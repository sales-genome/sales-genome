"""Run CPU baselines and a directional heterogeneous GraphSAGE on a frozen cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch_geometric
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_submission import artifacts as cohort

KINDS = ("vendor", "customer", "product", "capability", "pain_point")
WIDTH = 128
VERSIONS = ("raw", "raw_matched", "canonical")
MESSAGE_RELATIONS = {
    "offers": ("product", "vendor"),
    "uses_product": ("product", "customer"),
    "provides": ("capability", "product"),
    "experiences": ("pain_point", "customer"),
    "addresses": ("pain_point", "capability"),
}


def verify_snapshot(snapshot):
    manifest = cohort.load(snapshot / "manifest.json")
    if "held_out_source_observations_excluded_from_all_training_edges" not in manifest:
        raise ValueError(
            "Training requires a compatible frozen cohort with source-disjoint train_edges/splits. "
            "The unsplit paper sample is not training input; use paper_submission.gnn_smoke."
        )
    for name, artifact in manifest["artifacts"].items():
        raw = (snapshot / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError("Frozen artifact checksum mismatch: " + name)
        if len(raw) != artifact["bytes"]:
            raise ValueError("Frozen artifact size mismatch: " + name)
    return manifest


def words(text):
    result = torch.zeros(WIDTH)
    for token in set(re.findall(r"\w{3,}", text.casefold())):
        value = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        result[value % WIDTH] += 1 if value & (1 << 63) else -1
    return F.normalize(result, dim=0)


def graph_features(nodes, edges, required):
    triples = sorted({(edge["src"], edge["type"], edge["dst"]) for edge in edges})
    active = {endpoint for src, _, dst in triples for endpoint in [src, dst]} | required
    if not active <= nodes.keys():
        raise ValueError("Training relationships contain unknown endpoints.")
    selected = {key: nodes[key] for key in sorted(active)}
    text = {
        key: words(node["text"]) if node["type"] in {"capability", "pain_point"} else torch.zeros(WIDTH)
        for key, node in selected.items()
    }
    outgoing = defaultdict(set)
    incoming_count = Counter()
    outgoing_count = Counter()
    for src, relation, dst in triples:
        outgoing[(src, relation)].add(dst)
        incoming_count[dst] += 1
        outgoing_count[src] += 1
    for kind, relations in [
        ("capability", ["addresses"]), ("product", ["provides"]),
        ("vendor", ["offers"]), ("customer", ["experiences", "uses_product"]),
    ]:
        for key, node in selected.items():
            if node["type"] != kind:
                continue
            neighbours = sorted(set().union(*(outgoing[(key, relation)] for relation in relations)))
            if neighbours:
                text[key] = F.normalize(
                    text[key] + torch.stack([text[neighbour] for neighbour in neighbours]).mean(0), dim=0,
                )
    features = {}
    for key, node in selected.items():
        if node["type"] not in KINDS:
            raise ValueError("Unsupported node type: " + node["type"])
        kind_feature = torch.zeros(len(KINDS))
        kind_feature[KINDS.index(node["type"])] = 1
        features[key] = torch.cat([
            text[key], kind_feature,
            torch.tensor([math.log1p(incoming_count[key]), math.log1p(outgoing_count[key])]),
        ])
    return selected, features, triples


def load_graph(snapshot, version):
    manifest = verify_snapshot(snapshot)
    labels = list(cohort.lines(snapshot / "labels.jsonl"))
    if len({(row["src"], row["dst"]) for row in labels}) != len(labels):
        raise ValueError("Duplicate observed vendor/customer labels.")
    nodes = {row["id"]: row for row in cohort.lines(snapshot / version / "nodes.jsonl")}
    edges = list(cohort.lines(snapshot / version / "train_edges.jsonl"))
    held_sources = {
        observation for row in labels if row["split"] != "train" for observation in row["observation_ids"]
    }
    if held_sources != set(manifest["held_out_source_observations_excluded_from_all_training_edges"]):
        raise ValueError("Held-out source manifest differs from labels.")
    if any(held_sources & set(edge["observation_ids"]) for edge in edges):
        raise ValueError("Held-out case-study source leaked into training edges.")
    warm_vendors = {row["src"] for row in labels if row["split"] == "train"}
    warm_customers = {row["dst"] for row in labels if row["split"] == "train"}
    if any(row["src"] not in warm_vendors or row["dst"] not in warm_customers for row in labels):
        raise ValueError("This cohort protocol requires observed training evidence for every endpoint.")
    required = {endpoint for row in labels for endpoint in [row["src"], row["dst"]]}
    nodes, features, triples = graph_features(nodes, edges, required)
    maps = {
        kind: {key: index for index, key in enumerate(
            key for key, node in sorted(nodes.items()) if node["type"] == kind
        )} for kind in KINDS
    }
    data = HeteroData()
    for kind, mapping in maps.items():
        if mapping:
            data[kind].x = torch.stack([features[key] for key in mapping])
    relation_edges = defaultdict(list)
    for src, relation, dst in triples:
        if relation not in MESSAGE_RELATIONS:
            raise ValueError("Not an admissible reviewed business relation: " + relation)
        source_kind, target_kind = MESSAGE_RELATIONS[relation]
        if nodes[dst]["type"] != source_kind or nodes[src]["type"] != target_kind:
            raise ValueError("Wrong relationship endpoint types: " + relation)
        relation_edges[(source_kind, "reverse_" + relation, target_kind)].append(
            [maps[source_kind][dst], maps[target_kind][src]],
        )
    for edge_type, pairs in sorted(relation_edges.items()):
        data[edge_type].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    data.validate(raise_on_error=True)
    return data, maps, labels


class TwoTower(nn.Module):
    def __init__(self, width, hidden, nonlinear):
        super().__init__()
        def tower():
            if nonlinear:
                return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            return nn.Linear(width, hidden)
        self.vendor = tower()
        self.customer = tower()

    def forward(self, data):
        return self.vendor(data["vendor"].x), self.customer(data["customer"].x)


class DirectionalSAGE(nn.Module):
    def __init__(self, data, hidden):
        super().__init__()
        self.kinds = tuple(data.node_types)
        self.relations = tuple(data.edge_types)
        self.input = nn.ModuleDict({
            kind: nn.Linear(data[kind].num_features, hidden) for kind in self.kinds
        })
        self.layers = nn.ModuleList([
            nn.ModuleList([SAGEConv((hidden, hidden), hidden, root_weight=False)
                           for _ in self.relations]) for _ in range(3)
        ])

    def forward(self, data):
        values = {kind: F.relu(self.input[kind](data[kind].x)) for kind in self.kinds}
        for index, layer in enumerate(self.layers):
            incoming = defaultdict(list)
            for convolution, relation in zip(layer, self.relations):
                src, _, dst = relation
                incoming[dst].append(convolution(
                    (values[src], values[dst]), data[relation].edge_index,
                ))
            # Organizations cannot rebroadcast their case-study features through global concept hubs.
            values = {
                kind: value + torch.stack(incoming[kind]).mean(0)
                if kind in incoming else value for kind, value in values.items()
            }
            if index + 1 < len(self.layers):
                values = {kind: F.relu(value) for kind, value in values.items()}
        return values["vendor"], values["customer"]


def normalized_scores(vendor, customer):
    return 10 * F.normalize(vendor, dim=1) @ F.normalize(customer, dim=1).t()


def ranking(scores, labels, maps, split, excluded_pairs=()):
    positive = defaultdict(set)
    excluded = defaultdict(set)
    for vendor, customer in excluded_pairs:
        excluded[vendor].add(customer)
    for row in labels:
        if row["split"] == split:
            positive[row["src"]].add(row["dst"])
        else:
            excluded[row["src"]].add(row["dst"])
    queries = []
    for vendor, relevant in sorted(positive.items()):
        candidates = [
            customer for customer in maps["customer"] if customer not in excluded[vendor]
        ]
        ordered = sorted(candidates, key=lambda customer: (
            -float(scores[maps["vendor"][vendor], maps["customer"][customer]]), customer,
        ))
        hits, precision = 0, 0.0
        first_rank = None
        for index, customer in enumerate(ordered, 1):
            if customer in relevant:
                hits += 1
                precision += hits / index
                if first_rank is None:
                    first_rank = index
        if hits != len(relevant):
            raise ValueError("A positive label is absent from the ranking universe.")
        queries.append({
            "vendor": vendor, "positives": len(relevant), "candidates": len(candidates),
            "average_precision": precision / len(relevant),
            "reciprocal_rank": 1 / first_rank,
            "hits_at_3": sum(customer in relevant for customer in ordered[:3]) / len(relevant),
        })
    if not queries:
        raise ValueError("No queries for requested split: " + split)
    return {
        "queries": len(queries),
        **{key: sum(row[key] for row in queries) / len(queries)
           for key in ["average_precision", "reciprocal_rank", "hits_at_3"]},
        "per_query": queries,
    }


def training_pairs(labels, maps, seed, excluded_pairs=()):
    known = {(row["src"], row["dst"]) for row in labels} | set(excluded_pairs)
    positive, negative = [], []
    for row in labels:
        if row["split"] != "train":
            continue
        vendor = row["src"]
        positive.append([maps["vendor"][vendor], maps["customer"][row["dst"]]])
        unknown = sorted(
            [customer for customer in maps["customer"] if (vendor, customer) not in known],
            key=lambda customer: hashlib.sha256(
                f"{seed}:{vendor}:{row['dst']}:{customer}".encode(),
            ).hexdigest(),
        )[:4]
        if not unknown:
            raise ValueError("No unknown comparison pairs for training vendor: " + vendor)
        negative.extend([maps["vendor"][vendor], maps["customer"][customer]] for customer in unknown)
    pairs = torch.tensor(positive + negative, dtype=torch.long).t()
    targets = torch.tensor([1.] * len(positive) + [0.] * len(negative))
    return pairs, targets


def run(snapshot, destination, seeds, epochs):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = verify_snapshot(snapshot)
    excluded_pairs = {
        (row["src"], row["dst"]) for row in cohort.lines(snapshot / "excluded_observed_pairs.jsonl")
    }
    report = {
        "snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
        "snapshot_version": manifest["version"], "seeds": seeds, "epochs": epochs,
        "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_canonicalization_comparison": manifest["primary_canonicalization_comparison"],
        "coverage_control_graph": manifest["coverage_control"], "limitations": manifest["limitations"],
        "device": "cpu", "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "feature_width": WIDTH + len(KINDS) + 2,
        "hyperparameters": {
            "hidden": 32, "gnn_layers": 3, "learning_rate": 0.005, "weight_decay": 0.0001,
            "maximum_unknown_comparisons_per_positive": 4, "score_scale": 10,
        },
        "feature_encoder": "Fixed signed SHA256 word hashing; no learned external embeddings.",
        "message_direction": "Pain -> capability -> product -> vendor; pain/product -> customer. No organization -> concept messages.",
        "evaluation": "Filtered observed-label recovery, not qualification or verified-negative prospect ranking.",
        "additional_observed_pairs_excluded_from_unknown_comparisons": len(excluded_pairs),
        "hyperparameters_selected_on_test": False, "results": [],
    }
    for version in VERSIONS:
        data, maps, labels = load_graph(snapshot, version)
        cohort.save(destination / (version + "-node-maps.json"), maps)
        for seed in seeds:
            torch.manual_seed(seed)
            popularity = torch.zeros((len(maps["vendor"]), len(maps["customer"])))
            degree = Counter(row["dst"] for row in labels if row["split"] == "train")
            for customer, index in maps["customer"].items():
                popularity[:, index] = degree[customer]
            random = torch.rand(popularity.shape, generator=torch.Generator().manual_seed(seed))
            cosine = normalized_scores(data["vendor"].x, data["customer"].x)
            for family, scores in [("random", random), ("customer_popularity", popularity),
                                   ("feature_cosine", cosine)]:
                report["results"].append({
                    "graph": version, "seed": seed, "family": family,
                    "validation": ranking(scores, labels, maps, "validation", excluded_pairs),
                    "test": ranking(scores, labels, maps, "test", excluded_pairs),
                })
            pairs, targets = training_pairs(labels, maps, seed, excluded_pairs)
            for family in ["linear_two_tower", "mlp_two_tower", "heterogeneous_sage"]:
                torch.manual_seed(seed)
                if family == "heterogeneous_sage":
                    model = DirectionalSAGE(data, 32)
                else:
                    model = TwoTower(data["vendor"].num_features, 32, family == "mlp_two_tower")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
                losses = []
                model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    scores = normalized_scores(*model(data))
                    loss = F.binary_cross_entropy_with_logits(scores[pairs[0], pairs[1]], targets)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss.")
                    loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters()):
                        raise ValueError("Nonfinite model gradient.")
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    scores = normalized_scores(*model(data))
                checkpoint = destination / f"{version}-{family}-{seed}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(), "family": family, "hidden": 32,
                    "snapshot_manifest_sha256": report["snapshot_manifest_sha256"], "seed": seed,
                    "epochs": epochs, "node_map_file": version + "-node-maps.json",
                    "graph": version,
                }, checkpoint)
                report["results"].append({
                    "graph": version, "seed": seed, "family": family, "losses": losses,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "validation": ranking(scores, labels, maps, "validation", excluded_pairs),
                    "test": ranking(scores, labels, maps, "test", excluded_pairs),
                })
                print("BASELINE_TRAINED", version, family, seed, losses[0], losses[-1], flush=True)
    cohort.save(destination / "results.json", report)
    return report


def verify_run(snapshot, destination):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = cohort.load(destination / "results.json")
    if report["training_code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Training code changed after the model run.")
    manifest_checksum = hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    if report["snapshot_manifest_sha256"] != manifest_checksum:
        raise ValueError("Model results refer to a different graph snapshot.")
    loaded = {version: load_graph(snapshot, version) for version in VERSIONS}
    excluded_pairs = {
        (row["src"], row["dst"]) for row in cohort.lines(snapshot / "excluded_observed_pairs.jsonl")
    }
    for kind in ["vendor", "customer"]:
        if any(loaded["raw"][1][kind] != loaded[version][1][kind] for version in VERSIONS):
            raise ValueError("Raw/canonical target entity IDs differ.")
    reloaded = 0
    for result in report["results"]:
        if "checkpoint" not in result:
            continue
        path = destination / result["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
            raise ValueError("Model checkpoint changed: " + path.name)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["snapshot_manifest_sha256"] != manifest_checksum or state["graph"] != result["graph"]:
            raise ValueError("Model checkpoint lineage differs from results.")
        data, maps, labels = loaded[result["graph"]]
        if cohort.load(destination / state["node_map_file"]) != maps:
            raise ValueError("Model node IDs differ from the frozen node map.")
        if result["family"] == "heterogeneous_sage":
            model = DirectionalSAGE(data, state["hidden"])
        else:
            model = TwoTower(data["vendor"].num_features, state["hidden"],
                             result["family"] == "mlp_two_tower")
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            scores = normalized_scores(*model(data))
        for split in ["validation", "test"]:
            if ranking(scores, labels, maps, split, excluded_pairs) != result[split]:
                raise ValueError("Reloaded model does not reproduce ranking metrics: " + path.name)
        reloaded += 1
    if reloaded != 3 * len(VERSIONS) * len(report["seeds"]):
        raise ValueError("Missing a required raw/canonical model family or seed.")
    verification = {
        "verified_at": cohort.repair.source.now(), "model_checkpoints_reloaded": reloaded,
        "validation_and_test_rankings_reproduced": True,
        "raw_canonical_target_ids_identical": True, "held_source_exclusion_verified": True,
        "snapshot_manifest_sha256": manifest_checksum,
    }
    cohort.save(destination / "verification.json", verification)
    print("MODEL_RUN_VERIFIED", json.dumps(verification), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Use positive epochs and distinct seeds.")
    if args.verify:
        verify_run(args.snapshot, args.out)
    else:
        run(args.snapshot, args.out, args.seeds, args.epochs)


if __name__ == "__main__":
    main()
