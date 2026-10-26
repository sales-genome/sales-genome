"""Notebook/memo R-GCN, degree controls, training and fixed-seller resampling."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

from paper_submission.artifacts import save, sha256


@dataclass(frozen=True)
class Configuration:
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    temperature: float = 0.1
    negatives: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 30
    max_cap_product_degree: int = 100


def collapsed_graph(base, train_pairs, *, degree_limit=100, residual=True):
    import torch
    from experiments.graph_embeddings.build_graph_variants import build_collapsed
    if degree_limit <= 0:
        raise ValueError("Capability degree limit must be positive.")
    graph, vendor_products = build_collapsed(base, train_pairs, add_residual=residual)
    provides = graph["vendor_product", "provides", "cap_canonical"].edge_index
    unique = torch.unique(provides, dim=1)
    degree = torch.bincount(unique[1], minlength=graph["cap_canonical"].x.shape[0])
    keep = degree <= degree_limit
    remap = torch.full((keep.numel(),), -1, dtype=torch.long)
    remap[keep] = torch.arange(int(keep.sum()))
    graph["cap_canonical"].x = graph["cap_canonical"].x[keep]
    for relation in list(graph.edge_types):
        source, _, target = relation
        edge_index = graph[relation].edge_index
        valid = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if source == "cap_canonical":
            valid &= keep[edge_index[0]]
        if target == "cap_canonical":
            valid &= keep[edge_index[1]]
        edge_index = edge_index[:, valid].clone()
        if source == "cap_canonical":
            edge_index[0] = remap[edge_index[0]]
        if target == "cap_canonical":
            edge_index[1] = remap[edge_index[1]]
        graph[relation].edge_index = edge_index
    graph.validate(raise_on_error=True)
    n_vendors = int(base["vendor"].num_nodes)
    products = [vendor_products.get(i, []) for i in range(n_vendors)]
    return graph, products, {
        "capabilities_before": keep.numel(), "capabilities_after": int(keep.sum()),
        "degree_definition": "unique_product_neighbours_not_top_n_per_vendor",
        "degree_limit": degree_limit, "residual_training_pairs": len(train_pairs) if residual else 0,
    }


def negative_candidates(pairs, n_customers):
    known = defaultdict(set)
    for vendor, customer in pairs:
        known[int(vendor)].add(int(customer))
    return {vendor: sorted(set(range(n_customers)) - customers) for vendor, customers in known.items()}


def info_nce(vendors, customers, pairs, candidates, *, temperature, negatives, rng):
    import torch
    from torch.nn import functional as F
    if temperature <= 0 or negatives <= 0 or not pairs:
        raise ValueError("InfoNCE needs pairs, positive temperature and negative count.")
    if any(not candidates.get(int(vendor)) for vendor, _ in pairs):
        raise ValueError("A training vendor has no unlabeled customer candidates.")
    device = vendors.device
    indexes = torch.tensor(pairs, dtype=torch.long, device=device)
    negative = torch.tensor([
        rng.choices(candidates[int(vendor)], k=negatives) for vendor, _ in pairs
    ], dtype=torch.long, device=device)
    vendors = F.normalize(vendors, dim=-1)
    customers = F.normalize(customers, dim=-1)
    selected = vendors[indexes[:, 0]]
    positive = (selected * customers[indexes[:, 1]]).sum(-1, keepdim=True)
    noise = (selected[:, None, :] * customers[negative]).sum(-1)
    logits = torch.cat([positive, noise], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(pairs), dtype=torch.long, device=device))


def sampled_recovery(vendors, customers, train_pairs, test_pairs, *, seed,
                     resamples=500, sellers_per_sample=50, positives=10, negatives=90,
                     ks=(10, 20, 50)):
    import torch
    from torch.nn import functional as F
    if any(value <= 0 for value in (resamples, sellers_per_sample, positives, negatives)):
        raise ValueError("Resampling counts must be positive.")
    if any(k <= 0 or k > positives + negatives for k in ks):
        raise ValueError("Precision cutoffs must fit the sampled candidate pool.")
    vendors = F.normalize(vendors.detach().cpu(), dim=-1)
    customers = F.normalize(customers.detach().cpu(), dim=-1)
    known, heldout = defaultdict(set), defaultdict(set)
    for vendor, customer in train_pairs:
        known[int(vendor)].add(int(customer))
    for vendor, customer in test_pairs:
        if int(customer) in known[int(vendor)]:
            raise ValueError("Training and held-out pairs overlap.")
        heldout[int(vendor)].add(int(customer))
    population = set(range(customers.shape[0]))
    eligible = [
        vendor for vendor, targets in sorted(heldout.items())
        if len(targets) >= positives and len(population - targets - known[vendor]) >= negatives
    ]
    if len(eligible) < sellers_per_sample:
        raise ValueError(
            f"Need {sellers_per_sample} eligible sellers, found {len(eligible)}; "
            f"each needs {positives} held-out positives and {negatives} unlabeled controls."
        )
    rng, sums = random.Random(seed), defaultdict(float)
    for _ in range(resamples):
        for vendor in rng.sample(eligible, sellers_per_sample):
            target = set(rng.sample(sorted(heldout[vendor]), positives))
            pool = list(target) + rng.sample(sorted(population - heldout[vendor] - known[vendor]), negatives)
            rng.shuffle(pool)
            scores = (customers[pool] @ vendors[vendor]).tolist()
            ranked = sorted(zip(pool, scores, strict=True), key=lambda item: -item[1])
            for k in ks:
                sums[k] += sum(customer in target for customer, _ in ranked[:k]) / k
    denominator = resamples * sellers_per_sample
    return {
        "resamples": resamples, "seed": seed, "eligible_sellers": len(eligible),
        "sellers_per_sample": sellers_per_sample, "positive_candidates": positives,
        "unlabeled_candidates": negatives,
        "random_expected_precision": positives / (positives + negatives),
        "precision": {str(k): sums[k] / denominator for k in ks},
        "negative_interpretation": "unobserved_relationship_not_verified_noncustomer",
        "historical_reported_metrics_reproduced": False,
    }


def train(base, train_pairs, validation_pairs, *, config, seed, device="cpu"):
    import torch
    from experiments.graph_embeddings.train_v2 import LinkPredModel
    if not train_pairs or not validation_pairs:
        raise ValueError("Separate optimization and validation pairs are required.")
    if set(map(tuple, train_pairs)) & set(map(tuple, validation_pairs)):
        raise ValueError("Optimization and validation pairs overlap.")
    if config.epochs <= 0 or config.patience <= 0:
        raise ValueError("Epochs and patience must be positive.")
    torch.manual_seed(seed)
    graph, products, diagnostics = collapsed_graph(
        base, train_pairs, degree_limit=config.max_cap_product_degree,
    )
    graph = graph.to(device)
    model = LinkPredModel("collapsed_res_d100", graph, config.hidden_dim,
                          config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    training_candidates = negative_candidates(train_pairs, base["customer"].num_nodes)
    validation_candidates = negative_candidates(
        train_pairs + validation_pairs, base["customer"].num_nodes,
    )
    rng = random.Random(seed)
    best, best_epoch, state, history = float("inf"), 0, None, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        embedding = model.encode(graph)
        vendor, customer = model.get_vendor_customer(embedding, products, base["vendor"].num_nodes)
        loss = info_nce(vendor, customer, train_pairs, training_candidates,
                        temperature=config.temperature, negatives=config.negatives, rng=rng)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            embedding = model.encode(graph)
            vendor, customer = model.get_vendor_customer(embedding, products, base["vendor"].num_nodes)
            validation = info_nce(
                vendor, customer, validation_pairs, validation_candidates, temperature=config.temperature,
                negatives=config.negatives, rng=random.Random(seed + 1000003),
            ).item()
        history.append({"epoch": epoch, "loss": loss.item(), "validation_loss": validation})
        if validation < best:
            best, best_epoch = validation, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break
    if state is None:
        raise ValueError("No finite validation checkpoint was selected.")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        embedding = model.encode(graph)
        vendor, customer = model.get_vendor_customer(embedding, products, base["vendor"].num_nodes)
    return state, vendor, customer, {
        "configuration": asdict(config), "seed": seed, "device": device,
        "graph": diagnostics, "best_epoch": best_epoch, "history": history,
        "checkpoint_selection": "validation_only_not_final_test",
        "implementation": "captured_encoder_and_notebook_memo_training_reconstruction",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--trusted-local-input", action="store_true", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Training output must be new.")
    protocol = json.loads(args.protocol.read_text())
    if protocol["data_sha256"] != sha256(args.data):
        raise ValueError("Input data does not match the supplied frozen protocol.")
    if protocol.get("heldout_source_evidence_removed") is not True:
        raise ValueError("The protocol must identify a source-filtered message graph.")
    import torch
    checkpoint = torch.load(args.data, map_location="cpu", weights_only=False)
    config = Configuration(**protocol["configuration"])
    state, vendor, customer, report = train(
        checkpoint["data"], [tuple(pair) for pair in protocol["optimization_pairs"]],
        [tuple(pair) for pair in protocol["validation_pairs"]],
        config=config, seed=args.seed, device=args.device,
    )
    report["input_data_sha256"] = sha256(args.data)
    report["protocol_sha256"] = sha256(args.protocol)
    args.out.mkdir(parents=True)
    torch.save(state, args.out / "model.pt")
    save(args.out / "run.json", report)


if __name__ == "__main__":
    main()
