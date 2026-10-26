"""CPU graph/forward-pass check; no optimizer, training, checkpoint, or metrics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from paper_submission.artifacts import lines, save


def inspect(graph):
    import torch
    from torch_geometric.data import HeteroData
    from scripts import gnn_ready_baseline as baseline

    torch.set_num_threads(2)
    torch.manual_seed(17)
    nodes = {row["id"]: row for row in lines(Path(graph) / "nodes.jsonl") if row["type"] in baseline.KINDS}
    edges = [row for row in lines(Path(graph) / "edges.jsonl")
             if row["type"] in baseline.MESSAGE_RELATIONS
             and row["src"] in nodes and row["dst"] in nodes]
    required = {node_id for node_id, node in nodes.items() if node["type"] in {"vendor", "customer"}}
    nodes, features, triples = baseline.graph_features(nodes, edges, required)
    maps = {kind: {node_id: i for i, node_id in enumerate(
        item for item in sorted(nodes) if nodes[item]["type"] == kind
    )} for kind in baseline.KINDS}
    data = HeteroData()
    for kind, mapping in maps.items():
        if mapping:
            data[kind].x = torch.stack([features[node_id] for node_id in mapping])
    relation_edges = defaultdict(list)
    for src, relation, dst in triples:
        source_kind, target_kind = baseline.MESSAGE_RELATIONS[relation]
        relation_edges[(source_kind, "reverse_" + relation, target_kind)].append(
            [maps[source_kind][dst], maps[target_kind][src]],
        )
    for relation, pairs in relation_edges.items():
        data[relation].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    data.validate(raise_on_error=True)
    with torch.no_grad():
        model = baseline.DirectionalSAGE(data, hidden=32).eval()
        vendor, customer = model(data)
    if not torch.isfinite(vendor).all() or not torch.isfinite(customer).all():
        raise ValueError("Nonfinite forward-pass output.")
    return {
        "task": "untrained_CPU_shape_and_endpoint_check",
        "training_run": False, "experimental_metrics_computed": False,
        "feature_source": "selected graph phrase hashing; NOT independent cold-start attributes",
        "vendor_shape": list(vendor.shape), "customer_shape": list(customer.shape),
        "message_edges": len(triples), "direct_has_case_study_label_edges_used": False,
        "torch": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.graph)
    save(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
