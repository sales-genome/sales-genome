#!/usr/bin/env python3
"""Evaluate inductive checkpoints against exact-degree-matched negatives."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.graph_embeddings.inductive_baseline import (
    InductiveTwoTower,
    _encode_in_batches,
)


DEFAULT_CACHE = ROOT / "experiments" / "graph_embeddings" / "inductive_features.pt"


def build_degree_matched_comparisons(
    *,
    train_edges: list[tuple[int, int]],
    test_edges: list[tuple[int, int]],
    all_pairs: set[tuple[int, int]],
    n_customers: int,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build positive/negative triples with equal customer train degree."""
    degree = np.bincount(
        np.asarray(train_edges, dtype=np.int64)[:, 1],
        minlength=n_customers,
    )
    customers_by_degree: dict[int, list[int]] = defaultdict(list)
    for customer, value in enumerate(degree):
        customers_by_degree[int(value)].append(customer)

    rng = np.random.RandomState(seed)
    comparisons: list[tuple[int, int, int]] = []
    covered_edges = 0
    for vendor, positive_customer in test_edges:
        candidates = [
            customer
            for customer in customers_by_degree[int(degree[positive_customer])]
            if customer != positive_customer and (vendor, customer) not in all_pairs
        ]
        if not candidates:
            continue
        covered_edges += 1
        count = min(repeats, len(candidates))
        negatives = rng.choice(candidates, size=count, replace=False)
        comparisons.extend(
            (vendor, positive_customer, int(negative))
            for negative in negatives
        )

    array = np.asarray(comparisons, dtype=np.int64)
    diagnostics = {
        "n_test_edges": len(test_edges),
        "n_covered_test_edges": covered_edges,
        "coverage": round(covered_edges / len(test_edges), 6) if test_edges else 0.0,
        "n_comparisons": len(comparisons),
        "repeats": repeats,
    }
    return array, diagnostics


def evaluate_checkpoint(
    *,
    result_path: Path,
    cache: dict,
    comparisons: np.ndarray,
    diagnostics: dict[str, float | int],
    device: torch.device,
    batch_size: int,
) -> dict:
    result = json.loads(result_path.read_text())
    configuration = result["configuration"]
    if configuration.get("evaluation_protocol") != "edge_transfer":
        raise ValueError(f"Not an edge-transfer result: {result_path}")
    if result["cache_fingerprint"] != cache["fingerprint"]:
        raise ValueError(f"Cache fingerprint mismatch: {result_path}")

    vendor_feature_name = configuration["vendor_feature"]
    customer_feature_name = configuration["customer_feature"]
    vendor_values = cache["vendor_features"][vendor_feature_name].to(
        device=device,
        dtype=torch.float32,
    )
    customer_values = cache["customer_features"][customer_feature_name].to(
        device=device,
        dtype=torch.float32,
    )
    model = InductiveTwoTower(
        vendor_dim=vendor_values.shape[1],
        customer_dim=customer_values.shape[1],
        hidden_dim=configuration["hidden_dim"],
        output_dim=configuration["output_dim"],
        dropout=configuration["dropout"],
    ).to(device)
    checkpoint_path = result_path.with_suffix(".pt")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    vendor_embeddings = _encode_in_batches(
        model.vendor,
        vendor_values,
        batch_size=batch_size,
    )
    customer_embeddings = _encode_in_batches(
        model.customer,
        customer_values,
        batch_size=batch_size,
    )

    vendor = comparisons[:, 0]
    positive = comparisons[:, 1]
    negative = comparisons[:, 2]
    positive_scores = np.sum(
        vendor_embeddings[vendor] * customer_embeddings[positive],
        axis=1,
    )
    negative_scores = np.sum(
        vendor_embeddings[vendor] * customer_embeddings[negative],
        axis=1,
    )
    margins = positive_scores - negative_scores
    outcomes = (margins > 0).astype(np.float32)
    outcomes[margins == 0] = 0.5

    vendor_outcomes: dict[int, list[float]] = defaultdict(list)
    for vendor_index, outcome in zip(vendor, outcomes):
        vendor_outcomes[int(vendor_index)].append(float(outcome))
    per_vendor = np.asarray(
        [np.mean(values) for values in vendor_outcomes.values()],
        dtype=np.float32,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result": str(result_path),
        "run_name": result["run_name"],
        "cache_fingerprint": result["cache_fingerprint"],
        "configuration": configuration,
        "matching": diagnostics,
        "pair_accuracy": round(float(outcomes.mean()), 6),
        "pair_accuracy_lift_over_chance": round(float(outcomes.mean() / 0.5), 6),
        "mean_margin": round(float(margins.mean()), 6),
        "median_margin": round(float(np.median(margins)), 6),
        "n_vendors": len(vendor_outcomes),
        "mean_vendor_pair_accuracy": round(float(per_vendor.mean()), 6),
        "popularity_baseline_pair_accuracy": 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="edge_transfer__*.json")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_edges = cache["edge_transfer"]["train_edges"]
    test_edges = cache["edge_transfer"]["test_edges"]
    all_pairs = set(map(tuple, cache["pairs"]))
    comparisons, diagnostics = build_degree_matched_comparisons(
        train_edges=train_edges,
        test_edges=test_edges,
        all_pairs=all_pairs,
        n_customers=cache["customer_features"]["audience"].shape[0],
        repeats=args.repeats,
        seed=args.seed,
    )
    if comparisons.size == 0:
        raise SystemExit("No exact-degree-matched comparisons are available")

    summaries = []
    for result_path in sorted(args.result_dir.glob(args.pattern)):
        if result_path.name in {"matrix_manifest.json", "degree_matched_summary.json"}:
            continue
        summary = evaluate_checkpoint(
            result_path=result_path,
            cache=cache,
            comparisons=comparisons,
            diagnostics=diagnostics,
            device=torch.device(args.device),
            batch_size=args.batch_size,
        )
        output_path = result_path.with_suffix(".degree_matched.json")
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(
            f"{summary['run_name']}: "
            f"pair_accuracy={summary['pair_accuracy']:.4f}"
        )

    aggregate_path = args.result_dir / "degree_matched_summary.json"
    aggregate_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_fingerprint": cache["fingerprint"],
        "matching": diagnostics,
        "results": summaries,
    }, indent=2) + "\n")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
