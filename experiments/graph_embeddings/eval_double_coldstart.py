#!/usr/bin/env python3
"""Evaluate unseen product-description vendors against unseen researched customers."""

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

from experiments.graph_embeddings.inductive_baseline import (  # noqa: E402
    InductiveTwoTower,
    SliceDefinition,
    _encode_in_batches,
    evaluate_slice,
)


def build_double_definition(
    *,
    vendor_split: dict,
    profiled_customers: set[int],
) -> SliceDefinition:
    edges = [
        tuple(map(int, edge))
        for edge in vendor_split["test_slices"]["double_holdout"]
        if int(edge[1]) in profiled_customers
    ]
    return SliceDefinition(
        edges=edges,
        candidate_customers=sorted(profiled_customers),
        excluded_pairs=set(),
    )


def aggregate_runs(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["configuration"]].append(run)
    output = {}
    for configuration, values in sorted(grouped.items()):
        metrics = {}
        for name in (
            "mean_average_precision",
            "mean_ndcg_at_50",
            "mean_precision_at_10",
            "precision_at_10_lift",
            "mean_recall_at_50",
        ):
            array = np.asarray(
                [value["metrics"][name] for value in values],
                dtype=np.float64,
            )
            metrics[name] = {
                "mean": round(float(array.mean()), 6),
                "std": round(float(array.std()), 6),
            }
        output[configuration] = {
            "n_runs": len(values),
            "seeds": sorted(value["seed"] for value in values),
            "metrics": metrics,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-cache", type=Path, required=True)
    parser.add_argument("--customer-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--vendor-features",
        nargs="+",
        default=["product_only"],
    )
    args = parser.parse_args()

    vendor_cache = torch.load(
        args.vendor_cache,
        map_location="cpu",
        weights_only=False,
    )
    customer_cache = torch.load(
        args.customer_cache,
        map_location="cpu",
        weights_only=False,
    )
    if vendor_cache["split"]["train_edges"] != customer_cache["split"]["train_edges"]:
        raise ValueError("vendor and customer caches use different training edges")
    profiled_customers = set(
        map(int, customer_cache["split"]["holdout_customers"])
    )
    definition = build_double_definition(
        vendor_split=vendor_cache["split"],
        profiled_customers=profiled_customers,
    )
    if not definition.edges:
        raise ValueError("no independently profiled double-holdout edges")

    device = torch.device(args.device)
    known_pairs = set(map(tuple, vendor_cache["pairs"]))
    runs = []
    result_paths = sorted(
        args.checkpoint_dir.glob("oracle_node_holdout__*__*__seed*.json")
    )
    for result_path in result_paths:
        training_result = json.loads(result_path.read_text())
        if training_result["cache_fingerprint"] != vendor_cache["fingerprint"]:
            raise ValueError(f"checkpoint cache mismatch: {result_path}")
        configuration = training_result["configuration"]
        vendor_feature = configuration["vendor_feature"]
        if vendor_feature not in args.vendor_features:
            continue
        customer_feature = configuration["customer_feature"]
        vendor_values = vendor_cache["vendor_features"][vendor_feature].to(
            device=device,
            dtype=torch.float32,
        )
        customer_values = customer_cache["customer_features"][
            customer_feature
        ].to(device=device, dtype=torch.float32)
        model = InductiveTwoTower(
            vendor_dim=vendor_values.shape[1],
            customer_dim=customer_values.shape[1],
            hidden_dim=int(configuration["hidden_dim"]),
            output_dim=int(configuration["output_dim"]),
            dropout=float(configuration["dropout"]),
        ).to(device)
        model.load_state_dict(
            torch.load(
                result_path.with_suffix(".pt"),
                map_location=device,
                weights_only=True,
            )
        )
        vendor_embeddings = _encode_in_batches(
            model.vendor,
            vendor_values,
            batch_size=args.batch_size,
        )
        customer_embeddings = _encode_in_batches(
            model.customer,
            customer_values,
            batch_size=args.batch_size,
        )
        metrics = evaluate_slice(
            vendor_embeddings=vendor_embeddings,
            customer_embeddings=customer_embeddings,
            definition=definition,
            known_pairs=known_pairs,
            seed=int(configuration["seed"]),
        )
        runs.append(
            {
                "run_name": training_result["run_name"],
                "configuration": f"{vendor_feature}__{customer_feature}",
                "seed": int(configuration["seed"]),
                "metrics": metrics,
            }
        )

    semantic_vectors = customer_cache["customer_features"][
        "semantic_only"
    ].cpu().numpy()
    semantic_vectors /= np.maximum(
        np.linalg.norm(semantic_vectors, axis=1, keepdims=True),
        1e-8,
    )
    direct_semantic = {}
    for vendor_feature in args.vendor_features:
        vendor_vectors = vendor_cache["vendor_features"][vendor_feature].cpu().numpy()
        if vendor_vectors.shape[1] != semantic_vectors.shape[1]:
            continue
        vendor_vectors /= np.maximum(
            np.linalg.norm(vendor_vectors, axis=1, keepdims=True),
            1e-8,
        )
        direct_semantic[vendor_feature] = evaluate_slice(
            vendor_embeddings=vendor_vectors,
            customer_embeddings=semantic_vectors,
            definition=definition,
            known_pairs=known_pairs,
            seed=42,
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "independent_double_coldstart",
        "vendor_cache": {
            "path": str(args.vendor_cache),
            "fingerprint": vendor_cache["fingerprint"],
            "features": args.vendor_features,
            "provenance": vendor_cache.get("independent_vendor_profile", {}),
        },
        "customer_cache": {
            "path": str(args.customer_cache),
            "fingerprint": customer_cache["fingerprint"],
            "profile_source": customer_cache["split"].get(
                "profile_source",
                "unknown",
            ),
            "provenance": customer_cache.get("independent_profile", {}).get(
                "provenance",
                {},
            ),
        },
        "population": {
            "profiled_candidates": len(profiled_customers),
            "positive_edges": len(definition.edges),
            "vendor_queries": len({vendor for vendor, _ in definition.edges}),
            "positive_customers": len(
                {customer for _, customer in definition.edges}
            ),
        },
        "runs": runs,
        "aggregate": aggregate_runs(runs),
        "baselines": {
            "direct_vendor_feature_to_semantic_pain_cosine": direct_semantic,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
