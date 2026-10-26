#!/usr/bin/env python3
"""Evaluate training-customer centroid baselines on independent profiles."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.graph_embeddings.inductive_baseline import (  # noqa: E402
    SliceDefinition,
    evaluate_slice,
)


FEATURES = (
    "audience",
    "semantic_only",
    "tfidf_only",
    "semantic",
    "tfidf",
    "fusion",
)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def vendor_customer_centroids(
    *,
    customer_features: np.ndarray,
    train_edges: list[tuple[int, int]],
    n_vendors: int,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.zeros(
        (n_vendors, customer_features.shape[1]),
        dtype=np.float32,
    )
    counts = np.zeros(n_vendors, dtype=np.int64)
    for vendor, customer in train_edges:
        centroids[vendor] += customer_features[customer]
        counts[vendor] += 1
    centroids /= np.maximum(counts[:, None], 1)
    return normalize_rows(centroids), counts


def evaluate_centroids(cache: dict, *, seed: int) -> dict:
    split = cache["split"]
    candidates = split.get(
        "customer_holdout_candidates",
        split["holdout_customers"],
    )
    definition = SliceDefinition(
        edges=split["test_slices"]["customer_holdout"],
        candidate_customers=list(map(int, candidates)),
        excluded_pairs=set(),
    )
    train_edges = [tuple(map(int, edge)) for edge in split["train_edges"]]
    known_pairs = set(map(tuple, cache["pairs"]))
    n_vendors = next(iter(cache["vendor_features"].values())).shape[0]

    results = {}
    for feature_name in FEATURES:
        if feature_name not in cache["customer_features"]:
            continue
        customer_features = normalize_rows(
            cache["customer_features"][feature_name].cpu().numpy().astype(
                np.float32,
                copy=False,
            )
        )
        vendor_features, counts = vendor_customer_centroids(
            customer_features=customer_features,
            train_edges=train_edges,
            n_vendors=n_vendors,
        )
        metrics = evaluate_slice(
            vendor_embeddings=vendor_features,
            customer_embeddings=customer_features,
            definition=definition,
            known_pairs=known_pairs,
            seed=seed,
        )
        results[feature_name] = {
            "metrics": metrics,
            "vendors_with_training_customers": int((counts > 0).sum()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    results = evaluate_centroids(cache, seed=args.seed)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "independent_customer_holdout_centroids",
        "cache": str(args.cache),
        "cache_fingerprint": cache["fingerprint"],
        "source_fingerprint": cache["source_fingerprint"],
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
