#!/usr/bin/env python3
"""Evaluate pseudo-unseen prospect projection against the frozen GNN.

Each sampled graph customer is removed from its own lookalike set. The script
reconstructs a customer embedding from leakage-cleaned pain-subtheme and
audience features, then measures embedding agreement and held-out vendor
retrieval relative to the customer's direct frozen embedding.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.graph_embeddings.lead_prior import (
    DEFAULT_VARIANT,
    GraphPriorUnavailable,
    LeadGraphPrior,
)


MODES = {
    "pain_only": (1.0, 0.0),
    "audience_only": (0.0, 1.0),
    "combined": (0.7, 0.3),
}
K_VALUES = (1, 3, 5, 10, 20, 50)


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "n": len(values),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "p10": round(float(np.percentile(array, 10)), 6),
    }


def _retrieval_metrics(ranks: list[int]) -> dict[str, Any]:
    if not ranks:
        return {
            "n_edges": 0,
            "mean_reciprocal_rank": None,
            "hit_at_k": {},
        }
    return {
        "n_edges": len(ranks),
        "mean_reciprocal_rank": round(
            float(np.mean([1.0 / rank for rank in ranks])), 6
        ),
        "hit_at_k": {
            str(k): round(float(np.mean([rank <= k for rank in ranks])), 6)
            for k in K_VALUES
        },
    }


def evaluate_projection(
    prior: LeadGraphPrior,
    *,
    max_customers: int = 500,
    top_k_lookalikes: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    prior.load()
    assert prior.feature_space is not None
    assert prior.customer_embeddings is not None
    assert prior.vendor_embeddings is not None

    test_by_customer: dict[int, list[int]] = defaultdict(list)
    train_by_customer: dict[int, set[int]] = defaultdict(set)
    for vendor_index, customer_index in prior.test_vc:
        test_by_customer[customer_index].append(vendor_index)
    for vendor_index, customer_index in prior.train_vc:
        train_by_customer[customer_index].add(vendor_index)

    customer_indices = sorted(test_by_customer)
    if max_customers > 0 and len(customer_indices) > max_customers:
        customer_indices = sorted(
            random.Random(seed).sample(customer_indices, max_customers)
        )

    results: dict[str, Any] = {}
    for mode, (pain_weight, audience_weight) in MODES.items():
        agreement: list[float] = []
        projected_ranks: list[int] = []
        direct_ranks: list[int] = []
        status_counts: dict[str, int] = defaultdict(int)

        for customer_index in customer_indices:
            projection = prior.feature_space.project_vectors(
                prior.feature_space.customer_pain_features[customer_index],
                prior.feature_space.customer_audience_features[customer_index],
                top_k=top_k_lookalikes,
                pain_weight=pain_weight,
                audience_weight=audience_weight,
                exclude_customer_index=customer_index,
            )
            status_counts[projection.status] += 1
            if projection.embedding is None:
                continue

            direct_embedding = prior.customer_embeddings[customer_index]
            agreement.append(float(direct_embedding @ projection.embedding))

            projected_scores = prior.vendor_embeddings @ projection.embedding
            direct_scores = prior.vendor_embeddings @ direct_embedding
            excluded_vendors = train_by_customer.get(customer_index, set())
            if excluded_vendors:
                excluded = list(excluded_vendors)
                projected_scores[excluded] = -np.inf
                direct_scores[excluded] = -np.inf

            projected_order = np.argsort(-projected_scores)
            direct_order = np.argsort(-direct_scores)
            projected_position = {
                int(vendor_index): rank
                for rank, vendor_index in enumerate(projected_order, 1)
            }
            direct_position = {
                int(vendor_index): rank
                for rank, vendor_index in enumerate(direct_order, 1)
            }
            for vendor_index in test_by_customer[customer_index]:
                projected_ranks.append(projected_position[vendor_index])
                direct_ranks.append(direct_position[vendor_index])

        results[mode] = {
            "n_customers_requested": len(customer_indices),
            "n_customers_projected": len(agreement),
            "coverage": (
                round(len(agreement) / len(customer_indices), 6)
                if customer_indices
                else 0.0
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "embedding_cosine_agreement": _summarize(agreement),
            "projected_vendor_retrieval": _retrieval_metrics(projected_ranks),
            "direct_vendor_retrieval": _retrieval_metrics(direct_ranks),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": prior.variant,
        "model_fingerprint": prior.fingerprint,
        "seed": seed,
        "top_k_lookalikes": top_k_lookalikes,
        "n_test_customers_available": len(test_by_customer),
        "modes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--max-customers", type=int, default=500)
    parser.add_argument("--top-k-lookalikes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "graph_embeddings"
        / "results"
        / "lead_projection_eval.json",
    )
    args = parser.parse_args()

    prior = LeadGraphPrior(variant=args.variant)
    try:
        result = evaluate_projection(
            prior,
            max_customers=args.max_customers,
            top_k_lookalikes=args.top_k_lookalikes,
            seed=args.seed,
        )
    except GraphPriorUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
