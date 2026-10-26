from __future__ import annotations

import numpy as np
import torch

from experiments.graph_embeddings.inductive_baseline import (
    SliceDefinition,
    aggregate_neighbor_features,
    average_precision,
    evaluate_slice,
    make_node_split,
    ndcg_at_k,
)
from experiments.graph_embeddings.eval_inductive_degree_matched import (
    build_degree_matched_comparisons,
)


def test_aggregate_neighbor_features_uses_weighted_means() -> None:
    pooled, totals = aggregate_neighbor_features(
        source_index=torch.tensor([0, 0, 1]),
        target_index=torch.tensor([0, 1, 1]),
        target_features=torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        n_sources=3,
        edge_weights=torch.tensor([1.0, 3.0, 2.0]),
    )

    assert torch.allclose(totals, torch.tensor([4.0, 2.0, 0.0]))
    assert torch.allclose(pooled[0], torch.tensor([0.25, 1.5]))
    assert torch.allclose(pooled[1], torch.tensor([0.0, 2.0]))
    assert torch.allclose(pooled[2], torch.zeros(2))


def test_make_node_split_is_disjoint_and_complete() -> None:
    pairs = [
        (vendor, customer)
        for vendor in range(5)
        for customer in range(8)
        if (vendor + customer) % 3 == 0
    ]
    split = make_node_split(
        pairs,
        n_vendors=5,
        n_customers=8,
        holdout_fraction=0.25,
        seed=7,
    )

    reconstructed = set(split["train_edges"])
    for edges in split["test_slices"].values():
        reconstructed.update(edges)
    assert reconstructed == set(pairs)

    train_vendors = set(split["train_vendors"])
    holdout_vendors = set(split["holdout_vendors"])
    train_customers = set(split["train_customers"])
    holdout_customers = set(split["holdout_customers"])
    assert train_vendors.isdisjoint(holdout_vendors)
    assert train_customers.isdisjoint(holdout_customers)
    for vendor, customer in split["train_edges"]:
        assert vendor in train_vendors
        assert customer in train_customers
    for vendor, customer in split["test_slices"]["customer_holdout"]:
        assert vendor in train_vendors
        assert customer in holdout_customers
    for vendor, customer in split["test_slices"]["vendor_holdout"]:
        assert vendor in holdout_vendors
        assert customer in train_customers
    for vendor, customer in split["test_slices"]["double_holdout"]:
        assert vendor in holdout_vendors
        assert customer in holdout_customers


def test_ranking_metrics_reward_early_positives() -> None:
    best = np.array([1, 1, 0, 0], dtype=np.int8)
    worst = np.array([0, 0, 1, 1], dtype=np.int8)

    assert average_precision(best) > average_precision(worst)
    assert ndcg_at_k(best, 4) > ndcg_at_k(worst, 4)


def test_evaluate_slice_excludes_training_pairs() -> None:
    vendor_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    customer_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    definition = SliceDefinition(
        edges=[(0, 1)],
        candidate_customers=[0, 1, 2, 3],
        excluded_pairs={(0, 0)},
    )

    result = evaluate_slice(
        vendor_embeddings=vendor_embeddings,
        customer_embeddings=customer_embeddings,
        definition=definition,
        known_pairs={(0, 0), (0, 1)},
        seed=42,
    )

    assert result["n_queries"] == 1
    assert result["mean_average_precision"] == 1.0
    assert result["mean_candidate_customers"] == 3.0


def test_popularity_scores_can_use_scalar_embeddings() -> None:
    definition = SliceDefinition(
        edges=[(0, 1)],
        candidate_customers=[0, 1, 2],
        excluded_pairs={(0, 0)},
    )
    result = evaluate_slice(
        vendor_embeddings=np.ones((1, 1), dtype=np.float32),
        customer_embeddings=np.array([[10.0], [5.0], [1.0]], dtype=np.float32),
        definition=definition,
        known_pairs={(0, 0), (0, 1)},
        seed=42,
    )

    assert result["mean_average_precision"] == 1.0


def test_degree_matched_comparisons_hold_popularity_constant() -> None:
    train_edges = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    test_edges = [(2, 0)]
    comparisons, diagnostics = build_degree_matched_comparisons(
        train_edges=train_edges,
        test_edges=test_edges,
        all_pairs=set(train_edges + test_edges),
        n_customers=4,
        repeats=3,
        seed=42,
    )

    degree = np.bincount(np.asarray(train_edges)[:, 1], minlength=4)
    assert diagnostics["n_covered_test_edges"] == 1
    assert comparisons.shape[0] == 1
    _, positive, negative = comparisons[0]
    assert degree[positive] == degree[negative]
