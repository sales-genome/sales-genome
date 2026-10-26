#!/usr/bin/env python3
"""Prepare and train the first inductive vendor-customer ranking baselines.

The prepared features intentionally use directional semantic aggregation:
canonical capability and pain features are anchors read by products/vendors or
customers. Canonical nodes do not aggregate entity messages.

This is an oracle-profile benchmark. Graph customer pain edges originate from
case-study evidence, not independent prospect research, so cold-start results
measure representation viability rather than production-ready CRM transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "experiments" / "graph_embeddings"
DEFAULT_CACHE = GRAPH_DIR / "inductive_features.pt"
DEFAULT_RESULTS = GRAPH_DIR / "results" / "inductive"

VENDOR_FEATURES = (
    "product_only",
    "d100",
    "unpruned",
    "inverse_degree",
    "product_page_capability_only",
    "product_page_pain_only",
    "product_page_product_capability",
    "product_page_product_pain",
    "product_page_full",
    "product_page_canonical_pain_semantic",
    "product_page_canonical_pain_tfidf",
    "product_page_product_canonical_pain",
    "product_page_full_canonical_pain",
    "product_page_product_canonical_fusion",
)
CUSTOMER_FEATURES = (
    "audience",
    "semantic_only",
    "tfidf_only",
    "semantic",
    "tfidf",
    "fusion",
)
EVALUATION_PROTOCOLS = ("edge_transfer", "oracle_node_holdout")
SLICES = ("customer_holdout", "vendor_holdout", "double_holdout")


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, p=2, dim=1, eps=1e-8)


def aggregate_neighbor_features(
    *,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    target_features: torch.Tensor,
    n_sources: int,
    edge_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted mean target features and total source weights."""
    source_index = source_index.to(dtype=torch.long, device="cpu")
    target_index = target_index.to(dtype=torch.long, device="cpu")
    target_features = target_features.to(dtype=torch.float32, device="cpu")
    if edge_weights is None:
        edge_weights = torch.ones(source_index.shape[0], dtype=torch.float32)
    else:
        edge_weights = edge_weights.to(dtype=torch.float32, device="cpu")
    if source_index.shape != target_index.shape or source_index.shape != edge_weights.shape:
        raise ValueError("edge indexes and weights must have matching shapes")

    adjacency = torch.sparse_coo_tensor(
        torch.stack([source_index, target_index]),
        edge_weights,
        size=(n_sources, target_features.shape[0]),
    ).coalesce()
    totals = torch.sparse.sum(adjacency, dim=1).to_dense()
    pooled = torch.sparse.mm(adjacency, target_features)
    pooled /= totals.clamp_min(1e-8).unsqueeze(1)
    pooled[totals == 0] = 0
    return pooled, totals


def make_node_split(
    pairs: list[tuple[int, int]],
    *,
    n_vendors: int,
    n_customers: int,
    holdout_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Create one disjoint vendor/customer split used by every configuration."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")

    active_vendors = sorted({vendor for vendor, _ in pairs})
    active_customers = sorted({customer for _, customer in pairs})
    rng = np.random.RandomState(seed)
    rng.shuffle(active_vendors)
    rng.shuffle(active_customers)

    n_holdout_vendors = max(1, round(len(active_vendors) * holdout_fraction))
    n_holdout_customers = max(1, round(len(active_customers) * holdout_fraction))
    holdout_vendors = set(active_vendors[:n_holdout_vendors])
    holdout_customers = set(active_customers[:n_holdout_customers])

    train_edges: list[tuple[int, int]] = []
    slices: dict[str, list[tuple[int, int]]] = {name: [] for name in SLICES}
    for vendor, customer in pairs:
        vendor_heldout = vendor in holdout_vendors
        customer_heldout = customer in holdout_customers
        if not vendor_heldout and not customer_heldout:
            train_edges.append((vendor, customer))
        elif not vendor_heldout and customer_heldout:
            slices["customer_holdout"].append((vendor, customer))
        elif vendor_heldout and not customer_heldout:
            slices["vendor_holdout"].append((vendor, customer))
        else:
            slices["double_holdout"].append((vendor, customer))

    train_vendors = sorted(set(range(n_vendors)) - holdout_vendors)
    train_customers = sorted(set(range(n_customers)) - holdout_customers)
    return {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "train_vendors": train_vendors,
        "holdout_vendors": sorted(holdout_vendors),
        "train_customers": train_customers,
        "holdout_customers": sorted(holdout_customers),
        "train_edges": train_edges,
        "test_slices": slices,
    }


def average_precision(labels_in_rank_order: np.ndarray) -> float:
    positives = int(labels_in_rank_order.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(labels_in_rank_order)
    ranks = np.arange(1, labels_in_rank_order.shape[0] + 1)
    return float((cumulative / ranks * labels_in_rank_order).sum() / positives)


def ndcg_at_k(labels_in_rank_order: np.ndarray, k: int) -> float:
    labels = labels_in_rank_order[:k]
    if not labels.any():
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, labels.shape[0] + 2))
    dcg = float((labels * discounts).sum())
    ideal_count = min(int(labels_in_rank_order.sum()), k)
    ideal = float(discounts[:ideal_count].sum())
    return dcg / ideal if ideal > 0 else 0.0


def _artifact_fingerprint(paths: list[Path], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _edge_degree_weights(
    target_index: torch.Tensor,
    *,
    n_targets: int,
    n_sources: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    degree = torch.bincount(target_index.cpu(), minlength=n_targets).to(torch.float32)
    specificity = torch.log(
        torch.tensor(float(n_sources + 1)) / (degree + 1.0)
    ).clamp_min(0.05)
    return specificity[target_index.cpu()], degree


def _vendor_features(
    *,
    product_features: torch.Tensor,
    cap_features: torch.Tensor,
    provides: torch.Tensor,
    offers: torch.Tensor,
    n_vendors: int,
    cap_weighting: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n_products = product_features.shape[0]
    n_caps = cap_features.shape[0]
    edge_weights = None
    cap_degree = torch.bincount(provides[1].cpu(), minlength=n_caps)
    if cap_weighting == "inverse_degree":
        edge_weights, cap_degree_float = _edge_degree_weights(
            provides[1],
            n_targets=n_caps,
            n_sources=n_products,
        )
        cap_degree = cap_degree_float.to(torch.long)
    elif cap_weighting != "uniform":
        raise ValueError(f"Unknown cap weighting: {cap_weighting}")

    cap_pool, product_cap_weight = aggregate_neighbor_features(
        source_index=provides[0],
        target_index=provides[1],
        target_features=normalize_rows(cap_features),
        n_sources=n_products,
        edge_weights=edge_weights,
    )
    product_repr = torch.cat(
        [normalize_rows(product_features), normalize_rows(cap_pool)],
        dim=1,
    )
    vendor_repr, vendor_product_count = aggregate_neighbor_features(
        source_index=offers[0],
        target_index=offers[1],
        target_features=product_repr,
        n_sources=n_vendors,
    )
    nonzero_degree = cap_degree[cap_degree > 0]
    diagnostics = {
        "n_provides_edges": int(provides.shape[1]),
        "n_caps_with_product": int(nonzero_degree.shape[0]),
        "max_cap_product_degree": int(nonzero_degree.max().item())
        if nonzero_degree.numel()
        else 0,
        "p99_cap_product_degree": float(
            torch.quantile(nonzero_degree.float(), 0.99).item()
        )
        if nonzero_degree.numel()
        else 0.0,
        "products_with_caps": int((product_cap_weight > 0).sum().item()),
        "vendors_with_products": int((vendor_product_count > 0).sum().item()),
    }
    return vendor_repr, diagnostics


def _product_only_vendor_features(
    *,
    product_features: torch.Tensor,
    offers: torch.Tensor,
    n_vendors: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    vendor_repr, vendor_product_count = aggregate_neighbor_features(
        source_index=offers[0],
        target_index=offers[1],
        target_features=normalize_rows(product_features),
        n_sources=n_vendors,
    )
    return vendor_repr, {
        "vendors_with_products": int((vendor_product_count > 0).sum().item()),
        "max_products_per_vendor": int(vendor_product_count.max().item())
        if vendor_product_count.numel()
        else 0,
    }


def _customer_features(data: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    pain_features = normalize_rows(data["pain_canonical"].x.cpu())
    customer_pain = data["customer", "experiences", "pain_canonical"].edge_index.cpu()
    n_customers = data["customer"].x.shape[0]
    n_pains = pain_features.shape[0]

    tfidf = data["customer"].x.cpu().to(torch.float32)
    document_frequency = (tfidf != 0).sum(dim=0).to(torch.float32)
    subtheme_idf = (
        torch.log(torch.tensor(float(n_customers + 1)) / (document_frequency + 1.0))
        + 1.0
    )

    pain_subtheme = torch.full((n_pains,), -1, dtype=torch.long)
    subtheme_edges = data[
        "pain_canonical", "in_subtheme", "pain_subtheme"
    ].edge_index.cpu()
    pain_subtheme[subtheme_edges[0]] = subtheme_edges[1]
    pain_weights = torch.ones(customer_pain.shape[1], dtype=torch.float32)
    edge_subthemes = pain_subtheme[customer_pain[1]]
    mapped = edge_subthemes >= 0
    pain_weights[mapped] = subtheme_idf[edge_subthemes[mapped]]

    semantic, pain_weight_sum = aggregate_neighbor_features(
        source_index=customer_pain[0],
        target_index=customer_pain[1],
        target_features=pain_features,
        n_sources=n_customers,
        edge_weights=pain_weights,
    )

    n_audiences = int(data["audience"].num_nodes)
    audience = torch.zeros((n_customers, n_audiences), dtype=torch.float32)
    audience_edges = data["customer", "in_audience", "audience"].edge_index.cpu()
    audience[audience_edges[0], audience_edges[1]] = 1.0

    tfidf_norm = normalize_rows(tfidf)
    semantic_norm = normalize_rows(semantic)
    audience_norm = normalize_rows(audience)
    features = {
        "audience": audience_norm,
        "semantic_only": semantic_norm,
        "tfidf_only": tfidf_norm,
        "semantic": torch.cat([semantic_norm, audience_norm], dim=1)
        / math.sqrt(2.0),
        "tfidf": torch.cat([tfidf_norm, audience_norm], dim=1)
        / math.sqrt(2.0),
        "fusion": torch.cat(
            [semantic_norm, tfidf_norm, audience_norm],
            dim=1,
        )
        / math.sqrt(3.0),
    }
    diagnostics = {
        "n_experiences_edges": int(customer_pain.shape[1]),
        "customers_with_pains": int((pain_weight_sum > 0).sum().item()),
        "customers_with_audiences": int((audience.sum(dim=1) > 0).sum().item()),
        "audience_dim": int(features["audience"].shape[1]),
        "semantic_only_dim": int(features["semantic_only"].shape[1]),
        "tfidf_only_dim": int(features["tfidf_only"].shape[1]),
        "semantic_dim": int(features["semantic"].shape[1]),
        "tfidf_dim": int(features["tfidf"].shape[1]),
        "fusion_dim": int(features["fusion"].shape[1]),
    }
    return features, diagnostics


def prepare_cache(
    *,
    output: Path,
    split_seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    started = time.time()
    base_path = GRAPH_DIR / "hetero_data.pt"
    d100_path = GRAPH_DIR / "graph_variants" / "collapsed_res_d100.pt"
    print(f"Loading {base_path}...")
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    data = base_checkpoint["data"]
    all_pairs = sorted(
        set(map(tuple, base_checkpoint["train_vc"]))
        | set(map(tuple, base_checkpoint["test_vc"]))
    )
    n_vendors = int(data["vendor"].num_nodes)
    n_customers = int(data["customer"].x.shape[0])

    print("Building customer feature ablations...")
    customer_features, customer_diagnostics = _customer_features(data)

    offers = data["vendor", "offers", "product"].edge_index.cpu()
    product_features = data["product"].x.cpu().to(torch.float32)

    print("Building product-only vendor features...")
    product_only, product_only_diagnostics = _product_only_vendor_features(
        product_features=product_features,
        offers=offers,
        n_vendors=n_vendors,
    )

    print("Building unpruned vendor features...")
    unpruned, unpruned_diagnostics = _vendor_features(
        product_features=product_features,
        cap_features=data["cap_canonical"].x.cpu().to(torch.float32),
        provides=data["product", "provides", "cap_canonical"].edge_index.cpu(),
        offers=offers,
        n_vendors=n_vendors,
        cap_weighting="uniform",
    )

    print("Building inverse-degree vendor features...")
    inverse_degree, inverse_diagnostics = _vendor_features(
        product_features=product_features,
        cap_features=data["cap_canonical"].x.cpu().to(torch.float32),
        provides=data["product", "provides", "cap_canonical"].edge_index.cpu(),
        offers=offers,
        n_vendors=n_vendors,
        cap_weighting="inverse_degree",
    )

    print(f"Loading {d100_path}...")
    d100_checkpoint = torch.load(d100_path, map_location="cpu", weights_only=False)
    d100_data = d100_checkpoint["data"]
    print("Building d100 vendor features...")
    d100, d100_diagnostics = _vendor_features(
        product_features=d100_data["vendor_product"].x.cpu().to(torch.float32),
        cap_features=d100_data["cap_canonical"].x.cpu().to(torch.float32),
        provides=d100_data[
            "vendor_product", "provides", "cap_canonical"
        ].edge_index.cpu(),
        offers=offers,
        n_vendors=n_vendors,
        cap_weighting="uniform",
    )

    split = make_node_split(
        all_pairs,
        n_vendors=n_vendors,
        n_customers=n_customers,
        holdout_fraction=holdout_fraction,
        seed=split_seed,
    )
    fingerprint_payload = {
        "split_seed": split_seed,
        "holdout_fraction": holdout_fraction,
        "n_pairs": len(all_pairs),
        "n_vendors": n_vendors,
        "n_customers": n_customers,
    }
    fingerprint = _artifact_fingerprint(
        [
            base_path,
            d100_path,
            GRAPH_DIR / "features" / "metadata.json",
            Path(__file__),
        ],
        fingerprint_payload,
    )
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "oracle_profile_warning": (
            "Customer pain edges come from case-study graph evidence. These "
            "features compare representation choices but are not independent "
            "prospect research."
        ),
        "vendor_features": {
            "product_only": product_only,
            "d100": d100,
            "unpruned": unpruned,
            "inverse_degree": inverse_degree,
        },
        "customer_features": customer_features,
        "pairs": all_pairs,
        "edge_transfer": {
            "train_edges": sorted(set(map(tuple, base_checkpoint["train_vc"]))),
            "test_edges": sorted(set(map(tuple, base_checkpoint["test_vc"]))),
        },
        "split": split,
        "diagnostics": {
            "customer": customer_diagnostics,
            "vendor": {
                "product_only": product_only_diagnostics,
                "d100": d100_diagnostics,
                "unpruned": unpruned_diagnostics,
                "inverse_degree": inverse_diagnostics,
            },
            "elapsed_seconds": round(time.time() - started, 3),
            "edge_transfer": {
                "train_edges": len(base_checkpoint["train_vc"]),
                "test_edges": len(base_checkpoint["test_vc"]),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(json.dumps({
        "output": str(output),
        "fingerprint": fingerprint,
        "diagnostics": artifact["diagnostics"],
        "split_counts": {
            "train": len(split["train_edges"]),
            **{
                name: len(edges)
                for name, edges in split["test_slices"].items()
            },
        },
    }, indent=2))
    return artifact


class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(values), dim=-1)


class InductiveTwoTower(nn.Module):
    def __init__(
        self,
        vendor_dim: int,
        customer_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vendor = Tower(vendor_dim, hidden_dim, output_dim, dropout)
        self.customer = Tower(customer_dim, hidden_dim, output_dim, dropout)


@dataclass(frozen=True)
class SliceDefinition:
    edges: list[tuple[int, int]]
    candidate_customers: list[int]
    excluded_pairs: set[tuple[int, int]]


def _slice_definitions(split: dict[str, Any]) -> dict[str, SliceDefinition]:
    return {
        "customer_holdout": SliceDefinition(
            edges=split["test_slices"]["customer_holdout"],
            candidate_customers=split["holdout_customers"],
            excluded_pairs=set(),
        ),
        "vendor_holdout": SliceDefinition(
            edges=split["test_slices"]["vendor_holdout"],
            candidate_customers=split["train_customers"],
            excluded_pairs=set(),
        ),
        "double_holdout": SliceDefinition(
            edges=split["test_slices"]["double_holdout"],
            candidate_customers=split["holdout_customers"],
            excluded_pairs=set(),
        ),
    }


def _edge_transfer_definition(
    *,
    train_edges: list[tuple[int, int]],
    test_edges: list[tuple[int, int]],
    n_customers: int,
) -> dict[str, SliceDefinition]:
    return {
        "edge_transfer": SliceDefinition(
            edges=test_edges,
            candidate_customers=list(range(n_customers)),
            excluded_pairs=set(train_edges),
        )
    }


def evaluate_slice(
    *,
    vendor_embeddings: np.ndarray,
    customer_embeddings: np.ndarray,
    definition: SliceDefinition,
    known_pairs: set[tuple[int, int]],
    seed: int,
) -> dict[str, Any]:
    positives: dict[int, set[int]] = defaultdict(set)
    for vendor, customer in definition.edges:
        positives[vendor].add(customer)

    candidate_set = set(definition.candidate_customers)
    aps: list[float] = []
    ndcgs: list[float] = []
    p10s: list[float] = []
    recall50s: list[float] = []
    sampled_p10s: list[float] = []
    chance_p10s: list[float] = []
    candidate_counts: list[int] = []
    rng = np.random.RandomState(seed)

    for vendor, vendor_positives in sorted(positives.items()):
        valid_positives = vendor_positives & candidate_set
        if not valid_positives:
            continue
        candidates = np.asarray(
            [
                customer
                for customer in definition.candidate_customers
                if (vendor, customer) not in definition.excluded_pairs
            ],
            dtype=np.int64,
        )
        scores = customer_embeddings[candidates] @ vendor_embeddings[vendor]
        candidate_counts.append(len(candidates))
        labels = np.fromiter(
            (int(customer in valid_positives) for customer in candidates),
            dtype=np.int8,
        )
        order = np.argsort(-scores)
        ranked_labels = labels[order]
        aps.append(average_precision(ranked_labels))
        ndcgs.append(ndcg_at_k(ranked_labels, 50))
        p10s.append(float(ranked_labels[:10].mean()))
        chance_p10s.append(len(valid_positives) / max(len(candidates), 1))
        recall50s.append(
            float(ranked_labels[:50].sum() / max(len(valid_positives), 1))
        )

        if len(valid_positives) >= 3:
            negatives = [
                customer
                for customer in definition.candidate_customers
                if (vendor, customer) not in known_pairs
            ]
            if len(negatives) >= 27:
                positive_sample = rng.choice(
                    sorted(valid_positives),
                    size=min(3, len(valid_positives)),
                    replace=False,
                )
                negative_sample = rng.choice(negatives, size=27, replace=False)
                pool = np.concatenate([positive_sample, negative_sample])
                pool_labels = np.concatenate([
                    np.ones(len(positive_sample), dtype=np.int8),
                    np.zeros(len(negative_sample), dtype=np.int8),
                ])
                pool_scores = customer_embeddings[pool] @ vendor_embeddings[vendor]
                sampled_order = np.argsort(-pool_scores)
                sampled_p10s.append(float(pool_labels[sampled_order][:10].mean()))

    def summary(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 6) if values else None

    return {
        "n_queries": len(aps),
        "n_positive_edges": sum(len(values) for values in positives.values()),
        "mean_candidate_customers": summary(
            [float(count) for count in candidate_counts]
        ),
        "mean_average_precision": summary(aps),
        "mean_ndcg_at_50": summary(ndcgs),
        "mean_precision_at_10": summary(p10s),
        "mean_chance_precision_at_10": summary(chance_p10s),
        "precision_at_10_lift": (
            round(float(np.mean(p10s) / np.mean(chance_p10s)), 6)
            if p10s and chance_p10s and np.mean(chance_p10s) > 0
            else None
        ),
        "mean_recall_at_50": summary(recall50s),
        "sampled_10pct_precision_at_10": summary(sampled_p10s),
        "sampled_queries": len(sampled_p10s),
    }


def _encode_in_batches(
    tower: Tower,
    features: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    tower.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            outputs.append(tower(features[start:start + batch_size]).cpu())
    return torch.cat(outputs, dim=0).numpy()


def train_configuration(
    *,
    cache_path: Path,
    output_dir: Path,
    vendor_feature: str,
    customer_feature: str,
    evaluation_protocol: str,
    device_name: str,
    seed: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    negatives: int,
    learning_rate: float,
    patience: int,
) -> dict[str, Any]:
    if vendor_feature not in VENDOR_FEATURES:
        raise ValueError(f"Unknown vendor feature: {vendor_feature}")
    if customer_feature not in CUSTOMER_FEATURES:
        raise ValueError(f"Unknown customer feature: {customer_feature}")
    if evaluation_protocol not in EVALUATION_PROTOCOLS:
        raise ValueError(f"Unknown evaluation protocol: {evaluation_protocol}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name)
    artifact = torch.load(cache_path, map_location="cpu", weights_only=False)
    vendor_values = artifact["vendor_features"][vendor_feature].to(torch.float32)
    customer_values = artifact["customer_features"][customer_feature].to(torch.float32)
    if evaluation_protocol == "edge_transfer":
        train_edges_list = artifact["edge_transfer"]["train_edges"]
        test_definitions = _edge_transfer_definition(
            train_edges=train_edges_list,
            test_edges=artifact["edge_transfer"]["test_edges"],
            n_customers=customer_values.shape[0],
        )
        active_vendors = sorted({vendor for vendor, _ in train_edges_list})
        active_customers = sorted({customer for _, customer in train_edges_list})
    else:
        split = artifact["split"]
        train_edges_list = split["train_edges"]
        test_definitions = _slice_definitions(split)
        active_vendors = sorted({vendor for vendor, _ in train_edges_list})
        active_customers = sorted({customer for _, customer in train_edges_list})

    model = InductiveTwoTower(
        vendor_dim=vendor_values.shape[1],
        customer_dim=customer_values.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    edge_array = np.asarray(train_edges_list, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(edge_array)
    n_validation = max(1, round(edge_array.shape[0] * 0.1))
    validation_edges = torch.from_numpy(edge_array[:n_validation]).to(device)
    training_edges = edge_array[n_validation:]

    train_customer_ids = torch.tensor(
        active_customers, dtype=torch.long, device=device
    )
    train_vendor_ids = torch.tensor(
        active_vendors, dtype=torch.long, device=device
    )
    vendor_values = vendor_values.to(device)
    customer_values = customer_values.to(device)

    known_pair_mask = torch.zeros(
        (vendor_values.shape[0], customer_values.shape[0]),
        dtype=torch.bool,
        device=device,
    )
    training_pairs = torch.tensor(
        train_edges_list,
        dtype=torch.long,
        device=device,
    )
    known_pair_mask[training_pairs[:, 0], training_pairs[:, 1]] = True

    def sample_negatives(
        vendor_ids: torch.Tensor,
        customer_ids: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        negative_customer_positions = torch.randint(
            0,
            train_customer_ids.shape[0],
            (vendor_ids.shape[0], negatives),
            device=device,
            generator=generator,
        )
        negative_customers = train_customer_ids[negative_customer_positions]
        for _ in range(4):
            invalid = known_pair_mask[vendor_ids.unsqueeze(1), negative_customers]
            if not invalid.any():
                break
            negative_customer_positions[invalid] = torch.randint(
                0,
                train_customer_ids.shape[0],
                (int(invalid.sum().item()),),
                device=device,
                generator=generator,
            )
            negative_customers = train_customer_ids[negative_customer_positions]

        negative_vendor_positions = torch.randint(
            0,
            train_vendor_ids.shape[0],
            (vendor_ids.shape[0], negatives),
            device=device,
            generator=generator,
        )
        negative_vendors = train_vendor_ids[negative_vendor_positions]
        for _ in range(4):
            invalid = known_pair_mask[negative_vendors, customer_ids.unsqueeze(1)]
            if not invalid.any():
                break
            negative_vendor_positions[invalid] = torch.randint(
                0,
                train_vendor_ids.shape[0],
                (int(invalid.sum().item()),),
                device=device,
                generator=generator,
            )
            negative_vendors = train_vendor_ids[negative_vendor_positions]
        return negative_customers, negative_vendors

    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(seed + 1_000_003)
    validation_negative_customers, validation_negative_vendors = sample_negatives(
        validation_edges[:, 0],
        validation_edges[:, 1],
        generator=validation_generator,
    )

    def batch_loss(
        edges: torch.Tensor,
        *,
        training: bool,
        fixed_negative_customers: torch.Tensor | None = None,
        fixed_negative_vendors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vendor_ids = edges[:, 0]
        customer_ids = edges[:, 1]
        vendor_embedding = model.vendor(vendor_values[vendor_ids])
        customer_embedding = model.customer(customer_values[customer_ids])
        positive = (vendor_embedding * customer_embedding).sum(dim=1, keepdim=True)

        if fixed_negative_customers is None or fixed_negative_vendors is None:
            negative_customers, negative_vendors = sample_negatives(
                vendor_ids,
                customer_ids,
            )
        else:
            negative_customers = fixed_negative_customers
            negative_vendors = fixed_negative_vendors
        negative_customer_embedding = model.customer(
            customer_values[negative_customers.reshape(-1)]
        ).reshape(edges.shape[0], negatives, -1)
        customer_logits = torch.cat(
            [
                positive,
                torch.bmm(
                    negative_customer_embedding,
                    vendor_embedding.unsqueeze(-1),
                ).squeeze(-1),
            ],
            dim=1,
        ) / 0.1

        negative_vendor_embedding = model.vendor(
            vendor_values[negative_vendors.reshape(-1)]
        ).reshape(edges.shape[0], negatives, -1)
        vendor_logits = torch.cat(
            [
                positive,
                torch.bmm(
                    negative_vendor_embedding,
                    customer_embedding.unsqueeze(-1),
                ).squeeze(-1),
            ],
            dim=1,
        ) / 0.1
        labels = torch.zeros(edges.shape[0], dtype=torch.long, device=device)
        loss = (
            F.cross_entropy(customer_logits, labels)
            + F.cross_entropy(vendor_logits, labels)
        ) / 2.0
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return loss.detach()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{evaluation_protocol}__{vendor_feature}__{customer_feature}"
        f"__seed{seed}"
    )
    checkpoint_path = output_dir / f"{run_name}.pt"
    result_path = output_dir / f"{run_name}.json"

    best_validation = math.inf
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_edges = training_edges.copy()
        rng.shuffle(epoch_edges)
        losses = []
        for start in range(0, epoch_edges.shape[0], batch_size):
            edges = torch.from_numpy(
                epoch_edges[start:start + batch_size]
            ).to(device)
            losses.append(float(batch_loss(edges, training=True).item()))

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                batch_loss(
                    validation_edges,
                    training=False,
                    fixed_negative_customers=validation_negative_customers,
                    fixed_negative_vendors=validation_negative_vendors,
                ).item()
            )
        mean_loss = float(np.mean(losses))
        history.append({
            "epoch": epoch,
            "train_loss": round(mean_loss, 6),
            "validation_loss": round(validation_loss, 6),
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch - best_epoch >= patience:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{run_name} epoch={epoch} train={mean_loss:.4f} "
                f"validation={validation_loss:.4f}"
            )

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
    known_pairs = set(map(tuple, artifact["pairs"]))
    evaluations = {
        name: evaluate_slice(
            vendor_embeddings=vendor_embeddings,
            customer_embeddings=customer_embeddings,
            definition=definition,
            known_pairs=known_pairs,
            seed=seed,
        )
        for name, definition in test_definitions.items()
    }
    customer_train_degree = np.bincount(
        np.asarray(train_edges_list, dtype=np.int64)[:, 1],
        minlength=customer_values.shape[0],
    ).astype(np.float32)
    popularity_vendor_embeddings = np.ones(
        (vendor_values.shape[0], 1),
        dtype=np.float32,
    )
    popularity_customer_embeddings = customer_train_degree[:, None]
    popularity_baseline = {
        name: evaluate_slice(
            vendor_embeddings=popularity_vendor_embeddings,
            customer_embeddings=popularity_customer_embeddings,
            definition=definition,
            known_pairs=known_pairs,
            seed=seed,
        )
        for name, definition in test_definitions.items()
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "cache": str(cache_path),
        "cache_fingerprint": artifact["fingerprint"],
        "oracle_profile_warning": artifact["oracle_profile_warning"],
        "configuration": {
            "vendor_feature": vendor_feature,
            "customer_feature": customer_feature,
            "evaluation_protocol": evaluation_protocol,
            "device": str(device),
            "seed": seed,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "negatives": negatives,
            "learning_rate": learning_rate,
            "patience": patience,
        },
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_validation, 6),
        "elapsed_seconds": round(time.time() - started, 3),
        "evaluations": evaluations,
        "baselines": {
            "customer_train_degree_popularity": popularity_baseline,
        },
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "result": str(result_path),
        "best_epoch": best_epoch,
        "evaluations": evaluations,
    }, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    prepare.add_argument("--split-seed", type=int, default=42)
    prepare.add_argument("--holdout-fraction", type=float, default=0.2)

    train = subparsers.add_parser("train")
    train.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    train.add_argument("--vendor-feature", choices=VENDOR_FEATURES, required=True)
    train.add_argument("--customer-feature", choices=CUSTOMER_FEATURES, required=True)
    train.add_argument(
        "--evaluation-protocol",
        choices=EVALUATION_PROTOCOLS,
        default="edge_transfer",
    )
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--output-dim", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--negatives", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_cache(
            output=args.output,
            split_seed=args.split_seed,
            holdout_fraction=args.holdout_fraction,
        )
    elif args.command == "train":
        train_configuration(
            cache_path=args.cache,
            output_dir=args.output_dir,
            vendor_feature=args.vendor_feature,
            customer_feature=args.customer_feature,
            evaluation_protocol=args.evaluation_protocol,
            device_name=args.device,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            negatives=args.negatives,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
