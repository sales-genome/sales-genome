#!/usr/bin/env python3
"""Map product-page vendor pains into the v6 canonical pain feature space.

This is a label-independent cosine baseline. Vendor text comes exclusively
from ``page_type == "product"`` extraction records. Training-customer evidence
is used only to compute pain-subtheme document frequencies for IDF weighting.

Usage:
  python experiments/graph_embeddings/canonical_vendor_pain_features.py prepare \
    --gpu 4 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.graph_embeddings.product_page_vendor_features import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    TEXT_NORMALIZATION,
    _atomic_torch_save,
    _load_feature_mappings,
    _resolve_local_model,
    _sample_manifest,
    _sha256_file,
    _sha256_json,
    _sha256_tensor,
    _sha256_text,
    collect_product_page_samples,
    collect_product_page_samples_from_path,
)


GRAPH_DIR = ROOT / "experiments" / "graph_embeddings"
FEATURE_DIR = GRAPH_DIR / "features"

DEFAULT_EXTRACTIONS = ROOT / "graph" / "v3" / "extractions.jsonl"
DEFAULT_SOURCE_CACHE = (
    GRAPH_DIR / "inductive_product_page_vendor_features.pt"
)
DEFAULT_HETERO_DATA = GRAPH_DIR / "hetero_data.pt"
DEFAULT_FEATURE_MAPPINGS = FEATURE_DIR / "index_maps.pkl"
DEFAULT_PAIN_FEATURES = FEATURE_DIR / "pain_features.npy"
DEFAULT_OUTPUT = (
    GRAPH_DIR / "inductive_product_page_canonical_vendor_features.pt"
)
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

PIPELINE_VERSION = "canonical-product-page-vendor-pain-v1"
SCHEMA_VERSION = "inductive-canonical-product-page-vendor-pain-v1"
PAIN_LIMIT = 32
DEFAULT_SIMILARITY_THRESHOLD = 0.45
DEFAULT_CANONICAL_BATCH_SIZE = 8192
FEATURE_NAMES = (
    "product_page_canonical_pain_semantic",
    "product_page_canonical_pain_tfidf",
    "product_page_product_canonical_pain",
    "product_page_full_canonical_pain",
    "product_page_product_canonical_fusion",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(
        values.detach().cpu().to(torch.float32),
        p=2,
        dim=1,
        eps=1e-8,
    )


def _index_set_hash(indices: Sequence[int]) -> str:
    return _sha256_json(sorted(map(int, indices)))


def validate_source_sample_manifest(
    profiles: dict[str, dict[str, Any]],
    expected_manifest: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assert exact retained sample hashes/counts against the source cache."""
    actual_manifest = _sample_manifest(profiles)
    if actual_manifest == expected_manifest:
        return actual_manifest

    actual_vendors = set(actual_manifest)
    expected_vendors = set(expected_manifest)
    if actual_vendors != expected_vendors:
        missing = sorted(expected_vendors - actual_vendors)[:5]
        unexpected = sorted(actual_vendors - expected_vendors)[:5]
        raise AssertionError(
            "product-page sample manifest vendor IDs differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for vendor_id in sorted(expected_manifest):
        actual = actual_manifest[vendor_id]
        expected = expected_manifest[vendor_id]
        if actual != expected:
            differing_fields = sorted(
                key
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            raise AssertionError(
                "product-page sample manifest differs for "
                f"{vendor_id}: fields={differing_fields}"
            )
    raise AssertionError("product-page sample manifest differs")


def reconstruct_and_validate_samples(
    records: Iterable[dict[str, Any]],
    *,
    vendor_id_to_index: dict[str, int],
    expected_manifest: dict[str, dict[str, Any]],
    capability_limit: int,
    pain_limit: int = PAIN_LIMIT,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    """Reconstruct samples with the source collector and validate the manifest."""
    if pain_limit != PAIN_LIMIT:
        raise ValueError(
            f"canonical vendor pain features require pain_limit={PAIN_LIMIT}"
        )
    profiles, diagnostics = collect_product_page_samples(
        records,
        vendor_id_to_index=vendor_id_to_index,
        capability_limit=capability_limit,
        pain_limit=pain_limit,
    )
    manifest = validate_source_sample_manifest(profiles, expected_manifest)
    return profiles, diagnostics, manifest


def compute_training_subtheme_idf(
    customer_subtheme_features: torch.Tensor,
    train_customers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the independent-customer TF-IDF and semantic IDF formulas."""
    train_indices = list(map(int, train_customers))
    if not train_indices:
        raise ValueError("training customer population is empty")
    train_index = torch.tensor(train_indices, dtype=torch.long)
    training_rows = customer_subtheme_features.detach().cpu()[train_index]
    document_frequency = (
        (training_rows != 0).sum(dim=0).to(torch.float32)
    )
    n_documents = float(training_rows.shape[0])
    tfidf_idf = torch.log(
        torch.tensor(n_documents, dtype=torch.float32)
        / (document_frequency + 1.0)
    )
    semantic_idf = (
        torch.log(
            torch.tensor(n_documents + 1.0, dtype=torch.float32)
            / (document_frequency + 1.0)
        )
        + 1.0
    )
    return (
        tfidf_idf.numpy(),
        semantic_idf.numpy(),
        document_frequency.numpy(),
    )


def top1_cosine_matches(
    text_embeddings: np.ndarray | torch.Tensor,
    canonical_features: np.ndarray | torch.Tensor,
    *,
    threshold: float,
    device: str | torch.device,
    canonical_batch_size: int = DEFAULT_CANONICAL_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return thresholded top-1 indices and unthresholded cosine scores."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be between -1 and 1")
    if canonical_batch_size <= 0:
        raise ValueError("canonical_batch_size must be positive")

    text_values = torch.as_tensor(
        text_embeddings, dtype=torch.float32, device=device
    )
    canonical_values = torch.as_tensor(
        canonical_features, dtype=torch.float32, device=device
    )
    if text_values.ndim != 2 or canonical_values.ndim != 2:
        raise ValueError("text and canonical embeddings must be matrices")
    if text_values.shape[1] != canonical_values.shape[1]:
        raise ValueError("text and canonical embedding dimensions differ")
    if canonical_values.shape[0] == 0:
        raise ValueError("canonical feature matrix is empty")
    if not torch.isfinite(text_values).all():
        raise ValueError("text embeddings contain non-finite values")
    if not torch.isfinite(canonical_values).all():
        raise ValueError("canonical embeddings contain non-finite values")

    text_values = F.normalize(text_values, p=2, dim=1, eps=1e-8)
    canonical_values = F.normalize(
        canonical_values, p=2, dim=1, eps=1e-8
    )
    return _top1_cosine_normalized(
        text_values,
        canonical_values,
        threshold=threshold,
        canonical_batch_size=canonical_batch_size,
    )


def _top1_cosine_normalized(
    text_values: torch.Tensor,
    canonical_values: torch.Tensor,
    *,
    threshold: float,
    canonical_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized GPU tensors without recopying canonical features."""
    best_similarity = torch.full(
        (text_values.shape[0],),
        -float("inf"),
        dtype=torch.float32,
        device=text_values.device,
    )
    best_index = torch.full(
        (text_values.shape[0],),
        -1,
        dtype=torch.long,
        device=text_values.device,
    )
    for start in range(0, canonical_values.shape[0], canonical_batch_size):
        stop = min(start + canonical_batch_size, canonical_values.shape[0])
        similarities = text_values @ canonical_values[start:stop].T
        chunk_similarity, chunk_index = similarities.max(dim=1)
        improved = chunk_similarity > best_similarity
        best_similarity[improved] = chunk_similarity[improved]
        best_index[improved] = chunk_index[improved] + start

    matched_index = best_index.clone()
    matched_index[best_similarity < threshold] = -1
    return (
        matched_index.cpu().numpy(),
        best_similarity.cpu().numpy(),
    )


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {
            name: None
            for name in (
                "min",
                "p01",
                "p05",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
            )
        }
    quantile_values = np.quantile(
        array,
        [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
    )
    return {
        name: round(float(value), 6)
        for name, value in zip(
            (
                "min",
                "p01",
                "p05",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
            ),
            quantile_values,
        )
    }


def embed_and_map_retained_texts(
    *,
    profiles: dict[str, dict[str, Any]],
    model: Any,
    pain_features: np.ndarray,
    pain_index_to_id: dict[int, str],
    pain_to_subtheme: np.ndarray,
    subtheme_index_to_id: dict[int, str],
    batch_size: int,
    canonical_batch_size: int,
    threshold: float,
    device: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Embed every globally unique retained pain once and map by cosine."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if canonical_batch_size <= 0:
        raise ValueError("canonical_batch_size must be positive")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be between -1 and 1")
    texts = sorted(
        {
            sample["text"]
            for profile in profiles.values()
            for sample in profile["pains"]
        }
    )
    mappings: dict[str, dict[str, Any]] = {}
    similarities: list[float] = []
    matched_similarities: list[float] = []
    unmatched_similarities: list[float] = []
    canonical_values = F.normalize(
        torch.as_tensor(
            pain_features, dtype=torch.float32, device=device
        ),
        p=2,
        dim=1,
        eps=1e-8,
    )

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = np.asarray(
            model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if encoded.shape != (len(batch), pain_features.shape[1]):
            raise ValueError(
                f"unexpected embedding shape {encoded.shape}; expected "
                f"{(len(batch), pain_features.shape[1])}"
            )
        encoded_values = F.normalize(
            torch.as_tensor(
                encoded, dtype=torch.float32, device=device
            ),
            p=2,
            dim=1,
            eps=1e-8,
        )
        matched_indices, scores = _top1_cosine_normalized(
            encoded_values,
            canonical_values,
            threshold=threshold,
            canonical_batch_size=canonical_batch_size,
        )
        for text, pain_index, similarity in zip(
            batch, matched_indices, scores
        ):
            score = float(similarity)
            similarities.append(score)
            if pain_index < 0:
                matched = False
                canonical_pain_id = None
                subtheme_index = None
                subtheme_id = None
                unmatched_similarities.append(score)
            else:
                matched = True
                canonical_pain_id = pain_index_to_id[int(pain_index)]
                subtheme_index = int(pain_to_subtheme[int(pain_index)])
                subtheme_id = subtheme_index_to_id[subtheme_index]
                matched_similarities.append(score)
            mappings[text] = {
                "normalized_text": text,
                "text_sha256": _sha256_text(text),
                "top1_similarity": round(score, 8),
                "matched": matched,
                "canonical_pain_index": (
                    int(pain_index) if matched else None
                ),
                "canonical_pain_id": canonical_pain_id,
                "subtheme_index": subtheme_index,
                "subtheme_id": subtheme_id,
            }
        completed = min(start + len(batch), len(texts))
        if completed == len(texts) or start == 0 or start // batch_size % 10 == 0:
            print(
                f"Embedded and mapped {completed:,}/{len(texts):,} "
                "globally unique retained pains",
                flush=True,
            )

    diagnostics = {
        "global_unique_retained_pain_texts": len(texts),
        "matched_unique_texts": len(matched_similarities),
        "unmatched_unique_texts": len(unmatched_similarities),
        "matched_fraction": (
            round(len(matched_similarities) / len(texts), 6)
            if texts
            else None
        ),
        "threshold": threshold,
        "similarity_quantiles": {
            "all": _quantiles(similarities),
            "matched": _quantiles(matched_similarities),
            "unmatched": _quantiles(unmatched_similarities),
        },
        "global_text_hash_sha256": _sha256_json(
            [_sha256_text(text) for text in texts]
        ),
    }
    return mappings, diagnostics


def build_canonical_vendor_features(
    *,
    profiles: dict[str, dict[str, Any]],
    text_mappings: dict[str, dict[str, Any]],
    pain_features: np.ndarray,
    pain_to_subtheme: np.ndarray,
    tfidf_subtheme_idf: np.ndarray,
    semantic_subtheme_idf: np.ndarray,
    n_vendors: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, dict[str, Any]]]:
    """Build independent-customer-compatible semantic and TF-IDF blocks."""
    pain_values = np.asarray(pain_features, dtype=np.float32)
    if pain_values.ndim != 2:
        raise ValueError("pain features must be a matrix")
    pain_norms = np.linalg.norm(pain_values, axis=1, keepdims=True)
    normalized_pains = pain_values / np.maximum(pain_norms, 1e-8)
    n_subthemes = tfidf_subtheme_idf.shape[0]
    if semantic_subtheme_idf.shape != (n_subthemes,):
        raise ValueError("semantic and TF-IDF subtheme IDF dimensions differ")
    if pain_to_subtheme.shape != (pain_values.shape[0],):
        raise ValueError("pain-to-subtheme mapping dimension differs")

    semantic = np.zeros(
        (n_vendors, pain_values.shape[1]), dtype=np.float32
    )
    tfidf = np.zeros((n_vendors, n_subthemes), dtype=np.float32)
    vendor_stats: dict[int, dict[str, Any]] = {}

    for profile in sorted(
        profiles.values(), key=lambda values: int(values["vendor_index"])
    ):
        vendor_index = int(profile["vendor_index"])
        if not 0 <= vendor_index < n_vendors:
            raise ValueError(f"vendor index out of range: {vendor_index}")
        selected_indices: list[int] = []
        seen_indices: set[int] = set()
        matched_samples = 0
        unmatched_hashes: list[str] = []
        for sample in profile["pains"]:
            mapping = text_mappings[sample["text"]]
            pain_index = mapping["canonical_pain_index"]
            if pain_index is None:
                unmatched_hashes.append(sample["sha256"])
                continue
            matched_samples += 1
            pain_index = int(pain_index)
            if pain_index not in seen_indices:
                seen_indices.add(pain_index)
                selected_indices.append(pain_index)

        semantic_sum = np.zeros(pain_values.shape[1], dtype=np.float32)
        semantic_weight_sum = 0.0
        subtheme_counts = np.zeros(n_subthemes, dtype=np.float32)
        for pain_index in selected_indices:
            subtheme_index = int(pain_to_subtheme[pain_index])
            if not 0 <= subtheme_index < n_subthemes:
                raise ValueError(
                    f"canonical pain {pain_index} lacks a valid subtheme"
                )
            subtheme_counts[subtheme_index] += 1.0
            weight = float(semantic_subtheme_idf[subtheme_index])
            semantic_sum += normalized_pains[pain_index] * weight
            semantic_weight_sum += weight

        if semantic_weight_sum > 0:
            semantic_sum /= semantic_weight_sum
        semantic[vendor_index] = semantic_sum
        tfidf[vendor_index] = subtheme_counts * tfidf_subtheme_idf
        selected_subthemes = sorted(
            {int(pain_to_subtheme[index]) for index in selected_indices}
        )
        vendor_stats[vendor_index] = {
            "retained_pain_texts": len(profile["pains"]),
            "matched_pain_texts": matched_samples,
            "unmatched_pain_texts": len(unmatched_hashes),
            "unmatched_text_sha256": unmatched_hashes,
            "selected_unique_canonical_pains": len(selected_indices),
            "selected_canonical_pain_indices": selected_indices,
            "selected_unique_subthemes": len(selected_subthemes),
            "selected_subtheme_indices": selected_subthemes,
        }

    semantic_tensor = _normalize_rows(torch.from_numpy(semantic))
    tfidf_tensor = _normalize_rows(torch.from_numpy(tfidf))
    return semantic_tensor, tfidf_tensor, vendor_stats


def materialize_feature_variants(
    *,
    product: torch.Tensor,
    raw_capability: torch.Tensor,
    canonical_semantic: torch.Tensor,
    canonical_tfidf: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Create equal-block-norm variants consumed by inductive_baseline."""
    blocks = {
        "product": product,
        "raw_capability": raw_capability,
        "canonical_semantic": canonical_semantic,
        "canonical_tfidf": canonical_tfidf,
    }
    if any(values.ndim != 2 for values in blocks.values()):
        raise ValueError("vendor feature blocks must be matrices")
    row_counts = {values.shape[0] for values in blocks.values()}
    if len(row_counts) != 1:
        raise ValueError("vendor feature blocks disagree on row count")
    for name in ("product", "raw_capability", "canonical_semantic"):
        if blocks[name].shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"{name} must be {EMBEDDING_DIMENSION}d")

    product_norm = _normalize_rows(product)
    raw_capability_norm = _normalize_rows(raw_capability)
    semantic_norm = _normalize_rows(canonical_semantic)
    tfidf_norm = _normalize_rows(canonical_tfidf)
    return {
        "product_page_canonical_pain_semantic": semantic_norm,
        "product_page_canonical_pain_tfidf": tfidf_norm,
        "product_page_product_canonical_pain": torch.cat(
            [product_norm, semantic_norm], dim=1
        )
        / math.sqrt(2.0),
        "product_page_full_canonical_pain": torch.cat(
            [product_norm, raw_capability_norm, semantic_norm], dim=1
        )
        / math.sqrt(3.0),
        "product_page_product_canonical_fusion": torch.cat(
            [product_norm, semantic_norm, tfidf_norm], dim=1
        )
        / math.sqrt(3.0),
    }


def _mapping_coverage(
    *,
    indices: Iterable[int],
    vendor_stats: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    selected = sorted(set(map(int, indices)))
    totals: Counter[str] = Counter()
    for index in selected:
        stats = vendor_stats[index]
        for field in (
            "retained_pain_texts",
            "matched_pain_texts",
            "unmatched_pain_texts",
            "selected_unique_canonical_pains",
            "selected_unique_subthemes",
        ):
            totals[field] += int(stats[field])
        totals["vendors_with_retained_pains"] += int(
            stats["retained_pain_texts"] > 0
        )
        totals["vendors_with_matched_pains"] += int(
            stats["matched_pain_texts"] > 0
        )

    retained = int(totals["retained_pain_texts"])
    return {
        "vendors": len(selected),
        "vendor_indices_sha256": _index_set_hash(selected),
        "vendors_with_retained_pains": int(
            totals["vendors_with_retained_pains"]
        ),
        "vendors_with_matched_pains": int(
            totals["vendors_with_matched_pains"]
        ),
        "retained_pain_texts": retained,
        "matched_pain_texts": int(totals["matched_pain_texts"]),
        "unmatched_pain_texts": int(totals["unmatched_pain_texts"]),
        "mapping_fraction": (
            round(totals["matched_pain_texts"] / retained, 6)
            if retained
            else None
        ),
        "selected_unique_canonical_pains": int(
            totals["selected_unique_canonical_pains"]
        ),
        "selected_unique_subthemes": int(
            totals["selected_unique_subthemes"]
        ),
    }


def build_mapping_coverage(
    *,
    vendor_stats: dict[int, dict[str, Any]],
    split: dict[str, Any],
    n_vendors: int,
) -> dict[str, Any]:
    double_holdout_vendors = sorted(
        {
            int(vendor)
            for vendor, _ in split.get("test_slices", {}).get(
                "double_holdout", []
            )
        }
    )
    return {
        "overall": _mapping_coverage(
            indices=range(n_vendors), vendor_stats=vendor_stats
        ),
        "train": _mapping_coverage(
            indices=split["train_vendors"], vendor_stats=vendor_stats
        ),
        "holdout": _mapping_coverage(
            indices=split["holdout_vendors"], vendor_stats=vendor_stats
        ),
        "double_holdout": _mapping_coverage(
            indices=double_holdout_vendors, vendor_stats=vendor_stats
        ),
    }


def _load_graph_context(
    *,
    hetero_data_path: Path,
    vendor_id_to_index: dict[str, int],
    pain_features_path: Path,
) -> tuple[
    Any,
    np.ndarray,
    np.ndarray,
    dict[int, str],
    dict[int, str],
    dict[str, Any],
]:
    checkpoint = torch.load(
        hetero_data_path, map_location="cpu", weights_only=False
    )
    data = checkpoint["data"]
    node_maps = checkpoint.get("node_maps")
    if not isinstance(node_maps, dict):
        raise ValueError("hetero_data cache lacks node_maps")
    hetero_vendors = {
        str(key): int(value)
        for key, value in node_maps.get("vendor", {}).items()
    }
    if hetero_vendors != vendor_id_to_index:
        raise ValueError(
            "feature mappings and hetero_data vendor mappings disagree"
        )

    pain_id_to_index = {
        str(key): int(value)
        for key, value in node_maps.get("pain_canonical", {}).items()
    }
    subtheme_id_to_index = {
        str(key): int(value)
        for key, value in node_maps.get("pain_subtheme", {}).items()
    }
    pain_index_to_id = {
        index: node_id for node_id, index in pain_id_to_index.items()
    }
    subtheme_index_to_id = {
        index: node_id for node_id, index in subtheme_id_to_index.items()
    }
    if sorted(pain_index_to_id) != list(range(len(pain_index_to_id))):
        raise ValueError("canonical pain indices are not contiguous")
    if sorted(subtheme_index_to_id) != list(
        range(len(subtheme_index_to_id))
    ):
        raise ValueError("pain subtheme indices are not contiguous")

    pain_features = np.asarray(
        np.load(pain_features_path), dtype=np.float32
    )
    if pain_features.shape != (
        len(pain_index_to_id),
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            "pain feature shape does not match hetero pain mapping: "
            f"{pain_features.shape}"
        )
    if not np.isfinite(pain_features).all():
        raise ValueError("pain feature matrix contains non-finite values")
    pain_to_subtheme = np.full(
        pain_features.shape[0], -1, dtype=np.int64
    )
    subtheme_edges = data[
        "pain_canonical", "in_subtheme", "pain_subtheme"
    ].edge_index.cpu()
    pain_to_subtheme[
        subtheme_edges[0].numpy()
    ] = subtheme_edges[1].numpy()
    if np.any(pain_to_subtheme < 0):
        raise ValueError("one or more canonical pains lack a subtheme mapping")

    diagnostics = {
        "n_vendors": len(vendor_id_to_index),
        "n_canonical_pains": len(pain_index_to_id),
        "n_pain_subthemes": len(subtheme_index_to_id),
        "vendor_mapping_sha256": _sha256_json(vendor_id_to_index),
        "pain_mapping_sha256": _sha256_json(pain_id_to_index),
        "subtheme_mapping_sha256": _sha256_json(subtheme_id_to_index),
        "vendor_customer_edges_read_for_vendor_content": False,
        "training_customer_features_read_for_idf_only": True,
    }
    return (
        data,
        pain_features,
        pain_to_subtheme,
        pain_index_to_id,
        subtheme_index_to_id,
        diagnostics,
    )


def prepare_cache(
    *,
    extractions_path: Path,
    source_cache_path: Path,
    hetero_data_path: Path,
    feature_mappings_path: Path,
    pain_features_path: Path,
    output_path: Path,
    model_cache: Path,
    gpu: int,
    batch_size: int,
    canonical_batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    started = time.time()
    source = torch.load(
        source_cache_path, map_location="cpu", weights_only=False
    )
    source_profile = source.get("independent_product_page")
    if not isinstance(source_profile, dict):
        raise ValueError(
            "source cache lacks independent_product_page provenance"
        )
    expected_manifest = source_profile.get("sample_manifest")
    if not isinstance(expected_manifest, dict):
        raise ValueError("source cache lacks product-page sample manifest")
    source_config = source_profile.get("fingerprint_payload", {}).get(
        "config", {}
    )
    capability_limit = int(
        source_config.get("capability_limit_per_vendor", -1)
    )
    pain_limit = int(source_config.get("pain_limit_per_vendor", -1))
    if capability_limit < 0:
        raise ValueError("source cache lacks a valid capability sample limit")
    if pain_limit != PAIN_LIMIT:
        raise ValueError(
            f"source cache pain limit is {pain_limit}, expected {PAIN_LIMIT}"
        )

    vendor_id_to_index = _load_feature_mappings(feature_mappings_path)
    (
        data,
        pain_features,
        pain_to_subtheme,
        pain_index_to_id,
        subtheme_index_to_id,
        graph_diagnostics,
    ) = _load_graph_context(
        hetero_data_path=hetero_data_path,
        vendor_id_to_index=vendor_id_to_index,
        pain_features_path=pain_features_path,
    )

    print(f"Reconstructing product-page samples from {extractions_path}...")
    profiles, extraction_diagnostics, product_records_sha256 = (
        collect_product_page_samples_from_path(
            extractions_path,
            vendor_id_to_index=vendor_id_to_index,
            capability_limit=capability_limit,
            pain_limit=pain_limit,
        )
    )
    sample_manifest = validate_source_sample_manifest(
        profiles, expected_manifest
    )
    sample_manifest_sha256 = _sha256_json(sample_manifest)

    tfidf_subtheme_idf, semantic_subtheme_idf, document_frequency = (
        compute_training_subtheme_idf(
            data["customer"].x,
            source["split"]["train_customers"],
        )
    )
    if tfidf_subtheme_idf.shape[0] != len(subtheme_index_to_id):
        raise ValueError(
            "training customer subtheme features and graph subthemes differ"
        )

    model, model_provenance = _resolve_local_model(
        model_name=EMBEDDING_MODEL,
        model_cache=model_cache,
        gpu=gpu,
    )
    text_mappings, mapping_diagnostics = embed_and_map_retained_texts(
        profiles=profiles,
        model=model,
        pain_features=pain_features,
        pain_index_to_id=pain_index_to_id,
        pain_to_subtheme=pain_to_subtheme,
        subtheme_index_to_id=subtheme_index_to_id,
        batch_size=batch_size,
        canonical_batch_size=canonical_batch_size,
        threshold=threshold,
        device=f"cuda:{gpu}",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    canonical_semantic, canonical_tfidf, vendor_stats = (
        build_canonical_vendor_features(
            profiles=profiles,
            text_mappings=text_mappings,
            pain_features=pain_features,
            pain_to_subtheme=pain_to_subtheme,
            tfidf_subtheme_idf=tfidf_subtheme_idf,
            semantic_subtheme_idf=semantic_subtheme_idf,
            n_vendors=len(vendor_id_to_index),
        )
    )
    source_vendor_features = source.get("vendor_features", {})
    product = source_vendor_features.get("product_only")
    raw_capability = source_vendor_features.get(
        "product_page_capability_only"
    )
    if not isinstance(product, torch.Tensor):
        raise ValueError("source cache lacks vendor_features.product_only")
    if not isinstance(raw_capability, torch.Tensor):
        raise ValueError(
            "source cache lacks product_page_capability_only"
        )
    variants = materialize_feature_variants(
        product=product,
        raw_capability=raw_capability,
        canonical_semantic=canonical_semantic,
        canonical_tfidf=canonical_tfidf,
    )
    if tuple(variants) != FEATURE_NAMES:
        raise AssertionError("canonical vendor feature names changed unexpectedly")
    coverage = build_mapping_coverage(
        vendor_stats=vendor_stats,
        split=source["split"],
        n_vendors=len(vendor_id_to_index),
    )

    text_mapping_records = [
        text_mappings[text] for text in sorted(text_mappings)
    ]
    vendor_mapping_manifest = {
        vendor_id: vendor_stats[int(profile["vendor_index"])]
        for vendor_id, profile in sorted(profiles.items())
    }
    feature_hashes = {
        name: _sha256_tensor(values)
        for name, values in variants.items()
    }
    input_hashes = {
        "product_page_records_sha256": product_records_sha256,
        "source_cache_sha256": _sha256_file(source_cache_path),
        "hetero_data_sha256": _sha256_file(hetero_data_path),
        "feature_mappings_sha256": _sha256_file(feature_mappings_path),
        "pain_features_sha256": _sha256_file(pain_features_path),
        "code_sha256": _sha256_file(Path(__file__)),
        "sample_collector_code_sha256": _sha256_file(
            GRAPH_DIR / "product_page_vendor_features.py"
        ),
        "sample_manifest_sha256": sample_manifest_sha256,
        "text_mapping_sha256": _sha256_json(text_mapping_records),
        "vendor_mapping_manifest_sha256": _sha256_json(
            vendor_mapping_manifest
        ),
    }
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "model_snapshot_revision": model_provenance[
            "snapshot_revision"
        ],
        "text_normalization": TEXT_NORMALIZATION,
        "pain_limit_per_vendor": PAIN_LIMIT,
        "embedding_batch_size": batch_size,
        "canonical_cosine_batch_size": canonical_batch_size,
        "similarity_threshold": threshold,
        "canonical_selection": (
            "top-1 cosine; below threshold is explicitly unmatched"
        ),
        "vendor_canonical_deduplication": (
            "unique selected canonical pain IDs per vendor"
        ),
        "tfidf_idf_formula": (
            "log(n_train_customers / (train_subtheme_df + 1))"
        ),
        "semantic_pain_weight_formula": (
            "log((n_train_customers + 1) / "
            "(train_subtheme_df + 1)) + 1"
        ),
        "feature_block_scaling": {
            "single": "unit row norm",
            "two_blocks": (
                "unit-normalized blocks concatenated / sqrt(2)"
            ),
            "three_blocks": (
                "unit-normalized blocks concatenated / sqrt(3)"
            ),
        },
    }
    fingerprint_payload = {
        "source_cache_fingerprint": source.get("fingerprint"),
        "inputs": input_hashes,
        "config": config,
        "feature_hashes": feature_hashes,
    }
    fingerprint = _sha256_json(fingerprint_payload)

    artifact = dict(source)
    artifact["created_at"] = _utc_now()
    artifact["fingerprint"] = fingerprint
    artifact["source_fingerprint"] = source.get("fingerprint")
    artifact["vendor_features"] = dict(source["vendor_features"])
    artifact["vendor_features"].update(variants)
    artifact["customer_features"] = source["customer_features"]
    artifact["split"] = source["split"]
    artifact["pairs"] = source["pairs"]
    artifact["edge_transfer"] = source["edge_transfer"]
    artifact["oracle_profile_warning"] = (
        "Canonical product-page vendor pain features use only product-page "
        "pain_points_cited text plus source product/raw-capability blocks. "
        "Case-study records and vendor-customer edges do not construct vendor "
        "content. Training-customer pain-subtheme incidence is used only for "
        "IDF. Customer features retain the source cache's graph-derived evidence."
    )
    artifact["diagnostics"] = dict(source.get("diagnostics", {}))
    artifact["diagnostics"]["canonical_vendor_pain_features"] = {
        "extractions": extraction_diagnostics,
        "graph": graph_diagnostics,
        "mapping": mapping_diagnostics,
        "coverage": coverage,
        "idf": {
            "document_population": "source split train_customers only",
            "n_train_customers": len(source["split"]["train_customers"]),
            "subtheme_document_frequency_quantiles": _quantiles(
                document_frequency.tolist()
            ),
            "tfidf_idf_quantiles": _quantiles(
                tfidf_subtheme_idf.tolist()
            ),
            "semantic_idf_quantiles": _quantiles(
                semantic_subtheme_idf.tolist()
            ),
        },
        "feature_dimensions": {
            name: int(values.shape[1])
            for name, values in variants.items()
        },
        "feature_sha256": feature_hashes,
        "sample_manifest_sha256": sample_manifest_sha256,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    artifact["canonical_product_page"] = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "fingerprint_payload": fingerprint_payload,
        "sample_manifest_validation": {
            "status": "exact_match",
            "vendors": len(sample_manifest),
            "sample_manifest_sha256": sample_manifest_sha256,
            "source_sample_manifest_sha256": source.get(
                "diagnostics", {}
            ).get("product_page_vendor_features", {}).get(
                "sample_manifest_sha256"
            ),
        },
        "text_mappings": text_mapping_records,
        "vendor_mapping_manifest": vendor_mapping_manifest,
        "provenance": {
            "extractions": str(extractions_path.resolve()),
            "source_cache": str(source_cache_path.resolve()),
            "hetero_data": str(hetero_data_path.resolve()),
            "feature_mappings": str(feature_mappings_path.resolve()),
            "pain_features": str(pain_features_path.resolve()),
            "code": str(Path(__file__).resolve()),
            "sample_collector_code": str(
                (GRAPH_DIR / "product_page_vendor_features.py").resolve()
            ),
            "embedding_model": model_provenance,
        },
        "construction_contract": {
            "page_type_filter": "product",
            "vendor_text_field": "data.pain_points_cited",
            "retained_pain_limit_per_vendor": PAIN_LIMIT,
            "uses_case_study_records_for_vendor_features": False,
            "uses_vendor_customer_edges_for_vendor_features": False,
            "training_customer_edges_usage": (
                "pain-subtheme document frequency for IDF only"
            ),
            "source_product_block_reused": "product_only",
            "source_raw_capability_block_reused": (
                "product_page_capability_only"
            ),
            "source_other_vendor_blocks_modified": False,
            "customer_features_preserved": True,
            "split_preserved": True,
            "pairs_preserved": True,
            "global_text_deduplication_before_embedding": True,
            "cosine_refinement_model": None,
            "qwen_refinement_used": False,
            "below_threshold_policy": (
                "canonical_pain_id, canonical_pain_index, subtheme_id, "
                "and subtheme_index are persisted as null"
            ),
        },
    }

    _atomic_torch_save(output_path, artifact)
    summary = {
        "output": str(output_path),
        "fingerprint": fingerprint,
        "source_fingerprint": source.get("fingerprint"),
        "features": artifact["diagnostics"][
            "canonical_vendor_pain_features"
        ]["feature_dimensions"],
        "mapping": mapping_diagnostics,
        "coverage": coverage,
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--extractions", type=Path, default=DEFAULT_EXTRACTIONS
    )
    prepare.add_argument(
        "--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE
    )
    prepare.add_argument(
        "--hetero-data", type=Path, default=DEFAULT_HETERO_DATA
    )
    prepare.add_argument(
        "--feature-mappings", type=Path, default=DEFAULT_FEATURE_MAPPINGS
    )
    prepare.add_argument(
        "--pain-features", type=Path, default=DEFAULT_PAIN_FEATURES
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument(
        "--model-cache", type=Path, default=DEFAULT_MODEL_CACHE
    )
    prepare.add_argument("--gpu", type=int, required=True)
    prepare.add_argument("--batch-size", type=int, default=512)
    prepare.add_argument(
        "--canonical-batch-size",
        type=int,
        default=DEFAULT_CANONICAL_BATCH_SIZE,
    )
    prepare.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_cache(
            extractions_path=args.extractions,
            source_cache_path=args.source_cache,
            hetero_data_path=args.hetero_data,
            feature_mappings_path=args.feature_mappings,
            pain_features_path=args.pain_features,
            output_path=args.output,
            model_cache=args.model_cache,
            gpu=args.gpu,
            batch_size=args.batch_size,
            canonical_batch_size=args.canonical_batch_size,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
