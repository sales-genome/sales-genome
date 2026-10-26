#!/usr/bin/env python3
"""Build label-independent vendor features from product-page extractions.

Only ``page_type == "product"`` extraction records contribute text. The
resulting cache preserves the source customer features and evaluation splits,
while adding vendor feature blocks compatible with
``inductive_baseline.train_configuration``.

Usage:
  python experiments/graph_embeddings/product_page_vendor_features.py prepare \
    --gpu 4 --batch-size 1024
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "experiments" / "graph_embeddings"
FEATURE_DIR = GRAPH_DIR / "features"

DEFAULT_EXTRACTIONS = ROOT / "graph" / "v3" / "extractions.jsonl"
DEFAULT_SOURCE_CACHE = GRAPH_DIR / "inductive_features_vendor_base.pt"
DEFAULT_HETERO_DATA = GRAPH_DIR / "hetero_data.pt"
DEFAULT_FEATURE_MAPPINGS = FEATURE_DIR / "index_maps.pkl"
DEFAULT_OUTPUT = GRAPH_DIR / "inductive_features_product_page.pt"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024
PIPELINE_VERSION = "product-page-vendor-features-v1"
SCHEMA_VERSION = "inductive-product-page-vendor-features-v1"
TEXT_NORMALIZATION = "unicode_nfkc_whitespace_collapse_casefold"
FEATURE_NAMES = (
    "product_page_capability_only",
    "product_page_pain_only",
    "product_page_product_capability",
    "product_page_product_pain",
    "product_page_full",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _sha256_tensor(values: torch.Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def normalize_raw_text(value: Any) -> str:
    """Normalize text for exact matching without semantic name resolution."""
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def _normalize_vendor_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    slug = value.strip().casefold()
    if slug.startswith("vendor:"):
        slug = slug.removeprefix("vendor:")
    return f"vendor:{slug}" if slug else ""


def _normalized_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = ()
    for item in values:
        normalized = normalize_raw_text(item)
        if normalized:
            yield normalized


def _new_vendor_accumulator() -> dict[str, Any]:
    return {
        "product_record_count": 0,
        "source_urls": Counter(),
        "capabilities": Counter(),
        "pains": Counter(),
    }


def _rank_samples(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "text": text,
            "count": int(count),
            "sha256": _sha256_text(text),
        }
        for text, count in ranked[:limit]
    ]


def collect_product_page_samples(
    records: Iterable[dict[str, Any]],
    *,
    vendor_id_to_index: dict[str, int],
    capability_limit: int = 64,
    pain_limit: int = 32,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Collect deterministic, frequency-ranked exact product-page samples."""
    if capability_limit < 0 or pain_limit < 0:
        raise ValueError("sample limits must be non-negative")

    accumulators = {
        vendor_id: _new_vendor_accumulator()
        for vendor_id in vendor_id_to_index
    }
    page_type_counts: Counter[str] = Counter()
    unknown_product_vendors: Counter[str] = Counter()
    malformed_product_records = 0
    total_records = 0

    for record in records:
        total_records += 1
        page_type = record.get("page_type")
        page_type_counts[str(page_type or "<missing>")] += 1
        if page_type != "product":
            continue

        vendor_id = _normalize_vendor_id(record.get("vendor_slug"))
        if vendor_id not in accumulators:
            unknown_product_vendors[vendor_id or "<missing>"] += 1
            continue

        data = record.get("data")
        if not isinstance(data, dict):
            data = {}
            malformed_product_records += 1

        accumulator = accumulators[vendor_id]
        accumulator["product_record_count"] += 1
        url = record.get("url")
        if isinstance(url, str) and url.strip():
            accumulator["source_urls"][url.strip()] += 1
        accumulator["capabilities"].update(
            _normalized_values(data.get("capabilities_cited"))
        )
        accumulator["pains"].update(
            _normalized_values(data.get("pain_points_cited"))
        )

    profiles: dict[str, dict[str, Any]] = {}
    for vendor_id, vendor_index in sorted(
        vendor_id_to_index.items(), key=lambda item: item[1]
    ):
        accumulator = accumulators[vendor_id]
        capabilities = _rank_samples(
            accumulator["capabilities"], capability_limit
        )
        pains = _rank_samples(accumulator["pains"], pain_limit)
        source_urls = [
            {
                "sha256": _sha256_text(url),
                "count": int(count),
            }
            for url, count in sorted(
                accumulator["source_urls"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        profiles[vendor_id] = {
            "vendor_index": int(vendor_index),
            "product_record_count": int(accumulator["product_record_count"]),
            "source_urls": source_urls,
            "capability_total_mentions": int(
                sum(accumulator["capabilities"].values())
            ),
            "capability_unique_count": len(accumulator["capabilities"]),
            "capabilities": capabilities,
            "pain_total_mentions": int(sum(accumulator["pains"].values())),
            "pain_unique_count": len(accumulator["pains"]),
            "pains": pains,
        }

    diagnostics = {
        "total_extraction_records": total_records,
        "page_type_counts": dict(sorted(page_type_counts.items())),
        "product_records": int(page_type_counts["product"]),
        "ignored_non_product_records": total_records - page_type_counts["product"],
        "malformed_product_records": malformed_product_records,
        "unknown_product_vendor_records": int(
            sum(unknown_product_vendors.values())
        ),
        "unknown_product_vendors": dict(sorted(unknown_product_vendors.items())),
    }
    return profiles, diagnostics


def collect_product_page_samples_from_path(
    path: Path,
    *,
    vendor_id_to_index: dict[str, int],
    capability_limit: int,
    pain_limit: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    """Load JSONL and hash only the product records used by this pipeline."""
    product_record_digest = hashlib.sha256()

    def records() -> Iterator[dict[str, Any]]:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected an object in {path}:{line_number}"
                    )
                if record.get("page_type") == "product":
                    canonical = json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    product_record_digest.update(canonical.encode("utf-8"))
                    product_record_digest.update(b"\n")
                yield record

    profiles, diagnostics = collect_product_page_samples(
        records(),
        vendor_id_to_index=vendor_id_to_index,
        capability_limit=capability_limit,
        pain_limit=pain_limit,
    )
    return profiles, diagnostics, product_record_digest.hexdigest()


def _sample_manifest(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Remove raw text while retaining exact sample counts and hashes."""
    manifest: dict[str, dict[str, Any]] = {}
    for vendor_id, profile in profiles.items():
        capability_samples = [
            {"sha256": sample["sha256"], "count": sample["count"]}
            for sample in profile["capabilities"]
        ]
        pain_samples = [
            {"sha256": sample["sha256"], "count": sample["count"]}
            for sample in profile["pains"]
        ]
        values = {
            "vendor_index": profile["vendor_index"],
            "product_record_count": profile["product_record_count"],
            "source_urls": profile["source_urls"],
            "capability_total_mentions": profile["capability_total_mentions"],
            "capability_unique_count": profile["capability_unique_count"],
            "capability_retained_count": len(capability_samples),
            "capability_samples": capability_samples,
            "pain_total_mentions": profile["pain_total_mentions"],
            "pain_unique_count": profile["pain_unique_count"],
            "pain_retained_count": len(pain_samples),
            "pain_samples": pain_samples,
        }
        values["profile_sha256"] = _sha256_json(values)
        manifest[vendor_id] = values
    return manifest


def _load_feature_mappings(path: Path) -> dict[str, int]:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    else:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("feature mappings must contain an object")

    mapping = payload.get("vendor_id_to_idx")
    if mapping is None:
        mapping = payload.get("vendors")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "feature mappings lack vendor_id_to_idx (or vendors)"
        )
    normalized = {str(key): int(value) for key, value in mapping.items()}
    indices = sorted(normalized.values())
    if indices != list(range(len(indices))):
        raise ValueError("vendor feature mapping indices are not contiguous")
    return normalized


def _load_and_validate_hetero_vendor_mapping(
    path: Path,
    *,
    vendor_id_to_index: dict[str, int],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    node_maps = checkpoint.get("node_maps")
    if not isinstance(node_maps, dict):
        raise ValueError("hetero_data cache lacks node_maps")
    hetero_vendors = node_maps.get("vendor")
    if not isinstance(hetero_vendors, dict):
        raise ValueError("hetero_data node_maps lacks vendors")
    normalized = {str(key): int(value) for key, value in hetero_vendors.items()}
    if normalized != vendor_id_to_index:
        raise ValueError(
            "feature mappings and hetero_data vendor mappings disagree"
        )
    summary = {
        "n_vendors": len(normalized),
        "vendor_mapping_sha256": _sha256_json(normalized),
        "edges_read_for_feature_construction": False,
    }
    del checkpoint
    gc.collect()
    return summary


class _LocalBGEEncoder:
    """Minimal GPU BGE encoder that never consults a remote model source."""

    def __init__(self, snapshot: Path, *, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(snapshot),
            local_files_only=True,
        ).to(device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del show_progress_bar
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                output = self.model(**encoded)
                embeddings = output.last_hidden_state[:, 0]
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
            batches.append(embeddings.cpu().to(torch.float32).numpy())
        values = (
            np.vstack(batches)
            if batches
            else np.empty((0, self.dimension), dtype=np.float32)
        )
        if not convert_to_numpy:
            raise ValueError("local BGE encoder requires convert_to_numpy=True")
        return values


def _resolve_local_model(
    *,
    model_name: str,
    model_cache: Path,
    gpu: int,
) -> tuple[Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for product-page embedding")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {gpu} is unavailable; detected {torch.cuda.device_count()} devices"
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    import transformers

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(model_cache),
                local_files_only=True,
            )
        ).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} is not available in local cache {model_cache}; "
            "network downloads are disabled"
        ) from exc

    device = f"cuda:{gpu}"
    model = _LocalBGEEncoder(snapshot, device=device)
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}d {model_name} embeddings, "
            f"received {dimension}d"
        )
    return model, {
        "name": model_name,
        "dimension": dimension,
        "device": device,
        "local_files_only": True,
        "cache": str(model_cache.resolve()),
        "snapshot": str(snapshot),
        "snapshot_revision": snapshot.name,
        "encoder_implementation": "transformers_cls_pooling",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }


def _text_uses(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[int, str, int]]]:
    uses: dict[str, list[tuple[int, str, int]]] = {}
    for profile in profiles.values():
        vendor_index = int(profile["vendor_index"])
        for kind, key in (("capability", "capabilities"), ("pain", "pains")):
            for sample in profile[key]:
                uses.setdefault(sample["text"], []).append(
                    (vendor_index, kind, int(sample["count"]))
                )
    return uses


def embed_and_aggregate(
    *,
    profiles: dict[str, dict[str, Any]],
    n_vendors: int,
    model: Any,
    batch_size: int,
    dimension: int = EMBEDDING_DIMENSION,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Embed globally unique text once and build normalized vendor means."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    uses = _text_uses(profiles)
    texts = sorted(uses)
    capability_sum = np.zeros((n_vendors, dimension), dtype=np.float32)
    pain_sum = np.zeros((n_vendors, dimension), dtype=np.float32)
    capability_samples = np.zeros(n_vendors, dtype=np.int64)
    pain_samples = np.zeros(n_vendors, dtype=np.int64)
    capability_mentions = 0
    pain_mentions = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        encoded = np.asarray(encoded, dtype=np.float32)
        if encoded.shape != (len(batch), dimension):
            raise ValueError(
                f"Unexpected embedding shape {encoded.shape}; "
                f"expected {(len(batch), dimension)}"
            )
        if not np.isfinite(encoded).all():
            raise ValueError("embedding model returned non-finite values")

        for text, embedding in zip(batch, encoded):
            for vendor_index, kind, count in uses[text]:
                if kind == "capability":
                    capability_sum[vendor_index] += embedding
                    capability_samples[vendor_index] += 1
                    capability_mentions += count
                else:
                    pain_sum[vendor_index] += embedding
                    pain_samples[vendor_index] += 1
                    pain_mentions += count
        completed = min(start + len(batch), len(texts))
        if completed == len(texts) or start == 0 or start // batch_size % 10 == 0:
            print(f"Embedded {completed:,}/{len(texts):,} unique texts", flush=True)

    capability = F.normalize(
        torch.from_numpy(capability_sum), p=2, dim=1, eps=1e-8
    )
    pain = F.normalize(torch.from_numpy(pain_sum), p=2, dim=1, eps=1e-8)
    diagnostics = {
        "global_unique_texts_embedded": len(texts),
        "capability_unique_texts_retained": len(
            {
                sample["text"]
                for profile in profiles.values()
                for sample in profile["capabilities"]
            }
        ),
        "pain_unique_texts_retained": len(
            {
                sample["text"]
                for profile in profiles.values()
                for sample in profile["pains"]
            }
        ),
        "capability_pain_text_overlap": len(
            {
                sample["text"]
                for profile in profiles.values()
                for sample in profile["capabilities"]
            }
            & {
                sample["text"]
                for profile in profiles.values()
                for sample in profile["pains"]
            }
        ),
        "capability_retained_vendor_samples": int(capability_samples.sum()),
        "pain_retained_vendor_samples": int(pain_samples.sum()),
        "capability_mentions_represented": int(capability_mentions),
        "pain_mentions_represented": int(pain_mentions),
        "global_text_hash_sha256": _sha256_json(
            [_sha256_text(text) for text in texts]
        ),
    }
    return capability, pain, diagnostics


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(
        values.to(dtype=torch.float32, device="cpu"),
        p=2,
        dim=1,
        eps=1e-8,
    )


def materialize_feature_variants(
    *,
    product: torch.Tensor,
    capability: torch.Tensor,
    pain: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Create fixed equal-block-norm product-page vendor ablations."""
    if product.ndim != 2 or capability.ndim != 2 or pain.ndim != 2:
        raise ValueError("vendor feature blocks must be matrices")
    if len({product.shape[0], capability.shape[0], pain.shape[0]}) != 1:
        raise ValueError("vendor feature blocks disagree on row count")
    if (
        product.shape[1] != EMBEDDING_DIMENSION
        or capability.shape[1] != EMBEDDING_DIMENSION
        or pain.shape[1] != EMBEDDING_DIMENSION
    ):
        raise ValueError("all vendor feature blocks must be 1024-dimensional")

    product_norm = _normalize_rows(product)
    capability_norm = _normalize_rows(capability)
    pain_norm = _normalize_rows(pain)
    return {
        "product_page_capability_only": capability_norm,
        "product_page_pain_only": pain_norm,
        "product_page_product_capability": torch.cat(
            [product_norm, capability_norm], dim=1
        )
        / math.sqrt(2.0),
        "product_page_product_pain": torch.cat(
            [product_norm, pain_norm], dim=1
        )
        / math.sqrt(2.0),
        "product_page_full": torch.cat(
            [product_norm, capability_norm, pain_norm], dim=1
        )
        / math.sqrt(3.0),
    }


def _index_set_hash(indices: list[int]) -> str:
    return _sha256_json(sorted(map(int, indices)))


def _coverage_summary(
    *,
    indices: Iterable[int],
    profiles_by_index: dict[int, dict[str, Any]],
    product: torch.Tensor,
) -> dict[str, Any]:
    selected = sorted(set(map(int, indices)))
    product_nonzero = torch.linalg.vector_norm(product, dim=1) > 0
    counts = Counter()
    for index in selected:
        profile = profiles_by_index[index]
        has_record = profile["product_record_count"] > 0
        has_capability = bool(profile["capabilities"])
        has_pain = bool(profile["pains"])
        has_product = bool(product_nonzero[index].item())
        counts["product_page_record"] += int(has_record)
        counts["capability"] += int(has_capability)
        counts["pain"] += int(has_pain)
        counts["capability_or_pain"] += int(has_capability or has_pain)
        counts["capability_and_pain"] += int(has_capability and has_pain)
        counts["product_vector"] += int(has_product)
        counts["all_three_blocks"] += int(
            has_product and has_capability and has_pain
        )

    denominator = len(selected)
    return {
        "vendors": denominator,
        "vendor_indices_sha256": _index_set_hash(selected),
        **{
            name: {
                "count": int(counts[name]),
                "fraction": (
                    round(counts[name] / denominator, 6)
                    if denominator
                    else None
                ),
            }
            for name in (
                "product_page_record",
                "capability",
                "pain",
                "capability_or_pain",
                "capability_and_pain",
                "product_vector",
                "all_three_blocks",
            )
        },
    }


def build_coverage_diagnostics(
    *,
    profiles: dict[str, dict[str, Any]],
    product: torch.Tensor,
    split: dict[str, Any],
    edge_transfer: dict[str, Any],
) -> dict[str, Any]:
    profiles_by_index = {
        int(profile["vendor_index"]): profile
        for profile in profiles.values()
    }
    n_vendors = product.shape[0]
    holdout_vendors = list(map(int, split.get("holdout_vendors", [])))
    double_holdout_vendors = sorted(
        {
            int(vendor)
            for vendor, _ in (
                split.get("test_slices", {}).get("double_holdout", [])
            )
        }
    )
    edge_transfer_test_vendors = sorted(
        {
            int(vendor)
            for vendor, _ in edge_transfer.get("test_edges", [])
        }
    )
    return {
        "all_graph_vendors": _coverage_summary(
            indices=range(n_vendors),
            profiles_by_index=profiles_by_index,
            product=product,
        ),
        "holdout_vendors": _coverage_summary(
            indices=holdout_vendors,
            profiles_by_index=profiles_by_index,
            product=product,
        ),
        "double_holdout_vendors": _coverage_summary(
            indices=double_holdout_vendors,
            profiles_by_index=profiles_by_index,
            product=product,
        ),
        "edge_transfer_test_vendors": _coverage_summary(
            indices=edge_transfer_test_vendors,
            profiles_by_index=profiles_by_index,
            product=product,
        ),
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def prepare_cache(
    *,
    extractions_path: Path,
    source_cache_path: Path,
    hetero_data_path: Path,
    feature_mappings_path: Path,
    output_path: Path,
    model_cache: Path,
    gpu: int,
    batch_size: int,
    capability_limit: int,
    pain_limit: int,
) -> dict[str, Any]:
    started = time.time()
    vendor_id_to_index = _load_feature_mappings(feature_mappings_path)
    mapping_validation = _load_and_validate_hetero_vendor_mapping(
        hetero_data_path,
        vendor_id_to_index=vendor_id_to_index,
    )

    print(f"Reading product-page samples from {extractions_path}...")
    profiles, extraction_diagnostics, product_records_sha256 = (
        collect_product_page_samples_from_path(
            extractions_path,
            vendor_id_to_index=vendor_id_to_index,
            capability_limit=capability_limit,
            pain_limit=pain_limit,
        )
    )
    sample_manifest = _sample_manifest(profiles)
    sample_manifest_sha256 = _sha256_json(sample_manifest)

    model, model_provenance = _resolve_local_model(
        model_name=EMBEDDING_MODEL,
        model_cache=model_cache,
        gpu=gpu,
    )
    capability, pain, embedding_diagnostics = embed_and_aggregate(
        profiles=profiles,
        n_vendors=len(vendor_id_to_index),
        model=model,
        batch_size=batch_size,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading source cache {source_cache_path}...")
    source = torch.load(
        source_cache_path, map_location="cpu", weights_only=False
    )
    product = source.get("vendor_features", {}).get("product_only")
    if not isinstance(product, torch.Tensor):
        raise ValueError("source cache lacks vendor_features.product_only")
    if product.shape != (len(vendor_id_to_index), EMBEDDING_DIMENSION):
        raise ValueError(
            "source product_only features do not match the vendor mapping "
            f"or expected dimension: {tuple(product.shape)}"
        )
    variants = materialize_feature_variants(
        product=product,
        capability=capability,
        pain=pain,
    )
    coverage = build_coverage_diagnostics(
        profiles=profiles,
        product=product,
        split=source["split"],
        edge_transfer=source["edge_transfer"],
    )

    code_sha256 = _sha256_file(Path(__file__))
    input_hashes = {
        "product_page_records_sha256": product_records_sha256,
        "source_cache_sha256": _sha256_file(source_cache_path),
        "hetero_data_sha256": _sha256_file(hetero_data_path),
        "feature_mappings_sha256": _sha256_file(feature_mappings_path),
        "code_sha256": code_sha256,
        "sample_manifest_sha256": sample_manifest_sha256,
    }
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "model_snapshot_revision": model_provenance["snapshot_revision"],
        "text_normalization": TEXT_NORMALIZATION,
        "capability_limit_per_vendor": capability_limit,
        "pain_limit_per_vendor": pain_limit,
        "embedding_batch_size": batch_size,
        "aggregation": (
            "unweighted mean of retained unique unit text embeddings, then L2 "
            "normalize; exact frequencies are used for ranking and retained "
            "in the sample manifest"
        ),
        "feature_block_scaling": {
            "single": "unit row norm",
            "two_blocks": "unit-normalized blocks concatenated / sqrt(2)",
            "three_blocks": "unit-normalized blocks concatenated / sqrt(3)",
        },
    }
    feature_hashes = {
        name: _sha256_tensor(values)
        for name, values in variants.items()
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
        "Vendor features named product_page_* are built only from product-page "
        "extractions plus the source product_only block; case-study records and "
        "vendor-customer edges are excluded from vendor feature construction. "
        "Customer features retain the source cache's graph-derived evidence."
    )
    product_page_diagnostics = {
        "extractions": extraction_diagnostics,
        "embedding": embedding_diagnostics,
        "mapping_validation": mapping_validation,
        "coverage": coverage,
        "feature_dimensions": {
            name: int(values.shape[1])
            for name, values in variants.items()
        },
        "feature_sha256": feature_hashes,
        "sample_manifest_sha256": sample_manifest_sha256,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    artifact["diagnostics"] = dict(source.get("diagnostics", {}))
    artifact["diagnostics"]["product_page_vendor_features"] = (
        product_page_diagnostics
    )
    artifact["independent_product_page"] = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "fingerprint_payload": fingerprint_payload,
        "sample_manifest": sample_manifest,
        "provenance": {
            "extractions": str(extractions_path.resolve()),
            "source_cache": str(source_cache_path.resolve()),
            "hetero_data": str(hetero_data_path.resolve()),
            "feature_mappings": str(feature_mappings_path.resolve()),
            "code": str(Path(__file__).resolve()),
            "embedding_model": model_provenance,
        },
        "construction_contract": {
            "page_type_filter": "product",
            "uses_case_study_records_for_vendor_features": False,
            "uses_vendor_customer_edges_for_vendor_features": False,
            "hetero_data_usage": "vendor node mapping validation only",
            "source_cache_vendor_block_reused": "product_only",
            "source_cache_other_vendor_blocks_modified": False,
            "customer_features_preserved": True,
            "split_preserved": True,
            "pairs_preserved": True,
            "labels_used_for": "coverage diagnostics only",
            "global_text_deduplication_before_embedding": True,
            "exact_text_normalization": TEXT_NORMALIZATION,
        },
    }

    _atomic_torch_save(output_path, artifact)
    summary = {
        "output": str(output_path),
        "fingerprint": fingerprint,
        "source_fingerprint": source.get("fingerprint"),
        "features": artifact["diagnostics"][
            "product_page_vendor_features"
        ]["feature_dimensions"],
        "coverage": coverage,
        "unique_texts": embedding_diagnostics[
            "global_unique_texts_embedded"
        ],
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
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument(
        "--model-cache", type=Path, default=DEFAULT_MODEL_CACHE
    )
    prepare.add_argument("--gpu", type=int, required=True)
    prepare.add_argument("--batch-size", type=int, default=1024)
    prepare.add_argument("--capability-limit", type=int, default=64)
    prepare.add_argument("--pain-limit", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_cache(
            extractions_path=args.extractions,
            source_cache_path=args.source_cache,
            hetero_data_path=args.hetero_data,
            feature_mappings_path=args.feature_mappings,
            output_path=args.output,
            model_cache=args.model_cache,
            gpu=args.gpu,
            batch_size=args.batch_size,
            capability_limit=args.capability_limit,
            pain_limit=args.pain_limit,
        )


if __name__ == "__main__":
    main()
