#!/usr/bin/env python3
"""Vendor-blind independent-profile experiment for the fixed Adobe CRM cohort.

The command sequence is:

1. ``research``: preserve every input lead, deduplicate public research by exact
   normalized account name, and build vendor-blind external company profiles.
2. ``resolve``: retrieve BGE top-five canonical pain candidates and let local
   Qwen select only an exact supplied canonical ID or null.
3. ``score``: construct independent customer feature rows and score Adobe with
   the ten-seed audience and semantic inductive checkpoint ensembles.
4. ``evaluate``: compare structural scores with the fixed-sample pipeline score
   without fitting or tuning any parameters on this cohort.

This module intentionally does not import or modify production ``lead_signals``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.graph_embeddings.independent_profile_experiment import (
    DEFAULT_GRAPH_NODES,
    DEFAULT_HETERO_DATA,
    DEFAULT_INDEX_MAPS,
    DEFAULT_NODE_MAPPINGS,
    DEFAULT_PAIN_FEATURES,
    EMBEDDING_MODEL,
    LocalBGEEncoder,
    _append_jsonl,
    _atomic_write_json,
    _load_latest_jsonl,
    _load_node_catalog,
    _load_node_mappings,
    _research_customer,
    _resolve_customer,
    _sha256_file,
    _training_only_idf,
    build_independent_feature_row,
    cosine_shortlists,
    customer_name_rejection_reason,
    validate_checkpoint_configuration,
)
from experiments.graph_embeddings.inductive_baseline import InductiveTwoTower
from paper_submission.experiment_io import require_local


GRAPH_DIR = ROOT / "experiments" / "graph_embeddings"
DEFAULT_SOURCE_CACHE = GRAPH_DIR / "inductive_features_independent_base.pt"
DEFAULT_CHECKPOINT_DIR = (
    GRAPH_DIR / "results" / "independent_profile_training"
)

INPUT_LEADS_FILENAME = "input_leads.jsonl"
RESEARCH_FILENAME = "research_profiles.jsonl"
RESOLVED_FILENAME = "resolved_profiles.jsonl"
ACCOUNT_SCORES_FILENAME = "account_scores.jsonl"
SCORES_FILENAME = "scores.jsonl"
RESEARCH_MANIFEST = "research_manifest.json"
RESOLVE_MANIFEST = "resolve_manifest.json"
SCORE_MANIFEST = "score_manifest.json"

PIPELINE_VERSION = "crm-independent-profile-v1"
RESEARCH_SCHEMA_VERSION = "crm-independent-research-v1"
RESOLVE_SCHEMA_VERSION = "crm-independent-resolution-v1"
SCORE_SCHEMA_VERSION = "crm-independent-score-v1"
EVALUATION_SCHEMA_VERSION = "crm-independent-evaluation-v1"

TARGET_VENDOR_ID = "vendor:adobe"
TARGET_VENDOR_SLUG = "adobe"
STRUCTURAL_METHODS = ("inductive_audience", "inductive_semantic")
CHECKPOINT_CUSTOMER_FEATURES = {
    "inductive_audience": "audience",
    "inductive_semantic": "semantic",
}
P_AT_K = (1, 3, 5, 10, 20)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_account_name(value: Any) -> str:
    """Return the only account matching key used by this experiment.

    NFKC, whitespace collapsing, and case folding are normalization rather than
    entity resolution. Punctuation, suffixes, spelling, and tokens are retained;
    no fuzzy or alias matching is performed.
    """
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def _display_account_name(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _external_identity(account_key: str) -> tuple[str, int]:
    digest = hashlib.sha256(account_key.encode("utf-8")).hexdigest()
    return f"external:{digest[:24]}", int(digest[:15], 16)


def _strict_label(record: dict[str, Any]) -> bool | None:
    value = record.get("crm_qualified")
    return value if isinstance(value, bool) else None


def _json_fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            records.append(record)
    return records


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_file_metadata(path: Path, count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "record_count": count,
    }


def _cohort_fingerprint(source_files: Sequence[dict[str, Any]]) -> str:
    return _json_fingerprint(
        [
            {
                "sha256": source["sha256"],
                "size_bytes": source["size_bytes"],
                "record_count": source["record_count"],
            }
            for source in source_files
        ]
    )


def load_lead_cohort(
    lead_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load all lead records in concatenation order and attach exact account keys."""
    if not lead_paths:
        raise ValueError("At least one --leads path is required")

    leads: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in lead_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Lead JSONL not found: {path}")
        source_records = _read_jsonl(path)
        sources.append(_source_file_metadata(path, len(source_records)))
        for source_line, source_record in enumerate(source_records, 1):
            record = copy.deepcopy(source_record)
            account_name = _display_account_name(record.get("account_name"))
            account_key = normalize_account_name(account_name)
            external_id, external_index = _external_identity(account_key)
            label = _strict_label(record)
            record.update(
                {
                    "lead_index": len(leads),
                    "source_path": str(path.resolve()),
                    "source_line": source_line,
                    "account_name": account_name,
                    "account_normalized": account_key,
                    "external_account_id": external_id,
                    "external_account_index": external_index,
                    "evaluation_label": (
                        "qualified"
                        if label is True
                        else "genuine_nq"
                        if label is False
                        else None
                    ),
                }
            )
            leads.append(record)
    return leads, sources, _cohort_fingerprint(sources)


def build_external_accounts(
    leads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate research identities by exact normalized account name."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lead in leads:
        grouped[normalize_account_name(lead.get("account_name"))].append(lead)

    accounts: list[dict[str, Any]] = []
    seen_indices: dict[int, str] = {}
    for account_key in sorted(grouped):
        records = grouped[account_key]
        names = sorted(
            {_display_account_name(record.get("account_name")) for record in records},
            key=lambda value: (value.casefold(), value),
        )
        account_name = names[0] if names else ""
        external_id, external_index = _external_identity(account_key)
        collision = seen_indices.get(external_index)
        if collision is not None and collision != account_key:
            raise ValueError(
                "Stable external index collision between "
                f"{collision!r} and {account_key!r}"
            )
        seen_indices[external_index] = account_key
        labels = [_strict_label(record) for record in records]
        products = sorted(
            {
                str(record.get("product") or "all")
                for record in records
            }
        )
        accounts.append(
            {
                "customer_index": external_index,
                "customer_id": external_id,
                "customer_name": account_name,
                "account_normalized": account_key,
                "validation_error": customer_name_rejection_reason(account_name),
                "lead_count": len(records),
                "products": products,
                "qualified_leads": sum(label is True for label in labels),
                "genuine_nq_leads": sum(label is False for label in labels),
                "unlabeled_leads": sum(label is None for label in labels),
                "account_qualified": any(label is True for label in labels),
            }
        )
    return accounts


def _duplicate_product_summary(
    accounts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    duplicates = [
        {
            "account_normalized": account["account_normalized"],
            "account_name": account["customer_name"],
            "lead_count": account["lead_count"],
            "products": account["products"],
        }
        for account in accounts
        if account["lead_count"] > 1 or len(account["products"]) > 1
    ]
    return {
        "accounts_with_duplicate_records": sum(
            account["lead_count"] > 1 for account in accounts
        ),
        "accounts_with_multiple_products": sum(
            len(account["products"]) > 1 for account in accounts
        ),
        "extra_lead_records_after_account_dedupe": sum(
            max(0, int(account["lead_count"]) - 1) for account in accounts
        ),
        "accounts": duplicates,
    }


def _latest_records_by_account(path: Path) -> dict[str, dict[str, Any]]:
    return {
        normalize_account_name(record.get("account_normalized"))
        or normalize_account_name(record.get("customer_name")): record
        for record in _load_latest_jsonl(path).values()
    }


def _coverage_by_label(
    accounts: Sequence[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    buckets: dict[str, dict[str, int | float]] = {}
    for label_name, selector in (
        ("qualified", lambda account: bool(account["account_qualified"])),
        ("genuine_nq", lambda account: not bool(account["account_qualified"])),
        ("overall", lambda account: True),
    ):
        selected = [account for account in accounts if selector(account)]
        statuses = Counter(
            str(
                latest.get(account["account_normalized"], {}).get(
                    "status", "missing"
                )
            )
            for account in selected
        )
        successful = statuses.get("success", 0)
        buckets[label_name] = {
            "accounts": len(selected),
            "success": successful,
            "error": statuses.get("error", 0),
            "rejected_invalid_name": statuses.get(
                "rejected_invalid_name", 0
            ),
            "missing": statuses.get("missing", 0),
            "coverage": (
                round(successful / len(selected), 6) if selected else None
            ),
            "statuses": dict(sorted(statuses.items())),
        }
    return buckets


def research_external_account(
    account: dict[str, Any],
    *,
    audience_catalog: Sequence[dict[str, Any]],
    use_cache: bool,
    research_fn: Callable[..., dict[str, Any]] = _research_customer,
    **research_kwargs: Any,
) -> dict[str, Any]:
    """Call the shared researcher with a deliberately sanitized identity only."""
    safe_customer = {
        "customer_index": int(account["customer_index"]),
        "customer_id": str(account["customer_id"]),
        "customer_name": str(account["customer_name"]),
        "account_normalized": str(account["account_normalized"]),
    }
    result = research_fn(
        safe_customer,
        audience_catalog=audience_catalog,
        use_cache=use_cache,
        **research_kwargs,
    )
    result["account_normalized"] = safe_customer["account_normalized"]
    return result


def run_research(
    *,
    lead_paths: Sequence[Path],
    output_dir: Path,
    workers: int = 16,
    use_cache: bool = True,
    node_mappings_path: Path = DEFAULT_NODE_MAPPINGS,
    graph_nodes_path: Path = DEFAULT_GRAPH_NODES,
    research_fn: Callable[..., dict[str, Any]] = _research_customer,
    require_local_fn: Callable[[], Any] = require_local,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    require_local_fn()

    leads, sources, cohort_fingerprint = load_lead_cohort(lead_paths)
    accounts = build_external_accounts(leads)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / INPUT_LEADS_FILENAME, leads)

    node_mappings = _load_node_mappings(node_mappings_path)
    audience_nodes = _load_node_catalog(
        node_types={"audience"},
        path=graph_nodes_path,
    )
    audience_catalog = [
        audience_nodes[audience_id]
        for audience_id, _ in sorted(
            node_mappings["audiences"].items(), key=lambda item: item[1]
        )
    ]

    research_path = output_dir / RESEARCH_FILENAME
    existing = _latest_records_by_account(research_path)
    pending: list[dict[str, Any]] = []
    rejected_this_run = 0
    for account in accounts:
        previous = existing.get(account["account_normalized"], {})
        if account["validation_error"] is not None:
            if (
                previous.get("status") != "rejected_invalid_name"
                or previous.get("validation_error") != account["validation_error"]
            ):
                _append_jsonl(
                    research_path,
                    {
                        "customer_index": account["customer_index"],
                        "customer_id": account["customer_id"],
                        "customer_name": account["customer_name"],
                        "account_normalized": account["account_normalized"],
                        "stage": "research",
                        "status": "rejected_invalid_name",
                        "pipeline_version": PIPELINE_VERSION,
                        "schema_version": RESEARCH_SCHEMA_VERSION,
                        "created_at": _utc_now(),
                        "validation_error": account["validation_error"],
                        "profile": None,
                        "errors": [
                            "Explicit placeholder or generic invalid account name"
                        ],
                    },
                )
                rejected_this_run += 1
            continue
        reusable = (
            use_cache
            and previous.get("status") == "success"
            and previous.get("customer_id") == account["customer_id"]
            and normalize_account_name(previous.get("account_normalized"))
            == account["account_normalized"]
        )
        if not reusable:
            pending.append(account)

    succeeded = 0
    failed = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                research_external_account,
                account,
                audience_catalog=audience_catalog,
                use_cache=use_cache,
                research_fn=research_fn,
            ): account
            for account in pending
        }
        for future in as_completed(futures):
            account = futures[future]
            try:
                result = future.result()
                result["pipeline_version"] = PIPELINE_VERSION
                result["schema_version"] = RESEARCH_SCHEMA_VERSION
            except Exception as exc:  # noqa: BLE001 - persisted for resume
                result = {
                    "customer_index": account["customer_index"],
                    "customer_id": account["customer_id"],
                    "customer_name": account["customer_name"],
                    "account_normalized": account["account_normalized"],
                    "stage": "research",
                    "status": "error",
                    "pipeline_version": PIPELINE_VERSION,
                    "schema_version": RESEARCH_SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "profile": None,
                    "errors": [repr(exc)],
                }
            _append_jsonl(research_path, result)
            completed += 1
            succeeded += int(result.get("status") == "success")
            failed += int(result.get("status") != "success")
            print(
                f"[research {completed}/{len(pending)}] "
                f"{account['customer_name']}: {result.get('status')}",
                flush=True,
            )

    latest = _latest_records_by_account(research_path)
    label_counts = Counter(
        lead.get("evaluation_label") or "unlabeled" for lead in leads
    )
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "output": str(research_path),
        "input_leads": str(output_dir / INPUT_LEADS_FILENAME),
        "source_files": sources,
        "cohort_fingerprint": cohort_fingerprint,
        "input_counts": {
            "lead_records": len(leads),
            "unique_accounts": len(accounts),
            "qualified_leads": label_counts["qualified"],
            "genuine_nq_leads": label_counts["genuine_nq"],
            "unlabeled_leads": label_counts["unlabeled"],
            "qualified_accounts": sum(
                bool(account["account_qualified"]) for account in accounts
            ),
            "genuine_nq_accounts": sum(
                not bool(account["account_qualified"]) for account in accounts
            ),
        },
        "duplicate_products": _duplicate_product_summary(accounts),
        "workers": workers,
        "use_search_cache": use_cache,
        "already_successful": sum(
            existing.get(account["account_normalized"], {}).get("status")
            == "success"
            for account in accounts
        ),
        "attempted_this_run": len(pending),
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "rejected_this_run": rejected_this_run,
        "coverage_by_label": _coverage_by_label(accounts, latest),
        "identity_contract": {
            "research_identity": "stable external SHA-256 ID/index",
            "dedupe_key": (
                "NFKC + whitespace collapse + casefold; exact equality only"
            ),
            "graph_customer_identity_required": False,
            "target_vendor_or_product_in_research": False,
        },
    }
    _atomic_write_json(output_dir / RESEARCH_MANIFEST, summary)
    print(json.dumps(summary, indent=2))
    return summary


def _record_profile_fingerprint(record: dict[str, Any]) -> str:
    return _json_fingerprint(
        {
            "customer_id": record.get("customer_id"),
            "account_normalized": normalize_account_name(
                record.get("account_normalized")
            ),
            "profile": record.get("profile"),
        }
    )


def run_resolve(
    *,
    output_dir: Path,
    workers: int,
    device: str,
    node_mappings_path: Path = DEFAULT_NODE_MAPPINGS,
    pain_features_path: Path = DEFAULT_PAIN_FEATURES,
    graph_nodes_path: Path = DEFAULT_GRAPH_NODES,
    encoder_factory: Callable[..., Any] = LocalBGEEncoder,
    resolve_fn: Callable[..., dict[str, Any]] = _resolve_customer,
    require_local_fn: Callable[[], Any] = require_local,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    require_local_fn()

    research_manifest_path = output_dir / RESEARCH_MANIFEST
    if not research_manifest_path.exists():
        raise ValueError(f"Research manifest not found: {research_manifest_path}")
    research_manifest = json.loads(research_manifest_path.read_text())
    research_path = output_dir / RESEARCH_FILENAME
    resolved_path = output_dir / RESOLVED_FILENAME
    research_records = {
        key: record
        for key, record in _latest_records_by_account(research_path).items()
        if record.get("status") == "success"
    }
    existing = _latest_records_by_account(resolved_path)
    pending = [
        record
        for key, record in sorted(research_records.items())
        if not (
            existing.get(key, {}).get("status") == "success"
            and existing[key].get("customer_id") == record.get("customer_id")
            and existing[key].get("research_profile_fingerprint")
            == _record_profile_fingerprint(record)
        )
    ]

    if pending:
        node_mappings = _load_node_mappings(node_mappings_path)
        pain_id_to_index = {
            pain_id: int(index)
            for pain_id, index in node_mappings["pains"].items()
        }
        pain_ids = [
            pain_id
            for pain_id, _ in sorted(
                pain_id_to_index.items(), key=lambda item: item[1]
            )
        ]
        pain_catalog = _load_node_catalog(
            node_types={"pain_canonical"},
            path=graph_nodes_path,
        )
        if set(pain_ids) != set(pain_catalog):
            raise ValueError("pain node mapping and v6 pain catalog differ")
        canonical_embeddings = np.asarray(
            np.load(pain_features_path), dtype=np.float32
        )
        if canonical_embeddings.shape[0] != len(pain_ids):
            raise ValueError("pain feature rows do not match node mapping")

        flattened: list[tuple[int, int, str]] = []
        for record in pending:
            for pain_index, pain in enumerate(
                record["profile"]["contextual_pains"]
            ):
                flattened.append(
                    (int(record["customer_index"]), pain_index, str(pain))
                )
        encoder = encoder_factory(device=device)
        query_embeddings = encoder.encode(
            [pain for _, _, pain in flattened]
        )
        shortlist_indices, shortlist_scores = cosine_shortlists(
            query_embeddings,
            canonical_embeddings,
            top_k=5,
        )
        candidates_by_index = {
            int(record["customer_index"]): [
                {"contextual_pain": pain, "candidates": []}
                for pain in record["profile"]["contextual_pains"]
            ]
            for record in pending
        }
        for row, (external_index, pain_index, _) in enumerate(flattened):
            candidates = []
            for canonical_index, similarity in zip(
                shortlist_indices[row], shortlist_scores[row]
            ):
                canonical_id = pain_ids[int(canonical_index)]
                node = pain_catalog[canonical_id]
                candidates.append(
                    {
                        "canonical_id": canonical_id,
                        "canonical_name": node.get("canonical_name") or "",
                        "scope_definition": node.get("scope_definition") or "",
                        "cosine_similarity": round(float(similarity), 6),
                    }
                )
            candidates_by_index[external_index][pain_index][
                "candidates"
            ] = candidates
    else:
        candidates_by_index = {}

    completed = 0
    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                resolve_fn,
                record,
                candidates_by_index[int(record["customer_index"])],
            ): record
            for record in pending
        }
        for future in as_completed(futures):
            research_record = futures[future]
            try:
                resolved = future.result()
                resolved["pipeline_version"] = PIPELINE_VERSION
                resolved["schema_version"] = RESOLVE_SCHEMA_VERSION
            except Exception as exc:  # noqa: BLE001 - persisted for resume
                resolved = {
                    "customer_index": research_record["customer_index"],
                    "customer_id": research_record["customer_id"],
                    "customer_name": research_record["customer_name"],
                    "stage": "resolve",
                    "status": "error",
                    "pipeline_version": PIPELINE_VERSION,
                    "schema_version": RESOLVE_SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "profile": copy.deepcopy(research_record.get("profile")),
                    "resolved_pains": None,
                    "matched_canonical_ids": [],
                    "unmatched_pains": [],
                    "errors": [repr(exc)],
                }
            resolved["account_normalized"] = normalize_account_name(
                research_record.get("account_normalized")
            )
            resolved["research_profile_fingerprint"] = (
                _record_profile_fingerprint(research_record)
            )
            _append_jsonl(resolved_path, resolved)
            completed += 1
            succeeded += int(resolved.get("status") == "success")
            failed += int(resolved.get("status") != "success")
            print(
                f"[resolve {completed}/{len(pending)}] "
                f"{research_record['customer_name']}: {resolved.get('status')}",
                flush=True,
            )

    leads = _read_jsonl(output_dir / INPUT_LEADS_FILENAME)
    accounts = build_external_accounts(leads)
    latest = _latest_records_by_account(research_path)
    latest.update(_latest_records_by_account(resolved_path))
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": RESOLVE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "output": str(resolved_path),
        "cohort_fingerprint": research_manifest.get("cohort_fingerprint"),
        "embedding_model": EMBEDDING_MODEL,
        "device": device,
        "workers": workers,
        "successful_research_profiles": len(research_records),
        "already_successful": len(research_records) - len(pending),
        "attempted_this_run": len(pending),
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "coverage_by_label": _coverage_by_label(accounts, latest),
        "resolution_contract": {
            "candidate_retrieval": "BGE cosine top-5",
            "canonical_selection": "Qwen exact supplied ID or null",
            "invented_ids_allowed": False,
            "graph_customer_identity_required": False,
        },
    }
    _atomic_write_json(output_dir / RESOLVE_MANIFEST, summary)
    print(json.dumps(summary, indent=2))
    return summary


def _load_index_maps(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _validate_source_cohort(
    output_dir: Path,
    cohort_fingerprint: str,
) -> dict[str, Any]:
    manifest_path = output_dir / RESEARCH_MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"Research manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("cohort_fingerprint")
    if expected != cohort_fingerprint:
        raise ValueError(
            "Lead source fingerprint mismatch: research used "
            f"{expected!r}, scoring received {cohort_fingerprint!r}"
        )
    return manifest


def discover_scoring_checkpoint_runs(
    checkpoint_dir: Path,
    *,
    expected_seeds_per_method: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Discover only the two requested unpruned checkpoint ensembles."""
    runs_by_method: dict[str, list[dict[str, Any]]] = {}
    for method, customer_feature in CHECKPOINT_CUSTOMER_FEATURES.items():
        paths = sorted(
            checkpoint_dir.glob(
                "oracle_node_holdout__unpruned__"
                f"{customer_feature}__seed*.pt"
            )
        )
        if len(paths) != expected_seeds_per_method:
            raise ValueError(
                f"Expected {expected_seeds_per_method} {method} checkpoints in "
                f"{checkpoint_dir}, found {len(paths)}"
            )
        runs = []
        seeds: set[int] = set()
        for checkpoint_path in paths:
            result_path = checkpoint_path.with_suffix(".json")
            if not result_path.exists():
                raise ValueError(
                    f"Checkpoint has no training result JSON: {checkpoint_path}"
                )
            training_result = json.loads(result_path.read_text())
            run_name = checkpoint_path.stem
            if training_result.get("run_name") != run_name:
                raise ValueError(
                    f"Run name mismatch between {checkpoint_path} and {result_path}"
                )
            configuration = training_result.get("configuration")
            if not isinstance(configuration, dict):
                raise ValueError(f"Missing configuration in {result_path}")
            expected = {
                "vendor_feature": "unpruned",
                "customer_feature": customer_feature,
                "evaluation_protocol": "oracle_node_holdout",
            }
            for field, expected_value in expected.items():
                if configuration.get(field) != expected_value:
                    raise ValueError(
                        f"{checkpoint_path.name} has {field}="
                        f"{configuration.get(field)!r}, expected {expected_value!r}"
                    )
            seed = int(configuration["seed"])
            if seed in seeds:
                raise ValueError(f"Duplicate {method} checkpoint seed: {seed}")
            seeds.add(seed)
            runs.append(
                {
                    "method": method,
                    "run_name": run_name,
                    "checkpoint_path": checkpoint_path,
                    "result_path": result_path,
                    "training_result": training_result,
                    "configuration": configuration,
                }
            )
        runs_by_method[method] = sorted(
            runs, key=lambda run: int(run["configuration"]["seed"])
        )
    audience_seeds = {
        int(run["configuration"]["seed"])
        for run in runs_by_method["inductive_audience"]
    }
    semantic_seeds = {
        int(run["configuration"]["seed"])
        for run in runs_by_method["inductive_semantic"]
    }
    if audience_seeds != semantic_seeds:
        raise ValueError(
            "Audience and semantic checkpoint seed sets differ: "
            f"{sorted(audience_seeds)} vs {sorted(semantic_seeds)}"
        )
    return runs_by_method


def validate_scoring_checkpoint(
    *,
    run: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    source_cache: dict[str, Any],
) -> dict[str, int]:
    fingerprint = source_cache.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Source cache has no fingerprint")
    return validate_checkpoint_configuration(
        run=run,
        state_dict=state_dict,
        cache=source_cache,
        source_cache_fingerprint=fingerprint,
    )


def _feature_context(
    *,
    source_cache: dict[str, Any],
    hetero_data_path: Path,
    node_mappings_path: Path,
    pain_features_path: Path,
) -> dict[str, Any]:
    hetero_checkpoint = torch.load(
        hetero_data_path, map_location="cpu", weights_only=False
    )
    data = hetero_checkpoint["data"]
    node_mappings = _load_node_mappings(node_mappings_path)
    pain_id_to_index = {
        pain_id: int(index)
        for pain_id, index in node_mappings["pains"].items()
    }
    audience_id_to_index = {
        audience_id: int(index)
        for audience_id, index in node_mappings["audiences"].items()
    }
    pain_features = np.asarray(np.load(pain_features_path), dtype=np.float32)
    if pain_features.shape[0] != len(pain_id_to_index):
        raise ValueError("pain feature rows do not match node mappings")

    pain_to_subtheme = np.full(pain_features.shape[0], -1, dtype=np.int64)
    subtheme_edges = data[
        "pain_canonical", "in_subtheme", "pain_subtheme"
    ].edge_index.cpu()
    pain_to_subtheme[subtheme_edges[0].numpy()] = subtheme_edges[1].numpy()
    if np.any(pain_to_subtheme < 0):
        raise ValueError("one or more canonical pains lack a subtheme mapping")
    tfidf_idf, semantic_idf = _training_only_idf(
        data["customer"].x,
        source_cache["split"]["train_customers"],
    )
    return {
        "pain_id_to_index": pain_id_to_index,
        "pain_to_subtheme": pain_to_subtheme,
        "pain_features": pain_features,
        "audience_id_to_index": audience_id_to_index,
        "tfidf_subtheme_idf": tfidf_idf,
        "semantic_subtheme_idf": semantic_idf,
    }


def join_leads_with_account_scores(
    leads: Sequence[dict[str, Any]],
    account_scores: dict[str, dict[str, Any]],
    *,
    source_cache_fingerprint: str,
) -> list[dict[str, Any]]:
    """Join by the exact normalized account key and preserve every lead pair."""
    joined: list[dict[str, Any]] = []
    for lead in leads:
        account_key = normalize_account_name(lead.get("account_name"))
        account_score = account_scores.get(account_key)
        success = account_score is not None and account_score.get("status") == "success"
        record = copy.deepcopy(lead)
        record.update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "schema_version": SCORE_SCHEMA_VERSION,
                "account_normalized": account_key,
                "status": "success" if success else "unavailable",
                "unavailable_reason": (
                    None
                    if success
                    else (
                        account_score or {}
                    ).get("unavailable_reason", "no_resolved_external_profile")
                ),
                "score_scope": "adobe_vendor_wide_not_product_specific",
                "target_vendor_id": TARGET_VENDOR_ID,
                "product_specific_structural_score": False,
                "source_cache_fingerprint": source_cache_fingerprint,
            }
        )
        for method in STRUCTURAL_METHODS:
            record[method] = (
                account_score.get(method) if success else None
            )
            record[f"{method}_per_seed"] = (
                copy.deepcopy(account_score.get(f"{method}_per_seed", {}))
                if success
                else {}
            )
        joined.append(record)
    return joined


def run_score(
    *,
    output_dir: Path,
    lead_paths: Sequence[Path],
    source_cache_path: Path = DEFAULT_SOURCE_CACHE,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device_name: str = "cuda:4",
    hetero_data_path: Path = DEFAULT_HETERO_DATA,
    node_mappings_path: Path = DEFAULT_NODE_MAPPINGS,
    pain_features_path: Path = DEFAULT_PAIN_FEATURES,
    index_maps_path: Path = DEFAULT_INDEX_MAPS,
) -> dict[str, Any]:
    leads, sources, cohort_fingerprint = load_lead_cohort(lead_paths)
    _validate_source_cohort(output_dir, cohort_fingerprint)
    non_adobe = sorted(
        {
            normalize_account_name(lead.get("vendor_slug"))
            for lead in leads
            if normalize_account_name(lead.get("vendor_slug"))
            not in {"", TARGET_VENDOR_SLUG}
        }
    )
    if non_adobe:
        raise ValueError(f"Non-Adobe vendor slugs in fixed cohort: {non_adobe}")

    source_cache = torch.load(
        source_cache_path, map_location="cpu", weights_only=False
    )
    source_fingerprint = source_cache.get("fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("Source cache has no fingerprint")
    feature_context = _feature_context(
        source_cache=source_cache,
        hetero_data_path=hetero_data_path,
        node_mappings_path=node_mappings_path,
        pain_features_path=pain_features_path,
    )
    resolved = _latest_records_by_account(output_dir / RESOLVED_FILENAME)
    research = _latest_records_by_account(output_dir / RESEARCH_FILENAME)
    accounts = build_external_accounts(leads)

    feature_rows: dict[str, dict[str, torch.Tensor]] = {}
    unavailable: dict[str, str] = {}
    for account in accounts:
        key = account["account_normalized"]
        record = resolved.get(key)
        if record is None:
            research_status = research.get(key, {}).get("status")
            unavailable[key] = (
                f"research_{research_status}"
                if research_status and research_status != "success"
                else "no_resolution_record"
            )
            continue
        if record.get("status") != "success":
            unavailable[key] = "resolution_not_successful"
            continue
        if record.get("customer_id") != account["customer_id"]:
            unavailable[key] = "external_identity_mismatch"
            continue
        try:
            feature_rows[key] = build_independent_feature_row(
                resolved_record=record,
                **feature_context,
            )
        except Exception as exc:  # noqa: BLE001 - account-level unavailability
            unavailable[key] = f"feature_construction_error: {exc}"

    runs_by_method = discover_scoring_checkpoint_runs(checkpoint_dir)
    index_maps = _load_index_maps(index_maps_path)
    vendor_index = index_maps["vendor_id_to_idx"].get(TARGET_VENDOR_ID)
    if vendor_index is None:
        raise ValueError(f"Vendor mapping lacks {TARGET_VENDOR_ID}")
    vendor_index = int(vendor_index)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device_name}")
    account_keys = sorted(feature_rows)
    per_method: dict[str, dict[str, dict[int, float]]] = {
        method: {key: {} for key in account_keys}
        for method in STRUCTURAL_METHODS
    }
    checkpoint_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for method in STRUCTURAL_METHODS:
        customer_feature = CHECKPOINT_CUSTOMER_FEATURES[method]
        if account_keys:
            customer_values = torch.stack(
                [feature_rows[key][customer_feature] for key in account_keys]
            ).to(torch.float32)
        else:
            customer_values = torch.empty(
                (0, source_cache["customer_features"][customer_feature].shape[1]),
                dtype=torch.float32,
            )
        vendor_value = source_cache["vendor_features"]["unpruned"][
            vendor_index : vendor_index + 1
        ].to(torch.float32)
        for run in runs_by_method[method]:
            state_dict = torch.load(
                run["checkpoint_path"],
                map_location=device,
                weights_only=True,
            )
            dimensions = validate_scoring_checkpoint(
                run=run,
                state_dict=state_dict,
                source_cache=source_cache,
            )
            configuration = run["configuration"]
            model = InductiveTwoTower(
                vendor_dim=dimensions["vendor_input_dim"],
                customer_dim=dimensions["customer_input_dim"],
                hidden_dim=dimensions["hidden_dim"],
                output_dim=dimensions["output_dim"],
                dropout=float(configuration["dropout"]),
            ).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            with torch.no_grad():
                vendor_embedding = model.vendor(vendor_value.to(device))
                if account_keys:
                    customer_embedding = model.customer(
                        customer_values.to(device)
                    )
                    scores = (
                        customer_embedding @ vendor_embedding.T
                    ).squeeze(1).cpu().numpy()
                else:
                    scores = np.empty((0,), dtype=np.float32)
            seed = int(configuration["seed"])
            for key, score in zip(account_keys, scores):
                per_method[method][key][seed] = float(score)
            checkpoint_metadata[method].append(
                {
                    "seed": seed,
                    "checkpoint": str(run["checkpoint_path"]),
                    "checkpoint_sha256": _sha256_file(run["checkpoint_path"]),
                    "training_result": str(run["result_path"]),
                    "training_cache_fingerprint": run["training_result"].get(
                        "cache_fingerprint"
                    ),
                }
            )
            del model, state_dict
            if device.type == "cuda":
                torch.cuda.empty_cache()

    account_scores: dict[str, dict[str, Any]] = {}
    for account in accounts:
        key = account["account_normalized"]
        if key not in feature_rows:
            account_scores[key] = {
                "customer_index": account["customer_index"],
                "customer_id": account["customer_id"],
                "customer_name": account["customer_name"],
                "account_normalized": key,
                "status": "unavailable",
                "unavailable_reason": unavailable.get(
                    key, "no_resolved_external_profile"
                ),
                "score_scope": "adobe_vendor_wide_not_product_specific",
                "inductive_audience": None,
                "inductive_semantic": None,
                "inductive_audience_per_seed": {},
                "inductive_semantic_per_seed": {},
            }
            continue
        record: dict[str, Any] = {
            "customer_index": account["customer_index"],
            "customer_id": account["customer_id"],
            "customer_name": account["customer_name"],
            "account_normalized": key,
            "status": "success",
            "unavailable_reason": None,
            "score_scope": "adobe_vendor_wide_not_product_specific",
        }
        for method in STRUCTURAL_METHODS:
            seed_scores = per_method[method][key]
            if len(seed_scores) != len(runs_by_method[method]):
                raise ValueError(
                    f"Incomplete {method} seed scores for {account['customer_name']}"
                )
            record[method] = float(np.mean(list(seed_scores.values())))
            record[f"{method}_per_seed"] = {
                str(seed): seed_scores[seed] for seed in sorted(seed_scores)
            }
        account_scores[key] = record

    account_score_records = [
        {
            **account_scores[key],
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCORE_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "target_vendor_id": TARGET_VENDOR_ID,
            "source_cache_fingerprint": source_fingerprint,
        }
        for key in sorted(account_scores)
    ]
    pair_records = join_leads_with_account_scores(
        leads,
        account_scores,
        source_cache_fingerprint=source_fingerprint,
    )
    account_scores_path = output_dir / ACCOUNT_SCORES_FILENAME
    scores_path = output_dir / SCORES_FILENAME
    _atomic_write_jsonl(account_scores_path, account_score_records)
    _atomic_write_jsonl(scores_path, pair_records)

    label_coverage: dict[str, Any] = {}
    for label_name, predicate in (
        ("qualified", lambda lead: _strict_label(lead) is True),
        ("genuine_nq", lambda lead: _strict_label(lead) is False),
        ("overall", lambda lead: _strict_label(lead) is not None),
    ):
        selected = [lead for lead in pair_records if predicate(lead)]
        successful = sum(lead["status"] == "success" for lead in selected)
        label_coverage[label_name] = {
            "lead_records": len(selected),
            "successful": successful,
            "unavailable": len(selected) - successful,
            "coverage": (
                round(successful / len(selected), 6) if selected else None
            ),
        }

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCORE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "scores": str(scores_path),
        "account_scores": str(account_scores_path),
        "source_files": sources,
        "cohort_fingerprint": cohort_fingerprint,
        "source_cache": str(source_cache_path),
        "source_cache_sha256": _sha256_file(source_cache_path),
        "source_cache_fingerprint": source_fingerprint,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": dict(checkpoint_metadata),
        "device": str(device),
        "target_vendor_id": TARGET_VENDOR_ID,
        "score_scope": "vendor-wide; products retained but not scored separately",
        "lead_records": len(pair_records),
        "unique_accounts": len(account_scores),
        "successful_accounts": sum(
            record["status"] == "success" for record in account_scores.values()
        ),
        "unavailable_accounts": sum(
            record["status"] != "success" for record in account_scores.values()
        ),
        "coverage_by_label": label_coverage,
    }
    _atomic_write_json(output_dir / SCORE_MANIFEST, summary)
    print(json.dumps(summary, indent=2))
    return summary


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def aggregate_account_records(
    pair_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate duplicate account-products without duplicating structural scores."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_records:
        label = _strict_label(record)
        if label is None:
            continue
        grouped[normalize_account_name(record.get("account_name"))].append(record)

    accounts: list[dict[str, Any]] = []
    for account_key in sorted(grouped):
        records = grouped[account_key]
        structural: dict[str, float | None] = {}
        for method in STRUCTURAL_METHODS:
            values = [
                float(record[method])
                for record in records
                if _finite_number(record.get(method))
            ]
            unique = {round(value, 12) for value in values}
            if len(unique) > 1:
                raise ValueError(
                    f"Vendor-wide {method} differs across products for {account_key}"
                )
            structural[method] = values[0] if values else None
        pipeline_values = [
            float(record["pipeline_composite_score"])
            for record in records
            if _finite_number(record.get("pipeline_composite_score"))
        ]
        names = sorted(
            {_display_account_name(record.get("account_name")) for record in records},
            key=lambda value: (value.casefold(), value),
        )
        accounts.append(
            {
                "account_name": names[0],
                "account_normalized": account_key,
                "crm_qualified": any(
                    _strict_label(record) is True for record in records
                ),
                "evaluation_label": (
                    "qualified"
                    if any(_strict_label(record) is True for record in records)
                    else "genuine_nq"
                ),
                "pipeline_composite_score": (
                    max(pipeline_values) if pipeline_values else None
                ),
                **structural,
                "products": sorted(
                    {str(record.get("product") or "all") for record in records}
                ),
                "pair_count": len(records),
                "structural_score_scope": (
                    "adobe_vendor_wide_not_product_specific"
                ),
            }
        )
    return accounts


def percentile_ranks(
    records: Sequence[dict[str, Any]],
    score_key: str,
) -> dict[str, float]:
    """Return deterministic average-rank percentiles keyed by stable record ID."""
    def record_key(record: dict[str, Any]) -> str:
        account_key = normalize_account_name(record.get("account_name"))
        if "lead_index" in record:
            return f"{account_key}\0{int(record['lead_index'])}"
        return account_key

    values = [
        (
            record_key(record),
            float(record[score_key]),
        )
        for record in records
        if _finite_number(record.get(score_key))
    ]
    values.sort(key=lambda item: (item[1], item[0]))
    if not values:
        return {}
    if len(values) == 1:
        return {values[0][0]: 0.5}
    result: dict[str, float] = {}
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][1] == values[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        percentile = average_rank / (len(values) - 1)
        for index in range(start, end):
            result[values[index][0]] = percentile
        start = end
    return result


def add_percentile_controls(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add label-free fallback and fixed 0.5 percentile controls."""
    def record_key(record: dict[str, Any]) -> str:
        account_key = normalize_account_name(record.get("account_name"))
        if "lead_index" in record:
            return f"{account_key}\0{int(record['lead_index'])}"
        return account_key

    enriched = [copy.deepcopy(record) for record in records]
    pipeline_percentiles = percentile_ranks(
        enriched, "pipeline_composite_score"
    )
    for method in STRUCTURAL_METHODS:
        structural_percentiles = percentile_ranks(enriched, method)
        for record in enriched:
            key = record_key(record)
            structural = structural_percentiles[key]
            pipeline = pipeline_percentiles[key]
            record[f"{method}_percentile"] = structural
            record["pipeline_percentile"] = pipeline
            record[f"{method}_pipeline_zero_fallback"] = (
                pipeline
                if float(record["pipeline_composite_score"]) > 0
                else structural
            )
            record[f"{method}_pipeline_blend_0_5"] = (
                0.5 * structural + 0.5 * pipeline
            )
    return enriched


def _score_distribution(
    records: Sequence[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(record[score_key])
            for record in records
            if _finite_number(record.get(score_key))
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def ranking_metrics(
    records: Sequence[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    """Compute deterministic binary ranking metrics with account-name tie breaks."""
    scored = [
        record for record in records if _finite_number(record.get(score_key))
    ]
    ranked = sorted(
        scored,
        key=lambda record: (
            -float(record[score_key]),
            normalize_account_name(record.get("account_name")),
            int(record.get("lead_index", -1)),
        ),
    )
    labels = np.asarray(
        [int(bool(record["crm_qualified"])) for record in ranked],
        dtype=np.float64,
    )
    positives = int(labels.sum())
    precision_at_k: dict[str, float | None] = {}
    for k in P_AT_K:
        precision_at_k[str(k)] = (
            float(labels[:k].sum() / k) if len(ranked) >= k else None
        )
    if positives:
        cumulative = np.cumsum(labels)
        ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
        average_precision = float(
            ((cumulative / ranks) * labels).sum() / positives
        )
        discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
        dcg = float((labels * discounts).sum())
        ideal = float(discounts[:positives].sum())
        ndcg = dcg / ideal if ideal > 0 else None
    else:
        average_precision = None
        ndcg = None
    by_label = {
        label: _score_distribution(
            [
                record
                for record in ranked
                if (
                    bool(record["crm_qualified"])
                    if label == "qualified"
                    else not bool(record["crm_qualified"])
                )
            ],
            score_key,
        )
        for label in ("qualified", "genuine_nq")
    }
    return {
        "n": len(ranked),
        "qualified": positives,
        "genuine_nq": len(ranked) - positives,
        "base_rate": positives / len(ranked) if ranked else None,
        "precision_at_k": precision_at_k,
        "average_precision": average_precision,
        "ndcg": ndcg,
        "score_distribution": {
            "overall": _score_distribution(ranked, score_key),
            **by_label,
        },
    }


def _evaluate_methods(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    methods = [
        "inductive_audience",
        "inductive_semantic",
        "pipeline_composite_score",
        "inductive_audience_pipeline_zero_fallback",
        "inductive_semantic_pipeline_zero_fallback",
        "inductive_audience_pipeline_blend_0_5",
        "inductive_semantic_pipeline_blend_0_5",
    ]
    return {
        method: ranking_metrics(records, method)
        for method in methods
    }


def _evaluation_coverage(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        audience = sum(
            _finite_number(record.get("inductive_audience"))
            for record in selected
        )
        semantic = sum(
            _finite_number(record.get("inductive_semantic"))
            for record in selected
        )
        common = sum(
            _finite_number(record.get("inductive_audience"))
            and _finite_number(record.get("inductive_semantic"))
            and _finite_number(record.get("pipeline_composite_score"))
            for record in selected
        )
        return {
            "records": len(selected),
            "inductive_audience_scored": audience,
            "inductive_semantic_scored": semantic,
            "common_scored_with_pipeline": common,
            "inductive_audience_coverage": (
                audience / len(selected) if selected else None
            ),
            "inductive_semantic_coverage": (
                semantic / len(selected) if selected else None
            ),
            "common_coverage": common / len(selected) if selected else None,
        }

    labeled = [
        record for record in records if _strict_label(record) is not None
    ]
    return {
        "overall": summarize(labeled),
        "qualified": summarize(
            [record for record in labeled if _strict_label(record) is True]
        ),
        "genuine_nq": summarize(
            [record for record in labeled if _strict_label(record) is False]
        ),
    }


def run_evaluate(
    *,
    scores_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    pair_records = _read_jsonl(scores_path)
    labeled_pairs = [
        record for record in pair_records if _strict_label(record) is not None
    ]
    account_records = aggregate_account_records(labeled_pairs)
    common_accounts = [
        record
        for record in account_records
        if all(
            _finite_number(record.get(method))
            for method in STRUCTURAL_METHODS
        )
        and _finite_number(record.get("pipeline_composite_score"))
    ]
    common_pairs = [
        record
        for record in labeled_pairs
        if all(
            _finite_number(record.get(method))
            for method in STRUCTURAL_METHODS
        )
        and _finite_number(record.get("pipeline_composite_score"))
    ]
    account_controls = add_percentile_controls(common_accounts)
    pair_controls = add_percentile_controls(common_pairs)

    def slices(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "all_common_scored": _evaluate_methods(records),
            "pipeline_score_zero": _evaluate_methods(
                [
                    record
                    for record in records
                    if float(record["pipeline_composite_score"]) == 0.0
                ]
            ),
            "pipeline_score_positive": _evaluate_methods(
                [
                    record
                    for record in records
                    if float(record["pipeline_composite_score"]) > 0.0
                ]
            ),
        }

    duplicate_accounts = [
        record for record in account_records if int(record["pair_count"]) > 1
    ]
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "scores": str(scores_path),
        "evaluation_contract": {
            "labels": "strict boolean crm_qualified: qualified vs genuine fixed-sample NQ",
            "account_join": (
                "NFKC + whitespace collapse + casefold; exact equality only"
            ),
            "account_aggregation": {
                "structural": "one vendor-wide value per account",
                "pipeline": "maximum across duplicated account-products",
                "label": "qualified when any duplicated pair is qualified",
            },
            "common_subset": (
                "Both structural methods and pipeline score must be finite"
            ),
            "ranking_tie_break": "exact normalized account name, then lead index",
            "percentile_controls_are_label_free": True,
            "zero_fallback": (
                "pipeline score > 0 uses pipeline percentile; pipeline score == 0 "
                "uses the corresponding structural percentile"
            ),
            "fixed_blend": "0.5 structural percentile + 0.5 pipeline percentile",
            "parameter_tuning_on_cohort": False,
        },
        "population": {
            "input_pairs": len(pair_records),
            "labeled_pairs": len(labeled_pairs),
            "labeled_accounts": len(account_records),
            "common_scored_pairs": len(common_pairs),
            "common_scored_accounts": len(common_accounts),
            "duplicate_account_products": {
                "accounts": len(duplicate_accounts),
                "extra_pairs": sum(
                    int(record["pair_count"]) - 1
                    for record in duplicate_accounts
                ),
            },
        },
        "coverage": {
            "account_level": _evaluation_coverage(account_records),
            "pair_level_descriptive": _evaluation_coverage(labeled_pairs),
        },
        "account_level": slices(account_controls),
        "pair_level_descriptive": slices(pair_controls),
    }
    _atomic_write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "labeled_accounts": len(account_records),
                "common_scored_accounts": len(common_accounts),
                "account_coverage": result["coverage"]["account_level"],
            },
            indent=2,
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    research = subparsers.add_parser("research")
    research.add_argument("--leads", type=Path, nargs="+", required=True)
    research.add_argument("--output-dir", type=Path, required=True)
    research.add_argument("--workers", type=int, default=16)
    research.add_argument(
        "--fresh-search",
        action="store_true",
        help="Bypass cached SERP responses and redo successful account research.",
    )

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--output-dir", type=Path, required=True)
    resolve.add_argument("--workers", type=int, default=8)
    resolve.add_argument("--device", default="cuda:4")

    score = subparsers.add_parser("score")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--leads", type=Path, nargs="+", required=True)
    score.add_argument(
        "--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE
    )
    score.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR
    )
    score.add_argument("--device", default="cuda:4")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "research":
        run_research(
            lead_paths=args.leads,
            output_dir=args.output_dir,
            workers=args.workers,
            use_cache=not args.fresh_search,
        )
    elif args.command == "resolve":
        run_resolve(
            output_dir=args.output_dir,
            workers=args.workers,
            device=args.device,
        )
    elif args.command == "score":
        run_score(
            output_dir=args.output_dir,
            lead_paths=args.leads,
            source_cache_path=args.source_cache,
            checkpoint_dir=args.checkpoint_dir,
            device_name=args.device,
        )
    else:
        run_evaluate(
            scores_path=args.scores,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
