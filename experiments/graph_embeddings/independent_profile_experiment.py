#!/usr/bin/env python3
"""Build independent cold-start customer profiles for the inductive GNN.

The pipeline deliberately keeps target vendors out of customer research:

1. ``research`` gathers generic public company context and asks local Qwen for
   graph audience IDs plus contextual pain statements.
2. ``resolve`` retrieves canonical pain candidates with local BGE embeddings,
   then asks local Qwen to select one candidate or null for every pain.
3. ``build-cache`` replaces only successfully profiled holdout customer rows
   and emits a customer-holdout-only derivative of ``inductive_features.pt``.
4. ``evaluate`` applies completed oracle-node-holdout checkpoints to that
   independently profiled customer cohort and aggregates ranking metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from paper_submission.experiment_io import local_tool_call, require_local
from paper_submission.experiment_io import SearchResult, serp
from experiments.graph_embeddings.inductive_baseline import (
    InductiveTwoTower,
    SliceDefinition,
    evaluate_slice,
)


GRAPH_DIR = ROOT / "experiments" / "graph_embeddings"
FEATURE_DIR = GRAPH_DIR / "features"
DEFAULT_CACHE = GRAPH_DIR / "inductive_features.pt"
DEFAULT_HETERO_DATA = GRAPH_DIR / "hetero_data.pt"
DEFAULT_INDEX_MAPS = FEATURE_DIR / "index_maps.pkl"
DEFAULT_NODE_MAPPINGS = FEATURE_DIR / "node_mappings.json"
DEFAULT_PAIN_FEATURES = FEATURE_DIR / "pain_features.npy"
DEFAULT_GRAPH_NODES = ROOT / "graph" / "v6" / "nodes.jsonl"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

RESEARCH_FILENAME = "research_profiles.jsonl"
RESOLVED_FILENAME = "resolved_profiles.jsonl"
RESEARCH_MANIFEST = "research_manifest.json"
RESOLVE_MANIFEST = "resolve_manifest.json"

PIPELINE_VERSION = "independent-customer-profile-v1"
RESEARCH_SCHEMA_VERSION = "independent-customer-research-v1"
RESOLVE_SCHEMA_VERSION = "independent-canonical-resolution-v1"
FEATURE_SCHEMA_VERSION = "independent-customer-features-v1"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

_CASE_STUDY_TEXT = re.compile(
    r"\b(?:case[\s_-]*stud(?:y|ies)|customer[\s_-]*stor(?:y|ies)|"
    r"success[\s_-]*stor(?:y|ies)|client[\s_-]*stor(?:y|ies))\b",
    re.IGNORECASE,
)
_CASE_STUDY_URL = re.compile(
    r"/(?:case-stud(?:y|ies)|customer-stor(?:y|ies)|success-stor(?:y|ies)|"
    r"client-stor(?:y|ies)|customers?)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_UNUSABLE_NAMES = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "unknown",
    "anonymous",
    "confidential",
    "undisclosed",
    "unnamed",
    "customer",
    "client",
}
_GENERIC_NAME_PATTERNS = (
    re.compile(
        r"^(?:a|an)\s+(?:global|large|leading|major|prominent|fast-growing|"
        r"top-ranked|north american|private equity|business entity|company|"
        r"manufacturer|provider|supplier|contractor|operator|retailer|bank|"
        r"insurer|reinsurer|facility|plant|firm|organization)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:leading|major|large|global|prominent)\s+(?:manufacturer|provider|"
        r"supplier|contractor|operator|retailer|reinsurer|insurer|company|"
        r"organization|facility|dealership|business process|commercial truck|"
        r"north american|european retailer)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:this\s+industry|one\s+of\s+the)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:unnamed|anonymous|confidential|undisclosed|industry[- ]leading|"
        r"industry leader)\b",
        re.IGNORECASE,
    ),
)
_GENERIC_COMPANY_TOKENS = {
    "a",
    "an",
    "and",
    "business",
    "bus",
    "commercial",
    "companies",
    "company",
    "contracting",
    "contractor",
    "corp",
    "corporation",
    "dealership",
    "defense",
    "design",
    "development",
    "enterprise",
    "enterprises",
    "entity",
    "facility",
    "firm",
    "for",
    "global",
    "group",
    "gmbh",
    "in",
    "inc",
    "industry",
    "insurer",
    "international",
    "large",
    "leader",
    "leading",
    "limited",
    "llc",
    "llp",
    "lp",
    "major",
    "manufacturer",
    "manufacturing",
    "of",
    "operator",
    "organisation",
    "organization",
    "plant",
    "plc",
    "power",
    "producer",
    "prominent",
    "provider",
    "reinsurer",
    "retailer",
    "sa",
    "service",
    "services",
    "solutions",
    "supplier",
    "systems",
    "technology",
    "the",
    "this",
    "to",
    "transportation",
    "truck",
    "aerospace",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_value(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _usage_record(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    values: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            values[field] = int(value)
    return values or None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_latest_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    file_size = path.stat().st_size
    with open(path, encoding="utf-8") as handle:
        line_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if handle.tell() == file_size and not line.endswith("\n"):
                    print(
                        f"Warning: ignoring truncated final JSONL record in "
                        f"{path}:{line_number}",
                        file=sys.stderr,
                    )
                    break
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if "customer_index" not in record:
                raise ValueError(f"Missing customer_index in {path}:{line_number}")
            records[int(record["customer_index"])] = record
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_index_maps(path: Path = DEFAULT_INDEX_MAPS) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_node_mappings(path: Path = DEFAULT_NODE_MAPPINGS) -> dict[str, dict[str, int]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_node_catalog(
    *,
    node_types: set[str],
    path: Path = DEFAULT_GRAPH_NODES,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            node = json.loads(line)
            if node.get("type") in node_types:
                catalog[node["id"]] = node
    return catalog


def sid(value: str) -> str:
    """Slugify exactly as the v4/v6 graph builders do."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60]


def customer_name_rejection_reason(name: str | None) -> str | None:
    value = " ".join((name or "").split())
    if not value:
        return "missing_canonical_name"
    if value.casefold() in _UNUSABLE_NAMES or len(value) < 2:
        return "placeholder_name"
    if re.search(r"[A-Za-z]", value) is None:
        return "no_ascii_letters"
    if any(pattern.search(value) for pattern in _GENERIC_NAME_PATTERNS):
        return "generic_company_description"

    tokens = re.findall(r"[A-Za-z0-9]+", value.casefold())
    if len(tokens) > 10:
        return "sentence_like_name"
    distinctive_tokens = [
        token
        for token in tokens
        if any(character.isalpha() for character in token)
        and token not in _GENERIC_COMPANY_TOKENS
    ]
    if not distinctive_tokens:
        return "generic_company_name"
    return None


def is_usable_customer_name(name: str | None) -> bool:
    return customer_name_rejection_reason(name) is None


def customer_identity_rejection_reason(
    customer_id: str | None,
    node: dict[str, Any] | None,
) -> str | None:
    if not customer_id or not customer_id.startswith("customer:"):
        return "invalid_customer_id"
    if node is None:
        return "missing_v6_customer_node"
    if node.get("type") != "customer":
        return "invalid_v6_node_type"
    canonical_name = " ".join(str(node.get("name") or "").split())
    if not canonical_name:
        return "missing_canonical_name"
    if re.search(r"[A-Za-z]", canonical_name) is None:
        return "no_ascii_letters"
    expected_slug = customer_id.removeprefix("customer:")
    if sid(canonical_name) != expected_slug:
        return "canonical_name_slug_mismatch"
    return customer_name_rejection_reason(canonical_name)


def profile_record_identity_rejection_reason(
    record: dict[str, Any],
    customer_nodes: dict[str, dict[str, Any]],
) -> str | None:
    customer_id = record.get("customer_id")
    node = customer_nodes.get(str(customer_id or ""))
    reason = customer_identity_rejection_reason(customer_id, node)
    if reason is not None:
        return reason
    assert node is not None
    canonical_name = " ".join(str(node["name"]).split())
    record_name = " ".join(str(record.get("customer_name") or "").split())
    if record_name != canonical_name:
        return "record_name_differs_from_canonical_node"
    return None


def select_research_cohort(
    *,
    split: dict[str, Any],
    customer_idx_to_id: dict[int, str],
    customer_nodes: dict[str, dict[str, Any]],
    max_customers: int | None,
    seed: int,
    selection: str = "random",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a stable subset of eligible customer-holdout nodes."""
    if selection not in {"random", "high_support"}:
        raise ValueError(f"unknown cohort selection strategy: {selection}")
    positive_customers = {
        int(customer)
        for _, customer in split["test_slices"].get("customer_holdout", [])
    }
    holdout_customers = set(map(int, split["holdout_customers"]))
    identity_candidates = sorted(holdout_customers & positive_customers)
    eligible: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for customer_index in identity_candidates:
        customer_id = customer_idx_to_id.get(customer_index)
        node = customer_nodes.get(customer_id or "")
        reason = customer_identity_rejection_reason(customer_id, node)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        assert customer_id is not None and node is not None
        canonical_name = " ".join(str(node["name"]).split())
        canonical_slug = customer_id.removeprefix("customer:")
        eligible.append(
            {
                "customer_index": customer_index,
                "customer_id": customer_id,
                "customer_name": canonical_name,
                "graph_support": {
                    "n_case_studies": int(node.get("n_case_studies") or 0),
                    "n_vendors": len(node.get("vendors") or []),
                },
                "identity_validation": {
                    "name_source": "graph/v6/nodes.jsonl",
                    "canonical_name_sid": sid(canonical_name),
                    "customer_id_slug": canonical_slug,
                    "exact_slug_match": True,
                },
            }
        )
    accepted_before_limit = len(eligible)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if selection == "high_support":
        eligible.sort(
            key=lambda record: (
                -record["graph_support"]["n_vendors"],
                -record["graph_support"]["n_case_studies"],
            )
        )
    if max_customers is not None and max_customers >= 0:
        eligible = eligible[:max_customers]
    not_positive = len(holdout_customers - positive_customers)
    rejected_by_reason = {
        "no_customer_holdout_positive": not_positive,
        **dict(sorted(rejected.items())),
    }
    diagnostics = {
        "holdout_customers": len(holdout_customers),
        "customer_holdout_positive_customers": len(positive_customers),
        "identity_check_candidates": len(identity_candidates),
        "accepted_identity_safe_before_limit": accepted_before_limit,
        "selected_after_limit": len(eligible),
        "selection_strategy": selection,
        "rejected_total_before_limit": sum(rejected_by_reason.values()),
        "rejected_identity_total": sum(rejected.values()),
        "rejected_by_reason": rejected_by_reason,
        "rejected_identity_by_reason": dict(sorted(rejected.items())),
        "identity_contract": (
            "Use graph/v6 customer node name only when sid(name) exactly equals "
            "the customer ID slug; require ASCII letters and a non-generic "
            "company-like canonical name."
        ),
    }
    return eligible, diagnostics


def case_study_filter_reason(result: SearchResult | dict[str, Any]) -> str | None:
    url = str(_record_value(result, "url") or "")
    title = str(_record_value(result, "title") or "")
    snippet = str(_record_value(result, "snippet") or "")
    if _CASE_STUDY_URL.search(url):
        return "case_study_url"
    if _CASE_STUDY_TEXT.search(f"{title}\n{snippet}"):
        return "case_study_text"
    return None


def is_case_study_result(result: SearchResult | dict[str, Any]) -> bool:
    return case_study_filter_reason(result) is not None


def _serialize_search_result(result: SearchResult | dict[str, Any]) -> dict[str, Any]:
    return {
        "url": str(_record_value(result, "url") or ""),
        "title": str(_record_value(result, "title") or ""),
        "snippet": str(_record_value(result, "snippet") or ""),
        "published_date": _record_value(result, "published_date"),
    }


def _filter_search_results(
    results: Sequence[SearchResult | dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        serialized = _serialize_search_result(result)
        key = (serialized["url"], serialized["title"])
        if key in seen:
            continue
        seen.add(key)
        reason = case_study_filter_reason(result)
        if reason:
            excluded.append({**serialized, "filter_reason": reason})
        elif serialized["title"] or serialized["snippet"]:
            kept.append(serialized)
    return kept, excluded


def _generic_company_query(company_name: str, *, quoted: bool) -> str:
    subject = f'"{company_name}"' if quoted else company_name
    return (
        f"{subject} company overview industry revenue employees business model "
        "operational strategic challenges"
    )


def _research_schema(audience_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "audience_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(audience_ids)},
            },
            "contextual_pains": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "description",
            "audience_ids",
            "contextual_pains",
            "confidence",
        ],
        "additionalProperties": False,
    }


def validate_research_output(
    output: dict[str, Any] | None,
    *,
    valid_audience_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Qwen returned no structured profile")
    description = " ".join(str(output.get("description") or "").split())
    if not description:
        raise ValueError("profile description is empty")

    audience_ids = output.get("audience_ids")
    if isinstance(audience_ids, str):
        audience_ids = re.findall(r"audience:[a-z0-9:_-]+", audience_ids)
    if not isinstance(audience_ids, list):
        raise ValueError("audience_ids must be a list")
    normalized_audiences = list(dict.fromkeys(str(value) for value in audience_ids))
    invalid_audiences = [
        audience_id
        for audience_id in normalized_audiences
        if audience_id not in valid_audience_ids
    ]
    normalized_audiences = [
        audience_id
        for audience_id in normalized_audiences
        if audience_id in valid_audience_ids
    ]
    if not normalized_audiences:
        raise ValueError("profile selected no graph audience IDs")

    pains = output.get("contextual_pains")
    if not isinstance(pains, list):
        raise ValueError("contextual_pains must be a list")
    normalized_pains = list(
        dict.fromkeys(
            " ".join(str(pain).split())
            for pain in pains
            if " ".join(str(pain).split())
        )
    )
    if not 3 <= len(normalized_pains) <= 12:
        raise ValueError(
            f"contextual_pains must contain 3-12 non-empty unique strings; "
            f"received {len(normalized_pains)}"
        )

    confidence = str(output.get("confidence") or "").strip().casefold()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence: {confidence!r}")
    return {
        "description": description,
        "audience_ids": normalized_audiences,
        "contextual_pains": normalized_pains,
        "confidence": confidence,
        "ignored_invalid_audience_ids": invalid_audiences,
    }


def _research_customer(
    customer: dict[str, Any],
    *,
    audience_catalog: Sequence[dict[str, Any]],
    use_cache: bool,
    serp_fn: Callable[..., Sequence[SearchResult]] = serp,
    tool_call_fn: Callable[..., tuple[dict[str, Any] | None, Any]] = local_tool_call,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    kept_results: list[dict[str, Any]] = []
    excluded_results: list[dict[str, Any]] = []
    for quoted in (False, True):
        query = _generic_company_query(customer["customer_name"], quoted=quoted)
        attempt: dict[str, Any] = {"query": query, "quoted_fallback": quoted}
        try:
            raw_results = list(
                serp_fn(
                    query,
                    provider="submission_local_corpus",
                    max_results=10,
                    use_cache=use_cache,
                )
            )
            kept, excluded = _filter_search_results(raw_results)
            attempt.update(
                {
                    "raw_result_count": len(raw_results),
                    "kept_result_count": len(kept),
                    "filtered_result_count": len(excluded),
                }
            )
            kept_results = kept
            excluded_results.extend(excluded)
        except Exception as exc:  # noqa: BLE001 - persisted for resumability
            attempt["error"] = repr(exc)
        attempts.append(attempt)
        if kept_results:
            break

    search = {
        "provider": "submission_local_corpus",
        "attempts": attempts,
        "results": kept_results,
        "filtered_results": excluded_results,
    }
    base = {
        **customer,
        "stage": "research",
        "status": "error",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "search": search,
        "profile": None,
        "errors": [],
    }
    if not kept_results:
        base["errors"] = ["websearch returned no usable non-case-study results"]
        return base

    snippet_blocks = []
    for index, result in enumerate(kept_results[:10], start=1):
        snippet_blocks.append(
            f"[{index}] {result['title']}\nURL: {result['url']}\n"
            f"Snippet: {result['snippet'][:1200]}"
        )
    audience_lines = [
        {
            "id": audience["id"],
            "name": audience.get("canonical_name") or audience.get("name") or "",
            "description": audience.get("description") or "",
        }
        for audience in audience_catalog
    ]
    user_prompt = (
        f"Company: {customer['customer_name']}\n\n"
        "Generic public search evidence:\n"
        + "\n\n".join(snippet_blocks)
        + "\n\nAvailable graph audiences (select only exact IDs from this list):\n"
        + json.dumps(audience_lines, ensure_ascii=False)
        + "\n\nCreate a vendor-independent company profile. Infer the company's "
        "business context only from the generic evidence. Return a concise "
        "description, the applicable industry/size/region/business-model audience "
        "IDs, and 3-12 concrete operational or strategic pain statements. Do not "
        "mention, recommend, or assume any software vendor."
    )
    output, usage = tool_call_fn(
        system=(
            "You are a rigorous B2B company researcher. Use only the supplied "
            "generic public evidence. Never introduce a seller, product vendor, "
            "customer case study, or success story. If evidence is weak, lower "
            "confidence rather than inventing details."
        ),
        user=user_prompt,
        tool_name="emit_independent_customer_profile",
        schema=_research_schema([audience["id"] for audience in audience_catalog]),
        max_tokens=2500,
        description="Emit a vendor-independent cold-start customer profile.",
    )
    base["llm_usage"] = _usage_record(usage)
    if output is None:
        base["errors"] = ["local Qwen failed to return a structured profile"]
        return base
    try:
        base["profile"] = validate_research_output(
            output,
            valid_audience_ids={audience["id"] for audience in audience_catalog},
        )
    except ValueError as exc:
        base["errors"] = [str(exc)]
        base["rejected_output"] = output
        return base
    base["status"] = "success"
    return base


def run_research(
    *,
    output_dir: Path,
    max_customers: int,
    workers: int,
    seed: int,
    selection: str = "random",
    use_cache: bool = True,
    cache_path: Path = DEFAULT_CACHE,
    index_maps_path: Path = DEFAULT_INDEX_MAPS,
    node_mappings_path: Path = DEFAULT_NODE_MAPPINGS,
    graph_nodes_path: Path = DEFAULT_GRAPH_NODES,
) -> dict[str, Any]:
    require_local()
    if workers <= 0:
        raise ValueError("workers must be positive")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    index_maps = _load_index_maps(index_maps_path)
    node_mappings = _load_node_mappings(node_mappings_path)
    node_catalog = _load_node_catalog(
        node_types={"audience", "customer"},
        path=graph_nodes_path,
    )
    audience_catalog = [
        node_catalog[audience_id]
        for audience_id, _ in sorted(
            node_mappings["audiences"].items(), key=lambda item: item[1]
        )
    ]
    if len(audience_catalog) != 50:
        raise ValueError(f"Expected 50 graph audiences, found {len(audience_catalog)}")
    customer_nodes = {
        node_id: node
        for node_id, node in node_catalog.items()
        if node.get("type") == "customer"
    }

    cohort, cohort_diagnostics = select_research_cohort(
        split=cache["split"],
        customer_idx_to_id=index_maps["customer_idx_to_id"],
        customer_nodes=customer_nodes,
        max_customers=max_customers,
        seed=seed,
        selection=selection,
    )
    output_path = output_dir / RESEARCH_FILENAME
    existing = _load_latest_jsonl(output_path)
    pending = []
    for customer in cohort:
        previous = existing.get(customer["customer_index"], {})
        identity = previous.get("identity_validation") or {}
        reusable = (
            previous.get("status") == "success"
            and previous.get("customer_id") == customer["customer_id"]
            and previous.get("customer_name") == customer["customer_name"]
            and identity.get("exact_slug_match") is True
            and identity.get("canonical_name_sid")
            == customer["identity_validation"]["canonical_name_sid"]
        )
        if not reusable:
            pending.append(customer)

    completed = 0
    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _research_customer,
                customer,
                audience_catalog=audience_catalog,
                use_cache=use_cache,
            ): customer
            for customer in pending
        }
        for future in as_completed(futures):
            customer = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - persist worker failures
                record = {
                    **customer,
                    "stage": "research",
                    "status": "error",
                    "pipeline_version": PIPELINE_VERSION,
                    "schema_version": RESEARCH_SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "search": None,
                    "profile": None,
                    "errors": [repr(exc)],
                }
            _append_jsonl(output_path, record)
            completed += 1
            if record["status"] == "success":
                succeeded += 1
            else:
                failed += 1
            print(
                f"[research {completed}/{len(pending)}] "
                f"{record['customer_name']}: {record['status']}",
                flush=True,
            )

    latest = _load_latest_jsonl(output_path)
    selected_successes = sum(
        latest.get(customer["customer_index"], {}).get("status") == "success"
        for customer in cohort
    )
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "output": str(output_path),
        "seed": seed,
        "max_customers": max_customers,
        "workers": workers,
        "selection": selection,
        "use_cache": use_cache,
        "selected_customers": len(cohort),
        "already_successful": len(cohort) - len(pending),
        "attempted_this_run": len(pending),
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "successful_selected_customers": selected_successes,
        "cohort_selection": cohort_diagnostics,
        "customer_name_source": str(graph_nodes_path),
        "selected_customer_indices": [
            customer["customer_index"] for customer in cohort
        ],
        "selected_customer_ids": [
            customer["customer_id"] for customer in cohort
        ],
    }
    _atomic_write_json(output_dir / RESEARCH_MANIFEST, summary)
    print(json.dumps(summary, indent=2))
    return summary


class LocalBGEEncoder:
    """Load the cached BGE model without permitting a network download."""

    def __init__(
        self,
        *,
        model_name: str = EMBEDDING_MODEL,
        device: str,
        cache_folder: Path = DEFAULT_MODEL_CACHE,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._sentence_transformer: Any = None
        self._tokenizer: Any = None
        self._transformer: Any = None

        try:
            from sentence_transformers import SentenceTransformer

            self._sentence_transformer = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=str(cache_folder),
                local_files_only=True,
            )
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_folder),
                local_files_only=True,
            )
            self._transformer = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_folder),
                local_files_only=True,
            ).to(device)
            self._transformer.eval()

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        if self._sentence_transformer is not None:
            return np.asarray(
                self._sentence_transformer.encode(
                    list(texts),
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                ),
                dtype=np.float32,
            )

        assert self._tokenizer is not None and self._transformer is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self._tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._transformer(**encoded)
                embeddings = torch.nn.functional.normalize(
                    output.last_hidden_state[:, 0],
                    p=2,
                    dim=1,
                )
            batches.append(embeddings.cpu().numpy().astype(np.float32))
        return np.vstack(batches)


def cosine_shortlists(
    query_embeddings: np.ndarray,
    canonical_embeddings: np.ndarray,
    *,
    top_k: int = 5,
    score_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if query_embeddings.ndim != 2 or canonical_embeddings.ndim != 2:
        raise ValueError("query and canonical embeddings must be matrices")
    if query_embeddings.shape[1] != canonical_embeddings.shape[1]:
        raise ValueError("query and canonical embedding dimensions differ")
    if not 0 < top_k <= canonical_embeddings.shape[0]:
        raise ValueError("top_k is out of range")

    queries = np.asarray(query_embeddings, dtype=np.float32)
    canonicals = np.asarray(canonical_embeddings, dtype=np.float32)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    canonicals /= np.maximum(
        np.linalg.norm(canonicals, axis=1, keepdims=True), 1e-8
    )
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, queries.shape[0], score_batch_size):
        scores = queries[start : start + score_batch_size] @ canonicals.T
        candidates = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        all_indices.append(np.take_along_axis(candidates, order, axis=1))
        all_scores.append(np.take_along_axis(candidate_scores, order, axis=1))
    return np.vstack(all_indices), np.vstack(all_scores)


def _resolution_schema(candidate_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pain_index": {"type": "integer"},
                        "canonical_id": {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "enum": sorted(set(candidate_ids)),
                                },
                                {"type": "null"},
                            ]
                        },
                        "reason": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": [
                        "pain_index",
                        "canonical_id",
                        "reason",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["selections"],
        "additionalProperties": False,
    }


def validate_canonical_selections(
    pains_with_candidates: Sequence[dict[str, Any]],
    output: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Validate exact per-pain candidate membership and preserve null matches."""
    if not isinstance(output, dict) or not isinstance(output.get("selections"), list):
        raise ValueError("Qwen returned no canonical selections")
    selections = output["selections"]
    if len(selections) != len(pains_with_candidates):
        raise ValueError(
            f"expected {len(pains_with_candidates)} selections, received "
            f"{len(selections)}"
        )

    by_index: dict[int, dict[str, Any]] = {}
    for selection in selections:
        if not isinstance(selection, dict):
            raise ValueError("selection entries must be objects")
        pain_index = selection.get("pain_index")
        if not isinstance(pain_index, int):
            raise ValueError(f"invalid pain_index: {pain_index!r}")
        if pain_index in by_index:
            raise ValueError(f"duplicate pain_index: {pain_index}")
        by_index[pain_index] = selection
    expected = set(range(len(pains_with_candidates)))
    if set(by_index) != expected:
        raise ValueError(
            f"selection indexes differ from expected indexes: "
            f"{sorted(set(by_index) ^ expected)}"
        )

    resolved: list[dict[str, Any]] = []
    for pain_index, pain in enumerate(pains_with_candidates):
        selection = by_index[pain_index]
        candidate_ids = {
            str(candidate["canonical_id"]) for candidate in pain["candidates"]
        }
        canonical_id = selection.get("canonical_id")
        if canonical_id is not None and canonical_id not in candidate_ids:
            raise ValueError(
                f"invented canonical selection for pain {pain_index}: {canonical_id!r}"
            )
        confidence = selection.get("confidence")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(
                f"invalid selection confidence for pain {pain_index}: {confidence!r}"
            )
        reason = " ".join(str(selection.get("reason") or "").split())
        if not reason:
            raise ValueError(f"empty selection reason for pain {pain_index}")
        resolved.append(
            {
                **pain,
                "selected_canonical_id": canonical_id,
                "matched": canonical_id is not None,
                "selection_reason": reason,
                "selection_confidence": confidence,
            }
        )
    return resolved


def _resolve_customer(
    research_record: dict[str, Any],
    pains_with_candidates: list[dict[str, Any]],
    *,
    tool_call_fn: Callable[..., tuple[dict[str, Any] | None, Any]] = local_tool_call,
) -> dict[str, Any]:
    profile = research_record["profile"]
    prompt_pains = []
    for pain_index, pain in enumerate(pains_with_candidates):
        prompt_pains.append(
            {
                "pain_index": pain_index,
                "contextual_pain": pain["contextual_pain"],
                "candidates": [
                    {
                        "canonical_id": candidate["canonical_id"],
                        "canonical_name": candidate["canonical_name"],
                        "scope_definition": candidate["scope_definition"][:700],
                        "cosine_similarity": candidate["cosine_similarity"],
                    }
                    for candidate in pain["candidates"]
                ],
            }
        )
    candidate_ids = [
        candidate["canonical_id"]
        for pain in pains_with_candidates
        for candidate in pain["candidates"]
    ]
    output, usage = tool_call_fn(
        system=(
            "You resolve independently researched business pain statements to a "
            "fixed canonical taxonomy. For each pain, select at most one supplied "
            "canonical ID. Select null when none is a defensible semantic match. "
            "Never invent an ID and never force a weak match."
        ),
        user=(
            f"Company context: {profile['description']}\n\n"
            "Resolve every indexed contextual pain using only its supplied top-five "
            "candidates. Return exactly one selection per pain index.\n\n"
            + json.dumps(prompt_pains, ensure_ascii=False)
        ),
        tool_name="emit_canonical_pain_selections",
        schema=_resolution_schema(candidate_ids),
        max_tokens=3500,
        description="Select one supplied canonical pain ID or null per pain.",
    )
    base = {
        "customer_index": int(research_record["customer_index"]),
        "customer_id": research_record["customer_id"],
        "customer_name": research_record["customer_name"],
        "stage": "resolve",
        "status": "error",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": RESOLVE_SCHEMA_VERSION,
        "research_schema_version": research_record["schema_version"],
        "created_at": _utc_now(),
        "profile": copy.deepcopy(profile),
        "pain_candidates": copy.deepcopy(pains_with_candidates),
        "resolved_pains": None,
        "matched_canonical_ids": [],
        "unmatched_pains": [],
        "errors": [],
        "llm_usage": _usage_record(usage),
    }
    if output is None:
        base["errors"] = ["local Qwen failed to return canonical selections"]
        return base
    try:
        resolved = validate_canonical_selections(pains_with_candidates, output)
    except ValueError as exc:
        base["errors"] = [str(exc)]
        base["rejected_output"] = output
        return base

    base["resolved_pains"] = resolved
    base.pop("pain_candidates", None)
    base["matched_canonical_ids"] = list(
        dict.fromkeys(
            pain["selected_canonical_id"]
            for pain in resolved
            if pain["selected_canonical_id"] is not None
        )
    )
    base["unmatched_pains"] = [
        pain["contextual_pain"] for pain in resolved if not pain["matched"]
    ]
    base["status"] = "success"
    return base


def run_resolve(
    *,
    output_dir: Path,
    workers: int,
    device: str,
    node_mappings_path: Path = DEFAULT_NODE_MAPPINGS,
    pain_features_path: Path = DEFAULT_PAIN_FEATURES,
    graph_nodes_path: Path = DEFAULT_GRAPH_NODES,
) -> dict[str, Any]:
    require_local()
    if workers <= 0:
        raise ValueError("workers must be positive")

    research_path = output_dir / RESEARCH_FILENAME
    resolved_path = output_dir / RESOLVED_FILENAME
    research_manifest_path = output_dir / RESEARCH_MANIFEST
    if not research_manifest_path.exists():
        raise ValueError(
            f"Identity-safe research manifest not found: {research_manifest_path}"
        )
    with open(research_manifest_path, encoding="utf-8") as handle:
        research_manifest = json.load(handle)
    selected_indices = {
        int(index)
        for index in research_manifest.get("selected_customer_indices", [])
    }
    if not selected_indices:
        raise ValueError(
            "Research manifest has no identity-safe selected_customer_indices"
        )

    graph_catalog = _load_node_catalog(
        node_types={"customer", "pain_canonical"},
        path=graph_nodes_path,
    )
    customer_nodes = {
        node_id: node
        for node_id, node in graph_catalog.items()
        if node.get("type") == "customer"
    }
    research_records: dict[int, dict[str, Any]] = {}
    identity_rejections: dict[str, int] = {}
    for index, record in _load_latest_jsonl(research_path).items():
        if record.get("status") != "success":
            continue
        if index not in selected_indices:
            reason = "not_selected_identity_safe_cohort"
        elif (record.get("identity_validation") or {}).get(
            "exact_slug_match"
        ) is not True:
            reason = "missing_identity_validation"
        else:
            reason = profile_record_identity_rejection_reason(
                record,
                customer_nodes,
            )
        if reason is not None:
            identity_rejections[reason] = identity_rejections.get(reason, 0) + 1
            continue
        research_records[index] = record
    if not research_records:
        raise ValueError(
            f"No identity-safe successful research profiles found in {research_path}; "
            f"rejections={identity_rejections}"
        )
    existing = _load_latest_jsonl(resolved_path)
    pending = []
    for index, record in sorted(research_records.items()):
        previous = existing.get(index, {})
        reusable = (
            previous.get("status") == "success"
            and previous.get("customer_id") == record["customer_id"]
            and previous.get("customer_name") == record["customer_name"]
            and profile_record_identity_rejection_reason(
                previous,
                customer_nodes,
            )
            is None
        )
        if not reusable:
            pending.append(record)
    if not pending:
        summary = {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": RESOLVE_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "output": str(resolved_path),
            "successful_research_profiles": len(research_records),
            "attempted_this_run": 0,
            "already_successful": len(research_records),
            "identity_rejected_research_records": dict(
                sorted(identity_rejections.items())
            ),
        }
        _atomic_write_json(output_dir / RESOLVE_MANIFEST, summary)
        print(json.dumps(summary, indent=2))
        return summary

    node_mappings = _load_node_mappings(node_mappings_path)
    pain_id_to_index = {
        pain_id: int(index) for pain_id, index in node_mappings["pains"].items()
    }
    pain_ids = [
        pain_id
        for pain_id, _ in sorted(pain_id_to_index.items(), key=lambda item: item[1])
    ]
    pain_catalog = {
        node_id: node
        for node_id, node in graph_catalog.items()
        if node.get("type") == "pain_canonical"
    }
    if set(pain_ids) != set(pain_catalog):
        raise ValueError("pain node mapping and v6 pain catalog differ")
    canonical_embeddings = np.load(pain_features_path)
    if canonical_embeddings.shape[0] != len(pain_ids):
        raise ValueError("pain feature rows do not match node mapping")

    flattened: list[tuple[int, int, str]] = []
    for record in pending:
        for pain_index, pain in enumerate(record["profile"]["contextual_pains"]):
            flattened.append((int(record["customer_index"]), pain_index, pain))
    encoder = LocalBGEEncoder(device=device)
    query_embeddings = encoder.encode([pain for _, _, pain in flattened])
    shortlist_indices, shortlist_scores = cosine_shortlists(
        query_embeddings,
        canonical_embeddings,
        top_k=5,
    )

    candidates_by_customer: dict[int, list[dict[str, Any]]] = {
        int(record["customer_index"]): [
            {
                "contextual_pain": pain,
                "candidates": [],
            }
            for pain in record["profile"]["contextual_pains"]
        ]
        for record in pending
    }
    for row, (customer_index, pain_index, _) in enumerate(flattened):
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
        candidates_by_customer[customer_index][pain_index]["candidates"] = candidates

    completed = 0
    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _resolve_customer,
                record,
                candidates_by_customer[int(record["customer_index"])],
            ): record
            for record in pending
        }
        for future in as_completed(futures):
            research_record = futures[future]
            try:
                resolved_record = future.result()
            except Exception as exc:  # noqa: BLE001 - persisted for resumability
                resolved_record = {
                    "customer_index": int(research_record["customer_index"]),
                    "customer_id": research_record["customer_id"],
                    "customer_name": research_record["customer_name"],
                    "stage": "resolve",
                    "status": "error",
                    "pipeline_version": PIPELINE_VERSION,
                    "schema_version": RESOLVE_SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "profile": copy.deepcopy(research_record["profile"]),
                    "pain_candidates": candidates_by_customer[
                        int(research_record["customer_index"])
                    ],
                    "resolved_pains": None,
                    "matched_canonical_ids": [],
                    "unmatched_pains": [],
                    "errors": [repr(exc)],
                }
            _append_jsonl(resolved_path, resolved_record)
            completed += 1
            if resolved_record["status"] == "success":
                succeeded += 1
            else:
                failed += 1
            print(
                f"[resolve {completed}/{len(pending)}] "
                f"{resolved_record['customer_name']}: {resolved_record['status']}",
                flush=True,
            )

    latest = _load_latest_jsonl(resolved_path)
    successful = sum(
        latest.get(index, {}).get("status") == "success"
        for index in research_records
    )
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": RESOLVE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "output": str(resolved_path),
        "device": device,
        "workers": workers,
        "embedding_model": EMBEDDING_MODEL,
        "successful_research_profiles": len(research_records),
        "already_successful": len(research_records) - len(pending),
        "attempted_this_run": len(pending),
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "successful_resolved_profiles": successful,
        "identity_rejected_research_records": dict(
            sorted(identity_rejections.items())
        ),
    }
    _atomic_write_json(output_dir / RESOLVE_MANIFEST, summary)
    print(json.dumps(summary, indent=2))
    return summary


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return values / norm


def build_independent_feature_row(
    *,
    resolved_record: dict[str, Any],
    pain_id_to_index: dict[str, int],
    pain_to_subtheme: np.ndarray,
    pain_features: np.ndarray,
    audience_id_to_index: dict[str, int],
    tfidf_subtheme_idf: np.ndarray,
    semantic_subtheme_idf: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Construct all six normalized feature blocks without graph customer edges."""
    n_audiences = len(audience_id_to_index)
    n_subthemes = tfidf_subtheme_idf.shape[0]
    if semantic_subtheme_idf.shape != (n_subthemes,):
        raise ValueError("semantic and TF-IDF subtheme IDF dimensions differ")
    if pain_to_subtheme.shape[0] != pain_features.shape[0]:
        raise ValueError("pain-to-subtheme and pain feature counts differ")

    audience = np.zeros(n_audiences, dtype=np.float32)
    for audience_id in dict.fromkeys(resolved_record["profile"]["audience_ids"]):
        if audience_id not in audience_id_to_index:
            raise ValueError(f"unknown audience ID in resolved profile: {audience_id}")
        audience[audience_id_to_index[audience_id]] = 1.0

    selected_ids = list(
        dict.fromkeys(
            pain["selected_canonical_id"]
            for pain in resolved_record["resolved_pains"]
            if pain.get("selected_canonical_id") is not None
        )
    )
    selected_indices = []
    for pain_id in selected_ids:
        if pain_id not in pain_id_to_index:
            raise ValueError(f"unknown canonical pain ID in resolved profile: {pain_id}")
        selected_indices.append(pain_id_to_index[pain_id])

    subtheme_counts = np.zeros(n_subthemes, dtype=np.float32)
    semantic_sum = np.zeros(pain_features.shape[1], dtype=np.float32)
    semantic_weight_sum = 0.0
    for pain_index in selected_indices:
        subtheme_index = int(pain_to_subtheme[pain_index])
        if not 0 <= subtheme_index < n_subthemes:
            continue
        subtheme_counts[subtheme_index] += 1.0
        weight = float(semantic_subtheme_idf[subtheme_index])
        pain_feature = pain_features[pain_index]
        normalized_pain_feature = pain_feature / max(
            float(np.linalg.norm(pain_feature)),
            1e-8,
        )
        semantic_sum += normalized_pain_feature * weight
        semantic_weight_sum += weight

    if semantic_weight_sum > 0:
        semantic_sum /= semantic_weight_sum
    semantic_only = _normalize_vector(semantic_sum)
    tfidf_only = _normalize_vector(subtheme_counts * tfidf_subtheme_idf)
    audience_only = _normalize_vector(audience)
    semantic = np.concatenate([semantic_only, audience_only]) / math.sqrt(2.0)
    tfidf = np.concatenate([tfidf_only, audience_only]) / math.sqrt(2.0)
    fusion = (
        np.concatenate([semantic_only, tfidf_only, audience_only])
        / math.sqrt(3.0)
    )
    return {
        "audience": torch.from_numpy(audience_only),
        "semantic_only": torch.from_numpy(semantic_only),
        "tfidf_only": torch.from_numpy(tfidf_only),
        "semantic": torch.from_numpy(semantic.astype(np.float32)),
        "tfidf": torch.from_numpy(tfidf.astype(np.float32)),
        "fusion": torch.from_numpy(fusion.astype(np.float32)),
    }


def materialize_customer_feature_blocks(
    customer_features: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Clone existing matrices and recover legacy missing single-block variants."""
    required = {"audience", "semantic", "tfidf", "fusion"}
    missing = required - set(customer_features)
    if missing:
        raise ValueError(f"source cache is missing customer features: {sorted(missing)}")
    features = {
        name: values.detach().cpu().to(torch.float32).clone()
        for name, values in customer_features.items()
    }
    audience_dim = int(features["audience"].shape[1])
    semantic_only_dim = int(features["semantic"].shape[1] - audience_dim)
    tfidf_only_dim = int(features["tfidf"].shape[1] - audience_dim)
    if semantic_only_dim <= 0 or tfidf_only_dim <= 0:
        raise ValueError("combined customer feature dimensions are invalid")

    if "semantic_only" not in features:
        features["semantic_only"] = (
            features["semantic"][:, :semantic_only_dim] * math.sqrt(2.0)
        )
    if "tfidf_only" not in features:
        features["tfidf_only"] = (
            features["tfidf"][:, :tfidf_only_dim] * math.sqrt(2.0)
        )
    dimensions = {
        "audience": audience_dim,
        "semantic_only": semantic_only_dim,
        "tfidf_only": tfidf_only_dim,
        "semantic": semantic_only_dim + audience_dim,
        "tfidf": tfidf_only_dim + audience_dim,
        "fusion": semantic_only_dim + tfidf_only_dim + audience_dim,
    }
    for name, expected_dim in dimensions.items():
        if features[name].shape[1] != expected_dim:
            raise ValueError(
                f"{name} dimension is {features[name].shape[1]}, expected {expected_dim}"
            )
    return features, dimensions


def derive_customer_holdout_split(
    split: dict[str, Any],
    successful_customer_indices: Iterable[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep only independently profiled customer-holdout candidates and edges."""
    original_holdout = set(map(int, split["holdout_customers"]))
    customer_holdout_edges = [
        (int(vendor), int(customer))
        for vendor, customer in split["test_slices"].get("customer_holdout", [])
    ]
    positive_customers = {customer for _, customer in customer_holdout_edges}
    successful = set(map(int, successful_customer_indices))
    retained_customers = sorted(successful & original_holdout & positive_customers)
    retained_set = set(retained_customers)
    retained_edges = [
        edge for edge in customer_holdout_edges if edge[1] in retained_set
    ]

    derived = copy.deepcopy(split)
    derived["holdout_customers"] = retained_customers
    derived["customer_holdout_candidates"] = retained_customers
    derived["test_slices"] = {
        "customer_holdout": retained_edges,
        "vendor_holdout": [],
        "double_holdout": [],
    }
    derived["profile_source"] = "independent_public_research"
    derived["omitted_test_slices"] = {
        "vendor_holdout": (
            "Omitted: this first independent-profile artifact is scoped to "
            "customer_holdout."
        ),
        "double_holdout": (
            "Omitted: this first independent-profile artifact is scoped to "
            "customer_holdout."
        ),
    }
    diagnostics = {
        "original_holdout_customers": len(original_holdout),
        "original_customer_holdout_positive_customers": len(positive_customers),
        "successful_profile_records": len(successful),
        "retained_customer_holdout_customers": len(retained_customers),
        "original_customer_holdout_edges": len(customer_holdout_edges),
        "retained_customer_holdout_edges": len(retained_edges),
        "ignored_successful_profiles": sorted(
            successful - (original_holdout & positive_customers)
        ),
    }
    return derived, diagnostics


def _training_only_idf(
    customer_tfidf: torch.Tensor,
    train_customers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    train_index = torch.tensor(list(map(int, train_customers)), dtype=torch.long)
    training_rows = customer_tfidf.detach().cpu()[train_index]
    document_frequency = (training_rows != 0).sum(dim=0).to(torch.float32)
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
    return tfidf_idf.numpy(), semantic_idf.numpy()


def run_build_cache(
    *,
    output_dir: Path,
    cache_out: Path,
    source_cache_path: Path = DEFAULT_CACHE,
    hetero_data_path: Path = DEFAULT_HETERO_DATA,
    node_mappings_path: Path = DEFAULT_NODE_MAPPINGS,
    pain_features_path: Path = DEFAULT_PAIN_FEATURES,
    graph_nodes_path: Path = DEFAULT_GRAPH_NODES,
) -> dict[str, Any]:
    resolved_path = output_dir / RESOLVED_FILENAME
    customer_nodes = _load_node_catalog(
        node_types={"customer"},
        path=graph_nodes_path,
    )
    successful_records: dict[int, dict[str, Any]] = {}
    identity_rejections: dict[str, int] = {}
    for index, record in _load_latest_jsonl(resolved_path).items():
        if record.get("status") != "success":
            continue
        reason = profile_record_identity_rejection_reason(
            record,
            customer_nodes,
        )
        if reason is not None:
            identity_rejections[reason] = identity_rejections.get(reason, 0) + 1
            continue
        successful_records[index] = record
    if not successful_records:
        raise ValueError(
            f"No identity-safe successful resolved profiles found in "
            f"{resolved_path}; rejections={identity_rejections}"
        )

    source = torch.load(source_cache_path, map_location="cpu", weights_only=False)
    hetero_checkpoint = torch.load(
        hetero_data_path, map_location="cpu", weights_only=False
    )
    data = hetero_checkpoint["data"]
    node_mappings = _load_node_mappings(node_mappings_path)
    pain_id_to_index = {
        pain_id: int(index) for pain_id, index in node_mappings["pains"].items()
    }
    audience_id_to_index = {
        audience_id: int(index)
        for audience_id, index in node_mappings["audiences"].items()
    }
    pain_features = np.asarray(np.load(pain_features_path), dtype=np.float32)
    if pain_features.shape[0] != len(pain_id_to_index):
        raise ValueError("pain feature rows do not match node mappings")

    n_pains = pain_features.shape[0]
    pain_to_subtheme = np.full(n_pains, -1, dtype=np.int64)
    subtheme_edges = data[
        "pain_canonical", "in_subtheme", "pain_subtheme"
    ].edge_index.cpu()
    pain_to_subtheme[subtheme_edges[0].numpy()] = subtheme_edges[1].numpy()
    if np.any(pain_to_subtheme < 0):
        raise ValueError("one or more canonical pains lack a subtheme mapping")

    features, feature_dimensions = materialize_customer_feature_blocks(
        source["customer_features"]
    )
    tfidf_subtheme_idf, semantic_subtheme_idf = _training_only_idf(
        data["customer"].x,
        source["split"]["train_customers"],
    )
    if feature_dimensions["semantic_only"] != pain_features.shape[1]:
        raise ValueError("semantic feature dimension does not match pain embeddings")
    if feature_dimensions["tfidf_only"] != tfidf_subtheme_idf.shape[0]:
        raise ValueError("TF-IDF feature dimension does not match subthemes")
    if feature_dimensions["audience"] != len(audience_id_to_index):
        raise ValueError("audience feature dimension does not match graph audiences")

    derived_split, split_diagnostics = derive_customer_holdout_split(
        source["split"],
        successful_records,
    )
    retained_indices = set(derived_split["holdout_customers"])
    retained_records = {
        index: record
        for index, record in successful_records.items()
        if index in retained_indices
    }
    if not retained_records:
        raise ValueError(
            "Successful resolved profiles do not overlap eligible customer_holdout nodes"
        )

    matched_pain_count = 0
    unmatched_pain_count = 0
    profiles_with_matched_pains = 0
    profiles_with_audiences = 0
    for customer_index, record in sorted(retained_records.items()):
        row = build_independent_feature_row(
            resolved_record=record,
            pain_id_to_index=pain_id_to_index,
            pain_to_subtheme=pain_to_subtheme,
            pain_features=pain_features,
            audience_id_to_index=audience_id_to_index,
            tfidf_subtheme_idf=tfidf_subtheme_idf,
            semantic_subtheme_idf=semantic_subtheme_idf,
        )
        for feature_name, values in row.items():
            features[feature_name][customer_index] = values
        n_matched = len(record["matched_canonical_ids"])
        n_unmatched = len(record["unmatched_pains"])
        matched_pain_count += n_matched
        unmatched_pain_count += n_unmatched
        profiles_with_matched_pains += int(n_matched > 0)
        profiles_with_audiences += int(bool(record["profile"]["audience_ids"]))

    fingerprint_payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_cache_fingerprint": source.get("fingerprint"),
        "resolved_profiles_sha256": _sha256_file(resolved_path),
        "node_mappings_sha256": _sha256_file(node_mappings_path),
        "pain_features_size": pain_features_path.stat().st_size,
        "graph_nodes_size": graph_nodes_path.stat().st_size,
        "profiled_customer_indices": sorted(retained_records),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()

    artifact = dict(source)
    artifact.pop("edge_transfer", None)
    artifact["created_at"] = _utc_now()
    artifact["fingerprint"] = fingerprint
    artifact["source_fingerprint"] = source.get("fingerprint")
    artifact["customer_features"] = features
    artifact["split"] = derived_split
    artifact["oracle_profile_warning"] = (
        "Training-customer rows retain their original graph-derived features. "
        "Every customer_holdout candidate row was replaced using independent "
        "vendor-blind public research; unprofiled holdout customers are excluded."
    )
    source_diagnostics = dict(source.get("diagnostics", {}))
    profile_diagnostics = {
        **split_diagnostics,
        "coverage_fraction": (
            len(retained_records)
            / max(split_diagnostics["original_customer_holdout_positive_customers"], 1)
        ),
        "profiles_with_matched_pains": profiles_with_matched_pains,
        "profiles_with_audiences": profiles_with_audiences,
        "matched_unique_canonical_pains": matched_pain_count,
        "unmatched_contextual_pains": unmatched_pain_count,
        "identity_rejected_resolved_profiles": dict(
            sorted(identity_rejections.items())
        ),
        "feature_dimensions": feature_dimensions,
        "idf_document_population": "original split train_customers only",
        "omitted_protocols": ["edge_transfer", "vendor_holdout", "double_holdout"],
    }
    source_diagnostics["independent_profiles"] = profile_diagnostics
    artifact["diagnostics"] = source_diagnostics
    artifact["independent_profile"] = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "fingerprint_payload": fingerprint_payload,
        "profiled_customer_indices": sorted(retained_records),
        "provenance": {
            "research_profiles": str((output_dir / RESEARCH_FILENAME).resolve()),
            "resolved_profiles": str(resolved_path.resolve()),
            "source_cache": str(source_cache_path.resolve()),
            "hetero_data": str(hetero_data_path.resolve()),
            "node_mappings": str(node_mappings_path.resolve()),
            "pain_features": str(pain_features_path.resolve()),
            "graph_nodes": str(graph_nodes_path.resolve()),
        },
        "construction_contract": {
            "test_rows_use_heldout_graph_pains": False,
            "test_rows_use_heldout_graph_audiences": False,
            "training_rows_reused": True,
            "identity_safe_customer_names": True,
            "identity_rule": (
                "sid(v6_customer.name) == customer_id slug, with ASCII letters "
                "and a non-generic company-like canonical name"
            ),
            "tfidf_idf_formula": "log(n_train_customers / (train_df + 1))",
            "semantic_pain_weight_formula": (
                "log((n_train_customers + 1) / (train_df + 1)) + 1"
            ),
            "block_scaling": {
                "semantic": "semantic_only + audience, divided by sqrt(2)",
                "tfidf": "tfidf_only + audience, divided by sqrt(2)",
                "fusion": (
                    "semantic_only + tfidf_only + audience, divided by sqrt(3)"
                ),
            },
        },
    }
    _atomic_torch_save(cache_out, artifact)
    summary = {
        "output": str(cache_out),
        "fingerprint": fingerprint,
        "profile_diagnostics": profile_diagnostics,
        "provenance": artifact["independent_profile"]["provenance"],
    }
    print(json.dumps(summary, indent=2))
    return summary


_EVALUATION_METRICS = (
    "map",
    "ndcg_at_50",
    "precision_at_10",
    "chance_precision_at_10",
    "precision_at_10_lift",
    "recall_at_50",
    "profile_coverage",
)


def validate_evaluation_cache(
    cache: dict[str, Any],
) -> dict[str, Any]:
    """Validate the independent-cache contract and return coverage metadata."""
    fingerprint = cache.get("fingerprint")
    source_fingerprint = cache.get("source_fingerprint")
    metadata = cache.get("independent_profile")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("derived cache has no fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("derived cache has no source_fingerprint")
    if not isinstance(metadata, dict):
        raise ValueError("cache is not an independent-profile derived artifact")
    if metadata.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported independent profile schema: "
            f"{metadata.get('schema_version')!r}"
        )

    fingerprint_payload = metadata.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise ValueError("derived cache is missing its fingerprint payload")
    if fingerprint_payload.get("source_cache_fingerprint") != source_fingerprint:
        raise ValueError("source fingerprint disagrees with fingerprint payload")
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("derived cache fingerprint does not match its payload")

    contract = metadata.get("construction_contract")
    if not isinstance(contract, dict):
        raise ValueError("derived cache is missing its construction contract")
    if contract.get("test_rows_use_heldout_graph_pains") is not False:
        raise ValueError("cache does not prohibit heldout graph pains in test rows")
    if contract.get("test_rows_use_heldout_graph_audiences") is not False:
        raise ValueError("cache does not prohibit heldout graph audiences in test rows")
    if contract.get("training_rows_reused") is not True:
        raise ValueError("cache does not attest that training rows were reused")
    if contract.get("identity_safe_customer_names") is not True:
        raise ValueError("cache does not attest identity-safe customer names")

    provenance = metadata.get("provenance")
    required_provenance = {
        "research_profiles",
        "resolved_profiles",
        "source_cache",
        "hetero_data",
        "node_mappings",
        "pain_features",
        "graph_nodes",
    }
    if not isinstance(provenance, dict) or not required_provenance <= set(provenance):
        raise ValueError("derived cache has incomplete provenance paths")

    split = cache.get("split")
    if not isinstance(split, dict):
        raise ValueError("derived cache has no split")
    if split.get("profile_source") != "independent_public_research":
        raise ValueError("customer holdout split is not marked independently profiled")
    candidates = list(map(int, split.get("customer_holdout_candidates", [])))
    holdout_customers = list(map(int, split.get("holdout_customers", [])))
    if not candidates or candidates != holdout_customers:
        raise ValueError(
            "customer_holdout candidates must be the non-empty holdout customer list"
        )
    if len(candidates) != len(set(candidates)):
        raise ValueError("customer_holdout candidates contain duplicates")

    profiled = list(map(int, metadata.get("profiled_customer_indices", [])))
    if profiled != candidates:
        raise ValueError(
            "profiled_customer_indices differ from customer_holdout candidates"
        )
    test_slices = split.get("test_slices") or {}
    customer_edges = [
        (int(vendor), int(customer))
        for vendor, customer in test_slices.get("customer_holdout", [])
    ]
    if not customer_edges:
        raise ValueError("derived customer_holdout has no positive edges")
    candidate_set = set(candidates)
    if any(customer not in candidate_set for _, customer in customer_edges):
        raise ValueError("customer_holdout edges reference unprofiled customers")
    train_vendors = set(map(int, split.get("train_vendors", [])))
    if any(vendor not in train_vendors for vendor, _ in customer_edges):
        raise ValueError("customer_holdout edges reference heldout vendors")
    if test_slices.get("vendor_holdout") or test_slices.get("double_holdout"):
        raise ValueError(
            "derived cache must omit vendor_holdout and double_holdout evaluations"
        )

    customer_features = cache.get("customer_features")
    vendor_features = cache.get("vendor_features")
    if not isinstance(customer_features, dict) or not customer_features:
        raise ValueError("derived cache has no customer feature tensors")
    if not isinstance(vendor_features, dict) or not vendor_features:
        raise ValueError("derived cache has no vendor feature tensors")
    customer_counts = {
        int(values.shape[0]) for values in customer_features.values()
    }
    vendor_counts = {int(values.shape[0]) for values in vendor_features.values()}
    if len(customer_counts) != 1 or len(vendor_counts) != 1:
        raise ValueError("feature tensors disagree on node counts")
    n_customers = next(iter(customer_counts))
    n_vendors = next(iter(vendor_counts))
    if max(candidates) >= n_customers:
        raise ValueError("customer_holdout candidate exceeds customer feature rows")
    if max(vendor for vendor, _ in customer_edges) >= n_vendors:
        raise ValueError("customer_holdout vendor exceeds vendor feature rows")

    diagnostics = (
        cache.get("diagnostics", {}).get("independent_profiles", {})
    )
    if not isinstance(diagnostics, dict):
        raise ValueError("cache lacks independent profile diagnostics")
    original_positive_customers = int(
        diagnostics.get("original_customer_holdout_positive_customers", 0)
    )
    if original_positive_customers < len(candidates):
        raise ValueError("profile coverage denominator is smaller than the cohort")
    expected_coverage = len(candidates) / max(original_positive_customers, 1)
    recorded_coverage = float(diagnostics.get("coverage_fraction", -1.0))
    if not math.isclose(
        recorded_coverage,
        expected_coverage,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError("recorded profile coverage is inconsistent with the split")
    if int(diagnostics.get("retained_customer_holdout_edges", -1)) != len(
        customer_edges
    ):
        raise ValueError("recorded retained edge count is inconsistent with the split")

    provenance_validation: dict[str, Any] = {}
    checks = (
        ("resolved_profiles", "resolved_profiles_sha256"),
        ("node_mappings", "node_mappings_sha256"),
    )
    for path_key, hash_key in checks:
        path = Path(str(provenance[path_key]))
        expected_hash = fingerprint_payload.get(hash_key)
        if path.exists() and expected_hash:
            actual_hash = _sha256_file(path)
            provenance_validation[path_key] = {
                "path": str(path),
                "exists": True,
                "sha256_at_build": expected_hash,
                "current_sha256": actual_hash,
                "current_matches_build": actual_hash == expected_hash,
            }
        else:
            provenance_validation[path_key] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256_at_build": expected_hash,
                "current_sha256": None,
                "current_matches_build": None,
            }

    return {
        "derived_cache_fingerprint": fingerprint,
        "source_cache_fingerprint": source_fingerprint,
        "profiled_customers": len(candidates),
        "original_customer_holdout_positive_customers": original_positive_customers,
        "profile_coverage": expected_coverage,
        "customer_holdout_edges": len(customer_edges),
        "customer_holdout_vendors": len({vendor for vendor, _ in customer_edges}),
        "provenance_paths": provenance,
        "provenance_validation": provenance_validation,
    }


def discover_oracle_checkpoint_runs(
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    """Pair every oracle checkpoint with its completed training result JSON."""
    checkpoint_paths = sorted(
        checkpoint_dir.glob("oracle_node_holdout__*.pt")
    )
    if not checkpoint_paths:
        raise ValueError(
            f"No oracle_node_holdout checkpoints found in {checkpoint_dir}"
        )
    runs: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        result_path = checkpoint_path.with_suffix(".json")
        if not result_path.exists():
            raise ValueError(
                f"Checkpoint has no completed model configuration JSON: "
                f"{checkpoint_path}"
            )
        with open(result_path, encoding="utf-8") as handle:
            training_result = json.load(handle)
        run_name = checkpoint_path.stem
        if training_result.get("run_name") != run_name:
            raise ValueError(
                f"Run name mismatch between {checkpoint_path} and {result_path}"
            )
        configuration = training_result.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(f"Missing configuration in {result_path}")
        if configuration.get("evaluation_protocol") != "oracle_node_holdout":
            raise ValueError(f"Non-oracle checkpoint in oracle set: {checkpoint_path}")
        runs.append(
            {
                "run_name": run_name,
                "checkpoint_path": checkpoint_path,
                "result_path": result_path,
                "training_result": training_result,
                "configuration": configuration,
            }
        )
    return runs


def validate_checkpoint_configuration(
    *,
    run: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    cache: dict[str, Any],
    source_cache_fingerprint: str,
) -> dict[str, int]:
    """Validate checkpoint provenance, named feature blocks, and tower dimensions."""
    configuration = run["configuration"]
    training_result = run["training_result"]
    checkpoint_path = run["checkpoint_path"]
    if training_result.get("cache_fingerprint") != source_cache_fingerprint:
        raise ValueError(
            f"{checkpoint_path.name} was trained on cache fingerprint "
            f"{training_result.get('cache_fingerprint')!r}, expected derived "
            f"source fingerprint {source_cache_fingerprint!r}"
        )

    vendor_feature = configuration.get("vendor_feature")
    customer_feature = configuration.get("customer_feature")
    if vendor_feature not in cache["vendor_features"]:
        raise ValueError(
            f"{checkpoint_path.name} requests missing vendor feature "
            f"{vendor_feature!r}"
        )
    if customer_feature not in cache["customer_features"]:
        raise ValueError(
            f"{checkpoint_path.name} requests missing customer feature "
            f"{customer_feature!r}"
        )

    required_state = {
        "vendor.layers.0.weight",
        "vendor.layers.0.bias",
        "vendor.layers.3.weight",
        "vendor.layers.3.bias",
        "customer.layers.0.weight",
        "customer.layers.0.bias",
        "customer.layers.3.weight",
        "customer.layers.3.bias",
    }
    missing = required_state - set(state_dict)
    if missing:
        raise ValueError(
            f"{checkpoint_path.name} is missing model parameters: {sorted(missing)}"
        )

    vendor_input_dim = int(state_dict["vendor.layers.0.weight"].shape[1])
    customer_input_dim = int(state_dict["customer.layers.0.weight"].shape[1])
    vendor_hidden_dim = int(state_dict["vendor.layers.0.weight"].shape[0])
    customer_hidden_dim = int(state_dict["customer.layers.0.weight"].shape[0])
    vendor_output_dim = int(state_dict["vendor.layers.3.weight"].shape[0])
    customer_output_dim = int(state_dict["customer.layers.3.weight"].shape[0])
    if vendor_hidden_dim != customer_hidden_dim:
        raise ValueError(f"{checkpoint_path.name} tower hidden dimensions differ")
    if vendor_output_dim != customer_output_dim:
        raise ValueError(f"{checkpoint_path.name} tower output dimensions differ")

    cache_vendor_dim = int(cache["vendor_features"][vendor_feature].shape[1])
    cache_customer_dim = int(cache["customer_features"][customer_feature].shape[1])
    if vendor_input_dim != cache_vendor_dim:
        raise ValueError(
            f"{checkpoint_path.name} vendor input dimension {vendor_input_dim} "
            f"does not match cache {vendor_feature} dimension {cache_vendor_dim}"
        )
    if customer_input_dim != cache_customer_dim:
        raise ValueError(
            f"{checkpoint_path.name} customer input dimension {customer_input_dim} "
            f"does not match cache {customer_feature} dimension {cache_customer_dim}"
        )
    if int(configuration.get("hidden_dim", -1)) != vendor_hidden_dim:
        raise ValueError(f"{checkpoint_path.name} hidden_dim metadata is inconsistent")
    if int(configuration.get("output_dim", -1)) != vendor_output_dim:
        raise ValueError(f"{checkpoint_path.name} output_dim metadata is inconsistent")

    return {
        "vendor_input_dim": vendor_input_dim,
        "customer_input_dim": customer_input_dim,
        "hidden_dim": vendor_hidden_dim,
        "output_dim": vendor_output_dim,
    }


def _encode_tower_for_evaluation(
    tower: torch.nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    tower.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(
                device=device,
                dtype=torch.float32,
            )
            outputs.append(tower(batch).cpu())
    return torch.cat(outputs, dim=0).numpy()


def _aggregate_evaluation_runs(
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for metric in _EVALUATION_METRICS:
            values = [
                float(record["metrics"][metric])
                for record in records
                if record["metrics"].get(metric) is not None
            ]
            metrics[metric] = {
                "mean": round(float(np.mean(values)), 6) if values else None,
                "std": round(float(np.std(values)), 6) if values else None,
                "min": round(float(np.min(values)), 6) if values else None,
                "max": round(float(np.max(values)), 6) if values else None,
            }
        return {
            "n_runs": len(records),
            "seeds": sorted(
                {int(record["configuration"]["seed"]) for record in records}
            ),
            "metrics": metrics,
        }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (
            str(run["configuration"]["vendor_feature"]),
            str(run["configuration"]["customer_feature"]),
        )
        grouped.setdefault(key, []).append(run)
    return {
        "all_runs": summarize(runs),
        "by_configuration": {
            f"{vendor_feature}__{customer_feature}": {
                "vendor_feature": vendor_feature,
                "customer_feature": customer_feature,
                **summarize(records),
            }
            for (vendor_feature, customer_feature), records in sorted(
                grouped.items()
            )
        },
    }


def run_evaluate(
    *,
    cache_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    coverage = validate_evaluation_cache(cache)
    checkpoint_runs = discover_oracle_checkpoint_runs(checkpoint_dir)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device_name}")

    split = cache["split"]
    definition = SliceDefinition(
        edges=[
            (int(vendor), int(customer))
            for vendor, customer in split["test_slices"]["customer_holdout"]
        ],
        candidate_customers=list(
            map(int, split["customer_holdout_candidates"])
        ),
        excluded_pairs=set(),
    )
    known_pairs = {
        (int(vendor), int(customer))
        for vendor, customer in cache.get(
            "pairs",
            split["train_edges"] + definition.edges,
        )
    }
    evaluated_runs: list[dict[str, Any]] = []
    for run_number, run in enumerate(checkpoint_runs, start=1):
        state_dict = torch.load(
            run["checkpoint_path"],
            map_location=device,
            weights_only=True,
        )
        dimensions = validate_checkpoint_configuration(
            run=run,
            state_dict=state_dict,
            cache=cache,
            source_cache_fingerprint=coverage["source_cache_fingerprint"],
        )
        configuration = run["configuration"]
        vendor_feature = configuration["vendor_feature"]
        customer_feature = configuration["customer_feature"]
        model = InductiveTwoTower(
            vendor_dim=dimensions["vendor_input_dim"],
            customer_dim=dimensions["customer_input_dim"],
            hidden_dim=dimensions["hidden_dim"],
            output_dim=dimensions["output_dim"],
            dropout=float(configuration["dropout"]),
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        batch_size = max(1, int(configuration.get("batch_size", 2048)))
        vendor_embeddings = _encode_tower_for_evaluation(
            model.vendor,
            cache["vendor_features"][vendor_feature],
            device=device,
            batch_size=batch_size,
        )
        customer_embeddings = _encode_tower_for_evaluation(
            model.customer,
            cache["customer_features"][customer_feature],
            device=device,
            batch_size=batch_size,
        )
        raw_metrics = evaluate_slice(
            vendor_embeddings=vendor_embeddings,
            customer_embeddings=customer_embeddings,
            definition=definition,
            known_pairs=known_pairs,
            seed=int(configuration["seed"]),
        )
        metrics = {
            "map": raw_metrics["mean_average_precision"],
            "ndcg_at_50": raw_metrics["mean_ndcg_at_50"],
            "precision_at_10": raw_metrics["mean_precision_at_10"],
            "chance_precision_at_10": raw_metrics[
                "mean_chance_precision_at_10"
            ],
            "precision_at_10_lift": raw_metrics["precision_at_10_lift"],
            "recall_at_50": raw_metrics["mean_recall_at_50"],
            "profile_coverage": round(float(coverage["profile_coverage"]), 6),
        }
        evaluated_runs.append(
            {
                "run_name": run["run_name"],
                "checkpoint": str(run["checkpoint_path"]),
                "training_result": str(run["result_path"]),
                "training_cache_fingerprint": run["training_result"][
                    "cache_fingerprint"
                ],
                "configuration": {
                    "vendor_feature": vendor_feature,
                    "customer_feature": customer_feature,
                    "evaluation_protocol": "oracle_node_holdout",
                    "seed": int(configuration["seed"]),
                    "hidden_dim": dimensions["hidden_dim"],
                    "output_dim": dimensions["output_dim"],
                    "dropout": float(configuration["dropout"]),
                    "batch_size": batch_size,
                },
                "evaluation_population": {
                    "n_queries": raw_metrics["n_queries"],
                    "n_positive_edges": raw_metrics["n_positive_edges"],
                    "candidate_customers": coverage["profiled_customers"],
                    "mean_candidate_customers": raw_metrics[
                        "mean_candidate_customers"
                    ],
                },
                "metrics": metrics,
            }
        )
        del model, state_dict, vendor_embeddings, customer_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[evaluate {run_number}/{len(checkpoint_runs)}] "
            f"{run['run_name']} MAP={metrics['map']}",
            flush=True,
        )

    result = {
        "created_at": _utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "evaluation": "independent_customer_holdout",
        "cache": {
            "path": str(cache_path),
            **coverage,
        },
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "definition": {
            "fixed_vendor_queries": True,
            "candidate_population": (
                "successfully profiled customer_holdout customers only"
            ),
            "slice": "customer_holdout",
        },
        "runs": evaluated_runs,
        "aggregate": _aggregate_evaluation_runs(evaluated_runs),
    }
    _atomic_write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "runs": len(evaluated_runs),
                "profile_coverage": coverage["profile_coverage"],
            },
            indent=2,
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    research = subparsers.add_parser("research")
    research.add_argument("--output-dir", type=Path, required=True)
    research.add_argument("--max-customers", type=int, default=100)
    research.add_argument("--workers", type=int, default=8)
    research.add_argument("--seed", type=int, default=42)
    research.add_argument(
        "--selection",
        choices=["random", "high_support"],
        default="random",
    )
    research.add_argument(
        "--fresh-search",
        action="store_true",
        help="Bypass cached SERP responses.",
    )

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--output-dir", type=Path, required=True)
    resolve.add_argument("--workers", type=int, default=8)
    resolve.add_argument("--device", default="cuda:4")

    build_cache = subparsers.add_parser("build-cache")
    build_cache.add_argument("--output-dir", type=Path, required=True)
    build_cache.add_argument("--cache-out", type=Path, required=True)
    build_cache.add_argument("--source-cache", type=Path, default=DEFAULT_CACHE)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--checkpoint-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "research":
        run_research(
            output_dir=args.output_dir,
            max_customers=args.max_customers,
            workers=args.workers,
            seed=args.seed,
            selection=args.selection,
            use_cache=not args.fresh_search,
        )
    elif args.command == "resolve":
        run_resolve(
            output_dir=args.output_dir,
            workers=args.workers,
            device=args.device,
        )
    elif args.command == "build-cache":
        run_build_cache(
            output_dir=args.output_dir,
            cache_out=args.cache_out,
            source_cache_path=args.source_cache,
        )
    else:
        run_evaluate(
            cache_path=args.cache,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            device_name=args.device,
        )


if __name__ == "__main__":
    main()
