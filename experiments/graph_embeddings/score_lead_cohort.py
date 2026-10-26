#!/usr/bin/env python3
"""Generate frozen GNN score sidecars for lead JSONL cohorts.

The scorer never modifies lead_result.json. Unknown prospects require cached
account enrichment containing either canonical pain IDs or free-text pains.
Audience attributes are resolved conservatively by exact canonical aliases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.graph_embeddings.lead_prior import (
    DEFAULT_VARIANT,
    GraphPriorUnavailable,
    LeadGraphPrior,
)
from experiments.graph_embeddings.prospect_features import (
    exact_key,
    load_enrichment_cache,
)


DEFAULT_ENRICHMENTS = (
    ROOT
    / "experiments"
    / "graph_embeddings"
    / "results"
    / "crm_enrichments.jsonl"
)


def load_leads(paths: list[Path]) -> list[dict[str, Any]]:
    leads: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    leads.append(json.loads(line))
    return leads


def lead_key(record: dict[str, Any]) -> tuple[str, str, str]:
    product = record.get("product") or "all"
    return (
        exact_key(record.get("account_name", "")),
        record.get("vendor_slug", ""),
        exact_key(product),
    )


def _values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int, float)):
        return (value,)
    return tuple(value)


def profile_inputs(
    lead: dict[str, Any],
    enrichment: dict[str, Any] | None,
    prior: LeadGraphPrior,
    *,
    pain_threshold: float,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    enrichment = enrichment or {}

    pain_ids = (
        enrichment.get("pain_ids")
        or enrichment.get("matched_pain_ids")
        or lead.get("pain_ids")
        or ()
    )
    pain_ids = tuple(sorted({str(value) for value in _values(pain_ids) if value}))
    profile_error = None

    if not pain_ids:
        pain_texts = (
            enrichment.get("extracted_pains")
            or enrichment.get("pain_points")
            or lead.get("pain_points")
            or ()
        )
        pain_texts = tuple(str(value) for value in _values(pain_texts) if value)
        if pain_texts:
            try:
                pain_ids = prior.match_pain_texts(
                    pain_texts,
                    threshold=pain_threshold,
                )
            except RuntimeError as exc:
                profile_error = str(exc)

    audience_ids = {
        str(value)
        for value in _values(
            enrichment.get("audience_ids") or lead.get("audience_ids") or ()
        )
        if value
    }
    audience_values: list[str] = []
    for source in (lead, enrichment):
        industry = source.get("industry")
        if industry and industry != "unknown":
            audience_values.append(str(industry))
        attributes = source.get("attributes") or source.get("audience_attributes") or {}
        if isinstance(attributes, dict):
            for value in attributes.values():
                if isinstance(value, list):
                    audience_values.extend(str(item) for item in value if item)
                elif value:
                    audience_values.append(str(value))

    audience_ids.update(prior.resolve_audience_ids(audience_values))
    return pain_ids, tuple(sorted(audience_ids)), profile_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enrichments", type=Path, default=DEFAULT_ENRICHMENTS)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pain-threshold", type=float, default=0.45)
    parser.add_argument("--top-k-lookalikes", type=int, default=20)
    parser.add_argument("--product-weight", type=float, default=0.5)
    parser.add_argument("--competitor-prior-strength", type=float, default=25.0)
    args = parser.parse_args()

    leads = load_leads(args.leads)
    if args.limit > 0:
        leads = leads[: args.limit]
    enrichments = load_enrichment_cache(args.enrichments)

    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if args.resume and args.output.exists():
        with open(args.output) as handle:
            for line in handle:
                record = json.loads(line)
                existing[lead_key(record)] = record

    prior = LeadGraphPrior(variant=args.variant)
    try:
        prior.load()
    except GraphPriorUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    output_records = dict(existing)
    for index, lead in enumerate(leads, 1):
        key = lead_key(lead)
        if key in output_records:
            continue

        account_name = lead["account_name"]
        enrichment = enrichments.get(exact_key(account_name))
        pain_ids, audience_ids, profile_error = profile_inputs(
            lead,
            enrichment,
            prior,
            pain_threshold=args.pain_threshold,
        )
        input_product = lead.get("product")
        product_name = (
            None
            if not input_product or exact_key(str(input_product)) == "all"
            else str(input_product)
        )
        result = prior.score_lead(
            account_name=account_name,
            vendor_slug=lead["vendor_slug"],
            product_name=product_name,
            pain_ids=pain_ids,
            audience_ids=audience_ids,
            explicit_competitors=_values(lead.get("competitor_slugs")),
            top_k_lookalikes=args.top_k_lookalikes,
            product_weight=args.product_weight,
            competitor_prior_strength=args.competitor_prior_strength,
        )
        result["product"] = input_product
        result["profile_inputs"] = {
            "n_pain_ids": len(pain_ids),
            "n_audience_ids": len(audience_ids),
            "enrichment_found": enrichment is not None,
            "error": profile_error,
        }
        output_records[key] = result
        print(
            f"[{index}/{len(leads)}] {account_name}: "
            f"{result['status']} ({len(result.get('scores', {}))} scores)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w") as handle:
        for record in output_records.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()
