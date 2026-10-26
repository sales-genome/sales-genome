"""Recovered legacy retrieval/ranking/generation functions, not SDR-Bench/AET harness.

Callers must construct indexes and exclude target-company evidence BEFORE use.
The legacy step4 function does not itself implement the manuscript holdout.
Only the new product-aware offline adapter is used in package examples.
"""
from __future__ import annotations
import json
from typing import Any
from lib.budget import Budget
from lib.llm import tool_call as _llm_call
from lib.logging import log as _log

def log(message):
    _log("pitch-reference", message)

def step4_audience_pains(
    matched_audiences: list[dict],
    idx: dict[str, Any],
) -> list[dict]:
    log("Step 4: Collecting pain points from audience customers")

    pain_info: dict[str, dict] = {}

    for aud in matched_audiences:
        aud_id = aud["audience_id"]
        customer_slugs = idx["audience_customers"].get(aud_id, set())

        for slug in customer_slugs:
            cust_nid = f"customer:{slug}"
            cust_name = idx["customer_nodes"].get(cust_nid, {}).get("name", slug)
            pain_ids = idx["cust_pains"].get(cust_nid, set())

            for pid in pain_ids:
                if pid not in pain_info:
                    pain_info[pid] = {
                        "pain_id": pid,
                        "pain_text": idx["pain_text"].get(pid, ""),
                        "frequency": 0,
                        "experienced_by": [],
                        "via_audiences": set(),
                    }
                info = pain_info[pid]
                info["frequency"] += 1
                if cust_name not in info["experienced_by"]:
                    info["experienced_by"].append(cust_name)
                info["via_audiences"].add(aud.get("canonical_name", aud_id))

    result = []
    for pid, info in pain_info.items():
        info["via_audiences"] = sorted(info["via_audiences"])
        result.append(info)

    result.sort(key=lambda x: -x["frequency"])
    log(f"  {len(result)} unique pain points from {sum(len(idx['audience_customers'].get(a['audience_id'], set())) for a in matched_audiences)} audience customers")

    return result


def step5_vendor_pains(
    vendor_slug: str,
    idx: dict[str, Any],
) -> list[dict]:
    log(f"Step 5: Analyzing vendor '{vendor_slug}' capabilities → pains")

    vendor_nid = f"vendor:{vendor_slug}"
    cap_ids = idx["vendor_caps"].get(vendor_nid, set())

    pain_info: dict[str, dict] = {}

    for cap_id in cap_ids:
        cap_t = idx["cap_text"].get(cap_id, "")
        for addr_edge in idx["cap_pains"].get(cap_id, []):
            pid = addr_edge["dst"]
            if pid not in pain_info:
                pain_info[pid] = {
                    "pain_id": pid,
                    "pain_text": idx["pain_text"].get(pid, ""),
                    "addressed_by_capabilities": [],
                }
            pain_info[pid]["addressed_by_capabilities"].append({
                "capability_id": cap_id,
                "capability_text": cap_t,
                "source_urls": addr_edge.get("source_urls", []),
                "customer_names": addr_edge.get("customer_names", []),
                "confidence": addr_edge.get("avg_confidence", ""),
            })

    result = sorted(pain_info.values(), key=lambda x: -len(x["addressed_by_capabilities"]))
    log(f"  {len(cap_ids)} capabilities → {len(result)} addressable pains")

    return result


RANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranked_pains"],
    "properties": {
        "ranked_pains": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "score", "relevance", "reasoning"],
                "properties": {
                    "index": {"type": "integer", "description": "0-based index into the input pain list"},
                    "score": {"type": "integer", "description": "1-10 relevance score for this buyer-seller pair"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reasoning": {"type": "string", "description": "Why this pain matters for this specific buyer"},
                },
            },
        },
    },
}


def _rank_pains(
    pains: list[dict],
    profile: dict,
    entity_name: str,
    budget: Budget,
) -> list[dict]:
    """Rank pain points by relevance to a specific buyer-seller pair via LLM."""
    if not pains:
        return []

    lines = []
    for i, p in enumerate(pains):
        text = (p.get("pain_text", "") or p.get("text", "")).replace("\n", " ")
        caps = [c.get("capability_text", "") for c in p.get("addressed_by_capabilities", [])[:3]]
        caps_str = " | ".join(c[:80] for c in caps if c)
        custs = p.get("experienced_by", [])
        if not custs:
            custs = []
            for c in p.get("addressed_by_capabilities", []):
                custs.extend(c.get("customer_names", []))
        custs_str = ", ".join(custs[:5])
        lines.append(f"{i}|{text}|caps:{caps_str}|custs:{custs_str}")

    pains_block = "\n".join(lines)
    attrs = profile["attributes"]

    prompt = (
        f"You are ranking pain points for relevance to a specific buyer-seller combination.\n\n"
        f"BUYER: {profile['name']}\n"
        f"  Industry: {attrs.get('canonical_industry', 'unknown')}\n"
        f"  Revenue: {attrs.get('revenue_band', 'unknown')}\n"
        f"  Employees: {attrs.get('employee_band', 'unknown')}\n"
        f"  Region: {attrs.get('region', 'unknown')}\n"
        f"  Summary: {profile.get('summary', profile.get('description', ''))}\n\n"
        f"SELLER: {entity_name}\n\n"
        f"Below are {len(pains)} pain points that {entity_name} can address, "
        f"filtered to only pains experienced by companies similar to {profile['name']}.\n"
        f"Each line: index|pain_text|capabilities|customers_who_experienced_it\n\n"
        f"{pains_block}\n\n"
        f"Select and rank the TOP 25 most relevant pain points for selling {entity_name} "
        f"to {profile['name']} specifically. Score each 1-10.\n"
        f"Consider: Does this pain likely affect {profile['name']} given their industry, "
        f"scale, and business model? Is it a high-impact problem worth pitching?"
    )

    result = _llm_call(
        system="You are a B2B sales strategist who ranks pain points by relevance to a specific buyer.",
        user=prompt,
        tool_name="emit_ranked_pains",
        schema=RANK_SCHEMA,
        budget=budget,
        max_tokens=8192,
    )

    ranked = result.get("ranked_pains", [])
    enriched = []
    for r in ranked:
        idx_val = r.get("index", -1)
        if 0 <= idx_val < len(pains):
            item = dict(pains[idx_val])
            item["rank_score"] = r["score"]
            item["rank_relevance"] = r["relevance"]
            item["rank_reasoning"] = r["reasoning"]
            enriched.append(item)

    enriched.sort(key=lambda x: -x["rank_score"])
    return enriched


PITCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pitch_points", "competitive_summary", "unaddressed_gaps"],
    "properties": {
        "pitch_points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["headline", "pain", "capability", "reasoning",
                             "evidence", "competitor_comparison", "confidence"],
                "properties": {
                    "headline": {"type": "string"},
                    "pain": {"type": "string", "description": "The pain restated for this specific prospect"},
                    "capability": {"type": "string", "description": "The vendor capability that addresses this pain"},
                    "reasoning": {"type": "string"},
                    "evidence": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["pain_seen_in_customers", "pain_source_urls",
                                     "capability_proven_for", "capability_source_urls"],
                        "properties": {
                            "pain_seen_in_customers": {"type": "array", "items": {"type": "string"}},
                            "pain_source_urls": {"type": "array", "items": {"type": "string"}},
                            "capability_proven_for": {"type": "array", "items": {"type": "string"}},
                            "capability_source_urls": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "competitor_comparison": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
        "competitive_summary": {"type": "string"},
        "unaddressed_gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pain", "addressed_by_competitors", "reasoning"],
                "properties": {
                    "pain": {"type": "string"},
                    "addressed_by_competitors": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string", "description": "Why this gap matters for the prospect"},
                },
            },
        },
    },
}


def step8_generate_pitch(
    profile: dict,
    matched_audiences: list[dict],
    rankings: dict[str, Any],
    competitors: list[dict],
    vendor_slug: str,
    idx: dict[str, Any],
    budget: Budget,
) -> dict:
    log("Step 8: Generating pitch from ranked pains")

    vendor_name = idx["vendor_nodes"].get(f"vendor:{vendor_slug}", {}).get("name", vendor_slug)

    def _pain_to_json(p: dict) -> dict:
        caps = p.get("addressed_by_capabilities", [])
        return {
            "pain_text": p.get("pain_text", ""),
            "score": p.get("rank_score", 0),
            "relevance": p.get("rank_relevance", ""),
            "why_relevant": p.get("rank_reasoning", ""),
            "capabilities": [c.get("capability_text", "")[:100] for c in caps[:3]],
            "experienced_by": (p.get("experienced_by") or [])[:5],
            "source_urls": [u for c in caps[:2] for u in c.get("source_urls", [])[:2]],
            "customer_names": [n for c in caps[:2] for n in c.get("customer_names", [])[:3]],
        }

    vendor_ranked_json = json.dumps(
        [_pain_to_json(p) for p in rankings["vendor_ranked"]],
        indent=1,
    )

    comp_parts = []
    for comp in competitors:
        cr = rankings["competitor_ranked"].get(comp["competitor_slug"], [])
        if not cr:
            continue
        comp_parts.append(json.dumps({
            "competitor": comp["competitor_name"],
            "intensity": comp["intensity"],
            "ranked_pains": [_pain_to_json(p) for p in cr[:15]],
        }, indent=1))
    comp_text = "\n---\n".join(comp_parts) if comp_parts else "No competitor pains ranked."

    attrs = profile["attributes"]
    audiences_json = json.dumps([
        {"audience": m["canonical_name"], "relevance": m["relevance"], "reasoning": m["reasoning"]}
        for m in matched_audiences
    ], indent=1)

    prompt = f"""Generate a sales pitch for {vendor_name} targeting {profile['name']}.

## Prospect Profile
- Company: {profile['name']}
- Summary: {profile.get('summary', profile.get('description', ''))}
- Industry: {attrs['canonical_industry']}
- Revenue: {attrs['revenue_band']}  |  Employees: {attrs['employee_band']}
- Region: {attrs['region']}  |  Business Model: {attrs['business_model']}

## Matched Audiences
{audiences_json}

## {vendor_name} — Top Ranked Pain Points
(These pains are ranked by relevance to {profile['name']}. Each has a score (1-10),
the vendor capability that addresses it, and provenance from similar companies.)
{vendor_ranked_json}

## Competitor — Top Ranked Pain Points
(Pains ranked by relevance that competitors can address.)
{comp_text}

## Instructions

Create 10-15 pitch points from the vendor's top-ranked pains. For each:
- headline: compelling one-liner for {profile['name']}
- pain: restate the pain in {profile['name']}-specific terms
- capability: the vendor capability that addresses it
- reasoning: why this matters for {profile['name']} specifically
- evidence: customer_names and source_urls from the data (do NOT invent)
- competitor_comparison: which competitors also address this (from their ranked list) or don't
- confidence: high/medium/low

Also identify unaddressed_gaps: high-scoring competitor pains that {vendor_name} cannot address.

Rank pitch points by impact. Use ONLY evidence from the provided data."""

    result = _llm_call(
        system=(
            "You are an expert B2B sales strategist generating provenanced pitch points "
            "from pre-ranked pain points. Every claim must trace to provided data."
        ),
        user=prompt,
        tool_name="emit_pitch",
        schema=PITCH_SCHEMA,
        budget=budget,
        max_tokens=16384,
    )

    log(f"  {len(result.get('pitch_points', []))} pitch points, "
        f"{len(result.get('unaddressed_gaps', []))} gaps")

    return result
