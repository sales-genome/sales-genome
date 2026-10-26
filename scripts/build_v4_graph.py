#!/usr/bin/env python3
"""Build the v4 knowledge graph with an explicit product layer.

v4 inserts a `product` node between vendors and capabilities:

    vendor -[offers]-> product -[provides]-> capability -[addresses]-> pain_point

Reads (from v3, shared source of truth):
  graph/v3/extractions.jsonl
  graph/v3/crosslinks.json
  graph/poc/competitor_edges.jsonl

Reads (caches, falling back to v3):
  graph/v3/industry_canonicalization.json
  graph/v3/attribute_normalization.json

Writes:
  graph/v4/nodes.jsonl
  graph/v4/edges.jsonl
  graph/v4/summary.json
  graph/v4/product_normalization.json  (cached LLM product canonicalization)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import llm as _llm
from lib.budget import Budget
from lib.logging import log as _log

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "graph" / "v3"
V4 = ROOT / "graph" / "v4"
EXTRACTIONS = V3 / "extractions.jsonl"
CROSSLINKS = V3 / "crosslinks.json"

INDUSTRY_CACHE = V3 / "industry_canonicalization.json"
ATTR_CACHE = V3 / "attribute_normalization.json"
PRODUCT_CACHE = V4 / "product_normalization.json"
SPEND_FILE = V4 / ".graph_spend.json"

MIN_CAP_LEN = 5
MIN_PAIN_LEN = 10


def log(msg: str, **kw):
    _log("v4-graph", msg, **kw)


def sid(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def h8(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def load_jsonl(path: Path) -> list[dict]:
    items = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# LLM helper — delegates to lib.llm.tool_call, the shared throttle-aware
# dispatcher that already routes to the local vLLM provider when
# LLM_BACKEND=local (and to Azure otherwise). This script used to keep its own
# private Azure-only client here, which meant it silently ignored
# LLM_BACKEND=local and always hit Azure Foundry.
# ---------------------------------------------------------------------------

def _llm_call(*, system: str, user: str, tool_name: str, schema: dict,
              budget: Budget, max_tokens: int = 4096) -> dict:
    return _llm.tool_call(
        system=system, user=user, tool_name=tool_name, schema=schema,
        budget=budget, max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Phase 0: Load inputs
# ---------------------------------------------------------------------------

def phase0_load() -> tuple[list[dict], dict]:
    log("Loading extractions and crosslinks")
    extractions = load_jsonl(EXTRACTIONS)
    log(f"  {len(extractions)} extractions loaded")

    crosslinks_data: dict = {}
    if CROSSLINKS.exists():
        with open(CROSSLINKS) as f:
            crosslinks_data = json.load(f)
    cl_list = crosslinks_data.get("crosslinks", [])

    name_to_slug: dict[str, str] = {}
    for cl in cl_list:
        slug = cl.get("customer_slug", "")
        if not slug:
            continue
        for field in ("customer_name", "customer_name_matched", "raw_customer_name"):
            name = cl.get(field, "")
            if name:
                name_to_slug[name.lower().strip()] = slug
    log(f"  {len(name_to_slug)} crosslink name mappings, {len(cl_list)} crosslinks")

    return extractions, name_to_slug


# ---------------------------------------------------------------------------
# Phase 1: Vendor nodes
# ---------------------------------------------------------------------------

def phase1_vendors(extractions: list[dict]) -> dict[str, dict]:
    log("Phase 1: Building vendor nodes")
    vendor_stats: dict[str, dict] = {}
    for ext in extractions:
        v = ext.get("vendor_slug", "")
        if not v:
            continue
        if v not in vendor_stats:
            vendor_stats[v] = {"cs": 0, "pp": 0}
        if ext.get("page_type") == "case_study":
            vendor_stats[v]["cs"] += 1
        elif ext.get("page_type") == "product":
            vendor_stats[v]["pp"] += 1

    nodes = {}
    for slug, st in sorted(vendor_stats.items()):
        nid = f"vendor:{slug}"
        nodes[nid] = {
            "id": nid, "type": "vendor", "slug": slug,
            "name": slug.replace("-", " ").title(),
            "n_case_studies": st["cs"], "n_product_pages": st["pp"],
        }
    log(f"  {len(nodes)} vendor nodes")
    return nodes


# ---------------------------------------------------------------------------
# Phase 2: Capability + pain nodes (single pass)
# ---------------------------------------------------------------------------

def phase2_cap_pain(extractions: list[dict]) -> tuple[dict[str, dict], dict[str, dict], dict, dict]:
    log("Phase 2: Building capability + pain nodes (single pass)")

    cap_reg: dict[tuple[str, str], dict] = {}
    pain_reg: dict[str, dict] = {}

    for ext in extractions:
        v = ext.get("vendor_slug", "")
        d = ext.get("data", {})
        url = ext.get("url", "")
        customer = d.get("customer_name", "")
        products = d.get("vendor_products_used") or []

        cap_texts: set[str] = set()
        for c in (d.get("capabilities_cited") or []):
            if isinstance(c, str) and len(c.strip()) >= MIN_CAP_LEN:
                cap_texts.add(c.strip())
        for pair in (d.get("vp_pp_pairs") or []):
            if not isinstance(pair, dict):
                continue
            cap = pair.get("capability", "")
            if isinstance(cap, str) and len(cap.strip()) >= MIN_CAP_LEN:
                cap_texts.add(cap.strip())

        for ct in cap_texts:
            key = (v, ct.lower())
            if key not in cap_reg:
                cap_reg[key] = {
                    "text": ct, "frequency": 0,
                    "customers": [], "urls": [], "products": [],
                }
            rec = cap_reg[key]
            rec["frequency"] += 1
            if customer and customer not in rec["customers"]:
                rec["customers"].append(customer)
            if url and url not in rec["urls"]:
                rec["urls"].append(url)
            for p in products:
                if p and p not in rec["products"]:
                    rec["products"].append(p)

        pain_texts: set[str] = set()
        for p in (d.get("pain_points_cited") or []):
            if isinstance(p, str) and len(p.strip()) >= MIN_PAIN_LEN:
                pain_texts.add(p.strip())
        for pair in (d.get("vp_pp_pairs") or []):
            if not isinstance(pair, dict):
                continue
            pt = pair.get("pain_point", "")
            if isinstance(pt, str) and len(pt.strip()) >= MIN_PAIN_LEN:
                pain_texts.add(pt.strip())

        for pt in pain_texts:
            key = pt.lower()
            if key not in pain_reg:
                pain_reg[key] = {
                    "text": pt, "frequency": 0,
                    "customers": [], "vendor_slugs": [], "urls": [],
                }
            rec = pain_reg[key]
            rec["frequency"] += 1
            if customer and customer not in rec["customers"]:
                rec["customers"].append(customer)
            if v and v not in rec["vendor_slugs"]:
                rec["vendor_slugs"].append(v)
            if url and url not in rec["urls"]:
                rec["urls"].append(url)

    cap_nodes: dict[str, dict] = {}
    cap_lookup: dict[tuple[str, str], str] = {}

    for (vendor, lowered), rec in cap_reg.items():
        nid = f"cap:{vendor}:{h8(lowered)}"
        cap_nodes[nid] = {
            "id": nid, "type": "capability", "vendor_slug": vendor,
            "text": rec["text"], "frequency": rec["frequency"],
            "n_customers": len(rec["customers"]),
            "customers": rec["customers"][:10],
            "products_associated": rec["products"][:10],
        }
        cap_lookup[(vendor, lowered)] = nid

    pain_nodes: dict[str, dict] = {}
    pain_lookup: dict[str, str] = {}

    for lowered, rec in pain_reg.items():
        nid = f"pain:{h8(lowered)}"
        pain_nodes[nid] = {
            "id": nid, "type": "pain_point",
            "text": rec["text"], "frequency": rec["frequency"],
            "n_customers": len(rec["customers"]),
            "vendor_slugs": rec["vendor_slugs"],
        }
        pain_lookup[lowered] = nid

    log(f"  {len(cap_nodes)} capability nodes across {len(set(k[0] for k in cap_reg))} vendors")
    log(f"  {len(pain_nodes)} pain_point nodes")

    return cap_nodes, pain_nodes, cap_lookup, pain_lookup


# ---------------------------------------------------------------------------
# Phase 2B: Product nodes (LLM normalization of vendor_products_used)
# ---------------------------------------------------------------------------

PRODUCT_LINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["product_lines"],
    "properties": {
        "product_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical_name", "description", "raw_variants"],
                "properties": {
                    "canonical_name": {"type": "string"},
                    "description": {"type": "string"},
                    "raw_variants": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

PRODUCT_ASSIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assignments", "new_product_lines"],
    "properties": {
        "assignments": {
            "type": "array",
            "description": "Map each raw product name to an existing canonical product line.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["raw_name", "canonical_name"],
                "properties": {
                    "raw_name": {"type": "string"},
                    "canonical_name": {
                        "type": "string",
                        "description": "Must be one of the existing canonical names, or '_new' if proposing a new line.",
                    },
                },
            },
        },
        "new_product_lines": {
            "type": "array",
            "description": "New product lines not covered by existing ones. Only add if truly distinct.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical_name", "description", "raw_variants"],
                "properties": {
                    "canonical_name": {"type": "string"},
                    "description": {"type": "string"},
                    "raw_variants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Raw names from this batch that belong to this new line.",
                    },
                },
            },
        },
    },
}


def _normalize_products_for_vendor(
    vendor_name: str,
    sorted_prods: list[tuple[str, set[str]]],
    budget: Budget,
) -> list[dict]:
    """Two-phase LLM normalization of raw product names into canonical lines.

    Phase A: Send top ~60 products to establish initial canonical lines (up to 20).
    Phase B: Batch remaining products ~80 at a time, assigning to existing lines
             or proposing new ones.

    Returns list of product line dicts: [{canonical_name, description, raw_variants}, ...]
    """
    total_raw = len(sorted_prods)

    # --- Phase A: initial canonical lines from top 60 ---
    phase_a_size = min(60, total_raw)
    phase_a_prods = sorted_prods[:phase_a_size]

    prod_text = "\n".join(
        f"- {p} ({len(custs)} customers)"
        for p, custs in phase_a_prods
    )
    result = _llm_call(
        system=(
            "You normalize raw product names from case studies into canonical "
            "product lines. Group variants of the same product together."
        ),
        user=(
            f"Vendor: {vendor_name}\n\n"
            f"Raw product names (with customer count):\n{prod_text}\n\n"
            "Normalize into up to 20 canonical product lines. Large vendors may "
            "genuinely have 15-20 distinct product families — capture all of them."
        ),
        tool_name="emit_product_lines",
        schema=PRODUCT_LINE_SCHEMA,
        budget=budget,
        max_tokens=8192,
    )
    lines = result.get("product_lines", [])
    log(f"    Phase A: {len(lines)} initial lines from top {phase_a_size} products")

    if phase_a_size >= total_raw:
        return lines

    # --- Phase B: assign remaining products in batches ---
    remaining_prods = sorted_prods[phase_a_size:]
    phase_b_batch_size = 80
    n_batches = (len(remaining_prods) + phase_b_batch_size - 1) // phase_b_batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * phase_b_batch_size
        end = min(start + phase_b_batch_size, len(remaining_prods))
        batch = remaining_prods[start:end]

        # Build the current canonical lines summary for the LLM
        canon_summary = "\n".join(
            f"- {pl['canonical_name']}: {pl.get('description', '')}"
            for pl in lines
        )

        batch_text = "\n".join(
            f"- {p} ({len(custs)} customers)"
            for p, custs in batch
        )

        result = _llm_call(
            system=(
                "You assign raw product names to existing canonical product lines, "
                "or propose new product lines if a raw name does not fit any existing line."
            ),
            user=(
                f"Vendor: {vendor_name}\n\n"
                f"Existing canonical product lines:\n{canon_summary}\n\n"
                f"Raw product names to assign (with customer count):\n{batch_text}\n\n"
                "For each raw name, assign it to the best-matching existing canonical "
                "product line. If a raw product genuinely represents a distinct product "
                "family not covered by any existing line, propose it as a new product line. "
                "Use '_new' as the canonical_name in assignments for products that belong "
                "to a new line, and define the new line in new_product_lines."
            ),
            tool_name="emit_product_assignments",
            schema=PRODUCT_ASSIGN_SCHEMA,
            budget=budget,
            max_tokens=8192,
        )

        assignments = result.get("assignments", [])
        new_lines = result.get("new_product_lines", [])

        # Index existing lines by canonical_name for fast lookup
        lines_by_name: dict[str, dict] = {pl["canonical_name"].lower(): pl for pl in lines}

        # Process assignments: add raw names to existing lines
        n_assigned = 0
        for a in assignments:
            if not isinstance(a, dict):
                continue
            raw = a.get("raw_name", "")
            canon = a.get("canonical_name", "")
            if not raw or not canon or canon == "_new":
                continue
            pl = lines_by_name.get(canon.lower())
            if pl and raw not in pl["raw_variants"]:
                pl["raw_variants"].append(raw)
                n_assigned += 1

        # Merge new product lines
        for npl in new_lines:
            if not isinstance(npl, dict):
                continue
            new_name = npl.get("canonical_name", "")
            if not new_name:
                continue
            # Check if it accidentally duplicates an existing line
            if new_name.lower() in lines_by_name:
                existing = lines_by_name[new_name.lower()]
                for rv in npl.get("raw_variants", []):
                    if rv not in existing["raw_variants"]:
                        existing["raw_variants"].append(rv)
            else:
                lines.append(npl)
                lines_by_name[new_name.lower()] = npl

        log(f"    Phase B batch {batch_idx + 1}/{n_batches}: "
            f"assigned {n_assigned}, {len(new_lines)} new lines")

    return lines


def phase2b_products(
    extractions: list[dict],
    cap_nodes: dict[str, dict],
    vendor_nodes: dict[str, dict],
    budget: Budget,
    skip_llm: bool = False,
    refresh: bool = False,
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Normalize raw product names into canonical product lines per vendor.

    Returns:
        product_nodes: product_id -> node dict
        cap_to_products: cap_id -> [product_id, ...] mapping for edge construction
    """
    log("Phase 2B: Building product nodes via LLM normalization")

    # Collect raw product names + customer counts per vendor
    vendor_product_customers: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for ext in extractions:
        v = ext.get("vendor_slug", "")
        if not v:
            continue
        d = ext.get("data", {})
        customer = d.get("customer_name", "")
        for prod in d.get("vendor_products_used") or []:
            if prod and prod.strip():
                vendor_product_customers[v][prod.strip()].add(customer if customer else "_anon")

    # Check cache
    cached_normalization: dict[str, list[dict]] = {}
    if PRODUCT_CACHE.exists() and not refresh:
        with open(PRODUCT_CACHE) as f:
            cached_normalization = json.load(f)
        log(f"  Loaded cached product normalization for {len(cached_normalization)} vendors")

    product_nodes: dict[str, dict] = {}
    product_lookup: dict[tuple[str, str], str] = {}  # (vendor, raw_name_lower) -> product_id
    all_normalization: dict[str, list[dict]] = {}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _norm_lock = threading.Lock()
    _save_counter = 0

    def _normalize_one_vendor(vendor_slug: str) -> tuple[str, list[dict]]:
        """Normalize products for one vendor. Thread-safe — only touches local state + LLM calls."""
        nonlocal _save_counter
        v = vendor_slug.replace("vendor:", "")
        vpc = vendor_product_customers.get(v, {})
        vendor_name = vendor_nodes[vendor_slug]["name"]

        if not vpc:
            log(f"  {v}: no products found, creating _unspecified only")
            return v, []

        sorted_prods = sorted(vpc.items(), key=lambda x: len(x[1]), reverse=True)

        if v in cached_normalization and not refresh:
            return v, cached_normalization[v]
        elif skip_llm:
            return v, [{"canonical_name": p, "description": "", "raw_variants": [p]}
                       for p, _ in sorted_prods]
        else:
            log(f"  {v}: {len(sorted_prods)} raw products to normalize")
            try:
                lines = _normalize_products_for_vendor(vendor_name, sorted_prods, budget)
            except Exception as e:  # noqa: BLE001 - one vendor's LLM failure (e.g.
                # timeout) must not crash a multi-hour build; treat as empty so the
                # poison guard below skips caching it and it retries on resume.
                log(f"  {v}: normalization FAILED ({e!r}) - treating as empty, will retry on resume")
                lines = []
            log(f"  {v}: LLM returned {len(lines)} product lines")
            # Only cache a SUCCESSFUL (non-empty) normalization. This vendor has raw
            # products, so an empty result means the LLM call failed (e.g. server
            # crash); caching it would poison the cache and be skipped on resume.
            if lines:
                with _norm_lock:
                    all_normalization[v] = lines
                    _save_counter += 1
                    if _save_counter % 5 == 0:
                        PRODUCT_CACHE.write_text(json.dumps(all_normalization, indent=2, default=list))
            else:
                log(f"  {v}: EMPTY result — NOT caching (likely LLM/server failure); will retry on resume")
            return v, lines

    # Parallelize across vendors (LLM_CONCURRENCY from settings, default 5)
    vendors_to_process = sorted(vendor_nodes.keys())
    n_cached = sum(1 for vs in vendors_to_process
                   if vs.replace("vendor:", "") in cached_normalization and not refresh)
    n_llm = len(vendors_to_process) - n_cached
    log(f"  {len(vendors_to_process)} vendors ({n_cached} cached, {n_llm} need LLM)")

    concurrency = min(8, max(4, len(vendors_to_process) // 4))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_normalize_one_vendor, vs): vs for vs in vendors_to_process}
        for fut in as_completed(futures):
            v, lines = fut.result()
            vpc = vendor_product_customers.get(v, {})
            # Mirror the inner poison guard: only persist an empty result when the
            # vendor genuinely has no raw products (`not vpc`). An empty result for
            # a vendor that DOES have products means the LLM call failed; leaving
            # it out of all_normalization means it's absent from the final cache
            # dump below, so a re-run of this script retries it instead of
            # treating "LLM failed" as "vendor has zero products" forever.
            if lines or not vpc:
                all_normalization[v] = lines

            if not vpc:
                unspec_id = f"product:{v}:_unspecified"
                product_nodes[unspec_id] = {
                    "id": unspec_id, "type": "product", "vendor_slug": v,
                    "name": "(Unspecified)", "raw_variants": [],
                    "n_customers": 0, "customer_slugs": [],
                }
                continue

            # Build product nodes + lookup for this vendor
            assigned_raws: set[str] = set()
            for pl in lines:
                canon = pl["canonical_name"]
                pid = f"product:{v}:{h8(canon.lower())}"
                raw_variants = pl.get("raw_variants", [])

                custs: set[str] = set()
                for rv in raw_variants:
                    custs.update(vpc.get(rv, set()))
                    product_lookup[(v, rv.lower().strip())] = pid
                    assigned_raws.add(rv.lower().strip())

                custs.discard("_anon")

                product_nodes[pid] = {
                    "id": pid, "type": "product", "vendor_slug": v,
                    "name": canon,
                    "description": pl.get("description", ""),
                    "raw_variants": raw_variants,
                    "n_customers": len(custs),
                    "customer_slugs": sorted(custs)[:20],
                }

            # Map unassigned raw products to _unspecified
            unspec_id = f"product:{v}:_unspecified"
            unassigned_custs: set[str] = set()
            unassigned_raws: list[str] = []
            for raw_prod, custs in vpc.items():
                if raw_prod.lower().strip() not in assigned_raws:
                    product_lookup[(v, raw_prod.lower().strip())] = unspec_id
                    unassigned_custs.update(custs)
                    unassigned_raws.append(raw_prod)

            unassigned_custs.discard("_anon")
            product_nodes[unspec_id] = {
                "id": unspec_id, "type": "product", "vendor_slug": v,
                "name": "(Unspecified)",
                "raw_variants": unassigned_raws[:50],
                "n_customers": len(unassigned_custs),
                "customer_slugs": sorted(unassigned_custs)[:20],
            }

            total_prods = len(lines) + 1
            n_assigned = len(assigned_raws)
            n_unassigned = len(unassigned_raws)
            log(f"  {v}: {total_prods} product nodes ({len(lines)} canonical + _unspecified), "
                f"{n_assigned}/{n_assigned + n_unassigned} raw names assigned")

    # Save final cache
    PRODUCT_CACHE.write_text(json.dumps(all_normalization, indent=2, default=list))
    log(f"  Cached to {PRODUCT_CACHE.name}")

    # Build cap_to_products: which product(s) each capability belongs to
    cap_to_products: dict[str, list[str]] = {}
    for cap_id, cap_node in cap_nodes.items():
        vendor = cap_node["vendor_slug"]
        raw_prods = cap_node.get("products_associated", [])

        pids: list[str] = []
        for rp in raw_prods:
            pid = product_lookup.get((vendor, rp.lower().strip()))
            if pid and pid not in pids:
                pids.append(pid)

        if not pids:
            unspec = f"product:{vendor}:_unspecified"
            if unspec in product_nodes:
                pids = [unspec]

        cap_to_products[cap_id] = pids

    n_unspec = sum(1 for pids in cap_to_products.values()
                   if pids and all("_unspecified" in p for p in pids))
    log(f"  {len(product_nodes)} total product nodes")
    log(f"  {len(cap_to_products)} capabilities mapped ({n_unspec} to _unspecified)")

    return product_nodes, cap_to_products


# ---------------------------------------------------------------------------
# Phase 3: Customer nodes
# ---------------------------------------------------------------------------

def phase3_customers(
    extractions: list[dict],
    name_to_slug: dict[str, str],
) -> tuple[dict[str, dict], dict[str, str], dict[str, list[str]], dict[str, list[dict]]]:
    log("Phase 3: Building customer nodes")

    customer_data: dict[str, dict] = {}
    raw_name_to_slug: dict[str, str] = {}
    customer_raw_dims: dict[str, list[dict]] = {}

    for ext in extractions:
        if ext.get("page_type") != "case_study":
            continue
        d = ext.get("data", {})
        name = d.get("customer_name", "")
        if not name or name.lower().strip() in ("n/a", "none", ""):
            continue

        lowered = name.lower().strip()
        slug = name_to_slug.get(lowered, sid(name))
        raw_name_to_slug[lowered] = slug

        if slug not in customer_data:
            customer_data[slug] = {
                "raw_names": [], "industries": [],
                "n_case_studies": 0, "vendors": [],
            }
        cd = customer_data[slug]
        cd["n_case_studies"] += 1
        if name not in cd["raw_names"]:
            cd["raw_names"].append(name)
        v = ext.get("vendor_slug", "")
        if v and v not in cd["vendors"]:
            cd["vendors"].append(v)

        ad = d.get("audience_dimensions") or {}
        if not isinstance(ad, dict):
            ad = {}
        ind = ad.get("industry", "")
        if ind and ind.lower().strip() not in ("none", "n/a", ""):
            cd["industries"].append(ind.strip())
        if ad:
            customer_raw_dims.setdefault(slug, []).append(ad)

    nodes: dict[str, dict] = {}
    customer_industries: dict[str, list[str]] = {}

    crosslink_count = 0
    for slug, cd in sorted(customer_data.items()):
        nid = f"customer:{slug}"
        display_name = Counter(cd["raw_names"]).most_common(1)[0][0]
        is_crosslink = any(name_to_slug.get(n.lower().strip()) == slug for n in cd["raw_names"])
        if is_crosslink:
            crosslink_count += 1

        nodes[nid] = {
            "id": nid, "type": "customer", "slug": slug,
            "name": display_name,
            "raw_names": cd["raw_names"][:5],
            "n_case_studies": cd["n_case_studies"],
            "vendors": cd["vendors"],
            "crosslink_resolved": is_crosslink,
        }
        customer_industries[slug] = cd["industries"]

    log(f"  {len(nodes)} customer nodes ({crosslink_count} crosslink-resolved)")
    return nodes, raw_name_to_slug, customer_industries, customer_raw_dims


# ---------------------------------------------------------------------------
# Phase 4: Audience nodes (LLM industry canonicalization)
# Uses cached results from v3 — same extractions produce identical output.
# ---------------------------------------------------------------------------

INDUSTRY_TAXONOMY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["industries"],
    "properties": {
        "industries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical_name", "description"],
                "properties": {
                    "canonical_name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    },
}

INDUSTRY_ASSIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assignments"],
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["raw", "canonical"],
                "properties": {
                    "raw": {"type": "string"},
                    "canonical": {"type": "string"},
                },
            },
        },
    },
}


def phase4_audiences(
    customer_industries: dict[str, list[str]],
    customer_nodes: dict[str, dict],
    budget: Budget,
    skip_llm: bool = False,
    refresh: bool = False,
) -> dict[str, dict]:
    log("Phase 4: Building audience nodes via industry canonicalization")

    all_industries: list[str] = []
    for inds in customer_industries.values():
        all_industries.extend(inds)
    counts = Counter(i.lower().strip() for i in all_industries if i.strip())
    unique = sorted(counts.keys())
    log(f"  {len(all_industries)} mentions, {len(unique)} unique industries")

    raw_to_canonical: dict[str, str] = {}
    taxonomy: list[dict] = []

    if INDUSTRY_CACHE.exists() and not refresh:
        with open(INDUSTRY_CACHE) as f:
            cached = json.load(f)
        taxonomy = cached.get("taxonomy", [])
        raw_to_canonical = cached.get("mapping", {})
        log(f"  Loaded cached canonicalization: {len(taxonomy)} canonical, {len(raw_to_canonical)} mapped")
    elif skip_llm:
        log("  --skip-llm: using raw industries as audience nodes")
        for raw in unique:
            raw_to_canonical[raw] = raw
    else:
        top_industries = [ind for ind, _ in counts.most_common(90) if counts[ind] >= 5]
        log(f"  LLM taxonomy call: {len(top_industries)} top industries")

        taxonomy_prompt = (
            "Below are the most common B2B industry labels observed in case studies.\n"
            "Group them into 30-50 canonical industry segments. Each segment should be:\n"
            "- Broad enough to cover related sub-industries\n"
            "- Specific enough to be useful for sales targeting\n"
            "- Named clearly (e.g., 'Financial Services', 'Healthcare & Life Sciences')\n\n"
            "Industries:\n" + "\n".join(f"- {i} ({counts[i]}x)" for i in top_industries)
        )
        tax_result = _llm_call(
            system="You are a B2B industry classification expert.",
            user=taxonomy_prompt,
            tool_name="emit_taxonomy",
            schema=INDUSTRY_TAXONOMY_SCHEMA,
            budget=budget,
        )
        taxonomy = tax_result.get("industries", [])
        canonical_names = [t["canonical_name"] for t in taxonomy]
        log(f"  Taxonomy: {len(taxonomy)} canonical industries")

        batch_size = 50
        canonical_list = "\n".join(f"- {c}" for c in canonical_names)

        for i in range(0, len(unique), batch_size):
            batch = unique[i : i + batch_size]
            assign_prompt = (
                "Map each raw industry string to the single best canonical industry from the list below.\n"
                "If none fits well, pick the closest match.\n\n"
                f"Canonical industries:\n{canonical_list}\n\n"
                "Raw strings to classify:\n" +
                "\n".join(f"- {r}" for r in batch)
            )
            result = _llm_call(
                system="You are a B2B industry classification expert.",
                user=assign_prompt,
                tool_name="emit_assignments",
                schema=INDUSTRY_ASSIGN_SCHEMA,
                budget=budget,
            )
            for a in result.get("assignments", []):
                raw = a.get("raw", "").lower().strip()
                canon = a.get("canonical", "")
                if raw and canon:
                    raw_to_canonical[raw] = canon
            log(f"  Assigned batch {i // batch_size + 1}/{(len(unique) + batch_size - 1) // batch_size}")

        INDUSTRY_CACHE.write_text(json.dumps({
            "taxonomy": taxonomy,
            "mapping": raw_to_canonical,
        }, indent=2))
        log(f"  Cached to {INDUSTRY_CACHE.name}")

    audience_customer_count: dict[str, int] = Counter()
    for slug, inds in customer_industries.items():
        if not inds:
            continue
        canon_inds = []
        for i in inds:
            canon = raw_to_canonical.get(i.lower().strip())
            if canon:
                canon_inds.append(canon)
        if canon_inds:
            modal = Counter(canon_inds).most_common(1)[0][0]
            customer_nodes[f"customer:{slug}"]["canonical_industry"] = modal
            audience_customer_count[modal] += 1

    aud_nodes: dict[str, dict] = {}
    aud_raw_counts: dict[str, int] = Counter()
    for raw, canon in raw_to_canonical.items():
        aud_raw_counts[canon] += 1

    for canon, n_raw in aud_raw_counts.items():
        aid = f"audience:{sid(canon)}"
        desc = ""
        for t in taxonomy:
            if t["canonical_name"] == canon:
                desc = t.get("description", "")
                break
        aud_nodes[aid] = {
            "id": aid, "type": "audience",
            "canonical_name": canon, "description": desc,
            "n_customers": audience_customer_count.get(canon, 0),
            "n_raw_variants": n_raw,
        }

    log(f"  {len(aud_nodes)} audience nodes, {sum(audience_customer_count.values())} customers assigned")
    return aud_nodes


# ---------------------------------------------------------------------------
# Phase 4B: Attribute nodes from audience_dimensions (LLM normalization)
# Uses cached results from v3.
# ---------------------------------------------------------------------------

ATTR_DIMS = {
    "revenue_band": ["<$100M", "$100M-$1B", "$1B-$10B", "$10B+"],
    "employee_band": ["<100", "100-1K", "1K-10K", "10K-100K", "100K+"],
    "region": ["North America", "Europe", "Asia-Pacific", "Latin America",
               "Middle East & Africa", "Global"],
    "business_model": ["B2B", "B2C", "B2B2C", "Marketplace", "Platform"],
}

ATTR_NORMALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["customers"],
    "properties": {
        "customers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["customer_name", "revenue_band", "employee_band",
                             "region", "business_model"],
                "properties": {
                    "customer_name": {"type": "string"},
                    "revenue_band": {
                        "type": "string",
                        "enum": ATTR_DIMS["revenue_band"] + ["unknown"],
                    },
                    "employee_band": {
                        "type": "string",
                        "enum": ATTR_DIMS["employee_band"] + ["unknown"],
                    },
                    "region": {
                        "type": "string",
                        "enum": ATTR_DIMS["region"] + ["unknown"],
                    },
                    "business_model": {
                        "type": "string",
                        "enum": ATTR_DIMS["business_model"] + ["unknown"],
                    },
                },
            },
        },
    },
}


def _merge_raw_dims(dims_list: list[dict]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for field in ["company_size", "location", "revenue", "other"]:
        vals: list[str] = []
        for ad in dims_list:
            v = ad.get(field)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v if x)
            if v and str(v).lower().strip() not in ("none", "n/a", "", "null",
                                                      "not specified", "not mentioned"):
                vals.append(str(v).strip())
        unique = list(dict.fromkeys(vals))[:5]
        merged[field] = "; ".join(unique) if unique else ""
    return merged


def phase4b_attributes(
    customer_raw_dims: dict[str, list[dict]],
    customer_nodes: dict[str, dict],
    budget: Budget,
    skip_llm: bool = False,
    refresh: bool = False,
) -> tuple[dict[str, dict], dict[str, dict]]:
    log("Phase 4B: Building attribute nodes from audience_dimensions")

    if ATTR_CACHE.exists() and not refresh:
        with open(ATTR_CACHE) as f:
            cached = json.load(f)
        attr_nodes = cached.get("nodes", {})
        cust_attrs = cached.get("customer_attributes", {})
        log(f"  Loaded cached: {len(attr_nodes)} attribute nodes, {len(cust_attrs)} customer assignments")
        return attr_nodes, cust_attrs

    if skip_llm:
        log("  --skip-llm: skipping attribute normalization")
        return {}, {}

    slug_merged: dict[str, dict] = {}
    slug_name: dict[str, str] = {}

    for slug, dims_list in customer_raw_dims.items():
        cnode = customer_nodes.get(f"customer:{slug}")
        if not cnode:
            continue
        merged = _merge_raw_dims(dims_list)
        has_data = any(v for v in merged.values())
        if not has_data:
            continue
        slug_merged[slug] = merged
        slug_name[slug] = cnode["name"]

    log(f"  {len(slug_merged)} customers with non-empty audience dimensions")

    slugs = sorted(slug_merged.keys())
    batch_size = 30
    cust_attrs: dict[str, dict] = {}

    for i in range(0, len(slugs), batch_size):
        batch = slugs[i : i + batch_size]
        batch_text_parts = []
        for s in batch:
            m = slug_merged[s]
            name = slug_name.get(s, s)
            parts = [f"Customer: {name}"]
            if m.get("company_size"):
                parts.append(f"  Size: {m['company_size']}")
            if m.get("location"):
                parts.append(f"  Location: {m['location']}")
            if m.get("revenue"):
                parts.append(f"  Revenue: {m['revenue']}")
            if m.get("other"):
                parts.append(f"  Other: {m['other'][:200]}")
            batch_text_parts.append("\n".join(parts))

        prompt = (
            "Classify each customer into standardized buckets based ONLY on the "
            "information provided below. If a dimension cannot be determined from "
            "the provided data, output 'unknown'. Do NOT use outside knowledge.\n\n"
            "Buckets:\n"
            + "\n".join(f"- {k}: {', '.join(v)}, unknown" for k, v in ATTR_DIMS.items())
            + "\n\nCustomers:\n\n" + "\n\n".join(batch_text_parts)
        )

        result = _llm_call(
            system="You classify companies into attribute buckets using ONLY the "
                   "information provided. If a field is not evident from the data, "
                   "output 'unknown'.",
            user=prompt,
            tool_name="emit_attributes",
            schema=ATTR_NORMALIZE_SCHEMA,
            budget=budget,
            max_tokens=8192,
        )

        name_to_result: dict[str, dict] = {}
        for c in result.get("customers", []):
            name_to_result[c["customer_name"].lower().strip()] = c

        for s in batch:
            name = slug_name.get(s, s)
            r = name_to_result.get(name.lower().strip())
            if not r:
                for key, val in name_to_result.items():
                    if sid(key) == s:
                        r = val
                        break

            cust_attrs[s] = {
                "revenue_band": r["revenue_band"] if r else "unknown",
                "employee_band": r["employee_band"] if r else "unknown",
                "region": r["region"] if r else "unknown",
                "business_model": r["business_model"] if r else "unknown",
            }

        batch_num = i // batch_size + 1
        total_batches = (len(slugs) + batch_size - 1) // batch_size
        log(f"  Batch {batch_num}/{total_batches}")

    attr_nodes: dict[str, dict] = {}
    for dim, buckets in ATTR_DIMS.items():
        for val in buckets:
            nid = f"audience:{dim}:{sid(val)}"
            attr_nodes[nid] = {
                "id": nid, "type": "audience",
                "dimension": dim, "value": val,
                "canonical_name": f"{dim}:{val}",
                "n_customers": 0,
            }

    for slug, attrs in cust_attrs.items():
        for dim, val in attrs.items():
            if val and val != "unknown":
                nid = f"audience:{dim}:{sid(val)}"
                if nid in attr_nodes:
                    attr_nodes[nid]["n_customers"] += 1

    ATTR_CACHE.write_text(json.dumps({
        "nodes": attr_nodes,
        "customer_attributes": cust_attrs,
    }, indent=2))

    for dim in ATTR_DIMS:
        known = sum(1 for a in cust_attrs.values() if a.get(dim, "unknown") != "unknown")
        log(f"  {dim}: {known}/{len(cust_attrs)} known ({100*known/len(cust_attrs):.0f}%)")

    log(f"  {len(attr_nodes)} attribute nodes, {len(cust_attrs)} customer assignments")
    return attr_nodes, cust_attrs


# ---------------------------------------------------------------------------
# Phase 5: Build edges (v4: vendor→product→capability instead of vendor→capability)
# ---------------------------------------------------------------------------

def phase5_edges(
    extractions: list[dict],
    vendor_nodes: dict[str, dict],
    product_nodes: dict[str, dict],
    cap_nodes: dict[str, dict],
    pain_nodes: dict[str, dict],
    customer_nodes: dict[str, dict],
    aud_nodes: dict[str, dict],
    attr_nodes: dict[str, dict],
    cust_attrs: dict[str, dict],
    cap_lookup: dict[tuple[str, str], str],
    pain_lookup: dict[str, str],
    raw_name_to_slug: dict[str, str],
    cap_to_products: dict[str, list[str]],
) -> list[dict]:
    log("Phase 5: Building edges")
    edges: list[dict] = []

    # 5a: offers (vendor → product)
    for pid, pnode in product_nodes.items():
        vendor_nid = f"vendor:{pnode['vendor_slug']}"
        edges.append({"src": vendor_nid, "dst": pid, "type": "offers"})
    log(f"  5a: {len(edges)} offers edges (vendor → product)")

    # 5a2: provides (product → capability)
    provides_count = 0
    for cap_id, pids in cap_to_products.items():
        for pid in pids:
            edges.append({"src": pid, "dst": cap_id, "type": "provides"})
            provides_count += 1
    log(f"  5a2: {provides_count} provides edges (product → capability)")

    # 5b: addresses (capability → pain_point) from vp_pp_pairs
    addr_agg: dict[tuple[str, str], dict] = {}
    addr_miss = 0

    for ext in extractions:
        v = ext.get("vendor_slug", "")
        d = ext.get("data", {})
        url = ext.get("url", "")
        customer = d.get("customer_name", "")

        for pair in (d.get("vp_pp_pairs") or []):
            if not isinstance(pair, dict):
                continue
            cap_text = pair.get("capability", "").strip()
            pain_text = pair.get("pain_point", "").strip()
            if not cap_text or not pain_text:
                continue

            cap_id = cap_lookup.get((v, cap_text.lower()))
            pain_id = pain_lookup.get(pain_text.lower())

            if not cap_id or not pain_id:
                addr_miss += 1
                continue

            key = (cap_id, pain_id)
            if key not in addr_agg:
                addr_agg[key] = {
                    "weight": 0, "source_urls": [],
                    "customer_names": [], "confidences": [],
                }
            agg = addr_agg[key]
            agg["weight"] += 1
            if url and url not in agg["source_urls"]:
                agg["source_urls"].append(url)
            if customer and customer not in agg["customer_names"]:
                agg["customer_names"].append(customer)
            conf = pair.get("confidence", "")
            if conf:
                agg["confidences"].append(conf)

    for (cap_id, pain_id), agg in addr_agg.items():
        conf_counts = Counter(agg["confidences"])
        edges.append({
            "src": cap_id, "dst": pain_id, "type": "addresses",
            "weight": agg["weight"],
            "source_urls": agg["source_urls"][:10],
            "customer_names": agg["customer_names"][:10],
            "n_customers": len(agg["customer_names"]),
            "avg_confidence": conf_counts.most_common(1)[0][0] if conf_counts else "",
        })
    log(f"  5b: {len(addr_agg)} addresses edges ({addr_miss} pairs skipped)")

    # 5c: experiences (customer → pain_point)
    exp_data: dict[tuple[str, str], dict] = {}

    for ext in extractions:
        if ext.get("page_type") != "case_study":
            continue
        v = ext.get("vendor_slug", "")
        d = ext.get("data", {})
        customer = d.get("customer_name", "")
        if not customer or customer.lower().strip() in ("n/a", "none"):
            continue

        slug = raw_name_to_slug.get(customer.lower().strip())
        if not slug:
            continue
        cust_nid = f"customer:{slug}"
        if cust_nid not in customer_nodes:
            continue

        pain_texts: set[str] = set()
        for p in (d.get("pain_points_cited") or []):
            if isinstance(p, str) and len(p.strip()) >= MIN_PAIN_LEN:
                pain_texts.add(p.strip())
        for pair in (d.get("vp_pp_pairs") or []):
            if not isinstance(pair, dict):
                continue
            pt = pair.get("pain_point", "")
            if isinstance(pt, str) and len(pt.strip()) >= MIN_PAIN_LEN:
                pain_texts.add(pt.strip())

        for pt in pain_texts:
            pain_id = pain_lookup.get(pt.lower())
            if not pain_id:
                continue
            key = (cust_nid, pain_id)
            if key not in exp_data:
                exp_data[key] = {"n_sources": 0, "vendor_slugs": []}
            exp_data[key]["n_sources"] += 1
            if v and v not in exp_data[key]["vendor_slugs"]:
                exp_data[key]["vendor_slugs"].append(v)

    for (cust_nid, pain_id), ed in exp_data.items():
        edges.append({
            "src": cust_nid, "dst": pain_id, "type": "experiences",
            "n_sources": ed["n_sources"],
            "vendor_slugs": ed["vendor_slugs"],
        })
    log(f"  5c: {len(exp_data)} experiences edges")

    # 5d: in_audience (customer → audience)
    aud_count = 0
    for nid, cnode in customer_nodes.items():
        canon_ind = cnode.get("canonical_industry")
        if not canon_ind:
            continue
        aud_nid = f"audience:{sid(canon_ind)}"
        if aud_nid in aud_nodes:
            edges.append({"src": nid, "dst": aud_nid, "type": "in_audience"})
            aud_count += 1
    log(f"  5d: {aud_count} in_audience edges")

    # 5e: has_case_study (vendor → customer)
    cs_pairs: dict[tuple[str, str], dict] = {}
    for ext in extractions:
        if ext.get("page_type") != "case_study":
            continue
        v = ext.get("vendor_slug", "")
        d = ext.get("data", {})
        customer = d.get("customer_name", "")
        if not customer or customer.lower().strip() in ("n/a", "none"):
            continue
        slug = raw_name_to_slug.get(customer.lower().strip())
        if not slug:
            continue

        key = (v, slug)
        if key not in cs_pairs:
            cs_pairs[key] = {"n": 0, "urls": []}
        cs_pairs[key]["n"] += 1
        url = ext.get("url", "")
        if url and url not in cs_pairs[key]["urls"]:
            cs_pairs[key]["urls"].append(url)

    for (v, c_slug), csd in cs_pairs.items():
        edges.append({
            "src": f"vendor:{v}", "dst": f"customer:{c_slug}",
            "type": "has_case_study",
            "n_case_studies": csd["n"],
            "source_urls": csd["urls"][:10],
        })
    log(f"  5e: {len(cs_pairs)} has_case_study edges")

    # 5f: in_audience for non-industry attributes (customer → audience:dim:val)
    attr_aud_count = 0
    for slug, attrs in cust_attrs.items():
        cust_nid = f"customer:{slug}"
        if cust_nid not in customer_nodes:
            continue
        for dim, val in attrs.items():
            if val and val != "unknown":
                attr_nid = f"audience:{dim}:{sid(val)}"
                if attr_nid in attr_nodes:
                    edges.append({
                        "src": cust_nid, "dst": attr_nid,
                        "type": "in_audience",
                        "dimension": dim,
                    })
                    attr_aud_count += 1
    log(f"  5f: {attr_aud_count} in_audience (attribute) edges")

    # 5g: competes_with (vendor ↔ vendor) from POC competitor edges
    poc_comp_path = V3.parent / "poc" / "competitor_edges.jsonl"
    compete_count = 0
    if poc_comp_path.exists():
        vendor_nids = set(vendor_nodes.keys())
        for line in poc_comp_path.read_text().splitlines():
            if not line.strip():
                continue
            ce = json.loads(line)
            if ce.get("src_role") != "vendor":
                continue
            src_nid = f"vendor:{ce['src_slug']}"
            dst_nid = f"vendor:{ce['dst_slug']}"
            if src_nid in vendor_nids and dst_nid in vendor_nids:
                edges.append({
                    "src": src_nid, "dst": dst_nid,
                    "type": "competes_with",
                    "intensity": ce.get("intensity", ""),
                    "confidence": ce.get("confidence", ""),
                    "rationale": ce.get("rationale", ""),
                    "basis": ce.get("basis", ""),
                    "derived_via": ce.get("derived_via", ""),
                })
                compete_count += 1
    log(f"  5g: {compete_count} competes_with edges (from POC competitor analysis)")

    log(f"  Total: {len(edges)} edges")
    return edges


# ---------------------------------------------------------------------------
# Phase 6: Write outputs
# ---------------------------------------------------------------------------

def phase6_write(
    vendor_nodes: dict[str, dict],
    product_nodes: dict[str, dict],
    cap_nodes: dict[str, dict],
    pain_nodes: dict[str, dict],
    customer_nodes: dict[str, dict],
    aud_nodes: dict[str, dict],
    attr_nodes: dict[str, dict],
    edges: list[dict],
):
    log("Phase 6: Writing outputs")

    all_nodes = {}
    all_nodes.update(vendor_nodes)
    all_nodes.update(product_nodes)
    all_nodes.update(cap_nodes)
    all_nodes.update(pain_nodes)
    all_nodes.update(customer_nodes)
    all_nodes.update(aud_nodes)
    all_nodes.update(attr_nodes)

    nodes_path = V4 / "nodes.jsonl"
    edges_path = V4 / "edges.jsonl"
    summary_path = V4 / "summary.json"

    with open(nodes_path, "w") as f:
        for n in all_nodes.values():
            f.write(json.dumps(n) + "\n")
    with open(edges_path, "w") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")

    node_types = Counter(n["type"] for n in all_nodes.values())
    edge_types = Counter(e["type"] for e in edges)

    summary = {
        "version": "v4",
        "n_nodes": len(all_nodes),
        "n_edges": len(edges),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "description": "v4 graph adds product layer: vendor -[offers]-> product -[provides]-> capability",
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    log(f"Wrote {nodes_path.name} ({len(all_nodes)} nodes)")
    log(f"Wrote {edges_path.name} ({len(edges)} edges)")
    log(f"Wrote {summary_path.name}")
    log(f"Node types: {dict(node_types)}")
    log(f"Edge types: {dict(edge_types)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global V3, V4, EXTRACTIONS, CROSSLINKS, INDUSTRY_CACHE, ATTR_CACHE, PRODUCT_CACHE, SPEND_FILE
    parser = argparse.ArgumentParser(description="Build v4 knowledge graph (with product layer)")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Extraction/crosslink/cache directory; never modified")
    parser.add_argument("--out", type=Path, required=True, help="New package-local output directory")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM calls (use raw product names, skip uncached canonicalization)")
    parser.add_argument("--refresh-products", action="store_true",
                        help="Re-run product normalization LLM calls")
    parser.add_argument("--refresh-industries", action="store_true",
                        help="Re-run industry canonicalization (writes to v3 cache)")
    parser.add_argument("--refresh-attributes", action="store_true",
                        help="Re-run attribute normalization (writes to v3 cache)")
    args = parser.parse_args()

    V3, V4 = args.input_dir.resolve(), args.out.resolve()
    if V3 == V4 or V3 in V4.parents:
        raise ValueError("Keep outputs separate from extraction inputs.")
    if args.refresh_industries or args.refresh_attributes:
        raise ValueError("Submission copy does not overwrite input normalization caches.")
    EXTRACTIONS, CROSSLINKS = V3 / "extractions.jsonl", V3 / "crosslinks.json"
    INDUSTRY_CACHE, ATTR_CACHE = V4 / "industry_canonicalization.json", V4 / "attribute_normalization.json"
    PRODUCT_CACHE, SPEND_FILE = V4 / "product_normalization.json", V4 / ".local-usage.json"
    if not EXTRACTIONS.is_file():
        raise FileNotFoundError(EXTRACTIONS)
    V4.mkdir(parents=True, exist_ok=False)
    for filename in ["industry_canonicalization.json", "attribute_normalization.json", "product_normalization.json"]:
        if (V3 / filename).is_file():
            shutil.copyfile(V3 / filename, V4 / filename)

    budget = Budget.load(SPEND_FILE)
    log(f"Budget: ${budget.total_usd:.2f} spent so far")

    extractions, name_to_slug = phase0_load()
    vendor_nodes = phase1_vendors(extractions)
    cap_nodes, pain_nodes, cap_lookup, pain_lookup = phase2_cap_pain(extractions)
    product_nodes, cap_to_products = phase2b_products(
        extractions, cap_nodes, vendor_nodes, budget,
        skip_llm=args.skip_llm, refresh=args.refresh_products,
    )
    customer_nodes, raw_name_to_slug, customer_industries, customer_raw_dims = phase3_customers(
        extractions, name_to_slug,
    )
    aud_nodes = phase4_audiences(
        customer_industries, customer_nodes, budget,
        skip_llm=args.skip_llm, refresh=args.refresh_industries,
    )
    attr_nodes, cust_attrs = phase4b_attributes(
        customer_raw_dims, customer_nodes, budget,
        skip_llm=args.skip_llm, refresh=args.refresh_attributes,
    )
    edges = phase5_edges(
        extractions, vendor_nodes, product_nodes, cap_nodes, pain_nodes,
        customer_nodes, aud_nodes, attr_nodes, cust_attrs,
        cap_lookup, pain_lookup, raw_name_to_slug, cap_to_products,
    )
    phase6_write(
        vendor_nodes, product_nodes, cap_nodes, pain_nodes,
        customer_nodes, aud_nodes, attr_nodes, edges,
    )
    log(f"Done. Budget: ${budget.total_usd:.2f} total")


if __name__ == "__main__":
    main()
