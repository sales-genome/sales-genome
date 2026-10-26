#!/usr/bin/env python3
"""Build v5 canonical-first knowledge graph.

Merges the v4 base graph with the pain/capability taxonomy overlay to produce
a graph where canonical pains and capabilities are the primary nodes, and raw
atom texts are preserved as evidence on edges.

Key differences from v4:
  - 198K raw pain nodes → 4,717 pain_canonical nodes
  - 307K raw capability nodes → 5,269 cap_canonical nodes
  - Taxonomy hierarchy (domain → sub_theme → canonical) is navigable
  - customer -[experiences]→ pain_canonical edges carry the customer's own
    phrasing, the vendor that documented it, and the case study URL
  - product -[provides]→ cap_canonical edges carry raw capability texts
  - cap_canonical -[addresses]→ pain_canonical edges carry provenance

No LLM calls — pure Python merge of existing v4 + overlay data.

    .venv-exp/bin/python scripts/build_v5_graph.py
    .venv-exp/bin/python scripts/build_v5_graph.py --out graph/v5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "graph" / "v4"
PAIN_OVERLAY = V4 / "taxonomy" / "pain" / "all97" / "overlay"
CAP_OVERLAY = V4 / "taxonomy" / "capability" / "all97" / "overlay"

ALIAS = {
    "adobe-dme": "adobe", "adobe-cc": "adobe", "adobe-dx": "adobe",
    "marketo-engage-now-adobe-braze": "braze",
}

EVIDENCE_CAP_EXPERIENCES = 10
EVIDENCE_CAP_PROVIDES = 5
URLS_CAP = 15
CUSTOMERS_CAP_ADDR = 20
VENDORS_CAP_ADDR = 15


def a(slug: str) -> str:
    return ALIAS.get(slug, slug)


def log(msg: str):
    print(f"  {msg}", file=sys.stderr)


def jl_stream(p: Path):
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Phase 0: Load taxonomy overlay mappings
# ---------------------------------------------------------------------------

def phase0():
    """Load instance_of mappings + canonical/structural nodes from both overlays."""
    log("Phase 0: Loading taxonomy overlay mappings …")

    pain_to_canon: dict[str, str] = {}
    cap_to_canon: dict[str, str] = {}
    canon_pain_nodes: dict[str, dict] = {}
    canon_cap_nodes: dict[str, dict] = {}
    structural_nodes: list[dict] = []
    structural_edges: list[dict] = []

    # Pain overlay
    for n in jl_stream(PAIN_OVERLAY / "overlay_nodes.jsonl"):
        t = n["type"]
        if t == "pain_canonical":
            canon_pain_nodes[n["id"]] = n
        elif t in ("pain_domain", "pain_subtheme"):
            structural_nodes.append(n)

    for e in jl_stream(PAIN_OVERLAY / "overlay_edges.jsonl"):
        if e["type"] == "instance_of":
            pain_to_canon[e["src"]] = e["dst"]
        elif e["type"] in ("in_subtheme", "in_domain"):
            structural_edges.append(e)

    # Capability overlay
    for n in jl_stream(CAP_OVERLAY / "overlay_nodes.jsonl"):
        t = n["type"]
        if t == "cap_canonical":
            canon_cap_nodes[n["id"]] = n
        elif t in ("cap_domain", "cap_subtheme"):
            structural_nodes.append(n)

    for e in jl_stream(CAP_OVERLAY / "overlay_edges.jsonl"):
        if e["type"] == "instance_of":
            cap_to_canon[e["src"]] = e["dst"]
        elif e["type"] in ("in_subtheme", "in_domain"):
            structural_edges.append(e)

    log(f"  pain_to_canon: {len(pain_to_canon):,} mappings → "
        f"{len(canon_pain_nodes):,} canonicals")
    log(f"  cap_to_canon:  {len(cap_to_canon):,} mappings → "
        f"{len(canon_cap_nodes):,} canonicals")
    log(f"  structural: {len(structural_nodes)} nodes, {len(structural_edges)} edges")

    return (pain_to_canon, cap_to_canon, canon_pain_nodes, canon_cap_nodes,
            structural_nodes, structural_edges)


# ---------------------------------------------------------------------------
# Phase 1: Stream v4 nodes — keep non-atoms, collect atom texts
# ---------------------------------------------------------------------------

def phase1(pain_to_canon, cap_to_canon):
    """Returns v5_nodes (non-atom), pain_texts, cap_texts, cap_vendors,
    customer_name_to_slug."""
    log("Phase 1: Streaming v4 nodes …")

    v5_nodes: list[dict] = []
    pain_texts: dict[str, str] = {}
    cap_texts: dict[str, str] = {}
    cap_vendors: dict[str, str] = {}
    customer_name_to_slug: dict[str, str] = {}

    for n in jl_stream(V4 / "nodes.jsonl"):
        t = n.get("type")
        nid = n.get("id", "")

        if t in ("vendor", "product", "audience"):
            if t == "vendor":
                n["vendor_slug"] = a(n.get("vendor_slug", n.get("slug", "")))
            elif t == "product":
                n["vendor_slug"] = a(n.get("vendor_slug", ""))
            v5_nodes.append(n)

        elif t == "customer":
            v5_nodes.append(n)
            slug = nid
            name = n.get("name", "")
            if name:
                customer_name_to_slug[name.lower().strip()] = slug
            for rn in n.get("raw_names", []):
                if rn:
                    customer_name_to_slug[rn.lower().strip()] = slug
            bare = slug.replace("customer:", "")
            if bare:
                customer_name_to_slug[bare.lower()] = slug

        elif t == "pain_point" and nid in pain_to_canon:
            pain_texts[nid] = n.get("text", "")

        elif t == "capability" and nid in cap_to_canon:
            cap_texts[nid] = n.get("text", "")
            vs = n.get("vendor_slug", "")
            if not vs and ":" in nid:
                vs = nid.split(":")[1]
            cap_vendors[nid] = a(vs)

    log(f"  v5 nodes (non-atom): {len(v5_nodes):,}")
    log(f"  pain texts:  {len(pain_texts):,}")
    log(f"  cap texts:   {len(cap_texts):,}")
    log(f"  customer name→slug: {len(customer_name_to_slug):,}")

    return v5_nodes, pain_texts, cap_texts, cap_vendors, customer_name_to_slug


# ---------------------------------------------------------------------------
# Phase 2: Stream v4 edges — group by canonical, build attribution indexes
# ---------------------------------------------------------------------------

def phase2(pain_to_canon, cap_to_canon, customer_name_to_slug):
    """Returns grouped canonical edges + direct edges + attribution indexes."""
    log("Phase 2: Streaming v4 edges …")

    # Direct edges (pass through unchanged)
    direct_edges: list[dict] = []

    # Grouped canonical edges: key → list of evidence items
    # provides: (product_id, cap_canonical) → [raw_cap_id, ...]
    provides_grp: dict[tuple, list] = defaultdict(list)
    # addresses: (cap_canonical, pain_canonical) → [edge_data, ...]
    addresses_grp: dict[tuple, list] = defaultdict(list)
    # experiences: (customer_id, pain_canonical) → [(raw_pain_id, vendor_slugs), ...]
    experiences_grp: dict[tuple, list] = defaultdict(list)

    # Attribution: (raw_pain_id, vendor) → set(urls)
    pain_vendor_urls: dict[tuple, set] = defaultdict(set)

    # Stats
    counts = defaultdict(int)
    unmapped_pain = 0
    unmapped_cap = 0

    for e in jl_stream(V4 / "edges.jsonl"):
        t = e.get("type")
        counts[t] += 1

        if t == "offers":
            # vendor → product: keep as-is, alias vendor
            e["src"] = e["src"].replace(
                e["src"].split(":")[1],
                a(e["src"].split(":")[1])
            ) if ":" in e["src"] else e["src"]
            direct_edges.append(e)

        elif t == "has_case_study":
            src_parts = e["src"].split(":")
            if len(src_parts) >= 2:
                e["src"] = f"vendor:{a(src_parts[1])}"
            direct_edges.append(e)

        elif t == "competes_with":
            src_parts = e["src"].split(":")
            dst_parts = e["dst"].split(":")
            if len(src_parts) >= 2:
                e["src"] = f"vendor:{a(src_parts[1])}"
            if len(dst_parts) >= 2:
                e["dst"] = f"vendor:{a(dst_parts[1])}"
            direct_edges.append(e)

        elif t == "in_audience":
            direct_edges.append(e)

        elif t == "provides":
            cap_id = e["dst"]
            cc = cap_to_canon.get(cap_id)
            if cc:
                provides_grp[(e["src"], cc)].append(cap_id)
            else:
                unmapped_cap += 1

        elif t == "addresses":
            cap_id = e["src"]
            pain_id = e["dst"]
            cc = cap_to_canon.get(cap_id)
            pc = pain_to_canon.get(pain_id)
            if cc and pc:
                addresses_grp[(cc, pc)].append(e)
                # Build URL attribution index
                vendor = a(cap_id.split(":")[1]) if ":" in cap_id else ""
                urls = e.get("source_urls") or []
                if vendor and urls:
                    pain_vendor_urls[(pain_id, vendor)].update(urls[:3])
            else:
                if not cc:
                    unmapped_cap += 1
                if not pc:
                    unmapped_pain += 1

        elif t == "experiences":
            pain_id = e["dst"]
            pc = pain_to_canon.get(pain_id)
            if pc:
                vendor_slugs = [a(v) for v in (e.get("vendor_slugs") or [])]
                experiences_grp[(e["src"], pc)].append(
                    (pain_id, vendor_slugs)
                )
            else:
                unmapped_pain += 1

    log(f"  Edge counts: {dict(counts)}")
    log(f"  Unmapped: {unmapped_pain} pain refs, {unmapped_cap} cap refs")
    log(f"  provides groups:    {len(provides_grp):,}")
    log(f"  addresses groups:   {len(addresses_grp):,}")
    log(f"  experiences groups: {len(experiences_grp):,}")
    log(f"  pain_vendor_urls:   {len(pain_vendor_urls):,}")
    log(f"  direct edges:       {len(direct_edges):,}")

    return (direct_edges, provides_grp, addresses_grp, experiences_grp,
            pain_vendor_urls)


# ---------------------------------------------------------------------------
# Phase 3: Build v5 canonical edges with evidence
# ---------------------------------------------------------------------------

def build_provides_edges(provides_grp, cap_texts):
    """product → cap_canonical with raw capability text evidence."""
    edges = []
    for (product_id, cap_canon), raw_cap_ids in provides_grp.items():
        seen_texts = set()
        evidence = []
        for cid in raw_cap_ids:
            txt = cap_texts.get(cid, "")
            if txt and txt.lower() not in seen_texts:
                seen_texts.add(txt.lower())
                evidence.append({"text": txt[:300], "atom_id": cid})
                if len(evidence) >= EVIDENCE_CAP_PROVIDES:
                    break

        edges.append({
            "src": product_id,
            "dst": cap_canon,
            "type": "provides",
            "weight": len(raw_cap_ids),
            "n_variants": len(seen_texts),
            "evidence": evidence,
        })
    return edges


def build_addresses_edges(addresses_grp, cap_vendors, customer_name_to_slug):
    """cap_canonical → pain_canonical with aggregated provenance."""
    edges = []
    for (cap_canon, pain_canon), raw_edges in addresses_grp.items():
        all_vendors = set()
        all_customers = set()
        all_urls = set()
        confidences = []

        for e in raw_edges:
            cap_id = e.get("src", "")
            vendor = a(cap_id.split(":")[1]) if ":" in cap_id else ""
            if vendor:
                all_vendors.add(vendor)

            for cn in (e.get("customer_names") or []):
                slug = customer_name_to_slug.get(cn.lower().strip())
                if slug:
                    all_customers.add(slug)

            for u in (e.get("source_urls") or []):
                all_urls.add(u)

            conf = e.get("avg_confidence")
            if conf:
                confidences.append(conf)

        # Most common confidence
        avg_conf = "medium"
        if confidences:
            from collections import Counter
            avg_conf = Counter(confidences).most_common(1)[0][0]

        edges.append({
            "src": cap_canon,
            "dst": pain_canon,
            "type": "addresses",
            "weight": len(raw_edges),
            "n_vendors": len(all_vendors),
            "vendors": sorted(all_vendors)[:VENDORS_CAP_ADDR],
            "n_customers": len(all_customers),
            "customers": sorted(all_customers)[:CUSTOMERS_CAP_ADDR],
            "source_urls": sorted(all_urls)[:URLS_CAP],
            "avg_confidence": avg_conf,
        })
    return edges


def build_experiences_edges(experiences_grp, pain_texts, pain_vendor_urls):
    """customer → pain_canonical with raw text evidence + vendor attribution."""
    edges = []
    for (customer_id, pain_canon), items in experiences_grp.items():
        evidence = []
        seen_texts = set()
        all_vendors = set()

        for raw_pain_id, vendor_slugs in items:
            txt = pain_texts.get(raw_pain_id, "")
            txt_key = txt.lower().strip()

            for vendor in vendor_slugs:
                all_vendors.add(vendor)
                urls = pain_vendor_urls.get((raw_pain_id, vendor), set())
                url = sorted(urls)[0] if urls else ""

                if txt_key not in seen_texts and len(evidence) < EVIDENCE_CAP_EXPERIENCES:
                    seen_texts.add(txt_key)
                    evidence.append({
                        "text": txt[:300],
                        "vendor": vendor,
                        "url": url,
                        "atom_id": raw_pain_id,
                    })

            # If no vendor_slugs (shouldn't happen, but be safe)
            if not vendor_slugs and txt_key not in seen_texts and len(evidence) < EVIDENCE_CAP_EXPERIENCES:
                seen_texts.add(txt_key)
                evidence.append({
                    "text": txt[:300],
                    "vendor": None,
                    "url": "",
                    "atom_id": raw_pain_id,
                })

        # Total weight = number of raw pain atoms (not deduplicated by text)
        total_raw = sum(len(vs) or 1 for _, vs in items)

        edges.append({
            "src": customer_id,
            "dst": pain_canon,
            "type": "experiences",
            "weight": len(items),
            "n_sources": total_raw,
            "vendors": sorted(all_vendors),
            "evidence": evidence,
        })
    return edges


# ---------------------------------------------------------------------------
# Phase 4: Write v5 graph
# ---------------------------------------------------------------------------

def write_graph(out_dir: Path, v5_nodes, canon_pain_nodes, canon_cap_nodes,
                structural_nodes, structural_edges, direct_edges,
                provides_edges, addresses_edges, experiences_edges):
    log("Phase 4: Writing v5 graph …")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nodes
    all_nodes = []
    all_nodes.extend(v5_nodes)
    all_nodes.extend(canon_pain_nodes.values())
    all_nodes.extend(canon_cap_nodes.values())
    all_nodes.extend(structural_nodes)

    with (out_dir / "nodes.jsonl").open("w") as f:
        for n in all_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    # Edges
    all_edges = []
    all_edges.extend(direct_edges)
    all_edges.extend(structural_edges)
    all_edges.extend(provides_edges)
    all_edges.extend(addresses_edges)
    all_edges.extend(experiences_edges)

    with (out_dir / "edges.jsonl").open("w") as f:
        for e in all_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Count by type
    node_types = defaultdict(int)
    for n in all_nodes:
        node_types[n.get("type", "unknown")] += 1

    edge_types = defaultdict(int)
    for e in all_edges:
        edge_types[e.get("type", "unknown")] += 1

    summary = {
        "version": "v5",
        "description": "Canonical-first knowledge graph with evidence-attributed edges",
        "n_nodes": len(all_nodes),
        "n_edges": len(all_edges),
        "node_types": dict(sorted(node_types.items())),
        "edge_types": dict(sorted(edge_types.items())),
        "sources": {
            "v4_graph": str(V4),
            "pain_overlay": str(PAIN_OVERLAY),
            "cap_overlay": str(CAP_OVERLAY),
        },
        "evidence_caps": {
            "experiences": EVIDENCE_CAP_EXPERIENCES,
            "provides": EVIDENCE_CAP_PROVIDES,
            "urls_per_addresses": URLS_CAP,
            "customers_per_addresses": CUSTOMERS_CAP_ADDR,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


# ---------------------------------------------------------------------------
# Phase 5: Verification
# ---------------------------------------------------------------------------

def verify(out_dir, summary):
    log("Phase 5: Verification …")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"v5 graph written to {out_dir}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    nodes_mb = (out_dir / "nodes.jsonl").stat().st_size / 1024 / 1024
    edges_mb = (out_dir / "edges.jsonl").stat().st_size / 1024 / 1024
    print(f"\nFiles:", file=sys.stderr)
    print(f"  nodes.jsonl: {nodes_mb:.1f} MB", file=sys.stderr)
    print(f"  edges.jsonl: {edges_mb:.1f} MB", file=sys.stderr)

    print(f"\nNode types ({summary['n_nodes']:,} total):", file=sys.stderr)
    for t, c in sorted(summary["node_types"].items(), key=lambda x: -x[1]):
        print(f"  {t:25s} {c:>8,}", file=sys.stderr)

    print(f"\nEdge types ({summary['n_edges']:,} total):", file=sys.stderr)
    for t, c in sorted(summary["edge_types"].items(), key=lambda x: -x[1]):
        print(f"  {t:25s} {c:>8,}", file=sys.stderr)

    # Spot-check: sample an experiences edge
    print(f"\n--- Sample experiences edge ---", file=sys.stderr)
    with (out_dir / "edges.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "experiences" and len(e.get("evidence", [])) >= 2:
                print(json.dumps(e, indent=2, ensure_ascii=False)[:600], file=sys.stderr)
                break

    # Spot-check: sample an addresses edge
    print(f"\n--- Sample addresses edge ---", file=sys.stderr)
    with (out_dir / "edges.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "addresses" and e.get("n_vendors", 0) >= 2:
                print(json.dumps(e, indent=2, ensure_ascii=False)[:600], file=sys.stderr)
                break

    # Spot-check: sample a provides edge
    print(f"\n--- Sample provides edge ---", file=sys.stderr)
    with (out_dir / "edges.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "provides" and len(e.get("evidence", [])) >= 2:
                print(json.dumps(e, indent=2, ensure_ascii=False)[:600], file=sys.stderr)
                break

    # Compression ratio vs v4
    v4_nodes = sum(1 for _ in jl_stream(V4 / "nodes.jsonl"))
    v4_edges = sum(1 for _ in jl_stream(V4 / "edges.jsonl"))
    print(f"\n--- Compression vs v4 ---", file=sys.stderr)
    print(f"  Nodes: {v4_nodes:>10,} → {summary['n_nodes']:>10,}  "
          f"({summary['n_nodes']/v4_nodes*100:.1f}%)", file=sys.stderr)
    print(f"  Edges: {v4_edges:>10,} → {summary['n_edges']:>10,}  "
          f"({summary['n_edges']/v4_edges*100:.1f}%)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global V4, PAIN_OVERLAY, CAP_OVERLAY
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="Input raw graph directory")
    ap.add_argument("--out", type=Path, required=True, help="New canonical output directory")
    ap.add_argument("--pain-overlay", default=str(PAIN_OVERLAY),
                    help="Pain taxonomy overlay dir (default: v4 all97 — override for a v6 build)")
    ap.add_argument("--cap-overlay", default=str(CAP_OVERLAY),
                    help="Capability taxonomy overlay dir (default: v4 all97)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    V4 = args.base
    PAIN_OVERLAY = Path(args.pain_overlay)
    CAP_OVERLAY = Path(args.cap_overlay)
    out_dir = Path(args.out)
    if out_dir.exists():
        raise FileExistsError(out_dir)

    t0 = time.time()
    print(f"Building v5 canonical-first graph → {out_dir}", file=sys.stderr)

    # Phase 0
    (pain_to_canon, cap_to_canon, canon_pain_nodes, canon_cap_nodes,
     structural_nodes, structural_edges) = phase0()

    # Phase 1
    (v5_nodes, pain_texts, cap_texts, cap_vendors,
     customer_name_to_slug) = phase1(pain_to_canon, cap_to_canon)

    # Phase 2
    (direct_edges, provides_grp, addresses_grp, experiences_grp,
     pain_vendor_urls) = phase2(pain_to_canon, cap_to_canon, customer_name_to_slug)

    # Phase 3
    log("Phase 3: Building canonical edges with evidence …")
    provides_edges = build_provides_edges(provides_grp, cap_texts)
    log(f"  provides edges:    {len(provides_edges):,}")
    del provides_grp

    addresses_edges = build_addresses_edges(
        addresses_grp, cap_vendors, customer_name_to_slug)
    log(f"  addresses edges:   {len(addresses_edges):,}")
    del addresses_grp

    experiences_edges = build_experiences_edges(
        experiences_grp, pain_texts, pain_vendor_urls)
    log(f"  experiences edges: {len(experiences_edges):,}")
    del experiences_grp, pain_vendor_urls

    # Phase 4
    summary = write_graph(
        out_dir, v5_nodes, canon_pain_nodes, canon_cap_nodes,
        structural_nodes, structural_edges, direct_edges,
        provides_edges, addresses_edges, experiences_edges)

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s")

    # Phase 5
    if not args.skip_verify:
        verify(out_dir, summary)


if __name__ == "__main__":
    main()
