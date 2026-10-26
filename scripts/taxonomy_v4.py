#!/usr/bin/env python3
"""Schema-first taxonomy + canonicalization for v4 pains / capabilities / audiences.

Adapts taxonomy-gen's schema-first decomposition to the audience-genome v4 graph,
embedding-free, using gpt-5.4-mini. For one dimension over a chosen vendor set:

  A0  Load atoms from graph/v4 + text-dedup + held-out split (seeded)
  A1  Domain discovery   (LLM proposes ~15-28 canonical domains from a train sample)
  B1  Domain assignment  (batched LLM routing of every atom -> domain)
  A2  Sub-theme schema    (per domain: propose -> validate(fit) -> refine; train only)
  B2  Sub-theme assignment (batched, every atom -> sub-theme within its domain)
  C   Slot merge          (per (domain, sub-theme): whole-slot LLM merge into
                           self-contained canonical leaves; chunk+reconcile if big;
                           NO pruning, singletons kept)
  D   Eval + relations    (fit_rate, held-out coverage HOCR, compression;
                           vendor<->vendor via shared canonical capabilities,
                           customer<->customer via shared canonical pains)

Outputs a sidecar overlay under the run dir; does NOT mutate nodes/edges.jsonl.

Usage:
  .venv-exp/bin/python scripts/taxonomy_v4.py --dimension pain \
      --vendors datadog splunk new-relic dynatrace grafana-labs snowflake bynder \
      --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.logging import log
from lib.oai import tool_call, tracker

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "graph" / "v4"
NODES = V4 / "nodes.jsonl"
EDGES = V4 / "edges.jsonl"

MAX_TEXT = 320  # truncate atom text in prompts


# ───────────────────────────── A0: load + split ─────────────────────────────

def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def load_atoms(dimension: str, vendors: set[str]):
    """Return (atoms, cust_name) where atoms is a list of dicts:
    {id, text, vendors:[...], customers:[...], source_ids:[...], n_occ}."""
    cust_name: dict[str, str] = {}
    pain_customers: dict[str, set[str]] = defaultdict(set)

    if dimension == "pain":
        # pain -> customers via experiences edges; customer slug -> name
        with EDGES.open() as f:
            for line in f:
                if '"experiences"' not in line:
                    continue
                e = json.loads(line)
                if e.get("type") == "experiences":
                    cslug = e["src"].split(":", 1)[1]
                    if cslug:
                        pain_customers[e["dst"]].add(cslug)
        with NODES.open() as f:
            for line in f:
                if '"customer"' in line:
                    d = json.loads(line)
                    if d.get("type") == "customer" and d.get("slug"):
                        cust_name[d["slug"]] = d.get("name") or d["slug"]

        atoms = []
        with NODES.open() as f:
            for line in f:
                if '"pain_point"' not in line:
                    continue
                d = json.loads(line)
                if d.get("type") != "pain_point":
                    continue
                vs = sorted(set(d.get("vendor_slugs", [])) & vendors)
                if not vs:
                    continue
                custs = sorted(
                    cust_name.get(c, c) for c in pain_customers.get(d["id"], set())
                )
                atoms.append({
                    "id": d["id"], "text": d["text"], "vendors": vs,
                    "customers": custs, "source_ids": [d["id"]],
                    "n_occ": int(d.get("frequency", 1) or 1),
                })
        return atoms, cust_name

    if dimension == "audience":
        atoms_path = ROOT / "graph" / "v5" / "audience_atoms" / "atoms.jsonl"
        atoms = []
        with atoms_path.open() as f:
            for line in f:
                a = json.loads(line)
                vs = sorted(set(a.get("vendors", [])) & vendors) if vendors else a.get("vendors", [])
                if vendors and not vs:
                    continue
                custs = [c["name"] for c in a.get("customers", [])[:30]]
                atoms.append({
                    "id": a["atom_id"],
                    "text": a["text"],
                    "vendors": vs if vs else a.get("vendors", []),
                    "customers": custs,
                    "source_ids": [a["atom_id"]],
                    "n_occ": a.get("n_extractions", 1),
                })
        return atoms, cust_name

    if dimension == "persuasion":
        # Precomputed by scripts/persuasion_extract.py (P0). Records are already
        # atom-shaped: `text` is the concatenated persuasion fingerprint string
        # (clustering key), with raw_text/fingerprint/page_type preserved.
        spans_path = ROOT / "graph" / "v5" / "persuasion" / "spans.jsonl"
        atoms = []
        with spans_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                a = json.loads(line)
                vs = sorted(set(a.get("vendors", [])) & vendors) if vendors else a.get("vendors", [])
                if vendors and not vs:
                    continue
                atoms.append({
                    "id": a["id"],
                    "text": a["text"],
                    "vendors": vs if vs else a.get("vendors", []),
                    "customers": a.get("customers", [])[:30],
                    "source_ids": a.get("source_ids", [a["id"]]),
                    "n_occ": a.get("n_occ", 1),
                    # carry-through payload (not used for clustering, but kept so
                    # downstream stages / D-stage annotation can reach the raw quote)
                    "raw_text": a.get("raw_text", ""),
                    "page_type": a.get("page_type"),
                    "fingerprint": a.get("fingerprint", {}),
                    "claim": a.get("claim", ""),
                })
        return atoms, cust_name

    # capability: per-vendor nodes; pre-merge identical text across vendors
    merged: dict[str, dict] = {}
    with NODES.open() as f:
        for line in f:
            if '"capability"' not in line:
                continue
            d = json.loads(line)
            if d.get("type") != "capability" or d.get("vendor_slug") not in vendors:
                continue
            key = _norm(d["text"])
            m = merged.get(key)
            if m is None:
                merged[key] = {
                    "id": "cap:" + hashlib.sha1(key.encode()).hexdigest()[:12],
                    "text": d["text"], "vendors": {d["vendor_slug"]},
                    "customers": set(d.get("customers", [])),
                    "source_ids": [d["id"]], "n_occ": 1,
                }
            else:
                m["vendors"].add(d["vendor_slug"])
                m["customers"].update(d.get("customers", []))
                m["source_ids"].append(d["id"])
                m["n_occ"] += 1
    atoms = []
    for m in merged.values():
        m["vendors"] = sorted(m["vendors"])
        m["customers"] = sorted(m["customers"])
        atoms.append(m)
    return atoms, cust_name


# ───────────────────────────── prompt helpers ───────────────────────────────

def _atom_block(idxs, atoms, with_index=True):
    lines = []
    for j, i in enumerate(idxs):
        t = atoms[i]["text"].replace("\n", " ").strip()[:MAX_TEXT]
        lines.append(f"{j}: {t}" if with_index else f"- {t}")
    return "\n".join(lines)


NOUN = {"pain": "pain point (a problem/challenge a company faces)",
        "capability": "capability (something a product lets a customer do)",
        "audience": "audience attribute (a factual characteristic of a company or its operating environment used for B2B segmentation)",
        "persuasion": "persuasion strategy (HOW a vendor rhetorically frames evidence to influence a buyer — described by its rhetorical mechanism, substance basis, tone, target emotion, and warrant strength; grouped by STYLE, never by the product/topic it is about)"}


# ───────────────────────── A1: domain discovery ─────────────────────────────

def discover_domains(atoms, train_idx, dimension, sample_n, seed):
    rng = random.Random(seed)
    sample = rng.sample(train_idx, min(sample_n, len(train_idx)))
    noun = NOUN[dimension]
    if dimension == "persuasion":
        guidance = (
            "Propose 12-22 canonical DOMAINS that are families of PERSUASION STYLE, "
            "grouped by HOW the vendor persuades — the rhetorical mechanism, the "
            "kind of evidence leaned on, the tone/emotion worked on — and NEVER by "
            "the product, feature, or business topic the quote happens to concern.\n"
            "GOOD domains (style): 'Peer & Consensus Social Proof', 'Third-Party "
            "Authority & Credentials', 'Quantified Outcome & ROI Framing', 'Risk "
            "Reversal & Reassurance', 'Scarcity & Urgency', 'Aspiration & Vision "
            "Casting', 'Identity & Belonging Appeal', 'Ease & Low-Friction Framing', "
            "'Scale & Market Dominance', 'Before/After Transformation'.\n"
            "FORBIDDEN — these are TOPICS, not styles: anything named after a "
            "product area or business function ('Marketing', 'Security', "
            "'Analytics', 'Workflow Automation', 'Commerce', 'Customer Support', "
            "'AI'). If a proposed domain names a business function or outcome area, "
            "it is WRONG — recast it as the underlying rhetorical mechanism. Two "
            "quotes about totally different products that persuade the same way "
            "belong in the same domain."
        )
    else:
        guidance = (
            "Propose 14-26 canonical DOMAINS that together cover this space. A domain "
            "is a broad, mutually-distinct theme (e.g. for pains: 'Observability & "
            "Incident Response', 'Data Integration', 'Security & Threat Detection'). "
        )
    prompt = (
        f"You are designing the top level of a taxonomy of {noun}s.\n\n"
        f"Below are {len(sample)} real {dimension} statements sampled from B2B "
        f"vendor case studies and product pages:\n\n{_atom_block(sample, atoms, False)}\n\n"
        f"{guidance}\n"
        f"Rules: domains must be mutually exclusive and collectively cover the "
        f"sample; use Title Case names; give each a one-sentence description.\n\n"
        f"Return via emit_domains."
    )
    schema = {"type": "object", "properties": {"domains": {"type": "array", "items": {
        "type": "object", "properties": {
            "name": {"type": "string"}, "description": {"type": "string"}}}}}}
    out = tool_call(prompt, "emit_domains", schema, max_tokens=6000)
    domains = (out or {}).get("domains") or []
    # de-dup names defensively
    seen, clean = set(), []
    for d in domains:
        n = (d.get("name") or "").strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            clean.append({"name": n, "description": (d.get("description") or "").strip()})
    return clean


# ─────────────────────── B: batched classification ──────────────────────────

def _classify_batch(idxs, atoms, options, level_name):
    """Assign each atom idx to one option index (or -1). options: list of
    (name, description). Returns {atom_global_idx: option_idx}."""
    opt_block = "\n".join(f"{k}: {n} — {d}" for k, (n, d) in enumerate(options))
    prompt = (
        f"Assign each item to the single best {level_name} from this list. "
        f"If none fits, use -1.\n\n{level_name.upper()}S:\n{opt_block}\n\n"
        f"ITEMS:\n{_atom_block(range(len(idxs)), [atoms[i] for i in idxs])}\n\n"
        f"Return via emit_assignments with one entry per item index."
    )
    schema = {"type": "object", "properties": {"assignments": {"type": "array", "items": {
        "type": "object", "properties": {
            "item": {"type": "integer"}, "choice": {"type": "integer"}}}}}}
    out = tool_call(prompt, "emit_assignments", schema, max_tokens=max(6000, 60 * len(idxs)))
    res = {}
    for a in (out or {}).get("assignments", []):
        j, c = a.get("item"), a.get("choice")
        if isinstance(j, int) and 0 <= j < len(idxs):
            res[idxs[j]] = c if (isinstance(c, int) and 0 <= c < len(options)) else -1
    for i in idxs:
        res.setdefault(i, -1)
    return res


def classify_all(idxs, atoms, options, level_name, workers, batch):
    chunks = [idxs[i:i + batch] for i in range(0, len(idxs), batch)]
    out: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_classify_batch, c, atoms, options, level_name): c for c in chunks}
        done = 0
        for fut in as_completed(futs):
            out.update(fut.result())
            done += 1
            if done % 10 == 0 or done == len(chunks):
                log("classify", f"{level_name}: {done}/{len(chunks)} batches  ${tracker.usd:.2f}")
    return out


# ─────────────────────── A2: sub-theme schema per domain ─────────────────────

def discover_subthemes(domain, dom_train_idx, atoms, dimension, seed,
                       max_rounds=3, fit_target=0.90, val_n=180, prop_n=140):
    noun = NOUN[dimension]
    rng = random.Random(seed)
    prop = rng.sample(dom_train_idx, min(prop_n, len(dom_train_idx)))
    propose_schema = {"type": "object", "properties": {"sub_themes": {"type": "array", "items": {
        "type": "object", "properties": {
            "name": {"type": "string"}, "description": {"type": "string"}}}}}}
    prompt = (
        f"Within the domain \"{domain}\", design 3-8 canonical SUB-THEMES that "
        f"partition these {noun} statements into mutually-exclusive groups:\n\n"
        f"{_atom_block(prop, atoms, False)}\n\nUse short Title Case names + a "
        f"one-sentence description each. Return via emit_sub_themes."
    )
    out = tool_call(prompt, "emit_sub_themes", propose_schema, max_tokens=3000)
    subs = [(s["name"].strip(), (s.get("description") or "").strip())
            for s in (out or {}).get("sub_themes", []) if s.get("name")]
    if not subs:
        subs = [("General", f"General {domain} {dimension} statements.")]
    fit = None
    for _ in range(max_rounds - 1):
        val = rng.sample(dom_train_idx, min(val_n, len(dom_train_idx)))
        asg = _classify_batch(val, atoms, subs, "sub-theme")
        assigned = sum(1 for i in val if asg[i] >= 0)
        fit = assigned / max(1, len(val))
        loads = defaultdict(int)
        for i in val:
            if asg[i] >= 0:
                loads[asg[i]] += 1
        overloaded = [subs[k][0] for k, c in loads.items()
                      if c / max(1, assigned) > 0.45]
        if fit >= fit_target and not overloaded and len(subs) <= 9:
            break
        unfit = [i for i in val if asg[i] < 0][:25]
        refine_prompt = (
            f"Refine the sub-theme schema for domain \"{domain}\". Current "
            f"sub-themes:\n" + "\n".join(f"- {n}: {d}" for n, d in subs) + "\n\n"
            + (f"These {len(unfit)} statements did NOT fit any sub-theme — add or "
               f"adjust sub-themes to cover them:\n{_atom_block(unfit, atoms, False)}\n\n"
               if unfit else "")
            + (f"These sub-themes are overloaded; consider splitting: {overloaded}\n\n"
               if overloaded else "")
            + f"Return the FULL revised list (3-9 sub-themes) via emit_sub_themes."
        )
        out = tool_call(refine_prompt, "emit_sub_themes", propose_schema, max_tokens=3000)
        new = [(s["name"].strip(), (s.get("description") or "").strip())
               for s in (out or {}).get("sub_themes", []) if s.get("name")]
        if new:
            subs = new
    return {"sub_themes": [{"name": n, "description": d} for n, d in subs],
            "fit_rate": round(fit, 3) if fit is not None else None}


# ─────────────────────────── C: slot merge ──────────────────────────────────

def _merge_chunk(local_idxs, atoms, dimension):
    noun = NOUN[dimension]
    rule = ("Merge two statements into ONE group only if they describe the SAME "
            "underlying problem — same root cause, same affected actor/process, "
            "same impact — so that no meaningful business distinction exists "
            "between them." if dimension == "pain" else
            "Merge two statements into ONE group only if they describe the SAME "
            "underlying capability — the same thing a customer can do — regardless "
            "of vendor branding or wording." if dimension == "capability" else
            "Each statement is a persuasion STYLE FINGERPRINT (mechanism | "
            "substance basis | tone | target emotion | warrant) — it describes HOW "
            "a vendor persuades, never the product/topic it is about. Merge two "
            "into ONE group only if they represent the SAME underlying persuasion "
            "strategy — same rhetorical mechanism working on the same emotion via "
            "the same kind of evidence — so that no meaningful distinction in HOW "
            "they persuade remains. Do NOT merge across genuinely different "
            "mechanisms (e.g. 'flexes raw scale' vs 'de-risks the choice' are "
            "distinct). NEVER split or distinguish on the business topic/subject "
            "(a scale-flex about security and a scale-flex about marketing are the "
            "SAME strategy)." if dimension == "persuasion" else
            "Merge two statements into ONE group only if they describe the SAME "
            "underlying company characteristic — same attribute type, same scope, "
            "same discriminating power for B2B segmentation — regardless of "
            "wording differences. 'Founded in 2014' and 'Founded in 2015' are "
            "distinct (different founding years). 'AWS' and 'Amazon Web Services' "
            "are the same.")
    prompt = (
        f"Below are {len(local_idxs)} {noun} statements that already share a narrow "
        f"theme. {rule} Different severity, phrasing, product names, or examples are "
        f"NOT a distinction. Genuinely distinct items stay in their own group "
        f"(single-member groups are fine).\n\n"
        f"{_atom_block(range(len(local_idxs)), [atoms[i] for i in local_idxs])}\n\n"
        f"For each group return: canonical_name (ONE self-contained sentence stating "
        f"the {noun}, general enough to cover all members yet faithful to them), "
        f"scope_definition (one line: what's in/out of scope), key_discriminator "
        f"(what sets it apart from sibling groups), and members (the item indices). "
        f"Every item index must appear in exactly one group. Return via emit_groups."
    )
    schema = {"type": "object", "properties": {"groups": {"type": "array", "items": {
        "type": "object", "properties": {
            "canonical_name": {"type": "string"},
            "scope_definition": {"type": "string"},
            "key_discriminator": {"type": "string"},
            "members": {"type": "array", "items": {"type": "integer"}}}}}}}
    out = tool_call(prompt, "emit_groups", schema, max_tokens=16000)
    groups = []
    for g in (out or {}).get("groups", []):
        members = [local_idxs[m] for m in g.get("members", [])
                   if isinstance(m, int) and 0 <= m < len(local_idxs)]
        if members:
            groups.append({
                "canonical_name": (g.get("canonical_name") or "").strip(),
                "scope_definition": (g.get("scope_definition") or "").strip(),
                "key_discriminator": (g.get("key_discriminator") or "").strip(),
                "members": members})
    # fall back: any ungrouped atoms become singletons
    grouped = {i for g in groups for i in g["members"]}
    for i in local_idxs:
        if i not in grouped:
            groups.append({"canonical_name": atoms[i]["text"][:MAX_TEXT],
                           "scope_definition": "", "key_discriminator": "",
                           "members": [i]})
    return groups


def _reconcile_once(protos, dimension):
    """Merge equivalent chunk-level canonicals. protos carry atom-index `members`;
    returns a new, shorter proto list with unioned members."""
    noun = NOUN[dimension]
    block = "\n".join(f"{k}: {p['canonical_name']}  [{p['scope_definition']}]"
                      for k, p in enumerate(protos))
    prompt = (
        f"These are canonical {noun} labels produced from different batches of the "
        f"same narrow theme. Some are duplicates of each other. Group labels that "
        f"refer to the SAME underlying {noun} (no meaningful distinction). Distinct "
        f"ones stay alone.\n\n{block}\n\nFor each merged group return canonical_name "
        f"(pick/refine the clearest), scope_definition, and members (the label "
        f"indices). Every index appears once. Return via emit_merges."
    )
    schema = {"type": "object", "properties": {"merges": {"type": "array", "items": {
        "type": "object", "properties": {
            "canonical_name": {"type": "string"},
            "scope_definition": {"type": "string"},
            "members": {"type": "array", "items": {"type": "integer"}}}}}}}
    out = tool_call(prompt, "emit_merges", schema, max_tokens=8000)
    groups = []
    for m in (out or {}).get("merges", []):
        members = [k for k in m.get("members", []) if isinstance(k, int) and 0 <= k < len(protos)]
        if members:
            groups.append({"name": (m.get("canonical_name") or "").strip(),
                           "scope": (m.get("scope_definition") or "").strip(), "members": members})
    grouped = {k for g in groups for k in g["members"]}
    for k in range(len(protos)):
        if k not in grouped:
            groups.append({"name": protos[k]["canonical_name"],
                           "scope": protos[k]["scope_definition"], "members": [k]})
    newp = []
    for g in groups:
        atoms_m = sorted({i for k in g["members"] for i in protos[k]["members"]})
        disc = next((protos[k].get("key_discriminator", "") for k in g["members"]
                     if protos[k].get("key_discriminator")), "")
        newp.append({"canonical_name": g["name"] or protos[g["members"][0]]["canonical_name"],
                     "scope_definition": g["scope"], "key_discriminator": disc, "members": atoms_m})
    return newp


def _reconcile_protos(protos, dimension, max_n=100):
    """Hierarchical reconcile so huge slots don't overflow one prompt: reconcile in
    batches, then recurse on the (smaller) union until a single global pass fits."""
    if len(protos) <= 1:
        return protos
    if len(protos) <= max_n:
        return _reconcile_once(protos, dimension)
    out = []
    for i in range(0, len(protos), max_n):
        out.extend(_reconcile_once(protos[i:i + max_n], dimension))
    if len(out) >= len(protos):  # no progress — stop to avoid infinite recursion
        return out
    return _reconcile_protos(out, dimension, max_n)


def merge_slot(slot_idxs, atoms, dimension, chunk_size):
    chunks = [slot_idxs[i:i + chunk_size] for i in range(0, len(slot_idxs), chunk_size)]
    protos = []
    for c in chunks:
        protos.extend(_merge_chunk(c, atoms, dimension))
    if len(chunks) <= 1:
        return protos
    return _reconcile_protos(protos, dimension)


# ─────────────────────────── D: relations ───────────────────────────────────

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def build_relations(canon, atoms):
    vendor_sets = defaultdict(set)   # vendor -> canonical_ids
    cust_sets = defaultdict(set)     # customer -> canonical_ids
    for c in canon:
        for v in c["vendors"]:
            vendor_sets[v].add(c["canonical_id"])
        for cust in c["customers"]:
            cust_sets[cust].add(c["canonical_id"])

    def top_pairs(sets, kind, key="jaccard", n=40):
        pairs = []
        for x, y in itertools.combinations(sorted(sets), 2):
            sx, sy = sets[x], sets[y]
            shared = len(sx & sy)
            if shared == 0:
                continue
            pairs.append({kind: [x, y], "shared": shared,
                          "jaccard": round(jaccard(sx, sy), 4)})
        pairs.sort(key=lambda p: (-p[key], -p["shared"]))
        return pairs[:n]

    return {
        "vendor_canonical_counts": {v: len(s) for v, s in sorted(vendor_sets.items())},
        "vendor_similarity_top": top_pairs(vendor_sets, "vendors"),
        "customer_similarity_top": top_pairs(cust_sets, "customers", key="shared")[:40],
        "n_customers_with_canonical": len(cust_sets),
    }


# ─────────────────────────────── main ───────────────────────────────────────

def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    global V4, NODES, EDGES
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--dimension", required=True,
                    choices=["pain", "capability", "audience", "persuasion"])
    ap.add_argument("--vendors", nargs="+", default=None)
    ap.add_argument("--all-vendors", action="store_true", help="use every vendor in the v4 graph")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--held-out-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domain-sample", type=int, default=600)
    ap.add_argument("--chunk-size", type=int, default=170)
    ap.add_argument("--limit", type=int, default=0, help="cap atoms for a fast smoke test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints")
    args = ap.parse_args()
    V4 = args.graph_dir
    NODES, EDGES = V4 / "nodes.jsonl", V4 / "edges.jsonl"

    if args.all_vendors:
        if args.dimension in ("audience", "persuasion"):
            vendors = set()  # empty means "all" for precomputed-atom dimensions
        else:
            vendors = set()
            with NODES.open() as f:
                for line in f:
                    if '"vendor"' in line:
                        d = json.loads(line)
                        if d.get("type") == "vendor" and d.get("slug"):
                            vendors.add(d["slug"])
    elif args.vendors:
        vendors = set(args.vendors)
    else:
        ap.error("provide --vendors or --all-vendors")
    if args.output_dir:
        out = Path(args.output_dir)
    elif args.dimension == "persuasion":
        out = ROOT / "graph" / "v5" / "persuasion" / "taxonomy" / f"run_{int(time.time())}"
    else:
        out = V4 / "taxonomy" / args.dimension / f"run_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema").mkdir(exist_ok=True)
    log("init", f"dimension={args.dimension} vendors={len(vendors)} out={out}")

    # A0
    atoms, _ = load_atoms(args.dimension, vendors)
    rng = random.Random(args.seed)
    rng.shuffle(atoms)
    if args.limit:
        atoms = atoms[:args.limit]
    n = len(atoms)
    held = set(rng.sample(range(n), int(n * args.held_out_frac)))
    train_idx = [i for i in range(n) if i not in held]
    log("A0", f"{n} atoms ({len(train_idx)} train / {len(held)} held-out)")
    if n < 20:
        log("A0", "too few atoms; aborting"); return

    # A1 domains
    dom_path = out / "domains.json"
    domains = None if args.fresh else _load(dom_path)
    if not domains:
        domains = discover_domains(atoms, train_idx, args.dimension, args.domain_sample, args.seed)
        dom_path.write_text(json.dumps(domains, indent=2))
    log("A1", f"{len(domains)} domains  ${tracker.usd:.2f}")
    dom_opts = [(d["name"], d["description"]) for d in domains]

    # B1 domain assignment (all atoms)
    da_path = out / "atom_domain.json"
    da = None if args.fresh else _load(da_path)
    if da:
        atom_dom = {int(k): v for k, v in da.items()}
    else:
        atom_dom = classify_all(list(range(n)), atoms, dom_opts, "domain", args.workers, 60)
        da_path.write_text(json.dumps({str(k): v for k, v in atom_dom.items()}))
    by_dom = defaultdict(list)
    for i, d in atom_dom.items():
        by_dom[d].append(i)
    log("B1", "domain sizes: " + ", ".join(
        f"{domains[d]['name']}={len(ix)}" for d, ix in sorted(by_dom.items(), key=lambda x:-len(x[1])) if d >= 0))

    # A2 sub-theme schema per domain (train atoms only)
    schemas = {}
    def _a2(d):
        dom_train = [i for i in by_dom[d] if i not in held]
        if len(dom_train) < 8:
            return d, {"sub_themes": [{"name": "General", "description": ""}], "fit_rate": None}
        return d, discover_subthemes(domains[d]["name"], dom_train, atoms, args.dimension, args.seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_a2, d) for d in by_dom if d >= 0]):
            d, sc = fut.result()
            schemas[d] = sc
            (out / "schema" / f"domain_{d}.json").write_text(
                json.dumps({"domain": domains[d]["name"], **sc}, indent=2))
    log("A2", f"schemas for {len(schemas)} domains  ${tracker.usd:.2f}")

    # B2 sub-theme assignment within each domain (all atoms)
    sa_path = out / "atom_subtheme.json"
    sa = None if args.fresh else _load(sa_path)
    if sa:
        atom_sub = {int(k): v for k, v in sa.items()}
    else:
        atom_sub = {}
        for d, ix in by_dom.items():
            if d < 0:
                for i in ix:
                    atom_sub[i] = -1
                continue
            subs = [(s["name"], s.get("description", "")) for s in schemas[d]["sub_themes"]]
            if len(subs) == 1:
                for i in ix:
                    atom_sub[i] = 0
            else:
                atom_sub.update(classify_all(ix, atoms, subs, "sub-theme", args.workers, 50))
        sa_path.write_text(json.dumps({str(k): v for k, v in atom_sub.items()}))
    log("B2", f"sub-theme assignment done  ${tracker.usd:.2f}")

    # C slot merge
    slots = defaultdict(list)
    for i in range(n):
        d = atom_dom.get(i, -1)
        s = atom_sub.get(i, -1)
        if d >= 0 and s >= 0:
            slots[(d, s)].append(i)
    log("C", f"{len(slots)} non-empty slots; merging…")
    canon = []
    counter = itertools.count()
    lock_results = []
    def _merge(slot, idxs):
        return slot, merge_slot(idxs, atoms, args.dimension, args.chunk_size)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_merge, slot, idxs) for slot, idxs in slots.items()]
        done = 0
        for fut in as_completed(futs):
            lock_results.append(fut.result())
            done += 1
            if done % 5 == 0 or done == len(slots):
                log("C", f"{done}/{len(slots)} slots  ${tracker.usd:.2f}")

    atom_canon = {}
    for (d, s), groups in lock_results:
        dname = domains[d]["name"]
        sname = schemas[d]["sub_themes"][s]["name"] if s < len(schemas[d]["sub_themes"]) else "General"
        for g in groups:
            cid = f"{args.dimension[:3]}_c_{next(counter):04d}"
            members = g["members"]
            vset, cset = set(), set()
            for i in members:
                vset.update(atoms[i]["vendors"])
                cset.update(atoms[i]["customers"])
                atom_canon[i] = cid
            canon.append({
                "canonical_id": cid, "domain": dname, "sub_theme": sname,
                "canonical_name": g["canonical_name"],
                "scope_definition": g["scope_definition"],
                "key_discriminator": g.get("key_discriminator", ""),
                "n_atoms": len(members), "n_vendors": len(vset), "vendors": sorted(vset),
                "n_customers": len(cset), "customers": sorted(cset),
                "member_atom_ids": [atoms[i]["id"] for i in members],
            })

    # D eval + relations + outputs
    assigned = [i for i in range(n) if i in atom_canon]
    held_assigned = [i for i in held if i in atom_canon]
    n_singletons = sum(1 for c in canon if c["n_atoms"] == 1)
    cross_vendor = sum(1 for c in canon if c["n_vendors"] >= 2)
    stats = {
        "dimension": args.dimension, "vendors": sorted(vendors),
        "n_atoms": n, "n_train": len(train_idx), "n_held_out": len(held),
        "n_domains": len(domains),
        "n_canonical": len(canon),
        "fit_rate_all": round(len(assigned) / n, 3),
        "held_out_coverage_HOCR": round(len(held_assigned) / max(1, len(held)), 3),
        "compression_ratio": round(n / max(1, len(canon)), 2),
        "singleton_canonical_frac": round(n_singletons / max(1, len(canon)), 3),
        "cross_vendor_canonical_frac": round(cross_vendor / max(1, len(canon)), 3),
        "spend": tracker.report(),
    }
    relations = build_relations(canon, atoms)

    tree = defaultdict(lambda: defaultdict(list))
    for c in canon:
        tree[c["domain"]][c["sub_theme"]].append(
            {"canonical_id": c["canonical_id"], "canonical_name": c["canonical_name"],
             "n_atoms": c["n_atoms"], "n_vendors": c["n_vendors"], "vendors": c["vendors"]})

    (out / "canonical.jsonl").write_text("\n".join(json.dumps(c) for c in canon))
    (out / "assignments.jsonl").write_text(
        "\n".join(json.dumps({"atom_id": atoms[i]["id"], "canonical_id": atom_canon[i],
                              "held_out": i in held}) for i in assigned))
    (out / "taxonomy_tree.json").write_text(json.dumps(tree, indent=2))
    (out / "relations.json").write_text(json.dumps(relations, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    log("DONE", f"{len(canon)} canonical {args.dimension}s from {n} atoms "
                f"(CR {stats['compression_ratio']}x, fit {stats['fit_rate_all']}, "
                f"HOCR {stats['held_out_coverage_HOCR']}, ${tracker.usd:.2f})")
    print(json.dumps(stats, indent=2))
    print("\nTop vendor-similarity pairs (shared canonical " + args.dimension + "s):")
    for p in relations["vendor_similarity_top"][:12]:
        print(f"  {p['vendors'][0]:14s} ~ {p['vendors'][1]:14s}  jaccard={p['jaccard']:.3f}  shared={p['shared']}")


if __name__ == "__main__":
    main()
