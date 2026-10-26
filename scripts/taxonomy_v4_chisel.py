#!/usr/bin/env python3
"""Schema-first taxonomy + canonicalization — faithful "chisel" port of
taxonomy-gen's force-split mechanism, on top of the v4 graph.

This is `taxonomy_v4.py` with ONE gap fixed: taxonomy-gen's `discover_subfeatures`
(taxonomy-gen/src/schema.py) programmatically force-splits an overloaded flat
sub-theme into nested children via a dedicated LLM call, and re-verifies the
split actually happened before moving on. `taxonomy_v4.py::discover_subthemes`
only asked the LLM to "consider splitting" inside a flat re-propose — there was
no `children` field in its schema, no dedicated force-split call, and no
programmatic verification, so depth was structurally capped at exactly
domain -> sub-theme -> canonical (3 levels) for every dimension, regardless of
how overloaded any slot got.

This script restores that mechanism faithfully:
  - Real recursive `SubTheme(name, description, children)` schema (matches
    taxonomy-gen's `SubFeatureDef`).
  - A2 discovery loop validates/refines against TOP-LEVEL sub-theme names only
    (exactly like taxonomy-gen's `_validate_schema` / `_refine_schema` — nesting
    is not re-tested by the discovery loop itself).
  - When a slot is still overloaded after refine AND still has no children, a
    dedicated `_classify_into_subtheme_slots` call gathers real per-slot samples
    (falling back to generic validation samples only if too few), then a
    dedicated force-split call produces 2-4 children for THAT slot only
    (taxonomy-gen's `_force_split_slot`, not exp2_schema.py's shortcut of just
    reusing the generic validation batch).
  - Production classification (B2) classifies every atom against the FULL
    flattened leaf-path list (e.g. "Alert Fatigue > Noisy Low-Value Alerts"),
    mirroring taxonomy-gen's Phase B `schema_prompt_block()` — not a two-level
    domain-then-flat-subtheme classify.
  - Depth is now DATA-DRIVEN: a domain's sub-themes stay flat (depth 1) unless
    load actually demands a split; different domains/sub-themes can end up at
    different depths in the same run.

Everything else (A0 load+split, A1 domain discovery/assignment, C slot merge,
D relations) is unchanged and imported directly from `taxonomy_v4.py` — the 3
*documented, intentional* deviations from taxonomy-gen (rich full-sentence
canonical leaves, no min_brand_count pruning, reuse-not-optimized) are kept.

Outputs a sidecar overlay under the run dir; does NOT mutate nodes/edges.jsonl.

Usage:
  .venv-exp/bin/python scripts/taxonomy_v4_chisel.py --dimension pain \
      --all-vendors --workers 8
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.logging import log
from lib.oai import tool_call, tracker

from scripts import taxonomy_v4 as base_taxonomy
from scripts.taxonomy_v4 import (  # noqa: E402 — reuse the unchanged A0/A1/C/D machinery
    NODES, NOUN, V4,
    _atom_block, _classify_batch, _load,
    build_relations, classify_all, discover_domains, jaccard, load_atoms,
    merge_slot,
)

MAX_TEXT = 320


# ─────────────────────── recursive sub-theme schema ──────────────────────────

@dataclass
class SubTheme:
    """Mirrors taxonomy-gen's `SubFeatureDef` — a node that starts as a flat
    leaf and only gains `children` if load forces a split."""
    name: str
    description: str
    children: list["SubTheme"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """0 = still a flat leaf, 1 = has been force-split once."""
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    def leaf_paths(self, prefix: tuple = ()):
        path = prefix + (self.name,)
        if not self.children:
            yield path, self
        else:
            for c in self.children:
                yield from c.leaf_paths(path)

    def to_dict(self) -> dict:
        d = {"name": self.name, "description": self.description}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubTheme":
        return cls(name=d["name"], description=d.get("description", ""),
                   children=[cls.from_dict(c) for c in d.get("children", [])])


def schema_max_depth(subs: list[SubTheme]) -> int:
    """1 = flat schema, 2 = at least one slot was force-split once, etc."""
    if not subs:
        return 0
    return 1 + max(s.depth for s in subs)


def schema_leaf_paths(subs: list[SubTheme]) -> list[tuple[tuple[str, ...], SubTheme]]:
    out = []
    for s in subs:
        out.extend(s.leaf_paths())
    return out


# ───────────────────── A2 (faithful): propose/validate/refine/force-split ────

_SUBTHEME_SCHEMA = {"type": "object", "properties": {"sub_themes": {"type": "array", "items": {
    "type": "object", "properties": {
        "name": {"type": "string"}, "description": {"type": "string"}}}}}}

_CHILD_SCHEMA = {"type": "object", "properties": {"children": {"type": "array", "items": {
    "type": "object", "properties": {
        "name": {"type": "string"}, "description": {"type": "string"}}}}}}


def propose_subthemes(domain: str, prop_idx, atoms, dimension: str) -> list[SubTheme]:
    noun = NOUN[dimension]
    prompt = (
        f"Within the domain \"{domain}\", design 4-8 canonical SUB-THEMES that "
        f"partition these {noun} statements into mutually-exclusive, "
        f"collectively-exhaustive groups:\n\n{_atom_block(prop_idx, atoms, False)}\n\n"
        f"Rules:\n"
        f"  1. Each item should clearly belong to exactly ONE sub-theme.\n"
        f"  2. Keep a FLAT list at this stage — do not nest children yet.\n"
        f"  3. Use short Title Case names + a one-sentence description each.\n"
        f"Return via emit_sub_themes."
    )
    out = tool_call(prompt, "emit_sub_themes", _SUBTHEME_SCHEMA, max_tokens=3000)
    subs = [SubTheme(name=s["name"].strip(), description=(s.get("description") or "").strip())
            for s in (out or {}).get("sub_themes", []) if s.get("name")]
    if not subs:
        subs = [SubTheme(name="General", description=f"General {domain} {dimension} statements.")]
    return subs


def validate_subthemes(subs: list[SubTheme], val_idx, atoms, dimension: str):
    """Classify against TOP-LEVEL names only — matches taxonomy-gen's
    `_validate_schema`, which never tests nested children during discovery."""
    options = [(s.name, s.description) for s in subs]
    asg = _classify_batch(val_idx, atoms, options, "sub-theme")
    assigned = [i for i in val_idx if asg.get(i, -1) >= 0]
    fit = len(assigned) / max(1, len(val_idx))
    loads = defaultdict(int)
    for i in assigned:
        loads[asg[i]] += 1
    dist = {subs[k].name: c / max(1, len(assigned)) for k, c in loads.items()}
    unfit = [i for i in val_idx if asg.get(i, -1) < 0]
    return fit, dist, unfit


def classify_into_subtheme_slots(subs: list[SubTheme], val_idx, atoms, dimension: str):
    """Dedicated re-classify to get real per-slot samples for force-split —
    taxonomy-gen's `_classify_into_slots`, dropped in the earlier exp2 port."""
    options = [(s.name, s.description) for s in subs]
    asg = _classify_batch(val_idx, atoms, options, "sub-theme")
    by_slot = defaultdict(list)
    for i in val_idx:
        c = asg.get(i, -1)
        if c >= 0:
            by_slot[subs[c].name].append(i)
    return by_slot


def refine_subthemes(domain: str, subs: list[SubTheme], unfit_idx, overloaded: list[str],
                      atoms, dimension: str) -> list[SubTheme]:
    noun = NOUN[dimension]
    current = [{"name": s.name, "description": s.description} for s in subs]
    parts = [f'Current sub-themes for domain "{domain}":\n' + json.dumps(current, indent=2) + "\n"]
    if unfit_idx:
        parts.append(
            f"\n{len(unfit_idx)} {noun} statements did NOT fit any sub-theme. Examples:\n"
            f"{_atom_block(unfit_idx, atoms, False)}\n\nIf they represent a missing "
            f"sub-theme, add it.\n"
        )
    if overloaded:
        parts.append(
            f"\nThese sub-themes are overloaded (each absorbing a disproportionate "
            f"share of statements): {overloaded}.\nNote: at THIS stage keep the list "
            f"flat — do not add children. Overloaded slots will be split separately.\n"
        )
    parts.append(
        f"\nAdditional rules:\n"
        f"  - Merge sub-themes that overlap semantically.\n"
        f"  - Remove sub-themes that are not clearly distinct.\n"
        f"  - Aim for 4-8 top-level sub-themes.\n\n"
        f"Return the COMPLETE revised flat list via emit_sub_themes."
    )
    out = tool_call("".join(parts), "emit_sub_themes", _SUBTHEME_SCHEMA, max_tokens=3000)
    new = [SubTheme(name=s["name"].strip(), description=(s.get("description") or "").strip())
           for s in (out or {}).get("sub_themes", []) if s.get("name")]
    return new if new else subs


def force_split_subtheme(domain: str, sub: SubTheme, sample_idx, atoms, dimension: str,
                          max_children: int = 4, min_children: int = 2) -> list[SubTheme]:
    """taxonomy-gen's `_force_split_slot` — a dedicated call whose ONLY job is
    to partition one overloaded slot into genuine children."""
    noun = NOUN[dimension]
    prompt = (
        f"The sub-theme \"{sub.name}\" ({sub.description}) under domain \"{domain}\" "
        f"is too broad — it absorbs too many {noun} statements.\n\n"
        f"Here are {len(sample_idx)} statements currently classified under it:\n\n"
        f"{_atom_block(sample_idx, atoms, False)}\n\n"
        f"Split \"{sub.name}\" into 2-{max_children} CHILDREN sub-themes that partition "
        f"its scope into distinct, non-overlapping groups. Each child must be a "
        f"meaningful partition — do NOT just rename the parent. Return via emit_children."
    )
    out = tool_call(prompt, "emit_children", _CHILD_SCHEMA, max_tokens=2000)
    kids = [SubTheme(name=c["name"].strip(), description=(c.get("description") or "").strip())
            for c in (out or {}).get("children", []) if c.get("name")]
    return kids if len(kids) >= min_children else []


def discover_subthemes_faithful(domain: str, dom_train_idx, atoms, dimension: str, seed: int,
                                 max_rounds: int, fit_target: float, overload_frac: float,
                                 prop_n: int, val_n: int, force_split_min: int, max_children: int):
    rng = random.Random(seed)
    pool = dom_train_idx[:]
    rng.shuffle(pool)
    prop_idx = pool[:prop_n]
    remaining = pool[prop_n:]

    subs = propose_subthemes(domain, prop_idx, atoms, dimension)
    fit = None
    rounds_log = [{"round": 1, "n_subthemes": len(subs), "names": [s.name for s in subs]}]

    for rnd in range(2, max_rounds + 1):
        if len(remaining) < 20:
            break
        val_idx = remaining[:val_n]
        remaining = remaining[val_n:]

        fit, dist, unfit = validate_subthemes(subs, val_idx, atoms, dimension)
        overloaded = [name for name, frac in dist.items() if frac > overload_frac]
        rounds_log.append({"round": rnd, "fit_rate": round(fit, 3),
                           "overloaded": overloaded, "n_unfit": len(unfit)})

        if fit >= fit_target and not overloaded:
            break

        subs = refine_subthemes(domain, subs, unfit[:25], overloaded, atoms, dimension)

        if overloaded:
            by_name = {s.name: s for s in subs}
            still_flat = [by_name[n] for n in overloaded if n in by_name and not by_name[n].children]
            if still_flat:
                slot_samples = classify_into_subtheme_slots(subs, val_idx, atoms, dimension)
                for sub in still_flat:
                    samp = slot_samples.get(sub.name, [])
                    if len(samp) < force_split_min:
                        samp = val_idx[:30]
                    kids = force_split_subtheme(domain, sub, samp, atoms, dimension, max_children)
                    if kids:
                        sub.children = kids
                        rounds_log[-1].setdefault("force_split", []).append(
                            {"slot": sub.name, "children": [k.name for k in kids]})

    return {
        "sub_themes": [s.to_dict() for s in subs],
        "fit_rate": round(fit, 3) if fit is not None else None,
        "max_depth": schema_max_depth(subs),
        "n_leaf_slots": len(schema_leaf_paths(subs)),
        "rounds": rounds_log,
    }


# ───────────────────────────── main ───────────────────────────────────────

def _nested_tree_with_canonicals(sub_dicts: list[dict], canon_by_path: dict) -> list[dict]:
    """Deep-copy the schema tree and attach `canonicals` to every leaf node."""
    def walk(node: dict, path: tuple) -> dict:
        node = copy.deepcopy(node)
        path = path + (node["name"],)
        if node.get("children"):
            node["children"] = [walk(c, path) for c in node["children"]]
        else:
            node["canonicals"] = canon_by_path.get(path, [])
        return node
    return [walk(s, ()) for s in sub_dicts]


def main():
    global V4, NODES
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
    # Faithful A2 hyperparameters — taxonomy-gen defaults (schema.py / models.py Config)
    ap.add_argument("--fit-target", type=float, default=0.95)
    ap.add_argument("--overload-frac", type=float, default=0.20)
    ap.add_argument("--schema-max-rounds", type=int, default=5)
    ap.add_argument("--propose-sample", type=int, default=200)
    ap.add_argument("--validation-sample", type=int, default=300)
    ap.add_argument("--force-split-min-samples", type=int, default=5)
    ap.add_argument("--max-children-per-split", type=int, default=4)
    args = ap.parse_args()
    V4 = args.graph_dir
    NODES = V4 / "nodes.jsonl"
    base_taxonomy.V4, base_taxonomy.NODES, base_taxonomy.EDGES = V4, NODES, V4 / "edges.jsonl"

    from lib import providers
    providers.require_local()  # this script only ever calls an LLM; never let it hit Azure

    if args.all_vendors:
        if args.dimension in ("audience", "persuasion"):
            vendors = set()
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
        out = V4.parent / "v5" / "persuasion" / "taxonomy" / f"run_{int(time.time())}"
    else:
        out = V4 / "taxonomy_chisel" / args.dimension / f"run_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema").mkdir(exist_ok=True)
    log("init", f"dimension={args.dimension} vendors={len(vendors)} out={out}")

    # A0 (unchanged)
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

    # A1 domains (unchanged)
    dom_path = out / "domains.json"
    domains = None if args.fresh else _load(dom_path)
    if not domains:
        domains = discover_domains(atoms, train_idx, args.dimension, args.domain_sample, args.seed)
        dom_path.write_text(json.dumps(domains, indent=2))
    log("A1", f"{len(domains)} domains  ${tracker.usd:.2f}")
    dom_opts = [(d["name"], d["description"]) for d in domains]

    # B1 domain assignment (unchanged)
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
        f"{domains[d]['name']}={len(ix)}" for d, ix in sorted(by_dom.items(), key=lambda x: -len(x[1])) if d >= 0))

    # A2 (faithful): sub-theme schema per domain, with real force-split
    schemas = {}
    def _a2(d):
        dom_train = [i for i in by_dom[d] if i not in held]
        if len(dom_train) < 8:
            return d, {"sub_themes": [{"name": "General", "description": ""}],
                       "fit_rate": None, "max_depth": 1, "n_leaf_slots": 1, "rounds": []}
        return d, discover_subthemes_faithful(
            domains[d]["name"], dom_train, atoms, args.dimension, args.seed,
            max_rounds=args.schema_max_rounds, fit_target=args.fit_target,
            overload_frac=args.overload_frac, prop_n=args.propose_sample,
            val_n=args.validation_sample, force_split_min=args.force_split_min_samples,
            max_children=args.max_children_per_split)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_a2, d) for d in by_dom if d >= 0]):
            d, sc = fut.result()
            schemas[d] = sc
            (out / "schema" / f"domain_{d}.json").write_text(
                json.dumps({"domain": domains[d]["name"], **sc}, indent=2))
    depths = [sc["max_depth"] for sc in schemas.values() if sc.get("max_depth")]
    n_split_domains = sum(1 for dep in depths if dep > 1)
    log("A2", f"schemas for {len(schemas)} domains, {n_split_domains} force-split "
              f"(depth range {min(depths, default=1)}-{max(depths, default=1)})  ${tracker.usd:.2f}")

    # B2 (faithful): classify every atom against the FLATTENED LEAF PATHS
    leaf_by_dom: dict[int, list[tuple[tuple[str, ...], str]]] = {}
    for d, sc in schemas.items():
        subs = [SubTheme.from_dict(s) for s in sc["sub_themes"]]
        leaf_by_dom[d] = [(path, " > ".join(path)) for path, _ in schema_leaf_paths(subs)]

    sa_path = out / "atom_subtheme.json"
    sa = None if args.fresh else _load(sa_path)
    if sa:
        atom_leaf = {int(k): v for k, v in sa.items()}
    else:
        atom_leaf = {}
        for d, ix in by_dom.items():
            if d < 0:
                for i in ix:
                    atom_leaf[i] = -1
                continue
            leaves = leaf_by_dom[d]
            if len(leaves) == 1:
                for i in ix:
                    atom_leaf[i] = 0
            else:
                options = [(label, "") for _, label in leaves]
                atom_leaf.update(classify_all(ix, atoms, options, "sub-theme leaf", args.workers, 50))
        sa_path.write_text(json.dumps({str(k): v for k, v in atom_leaf.items()}))
    log("B2", f"leaf-path assignment done  ${tracker.usd:.2f}")

    # C slot merge (unchanged; keyed by (domain, leaf_path) instead of (domain, subtheme_idx))
    slots = defaultdict(list)
    for i in range(n):
        d = atom_dom.get(i, -1)
        s = atom_leaf.get(i, -1)
        if d >= 0 and s >= 0 and s < len(leaf_by_dom.get(d, [])):
            slots[(d, s)].append(i)
    log("C", f"{len(slots)} non-empty leaf slots; merging…")
    canon = []
    counter = itertools.count()
    def _merge(slot, idxs):
        return slot, merge_slot(idxs, atoms, args.dimension, args.chunk_size)
    lock_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_merge, slot, idxs) for slot, idxs in slots.items()]
        done = 0
        for fut in as_completed(futs):
            lock_results.append(fut.result())
            done += 1
            if done % 5 == 0 or done == len(slots):
                log("C", f"{done}/{len(slots)} slots  ${tracker.usd:.2f}")

    canon_by_path: dict = defaultdict(list)  # (domain_name, *leaf_path) -> [{...}]
    atom_canon = {}
    for (d, s), groups in lock_results:
        dname = domains[d]["name"]
        path = leaf_by_dom[d][s][0]
        sname = " > ".join(path)
        for g in groups:
            cid = f"{args.dimension[:3]}_c_{next(counter):04d}"
            members = g["members"]
            vset, cset = set(), set()
            for i in members:
                vset.update(atoms[i]["vendors"])
                cset.update(atoms[i]["customers"])
                atom_canon[i] = cid
            entry = {
                "canonical_id": cid, "domain": dname, "sub_theme": sname,
                "sub_theme_path": list(path), "schema_depth": len(path),
                "canonical_name": g["canonical_name"],
                "scope_definition": g["scope_definition"],
                "key_discriminator": g.get("key_discriminator", ""),
                "n_atoms": len(members), "n_vendors": len(vset), "vendors": sorted(vset),
                "n_customers": len(cset), "customers": sorted(cset),
                "member_atom_ids": [atoms[i]["id"] for i in members],
            }
            canon.append(entry)
            canon_by_path[(dname, *path)].append({
                "canonical_id": cid, "canonical_name": g["canonical_name"],
                "n_atoms": len(members), "n_vendors": len(vset), "vendors": sorted(vset)})

    # D eval + relations + outputs (unchanged metrics + new depth stats)
    assigned = [i for i in range(n) if i in atom_canon]
    held_assigned = [i for i in held if i in atom_canon]
    n_singletons = sum(1 for c in canon if c["n_atoms"] == 1)
    cross_vendor = sum(1 for c in canon if c["n_vendors"] >= 2)
    depth_counts = defaultdict(int)
    for c in canon:
        depth_counts[c["schema_depth"]] += 1
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
        "schema_depth_distribution": dict(sorted(depth_counts.items())),
        "n_domains_force_split": n_split_domains,
        "schema_max_depth_overall": max(depths, default=1),
        "params": {"fit_target": args.fit_target, "overload_frac": args.overload_frac,
                   "schema_max_rounds": args.schema_max_rounds},
        "spend": tracker.report(),
    }
    relations = build_relations(canon, atoms)

    tree = {}
    for d, sc in schemas.items():
        dname = domains[d]["name"]
        canon_by_path_for_domain = {k[1:]: v for k, v in canon_by_path.items() if k[0] == dname}
        tree[dname] = _nested_tree_with_canonicals(sc["sub_themes"], canon_by_path_for_domain)

    (out / "canonical.jsonl").write_text("\n".join(json.dumps(c) for c in canon))
    (out / "assignments.jsonl").write_text(
        "\n".join(json.dumps({"atom_id": atoms[i]["id"], "canonical_id": atom_canon[i],
                              "held_out": i in held}) for i in assigned))
    (out / "taxonomy_tree.json").write_text(json.dumps(tree, indent=2))
    (out / "relations.json").write_text(json.dumps(relations, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    log("DONE", f"{len(canon)} canonical {args.dimension}s from {n} atoms "
                f"(CR {stats['compression_ratio']}x, fit {stats['fit_rate_all']}, "
                f"HOCR {stats['held_out_coverage_HOCR']}, "
                f"{n_split_domains}/{len(domains)} domains force-split, "
                f"${tracker.usd:.2f})")
    print(json.dumps(stats, indent=2))
    print("\nTop vendor-similarity pairs (shared canonical " + args.dimension + "s):")
    for p in relations["vendor_similarity_top"][:12]:
        print(f"  {p['vendors'][0]:14s} ~ {p['vendors'][1]:14s}  jaccard={p['jaccard']:.3f}  shared={p['shared']}")


if __name__ == "__main__":
    main()
