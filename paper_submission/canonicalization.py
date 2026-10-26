"""Provenance-preserving raw, canonical and raw-preserving hybrid projections."""
from __future__ import annotations

import copy
from collections import defaultdict

from paper_submission.composition import ATOM_TYPES
from paper_submission.construction import ContractError, DIMENSIONS, digest
from paper_submission.induction import CANONICAL_TYPES

BUSINESS = {
    "offers": ({"vendor"}, {"product"}),
    "provides": ({"product"}, {"capability", "cap_canonical"}),
    "addresses": ({"capability", "cap_canonical"}, {"pain_point", "pain_canonical"}),
    "experiences": ({"customer"}, {"pain_point", "pain_canonical"}),
    "in_audience": ({"customer"}, {"audience"}),
    "has_case_study": ({"vendor"}, {"customer"}),
    "competes_with": ({"vendor"}, {"vendor"}),
    "asserts": ({"vendor"}, {"strategy"}),
    "cites": ({"strategy"}, {"customer"}),
    "targets": ({"strategy"}, {"audience"}),
}


def validate_graph(graph):
    nodes = {}
    for node in graph["nodes"]:
        if node["id"] in nodes:
            raise ContractError("Duplicate graph node ID.")
        nodes[node["id"]] = node
        kind, dimension, level = node["type"], node.get("dimension"), node.get("level")
        if kind in {"vendor", "product", "customer"}:
            if dimension is not None:
                raise ContractError("Company/product identities are not semantic atoms.")
        elif kind in {"capability", "pain_point", "cap_canonical", "pain_canonical", "audience", "strategy"}:
            expected_dimension = {
                "capability": "capability", "cap_canonical": "capability", "pain_point": "pain",
                "pain_canonical": "pain", "audience": "audience", "strategy": "strategy",
            }[kind]
            expected_levels = {"raw", "canonical"} if kind in {"audience", "strategy"} else {
                "canonical" if kind in {"cap_canonical", "pain_canonical"} else "raw"}
            if dimension != expected_dimension or level not in expected_levels:
                raise ContractError("Invalid semantic node type/dimension/level.")
        elif kind.endswith(("_domain", "_subtheme")):
            prefix = {"pain": "pain", "capability": "cap", "audience": "audience", "strategy": "strategy"}.get(dimension)
            if not prefix or kind not in {prefix + "_domain", prefix + "_subtheme"} or level != "category":
                raise ContractError("Invalid hierarchy category type/dimension.")
        else:
            raise ContractError("Unsupported graph node type.")
    sources = {row["id"]: row for row in graph.get("sources", [])}
    if len(sources) != len(graph.get("sources", [])):
        raise ContractError("Duplicate source-version record.")
    seen, hierarchy = set(), defaultdict(list)
    for row in graph["edges"]:
        triple = row["src"], row["type"], row["dst"]
        if triple in seen:
            raise ContractError("Duplicate typed graph edge.")
        seen.add(triple)
        if row["src"] not in nodes or row["dst"] not in nodes:
            raise ContractError("Dangling graph edge endpoint.")
        src, dst = nodes[row["src"]], nodes[row["dst"]]
        if row["type"] in BUSINESS:
            source_types, destination_types = BUSINESS[row["type"]]
            if src["type"] not in source_types or dst["type"] not in destination_types:
                raise ContractError("Business relation endpoint signature mismatch.")
            if not row.get("claims"):
                raise ContractError("Business relations require source-specific claims.")
            for claim in row["claims"]:
                if not claim.get("id") or not claim.get("evidence", {}).get("source_id"):
                    raise ContractError("Incomplete business-claim provenance.")
                if not all(claim.get(field) for field in ["raw_src", "raw_type", "raw_dst"]):
                    raise ContractError("Business claims require original raw-atom/edge lineage.")
                evidence = claim["evidence"]
                source = sources.get(evidence["source_id"])
                if (not source or any(evidence.get(key) != source.get(key)
                                      for key in ["vendor_slug", "url", "content_sha256"])
                        or claim["raw_type"] != row["type"]
                        or claim["id"] != "claim:" + digest({
                            "edge": (claim["raw_src"], claim["raw_type"], claim["raw_dst"]),
                            "evidence": evidence, "role": claim["role"],
                        })[:24]):
                    raise ContractError("Claim hash/source-version/raw relation lineage is inconsistent.")
            source_records = sorted(
                {(record["vendor_slug"], record["url"]) for record in row.get("source_records", [])},
            )
            claim_sources = sorted({(claim["evidence"]["vendor_slug"], claim["evidence"]["url"])
                                    for claim in row["claims"]})
            if source_records != claim_sources or len(row.get("source_records", [])) != len(source_records):
                raise ContractError("Exact source-record index differs from supporting claims.")
        elif row["type"] in {"is_a", "instance_of"}:
            dimension = row.get("dimension")
            if dimension not in DIMENSIONS or src.get("dimension") != dimension or dst.get("dimension") != dimension:
                raise ContractError("Cross-dimension semantic mapping/hierarchy.")
            if row["type"] == "instance_of":
                if src.get("level") != "raw" or dst.get("level") != "canonical":
                    raise ContractError("instance_of must link raw atoms to canonical concepts.")
            else:
                if dst.get("level") != "category" or src.get("level") not in {"category", "canonical"}:
                    raise ContractError("is_a must point from a concept/category to a category.")
            hierarchy[row["src"]].append(row["dst"])
        else:
            raise ContractError("Unknown graph relation: " + row["type"])
    visiting, visited = set(), set()

    def visit(identifier):
        if identifier in visiting:
            raise ContractError("Cyclic semantic hierarchy.")
        if identifier in visited:
            return
        visiting.add(identifier)
        for target in hierarchy[identifier]:
            visit(target)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in list(nodes):
        visit(identifier)
    return {"nodes": len(nodes), "edges": len(seen), "endpoint_integrity": True, "acyclic_semantic_links": True}


def project(raw, induction):
    validate_graph(raw)
    for edge in raw["edges"]:
        if edge["type"] not in BUSINESS or any(
            (claim["raw_src"], claim["raw_type"], claim["raw_dst"]) != (edge["src"], edge["type"], edge["dst"])
            for claim in edge["claims"]
        ):
            raise ContractError("Raw input must contain only original native source claims.")
    raw_nodes = {node["id"]: node for node in raw["nodes"]}
    learned = {}
    for node in induction["nodes"]:
        if node["id"] in raw_nodes or node["id"] in learned:
            raise ContractError("Induced node ID is duplicated or collides with a raw entity.")
        learned[node["id"]] = copy.deepcopy(node)
    mappings = {}
    for row in induction["mappings"]:
        if set(row) - {"src", "type", "dst", "dimension", "status"} or row["type"] != "instance_of":
            raise ContractError("Invalid canonical assignment record.")
        src, dst, dimension = row["src"], row["dst"], row["dimension"]
        if src in mappings or src not in raw_nodes or dst not in learned or dimension not in DIMENSIONS:
            raise ContractError("Duplicate, unknown or unsupported mapping endpoint.")
        if (raw_nodes[src]["type"] != ATOM_TYPES[dimension]
                or raw_nodes[src].get("dimension") != dimension
                or learned[dst]["type"] != CANONICAL_TYPES[dimension]
                or learned[dst].get("dimension") != dimension
                or learned[dst].get("level") != "canonical"):
            raise ContractError("Mapping dimension/type mismatch.")
        mappings[src] = dst
    expected_members = defaultdict(set)
    for src, dst in mappings.items():
        expected_members[dst].add(src)
    for identifier, node in learned.items():
        if node.get("level") == "canonical":
            members = node.get("raw_ids", [])
            if not members or len(set(members)) != len(members) or set(members) != expected_members[identifier]:
                raise ContractError("Canonical membership and assignment manifest disagree.")
    for edge in induction["edges"]:
        if (set(edge) != {"src", "type", "dst", "dimension"} or edge["type"] != "is_a"
                or edge["src"] not in learned or edge["dst"] not in learned):
            raise ContractError("Induction can add definition hierarchy, never native company claims.")
    declared_unmapped = [row["id"] for row in induction.get("unmapped", [])]
    if len(declared_unmapped) != len(set(declared_unmapped)) or any(identifier in mappings for identifier in declared_unmapped):
        raise ContractError("Unmapped atoms overlap mappings or are duplicated.")
    for identifier in declared_unmapped:
        if identifier not in raw_nodes:
            raise ContractError("Unknown unmapped atom.")
    for row in induction.get("unmapped", []):
        if row.get("dimension") != raw_nodes[row["id"]].get("dimension"):
            raise ContractError("Unmapped-atom dimension mismatch.")
    if "dimensions" in induction:
        dimensions = {row["dimension"] for row in induction["dimensions"]}
        expected_atoms = {identifier for identifier, node in raw_nodes.items() if node.get("dimension") in dimensions}
        if set(mappings) | set(declared_unmapped) != expected_atoms:
            raise ContractError("Incomplete explicit mapping/unmapped manifest.")
    canonical_nodes = {identifier: copy.deepcopy(node) for identifier, node in raw_nodes.items()
                       if identifier not in mappings}
    canonical_nodes.update(learned)
    grouped = {}
    for row in raw["edges"]:
        triple = mappings.get(row["src"], row["src"]), row["type"], mappings.get(row["dst"], row["dst"])
        target = grouped.setdefault(triple, {"src": triple[0], "type": triple[1], "dst": triple[2],
                                             "claims": [], "contributing_raw_edges": [],
                                             "source_records": []})
        for claim in row["claims"]:
            if claim not in target["claims"]:
                target["claims"].append(copy.deepcopy(claim))
        target["contributing_raw_edges"].append({
            "src": row["src"], "type": row["type"], "dst": row["dst"],
            "claim_ids": sorted(claim["id"] for claim in row["claims"]),
        })
    for row in grouped.values():
        row["claims"].sort(key=lambda claim: claim["id"])
        row["contributing_raw_edges"].sort(key=lambda item: (item["src"], item["type"], item["dst"]))
        row["source_records"] = [
            {"vendor_slug": vendor, "url": url}
            for vendor, url in sorted({(claim["evidence"]["vendor_slug"], claim["evidence"]["url"])
                                      for claim in row["claims"]})
        ]
    canonical = {
        "nodes": [canonical_nodes[identifier] for identifier in sorted(canonical_nodes)],
        "edges": [grouped[triple] for triple in sorted(grouped)] + copy.deepcopy(induction["edges"]),
        "sources": copy.deepcopy(raw.get("sources", [])), "projection": "pi(raw edges) plus learned hierarchy",
    }
    hybrid = {
        "nodes": copy.deepcopy(raw["nodes"]) + [learned[identifier] for identifier in sorted(learned)],
        "edges": copy.deepcopy(raw["edges"]) + copy.deepcopy(induction["edges"]) + copy.deepcopy(induction["mappings"]),
        "sources": copy.deepcopy(raw.get("sources", [])),
        "projection": "native raw claims unchanged; canonical definitions linked by instance_of",
    }
    reports = {"raw": validate_graph(raw), "canonical": validate_graph(canonical), "hybrid": validate_graph(hybrid)}
    native = {(edge["src"], edge["type"], edge["dst"]): digest(edge) for edge in raw["edges"]}
    hybrid_native = {(edge["src"], edge["type"], edge["dst"]): digest(edge) for edge in hybrid["edges"]
                     if edge["type"] in BUSINESS}
    if native != hybrid_native:
        raise ContractError("Hybrid changed a native business claim.")
    canonical_claim_ids = {claim["id"] for edge in canonical["edges"] for claim in edge.get("claims", [])}
    raw_claim_ids = {claim["id"] for edge in raw["edges"] for claim in edge["claims"]}
    if canonical_claim_ids != raw_claim_ids:
        raise ContractError("Canonical projection lost or fabricated native claims.")
    return {
        "raw": copy.deepcopy(raw), "canonical": canonical, "hybrid": hybrid,
        "report": {**reports, "native_claim_ids_preserved": True, "hybrid_native_edges_unchanged": True,
                   "unmapped_raw_atoms_retained": sorted(
                       identifier for identifier, node in raw_nodes.items()
                       if node.get("dimension") in DIMENSIONS and identifier not in mappings)},
    }
