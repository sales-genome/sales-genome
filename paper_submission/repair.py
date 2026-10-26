"""Offline exact-decision adapter; not a substitute for historical source review."""
from collections import defaultdict


def apply_identity_decisions(nodes, edges, decisions):
    existing = {node["id"]: dict(node) for node in nodes}
    replacements = {}
    for decision in decisions:
        if decision.get("decision") != "merge" or decision.get("review_status") != "accepted":
            continue
        source, target = decision["source_id"], decision["target_id"]
        if source not in existing or target not in existing:
            raise ValueError("Explicit identity decision has an unknown endpoint.")
        if existing[source]["type"] != existing[target]["type"]:
            raise ValueError("Cross-type identity merges are not allowed.")
        if source == target or source in replacements:
            raise ValueError("Duplicate or reflexive identity decision.")
        replacements[source] = target
    for source in replacements:
        visited, target = {source}, replacements[source]
        while target in replacements:
            if target in visited:
                raise ValueError("Cyclic identity decisions.")
            visited.add(target)
            target = replacements[target]
        replacements[source] = target
    rewritten = defaultdict(list)
    for edge in edges:
        if edge["src"] not in existing or edge["dst"] not in existing:
            raise ValueError("Unknown input edge endpoint.")
        src, dst = replacements.get(edge["src"], edge["src"]), replacements.get(edge["dst"], edge["dst"])
        rewritten[(src, edge["type"], dst)].append(dict(edge))
    return (
        [existing[item] for item in sorted(set(existing) - set(replacements))],
        [{"src": src, "type": relation, "dst": dst, "input_claims": claims}
         for (src, relation, dst), claims in sorted(rewritten.items())],
    )


def retain_reviewed_relationships(proposals):
    accepted, withheld = [], []
    for proposal in proposals:
        row = dict(proposal)
        if row.get("decision") == "supported" and row.get("review_status") == "accepted":
            if not row.get("source_url") or not row.get("review_evidence_hash"):
                raise ValueError("Accepted decisions require source URL and evidence hash.")
            accepted.append(row)
        else:
            withheld.append(row)
    return accepted, withheld
