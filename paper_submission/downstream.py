"""Offline task adapters; structural retrieval is not a generated sales claim."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from paper_submission.artifacts import lines, save

WEIGHTS = {"CONFIRMED": 10, "PROBABLE": 5, "SUGGESTIVE": 2, "INSUFFICIENT": 0}


def account_score(grades):
    grades = list(grades)
    if any(grade not in WEIGHTS for grade in grades):
        raise ValueError("Unknown evidence grade.")
    raw = sum(WEIGHTS[grade] for grade in grades)
    coverage = sum(grade != "INSUFFICIENT" for grade in grades) / len(grades) if grades else 0
    return {"raw_score": raw, "coverage": coverage, "score": raw * (0.5 + 0.5 * coverage)}


def retrieve_pitch_context(nodes, edges, vendor, audience_ids, exclude_customer=None):
    index = {node["id"]: node for node in nodes}
    outgoing, addresses = defaultdict(set), {}
    peer_customers = set()
    for edge in edges:
        outgoing[(edge["src"], edge["type"])].add(edge["dst"])
        if edge["type"] == "addresses":
            addresses[(edge["src"], edge["dst"])] = edge
        if edge["type"] == "in_audience" and edge["dst"] in audience_ids:
            if edge["src"] != exclude_customer:
                peer_customers.add(edge["src"])
    pain_peers = defaultdict(set)
    for customer in peer_customers:
        for pain in outgoing[(customer, "experiences")]:
            pain_peers[pain].add(customer)
    candidates = []
    for product in sorted(outgoing[(vendor, "offers")]):
        for capability in sorted(outgoing[(product, "provides")]):
            for pain in sorted(outgoing[(capability, "addresses")]):
                if pain not in pain_peers:
                    continue
                evidence = addresses[(capability, pain)]
                candidates.append({
                    "seller_id": vendor, "product_id": product, "capability_id": capability,
                    "pain_id": pain, "pain": index[pain].get("text", index[pain].get("canonical_name", "")),
                    "peer_customer_ids": sorted(pain_peers[pain]),
                    "source_urls": evidence.get("source_urls", []),
                    "interpretation": "structural_candidate_not_prospect_fact_or_verified_product_specific_proof",
                })
    return sorted(candidates, key=lambda row: (
        -len(row["peer_customer_ids"]), row["pain_id"], row["product_id"], row["capability_id"],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="Raw or canonical sample view")
    parser.add_argument("--vendor", required=True, help="Exact vendor node ID, including vendor:")
    parser.add_argument("--audience", nargs="+", required=True, help="Exact audience IDs")
    parser.add_argument("--exclude-customer")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    context = retrieve_pitch_context(
        list(lines(args.graph / "nodes.jsonl")), list(lines(args.graph / "edges.jsonl")),
        args.vendor, set(args.audience), args.exclude_customer,
    )
    save(args.out, {"kind": "offline_retrieval_not_LLM_pitch", "candidates": context})
    print("Retrieved", len(context), "structural candidates.")


if __name__ == "__main__":
    main()
