"""Exact source-joined, quotation-deduplicated persuasion-family profiles."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from paper_submission.artifacts import lines, save


def profiles(spans, assignments, source_customers, customer_audiences):
    span_index = {}
    for span in spans:
        span_id = span["span_id"]
        if span_id in span_index and span_index[span_id] != span:
            raise ValueError("Conflicting span ID.")
        if not isinstance(span["text"], str) or not span["text"].strip():
            raise ValueError("Quotation text is required for deduplication.")
        span_index[span_id] = span
    membership, strategy_coordinates = {}, {}
    strategy_sellers = defaultdict(set)
    for row in assignments:
        if row["span_id"] not in span_index:
            raise ValueError("Assignment references an unknown span.")
        coordinates = row["family_id"], row["subtheme_id"]
        previous = membership.setdefault(row["span_id"], coordinates)
        if previous != coordinates:
            raise ValueError("Overlapping strategy memberships disagree on family/subtheme.")
        strategy_previous = strategy_coordinates.setdefault(row["strategy_id"], coordinates)
        if strategy_previous != coordinates:
            raise ValueError("Strategy ID belongs to inconsistent hierarchy slots.")
        span = span_index[row["span_id"]]
        occurrences = span.get("occurrences", [span])
        strategy_sellers[row["strategy_id"]].update(item["vendor_slug"] for item in occurrences)
    owners = defaultdict(set)
    for row in source_customers:
        owners[(row["vendor_slug"], row["url"])].update(row["customer_ids"])
    audiences = defaultdict(set)
    for row in customer_audiences:
        audiences[row["customer_id"]].add(row["audience_id"])
    seller_counts = defaultdict(lambda: defaultdict(set))
    industry_counts = defaultdict(lambda: defaultdict(set))
    ambiguous, matched, customers, sellers = set(), set(), set(), set()
    for span_id, (family, _) in membership.items():
        span = span_index[span_id]
        quote = " ".join(span["text"].casefold().split())
        for occurrence in span.get("occurrences", [span]):
            vendor = occurrence["vendor_slug"]
            seller_counts[vendor][family].add(quote)
            if occurrence.get("page_type") != "case_study":
                continue
            key = vendor, occurrence["url"]
            linked = owners[key]
            if len(linked) != 1:
                if len(linked) > 1:
                    ambiguous.add(key)
                continue
            customer = next(iter(linked))
            if not audiences[customer]:
                continue
            matched.add(span_id)
            customers.add(customer)
            sellers.add(vendor)
            for audience in audiences[customer]:
                industry_counts[audience][family].add(span_id)

    def summary(groups):
        result = {}
        for entity, families in sorted(groups.items()):
            counts = {family: len(values) for family, values in sorted(families.items())}
            total = sum(counts.values())
            result[entity] = {
                "total": total, "counts": counts,
                "shares": {family: count / total for family, count in counts.items()},
            }
        return result

    return {
        "seller_profiles": summary(seller_counts), "audience_profiles": summary(industry_counts),
        "strategies": len(strategy_sellers),
        "multi_seller_strategies": sum(len(values) > 1 for values in strategy_sellers.values()),
        "matched_case_spans": len(matched), "matched_customers": len(customers),
        "matched_sellers": len(sellers), "ambiguous_source_groups_excluded": len(ambiguous),
        "join": "exact_seller_url_then_single_customer_then_audience_no_alias_normalization",
        "interpretation": "seller_communication_profiles_not_customer_preferences_or_persuasion_effects",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spans", "assignments", "source-customers", "customer-audiences", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Analysis output must be new.")
    save(args.out, profiles(
        lines(args.spans), lines(args.assignments), lines(args.source_customers),
        lines(args.customer_audiences),
    ))


if __name__ == "__main__":
    main()
