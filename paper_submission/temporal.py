"""Publication-date sidecars, calendar cutoffs and source-filtered graph views."""
import argparse
import calendar
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from paper_submission.artifacts import lines, save, write_lines


def months_before(value, months=6):
    value = date.fromisoformat(value)
    if type(months) is not int or months < 0:
        raise ValueError("Months must be a nonnegative integer.")
    absolute = value.year * 12 + value.month - 1 - months
    year, zero_month = divmod(absolute, 12)
    month = zero_month + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1])).isoformat()


def eligible_sources_before(rows, cutoff):
    cutoff_date = date.fromisoformat(cutoff)
    observed = {}
    for row in rows:
        if row.get("temporal_use") != "eligible" or row.get("date_semantics") != "editorial_publication":
            continue
        published = row.get("reported_published_date")
        if not published:
            continue
        parsed = date.fromisoformat(published)
        key = row["vendor_slug"], row["url"]
        if key in observed and observed[key] != parsed:
            raise ValueError("Conflicting eligible publication dates for one exact source.")
        observed[key] = parsed
    return {key: parsed for key, parsed in observed.items() if parsed <= cutoff_date}


def source_references(edge):
    """Normalize exact source keys from construction or sampled graph records."""
    result = set()
    for field in ("sources", "source_records", "provenance", "evidence"):
        for row in edge.get(field, []):
            if isinstance(row, dict) and row.get("vendor_slug") and row.get("url"):
                result.add((row["vendor_slug"], row["url"]))
    if edge.get("vendor_slug"):
        for url in edge.get("source_urls", []):
            result.add((edge["vendor_slug"], url))
    return result


def graph_before(nodes, edges, sources, dates, *, cutoff, excluded_customers=(), excluded_urls=()):
    """Withhold unscoped claims; retain only exact dated, non-target support."""
    eligible = eligible_sources_before(dates, cutoff)
    excluded = set(excluded_customers)
    nodes, edges = list(nodes), list(edges)
    index = {node["id"]: dict(node) for node in nodes}
    if len(index) != len(nodes):
        raise ValueError("Duplicate graph node ID.")
    known, ownership = set(), defaultdict(set)
    for row in sources:
        key = row["vendor_slug"], row["url"]
        known.add(key)
        ownership[key].update(row.get("customer_ids", []))
    for edge in edges:
        if edge["src"] not in index or edge["dst"] not in index:
            raise ValueError("Dangling graph endpoint.")
        customers = {
            edge[key] for key in ("src", "dst") if index[edge[key]]["type"] == "customer"
        }
        for source in source_references(edge):
            ownership[source].update(customers)
    target_sources = {key for key in known if ownership[key] & excluded}
    excluded_urls = set(excluded_urls)
    reference_sources = {key for key in known if key[1] in excluded_urls}
    allowed = (known & eligible.keys()) - target_sources - reference_sources
    kept, withheld = [], Counter()
    for edge in edges:
        if edge["src"] in excluded or edge["dst"] in excluded:
            withheld["target_endpoint"] += 1
            continue
        references = source_references(edge)
        if not references:
            withheld["unscoped_source_support"] += 1
            continue
        if not references <= known:
            raise ValueError("Graph source is absent from supplied source index.")
        retained = references & allowed
        if not retained:
            withheld["no_eligible_non_target_support"] += 1
            continue
        output = {key: edge[key] for key in ("src", "type", "dst")}
        output["source_records"] = [
            {"vendor_slug": vendor, "url": url, "published_date": eligible[(vendor, url)].isoformat()}
            for vendor, url in sorted(retained)
        ]
        output["source_urls"] = sorted({url for _, url in retained})
        claims = [
            claim for claim in edge.get("claims", [])
            if source_references({"evidence": [claim.get("evidence", {})]}) & retained
        ]
        if claims:
            output["claims"] = claims
            output["contributing_raw_edges"] = [
                {"src": src, "type": relation, "dst": dst}
                for src, relation, dst in sorted({
                    (claim["raw_src"], claim["raw_type"], claim["raw_dst"]) for claim in claims
                })
            ]
        kept.append(output)
    incident = {edge[key] for edge in kept for key in ("src", "dst")}
    if not incident <= index.keys():
        raise ValueError("Dangling graph endpoint.")
    scoped_nodes = []
    for key in sorted(incident):
        node = index[key]
        output = {
            field: node[field] for field in
            ("id", "type", "text", "name", "canonical_name", "dimension", "attribute_type", "level")
            if field in node
        }
        evidence = [
            row for row in node.get("evidence", [])
            if source_references({"evidence": [row]}) & allowed
        ]
        if evidence:
            output["evidence"] = evidence
        scoped_nodes.append(output)
    return scoped_nodes, kept, {
        "cutoff": cutoff, "date_semantics": "editorial_publication",
        "retained_edges": len(kept), "withheld_edges": dict(withheld),
        "excluded_target_source_groups": len(target_sources),
        "excluded_reference_source_groups": len(reference_sources),
        "excluded_customer_ids": sorted(excluded),
        "node_metadata_policy": "identity_labels_and_eligible_evidence_only",
        "interpretation": "publication_snapshot_not_adoption_or_relationship_start",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--dates", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--exclude-customer", action="append", default=[])
    parser.add_argument("--exclude-url", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Snapshot output must be new.")
    nodes, edges, report = graph_before(
        lines(args.graph / "nodes.jsonl"), lines(args.graph / "edges.jsonl"),
        lines(args.sources), lines(args.dates), cutoff=args.cutoff,
        excluded_customers=args.exclude_customer,
        excluded_urls=args.exclude_url,
    )
    args.out.mkdir(parents=True)
    write_lines(args.out / "nodes.jsonl", nodes)
    write_lines(args.out / "edges.jsonl", edges)
    save(args.out / "manifest.json", report)


if __name__ == "__main__":
    main()
