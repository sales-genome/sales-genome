"""Validate standalone payload integrity, cross-view correspondence, and scope."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from paper_submission.artifacts import lines, load, package_paths, save, sha256
from paper_submission.sample_graph import Bundle, DATE_FIELDS, NODE_FIELDS, key, safe_url, select_customers

EDGE_FIELDS = {
    "src", "dst", "type", "native_record_sha256", "evidence_type", "customer_ids",
    "source_status", "source_urls", "vendor_slug", "context_role", "evidence",
    "sample_native_raw_edge_count",
}
SIGNATURES = {
    "offers": ({"vendor"}, {"product"}),
    "provides": ({"product"}, {"capability", "cap_canonical"}),
    "addresses": ({"capability", "cap_canonical"}, {"pain_point", "pain_canonical"}),
    "experiences": ({"customer"}, {"pain_point", "pain_canonical"}),
    "has_case_study": ({"vendor"}, {"customer"}),
    "in_audience": ({"customer"}, {"audience"}),
    "in_subtheme": ({"pain_canonical", "cap_canonical"}, {"pain_subtheme", "cap_subtheme"}),
    "in_domain": ({"pain_subtheme", "cap_subtheme"}, {"pain_domain", "cap_domain"}),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def standalone(root):
    root = Path(root)
    checked = 0
    forbidden_names = {".env", "server-credentials.json", "catalog.sqlite", "audience_genome.duckdb"}
    for path in package_paths(root):
        require(not path.is_symlink(), "Symlink in standalone package: " + str(path))
        if path.is_file():
            require(path.name not in forbidden_names, "Forbidden file: " + str(path))
            require(path.suffix not in {".sqlite", ".sqlite3", ".duckdb", ".pt", ".safetensors"}, "Forbidden database/model: " + str(path))
            checked += 1
    return checked


def validate_sample(root, source_bundle=None):
    root = Path(root)
    standalone(root)
    manifest = load(root / "manifest.json")
    require(manifest["sampler_sha256"] == sha256(Path(__file__).with_name("sample_graph.py")),
            "Sampler implementation differs from pinned sample manifest.")
    require(manifest["sampling_policy_approved_by_user"] is False, "Provisional policy must not claim user approval.")
    selected = {row["id"] for row in lines(root / "selection/customers.jsonl")}
    require("customer:" not in selected, "Invalid empty customer included.")
    require(len(selected) == manifest["denominator_valid_customer_groups"] // 5, "Wrong 20% customer-group count.")
    require(len(selected) == manifest["selected_customer_groups"], "Selection count mismatch.")
    require(sha256(root / "selection/customers.jsonl") == manifest["selection_sha256"], "Selection hash mismatch.")
    source_selection_verified = False
    if source_bundle is not None:
        source = Bundle(source_bundle)
        population = {
            row["id"] for row in source.stream("raw_v7/nodes.jsonl")
            if row["type"] == "customer" and row["id"] != "customer:"
        }
        require(len(population) == manifest["denominator_valid_customer_groups"], "Customer denominator differs from source.")
        require(set(select_customers(population, manifest["seed"])) == selected,
                "Exported selection differs from deterministic full-population selection.")
        source_selection_verified = True
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(actual_files == set(manifest["artifacts"]) | {"manifest.json"}, "Unexpected or missing sample payload file.")
    for relative, artifact in manifest["artifacts"].items():
        path = root / relative
        require(sha256(path) == artifact["sha256"], "Checksum mismatch: " + relative)
        require(path.stat().st_size == artifact["bytes"], "Size mismatch: " + relative)
        require(sum(1 for _ in lines(path)) == artifact["records"], "Record mismatch: " + relative)
    sources = {}
    for row in lines(root / "sources.jsonl"):
        require(set(row) == {"vendor_slug", "url", "customer_ids"}, "Unexpected source field.")
        require(safe_url(row["url"]), "Private or unsafe source URL.")
        require(set(row["customer_ids"]) <= selected and row["customer_ids"], "Unsampled source owner.")
        source = row["vendor_slug"], row["url"]
        require(source not in sources, "Duplicate exact source key.")
        sources[source] = set(row["customer_ids"])
    permitted_urls = {url for _, url in sources}
    views, view_edges = {}, {}
    for view in ["raw", "raw_matched", "canonical"]:
        nodes = {}
        for row in lines(root / view / "nodes.jsonl"):
            require(row["id"] not in nodes, "Duplicate node.")
            require(set(row) <= NODE_FIELDS, "Unfiltered node aggregate or evidence.")
            require(all(not isinstance(value, (list, dict)) for value in row.values()), "Nested node payload not permitted.")
            if row["type"] == "customer":
                require(row["id"] in selected, "Unsampled customer node.")
            nodes[row["id"]] = row
        require({node_id for node_id, node in nodes.items() if node["type"] == "customer"} == selected,
                "Customer IDs differ across views.")
        edges = {}
        for row in lines(root / view / "edges.jsonl"):
            triple = key(row)
            require(triple not in edges, "Duplicate typed edge.")
            require(set(row) <= EDGE_FIELDS, "Unexpected edge field.")
            require(row["src"] in nodes and row["dst"] in nodes, "Dangling edge endpoint.")
            require(row["type"] in SIGNATURES, "Unexpected native relation.")
            src_types, dst_types = SIGNATURES[row["type"]]
            require(nodes[row["src"]]["type"] in src_types and nodes[row["dst"]]["type"] in dst_types,
                    "Relation endpoint signature mismatch.")
            require(set(row.get("customer_ids", [])) <= selected, "Unsampled edge membership.")
            require(set(row.get("source_urls", [])) <= permitted_urls, "Source URL outside authorized source groups.")
            for evidence in row.get("evidence", []):
                require(set(evidence) <= {"atom_id", "vendor_slug", "url", "evidence_type"}, "Bulk citation or unsupported field.")
                require((evidence["vendor_slug"], evidence["url"]) in sources, "Evidence source outside scope.")
                require(evidence["atom_id"].startswith(("pain:", "cap:")), "Unexpected evidence atom ID.")
            edges[triple] = row
        require(len(nodes) == manifest["views"][view]["nodes"], "Manifest node count differs.")
        require(len(edges) == manifest["views"][view]["edges"], "Manifest edge count differs.")
        metadata = manifest["views"][view]
        require(metadata["node_fraction"] == len(nodes) / metadata["source_nodes"], "Wrong node disclosure fraction.")
        require(metadata["edge_fraction"] == len(edges) / metadata["source_edges"], "Wrong edge disclosure fraction.")
        actual_bytes = sum((root / view / filename).stat().st_size for filename in ["nodes.jsonl", "edges.jsonl"])
        require(metadata["bytes"] == actual_bytes and metadata["byte_fraction"] == actual_bytes / metadata["source_bytes"],
                "Wrong physical byte disclosure fraction.")
        views[view], view_edges[view] = nodes, edges
    shared_raw = {node_id for node_id, node in views["raw"].items()
                  if node["type"] in {"vendor", "product", "customer", "audience"}}
    metadata_differences = {}
    for view in ["raw_matched", "canonical"]:
        shared = {node_id for node_id, node in views[view].items()
                  if node["type"] in {"vendor", "product", "customer", "audience"}}
        require(shared == shared_raw, "Shared identity IDs differ across views.")
        metadata_differences[view] = sum(
            views[view][node_id] != views["raw"][node_id] for node_id in shared
        )
        require(all(views[view][node_id]["type"] == views["raw"][node_id]["type"] for node_id in shared),
                "Shared identity types differ across views.")
    mapping = {}
    for row in lines(root / "taxonomy/mappings.jsonl"):
        require(set(row) == {"src", "type", "dst"} and row["type"] == "instance_of", "Bad mapping record.")
        require(row["src"] in views["raw"] and row["dst"] in views["canonical"], "Mapping endpoint missing.")
        require(row["src"] not in mapping, "Ambiguous exported mapping.")
        mapping[row["src"]] = row["dst"]
    require(len(mapping) == manifest["mapping_count"], "Mapping count mismatch.")
    expected = Counter()
    for triple, row in view_edges["raw_matched"].items():
        require(triple in view_edges["raw"] and row == view_edges["raw"][triple], "Matched view claim differs from raw.")
        expected[(mapping.get(row["src"], row["src"]), row["type"], mapping.get(row["dst"], row["dst"]))] += 1
    canonical_business = {triple for triple in view_edges["canonical"] if triple[1] not in {"in_subtheme", "in_domain"}}
    require(set(expected) == canonical_business, "Canonical projection differs from matched raw claims.")
    require(all(view_edges["canonical"][triple]["sample_native_raw_edge_count"] == count
                for triple, count in expected.items()), "Incorrect canonical contribution count.")
    for row in view_edges["canonical"].values():
        if row["type"] == "experiences":
            require(all(mapping[evidence["atom_id"]] == row["dst"] for evidence in row.get("evidence", [])),
                    "Canonical source evidence has a mismatched raw atom.")
    for edges in view_edges.values():
        for row in edges.values():
            for evidence in row.get("evidence", []):
                require(evidence["atom_id"] in views["raw"], "Evidence references unreleased raw atom.")
                if row["type"] == "experiences":
                    require(row["src"] in sources[(evidence["vendor_slug"], evidence["url"])],
                            "Experience source not owned by its sampled customer.")
    labels = list(lines(root / "labels.jsonl"))
    pairs = {(row["src"], row["dst"]) for row in labels}
    require(len(labels) == len(pairs), "Duplicate labels or split overlap.")
    require(pairs == {(src, dst) for src, relation, dst in view_edges["raw"] if relation == "has_case_study"},
            "Labels and sampled relationship claims differ.")
    require(all(row["split"] == "unsplit" and row["dst"] in selected and set(row["source_urls"]) <= permitted_urls
                for row in labels), "False split or unsampled label.")
    require(not any(root.rglob("train_edges.jsonl")), "All-source sample must not masquerade as training graph.")
    temporal_count = 0
    for row in lines(root / "temporal/evidence_dates.jsonl"):
        require(set(row) <= DATE_FIELDS, "Unexpected temporal payload or aggregate field.")
        require((row["vendor_slug"], row["url"]) in sources, "Unsampled temporal source.")
        require(row["temporal_use"] == "eligible", "Review-only date leaked into normal sidecar.")
        require("reference_counts" not in row, "Global temporal aggregates retained.")
        temporal_count += 1
    native_count = len(view_edges["raw"]) + len(view_edges["canonical"])
    require(native_count == manifest["disclosure"]["distinct_native_raw_plus_canonical_edge_records"],
            "Native edge disclosure count differs.")
    require(native_count / manifest["disclosure"]["source_raw_plus_canonical_edge_records"]
            == manifest["disclosure"]["native_edge_record_fraction"], "Wrong combined native-edge disclosure fraction.")
    for path in root.rglob("*.jsonl"):
        with path.open() as stream:
            for line in stream:
                require(not re.search(r"(?<![\w:/.-])/(?:home|mnt|root|tmp)/", line),
                        "Private absolute path in payload.")
                require(len(line) <= 65536, "Unexpected bulk payload record; investigate before release.")
    return {
        "status": "PASS_LOCAL_DRAFT_INVARIANTS", "selected_customer_groups": len(selected),
        "exclusive_source_groups": len(sources), "label_pairs": len(labels),
        "labels_are_unsplit": True, "temporal_eligible_records": temporal_count,
        "projection_triples_checked": len(expected), "endpoint_signature_integrity": True,
        "shared_identity_ids_and_types_match": True,
        "source_view_metadata_differences_preserved_not_identity_rematched": metadata_differences,
        "bulk_citation_bodies_and_private_paths_absent": True,
        "sample_checksums_verified": len(manifest["artifacts"]),
        "deterministic_selection_recomputed_from_verified_full_population": source_selection_verified,
        "native_edge_disclosure_fraction": manifest["disclosure"]["native_edge_record_fraction"],
        "sampling_policy_approval_still_required_before_release": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-bundle", type=Path,
                        help="Optional read-only full population check; not needed for ordinary standalone validation")
    args = parser.parse_args()
    report = validate_sample(args.sample, args.source_bundle)
    if args.report:
        save(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
