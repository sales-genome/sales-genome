"""Deterministic, source-scoped customer-group sampler for the paper graph."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import ipaddress
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from paper_submission.artifacts import lines, load, save, sha256, write_lines

SEED = "audience-genome-paper-submission-20261026-v1"
EXPECTED_BUNDLE_MANIFEST_SHA256 = "474ff3b0713be8ed00fdce6a949a394dc2302145358b7bca3555beeb909b83fd"
NODE_FIELDS = {
    "id", "type", "name", "text", "canonical_name", "slug", "vendor_slug",
    "canonical_industry", "dimension", "value", "domain", "sub_theme",
}
DATE_FIELDS = {
    "vendor_slug", "url", "hash", "reported_published_date", "modified_date",
    "date_semantics", "temporal_use", "extraction_confidence", "historical_confidence",
    "metadata_confidence", "anomaly_flags", "quality_reasons", "page_type", "page_class",
    "published_source", "modified_source",
}
ATOM_TYPES = {"capability", "pain_point"}
BUSINESS_TYPES = {"offers", "provides", "addresses", "experiences", "has_case_study", "in_audience"}


def key(row):
    return row["src"], row["type"], row["dst"]


def safe_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            return False
        if host == "localhost" or host.endswith((".local", ".internal")):
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            pass
        forbidden = {"token", "api_key", "apikey", "password", "secret", "authorization", "signature"}
        return not any(name.casefold() in forbidden for name, _ in parse_qsl(parsed.query))
    except ValueError:
        return False


def select_customers(ids, seed=SEED):
    ordered = sorted(ids, key=lambda item: (
        hashlib.sha256((seed + "\0" + item).encode()).hexdigest(), item,
    ))
    return sorted(ordered[:len(ordered) // 5])


def clean_node(row):
    return {name: value for name, value in row.items() if name in NODE_FIELDS}


def native_edge(row, **extra):
    return {
        "src": row["src"], "type": row["type"], "dst": row["dst"],
        "native_record_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode(),
        ).hexdigest(),
        "evidence_type": "native_builder_claim_not_independently_verified",
        **extra,
    }


class Bundle:
    def __init__(self, root):
        self.root = Path(root)
        if sha256(self.root / "manifest.json") != EXPECTED_BUNDLE_MANIFEST_SHA256:
            raise ValueError("Source bundle manifest is not the pinned manuscript-matched snapshot.")
        self.manifest = load(self.root / "manifest.json")
        self.inventory = {row["path"]: row for row in self.manifest["files"]}
        self.inputs = {}
        raw, canonical = load(self.root / "raw_v7/summary.json"), load(self.root / "canonical_v8/summary.json")
        if (raw["nodes"], raw["edges"], canonical["n_nodes"], canonical["n_edges"]) != (
            4525703, 6993277, 214236, 4559801,
        ):
            raise ValueError("This sampler requires the manuscript-matched Sept30 raw_v7/canonical_v8 bundle.")
        self.raw_summary, self.canonical_summary = raw, canonical

    def stream(self, relative):
        expected = self.inventory[relative]
        digest, records, size = hashlib.sha256(), 0, 0
        with (self.root / relative).open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                size += len(raw)
                if raw.strip():
                    records += 1
                    yield json.loads(raw)
        actual = {"sha256": digest.hexdigest(), "records": records, "bytes": size}
        if actual["sha256"] != expected["sha256"] or size != expected["size_bytes"]:
            raise ValueError("Input checksum mismatch: " + relative)
        if expected.get("records") is not None and records != expected["records"]:
            raise ValueError("Input record count mismatch: " + relative)
        self.inputs[relative] = actual


def enrich_manifest(manifest, destination, bundle):
    """Add per-table disclosure metrics without changing any selected payload."""
    manifest["sampler_sha256"] = sha256(Path(__file__))
    manifest["source_ownership_limit"] = (
        "Exclusivity is established only among saved native has_case_study references, "
        "whose source URL lists are capped; complete extraction-corpus ownership is not certified."
    )
    for name, view in manifest["views"].items():
        original = "canonical_v8" if name == "canonical" else "raw_v7"
        summary = bundle.canonical_summary if name == "canonical" else bundle.raw_summary
        view["node_type_fractions"] = {
            kind: count / summary["node_types"][kind] for kind, count in view["node_types"].items()
        }
        view["edge_type_fractions"] = {
            kind: count / summary["edge_types"][kind] for kind, count in view["edge_types"].items()
        }
        for table in ["nodes", "edges"]:
            output_bytes = (Path(destination) / name / (table + ".jsonl")).stat().st_size
            source_bytes = bundle.inventory[original + "/" + table + ".jsonl"]["size_bytes"]
            view[table + "_bytes"] = output_bytes
            view[table + "_byte_fraction"] = output_bytes / source_bytes
    for relative in ["raw_v7/summary.json", "canonical_v8/summary.json"]:
        path = bundle.root / relative
        digest = sha256(path)
        if digest != bundle.inventory[relative]["sha256"]:
            raise ValueError("Source summary hash mismatch: " + relative)
        manifest["source_inputs_verified_by_full_sha256"][relative] = {
            "sha256": digest, "bytes": path.stat().st_size, "records": None,
        }
    if not manifest["disclosure"]["at_most_20_percent_native_edges_each_view_and_combined"]:
        manifest["release_blockers"].append(
            "The conservative 20% native-edge bound was exceeded; explicit disclosure-policy approval required."
        )
    return manifest


def export(bundle_path, destination, seed=SEED):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Never overwrite an existing sample: " + str(destination))
    bundle = Bundle(bundle_path)
    customers = {}
    for node in bundle.stream("raw_v7/nodes.jsonl"):
        if node["type"] == "customer" and node["id"] != "customer:":
            customers[node["id"]] = clean_node(node)
    selected = set(select_customers(customers, seed))
    print("Selected", len(selected), "of", len(customers), "valid customer groups", flush=True)
    raw_edges, source_owners, case_rows = {}, defaultdict(set), []
    for edge in bundle.stream("raw_v7/edges.jsonl"):
        relation = edge["type"]
        if relation == "has_case_study":
            vendor = edge["src"].removeprefix("vendor:")
            for url in edge.get("source_urls", []):
                if safe_url(url):
                    source_owners[(vendor, url)].add(edge["dst"])
            if edge["dst"] in selected:
                case_rows.append(edge)
        elif relation in {"experiences", "in_audience"} and edge["src"] in selected:
            raw_edges[key(edge)] = native_edge(
                edge, customer_ids=[edge["src"]],
                source_status="customer_scoped_native_claim; complete_source_records_not_released",
            )
    sources = {
        source: owners for source, owners in source_owners.items()
        if owners and owners <= selected
    }
    mixed_sources_excluded = sum(bool(owners & selected) and not owners <= selected
                                 for owners in source_owners.values())
    for edge in case_rows:
        vendor = edge["src"].removeprefix("vendor:")
        urls = sorted(url for url in edge.get("source_urls", []) if (vendor, url) in sources)
        raw_edges[key(edge)] = native_edge(
            edge, customer_ids=[edge["dst"]], source_urls=urls,
            source_status="retained_exact_source_references_only" if urls else "native_claim_no_exclusive_source_reference",
        )
    selected_caps = set()
    for edge in bundle.stream("raw_v7/edges.jsonl"):
        if edge["type"] != "addresses":
            continue
        parts = edge["src"].split(":")
        if len(parts) != 3 or parts[0] != "cap":
            continue
        vendor = parts[1]
        urls = sorted(url for url in edge.get("source_urls", []) if (vendor, url) in sources)
        if not urls:
            continue
        owners = sorted(set().union(*(sources[(vendor, url)] for url in urls)))
        raw_edges[key(edge)] = native_edge(
            edge, source_urls=urls, vendor_slug=vendor, customer_ids=owners,
            source_status="exact_vendor_url_in_sampled_case_study_group; not_source_verified",
        )
        selected_caps.add(edge["src"])
    products, offers = set(), []
    for edge in bundle.stream("raw_v7/edges.jsonl"):
        if edge["type"] == "provides" and edge["dst"] in selected_caps:
            products.add(edge["src"])
            raw_edges[key(edge)] = native_edge(
                edge, context_role="shared_product_association_not_exclusive_source_attribution",
            )
        elif edge["type"] == "offers":
            offers.append(edge)
    for edge in offers:
        if edge["dst"] in products:
            raw_edges[key(edge)] = native_edge(edge, context_role="required_seller_product_endpoint_closure")
    required = selected | {endpoint for edge in raw_edges.values() for endpoint in (edge["src"], edge["dst"])}
    raw_nodes = {}
    for node in bundle.stream("raw_v7/nodes.jsonl"):
        if node["id"] in required:
            raw_nodes[node["id"]] = clean_node(node)
    if set(raw_nodes) != required:
        raise ValueError("Raw graph endpoint closure failed.")
    print("Retained raw business edges:", len(raw_edges), flush=True)
    mappings = {}
    for relative in [
        "taxonomy/pain_v6_overlay/overlay_edges.jsonl",
        "taxonomy/capability_v6_overlay/overlay_edges.jsonl",
    ]:
        for edge in bundle.stream(relative):
            if edge["type"] == "instance_of" and edge["src"] in raw_nodes:
                mappings[edge["src"]] = edge["dst"]
    matched_edges = {}
    projection = defaultdict(list)
    for triple, edge in raw_edges.items():
        if any(raw_nodes[endpoint]["type"] in ATOM_TYPES and endpoint not in mappings
               for endpoint in (edge["src"], edge["dst"])):
            continue
        mapped = mappings.get(edge["src"], edge["src"]), edge["type"], mappings.get(edge["dst"], edge["dst"])
        matched_edges[triple] = edge
        projection[mapped].append(edge)
    hierarchy, observed_canonical, recovered_evidence = [], set(), defaultdict(list)
    for edge in bundle.stream("canonical_v8/edges.jsonl"):
        triple = key(edge)
        if edge["type"] in {"in_subtheme", "in_domain"}:
            hierarchy.append({name: edge[name] for name in ("src", "type", "dst")})
        if triple in projection:
            observed_canonical.add(triple)
            if edge["type"] == "experiences":
                for evidence in edge.get("evidence", []):
                    atom = evidence.get("atom_id")
                    source = evidence.get("vendor"), evidence.get("url")
                    if atom in raw_nodes and source in sources and edge["src"] in sources[source]:
                        recovered_evidence[(edge["src"], atom)].append({
                            "atom_id": atom, "vendor_slug": source[0], "url": source[1],
                            "evidence_type": "retained_source_reference; quoted_page_text_omitted",
                        })
    absent = set(projection) - observed_canonical
    for triple in list(matched_edges):
        src, relation, dst = triple
        if (mappings.get(src, src), relation, mappings.get(dst, dst)) in absent:
            del matched_edges[triple]
    for pair, evidence in recovered_evidence.items():
        edge = raw_edges.get((pair[0], "experiences", pair[1]))
        if edge is not None:
            edge["evidence"] = evidence
    canonical_edges = {}
    for triple in sorted(observed_canonical):
        contributors = projection[triple]
        urls = sorted({url for edge in contributors for url in edge.get("source_urls", [])})
        owners = sorted({customer for edge in contributors for customer in edge.get("customer_ids", [])})
        canonical_edges[triple] = {
            "src": triple[0], "type": triple[1], "dst": triple[2],
            "sample_native_raw_edge_count": len(contributors),
            "evidence_type": "projection_of_sampled_native_claims; not_independent_verification",
            **({"source_urls": urls} if urls else {}),
            **({"customer_ids": owners} if owners else {}),
        }
        refs = [item for edge in contributors for item in edge.get("evidence", [])]
        if refs:
            canonical_edges[triple]["evidence"] = refs
    shared = {node_id for node_id, node in raw_nodes.items() if node["type"] not in ATOM_TYPES}
    canonical_required = shared | set(mappings.values())
    changed = True
    while changed:
        changed = False
        for edge in hierarchy:
            if edge["src"] in canonical_required:
                canonical_edges[key(edge)] = edge
                if edge["dst"] not in canonical_required:
                    canonical_required.add(edge["dst"])
                    changed = True
    canonical_nodes = {
        node["id"]: clean_node(node) for node in bundle.stream("canonical_v8/nodes.jsonl")
        if node["id"] in canonical_required
    }
    if set(canonical_nodes) != canonical_required:
        raise ValueError("Canonical hierarchy endpoint closure failed.")
    raw_matched_nodes = {
        node_id: node for node_id, node in raw_nodes.items()
        if node["type"] not in ATOM_TYPES or node_id in mappings
    }
    labels = [{
        "src": edge["src"], "dst": edge["dst"], "split": "unsplit",
        "source_urls": edge.get("source_urls", []),
        "label_kind": "native_documented_case_study_link_not_sales_qualification",
    } for edge in raw_edges.values() if edge["type"] == "has_case_study"]
    dates = []
    for row in bundle.stream("temporal/v1/evidence_dates.jsonl"):
        if (row.get("vendor_slug"), row.get("url")) in sources and row.get("temporal_use") == "eligible":
            dates.append({name: value for name, value in row.items() if name in DATE_FIELDS})
    # Payload creation starts only after the source and all correspondence checks pass.
    destination.mkdir(parents=True, exist_ok=False)
    write_lines(destination / "selection/customers.jsonl", ({"id": item} for item in sorted(selected)))
    write_lines(destination / "sources.jsonl", ({
        "vendor_slug": vendor, "url": url, "customer_ids": sorted(owners),
    } for (vendor, url), owners in sorted(sources.items())))
    write_lines(destination / "labels.jsonl", sorted(labels, key=lambda row: (row["src"], row["dst"])))
    write_lines(destination / "taxonomy/mappings.jsonl", ({
        "src": atom, "type": "instance_of", "dst": canonical,
    } for atom, canonical in sorted(mappings.items())))
    write_lines(destination / "temporal/evidence_dates.jsonl", sorted(dates, key=lambda row: (row["vendor_slug"], row["url"])))
    views = {}
    for name, nodes, edges, original in [
        ("raw", raw_nodes, raw_edges, "raw_v7"),
        ("raw_matched", raw_matched_nodes, matched_edges, "raw_v7"),
        ("canonical", canonical_nodes, canonical_edges, "canonical_v8"),
    ]:
        write_lines(destination / name / "nodes.jsonl", (nodes[item] for item in sorted(nodes)))
        write_lines(destination / name / "edges.jsonl", (edges[item] for item in sorted(edges)))
        node_source, edge_source = bundle.inputs[original + "/nodes.jsonl"], bundle.inputs[original + "/edges.jsonl"]
        output_bytes = sum((destination / name / filename).stat().st_size for filename in ["nodes.jsonl", "edges.jsonl"])
        views[name] = {
            "nodes": len(nodes), "edges": len(edges), "bytes": output_bytes,
            "node_types": dict(Counter(node["type"] for node in nodes.values())),
            "edge_types": dict(Counter(edge["type"] for edge in edges.values())),
            "source_nodes": node_source["records"], "source_edges": edge_source["records"],
            "source_bytes": node_source["bytes"] + edge_source["bytes"],
            "node_fraction": len(nodes) / node_source["records"],
            "edge_fraction": len(edges) / edge_source["records"],
            "byte_fraction": output_bytes / (node_source["bytes"] + edge_source["bytes"]),
        }
    native_edges = len(raw_edges) + len(canonical_edges)
    native_denominator = bundle.raw_summary["edges"] + bundle.canonical_summary["n_edges"]
    physical_edges = native_edges + len(matched_edges) + len(mappings)
    manifest = {
        "status": "LOCAL DRAFT — NOT A PUBLIC RELEASE",
        "sampling_policy": "PROVISIONAL engineering default: 20% of customer groups",
        "sampling_policy_approved_by_user": False,
        "release_blockers": ["User approval of customer-group denominator and disclosure policy is required before release.",
                             "Data rights/license and source-support review are required before redistribution."],
        "source_bundle": "audience-genome-full-2026-09-30",
        "source_bundle_manifest_sha256": sha256(bundle.root / "manifest.json"),
        "source_lineage": bundle.manifest["lineage"],
        "manuscript_inventory_match": {"raw_nodes": 4525703, "raw_edges": 6993277,
                                       "canonical_nodes": 214236, "canonical_edges": 4559801},
        "seed": seed,
        "selection_algorithm": "sort by (SHA256(seed + NUL + customer_id), customer_id); take floor(N/5)",
        "unit": "existing cleaned customer node ID; NOT independent source records, edges, nodes, or bytes",
        "customer_groups_including_invalid_empty_id": len(customers) + 1,
        "denominator_valid_customer_groups": len(customers),
        "excluded_invalid_customer_ids": ["customer:"],
        "selected_customer_groups": len(selected),
        "actual_customer_group_fraction": len(selected) / len(customers),
        "selection_sha256": sha256(destination / "selection/customers.jsonl"),
        "closure": [
            "All native experiences/in_audience/has_case_study edges for selected customers, without original aggregates.",
            "Addresses restricted to exact (capability vendor, URL) references owned exclusively by sampled customer groups.",
            "Mixed sampled/unsampled source groups are excluded; no URL/name fuzzy matching.",
            "Provides/offers are native shared endpoint context, NOT exclusive source-level product attribution.",
            "Raw atom text, selected identities, and minimal concept labels only; aggregate memberships/counts removed.",
            "Raw_matched and canonical use the SAME selected raw claims and saved last-instance_of mapping.",
            "Canonical business edges must also exist in the native canonical graph; required hierarchy ancestors retained.",
        ],
        "exclusive_source_groups": len(sources), "mixed_source_groups_excluded": mixed_sources_excluded,
        "labels": {"count": len(labels), "splits": {"unsplit": len(labels)},
                   "original_paper_splits_available": False, "train_edges_created": False},
        "mapping_count": len(mappings),
        "unmapped_selected_raw_atoms": sum(node["type"] in ATOM_TYPES and node_id not in mappings
                                           for node_id, node in raw_nodes.items()),
        "projected_triples_absent_from_native_canonical_excluded": len(absent),
        "temporal_eligible_records": len(dates),
        "views": views,
        "disclosure": {
            "distinct_native_raw_plus_canonical_edge_records": native_edges,
            "source_raw_plus_canonical_edge_records": native_denominator,
            "native_edge_record_fraction": native_edges / native_denominator,
            "physical_edge_records_including_matched_duplicates_and_mappings": physical_edges,
            "physical_edge_fraction_vs_source_base_edges": physical_edges / native_denominator,
            "native_raw_edge_fraction": views["raw"]["edge_fraction"],
            "native_canonical_edge_fraction": views["canonical"]["edge_fraction"],
            "at_most_20_percent_native_edges_each_view_and_combined": all(
                value <= .2 for value in [views["raw"]["edge_fraction"],
                                         views["canonical"]["edge_fraction"], native_edges / native_denominator]),
        },
        "source_inputs_verified_by_full_sha256": bundle.inputs,
        "known_source_issues": bundle.manifest["known_issues"],
        "exclusions": [
            "All unsampled customer nodes and relationship memberships; invalid empty customer.",
            "Full source pages/extractions, source databases, citations' bulk text and absolute paths.",
            "Whole node aggregates, customers/vendors lists, raw_variants, source/global counts, pooled confidence.",
            "Original vendor_pain_evidence, product-only source records, competitor model-inferred edges.",
            "Persuasion sidecars/taxonomy work, extraction_dates, review-only dates, model caches/embeddings/checkpoints.",
            "Private CRM/leads, credentials, endpoint registries, model-serving or session receipts.",
        ],
        "reproduction_claim": "Illustrative sample only. No manuscript experimental result is reproduced.",
    }
    manifest["artifacts"] = {
        str(path.relative_to(destination)): {
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "records": sum(1 for _ in lines(path)),
        } for path in sorted(destination.rglob("*.jsonl"))
    }
    manifest["total_payload_bytes_excluding_manifest"] = sum(row["bytes"] for row in manifest["artifacts"].values())
    manifest["payload_byte_fraction_vs_source_raw_and_canonical_tables"] = (
        manifest["total_payload_bytes_excluding_manifest"]
        / (views["raw"]["source_bytes"] + views["canonical"]["source_bytes"])
    )
    enrich_manifest(manifest, destination, bundle)
    save(destination / "manifest.json", manifest)
    print(json.dumps({"destination": str(destination), "views": views, "disclosure": manifest["disclosure"]}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Locally authorized Sept30 bundle; read-only")
    parser.add_argument("--out", type=Path, required=True, help="New standalone sample directory")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--acknowledge-provisional-local-draft", action="store_true", required=True)
    args = parser.parse_args()
    export(args.bundle, args.out, args.seed)


if __name__ == "__main__":
    main()
