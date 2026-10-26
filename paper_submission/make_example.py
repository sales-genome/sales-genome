"""Write explicitly synthetic extraction fixtures; no unreleased graph rows."""
import argparse
from pathlib import Path

from jsonschema import Draft202012Validator

from paper_submission.artifacts import load, save, write_lines


def create(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    schema = load(Path(__file__).resolve().parents[1] / "schemas/unified/case_study_v2.json")
    observations = []
    for index, customer in enumerate(["Synthetic Alpha", "Synthetic Beta", "Synthetic Gamma"]):
        capability = ["Automate synthetic ticket tracking", "Automated synthetic ticket routing"][index % 2]
        pain = ["Manual synthetic ticket processing causes delays.",
                "Manual synthetic ticket triage creates delays."][index % 2]
        data = {
            "customer_name": customer, "inferred_target_persona": None,
            "audience_dimensions": {"industry": "Synthetic Industry", "company_size": None,
                                    "location": None, "revenue": None, "tech_stack": [], "other": None},
            "vendor_products_used": ["Synthetic Service"],
            "capabilities_cited": [capability], "pain_points_cited": [pain],
            "vp_pp_pairs": [{"capability": capability, "pain_point": pain, "confidence": "high"}],
            "outcomes": [], "confidence": "high",
        }
        Draft202012Validator(schema).validate(data)
        observations.append({
            "vendor_slug": "synthetic-vendor", "url": f"https://example.org/synthetic/{index}",
            "page_type": "case_study", "data": data,
        })
    write_lines(destination / "extractions.jsonl", observations)
    save(destination / "crosslinks.json", {"crosslinks": []})
    save(destination / "fixture-info.json", {
        "synthetic": True, "sampled_from_research_graph": False,
        "description": "Invented software fixtures, not companies, source evidence, labels, or experimental results.",
    })


def overlays(graph, destination):
    from paper_submission.artifacts import lines
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    nodes = list(lines(Path(graph) / "nodes.jsonl"))
    for dimension, raw_kind, canonical_kind in [
        ("pain", "pain_point", "pain_canonical"), ("capability", "capability", "cap_canonical"),
    ]:
        canonical = f"synthetic:canonical:{dimension}"
        parent, domain = f"synthetic:subtheme:{dimension}", f"synthetic:domain:{dimension}"
        prefix = "pain" if dimension == "pain" else "cap"
        overlay_nodes = [
            {"id": canonical, "type": canonical_kind, "canonical_name": f"Synthetic {dimension} concept"},
            {"id": parent, "type": prefix + "_subtheme", "name": "Synthetic subtheme"},
            {"id": domain, "type": prefix + "_domain", "name": "Synthetic domain"},
        ]
        edges = [{"src": node["id"], "type": "instance_of", "dst": canonical}
                 for node in nodes if node["type"] == raw_kind]
        edges.extend([{"src": canonical, "type": "in_subtheme", "dst": parent},
                      {"src": parent, "type": "in_domain", "dst": domain}])
        write_lines(destination / dimension / "overlay_nodes.jsonl", overlay_nodes)
        write_lines(destination / dimension / "overlay_edges.jsonl", edges)
    save(destination / "fixture-info.json", {
        "synthetic": True, "learned_taxonomy": False,
        "description": "Explicit test mappings exercise graph projection; no LLM induction was run.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph", type=Path, help="If supplied, write synthetic overlays for this synthetic graph")
    args = parser.parse_args()
    if args.graph:
        overlays(args.graph, args.out)
    else:
        create(args.out)


if __name__ == "__main__":
    main()
