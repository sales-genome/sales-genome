"""Run an offline construction-to-task workflow using synthetic observations."""
import argparse
from pathlib import Path
import subprocess
import sys

from paper_submission.artifacts import lines, save
from paper_submission.downstream import account_score, retrieve_pitch_context
from paper_submission.make_example import create, overlays
from paper_submission.temporal import eligible_sources_before


def run(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    create(destination / "input")
    subprocess.run([
        sys.executable, "scripts/build_v4_graph.py", "--input-dir", str(destination / "input"),
        "--out", str(destination / "raw"), "--skip-llm",
    ], check=True)
    overlays(destination / "raw", destination / "overlay")
    subprocess.run([
        sys.executable, "scripts/build_v5_graph.py", "--base", str(destination / "raw"),
        "--pain-overlay", str(destination / "overlay/pain"),
        "--cap-overlay", str(destination / "overlay/capability"),
        "--out", str(destination / "canonical"),
    ], check=True)
    nodes, edges = list(lines(destination / "canonical/nodes.jsonl")), list(lines(destination / "canonical/edges.jsonl"))
    contexts = retrieve_pitch_context(
        nodes, edges, "vendor:synthetic-vendor", {"audience:synthetic-industry"},
        exclude_customer="customer:synthetic-alpha",
    )
    if not contexts or any("customer:synthetic-alpha" in row["peer_customer_ids"] for row in contexts):
        raise ValueError("Synthetic peer exclusion/retrieval failed.")
    dates = [{
        "vendor_slug": "synthetic-vendor", "url": "https://example.org/synthetic/0",
        "reported_published_date": "2020-01-01", "temporal_use": "eligible",
        "date_semantics": "editorial_publication",
    }]
    report = {
        "synthetic": True, "sampled_from_research_graph": False,
        "network_calls": 0, "models_trained": 0,
        "extraction": "pre-authored schema-valid fixtures, NOT LLM extraction",
        "induction": "explicit fixture mappings, NOT a learned taxonomy",
        "raw_and_canonical_endpoint_integrity": True,
        "prospect_excluded_from_peer_evidence": True,
        "retrieval_contexts": contexts,
        "synthetic_account_score": account_score(["CONFIRMED", "PROBABLE", "INSUFFICIENT"]),
        "eligible_exact_source_date_joins": len(eligible_sources_before(dates, "2021-01-01")),
        "reproduces_paper_metrics": False,
    }
    save(destination / "smoke-report.json", report)
    print("OFFLINE_SYNTHETIC_SMOKE_OK", destination / "smoke-report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/offline-demo"))
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
