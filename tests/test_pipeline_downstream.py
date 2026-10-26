import json
import subprocess
import sys

from paper_submission.artifacts import lines
from paper_submission.experiment_io import LocalCorpus
from paper_submission.tasks import Model, pitch, prioritize_account
from paper_submission.temporal import graph_before


def test_construction_induction_temporal_pitch_and_account_connect(tmp_path):
    out = tmp_path / "core"
    subprocess.run([
        sys.executable, "-m", "paper_submission.pipeline", "demo", "--out", str(out),
    ], check=True, capture_output=True, text=True)
    sources = list(lines(out / "sources.jsonl"))
    dates = [
        {**source, "temporal_use": "eligible", "date_semantics": "editorial_publication",
         "reported_published_date": source["published_date"]}
        for source in sources if source.get("published_date")
    ]
    nodes = list(lines(out / "canonical/nodes.jsonl"))
    edges = list(lines(out / "canonical/edges.jsonl"))
    seller = min(edge["src"] for edge in edges if edge["type"] == "offers")
    target = min(edge["dst"] for edge in edges
                 if edge["src"] == seller and edge["type"] == "has_case_study")
    audiences = {edge["dst"] for edge in edges
                 if edge["src"] == target and edge["type"] == "in_audience"}
    target_urls = {
        source["url"] for edge in edges if target in (edge["src"], edge["dst"])
        for source in edge["source_records"]
    }
    scoped_nodes, scoped_edges, report = graph_before(
        nodes, edges, sources, dates, cutoff="2020-07-01", excluded_customers=[target],
    )
    assert report["excluded_target_source_groups"] == len(target_urls) > 0
    serialized = json.dumps([scoped_nodes, scoped_edges])
    assert not any(url in serialized for url in target_urls)
    assert any(edge.get("claims") for edge in scoped_edges)
    assert all(edge["src"] != target and edge["dst"] != target for edge in scoped_edges)
    corpus = LocalCorpus([
        {"id": f"doc_{i}", "url": f"https://example.org/independent-{i}",
         "title": "Synthetic Prospect ticket routing", "text": "Support routing delays.",
         "published_date": "2020-01-01", "independence_group": f"group-{i}"}
        for i in range(3)
    ])

    def call(stage, payload, schema):
        if stage == "rank_graph_pains":
            return {"ranked_ids": list(payload["candidates"])}
        if stage == "aet_questions":
            return {"questions": ["Synthetic Prospect ticket routing"] if payload["round"] == 0 else []}
        if stage == "merge_pitch":
            return {"selected_ids": [f"graph_{i}" for i in range(5)] + [f"aet_{i}" for i in range(5)]}
        if stage == "predict_account_signals":
            return {"queries": ["Synthetic Prospect ticket routing"]}
        if stage == "grade_account_evidence":
            return {"grade": "CONFIRMED", "evidence_ids": ["doc_0", "doc_1", "doc_2"],
                    "narrative": "Synthetic independent signals.", "hindsight_concerns": ""}
        return {"pitch_points": [{
            "headline": f"{stage} synthetic point {i}", "pain": "Routing delays",
            "capability": "Automated routing", "reasoning": "Synthetic fixture hypothesis.",
            "candidate_ids": list(payload["graph_candidates"]),
            "evidence_ids": list(payload["source_evidence"]),
        } for i in range(payload["count"])]}

    result = pitch(
        {"seller_id": seller, "prospect_id": target, "audience_ids": sorted(audiences),
         "reference_date": "2021-01-01", "reference_source_urls": sorted(target_urls)},
        scoped_nodes, scoped_edges, corpus, Model(call),
    )
    assert len(result["pitch_points"]) == 10
    assert all(target not in point["peer_customer_ids"] for point in result["pitch_points"])
    score = prioritize_account(
        {"seller_id": seller, "account_name": "Synthetic Prospect", "cutoff": "2021-01-01"},
        scoped_nodes, scoped_edges, corpus, Model(call),
    )
    assert score["score"]["score"] > 0
