import unittest

from paper_submission.evaluate import (
    pitch_coverage, qualification_labels, ranking_metrics, stratified_accounts,
)
from paper_submission.experiment_io import LocalCorpus
from paper_submission.tasks import Model, aet, judge_pitch, pitch, prioritize_account
from paper_submission.temporal import graph_before, months_before


def graph_fixture():
    nodes = [{"id": key, "type": kind, "text": text} for key, kind, text in [
        ("seller", "vendor", "Synthetic Seller"), ("product", "product", "Synthetic Service"),
        ("cap", "capability", "Automate ticket routing"),
        ("pain", "pain_point", "Manual ticket routing delays service"),
        ("peer", "customer", "Synthetic Peer"), ("target", "customer", "Synthetic Target"),
        ("audience", "audience", "Synthetic Industry"),
    ]]
    edges = [
        {"src": source, "type": relation, "dst": target,
         "source_records": [{"vendor_slug": "synthetic", "url": "https://example.org/peer"}]}
        for source, relation, target in [
            ("seller", "offers", "product"), ("product", "provides", "cap"),
            ("cap", "addresses", "pain"), ("peer", "experiences", "pain"),
            ("peer", "in_audience", "audience"),
        ]
    ]
    return nodes, edges


def corpus_fixture():
    return LocalCorpus([{
        "id": f"doc_{i}", "url": f"https://example.org/evidence-{i}",
        "title": "Synthetic Target ticket routing", "text": "Manual ticket routing delays service.",
        "published_date": "2020-01-01", "independence_group": f"independent-{i}",
    } for i in range(3)])


class EvaluationTests(unittest.TestCase):
    def test_coverage_shape_and_denominator(self):
        self.assertEqual(pitch_coverage([5, 0, 5]), 2 / 3)
        for invalid in ([], [6], [-1], [True], [2.5]):
            with self.assertRaises(ValueError):
                pitch_coverage(invalid)

    def test_vendor_axis_ranking(self):
        result = ranking_metrics({"a": 3.0, "b": 2.0, "c": 1.0}, {"a"}, ks=(1, 2))
        self.assertEqual(result["average_precision"], 1)
        self.assertEqual(result["precision_at_1"], 1)
        self.assertEqual(result["precision_at_2"], 0.5)
        with self.assertRaises(ValueError):
            ranking_metrics({"a": float("nan")}, {"a"})

    def test_account_labels_are_contact_aggregated_and_positive_wins(self):
        rows = [
            {"Account Name": "Acme", "Status": "Qualified"},
            {"Account Name": "Acme", "Status Reason": "Not Interested"},
            {"Account Name": "Other", "Status Reason": "No Budget/No Need"},
            {"Account Name": "Neither", "Status": "Open"},
        ]
        result = qualification_labels(rows)
        self.assertTrue(result["acme"]["qualified"])
        self.assertFalse(result["acme"]["negative"])
        self.assertTrue(result["other"]["negative"])
        self.assertFalse(result["neither"]["eligible"])

    def test_ties_are_label_blind_and_zero_scores_are_retained(self):
        rows = [{"account_id": str(i), "qualified": i < 10, "score": 0.0} for i in range(100)]
        result = stratified_accounts(rows, positive_share=0.1, draws=1000, ks=(10,))
        self.assertEqual(result["n_positive_pool"], 10)
        self.assertLess(abs(result["metrics"]["10"]["precision"] - 0.1), 0.03)
        self.assertEqual(result, stratified_accounts(rows, positive_share=0.1, draws=1000, ks=(10,)))
        with self.assertRaises(ValueError):
            stratified_accounts(rows + [rows[0]], positive_share=0.1)


class TemporalTests(unittest.TestCase):
    def test_six_month_cutoff_is_calendar_based(self):
        self.assertEqual(months_before("2024-08-31"), "2024-02-29")
        self.assertEqual(months_before("2023-08-31"), "2023-02-28")

    def test_target_sources_are_removed_from_all_relations(self):
        nodes, edges = graph_fixture()
        target_source = {"vendor_slug": "synthetic", "url": "https://example.org/target"}
        edges.append({"src": "cap", "type": "addresses", "dst": "pain", "source_records": [target_source]})
        edges.append({"src": "seller", "type": "offers", "dst": "product"})
        sources = [
            {"vendor_slug": "synthetic", "url": "https://example.org/peer", "customer_ids": ["peer"]},
            {**target_source, "customer_ids": ["target"]},
        ]
        dates = [{**row, "temporal_use": "eligible", "date_semantics": "editorial_publication",
                  "reported_published_date": "2020-01-01"} for row in sources]
        _, kept, report = graph_before(nodes, edges, sources, dates, cutoff="2020-07-01",
                                       excluded_customers=["target"])
        self.assertEqual(len(kept), 5)
        self.assertEqual(report["excluded_target_source_groups"], 1)
        self.assertEqual(report["withheld_edges"]["unscoped_source_support"], 1)
        self.assertFalse(any(target_source["url"] in edge["source_urls"] for edge in kept))

    def test_cutoff_search_rejects_missing_and_future_dates(self):
        corpus = LocalCorpus([
            {"id": "past", "url": "https://example.org/a", "title": "ticket", "text": "routing",
             "published_date": "2020-01-01"},
            {"id": "future", "url": "https://example.org/b", "title": "ticket", "text": "routing",
             "published_date": "2030-01-01"},
            {"id": "undated", "url": "https://example.org/c", "title": "ticket", "text": "routing"},
        ])
        self.assertEqual([row["id"] for row in corpus.search("ticket", cutoff="2021-01-01")], ["past"])

    def test_reference_url_is_excluded_even_without_customer_ownership(self):
        nodes, edges = graph_fixture()
        source = {"vendor_slug": "synthetic", "url": "https://example.org/peer"}
        dates = [{**source, "temporal_use": "eligible", "date_semantics": "editorial_publication",
                  "reported_published_date": "2020-01-01"}]
        retained, kept, report = graph_before(
            nodes, edges, [source], dates, cutoff="2021-01-01",
            excluded_customers=["unknown-prospect"], excluded_urls=[source["url"]],
        )
        self.assertEqual((retained, kept), ([], []))
        self.assertEqual(report["excluded_reference_source_groups"], 1)


class TaskTests(unittest.TestCase):
    def test_graph_aet_generation_merges_ten_points_with_citations(self):
        calls = []

        def call(stage, payload, schema):
            calls.append(stage)
            if stage == "rank_graph_pains":
                return {"ranked_ids": list(payload["candidates"])}
            if stage == "aet_questions":
                return {"questions": ["Synthetic Target ticket routing"] if payload["round"] == 0 else []}
            if stage == "merge_pitch":
                return {"selected_ids": [f"graph_{i}" for i in range(5)] + [f"aet_{i}" for i in range(5)]}
            return {"pitch_points": [{
                "headline": f"{stage} distinct point {i}", "pain": "Potential routing delays",
                "capability": "Routing automation", "reasoning": "Peer evidence motivates a hypothesis.",
                "candidate_ids": list(payload["graph_candidates"]),
                "evidence_ids": list(payload["source_evidence"]),
            } for i in range(payload["count"])]}

        nodes, edges = graph_fixture()
        request = {"seller_id": "seller", "prospect_id": "target", "audience_ids": ["audience"],
                   "reference_date": "2021-01-01"}
        result = pitch(request, nodes, edges, corpus_fixture(), Model(call))
        self.assertEqual(len(result["pitch_points"]), 10)
        self.assertEqual(result["cutoff"], "2020-07-01")
        self.assertEqual(result["pitch_points"][0]["peer_customer_ids"], ["peer"])
        self.assertEqual(len(result["pitch_points"][-1]["source_urls"]), 3)
        self.assertIn("merge_pitch", calls)

    def test_aet_has_finite_query_budget(self):
        model = Model(lambda stage, payload, schema: {"questions": [f"ticket {payload['round']}"]})
        evidence, history = aet({}, corpus_fixture(), model, cutoff="2020-07-01", rounds=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(evidence), 3)

    def test_account_grading_and_weighted_score(self):
        def call(stage, payload, schema):
            if stage == "predict_account_signals":
                return {"queries": ["Synthetic Target ticket routing"]}
            return {"grade": "CONFIRMED", "evidence_ids": ["doc_0", "doc_1", "doc_2"],
                    "narrative": "Three independently supplied problem signals.", "hindsight_concerns": ""}

        nodes, edges = graph_fixture()
        result = prioritize_account(
            {"seller_id": "seller", "account_name": "Synthetic Target", "cutoff": "2021-01-01"},
            nodes, edges, corpus_fixture(), Model(call),
        )
        self.assertEqual(result["score"]["score"], 10)

    def test_grade_cannot_claim_three_independent_signals_from_one_group(self):
        def call(stage, payload, schema):
            if stage == "predict_account_signals":
                return {"queries": ["ticket"]}
            return {"grade": "CONFIRMED", "evidence_ids": ["doc_0"],
                    "narrative": "Unsupported strong grade.", "hindsight_concerns": ""}
        nodes, edges = graph_fixture()
        with self.assertRaisesRegex(ValueError, "independent"):
            prioritize_account(
                {"seller_id": "seller", "account_name": "Synthetic Target", "cutoff": "2021-01-01"},
                nodes, edges, corpus_fixture(), Model(call),
            )

    def test_pitch_judge_requires_each_reference_rating(self):
        model = Model(lambda stage, payload, schema: {"ratings": {"r0": 5, "r1": 0}})
        self.assertEqual(judge_pitch({"r0": "a", "r1": "b"}, [], model)["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
