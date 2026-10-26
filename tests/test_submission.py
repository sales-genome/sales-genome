import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_submission.downstream import account_score, retrieve_pitch_context
from paper_submission.repair import apply_identity_decisions, retain_reviewed_relationships
from paper_submission.sample_graph import clean_node, safe_url, select_customers
from paper_submission.temporal import eligible_sources_before


class SamplingTests(unittest.TestCase):
    def test_customer_selection_is_exact_deterministic_and_order_independent(self):
        ids = [f"customer:synthetic-{i}" for i in range(27)]
        self.assertEqual(len(select_customers(ids, "test")), 5)
        self.assertEqual(select_customers(ids, "test"), select_customers(reversed(ids), "test"))
        self.assertNotEqual(select_customers(ids, "test"), select_customers(ids, "other"))

    def test_aggregate_attributes_are_not_carried_by_shared_nodes(self):
        result = clean_node({
            "id": "cap_c_1", "type": "cap_canonical", "canonical_name": "Synthetic concept",
            "vendors": ["not-sampled"], "customers": ["not-sampled"], "n_atoms": 100000,
            "citations": [{"page_body": "PRIVATE FULL PAGE", "path": "PRIVATE MACHINE PATH"}],
            "raw_variants": ["not-sampled"],
        })
        self.assertEqual(set(result), {"id", "type", "canonical_name"})

    def test_private_urls_and_credential_queries_are_rejected(self):
        for url in ["file:///private/file", "https://localhost/a", "http://127.0.0.1/a",
                    "https://10.1.2.3/a", "https://user:pass@example.org/a",
                    "https://example.org/a?token=private", "https://service.internal/a"]:
            self.assertFalse(safe_url(url), url)
        self.assertTrue(safe_url("https://example.org/a?lang=en"))


class RepairTests(unittest.TestCase):
    def test_only_explicit_accepted_same_type_merges_are_applied(self):
        nodes = [{"id": "a", "type": "customer"}, {"id": "b", "type": "customer"},
                 {"id": "v", "type": "vendor"}]
        edges = [{"src": "v", "type": "has_case_study", "dst": "a"}]
        decisions = [{"source_id": "a", "target_id": "b", "decision": "merge", "review_status": "accepted"}]
        output_nodes, output_edges = apply_identity_decisions(nodes, edges, decisions)
        self.assertEqual({row["id"] for row in output_nodes}, {"b", "v"})
        self.assertEqual(output_edges[0]["dst"], "b")
        self.assertEqual(output_edges[0]["input_claims"], edges)
        with self.assertRaises(ValueError):
            apply_identity_decisions(nodes, edges, decisions + [
                {"source_id": "b", "target_id": "a", "decision": "merge", "review_status": "accepted"},
            ])

    def test_unsupported_relationships_are_withheld_not_promoted(self):
        accepted, withheld = retain_reviewed_relationships([{"decision": "unclear"}])
        self.assertEqual(accepted, [])
        self.assertEqual(len(withheld), 1)
        with self.assertRaises(ValueError):
            retain_reviewed_relationships([{"decision": "supported", "review_status": "accepted"}])


class DownstreamTests(unittest.TestCase):
    def test_score_matches_recovered_implementation(self):
        from reference.account_scoring import _score_lead
        for grades in [[], ["INSUFFICIENT"], ["CONFIRMED", "PROBABLE", "INSUFFICIENT"],
                       ["SUGGESTIVE", "PROBABLE", "CONFIRMED", "SUGGESTIVE"]]:
            recovered = _score_lead(
                [{"assessment": {"grade": grade}} for grade in grades], cutoff_date="2020-01-01",
            )
            self.assertEqual(round(account_score(grades)["score"], 1), recovered["composite_score"])
        with self.assertRaises(ValueError):
            account_score(["MADE_UP"])

    def test_prospect_is_not_used_as_peer(self):
        nodes = [{"id": item, "type": kind, "text": "Synthetic pain"} for item, kind in [
            ("v", "vendor"), ("p", "product"), ("c", "cap_canonical"),
            ("pain", "pain_canonical"), ("peer", "customer"), ("target", "customer"),
            ("aud", "audience"),
        ]]
        edges = [{"src": src, "type": relation, "dst": dst} for src, relation, dst in [
            ("v", "offers", "p"), ("p", "provides", "c"), ("c", "addresses", "pain"),
            ("peer", "in_audience", "aud"), ("target", "in_audience", "aud"),
            ("peer", "experiences", "pain"), ("target", "experiences", "pain"),
        ]]
        contexts = retrieve_pitch_context(nodes, edges, "v", {"aud"}, "target")
        self.assertEqual(contexts[0]["peer_customer_ids"], ["peer"])

    def test_exact_date_join_excludes_review_and_non_editorial_dates(self):
        base = {"vendor_slug": "synthetic", "url": "https://example.org/a", "temporal_use": "eligible",
                "date_semantics": "editorial_publication", "reported_published_date": "2020-01-01"}
        result = eligible_sources_before([
            base, {**base, "url": "https://example.org/b", "temporal_use": "review"},
            {**base, "url": "https://example.org/c", "date_semantics": "event_date"},
            {**base, "url": "https://example.org/d", "reported_published_date": "2030-01-01"},
        ], "2021-01-01")
        self.assertEqual(set(result), {("synthetic", "https://example.org/a")})


class EnvironmentTests(unittest.TestCase):
    def test_all_python_sources_compile_without_writing_bytecode(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            compile(path.read_text(), str(path.relative_to(root)), "exec")

    def test_recovered_modules_import_without_service_configuration(self):
        for name in [
            "scripts.build_v4_graph", "scripts.build_v5_graph", "scripts.taxonomy_v4",
            "scripts.taxonomy_v4_chisel", "scripts.ingest.extract_corpus",
            "reference.pitch_point_functions", "reference.account_scoring",
        ]:
            importlib.import_module(name)

    def test_provider_fails_closed_without_opt_in(self):
        from lib.providers import local_tool_call
        with patch.dict("os.environ", {"SUBMISSION_ALLOW_LOCAL_LLM": "0"}):
            with self.assertRaisesRegex(RuntimeError, "Network disabled"):
                local_tool_call(system="", user="", tool_name="synthetic", schema={})

    def test_local_provider_rejects_non_loopback_endpoint(self):
        from lib.providers import _NoRedirect, local_tool_call
        with patch.dict("os.environ", {
            "SUBMISSION_ALLOW_LOCAL_LLM": "1", "SUBMISSION_LOCAL_MODEL": "synthetic",
            "SUBMISSION_LOCAL_BASE_URL": "https://example.org/v1",
        }):
            with self.assertRaisesRegex(ValueError, "loopback"):
                local_tool_call(system="", user="", tool_name="synthetic", schema={})
        with self.assertRaisesRegex(ValueError, "redirects"):
            _NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.org/")

    def test_schema_is_valid(self):
        from jsonschema import Draft202012Validator
        from paper_submission.artifacts import load
        Draft202012Validator.check_schema(load(
            Path(__file__).resolve().parents[1] / "schemas/unified/case_study_v2.json",
        ))
        root = Path(__file__).resolve().parents[1]
        Draft202012Validator.check_schema(load(root / "reference/account-grading-contract.json")["schema"])
        for row in load(root / "reference/repair-output-schemas.json")["schemas"]:
            Draft202012Validator.check_schema(row["schema"])


if __name__ == "__main__":
    unittest.main()
