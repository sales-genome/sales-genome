import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from jsonschema import Draft202012Validator

from paper_submission.canonicalization import project, validate_graph
from paper_submission.composition import compose, identity_policy, public_observation_schema
from paper_submission.construction import (
    ContractError, FINGERPRINT_AXES, ModelFailure, ModelSession, RecordedModel, digest,
    extract_pages, extraction_schema,
)
from paper_submission.induction import (
    InductionSettings, _group, _hierarchy_paths, group_schema, hierarchy_schema, induce, learn_dimension,
)
from paper_submission.pipeline import SyntheticResponseModel, run_pipeline


class ChangedModel:
    def __init__(self, operation, change):
        self.base = SyntheticResponseModel()
        self.settings = self.base.settings
        self.operation, self.change = operation, change

    def call(self, operation, payload, schema):
        response = self.base.call(operation, payload, schema)
        if operation == self.operation:
            self.change(response, payload)
        return response


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path("outputs") / ("core-pipeline-tests-" + uuid.uuid4().hex)
        cls.scratch.mkdir(parents=True, exist_ok=False)
        cls.model = SyntheticResponseModel()
        cls.settings = InductionSettings(sample_size=4, classification_batch=3, group_batch=2,
                                        reconciliation_batch=2, max_leaf_fraction=.6, min_split_size=2)
        cls.session = ModelSession(cls.model, attempts=1)
        cls.observations = extract_pages(cls.model.pages, cls.session)
        cls.raw = compose(cls.observations, aliases=cls.model.aliases)
        cls.learned = induce(cls.raw, cls.session, settings=cls.settings)
        cls.views = project(cls.raw, cls.learned)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch)

    def normalized(self, change):
        observations = copy.deepcopy(self.observations)
        change(observations)
        for observation in observations:
            observation["observation_sha256"] = digest({
                key: value for key, value in observation.items() if key != "observation_sha256"})
        return observations

    def test_full_workflow_replays_identically_and_preserves_all_claims(self):
        recording = self.model.recording()
        replay = RecordedModel(recording)
        result = run_pipeline(self.model.pages, self.scratch / "replay", replay,
                              aliases=self.model.aliases, settings=self.settings, attempts=1)
        self.assertTrue(result["manifest"]["semantic_success"])
        for name in ["raw", "canonical", "hybrid"]:
            self.assertEqual(result["views"][name], self.views[name])
            self.assertTrue((self.scratch / "replay" / name / "nodes.jsonl").is_file())
        self.assertEqual({row["operation"] for row in self.session.receipts}, {
            "extract_observation", "extract_fingerprints", "propose_hierarchy", "refine_hierarchy",
            "assign_hierarchy", "group_canonicals", "reconcile_canonicals",
        })
        self.assertTrue(self.views["report"]["native_claim_ids_preserved"])
        self.assertEqual(len(self.raw["nodes"]), 49)
        self.assertEqual(len(self.raw["edges"]), 96)

    def test_all_four_dimensions_refine_for_coverage_and_balance(self):
        self.assertEqual({row["dimension"] for row in self.learned["dimensions"]},
                         {"pain", "capability", "audience", "strategy"})
        for result in self.learned["dimensions"]:
            self.assertEqual(result["status"], "coverage_and_balance_targets_met")
            self.assertEqual(len(result["rounds"]), 2)
            self.assertLess(result["rounds"][0]["coverage"], 1)
            self.assertTrue(result["rounds"][0]["overloaded_leaves"])
            self.assertEqual(result["rounds"][1]["coverage"], 1)
            self.assertFalse(result["rounds"][1]["overloaded_leaves"])
            self.assertGreater(len(result["mappings"]), 0)

    def test_canonical_cross_chunk_reconciliation_is_semantic_not_exact_string(self):
        capabilities = next(row for row in self.learned["dimensions"] if row["dimension"] == "capability")
        concepts = [row for row in capabilities["nodes"] if row.get("level") == "canonical"]
        self.assertEqual(sorted(len(row["raw_ids"]) for row in concepts), [1, 8, 8])
        self.assertGreater(sum(row["operation"] == "reconcile_canonicals" for row in self.session.receipts), 2)
        raw_by_id = {node["id"]: node for node in self.raw["nodes"]}
        for concept in concepts:
            if len(concept["raw_ids"]) > 1:
                self.assertEqual(len({raw_by_id[identifier]["text"] for identifier in concept["raw_ids"]}), 2)
                self.assertEqual(len({raw_by_id[identifier]["vendor_slug"] for identifier in concept["raw_ids"]}), 4)

    def test_product_attribution_is_not_cartesian_and_unbound_uses_unspecified(self):
        provides = [edge for edge in self.raw["edges"] if edge["type"] == "provides"]
        nodes = {node["id"]: node for node in self.raw["nodes"]}
        self.assertEqual(len(provides), 17)
        for edge in provides:
            capability = nodes[edge["dst"]]["text"]
            product = nodes[edge["src"]]
            if "invoice" in capability.lower():
                self.assertEqual(product["name"], "Synthetic Ledger")
            elif "reports" in capability.lower():
                self.assertTrue(product["id"].endswith(":unspecified"))
            else:
                self.assertEqual(product["name"], "Synthetic Route")
        customers = [node for node in self.raw["nodes"] if node["type"] == "customer"]
        self.assertEqual(len(customers), 4)
        generic = next(node["id"] for node in self.raw["nodes"] if node.get("text", "").startswith("Analysts lacked"))
        self.assertFalse(any(edge["dst"] == generic and edge["type"] == "experiences"
                             for edge in self.raw["edges"]))

    def test_aliases_require_approval_and_preserve_scope(self):
        self.assertIn("customer:synthetic-client-0", {node["id"] for node in self.raw["nodes"]})
        self.assertNotIn("customer:synthetic-client-zero-llc", {node["id"] for node in self.raw["nodes"]})
        with self.assertRaises(ContractError):
            identity_policy({"customer_crosslinks": [{**self.model.aliases["customer_crosslinks"][0], "approved": False}]})
        _, aliases = identity_policy({"product_aliases": [
            {"vendor_slug": "seller", "name": "Alias", "canonical_name": "Middle", "approved": True},
            {"vendor_slug": "seller", "name": "Middle", "canonical_name": "Final", "approved": True},
        ]})
        self.assertEqual(aliases[("seller", "alias")], "Final")
        self.assertNotIn(("other-seller", "alias"), aliases)
        with self.assertRaises(ContractError):
            identity_policy({"product_aliases": [
                {"vendor_slug": "seller", "name": "One", "canonical_name": "Two", "approved": True},
                {"vendor_slug": "seller", "name": "Two", "canonical_name": "One", "approved": True},
            ]})

    def test_slug_collisions_do_not_fuzzy_merge_companies(self):
        observations = self.normalized(lambda rows: [
            rows[index]["data"].update(customer_name=name)
            for index, name in [(0, "Synthetic A.B"), (1, "Synthetic A B")]
        ])
        with self.assertRaises(ContractError):
            compose(observations, aliases=self.model.aliases)

    def test_malformed_or_incomplete_normalized_records_are_rejected(self):
        mutations = [
            lambda rows: rows[0]["support"]["products"].pop(),
            lambda rows: rows[0]["support"]["capability_products"][0].update(product_index=500),
            lambda rows: rows[0]["support"]["pains"][0]["evidence"].update(vendor_slug="other"),
            lambda rows: rows[0]["support"]["pains"][0].update(quote="FORBIDDEN BODY"),
            lambda rows: rows[0]["data"].update(inferred_target_persona="file://synthetic/persona"),
            lambda rows: rows[0]["fingerprints"].clear(),
            lambda rows: rows[0]["fingerprints"][0].update(outcome_index=999),
            lambda rows: rows[0]["fingerprints"][0].update(target_audience_indices=[999]),
        ]
        for change in mutations:
            with self.subTest(change=change):
                with self.assertRaises(ContractError):
                    compose(self.normalized(change), aliases=self.model.aliases)

    def test_extraction_rejects_unsupported_pains_products_and_pairs(self):
        mutations = [
            lambda response, payload: response["support"]["pains"][0].update(role="customer_problem"),
            lambda response, payload: response["data"]["vp_pp_pairs"][0].update(pain_point="Invented pain."),
            lambda response, payload: response["support"]["capability_products"].append({
                "capability_index": 0, "product_index": 100, "quote": payload["content"][:20]}),
            lambda response, payload: response["support"]["pains"][0].update(quote="NOT PRESENT IN THE PAGE"),
        ]
        for change in mutations:
            model = ChangedModel("extract_observation", change)
            with self.subTest(change=change):
                with self.assertRaises(ModelFailure):
                    extract_pages([model.base.pages[-1]], ModelSession(model, attempts=1))

    def test_model_identifiers_cannot_be_injected_as_names_or_mappings(self):
        model = ChangedModel("group_canonicals", lambda response, payload: response["groups"][0].update(
            name="customer:attacker-controlled"))
        with self.assertRaises(ModelFailure):
            learn_dimension(self.raw, "pain", ModelSession(model, attempts=1), self.settings)
        model = ChangedModel("propose_hierarchy", lambda response, payload: response.update(dimension="strategy"))
        with self.assertRaises(ModelFailure):
            learn_dimension(self.raw, "pain", ModelSession(model, attempts=1), self.settings)

    def test_missing_duplicate_unknown_and_noninteger_assignments_fail_closed(self):
        changes = [
            lambda response, payload: response["assignments"].pop(),
            lambda response, payload: response["assignments"].append(copy.deepcopy(response["assignments"][0])),
            lambda response, payload: response["assignments"][0].update(item=999),
            lambda response, payload: response["assignments"][0].update(choice=999),
            lambda response, payload: response["assignments"][0].update(choice=0.0),
        ]
        for change in changes:
            model = ChangedModel("assign_hierarchy", change)
            session = ModelSession(model, attempts=1)
            with self.subTest(change=change):
                with self.assertRaises(ModelFailure):
                    learn_dimension(self.raw, "pain", session, self.settings)
                self.assertEqual(session.receipts[-1]["status"], "rejected")

    def test_canonical_grouping_requires_a_complete_disjoint_partition(self):
        changes = [
            lambda response, payload: response["groups"][0]["members"].pop(),
            lambda response, payload: response["groups"][0]["members"].append(response["groups"][0]["members"][0]),
            lambda response, payload: response["groups"][0]["members"].append(999),
            lambda response, payload: response["groups"][0].update(members=["pain:untrusted"]),
        ]
        for change in changes:
            model = ChangedModel("group_canonicals", change)
            with self.subTest(change=change):
                with self.assertRaises(ModelFailure):
                    learn_dimension(self.raw, "pain", ModelSession(model, attempts=1), self.settings)

    def test_default_failure_is_not_success_and_explicit_fallback_is_labeled(self):
        def empty(response, payload):
            response["assignments"] = []
        model = ChangedModel("assign_hierarchy", empty)
        session = ModelSession(model, attempts=1)
        fallback = learn_dimension(self.raw, "pain", session, replace(
            self.settings, explicit_singleton_fallback=True))
        self.assertEqual(fallback["status"], "explicit_fallback_not_model_success")
        self.assertEqual(len(fallback["mappings"]), 5)
        self.assertTrue(all(row["status"] == "explicit_singleton_fallback" for row in fallback["mappings"]))
        self.assertEqual(session.receipts[-1]["status"], "rejected")
        model = ChangedModel("assign_hierarchy", empty)
        with self.assertRaises(ModelFailure):
            run_pipeline(model.base.pages, self.scratch / "failed", model,
                         aliases=model.base.aliases, settings=self.settings, attempts=1)
        failure = json.loads((self.scratch / "failed" / "failure.json").read_text())
        self.assertEqual(failure["status"], "FAILED_NOT_A_SUCCESSFUL_INDUCTION")
        self.assertFalse((self.scratch / "failed" / "manifest.json").exists())
        self.assertNotIn("content", failure)

    def test_unmapped_atoms_survive_every_projection_at_round_limit(self):
        model = SyntheticResponseModel()
        learned = induce(self.raw, ModelSession(model, attempts=1), settings=replace(self.settings, max_rounds=1))
        self.assertTrue(learned["unmapped"])
        result = project(self.raw, learned)
        canonical_ids = {row["id"] for row in result["canonical"]["nodes"]}
        self.assertTrue(all(row["id"] in canonical_ids for row in learned["unmapped"]))
        self.assertTrue(all(row["status"] == "maximum_rounds_reached" for row in learned["dimensions"]))
        self.assertTrue(result["report"]["native_claim_ids_preserved"])

    def test_no_schema_progress_stops_without_claiming_target_success(self):
        def unchanged(response, payload):
            response.clear()
            response.update(copy.deepcopy(payload["previous"]))
        model = ChangedModel("refine_hierarchy", unchanged)
        learned = learn_dimension(self.raw, "pain", ModelSession(model, attempts=1), self.settings)
        self.assertEqual(learned["status"], "no_schema_progress_targets_not_met")
        self.assertEqual(len(learned["rounds"]), 2)
        self.assertTrue(learned["unmapped"])

    def test_recursive_reconciliation_rejects_partial_responses(self):
        model = ChangedModel("reconcile_canonicals", lambda response, payload: response["groups"][0]["members"].pop())
        with self.assertRaises(ModelFailure):
            learn_dimension(self.raw, "capability", ModelSession(model, attempts=1), self.settings)

    def test_canonical_projection_keeps_exact_source_claims_and_raw_lineage(self):
        raw_claims = {claim["id"]: claim for row in self.raw["edges"] for claim in row["claims"]}
        canonical = self.views["canonical"]
        found_merged_sources = False
        for edge in canonical["edges"]:
            if not edge.get("claims"):
                continue
            exact = {(claim["evidence"]["vendor_slug"], claim["evidence"]["url"]) for claim in edge["claims"]}
            self.assertEqual(exact, {(row["vendor_slug"], row["url"]) for row in edge["source_records"]})
            self.assertEqual({claim["id"] for claim in edge["claims"]},
                             {identifier for row in edge["contributing_raw_edges"] for identifier in row["claim_ids"]})
            for claim in edge["claims"]:
                self.assertEqual(claim, raw_claims[claim["id"]])
                self.assertTrue(all(claim[key] for key in ["raw_src", "raw_type", "raw_dst"]))
            found_merged_sources |= len(exact) >= 4
        self.assertTrue(found_merged_sources)
        native = [row for row in self.views["hybrid"]["edges"] if row["type"] not in {"is_a", "instance_of"}]
        self.assertEqual(native, self.raw["edges"])

    def test_anchor_hashes_and_offsets_match_synthetic_sources_without_bodies(self):
        pages = {(row["vendor_slug"], row["url"]): row for row in self.model.pages}
        for observation in self.observations:
            page = pages[observation["source"]["vendor_slug"], observation["source"]["url"]]
            self.assertEqual(observation["source"]["content_sha256"],
                             hashlib.sha256(page["content"].encode()).hexdigest())
            for role, rows in observation["support"].items():
                records = [rows] if role == "customer" and rows else [] if role == "customer" else rows
                for row in records:
                    anchor = row["evidence"]
                    quote = page["content"][anchor["start"]:anchor["end"]]
                    self.assertEqual(hashlib.sha256(quote.encode()).hexdigest(), anchor["quote_sha256"])
                    self.assertNotIn("quote", row)
            self.assertTrue(all(row["quote"] is None for row in observation["data"]["outcomes"]))
            serialized = json.dumps(observation)
            self.assertNotIn(page["content"], serialized)
        self.assertNotIn('"content":', json.dumps(self.session.receipts))

    def test_unicode_offsets_are_characters_and_hashes_use_utf8(self):
        model = SyntheticResponseModel()
        page = copy.deepcopy(model.pages[0])
        old_body = page["content"]
        page["content"] = "Résumé ✨\n" + old_body
        model.extractions[page["content"]] = copy.deepcopy(model.extractions[old_body])
        observation = extract_pages([page], ModelSession(model, attempts=1))[0]
        evidence = observation["support"]["customer"]["evidence"]
        self.assertEqual(evidence["start"], len("Résumé ✨\n"))
        quoted = page["content"][evidence["start"]:evidence["end"]]
        self.assertEqual(evidence["quote_sha256"], hashlib.sha256(quoted.encode("utf-8")).hexdigest())

    def test_strategy_business_subject_has_no_effect_on_graph_or_semantic_requests(self):
        def alter(rows):
            for observation in rows:
                for fingerprint in observation["fingerprints"]:
                    fingerprint["fingerprint"]["business_subject"] = "A wholly unrelated business topic"
        graph = compose(self.normalized(alter), aliases=self.model.aliases)
        self.assertEqual(graph["nodes"], self.raw["nodes"])
        self.assertEqual(graph["edges"], self.raw["edges"])
        model = SyntheticResponseModel()
        learn_dimension(graph, "strategy", ModelSession(model, attempts=1), self.settings)
        original = [row["request_sha256"] for row in self.model.responses
                    if row["operation"] not in {"extract_observation", "extract_fingerprints"}
                    and row["response"].get("dimension") == "strategy"]
        self.assertEqual([row["request_sha256"] for row in model.responses], original)
        for node in graph["nodes"]:
            if node["type"] == "strategy":
                self.assertEqual(set(node["fingerprint"]), set(FINGERPRINT_AXES))

    def test_projection_rejects_duplicate_wrong_dimension_unknown_or_partial_mappings(self):
        mutations = [
            lambda learned: learned["mappings"].append(copy.deepcopy(learned["mappings"][0])),
            lambda learned: learned["mappings"][0].update(dimension="strategy"),
            lambda learned: learned["mappings"][0].update(dst="canonical:unknown"),
            lambda learned: learned["mappings"].pop(),
            lambda learned: learned["edges"].append(copy.deepcopy(self.raw["edges"][0])),
        ]
        for mutate in mutations:
            learned = copy.deepcopy(self.learned)
            mutate(learned)
            with self.subTest(mutate=mutate):
                with self.assertRaises(ContractError):
                    project(self.raw, learned)

    def test_hierarchy_cycle_duplicate_names_and_depth_are_rejected(self):
        learned = copy.deepcopy(self.learned)
        categories = [row for row in learned["nodes"] if row["dimension"] == "pain" and row["level"] == "category"]
        hierarchy_edge = next(row for row in learned["edges"] if row["src"] in {node["id"] for node in categories})
        learned["edges"].append({**hierarchy_edge, "src": hierarchy_edge["dst"], "dst": hierarchy_edge["src"]})
        with self.assertRaises(ContractError):
            project(self.raw, learned)
        hierarchy = copy.deepcopy(self.learned["dimensions"][0]["hierarchy"])
        hierarchy["domains"].append(copy.deepcopy(hierarchy["domains"][0]))
        with self.assertRaises(ContractError):
            _hierarchy_paths(hierarchy, "pain", 6)
        with self.assertRaises(ContractError):
            _hierarchy_paths(self.learned["dimensions"][0]["hierarchy"], "pain", 2)

    def test_recording_request_and_schema_are_exact_and_unused_rows_are_errors(self):
        row = self.model.responses[0]
        replay = RecordedModel(self.model.recording())
        with self.assertRaises(ModelFailure):
            replay.call(row["operation"], {"different": "request"}, extraction_schema())
        self.assertEqual(replay.position, 0)
        with self.assertRaises(ContractError):
            replay.assert_consumed()
        replay = RecordedModel({"responses": [{
            "operation": "synthetic", "request_sha256": digest({}),
            "schema_sha256": digest({"type": "object"}), "response": {},
        }]})
        with self.assertRaises(ModelFailure):
            replay.call("synthetic", {}, {"type": "array"})

    def test_retry_records_each_failure_and_usage_is_sanitized(self):
        class RetryModel:
            settings = {"backend": "synthetic"}
            count = 0
            last_usage = {"prompt_tokens": 4, "content": "FORBIDDEN BODY",
                          "private_path": "synthetic/unused-provider-location"}

            def call(self, operation, payload, schema):
                self.count += 1
                return {"ok": self.count > 1}
        model, receipts = RetryModel(), []
        session = ModelSession(model, attempts=2, receipt_callback=lambda rows: receipts.append(copy.deepcopy(rows)))
        value, receipt = session.ask("synthetic", {}, {"type": "object", "properties": {"ok": {"const": True}}})
        self.assertEqual(value, {"ok": True})
        self.assertEqual(receipt, 1)
        self.assertEqual([row["status"] for row in session.receipts], ["rejected", "accepted"])
        self.assertEqual(session.receipts[-1]["usage"], {"prompt_tokens": 4})
        self.assertEqual(len(receipts), 2)
        self.assertNotIn("FORBIDDEN BODY", json.dumps(session.receipts))

    def test_audience_attribute_type_cannot_be_merged_by_a_model(self):
        class MergeEverything:
            settings = {"backend": "synthetic"}

            def call(self, operation, payload, schema):
                return {"dimension": "audience", "groups": [{
                    "name": "Combined", "scope": "Wrong scope", "discriminator": "Wrong merge",
                    "members": [0, 1],
                }]}
        items = [{"name": kind, "scope": kind, "discriminator": kind, "members": [kind],
                  "attribute_types": [kind], "representatives": [kind], "receipts": []}
                 for kind in ["industry", "location"]]
        with self.assertRaises(ModelFailure):
            _group(items, "audience", ModelSession(MergeEverything(), attempts=1),
                   operation="group_canonicals", path=("test",))

    def test_schemas_compile_and_configuration_is_bounded(self):
        for dimension in ["pain", "capability", "audience", "strategy"]:
            Draft202012Validator.check_schema(hierarchy_schema(dimension))
            Draft202012Validator.check_schema(group_schema(dimension))
        Draft202012Validator.check_schema(public_observation_schema())
        for change in [{"max_rounds": 0}, {"reconciliation_batch": 1}, {"coverage_target": float("nan")},
                       {"max_depth": 20}, {"max_leaf_fraction": -1}, {"coverage_target": True},
                       {"seed": "file://synthetic/seed"}, {"explicit_singleton_fallback": "yes"}]:
            with self.assertRaises(ContractError):
                InductionSettings(**change)
        from paper_submission.construction import LocalModel
        with patch.dict("os.environ", {"SUBMISSION_ALLOW_LOCAL_LLM": "0"}):
            with self.assertRaises(RuntimeError):
                LocalModel()

    def test_configured_local_backend_uses_actual_structured_transport_contract(self):
        from paper_submission.construction import LocalModel
        with patch.dict("os.environ", {"SUBMISSION_ALLOW_LOCAL_LLM": "1", "SUBMISSION_LOCAL_MODEL": "synthetic-test"}):
            with patch("lib.providers.local_tool_call", return_value=({"ok": True}, {"total_tokens": 5})) as call:
                model = LocalModel(max_tokens=2048)
                result = model.call("synthetic", {"instruction": "Test only"}, {"type": "object"})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call.call_args.kwargs["tool_name"], "emit_synthetic")
        self.assertEqual(call.call_args.kwargs["max_tokens"], 2048)
        self.assertEqual(model.last_usage["total_tokens"], 5)
        self.assertEqual(model.settings["model"], "synthetic-test")

    def test_private_text_and_nonpublic_urls_are_not_persisted(self):
        from paper_submission.construction import public_url, text
        for value in ["file://synthetic/fixture", "file:/synthetic-fixture", "http://10.0.0.1:8000/v1",
                      "C:\\private\\cache", "password: secret-value", "API token at 192.168.1.2"]:
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    text(value)
        for value in ["https://localhost./a", "https://intranet/a", "https://example.org/?access_token=secret"]:
            self.assertFalse(public_url(value))
        self.assertTrue(public_url("https://example.org/public-page"))

    def test_output_directory_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            run_pipeline([], self.scratch, SyntheticResponseModel(), settings=self.settings)


if __name__ == "__main__":
    unittest.main()
