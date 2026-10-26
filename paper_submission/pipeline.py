"""Executable source extraction -> composition -> induction -> graph projections."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

from paper_submission.artifacts import lines, load, save, sha256, write_lines
from paper_submission.canonicalization import project
from paper_submission.composition import compose
from paper_submission.construction import (
    ContractError, FINGERPRINT_AXES, LocalModel, ModelSession, RecordedModel, digest, extract_pages,
)
from paper_submission.induction import InductionSettings, induce

PIPELINE_VERSION = "executable-construction-to-canonical-1"


def _save_manifest(destination, manifest):
    manifest["artifacts"] = {
        str(path.relative_to(destination)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }
    save(destination / "manifest.json", manifest)


def run_pipeline(pages, destination, model, *, aliases=None, settings=None, attempts=2):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Pipeline outputs must be new: " + str(destination))
    destination.mkdir(parents=True, exist_ok=False)
    settings = settings or InductionSettings()
    session = ModelSession(model, attempts=attempts,
                           receipt_callback=lambda receipts: save(destination / "model-receipts.json", receipts))
    try:
        save(destination / "model-receipts.json", [])
        observations = extract_pages(pages, session)
        if not observations:
            raise ContractError("At least one supplied source page is required.")
        write_lines(destination / "observations.jsonl", observations)
        raw = compose(observations, aliases=aliases)
        learned = induce(raw, session, settings=settings)
        views = project(raw, learned)
        if isinstance(model, RecordedModel):
            model.assert_consumed()
        write_lines(destination / "sources.jsonl", raw["sources"])
        write_lines(destination / "mappings.jsonl", learned["mappings"])
        write_lines(destination / "unmapped.jsonl", learned["unmapped"])
        save(destination / "induction.json", learned)
        save(destination / "graph-validation.json", views["report"])
        for name in ["raw", "canonical", "hybrid"]:
            write_lines(destination / name / "nodes.jsonl", views[name]["nodes"])
            write_lines(destination / name / "edges.jsonl", views[name]["edges"])
        manifest = {
            "version": PIPELINE_VERSION, "historical_experiment_reproduction": False,
            "settings": asdict(settings), "identity_policy_sha256": digest(aliases or {}),
            "semantic_success": all(result["status"] in {
                "coverage_and_balance_targets_met", "empty_dimension",
            } for result in learned["dimensions"]),
            "model": session.model_settings, "receipts": len(session.receipts),
            "source_records": len(raw["sources"]),
            "dimensions": {result["dimension"]: {
                "status": result["status"], "rounds": result["rounds"],
                "mappings": len(result["mappings"]), "unmapped": len(result["unmapped"]),
            } for result in learned["dimensions"]},
            "views": {name: {"nodes": len(views[name]["nodes"]), "edges": len(views[name]["edges"])}
                      for name in ["raw", "canonical", "hybrid"]},
            "provenance_contract": "edge.source_records indexes exact vendor_slug/url; edge.claims retains "
                                   "per-source evidence and raw_src/raw_type/raw_dst; no source bodies persisted",
            "input_observation_sha256": digest(observations),
            "code_sha256": {name: sha256(Path(__file__).with_name(name + ".py"))
                            for name in ["construction", "composition", "induction", "canonicalization", "pipeline"]},
        }
        _save_manifest(destination, manifest)
        return {"manifest": manifest, "views": views, "induction": learned, "observations": observations,
                "receipts": session.receipts}
    except Exception as error:
        save(destination / "failure.json", {
            "version": PIPELINE_VERSION, "status": "FAILED_NOT_A_SUCCESSFUL_INDUCTION",
            "error_class": type(error).__name__, "accepted_model_calls": sum(
                row["status"] == "accepted" for row in session.receipts),
            "rejected_model_calls": sum(row["status"] != "accepted" for row in session.receipts),
            "source_bodies_copied": False,
        })
        raise


def synthetic_fixture():
    """Authored source/response examples; no real corpus records or graph overlays."""
    pages, extractions, fingerprints, semantics = [], {}, {}, {}
    cap = [
        "Automatically route incoming support tickets.",
        "Assign new service requests without manual sorting.",
        "Capture supplier invoices automatically.",
        "Digitize incoming invoices without manual entry.",
    ]
    pains = [
        "Staff manually sorted support tickets, delaying responses.",
        "Manual support-request triage delayed replies.",
        "Manual invoice entry caused processing delays.",
        "Staff retyped supplier bills, slowing finance operations.",
    ]
    styles = [
        ("Numerical improvement", "Before and after measurements", "Confident", "Confidence", "Measured result"),
        ("Risk reassurance", "Reported elimination of errors", "Reassuring", "Safety", "Customer experience"),
        ("Quantified progress", "Comparison of measured outcomes", "Assured", "Confidence", "Measured result"),
        ("De-risking reassurance", "Reported removal of errors", "Calm", "Safety", "Customer experience"),
    ]
    for index, value in enumerate(cap):
        semantics[value] = ("ticket" if index < 2 else "invoice", "ticket" if index < 2 else "invoice")
    for index, value in enumerate(pains):
        semantics[value] = ("ticket" if index < 2 else "invoice", "ticket" if index < 2 else "invoice")
    for variant, values in enumerate(styles):
        matching = " | ".join(axis + ": " + value for axis, value in zip(FINGERPRINT_AXES, values))
        semantics[matching] = ("quantified" if variant % 2 == 0 else "reassurance",
                               "quantified" if variant % 2 == 0 else "reassurance")
    for value in ["Manufacturing", "Industrial manufacturing"]:
        semantics["industry: " + value] = ("industry", "manufacturing")
    for value in ["United States", "USA"]:
        semantics["location: " + value] = ("location", "united_states")
    semantics["industry: Government agencies"] = ("industry", "government")

    def record(vendor_index, scenario):
        vendor = "synthetic-" + chr(ord("a") + vendor_index)
        customer = "Synthetic Client " + str(scenario)
        if vendor_index == 1 and scenario == 0:
            customer = "Synthetic Client Zero LLC"
        product_index = 0 if scenario < 2 else 1
        products = ["Synthetic Route", "Synthetic Ledger"]
        subject = customer + " is the subject of this synthetic customer story."
        offered = "Synthetic Route and Synthetic Ledger are available offerings."
        binding = products[product_index] + " enables this function: " + cap[scenario]
        industry = "Manufacturing" if scenario % 2 == 0 else "Industrial manufacturing"
        location = "United States" if scenario % 2 == 0 else "USA"
        industry_quote, location_quote = "Industry: " + industry + ".", "Location: " + location + "."
        business_subject = "Ticket operations" if scenario < 2 else "Invoice operations"
        outcome = business_subject + (" processing time fell by 40 percent." if scenario % 2 == 0
                                      else " errors were eliminated, lowering operational risk.")
        body = "\n".join([subject, offered, binding, pains[scenario], industry_quote, location_quote, outcome])
        page = {"vendor_slug": vendor, "vendor_name": vendor.title(), "page_type": "case_study",
                "url": f"https://example.org/{vendor}/case-{scenario}", "content": body,
                "published_date": "2020-01-01"}
        data = {
            "customer_name": customer, "inferred_target_persona": None,
            "audience_dimensions": {"industry": industry, "company_size": None, "location": location,
                                    "revenue": None, "tech_stack": [], "other": None},
            "vendor_products_used": products, "capabilities_cited": [cap[scenario]],
            "pain_points_cited": [pains[scenario]],
            "vp_pp_pairs": [{"capability": cap[scenario], "pain_point": pains[scenario], "confidence": "high"}],
            "outcomes": [{"metric": "Synthetic outcome", "value": "Synthetic fixture only", "quote": outcome}],
            "confidence": "high",
        }
        result = {"data": data, "support": {
            "customer": {"quote": subject, "is_case_subject": True},
            "products": [{"index": i, "quote": offered} for i in range(2)],
            "capabilities": [{"index": 0, "quote": binding}],
            "pains": [{"index": 0, "quote": pains[scenario], "role": "customer_problem"}],
            "pairs": [{"index": 0, "quote": binding + "\n" + pains[scenario]}],
            "outcomes": [{"index": 0, "quote": outcome}],
            "audiences": [
                {"attribute": "industry", "value": industry, "quote": industry_quote, "customer_supported": True},
                {"attribute": "location", "value": location, "quote": location_quote, "customer_supported": True},
            ],
            "capability_products": [{"capability_index": 0, "product_index": product_index, "quote": binding}],
        }}
        fp = dict(zip(FINGERPRINT_AXES, styles[scenario]))
        fp["business_subject"] = business_subject
        fingerprints[outcome] = {"outcome_index": 0, "persuasive": True,
                                 "claim": outcome, "fingerprint": fp, "cites_case_customer": True,
                                 "target_audience_indices": [0]}
        pages.append(page)
        extractions[body] = result

    for vendor_index in range(4):
        for scenario in range(4):
            record(vendor_index, scenario)
    reporting_cap = "Export consistent audit reports."
    reporting_pain = "Analysts lacked consistent reporting across review cycles."
    semantics[reporting_cap] = ("reporting", "reporting")
    semantics[reporting_pain] = ("reporting", "reporting")
    body = "\n".join(["Synthetic Reports is an available offering.", reporting_cap, reporting_pain,
                      "Industry: Government agencies."])
    page = {"vendor_slug": "synthetic-a", "vendor_name": "Synthetic-A", "page_type": "product",
            "url": "https://example.org/synthetic-a/reporting", "content": body}
    pages.append(page)
    extractions[body] = {
        "data": {
            "customer_name": None, "inferred_target_persona": None,
            "audience_dimensions": {"industry": "Government agencies", "company_size": None, "location": None,
                                    "revenue": None, "tech_stack": [], "other": None},
            "vendor_products_used": ["Synthetic Reports"], "capabilities_cited": [reporting_cap],
            "pain_points_cited": [reporting_pain],
            "vp_pp_pairs": [{"capability": reporting_cap, "pain_point": reporting_pain, "confidence": "high"}],
            "outcomes": [], "confidence": "high",
        },
        "support": {
            "customer": None, "products": [{"index": 0, "quote": "Synthetic Reports is an available offering."}],
            "capabilities": [{"index": 0, "quote": reporting_cap}],
            "pains": [{"index": 0, "quote": reporting_pain, "role": "general_problem"}],
            "pairs": [{"index": 0, "quote": reporting_cap + "\n" + reporting_pain}], "outcomes": [],
            "audiences": [{"attribute": "industry", "value": "Government agencies",
                           "quote": "Industry: Government agencies.", "customer_supported": False}],
            "capability_products": [],
        },
    }
    aliases = {"customer_crosslinks": [{
        "name": "Synthetic Client Zero LLC", "customer_id": "customer:synthetic-client-0",
        "canonical_name": "Synthetic Client 0", "approved": True,
    }]}
    return pages, aliases, extractions, fingerprints, semantics


class SyntheticResponseModel:
    """Authored semantic response fixture, NOT a general-purpose clustering model.

    Every response is logged with its exact request digest and can then be replayed
    through RecordedModel. No ready-made graph, mapping, or overlay is supplied.
    """
    def __init__(self):
        self.pages, self.aliases, self.extractions, self.fingerprints, self.semantics = synthetic_fixture()
        self.responses = []
        self.settings = {"backend": "synthetic_recorded_responses", "model": "authored-semantic-fixture-v1",
                         "live_calls": False, "synthetic": True}

    def _meaning(self, label):
        if label not in self.semantics:
            raise ContractError("Unknown synthetic fixture description; this backend cannot infer new meanings.")
        return self.semantics[label]

    def call(self, operation, payload, schema):
        if operation == "extract_observation":
            response = copy.deepcopy(self.extractions[payload["content"]])
        elif operation == "extract_fingerprints":
            response = {"fingerprints": [copy.deepcopy(self.fingerprints[row["quote"]])
                                         for row in payload["outcomes"]]}
        elif operation in {"propose_hierarchy", "refine_hierarchy"}:
            dimension = payload["dimension"]
            if operation == "propose_hierarchy":
                children = [{"name": "Initial broad slot", "description": "Initial synthetic semantic slot.",
                             "children": []}]
            else:
                categories = {
                    "pain": ["ticket", "invoice", "reporting"],
                    "capability": ["ticket", "invoice", "reporting"],
                    "audience": ["industry", "location"],
                    "strategy": ["quantified", "reassurance"],
                }[dimension]
                children = [{"name": "Refined meanings", "description": "Force-split overloaded semantic branch.",
                             "children": [{"name": category, "description": "Synthetic " + category + " descriptions.",
                                           "children": []} for category in categories]}]
            response = {"dimension": dimension, "domains": [{
                "name": "Synthetic " + dimension + " domain", "description": "Authored synthetic semantic responses.",
                "children": children,
            }]}
        elif operation == "assign_hierarchy":
            options, dimension = payload["options"], payload["dimension"]
            initial = len(options) == 1 and options[0]["path"][-1] == "Initial broad slot"
            preferred = {"pain": "ticket", "capability": "ticket", "audience": "industry",
                         "strategy": "quantified"}[dimension]
            assignments = []
            for row in payload["items"]:
                category, _ = self._meaning(row["text"])
                if initial:
                    choice = 0 if category == preferred else -1
                else:
                    choice = next(option["choice"] for option in options if option["path"][-1] == category)
                assignments.append({"item": row["index"], "choice": choice})
            response = {"dimension": dimension, "assignments": assignments}
        elif operation in {"group_canonicals", "reconcile_canonicals"}:
            groups = {}
            for row in payload["items"]:
                meanings = {self._meaning(label)[1] for label in row["representative_texts"]}
                if len(meanings) != 1:
                    raise ContractError("Fixture prototype contains inconsistent semantic meanings.")
                meaning = meanings.pop()
                groups.setdefault(meaning, []).append(row["index"])
            response = {"dimension": payload["dimension"], "groups": [{
                "name": "Synthetic " + payload["dimension"] + " " + meaning,
                "scope": "Equivalent synthetic " + meaning + " meaning only.",
                "discriminator": "Distinct from other synthetic meanings.",
                "members": indices,
            } for meaning, indices in sorted(groups.items())]}
        else:
            raise ContractError("Unknown synthetic fixture operation.")
        self.responses.append({"operation": operation, "request_sha256": digest(payload), "schema_sha256": digest(schema),
                               "response": copy.deepcopy(response)})
        return response

    def recording(self):
        return {"model": self.settings["model"], "synthetic": True, "responses": self.responses}


def demo(destination):
    model = SyntheticResponseModel()
    settings = InductionSettings(sample_size=4, classification_batch=3, group_batch=2,
                                 reconciliation_batch=2, max_rounds=4, max_leaf_fraction=.6, min_split_size=2)
    result = run_pipeline(model.pages, destination, model, aliases=model.aliases, settings=settings, attempts=1)
    save(Path(destination) / "synthetic-recording.json", model.recording())
    save(Path(destination) / "demo-inputs.json", {"synthetic": True, "pages": model.pages, "aliases": model.aliases})
    save(Path(destination) / "demo-report.json", {
        "synthetic": True, "live_calls": False, "manually_supplied_graph_overlays": False,
        "all_four_dimensions_executed": all(
            result["manifest"]["dimensions"][name]["mappings"] > 0
            for name in ["pain", "capability", "audience", "strategy"]),
        "operations_executed": sorted({row["operation"] for row in result["receipts"]}),
        "views": result["manifest"]["views"],
        "status": "PASS" if result["views"]["report"]["native_claim_ids_preserved"] else "FAIL",
    })
    _save_manifest(Path(destination), result["manifest"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("demo", help="Full deterministic synthetic recorded-response pipeline")
    sample.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="Extract supplied local page records with configured local model or strict recording")
    run.add_argument("--pages", type=Path, required=True, help="JSONL source records with inline in-memory content")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--backend", choices=["local", "recorded"], required=True)
    run.add_argument("--recording", type=Path)
    run.add_argument("--aliases", type=Path)
    run.add_argument("--settings", type=Path, help="JSON InductionSettings overrides")
    run.add_argument("--attempts", type=int, default=2)
    run.add_argument("--max-tokens", type=int, default=12000)
    replay = sub.add_parser("replay-demo", help="Replay the exact synthetic demo requests from its saved recording")
    replay.add_argument("--demo", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        result = demo(args.out)
    elif args.command == "replay-demo":
        inputs, manifest = load(args.demo / "demo-inputs.json"), load(args.demo / "manifest.json")
        result = run_pipeline(
            inputs["pages"], args.out, RecordedModel(args.demo / "synthetic-recording.json"),
            aliases=inputs["aliases"], settings=InductionSettings(**manifest["settings"]), attempts=1,
        )
        for view in ["raw", "canonical", "hybrid"]:
            for table in ["nodes.jsonl", "edges.jsonl"]:
                if sha256(args.out / view / table) != sha256(args.demo / view / table):
                    raise ContractError("Recorded replay changed a graph artifact.")
    else:
        if args.backend == "recorded" and not args.recording:
            parser.error("--recording is required for recorded mode")
        model = RecordedModel(args.recording) if args.backend == "recorded" else LocalModel(max_tokens=args.max_tokens)
        result = run_pipeline(
            list(lines(args.pages)), args.out, model, aliases=load(args.aliases) if args.aliases else None,
            settings=InductionSettings(**load(args.settings)) if args.settings else None, attempts=args.attempts,
        )
    print(json.dumps({"status": "COMPLETE", "out": str(args.out), "views": result["manifest"]["views"],
                      "dimensions": {name: row["status"] for name, row in result["manifest"]["dimensions"].items()}},
                     indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error_class": type(error).__name__,
                          "details": "Inspect redacted failure.json/model-receipts.json in the new output directory."}))
        raise SystemExit(1) from None
