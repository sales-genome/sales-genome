from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.graph_embeddings.independent_profile_experiment import (
    FEATURE_SCHEMA_VERSION,
    build_independent_feature_row,
    customer_identity_rejection_reason,
    derive_customer_holdout_split,
    is_case_study_result,
    run_evaluate,
    select_research_cohort,
    validate_canonical_selections,
    validate_checkpoint_configuration,
    validate_research_output,
)
from experiments.graph_embeddings.inductive_baseline import InductiveTwoTower


def test_case_study_filtering_checks_url_title_and_snippet() -> None:
    assert is_case_study_result(
        {
            "url": "https://vendor.example/case-studies/acme",
            "title": "Acme grows",
            "snippet": "A customer result.",
        }
    )
    assert is_case_study_result(
        {
            "url": "https://vendor.example/resources/case-study-acme",
            "title": "Acme grows",
            "snippet": "A customer result.",
        }
    )
    assert is_case_study_result(
        {
            "url": "https://news.example/acme",
            "title": "Acme customer story",
            "snippet": "A success profile.",
        }
    )
    assert is_case_study_result(
        {
            "url": "https://news.example/acme",
            "title": "Acme company update",
            "snippet": "Read this success story about Acme.",
        }
    )
    assert not is_case_study_result(
        {
            "url": "https://news.example/acme-expansion",
            "title": "Acme expands European operations",
            "snippet": "The manufacturer is hiring 500 employees.",
        }
    )


def test_deterministic_cohort_uses_only_named_positive_holdouts() -> None:
    split = {
        "holdout_customers": [1, 2, 3, 4, 5, 6],
        "test_slices": {
            "customer_holdout": [
                (10, 1),
                (11, 2),
                (12, 4),
                (13, 5),
                (14, 6),
            ]
        },
    }
    customer_idx_to_id = {
        1: "customer:one-corp",
        2: "customer:japanese-company",
        3: "customer:three-corp",
        4: "customer:a-leading-manufacturer",
        5: "customer:five-gmbh",
        6: "customer:alpha-systems",
    }
    customer_nodes = {
        "customer:one-corp": {
            "id": "customer:one-corp",
            "type": "customer",
            "name": "One Corp",
        },
        "customer:japanese-company": {
            "id": "customer:japanese-company",
            "type": "customer",
            "name": "株式会社",
        },
        "customer:three-corp": {
            "id": "customer:three-corp",
            "type": "customer",
            "name": "Three Corp",
        },
        "customer:a-leading-manufacturer": {
            "id": "customer:a-leading-manufacturer",
            "type": "customer",
            "name": "A leading manufacturer",
        },
        "customer:five-gmbh": {
            "id": "customer:five-gmbh",
            "type": "customer",
            "name": "Five GmbH",
        },
        "customer:alpha-systems": {
            "id": "customer:alpha-systems",
            "type": "customer",
            "name": "Beta Systems",
        },
    }
    first, first_diagnostics = select_research_cohort(
        split=split,
        customer_idx_to_id=customer_idx_to_id,
        customer_nodes=customer_nodes,
        max_customers=2,
        seed=42,
    )
    second, second_diagnostics = select_research_cohort(
        split=split,
        customer_idx_to_id=customer_idx_to_id,
        customer_nodes=customer_nodes,
        max_customers=2,
        seed=42,
    )

    assert first == second
    assert first_diagnostics == second_diagnostics
    assert {record["customer_index"] for record in first} == {1, 5}
    assert all(record["customer_index"] != 3 for record in first)
    assert first_diagnostics["rejected_by_reason"] == {
        "canonical_name_slug_mismatch": 1,
        "generic_company_description": 1,
        "no_ascii_letters": 1,
        "no_customer_holdout_positive": 1,
    }
    assert all(
        record["identity_validation"]["exact_slug_match"] is True
        for record in first
    )


def test_known_bad_crosslink_merge_fails_exact_slug_identity_check() -> None:
    reason = customer_identity_rejection_reason(
        "customer:3d-systems",
        {
            "id": "customer:3d-systems",
            "type": "customer",
            "name": "Arvato Systems",
        },
    )
    assert reason == "canonical_name_slug_mismatch"


def test_high_support_selection_and_output_normalization() -> None:
    split = {
        "holdout_customers": [1, 2],
        "test_slices": {"customer_holdout": [(10, 1), (11, 2)]},
    }
    customer_idx_to_id = {
        1: "customer:one-corp",
        2: "customer:two-corp",
    }
    customer_nodes = {
        "customer:one-corp": {
            "id": "customer:one-corp",
            "type": "customer",
            "name": "One Corp",
            "n_case_studies": 1,
            "vendors": ["vendor:a"],
        },
        "customer:two-corp": {
            "id": "customer:two-corp",
            "type": "customer",
            "name": "Two Corp",
            "n_case_studies": 5,
            "vendors": ["vendor:a", "vendor:b"],
        },
    }
    cohort, diagnostics = select_research_cohort(
        split=split,
        customer_idx_to_id=customer_idx_to_id,
        customer_nodes=customer_nodes,
        max_customers=1,
        seed=42,
        selection="high_support",
    )
    assert cohort[0]["customer_index"] == 2
    assert diagnostics["selection_strategy"] == "high_support"

    profile = validate_research_output(
        {
            "description": "A software company.",
            "audience_ids": (
                "<item=audience:software>\n"
                "<item=audience:invented-segment>"
            ),
            "contextual_pains": ["Pain one", "Pain two", "Pain three"],
            "confidence": "  High\n",
        },
        valid_audience_ids={"audience:software"},
    )
    assert profile["audience_ids"] == ["audience:software"]
    assert profile["confidence"] == "high"
    assert profile["ignored_invalid_audience_ids"] == [
        "audience:invented-segment"
    ]


def test_canonical_selection_rejects_invented_ids_and_preserves_null() -> None:
    pains = [
        {
            "contextual_pain": "Slow planning",
            "candidates": [
                {"canonical_id": "pai_c_0001"},
                {"canonical_id": "pai_c_0002"},
            ],
        },
        {
            "contextual_pain": "Hiring pressure",
            "candidates": [{"canonical_id": "pai_c_0003"}],
        },
    ]
    valid = validate_canonical_selections(
        pains,
        {
            "selections": [
                {
                    "pain_index": 0,
                    "canonical_id": "pai_c_0002",
                    "reason": "Exact planning match",
                    "confidence": "high",
                },
                {
                    "pain_index": 1,
                    "canonical_id": None,
                    "reason": "No supplied candidate fits",
                    "confidence": "medium",
                },
            ]
        },
    )
    assert valid[0]["selected_canonical_id"] == "pai_c_0002"
    assert valid[1]["matched"] is False

    try:
        validate_canonical_selections(
            pains,
            {
                "selections": [
                    {
                        "pain_index": 0,
                        "canonical_id": "pai_c_invented",
                        "reason": "Hallucinated",
                        "confidence": "high",
                    },
                    {
                        "pain_index": 1,
                        "canonical_id": None,
                        "reason": "No match",
                        "confidence": "low",
                    },
                ]
            },
        )
    except ValueError as exc:
        assert "invented canonical selection" in str(exc)
    else:
        raise AssertionError("invented canonical ID was accepted")


def test_feature_blocks_have_expected_dimensions_and_norms() -> None:
    resolved_record = {
        "profile": {"audience_ids": ["audience:industry"]},
        "resolved_pains": [
            {"selected_canonical_id": "pai_c_0000"},
            {"selected_canonical_id": "pai_c_0001"},
            {"selected_canonical_id": None},
        ],
    }
    rows = build_independent_feature_row(
        resolved_record=resolved_record,
        pain_id_to_index={"pai_c_0000": 0, "pai_c_0001": 1},
        pain_to_subtheme=np.array([0, 1], dtype=np.int64),
        pain_features=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        audience_id_to_index={"audience:industry": 0, "audience:region": 1},
        tfidf_subtheme_idf=np.array([2.0, 1.0], dtype=np.float32),
        semantic_subtheme_idf=np.array([1.0, 3.0], dtype=np.float32),
    )

    expected_dimensions = {
        "audience": 2,
        "semantic_only": 2,
        "tfidf_only": 2,
        "semantic": 4,
        "tfidf": 4,
        "fusion": 6,
    }
    assert {name: values.shape[0] for name, values in rows.items()} == expected_dimensions
    for name in ("audience", "semantic_only", "tfidf_only"):
        assert torch.isclose(torch.linalg.vector_norm(rows[name]), torch.tensor(1.0))
    for name in ("semantic", "tfidf", "fusion"):
        assert torch.isclose(
            torch.linalg.vector_norm(rows[name]),
            torch.tensor(1.0),
            atol=1e-6,
        )
    expected_semantic = torch.tensor([1.0, 3.0])
    expected_semantic /= torch.linalg.vector_norm(expected_semantic)
    assert torch.allclose(rows["semantic_only"], expected_semantic)
    assert math.isclose(float(rows["audience"][0]), 1.0)


def test_split_filters_to_successfully_profiled_customer_holdouts() -> None:
    split = {
        "seed": 42,
        "holdout_customers": [2, 3, 4],
        "train_customers": [0, 1],
        "train_vendors": [0, 1],
        "holdout_vendors": [2],
        "train_edges": [(0, 0), (1, 1)],
        "test_slices": {
            "customer_holdout": [(0, 2), (1, 3), (1, 4)],
            "vendor_holdout": [(2, 0)],
            "double_holdout": [(2, 4)],
        },
    }
    derived, diagnostics = derive_customer_holdout_split(split, {3, 99})

    assert derived["holdout_customers"] == [3]
    assert derived["customer_holdout_candidates"] == [3]
    assert derived["test_slices"]["customer_holdout"] == [(1, 3)]
    assert derived["test_slices"]["vendor_holdout"] == []
    assert derived["test_slices"]["double_holdout"] == []
    assert diagnostics["retained_customer_holdout_customers"] == 1
    assert diagnostics["ignored_successful_profiles"] == [99]
    assert split["holdout_customers"] == [2, 3, 4]


def test_evaluate_loads_oracle_checkpoint_and_emits_aggregate() -> None:
    work_dir = Path(__file__).resolve().parent / ".independent_profile_eval_fixture"
    shutil.rmtree(work_dir, ignore_errors=True)
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    cache_path = work_dir / "derived.pt"
    output_path = work_dir / "evaluation.json"
    source_fingerprint = "base-cache-fingerprint"
    run_name = "oracle_node_holdout__unpruned__audience__seed42"
    fingerprint_payload = {
        "source_cache_fingerprint": source_fingerprint,
    }
    derived_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()

    cache = {
        "fingerprint": derived_fingerprint,
        "source_fingerprint": source_fingerprint,
        "vendor_features": {
            "unpruned": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            )
        },
        "customer_features": {
            "audience": torch.tensor(
                [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            )
        },
        "pairs": [(0, 0), (0, 1), (1, 2)],
        "split": {
            "profile_source": "independent_public_research",
            "train_vendors": [0, 1],
            "holdout_vendors": [],
            "train_customers": [0],
            "holdout_customers": [1, 2],
            "customer_holdout_candidates": [1, 2],
            "train_edges": [(0, 0)],
            "test_slices": {
                "customer_holdout": [(0, 1), (1, 2)],
                "vendor_holdout": [],
                "double_holdout": [],
            },
        },
        "diagnostics": {
            "independent_profiles": {
                "original_customer_holdout_positive_customers": 4,
                "retained_customer_holdout_edges": 2,
                "coverage_fraction": 0.5,
            }
        },
        "independent_profile": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "profiled_customer_indices": [1, 2],
            "fingerprint_payload": fingerprint_payload,
            "provenance": {
                "research_profiles": str(work_dir / "research.jsonl"),
                "resolved_profiles": str(work_dir / "resolved.jsonl"),
                "source_cache": str(work_dir / "base.pt"),
                "hetero_data": str(work_dir / "hetero.pt"),
                "node_mappings": str(work_dir / "node_mappings.json"),
                "pain_features": str(work_dir / "pain_features.npy"),
                "graph_nodes": str(work_dir / "nodes.jsonl"),
            },
            "construction_contract": {
                "test_rows_use_heldout_graph_pains": False,
                "test_rows_use_heldout_graph_audiences": False,
                "training_rows_reused": True,
                "identity_safe_customer_names": True,
            },
        },
    }
    torch.save(cache, cache_path)

    torch.manual_seed(42)
    model = InductiveTwoTower(
        vendor_dim=2,
        customer_dim=2,
        hidden_dim=3,
        output_dim=2,
        dropout=0.0,
    )
    torch.save(model.state_dict(), checkpoint_dir / f"{run_name}.pt")
    training_result = {
        "run_name": run_name,
        "cache_fingerprint": source_fingerprint,
        "configuration": {
            "vendor_feature": "unpruned",
            "customer_feature": "audience",
            "evaluation_protocol": "oracle_node_holdout",
            "seed": 42,
            "hidden_dim": 3,
            "output_dim": 2,
            "dropout": 0.0,
            "batch_size": 2,
        },
    }
    (checkpoint_dir / f"{run_name}.json").write_text(
        json.dumps(training_result),
        encoding="utf-8",
    )

    try:
        result = run_evaluate(
            cache_path=cache_path,
            checkpoint_dir=checkpoint_dir,
            output_path=output_path,
            device_name="cpu",
        )
        assert output_path.exists()
        assert len(result["runs"]) == 1
        metrics = result["runs"][0]["metrics"]
        assert set(metrics) == {
            "map",
            "ndcg_at_50",
            "precision_at_10",
            "chance_precision_at_10",
            "precision_at_10_lift",
            "recall_at_50",
            "profile_coverage",
        }
        assert metrics["profile_coverage"] == 0.5
        aggregate = result["aggregate"]["by_configuration"]
        assert "unpruned__audience" in aggregate
        assert aggregate["unpruned__audience"]["n_runs"] == 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_checkpoint_validation_rejects_provenance_and_dimension_mismatches() -> None:
    model = InductiveTwoTower(
        vendor_dim=2,
        customer_dim=3,
        hidden_dim=4,
        output_dim=2,
        dropout=0.0,
    )
    cache = {
        "vendor_features": {"unpruned": torch.zeros((2, 2))},
        "customer_features": {"audience": torch.zeros((3, 2))},
    }
    run = {
        "checkpoint_path": Path("oracle_node_holdout__unpruned__audience__seed42.pt"),
        "configuration": {
            "vendor_feature": "unpruned",
            "customer_feature": "audience",
            "hidden_dim": 4,
            "output_dim": 2,
        },
        "training_result": {"cache_fingerprint": "wrong-base"},
    }
    try:
        validate_checkpoint_configuration(
            run=run,
            state_dict=model.state_dict(),
            cache=cache,
            source_cache_fingerprint="expected-base",
        )
    except ValueError as exc:
        assert "expected derived source fingerprint" in str(exc)
    else:
        raise AssertionError("checkpoint provenance mismatch was accepted")

    run["training_result"]["cache_fingerprint"] = "expected-base"
    try:
        validate_checkpoint_configuration(
            run=run,
            state_dict=model.state_dict(),
            cache=cache,
            source_cache_fingerprint="expected-base",
        )
    except ValueError as exc:
        assert "customer input dimension" in str(exc)
    else:
        raise AssertionError("checkpoint feature dimension mismatch was accepted")


def _run_direct_tests() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    _run_direct_tests()
