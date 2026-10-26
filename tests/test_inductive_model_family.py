from __future__ import annotations

import hashlib
import json

import torch

from experiments.graph_embeddings.inductive_model_family import (
    BlockSpec,
    InductiveModelFamily,
    _validate_evaluation_cache,
    build_evaluation_definition,
    build_train_only_pair_mask,
    parse_block_specs,
    sample_train_only_negatives,
    validate_block_dimensions,
)


def test_block_dimension_validation() -> None:
    specs = parse_block_specs(["product:4", "capability:3"])
    assert specs == (
        BlockSpec(name="product", dim=4),
        BlockSpec(name="capability", dim=3),
    )
    assert parse_block_specs(["4,3"]) == (
        BlockSpec(name="block_0", dim=4),
        BlockSpec(name="block_1", dim=3),
    )
    assert validate_block_dimensions(specs, 7, tower_name="vendor") == specs

    for invalid in (
        ["product:4", "product:3"],
        ["product:0"],
        ["not-a-name:4"],
    ):
        try:
            parse_block_specs(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid blocks were accepted: {invalid}")

    try:
        validate_block_dimensions(specs, 8, tower_name="vendor")
    except ValueError as exc:
        assert "total 7" in str(exc)
    else:
        raise AssertionError("mismatched block dimensions were accepted")


def test_relation_towers_are_deterministic_normalized_and_shaped() -> None:
    vendor_values = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 1.0, 0.1, 0.2, 0.3],
            [0.5, 0.5, 0.3, 0.0, 0.4],
        ]
    )
    customer_values = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.8],
            [0.0, 1.0, 0.7, 0.3],
            [0.5, 0.5, 0.4, 0.6],
        ]
    )
    for family in ("relation_sage", "relation_attention"):
        torch.manual_seed(123)
        first = InductiveModelFamily(
            family=family,
            vendor_dim=5,
            customer_dim=4,
            vendor_blocks=(BlockSpec("self", 2), BlockSpec("capability", 3)),
            customer_blocks=(BlockSpec("pain", 2), BlockSpec("audience", 2)),
            hidden_dim=6,
            output_dim=4,
            dropout=0.0,
        ).eval()
        torch.manual_seed(123)
        second = InductiveModelFamily(
            family=family,
            vendor_dim=5,
            customer_dim=4,
            vendor_blocks=(BlockSpec("self", 2), BlockSpec("capability", 3)),
            customer_blocks=(BlockSpec("pain", 2), BlockSpec("audience", 2)),
            hidden_dim=6,
            output_dim=4,
            dropout=0.0,
        ).eval()

        first_vendor = first.vendor(vendor_values)
        second_vendor = second.vendor(vendor_values)
        first_customer = first.customer(customer_values)
        second_customer = second.customer(customer_values)
        assert first_vendor.shape == (3, 4)
        assert first_customer.shape == (3, 4)
        assert torch.allclose(first_vendor, second_vendor)
        assert torch.allclose(first_customer, second_customer)
        assert torch.allclose(
            torch.linalg.vector_norm(first_vendor, dim=1),
            torch.ones(3),
            atol=1e-6,
        )
        assert torch.allclose(
            torch.linalg.vector_norm(first_customer, dim=1),
            torch.ones(3),
            atol=1e-6,
        )


def _evaluation_fixture() -> tuple[dict, dict]:
    source_fingerprint = "source-cache-fingerprint"
    train_edges = [(0, 0), (0, 1), (1, 1)]
    source = {
        "fingerprint": source_fingerprint,
        "vendor_features": {"vendor": torch.zeros((3, 4))},
        "customer_features": {
            "customer": torch.arange(24, dtype=torch.float32).reshape(6, 4)
        },
        "pairs": train_edges + [(0, 2), (2, 2), (2, 4)],
        "split": {
            "train_edges": train_edges,
            "train_vendors": [0, 1],
            "holdout_vendors": [2],
            "train_customers": [0, 1],
            "holdout_customers": [2, 3, 4, 5],
            "test_slices": {
                "customer_holdout": [(0, 2), (1, 4)],
                "vendor_holdout": [],
                "double_holdout": [(2, 2), (2, 4)],
            },
        },
    }
    fingerprint_payload = {
        "source_cache_fingerprint": source_fingerprint,
        "profiled_customer_indices": [2, 3],
        "fixture": True,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    evaluation_values = source["customer_features"]["customer"].clone()
    evaluation_values[2] = torch.tensor([9.0, 9.0, 9.0, 9.0])
    evaluation_values[3] = torch.tensor([8.0, 8.0, 8.0, 8.0])
    evaluation = {
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "customer_features": {"customer": evaluation_values},
        "split": {
            "train_edges": train_edges,
            "train_vendors": [0, 1],
            "holdout_vendors": [2],
            "train_customers": [0, 1],
            "holdout_customers": [2, 3],
            "customer_holdout_candidates": [2, 3],
            "profile_source": "independent_public_research",
            "test_slices": {
                "customer_holdout": [(0, 2)],
                "vendor_holdout": [],
                "double_holdout": [],
            },
        },
        "diagnostics": {
            "independent_profiles": {
                "original_customer_holdout_positive_customers": 4,
                "coverage_fraction": 0.5,
            }
        },
        "independent_profile": {
            "fingerprint_payload": fingerprint_payload,
            "profiled_customer_indices": [2, 3],
            "provenance": {
                "research_profiles": "research.jsonl",
                "resolved_profiles": "resolved.jsonl",
                "source_cache": "source.pt",
                "hetero_data": "hetero.pt",
                "node_mappings": "mappings.json",
                "pain_features": "pain.npy",
                "graph_nodes": "nodes.jsonl",
            },
            "construction_contract": {
                "test_rows_use_heldout_graph_pains": False,
                "test_rows_use_heldout_graph_audiences": False,
                "training_rows_reused": True,
                "identity_safe_customer_names": True,
            },
        },
    }
    return source, evaluation


def test_evaluation_definitions_filter_profiled_customers_by_slice() -> None:
    source, evaluation = _evaluation_fixture()
    validated = _validate_evaluation_cache(
        source_cache=source,
        evaluation_cache=evaluation,
        source_fingerprint=source["fingerprint"],
        customer_feature="customer",
        same_artifact=False,
    )
    customer_definition = build_evaluation_definition(
        source_cache=source,
        evaluation_cache=evaluation,
        evaluation_slice="customer_holdout",
        profiled_customers=validated["profiled_customers"],
        profile_coverage=validated["profile_coverage"],
        source_positive_customers=validated["original_positive_customers"],
    )
    double_definition = build_evaluation_definition(
        source_cache=source,
        evaluation_cache=evaluation,
        evaluation_slice="double_holdout",
        profiled_customers=validated["profiled_customers"],
        profile_coverage=validated["profile_coverage"],
        source_positive_customers=validated["original_positive_customers"],
    )

    assert customer_definition.slice_definition.candidate_customers == [2, 3]
    assert customer_definition.slice_definition.edges == [(0, 2)]
    assert double_definition.slice_definition.candidate_customers == [2, 3]
    assert double_definition.slice_definition.edges == [(2, 2)]
    assert double_definition.profile_coverage == 0.5


def test_negative_mask_and_sampling_use_training_labels_only() -> None:
    train_edges = [(0, 0), (0, 1), (1, 2)]
    heldout_edges = [(0, 2), (1, 3)]
    mask = build_train_only_pair_mask(
        train_edges=train_edges,
        n_vendors=2,
        n_customers=4,
        device="cpu",
    )
    assert all(mask[vendor, customer] for vendor, customer in train_edges)
    assert all(not mask[vendor, customer] for vendor, customer in heldout_edges)
    assert int(mask.sum()) == len(train_edges)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    negative_customers, negative_vendors = sample_train_only_negatives(
        vendor_ids=torch.tensor([0, 1]),
        customer_ids=torch.tensor([0, 2]),
        active_vendor_ids=torch.tensor([0, 1]),
        active_customer_ids=torch.tensor([0, 1, 2, 3]),
        known_pair_mask=mask,
        negatives=8,
        generator=generator,
    )
    for row, vendor in enumerate((0, 1)):
        assert not mask[vendor, negative_customers[row]].any()
    for row, customer in enumerate((0, 2)):
        assert not mask[negative_vendors[row], customer].any()


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
