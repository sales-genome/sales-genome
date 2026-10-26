#!/usr/bin/env python3
"""Train inductive two-tower model families on declared relation blocks.

``relation_sage`` and ``relation_attention`` are directional one-hop message
aggregation over already pre-aggregated relation-neighborhood feature blocks.
They do not materialize symmetric message passing over the million-node graph
and are not the old transductive symmetric R-GCN.

Training deliberately reuses the leakage-safe behavior from
``inductive_baseline``: a train/validation split within source-cache training
edges, symmetric InfoNCE, fixed validation negatives, and negative rejection
against training labels only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.graph_embeddings.inductive_baseline import (  # noqa: E402
    SliceDefinition,
    evaluate_slice,
)


GRAPH_DIR = ROOT / "experiments" / "graph_embeddings"
DEFAULT_OUTPUT = GRAPH_DIR / "results" / "inductive_model_family"
MODEL_FAMILIES = (
    "linear",
    "mlp",
    "relation_sage",
    "relation_attention",
)
EVALUATION_SLICES = ("customer_holdout", "double_holdout")
MODEL_INTERPRETATION = (
    "Directional one-hop message aggregation over pre-aggregated relation "
    "blocks. No million-node symmetric message passing is materialized; this "
    "is not the old transductive symmetric R-GCN."
)
_BLOCK_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_REQUIRED_PROFILE_PROVENANCE = {
    "research_profiles",
    "resolved_profiles",
    "source_cache",
    "hetero_data",
    "node_mappings",
    "pain_features",
    "graph_nodes",
}


@dataclass(frozen=True)
class BlockSpec:
    """Named contiguous input block representing one relation neighborhood."""

    name: str
    dim: int


@dataclass(frozen=True)
class EvaluationDefinition:
    """Evaluation slice plus independently profiled population metadata."""

    slice_definition: SliceDefinition
    profiled_customers: list[int]
    profile_coverage: float
    source_positive_customers: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_block_specs(values: Sequence[str]) -> tuple[BlockSpec, ...]:
    """Parse named blocks or plain dimensions in their given order.

    Plain dimensions receive stable ``block_N`` names, so both
    ``product:1024 capability:1024`` and ``1024,1024`` are accepted.
    """
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    if not tokens:
        raise ValueError("at least one relation block must be declared")

    specs: list[BlockSpec] = []
    seen: set[str] = set()
    for index, token in enumerate(tokens):
        try:
            if ":" in token:
                name, raw_dim = token.rsplit(":", 1)
            else:
                name, raw_dim = f"block_{index}", token
            dim = int(raw_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid relation block {token!r}; expected name:positive_dim"
            ) from exc
        name = name.strip()
        if not _BLOCK_NAME.fullmatch(name):
            raise ValueError(
                f"invalid relation block name {name!r}; use letters, digits, "
                "and underscores, beginning with a letter"
            )
        if name in seen:
            raise ValueError(f"duplicate relation block name: {name}")
        if dim <= 0:
            raise ValueError(f"relation block {name!r} must have positive dimension")
        seen.add(name)
        specs.append(BlockSpec(name=name, dim=dim))
    return tuple(specs)


def validate_block_dimensions(
    specs: Sequence[BlockSpec],
    input_dim: int,
    *,
    tower_name: str,
) -> tuple[BlockSpec, ...]:
    """Validate unique positive blocks that exactly partition an input row."""
    normalized = tuple(specs)
    if not normalized:
        raise ValueError(f"{tower_name} must declare at least one relation block")
    names = [spec.name for spec in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"{tower_name} relation block names must be unique")
    if any(not _BLOCK_NAME.fullmatch(name) for name in names):
        raise ValueError(f"{tower_name} relation block names are invalid")
    invalid = [spec for spec in normalized if spec.dim <= 0]
    if invalid:
        raise ValueError(f"{tower_name} relation block dimensions must be positive")
    declared_dim = sum(spec.dim for spec in normalized)
    if declared_dim != input_dim:
        raise ValueError(
            f"{tower_name} relation blocks total {declared_dim}, but feature "
            f"dimension is {input_dim}"
        )
    return normalized


def split_relation_blocks(
    values: torch.Tensor,
    specs: Sequence[BlockSpec],
) -> tuple[torch.Tensor, ...]:
    """Split a feature matrix according to its validated contiguous blocks."""
    validated = validate_block_dimensions(
        specs,
        int(values.shape[-1]),
        tower_name="input",
    )
    return tuple(torch.split(values, [spec.dim for spec in validated], dim=-1))


class LinearTower(nn.Module):
    """One normalized linear projection."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(values), p=2, dim=-1, eps=1e-8)


class MLPTower(nn.Module):
    """The existing two-layer GELU/dropout tower behavior."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(values), p=2, dim=-1, eps=1e-8)


class RelationSageTower(nn.Module):
    """Directional one-hop GraphSAGE over pre-aggregated relation blocks."""

    def __init__(
        self,
        block_specs: Sequence[BlockSpec],
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_specs = tuple(block_specs)
        self.relation_transforms = nn.ModuleList(
            nn.Linear(spec.dim, hidden_dim) for spec in self.block_specs
        )
        self.relation_weights = nn.Parameter(
            torch.ones(len(self.block_specs), dtype=torch.float32)
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        blocks = split_relation_blocks(values, self.block_specs)
        transformed = torch.stack(
            [
                transform(block) * self.relation_weights[index]
                for index, (transform, block) in enumerate(
                    zip(self.relation_transforms, blocks, strict=True)
                )
            ],
            dim=0,
        ).sum(dim=0)
        output = self.output_projection(self.dropout(F.gelu(transformed)))
        return F.normalize(output, p=2, dim=-1, eps=1e-8)


class RelationAttentionTower(nn.Module):
    """Sample-specific softmax gating across relation-neighborhood blocks."""

    def __init__(
        self,
        block_specs: Sequence[BlockSpec],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_specs = tuple(block_specs)
        self.relation_projections = nn.ModuleList(
            nn.Linear(spec.dim, output_dim) for spec in self.block_specs
        )
        self.relation_gates = nn.ModuleList(
            nn.Linear(spec.dim, 1) for spec in self.block_specs
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        blocks = split_relation_blocks(values, self.block_specs)
        projected = torch.stack(
            [
                projection(block)
                for projection, block in zip(
                    self.relation_projections,
                    blocks,
                    strict=True,
                )
            ],
            dim=1,
        )
        raw_gates = torch.cat(
            [
                gate(block)
                for gate, block in zip(
                    self.relation_gates,
                    blocks,
                    strict=True,
                )
            ],
            dim=1,
        )
        weights = torch.softmax(raw_gates, dim=1).unsqueeze(-1)
        output = (self.dropout(projected) * weights).sum(dim=1)
        return F.normalize(output, p=2, dim=-1, eps=1e-8)


def build_tower(
    *,
    family: str,
    input_dim: int,
    block_specs: Sequence[BlockSpec],
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    """Build one tower after validating the declared relation partition."""
    specs = validate_block_dimensions(
        block_specs,
        input_dim,
        tower_name=f"{family} tower",
    )
    if family == "linear":
        return LinearTower(input_dim, output_dim)
    if family == "mlp":
        return MLPTower(input_dim, hidden_dim, output_dim, dropout)
    if family == "relation_sage":
        return RelationSageTower(specs, hidden_dim, output_dim, dropout)
    if family == "relation_attention":
        return RelationAttentionTower(specs, output_dim, dropout)
    raise ValueError(f"unknown model family: {family}")


class InductiveModelFamily(nn.Module):
    """Vendor/customer towers sharing a family but not parameters."""

    def __init__(
        self,
        *,
        family: str,
        vendor_dim: int,
        customer_dim: int,
        vendor_blocks: Sequence[BlockSpec],
        customer_blocks: Sequence[BlockSpec],
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.family = family
        self.vendor = build_tower(
            family=family,
            input_dim=vendor_dim,
            block_specs=vendor_blocks,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.customer = build_tower(
            family=family,
            input_dim=customer_dim,
            block_specs=customer_blocks,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


def _normalized_edges(edges: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    return sorted({(int(edge[0]), int(edge[1])) for edge in edges})


def build_train_only_pair_mask(
    *,
    train_edges: Iterable[Sequence[int]],
    n_vendors: int,
    n_customers: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a rejection mask from training labels and no heldout labels."""
    edges = _normalized_edges(train_edges)
    if not edges:
        raise ValueError("training edges are empty")
    tensor = torch.tensor(edges, dtype=torch.long, device=device)
    if int(tensor[:, 0].min()) < 0 or int(tensor[:, 0].max()) >= n_vendors:
        raise ValueError("training edge vendor index is out of range")
    if int(tensor[:, 1].min()) < 0 or int(tensor[:, 1].max()) >= n_customers:
        raise ValueError("training edge customer index is out of range")
    mask = torch.zeros(
        (n_vendors, n_customers),
        dtype=torch.bool,
        device=device,
    )
    mask[tensor[:, 0], tensor[:, 1]] = True
    return mask


def _sample_rejected_ids(
    *,
    first_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    known_pair_mask: torch.Tensor,
    negatives: int,
    generator: torch.Generator | None,
    candidate_is_customer: bool,
) -> torch.Tensor:
    positions = torch.randint(
        0,
        candidate_ids.shape[0],
        (first_ids.shape[0], negatives),
        device=first_ids.device,
        generator=generator,
    )
    sampled = candidate_ids[positions]
    for _ in range(32):
        invalid = (
            known_pair_mask[first_ids.unsqueeze(1), sampled]
            if candidate_is_customer
            else known_pair_mask[sampled, first_ids.unsqueeze(1)]
        )
        if not invalid.any():
            return sampled
        positions[invalid] = torch.randint(
            0,
            candidate_ids.shape[0],
            (int(invalid.sum().item()),),
            device=first_ids.device,
            generator=generator,
        )
        sampled = candidate_ids[positions]

    invalid_rows, invalid_columns = torch.where(
        known_pair_mask[first_ids.unsqueeze(1), sampled]
        if candidate_is_customer
        else known_pair_mask[sampled, first_ids.unsqueeze(1)]
    )
    for row, column in zip(
        invalid_rows.tolist(),
        invalid_columns.tolist(),
        strict=True,
    ):
        first_id = int(first_ids[row])
        allowed = (
            candidate_ids[~known_pair_mask[first_id, candidate_ids]]
            if candidate_is_customer
            else candidate_ids[~known_pair_mask[candidate_ids, first_id]]
        )
        if allowed.numel() == 0:
            side = "customers" if candidate_is_customer else "vendors"
            raise ValueError(
                f"no train-negative {side} remain for index {first_id}"
            )
        sampled[row, column] = allowed[column % allowed.numel()]
    return sampled


def sample_train_only_negatives(
    *,
    vendor_ids: torch.Tensor,
    customer_ids: torch.Tensor,
    active_vendor_ids: torch.Tensor,
    active_customer_ids: torch.Tensor,
    known_pair_mask: torch.Tensor,
    negatives: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample symmetric negatives rejected only by the train-label mask."""
    if negatives <= 0:
        raise ValueError("negatives must be positive")
    negative_customers = _sample_rejected_ids(
        first_ids=vendor_ids,
        candidate_ids=active_customer_ids,
        known_pair_mask=known_pair_mask,
        negatives=negatives,
        generator=generator,
        candidate_is_customer=True,
    )
    negative_vendors = _sample_rejected_ids(
        first_ids=customer_ids,
        candidate_ids=active_vendor_ids,
        known_pair_mask=known_pair_mask,
        negatives=negatives,
        generator=generator,
        candidate_is_customer=False,
    )
    return negative_customers, negative_vendors


def _validate_source_cache(
    cache: dict[str, Any],
    *,
    vendor_feature: str,
    customer_feature: str,
) -> dict[str, Any]:
    fingerprint = cache.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("source cache has no fingerprint")
    vendor_features = cache.get("vendor_features")
    customer_features = cache.get("customer_features")
    if not isinstance(vendor_features, dict) or vendor_feature not in vendor_features:
        raise ValueError(f"source cache lacks vendor feature {vendor_feature!r}")
    if (
        not isinstance(customer_features, dict)
        or customer_feature not in customer_features
    ):
        raise ValueError(f"source cache lacks customer feature {customer_feature!r}")
    vendor_values = vendor_features[vendor_feature]
    customer_values = customer_features[customer_feature]
    if vendor_values.ndim != 2 or customer_values.ndim != 2:
        raise ValueError("source feature tensors must be matrices")

    split = cache.get("split")
    if not isinstance(split, dict):
        raise ValueError("source cache has no split")
    train_edges = _normalized_edges(split.get("train_edges", []))
    if len(train_edges) < 2:
        raise ValueError("source cache needs at least two training edges")
    test_slices = split.get("test_slices")
    if not isinstance(test_slices, dict):
        raise ValueError("source cache split has no test_slices")
    return {
        "fingerprint": fingerprint,
        "vendor_values": vendor_values.detach().cpu().to(torch.float32),
        "customer_values": customer_values.detach().cpu().to(torch.float32),
        "split": split,
        "train_edges": train_edges,
    }


def _validate_evaluation_cache(
    *,
    source_cache: dict[str, Any],
    evaluation_cache: dict[str, Any],
    source_fingerprint: str,
    customer_feature: str,
    same_artifact: bool,
) -> dict[str, Any]:
    fingerprint = evaluation_cache.get("fingerprint")
    linked_source = evaluation_cache.get("source_fingerprint")
    metadata = evaluation_cache.get("independent_profile")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("evaluation cache has no fingerprint")
    if not isinstance(linked_source, str) or not linked_source:
        raise ValueError("evaluation cache has no source_fingerprint")
    if not same_artifact and linked_source != source_fingerprint:
        raise ValueError(
            "evaluation cache source fingerprint does not match the training "
            f"source cache: {linked_source!r} != {source_fingerprint!r}"
        )
    if not isinstance(metadata, dict):
        raise ValueError("evaluation cache is not an independent-profile artifact")
    fingerprint_payload = metadata.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise ValueError("evaluation cache lacks its fingerprint payload")
    if fingerprint_payload.get("source_cache_fingerprint") != linked_source:
        raise ValueError("evaluation source fingerprint disagrees with its payload")
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("evaluation cache fingerprint does not match its payload")

    contract = metadata.get("construction_contract")
    if not isinstance(contract, dict):
        raise ValueError("evaluation cache lacks its construction contract")
    required_contract = {
        "test_rows_use_heldout_graph_pains": False,
        "test_rows_use_heldout_graph_audiences": False,
        "training_rows_reused": True,
        "identity_safe_customer_names": True,
    }
    for key, expected in required_contract.items():
        if contract.get(key) is not expected:
            raise ValueError(
                f"evaluation construction contract does not attest {key}={expected}"
            )
    provenance = metadata.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not _REQUIRED_PROFILE_PROVENANCE <= set(provenance)
    ):
        raise ValueError("evaluation cache has incomplete profile provenance")

    source_split = source_cache.get("split") or {}
    evaluation_split = evaluation_cache.get("split") or {}
    if evaluation_split.get("profile_source") != "independent_public_research":
        raise ValueError("evaluation cache is not marked independently profiled")
    if _normalized_edges(source_split.get("train_edges", [])) != _normalized_edges(
        evaluation_split.get("train_edges", [])
    ):
        raise ValueError("source and evaluation caches use different training edges")

    profiled = list(map(int, metadata.get("profiled_customer_indices", [])))
    candidates = list(
        map(
            int,
            evaluation_split.get(
                "customer_holdout_candidates",
                evaluation_split.get("holdout_customers", []),
            ),
        )
    )
    if not profiled or profiled != candidates or len(profiled) != len(set(profiled)):
        raise ValueError(
            "evaluation profiled customers must equal unique holdout candidates"
        )

    evaluation_features = evaluation_cache.get("customer_features")
    if (
        not isinstance(evaluation_features, dict)
        or customer_feature not in evaluation_features
    ):
        raise ValueError(
            f"evaluation cache lacks customer feature {customer_feature!r}"
        )
    source_values = source_cache["customer_features"][customer_feature]
    evaluation_values = evaluation_features[customer_feature]
    if source_values.shape != evaluation_values.shape:
        raise ValueError("source and evaluation customer feature shapes differ")
    train_customers = list(map(int, source_split.get("train_customers", [])))
    if train_customers:
        train_index = torch.tensor(train_customers, dtype=torch.long)
        if not torch.equal(
            source_values.detach().cpu()[train_index],
            evaluation_values.detach().cpu()[train_index],
        ):
            raise ValueError(
                "evaluation cache modified one or more training-customer rows"
            )

    diagnostics = (
        evaluation_cache.get("diagnostics", {}).get("independent_profiles", {})
    )
    original_positive_customers = int(
        diagnostics.get("original_customer_holdout_positive_customers", 0)
    )
    recorded_coverage = float(diagnostics.get("coverage_fraction", -1.0))
    expected_coverage = len(profiled) / max(original_positive_customers, 1)
    if original_positive_customers < len(profiled) or not math.isclose(
        recorded_coverage,
        expected_coverage,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError("evaluation cache profile coverage is inconsistent")
    return {
        "fingerprint": fingerprint,
        "source_fingerprint": linked_source,
        "profiled_customers": profiled,
        "profile_coverage": expected_coverage,
        "original_positive_customers": original_positive_customers,
        "customer_values": evaluation_values.detach().cpu().to(torch.float32),
        "split": evaluation_split,
        "provenance": provenance,
    }


def build_evaluation_definition(
    *,
    source_cache: dict[str, Any],
    evaluation_cache: dict[str, Any],
    evaluation_slice: str,
    profiled_customers: Sequence[int],
    profile_coverage: float,
    source_positive_customers: int,
) -> EvaluationDefinition:
    """Construct the requested independently profiled cold-start evaluation."""
    if evaluation_slice not in EVALUATION_SLICES:
        raise ValueError(f"unknown evaluation slice: {evaluation_slice}")
    candidates = sorted(set(map(int, profiled_customers)))
    candidate_set = set(candidates)
    if evaluation_slice == "customer_holdout":
        raw_edges = (
            evaluation_cache.get("split", {})
            .get("test_slices", {})
            .get("customer_holdout", [])
        )
    else:
        raw_edges = (
            source_cache.get("split", {})
            .get("test_slices", {})
            .get("double_holdout", [])
        )
    edges = [
        (int(vendor), int(customer))
        for vendor, customer in raw_edges
        if int(customer) in candidate_set
    ]
    if not edges:
        raise ValueError(
            f"{evaluation_slice} has no positive edges for profiled customers"
        )
    if evaluation_slice == "customer_holdout":
        train_vendors = set(
            map(int, source_cache.get("split", {}).get("train_vendors", []))
        )
        if train_vendors and any(vendor not in train_vendors for vendor, _ in edges):
            raise ValueError("customer_holdout includes a heldout vendor")
    else:
        heldout_vendors = set(
            map(int, source_cache.get("split", {}).get("holdout_vendors", []))
        )
        if heldout_vendors and any(
            vendor not in heldout_vendors for vendor, _ in edges
        ):
            raise ValueError("double_holdout includes a training vendor")
    return EvaluationDefinition(
        slice_definition=SliceDefinition(
            edges=sorted(set(edges)),
            candidate_customers=candidates,
            excluded_pairs=set(),
        ),
        profiled_customers=candidates,
        profile_coverage=profile_coverage,
        source_positive_customers=source_positive_customers,
    )


def _encode_in_batches(
    tower: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    tower.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(
                device=device,
                dtype=torch.float32,
            )
            outputs.append(tower(batch).cpu())
    return torch.cat(outputs, dim=0).numpy()


def _symmetric_infonce_loss(
    *,
    model: InductiveModelFamily,
    vendor_values: torch.Tensor,
    customer_values: torch.Tensor,
    edges: torch.Tensor,
    negative_customers: torch.Tensor,
    negative_vendors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    vendor_ids = edges[:, 0]
    customer_ids = edges[:, 1]
    vendor_embedding = model.vendor(vendor_values[vendor_ids])
    customer_embedding = model.customer(customer_values[customer_ids])
    positive = (vendor_embedding * customer_embedding).sum(dim=1, keepdim=True)

    n_negatives = negative_customers.shape[1]
    negative_customer_embedding = model.customer(
        customer_values[negative_customers.reshape(-1)]
    ).reshape(edges.shape[0], n_negatives, -1)
    customer_logits = torch.cat(
        [
            positive,
            torch.bmm(
                negative_customer_embedding,
                vendor_embedding.unsqueeze(-1),
            ).squeeze(-1),
        ],
        dim=1,
    ) / temperature

    negative_vendor_embedding = model.vendor(
        vendor_values[negative_vendors.reshape(-1)]
    ).reshape(edges.shape[0], n_negatives, -1)
    vendor_logits = torch.cat(
        [
            positive,
            torch.bmm(
                negative_vendor_embedding,
                customer_embedding.unsqueeze(-1),
            ).squeeze(-1),
        ],
        dim=1,
    ) / temperature
    labels = torch.zeros(edges.shape[0], dtype=torch.long, device=edges.device)
    return (
        F.cross_entropy(customer_logits, labels)
        + F.cross_entropy(vendor_logits, labels)
    ) / 2.0


def _metric_projection(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "map": raw_metrics["mean_average_precision"],
        "ndcg_at_50": raw_metrics["mean_ndcg_at_50"],
        "precision_at_10": raw_metrics["mean_precision_at_10"],
        "chance_precision_at_10": raw_metrics[
            "mean_chance_precision_at_10"
        ],
        "precision_at_10_lift": raw_metrics["precision_at_10_lift"],
        "recall_at_50": raw_metrics["mean_recall_at_50"],
    }


def train_configuration(
    *,
    source_cache_path: Path,
    evaluation_cache_path: Path | None,
    output_dir: Path,
    vendor_feature: str,
    customer_feature: str,
    vendor_blocks: Sequence[BlockSpec],
    customer_blocks: Sequence[BlockSpec],
    evaluation_slice: str,
    family: str,
    device_name: str,
    seed: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    negatives: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    validation_fraction: float,
    temperature: float,
) -> dict[str, Any]:
    """Train one model-family configuration and evaluate one cold-start slice."""
    started_total = time.perf_counter()
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family: {family}")
    if evaluation_slice not in EVALUATION_SLICES:
        raise ValueError(f"unknown evaluation slice: {evaluation_slice}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if hidden_dim <= 0 or output_dim <= 0:
        raise ValueError("hidden_dim and output_dim must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device_name}")

    load_started = time.perf_counter()
    source_cache = torch.load(
        source_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    source = _validate_source_cache(
        source_cache,
        vendor_feature=vendor_feature,
        customer_feature=customer_feature,
    )
    if evaluation_cache_path is None:
        evaluation_cache = source_cache
        same_artifact = True
        resolved_evaluation_path = source_cache_path
    else:
        evaluation_cache = torch.load(
            evaluation_cache_path,
            map_location="cpu",
            weights_only=False,
        )
        same_artifact = source_cache_path.resolve() == evaluation_cache_path.resolve()
        resolved_evaluation_path = evaluation_cache_path
    evaluation = _validate_evaluation_cache(
        source_cache=source_cache,
        evaluation_cache=evaluation_cache,
        source_fingerprint=source["fingerprint"],
        customer_feature=customer_feature,
        same_artifact=same_artifact,
    )
    definition = build_evaluation_definition(
        source_cache=source_cache,
        evaluation_cache=evaluation_cache,
        evaluation_slice=evaluation_slice,
        profiled_customers=evaluation["profiled_customers"],
        profile_coverage=evaluation["profile_coverage"],
        source_positive_customers=evaluation["original_positive_customers"],
    )

    vendor_values_cpu = source["vendor_values"]
    customer_values_cpu = source["customer_values"]
    evaluation_customer_values = customer_values_cpu.clone()
    profiled_index = torch.tensor(
        definition.profiled_customers,
        dtype=torch.long,
    )
    evaluation_customer_values[profiled_index] = evaluation["customer_values"][
        profiled_index
    ]
    vendor_blocks = validate_block_dimensions(
        vendor_blocks,
        int(vendor_values_cpu.shape[1]),
        tower_name="vendor",
    )
    customer_blocks = validate_block_dimensions(
        customer_blocks,
        int(customer_values_cpu.shape[1]),
        tower_name="customer",
    )
    load_seconds = time.perf_counter() - load_started

    train_edges_list = source["train_edges"]
    edge_array = np.asarray(train_edges_list, dtype=np.int64)
    split_rng = np.random.RandomState(seed)
    split_rng.shuffle(edge_array)
    n_validation = max(1, round(edge_array.shape[0] * validation_fraction))
    if n_validation >= edge_array.shape[0]:
        raise ValueError("validation split leaves no optimization edges")
    validation_edges = torch.from_numpy(edge_array[:n_validation]).to(device)
    optimization_edges = edge_array[n_validation:]

    active_vendors = sorted({vendor for vendor, _ in train_edges_list})
    active_customers = sorted({customer for _, customer in train_edges_list})
    active_vendor_ids = torch.tensor(
        active_vendors,
        dtype=torch.long,
        device=device,
    )
    active_customer_ids = torch.tensor(
        active_customers,
        dtype=torch.long,
        device=device,
    )
    known_pair_mask = build_train_only_pair_mask(
        train_edges=train_edges_list,
        n_vendors=vendor_values_cpu.shape[0],
        n_customers=customer_values_cpu.shape[0],
        device=device,
    )
    vendor_values = vendor_values_cpu.to(device)
    customer_values = customer_values_cpu.to(device)

    model = InductiveModelFamily(
        family=family,
        vendor_dim=vendor_values.shape[1],
        customer_dim=customer_values.shape[1],
        vendor_blocks=vendor_blocks,
        customer_blocks=customer_blocks,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(seed + 1_000_003)
    validation_negative_customers, validation_negative_vendors = (
        sample_train_only_negatives(
            vendor_ids=validation_edges[:, 0],
            customer_ids=validation_edges[:, 1],
            active_vendor_ids=active_vendor_ids,
            active_customer_ids=active_customer_ids,
            known_pair_mask=known_pair_mask,
            negatives=negatives,
            generator=validation_generator,
        )
    )

    run_name = (
        f"{family}__{vendor_feature}__{customer_feature}"
        f"__{evaluation_slice}__seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}.pt"
    result_path = output_dir / f"{run_name}.json"

    train_started = time.perf_counter()
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    epoch_rng = np.random.RandomState(seed + 17)
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_edges = optimization_edges.copy()
        epoch_rng.shuffle(epoch_edges)
        losses: list[float] = []
        for start in range(0, epoch_edges.shape[0], batch_size):
            edges = torch.from_numpy(
                epoch_edges[start : start + batch_size]
            ).to(device)
            negative_customers, negative_vendors = sample_train_only_negatives(
                vendor_ids=edges[:, 0],
                customer_ids=edges[:, 1],
                active_vendor_ids=active_vendor_ids,
                active_customer_ids=active_customer_ids,
                known_pair_mask=known_pair_mask,
                negatives=negatives,
            )
            loss = _symmetric_infonce_loss(
                model=model,
                vendor_values=vendor_values,
                customer_values=customer_values,
                edges=edges,
                negative_customers=negative_customers,
                negative_vendors=negative_vendors,
                temperature=temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _symmetric_infonce_loss(
                    model=model,
                    vendor_values=vendor_values,
                    customer_values=customer_values,
                    edges=validation_edges,
                    negative_customers=validation_negative_customers,
                    negative_vendors=validation_negative_vendors,
                    temperature=temperature,
                ).item()
            )
        mean_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(mean_loss, 6),
                "validation_loss": round(validation_loss, 6),
                "elapsed_seconds": round(
                    time.perf_counter() - epoch_started,
                    3,
                ),
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{run_name} epoch={epoch} train={mean_loss:.4f} "
                f"validation={validation_loss:.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)

    encode_started = time.perf_counter()
    vendor_embeddings = _encode_in_batches(
        model.vendor,
        vendor_values_cpu,
        device=device,
        batch_size=batch_size,
    )
    customer_embeddings = _encode_in_batches(
        model.customer,
        evaluation_customer_values,
        device=device,
        batch_size=batch_size,
    )
    encoding_seconds = time.perf_counter() - encode_started

    evaluation_started = time.perf_counter()
    known_pairs = {
        (int(vendor), int(customer))
        for vendor, customer in source_cache.get(
            "pairs",
            train_edges_list + definition.slice_definition.edges,
        )
    }
    raw_metrics = evaluate_slice(
        vendor_embeddings=vendor_embeddings,
        customer_embeddings=customer_embeddings,
        definition=definition.slice_definition,
        known_pairs=known_pairs,
        seed=seed,
    )
    metrics = _metric_projection(raw_metrics)
    customer_train_degree = np.bincount(
        np.asarray(train_edges_list, dtype=np.int64)[:, 1],
        minlength=customer_values_cpu.shape[0],
    ).astype(np.float32)
    popularity_raw = evaluate_slice(
        vendor_embeddings=np.ones(
            (vendor_values_cpu.shape[0], 1),
            dtype=np.float32,
        ),
        customer_embeddings=customer_train_degree[:, None],
        definition=definition.slice_definition,
        known_pairs=known_pairs,
        seed=seed,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started

    configuration = {
        "family": family,
        "vendor_feature": vendor_feature,
        "customer_feature": customer_feature,
        "vendor_blocks": [asdict(spec) for spec in vendor_blocks],
        "customer_blocks": [asdict(spec) for spec in customer_blocks],
        "evaluation_slice": evaluation_slice,
        "device": str(device),
        "seed": seed,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "dropout": dropout,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "negatives": negatives,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "validation_fraction": validation_fraction,
        "temperature": temperature,
    }
    checkpoint = {
        "created_at": _utc_now(),
        "run_name": run_name,
        "model_state_dict": best_state,
        "configuration": configuration,
        "source_cache_fingerprint": source["fingerprint"],
        "evaluation_cache_fingerprint": evaluation["fingerprint"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "model_interpretation": MODEL_INTERPRETATION,
    }
    torch.save(checkpoint, checkpoint_path)

    population = {
        "vendors": int(vendor_values_cpu.shape[0]),
        "customers": int(customer_values_cpu.shape[0]),
        "source_training_edges": len(train_edges_list),
        "optimization_edges": int(optimization_edges.shape[0]),
        "validation_edges": int(validation_edges.shape[0]),
        "active_training_vendors": len(active_vendors),
        "active_training_customers": len(active_customers),
        "profiled_candidate_customers": len(definition.profiled_customers),
        "source_profile_eligible_customers": (
            definition.source_positive_customers
        ),
        "profile_coverage": round(definition.profile_coverage, 6),
        "evaluation_positive_edges": raw_metrics["n_positive_edges"],
        "evaluation_vendor_queries": raw_metrics["n_queries"],
        "mean_candidate_customers": raw_metrics["mean_candidate_customers"],
    }
    timings = {
        "load_and_validate_seconds": round(load_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "total_seconds": round(time.perf_counter() - started_total, 3),
    }
    result = {
        "created_at": _utc_now(),
        "run_name": run_name,
        "model_interpretation": MODEL_INTERPRETATION,
        "source_cache": {
            "path": str(source_cache_path),
            "fingerprint": source["fingerprint"],
        },
        "evaluation_cache": {
            "path": str(resolved_evaluation_path),
            "fingerprint": evaluation["fingerprint"],
            "source_fingerprint": evaluation["source_fingerprint"],
            "provenance": evaluation["provenance"],
        },
        "configuration": configuration,
        "checkpoint": {
            "path": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_validation, 6),
        },
        "population": population,
        "metrics": metrics,
        "baselines": {
            "customer_train_degree_popularity": _metric_projection(
                popularity_raw
            )
        },
        "timings": timings,
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "metrics": metrics,
                "population": population,
                "timings": timings,
            },
            indent=2,
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--source-cache", type=Path, required=True)
    train.add_argument("--evaluation-cache", type=Path)
    train.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    train.add_argument("--vendor-feature", required=True)
    train.add_argument("--customer-feature", required=True)
    train.add_argument(
        "--vendor-block-dims",
        nargs="+",
        required=True,
        metavar="RELATION:DIM",
    )
    train.add_argument(
        "--customer-block-dims",
        nargs="+",
        required=True,
        metavar="RELATION:DIM",
    )
    train.add_argument(
        "--evaluation-slice",
        choices=EVALUATION_SLICES,
        required=True,
    )
    train.add_argument("--family", choices=MODEL_FAMILIES, required=True)
    train.add_argument("--device", default="cuda:4")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--output-dim", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--negatives", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--temperature", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train_configuration(
            source_cache_path=args.source_cache,
            evaluation_cache_path=args.evaluation_cache,
            output_dir=args.output_dir,
            vendor_feature=args.vendor_feature,
            customer_feature=args.customer_feature,
            vendor_blocks=parse_block_specs(args.vendor_block_dims),
            customer_blocks=parse_block_specs(args.customer_block_dims),
            evaluation_slice=args.evaluation_slice,
            family=args.family,
            device_name=args.device,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            negatives=args.negatives,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            validation_fraction=args.validation_fraction,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
