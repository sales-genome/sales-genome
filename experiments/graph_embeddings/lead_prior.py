"""Frozen GNN prior for lead-scoring experiments.

This module exposes the trained graph model without coupling it to the
production lead pipeline. PyTorch and PyG are imported lazily so evaluation
helpers and tests can run in environments that do not have the GNN stack.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.graph_embeddings.prospect_features import (
    PainCanonicalMatcher,
    ProspectFeatureSpace,
    ProspectProfile,
    UniqueExactIndex,
    exact_key,
)


ROOT = Path(__file__).resolve().parents[2]
GRAPH_EMBEDDINGS_DIR = ROOT / "experiments" / "graph_embeddings"
DEFAULT_VARIANT = "collapsed_res_d100"


class GraphPriorUnavailable(RuntimeError):
    """Raised when the frozen graph-prior artifacts cannot be loaded."""


class LeadGraphPrior:
    """Load frozen embeddings and score vendor/product-to-prospect fit."""

    def __init__(
        self,
        *,
        variant: str = DEFAULT_VARIANT,
        graph_embeddings_dir: Path = GRAPH_EMBEDDINGS_DIR,
    ) -> None:
        self.variant = variant
        self.graph_embeddings_dir = graph_embeddings_dir
        self.features_dir = graph_embeddings_dir / "features"
        self.results_dir = graph_embeddings_dir / "results"
        self.variants_dir = graph_embeddings_dir / "graph_variants"
        self.base_data_path = graph_embeddings_dir / "hetero_data.pt"
        self.variant_path = self.variants_dir / f"{variant}.pt"
        self.checkpoint_path = self.results_dir / f"{variant}_best.pt"
        self.results_path = self.results_dir / f"{variant}_results.json"

        self._loaded = False
        self.vendor_embeddings: np.ndarray | None = None
        self.product_embeddings: np.ndarray | None = None
        self.customer_embeddings: np.ndarray | None = None
        self.feature_space: ProspectFeatureSpace | None = None
        self.pain_matcher: PainCanonicalMatcher | None = None

        self.node_maps: dict[str, dict[str, int]] = {}
        self.vendor_product_lists: list[list[int]] = []
        self.vendor_support: Counter[int] = Counter()
        self.product_support: Counter[int] = Counter()
        self.customer_names: dict[int, str] = {}
        self.customer_index = UniqueExactIndex()
        self.audience_index = UniqueExactIndex()
        self.product_aliases: dict[tuple[str, str], set[int]] = defaultdict(set)
        self.product_names: dict[int, str] = {}
        self.product_vendor: dict[int, str] = {}
        self.competitors: dict[str, set[str]] = defaultdict(set)
        self.train_vc: tuple[tuple[int, int], ...] = ()
        self.test_vc: tuple[tuple[int, int], ...] = ()
        self.fingerprint = ""

    def load(self) -> None:
        if self._loaded:
            return
        missing = [
            path
            for path in (
                self.base_data_path,
                self.variant_path,
                self.checkpoint_path,
                self.results_path,
            )
            if not path.exists()
        ]
        if missing:
            raise GraphPriorUnavailable(
                "Missing graph-prior artifacts: " + ", ".join(map(str, missing))
            )

        try:
            import torch
            import torch.nn.functional as functional
            from experiments.graph_embeddings.train_v2 import LinkPredModel
        except ImportError as exc:
            raise GraphPriorUnavailable(
                "The graph prior requires torch and torch-geometric. "
                "Run it in the graph-embedding environment."
            ) from exc

        base_checkpoint = torch.load(
            self.base_data_path, map_location="cpu", weights_only=False
        )
        variant_checkpoint = torch.load(
            self.variant_path, map_location="cpu", weights_only=False
        )
        data = variant_checkpoint["data"]
        n_vendors = int(variant_checkpoint["n_vendors"])
        vendor_to_products = variant_checkpoint.get("vendor_to_products", {})
        self.vendor_product_lists = [
            list(vendor_to_products.get(index, [])) for index in range(n_vendors)
        ]

        training_config = json.loads(self.results_path.read_text())
        model = LinkPredModel(
            self.variant,
            data,
            hidden_dim=int(training_config["hidden_dim"]),
            num_layers=int(training_config["num_layers"]),
        )
        model.load_state_dict(
            torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            embedding_dict = model.encode(data)
            vendor, customer = model.get_vendor_customer(
                embedding_dict,
                self.vendor_product_lists,
                n_vendors,
            )
            product = embedding_dict.get("vendor_product")
            if product is None:
                raise GraphPriorUnavailable(
                    f"{self.variant} does not expose vendor_product embeddings"
                )
            self.vendor_embeddings = functional.normalize(
                vendor, dim=-1
            ).cpu().numpy()
            self.product_embeddings = functional.normalize(
                product, dim=-1
            ).cpu().numpy()
            self.customer_embeddings = functional.normalize(
                customer, dim=-1
            ).cpu().numpy()

        self.node_maps = {
            node_type: dict(mapping)
            for node_type, mapping in base_checkpoint["node_maps"].items()
        }
        self.train_vc = tuple(
            (int(vendor), int(customer))
            for vendor, customer in base_checkpoint["train_vc"]
        )
        self.test_vc = tuple(
            (int(vendor), int(customer))
            for vendor, customer in base_checkpoint["test_vc"]
        )
        self.vendor_support.update(vendor for vendor, _ in self.train_vc)
        residual_edge_type = ("vendor_product", "case_study", "customer")
        if residual_edge_type in data.edge_types:
            residual_edges = _as_numpy(data[residual_edge_type].edge_index)
            self.product_support.update(int(index) for index in residual_edges[0])
        self._load_node_metadata()
        self._load_competitors()
        self._build_feature_space(base_checkpoint["data"])
        self.fingerprint = self._artifact_fingerprint()
        self._loaded = True

    def score_lead(
        self,
        *,
        account_name: str,
        vendor_slug: str,
        product_name: str | None = None,
        pain_ids: Iterable[str] = (),
        audience_ids: Iterable[str] = (),
        explicit_competitors: Iterable[str] = (),
        top_k_lookalikes: int = 20,
        product_weight: float = 0.5,
        competitor_prior_strength: float = 25.0,
    ) -> dict[str, Any]:
        self.load()
        assert self.vendor_embeddings is not None
        assert self.product_embeddings is not None
        assert self.customer_embeddings is not None
        assert self.feature_space is not None
        if not 0.0 <= product_weight <= 1.0:
            raise ValueError("product_weight must be between 0 and 1")
        if competitor_prior_strength <= 0:
            raise ValueError("competitor_prior_strength must be positive")

        vendor_id = f"vendor:{vendor_slug}"
        vendor_index = self.node_maps.get("vendor", {}).get(vendor_id)
        if vendor_index is None:
            return {
                "status": "vendor_not_in_graph",
                "account_name": account_name,
                "vendor_slug": vendor_slug,
                "product": product_name,
                "scores": {},
                "model_fingerprint": self.fingerprint,
            }

        exact_customer_index = self.customer_index.resolve(account_name)
        if exact_customer_index is not None:
            prospect_embedding = self.customer_embeddings[exact_customer_index]
            prospect = {
                "status": "exact_graph_customer",
                "customer_index": exact_customer_index,
                "customer_name": self.customer_names.get(
                    exact_customer_index, account_name
                ),
                "profile_coverage": 1.0,
                "lookalikes": [],
            }
        else:
            profile = ProspectProfile(
                account_name=account_name,
                pain_ids=tuple(sorted(set(pain_ids))),
                audience_ids=tuple(sorted(set(audience_ids))),
                source="external_enrichment",
            )
            projection = self.feature_space.project(
                profile,
                top_k=top_k_lookalikes,
            )
            if projection.embedding is None:
                return {
                    "status": projection.status,
                    "account_name": account_name,
                    "vendor_slug": vendor_slug,
                    "product": product_name,
                    "scores": {},
                    "prospect": {
                        "status": projection.status,
                        "profile_coverage": projection.profile_coverage,
                        "pain_coverage": projection.pain_coverage,
                        "audience_coverage": projection.audience_coverage,
                    },
                    "model_fingerprint": self.fingerprint,
                }
            prospect_embedding = projection.embedding
            prospect = {
                "status": "out_of_sample_lookalike",
                "profile_coverage": projection.profile_coverage,
                "pain_coverage": projection.pain_coverage,
                "audience_coverage": projection.audience_coverage,
                "lookalikes": [
                    {
                        **item,
                        "customer_name": self.customer_names.get(
                            int(item["customer_index"]), ""
                        ),
                    }
                    for item in projection.lookalikes
                ],
            }

        pooled_vendor = self.vendor_embeddings[vendor_index]
        scores: dict[str, float] = {
            "gnn_vendor_pooled": _cosine(pooled_vendor, prospect_embedding),
        }
        embedding_sources: dict[str, Any] = {
            "gnn_vendor_pooled": {
                "vendor_id": vendor_id,
                "support_customers": self.vendor_support[vendor_index],
            }
        }

        product_index = self.resolve_product(vendor_slug, product_name)
        if product_index is not None:
            product_embedding = self.product_embeddings[product_index]
            scores["gnn_product_direct"] = _cosine(
                product_embedding, prospect_embedding
            )
            shrunken = _normalize(
                product_weight * product_embedding
                + (1.0 - product_weight) * pooled_vendor
            )
            scores["gnn_product_vendor_shrinkage"] = _cosine(
                shrunken, prospect_embedding
            )
            embedding_sources["gnn_product_direct"] = {
                "product_index": product_index,
                "product_name": self.product_names.get(product_index, product_name),
                "support_customers": self.product_support[product_index],
            }
            embedding_sources["gnn_product_vendor_shrinkage"] = {
                "product_weight": product_weight,
                "product_index": product_index,
            }

        competitor_slugs = set(self.competitors.get(vendor_slug, set()))
        competitor_slugs.update(slug for slug in explicit_competitors if slug)
        competitor_parts = []
        for competitor_slug in sorted(competitor_slugs):
            competitor_index = self.node_maps.get("vendor", {}).get(
                f"vendor:{competitor_slug}"
            )
            if competitor_index is not None and self.vendor_support[competitor_index] > 0:
                competitor_parts.append(
                    (
                        competitor_slug,
                        competitor_index,
                        self.vendor_embeddings[competitor_index],
                        self.vendor_support[competitor_index],
                    )
                )

        if competitor_parts:
            weights = np.asarray(
                [math.sqrt(part[3]) for part in competitor_parts],
                dtype=np.float32,
            )
            weights /= weights.sum()
            competitor_centroid = _normalize(
                sum(
                    weight * part[2]
                    for weight, part in zip(weights, competitor_parts)
                )
            )
            scores["gnn_competitor_centroid"] = _cosine(
                competitor_centroid, prospect_embedding
            )

            support = float(self.vendor_support[vendor_index])
            own_weight = support / (support + competitor_prior_strength)
            sparse_vendor = _normalize(
                own_weight * pooled_vendor
                + (1.0 - own_weight) * competitor_centroid
            )
            scores["gnn_sparse_vendor_shrinkage"] = _cosine(
                sparse_vendor, prospect_embedding
            )
            contributors = [
                {
                    "vendor_slug": part[0],
                    "support_customers": part[3],
                    "weight": round(float(weight), 6),
                }
                for part, weight in zip(competitor_parts, weights)
            ]
            embedding_sources["gnn_competitor_centroid"] = {
                "contributors": contributors,
            }
            embedding_sources["gnn_sparse_vendor_shrinkage"] = {
                "own_weight": round(own_weight, 6),
                "prior_strength": competitor_prior_strength,
                "contributors": contributors,
            }

        return {
            "status": "ok",
            "account_name": account_name,
            "vendor_slug": vendor_slug,
            "product": product_name,
            "scores": {name: round(value, 8) for name, value in scores.items()},
            "prospect": prospect,
            "support": {
                "vendor_training_customers": self.vendor_support[vendor_index],
                "product_training_edges": (
                    self.product_support[product_index]
                    if product_index is not None
                    else None
                ),
                "n_products": len(self.vendor_product_lists[vendor_index]),
            },
            "embedding_sources": embedding_sources,
            "model_fingerprint": self.fingerprint,
        }

    def resolve_product(
        self, vendor_slug: str, product_name: str | None
    ) -> int | None:
        if not product_name:
            return None
        matches = self.product_aliases.get(
            (vendor_slug, exact_key(product_name)), set()
        )
        if len(matches) != 1:
            return None
        return next(iter(matches))

    def resolve_audience_ids(self, values: Iterable[str]) -> tuple[str, ...]:
        self.load()
        index_to_id = {
            index: audience_id
            for audience_id, index in self.node_maps.get("audience", {}).items()
        }
        resolved = {
            index_to_id[index]
            for value in values
            if (index := self.audience_index.resolve(value)) is not None
        }
        return tuple(sorted(resolved))

    def match_pain_texts(
        self,
        pain_texts: Iterable[str],
        *,
        top_k: int = 5,
        threshold: float = 0.45,
    ) -> tuple[str, ...]:
        self.load()
        assert self.pain_matcher is not None
        return self.pain_matcher.match(
            pain_texts, top_k=top_k, threshold=threshold
        )

    def _build_feature_space(self, base_data: Any) -> None:
        assert self.customer_embeddings is not None
        customer_pain = _as_numpy(base_data["customer"].x)
        n_customers = customer_pain.shape[0]
        n_audiences = len(self.node_maps["audience"])
        customer_audience = np.zeros(
            (n_customers, n_audiences), dtype=np.float32
        )
        audience_edges = _as_numpy(
            base_data[("customer", "in_audience", "audience")].edge_index
        )
        customer_audience[audience_edges[0], audience_edges[1]] = 1.0

        pain_subtheme_edges = _as_numpy(
            base_data[
                ("pain_canonical", "in_subtheme", "pain_subtheme")
            ].edge_index
        )
        pain_index_to_subtheme = {
            int(pain_index): int(subtheme_index)
            for pain_index, subtheme_index in zip(
                pain_subtheme_edges[0], pain_subtheme_edges[1]
            )
        }
        pain_id_to_subtheme = {
            pain_id: pain_index_to_subtheme[pain_index]
            for pain_id, pain_index in self.node_maps["pain_canonical"].items()
            if pain_index in pain_index_to_subtheme
        }
        self.feature_space = ProspectFeatureSpace(
            customer_pain_features=customer_pain,
            customer_audience_features=customer_audience,
            customer_embeddings=self.customer_embeddings,
            pain_id_to_subtheme=pain_id_to_subtheme,
            audience_id_to_index=self.node_maps["audience"],
        )

        pain_ids = [
            pain_id
            for pain_id, _ in sorted(
                self.node_maps["pain_canonical"].items(),
                key=lambda item: item[1],
            )
        ]
        metadata = json.loads((self.features_dir / "metadata.json").read_text())
        self.pain_matcher = PainCanonicalMatcher(
            pain_features_path=self.features_dir / "pain_features.npy",
            pain_ids=pain_ids,
            model_name=metadata["model"],
        )

    def _load_node_metadata(self) -> None:
        nodes_path = ROOT / "graph" / "v6" / "nodes.jsonl"
        with open(nodes_path) as handle:
            for line in handle:
                node = json.loads(line)
                node_id = node["id"]
                node_type = node.get("type")
                if node_type == "customer":
                    index = self.node_maps["customer"].get(node_id)
                    if index is None:
                        continue
                    name = node.get("name") or node.get("canonical_name") or node_id
                    self.customer_names[index] = name
                    for alias in [name, node_id.removeprefix("customer:")]:
                        self.customer_index.add(alias, index)
                    for alias in node.get("raw_names", []):
                        self.customer_index.add(alias, index)
                elif node_type == "audience":
                    index = self.node_maps["audience"].get(node_id)
                    if index is None:
                        continue
                    aliases = [
                        node_id,
                        node_id.removeprefix("audience:"),
                        node.get("name", ""),
                        node.get("canonical_name", ""),
                        node.get("value", ""),
                    ]
                    for alias in aliases:
                        self.audience_index.add(alias, index)
                elif node_type == "product":
                    product_index = self.node_maps["product"].get(node_id)
                    if product_index is None:
                        continue
                    vendor_slug = node.get("vendor_slug", "")
                    name = node.get("name", "")
                    self.product_names[product_index] = name
                    self.product_vendor[product_index] = vendor_slug
                    for alias in [name, *node.get("raw_variants", [])]:
                        key = exact_key(alias)
                        if key:
                            self.product_aliases[(vendor_slug, key)].add(
                                product_index
                            )

    def _load_competitors(self) -> None:
        path = ROOT / "graph" / "poc" / "competitor_edges.jsonl"
        if not path.exists():
            return
        with open(path) as handle:
            for line in handle:
                edge = json.loads(line)
                if edge.get("src_role") != "vendor":
                    continue
                source = edge.get("src_slug", "")
                target = edge.get("dst_slug", "")
                if source and target:
                    self.competitors[source].add(target)
                    self.competitors[target].add(source)

    def _artifact_fingerprint(self) -> str:
        parts = [self.variant]
        for path in (
            self.base_data_path,
            self.variant_path,
            self.checkpoint_path,
            self.results_path,
            self.features_dir / "metadata.json",
        ):
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(_normalize(left) @ _normalize(right))
