"""Feature construction for graph-prior scoring of unseen prospects.

The trained GNN has embeddings only for graph customer nodes. This module
projects an unseen prospect into that space by finding graph customers with
similar pain-subtheme and audience profiles, then averaging their frozen GNN
embeddings. It intentionally contains no vendor-specific research logic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def exact_key(value: str) -> str:
    """Normalize only case and whitespace for conservative exact matching."""
    return " ".join((value or "").casefold().split())


class UniqueExactIndex:
    """Resolve aliases only when an exact normalized key maps uniquely."""

    def __init__(self) -> None:
        self._values: dict[str, set[int]] = {}

    def add(self, alias: str, index: int) -> None:
        key = exact_key(alias)
        if key:
            self._values.setdefault(key, set()).add(index)

    def resolve(self, alias: str) -> int | None:
        matches = self._values.get(exact_key(alias), set())
        if len(matches) != 1:
            return None
        return next(iter(matches))


@dataclass(frozen=True)
class ProspectProfile:
    account_name: str
    pain_ids: tuple[str, ...] = ()
    audience_ids: tuple[str, ...] = ()
    source: str = "provided"


@dataclass(frozen=True)
class LookalikeProjection:
    embedding: np.ndarray | None
    status: str
    profile_coverage: float
    pain_coverage: float
    audience_coverage: float
    lookalikes: tuple[dict[str, float | int], ...]


class ProspectFeatureSpace:
    """Build prospect vectors and project them into frozen customer GNN space."""

    def __init__(
        self,
        *,
        customer_pain_features: np.ndarray,
        customer_audience_features: np.ndarray,
        customer_embeddings: np.ndarray,
        pain_id_to_subtheme: dict[str, int],
        audience_id_to_index: dict[str, int],
        subtheme_idf: np.ndarray | None = None,
    ) -> None:
        if customer_pain_features.ndim != 2:
            raise ValueError("customer_pain_features must be a 2D array")
        if customer_audience_features.ndim != 2:
            raise ValueError("customer_audience_features must be a 2D array")
        if customer_embeddings.ndim != 2:
            raise ValueError("customer_embeddings must be a 2D array")

        n_customers = customer_pain_features.shape[0]
        if customer_audience_features.shape[0] != n_customers:
            raise ValueError("pain and audience customer counts differ")
        if customer_embeddings.shape[0] != n_customers:
            raise ValueError("feature and embedding customer counts differ")

        self.customer_pain_features = np.asarray(customer_pain_features, dtype=np.float32)
        self.customer_audience_features = np.asarray(
            customer_audience_features, dtype=np.float32
        )
        self.customer_embeddings = _normalize_rows(
            np.asarray(customer_embeddings, dtype=np.float32)
        )
        self.pain_id_to_subtheme = dict(pain_id_to_subtheme)
        self.audience_id_to_index = dict(audience_id_to_index)

        if subtheme_idf is None:
            document_frequency = np.count_nonzero(
                self.customer_pain_features > 0, axis=0
            )
            subtheme_idf = np.log(
                n_customers / (document_frequency.astype(np.float32) + 1.0)
            )
        self.subtheme_idf = np.asarray(subtheme_idf, dtype=np.float32)

        if self.subtheme_idf.shape != (self.customer_pain_features.shape[1],):
            raise ValueError("subtheme_idf has the wrong shape")

        self._pain_norm = _normalize_rows(self.customer_pain_features)
        self._audience_norm = _normalize_rows(self.customer_audience_features)

    def build_vectors(self, profile: ProspectProfile) -> tuple[np.ndarray, np.ndarray]:
        pain = np.zeros(self.customer_pain_features.shape[1], dtype=np.float32)
        for pain_id in set(profile.pain_ids):
            subtheme_index = self.pain_id_to_subtheme.get(pain_id)
            if subtheme_index is not None:
                pain[subtheme_index] += 1.0
        pain *= self.subtheme_idf

        audience = np.zeros(
            self.customer_audience_features.shape[1], dtype=np.float32
        )
        for audience_id in set(profile.audience_ids):
            audience_index = self.audience_id_to_index.get(audience_id)
            if audience_index is not None:
                audience[audience_index] = 1.0
        return pain, audience

    def project(
        self,
        profile: ProspectProfile,
        *,
        top_k: int = 20,
        pain_weight: float = 0.7,
        audience_weight: float = 0.3,
        min_similarity: float = 0.0,
        exclude_customer_index: int | None = None,
    ) -> LookalikeProjection:
        pain, audience = self.build_vectors(profile)
        return self.project_vectors(
            pain,
            audience,
            top_k=top_k,
            pain_weight=pain_weight,
            audience_weight=audience_weight,
            min_similarity=min_similarity,
            exclude_customer_index=exclude_customer_index,
        )

    def project_vectors(
        self,
        pain: np.ndarray,
        audience: np.ndarray,
        *,
        top_k: int = 20,
        pain_weight: float = 0.7,
        audience_weight: float = 0.3,
        min_similarity: float = 0.0,
        exclude_customer_index: int | None = None,
    ) -> LookalikeProjection:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if pain_weight < 0 or audience_weight < 0:
            raise ValueError("feature weights must be non-negative")

        pain = np.asarray(pain, dtype=np.float32)
        audience = np.asarray(audience, dtype=np.float32)
        if pain.shape != (self.customer_pain_features.shape[1],):
            raise ValueError("pain vector has the wrong shape")
        if audience.shape != (self.customer_audience_features.shape[1],):
            raise ValueError("audience vector has the wrong shape")
        has_pain = bool(np.linalg.norm(pain) > 0)
        has_audience = bool(np.linalg.norm(audience) > 0)

        if not has_pain and not has_audience:
            return LookalikeProjection(
                embedding=None,
                status="no_profile_features",
                profile_coverage=0.0,
                pain_coverage=0.0,
                audience_coverage=0.0,
                lookalikes=(),
            )

        active_weights: list[tuple[float, np.ndarray]] = []
        if has_pain and pain_weight > 0:
            active_weights.append((pain_weight, self._pain_norm @ _normalize(pain)))
        if has_audience and audience_weight > 0:
            active_weights.append(
                (audience_weight, self._audience_norm @ _normalize(audience))
            )
        if not active_weights:
            return LookalikeProjection(
                embedding=None,
                status="zero_active_weight",
                profile_coverage=0.0,
                pain_coverage=float(has_pain),
                audience_coverage=float(has_audience),
                lookalikes=(),
            )

        total_weight = sum(weight for weight, _ in active_weights)
        similarities = sum(
            weight * scores for weight, scores in active_weights
        ) / total_weight
        similarities = np.asarray(similarities, dtype=np.float32)

        if exclude_customer_index is not None:
            if 0 <= exclude_customer_index < similarities.shape[0]:
                similarities[exclude_customer_index] = -np.inf

        ranked = np.argsort(-similarities)
        selected = [
            int(index)
            for index in ranked
            if math.isfinite(float(similarities[index]))
            and float(similarities[index]) > min_similarity
        ][:top_k]

        if not selected:
            return LookalikeProjection(
                embedding=None,
                status="no_positive_lookalikes",
                profile_coverage=(float(has_pain) + float(has_audience)) / 2.0,
                pain_coverage=float(has_pain),
                audience_coverage=float(has_audience),
                lookalikes=(),
            )

        weights = np.asarray(
            [max(float(similarities[index]), 0.0) for index in selected],
            dtype=np.float32,
        )
        if float(weights.sum()) <= 0:
            weights = np.ones(len(selected), dtype=np.float32)
        weights /= weights.sum()

        embedding = weights @ self.customer_embeddings[selected]
        embedding = _normalize(embedding)
        lookalikes = tuple(
            {
                "customer_index": index,
                "similarity": round(float(similarities[index]), 6),
                "weight": round(float(weight), 6),
            }
            for index, weight in zip(selected, weights)
        )
        return LookalikeProjection(
            embedding=embedding,
            status="ok",
            profile_coverage=(float(has_pain) + float(has_audience)) / 2.0,
            pain_coverage=float(has_pain),
            audience_coverage=float(has_audience),
            lookalikes=lookalikes,
        )


class PainCanonicalMatcher:
    """Map free-text pains to canonical pain IDs using precomputed BGE vectors."""

    def __init__(
        self,
        *,
        pain_features_path: Path,
        pain_ids: Iterable[str],
        model_name: str,
        cache_folder: str | None = None,
    ) -> None:
        self.pain_features_path = pain_features_path
        self.pain_ids = tuple(pain_ids)
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._features: np.ndarray | None = None
        self._model: Any = None

    def _load(self) -> None:
        if self._features is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            SentenceTransformer = None

        features = np.load(self.pain_features_path)
        if features.shape[0] != len(self.pain_ids):
            raise ValueError("pain feature rows do not match pain ID ordering")
        self._features = _normalize_rows(np.asarray(features, dtype=np.float32))
        if SentenceTransformer is not None:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
            )
        else:
            self._model = _TransformersEncoder(
                self.model_name,
                cache_folder=self.cache_folder,
            )

    def match(
        self,
        pain_texts: Iterable[str],
        *,
        top_k: int = 5,
        threshold: float = 0.45,
    ) -> tuple[str, ...]:
        texts = tuple(text.strip() for text in pain_texts if text and text.strip())
        if not texts:
            return ()
        self._load()
        assert self._features is not None
        query = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matched: set[str] = set()
        for vector in np.asarray(query, dtype=np.float32):
            scores = self._features @ vector
            for index in np.argsort(-scores)[:top_k]:
                if float(scores[index]) >= threshold:
                    matched.add(self.pain_ids[int(index)])
        return tuple(sorted(matched))


def load_enrichment_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest JSONL enrichment record for each exact account key."""
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record.get("account_name") or record.get("name") or ""
            key = exact_key(name)
            if key:
                records[key] = record
    return records


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > 1e-8,
    )


class _TransformersEncoder:
    """Minimal SentenceTransformer-compatible BGE encoder fallback."""

    def __init__(self, model_name: str, *, cache_folder: str | None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers or transformers is required to map pains"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_folder,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_folder,
        )
        self._model.eval()

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        del show_progress_bar
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            output = self._model(**encoded)
            embeddings = output.last_hidden_state[:, 0]
            if normalize_embeddings:
                embeddings = self._torch.nn.functional.normalize(
                    embeddings, dim=-1
                )
        return embeddings.cpu().numpy()
