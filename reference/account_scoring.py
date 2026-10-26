"""Verbatim recovered scoring functions; no CRM inputs or search clients bundled."""
from __future__ import annotations
from datetime import date
from typing import Any

def _recency_multiplier(assessment: dict[str, Any], *, reference_date: date | None = None) -> float:
    """Compute a recency multiplier (1.0–2.0) based on how recent the evidence is.

    Evidence within 12 months of reference_date gets up to 2.0x;
    evidence 12–24 months gets 1.5x; 24–36 months gets 1.2x; older gets 1.0x.
    Uses the most recent dated evidence item.
    """
    ref = reference_date or date.today()
    best_date: date | None = None

    for hit in assessment.get("evidence_hits", []):
        d = hit.get("published_date")
        if not d or d == "undated":
            continue
        try:
            parsed = date.fromisoformat(d[:10])
            if best_date is None or parsed > best_date:
                best_date = parsed
        except (ValueError, TypeError):
            continue

    if best_date is None:
        return 1.0

    months_ago = (ref.year - best_date.year) * 12 + (ref.month - best_date.month)
    if months_ago <= 12:
        return 2.0
    elif months_ago <= 24:
        return 1.5
    elif months_ago <= 36:
        return 1.2
    return 1.0


def _score_lead(assessments: list[dict[str, Any]], *, cutoff_date: str | None = None) -> dict[str, Any]:
    """Compute a lead score from evidence assessments with recency weighting."""
    grade_weights = {"CONFIRMED": 10, "PROBABLE": 5, "SUGGESTIVE": 2, "INSUFFICIENT": 0}
    ref = date.fromisoformat(cutoff_date) if cutoff_date else date.today()

    grades = [a.get("assessment", {}).get("grade", "INSUFFICIENT") for a in assessments]
    confidences = [
        a.get("assessment", {}).get("confidence", 0.0) for a in assessments
        if a.get("assessment", {}).get("grade") != "INSUFFICIENT"
    ]

    raw_score = sum(grade_weights.get(g, 0) for g in grades)
    recency_score = sum(
        grade_weights.get(g, 0) * _recency_multiplier(a, reference_date=ref)
        for a, g in zip(assessments, grades)
    )
    n_with_evidence = sum(1 for g in grades if g != "INSUFFICIENT")
    coverage = n_with_evidence / len(grades) if grades else 0
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0

    composite = raw_score * (0.5 + 0.5 * coverage)
    recency_composite = recency_score * (0.5 + 0.5 * coverage)

    return {
        "composite_score": round(composite, 1),
        "recency_score": round(recency_composite, 1),
        "raw_score": raw_score,
        "coverage": round(coverage, 3),
        "mean_confidence": round(mean_confidence, 3),
        "n_pains": len(grades),
        "n_with_evidence": n_with_evidence,
        "grades": {g: grades.count(g) for g in ["CONFIRMED", "PROBABLE", "SUGGESTIVE", "INSUFFICIENT"]},
    }
