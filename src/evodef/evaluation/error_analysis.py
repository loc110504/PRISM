"""Automatic error taxonomy for wrong predictions (05_EXPERIMENT_PLAN.md #7).

Each wrong case is assigned to the FIRST applicable category, checked in the
exact order the spec lists them.
"""
from __future__ import annotations

from collections import Counter

ERROR_TYPES = [
    "RETRIEVAL_MISS",
    "FORMALIZATION_ERROR",
    "FACT_GROUNDING_ERROR",
    "EXCEPTION_ERROR",
    "SYMBOLIC_ERROR",
    "LABEL_MAPPING_ERROR",
    "OTHER",
]


def classify_error(prediction: dict, case: dict) -> str | None:
    """Returns None if the prediction was correct (not an error)."""
    if prediction["prediction"] == prediction["gold"]:
        return None

    gold_chunks = set(case.get("gold_chunk_ids", []))
    retrieved = set(prediction.get("retrieved_initial", [])) | set(prediction.get("retrieved_targeted", []))
    if gold_chunks and not (gold_chunks & retrieved):
        return "RETRIEVAL_MISS"

    proof = prediction.get("proof")
    if proof is None or not prediction.get("rules_used"):
        return "FORMALIZATION_ERROR"

    if proof.get("missing_conditions"):
        return "FACT_GROUNDING_ERROR"

    if any(ec.get("status") == "unknown" for ec in proof.get("exceptions_checked", [])):
        return "EXCEPTION_ERROR"

    if not proof.get("proof_valid", True):
        return "SYMBOLIC_ERROR"

    if prediction.get("undetermined"):
        return "LABEL_MAPPING_ERROR"

    return "OTHER"


def build_error_breakdown(predictions: list[dict], cases_by_id: dict[str, dict]) -> dict[str, int]:
    counts: Counter[str] = Counter({t: 0 for t in ERROR_TYPES})
    for p in predictions:
        case = cases_by_id.get(p["case_id"])
        if case is None:
            continue
        error_type = classify_error(p, case)
        if error_type is not None:
            counts[error_type] += 1
    return dict(counts)
