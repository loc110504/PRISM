"""Deterministic metrics (06_EVALUATION_AND_STATS_SPEC.md). No LLM judge
anywhere in this module (06_EVALUATION_AND_STATS_SPEC.md #9).

Every function takes plain dicts (as read from a `PredictionRow`/`Case`
JSONL file) rather than the Pydantic models, so scripts can compute metrics
straight off `outputs/raw_predictions/*.jsonl` without re-importing the
pipeline.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Classification (06_EVALUATION_AND_STATS_SPEC.md #1)
# ---------------------------------------------------------------------------


def accuracy(predictions: list[dict]) -> float:
    if not predictions:
        return float("nan")
    correct = sum(1 for p in predictions if p["prediction"] == p["gold"])
    return correct / len(predictions)


def _per_class_prf1(predictions: list[dict], labels: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for label in labels:
        tp = sum(1 for p in predictions if p["prediction"] == label and p["gold"] == label)
        fp = sum(1 for p in predictions if p["prediction"] == label and p["gold"] != label)
        fn = sum(1 for p in predictions if p["prediction"] != label and p["gold"] == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def macro_f1(predictions: list[dict], labels: list[str] | None = None) -> float:
    if not predictions:
        return float("nan")
    labels = labels or sorted({p["gold"] for p in predictions} | {p["prediction"] for p in predictions})
    per_class = _per_class_prf1(predictions, labels)
    return sum(v["f1"] for v in per_class.values()) / len(labels)


def per_class_metrics(predictions: list[dict], labels: list[str] | None = None) -> dict[str, dict[str, float]]:
    labels = labels or sorted({p["gold"] for p in predictions} | {p["prediction"] for p in predictions})
    return _per_class_prf1(predictions, labels)


def citation_coverage(predictions: list[dict]) -> float:
    """Fraction of ALL predictions whose certificate cites >=1 source
    (00_MASTER_SPEC.md Table 1 "Citation Coverage"). Methods with no
    symbolic proof (M0/M1/M2) score 0 here by definition - they have no
    formal citation mechanism, which is itself part of the comparison.
    """
    if not predictions:
        return float("nan")
    cited = sum(1 for p in predictions if p.get("proof") and p["proof"].get("supporting_sources"))
    return cited / len(predictions)


def undetermined_rate(predictions: list[dict]) -> float:
    if not predictions:
        return float("nan")
    return sum(1 for p in predictions if p.get("undetermined")) / len(predictions)


# ---------------------------------------------------------------------------
# Retrieval (06_EVALUATION_AND_STATS_SPEC.md #2)
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    top_k = set(retrieved[:k])
    hit = sum(1 for g in gold if g in top_k)
    return hit / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return float("nan")
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    top_k = set(retrieved[:k])
    return 1.0 if any(g in top_k for g in gold) else 0.0


def complete_evidence_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    top_k = set(retrieved[:k])
    return 1.0 if all(g in top_k for g in gold) else 0.0


def retrieval_metrics_for_cases(
    cases_by_id: dict[str, dict],
    predictions: list[dict],
    retrieved_field: str = "retrieved_initial",
) -> dict[str, float]:
    """Averages Recall@1/3/5, MRR, Hit@5, CompleteEvidence@5 over all
    predictions whose case has at least one gold chunk id.
    """
    rows = []
    for p in predictions:
        case = cases_by_id.get(p["case_id"])
        if case is None or not case.get("gold_chunk_ids"):
            continue
        retrieved = p.get(retrieved_field, []) or []
        gold = case["gold_chunk_ids"]
        rows.append(
            {
                "recall_at_1": recall_at_k(retrieved, gold, 1),
                "recall_at_3": recall_at_k(retrieved, gold, 3),
                "recall_at_5": recall_at_k(retrieved, gold, 5),
                "mrr": mrr(retrieved, gold),
                "hit_at_5": hit_at_k(retrieved, gold, 5),
                "complete_evidence_at_5": complete_evidence_at_k(retrieved, gold, 5),
            }
        )
    if not rows:
        return {k: float("nan") for k in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr", "hit_at_5", "complete_evidence_at_5"]}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


# ---------------------------------------------------------------------------
# XAI / auditability (06_EVALUATION_AND_STATS_SPEC.md #4, 05_EXPERIMENT_PLAN.md #6)
#
# Each metric re-derives its verdict from the STORED proof trace rather than
# trusting the pipeline's self-reported `decision`/`proof_valid` fields, so
# these are an audit, not an echo, of what the pipeline claimed.
# ---------------------------------------------------------------------------


def _status_to_decision(status: str) -> str:
    if status == "true":
        return "ENTAILED"
    if status in ("false", "defeated"):
        return "NOT_ENTAILED"
    return "UNDETERMINED"


def proof_executable_rate(predictions: list[dict]) -> float:
    """PER = valid re-executions / certificates emitted.

    "Re-execution" here means: independently re-deriving the decision from
    the certificate's own stored root trace and checking it reproduces the
    stored `decision` - i.e. the trace is internally consistent, not
    fabricated post-hoc.
    """
    certs = [p["proof"] for p in predictions if p.get("proof")]
    if not certs:
        return float("nan")
    valid = 0
    for cert in certs:
        root = cert.get("root")
        if root is None:
            continue
        if _status_to_decision(root["status"]) == cert["decision"]:
            valid += 1
    return valid / len(certs)


def source_link_coverage(predictions: list[dict]) -> float:
    """SLC = applied rule nodes with a non-empty source_id / all applied rule nodes."""
    total = 0
    covered = 0

    def walk(node):
        nonlocal total, covered
        if node.get("via_rule_id"):
            total += 1
            if node.get("source_ids"):
                covered += 1
        for child in node.get("children", []):
            walk(child)

    for p in predictions:
        cert = p.get("proof")
        if cert and cert.get("root"):
            walk(cert["root"])
    return covered / total if total else float("nan")


def counterfactual_validity(predictions: list[dict]) -> float:
    """CFV = verified-flip counterfactuals / emitted counterfactuals."""
    emitted = 0
    verified = 0
    for p in predictions:
        cert = p.get("proof")
        if not cert:
            continue
        for cf in cert.get("counterfactuals", []):
            emitted += 1
            if cf.get("verified_by_reexecution"):
                verified += 1
    return verified / emitted if emitted else float("nan")


def exception_coverage(predictions: list[dict]) -> float:
    """EC = proofs whose certificate reports >=1 exception status /
    proofs where at least one rule was applied at all.

    `_build_certificate` (pipeline.py) always appends to `exceptions_checked`
    whenever an applied rule declares an exception - by construction, this
    system either reports every exception it applies, or it applied none.
    A denominator of "gold rule has an exception" would need SARA-v2
    structural annotations, which 02_DATA_AND_SPLITS_SPEC.md #7 scopes to
    optional diagnostics only, so the achievable proxy here is "among
    reasoned cases, how many involved a reported exception check".
    """
    reasoned = 0
    reporting = 0
    for p in predictions:
        cert = p.get("proof")
        if not cert or not cert.get("supporting_rules"):
            continue
        reasoned += 1
        if cert.get("exceptions_checked"):
            reporting += 1
    return reporting / reasoned if reasoned else float("nan")
