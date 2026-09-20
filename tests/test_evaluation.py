from __future__ import annotations

from evodef.evaluation.error_analysis import build_error_breakdown, classify_error
from evodef.evaluation.metrics import (
    accuracy,
    citation_coverage,
    complete_evidence_at_k,
    counterfactual_validity,
    exception_coverage,
    hit_at_k,
    macro_f1,
    mrr,
    proof_executable_rate,
    recall_at_k,
    retrieval_metrics_for_cases,
    source_link_coverage,
    undetermined_rate,
)
from evodef.evaluation.stats import mcnemar_test, paired_bootstrap_ci


class TestClassificationMetrics:
    def test_accuracy(self):
        preds = [{"prediction": "A", "gold": "A"}, {"prediction": "A", "gold": "B"}]
        assert accuracy(preds) == 0.5

    def test_macro_f1_perfect(self):
        preds = [{"prediction": "A", "gold": "A"}, {"prediction": "B", "gold": "B"}]
        assert macro_f1(preds, ["A", "B"]) == 1.0

    def test_undetermined_rate(self):
        preds = [{"undetermined": True}, {"undetermined": False}, {}]
        assert undetermined_rate(preds) == 1 / 3

    def test_empty_predictions_is_nan(self):
        import math

        assert math.isnan(accuracy([]))


class TestRetrievalMetrics:
    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], ["a", "c"], 2) == 0.5
        assert recall_at_k(["a", "b"], ["a"], 5) == 1.0

    def test_mrr(self):
        assert mrr(["x", "a", "b"], ["a"]) == 0.5
        assert mrr(["a"], ["a"]) == 1.0
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_hit_at_k(self):
        assert hit_at_k(["a", "b"], ["c", "b"], 5) == 1.0
        assert hit_at_k(["a"], ["c"], 5) == 0.0

    def test_complete_evidence(self):
        assert complete_evidence_at_k(["a", "b"], ["a", "b"], 5) == 1.0
        assert complete_evidence_at_k(["a"], ["a", "b"], 5) == 0.0

    def test_retrieval_metrics_for_cases_averages(self):
        cases_by_id = {"c1": {"gold_chunk_ids": ["g1"]}, "c2": {"gold_chunk_ids": ["g2"]}}
        preds = [
            {"case_id": "c1", "retrieved_initial": ["g1"]},
            {"case_id": "c2", "retrieved_initial": ["x", "y"]},
        ]
        metrics = retrieval_metrics_for_cases(cases_by_id, preds)
        assert metrics["recall_at_5"] == 0.5

    def test_cases_without_gold_chunks_are_excluded(self):
        cases_by_id = {"c1": {"gold_chunk_ids": []}}
        preds = [{"case_id": "c1", "retrieved_initial": []}]
        import math

        metrics = retrieval_metrics_for_cases(cases_by_id, preds)
        assert math.isnan(metrics["recall_at_5"])


class TestXaiMetrics:
    def _pred(self, decision, root_status, exceptions_checked=None, counterfactuals=None, supporting_rules=None, proof_valid=True):
        return {
            "proof": {
                "decision": decision,
                "root": {"predicate": "p", "status": root_status, "children": [], "via_rule_id": "r1", "source_ids": ["s1"]},
                "exceptions_checked": exceptions_checked or [],
                "counterfactuals": counterfactuals or [],
                "supporting_rules": supporting_rules if supporting_rules is not None else ["r1"],
                "proof_valid": proof_valid,
            }
        }

    def test_proof_executable_rate_consistent_trace(self):
        preds = [self._pred("ENTAILED", "true"), self._pred("NOT_ENTAILED", "false")]
        assert proof_executable_rate(preds) == 1.0

    def test_proof_executable_rate_catches_inconsistent_trace(self):
        preds = [self._pred("ENTAILED", "false")]  # decision doesn't match trace
        assert proof_executable_rate(preds) == 0.0

    def test_source_link_coverage(self):
        preds = [self._pred("ENTAILED", "true")]
        assert source_link_coverage(preds) == 1.0

    def test_source_link_coverage_missing_source(self):
        pred = self._pred("ENTAILED", "true")
        pred["proof"]["root"]["source_ids"] = []
        assert source_link_coverage([pred]) == 0.0

    def test_counterfactual_validity(self):
        preds = [self._pred("ENTAILED", "true", counterfactuals=[{"verified_by_reexecution": True}, {"verified_by_reexecution": False}])]
        assert counterfactual_validity(preds) == 0.5

    def test_exception_coverage(self):
        with_exc = self._pred("ENTAILED", "true", exceptions_checked=[{"predicate": "e", "status": "false"}])
        without_exc = self._pred("ENTAILED", "true", exceptions_checked=[])
        assert exception_coverage([with_exc, without_exc]) == 0.5


class TestCitationCoverage:
    def test_counts_predictions_with_sources(self):
        preds = [
            {"proof": {"supporting_sources": ["s1"]}},
            {"proof": {"supporting_sources": []}},
            {"proof": None},
        ]
        assert citation_coverage(preds) == 1 / 3

    def test_empty_is_nan(self):
        import math

        assert math.isnan(citation_coverage([]))


class TestErrorAnalysis:
    def test_correct_prediction_is_not_an_error(self):
        assert classify_error({"prediction": "A", "gold": "A"}, {}) is None

    def test_retrieval_miss(self):
        pred = {"prediction": "A", "gold": "B", "retrieved_initial": ["x"], "retrieved_targeted": []}
        case = {"gold_chunk_ids": ["g1"]}
        assert classify_error(pred, case) == "RETRIEVAL_MISS"

    def test_formalization_error_when_no_rules(self):
        pred = {"prediction": "A", "gold": "B", "retrieved_initial": ["g1"], "retrieved_targeted": [], "rules_used": [], "proof": None}
        case = {"gold_chunk_ids": ["g1"]}
        assert classify_error(pred, case) == "FORMALIZATION_ERROR"

    def test_fact_grounding_error(self):
        pred = {
            "prediction": "A", "gold": "B", "retrieved_initial": ["g1"], "retrieved_targeted": [],
            "rules_used": ["r1"], "proof": {"missing_conditions": ["x"], "exceptions_checked": [], "proof_valid": True},
        }
        case = {"gold_chunk_ids": ["g1"]}
        assert classify_error(pred, case) == "FACT_GROUNDING_ERROR"

    def test_exception_error(self):
        pred = {
            "prediction": "A", "gold": "B", "retrieved_initial": ["g1"], "retrieved_targeted": [],
            "rules_used": ["r1"], "proof": {"missing_conditions": [], "exceptions_checked": [{"status": "unknown"}], "proof_valid": True},
        }
        case = {"gold_chunk_ids": ["g1"]}
        assert classify_error(pred, case) == "EXCEPTION_ERROR"

    def test_label_mapping_error(self):
        pred = {
            "prediction": "A", "gold": "B", "retrieved_initial": ["g1"], "retrieved_targeted": [],
            "rules_used": ["r1"], "proof": {"missing_conditions": [], "exceptions_checked": [], "proof_valid": True},
            "undetermined": True,
        }
        case = {"gold_chunk_ids": ["g1"]}
        assert classify_error(pred, case) == "LABEL_MAPPING_ERROR"

    def test_build_error_breakdown_counts_all_types(self):
        cases_by_id = {"c1": {"gold_chunk_ids": ["g1"]}}
        preds = [
            {"case_id": "c1", "prediction": "A", "gold": "B", "retrieved_initial": [], "retrieved_targeted": []},
        ]
        breakdown = build_error_breakdown(preds, cases_by_id)
        assert breakdown["RETRIEVAL_MISS"] == 1
        assert sum(breakdown.values()) == 1


class TestStats:
    def test_mcnemar_no_discordant_pairs(self):
        result = mcnemar_test([True, True], [True, True])
        assert result["p_value"] == 1.0
        assert result["n_discordant"] == 0

    def test_mcnemar_detects_asymmetry(self):
        # B is right in 8 cases where A is wrong, A never uniquely right
        correct_a = [False] * 8 + [True] * 2
        correct_b = [True] * 8 + [True] * 2
        result = mcnemar_test(correct_a, correct_b)
        assert result["n_01_a_wrong_b_right"] == 8
        assert result["n_10_a_right_b_wrong"] == 0
        assert result["p_value"] < 0.05

    def test_mcnemar_requires_equal_length(self):
        import pytest

        with pytest.raises(ValueError):
            mcnemar_test([True], [True, False])

    def test_bootstrap_ci_contains_point_estimate(self):
        values_a = [0.0] * 5 + [1.0] * 5
        values_b = [1.0] * 10
        result = paired_bootstrap_ci(values_a, values_b, n_resamples=500, seed=1)
        assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
        assert result["point_estimate"] == 0.5

    def test_bootstrap_ci_deterministic_with_seed(self):
        values_a = [0.0, 1.0, 0.0, 1.0]
        values_b = [1.0, 1.0, 0.0, 0.0]
        r1 = paired_bootstrap_ci(values_a, values_b, n_resamples=200, seed=42)
        r2 = paired_bootstrap_ci(values_a, values_b, n_resamples=200, seed=42)
        assert r1 == r2
