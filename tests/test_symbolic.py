"""Milestone C acceptance: traced/official Boolean equivalence, gap typing,
counterfactual re-execution (03_IMPLEMENTATION_SPEC.md #5/#9, #01_METHOD_SPEC #3E/#7)."""
from __future__ import annotations

import itertools

from evodef.schemas import DefeasibleRule
from evodef.symbolic.counterfactual import generate_counterfactuals
from evodef.symbolic.gap_analyzer import analyze_gaps, detect_conflicting_rules
from evodef.symbolic.pythen_adapter import build_evaluator, dedupe_rules, evaluate_boolean
from evodef.symbolic.traced_evaluator import TracedRuleEvaluator


def _rule(rule_id, p, op, conditions, exceptions, source_id="s1", confidence=0.9, needs_review=False):
    return DefeasibleRule(
        rule_id=rule_id,
        p=p,
        op=op,
        conditions=conditions,
        exceptions=exceptions,
        references=[],
        source_id=source_id,
        source_quote="q",
        confidence=confidence,
        needs_review=needs_review,
    )


CONTRACT_RULES = [
    _rule("r1", "contract_voidable", "ANY", ["is_minor", "is_incapacitated"], ["for_necessities"]),
]

QUALIFYING_RULES = [
    _rule(
        "r1",
        "qualifying_relative(Person,Taxpayer,Year)",
        "ALL",
        ["relationship_test(Person,Taxpayer)", "gross_income_test(Person,Year)"],
        ["qualifying_child_of_other_taxpayer(Person,Year)"],
        source_id="sec_152_d_1",
    ),
]


class TestPythenEquivalence:
    """official_boolean == traced_boolean for every combination of true facts
    (03_IMPLEMENTATION_SPEC.md #5: ">=99.9% of executable cases")."""

    def test_all_boolean_combinations_contract_voidable(self):
        atoms = ["is_minor", "is_incapacitated", "for_necessities"]
        for bits in itertools.product([False, True], repeat=len(atoms)):
            true_facts = [a for a, b in zip(atoms, bits) if b]
            official = evaluate_boolean(CONTRACT_RULES, true_facts, "contract_voidable")
            facts_dict = {a: ("true" if b else "false") for a, b in zip(atoms, bits)}
            traced = TracedRuleEvaluator(CONTRACT_RULES).evaluate(facts_dict, "contract_voidable", closed_world=True)
            assert official == (traced.status == "true"), (bits, official, traced.status)

    def test_all_boolean_combinations_qualifying_relative(self):
        atoms = ["relationship_test", "gross_income_test", "qualifying_child_of_other_taxpayer"]
        for bits in itertools.product([False, True], repeat=len(atoms)):
            true_facts = [a for a, b in zip(atoms, bits) if b]
            official = evaluate_boolean(QUALIFYING_RULES, true_facts, "qualifying_relative")
            facts_dict = {a: ("true" if b else "false") for a, b in zip(atoms, bits)}
            traced = TracedRuleEvaluator(QUALIFYING_RULES).evaluate(facts_dict, "qualifying_relative", closed_world=True)
            assert official == (traced.status == "true"), (bits, official, traced.status)

    def test_empty_rule_list_is_false_not_a_crash(self):
        assert evaluate_boolean([], ["anything"], "target") is False
        assert build_evaluator([]) is None

    def test_argument_lists_are_normalized_away(self):
        # "qualifying_relative(Person,Taxpayer,Year)" must match a case-grounded
        # "qualifying_relative(alice,bob,2018)" target, and case-grounded fact
        # atoms with concrete args must satisfy Variable-named conditions.
        official = evaluate_boolean(
            QUALIFYING_RULES,
            ["relationship_test(alice,bob)", "gross_income_test(alice,2018)"],
            "qualifying_relative(alice,bob,2018)",
        )
        assert official is True


class TestTracedEvaluatorOpenWorld:
    def test_unknown_condition_yields_unknown_status_and_defeating_exception_none(self):
        facts = {"relationship_test": "true", "qualifying_child_of_other_taxpayer": "false"}
        node = TracedRuleEvaluator(QUALIFYING_RULES).evaluate(facts, "qualifying_relative")
        assert node.status == "unknown"
        assert node.defeating_exception is None  # blocked by a missing SUPPORT condition, not an exception

    def test_unresolved_exception_sets_defeating_exception(self):
        facts = {"relationship_test": "true", "gross_income_test": "true"}
        node = TracedRuleEvaluator(QUALIFYING_RULES).evaluate(facts, "qualifying_relative")
        assert node.status == "unknown"
        assert node.defeating_exception == "qualifying_child_of_other_taxpayer(Person,Year)"

    def test_dedupe_keeps_highest_confidence(self):
        low = _rule("r_low", "p", "ALL", ["a"], [], confidence=0.3)
        high = _rule("r_high", "p", "ALL", ["b"], [], confidence=0.9)
        deduped = dedupe_rules([low, high])
        assert len(deduped) == 1
        assert deduped[0].rule_id == "r_high"

    def test_cycle_does_not_infinite_loop(self):
        cyclic = [
            _rule("r1", "a", "ALL", ["b"], []),
            _rule("r2", "b", "ALL", ["a"], []),
        ]
        node = TracedRuleEvaluator(cyclic).evaluate({}, "a")
        assert node.status == "unknown"


class TestGapAnalyzer:
    def test_missing_support_gap(self):
        node = TracedRuleEvaluator(QUALIFYING_RULES).evaluate({"relationship_test": "true"}, "qualifying_relative")
        rules_by_id = {r.rule_id: r for r in QUALIFYING_RULES}
        gaps = analyze_gaps(node, rules_by_id, known_source_ids={"sec_152_d_1"})
        types = {g.gap_type for g in gaps}
        assert "MISSING_SUPPORT" in types
        support_gap = next(g for g in gaps if g.gap_type == "MISSING_SUPPORT")
        assert support_gap.target_predicate == "gross_income_test(Person,Year)"

    def test_unresolved_exception_gap(self):
        facts = {"relationship_test": "true", "gross_income_test": "true"}
        node = TracedRuleEvaluator(QUALIFYING_RULES).evaluate(facts, "qualifying_relative")
        rules_by_id = {r.rule_id: r for r in QUALIFYING_RULES}
        gaps = analyze_gaps(node, rules_by_id, known_source_ids={"sec_152_d_1"})
        assert any(g.gap_type == "UNRESOLVED_EXCEPTION" for g in gaps)

    def test_missing_reference_gap(self):
        rule = _rule("r1", "p", "ALL", ["a"], [], source_id="sec_1")
        rule.references = ["7703"]
        node = TracedRuleEvaluator([rule]).evaluate({"a": "true"}, "p")
        gaps = analyze_gaps(node, {"r1": rule}, known_source_ids={"sec_1"})
        assert any(g.gap_type == "MISSING_REFERENCE" and g.source_id == "7703" for g in gaps)
        # referenced section already in evidence -> no gap
        gaps2 = analyze_gaps(node, {"r1": rule}, known_source_ids={"sec_1", "7703"})
        assert not any(g.gap_type == "MISSING_REFERENCE" for g in gaps2)

    def test_formalization_uncertain_gap(self):
        rule = _rule("r1", "p", "ALL", ["a"], [], confidence=0.9, needs_review=True)
        node = TracedRuleEvaluator([rule]).evaluate({"a": "true"}, "p")
        gaps = analyze_gaps(node, {"r1": rule}, known_source_ids=set())
        assert any(g.gap_type == "FORMALIZATION_UNCERTAIN" for g in gaps)

    def test_conflict_detection(self):
        r1 = _rule("r1", "p", "ALL", ["a"], [])
        r2 = _rule("r2", "p", "ANY", ["b"], [])
        conflicts = detect_conflicting_rules([r1, r2])
        assert len(conflicts) == 1
        assert conflicts[0].gap_type == "CONFLICT"

    def test_no_conflict_when_rules_identical(self):
        r1 = _rule("r1", "p", "ALL", ["a"], [])
        r2 = _rule("r2", "p", "ALL", ["a"], [])
        assert detect_conflicting_rules([r1, r2]) == []


class TestCounterfactual:
    def test_removing_support_flips_true_to_not_true(self):
        facts = {"relationship_test": "true", "gross_income_test": "true", "qualifying_child_of_other_taxpayer": "false"}
        root = TracedRuleEvaluator(QUALIFYING_RULES).evaluate(facts, "qualifying_relative")
        assert root.status == "true"
        cfs = generate_counterfactuals(QUALIFYING_RULES, facts, "qualifying_relative", root)
        assert any(cf.verified_by_reexecution for cf in cfs)
        verified_predicates = {cf.intervention for cf in cfs if cf.verified_by_reexecution}
        assert any("gross_income_test" in v or "relationship_test" in v for v in verified_predicates)

    def test_activating_exception_flips_decision(self):
        facts = {"relationship_test": "true", "gross_income_test": "true", "qualifying_child_of_other_taxpayer": "false"}
        root = TracedRuleEvaluator(QUALIFYING_RULES).evaluate(facts, "qualifying_relative")
        cfs = generate_counterfactuals(QUALIFYING_RULES, facts, "qualifying_relative", root)
        exception_cfs = [cf for cf in cfs if "qualifying_child_of_other_taxpayer" in cf.intervention]
        assert len(exception_cfs) == 1
        assert exception_cfs[0].verified_by_reexecution is True

    def test_no_spurious_counterfactuals_when_no_rules(self):
        root = TracedRuleEvaluator([]).evaluate({}, "target")
        cfs = generate_counterfactuals([], {}, "target", root)
        assert cfs == []
