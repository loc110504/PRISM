"""Adapter around the official PYTHEN `RuleTreeEvaluator` (03_IMPLEMENTATION_SPEC.md #5).

PYTHEN is closed-world Boolean: `evaluate(facts, target)` treats every
predicate not explicitly in `facts` as false, and rejects an empty rule list
or duplicate propositions. This module strips EvoDef-RAG's provenance
metadata and handles those two edge cases deterministically so callers never
need to special-case "no rules retrieved yet".
"""
from __future__ import annotations

from pythen import RuleTreeEvaluator

from ..schemas import DefeasibleRule
from ..utils import normalize_predicate


def dedupe_rules(rules: list[DefeasibleRule]) -> list[DefeasibleRule]:
    """Keep one rule per NORMALIZED proposition (PYTHEN forbids duplicate
    propositions, and `to_pythen_dict()` normalizes `p` before evaluation, so
    two rules that only differ by argument list - e.g. `foo(X)` vs `foo(Y)` -
    must be deduped here too or the official evaluator would raise on them).

    Highest confidence wins; ties broken by rule_id for determinism
    (03_IMPLEMENTATION_SPEC.md #6).
    """
    best: dict[str, DefeasibleRule] = {}
    for rule in rules:
        key = normalize_predicate(rule.p)
        current = best.get(key)
        if current is None or (rule.confidence, rule.rule_id) > (current.confidence, current.rule_id):
            best[key] = rule
    return list(best.values())


def build_evaluator(rules: list[DefeasibleRule], enable_trace: bool = False) -> RuleTreeEvaluator | None:
    """Returns None for an empty rule set (PYTHEN refuses to construct on `[]`)."""
    deduped = dedupe_rules(rules)
    if not deduped:
        return None
    return RuleTreeEvaluator([r.to_pythen_dict() for r in deduped], enable_trace=enable_trace)


def evaluate_boolean(rules: list[DefeasibleRule], true_facts: list[str], target_predicate: str) -> bool:
    """Closed-world Boolean evaluation, used only for the equivalence unit
    test against `TracedRuleEvaluator` (03_IMPLEMENTATION_SPEC.md #5/#9).

    `true_facts` and `target_predicate` are normalized to match the
    normalized rule dicts PYTHEN actually evaluates against.
    """
    evaluator = build_evaluator(rules)
    if evaluator is None:
        return False
    normalized_facts = [normalize_predicate(f) for f in true_facts]
    return evaluator.evaluate(normalized_facts, normalize_predicate(target_predicate))
