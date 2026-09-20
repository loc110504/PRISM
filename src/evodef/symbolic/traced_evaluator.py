"""Traced, 3-valued (true/false/unknown) recursive evaluator with exception-first
defeasible semantics (01_METHOD_SPEC.md #2.4, #3 Phase D).

Mirrors the official PYTHEN `RuleTreeEvaluator`'s exception-first, ALL/ANY
Boolean semantics (see symbolic/pythen_adapter.py docstring) but adds an
explicit UNKNOWN status for facts the case narrative did not resolve, which
the official closed-world evaluator cannot represent. `evaluate(..., closed_world=True)`
degrades to the official semantics (missing fact -> false) specifically so
tests can assert Boolean equivalence (03_IMPLEMENTATION_SPEC.md #5/#9).
"""
from __future__ import annotations

from ..schemas import DefeasibleRule, ProofNode
from ..utils import normalize_predicate
from .pythen_adapter import dedupe_rules

FactStatus = str  # "true" | "false" | "unknown"


class TracedRuleEvaluator:
    """Predicate lookups are argument-insensitive; see `utils.normalize_predicate`."""

    def __init__(self, rules: list[DefeasibleRule]):
        self.rules = dedupe_rules(rules)
        self.node_map: dict[str, DefeasibleRule] = {normalize_predicate(r.p): r for r in self.rules}

    def evaluate(
        self,
        facts: dict[str, FactStatus],
        target_predicate: str,
        closed_world: bool = False,
    ) -> ProofNode:
        return self._eval(target_predicate, facts, closed_world, frozenset())

    # ------------------------------------------------------------------
    def _fact_status(self, predicate: str, facts: dict[str, FactStatus], closed_world: bool) -> FactStatus:
        status = facts.get(normalize_predicate(predicate))
        if status is None:
            return "false" if closed_world else "unknown"
        status = status.lower()
        return status if status in ("true", "false", "unknown") else "unknown"

    def _eval(
        self,
        predicate: str,
        facts: dict[str, FactStatus],
        closed_world: bool,
        visiting: frozenset[str],
    ) -> ProofNode:
        norm_predicate = normalize_predicate(predicate)
        if norm_predicate in visiting:
            # Cyclic rule reference: cannot be resolved deterministically.
            return ProofNode(predicate=predicate, status="unknown")

        rule = self.node_map.get(norm_predicate)
        if rule is None:
            status = self._fact_status(predicate, facts, closed_world)
            return ProofNode(predicate=predicate, status=status)

        next_visiting = visiting | {norm_predicate}
        condition_children = [self._eval(c, facts, closed_world, next_visiting) for c in rule.conditions]
        exception_children = [self._eval(e, facts, closed_world, next_visiting) for e in rule.exceptions]

        exception_true = any(ch.status == "true" for ch in exception_children)
        exception_unknown_pred = next((ch.predicate for ch in exception_children if ch.status == "unknown"), None)

        if rule.op == "ANY":
            cond_true = any(ch.status == "true" for ch in condition_children)
            cond_false = (not cond_true) and all(ch.status == "false" for ch in condition_children) if condition_children else False
        else:  # ALL (mirrors PYTHEN treating unlisted/legacy "AND" the same way)
            cond_false = any(ch.status == "false" for ch in condition_children)
            cond_true = (not cond_false) and all(ch.status == "true" for ch in condition_children)

        defeating_exception = None
        if exception_true:
            status = "defeated"
            defeating_exception = next(ch.predicate for ch in exception_children if ch.status == "true")
        elif cond_false:
            status = "false"
        elif cond_true and exception_unknown_pred is None:
            status = "true"
        else:
            status = "unknown"
            defeating_exception = exception_unknown_pred

        return ProofNode(
            predicate=predicate,
            status=status,
            via_rule_id=rule.rule_id,
            source_ids=[rule.source_id],
            children=condition_children + exception_children,
            defeating_exception=defeating_exception,
        )
