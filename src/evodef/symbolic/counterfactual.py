"""Deterministic counterfactual generation + re-execution (01_METHOD_SPEC.md #7).

Candidates are read off the proof tree, not guessed by an LLM:
  - remove one necessary support fact/condition currently TRUE,
  - activate one currently FALSE/UNKNOWN exception.
Each candidate is re-executed through the symbolic engine; only flips are
kept in the final certificate (the caller filters on `verified_by_reexecution`).
"""
from __future__ import annotations

from ..schemas import Counterfactual, DefeasibleRule, ProofNode
from ..utils import normalize_predicate
from .traced_evaluator import TracedRuleEvaluator


def _walk(
    node: ProofNode,
    rules_by_id: dict[str, DefeasibleRule],
    role: str | None,
    condition_leaves: list[ProofNode],
    exception_leaves: list[ProofNode],
) -> None:
    if node.via_rule_id is None:
        if role == "condition":
            condition_leaves.append(node)
        elif role == "exception":
            exception_leaves.append(node)
        return
    rule = rules_by_id.get(node.via_rule_id)
    if rule is None:
        return
    condition_names = set(rule.conditions)
    exception_names = set(rule.exceptions)
    for child in node.children:
        if child.predicate in condition_names:
            child_role = "condition"
        elif child.predicate in exception_names:
            child_role = "exception"
        else:
            child_role = role
        _walk(child, rules_by_id, child_role, condition_leaves, exception_leaves)


def generate_counterfactuals(
    rules: list[DefeasibleRule],
    facts: dict[str, str],
    target: str,
    root: ProofNode,
) -> list[Counterfactual]:
    rules_by_id = {r.rule_id: r for r in rules}
    condition_leaves: list[ProofNode] = []
    exception_leaves: list[ProofNode] = []
    _walk(root, rules_by_id, None, condition_leaves, exception_leaves)

    evaluator = TracedRuleEvaluator(rules)
    original_status = root.status

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node in condition_leaves:
        key = normalize_predicate(node.predicate)
        if node.status == "true" and key not in seen:
            seen.add(key)
            candidates.append(("remove_support", node.predicate))
    for node in exception_leaves:
        key = normalize_predicate(node.predicate)
        if node.status in ("false", "unknown") and key not in seen:
            seen.add(key)
            candidates.append(("activate_exception", node.predicate))

    counterfactuals: list[Counterfactual] = []
    for kind, predicate in candidates:
        new_facts = dict(facts)
        norm_key = normalize_predicate(predicate)
        if kind == "remove_support":
            new_facts[norm_key] = "false"
            intervention = f"set {predicate}=false"
            predicted_effect = f"decision for {target} is no longer supported"
        else:
            new_facts[norm_key] = "true"
            intervention = f"set {predicate}=true"
            predicted_effect = f"decision for {target} flips due to the activated exception"

        new_status = evaluator.evaluate(new_facts, target).status
        verified = new_status != original_status
        counterfactuals.append(
            Counterfactual(
                intervention=intervention,
                predicted_effect=predicted_effect,
                verified_by_reexecution=verified,
            )
        )

    return counterfactuals
