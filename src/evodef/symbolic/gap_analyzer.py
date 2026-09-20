"""Deterministic typed proof-gap analysis (01_METHOD_SPEC.md #3 Phase E).

No LLM decides the gap type or whether a gap exists; everything here is a
pure function of the proof trace plus the rule/chunk metadata that produced
it. `prompts/gap_query.txt` only *rewrites* an already-decided GapRecord into
retrieval queries (Phase F) - it is never consulted here.
"""
from __future__ import annotations

from ..schemas import DefeasibleRule, GapRecord, ProofNode

FORMALIZATION_CONFIDENCE_THRESHOLD = 0.5

_GAP_PRIORITY = {
    "UNRESOLVED_EXCEPTION": 1,
    "MISSING_REFERENCE": 1,
    "CONFLICT": 1,
    "FORMALIZATION_UNCERTAIN": 2,
    "MISSING_SUPPORT": 3,
}


def _iter_nodes(node: ProofNode):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def analyze_gaps(
    root: ProofNode,
    rules_by_id: dict[str, DefeasibleRule],
    known_source_ids: set[str],
) -> list[GapRecord]:
    """Walk the proof tree and emit one GapRecord per unresolved node.

    `known_source_ids` is the set of `source_id`s already present in the
    current evidence set (used to detect MISSING_REFERENCE: a rule
    references a section that is not yet in evidence).
    """
    gaps: list[GapRecord] = []
    seen_signatures: set[tuple[str, str]] = set()

    for node in _iter_nodes(root):
        if node.via_rule_id is None:
            continue  # leaf fact node, not a rule application
        rule = rules_by_id.get(node.via_rule_id)
        if rule is None:
            continue

        condition_names = set(rule.conditions)
        exception_names = set(rule.exceptions)
        by_predicate = {c.predicate: c for c in node.children}

        # --- FORMALIZATION_UNCERTAIN: the rule itself is uncertain -------
        if rule.needs_review or rule.confidence < FORMALIZATION_CONFIDENCE_THRESHOLD:
            _add(
                gaps,
                seen_signatures,
                GapRecord(
                    gap_type="FORMALIZATION_UNCERTAIN",
                    target_predicate=rule.p,
                    current_rule_id=rule.rule_id,
                    source_id=rule.source_id,
                    query_terms=[rule.p, rule.source_id],
                    priority=_GAP_PRIORITY["FORMALIZATION_UNCERTAIN"],
                ),
            )

        # --- MISSING_REFERENCE: referenced section not yet in evidence ---
        for ref in rule.references:
            if ref not in known_source_ids:
                _add(
                    gaps,
                    seen_signatures,
                    GapRecord(
                        gap_type="MISSING_REFERENCE",
                        target_predicate=rule.p,
                        current_rule_id=rule.rule_id,
                        source_id=ref,
                        query_terms=[ref, rule.p],
                        priority=_GAP_PRIORITY["MISSING_REFERENCE"],
                    ),
                )

        if node.status not in ("unknown",):
            continue  # only unresolved nodes generate support/exception gaps

        # --- UNRESOLVED_EXCEPTION ----------------------------------------
        for exc_name in exception_names:
            child = by_predicate.get(exc_name)
            if child is not None and child.status == "unknown":
                _add(
                    gaps,
                    seen_signatures,
                    GapRecord(
                        gap_type="UNRESOLVED_EXCEPTION",
                        target_predicate=exc_name,
                        current_rule_id=rule.rule_id,
                        source_id=rule.source_id,
                        query_terms=[exc_name, "exception", "exclusion"],
                        priority=_GAP_PRIORITY["UNRESOLVED_EXCEPTION"],
                    ),
                )

        # --- MISSING_SUPPORT: an unresolved condition ---------------------
        for cond_name in condition_names:
            child = by_predicate.get(cond_name)
            if child is not None and child.status == "unknown":
                _add(
                    gaps,
                    seen_signatures,
                    GapRecord(
                        gap_type="MISSING_SUPPORT",
                        target_predicate=cond_name,
                        current_rule_id=rule.rule_id,
                        source_id=rule.source_id,
                        query_terms=[cond_name, "definition", "condition"],
                        priority=_GAP_PRIORITY["MISSING_SUPPORT"],
                    ),
                )

    gaps.extend(detect_conflicting_rules(list(rules_by_id.values())))
    gaps.sort(key=lambda g: (g.priority, g.gap_type, g.target_predicate))
    return gaps


def _add(gaps: list[GapRecord], seen: set[tuple[str, str]], gap: GapRecord) -> None:
    sig = (gap.gap_type, gap.target_predicate)
    if sig in seen:
        return
    seen.add(sig)
    gaps.append(gap)


def detect_conflicting_rules(rules: list[DefeasibleRule]) -> list[GapRecord]:
    """CONFLICT: two extracted rules disagree on the definition of the same
    proposition (different op/conditions/exceptions) before deduplication
    would silently pick one.
    """
    by_predicate: dict[str, list[DefeasibleRule]] = {}
    for rule in rules:
        by_predicate.setdefault(rule.p, []).append(rule)

    conflicts: list[GapRecord] = []
    for predicate, defs in by_predicate.items():
        if len(defs) < 2:
            continue
        signatures = {(d.op, tuple(sorted(d.conditions)), tuple(sorted(d.exceptions))) for d in defs}
        if len(signatures) > 1:
            defs_sorted = sorted(defs, key=lambda d: d.rule_id)
            conflicts.append(
                GapRecord(
                    gap_type="CONFLICT",
                    target_predicate=predicate,
                    current_rule_id=defs_sorted[0].rule_id,
                    source_id=defs_sorted[0].source_id,
                    query_terms=[predicate, "conflicting definition"],
                    priority=_GAP_PRIORITY["CONFLICT"],
                )
            )
    return conflicts
