"""Programmatic self-checks for extracted rules/facts (04_PROMPT_ENGINEERING_SPEC.md #7,
01_METHOD_SPEC.md #3 Phase B).

These run BEFORE any repair LLM call and decide whether a repair call is
even attempted. They are pure functions over already-parsed Pydantic
objects, so they are fully unit-testable without Ollama.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import DefeasibleRule, GroundedFact

_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]*(\([^)]*\))?$", re.IGNORECASE)
_SECTION_REF_RE = re.compile(
    r"\bsection\s+(\d+[a-zA-Z]?(?:\([a-zA-Z0-9]+\))*)|"  # "section 152(d)(1)"
    r"§\s*(\d+[a-zA-Z]?(?:\([a-zA-Z0-9]+\))*)",  # "§152(d)(1)"
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def find_referenced_sections(text: str) -> list[str]:
    """Heuristically extract section identifiers mentioned in free text."""
    found = []
    for match in _SECTION_REF_RE.finditer(text):
        ident = match.group(1) or match.group(2)
        if ident:
            found.append(ident)
    # de-duplicate, preserve order
    seen: set[str] = set()
    ordered = []
    for ident in found:
        if ident not in seen:
            seen.add(ident)
            ordered.append(ident)
    return ordered


def is_exact_quote(quote: str, source_text: str) -> bool:
    return bool(quote) and quote in source_text


def _is_well_formed_predicate(name: str) -> bool:
    return bool(name) and bool(_PREDICATE_RE.match(name.strip()))


def validate_rule_against_source(rule: DefeasibleRule, source_text: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not is_exact_quote(rule.source_quote, source_text):
        errors.append("source_quote is not an exact substring of the statute text")

    if rule.op not in ("ALL", "ANY"):
        errors.append(f"op must be ALL or ANY, got {rule.op!r}")

    if not rule.needs_review and not rule.conditions:
        errors.append("conditions must be non-empty for a rule that is not marked needs_review")

    if not _is_well_formed_predicate(rule.p):
        errors.append(f"proposition {rule.p!r} is not a well-formed predicate name")

    for cond in rule.conditions:
        if not _is_well_formed_predicate(cond):
            errors.append(f"condition {cond!r} is not a well-formed predicate name")

    for exc in rule.exceptions:
        if not _is_well_formed_predicate(exc):
            errors.append(f"exception {exc!r} is not a well-formed predicate name")

    mentioned_sections = find_referenced_sections(source_text)
    if mentioned_sections and not rule.references:
        warnings.append(
            f"source text mentions section reference(s) {mentioned_sections} "
            "but rule.references is empty"
        )

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_grounded_fact(fact: GroundedFact, facts_text: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not _is_well_formed_predicate(fact.predicate):
        errors.append(f"predicate {fact.predicate!r} is not a well-formed predicate name")

    if fact.status in ("TRUE", "FALSE"):
        if not is_exact_quote(fact.source_quote, facts_text):
            errors.append(
                f"fact {fact.predicate!r} has status {fact.status} but source_quote "
                "is not an exact substring of the case facts text"
            )
    elif fact.status == "UNKNOWN" and fact.source_quote:
        warnings.append(f"fact {fact.predicate!r} is UNKNOWN but carries a source_quote")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)
