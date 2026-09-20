"""Source-bounded autoformalization: statute chunk -> DefeasibleRule list.

Implements 01_METHOD_SPEC.md #3 Phase B and the F0-F3 prompt-tuning grid from
04_PROMPT_ENGINEERING_SPEC.md #6/#7 (schema-constrained extraction, at most
one deterministic-check-triggered repair call).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ollama_client import LLMCallRecord, OllamaClient
from ..prompt_templates import PromptTemplate
from ..schemas import DefeasibleRule, RuleExtractionOutput, StatuteChunk
from .validators import ValidationResult, validate_rule_against_source

CONCISE_OPERATOR_NOTE = (
    "Reminder: op=\"ALL\" means every listed condition is jointly required "
    "(conjunction); op=\"ANY\" means the text gives explicit alternative "
    "sufficient paths (disjunction). Do not use ANY for a plain list of "
    "required elements."
)

FORBID_XREF_NOTE = (
    "If the passage references another section, record the section identifier "
    "in `references` and leave the referenced predicate as an unresolved "
    "condition/exception name. Never fill in what you believe the referenced "
    "section says."
)

# Synthetic legal micro-rules (04_PROMPT_ENGINEERING_SPEC.md #5). Kept out of
# SARA text entirely to avoid any test leakage.
_SYNTHETIC_DEMOS: list[tuple[str, dict]] = [
    (
        "A benefit applies if the person is a resident and has valid registration, "
        "unless the registration has been revoked.",
        {
            "rules": [
                {
                    "rule_id": "R_demo_1",
                    "p": "benefit_applies(Person)",
                    "op": "ALL",
                    "conditions": ["is_resident(Person)", "has_valid_registration(Person)"],
                    "exceptions": ["registration_revoked(Person)"],
                    "references": [],
                    "source_id": "demo_1",
                    "source_quote": (
                        "A benefit applies if the person is a resident and has valid "
                        "registration, unless the registration has been revoked."
                    ),
                    "confidence": 1.0,
                    "needs_review": False,
                }
            ]
        },
    ),
    (
        "A license applies if the applicant holds a class A permit or a class B permit.",
        {
            "rules": [
                {
                    "rule_id": "R_demo_2",
                    "p": "license_applies(Applicant)",
                    "op": "ANY",
                    "conditions": ["class_a_permit(Applicant)", "class_b_permit(Applicant)"],
                    "exceptions": [],
                    "references": [],
                    "source_id": "demo_2",
                    "source_quote": (
                        "A license applies if the applicant holds a class A permit "
                        "or a class B permit."
                    ),
                    "confidence": 1.0,
                    "needs_review": False,
                }
            ]
        },
    ),
    (
        "The exemption is subject to the requirements described in Section 9.",
        {
            "rules": [
                {
                    "rule_id": "R_demo_3",
                    "p": "exemption_applies(Person)",
                    "op": "ALL",
                    "conditions": ["meets_section_9_requirements(Person)"],
                    "exceptions": [],
                    "references": ["9"],
                    "source_id": "demo_3",
                    "source_quote": "The exemption is subject to the requirements described in Section 9.",
                    "confidence": 0.6,
                    "needs_review": True,
                }
            ]
        },
    ),
]


def _demo_block(n_demos: int) -> str:
    if n_demos <= 0:
        return ""
    import json

    parts = ["DEMONSTRATIONS (synthetic, not from the case corpus):"]
    for text, rules_json in _SYNTHETIC_DEMOS[:n_demos]:
        parts.append(f"STATUTE TEXT:\n{text}\nOUTPUT:\n{json.dumps(rules_json)}")
    return "\n\n".join(parts)


@dataclass
class FormalizationResult:
    rules: list[DefeasibleRule] = field(default_factory=list)
    schema_valid: bool = False
    repair_used: bool = False
    accepted_count: int = 0
    rejected_count: int = 0
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    calls: list[LLMCallRecord] = field(default_factory=list)


def _filter_valid(output: RuleExtractionOutput | None, chunk: StatuteChunk) -> tuple[list[DefeasibleRule], list[str], list[str]]:
    if output is None:
        return [], ["schema parse failed"], []
    accepted: list[DefeasibleRule] = []
    errors: list[str] = []
    warnings: list[str] = []
    for rule in output.rules:
        result: ValidationResult = validate_rule_against_source(rule, chunk.text)
        warnings.extend(result.warnings)
        if result.valid:
            accepted.append(rule)
        else:
            errors.extend(f"rule {rule.rule_id}: {e}" for e in result.errors)
    return accepted, errors, warnings


def formalize_chunk(
    client: OllamaClient,
    model: str,
    template: PromptTemplate,
    chunk: StatuteChunk,
    variant: dict | None = None,
    temperature: float = 0,
) -> FormalizationResult:
    variant = variant or {}
    system = template.system
    if variant.get("concise_operator_instructions"):
        system = system + "\n\n" + CONCISE_OPERATOR_NOTE
    if variant.get("forbid_cross_reference_resolution"):
        system = system + "\n\n" + FORBID_XREF_NOTE

    user = template.render_user(source_id=chunk.chunk_id, statute_text=chunk.text)
    demo_block = _demo_block(variant.get("demonstrations", 0))
    if demo_block:
        user = demo_block + "\n\n" + user

    calls: list[LLMCallRecord] = []
    parsed, record = client.chat_json(model, system, user, RuleExtractionOutput, temperature=temperature)
    calls.append(record)
    accepted, errors, warnings = _filter_valid(parsed, chunk)
    schema_valid = parsed is not None
    repair_used = False

    if parsed is None or errors:
        repair_used = True
        error_list = "\n".join(f"- {e}" for e in errors) or "- output did not match the required schema"
        repair_user = (
            f"{user}\n\nYour previous output had validation errors:\n{error_list}\n"
            "Return corrected JSON only according to the supplied schema."
        )
        parsed2, record2 = client.chat_json(model, system, repair_user, RuleExtractionOutput, temperature=temperature)
        calls.append(record2)
        if parsed2 is not None:
            schema_valid = True
            accepted, errors, warnings = _filter_valid(parsed2, chunk)

    return FormalizationResult(
        rules=accepted,
        schema_valid=schema_valid,
        repair_used=repair_used,
        accepted_count=len(accepted),
        rejected_count=len(errors),
        validation_errors=errors,
        validation_warnings=warnings,
        calls=calls,
    )
