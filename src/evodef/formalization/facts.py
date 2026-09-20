"""Fact grounding: case narrative -> GroundedFact list (01_METHOD_SPEC.md #3 Phase C).

G0/G1 variants from 04_PROMPT_ENGINEERING_SPEC.md #6:
  G0 = all_case_facts: ask for every fact the text supports, unconstrained.
  G1 = predicate_targeted: ask only for the specific predicates the current
       rule vocabulary needs (the default for the full pipeline, since Phase C
       says "extract only predicates needed by current rules").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ollama_client import LLMCallRecord, OllamaClient
from ..prompt_templates import PromptTemplate
from ..schemas import FactGroundingOutput, GroundedFact
from ..utils import normalize_predicate
from .validators import validate_grounded_fact


def facts_to_dict(facts: list[GroundedFact]) -> dict[str, str]:
    """Normalized-predicate -> lowercase status ("true"/"false"/"unknown"),
    the shape `symbolic.traced_evaluator.TracedRuleEvaluator.evaluate` expects.
    """
    return {normalize_predicate(f.predicate): f.status.lower() for f in facts}


@dataclass
class FactGroundingResult:
    facts: list[GroundedFact] = field(default_factory=list)
    schema_valid: bool = False
    validation_warnings: list[str] = field(default_factory=list)
    calls: list[LLMCallRecord] = field(default_factory=list)


def _predicate_request_block(predicates: list[str], mode: str) -> str:
    if mode == "all_case_facts" or not predicates:
        return "Extract every fact predicate the case narrative directly supports or contradicts."
    return "\n".join(f"- {p}" for p in predicates)


def ground_facts(
    client: OllamaClient,
    model: str,
    template: PromptTemplate,
    facts_text: str,
    required_predicates: list[str],
    variant: dict | None = None,
    temperature: float = 0,
) -> FactGroundingResult:
    variant = variant or {}
    mode = variant.get("mode", "predicate_targeted")
    predicate_block = _predicate_request_block(required_predicates, mode)
    user = template.render_user(facts_text=facts_text, predicate_requests=predicate_block)

    parsed, record = client.chat_json(model, template.system, user, FactGroundingOutput, temperature=temperature)
    calls = [record]

    if parsed is None:
        return FactGroundingResult(facts=[], schema_valid=False, calls=calls)

    accepted: list[GroundedFact] = []
    warnings: list[str] = []
    requested = {normalize_predicate(p) for p in required_predicates}
    for fact in parsed.facts:
        if mode != "all_case_facts" and normalize_predicate(fact.predicate) not in requested:
            warnings.append(f"dropped unrequested predicate {fact.predicate}")
            continue
        result = validate_grounded_fact(fact, facts_text)
        warnings.extend(result.warnings)
        if result.valid:
            accepted.append(fact)
        else:
            warnings.extend(f"dropped fact {fact.predicate}: {e}" for e in result.errors)

    return FactGroundingResult(facts=accepted, schema_valid=True, validation_warnings=warnings, calls=calls)
