from __future__ import annotations

import json

from evodef.formalization.facts import facts_to_dict, ground_facts
from evodef.formalization.rules import formalize_chunk
from evodef.formalization.validators import (
    find_referenced_sections,
    is_exact_quote,
    validate_grounded_fact,
    validate_rule_against_source,
)
from evodef.ollama_client import OllamaClient
from evodef.prompt_templates import load_prompt_template
from evodef.schemas import DefeasibleRule, GroundedFact, StatuteChunk

from conftest import FakeOllamaBackend, json_response, REPO_ROOT

FORMALIZE_TEMPLATE = load_prompt_template(REPO_ROOT / "prompts" / "formalize_rule.txt")
GROUND_TEMPLATE = load_prompt_template(REPO_ROOT / "prompts" / "ground_facts.txt")

CHUNK = StatuteChunk(
    chunk_id="sec_152_d_1",
    section_id="152(d)(1)",
    text="A qualifying relative means an individual who bears a relationship and has income under section 7703.",
    source_file="section152",
)


def _valid_rule_payload():
    return {
        "rules": [
            {
                "rule_id": "R1",
                "p": "qualifying_relative",
                "op": "ALL",
                "conditions": ["relationship_test", "income_test"],
                "exceptions": [],
                "references": ["7703"],
                "source_id": "sec_152_d_1",
                "source_quote": "A qualifying relative means an individual who bears a relationship and has income under section 7703.",
                "confidence": 0.9,
                "needs_review": False,
            }
        ]
    }


class TestValidators:
    def test_find_referenced_sections(self):
        assert find_referenced_sections("as defined in section 7703") == ["7703"]
        assert find_referenced_sections("under §152(c)") == ["152(c)"]
        assert find_referenced_sections("no reference here") == []

    def test_is_exact_quote(self):
        assert is_exact_quote("hello", "say hello world")
        assert not is_exact_quote("goodbye", "say hello world")
        assert not is_exact_quote("", "say hello world")

    def test_rejects_inexact_quote(self):
        rule = DefeasibleRule(
            rule_id="r1", p="p", op="ALL", conditions=["a"], exceptions=[], references=[],
            source_id="s1", source_quote="not in source", confidence=0.9, needs_review=False,
        )
        result = validate_rule_against_source(rule, "the actual statute text")
        assert not result.valid
        assert any("source_quote" in e for e in result.errors)

    def test_rejects_empty_conditions_unless_needs_review(self):
        rule = DefeasibleRule(
            rule_id="r1", p="p", op="ALL", conditions=[], exceptions=[], references=[],
            source_id="s1", source_quote="text", confidence=0.9, needs_review=False,
        )
        assert not validate_rule_against_source(rule, "text").valid

    def test_grounded_fact_requires_exact_quote_when_true_or_false(self):
        fact = GroundedFact(predicate="p", status="TRUE", source_quote="not present", confidence=0.9)
        assert not validate_grounded_fact(fact, "case facts text").valid
        fact2 = GroundedFact(predicate="p", status="UNKNOWN", source_quote="", confidence=0.0)
        assert validate_grounded_fact(fact2, "case facts text").valid


class TestFormalizeChunk:
    def test_accepts_valid_rule_on_first_try(self):
        backend = FakeOllamaBackend(responses=[json_response(_valid_rule_payload())])
        client = OllamaClient(_backend=backend)
        result = formalize_chunk(client, "test-model", FORMALIZE_TEMPLATE, CHUNK)
        assert result.schema_valid
        assert not result.repair_used
        assert len(result.rules) == 1
        assert len(result.calls) == 1

    def test_repairs_once_on_bad_quote_then_accepts(self):
        bad_payload = _valid_rule_payload()
        bad_payload["rules"][0]["source_quote"] = "this text is not in the chunk"
        good_payload = _valid_rule_payload()
        backend = FakeOllamaBackend(responses=[json_response(bad_payload), json_response(good_payload)])
        client = OllamaClient(_backend=backend)
        result = formalize_chunk(client, "test-model", FORMALIZE_TEMPLATE, CHUNK)
        assert result.repair_used
        assert len(result.calls) == 2
        assert len(result.rules) == 1

    def test_never_crashes_on_unparseable_output(self):
        backend = FakeOllamaBackend(
            responses=[
                {"message": {"content": "not json at all"}},
                {"message": {"content": "still not json"}},
            ]
        )
        client = OllamaClient(_backend=backend, max_schema_retries=0)
        result = formalize_chunk(client, "test-model", FORMALIZE_TEMPLATE, CHUNK)
        assert result.rules == []
        assert result.schema_valid is False

    def test_needs_review_rule_still_requires_exact_source_quote(self):
        payload = {
            "rules": [
                {
                    "rule_id": "R1", "p": "ambiguous_p", "op": "ALL", "conditions": [],
                    "exceptions": [], "references": [], "source_id": "sec_152_d_1",
                    "source_quote": "not exact", "confidence": 0.4, "needs_review": True,
                }
            ]
        }
        backend = FakeOllamaBackend(responses=[json_response(payload), json_response(payload)])
        client = OllamaClient(_backend=backend)
        result = formalize_chunk(client, "test-model", FORMALIZE_TEMPLATE, CHUNK)
        assert result.rules == []
        assert result.repair_used

    def test_targeted_grounding_drops_unrequested_predicates(self):
        payload = {
            "facts": [
                {"predicate": "requested", "status": "UNKNOWN", "source_quote": "", "confidence": 0.0},
                {"predicate": "unrequested", "status": "UNKNOWN", "source_quote": "", "confidence": 0.0},
            ]
        }
        backend = FakeOllamaBackend(responses=[json_response(payload)])
        client = OllamaClient(_backend=backend)
        result = ground_facts(client, "test-model", GROUND_TEMPLATE, "case facts", ["requested"])
        assert [fact.predicate for fact in result.facts] == ["requested"]
        assert any("unrequested" in warning for warning in result.validation_warnings)


class TestGroundFacts:
    def test_grounds_facts_from_valid_response(self):
        payload = {
            "facts": [
                {"predicate": "relationship_test", "status": "TRUE", "source_quote": "Alice is Bob's cousin", "confidence": 0.9},
                {"predicate": "income_test", "status": "UNKNOWN", "source_quote": "", "confidence": 0.0},
            ]
        }
        backend = FakeOllamaBackend(responses=[json_response(payload)])
        client = OllamaClient(_backend=backend)
        result = ground_facts(client, "test-model", GROUND_TEMPLATE, "Alice is Bob's cousin.", ["relationship_test", "income_test"])
        assert result.schema_valid
        assert len(result.facts) == 2
        d = facts_to_dict(result.facts)
        assert d["relationship_test"] == "true"
        assert d["income_test"] == "unknown"

    def test_drops_facts_with_fabricated_quote(self):
        payload = {
            "facts": [
                {"predicate": "p", "status": "TRUE", "source_quote": "fabricated text", "confidence": 0.9},
            ]
        }
        backend = FakeOllamaBackend(responses=[json_response(payload)])
        client = OllamaClient(_backend=backend)
        result = ground_facts(client, "test-model", GROUND_TEMPLATE, "real case text", ["p"])
        assert result.facts == []
