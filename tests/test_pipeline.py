"""Integration tests for pipeline.py driving every method through a scripted
FakeOllamaBackend. No real Ollama server is used (see conftest.py)."""
from __future__ import annotations

import json

from conftest import FakeOllamaBackend
from evodef.memory.gap_memory import GapMemory
from evodef.memory.norm_memory import NormMemory
from evodef.ollama_client import OllamaClient
from evodef.pipeline import PipelineResources, run_case
from evodef.prompt_templates import load_prompt_dir
from evodef.retrieval.bm25 import BM25Index
from evodef.schemas import Case, StatuteChunk
from evodef.utils import load_config

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

MAIN_CHUNK = StatuteChunk(
    chunk_id="sec_152_d_1",
    section_id="152(d)(1)",
    text="An individual is a qualifying relative if the relationship test and the gross income test are met.",
    source_file="section152",
)
EXCEPTION_CHUNK = StatuteChunk(
    chunk_id="sec_152_d_1_D",
    section_id="152(d)(1)(D)",
    text="An individual is not a qualifying relative if that individual is a qualifying child of another taxpayer.",
    source_file="section152",
)

CORPUS = {c.chunk_id: c for c in [MAIN_CHUNK, EXCEPTION_CHUNK]}

CASE = Case(
    case_id="s152_d_1_pos",
    split="test",
    facts_text="Alice bears the relationship to Bob and has no gross income for the year. Alice is not the qualifying child of any other taxpayer.",
    assertion="Alice is a qualifying relative of Bob.",
    gold_label="ENTAILMENT",
    gold_chunk_ids=["sec_152_d_1"],
)


def _content(payload: dict) -> dict:
    return {"message": {"content": json.dumps(payload)}, "prompt_eval_count": 1, "eval_count": 1}


def _dispatch_handler(rule_for_main=True, rule_for_exception=True, ground_relationship="TRUE", ground_income="TRUE", ground_exception="UNKNOWN", ground_exception2="FALSE"):
    """Routes each FakeOllamaBackend.chat() call to a canned response based on
    which prompt's system text (stable per template) is being invoked, and
    for ground_facts, on whether the exception predicate was already asked
    about once (simulates targeted-retrieval producing a second grounding
    call that resolves it).
    """
    state = {"ground_calls": 0}

    def handler(model, messages):
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "legal retrieval query planner" in system:
            return _content({"queries": ["qualifying relative test", "qualifying relative exception"]})

        if "label retrieved statutory passages" in system:
            items = []
            if "sec_152_d_1_D" in user:
                items.append({"chunk_id": "sec_152_d_1_D", "label": "EXCEPTION"})
            if "sec_152_d_1]" in user or "(152(d)(1))" in user:
                items.append({"chunk_id": "sec_152_d_1", "label": "DIRECT"})
            return _content({"items": items})

        if "convert one statutory passage" in system:
            if "SOURCE ID: sec_152_d_1_D" in user or "not a qualifying relative if" in user:
                if not rule_for_exception:
                    return _content({"rules": []})
                return _content(
                    {
                        "rules": [
                            {
                                "rule_id": "R_exc",
                                "p": "qualifying_child_of_other_taxpayer",
                                "op": "ALL",
                                "conditions": ["is_qualifying_child_of_other"],
                                "exceptions": [],
                                "references": [],
                                "source_id": "sec_152_d_1_D",
                                "source_quote": EXCEPTION_CHUNK.text,
                                "confidence": 0.9,
                                "needs_review": False,
                            }
                        ]
                    }
                )
            if not rule_for_main:
                return _content({"rules": []})
            return _content(
                {
                    "rules": [
                        {
                            "rule_id": "R_main",
                            "p": "qualifying_relative",
                            "op": "ALL",
                            "conditions": ["relationship_test", "gross_income_test"],
                            "exceptions": ["qualifying_child_of_other_taxpayer"],
                            "references": [],
                            "source_id": "sec_152_d_1",
                            "source_quote": MAIN_CHUNK.text,
                            "confidence": 0.9,
                            "needs_review": False,
                        }
                    ]
                }
            )

        if "ground a case narrative" in system:
            state["ground_calls"] += 1
            facts = []
            if "relationship_test" in user:
                facts.append({"predicate": "relationship_test", "status": ground_relationship, "source_quote": "bears the relationship", "confidence": 0.9})
            if "gross_income_test" in user:
                facts.append({"predicate": "gross_income_test", "status": ground_income, "source_quote": "no gross income", "confidence": 0.9})
            if "qualifying_child_of_other_taxpayer" in user and state["ground_calls"] == 1:
                facts.append({"predicate": "qualifying_child_of_other_taxpayer", "status": ground_exception, "source_quote": "", "confidence": 0.0})
            if "is_qualifying_child_of_other" in user:
                status = ground_exception2 if state["ground_calls"] > 1 else ground_exception
                quote = "not the qualifying child of any other taxpayer" if status != "UNKNOWN" else ""
                facts.append({"predicate": "is_qualifying_child_of_other", "status": status, "source_quote": quote, "confidence": 0.8})
            return _content({"facts": facts})

        if "rewrite a deterministic symbolic proof gap" in system:
            return _content({"queries": ["qualifying child of another taxpayer exception"]})

        if "decide whether a legal assertion is entailed" in system:
            return _content({"decision": "ENTAILMENT", "rationale": "facts support the assertion"})

        raise AssertionError(f"unrouted system prompt: {system[:60]}")

    return handler


def _make_resources(handler, retrieval_overrides=None, **memory_kwargs):
    config = load_config(root=REPO_ROOT)
    if retrieval_overrides:
        config["retrieval"] = {**config["retrieval"], **retrieval_overrides}
    prompts = load_prompt_dir(REPO_ROOT / "prompts")
    backend = FakeOllamaBackend(handler=handler)
    client = OllamaClient(_backend=backend)
    bm25 = BM25Index.build(list(CORPUS.values()))
    return PipelineResources(
        corpus=CORPUS,
        bm25_index=bm25,
        client=client,
        generator_model="test-model",
        prompts=prompts,
        config=config,
        norm_memory=memory_kwargs.get("norm_memory", NormMemory()),
        gap_memory=memory_kwargs.get("gap_memory", GapMemory()),
        formalization_variant={"demonstrations": 0, "concise_operator_instructions": True, "forbid_cross_reference_resolution": False},
        grounding_variant={"mode": "predicate_targeted"},
    )


class TestDirectLLM:
    def test_direct_llm_answers_without_retrieval(self):
        resources = _make_resources(_dispatch_handler())
        row = run_case(resources, CASE, "direct_llm")
        assert row.prediction == "ENTAILMENT"
        assert row.retrieved_initial == []
        assert row.llm_calls == 1
        assert [event["stage"] for event in row.inference_trace] == ["case_start", "direct_answer"]


class TestVanillaAndPromptedRag:
    def test_vanilla_rag_retrieves_and_answers_directly(self):
        resources = _make_resources(_dispatch_handler())
        row = run_case(resources, CASE, "vanilla_rag")
        assert row.prediction == "ENTAILMENT"
        assert "sec_152_d_1" in row.retrieved_initial
        assert row.proof is None

    def test_prompted_rag_uses_query_expansion_and_rerank(self):
        resources = _make_resources(_dispatch_handler())
        row = run_case(resources, CASE, "prompted_rag")
        assert row.prediction == "ENTAILMENT"
        assert set(row.retrieved_initial) <= {"sec_152_d_1", "sec_152_d_1_D"}


class TestStaticNesyRag:
    def test_undetermined_without_targeted_retrieval(self):
        # exception left UNKNOWN by grounding, and static_nesy_rag never runs
        # a second retrieval round, so the proof cannot resolve past "unknown".
        resources = _make_resources(_dispatch_handler(ground_exception="UNKNOWN"))
        row = run_case(resources, CASE, "static_nesy_rag")
        assert row.undetermined is True
        assert row.proof["decision"] == "UNDETERMINED"
        assert row.retrieved_targeted == []

    def test_entailed_when_exception_resolved_by_initial_grounding(self):
        resources = _make_resources(_dispatch_handler(ground_exception="FALSE"))
        row = run_case(resources, CASE, "static_nesy_rag")
        assert row.prediction == "ENTAILMENT"
        assert row.proof["decision"] == "ENTAILED"
        assert row.proof["exceptions_checked"] == [
            {"predicate": "qualifying_child_of_other_taxpayer", "status": "false"}
        ]


class TestEvoDefFull:
    def test_targeted_retrieval_resolves_exception_to_entailed(self):
        # final_topk=1 forces phase-A to surface only the main chunk, so the
        # exception provision is genuinely missing until targeted retrieval.
        resources = _make_resources(
            _dispatch_handler(ground_exception="UNKNOWN", ground_exception2="FALSE"),
            retrieval_overrides={"final_topk": 1},
        )
        row = run_case(resources, CASE, "evodef_full")
        assert row.prediction == "ENTAILMENT"
        assert row.proof["decision"] == "ENTAILED"
        assert "sec_152_d_1_D" in row.retrieved_targeted
        assert any(cf["verified_by_reexecution"] for cf in row.proof["counterfactuals"])
        stages = [event["stage"] for event in row.inference_trace]
        for required in ["issue_query", "retrieval_query", "rrf_fusion", "rerank", "formalization", "fact_grounding", "symbolic_proof", "gap_query", "final_decision"]:
            assert required in stages
        targeted_selection = [event for event in row.inference_trace if event["stage"] == "retrieval_selected" and event.get("phase") == "targeted"]
        assert targeted_selection and "sec_152_d_1_D" in targeted_selection[0]["selected_chunk_ids"]

    def test_no_proof_gap_ablation_never_retrieves_targeted(self):
        resources = _make_resources(_dispatch_handler(ground_exception="UNKNOWN"))
        row = run_case(resources, CASE, "no_proof_gap")
        assert row.retrieved_targeted == []
        assert row.undetermined is True

    def test_norm_memory_hit_skips_formalization_call(self):
        norm_memory = NormMemory()
        from evodef.schemas import DefeasibleRule

        rule = DefeasibleRule(
            rule_id="R_cached", p="qualifying_relative", op="ALL",
            conditions=["relationship_test", "gross_income_test"], exceptions=["qualifying_child_of_other_taxpayer"],
            references=[], source_id="sec_152_d_1", source_quote=MAIN_CHUNK.text, confidence=0.95, needs_review=False,
        )
        norm_memory.commit(norm_memory.make_entry(MAIN_CHUNK, [rule]))
        resources = _make_resources(_dispatch_handler(ground_exception="FALSE"), norm_memory=norm_memory)
        row = run_case(resources, CASE, "static_nesy_rag")
        # static_nesy_rag has use_norm_memory=False per experiments.yaml, so
        # memory is NOT consulted even though it's populated - confirms the
        # "no persistent evolution memory ... for fairness" spec requirement.
        assert row.memory_hits == 0

        resources2 = _make_resources(_dispatch_handler(ground_exception="FALSE"), norm_memory=norm_memory)
        row2 = run_case(resources2, CASE, "evodef_full")
        assert row2.memory_hits >= 1

    def test_gap_trace_records_signature_and_queries(self):
        resources = _make_resources(
            _dispatch_handler(ground_exception="UNKNOWN", ground_exception2="FALSE"),
            retrieval_overrides={"final_topk": 1},
        )
        gap_trace: list[dict] = []
        run_case(resources, CASE, "evodef_full", gap_trace=gap_trace)
        assert len(gap_trace) >= 1
        assert gap_trace[0]["signature"].startswith("UNRESOLVED_EXCEPTION|")
        assert gap_trace[0]["queries"]
        assert gap_trace[0]["used_memory"] is False

    def test_no_memory_ablation_ignores_populated_memory(self):
        norm_memory = NormMemory()
        from evodef.schemas import DefeasibleRule

        rule = DefeasibleRule(
            rule_id="R_cached", p="qualifying_relative", op="ALL",
            conditions=["relationship_test"], exceptions=[], references=[],
            source_id="sec_152_d_1", source_quote=MAIN_CHUNK.text, confidence=0.95, needs_review=False,
        )
        norm_memory.commit(norm_memory.make_entry(MAIN_CHUNK, [rule]))
        resources = _make_resources(_dispatch_handler(ground_exception="FALSE"), norm_memory=norm_memory)
        row = run_case(resources, CASE, "no_memory")
        assert row.memory_hits == 0
