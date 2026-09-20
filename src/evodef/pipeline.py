"""End-to-end pipeline: one function, `run_case`, drives every method
(direct_llm, vanilla_rag, prompted_rag, static_nesy_rag, evodef_full) and
every ablation, since they are all the same pipeline with stages toggled on
or off by `configs/experiments.yaml` (03_IMPLEMENTATION_SPEC.md, 05_EXPERIMENT_PLAN.md).

Modeling decisions not fully pinned down by the spec bundle, made explicitly
here and documented for the paper's limitations section:

  * Predicates are matched by name only, arguments stripped
    (see `utils.normalize_predicate` - PYTHEN has no unification).
  * The proof target is the (normalized) proposition of the rule extracted
    from the top-ranked retrieved+formalized chunk. This is a retrieval-rank
    heuristic, not derived from gold labels.
  * UNDETERMINED is mapped to CONTRADICTION for the binary SARA metric (a
    fixed, predeclared rule, per 01_METHOD_SPEC.md #8); the internal
    abstention rate is reported separately via `PredictionRow.undetermined`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .formalization.facts import FactGroundingResult, facts_to_dict, ground_facts
from .formalization.rules import FormalizationResult, formalize_chunk
from .memory.gap_memory import GapMemory, generalize_signature
from .memory.norm_memory import NormMemory, validate_candidate
from .ollama_client import LLMCallRecord, OllamaClient
from .prompt_templates import PromptTemplate
from .retrieval.bm25 import BM25Index
from .retrieval.dense import DenseIndex
from .retrieval.hybrid import rrf_fuse
from .retrieval.llm_reranker import rerank
from .schemas import (
    Case,
    Counterfactual,
    DefeasibleRule,
    DirectAnswerOutput,
    ExceptionCheck,
    GapRecord,
    ProofCertificate,
    ProofNode,
    PredictionRow,
    QueryOutput,
    StatuteChunk,
)
from .symbolic.counterfactual import generate_counterfactuals
from .symbolic.gap_analyzer import analyze_gaps
from .symbolic.traced_evaluator import TracedRuleEvaluator
from .utils import normalize_predicate, sha256_json

UNDETERMINED_MAPS_TO = "CONTRADICTION"  # fixed predeclared rule, 01_METHOD_SPEC.md #8


@dataclass
class PipelineResources:
    corpus: dict[str, StatuteChunk]
    bm25_index: BM25Index
    client: OllamaClient
    generator_model: str
    prompts: dict[str, PromptTemplate]
    config: dict
    dense_index: DenseIndex | None = None
    norm_memory: NormMemory = field(default_factory=NormMemory)
    gap_memory: GapMemory = field(default_factory=GapMemory)
    formalization_variant: dict = field(default_factory=dict)
    grounding_variant: dict = field(default_factory=dict)
    write_memory: bool = False  # True only while running scripts/03_run_evolution.py
    diagnostics: dict | None = None  # set by scripts/02_tune_prompts_dev.py to collect extraction-quality counters

    def config_hash(self) -> str:
        return sha256_json(
            {
                "retrieval": self.config.get("retrieval"),
                "memory": self.config.get("memory"),
                "formalization_variant": self.formalization_variant,
                "grounding_variant": self.grounding_variant,
            }
        )


def run_case(
    resources: PipelineResources,
    case: Case,
    method_name: str,
    method_cfg_override: dict | None = None,
    gap_trace: list[dict] | None = None,
) -> PredictionRow:
    """`method_cfg_override` lets scripts/02_tune_prompts_dev.py evaluate a
    grid combination that is not one of the named methods in
    configs/experiments.yaml, while still recording `method_name` as a label.
    `gap_trace`: see `_targeted_retrieval`'s docstring.
    """
    method_cfg = method_cfg_override or resources.config["_experiments"]["methods"][method_name]
    t0 = time.monotonic()
    stats = _RunStats()
    prompt_hashes: dict[str, str] = {}
    inference_trace: list[dict] = [{
        "stage": "case_start",
        "case_id": case.case_id,
        "method": method_name,
        "config_hash": resources.config_hash(),
    }]

    if method_cfg["retrieval_query_variant"] == "none":
        prediction, errors = _direct_llm(resources, case, stats, prompt_hashes)
        inference_trace.append({"stage": "direct_answer", "prediction": prediction, "errors": errors})
        return _finalize(resources, case, method_name, t0, stats, prompt_hashes, [], [], [], None, prediction, errors, inference_trace)

    retrieved_initial = _retrieve_initial(resources, case, method_cfg, stats, prompt_hashes, inference_trace)

    if not method_cfg["symbolic_reasoning"]:
        prediction, errors = _answer_from_retrieved(resources, case, retrieved_initial, stats, prompt_hashes)
        inference_trace.append({"stage": "direct_answer_from_retrieval", "prediction": prediction, "errors": errors})
        return _finalize(resources, case, method_name, t0, stats, prompt_hashes, retrieved_initial, [], [], None, prediction, errors, inference_trace)

    return _run_symbolic(resources, case, method_name, method_cfg, retrieved_initial, t0, stats, prompt_hashes, inference_trace, gap_trace=gap_trace)


@dataclass
class _RunStats:
    llm_calls: int = 0
    formalization_calls: int = 0
    memory_hits: int = 0
    errors: list[str] = field(default_factory=list)


def _finalize(
    resources: PipelineResources,
    case: Case,
    method_name: str,
    t0: float,
    stats: _RunStats,
    prompt_hashes: dict[str, str],
    retrieved_initial: list[str],
    retrieved_targeted: list[str],
    rules_used: list[str],
    proof: ProofCertificate | None,
    prediction: str,
    errors: list[str],
    inference_trace: list[dict],
) -> PredictionRow:
    return PredictionRow(
        case_id=case.case_id,
        method=method_name,
        config_hash=resources.config_hash(),
        prompt_hashes=prompt_hashes,
        model=resources.generator_model,
        retrieved_initial=retrieved_initial,
        retrieved_targeted=retrieved_targeted,
        rules_used=rules_used,
        proof=proof.model_dump() if proof is not None else None,
        prediction=prediction,
        gold=case.gold_label,
        latency_s=time.monotonic() - t0,
        llm_calls=stats.llm_calls,
        formalization_calls=stats.formalization_calls,
        memory_hits=stats.memory_hits,
        undetermined=proof.decision == "UNDETERMINED" if proof is not None else False,
        errors=errors,
        inference_trace=inference_trace,
    )


# ---------------------------------------------------------------------------
# M0 Direct-LLM
# ---------------------------------------------------------------------------


def _direct_llm(resources, case: Case, stats: _RunStats, prompt_hashes: dict) -> tuple[str, list[str]]:
    template = resources.prompts["direct_answer"]
    prompt_hashes["direct_answer"] = template.hash
    user = template.render_user(facts_text=case.facts_text, assertion=case.assertion, retrieved_text="(none - no retrieval)")
    parsed, _ = resources.client.chat_json(resources.generator_model, template.system, user, DirectAnswerOutput)
    stats.llm_calls += 1
    if parsed is None:
        return UNDETERMINED_MAPS_TO, ["direct_llm: schema parse failed"]
    return parsed.decision, []


def _answer_from_retrieved(resources, case: Case, retrieved_ids: list[str], stats: _RunStats, prompt_hashes: dict) -> tuple[str, list[str]]:
    template = resources.prompts["direct_answer"]
    prompt_hashes["direct_answer"] = template.hash
    chunks = [resources.corpus[cid] for cid in retrieved_ids if cid in resources.corpus]
    retrieved_text = "\n\n".join(f"[{c.chunk_id}] ({c.section_id}) {c.text}" for c in chunks) or "(none retrieved)"
    user = template.render_user(facts_text=case.facts_text, assertion=case.assertion, retrieved_text=retrieved_text)
    parsed, _ = resources.client.chat_json(resources.generator_model, template.system, user, DirectAnswerOutput)
    stats.llm_calls += 1
    if parsed is None:
        return UNDETERMINED_MAPS_TO, ["answer_from_retrieved: schema parse failed"]
    return parsed.decision, []


# ---------------------------------------------------------------------------
# Phase A: retrieval
# ---------------------------------------------------------------------------


def _build_issue_queries(resources, case: Case, exception_focused: bool, stats: _RunStats, prompt_hashes: dict, inference_trace: list[dict]) -> list[str]:
    template = resources.prompts["issue_query"]
    prompt_hashes["issue_query"] = template.hash
    user = template.render_user(facts_text=case.facts_text, assertion=case.assertion)
    if not exception_focused:
        lines = [
            l for l in user.splitlines()
            if not l.strip().startswith("3. Query 3")
        ]
        user = "\n".join(lines).replace("up to 3 concise", "up to 2 concise")
    parsed, _ = resources.client.chat_json(resources.generator_model, template.system, user, QueryOutput)
    stats.llm_calls += 1
    if parsed is None or not parsed.queries:
        queries = [f"{case.assertion}"]
        inference_trace.append({"stage": "issue_query", "queries": queries, "fallback": True})
        return queries
    inference_trace.append({"stage": "issue_query", "queries": parsed.queries, "fallback": False})
    return parsed.queries


def _trace_ranked(ranked: list[tuple[str, float]], top_k: int) -> list[dict]:
    """Keep enough ranked evidence for a figure without serializing full lists."""
    return [{"chunk_id": chunk_id, "score": round(float(score), 8)} for chunk_id, score in ranked[:top_k]]


def _hybrid_retrieve(resources, queries: list[str], top_k: int, inference_trace: list[dict], phase: str) -> list[tuple[str, float]]:
    retrieval_cfg = resources.config["retrieval"]
    ranked_lists: list[list[tuple[str, float]]] = []
    for query in queries:
        sparse = resources.bm25_index.search(query, top_k=retrieval_cfg["bm25_topk"])
        ranked_lists.append(sparse)
        event = {
            "stage": "retrieval_query",
            "phase": phase,
            "query": query,
            "bm25_top": _trace_ranked(sparse, top_k),
        }
        if resources.dense_index is not None:
            query_vec = resources.client.embed(resources.config["models"]["embedder"], [query])[0]
            dense = resources.dense_index.search(query_vec, top_k=retrieval_cfg["dense_topk"])
            ranked_lists.append(dense)
            event["dense_top"] = _trace_ranked(dense, top_k)
        inference_trace.append(event)
    fused = rrf_fuse(ranked_lists, k_rrf=retrieval_cfg["rrf_k"], top_k=top_k)
    inference_trace.append({"stage": "rrf_fusion", "phase": phase, "fused_top": _trace_ranked(fused, top_k)})
    return fused


def _retrieve_initial(resources, case: Case, method_cfg: dict, stats: _RunStats, prompt_hashes: dict, inference_trace: list[dict]) -> list[str]:
    retrieval_cfg = resources.config["retrieval"]
    if method_cfg["retrieval_query_variant"] == "raw":
        queries = [f"{case.assertion} {case.facts_text}"]
        inference_trace.append({"stage": "issue_query", "queries": queries, "mode": "raw", "fallback": False})
    else:
        queries = _build_issue_queries(resources, case, method_cfg["exception_focused_query"], stats, prompt_hashes, inference_trace)

    fused = _hybrid_retrieve(resources, queries, retrieval_cfg["fused_topk_before_rerank"], inference_trace, "initial")

    if method_cfg["llm_rerank"]:
        candidates = [resources.corpus[cid] for cid, _ in fused if cid in resources.corpus]
        template = resources.prompts["rerank_chunks"]
        prompt_hashes["rerank_chunks"] = template.hash
        kept_ids, parsed, _ = rerank(
            resources.client,
            resources.generator_model,
            template,
            case.facts_text,
            case.assertion,
            candidates,
            keep_top_k=retrieval_cfg["final_topk"],
        )
        stats.llm_calls += 1
        inference_trace.append({
            "stage": "rerank",
            "phase": "initial",
            "labels": [item.model_dump() for item in parsed.items] if parsed is not None else [],
            "selected_chunk_ids": kept_ids,
            "fallback": parsed is None or not parsed.items,
        })
        return kept_ids

    selected = [cid for cid, _ in fused[: retrieval_cfg["final_topk"]]]
    inference_trace.append({"stage": "retrieval_selected", "phase": "initial", "selected_chunk_ids": selected})
    return selected


# ---------------------------------------------------------------------------
# Phase B/C/D/E/F: formalize, ground, prove, diagnose, targeted retrieval
# ---------------------------------------------------------------------------


def _formalize_chunks(resources, chunks: list[StatuteChunk], stats: _RunStats, use_memory: bool, inference_trace: list[dict]) -> list[DefeasibleRule]:
    rules: list[DefeasibleRule] = []
    for chunk in chunks:
        if use_memory:
            entry = resources.norm_memory.get(chunk)
            if entry is not None:
                stats.memory_hits += 1
                rules.extend(entry.rules)
                inference_trace.append({
                    "stage": "formalization",
                    "chunk_id": chunk.chunk_id,
                    "source": "norm_memory",
                    "rule_ids": [rule.rule_id for rule in entry.rules],
                })
                continue
        template = resources.prompts["formalize_rule"]
        result: FormalizationResult = formalize_chunk(
            resources.client,
            resources.generator_model,
            template,
            chunk,
            variant=resources.formalization_variant,
        )
        stats.llm_calls += len(result.calls)
        stats.formalization_calls += len(result.calls)
        rules.extend(result.rules)
        inference_trace.append({
            "stage": "formalization",
            "chunk_id": chunk.chunk_id,
            "source": "llm",
            "schema_valid": result.schema_valid,
            "repair_used": result.repair_used,
            "rule_ids": [rule.rule_id for rule in result.rules],
            "needs_review_rule_ids": [rule.rule_id for rule in result.rules if rule.needs_review],
            "validation_errors": result.validation_errors,
        })
        stats.errors.extend(
            f"{chunk.chunk_id}: {error}" for error in result.validation_errors
        )
        if not result.schema_valid:
            stats.errors.append(f"{chunk.chunk_id}: formalization schema parse failed")
        if resources.diagnostics is not None:
            d = resources.diagnostics
            d["chunks_formalized"] = d.get("chunks_formalized", 0) + 1
            d["schema_valid"] = d.get("schema_valid", 0) + int(result.schema_valid)
            d["rules_accepted"] = d.get("rules_accepted", 0) + result.accepted_count
            d["rules_rejected"] = d.get("rules_rejected", 0) + result.rejected_count
        if resources.write_memory and result.rules:
            known_section_ids = {
                value
                for source in resources.corpus.values()
                for value in (source.chunk_id, source.section_id)
            }
            executable = [rule for rule in result.rules if not rule.needs_review]
            admission = validate_candidate(chunk, executable, known_section_ids)
            if executable and admission.valid:
                resources.norm_memory.commit(resources.norm_memory.make_entry(chunk, executable))
            elif executable:
                stats.errors.extend(
                    f"{chunk.chunk_id}: memory admission rejected: {error}"
                    for error in admission.errors
                )
    return rules


def _required_predicates(rules: list[DefeasibleRule]) -> list[str]:
    names = set()
    for r in rules:
        names.add(normalize_predicate(r.p))
        names.update(normalize_predicate(c) for c in r.conditions)
        names.update(normalize_predicate(e) for e in r.exceptions)
    return sorted(names)


def _ground(resources, case: Case, predicates: list[str], stats: _RunStats, prompt_hashes: dict, inference_trace: list[dict], phase: str) -> dict[str, str]:
    if not predicates:
        return {}
    template = resources.prompts["ground_facts"]
    prompt_hashes["ground_facts"] = template.hash
    result: FactGroundingResult = ground_facts(
        resources.client,
        resources.generator_model,
        template,
        case.facts_text,
        predicates,
        variant=resources.grounding_variant,
    )
    stats.llm_calls += len(result.calls)
    facts = facts_to_dict(result.facts)
    inference_trace.append({
        "stage": "fact_grounding",
        "phase": phase,
        "requested_predicates": predicates,
        "grounded_statuses": facts,
        "schema_valid": result.schema_valid,
        "validation_warnings": result.validation_warnings,
    })
    return facts


def _select_target(rules: list[DefeasibleRule], retrieved_order: list[str]) -> str | None:
    """The proposition of the rule from the highest-ranked retrieved chunk
    (see module docstring for why this heuristic is needed)."""
    rank = {cid: i for i, cid in enumerate(retrieved_order)}
    candidates = [r for r in rules if not r.needs_review]
    if not candidates:
        candidates = rules
    if not candidates:
        return None
    candidates.sort(key=lambda r: (rank.get(r.source_id, len(retrieved_order)), r.rule_id))
    return normalize_predicate(candidates[0].p)


def _decision_from_status(status: str) -> str:
    if status == "true":
        return "ENTAILED"
    if status in ("false", "defeated"):
        return "NOT_ENTAILED"
    return "UNDETERMINED"


def _decision_to_label(decision: str) -> str:
    if decision == "ENTAILED":
        return "ENTAILMENT"
    if decision == "NOT_ENTAILED":
        return "CONTRADICTION"
    return UNDETERMINED_MAPS_TO


def _iter_nodes(node: ProofNode):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _build_certificate(
    root: ProofNode,
    rules: list[DefeasibleRule],
    target: str,
    counterfactuals: list[Counterfactual],
) -> ProofCertificate:
    decision = _decision_from_status(root.status)
    rules_by_id = {r.rule_id: r for r in rules}
    supporting_rules, supporting_sources = [], []
    satisfied, missing, exceptions_checked = [], [], []

    seen_exception_checks: set[tuple[str, str]] = set()
    for node in _iter_nodes(root):
        if node.via_rule_id is None:
            continue
        rule = rules_by_id.get(node.via_rule_id)
        if rule is None:
            continue
        if node.via_rule_id not in supporting_rules:
            supporting_rules.append(node.via_rule_id)
        if rule.source_id not in supporting_sources:
            supporting_sources.append(rule.source_id)
        if node.status == "true":
            satisfied.append(node.predicate)
        elif node.status == "unknown":
            missing.append(node.predicate)
        exception_names = {normalize_predicate(name) for name in rule.exceptions}
        for child in node.children:
            if normalize_predicate(child.predicate) not in exception_names:
                continue
            key = (child.predicate, child.status)
            if key not in seen_exception_checks:
                seen_exception_checks.add(key)
                exceptions_checked.append(ExceptionCheck(predicate=child.predicate, status=child.status))

    return ProofCertificate(
        decision=decision,
        target=target,
        supporting_rules=supporting_rules,
        supporting_sources=supporting_sources,
        satisfied_conditions=sorted(set(satisfied)),
        missing_conditions=sorted(set(missing)),
        exceptions_checked=exceptions_checked,
        proof_valid=True,
        counterfactuals=counterfactuals,
        root=root,
    )


def _targeted_retrieval(
    resources,
    case: Case,
    gaps: list[GapRecord],
    already_retrieved: set[str],
    method_cfg: dict,
    stats: _RunStats,
    prompt_hashes: dict,
    inference_trace: list[dict],
    gap_trace: list[dict] | None = None,
) -> list[str]:
    """`gap_trace`, when given, is appended with one {signature, queries,
    used_memory} dict per gap actually queried - scripts/03_run_evolution.py
    uses this to know which Gap Policy Memory entry to reward/penalize
    (01_METHOD_SPEC.md #4.2), which is not otherwise recoverable from a
    PredictionRow.
    """
    retrieval_cfg = resources.config["retrieval"]
    applicable_gaps = gaps
    if not method_cfg["exception_focused_query"]:
        applicable_gaps = [g for g in gaps if g.gap_type != "UNRESOLVED_EXCEPTION"]
    if not applicable_gaps:
        inference_trace.append({"stage": "targeted_retrieval", "skipped": True, "reason": "no applicable proof gaps"})
        return []

    top_gaps = applicable_gaps[:2]
    all_queries: list[str] = []
    for gap in top_gaps:
        signature = generalize_signature(gap)
        memory_entry = resources.gap_memory.get(signature) if method_cfg["use_gap_memory"] else None
        used_memory = memory_entry is not None and bool(memory_entry.query_templates)
        if used_memory:
            # Do not use ``str.format`` on model-generated templates: braces
            # unrelated to our two placeholders would otherwise abort a run.
            queries = [
                t.replace("{predicate}", gap.target_predicate).replace("{source_section}", gap.source_id or "")
                for t in memory_entry.query_templates
            ]
        else:
            template = resources.prompts["gap_query"]
            prompt_hashes["gap_query"] = template.hash
            user = template.render_user(
                gap_type=gap.gap_type,
                target_predicate=gap.target_predicate,
                rule_id=gap.current_rule_id or "",
                source_id=gap.source_id or "",
                query_terms=", ".join(gap.query_terms),
                known_references=gap.source_id or "",
            )
            parsed, _ = resources.client.chat_json(resources.generator_model, template.system, user, QueryOutput)
            stats.llm_calls += 1
            queries = parsed.queries if parsed is not None else gap.query_terms
        queries = queries[: retrieval_cfg["targeted_queries_max"]]
        all_queries.extend(queries)
        inference_trace.append({
            "stage": "gap_query",
            "gap": gap.model_dump(),
            "queries": queries,
            "used_gap_memory": used_memory,
        })
        if gap_trace is not None:
            gap_trace.append({"signature": signature, "queries": queries, "used_memory": used_memory})

    fused = _hybrid_retrieve(
        resources,
        all_queries,
        retrieval_cfg["targeted_new_chunks_max"] + len(already_retrieved),
        inference_trace,
        "targeted",
    )
    new_ids = [cid for cid, _ in fused if cid not in already_retrieved]
    selected = new_ids[: retrieval_cfg["targeted_new_chunks_max"]]
    inference_trace.append({"stage": "retrieval_selected", "phase": "targeted", "selected_chunk_ids": selected})
    return selected


def _run_symbolic(
    resources,
    case: Case,
    method_name: str,
    method_cfg: dict,
    retrieved_initial: list[str],
    t0: float,
    stats: _RunStats,
    prompt_hashes: dict,
    inference_trace: list[dict],
    gap_trace: list[dict] | None = None,
) -> PredictionRow:
    use_memory = method_cfg["use_norm_memory"]
    evidence_chunks = [resources.corpus[cid] for cid in retrieved_initial if cid in resources.corpus]
    rules = _formalize_chunks(resources, evidence_chunks, stats, use_memory, inference_trace)
    executable_rules = [rule for rule in rules if not rule.needs_review]

    predicates = _required_predicates(executable_rules)
    facts = _ground(resources, case, predicates, stats, prompt_hashes, inference_trace, "initial")

    target = _select_target(executable_rules, retrieved_initial)
    retrieved_targeted: list[str] = []

    if target is None:
        proof = ProofCertificate(decision="UNDETERMINED", target="", proof_valid=False)
        return _finalize(
            resources, case, method_name, t0, stats, prompt_hashes,
            retrieved_initial, retrieved_targeted, [], proof, _decision_to_label("UNDETERMINED"),
            ["no executable rules formalized from retrieved evidence", *stats.errors], inference_trace,
        )

    root = TracedRuleEvaluator(executable_rules).evaluate(facts, target)
    known_source_ids = {c.chunk_id for c in evidence_chunks} | {c.section_id for c in evidence_chunks}
    rules_by_id = {r.rule_id: r for r in rules}
    gaps = analyze_gaps(root, rules_by_id, known_source_ids)
    inference_trace.append({
        "stage": "symbolic_proof",
        "phase": "initial",
        "target": target,
        "decision": _decision_from_status(root.status),
        "root_status": root.status,
        "gaps": [gap.model_dump() for gap in gaps],
    })

    if method_cfg["targeted_retrieval"] and root.status == "unknown" and gaps:
        already = set(retrieved_initial)
        retrieved_targeted = _targeted_retrieval(resources, case, gaps, already, method_cfg, stats, prompt_hashes, inference_trace, gap_trace=gap_trace)
        new_chunks = [resources.corpus[cid] for cid in retrieved_targeted if cid in resources.corpus]
        if new_chunks:
            new_rules = _formalize_chunks(resources, new_chunks, stats, use_memory, inference_trace)
            rules = rules + new_rules
            executable_rules = [rule for rule in rules if not rule.needs_review]
            rules_by_id = {r.rule_id: r for r in rules}
            new_predicates = [p for p in _required_predicates(executable_rules) if p not in facts]
            facts.update(_ground(resources, case, new_predicates, stats, prompt_hashes, inference_trace, "targeted"))
            root = TracedRuleEvaluator(executable_rules).evaluate(facts, target)
            known_source_ids |= {c.chunk_id for c in new_chunks} | {c.section_id for c in new_chunks}
            gaps = analyze_gaps(root, rules_by_id, known_source_ids)
            inference_trace.append({
                "stage": "symbolic_proof",
                "phase": "after_targeted_retrieval",
                "target": target,
                "decision": _decision_from_status(root.status),
                "root_status": root.status,
                "remaining_gaps": [gap.model_dump() for gap in gaps],
            })
    elif method_cfg["targeted_retrieval"]:
        inference_trace.append({
            "stage": "targeted_retrieval",
            "skipped": True,
            "reason": "initial proof resolved" if root.status != "unknown" else "no deterministic proof gap",
        })

    counterfactuals = [cf for cf in generate_counterfactuals(executable_rules, facts, target, root) if cf.verified_by_reexecution]
    proof = _build_certificate(root, executable_rules, target, counterfactuals)
    prediction = _decision_to_label(proof.decision)
    inference_trace.append({
        "stage": "final_decision",
        "proof_decision": proof.decision,
        "prediction": prediction,
        "counterfactual_count": len(counterfactuals),
    })

    return _finalize(
        resources, case, method_name, t0, stats, prompt_hashes,
        retrieved_initial, retrieved_targeted, [r.rule_id for r in executable_rules], proof, prediction, stats.errors, inference_trace,
    )
