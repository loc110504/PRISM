#!/usr/bin/env python3
"""Feasibility scan for the controlled proof-gap diagnostic (paper Table III).

NOT the final diagnostic-set generator. SARA's gold_chunk_ids are all
single-chunk (verified against data/processed/cases.jsonl), so naive
leave-one-out over a case's own gold evidence can never produce a real
MISSING_SUPPORT/UNRESOLVED_EXCEPTION gap (removing the only chunk removes
the whole rule, not one of its conditions - analyze_gaps() requires a
still-standing parent rule with exactly one unresolved child).

Instead this scans hand-picked statute "families" (a root section plus its
sub-clauses, spanning genuine ALL-conditions / exceptions / cross-references
in the same passage, e.g. the surviving-spouse definition sec_2_a_1/_2 and
its (A)/(B) sub-clauses). Each family is formalized as ONE passage (a single
formalize_chunk call, reusing the existing agent/prompt - no new prompting),
so every declared condition/exception/rule proposition name is internally
consistent by construction. Each declared rule is then linked back to
whichever child chunk's text contains its `source_quote` (exact-substring,
deterministic - not an LLM judgment). For every linkable condition/exception,
the predicate's own defining rule is removed and the REAL, unmodified
`evodef.symbolic.traced_evaluator.TracedRuleEvaluator` +
`evodef.symbolic.gap_analyzer.analyze_gaps` (no LLM, pure code, same code
path the live pipeline uses) decide whether a clean typed gap results.

This only measures structural yield (does the family formalize into
linkable, removable rules at all) - it does NOT ground real case facts or
run the repair-check; that belongs to the full generator once yield looks
workable. Prints per-family and total counts; writes
outputs/results/proof_gap_feasibility_scan.json + a detail JSONL for
inspection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.formalization.rules import formalize_chunk  # noqa: E402
from evodef.ollama_client import OllamaClient  # noqa: E402
from evodef.prompt_templates import load_prompt_dir  # noqa: E402
from evodef.schemas import DefeasibleRule, StatuteChunk  # noqa: E402
from evodef.symbolic.gap_analyzer import analyze_gaps  # noqa: E402
from evodef.symbolic.traced_evaluator import TracedRuleEvaluator  # noqa: E402
from evodef.utils import load_config, normalize_predicate, read_jsonl, resolve_path, write_json, write_jsonl  # noqa: E402

# Hand-picked (not LLM-selected): each family's chunks share a common
# statutory ancestor and, read together, contain real ALL-conditions and/or
# "unless"-exceptions and/or "section N" cross-references - the three things
# scripts/../symbolic/gap_analyzer.py can type. Pure numeric tax-bracket
# families (sec_1_a_i.. etc.) are deliberately excluded: they are
# alternative dollar thresholds, not legal conditions/exceptions.
FAMILIES: dict[str, list[str]] = {
    "surviving_spouse": [
        "sec_2_a_1", "sec_2_a_1_A", "sec_2_a_1_B",
        "sec_2_a_2", "sec_2_a_2_A", "sec_2_a_2_B",
    ],
    "head_of_household": [
        "sec_2_b_1", "sec_2_b_1_A", "sec_2_b_1_A_i", "sec_2_b_1_A_i_I", "sec_2_b_1_A_i_II",
        "sec_2_b_1_A_ii", "sec_2_b_1_B", "sec_2_b_2_A", "sec_2_b_2_B", "sec_2_b_2_C",
        "sec_2_b_3_A", "sec_2_b_3_B",
    ],
    "qualifying_child": [
        "sec_152_c_1", "sec_152_c_1_A", "sec_152_c_1_B", "sec_152_c_1_C", "sec_152_c_1_E",
        "sec_152_c_2", "sec_152_c_2_A", "sec_152_c_2_B", "sec_152_c_3",
    ],
    "qualifying_relative": [
        "sec_152_d_1", "sec_152_d_1_A", "sec_152_d_1_B", "sec_152_d_1_D", "sec_152_d_2",
    ],
    "dependents_general": [
        "sec_152_a", "sec_152_a_1", "sec_152_a_2", "sec_152_b_1", "sec_152_b_2",
    ],
    "standard_deduction_ineligible": [
        "sec_63_c_6", "sec_63_c_6_A", "sec_63_c_6_B", "sec_63_c_6_D",
    ],
    "employer_definition": [
        "sec_3306_a_1", "sec_3306_a_1_A", "sec_3306_a_1_B",
        "sec_3306_a_2", "sec_3306_a_2_A", "sec_3306_a_2_B",
    ],
    "personal_exemption": [
        "sec_151_a", "sec_151_b", "sec_151_c",
    ],
}

QUANT_GAP_TYPES = ("MISSING_SUPPORT", "UNRESOLVED_EXCEPTION", "MISSING_REFERENCE")


def _build_family_chunk(family_name: str, chunk_ids: list[str], corpus: dict[str, StatuteChunk]) -> tuple[StatuteChunk, list[StatuteChunk]]:
    children = [corpus[cid] for cid in chunk_ids]
    text = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in children)
    family_chunk = StatuteChunk(chunk_id=f"family_{family_name}", section_id=family_name, text=text)
    return family_chunk, children


def _link_child(rule: DefeasibleRule, children: list[StatuteChunk]) -> str | None:
    """Deterministic: which child's own text contains this rule's exact
    source_quote. None if the quote spans/paraphrases across children (a
    real signal that this rule is not cleanly attributable to one chunk).
    """
    if not rule.source_quote:
        return None
    for child in children:
        if rule.source_quote in child.text:
            return child.chunk_id
    return None


def _scan_family(client: OllamaClient, model: str, template, family_name: str, chunk_ids: list[str], corpus: dict[str, StatuteChunk]) -> tuple[dict[str, int], list[dict]]:
    family_chunk, children = _build_family_chunk(family_name, chunk_ids, corpus)
    result = formalize_chunk(client, model, template, family_chunk)
    counts = {t: 0 for t in QUANT_GAP_TYPES}
    detail: list[dict] = []

    if not result.rules:
        print(f"  [{family_name}] no valid rules extracted (schema_valid={result.schema_valid}, "
              f"rejected={result.rejected_count}) - skipping")
        return counts, detail

    rules = result.rules
    known_source_ids = {family_chunk.chunk_id, family_chunk.section_id} | {c.chunk_id for c in children} | {c.section_id for c in children}
    predicate_to_child = {normalize_predicate(r.p): _link_child(r, children) for r in rules}

    for target_rule in rules:
        target = normalize_predicate(target_rule.p)
        sub_predicates = [normalize_predicate(x) for x in (target_rule.conditions + target_rule.exceptions)]

        # MISSING_REFERENCE: a cross-reference the family passage does not resolve.
        for ref in target_rule.references:
            if ref not in known_source_ids:
                counts["MISSING_REFERENCE"] += 1
                detail.append({
                    "family": family_name, "target": target, "gap_type": "MISSING_REFERENCE",
                    "withheld_predicate": None, "withheld_chunk": None, "reference": ref,
                    "rule_id": target_rule.rule_id, "source_quote": target_rule.source_quote,
                })

        # MISSING_SUPPORT / UNRESOLVED_EXCEPTION: remove one linkable
        # sub-predicate's own rule and its fact, leave every other
        # sub-predicate of this rule asserted TRUE, and ask the REAL
        # evaluator + gap analyzer what happens.
        for cond in sub_predicates:
            child_id = predicate_to_child.get(cond)
            if child_id is None:
                continue  # not attributable to one child chunk - skip, don't guess
            reduced_rules = [r for r in rules if normalize_predicate(r.p) != cond]
            facts = {p: "true" for p in sub_predicates if p != cond}
            reduced_by_id = {r.rule_id: r for r in reduced_rules}
            root = TracedRuleEvaluator(reduced_rules).evaluate(facts, target)
            gaps = analyze_gaps(root, reduced_by_id, known_source_ids)
            match = next((g for g in gaps if normalize_predicate(g.target_predicate) == cond), None)
            if match is not None and match.gap_type in counts:
                counts[match.gap_type] += 1
                detail.append({
                    "family": family_name, "target": target, "gap_type": match.gap_type,
                    "withheld_predicate": cond, "withheld_chunk": child_id, "reference": None,
                    "rule_id": target_rule.rule_id, "source_quote": target_rule.source_quote,
                })

    return counts, detail


def main() -> None:
    config = load_config()
    corpus_rows = read_jsonl(resolve_path(config, "processed_dir") / "statutes.jsonl")
    corpus = {row["chunk_id"]: StatuteChunk.model_validate(row) for row in corpus_rows}
    prompts = load_prompt_dir(REPO_ROOT / config["paths"]["prompts_dir"])
    template = prompts["formalize_rule"]

    import os

    client = OllamaClient(
        base_url=os.environ.get("OLLAMA_HOST", config["ollama"]["base_url"]),
        max_schema_retries=config["ollama"]["max_schema_retries"],
        request_timeout_s=config["ollama"]["request_timeout_s"],
        max_network_retries=config["ollama"].get("max_network_retries", 2),
        network_retry_backoff_s=config["ollama"].get("network_retry_backoff_s", 5.0),
        provider=config.get("llm", {}).get("provider", "ollama"),
        embed_provider=config.get("llm", {}).get("embed_provider"),
    )

    totals = {t: 0 for t in QUANT_GAP_TYPES}
    all_detail: list[dict] = []
    for family_name, chunk_ids in FAMILIES.items():
        counts, detail = _scan_family(client, config["models"]["generator"], template, family_name, chunk_ids, corpus)
        print(f"  [{family_name}] {counts}")
        for t in QUANT_GAP_TYPES:
            totals[t] += counts[t]
        all_detail.extend(detail)

    print("\n=== Feasibility scan totals (structural yield, no case grounding yet) ===")
    print(json.dumps(totals, indent=2))
    print(f"total linkable candidates: {sum(totals.values())}")

    results_dir = resolve_path(config, "results_dir")
    write_json(results_dir / "proof_gap_feasibility_scan.json", {
        "families_scanned": list(FAMILIES.keys()),
        "totals": totals,
        "model": config["models"]["generator"],
    })
    write_jsonl(results_dir / "proof_gap_feasibility_scan_detail.jsonl", all_detail)
    print(f"\nWrote {results_dir / 'proof_gap_feasibility_scan.json'}")
    print(f"Wrote {results_dir / 'proof_gap_feasibility_scan_detail.jsonl'}")


if __name__ == "__main__":
    main()
