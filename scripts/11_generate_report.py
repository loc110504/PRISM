#!/usr/bin/env python3
"""Generate the paper-facing results report from produced artifacts only.

This script never runs inference and never fabricates a number. Run it after
scripts/08_make_tables.py; it writes ``outputs/PAPER_RESULTS_REPORT.md``, the
only approved numeric source for paper writing (08_AGENT_EXECUTION_CHECKLIST,
Milestone I).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_corpus_and_cases  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.utils import load_config, read_json, read_jsonl, resolve_path, write_json  # noqa: E402


def _markdown_table(path: Path) -> str:
    if not path.exists():
        return f"_Not available: `{path.relative_to(path.parents[2])}`._"
    frame = pd.read_csv(path)
    if frame.empty:
        return "_Available but empty._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.fillna("NA").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def _claim(main_results: pd.DataFrame | None, ablations: pd.DataFrame | None) -> list[str]:
    claims: list[str] = []
    if main_results is not None and {"evodef_full", "static_nesy_rag"}.issubset(set(main_results["method"])):
        by_method = main_results.set_index("method")
        full, static = by_method.loc["evodef_full"], by_method.loc["static_nesy_rag"]
        if full["accuracy"] > static["accuracy"]:
            claims.append("EvoDef-RAG exceeds Static-NeSy-RAG accuracy on the held-out test split.")
        if full.get("recall_at_5", float("nan")) > static.get("recall_at_5", float("nan")):
            claims.append("EvoDef-RAG improves initial Recall@5 over Static-NeSy-RAG on the held-out test split.")
    if ablations is not None and {"evodef_full", "no_proof_gap"}.issubset(set(ablations["ablation"])):
        by_ablation = ablations.set_index("ablation")
        if by_ablation.loc["evodef_full", "accuracy"] > by_ablation.loc["no_proof_gap", "accuracy"]:
            claims.append("Removing proof-gap retrieval reduces held-out accuracy in the preregistered ablation.")
    return claims[:3]


def _representative_cases(config: dict) -> dict[str, str | None]:
    path = resolve_path(config, "predictions_dir") / "evodef_full_test.jsonl"
    rows = read_jsonl(path)
    good = next((row.get("case_id") for row in rows if row.get("prediction") == row.get("gold")), None)
    bad = next((row.get("case_id") for row in rows if row.get("prediction") != row.get("gold")), None)
    return {"successful_case_id": good, "failure_case_id": bad, "prediction_file": str(path)}


def main() -> None:
    config = load_config()
    _, cases = load_corpus_and_cases(config)
    processed = resolve_path(config, "processed_dir")
    manifest_path = processed / "data_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    checkpoints = resolve_path(config, "checkpoints_dir") / "frozen_checkpoint_manifest.json"
    checkpoint_manifest = read_json(checkpoints) if checkpoints.exists() else {}
    results_dir = resolve_path(config, "results_dir")
    tables_dir = resolve_path(config, "tables_dir")
    main_path = results_dir / "main_results.csv"
    ablation_path = results_dir / "ablation_results.csv"
    main_results = pd.read_csv(main_path) if main_path.exists() else None
    ablations = pd.read_csv(ablation_path) if ablation_path.exists() else None
    claims = _claim(main_results, ablations)
    claim_lines = [f"- {claim}" for claim in claims] or [
        "- No comparative claim is supported until all required result artifacts are available."
    ]
    representative = _representative_cases(config)

    sections = [
        "# EvoDef-RAG Experimental Results Report",
        "",
        "This report is generated from experiment artifacts. Do not replace missing values with estimates or use any other file as a numeric source for the paper.",
        "",
        "## Run provenance",
        "",
        f"- Dataset cases loaded: {len(cases)}",
        f"- Dataset manifest: `{manifest_path}`",
        f"- Dataset archive SHA-256: `{manifest.get('source_sha256', 'not available')}`",
        f"- Generator: `{config['models']['generator']}`; embedder: `{config['models']['embedder']}`; seed: `{config['seed']}`",
        f"- Frozen checkpoint manifest: `{checkpoints}`",
        "",
        "```json",
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True),
        "```",
        "",
        "## Main results (Table 1)",
        "",
        _markdown_table(tables_dir / "table1_main_results.csv"),
        "",
        "## Ablations (Table 2)",
        "",
        _markdown_table(tables_dir / "table2_ablations.csv"),
        "",
        "## Auditability (Table 3)",
        "",
        _markdown_table(tables_dir / "table3_xai.csv"),
        "",
        "## Evolution, efficiency, error, and subgroup diagnostics",
        "",
        "### Evolution curve",
        _markdown_table(results_dir / "evolution_curve.csv"),
        "",
        "### Efficiency",
        _markdown_table(results_dir / "efficiency_results.csv"),
        "",
        "### Error breakdown",
        _markdown_table(results_dir / "error_breakdown.csv"),
        "",
        "### Subgroups",
        _markdown_table(results_dir / "subgroup_results.csv"),
        "",
        "### Statistical tests",
        "```json",
        json.dumps(read_json(results_dir / "statistical_tests.json") if (results_dir / "statistical_tests.json").exists() else {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Supported claims",
        "",
        *claim_lines,
        "",
        "## Limitations",
        "",
        "- SARA is a small, tax-law-focused benchmark; results may not generalize to other legal domains.",
        "- Source-grounded rule validation is not lawyer validation, and model extraction can still fail.",
        "- The symbolic layer uses argument-insensitive predicate matching, a deliberate PoC limitation documented in the pipeline.",
        "",
        "## Representative cases",
        "",
        f"- Successful case ID: `{representative['successful_case_id'] or 'not available'}`",
        f"- Failure case ID: `{representative['failure_case_id'] or 'not available'}`",
        f"- Raw predictions: `{representative['prediction_file']}`",
        f"- Case-study artifact: `{resolve_path(config, 'figures_dir') / 'figure3_case_study.json'}`",
    ]
    out_path = resolve_path(config, "outputs_dir") / "PAPER_RESULTS_REPORT.md"
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    write_json(resolve_path(config, "outputs_dir") / "paper_results_report_manifest.json", {
        "report": str(out_path), "representative_cases": representative, "supported_claims": claims,
    })
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
