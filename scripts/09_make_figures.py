#!/usr/bin/env python3
"""Milestone H: Figure 1 (architecture, as a Mermaid source file - "acceptable"
per 08_AGENT_EXECUTION_CHECKLIST.md), Figure 2 (evolution curve, from
outputs/results/evolution_curve.csv), Figure 3 (one case study, from a real
evodef_full test prediction)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_corpus_and_cases  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.utils import load_config, read_jsonl, resolve_path  # noqa: E402

ARCHITECTURE_MERMAID = """\
flowchart LR
    Q[Fact pattern + assertion] --> R1[Hybrid retrieval\\nBM25 + dense + RRF + LLM rerank]
    R1 --> F[Source-bounded formalization\\nPYTHEN-compatible rules]
    F --> G[Fact grounding]
    G --> P[Symbolic proof\\nTracedRuleEvaluator]
    P --> D{Proof status}
    D -->|true/false/defeated| C[Proof certificate\\n+ counterfactuals]
    D -->|unknown| GA[Typed proof-gap analysis]
    GA --> R2[Targeted retrieval\\ngap_query.txt / Gap Policy Memory]
    R2 --> F
    F -.validated.-> MN[(Norm Memory)]
    GA -.reward/penalize.-> MG[(Gap Policy Memory)]
    MN -.reuse.-> F
    MG -.reuse.-> R2
"""


def make_figure1(config) -> None:
    path = resolve_path(config, "figures_dir") / "figure1_architecture.mmd"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ARCHITECTURE_MERMAID, encoding="utf-8")
    print(f"Wrote {path}")


def make_figure2(config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    curve_path = resolve_path(config, "results_dir") / "evolution_curve.csv"
    if not curve_path.exists():
        print(f"SKIP figure2: {curve_path} missing - run scripts/03_run_evolution.py first.")
        return

    df = pd.read_csv(curve_path).sort_values("pct_evolution_train_processed")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    axes[0].plot(df["pct_evolution_train_processed"], df["dev_accuracy"], marker="o")
    axes[0].set_title("Dev accuracy")
    axes[0].set_xlabel("% evolution-train processed")

    axes[1].plot(df["pct_evolution_train_processed"], df["dev_recall_at_5"], marker="o", color="darkorange")
    axes[1].set_title("Dev Recall@5")
    axes[1].set_xlabel("% evolution-train processed")

    axes[2].plot(df["pct_evolution_train_processed"], df["avg_formalization_calls"], marker="o", color="green")
    axes[2].set_title("Avg. formalization LLM calls / case")
    axes[2].set_xlabel("% evolution-train processed")

    fig.tight_layout()
    out_path = resolve_path(config, "figures_dir") / "figure2_evolution_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_figure3(config) -> None:
    predictions_path = resolve_path(config, "predictions_dir") / "evodef_full_test.jsonl"
    if not predictions_path.exists():
        print(f"SKIP figure3: {predictions_path} missing - run scripts/05_run_test.py first.")
        return

    _, cases = load_corpus_and_cases(config)
    cases_by_id = {c.case_id: c for c in cases}

    predictions = read_jsonl(predictions_path)
    # Prefer a correctly-decided case whose proof involved a targeted
    # retrieval round AND a verified counterfactual - the most illustrative
    # story for the paper (proof-gap retrieval + auditable explanation).
    def score(p: dict) -> tuple:
        proof = p.get("proof") or {}
        return (
            p["prediction"] == p["gold"],
            bool(p.get("retrieved_targeted")),
            bool(proof.get("counterfactuals")),
        )

    candidates = sorted(predictions, key=score, reverse=True)
    if not candidates:
        print("SKIP figure3: no predictions found.")
        return
    chosen = candidates[0]
    case = cases_by_id.get(chosen["case_id"])

    case_study = {
        "case_id": chosen["case_id"],
        "facts_text": case.facts_text if case else None,
        "assertion": case.assertion if case else None,
        "gold_label": chosen["gold"],
        "prediction": chosen["prediction"],
        "retrieved_initial": chosen["retrieved_initial"],
        "retrieved_targeted": chosen["retrieved_targeted"],
        "proof_certificate": chosen["proof"],
    }
    out_path = resolve_path(config, "figures_dir") / "figure3_case_study.json"
    out_path.write_text(json.dumps(case_study, indent=2), encoding="utf-8")
    proof = chosen.get("proof") or {}

    def _node_lines(node: dict, parent: str | None = None, index: list[int] | None = None) -> list[str]:
        index = index if index is not None else [0]
        node_id = f"n{index[0]}"
        index[0] += 1
        label = f"{node.get('predicate', 'unknown')}\\n{node.get('status', 'unknown')}".replace('"', "'")
        lines = [f'    {node_id}["{label}"]']
        if parent is not None:
            lines.append(f"    {parent} --> {node_id}")
        for child in node.get("children", []):
            lines.extend(_node_lines(child, node_id, index))
        return lines

    root = proof.get("root")
    if root:
        mermaid = ["flowchart TD", *_node_lines(root)]
        mermaid_path = resolve_path(config, "figures_dir") / "figure3_proof_graph.mmd"
        mermaid_path.write_text("\n".join(mermaid) + "\n", encoding="utf-8")
        print(f"Wrote {mermaid_path}")
    print(f"Wrote {out_path} (case_id={chosen['case_id']})")


def main() -> None:
    config = load_config()
    make_figure1(config)
    make_figure2(config)
    make_figure3(config)


if __name__ == "__main__":
    main()
