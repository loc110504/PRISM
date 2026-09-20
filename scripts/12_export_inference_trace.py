#!/usr/bin/env python3
"""Export one stored inference trace as compact JSON and Mermaid.

Example (after a test run):
    python scripts/12_export_inference_trace.py --case-id s152_d_1_pos

The script makes no model call. It reads the trace embedded in
``outputs/raw_predictions/evodef_full_test.jsonl`` and writes figure-ready
artifacts under ``outputs/figures/inference_traces/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.utils import load_config, read_jsonl, resolve_path  # noqa: E402


def _compact(value: object, limit: int = 80) -> str:
    text = str(value).replace('"', "'").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _event_label(event: dict) -> str:
    stage = event["stage"]
    if stage == "issue_query":
        return f"Issue query\\n{_compact('; '.join(event.get('queries', [])))}"
    if stage == "retrieval_selected":
        return f"{event.get('phase', 'initial')} retrieval\\n{', '.join(event.get('selected_chunk_ids', [])) or 'none'}"
    if stage == "formalization":
        return f"Formalize {event.get('chunk_id')}\\n{event.get('source')} · {', '.join(event.get('rule_ids', [])) or 'no rule'}"
    if stage == "fact_grounding":
        statuses = ", ".join(f"{key}={value}" for key, value in event.get("grounded_statuses", {}).items())
        return f"Ground facts ({event.get('phase')})\\n{_compact(statuses)}"
    if stage == "symbolic_proof":
        return f"Proof ({event.get('phase')})\\n{event.get('target')} → {event.get('root_status')}"
    if stage == "gap_query":
        gap = event.get("gap", {})
        return f"{gap.get('gap_type')}\\n{gap.get('target_predicate')}"
    if stage == "rerank":
        return f"LLM rerank\\n{', '.join(event.get('selected_chunk_ids', []))}"
    if stage == "final_decision":
        return f"Final decision\\n{event.get('prediction')}"
    return stage.replace("_", " ")


def _mermaid(trace: list[dict]) -> str:
    lines = ["flowchart TD"]
    previous = None
    for index, event in enumerate(trace):
        node_id = f"n{index}"
        lines.append(f'    {node_id}["{_event_label(event)}"]')
        if previous is not None:
            lines.append(f"    {previous} --> {node_id}")
        previous = node_id
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True, help="Case ID to export.")
    parser.add_argument("--method", default="evodef_full", help="Prediction method/file prefix (default: evodef_full).")
    parser.add_argument("--split", default="test", help="Prediction split/file suffix (default: test).")
    args = parser.parse_args()

    config = load_config()
    predictions_path = resolve_path(config, "predictions_dir") / f"{args.method}_{args.split}.jsonl"
    rows = read_jsonl(predictions_path)
    row = next((item for item in rows if item.get("case_id") == args.case_id), None)
    if row is None:
        raise SystemExit(f"Case {args.case_id!r} is not present in {predictions_path}")
    trace = row.get("inference_trace", [])
    if not trace:
        raise SystemExit(
            "This prediction has no inference_trace. Re-run the experiment with the current pipeline."
        )

    output_dir = resolve_path(config, "figures_dir") / "inference_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.method}_{args.split}_{args.case_id}.json"
    mermaid_path = output_dir / f"{args.method}_{args.split}_{args.case_id}.mmd"
    json_path.write_text(json.dumps({"prediction": row, "inference_trace": trace}, indent=2), encoding="utf-8")
    mermaid_path.write_text(_mermaid(trace), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {mermaid_path}")


if __name__ == "__main__":
    main()
