#!/usr/bin/env python3
"""Leakage audit (02_DATA_AND_SPLITS_SPEC.md #9). Run BEFORE scripts/05-07
(Milestone F pre-flight) and again after, to confirm nothing changed.

Every check degrades to `"status": "not_yet_run"` instead of failing when
its required artifact does not exist yet, so this can be run at any point in
the pipeline to see what is/isn't verifiable so far.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cases_by_split, load_corpus_and_cases  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.utils import load_config, read_jsonl, resolve_path, sha256_text, write_json  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
PROMPTS_DIR_NAME = "prompts"


def check_no_test_ids_in_prompts(config, test_case_ids: list[str]) -> dict:
    prompts_dir = Path(config["_root"]) / config["paths"]["prompts_dir"]
    hits = []
    for path in sorted(prompts_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for case_id in test_case_ids:
            if case_id in text:
                hits.append({"file": path.name, "case_id": case_id})
    return {"status": "pass" if not hits else "FAIL", "hits": hits}


def check_no_test_text_in_memory(config, test_cases: list) -> dict:
    checkpoints_dir = resolve_path(config, "checkpoints_dir")
    memory_files = sorted(checkpoints_dir.glob("norm_memory_*.json")) + sorted(checkpoints_dir.glob("gap_memory_*.json"))
    if not memory_files:
        return {"status": "not_yet_run", "reason": "no memory checkpoint files found - run scripts/03_run_evolution.py"}

    test_texts = [t for c in test_cases for t in (c.facts_text, c.assertion) if len(t) > 30]
    hits = []
    for path in memory_files:
        blob = path.read_text(encoding="utf-8")
        for text in test_texts:
            if text in blob:
                hits.append({"file": path.name, "text_snippet": text[:60]})
    return {"status": "pass" if not hits else "FAIL", "checked_files": [p.name for p in memory_files], "hits": hits}


def check_no_accepted_checkpoint_labeled_test(config) -> dict:
    checkpoints_dir = resolve_path(config, "checkpoints_dir")
    memory_files = sorted(checkpoints_dir.glob("norm_memory_*.json"))
    if not memory_files:
        return {"status": "not_yet_run"}
    import json

    hits = []
    for path in memory_files:
        data = json.loads(path.read_text())
        for key, entry in data.items():
            checkpoint_label = entry.get("accepted_checkpoint")
            if checkpoint_label and "test" in str(checkpoint_label).lower():
                hits.append({"file": path.name, "key": key, "accepted_checkpoint": checkpoint_label})
    return {"status": "pass" if not hits else "FAIL", "hits": hits}


def check_tuning_and_evolution_never_read_test_split() -> dict:
    """Static check: scripts/00-04 must never call cases_by_split(cases, "test")."""
    pattern = re.compile(r'cases_by_split\([^)]*["\']test["\']')
    offending = []
    for script_name in ["00_prepare_sara.py", "01_build_index.py", "02_tune_prompts_dev.py", "03_run_evolution.py", "04_freeze_checkpoint.py"]:
        path = SCRIPTS_DIR / script_name
        if not path.exists():
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offending.append(script_name)
    return {"status": "pass" if not offending else "FAIL", "offending_scripts": offending}


def check_write_memory_disabled_at_test() -> dict:
    """Static check: scripts/05-07 must construct resources with write_memory=False."""
    offending = []
    for script_name in ["05_run_test.py", "06_run_baselines.py", "07_run_ablations.py"]:
        path = SCRIPTS_DIR / script_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "write_memory=False" not in text:
            offending.append(script_name)
    return {"status": "pass" if not offending else "FAIL", "offending_scripts": offending}


def check_prompt_hashes_match_current_files(config) -> dict:
    predictions_dir = resolve_path(config, "predictions_dir")
    test_files = sorted(predictions_dir.glob("*_test.jsonl"))
    if not test_files:
        return {"status": "not_yet_run", "reason": "no test predictions found - run scripts/05/06_run_*.py"}

    from evodef.prompt_templates import load_prompt_dir

    prompts_dir = Path(config["_root"]) / config["paths"]["prompts_dir"]
    current_hashes = {name: t.hash for name, t in load_prompt_dir(prompts_dir).items()}
    frozen_path = resolve_path(config, "checkpoints_dir") / "frozen_prompt_config.json"
    frozen_hashes = None
    if frozen_path.exists():
        import json

        frozen_hashes = json.loads(frozen_path.read_text(encoding="utf-8")).get("prompt_hashes")
    if frozen_hashes is None:
        return {
            "status": "FAIL",
            "reason": "frozen prompt hashes are missing; run scripts/02_tune_prompts_dev.py before test evaluation",
        }

    mismatches = []
    for path in test_files:
        for row in read_jsonl(path):
            for prompt_name, recorded_hash in row.get("prompt_hashes", {}).items():
                current = current_hashes.get(prompt_name)
                frozen = frozen_hashes.get(prompt_name)
                if current is None or recorded_hash != current or recorded_hash != frozen:
                    mismatches.append({"file": path.name, "case_id": row["case_id"], "prompt": prompt_name})
    current_differs_from_frozen = current_hashes != frozen_hashes
    return {
        "status": "pass" if not mismatches and not current_differs_from_frozen else "FAIL",
        "checked_files": [p.name for p in test_files],
        "current_differs_from_frozen": current_differs_from_frozen,
        "mismatches": mismatches[:20],
    }


def check_config_hash_consistent_per_method(config) -> dict:
    predictions_dir = resolve_path(config, "predictions_dir")
    test_files = sorted(predictions_dir.glob("*_test.jsonl"))
    if not test_files:
        return {"status": "not_yet_run"}
    inconsistent = []
    for path in test_files:
        hashes = {row["config_hash"] for row in read_jsonl(path)}
        if len(hashes) > 1:
            inconsistent.append({"file": path.name, "distinct_config_hashes": len(hashes)})
    return {"status": "pass" if not inconsistent else "FAIL", "inconsistent": inconsistent}


def main() -> None:
    config = load_config()
    _, cases = load_corpus_and_cases(config)
    test_cases = cases_by_split(cases, "test")
    test_case_ids = [c.case_id for c in test_cases]

    results = {
        "no_test_ids_in_prompts": check_no_test_ids_in_prompts(config, test_case_ids),
        "no_test_text_in_memory": check_no_test_text_in_memory(config, test_cases),
        "no_accepted_checkpoint_labeled_test": check_no_accepted_checkpoint_labeled_test(config),
        "tuning_and_evolution_never_read_test_split": check_tuning_and_evolution_never_read_test_split(),
        "write_memory_disabled_at_test": check_write_memory_disabled_at_test(),
        "prompt_hashes_match_current_files": check_prompt_hashes_match_current_files(config),
        "config_hash_consistent_per_method": check_config_hash_consistent_per_method(config),
    }

    overall = "PASS"
    for name, result in results.items():
        print(f"{name}: {result['status']}")
        if result["status"] == "FAIL":
            overall = "FAIL"
            print(f"  -> {result}")

    out = {"overall": overall, "checks": results}
    out_path = resolve_path(config, "outputs_dir") / "leakage_audit.json"
    write_json(out_path, out)
    print(f"\nOverall: {overall}")
    print(f"Wrote {out_path}")

    if overall == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
