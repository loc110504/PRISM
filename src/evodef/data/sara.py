"""Official JHU SARA loader (02_DATA_AND_SPLITS_SPEC.md).

Downloads/extracts `https://nlp.jhu.edu/law/sara/sara.tar.gz`, filters to the
**binary entailment** cases (`s<section>..._pos.pl` / `..._neg.pl`; the
`tax_case_N.pl` numerical-QA cases are out of scope per
00_MASTER_SPEC.md #2), preserves the official test split untouched, and
splits the official train split into evolution-train/dev grouped by the
top-level statute section so paired pos/neg variants of the same section
never straddle the split.
"""
from __future__ import annotations

import random
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ..schemas import Case
from ..utils import sha256_file, sha256_text

SARA_URL = "https://nlp.jhu.edu/law/sara/sara.tar.gz"

_QUESTION_LABEL_RE = re.compile(r"\b(Entailment|Contradiction)\s*\.?\s*$", re.IGNORECASE)
_GOAL_RE = re.compile(r"^\s*:-\s*(?:\\\+\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_TOP_SECTION_RE = re.compile(r"^sec_(\d+[A-Za-z]?)")


@dataclass
class DownloadInfo:
    url: str
    sha256: str
    archive_path: Path
    extracted_dir: Path


def download_sara(raw_dir: str | Path, force: bool = False) -> DownloadInfo:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "sara.tar.gz"
    extracted_dir = raw_dir / "sara_extracted" / "sara"

    if force or not archive_path.exists():
        response = requests.get(SARA_URL, timeout=180)
        response.raise_for_status()
        archive_path.write_bytes(response.content)

    if force or not extracted_dir.exists():
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(raw_dir / "sara_extracted")

    return DownloadInfo(
        url=SARA_URL,
        sha256=sha256_file(archive_path),
        archive_path=archive_path,
        extracted_dir=extracted_dir,
    )


def _predicate_to_chunk_id(predicate: str) -> str:
    body = predicate[1:] if predicate.startswith("s") else predicate
    return f"sec_{body}"


def parse_case_file(path: str | Path) -> dict[str, Any] | None:
    """Returns a raw case dict, or None for a non-binary-entailment case
    (e.g. `tax_case_N.pl`, which has no Entailment/Contradiction verdict).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {"Text": [], "Question": [], "Facts": [], "Test": []}
    current: str | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("% Text"):
            current = "Text"
            continue
        if stripped.startswith("% Question"):
            current = "Question"
            continue
        if stripped.startswith("% Facts"):
            current = "Facts"
            continue
        if stripped.startswith("% Test"):
            current = "Test"
            continue
        if current in ("Text", "Question"):
            if stripped.startswith("%"):
                sections[current].append(stripped[1:].strip())
        elif current in ("Facts", "Test") and stripped:
            sections[current].append(line)

    facts_text = " ".join(sections["Text"]).strip()
    question_text = " ".join(sections["Question"]).strip()

    label_match = _QUESTION_LABEL_RE.search(question_text)
    if label_match is None:
        return None  # numerical case, e.g. tax_case_8.pl - not a binary entailment case

    gold_label = "ENTAILMENT" if label_match.group(1).lower() == "entailment" else "CONTRADICTION"
    assertion = _QUESTION_LABEL_RE.sub("", question_text).strip()

    predicates: list[str] = []
    for line in sections["Test"]:
        goal_match = _GOAL_RE.match(line)
        if goal_match and goal_match.group(1) != "halt":
            predicates.append(goal_match.group(1))

    gold_chunk_ids = sorted({_predicate_to_chunk_id(p) for p in predicates})
    gold_prolog_test = "\n".join(sections["Test"]).strip()

    return {
        "facts_text": facts_text,
        "assertion": assertion,
        "gold_label": gold_label,
        "gold_chunk_ids": gold_chunk_ids,
        "gold_prolog_test": gold_prolog_test,
    }


def _read_split_ids(splits_dir: Path, name: str) -> list[str]:
    path = splits_dir / name
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_raw_binary_cases(sara_dir: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Returns (official_train_cases, official_test_cases) as {case_id: raw_case_dict},
    filtered to binary entailment cases only.
    """
    sara_dir = Path(sara_dir)
    cases_dir = sara_dir / "cases"
    splits_dir = sara_dir / "splits"

    train_ids = set(_read_split_ids(splits_dir, "train"))
    test_ids = set(_read_split_ids(splits_dir, "test"))

    official_train: dict[str, dict] = {}
    official_test: dict[str, dict] = {}
    for case_id in sorted(train_ids | test_ids):
        parsed = parse_case_file(cases_dir / f"{case_id}.pl")
        if parsed is None:
            continue  # numerical case, excluded (00_MASTER_SPEC.md #2)
        target = official_train if case_id in train_ids else official_test
        target[case_id] = parsed

    return official_train, official_test


def _family_of(raw_case: dict) -> str:
    """Top-level statute section family for grouped splitting."""
    if not raw_case["gold_chunk_ids"]:
        return "unknown"
    match = _TOP_SECTION_RE.match(raw_case["gold_chunk_ids"][0])
    return match.group(1) if match else raw_case["gold_chunk_ids"][0]


def make_evolution_dev_split(
    official_train: dict[str, dict],
    seed: int,
    evolution_train_fraction: float = 0.8,
) -> dict[str, str]:
    """Groups official-train cases by statute family and greedily assigns
    whole families to evolution_train/dev to hit the target fraction without
    splitting a family (02_DATA_AND_SPLITS_SPEC.md #3).

    Returns {case_id: "evolution_train" | "dev"}.
    """
    families: dict[str, list[str]] = {}
    for case_id, raw in official_train.items():
        families.setdefault(_family_of(raw), []).append(case_id)

    family_names = sorted(families.keys())
    rng = random.Random(seed)
    rng.shuffle(family_names)

    total = len(official_train)
    target_evolution = round(total * evolution_train_fraction)

    assignment: dict[str, str] = {}
    evolution_count = 0
    for family in family_names:
        ids = families[family]
        if evolution_count < target_evolution:
            split = "evolution_train"
        else:
            split = "dev"
        for case_id in ids:
            assignment[case_id] = split
        if split == "evolution_train":
            evolution_count += len(ids)

    return assignment


def build_cases(sara_dir: str | Path, seed: int, evolution_train_fraction: float = 0.8) -> list[Case]:
    official_train, official_test = load_raw_binary_cases(sara_dir)
    split_assignment = make_evolution_dev_split(official_train, seed, evolution_train_fraction)

    cases: list[Case] = []
    for case_id, raw in official_train.items():
        cases.append(
            Case(
                case_id=case_id,
                split=split_assignment[case_id],  # type: ignore[arg-type]
                facts_text=raw["facts_text"],
                assertion=raw["assertion"],
                gold_label=raw["gold_label"],
                gold_chunk_ids=raw["gold_chunk_ids"],
                gold_prolog_test=raw["gold_prolog_test"],
                metadata={"family": _family_of(raw)},
            )
        )
    for case_id, raw in official_test.items():
        cases.append(
            Case(
                case_id=case_id,
                split="test",
                facts_text=raw["facts_text"],
                assertion=raw["assertion"],
                gold_label=raw["gold_label"],
                gold_chunk_ids=raw["gold_chunk_ids"],
                gold_prolog_test=raw["gold_prolog_test"],
                metadata={"family": _family_of(raw)},
            )
        )
    return cases


def config_hash(seed: int, evolution_train_fraction: float) -> str:
    return sha256_text(f"seed={seed};evolution_train_fraction={evolution_train_fraction}")
