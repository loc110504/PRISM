"""Small shared utilities: config loading, hashing, seeding, IO helpers.

Kept dependency-light and side-effect-free so every other module (and every
test) can import from here without pulling in Ollama/PYTHEN.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: str | Path) -> None:
    """Load simple KEY=VALUE entries without overwriting exported variables."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_config(
    base_path: str | Path = "configs/base.yaml",
    experiments_path: str | Path = "configs/experiments.yaml",
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load base + experiments config, merged under one dict.

    `root` defaults to the repo root so scripts can be invoked from anywhere.
    """
    root = Path(root) if root is not None else REPO_ROOT
    load_dotenv(root / ".env")
    base = load_yaml(root / base_path)
    experiments = load_yaml(root / experiments_path)
    base["_experiments"] = experiments
    base["_root"] = str(root)
    provider = os.environ.get("LLM_PROVIDER", base.get("llm", {}).get("provider", "ollama")).lower()
    if provider not in {"ollama", "openai"}:
        raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}; expected 'ollama' or 'openai'.")
    base.setdefault("llm", {})["provider"] = provider
    if provider == "openai":
        base["models"]["generator"] = os.environ.get("OPENAI_GENERATOR_MODEL", "gpt-4o-mini")
        base["models"]["embedder"] = os.environ.get("OPENAI_EMBEDDER_MODEL", "text-embedding-3-small")
    return base


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a `paths.<key>` entry in config relative to the repo root."""
    root = Path(config.get("_root", REPO_ROOT))
    rel = config["paths"][key]
    return root / rel


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys, no whitespace)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(blob)


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_predicate(name: str) -> str:
    """Strip Prolog-style argument lists and whitespace from a predicate name.

    PYTHEN's `RuleTreeEvaluator` matches predicates by exact string equality
    with no unification (see symbolic/pythen_adapter.py docstring), so a
    rule's `qualifying_relative(Person,Taxpayer,Year)` would never match a
    grounded fact `qualifying_relative(alice,bob,2018)`. Since each SARA
    case concerns exactly one scenario, EvoDef-RAG treats every predicate as
    a per-case boolean flag: arguments are dropped everywhere the symbolic
    engine looks a predicate up, while the original (possibly argument-ful)
    string extracted by the LLM is still kept for provenance/XAI display.
    """
    return name.split("(", 1)[0].strip()


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25 (03_IMPLEMENTATION_SPEC.md #4)."""
    return _WORD_RE.findall(text.lower())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str))
        f.write("\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_sort_by_id(items: list[dict[str, Any]], id_key: str, score_key: str, reverse: bool = True) -> list[dict[str, Any]]:
    """Sort by score, breaking ties deterministically by id (03_IMPLEMENTATION_SPEC.md #6)."""
    return sorted(items, key=lambda x: (-x[score_key] if reverse else x[score_key], x[id_key]))
