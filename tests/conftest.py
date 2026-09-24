"""Shared test fixtures. No test in this suite talks to a real Ollama server -
the user explicitly runs the pipeline against their own remote server later
(see README.md). `FakeOllamaBackend` gives `OllamaClient` a scripted,
deterministic transport so every LLM-dependent code path is still exercised.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeOllamaBackend:
    """Replays a queue of canned responses, or calls a handler function.

    Usage:
        backend = FakeOllamaBackend(responses=[{"message": {"content": "..."}}])
        client = OllamaClient(_backend=backend)
    """

    responses: list[dict[str, Any]] = field(default_factory=list)
    handler: Callable[[str, list[dict[str, str]]], dict[str, Any]] | None = None
    embed_vectors: dict[str, list[float]] | None = None
    embed_handler: Callable[[str, list[str]], dict[str, Any]] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, model: str, messages: list[dict[str, str]], format: Any = None, options: dict | None = None) -> dict:
        self.calls.append({"model": model, "messages": messages, "format": format, "options": options})
        if self.handler is not None:
            return self.handler(model, messages)
        if not self.responses:
            raise AssertionError("FakeOllamaBackend ran out of scripted responses")
        return self.responses.pop(0)

    def embed(self, model: str, input: list[str]) -> dict:
        self.calls.append({"model": model, "input": input})
        if self.embed_handler is not None:
            return self.embed_handler(model, input)
        if self.embed_vectors is not None:
            return {"embeddings": [self.embed_vectors.get(text, [0.0, 0.0]) for text in input]}
        # deterministic hash-based fake embedding
        import hashlib

        def fake_vec(text: str) -> list[float]:
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:8]]

        return {"embeddings": [fake_vec(t) for t in input]}


def json_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {"message": {"content": json.dumps(payload)}, "prompt_eval_count": 10, "eval_count": 5}


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


_LLM_ENV_VARS = [
    "LLM_PROVIDER", "LLM_EMBED_PROVIDER", "OLLAMA_HOST", "OLLAMA_EMBEDDER_MODEL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_GENERATOR_MODEL", "OPENAI_EMBEDDER_MODEL",
]


@pytest.fixture(autouse=True)
def _isolated_llm_env(monkeypatch):
    """`load_config()` reads the repo's real `.env` by default (root=REPO_ROOT),
    which would otherwise leak a developer's own LLM_PROVIDER/OPENAI_*/OLLAMA_*
    choices into every test that calls `load_config()` without explicitly
    overriding them. Stub out the `.env` read entirely so tests only ever see
    what they monkeypatch themselves, regardless of the real `.env` content.
    """
    for key in _LLM_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("evodef.utils.load_dotenv", lambda path: None)
