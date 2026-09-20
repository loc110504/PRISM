"""Thin wrapper around the Ollama API (03_IMPLEMENTATION_SPEC.md #3).

Every LLM call in the pipeline goes through `OllamaClient.chat_json`,
`chat_text`, or `embed` so that retries, hashing, latency, and token
accounting are handled in exactly one place.

`base_url` defaults to `configs/base.yaml`'s `ollama.base_url` but is
overridden by the `OLLAMA_HOST` environment variable when set, so the same
code runs unmodified against a remote Ollama server
(see README.md "Running against a remote Ollama server").
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .utils import sha256_text

T = TypeVar("T", bound=BaseModel)


class SchemaValidationFailure(RuntimeError):
    """Raised when structured output fails schema validation after all retries."""

    def __init__(self, message: str, raw_responses: list[str]):
        super().__init__(message)
        self.raw_responses = raw_responses


@dataclass
class LLMCallRecord:
    """Everything needed to reproduce and audit one LLM call."""

    model: str
    prompt_hash: str
    system_hash: str
    raw_response: str
    parsed: Any
    latency_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    attempts: int
    temperature: float
    seed: int | None


@dataclass
class OllamaClient:
    """Wrapper with retries, hashing, and latency/token accounting.

    The actual transport is delegated to `self._backend`, an object exposing
    `chat(model, messages, format=None, options=None) -> dict` and
    `embed(model, input) -> dict`, matching the `ollama` python package's
    `Client` interface. Tests inject a `FakeOllamaBackend` instead of hitting
    a real server.
    """

    base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    max_schema_retries: int = 1
    request_timeout_s: int = 300
    _backend: Any = None
    last_calls: list[LLMCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self._backend is None:
            self._backend = _RealOllamaBackend(self.base_url, self.request_timeout_s)

    # ------------------------------------------------------------------
    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        pydantic_model: Type[T],
        temperature: float = 0,
        seed: int | None = None,
    ) -> tuple[T | None, LLMCallRecord]:
        """Structured-output chat call. Returns (parsed_or_None, record).

        Retries at most `max_schema_retries` times on schema/JSON parse
        failure. Never raises on failure: callers must check for `None` and
        follow 03_IMPLEMENTATION_SPEC.md #8 (never crash a benchmark run on
        one malformed LLM output).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        schema = pydantic_model.model_json_schema()
        raw_responses: list[str] = []
        parsed: T | None = None
        attempts = 0
        last_prompt_tokens = None
        last_output_tokens = None
        t0 = time.monotonic()

        for attempt in range(self.max_schema_retries + 1):
            attempts += 1
            if attempt > 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous output did not match the required JSON schema. "
                            f"Schema: {schema}. Errors: {raw_responses[-1] if raw_responses else ''}. "
                            "Return corrected JSON only."
                        ),
                    }
                ]
            response = self._backend.chat(
                model=model,
                messages=messages,
                format=schema,
                options={"temperature": temperature, "seed": seed},
            )
            content = _extract_content(response)
            raw_responses.append(content)
            last_prompt_tokens = response.get("prompt_eval_count", last_prompt_tokens)
            last_output_tokens = response.get("eval_count", last_output_tokens)
            try:
                parsed = pydantic_model.model_validate_json(content)
                break
            except ValidationError as e:
                raw_responses[-1] = f"{content}\n---VALIDATION ERROR---\n{e}"
                parsed = None
                continue

        latency_s = time.monotonic() - t0
        record = LLMCallRecord(
            model=model,
            prompt_hash=sha256_text(user),
            system_hash=sha256_text(system),
            raw_response=raw_responses[-1] if raw_responses else "",
            parsed=parsed.model_dump() if parsed is not None else None,
            latency_s=latency_s,
            prompt_tokens=last_prompt_tokens,
            output_tokens=last_output_tokens,
            attempts=attempts,
            temperature=temperature,
            seed=seed,
        )
        self.last_calls.append(record)
        return parsed, record

    # ------------------------------------------------------------------
    def chat_text(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0,
        seed: int | None = None,
    ) -> tuple[str, LLMCallRecord]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        t0 = time.monotonic()
        response = self._backend.chat(
            model=model,
            messages=messages,
            format=None,
            options={"temperature": temperature, "seed": seed},
        )
        latency_s = time.monotonic() - t0
        content = _extract_content(response)
        record = LLMCallRecord(
            model=model,
            prompt_hash=sha256_text(user),
            system_hash=sha256_text(system),
            raw_response=content,
            parsed=content,
            latency_s=latency_s,
            prompt_tokens=response.get("prompt_eval_count"),
            output_tokens=response.get("eval_count"),
            attempts=1,
            temperature=temperature,
            seed=seed,
        )
        self.last_calls.append(record)
        return content, record

    # ------------------------------------------------------------------
    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._backend.embed(model=model, input=texts)
        embeddings = response.get("embeddings")
        if embeddings is None:
            # Single-text legacy shape: {"embedding": [...]}.
            embeddings = [response["embedding"]]
        return embeddings


def _extract_content(response: dict[str, Any]) -> str:
    if "message" in response:
        return response["message"]["content"]
    if "response" in response:
        return response["response"]
    raise KeyError(f"Unrecognized Ollama response shape: {list(response.keys())}")


class _RealOllamaBackend:
    """Delegates to the `ollama` python package's Client."""

    def __init__(self, base_url: str, timeout_s: int):
        import ollama  # imported lazily so unit tests don't require it configured

        self._client = ollama.Client(host=base_url, timeout=timeout_s)

    def chat(self, model: str, messages: list[dict[str, str]], format: Any = None, options: dict | None = None) -> dict:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "options": options or {}}
        if format is not None:
            kwargs["format"] = format
        result = self._client.chat(**kwargs)
        return dict(result)

    def embed(self, model: str, input: list[str]) -> dict:
        result = self._client.embed(model=model, input=input)
        return dict(result)
