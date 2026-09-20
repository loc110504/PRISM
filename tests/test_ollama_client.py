from __future__ import annotations

from pydantic import BaseModel

from conftest import FakeOllamaBackend, json_response
from evodef.ollama_client import OllamaClient


class Answer(BaseModel):
    value: int


class TestChatJson:
    def test_parses_valid_response_first_try(self):
        backend = FakeOllamaBackend(responses=[json_response({"value": 42})])
        client = OllamaClient(_backend=backend, max_schema_retries=1)
        parsed, record = client.chat_json("m", "sys", "user", Answer)
        assert parsed.value == 42
        assert record.attempts == 1
        assert record.parsed == {"value": 42}

    def test_retries_then_succeeds(self):
        backend = FakeOllamaBackend(
            responses=[{"message": {"content": "garbage"}}, json_response({"value": 7})]
        )
        client = OllamaClient(_backend=backend, max_schema_retries=1)
        parsed, record = client.chat_json("m", "sys", "user", Answer)
        assert parsed.value == 7
        assert record.attempts == 2

    def test_never_raises_returns_none_after_exhausting_retries(self):
        backend = FakeOllamaBackend(responses=[{"message": {"content": "x"}}, {"message": {"content": "y"}}])
        client = OllamaClient(_backend=backend, max_schema_retries=1)
        parsed, record = client.chat_json("m", "sys", "user", Answer)
        assert parsed is None
        assert record.attempts == 2

    def test_prompt_hash_is_stable_for_same_input(self):
        backend = FakeOllamaBackend(responses=[json_response({"value": 1}), json_response({"value": 1})])
        client = OllamaClient(_backend=backend)
        _, r1 = client.chat_json("m", "sys", "same user text", Answer)
        _, r2 = client.chat_json("m", "sys", "same user text", Answer)
        assert r1.prompt_hash == r2.prompt_hash

    def test_records_accumulate_in_last_calls(self):
        backend = FakeOllamaBackend(responses=[json_response({"value": 1})])
        client = OllamaClient(_backend=backend)
        client.chat_json("m", "sys", "user", Answer)
        assert len(client.last_calls) == 1


class TestChatText:
    def test_returns_raw_content(self):
        backend = FakeOllamaBackend(responses=[{"message": {"content": "hello world"}}])
        client = OllamaClient(_backend=backend)
        text, record = client.chat_text("m", "sys", "user")
        assert text == "hello world"
        assert record.attempts == 1


class TestEmbed:
    def test_returns_embeddings_list(self):
        backend = FakeOllamaBackend(embed_vectors={"a": [1.0, 2.0], "b": [3.0, 4.0]})
        client = OllamaClient(_backend=backend)
        vectors = client.embed("embed-model", ["a", "b"])
        assert vectors == [[1.0, 2.0], [3.0, 4.0]]

    def test_empty_input_returns_empty(self):
        client = OllamaClient(_backend=FakeOllamaBackend())
        assert client.embed("embed-model", []) == []
