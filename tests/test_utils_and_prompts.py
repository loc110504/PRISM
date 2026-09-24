from __future__ import annotations

from conftest import FakeOllamaBackend, json_response, REPO_ROOT
from evodef.ollama_client import OllamaClient
from evodef.prompt_templates import load_prompt_dir, load_prompt_template
from evodef.retrieval.llm_reranker import rerank
from evodef.schemas import StatuteChunk
from evodef.utils import load_config, normalize_predicate, sha256_json, stable_sort_by_id, tokenize, write_jsonl, read_jsonl


class TestUtils:
    def test_openai_environment_overrides_models(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_GENERATOR_MODEL", "test-openai-generator")
        monkeypatch.setenv("OPENAI_EMBEDDER_MODEL", "test-openai-embedder")
        config = load_config()
        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["embed_provider"] == "openai"
        assert config["models"]["generator"] == "test-openai-generator"
        assert config["models"]["embedder"] == "test-openai-embedder"

    def test_embed_provider_can_diverge_from_generator_provider(self, monkeypatch):
        # e.g. generator on Together (via the openai-compatible provider)
        # while embeddings stay on the local Ollama server.
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_EMBED_PROVIDER", "ollama")
        monkeypatch.setenv("OPENAI_GENERATOR_MODEL", "test-openai-generator")
        config = load_config()
        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["embed_provider"] == "ollama"
        assert config["models"]["generator"] == "test-openai-generator"
        assert config["models"]["embedder"] == "qwen3-embedding:0.6b"

    def test_ollama_embedder_model_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_EMBED_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_EMBEDDER_MODEL", "qwen3-embedding:4b")
        config = load_config()
        assert config["llm"]["embed_provider"] == "ollama"
        assert config["models"]["embedder"] == "qwen3-embedding:4b"

    def test_normalize_predicate_strips_args(self):
        assert normalize_predicate("foo(Person,Year)") == "foo"
        assert normalize_predicate("bare_flag") == "bare_flag"
        assert normalize_predicate("  spaced  ") == "spaced"

    def test_tokenize_lowercases_and_splits(self):
        assert tokenize("Qualifying Relative, Section 152!") == ["qualifying", "relative", "section", "152"]

    def test_sha256_json_stable_regardless_of_key_order(self):
        a = sha256_json({"x": 1, "y": 2})
        b = sha256_json({"y": 2, "x": 1})
        assert a == b

    def test_stable_sort_by_id_breaks_ties(self):
        items = [{"id": "b", "score": 1.0}, {"id": "a", "score": 1.0}, {"id": "c", "score": 2.0}]
        ranked = stable_sort_by_id(items, "id", "score", reverse=True)
        assert [i["id"] for i in ranked] == ["c", "a", "b"]

    def test_jsonl_roundtrip(self, tmp_path):
        path = tmp_path / "x.jsonl"
        write_jsonl(path, [{"a": 1}, {"a": 2}])
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}]

    def test_read_missing_jsonl_returns_empty(self, tmp_path):
        assert read_jsonl(tmp_path / "missing.jsonl") == []


class TestPromptTemplates:
    def test_load_all_shipped_prompts(self):
        templates = load_prompt_dir(REPO_ROOT / "prompts")
        expected = {"formalize_rule", "gap_query", "ground_facts", "issue_query", "verbalize", "rerank_chunks"}
        assert expected.issubset(templates.keys())
        for t in templates.values():
            assert t.system
            assert t.user_template

    def test_render_user_substitutes_placeholders(self):
        template = load_prompt_template(REPO_ROOT / "prompts" / "issue_query.txt")
        rendered = template.render_user(facts_text="FACTS_X", assertion="ASSERTION_Y")
        assert "FACTS_X" in rendered
        assert "ASSERTION_Y" in rendered
        assert "{{" not in rendered

    def test_hash_changes_when_content_changes(self):
        t1 = load_prompt_template(REPO_ROOT / "prompts" / "issue_query.txt")
        from evodef.prompt_templates import PromptTemplate

        t2 = PromptTemplate(name="x", system=t1.system + " extra", user_template=t1.user_template)
        assert t1.hash != t2.hash


class TestReranker:
    def _chunks(self):
        return [
            StatuteChunk(chunk_id="c1", section_id="1", text="direct rule text", source_file="f"),
            StatuteChunk(chunk_id="c2", section_id="2", text="exception text", source_file="f"),
            StatuteChunk(chunk_id="c3", section_id="3", text="irrelevant text", source_file="f"),
        ]

    def test_orders_by_label_priority(self):
        payload = {
            "items": [
                {"chunk_id": "c3", "label": "IRRELEVANT"},
                {"chunk_id": "c1", "label": "DIRECT"},
                {"chunk_id": "c2", "label": "EXCEPTION"},
            ]
        }
        backend = FakeOllamaBackend(responses=[json_response(payload)])
        client = OllamaClient(_backend=backend)
        template = load_prompt_template(REPO_ROOT / "prompts" / "rerank_chunks.txt")
        kept, parsed, _ = rerank(client, "m", template, "facts", "assertion", self._chunks(), keep_top_k=5)
        assert kept == ["c1", "c2", "c3"]

    def test_falls_back_to_fused_order_on_schema_failure(self):
        backend = FakeOllamaBackend(responses=[{"message": {"content": "not json"}}])
        client = OllamaClient(_backend=backend, max_schema_retries=0)
        template = load_prompt_template(REPO_ROOT / "prompts" / "rerank_chunks.txt")
        chunks = self._chunks()
        kept, parsed, _ = rerank(client, "m", template, "facts", "assertion", chunks, keep_top_k=2)
        assert kept == ["c1", "c2"]
        assert parsed is None

    def test_unlabeled_candidate_appended_after_labeled(self):
        payload = {"items": [{"chunk_id": "c1", "label": "DIRECT"}]}
        backend = FakeOllamaBackend(responses=[json_response(payload)])
        client = OllamaClient(_backend=backend)
        template = load_prompt_template(REPO_ROOT / "prompts" / "rerank_chunks.txt")
        kept, _, _ = rerank(client, "m", template, "facts", "assertion", self._chunks(), keep_top_k=5)
        assert kept[0] == "c1"
        assert set(kept) == {"c1", "c2", "c3"}
