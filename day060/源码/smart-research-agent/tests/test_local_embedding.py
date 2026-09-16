"""本地 Ollama embedding 测试（day045）：惰性维度探测与工厂选型（全离线）."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smart_research_agent.config import settings
from smart_research_agent.llm.embedding import (
    CharNgramEmbedding,
    OllamaEmbedding,
    default_embedding,
)


class _StubEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(vector)) for vector in self.vectors]
        )


class StubEmbeddingClient:
    def __init__(self, vectors):
        self.embeddings = _StubEmbeddings(vectors)


class TestOllamaEmbedding:
    def test_explicit_dimension_skips_probe(self):
        """显式给出维度时不应发起探测请求（省一次网络往返）."""
        client = StubEmbeddingClient([])
        provider = OllamaEmbedding(dimension=768, client=client)
        assert provider.dimension == 768
        assert client.embeddings.calls == []

    def test_dimension_probed_lazily_from_response(self):
        """本地模型维度取决于权重（nomic-embed-text 为 768），只能探测."""
        provider = OllamaEmbedding(client=StubEmbeddingClient([[0.1] * 768]))
        assert provider.dimension == 768
        assert len(provider._client.embeddings.calls) == 1

    def test_embed_returns_first_vector_and_records_model(self):
        provider = OllamaEmbedding(
            model="nomic-embed-text", client=StubEmbeddingClient([[0.5, 0.5]])
        )
        assert provider.embed("你好") == [0.5, 0.5]
        assert provider.model == "nomic-embed-text"
        assert provider._client.embeddings.calls[0]["model"] == "nomic-embed-text"

    def test_embed_batch_sends_all_texts_in_one_call(self):
        provider = OllamaEmbedding(
            client=StubEmbeddingClient([[1.0, 0.0], [0.0, 1.0]]),
            dimension=2,
        )
        vectors = provider.embed_batch(["a", "b"])
        assert vectors == [[1.0, 0.0], [0.0, 1.0]]
        assert provider._client.embeddings.calls[0]["input"] == ["a", "b"]

    def test_embed_batch_empty_short_circuits(self):
        provider = OllamaEmbedding(client=StubEmbeddingClient([]))
        assert provider.embed_batch([]) == []
        assert provider._client.embeddings.calls == []


class TestDefaultEmbeddingFactory:
    def test_ollama_provider_uses_settings_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "embedding_provider", "ollama")
        monkeypatch.setattr(settings, "local_embedding_model", "nomic-embed-text")
        provider = default_embedding()
        assert isinstance(provider, OllamaEmbedding)
        assert provider.model == "nomic-embed-text"

    def test_unknown_provider_lists_ollama(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "embedding_provider", "tei")
        with pytest.raises(ValueError, match="ollama"):
            default_embedding()

    def test_offline_default_unchanged(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "embedding_provider", "char-ngram")
        assert isinstance(default_embedding(), CharNgramEmbedding)
