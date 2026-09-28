"""Embedding 提供方测试（day041）：抽象、四种实现与工厂选型.

全部离线：神经提供方用注入的假模型对象覆盖，云端提供方用假客户端
覆盖，工厂测试通过 monkeypatch 切换 settings——不需要任何真实权重
或网络请求。
"""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import pytest

from smart_research_agent.config import settings
from smart_research_agent.llm.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    CharNgramEmbedding,
    EmbeddingProvider,
    MockEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    default_embedding,
    l2_normalize,
)
from smart_research_agent.memory.long_term import LongTermMemory
from smart_research_agent.memory.vector_store import InMemoryVectorStore, VectorRecord


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


class TestL2Normalize:
    def test_unit_norm(self):
        assert norm(l2_normalize([3.0, 4.0])) == pytest.approx(1.0)

    def test_direction_preserved(self):
        assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])

    def test_zero_vector_returned_as_is(self):
        assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_returns_copy(self):
        original = [0.0, 0.0]
        assert l2_normalize(original) is not original


class TestMockEmbeddingExtras:
    def test_values_in_unit_range(self):
        vector = MockEmbedding(dimension=64).embed("任意文本")
        assert all(-1.0 <= x <= 1.0 for x in vector)

    def test_different_texts_differ(self):
        emb = MockEmbedding(dimension=64)
        assert emb.embed("文本甲") != emb.embed("文本乙")


class TestCharNgramEmbedding:
    def test_dimension_and_deterministic(self):
        emb = CharNgramEmbedding(dimension=128)
        assert emb.dimension == 128
        assert emb.embed("机器学习") == emb.embed("机器学习")

    def test_normalized_to_unit_length(self):
        vector = CharNgramEmbedding().embed("一段非空的文本")
        assert norm(vector) == pytest.approx(1.0)

    def test_empty_text_gives_zero_vector(self):
        emb = CharNgramEmbedding(dimension=32)
        assert emb.embed("") == [0.0] * 32

    def test_short_text_not_dropped(self):
        """单字文本比所有 n 都短，也应得到非零向量."""
        assert any(x != 0.0 for x in CharNgramEmbedding().embed("猫"))

    def test_identical_texts_similarity_one(self):
        from smart_research_agent.memory.vector_store import cosine_similarity

        emb = CharNgramEmbedding()
        a, b = emb.embed_batch(["同一段文本", "同一段文本"])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_shared_ngrams_imply_higher_similarity(self):
        """词汇层面语义：字面共享越多，向量越接近（day041 核心性质）."""
        from smart_research_agent.memory.vector_store import cosine_similarity

        emb = CharNgramEmbedding()
        anchor, similar, unrelated = emb.embed_batch(
            ["我喜欢学习机器学习", "我爱学习机器学习", "今天晚饭吃什么"]
        )
        sim_close = cosine_similarity(anchor, similar)
        sim_far = cosine_similarity(anchor, unrelated)
        assert sim_close > 0.5
        assert sim_close > sim_far

    def test_embed_batch_matches_embed(self):
        emb = CharNgramEmbedding()
        texts = ["甲", "乙丙丁", "戊己庚辛"]
        assert emb.embed_batch(texts) == [emb.embed(t) for t in texts]

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            CharNgramEmbedding(dimension=0)

    def test_invalid_ngram_sizes_raises(self):
        with pytest.raises(ValueError):
            CharNgramEmbedding(ngram_sizes=())
        with pytest.raises(ValueError):
            CharNgramEmbedding(ngram_sizes=(0, 2))

    def test_custom_ngram_sizes(self):
        emb = CharNgramEmbedding(dimension=64, ngram_sizes=(1,))
        assert norm(emb.embed("单字符切分")) == pytest.approx(1.0)


class FakeSTModel:
    """sentence-transformers 模型的测试替身.

    ``new_api`` 控制是否暴露 v5.4+ 的 get_embedding_dimension——
    用于覆盖新旧两套维度解析路径。
    """

    def __init__(self, dim: int = 4, new_api: bool = True):
        self._dim = dim
        self.calls: list[tuple[list[str], bool]] = []
        if new_api:
            self.get_embedding_dimension = lambda: dim
        else:
            self.get_sentence_embedding_dimension = lambda: dim

    def encode(self, sentences, normalize_embeddings: bool = False):
        self.calls.append((list(sentences), normalize_embeddings))
        return [[float(i + 1)] * self._dim for i in range(len(sentences))]


class TestSentenceTransformerEmbedding:
    def test_injected_model_new_api_dimension(self):
        emb = SentenceTransformerEmbedding(model=FakeSTModel(dim=8))
        assert emb.dimension == 8

    def test_injected_model_old_api_dimension(self):
        emb = SentenceTransformerEmbedding(model=FakeSTModel(dim=6, new_api=False))
        assert emb.dimension == 6

    def test_embed_batch_requests_normalized_vectors(self):
        fake = FakeSTModel()
        emb = SentenceTransformerEmbedding(model=fake)
        vectors = emb.embed_batch(["甲", "乙"])
        assert vectors == [[1.0] * 4, [2.0] * 4]
        assert fake.calls[-1] == (["甲", "乙"], True)

    def test_embed_single_goes_through_list_entry(self):
        """encode 对字符串与列表返回不同形状，embed 一律走列表入口."""
        fake = FakeSTModel()
        emb = SentenceTransformerEmbedding(model=fake)
        assert emb.embed("单条") == [1.0] * 4
        assert fake.calls[-1][0] == ["单条"]

    def test_unknown_dimension_raises(self):
        class NoDimModel:
            def get_embedding_dimension(self):
                return None

        with pytest.raises(RuntimeError):
            SentenceTransformerEmbedding(model=NoDimModel())

    def test_missing_package_raises_with_install_hint(self, monkeypatch):
        """sys.modules[name] = None 让 import 抛 ImportError，模拟未安装."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
            SentenceTransformerEmbedding()

    def test_lazy_import_success_path(self, monkeypatch):
        """装上假的 sentence_transformers 模块，验证惰性导入与构造链路."""
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = lambda name: FakeSTModel(dim=4)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        emb = SentenceTransformerEmbedding(model_name="fake-model")
        assert emb.model_name == "fake-model"
        assert emb.dimension == 4


class FakeEmbeddingResource:
    """openai client.embeddings 的测试替身：记录请求、按序造向量."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.requests: list[dict] = []

    def create(self, model: str, input: list[str]):
        self.requests.append({"model": model, "input": list(input)})
        data = [
            SimpleNamespace(embedding=[float(i + 1)] * self.dim) for i in range(len(input))
        ]
        return SimpleNamespace(data=data)


class FakeOpenAIClient:
    def __init__(self, dim: int = 4):
        self.embeddings = FakeEmbeddingResource(dim)


class TestOpenAIEmbedding:
    def test_known_model_dimensions(self):
        assert OpenAIEmbedding(client=FakeOpenAIClient()).dimension == 1536
        assert (
            OpenAIEmbedding(model="text-embedding-3-large", client=FakeOpenAIClient()).dimension
            == 3072
        )

    def test_dimension_table_matches_official_specs(self):
        assert OPENAI_EMBEDDING_DIMENSIONS == {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def test_unknown_model_requires_explicit_dimension(self):
        with pytest.raises(ValueError, match="dimension"):
            OpenAIEmbedding(model="custom-embed-v9", client=FakeOpenAIClient())
        emb = OpenAIEmbedding(model="custom-embed-v9", client=FakeOpenAIClient(), dimension=96)
        assert emb.dimension == 96

    def test_embed_batch_single_request_in_order(self):
        client = FakeOpenAIClient()
        emb = OpenAIEmbedding(client=client)
        vectors = emb.embed_batch(["甲", "乙", "丙"])
        assert len(client.embeddings.requests) == 1
        assert client.embeddings.requests[0] == {
            "model": "text-embedding-3-small",
            "input": ["甲", "乙", "丙"],
        }
        assert vectors == [[1.0] * 4, [2.0] * 4, [3.0] * 4]

    def test_embed_single_text(self):
        emb = OpenAIEmbedding(client=FakeOpenAIClient())
        assert emb.embed("单条") == [1.0] * 4

    def test_empty_batch_sends_no_request(self):
        client = FakeOpenAIClient()
        assert OpenAIEmbedding(client=client).embed_batch([]) == []
        assert client.embeddings.requests == []

    def test_model_property(self):
        emb = OpenAIEmbedding(model="text-embedding-3-large", client=FakeOpenAIClient())
        assert emb.model == "text-embedding-3-large"


class CountingEmbedding(EmbeddingProvider):
    """记录 embed 调用次数的最小提供方：验证 embed_batch 默认实现."""

    def __init__(self):
        self.embed_calls = 0

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [float(len(text)), 1.0]


class TestEmbedBatchDefault:
    def test_default_batch_calls_embed_per_item(self):
        emb = CountingEmbedding()
        vectors = emb.embed_batch(["a", "bb", "ccc"])
        assert emb.embed_calls == 3
        assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]

    def test_default_batch_empty_input(self):
        emb = CountingEmbedding()
        assert emb.embed_batch([]) == []
        assert emb.embed_calls == 0


class TestDefaultEmbeddingFactory:
    def test_mock_provider(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "mock")
        assert isinstance(default_embedding(), MockEmbedding)

    def test_char_ngram_provider(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "char-ngram")
        assert isinstance(default_embedding(), CharNgramEmbedding)

    def test_default_provider_is_char_ngram(self):
        """settings 的默认值就是 char-ngram：离线也有词汇层面语义."""
        assert settings.embedding_provider == "char-ngram"
        assert isinstance(default_embedding(), CharNgramEmbedding)

    def test_sentence_transformer_missing_package(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "sentence-transformer")
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(RuntimeError):
            default_embedding()

    def test_openai_without_key_falls_back_to_char_ngram(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", None)
        assert isinstance(default_embedding(), CharNgramEmbedding)

    def test_openai_with_key_returns_openai_embedding(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        emb = default_embedding()
        assert isinstance(emb, OpenAIEmbedding)
        assert emb.model == settings.openai_embedding_model

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "quantum")
        with pytest.raises(ValueError, match="embedding_provider"):
            default_embedding()


class TestVectorStoreMinScore:
    def test_min_score_filters_low_similarity(self):
        store = InMemoryVectorStore()
        store.add(VectorRecord(id="a", text="A", vector=[1.0, 0.0]))
        store.add(VectorRecord(id="b", text="B", vector=[0.0, 1.0]))
        results = store.search([1.0, 0.0], top_k=5, min_score=0.5)
        assert [r.id for r, _ in results] == ["a"]

    def test_min_score_none_keeps_all(self):
        store = InMemoryVectorStore()
        store.add(VectorRecord(id="a", text="A", vector=[1.0, 0.0]))
        store.add(VectorRecord(id="b", text="B", vector=[0.0, 1.0]))
        assert len(store.search([1.0, 0.0], top_k=5)) == 2

    def test_min_score_boundary_inclusive(self):
        store = InMemoryVectorStore()
        store.add(VectorRecord(id="a", text="A", vector=[1.0, 0.0]))
        assert len(store.search([1.0, 0.0], min_score=1.0)) == 1


class TestLongTermMemoryBatch:
    def test_remember_many_and_recall(self):
        emb = CharNgramEmbedding()
        mem = LongTermMemory(embedding=emb)
        ids = mem.remember_many(["机器学习是人工智能的分支", "深度学习使用神经网络"])
        assert len(ids) == 2 and len(set(ids)) == 2
        hits = mem.recall("机器学习是什么", top_k=1)
        assert hits[0] == "机器学习是人工智能的分支"

    def test_remember_many_stores_metadatas(self):
        emb = CharNgramEmbedding()
        store = InMemoryVectorStore()
        mem = LongTermMemory(embedding=emb, store=store)
        mem.remember_many(["巴黎是法国的首都"], metadatas=[{"topic": "geo"}])
        results = store.search(emb.embed("巴黎是法国的首都"), top_k=1)
        assert results[0][0].metadata == {"topic": "geo"}

    def test_remember_many_metadata_length_mismatch_raises(self):
        mem = LongTermMemory(embedding=CharNgramEmbedding())
        with pytest.raises(ValueError, match="metadatas"):
            mem.remember_many(["甲", "乙"], metadatas=[{}])

    def test_recall_min_score_exact_match(self):
        """min_score=0.99 只留得下完全同文的记忆（相似度恰为 1.0）."""
        mem = LongTermMemory(embedding=CharNgramEmbedding())
        mem.remember_many(["锚点句子", "另一段毫不相关的长篇内容"])
        hits = mem.recall("锚点句子", top_k=2, min_score=0.99)
        assert hits == ["锚点句子"]
