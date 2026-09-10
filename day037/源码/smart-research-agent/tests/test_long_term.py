"""长期记忆与向量检索测试."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.embedding import MockEmbedding
from smart_research_agent.memory.long_term import LongTermMemory
from smart_research_agent.memory.vector_store import (
    InMemoryVectorStore,
    VectorRecord,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestVectorStore:
    def test_search_orders_by_similarity(self):
        store = InMemoryVectorStore()
        store.add(VectorRecord(id="a", text="A", vector=[1.0, 0.0]))
        store.add(VectorRecord(id="b", text="B", vector=[0.0, 1.0]))
        results = store.search([0.9, 0.1], top_k=2)
        assert results[0][0].id == "a"

    def test_top_k_limits_results(self):
        store = InMemoryVectorStore()
        for i in range(5):
            store.add(VectorRecord(id=str(i), text=str(i), vector=[float(i + 1), 0.0]))
        assert len(store.search([1.0, 0.0], top_k=2)) == 2

    def test_delete(self):
        store = InMemoryVectorStore()
        store.add(VectorRecord(id="a", text="A", vector=[1.0]))
        store.delete("a")
        assert len(store) == 0


class TestMockEmbedding:
    def test_deterministic(self):
        emb = MockEmbedding(dimension=32)
        assert emb.embed("你好") == emb.embed("你好")

    def test_dimension(self):
        emb = MockEmbedding(dimension=32)
        assert len(emb.embed("任意文本")) == 32


class TestLongTermMemory:
    def test_remember_and_recall_exact(self):
        mem = LongTermMemory(embedding=MockEmbedding())
        mem.remember("Python 是一种编程语言")
        mem.remember("今天天气不错")
        # 相同文本的伪向量完全一致，recall 精确文本必然排第一
        results = mem.recall("Python 是一种编程语言", top_k=1)
        assert results[0] == "Python 是一种编程语言"
