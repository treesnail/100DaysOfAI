"""day043 语义缓存测试：精确/语义命中、LRU 淘汰、CachedLLM 省成本归因."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.cache import CachedLLM, SemanticCache
from smart_research_agent.llm.embedding import CharNgramEmbedding
from smart_research_agent.llm.mock import MockLLM

# ----------------------------------------------------------------------
# SemanticCache：命中与统计
# ----------------------------------------------------------------------


class TestSemanticCache:
    def test_exact_hit(self):
        cache = SemanticCache(CharNgramEmbedding())
        cache.put("你好", "回复")
        assert cache.get("你好") == "回复"
        assert cache.exact_hits == 1

    def test_exact_hit_normalizes_whitespace(self):
        """strip 归一化：首尾空白不影响命中."""
        cache = SemanticCache(CharNgramEmbedding())
        cache.put("你好", "回复")
        assert cache.get("  你好  ") == "回复"
        assert cache.exact_hits == 1

    def test_miss_when_empty(self):
        cache = SemanticCache(CharNgramEmbedding())
        assert cache.get("任意问题") is None
        assert cache.misses == 1

    def test_semantic_hit_for_reworded_query(self):
        """语义命中：措辞不同但字面高度重叠的查询复用缓存回复."""
        cache = SemanticCache(CharNgramEmbedding(), similarity_threshold=0.5)
        cache.put("机器学习是什么", "机器学习是一门学科")
        assert cache.get("什么是机器学习") == "机器学习是一门学科"
        assert cache.semantic_hits == 1
        assert cache.exact_hits == 0

    def test_semantic_miss_for_unrelated_query(self):
        cache = SemanticCache(CharNgramEmbedding(), similarity_threshold=0.5)
        cache.put("机器学习是什么", "机器学习是一门学科")
        assert cache.get("今天晚饭吃什么") is None
        assert cache.misses == 1

    def test_threshold_one_allows_only_exact(self):
        """阈值取 1.0 时退化为纯精确匹配：同义改写不再命中."""
        cache = SemanticCache(CharNgramEmbedding(), similarity_threshold=1.0)
        cache.put("机器学习是什么", "答案")
        assert cache.get("什么是机器学习") is None
        assert cache.semantic_hits == 0

    def test_lru_eviction(self):
        cache = SemanticCache(CharNgramEmbedding(), max_size=2)
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.put("q3", "r3")  # 淘汰最久未用的 q1
        assert cache.size == 2
        assert cache.get("q1") is None  # 已被淘汰
        assert cache.evictions == 1
        assert cache.get("q2") == "r2"
        assert cache.get("q3") == "r3"

    def test_lru_refreshed_on_access(self):
        """访问过的 key 更新到队尾，避免被优先淘汰."""
        cache = SemanticCache(CharNgramEmbedding(), max_size=2)
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.get("q1")  # 刷新 q1 为最近使用
        cache.put("q3", "r3")  # 应淘汰 q2 而非 q1
        assert cache.get("q1") == "r1"
        assert cache.get("q2") is None

    def test_put_updates_existing_key_without_eviction(self):
        cache = SemanticCache(CharNgramEmbedding(), max_size=2)
        cache.put("q1", "r1-v1")
        cache.put("q1", "r1-v2")
        assert cache.size == 1
        assert cache.evictions == 0
        assert cache.get("q1") == "r1-v2"

    def test_hit_rate(self):
        cache = SemanticCache(CharNgramEmbedding())
        assert cache.hit_rate == 0.0  # 无请求时命中率为 0，不做除零
        cache.put("q", "r")
        cache.get("q")  # 命中
        cache.get("nope")  # 未命中
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == pytest.approx(0.5)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(CharNgramEmbedding(), similarity_threshold=1.5)
        with pytest.raises(ValueError):
            SemanticCache(CharNgramEmbedding(), similarity_threshold=-0.1)

    def test_invalid_max_size_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(CharNgramEmbedding(), max_size=0)


# ----------------------------------------------------------------------
# CachedLLM：透明包装与省成本归因
# ----------------------------------------------------------------------


class TestCachedLLM:
    def _make(self, responses=None, threshold=0.5):
        llm = MockLLM(responses=responses if responses is not None else ["生成回复"])
        cache = SemanticCache(CharNgramEmbedding(), similarity_threshold=threshold)
        cached = CachedLLM(llm, cache, model="gpt-4o-mini")
        return cached, llm, cache

    def test_miss_calls_underlying_llm_and_stores(self):
        cached, llm, _ = self._make()
        reply = cached.chat([Message(role="user", content="你好")])
        assert reply == "生成回复"
        assert len(llm.calls) == 1
        assert cached.cache_misses == 1
        assert cached.cache_hits == 0

    def test_exact_hit_skips_llm_and_accumulates_savings(self):
        cached, llm, _ = self._make()
        first = cached.chat([Message(role="user", content="你好世界")])
        second = cached.chat([Message(role="user", content="你好世界")])
        assert first == second
        assert len(llm.calls) == 1  # 底层只被真实调用一次
        assert cached.cache_hits == 1
        assert cached.savings_usd > 0

    def test_semantic_hit_skips_llm(self):
        cached, llm, _ = self._make()
        cached.chat([Message(role="user", content="什么是机器学习")])
        reply = cached.chat([Message(role="user", content="机器学习是什么")])
        assert reply == "生成回复"
        assert len(llm.calls) == 1  # 第二次是语义命中，未走底层
        assert cached.cache_hits == 1

    def test_savings_zero_until_first_hit(self):
        cached, _, _ = self._make()
        assert cached.savings_usd == 0.0
        cached.chat([Message(role="user", content="未命中问题")])
        assert cached.savings_usd == 0.0  # 未命中不产生节省

    def test_works_as_base_llm_stream_fallback(self):
        """CachedLLM 继承 BaseLLM 的 stream 默认实现，回退到被缓存的 chat."""
        cached, llm, _ = self._make()
        chunks = list(cached.stream([Message(role="user", content="你好世界")]))
        assert "".join(chunks) == "生成回复"
        assert len(llm.calls) == 1
