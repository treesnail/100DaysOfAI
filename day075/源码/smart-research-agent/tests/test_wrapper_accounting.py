"""包装器的成本归因测试（day048）：``CachedLLM`` 必须交代"我替谁说话".

背景（day047 复盘遗留项）：``CostTracker.record_from_llm`` 读的是
``llm.usage_log``，而 ``CachedLLM`` 是包装器、自己不产生 token。把它直接
交给计费器就会在"包装器没有 usage_log"处断链，真实用量被记成 0。
本文件验证 ``IntegratedPipeline._account`` 能穿过 ``CachedLLM`` 找到叶子，
并验证修复**没有改变任何既有行为**（缓存命中路径依旧不产生用量）。
"""

from __future__ import annotations

import logging

from smart_research_agent.integration.pipeline import IntegratedPipeline
from smart_research_agent.llm.base import Message
from smart_research_agent.llm.cache import CachedLLM, SemanticCache
from smart_research_agent.llm.embedding import CharNgramEmbedding
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.router import ModelRouter, ModelSpec
from smart_research_agent.observability.cost_tracker import CostTracker

PIPELINE_LOGGER = "smart_research_agent.integration.pipeline"

#: 断言"没有断链告警"用的关键词（修复前它会被打印出来）
BROKEN_CHAIN_WARNING = "不提供 usage_log"


def make_cache() -> SemanticCache:
    """精确匹配缓存：阈值 1.0，同一条查询第二次必然命中."""
    return SemanticCache(
        embedding=CharNgramEmbedding(), similarity_threshold=1.0, max_size=8
    )


def make_cached() -> tuple[CachedLLM, MockLLM]:
    """构造 ``CachedLLM(MockLLM)`` 与内层叶子，便于断言归因对象."""
    leaf = MockLLM(responses=["来自叶子的真实回复"])
    return CachedLLM(leaf, make_cache()), leaf


class TestCachedLLMLastUsedLlm:
    """``last_used_llm`` 的透传语义."""

    def test_points_to_inner_leaf(self):
        cached, leaf = make_cached()
        assert cached.last_used_llm is leaf

    def test_passes_through_router_leaf(self):
        """包住 ModelRouter 时透传路由真正选中的叶子."""
        cheap = MockLLM(default="便宜模型的回答")
        router = ModelRouter([ModelSpec("cheap", cheap, capability=5, cost_per_1k=0.001)])
        cached = CachedLLM(router, make_cache())
        cached.chat([Message(role="user", content="1 + 1 等于几")])
        assert router.last_used_llm is cheap
        assert cached.last_used_llm is cheap

    def test_router_before_any_decision_falls_back_to_router(self):
        """路由尚未决策时退化为"账记在路由器头上"，由调用方兜底处理."""
        router = ModelRouter([ModelSpec("cheap", MockLLM(), capability=5, cost_per_1k=0.001)])
        cached = CachedLLM(router, make_cache())
        assert router.last_used_llm is None
        assert cached.last_used_llm is router

    def test_nested_wrappers_resolve_to_leaf(self):
        leaf = MockLLM()
        inner = CachedLLM(leaf, make_cache())
        outer = CachedLLM(inner, make_cache())
        assert outer.last_used_llm is leaf

    def test_wrapper_itself_has_no_usage_log(self):
        """断链的根因：包装器没有 usage_log，所以必须把叶子交出去."""
        cached, _ = make_cached()
        assert not hasattr(cached, "usage_log")
        assert hasattr(cached.last_used_llm, "usage_log")


class TestPipelineAccountingThroughCachedLLM:
    """流水线视角：成本归因落在叶子上，而不是记成 0."""

    def test_cost_is_attributed_with_exact_tokens(self, caplog):
        cached, leaf = make_cached()
        tracker = CostTracker()
        pipeline = IntegratedPipeline(cached, tracker=tracker)

        with caplog.at_level(logging.WARNING, logger=PIPELINE_LOGGER):
            result = pipeline.run("请解释一下语义缓存的成本收益")

        # CachedLLM 自述模型名为 gpt-4o-mini（价格表中存在），因此成本非 0
        assert result.model == "gpt-4o-mini"
        assert result.cost_usd > 0
        assert round(tracker.total_cost, 8) == result.cost_usd
        # token 数与叶子实际记录的用量逐位一致
        assert result.prompt_tokens == leaf.usage_log[0]["prompt_tokens"] > 0
        assert result.completion_tokens == leaf.usage_log[0]["completion_tokens"]
        assert "tokens=" in (result.stage("account").detail)  # type: ignore[union-attr]
        # 修复前这里会有"不提供 usage_log"的告警
        assert not any(BROKEN_CHAIN_WARNING in record.message for record in caplog.records)

    def test_cache_hit_produces_no_new_cost(self, caplog):
        cached, leaf = make_cached()
        tracker = CostTracker()
        pipeline = IntegratedPipeline(cached, tracker=tracker)

        with caplog.at_level(logging.WARNING, logger=PIPELINE_LOGGER):
            first = pipeline.run("重复的问题")
            second = pipeline.run("重复的问题")

        assert first.cost_usd > 0
        assert first.prompt_tokens > 0
        # 第二次命中 CachedLLM 的缓存：不调用模型、不新增用量、成本为 0
        assert second.cost_usd == 0.0
        assert second.prompt_tokens == 0
        assert second.completion_tokens == 0
        assert len(leaf.calls) == 1
        assert cached.cache_hits == 1
        assert not any(BROKEN_CHAIN_WARNING in record.message for record in caplog.records)

    def test_tracker_can_consume_the_leaf_exposed_by_the_wrapper(self):
        """把 ``cached.last_used_llm`` 交给计费器即可正常记账."""
        cached, _ = make_cached()
        cached.chat([Message(role="user", content="你好")])
        tracker = CostTracker()
        assert tracker.record_from_llm(cached.last_used_llm, model="gpt-4o-mini") == 1
        assert tracker.total_tokens > 0
