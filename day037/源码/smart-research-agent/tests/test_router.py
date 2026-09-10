"""day033 多模型路由测试：复杂度估计、策略选择、降级链与决策可观测."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.router import (
    CostFirstStrategy,
    ModelRouter,
    ModelSpec,
    QualityFirstStrategy,
    estimate_complexity,
)


class FailingLLM(BaseLLM):
    """总是抛异常的 LLM，用于测试降级链."""

    def __init__(self, error: str = "connection refused"):
        self.error = error
        self.calls: list = []

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        self.calls.append(messages)
        raise ConnectionError(self.error)


def make_models() -> list[ModelSpec]:
    """三档模型：便宜弱 / 中等 / 贵强."""
    return [
        ModelSpec("cheap", MockLLM(default="便宜模型的回答"), capability=2, cost_per_1k=0.001),
        ModelSpec("mid", MockLLM(default="中档模型的回答"), capability=3, cost_per_1k=0.01),
        ModelSpec("strong", MockLLM(default="强模型的回答"), capability=5, cost_per_1k=0.1),
    ]


class TestEstimateComplexity:
    def test_short_simple_task(self):
        assert estimate_complexity("2+3 等于几") == 1

    def test_complex_keywords_raise_score(self):
        assert estimate_complexity("分析 Transformer 架构") >= 3
        assert estimate_complexity("对比并推理两种方案的优劣") >= 4

    def test_long_task_raises_score(self):
        assert estimate_complexity("查一下" + "某资料" * 30) >= 2

    def test_capped_at_five(self):
        task = "分析对比推理证明设计评估综述为什么推导" + "长文" * 200
        assert estimate_complexity(task) == 5


class TestCostFirstStrategy:
    def test_simple_task_routes_to_cheapest_capable(self):
        router = ModelRouter(make_models())
        decision, chain = router.route("2+3 等于几")
        assert decision.chosen == "cheap"
        assert "最便宜" in decision.reason

    def test_complex_task_routes_to_strong(self):
        router = ModelRouter(make_models())
        decision, _ = router.route("请深入分析和对比 RAG 与微调的取舍，给出推理过程")
        assert decision.chosen == "strong"
        assert decision.complexity >= 4

    def test_unavailable_model_skipped(self):
        models = make_models()
        models[0].available = False  # cheap 下线
        router = ModelRouter(models)
        decision, _ = router.route("2+3 等于几")
        assert decision.chosen == "mid"

    def test_no_capable_model_uses_strongest_with_reason(self):
        models = make_models()
        for m in models:
            m.capability = 2  # 所有模型能力上限 2
        router = ModelRouter(models)
        decision, _ = router.route("请深入分析和对比 RAG 与微调的取舍，给出推理过程")
        assert decision.chosen in {"cheap", "mid", "strong"}
        assert "超出所有可用模型能力上限" in decision.reason

    def test_no_available_model_raises(self):
        models = make_models()
        for m in models:
            m.available = False
        router = ModelRouter(models)
        with pytest.raises(RuntimeError, match="没有可用模型"):
            router.route("任意任务")

    def test_fallback_chain_ordered_by_capability(self):
        router = ModelRouter(make_models())
        decision, chain = router.route("2+3 等于几")
        assert chain[0].name == "cheap"
        # 兜底方向向上：强模型在前
        assert [m.name for m in chain[1:]] == ["strong", "mid"]
        assert decision.fallbacks == ["strong", "mid"]


class TestQualityFirstStrategy:
    def test_always_strongest(self):
        router = ModelRouter(make_models(), strategy=QualityFirstStrategy())
        decision, _ = router.route("2+3 等于几")
        assert decision.chosen == "strong"


class TestRouterChat:
    def test_chat_returns_primary_reply(self):
        router = ModelRouter(make_models())
        reply = router.chat([Message(role="user", content="2+3 等于几")])
        assert reply == "便宜模型的回答"
        assert router.decision_log[-1].fallback_used is None

    def test_fallback_on_failure(self):
        models = make_models()
        models[0] = ModelSpec("cheap", FailingLLM(), capability=2, cost_per_1k=0.001)
        router = ModelRouter(models)
        reply = router.chat([Message(role="user", content="2+3 等于几")])
        # cheap 失败，降级链按能力从强到弱：strong 先生效
        assert reply == "强模型的回答"
        decision = router.decision_log[-1]
        assert decision.chosen == "cheap"
        assert decision.fallback_used == "strong"

    def test_all_fail_raises_with_details(self):
        models = [
            ModelSpec("a", FailingLLM("超时"), capability=2, cost_per_1k=0.001),
            ModelSpec("b", FailingLLM("限流"), capability=3, cost_per_1k=0.01),
        ]
        router = ModelRouter(models)
        with pytest.raises(RuntimeError, match="所有候选模型均调用失败") as exc_info:
            router.chat([Message(role="user", content="hi")])
        assert "超时" in str(exc_info.value) and "限流" in str(exc_info.value)

    def test_decision_log_accumulates(self):
        router = ModelRouter(make_models())
        router.chat([Message(role="user", content="2+3 等于几")])
        router.chat([Message(role="user", content="分析对比两种方案并给出推理")])
        assert len(router.decision_log) == 2
        assert router.decision_log[0].chosen == "cheap"
        assert router.decision_log[1].chosen != "cheap"

    def test_router_is_base_llm_for_agent(self):
        """ModelRouter 实现 BaseLLM，可直接装配进 ReactAgent."""
        from smart_research_agent.agent.react_agent import ReactAgent
        from smart_research_agent.tools.calculator import CalculatorTool
        from smart_research_agent.tools.registry import ToolRegistry

        models = [
            ModelSpec(
                "cheap",
                MockLLM(responses=["Thought: 直接答\nFinal Answer: 5"]),
                capability=2,
                cost_per_1k=0.001,
            )
        ]
        router = ModelRouter(models)
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        agent = ReactAgent(llm=router, registry=registry)
        assert agent.run("2+3 等于几") == "5"
        assert router.decision_log[-1].chosen == "cheap"

    def test_empty_models_rejected(self):
        with pytest.raises(ValueError, match="至少需要一个模型"):
            ModelRouter([])
