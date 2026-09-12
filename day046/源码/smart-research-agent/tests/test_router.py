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


class FailingVisionLLM(BaseLLM):
    """声明支持视觉、但调用总失败，用于测试视觉降级链."""

    supports_vision = True  # 类属性遮蔽父类 property

    def __init__(self, error: str = "vision api down"):
        self.error = error

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        raise ConnectionError(self.error)

    def chat_vision(
        self, messages, image_data_url, temperature: float = 0.7, max_tokens: int = 1024
    ) -> str:
        raise ConnectionError(self.error)


class TextOnlyLLM(BaseLLM):
    """纯文本模型：实现 chat，但不声明 supports_vision（默认 False）."""

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        return "文本模型的回答"


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


class TestRouterStream:
    """day039：流式路由——stream 也纳入透明路由与降级."""

    def test_stream_routes_to_cheapest_and_reassembles(self):
        router = ModelRouter(make_models())
        chunks = list(router.stream([Message(role="user", content="2+3 等于几")]))
        assert "".join(chunks) == "便宜模型的回答"
        assert router.decision_log[-1].fallback_used is None

    def test_stream_fallback_on_failure(self):
        models = make_models()
        models[0] = ModelSpec("cheap", FailingLLM(), capability=2, cost_per_1k=0.001)
        router = ModelRouter(models)
        chunks = list(router.stream([Message(role="user", content="2+3 等于几")]))
        assert "".join(chunks) == "强模型的回答"
        decision = router.decision_log[-1]
        assert decision.chosen == "cheap"
        assert decision.fallback_used == "strong"

    def test_stream_reassembled_equals_chat(self):
        """同一路由决策下，流式拼接 == 一次性回复."""
        router = ModelRouter(make_models())
        one_shot = router.chat([Message(role="user", content="2+3 等于几")])
        streamed = "".join(
            router.stream([Message(role="user", content="2+3 等于几")])
        )
        assert streamed == one_shot


class TestRouterChatWithTools:
    """day039：function calling 也纳入路由."""

    def test_chat_with_tools_delegates_and_returns_normalized(self):
        mock = MockLLM(tool_call_responses=[{"content": "直接回答"}])
        router = ModelRouter([ModelSpec("m", mock, capability=3, cost_per_1k=0.01)])
        resp = router.chat_with_tools(
            [Message(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "calculator"}}],
        )
        assert resp == {"content": "直接回答"}
        assert router.decision_log[-1].chosen == "m"

    def test_chat_with_tools_routes_tool_call(self):
        mock = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]}
            ]
        )
        router = ModelRouter([ModelSpec("m", mock, capability=3, cost_per_1k=0.01)])
        resp = router.chat_with_tools(
            [Message(role="user", content="1+1 等于几")], tools=[]
        )
        assert resp["tool_calls"][0]["function"]["name"] == "calculator"


class TestGetModel:
    """day039：按名查模型，供 API 层 ``model`` 参数解析."""

    def test_get_model_returns_spec_llm(self):
        models = make_models()
        router = ModelRouter(models)
        assert router.get_model("strong") is models[2].llm
        assert router.get_model("mid") is models[1].llm

    def test_get_model_unknown_returns_none(self):
        assert ModelRouter(make_models()).get_model("nope") is None


class TestRouterObservability:
    """day046：``last_used_llm`` —— 计费要落到真正干活的叶子模型.

    背景：``CostTracker.record_from_llm`` 按 ``llm.usage_log`` 归集用量。
    路由器是包装器、不持有用量，若上层把路由器直接交给计费器就会断链。
    因此路由器必须能回答"谁真的花钱了"，且四条调用路径（chat / stream /
    function calling / vision）都要如实记录。
    """

    def test_no_call_yet_returns_none(self):
        assert ModelRouter(make_models()).last_used_llm is None

    def test_chat_records_leaf(self):
        models = make_models()
        router = ModelRouter(models)
        router.chat([Message(role="user", content="2+3 等于几")])
        assert router.last_used_llm is models[0].llm

    def test_stream_records_leaf(self):
        models = make_models()
        router = ModelRouter(models)
        list(router.stream([Message(role="user", content="2+3 等于几")]))
        assert router.last_used_llm is models[0].llm

    def test_chat_with_tools_records_leaf(self):
        models = make_models()
        router = ModelRouter(models)
        router.chat_with_tools([Message(role="user", content="2+3 等于几")], tools=[])
        assert router.last_used_llm is models[0].llm

    def test_chat_vision_records_vision_leaf(self):
        router = ModelRouter(
            [
                ModelSpec("text", TextOnlyLLM(), capability=2, cost_per_1k=0.001),
                ModelSpec(
                    "vision", MockLLM(default="视觉回复"), capability=4, cost_per_1k=0.05
                ),
            ]
        )
        router.chat_vision(
            [Message(role="user", content="描述图片")],
            image_data_url="data:image/png;base64,AAAA",
        )
        assert router.last_used_llm is router.get_model("vision")

    def test_fallback_records_the_model_that_actually_answered(self):
        """降级发生时，记的是**真正回答**的那个模型（而不是首选）.

        注意降级方向：``CostFirstStrategy`` 把首选之后的候选按能力**从强到弱**
        排列，即"降级往上"——多花钱总比答错安全。所以便宜模型失败后落到的是
        最强的 strong，而不是中间的 mid。
        """
        models = make_models()
        models[0] = ModelSpec("cheap", FailingLLM(), capability=2, cost_per_1k=0.001)
        router = ModelRouter(models)
        router.chat([Message(role="user", content="2+3 等于几")])
        assert router.decision_log[-1].fallback_used == "strong"
        assert router.last_used_llm is models[2].llm

    def test_leaf_tracks_the_most_recent_call(self):
        models = make_models()
        router = ModelRouter(models)
        router.chat([Message(role="user", content="2+3 等于几")])  # 复杂度低 -> cheap
        assert router.last_used_llm is models[0].llm
        router.chat([Message(role="user", content="请深入分析并对比两种方案")])  # -> strong
        assert router.last_used_llm is models[2].llm

    def test_all_candidates_failing_keeps_no_leaf(self):
        """全部候选失败时不产生"用过的模型"（失败者不应被计入成本归因）."""
        router = ModelRouter(
            [
                ModelSpec("cheap", FailingLLM(), capability=2, cost_per_1k=0.001),
                ModelSpec("mid", FailingLLM(), capability=3, cost_per_1k=0.01),
                ModelSpec("strong", FailingLLM(), capability=5, cost_per_1k=0.1),
            ]
        )
        with pytest.raises(RuntimeError, match="所有候选模型均调用失败"):
            router.chat([Message(role="user", content="2+3 等于几")])
        assert router.last_used_llm is None


class TestRouterVision:
    """day040：视觉请求路由——图像请求只发给支持视觉的候选."""

    def make_mixed_models(self) -> list[ModelSpec]:
        """文本模型更便宜（会被成本策略首选），但视觉链必须跳过它."""
        return [
            ModelSpec("text", TextOnlyLLM(), capability=2, cost_per_1k=0.001),
            ModelSpec("vision", MockLLM(default="视觉模型的回答"), capability=4, cost_per_1k=0.05),
        ]

    def test_supports_vision_true_when_any_candidate_has_vision(self):
        assert ModelRouter(self.make_mixed_models()).supports_vision is True

    def test_supports_vision_false_when_none(self):
        models = [
            ModelSpec("a", TextOnlyLLM(), capability=2, cost_per_1k=0.001),
            ModelSpec("b", TextOnlyLLM(), capability=3, cost_per_1k=0.01),
        ]
        assert ModelRouter(models).supports_vision is False

    def test_chat_vision_skips_text_models(self):
        """路由决策首选便宜的文本模型，但视觉链跳过它、落到视觉模型."""
        router = ModelRouter(self.make_mixed_models())
        reply = router.chat_vision(
            [Message(role="user", content="图片里有什么")],
            image_data_url="data:image/png;base64,AAAA",
        )
        assert reply == "视觉模型的回答"
        decision = router.decision_log[-1]
        assert decision.chosen == "text"  # 成本策略仍选了文本模型
        assert decision.fallback_used == "vision"  # 视觉链实际落到 vision

    def test_chat_vision_uses_vision_model_directly(self):
        router = ModelRouter(
            [ModelSpec("v", MockLLM(default="视觉回复"), capability=3, cost_per_1k=0.01)]
        )
        reply = router.chat_vision(
            [Message(role="user", content="描述图片")],
            image_data_url="data:image/png;base64,AAAA",
        )
        assert reply == "视觉回复"
        assert router.decision_log[-1].fallback_used is None

    def test_chat_vision_all_text_models_raises(self):
        router = ModelRouter(
            [
                ModelSpec("a", TextOnlyLLM(), capability=2, cost_per_1k=0.001),
                ModelSpec("b", TextOnlyLLM(), capability=3, cost_per_1k=0.01),
            ]
        )
        with pytest.raises(RuntimeError, match="候选模型均不支持视觉"):
            router.chat_vision(
                [Message(role="user", content="描述图片")],
                image_data_url="data:image/png;base64,AAAA",
            )

    def test_chat_vision_fallback_on_vision_failure(self):
        """视觉首选失败时沿降级链找下一个视觉候选."""
        router = ModelRouter(
            [
                ModelSpec("v1", FailingVisionLLM("visual api 500"), capability=4, cost_per_1k=0.01),
                ModelSpec("v2", MockLLM(default="备用视觉回复"), capability=3, cost_per_1k=0.02),
            ]
        )
        reply = router.chat_vision(
            [Message(role="user", content="图片里有什么")],
            image_data_url="data:image/png;base64,AAAA",
        )
        assert reply == "备用视觉回复"
        decision = router.decision_log[-1]
        assert decision.chosen == "v1"
        assert decision.fallback_used == "v2"

    def test_chat_vision_all_vision_fail_raises_details(self):
        router = ModelRouter(
            [
                ModelSpec("v1", FailingVisionLLM("超时"), capability=4, cost_per_1k=0.05),
                ModelSpec("v2", FailingVisionLLM("限流"), capability=3, cost_per_1k=0.02),
            ]
        )
        with pytest.raises(RuntimeError, match="所有视觉模型均调用失败") as exc_info:
            router.chat_vision(
                [Message(role="user", content="描述图片")],
                image_data_url="data:image/png;base64,AAAA",
            )
        assert "超时" in str(exc_info.value) and "限流" in str(exc_info.value)
