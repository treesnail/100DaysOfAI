"""多模型路由：按能力/成本/可用性选择模型，失败自动降级，决策全程可观测（day033）.

路由要解决的核心矛盾：强模型贵、弱模型笨。把所有请求都发给最强模型
是最贵的偷懒；都发最弱模型则复杂任务翻车。``ModelRouter`` 用一个
可插拔的策略（``RoutingStrategy``）在每次调用前回答"这个任务值得
用多强的模型"，并把答案连同理由记入 ``decision_log`` 供事后审计。

``ModelRouter`` 本身实现 ``BaseLLM`` 接口，可直接替换 ReactAgent 里
的任何单一模型——对 Agent 而言，路由层是透明的。

day039 起补齐两个能力：
  - ``stream`` / ``chat_with_tools``：把流式与 function calling 也纳入路由，
    使路由器成为 BaseLLM 的完整替身，无论走哪条协议都能透明路由；
  - ``get_model``：按名查模型，供 API 层把 ``model`` 参数解析为具体客户端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message

#: 复杂任务的标志性动词（启发式，可随业务扩充）
COMPLEX_KEYWORDS = ("分析", "对比", "比较", "推理", "证明", "设计", "评估", "综述", "为什么", "推导")


def estimate_complexity(task: str) -> int:
    """估计任务复杂度（1~5）：基于文本长度与复杂关键词的确定性启发式.

    简单任务（查个事实、算个数）→ 1~2；含一个复杂动词 → 3；
    多步分析/推理/设计 → 4~5。
    真实系统可换成分类模型或规则组合；启发式的价值是零成本、可解释、
    在测试里完全确定。
    """
    score = 1
    if len(task) > 50:
        score += 1
    if len(task) > 200:
        score += 1
    hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in task)
    score += min(2 * hits, 3)
    return min(score, 5)


@dataclass
class ModelSpec:
    """一个可选模型的档案：能力档位、综合成本与可用性."""

    name: str
    llm: BaseLLM
    capability: int  # 1~5，越大越强
    cost_per_1k: float  # 综合成本（美元/1K token，输入输出加权后的单价）
    available: bool = True  # 可用性（熔断/下线时置 False，路由自动绕开）


@dataclass
class RouteDecision:
    """一次路由决策的完整记录：选了谁、为什么、降级链是什么."""

    task: str
    complexity: int
    chosen: str
    reason: str
    fallbacks: list[str] = field(default_factory=list)
    fallback_used: str | None = None  # 实际发生降级时，记录最终生效的模型


class RoutingStrategy(ABC):
    """路由策略接口：给定可用模型与任务复杂度，返回有序的选择链（首选在前）."""

    @abstractmethod
    def choose(self, models: list[ModelSpec], complexity: int) -> tuple[list[ModelSpec], str]:
        """返回 (有序模型链, 人类可读的决策理由)."""


class CostFirstStrategy(RoutingStrategy):
    """成本优先：首选"满足能力要求的最便宜模型"，降级链按能力从强到弱.

    典型的"简单任务走便宜模型、复杂任务走强模型"规则路由：
    复杂度 2 的任务不会浪费 capability=5 的贵模型；但兜底方向永远是
    向上（更强）而非向下——降级到更弱模型可能答错，降级到更强模型
    只是多花钱，失败安全的方向是多花钱。
    """

    def choose(self, models: list[ModelSpec], complexity: int) -> tuple[list[ModelSpec], str]:
        available = [m for m in models if m.available]
        if not available:
            raise RuntimeError("没有可用模型，无法路由")
        capable = [m for m in available if m.capability >= complexity]
        if capable:
            primary = min(capable, key=lambda m: m.cost_per_1k)
            reason = (
                f"任务复杂度 {complexity}，选择满足能力要求（capability>={complexity}）"
                f"的最便宜模型 {primary.name}（capability={primary.capability}，"
                f"${primary.cost_per_1k}/1K）"
            )
        else:
            primary = max(available, key=lambda m: m.capability)
            reason = (
                f"任务复杂度 {complexity} 超出所有可用模型能力上限，"
                f"降级使用最强可用模型 {primary.name}（capability={primary.capability}）"
            )
        rest = [m for m in available if m is not primary]
        rest.sort(key=lambda m: m.capability, reverse=True)
        return [primary, *rest], reason


class QualityFirstStrategy(RoutingStrategy):
    """质量优先：无视成本永远选最强可用模型（对照组，用于度量成本策略的收益）."""

    def choose(self, models: list[ModelSpec], complexity: int) -> tuple[list[ModelSpec], str]:
        available = [m for m in models if m.available]
        if not available:
            raise RuntimeError("没有可用模型，无法路由")
        chain = sorted(available, key=lambda m: m.capability, reverse=True)
        reason = (
            f"质量优先策略：选择最强可用模型 {chain[0].name}"
            f"（capability={chain[0].capability}），不考虑成本"
        )
        return chain, reason


class ModelRouter(BaseLLM):
    """多模型路由器：每次调用先路由再调用，失败沿降级链自动切换.

    可观测性：每次决策（含理由与降级链）追加到 ``decision_log``；
    发生降级时 ``fallback_used`` 记录最终生效的模型——"为什么选了它"
    永远有案可查。
    """

    def __init__(self, models: list[ModelSpec], strategy: RoutingStrategy | None = None):
        if not models:
            raise ValueError("至少需要一个模型")
        self.models = list(models)
        self.strategy = strategy or CostFirstStrategy()
        self.decision_log: list[RouteDecision] = []

    def route(self, task: str) -> tuple[RouteDecision, list[ModelSpec]]:
        """做一次路由决策：估计复杂度 → 策略选择链 → 记录决策."""
        complexity = estimate_complexity(task)
        chain, reason = self.strategy.choose(self.models, complexity)
        decision = RouteDecision(
            task=task,
            complexity=complexity,
            chosen=chain[0].name,
            reason=reason,
            fallbacks=[m.name for m in chain[1:]],
        )
        self.decision_log.append(decision)
        return decision, chain

    @staticmethod
    def _last_user_task(messages: list[Message]) -> str:
        """取最近一条 user 消息作为路由决策的输入（估复杂度的依据）."""
        return next((m.content for m in reversed(messages) if m.role == "user"), "")

    def get_model(self, name: str) -> BaseLLM | None:
        """按名查模型：API 层的显式模型切换（``model`` 参数）用它解析目标客户端."""
        for spec in self.models:
            if spec.name == name:
                return spec.llm
        return None

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """按最近一条 user 消息路由；首选模型异常时沿降级链依次尝试."""
        decision, chain = self.route(self._last_user_task(messages))
        errors: list[str] = []
        for spec in chain:
            try:
                reply = spec.llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{spec.name}: {exc}")
                continue
            if spec.name != decision.chosen:
                decision.fallback_used = spec.name
            return reply
        raise RuntimeError(f"所有候选模型均调用失败: {'; '.join(errors)}")

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式路由：先路由，再逐片段转发首选模型的流式输出（day039）.

        与 chat 相同的降级语义：首选模型在流式过程中抛异常（含首片未产出
        即失败，因为 BaseLLM.stream 的默认实现是惰性的）时，沿降级链换下一个
        模型重新流式；成功产出处理完毕即返回。降级事实记入 decision。
        """
        decision, chain = self.route(self._last_user_task(messages))
        errors: list[str] = []
        for spec in chain:
            try:
                for chunk in spec.llm.stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    if spec.name != decision.chosen:
                        decision.fallback_used = spec.name
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{spec.name}: {exc}")
                continue
        raise RuntimeError(f"所有候选模型均调用失败: {'; '.join(errors)}")

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """工具调用路由：把 function calling 也纳入路由（day039）.

        与 chat 对称：按最近一条 user 消息路由，沿降级链尝试 chat_with_tools。
        Agent 无论走文本协议还是 function calling，都能透明享受多模型路由。
        """
        decision, chain = self.route(self._last_user_task(messages))
        errors: list[str] = []
        for spec in chain:
            try:
                reply = spec.llm.chat_with_tools(
                    messages, tools=tools, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{spec.name}: {exc}")
                continue
            if spec.name != decision.chosen:
                decision.fallback_used = spec.name
            return reply
        raise RuntimeError(f"所有候选模型均调用失败: {'; '.join(errors)}")