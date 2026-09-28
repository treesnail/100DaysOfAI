"""AgentEvalHarness：把 ReactAgent 包装进评估框架的具体实现."""

from __future__ import annotations

import time
from collections.abc import Callable

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation.harness import (
    CaseResult,
    EvalCase,
    EvaluationHarness,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

Scorer = Callable[[str, str], float]
"""打分函数签名：(实际输出, 期望输出) -> [0, 1] 之间的分数."""


def exact_match_scorer(output: str, expected: str) -> float:
    """精确匹配打分：完全一致得 1 分，否则 0 分."""
    return 1.0 if output.strip() == expected.strip() else 0.0


def contains_scorer(output: str, expected: str) -> float:
    """包含匹配打分：输出中包含期望内容得 1 分，否则 0 分."""
    return 1.0 if expected.strip() in output else 0.0


class AgentEvalHarness(EvaluationHarness):
    """面向 ReactAgent 的评估实现.

    agent_factory 是"按用例构造一个全新 Agent"的工厂函数，
    保证每条用例的 history 相互隔离（也便于 MockLLM 按用例注入脚本）。
    scorer 决定如何把 Agent 输出与期望答案对比打分，默认包含匹配。
    """

    def __init__(
        self,
        agent_factory: Callable[[EvalCase], ReactAgent],
        scorer: Scorer = contains_scorer,
        pass_threshold: float = 1.0,
    ) -> None:
        super().__init__(pass_threshold=pass_threshold)
        self.agent_factory = agent_factory
        self.scorer = scorer

    def evaluate(self, case: EvalCase) -> CaseResult:
        """运行 Agent 并对输出打分，记录延迟."""
        agent = self.agent_factory(case)
        start = time.perf_counter()
        output = agent.run(case.input)
        latency = time.perf_counter() - start

        score = self.scorer(output, case.expected)
        return CaseResult(
            case_id=case.id,
            output=output,
            score=score,
            passed=score >= self.pass_threshold,
            latency_seconds=latency,
            details={"expected": case.expected},
        )
