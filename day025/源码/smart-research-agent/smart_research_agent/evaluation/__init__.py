"""评估框架：EvaluationHarness 基类、数据模型与 Agent 评估实现."""

from __future__ import annotations

from smart_research_agent.evaluation.agent_harness import AgentEvalHarness
from smart_research_agent.evaluation.harness import (
    CaseResult,
    EvalCase,
    EvaluationHarness,
)

__all__ = [
    "AgentEvalHarness",
    "CaseResult",
    "EvalCase",
    "EvaluationHarness",
]
