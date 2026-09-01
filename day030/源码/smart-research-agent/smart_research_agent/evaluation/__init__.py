"""评估框架：EvaluationHarness 基类、数据模型、Agent 评估实现、输出质量评估与运行指标聚合."""

from __future__ import annotations

from smart_research_agent.evaluation.agent_harness import AgentEvalHarness
from smart_research_agent.evaluation.harness import (
    CaseResult,
    EvalCase,
    EvaluationHarness,
)
from smart_research_agent.evaluation.metrics import MetricsTracker, RunRecord
from smart_research_agent.evaluation.output_eval import OutputEvaluator, OutputScore
from smart_research_agent.evaluation.quality_logger import OutputQualityLogger

__all__ = [
    "AgentEvalHarness",
    "CaseResult",
    "EvalCase",
    "EvaluationHarness",
    "MetricsTracker",
    "OutputEvaluator",
    "OutputQualityLogger",
    "OutputScore",
    "RunRecord",
]
