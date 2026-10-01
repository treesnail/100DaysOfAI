"""M4 集成层（day046）：把分散的能力装配成一条可运行的调用链."""

from __future__ import annotations

from smart_research_agent.integration.pipeline import (
    CACHE_MODEL_LABEL,
    REFUSAL_REPLY,
    STAGE_ORDER,
    IntegratedPipeline,
    PipelineResult,
    StageRecord,
    build_hybrid_router,
    default_pipeline,
)

__all__ = [
    "CACHE_MODEL_LABEL",
    "REFUSAL_REPLY",
    "STAGE_ORDER",
    "IntegratedPipeline",
    "PipelineResult",
    "StageRecord",
    "build_hybrid_router",
    "default_pipeline",
]
