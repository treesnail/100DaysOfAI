"""可观测性：成本追踪、轻量 trace 与告警判定（day032）.

三个子模块分别对应可观测性的三个支柱在本项目中的落地：
- ``cost_tracker``：按模型/按接口归因 token 用量与费用（对应"指标"）；
- ``tracing``：每次 Agent 运行生成 trace_id，记录各步骤 span 到 jsonl（对应"追踪"）；
- ``alerting``：评估指标低于阈值时产出告警（把"指标"接入 CI 门禁）。
"""

from smart_research_agent.observability.alerting import Alert, check_thresholds, render_alerts
from smart_research_agent.observability.cost_tracker import CostTracker, UsageRecord
from smart_research_agent.observability.tracing import Span, Tracer, run_with_trace

__all__ = [
    "Alert",
    "CostTracker",
    "Span",
    "Tracer",
    "UsageRecord",
    "check_thresholds",
    "render_alerts",
    "run_with_trace",
]
