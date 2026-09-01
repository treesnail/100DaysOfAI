"""评估指标：记录每次 Agent 运行的数据，聚合成功率/步数/延迟."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunRecord:
    """一次 Agent 运行的追踪数据."""

    success: bool
    steps: int
    latency_seconds: float
    tool_calls: int = 0


class MetricsTracker:
    """指标聚合器：累计 RunRecord，计算汇总统计并导出报告."""

    def __init__(self) -> None:
        self._records: list[RunRecord] = []

    def record(self, record: RunRecord) -> None:
        """登记一次运行记录."""
        self._records.append(record)

    def record_run(
        self,
        success: bool,
        steps: int,
        latency_seconds: float,
        tool_calls: int = 0,
    ) -> RunRecord:
        """便捷方法：直接以字段登记一次运行."""
        record = RunRecord(
            success=success,
            steps=steps,
            latency_seconds=latency_seconds,
            tool_calls=tool_calls,
        )
        self.record(record)
        return record

    @property
    def total_runs(self) -> int:
        return len(self._records)

    @property
    def success_rate(self) -> float:
        """成功率：成功次数 / 总次数；无记录时为 0.0."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.success) / len(self._records)

    @property
    def avg_steps(self) -> float:
        """平均步数；无记录时为 0.0."""
        if not self._records:
            return 0.0
        return sum(r.steps for r in self._records) / len(self._records)

    @property
    def avg_latency(self) -> float:
        """平均延迟（秒）；无记录时为 0.0."""
        if not self._records:
            return 0.0
        return sum(r.latency_seconds for r in self._records) / len(self._records)

    @property
    def avg_tool_calls(self) -> float:
        """平均每次运行的工具调用次数；无记录时为 0.0."""
        if not self._records:
            return 0.0
        return sum(r.tool_calls for r in self._records) / len(self._records)

    def report(self) -> dict:
        """导出聚合报告（可直接 json.dumps 或打印）."""
        return {
            "total_runs": self.total_runs,
            "success_rate": round(self.success_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_latency_seconds": round(self.avg_latency, 4),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
        }
