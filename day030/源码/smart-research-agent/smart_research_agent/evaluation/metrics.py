"""评估指标合并基准版：day015 的运行追踪聚合 + day030 的纯数据指标函数.

两部分无命名冲突，直接并集：
- ``RunRecord`` / ``MetricsTracker``（day015）：记录每次 Agent 运行的数据，
  聚合成功率/步数/延迟/工具调用次数；
- ``accuracy`` / ``contains_rate`` / ``mean``（day030）：操作纯数据
  （参考答案与预测结果的列表）的通用指标函数，与被测系统解耦。
"""

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


# ---------------------------------------------------------------------------
# day030：通用评估指标函数（操作纯数据，与被测系统解耦）
# ---------------------------------------------------------------------------


def accuracy(references: list[str], predictions: list[str]) -> float:
    """完全匹配准确率：预测与参考逐条相等（忽略首尾空白）的比例."""
    if len(references) != len(predictions):
        raise ValueError("references 与 predictions 长度必须一致")
    if not references:
        return 0.0
    hits = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return hits / len(references)


def contains_rate(references: list[str], predictions: list[str]) -> float:
    """包含率：参考答案作为子串出现在预测中的比例（宽松匹配）."""
    if len(references) != len(predictions):
        raise ValueError("references 与 predictions 长度必须一致")
    if not references:
        return 0.0
    hits = sum(1 for ref, pred in zip(references, predictions) if ref.strip() in pred)
    return hits / len(references)


def mean(values: list[float]) -> float:
    """算术平均值，空列表返回 0.0 而不是抛异常（评估时"没有数据"应得 0 分而非崩溃）."""
    if not values:
        return 0.0
    return sum(values) / len(values)
