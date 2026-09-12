"""性能基线（day046）：把"变慢了"从主观感受变成可判定的工程事实.

集成之后必然要问的问题不是"跑得通吗"（测试已回答），而是"**比昨天慢了吗、
贵了吗**"。要回答它，需要三样东西：

1. **可复现的度量**（``measure``）：在一组固定任务上跑流水线，采集每个阶段的
   耗时、token 数与费用。指标定义必须固定，否则数字之间不可比；
2. **一份存档**（``PerfBaseline.save/load``）：基线写进 JSON 文件随仓库提交，
   它就是这次迭代的"体检报告底片"；
3. **一条判定规则**（``PerformanceGuard``）：当前值 / 基线值 超过容忍倍数即
   判为回归。

关于百分位数的取法：这里用**最近秩法**（nearest-rank）而不是线性插值。
原因是样本量小（一次评测往往十几条任务），插值会凭空捏造出样本里并不存在
的耗时值；最近秩法永远返回一个真实观测值，更适合"小样本 + 人工复核"的场景。

关于**判定下限**（``min_latency_ms`` / ``min_tokens``）：微秒级的基线做比值
毫无意义——1ms 变成 2ms 是 2 倍回归，但对用户完全无感，而且会被调度抖动
随机触发。所以只有当基线值本身超过下限时才参与比值判定；这叫"防抖动阈值"，
是性能门禁能不能长期用下去的关键：一个天天误报的门禁，最后一定会被人关掉。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.integration.pipeline import IntegratedPipeline
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 总耗时在 ``latency_ms`` 里的键名（与流水线阶段名同处一个字典）
TOTAL_KEY = "total"

#: 默认判定下限：低于这些值的基线不参与比值判定（防抖动）
DEFAULT_MIN_LATENCY_MS = 5.0
DEFAULT_MIN_TOKENS = 1.0


def percentile(values: list[float], q: float) -> float:
    """最近秩法百分位数：返回样本中真实存在的那个值（空列表返回 0.0）.

    ``q`` 取 0~100。实现刻意不做插值：小样本下插值会造出"观测不到的耗时"，
    而性能门禁的结论必须能回溯到某一次真实请求。``q=0`` 取最小值、
    ``q=100`` 取最大值，边界行为与直觉一致。
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError("q 必须在 0~100 之间")
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered)) - 1
    index = max(0, min(rank, len(ordered) - 1))
    return ordered[index]


@dataclass
class PerfBaseline:
    """一次性能采集的存档：阶段耗时 + token + 费用.

    ``latency_ms`` 与 ``p95_latency_ms`` 都含 ``"total"`` 键（整条流水线的
    总耗时），其余键为阶段名——阶段与总览同表存放，比较时不必写两套逻辑。
    """

    label: str
    samples: int
    latency_ms: dict[str, float] = field(default_factory=dict)
    p95_latency_ms: dict[str, float] = field(default_factory=dict)
    mean_tokens: float = 0.0
    mean_cost_usd: float = 0.0
    mean_stages: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "samples": self.samples,
            "latency_ms": {k: round(v, 4) for k, v in self.latency_ms.items()},
            "p95_latency_ms": {k: round(v, 4) for k, v in self.p95_latency_ms.items()},
            "mean_tokens": round(self.mean_tokens, 4),
            "mean_cost_usd": round(self.mean_cost_usd, 8),
            "mean_stages": round(self.mean_stages, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerfBaseline:
        """从字典还原（``save`` 的逆操作，用于读取已提交的基线文件）."""
        return cls(
            label=str(data.get("label", "pipeline")),
            samples=int(data.get("samples", 0)),
            latency_ms={
                str(k): float(v) for k, v in (data.get("latency_ms") or {}).items()
            },
            p95_latency_ms={
                str(k): float(v) for k, v in (data.get("p95_latency_ms") or {}).items()
            },
            mean_tokens=float(data.get("mean_tokens", 0.0)),
            mean_cost_usd=float(data.get("mean_cost_usd", 0.0)),
            mean_stages=float(data.get("mean_stages", 0.0)),
        )

    def save(self, path: str | Path) -> Path:
        """写入 JSON（自动创建父目录），返回实际写入的路径."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("性能基线已保存: %s（%d 个样本）", target, self.samples)
        return target

    @classmethod
    def load(cls, path: str | Path) -> PerfBaseline:
        """读取基线文件；文件不存在时抛 ``FileNotFoundError``.

        刻意不返回"空基线"兜底：**没有基线**与"基线全为零"是两回事，
        前者应当明确报错（提示先跑一次采集），后者会静默地把所有回归
        判定都跳过——那是最危险的失败方式。
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"性能基线文件不存在: {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


def measure(
    pipeline: IntegratedPipeline,
    tasks: list[str],
    *,
    label: str = "pipeline",
    system_prompt: str | None = None,
) -> PerfBaseline:
    """在一组任务上运行流水线并采集基线（空任务集返回全零基线）.

    耗时取流水线自报的 ``total_ms`` 与各 ``StageRecord.duration_ms``——
    复用同一次运行的数据，而不是在外部再包一层计时，避免"测到的耗时"
    与"响应里报告的耗时"对不上。

    token 与费用的**唯一来源是成本归集阶段**（它读 LLM 的 ``usage_log``）：
    流水线未装配 ``CostTracker`` 时这两个指标恒为 0。这是刻意的口径声明——
    没有账本就没有可核对的用量，绝不用"字符数除以 4"之类的估算顶替，
    否则基线与真实账单会在同一张表里混成两种口径。
    """
    total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    tokens: list[float] = []
    costs: list[float] = []
    stage_counts: list[float] = []

    for task in tasks:
        result = pipeline.run(task, system_prompt=system_prompt)
        total_ms.append(result.total_ms)
        tokens.append(float(result.total_tokens))
        costs.append(result.cost_usd)
        stage_counts.append(float(len(result.stages)))
        for stage in result.stages:
            stage_ms.setdefault(stage.name, []).append(stage.duration_ms)

    latency_ms = {TOTAL_KEY: _mean(total_ms)}
    latency_ms.update({name: _mean(values) for name, values in stage_ms.items()})
    p95 = {TOTAL_KEY: percentile(total_ms, 95)}
    p95.update({name: percentile(values, 95) for name, values in stage_ms.items()})

    baseline = PerfBaseline(
        label=label,
        samples=len(tasks),
        latency_ms={k: round(v, 4) for k, v in latency_ms.items()},
        p95_latency_ms={k: round(v, 4) for k, v in p95.items()},
        mean_tokens=round(_mean(tokens), 4),
        mean_cost_usd=round(_mean(costs), 8),
        mean_stages=round(_mean(stage_counts), 4),
    )
    logger.info("性能采集完成: %s（%d 个样本）", label, baseline.samples)
    return baseline


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class Regression:
    """一条被判定为回归的指标."""

    metric: str
    baseline: float
    current: float
    ratio: float
    threshold: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "ratio": self.ratio,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class BaselineComparison:
    """当前采集 vs 参考基线的对比结论."""

    baseline: PerfBaseline
    current: PerfBaseline
    regressions: list[Regression] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无回归即通过（可作为 CI 的断言入口）."""
        return not self.regressions

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "baseline_label": self.baseline.label,
            "current_label": self.current.label,
            "regressions": [r.to_dict() for r in self.regressions],
            "current": self.current.to_dict(),
        }

    def summary(self) -> str:
        """一句话结论，供 CI 日志与演示脚本直接打印."""
        if self.ok:
            return (
                f"性能未回归：当前 {self.current.samples} 个样本，"
                f"平均总耗时 {self.current.latency_ms.get(TOTAL_KEY, 0.0)}ms、"
                f"平均 {self.current.mean_tokens} token、"
                f"平均 ${self.current.mean_cost_usd:.6f}"
            )
        return "性能回归 {0} 项：" + "; ".join(r.message for r in self.regressions)


class PerformanceGuard:
    """性能门禁：用容忍倍数把"抖动"与"真回归"分开.

    判定规则统一为比值：``current / baseline > tolerance`` 即回归，
    且基线值必须超过对应的下限（``min_latency_ms`` / ``min_tokens``）。
    成本基数为 0 时（全本地模型的常见情形）不参与比值判定——0 的任意
    倍数都还是 0，比值判定在这里没有意义。
    """

    def __init__(
        self,
        baseline: PerfBaseline,
        *,
        latency_tolerance: float = 1.5,
        token_tolerance: float = 1.2,
        cost_tolerance: float = 1.2,
        min_latency_ms: float = DEFAULT_MIN_LATENCY_MS,
        min_tokens: float = DEFAULT_MIN_TOKENS,
    ):
        for name, value in (
            ("latency_tolerance", latency_tolerance),
            ("token_tolerance", token_tolerance),
            ("cost_tolerance", cost_tolerance),
        ):
            if value <= 0:
                raise ValueError(f"{name} 必须为正数")
        self.baseline = baseline
        self.latency_tolerance = latency_tolerance
        self.token_tolerance = token_tolerance
        self.cost_tolerance = cost_tolerance
        self.min_latency_ms = min_latency_ms
        self.min_tokens = min_tokens

    def compare(self, current: PerfBaseline) -> BaselineComparison:
        """逐指标比对并汇总所有回归项（不短路——一次看全，便于批量修复）."""
        regressions: list[Regression] = []

        # 总耗时与各阶段耗时：基线值低于下限的指标跳过（防抖动）
        for metric, base_value in self.baseline.latency_ms.items():
            current_value = current.latency_ms.get(metric)
            if current_value is None or base_value < self.min_latency_ms:
                continue
            ratio = current_value / base_value
            if ratio > self.latency_tolerance:
                regressions.append(
                    Regression(
                        metric=f"latency_ms[{metric}]",
                        baseline=base_value,
                        current=current_value,
                        ratio=round(ratio, 4),
                        threshold=self.latency_tolerance,
                        message=(
                            f"{metric} 耗时 {base_value}ms -> {current_value}ms"
                            f"（{ratio:.2f}x > {self.latency_tolerance}x）"
                        ),
                    )
                )

        if self.baseline.mean_tokens >= self.min_tokens:
            ratio = current.mean_tokens / self.baseline.mean_tokens
            if ratio > self.token_tolerance:
                regressions.append(
                    Regression(
                        metric="mean_tokens",
                        baseline=self.baseline.mean_tokens,
                        current=current.mean_tokens,
                        ratio=round(ratio, 4),
                        threshold=self.token_tolerance,
                        message=(
                            f"平均 token {self.baseline.mean_tokens} -> {current.mean_tokens}"
                            f"（{ratio:.2f}x > {self.token_tolerance}x）"
                        ),
                    )
                )

        if self.baseline.mean_cost_usd > 0:
            ratio = current.mean_cost_usd / self.baseline.mean_cost_usd
            if ratio > self.cost_tolerance:
                regressions.append(
                    Regression(
                        metric="mean_cost_usd",
                        baseline=self.baseline.mean_cost_usd,
                        current=current.mean_cost_usd,
                        ratio=round(ratio, 4),
                        threshold=self.cost_tolerance,
                        message=(
                            f"平均成本 ${self.baseline.mean_cost_usd:.6f} -> "
                            f"${current.mean_cost_usd:.6f}（{ratio:.2f}x > {self.cost_tolerance}x）"
                        ),
                    )
                )

        comparison = BaselineComparison(
            baseline=self.baseline, current=current, regressions=regressions
        )
        if comparison.ok:
            logger.info("性能对比通过: %s", comparison.summary())
        else:
            logger.warning("性能对比发现回归: %s", comparison.summary())
        return comparison


def guard_from_settings(baseline: PerfBaseline) -> PerformanceGuard:
    """按 ``settings`` 的容忍度构建门禁（阈值外置，调参不用改代码）."""
    from smart_research_agent.config import settings

    return PerformanceGuard(
        baseline,
        latency_tolerance=settings.perf_latency_tolerance,
        token_tolerance=settings.perf_token_tolerance,
        cost_tolerance=settings.perf_cost_tolerance,
    )


def compare_with_file(
    current: PerfBaseline,
    baseline_path: str | Path,
    *,
    clock: Callable[[], float] | None = None,  # noqa: ARG001 - 保留给调用方统一签名
) -> BaselineComparison:
    """读取基线文件并与当前采集对比（CI 与演示脚本的一行入口）.

    ``clock`` 参数仅为与其它采集函数保持一致的调用签名而保留，不参与计算
    （对比是纯算术）；显式保留可以避免调用方在"采集/对比"之间切换时改参数。
    """
    baseline = PerfBaseline.load(baseline_path)
    return guard_from_settings(baseline).compare(current)
