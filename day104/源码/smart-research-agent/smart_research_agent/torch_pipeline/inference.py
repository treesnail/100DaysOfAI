"""``inference``：把"训好的参数 → 一条预测"包成一个带计数器的引擎（day096 / M8-D7）.

```text
InferenceEngine     参数 + running 统计量 + **可注入的时钟** + 三个计数器
predict_one         一条样本的预测（**只走推理相**：phase="eval"）
predict_batch       一批样本的预测（一次前向，与逐条的顺序调用给出同一串结果）
predict_dataset     整份数据集 → argmax 标签
LatencyReport       延迟读数：seconds_per_sample / throughput
```

## 一、只走推理相

```text
phase="eval"  ⇒  BatchNorm 用 running 统计量、dropout 恒等
phase="train" ⇒  BatchNorm 用**这一批自己的**统计量、dropout 置零
```

推理时用训练相**不会报错**：它只让"单条样本的预测"随批里还有谁而变，
而所有形状检查都会通过。因此本模块把 ``phase`` 写死成 ``eval``，
不给调用方留一个可以填错的参数。

## 二、延迟读数怎么才可复现

```text
默认 clock = time.perf_counter（真实墙钟，读数不可复现）
测试 / 性质 注入一个假 clock：它每次返回一个**预先写死的**数
  ⇒ seconds_per_sample = seconds / samples 变成逐位可复算的读数
```

把时钟做成**构造参数**而不是模块级全局，是"一个量只写一遍"在时间维度上的样子：
读数只有一个来源，而那个来源可以被替换成一个确定的替身。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smart_research_agent.regularization.normalization import (
    RunningStatistics,
    initial_running,
)
from smart_research_agent.regularization.network import (
    RegularizedParams,
    as_batch,
    regularized_forward,
)
from smart_research_agent.regularization.types import PHASE_EVAL
from smart_research_agent.torch_pipeline.dataloader import Batch as LoaderBatch
from smart_research_agent.torch_pipeline.dataloader import batch_samples
from smart_research_agent.torch_pipeline.datasets import Sample, TabularDataset
from smart_research_agent.torch_pipeline.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)

#: 时钟：不带参数、返回一个单调递增的秒数（默认 :func:`time.perf_counter`）.
Clock = Callable[[], float]

#: 推理只走这一个相（**写死，不给调用方填错的余地**）.
INFERENCE_PHASE = PHASE_EVAL

#: 假时钟的默认步长（每秒 1e-6 的推进，读起来像"一次很小的调用"）.
DEFAULT_CLOCK_STEP = 1e-6


def stepping_clock(step: float = DEFAULT_CLOCK_STEP) -> Clock:
    """一个**确定**的假时钟：每次调用都往前迈固定的一步.

    它存在的理由与 LCG 一样：让"延迟读数"可以被逐位复现。
    注入它之后，``seconds_per_sample`` 变成一个**算得出来**的数。
    """
    if not math.isfinite(step) or step <= 0.0:
        raise ParameterError(f"step 必须是正的有限数，收到 {step!r}。")
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step
        return state["now"]

    return clock


@dataclass
class InferenceEngine:
    """推理封装：持有网络 + running 统计量 + 可注入时钟 + 三个计数器（**非 frozen**）."""

    params: RegularizedParams
    running: RunningStatistics
    clock: Clock = time.perf_counter
    calls: int = 0
    samples: int = 0
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.params, RegularizedParams):
            raise ParameterError(
                f"params 必须是 RegularizedParams，收到 {type(self.params).__name__}。"
            )
        if not isinstance(self.running, RunningStatistics):
            raise ParameterError(
                f"running 必须是 RunningStatistics，收到 {type(self.running).__name__}。"
            )
        if self.running.features != self.params.features:
            raise ShapeError(
                f"running 的宽度 {self.running.features} 与网络的宽 "
                f"{self.params.features} 不一致：推理相的归一化读的就是它。"
            )
        if not callable(self.clock):
            raise ParameterError(f"clock 必须可调用，收到 {type(self.clock).__name__}。")

    @property
    def model(self) -> RegularizedParams:
        """本课的"模型"就是 ``regularization`` 的那一份网络（**没有第二份模型**）."""
        return self.params

    @property
    def features(self) -> int:
        """网络宽度（= BatchNorm 的特征数）."""
        return self.params.features

    def line(self) -> str:
        """一行说明：``推理封装 | lstm(H=4,…) | 调用 4 次 | 样本 9 条 | 累计 0.000018s``."""
        return (
            f"推理封装 | {self.params.line()} | 调用 {self.calls} 次 | "
            f"样本 {self.samples} 条 | 累计 {self.latency_seconds:.6f}s"
        )


def build_inference_engine(
    params: RegularizedParams,
    *,
    running: RunningStatistics | None = None,
    clock: Clock | None = None,
) -> InferenceEngine:
    """造一个推理引擎（``running`` 缺省取"什么都没见过"的初值 ``μ=0、σ²=1``）."""
    if not isinstance(params, RegularizedParams):
        raise ParameterError(f"params 必须是 RegularizedParams，收到 {type(params).__name__}。")
    resolved_running = running if running is not None else initial_running(params.features)
    resolved_clock = clock if clock is not None else time.perf_counter
    return InferenceEngine(params=params, running=resolved_running, clock=resolved_clock)


def _tick(engine: InferenceEngine, start: float) -> float:
    """计算一次调用的耗时并累加（**非有限 / 倒退的时钟当场拒绝**）."""
    elapsed = float(engine.clock()) - start
    if not math.isfinite(elapsed):
        raise NumericError(
            "这一次调用的耗时不是有限数：时钟返回了 nan / inf，"
            "任何延迟读数都会被它污染。"
        )
    if elapsed < 0.0:
        raise NumericError(
            f"这一次调用的耗时是负数（{elapsed!r}）：时钟倒退了——"
            "延迟读数的第一条前提是它非负。"
        )
    engine.latency_seconds += elapsed
    return elapsed


def _as_batch_of_samples(batch: object) -> tuple[Sample, ...]:
    """把"一批样本"或一个 :class:`dataloader.Batch` 收敛成 ``tuple[Sample, ...]``."""
    if isinstance(batch, LoaderBatch):
        return batch_samples(batch)
    try:
        return as_batch(batch, name="batch")
    except ValueError as exc:
        raise ShapeError(f"batch 不是一批样本：{exc}") from exc


def predict_one(engine: InferenceEngine, sample: Sample) -> tuple[float, ...]:
    """一条样本的预测：**只走推理相**，返回这一条的 logits（同时把计数与耗时加上去）."""
    if not isinstance(engine, InferenceEngine):
        raise ParameterError(f"engine 必须是 InferenceEngine，收到 {type(engine).__name__}。")
    checked = _as_batch_of_samples((sample,))
    start = float(engine.clock())
    logits = regularized_forward(
        engine.params, checked, phase=INFERENCE_PHASE, running=engine.running
    )
    engine.calls += 1
    engine.samples += 1
    _tick(engine, start)
    return tuple(logits[0])


def predict_batch(
    engine: InferenceEngine, batch: object
) -> tuple[tuple[float, ...], ...]:
    """一批样本的预测：**一次前向**给出每一行的 logits（与逐条调用逐位相同）."""
    if not isinstance(engine, InferenceEngine):
        raise ParameterError(f"engine 必须是 InferenceEngine，收到 {type(engine).__name__}。")
    checked = _as_batch_of_samples(batch)
    start = float(engine.clock())
    logits = regularized_forward(
        engine.params, checked, phase=INFERENCE_PHASE, running=engine.running
    )
    engine.calls += 1
    engine.samples += len(checked)
    _tick(engine, start)
    return tuple(tuple(row) for row in logits)


def predict_dataset(
    engine: InferenceEngine, dataset: TabularDataset, *, batch_size: int
) -> tuple[int, ...]:
    """整份数据集的预测标签（按 ``batch_size`` 分批，返回的顺序与数据集一致）."""
    if not isinstance(engine, InferenceEngine):
        raise ParameterError(f"engine 必须是 InferenceEngine，收到 {type(engine).__name__}。")
    if not isinstance(dataset, TabularDataset):
        raise ParameterError(f"dataset 必须是 TabularDataset，收到 {type(dataset).__name__}。")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ParameterError(f"batch_size 必须是 >= 1 的整数，收到 {batch_size!r}。")
    labels: list[int] = []
    for start in range(0, len(dataset), batch_size):
        chunk = tuple(
            dataset[index] for index in range(start, min(start + batch_size, len(dataset)))
        )
        for row in predict_batch(engine, chunk):
            if not row:  # pragma: no cover - 分类头至少给一个类别
                raise ShapeError("预测的 logits 是空向量：无法取 argmax。")
            labels.append(max(range(len(row)), key=lambda index: row[index]))
    return tuple(labels)


@dataclass(frozen=True)
class LatencyReport:
    """一次会话的延迟读数：调用次数 / 样本数 / 累计秒数（两个派生量现算）."""

    calls: int
    samples: int
    seconds: float

    def __post_init__(self) -> None:
        for name in ("calls", "samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ParameterError(f"{name} 必须是 >= 0 的整数，收到 {value!r}。")
        if not math.isfinite(self.seconds) or self.seconds < 0.0:
            raise NumericError(f"seconds 必须是非负的有限数，收到 {self.seconds!r}。")

    @property
    def seconds_per_sample(self) -> float:
        """每个样本的平均秒数（没有样本时记 0.0，**不除零**）."""
        if self.samples == 0:
            return 0.0
        return self.seconds / self.samples

    @property
    def throughput(self) -> float:
        """每秒样本数（累计秒数为 0 时记 0.0，**不给出 inf**）."""
        if self.seconds == 0.0:
            return 0.0
        return self.samples / self.seconds

    def line(self) -> str:
        """一行说明：``延迟 | 4 次调用 | 9 条样本 | 0.000018s | 每条 0.000002s | 吞吐 500000/s``."""
        return (
            f"延迟 | {self.calls} 次调用 | {self.samples} 条样本 | {self.seconds:.6f}s | "
            f"每条 {self.seconds_per_sample:.9f}s | 吞吐 {self.throughput:.1f}/s"
        )


def latency_report(engine: InferenceEngine) -> LatencyReport:
    """把引擎上的三个计数器包成一份 :class:`LatencyReport`（读数只有一个来源）."""
    if not isinstance(engine, InferenceEngine):
        raise ParameterError(f"engine 必须是 InferenceEngine，收到 {type(engine).__name__}。")
    return LatencyReport(calls=engine.calls, samples=engine.samples, seconds=engine.latency_seconds)


def reset_counters(engine: InferenceEngine) -> None:
    """清空三个计数器（**参数与 running 不动**）——换一段测量时先清一次."""
    if not isinstance(engine, InferenceEngine):
        raise ParameterError(f"engine 必须是 InferenceEngine，收到 {type(engine).__name__}。")
    engine.calls = 0
    engine.samples = 0
    engine.latency_seconds = 0.0


__all__ = [
    "Clock",
    "DEFAULT_CLOCK_STEP",
    "INFERENCE_PHASE",
    "InferenceEngine",
    "LatencyReport",
    "build_inference_engine",
    "latency_report",
    "predict_batch",
    "predict_dataset",
    "predict_one",
    "reset_counters",
    "stepping_clock",
]
