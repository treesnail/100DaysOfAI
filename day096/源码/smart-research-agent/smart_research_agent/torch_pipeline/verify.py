"""``verify``：七条性质与三类判据（day096 / M8-D7）.

```text
相等（逐位 / 整数）  ① 同 seed 两次 Sampler.order(epoch) 的差异项数 = 0
                     ③ 批数与 n // b 的差 = 0（drop_last）
                     ⑤ save → load 之后压平参数逐位相同
                     ⑥ 中途恢复与一口气训到第 N 轮的参数最大绝对差 = 0   ← 本课最值钱的一条
                     ⑦ predict_batch 与逐条 predict_one 的输出最大绝对差 = 0
上界                 ④ DevicePlan.total_bytes 与手算的相对差 <= 1e-9
下界                 ② worker 分片的并集覆盖 = 1.0 且不相交项数 = 0
```

判据三类与 day092 / day094 / day095 同源：**相等**、**不超过上界**、**不低于下界**。

## 为什么第 ② 条必须是下界

因为"分片"这件事的可证伪形式是"它们**合起来刚好是全集**"：

```text
把取模分片写成切块分片（range(k·b, (k+1)·b)）
  ⇒ 换个 workers 就会漏样本 / 重叠
  ⇒ 而它**不会报错**：只是少数样本被喂了两遍、或几条从来没被训过
  ⇒ 只有"并集覆盖 = 1.0"这条下界能把它钉死
```

读数取 ``覆盖 − 重叠占比``：互不相交且覆盖全集时它恰好是 ``1.0``；
漏样本会把它压低，重叠也会把它压低——一条读数同时盯住两件事。

## 第 ⑥ 条为什么是本课最值钱的一条

它一次性检验了四件事是否全都对：打乱只依赖 ``(seed, epoch)``、
训练相的归一化不依赖 running、前向里没有随机性、以及**优化器的跨步状态真的被装回去了**。
其中任何一件出错，读数都会从 ``0.0`` 变成一个非零数——
而其余六条性质会**照常通过**（它们各自只看一小段）。
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from smart_research_agent.optimizers.optimizer import make_train_optimizer
from smart_research_agent.regularization.normalization import initial_running
from smart_research_agent.regularization.network import build_regularized, flatten_params
from smart_research_agent.sequence_models.train import make_sign_dataset
from smart_research_agent.torch_pipeline.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from smart_research_agent.torch_pipeline.dataloader import DataLoader
from smart_research_agent.torch_pipeline.datasets import make_dataset
from smart_research_agent.torch_pipeline.device import (
    activation_bytes,
    parameter_bytes,
    plan_device,
)
from smart_research_agent.torch_pipeline.errors import TorchPipelineError
from smart_research_agent.torch_pipeline.inference import (
    DEFAULT_CLOCK_STEP,
    build_inference_engine,
    predict_batch,
    predict_one,
    stepping_clock,
)
from smart_research_agent.torch_pipeline.sampler import Sampler, shard_indices
from smart_research_agent.torch_pipeline.train import PipelineConfig, resume_training, train_pipeline
from smart_research_agent.torch_pipeline.types import (
    DEVICE_BYTES_TOLERANCE,
    EXACT_TOLERANCE,
    PIPELINE_PROPERTIES,
    PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DEVICE_BYTES_MATCH_FORMULA,
    PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA,
    PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL,
    PROPERTY_RESUME_MATCHES_UNINTERRUPTED,
    PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC,
    PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET,
    SHARD_COVERAGE_LOWER_BOUND,
)

#: 第 ①② 条用的写死尺寸（换一个尺寸，所有读数都会变）.
SAMPLE_SIZE = 32
SAMPLE_WORKERS = 4
SAMPLE_SEED = 100
SAMPLE_BATCH_SIZE = 8
SAMPLE_EPOCH = 3

#: 第 ④ 条用的写死账（参数量 / 批大小 / 宽度 / 优化器倍数）.
DEVICE_PARAMS = 44
DEVICE_BATCH = 8
DEVICE_WIDTH = 4
DEVICE_MULTIPLIER = 3
DEVICE_KIND = "cuda"
DEVICE_DTYPE = "float32"

#: 第 ⑤ ⑥ ⑦ 条用的小网络与数据（都取小一点，让现场调用便宜）.
CHAIN_SEED = 5
CHAIN_HIDDEN = 3
SIGN_LENGTH = 5

#: 第 ⑥ 条用的训练配置（**同一 `seed` 保证切分与打乱逐位相同**）.
RESUME_SEED = 11
RESUME_EPOCHS = 6
RESUME_MID_EPOCHS = 3
RESUME_BATCH_SIZE = 8
RESUME_EVAL_RATIO = 0.25

#: 第 ⑦ 条注入的假 clock 步长（每次调用前进 1e-6 秒）.
FAKE_CLOCK_STEP = DEFAULT_CLOCK_STEP


@dataclass(frozen=True)
class Check:
    """一次性质校验的读数与判据（三类：相等 / 上界 / 下界）."""

    reading: float
    upper_bound: float | None = None
    lower_bound: float | None = None
    left: str = "-"
    right: str = "-"

    def passed(self) -> bool:
        """读数是否落在判据内（上界与下界同时生效）."""
        if self.upper_bound is not None and self.reading > self.upper_bound:
            return False
        return not (self.lower_bound is not None and self.reading < self.lower_bound)

    def bound_text(self) -> str:
        """判据的一行文本（上界用 ``<=``、下界用 ``>=``、相等用 ``== 0``）."""
        if self.lower_bound is not None and self.upper_bound is not None:
            return f"∈ [{self.lower_bound:.1e}, {self.upper_bound:.1e}]"
        if self.lower_bound is not None:
            return f">= {self.lower_bound:.1e}"
        if self.upper_bound is None:
            return "== 逐位"
        if self.upper_bound == 0.0:
            return "== 0"
        return f"<= {self.upper_bound:.1e}"


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：是否通过 + 现场读数 + 两个来源."""

    name: str
    passed: bool
    check: Check

    def line(self) -> str:
        """``通过 shuffle_order_is_deterministic | 读数 0.000e+00 == 0 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return (
            f"{mark} {self.name:<48} | 读数 {self.check.reading:.3e} "
            f"{self.check.bound_text()} | {self.check.left} vs {self.check.right}"
        )


@dataclass
class PropertyReport:
    """七条性质的汇总报告."""

    outcomes: list[PropertyOutcome]

    @property
    def passed(self) -> int:
        """通过的条数."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def total(self) -> int:
        """总条数."""
        return len(self.outcomes)

    def all_passed(self) -> bool:
        """是否全部通过."""
        return self.passed == self.total

    def lines(self) -> tuple[str, ...]:
        """逐行印出（演示脚本与教程引用的是同一批读数）."""
        return tuple(outcome.line() for outcome in self.outcomes)


def _max_abs_gap(left, right, *, rows: bool = False) -> float:
    """两组数（或两组行）的最大绝对差（长度不一致时直接记 ``inf``，让判据变红）."""
    if len(left) != len(right):
        return math.inf
    if rows:
        return max(
            (
                abs(a - b)
                for row_left, row_right in zip(left, right, strict=True)
                for a, b in zip(row_left, row_right, strict=True)
            ),
            default=0.0,
        )
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def _stepping_clock(step: float = FAKE_CLOCK_STEP) -> Callable[[], float]:
    """假时钟的**唯一实现**在 :mod:`inference`（:func:`inference.stepping_clock`）里——本处只转发."""
    return stepping_clock(step)


# --------------------------------------------------------------------------- #
# 单条性质
# --------------------------------------------------------------------------- #


def check_shuffle_order_is_deterministic() -> PropertyOutcome:
    """① 同 seed 两次 ``Sampler.order(epoch)`` 的差异项数 = 0（读数 = 差异项数）."""
    first = Sampler(size=SAMPLE_SIZE, shuffle=True, seed=SAMPLE_SEED).order(SAMPLE_EPOCH)
    second = Sampler(size=SAMPLE_SIZE, shuffle=True, seed=SAMPLE_SEED).order(SAMPLE_EPOCH)
    mismatches = sum(1 for a, b in zip(first, second, strict=True) if a != b)
    mismatches += abs(len(first) - len(second))
    check = Check(
        reading=float(mismatches),
        upper_bound=EXACT_TOLERANCE,
        left="Sampler(seed=100).order(3) 第一次",
        right="Sampler(seed=100).order(3) 第二次",
    )
    return PropertyOutcome(PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC, check.passed(), check)


def check_worker_shards_partition_the_dataset() -> PropertyOutcome:
    """② worker 分片的并集覆盖 = 1.0 且不相交项数 = 0（读数 = 覆盖 − 重叠占比，下界 1.0）."""
    shards = [
        shard_indices(SAMPLE_SIZE, workers=SAMPLE_WORKERS, worker_id=worker)
        for worker in range(SAMPLE_WORKERS)
    ]
    union = set().union(*shards)
    coverage = len(union) / SAMPLE_SIZE
    overlap = sum(len(shard) for shard in shards) - len(union)
    reading = coverage - overlap / SAMPLE_SIZE
    check = Check(
        reading=reading,
        lower_bound=SHARD_COVERAGE_LOWER_BOUND,
        left=f"{SAMPLE_WORKERS} 个分片的并集覆盖 {coverage:.1f}",
        right=f"重叠项数 {overlap}",
    )
    return PropertyOutcome(PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET, check.passed(), check)


def check_drop_last_matches_batch_formula() -> PropertyOutcome:
    """③ ``DataLoader.batches(epoch)`` 的长度与 ``n // b`` 一致（读数 = 差异项数）."""
    dataset = make_dataset(make_sign_dataset(SIGN_LENGTH), name="sign")
    loader = DataLoader(
        dataset, SAMPLE_BATCH_SIZE, shuffle=True, drop_last=True, seed=SAMPLE_SEED
    )
    produced = len(loader.batches(SAMPLE_EPOCH))
    expected = len(dataset) // SAMPLE_BATCH_SIZE
    check = Check(
        reading=float(abs(produced - expected)),
        upper_bound=EXACT_TOLERANCE,
        left=f"DataLoader.batches(3) 的批数 {produced}",
        right=f"n // b = {len(dataset)} // {SAMPLE_BATCH_SIZE} = {expected}",
    )
    return PropertyOutcome(PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA, check.passed(), check)


def check_device_bytes_match_formula() -> PropertyOutcome:
    """④ ``DevicePlan.total_bytes`` 与手算的相对差 <= 1e-9（**上界**判据）."""
    plan = plan_device(
        DEVICE_PARAMS,
        batch_size=DEVICE_BATCH,
        width=DEVICE_WIDTH,
        device=DEVICE_KIND,
        dtype=DEVICE_DTYPE,
        optimizer_multiplier=DEVICE_MULTIPLIER,
    )
    manual = (
        parameter_bytes(DEVICE_PARAMS, dtype=DEVICE_DTYPE)
        + activation_bytes(DEVICE_BATCH, DEVICE_WIDTH, dtype=DEVICE_DTYPE)
        + DEVICE_MULTIPLIER * parameter_bytes(DEVICE_PARAMS, dtype=DEVICE_DTYPE)
    )
    reading = abs(plan.total_bytes - manual) / manual
    check = Check(
        reading=reading,
        upper_bound=DEVICE_BYTES_TOLERANCE,
        left="DevicePlan.total_bytes",
        right=f"手算 params + activations + {DEVICE_MULTIPLIER} × params = {manual}",
    )
    return PropertyOutcome(PROPERTY_DEVICE_BYTES_MATCH_FORMULA, check.passed(), check)


def check_checkpoint_round_trip_is_bitwise() -> PropertyOutcome:
    """⑤ ``save → load`` 之后压平参数逐位相同（读数 = 最大绝对差）."""
    model = build_regularized(
        "lstm", input_size=1, hidden_size=CHAIN_HIDDEN, classes=2, seed=CHAIN_SEED
    )
    optimizer = make_train_optimizer("adam", 0.01)
    flat = flatten_params(model)
    optimizer.step(flat, tuple(0.0 for _ in flat))
    with tempfile.TemporaryDirectory() as directory:
        manifest = save_checkpoint(
            directory,
            params=model,
            optimizer_state=optimizer.state(),
            step=0,
            metrics={"epoch": 0, "step": 0},
            history=(),
        )
        loaded = load_checkpoint(directory)
        verify_checkpoint(directory, loaded.manifest)
    gap = _max_abs_gap(flat, loaded.params)
    check = Check(
        reading=gap,
        upper_bound=EXACT_TOLERANCE,
        left="save → load 的压平参数",
        right=f"原始压平参数（清单 {manifest.params_sha256[:12]}…）",
    )
    return PropertyOutcome(PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE, check.passed(), check)


def check_resume_matches_uninterrupted() -> PropertyOutcome:
    """⑥ 中途落检查点再恢复训到第 N 轮，与一口气训到第 N 轮，参数最大绝对差 = 0."""
    dataset = make_dataset(make_sign_dataset(SIGN_LENGTH), name="sign")
    full_config = PipelineConfig(
        epochs=RESUME_EPOCHS,
        batch_size=RESUME_BATCH_SIZE,
        optimizer="adam",
        shuffle=True,
        eval_ratio=RESUME_EVAL_RATIO,
        seed=RESUME_SEED,
        early_stop_patience=0,
    )
    mid_config = replace(full_config, epochs=RESUME_MID_EPOCHS)
    with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as mid_dir:
        train_pipeline(dataset, config=full_config, checkpoint_dir=full_dir)
        train_pipeline(dataset, config=mid_config, checkpoint_dir=mid_dir)
        resume_training(dataset, checkpoint_dir=mid_dir, config=full_config)
        expected = load_checkpoint(full_dir).params
        resumed = load_checkpoint(mid_dir).params
    gap = _max_abs_gap(expected, resumed)
    check = Check(
        reading=gap,
        upper_bound=EXACT_TOLERANCE,
        left=f"一口气训到第 {RESUME_EPOCHS} 轮的参数",
        right=f"第 {RESUME_MID_EPOCHS} 轮落检查点后恢复的最终参数",
    )
    return PropertyOutcome(PROPERTY_RESUME_MATCHES_UNINTERRUPTED, check.passed(), check)


def check_predict_batch_equals_sequential() -> PropertyOutcome:
    """⑦ ``predict_batch`` 与逐条 ``predict_one`` 的输出最大绝对差 = 0."""
    model = build_regularized(
        "lstm", input_size=1, hidden_size=CHAIN_HIDDEN, classes=2, seed=CHAIN_SEED
    )
    dataset = make_dataset(make_sign_dataset(SIGN_LENGTH), name="sign")
    samples = tuple(dataset[index] for index in range(4))
    batch_engine = build_inference_engine(
        model, running=initial_running(model.features), clock=_stepping_clock()
    )
    sequential_engine = build_inference_engine(
        model, running=initial_running(model.features), clock=_stepping_clock()
    )
    batched = predict_batch(batch_engine, samples)
    sequential = tuple(predict_one(sequential_engine, sample) for sample in samples)
    gap = _max_abs_gap(batched, sequential, rows=True)
    check = Check(
        reading=gap,
        upper_bound=EXACT_TOLERANCE,
        left="predict_batch 的输出",
        right="逐条 predict_one 的输出",
    )
    return PropertyOutcome(PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL, check.passed(), check)


#: 七条性质的名字 -> 检查函数（键顺序 = :data:`types.PIPELINE_PROPERTIES`）.
CHECKS: dict[str, Callable[[], PropertyOutcome]] = {
    PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC: check_shuffle_order_is_deterministic,
    PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET: check_worker_shards_partition_the_dataset,
    PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA: check_drop_last_matches_batch_formula,
    PROPERTY_DEVICE_BYTES_MATCH_FORMULA: check_device_bytes_match_formula,
    PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE: check_checkpoint_round_trip_is_bitwise,
    PROPERTY_RESUME_MATCHES_UNINTERRUPTED: check_resume_matches_uninterrupted,
    PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL: check_predict_batch_equals_sequential,
}

if set(CHECKS) != set(PIPELINE_PROPERTIES):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError(
        "性质名单与检查函数表不一致：少一条的性质会静默地不在报告里出现，"
        "而'少一条'与'它通过了'在读报告时长得一样。"
    )

if set(PROPERTY_DESCRIPTIONS) != set(PIPELINE_PROPERTIES):  # pragma: no cover
    raise TorchPipelineError("性质说明表与名单不一致。")


def check_all() -> PropertyReport:
    """跑完七条性质，返回汇总报告（顺序与 :data:`types.PIPELINE_PROPERTIES` 一致）."""
    return PropertyReport(outcomes=[CHECKS[name]() for name in PIPELINE_PROPERTIES])


__all__ = [
    "CHAIN_HIDDEN",
    "CHAIN_SEED",
    "CHECKS",
    "DEVICE_BATCH",
    "DEVICE_DTYPE",
    "DEVICE_KIND",
    "DEVICE_MULTIPLIER",
    "DEVICE_PARAMS",
    "DEVICE_WIDTH",
    "FAKE_CLOCK_STEP",
    "RESUME_BATCH_SIZE",
    "RESUME_EPOCHS",
    "RESUME_EVAL_RATIO",
    "RESUME_MID_EPOCHS",
    "RESUME_SEED",
    "SAMPLE_BATCH_SIZE",
    "SAMPLE_EPOCH",
    "SAMPLE_SEED",
    "SAMPLE_SIZE",
    "SAMPLE_WORKERS",
    "SIGN_LENGTH",
    "Check",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_checkpoint_round_trip_is_bitwise",
    "check_device_bytes_match_formula",
    "check_drop_last_matches_batch_formula",
    "check_predict_batch_equals_sequential",
    "check_resume_matches_uninterrupted",
    "check_shuffle_order_is_deterministic",
    "check_worker_shards_partition_the_dataset",
]
