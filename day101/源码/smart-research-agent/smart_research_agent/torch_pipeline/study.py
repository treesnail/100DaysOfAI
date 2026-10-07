"""``study``：六张表 + 一条阶段线（day096 / M8-D7）.

```text
① 数据集表     全集 / 训练 / 评估三份的规模、正负比、内容指纹
② 批次表       drop_last 两个分支的批数与丢弃数
③ 设备表       cpu / cuda / mps 上同一份账的字节数与 fits
④ 检查点表     五件套各自的字节数与是否在场
⑤ 推理表       三种批大小下的调用次数、样本数、每样本秒数与吞吐（假 clock）
⑥ 性质表       七条性质是否通过 / 现场读数 / 两个来源
+  阶段线       七个阶段各由谁实现（pipeline_lines）
```

## 一条纪律：每一行都要带**数字**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
第 ① ③ ⑤ 张表尤其如此：它们给出的每一行都同时有"规模"与"派生读数"，
因此"换一个 batch_size 会怎样"这类问题可以**直接读出行**，而不是靠猜。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from smart_research_agent.optimizers.optimizer import make_train_optimizer
from smart_research_agent.regularization.normalization import initial_running
from smart_research_agent.regularization.network import build_regularized
from smart_research_agent.sequence_models.train import make_sign_dataset
from smart_research_agent.torch_pipeline import verify
from smart_research_agent.torch_pipeline.checkpoint import CHECKPOINT_FILES, save_checkpoint
from smart_research_agent.torch_pipeline.dataloader import DataLoader
from smart_research_agent.torch_pipeline.datasets import TabularDataset, make_dataset, split_dataset
from smart_research_agent.torch_pipeline.device import DevicePlan, plan_device
from smart_research_agent.torch_pipeline.inference import (
    build_inference_engine,
    latency_report,
    predict_dataset,
    stepping_clock,
)
from smart_research_agent.torch_pipeline.train import (
    PipelineConfig,
    PipelineReport,
    compare_configs,
)
from smart_research_agent.torch_pipeline.types import (
    DEVICE_CPU,
    DEVICE_KINDS,
    PIPELINE_PROPERTIES,
    STAGE_DESCRIPTIONS,
    STAGE_OWNERS,
    STAGES,
)

#: ① ② ⑤ 用的数据集与批（与演示脚本共用同一批数字）.
SIGN_LENGTH = 5
DATASET_SEED = 100
BATCH_SIZE = 8
EVAL_RATIO = 0.25

#: ② 用的批大小：刻意取一个**除不尽** 32 的数，好让 drop_last 的两个分支给出不同读数.
BATCH_TABLE_SIZE = 7

#: ⑥ 用的批大小（同样除不尽 24，于是"丢尾批"与"基准"会给出不同读数）.
ABLATION_BATCH_SIZE = 7

#: ③ 用的三份账（参数量 / 批大小 / 宽度 / 优化器倍数）.
DEVICE_PARAMS = 44
DEVICE_WIDTH = 4
DEVICE_MULTIPLIER = 3

#: ⑤ 用的三种批大小（越小调用次数越多、越慢）.
INFERENCE_BATCH_SIZES: tuple[int, ...] = (1, 4, 8)

#: ⑤ 用的小网络.
INFERENCE_SEED = 100
INFERENCE_HIDDEN = 4

#: ⑥ 用的三种消融变体（**一次只关一件事**）.
ABLATION_EPOCHS = 6
ABLATION_EVAL_RATIO = 0.25
ABLATION_SEED = 100
ABLATION_VARIANTS: dict[str, PipelineConfig] = {
    "基准": PipelineConfig(
        epochs=ABLATION_EPOCHS, batch_size=ABLATION_BATCH_SIZE, eval_ratio=ABLATION_EVAL_RATIO,
        seed=ABLATION_SEED, shuffle=True, drop_last=False,
    ),
    "不洗牌": PipelineConfig(
        epochs=ABLATION_EPOCHS, batch_size=ABLATION_BATCH_SIZE, eval_ratio=ABLATION_EVAL_RATIO,
        seed=ABLATION_SEED, shuffle=False, drop_last=False,
    ),
    "丢尾批": PipelineConfig(
        epochs=ABLATION_EPOCHS, batch_size=ABLATION_BATCH_SIZE, eval_ratio=ABLATION_EVAL_RATIO,
        seed=ABLATION_SEED, shuffle=True, drop_last=True,
    ),
}

#: 本课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTY_NAMES = PIPELINE_PROPERTIES


@dataclass(frozen=True)
class DatasetRow:
    """① 数据集表的一行."""

    name: str
    size: int
    positives: int
    negatives: int
    fingerprint: str

    def line(self) -> str:
        """``sign/train | 24 条 | 正 12 / 负 12 | 指纹 a1b2c3d4e5f60718``."""
        return (
            f"{self.name:<14} | {self.size:>3} 条 | 正 {self.positives} / 负 {self.negatives}"
            f" | 指纹 {self.fingerprint}"
        )


@dataclass(frozen=True)
class BatchRow:
    """② 批次表的一行."""

    dataset: str
    batch_size: int
    shuffle: bool
    drop_last: bool
    workers: int
    batch_count: int
    dropped: int

    def line(self) -> str:
        """``sign | b=8 | shuffle=True | drop_last=False | 批数 4 | 丢弃 0``."""
        return (
            f"{self.dataset:<10} | b={self.batch_size} | shuffle={self.shuffle} | "
            f"drop_last={self.drop_last} | workers={self.workers} | "
            f"批数 {self.batch_count} | 丢弃 {self.dropped}"
        )


@dataclass(frozen=True)
class DeviceRow:
    """③ 设备表的一行（三个字节数 + 是否装得下）."""

    device: str
    dtype: str
    params: int
    activations: int
    optimizer_bytes: int
    total_bytes: int
    budget_bytes: int
    fits: bool

    def line(self) -> str:
        """``cuda | float32 | 参数 176 B + 激活 128 B + 优化器 528 B = 832 B | 4.0 GiB | fits=True``."""
        mark = "fits=True" if self.fits else "fits=False"
        return (
            f"{self.device:<5} | {self.dtype} | 参数 {self.params} B + 激活 {self.activations} B + "
            f"优化器 {self.optimizer_bytes} B = {self.total_bytes} B | "
            f"{self.budget_bytes / 1024**3:.1f} GiB | {mark}"
        )


@dataclass(frozen=True)
class CheckpointRow:
    """④ 检查点表的一行."""

    file: str
    size_bytes: int
    present: bool

    def line(self) -> str:
        """``params.json | 1204 字节 | 在场=True``."""
        return f"{self.file:<16} | {self.size_bytes:>6} 字节 | 在场={self.present}"


@dataclass(frozen=True)
class InferenceRow:
    """⑤ 推理表的一行."""

    batch_size: int
    calls: int
    samples: int
    seconds: float
    seconds_per_sample: float
    throughput: float

    def line(self) -> str:
        """``b=8 | 调用 4 次 | 样本 32 | 0.000004s | 每条 0.000000125s | 吞吐 8000000/s``."""
        return (
            f"b={self.batch_size:<2} | 调用 {self.calls:>2} 次 | 样本 {self.samples:>2} | "
            f"{self.seconds:.6f}s | 每条 {self.seconds_per_sample:.9f}s | "
            f"吞吐 {self.throughput:.1f}/s"
        )


@dataclass(frozen=True)
class AblationRow:
    """⑥ 消融表的一行."""

    variant: str
    epochs_run: int
    steps: int
    initial_loss: float
    final_train_loss: float
    final_eval_loss: float
    improvement: float
    dropped: int

    def line(self) -> str:
        """``基准 | 6 轮 × 9 步 | 训练 0.69 → 0.12（↓82%）| 推理 0.20 | 丢弃 0``."""
        return (
            f"{self.variant:<4} | {self.epochs_run} 轮 × {self.steps} 步 | "
            f"训练 {self.initial_loss:.3e} → {self.final_train_loss:.3e}"
            f"（↓{self.improvement:.1%}）| 推理 {self.final_eval_loss:.3e} | 丢弃 {self.dropped}"
        )


@dataclass(frozen=True)
class PropertyRow:
    """⑦ 性质表的一行."""

    name: str
    passed: bool
    reading: float
    bound: str
    cross_check: str

    def line(self) -> str:
        """``通过 shuffle_order_is_deterministic | 读数 0.000e+00 == 0 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return (
            f"{mark} {self.name:<48} | 读数 {self.reading:.3e} {self.bound} | {self.cross_check}"
        )


def _default_dataset() -> TabularDataset:
    """缺省数据集：day094 的 ±1 符号任务（**完全确定**）."""
    return make_dataset(make_sign_dataset(SIGN_LENGTH), name="sign")


def dataset_rows(dataset: TabularDataset | None = None) -> tuple[DatasetRow, ...]:
    """① 数据集表：全集 / 训练 / 评估三份的规模、正负比与内容指纹."""
    resolved = dataset if dataset is not None else _default_dataset()
    train, evaluation = split_dataset(resolved, eval_ratio=EVAL_RATIO, seed=DATASET_SEED)
    rows: list[DatasetRow] = []
    for part in (resolved, train, evaluation):
        labels = part.labels()
        rows.append(
            DatasetRow(
                name=part.name,
                size=len(part),
                positives=sum(labels),
                negatives=len(labels) - sum(labels),
                fingerprint=part.fingerprint(),
            )
        )
    return tuple(rows)


def batch_rows(dataset: TabularDataset | None = None) -> tuple[BatchRow, ...]:
    """② 批次表：``drop_last`` 两个分支的批数与丢弃数（同一个 ``batch_size``）."""
    resolved = dataset if dataset is not None else _default_dataset()
    rows: list[BatchRow] = []
    for drop_last in (False, True):
        loader = DataLoader(
            resolved, BATCH_TABLE_SIZE, shuffle=True, drop_last=drop_last, seed=DATASET_SEED
        )
        rows.append(
            BatchRow(
                dataset=resolved.name,
                batch_size=BATCH_TABLE_SIZE,
                shuffle=True,
                drop_last=drop_last,
                workers=1,
                batch_count=len(loader),
                dropped=loader.dropped_samples(),
            )
        )
    return tuple(rows)


def device_rows() -> tuple[DeviceRow, ...]:
    """③ 设备表：三个设备上同一份账的字节数与 ``fits``（**预算是算术表**）."""
    rows: list[DeviceRow] = []
    for device in DEVICE_KINDS:
        plan: DevicePlan = plan_device(
            DEVICE_PARAMS,
            batch_size=BATCH_SIZE,
            width=DEVICE_WIDTH,
            device=device,
            optimizer_multiplier=DEVICE_MULTIPLIER,
        )
        rows.append(
            DeviceRow(
                device=plan.device,
                dtype=plan.dtype,
                params=plan.params,
                activations=plan.activations,
                optimizer_bytes=plan.optimizer_bytes,
                total_bytes=plan.total_bytes,
                budget_bytes=plan.budget_bytes,
                fits=plan.fits,
            )
        )
    return tuple(rows)


def checkpoint_rows(directory: str | Path | None = None) -> tuple[CheckpointRow, ...]:
    """④ 检查点表：五件套各自的字节数与是否在场.

    ``directory`` 缺省时先在一个临时目录里**真的写一份**小检查点再读它——
    因此表里的字节数来自真实写盘，而不是估计。
    """
    if directory is None:
        with tempfile.TemporaryDirectory() as temporary:
            return _checkpoint_rows_in(temporary)
    return _checkpoint_rows_in(directory)


def _checkpoint_rows_in(directory: str | Path) -> tuple[CheckpointRow, ...]:
    """在给定目录上：缺文件时先写一份，再逐文件列出字节数."""
    target = Path(directory)
    if not all((target / name).exists() for name in CHECKPOINT_FILES):
        model = build_regularized(
            "lstm", input_size=1, hidden_size=INFERENCE_HIDDEN, classes=2, seed=INFERENCE_SEED
        )
        optimizer = make_train_optimizer("adam", 0.01)
        save_checkpoint(
            target,
            params=model,
            optimizer_state=optimizer.state(),
            step=0,
            metrics={"epoch": 0, "step": 0},
            history=(),
        )
    rows: list[CheckpointRow] = []
    for name in CHECKPOINT_FILES:
        path = target / name
        rows.append(
            CheckpointRow(
                file=name,
                size_bytes=path.stat().st_size if path.exists() else 0,
                present=path.exists(),
            )
        )
    return tuple(rows)


def inference_rows(dataset: TabularDataset | None = None) -> tuple[InferenceRow, ...]:
    """⑤ 推理表：三种批大小下的调用次数 / 样本数 / 延迟读数（**注入假 clock**）."""
    resolved = dataset if dataset is not None else _default_dataset()
    model = build_regularized(
        "lstm", input_size=1, hidden_size=INFERENCE_HIDDEN, classes=2, seed=INFERENCE_SEED
    )
    rows: list[InferenceRow] = []
    for batch_size in INFERENCE_BATCH_SIZES:
        engine = build_inference_engine(
            model, running=initial_running(model.features), clock=stepping_clock()
        )
        predict_dataset(engine, resolved, batch_size=batch_size)
        report = latency_report(engine)
        rows.append(
            InferenceRow(
                batch_size=batch_size,
                calls=report.calls,
                samples=report.samples,
                seconds=report.seconds,
                seconds_per_sample=report.seconds_per_sample,
                throughput=report.throughput,
            )
        )
    return tuple(rows)


def ablation_reports() -> dict[str, PipelineReport]:
    """⑥ 消融表：三个变体各训一次（**最贵的一张表**）."""
    return compare_configs(_default_dataset(), ABLATION_VARIANTS)


def ablation_rows(reports: dict[str, PipelineReport] | None = None) -> tuple[AblationRow, ...]:
    """⑥ 消融表的行（``reports`` 可以外部传入，避免同一次会话里训练两遍）."""
    resolved = reports if reports is not None else ablation_reports()
    rows: list[AblationRow] = []
    for variant, report in resolved.items():
        rows.append(
            AblationRow(
                variant=variant,
                epochs_run=report.epochs_run,
                steps=report.steps,
                initial_loss=report.initial_loss,
                final_train_loss=report.final_train_loss,
                final_eval_loss=report.final_eval_loss,
                improvement=report.improvement,
                dropped=report.dropped_samples,
            )
        )
    return tuple(rows)


def property_rows() -> tuple[PropertyRow, ...]:
    """⑦ 性质表：七条性质逐行（读数来自 :func:`verify.check_all`）."""
    report = verify.check_all()
    rows: list[PropertyRow] = []
    for outcome in report.outcomes:
        check = outcome.check
        rows.append(
            PropertyRow(
                name=outcome.name,
                passed=outcome.passed,
                reading=check.reading,
                bound=check.bound_text(),
                cross_check=f"{check.left} vs {check.right}",
            )
        )
    return tuple(rows)


def pipeline_lines() -> tuple[str, ...]:
    """阶段线：七个阶段各由谁实现（**这一课的"转发而不重写"清单**）."""
    lines: list[str] = []
    for index, stage in enumerate(STAGES, start=1):
        lines.append(f"{index}. {stage:<10} | {STAGE_DESCRIPTIONS[stage]}")
        lines.append(f"   由谁实现：{STAGE_OWNERS[stage]}")
    return tuple(lines)


def study_lines(reports: dict[str, PipelineReport] | None = None) -> tuple[str, ...]:
    """一次跑完六张表 + 阶段线（演示脚本与教程引用的是同一批读数）.

    ``reports`` 可以外部传入（避免同一个训练在一次会话里跑两遍）——
    没有它时第 ⑥ 张表会**自己训一遍**。
    """
    lines: list[str] = []
    lines.append("== 1. 数据集表（全集 / 训练 / 评估）")
    for row in dataset_rows():
        lines.append("  " + row.line())
    lines.append("== 2. 批次表（drop_last 的两个分支）")
    for row in batch_rows():
        lines.append("  " + row.line())
    lines.append("== 3. 设备表（三个设备上同一份账）")
    for row in device_rows():
        lines.append("  " + row.line())
    lines.append("== 4. 检查点表（五件套的字节数）")
    for row in checkpoint_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 推理表（三种批大小下的延迟读数）")
    for row in inference_rows():
        lines.append("  " + row.line())
    lines.append("== 6. 消融表（三个变体各训一次）")
    for row in ablation_rows(reports):
        lines.append("  " + row.line())
    lines.append("== 7. 性质表（七条性质：是否通过 / 读数 / 两个来源）")
    for row in property_rows():
        lines.append("  " + row.line())
    lines.append("== 8. 阶段线（七个阶段各由谁实现）")
    for line in pipeline_lines():
        lines.append("  " + line)
    return tuple(lines)


#: 一个"什么都不改"的设备（③ 的对照：预算内的 CPU）.
FRESH_DEVICE = DEVICE_CPU

__all__ = [
    "ABLATION_BATCH_SIZE",
    "ABLATION_EPOCHS",
    "ABLATION_EVAL_RATIO",
    "ABLATION_SEED",
    "ABLATION_VARIANTS",
    "BATCH_SIZE",
    "BATCH_TABLE_SIZE",
    "DATASET_SEED",
    "DEVICE_MULTIPLIER",
    "DEVICE_PARAMS",
    "DEVICE_WIDTH",
    "EVAL_RATIO",
    "FRESH_DEVICE",
    "INFERENCE_BATCH_SIZES",
    "INFERENCE_HIDDEN",
    "INFERENCE_SEED",
    "PROPERTY_NAMES",
    "SIGN_LENGTH",
    "AblationRow",
    "BatchRow",
    "CheckpointRow",
    "DatasetRow",
    "DeviceRow",
    "InferenceRow",
    "PropertyRow",
    "ablation_reports",
    "ablation_rows",
    "batch_rows",
    "checkpoint_rows",
    "dataset_rows",
    "device_rows",
    "inference_rows",
    "pipeline_lines",
    "property_rows",
    "study_lines",
]
