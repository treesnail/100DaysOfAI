"""``train``：把"数据 → 设备 → 训练 → 检查点"串成一条回路（day096 / M8-D7）.

```text
构造 DataLoader → 每个 epoch 打乱 → 逐批前向 / 反向（复用 day095 的 loss_and_grad）
  → 调用 day094 的 check_gradient_norm 守卫 → 优化器 step（make_train_optimizer）
  → 每 epoch 用推理相评一次 → 若给了 checkpoint_dir 则落一次五件套检查点
```

## 一、回路顺序是**写死**的（不给调用方自己拼的机会）

```text
取梯度（day095） → 过范数守卫（day094） → 优化器 step（day092） → 传回参数
```

四步的顺序与 day092 / day094 / day095 的回路逐字相同：先看范数再走一步，
"爆炸的那一步"因此不会进到参数里。本课**不重写**这四步中的任何一步——
它只是把"每一步喂哪一批"这件事换成 DataLoader 给的批。

## 二、训练损失与评估损失是两把尺子

```text
训练损失   训练相：BatchNorm 用**本批统计量**、dropout 置零            （每批一次）
评估损失   推理相：BatchNorm 用 **running 统计量**、dropout 恒等       （每轮一次）
```

只报其中一把会把两件事看反：训练损失低不代表推理损失低（day095 已经撞过这堵墙）。
本课的 :class:`PipelineReport` 因此把初始损失、训练损失、评估损失、最好评估损失四个数都给出来。

## 三、恢复训练为什么能逐位相等

```text
① 打乱只依赖 (seed, epoch)          ⇒ 第 k+1 轮的批与不中断时**逐位相同**
② 训练相的归一化只用本批统计量       ⇒ 训练轨迹与 running 统计量无关
③ dropout 恒等（本课不设丢弃率）     ⇒ 前向没有随机性
④ 检查点装回了**优化器的跨步状态**   ⇒ Adam 的一阶 / 二阶动量不会从零开始
```

四条合起来，让"中途停下再恢复"与"一口气跑完"给出同一批参数——
这是本课最值钱的一条性质（:data:`types.PROPERTY_RESUME_MATCHES_UNINTERRUPTED`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.neural_basics.losses import cross_entropy
from smart_research_agent.optimizers.optimizer import (
    SUPPORTED_OPTIMIZERS,
    TrainingOptimizer,
    make_train_optimizer,
)
from smart_research_agent.optimizers.types import OPTIMIZER_STATE_KEYS
from smart_research_agent.regularization.normalization import (
    RunningStatistics,
    initial_running,
)
from smart_research_agent.regularization.network import (
    RegularizedParams,
    build_regularized,
    flatten_gradients,
    flatten_params,
    loss_and_grad,
    regularized_forward,
    unflatten_params,
)
from smart_research_agent.regularization.train import EpochRecord
from smart_research_agent.regularization.types import PHASE_EVAL, PHASE_TRAIN
from smart_research_agent.sequence_models.train import (
    MAX_GRADIENT_NORM,
    check_gradient_norm,
)
from smart_research_agent.torch_pipeline.checkpoint import (
    load_checkpoint,
    record_from_payload,
    save_checkpoint,
)
from smart_research_agent.torch_pipeline.dataloader import DataLoader, batch_samples
from smart_research_agent.torch_pipeline.datasets import TabularDataset, split_dataset
from smart_research_agent.torch_pipeline.device import DevicePlan, plan_device
from smart_research_agent.torch_pipeline.errors import (
    CheckpointError,
    DeviceError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.torch_pipeline.types import (
    DEVICE_CPU,
    DEVICE_KINDS,
    DTYPE_BYTES,
    DTYPE_FLOAT32,
)

#: 缺省配置（**每一项都可以单独改**——它是"这一次训练"的决定，不是服务级默认值）.
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EVAL_RATIO = 0.25
DEFAULT_SEED = 100
DEFAULT_HIDDEN_SIZE = 4

#: 缺省的循环单元（与 day094 / day095 的实验对象一致）.
DEFAULT_CELL = "lstm"

#: 一次 BPTT 的梯度整体范数上界（**day094 的护栏**，本课照原样使用）.
GRADIENT_NORM_GUARD = MAX_GRADIENT_NORM


@dataclass(frozen=True)
class PipelineConfig:
    """一次端到端训练的全部旋钮（``validate()`` 是唯一一处判据）."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: str = "adam"
    shuffle: bool = True
    drop_last: bool = False
    workers: int = 1
    eval_ratio: float = DEFAULT_EVAL_RATIO
    seed: int = DEFAULT_SEED
    device: str = DEVICE_CPU
    dtype: str = DTYPE_FLOAT32
    early_stop_patience: int = 0

    def validate(self) -> None:
        """逐项校验（非法值一律 **当场拒绝**，不给一个静默的默认值）."""
        _require_int(self.epochs, name="epochs", minimum=1)
        _require_int(self.batch_size, name="batch_size", minimum=1)
        _require_int(self.workers, name="workers", minimum=1)
        _require_int(self.early_stop_patience, name="early_stop_patience", minimum=0)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ParameterError(f"seed 必须是整数，收到 {self.seed!r}。")
        if not isinstance(self.shuffle, bool):
            raise ParameterError(f"shuffle 必须是布尔值，收到 {self.shuffle!r}。")
        if not isinstance(self.drop_last, bool):
            raise ParameterError(f"drop_last 必须是布尔值，收到 {self.drop_last!r}。")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ParameterError(
                f"learning_rate 必须是正的有限数，收到 {self.learning_rate!r}。"
            )
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ParameterError(
                f"未知的优化器 {self.optimizer!r}：可用取值 {list(SUPPORTED_OPTIMIZERS)}。"
            )
        if (
            isinstance(self.eval_ratio, bool)
            or not isinstance(self.eval_ratio, (int, float))
            or not math.isfinite(self.eval_ratio)
            or not 0.0 <= self.eval_ratio < 1.0
        ):
            raise ParameterError(
                f"eval_ratio 必须落在 [0, 1)，收到 {self.eval_ratio!r}："
                "取 1 时训练集为空、取负数没有意义。"
            )
        if self.device not in DEVICE_KINDS:
            raise DeviceError(
                f"未知的设备 {self.device!r}：可用取值 {list(DEVICE_KINDS)}。"
            )
        if self.dtype not in DTYPE_BYTES:
            raise DeviceError(
                f"未知的精度 {self.dtype!r}：可用取值 {sorted(DTYPE_BYTES)}。"
            )

    def line(self) -> str:
        """一行说明：``20 轮 × b8 | adam lr=0.05 | shuffle=True drop_last=False | eval_ratio=0.25 | cpu/float32``."""
        return (
            f"{self.epochs} 轮 × b{self.batch_size} | {self.optimizer} lr={self.learning_rate:g} | "
            f"shuffle={self.shuffle} drop_last={self.drop_last} workers={self.workers} | "
            f"eval_ratio={self.eval_ratio:g} | {self.device}/{self.dtype} | "
            f"patience={self.early_stop_patience}"
        )


@dataclass(frozen=True)
class PipelineReport:
    """一次端到端训练的全过程读数（**每个数字都来自现场调用**）."""

    epochs_run: int
    steps: int
    batch_count: int
    dropped_samples: int
    initial_loss: float
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    improvement: float
    device_plan: DevicePlan
    dataset_fingerprint: str
    history: tuple[EpochRecord, ...]

    def summary_line(self) -> str:
        """一行说明：``20 轮 × 3 批 | 训练 6.9e-01 → 1.2e-02（↓98%）| 推理 8.1e-02 | 60 步 | 丢弃 0``."""
        return (
            f"{self.epochs_run} 轮 × {self.batch_count} 批 | "
            f"训练 {self.initial_loss:.3e} → {self.final_train_loss:.3e}"
            f"（↓{self.improvement:.1%}）| 推理 {self.final_eval_loss:.3e}"
            f"（最好 {self.best_eval_loss:.3e}）| {self.steps} 步 | 丢弃 {self.dropped_samples}"
        )

    def lines(self) -> tuple[str, ...]:
        """逐行印出（演示脚本与教程引用的是同一批读数）."""
        rows = [
            self.summary_line(),
            f"设备：{self.device_plan.line()}",
            f"数据集指纹：{self.dataset_fingerprint}",
        ]
        rows.extend(record.line() for record in self.history)
        return tuple(rows)


def _require_int(value: object, *, name: str, minimum: int) -> int:
    """整数且不小于 ``minimum``（布尔不是整数）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < minimum:
        raise ParameterError(f"{name} 必须 >= {minimum}，收到 {value}。")
    return int(value)


def _split_for_pipeline(
    dataset: TabularDataset, config: PipelineConfig
) -> tuple[TabularDataset, TabularDataset]:
    """按配置切分：``eval_ratio=0`` 时**不切**（训练集就是评估集），否则走 ``split_dataset``."""
    if config.eval_ratio == 0.0:
        return dataset, dataset
    return split_dataset(dataset, eval_ratio=config.eval_ratio, seed=config.seed)


def _build_model(dataset: TabularDataset, config: PipelineConfig) -> RegularizedParams:
    """按数据集的输入宽度造一个"装上 day095 旋钮"的分类器（**不重写网络**）."""
    input_size = len(dataset[0][0][0])
    return build_regularized(
        DEFAULT_CELL,
        input_size=input_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        classes=2,
        seed=config.seed,
    )


def _mean_loss(
    params: RegularizedParams,
    loader: DataLoader,
    *,
    epoch: int,
    running: RunningStatistics,
    phase: str,
) -> float:
    """在 ``loader`` 的第 ``epoch`` 轮批上算平均交叉熵（**前向相由调用方给**）."""
    total = 0.0
    count = 0
    for batch in loader.batches(epoch):
        logits = regularized_forward(params, batch_samples(batch), phase=phase, running=running)
        for row, label in zip(logits, batch.labels, strict=True):
            total += cross_entropy(row, label)
            count += 1
    return total / max(count, 1)


def evaluate(
    params: RegularizedParams,
    dataset: TabularDataset,
    *,
    running: RunningStatistics,
    batch_size: int,
) -> tuple[float, float]:
    """**推理相**地评估：返回 ``(平均损失, 准确率)``（BatchNorm 用 running、dropout 恒等）."""
    loader = DataLoader(dataset, batch_size, shuffle=False, seed=0)
    total = 0.0
    hits = 0
    count = 0
    for batch in loader.batches(0):
        logits = regularized_forward(
            params, batch_samples(batch), phase=PHASE_EVAL, running=running
        )
        for row, label in zip(logits, batch.labels, strict=True):
            total += cross_entropy(row, label)
            if max(range(len(row)), key=lambda index: row[index]) == label:
                hits += 1
            count += 1
    return total / max(count, 1), hits / max(count, 1)


def _checkpoint_metrics(
    *,
    epoch: int,
    step: int,
    initial_loss: float,
    record: EpochRecord,
    running: RunningStatistics,
) -> dict[str, Any]:
    """这一轮要写进 ``metrics.json`` 的读数（含 running 统计量，恢复训练要用它）."""
    return {
        "epoch": epoch,
        "step": step,
        "initial_loss": initial_loss,
        "train_loss": record.train_loss,
        "eval_loss": record.eval_loss,
        "eval_accuracy": record.eval_accuracy,
        "learning_rate": record.learning_rate,
        "running": {
            "mean": list(running.mean),
            "variance": list(running.variance),
            "batches": running.batches,
        },
    }


def _restore_running(payload: object, features: int) -> RunningStatistics:
    """从 ``metrics.json`` 的 running 字段还原统计量（缺字段时退回初值）."""
    if not isinstance(payload, dict):
        return initial_running(features)
    mean = tuple(float(value) for value in payload.get("mean", [0.0] * features))
    variance = tuple(float(value) for value in payload.get("variance", [1.0] * features))
    if len(mean) != features or len(variance) != features:
        raise ShapeError(
            f"检查点里的 running 宽度 {len(mean)}/{len(variance)} 与网络的宽 {features} "
            "不一致：推理相的归一化读的就是它。"
        )
    return RunningStatistics(mean=mean, variance=variance, batches=int(payload.get("batches", 0)))


def _restore_optimizer(optimizer: TrainingOptimizer, payload: dict[str, Any]) -> None:
    """把检查点里的跨步状态装回优化器（**存什么由 ``OPTIMIZER_STATE_KEYS`` 决定**）.

    ``day092`` 的 ``TrainingOptimizer`` 把共享规则的状态放在内部 ``_base``（day074 的对象）里，
    因此这里要装两处：``base`` 里的动量，以及顶层的新规则状态（nesterov / rmsprop）。
    """
    if not isinstance(payload, dict):
        raise CheckpointError(f"optimizer_state 必须是 dict，收到 {type(payload).__name__}。")
    name = payload.get("name")
    if name not in OPTIMIZER_STATE_KEYS:
        raise CheckpointError(
            f"优化器状态里的 name={name!r} 不在 {list(SUPPORTED_OPTIMIZERS)} 里："
            "不知道有哪些跨步状态，就无法把它装回去。"
        )
    if name != optimizer.name:
        raise CheckpointError(
            f"检查点里的优化器是 {name!r}，而配置要的是 {optimizer.name!r}："
            "换优化器恢复会得到一个'看着能跑、其实动量对不上'的起点。"
        )
    optimizer.step_count = int(payload.get("step_count", 0))
    inner = getattr(optimizer, "_base", None)
    base = payload.get("base")
    if inner is not None and isinstance(base, dict):
        inner.step_count = int(base.get("step_count", 0))
        for attribute in ("velocity", "first_moment", "second_moment"):
            if base.get(attribute):
                setattr(inner, attribute, tuple(float(value) for value in base[attribute]))
    for attribute in OPTIMIZER_STATE_KEYS[name]:
        if payload.get(attribute):
            setattr(optimizer, attribute, tuple(float(value) for value in payload[attribute]))


def _run_epochs(
    *,
    model: RegularizedParams,
    optimizer: TrainingOptimizer,
    running: RunningStatistics,
    loader: DataLoader,
    eval_dataset: TabularDataset,
    config: PipelineConfig,
    history: tuple[EpochRecord, ...],
    steps: int,
    start_epoch: int,
    epochs_total: int,
    checkpoint_dir: str | Path | None,
    initial_loss: float,
    plan: DevicePlan,
    fingerprint: str,
) -> PipelineReport:
    """回路的主体：从 ``start_epoch + 1`` 跑到 ``epochs_total``（训练与恢复**共用**它）."""
    best_eval = min((record.eval_loss for record in history), default=math.inf)
    no_improve = 0
    epochs_run = start_epoch
    for epoch in range(start_epoch + 1, epochs_total + 1):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch in loader.batches(epoch):
            flat_params = flatten_params(model)
            loss, grads, cache = loss_and_grad(
                model, batch_samples(batch), phase=PHASE_TRAIN, running=running
            )
            running = cache.running
            flat_grads = flatten_gradients(grads)
            check_gradient_norm(flat_grads, max_norm=GRADIENT_NORM_GUARD, step=steps + 1)
            updated = optimizer.step(flat_params, flat_grads)
            model = unflatten_params(updated, template=model)
            steps += 1
            epoch_loss_sum += loss
            epoch_batches += 1
        epochs_run = epoch
        train_loss = epoch_loss_sum / max(epoch_batches, 1)
        eval_loss, eval_accuracy = evaluate(
            model, eval_dataset, running=running, batch_size=config.batch_size
        )
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            eval_loss=eval_loss,
            eval_accuracy=eval_accuracy,
            learning_rate=optimizer.learning_rate,
        )
        history = history + (record,)
        if eval_loss < best_eval:
            best_eval = eval_loss
            no_improve = 0
        else:
            no_improve += 1
        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir,
                params=model,
                optimizer_state=optimizer.state(),
                step=steps,
                metrics=_checkpoint_metrics(
                    epoch=epoch,
                    step=steps,
                    initial_loss=initial_loss,
                    record=record,
                    running=running,
                ),
                history=history,
            )
        if config.early_stop_patience > 0 and no_improve >= config.early_stop_patience:
            break
    if not history:  # pragma: no cover - epochs_total >= 1 且 start_epoch >= 0 时必然非空
        raise ParameterError("一个 epoch 都没有跑：epochs 必须大于恢复时的起点。")
    final = history[-1]
    improvement = 0.0 if initial_loss == 0.0 else (initial_loss - final.train_loss) / abs(initial_loss)
    return PipelineReport(
        epochs_run=epochs_run,
        steps=steps,
        batch_count=len(loader),
        dropped_samples=loader.dropped_samples(),
        initial_loss=initial_loss,
        final_train_loss=final.train_loss,
        final_eval_loss=final.eval_loss,
        best_eval_loss=best_eval,
        improvement=improvement,
        device_plan=plan,
        dataset_fingerprint=fingerprint,
        history=history,
    )


def train_pipeline(
    dataset: TabularDataset,
    *,
    config: PipelineConfig | None = None,
    model: RegularizedParams | None = None,
    checkpoint_dir: str | Path | None = None,
) -> PipelineReport:
    """端到端训练一次：切分 → 造模型 → 造装载器 → 跑回路（可选落检查点）.

    ``model`` 缺省时按数据集的输入宽度造一个 ``lstm`` 分类器；给了就以它为准
    （因此调用方可以重复使用同一个模型对象，也可以换一个宽度做对照）。
    """
    if not isinstance(dataset, TabularDataset):
        raise ParameterError(f"dataset 必须是 TabularDataset，收到 {type(dataset).__name__}。")
    resolved = config if config is not None else PipelineConfig()
    if not isinstance(resolved, PipelineConfig):
        raise ParameterError(f"config 必须是 PipelineConfig，收到 {type(resolved).__name__}。")
    resolved.validate()
    train_dataset, eval_dataset = _split_for_pipeline(dataset, resolved)
    resolved_model = model if model is not None else _build_model(dataset, resolved)
    if not isinstance(resolved_model, RegularizedParams):
        raise ParameterError(f"model 必须是 RegularizedParams，收到 {type(resolved_model).__name__}。")
    plan = plan_device(
        resolved_model.parameter_count,
        batch_size=resolved.batch_size,
        width=resolved_model.features,
        device=resolved.device,
        dtype=resolved.dtype,
    )
    loader = DataLoader(
        train_dataset,
        resolved.batch_size,
        shuffle=resolved.shuffle,
        drop_last=resolved.drop_last,
        seed=resolved.seed,
        workers=resolved.workers,
    )
    optimizer = make_train_optimizer(resolved.optimizer, resolved.learning_rate)
    running = initial_running(resolved_model.features)
    initial_loss = _mean_loss(
        resolved_model, loader, epoch=1, running=running, phase=PHASE_TRAIN
    )
    return _run_epochs(
        model=resolved_model,
        optimizer=optimizer,
        running=running,
        loader=loader,
        eval_dataset=eval_dataset,
        config=resolved,
        history=(),
        steps=0,
        start_epoch=0,
        epochs_total=resolved.epochs,
        checkpoint_dir=checkpoint_dir,
        initial_loss=initial_loss,
        plan=plan,
        fingerprint=dataset.fingerprint(),
    )


def resume_training(
    dataset: TabularDataset,
    *,
    checkpoint_dir: str | Path,
    config: PipelineConfig | None = None,
    model: RegularizedParams | None = None,
) -> PipelineReport:
    """从 ``checkpoint_dir`` 的五件套恢复，继续训到 ``config.epochs``（**写回同一个目录**）.

    恢复的三件事：① 用检查点里的权重替换模型参数；② 装回优化器的跨步状态；
    ③ 从 ``metrics.json`` 记录的 epoch 号继续——打乱用的是全局 epoch 号，
    因此第 ``k+1`` 轮的批与"不中断时"逐位相同。
    """
    if not isinstance(dataset, TabularDataset):
        raise ParameterError(f"dataset 必须是 TabularDataset，收到 {type(dataset).__name__}。")
    resolved = config if config is not None else PipelineConfig()
    if not isinstance(resolved, PipelineConfig):
        raise ParameterError(f"config 必须是 PipelineConfig，收到 {type(resolved).__name__}。")
    resolved.validate()
    checkpoint = load_checkpoint(checkpoint_dir)
    train_dataset, eval_dataset = _split_for_pipeline(dataset, resolved)
    template = model if model is not None else _build_model(dataset, resolved)
    if len(checkpoint.params) != template.parameter_count:
        raise ShapeError(
            f"检查点里有 {len(checkpoint.params)} 个参数，而模板需要 "
            f"{template.parameter_count} 个：形状对不上时'按顺序切'会静默填错位置。"
        )
    restored_model = unflatten_params(checkpoint.params, template=template)
    optimizer = make_train_optimizer(resolved.optimizer, resolved.learning_rate)
    _restore_optimizer(optimizer, checkpoint.optimizer_state)
    running = _restore_running(checkpoint.metrics.get("running"), template.features)
    start_epoch = int(checkpoint.metrics.get("epoch", 0))
    initial_loss = float(checkpoint.metrics.get("initial_loss", 0.0))
    plan = plan_device(
        template.parameter_count,
        batch_size=resolved.batch_size,
        width=template.features,
        device=resolved.device,
        dtype=resolved.dtype,
    )
    loader = DataLoader(
        train_dataset,
        resolved.batch_size,
        shuffle=resolved.shuffle,
        drop_last=resolved.drop_last,
        seed=resolved.seed,
        workers=resolved.workers,
    )
    history = tuple(record_from_payload(payload) for payload in checkpoint.history)
    return _run_epochs(
        model=restored_model,
        optimizer=optimizer,
        running=running,
        loader=loader,
        eval_dataset=eval_dataset,
        config=resolved,
        history=history,
        steps=checkpoint.step,
        start_epoch=start_epoch,
        epochs_total=resolved.epochs,
        checkpoint_dir=checkpoint_dir,
        initial_loss=initial_loss,
        plan=plan,
        fingerprint=dataset.fingerprint(),
    )


def compare_configs(
    dataset: TabularDataset,
    variants: dict[str, PipelineConfig],
    **kwargs: Any,
) -> dict[str, PipelineReport]:
    """把若干份配置各训一次（**同一颗种子、同一份数据**），返回"名字 → 报告"."""
    if not variants:
        raise ParameterError("至少要给一份配置。")
    reports: dict[str, PipelineReport] = {}
    for name, config in variants.items():
        reports[str(name)] = train_pipeline(dataset, config=config, **kwargs)
    return reports


#: 从 types 转发，避免各写一遍.
__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CELL",
    "DEFAULT_EPOCHS",
    "DEFAULT_EVAL_RATIO",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SEED",
    "GRADIENT_NORM_GUARD",
    "PipelineConfig",
    "PipelineReport",
    "compare_configs",
    "evaluate",
    "resume_training",
    "train_pipeline",
]
