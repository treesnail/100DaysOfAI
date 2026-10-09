"""``train``：在一批一批的序列上真训练，并把四个旋钮逐个装上（day095 / M8-D6）.

```text
数据集    day094 的 ±1 序列任务（长度 3：8 条；长度 5：32 条）——**完全确定**
损失      批均值交叉熵（day089 的单点交叉熵 + day090 的批梯度）
优化器    day092 的 TrainingOptimizer（默认 adam，**装 day074 的调度**）
护栏      day094 的 ``check_gradient_norm``（本课只是**调用**它）
参数流转  flatten_params → optimizer.step → unflatten_params（day090 的压平契约）
两相      训练相：BN 用本批统计、dropout 置零；推理相：BN 用 running、dropout 恒等
```

## 一、为什么必须分批

```text
BatchNorm 的统计量来自批 ⇒ 批越大越稳、批大小为 1 时整层塌成常数
"一次一条"的训练在数学上没有错，但它让 BatchNorm **失去意义**
```

因此 :func:`shard_batches` 把数据集切成固定大小的批；每个 epoch 把每批过一遍，
一个 epoch 结束时用**推理相**评一次（BN 用 running、dropout 恒等）——
训练损失与推理损失是两把尺子，只报其中一把会把 dropout 的代价看反。

## 二、学习率衰减由优化器自己执行

day092 的 ``TrainingOptimizer`` 原生支持 ``set_schedule``：它在**每一步之前**
取一次学习率，因此报告里的 ``learning_rates`` 就是真用上的那一个。
本课不自己乘一个系数——那样对 Adam 是错的（Adam 的等效步长不是 lr 的线性函数）。

## 三、早停只在**推理损失**上判

```text
判据是"连续 patience 个 epoch 的推理损失没有明显变好"
⇒ 它判的是"这个模型还会不会更好"，而不是"这一批拟合得怎么样"
```

早停**只给建议**（``report.early_stop``），不自己在循环里 break——
因为"停下之后用哪一份参数"是调用方的决定（day081 把 best_step 一起交出来，正是为此）。
本课**只记录、不回滚**，并把这件事写进报告的 notes。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.math_foundations.types import Vector
from smart_research_agent.optimizers.optimizer import make_train_optimizer
from smart_research_agent.regularization import network as reg_network
from smart_research_agent.regularization.errors import ParameterError
from smart_research_agent.regularization.normalization import (
    initial_running,
    RunningStatistics,
)
from smart_research_agent.regularization.network import (
    Batch,
    RegularizedParams,
    build_regularized,
)
from smart_research_agent.regularization.techniques import (
    early_stopping_watch,
    RegularizationConfig,
)
from smart_research_agent.regularization.types import PHASE_EVAL, PHASE_TRAIN
from smart_research_agent.sequence_models.train import (
    check_gradient_norm,
    MAX_GRADIENT_NORM,
    Sample,
)
from smart_research_agent.sequence_models.types import CELL_TYPES
from smart_research_agent.training_optim.types import EarlyStopReport

#: 缺省配置（**每一项都可以单独关掉**——这就是"四个旋钮"的意思）.
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 8
DEFAULT_HIDDEN_SIZE = 4
DEFAULT_SEED = 100

#: 一次 BPTT 的梯度整体范数上界（**day094 的护栏**，本课照原样使用）.
GRADIENT_NORM_GUARD = MAX_GRADIENT_NORM


def shard_batches(dataset: tuple[Sample, ...], batch_size: int) -> tuple[Batch, ...]:
    """把数据集切成固定大小的批（**顺序切**、不洗牌：整条链因此逐位可复现）.

    最后一批可以不满——BatchNorm 只要批里有 **>= 2** 条样本就有定义
    （1 条会让方差恒为 0）。本函数因此拒绝"批大小为 1"这个配置。
    """
    if not dataset:
        raise ParameterError("空数据集切不出任何批。")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
        raise ParameterError(
            f"batch_size 必须是 >= 2 的整数，收到 {batch_size!r}："
            "批大小为 1 时 BatchNorm 的批内方差恒为 0，整层输出会被压成常数。"
        )
    return tuple(
        tuple(dataset[start : start + batch_size])
        for start in range(0, len(dataset), batch_size)
    )


@dataclass(frozen=True)
class EpochRecord:
    """一个 epoch 的读数（**训练损失与推理损失都给**）."""

    epoch: int
    train_loss: float
    eval_loss: float
    eval_accuracy: float
    learning_rate: float

    def line(self) -> str:
        """一行说明：``epoch  12 | 训练 0.123456 | 推理 0.234567 | 准确率 87.5% | lr 0.0412``."""
        return (
            f"epoch {self.epoch:>3} | 训练 {self.train_loss:.6f} | 推理 {self.eval_loss:.6f} | "
            f"准确率 {self.eval_accuracy:.1%} | lr {self.learning_rate:.6f}"
        )


@dataclass(frozen=True)
class TrainReport:
    """一次训练的全过程读数."""

    cell_type: str
    config: RegularizationConfig
    steps: int
    batch_size: int
    losses: tuple[float, ...]
    learning_rates: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    epochs: tuple[EpochRecord, ...]
    initial_loss: float
    final_loss: float
    eval_loss: float
    accuracy: float
    early_stop: EarlyStopReport | None
    running: RunningStatistics
    notes: tuple[str, ...] = field(default=())

    @property
    def improvement(self) -> float:
        """相对下降比例 ``(L₀ − L_T) / L₀``（初始损失为 0 时记 0.0）."""
        if self.initial_loss == 0.0:
            return 0.0
        return (self.initial_loss - self.final_loss) / abs(self.initial_loss)

    @property
    def max_gradient_norm(self) -> float:
        """整段训练里最大的一步梯度范数（"离爆炸还有多远"的读数）."""
        return max(self.gradient_norms) if self.gradient_norms else 0.0

    @property
    def loss_gap(self) -> float:
        """训练损失与推理损失的差（``rate=0`` 时它应当接近 0）."""
        return self.eval_loss - self.final_loss

    def summary_line(self) -> str:
        """一行说明：``rnn | 240 步 × 8 | 训练 6.9e-01 → 1.2e-02（↓98%）| 推理 8.1e-02 | 准确率 100.0%``."""
        return (
            f"{self.cell_type} | {self.steps} 步 × {self.batch_size} | "
            f"训练 {self.initial_loss:.3e} → {self.final_loss:.3e}（↓{self.improvement:.1%}）| "
            f"推理 {self.eval_loss:.3e} | 准确率 {self.accuracy:.1%}"
        )


def evaluate(
    params: RegularizedParams,
    dataset: tuple[Sample, ...],
    *,
    running: RunningStatistics,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_norm: bool = True,
) -> tuple[float, float]:
    """**推理相**地评估：BatchNorm 用 running 统计量、dropout 恒等（返回 ``(损失, 准确率)``）."""
    batches = shard_batches(dataset, batch_size)
    total = 0.0
    hits = 0
    count = 0
    for batch in batches:
        logits = reg_network.regularized_forward(
            params, batch, phase=PHASE_EVAL, running=running, use_norm=use_norm
        )
        labels = tuple(label for _inputs, label in batch)
        for row, label in zip(logits, labels, strict=True):
            total += _cross_entropy_row(row, label)
            if max(range(len(row)), key=lambda index: row[index]) == label:
                hits += 1
            count += 1
    if count == 0:  # pragma: no cover - shard_batches 已保证非空
        raise ParameterError("空数据集没有损失与准确率。")
    return total / count, hits / count


def _cross_entropy_row(logits: Vector, label: int) -> float:
    """单行交叉熵（复用 day089 的实现，**不重写**）."""
    from smart_research_agent.neural_basics.losses import cross_entropy

    return cross_entropy(logits, label)


def train_regularized(
    cell_type: str,
    dataset: tuple[Sample, ...],
    *,
    config: RegularizationConfig | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    optimizer_name: str = "adam",
    max_grad_norm: float = GRADIENT_NORM_GUARD,
) -> tuple[RegularizedParams, TrainReport]:
    """训练一个"装上旋钮"的分类器，返回 ``(训练后的参数, 报告)``.

    每一步的顺序是写死的（与 day092 的优化器内部顺序一致）：

```text
取 lr（调度在优化器内部） → 算这一步的批均值梯度 → **过 day094 的梯度范数守卫**
  → optimizer.step → unflatten
```
    """
    if cell_type not in CELL_TYPES:
        raise ParameterError(f"未知的循环单元 {cell_type!r}：可用取值 {list(CELL_TYPES)}。")
    if not dataset:
        raise ParameterError("空数据集训练不了任何东西。")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ParameterError(f"epochs 必须是 >= 1 的整数，收到 {epochs!r}。")
    resolved = config if config is not None else RegularizationConfig()
    batches = shard_batches(dataset, batch_size)
    total_steps = len(batches) * epochs
    input_size = len(dataset[0][0][0])
    params = build_regularized(
        cell_type, input_size=input_size, hidden_size=hidden_size, classes=2, seed=seed
    )
    features = params.features
    optimizer = make_train_optimizer(optimizer_name, resolved.base_lr)
    optimizer.set_schedule(resolved.schedule, **resolved.schedule_kwargs(total_steps))
    running = initial_running(features)
    warmup_loss, _warmup_acc = evaluate(params, dataset, running=running, batch_size=batch_size)
    losses: list[float] = []
    learning_rates: list[float] = []
    gradient_norms: list[float] = []
    epoch_records: list[EpochRecord] = []
    step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_index, batch in enumerate(batches):
            flat_params = reg_network.flatten_params(params)
            loss, grads, cache = reg_network.loss_and_grad(
                params,
                batch,
                phase=PHASE_TRAIN,
                running=running,
                dropout_rate=resolved.dropout_rate,
                dropout_seed=resolved.dropout_seed + epoch * 1000 + batch_index,
                use_norm=resolved.norm,
                bn_momentum=resolved.bn_momentum,
            )
            # running 统计量**跨步保留**（它就是要被"折"进历史的那一份）
            running = cache.running
            flat_grads = reg_network.flatten_gradients(grads)
            gradient_norms.append(
                check_gradient_norm(flat_grads, max_norm=max_grad_norm, step=step + 1)
            )
            updated = optimizer.step(flat_params, flat_grads)
            params = reg_network.unflatten_params(updated, template=params)
            step += 1
            losses.append(loss)
            learning_rates.append(optimizer.learning_rate)
            epoch_loss += loss
        eval_loss, eval_accuracy = evaluate(
            params, dataset, running=running, batch_size=batch_size, use_norm=resolved.norm
        )
        epoch_records.append(
            EpochRecord(
                epoch=epoch,
                train_loss=epoch_loss / len(batches),
                eval_loss=eval_loss,
                eval_accuracy=eval_accuracy,
                learning_rate=optimizer.learning_rate,
            )
        )
    early_stop: EarlyStopReport | None = None
    if resolved.patience > 0:
        early_stop = early_stopping_watch(
            tuple(record.eval_loss for record in epoch_records),
            patience=resolved.patience,
            min_delta=resolved.min_delta,
        )
    final_eval_loss, final_accuracy = evaluate(
        params, dataset, running=running, batch_size=batch_size, use_norm=resolved.norm
    )
    report = TrainReport(
        cell_type=cell_type,
        config=resolved,
        steps=total_steps,
        batch_size=batch_size,
        losses=tuple(losses),
        learning_rates=tuple(learning_rates),
        gradient_norms=tuple(gradient_norms),
        epochs=tuple(epoch_records),
        initial_loss=warmup_loss,
        final_loss=losses[-1],
        eval_loss=final_eval_loss,
        accuracy=final_accuracy,
        early_stop=early_stop,
        running=running,
        notes=(
            "梯度来自 day094 的 BPTT（本课没有重写任何反向）",
            "BatchNorm 的反向是本课新写的（normalization.batch_norm_backward），训练相三项",
            "dropout 与早停、调度分别转发 day081 与 day074",
            "每一步都过 day094 的 check_gradient_norm（本课只是调用它）",
            "训练损失与推理损失都给：只看训练损失会把 dropout 的代价看反",
            f"初始损失是在**未更新过**的参数上、用推理相算的整轮均值",
        ),
    )
    return params, report


def compare_configs(
    cell_type: str,
    dataset: tuple[Sample, ...],
    variants: dict[str, RegularizationConfig],
    **kwargs,
) -> dict[str, TrainReport]:
    """把若干份配置各训一次（**同一颗种子、同一份数据**），返回"名字 → 报告"."""
    if not variants:
        raise ParameterError("至少要给一份配置。")
    reports: dict[str, TrainReport] = {}
    for name, config in variants.items():
        _params, report = train_regularized(
            cell_type, dataset, config=config, **kwargs
        )
        reports[str(name)] = report
    return reports


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_SEED",
    "GRADIENT_NORM_GUARD",
    "EpochRecord",
    "TrainReport",
    "compare_configs",
    "evaluate",
    "shard_batches",
    "train_regularized",
]
