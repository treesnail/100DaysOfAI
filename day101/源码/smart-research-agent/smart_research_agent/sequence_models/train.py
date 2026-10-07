"""``train``：在一个**合成的序列任务**上真的训练一次（day094 / M8-D5）.

```text
数据集    长度 T=5 的 ±1 序列，两类：和为正 vs 和为负（D=1，每一步一个标量）
损失      day089 的交叉熵（单个样本 -log p[标签]）
优化器    day092 的 TrainingOptimizer（默认 adam）——**同一套更新规则**
参数流转  flatten_params → optimizer.step → unflatten_params（day090 的压平契约）
护栏      **check_gradient_norm**：一次 BPTT 的范数越界 ⇒ 抛 GradientError（当场停下）
```

## 为什么用这个任务

```text
① 可复现    序列是枚举出来的（2^5 = 32 条），没有随机性
② 可判定    标签 = "和的正负"，而 RNN 的隐状态天然可以做一个累加器 ⇒ 学会了就是学会了
③ 有对照    同一个任务在 rnn 与 lstm 上各训一次，两条曲线直接可比
```

长度取 **5（奇数）**：这样"和"永远不为 0，"正 / 负"两类的边界不会落在未定义的地方。
标签按 `和 > 0 ⇒ 1` 定义，两类各 16 条（完全平衡）。

## 这一课不承诺的事

它不承诺"任意任务都能训到 100%"：这是一个 **H=4 的单元 + 一个线性分类头**
的最小网络，数据只有 32 条、序列只有 5 步。它的目标是让整条链
**跑通、可对账、可复现**，并把"梯度爆炸"这条护栏装上去——
真正的长序列依赖（几百步）需要截断 BPTT 与门控结构（day095 / day096 再谈）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.math_foundations.types import Vector
from smart_research_agent.optimizers.optimizer import make_train_optimizer
from smart_research_agent.sequence_models.errors import GradientError, ParameterError
from smart_research_agent.sequence_models.network import (
    SequenceGradients,
    SequenceNetParams,
    build_sequence_net,
    flatten_gradients,
    flatten_params,
    loss_and_grad,
    predict,
    unflatten_params,
)
from smart_research_agent.sequence_models.types import CELL_TYPES

#: 数据集与网络的缺省尺寸（写进常量：换一个尺寸，所有读数都会变）.
DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_HIDDEN_SIZE = 4
DEFAULT_STEPS = 250
DEFAULT_LEARNING_RATE = 0.05

#: 一次 BPTT 的梯度整体范数上界（**训练期的护栏**：超过它就抛 ``GradientError``）.
#:
#: 这个数不是"越大越好"：范数 50 已经意味着某一步的更新会把权重推到它原来的几十倍之外。
#: 它进的是**函数参数**而不是配置项——"这次训练允许多大的梯度"是调用点的一次决定。
MAX_GRADIENT_NORM = 50.0

#: 一条样本：`(T 步的输入, 标签)`.
Sample = tuple[tuple[Vector, ...], int]


def make_sign_sample(values: tuple[int, ...]) -> Sample:
    """把一串 ``±1`` 变成一条样本：标签 1 表示和为正.

    每一步的输入是**长度 1** 的向量 ``(value,)``——把"一个标量序列"写成"一串 D=1 的向量"
    是为了与"输入维 D"这个通用口径一致（换成一个更大的 D 也不改任何一行实现）。
    """
    if not values:
        raise ParameterError("一条序列至少要有一个符号。")
    cleaned: list[Vector] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or value not in (1, -1):
            raise ParameterError(f"第 {index} 个符号必须是 +1 或 −1，收到 {value!r}。")
        cleaned.append((float(value),))
    total = sum(values)
    if total == 0:
        raise ParameterError(
            "这条序列的和恰好是 0：本课的标签是「和的正负」，"
            "而 0 不属于任何一类——请把长度取成奇数。"
        )
    return tuple(cleaned), 1 if total > 0 else 0


def make_sign_dataset(length: int = DEFAULT_SEQUENCE_LENGTH) -> tuple[Sample, ...]:
    """枚举全部 ``2^length`` 条 ``±1`` 序列（**完全确定，没有随机性**）.

    长度必须是**奇数**：偶数长度会出现"和为 0"的序列，而它的标签没有定义。
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ParameterError(f"序列长度必须是 >= 1 的整数，收到 {length!r}。")
    if length % 2 == 0:
        raise ParameterError(
            f"序列长度必须是奇数，收到 {length}：偶数长度会出现和为 0 的序列，"
            "而「和的正负」在 0 处没有定义。"
        )
    samples: list[Sample] = []
    for mask in range(2**length):
        values = tuple(1 if (mask >> index) & 1 else -1 for index in range(length))
        samples.append(make_sign_sample(values))
    return tuple(samples)


def check_gradient_norm(
    values: Vector,
    *,
    max_norm: float = MAX_GRADIENT_NORM,
    step: int = 0,
) -> float:
    """**本课的运行期守卫**：一次 BPTT 的梯度范数非有限或超过 ``max_norm`` ⇒ 抛 ``GradientError``.

    它是 ``GradientError`` 今天"回来"的地方：

    ```text
    day090  抛它是因为"解析梯度与数值差分对不上"   （离线对账的结论）
    day094  抛它是因为"梯度真的爆了"               （这一步就不该继续）
    ```

    返回范数本身，因此调用方可以把它记进训练报告（"哪一步开始不对"是可读的）。
    """
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ParameterError(f"max_norm 必须是正的有限数，收到 {max_norm!r}。")
    checked = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in checked):
        raise GradientError(
            f"第 {step} 步的梯度里出现非有限数（nan / inf）："
            "一次 BPTT 把同一个循环权重乘了 T 次，爆炸之后所有下游读数都会变成 nan。"
            "先降学习率、缩短展开长度，或加上裁剪。"
        )
    norm = math.sqrt(math.fsum(value * value for value in checked))
    if norm > max_norm:
        raise GradientError(
            f"第 {step} 步的梯度范数 {norm:.6e} 超过上界 {max_norm:.6g}："
            "BPTT 的范数随展开长度指数增长，一次越界就足以把参数甩出可训练的区域。"
            "先降学习率 / 缩短展开长度 / 加裁剪。"
        )
    return norm


@dataclass(frozen=True)
class TrainReport:
    """一次训练的全过程读数."""

    cell_type: str
    steps: int
    losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    initial_loss: float
    final_loss: float
    best_loss: float
    accuracy: float
    optimizer: str
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

    def summary_line(self) -> str:
        """一行说明：``rnn | 400 步 | 损失 6.9e-01 → 1.2e-02（↓98.3%）| 准确率 100.0% | adam``."""
        return (
            f"{self.cell_type} | {self.steps} 步 | 损失 {self.initial_loss:.3e} → "
            f"{self.final_loss:.3e}（↓{self.improvement:.1%}）| 准确率 {self.accuracy:.1%} | "
            f"{self.optimizer}"
        )


def mean_loss(
    params: SequenceNetParams, dataset: tuple[Sample, ...]
) -> float:
    """数据集上的平均交叉熵（**整轮的读数**，而不是单样本的读数）."""
    if not dataset:
        raise ParameterError("空数据集没有损失。")
    total = 0.0
    for inputs, label in dataset:
        loss, _grads = loss_and_grad(params, inputs, label)
        total += loss
    return total / len(dataset)


def accuracy(params: SequenceNetParams, dataset: tuple[Sample, ...]) -> float:
    """数据集上的分类准确率（预测 = logits 的最大值下标）."""
    if not dataset:
        raise ParameterError("空数据集没有准确率。")
    hits = 0
    for inputs, label in dataset:
        if predict(params, inputs) == label:
            hits += 1
    return hits / len(dataset)


def _average_gradients(
    params: SequenceNetParams, dataset: tuple[Sample, ...]
) -> tuple[float, Vector]:
    """把整个数据集的梯度求平均，返回 ``(损失之和, 平均梯度)``.

    损失用 ``fsum`` 累加之后**不在这里除**——调用方需要"和"来决定要不要再算均值，
    而"梯度求平均"这件事只应该有一处实现。
    """
    flat = flatten_params(params)
    accumulator = [0.0] * len(flat)
    total_loss = 0.0
    for inputs, label in dataset:
        loss, grads = loss_and_grad(params, inputs, label)
        total_loss += loss
        for index, value in enumerate(flatten_gradients(grads)):
            accumulator[index] += value
    count = len(dataset)
    return total_loss, tuple(value / count for value in accumulator)


def gradient_and_loss(
    params: SequenceNetParams, dataset: tuple[Sample, ...]
) -> tuple[float, SequenceGradients, Vector]:
    """一个便于测试的复合读数：``(平均损失, 最后一条样本的梯度, 平均梯度的压平向量)``."""
    total_loss, flat = _average_gradients(params, dataset)
    _last_loss, last_grads = loss_and_grad(params, *dataset[-1])
    return total_loss / len(dataset), last_grads, flat


def train_sequence(
    cell_type: str,
    dataset: tuple[Sample, ...],
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    optimizer_name: str = "adam",
    seed: int = 100,
    max_grad_norm: float = MAX_GRADIENT_NORM,
    **optimizer_kwargs,
) -> tuple[SequenceNetParams, TrainReport]:
    """训练一个序列分类器，返回 ``(训练后的参数, 报告)``.

    每一步都是**全批量**（把整个数据集的梯度求平均再走一步）：32 条样本既便宜又稳定——
    它把"单样本 SGD 的抖动"这个与循环结构无关的变量先消掉。
    损失曲线每一步记一条（该步**更新前**的整轮均值），梯度范数也逐步记下。
    """
    if cell_type not in CELL_TYPES:
        raise ParameterError(f"未知的循环单元 {cell_type!r}：可用取值 {list(CELL_TYPES)}。")
    if not dataset:
        raise ParameterError("空数据集训练不了任何东西。")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ParameterError(f"steps 必须是 >= 1 的整数，收到 {steps!r}。")
    input_size = len(dataset[0][0][0])
    params = build_sequence_net(
        cell_type, input_size=input_size, hidden_size=hidden_size, classes=2, seed=seed
    )
    optimizer = make_train_optimizer(optimizer_name, learning_rate, **optimizer_kwargs)
    initial_loss = mean_loss(params, dataset)
    losses: list[float] = []
    norms: list[float] = []
    for step in range(steps):
        flat_params = flatten_params(params)
        total_loss, flat_grads = _average_gradients(params, dataset)
        losses.append(total_loss / len(dataset))
        # **护栏**：进优化器之前先看范数（爆炸当场停下，而不是让 nan 传下去）
        norms.append(check_gradient_norm(flat_grads, max_norm=max_grad_norm, step=step + 1))
        updated = optimizer.step(flat_params, flat_grads)
        params = unflatten_params(updated, template=params)
    final_accuracy = accuracy(params, dataset)
    report = TrainReport(
        cell_type=cell_type,
        steps=steps,
        losses=tuple(losses),
        gradient_norms=tuple(norms),
        initial_loss=initial_loss,
        final_loss=losses[-1],
        best_loss=min(losses),
        accuracy=final_accuracy,
        optimizer=optimizer_name,
        notes=(
            "梯度来自本包的 BPTT（gradients.rnn_bptt / lstm_bptt）——不是数值差分",
            "优化器复用 day092 的 TrainingOptimizer，参数按 day090 的压平契约来回搬运",
            "每一步是全批量：把 32 条样本的梯度求平均再走一步（先消掉单样本抖动）",
            "每一步都过 check_gradient_norm：范数越界当场抛 GradientError，把 nan 挡在门外",
        ),
    )
    return params, report


__all__ = [
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_STEPS",
    "MAX_GRADIENT_NORM",
    "Sample",
    "TrainReport",
    "accuracy",
    "check_gradient_norm",
    "gradient_and_loss",
    "make_sign_dataset",
    "make_sign_sample",
    "mean_loss",
    "train_sequence",
]
