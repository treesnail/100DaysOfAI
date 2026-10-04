"""``train``：在一个**合成的小图像集**上真的训练一次 CNN（day093 / M8-D4）.

```text
数据集    1×6×6 的灰度图，两类：竖线 vs 横线（LCG 加确定性噪声）
损失      day089 的交叉熵（单个样本 -log p[标签]）
优化器    day092 的 TrainingOptimizer（默认 adam）——**同一套更新规则**
参数流转  flatten_params → optimizer.step → unflatten_params（day090 的压平契约）
```

## 为什么用合成数据

真实图像集（MNIST / CIFAR）需要下载与解析，而这一课要讲的是**卷积本身**：
权重共享、局部连接、感受野、反向。合成数据有两个直接好处：

```text
① 可复现    图像由 LCG 生成 ⇒ 每一次训练曲线都逐位相同（不需要随机种子运气）
② 可判定    "竖线 vs 横线"是局部纹理差异，卷积核恰好能学到方向性 —— 学会了就是学会了
```

## 这一课不承诺的事

它不承诺"训练到 100% 准确率"：这是一个**单层卷积 + 2×2 池化 + 一个小全连接头**
的最小网络，数据集也只有若干个样本。它的目标是让整条链**跑通、可对账、可复现**，
而不是刷一个准确率数字——那需要更大的网络与更多的数据（day095 / day096 再谈）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.conv_net.errors import ParameterError
from smart_research_agent.conv_net.layers import ConvSpec
from smart_research_agent.conv_net.network import (
    CNNParams,
    build_cnn,
    cnn_forward,
    flatten_gradients,
    flatten_params,
    loss_and_grad,
    unflatten_params,
)
from smart_research_agent.conv_net.ops import as_matrix
from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.neural_basics.layers import lcg_stream
from smart_research_agent.optimizers.optimizer import make_train_optimizer

#: 数据集与网络的缺省尺寸（写进常量：换一个尺寸，所有读数都会变）.
DEFAULT_IMAGE_SIZE = 6
DEFAULT_POOL_WINDOW = 2

#: 一条样本：`(单通道图像, 标签)`.
Sample = tuple[Matrix, int]


def make_stripe_sample(size: int, label: int, *, noise: float = 0.0, seed: int = 0) -> Sample:
    """造一张 ``size×size`` 的图：标签 0 = 竖线，1 = 横线（可加确定性噪声）.

    线条放在**中间那一行 / 一列**；噪声来自 day089 的 LCG，因此整张图逐位可复现。
    """
    if label not in (0, 1):
        raise ParameterError(f"stripe 数据集只有两个类别（0=竖线 / 1=横线），收到 {label}。")
    if size < 3:
        raise ParameterError(f"图像尺寸必须 >= 3，收到 {size}（太小了放不下一根线）。")
    if not math.isfinite(noise) or noise < 0.0:
        raise ParameterError(f"noise 必须是非负的有限数，收到 {noise!r}。")
    jitter = lcg_stream(seed, size * size) if noise > 0 else (0.0,) * (size * size)
    centre = size // 2
    rows: list[tuple[float, ...]] = []
    for row in range(size):
        values: list[float] = []
        for column in range(size):
            base = 1.0 if ((row == centre) if label == 1 else (column == centre)) else 0.0
            values.append(base + (jitter[row * size + column] * 2.0 - 1.0) * noise)
        rows.append(tuple(values))
    return tuple(rows), label


def make_stripe_dataset(
    size: int = DEFAULT_IMAGE_SIZE, *, per_class: int = 4, noise: float = 0.0, seed: int = 0
) -> tuple[Sample, ...]:
    """造一个两类的小数据集（每类 ``per_class`` 条，噪声按样本序号错开种子）."""
    if per_class < 1:
        raise ParameterError(f"per_class 必须 >= 1，收到 {per_class}。")
    samples: list[Sample] = []
    for label in (0, 1):
        for index in range(per_class):
            samples.append(
                make_stripe_sample(size, label, noise=noise, seed=seed + label * 100 + index)
            )
    return tuple(samples)


@dataclass(frozen=True)
class TrainReport:
    """一次训练的全过程读数."""

    steps: int
    losses: tuple[float, ...]
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

    def summary_line(self) -> str:
        """一行说明：``60 步 | 损失 6.93e-01 → 1.2e-02（↓98.3%）| 准确率 100.0% | adam``."""
        return (
            f"{self.steps} 步 | 损失 {self.initial_loss:.3e} → {self.final_loss:.3e}"
            f"（↓{self.improvement:.1%}）| 准确率 {self.accuracy:.1%} | {self.optimizer}"
        )


def _as_channels(image: Matrix) -> tuple[Matrix, ...]:
    """单通道图 → 通道元组（本课的网络是单通道输入）."""
    return (as_matrix(image, name="image"),)


def accuracy(
    params: CNNParams, spec: ConvSpec, dataset: tuple[Sample, ...], *, pool_window: int = DEFAULT_POOL_WINDOW
) -> float:
    """数据集上的分类准确率（预测 = logits 的最大值下标）."""
    if not dataset:
        raise ParameterError("空数据集没有准确率。")
    hits = 0
    for image, label in dataset:
        logits = cnn_forward(params, spec, _as_channels(image), pool_window=pool_window)
        if max(range(len(logits)), key=lambda index: logits[index]) == label:
            hits += 1
    return hits / len(dataset)


def mean_loss(
    params: CNNParams, spec: ConvSpec, dataset: tuple[Sample, ...], *, pool_window: int = DEFAULT_POOL_WINDOW
) -> float:
    """数据集上的平均交叉熵（**整轮的读数**，而不是单样本的读数）."""
    if not dataset:
        raise ParameterError("空数据集没有损失。")
    total = 0.0
    for image, label in dataset:
        loss, _grads = loss_and_grad(params, spec, _as_channels(image), label, pool_window=pool_window)
        total += loss
    return total / len(dataset)


def train_cnn(
    spec: ConvSpec,
    dataset: tuple[Sample, ...],
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    pool_window: int = DEFAULT_POOL_WINDOW,
    steps: int = 200,
    learning_rate: float = 0.05,
    optimizer_name: str = "adam",
    seed: int = 100,
    **optimizer_kwargs,
) -> tuple[CNNParams, TrainReport]:
    """训练一个小 CNN，返回 ``(训练后的参数, 报告)``.

    每一步都是**全批量**（把整个小数据集的梯度求平均再走一步）：数据集只有若干条，
    全批量既便宜又稳定——它把"单样本 SGD 的抖动"这个与卷积无关的变量先消掉，
    让这一课的读数只在讲"卷积学没学会"。损失曲线每一步记一条（该步**更新前**的整轮均值）。
    """
    if not dataset:
        raise ParameterError("空数据集训练不了任何东西。")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ParameterError(f"steps 必须是 >= 1 的整数，收到 {steps!r}。")
    params = build_cnn(spec, (image_size, image_size), classes=2, pool_window=pool_window, seed=seed)
    optimizer = make_train_optimizer(optimizer_name, learning_rate, **optimizer_kwargs)
    # 起点损失：在**未被更新过**的初始参数上算整轮均值（"起点在哪"要可比）
    initial_loss = mean_loss(params, spec, dataset, pool_window=pool_window)
    losses: list[float] = []
    for _step in range(steps):
        flat_params = flatten_params(params)
        accumulator = [0.0] * len(flat_params)
        total_loss = 0.0
        for image, label in dataset:
            loss, grads = loss_and_grad(params, spec, _as_channels(image), label, pool_window=pool_window)
            total_loss += loss
            flat_grads = flatten_gradients(grads)
            for index, value in enumerate(flat_grads):
                accumulator[index] += value
        losses.append(total_loss / len(dataset))
        average_grads = tuple(value / len(dataset) for value in accumulator)
        updated = optimizer.step(flat_params, average_grads)
        params = unflatten_params(updated, template=params)
    final_accuracy = accuracy(params, spec, dataset, pool_window=pool_window)
    report = TrainReport(
        steps=steps,
        losses=tuple(losses),
        initial_loss=initial_loss,
        final_loss=losses[-1],
        best_loss=min(losses),
        accuracy=final_accuracy,
        optimizer=optimizer_name,
        notes=(
            "梯度来自本包的解析反向（gradients.conv_block_backward / pool2d_backward）——不是数值差分",
            "优化器复用 day092 的 TrainingOptimizer，参数按 day090 的压平契约来回搬运",
            "每一步是全批量：把整个小数据集的梯度求平均再走一步（先消掉单样本抖动）",
            f"初始损失是整轮均值（{len(dataset)} 条），最终损失是最后一步**更新前**的整轮均值",
        ),
    )
    return params, report


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_POOL_WINDOW",
    "Sample",
    "TrainReport",
    "accuracy",
    "make_stripe_dataset",
    "make_stripe_sample",
    "mean_loss",
    "train_cnn",
]
