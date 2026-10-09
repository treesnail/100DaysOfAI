"""``study``：五张表（day089 / M8-D1）.

每一天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天五张表各回答一个"从神经元到 FFN 这条链搭对了吗"的问题：

```text
激活表      六个激活：是否饱和 / 是否零中心 / 值域 / 公式 / 现场读数
初始化表    五种初始化：说明 / 首元素 / 首偏置 / 权重个数（同一份 spec、同一个种子）
层次账表    一个 4→8→3 的网络：逐层的形状、权重数、偏置数、参数总数
损失表      三个损失：说明 / 现场读数
性质表      七条性质：是否通过 / 现场读数（读数来自跨天对账或结构事实）
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：**读数**与**参照**（口径 / 形状 / 期望 / 跨包对象）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.neural_basics import activations, layers, losses, verify
from smart_research_agent.neural_basics.types import (
    ACTIVATIONS,
    ACTIVATION_PROFILES,
    INITIALIZATIONS,
    INITIALIZATION_DESCRIPTIONS,
    LOSSES,
    LOSS_CROSS_ENTROPY,
    LOSS_DESCRIPTIONS,
    LOSS_MAE,
    LOSS_MSE,
    DenseSpec,
    MLPSpec,
)

#: 初始化表的样本层：``4 → 3``、种子固定，因此每个读数都可重新跑出来.
INIT_SAMPLE = DenseSpec(in_features=4, out_features=3, activation=None, init="xavier", seed=5)

#: 层次账表的样本网络：``4 → 8 → 3``（第一层 relu、第二层恒等）.
LAYER_SAMPLE = MLPSpec(
    layers=(
        DenseSpec(4, 8, activation="relu", init="xavier", seed=21),
        DenseSpec(8, 3, activation=None, init="xavier", seed=22),
    )
)

#: 损失表的样本：mse / mae 用预测-目标对，交叉熵用一行 logits + 标签.
LOSS_PRED: tuple[tuple[float, ...], ...] = ((1.0, 2.0), (3.0, 4.0))
LOSS_TARGET: tuple[tuple[float, ...], ...] = ((1.5, 1.5), (4.0, 3.0))
LOSS_LOGITS: tuple[float, ...] = (1.0, 2.0, 3.0)
LOSS_TARGET_INDEX = 0

#: 激活表的读数点（软读数：逐元素激活取 ``x=1.0``；softmax 取一整行）.
ACTIVATION_SAMPLE_X = 1.0
ACTIVATION_SOFTMAX_ROW: tuple[float, ...] = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ActivationRow:
    """激活表的一行：口径 + 现场读数."""

    name: str
    profile: str
    reading: str

    def line(self) -> str:
        """``relu        饱和 否 | 零中心 否 | 值域 [0, +inf) | 公式 max(0, x) | 读数 1``."""
        return f"{self.name:<11} {self.profile} | 读数 {self.reading}"


@dataclass(frozen=True)
class InitializationRow:
    """初始化表的一行：说明 + 首元素读数."""

    name: str
    description: str
    first_weight: float
    first_bias: float
    weight_count: int

    def line(self) -> str:
        """``xavier     W[0][0]=... b[0]=... 权重 12 个 | ...``."""
        return (
            f"{self.name:<9} W[0][0]={self.first_weight:+.6f} b[0]={self.first_bias:+.6f} "
            f"权重 {self.weight_count} 个 | {self.description}"
        )


@dataclass(frozen=True)
class LayerRow:
    """层次账表的一行：一层的形状与参数."""

    index: int
    in_features: int
    out_features: int
    weight_shape: str
    weights: int
    biases: int
    total: int

    def line(self) -> str:
        """``第 1 层  dense(4→8) | W (8, 4) | 权重 32 + 偏置 8 = 40``."""
        return (
            f"第 {self.index} 层  dense({self.in_features}→{self.out_features}) | "
            f"W {self.weight_shape} | 权重 {self.weights} + 偏置 {self.biases} = {self.total}"
        )


@dataclass(frozen=True)
class LossRow:
    """损失表的一行：说明 + 现场读数."""

    name: str
    description: str
    reading: float

    def line(self) -> str:
        """``mse = 6.250000e-01 | ...``."""
        return f"{self.name:<14} = {self.reading:.6e} | {self.description}"


@dataclass(frozen=True)
class PropertyRow:
    """性质表的一行：是否通过 + 现场读数 + 跨包对象."""

    name: str
    passed: bool
    reading: float
    cross_check: str

    def line(self) -> str:
        """``通过 activations_are_finite | 读数 0 | ...``."""
        mark = "通过" if self.passed else "失败"
        return f"{mark} {self.name:<38} | 读数 {self.reading:g} | {self.cross_check}"


def activation_rows() -> tuple[ActivationRow, ...]:
    """激活表：六个激活逐行（读数来自 ``activations``）."""
    rows: list[ActivationRow] = []
    for name in ACTIVATIONS:
        profile = ACTIVATION_PROFILES[name]
        if name == "softmax":
            distribution = activations.softmax(ACTIVATION_SOFTMAX_ROW)
            reading = f"行和 {sum(distribution):.12f}"
        else:
            reading = f"act({ACTIVATION_SAMPLE_X:g}) = {activations.activate(name, (ACTIVATION_SAMPLE_X,))[0]:+.6f}"
        rows.append(ActivationRow(name=name, profile=profile.line(), reading=reading))
    return tuple(rows)


def initialization_rows() -> tuple[InitializationRow, ...]:
    """初始化表：五种初始化逐行（同一份 spec、同一个种子）."""
    rows: list[InitializationRow] = []
    for name in INITIALIZATIONS:
        weight, bias = layers.initialize(
            name, INIT_SAMPLE.out_features, INIT_SAMPLE.in_features, seed=INIT_SAMPLE.seed
        )
        rows.append(
            InitializationRow(
                name=name,
                description=INITIALIZATION_DESCRIPTIONS[name],
                first_weight=weight[0][0],
                first_bias=bias[0],
                weight_count=INIT_SAMPLE.weight_count,
            )
        )
    return tuple(rows)


def layer_rows(spec: MLPSpec | None = None) -> tuple[LayerRow, ...]:
    """层次账表：逐层的形状与参数（样本网络默认 ``4 → 8 → 3``）."""
    resolved = LAYER_SAMPLE if spec is None else spec
    rows: list[LayerRow] = []
    for index, layer_spec in enumerate(resolved.layers, start=1):
        accounting = layers.layer_accounting(layer_spec)
        rows.append(
            LayerRow(
                index=index,
                in_features=layer_spec.in_features,
                out_features=layer_spec.out_features,
                weight_shape=str(accounting["weight_shape"]),
                weights=int(accounting["weights"]),
                biases=int(accounting["biases"]),
                total=int(accounting["total"]),
            )
        )
    return tuple(rows)


def loss_rows() -> tuple[LossRow, ...]:
    """损失表：三个损失逐行（现场读数）."""
    values: dict[str, float] = {
        LOSS_MSE: losses.mse(LOSS_PRED, LOSS_TARGET),
        LOSS_MAE: losses.mae(LOSS_PRED, LOSS_TARGET),
        LOSS_CROSS_ENTROPY: losses.cross_entropy(LOSS_LOGITS, LOSS_TARGET_INDEX),
    }
    return tuple(
        LossRow(name=name, description=LOSS_DESCRIPTIONS[name], reading=values[name])
        for name in LOSSES
    )


def property_rows() -> tuple[PropertyRow, ...]:
    """性质表：七条性质逐行（读数来自 :func:`verify.check_all`）."""
    report = verify.check_all()
    rows: list[PropertyRow] = []
    for outcome in report.outcomes:
        reading = outcome.cross_check.reading if outcome.cross_check is not None else 0.0
        cross_check = f"{outcome.cross_check.left} vs {outcome.cross_check.right}" if outcome.cross_check else "-"
        rows.append(
            PropertyRow(
                name=outcome.name,
                passed=outcome.passed,
                reading=reading,
                cross_check=cross_check,
            )
        )
    return tuple(rows)


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """十条笔记逐行印出（顺序即写入顺序）."""
    from smart_research_agent.neural_basics.types import NEURAL_NOTES, NEURAL_NOTES_ORDER

    keys = NEURAL_NOTES_ORDER if limit is None else NEURAL_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {NEURAL_NOTES[key]}" for index, key in enumerate(keys, start=1))


def study_lines() -> tuple[str, ...]:
    """一次跑完五张表（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("== 1. 激活表（是否饱和 / 是否零中心 / 值域 / 公式 / 读数）")
    for row in activation_rows():
        lines.append("  " + row.line())
    lines.append("== 2. 初始化表（说明 / 首元素 / 首偏置 / 权重个数）")
    for row in initialization_rows():
        lines.append("  " + row.line())
    lines.append("== 3. 层次账表（4→8→3：逐层的形状与参数）")
    for row in layer_rows():
        lines.append("  " + row.line())
    lines.append(f"  参数总数 {LAYER_SAMPLE.parameter_count}（{LAYER_SAMPLE.line()}）")
    lines.append("== 4. 损失表（三个损失的现场读数）")
    for row in loss_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 性质表（七条性质：是否通过 / 读数 / 跨包对象）")
    for row in property_rows():
        lines.append("  " + row.line())
    return tuple(lines)


__all__ = [
    "ACTIVATION_SAMPLE_X",
    "ACTIVATION_SOFTMAX_ROW",
    "INIT_SAMPLE",
    "LAYER_SAMPLE",
    "LOSS_LOGITS",
    "LOSS_PRED",
    "LOSS_TARGET",
    "LOSS_TARGET_INDEX",
    "ActivationRow",
    "InitializationRow",
    "LayerRow",
    "LossRow",
    "PropertyRow",
    "activation_rows",
    "initialization_rows",
    "layer_rows",
    "loss_rows",
    "note_lines",
    "property_rows",
    "study_lines",
]
