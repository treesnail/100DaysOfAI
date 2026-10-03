"""``study``：六张表（day093 / M8-D4）.

```text
① 层表       一层的画像：通道 / 核 / 步长 / 填充 / 激活 / 参数 / 输出尺寸
② 尺寸表     输出尺寸公式在几组 (核, 步长, 填充, 膨胀) 上的读数
③ 感受野表   几组层序列的递推公式读数（"堆两层 3×3"为什么等于 5×5）
④ 特征图表   一次卷积之后的特征图画像（形状 / 均值 / 最大值 / 非零比例）
⑤ 池化表     max / avg 在同一次输入上的读数（降采样率与"保留了什么"）
⑥ 训练表     合成数据集上真训练一次的损失曲线与准确率
⑦ 性质表     七条性质是否通过 / 现场读数 / 两个来源
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
每张表的每一行至少有"读数"与"参照"两列。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.conv_net import verify
from smart_research_agent.conv_net.layers import ConvSpec, initialize_conv
from smart_research_agent.conv_net.ops import (
    conv2d,
    feature_size,
    feature_stats,
    output_size,
    pool2d,
    receptive_field,
)
from smart_research_agent.conv_net.train import TrainReport, make_stripe_dataset, train_cnn
from smart_research_agent.conv_net.types import (
    CONV_PROPERTIES,
    PADDING_SAME,
    POOL_AVG,
    POOL_MAX,
)

#: ① / ④ 用的样本层（1→2 通道、3×3 核、same 填充、relu、种子固定）.
SAMPLE_SPEC = ConvSpec(1, 2, 3, padding=PADDING_SAME, activation="relu", seed=3)

#: ① / ④ 用的样本图（6×6 的竖线图）.
SAMPLE_IMAGE: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
)

#: ② 用的几组参数（核, 步长, 填充, 膨胀）.
SIZE_CASES: tuple[tuple[int, int, int, int], ...] = (
    (3, 1, 0, 1),
    (3, 1, 1, 1),
    (3, 2, 1, 1),
    (2, 2, 0, 1),
    (3, 1, 1, 2),
)

#: ③ 用的几组层序列.
FIELD_CASES: dict[str, tuple[tuple[int, int], ...]] = {
    "一层 3×3": ((3, 1),),
    "两层 3×3": ((3, 1), (3, 1)),
    "三层 3×3": ((3, 1), (3, 1), (3, 1)),
    "两层 3×3 + 池化 2×2": ((3, 1), (3, 1), (2, 2)),
}

#: ⑥ 用的训练配置.
TRAIN_SPEC = ConvSpec(1, 2, 3, padding=PADDING_SAME, activation="relu", seed=3)
TRAIN_STEPS = 200
TRAIN_LR = 0.05


@dataclass(frozen=True)
class LayerRow:
    """① 层表的一行."""

    description: str
    output: str
    parameters: int

    def line(self) -> str:
        """``conv(1→2, k=3, s=1, pad=same, act=relu) | 参数 20 | 输出 2×6×6``."""
        return f"{self.description} | 输出 {self.output}"


@dataclass(frozen=True)
class SizeRow:
    """② 尺寸表的一行."""

    kernel: int
    stride: int
    padding: int
    dilation: int
    size: int

    def line(self) -> str:
        """``k=3 s=1 p=1 d=1 | 6 → 6``."""
        return (
            f"k={self.kernel} s={self.stride} p={self.padding} d={self.dilation} | 6 → {self.size}"
        )


@dataclass(frozen=True)
class FieldRow:
    """③ 感受野表的一行."""

    name: str
    field: int
    direct: int

    def line(self) -> str:
        """``两层 3×3 | 感受野 5 | 区间传播 5``."""
        return f"{self.name:<22} | 感受野 {self.field} | 区间传播 {self.direct}"


@dataclass(frozen=True)
class FeatureRow:
    """④ 特征图表的一行."""

    channel: int
    height: int
    width: int
    mean: float
    maximum: float
    nonzero: float

    def line(self) -> str:
        """``通道 0 | 6×6 | 均值 0.123456 | 最大 1.234567 | 非零 0.500``."""
        return (
            f"通道 {self.channel} | {self.height:g}×{self.width:g} | 均值 {self.mean:+.6f} | "
            f"最大 {self.maximum:+.6f} | 非零 {self.nonzero:.3f}"
        )


@dataclass(frozen=True)
class PoolRow:
    """⑤ 池化表的一行."""

    mode: str
    height: int
    width: int
    mean: float

    def line(self) -> str:
        """``max | 3×3 | 均值 +0.123456``."""
        return f"{self.mode:<4} | {self.height:g}×{self.width:g} | 均值 {self.mean:+.6f}"


@dataclass(frozen=True)
class PropertyRow:
    """⑦ 性质表的一行."""

    name: str
    passed: bool
    reading: float
    cross_check: str

    def line(self) -> str:
        """``通过 conv_matches_sliding_dot | 读数 0.000e+00 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return f"{mark} {self.name:<42} | 读数 {self.reading:.3e} | {self.cross_check}"


def layer_rows() -> tuple[LayerRow, ...]:
    """① 层表：一层的画像（输出尺寸由公式现场算出）."""
    size = len(SAMPLE_IMAGE)
    out = feature_size(size, SAMPLE_SPEC.kernel_size, stride=SAMPLE_SPEC.stride, padding=SAMPLE_SPEC.padding)
    spec_line = SAMPLE_SPEC.line()
    return (
        LayerRow(
            description=spec_line,
            output=f"{SAMPLE_SPEC.out_channels}×{out}×{out}",
            parameters=SAMPLE_SPEC.parameter_count,
        ),
    )


def size_rows() -> tuple[SizeRow, ...]:
    """② 尺寸表：几组参数下 6×6 输入的输出尺寸."""
    return tuple(
        SizeRow(
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            size=output_size(6, kernel, stride=stride, padding=padding, dilation=dilation),
        )
        for kernel, stride, padding, dilation in SIZE_CASES
    )


def field_rows() -> tuple[FieldRow, ...]:
    """③ 感受野表：递推公式与区间传播两条路径的读数."""
    return tuple(
        FieldRow(name=name, field=receptive_field(layers), direct=verify.direct_receptive_field(layers))
        for name, layers in FIELD_CASES.items()
    )


def feature_rows() -> tuple[FeatureRow, ...]:
    """④ 特征图表：一次卷积（按样本层）之后的每张特征图画像."""
    params = initialize_conv(SAMPLE_SPEC)
    from smart_research_agent.conv_net.layers import conv_block_forward

    outputs = conv_block_forward(params, (SAMPLE_IMAGE,), SAMPLE_SPEC)
    rows: list[FeatureRow] = []
    for index, channel in enumerate(outputs):
        stats = feature_stats(channel)
        rows.append(
            FeatureRow(
                channel=index,
                height=stats["height"],
                width=stats["width"],
                mean=stats["mean"],
                maximum=stats["max"],
                nonzero=stats["nonzero_ratio"],
            )
        )
    return tuple(rows)


def pool_rows() -> tuple[PoolRow, ...]:
    """⑤ 池化表：同一次 6×6 输入上 max / avg 的读数."""
    rows: list[PoolRow] = []
    for mode in (POOL_MAX, POOL_AVG):
        pooled = pool2d(SAMPLE_IMAGE, mode=mode, window=2)
        stats = feature_stats(pooled)
        rows.append(
            PoolRow(
                mode=mode,
                height=stats["height"],
                width=stats["width"],
                mean=stats["mean"],
            )
        )
    return tuple(rows)


def train_report() -> TrainReport:
    """⑥ 训练表：在合成数据集上真训练一次（读数来自 ``train.train_cnn``）."""
    dataset = make_stripe_dataset(noise=0.05, seed=5)
    _params, report = train_cnn(
        TRAIN_SPEC, dataset, steps=TRAIN_STEPS, learning_rate=TRAIN_LR, optimizer_name="adam", seed=100
    )
    return report


def train_rows(report: TrainReport | None = None) -> tuple[str, ...]:
    """⑥ 训练表：把损失曲线的几个点与总结行印出来."""
    resolved = report if report is not None else train_report()
    lines: list[str] = [resolved.summary_line()]
    total = len(resolved.losses)
    for index in (0, total // 4, total // 2, (3 * total) // 4, total - 1):
        lines.append(f"  step {index + 1:>3} | loss {resolved.losses[index]:.6f}")
    return tuple(lines)


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
                cross_check=f"{check.left} vs {check.right}",
            )
        )
    return tuple(rows)


def study_lines() -> tuple[str, ...]:
    """一次跑完七张表（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("== 1. 层表（通道 / 核 / 步长 / 填充 / 激活 / 参数 / 输出）")
    for row in layer_rows():
        lines.append("  " + row.line())
    lines.append("== 2. 尺寸表（6×6 输入在各种 (k, s, p, d) 下的输出尺寸）")
    for row in size_rows():
        lines.append("  " + row.line())
    lines.append("== 3. 感受野表（递推公式 vs 区间传播）")
    for row in field_rows():
        lines.append("  " + row.line())
    lines.append("== 4. 特征图表（一次卷积之后每张特征图的画像）")
    for row in feature_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 池化表（max / avg 在同一次输入上）")
    for row in pool_rows():
        lines.append("  " + row.line())
    lines.append("== 6. 训练表（合成数据集：竖线 vs 横线）")
    for line in train_rows():
        lines.append("  " + line if not line.startswith("  ") else line)
    lines.append("== 7. 性质表（七条性质：是否通过 / 读数 / 两个来源）")
    for row in property_rows():
        lines.append("  " + row.line())
    return tuple(lines)


#: 本课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTY_NAMES = CONV_PROPERTIES

__all__ = [
    "FIELD_CASES",
    "PROPERTY_NAMES",
    "SAMPLE_IMAGE",
    "SAMPLE_SPEC",
    "SIZE_CASES",
    "TRAIN_LR",
    "TRAIN_SPEC",
    "TRAIN_STEPS",
    "FeatureRow",
    "FieldRow",
    "LayerRow",
    "PoolRow",
    "PropertyRow",
    "SizeRow",
    "feature_rows",
    "field_rows",
    "layer_rows",
    "pool_rows",
    "property_rows",
    "size_rows",
    "study_lines",
    "train_report",
    "train_rows",
]
