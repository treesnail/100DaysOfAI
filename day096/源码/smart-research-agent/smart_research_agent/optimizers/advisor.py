"""``advisor``：为 SmartResearch Agent 的微调脚本**选一个优化器**（day092 / M8-D3）.

curriculum 给 day092 的第三条目标是"**为 SmartResearch Agent 的微调脚本选择合适优化器**"。
这一模块把前面六个规则与三种衰减**落到本项目真实存在的几类训练场景**上：
LoRA 微调、全参微调、DPO 对齐、从零训练小网络、病态目标、纯 CPU 小实验。

```text
场景           推荐优化器      起点学习率   衰减            裁剪        为什么
sft-lora       adamw          1e-4        解耦 0.01       1.0         解耦衰减不随梯度尺度变化
sft-full       adamw          2e-5        解耦 0.10       1.0         全参更新更敏感，lr 要更小
dpo            adamw          5e-7        解耦 0.00       1.0         偏好优化对 lr 极敏感
scratch-mlp    adam           1e-2        无              none        小网络用 Adam 起步快
ill-conditioned nesterov      1e-2        无              1.0         病态峡谷靠动量与 lookahead
cpu-tiny       rmsprop        1e-2        无              none        只归一化尺度、最省内存
```

> **这些数字是本仓库的"起点配置"，不是"最优配置"。** 上表里没有一个数字来自
> 某篇论文的基准表或某次实验的排名——它们是"从更新公式能读出来的默认值 +
> 工程惯例"，加一句"为什么"。真正的调参必须在自己的数据上跑，本模块给的是
> **出发点**，而不是结论。把这句话写进文档，是为了不让"推荐值"被误读成"最优值"。

## 每条推荐至少要能回答"为什么是它"

推荐不是"查表给个数"，而是"从更新公式推出一个方向"。例如：

```text
LoRA / 全参 / DPO   都在大模型上微调 ⇒ 默认 AdamW：它的解耦衰减不会因为
                    某层梯度尺度不同而被 √v̂ 除成不同的强度
从零训练小网络       参数少、目标简单 ⇒ Adam 起步最快；加衰减反而拖慢
病态峡谷             陡方向来回横跳 ⇒ 动量/lookahead 把横跳平均掉
```

这三条都能在 :mod:`optimizers.types` 的公式表里找到依据，而不是"经验告诉我"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_research_agent.optimizers.errors import ParameterError
from smart_research_agent.optimizers.optimizer import TrainingOptimizer, make_train_optimizer
from smart_research_agent.optimizers.types import (
    DECAY_DECOUPLED,
    DECAY_NONE,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_NESTEROV,
    OPTIMIZER_RMSPROP,
    TRAIN_OPTIMIZERS,
)


@dataclass(frozen=True)
class TrainingScenario:
    """一类训练场景的画像（决定"该选什么优化器"的三件事实）."""

    name: str
    description: str
    model_scale: str
    conditioning: str
    gradient_noise: str

    def line(self) -> str:
        """``sft-lora      | 规模 大（LoRA 适配器） | 条件 良态 | 噪声 中``."""
        return (
            f"{self.name:<15} | 规模 {self.model_scale:<18} | "
            f"条件 {self.conditioning:<6} | 噪声 {self.gradient_noise}"
        )


@dataclass(frozen=True)
class Recommendation:
    """一条优化器推荐：选谁 + 起点超参 + **为什么**."""

    scenario: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    decay_mode: str
    max_grad_norm: float | None
    why: tuple[str, ...] = field(default_factory=tuple)

    def line(self) -> str:
        """``sft-lora      → adamw  lr=1e-4  decay=decoupled:0.01  clip=1.0``."""
        clip = "none" if self.max_grad_norm is None else f"{self.max_grad_norm:g}"
        return (
            f"{self.scenario:<15} → {self.optimizer:<8} lr={self.learning_rate:g}  "
            f"decay={self.decay_mode}:{self.weight_decay:g}  clip={clip}"
        )

    def build(self) -> TrainingOptimizer:
        """按这条推荐造一个可用的优化器（**推荐可执行**，不是一段散文）."""
        return make_train_optimizer(
            self.optimizer,
            self.learning_rate,
            weight_decay=self.weight_decay,
            decay_mode=self.decay_mode,
            max_grad_norm=self.max_grad_norm,
        )


#: 六类场景的画像（顺序 = 从"大模型微调"到"纯 CPU 小实验"）.
SCENARIOS: dict[str, TrainingScenario] = {
    "sft-lora": TrainingScenario(
        "sft-lora", "LoRA 适配器微调（本项目 sft / peft 包）", "大（适配器）", "良态", "中"
    ),
    "sft-full": TrainingScenario(
        "sft-full", "全参监督微调（本项目 sft 包）", "大（全部参数）", "良态", "中"
    ),
    "dpo": TrainingScenario("dpo", "DPO 偏好对齐（本项目 alignment / dpo 包）", "大", "良态", "高"),
    "scratch-mlp": TrainingScenario(
        "scratch-mlp", "从零训练小网络（day089 / day090 的 MLP）", "小", "良态", "低"
    ),
    "ill-conditioned": TrainingScenario(
        "ill-conditioned", "病态目标（day092 的峡谷 / Rosenbrock）", "小", "病态", "低"
    ),
    "cpu-tiny": TrainingScenario(
        "cpu-tiny", "纯 CPU 小实验（离线可跑的最小回路）", "极小", "良态", "低"
    ),
}

#: 场景 -> 推荐（**每条的 why 都能在公式表里找到依据**）.
RECOMMENDATIONS: dict[str, Recommendation] = {
    "sft-lora": Recommendation(
        scenario="sft-lora",
        optimizer=OPTIMIZER_ADAMW,
        learning_rate=1e-4,
        weight_decay=0.01,
        decay_mode=DECAY_DECOUPLED,
        max_grad_norm=1.0,
        why=(
            "大模型微调的默认起点是 AdamW：它的解耦衰减不随梯度尺度变化——"
            "若用耦合 L2，某层梯度大时衰减会被 √v̂ 除得更弱，各层衰减强度就不再一致。",
            "适配器层数少、更新量小，lr=1e-4 是 LoRA 常用的量级（比全参更激进）。",
            "整体范数裁剪 1.0 用来挡住偶发的梯度尖峰，不改变方向。",
        ),
    ),
    "sft-full": Recommendation(
        scenario="sft-full",
        optimizer=OPTIMIZER_ADAMW,
        learning_rate=2e-5,
        weight_decay=0.10,
        decay_mode=DECAY_DECOUPLED,
        max_grad_norm=1.0,
        why=(
            "全参微调时每一个参数都在动，lr 必须比 LoRA 小一个量级（2e-5 量级）。",
            "衰减取 0.10：全参更容易过拟合，解耦衰减起到正则作用。",
            "仍用整体范数裁剪 1.0——大模型早期几步的梯度尖峰更常见。",
        ),
    ),
    "dpo": Recommendation(
        scenario="dpo",
        optimizer=OPTIMIZER_ADAMW,
        learning_rate=5e-7,
        weight_decay=0.0,
        decay_mode=DECAY_DECOUPLED,
        max_grad_norm=1.0,
        why=(
            "偏好对齐的目标比 SFT 更尖锐，lr 通常再小两三个数量级（5e-7 量级）。",
            "DPO 常把 weight_decay 设为 0：目标里已经含了相对奖励，再叠正则容易把策略拉回参考模型。",
            "裁剪仍保留：对齐早期梯度噪声大，1.0 是保守的兜底。",
        ),
    ),
    "scratch-mlp": Recommendation(
        scenario="scratch-mlp",
        optimizer=OPTIMIZER_ADAM,
        learning_rate=1e-2,
        weight_decay=0.0,
        decay_mode=DECAY_NONE,
        max_grad_norm=None,
        why=(
            "小网络目标简单、参数少：Adam 的每参数自适应步长让它起步远快于 SGD。",
            "不做衰减：小模型不怕过拟合，加衰减只会拖慢它走到零损失的时间。",
            "不做裁剪：这条链上的梯度由 day090 的反向给出，量级可控。",
        ),
    ),
    "ill-conditioned": Recommendation(
        scenario="ill-conditioned",
        optimizer=OPTIMIZER_NESTEROV,
        learning_rate=1e-2,
        weight_decay=0.0,
        decay_mode=DECAY_NONE,
        max_grad_norm=1.0,
        why=(
            "病态峡谷里 SGD 在陡方向来回横跳：动量把横跳平均掉。",
            "选 Nesterov 而不是普通动量：它在接近过冲时会**提前**刹车（lookahead）。",
            "裁剪 1.0 兜底：峡谷里偶尔会出现很大的梯度。",
        ),
    ),
    "cpu-tiny": Recommendation(
        scenario="cpu-tiny",
        optimizer=OPTIMIZER_RMSPROP,
        learning_rate=1e-2,
        weight_decay=0.0,
        decay_mode=DECAY_NONE,
        max_grad_norm=None,
        why=(
            "RMSProp 只保留一个二阶动量：状态内存最省，适合纯 CPU 的手边实验。",
            "它把每个参数的尺度归一，因此对 lr 的量级不敏感（1e-2 起步即可）。",
            "无裁剪、无衰减：这一步的目标是把回路跑通，而不是刷指标。",
        ),
    ),
}

if set(SCENARIOS) != set(RECOMMENDATIONS):  # pragma: no cover - 导入期不变式
    raise ParameterError(
        "场景表与推荐表不一致：少一个键的场景会静默地没有推荐，"
        "而'没有推荐'与'推荐为空'在读报告时长得一样。"
    )

if not set(RECOMMENDATIONS) <= set(SCENARIOS):  # pragma: no cover - 导入期不变式
    raise ParameterError("推荐表里有不属于任何场景的键。")


def recommend(scenario: str) -> Recommendation:
    """按场景名取一条推荐（不认识的名字当场报错，而不是给个默认值）."""
    if scenario not in RECOMMENDATIONS:
        raise ParameterError(
            f"不认识的训练场景 {scenario!r}：可用取值 {list(SCENARIOS)}。"
        )
    return RECOMMENDATIONS[scenario]


def recommendation_lines() -> tuple[str, ...]:
    """把六条推荐逐行印出（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    for name in SCENARIOS:
        lines.append("  " + SCENARIOS[name].line())
        lines.append("    " + RECOMMENDATIONS[name].line())
        for reason in RECOMMENDATIONS[name].why:
            lines.append("      · " + reason)
    return tuple(lines)


#: 推荐里出现过的优化器（供测试检查它们都属于 :data:`types.TRAIN_OPTIMIZERS`）.
RECOMMENDED_OPTIMIZERS: tuple[str, ...] = tuple(
    sorted({rec.optimizer for rec in RECOMMENDATIONS.values()})
)

__all__ = [
    "RECOMMENDATIONS",
    "RECOMMENDED_OPTIMIZERS",
    "SCENARIOS",
    "Recommendation",
    "TrainingScenario",
    "recommend",
    "recommendation_lines",
]
