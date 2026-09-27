"""``training_optim`` 的记录与旋钮表：一段训练过程（day081 / M7-D6）.

## 今天新增的结构：一段**过程**

```text
day080   一条链 = 一个块被复制 N 份         ——一个**结构**
day081   一段训练 = 一条链被更新 T 次        ——一段**过程**
```

过程的记录与前几天的记录有一处根本差别：它必须留下**每一步**，
因为要回答的问题全是关于"曲线"的：

```text
损失在下降吗       需要 first / best / last 三个数，而不是最后那一个
它有没有回头       需要整条序列（发散的定义就是"涨到初始值的若干倍"）
超参改了会怎样     需要两条曲线并排看（day081 的实验就是干这个的）
```

## 五个旋钮（本课的全部超参都在这一张表里）

```text
初始化    每一层权重的幅度        决定"信号穿过去是变大还是变小"（day080 的增益读数）
Dropout   训练时丢掉多少单元      决定"训练损失与推理损失差多少"
学习率    每一步走多远            决定"能不能收敛"（太大就发散）
调度      学习率随步数怎么变      决定"前期敢不敢走、后期稳不稳"
裁剪      一步最长能有多长        决定"一次坏梯度能不能把参数甩出去"
早停      什么时候该放弃           决定"训练多久才不算浪费"
```

后五个各自有**独立的一族失败**，而它们的修法都是"改超参"——
这正是 :class:`training_optim.errors.DivergenceError` 单独成族的理由。

## 三条必须写下来的边界

```text
① 本课训练的是**块参数**（前馈 + 两个 LN 的 γ/β），注意力那一层是给定函数
   ——与 day079/080 同一条口径。因此"参数量 1368"这个数在报告里会出现
② 记录里的损失有两个口径：`loss`（训练相、带 dropout）与 `eval_loss`（推理相、无 dropout）
   ——两者之差就是 dropout 的效果，而"只看训练损失"会让 dropout 看起来有害
③ 发散阈值是一个**判据**，不是事实：`divergence_factor = 1e3` 意味着
   "损失涨到初始值的 1000 倍以上"才算发散——这条线写进配置，因此可比
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping

from smart_research_agent.math_foundations.types import OPTIMIZERS, SCHEDULES
from smart_research_agent.training_optim.dropout import (
    DEFAULT_DROPOUT_RATE,
    _checked_rate,
)
from smart_research_agent.training_optim.errors import (
    DivergenceError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.training_optim.init import (
    INIT_SCHEMES,
    SCHEME_UNIFORM,
    _checked_scheme,
)

#: 五个旋钮的名字（报告里要能读出"这一次改的是哪一个"）.
CONTROL_INIT = "init"
CONTROL_DROPOUT = "dropout"
CONTROL_LEARNING_RATE = "learning_rate"
CONTROL_SCHEDULE = "schedule"
CONTROL_CLIP = "clip"
CONTROL_EARLY_STOP = "early_stop"

CONTROLS: tuple[str, ...] = (
    CONTROL_INIT,
    CONTROL_DROPOUT,
    CONTROL_LEARNING_RATE,
    CONTROL_SCHEDULE,
    CONTROL_CLIP,
    CONTROL_EARLY_STOP,
)

CONTROL_DESCRIPTIONS: dict[str, str] = {
    CONTROL_INIT: "每一层权重的幅度：决定信号穿过去是放大还是缩小（day080 的增益读数）",
    CONTROL_DROPOUT: "训练时丢掉多少单元：决定训练损失与推理损失差多少",
    CONTROL_LEARNING_RATE: "每一步走多远：太大就发散（DivergenceError），太小就学不动",
    CONTROL_SCHEDULE: "学习率随步数怎么变：前期敢走、后期稳",
    CONTROL_CLIP: "一步最长能有多长：一次坏梯度不该把参数甩出去",
    CONTROL_EARLY_STOP: "什么时候该放弃：判据是'连续多少步没变好'",
}

# ---------------------------------------------------------------------- 六条性质

PROPERTY_DETERMINISTIC = "training_is_deterministic"
PROPERTY_RATE_ZERO_MATCHES_STACK = "dropout_zero_matches_the_plain_stack"
PROPERTY_DROPOUT_EXPECTATION = "dropout_preserves_the_expectation"
PROPERTY_CLIP_PRESERVES_DIRECTION = "clipping_preserves_the_direction"
PROPERTY_SCHEDULE_MATCHES_BACKBONE = "schedule_matches_the_backbone"
PROPERTY_EARLY_STOP_ON_MONOTONE = "early_stopping_never_fires_on_a_monotone_curve"

TRAINING_PROPERTIES: tuple[str, ...] = (
    PROPERTY_DETERMINISTIC,
    PROPERTY_RATE_ZERO_MATCHES_STACK,
    PROPERTY_DROPOUT_EXPECTATION,
    PROPERTY_CLIP_PRESERVES_DIRECTION,
    PROPERTY_SCHEDULE_MATCHES_BACKBONE,
    PROPERTY_EARLY_STOP_ON_MONOTONE,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_DETERMINISTIC: "同一份配置跑两次：每一步的损失与学习率**逐位**相同",
    PROPERTY_RATE_ZERO_MATCHES_STACK: "``dropout = 0`` 时前向与 ``stack_forward`` 逐位一致（否则训的不是同一个模型）",
    PROPERTY_DROPOUT_EXPECTATION: "``E[掩码 × 缩放] = 1``：inverted dropout 在期望上不改变这一层",
    PROPERTY_CLIP_PRESERVES_DIRECTION: "裁剪只改一步的长度、不改方向（整体范数裁剪的定义）",
    PROPERTY_SCHEDULE_MATCHES_BACKBONE: "本课的 ``learning_rate_at`` 与 day074 的调度函数逐位一致",
    PROPERTY_EARLY_STOP_ON_MONOTONE: "单调下降的曲线上早停永不触发（它一旦触发就说明 patience 太小）",
}

TRAINING_NOTES: tuple[str, ...] = {
    "本课训练的是块参数（前馈 + 两个 LN 的 γ/β），注意力那一层是给定函数（day079/080 同一条口径）",
    "记录里的 loss 是训练相（带 dropout）、eval_loss 是推理相（无 dropout）——两者之差才是 dropout 的效果",
    "发散阈值是一个**判据**（默认 1e3 倍），写进配置因此可比；它不是'事实'",
}

#: 默认的发散阈值（损失涨到初始值的这个倍数以上就算发散）.
DEFAULT_DIVERGENCE_FACTOR = 1e3


def _checked_positive_int(value: Any, *, name: str) -> int:
    """正整数（步数、耐心、批大小这一类）."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ParameterError(f"{name} 必须是 >= 1 的整数，收到 {value!r}。")
    return value


def _checked_positive_float(value: Any, *, name: str) -> float:
    """正的有限数（学习率、裁剪阈值、发散阈值这一类）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ParameterError(
            f"{name} 必须是正的有限数，收到 {value!r}："
            "负的学习率会把下降变成上升，而它不会报错——只会让损失一路变大。"
        )
    return resolved


def _checked_non_negative_float(value: Any, *, name: str) -> float:
    """非负的有限数（``min_delta`` 这一类）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ParameterError(f"{name} 必须是非负的有限数，收到 {value!r}。")
    return resolved


def _checked_init_scale(value: Any) -> float:
    """初始化的幅度：必须落在 ``(0, 1)``（**与 day079 的 ``default_parameters`` 同一条纪律**）.

    上限不是随意的：``uniform`` 方案的半宽是 ``scale``，而注意力那一层走的
    ``default_parameters`` 自己就要求 ``0 < scale < 1``。配置放行一个 1.5，
    会在**几步之后**于另一个模块里炸——报错位置与出错位置因此离得很远。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"init_scale 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ParameterError(
            f"init_scale 必须落在 (0, 1)，收到 {value!r}："
            "初始化幅度太大时第一层的打分会被推到 softmax 的饱和区。"
        )
    return resolved


def _pairs(payload: Mapping[str, Any] | None) -> tuple[tuple[str, float], ...]:
    """把参数字典变成**有序的键值对**（冻结、可 hash、可 json）."""
    if payload is None:
        return ()
    items = tuple(sorted((str(key), float(value)) for key, value in payload.items()))
    for key, value in items:
        if not math.isfinite(value):
            raise ParameterError(f"调度/优化器参数 {key!r} 必须是有限数，收到 {value!r}。")
    return items


def _restore(value: float) -> float | int:
    """把整数值还原成 ``int``（**day074 的调度对 ``warmup_steps`` 要求整数**）.

    配置里的数字一律被冻结成有限数（这样它可以被 hash、被比对），
    而"这一步是第几步""一共多少步"这类参数在 day074 的签名里是 ``int``——
    于是出门时要还原。``total_steps=40.0`` 会被 day074 自己拒绝
    （"必须是 >= 2 的整数"），而那条报错的位置与错因隔着一个模块。
    """
    as_int = int(value)
    return as_int if float(as_int) == float(value) else value


@dataclass(frozen=True)
class TrainingConfig:
    """一份训练配置：**五个旋钮 + 两个记录口径**（全部进记录，因此可比）.

    它是 ``frozen`` 且字段全部可 hash：一份配置可以被放进集合、
    也可以被写进报告的 ``config`` 字段，而"这一次是怎么训的"不需要靠文件名去猜。
    """

    learning_rate: float = 0.05
    schedule: str = "constant"
    schedule_params: tuple[tuple[str, float], ...] = ()
    optimizer: str = "sgd"
    optimizer_params: tuple[tuple[str, float], ...] = ()
    dropout: float = DEFAULT_DROPOUT_RATE
    init_scheme: str = SCHEME_UNIFORM
    init_scale: float = 0.25
    max_norm: float | None = None
    steps: int = 40
    patience: int | None = None
    min_delta: float = 0.0
    seed: int = 7
    divergence_factor: float = DEFAULT_DIVERGENCE_FACTOR

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_rate", _checked_positive_float(self.learning_rate, name="learning_rate")
        )
        if self.schedule not in SCHEDULES:
            raise ParameterError(
                f"不认识的学习率调度 {self.schedule!r}：可用取值 {list(SCHEDULES)}。"
            )
        if self.optimizer not in OPTIMIZERS:
            raise ParameterError(
                f"不认识的优化器 {self.optimizer!r}：可用取值 {list(OPTIMIZERS)}。"
            )
        object.__setattr__(self, "schedule_params", _pairs(dict(self.schedule_params)))
        object.__setattr__(self, "optimizer_params", _pairs(dict(self.optimizer_params)))
        object.__setattr__(self, "dropout", _checked_rate(self.dropout))
        object.__setattr__(self, "init_scheme", _checked_scheme(self.init_scheme))
        object.__setattr__(self, "init_scale", _checked_init_scale(self.init_scale))
        if self.max_norm is not None:
            object.__setattr__(self, "max_norm", _checked_positive_float(self.max_norm, name="max_norm"))
        object.__setattr__(self, "steps", _checked_positive_int(self.steps, name="steps"))
        if self.patience is not None:
            object.__setattr__(
                self, "patience", _checked_positive_int(self.patience, name="patience")
            )
        object.__setattr__(self, "min_delta", _checked_non_negative_float(self.min_delta, name="min_delta"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ParameterError(f"seed 必须是整数，收到 {self.seed!r}。")
        object.__setattr__(
            self,
            "divergence_factor",
            _checked_positive_float(self.divergence_factor, name="divergence_factor"),
        )

    @property
    def schedule_kwargs(self) -> dict[str, float | int]:
        """调度函数的参数（交给 ``make_schedule``；整数值还原成 ``int``）."""
        return {key: _restore(value) for key, value in self.schedule_params}

    @property
    def optimizer_kwargs(self) -> dict[str, float | int]:
        """优化器的额外参数（交给 ``make_optimizer``）."""
        return {key: _restore(value) for key, value in self.optimizer_params}

    @property
    def clipping_on(self) -> bool:
        """这一步有没有开裁剪（``max_norm`` 为 ``None`` 时是关的）."""
        return self.max_norm is not None

    @property
    def early_stop_on(self) -> bool:
        """有没有开早停."""
        return self.patience is not None

    def replaced(self, **changes: Any) -> TrainingConfig:
        """换掉若干个字段（**实验靠它一次只改一个旋钮**）."""
        unknown = set(changes) - {item.name for item in fields(self)}
        if unknown:
            raise ParameterError(f"不认识的配置项 {sorted(unknown)}。")
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "learning_rate": self.learning_rate,
            "schedule": self.schedule,
            "schedule_params": dict(self.schedule_params),
            "optimizer": self.optimizer,
            "optimizer_params": dict(self.optimizer_params),
            "dropout": self.dropout,
            "init_scheme": self.init_scheme,
            "init_scale": self.init_scale,
            "max_norm": self.max_norm,
            "steps": self.steps,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "seed": self.seed,
            "divergence_factor": self.divergence_factor,
        }

    def summary_line(self) -> str:
        """一行说明：``40 步 | sgd lr=0.05 constant | dropout 0.00 | 初始化 uniform | 无裁剪 | 无早停``."""
        schedule = self.schedule
        if self.schedule_params:
            schedule += "(" + ", ".join(f"{k}={v:g}" for k, v in self.schedule_params) + ")"
        clip = f"裁剪 {self.max_norm:g}" if self.clipping_on else "无裁剪"
        stop = f"早停 {self.patience}" if self.early_stop_on else "无早停"
        return (
            f"{self.steps} 步 | {self.optimizer} lr={self.learning_rate:g} {schedule} | "
            f"dropout {self.dropout:.2f} | 初始化 {self.init_scheme} | {clip} | {stop}"
        )


@dataclass(frozen=True)
class ClipReport:
    """一次裁剪的读数（**缩放系数与范数都要留下**，否则"有没有裁剪"不可读）."""

    original_norm: float
    max_norm: float | None
    scale: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.original_norm)) or float(self.original_norm) < 0.0:
            raise NumericError(f"original_norm 必须是非负有限数，收到 {self.original_norm!r}。")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise NumericError(f"scale 必须是正的有限数，收到 {self.scale!r}。")
        if self.max_norm is not None:
            _checked_positive_float(self.max_norm, name="max_norm")

    @property
    def applied(self) -> bool:
        """这一次裁剪有没有真的生效（缩放系数严格小于 1 才算生效）."""
        return self.scale < 1.0

    @property
    def clipped_norm(self) -> float:
        """裁剪之后的范数（= 原范数 × 缩放系数）."""
        return self.original_norm * self.scale

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "original_norm": self.original_norm,
            "max_norm": self.max_norm,
            "scale": self.scale,
            "clipped_norm": self.clipped_norm,
            "applied": self.applied,
        }

    def summary_line(self) -> str:
        """一行说明：``‖g‖ 2.345678 → 1.000000（缩放 0.426，已裁剪）``."""
        mark = "已裁剪" if self.applied else "未裁剪"
        limit = "无阈值" if self.max_norm is None else f"{self.max_norm:g}"
        return (
            f"‖g‖ {self.original_norm:.6f} → {self.clipped_norm:.6f}"
            f"（缩放 {self.scale:.6f}，阈值 {limit}，{mark}）"
        )


@dataclass(frozen=True)
class EarlyStopReport:
    """早停的读数：**它是"什么时候停下来"的判据，而不是结果**."""

    patience: int
    best_step: int
    best_loss: float
    waited: int
    triggered: bool
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        _checked_positive_int(self.patience, name="patience")
        _checked_positive_int(self.best_step, name="best_step")
        if not math.isfinite(float(self.best_loss)):
            raise NumericError(f"best_loss 必须是有限数，收到 {self.best_loss!r}。")
        if isinstance(self.waited, bool) or not isinstance(self.waited, int) or self.waited < 0:
            raise ParameterError(f"waited 必须是非负整数，收到 {self.waited!r}。")
        if not isinstance(self.triggered, bool):
            raise ParameterError(f"triggered 必须是布尔量，收到 {self.triggered!r}。")
        object.__setattr__(self, "min_delta", _checked_non_negative_float(self.min_delta, name="min_delta"))

    @property
    def reason(self) -> str:
        """一行原因（报告里要能读出"为什么停在这里"）."""
        if self.triggered:
            return f"连续 {self.patience} 步没有比第 {self.best_step} 步好 {self.min_delta:g} 以上"
        return f"跑完了计划步数；最好的一步是第 {self.best_step} 步"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "patience": self.patience,
            "best_step": self.best_step,
            "best_loss": self.best_loss,
            "waited": self.waited,
            "triggered": self.triggered,
            "min_delta": self.min_delta,
            "reason": self.reason,
        }

    def summary_line(self) -> str:
        """一行说明."""
        mark = "已触发" if self.triggered else "未触发"
        return (
            f"早停（耐心 {self.patience}，min_delta {self.min_delta:g}）{mark} | "
            f"最好的一步 {self.best_step}（损失 {self.best_loss:.6f}） | 等了 {self.waited} 步"
        )


@dataclass(frozen=True)
class EpochRecord:
    """一步训练的读数（**每一步都要有**：曲线的全部信息在这里）."""

    step: int
    loss: float
    eval_loss: float
    learning_rate: float
    grad_norm: float
    clip_scale: float
    kept_fraction: float

    def __post_init__(self) -> None:
        _checked_positive_int(self.step, name="step")
        for name in ("loss", "eval_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise NumericError(f"{name} 必须是有限数，收到 {getattr(self, name)!r}。")
        object.__setattr__(
            self,
            "learning_rate",
            _checked_non_negative_float(self.learning_rate, name="learning_rate"),
        )
        if not math.isfinite(float(self.grad_norm)) or float(self.grad_norm) < 0.0:
            raise NumericError(f"grad_norm 必须是非负有限数，收到 {self.grad_norm!r}。")
        if not 0.0 < float(self.clip_scale) <= 1.0:
            raise ParameterError(f"clip_scale 必须落在 (0, 1]，收到 {self.clip_scale!r}。")
        if not 0.0 <= float(self.kept_fraction) <= 1.0:
            raise ParameterError(f"kept_fraction 必须落在 [0, 1]，收到 {self.kept_fraction!r}。")

    @property
    def gap(self) -> float:
        """训练损失与推理损失之差（**dropout 的效果就在这个数上**）."""
        return self.eval_loss - self.loss

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "step": self.step,
            "loss": self.loss,
            "eval_loss": self.eval_loss,
            "gap": self.gap,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "clip_scale": self.clip_scale,
            "kept_fraction": self.kept_fraction,
        }

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"第 {self.step:>3} 步 | 损失 {self.loss:.6f}（推理 {self.eval_loss:.6f}，差 {self.gap:+.6f}）"
            f" | lr {self.learning_rate:.6f} | ‖g‖ {self.grad_norm:.6f}"
            f"（缩放 {self.clip_scale:.6f}） | 保留 {self.kept_fraction:.3f}"
        )


@dataclass(frozen=True)
class TrainingCurve:
    """一段训练的完整记录（配置 + 每一步 + 早停报告）."""

    config: TrainingConfig
    shape: Any
    records: tuple[EpochRecord, ...]
    early_stop: EarlyStopReport | None = None
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.records)
        if not resolved:
            raise ParameterError("一段训练至少要有一个读数。")
        steps = [item.step for item in resolved]
        if steps != sorted(steps):
            raise ShapeError(f"步号必须递增，收到 {steps}。")
        if len(set(steps)) != len(steps):
            raise NumericError("步号不能重复：重复的那一步会污染 best_step。")
        object.__setattr__(self, "records", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def first_loss(self) -> float:
        """第一步的训练损失."""
        return self.records[0].loss

    @property
    def last_loss(self) -> float:
        """最后一步的训练损失."""
        return self.records[-1].loss

    @property
    def best(self) -> EpochRecord:
        """训练损失最小的那一步."""
        return min(self.records, key=lambda item: item.loss)

    @property
    def best_loss(self) -> float:
        """最好的训练损失."""
        return self.best.loss

    @property
    def best_step(self) -> int:
        """最好那一步的步号."""
        return self.best.step

    @property
    def first_eval_loss(self) -> float:
        """第一步的推理损失（无 dropout）."""
        return self.records[0].eval_loss

    @property
    def last_eval_loss(self) -> float:
        """最后一步的推理损失."""
        return self.records[-1].eval_loss

    @property
    def improvement_ratio(self) -> float:
        """``首次损失 / 最好损失``（**大于 1 才叫"变好了"**）."""
        base = self.best_loss
        if base <= 0.0:
            return math.inf if self.first_loss > 0.0 else 1.0
        return self.first_loss / base

    @property
    def decreased(self) -> bool:
        """终点比起点低（严格更低）."""
        return self.last_loss < self.first_loss

    @property
    def best_is_before_the_end(self) -> bool:
        """最好那一步出现在终点之前（说明后期在变差）."""
        return self.best_step < self.records[-1].step

    @property
    def diverged(self) -> bool:
        """有没有发散（损失涨到初始值的 ``divergence_factor`` 倍以上）."""
        threshold = self.first_loss * self.config.divergence_factor
        return any(
            not math.isfinite(item.loss) or item.loss >= threshold for item in self.records
        )

    @property
    def eval_decreased(self) -> bool:
        """推理损失（无 dropout）的终点比起点低."""
        return self.last_eval_loss < self.first_eval_loss

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "config": self.config.to_dict(),
            "records": [item.to_dict() for item in self.records],
            "first_loss": self.first_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "last_loss": self.last_loss,
            "improvement_ratio": self.improvement_ratio,
            "diverged": self.diverged,
            "early_stop": None if self.early_stop is None else self.early_stop.to_dict(),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``40 步 | 损失 0.123456 → 0.045678（最好第 12 步 0.041234，改善 3.00 倍）``."""
        return (
            f"{len(self.records)} 步 | 损失 {self.first_loss:.6f} → {self.last_loss:.6f}"
            f"（最好第 {self.best_step} 步 {self.best_loss:.6f}，改善 {self.improvement_ratio:.6f} 倍）"
            + (" | **已发散**" if self.diverged else "")
        )

    def table_lines(self, *, every: int = 1) -> tuple[str, ...]:
        """把曲线排成一张表（``every`` 控制抽稀：默认每一步都印）."""
        _checked_positive_int(every, name="every")
        header = (
            f"  {'步':>4} | {'训练损失':>11} | {'推理损失':>11} | {'之差':>10} | "
            f"{'lr':>10} | {'‖g‖':>10} | {'缩放':>8}"
        )
        lines = [header, "  " + "-" * 78]
        for index, item in enumerate(self.records):
            if index % every:
                continue
            lines.append(
                f"  {item.step:>4} | {item.loss:>11.6f} | {item.eval_loss:>11.6f} | "
                f"{item.gap:>+10.6f} | {item.learning_rate:>10.6f} | "
                f"{item.grad_norm:>10.6f} | {item.clip_scale:>8.6f}"
            )
        return tuple(lines)

    def mean_gap(self) -> float:
        """平均的训练/推理损失差（dropout 的一个汇总读数）."""
        return math.fsum(item.gap for item in self.records) / len(self.records)


def check_no_divergence(curve: TrainingCurve) -> None:
    """发散时抛 :class:`DivergenceError`（**控制流**：这一段训练不该被当成结果）."""
    if not isinstance(curve, TrainingCurve):
        raise ParameterError(f"curve 必须是 TrainingCurve，收到 {type(curve).__name__}。")
    if curve.diverged:
        threshold = curve.first_loss * curve.config.divergence_factor
        worst = max(item.loss for item in curve.records if math.isfinite(item.loss))
        raise DivergenceError(
            f"训练发散：损失从 {curve.first_loss:.6f} 涨到 {worst:.6f}"
            f"（阈值是初始值的 {curve.config.divergence_factor:g} 倍 = {threshold:.6f}）。"
            "每一**步**都是合法的，问题出在把那些步连起来这件事上——"
            "修法是改超参（学习率、裁剪阈值、热身步数），而不是改数据或改一行公式。"
        )


def describe_controls() -> tuple[str, ...]:
    """六个旋钮各一行（演示脚本开头打印它，教材用的是同一张表）."""
    return tuple(f"{name:<14} | {CONTROL_DESCRIPTIONS[name]}" for name in CONTROLS)


def shape_size(shape: Any) -> int:
    """形状声明的参数量（转发 ``StackShape.total_parameter_count``，越界当场报错）."""
    if not hasattr(shape, "total_parameter_count"):
        raise ParameterError(
            f"shape 必须有 total_parameter_count（收到 {type(shape).__name__}）。"
        )
    return int(shape.total_parameter_count)


def record_count(curve: TrainingCurve) -> int:
    """曲线里的读数条数."""
    if not isinstance(curve, TrainingCurve):
        raise ParameterError(f"curve 必须是 TrainingCurve，收到 {type(curve).__name__}。")
    return len(curve.records)


def steps_of(curve: TrainingCurve) -> tuple[int, ...]:
    """曲线里的步号序列."""
    if not isinstance(curve, TrainingCurve):
        raise ParameterError(f"curve 必须是 TrainingCurve，收到 {type(curve).__name__}。")
    return tuple(item.step for item in curve.records)


__all__ = [
    "CONTROLS",
    "CONTROL_DESCRIPTIONS",
    "CONTROL_CLIP",
    "CONTROL_DROPOUT",
    "CONTROL_EARLY_STOP",
    "CONTROL_INIT",
    "CONTROL_LEARNING_RATE",
    "CONTROL_SCHEDULE",
    "DEFAULT_DIVERGENCE_FACTOR",
    "PROPERTY_CLIP_PRESERVES_DIRECTION",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_DROPOUT_EXPECTATION",
    "PROPERTY_EARLY_STOP_ON_MONOTONE",
    "PROPERTY_RATE_ZERO_MATCHES_STACK",
    "PROPERTY_SCHEDULE_MATCHES_BACKBONE",
    "TRAINING_NOTES",
    "TRAINING_PROPERTIES",
    "ClipReport",
    "EarlyStopReport",
    "EpochRecord",
    "TrainingConfig",
    "TrainingCurve",
    "check_no_divergence",
    "describe_controls",
    "record_count",
    "shape_size",
    "steps_of",
]
