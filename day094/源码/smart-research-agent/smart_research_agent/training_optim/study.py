"""``training_optim`` 的对照实验：四个旋钮各一组（day081 / M7-D6）.

四组实验共用一个问题：**换掉一个旋钮，曲线会变成什么样？**

```text
① 初始化   xavier / kaiming / normal / uniform    ——它同时给出 day080 的"逐层增益"读数
② 调度     constant / warmup_cosine              ——**day079 留下的那个问题**：post-LN 需要热身吗
③ 裁剪     关 / 开（同一个学习率）                ——一次坏梯度该不该把参数甩出去
④ Dropout  0.0 / 0.2                            ——训练损失与推理损失差多少
```

## 四组实验的纪律（与 day076~080 逐字相同）

```text
一次只改一个旋钮        其余字段由 replaced() 明确列出，而不是重写一份 config
同一个目标与同一组数据   四组共用同一个 (inputs, target)
同一颗种子              于是"换旋钮"与"换随机性"两件事不会混在一起
```

## 这份表**不回答**什么

```text
它回答      "换掉这一个旋钮之后，损失曲线与几个读数怎么动"
它不回答    "哪个旋钮最好"——一颗种子、一个目标、一条 40 步的计划
           ⇒ 要下"哪个更好"的结论需要多种子与统计检验
```

真正的结论只在**差别足够大**的地方才敢下，而"足够大"这条线在每组实验里
由一条**事先写在判据里的**阈值决定（例如"发散"是损失涨到初始值的 1000 倍）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from smart_research_agent.math_foundations.types import Matrix, validate_matrix
from smart_research_agent.transformer_core.layers import mean_squared_error
from smart_research_agent.transformer_stack import (
    StackParameters,
    StackShape,
    stack_forward,
)
from smart_research_agent.training_optim.dropout import PHASE_TRAIN
from smart_research_agent.training_optim.errors import DivergenceError, ParameterError
from smart_research_agent.training_optim.init import (
    INIT_SCHEMES,
    SCHEME_UNIFORM,
    _checked_scheme,
    initialize_parameters,
)
from smart_research_agent.training_optim.types import TrainingConfig
from smart_research_agent.training_optim.train import train, training_forward

#: 四组实验的名字（报告里要能按组读）。
CONTROL_INIT = "init"
CONTROL_SCHEDULE = "schedule"
CONTROL_CLIP = "clip"
CONTROL_DROPOUT = "dropout"

CONTROL_GROUPS: tuple[str, ...] = (
    CONTROL_INIT,
    CONTROL_SCHEDULE,
    CONTROL_CLIP,
    CONTROL_DROPOUT,
)

CONTROL_GROUP_DESCRIPTIONS: dict[str, str] = {
    CONTROL_INIT: "初始化方案：它同时给出 day080 的逐层增益（第一层与最后一层）",
    CONTROL_SCHEDULE: "学习率调度：热身能不能救下 post-LN（day079 留下的问题）",
    CONTROL_CLIP: "梯度裁剪：同一个学习率下，裁剪有没有把曲线拉回来",
    CONTROL_DROPOUT: "Dropout：训练损失与推理损失之差（只看训练损失会让它看起来有害）",
}


def _checked_inputs(inputs: Matrix) -> Matrix:
    """样本输入（形状由调用方保证，这里只做一次统一校验）."""
    return validate_matrix(inputs, name="inputs")


@dataclass(frozen=True)
class StudyRow:
    """一行读数：**一个旋钮的一个取值**在一段训练上的结果."""

    control: str
    variant: str
    steps: int
    first_loss: float
    best_loss: float
    best_step: int
    last_loss: float
    improvement_ratio: float
    diverged: bool
    failure: str = ""
    extras: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.control not in CONTROL_GROUPS:
            raise ParameterError(f"不认识的对照组 {self.control!r}：可选 {list(CONTROL_GROUPS)}。")
        if self.steps < 0:
            raise ParameterError(
                f"steps 必须 >= 0（0 表示'还没更新就发散了'），收到 {self.steps}。"
            )
        for name in ("first_loss", "best_loss", "last_loss", "improvement_ratio"):
            value = float(getattr(self, name))
            if math.isnan(value) or value < 0.0:
                raise ParameterError(f"{name} 必须是非负的数，收到 {value!r}。")
        if self.best_step < 1:
            raise ParameterError(f"best_step 必须 >= 1，收到 {self.best_step}。")
        object.__setattr__(self, "failure", str(self.failure))
        object.__setattr__(self, "extras", tuple((str(k), float(v)) for k, v in self.extras))

    def extra(self, name: str) -> float:
        """取一个附加读数（不在表里时返回 ``nan``，让"没测"与"测到 0"分开）."""
        for key, value in self.extras:
            if key == name:
                return value
        return float("nan")

    @property
    def settled(self) -> bool:
        """收敛了吗（没发散，且最好的一步明显优于第一步）."""
        return not self.diverged and self.improvement_ratio > 1.05

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "control": self.control,
            "variant": self.variant,
            "steps": self.steps,
            "first_loss": self.first_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "last_loss": self.last_loss,
            "improvement_ratio": self.improvement_ratio,
            "diverged": self.diverged,
            "settled": self.settled,
            "failure": self.failure,
            "extras": dict(self.extras),
        }

    def summary_line(self) -> str:
        """一行说明."""
        if self.failure:
            return f"{self.control:<8} {self.variant:<24} | 发散：{self.failure[:72]}…"
        mark = "收敛" if self.settled else "未收敛"
        return (
            f"{self.control:<8} {self.variant:<24} | {self.steps:>3} 步 | "
            f"损失 {self.first_loss:.6f} → {self.last_loss:.6f}"
            f"（最好第 {self.best_step} 步 {self.best_loss:.6f}） | "
            f"改善 {self.improvement_ratio:.4f} 倍 | {mark}"
        )


@dataclass(frozen=True)
class TrainingStudy:
    """四组实验的总表."""

    rows: tuple[StudyRow, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.rows)
        if not resolved:
            raise ParameterError("对照实验至少要有一行读数。")
        groups = {row.control for row in resolved}
        if groups != set(CONTROL_GROUPS):
            raise ParameterError(
                f"四组必须齐备：缺 {sorted(set(CONTROL_GROUPS) - groups)}——"
                "缺一组的那张表说不出'换这一个旋钮有没有用'。"
            )
        object.__setattr__(self, "rows", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def rows_of(self, control: str) -> tuple[StudyRow, ...]:
        """某一组的所有行（按运行顺序）."""
        if control not in CONTROL_GROUPS:
            raise ParameterError(f"不认识的对照组 {control!r}：可选 {list(CONTROL_GROUPS)}。")
        return tuple(row for row in self.rows if row.control == control)

    def row(self, control: str, variant: str) -> StudyRow:
        """按 (组, 取值) 取一行（找不到就报错，而不是返回一个默认值）."""
        for item in self.rows_of(control):
            if item.variant == variant:
                return item
        raise ParameterError(f"{control} 这一组里没有取值 {variant!r}。")

    @property
    def diverged_variants(self) -> tuple[str, ...]:
        """所有发散的行（"哪几组超参不能这么设"的名单）."""
        return tuple(f"{row.control}:{row.variant}" for row in self.rows if row.diverged)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "rows": [row.to_dict() for row in self.rows],
            "diverged_variants": list(self.diverged_variants),
            "notes": list(self.notes),
        }

    def table_lines(self) -> tuple[str, ...]:
        """把四组排成一张表."""
        header = (
            f"  {'组':<8} | {'取值':<24} | {'步':>3} | {'首个损失':>10} | "
            f"{'最好损失':>10} | {'改善':>8} | {'末损失':>10} | 状态"
        )
        lines = [header, "  " + "-" * 96]
        for row in self.rows:
            state = "发散" if row.diverged else ("收敛" if row.settled else "未收敛")
            lines.append(
                f"  {row.control:<8} | {row.variant:<24} | {row.steps:>3} | "
                f"{row.first_loss:>10.6f} | {row.best_loss:>10.6f} | "
                f"{row.improvement_ratio:>8.4f} | {row.last_loss:>10.6f} | {state}"
            )
        return tuple(lines)

    def summary_line(self) -> str:
        """一行说明：``四组实验 12 行 | 发散 0 行 | 收敛 9 行``."""
        diverged = len(self.diverged_variants)
        settled = sum(1 for row in self.rows if row.settled)
        return (
            f"四组实验 {len(self.rows)} 行 | 发散 {diverged} 行 | 收敛 {settled} 行"
        )


def _row_of(
    control: str,
    variant: str,
    curve: Any,
    *,
    extras: Sequence[tuple[str, float]] = (),
) -> StudyRow:
    """把一条曲线压成一行读数（**四组共用的那一处压法**）."""
    return StudyRow(
        control=control,
        variant=variant,
        steps=len(curve.records),
        first_loss=curve.first_loss,
        best_loss=curve.best_loss,
        best_step=curve.best_step,
        last_loss=curve.last_loss,
        improvement_ratio=curve.improvement_ratio,
        diverged=curve.diverged,
        extras=tuple(extras),
    )


def _initial_train_loss(
    params: StackParameters,
    sample: Matrix,
    goal: Matrix,
    config: TrainingConfig,
    *,
    placement: str = "pre",
) -> float:
    """第 1 步**更新之前**的训练损失（发散行里用它当占位读数，而不是填 0）."""
    cache = training_forward(
        params, sample, config=config, step=1, phase=PHASE_TRAIN, placement=placement
    )
    return mean_squared_error(cache.output, goal)


def _run(
    control: str,
    variant: str,
    shape: StackShape,
    sample: Matrix,
    goal: Matrix,
    config: TrainingConfig,
    params: StackParameters,
    *,
    extras: Sequence[tuple[str, float]] = (),
    extras_from_curve: Callable[[Any], Sequence[tuple[str, float]]] | None = None,
    placement: str = "pre",
) -> StudyRow:
    """跑一条曲线，**发散时也返回一行**（而不是让整张表崩掉）.

    "这一组发散"本身就是一个读数：一张表里如果缺了那一行，
    读者会以为"没测"，而实际上它是"测了，而且炸了"。

    ``extras_from_curve`` 让附加读数**从同一条曲线上取**（例如裁剪那一组
    需要实测的缩放系数）——因此每条曲线只跑一次。
    """
    try:
        curve = train(shape, sample, goal, config=config, params=params, placement=placement)
    except DivergenceError as error:
        initial = _initial_train_loss(
            params, sample, goal, config, placement=placement
        )
        return StudyRow(
            control=control,
            variant=variant,
            steps=0,
            first_loss=initial,
            best_loss=initial,
            best_step=1,
            last_loss=initial,
            improvement_ratio=1.0,
            diverged=True,
            failure=str(error),
            extras=tuple(extras),
        )
    resolved_extras = (
        tuple(extras) if extras_from_curve is None else tuple(extras_from_curve(curve))
    )
    return _row_of(control, variant, curve, extras=resolved_extras)


def _dropout_extras(curve: Any) -> tuple[tuple[str, float], ...]:
    """Dropout 那一组的附加读数：**训练/推理损失之差**（只看训练损失会误判）."""
    return (
        ("mean_gap", curve.mean_gap()),
        ("last_gap", curve.records[-1].gap),
        ("last_eval_loss", curve.last_eval_loss),
    )


def _clip_scale_extras(curve: Any) -> tuple[tuple[str, float], ...]:
    """裁剪那一组的附加读数：**实测的缩放系数**（有没有真的裁到只能从这里读）."""
    scales = [record.clip_scale for record in curve.records]
    return (
        ("mean_clip_scale", sum(scales) / len(scales)),
        ("min_clip_scale", min(scales)),
    )


def compare_initializations(
    shape: StackShape,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig | None = None,
    schemes: Sequence[str] = INIT_SCHEMES,
) -> tuple[StudyRow, ...]:
    """第一组：换初始化方案，其余不动（同时量 day080 的逐层增益）."""
    resolved_config = TrainingConfig() if config is None else config
    sample = _checked_inputs(inputs)
    goal = validate_matrix(target, name="target")
    rows: list[StudyRow] = []
    for scheme in schemes:
        resolved_scheme = _checked_scheme(scheme)
        params = initialize_parameters(
            shape,
            scheme=resolved_scheme,
            seed=resolved_config.seed,
            scale=resolved_config.init_scale,
        )
        gains = stack_forward(params, sample).gain_profile()
        rows.append(
            _run(
                CONTROL_INIT,
                resolved_scheme,
                shape,
                sample,
                goal,
                resolved_config.replaced(init_scheme=resolved_scheme),
                params,
                extras=(
                    ("gain_first", gains[0]),
                    ("gain_last", gains[-1]),
                    ("gain_max", max(gains)),
                ),
            )
        )
    return tuple(rows)


def compare_schedules(
    shape: StackShape,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig | None = None,
    placements: Sequence[str] = ("pre", "post"),
    warmup_steps: int = 5,
) -> tuple[StudyRow, ...]:
    """第二组：**day079 留下的那个问题**——post-LN 在没有热身时会更难吗？

    每一档摆放位置各跑两条曲线（``constant`` 与 ``warmup_cosine``），
    两者只差 ``schedule`` 与它的参数。学习率与步数完全一样，
    因此"热身有没有救下它"这句话有一个可比较的参照物。
    """
    resolved_config = TrainingConfig() if config is None else config
    sample = _checked_inputs(inputs)
    goal = validate_matrix(target, name="target")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ParameterError(f"warmup_steps 必须是 >= 1 的整数，收到 {warmup_steps!r}。")
    steps = resolved_config.steps
    if warmup_steps >= steps:
        raise ParameterError(
            f"warmup_steps（{warmup_steps}）必须小于步数（{steps}）："
            "热身占满全程时没有退火段，那它就不是'热身+退火'。"
        )
    rows: list[StudyRow] = []
    base_params = initialize_parameters(
        shape, scheme=resolved_config.init_scheme, seed=resolved_config.seed, scale=resolved_config.init_scale
    )
    for placement in placements:
        plain = resolved_config.replaced(schedule="constant", schedule_params={})
        rows.append(
            _run(
                CONTROL_SCHEDULE,
                f"{placement}-LN constant",
                shape,
                sample,
                goal,
                plain,
                base_params,
                placement=placement,
            )
        )
        warmed = resolved_config.replaced(
            schedule="warmup_cosine",
            schedule_params=(
                ("warmup_steps", float(warmup_steps)),
                ("total_steps", float(steps)),
            ),
        )
        rows.append(
            _run(
                CONTROL_SCHEDULE,
                f"{placement}-LN warmup_cosine",
                shape,
                sample,
                goal,
                warmed,
                base_params,
                placement=placement,
            )
        )
    return tuple(rows)


def compare_clipping(
    shape: StackShape,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig | None = None,
    max_norms: Sequence[float] = (0.5,),
) -> tuple[StudyRow, ...]:
    """第三组：同一个学习率下，裁剪关/开各跑一条曲线."""
    resolved_config = TrainingConfig() if config is None else config
    sample = _checked_inputs(inputs)
    goal = validate_matrix(target, name="target")
    rows: list[StudyRow] = []
    base_params = initialize_parameters(
        shape, scheme=resolved_config.init_scheme, seed=resolved_config.seed, scale=resolved_config.init_scale
    )
    rows.append(
        _run(
            CONTROL_CLIP,
            "无裁剪",
            shape,
            sample,
            goal,
            resolved_config.replaced(max_norm=None),
            base_params,
            extras=(("mean_clip_scale", 1.0), ("min_clip_scale", 1.0)),
        )
    )
    for max_norm in max_norms:
        rows.append(
            _run(
                CONTROL_CLIP,
                f"裁剪 {float(max_norm):g}",
                shape,
                sample,
                goal,
                resolved_config.replaced(max_norm=float(max_norm)),
                base_params,
                extras_from_curve=_clip_scale_extras,
            )
        )
    return tuple(rows)


def compare_dropout(
    shape: StackShape,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig | None = None,
    rates: Sequence[float] = (0.0, 0.2),
) -> tuple[StudyRow, ...]:
    """第四组：dropout 关/开各跑一条曲线，并把训练/推理损失之差量出来."""
    resolved_config = TrainingConfig() if config is None else config
    sample = _checked_inputs(inputs)
    goal = validate_matrix(target, name="target")
    rows: list[StudyRow] = []
    base_params = initialize_parameters(
        shape, scheme=resolved_config.init_scheme, seed=resolved_config.seed, scale=resolved_config.init_scale
    )
    for rate in rates:
        rows.append(
            _run(
                CONTROL_DROPOUT,
                f"rate {float(rate):.2f}",
                shape,
                sample,
                goal,
                resolved_config.replaced(dropout=float(rate)),
                base_params,
                extras_from_curve=_dropout_extras,
            )
        )
    return tuple(rows)


def training_study(
    shape: StackShape,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig | None = None,
    schemes: Sequence[str] = INIT_SCHEMES,
    placements: Sequence[str] = ("pre", "post"),
    warmup_steps: int = 5,
    max_norms: Sequence[float] = (0.5,),
    rates: Sequence[float] = (0.0, 0.2),
) -> TrainingStudy:
    """把四组实验跑一遍，返回总表（**这一课的全部读数都在这里**）."""
    resolved_config = TrainingConfig() if config is None else config
    sample = _checked_inputs(inputs)
    goal = validate_matrix(target, name="target")
    rows: list[StudyRow] = []
    rows.extend(
        compare_initializations(shape, sample, goal, config=resolved_config, schemes=schemes)
    )
    rows.extend(
        compare_schedules(
            shape,
            sample,
            goal,
            config=resolved_config,
            placements=placements,
            warmup_steps=warmup_steps,
        )
    )
    rows.extend(
        compare_clipping(shape, sample, goal, config=resolved_config, max_norms=max_norms)
    )
    rows.extend(
        compare_dropout(shape, sample, goal, config=resolved_config, rates=rates)
    )
    return TrainingStudy(
        rows=tuple(rows),
        notes=(
            "四组共用同一颗种子、同一个目标与同一份配置，只改一个旋钮",
            "diverged 的判据是损失涨到初始值的 divergence_factor 倍（默认 1e3）",
            "这份表是一次观测：要下'哪个旋钮更好'的结论需要多种子与统计检验",
        ),
    )


def gain_profile_of(
    shape: StackShape,
    inputs: Matrix,
    *,
    scheme: str = SCHEME_UNIFORM,
    seed: int = 7,
    scale: float = 0.25,
) -> tuple[float, ...]:
    """某个初始化方案下的逐层增益（day080 的读数，day081 用它解释初始化）。"""
    resolved_scheme = _checked_scheme(scheme)
    sample = _checked_inputs(inputs)
    params = initialize_parameters(shape, scheme=resolved_scheme, seed=seed, scale=scale)
    return stack_forward(params, sample).gain_profile()


def stack_of(shape: StackShape, *, scheme: str, seed: int, scale: float) -> StackParameters:
    """按方案造一摞参数（演示脚本按方案逐个打印增益时用）."""
    return initialize_parameters(
        shape, scheme=_checked_scheme(scheme), seed=seed, scale=scale
    )


__all__ = [
    "CONTROL_GROUPS",
    "CONTROL_GROUP_DESCRIPTIONS",
    "CONTROL_CLIP",
    "CONTROL_DROPOUT",
    "CONTROL_INIT",
    "CONTROL_SCHEDULE",
    "StudyRow",
    "TrainingStudy",
    "compare_clipping",
    "compare_dropout",
    "compare_initializations",
    "compare_schedules",
    "gain_profile_of",
    "stack_of",
    "training_study",
]
