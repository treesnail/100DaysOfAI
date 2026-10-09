"""``study``：七张表（day095 / M8-D6）.

```text
① 轴表       两条归一化轴（batch / feature）的公式与"签名"
② 相表       同一个 batch 上训练相与推理相的读数（μ/σ 来自哪里、差多少）
③ 技巧表     四个技巧：谁实现、开在哪一步
④ 参数表     BatchNorm 加了多少参数（2H），以及隐藏宽变化时它怎么变
⑤ 消融表     四个变体在 rnn / lstm 上的训练损失 / 推理损失 / 间隔 / 准确率  ← **核心**
⑥ 日志表     sparkline / ASCII 折线 / 条形图 / 汇总裁剪（训练日志的四张脸）
⑦ 性质表     七条性质是否通过 / 现场读数 / 两个来源
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
第 ⑤ 张表尤其如此：它给出的每一行都同时有**训练损失**与**推理损失**，
以及两者的差 ``loss_gap``——**只看训练损失会把 dropout 的代价看反**
（这正是 day081 留下"两个损失口径都要给"那条纪律的原因）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.regularization import train as reg_train
from smart_research_agent.regularization import verify
from smart_research_agent.regularization import visualize as vz
from smart_research_agent.regularization.normalization import (
    batch_norm_forward,
    initial_running,
    RunningStatistics,
)
from smart_research_agent.regularization.techniques import RegularizationConfig
from smart_research_agent.regularization.types import (
    AXIS_BATCH,
    AXIS_DESCRIPTIONS,
    AXIS_FEATURE,
    AXIS_FORMULAS,
    AXIS_SIGNATURES,
    BN_BACKWARD_FORMULA,
    BN_EVAL_BACKWARD_FORMULA,
    BN_FORMULA,
    NORM_AXES,
    PHASE_DESCRIPTIONS,
    PHASE_EVAL,
    PHASE_STATISTIC_SOURCES,
    PHASE_TRAIN,
    PHASES,
    REGULARIZATION_PROPERTIES,
    TECHNIQUES,
    TECHNIQUE_DESCRIPTIONS,
    TECHNIQUE_SOURCES,
)
from smart_research_agent.sequence_models.train import make_sign_dataset

#: ② 用的样本批（与 verify 的样本同一批，便于互相对照）.
PHASE_BATCH = verify.SAMPLE_BATCH
PHASE_GAMMA = verify.SAMPLE_GAMMA
PHASE_BETA = verify.SAMPLE_BETA

#: ④ 用的三个隐藏宽（参数量 = 2H 随它线性增长）.
HIDDEN_CASES: tuple[int, ...] = (2, 4, 8)

#: ⑤ 用的四个变体（**一次只关一个旋钮**）.
ABLATION_VARIANTS: dict[str, RegularizationConfig] = {
    "全开": RegularizationConfig(
        norm=True, dropout_rate=0.2, schedule="cosine", base_lr=0.1, min_lr=0.01, patience=8
    ),
    "无归一化": RegularizationConfig(
        norm=False, dropout_rate=0.2, schedule="cosine", base_lr=0.1, min_lr=0.01, patience=8
    ),
    "无丢弃": RegularizationConfig(
        norm=True, dropout_rate=0.0, schedule="cosine", base_lr=0.1, min_lr=0.01, patience=8
    ),
    "无调度": RegularizationConfig(
        norm=True, dropout_rate=0.2, schedule="constant", base_lr=0.1, patience=8
    ),
}

#: ⑤ 用的训练配置（与演示脚本共用同一批数字）.
ABLATION_CELLS: tuple[str, ...] = ("rnn", "lstm")
ABLATION_EPOCHS = 40
ABLATION_BATCH_SIZE = 8
ABLATION_HIDDEN = 4
ABLATION_SEED = 100


@dataclass(frozen=True)
class AxisRow:
    """① 轴表的一行."""

    axis: str
    formula: str
    signature: str

    def line(self) -> str:
        """``batch   | μ_j = ... | 批大小 1 时方差恒为 0 ...``."""
        return f"{self.axis:<8} | {self.formula} | {self.signature}"


@dataclass(frozen=True)
class PhaseRow:
    """② 相表的一行."""

    phase: str
    statistic: str
    mean_of_means: float
    max_abs: float

    def line(self) -> str:
        """``train | 本批统计量 | μ 均值 +0.000000 | 最大 |y| 1.234567``."""
        return (
            f"{self.phase:<6} | {self.statistic:<28} | μ 均值 {self.mean_of_means:+.6f} | "
            f"最大 |y| {self.max_abs:.6f}"
        )


@dataclass(frozen=True)
class TechniqueRow:
    """③ 技巧表的一行."""

    technique: str
    description: str
    source: str

    def line(self) -> str:
        """``batchnorm | 批量归一化：... | 本包新建（...）``."""
        return f"{self.technique:<10} | {self.source}"


@dataclass(frozen=True)
class ParameterRow:
    """④ 参数表的一行."""

    hidden: int
    batch_norm_parameters: int

    def line(self) -> str:
        """``H = 4 | BatchNorm 参数 8（2H）``."""
        return f"H = {self.hidden:<2} | BatchNorm 参数 {self.batch_norm_parameters}（2H）"


@dataclass(frozen=True)
class AblationRow:
    """⑤ 消融表的一行."""

    variant: str
    cell: str
    train_loss: float
    eval_loss: float
    gap: float
    accuracy: float
    early_stop_step: int | None

    def line(self) -> str:
        """``rnn  | 全开 | 训练 0.262312 | 推理 3.892314 | 间隔 +3.630002 | 准确率 50.0% | 早停@2``."""
        stop = "—" if self.early_stop_step is None else f"早停@{self.early_stop_step}"
        return (
            f"{self.cell:<5} | {self.variant:<6} | 训练 {self.train_loss:.6f} | "
            f"推理 {self.eval_loss:.6f} | 间隔 {self.gap:+.6f} | 准确率 {self.accuracy:.1%} | {stop}"
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
        """``通过 batchnorm_matches_manual_standardization | 读数 0.000e+00 == 0 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return (
            f"{mark} {self.name:<48} | 读数 {self.reading:.3e} {self.bound} | {self.cross_check}"
        )


def axis_rows() -> tuple[AxisRow, ...]:
    """① 轴表：两条归一化轴的公式与签名."""
    return tuple(
        AxisRow(
            axis=axis,
            formula=AXIS_FORMULAS[axis],
            signature=AXIS_SIGNATURES[axis],
        )
        for axis in NORM_AXES
    )


def phase_rows() -> tuple[PhaseRow, ...]:
    """② 相表：同一个 batch 上两个相的输出（**μ/σ 来源不同**）."""
    training, _cache, running = batch_norm_forward(
        PHASE_BATCH, gamma=PHASE_GAMMA, beta=PHASE_BETA, phase=PHASE_TRAIN
    )
    inference, _inference_cache, _same = batch_norm_forward(
        PHASE_BATCH, gamma=PHASE_GAMMA, beta=PHASE_BETA, phase=PHASE_EVAL, running=running
    )
    rows: list[PhaseRow] = []
    for phase, output in ((PHASE_TRAIN, training), (PHASE_EVAL, inference)):
        flat = [value for row in output for value in row]
        rows.append(
            PhaseRow(
                phase=phase,
                statistic=PHASE_STATISTIC_SOURCES[phase],
                mean_of_means=math.fsum(flat) / len(flat),
                max_abs=max(abs(value) for value in flat),
            )
        )
    return tuple(rows)


def technique_rows() -> tuple[TechniqueRow, ...]:
    """③ 技巧表：四个技巧与它们的实现者."""
    return tuple(
        TechniqueRow(
            technique=technique,
            description=TECHNIQUE_DESCRIPTIONS[technique],
            source=TECHNIQUE_SOURCES[technique],
        )
        for technique in TECHNIQUES
    )


def parameter_rows() -> tuple[ParameterRow, ...]:
    """④ 参数表：BatchNorm 的参数量随隐藏宽线性增长（2H：γ 与 β 各 H 个）."""
    return tuple(
        ParameterRow(hidden=hidden, batch_norm_parameters=2 * hidden) for hidden in HIDDEN_CASES
    )


def ablation_reports() -> dict[str, dict[str, reg_train.TrainReport]]:
    """⑤ 消融表：四个变体在两种单元上各训一次（**最贵的一张表**）."""
    dataset = make_sign_dataset()
    reports: dict[str, dict[str, reg_train.TrainReport]] = {}
    for cell in ABLATION_CELLS:
        reports[cell] = reg_train.compare_configs(
            cell,
            dataset,
            ABLATION_VARIANTS,
            hidden_size=ABLATION_HIDDEN,
            epochs=ABLATION_EPOCHS,
            batch_size=ABLATION_BATCH_SIZE,
            seed=ABLATION_SEED,
        )
    return reports


def ablation_rows(
    reports: dict[str, dict[str, reg_train.TrainReport]] | None = None,
) -> tuple[AblationRow, ...]:
    """⑤ 消融表的行（``reports`` 可以外部传入，避免同一次会话里训练两遍）."""
    resolved = reports if reports is not None else ablation_reports()
    rows: list[AblationRow] = []
    for cell in ABLATION_CELLS:
        for variant, report in resolved[cell].items():
            early = None if report.early_stop is None else report.early_stop.best_step
            rows.append(
                AblationRow(
                    variant=variant,
                    cell=cell,
                    train_loss=report.final_loss,
                    eval_loss=report.eval_loss,
                    gap=report.loss_gap,
                    accuracy=report.accuracy,
                    early_stop_step=early if report.early_stop and report.early_stop.triggered else None,
                )
            )
    return tuple(rows)


def log_lines(
    reports: dict[str, dict[str, reg_train.TrainReport]] | None = None,
) -> tuple[str, ...]:
    """⑥ 日志表：把"全开"那一组的推理损失画成四张脸."""
    resolved = reports if reports is not None else ablation_reports()
    curve = tuple(record.eval_loss for record in resolved["lstm"]["全开"].epochs)
    lines: list[str] = []
    lines.append("sparkline（推理损失，越低越好）:")
    lines.append("  " + vz.sparkline(curve, width=len(curve)))
    lines.append("ASCII 折线:")
    for line in vz.loss_curve(curve, height=5, width=40):
        lines.append("  " + line)
    lines.append("条形图（最终准确率，越长越好）:")
    for line in vz.bar_chart(
        tuple(
            (f"{cell}-{variant}", report.accuracy)
            for cell in ABLATION_CELLS
            for variant, report in resolved[cell].items()
        ),
        width=20,
    ):
        lines.append("  " + line)
    lines.append("汇总裁剪（lstm / 全开）:")
    lines.append("  " + vz.curve_summary(curve))
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
                bound=check.bound_text(),
                cross_check=f"{check.left} vs {check.right}",
            )
        )
    return tuple(rows)


def study_lines(
    reports: dict[str, dict[str, reg_train.TrainReport]] | None = None,
) -> tuple[str, ...]:
    """一次跑完七张表（演示脚本与教程引用的是同一批读数）.

    ``reports`` 可以外部传入（避免同一个训练在一次会话里跑两遍）——
    没有它时第 ⑤ ⑥ 张表会**自己训一遍**。
    """
    lines: list[str] = []
    lines.append("== 1. 轴表（两条归一化轴：公式与签名）")
    for row in axis_rows():
        lines.append("  " + row.line())
    for axis in NORM_AXES:
        lines.append(f"  {axis:<8} : {AXIS_DESCRIPTIONS[axis]}")
    lines.append("== 2. 相表（同一个 batch 上两个相的输出）")
    for row in phase_rows():
        lines.append("  " + row.line())
    lines.append(f"  训练相：{BN_FORMULA}")
    lines.append(f"  推理相：{BN_EVAL_BACKWARD_FORMULA.split('；')[0]}")
    lines.append(f"  反向（训练相）：{BN_BACKWARD_FORMULA}")
    lines.append("== 3. 技巧表（四个技巧：谁实现）")
    for row in technique_rows():
        lines.append("  " + row.line())
    lines.append("== 4. 参数表（BatchNorm 加了几个参数）")
    for row in parameter_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 消融表（四个变体 × 两种单元：训练 / 推理 / 间隔 / 准确率）")
    for row in ablation_rows(reports):
        lines.append("  " + row.line())
    lines.append("== 6. 日志表（训练日志的四张脸）")
    for line in log_lines(reports):
        lines.append("  " + line)
    lines.append("== 7. 性质表（七条性质：是否通过 / 读数 / 两个来源）")
    for row in property_rows():
        lines.append("  " + row.line())
    return tuple(lines)


#: 本课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTY_NAMES = REGULARIZATION_PROPERTIES

#: 两个相的解释（供报告引用，不必各写一遍字符串）.
PHASE_NOTES = PHASE_DESCRIPTIONS

#: 一个"什么都不改"的 running 统计量（初值：μ=0、σ²=1）.
FRESH_RUNNING = initial_running(3)

__all__ = [
    "ABLATION_BATCH_SIZE",
    "ABLATION_CELLS",
    "ABLATION_EPOCHS",
    "ABLATION_HIDDEN",
    "ABLATION_SEED",
    "ABLATION_VARIANTS",
    "FRESH_RUNNING",
    "HIDDEN_CASES",
    "PHASE_BATCH",
    "PHASE_BETA",
    "PHASE_GAMMA",
    "PHASE_NOTES",
    "PROPERTY_NAMES",
    "AblationRow",
    "AxisRow",
    "ParameterRow",
    "PhaseRow",
    "PropertyRow",
    "TechniqueRow",
    "ablation_reports",
    "ablation_rows",
    "axis_rows",
    "log_lines",
    "parameter_rows",
    "phase_rows",
    "property_rows",
    "study_lines",
    "technique_rows",
]
