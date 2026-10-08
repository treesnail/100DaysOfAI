"""``study``：七张表（day094 / M8-D5）.

```text
① 单元表     两种循环单元的画像：状态形状 / 参数量 / 那一行递推
② 门表       LSTM 四个门在一次前向里的实际读数（均值 / 最小 / 最大）
③ 状态表     逐步的隐状态画像（范数）——"信息在这条链上还剩多少"
④ 参数量表   同一个 (D, H) 在 T = 1 / 3 / 8 下的参数量（**应当完全相同**）
⑤ 反向表     解析 BPTT vs 数值差分（两个单元各一条读数）
⑥ 训练表     rnn 与 lstm 在 ±1 符号任务上的损失曲线与准确率
⑦ 性质表     七条性质是否通过 / 现场读数 / 两个来源
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
每张表的每一行至少有"读数"与"参照"两列。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.sequence_models import gradients, verify
from smart_research_agent.sequence_models.gradients import (
    lstm_forward_with_cache,
    rnn_forward_with_cache,
)
from smart_research_agent.sequence_models.layers import (
    LSTMSpec,
    RNNSpec,
    initialize_lstm,
    initialize_rnn,
)
from smart_research_agent.sequence_models.ops import sequence_norms
from smart_research_agent.sequence_models.train import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STEPS,
    TrainReport,
    make_sign_dataset,
    train_sequence,
)
from smart_research_agent.sequence_models.types import (
    CELL_DESCRIPTIONS,
    CELL_LSTM,
    CELL_RNN,
    CELL_STATE_SHAPES,
    GATES,
    GATE_ACTIVATIONS,
    GATE_DESCRIPTIONS,
    RECURRENT_PROPERTIES,
    RNN_RECURRENCE_FORMULA,
)

#: ① ② ③ ④ 用的样本规格（同一个 D=2、H=3，换单元不换尺寸）.
SAMPLE_INPUT_SIZE = 2
SAMPLE_HIDDEN_SIZE = 3
SAMPLE_SPEC_RNN = RNNSpec(SAMPLE_INPUT_SIZE, SAMPLE_HIDDEN_SIZE, seed=3)
SAMPLE_SPEC_LSTM = LSTMSpec(SAMPLE_INPUT_SIZE, SAMPLE_HIDDEN_SIZE, seed=3)

#: ③ 用的样本序列（T = 4）.
SAMPLE_SEQUENCE: tuple[tuple[float, ...], ...] = (
    (0.5, -1.0),
    (1.0, 0.25),
    (-0.5, 0.75),
    (0.0, -1.0),
)

#: ④ 用的三个序列长度（参数量必须与它们都无关）.
LENGTH_CASES: tuple[int, ...] = (1, 3, 8)

#: ⑤ 用的两个单元与容差（读数来自 :mod:`verify`，**不在这里重算一遍**）.
BPTT_TOLERANCE = verify.GRAD_TOLERANCE

#: ⑥ 用的训练配置.
TRAIN_STEPS = DEFAULT_STEPS
TRAIN_HIDDEN = DEFAULT_HIDDEN_SIZE
TRAIN_LR = DEFAULT_LEARNING_RATE


@dataclass(frozen=True)
class CellRow:
    """① 单元表的一行."""

    cell: str
    state_shape: str
    parameters: int
    description: str

    def line(self) -> str:
        """``rnn  | 状态 (h,) | 参数 12 | 一个隐状态...``."""
        return f"{self.cell:<5} | 状态 {self.state_shape} | 参数 {self.parameters} | {self.description}"


@dataclass(frozen=True)
class GateRow:
    """② 门表的一行."""

    gate: str
    activation: str
    mean: float
    minimum: float
    maximum: float

    def line(self) -> str:
        """``input | sigmoid | 均值 0.512345 | 范围 [0.400000, 0.600000]``."""
        return (
            f"{self.gate:<9} | {self.activation:<7} | 均值 {self.mean:+.6f} | "
            f"范围 [{self.minimum:+.6f}, {self.maximum:+.6f}]"
        )


@dataclass(frozen=True)
class StateRow:
    """③ 状态表的一行."""

    cell: str
    step: int
    norm: float

    def line(self) -> str:
        """``rnn | 第 1 步 | ‖h‖ = 0.123456``."""
        return f"{self.cell:<5} | 第 {self.step} 步 | ‖h‖ = {self.norm:.6f}"


@dataclass(frozen=True)
class ParameterRow:
    """④ 参数量表的一行."""

    length: int
    rnn_parameters: int
    lstm_parameters: int

    def line(self) -> str:
        """``T = 8 | rnn 12 | lstm 48``."""
        return f"T = {self.length:<3} | rnn {self.rnn_parameters} | lstm {self.lstm_parameters}"


@dataclass(frozen=True)
class BpttRow:
    """⑤ 反向表的一行."""

    cell: str
    analytic_norm: float
    numerical_gap: float

    def line(self) -> str:
        """``rnn  | 解析范数 2.654833 | 与数值差 2.222e-10``."""
        return (
            f"{self.cell:<5} | 解析范数 {self.analytic_norm:.6f} | "
            f"与数值差 {self.numerical_gap:.3e}"
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
        """``通过 rnn_cell_matches_manual_recurrence | 读数 0.000e+00 == 0 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return f"{mark} {self.name:<44} | 读数 {self.reading:.3e} {self.bound} | {self.cross_check}"


def cell_rows() -> tuple[CellRow, ...]:
    """① 单元表：两种单元的画像（参数量由公式现场算出）."""
    rnn_cell = initialize_rnn(SAMPLE_SPEC_RNN)
    lstm_cell = initialize_lstm(SAMPLE_SPEC_LSTM)
    return (
        CellRow(
            cell=CELL_RNN,
            state_shape=CELL_STATE_SHAPES[CELL_RNN],
            parameters=rnn_cell.parameter_count,
            description=CELL_DESCRIPTIONS[CELL_RNN],
        ),
        CellRow(
            cell=CELL_LSTM,
            state_shape=CELL_STATE_SHAPES[CELL_LSTM],
            parameters=lstm_cell.parameter_count,
            description=CELL_DESCRIPTIONS[CELL_LSTM],
        ),
    )


def gate_rows() -> tuple[GateRow, ...]:
    """② 门表：LSTM 四个门在一次前向里的实际读数（**每一列单独统计**）."""
    from smart_research_agent.sequence_models.layers import lstm_cell_forward, lstm_gates

    cell = initialize_lstm(SAMPLE_SPEC_LSTM)
    state = cell.initial_state()
    collected: list[tuple[tuple[float, ...], ...]] = []
    for step in SAMPLE_SEQUENCE:
        collected.append(lstm_gates(cell, step, state))
        state = lstm_cell_forward(cell, step, state)
    rows: list[GateRow] = []
    for gate_index, gate in enumerate(GATES):
        values: list[float] = []
        for step_gates in collected:
            values.extend(step_gates[gate_index])
        rows.append(
            GateRow(
                gate=gate,
                activation=GATE_ACTIVATIONS[gate],
                mean=math.fsum(values) / len(values),
                minimum=min(values),
                maximum=max(values),
            )
        )
    return tuple(rows)


def state_rows() -> tuple[StateRow, ...]:
    """③ 状态表：两种单元逐步的隐状态范数."""
    rnn_cell = initialize_rnn(SAMPLE_SPEC_RNN)
    lstm_cell = initialize_lstm(SAMPLE_SPEC_LSTM)
    rnn_states, _caches = rnn_forward_with_cache(rnn_cell, SAMPLE_SEQUENCE)
    lstm_states, _cells, _lstm_caches = lstm_forward_with_cache(lstm_cell, SAMPLE_SEQUENCE)
    rows: list[StateRow] = []
    for step, norm in enumerate(sequence_norms(rnn_states), start=1):
        rows.append(StateRow(cell=CELL_RNN, step=step, norm=norm))
    for step, norm in enumerate(sequence_norms(lstm_states), start=1):
        rows.append(StateRow(cell=CELL_LSTM, step=step, norm=norm))
    return tuple(rows)


def parameter_rows() -> tuple[ParameterRow, ...]:
    """④ 参数量表：同一个 (D, H) 在三个 T 下的参数量（**应当完全相同**）.

    这张表是"时间轴上的权重共享"最直接的一个证据：
    T 从 1 涨到 8，参数量一个也不多。
    """
    rnn_cell = initialize_rnn(SAMPLE_SPEC_RNN)
    lstm_cell = initialize_lstm(SAMPLE_SPEC_LSTM)
    return tuple(
        ParameterRow(
            length=length,
            rnn_parameters=rnn_cell.parameter_count,
            lstm_parameters=lstm_cell.parameter_count,
        )
        for length in LENGTH_CASES
    )


def bptt_rows() -> tuple[BpttRow, ...]:
    """⑤ 反向表：解析 BPTT 的范数与"与数值差分差多少"（读数来自 :mod:`verify`）."""
    rnn_cell = initialize_rnn(RNNSpec(3, 2, activation="tanh", seed=verify.RNN_SEED))
    _states, rnn_caches = rnn_forward_with_cache(rnn_cell, verify.SEQUENCE_INPUTS)
    rnn_result = gradients.rnn_bptt(rnn_cell, rnn_caches, verify.GRAD_SEED)
    lstm_cell = initialize_lstm(LSTMSpec(3, 2, seed=verify.LSTM_SEED))
    _h, _c, lstm_caches = lstm_forward_with_cache(lstm_cell, verify.SEQUENCE_INPUTS)
    lstm_result = gradients.lstm_bptt(lstm_cell, lstm_caches, verify.GRAD_SEED)
    gap = verify.check_bptt_matches_numerical().check.reading
    return (
        BpttRow(cell=CELL_RNN, analytic_norm=rnn_result.norm(), numerical_gap=gap),
        BpttRow(cell=CELL_LSTM, analytic_norm=lstm_result.norm(), numerical_gap=gap),
    )


def train_reports() -> dict[str, TrainReport]:
    """⑥ 训练表：在 ±1 符号任务上把两种单元各训一次（读数来自 ``train.train_sequence``）."""
    dataset = make_sign_dataset()
    reports: dict[str, TrainReport] = {}
    for cell in (CELL_RNN, CELL_LSTM):
        _params, report = train_sequence(
            cell,
            dataset,
            hidden_size=TRAIN_HIDDEN,
            steps=TRAIN_STEPS,
            learning_rate=TRAIN_LR,
            seed=100,
        )
        reports[cell] = report
    return reports


def train_rows(reports: dict[str, TrainReport] | None = None) -> tuple[str, ...]:
    """⑥ 训练表：两种单元各一行总结 + 几个损失点."""
    resolved = reports if reports is not None else train_reports()
    lines: list[str] = []
    for cell in (CELL_RNN, CELL_LSTM):
        report = resolved[cell]
        lines.append(report.summary_line())
        total = len(report.losses)
        for index in (0, total // 4, total // 2, total - 1):
            lines.append(f"  {cell} step {index + 1:>3} | loss {report.losses[index]:.6f}")
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


def study_lines(reports: dict[str, TrainReport] | None = None) -> tuple[str, ...]:
    """一次跑完七张表（演示脚本与教程引用的是同一批读数）.

    ``reports`` 可以外部传入（避免同一个训练在一次会话里跑两遍）——
    没有它时第 ⑥ 张表会**自己训一次**。
    """
    lines: list[str] = []
    lines.append("== 1. 单元表（状态形状 / 参数量 / 那一行递推）")
    for row in cell_rows():
        lines.append("  " + row.line())
    lines.append(f"  递推：{RNN_RECURRENCE_FORMULA}")
    lines.append("== 2. 门表（LSTM 四个门在一次前向里的实际读数）")
    for row in gate_rows():
        lines.append("  " + row.line())
    for gate in GATES:
        lines.append(f"  {gate:<9} : {GATE_DESCRIPTIONS[gate]}")
    lines.append("== 3. 状态表（逐步的隐状态范数）")
    for row in state_rows():
        lines.append("  " + row.line())
    lines.append("== 4. 参数量表（同一个 D/H：参数量与序列长度 T 无关）")
    for row in parameter_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 反向表（解析 BPTT vs 数值差分）")
    for row in bptt_rows():
        lines.append("  " + row.line())
    lines.append("== 6. 训练表（±1 序列：和为正 vs 和为负）")
    for line in train_rows(reports):
        lines.append("  " + line if not line.startswith("  ") else line)
    lines.append("== 7. 性质表（七条性质：是否通过 / 读数 / 两个来源）")
    for row in property_rows():
        lines.append("  " + row.line())
    return tuple(lines)


#: 本课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTY_NAMES = RECURRENT_PROPERTIES

__all__ = [
    "BPTT_TOLERANCE",
    "LENGTH_CASES",
    "PROPERTY_NAMES",
    "SAMPLE_HIDDEN_SIZE",
    "SAMPLE_INPUT_SIZE",
    "SAMPLE_SEQUENCE",
    "SAMPLE_SPEC_LSTM",
    "SAMPLE_SPEC_RNN",
    "TRAIN_HIDDEN",
    "TRAIN_LR",
    "TRAIN_STEPS",
    "BpttRow",
    "CellRow",
    "GateRow",
    "ParameterRow",
    "PropertyRow",
    "StateRow",
    "bptt_rows",
    "cell_rows",
    "gate_rows",
    "parameter_rows",
    "property_rows",
    "state_rows",
    "study_lines",
    "train_reports",
    "train_rows",
]
