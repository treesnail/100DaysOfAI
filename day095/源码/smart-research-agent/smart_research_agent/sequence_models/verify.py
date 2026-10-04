"""``verify``：七条性质与三类判据（day094 / M8-D5）.

```text
相等（逐位 / 整数）  ① rnn_cell_forward 与手写递推逐位一致
                     ② i=0、f=1、o=1 时 LSTM 的记忆原样冻结（逐位）
                     ③ T 步与 1 步用同一组权重（逐位 + 整数）
                     ④ T 个输入 ⇒ T 个状态：h₀ 是初值不是输出（整数）
上界                 ⑤ ∂c_T/∂c_0 = ∏f_t（偏差 <= 1e-12）
                     ⑥ BPTT 的解析梯度与 day074 的数值差分一致（<= 1e-9）
下界                 ⑦ LSTM 的长程信号必须**明显强于** tanh-RNN（差 >= 0.5）
```

判据三类与 day092 同源：**相等**、**不超过上界**与**不超过下界**。
本课第一次出现"下界"判据的理由很具体：

```text
把 LSTM 与 RNN 写成同一个（记忆通道没有真正独立）时——
  ⑤ 仍然通过（两条路径都在这个"假的 LSTM"上算）
  ⑥ 仍然通过（梯度照样对）
  ④ 仍然通过（步数照样对）
  只有 ⑦ 会红：因为"LSTM 必须比 RNN 强"这件事被写成了一条不能少的断言。
```

这就是"没有下界判据时，把两个不同的东西写成同一个，报告会全绿"。

## 第 ⑥ 条为什么要用数值差分

BPTT 里最容易写串的是 **``dW`` 的累加**与 **``x`` / ``h`` 的转置**：
把 ``dW_x`` 与 ``dW_h`` 的第二个乘子互换、或把 ``dy·W`` 写成 ``W·dy``，
**形状全部仍然正确**。唯一能区分它们的是一把独立于这段推导的尺子——
day074 的数值差分。这一条同时钉住 RNN 与 LSTM 两条链。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.math_foundations.calculus import gradient
from smart_research_agent.sequence_models import gradients as gradients_module
from smart_research_agent.sequence_models import layers as layers_module
from smart_research_agent.sequence_models.errors import RecurrentError
from smart_research_agent.sequence_models.layers import (
    LSTMCell,
    LSTMSpec,
    RNNCell,
    RNNSpec,
    initialize_lstm,
    initialize_rnn,
    lstm_cell_forward,
    rnn_cell_forward,
)
from smart_research_agent.sequence_models.ops import parameter_count
from smart_research_agent.sequence_models.types import (
    CELL_LSTM,
    CELL_RNN,
    PROPERTY_BPTT_MATCHES_NUMERICAL,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT,
    PROPERTY_LSTM_GATES_CONTROL_STATE,
    PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE,
    PROPERTY_RNN_CELL_MATCHES_MANUAL,
    PROPERTY_SEQUENCE_ALIGNMENT,
    PROPERTY_WEIGHT_SHARING_ACROSS_TIME,
    RECURRENT_PROPERTIES,
)

#: 第 ①③④ 条用的写死样本（3 步、D=3、H=2）.
SEQUENCE_INPUTS: tuple[tuple[float, ...], ...] = (
    (0.5, -1.0, 0.25),
    (1.0, 0.0, -0.5),
    (-0.25, 0.75, 0.5),
)
RNN_SEED = 11
LSTM_SEED = 7

#: 第 ② 条用的"饱和门"偏置：``σ(40)`` 在双精度里**恰好是 1.0**（``e^{−40} < eps/2``），
#: ``σ(−40)`` 是 4.25e-18（配上 ``g = tanh(0) = 0`` 之后 ``i⊙g`` 恰好是 0.0）.
SATURATED_GATE_BIAS = 40.0

#: 第 ⑤ 条用的序列（4 步、D=2、H=2）.
FORGET_INPUTS: tuple[tuple[float, ...], ...] = (
    (1.0, -0.5),
    (-0.25, 1.5),
    (0.75, 0.0),
    (-1.0, -1.0),
)
FORGET_TOLERANCE = 1e-12

#: 第 ⑥ 条用的样本与容差（与 day093 同一把尺子、同一个量级）.
GRAD_TOLERANCE = 1e-9
GRAD_SEED: tuple[tuple[float, ...], ...] = (
    (1.0, -0.5),
    (0.25, 2.0),
    (-1.0, 0.5),
)

#: 第 ⑦ 条用的长程序列长度与两侧的读数.
LONG_RANGE_STEPS = 8
RNN_DECAY_WEIGHT = 0.5
LONG_RANGE_GAP = 0.5


@dataclass(frozen=True)
class Check:
    """一次性质校验的读数与判据（三类：相等 / 上界 / 下界）."""

    reading: float
    upper_bound: float | None = None
    lower_bound: float | None = None
    left: str = "-"
    right: str = "-"

    def passed(self) -> bool:
        """读数是否落在判据内（上界与下界同时生效）."""
        if self.upper_bound is not None and self.reading > self.upper_bound:
            return False
        return not (self.lower_bound is not None and self.reading < self.lower_bound)

    def bound_text(self) -> str:
        """判据的一行文本."""
        if self.lower_bound is not None and self.upper_bound is not None:
            return f"∈ [{self.lower_bound:.1e}, {self.upper_bound:.1e}]"
        if self.lower_bound is not None:
            return f">= {self.lower_bound:.1e}"
        if self.upper_bound is None:
            return "== 逐位"
        if self.upper_bound == 0.0:
            return "== 0"
        return f"<= {self.upper_bound:.1e}"


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：是否通过 + 现场读数 + 两个来源."""

    name: str
    passed: bool
    check: Check

    def line(self) -> str:
        """``通过 rnn_cell_matches_manual_recurrence | 读数 0.000e+00 == 0 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return (
            f"{mark} {self.name:<44} | 读数 {self.check.reading:.3e} "
            f"{self.check.bound_text()} | {self.check.left} vs {self.check.right}"
        )


@dataclass
class PropertyReport:
    """七条性质的汇总报告."""

    outcomes: list[PropertyOutcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        """通过的条数."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def total(self) -> int:
        """总条数."""
        return len(self.outcomes)

    def all_passed(self) -> bool:
        """是否全部通过."""
        return self.passed == self.total

    def lines(self) -> tuple[str, ...]:
        """逐行印出（演示脚本与教程引用的是同一批读数）."""
        return tuple(outcome.line() for outcome in self.outcomes)


# --------------------------------------------------------------------------- #
# 单条性质
# --------------------------------------------------------------------------- #


def _manual_rnn_step(cell: RNNCell, inputs, state, activation: str) -> tuple[float, ...]:
    """**独立于 layers.rnn_cell_forward** 的第二条实现：按定义显式写出那一行递推.

    它刻意不走 ``ops.matvec`` / ``ops.add_triple`` / ``activate``——一条性质要成立，
    "两条路径独立"这件事必须是真的（自证不算对账）。

    唯一与实现**共享**的是"求和用 ``math.fsum``"这条数值约定（它写在 ``ops`` 里）。
    因此第 ① 条把逐位比较钉在**结构**上：哪一个权重乘哪一个变量、三项以什么顺序相加。
    把 ``W_x`` 与 ``W_h`` 写反、或把矩阵转置写错，都会在这里现形；
    而"浮点求值顺序不同"（那属于数值而不是结构）不会。
    """
    hidden, width = cell.hidden_size, cell.input_size
    result: list[float] = []
    for i in range(hidden):
        from_inputs = math.fsum(cell.wx[i][j] * inputs[j] for j in range(width))
        from_state = math.fsum(cell.wh[i][j] * state[j] for j in range(hidden))
        total = from_inputs + from_state + cell.bias[i]
        if activation == "tanh":
            result.append(math.tanh(total))
        else:  # relu
            result.append(total if total > 0.0 else 0.0)
    return tuple(result)


def check_rnn_cell_matches_manual() -> PropertyOutcome:
    """① ``rnn_cell_forward`` 的一步与手写那一行递推逐位一致（读数 = 最大绝对差）."""
    cell = initialize_rnn(RNNSpec(3, 2, activation="tanh", seed=RNN_SEED))
    state = cell.initial_state()
    gap = 0.0
    for step in SEQUENCE_INPUTS:
        produced = rnn_cell_forward(cell, step, state)
        manual = _manual_rnn_step(cell, step, state, "tanh")
        gap = max(gap, max(abs(a - b) for a, b in zip(produced, manual, strict=True)))
        state = produced
    check = Check(reading=gap, upper_bound=0.0, left="layers.rnn_cell_forward", right="手写递推")
    return PropertyOutcome(PROPERTY_RNN_CELL_MATCHES_MANUAL, check.passed(), check)


def saturated_lstm(hidden_size: int = 2, input_size: int = 3) -> LSTMCell:
    """造一个"门被钉死"的 LSTM：``i = 0``、``f = 1``、``o = 1``、候选 ``g = 0``.

    所有权重为 0、四个偏置分别为 ``−40 / +40 / +40 / 0``：

    ```text
    σ(+40) = 1.0                  （恰好，因为 e^{−40} ≈ 4.25e-18 < eps/2）
    σ(−40) = 4.25e-18             （不是 0，但 g = tanh(0) = 0 ⇒ i⊙g 恰好是 0.0）
    ⇒ c_t = 1.0·c_{t−1} + 0.0 = c_{t−1}   （**逐位**相等）
    ```
    """
    rows = 4 * hidden_size
    zero_wx = tuple((0.0,) * input_size for _ in range(rows))
    zero_wh = tuple((0.0,) * hidden_size for _ in range(rows))
    bias = (
        (-SATURATED_GATE_BIAS,) * hidden_size
        + (SATURATED_GATE_BIAS,) * hidden_size
        + (SATURATED_GATE_BIAS,) * hidden_size
        + (0.0,) * hidden_size
    )
    return LSTMCell(wx=zero_wx, wh=zero_wh, bias=bias)


def check_lstm_gates_control_state() -> PropertyOutcome:
    """② ``i=0、f=1、o=1`` 时 LSTM 的记忆**原样冻结**（读数 = 最大漂移）."""
    cell = saturated_lstm(2)
    initial = (0.75, -0.4)
    state = ((0.0, 0.0), initial)
    drift = 0.0
    for step in SEQUENCE_INPUTS:
        new_hidden, new_cell = lstm_cell_forward(cell, step, state)
        drift = max(drift, max(abs(a - b) for a, b in zip(new_cell, initial, strict=True)))
        expected_hidden = tuple(math.tanh(value) for value in new_cell)
        drift = max(drift, max(abs(a - b) for a, b in zip(new_hidden, expected_hidden, strict=True)))
        state = (new_hidden, new_cell)
    check = Check(
        reading=drift,
        upper_bound=0.0,
        left="c_t（三门钉死）",
        right="c₀ = (0.75, −0.4)",
    )
    return PropertyOutcome(PROPERTY_LSTM_GATES_CONTROL_STATE, check.passed(), check)


def check_weight_sharing_across_time() -> PropertyOutcome:
    """③ T 步与 1 步用同一组权重：参数量与 T 无关，且每一步都能单独复算（逐位 + 整数）.

    读数 = ``|参数量(同一个 H,D 在不同 T 下) − 常数|`` 的最大值，
    再加上"第 t 步单独复算"与"整段第 t 步"的最大绝对差。
    """
    cell = initialize_rnn(RNNSpec(3, 2, activation="tanh", seed=RNN_SEED))
    # 参数量由公式给出，与 T **完全无关**：把它与参数本体自己数出来的数对齐
    count_gap = abs(parameter_count(CELL_RNN, cell.input_size, cell.hidden_size) - cell.parameter_count)
    states = layers_module.rnn_forward(cell, SEQUENCE_INPUTS)
    replay_gap = 0.0
    previous = cell.initial_state()
    for index, step in enumerate(SEQUENCE_INPUTS):
        # 用**单步函数**从 states[t−1] 复算第 t 步
        single = rnn_cell_forward(cell, step, previous)
        replay_gap = max(
            replay_gap, max(abs(a - b) for a, b in zip(single, states[index], strict=True))
        )
        previous = states[index]
    check = Check(
        reading=abs(count_gap) + replay_gap,
        upper_bound=0.0,
        left=f"参数量 {cell.parameter_count}（与 T 无关）+ 逐步复算",
        right="同一组权重",
    )
    return PropertyOutcome(PROPERTY_WEIGHT_SHARING_ACROSS_TIME, check.passed(), check)


def check_sequence_alignment() -> PropertyOutcome:
    """④ T 个输入 ⇒ T 个状态：``h₀`` 是初值，不出现在输出里（读数 = 步数差）."""
    rnn_cell = initialize_rnn(RNNSpec(3, 2, seed=RNN_SEED))
    lstm_cell = initialize_lstm(LSTMSpec(3, 2, seed=LSTM_SEED))
    expected = len(SEQUENCE_INPUTS)
    rnn_states = layers_module.rnn_forward(rnn_cell, SEQUENCE_INPUTS)
    hidden_states, cell_states = layers_module.lstm_forward(lstm_cell, SEQUENCE_INPUTS)
    gap = (
        abs(len(rnn_states) - expected)
        + abs(len(hidden_states) - expected)
        + abs(len(cell_states) - expected)
    )
    check = Check(
        reading=float(gap),
        upper_bound=0.0,
        left=f"rnn {len(rnn_states)} / lstm {len(hidden_states)} 个状态",
        right=f"T = {expected}",
    )
    return PropertyOutcome(PROPERTY_SEQUENCE_ALIGNMENT, check.passed(), check)


def check_lstm_cell_gradient_is_forget_product() -> PropertyOutcome:
    """⑤ ``∂c_T/∂c_0 = ∏f_t``：BPTT 的 carry 与"把遗忘门乘起来"一致（<= 1e-12）.

    为了让它**精确**成立，这里刻意把 ``W_h`` 置零：此时隐状态那条链不再回传
    （``W_hᵀ·dz = 0``），记忆通道上只剩 ``dC_{t−1} = dC_t ⊙ f_t``——
    于是两条路径（BPTT 的累乘 vs 把前向门读出来直接相乘）必须给出同一个向量。
    """
    cell = initialize_lstm(LSTMSpec(2, 2, seed=5))
    zero_wh = tuple((0.0,) * 2 for _ in range(4 * 2))
    cell = LSTMCell(wx=cell.wx, wh=zero_wh, bias=cell.bias)
    _hidden, _cells, caches = gradients_module.lstm_forward_with_cache(cell, FORGET_INPUTS)
    length = len(caches)
    seed = (1.0, 1.0)
    grad_hidden = tuple((0.0, 0.0) for _ in range(length))
    grad_cell = tuple(seed if index == length - 1 else (0.0, 0.0) for index in range(length))
    result = gradients_module.lstm_bptt(cell, caches, grad_hidden, grad_cell=grad_cell)
    product = [1.0, 1.0]
    for cache in caches:
        for index in range(2):
            product[index] *= cache.forget_gate[index]
    gap = max(abs(result.d_cell0[index] - product[index]) for index in range(2))
    check = Check(
        reading=gap,
        upper_bound=FORGET_TOLERANCE,
        left="BPTT 的 ∂L/∂c₀",
        right=f"∏f_t = ({product[0]:.6f}, {product[1]:.6f})",
    )
    return PropertyOutcome(PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT, check.passed(), check)


def _flatten_cell(cell) -> tuple[float, ...]:
    """把循环单元的权重压成一串数（**只为数值差分服务**，不进正式参数契约）."""
    flat: list[float] = []
    for row in cell.wx:
        flat.extend(row)
    for row in cell.wh:
        flat.extend(row)
    flat.extend(cell.bias)
    return tuple(flat)


def _unflatten_rnn(flat: tuple[float, ...], template: RNNCell) -> RNNCell:
    """按模板把一串数还原成 :class:`RNNCell`（**只为数值差分服务**）."""
    cursor = 0
    wx: list[tuple[float, ...]] = []
    for row in template.wx:
        wx.append(tuple(flat[cursor : cursor + len(row)]))
        cursor += len(row)
    wh: list[tuple[float, ...]] = []
    for row in template.wh:
        wh.append(tuple(flat[cursor : cursor + len(row)]))
        cursor += len(row)
    bias = tuple(flat[cursor : cursor + len(template.bias)])
    return RNNCell(wx=tuple(wx), wh=tuple(wh), bias=bias)


def _unflatten_lstm(flat: tuple[float, ...], template: LSTMCell) -> LSTMCell:
    """按模板把一串数还原成 :class:`LSTMCell`（**只为数值差分服务**）."""
    cursor = 0
    wx: list[tuple[float, ...]] = []
    for row in template.wx:
        wx.append(tuple(flat[cursor : cursor + len(row)]))
        cursor += len(row)
    wh: list[tuple[float, ...]] = []
    for row in template.wh:
        wh.append(tuple(flat[cursor : cursor + len(row)]))
        cursor += len(row)
    bias = tuple(flat[cursor : cursor + len(template.bias)])
    return LSTMCell(wx=tuple(wx), wh=tuple(wh), bias=bias)


def _weighted_sum(states, seeds) -> float:
    """``Σ_t Σ_i states[t][i]·seeds[t][i]``——数值差分要的那个标量目标."""
    return math.fsum(
        states[t][i] * seeds[t][i]
        for t in range(len(states))
        for i in range(len(states[0]))
    )


def check_bptt_matches_numerical() -> PropertyOutcome:
    """⑥ BPTT 的解析梯度与 day074 的数值差分一致（读数 = 两个单元的最大绝对差）."""
    rnn_cell = initialize_rnn(RNNSpec(3, 2, activation="tanh", seed=RNN_SEED))
    rnn_result = gradients_module.rnn_bptt(
        rnn_cell, gradients_module.rnn_forward_with_cache(rnn_cell, SEQUENCE_INPUTS)[1], GRAD_SEED
    )

    def rnn_objective(flat: tuple[float, ...]) -> float:
        probe = _unflatten_rnn(flat, rnn_cell)
        states = layers_module.rnn_forward(probe, SEQUENCE_INPUTS)
        return _weighted_sum(states, GRAD_SEED)

    numeric_rnn = gradient(rnn_objective, _flatten_cell(rnn_cell))
    worst = max(
        abs(a - b) for a, b in zip(_flatten_cell(rnn_result.cell), numeric_rnn, strict=True)
    )

    lstm_cell = initialize_lstm(LSTMSpec(3, 2, seed=LSTM_SEED))
    _hidden, _cells, lstm_caches = gradients_module.lstm_forward_with_cache(lstm_cell, SEQUENCE_INPUTS)
    lstm_result = gradients_module.lstm_bptt(lstm_cell, lstm_caches, GRAD_SEED)

    def lstm_objective(flat: tuple[float, ...]) -> float:
        probe = _unflatten_lstm(flat, lstm_cell)
        states = layers_module.lstm_forward(probe, SEQUENCE_INPUTS)[0]
        return _weighted_sum(states, GRAD_SEED)

    numeric_lstm = gradient(lstm_objective, _flatten_cell(lstm_cell))
    worst = max(
        worst,
        max(abs(a - b) for a, b in zip(_flatten_cell(lstm_result.cell), numeric_lstm, strict=True)),
    )
    check = Check(
        reading=worst,
        upper_bound=GRAD_TOLERANCE,
        left="rnn_bptt / lstm_bptt（解析）",
        right="math_foundations.calculus.gradient（数值）",
    )
    return PropertyOutcome(PROPERTY_BPTT_MATCHES_NUMERICAL, check.passed(), check)


def check_lstm_outlasts_rnn_long_range() -> PropertyOutcome:
    """⑦ LSTM 的长程信号必须明显强于 tanh-RNN（读数 = 两者之差，下界 0.5）.

    ```text
    RNN    W_h = 0.5、其余为 0、h₀ = 1      ⇒ ‖∂h_T/∂h_0‖ ≤ 0.5^T（T = 8 ⇒ ≤ 3.91e-3）
    LSTM   权重全 0、f = 1、i = 0、c₀ = 1   ⇒ ‖∂L/∂c₀‖ = 1.0（记忆通道原样带回）
    ```
    """
    inputs = tuple((0.0,) for _ in range(LONG_RANGE_STEPS))
    rnn_cell = RNNCell(
        wx=((0.0,),),
        wh=((RNN_DECAY_WEIGHT,),),
        bias=(0.0,),
    )
    rnn_carry = gradients_module.rnn_carry_norm(rnn_cell, inputs, state0=(1.0,))
    lstm_cell = LSTMCell(
        wx=((0.0,), (0.0,), (0.0,), (0.0,)),
        wh=((0.0,), (0.0,), (0.0,), (0.0,)),
        bias=(-SATURATED_GATE_BIAS, SATURATED_GATE_BIAS, SATURATED_GATE_BIAS, 0.0),
    )
    lstm_carry = gradients_module.lstm_cell_carry_norm(
        lstm_cell, inputs, state0=((0.0,), (1.0,))
    )
    check = Check(
        reading=lstm_carry - rnn_carry,
        lower_bound=LONG_RANGE_GAP,
        left=f"LSTM carry {lstm_carry:.6f}",
        right=f"tanh-RNN carry {rnn_carry:.3e}（≤ {RNN_DECAY_WEIGHT**LONG_RANGE_STEPS:.2e}）",
    )
    return PropertyOutcome(PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE, check.passed(), check)


#: 七条性质的名字 -> 检查函数（键顺序 = :data:`types.RECURRENT_PROPERTIES`）.
CHECKS = {
    PROPERTY_RNN_CELL_MATCHES_MANUAL: check_rnn_cell_matches_manual,
    PROPERTY_LSTM_GATES_CONTROL_STATE: check_lstm_gates_control_state,
    PROPERTY_WEIGHT_SHARING_ACROSS_TIME: check_weight_sharing_across_time,
    PROPERTY_SEQUENCE_ALIGNMENT: check_sequence_alignment,
    PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT: check_lstm_cell_gradient_is_forget_product,
    PROPERTY_BPTT_MATCHES_NUMERICAL: check_bptt_matches_numerical,
    PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE: check_lstm_outlasts_rnn_long_range,
}

if set(CHECKS) != set(RECURRENT_PROPERTIES):  # pragma: no cover - 导入期不变式
    raise RecurrentError(
        "性质名单与检查函数表不一致：少一条的性质会静默地不在报告里出现，"
        "而'少一条'与'它通过了'在读报告时长得一样。"
    )

if set(PROPERTY_DESCRIPTIONS) != set(RECURRENT_PROPERTIES):  # pragma: no cover - 导入期不变式
    raise RecurrentError("性质说明表与名单不一致。")


def check_all() -> PropertyReport:
    """跑完七条性质，返回汇总报告（顺序与 :data:`types.RECURRENT_PROPERTIES` 一致）."""
    report = PropertyReport()
    for name in RECURRENT_PROPERTIES:
        report.outcomes.append(CHECKS[name]())
    return report


#: 第 ① 条里被对照的单元名（供 study / demo 引用，避免各写一遍字符串）.
MANUAL_CELL = CELL_RNN
NUMERICAL_CELL = CELL_LSTM

__all__ = [
    "CHECKS",
    "FORGET_INPUTS",
    "FORGET_TOLERANCE",
    "GRAD_SEED",
    "GRAD_TOLERANCE",
    "LONG_RANGE_GAP",
    "LONG_RANGE_STEPS",
    "MANUAL_CELL",
    "NUMERICAL_CELL",
    "RNN_DECAY_WEIGHT",
    "RNN_SEED",
    "SATURATED_GATE_BIAS",
    "SEQUENCE_INPUTS",
    "Check",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_bptt_matches_numerical",
    "check_lstm_cell_gradient_is_forget_product",
    "check_lstm_gates_control_state",
    "check_lstm_outlasts_rnn_long_range",
    "check_rnn_cell_matches_manual",
    "check_sequence_alignment",
    "check_weight_sharing_across_time",
    "saturated_lstm",
]
