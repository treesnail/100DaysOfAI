"""``layers``：一个循环单元的**可复现实现**（day094 / M8-D5）.

```text
RNNSpec / RNNCell     一个状态：h_t = act(W_h·h_{t−1} + W_x·x_t + b)
LSTMSpec / LSTMCell   一对状态：(h_t, c_t)，四个门各有一套权重
initialize_rnn / initialize_lstm   用 day089 的 LCG 造权重（**同一份 spec 两次逐位相同**）
rnn_cell_forward / lstm_cell_forward   一步
rnn_forward / lstm_forward             整段（**T 个输入 ⇒ T 个状态**）
```

## 一、权重是怎么放的：两个面板拼成一个矩阵

```text
RNN    wx (H, D)      wh (H, H)      b (H,)
LSTM   wx (4H, D)     wh (4H, H)     b (4H,)      ← 4H 行按 i / f / o / g 分四块
```

把 ``W_x`` 与 ``W_h`` 的贡献加在一起之后，一步就是一次仿射——
这正是"同一组权重在每一个时刻被复用"在代码里的样子：
**循环体只有一个**（:func:`rnn_cell_forward` / :func:`lstm_cell_forward`），
它被 for 循环调用了 T 次。

## 二、初始化沿用 day089 的 LCG

```text
limit = 1/√fan_in，   fan_in = D + H（这一步的输入 + 上一步的状态）
权重 ~ U(−limit, +limit)，偏置同尺度
```

## 三、一个**刻意的例外**：遗忘门偏置初始化为 1

它是本课唯一一处"不由 LCG 决定"的常数（:data:`types.DEFAULT_FORGET_BIAS`）：

```text
f = σ(b_f)；b_f = 0 ⇒ f ≈ 0.5（记忆每一步被砍掉一半）
              b_f = 1 ⇒ f ≈ 0.731（记忆通道一开始就基本连通）
```

这是 PyTorch / fastai 圈子里最常见的做法，本包把它写成一个常量并逐字记进笔记——
**它是刻意写死的，不是漏掉了随机性**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import activate
from smart_research_agent.neural_basics.layers import lcg_stream
from smart_research_agent.sequence_models.errors import ParameterError, ShapeError
from smart_research_agent.sequence_models.ops import (
    add_triple,
    as_matrix,
    as_sequence,
    as_vector,
    checked_positive_int,
    gate_blocks,
    matvec,
    parameter_count,
    vec_add,
    vec_hadamard,
    zeros,
)
from smart_research_agent.sequence_models.types import (
    CELL_LSTM,
    CELL_RNN,
    CELL_TYPES,
    DEFAULT_FORGET_BIAS,
    GATE_ACTIVATIONS,
    GATES,
    RNN_ACTIVATIONS,
)

# --------------------------------------------------------------------------- #
# 1. 朴素循环单元
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RNNSpec:
    """一个朴素循环单元的画像（``activation`` 只能是 :data:`types.RNN_ACTIVATIONS` 里的一个）."""

    input_size: int
    hidden_size: int
    activation: str = "tanh"
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_size", checked_positive_int(self.input_size, name="input_size"))
        object.__setattr__(self, "hidden_size", checked_positive_int(self.hidden_size, name="hidden_size"))
        if self.activation not in RNN_ACTIVATIONS:
            raise ParameterError(
                f"未知的循环激活 {self.activation!r}：可用取值 {list(RNN_ACTIVATIONS)}"
                "（softmax 是逐行激活，用它会破坏「每个单元一个状态」这件事）。"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ParameterError(f"seed 必须是非负整数，收到 {self.seed!r}。")

    @property
    def cell(self) -> str:
        """单元名（本课的两个单元之一）."""
        return CELL_RNN

    @property
    def fan_in(self) -> int:
        """每一行相连的输入个数 ``D + H``（这一步的输入 + 上一步的状态）."""
        return self.input_size + self.hidden_size

    @property
    def parameter_count(self) -> int:
        """参数量 ``H·(D + H + 1)``——**与序列长度无关**."""
        return parameter_count(CELL_RNN, self.input_size, self.hidden_size)

    def line(self) -> str:
        """一行说明：``rnn(D=2→H=3, act=tanh) | 参数 18 | 与 T 无关``."""
        return (
            f"rnn(D={self.input_size}→H={self.hidden_size}, act={self.activation}) | "
            f"参数 {self.parameter_count} | 与 T 无关"
        )


@dataclass(frozen=True)
class RNNCell:
    """朴素循环单元的参数本体：``wx`` ``wh`` 与偏置 ``b``."""

    wx: Matrix
    wh: Matrix
    bias: Vector

    def __post_init__(self) -> None:
        checked_wx = as_matrix(self.wx, name="wx")
        checked_wh = as_matrix(self.wh, name="wh")
        checked_bias = as_vector(self.bias, name="bias")
        hidden = len(checked_wx)
        if len(checked_wh) != hidden:
            raise ShapeError(
                f"wx 有 {hidden} 行而 wh 有 {len(checked_wh)} 行：两者都必须是隐藏宽 H。"
            )
        if len(checked_wh[0]) != hidden:
            raise ShapeError(
                f"wh 的列数 {len(checked_wh[0])} 与隐藏宽 {hidden} 不一致："
                "W_h 把 h_{t−1} 映回 H 维，因此它必须是方阵。"
            )
        if len(checked_bias) != hidden:
            raise ShapeError(f"bias 长度 {len(checked_bias)} 与隐藏宽 {hidden} 不一致。")
        object.__setattr__(self, "wx", checked_wx)
        object.__setattr__(self, "wh", checked_wh)
        object.__setattr__(self, "bias", checked_bias)

    @property
    def input_size(self) -> int:
        """输入维 D."""
        return len(self.wx[0])

    @property
    def hidden_size(self) -> int:
        """隐藏宽 H."""
        return len(self.wx)

    @property
    def parameter_count(self) -> int:
        """参数量 = wx 的元素 + wh 的元素 + 偏置个数."""
        return len(self.wx) * len(self.wx[0]) + len(self.wh) * len(self.wh[0]) + len(self.bias)

    def initial_state(self) -> Vector:
        """缺省初值 ``h₀``（全零）."""
        return zeros(self.hidden_size)

    def line(self) -> str:
        """一行说明：``rnn cell H=3, D=2 | 参数 18``."""
        return f"rnn cell H={self.hidden_size}, D={self.input_size} | 参数 {self.parameter_count}"


def initialize_rnn(spec: RNNSpec) -> RNNCell:
    """按 spec 造一个循环单元（LCG + 可注入种子 ⇒ **逐位可复现**）."""
    if not isinstance(spec, RNNSpec):
        raise ParameterError(f"initialize_rnn 需要一个 RNNSpec，收到 {type(spec).__name__}。")
    hidden, width = spec.hidden_size, spec.input_size
    limit = 1.0 / math.sqrt(spec.fan_in)
    uniform = lcg_stream(spec.seed, hidden * width + hidden * hidden + hidden)
    cursor = 0
    wx_rows: list[tuple[float, ...]] = []
    for _ in range(hidden):
        row = tuple(uniform[cursor + column] * 2.0 * limit - limit for column in range(width))
        cursor += width
        wx_rows.append(row)
    wh_rows: list[tuple[float, ...]] = []
    for _ in range(hidden):
        row = tuple(uniform[cursor + column] * 2.0 * limit - limit for column in range(hidden))
        cursor += hidden
        wh_rows.append(row)
    bias = tuple(uniform[cursor + index] * 2.0 * limit - limit for index in range(hidden))
    return RNNCell(wx=tuple(wx_rows), wh=tuple(wh_rows), bias=bias)


def rnn_pre_activation(cell: RNNCell, inputs: Vector, state: Vector) -> Vector:
    """一步的**激活之前**的值 ``z_t = W_x·x_t + W_h·h_{t−1} + b``.

    反向需要它（``tanh'`` / ``relu'`` 都要看激活前的符号），因此这一行被单独提出来，
    而不是在 :func:`rnn_cell_forward` 里算完就丢。
    """
    checked_inputs = as_vector(inputs, name="inputs")
    checked_state = as_vector(state, name="state")
    if len(checked_inputs) != cell.input_size:
        raise ShapeError(
            f"这一步的输入宽度 {len(checked_inputs)} 与该单元的输入维 {cell.input_size} 不一致。"
        )
    if len(checked_state) != cell.hidden_size:
        raise ShapeError(
            f"上一步的状态宽度 {len(checked_state)} 与该单元的隐藏宽 {cell.hidden_size} 不一致。"
        )
    return add_triple(matvec(cell.wx, checked_inputs), matvec(cell.wh, checked_state), cell.bias)


def rnn_cell_forward(
    cell: RNNCell, inputs: Vector, state: Vector, *, activation: str = "tanh"
) -> Vector:
    """朴素循环单元的**一步**：``h_t = act(W_h·h_{t−1} + W_x·x_t + b)``.

    激活复用 day089 的 :func:`neural_basics.activations.activate`：
    本模块不重写 tanh，也不重写 relu。
    """
    if activation not in RNN_ACTIVATIONS:
        raise ParameterError(
            f"未知的循环激活 {activation!r}：可用取值 {list(RNN_ACTIVATIONS)}。"
        )
    return activate(activation, rnn_pre_activation(cell, inputs, state))


def rnn_forward(
    cell: RNNCell,
    inputs: tuple[Vector, ...],
    *,
    state0: Vector | None = None,
    activation: str = "tanh",
) -> tuple[Vector, ...]:
    """整段前向：**T 个输入 ⇒ T 个状态**（``h₁ … h_T``；``h₀`` 是初值，不在输出里）.

    这条约定是反向能对齐的前提：``states[t]`` 与 ``inputs[t]`` 必须对应同一个时刻。
    把 ``h₀`` 也算进输出会得到 ``T+1`` 个状态——它**不会报错**，
    只会让全部状态整体错位一格（见 :class:`errors.TimeStepError`）。
    """
    checked_inputs = as_sequence(inputs, name="inputs")
    initial = cell.initial_state() if state0 is None else as_vector(state0, name="state0")
    if len(initial) != cell.hidden_size:
        raise ShapeError(f"初值宽度 {len(initial)} 与该单元的隐藏宽 {cell.hidden_size} 不一致。")
    states: list[Vector] = []
    current = initial
    for step in checked_inputs:
        current = rnn_cell_forward(cell, step, current, activation=activation)
        states.append(current)
    return tuple(states)


# --------------------------------------------------------------------------- #
# 2. 长短期记忆单元
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LSTMSpec:
    """一个 LSTM 单元的画像（门激活固定为 ``sigmoid``、候选固定为 ``tanh``）."""

    input_size: int
    hidden_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_size", checked_positive_int(self.input_size, name="input_size"))
        object.__setattr__(self, "hidden_size", checked_positive_int(self.hidden_size, name="hidden_size"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ParameterError(f"seed 必须是非负整数，收到 {self.seed!r}。")

    @property
    def cell(self) -> str:
        """单元名."""
        return CELL_LSTM

    @property
    def fan_in(self) -> int:
        """每一行相连的输入个数 ``D + H``（四个门各自如此）."""
        return self.input_size + self.hidden_size

    @property
    def parameter_count(self) -> int:
        """参数量 ``4H·(D + H + 1)``——**与序列长度无关**."""
        return parameter_count(CELL_LSTM, self.input_size, self.hidden_size)

    def line(self) -> str:
        """一行说明：``lstm(D=2→H=3) | 参数 72 | 与 T 无关``."""
        return (
            f"lstm(D={self.input_size}→H={self.hidden_size}) | "
            f"参数 {self.parameter_count} | 与 T 无关"
        )


@dataclass(frozen=True)
class LSTMCell:
    """LSTM 的参数本体：``wx`` ``wh`` 与偏置都是 ``4H`` 行，按 i / f / o / g 分块.

    形状约定与 ``torch.nn.LSTMCell`` 的四个 ``weight_ih_* / weight_hh_*`` 拼接后一致——
    只是 torch 把它们存成四个独立的张量，本包把四块**拼成一个矩阵**并把顺序写进
    :data:`types.GATES` 与 :data:`types.GATE_ROW_BLOCKS`。
    """

    wx: Matrix
    wh: Matrix
    bias: Vector

    def __post_init__(self) -> None:
        checked_wx = as_matrix(self.wx, name="wx")
        checked_wh = as_matrix(self.wh, name="wh")
        checked_bias = as_vector(self.bias, name="bias")
        rows = len(checked_wx)
        if rows % 4 != 0:
            raise ShapeError(f"wx 有 {rows} 行，不能被 4 整除：LSTM 的四个门要求行数是 4H。")
        if len(checked_wh) != rows:
            raise ShapeError(f"wh 有 {len(checked_wh)} 行而 wx 有 {rows} 行：两者必须同为 4H。")
        if len(checked_wh[0]) != rows // 4:
            raise ShapeError(
                f"wh 的列数 {len(checked_wh[0])} 与 H = 4H/4 = {rows // 4} 不一致："
                "W_h 把 h_{t−1} 映回 H 维，因此它必须是 (4H, H)。"
            )
        if len(checked_bias) != rows:
            raise ShapeError(f"bias 长度 {len(checked_bias)} 与 4H = {rows} 不一致。")
        object.__setattr__(self, "wx", checked_wx)
        object.__setattr__(self, "wh", checked_wh)
        object.__setattr__(self, "bias", checked_bias)

    @property
    def input_size(self) -> int:
        """输入维 D."""
        return len(self.wx[0])

    @property
    def hidden_size(self) -> int:
        """隐藏宽 H."""
        return len(self.wx) // 4

    @property
    def parameter_count(self) -> int:
        """参数量 = wx 元素 + wh 元素 + 偏置个数."""
        return len(self.wx) * len(self.wx[0]) + len(self.wh) * len(self.wh[0]) + len(self.bias)

    def initial_state(self) -> tuple[Vector, Vector]:
        """缺省初值 ``(h₀, c₀)``（两个全零向量）."""
        return zeros(self.hidden_size), zeros(self.hidden_size)

    def line(self) -> str:
        """一行说明：``lstm cell H=3, D=2 | 参数 72``."""
        return f"lstm cell H={self.hidden_size}, D={self.input_size} | 参数 {self.parameter_count}"


def initialize_lstm(spec: LSTMSpec) -> LSTMCell:
    """按 spec 造一个 LSTM 单元（LCG + **遗忘门偏置 = 1** 这一个刻意的例外）."""
    if not isinstance(spec, LSTMSpec):
        raise ParameterError(f"initialize_lstm 需要一个 LSTMSpec，收到 {type(spec).__name__}。")
    hidden, width = spec.hidden_size, spec.input_size
    rows = 4 * hidden
    limit = 1.0 / math.sqrt(spec.fan_in)
    uniform = lcg_stream(spec.seed, rows * width + rows * hidden + rows)
    cursor = 0
    wx_rows: list[tuple[float, ...]] = []
    for _ in range(rows):
        row = tuple(uniform[cursor + column] * 2.0 * limit - limit for column in range(width))
        cursor += width
        wx_rows.append(row)
    wh_rows: list[tuple[float, ...]] = []
    for _ in range(rows):
        row = tuple(uniform[cursor + column] * 2.0 * limit - limit for column in range(hidden))
        cursor += hidden
        wh_rows.append(row)
    bias = [uniform[cursor + index] * 2.0 * limit - limit for index in range(rows)]
    # 刻意的例外：遗忘门那一块（行 [H, 2H)）写成 1.0，不是 0，也不是 LCG 的数
    for index in range(hidden, 2 * hidden):
        bias[index] = DEFAULT_FORGET_BIAS
    return LSTMCell(wx=tuple(wx_rows), wh=tuple(wh_rows), bias=tuple(bias))


def lstm_pre_activation(
    cell: LSTMCell, inputs: Vector, state: tuple[Vector, Vector]
) -> Vector:
    """这一步四个门的**打分** ``z = W_x·x_t + W_h·h_{t−1} + b``（长度 4H）.

    门的激活在 :func:`lstm_gates` 里做——把"打分"与"激活"分成两步，
    是因为反向对这两步的处理完全不同（激活要过导数，线性部分只要转置）。
    """
    checked_inputs = as_vector(inputs, name="inputs")
    checked_h, checked_c = state
    if len(checked_inputs) != cell.input_size:
        raise ShapeError(
            f"这一步的输入宽度 {len(checked_inputs)} 与该单元的输入维 {cell.input_size} 不一致。"
        )
    if len(checked_h) != cell.hidden_size or len(checked_c) != cell.hidden_size:
        raise ShapeError(
            f"上一步的 (h, c) 宽度必须是 H = {cell.hidden_size}，收到 "
            f"({len(checked_h)}, {len(checked_c)})。"
        )
    return add_triple(matvec(cell.wx, checked_inputs), matvec(cell.wh, checked_h), cell.bias)


def lstm_gates(
    cell: LSTMCell, inputs: Vector, state: tuple[Vector, Vector]
) -> tuple[Vector, Vector, Vector, Vector]:
    """算出一个 LSTM 单元在这一步的四个门 ``(i, f, o, g)``.

    ```text
    z = 4H 的打分   →   切成四块（顺序 i / f / o / g）
    i = σ(z_i)   f = σ(z_f)   o = σ(z_o)   g = tanh(z_g)
    ```
    """
    raw_i, raw_f, raw_o, raw_g = gate_blocks(
        lstm_pre_activation(cell, inputs, state), cell.hidden_size
    )
    return (
        activate(GATE_ACTIVATIONS[GATES[0]], raw_i),
        activate(GATE_ACTIVATIONS[GATES[1]], raw_f),
        activate(GATE_ACTIVATIONS[GATES[2]], raw_o),
        activate(GATE_ACTIVATIONS[GATES[3]], raw_g),
    )


def lstm_cell_forward(
    cell: LSTMCell, inputs: Vector, state: tuple[Vector, Vector]
) -> tuple[Vector, Vector]:
    """LSTM 的**一步**：``c_t = f⊙c_{t−1} + i⊙g``、``h_t = o⊙tanh(c_t)``.

    返回 ``(h_t, c_t)``。记忆通道的那一行是**加法**：

    ```text
    c_t = f ⊙ c_{t−1} + i ⊙ g
      ↑      ↑             ↑
      |      保留多少旧记忆   写多少新记忆
      这条加法通道就是"梯度高速公路"——反向时梯度沿它走只乘 f
    ```
    """
    checked_h, checked_c = state
    input_gate, forget_gate, output_gate, candidate = lstm_gates(
        cell, inputs, (checked_h, checked_c)
    )
    new_cell = vec_add(
        vec_hadamard(forget_gate, checked_c), vec_hadamard(input_gate, candidate)
    )
    new_hidden = vec_hadamard(output_gate, activate("tanh", new_cell))
    return new_hidden, new_cell


def lstm_forward(
    cell: LSTMCell,
    inputs: tuple[Vector, ...],
    *,
    state0: tuple[Vector, Vector] | None = None,
) -> tuple[tuple[Vector, ...], tuple[Vector, ...]]:
    """整段前向：**T 个输入 ⇒ T 个 h 与 T 个 c**（``h₀ / c₀`` 是初值，不在输出里）.

    返回 ``(hidden_states, cell_states)``。两者都是 ``T`` 步——
    这就是 :data:`types.PROPERTY_SEQUENCE_ALIGNMENT` 在 LSTM 上的形态。
    """
    checked_inputs = as_sequence(inputs, name="inputs")
    initial_h, initial_c = cell.initial_state() if state0 is None else state0
    if len(initial_h) != cell.hidden_size or len(initial_c) != cell.hidden_size:
        raise ShapeError("初值 (h₀, c₀) 的宽度必须都等于隐藏宽 H。")
    hidden_states: list[Vector] = []
    cell_states: list[Vector] = []
    current_h = as_vector(initial_h, name="h0")
    current_c = as_vector(initial_c, name="c0")
    for step in checked_inputs:
        current_h, current_c = lstm_cell_forward(cell, step, (current_h, current_c))
        hidden_states.append(current_h)
        cell_states.append(current_c)
    return tuple(hidden_states), tuple(cell_states)


def sequence_length(inputs: tuple[Vector, ...]) -> int:
    """序列长度 T（空序列抛 :class:`errors.TimeStepError`）."""
    return len(as_sequence(inputs, name="inputs"))


def cell_types() -> tuple[str, ...]:
    """本课支持的两种单元（转发 :data:`types.CELL_TYPES`，让报告有一处可读的名单）."""
    return CELL_TYPES


__all__ = [
    "LSTMCell",
    "LSTMSpec",
    "RNNCell",
    "RNNSpec",
    "cell_types",
    "initialize_lstm",
    "initialize_rnn",
    "lstm_cell_forward",
    "lstm_forward",
    "lstm_gates",
    "lstm_pre_activation",
    "rnn_cell_forward",
    "rnn_forward",
    "rnn_pre_activation",
    "sequence_length",
]
