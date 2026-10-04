"""``gradients``：循环网络的**反向**，也就是 BPTT（day094 / M8-D5）.

```text
rnn_forward_with_cache   前向并把每一步的 (x_t, h_{t−1}, z_t) 留下
rnn_bptt                 一层链的反向：dz → dW_x / dW_h / db（**累加**）/ dx / dh_{t−1}
lstm_forward_with_cache  前向并把每一步的四个门与 (c_{t−1}, c_t) 留下
lstm_bptt                LSTM 的反向：四块门的导数 → 同一个线性部分 → 同一条累加纪律
```

## 一、一条纪律，与 day090 / day093 逐字相同

```text
dW ← Σ_{t=1..T} dz_t ⊗ x_t        （**累加**，不是赋值）
```

写成 `=` 的后果不是报错，而是"只留下最后一步的贡献"——梯度整体偏小，
看起来像"学习率设小了"。这就是 :data:`types.BPTT_FORMULA` 要写下来的原因。

## 二、BPTT 到底把什么乘了 T 次

```text
∂h_T/∂h_0 = ∏_{t=1..T} (∂h_t/∂h_{t−1}) = ∏ (diag(act'(z_t)) · W_h)
```

对 tanh 而言每个因子的范数 ≤ ‖W_h‖（因为 ``|tanh'| ≤ 1``），
于是 **T 步连乘 ⇒ 指数衰减**（第 ⑦ 条性质把它量出来）。
LSTM 多了一条加法通道，梯度沿 c 走时只乘遗忘门 ``f``，
因此 ``∂c_T/∂c_0 = ∏ f_t``——f 接近 1 时这条通道几乎不衰减
（同一条性质的另一半）。

## 三、激活的反向导数不重写

``tanh`` / ``relu`` / ``sigmoid`` 的导数**只有一份**，写在 day090 的
``backprop.gradients``。本模块直接调用
:func:`backprop.gradients.elementwise_backward`，连 LSTM 的三个门也不例外——
``sigmoid`` 与 ``tanh`` 都是那张表里的逐元素激活。

## 四、诊断事件：梯度爆炸

:func:`gradient_norm` 与 :func:`step_gradient_norms` 只**读数**；
把"范数越界"变成控制流事件的是 ``train.check_gradient_norm``
（它抛 day075 的 ``GradientError``——本课让这一族**回来**的地方）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.backprop import gradients as backprop_gradients
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import activate
from smart_research_agent.sequence_models.errors import (
    BackwardError,
    ParameterError,
    ShapeError,
    TimeStepError,
)
from smart_research_agent.sequence_models.layers import (
    LSTMCell,
    RNNCell,
    lstm_pre_activation,
    rnn_pre_activation,
)
from smart_research_agent.sequence_models.ops import (
    as_finite_vector,
    as_sequence,
    as_vector,
    gate_blocks,
    matvec_transpose,
    outer,
    vec_add,
    vec_hadamard,
    zeros,
)
from smart_research_agent.sequence_models.types import (
    GATE_ACTIVATIONS,
    GATES,
    RNN_ACTIVATIONS,
)


def _mat_add(left: Matrix, right: Matrix) -> Matrix:
    """两个同形矩阵逐元素相加（``dW += 这一步的一片`` 走它）."""
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ShapeError("矩阵相加要求同形。")
    return tuple(
        tuple(a + b for a, b in zip(row_a, row_b, strict=True))
        for row_a, row_b in zip(left, right, strict=True)
    )


def _zeros_matrix(rows: int, columns: int) -> Matrix:
    """全零矩阵（梯度的初值：**从 0 开始累加**）."""
    return tuple((0.0,) * columns for _ in range(rows))


# --------------------------------------------------------------------------- #
# 1. 朴素循环单元
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RNNStepCache:
    """一步前向留下的三样中间量（反向需要的**全部**东西）."""

    inputs: Vector
    state_prev: Vector
    pre_activation: Vector


@dataclass(frozen=True)
class RNNGradients:
    """累积到整段之后的三个梯度（``wx`` / ``wh`` / ``bias``）."""

    wx: Matrix
    wh: Matrix
    bias: Vector

    def norm(self) -> float:
        """全部梯度的整体 L2 范数（"这一步被推动了多大"）."""
        return math.sqrt(
            math.fsum(value * value for row in self.wx for value in row)
            + math.fsum(value * value for row in self.wh for value in row)
            + math.fsum(value * value for value in self.bias)
        )


@dataclass(frozen=True)
class RNNBPTTResult:
    """一次 BPTT 的全部产物：逐步的 ``dx``、回到起点的 ``dh₀`` 与累积的权重梯度."""

    dx: tuple[Vector, ...]
    d_state0: Vector
    cell: RNNGradients
    notes: tuple[str, ...] = field(default=())

    def norm(self) -> float:
        """权重梯度的整体范数."""
        return self.cell.norm()

    def carry(self) -> float:
        """``‖∂h_T/∂h_0‖`` 的直接读数（本课用来量"梯度衰减了多少"）."""
        return math.sqrt(math.fsum(value * value for value in self.d_state0))


def rnn_forward_with_cache(
    cell: RNNCell,
    inputs: tuple[Vector, ...],
    *,
    state0: Vector | None = None,
    activation: str = "tanh",
) -> tuple[tuple[Vector, ...], tuple[RNNStepCache, ...]]:
    """前向并返回 ``(T 个状态, T 份缓存)``.

    它在数值上与 ``layers.rnn_forward`` 完全一致（只是多留了每一步的中间量），
    因此"前向只写一遍"这件事仍然成立：``z_t`` 由同一个
    :func:`layers.rnn_pre_activation` 产出。
    """
    if activation not in RNN_ACTIVATIONS:
        raise ParameterError(f"未知的循环激活 {activation!r}：可用取值 {list(RNN_ACTIVATIONS)}。")
    checked_inputs = as_sequence(inputs, name="inputs")
    initial = cell.initial_state() if state0 is None else as_vector(state0, name="state0")
    if len(initial) != cell.hidden_size:
        raise ShapeError(f"初值宽度 {len(initial)} 与该单元的隐藏宽 {cell.hidden_size} 不一致。")
    states: list[Vector] = []
    caches: list[RNNStepCache] = []
    current = initial
    for step in checked_inputs:
        pre = rnn_pre_activation(cell, step, current)
        caches.append(RNNStepCache(inputs=step, state_prev=current, pre_activation=pre))
        current = activate(activation, pre)
        states.append(current)
    return tuple(states), tuple(caches)


def rnn_bptt(
    cell: RNNCell,
    caches: tuple[RNNStepCache, ...],
    grad_states: tuple[Vector, ...],
    *,
    activation: str = "tanh",
) -> RNNBPTTResult:
    """``RNNCell`` 的 BPTT：给定每一步回传的 ``∂L/∂h_t``，反着走回 ``h₀``.

    ```text
    对 t = T … 1（倒序）：
        d_h  = grad_states[t] + 从 t+1 带上来的那部分
        dz_t = d_h ⊙ act'(z_t)                 ← 激活的导数是 day090 的
        dW_x += dz_t ⊗ x_t                     ← **累加**
        dW_h += dz_t ⊗ h_{t−1}
        db   += dz_t
        dx_t  = W_xᵀ·dz_t
        h_{t−1} ← W_hᵀ·dz_t                     （继续往前带）
    ```

    ``grad_states`` 的长度必须等于缓存步数（也就是序列长度 T）。
    对不上抛 :class:`TimeStepError`——**这一条是"形状都对、只是错位一格"的唯一闸门**。
    """
    if activation not in RNN_ACTIVATIONS:
        raise ParameterError(f"未知的循环激活 {activation!r}：可用取值 {list(RNN_ACTIVATIONS)}。")
    if len(grad_states) != len(caches):
        raise TimeStepError(
            f"回传梯度有 {len(grad_states)} 步而缓存有 {len(caches)} 步："
            "两者必须是同一个 T——'差一个'多半是把初值 h₀ 也算进了输出。"
        )
    if not caches:
        raise TimeStepError("缓存为空：零步的序列没有 BPTT 可言。")
    hidden = cell.hidden_size
    checked_grads = tuple(as_vector(grad, name=f"grad_states[{t}]") for t, grad in enumerate(grad_states))
    for index, grad in enumerate(checked_grads):
        if len(grad) != hidden:
            raise BackwardError(
                f"第 {index} 步的回传梯度宽度 {len(grad)} 与隐藏宽 {hidden} 不一致："
                "反向的梯度必须与**同一次**前向的状态配对。"
            )
    d_wx = _zeros_matrix(hidden, cell.input_size)
    d_wh = _zeros_matrix(hidden, hidden)
    d_bias = zeros(hidden)
    dx: list[Vector] = [zeros(cell.input_size) for _ in caches]
    carry = zeros(hidden)
    for index in range(len(caches) - 1, -1, -1):
        cache = caches[index]
        combined = vec_add(checked_grads[index], carry)
        dz = backprop_gradients.elementwise_backward(
            activation, (cache.pre_activation,), (combined,)
        )[0]
        d_wx = _mat_add(d_wx, outer(dz, cache.inputs))
        d_wh = _mat_add(d_wh, outer(dz, cache.state_prev))
        d_bias = vec_add(d_bias, dz)
        dx[index] = matvec_transpose(cell.wx, dz)
        carry = matvec_transpose(cell.wh, dz)
    return RNNBPTTResult(
        dx=tuple(dx),
        d_state0=carry,
        cell=RNNGradients(wx=d_wx, wh=d_wh, bias=d_bias),
        notes=(
            "dW_x / dW_h / db 在整段上**累加**（+= 而不是 =）",
            "激活的局部导数来自 day090 的 elementwise_backward",
            "最后的 carry 就是 ∂L/∂h₀（第 ⑦ 条性质的读数）",
        ),
    )


# --------------------------------------------------------------------------- #
# 2. 长短期记忆单元
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LSTMStepCache:
    """LSTM 一步前向留下的全部中间量（四个门 + 两条状态）."""

    inputs: Vector
    hidden_prev: Vector
    cell_prev: Vector
    pre_activation: Vector
    input_gate: Vector
    forget_gate: Vector
    output_gate: Vector
    candidate: Vector
    cell: Vector

    def tanh_cell(self) -> Vector:
        """``tanh(c_t)``（``h_t = o_t ⊙ tanh(c_t)`` 里那一项，反向要用两次）."""
        return activate("tanh", self.cell)


@dataclass(frozen=True)
class LSTMGradients:
    """累积到整段之后的三个梯度（``wx`` / ``wh`` / ``bias``）."""

    wx: Matrix
    wh: Matrix
    bias: Vector

    def norm(self) -> float:
        """全部梯度的整体 L2 范数."""
        return math.sqrt(
            math.fsum(value * value for row in self.wx for value in row)
            + math.fsum(value * value for row in self.wh for value in row)
            + math.fsum(value * value for value in self.bias)
        )


@dataclass(frozen=True)
class LSTMBPTTResult:
    """一次 LSTM 的 BPTT 的产物：``dx``、回到起点的 ``dh₀`` 与 ``dc₀``、权重梯度."""

    dx: tuple[Vector, ...]
    d_hidden0: Vector
    d_cell0: Vector
    cell: LSTMGradients
    notes: tuple[str, ...] = field(default=())

    def norm(self) -> float:
        """权重梯度的整体范数."""
        return self.cell.norm()

    def cell_carry(self) -> float:
        """``‖∂L/∂c_0‖`` 的直接读数（"记忆通道带回多少"）."""
        return math.sqrt(math.fsum(value * value for value in self.d_cell0))


def lstm_forward_with_cache(
    cell: LSTMCell,
    inputs: tuple[Vector, ...],
    *,
    state0: tuple[Vector, Vector] | None = None,
) -> tuple[tuple[Vector, ...], tuple[Vector, ...], tuple[LSTMStepCache, ...]]:
    """前向并返回 ``(T 个 h, T 个 c, T 份缓存)``."""
    checked_inputs = as_sequence(inputs, name="inputs")
    initial_h, initial_c = cell.initial_state() if state0 is None else state0
    if len(initial_h) != cell.hidden_size or len(initial_c) != cell.hidden_size:
        raise ShapeError("初值 (h₀, c₀) 的宽度必须都等于隐藏宽 H。")
    current_h = as_vector(initial_h, name="h0")
    current_c = as_vector(initial_c, name="c0")
    hidden_states: list[Vector] = []
    cell_states: list[Vector] = []
    caches: list[LSTMStepCache] = []
    for step in checked_inputs:
        previous_h, previous_c = current_h, current_c
        pre = lstm_pre_activation(cell, step, (previous_h, previous_c))
        raw_i, raw_f, raw_o, raw_g = gate_blocks(pre, cell.hidden_size)
        input_gate = activate(GATE_ACTIVATIONS[GATES[0]], raw_i)
        forget_gate = activate(GATE_ACTIVATIONS[GATES[1]], raw_f)
        output_gate = activate(GATE_ACTIVATIONS[GATES[2]], raw_o)
        candidate = activate(GATE_ACTIVATIONS[GATES[3]], raw_g)
        new_cell = vec_add(
            vec_hadamard(forget_gate, previous_c), vec_hadamard(input_gate, candidate)
        )
        new_hidden = vec_hadamard(output_gate, activate("tanh", new_cell))
        caches.append(
            LSTMStepCache(
                inputs=step,
                hidden_prev=previous_h,
                cell_prev=previous_c,
                pre_activation=pre,
                input_gate=input_gate,
                forget_gate=forget_gate,
                output_gate=output_gate,
                candidate=candidate,
                cell=new_cell,
            )
        )
        current_h, current_c = new_hidden, new_cell
        hidden_states.append(new_hidden)
        cell_states.append(new_cell)
    return tuple(hidden_states), tuple(cell_states), tuple(caches)


def _gate_derivative(name: str, pre: Vector, grad: Vector) -> Vector:
    """一个门在**激活之后**的梯度 → 激活**之前**的梯度（复用 day090 的逐元素反向）."""
    return backprop_gradients.elementwise_backward(name, (pre,), (grad,))[0]


def lstm_bptt(
    cell: LSTMCell,
    caches: tuple[LSTMStepCache, ...],
    grad_hidden: tuple[Vector, ...],
    *,
    grad_cell: tuple[Vector, ...] | None = None,
) -> LSTMBPTTResult:
    """``LSTMCell`` 的 BPTT：两条通道（h 与 c）一起往回走.

    ```text
    对 t = T … 1（倒序）：
        dh = grad_hidden[t] + 从 t+1 带上来的 dh
        dc = grad_cell[t]   + 从 t+1 带上来的 dc
        ── h_t = o ⊙ tanh(c_t)  ─────────────────────────────
        d_o       = dh ⊙ tanh(c_t)
        dc       += dh ⊙ o ⊙ (1 − tanh(c_t)²)
        ── c_t = f ⊙ c_{t−1} + i ⊙ g  ───────────────────────
        d_f       = dc ⊙ c_{t−1}
        dc_{t−1}  = dc ⊙ f                    ← **记忆通道：只乘 f**
        d_i       = dc ⊙ g
        d_g       = dc ⊙ i
        ── 四个门各自过激活的导数，拼回 4H ────────────────────
        dz = [σ'(z_i)⊙d_i | σ'(z_f)⊙d_f | σ'(z_o)⊙d_o | tanh'(z_g)⊙d_g]
        dW_x += dz ⊗ x_t；dW_h += dz ⊗ h_{t−1}；db += dz
    ```

    与 :func:`rnn_bptt` 的两处不同：**多带一条 ``dc``**，以及
    ``h_t`` 通过 ``tanh(c_t)`` **间接**依赖 ``c_t``（那一项 ``dh ⊙ o ⊙ (1 − tanh²)``
    是最容易漏的一条链）。
    """
    if grad_cell is not None and len(grad_cell) != len(caches):
        raise TimeStepError(
            f"c 的回传梯度有 {len(grad_cell)} 步而缓存有 {len(caches)} 步：两者必须是同一个 T。"
        )
    if len(grad_hidden) != len(caches):
        raise TimeStepError(
            f"h 的回传梯度有 {len(grad_hidden)} 步而缓存有 {len(caches)} 步："
            "两者必须是同一个 T——'差一个'多半是把初值 h₀ 也算进了输出。"
        )
    if not caches:
        raise TimeStepError("缓存为空：零步的序列没有 BPTT 可言。")
    hidden = cell.hidden_size
    checked_h = tuple(as_vector(grad, name=f"grad_hidden[{t}]") for t, grad in enumerate(grad_hidden))
    checked_c = (
        tuple(zeros(hidden) for _ in caches)
        if grad_cell is None
        else tuple(as_vector(grad, name=f"grad_cell[{t}]") for t, grad in enumerate(grad_cell))
    )
    for index, (grad_h, grad_c) in enumerate(zip(checked_h, checked_c, strict=True)):
        if len(grad_h) != hidden or len(grad_c) != hidden:
            raise BackwardError(
                f"第 {index} 步的回传梯度宽度与隐藏宽 {hidden} 不一致："
                "反向的梯度必须与**同一次**前向的状态配对。"
            )
    d_wx = _zeros_matrix(4 * hidden, cell.input_size)
    d_wh = _zeros_matrix(4 * hidden, hidden)
    d_bias = zeros(4 * hidden)
    dx: list[Vector] = [zeros(cell.input_size) for _ in caches]
    carry_h = zeros(hidden)
    carry_c = zeros(hidden)
    for index in range(len(caches) - 1, -1, -1):
        cache = caches[index]
        tanh_cell = cache.tanh_cell()
        d_h = vec_add(checked_h[index], carry_h)
        d_c = vec_add(checked_c[index], carry_c)
        # h_t = o ⊙ tanh(c_t)
        d_output = vec_hadamard(d_h, tanh_cell)
        to_cell = vec_hadamard(
            d_h,
            vec_hadamard(
                cache.output_gate,
                tuple(1.0 - value * value for value in tanh_cell),
            ),
        )
        d_c = vec_add(d_c, to_cell)
        # c_t = f ⊙ c_{t−1} + i ⊙ g
        d_forget = vec_hadamard(d_c, cache.cell_prev)
        d_cell_prev = vec_hadamard(d_c, cache.forget_gate)
        d_input = vec_hadamard(d_c, cache.candidate)
        d_candidate = vec_hadamard(d_c, cache.input_gate)
        # 四个门各自过激活的导数（顺序必须与权重行块一致）
        raw_i, raw_f, raw_o, raw_g = gate_blocks(cache.pre_activation, hidden)
        d_raw = add_gate_blocks(
            _gate_derivative(GATE_ACTIVATIONS[GATES[0]], raw_i, d_input),
            _gate_derivative(GATE_ACTIVATIONS[GATES[1]], raw_f, d_forget),
            _gate_derivative(GATE_ACTIVATIONS[GATES[2]], raw_o, d_output),
            _gate_derivative(GATE_ACTIVATIONS[GATES[3]], raw_g, d_candidate),
        )
        d_wx = _mat_add(d_wx, outer(d_raw, cache.inputs))
        d_wh = _mat_add(d_wh, outer(d_raw, cache.hidden_prev))
        d_bias = vec_add(d_bias, d_raw)
        dx[index] = matvec_transpose(cell.wx, d_raw)
        carry_h = matvec_transpose(cell.wh, d_raw)
        carry_c = d_cell_prev
    return LSTMBPTTResult(
        dx=tuple(dx),
        d_hidden0=carry_h,
        d_cell0=carry_c,
        cell=LSTMGradients(wx=d_wx, wh=d_wh, bias=d_bias),
        notes=(
            "dW_x / dW_h / db 在整段上**累加**（+= 而不是 =）",
            "记忆通道的梯度只乘遗忘门：dC_{t−1} = dC_t ⊙ f_t",
            "h_t 通过 tanh(c_t) 间接依赖 c_t，那一条链（dh ⊙ o ⊙ (1 − tanh²)）最容易漏",
            "四个门的激活导数来自 day090 的 elementwise_backward",
        ),
    )


def add_gate_blocks(
    input_grad: Vector, forget_grad: Vector, output_grad: Vector, candidate_grad: Vector
) -> Vector:
    """把四个门的梯度**按行块顺序**拼回一个 ``4H`` 向量（拼错顺序不会报错）."""
    hidden = len(as_vector(input_grad, name="input_grad"))
    for name, block in (
        ("forget_grad", forget_grad),
        ("output_grad", output_grad),
        ("candidate_grad", candidate_grad),
    ):
        if len(block) != hidden:
            raise ShapeError(f"{name} 的长度 {len(block)} 与 input_grad 的 {hidden} 不一致。")
    return tuple(input_grad) + tuple(forget_grad) + tuple(output_grad) + tuple(candidate_grad)


def gradient_norm(values: tuple[float, ...]) -> float:
    """一串梯度的整体 L2 范数（爆炸判据的读数；元素非有限抛 :class:`BackwardError`）."""
    flat = as_finite_vector(values, name="gradient")
    return math.sqrt(math.fsum(value * value for value in flat))


def rnn_carry_norm(
    cell: RNNCell,
    inputs: tuple[Vector, ...],
    *,
    activation: str = "tanh",
    state0: Vector | None = None,
) -> float:
    """``‖∂h_T/∂h_0‖`` 的读数：**只给末步一个全 1 的回传**，看它回到起点还剩多少.

    它是第 ⑦ 条性质的左半边：tanh-RNN 的这个数随 T **指数衰减**。
    """
    _states, caches = rnn_forward_with_cache(cell, inputs, state0=state0, activation=activation)
    length = len(caches)
    seed = (1.0,) * cell.hidden_size
    grad_states = tuple(
        seed if index == length - 1 else zeros(cell.hidden_size) for index in range(length)
    )
    return rnn_bptt(cell, caches, grad_states, activation=activation).carry()


def lstm_cell_carry_norm(
    cell: LSTMCell,
    inputs: tuple[Vector, ...],
    *,
    state0: tuple[Vector, Vector] | None = None,
) -> float:
    """``‖∂L/∂c_0‖`` 的读数（梯度只从末步的 ``c_T`` 出发）.

    它是第 ⑦ 条性质的右半边：``∂c_T/∂c_0 = ∏f_t``，f 接近 1 时几乎不衰减。
    注意它的入参只有 **c** 的种子（``h`` 的回传全零）——这样量到的正好是
    "记忆通道自己能把信号带回多远"。
    """
    _hidden, _cell_states, caches = lstm_forward_with_cache(cell, inputs, state0=state0)
    length = len(caches)
    seed = (1.0,) * cell.hidden_size
    grad_hidden = tuple(zeros(cell.hidden_size) for _ in range(length))
    grad_cell = tuple(
        seed if index == length - 1 else zeros(cell.hidden_size) for index in range(length)
    )
    return lstm_bptt(cell, caches, grad_hidden, grad_cell=grad_cell).cell_carry()


__all__ = [
    "LSTMGradients",
    "LSTMBPTTResult",
    "LSTMStepCache",
    "RNNGradients",
    "RNNBPTTResult",
    "RNNStepCache",
    "add_gate_blocks",
    "gradient_norm",
    "lstm_bptt",
    "lstm_cell_carry_norm",
    "lstm_forward_with_cache",
    "rnn_bptt",
    "rnn_carry_norm",
    "rnn_forward_with_cache",
]
