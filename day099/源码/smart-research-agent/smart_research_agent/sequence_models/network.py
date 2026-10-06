"""``network``：一个**序列分类器**的前向、反向与参数压平（day094 / M8-D5）.

网络结构（**写死一条链**，为了可读与可对账）：

```text
输入 x₁…x_T  →  循环单元逐步吃进去  →  取**最后一步**的 h_T  →  dense(H→C)  →  logits
                                                                              ↓
                                                              softmax + 交叉熵（day089）
```

## 三个"接缝"都在这里被钉住

```text
① 整段与单步    循环体被 for 循环调用 T 次；每一步的缓存（z_t 或四个门）都留下来
② 循环与全连接   只取最后一步的 h_T 送进 dense ——分类头读的是"看完整条序列之后的状态"
③ 反向           dense 的 dh_T ⇒ 从末步往前走的 BPTT（grad 只挂在最后一步）
```

## 为什么取"最后一步"

对"序列 → 一个标签"的任务，``h_T`` 是"看完整条序列之后的摘要"。
换成"每一步都分类"（序列标注）只需要把每一步的 ``h_t`` 都送进一个 dense——
本课的路由复用同一套 BPTT，因此那是一个**接法**问题，不是数学问题。

## 参数压平：沿用 day090 的契约

``flatten_params`` / ``unflatten_params`` 的顺序是**写死**的：

```text
cell.wx → cell.wh → cell.bias → head_weight → head_bias
```

还原时用**参数本体当形状表**（template）：长度对不上当场抛 ``ShapeError``，
而不是"按顺序切"——后者会给出若干"形状正确、数值错位"的参数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.backprop.gradients import cross_entropy_grad
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.layers import initialize
from smart_research_agent.neural_basics.losses import cross_entropy
from smart_research_agent.sequence_models.errors import (
    BackwardError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.sequence_models.gradients import (
    LSTMGradients,
    LSTMStepCache,
    RNNGradients,
    RNNStepCache,
    lstm_bptt,
    lstm_forward_with_cache,
    rnn_bptt,
    rnn_forward_with_cache,
)
from smart_research_agent.sequence_models.layers import (
    LSTMCell,
    LSTMSpec,
    RNNCell,
    RNNSpec,
    initialize_lstm,
    initialize_rnn,
)
from smart_research_agent.sequence_models.ops import (
    as_matrix,
    as_sequence,
    as_vector,
    matvec,
    matvec_transpose,
    outer,
    sequence_norms,
    vec_add,
)
from smart_research_agent.sequence_models.types import CELL_LSTM, CELL_RNN, CELL_TYPES

#: 分类头的缺省初始化（复用 day089 的 ``initialize``；xavier 对"循环输出 → 类别"这一层稳妥）.
HEAD_INIT = "xavier"


@dataclass(frozen=True)
class SequenceNetParams:
    """一个序列分类器的全部参数：一个循环单元 + 一个全连接头."""

    cell: RNNCell | LSTMCell
    head_weight: Matrix
    head_bias: Vector

    def __post_init__(self) -> None:
        if not isinstance(self.cell, (RNNCell, LSTMCell)):
            raise ParameterError(
                f"cell 必须是 RNNCell 或 LSTMCell，收到 {type(self.cell).__name__}。"
            )
        checked = as_matrix(self.head_weight, name="head_weight")
        if len(checked[0]) != self.cell.hidden_size:
            raise ShapeError(
                f"head_weight 的列数 {len(checked[0])} 与隐藏宽 {self.cell.hidden_size} 不一致："
                "分类头读的是最后一步的 h_T。"
            )
        if len(self.head_bias) != len(checked):
            raise ShapeError(
                f"head_bias 长度 {len(self.head_bias)} 与 head_weight 的行数 {len(checked)} 不一致。"
            )
        object.__setattr__(self, "head_weight", checked)
        object.__setattr__(self, "head_bias", tuple(float(value) for value in self.head_bias))

    @property
    def cell_type(self) -> str:
        """单元名（由参数本体的类型决定，**不单独存一份**，避免两处说不一致）."""
        return CELL_LSTM if isinstance(self.cell, LSTMCell) else CELL_RNN

    @property
    def classes(self) -> int:
        """类别数（分类头的输出宽度）."""
        return len(self.head_weight)

    @property
    def hidden_size(self) -> int:
        """隐藏宽 H."""
        return self.cell.hidden_size

    @property
    def input_size(self) -> int:
        """输入维 D."""
        return self.cell.input_size

    @property
    def parameter_count(self) -> int:
        """全部参数量 = 循环单元 + 分类头."""
        return self.cell.parameter_count + self.classes * self.hidden_size + self.classes

    def line(self) -> str:
        """一行说明：``lstm(H=4, D=1) → dense(4→2) | 参数 ...``."""
        return (
            f"{self.cell_type}(H={self.hidden_size}, D={self.input_size}) → "
            f"dense({self.hidden_size}→{self.classes}) | 参数 {self.parameter_count}"
        )


@dataclass(frozen=True)
class SequenceCache:
    """一次前向留下的中间量（两种单元的缓存按 :attr:`cell_type` 取用）."""

    cell_type: str
    length: int
    last_hidden: Vector
    rnn_steps: tuple[RNNStepCache, ...] | None = None
    lstm_steps: tuple[LSTMStepCache, ...] | None = None

    def __post_init__(self) -> None:
        if self.cell_type not in CELL_TYPES:
            raise ParameterError(f"未知的循环单元 {self.cell_type!r}：可用取值 {list(CELL_TYPES)}。")
        if self.cell_type == CELL_RNN and self.rnn_steps is None:
            raise BackwardError("rnn 的缓存不能为空：反向需要每一步的 (x_t, h_{t−1}, z_t)。")
        if self.cell_type == CELL_LSTM and self.lstm_steps is None:
            raise BackwardError("lstm 的缓存不能为空：反向需要每一步的四个门与 (c_{t−1}, c_t)。")
        if len(self.rnn_steps or self.lstm_steps or ()) != self.length:
            raise BackwardError("缓存的步数与声明的长度不一致。")
        object.__setattr__(self, "last_hidden", as_vector(self.last_hidden, name="last_hidden"))


@dataclass(frozen=True)
class SequenceGradients:
    """一次反向的三块梯度（循环单元 + 分类头的两块）."""

    cell: RNNGradients | LSTMGradients
    head_weight: Matrix
    head_bias: Vector
    length: int

    def total_norm(self) -> float:
        """全部梯度的整体范数（"这一步被推动了多大"）."""
        head = math.sqrt(
            math.fsum(value * value for row in self.head_weight for value in row)
            + math.fsum(value * value for value in self.head_bias)
        )
        return math.sqrt(self.cell.norm() ** 2 + head**2)


def build_sequence_net(
    cell_type: str,
    *,
    input_size: int,
    hidden_size: int,
    classes: int = 2,
    seed: int = 100,
) -> SequenceNetParams:
    """按单元名造一个序列分类器（循环权重与分类头都由 LCG 生成 ⇒ **逐位可复现**）."""
    if cell_type not in CELL_TYPES:
        raise ParameterError(f"未知的循环单元 {cell_type!r}：可用取值 {list(CELL_TYPES)}。")
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ParameterError(f"classes 必须是 >= 2 的整数，收到 {classes!r}。")
    if cell_type == CELL_RNN:
        cell: RNNCell | LSTMCell = initialize_rnn(RNNSpec(input_size, hidden_size, seed=seed))
    else:
        cell = initialize_lstm(LSTMSpec(input_size, hidden_size, seed=seed))
    head_weight, head_bias = initialize(
        HEAD_INIT, classes, cell.hidden_size, seed=seed + 7
    )
    return SequenceNetParams(cell=cell, head_weight=head_weight, head_bias=head_bias)


def sequence_forward_cached(
    params: SequenceNetParams, inputs: tuple[Vector, ...]
) -> tuple[Vector, SequenceCache]:
    """前向并返回 ``(logits, cache)``——cache 是反向需要的全部中间量."""
    checked = as_sequence(inputs, name="inputs")
    if len(checked[0]) != params.input_size:
        raise ShapeError(
            f"输入宽度 {len(checked[0])} 与该网络的输入维 {params.input_size} 不一致。"
        )
    if params.cell_type == CELL_RNN:
        rnn_cell = params.cell
        assert isinstance(rnn_cell, RNNCell)
        states, rnn_steps = rnn_forward_with_cache(rnn_cell, checked)
        last_hidden = states[-1]
        cache = SequenceCache(
            cell_type=CELL_RNN, length=len(checked), last_hidden=last_hidden, rnn_steps=rnn_steps
        )
    else:
        lstm_cell = params.cell
        assert isinstance(lstm_cell, LSTMCell)
        hidden_states, _cell_states, lstm_steps = lstm_forward_with_cache(lstm_cell, checked)
        last_hidden = hidden_states[-1]
        cache = SequenceCache(
            cell_type=CELL_LSTM,
            length=len(checked),
            last_hidden=last_hidden,
            lstm_steps=lstm_steps,
        )
    logits = vec_add(
        matvec(params.head_weight, last_hidden),
        as_vector(params.head_bias, name="head_bias"),
    )
    return logits, cache


def sequence_forward(params: SequenceNetParams, inputs: tuple[Vector, ...]) -> Vector:
    """只要 logits 的前向（:func:`sequence_forward_cached` 的薄包装）."""
    logits, _cache = sequence_forward_cached(params, inputs)
    return logits


def sequence_backward(
    params: SequenceNetParams,
    inputs: tuple[Vector, ...],
    grad_logits: Vector,
    *,
    cache: SequenceCache | None = None,
) -> SequenceGradients:
    """完整反向：``logits → 分类头 → BPTT``（每一步都复用已有的反向实现）.

    ``cache`` 缺省时**重跑一次前向**取得中间量（保真优先于省一次前向）。
    """
    checked_grad = as_vector(grad_logits, name="grad_logits")
    if len(checked_grad) != params.classes:
        raise BackwardError(
            f"grad_logits 长度 {len(checked_grad)} 与类别数 {params.classes} 不一致。"
        )
    checked_inputs = as_sequence(inputs, name="inputs")
    if cache is None:
        _logits, cache = sequence_forward_cached(params, checked_inputs)
    if cache.length != len(checked_inputs):
        raise BackwardError(
            f"缓存有 {cache.length} 步而输入有 {len(checked_inputs)} 步："
            "反向的梯度必须与**同一次**前向的缓存配对。"
        )
    # ① 分类头：dW = g ⊗ h_T；dh_T = Wᵀ·g；db = g
    head_weight_grad = outer(checked_grad, cache.last_hidden)
    d_last = matvec_transpose(params.head_weight, checked_grad)
    head_bias_grad = checked_grad
    # ② 循环部分：梯度**只挂在最后一步**（分类头只读 h_T）
    length = cache.length
    if params.cell_type == CELL_RNN:
        rnn_cell = params.cell
        assert isinstance(rnn_cell, RNNCell)
        if cache.rnn_steps is None:  # pragma: no cover - 构造期已挡住
            raise BackwardError("rnn 缓存缺失。")
        grad_states = tuple(
            d_last if index == length - 1 else (0.0,) * rnn_cell.hidden_size
            for index in range(length)
        )
        result = rnn_bptt(rnn_cell, cache.rnn_steps, grad_states)
        cell_grads: RNNGradients | LSTMGradients = result.cell
    else:
        lstm_cell = params.cell
        assert isinstance(lstm_cell, LSTMCell)
        if cache.lstm_steps is None:  # pragma: no cover - 构造期已挡住
            raise BackwardError("lstm 缓存缺失。")
        grad_hidden = tuple(
            d_last if index == length - 1 else (0.0,) * lstm_cell.hidden_size
            for index in range(length)
        )
        lstm_result = lstm_bptt(lstm_cell, cache.lstm_steps, grad_hidden)
        cell_grads = lstm_result.cell
    return SequenceGradients(
        cell=cell_grads,
        head_weight=head_weight_grad,
        head_bias=head_bias_grad,
        length=length,
    )


def predict(params: SequenceNetParams, inputs: tuple[Vector, ...]) -> int:
    """取 logits 的最大值下标（**预测类别**）."""
    logits = sequence_forward(params, inputs)
    return max(range(len(logits)), key=lambda index: logits[index])


def loss_and_grad(
    params: SequenceNetParams, inputs: tuple[Vector, ...], label: int
) -> tuple[float, SequenceGradients]:
    """一次"前向 → 交叉熵 → 反向"，返回 ``(损失, 梯度)``.

    损失复用 day089 的 :func:`neural_basics.losses.cross_entropy`（`-log p[标签]`），
    它对 logits 的梯度复用 day090 的
    :func:`backprop.gradients.cross_entropy_grad`（``p − onehot``）——
    **两条都只有一份实现**。
    """
    logits, cache = sequence_forward_cached(params, inputs)
    loss = cross_entropy(logits, label)
    grad_logits = cross_entropy_grad(logits, label)
    grads = sequence_backward(params, inputs, grad_logits, cache=cache)
    return loss, grads


def step_norms(params: SequenceNetParams, inputs: tuple[Vector, ...]) -> Vector:
    """每一步隐状态的 L2 范数（"信息在这条链上还剩多少"的读数）."""
    checked = as_sequence(inputs, name="inputs")
    if params.cell_type == CELL_RNN:
        rnn_cell = params.cell
        assert isinstance(rnn_cell, RNNCell)
        states, _steps = rnn_forward_with_cache(rnn_cell, checked)
    else:
        lstm_cell = params.cell
        assert isinstance(lstm_cell, LSTMCell)
        states, _cells, _steps = lstm_forward_with_cache(lstm_cell, checked)
    return sequence_norms(states)


def flatten_params(params: SequenceNetParams) -> Vector:
    """按写死顺序压平全部参数：``cell.wx → cell.wh → cell.bias → head → head_bias``."""
    flat: list[float] = []
    for row in params.cell.wx:
        flat.extend(row)
    for row in params.cell.wh:
        flat.extend(row)
    flat.extend(params.cell.bias)
    for row in params.head_weight:
        flat.extend(row)
    flat.extend(params.head_bias)
    return tuple(flat)


def flatten_gradients(grads: SequenceGradients) -> Vector:
    """按与 :func:`flatten_params` **相同**的顺序压平梯度（两者必须能逐位相加）."""
    flat: list[float] = []
    for row in grads.cell.wx:
        flat.extend(row)
    for row in grads.cell.wh:
        flat.extend(row)
    flat.extend(grads.cell.bias)
    for row in grads.head_weight:
        flat.extend(row)
    flat.extend(grads.head_bias)
    return tuple(flat)


def unflatten_params(flat: Vector, template: SequenceNetParams) -> SequenceNetParams:
    """按模板（**参数本体当形状表**）把一串数还原成 :class:`SequenceNetParams`."""
    values = tuple(float(value) for value in flat)
    if any(not math.isfinite(value) for value in values):
        raise ShapeError("压平的参数里出现非有限数（nan / inf）：先查那一步的学习率。")
    required = template.parameter_count
    if len(values) != required:
        raise ShapeError(
            f"压平的向量有 {len(values)} 个数，而模板需要 {required} 个："
            "长度对不上时'按顺序切'会静默地填错位置，给出形状正确、数值错位的参数。"
        )
    cursor = 0

    def take_row(width: int) -> tuple[float, ...]:
        nonlocal cursor
        row = values[cursor : cursor + width]
        cursor += width
        return tuple(row)

    wx = tuple(take_row(len(row)) for row in template.cell.wx)
    wh = tuple(take_row(len(row)) for row in template.cell.wh)
    bias = tuple(values[cursor : cursor + len(template.cell.bias)])
    cursor += len(template.cell.bias)
    head_weight = tuple(take_row(len(row)) for row in template.head_weight)
    head_bias = tuple(values[cursor : cursor + template.classes])
    cursor += template.classes
    if template.cell_type == CELL_RNN:
        cell: RNNCell | LSTMCell = RNNCell(wx=wx, wh=wh, bias=bias)
    else:
        cell = LSTMCell(wx=wx, wh=wh, bias=bias)
    return SequenceNetParams(cell=cell, head_weight=head_weight, head_bias=head_bias)


__all__ = [
    "HEAD_INIT",
    "SequenceCache",
    "SequenceGradients",
    "SequenceNetParams",
    "build_sequence_net",
    "flatten_gradients",
    "flatten_params",
    "loss_and_grad",
    "predict",
    "sequence_backward",
    "sequence_forward",
    "sequence_forward_cached",
    "step_norms",
    "unflatten_params",
]
