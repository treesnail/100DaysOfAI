"""``network``：把 day094 的链**装上旋钮**之后的前向 / 反向 / 参数压平（day095 / M8-D6）.

```text
一批样本 x₁…x_T  →  day094 的循环单元（**一行不改**）  →  每条的 h_T
                                                        ↓
                                              BatchNorm(γ, β)      ← 本包新写
                                                        ↓
                                              Dropout（训练相）     ← 转发 day081
                                                        ↓
                                              dense(H→C)           ← day094 的分类头
                                                        ↓
                                                      logits
```

## 为什么这一层**必须是批**的（本课真的撞到的第一堵墙）

BatchNorm 的统计量来自**批**。而 day094 的分类器一次只吃一条序列——
如果把 BN 直接插在"单条序列的 h_T"上，批大小就是 1：

```text
批大小 1 ⇒ 每个特征的批内方差恒为 0
        ⇒ x̂ = (x − x)/√(0 + ε) = 0
        ⇒ y = γ⊙0 + β = β = 0
        ⇒ 整层输出是一个常数，与输入完全无关
```

第一版实现就是这样：三条不同的配置（rnn / lstm / 带不带 dropout）给出**同一个 loss**
（`0.542374`），因为 BN 把 h_T 全抹平了。这不是 bug，而是
"BatchNorm 需要批"最直接的证据——所以本课的网络一次前向处理**一批样本**，
而 `normalization.batch_size_one_report` 把这件事变成一个可读的读数。

## 三个接缝都在这里被钉住

```text
① 循环与归一化   每条样本各自的 h_T 竖着堆成一个 (N, H) 矩阵，再交给 BatchNorm
② 归一化与丢弃   BatchNorm 的输出整批交给 dropout（掩码也是 (N, H)）
③ 反向          dense → dropout → BN 得到 (N, H) 的 dh_T，
                再**逐条**把 dh_T 的第 i 行交回 day094 的 BPTT
```

第 ③ 条是本课最值得看的一处：本包**没有**重写 BPTT，也没有重写"批"——
它把每条样本的 `dh_T` 放进一条"只在最后一步有梯度"的向量里，交给 day094，
再把 N 条样本的权重梯度**加起来**（因为损失是批均值）。

## 参数压平：day090 的契约第三次被用上

```text
flatten    day094 的 ``flatten_params``（cell → head）+ γ + β
unflatten  先切出末尾的 2H 个数（γ / β），其余原样交回 day094 的 ``unflatten_params``
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.backprop.gradients import cross_entropy_grad_rows
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.losses import cross_entropy
from smart_research_agent.regularization.errors import ParameterError, PhaseError, ShapeError
from smart_research_agent.regularization.normalization import (
    as_matrix,
    as_vector,
    batch_norm_backward,
    batch_norm_forward,
    BatchNormCache,
    initial_running,
    matrix_shape,
    RunningStatistics,
)
from smart_research_agent.regularization.techniques import (
    dropout_vector_backward_matrix,
)
from smart_research_agent.regularization.types import PHASE_TRAIN, PHASES
from smart_research_agent.sequence_models import gradients as sequence_gradients
from smart_research_agent.sequence_models.layers import LSTMCell, RNNCell
from smart_research_agent.sequence_models.network import (
    SequenceCache,
    SequenceGradients,
    SequenceNetParams,
    build_sequence_net,
    flatten_gradients as flatten_sequence_gradients,
    flatten_params as flatten_sequence_params,
    sequence_forward_cached,
    unflatten_params as unflatten_sequence_params,
)
from smart_research_agent.sequence_models.ops import as_sequence, zeros
from smart_research_agent.sequence_models.types import CELL_RNN
from smart_research_agent.training_optim.dropout import dropout_forward

#: 一批样本：``((序列, 标签), …)``（与 day094 的 ``Sample`` 同形，只是装进元组）。
Batch = tuple[tuple[tuple[Vector, ...], int], ...]


def _checked_phase(phase: object) -> str:
    """相的名字必须是 ``train`` / ``eval`` 之一（**不给它挑一个默认值**）."""
    if phase not in PHASES:
        raise PhaseError(f"不认识的相 {phase!r}：可用取值 {list(PHASES)}。")
    return str(phase)


def as_batch(batch: object, *, name: str = "batch") -> Batch:
    """收敛成一批样本（非空；每条样本都是 ``(序列, 标签)``）."""
    if isinstance(batch, (str, bytes)) or not hasattr(batch, "__iter__"):
        raise ShapeError(f"{name} 必须是一批样本（可迭代），收到 {type(batch).__name__}。")
    items = list(batch)
    if not items:
        raise ShapeError(f"{name} 不能为空：空批的 BatchNorm 统计量没有定义。")
    checked: list[tuple[tuple[Vector, ...], int]] = []
    for index, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ShapeError(f"{name} 的第 {index} 项必须是 (序列, 标签)，收到 {item!r}。")
        inputs, label = item
        if isinstance(label, bool) or not isinstance(label, int):
            raise ShapeError(f"{name} 的第 {index} 项标签必须是整数，收到 {label!r}。")
        checked.append((as_sequence(inputs, name=f"{name}[{index}].inputs"), label))
    return tuple(checked)


@dataclass(frozen=True)
class RegularizedParams:
    """一个"装上旋钮"的分类器：day094 的序列网络 + BatchNorm 的 γ/β."""

    sequence: SequenceNetParams
    gamma: Vector
    beta: Vector

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, SequenceNetParams):
            raise ParameterError(f"sequence 必须是 SequenceNetParams，收到 {type(self.sequence).__name__}。")
        checked_gamma = as_vector(self.gamma, name="gamma")
        checked_beta = as_vector(self.beta, name="beta")
        features = self.sequence.hidden_size
        if len(checked_gamma) != features:
            raise ShapeError(
                f"gamma 长度 {len(checked_gamma)} 与隐藏宽 {features} 不一致："
                "BatchNorm 的 γ/β 是**逐特征**的。"
            )
        if len(checked_beta) != features:
            raise ShapeError(f"beta 长度 {len(checked_beta)} 与隐藏宽 {features} 不一致。")
        object.__setattr__(self, "gamma", checked_gamma)
        object.__setattr__(self, "beta", checked_beta)

    @property
    def features(self) -> int:
        """BatchNorm 处理的特征数（= 隐藏宽 H）."""
        return self.sequence.hidden_size

    @property
    def cell_type(self) -> str:
        """循环单元名."""
        return self.sequence.cell_type

    @property
    def classes(self) -> int:
        """类别数."""
        return self.sequence.classes

    @property
    def parameter_count(self) -> int:
        """全部参数量 = day094 的网络 + γ/β 两块."""
        return self.sequence.parameter_count + 2 * self.features

    def line(self) -> str:
        """一行说明：``lstm(H=4, D=1) → bn(4) → dropout → dense(4→2) | 参数 ...``."""
        return (
            f"{self.sequence.cell_type}(H={self.features}, D={self.sequence.input_size}) → "
            f"bn({self.features}) → dropout → dense({self.features}→{self.classes}) | "
            f"参数 {self.parameter_count}"
        )


@dataclass(frozen=True)
class RegularizedCache:
    """一次前向留下的全部中间量（反向需要**它们**，不需要重跑前向）."""

    phase: str
    running: RunningStatistics
    sequences: tuple[SequenceCache, ...]
    last_hidden: Matrix
    bn: BatchNormCache | None
    dropout_mask: Matrix
    dropout_scale: float
    dropped: Matrix

    @property
    def rows(self) -> int:
        """批大小 N."""
        return len(self.last_hidden)

    @property
    def features(self) -> int:
        """特征数 H."""
        return len(self.last_hidden[0])

    @property
    def normalized(self) -> Matrix:
        """BatchNorm 的输出（``x̂`` 过完 γ/β 之后的那一份；关掉 BN 时就是 ``h_T``）."""
        return self.last_hidden if self.bn is None else self.bn.normalized


@dataclass(frozen=True)
class RegularizedGradients:
    """一次反向的产物：day094 的那一份 + γ/β 两块."""

    sequence: SequenceGradients
    d_gamma: Vector
    d_beta: Vector

    def total_norm(self) -> float:
        """全部梯度的整体范数."""
        return math.sqrt(
            self.sequence.total_norm() ** 2
            + math.fsum(value * value for value in self.d_gamma)
            + math.fsum(value * value for value in self.d_beta)
        )


def build_regularized(
    cell_type: str,
    *,
    input_size: int,
    hidden_size: int,
    classes: int = 2,
    seed: int = 100,
) -> RegularizedParams:
    """按单元名造一个"装上旋钮"的分类器（**循环部分由 day094 造**，γ/β 由常量定）."""
    sequence = build_sequence_net(
        cell_type, input_size=input_size, hidden_size=hidden_size, classes=classes, seed=seed
    )
    # γ 的初值是 1、β 的初值是 0：此时 BN 的输出就是"纯粹的标准化结果"。
    # 刻意**不加随机扰动**——"开局等于标准化"是一条可断言的起点。
    return RegularizedParams(
        sequence=sequence,
        gamma=(1.0,) * sequence.hidden_size,
        beta=(0.0,) * sequence.hidden_size,
    )


def regularized_forward_cached(
    params: RegularizedParams,
    batch: Batch,
    *,
    phase: str = PHASE_TRAIN,
    running: RunningStatistics | None = None,
    dropout_rate: float = 0.0,
    dropout_seed: int = 0,
    epsilon: float | None = None,
    use_norm: bool = True,
    bn_momentum: float = 0.1,
) -> tuple[Matrix, RegularizedCache]:
    """一批样本的前向，返回 ``(logits (N×C), cache)``.

    ``use_norm=False`` 表示**把 BatchNorm 整层跳过**（恒等）——
    它是消融实验里的"没有归一化"那一组，而不是"用一个接近恒等的 BN 冒充"。
    """
    resolved_phase = _checked_phase(phase)
    checked = as_batch(batch)
    # ① 序列部分（day094 的实现；它顺带算了一次 head，本课不用那一份）
    last_hidden: list[Vector] = []
    sequences: list[SequenceCache] = []
    for inputs, _label in checked:
        _unused, cache = sequence_forward_cached(params.sequence, inputs)
        sequences.append(cache)
        last_hidden.append(cache.last_hidden)
    hidden_matrix = tuple(last_hidden)
    # ② BatchNorm（本包新写的唯一一件数学）。
    #    running 原样传下去：训练相为 None 时从 (μ=0, σ²=1) 起步；
    #    **推理相为 None 时 batch_norm_forward 会抛 PhaseError**（那正是它该拦的事）。
    kwargs = {} if epsilon is None else {"epsilon": epsilon}
    bn_cache: BatchNormCache | None
    if use_norm:
        normalized, bn_cache, running_out = batch_norm_forward(
            hidden_matrix,
            gamma=params.gamma,
            beta=params.beta,
            phase=resolved_phase,
            running=running,
            momentum=bn_momentum,
            **kwargs,
        )
    else:
        normalized, bn_cache = hidden_matrix, None
        running_out = initial_running(params.features) if running is None else running
    # ③ Dropout（转发 day081，整批一次）
    dropped, mask, scale = dropout_forward(
        normalized, rate=dropout_rate, seed=dropout_seed, phase=resolved_phase
    )
    # ④ 分类头（逐行）
    logits = tuple(
        tuple(
            math.fsum(weight * value for weight, value in zip(row, dropped[index], strict=True))
            + params.sequence.head_bias[class_index]
            for class_index, row in enumerate(params.sequence.head_weight)
        )
        for index in range(len(dropped))
    )
    cache = RegularizedCache(
        phase=resolved_phase,
        running=running_out,
        sequences=tuple(sequences),
        last_hidden=hidden_matrix,
        bn=bn_cache,
        dropout_mask=mask,
        dropout_scale=scale,
        dropped=dropped,
    )
    return logits, cache


def regularized_forward(
    params: RegularizedParams, batch: Batch, **kwargs
) -> Matrix:
    """只要 logits 的前向（:func:`regularized_forward_cached` 的薄包装）."""
    logits, _cache = regularized_forward_cached(params, batch, **kwargs)
    return logits


def _accumulate_cell_gradients(
    acc: tuple[list[list[float]], list[list[float]], list[float]],
    single: object,
) -> None:
    """把**一条样本**的循环权重梯度加进累加器（因为损失是批均值）."""
    wx_acc, wh_acc, bias_acc = acc
    for row_index, row in enumerate(single.wx):  # type: ignore[attr-defined]
        for column_index, value in enumerate(row):
            wx_acc[row_index][column_index] += value
    for row_index, row in enumerate(single.wh):  # type: ignore[attr-defined]
        for column_index, value in enumerate(row):
            wh_acc[row_index][column_index] += value
    for index, value in enumerate(single.bias):  # type: ignore[attr-defined]
        bias_acc[index] += value


def regularized_backward(
    params: RegularizedParams, cache: RegularizedCache, grad_logits: Matrix
) -> RegularizedGradients:
    """完整反向：``logits → dense → dropout → BN → 逐条 dh_T → day094 的 BPTT``.

    ``grad_logits`` 是**批均值损失**对 logits 的梯度（``cross_entropy_grad_rows``
    已经除过批大小），因此逐条累加之后不需要再除一次。
    """
    if not isinstance(cache, RegularizedCache):
        raise ParameterError(f"cache 必须是 RegularizedCache，收到 {type(cache).__name__}。")
    checked_grad = as_matrix(grad_logits, name="grad_logits")
    rows, columns = matrix_shape(checked_grad)
    if rows != cache.rows:
        raise ShapeError(
            f"grad_logits 有 {rows} 行而缓存有 {cache.rows} 行：反向的梯度必须与**同一次**前向配对。"
        )
    if columns != params.classes:
        raise ShapeError(f"grad_logits 有 {columns} 列而与类别数 {params.classes} 不一致。")
    feature_count = params.features
    # ① 分类头：dW = Σ_i g_i ⊗ dropped_i；d_dropped_i = Wᵀ·g_i；db = Σ_i g_i
    head_weight_grad = tuple(
        tuple(
            math.fsum(checked_grad[row_index][c] * cache.dropped[row_index][f] for row_index in range(rows))
            for f in range(feature_count)
        )
        for c in range(params.classes)
    )
    head_bias_grad = tuple(
        math.fsum(checked_grad[row_index][c] for row_index in range(rows))
        for c in range(params.classes)
    )
    d_dropped = tuple(
        tuple(
            math.fsum(
                params.sequence.head_weight[c][f] * checked_grad[row_index][c]
                for c in range(params.classes)
            )
            for f in range(feature_count)
        )
        for row_index in range(rows)
    )
    # ② Dropout 反向（转发 day081，整批一次）
    d_normalized = dropout_vector_backward_matrix(d_dropped, cache.dropout_mask, cache.dropout_scale)
    # ③ BatchNorm 反向（训练相三项、推理相一项）；关掉 BN 时是恒等
    if cache.bn is None:
        d_inputs = d_normalized
        d_gamma = zeros(feature_count)
        d_beta = zeros(feature_count)
    else:
        bn_grads = batch_norm_backward(cache.bn, d_normalized, beta=params.beta)
        d_inputs = bn_grads.d_inputs
        d_gamma = bn_grads.d_gamma
        d_beta = bn_grads.d_beta
    # ④ 把每一行的 dh_T 交回 day094 的 BPTT（**逐条**），权重梯度相加
    cell = params.sequence.cell
    accumulator = (
        [[0.0] * len(cell.wx[0]) for _ in cell.wx],
        [[0.0] * len(cell.wh[0]) for _ in cell.wh],
        [0.0] * len(cell.bias),
    )
    lengths: list[int] = []
    for index, sequence_cache in enumerate(cache.sequences):
        d_hidden = d_inputs[index]
        length = sequence_cache.length
        lengths.append(length)
        if params.cell_type == CELL_RNN:
            rnn_cell = cell
            assert isinstance(rnn_cell, RNNCell)
            if sequence_cache.rnn_steps is None:  # pragma: no cover - 构造期已挡住
                raise ShapeError("rnn 缓存缺失。")
            grad_states = tuple(
                d_hidden if step == length - 1 else zeros(rnn_cell.hidden_size)
                for step in range(length)
            )
            result = sequence_gradients.rnn_bptt(rnn_cell, sequence_cache.rnn_steps, grad_states)
            _accumulate_cell_gradients(accumulator, result.cell)
        else:
            lstm_cell = cell
            assert isinstance(lstm_cell, LSTMCell)
            if sequence_cache.lstm_steps is None:  # pragma: no cover - 构造期已挡住
                raise ShapeError("lstm 缓存缺失。")
            grad_hidden = tuple(
                d_hidden if step == length - 1 else zeros(lstm_cell.hidden_size)
                for step in range(length)
            )
            lstm_result = sequence_gradients.lstm_bptt(
                lstm_cell, sequence_cache.lstm_steps, grad_hidden
            )
            _accumulate_cell_gradients(accumulator, lstm_result.cell)
    if len(set(lengths)) != 1:
        raise ShapeError(
            "这一批里的序列长度不一致：BatchNorm 只关心 h_T 的批，但 day094 的 BPTT "
            "要求一批里每条样本的展开长度相同（变长批请先 padding / 分桶）。"
        )
    wx_acc, wh_acc, bias_acc = accumulator
    if params.cell_type == CELL_RNN:
        cell_grads: object = sequence_gradients.RNNGradients(
            wx=tuple(tuple(row) for row in wx_acc),
            wh=tuple(tuple(row) for row in wh_acc),
            bias=tuple(bias_acc),
        )
    else:
        cell_grads = sequence_gradients.LSTMGradients(
            wx=tuple(tuple(row) for row in wx_acc),
            wh=tuple(tuple(row) for row in wh_acc),
            bias=tuple(bias_acc),
        )
    sequence_result = SequenceGradients(
        cell=cell_grads,  # type: ignore[arg-type]
        head_weight=head_weight_grad,
        head_bias=head_bias_grad,
        length=lengths[0],
    )
    return RegularizedGradients(
        sequence=sequence_result, d_gamma=d_gamma, d_beta=d_beta
    )


def loss_and_grad(
    params: RegularizedParams,
    batch: Batch,
    *,
    phase: str = PHASE_TRAIN,
    running: RunningStatistics | None = None,
    dropout_rate: float = 0.0,
    dropout_seed: int = 0,
    **kwargs,
) -> tuple[float, RegularizedGradients, RegularizedCache]:
    """一批样本的"前向 → 批均值交叉熵 → 反向"，返回 ``(损失, 梯度, 缓存)``.

    损失与它的梯度都**不重写**：``cross_entropy``（day089）与
    ``cross_entropy_grad_rows``（day090，已除过批大小）各只有一份实现。
    """
    checked = as_batch(batch)
    logits, cache = regularized_forward_cached(
        params,
        checked,
        phase=phase,
        running=running,
        dropout_rate=dropout_rate,
        dropout_seed=dropout_seed,
        **kwargs,
    )
    labels = tuple(label for _inputs, label in checked)
    loss = math.fsum(
        cross_entropy(row, label) for row, label in zip(logits, labels, strict=True)
    ) / len(checked)
    grad_logits = cross_entropy_grad_rows(logits, labels)
    grads = regularized_backward(params, cache, grad_logits)
    return loss, grads, cache


def accuracy(
    params: RegularizedParams,
    batch: Batch,
    *,
    running: RunningStatistics | None = None,
    phase: str = PHASE_TRAIN,
    dropout_rate: float = 0.0,
    use_norm: bool = True,
) -> float:
    """一批样本上的分类准确率（**推理相**：dropout 恒等、BN 用 running）."""
    checked = as_batch(batch)
    logits, _cache = regularized_forward_cached(
        params,
        checked,
        phase=phase,
        running=running,
        dropout_rate=dropout_rate,
        dropout_seed=0,
        use_norm=use_norm,
    )
    hits = sum(
        1
        for row, (_inputs, label) in zip(logits, checked, strict=True)
        if max(range(len(row)), key=lambda index: row[index]) == label
    )
    return hits / len(checked)


def flatten_params(params: RegularizedParams) -> Vector:
    """按写死顺序压平：``day094 的 flatten`` → γ → β."""
    return flatten_sequence_params(params.sequence) + tuple(params.gamma) + tuple(params.beta)


def flatten_gradients(grads: RegularizedGradients) -> Vector:
    """按与 :func:`flatten_params` **相同**的顺序压平梯度（两者必须能逐位相加）."""
    return (
        flatten_sequence_gradients(grads.sequence)
        + tuple(grads.d_gamma)
        + tuple(grads.d_beta)
    )


def unflatten_params(flat: Vector, template: RegularizedParams) -> RegularizedParams:
    """按模板把一串数还原成 :class:`RegularizedParams`（**长度对不上当场拒绝**）."""
    values = tuple(float(value) for value in flat)
    if any(not math.isfinite(value) for value in values):
        raise ShapeError("压平的参数里出现非有限数（nan / inf）：先查那一步的学习率。")
    required = template.parameter_count
    if len(values) != required:
        raise ShapeError(
            f"压平的向量有 {len(values)} 个数，而模板需要 {required} 个："
            "长度对不上时'按顺序切'会静默地填错位置，给出形状正确、数值错位的参数。"
        )
    split = len(values) - 2 * template.features
    sequence = unflatten_sequence_params(values[:split], template=template.sequence)
    gamma = values[split : split + template.features]
    beta = values[split + template.features :]
    return RegularizedParams(sequence=sequence, gamma=gamma, beta=beta)


__all__ = [
    "Batch",
    "RegularizedCache",
    "RegularizedGradients",
    "RegularizedParams",
    "accuracy",
    "as_batch",
    "build_regularized",
    "flatten_gradients",
    "flatten_params",
    "loss_and_grad",
    "regularized_backward",
    "regularized_forward",
    "regularized_forward_cached",
    "unflatten_params",
]
