"""可训练的自注意力：前向七步 + 反向七步（day075 / M7-D1）.

day073 的 ``math_foundations.attention`` 把 ``softmax(QKᵀ/√d_k)·V`` 算得明明白白，
但那 4 个字母里 **Q/K/V 是给定的**。今天把它们换成参数投影出来的：

```text
day073    Attention(Q, K, V)         Q/K/V 是输入          只能验证"公式算得对"
day075    SelfAttention(x; W)        Q = x·W_qᵀ 等四个矩阵   可以问"W 该往哪调"
```

差别只在一处——**多了四个参数**，但它带来一件全新的工作：
**反向传播**。而反向传播的每一行都能写成一句"把上游梯度乘上局部导数"：

```text
前向（七步）                        反向（七步，顺序恰好相反）
① Q/K/V = x·Wᵀ                     ⑦ dW_o = dOutᵀ·context ；dContext = dOut·W_o
② raw = Q·Kᵀ                        ⑥ dWeights = dContext·Vᵀ ；dV = Weightsᵀ·dContext
③ scores = raw/√d_k                 ⑤ dScores = softmax 反向（逐行）
④ 掩码：上三角标成不允许看           ④ 掩码位置梯度置 0
⑤ weights = 逐行 softmax            ③ dRaw = dScores/√d_k
⑥ context = weights·V               ② dQ = dRaw·K ；dK = dRawᵀ·Q
⑦ output = context·W_oᵀ             ① dW_q = dQᵀ·x 等三块 + dInputs = 三条链之和
```

## 三处"漏掉不会报错"的地方

这一课的全部风险集中在这三处，它们各自有独立的测试：

```text
① softmax 的雅可比    dWeights → dScores 不是恒等映射，
                     写成 dScores = dWeights 的后果是"梯度方向大致对、幅度错"
② 缩放系数            前向除了 √d_k，反向就必须乘回 1/√d_k；
                     漏掉的后果是整体差一个 √d_k 倍（比如 2.45 倍）
③ 掩码位置的梯度      被掩码的位置在**前向**权重是 0，
                     但它的 dWeights 一般不为 0；不置 0 的后果是
                     位置 i 会通过一条"它看不到的路径"收到梯度
```

第 ③ 条最隐蔽：它**不会**让"权重"出现非零（前向掩码仍然生效），
只会让 key 的梯度里混进"未来位置"的贡献——而损失曲线看起来一切正常。

## 一句纪律：全零输入得到全零输出

四个投影都没有偏置（现代 Transformer 的常见选择），因此
``SelfAttention(0) = 0``。这条性质有两个直接好处：可被断言、
以及"某个位置输出为 0"一定来自"它那一行的混合结果确实是 0"，
而不是偏置的残留。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.math_foundations.attention import (
    causal_mask,
    masked_softmax_rows,
)
from smart_research_agent.math_foundations.linalg import argmax, matmul, transpose
from smart_research_agent.math_foundations.probability import entropy
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.errors import NumericError, ShapeError
from smart_research_agent.transformer_core.types import (
    AttentionForward,
    AttentionParams,
    ParameterGradients,
    check_weights_are_a_distribution,
    project,
)


def full_mask(size: int) -> tuple[tuple[bool, ...], ...]:
    """全开掩码（每一行都能看到所有位置）——**显式给出**，而不是 ``None``.

    与 day073 的因果掩码同一取向：掩码是一张可以被打印出来看的表，
    "没有掩码"也应当是那张表（全是 ``True``），而不是一个需要读代码才知道含义的 ``None``。
    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise ShapeError(f"掩码边长必须是整数，收到 {size!r}。")
    if size < 1:
        raise ShapeError(f"掩码边长必须 >= 1，收到 {size}。")
    return tuple(tuple(True for _ in range(size)) for _ in range(size))


def resolve_mask(
    size: int,
    *,
    causal: bool,
    mask: tuple[tuple[bool, ...], ...] | None = None,
) -> tuple[tuple[bool, ...], ...]:
    """确定用哪张掩码：显式传入优先，否则按 ``causal`` 生成.

    两条规则：

    ```text
    ① ``causal=True`` 与显式掩码**同时**给 → 报错
       两个来源同时生效时，"到底用的是哪张"要靠读代码才知道——
       而"多看到一个位置"这件事在输出上只表现为"数值略有不同"
    ② 掩码必须能保证**每一行至少有一个允许的位置**
       （因果掩码天然满足：第 0 行至少能看到自己）
    ```
    """
    if causal and mask is not None:
        raise ShapeError(
            "``causal=True`` 与显式掩码不能同时给：两个来源同时生效时，"
            "到底用的是哪一张需要读代码才知道，而'多看到一个位置'在输出上"
            "只表现为数值略有不同——这类分歧没有异常，只有更难查的结果。"
        )
    resolved = causal_mask(size) if causal else full_mask(size)
    if mask is None:
        return resolved
    checked = tuple(tuple(bool(value) for value in row) for row in mask)
    if len(checked) != size or any(len(row) != size for row in checked):
        raise ShapeError(
            f"掩码形状 {len(checked)}×{len(checked[0]) if checked else 0} 与 "
            f"({size}, {size}) 不一致。"
        )
    for index, row in enumerate(checked):
        if not any(row):
            raise NumericError(
                f"掩码第 {index} 行没有任何允许的位置：这一行的注意力分布没有定义——"
                "因果掩码应当保证第 0 行至少能看到自己。"
            )
    return checked


def _check_inputs(inputs: Matrix) -> Matrix:
    """校验输入：非空、每行等长、**没有全零行**.

    全零行的处理是这一层的一个显式决定：它会让这一行的打分全是 0、
    softmax 给出均匀分布，而"均匀分布"看起来像"模型还没学到"——
    这两种情况的修法人完全不同（一个改数据、一个改训练）。
    """
    checked = validate_matrix(inputs, name="inputs")
    for index, row in enumerate(checked):
        if all(value == 0.0 for value in row):
            raise NumericError(
                f"输入第 {index} 行全零：它会让这一行的打分全是 0、softmax 给出均匀分布，"
                "而均匀分布看起来像'模型还没学到'。"
                "如果这是 padding，请显式裁剪掉或用掩码处理，不要交给这一层猜。"
            )
    return checked


def self_attention(
    params: AttentionParams,
    inputs: Matrix,
    *,
    causal: bool = False,
    mask: tuple[tuple[bool, ...], ...] | None = None,
) -> AttentionForward:
    """一层自注意力的前向（返回**全部中间量**，因为反向要用）.

    七步见模块说明。三处细节值得单独指出：

    ```text
    ① 缩放作用在**打分**上（而不是在 softmax 里）：这样反向只需乘回 1/√d_k
    ② 掩码用 day073 的 masked_softmax_rows：被掩码的位置权重**恰好是 0.0**、
       且不计入分母（而不是写 -inf 再指望 exp 把它变成 0）
    ③ 输出投影 W_o 是"多头/单头拼接之后那一层"的最小版本：
       没有它，输出的每个维度都被限制成 value 的线性组合，
       而加上它之后模型可以自己决定"怎么用混合结果"
    ```
    """
    if not isinstance(params, AttentionParams):
        raise ShapeError(f"params 必须是 AttentionParams，收到 {type(params).__name__}。")
    checked_inputs = _check_inputs(inputs)
    rows = len(checked_inputs)
    resolved_mask = resolve_mask(rows, causal=causal, mask=mask)

    queries = project(checked_inputs, params.w_query)
    keys = project(checked_inputs, params.w_key)
    values = project(checked_inputs, params.w_value)

    scale = params.shape.scale
    raw = matmul(queries, transpose(keys))
    scores = tuple(tuple(value * scale for value in row) for row in raw)

    weights = masked_softmax_rows(scores, resolved_mask)
    check_weights_are_a_distribution(weights)

    context = matmul(weights, values)
    output = project(context, params.w_output)

    entropies = tuple(entropy(row) for row in weights)
    peaks = tuple(max(row) for row in weights)
    indices = tuple(argmax(row) for row in weights)

    notes = [
        f"缩放系数 {scale:.6f}（1/√d_k，d_k={params.shape.keys}）作用在**打分**上，"
        "因此反向只需乘回同一个数",
        "被掩码的位置权重恰好是 0.0、且不计入 softmax 的分母"
        "（显式掩码，不是把打分写成 -inf）",
        "四个投影都没有偏置，因此全零输入得到全零输出",
    ]
    if causal:
        notes.append(
            "因果掩码：上三角（j > i）恰好为 0.0——位置 i 看不到它之后的任何位置"
        )
    return AttentionForward(
        params=params,
        inputs=checked_inputs,
        queries=queries,
        keys=keys,
        values=values,
        raw=raw,
        scores=scores,
        weights=weights,
        context=context,
        output=output,
        causal=causal,
        mask=resolved_mask,
        scale=scale,
        row_entropies=entropies,
        peak_weights=peaks,
        peak_indices=indices,
        notes=tuple(notes),
    )


def softmax_backward_row(weights_row: Vector, grad_row: Vector) -> Vector:
    """softmax 的**雅可比-向量积**（逐行）：``dScores_i = s_i(g_i − Σ_j g_j s_j)``.

    这是这一课最容易被简化掉的一步。完整的雅可比是 ``∂p_i/∂z_j = p_i(δ_ij − p_j)``
    （day073 的 ``gradcheck`` 单独验证过它），而我们要的是"某个下游梯度 ``g`` 怎么传回 ``z``"：

    ```text
    dScores_i = Σ_j g_j · ∂p_j/∂z_i = Σ_j g_j · p_j(δ_ji − p_i) = p_i(g_i − Σ_j g_j p_j)
    ```

    三条值得记住的性质：

    ```text
    ① 写成 dScores = dWeights 是最常见的简化。它不会报错，方向也"大致"对，
       因此训练仍然能下降——只是每一步的幅度不对（差一个"减去行平均"的量）
    ② 括号里那一项 Σ_j g_j p_j 是**整行的加权平均**，不是逐元素的
    ③ 权重为 0 的位置（被掩码）自动得到 0 梯度：0·(g − 平均) = 0
    ```
    """
    if len(weights_row) != len(grad_row):
        raise ShapeError(
            f"权重行 {len(weights_row)} 维而梯度行 {len(grad_row)} 维："
            "softmax 的反向要求两者逐位对应。"
        )
    average = math.fsum(g * s for g, s in zip(grad_row, weights_row))
    return tuple(s * (g - average) for s, g in zip(weights_row, grad_row))


def attention_backward(
    forward: AttentionForward,
    grad_output: Matrix,
) -> ParameterGradients:
    """反向传播：从"损失对输出的梯度"一路回到四块参数梯度与输入梯度.

    ``grad_output`` 的形状必须与 ``forward.output`` 一致。
    返回值里既有四块参数梯度（训练用），也有输入梯度
    （让这一层能被**堆叠**——day079/080 的 Encoder 就是两层这样的层串起来）。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.output):
        raise ShapeError(
            f"输出梯度的形状 {matrix_shape(checked_grad)} 与前向输出的形状 "
            f"{matrix_shape(forward.output)} 不一致。"
        )

    params = forward.params

    # ⑦ 输出投影：output = context·W_oᵀ
    grad_w_output = matmul(transpose(checked_grad), forward.context)
    grad_context = matmul(checked_grad, params.w_output)

    # ⑥ 加权混合：context = weights·V
    grad_weights = matmul(grad_context, transpose(forward.values))
    grad_values = matmul(transpose(forward.weights), grad_context)

    # ⑤④ softmax 反向 + 掩码：被掩码的位置梯度必须是 0
    grad_scores = tuple(
        tuple(
            value if forward.mask[row][column] else 0.0
            for column, value in enumerate(
                softmax_backward_row(forward.weights[row], grad_weights[row])
            )
        )
        for row in range(forward.tokens)
    )

    # ③ 缩放：scores = raw · scale
    grad_raw = tuple(tuple(value * forward.scale for value in row) for row in grad_scores)

    # ② 打分：raw = Q·Kᵀ
    grad_queries = matmul(grad_raw, forward.keys)
    grad_keys = matmul(transpose(grad_raw), forward.queries)

    # ① 投影：q = x·W_qᵀ 等（dW = gradᵀ·x，dx = grad·W）
    grad_w_query = matmul(transpose(grad_queries), forward.inputs)
    grad_w_key = matmul(transpose(grad_keys), forward.inputs)
    grad_w_value = matmul(transpose(grad_values), forward.inputs)
    grad_inputs = _matrix_sum(
        matmul(grad_queries, params.w_query),
        matmul(grad_keys, params.w_key),
        matmul(grad_values, params.w_value),
    )

    return ParameterGradients(
        grad_w_query=grad_w_query,
        grad_w_key=grad_w_key,
        grad_w_value=grad_w_value,
        grad_w_output=grad_w_output,
        grad_inputs=grad_inputs,
    )


def _matrix_sum(*matrices: Matrix) -> Matrix:
    """逐元素相加（形状必须一致）.

    ``grad_inputs`` 是**三条链之和**（q/k/v 各一条）。
    少任何一条都不会报错——它只会让输入梯度偏小，
    而输入梯度偏小在多层层叠里表现为"靠近输入的那几层学得慢"。
    """
    checked = tuple(validate_matrix(matrix, name="matrix") for matrix in matrices)
    rows, columns = matrix_shape(checked[0])
    for index, matrix in enumerate(checked):
        if matrix_shape(matrix) != (rows, columns):
            raise ShapeError(
                f"第 {index} 块矩阵形状 {matrix_shape(matrix)} 与第一块 {(rows, columns)} 不一致。"
            )
    return tuple(
        tuple(math.fsum(matrix[row][column] for matrix in checked) for column in range(columns))
        for row in range(rows)
    )


# --------------------------------------------------------------------------- #
# 目标函数：这一课用逐元素 MSE（它的梯度是零阶/一阶都平凡的，便于手算复核）
# --------------------------------------------------------------------------- #


def mean_squared_error(output: Matrix, target: Matrix) -> float:
    """逐元素均方误差 ``mean((y − t)²)``（**对所有行与列**）.

    为什么这一课用 MSE 而不是交叉熵：MSE 的梯度是 ``2(y − t)/N``——
    **一行就能手算**，因此"注意力那复杂的反向"里唯一"外部输入的梯度"是平凡的。
    这样任何梯度校验失败都只可能来自注意力那七步，而不是来自损失函数的定义。
    """
    checked_output = validate_matrix(output, name="output")
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_output) != matrix_shape(checked_target):
        raise ShapeError(
            f"输出形状 {matrix_shape(checked_output)} 与目标形状 "
            f"{matrix_shape(checked_target)} 不一致。"
        )
    rows, columns = matrix_shape(checked_output)
    total = 0.0
    for output_row, target_row in zip(checked_output, checked_target, strict=True):
        for value, expected in zip(output_row, target_row, strict=True):
            total += (value - expected) ** 2
    return total / (rows * columns)


def mse_gradient(output: Matrix, target: Matrix) -> Matrix:
    """``∂MSE/∂y = 2(y − t)/N``（``N = 行数 × 列数``）."""
    checked_output = validate_matrix(output, name="output")
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_output) != matrix_shape(checked_target):
        raise ShapeError(
            f"输出形状 {matrix_shape(checked_output)} 与目标形状 "
            f"{matrix_shape(checked_target)} 不一致。"
        )
    rows, columns = matrix_shape(checked_output)
    count = rows * columns
    return tuple(
        tuple(2.0 * (value - expected) / count for value, expected in zip(o, t, strict=True))
        for o, t in zip(checked_output, checked_target, strict=True)
    )


def _check_supervised_rows(rows: Sequence[int], total: int) -> tuple[int, ...]:
    """校验监督行下标：非空、不重复、落在范围内（**顺序被保留**）."""
    if not rows:
        raise ShapeError(
            "监督行不能为空：没有任何监督行时损失没有定义——"
            "返回 0.0 会让'这一批数据全被屏蔽了'伪装成'损失完美'。"
        )
    checked: list[int] = []
    for row in rows:
        if isinstance(row, bool) or not isinstance(row, int):
            raise ShapeError(f"监督行下标必须是整数，收到 {row!r}。")
        if not 0 <= row < total:
            raise ShapeError(f"监督行下标 {row} 落在 [0, {total}) 之外。")
        if row in checked:
            raise ShapeError(
                f"监督行下标 {row} 重复：重复计数会让这一行的损失被算两次，"
                "而'某个位置更重要'这件事应当由权重表达，不是靠重复。"
            )
        checked.append(row)
    return tuple(checked)


def masked_mean_squared_error(
    output: Matrix, target: Matrix, rows: Sequence[int]
) -> float:
    """只在若干**监督行**上算 MSE（分母是监督行数 × 列数）.

    分母的选择与 day050 的 ``masked_cross_entropy`` 是同一个问题：
    用全部行数当分母会让损失被系统性地**压低**（分子只算了几行），
    而"损失从 1.2 降到 0.4"看起来仍然很美——它意味着梯度被偷偷缩小了。
    """
    checked_output = validate_matrix(output, name="output")
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_output) != matrix_shape(checked_target):
        raise ShapeError("输出与目标的形状必须一致。")
    total_rows, columns = matrix_shape(checked_output)
    selected = _check_supervised_rows(rows, total_rows)
    total = 0.0
    for row in selected:
        for value, expected in zip(checked_output[row], checked_target[row], strict=True):
            total += (value - expected) ** 2
    return total / (len(selected) * columns)


def masked_mse_gradient(output: Matrix, target: Matrix, rows: Sequence[int]) -> Matrix:
    """``∂masked_MSE/∂y``：只在监督行上非零（其余行**恰好是 0.0**）.

    "其余行是 0"这一点值得写下来：它意味着**未被监督的位置不会收到任何梯度**。
    如果实现里不小心让它们非零，那些行会开始"朝着一个没有定义的目标"移动——
    而损失曲线只会稍微好看一点。
    """
    checked_output = validate_matrix(output, name="output")
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_output) != matrix_shape(checked_target):
        raise ShapeError("输出与目标的形状必须一致。")
    total_rows, columns = matrix_shape(checked_output)
    selected = _check_supervised_rows(rows, total_rows)
    count = len(selected) * columns
    chosen = set(selected)
    return tuple(
        tuple(
            2.0 * (value - expected) / count if row in chosen else 0.0
            for value, expected in zip(o, t, strict=True)
        )
        for row, (o, t) in enumerate(zip(checked_output, checked_target, strict=True))
    )


def row_argmax_hits(output: Matrix, target: Matrix, rows: Sequence[int]) -> float:
    """监督行上的**最高分命中率**（``argmax`` 是否一致）.

    它是比损失更直观的训练进度指标：损失是"混出来的向量离目标多远"，
    命中率是"最大的那个分量对上了吗"。两者一起看才能区分两种情况：

    ```text
    损失降但命中率不涨   混合比例在接近，但还没把目标顶到第一位（早期正常）
    命中率涨但损失不降   目标顶上去了，但其余分量还有残差（继续训练会好）
    ```
    """
    checked_output = validate_matrix(output, name="output")
    checked_target = validate_matrix(target, name="target")
    total_rows, _columns = matrix_shape(checked_output)
    selected = _check_supervised_rows(rows, total_rows)
    hits = sum(
        1 for row in selected if argmax(checked_output[row]) == argmax(checked_target[row])
    )
    return hits / len(selected)


__all__ = [
    "attention_backward",
    "full_mask",
    "masked_mean_squared_error",
    "masked_mse_gradient",
    "mean_squared_error",
    "mse_gradient",
    "resolve_mask",
    "row_argmax_hits",
    "self_attention",
    "softmax_backward_row",
]
