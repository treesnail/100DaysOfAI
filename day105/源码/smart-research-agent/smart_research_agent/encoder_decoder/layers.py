"""``encoder_decoder`` 的三个子层与块的前向/反向（day079 / M7-D4）.

这一章的主体是三个算子，而它们各自只解决一件事：

```text
LayerNorm      每一行标准化          → 它**逐行**，因此不跨行取统计量
前馈            逐位置的两层全连接      → 它也是**逐行**的
残差            y = x + F(x)         → 它在反向里多出一条**不减值的路**
```

前两个算子的“逐行”性质合起来给出一个跨天的结论：**打破置换等变性的只有
位置编码与掩码**（day078 的第 4 条性质），LN 与前馈都不是。
第三个算子是本课唯一“改一行、效果差一个数量级”的地方（第 9 章量它）。

## 一次块前向的六个阶段（pre-LN）

```text
① norm1 = LN(inputs)        ② branch1 = attention(norm1)   ③ residual1 = inputs + branch1
④ norm2 = LN(residual1)     ⑤ branch2 = ffn(norm2)         ⑥ output    = residual1 + branch2
```

post-LN 的差别只在 **LN 的位置**：它是 `branch → add → norm`，
因此六个阶段的顺序变成 `branch1 → add1 → norm1 → branch2 → add2 → norm2`。
两者的**形状完全一样**——这是第 9 章那个实验的全部意义。

## 反向里唯一“漏掉不会报错”的一处

```text
dResidual1 = dOutput + dNorm2             ← 前一项来自残差那条 +1 的路
dInputs    = dResidual1 + dNorm1_inputs   ← 后一项来自第一个 LN 的那条链
```

漏掉任何一项都不会报错：形状全对、训练也真的在动，
只是**梯度少了一条不减值的路**——而在深堆叠里，那一条路正是梯度能不能活下来的原因。

## 一处与自注意力的关键差别

```text
自注意力     Q/K/V 都来自同一路 → 四个投影的**列数相同**（day075 的 AttentionParams 要求这一点）
交叉注意力   Q 来自 target、K/V 来自 source → W_q 与 W_k/W_v 的列数**可以不同**
```

因此本课给交叉注意力单独做了一个参数记录（``CrossParameters``），
而“照抄 ``AttentionParams``”会在构造那一刻就被拒。

## 交叉注意力**绝不能**加因果掩码

```text
自注意力（解码器）   Q/K/V 都来自同一路 → 必须加因果掩码（不许看未来）
交叉注意力           Q 来自解码器、K/V 来自编码器 → 因果掩码**没有意义**
```

本包把“给它加因果掩码”做成一条**显式拒绝**（``AssemblyError``），
理由是它的失效方式很坏：只有在 ``n_tgt == n_src`` 时才不报错，
而那时它悄悄把源序列的后半段从注意范围里删掉了。
"""

from __future__ import annotations

import math
from typing import Any

from smart_research_agent.encoder_decoder.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    DEFAULT_EPSILON,
    ENCODER_BLOCK_STAGES,
    FFNCache,
    FFNGradients,
    FFNWeights,
    NORM_PRE,
    BlockForward,
    BlockGradients,
    BlockParameters,
    BlockShape,
    CrossForward,
    CrossGradients,
    CrossParameters,
    CrossShape,
    DecoderGradients,
    NormCache,
    NormGradients,
    _checked_activation,
    _checked_placement,
    validate_epsilon,
)
from smart_research_agent.math_foundations.linalg import matmul, softmax, transpose
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
    validate_vector,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    self_attention,
    softmax_backward_row,
)
from smart_research_agent.transformer_core.types import AttentionForward, AttentionParams


def _checked_epsilon(epsilon: Any) -> float:
    """校验 ``eps``（转发 ``types.validate_epsilon``）."""
    return validate_epsilon(epsilon)


def zero_matrix(rows: int, columns: int) -> Matrix:
    """零矩阵（反向里“这一项不存在”就用它，而不是到处写分支）."""
    if rows < 1 or columns < 1:
        raise ShapeError(f"零矩阵的形状必须为正：收到 ({rows}, {columns})。")
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


# ---------------------------------------------------------------------- 激活函数


def activate(values: Matrix, activation: str = ACTIVATION_RELU) -> Matrix:
    """逐元素激活（``relu`` / ``gelu``）——**它逐元素，因此它逐行**."""
    checked = validate_matrix(values, name="values")
    resolved = _checked_activation(activation)
    if resolved == ACTIVATION_RELU:
        return tuple(tuple(value if value > 0.0 else 0.0 for value in row) for row in checked)
    root_two = math.sqrt(2.0)
    return tuple(
        tuple(0.5 * value * (1.0 + math.erf(value / root_two)) for value in row)
        for row in checked
    )


def activation_backward(
    pre_activation: Matrix,
    grad_output: Matrix,
    activation: str = ACTIVATION_RELU,
) -> Matrix:
    """激活的导数**逐元素**乘上回传梯度.

    ```text
    relu   在 x > 0 处是 1、在 x < 0 处是 0，x = 0 处取**次梯度 0**
    gelu   d/dx = 0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)      在 0 处恰好是 0.5
    ```

    零点处的取值不是细节：ReLU 在 ``x = 0`` 处不可导，本包取 0
    （“关掉的神经元在反向里梯度**恰好**是 0”），
    这也让 :attr:`FFNCache.summary_line` 里那个“激活后零点占比”成为一个有意义的读数。
    """
    checked_pre = validate_matrix(pre_activation, name="pre_activation")
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_pre) != matrix_shape(checked_grad):
        raise ShapeError(
            f"激活前值 {matrix_shape(checked_pre)} 与回传梯度 "
            f"{matrix_shape(checked_grad)} 形状必须一致（激活是逐元素的）。"
        )
    resolved = _checked_activation(activation)
    if resolved == ACTIVATION_RELU:
        return tuple(
            tuple(
                grad if value > 0.0 else 0.0
                for value, grad in zip(row, grad_row, strict=True)
            )
            for row, grad_row in zip(checked_pre, checked_grad, strict=True)
        )
    root_two = math.sqrt(2.0)
    root_two_pi = math.sqrt(2.0 * math.pi)
    rows: list[Vector] = []
    for row, grad_row in zip(checked_pre, checked_grad, strict=True):
        cells: list[float] = []
        for value, grad in zip(row, grad_row, strict=True):
            derivative = 0.5 * (1.0 + math.erf(value / root_two)) + (
                value * math.exp(-0.5 * value * value) / root_two_pi
            )
            cells.append(grad * derivative)
        rows.append(tuple(cells))
    return tuple(rows)


# ---------------------------------------------------------------------- LayerNorm


def layer_norm(
    inputs: Matrix,
    *,
    gamma: Vector | None = None,
    beta: Vector | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Matrix, NormCache]:
    """逐 token 的 LayerNorm：``y = γ ⊙ (x − μ)/√(σ² + eps) + β``.

    ```text
    μ_i  = (1/d)·Σ_j x_ij                    **逐行**的均值（不跨行）
    σ²_i = (1/d)·Σ_j (x_ij − μ_i)²           **有偏**方差（除以 d，与 PyTorch 一致）
    x̂_ij = (x_ij − μ_i)/√(σ²_i + eps)        于是每一行的均值是 0、方差是 1
    y_ij = γ_j·x̂_ij + β_j                    γ/β 是**逐维**的（长度 = 隐藏维）
    ```

    三条立刻可测的性质（第 8 章）：每一行标准化、**平移不变**、**正尺度不变**。
    后两条也是它“丢掉信息”的地方：它们说明 LayerNorm 之后
    “这一行整体加了多大一个常数”与“乘了多大一个正数”都**不再可读**——
    这不是缺陷，而是把这两样东西从“要学的东西”里划掉。

    ``σ² = 0``（整行取值相同）时本函数**不抛异常**：`eps` 的目的正是让这一行有定义
    （标准化结果恰好是 0，于是输出等于 β）。
    """
    checked = validate_matrix(inputs, name="inputs")
    rows, columns = matrix_shape(checked)
    epsilon_value = _checked_epsilon(epsilon)
    resolved_gamma = (
        tuple(1.0 for _ in range(columns))
        if gamma is None
        else validate_vector(gamma, name="gamma")
    )
    resolved_beta = (
        tuple(0.0 for _ in range(columns))
        if beta is None
        else validate_vector(beta, name="beta")
    )
    if len(resolved_gamma) != columns or len(resolved_beta) != columns:
        raise ShapeError(
            f"γ/β 的长度必须等于列数 {columns}，收到 {len(resolved_gamma)} 与 "
            f"{len(resolved_beta)}：LayerNorm 的 γ/β 是**逐维**的。"
        )
    means: list[float] = []
    variances: list[float] = []
    normalized: list[Vector] = []
    outputs: list[Vector] = []
    for row in checked:
        mean = math.fsum(row) / columns
        variance = math.fsum((value - mean) ** 2 for value in row) / columns
        if not math.isfinite(variance):
            raise NumericError(
                f"某一行的方差是 {variance!r}：LayerNorm 的分母没有定义——"
                "先检查这一行里有没有非有限数。"
            )
        scale = math.sqrt(variance + epsilon_value)
        checked_row = tuple((value - mean) / scale for value in row)
        means.append(mean)
        variances.append(variance)
        normalized.append(checked_row)
        outputs.append(
            tuple(
                g * value + b
                for g, b, value in zip(resolved_gamma, resolved_beta, checked_row, strict=True)
            )
        )
    cache = NormCache(
        inputs=checked,
        mean=tuple(means),
        variance=tuple(variances),
        normalized=tuple(normalized),
        epsilon=epsilon_value,
        gamma=resolved_gamma,
    )
    return tuple(outputs), cache


def layer_norm_backward(
    cache: NormCache,
    grad_output: Matrix,
    *,
    beta: Vector | None = None,
) -> NormGradients:
    """LayerNorm 的反向（三块：``dγ``、``dβ``、``dx``）.

    ```text
    dβ_j  = Σ_i dY_ij                                     ← 纯求和（不经过 x̂）
    dγ_j  = Σ_i dY_ij · x̂_ij                               ← 唯一乘以 x̂ 的那一项
    dX̂_ij = dY_ij · γ_j
    dx_ij = (1/σ_i)·( dX̂_ij − mean_j(dX̂_i) − x̂_ij·mean_j(dX̂_i ⊙ x̂_i) )
    ```

    第三条公式的**两个减项**是它最容易漏的地方：

    ```text
    只写 dX̂/σ        漏掉“标准化把均值减掉了”这一层 → 梯度偏大，而形状全对
    漏掉第二个减项    少了“方差也依赖 x”那一条链 → 前向仍然对，训练会慢
    ```

    两项都用到**行均值**（对列取平均），因此这一条反向也是“逐行”的——
    它与前向的逐行性质在代码里是同一条约束。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(cache.inputs):
        raise ShapeError(
            f"回传梯度 {matrix_shape(checked_grad)} 与输入 "
            f"{matrix_shape(cache.inputs)} 形状不一致。"
        )
    rows, columns = matrix_shape(cache.inputs)
    gamma = cache.gamma if cache.gamma is not None else tuple(1.0 for _ in range(columns))
    if beta is not None:
        checked_beta = validate_vector(beta, name="beta")
        if len(checked_beta) != columns:
            raise ShapeError(f"beta 的长度 {len(checked_beta)} 与列数 {columns} 不一致。")
    grad_gamma = [0.0] * columns
    grad_beta = [0.0] * columns
    grad_inputs: list[Vector] = []
    for index in range(rows):
        row_grad = checked_grad[index]
        row_norm = cache.normalized[index]
        scaled = tuple(g * value for g, value in zip(gamma, row_grad, strict=True))
        mean_scaled = math.fsum(scaled) / columns
        mean_scaled_norm = (
            math.fsum(s * z for s, z in zip(scaled, row_norm, strict=True)) / columns
        )
        scale = math.sqrt(cache.variance[index] + cache.epsilon)
        grad_inputs.append(
            tuple(
                (s - mean_scaled - z * mean_scaled_norm) / scale
                for s, z in zip(scaled, row_norm, strict=True)
            )
        )
        for column in range(columns):
            grad_gamma[column] += row_grad[column] * row_norm[column]
            grad_beta[column] += row_grad[column]
    return NormGradients(
        grad_gamma=tuple(grad_gamma),
        grad_beta=tuple(grad_beta),
        grad_inputs=tuple(grad_inputs),
    )


# ---------------------------------------------------------------------- 前馈


def feed_forward(
    inputs: Matrix,
    weights: FFNWeights,
    *,
    activation: str = ACTIVATION_RELU,
) -> tuple[Matrix, FFNCache]:
    """逐位置前馈：``y = act(x·W_inᵀ + b_in)·W_outᵀ + b_out``.

    ``d_ff`` 通常是 ``d`` 的 4 倍（原论文的取值）——先扩张再压回，
    于是它给每一行提供了一个“在更宽的空间里做非线性”的机会。
    这一层**逐行**：第 ``i`` 行的输出只依赖第 ``i`` 行，因此它不破坏置换等变性
    （第 8 章第 5 条性质把它写成一条**逐位**断言）。
    """
    checked = validate_matrix(inputs, name="inputs")
    resolved = _checked_activation(activation)
    rows, columns = matrix_shape(checked)
    if matrix_shape(weights.w_in)[1] != columns:
        raise ShapeError(
            f"W_in 的列数 {matrix_shape(weights.w_in)[1]} 与输入列数 {columns} 不一致。"
        )
    pre_activation: list[Vector] = []
    for row in checked:
        pre_activation.append(
            tuple(
                math.fsum(w * value for w, value in zip(weight_row, row, strict=True)) + bias
                for weight_row, bias in zip(weights.w_in, weights.b_in, strict=True)
            )
        )
    activated = activate(tuple(pre_activation), resolved)
    outputs: list[Vector] = []
    for row in activated:
        outputs.append(
            tuple(
                math.fsum(w * value for w, value in zip(weight_row, row, strict=True)) + bias
                for weight_row, bias in zip(weights.w_out, weights.b_out, strict=True)
            )
        )
    cache = FFNCache(
        inputs=checked,
        pre_activation=tuple(pre_activation),
        hidden=activated,
        weights=weights,
        activation=resolved,
    )
    return tuple(outputs), cache


def feed_forward_backward(cache: FFNCache, grad_output: Matrix) -> FFNGradients:
    """前馈的反向（五块：``dW_in``、``db_in``、``dW_out``、``db_out``、``dx``）.

    两个偏置的梯度都是**按行相加**（它们被所有行共用），
    两个权重的梯度都是“回传梯度ᵀ × 输入”——
    四条式子里没有一处新东西，全部是 day073 的 `matmul` 与 day074 的链式法则。

    与 day075 的一条口径差别值得注意：那一层是**整个矩阵**参与 `matmul`，
    而这里逐行算——**逐行算不是笔误**：前馈的定义就是“每一行各自过一个两层网络”，
    写成矩阵乘法只是把它向量化之后的等价形式（第 8 章第 5 条性质断言这两者一致）。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(cache.inputs):
        raise ShapeError(
            f"回传梯度 {matrix_shape(checked_grad)} 与输入 "
            f"{matrix_shape(cache.inputs)} 形状不一致。"
        )
    rows, columns = matrix_shape(cache.inputs)
    ffn = matrix_shape(cache.hidden)[1]
    grad_w_out: list[list[float]] = [[0.0] * ffn for _ in range(columns)]
    grad_b_out = [0.0] * columns
    for row_grad, hidden_row in zip(checked_grad, cache.hidden, strict=True):
        for column in range(columns):
            grad_b_out[column] += row_grad[column]
            for index in range(ffn):
                grad_w_out[column][index] += row_grad[column] * hidden_row[index]
    grad_hidden: list[Vector] = []
    for row_grad in checked_grad:
        # dh_j = Σ_c dY_c · W_out[c][j]：遍历的是 W_out 的**列**，因此下标顺序不能反
        grad_hidden.append(
            tuple(
                math.fsum(
                    row_grad[column] * cache.weights.w_out[column][index]
                    for column in range(columns)
                )
                for index in range(ffn)
            )
        )
    grad_pre = activation_backward(cache.pre_activation, tuple(grad_hidden), cache.activation)
    grad_w_in: list[list[float]] = [[0.0] * columns for _ in range(ffn)]
    grad_b_in = [0.0] * ffn
    for row_grad, input_row in zip(grad_pre, cache.inputs, strict=True):
        for index in range(ffn):
            grad_b_in[index] += row_grad[index]
            for column in range(columns):
                grad_w_in[index][column] += row_grad[index] * input_row[column]
    grad_inputs = tuple(
        # dx_c = Σ_j dPre_j · W_in[j][c]：同样按**列**收，而不是按行
        tuple(
            math.fsum(
                row_grad[index] * cache.weights.w_in[index][column]
                for index in range(ffn)
            )
            for column in range(columns)
        )
        for row_grad in grad_pre
    )
    del rows
    return FFNGradients(
        grad_w_in=tuple(tuple(row) for row in grad_w_in),
        grad_b_in=tuple(grad_b_in),
        grad_w_out=tuple(tuple(row) for row in grad_w_out),
        grad_b_out=tuple(grad_b_out),
        grad_inputs=grad_inputs,
    )


# ---------------------------------------------------------------------- 残差


def add_residual(left: Matrix, right: Matrix, *, use_residual: bool = True) -> Matrix:
    """残差：``y = x + F(x)``（``use_residual=False`` 时退化成 ``y = F(x)``）.

    两件事必须一起说清：

    ```text
    ① 形状必须一致        它不是“广播”，两个 (n, d) 逐位相加
    ② 关掉残差是一个**开关**，而不是另一份实现 ——
      关掉之后整段代码路径不变，只是那一项 +x 没有了（第 9 章的对照就靠它）
    ```
    """
    checked_left = validate_matrix(left, name="left")
    checked_right = validate_matrix(right, name="right")
    if matrix_shape(checked_left) != matrix_shape(checked_right):
        raise ShapeError(
            f"残差相加的两边形状不同：{matrix_shape(checked_left)} 与 "
            f"{matrix_shape(checked_right)}——残差要求两条支路逐位对齐。"
        )
    if not use_residual:
        return checked_right
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(checked_left, checked_right, strict=True)
    )


def add_matrices(left: Matrix, right: Matrix) -> Matrix:
    """逐位相加（梯度累加用；形状不同抛 ``ShapeError``）."""
    return add_residual(left, right, use_residual=True)


# ---------------------------------------------------------------------- 编码器块


def block_attention(
    params: BlockParameters,
    inputs: Matrix,
    attention_params: AttentionParams,
    *,
    placement: str = NORM_PRE,
    causal: bool = False,
) -> AttentionForward:
    """按摆放位置给出**正确**的注意力账：pre 作用在 ``LN(x)`` 上、post 作用在 ``x`` 上.

    这个函数存在的原因是第一版踩到的一个坑，而它的表现是“``norm1_gamma`` 那一项
    差 1e-1 量级“——最像 bug 的量级。根因是：

    ```text
    pre-LN 时   注意力作用在 LN(x) 上 ⇒ γ₁/β₁ 会改变注意力的输出
    若把算好的注意力账从外面传进来 ⇒ 数值侧扰动 γ₁ 时分支一**不变**，
    而解析侧（用 attn 的 grad_inputs）却算进了那一条链 ⇒ 两边算的不是同一个函数
    ```

    修法就是把“造注意力”这件事放在**同一个函数**里：调用方只提供
    ``attention_params``，块与数值差分各自用它造出同一个前向。
    """
    resolved_placement = _checked_placement(placement)
    checked_inputs = validate_matrix(inputs, name="inputs")
    if resolved_placement == NORM_PRE:
        normed, _cache = layer_norm(
            checked_inputs, gamma=params.norm1_gamma, beta=params.norm1_beta
        )
    else:
        normed = checked_inputs
    return self_attention(attention_params, normed, causal=causal)


def stage_order(placement: str) -> tuple[str, ...]:
    """按 ``placement`` 给出六个阶段的**实际顺序**.

    ```text
    pre-LN    norm1 → branch1 → add1 → norm2 → branch2 → add2
    post-LN   branch1 → add1 → norm1 → branch2 → add2 → norm2
    ```

    同一批阶段名，只是 LN 的位置转过一格——这是“pre 与 post 形状完全一样”
    这句话在代码里的样子。
    """
    resolved = _checked_placement(placement)
    if resolved == NORM_PRE:
        return ENCODER_BLOCK_STAGES
    return ("branch1", "add1", "norm1", "branch2", "add2", "norm2")


def _block_caches(
    forward: BlockForward,
    params: BlockParameters,
    activation: str,
) -> tuple[NormCache, NormCache, FFNCache]:
    """按记录里的中间量**重算**三个子层的账（反向需要它们）.

    重算而不是把账塞进 ``BlockForward``：前向是确定性的，而记录里已经留下了
    每个子层的输入与输出——反向拿它们重跑一遍，代价是几毫秒，
    换来的是记录保持“可以说清”的大小。
    """
    if forward.placement == NORM_PRE:
        _norm1_out, norm1_cache = layer_norm(
            forward.inputs, gamma=params.norm1_gamma, beta=params.norm1_beta
        )
        second_input = forward.residual1
        _norm2_out, norm2_cache = layer_norm(
            second_input, gamma=params.norm2_gamma, beta=params.norm2_beta
        )
        _branch2_out, ffn_cache = feed_forward(
            forward.norm2, params.ffn, activation=activation
        )
        return norm1_cache, norm2_cache, ffn_cache
    second_input = forward.norm1
    residual1 = forward.residual1
    _norm1_out, norm1_cache = layer_norm(
        residual1, gamma=params.norm1_gamma, beta=params.norm1_beta
    )
    output_pre = add_residual(second_input, forward.branch2, use_residual=forward.use_residual)
    _norm2_out, norm2_cache = layer_norm(
        output_pre, gamma=params.norm2_gamma, beta=params.norm2_beta
    )
    _branch2_out, ffn_cache = feed_forward(
        second_input, params.ffn, activation=activation
    )
    return norm1_cache, norm2_cache, ffn_cache


def encoder_block(
    params: BlockParameters,
    inputs: Matrix,
    attention: AttentionForward,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> BlockForward:
    """一个编码器块的前向（六个阶段）.

    ``attention`` 是一个**已经把前向算好的** day075 注意力账——这样设计有具体好处：
    “块不认识注意力内部”这一点让第一个子层可以被替换成任何逐行算子，
    而第 9 章的对照换的就是它（把注意力换成别的东西，块的其他部分一个字不改）。
    """
    if not isinstance(params, BlockParameters):
        raise ParameterError(f"params 必须是 BlockParameters，收到 {type(params).__name__}。")
    if not isinstance(attention, AttentionForward):
        raise ParameterError(
            f"attention 必须是 day075 的 AttentionForward，收到 {type(attention).__name__}："
            "本课把注意力那一层当**已算好的给定函数**（它的反向在 day075 已逐项验过）。"
        )
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    checked_inputs = validate_matrix(inputs, name="inputs")
    rows, columns = matrix_shape(checked_inputs)
    if columns != params.hidden:
        raise ShapeError(f"输入的列数 {columns} 与块的隐藏维 {params.hidden} 不一致。")
    if matrix_shape(attention.inputs) != matrix_shape(checked_inputs):
        raise ShapeError(
            f"注意力那一层的输入形状 {matrix_shape(attention.inputs)} 与本块的输入 "
            f"{matrix_shape(checked_inputs)} 不一致：块内的第一个子层作用在同一个流上。"
        )
    shape = BlockShape(hidden=columns, ffn=matrix_shape(params.ffn_w_in)[0], tokens=rows)
    branch1 = attention.output
    if resolved_placement == NORM_PRE:
        norm1, _cache1 = layer_norm(
            checked_inputs, gamma=params.norm1_gamma, beta=params.norm1_beta
        )
        residual1 = add_residual(checked_inputs, branch1, use_residual=use_residual)
        second_input = residual1
        norm2, _cache2 = layer_norm(
            second_input, gamma=params.norm2_gamma, beta=params.norm2_beta
        )
        branch2, _cache3 = feed_forward(norm2, params.ffn, activation=resolved_activation)
        output = add_residual(second_input, branch2, use_residual=use_residual)
    else:
        residual1 = add_residual(checked_inputs, branch1, use_residual=use_residual)
        norm1, _cache1 = layer_norm(
            residual1, gamma=params.norm1_gamma, beta=params.norm1_beta
        )
        second_input = norm1
        branch2, _cache3 = feed_forward(
            second_input, params.ffn, activation=resolved_activation
        )
        output_pre = add_residual(second_input, branch2, use_residual=use_residual)
        norm2, _cache2 = layer_norm(
            output_pre, gamma=params.norm2_gamma, beta=params.norm2_beta
        )
        output = norm2
    if matrix_shape(output) != matrix_shape(checked_inputs):
        raise ShapeError(
            f"块改变了形状：{matrix_shape(checked_inputs)} → {matrix_shape(output)}："
            "块必须**保形**，否则它没法堆叠。"
        )
    return BlockForward(
        shape=shape,
        placement=resolved_placement,
        use_residual=use_residual,
        inputs=checked_inputs,
        attention=attention,
        norm1=norm1,
        branch1=branch1,
        residual1=residual1,
        norm2=norm2,
        branch2=branch2,
        output=output,
        notes=(
            f"{resolved_placement}-LN，阶段顺序 {' → '.join(stage_order(resolved_placement))}",
            f"残差 {'开' if use_residual else '**关**'}；激活 {resolved_activation}",
        ),
    )


def encoder_block_backward(
    forward: BlockForward,
    params: BlockParameters,
    grad_output: Matrix,
    *,
    activation: str = ACTIVATION_RELU,
) -> BlockGradients:
    """一个编码器块的反向（九块：八块参数 + 输入）.

    pre-LN 的两行关键式子（**两项都是“漏掉不会报错”的**）：

    ```text
    dResidual1 = dOutput + dNorm2              ← 前一项就是残差那条 +1 的路
    dInputs    = dResidual1 + dNorm1_inputs    ← 后一项来自第一个 LN 的那条链
    ```

    post-LN 的第一站反过来：``output = norm2``，因此先把 LN 的梯度还原成
    ``dOutputPre``，再分成“残差那条路”与“分支那条路”两半。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.output):
        raise ShapeError(
            f"回传梯度 {matrix_shape(checked_grad)} 与块输出 "
            f"{matrix_shape(forward.output)} 形状不一致。"
        )
    resolved_activation = _checked_activation(activation)
    use_residual = forward.use_residual
    norm1_cache, norm2_cache, ffn_cache = _block_caches(forward, params, resolved_activation)
    if forward.placement == NORM_PRE:
        ffn_grads = feed_forward_backward(ffn_cache, checked_grad)
        # 第二个 LN 的输入是 residual1、它的上游梯度**只有** dBranch2（来自前馈那一条）
        norm2_grads = layer_norm_backward(norm2_cache, ffn_grads.grad_inputs)
        # 而 residual1 拿到的是**两条**：残差那条 +1 的路（dOutput）与 LN2 那条链
        d_residual1 = add_matrices(
            norm2_grads.grad_inputs,
            checked_grad if use_residual else zero_matrix(*matrix_shape(checked_grad)),
        )
        attn_grads = attention_backward(forward.attention, d_residual1)
        norm1_grads = layer_norm_backward(norm1_cache, attn_grads.grad_inputs)
        d_inputs = (
            add_matrices(d_residual1, norm1_grads.grad_inputs)
            if use_residual
            else norm1_grads.grad_inputs
        )
    else:
        norm2_grads = layer_norm_backward(norm2_cache, checked_grad)
        d_output_pre = norm2_grads.grad_inputs
        ffn_grads = feed_forward_backward(ffn_cache, d_output_pre)
        d_second_input = (
            add_matrices(d_output_pre, ffn_grads.grad_inputs)
            if use_residual
            else ffn_grads.grad_inputs
        )
        norm1_grads = layer_norm_backward(norm1_cache, d_second_input)
        d_residual1 = norm1_grads.grad_inputs
        attn_grads = attention_backward(forward.attention, d_residual1)
        d_inputs = (
            add_matrices(d_residual1, attn_grads.grad_inputs)
            if use_residual
            else attn_grads.grad_inputs
        )
    return BlockGradients(
        grad_norm1_gamma=norm1_grads.grad_gamma,
        grad_norm1_beta=norm1_grads.grad_beta,
        grad_ffn_w_in=ffn_grads.grad_w_in,
        grad_ffn_b_in=ffn_grads.grad_b_in,
        grad_ffn_w_out=ffn_grads.grad_w_out,
        grad_ffn_b_out=ffn_grads.grad_b_out,
        grad_norm2_gamma=norm2_grads.grad_gamma,
        grad_norm2_beta=norm2_grads.grad_beta,
        grad_inputs=d_inputs,
    )


# ---------------------------------------------------------------------- 交叉注意力


def cross_attention(
    params: CrossParameters,
    target_inputs: Matrix,
    source_inputs: Matrix,
    *,
    causal: bool = False,
) -> CrossForward:
    """交叉注意力：``Q`` 来自 target、``K/V`` 来自 source（**两路**）.

    ```text
    Q = target·W_qᵀ     (n_tgt, d)      K = source·W_kᵀ   V = source·W_vᵀ   (n_src, d)
    scores  = Q·Kᵀ/√d                  (n_tgt, n_src)     ← **长方形**
    weights = softmax_rows(scores)     每一行和为 1
    output  = (weights·V)·W_oᵀ         (n_tgt, d)
    ```

    ``causal=True`` **当场拒绝**（``AssemblyError``）：因果掩码是“不许看**未来**”，
    而交叉注意力面对的是一段**已经编码好的**源序列——
    它没有“未来”可言，加上它只会把源序列的后半段删掉。
    这个拒绝的理由写在了错误消息里，因为它的失效方式很坏（第 8 章量过）。
    """
    if not isinstance(params, CrossParameters):
        raise ParameterError(f"params 必须是 CrossParameters，收到 {type(params).__name__}。")
    if causal:
        raise AssemblyError(
            "交叉注意力不能被赋予因果掩码：因果性是**同一段序列内部**的性质"
            "（位置 i 不许看 i 之后），而交叉注意力的 K/V 来自另一路、已经编码完毕，"
            "它没有'未来'——加掩码只会把源序列的后半段删掉。"
            "注意这个错误在 n_tgt == n_src 时**不会**以形状错误的形式暴露出来，"
            "因此本包在这里直接拒绝。"
        )
    checked_target = validate_matrix(target_inputs, name="target_inputs")
    checked_source = validate_matrix(source_inputs, name="source_inputs")
    targets, target_width = matrix_shape(checked_target)
    sources, source_width = matrix_shape(checked_source)
    if matrix_shape(params.w_query)[1] != target_width:
        raise ShapeError(
            f"W_q 的列数 {matrix_shape(params.w_query)[1]} 与 target 的列数 "
            f"{target_width} 不一致。"
        )
    if matrix_shape(params.w_key)[1] != source_width:
        raise ShapeError(
            f"W_k 的列数 {matrix_shape(params.w_key)[1]} 与 source 的列数 "
            f"{source_width} 不一致。"
        )
    head_dim = matrix_shape(params.w_query)[0]
    shape = CrossShape(targets=targets, sources=sources, dimension=head_dim)
    queries = matmul(checked_target, transpose(params.w_query))
    keys = matmul(checked_source, transpose(params.w_key))
    values = matmul(checked_source, transpose(params.w_value))
    scale = 1.0 / math.sqrt(head_dim)
    scores = tuple(
        tuple(value * scale for value in row) for row in matmul(queries, transpose(keys))
    )
    weights = tuple(softmax(row) for row in scores)
    context = matmul(weights, values)
    output = matmul(context, transpose(params.w_output))
    return CrossForward(
        shape=shape,
        params=params,
        target_inputs=checked_target,
        source_inputs=checked_source,
        queries=queries,
        keys=keys,
        values=values,
        scores=scores,
        weights=weights,
        context=context,
        output=output,
        scale=scale,
        notes=(
            f"权重形状 {shape.weights_shape}（**长方形**：行数来自 target、列数来自 source）",
            "本层没有掩码：K/V 是另一路的全部位置",
        ),
    )


def cross_attention_backward(
    forward: CrossForward,
    grad_output: Matrix,
) -> CrossGradients:
    """交叉注意力的反向（六块：四个投影 + **两路各自的输入**）.

    ```text
    dW_o = dOutᵀ·context ；dContext = dOut·W_o
    dWeights = dContext·Vᵀ ；dV = weightsᵀ·dContext
    dScores ← softmax 逐行反向（与 day075 **同一个函数**）
    dRaw = dScores·scale ；dQ = dRaw·K ；dK = dRawᵀ·Q
    dW_q = dQᵀ·target ；dW_k = dKᵀ·source ；dW_v = dVᵀ·source
    dTarget = dQ·W_q                            ← **只有一条链**
    dSource = dK·W_k + dV·W_v                   ← **两条链之和**（少一条不报错）
    ```

    最后两行是交叉注意力与自注意力唯一的实质差别：
    **同一个 ``dRaw`` 分给两个来源**，而 source 那一侧拿到的是两条链。
    这也解释了形状上的不对称：``dTarget`` 与 target 同行数、``dSource`` 与 source 同行数。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.output):
        raise ShapeError(
            f"回传梯度 {matrix_shape(checked_grad)} 与输出 "
            f"{matrix_shape(forward.output)} 形状不一致。"
        )
    params = forward.params
    grad_w_output = matmul(transpose(checked_grad), forward.context)
    grad_context = matmul(checked_grad, params.w_output)
    grad_weights = matmul(grad_context, transpose(forward.values))
    grad_values = matmul(transpose(forward.weights), grad_context)
    grad_scores = tuple(
        softmax_backward_row(forward.weights[index], grad_weights[index])
        for index in range(matrix_shape(forward.weights)[0])
    )
    grad_raw = tuple(tuple(value * forward.scale for value in row) for row in grad_scores)
    grad_queries = matmul(grad_raw, forward.keys)
    grad_keys = matmul(transpose(grad_raw), forward.queries)
    grad_w_query = matmul(transpose(grad_queries), forward.target_inputs)
    grad_w_key = matmul(transpose(grad_keys), forward.source_inputs)
    grad_w_value = matmul(transpose(grad_values), forward.source_inputs)
    grad_target = matmul(grad_queries, params.w_query)
    grad_source = add_matrices(
        matmul(grad_keys, params.w_key), matmul(grad_values, params.w_value)
    )
    return CrossGradients(
        grad_w_query=grad_w_query,
        grad_w_key=grad_w_key,
        grad_w_value=grad_w_value,
        grad_w_output=grad_w_output,
        grad_target_inputs=grad_target,
        grad_source_inputs=grad_source,
    )


# ---------------------------------------------------------------------- 解码器块


def decoder_block(
    self_attention_forward: AttentionForward,
    cross_params: CrossParameters,
    gammas: tuple[Vector, Vector, Vector],
    betas: tuple[Vector, Vector, Vector],
    ffn: FFNWeights,
    decoder_inputs: Matrix,
    encoder_outputs: Matrix,
    *,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> tuple[Matrix, CrossForward, tuple[NormCache, ...], FFNCache]:
    """一个解码器块的前向（**九个阶段**）：因果自注意力 → 交叉注意力 → 前馈.

    ```text
    ① norm1 = LN(x)         ② branch1 = 因果自注意力(x)     ③ residual1 = x + branch1
    ④ norm2 = LN(residual1) ⑤ branch2 = 交叉注意力(·, encoder) ⑥ residual2 = residual1 + branch2
    ⑦ norm3 = LN(residual2) ⑧ branch3 = 前馈                ⑨ output   = residual2 + branch3
    ```

    解码器与编码器的差别只有一处：**它多了第二个子层，而那个子层的 K/V 来自另一路**。
    参数刻意分成三份：``cross_params``（作用于两路之间）、
    ``gammas``/``betas``（三个 LN，各一个向量）、``ffn``（只作用在解码器自己的流上）——
    而“哪一个参数属于哪一路”正是这一层最容易搞混的地方。

    两条**组装层面的拒绝**在这里生效（都是 ``AssemblyError``）：

    ```text
    自注意力不是因果的      → 解码器会在训练时直接看到要预测的下一个 token
    编码器输出与解码器同宽？ → 不同宽时交叉注意力根本没有定义（两个数各自都合法）
    ```
    """
    if not isinstance(self_attention_forward, AttentionForward):
        raise ParameterError(
            f"self_attention_forward 必须是 AttentionForward，收到 "
            f"{type(self_attention_forward).__name__}。"
        )
    if not self_attention_forward.causal:
        raise AssemblyError(
            "解码器块的自注意力**必须**是因果的（它不许看未来）："
            "收到的这一份 forward.causal=False，而一个非因果的自注意力"
            "会让解码器在训练时直接看到要预测的下一个 token。"
        )
    if len(gammas) != 3 or len(betas) != 3:
        raise AssemblyError(
            f"解码器块有三个 LN，因此需要三个 γ 与三个 β，收到 {len(gammas)} 与 "
            f"{len(betas)}：pre-LN 的每个子层各有自己的 LN。"
        )
    checked_decoder = validate_matrix(decoder_inputs, name="decoder_inputs")
    checked_encoder = validate_matrix(encoder_outputs, name="encoder_outputs")
    if matrix_shape(checked_encoder)[1] != matrix_shape(checked_decoder)[1]:
        raise AssemblyError(
            f"编码器输出的列数 {matrix_shape(checked_encoder)[1]} 与解码器的隐藏维 "
            f"{matrix_shape(checked_decoder)[1]} 不一致：交叉注意力要求 K/V 与 Q "
            "落在同一个空间里——两个数各自都合法，但**拼起来**不成立。"
        )
    resolved_activation = _checked_activation(activation)
    norm1, cache1 = layer_norm(checked_decoder, gamma=gammas[0], beta=betas[0])
    residual1 = add_residual(
        checked_decoder, self_attention_forward.output, use_residual=use_residual
    )
    norm2, cache2 = layer_norm(residual1, gamma=gammas[1], beta=betas[1])
    cross_forward = cross_attention(cross_params, norm2, checked_encoder)
    residual2 = add_residual(residual1, cross_forward.output, use_residual=use_residual)
    norm3, cache3 = layer_norm(residual2, gamma=gammas[2], beta=betas[2])
    output_pre, ffn_cache = feed_forward(norm3, ffn, activation=resolved_activation)
    output = add_residual(residual2, output_pre, use_residual=use_residual)
    return output, cross_forward, (cache1, cache2, cache3), ffn_cache


def decoder_block_backward(
    self_attention_forward: AttentionForward,
    cross_forward: CrossForward,
    caches: tuple[NormCache, ...],
    ffn_cache: FFNCache,
    grad_output: Matrix,
    *,
    use_residual: bool = True,
) -> DecoderGradients:
    """解码器块的反向：**九步倒着走**，而最后得到**两路**输入梯度.

    ```text
    ⑨ dResidual2 = dOut；dBranch3 = dOut
    ⑧ dBranch3 → ffn 反向 → dNorm3
    ⑦ dNorm3 → LN 反向 → 加到 dResidual2 上
    ⑥ dResidual1 = dResidual2；dCross = dResidual2
    ⑤ dCross → 交叉注意力反向 → dNorm2（target 那一路）与 **dEncoder**（source 那一路）
    ④ dNorm2 → LN 反向 → 加到 dResidual1 上
    ③ dX = dResidual1；dBranch1 = dResidual1
    ② dBranch1 → 注意力反向 → dNorm1
    ① dNorm1 → LN 反向 → 加到 dX 上
    ```

    三处 ``+`` 就是三条残差路（第 ⑦、④、① 步各一处），
    而 ``dEncoder`` 是这一层独有的输出：**编码器会收到梯度**。
    """
    if len(caches) != 3:
        raise AssemblyError(f"解码器块有三个 LN 的账，收到 {len(caches)} 份。")
    checked_grad = validate_matrix(grad_output, name="grad_output")
    cache1, cache2, cache3 = caches
    if matrix_shape(checked_grad) != matrix_shape(cache1.inputs):
        raise ShapeError(
            f"回传梯度 {matrix_shape(checked_grad)} 与解码器输入 "
            f"{matrix_shape(cache1.inputs)} 形状不一致。"
        )
    neutral = zero_matrix(*matrix_shape(checked_grad))
    # ⑨ output = residual2 + branch3
    d_residual2 = checked_grad if use_residual else neutral
    d_branch3 = checked_grad
    # ⑧ branch3 = ffn(norm3)
    ffn_grads = feed_forward_backward(ffn_cache, d_branch3)
    # ⑦ norm3 = LN(residual2)
    ln3_grads = layer_norm_backward(cache3, ffn_grads.grad_inputs)
    d_residual2 = add_matrices(d_residual2, ln3_grads.grad_inputs)
    # ⑥ residual2 = residual1 + cross_out
    d_residual1 = d_residual2 if use_residual else neutral
    d_cross = d_residual2
    # ⑤ 交叉注意力反向
    cross_grads = cross_attention_backward(cross_forward, d_cross)
    # ④ norm2 = LN(residual1)
    ln2_grads = layer_norm_backward(cache2, cross_grads.grad_target_inputs)
    d_residual1 = add_matrices(d_residual1, ln2_grads.grad_inputs)
    # ③ residual1 = decoder_inputs + branch1
    d_decoder = d_residual1 if use_residual else neutral
    d_branch1 = d_residual1
    # ② branch1 = 因果自注意力(decoder_inputs)
    self_grads = attention_backward(self_attention_forward, d_branch1)
    # ① norm1 = LN(decoder_inputs)
    ln1_grads = layer_norm_backward(cache1, self_grads.grad_inputs)
    d_decoder = add_matrices(d_decoder, ln1_grads.grad_inputs)
    return DecoderGradients(
        grad_decoder_inputs=d_decoder,
        grad_encoder_outputs=cross_grads.grad_source_inputs,
        grad_self_params=self_grads,
        grad_cross=cross_grads,
        grad_ffn=ffn_grads,
        grad_norm_gammas=(
            ln1_grads.grad_gamma,
            ln2_grads.grad_gamma,
            ln3_grads.grad_gamma,
        ),
        grad_norm_betas=(ln1_grads.grad_beta, ln2_grads.grad_beta, ln3_grads.grad_beta),
    )


__all__ = [
    "activate",
    "activation_backward",
    "add_matrices",
    "add_residual",
    "cross_attention",
    "cross_attention_backward",
    "decoder_block",
    "decoder_block_backward",
    "encoder_block",
    "encoder_block_backward",
    "feed_forward",
    "feed_forward_backward",
    "layer_norm",
    "layer_norm_backward",
    "stage_order",
    "zero_matrix",
]
