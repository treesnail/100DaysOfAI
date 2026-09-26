"""多头注意力：前向九步 + 反向九步（day076 / M7-D2）.

day075 交出一层可训练的自注意力；那一层里每一行只有**一份**混合系数。
今天把"一份"改成"``heads`` 份"，而改动只落在两处**新算术**上：

```text
前向（九步）                                    反向（九步，顺序恰好相反）
① Q/K/V = x·Wᵀ                                 ⑨ dW_o = dOutᵀ·merged ；dMerged = dOut·W_o
② 按行切成 heads 段（Q/K 用 d_k 的划分，V 用 d_v）  ⑧ 按列把 dMerged 切回 heads 段
③ 每头 raw_h = Q_h·K_hᵀ                          ⑦ 每头 dWeights_h = dCtx_h·V_hᵀ
④ 每头 scores_h = raw_h / √d_h                   ⑥ 每头 dV_h = Weights_hᵀ·dCtx_h
⑤ 每头过同一张掩码表                               ⑤ 每头 dScores_h ← softmax 反向 + 掩码置 0
⑥ 每头按行 softmax（每行 heads 个分布）             ④ 每头 dRaw_h = dScores_h / √d_h
⑦ 每头 ctx_h = Weights_h·V_h                     ③ 每头 dQ_h、dK_h
⑧ heads 段按列拼成 merged                        ② **按行拼回**融合的 dQ / dK / dV
⑨ output = merged·W_oᵀ                           ① 每头 dW_h = d*_hᵀ·x，**heads 条链相加**
```

## 四处"漏掉不会报错"的地方

day075 有三处（softmax 的雅可比、缩放系数、掩码位置的梯度），今天再加一处：

```text
① 每头的 softmax 雅可比   仍然是 dScores_i = s_i(g_i − 平均)，但现在是 heads 份
② 每头缩放分母是 √d_h     **不是 √d_k**；用错时打分整体偏小 √heads 倍（heads=4 时 2 倍）
③ 每头掩码位置置 0        同一张掩码要发给每一头
④ 分头之后必须**相加**     四个投影被所有头共用：漏掉一头不报错，
                          只会让那一头对应的参数更新偏小（"某一头学得慢"）
```

第 ④ 条是今天独有的，也是它最"像正确实现"的地方——
一个只回了第一头梯度的实现，形状完全正确、损失也真的在降。

## 一处**必须**诚实说明的地方：merge-then-project 恒等式是近似的

```text
matmul(merged, W_oᵀ)  ==  Σ_h matmul(ctx_h, (W_o^h)ᵀ)
```

这两个式子**在实数上严格相等**，在浮点下只到 ``1e-18`` 量级（本样本 6.94e-18）——
因为两种写法的**求和顺序不同**（前者一次遍历 d_v 列，后者先按头分组再相加），
而浮点加法不满足结合律。测试因此断言"最大差 <= 1e-12"，并把观测值写进证据：

```text
不是"这里有个 bug"，而是"两个都对的写法本来就差一点点"——
把它写成 >= 的断言会让某一次无关的重构（比如换一个求和顺序）变成一次假失败。
```

## 一条纪律：heads=1 必须**逐位**等于 day075

这一条在实现上是"顺带成立"的：当 ``heads=1`` 时，
``split_rows`` 返回的那一块**就是**整张矩阵（连续块的唯一划分），
于是每一步的算子、顺序、甚至循环都退化回 day075 的那七个阶段。
因此本包把它写成 ``==`` 而不是 ```approx``——
**一条"逐位相等"的断言比一条"误差很小"的断言难伪造得多。**
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.math_foundations.attention import masked_softmax_rows
from smart_research_agent.math_foundations.linalg import argmax, matmul, transpose
from smart_research_agent.math_foundations.probability import entropy
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.multi_head.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.multi_head.types import (
    HeadGradients,
    MultiHeadForward,
    MultiHeadShape,
)
from smart_research_agent.transformer_core.layers import (
    resolve_mask,
    softmax_backward_row,
)
from smart_research_agent.transformer_core.types import (
    AttentionParams,
    ParameterGradients,
    check_weights_are_a_distribution,
    project,
)


def head_scale(head_dim: int) -> float:
    """多头里每一头的缩放系数 ``1/√head_dim``（**不是** ``1/√d_k``）.

    这个函数在 day075 里叫 ``softmax_shape_scale``；今天重写一遍是因为
    **参数的含义变了**：day075 传进来的是整个 ``d_k``，
    今天传进来的是 ``d_k / heads``。两处实现的存在有一处风险——
    "某天有人改了一处"——因此 ``tests/test_multihead_gradients.py``
    里有一条断言把 ``head_scale(head_dim)`` 与
    ``transformer_core.types.softmax_shape_scale(head_dim)`` 逐位对上：
    **同一个输入必须给出同一个数**，剩下的差别只在调用点传什么。
    """
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise ParameterError(f"head_dim 必须是整数，收到 {head_dim!r}。")
    if head_dim < 1:
        raise ParameterError(
            f"head_dim 必须 >= 1，收到 {head_dim}："
            "零宽的子空间上没有点积，而'每头零维'会让打分矩阵变成一片 0。"
        )
    return 1.0 / math.sqrt(head_dim)


def _check_inputs(inputs: Matrix) -> Matrix:
    """校验输入：非空、每行等长、**没有全零行**.

    这一条与 day075 ``transformer_core.layers._check_inputs`` 是**故意的重复**：
    它是"不替调用方猜"那条纪律的落点，而两处若各写一半（比如这里允许全零行），
    同一批数据在两层上会得到不同的对待——一个改数据、一个改训练。
    测试里有一条断言把两者钉在一起（``test_both_layers_reject_the_same_zero_row``）。
    """
    checked = validate_matrix(inputs, name="inputs")
    for index, row in enumerate(checked):
        if all(value == 0.0 for value in row):
            raise NumericError(
                f"输入第 {index} 行全零：它会让这一行（在**每一头**上）打分全是 0、"
                "softmax 给出均匀分布，而均匀分布看起来像'模型还没学到'。"
                "分头不会改变这件事：heads 份均匀分布仍然是均匀分布。"
                "如果这是 padding，请显式裁剪掉或用掩码处理，不要交给这一层猜。"
            )
    return checked


def multi_head_attention(
    params: AttentionParams,
    inputs: Matrix,
    *,
    heads: int = 1,
    causal: bool = False,
    mask: tuple[tuple[bool, ...], ...] | None = None,
) -> MultiHeadForward:
    """一层多头自注意力的前向（返回**全部中间量**，因为反向要用）.

    九步见模块说明。四处细节值得单独指出：

    ```text
    ① 缩放作用在**每一头自己的**打分上（分母 √d_h）：这样反向只需乘回同一个数
    ② 掩码是同一张表发给每一头——因果性是**位置**的性质，不是某一头的偏好
    ③ 拼接（merge）只是 split 的逆：它不引入任何新的算术，
       因此"每一头看到了哪些维度"这件事完全由划分决定、可以被打印出来看
    ④ 输出投影只有**一次**（不在每一头各做一次）：
       merged 先拼好再乘 W_o，与"每头各投影再相加"在实数上等价（见模块说明）
    ```
    """
    if not isinstance(params, AttentionParams):
        raise ShapeError(f"params 必须是 AttentionParams，收到 {type(params).__name__}。")
    shape = MultiHeadShape(params.shape, heads)
    checked_inputs = _check_inputs(inputs)
    rows = len(checked_inputs)
    resolved_mask = resolve_mask(rows, causal=causal, mask=mask)

    queries = project(checked_inputs, params.w_query)
    keys = project(checked_inputs, params.w_key)
    values = project(checked_inputs, params.w_value)

    keys_partition = shape.keys_partition
    values_partition = shape.values_partition
    # **权重按行切、激活按列切**：W_q 的形状是 (d_k, d_in)，d_k 在它的行上；
    # 而 Q = x·W_qᵀ 的形状是 (n, d_k)，d_k 在它的列上。同一个维度、两张不同的表。
    head_queries = keys_partition.split_columns(queries)
    head_keys = keys_partition.split_columns(keys)
    head_values = values_partition.split_columns(values)

    scale = shape.scale
    head_raw: list[Matrix] = []
    head_scores: list[Matrix] = []
    head_weights: list[Matrix] = []
    head_contexts: list[Matrix] = []
    head_entropies: list[Vector] = []
    head_peaks: list[Vector] = []
    head_indices: list[tuple[int, ...]] = []

    for head in range(shape.heads):
        raw = matmul(head_queries[head], transpose(head_keys[head]))
        scores = tuple(tuple(value * scale for value in row) for row in raw)
        weights = masked_softmax_rows(scores, resolved_mask)
        check_weights_are_a_distribution(weights)
        context = matmul(weights, head_values[head])
        head_raw.append(raw)
        head_scores.append(scores)
        head_weights.append(weights)
        head_contexts.append(context)
        head_entropies.append(tuple(entropy(row) for row in weights))
        head_peaks.append(tuple(max(row) for row in weights))
        head_indices.append(tuple(argmax(row) for row in weights))

    merged_context = values_partition.merge_columns(head_contexts)
    output = project(merged_context, params.w_output)

    notes = [
        f"每头宽度 {shape.head_dim}（d_k={shape.attention.keys} ÷ heads={shape.heads}），"
        f"因此缩放系数是 1/√{shape.head_dim} = {scale:.6f}，"
        f"是单头 1/√d_k = {shape.single_head_scale:.6f} 的 {shape.scale_ratio:.4f} 倍",
        f"这一层一共产生 {shape.heads} × {rows} = {shape.heads * rows} 个条件分布"
        f"（单头只有 {rows} 个）：'每一行是一个分布'这句话在多头下不再成立",
        "掩码是同一张表发给每一头：因果性是位置的性质，不是某一头的偏好",
        "四个投影被所有头共用，因此反向时 heads 条链必须相加——"
        "漏掉一头不报错，只会让那一头的参数更新偏小",
    ]
    if causal:
        notes.append("因果掩码：每一头的上三角（j > i）恰好为 0.0——位置 i 看不到未来")
    return MultiHeadForward(
        params=params,
        shape=shape,
        inputs=checked_inputs,
        queries=queries,
        keys=keys,
        values=values,
        head_queries=tuple(head_queries),
        head_keys=tuple(head_keys),
        head_values=tuple(head_values),
        head_raw=tuple(head_raw),
        head_scores=tuple(head_scores),
        head_weights=tuple(head_weights),
        head_contexts=tuple(head_contexts),
        merged_context=merged_context,
        output=output,
        causal=causal,
        mask=resolved_mask,
        scale=scale,
        head_entropies=tuple(head_entropies),
        head_peaks=tuple(head_peaks),
        head_indices=tuple(head_indices),
        notes=tuple(notes),
    )


def head_parameter_gradients(
    forward: MultiHeadForward,
    grad_output: Matrix,
) -> tuple[HeadGradients, ...]:
    """**逐头**的反向：从"损失对输出的梯度"回到每一头的三块参数梯度与输入梯度.

    它是 :func:`multi_head_backward` 的实现主体，也单独对外导出——
    因为"哪一头在学"这个问题只有它答得了：

    ```text
    grad_w_output   不在这里：W_o 消费的是拼接结果，**不属于任何一头**
    grad_w_query    第 h 头占有的那 head_dim 行
    grad_w_key      同上
    grad_w_value    第 h 头占有的那 head_value 行
    grad_inputs     这一头传回输入的梯度（它要参与"输入梯度是 heads·3 条链之和"）
    ```

    三处容易写错的地方，各自都"不报错"：

    ```text
    ① 掩码位置的梯度：softmax 的反向已经给出 0（因为那里的权重是 0），
       这里仍然显式再置一次——多写一行的代价是零，依赖隐含性质的代价是"某天漏了"
    ② 缩放：dRaw_h = dScores_h · scale；漏掉则整体差 √heads 倍（heads=4 时 2 倍）
    ③ 输入的梯度：每一头有 q/k/v 三条链，heads 头加起来是 heads×3 条
    ```
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.output):
        raise ShapeError(
            f"输出梯度的形状 {matrix_shape(checked_grad)} 与前向输出的形状 "
            f"{matrix_shape(forward.output)} 不一致。"
        )
    params = forward.params
    shape = forward.shape
    keys_partition = shape.keys_partition
    values_partition = shape.values_partition
    query_blocks = keys_partition.split_rows(params.w_query)
    key_blocks = keys_partition.split_rows(params.w_key)
    value_blocks = values_partition.split_rows(params.w_value)

    # ⑨ 输出投影只做一次：dMerged = dOut · W_o（W_o 不分头）
    grad_merged = matmul(checked_grad, params.w_output)
    # ⑧ 把 dMerged 按列切回 heads 段（与 heads 的 context 一一对应）
    grad_head_contexts = values_partition.split_columns(grad_merged)

    results: list[HeadGradients] = []
    for head in range(shape.heads):
        grad_context = grad_head_contexts[head]
        weights = forward.head_weights[head]
        values = forward.head_values[head]

        # ⑦ 加权混合：ctx_h = weights_h · V_h
        grad_weights = matmul(grad_context, transpose(values))
        grad_values = matmul(transpose(weights), grad_context)

        # ⑥⑤ softmax 反向 + 掩码：被掩码的位置梯度必须是 0
        grad_scores = tuple(
            tuple(
                value if forward.mask[row][column] else 0.0
                for column, value in enumerate(
                    softmax_backward_row(weights[row], grad_weights[row])
                )
            )
            for row in range(forward.tokens)
        )

        # ④ 缩放：scores_h = raw_h · scale（scale = 1/√head_dim）
        grad_raw = tuple(tuple(value * forward.scale for value in row) for row in grad_scores)

        # ③ 打分：raw_h = Q_h·K_hᵀ
        grad_queries = matmul(grad_raw, forward.head_keys[head])
        grad_keys = matmul(transpose(grad_raw), forward.head_queries[head])

        # ② 投影：三块参数梯度都只是"这一头占有的那几行"
        grad_w_query = matmul(transpose(grad_queries), forward.inputs)
        grad_w_key = matmul(transpose(grad_keys), forward.inputs)
        grad_w_value = matmul(transpose(grad_values), forward.inputs)
        # ① 输入：这一头的三条链之和
        grad_inputs = _matrix_sum(
            matmul(grad_queries, query_blocks[head]),
            matmul(grad_keys, key_blocks[head]),
            matmul(grad_values, value_blocks[head]),
        )
        results.append(
            HeadGradients(
                head=head,
                grad_w_query=grad_w_query,
                grad_w_key=grad_w_key,
                grad_w_value=grad_w_value,
                grad_inputs=grad_inputs,
            )
        )
    return tuple(results)


def multi_head_backward(
    forward: MultiHeadForward,
    grad_output: Matrix,
) -> ParameterGradients:
    """多头反向：heads 条链各自走完，再**按行拼回**融合的那三块矩阵.

    返回值的形状与 day075 的 :class:`ParameterGradients` **完全一致**
    （四个参数矩阵 + 输入梯度），因此 day074 的参数压平与优化器一行都不用改：

    ```text
    dW_q = 按行拼 [dW_q^0 … dW_q^(H−1)]      （每一行只由一头贡献 → 没有求和歧义）
    dW_k = 同上
    dW_v = 同上（行数用 d_v 的划分）
    dW_o = dOutᵀ · merged                    （**它不分头**）
    dx   = Σ_h dx_h                           （heads 条链相加）
    ```

    ## 为什么"拼回"比"求和"更值得写出来

    四个投影被所有头共用，等价的说法是"第 h 头只拥有其中一段行"。
    既然每一行只属于一头，那么"融合梯度"就**恰好是 heads 份块按行拼接**——
    于是"某一行拿到的梯度是不是来自正确的那一头"变成一条可以断言的等式，
    而不是一次需要读代码确认的求和。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.output):
        raise ShapeError(
            f"输出梯度的形状 {matrix_shape(checked_grad)} 与前向输出的形状 "
            f"{matrix_shape(forward.output)} 不一致。"
        )
    shape = forward.shape
    heads = head_parameter_gradients(forward, checked_grad)
    # 权重梯度按**行**拼（d_k / d_v 是权重矩阵的行），与 heads=1 时逐位退化一致
    grad_w_query = shape.keys_partition.merge_rows([item.grad_w_query for item in heads])
    grad_w_key = shape.keys_partition.merge_rows([item.grad_w_key for item in heads])
    grad_w_value = shape.values_partition.merge_rows([item.grad_w_value for item in heads])
    grad_w_output = matmul(transpose(checked_grad), forward.merged_context)
    grad_inputs = _matrix_sum(*(item.grad_inputs for item in heads))
    return ParameterGradients(
        grad_w_query=grad_w_query,
        grad_w_key=grad_w_key,
        grad_w_value=grad_w_value,
        grad_w_output=grad_w_output,
        grad_inputs=grad_inputs,
    )


def project_heads_separately(forward: MultiHeadForward) -> Matrix:
    """``merge → project`` 的另一种写法：每一头**各自**投影，再把结果相加.

    两个式子在实数上严格相等：

    ```text
    merged = [ctx_0 | ctx_1 | … ]      W_o = [W_o^0 | W_o^1 | … ]（按列切）
    merged · W_oᵀ = Σ_h ctx_h · (W_o^h)ᵀ
    ```

    这条恒等式值得单独写成一个函数，因为它把"输出投影**不属于任何一头**"
    这句话变成了一个可核对的事实：``W_o`` 的列被切成 heads 份，
    第 h 份只作用于第 h 头的 context。

    **浮点上它是近似的**：两种写法的求和顺序不同（一个遍历 d_v 列，
    一个先按头分组），而浮点加法不满足结合律——实测差在 ``1e-18`` 量级。
    测试因此断言 ``<= 1e-12`` 并且把观测值写进证据：
    "两个都对的写法本来就差一点点"不该被写成一次假失败。
    """
    values_partition = forward.shape.values_partition
    output_blocks = values_partition.split_columns(forward.params.w_output)
    parts = tuple(
        project(forward.head_contexts[head], output_blocks[head])
        for head in range(forward.shape.heads)
    )
    return _matrix_sum(*parts)


def head_value_of(
    forward: MultiHeadForward,
    head: int,
    position: int,
) -> Vector:
    """第 ``head`` 头在第 ``position`` 个位置上的 value 向量（宽度 ``head_value``）.

    它是"这一头的可达集合"的原材料：如果第 h 头整行都盯着位置 j，
    那么它交给 ``W_o`` 的就是这一条向量（见 ``reachability.py``）。
    """
    if isinstance(head, bool) or not isinstance(head, int):
        raise ParameterError(f"头号必须是整数，收到 {head!r}。")
    if not 0 <= head < forward.heads:
        raise ParameterError(f"头号 {head} 落在 [0, {forward.heads}) 之外。")
    if not 0 <= position < forward.tokens:
        raise ParameterError(f"位置 {position} 落在 [0, {forward.tokens}) 之外。")
    return forward.head_values[head][position]


def _matrix_sum(*matrices: Matrix) -> Matrix:
    """逐元素相加（形状必须一致；**至少一块**）.

    多头里只有两处用它：输入梯度（heads 条链）与"逐头投影再相加"。
    两处都**至少有一块**——零块会让"空集合的和"变成一个需要约定 0 的边界，
    而那个约定没有任何用处（heads >= 1 恒成立）。
    """
    if not matrices:
        raise ShapeError(
            "至少要有一块矩阵：零块的和需要一条'空集合 = 全零'的约定，"
            "而 heads >= 1 让这条约定永远用不上。"
        )
    checked = tuple(validate_matrix(matrix, name="matrix") for matrix in matrices)
    rows, columns = matrix_shape(checked[0])
    for index, matrix in enumerate(checked):
        if matrix_shape(matrix) != (rows, columns):
            raise ShapeError(
                f"第 {index} 块矩阵形状 {matrix_shape(matrix)} 与第一块 "
                f"{(rows, columns)} 不一致。"
            )
    return tuple(
        tuple(
            math.fsum(matrix[row][column] for matrix in checked) for column in range(columns)
        )
        for row in range(rows)
    )


def head_sequence(count: Sequence[int] | int) -> tuple[int, ...]:
    """把头号规范成一个元组：给一个整数就展开成 ``0..count−1``.

    报告与演示里反复要"对每一头做一件事"，而两种写法（传头数还是传显式头号列表）
    都会出现。统一在这里之后，调用点只需要一种形状——
    这与 day075 的 ``_resolve_trainable`` 是同一种"把可变参数规范成一种形状"的手法。
    """
    if isinstance(count, int) and not isinstance(count, bool):
        if count < 1:
            raise ParameterError(f"头数必须 >= 1，收到 {count}。")
        return tuple(range(count))
    checked: list[int] = []
    for head in count:
        if isinstance(head, bool) or not isinstance(head, int):
            raise ParameterError(f"头号必须是整数，收到 {head!r}。")
        if head < 0:
            raise ParameterError(f"头号必须 >= 0，收到 {head}。")
        if head in checked:
            raise ParameterError(
                f"头号 {head} 重复：重复不会让它'更重要'，只会让读数难以解释。"
            )
        checked.append(head)
    if not checked:
        raise ParameterError("头号列表不能为空。")
    return tuple(checked)


__all__ = [
    "head_parameter_gradients",
    "head_scale",
    "head_sequence",
    "head_value_of",
    "multi_head_attention",
    "multi_head_backward",
    "project_heads_separately",
]
