"""HF 的注意力：**分头发生在投影内部**（day085 / M7-D9）.

day079 与 day083 都记下过同一条**接口边界**：

```text
"一个块只支持一种掩码"——因此 day079 的 decoder_block 要求自注意力是因果的，
而 day082 的 resolve_mask 拒绝"因果 + 显式掩码"同时给。
```

Hugging Face 的注意力并不接受这条边界，而它绕开的办法非常具体：

```python
# modeling_gpt2.py（语义等价的重写，不是逐行抄写）
qkv   = c_attn(hidden)                       # (n, 3·hidden)
query, key, value = qkv.split(hidden, dim=-1) # 三个 (n, hidden)
query = query.view(n, heads, head_dim).transpose(0, 1)    # ← 就在这里
key   = key.view(n, heads, head_dim).transpose(0, 1)
value = value.view(n, heads, head_dim).transpose(0, 1)
attn  = attn_weights = query @ key.transpose(-1, -2)
attn  = attn * scale                          # scale = 1/√head_dim
attn  = attn + bias                           # ← 加性掩码
attn  = softmax(attn, dim=-1)
context = attn @ value
context = context.transpose(0, 1).reshape(n, hidden)
output  = c_proj(context)
```

**"分头"只是两次形状重排**（`view` + `transpose`），它没有换块类型、没有换掩码口径、
没有换缩放——因此同一个 `GPT2Attention` 既能跑 `heads=12` 又能跑因果掩码。
这就是 day083 说的"接口事实"在源码里的样子。

## 三条本模块要钉住的判据

```text
① 融合与分离等价   fused = concat(W_q, W_k, W_v) 之后按 hidden 切回去，
                 必须与三次独立投影**逐位相同**（切法错了不会报错）
② 加性掩码与显式掩码等价   两者算出的权重**逐位相同**（理由见 :func:`additive_bias_of`）
③ 分头与拼回是恒等   split → merge 必须逐位还原（分头只是记账）
```

## 一处刻意近似

本包与 day076 的 `multi_head_attention` 是**两份独立实现**：一处乘 `1/√head_dim`、
一处除 `√head_dim`，两处的求和顺序也不必相同。因此那一条对账是**容差判据**
（`SINGLE_HEAD_TOLERANCE = 1e-12`），而不是逐位判据——理由与 day080 的
"参数差值必须被逐项解释"同源：**两份实现之间的差异要么被消灭，要么被写下来。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.hf_source.errors import AssemblyError, ParameterError, ShapeError
from smart_research_agent.hf_source.types import SourceShape
from smart_research_agent.math_foundations.linalg import matmul, transpose
from smart_research_agent.math_foundations.types import Matrix, validate_matrix
from smart_research_agent.transformer_core.types import AttentionParams

#: 加性掩码用的"负无穷替身".
#:
#: HF 在 fp16 下用 ``torch.finfo(dtype).min``、在 fp32/fp64 下用 ``-inf``；
#: 本包用 ``-1e9``，而这个数**恰好等价**——理由见 :func:`additive_bias_of`。
BIAS_FLOOR = 1e9

#: 与 day076 对账时的容差（两份独立实现之间的"刻意近似"）.
SINGLE_HEAD_TOLERANCE = 1e-12


def hf_scale(head_dim: int) -> float:
    """HF 的缩放系数 ``1/√head_dim``（**不是** ``1/√hidden``）.

    这个量与 day076 的 ``head_scale``、day075 的 ``softmax_shape_scale``、
    day073 的 ``scaling_factor`` 是同一个量的四个名字——它由 `head_dim` **派生**，
    因此可以被断言（测试里有一条把四者逐位对上）。
    """
    if head_dim <= 0:
        raise ParameterError(f"head_dim 必须为正，收到 {head_dim}。")
    return 1.0 / math.sqrt(float(head_dim))


def resolve_heads(shape: SourceShape) -> int:
    """校验"隐藏维必须能被头数整除"（不能整除时 HF 会在**前向**才炸）."""
    if shape.heads <= 0:
        raise ParameterError(f"heads 必须为正，收到 {shape.heads}。")
    if shape.hidden % shape.heads != 0:
        raise ShapeError(
            f"hidden={shape.hidden} 不能被 heads={shape.heads} 整除。"
            "HF 在这种情况下会在 reshape 时才报错（那时离'写错一行'已经很远），"
            "本包把它挡在入口。"
        )
    return shape.heads


def fused_weight(
    w_query: Matrix,
    w_key: Matrix,
    w_value: Matrix,
) -> Matrix:
    """把三个投影**按行拼**成融合投影 ``c_attn.weight``（形状 ``(3·hidden, hidden)``）.

    拼的顺序就是 HF 切开的顺序：前三块依次是 q / k / v。反过来的话——
    `split` 之后拿到的 q 其实是 v——**不会报错**，只会让注意力的三个角色互换。
    """
    query = validate_matrix(w_query, name="w_query")
    key = validate_matrix(w_key, name="w_key")
    value = validate_matrix(w_value, name="w_value")
    if len(query) != len(key) or len(query) != len(value):
        raise AssemblyError(
            f"三个投影的行数必须相同（都是 d_out），收到 {len(query)} / {len(key)} / {len(value)}。"
        )
    if query and key and len(query[0]) != len(key[0]):
        raise AssemblyError("三个投影的列数必须相同（都是 d_in）。")
    return tuple(query) + tuple(key) + tuple(value)


def split_fused(fused: Matrix, hidden: int) -> tuple[Matrix, Matrix, Matrix]:
    """把 ``(3·hidden, d_in)`` 的融合权重按**行**切成三块 ``(hidden, d_in)``.

    这是"切法错了不会报错"那一处的守卫：三块永远凑得出来，
    因此本包在这里校验"总行数恰好是 3·hidden"——**多一行或少了都不接受**。
    """
    checked = validate_matrix(fused, name="fused")
    if hidden <= 0:
        raise ParameterError(f"hidden 必须为正，收到 {hidden}。")
    if len(checked) != 3 * hidden:
        raise ShapeError(
            f"融合投影的行数是 {len(checked)}，但 3·hidden = {3 * hidden}。"
            "HF 的 c_attn 一次算出 3·hidden 维再 split(hidden, dim=-1)——"
            "行数对不上时切开的三块仍然凑得出来，只是其中一块会混进别人的维度。"
        )
    return checked[:hidden], checked[hidden : 2 * hidden], checked[2 * hidden :]


def project(inputs: Matrix, weight: Matrix) -> Matrix:
    """线性投影 ``y = x·Wᵀ``（与 day075 起沿用的**唯一口径**一致）."""
    return matmul(inputs, transpose(weight))


def add_bias(inputs: Matrix, bias: tuple[float, ...] | None) -> Matrix:
    """逐行加一个偏置向量（``bias=None`` 时是恒等）."""
    checked = validate_matrix(inputs, name="inputs")
    if bias is None:
        return checked
    if len(bias) != len(checked[0]):
        raise ShapeError(
            f"偏置长度 {len(bias)} 与宽度 {len(checked[0])} 不一致："
            "HF 里偏置是**逐维**的（形状 (hidden,)），广播成 (n, hidden)。"
        )
    return tuple(tuple(value + offset for value, offset in zip(row, bias)) for row in checked)


def split_heads(matrix: Matrix, heads: int) -> tuple[Matrix, ...]:
    """``view(n, heads, head_dim).transpose(0, 1)``——**分头发生在投影内部**.

    等价于"把每一行按 ``head_dim`` 一段段切开"，然后按头重新分组。
    切出来的头按段号排列，因此 :func:`merge_heads` 能逐位还原。
    """
    checked = validate_matrix(matrix, name="matrix")
    if heads <= 0:
        raise ParameterError(f"heads 必须为正，收到 {heads}。")
    width = len(checked[0])
    if width % heads != 0:
        raise ShapeError(
            f"宽度 {width} 不能被 heads={heads} 整除："
            "HF 的 view(n, heads, head_dim) 在最后一维不整除时会抛形状错误，"
            "而本包把它挡在这里（消息里带上两个数，比一句 reshape failed 有用）。"
        )
    head_dim = width // heads
    blocks: list[Matrix] = []
    for index in range(heads):
        start = index * head_dim
        blocks.append(tuple(row[start : start + head_dim] for row in checked))
    return tuple(blocks)


def merge_heads(heads: tuple[Matrix, ...]) -> Matrix:
    """``transpose(0, 1).reshape(n, hidden)``——把若干头**逐位**拼回.

    拼接顺序与 :func:`split_heads` 的切分顺序相同，因此 `merge(split(x)) == x`
    是一条**逐位**性质（不是"形状一致"性质）。
    """
    if not heads:
        raise ShapeError("heads 为空：一次多头注意力至少要有一头。")
    rows = len(heads[0])
    width = len(heads[0][0]) if rows else 0
    if width == 0 or rows == 0:
        raise ShapeError("头的行数或宽度为 0：拼回去只会得到一个空矩阵。")
    for index, head in enumerate(heads):
        if len(head) != rows or len(head[0]) != width:
            raise ShapeError(
                f"第 {index} 头的形状与本组其它头不一致："
                "拼回时形状不一致会被 zip 静默截断（少拼几列），而结果看起来是一组正常的数。"
            )
    return tuple(
        tuple(value for head in heads for value in head[row]) for row in range(rows)
    )


def checked_mask(mask: tuple[tuple[bool, ...], ...]) -> tuple[tuple[bool, ...], ...]:
    """校验一张显式掩码的**结构**（它是布尔的表，因此不走数值校验器）.

    前九天的数值校验器（``math_foundations.types.validate_matrix``）要求元素是
    有限实数——``True`` / ``False`` 会被它正确地拒绝。掩码因此需要自己的入口，
    而这件事本身就是一条事实：**掩码是"允许集合"，不是"打分"**，
    把两者塞进同一个校验器会得到一个"布尔掩码不能用"的荒唐结果。
    """
    if isinstance(mask, (str, bytes)) or not hasattr(mask, "__iter__"):
        raise ShapeError("掩码必须是若干行（True / False 的序列），收到一个不可迭代的对象。")
    rows = tuple(mask)
    if not rows:
        raise ShapeError("掩码没有任何行：空掩码无法描述'允许哪些位置'。")
    width = len(rows[0])
    if width == 0:
        raise ShapeError("掩码的每一行为空：没有任何位置可允许。")
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ShapeError(
                f"掩码第 {index} 行的长度 {len(row)} 与第 0 行的 {width} 不一致："
                "长度不齐会被 zip 静默截断（少看几个位置），而形状看起来是合法的。"
            )
        for column, value in enumerate(row):
            if not isinstance(value, bool):
                raise ShapeError(
                    f"掩码的 (row={index}, column={column}) 是 {value!r}，不是布尔值。"
                    "掩码回答的是'允许 / 不允许'，把它写成 0.0 / -1e9 会让"
                    "'一张掩码'与'一张已经译好的偏置'变成两种不同的东西。"
                )
    return rows


def additive_bias_of(
    mask: tuple[tuple[bool, ...], ...],
    *,
    floor: float = BIAS_FLOOR,
) -> Matrix:
    """把一张**显式掩码**翻译成**加性偏置**（``0.0`` 或 ``-floor``）.

    ## 为什么 ``-1e9`` 与 ``-inf`` 在这里等价（这是本模块最想讲清的一处）

    HF 的 softmax 前会先减掉**本行的最大值** ``m``。被挡住的格子里
    ``v = s − 1e9 ≤ m − 1e9``，于是 ``v − m ≤ −1e9``，而

    ```text
    math.exp(-1e9) == 0.0        ← 直接下溢到精确的 0，
    ```

    因此被挡格子的**权重恰好是 0.0**，与"显式掩码 + 只对允许位置归一化"**逐位相同**。
    这条等价不是"数值上差不多"——它是 `exp` 下溢带来的**精确**结果，
    而它成立的前提有两条：

    ```text
    ① 本行至少有一个允许位置（否则 m = -1e9 + s，减完之后整行都是 0/0）
    ② 打分的量级远小于 floor（本包直接校验 score 的绝对值不超过 floor/2）
    ```
    """
    checked = checked_mask(mask)
    allowed = sum(1 for row in checked for value in row if value)
    if allowed == 0:
        raise AssemblyError(
            "掩码把**所有**位置都挡掉了：softmax 的分母会是 0。"
            "HF 在 top_p 里用'至少保留一个 token'来避免这类空核，"
            "而注意力掩码这一侧没有这种兜底——因此本包在这里拒绝。"
        )
    return tuple(
        tuple(0.0 if value else -float(floor) for value in row) for row in checked
    )


def causal_bias(count: int, *, floor: float = BIAS_FLOOR) -> Matrix:
    """因果加性偏置：上三角（``j > i``）为 ``-floor``，其余为 ``0.0``.

    注意它与 day073 的 ``causal_mask`` 是同一张表的两副面孔：
    这里是"加什么"，那里是"允许什么"。两者必须**互为补集**——
    行列搞反时形状完全合法，只是模型开始看未来。
    """
    if count <= 0:
        raise ParameterError(f"序列长度必须为正，收到 {count}。")
    return tuple(
        tuple(0.0 if column <= row else -float(floor) for column in range(count))
        for row in range(count)
    )


def check_score_magnitude(scores: Matrix, *, floor: float = BIAS_FLOOR) -> None:
    """校验打分**远离** ``-floor``（这是"加性掩码 = 显式掩码"的第二条前提）.

    如果某个合法位置的打分本身就接近 ``-1e9``，那么减掉行最大值之后，
    被挡格子与合法格子会落在同一个下溢区间里——那时"恰好 0.0"就不再成立。
    现实里打分是 ``O(1)``，但"现实里总是对的"不是一条判据。
    """
    ceiling = float(floor) / 2.0
    for row in scores:
        for value in row:
            if not math.isfinite(value):
                raise AssemblyError(
                    "打分里出现了非有限数：HF 会在 softmax 之后得到 nan，"
                    "而一个 nan 会顺着'平均熵'污染整张表（day073 第七章的同一条）。"
                )
            if abs(value) >= ceiling:
                raise AssemblyError(
                    f"打分 {value!r} 的绝对值不小于 floor/2 = {ceiling:g}："
                    "这时'加性掩码'与'显式掩码'不再逐位等价"
                    "（被挡格子与合法格子会挤进同一个下溢区间）。"
                )


def checked_scores(scores: object) -> Matrix:
    """校验打分的**结构与有限性**，并归入本包自己的失败族.

    理由与 :func:`generation.checked_logits` 相同：本包的入口要用**本包的名字**报错，
    否则调用方 ``except hf_source.errors.AssemblyError`` 兜不住一个"其实同族"的错误。
    """
    if isinstance(scores, (str, bytes)) or not hasattr(scores, "__iter__"):
        raise ShapeError(f"打分必须是若干行，收到 {type(scores).__name__}。")
    rows = tuple(scores)  # type: ignore[arg-type]
    if not rows:
        raise ShapeError("打分没有任何行。")
    width = len(rows[0])
    if width == 0:
        raise ShapeError("打分的每一行为空。")
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ShapeError(f"打分第 {index} 行的长度与第 0 行不一致（zip 会静默截断）。")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise AssemblyError(
                    f"打分里出现了不可用的数（收到 {value!r}）：softmax 之后会得到 nan，"
                    "而一个 nan 会顺着'平均熵'污染整张表（day073 第七章的同一条）。"
                )
    return rows


def row_softmax(scores: Matrix, bias: Matrix | None = None) -> Matrix:
    """带加性偏置的**逐行稳定 softmax**（先减本行最大值，求和用 ``math.fsum``）.

    这三步的顺序就是 HF 的顺序（``attn + bias`` → 减最大值 → exp → 归一化），
    因此它与 day073 的 ``masked_softmax_rows`` 在"掩码一致"时结果逐位相同。
    """
    checked = checked_scores(scores)
    width = len(checked[0])
    if bias is not None and len(bias) != len(checked):
        raise ShapeError(
            f"偏置有 {len(bias)} 行、打分有 {len(checked)} 行："
            "行数不一致时按行广播会静默地把偏置错位一整行。"
        )
    result: list[tuple[float, ...]] = []
    for index, row in enumerate(checked):
        biased = row if bias is None else tuple(value + b for value, b in zip(row, bias[index]))
        if len(biased) != width:
            raise ShapeError("偏置列数与打分列数不一致。")
        peak = max(biased)
        shifted = tuple(math.exp(value - peak) for value in biased)
        total = math.fsum(shifted)
        result.append(tuple(value / total for value in shifted))
    return tuple(result)


@dataclass(frozen=True)
class HfAttentionForward:
    """一次注意力的全部中间量（**只读**：本课不做反向）."""

    inputs: Matrix
    queries: Matrix
    keys: Matrix
    values: Matrix
    head_queries: tuple[Matrix, ...]
    head_keys: tuple[Matrix, ...]
    head_values: tuple[Matrix, ...]
    raw_scores: tuple[Matrix, ...]
    scores: tuple[Matrix, ...]
    weights: tuple[Matrix, ...]
    contexts: tuple[Matrix, ...]
    merged: Matrix
    output: Matrix
    bias: Matrix | None
    causal: bool
    heads: int

    @property
    def depth(self) -> int:
        """头数（读数的别名，方便报告里直接写 ``forward.depth``）."""
        return self.heads

    def average_weights(self) -> Matrix:
        """把各头的权重**逐位平均**——只在 heads 相同、掩码相同时才有意义.

        它存在的理由是"报告里要一张表"：热力图、熵、峰值都只需要一张。
        而它**不能**用来做跨掩码的比较（day083 第 5.2 节那条纪律）。
        """
        total = len(self.weights)
        rows = len(self.weights[0])
        width = len(self.weights[0][0])
        return tuple(
            tuple(
                math.fsum(head[row][column] for head in self.weights) / total
                for column in range(width)
            )
            for row in range(rows)
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**权重本体）."""
        return {
            "tokens": len(self.inputs),
            "hidden": len(self.inputs[0]),
            "heads": self.heads,
            "head_dim": len(self.head_queries[0][0]),
            "causal": self.causal,
            "has_explicit_bias": self.bias is not None,
        }


def hf_attention(
    params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    causal: bool = False,
    mask: tuple[tuple[bool, ...], ...] | None = None,
) -> HfAttentionForward:
    """一次 HF 语义的自注意力前向（十个阶段，见 :data:`types.ATTENTION_STAGES`）.

    ``params`` 直接复用 day075 的 :class:`~smart_research_agent.transformer_core.types.AttentionParams`
    （四个投影）——本课**没有**新造一套参数形状，因为 HF 的四个投影与 day075 的四个投影
    是同一个东西（`hidden → hidden` 的四个方阵）。

    ``causal=True`` 与显式 ``mask`` **不能同时给**：两者都是"允许集合"的来源，
    同时给时"谁生效"取决于实现顺序——那正是 day079/day082 反复记下的那类陷阱。
    """
    if causal and mask is not None:
        raise AssemblyError(
            "causal=True 与显式 mask 同时给了：两者都描述'允许集合'，"
            "同时给会让'谁生效'取决于实现顺序，而两种顺序的输出形状完全一样。"
        )
    resolved_heads = resolve_heads(shape)
    checked = validate_matrix(inputs, name="inputs")
    if len(checked) != shape.tokens:
        raise ShapeError(f"输入行数 {len(checked)} 与 shape.tokens {shape.tokens} 不一致。")
    if len(checked[0]) != shape.hidden:
        raise ShapeError(f"输入宽度 {len(checked[0])} 与 shape.hidden {shape.hidden} 不一致。")

    queries = project(checked, params.w_query)
    keys = project(checked, params.w_key)
    values = project(checked, params.w_value)
    head_queries = split_heads(queries, resolved_heads)
    head_keys = split_heads(keys, resolved_heads)
    head_values = split_heads(values, resolved_heads)

    bias_rows: Matrix | None
    if causal:
        bias_rows = causal_bias(shape.tokens)
    elif mask is not None:
        bias_rows = additive_bias_of(mask)
    else:
        bias_rows = None

    scale = hf_scale(shape.head_dim)
    raw_scores: list[Matrix] = []
    scaled_scores: list[Matrix] = []
    weights: list[Matrix] = []
    contexts: list[Matrix] = []
    for index in range(resolved_heads):
        raw = matmul(head_queries[index], transpose(head_keys[index]))
        scaled = tuple(tuple(value * scale for value in row) for row in raw)
        check_score_magnitude(scaled)
        head_weights = row_softmax(scaled, bias_rows)
        context = matmul(head_weights, head_values[index])
        raw_scores.append(raw)
        scaled_scores.append(scaled)
        weights.append(head_weights)
        contexts.append(context)

    merged = merge_heads(tuple(contexts))
    output = project(merged, params.w_output)
    return HfAttentionForward(
        inputs=checked,
        queries=queries,
        keys=keys,
        values=values,
        head_queries=head_queries,
        head_keys=head_keys,
        head_values=head_values,
        raw_scores=tuple(raw_scores),
        scores=tuple(scaled_scores),
        weights=tuple(weights),
        contexts=tuple(contexts),
        merged=merged,
        output=output,
        bias=bias_rows,
        causal=causal,
        heads=resolved_heads,
    )


__all__ = [
    "BIAS_FLOOR",
    "SINGLE_HEAD_TOLERANCE",
    "HfAttentionForward",
    "add_bias",
    "additive_bias_of",
    "causal_bias",
    "check_score_magnitude",
    "checked_mask",
    "checked_scores",
    "fused_weight",
    "hf_attention",
    "hf_scale",
    "merge_heads",
    "project",
    "resolve_heads",
    "row_softmax",
    "split_fused",
    "split_heads",
]
