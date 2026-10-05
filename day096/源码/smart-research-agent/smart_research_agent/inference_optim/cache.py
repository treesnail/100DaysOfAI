"""``cache``：别再重算已经算过的 K/V（day087 / M7-D11）.

自回归生成的第 t 步只需要**一个新位置**的 hidden state，而它的注意力要看
前面 t 个位置的 K/V。不缓存时，每步都要把前面 t 个位置**重算一遍**——
总计 O(T²) 次投影；缓存之后每层每步只投影**一行**，总计 O(T) 次。

```text
prefill     把提示词一次算完（**它是一次普通的整段前向**，只是顺手把 K/V 存下来）
decode      每次一个 token：每层投影一行、注意力读缓存、把新的 K/V 追加进去
```

## 缓存里存的**不是** hidden state，而是投影之后的 K/V

```python
keys[index]   = project(ln_1(x), w_key)      # (T, hidden)，**已投影**
values[index] = project(ln_1(x), w_value)
```

存投影结果而不是归一化之后的输入，是这一课最省事的一处收益：
每步只投影**新的一行**（`(1, h) × (h, h)`），而不是把 T 行再投影一遍。
它也带来第二条纪律：**归一化之后才投影**（pre-LN 块里注意力吃的是 `ln_1(x)`）。
在归一化之前缓存，形状一模一样、数值全变。

## 一条必须写下来的比较：缓存的 decode 与"整段重算"**逐位**相同

本模块的算术与 day085 的 ``hf_attention`` 逐字相同（先投影、再分头、再缩放、
再 softmax、再混合、再拼回、最后输出投影），因此"用缓存的最后一步"
与"把整段重跑一遍的最后一行"必须**逐位相等**——不是"数值相近"。
这条等式是"我的缓存接对了"的最强证据：接错一侧（在归一化之前取、或者把
K 与 V 弄反）时形状完全合法，而数值全变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.hf_integration.forward import (
    ModelWeights,
    embed_tokens,
    run_one_block,
)
from smart_research_agent.hf_integration.types import ModelCard
from smart_research_agent.hf_source.attention import (
    hf_scale,
    merge_heads,
    row_softmax,
    split_heads,
)
from smart_research_agent.inference_optim.errors import (
    AssemblyError,
    CacheError,
    ShapeError,
)
from smart_research_agent.inference_optim.types import (
    DECODE_TOKENS_PER_STEP,
    LEVEL_FP32,
    STAGE_DECODE,
    STAGE_PREFILL,
    CacheLayer,
    CacheState,
    StepRow,
    cache_bytes,
    element_bytes,
)
from smart_research_agent.math_foundations.linalg import matmul, transpose
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.transformer_core.types import AttentionParams, project

#: 默认容量（位置表长度就是模型的硬上界，因此它是天然的那个默认值）。
DEFAULT_CAPACITY_RATIO = 1.0


@dataclass
class KVCache:
    """一个可变的 KV 缓存：每层两张表，行数 = 已处理的 token 数.

    它是本课程里**少数几个可变的物件**（其余都是 frozen dataclass），
    因为"缓存"这件事的本质就是"一个会长的东西"。可变的代价是"状态从哪来"变得不明显，
    因此本类把三件事写死成方法：``append``（唯一的写入路径）、``record``（唯一的只读出口）、
    ``byte_report``（唯一的字节账）。
    """

    hidden: int
    heads: int
    capacity: int
    level: str = LEVEL_FP32
    keys: list[Matrix] = field(default_factory=list)
    values: list[Matrix] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise CacheError(f"容量不能为负，收到 {self.capacity}。")
        if self.hidden < 1 or self.heads < 1:
            raise CacheError(f"hidden 与 heads 必须为正，收到 {self.hidden}/{self.heads}。")
        if self.hidden % self.heads != 0:
            raise ShapeError(
                f"hidden={self.hidden} 不能被 heads={self.heads} 整除："
                "分头是一道整除操作，不整除时缓存里的行宽也切不开。"
            )
        if not self.keys and not self.values:
            return
        # 允许"从外部塞进来的表"，但要逐层核对形状（那是唯一可能出错的地方）
        if len(self.keys) != len(self.values):
            raise ShapeError(
                f"K 有 {len(self.keys)} 层、V 有 {len(self.values)} 层："
                "两者必须一一对应，否则某一层会拿到别人的 value。"
            )
        for index, (keys, values) in enumerate(zip(self.keys, self.values, strict=True)):
            self._check_table(index, keys, "keys")
            self._check_table(index, values, "values")
            if len(keys) != len(values):
                raise ShapeError(
                    f"第 {index} 层的 K 有 {len(keys)} 行、V 有 {len(values)} 行："
                    "长度不齐时'当前位置'这个数就没有意义了。"
                )

    def _check_table(self, index: int, table: Matrix, name: str) -> None:
        if len(table) > self.capacity:
            raise CacheError(
                f"第 {index} 层的 {name} 有 {len(table)} 行，超过容量 {self.capacity}："
                "容量是开局给定的一个硬上界，超了只能换策略（开大容量或缩短上下文）。"
            )
        for row in table:
            if len(row) != self.hidden:
                raise ShapeError(
                    f"第 {index} 层的 {name} 行宽 {len(row)} 与 hidden {self.hidden} 不一致。"
                )

    # -- 只读读数 ---------------------------------------------------------------

    @property
    def depth(self) -> int:
        """层数（由表决定；空缓存时是 0）."""
        return len(self.keys)

    @property
    def length(self) -> int:
        """当前长度（各层相同）."""
        return len(self.keys[0]) if self.keys else 0

    @property
    def free(self) -> int:
        """还能装多少个位置."""
        return max(self.capacity - self.length, 0)

    @property
    def bytes_per_step(self) -> int:
        """**每处理一个位置**新增多少字节（这就是"缓存账"的公式）.

        它就是 :func:`types.cache_bytes` 对 tokens=1 的取值——一处实现、两处调用。
        """
        return cache_bytes(max(self.depth, 1), 1, self.hidden, self.level)

    @property
    def total_bytes(self) -> int:
        """整个缓存占多少字节（逐层相加）."""
        return cache_bytes(self.depth, self.length, self.hidden, self.level)

    def keys_of(self, index: int) -> Matrix:
        """第 ``index`` 层的 K 表（越界当场拒绝）."""
        return self._table_of(index, self.keys, "keys")

    def values_of(self, index: int) -> Matrix:
        """第 ``index`` 层的 V 表."""
        return self._table_of(index, self.values, "values")

    def _table_of(self, index: int, tables: list[Matrix], name: str) -> Matrix:
        if not 0 <= index < len(tables):
            raise ShapeError(
                f"层号 {index} 越界：这份缓存只有 {len(tables)} 层"
                f"（配置里的层数是它应该等于的那个数）。"
            )
        return tables[index]

    def record(self) -> CacheState:
        """把这份可变缓存摊成一份**只读记录**（报告与测试读它）."""
        rows = tuple(
            CacheLayer(
                layer=index,
                length=len(self.keys[index]),
                capacity=self.capacity,
                hidden=self.hidden,
                bytes_per_element=element_bytes(self.level),
            )
            for index in range(self.depth)
        )
        return CacheState(layers=rows, level=self.level)

    def lines(self) -> tuple[str, ...]:
        """逐层一行（报告里读它）."""
        return self.record().lines()

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return self.record().to_dict()

    # -- 唯一的写入路径 ---------------------------------------------------------

    def ensure_layers(self, count: int) -> None:
        """按层数开好空表（**只允许开一次**：重复开会让已有内容消失）."""
        if count < 1:
            raise CacheError(f"层数必须为正，收到 {count}。")
        if self.depth and self.depth != count:
            raise CacheError(
                f"这份缓存已经有 {self.depth} 层，不能再按 {count} 层初始化："
                "两份不同层数的表拼在一起会让某些层拿到别人的 value。"
            )
        if not self.keys:
            self.keys = [() for _ in range(count)]
            self.values = [() for _ in range(count)]

    def append(self, index: int, keys_row: Vector, values_row: Vector) -> int:
        """追加**一行** K 与 V（返回追加之后的位置号）.

        这是唯一的写入路径，因此三条护栏都在这里：
        层号必须存在、行宽必须等于 hidden、**这一层**的长度不能超过容量。

        最后一条刻意按**层**检查而不是按"全局长度"：逐层推进是合法的用法
        （先给第 0 层追加、再给第 1 层追加），而按全局长度检查会在**第二层**上
        误报"已满"——那时第一层的长度正好等于容量，而第二层还是空的。
        """
        if not 0 <= index < self.depth:
            raise ShapeError(
                f"层号 {index} 越界：这份缓存有 {self.depth} 层"
                "（先调 ``ensure_layers`` 把表开好）。"
            )
        if len(keys_row) != self.hidden or len(values_row) != self.hidden:
            raise ShapeError(
                f"要追加的 K/V 行宽 {len(keys_row)}/{len(values_row)} 与 hidden "
                f"{self.hidden} 不一致：行宽不对时它会安静地拼到表尾（形状变得合法）。"
            )
        if len(self.keys[index]) >= self.capacity:
            raise CacheError(
                f"第 {index} 层的缓存已满（{len(self.keys[index])}/{self.capacity}）：追加会越界。"
                "出路是换策略——把容量开大、把上下文缩短、或者换一条能分页的路线；"
                "而不是把新的一行丢掉（那样输出会偏，而形状完全合法）。"
            )
        position = len(self.keys[index])
        self.keys[index] = self.keys[index] + (tuple(keys_row),)
        self.values[index] = self.values[index] + (tuple(values_row),)
        return position


def make_cache(
    card: ModelCard,
    *,
    capacity: int | None = None,
    level: str = LEVEL_FP32,
) -> KVCache:
    """按卡片开一份缓存（**容量默认取位置表长度**——那是模型的硬上界）."""
    resolved = card.positions if capacity is None else capacity
    cache = KVCache(hidden=card.hidden, heads=card.heads, capacity=resolved, level=level)
    cache.ensure_layers(card.layers)
    return cache


def attend_projected(
    attention_params: AttentionParams,
    query_rows: Matrix,
    key_rows: Matrix,
    value_rows: Matrix,
    *,
    heads: int,
) -> Matrix:
    """用**已经投影好**的 K/V 做一次注意力（缓存路径的核心）.

    它与 day085 的 ``hf_attention`` 的算术逐字相同——差别只有"K/V 从哪来"：

    ```text
    hf_attention       把 K/V 从输入现场投影出来（整段路径）
    attend_projected   K/V 从参数里传进来（缓存路径：老的行来自缓存、新的行刚算出来）
    ```

    七个阶段（投影 / 分头 / 打分 / 缩放 / softmax / 混合 / 拼回 + 输出投影）一个不少，
    因此两条路径的输出必须**逐位相同**。

    这里**没有掩码参数**：缓存路径上的每一次调用都发生在"最后一个位置"，
    而它能看到前面所有位置——因果性由"缓存里只有过去"这件事自动成立。
    想给一个中间位置算输出（例如逐 token 的 prefill）请走整段路径。
    """
    if heads < 1:
        raise ShapeError(f"heads 必须为正，收到 {heads}。")
    head_q = split_heads(project(query_rows, attention_params.w_query), heads)
    head_k = split_heads(key_rows, heads)
    head_v = split_heads(value_rows, heads)
    head_dim = len(head_q[0][0])
    scale = hf_scale(head_dim)
    contexts: list[Matrix] = []
    for query, keys, values in zip(head_q, head_k, head_v, strict=True):
        raw = matmul(query, transpose(keys))
        scores = tuple(tuple(value * scale for value in row) for row in raw)
        weights = row_softmax(scores)
        contexts.append(matmul(weights, values))
    merged = merge_heads(tuple(contexts))
    return project(merged, attention_params.w_output)


def _norm_and_project(
    card: ModelCard,
    weights: ModelWeights,
    index: int,
    inputs: Matrix,
) -> tuple[Matrix, Matrix, Matrix]:
    """一层里"归一化之后投影"的三件事：``normed``、``keys``、``values``.

    三者一次算出来（而不是各算各的），因为**它们必须来自同一个 normed**：
    分三次调用同一个函数是安全的（确定性），但分成三个不同的入口就迟早会走散。
    """
    from smart_research_agent.encoder_decoder.layers import layer_norm

    block, attention = weights.layer_of(index)
    normed, _cache = layer_norm(
        inputs, gamma=block.norm1_gamma, beta=block.norm1_beta, epsilon=card.ln_eps
    )
    return normed, project(normed, attention.w_key), project(normed, attention.w_value)


def _embed_at(
    card: ModelCard,
    weights: ModelWeights,
    token_id: int,
    position: int,
) -> Matrix:
    """把**一个 token 放在指定位置**上做嵌入（``embed_tokens`` 的单行版本）.

    ## 为什么必须有这个函数（缓存这一课的关键细节之一）

    day086 的 ``embed_tokens`` 按行内下标编号位置（``range(len(row))``），
    这在"一次算一整段"时是对的。而 decode 一步只喂**一个** token——
    若沿用那个函数，它会把这个新 token 当成**位置 0**，于是位置嵌入取错。
    形状完全合法、不报错，而输出会整体偏掉。

    ```text
    整段路径   第 5 个 token（下标 5）拿到位置表第 5 行
    decode     必须**显式**告诉它"你在第 5 位"，否则它会拿第 0 行
    ```

    这条差别是"缓存里必须记住当前位置"这句话在代码里的样子：
    **位置不是从 0 重新数起的**，它由缓存长度决定。
    """
    from smart_research_agent.encoder_decoder.layers import add_residual, layer_norm
    from smart_research_agent.hf_integration.types import ARCHITECTURE_BERT
    from smart_research_agent.hf_source.vocab import gather_rows

    if not 0 <= position < card.positions:
        raise CacheError(
            f"位置 {position} 超出位置表长度 {card.positions}：本包不静默回绕、也不截断。"
        )
    row = gather_rows(weights.word, (int(token_id),), name="word")
    offset = gather_rows(weights.position, (position,), name="position")
    combined = add_residual(row, offset, use_residual=True)
    if weights.token_type is not None:
        types = gather_rows(weights.token_type, (0,), name="token_type")
        combined = add_residual(combined, types, use_residual=True)
    if card.model_type == ARCHITECTURE_BERT:
        combined, _cache = layer_norm(
            combined,
            gamma=weights.embed_gamma,
            beta=weights.embed_beta,
            epsilon=card.ln_eps,
        )
    return combined


def _final_norm(card: ModelCard, weights: ModelWeights, states: Matrix) -> Matrix:
    """栈尾那一次 LayerNorm（**只有 GPT-2 有**；BERT 的块自己以 LN 收尾）.

    它单独成一个函数，是为了让"缓存路径"与"整段路径"的收尾**逐字相同**——
    day086 的 ``hidden_states`` 也做这一件事，因此两边的输出行可以直接比。
    """
    from smart_research_agent.encoder_decoder.layers import layer_norm
    from smart_research_agent.hf_integration.types import ARCHITECTURE_GPT2

    if card.model_type != ARCHITECTURE_GPT2:
        return states
    normed, _cache = layer_norm(
        states, gamma=weights.final_gamma, beta=weights.final_beta, epsilon=card.ln_eps
    )
    return normed


def prefill(
    card: ModelCard,
    weights: ModelWeights,
    cache: KVCache,
    token_ids: tuple[int, ...] | list[int],
) -> tuple[Matrix, Vector]:
    """把提示词一次算完，并把每一层的 K/V 存进缓存.

    它调 day086 的 ``run_one_block``（**整段路径**）来算输出，
    并额外把每一层的 K/V 顺手存下来——两件事共用同一个 ``normed``，
    因此缓存里的行与整段路径里用到的行**逐位相同**。

    返回值是 ``(收尾归一化之后的 hidden states, 最后一行的 logits)``——
    与 day086 的 ``hidden_states`` 同一套口径（都含栈尾那次 LN）。
    """
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids:
        raise ShapeError("prefill 需要至少一个 token：空提示词没有东西可以预填。")
    if cache.depth != card.layers:
        raise AssemblyError(
            f"缓存有 {cache.depth} 层、卡片有 {card.layers} 层："
            "两者必须来自同一次装载（层数对上而隐藏维不同时，前向照样跑完）。"
        )
    if len(ids) > cache.capacity:
        raise CacheError(
            f"提示词有 {len(ids)} 个 token，超过容量 {cache.capacity}："
            "prefill 之后还要留给生成，因此提示长度必须小于容量。"
        )
    states = embed_tokens(card, weights, (ids,))
    current = states
    for index in range(card.layers):
        _normed, keys, values = _norm_and_project(card, weights, index, current)
        for row_k, row_v in zip(keys, values, strict=True):
            cache.append(index, row_k, row_v)
        current = run_one_block(card, weights, index, current, causal=card.causal)
    final = _final_norm(card, weights, current)
    from smart_research_agent.inference_optim.quantize import logits_of

    return final, logits_of(card, weights, final[-1])


def decode_step(
    card: ModelCard,
    weights: ModelWeights,
    cache: KVCache,
    token_id: int,
) -> tuple[Matrix, Vector, int]:
    """推进一步（**每层只投影一行**），把新的 K/V 追加进缓存.

    返回 ``(这一行收尾归一化之后的 hidden state, logits, 位置号)``。
    位置号来自缓存长度——它必须与"这一行在序列里的下标"一致，
    否则位置嵌入会取错（而形状完全合法）。
    """
    if cache.depth != card.layers:
        raise AssemblyError(
            f"缓存有 {cache.depth} 层、卡片有 {card.layers} 层：两者必须来自同一次装载。"
        )
    if cache.length >= cache.capacity:
        raise CacheError(
            f"缓存已满（{cache.length}/{cache.capacity}）：这一步放不下。"
            "出路是改策略，而不是把这一步丢掉。"
        )
    position = cache.length
    states = _embed_at(card, weights, int(token_id), position)
    current = states
    for index in range(card.layers):
        attention = weights.layer_of(index)[1]
        normed, keys_new, values_new = _norm_and_project(card, weights, index, current)
        cache.append(index, keys_new[0], values_new[0])
        attended = attend_projected(
            attention,
            normed,
            cache.keys_of(index),
            cache.values_of(index),
            heads=card.heads,
        )
        current = _finish_block(card, weights, index, current, attended)
    final = _final_norm(card, weights, current)
    from smart_research_agent.inference_optim.quantize import logits_of

    return final, logits_of(card, weights, final[-1]), position


def _finish_block(
    card: ModelCard,
    weights: ModelWeights,
    index: int,
    inputs: Matrix,
    attended: Matrix,
) -> Matrix:
    """一层里注意力之后的那一半（残差 + 第二个子层），**两个架构共用**.

    它必须与 day086 的 ``run_one_block`` 的后半段逐字相同：
    pre-LN 是 ``x = x + attn`` 再 ``x = x + mlp(ln_2(x))``，
    post-LN 是 ``x = ln_1(x + attn)`` 再 ``x = ln_2(x + mlp(x))``。
    两处略有不同，因此这里按架构分开写——而不是"看起来差不多就共用一条"。
    """
    from smart_research_agent.encoder_decoder.layers import add_residual, layer_norm
    from smart_research_agent.encoder_decoder.types import FFNWeights
    from smart_research_agent.hf_integration.types import ARCHITECTURE_BERT
    from smart_research_agent.hf_source.blocks import gpt2_mlp

    block = weights.layer_of(index)[0]
    ffn = FFNWeights(
        w_in=block.ffn_w_in, b_in=block.ffn_b_in, w_out=block.ffn_w_out, b_out=block.ffn_b_out
    )
    if card.model_type == ARCHITECTURE_BERT:
        after, _cache = layer_norm(
            add_residual(inputs, attended, use_residual=True),
            gamma=block.norm1_gamma,
            beta=block.norm1_beta,
            epsilon=card.ln_eps,
        )
        mlp = gpt2_mlp(after, ffn, activation=card.activation)
        output, _cache_two = layer_norm(
            add_residual(after, mlp, use_residual=True),
            gamma=block.norm2_gamma,
            beta=block.norm2_beta,
            epsilon=card.ln_eps,
        )
        return output
    after = add_residual(inputs, attended, use_residual=True)
    normed_two, _cache = layer_norm(
        after, gamma=block.norm2_gamma, beta=block.norm2_beta, epsilon=card.ln_eps
    )
    return add_residual(
        after, gpt2_mlp(normed_two, ffn, activation=card.activation), use_residual=True
    )


def generate_with_cache(
    card: ModelCard,
    weights: ModelWeights,
    prompt_ids: tuple[int, ...] | list[int],
    *,
    steps: int,
    capacity: int | None = None,
    level: str = LEVEL_FP32,
) -> tuple[tuple[Vector, ...], CacheState, tuple[StepRow, ...]]:
    """贪心生成 ``steps`` 步，并留下**每一步的字节账**.

    选贪心（而不是四种策略）是刻意的：本课要量的是**缓存的账**，
    而策略的差别已经在 day085/day086 量过了。由此这条函数只依赖
    "哪一步选哪个 token"，与"怎么选"无关。
    """
    if steps < 0:
        raise ShapeError(f"steps 不能为负，收到 {steps}。")
    cache = make_cache(card, capacity=capacity, level=level)
    _states, logits = prefill(card, weights, cache, prompt_ids)
    rows: list[StepRow] = [
        StepRow(
            step=0,
            position=cache.length - 1,
            cache_bytes=cache.total_bytes,
            delta_bytes=cache.total_bytes,
            stage=STAGE_PREFILL,
        )
    ]
    chosen: list[Vector] = []
    for step in range(steps):
        token = max(range(len(logits)), key=lambda index: (logits[index], -index))
        chosen.append(logits)
        _state, logits, position = decode_step(card, weights, cache, token)
        rows.append(
            StepRow(
                step=step + 1,
                position=position,
                cache_bytes=cache.total_bytes,
                delta_bytes=cache.bytes_per_step,
                stage=STAGE_DECODE,
            )
        )
    return tuple(chosen), cache.record(), tuple(rows)


def cache_account_line(
    card: ModelCard,
    *,
    tokens: int,
    level: str = LEVEL_FP32,
) -> str:
    """一行读数：给定长度下缓存的字节数（**公式与逐层相加两个数必须相等**）."""
    formula = cache_bytes(card.layers, tokens, card.hidden, level)
    by_layer = sum(
        CacheLayer(
            layer=index,
            length=tokens,
            capacity=max(tokens, card.positions),
            hidden=card.hidden,
            bytes_per_element=element_bytes(level),
        ).bytes
        for index in range(card.layers)
    )
    return (
        f"{level}：{card.layers} 层 × {tokens} 位置 × {card.hidden} 维 | "
        f"公式 {formula} 字节 | 逐层相加 {by_layer} 字节 | 相等 {formula == by_layer}"
    )


def compare_with_recompute(
    card: ModelCard,
    weights: ModelWeights,
    prompt_ids: tuple[int, ...],
    new_token: int,
) -> tuple[Vector, Vector]:
    """两条路径的最后一步 logits：``(缓存路径, 整段重算路径)``.

    这是本课最强的一条对账素材：两条路径的算术逐字相同，
    因此两个向量必须**逐位相等**。它抓的错法是"缓存在归一化之前取"或"K/V 弄反"——
    两种错法下形状全部合法。
    """
    from smart_research_agent.hf_integration.forward import logits as full_logits

    cache = make_cache(card)
    _states, _logits = prefill(card, weights, cache, prompt_ids)
    _state, cached_logits, _position = decode_step(card, weights, cache, new_token)
    full = tuple(prompt_ids) + (int(new_token),)
    return cached_logits, full_logits(card, weights, full)


def step_delta_is_a_formula(card: ModelCard, *, level: str = LEVEL_FP32) -> str:
    """一行读数：每步的增量 == ``2·L·h·bytes``（整数相等）."""
    cache = make_cache(card, level=level)
    expected = int(2 * card.layers * card.hidden * element_bytes(level))
    return (
        f"{level}：L={card.layers} h={card.hidden} ⇒ 每步 {cache.bytes_per_step} 字节 "
        f"| 2·L·h·bytes = {expected} | 相等 {cache.bytes_per_step == expected} | "
        f"每步的查询行数 {DECODE_TOKENS_PER_STEP}（这就是省时间的全部来源）"
    )


def position_of(cache: KVCache) -> int:
    """缓存当前长度（= 下一个 token 的位置号）."""
    return cache.length


def tokens_per_second(tokens: int, seconds: float) -> float:
    """吞吐（token/秒）—— 一个 **neutral** 方向的读数（越大越好，但要与延迟一起看）."""
    if seconds <= 0:
        raise ShapeError(f"时间必须为正，收到 {seconds}。")
    return tokens / seconds


def attention_work(tokens: int) -> tuple[int, int]:
    """两种做法各自要算多少次**打分**：``(重算, 缓存)``.

    ```text
    重算     第 t 步要把前面 t 个位置重算一遍 ⇒ 1 + 2 + … + T = T(T+1)/2 次
    缓存     第 t 步只算新位置对前面 T 个位置 ⇒ 每步 T 次 ⇒ 总共 T 次
    ```

    两个都是整数，因此"省了多少"是一个能被算出来、而不是被形容出来的数。
    注意它只数**打分**（`q·k`），不数投影——投影那一侧从 O(T²) 降到 O(T)，
    降得更多（那是缓存的另一半收益）。
    """
    if tokens < 1:
        raise ShapeError(f"长度必须为正，收到 {tokens}。")
    return tokens * (tokens + 1) // 2, tokens


__all__ = [
    "DEFAULT_CAPACITY_RATIO",
    "KVCache",
    "attend_projected",
    "attention_work",
    "cache_account_line",
    "compare_with_recompute",
    "decode_step",
    "generate_with_cache",
    "make_cache",
    "position_of",
    "prefill",
    "step_delta_is_a_formula",
    "tokens_per_second",
]
