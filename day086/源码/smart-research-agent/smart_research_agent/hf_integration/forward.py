"""``forward``：把那份配置变成**一次真的前向**（day086 / M7-D10）.

前四步（找文件、解释配置、构造分词器、池化）都还没碰到"模型"。本模块才碰：
它按 :class:`~smart_research_agent.hf_integration.types.ModelCard` 的形状
造一份确定性权重，然后把 day079~085 写过的那些零件**按配置串起来**：

```text
GPT-2    embed   = wte[id] + wpe[pos]                      （不加 LN）
         L ×      x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
         ln_f     最后一次 LayerNorm
         lm_head  与 wte **共享**（tie_word_embeddings=True）

BERT     embed   = LN(wte[id] + wte_type[0] + wpe[pos])     （多一次 LN）
         L ×      x = ln_1(x + attn(x)); x = ln_2(x + mlp(x))
         pooler   tanh(W·x[0] + b)
```

## 一个"差值必须被逐项解释"的兑现（day080 那条纪律的第四次）

本包的前向**省略了注意力的偏置**（day075 的四个投影都不带偏置），
所以它的真实参数量比那份配置算出来的少。差值是多少？可以被**算出来**：

```text
配置口径的每一层    12h² + 13h
本包权重每一层      12h² +  9h
差                 4h        = 注意力的偏置
```

而 4h 在两边的**拆法不同**，这正是本课想留下的一句话：

```text
GPT-2   c_attn 的偏置 3h  +  c_proj 的偏置 h        （一次融合投影，分成两段）
BERT    query/key/value 各 h  +  output.dense 的 h   （四次独立投影，各一）
```

因此 `4·hidden·layers` 是一个**能被逐项解释**的整数，
而不是一句"差不多"——``check_weights_gap_is_explained`` 会把它钉住。

## 填充什么时候是惰性的（本课最值钱的一条）

```text
因果模型     填充排在真实位置**之后**，因果掩码让它们一格都传不回来
             ⇒ 同一段之内的填充是惰性的
双向模型     填充会真的改写真实位置那一行的输出
             ⇒ 必须先按真实长度**切开**再跑（本包的 safe 策略就是这么做的）
```

但"因果 ⇒ 填充惰性"这句话**只在单条样本内成立**。三条样本拼成一条长序列之后，
那张 ``(n, n)`` 的下三角会让第 2 条样本看到第 1 条的**全部** token——
因为掩码不知道"段"这件事。正确做法是把掩码升级成**段内下三角**
（:func:`block_causal_mask`），也就是真实实现里那张 ``(batch, 1, seq, seq)``。

``policy="naive"`` 是**故意漏掉这一步**的那一版：它在因果侧漏掉"段"、
在双向侧漏掉"切开"，两种错法各有各的读数，而两者都**不会报错**。
它存在不是为了用，而是为了让安全策略**可被证明是必要的**。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from smart_research_agent.encoder_decoder.depth import (
    DEFAULT_INIT_SCALE,
    make_block_parameters,
)
from smart_research_agent.encoder_decoder.layers import add_residual, layer_norm
from smart_research_agent.encoder_decoder.types import (
    BlockParameters,
    BlockShape,
    FFNWeights,
)
from smart_research_agent.hf_integration.errors import ShapeError, TokenError
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_BERT,
    ARCHITECTURE_GPT2,
    ModelCard,
)
from smart_research_agent.hf_source.attention import hf_attention
from smart_research_agent.hf_source.blocks import gpt2_mlp
from smart_research_agent.hf_source.types import SourceShape
from smart_research_agent.hf_source.vocab import gather_rows
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams

#: 安全策略：因果侧整批跑（填充惰性），双向侧先按真实长度切开。
POLICY_SAFE = "safe"

#: 故意违反上一条规矩的那一版（**只用于把代价印出来**）。
POLICY_NAIVE = "naive"

POLICIES: tuple[str, ...] = (POLICY_SAFE, POLICY_NAIVE)

#: 每一层的配置口径参数量（``12h² + 13h``）——两个架构**恰好相同**，这是本课的一个结论。
LAYER_CONFIG_SCALE = (12, 13)

#: 每一层的本包权重参数量（``12h² + 9h``）：少的正是注意力的偏置 ``4h``。
LAYER_WEIGHTS_SCALE = (12, 9)

#: 每一层注意力偏置的项数（两个架构都是 4 项，但**拆法不同**）。
BIAS_ITEMS_PER_LAYER = 4

#: 两侧的逐项拆法（报告里读它，因为"同一个 4h"的来源不一样）。
BIAS_BREAKDOWN: dict[str, str] = {
    ARCHITECTURE_GPT2: "c_attn 的偏置 3h + c_proj 的偏置 h（一次融合投影分成两段）",
    ARCHITECTURE_BERT: "query / key / value 各 h + output.dense 的 h（四次独立投影各一）",
}


def _matrix(rows: int, columns: int, *, seed: int, scale: float = DEFAULT_INIT_SCALE) -> Matrix:
    """确定性随机矩阵（``[-scale, scale)``，用 day073 那串 LCG 随机数）.

    与 day079 的 ``_random_matrix`` 同一口径，因此"同样的种子得到同样的权重"
    是一条可以被复核的结论——不是"上次跑出来的值"。
    """
    if rows < 1 or columns < 1:
        raise ShapeError(f"矩阵形状必须为正，收到 ({rows}, {columns})。")
    if not math.isfinite(scale) or scale <= 0:
        raise ShapeError(f"scale 必须是正的有限数，收到 {scale!r}。")
    raw = uniforms(rows * columns, seed=seed)
    return tuple(
        tuple(value * 2.0 * scale - scale for value in raw[index * columns : (index + 1) * columns])
        for index in range(rows)
    )


@dataclass(frozen=True)
class ModelWeights:
    """一份被造出来的权重：嵌入三张表 + 每一层 + 收尾.

    ``head`` 在 ``tie_word_embeddings=True`` 时**就是** ``word``——
    共享不是"再复制一份"，因此它与 ``word`` 是同一个对象（测试里用 ``is`` 查这件事）。
    """

    card: ModelCard
    word: Matrix
    position: Matrix
    token_type: Matrix | None
    layers: tuple[tuple[BlockParameters, AttentionParams], ...]
    final_gamma: Vector
    final_beta: Vector
    head: Matrix
    pooler_w: Matrix | None
    pooler_b: Vector | None
    tied: bool
    seed: int
    embed_gamma: Vector | None = None
    embed_beta: Vector | None = None

    @property
    def hidden(self) -> int:
        """隐藏维（每一层所有张量的宽度）."""
        return self.card.hidden

    @property
    def parameter_count(self) -> int:
        """本包权重真实持有的参数量（**按张量逐个累加**，不是按公式套）."""
        total = len(self.word) * len(self.word[0]) + len(self.position) * len(self.position[0])
        if self.token_type is not None:
            total += len(self.token_type) * len(self.token_type[0])
        hidden = self.hidden
        ffn = self.card.ffn
        per_layer = (
            4 * hidden * hidden  # 四个投影（**无偏置**）
            + 4 * hidden  # 两个 LayerNorm 的 γ/β
            + ffn * hidden  # ffn_w_in
            + ffn  # ffn_b_in（值是 0，但它是一个参数）
            + hidden * ffn  # ffn_w_out
            + hidden  # ffn_b_out
        )
        total += self.card.layers * per_layer
        if self.card.model_type == ARCHITECTURE_GPT2:
            total += 2 * hidden  # ln_f
        if self.embed_gamma is not None:
            total += 2 * hidden  # BERT 嵌入层那一次 LayerNorm 的 γ/β
        if self.pooler_w is not None:
            total += hidden * hidden + hidden
        if not self.tied:
            total += len(self.head) * len(self.head[0])
        return total

    @property
    def expected_gap(self) -> int:
        """配置口径与本包权重之间**应当**存在的差值（``4·hidden·layers``，逐项可解释）.

        两侧的拆法不同（见 :data:`BIAS_BREAKDOWN`），但**总数相同**——
        因为两个架构的注意力偏置都是每层 ``4h`` 项。
        """
        return BIAS_ITEMS_PER_LAYER * self.hidden * self.card.layers

    @property
    def gap_is_explained(self) -> bool:
        """两个口径之差是否恰好等于注意力偏置那一项."""
        return self.parameter_count + self.expected_gap == self.card.parameter_count

    def layer_of(self, index: int) -> tuple[BlockParameters, AttentionParams]:
        """取第 ``index`` 层的两份参数（越界当场拒绝）."""
        if not 0 <= index < len(self.layers):
            raise ShapeError(
                f"层号 {index} 越界：这份权重只有 {len(self.layers)} 层"
                f"（配置里写的是 n_layer / num_hidden_layers = {self.card.layers}）。"
            )
        return self.layers[index]

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**张量本体，只印形状与两个参数量口径）."""
        return {
            "model_type": self.card.model_type,
            "layers": len(self.layers),
            "hidden": self.hidden,
            "vocab": len(self.word),
            "positions": len(self.position),
            "tied": self.tied,
            "has_token_type": self.token_type is not None,
            "has_pooler": self.pooler_w is not None,
            "parameters": self.parameter_count,
            "config_parameters": self.card.parameter_count,
            "expected_gap": self.expected_gap,
            "gap_is_explained": self.gap_is_explained,
            "seed": self.seed,
        }


def make_weights(card: ModelCard, *, seed: int = 7) -> ModelWeights:
    """按配置造一份确定性权重（**层与层不同、同一种子可复现**）.

    每一层用两个不同的种子偏移（``seed + i*17`` 与 ``seed + i*17 + 3``），
    因此"层与层完全一样"这种错误会让读数**看起来正常**而实际退化——
    本包用"层参数逐位不同"作为一条自检（见 :mod:`verify`）。
    """
    hidden = card.hidden
    word = _matrix(card.vocab, hidden, seed=seed)
    position = _matrix(card.positions, hidden, seed=seed + 1)
    token_type = None
    if card.uses_token_type:
        token_type = _matrix(max(card.type_vocab, 1), hidden, seed=seed + 2)
    shape = BlockShape(hidden=hidden, ffn=card.ffn, tokens=1)
    layers: list[tuple[BlockParameters, AttentionParams]] = []
    for index in range(card.layers):
        offset = seed + 17 * (index + 1)
        layers.append(
            (
                make_block_parameters(shape, seed=offset, scale=DEFAULT_INIT_SCALE),
                default_parameters(hidden, seed=offset + 3, scale=DEFAULT_INIT_SCALE),
            )
        )
    # 共享与否是**配置的属性**，不是架构的属性：
    # 真实的 bert-base-uncased 也共享（它的 config.json 里没有这个键，于是默认 True 生效）。
    tied = card.tie_word_embeddings
    head = word if tied else _matrix(card.vocab, hidden, seed=seed + 5)
    pooler_w = None
    pooler_b = None
    embed_gamma = None
    embed_beta = None
    if card.model_type == ARCHITECTURE_BERT:
        pooler_w = _matrix(hidden, hidden, seed=seed + 7)
        pooler_b = tuple(0.0 for _ in range(hidden))
        # BERT 的嵌入层那一次 LN 是**带参数**的（这是它的参数量公式里那 2h 的来源）
        embed_gamma = tuple(1.0 for _ in range(hidden))
        embed_beta = tuple(0.0 for _ in range(hidden))
    return ModelWeights(
        card=card,
        word=word,
        position=position,
        token_type=token_type,
        layers=tuple(layers),
        final_gamma=tuple(1.0 for _ in range(hidden)),
        final_beta=tuple(0.0 for _ in range(hidden)),
        head=head,
        pooler_w=pooler_w,
        pooler_b=pooler_b,
        tied=tied,
        seed=seed,
        embed_gamma=embed_gamma,
        embed_beta=embed_beta,
    )


@dataclass(frozen=True)
class HiddenBatch:
    """一次前向的结果：**每条样本只保留真实长度**的那些行.

    刻意**不留填充行**：填充行没有语义（它们的值取决于 pad id 与后面有几行），
    把它们留在结果里，任何池化都会"看起来对"而实际算进了垃圾。
    需要整批张量形状时用 :meth:`padded` 一次拼出来。
    """

    rows: tuple[Matrix, ...]
    policy: str
    batched: bool

    @property
    def batch_size(self) -> int:
        """几条样本."""
        return len(self.rows)

    @property
    def hidden(self) -> int:
        """隐藏维."""
        return len(self.rows[0][0]) if self.rows and self.rows[0] else 0

    def lengths(self) -> tuple[int, ...]:
        """每条样本的真实长度."""
        return tuple(len(row) for row in self.rows)

    def padded(self) -> Matrix:
        """把各条按真实长度首尾相接（**逐位不含任何填充行**）."""
        return tuple(value for row in self.rows for value in row)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**张量本体）."""
        return {
            "batch_size": self.batch_size,
            "hidden": self.hidden,
            "lengths": list(self.lengths()),
            "policy": self.policy,
            "batched": self.batched,
        }


def _normalize_rows(
    input_ids: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    attention_mask: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """把若干不等长的行整理成"行 + 掩码"（掩码缺省时按真实长度推出来）."""
    rows = tuple(tuple(int(token_id) for token_id in row) for row in input_ids)
    if not rows:
        raise ShapeError("一次前向至少要有一条样本：空批次没有行可以算。")
    for index, row in enumerate(rows):
        if not row:
            raise ShapeError(f"第 {index} 条样本是空的：零个 token 没有 hidden state。")
    if attention_mask is None:
        return rows, tuple(tuple(1 for _ in row) for row in rows)
    masks = tuple(tuple(int(flag) for flag in row) for row in attention_mask)
    if len(masks) != len(rows):
        raise ShapeError(
            f"掩码有 {len(masks)} 行、输入有 {len(rows)} 行："
            "行数不一致时按行配对会静默地把掩码错位一整条样本。"
        )
    for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
        if len(mask) != len(row):
            raise ShapeError(
                f"第 {index} 条的掩码长度 {len(mask)} 与输入长度 {len(row)} 不一致："
                "长度不齐会被 zip 静默截断（少看几个 token），而形状看起来是合法的。"
            )
        if sum(mask) == 0:
            raise ShapeError(
                f"第 {index} 条的掩码全是 0：它没有任何真实位置，"
                "因此它的池化会是一次 0/0（day086 的 NumericError 那一族的前一站）。"
            )
    return rows, masks


def _check_ids(card: ModelCard, rows: tuple[tuple[int, ...], ...]) -> None:
    """校验 id 在词表内、长度不超过位置表——**两条硬上界都在这里挡**."""
    for index, row in enumerate(rows):
        if len(row) > card.positions:
            raise ShapeError(
                f"第 {index} 条样本有 {len(row)} 个 token，超过位置表长度 "
                f"{card.positions}：位置表的长度是一道**硬上界**"
                "（day085 的 vocab 模块记过同一条）。"
            )
        for token_id in row:
            if token_id < 0 or token_id >= card.vocab:
                raise TokenError(
                    f"token id {token_id} 不在词表里（词表大小 {card.vocab}）："
                    "它可能来自另一个分词器——那时嵌入层会取到别人的那一行，"
                    "而前向照样跑完。"
                )


def _embed(card: ModelCard, weights: ModelWeights, rows: tuple[tuple[int, ...], ...]) -> Matrix:
    """一次嵌入：``wte[id] + wpe[pos]``（BERT 侧再加 token_type 与一次 LN）."""
    hidden = card.hidden
    out: list[tuple[float, ...]] = []
    for row in rows:
        token_rows = gather_rows(weights.word, row, name="word")
        position_rows = gather_rows(weights.position, tuple(range(len(row))), name="position")
        combined = add_residual(token_rows, position_rows, use_residual=True)
        if weights.token_type is not None:
            types = gather_rows(
                weights.token_type, tuple(0 for _ in row), name="token_type"
            )
            combined = add_residual(combined, types, use_residual=True)
        if card.model_type == ARCHITECTURE_BERT:
            # BERT 的嵌入层里有一次 LayerNorm（GPT-2 没有）——day085 记过的那处分岔
            combined, _cache = layer_norm(
                combined,
                gamma=weights.embed_gamma,
                beta=weights.embed_beta,
                epsilon=card.ln_eps,
            )
        out.extend(combined)
    if len(out[0]) != hidden:  # pragma: no cover - 形状由 gather_rows 保证
        raise ShapeError("嵌入结果的宽度与配置里的 hidden 不一致。")
    return tuple(out)


def block_causal_mask(lengths: tuple[int, ...]) -> tuple[tuple[bool, ...], ...]:
    """把若干条样本首尾相接之后，"允许集合"是**同一段之内的下三角**.

    ## 为什么必须自己做这张表（这是本课最值钱的一条工程结论）

    单条样本的因果掩码是一张 ``(n, n)`` 的下三角。把三条样本拼成一条长序列之后，
    如果你仍然只用那张 ``(n, n)`` 的下三角，那么：

    ```text
    第 2 条样本的第 0 个 token 会看到第 1 条样本的**全部** token
      —— 因为它落在"下三角之内"，而掩码不知道"段"这件事
    ```

    它不会报错，也不会给出任何异常读数：第 2 条的输出只是**偏了**，
    而偏的方向取决于同一批里前面那条样本的内容。真实实现因此把掩码升级成
    ``(batch, 1, seq, seq)``——本包用同一张 ``(total, total)`` 的显式掩码表达同一件事。

    第 1 条样本（段内第一条）**碰巧是对的**，这正是这类 bug 最难被发现的地方：
    拿一条样本试，永远试不出来。
    """
    if not lengths:
        raise ShapeError("掩码需要至少一条样本的长度。")
    if any(length < 1 for length in lengths):
        raise ShapeError("每条样本的长度必须 >= 1（零长度的段没有掩码可描述）。")
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for length in lengths:
        bounds.append((cursor, cursor + length))
        cursor += length
    total = cursor
    rows: list[tuple[bool, ...]] = []
    for start, stop in bounds:
        for index in range(start, stop):
            rows.append(tuple(start <= column <= index for column in range(total)))
    return tuple(rows)


def _ffn_of(block: BlockParameters) -> FFNWeights:
    """把 day079 块参数里的前馈那四块取出来（**只换容器，不换值**）."""
    return FFNWeights(
        w_in=block.ffn_w_in,
        b_in=block.ffn_b_in,
        w_out=block.ffn_w_out,
        b_out=block.ffn_b_out,
    )


def run_one_block(
    card: ModelCard,
    weights: ModelWeights,
    index: int,
    inputs: Matrix,
    *,
    causal: bool = True,
    mask: tuple[tuple[bool, ...], ...] | None = None,
) -> Matrix:
    """跑一层（**两个架构的接线只差 LayerNorm 的位置**，与 day085 的块逐字相同）.

    与 day085 的 ``gpt2_block`` / ``bert_layer`` 的唯一差别是：本函数允许传入一张
    **显式掩码**（:func:`block_causal_mask`），而 day085 的块只接受 "causal 或 全双向"。
    这条扩展是必需的——批量之后"因果"这件事就不再是一张 ``(n, n)`` 的表了。
    """
    block, attention = weights.layer_of(index)
    shape = SourceShape(
        tokens=len(inputs),
        hidden=card.hidden,
        heads=card.heads,
        vocab=card.vocab,
        causal_default=causal,
    )
    ffn = _ffn_of(block)
    if card.model_type == ARCHITECTURE_GPT2:
        # pre-LN：x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
        normed, _cache = layer_norm(
            inputs, gamma=block.norm1_gamma, beta=block.norm1_beta, epsilon=card.ln_eps
        )
        forward = hf_attention(
            attention, normed, shape, causal=causal and mask is None, mask=mask
        )
        after = add_residual(inputs, forward.output, use_residual=True)
        normed_two, _cache_two = layer_norm(
            after, gamma=block.norm2_gamma, beta=block.norm2_beta, epsilon=card.ln_eps
        )
        return add_residual(
            after, gpt2_mlp(normed_two, ffn, activation=card.activation), use_residual=True
        )
    if card.model_type == ARCHITECTURE_BERT:
        # post-LN：x = ln_1(x + attn(x)); x = ln_2(x + mlp(x))
        forward = hf_attention(
            attention, inputs, shape, causal=causal and mask is None, mask=mask
        )
        after, _cache = layer_norm(
            add_residual(inputs, forward.output, use_residual=True),
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
    from smart_research_agent.hf_integration.errors import ConfigError

    raise ConfigError(  # pragma: no cover - 卡片来自 parse_config，架构必然被认得
        f"未知架构 {card.model_type!r}：本包只给了两种块（{list(ARCHITECTURE_GPT2)} 与 BERT）。"
    )


def embed_tokens(
    card: ModelCard,
    weights: ModelWeights,
    rows: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
) -> Matrix:
    """嵌入层（**公开入口**：``wte[id] + wpe[pos]``，BERT 侧再加 token_type 与一次 LN）.

    它单独暴露出来的理由与 :func:`run_one_block` 相同：跨天对账需要
    "拿同一份输入、同一份权重"从两个实现各跑一遍，而嵌入是那条链的起点。
    """
    return _embed(card, weights, tuple(tuple(row) for row in rows))


def _run_layers(
    card: ModelCard,
    weights: ModelWeights,
    states: Matrix,
    *,
    causal: bool,
    mask: tuple[tuple[bool, ...], ...] | None = None,
) -> Matrix:
    """把 ``card.layers`` 层跑完（**每一层都不改变形状**）."""
    current = states
    for index in range(card.layers):
        current = run_one_block(card, weights, index, current, causal=causal, mask=mask)
    if card.model_type == ARCHITECTURE_GPT2:
        current, _cache = layer_norm(
            current, gamma=weights.final_gamma, beta=weights.final_beta, epsilon=card.ln_eps
        )
    return current


def hidden_states(
    card: ModelCard,
    weights: ModelWeights,
    input_ids: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    attention_mask: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None = None,
    *,
    policy: str = POLICY_SAFE,
) -> HiddenBatch:
    """一次前向：``(batch, seq)`` 的 id → 每条样本真实长度的 hidden states.

    ``policy`` 决定**填充与样本边界怎么被对待**（见本模块文档第三段）：

    ```text
    safe    因果模型：整批一次跑，但掩码是**段内下三角**（block_causal_mask）
                     ⇒ 填充传不回来、别的样本也传不过来
            双向模型：先按真实长度切开，逐条跑（**这是必须的，不是优化**）
    naive   一律整批一次跑：因果侧漏掉"段"这件事（第 2 条会看到第 1 条），
            双向侧把填充一起喂进注意力（真实位置那一行被改写）
    ```
    """
    if policy not in POLICIES:
        raise ShapeError(f"未知的填充策略 {policy!r}：可选 {list(POLICIES)}。")
    rows, masks = _normalize_rows(input_ids, attention_mask)
    _check_ids(card, rows)
    lengths = tuple(sum(mask) for mask in masks)
    for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
        if sum(mask) != len(row) and any(flag == 0 for flag in mask[: sum(mask)]):
            raise ShapeError(
                f"第 {index} 条的掩码里有'空洞'（真实位之间夹着 0）："
                "本包按'填充只在右边'的约定工作（与真实库的 padding_side='right' 一致）。"
            )
    if policy == POLICY_SAFE and not card.causal:
        states = tuple(
            _run_layers(
                card,
                weights,
                _embed(card, weights, (row[:length],)),
                causal=False,
            )
            for row, length in zip(rows, lengths, strict=True)
        )
        return HiddenBatch(rows=states, policy=policy, batched=False)
    width = max(len(row) for row in rows)
    padded_rows = tuple(
        row + tuple(0 for _ in range(width - len(row))) for row in rows
    )
    # 因果侧：**必须**给一张"段内下三角"的掩码。只给 causal=True 的后果是
    # 第 2 条样本会看到第 1 条样本的全部 token（见 block_causal_mask 的说明），
    # 而它的输出看起来完全正常。naive 策略故意漏掉这一步。
    block = (
        block_causal_mask(tuple(width for _ in padded_rows))
        if card.causal and policy == POLICY_SAFE
        else None
    )
    state = _run_layers(
        card,
        weights,
        _embed(card, weights, padded_rows),
        causal=card.causal,
        mask=block,
    )
    sliced: list[Matrix] = []
    cursor = 0
    for length in lengths:
        sliced.append(tuple(state[cursor : cursor + length]))
        cursor += width
    return HiddenBatch(rows=tuple(sliced), policy=policy, batched=True)


def pooler_output(weights: ModelWeights, batch: HiddenBatch) -> Matrix:
    """BERT 的 ``pooler_output``：取**第一个真实位置**再过一层 ``tanh`` 投影.

    它只对 BERT 侧存在（GPT-2 没有 pooler），而它的输入是 ``x[:, 0]``——
    这正是"填充必须排在右边"的另一个理由：取错位置会取到一个填充行。
    """
    if weights.pooler_w is None or weights.pooler_b is None:
        raise TokenError(
            "这份权重没有 pooler（只有 BERT 侧才有）："
            "GPT-2 的特征抽取要从 hidden states 里自己池化，"
            "而不是读一个 pooler_output——两者是不同的东西。"
        )
    pooled: list[tuple[float, ...]] = []
    for row in batch.rows:
        first = row[0]
        projected = tuple(
            math.fsum(value * weight for value, weight in zip(first, unit, strict=True)) + offset
            for unit, offset in zip(weights.pooler_w, weights.pooler_b, strict=True)
        )
        pooled.append(tuple(math.tanh(value) for value in projected))
    return tuple(pooled)


def logits(card: ModelCard, weights: ModelWeights, token_ids: tuple[int, ...]) -> Vector:
    """最后一步：末位 hidden state · ``lm_head``ᵀ → 长度 ``vocab`` 的 logits.

    共享词嵌入时 ``lm_head`` **就是** ``wte``，因此这里不需要任何"额外权重"
    就能把 hidden 投回词表空间——这正是 tying 省下的那 ``vocab × hidden`` 个参数。
    """
    if not token_ids:
        raise ShapeError("logits 需要至少一个 token：空序列没有末位。")
    batch = hidden_states(card, weights, (token_ids,))
    last = batch.rows[0][-1]
    return tuple(
        math.fsum(value * weight for value, weight in zip(last, unit, strict=True))
        for unit in weights.head
    )


def make_logits_fn(
    card: ModelCard,
    weights: ModelWeights,
) -> Callable[[tuple[int, ...]], Vector]:
    """给 day085 的 ``generate`` 用的那个"模型"（``tokens → logits``）.

    适配器的全部内容就是一次 :func:`logits` 调用，因此"本包的生成"
    与"day085 的生成"之间**只差一个模型**，而不是两套策略。
    """

    def logits_fn(tokens: tuple[int, ...]) -> Vector:
        return logits(card, weights, tokens)

    return logits_fn


def embedding_norm_line(card: ModelCard, weights: ModelWeights) -> str:
    """一行读数：嵌入后到底有没有那次 LN（两个架构的第一处分岔）."""
    return (
        f"{card.model_type}：嵌入后 LN = "
        f"{'有（BERT 的 BertEmbeddings）' if card.model_type == ARCHITECTURE_BERT else '无（GPT-2 只在栈尾 ln_f）'}"
        f" | token_type 表 {'有' if weights.token_type is not None else '无'}"
    )


def bias_breakdown_line(card: ModelCard, weights: ModelWeights) -> str:
    """一行读数：那个"差值从哪来"（**逐项**，不是一句"少了一点"）."""
    hidden = card.hidden
    return (
        f"{card.model_type}：权重 {weights.parameter_count} + 注意力偏置 "
        f"{BIAS_ITEMS_PER_LAYER}·{hidden}·{card.layers} = {weights.card.parameter_count} "
        f"| 逐项：{BIAS_BREAKDOWN[card.model_type]} | 成立：{weights.gap_is_explained}"
    )


__all__ = [
    "BIAS_BREAKDOWN",
    "BIAS_ITEMS_PER_LAYER",
    "LAYER_CONFIG_SCALE",
    "LAYER_WEIGHTS_SCALE",
    "POLICIES",
    "POLICY_NAIVE",
    "POLICY_SAFE",
    "HiddenBatch",
    "ModelWeights",
    "bias_breakdown_line",
    "block_causal_mask",
    "embed_tokens",
    "embedding_norm_line",
    "hidden_states",
    "logits",
    "make_logits_fn",
    "make_weights",
    "pooler_output",
    "run_one_block",
]
