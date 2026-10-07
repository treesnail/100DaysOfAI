"""词表与嵌入层：**同一件事，两个模型的第一个分岔**（day085 / M7-D9）.

`modeling_gpt2.py` 与 `modeling_bert.py` 的第一段差异不在注意力上，而在**嵌入之后**：

```text
GPT-2    hidden = wte[token] + wpe[position]                      ← 到此为止
BERT     hidden = LN(wte[token] + wte_type[token_type] + wpe[position])   ← 多了一次 LN
```

这一条差别有三个可被断言的后果，而它们都不是"实现的自由"：

```text
① 第一层的输入尺度不同      GPT-2 的嵌入是两行之和，BERT 的是"标准化过"的两/三行之和
② 位置表的长度是硬上界      两者的位置表都是定长的（max_position_embeddings），越界必须拒绝
③ token_type 只在 BERT 侧存在  GPT-2 没有"句子 A / 句子 B"的概念
```

还有一条**同名不同默认值**的事实值得单独记：LayerNorm 的 `eps`。

```text
GPT-2   layer_norm_epsilon = 1e-5      （与本课程 day079 的 DEFAULT_EPSILON 同源）
BERT    layer_norm_eps     = 1e-12
T5      layer_norm_epsilon = 1e-6      （一并记下，作为第三个数）
```

`eps` 越小，`x̂` 的方差越接近 1（day079 第 3.2 节那条"方差是 σ²/(σ²+eps)"）。
因此"BERT 的 LayerNorm 与 GPT-2 的 LayerNorm 是同一条公式、两个默认值，
而它们的输出**并不相等**"——这件事可以被量出来，本包把它写成一张表（:mod:`study`）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.encoder_decoder.layers import add_matrices, layer_norm
from smart_research_agent.hf_source.errors import ShapeError
from smart_research_agent.hf_source.types import PROFILES, LN_EPS_DEFAULTS, profile_of
from smart_research_agent.math_foundations.types import Matrix, validate_matrix

#: BERT 侧"句子 A / 句子 B"的默认类型 id（``BertEmbeddings.token_type_ids`` 的缺省 0）.
DEFAULT_TOKEN_TYPE = 0

#: GPT-2 的位置表长度（``n_positions`` 的经典取值）；本包只用它做越界演示，不承诺它就是所有版本。
GPT2_N_POSITIONS = 1024

#: BERT 的位置表长度（``max_position_embeddings``）。
BERT_MAX_POSITION_EMBEDDINGS = 512


@dataclass(frozen=True)
class EmbeddingTables:
    """一次嵌入需要的三张表（``token_type`` 只在 BERT 侧被用到）."""

    word: Matrix
    position: Matrix
    token_type: Matrix | None = None

    @property
    def token_count(self) -> int:
        """词表大小（``wte`` 的行数）."""
        return len(self.word)

    @property
    def position_count(self) -> int:
        """位置表长度（``wpe`` 的行数）——**这就是那道硬上界**."""
        return len(self.position)

    def to_dict(self) -> dict[str, object]:
        """把三张表的形状摊平成一行字段（**不含**任何权重值）."""
        return {
            "vocab": self.token_count,
            "positions": self.position_count,
            "has_token_type": self.token_type is not None,
        }


def gather_rows(table: Matrix, indices: tuple[int, ...], *, name: str = "table") -> Matrix:
    """按行号取若干行——**越界当场拒绝**，而不是静默回绕或截断.

    这与 `nn.Embedding` 的语义一致（越界抛索引错误），只是本包把"改哪个数字"说清楚：
    越界是**调用点**的问题（``位置数超过表长``），因此归 :class:`ShapeError`。
    """
    checked = validate_matrix(table, name=name)
    if not indices:
        raise ShapeError("要取的行号为空：一次嵌入至少要有一个 token，返回空矩阵只会把错误推给下游。")
    collected: list[tuple[float, ...]] = []
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ShapeError(f"行号必须是整数，收到 {index!r}。")
        if index < 0 or index >= len(checked):
            raise ShapeError(
                f"行号 {index} 越界：{name} 只有 {len(checked)} 行。"
                "位置表的长度是一道**硬上界**（max_position_embeddings / n_positions），"
                "越界时正确做法是拒绝，而不是回绕或截断——"
                "回绕会让第 1025 个 token 拿到第 1 个位置的编码，而形状完全合法。"
            )
        collected.append(checked[index])
    return tuple(collected)


def position_ids(count: int) -> tuple[int, ...]:
    """``(0, 1, …, count-1)``——HF 在单序列前向里就是这么造 ``position_ids`` 的."""
    if count <= 0:
        raise ShapeError(f"序列长度必须为正，收到 {count}。")
    return tuple(range(count))


def default_token_types(count: int) -> tuple[int, ...]:
    """BERT 的默认 ``token_type_ids``：整条序列都是句子 A（0）."""
    if count <= 0:
        raise ShapeError(f"序列长度必须为正，收到 {count}。")
    return (DEFAULT_TOKEN_TYPE,) * count


def gpt2_embed(
    tables: EmbeddingTables,
    token_ids: tuple[int, ...],
    *,
    offset: int = 0,
) -> Matrix:
    """GPT-2 的嵌入：**两行之和，没有 LN**（``wte[token] + wpe[position]``）.

    ``offset`` 是"这一批 token 在整条序列里的起始位置"——增量解码（KV Cache）时
    它就是"已经生成了多少个 token"，因此它必须被显式传进来，
    而不是每次都从 0 开始（那样第二个 token 会拿到第一个位置的编码）。
    """
    positions = tuple(offset + index for index in range(len(token_ids)))
    words = gather_rows(tables.word, token_ids, name="word")
    place = gather_rows(tables.position, positions, name="position")
    return add_matrices(words, place)


def bert_embed(
    tables: EmbeddingTables,
    token_ids: tuple[int, ...],
    token_types: tuple[int, ...] | None = None,
    *,
    epsilon: float = LN_EPS_DEFAULTS["bert"],
) -> Matrix:
    """BERT 的嵌入：三行之和之后再**做一次 LayerNorm**（``LN(word + type + position)``）.

    ``token_type`` 表缺失时退化成"只有句子 A"——这正是 ``BertEmbeddings`` 在
    ``token_type_ids=None`` 时的行为（它把整条序列当作句子 A），
    而不是"跳过这一项加法"。
    """
    resolved_types = default_token_types(len(token_ids)) if token_types is None else token_types
    if len(resolved_types) != len(token_ids):
        raise ShapeError(
            f"token_type_ids 的个数 {len(resolved_types)} 与 token 个数 {len(token_ids)} 不一致："
            "HF 里这两者必须等长（不足时会广播，而广播出来的结果看起来正常）。"
        )
    words = gather_rows(tables.word, token_ids, name="word")
    place = gather_rows(tables.position, position_ids(len(token_ids)), name="position")
    summed = add_matrices(words, place)
    if tables.token_type is not None:
        types = gather_rows(tables.token_type, resolved_types, name="token_type")
        summed = add_matrices(summed, types)
    normalized, _cache = layer_norm(summed, epsilon=epsilon)
    return normalized


def embedding_of(
    name: str,
    tables: EmbeddingTables,
    token_ids: tuple[int, ...],
    token_types: tuple[int, ...] | None = None,
) -> Matrix:
    """按模型名分派嵌入（**画像驱动**：新增一个模型时只改 ``PROFILES``）."""
    profile = profile_of(name)
    if profile.normalize_embeddings:
        return bert_embed(tables, token_ids, token_types, epsilon=profile.ln_eps)
    if token_types is not None:
        raise ShapeError(
            f"{name} 的嵌入不接受 token_type_ids：它没有'句子 A / 句子 B'这一维。"
            "接受之后静默丢掉，会让'两个句子'的模型看起来与'一个句子'的完全一样。"
        )
    return gpt2_embed(tables, token_ids)


def embedding_note(name: str) -> str:
    """返回"这个模型的嵌入层与另一个差在哪"——直接引用画像里的两个布尔量."""
    profile = profile_of(name)
    other = "bert" if name == "gpt2" else "gpt2"
    other_profile = PROFILES[other]
    lines = [
        f"{name}（{profile.source_file}）",
        f"  位置摆放   {profile.norm_placement}（另一个模型：{other_profile.norm_placement}）",
        f"  LayerNorm eps={profile.ln_eps:g}（另一个模型：{other_profile.ln_eps:g}）",
        f"  嵌入后归一化 {profile.normalize_embeddings}（另一个模型：{other_profile.normalize_embeddings}）",
        f"  token_type  {profile.uses_token_type}（另一个模型：{other_profile.uses_token_type}）",
    ]
    return "\n".join(lines)


__all__ = [
    "BERT_MAX_POSITION_EMBEDDINGS",
    "DEFAULT_TOKEN_TYPE",
    "GPT2_N_POSITIONS",
    "EmbeddingTables",
    "bert_embed",
    "default_token_types",
    "embedding_note",
    "embedding_of",
    "gather_rows",
    "gpt2_embed",
    "position_ids",
]
