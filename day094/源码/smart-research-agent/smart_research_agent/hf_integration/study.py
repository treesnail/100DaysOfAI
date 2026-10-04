"""``study``：六张表（day086 / M7-D10）.

前十四天每天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天六张表各回答一个"接进来"的问题：

```text
缓存表      一次解析到底下了几个文件、其中几个是命中（逐文件记账）
参数量表    两个架构 × 两个口径（配置公式 / 真实库 / 本包权重 / 差值）
分词表      一段文本在**这份词表**下要几个 token（并排看中英与 emoji）
池化表      三法 × 两种填充宽度：正确池化的差恒为 0，违反 mask 的差不为 0
批量表      因果侧 vs 双向侧：填充是惰性的，而"双向侧不是"这件事有个数
生成表      四种策略各自选了什么、保留了几个候选（与 day085 的表同一取向）
```

## 一条纪律：每一行都要带**两个数**

与 day082 的探针同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：一个**读数**与一个**参照**
（命中/下载、配置/库、真实长度/总宽度、safe/naive）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.hf_integration.config import reference_cards, tiny_card
from smart_research_agent.hf_integration.features import compare_pooling, pool_rows
from smart_research_agent.hf_integration.forward import (
    ModelWeights,
    hidden_states,
    make_weights,
)
from smart_research_agent.hf_integration.hub import HubResolver
from smart_research_agent.hf_integration.pipeline import text_generation
from smart_research_agent.hf_integration.tokenizer import ByteBPETokenizer
from smart_research_agent.hf_integration.types import (
    INTEGRATION_NOTES,
    POOLING_STRATEGIES,
    POOLING_TOLERANCE,
    ModelCard,
    Snapshot,
)
from smart_research_agent.hf_source.generation import strategy_of
from smart_research_agent.hf_source.types import GenerationSettings

#: 分词表的默认样本（**四类各一条**：纯 ASCII、缩写、中文、emoji）.
TOKEN_SAMPLES: tuple[str, ...] = (
    "hello world",
    "don't stop",
    "你好，世界",
    "emoji 🙂 ok",
)

#: 池化表与批量表用的三条文本（**长度必须两两不同**，否则这一批没有填充）.
BATCH_TEXTS: tuple[str, ...] = (
    "hello world",
    "hey",
    "hello there world",
)

#: 生成表用的四条配置（每一条只动一个旋钮，因此差别可归因）.
#: 后三条都打开了 ``do_sample``——因为 ``top_k`` / ``top_p`` 只在采样时影响**选择**，
#: 贪心下它们只改变"保留几个候选"而不改变"选了谁"（day085 的 ``strategy_of`` 口径）。
GENERATION_CASES: tuple[GenerationSettings, ...] = (
    GenerationSettings(max_new_tokens=3, do_sample=False, seed=7),
    GenerationSettings(max_new_tokens=3, do_sample=True, top_k=2, seed=7),
    GenerationSettings(max_new_tokens=3, do_sample=True, temperature=0.7, seed=7),
    GenerationSettings(max_new_tokens=3, do_sample=True, top_p=0.9, seed=7),
)


@dataclass(frozen=True)
class CacheRow:
    """缓存表的一行：一个文件的四个数."""

    path: str
    size: int
    cached: bool
    pass_number: int

    def line(self) -> str:
        """``第1次  config.json      384 B  命中=否``."""
        mark = "命中" if self.cached else "新下"
        return f"第{self.pass_number}次  {self.path:<22} {self.size:>6} B  {mark}"


@dataclass(frozen=True)
class ParamRow:
    """参数量表的一行：四个口径一起看.

    ``weights`` 与 ``gap`` 是 ``None`` 的时候表示"这一行没有造权重"——
    真实模型有 1.2 亿个数，纯 Python 造一遍要几分钟，而本表要印的是**口径之间的关系**，
    不是那 1.2 亿个数本身。"差值 = 4h·L"这条性质在玩具卡片上被逐项验证（见 :mod:`verify`）。
    """

    name: str
    model_type: str
    config: int
    library: int | None
    expected_gap: int
    weights: int | None = None
    gap: int | None = None

    def line(self) -> str:
        """``gpt2  config 124 439 808 | 库 124 439 808 一致 | 期望差 4h·L 36 864 | 本包权重 —``."""
        library = "（无库读数）" if self.library is None else str(self.library)
        same = "" if self.library is None else ("一致" if self.library == self.config else "不一致")
        weights = "—" if self.weights is None else str(self.weights)
        gap = "—" if self.gap is None else str(self.gap)
        return (
            f"{self.name:<18} {self.model_type:<6} config {self.config:>10} | "
            f"库 {library:>10} {same:<6} | 期望差 4h·L {self.expected_gap:>7} | "
            f"本包权重 {weights:>10} | 实测差 {gap:>7}"
        )


@dataclass(frozen=True)
class TokenRow:
    """分词表的一行：字符数、token 数、比例、往返."""

    text: str
    characters: int
    tokens: int
    pieces: tuple[str, ...]
    round_trip: bool

    def line(self) -> str:
        """``'你好，世界'  5 字符 → 15 token（3.0 token/字符）| 往返 True``."""
        ratio = 0.0 if self.characters == 0 else self.tokens / self.characters
        head = self.pieces[:3]
        return (
            f"{self.text!r:<22} {self.characters:>3} 字符 → {self.tokens:>3} token"
            f"（{ratio:.2f} token/字符）| 往返 {self.round_trip} | 前几个 {list(head)}"
        )


@dataclass(frozen=True)
class PoolingRow:
    """池化表的一行：一个策略 × 两种填充宽度."""

    strategy: str
    gap_correct: float
    gap_buggy_short: float
    gap_buggy_long: float
    padding_waste: float

    def line(self) -> str:
        """``mean  正确版两种填充的差 0.000000e+00 | 违反版 1.2e-01 / 2.3e-01``."""
        return (
            f"{self.strategy:<11} 正确版两种填充的差 {self.gap_correct:.6e} | "
            f"违反版 {self.gap_buggy_short:.6e} / {self.gap_buggy_long:.6e} | "
            f"填充占比 {self.padding_waste:.1%}"
        )


@dataclass(frozen=True)
class BatchRow:
    """批量表的一行：一个架构上"填充与样本边界是否被正确对待"."""

    model_type: str
    safe_matches_single: bool
    naive_gap: float
    padding_waste: float
    naive_reason: str

    def line(self) -> str:
        """``gpt2  safe 与逐条一致 True | naive 的差 4.1e-01（跨样本串味）``."""
        verdict = "无差别" if self.naive_gap == 0.0 else self.naive_reason
        return (
            f"{self.model_type:<6} safe 与逐条一致 {self.safe_matches_single!s:<5} | "
            f"naive 的差 {self.naive_gap:.6e}（{verdict}）| "
            f"填充占比 {self.padding_waste:.1%}"
        )


@dataclass(frozen=True)
class GenerationRow:
    """生成表的一行：一条配置下选了什么."""

    settings: GenerationSettings
    strategy: str
    tokens: tuple[int, ...]
    generated: tuple[int, ...]
    kept: int
    text: str

    def line(self) -> str:
        """``greedy  新生成 3 个 token | 首步保留 100 个候选 | 文本 '...'``."""
        return (
            f"{self.strategy:<11} 新生成 {len(self.generated)} 个 token | "
            f"末步保留 {self.kept} 个候选 | 文本 {self.text!r}"
        )


def cache_rows(
    resolver: HubResolver,
    client: object,
    repo_id: str,
    *,
    revision: str = "main",
) -> tuple[CacheRow, ...]:
    """缓存表：**两轮解析共用一个缓存**，每一轮印出每一个文件的命中情况.

    第一轮的文件全部"新下"、第二轮全部"命中"——两行放在一起看，
    才看得出"逐文件"这三个字的含义。
    **共用一个缓存**是这条读数的前提：给两轮各一个新缓存，
    "命中"这件事就永远不会发生（而两轮看起来都正常）。
    """
    del client  # 只借 resolver 与它背后的假仓库；这里不再多拿一个引用
    target = HubResolver(hub=resolver.hub, cache_dir=resolver.cache_dir)
    first = target.resolve(repo_id, revision=revision, local_files_only=False)
    second = target.resolve(repo_id, revision=revision, local_files_only=False)
    rows: list[CacheRow] = []
    for path in first.names():
        entry = second.entry(path)
        rows.append(CacheRow(path=path, size=entry.size, cached=entry.cached, pass_number=2))
    for path in second.names():
        entry = first.entry(path)
        rows.append(CacheRow(path=path, size=entry.size, cached=entry.cached, pass_number=1))
    return tuple(sorted(rows, key=lambda row: (row.pass_number, row.path)))


def param_rows(*, weights_for_toy: bool = True) -> tuple[ParamRow, ...]:
    """参数量表：两张真实卡片 + 两张玩具卡片（四张一起看才看得出 tying 的代价）.

    真实卡片那一行**不造权重**（1.2 亿个数，纯 Python 要几分钟），
    它印的是"配置 / 库 / 期望差"三个整数——而"实测差恰好是 4h·L"这件事
    在那两张玩具卡片上真的被算了出来（本列的最后两格）。
    """
    from smart_research_agent.hf_integration.verify import LIBRARY_RECORDS

    rows: list[ParamRow] = []
    cards: list[ModelCard] = list(reference_cards().values())
    cards.extend((tiny_card("gpt2"), tiny_card("bert")))
    for card in cards:
        expected_gap = 4 * card.hidden * card.layers
        weights: int | None = None
        gap: int | None = None
        if weights_for_toy and card.name.endswith("-tiny"):
            built = make_weights(card)
            weights = built.parameter_count
            gap = card.parameter_count - built.parameter_count
        rows.append(
            ParamRow(
                name=card.name,
                model_type=card.model_type,
                config=card.parameter_count,
                library=LIBRARY_RECORDS.get(card.name),
                expected_gap=expected_gap,
                weights=weights,
                gap=gap,
            )
        )
    return tuple(rows)


def token_rows(
    tokenizer: ByteBPETokenizer,
    samples: tuple[str, ...] = TOKEN_SAMPLES,
) -> tuple[TokenRow, ...]:
    """分词表：**这份词表**下每段文本要几个 token（比例与往返一起印）."""
    rows: list[TokenRow] = []
    for text in samples:
        ids = tokenizer.encode(text)
        rows.append(
            TokenRow(
                text=text,
                characters=len(text),
                tokens=len(ids),
                pieces=tokenizer.tokenize(text),
                round_trip=tokenizer.decode(ids) == text,
            )
        )
    return tuple(rows)


def _flat_padded(
    safe_rows: tuple[tuple[tuple[float, ...], ...], ...],
    masks: tuple[tuple[int, ...], ...],
    *,
    width: int,
    hidden: int,
) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    """把安全前向的**真实行**放回一个带填充的张量（填充行用零行代表）.

    填充行在安全路径上不存在，因此"它们等于什么"是一个我们选的值：
    本表要量的是**池化分不分得清真实位置**，而不是模型对 pad id 输出什么。
    """
    zero = (0.0,) * hidden
    flat: list[tuple[float, ...]] = []
    flags: list[int] = []
    for row, mask in zip(safe_rows, masks, strict=True):
        flat.extend(row)
        flat.extend(zero for _ in range(width - len(row)))
        flags.extend(mask)
    return tuple(flat), tuple(flags)


def _pair_gap(left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]) -> float:
    """两组向量的最大绝对差（只用来回答"两种填充宽度下读数是否相同"）."""
    worst = 0.0
    for row_left, row_right in zip(left, right, strict=True):
        for a, b in zip(row_left, row_right, strict=True):
            worst = max(worst, abs(a - b))
    return worst


def pooling_rows(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    texts: tuple[str, ...] = BATCH_TEXTS,
) -> tuple[PoolingRow, ...]:
    """池化表：三法 × 两种填充宽度（正确版恒为 0，违反版不为 0）."""
    ids = [tokenizer.encode(text) for text in texts]
    real_width = max(len(row) for row in ids)
    rows: list[PoolingRow] = []
    for strategy in POOLING_STRATEGIES:
        corrects: list[tuple[tuple[float, ...], ...]] = []
        buggy_gaps: list[float] = []
        waste = 0.0
        for width in (real_width + 3, real_width + 7):
            padded = tuple(tuple(row) + (0,) * (width - len(row)) for row in ids)
            masks = tuple((1,) * len(row) + (0,) * (width - len(row)) for row in ids)
            safe = hidden_states(card, weights, padded, masks)
            flat, flags = _flat_padded(safe.rows, masks, width=width, hidden=card.hidden)
            comparison = compare_pooling(flat, flags, strategy, width=width)
            corrects.append(pool_rows(safe.rows, strategy))
            buggy_gaps.append(comparison.gap)
            waste = comparison.padding_waste
        rows.append(
            PoolingRow(
                strategy=strategy,
                gap_correct=_pair_gap(corrects[0], corrects[1]),
                gap_buggy_short=buggy_gaps[0],
                gap_buggy_long=buggy_gaps[1],
                padding_waste=waste,
            )
        )
    return tuple(rows)


def batch_rows(
    tokenizer: ByteBPETokenizer,
    texts: tuple[str, ...] = BATCH_TEXTS,
    *,
    seed: int = 7,
) -> tuple[BatchRow, ...]:
    """批量表：**同一个种子**下因果侧与双向侧各跑一次（差别只来自架构）.

    两张卡片当场按**分词器的词表大小**造出来：词表对不上时 ``encode`` 会给出
    越界的 id（那不是这一课要考的东西，而它会以 ``TokenError`` 的形式盖住真正的读数）。
    """
    ids = [tokenizer.encode(text) for text in texts]
    width = max(len(row) for row in ids)
    padded = tuple(tuple(row) + (0,) * (width - len(row)) for row in ids)
    masks = tuple((1,) * len(row) + (0,) * (width - len(row)) for row in ids)
    rows: list[BatchRow] = []
    for model_type in ("gpt2", "bert"):
        card = tiny_card(model_type, vocab=tokenizer.vocab_size)
        weights = make_weights(card, seed=seed)
        safe = hidden_states(card, weights, padded, masks)
        singles = tuple(
            hidden_states(card, weights, (row,), (mask,)).rows[0]
            for row, mask in zip(padded, masks, strict=True)
        )
        naive = hidden_states(card, weights, padded, masks, policy="naive")
        gap = 0.0
        for left_row, right_row in zip(safe.padded(), naive.padded(), strict=True):
            for a, b in zip(left_row, right_row, strict=True):
                gap = max(gap, abs(a - b))
        real = sum(sum(mask) for mask in masks)
        rows.append(
            BatchRow(
                model_type=model_type,
                safe_matches_single=safe.rows == singles,
                naive_gap=gap,
                padding_waste=1.0 - real / (len(masks) * width),
                naive_reason=(
                    "跨样本串味（naive 漏掉'段'）"
                    if card.causal
                    else "填充污染（naive 没有切开）"
                ),
            )
        )
    return tuple(rows)


def generation_rows(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    prompt: str = "hello",
    cases: tuple[GenerationSettings, ...] = GENERATION_CASES,
) -> tuple[GenerationRow, ...]:
    """生成表：四条配置各跑一次（**每一条只动一个旋钮**）."""
    rows: list[GenerationRow] = []
    for settings in cases:
        output = text_generation(card, weights, tokenizer, (prompt,), settings)
        result = output.results[0]
        rows.append(
            GenerationRow(
                settings=settings,
                strategy=strategy_of(settings),
                tokens=result.token_ids,
                generated=result.generated,
                kept=result.steps[-1].kept if result.steps else card.vocab,
                text=output.texts[0],
            )
        )
    return tuple(rows)


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """生态笔记逐行印出（十二条，顺序即写入顺序）."""
    items = tuple(INTEGRATION_NOTES.items())
    if limit is not None:
        items = items[:limit]
    return tuple(f"{index:>2}. {value}" for index, (_key, value) in enumerate(items, start=1))


def study_lines(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    snapshot: Snapshot,
    resolver: HubResolver,
) -> tuple[str, ...]:
    """一次跑完六张表（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append(f"== 1. 缓存表：{snapshot.repo_id}@{snapshot.revision}（commit {snapshot.commit[:8]}）")
    for row in cache_rows(resolver, resolver.hub, snapshot.repo_id, revision=snapshot.revision):
        lines.append("  " + row.line())
    lines.append("== 2. 参数量表（config / 库 / 本包权重 / 差）")
    for row in param_rows():
        lines.append("  " + row.line())
    lines.append("== 3. 分词表（这份词表下要几个 token）")
    for row in token_rows(tokenizer):
        lines.append("  " + row.line())
    lines.append(f"== 4. 池化表（容差 {POOLING_TOLERANCE:.0e}）")
    for row in pooling_rows(card, weights, tokenizer):
        lines.append("  " + row.line())
    lines.append("== 5. 批量表（填充与样本边界在两种架构下各自要什么）")
    for row in batch_rows(tokenizer):
        lines.append("  " + row.line())
    lines.append("== 6. 生成表（四条配置各动一个旋钮）")
    for row in generation_rows(card, weights, tokenizer):
        lines.append("  " + row.line())
    return tuple(lines)


__all__ = [
    "BATCH_TEXTS",
    "GENERATION_CASES",
    "TOKEN_SAMPLES",
    "BatchRow",
    "CacheRow",
    "GenerationRow",
    "ParamRow",
    "PoolingRow",
    "TokenRow",
    "batch_rows",
    "cache_rows",
    "generation_rows",
    "note_lines",
    "param_rows",
    "pooling_rows",
    "study_lines",
    "token_rows",
]
