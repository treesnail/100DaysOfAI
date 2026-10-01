"""十条性质与三类对账（day086 / M7-D10）.

本模块是"接进来了吗"的判据所在。它有三类，而**三类用不同的比法**：

```text
逐位（``==``）    两个量来自同一次算术：池化在两种填充下的结果、
                  批量与逐条的结果、encode/decode 往返、
                  本包的一层与 day085 的那个块
整数相等          参数量是两个整数：本包的公式 vs 库的读数、权重 vs 配置口径
逐名对齐          接口面是一串名字：在进程模型 vs LocalModel
```

day085 那条纪律在这里换了形式：**两份东西之间的差异要么被消灭、要么被写下来。**
今天的"两份东西"不再是两份实现，而是**两个来源**——
配置与源码、本地与远端、批量与单条、正确与故意违例。

## 三条跨天对账

```text
画像 vs 配置      day085 从源码读出的默认值 ↔ 今天从 config.json 解析出来的卡片
生成策略          pipeline 的生成 ↔ day085 的 ``generate``（同一个 token 序列）
块的接线          本包的一层 ↔ day085 的 ``gpt2_block``（**逐位**）
```

第三条最强：两条路径的差别只有一处（加性偏置 vs 显式掩码），
而它们必须在每一个浮点数上一致——接线走散时形状完全看不出来。

## "不适用"仍然必须与"通过"分开

``PropertyOutcome`` 沿用 day085 的契约：``applicable=False`` 时 ``passed`` 也必须是
``False``（构造期就拒绝）。今天有**两条**真的会走到"不适用"：

```text
参数量对账        只对两个**记录在案**的真实模型适用（玩具卡片没有库读数可对）
块对账            只在 L=1 且 heads=1 的 GPT-2 卡片上适用
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.hf_integration.bridge import (
    PROTOCOL_SURFACE,
    InProcessModel,
)
from smart_research_agent.hf_integration.config import reference_cards
from smart_research_agent.hf_integration.errors import AssemblyError, NumericError
from smart_research_agent.hf_integration.features import (
    compare_pooling,
    max_gap,
    pool_masked,
)
from smart_research_agent.hf_integration.forward import (
    ModelWeights,
    embed_tokens,
    hidden_states,
    make_logits_fn,
    run_one_block,
)
from smart_research_agent.hf_integration.hub import HubResolver, flat_paths
from smart_research_agent.hf_integration.pipeline import (
    check_head_available,
    text_generation,
)
from smart_research_agent.hf_integration.tokenizer import ByteBPETokenizer
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_GPT2,
    BERT_BASE_PARAMETERS,
    GPT2_SMALL_PARAMETERS,
    INTEGRATION_PROPERTIES,
    PROPERTY_BATCH_MATCHES_SINGLE,
    PROPERTY_BLOCK_MATCHES_DAY085,
    PROPERTY_CACHE_IS_PER_FILE,
    PROPERTY_GENERATION_MATCHES_DAY085,
    PROPERTY_PARAMS_MATCH_LIBRARY,
    PROPERTY_POOLING_RESPECTS_MASK,
    PROPERTY_PROTOCOL_SURFACE_MATCHES,
    PROPERTY_ROUND_TRIP_IS_EXACT,
    PROPERTY_SNAPSHOT_IS_FLAT,
    PROPERTY_WEIGHTS_GAP_IS_EXPLAINED,
    TASK_TEXT_GENERATION,
    ModelCard,
    Snapshot,
)
from smart_research_agent.hf_source.blocks import gpt2_block
from smart_research_agent.hf_source.generation import generate
from smart_research_agent.hf_source.types import GenerationSettings, SourceShape

#: 池化对账的容差（与 ``types.POOLING_TOLERANCE`` 同值，这里给一个本地别名便于阅读）.
POOLING_TOLERANCE = 1e-12

#: 库记录值的来源（写进证据行，否则"与库一致"这句话无法被追溯）.
LIBRARY_RECORD_SOURCE = "huggingface/transformers 5.17.0 实测（本机 venv）"

#: 两份真实模型的读数（**这两条是"与库一致"可被反驳的形式**）.
LIBRARY_RECORDS: dict[str, int] = {
    "gpt2": GPT2_SMALL_PARAMETERS,
    "bert-base-uncased": BERT_BASE_PARAMETERS,
}

#: 默认的往返样本（含中文、emoji、缩写、标点：它们分别会踩到字节级与预分词那两条）。
ROUND_TRIP_SAMPLES: tuple[str, ...] = (
    "hello world",
    "don't stop",
    "你好，世界",
    "emoji 🙂 与符号 →",
    "a1b2c3",
    "  leading and trailing  ",
    "hello,world!",
    "tab\tand\nnewline",
)


@dataclass(frozen=True)
class CrossCheck:
    """两个**来源**之间的一次对账：来源、读数、判据.

    与 day085 的同类只有一个区别：今天的"右边"常常不是另一份实现，
    而是一个**记录下来的库读数**（或一份配置文件）。因此 ``right`` 里
    必须写清"这个数是从哪来的"。
    """

    name: str
    left: str
    right: str
    reading: float | int
    expected: float | int
    exact: bool = True

    @property
    def passed(self) -> bool:
        """逐位（``exact``）或容差（非 ``exact``）是否通过."""
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= POOLING_TOLERANCE

    def line(self) -> str:
        """一行可读的读数（**带上两个来源**，否则这个数无法被追溯）."""
        verdict = "一致" if self.passed else "不一致"
        return (
            f"[{verdict}] {self.name}: {self.left} vs {self.right} | "
            f"读数 {self.reading} / 期望 {self.expected}"
            f"{'' if self.exact else '（容差 1e-12）'}"
        )


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：适用性 + 结果 + 证据行."""

    name: str
    applicable: bool
    passed: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)
    cross_check: CrossCheck | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.passed:
            raise NumericError(
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开——"
                "否则'把这条检查删掉'与'它通过了'在报告里长得一模一样。"
            )

    def line(self) -> str:
        """一行可读的结论（不适用也要印出来）."""
        if not self.applicable:
            return f"[不适用] {self.name} | {'；'.join(self.evidence)}"
        verdict = "通过" if self.passed else "失败"
        detail = "；".join(self.evidence)
        suffix = f" | {detail}" if detail else ""
        return f"[{verdict}] {self.name}{suffix}"


@dataclass(frozen=True)
class PropertyReport:
    """一组性质的报告（``ok`` 要求**所有适用**的都通过）."""

    outcomes: tuple[PropertyOutcome, ...]

    @property
    def applicable(self) -> tuple[PropertyOutcome, ...]:
        """适用（``applicable=True``）的那些性质."""
        return tuple(outcome for outcome in self.outcomes if outcome.applicable)

    @property
    def ok(self) -> bool:
        """是否全部通过——不适用不算通过、也不算失败."""
        return all(outcome.passed for outcome in self.applicable)

    def require_ok(self) -> None:
        """不通过时抛错（消息里带上失败那几条的原文）."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise AssemblyError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**，让"这一次没查它"一眼可见）."""
        return tuple(
            outcome.line()
            for outcome in sorted(self.outcomes, key=lambda outcome: outcome.applicable)
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "ok": self.ok,
            "counts": {"total": len(self.outcomes), "applicable": len(self.applicable)},
            "lines": list(self.lines()),
        }


def check_cache_is_per_file(
    resolver: HubResolver,
    repo_id: str,
    *,
    revision: str = "main",
    fresh: bool = True,
) -> PropertyOutcome:
    """**逐文件**命中：第二次解析同一个 commit 时，下载字节数恰好为 0.

    它以"先联网一次、再联网一次"的形式断言，而不是"断网也能跑"——
    后者只证明缓存**存在**，不证明它是**逐文件**复用的。

    ``fresh=True`` 会用一份**空缓存**从零开始。这一步不能省：
    拿一个已经填好的 store 去跑，"第一次"本身就是一次命中，
    于是"下载 0 个"这句话**不可能失败**——那正是 day085 记过的
    "不可能失败的读数也要印出来"的反面：一条不可能失败的判据不该被当成判据。
    """
    target = (
        HubResolver(hub=resolver.hub, cache_dir=resolver.cache_dir) if fresh else resolver
    )
    first = target.resolve(repo_id, revision=revision, local_files_only=False)
    second = target.resolve(repo_id, revision=revision, local_files_only=False)
    offline = target.resolve(repo_id, revision=revision)
    check = CrossCheck(
        name="第二次解析的下载量",
        left="第二次 resolve(local_files_only=False)",
        right="逐文件命中（下载 0 个）",
        reading=second.downloaded_count,
        expected=0,
    )
    return PropertyOutcome(
        name=PROPERTY_CACHE_IS_PER_FILE,
        applicable=True,
        passed=check.passed and offline.downloaded_count == 0 and first.downloaded_count > 0,
        evidence=(
            f"首次：下 {first.downloaded_count} 个 / 命中 {first.cached_count} 个",
            f"再次：下 {second.downloaded_count} 个 / 命中 {second.cached_count} 个",
            f"离线：下 {offline.downloaded_count} 个 / 命中 {offline.cached_count} 个",
            f"实体文件 {target.store.blob_count} 个、共 {target.store.total_bytes()} 字节",
        ),
        cross_check=check,
    )


def check_snapshot_is_flat(snapshot: Snapshot) -> PropertyOutcome:
    """快照里的相对路径与仓库一致（**不带** blobs / snapshots 前缀）.

    "平铺"这件事的可断言形式是：``flat_paths`` 与全部文件名**逐项相等**。
    只要有一个名字落在 ``blobs/`` 里，``join(root, name)`` 就找不到它——
    而那时 ``config.json`` 的缺席会被报成"这个仓库里没有配置文件"，指错方向。
    """
    names = snapshot.names()
    flat = flat_paths(snapshot)
    return PropertyOutcome(
        name=PROPERTY_SNAPSHOT_IS_FLAT,
        applicable=True,
        passed=flat == names and len(names) > 0,
        evidence=(
            f"快照里共 {len(names)} 个文件：{list(names)}",
            f"平铺（无内部前缀）的有 {len(flat)} 个",
            f"root = {snapshot.root}",
        ),
    )


def check_params_match_library(card: ModelCard) -> PropertyOutcome:
    """参数量公式与真实库的读数**整数相等**（只对两个记录在案的模型适用）.

    两张卡片之外的模型**不适用**——因为"与库一致"这句话需要**一个库读数**，
    而本包只为两个真实模型记录了读数（其余卡片没有可对的数）。
    让它"通过"的后果是：报告里那句"参数量与库一致"会对一张玩具卡片也成立，
    而玩具卡片一共只有 8704 个参数。
    """
    record = LIBRARY_RECORDS.get(card.name)
    if record is None:
        return PropertyOutcome(
            name=PROPERTY_PARAMS_MATCH_LIBRARY,
            applicable=False,
            passed=False,
            evidence=(
                f"{card.name} 不在记录表里（只有 {sorted(LIBRARY_RECORDS)} 有库读数）："
                "没有库读数时'与库一致'这句话无法被反驳",
            ),
        )
    check = CrossCheck(
        name=f"{card.name} 的参数量",
        left="本包的公式（由 config.json 算出）",
        right=LIBRARY_RECORD_SOURCE,
        reading=card.parameter_count,
        expected=record,
    )
    return PropertyOutcome(
        name=PROPERTY_PARAMS_MATCH_LIBRARY,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"配置口径 {card.parameter_count} vs 库 {record}",
            f"来源：{LIBRARY_RECORD_SOURCE}",
        ),
        cross_check=check,
    )


def check_round_trip_is_exact(
    tokenizer: ByteBPETokenizer,
    samples: tuple[str, ...] = ROUND_TRIP_SAMPLES,
) -> PropertyOutcome:
    """``decode(encode(text)) == text`` 对每一条样本都必须成立（**逐位**）.

    字节级 BPE 是无损的，因此这条判据**没有容差**。它对中文与 emoji 尤其关键：
    两者在 UTF-8 下都是多字节，丢掉字节级映射时**先坏的就是它们**。
    """
    bad = [text for text in samples if not tokenizer.round_trip(text)]
    counts = tuple(len(tokenizer.encode(text)) for text in samples)
    return PropertyOutcome(
        name=PROPERTY_ROUND_TRIP_IS_EXACT,
        applicable=True,
        passed=not bad,
        evidence=(
            f"{len(samples)} 条样本，往返全部还原：{'是' if not bad else f'否（{bad}）'}",
            f"token 数：{list(counts)}（词表 {tokenizer.vocab_size}，基元 {tokenizer.base_size}）",
        ),
    )


def check_pooling_respects_mask(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    texts: tuple[str, ...],
    *,
    strategy: str = "mean",
) -> PropertyOutcome:
    """把填充长度改掉，池化结果**逐位不变**；同时把"违反 mask 的代价"量出来.

    这条判据的两个量来自**同一次算术**（同一批数据、同一份权重），
    因此用 ``==`` 而不是容差：只对真实位置求和时，填充多少**一个字都不改**。
    证据行里额外印出"违反版"的差——那是这条规矩的必要性证明。
    """
    if len(texts) < 2:
        raise NumericError(
            "池化掩码性质需要至少两条长度不同的文本："
            "长度全相同时这一批没有填充，等于什么都没查。"
        )
    ids = [tokenizer.encode(text) for text in texts]
    real_width = max(len(row) for row in ids)
    lengths = tuple(len(row) for row in ids)
    short = real_width + 3
    long = real_width + 7

    def padded_at(width: int) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
        """安全前向之后，把**真实行**放回一个带填充的张量（填充行用零行代表）.

        填充行在安全路径上根本不存在，因此"它们等于什么"是一个我们选的值。
        本判据要量的是**池化分不分得清真实位置**，不是"模型对 pad id 输出什么"——
        把两者混在一起，读到的差就无法归因（day070 起那条纪律）。
        """
        rows = tuple(tuple(row) + (0,) * (width - len(row)) for row in ids)
        masks = tuple((1,) * len(row) + (0,) * (width - len(row)) for row in ids)
        safe = hidden_states(card, weights, rows, masks)
        zero = (0.0,) * card.hidden
        flat: list[tuple[float, ...]] = []
        flags: list[int] = []
        for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
            real = safe.rows[index]
            flat.extend(real)
            flat.extend(zero for _ in range(width - len(real)))
            flags.extend(mask)
        return tuple(flat), tuple(flags)

    flat_short, mask_short = padded_at(short)
    flat_long, mask_long = padded_at(long)
    correct_short, _evidence = pool_masked(flat_short, mask_short, strategy, width=short)
    correct_long, _ = pool_masked(flat_long, mask_long, strategy, width=long)
    gap = max_gap(correct_short, correct_long)
    comparison = compare_pooling(flat_short, mask_short, strategy, width=short)
    comparison_long = compare_pooling(flat_long, mask_long, strategy, width=long)
    check = CrossCheck(
        name=f"{strategy} 池化在两种填充宽度下的读数",
        left=f"填充到 {short}（真实 {real_width}）",
        right=f"填充到 {long}（真实 {real_width}）",
        reading=gap,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_POOLING_RESPECTS_MASK,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"各条真实长度 {list(lengths)}，填充宽度 {short} 与 {long}",
            f"正确池化在两种填充下的最大差 {gap:.6e}（**逐位**要求 0）",
            f"违反 mask 的代价：宽 {short} 时 {comparison.gap:.6e}"
            f"（填充占比 {comparison.padding_waste:.1%}）、"
            f"宽 {long} 时 {comparison_long.gap:.6e}",
        ),
        cross_check=check,
    )


def check_batch_matches_single(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    texts: tuple[str, ...],
) -> PropertyOutcome:
    """批量前向与逐条前向**逐位一致**（safe 策略下：因果靠段内掩码、双向靠先切开）.

    同时印出 naive 策略的差——那是"为什么安全策略是必要的"的唯一证据。
    两种架构的 naive 错法**不同**，而这一点必须写下来，否则读到的差无法归因：

    ```text
    因果卡片   naive 漏掉"段"⇒ 第 2 条样本看到第 1 条的 token（跨样本串味）
    双向卡片   naive 不切开  ⇒ 填充改写真实位置那一行（填充污染）
    ```
    """
    if len(texts) < 2:
        raise NumericError("批量性质需要至少两条样本（一条样本没有'批'这回事）。")
    ids = [tokenizer.encode(text) for text in texts]
    width = max(len(row) for row in ids)
    rows = tuple(tuple(row) + tuple(0 for _ in range(width - len(row))) for row in ids)
    masks = tuple(tuple(1 for _ in row) + tuple(0 for _ in range(width - len(row))) for row in ids)

    batch = hidden_states(card, weights, rows, masks)
    singles = tuple(
        hidden_states(card, weights, (row,), (mask,)).rows[0]
        for row, mask in zip(rows, masks, strict=True)
    )
    same = batch.rows == singles
    naive = hidden_states(card, weights, rows, masks, policy="naive")
    naive_gap = max_gap(batch.padded(), naive.padded())
    reason = "跨样本串味（漏掉'段'）" if card.causal else "填充污染（没有切开）"
    check = CrossCheck(
        name="批量 vs 逐条",
        left=f"批量（{len(texts)} 条，宽 {width}）",
        right="逐条（每条各自真实长度）",
        reading=0.0 if same else 1.0,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_BATCH_MATCHES_SINGLE,
        applicable=True,
        passed=same,
        evidence=(
            f"{card.model_type}：批量与逐条逐位一致：{same}",
            f"naive 策略与 safe 的最大差 {naive_gap:.6e}（归因：{reason}）",
        ),
        cross_check=check,
    )


def check_generation_matches_day085(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    settings: GenerationSettings | None = None,
) -> PropertyOutcome:
    """本包的生成与 day085 的 ``generate`` 落在**同一个 token 序列**上.

    "本包的生成"指走 :func:`~smart_research_agent.hf_integration.pipeline.text_generation`
    那条路（含分词、配置填充、解码），"day085 的生成"指直接调 ``generate``。
    两者**必须逐位相同**——因为它们本来就共用同一份策略实现，
    而这条判据抓的是"管线在透传 settings 时偷偷改了东西"
    （忘了透传 top_k、把温度重置成 1.0、换了一个种子）。
    """
    check_head_available(card, TASK_TEXT_GENERATION, weights)
    resolved = settings or GenerationSettings(max_new_tokens=4, do_sample=False, top_k=3, seed=11)
    output = text_generation(card, weights, tokenizer, (prompt,), resolved)
    from dataclasses import replace

    pipeline_settings = replace(resolved, eos_token=tokenizer.eos_id)
    direct = generate(
        make_logits_fn(card, weights),
        tuple(tokenizer.encode(prompt)),
        pipeline_settings,
        vocab=card.vocab,
    )
    same = output.results[0].token_ids == direct.token_ids
    check = CrossCheck(
        name="pipeline.text_generation vs hf_source.generate",
        left="pipeline（分词 + 透传 settings + 解码）",
        right="hf_source（同一个 logits_fn、同一份 settings）",
        reading=0.0 if same else 1.0,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_GENERATION_MATCHES_DAY085,
        applicable=True,
        passed=same,
        evidence=(
            f"token 序列 {'一致' if same else '不一致'}：{list(output.results[0].token_ids)}",
            f"策略 {output.strategy} | 新生成 {len(output.results[0].generated)} 个 token",
            f"解码出的文本 {output.texts[0]!r}",
        ),
        cross_check=check,
    )


def check_weights_gap_is_explained(card: ModelCard, weights: ModelWeights) -> PropertyOutcome:
    """本包权重与配置口径的参数量之差**恰好**是注意力偏置 ``4·hidden·layers``.

    这条判据兑现的是 day080 那句"参数量差值必须被逐项解释"：
    差值不是一个"没对上"的余数，而是一个**能被拆开**的整数
    （GPT-2 是 c_attn 的 3h 与 c_proj 的 h；BERT 是四次投影各 h）。
    """
    hidden = card.hidden
    check = CrossCheck(
        name=f"{card.model_type} 的参数量口径差",
        left=f"配置口径 {card.parameter_count} − 权重 {weights.parameter_count}",
        right=f"4·hidden·layers = 4·{hidden}·{card.layers}",
        reading=card.parameter_count - weights.parameter_count,
        expected=4 * hidden * card.layers,
    )
    return PropertyOutcome(
        name=PROPERTY_WEIGHTS_GAP_IS_EXPLAINED,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"配置 {card.parameter_count} / 权重 {weights.parameter_count} / "
            f"差 {card.parameter_count - weights.parameter_count}",
            f"每层配置 12h²+13h、权重 12h²+9h，差 4h = 注意力偏置（逐项：见 forward.BIAS_BREAKDOWN）",
        ),
        cross_check=check,
    )


def check_protocol_surface_matches(model: InProcessModel) -> PropertyOutcome:
    """在进程模型与 ``LocalModel`` 的**方法面逐名对齐**.

    这是"接进来"这件事的可断言形式：上层代码只依赖这八个名字，
    因此**名字齐了就等于接上了**（语义由各自的返回值保证）。
    反过来，少一个 ``describe`` 的后果是监控里安静地少一行读数——
    而"少一行"与"这一项没问题"读起来一样。
    """
    from smart_research_agent.llm.local_model import LocalModel

    missing_here = [
        name for name in PROTOCOL_SURFACE if not hasattr(model, name)
    ]
    missing_there = [
        name for name in PROTOCOL_SURFACE if not hasattr(LocalModel, name)
    ]
    check = CrossCheck(
        name="方法面逐名对齐",
        left="InProcessModel",
        right="llm.local_model.LocalModel",
        reading=len(missing_here) + len(missing_there),
        expected=0,
    )
    return PropertyOutcome(
        name=PROPERTY_PROTOCOL_SURFACE_MATCHES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"清单（{len(PROTOCOL_SURFACE)} 项）：{list(PROTOCOL_SURFACE)}",
            f"在进程缺 {missing_here}；LocalModel 缺 {missing_there}",
            f"上下文窗口来源：在进程 = 位置表 {model.context_length}",
        ),
        cross_check=check,
    )


def check_block_matches_day085(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    text: str,
) -> PropertyOutcome:
    """单条样本、因果、同一份参数下，本包的一层与 day085 的 ``gpt2_block`` **逐位**相同.

    这是本课的**第二条跨天对账**，而它比第一条更强：day085 的 ``gpt2_block`` 走的是
    ``causal=True`` 那条路（内部造一张加性偏置），本包走的是**显式掩码**那条路
    （:func:`~smart_research_agent.hf_integration.forward.block_causal_mask`）。
    两条不同的路径给出**同一个浮点数**——这才说明"接线顺序没有走散"，
    而接线走散（LN 放错一侧、残差少加一次）在形状上完全看不出来。

    ``heads=1`` 才适用：day085 的块与 ``multi_head`` 的单头实现共用同一份账。
    """
    if card.model_type != ARCHITECTURE_GPT2:
        return PropertyOutcome(
            name=PROPERTY_BLOCK_MATCHES_DAY085,
            applicable=False,
            passed=False,
            evidence=(
                f"{card.model_type} 卡片：day085 的 ``gpt2_block`` 只实现 pre-LN 因果那一侧",
            ),
        )
    if card.layers != 1 or card.heads != 1:
        return PropertyOutcome(
            name=PROPERTY_BLOCK_MATCHES_DAY085,
            applicable=False,
            passed=False,
            evidence=(
                f"这一次 L={card.layers}、heads={card.heads}："
                "对账要求 L=1 且 heads=1（day085 的块是单层、且只与单头对得上）",
            ),
        )
    ids = tokenizer.encode(text)
    states = embed_tokens(card, weights, (tuple(ids),))
    mine = run_one_block(card, weights, 0, states, causal=True)
    block_params, attention_params = weights.layer_of(0)
    shape = SourceShape(
        tokens=len(ids),
        hidden=card.hidden,
        heads=card.heads,
        vocab=card.vocab,
        causal_default=True,
    )
    reference = gpt2_block(
        block_params, attention_params, states, shape, activation=card.activation, causal=True
    ).output
    same = mine == reference
    check = CrossCheck(
        name="本包的一层 vs day085 的 gpt2_block",
        left="hf_integration.forward.run_one_block（显式掩码）",
        right="hf_source.blocks.gpt2_block（causal=True）",
        reading=0.0 if same else 1.0,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_BLOCK_MATCHES_DAY085,
        applicable=True,
        passed=same,
        evidence=(
            f"L=1、heads=1、pre-LN：逐位相同 {same}",
            "两条路径的差异只有一处：day085 造一张加性偏置、本包给一张显式掩码",
        ),
        cross_check=check,
    )


def check_all(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    resolver: HubResolver,
    snapshot: Snapshot,
    model: InProcessModel,
    *,
    texts: tuple[str, ...],
    prompt: str,
) -> PropertyReport:
    """一次跑完十条性质（顺序与 :data:`types.INTEGRATION_PROPERTIES` 一致）.

    两条跨天对账（生成策略的来源、块的接线）都在其中；其中块那条
    只在 ``L=1 且 heads=1`` 的 GPT-2 卡片上适用——**不适用也要印出来**。
    """
    outcomes = (
        check_cache_is_per_file(resolver, snapshot.repo_id, revision=snapshot.revision),
        check_snapshot_is_flat(snapshot),
        check_params_match_library(card),
        check_round_trip_is_exact(tokenizer),
        check_pooling_respects_mask(card, weights, tokenizer, texts),
        check_batch_matches_single(card, weights, tokenizer, texts),
        check_generation_matches_day085(card, weights, tokenizer, prompt),
        check_weights_gap_is_explained(card, weights),
        check_protocol_surface_matches(model),
        check_block_matches_day085(card, weights, tokenizer, prompt),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(INTEGRATION_PROPERTIES)
    missing = set(INTEGRATION_PROPERTIES) - names
    if extra or missing:
        # 复用一个本包已有的失败族（不新造异常类型：族越多、出路越难写清）
        raise AssemblyError(
            "性质名单与 types.INTEGRATION_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
            "名单对不上时，报告里那几行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=tuple(outcomes))


def reference_agreement_lines() -> tuple[str, ...]:
    """两条"配置 ↔ 源码"的对账（day085 的画像 vs 今天的配置）.

    这是本课的**第三条跨天对账**：day085 的画像来自 ``modeling_*.py`` 的默认值，
    今天的卡片来自 ``config.json``。两者必须说同一件事——
    否则"我读的那份源码"与"我装的那个模型"不是同一个东西，而它们的读数**看起来都对**。
    """
    from smart_research_agent.hf_integration.config import profile_agreement

    lines: list[str] = []
    for name, card in reference_cards().items():
        agreement = profile_agreement(card)
        lines.append(
            f"{name}: 激活 {agreement['activation']} | eps {agreement['ln_eps']} | "
            f"融合QKV {agreement['fused_qkv']} | token_type {agreement['uses_token_type']} | "
            f"两侧一致：{agreement['agreed']}"
        )
    return tuple(lines)


def finite_or_raise(values: tuple[float, ...], *, name: str) -> None:
    """校验一串数全为有限（报告端的最后一道闸）."""
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise NumericError(f"{name} 的第 {index} 个值不是有限数：{value!r}。")


__all__ = [
    "LIBRARY_RECORDS",
    "LIBRARY_RECORD_SOURCE",
    "POOLING_TOLERANCE",
    "ROUND_TRIP_SAMPLES",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_batch_matches_single",
    "check_block_matches_day085",
    "check_cache_is_per_file",
    "check_generation_matches_day085",
    "check_params_match_library",
    "check_pooling_respects_mask",
    "check_protocol_surface_matches",
    "check_round_trip_is_exact",
    "check_snapshot_is_flat",
    "check_weights_gap_is_explained",
    "finite_or_raise",
    "reference_agreement_lines",
]
