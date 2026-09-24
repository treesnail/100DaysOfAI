"""融合与去重：两路的名次怎么合成一份名单（M6-D6）.

## 融合要解决的第一件事：**跨通道的分数量纲不可比**

```text
向量路   余弦相似度 ∈ [-1, 1]     "0.82" 是一个有绝对上界的数
关键词路 BM25 ∈ [0, ∞)            "7.31" 是一个**没有上界**的数
```

两个数字放在一起比较是**类型错误**，而它不会报错：``0.82`` 与 ``7.31`` 都是浮点数，
``sorted`` 照单全收。于是"加权求和"这个看起来最自然的融合方式会把
**量纲差当成相关性差**——BM25 只要分数普遍偏大（语料里的词频一高就会），
它就会单方面决定名单顺序，而 alpha 那个权重根本没起作用。

因此本模块给两种策略，它们的差别正是"怎么处理量纲"：

```text
rrf        Reciprocal Rank Fusion：**完全不看分数**，只看名次
           贡献 = 1 / (k + rank + 1)，两路的量纲差异在定义上就不存在
weighted   先对每一路做 min-max 归一化到 [0, 1]，再加权求和
           归一化把"量纲"换成"这一路内部相对位置"，于是权重才有意义
```

## RRF 为什么是稳的（以及它牺牲了什么）

``rrf`` 的贡献只跟名次有关，因此它有三个好性质：**免标定**（不需要知道分数的
分布，换个编码器、换份语料都不用重调）、**免归一化**（不存在除零与上下界问题）、
**抗异常值**（某一路给出一条分数虚高 100 倍的记录，它在 RRF 里只是"第 1 名"）。

代价同样要说清楚，否则它会变成一个"永远该选 rrf"的教条：

```text
丢掉"差多少"   第 1 名与第 2 名差 0.001 还是差 0.5，RRF 一视同仁
丢掉了"只有一路有"这个信号   某条被两路同时命中（最强证据）与
                            分别被两路各自命中一次的**两条不同记录**，
                            在 RRF 里得分相近（都是两个 1/(k+r) 之和）
```

第 2 条正是 ``weighted`` 存在的理由之一，也是本模块把两路证据
（``channel_ranks`` / ``channel_scores`` / ``contributions``）逐条留下的理由：
**融合之后，"为什么它排这个位置"必须还能被回答**（见 ``FusedHit``）。

``k`` 的缺省 60 出自 RRF 的原论文（Cormack et al., 2009）。
它的作用是**压平头部差距**：``k=60`` 时第 1 名贡献 ``1/61 ≈ 0.0164``、
第 2 名 ``1/62 ≈ 0.0161``（差 1.6%），而 ``k=1`` 时是 ``1/2`` 与 ``1/3``（差 33%）。
k 越小越"只信排名第一的那条"，越大越接近"看谁被更多路提到"。

## 去重的语义：同一条出现在多路，不是重复，是**最强的证据**

"去重"这个词容易让人以为目标是"把重复的去掉，让名单短一点"。在这里它的含义
恰好相反：**同一条记录被两路独立召回，是融合能给出的最有价值的一条结论**
（两条互相独立的检索路径都指向它）。因此合并时不是"丢掉一条"，而是：

```text
一条命中          合并成一条 FusedHit，**保留全部通道的证据**
分数              两路贡献之和（rrf 是两个 1/(k+r+1)；weighted 是两个加权归一化分）
名次              两路里最好的那个（best_rank）——"两路里最强的那个说法"就是它的名次
channels          按名次好坏排序的通道名（因此"两路都召回了它"一眼可见）
```

``RetrievalResult.candidates``（混合模式下的口径）就是这条语义的算术表达：
它是**两路候选的并集大小**，因此"两路之和 − 并集"正好等于被合并掉的重复条数。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.retrieval.errors import FusionError
from smart_research_agent.retrieval.types import CHANNEL_BM25, CHANNEL_VECTOR, CHANNELS

#: 倒数名次融合（RRF）。字符串值会出现在报告与 ``RetrievalResult.fusion`` 里，
#: 因此它是一份**协议**而不是一个内部枚举（改它等于改了报告的口径）。
FUSION_RRF = "rrf"

#: 归一化加权融合。名字里刻意不含"分数"两个字：它加权的是**归一化之后**的值，
#: 而不是原始分数——把这一点写进名字，能少一次"为什么我的 alpha 不起作用"的排查。
FUSION_WEIGHTED = "weighted"

#: 全部策略（顺序 = 报告里的顺序）。``RetrievalQuery.extra`` 与端点的 strategy
#: 都必须落在这个元组里，否则 ``fuse`` 当场报 ``FusionError``。
FUSION_STRATEGIES: tuple[str, ...] = (FUSION_RRF, FUSION_WEIGHTED)

#: RRF 的平滑常数。缺省 60 出自 Cormack、Clarke & Buettcher（2009）的原文
#: （``k=60`` 是他们报告里效果最稳的取值），理由见模块 docstring。
DEFAULT_RRF_K = 60

#: ``weighted`` 策略下**向量通道**的权重缺省值（关键词通道是 ``1 - alpha``）。
#: 0.5 是"两边都不占先"的起点：它不代表两路一样有用，只代表**还没有证据**
#: 说明该偏向哪一边——先跑一遍 ``rrf`` 看清两路各自贡献了哪些条，再决定调它。
DEFAULT_ALPHA = 0.5

#: ``fuse`` 允许从 ``RetrievalQuery.extra`` 读到的键（见 ``hybrid``）。
#: **封闭清单**：多一个键就报错，理由与 ``RagPipeline.OVERRIDE_KEYS`` 相同——
#: 静默忽略一个覆盖参数会让调用方以为它生效了（"我把 alpha 拼成 alhpa，
#: 结果看起来只是没有变化"）。
FUSION_OVERRIDE_KEYS: tuple[str, ...] = ("strategy", "alpha", "k_rrf", "weights")


@dataclass(frozen=True)
class FusedHit:
    """融合后的一条命中：**排序结果 + 它为什么排在这里的全部输入**.

    七个字段里只有一个（``score``）是能不能排序的问题，其余六个都是
    "这条凭什么在这里"的问题——这正是本课的主题：一个只会
    ``sorted(hits, key=score)`` 的融合器，在"这条为什么第一"面前是哑的。

    ```text
    record_id       哪一条
    score           融合后的分数（策略定义的口径，不是一个"相关性"）
    rank            融合后的名次（0 起连续）
    channels        被哪几路召回（**按名次好坏排序**）——多路命中一眼可见
    channel_ranks   每一路给它的名次（{"vector": 3, "bm25": 0}）
    channel_scores  每一路的**原始**分数（"两路的分数量纲不同"因此可以对着看）
    contributions   每一路对最终分数的贡献（两条一起解释"为什么是 0.0315"）
    ```

    ``contributions`` 是这份形状存在的核心理由。没有它时，"向量 alpha=0.3
    的结果为什么与 alpha=0.7 一样"这个问题只能靠重跑两次来回答；
    有它时，读一眼 ``{"vector": 0.6, "bm25": 0.0}`` 就知道关键词路
    这次**什么都没贡献**（那种情况要调的不是 alpha，而是查询说法或语料）。
    """

    record_id: str
    score: float
    rank: int
    channels: tuple[str, ...] = ()
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise FusionError(
                f"FusedHit.record_id 必须是非空字符串，收到 {self.record_id!r}："
                "融合的第一步就是按 id 去重，没有 id 就没有'同一条'这件事。"
            )
        if not math.isfinite(float(self.score)):
            raise FusionError(
                f"FusedHit.score 必须是有限数，收到 {self.score!r}："
                "排序依赖它，nan 会让这条落到任意位置。"
            )
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise FusionError(
                f"FusedHit.rank 必须是非负整数（从 0 起），收到 {self.rank!r}"
            )
        for label, payload in (
            ("channel_ranks", self.channel_ranks),
            ("channel_scores", self.channel_scores),
            ("contributions", self.contributions),
        ):
            if not isinstance(payload, dict):
                raise FusionError(
                    f"FusedHit.{label} 必须是字典，收到 {type(payload).__name__}"
                )
        if set(self.channel_ranks) != set(self.contributions):
            raise FusionError(
                f"FusedHit 的通道证据不自洽：channel_ranks={sorted(self.channel_ranks)} "
                f"与 contributions={sorted(self.contributions)} 的通道集合不同。"
                "两处不一致时'这条为什么排这里'就无法回答——那正是这个形状存在的理由。"
            )

    @property
    def best_rank(self) -> int:
        """两路里最好的名次（``min(channel_ranks.values())``；没有通道时 ``0``）.

        它同时是排序的第二个键：分数并列时，"在某一路上排得更靠前的那条"
        更可能是真的相关（这条规则让融合的结果不依赖通道名的字典序）。
        """
        if not self.channel_ranks:
            return 0
        return min(self.channel_ranks.values())

    @property
    def channel_count(self) -> int:
        """被几路召回（``2`` = 两路都召回了它，融合里最强的证据）."""
        return len(self.channels)

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_evidence=False`` 只留"排序结果"（``record_id`` / ``score`` /
        ``rank`` / ``channels``）：逐条对比两次运行的融合次序时，
        三张证据表只会把 diff 淹掉。
        """
        payload: dict[str, Any] = {
            "rank": self.rank,
            "record_id": self.record_id,
            "score": round(self.score, 6),
            "channels": list(self.channels),
        }
        if include_evidence:
            payload["channel_ranks"] = dict(self.channel_ranks)
            payload["channel_scores"] = {
                name: round(value, 6) for name, value in self.channel_scores.items()
            }
            payload["contributions"] = {
                name: round(value, 6) for name, value in self.contributions.items()
            }
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        evidence = "、".join(
            f"{name}#{self.channel_ranks[name]}"
            f"(+{self.contributions.get(name, 0.0):.4f})"
            for name in self.channels
        )
        return (
            f"#{self.rank} {self.record_id} score={self.score:.6f} | "
            f"通道 {len(self.channels)} 路：{evidence or '（无）'}"
        )


# --------------------------------------------------------------------------- #
# 两种策略
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(
    channel_hits: Mapping[str, Sequence[Any]],
    k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """倒数名次融合：``score = Σ 1 / (k + rank + 1)``（rank 从 0 起）.

    **它只看名次、不看分数，这不是简化而是本模块最重要的一条设计决定**：
    余弦相似度 ∈ ``[-1, 1]``、BM25 ∈ ``[0, ∞)``——两个区间的数字直接相加，
    加出来的东西没有任何解释（"0.82 + 7.31 = 8.13"是什么的相关性？）。
    要让分数可比必须做归一化，而归一化需要一个**分布假设**
    （min-max 假设"这一路的最大值与最小值是有意义的边界"，而 BM25 的最小值
    恒为 0、向量路的 cos 最小值在一份小语料上可能只有 0.1）——
    参数一多，"结果不同"就不再指向唯一的原因。

    RRF 用"名次"这个**已经跨通道可比**的量绕开了整个问题：
    名次没有量纲，第 1 名在第 1 路与第 2 路是同一种东西。

    ``k`` 的作用是压平头部（见模块 docstring）：``k`` 越大，第 1 名与
    第 20 名的贡献差距越小，于是"被多路提到"比"在某一路排第一"更重要。

    ``channel_hits`` 的形状是 ``{通道名: 命中列表}``，命中只需要有
    ``record_id`` / ``rank`` / ``score`` 三个属性（``RetrievalHit`` 与
    ``LexicalHit`` 都满足）——**融合层刻意不认识这两路的类型**：
    认识它们就得为每一路写一份适配，而"融合是通用的一张表"这件事实
    会因此消失（第三路出现时要改的是融合层，而不是加一路）。

    同一路里出现重复的 ``record_id`` 直接报 ``FusionError``：那说明这一路
    自己没有去重，而两份不同的名次会让"它的名次"变成一个未定义的值。
    """
    _check_k(k)
    table = _extract_channels(channel_hits)

    contributions: dict[str, dict[str, float]] = {}
    for channel, hits in table.items():
        for record_id, (rank, _score) in hits.items():
            contributions.setdefault(record_id, {})[channel] = 1.0 / (k + rank + 1)

    scores = {
        record_id: math.fsum(per_channel.values())
        for record_id, per_channel in contributions.items()
    }
    ranks = {
        record_id: {channel: table[channel][record_id][0] for channel in per_channel}
        for record_id, per_channel in contributions.items()
    }
    raw_scores = {
        record_id: {channel: table[channel][record_id][1] for channel in per_channel}
        for record_id, per_channel in contributions.items()
    }
    return _assemble(scores, ranks, raw_scores, contributions)


def weighted_score_fusion(
    channel_hits: Mapping[str, Sequence[Any]],
    weights: Mapping[str, float],
) -> list[FusedHit]:
    """归一化加权融合：每路**先 min-max 到 [0, 1]**，再按权重求和.

    归一化的两个细节都不是可选项：

    ```text
    1. 归一化必须在**每一路**内部做（而不是对合并后的分数做）
       对合并后的分数做归一化时，"这一路的最大值"里混进了另一路的量纲，
       于是 alpha 调整的是"两路混在一起之后的重心"，仍然不是权重。
    2. min == max 时定义为 1.0（**不是除零**）
       这一路只有一条命中时必然 min == max；把它归一化成 0.0 会让
       "唯一的那条证据"在加权和里消失（0 × alpha = 0），
       而它明明是该路的第一名。定义成 1.0 的意思是"这一路内部，
       它就是最相关的那条"——归一化保序，把单点映射到 1 是保序的。
    ```

    代价也要写清楚：min-max 的上下界来自**这次的候选集合**，
    因此"同一路换个查询，同一份文档的归一化分不同"。
    这是归一化的固有性质（相对量），而不是实现缺陷——只要求
    **同一次融合内部**可比，这正是加权求和需要的全部。

    ``weights`` 必须覆盖 ``channel_hits`` 里的**每一个**通道：缺一路的权重
    直接报 ``FusionError``，不默认为 0。默认 0 意味着"这一路的证据静默消失"，
    而它的表现是"结果看起来正常，只是那一类查询突然变差了"。
    """
    table = _extract_channels(channel_hits)
    resolved = _resolve_weights(weights, table, label="weighted_score_fusion")

    contributions: dict[str, dict[str, float]] = {}
    for channel, hits in table.items():
        weight = resolved[channel]
        normalized = _min_max_normalize({rid: payload[1] for rid, payload in hits.items()})
        for record_id, value in normalized.items():
            contributions.setdefault(record_id, {})[channel] = weight * value

    scores = {
        record_id: math.fsum(per_channel.values())
        for record_id, per_channel in contributions.items()
    }
    ranks = {
        record_id: {channel: table[channel][record_id][0] for channel in per_channel}
        for record_id, per_channel in contributions.items()
    }
    raw_scores = {
        record_id: {channel: table[channel][record_id][1] for channel in per_channel}
        for record_id, per_channel in contributions.items()
    }
    return _assemble(scores, ranks, raw_scores, contributions)


def fuse(
    channel_hits: Mapping[str, Sequence[Any]],
    *,
    strategy: str = FUSION_RRF,
    alpha: float = DEFAULT_ALPHA,
    k_rrf: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """融合的统一入口（**混合检索只走这一个函数**）.

    ```text
    strategy="rrf"       用 k_rrf；alpha / weights 必须**不参与**（给了就报错）
    strategy="weighted"  weights 缺省时按 {"vector": alpha, "bm25": 1 - alpha} 展开
    ```

    ``strategy="rrf"`` 时给了 ``weights`` 会直接报 ``FusionError`` 而不是
    "忽略它"：RRF 的定义里没有权重（它的全部理由就是"不看分数、不加权"），
    而"我设了 weights 但结果没变"是一个会让人怀疑代码而不是怀疑参数的现象。
    ``alpha`` 则始终参与校验（即使在 rrf 下）：它越界说明**配置错了**，
    而错配置不该因为"这次用的是另一种策略"而被放过。

    通道名必须在 ``types.CHANNELS`` 这张封闭清单里：不在里面时，要么是
    调用方拼错了名字（``"bm25x"``），要么是给某一路漏了权重——
    **两种都必须当场响**，而不是凭空多出一路只有一半证据的通道。
    """
    if strategy not in FUSION_STRATEGIES:
        raise FusionError(
            f"未知融合策略 {strategy!r}：可用策略是 {'、'.join(FUSION_STRATEGIES)}。"
            "（rrf 只看名次、免标定；weighted 先归一化再加权，需要 alpha/weights）"
        )
    resolved_alpha = _check_alpha(alpha)
    _check_k(k_rrf)
    _check_channels(channel_hits)

    if strategy == FUSION_RRF:
        if weights is not None:
            raise FusionError(
                "strategy='rrf' 不接受 weights：RRF 的全部理由就是"
                "**只看名次、不做加权**（跨通道的分数量纲不可比，见模块 docstring）。"
                "要给某一路更高的话语权请改用 strategy='weighted'（配 alpha 或 weights）。"
            )
        return reciprocal_rank_fusion(channel_hits, k=k_rrf)

    if weights is not None:
        return weighted_score_fusion(channel_hits, weights)
    return weighted_score_fusion(
        channel_hits,
        {CHANNEL_VECTOR: resolved_alpha, CHANNEL_BM25: 1.0 - resolved_alpha},
    )


def contribution_counts(hits: Sequence[FusedHit]) -> dict[str, int]:
    """每一路在最终名单里"参与过多少条"（融合摘要里的一项）.

    它与"每路召回条数"是两个数字，差别正是融合的价值所在：

    ```text
    某路召回 3 条、贡献 0 条   它召回的东西全被别的路挤出了名单（这一路这次没用）
    某路召回 3 条、贡献 3 条   它的每一条都进了名单
    ```

    单看召回条数时这两种情况长得一样（都是 3）。
    """
    counts: dict[str, int] = {}
    for hit in hits:
        for channel in hit.channels:
            counts[channel] = counts.get(channel, 0) + 1
    return {channel: counts[channel] for channel in sorted(counts)}


# --------------------------------------------------------------------------- #
# 公共骨架（两种策略共用，因此"确定性"只写一份）
# --------------------------------------------------------------------------- #


def _assemble(
    scores: dict[str, float],
    ranks: dict[str, dict[str, int]],
    raw_scores: dict[str, dict[str, float]],
    contributions: dict[str, dict[str, float]],
) -> list[FusedHit]:
    """把四张表装配成排好序的 ``FusedHit`` 列表（排序规则在这里只写一份）.

    排序键是 ``(-score, best_rank, record_id)``：

    ```text
    -score      融合分数降序
    best_rank   分数并列时，在某一路上排得更靠前的那条优先
    record_id   还并列时按 id 升序（与 retriever / lexical 的第三键一致）
    ```

    三个键缺一不可：只有 ``-score`` 时会出现"同一份数据两次运行给出不同 top-1"
    （而原因不是算法，是没写下来的排序规则）；只有前两个时，两条路线都被
    同一名次命中的记录仍然无法定序。
    """
    ordered = sorted(
        scores,
        key=lambda record_id: (
            -scores[record_id],
            _best_rank(ranks[record_id]),
            record_id,
        ),
    )
    return [
        FusedHit(
            record_id=record_id,
            score=scores[record_id],
            rank=position,
            channels=tuple(_ordered_channels(ranks[record_id])),
            channel_ranks=dict(ranks[record_id]),
            channel_scores=dict(raw_scores[record_id]),
            contributions=dict(contributions[record_id]),
        )
        for position, record_id in enumerate(ordered)
    ]


def _best_rank(per_channel: dict[str, int]) -> int:
    """一条记录在两路里的最好名次（排序第二键；无通道时按 0 处理）."""
    return min(per_channel.values()) if per_channel else 0


def _ordered_channels(per_channel: dict[str, int]) -> list[str]:
    """通道名按"名次好坏"排序；名次相同时**向量优先**，再按名字兜底.

    向量优先不是偏袒：它与 ``RetrievalHit.channel`` 的归属规则
    （"贡献最大者，并列时按 ``CHANNEL_VECTOR`` 优先"）必须是同一条，
    否则 ``channels[0]`` 与 ``channel`` 会在并列时给出两个不同的答案——
    而它们是同一个问题的两个问法。
    """
    return sorted(per_channel, key=lambda name: (per_channel[name], name != CHANNEL_VECTOR, name))


def _extract_channels(
    channel_hits: Mapping[str, Sequence[Any]],
) -> dict[str, dict[str, tuple[int, float]]]:
    """把 ``{通道名: 命中列表}`` 收敛成 ``{通道名: {id: (rank, score)}}``（并校验形状）.

    这里做的四件校验各自对应一种真实错误：

    ```text
    不是 Mapping          调用方给了一个列表（"我想传两路，所以传了 [a, b]"）
    通道名不是非空字符串   通道名会进报告与权重表，空名字无法被引用
    命中缺三个属性之一     传了一个 dict（{"record_id": ...}）而不是命中对象
    同一路里 id 重复       那一路自己没去重，"它的名次"因此没有定义
    ```
    """
    if not isinstance(channel_hits, Mapping):
        raise FusionError(
            f"channel_hits 必须是 {{通道名: 命中列表}} 这样的映射，"
            f"收到 {type(channel_hits).__name__}。"
            "两路要写成 {'vector': [...], 'bm25': [...]}——"
            "列表形状里丢掉了'哪一路'这件事，而融合的第一步就是按通道分配名次。"
        )
    table: dict[str, dict[str, tuple[int, float]]] = {}
    for channel, hits in channel_hits.items():
        if not isinstance(channel, str) or not channel.strip():
            raise FusionError(f"通道名必须是非空字符串，收到 {channel!r}")
        if isinstance(hits, (str, bytes)) or not isinstance(hits, Sequence):
            raise FusionError(
                f"通道 {channel!r} 的命中必须是序列，收到 {type(hits).__name__}。"
                "空列表是合法的——它表示'这一路一条都没召回'（融合必须能处理这件事）。"
            )
        per_channel: dict[str, tuple[int, float]] = {}
        for position, hit in enumerate(hits):
            record_id, rank, score = _read_hit(hit, channel=channel, position=position)
            if record_id in per_channel:
                raise FusionError(
                    f"通道 {channel!r} 里出现了重复的 record_id {record_id!r}"
                    f"（位置 {position}）：同一路必须先自己去重。"
                    "两份不同的名次会让'它的名次'变成一个未定义的值，"
                    "而融合的贡献正是由名次算出来的。"
                )
            per_channel[record_id] = (rank, score)
        table[str(channel).strip()] = per_channel
    return table


def _read_hit(hit: Any, *, channel: str, position: int) -> tuple[str, int, float]:
    """从一个命中对象里读出 ``(record_id, rank, score)``（缺什么就说什么）."""
    where = f"通道 {channel!r} 的第 {position} 个命中"
    for attribute in ("record_id", "rank", "score"):
        if not hasattr(hit, attribute):
            raise FusionError(
                f"{where} 没有 {attribute} 属性（收到 {type(hit).__name__}）："
                "融合只要求命中提供 record_id / rank / score 三样东西，"
                "RetrievalHit 与 LexicalHit 都满足——请传这两者之一，"
                "不要传向量库的 SearchHit（它的 id 藏在 .record.record_id 里）。"
            )
    record_id = getattr(hit, "record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise FusionError(f"{where} 的 record_id 必须是非空字符串，收到 {record_id!r}")
    rank = getattr(hit, "rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise FusionError(
            f"{where} 的 rank 必须是非负整数（从 0 起），收到 {rank!r}："
            "RRF 的贡献 1/(k+rank+1) 直接由它算出，rank 从 1 起会让每条都多算一档。"
        )
    score = getattr(hit, "score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise FusionError(f"{where} 的 score 必须是数字，收到 {type(score).__name__}")
    if not math.isfinite(float(score)):
        raise FusionError(
            f"{where} 的 score 必须是有限数，收到 {score!r}："
            "nan 会让'这一路的最大值'变成 nan，归一化之后整路都成了 nan。"
        )
    return record_id, rank, float(score)


def _min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """把一路的分数 min-max 归一化到 [0, 1]（``min == max`` → 全部 1.0）.

    ``min == max`` 的两种情况都要按 1.0 处理：**只有一条命中**（这一路的第一名
    就是它），以及**多条同分**（这一路自己没区分开它们）。两种情况下
    "这一路内部最相关的"都是它们全体，归一化成 0.0 会让权重白给。

    判据是 ``high - low <= 0.0``——**精确的零跨度**，而不是"小于某个 epsilon"：
    一个凭空取出的 epsilon 会变成第三个需要标定的参数，而它的表现是
    "某些几乎相同的文档突然被归一化成 0"（比放大噪声更难解释）。

    代价写在明面上：跨度**非零但极小**（例如浮点误差级别的 1e-15）时，
    归一化会把它拉满到 ``[0, 1]``——这是 min-max 的固有性质，不是实现缺陷
    （归一化保序，也保"极差为 1"）。真要避免它，办法是在打分那一侧处理
    （例如同分文档本来就该给出同分），而不是在这里塞一个阈值。
    """
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high - low <= 0.0:
        return {record_id: 1.0 for record_id in scores}
    span = high - low
    return {record_id: (value - low) / span for record_id, value in scores.items()}


def _resolve_weights(
    weights: Mapping[str, float],
    table: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    """校验权重表：必须是映射、值必须有限且非负、**必须覆盖每一个通道**.

    "必须覆盖"是本函数最重要的一条：缺一路时报错而不是当 0，
    因为默认 0 会让那一路的证据**静默消失**（结果看起来正常，
    只是某一类查询突然变差了——一个查不到病因的现象）。
    """
    if not isinstance(weights, Mapping):
        raise FusionError(
            f"{label} 的 weights 必须是 {{通道名: 权重}} 这样的映射，"
            f"收到 {type(weights).__name__}（写法：{{'vector': 0.5, 'bm25': 0.5}}）"
        )
    resolved: dict[str, float] = {}
    for channel in table:
        if channel not in weights:
            raise FusionError(
                f"通道 {channel!r} 没有权重：weights 必须覆盖全部通道"
                f"（这次有 {'、'.join(sorted(table))}）。"
                "缺一路时**不能**默认为 0——那会让这一路的证据静默消失，"
                "而它的表现只是'某一类查询变差了'。"
            )
        value = weights[channel]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FusionError(
                f"weights[{channel!r}] 必须是数字，收到 {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise FusionError(f"weights[{channel!r}] 必须是有限数，收到 {value!r}")
        if float(value) < 0.0:
            raise FusionError(
                f"weights[{channel!r}]={value} 是负数：权重不允许为负。"
                "负权重意味着'这一路召回得越好、最终排名越差'——"
                "那不是一个可以用权重表达的需求，请改用显式的过滤条件。"
            )
        resolved[channel] = float(value)
    return resolved


def _check_alpha(alpha: float) -> float:
    """``alpha`` 必须是 ``[0, 1]`` 里的有限数（它是**向量通道的权重**）."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise FusionError(f"alpha 必须是数字，收到 {type(alpha).__name__}")
    value = float(alpha)
    if not math.isfinite(value):
        raise FusionError(f"alpha 必须是有限数，收到 {alpha!r}")
    if not 0.0 <= value <= 1.0:
        raise FusionError(
            f"alpha={value} 越界：alpha 是**向量通道的权重**，必须落在 [0, 1]"
            "（关键词通道的权重是 1 - alpha，因此给它 1.5 等于给关键词路 -0.5）。"
            "要表达'只看向量路'请给 alpha=1.0，'只看关键词路'给 alpha=0.0。"
        )
    return value


def _check_k(k: int) -> None:
    """``k_rrf`` 必须是 >= 1 的整数（``k=0`` 会让 rank=0 的贡献变成 1/1，过度偏向头部）."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise FusionError(
            f"k_rrf 必须是 >= 1 的整数，收到 {k!r}。"
            "它是 RRF 的平滑常数（缺省 60，出自 RRF 原论文）："
            "k 越小越只信'排名第一的那条'，k=0 会让贡献退化成 1/(rank+1) "
            "（第 1 名 1.0、第 2 名 0.5，头部权重过大）。"
        )


def _check_channels(channel_hits: Mapping[str, Sequence[Any]]) -> None:
    """通道名必须落在 ``types.CHANNELS`` 这张封闭清单里（理由见 ``fuse`` 的 docstring）."""
    if not isinstance(channel_hits, Mapping):
        return  # 形状问题交给 _extract_channels 报，那里的消息更具体
    unknown = sorted(
        str(name) for name in channel_hits if name not in CHANNELS
    )
    if unknown:
        raise FusionError(
            f"未知通道名 {unknown}：本层认识的通道是 {'、'.join(CHANNELS)}。"
            "拼错一个名字会让融合凭空多出一路只有一半证据的通道，"
            "而结果看起来完全正常——请核对通道名（或先把它加进 types.CHANNELS）。"
        )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RRF_K",
    "FUSION_OVERRIDE_KEYS",
    "FUSION_RRF",
    "FUSION_STRATEGIES",
    "FUSION_WEIGHTED",
    "FusedHit",
    "contribution_counts",
    "fuse",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
