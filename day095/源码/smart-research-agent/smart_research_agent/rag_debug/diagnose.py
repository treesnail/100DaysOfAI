"""归因：把"答案不好"折成**一个标签 + 一个动作**（day071）.

本模块只有一个公开函数 ``diagnose``，而它是**纯函数**：不读文件、不碰网络、
不看 ``settings``——输入一条 ``CaseOutcome``，输出一条 ``BadCase`` 或 ``None``。

## 判据的排列就是诊断的顺序

归因的逻辑短得可以写完，真正有信息量的是**排列**（``BAD_CASE_PRIORITY``）：
它逐条判、命中即返回，因此链路上**先出问题的层先被点名**：

```text
①  no_data                 库是空的             → 这是评测的前置条件，不是模型问题
②  retrieval_empty         检索为空（三种原因）   → 处置动作看 empty_reason 本身
③  retrieval_miss          金标准一条都没抓到     → 编码器 / 分块粒度 / 索引版本
④  rerank_cut              被第四道减法切掉       → 重排阈值那一个旋钮
⑤  packed_away             抓到了但没进提示词     → 预算（这一课独有的那一门失败）
⑥  rank_bad                进了提示词但排太后     → 排序侧（重排 / 融合）
⑦  truncated_chunk         片段是半截的           → 单块上限
⑧  generation_fallback     走了六种回退之一       → 按原因选动作（FALLBACK_ACTIONS）
⑨  hallucinated_citation   引用了不存在的编号     → 提示词 / 模型
⑩  ungrounded              一个有效引用都没有     → 引用要求（或 require_citation）
⑪  low_coverage            覆盖率低于下限         → 先看检索给得贴不贴题
⑫  weak_faithfulness       与参考答案比忠实度低   → 逐句对照
```

**"只给一个标签"这条纪律的理由**：一条坏例往往同时满足多个判据
（检索没抓到的时候，答案当然也没有引用），但它们的因果顺序是固定的——
⑩ 是 ③ 的**后果**。一次给出全部标签会把因果关系摊平成清单，
而读的人只会去修那个最显眼的（最后一个）。**先修因、再修果**，
所以归因只报最早的哪一层。

## 为什么判据是"读账"而不是"再算一次"

每一条 ``reason`` 里出现的数字全部来自 ``CaseOutcome`` 上已有的字段
（``retrieved.recall`` / ``packed.recall`` / ``dropped_by_rerank`` /
``hallucinated`` ...），归因不做任何新的计算。理由与 ``grounded``
"是算出来的而不是存下来的"类似，但方向相反：**归因必须是可复现的读数**——
如果它在内部重算一遍召回率，那么报告里的"召回 0.5"与"归因说召回为 0"
就有可能是两个数（一次算法改动、一次参数不同），而两个互相矛盾的读数
比一个错读数更坏。
"""

from __future__ import annotations

from collections.abc import Sequence

from smart_research_agent.rag_debug.errors import DiagnosisError
from smart_research_agent.rag_debug.types import (
    BAD_CASE_ACTIONS,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_FAITHFULNESS,
    DEFAULT_MIN_RECIPROCAL_RANK,
    FALLBACK_ACTIONS,
    TAG_GENERATION_FALLBACK,
    TAG_HALLUCINATED_CITATION,
    TAG_LOW_COVERAGE,
    TAG_NO_DATA,
    TAG_PACKED_AWAY,
    TAG_RANK_BAD,
    TAG_RERANK_CUT,
    TAG_RETRIEVAL_EMPTY,
    TAG_RETRIEVAL_MISS,
    TAG_TRUNCATED_CHUNK,
    TAG_UNGROUNDED,
    TAG_WEAK_FAITHFULNESS,
    BadCase,
    CaseOutcome,
)
from smart_research_agent.retrieval.types import (
    EMPTY_REASON_DESCRIPTIONS,
    EMPTY_REASON_NO_DATA,
)


def diagnose(
    outcome: CaseOutcome,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_reciprocal_rank: float = DEFAULT_MIN_RECIPROCAL_RANK,
    min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
    require_grounded: bool = True,
) -> BadCase | None:
    """把一条运行结果归因成**一个**坏例标签（没有问题就返回 ``None``）.

    四个阈值都有缺省值，且都可以逐次覆盖：**它们不是"质量的定义"，
    而是"这一批实验的判据"**——换一套更严的判据本身就该是一次显式决定，
    因此它们进的是函数参数（可被实验记录），而不是散落在代码里的字面量。

    ``require_grounded=False`` 的用法只有一种：**先量一次"不做接地要求时的
    可核对率"**（day069 把 ``retrieval_require_citation`` 的缺省设成关闭，
    理由正是"先让默认可核对率跑一段时间"）。因此这条开关存在的意义是
    "能复现那次测量"，而不是"放宽质量"。

    阈值的取值口径：

```text
min_coverage          覆盖率低于它算低覆盖（缺省 0.5：给出去的片段用不到一半）
min_reciprocal_rank   倒数排名低于它算"排太后"（缺省 0.5：金标准落在前 2 名之外）
min_faithfulness      忠实度低于它算弱（缺省 0.5，与 faithfulness 的三档口径一致）
```
    """
    _check_ratio("min_coverage", min_coverage)
    _check_ratio("min_faithfulness", min_faithfulness)
    if (
        isinstance(min_reciprocal_rank, bool)
        or not isinstance(min_reciprocal_rank, (int, float))
        or not 0.0 < float(min_reciprocal_rank) <= 1.0
    ):
        raise DiagnosisError(
            f"min_reciprocal_rank 必须落在 (0, 1]，收到 {min_reciprocal_rank!r}："
            "它是倒数排名的下限，1.0 表示'必须排第 1'、0.5 表示'前 2 名之外就算坏'；"
            "取 0 会让这条判据永远不成立（等于关掉它），而'关掉一条判据'应该是显式的。"
        )
    if not isinstance(require_grounded, bool):
        raise DiagnosisError(
            f"require_grounded 必须是布尔值，收到 {type(require_grounded).__name__}"
        )

    query = outcome.query
    retrieved = outcome.retrieved
    packed = outcome.packed

    # ① 库是空的：它不是模型问题，而是这次评估的前置条件没满足。
    if outcome.empty_reason == EMPTY_REASON_NO_DATA:
        return BadCase(
            query=query,
            tag=TAG_NO_DATA,
            reason=(
                "库是空的：这次检索一条候选都没有，"
                "因此这一条用例的四个指标全是 0——它衡量的是语料有没有灌进来，"
                "不是检索或生成的质量"
            ),
            action=BAD_CASE_ACTIONS[TAG_NO_DATA],
            evidence={
                "empty_reason": outcome.empty_reason,
                "retrieved_ids": ",".join(retrieved.ids) or "（无）",
            },
        )

    # ② 检索为空（其余三种原因）：处置动作看那一个原因本身。
    if outcome.empty_reason:
        described = EMPTY_REASON_DESCRIPTIONS.get(outcome.empty_reason, "未知原因")
        return BadCase(
            query=query,
            tag=TAG_RETRIEVAL_EMPTY,
            reason=(
                f"检索为空（empty_reason={outcome.empty_reason}）：{described}——"
                "空结果只给一个原因，因此这一条的处置动作只有那一个"
            ),
            action=BAD_CASE_ACTIONS[TAG_RETRIEVAL_EMPTY],
            evidence={
                "empty_reason": outcome.empty_reason,
                "dropped_by_top_k": outcome.dropped_by_top_k,
                "dropped_by_rerank": outcome.dropped_by_rerank,
            },
        )

    # ③ 一条金标准都没抓到：这是检索侧的问题（编码器 / 分块 / 索引版本）。
    if retrieved.recall == 0.0:
        return BadCase(
            query=query,
            tag=TAG_RETRIEVAL_MISS,
            reason=(
                f"检索没抓到：金标准 {len(outcome.relevant)} 条一条都没进名单"
                f"（召回 0.0，名单 {retrieved.count} 条）——"
                "这一条要先怀疑第 ③④ 层：编码器换过没重建索引、分块把答案切碎了、"
                "或者索引版本根本不含这批资料"
            ),
            action=BAD_CASE_ACTIONS[TAG_RETRIEVAL_MISS],
            evidence={
                "recall": retrieved.recall,
                "retrieved_ids": ",".join(retrieved.ids) or "（无）",
                "relevant": ",".join(outcome.relevant),
            },
        )

    # ④ 有金标准被重排阈值切掉（第四道减法）——它有自己的旋钮。
    if outcome.dropped_by_rerank > 0 and retrieved.recall < 1.0:
        return BadCase(
            query=query,
            tag=TAG_RERANK_CUT,
            reason=(
                f"重排阈值切掉了 {outcome.dropped_by_rerank} 条，"
                f"而金标准没被收全（召回 {retrieved.recall:.4f}）——"
                "第四道减法有它自己的旋钮，不该并进 top_k 那一笔账"
            ),
            action=BAD_CASE_ACTIONS[TAG_RERANK_CUT],
            evidence={
                "dropped_by_rerank": outcome.dropped_by_rerank,
                "recall": retrieved.recall,
            },
        )

    # ⑤ 抓到了但没进提示词：这一课独有的那一门失败（预算的错）。
    if packed.recall < retrieved.recall:
        return BadCase(
            query=query,
            tag=TAG_PACKED_AWAY,
            reason=(
                f"金标准被预算丢在了提示词之外：检索召回 {retrieved.recall:.4f}，"
                f"进了提示词之后只剩 {packed.recall:.4f}"
                f"（丢掉 {outcome.packed_away} 条依据，"
                f"dropped_hits={outcome.dropped_hits}）——"
                "检索是对的，模型却看不到那一条"
            ),
            action=BAD_CASE_ACTIONS[TAG_PACKED_AWAY],
            evidence={
                "retrieved_recall": retrieved.recall,
                "packed_recall": packed.recall,
                "dropped_hits": outcome.dropped_hits,
                "given": outcome.given,
            },
        )

    # ⑥ 进了提示词但排得太靠后：这是排序侧（重排 / 融合）的问题。
    if packed.recall == 1.0 and packed.reciprocal_rank < min_reciprocal_rank:
        return BadCase(
            query=query,
            tag=TAG_RANK_BAD,
            reason=(
                f"金标准进了提示词但排在第 {_rank_of(packed.reciprocal_rank)} 位"
                f"（倒数排名 {packed.reciprocal_rank:.4f} < {min_reciprocal_rank}）——"
                "召回没问题，次序不对"
            ),
            action=BAD_CASE_ACTIONS[TAG_RANK_BAD],
            evidence={
                "packed_reciprocal_rank": packed.reciprocal_rank,
                "packed_ndcg": packed.ndcg,
                "packed_ids": ",".join(packed.ids),
            },
        )

    # ⑦ 片段是半截的：截断（单块上限）与低覆盖同时出现时才点名，单独一个不算坏例。
    if outcome.truncated_hits > 0 and outcome.coverage < min_coverage:
        return BadCase(
            query=query,
            tag=TAG_TRUNCATED_CHUNK,
            reason=(
                f"有 {outcome.truncated_hits} 条片段被单块上限截断，"
                f"而覆盖率只有 {outcome.coverage:.4f}（下限 {min_coverage}）——"
                "依据是半截的，答案可能缺一句结论"
            ),
            action=BAD_CASE_ACTIONS[TAG_TRUNCATED_CHUNK],
            evidence={
                "truncated_hits": outcome.truncated_hits,
                "coverage": outcome.coverage,
            },
        )

    # ⑧ 走了回退：六种原因六种动作（第三种因素本身就是原因，不在这里细分）。
    if outcome.fallback_reason:
        action = FALLBACK_ACTIONS.get(outcome.fallback_reason, "")
        return BadCase(
            query=query,
            tag=TAG_GENERATION_FALLBACK,
            reason=(
                f"有片段，但这次答案走了回退（fallback_reason="
                f"{outcome.fallback_reason}）："
                f"{_fallback_phrase(outcome.fallback_reason)}"
            ),
            action=action or BAD_CASE_ACTIONS[TAG_GENERATION_FALLBACK],
            evidence={
                "fallback_reason": outcome.fallback_reason,
                "llm_called": outcome.llm_called,
                "given": outcome.given,
            },
        )

    # ⑨ 引用了不存在的编号：幻觉引用（它比"没引用"更坏，因此排在前面）。
    if outcome.hallucinated > 0:
        return BadCase(
            query=query,
            tag=TAG_HALLUCINATED_CITATION,
            reason=(
                f"答案里有 {outcome.hallucinated} 个幻觉引用："
                "那些编号不在那次提示词里——"
                "它不是'多写了几个数字'，而是**指到了一份并不存在的依据**"
            ),
            action=BAD_CASE_ACTIONS[TAG_HALLUCINATED_CITATION],
            evidence={
                "hallucinated": outcome.hallucinated,
                "cited": outcome.cited,
                "given": outcome.given,
            },
        )

    # ⑩ 一个有效引用都没有：无从核对（模型自由发挥了，或提示词没逼出引用）。
    if require_grounded and not outcome.grounded:
        return BadCase(
            query=query,
            tag=TAG_UNGROUNDED,
            reason=(
                f"答案没有任何有效引用（valid 为空、invalid 也为空）："
                f"给了 {outcome.given} 条片段、答案引用了 {outcome.cited} 条——"
                "这份答案无法被片段核对"
            ),
            action=BAD_CASE_ACTIONS[TAG_UNGROUNDED],
            evidence={
                "given": outcome.given,
                "cited": outcome.cited,
                "llm_called": outcome.llm_called,
            },
        )

    # ⑪ 覆盖率低：先看检索给得贴不贴题（unused 是覆盖率的分母）。
    if outcome.coverage < min_coverage:
        return BadCase(
            query=query,
            tag=TAG_LOW_COVERAGE,
            reason=(
                f"覆盖率 {outcome.coverage:.4f} 低于下限 {min_coverage}："
                f"给了 {outcome.given} 条片段，有效引用 {outcome.valid_count} 条、"
                f"未引用 {outcome.unused_count} 条——"
                "给了不少、用上的很少，先看检索给的片段贴不贴题"
            ),
            action=BAD_CASE_ACTIONS[TAG_LOW_COVERAGE],
            evidence={
                "coverage": outcome.coverage,
                "given": outcome.given,
                "cited": outcome.cited,
            },
        )

    # ⑫ 忠实度偏低：与参考答案逐句对照（生成侧的最后一关）。
    if outcome.faithfulness is not None and outcome.faithfulness < min_faithfulness:
        return BadCase(
            query=query,
            tag=TAG_WEAK_FAITHFULNESS,
            reason=(
                f"与参考答案对比，忠实度 {outcome.faithfulness:.4f} "
                f"低于下限 {min_faithfulness}——"
                "检索可能已经给对了片段，问题在生成侧"
            ),
            action=BAD_CASE_ACTIONS[TAG_WEAK_FAITHFULNESS],
            evidence={
                "faithfulness": outcome.faithfulness,
                "packed_recall": packed.recall,
            },
        )

    return None


def diagnose_all(
    outcomes: Sequence[CaseOutcome],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_reciprocal_rank: float = DEFAULT_MIN_RECIPROCAL_RANK,
    min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
    require_grounded: bool = True,
) -> tuple[BadCase, ...]:
    """逐条归因（**只收坏例**：通过的用例不产生记录）.

    为什么通过的用例不进这份列表：报告的读者要找的是"要动哪几个旋钮"，
    而"这一条没问题"对这个问题没有贡献。**通过率**由报告单独给出一个数字，
    两条信息因此各归其位（与 ``aggregate_results`` 把
    "空结果次数"与"平均命中数"分成两个键是同一条纪律）。
    """
    collected: list[BadCase] = []
    for outcome in outcomes:
        case = diagnose(
            outcome,
            min_coverage=min_coverage,
            min_reciprocal_rank=min_reciprocal_rank,
            min_faithfulness=min_faithfulness,
            require_grounded=require_grounded,
        )
        if case is not None:
            collected.append(case)
    return tuple(collected)


def tag_counts(bad_cases: Sequence[BadCase]) -> dict[str, int]:
    """按标签统计条数（只列**真的出现过**的标签，键按标签的固定顺序排）."""
    counts: dict[str, int] = {}
    for case in bad_cases:
        counts[case.tag] = counts.get(case.tag, 0) + 1
    return {tag: counts[tag] for tag in sorted(counts)}


def action_plan(bad_cases: Sequence[BadCase]) -> tuple[str, ...]:
    """把坏例汇总成一份**去重后的动作清单**（按第一次出现的顺序）.

    报告的最后一节应当是"下一步做什么"，而不是"有多少条坏"。同一类坏例出现
    12 次只需要一个动作——把 12 条记录原样列出来，等于让读的人自己做去重，
    而人在做去重时一定会漏（漏掉的那一类恰好是最少见、也最可能是新问题的那类）。
    """
    seen: list[str] = []
    for case in bad_cases:
        if case.action not in seen:
            seen.append(case.action)
    return tuple(seen)


def _check_ratio(label: str, value: float) -> None:
    """校验一个 [0, 1] 比例参数（两个阈值共用同一套话术）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosisError(
            f"{label} 必须是数字，收到 {type(value).__name__}（它是比例，取值 [0, 1]）"
        )
    number = float(value)
    if number != number or not 0.0 <= number <= 1.0:
        raise DiagnosisError(
            f"{label}={value!r} 必须落在 [0, 1]："
            "它是比例类的判据（覆盖率 / 忠实度），越界会让那条判据永远成立或不成立。"
        )


def _rank_of(reciprocal_rank: float) -> int:
    """由倒数排名反推名次（仅用于把话说清楚：1.0 → 第 1 位、0.5 → 第 2 位）."""
    if reciprocal_rank <= 0.0:
        return 0
    return int(round(1.0 / reciprocal_rank))


def _fallback_phrase(reason: str) -> str:
    """把回退原因翻成一句人话（读的是 generation 的那份说明，不重写一份）."""
    from smart_research_agent.retrieval.generation import FALLBACK_REASON_DESCRIPTIONS

    return FALLBACK_REASON_DESCRIPTIONS.get(reason, "（未知原因）")


__all__ = [
    "action_plan",
    "diagnose",
    "diagnose_all",
    "tag_counts",
]
