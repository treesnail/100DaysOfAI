"""day068 重排与两条流水线的集成测试（``rerank_hits`` + ``Retriever`` + ``HybridRetriever``）.

这一层要钉住的是**"重排被放在哪一步、它动了哪些字段、关掉时是否逐位不变"**——
也就是形状层证明不了的那一半。断言分九组：

```text
窗口纪律      top_n=3 / 候选 5 → 只有 3 条被打分，尾部一条都没碰；
              top_n > 候选 时全部打分；top_n=1 的极端
空输入        0 条候选 → **不调用重排器**（calls 保持 0）且 empty_reason 说清成因
次序与重编    replace 的名单来自手算排序键；rank 0 起连续；并列由 stage1_rank 兜底
blend        窗口内 min-max 的基准是**全部被打分的条**，且在 min_score 落刀之前
阈值         只切 scored=True 的那些；dropped_by_min_score 的实数；切空时的 empty_reason
weight       replace 下记录并进 notes；blend 下 weight=0/1 两个端点的次序
参数校验     四个参数的非法取值在**装配那一刻**就响（不必等到第一次检索）
注记纪律     窗口那条注记无条件出现（关掉重排时一条都不出现）
单路集成     回填 rerank_score / stage1_rank；摘要与 dropped_by_rerank；
             关掉时**逐位不变**（与 day067 同一份口径的回归用例）
混合集成     重排在多样性**之前**落刀；channels 不丢；extra 覆盖与封闭清单；
             RetrievalResult.explain() 多出的那一行
```

期望值全部来自 ``tests/rerank_samples.py`` 里的手算常量（十四个名单/分数）。

全部离线、确定性、零网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.retrieval.errors import FusionError, QueryError, RerankError
from smart_research_agent.retrieval.hybrid import HybridRetriever
from smart_research_agent.retrieval.rerank import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_N,
    RERANK_MODE_BLEND,
    RERANK_MODE_REPLACE,
    RERANK_OVERRIDE_KEYS,
    CrossEncoderReranker,
)
from smart_research_agent.retrieval.types import (
    EMPTY_REASON_BELOW_THRESHOLD,
    EMPTY_REASON_NO_DATA,
)
from tests.rerank_samples import (
    DOC_WINDOW,
    HAND_BLEND_ORDER,
    HAND_DIVERSITY_DROPPED,
    HAND_DIVERSITY_KEPT_WITH_RERANK,
    HAND_DIVERSITY_KEPT_WITHOUT_RERANK,
    HAND_FUSED_ORDER,
    HAND_HYBRID_RERANK_ORDER,
    HAND_RERANK_MOVED,
    HAND_RERANK_ORDER,
    HAND_RERANK_SCORES,
    HAND_RERANK_STAGE1_RANKS,
    HAND_SCORES,
    HAND_STAGE1_ORDER,
    HAND_WINDOW3_ORDER,
    RERANK_QUERY,
    STAGE1_SCORES,
    empty_store,
    query_of,
    rerank_hybrid_retriever,
    rerank_lexical,
    rerank_vector_retriever,
    stage1_hits,
)

# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

#: 只取前五条候选（"窗口纪律：候选 5 而 top_n=3"那条用例的输入）.
HAND_STAGE1_HEAD: tuple[str, ...] = HAND_STAGE1_ORDER[:5]

#: 窗口外那一条在 ``top_n=3`` 时的 7 条尾部（它必须全部 ``scored=False``）.
TAIL_IDS: tuple[str, ...] = HAND_STAGE1_ORDER[3:]


def rerank_summary_of(result: Any) -> dict[str, Any]:
    """从一份 ``RetrievalResult`` 里取出重排摘要（空字典 = 这次没开重排）."""
    return dict(result.rerank)


def notes_starting_with(result: Any, prefix: str) -> list[str]:
    """结果里以某个前缀开头的注记（"这条注记出现了几次"要用它来断言）."""
    return [note for note in result.notes if note.startswith(prefix)]


# --------------------------------------------------------------------------- #
# 窗口纪律
# --------------------------------------------------------------------------- #


class TestWindowDiscipline:
    """只对前 N 条打分：这一条纪律的账由三个计数器与 ``scored`` 标记回答."""

    def test_candidates_five_with_top_n_three(self) -> None:
        """候选 5、窗口 3：只有 3 条被打分，另外 2 条**一条都没碰**."""
        reranker = CrossEncoderReranker()
        result = rerank_hits_head(reranker, top_n=3)

        assert result.candidates == len(HAND_STAGE1_HEAD) == 5
        assert result.scored == 3
        assert result.window == 3
        assert reranker.calls == 1
        assert reranker.scored_pairs == 3
        assert reranker.batches == 1

    def test_the_two_candidates_outside_the_window_are_marked_unscored(self) -> None:
        """窗口外的两条：``scored=False`` / ``score=0.0`` / ``features={}`` / ``blended=None``."""
        result = rerank_hits_head(CrossEncoderReranker(), top_n=3)
        tail = [hit for hit in result.hits if not hit.scored]

        assert [hit.record_id for hit in tail] == ["r-t-04", "r-t-05"]
        for hit in tail:

            assert hit.score == 0.0
            assert hit.features == {}
            assert hit.blended is None
            assert hit.rank == hit.stage1_rank

    def test_top_n_larger_than_the_candidates_scores_everything(self) -> None:
        """窗口比候选大：五条全被打分（并且不会因为"取不满"而报错）."""
        reranker = CrossEncoderReranker()
        result = rerank_hits_head(reranker, top_n=50)

        assert result.scored == 5
        assert result.window == 5
        assert reranker.scored_pairs == 5
        assert all(hit.scored for hit in result.hits)

    def test_top_n_one_is_the_extreme(self) -> None:
        """``top_n=1``：只有第一条被打分，第一条的分数决定它自己的名次（0）."""
        reranker = CrossEncoderReranker()
        result = rerank_hits_head(reranker, top_n=1)

        assert result.scored == 1
        assert result.window == 1
        assert reranker.scored_pairs == 1
        assert result.hits[0].record_id == "r-t-01"
        assert result.hits[0].features == {
            "term_coverage": 0.0,
            "exact_phrase": 0.0,
            "proximity": 0.0,
            "length_penalty": 1.0,
        }

    def test_the_retriever_reports_the_same_window_account(self) -> None:
        """同一条纪律在单路流水线上也成立（``RetrievalResult.rerank`` 里的三个数）."""
        reranker = CrossEncoderReranker()
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=3, reranker=reranker)
        result = retriever.retrieve(query_of())

        assert reranker.scored_pairs == 3
        assert reranker.calls == 1
        assert rerank_summary_of(result)["candidates"] == 10
        assert rerank_summary_of(result)["scored"] == 3
        assert rerank_summary_of(result)["window"] == 3
        assert [hit.record_id for hit in result.hits] == list(HAND_WINDOW3_ORDER)

    def test_the_window_note_is_unconditional_on_both_pipelines(self) -> None:
        """窗口那条注记在单路与混合上都必须出现（它说的是"这次看不见什么"）."""
        single = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=4).retrieve(query_of())
        hybrid = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=4).retrieve(query_of())

        assert notes_starting_with(single, "重排：只对前 N 条打分")
        assert notes_starting_with(hybrid, "重排：只对前 N 条打分")

    def test_the_window_note_never_appears_when_rerank_is_off(self) -> None:
        """关掉重排时窗口注记一条都不该有（它只在重排真的跑过之后才有含义）."""
        single = rerank_vector_retriever().retrieve(query_of())
        hybrid = rerank_hybrid_retriever().retrieve(query_of())

        assert notes_starting_with(single, "重排：") == []
        assert notes_starting_with(hybrid, "重排：") == []


def rerank_hits_head(reranker: CrossEncoderReranker, *, top_n: int) -> Any:
    """对前五条候选做一次重排（窗口纪律那一组用例的公共起点）."""
    from smart_research_agent.retrieval.rerank import rerank_hits

    return rerank_hits(stage1_hits()[:5], RERANK_QUERY, reranker=reranker, top_n=top_n)


# --------------------------------------------------------------------------- #
# 空输入
# --------------------------------------------------------------------------- #


class TestEmptyInput:
    """0 条候选：不调用重排器，并把"是谁空着"写进 ``empty_reason``."""

    def test_no_candidates_never_calls_the_reranker(self) -> None:
        """``calls`` 与 ``scored_pairs`` 都保持 0——一次都不打分为空结果负责."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        reranker = CrossEncoderReranker()
        result = rerank_hits([], RERANK_QUERY, reranker=reranker)

        assert reranker.calls == 0
        assert reranker.scored_pairs == 0
        assert reranker.batches == 0
        assert result.candidates == 0
        assert result.window == 0

    def test_empty_reason_names_the_upstream_deductions(self) -> None:
        """文案要点出"上游那两道减法"，否则读的人只会以为重排没起作用."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits([], RERANK_QUERY)

        assert "没有候选可比" in result.empty_reason
        assert "dropped_below_threshold" in result.notes[0]
        assert "dropped_by_diversity" in result.notes[0]

    def test_an_empty_store_returns_no_data_without_reranking(self) -> None:
        """库是空的：早退发生在重排之前，因此摘要仍是空字典、第四个 dropped 数仍是 0."""
        reranker = CrossEncoderReranker()
        retriever = rerank_vector_retriever(
            empty_store(), rerank_enabled=True, reranker=reranker
        )
        result = retriever.retrieve(query_of())

        assert result.empty_reason == EMPTY_REASON_NO_DATA
        assert result.rerank == {}
        assert result.dropped_by_rerank == 0
        assert reranker.calls == 0

    def test_an_empty_store_on_the_hybrid_pipeline_too(self) -> None:
        """混合流水线的空库早退同样在重排之前（两层的口径必须一致）."""
        reranker = CrossEncoderReranker()
        store = empty_store()
        hybrid = HybridRetriever(
            rerank_vector_retriever(store, reranker=reranker),
            rerank_lexical(store),
            rerank_enabled=True,
            reranker=reranker,
        )
        result = hybrid.retrieve(query_of())

        assert result.empty_reason == EMPTY_REASON_NO_DATA
        assert result.rerank == {}
        assert reranker.calls == 0


# --------------------------------------------------------------------------- #
# replace 的次序与名次重编
# --------------------------------------------------------------------------- #


class TestReplaceOrdering:
    """排序键 ``(-score, stage1_rank, record_id)`` 三项各有可核对的用例."""

    def test_the_hand_computed_list(self) -> None:
        """十条的完整名单与逐条分数/原名次/移动量都对照手算常量."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert [hit.record_id for hit in result.hits] == list(HAND_RERANK_ORDER)
        assert [round(hit.score, 6) for hit in result.hits] == list(HAND_RERANK_SCORES)
        assert [hit.stage1_rank for hit in result.hits] == list(HAND_RERANK_STAGE1_RANKS)
        assert [hit.moved for hit in result.hits] == list(HAND_RERANK_MOVED)

    def test_ranks_are_renumbered_from_zero(self) -> None:
        """重排之后 ``rank`` 在**整份名单**上重编成 0 起连续（含窗口外的尾部）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=4)

        assert [hit.rank for hit in result.hits] == list(range(10))

    def test_ties_fall_back_to_stage1_rank(self) -> None:
        """三个 0.375 精确并列：次序 = 输入次序（第二键是 ``stage1_rank``）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)
        tied = [hit.record_id for hit in result.hits if round(hit.score, 6) == 0.375]

        assert tied == ["r-t-02", "r-t-05", "r-t-07"]
        assert [hit.stage1_rank for hit in result.hits if round(hit.score, 6) == 0.375] == [
            1,
            4,
            6,
        ]

    def test_ties_ignore_the_record_id_when_the_input_order_says_otherwise(self) -> None:
        """第二键不是 ``record_id`` 的判别用例：输入倒过来，并列次序也跟着倒过来.

        这一条拦的是"有人把第二键改成 ``record_id``"——那会让同分并列的名次
        重新由字典序决定，而这与"第一阶段怎么排的"无关。
        """
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(list(reversed(stage1_hits())), RERANK_QUERY, top_n=10)
        tied = [hit.record_id for hit in result.hits if round(hit.score, 6) == 0.375]

        assert tied == ["r-t-07", "r-t-05", "r-t-02"]
        assert tied != sorted(tied)

    def test_record_id_is_the_last_resort_key_and_is_unreachable_here(self) -> None:
        """第三键（``record_id``）在单次调用里**取不到**：``stage1_rank`` 是位置、必然唯一.

        这条用例把这个事实写下来：十条输入里任意两条的 ``(score, stage1_rank)``
        都不会同时相等，因此次序永远由前两键决定（``record_id`` 只是防"两次运行的
        名单不同"的兜底）。
        """
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)
        pairs = [(round(hit.score, 6), hit.stage1_rank) for hit in result.hits]

        assert len(set(pairs)) == len(pairs) == 10

    def test_the_exact_phrase_record_is_pulled_to_the_front(self) -> None:
        """主案例：第一阶段第 7 名（r-t-08）被顶到第 0 名（+7）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        top = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10).hits[0]

        assert top.record_id == "r-t-08"
        assert top.stage1_rank == 7
        assert top.moved == 7
        assert STAGE1_SCORES["r-t-08"] == 0.0
        assert top.stage1_score == 0.0

    def test_stage1_score_is_carried_through_unchanged(self) -> None:
        """``stage1_score`` 原样带来（用来回答"变了多少"），与 ``score`` 是两回事."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert [hit.stage1_score for hit in result.hits] == [
            STAGE1_SCORES[record_id] for record_id in HAND_RERANK_ORDER
        ]


# --------------------------------------------------------------------------- #
# blend：归一化基准
# --------------------------------------------------------------------------- #


class TestBlendNormalisation:
    """``blend`` 的两份分数各自在**窗口内** min-max 归一化，且发生在阈值之前."""

    def test_the_hand_computed_blend_list(self) -> None:
        """blend 的名单与加权分逐项对照手算常量（它的第一名与 replace 不同）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.5
        )

        assert [hit.record_id for hit in result.hits] == list(HAND_BLEND_ORDER)
        assert [round(hit.blended, 6) for hit in result.hits] == [
            0.568367,
            0.515306,
            0.5,
            0.346939,
            0.315306,
            0.168367,
            0.168367,
            0.015306,
            0.015306,
            0.0,
        ]

    def test_blend_overrides_the_replace_winner(self) -> None:
        """两个模式的第一名不同：replace 是 r-t-08，blend 是 r-t-02（第一阶段也占一半）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        replace = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)
        blend = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.5
        )

        assert replace.hits[0].record_id == "r-t-08"
        assert blend.hits[0].record_id == "r-t-02"
        assert replace.hits[0].blended is None
        assert blend.hits[0].blended == pytest.approx(0.568367, rel=1e-6)

    def test_normalisation_happens_before_the_threshold_cuts(self) -> None:
        """**先归一化、后落刀**：``r-t-06`` 的 0.346939 只在"用全部十条做基准"时成立.

        若实现先把 < 0.2 的条切掉再归一化，基准会变成 min=0.375 / max=11/12，
        同一个 r-t-06 会得到 0.269231——两个数字差得一眼看得出来。
        """
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(
            stage1_hits(),
            RERANK_QUERY,
            top_n=10,
            mode=RERANK_MODE_BLEND,
            weight=0.5,
            min_score=0.2,
        )
        by_id = {hit.record_id: hit for hit in result.hits}

        assert set(by_id) == {"r-t-02", "r-t-08", "r-t-06", "r-t-05", "r-t-07"}
        assert by_id["r-t-06"].blended == pytest.approx(0.3469387755, rel=1e-9)
        assert by_id["r-t-06"].blended != pytest.approx(0.269231, rel=1e-6)

    def test_normalisation_uses_the_window_only(self) -> None:
        """归一化的基准是**窗口内**的条：同一个 top_n 变小，基准随之改变（相对量的代价）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        wide = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.5
        )
        narrow = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=3, mode=RERANK_MODE_BLEND, weight=0.5
        )
        wide_top = next(hit for hit in wide.hits if hit.record_id == "r-t-02")
        narrow_top = next(hit for hit in narrow.hits if hit.record_id == "r-t-02")

        # 两次的窗口不同（0.1/11/12 对 0.125/0.375；第一阶段 0.0/1.0 对 0.6/1.0），
        # 因此同一个 r-t-02 得到的加权分不同 —— 这正是"归一化分随窗口变化"的证据：
        #   宽窗口 0.5·((3/8−1/10)÷49/60) + 0.5·0.8 = 0.568367
        #   窄窗口 0.5·1.0                + 0.5·0.5 = 0.75
        assert wide_top.blended == pytest.approx(0.568367, rel=1e-6)
        assert narrow_top.blended == pytest.approx(0.75, rel=1e-12)
        assert wide_top.blended != narrow_top.blended

    @pytest.mark.parametrize(
        ("weight", "expected_first"),
        [(0.0, "r-t-01"), (0.5, "r-t-02"), (1.0, "r-t-08")],
    )
    def test_weight_sweeps_the_winner(self, weight: float, expected_first: str) -> None:
        """weight 的两个端点与中点：0 = 只信第一阶段，1 = 只信重排分（次序同 replace）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=weight
        )

        assert result.hits[0].record_id == expected_first

    def test_weight_one_is_the_same_order_as_replace(self) -> None:
        """``weight=1.0`` 时 blend 的次序逐项等于 replace（归一化保序）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        blend = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=1.0
        )
        replace = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert [hit.record_id for hit in blend.hits] == [hit.record_id for hit in replace.hits]

    def test_weight_zero_is_the_same_order_as_stage_one(self) -> None:
        """``weight=0.0`` 时次序回到第一阶段（重排分完全不参与）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.0
        )

        assert [hit.record_id for hit in result.hits] == list(HAND_STAGE1_ORDER)

    def test_blend_through_the_retriever_matches_the_hand_computed_list(self) -> None:
        """混合模式在单路流水线上也要能端到端跑出手算名单."""
        retriever = rerank_vector_retriever(
            rerank_enabled=True,
            rerank_top_n=10,
            rerank_mode=RERANK_MODE_BLEND,
            rerank_weight=0.5,
        )
        result = retriever.retrieve(query_of())

        assert [hit.record_id for hit in result.hits] == list(HAND_BLEND_ORDER)
        assert rerank_summary_of(result)["mode"] == RERANK_MODE_BLEND
        assert rerank_summary_of(result)["params"] == {"top_n": 10, "weight": 0.5}


# --------------------------------------------------------------------------- #
# 阈值与权重
# --------------------------------------------------------------------------- #


class TestMinScoreAndWeight:
    """``min_score`` 只切被打分的条；``weight`` 在 replace 下记录、在 blend 下生效."""

    def test_min_score_drops_only_the_scored_ones(self) -> None:
        """切掉 8 条（两个高分留下），而账本记下了确切数字."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, min_score=0.5)

        assert [hit.record_id for hit in result.hits] == ["r-t-08", "r-t-06"]
        assert result.dropped_by_min_score == 8
        assert result.scored == 10

    def test_min_score_is_inclusive(self) -> None:
        """阈值用 ``<`` 而不是 ``<=``：恰好等于阈值的三条要留下（那条并列的 0.375）."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, min_score=0.375)

        assert [hit.record_id for hit in result.hits] == [
            "r-t-08",
            "r-t-06",
            "r-t-02",
            "r-t-05",
            "r-t-07",
        ]
        assert result.dropped_by_min_score == 5

    def test_the_tail_is_immune_to_the_threshold(self) -> None:
        """窗口外的条不受阈值影响：``top_n=3`` + ``min_score=0.5`` 时窗口被切空，尾部还在."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3, min_score=0.5)

        assert [hit.record_id for hit in result.hits] == list(TAIL_IDS)
        assert result.dropped_by_min_score == 3
        assert result.empty_reason == ""
        assert all(hit.scored is False for hit in result.hits)

    def test_threshold_emptying_the_window_reports_below_threshold(self) -> None:
        """窗口被切空且没有尾部：``empty_reason`` 必须是那五种之一里的 ``below_threshold``."""
        retriever = rerank_vector_retriever(
            rerank_enabled=True, rerank_top_n=10, rerank_min_score=1.0
        )
        result = retriever.retrieve(query_of())

        assert result.hits == ()
        assert result.empty_reason == EMPTY_REASON_BELOW_THRESHOLD
        assert result.dropped_by_rerank == 10
        assert rerank_summary_of(result)["empty_reason"].startswith("重排阈值 1.0")
        assert any("而是阈值定得比全部分数都高" in note for note in result.notes)

    def test_min_score_through_the_retriever_reports_the_fourth_dropped_number(self) -> None:
        """第四个 dropped 数（``dropped_by_rerank``）不许并进 ``dropped_by_top_k``."""
        retriever = rerank_vector_retriever(
            rerank_enabled=True, rerank_top_n=10, rerank_min_score=0.5
        )
        result = retriever.retrieve(query_of())

        assert [hit.record_id for hit in result.hits] == ["r-t-08", "r-t-06"]
        assert result.dropped_by_rerank == 8
        assert result.dropped_by_top_k == 0
        assert result.dropped_below_threshold == 0
        assert result.dropped_by_diversity == 0
        assert result.to_dict()["dropped_by_rerank"] == 8

    def test_replace_records_the_weight_and_says_it_did_not_participate(self) -> None:
        """replace 下权重照样被记录，并由 notes 说明"它没有参与排序"."""
        retriever = rerank_vector_retriever(
            rerank_enabled=True, rerank_top_n=10, rerank_weight=0.25
        )
        result = retriever.retrieve(query_of())

        assert rerank_summary_of(result)["params"] == {"top_n": 10, "weight": 0.25}
        assert any("mode='replace' 不看权重" in note for note in result.notes)
        assert any("weight=0.25 已记录在结果里但没有参与排序" in note for note in result.notes)
        assert [hit.record_id for hit in result.hits] == list(HAND_RERANK_ORDER)

    def test_blend_weight_is_used_not_merely_recorded(self) -> None:
        """blend 下权重真的参与排序（0.0 与 1.0 给出两个不同的第一名）."""
        stage_one_first = rerank_vector_retriever(
            rerank_enabled=True,
            rerank_top_n=10,
            rerank_mode=RERANK_MODE_BLEND,
            rerank_weight=0.0,
        ).retrieve(query_of())
        rerank_first = rerank_vector_retriever(
            rerank_enabled=True,
            rerank_top_n=10,
            rerank_mode=RERANK_MODE_BLEND,
            rerank_weight=1.0,
        ).retrieve(query_of())

        assert stage_one_first.ids() == list(HAND_STAGE1_ORDER)
        assert stage_one_first.ids()[0] == "r-t-01"
        assert rerank_first.ids() == list(HAND_RERANK_ORDER)
        assert rerank_first.ids()[0] == "r-t-08"

    def test_a_weighted_replace_does_not_add_a_blend_value(self) -> None:
        """记录权重 ≠ 算了一个加权分：replace 下每条的 ``blended`` 仍是 ``None``."""
        from smart_research_agent.retrieval.rerank import rerank_hits

        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, weight=0.25)

        assert result.weight == 0.25
        assert [hit.blended for hit in result.hits] == [None] * 10


# --------------------------------------------------------------------------- #
# 参数校验（装配那一刻就响）
# --------------------------------------------------------------------------- #


class TestConstructionTimeValidation:
    """四个重排参数写错时，``Retriever`` / ``HybridRetriever`` 的构造期就要报."""

    @pytest.mark.parametrize("top_n", [0, -1, 1.5, "3", True])
    def test_single_route_rejects_bad_top_n_at_construction(self, top_n: Any) -> None:
        """构造期校验用一份空输入跑一遍 ``rerank_hits``（零开销，但错误当场响）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_vector_retriever(rerank_enabled=True, rerank_top_n=top_n)

        assert "top_n 必须是 >= 1 的整数" in str(excinfo.value)

    @pytest.mark.parametrize("mode", ["mod", "", "Replace", 3])
    def test_single_route_rejects_unknown_mode(self, mode: Any) -> None:
        """模式名拼错不能让"我换了一种策略"变成一个查不出病因的现象."""
        with pytest.raises(RerankError) as excinfo:
            rerank_vector_retriever(rerank_enabled=True, rerank_mode=mode)

        assert "可用模式是 replace、blend" in str(excinfo.value)

    @pytest.mark.parametrize("weight", [-0.1, 1.5, float("nan"), "0.5"])
    def test_single_route_rejects_bad_weight(self, weight: Any) -> None:
        """权重越界要给出一条可执行的出路."""
        with pytest.raises(RerankError) as excinfo:
            rerank_vector_retriever(rerank_enabled=True, rerank_weight=weight)

        assert "weight" in str(excinfo.value)

    @pytest.mark.parametrize("reranker", ["cross-encoder", 3, "bge-reranker"])
    def test_single_route_rejects_a_non_reranker(self, reranker: Any) -> None:
        """不按名字加载模型：传字符串模型名要当场报."""
        with pytest.raises(RerankError) as excinfo:
            rerank_vector_retriever(reranker=reranker)

        assert "reranker 必须是 BaseReranker" in str(excinfo.value)

    def test_min_score_nan_is_rejected(self) -> None:
        """nan 阈值会静默地一条都不留，因此必须当场拒."""
        with pytest.raises(RerankError) as excinfo:
            rerank_vector_retriever(rerank_enabled=True, rerank_min_score=float("nan"))

        assert "nan 与任何分数比较都返回 False" in str(excinfo.value)

    def test_validation_happens_even_when_rerank_is_off(self) -> None:
        """一个写错的参数不该因为"这次没开"而被放过（它在打开的那一刻就会生效）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_vector_retriever(rerank_top_n=0)

        assert "top_n 必须是 >= 1 的整数" in str(excinfo.value)

    @pytest.mark.parametrize("top_n", [0, -1, 1.5])
    def test_hybrid_rejects_bad_top_n_at_construction(self, top_n: Any) -> None:
        """混合那一层的装配同样当场校验（理由与单路逐字相同）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=top_n)

        assert "top_n 必须是 >= 1 的整数" in str(excinfo.value)

    def test_hybrid_rejects_an_unknown_mode(self) -> None:
        """混合层的 mode 校验与单路共用同一份实现."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hybrid_retriever(rerank_enabled=True, rerank_mode="hybrid")

        assert "可用模式是 replace、blend" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 单路 Retriever 集成
# --------------------------------------------------------------------------- #


class TestVectorRetrieverIntegration:
    """``Retriever`` 第 6.5 步：回填两个字段 + 一份摘要 + 第四个 dropped 数."""

    def test_rerank_backfills_the_two_fields(self) -> None:
        """``rerank_score`` / ``stage1_rank`` 回填，而 ``score`` 仍是第一阶段的分数."""
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())

        assert [hit.record_id for hit in result.hits] == list(HAND_RERANK_ORDER)
        assert [round(hit.rerank_score, 6) for hit in result.hits] == list(HAND_RERANK_SCORES)
        assert [hit.stage1_rank for hit in result.hits] == list(HAND_RERANK_STAGE1_RANKS)

    def test_the_stage_one_score_is_not_overwritten(self) -> None:
        """``score`` 的口径不变（重排分另占一个字段）：重排后的名单里它不再单调."""
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())

        assert [hit.score for hit in result.hits] == [
            STAGE1_SCORES[record_id] for record_id in HAND_RERANK_ORDER
        ]
        scores = [hit.score for hit in result.hits]
        assert scores != sorted(scores, reverse=True)

    def test_ranks_are_contiguous_after_rerank(self) -> None:
        """第 8 步只重编号、不重排：重排开的名单里 ``rank`` 从 0 连续."""
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())

        assert [hit.rank for hit in result.hits] == list(range(10))

    def test_the_summary_is_the_shape_projection(self) -> None:
        """``RetrievalResult.rerank`` 逐键等于 ``RerankResult.to_summary()``."""
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())

        assert rerank_summary_of(result) == {
            "model": DEFAULT_RERANK_MODEL,
            "mode": RERANK_MODE_REPLACE,
            "params": {"top_n": 10, "weight": 0.5},
            "candidates": 10,
            "scored": 10,
            "window": 10,
            "dropped_by_min_score": 0,
            "moved": 10,
            "empty_reason": "",
        }

    def test_the_two_candidates_fields_are_different_ledgers(self) -> None:
        """同名的两个 ``candidates`` **不是同一个口径**：库侧候选 / 重排的输入条数.

        用第一阶段的阈值把它们分开：``min_score=0.5`` 先把 10 条切到 3 条，
        于是 ``RetrievalResult.candidates`` 仍是 10，而重排看到的只有 3 条。
        把这两个数当成一个读，会得到"重排吃掉了 7 条"这种错误结论。
        """
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of(min_score=0.5))

        assert result.candidates == 10
        assert result.dropped_below_threshold == 7
        assert result.rerank["candidates"] == 3
        assert result.rerank["scored"] == 3
        assert [hit.record_id for hit in result.hits] == ["r-t-02", "r-t-01", "r-t-03"]

    def test_the_window_summary_reports_the_windowed_numbers(self) -> None:
        """``top_n=3`` 时的摘要：``scored`` / ``window`` 都是 3，``moved`` 是 2."""
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=3)
        result = retriever.retrieve(query_of())
        summary = rerank_summary_of(result)

        assert (summary["candidates"], summary["scored"], summary["window"]) == (10, 3, 3)
        assert summary["moved"] == 2
        assert summary["params"] == {"top_n": 3, "weight": 0.5}

    def test_rerank_notes_are_prefixed_and_attached_to_the_result(self) -> None:
        """重排自己的注记要带前缀进 ``notes``（否则分不清是谁说的）."""
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=3)
        result = retriever.retrieve(query_of())
        rerank_notes = notes_starting_with(result, "重排：")

        assert len(rerank_notes) >= 2
        assert any("一条都没打分" in note for note in rerank_notes)
        assert any("按**原顺序**接在重排结果之后" in note for note in rerank_notes)
        assert any("mode='replace' 不看权重" in note for note in rerank_notes)

    def test_the_reranker_instance_is_reused_and_counted(self) -> None:
        """检索器复用同一个重排器实例：跑两次之后它的计数器是两次的账."""
        reranker = CrossEncoderReranker()
        retriever = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=10, reranker=reranker)
        retriever.retrieve(query_of())
        retriever.retrieve(query_of())

        assert reranker.calls == 2
        assert reranker.scored_pairs == 20

    def test_describe_reports_the_rerank_group(self) -> None:
        """``describe()`` 的重排那一组里要有 enabled 与五个参数（关着时也要能读出来）."""
        described = rerank_vector_retriever(rerank_enabled=True, rerank_top_n=7).describe()
        rerank = described["rerank"]

        assert rerank["enabled"] is True
        assert rerank["top_n"] == 7
        assert rerank["mode"] == RERANK_MODE_REPLACE
        assert rerank["weight"] == 0.5
        assert rerank["min_score"] is None
        assert rerank["model"]["name"] == DEFAULT_RERANK_MODEL
        assert rerank["model"]["kind"] == "cross-encoder(teaching)"

    def test_describe_keeps_reporting_the_parameters_when_off(self) -> None:
        """关着重排时那五个数字仍然有意义（它们会在打开的那一刻生效）."""
        rerank = rerank_vector_retriever(rerank_top_n=7).describe()["rerank"]

        assert rerank["enabled"] is False
        assert rerank["top_n"] == 7


class TestVectorRetrieverRegression:
    """**最重要的一条**：关掉重排时结果与 day067 逐位一致（一个字段都不改写）."""

    def test_the_off_shape_is_the_day067_shape(self) -> None:
        """关闭时：名单 = 分数降序 + id 升序，``rank`` 连续，两个新字段都是 ``None``."""
        result = rerank_vector_retriever().retrieve(query_of())

        assert result.ids() == list(HAND_STAGE1_ORDER)
        assert [hit.score for hit in result.hits] == [
            STAGE1_SCORES[record_id] for record_id in HAND_STAGE1_ORDER
        ]
        assert [hit.rank for hit in result.hits] == list(range(10))
        assert [hit.rerank_score for hit in result.hits] == [None] * 10
        assert [hit.stage1_rank for hit in result.hits] == [None] * 10

    def test_the_off_ledgers_are_empty(self) -> None:
        """关闭时三笔账都是"这次没开重排"：空字典与 0（与 ``channels=()`` 同一套写法）."""
        result = rerank_vector_retriever().retrieve(query_of())

        assert result.rerank == {}
        assert result.dropped_by_rerank == 0
        assert result.to_dict()["rerank"] == {}
        assert result.to_dict()["dropped_by_rerank"] == 0

    def test_the_off_json_keeps_the_two_keys_as_null(self) -> None:
        """``to_dict()`` 的两个键**恒存在**（值可能是 null），下游不必先判断有没有这个键."""
        payload = rerank_vector_retriever().retrieve(query_of()).to_dict(include_text=True)

        assert "rerank_score" in payload["hits"][0]
        assert "stage1_rank" in payload["hits"][0]
        assert payload["hits"][0]["rerank_score"] is None
        assert payload["hits"][0]["stage1_rank"] is None

    def test_the_off_result_adds_no_rerank_lines_to_explain(self) -> None:
        """``result.explain()`` 一条重排行都不加，而第四个 dropped 数在共享的那一行里是 0."""
        lines = rerank_vector_retriever().retrieve(query_of()).explain()

        assert not any(line.startswith("重排：") for line in lines)
        assert any("重排阈值切掉 0 条" in line for line in lines)
        assert any("条数：10 / top_k=10" in line for line in lines)

    def test_the_off_run_is_bit_for_bit_the_same_twice(self) -> None:
        """同一个输入两次给出逐位相同的名单与分数（day067 的可复现性口径）."""
        first = rerank_vector_retriever().retrieve(query_of())
        second = rerank_vector_retriever().retrieve(query_of())

        assert first.ids() == second.ids()
        assert [hit.score for hit in first.hits] == [hit.score for hit in second.hits]

    def test_the_rerank_line_is_still_readable_when_off(self) -> None:
        """``Retriever.explain()`` 永远有"重排口径"那一行（一个关着的旋钮也要能读出来）."""
        retriever = rerank_vector_retriever()
        result = retriever.retrieve(query_of())
        lines = retriever.explain(result)

        assert any("重排口径：**关闭**（缺省不生效）" in line for line in lines)
        assert any("打开后会用 'cross-encoder-teaching-v1' 只对前 20 条重排" in line
                   for line in lines)

    def test_the_rerank_line_reports_the_live_parameters_when_on(self) -> None:
        """开着时的口径行要写清模型、模式、窗口、权重与阈值，并点明它排在哪一步."""
        retriever = rerank_vector_retriever(
            rerank_enabled=True, rerank_top_n=6, rerank_weight=0.25
        )
        result = retriever.retrieve(query_of())
        line = next(line for line in retriever.explain(result) if line.startswith("重排口径"))

        assert "重排口径：开启" in line
        assert "4 维可手算特征，批大小 16" in line
        assert "模式 'replace'、只对前 6 条打分、权重 0.25、重排阈值 不设阈值" in line
        assert "它发生在阈值落刀之后、多样性裁剪之前" in line


# --------------------------------------------------------------------------- #
# 混合 HybridRetriever 集成
# --------------------------------------------------------------------------- #


class TestHybridRetrieverIntegration:
    """``HybridRetriever`` 第 10.5 步：融合之后、多样性之前."""

    def test_rerank_off_matches_the_fused_order(self) -> None:
        """关掉时名单与融合次序逐位相同（多路证据也在）."""
        result = rerank_hybrid_retriever().retrieve(query_of())

        assert result.ids() == list(HAND_FUSED_ORDER)
        assert result.rerank == {}
        assert result.dropped_by_rerank == 0
        assert [hit.channels for hit in result.hits[:5]] == [
            ("vector", "bm25"),
            ("bm25", "vector"),
            ("bm25", "vector"),
            ("bm25", "vector"),
            ("vector", "bm25"),
        ]

    def test_rerank_on_matches_the_hand_computed_hybrid_order(self) -> None:
        """打开后名单是"融合次序再重排"的手算常量，且 ``stage1_rank`` 是融合名次."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())

        assert result.ids() == list(HAND_HYBRID_RERANK_ORDER)
        assert [hit.stage1_rank for hit in result.hits] == [2, 1, 0, 3, 4, 5, 6, 7, 9, 8]
        assert [round(hit.rerank_score, 6) for hit in result.hits] == [
            round(HAND_SCORES[record_id], 6) for record_id in HAND_HYBRID_RERANK_ORDER
        ]

    def test_the_hybrid_summary_reports_the_fused_window(self) -> None:
        """摘要里的 ``candidates`` 是**重排的输入条数**（融合之后的并集），不是库侧候选."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())
        summary = rerank_summary_of(result)

        assert summary["candidates"] == 10
        assert summary["scored"] == 10
        assert summary["window"] == 10
        assert summary["moved"] == 4
        assert summary["mode"] == RERANK_MODE_REPLACE

    def test_channels_are_not_lost_by_the_rerank(self) -> None:
        """重排之后每条的多路证据必须与关掉时逐条相同（融合证据不能在最后一步丢）."""
        off = rerank_hybrid_retriever().retrieve(query_of())
        on = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10).retrieve(query_of())
        expected = {hit.record_id: hit.channels for hit in off.hits}

        assert [hit.channels for hit in on.hits] == [
            expected[hit.record_id] for hit in on.hits
        ]
        assert any(hit.channels == ("bm25", "vector") for hit in on.hits)
        assert all(hit.channels for hit in on.hits)

    def test_the_primary_channel_values_are_unchanged(self) -> None:
        """``channel``（贡献最大的那一路）也不该被重排改写."""
        off = rerank_hybrid_retriever().retrieve(query_of())
        on = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10).retrieve(query_of())
        expected = {hit.record_id: hit.channel for hit in off.hits}

        assert [hit.channel for hit in on.hits] == [
            expected[hit.record_id] for hit in on.hits
        ]

    def test_rerank_lands_before_the_diversity_trim(self) -> None:
        """**重排在多样性之前落刀**：同一个 ``max_per_doc=1``，开关一翻，留下的那条就换了人.

        融合次序里 ``DOC_WINDOW`` 的三条（r-t-05 / r-t-06 / r-t-08）中最好的是 r-t-06；
        重排把 r-t-08 顶到整份名单第 0 位之后，多样性保的就变成了 r-t-08。
        若实现"先裁后重排"，两种开关下留下的都会是 r-t-06——那时这条用例会红。
        """
        off = rerank_hybrid_retriever().retrieve(query_of(max_per_doc=1))
        on = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10).retrieve(
            query_of(max_per_doc=1)
        )

        assert HAND_DIVERSITY_KEPT_WITHOUT_RERANK in off.ids()
        assert HAND_DIVERSITY_KEPT_WITH_RERANK not in off.ids()
        assert HAND_DIVERSITY_KEPT_WITH_RERANK in on.ids()
        assert HAND_DIVERSITY_KEPT_WITHOUT_RERANK not in on.ids()
        assert off.dropped_by_diversity == on.dropped_by_diversity == HAND_DIVERSITY_DROPPED
        assert off.ids() != on.ids()

    def test_the_diversity_trim_still_runs_after_the_rerank(self) -> None:
        """两个开关下都只有一条 ``DOC_WINDOW`` 成员活下来（裁剪本身没有被跳过）."""
        on = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10).retrieve(
            query_of(max_per_doc=1)
        )
        window_members = [record_id for record_id in on.ids() if record_id in
                          {"r-t-05", "r-t-06", "r-t-08"}]

        assert window_members == [HAND_DIVERSITY_KEPT_WITH_RERANK]
        assert DOC_WINDOW == "doc-window"

    def test_ranks_are_renumbered_after_the_diversity_trim(self) -> None:
        """多样性丢掉几条之后名次必然有空洞，第 12 步必须补上（0 起连续）."""
        on = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10).retrieve(
            query_of(max_per_doc=1)
        )

        assert [hit.rank for hit in on.hits] == list(range(len(on.hits)))
        assert len(on.hits) == 8

    def test_the_hybrid_explain_line_reports_the_rerank_stage(self) -> None:
        """``RetrievalResult.explain()`` 多出的那一行要把四笔账都写出来."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of())
        line = next(line for line in result.explain() if line.startswith("重排："))

        assert line.startswith("重排：cross-encoder-teaching-v1（top_n=10、weight=0.5）")
        assert "候选 10 条" in line
        assert "真正打分 10 条" in line
        assert "名次动了 4 条" in line
        assert "重排阈值切掉 0 条" in line

    def test_the_explain_line_renders_the_window_size(self) -> None:
        """那一行里的"只对前 N 条"必须是**真的窗口**（``params["top_n"]``）.

        ``RetrievalResult.explain()`` 读的是摘要里的 ``params["top_n"]``（回落 ``window``），
        不是顶层键——摘要顶层没有 ``top_n``，多一个同义副本迟早会与 ``params`` 分家。
        这条用例同时断言"两个数字逐位对上"，因此"打印 0"或"打印别的数"都会红。
        """
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=6)
        result = retriever.retrieve(query_of())
        line = next(line for line in result.explain() if line.startswith("重排："))

        assert "只对前 6 条打分" in line
        assert result.rerank["params"]["top_n"] == 6
        assert result.rerank["window"] == 6

    def test_the_explain_line_is_absent_when_rerank_is_off(self) -> None:
        """关掉时诊断里没有重排行（空字典 = 这次没开重排：非空才渲染）."""
        result = rerank_hybrid_retriever().retrieve(query_of())

        assert not any(line.startswith("重排：") for line in result.explain())

    def test_the_hybrid_explain_already_lists_the_rerank_scope(self) -> None:
        """混合层的 ``explain()`` 本来就有一行"重排口径：…（第 10.5 步）"."""
        retriever = rerank_hybrid_retriever()
        result = retriever.retrieve(query_of())
        line = next(line for line in retriever.explain(result) if line.startswith("重排口径"))

        assert "重排在**融合之后、多样性之前**（第 10.5 步）" in line
        assert "关着的时候一个字段都不会被改写（结果与 day067 逐位相同）" in line


class TestHybridQueryOverrides:
    """``RetrievalQuery.extra`` 的五个键：两级优先级、封闭清单、必须进 notes."""

    def test_extra_overrides_the_constructor_parameters(self) -> None:
        """本次覆盖 mode / top_n / weight：摘要里读到的就是覆盖之后的那一组."""
        retriever = rerank_hybrid_retriever(
            rerank_enabled=True, rerank_top_n=1, rerank_mode=RERANK_MODE_REPLACE
        )
        result = retriever.retrieve(
            query_of(extra={"mode": RERANK_MODE_BLEND, "top_n": 10, "weight": 0.9})
        )

        assert rerank_summary_of(result)["mode"] == RERANK_MODE_BLEND
        assert rerank_summary_of(result)["params"] == {"top_n": 10, "weight": 0.9}
        assert rerank_summary_of(result)["scored"] == 10

    def test_extra_overrides_are_written_into_the_notes(self) -> None:
        """一次"这次用的是哪组参数"的改写必须可见（否则同一条查询两处结果不同却无法解释）."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=1)
        result = retriever.retrieve(query_of(extra={"top_n": 10}))
        note = next(note for note in result.notes if note.startswith("重排参数来自"))

        assert "本层默认 enabled=True、top_n=1、mode='replace'、weight=0.5" in note
        assert "本次覆盖了 top_n" in note

    def test_extra_can_turn_the_whole_layer_on(self) -> None:
        """``enabled=True`` 能在这一次打开整层（构造参数关着也可以）."""
        retriever = rerank_hybrid_retriever()
        result = retriever.retrieve(query_of(extra={"enabled": True, "top_n": 4}))

        assert rerank_summary_of(result)["params"]["top_n"] == 4
        assert rerank_summary_of(result)["scored"] == 4
        assert notes_starting_with(result, "重排参数来自")

    def test_extra_can_turn_the_whole_layer_off_for_one_call(self) -> None:
        """``enabled=False`` 能在这一次关掉整层，且**明说**那些覆盖参数没生效."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=10)
        result = retriever.retrieve(query_of(extra={"enabled": False, "top_n": 4}))
        note = next(note for note in result.notes if note.startswith("重排未启用"))

        assert result.rerank == {}
        assert result.ids() == list(HAND_FUSED_ORDER)
        assert "extra 里的 enabled、top_n 这次**没有作用**" in note
        assert "要在这一次打开重排请同时给 enabled=True" in note

    @pytest.mark.parametrize("enabled", ["yes", 1, 0, None])
    def test_extra_enabled_must_be_a_real_bool(self, enabled: Any) -> None:
        """``bool('no')`` 是 True：把"关闭"写成字符串会被静默读成"打开"."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)

        with pytest.raises(RerankError) as excinfo:
            retriever.retrieve(query_of(extra={"enabled": enabled}))

        assert "RetrievalQuery.extra['enabled'] 必须是布尔值" in str(excinfo.value)

    @pytest.mark.parametrize("key", ["mod", "weigth", "Top_n", "models"])
    def test_unknown_override_keys_are_rejected(self, key: str) -> None:
        """清单外的键当场报错，并把两份合法清单都列出来（融合层 + 重排层）."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)

        with pytest.raises(FusionError) as excinfo:
            retriever.retrieve(query_of(extra={key: "x"}))

        message = str(excinfo.value)
        assert f"本层不认识的键 ['{key}']" in message
        assert "重排层只认 enabled、mode、model、top_n、weight" in message

    def test_the_unknown_key_error_is_a_rerank_error_family_member(self) -> None:
        """它属于"参数写错"那一族（``QueryError``），端点那条 400 通道因此不必加分支."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)

        with pytest.raises(QueryError) as excinfo:
            retriever.retrieve(query_of(extra={"mod": "blend"}))

        assert not isinstance(excinfo.value, RerankError)

    def test_the_five_override_keys_are_accepted(self) -> None:
        """五个键各自都能被认领（``enabled`` / ``mode`` / ``model`` / ``top_n`` / ``weight``）."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)
        extra = {
            "enabled": True,
            "mode": RERANK_MODE_BLEND,
            "model": "my-gateway-alias",
            "top_n": 5,
            "weight": 0.25,
        }
        result = retriever.retrieve(query_of(extra=extra))
        summary = rerank_summary_of(result)

        assert summary["model"] == "my-gateway-alias"
        assert summary["params"] == {"top_n": 5, "weight": 0.25}
        assert set(extra) == set(RERANK_OVERRIDE_KEYS)

    def test_a_model_alias_only_changes_the_report(self) -> None:
        """``model`` 只进报告：它应当同时触发那条"名字不一致"的注记."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)
        result = retriever.retrieve(query_of(extra={"model": "my-gateway-alias"}))

        assert rerank_summary_of(result)["model"] == "my-gateway-alias"
        assert any("与实现的名字 'cross-encoder-teaching-v1' 不一致" in note
                   for note in result.notes)

    @pytest.mark.parametrize("top_n", [0, -1, 1.5, "4"])
    def test_bad_override_values_raise_from_the_query(self, top_n: Any) -> None:
        """覆盖值本身也要过同一份校验（覆盖不是"绕过校验"的通道）."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)

        with pytest.raises(RerankError) as excinfo:
            retriever.retrieve(query_of(extra={"top_n": top_n}))

        assert "top_n 必须是 >= 1 的整数" in str(excinfo.value)

    def test_an_empty_extra_changes_nothing(self) -> None:
        """``extra={}`` 与不带 ``extra`` 是同一件事（没有"看起来改了"的空覆盖）."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True, rerank_top_n=3)
        plain = retriever.retrieve(query_of())
        empty = retriever.retrieve(query_of(extra={}))

        assert plain.ids() == empty.ids()
        assert plain.rerank == empty.rerank
        assert not any(note.startswith("重排参数来自") for note in empty.notes)

    def test_the_default_top_n_is_twenty_when_nothing_is_given(self) -> None:
        """什么都不给时窗口是 ``DEFAULT_RERANK_TOP_N``（十条候选全进来）."""
        retriever = rerank_hybrid_retriever(rerank_enabled=True)

        assert retriever.retrieve(query_of()).rerank["params"]["top_n"] == DEFAULT_RERANK_TOP_N
