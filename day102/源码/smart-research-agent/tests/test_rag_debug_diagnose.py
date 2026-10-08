"""``rag_debug.diagnose`` 的归因判据（day071）.

这一份测试的重心不是"跑到某个分支"，而是**优先级**：一条坏例往往同时满足
多个判据，而函数必须报**最早的**那一层（检索没抓到的时候，"答案没有引用"
只是它的后果）。因此下面每一组用例都刻意把"多个判据同时成立"造出来，
再断言拿到的是链路顺序上更靠前的那个标签。
"""

from __future__ import annotations

import pytest

from smart_research_agent.rag_debug.diagnose import (
    action_plan,
    diagnose,
    diagnose_all,
    tag_counts,
)
from smart_research_agent.rag_debug.errors import DiagnosisError
from smart_research_agent.rag_debug.types import (
    BAD_CASE_PRIORITY,
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
)
from smart_research_agent.retrieval.generation import (
    FALLBACK_REASON_EMPTY_REPLY,
    FALLBACK_REASON_LLM_ERROR,
    FALLBACK_REASON_NO_CONTEXT,
)
from tests.rag_debug_samples import outcome


class TestPriorityOrder:
    def test_priority_covers_every_tag(self):
        assert set(BAD_CASE_PRIORITY) == {
            "no_data",
            "retrieval_empty",
            "retrieval_miss",
            "rerank_cut",
            "packed_away",
            "rank_bad",
            "truncated_chunk",
            "generation_fallback",
            "hallucinated_citation",
            "ungrounded",
            "low_coverage",
            "weak_faithfulness",
        }


class TestEmptyAndMiss:
    def test_no_data_wins_over_everything_else(self):
        # 库为空时"答案没有引用"只是后果，而它同时成立——必须先报库为空。
        case = diagnose(outcome(empty_reason="no_data", grounded=False, coverage=0.0))
        assert case is not None
        assert case.tag == TAG_NO_DATA
        assert "先灌语料" in case.action

    def test_retrieval_empty_reports_the_reason(self):
        case = diagnose(outcome(empty_reason="filtered_out", grounded=False, coverage=0.0))
        assert case is not None
        assert case.tag == TAG_RETRIEVAL_EMPTY
        assert case.evidence["empty_reason"] == "filtered_out"
        assert "filtered_out" in case.reason

    def test_retrieval_miss(self):
        case = diagnose(outcome(retrieved_ids=("d-2", "d-3"), packed_ids=()))
        assert case is not None
        assert case.tag == TAG_RETRIEVAL_MISS
        assert case.evidence["recall"] == 0.0

    def test_rerank_cut_before_packed_away(self):
        # 两条金标准只召回一条（recall 0.5）且重排切掉过候选 → 报第四道减法那一层。
        case = diagnose(
            outcome(
                relevant=("d-1", "d-2"),
                query="甲",
                retrieved_ids=("d-1",),
                packed_ids=("d-1",),
                dropped_by_rerank=3,
            )
        )
        assert case is not None
        assert case.tag == TAG_RERANK_CUT
        assert case.evidence["dropped_by_rerank"] == 3

    def test_rerank_cut_ignored_when_recall_is_full(self):
        # 金标准全在名单里时，重排切掉的只是无关候选——不该判坏例。
        case = diagnose(
            outcome(
                retrieved_ids=("d-1", "d-2"),
                packed_ids=("d-1", "d-2"),
                dropped_by_rerank=2,
            )
        )
        assert case is None

    def test_retrieval_miss_wins_over_rerank_cut(self):
        case = diagnose(
            outcome(retrieved_ids=("d-2",), packed_ids=(), dropped_by_rerank=1)
        )
        assert case is not None
        assert case.tag == TAG_RETRIEVAL_MISS


class TestPackingAndRanking:
    def test_packed_away_is_its_own_failure(self):
        # 检索召回 1.0，packed 召回 0.0：抓到了但模型看不到。
        case = diagnose(
            outcome(
                retrieved_ids=("d-1", "d-2"),
                packed_ids=("d-2",),
                dropped_hits=1,
                given=1,
                cited=1,
                valid_count=1,
                unused_count=0,
            )
        )
        assert case is not None
        assert case.tag == TAG_PACKED_AWAY
        assert "预算" in case.action
        assert case.evidence["retrieved_recall"] == pytest.approx(1.0)

    def test_packed_away_wins_over_ungrounded(self):
        case = diagnose(
            outcome(
                retrieved_ids=("d-1", "d-2"),
                packed_ids=("d-2",),
                grounded=False,
                coverage=0.0,
                given=1,
                cited=0,
                valid_count=0,
                unused_count=1,
            )
        )
        assert case is not None
        assert case.tag == TAG_PACKED_AWAY

    def test_rank_bad_when_golden_is_late(self):
        # 金标准进了提示词但排在第 2 位（RR = 0.5，下限 0.5 → 不算；用更靠后造）
        case = diagnose(
            outcome(
                retrieved_ids=("d-2", "d-3", "d-1"),
                packed_ids=("d-2", "d-3", "d-1"),
            )
        )
        assert case is not None
        assert case.tag == TAG_RANK_BAD
        assert "第 3 位" in case.reason

    def test_rank_bad_respects_the_threshold(self):
        # 排在第 2 位（RR = 0.5）时按缺省下限不算坏例。
        assert diagnose(outcome(retrieved_ids=("d-2", "d-1"), packed_ids=("d-2", "d-1"))) is None
        # 把下限提到 1.0 之后必须排第 1 才通过。
        case = diagnose(
            outcome(retrieved_ids=("d-2", "d-1"), packed_ids=("d-2", "d-1")),
            min_reciprocal_rank=1.0,
        )
        assert case is not None
        assert case.tag == TAG_RANK_BAD


class TestGenerationSide:
    def test_truncated_chunk_needs_both_signals(self):
        case = diagnose(
            outcome(truncated_hits=2, coverage=0.4, given=5, cited=2, valid_count=2, unused_count=3)
        )
        assert case is not None
        assert case.tag == TAG_TRUNCATED_CHUNK
        assert case.evidence["truncated_hits"] == 2

    def test_truncation_alone_is_not_a_bad_case(self):
        # 截断但覆盖率足够高（答案用上了给出去的片段）→ 不判坏例。
        assert (
            diagnose(
                outcome(
                    truncated_hits=1,
                    coverage=1.0,
                    given=2,
                    cited=2,
                    valid_count=2,
                    unused_count=0,
                )
            )
            is None
        )

    def test_fallback_uses_the_action_table(self):
        case = diagnose(outcome(fallback_reason=FALLBACK_REASON_LLM_ERROR, llm_called=False))
        assert case is not None
        assert case.tag == TAG_GENERATION_FALLBACK
        assert "查提供方" in case.action
        assert case.evidence["fallback_reason"] == FALLBACK_REASON_LLM_ERROR

    def test_fallback_no_context_from_generator_side(self):
        # 检索非空但打包为空 → 生成器报 no_context，动作落回那张表的第一条。
        case = diagnose(outcome(fallback_reason=FALLBACK_REASON_NO_CONTEXT, llm_called=False))
        assert case is not None
        assert case.tag == TAG_GENERATION_FALLBACK
        assert "回到检索层" in case.action

    def test_fallback_empty_reply(self):
        case = diagnose(outcome(fallback_reason=FALLBACK_REASON_EMPTY_REPLY))
        assert case is not None
        assert case.tag == TAG_GENERATION_FALLBACK
        assert "空串" in case.reason

    def test_hallucinated_citation_wins_over_ungrounded(self):
        case = diagnose(
            outcome(
                grounded=False,
                hallucinated=1,
                cited=3,
                given=2,
                valid_count=2,
                unused_count=0,
            )
        )
        assert case is not None
        assert case.tag == TAG_HALLUCINATED_CITATION
        assert case.evidence["hallucinated"] == 1

    def test_ungrounded(self):
        case = diagnose(
            outcome(
                grounded=False,
                cited=0,
                given=2,
                valid_count=0,
                unused_count=2,
            )
        )
        assert case is not None
        assert case.tag == TAG_UNGROUNDED
        assert case.evidence["given"] == 2

    def test_ungrounded_can_be_turned_off(self):
        # require_grounded=False 时只量"不做接地要求"的那一份数字：
        # 这一条不再判 ungrounded，但它同时覆盖率也只有 0.25 → 落到低覆盖那一档。
        payload = {
            "grounded": False,
            "cited": 0,
            "given": 2,
            "valid_count": 0,
            "unused_count": 2,
            "coverage": 0.25,
        }
        assert diagnose(outcome(**payload)) is not None  # 打开时判 ungrounded
        strict = diagnose(outcome(**payload))
        assert strict is not None
        assert strict.tag == TAG_UNGROUNDED
        loose = diagnose(outcome(**payload), require_grounded=False)
        assert loose is not None
        assert loose.tag == TAG_LOW_COVERAGE

    def test_low_coverage(self):
        case = diagnose(
            outcome(coverage=0.25, given=4, cited=1, valid_count=1, unused_count=3)
        )
        assert case is not None
        assert case.tag == TAG_LOW_COVERAGE
        assert case.evidence["given"] == 4
        assert case.evidence["coverage"] == pytest.approx(0.25)

    def test_weak_faithfulness_is_the_last_resort(self):
        case = diagnose(
            outcome(faithfulness=0.2, coverage=1.0, given=2, cited=2, valid_count=2, unused_count=0)
        )
        assert case is not None
        assert case.tag == TAG_WEAK_FAITHFULNESS
        assert case.evidence["faithfulness"] == pytest.approx(0.2)

    def test_clean_case_returns_none(self):
        assert (
            diagnose(
                outcome(
                    faithfulness=1.0,
                    coverage=1.0,
                    given=2,
                    cited=2,
                    valid_count=2,
                    unused_count=0,
                )
            )
            is None
        )


class TestThresholdValidation:
    def test_min_coverage_must_be_ratio(self):
        with pytest.raises(DiagnosisError, match="落在"):
            diagnose(outcome(), min_coverage=1.5)

    def test_min_coverage_must_be_number(self):
        with pytest.raises(DiagnosisError, match="必须是数字"):
            diagnose(outcome(), min_coverage="高")

    def test_min_faithfulness_must_be_ratio(self):
        with pytest.raises(DiagnosisError, match="落在"):
            diagnose(outcome(), min_faithfulness=-0.1)

    def test_min_reciprocal_rank_must_be_positive(self):
        with pytest.raises(DiagnosisError, match=r"\(0, 1\]"):
            diagnose(outcome(), min_reciprocal_rank=0.0)

    def test_require_grounded_must_be_bool(self):
        with pytest.raises(DiagnosisError, match="必须是布尔值"):
            diagnose(outcome(), require_grounded="yes")


class TestBatchHelpers:
    def test_diagnose_all_only_collects_bad_cases(self):
        outcomes = (
            outcome(),
            outcome(retrieved_ids=("d-2",), packed_ids=()),
            outcome(empty_reason="no_data", grounded=False, coverage=0.0),
        )
        cases = diagnose_all(outcomes)
        assert [item.tag for item in cases] == [TAG_RETRIEVAL_MISS, TAG_NO_DATA]

    def test_tag_counts_is_sorted_and_sparse(self):
        cases = (
            BadCase(query="a", tag=TAG_NO_DATA, reason="r", action="动作 A"),
            BadCase(query="b", tag=TAG_NO_DATA, reason="r", action="动作 A"),
            BadCase(query="c", tag=TAG_LOW_COVERAGE, reason="r", action="动作 B"),
        )
        assert tag_counts(cases) == {TAG_LOW_COVERAGE: 1, TAG_NO_DATA: 2}

    def test_action_plan_deduplicates_in_order(self):
        cases = (
            BadCase(query="a", tag=TAG_NO_DATA, reason="r", action="动作 A"),
            BadCase(query="b", tag=TAG_LOW_COVERAGE, reason="r", action="动作 B"),
            BadCase(query="c", tag=TAG_NO_DATA, reason="r", action="动作 A"),
        )
        assert action_plan(cases) == ("动作 A", "动作 B")

    def test_action_plan_on_empty_input(self):
        assert action_plan(()) == ()
