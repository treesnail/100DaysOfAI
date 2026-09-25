"""``rag_debug.types`` 的形状与三张封闭表（day071）.

每个期望值都是**手算或按定义**得出的（见 ``rag_debug_samples`` 的模块
docstring），而所有"校验类"用例断言的是**报错消息里有没有那句能照做的话**——
一条只说"参数错"的报错在排查时等于没说。
"""

from __future__ import annotations

import pytest

from smart_research_agent.rag_debug.errors import CaseDataError, DiagnosisError
from smart_research_agent.rag_debug.types import (
    BAD_CASE_ACTIONS,
    BAD_CASE_DESCRIPTIONS,
    BAD_CASE_PRIORITY,
    BAD_CASE_TAGS,
    DEFAULT_MIN_COVERAGE,
    FALLBACK_ACTIONS,
    RETRIEVAL_METRICS,
    STAGE_PACKED,
    STAGE_RETRIEVED,
    STAGES,
    TAG_NO_DATA,
    TAG_PACKED_AWAY,
    BadCase,
    CaseOutcome,
    RagEvalCase,
    StageMetrics,
    bad_case_rate,
    described_tags,
    mean_of,
    measure_ids,
)
from smart_research_agent.retrieval.generation import FALLBACK_REASON_DESCRIPTIONS
from tests.rag_debug_samples import EXPECTED_CASE1_METRICS, EXPECTED_IDS, outcome

# --------------------------------------------------------------------------- #
# 三张封闭表
# --------------------------------------------------------------------------- #


class TestClosedTables:
    def test_tags_descriptions_actions_align(self):
        # 三张表逐键对齐：少一个键意味着某类坏例没有解释或没有动作。
        assert set(BAD_CASE_TAGS) == set(BAD_CASE_DESCRIPTIONS) == set(BAD_CASE_ACTIONS)

    def test_priority_is_the_same_order_as_tags(self):
        # 归因优先级就是标签表的排列（链路顺序）——单列一份是为了可读，
        # 但两者必须逐位相同，否则"先查检索"这条纪律就只写在注释里。
        assert BAD_CASE_PRIORITY == BAD_CASE_TAGS

    def test_fallback_actions_cover_every_reason(self):
        # 六种回退 + "没有回退"那一格都要有处置动作。
        assert set(FALLBACK_ACTIONS) == set(FALLBACK_REASON_DESCRIPTIONS)

    def test_described_tags_shape(self):
        table = described_tags()
        assert set(table) == set(BAD_CASE_TAGS)
        assert table[TAG_PACKED_AWAY]["action"] == BAD_CASE_ACTIONS[TAG_PACKED_AWAY]
        assert "预算" in table[TAG_PACKED_AWAY]["description"]

    def test_stage_names(self):
        assert STAGES == (STAGE_RETRIEVED, STAGE_PACKED)
        assert RETRIEVAL_METRICS == ("recall", "precision", "reciprocal_rank", "ndcg")
        assert DEFAULT_MIN_COVERAGE == 0.5


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #


class TestRagEvalCase:
    def _case(self, **overrides):
        payload = {
            "query": "甲",
            "relevant": ("d-1",),
            "reference": "参考答案",
            "grades": {"d-1": 2.0},
        }
        payload.update(overrides)
        return RagEvalCase(**payload)

    def test_valid_case_projects_and_summarizes(self):
        case = self._case()
        assert case.summary_line().startswith("用例 '甲'")
        assert case.to_dict()["relevant"] == ["d-1"]
        assert case.to_dict()["grades"] == {"d-1": 2.0}

    def test_empty_query_rejected(self):
        with pytest.raises(CaseDataError, match="非空字符串"):
            self._case(query="   ")

    def test_relevant_must_be_tuple(self):
        with pytest.raises(CaseDataError, match="必须是 tuple"):
            self._case(relevant=["d-1"])

    def test_relevant_cannot_be_empty(self):
        with pytest.raises(CaseDataError, match="不能为空"):
            self._case(relevant=())

    def test_relevant_items_must_be_non_empty_strings(self):
        with pytest.raises(CaseDataError, match="非字符串或空串"):
            self._case(relevant=("d-1", " "), grades={"d-1": 1.0})

    def test_reference_must_be_string(self):
        with pytest.raises(CaseDataError, match="必须是字符串"):
            self._case(reference=None)

    def test_grades_must_be_dict(self):
        with pytest.raises(CaseDataError, match="必须是字典"):
            self._case(grades=[("d-1", 1.0)])

    def test_grade_keys_must_be_non_empty_strings(self):
        with pytest.raises(CaseDataError, match="空串键"):
            self._case(grades={"": 1.0})

    def test_grade_values_must_be_numbers(self):
        with pytest.raises(CaseDataError, match="必须是数字"):
            self._case(grades={"d-1": "很高"})

    def test_grade_values_must_be_non_negative_finite(self):
        with pytest.raises(CaseDataError, match="非负有限数"):
            self._case(grades={"d-1": -1.0})

    def test_all_zero_grades_rejected(self):
        # IDCG = 0 时 NDCG 恒为 0，而那个 0 读起来像"排序很差"。
        with pytest.raises(CaseDataError, match="全是 0"):
            self._case(grades={"d-1": 0.0})

    def test_grades_may_include_records_outside_relevant(self):
        # 项目自带评测集就是这个形态：等级 1 = 部分相关，不进 recall 的二值口径。
        case = self._case(grades={"d-1": 2.0, "d-2": 1.0})
        assert case.grades == {"d-1": 2.0, "d-2": 1.0}
        assert case.relevant == ("d-1",)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


class TestMeasureIds:
    def test_hand_computed_case_one(self):
        result = measure_ids(
            EXPECTED_IDS, ("d-1",), 3, grades={"d-1": 2, "d-2": 1}
        )
        assert result["recall"] == pytest.approx(EXPECTED_CASE1_METRICS["recall"])
        assert result["precision"] == pytest.approx(EXPECTED_CASE1_METRICS["precision"])
        assert result["reciprocal_rank"] == pytest.approx(
            EXPECTED_CASE1_METRICS["reciprocal_rank"]
        )
        assert result["ndcg"] == pytest.approx(EXPECTED_CASE1_METRICS["ndcg"])

    def test_binary_ndcg_when_no_grades(self):
        # 没有分级标注时走 day068 的二值 @k 版本：命中记 1、理想排序按金标准条数。
        result = measure_ids(("d-1", "d-2"), ("d-1",), 3)
        assert result["ndcg"] == pytest.approx(1.0)

    def test_precision_only_counts_first_k(self):
        # 名单 5 条、k=2、命中 1 条 → 1/2（不是 1/5："给得多"不该被惩罚）。
        result = measure_ids(("d-1", "d-2", "d-3", "d-4", "d-5"), ("d-1",), 2)
        assert result["precision"] == pytest.approx(0.5)

    def test_recall_only_counts_first_k(self):
        # 金标准排在 k 之外时 recall@k 记 0——recall 也是 @k 指标。
        result = measure_ids(("d-2", "d-3", "d-4", "d-1"), ("d-1",), 3)
        assert result["recall"] == pytest.approx(0.0)

    def test_reciprocal_rank_counts_position(self):
        result = measure_ids(("d-2", "d-1"), ("d-1",), 3)
        assert result["reciprocal_rank"] == pytest.approx(0.5)

    def test_invalid_k_rejected(self):
        with pytest.raises(CaseDataError, match=">= 1 的整数"):
            measure_ids(("d-1",), ("d-1",), 0)


# --------------------------------------------------------------------------- #
# 两段指标
# --------------------------------------------------------------------------- #


class TestStageMetrics:
    def _metrics(self, **overrides):
        payload = {
            "stage": STAGE_RETRIEVED,
            "ids": ("d-1",),
            "recall": 1.0,
            "precision": 0.5,
            "reciprocal_rank": 1.0,
            "ndcg": 1.0,
        }
        payload.update(overrides)
        return StageMetrics(**payload)

    def test_projects_and_summarizes(self):
        metrics = self._metrics()
        assert metrics.count == 1
        assert metrics.value("recall") == 1.0
        assert metrics.to_dict()["stage"] == STAGE_RETRIEVED
        assert "recall 1.0000" in metrics.summary_line()

    def test_unknown_stage_rejected(self):
        with pytest.raises(DiagnosisError, match="retrieved"):
            self._metrics(stage="merged")

    def test_ids_must_be_tuple(self):
        with pytest.raises(DiagnosisError, match="必须是 tuple"):
            self._metrics(ids=["d-1"])

    def test_ids_items_must_be_non_empty(self):
        with pytest.raises(DiagnosisError, match="非字符串或空串"):
            self._metrics(ids=("d-1", ""))

    def test_metric_must_be_number(self):
        with pytest.raises(DiagnosisError, match="必须是数字"):
            self._metrics(recall="很高")

    def test_metric_must_be_within_unit_interval(self):
        with pytest.raises(DiagnosisError, match=r"落在 \[0, 1\]"):
            self._metrics(precision=1.5)

    def test_unknown_metric_name_rejected(self):
        with pytest.raises(DiagnosisError, match="不认识的指标名"):
            self._metrics().value("hit_rate")


# --------------------------------------------------------------------------- #
# 一次运行的结果
# --------------------------------------------------------------------------- #


class TestCaseOutcome:
    def test_valid_outcome_projects(self):
        result = outcome()
        data = result.to_dict()
        assert data["retrieved"]["count"] == 2
        assert data["packed"]["stage"] == STAGE_PACKED
        assert data["packed_away"] == 0
        assert "召回" in result.summary_line()

    def test_packed_away_counts_missing_golden_records(self):
        # 检索召回 1.0、packed 召回 0.5（金标准 1 条）→ 丢掉 1 条依据。
        result = outcome(retrieved_ids=("d-1", "d-2"), packed_ids=("d-2",))
        assert result.packed_away == 1

    def test_query_must_be_non_empty(self):
        with pytest.raises(CaseDataError, match="非空字符串"):
            outcome(query=" ")

    def test_two_stages_must_be_stage_metrics(self):
        with pytest.raises(DiagnosisError, match="必须是 StageMetrics"):
            CaseOutcome(query="甲", relevant=("d-1",), retrieved=("d-1",), packed=("d-1",))

    def test_two_stages_cannot_be_swapped(self):
        result = outcome()
        with pytest.raises(DiagnosisError, match="装反了"):
            CaseOutcome(
                query="甲",
                relevant=("d-1",),
                retrieved=result.packed,
                packed=result.retrieved,
            )

    def test_empty_reason_must_come_from_closed_list(self):
        with pytest.raises(CaseDataError, match="empty_reason"):
            outcome(empty_reason="没找到")

    def test_fallback_reason_must_come_from_closed_list(self):
        with pytest.raises(CaseDataError, match="fallback_reason"):
            outcome(fallback_reason="模型没答")

    def test_bool_fields_are_checked(self):
        with pytest.raises(CaseDataError, match="llm_called 必须是布尔值"):
            outcome(llm_called="yes")

    def test_coverage_must_be_number(self):
        with pytest.raises(CaseDataError, match="coverage 必须是数字"):
            outcome(coverage="高")

    def test_coverage_must_be_within_unit_interval(self):
        with pytest.raises(CaseDataError, match=r"落在 \[0, 1\]"):
            outcome(coverage=1.2)

    def test_counts_must_be_non_negative_ints(self):
        with pytest.raises(CaseDataError, match="必须是非负整数"):
            outcome(given=-1)

    def test_valid_plus_unused_must_equal_given(self):
        with pytest.raises(CaseDataError, match="账不自洽"):
            outcome(given=3, cited=2, valid_count=2, unused_count=0)

    def test_valid_cannot_exceed_cited(self):
        with pytest.raises(CaseDataError, match="不可能大于"):
            outcome(given=2, cited=1, valid_count=2, unused_count=0)

    def test_cited_cannot_exceed_given_plus_hallucinated(self):
        with pytest.raises(CaseDataError, match="账不自洽"):
            outcome(given=1, cited=3, valid_count=1, unused_count=0, hallucinated=1)

    def test_answer_chars_must_be_non_negative_int(self):
        with pytest.raises(CaseDataError, match="answer_chars"):
            outcome(answer_chars=-3)

    def test_latency_must_be_finite_non_negative(self):
        with pytest.raises(CaseDataError, match="非负有限数"):
            outcome(latency_ms=float("inf"))

    def test_latency_must_be_number(self):
        with pytest.raises(CaseDataError, match="latency_ms 必须是数字"):
            outcome(latency_ms="很快")

    def test_faithfulness_must_be_number_or_none(self):
        with pytest.raises(CaseDataError, match="必须是数字或 None"):
            outcome(faithfulness="好")

    def test_faithfulness_must_be_within_unit_interval(self):
        with pytest.raises(CaseDataError, match=r"落在 \[0, 1\]"):
            outcome(faithfulness=1.5)

    def test_notes_must_be_tuple(self):
        with pytest.raises(CaseDataError, match="notes 必须是 tuple"):
            outcome(notes=["一句"])

    def test_notes_cannot_contain_blank_items(self):
        with pytest.raises(CaseDataError, match="空条目"):
            outcome(notes=("  ",))

    def test_faithfulness_appears_in_summary_and_projection(self):
        result = outcome(faithfulness=0.5, notes=("评审失败",))
        assert "忠实度 0.50" in result.summary_line()
        assert result.to_dict()["faithfulness"] == pytest.approx(0.5)
        assert result.to_dict()["notes"] == ["评审失败"]


# --------------------------------------------------------------------------- #
# 归因记录
# --------------------------------------------------------------------------- #


class TestBadCase:
    def test_projects_with_description(self):
        case = BadCase(
            query="甲",
            tag=TAG_NO_DATA,
            reason="库是空的",
            action="先建索引",
            evidence={"empty_reason": "no_data"},
        )
        data = case.to_dict()
        assert data["description"] == BAD_CASE_DESCRIPTIONS[TAG_NO_DATA]
        assert data["evidence"] == {"empty_reason": "no_data"}
        assert case.summary_line().startswith("[no_data]")

    def test_query_must_be_non_empty(self):
        with pytest.raises(CaseDataError, match="非空字符串"):
            BadCase(query=" ", tag=TAG_NO_DATA, reason="r", action="a")

    def test_unknown_tag_rejected(self):
        with pytest.raises(DiagnosisError, match="封闭清单|不认识的坏例标签"):
            BadCase(query="甲", tag="something_else", reason="r", action="a")

    def test_reason_and_action_must_be_non_empty(self):
        with pytest.raises(DiagnosisError, match="必须是非空字符串"):
            BadCase(query="甲", tag=TAG_NO_DATA, reason="r", action=" ")
        with pytest.raises(DiagnosisError, match="必须是非空字符串"):
            BadCase(query="甲", tag=TAG_NO_DATA, reason="", action="a")

    def test_evidence_must_be_dict(self):
        with pytest.raises(DiagnosisError, match="必须是字典"):
            BadCase(query="甲", tag=TAG_NO_DATA, reason="r", action="a", evidence=["x"])


# --------------------------------------------------------------------------- #
# 两个算术小工具
# --------------------------------------------------------------------------- #


class TestArithmeticHelpers:
    def test_bad_case_rate_zero_cases(self):
        # 0 条用例时返回 0.0 而不是除零异常；"评了 0 条"由报告的 cases 字段回答。
        assert bad_case_rate(0, 0) == 0.0

    def test_bad_case_rate_rounds(self):
        assert bad_case_rate(1, 3) == pytest.approx(0.3333)

    def test_bad_case_rate_rejects_bool(self):
        with pytest.raises(CaseDataError, match="整数"):
            bad_case_rate(True, 3)

    def test_bad_case_rate_rejects_negative(self):
        with pytest.raises(CaseDataError, match="非负整数"):
            bad_case_rate(-1, 3)

    def test_bad_case_rate_rejects_more_cases_than_total(self):
        with pytest.raises(CaseDataError, match="大于用例总数"):
            bad_case_rate(5, 3)

    def test_mean_of_empty_is_zero(self):
        assert mean_of([]) == 0.0

    def test_mean_of_rounds_to_four_digits(self):
        assert mean_of([1.0, 0.0, 0.0]) == pytest.approx(0.3333)
