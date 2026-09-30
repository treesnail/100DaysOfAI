"""day057 两级去重测试：精确键先挡、MinHash 再挡近重复（全部离线、确定性）.

本文件守的是**去重的记账纪律**与**边界行为**：

1. **两级顺序不能反**：一条逐字相同的样本必须由精确键（确定性）裁决，
   不能因为签名恰好差一位而被放行；
2. **保留首次出现**：重复样本不入库，因此索引里只有"已接受"的样本，
   增量场景下"重复的重复"才会被同一套判据继续挡掉；
3. **报告恒等式** ``exact_duplicates + near_duplicates + kept == total``：
   三个数对不上就说明有样本在统计里静默消失，这是最难查的一类故障；
4. **阈值边界**：0.7 是本课实测标定出来的（0.8 会让近重复链路空转），
   所以必须拿一对真实的近重复样本钉住"0.7 判重复、0.95 判保留"。

样本全部由 ``TrainingExample(...)`` 现场构造，不读任何真实数据文件。
"""

from __future__ import annotations

import pytest

from smart_research_agent.domain_data.dedup import (
    DEFAULT_NEAR_DUP_THRESHOLD,
    STATUS_EXACT,
    STATUS_KEPT,
    STATUS_NEAR,
    DedupeReport,
    NearDuplicateIndex,
    dedupe_examples,
)
from smart_research_agent.domain_data.fingerprint import DEFAULT_SHINGLE_K
from smart_research_agent.finetune.schema import TrainingExample

#: 基准提问（15 字，k=3 时 13 个 shingle）
PROMPT = "如何评估 RAG 的检索质量？"

#: "同一份语料被加了两字前缀"——实测估计相似度 0.8906（57/64），
#: 是阈值边界测试的主角：0.7 判近重复、0.95 判保留。
PROMPT_WITH_TWO_CHAR_PREFIX = "请问" + PROMPT

#: "掉了最后一个问号"的副本——实测估计相似度 0.9375（60/64）
PROMPT_TRUNCATED = PROMPT[:-1]

#: 字面完全无关的对照样本（实测估计相似度 0.0）
FAR_PROMPT = "如何配置 MCP 服务器并接入本地文件系统？请给出可复现的步骤。"

ANSWER = "可以从命中率、召回率与 MRR 三个指标评估检索质量。"


def make_example(prompt: str, output: str = ANSWER, source: str = "docs") -> TrainingExample:
    """构造一条最小样本（同一 prompt 配不同 output 是"冲突"而不是"冗余"）."""
    return TrainingExample(instruction=prompt, output=output, source=source)


class TestConstants:
    """常量即策略，逐个钉住."""

    def test_near_duplicate_threshold_is_calibrated_to_seven_tenths(self):
        """0.7 是本课程语料的标定值：0.8 会一条近重复都检不出来."""
        assert DEFAULT_NEAR_DUP_THRESHOLD == 0.7

    def test_status_constants_are_distinct(self):
        assert STATUS_KEPT == "kept"
        assert STATUS_EXACT == "exact_duplicate"
        assert STATUS_NEAR == "near_duplicate"
        assert len({STATUS_KEPT, STATUS_EXACT, STATUS_NEAR}) == 3

    def test_index_defaults_follow_fingerprint_module(self):
        index = NearDuplicateIndex()
        assert index.threshold == DEFAULT_NEAR_DUP_THRESHOLD
        assert index.k == DEFAULT_SHINGLE_K


class TestThresholdValidation:
    """阈值必须落在 ``(0, 1]``：0 会清空整批，>1 则近重复静默失效."""

    @pytest.mark.parametrize("threshold", [0.0, -0.1, -1.0, 1.5, 2.0])
    def test_out_of_range_rejected_at_construction(self, threshold):
        with pytest.raises(ValueError, match=r"近重复阈值必须落在 \(0, 1\] 区间"):
            NearDuplicateIndex(threshold=threshold)

    @pytest.mark.parametrize("threshold", [1.0, 0.7, 0.01])
    def test_boundaries_are_accepted(self, threshold):
        """``1.0`` 是闭区间上界（只留精确重复），必须被接受."""
        assert NearDuplicateIndex(threshold=threshold).threshold == threshold


class TestAdd:
    """``add``：判定 + 入库，重复样本不入库."""

    def test_first_occurrence_is_kept_and_indexed(self):
        index = NearDuplicateIndex()
        decision = index.add(make_example(PROMPT))
        assert decision.status == STATUS_KEPT
        assert decision.similarity == 0.0  # 空索引里没有比较对象
        assert decision.partner_index is None
        assert decision.is_duplicate is False
        assert len(index) == 1

    def test_exact_repeat_is_exact_duplicate_with_full_similarity(self):
        """同一 prompt 再来一次 → 精确重复、相似度 1.0、指回第 0 条."""
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        decision = index.add(make_example(PROMPT, output="换一个答案，长度也完全不同。"))
        assert decision.status == STATUS_EXACT
        assert decision.similarity == 1.0
        assert decision.partner_index == 0
        assert decision.is_duplicate is True

    def test_duplicate_is_not_stored(self):
        """重复样本不入库：入库后它会成为后续样本的比较对象，重复会被反复计数."""
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        index.add(make_example(PROMPT))
        assert len(index) == 1

    def test_whitespace_variant_counts_as_exact(self):
        """归一化后相同即精确重复——"多打了两个空格"不是一条新数据."""
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        assert index.add(make_example(f"  {PROMPT}  ")).status == STATUS_EXACT

    def test_near_prefix_duplicate_is_detected(self):
        """加两字前缀的同一条语料：估计 0.8906 ≥ 0.7 → 近重复，并指回第 0 条."""
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        decision = index.add(make_example(PROMPT_WITH_TWO_CHAR_PREFIX))
        assert decision.status == STATUS_NEAR
        assert decision.similarity == pytest.approx(0.8906, abs=1e-4)
        assert decision.partner_index == 0
        assert len(index) == 1

    def test_unrelated_sample_is_kept_and_indexed(self):
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        decision = index.add(make_example(FAR_PROMPT))
        assert decision.status == STATUS_KEPT
        assert decision.similarity == 0.0
        assert len(index) == 2

    def test_decision_to_dict_is_json_ready(self):
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        decision = index.add(make_example(PROMPT))
        payload = decision.to_dict()
        assert payload["status"] == STATUS_EXACT
        assert payload["similarity"] == 1.0
        assert payload["source"] == "docs"
        assert payload["preview"] == PROMPT
        assert set(payload) == {"status", "similarity", "partner_index", "source", "preview"}


class TestClassifyIsReadOnly:
    """``classify`` 是只读查询：问"这条会不会被挡"，不改索引."""

    def test_classify_does_not_grow_index(self):
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        decision = index.classify(make_example(PROMPT))
        assert decision.status == STATUS_EXACT
        assert len(index) == 1

    def test_classify_reports_near_and_kept_without_indexing(self):
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        assert index.classify(make_example(PROMPT_WITH_TWO_CHAR_PREFIX)).status == STATUS_NEAR
        assert index.classify(make_example(FAR_PROMPT)).status == STATUS_KEPT
        assert len(index) == 1

    def test_classify_on_empty_index_is_kept(self):
        assert NearDuplicateIndex().classify(make_example(PROMPT)).status == STATUS_KEPT

    def test_fingerprints_property_returns_a_copy(self):
        """返回的是副本：调用方往列表里塞东西不该污染索引（报告/诊断用）."""
        index = NearDuplicateIndex()
        index.add(make_example(PROMPT))
        snapshot = index.fingerprints
        snapshot.clear()
        assert len(index) == 1


class TestIncrementalIndex:
    """增量：历史先进索引，新批次 ``filter`` 时跨批次判重."""

    def test_add_many_returns_accepted_count(self):
        index = NearDuplicateIndex()
        accepted = index.add_many(
            [make_example(PROMPT), make_example(PROMPT), make_example(FAR_PROMPT)]
        )
        assert accepted == 2
        assert len(index) == 2

    def test_cross_batch_exact_duplicate_detected(self):
        """核心场景：不重读历史，新批次里的逐字副本照样被挡下."""
        index = NearDuplicateIndex()
        index.add_many([make_example(PROMPT)])
        kept, report = index.filter([make_example(PROMPT, output="另一份答案。")])
        assert kept == []
        assert report.exact_duplicates == 1
        assert report.decisions[0].status == STATUS_EXACT

    def test_cross_batch_near_duplicate_detected(self):
        index = NearDuplicateIndex()
        index.add_many([make_example(PROMPT)])
        kept, report = index.filter([make_example(PROMPT_WITH_TWO_CHAR_PREFIX)])
        assert kept == []
        assert report.near_duplicates == 1
        assert report.decisions[0].similarity == pytest.approx(0.8906, abs=1e-4)

    def test_filter_indexes_what_it_keeps(self):
        """``filter`` 会把保留样本一并入库，因此批内重复也能被挡下."""
        index = NearDuplicateIndex()
        index.filter([make_example(PROMPT)])
        assert len(index) == 1
        assert index.add(make_example(PROMPT)).status == STATUS_EXACT


class TestFilterReport:
    """``filter`` 的记账：恒等式、顺序、保留率与丢弃原因."""

    def test_counts_satisfy_the_identity(self):
        """``精确 + 近重复 + 保留 == 总数``：样本不许在统计里静默消失."""
        kept, report = dedupe_examples(
            [
                make_example(PROMPT),
                make_example(PROMPT),
                make_example(PROMPT_WITH_TWO_CHAR_PREFIX),
                make_example(PROMPT_TRUNCATED),
                make_example(FAR_PROMPT),
            ]
        )
        assert report.total == 5
        assert report.exact_duplicates == 1
        assert report.near_duplicates == 2
        assert report.kept == len(kept) == 2
        assert report.exact_duplicates + report.near_duplicates + report.kept == report.total
        assert report.duplicates == 3

    def test_decisions_preserve_input_order_and_match_counts(self):
        _, report = dedupe_examples(
            [
                make_example(PROMPT),
                make_example(PROMPT),
                make_example(PROMPT_WITH_TWO_CHAR_PREFIX),
            ]
        )
        assert [decision.status for decision in report.decisions] == [
            STATUS_KEPT,
            STATUS_EXACT,
            STATUS_NEAR,
        ]
        assert report.decisions[2].partner_index == 0

    def test_kept_samples_keep_relative_order(self):
        first = make_example(PROMPT)
        second = make_example(FAR_PROMPT)
        kept, _ = dedupe_examples([first, second])
        assert kept == [first, second]

    def test_drop_reasons_only_lists_non_zero_counts(self):
        _, exact_only = dedupe_examples([make_example(PROMPT), make_example(PROMPT)])
        assert exact_only.drop_reasons() == {STATUS_EXACT: 1}

        _, clean = dedupe_examples([make_example(PROMPT)])
        assert clean.drop_reasons() == {}

    def test_keep_rate_and_threshold_are_reported(self):
        _, report = dedupe_examples(
            [make_example(PROMPT), make_example(PROMPT), make_example(FAR_PROMPT)]
        )
        assert report.keep_rate == pytest.approx(2 / 3)
        assert report.threshold == DEFAULT_NEAR_DUP_THRESHOLD

    def test_empty_batch_reports_zero_keep_rate(self):
        """空输入不抛错，且保留率按 0.0 计（与 day048 的 ``FilterReport`` 同一约定）."""
        kept, report = dedupe_examples([])
        assert kept == []
        assert report.total == 0
        assert report.keep_rate == 0.0
        assert report.decisions == []

    def test_to_dict_only_lists_duplicate_decisions(self):
        """报告只列被丢弃的决策：报告是"丢了什么"的档案，不该混进保留项."""
        _, report = dedupe_examples(
            [make_example(PROMPT), make_example(PROMPT), make_example(FAR_PROMPT)]
        )
        payload = report.to_dict()
        assert payload["total"] == 3
        assert payload["kept"] == 2
        assert payload["duplicates"] == 1
        assert len(payload["decisions"]) == payload["duplicates"] == 1
        assert payload["keep_rate"] == pytest.approx(0.6667, abs=1e-4)

    def test_summary_line_reports_counts_and_threshold(self):
        _, report = dedupe_examples([make_example(PROMPT), make_example(PROMPT)])
        line = report.summary_line()
        assert "去重：2 条进 / 1 条留" in line
        assert "精确重复 1" in line
        assert "0.7" in line

    def test_convenience_wrapper_matches_explicit_index(self):
        """``dedupe_examples`` 只是"新建空索引 + filter"的薄封装，两者必须同结论."""
        examples = [make_example(PROMPT), make_example(PROMPT_WITH_TWO_CHAR_PREFIX)]
        kept, report = dedupe_examples(examples)
        index = NearDuplicateIndex()
        kept_explicit, report_explicit = index.filter(examples)
        assert [decision.status for decision in report_explicit.decisions] == [
            decision.status for decision in report.decisions
        ]
        assert len(kept) == len(kept_explicit) == report_explicit.kept
        assert report.to_dict() == report_explicit.to_dict()

    def test_report_dataclass_can_be_assembled_by_hand(self):
        """恒等式是**可断言的契约**：手工装配一个报告也要满足它（防止把恒等式写进代码里）."""
        report = DedupeReport(total=4, kept=1, exact_duplicates=2, near_duplicates=1, threshold=0.7)
        assert report.exact_duplicates + report.near_duplicates + report.kept == report.total
        assert report.duplicates == 3
        assert report.keep_rate == 0.25


class TestThresholdBoundary:
    """阈值边界行为：同一对样本在 0.7 判重复、在 0.95 判保留."""

    def test_two_char_prefix_pair_at_default_threshold(self):
        """估计 0.8906：默认阈值 0.7 下必须判近重复（否则近重复链路是空转的）."""
        _, report = dedupe_examples(
            [make_example(PROMPT), make_example(PROMPT_WITH_TWO_CHAR_PREFIX)]
        )
        assert report.near_duplicates == 1
        assert report.kept == 1

    def test_same_pair_is_kept_at_strict_threshold(self):
        """同一对样本把阈值抬到 0.95：0.8906 < 0.95 → 保留。

        这不是"阈值越高越安全"：它说明阈值是**召回与误伤的旋钮**，
        而不是一条"正确/错误"的界线。
        """
        _, report = dedupe_examples(
            [make_example(PROMPT), make_example(PROMPT_WITH_TWO_CHAR_PREFIX)], threshold=0.95
        )
        assert report.near_duplicates == 0
        assert report.kept == 2
        assert report.decisions[1].similarity == pytest.approx(0.8906, abs=1e-4)

    @pytest.mark.parametrize(
        "threshold, expected_status, expected_kept",
        [
            (0.7, STATUS_NEAR, 1),
            (0.8, STATUS_NEAR, 1),
            (0.9, STATUS_KEPT, 2),
            (0.95, STATUS_KEPT, 2),
            (1.0, STATUS_KEPT, 2),
        ],
    )
    def test_status_flips_exactly_at_the_estimated_similarity(
        self, threshold, expected_status, expected_kept
    ):
        """阈值扫描：0.8906 这对样本的翻转点必须落在估计值上（0.9 起判保留）."""
        _, report = dedupe_examples(
            [make_example(PROMPT), make_example(PROMPT_WITH_TWO_CHAR_PREFIX)],
            threshold=threshold,
        )
        assert report.decisions[1].status == expected_status
        assert report.kept == expected_kept

    def test_exact_duplicate_survives_the_strictest_threshold(self):
        """阈值 1.0 时精确重复仍然被挡——"精确优先"不能受阈值影响."""
        _, report = dedupe_examples([make_example(PROMPT), make_example(PROMPT)], threshold=1.0)
        assert report.exact_duplicates == 1
        assert report.kept == 1


class TestKeepFirstAppearance:
    """保留首次出现：重复项出现时留下的是先到的那条对象."""

    def test_first_object_wins_identity(self):
        first = make_example(PROMPT, output="第一次出现的答案。")
        second = make_example(PROMPT, output="第二次出现的、完全不同的答案文本。")
        kept, report = dedupe_examples([first, second])
        assert report.exact_duplicates == 1
        assert len(kept) == 1
        assert kept[0] is first, "保留的必须是先到的那条对象本身"

    def test_order_inside_batch_decides_the_winner(self):
        """同一批数据换个顺序会换赢家——这是"保留首次出现"的确定性代价.

        钉住它是为了让"批次内顺序必须稳定"这条前提有据可依。
        """
        first = make_example(PROMPT, output="答案甲。")
        second = make_example(PROMPT, output="答案乙。")
        kept_forward, _ = dedupe_examples([first, second])
        kept_backward, _ = dedupe_examples([second, first])
        assert kept_forward[0].output == "答案甲。"
        assert kept_backward[0].output == "答案乙。"

    def test_near_duplicate_keeps_the_original(self):
        """近重复同理：留下的是基准提问，被丢弃的是加前缀的那条副本."""
        original = make_example(PROMPT)
        prefixed = make_example(PROMPT_WITH_TWO_CHAR_PREFIX)
        kept, report = dedupe_examples([original, prefixed])
        assert report.near_duplicates == 1
        assert kept[0] is original
        assert report.decisions[1].partner_index == 0
