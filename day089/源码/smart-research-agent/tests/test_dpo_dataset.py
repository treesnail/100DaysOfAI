"""day055 偏好数据构造测试：劣化算子、打分器、自洽体检与 TRL 落盘（全部离线）.

用 day048 的 SFT 种子样本（``data/finetune/seed_examples.jsonl``）与
由红队 payload 派生的安全偏好数据（``data/eval/safety_pairs.jsonl``），
在同一条确定性流水线上验证"偏好对从哪来、怎么保证 chosen 真的更好"。
"""

from __future__ import annotations

import pytest

from smart_research_agent.dpo import (
    DEFAULT_SELECTION,
    DEGRADATION_OPS,
    MIN_PHRASE_CHARS,
    MIN_SCORE_GAP,
    OP_DIMENSION,
    OP_REASONS,
    OVERLENGTH_PENALTY,
    SELECTION_MODES,
    TRL_COLUMNS,
    DPOError,
    build_preferences,
    candidates_for,
    dataset_report,
    degrade,
    key_phrases,
    load_seed_examples,
    pick_rejected,
    read_safety_pairs,
    read_trl_dataset,
    safety_category_table,
    score_candidate,
    to_trl_rows,
    write_trl_dataset,
)

TEXT = "RAG 的检索阶段通常先用向量相似度召回，再用交叉编码器重排。"


class TestConstants:
    def test_four_degradation_ops_in_fixed_order(self):
        assert DEGRADATION_OPS == (
            "drop_tail",
            "vague_specifics",
            "pad_verbose",
            "add_overclaim",
        )

    def test_selection_modes(self):
        assert set(SELECTION_MODES) == {"worst", "rotate"}
        assert DEFAULT_SELECTION == "rotate"

    def test_every_op_has_reason_and_dimension(self):
        for op in DEGRADATION_OPS:
            assert OP_REASONS[op] and OP_DIMENSION[op]

    def test_trl_columns_are_exactly_three(self):
        assert TRL_COLUMNS == ("prompt", "chosen", "rejected")

    def test_gap_threshold_is_positive(self):
        assert MIN_SCORE_GAP > 0
        assert OVERLENGTH_PENALTY > 0
        assert MIN_PHRASE_CHARS >= 1


class TestKeyPhrases:
    def test_short_fragments_are_filtered_out(self):
        """低于 MIN_PHRASE_CHARS 的片段多半是虚词，必须剔除。"""
        phrases = key_phrases("这是三字短语与数字 42。")
        assert phrases == ("这是三字短语与数字 42",)

    def test_empty_text_yields_no_phrases(self):
        assert key_phrases("。。。") == ()

    def test_repeated_phrases_are_deduplicated(self):
        phrases = key_phrases("向量检索很重要。向量检索很快。")
        assert len(phrases) == len(set(phrases))


class TestScoreCandidate:
    def test_gold_scores_one(self):
        score = score_candidate(TEXT, TEXT)
        assert score.op == "gold"
        assert score.coverage == pytest.approx(1.0)
        assert score.length_ratio == pytest.approx(1.0)
        assert score.penalty == 0.0
        assert score.score == pytest.approx(1.0)

    def test_empty_candidate_rejected_not_scored_zero(self):
        """空回答的打分**无定义**，不是 0 分——静默给 0 会让"空答案"混进偏好数据。"""
        with pytest.raises(DPOError, match="候选回答不能为空"):
            score_candidate(TEXT, "")

    def test_empty_gold_rejected(self):
        with pytest.raises(DPOError, match="金标准答案不能为空"):
            score_candidate("   ", TEXT)

    def test_overlength_is_penalised(self):
        """两个追加类算子覆盖度与金标准相同，长度惩罚是它们唯一的区分度来源。"""
        padded = TEXT + "总之" * 60
        score = score_candidate(TEXT, padded)
        assert score.length_ratio > 1.0
        assert score.penalty > 0
        assert score.score < 1.0


class TestDegrade:
    def test_single_sentence_drop_tail_returns_none(self):
        """只有一句时"删掉最后一句"会把答案删空，因此算子判定为不适用。"""
        assert degrade("只有一句话。", "drop_tail") is None

    def test_multi_sentence_drop_tail_shorter(self):
        degraded = degrade("第一句内容。第二句内容。第三句内容。", "drop_tail")
        assert degraded is not None and len(degraded) < len("第一句内容。第二句内容。第三句内容。")

    def test_pad_verbose_is_longer(self):
        degraded = degrade("短句。", "pad_verbose")
        assert degraded is not None and len(degraded) > len("短句。")

    def test_unknown_op_rejected(self):
        with pytest.raises(DPOError):
            degrade(TEXT, "make_it_worse")

    def test_degradation_is_deterministic(self):
        assert degrade(TEXT, "pad_verbose") == degrade(TEXT, "pad_verbose")


class TestCandidates:
    def test_gold_is_always_first(self):
        example = load_seed_examples()[0]
        candidates = candidates_for(example)
        assert candidates[0].op == "gold"

    def test_candidates_are_sorted_by_score_descending(self):
        example = load_seed_examples()[0]
        scores = [item.score for item in candidates_for(example)]
        assert scores == sorted(scores, reverse=True)

    def test_applicable_ops_subset_of_all_ops(self):
        example = load_seed_examples()[0]
        ops = {item.op for item in candidates_for(example)} - {"gold"}
        assert ops <= set(DEGRADATION_OPS)


class TestPickRejected:
    def test_rotate_is_index_driven_and_reproducible(self):
        example = load_seed_examples()[0]
        pool = candidates_for(example)
        first = pick_rejected(pool, index=0)
        again = pick_rejected(pool, index=0)
        assert first is not None and first == again

    def test_worst_returns_lowest_scored(self):
        example = load_seed_examples()[0]
        pool = candidates_for(example)
        picked = pick_rejected(pool, index=0, selection="worst")
        degraded = [item for item in pool if item.op != "gold"]
        assert picked is not None and picked.score == min(item.score for item in degraded)

    def test_unknown_selection_rejected(self):
        pool = candidates_for(load_seed_examples()[0])
        with pytest.raises(DPOError, match="未知的挑选方式"):
            pick_rejected(pool, index=0, selection="random")


class TestBuildPreferences:
    def test_builds_pairs_from_real_seed_data(self):
        pairs, report = build_preferences(load_seed_examples())
        assert report.total_examples == 18
        assert report.cleaned_examples == 16
        assert report.built == len(pairs) == 16

    def test_report_counts_every_drop_reason(self):
        _, report = build_preferences(load_seed_examples())
        assert sum(report.by_op.values()) == report.built
        assert set(report.by_op) == set(DEGRADATION_OPS)

    def test_rotate_covers_most_operators(self):
        """覆盖度比单条样本的极端性更重要：轮转后多数算子都有样本。"""
        _, report = build_preferences(load_seed_examples())
        assert sum(1 for value in report.by_op.values() if value > 0) >= 3

    def test_every_pair_has_distinct_chosen_and_rejected(self):
        pairs, _ = build_preferences(load_seed_examples())
        for pair in pairs:
            assert pair.chosen != pair.rejected
            assert pair.dimension

    def test_skip_rate_matches_report(self):
        _, report = build_preferences(load_seed_examples())
        assert report.skip_rate == pytest.approx(1 - report.built / report.total_examples)

    def test_empty_input_rejected(self):
        with pytest.raises(DPOError, match="至少需要一条 SFT 样本"):
            build_preferences([])

    def test_negative_gap_rejected(self):
        with pytest.raises(DPOError, match="min_gap 不能为负数"):
            build_preferences(load_seed_examples(), min_gap=-0.1)

    def test_huge_gap_threshold_drops_everything(self):
        """把区分度门槛抬到不可能达到的高度，报告里必须出现 insufficient_gap。"""
        pairs, report = build_preferences(load_seed_examples(), min_gap=10.0)
        assert pairs == []
        assert report.drop_reasons.get("insufficient_gap", 0) > 0

    def test_summary_line_is_human_readable(self):
        _, report = build_preferences(load_seed_examples())
        line = report.summary_line()
        assert "偏好数据构造" in line and "成对" in line


class TestTRLRoundTrip:
    def test_rows_contain_exactly_trl_columns(self):
        pairs, _ = build_preferences(load_seed_examples())
        rows = to_trl_rows(pairs)
        assert len(rows) == len(pairs)
        for row in rows:
            assert tuple(sorted(row)) == tuple(sorted(TRL_COLUMNS))

    def test_write_then_read(self, tmp_path):
        pairs, _ = build_preferences(load_seed_examples())
        path = tmp_path / "prefs.jsonl"
        write_trl_dataset(path, pairs)
        back = read_trl_dataset(path)
        assert len(back) == len(pairs)
        assert set(back[0]) == set(TRL_COLUMNS)

    def test_written_file_keeps_chosen_and_rejected(self, tmp_path):
        pairs, _ = build_preferences(load_seed_examples())
        path = tmp_path / "prefs.jsonl"
        write_trl_dataset(path, pairs)
        back = read_trl_dataset(path)
        assert back[0]["chosen"] == pairs[0].chosen
        assert back[0]["rejected"] == pairs[0].rejected


class TestSafetyPairs:
    def test_reads_redteam_derived_pairs(self):
        pairs = read_safety_pairs()
        assert len(pairs) == 16
        for pair in pairs:
            assert pair.chosen != pair.rejected

    def test_category_table_covers_four_families(self):
        table = safety_category_table(read_safety_pairs())
        assert table == {
            "prompt_injection": 5,
            "jailbreak": 5,
            "pii_leak": 5,
            "tool_abuse": 1,
        }


class TestDatasetReport:
    def test_report_has_all_summary_fields(self):
        pairs, _ = build_preferences(load_seed_examples())
        report = dataset_report(pairs)
        assert report["total"] == len(pairs)
        for key in (
            "by_op",
            "by_dimension",
            "fingerprint",
            "mean_length_ratio",
            "mean_score_gap",
            "min_score_gap",
            "rejected_longer_fraction",
        ):
            assert key in report

    def test_min_score_gap_respects_threshold(self):
        pairs, _ = build_preferences(load_seed_examples())
        report = dataset_report(pairs)
        assert report["min_score_gap"] >= MIN_SCORE_GAP

    def test_fingerprint_is_stable_for_same_data(self):
        pairs, _ = build_preferences(load_seed_examples())
        assert dataset_report(pairs)["fingerprint"] == dataset_report(pairs)["fingerprint"]
