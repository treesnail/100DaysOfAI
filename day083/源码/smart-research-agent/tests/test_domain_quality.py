"""day057 五维质量打分测试：梯形分 / 反向分 / 门槛语义 / 过滤记账（全部离线）.

本文件守的是**"分数为什么是这个数"**，也就是打分的可解释性：

1. 每个维度的分段都是**能手算复核**的（梯形分四段点、反向分两段点、
   三档结构分、手算的 3-gram 重复率）——报告里写"长度分 0.375"，
   评审必须能一眼还原成"输出 25 字，落在 8~40 的上升段"；
2. **门槛语义**必须算清楚：任意**单个**维度归零，总分都还在 0.75 以上、
   不会被 0.6 的门槛拒绝。这不是漏洞，而是与 day048 七条硬规则的分工
   ——单点致命问题由一票否决的布尔规则管，质量分只管"都合格但有优劣"。
   因此这条性质要参数化地钉住，否则"某天把权重调大"会悄悄把质量分变成门禁；
3. **实测脏样本的分数**（0.4828 / 0.2906）与**拒绝归因**（placeholder /
   repetition）是常量标定的依据，必须逐条断言；
4. 维度对照表的权重列**由代码算出**（``default_weights().as_dict()``），
   断言它就是在守"文档数字不是抄的"。

样本全部由 ``TrainingExample(...)`` 现场构造，不读真实数据文件、不用随机数。
"""

from __future__ import annotations

import pytest

from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.quality import (
    DEFAULT_QUALITY_THRESHOLD,
    DIMENSIONS,
    LENGTH_HIGH,
    LENGTH_IDEAL_HIGH,
    LENGTH_IDEAL_LOW,
    LENGTH_LOW,
    REPETITION_BEST,
    REPETITION_NGRAM,
    REPETITION_WORST,
    STRUCTURE_MARKERS,
    STRUCTURE_OPEN_ENDED,
    STRUCTURE_WITH_MARKER,
    STRUCTURE_WITH_TERMINAL,
    QualityScore,
    QualityWeights,
    band_score,
    default_weights,
    descending_score,
    dimension_table,
    extract_signals,
    filter_by_quality,
    has_structure_marker,
    has_terminal_punctuation,
    ngram_repetition,
    quality_scores,
    render_dimension_table,
    score_example,
    structure_score,
)
from smart_research_agent.evaluation.perf_baseline import percentile
from smart_research_agent.finetune.schema import TrainingExample

#: 梯形分的四段点（用模块常量构造，保证与实现同源）
BAND = {
    "low": LENGTH_LOW,
    "ideal_low": LENGTH_IDEAL_LOW,
    "ideal_high": LENGTH_IDEAL_HIGH,
    "high": LENGTH_HIGH,
}

#: 一条"样样合格"的样本：48 字、3-gram 重复率 0、结构化收尾、来源与许可证齐全
CLEAN_ANSWER = "Final Answer: 评估 RAG 检索质量可以同时看命中率、召回率与 MRR 三个指标。"
CLEAN = TrainingExample(
    instruction="如何评估 RAG 的检索质量？",
    output=CLEAN_ANSWER,
    source="docs",
    license="MIT",
)

#: 实测脏样本一：占位符 + 输出太短（9 字），source 有值但无 license
DIRTY_PLACEHOLDER = TrainingExample(instruction="问题？", output="TODO：待补充。", source="probe")

#: 实测脏样本二：车轱辘话 + 无句末标点（10 字，8 个 3-gram 里只有 2 个唯一）
DIRTY_REPETITION = TrainingExample(
    instruction="问题？", output="好的好的好的好的好的", source="probe"
)

#: "只有占位符一维不合格"的输出：结构化收尾、长度落在舒适区、无自我重复
PLACEHOLDER_ONLY_ANSWER = (
    "Final Answer: 关于 TODO 待补充的评估指标，可以先看命中率与召回率两个基础量。"
)


class TestBandScore:
    """梯形分：四段点合法性与每一段的手算值."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, 0.0),
            (LENGTH_LOW, 0.0),  # value <= low → 0
            (LENGTH_LOW + 1, pytest.approx(1 / 32, abs=1e-6)),  # 9 字 → (9-8)/(40-8)
            (24, pytest.approx(0.5)),  # 手算：(24-8)/(40-8)
            (LENGTH_IDEAL_LOW, 1.0),
            (400, 1.0),
            (LENGTH_IDEAL_HIGH, 1.0),
            (2400, pytest.approx(0.5)),  # 手算：(4000-2400)/(4000-800)
            (LENGTH_HIGH, 0.0),  # value >= high → 0
            (999999, 0.0),
        ],
    )
    def test_each_segment_is_hand_checkable(self, value, expected):
        assert band_score(value, **BAND) == expected

    def test_rising_segment_formula(self):
        """上升段：``(value - low) / (ideal_low - low)``——把公式与实现对照着钉住."""
        for value in (10, 16, 25, 39):
            expected = (value - LENGTH_LOW) / (LENGTH_IDEAL_LOW - LENGTH_LOW)
            assert band_score(value, **BAND) == pytest.approx(expected)

    def test_falling_segment_formula(self):
        """下降段：``(high - value) / (high - ideal_high)``."""
        for value in (1000, 2000, 3999):
            expected = (LENGTH_HIGH - value) / (LENGTH_HIGH - LENGTH_IDEAL_HIGH)
            assert band_score(value, **BAND) == pytest.approx(expected)

    def test_score_is_bounded(self):
        """任何输入都落在 [0, 1]：总分才不会超过 1（门槛才有效）."""
        for value in (0, 8, 40, 800, 4000, 10**9):
            assert 0.0 <= band_score(value, **BAND) <= 1.0

    @pytest.mark.parametrize(
        "low, ideal_low, ideal_high, high",
        [
            (40, 8, 800, 4000),  # low > ideal_low
            (8, 800, 40, 4000),  # ideal_low > ideal_high
            (8, 40, 5000, 4000),  # ideal_high > high
        ],
    )
    def test_degenerate_four_points_rejected(self, low, ideal_low, ideal_high, high):
        """退化区间会让函数静默返回错误分数（甚至除零），必须在入口拒绝."""
        with pytest.raises(DomainDataError, match="梯形分的四段点必须满足"):
            band_score(100, low=low, ideal_low=ideal_low, ideal_high=ideal_high, high=high)

    def test_collapsed_band_does_not_divide_by_zero(self):
        """四段点全部相等是合法的退化平台：任何值都应落到边界，而不是 ZeroDivisionError."""
        assert band_score(5, low=10, ideal_low=10, ideal_high=10, high=10) == 0.0


class TestDescendingScore:
    """反向线性分（重复率、噪声比例这类"越小越好"的指标用它）."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, 1.0),
            (REPETITION_BEST, 1.0),  # <= best → 满分
            (0.2, pytest.approx(0.6)),  # 手算：(0.35-0.2)/(0.35-0.1)
            (0.225, pytest.approx(0.5)),
            (REPETITION_WORST, 0.0),  # >= worst → 0
            (0.9, 0.0),
        ],
    )
    def test_segments_are_hand_checkable(self, value, expected):
        """线性段必须用容差比较：``0.225`` 的浮点结果是 ``0.49999999999999994``.

        这不是实现的问题——浮点减法/除法本来就不可结合。但报告里的数字与
        断言都必须留容差，否则会得到"同一公式两次算出不同结论"的假故障。
        """
        assert descending_score(value, best=REPETITION_BEST, worst=REPETITION_WORST) == expected

    def test_linear_segment_formula(self):
        for value in (0.11, 0.15, 0.3, 0.34):
            expected = (REPETITION_WORST - value) / (REPETITION_WORST - REPETITION_BEST)
            assert descending_score(
                value, best=REPETITION_BEST, worst=REPETITION_WORST
            ) == pytest.approx(expected)

    def test_best_greater_than_worst_rejected(self):
        """方向反了会得到负数分数（"越差越高分"），必须在入口拒绝."""
        with pytest.raises(DomainDataError, match="best 必须不大于 worst"):
            descending_score(0.5, best=0.9, worst=0.1)

    def test_collapsed_interval_only_has_two_outcomes(self):
        """``best == worst`` 时最优平台收缩成一个点：不是满分就是 0 分."""
        assert descending_score(0.3, best=0.3, worst=0.3) == 1.0
        assert descending_score(0.4, best=0.3, worst=0.3) == 0.0


class TestNgramRepetition:
    """3-gram 重复率 = ``1 - 唯一 n-gram 数 / n-gram 总数``."""

    def test_hand_calculated_degenerate_sentence(self):
        """``"好的好的好的好的好的"``：10 字 → 8 个 3-gram，其中只有 2 个唯一 → 0.75.

        （"好的好"与"的好好"交替出现；8 个 / 2 个都能数出来，所以这个数字
        不是"测出来的"而是"算出来的"。）
        """
        assert ngram_repetition("好的好的好的好的好的") == pytest.approx(0.75)

    def test_no_repetition_scores_zero(self):
        """每个 3-gram 都只出现一次 → 重复率 0.0（不是"扣一点分"）."""
        assert ngram_repetition("检索质量评估指标命中率") == 0.0

    @pytest.mark.parametrize("text", ["好的", "好", ""])
    def test_text_shorter_than_window_scores_zero(self, text):
        """短于 n 时记 0.0 而不是 1.0：记 1.0 会让所有极短样本被判死."""
        assert ngram_repetition(text) == 0.0

    def test_default_window_matches_fingerprint_shingle_k(self):
        """同一个 k 让"两条文本像不像"与"这段话自己重复了几遍"可以互相印证."""
        assert REPETITION_NGRAM == 3

    @pytest.mark.parametrize("n", [0, -1])
    def test_window_below_one_rejected(self, n):
        with pytest.raises(DomainDataError, match="n-gram 窗口必须 >= 1"):
            ngram_repetition("好的好的", n=n)

    def test_case_and_width_are_normalized(self):
        """归一化与指纹共用：全角/大小写差异不该算出两个重复率."""
        assert ngram_repetition("ＲＡＧＲＡＧ") == ngram_repetition("RAGRAG")


class TestStructureScore:
    """结构完整性三档：1.0 / 0.75 / 0.25（最低档不清零，交给总分裁决）."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Thought: 先检索。\nFinal Answer: 先看命中率。", STRUCTURE_WITH_MARKER),
            ("结论：先看命中率。", STRUCTURE_WITH_MARKER),
            ("总结：先看命中率", STRUCTURE_WITH_MARKER),
            ("答案是这样。", STRUCTURE_WITH_TERMINAL),
            ("答案是这样！", STRUCTURE_WITH_TERMINAL),
            ("答案是这样。   ", STRUCTURE_WITH_TERMINAL),  # 去尾部空白后判最后一个字符
            ("答案没有标点", STRUCTURE_OPEN_ENDED),
            ("", STRUCTURE_OPEN_ENDED),
        ],
    )
    def test_three_tiers(self, text, expected):
        assert structure_score(text) == expected

    def test_tier_constants_are_pinned(self):
        assert STRUCTURE_WITH_MARKER == 1.0
        assert STRUCTURE_WITH_TERMINAL == 0.75
        assert STRUCTURE_OPEN_ENDED == 0.25
        assert "Final Answer:" in STRUCTURE_MARKERS

    def test_helper_predicates(self):
        assert has_structure_marker("结论：x") is True
        assert has_structure_marker("x") is False
        assert has_terminal_punctuation("答案。）") is True
        assert has_terminal_punctuation("答案") is False
        assert has_terminal_punctuation("   ") is False


class TestQualityWeights:
    """权重：非负且和恰为 1.0（构造期校验，不是运行时才发现）."""

    def test_default_weights_sum_to_one(self):
        weights = default_weights()
        assert weights.total() == pytest.approx(1.0)
        assert weights.as_dict() == {
            "length": 0.25,
            "repetition": 0.25,
            "structure": 0.2,
            "traceability": 0.15,
            "placeholder": 0.15,
        }

    def test_as_dict_follows_dimension_order(self):
        """键顺序与 ``DIMENSIONS`` 同源：报告、对照表、代码三处不许各说一套."""
        assert tuple(default_weights().as_dict()) == DIMENSIONS

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"length": 0.5},  # 和 = 1.25
            {"length": 0.1, "repetition": 0.1, "structure": 0.1},  # 和 = 0.45
            {"length": 0.3, "repetition": 0.25, "structure": 0.2},  # 和 = 1.05
        ],
    )
    def test_sum_must_be_one(self, kwargs):
        """和不为 1.0 会让不同批次的总分不在同一尺度上，必须在构造期拦下."""
        with pytest.raises(DomainDataError, match="五个维度权重之和必须为 1.0"):
            QualityWeights(**kwargs)

    def test_negative_weight_rejected(self):
        with pytest.raises(DomainDataError, match="权重不能为负"):
            QualityWeights(length=-0.1, repetition=0.6, structure=0.2)

    def test_valid_custom_weights_accepted(self):
        weights = QualityWeights(
            length=0.4, repetition=0.2, structure=0.2, traceability=0.1, placeholder=0.1
        )
        assert weights.as_dict()["length"] == 0.4
        assert weights.total() == pytest.approx(1.0)

    def test_error_is_value_error_subclass(self):
        with pytest.raises(ValueError):
            QualityWeights(length=0.5)


class TestScoreExample:
    """单条打分：维度集合、加权和与门槛判定."""

    def test_dimensions_cover_every_dimension(self):
        score = score_example(CLEAN)
        assert tuple(score.dimensions) == DIMENSIONS
        assert set(score.dimensions) == set(DIMENSIONS)

    def test_total_is_weighted_sum_of_dimensions(self):
        score = score_example(CLEAN)
        weights = default_weights()
        expected = sum(score.dimensions[name] * getattr(weights, name) for name in DIMENSIONS)
        assert score.total == pytest.approx(expected)

    def test_clean_sample_scores_perfectly(self):
        """48 字、无重复、有结构化收尾、来源与许可证齐全 → 五维全 1.0."""
        score = score_example(CLEAN)
        assert score.dimensions == {name: 1.0 for name in DIMENSIONS}
        assert score.total == pytest.approx(1.0)
        assert score.passed is True

    def test_threshold_travels_with_the_score(self):
        """分数自带判定依据：门槛改了以后，老报告里的 0.62 才读得懂."""
        assert score_example(CLEAN).threshold == DEFAULT_QUALITY_THRESHOLD
        assert score_example(CLEAN, threshold=0.99).threshold == 0.99

    def test_passed_uses_greater_or_equal(self):
        """门槛值本身算通过（``>=``）：0.6 的样本不能被判拒."""
        score = QualityScore(
            dimensions={name: 0.6 for name in DIMENSIONS}, total=0.6, threshold=0.6
        )
        assert score.passed is True

    def test_weakest_breaks_ties_by_dimension_order(self):
        """并列最低时按 ``DIMENSIONS`` 顺序取第一个，拒绝归因才可复现."""
        tie = QualityScore(dimensions={name: 0.4 for name in DIMENSIONS}, total=0.4, threshold=0.6)
        assert tie.weakest() == "length"

    def test_to_dict_rounds_and_carries_verdict(self):
        payload = score_example(DIRTY_PLACEHOLDER).to_dict()
        assert payload["passed"] is False
        assert payload["weakest"] == "placeholder"
        assert payload["total"] == pytest.approx(0.4828, abs=1e-4)
        assert len(payload["dimensions"]) == len(DIMENSIONS)

    def test_signals_are_raw_facts(self):
        """信号是事实、分数是判断：原始量必须直接可见（评审据此定位问题）."""
        signals = extract_signals(DIRTY_PLACEHOLDER)
        assert signals.output_chars == len(DIRTY_PLACEHOLDER.output) == 9
        assert signals.instruction_chars == 3
        assert signals.has_source is True
        assert signals.has_license is False
        assert signals.has_placeholder is True
        assert signals.structure == STRUCTURE_WITH_TERMINAL

    def test_signals_to_dict_is_json_ready(self):
        payload = extract_signals(CLEAN).to_dict()
        assert set(payload) == {
            "output_chars",
            "instruction_chars",
            "repetition",
            "structure",
            "has_source",
            "has_license",
            "has_placeholder",
        }
        assert payload["has_placeholder"] is False


class TestMeasuredDirtySamples:
    """实测脏样本：分数与拒绝归因必须逐条对得上（常量标定的依据）."""

    def test_placeholder_sample_scores_measured_total(self):
        """``"TODO：待补充。"`` + ``source="probe"`` + 无 license → 0.4828，被拒.

        构成：length 0.03125（9 字，落在 8~40 上升段）+ repetition 1.0
        + structure 0.75 + traceability 0.5 + placeholder 0.0
        → 0.25·0.03125 + 0.25 + 0.2·0.75 + 0.15·0.5 = 0.4828125。
        """
        score = score_example(DIRTY_PLACEHOLDER)
        assert score.dimensions["placeholder"] == 0.0
        assert score.dimensions["length"] == pytest.approx(0.03125, abs=1e-6)
        assert score.total == pytest.approx(0.4828, abs=1e-4)
        assert score.passed is False
        assert score.weakest() == "placeholder"

    def test_repetition_sample_scores_measured_total(self):
        """``"好的好的好的好的好的"`` → 0.2906，被拒，最低维 repetition.

        构成：length 0.0625（10 字）+ repetition 0.0（0.75 > 0.35）+ structure 0.25
        + traceability 0.5 + placeholder 1.0 = 0.290625。
        """
        score = score_example(DIRTY_REPETITION)
        assert score.dimensions["repetition"] == 0.0
        assert score.dimensions["structure"] == STRUCTURE_OPEN_ENDED
        assert score.total == pytest.approx(0.2906, abs=1e-4)
        assert score.passed is False
        assert score.weakest() == "repetition"


class TestThresholdSemantics:
    """门槛语义：单维度归零不会被拒（0.6 的门槛等价于"加权损失 > 0.4 才拒"）."""

    @pytest.mark.parametrize("dimension", DIMENSIONS)
    def test_single_zero_dimension_stays_above_the_floor(self, dimension):
        """把五个维度中任意**单个**维度归零：总分 >= 0.75，不会被 0.6 的门槛拒绝.

        权重最大的一维是 0.25，因此最坏情况是 ``1 - 0.25 = 0.75``。
        这条性质是刻意的分工（单点致命问题交给 day048 的硬规则一票否决），
        不是"门槛太松"——一旦它被破坏，质量分就变成了第二道门禁。
        """
        weights = default_weights().as_dict()
        dimensions = {name: (0.0 if name == dimension else 1.0) for name in DIMENSIONS}
        total = sum(dimensions[name] * weights[name] for name in DIMENSIONS)
        score = QualityScore(
            dimensions=dimensions, total=total, threshold=DEFAULT_QUALITY_THRESHOLD
        )
        assert total == pytest.approx(1.0 - weights[dimension])
        assert total >= 0.75
        assert score.passed is True
        assert score.weakest() == dimension

    def test_missing_metadata_is_not_fatal(self):
        """真实样本验证：来源/许可证全空（traceability = 0）→ 0.85，仍然通过."""
        score = score_example(TrainingExample(instruction=CLEAN.instruction, output=CLEAN_ANSWER))
        assert score.dimensions["traceability"] == 0.0
        assert score.total == pytest.approx(0.85, abs=1e-6)
        assert score.passed is True
        assert score.weakest() == "traceability"

    def test_placeholder_alone_is_not_fatal_for_the_scorer(self):
        """真实样本验证：只有占位符一维归零 → 0.85，仍然通过.

        这正是"打分不能冒充门禁"的实测形态：占位符样本必须由清洗器
        （``placeholder_output`` 硬规则）拦下，而不是指望质量分扣分后
        侥幸不过线。
        """
        score = score_example(
            TrainingExample(
                instruction=CLEAN.instruction,
                output=PLACEHOLDER_ONLY_ANSWER,
                source="docs",
                license="MIT",
            )
        )
        assert score.dimensions["placeholder"] == 0.0
        assert score.total == pytest.approx(0.85, abs=1e-6)
        assert score.passed is True

    def test_two_dimensions_down_is_where_it_gets_rejected(self):
        """单维不致命，但"多维度同时不合格"会被拒——门槛真正拦的是这种样本."""
        assert score_example(DIRTY_PLACEHOLDER).passed is False
        assert score_example(DIRTY_REPETITION).passed is False

    def test_default_threshold_is_six_tenths(self):
        assert DEFAULT_QUALITY_THRESHOLD == 0.6


class TestDimensionTable:
    """维度对照表：数字由代码算出，不是抄的."""

    def test_weight_column_equals_default_weights(self):
        """表里的权重必须**逐个等于** ``default_weights().as_dict()``.

        这条断言是"文档由代码生成"的守卫：权重改了而文档没改，这里立刻变红。
        """
        table = dimension_table()
        assert [row["weight"] for row in table] == list(default_weights().as_dict().values())

    def test_row_order_follows_dimensions(self):
        assert [row["dimension"] for row in dimension_table()] == list(DIMENSIONS)

    def test_every_row_is_explained(self):
        """每一行都要有含义、计算方式与"挡住什么故障"——缺一列就等于没解释."""
        for row in dimension_table():
            assert row["meaning"] and row["computation"] and row["guards_against"]

    def test_custom_weights_flow_into_the_table(self):
        custom = QualityWeights(
            length=0.4, repetition=0.2, structure=0.2, traceability=0.1, placeholder=0.1
        )
        assert [row["weight"] for row in dimension_table(custom)] == [0.4, 0.2, 0.2, 0.1, 0.1]

    def test_rendered_markdown_mentions_every_dimension(self):
        rendered = render_dimension_table()
        for name in DIMENSIONS:
            assert f"`{name}`" in rendered
        assert "| 维度 | 含义 | 计算 | 挡住的故障 | 缺省权重 |" in rendered

    def test_rendered_numbers_come_from_constants(self):
        """渲染文本里的四段点/三档取值必须能在模块常量里找到（防止手写漂移）."""
        rendered = render_dimension_table()
        assert f"<{LENGTH_LOW} / {LENGTH_LOW}~{LENGTH_IDEAL_LOW}" in rendered
        assert f"<={REPETITION_BEST}" in rendered
        assert f"{STRUCTURE_OPEN_ENDED}" in rendered


class TestFilterByQuality:
    """批量过滤：等长同序的分数、拒绝归因与分布."""

    def test_scores_are_same_length_and_order_as_input(self):
        """``scores`` 与入参**等长同序**（含被拒项）：
        "第 i 条为什么被拒"必须当场可答，被拒样本的原因不许消失在报告之外。
        """
        examples = [DIRTY_PLACEHOLDER, CLEAN, DIRTY_REPETITION]
        result = filter_by_quality(examples)
        assert result.total == len(examples)
        assert len(result.scores) == len(examples)
        assert [score.total for score in result.scores] == [
            score_example(example).total for example in examples
        ]
        assert [round(score.total, 4) for score in result.scores] == [
            pytest.approx(0.4828, abs=1e-4),
            pytest.approx(1.0, abs=1e-6),
            pytest.approx(0.2906, abs=1e-4),
        ]

    def test_kept_and_rejected_indices_partition_the_batch(self):
        examples = [DIRTY_PLACEHOLDER, CLEAN, DIRTY_REPETITION]
        result = filter_by_quality(examples)
        assert result.kept_indices == [1]
        assert result.rejected_indices == [0, 2]
        assert result.rejected == 2
        assert len(result.kept) == 1
        assert result.kept[0] is examples[1]

    def test_rejection_is_attributed_to_the_weakest_dimension(self):
        """每一次丢弃都要有名字，且名字取自**最低分维度**（不是"第一个不达标的"）."""
        result = filter_by_quality([DIRTY_PLACEHOLDER, CLEAN, DIRTY_REPETITION])
        assert result.rejection_by_dimension() == {"placeholder": 1, "repetition": 1}
        assert sum(result.rejection_by_dimension().values()) == result.rejected

    def test_rejection_attribution_is_reproducible(self):
        """同一份数据跑两次，归因必须逐键相同（并列取 DIMENSIONS 顺序的收益）."""
        examples = [DIRTY_PLACEHOLDER, DIRTY_REPETITION]
        first = filter_by_quality(examples).rejection_by_dimension()
        second = filter_by_quality(examples).rejection_by_dimension()
        assert first == second

    def test_distribution_uses_nearest_rank_percentile(self):
        """分布复用 day046 的最近秩法（**不插值**）：每个分位数都必须与
        ``evaluation.perf_baseline.percentile`` 逐个对得上，即"确实是一条真实样本的分数".
        """
        examples = [DIRTY_PLACEHOLDER, CLEAN, DIRTY_REPETITION]
        result = filter_by_quality(examples)
        totals = [score.total for score in result.scores]
        distribution = result.distribution()
        assert distribution["min"] == round(min(totals), 4)
        assert distribution["max"] == round(max(totals), 4)
        assert distribution["mean"] == round(sum(totals) / len(totals), 4)
        for key, q in (("p25", 25), ("p50", 50), ("p75", 75)):
            assert distribution[key] == round(percentile(totals, q), 4)
        assert distribution["p25"] == distribution["min"]  # 3 条样本的 P25 落在最小值上

    def test_empty_input_is_not_an_error(self):
        """空批次不抛错，保留率按 0.0 计（与 day048 同一约定）."""
        result = filter_by_quality([])
        assert result.total == 0
        assert result.keep_rate == 0.0
        assert result.kept_indices == []
        assert result.rejected_indices == []
        assert result.rejection_by_dimension() == {}
        assert result.distribution() == {
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }

    def test_keep_rate_matches_indices(self):
        result = filter_by_quality([DIRTY_PLACEHOLDER, CLEAN])
        assert result.keep_rate == pytest.approx(0.5)

    def test_custom_weights_and_threshold_are_honoured(self):
        """权重与门槛都要真的生效：把门槛抬到 0.99 后只有满分样本能过."""
        result = filter_by_quality(
            [CLEAN, DIRTY_PLACEHOLDER], threshold=0.99, weights=default_weights()
        )
        assert result.kept_indices == [0]
        assert result.threshold == 0.99
        assert result.weights.as_dict() == default_weights().as_dict()

    def test_to_dict_carries_the_whole_accounting(self):
        result = filter_by_quality([DIRTY_PLACEHOLDER, CLEAN])
        payload = result.to_dict()
        assert payload["total"] == 2
        assert payload["kept"] == 1
        assert payload["rejected"] == 1
        assert payload["keep_rate"] == pytest.approx(0.5)
        assert payload["threshold"] == DEFAULT_QUALITY_THRESHOLD
        assert payload["weights"] == default_weights().as_dict()
        assert payload["rejection_by_dimension"] == {"placeholder": 1}
        assert "distribution" in payload

    def test_summary_line_is_human_readable(self):
        line = filter_by_quality([DIRTY_PLACEHOLDER, CLEAN]).summary_line()
        assert "质量过滤：2 条进 / 1 条留" in line
        assert "placeholder" in line

    def test_quality_scores_only_returns_totals(self):
        """薄接口：配比削减只需要排序键，不应顺手把五维明细也算一遍暴露出去."""
        examples = [DIRTY_PLACEHOLDER, CLEAN, DIRTY_REPETITION]
        totals = quality_scores(examples)
        assert totals == [score_example(example).total for example in examples]
        assert len(totals) == len(examples)
