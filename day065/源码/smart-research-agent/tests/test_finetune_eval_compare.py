"""配对比较与回归门禁测试（day053）：``finetune_eval/compare.py`` 的逐项核对.

这个文件只测一件事：**"微调后变好了"这句话是怎么被算出来的、又是在什么
条件下被拒绝的**。三块内容各有它自己的失败模式：

- ``binomial_two_sided_p_value`` / ``mcnemar_exact``：翻转型检验的**手算值**
  必须逐位对上。这些数字全部可以在纸质上复核（``2·Σ C(n,k)/2^n`` 与
  ``(|b-c|-1)²/(b+c)``），所以断言写的是精确值而不是区间——**一个算错的
  p 值不会让任何东西崩掉，只会让报告里的"显著"变成一句错话**。
- ``percentile`` / ``paired_bootstrap_delta``：区间的可复现性是硬要求。
  同一个种子两次调用必须**逐位相同**（``==`` 而不是 ``approx``），
  否则"这份报告里的区间"就不是一个可以引用的数字。
- ``compare_runs``：门禁的**失败路径**必须被真实构造出来。本文件里那条
  "总体合格率上升、但 refusal 桶掉了 100 个百分点"的两臂是关键用例——
  它正是 ``max_regression`` 存在的理由（缺省 0.0 = 一处都不许掉）。

构造两臂时直接往 ``outcomes`` 里塞 ``ItemOutcome``：这些用例要控制的是
"哪一条过、哪一条不过"，而不是让脚本化对照臂去决定（那是
``test_finetune_eval_report_pipeline.py`` 的事）。
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from smart_research_agent.finetune_eval import (
    COMPONENT_NAMES,
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES,
    RATE_TOLERANCE,
    BucketDelta,
    ComparisonReport,
    EvalRun,
    FinetuneEvalError,
    ItemOutcome,
    MetricBreakdown,
    binomial_two_sided_p_value,
    compare_runs,
    mcnemar_exact,
    paired_bootstrap_delta,
    percentile,
)

#: 构造两臂时用的"合格条目的总分"，与"不合格条目的总分"分开，
#: 让 ``score_delta`` 与 ``pass_rate_delta`` 不是同一件事的两种写法。
PASS_SCORE = 0.9
FAIL_SCORE = 0.2


def make_outcome(
    item_id: str,
    *,
    bucket: str = "citation",
    difficulty: str = "normal",
    passed: bool = True,
    score: float | None = None,
) -> ItemOutcome:
    """构造一条 ``ItemOutcome``（六个分量取同一个值，总分单独给）.

    ``metric`` 必须是 ``MetricBreakdown``：``ItemOutcome.to_dict()`` 会按
    ``COMPONENT_NAMES`` 逐个取值，所以六个分量一个都不能少。
    """
    total = (PASS_SCORE if passed else FAIL_SCORE) if score is None else score
    return ItemOutcome(
        item_id=item_id,
        bucket=bucket,
        difficulty=difficulty,
        metric=MetricBreakdown(
            components={name: total for name in COMPONENT_NAMES}, total=total
        ),
        passed=passed,
    )


def make_run(name: str, rows: list[tuple[str, str, str, bool]]) -> EvalRun:
    """按 ``(id, 桶, 难度, 是否合格)`` 行构造一次运行."""
    return EvalRun(
        name=name,
        outcomes=[
            make_outcome(item_id, bucket=bucket, difficulty=difficulty, passed=passed)
            for item_id, bucket, difficulty, passed in rows
        ],
    )


def make_regressing_pair() -> tuple[EvalRun, EvalRun]:
    """构造"总体上升、但 refusal 桶全程回退"的两臂（门禁必须拦住的那份数据）.

    逐桶算术：

    ============  ==========  ==========  ==============
    桶            前合格率     后合格率     差值
    ============  ==========  ==========  ==============
    citation      0/3 = 0%    3/3 = 100%  +100%
    refusal       2/2 = 100%  0/2 = 0%    -100%
    **总体**      2/5 = 40%   3/5 = 60%   +20%
    ============  ==========  ==========  ==============

    这正是"总体涨了、某一类掉得厉害"的典型形态：只看总分，这份微调是
    成功的；按桶设门禁，它必须被拒绝。
    """
    before = make_run(
        "before",
        [
            ("cite-01", "citation", "normal", False),
            ("cite-02", "citation", "normal", False),
            ("cite-03", "citation", "hard", False),
            ("ref-01", "refusal", "easy", True),
            ("ref-02", "refusal", "easy", True),
        ],
    )
    after = make_run(
        "after",
        [
            ("cite-01", "citation", "normal", True),
            ("cite-02", "citation", "normal", True),
            ("cite-03", "citation", "hard", True),
            ("ref-01", "refusal", "easy", False),
            ("ref-02", "refusal", "easy", False),
        ],
    )
    return before, after


class TestBinomialTwoSidedPValue:
    """``binomial_two_sided_p_value``：``p = 0.5`` 下的两尾精确二项检验."""

    def test_zero_total_returns_one(self):
        """``total = 0``（没有不一致对）→ ``1.0``：无法拒绝"两臂一样"."""
        assert binomial_two_sided_p_value(0, 0) == 1.0

    @pytest.mark.parametrize("total", [-1, -5, -100])
    def test_negative_total_raises(self, total):
        """负的 ``total`` 是非法输入，抛 ``FinetuneEvalError``."""
        with pytest.raises(FinetuneEvalError):
            binomial_two_sided_p_value(0, total)

    @pytest.mark.parametrize(
        ("successes", "total"),
        [(-1, 5), (6, 5), (3, 2), (1, 0), (100, 18)],
    )
    def test_successes_out_of_range_raises(self, successes, total):
        """``successes`` 必须落在 ``[0, total]``，越界一律抛错."""
        with pytest.raises(FinetuneEvalError):
            binomial_two_sided_p_value(successes, total)

    @pytest.mark.parametrize(
        ("successes", "total", "expected"),
        [
            (0, 1, 1.0),  # 2·C(1,0)/2¹ = 1
            (0, 3, 0.25),  # 2·1/8
            (0, 2, 0.5),  # 2·1/4
            (0, 10, 0.001953125),  # 2/1024
            (1, 10, 0.021484375),  # 2·(1+10)/1024
            (0, 5, 0.0625),  # 2/32
            (2, 6, 0.6875),  # 2·(1+6+15)/64
        ],
    )
    def test_hand_computed_values(self, successes, total, expected):
        """手算值逐位对上：``min(1, 2·Σ_{k≤min(s,n-s)} C(n,k)/2ⁿ)``."""
        assert binomial_two_sided_p_value(successes, total) == expected

    def test_two_tailed_sum_is_capped_at_one(self):
        """``n=4, k=2``：原始两尾求和是 ``2·(1+4+6)/16 = 1.375``，结果必须截到 1.0.

        这是离散分布做两尾检验的常规处理（不是修补）：不截断会得到一个
        大于 1 的"概率"，而它会被下游直接写进报告。
        """
        raw = 2 * sum(math.comb(4, k) for k in range(3)) / 2**4
        assert raw == 1.375
        assert binomial_two_sided_p_value(2, 4) == 1.0

    @pytest.mark.parametrize(("successes", "total"), [(0, 8), (1, 8), (3, 8), (4, 8)])
    def test_is_symmetric_in_the_two_tails(self, successes, total):
        """``p(k, n) == p(n-k, n)``：函数内部取 ``min(k, n-k)`` 的必然结果."""
        assert binomial_two_sided_p_value(successes, total) == binomial_two_sided_p_value(
            total - successes, total
        )

    @pytest.mark.parametrize("successes", list(range(0, 19)))
    def test_always_within_unit_interval(self, successes):
        """n=18（本课评估集规模）上所有可能的 k 都必须落在 ``[0, 1]``."""
        value = binomial_two_sided_p_value(successes, 18)
        assert 0.0 <= value <= 1.0

    def test_is_a_finetune_eval_error(self):
        """异常类型必须是 ``FinetuneEvalError``（同时是 ``ValueError`` 的子类）."""
        with pytest.raises(FinetuneEvalError) as excinfo:
            binomial_two_sided_p_value(0, -1)
        assert isinstance(excinfo.value, ValueError)
        assert "total" in str(excinfo.value)


class TestMcNemarExact:
    """``mcnemar_exact``：只看翻转两格的精确检验 + 连续性修正的卡方参考值."""

    @pytest.mark.parametrize(("before_fail", "after_pass"), [(-1, 0), (0, -1), (-3, -4)])
    def test_negative_counts_raise(self, before_fail, after_pass):
        """两个计数都不能为负."""
        with pytest.raises(FinetuneEvalError):
            mcnemar_exact(before_fail, after_pass)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_invalid_alpha_raises(self, alpha):
        """``alpha`` 必须严格落在 ``(0, 1)``——0 与 1 都要被拒绝."""
        with pytest.raises(FinetuneEvalError):
            mcnemar_exact(1, 1, alpha=alpha)

    def test_no_discordance_gives_p_one_and_chi_zero(self):
        """零不一致 → ``p_value=1.0``、``chi_square=0.0``（两臂逐条相同）."""
        result = mcnemar_exact(0, 0)
        assert result.discordant == 0
        assert result.p_value == 1.0
        assert result.chi_square == 0.0
        assert result.significant is False

    def test_zero_to_three_gives_quarter(self):
        """``(0, 3)`` → ``p = 0.25``（= 2·C(3,0)/2³），卡方 = 4/3."""
        result = mcnemar_exact(0, 3)
        assert result.before_pass_after_fail == 0
        assert result.before_fail_after_pass == 3
        assert result.discordant == 3
        assert result.p_value == 0.25
        assert result.chi_square == pytest.approx(1.3333333333333333)

    @pytest.mark.parametrize(
        ("before_pass_after_fail", "before_fail_after_pass", "expected"),
        [
            (0, 0, 0.0),  # 无不一致：卡方定义为 0，不做除法
            (0, 3, 4 / 3),  # (3-1)²/3
            (1, 3, 0.25),  # (2-1)²/4
            (3, 1, 0.25),  # 对称
            (2, 2, 0.0),  # 差值 1 → max(0, 1-1) = 0
            (0, 1, 0.0),  # 同上：差 1 被连续性修正吃掉了
            (5, 1, 1.5),  # (4-1)²/6
        ],
    )
    def test_chi_square_continuity_correction(
        self, before_pass_after_fail, before_fail_after_pass, expected
    ):
        """卡方 = ``max(0, |b-c|-1)²/(b+c)``：连续性修正后的手算值."""
        result = mcnemar_exact(before_pass_after_fail, before_fail_after_pass)
        assert result.chi_square == pytest.approx(expected)

    @pytest.mark.parametrize(("before_pass_after_fail", "expected_p"), [(5, 0.0625), (6, 0.03125)])
    def test_p_value_follows_fail_to_pass_count(self, before_pass_after_fail, expected_p):
        """p 值只看"不过→过"的条数（此处与"过→不过"对称，取 min 后一致）."""
        result = mcnemar_exact(before_pass_after_fail, 0)
        assert result.p_value == expected_p
        # 交换两格不改变 p（两尾检验对方向不敏感）
        assert mcnemar_exact(0, before_pass_after_fail).p_value == expected_p

    @pytest.mark.parametrize(
        ("before_pass_after_fail", "before_fail_after_pass", "alpha", "expected"),
        [
            (0, 3, 0.05, False),  # p = 0.25
            (0, 6, 0.05, True),  # p = 0.03125 < 0.05
            (0, 6, 0.01, False),  # 0.03125 > 0.01
            (0, 3, 0.25, False),  # 边界：p < alpha 为假
            (0, 3, 0.26, True),
        ],
    )
    def test_significant_matches_p_less_than_alpha(
        self, before_pass_after_fail, before_fail_after_pass, alpha, expected
    ):
        """``significant`` 与 ``p < alpha`` 逐例一致（含边界 ``p == alpha``）."""
        result = mcnemar_exact(
            before_pass_after_fail, before_fail_after_pass, alpha=alpha
        )
        assert result.significant is (result.p_value < result.alpha)
        assert result.significant is expected

    def test_alpha_is_recorded_verbatim(self):
        """``alpha`` 原样进结果，不会被归一化或改写."""
        result = mcnemar_exact(1, 2, alpha=0.2)
        assert result.alpha == 0.2
        assert result.alpha != DEFAULT_ALPHA

    def test_summary_line_is_printable(self):
        """``summary_line`` 含两格计数、不一致条数与 p 值，可直接进日志."""
        line = mcnemar_exact(0, 3).summary_line()
        assert "McNemar 精确检验" in line
        assert "过→不过 0 条" in line
        assert "不过→过 3 条" in line
        assert "不一致 3 条" in line
        assert "0.250000" in line

    def test_to_dict_keys_and_json_roundtrip(self):
        """``to_dict`` 六个字段齐全，且 ``significant`` 是布尔值（不是 numpy 布尔）."""
        payload = mcnemar_exact(2, 1, alpha=0.1).to_dict()
        assert set(payload) == {
            "before_pass_after_fail",
            "before_fail_after_pass",
            "discordant",
            "chi_square",
            "p_value",
            "alpha",
            "significant",
        }
        assert payload["significant"] is False
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_result_is_frozen(self):
        """结果对象是 frozen dataclass：报告里的数字不该被事后改写."""
        result = mcnemar_exact(1, 1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.p_value = 0.0  # type: ignore[misc]


class TestPercentile:
    """``percentile``：秩 = ``q·(n-1)`` 再线性插值的分位数."""

    def test_empty_sequence_raises(self):
        """空序列没有分位数可取."""
        with pytest.raises(FinetuneEvalError):
            percentile([], 0.5)

    @pytest.mark.parametrize("quantile", [-0.1, 1.1, 2.0, -1.0])
    def test_quantile_out_of_range_raises(self, quantile):
        """``quantile`` 必须落在 ``[0, 1]``."""
        with pytest.raises(FinetuneEvalError):
            percentile([1.0, 2.0], quantile)

    @pytest.mark.parametrize("quantile", [0.0, 0.25, 0.5, 0.99, 1.0])
    def test_single_element_returns_that_element(self, quantile):
        """单元素序列：任何分位都返回该元素（秩恒为 0）."""
        assert percentile([7.5], quantile) == 7.5

    @pytest.mark.parametrize(
        ("quantile", "expected"),
        [(0.0, 0.0), (0.25, 2.5), (0.5, 5.0), (0.75, 7.5), (1.0, 10.0)],
    )
    def test_linear_interpolation(self, quantile, expected):
        """``[0, 10]`` 上的线性插值：0.25 → 2.5（不是取最近秩的 0 或 10）."""
        assert percentile([0.0, 10.0], quantile) == pytest.approx(expected)

    def test_three_elements_quarter_point(self):
        """``[0, 10, 20]`` 的 0.25 分位：秩 0.5 → ``0×0.5 + 10×0.5 = 5``."""
        assert percentile([0.0, 10.0, 20.0], 0.25) == pytest.approx(5.0)

    def test_endpoints_are_min_and_max(self):
        """两个端点必须原样返回（不留浮点残差）."""
        values = [0.1, 0.3, 0.9, 1.7]
        assert percentile(values, 0.0) == values[0]
        assert percentile(values, 1.0) == values[-1]

    @pytest.mark.parametrize("quantile", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_matches_manual_rank_formula(self, quantile):
        """与手算的"秩 + 权重"公式逐位一致."""
        values = [1.0, 2.0, 4.0, 8.0, 16.0]
        position = quantile * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        expected = (
            values[lower]
            if lower == upper
            else values[lower] * (1 - (position - lower)) + values[upper] * (position - lower)
        )
        assert percentile(values, quantile) == pytest.approx(expected)

    def test_is_monotone_in_quantile(self):
        """分位数对 ``q`` 单调不减（区间 ``[低, 高]`` 才成立）."""
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        results = [percentile(values, q) for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        assert results == sorted(results)


class TestPairedBootstrapDelta:
    """``paired_bootstrap_delta``：对逐条差值重采样的配对自助法."""

    def test_length_mismatch_raises(self):
        """两臂必须逐条对应（配对信息不能被丢掉）."""
        with pytest.raises(FinetuneEvalError):
            paired_bootstrap_delta([0.0, 1.0], [0.0])

    def test_empty_input_raises(self):
        """空列表：没有可重采样的条目."""
        with pytest.raises(FinetuneEvalError):
            paired_bootstrap_delta([], [])

    @pytest.mark.parametrize("samples", [0, -1, -100])
    def test_non_positive_samples_raises(self, samples):
        """重采样次数必须为正整数."""
        with pytest.raises(FinetuneEvalError):
            paired_bootstrap_delta([0.0, 1.0], [1.0, 2.0], samples=samples)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5])
    def test_invalid_alpha_raises(self, alpha):
        """``alpha`` 必须严格落在 ``(0, 1)``."""
        with pytest.raises(FinetuneEvalError):
            paired_bootstrap_delta([0.0, 1.0], [1.0, 2.0], alpha=alpha)

    def test_same_seed_is_bitwise_reproducible(self):
        """同 seed 两次结果**逐位相同**——报告里的区间必须是可复现的数字.

        断言用 ``==`` 而不是 ``pytest.approx``：自助法的区间一旦允许"近似
        相等"，它就不再是一份可以被引用的证据。
        """
        before = [0.0, 0.3, 0.7, 1.0, 0.5]
        after = [0.2, 0.9, 0.4, 1.0, 0.8]
        first = paired_bootstrap_delta(before, after, samples=200, seed=123)
        second = paired_bootstrap_delta(before, after, samples=200, seed=123)
        assert first == second
        assert first.to_dict() == second.to_dict()
        assert first.ci_low == second.ci_low
        assert first.ci_high == second.ci_high

    def test_mean_delta_equals_arithmetic_mean_of_pairs(self):
        """``mean_delta`` = 逐条差值的算术平均（不是两臂各自的均值之差）.

        在这个测试里两种写法恰好相等；下面再用长度不等的权重破坏对称性。
        """
        before = [0.0, 1.0, 2.0]
        after = [1.0, 2.0, 3.5]
        result = paired_bootstrap_delta(before, after, samples=100, seed=1)
        deltas = [a - b for b, a in zip(before, after, strict=True)]
        assert result.mean_delta == pytest.approx(sum(deltas) / len(deltas))
        assert result.mean_delta == pytest.approx(1.1666666666666667)

    def test_all_positive_deltas_exclude_zero(self):
        """全为正的差值 → ``ci_low > 0`` 且 ``excludes_zero is True``."""
        result = paired_bootstrap_delta(
            [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], samples=200, seed=42
        )
        assert result.mean_delta == pytest.approx(1.0)
        assert result.ci_low > 0
        assert result.ci_high >= result.ci_low
        assert result.excludes_zero is True

    def test_all_negative_deltas_exclude_zero(self):
        """全为负的差值 → ``ci_high < 0`` 且 ``excludes_zero is True``."""
        result = paired_bootstrap_delta(
            [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], samples=200, seed=42
        )
        assert result.ci_high < 0
        assert result.excludes_zero is True

    def test_zero_deltas_do_not_exclude_zero(self):
        """差值为 0 → 区间 ``[0, 0]``，``excludes_zero is False``（没有方向）."""
        result = paired_bootstrap_delta([1.0, 2.0], [1.0, 2.0], samples=200, seed=42)
        assert result.mean_delta == 0.0
        assert result.ci_low == 0.0
        assert result.ci_high == 0.0
        assert result.excludes_zero is False

    def test_interval_brackets_the_mean(self):
        """均值必须落在区间内（重采样均值的分布以原样本均值为中心）."""
        before = [0.0, 1.0, 2.0, 3.0, 4.0]
        after = [0.1, 1.4, 1.8, 3.3, 4.6]
        result = paired_bootstrap_delta(before, after, samples=500, seed=7)
        assert result.ci_low <= result.mean_delta <= result.ci_high

    def test_samples_and_alpha_are_recorded(self):
        """``samples`` / ``alpha`` 原样进结果（报告要能说清这是几次重采样）."""
        result = paired_bootstrap_delta([0.0, 1.0], [1.0, 2.0], samples=100, alpha=0.2)
        assert result.samples == 100
        assert result.alpha == 0.2
        assert result.samples != DEFAULT_BOOTSTRAP_SAMPLES

    @pytest.mark.parametrize(("alpha", "level_text"), [(0.05, "95%"), (0.1, "90%"), (0.2, "80%")])
    def test_summary_line_reports_confidence_level(self, alpha, level_text):
        """``summary_line`` 里的置信水平 = ``(1 - alpha)·100``."""
        result = paired_bootstrap_delta(
            [0.0, 1.0], [1.0, 2.0], samples=50, alpha=alpha
        )
        line = result.summary_line()
        assert line.startswith("配对自助法（50 次）")
        assert level_text in line
        assert "不含 0" in line

    def test_to_dict_keys_and_json_roundtrip(self):
        """``to_dict`` 六个字段齐全且可 json.dumps."""
        payload = paired_bootstrap_delta([0.0, 1.0], [1.0, 3.0], samples=50).to_dict()
        assert set(payload) == {
            "mean_delta",
            "ci_low",
            "ci_high",
            "samples",
            "alpha",
            "excludes_zero",
        }
        assert isinstance(payload["excludes_zero"], bool)
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_single_sample_still_returns_an_interval(self):
        """``samples=1`` 也要给出合法区间（下界不高于上界），不能崩."""
        result = paired_bootstrap_delta([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], samples=1)
        assert result.ci_low <= result.ci_high
        assert result.mean_delta == pytest.approx(0.5)


class TestCompareRunsAlignment:
    """``compare_runs`` 的输入校验：不同的运行不许被拿来比较."""

    def test_length_mismatch_raises(self):
        """两臂条数不一致 → 抛错（配对比较的前提被破坏）."""
        before = make_run("before", [("a", "citation", "normal", True)])
        after = make_run(
            "after",
            [("a", "citation", "normal", True), ("b", "citation", "normal", True)],
        )
        with pytest.raises(FinetuneEvalError):
            compare_runs(before, after)

    def test_same_length_but_different_ids_raises(self):
        """条数一致但顺序/内容不一致（id 不同）→ 抛错.

        ``format_lines`` 是按顺序逐行渲染的；允许乱序会让同一份数据渲染出
        两种表格，而人眼对比表格时不会去核对每一行的 id。
        """
        before = make_run(
            "before",
            [("cite-01", "citation", "normal", True), ("cite-02", "citation", "normal", True)],
        )
        after = make_run(
            "after",
            [("cite-02", "citation", "normal", True), ("cite-01", "citation", "normal", True)],
        )
        with pytest.raises(FinetuneEvalError) as excinfo:
            compare_runs(before, after)
        assert "顺序不一致" in str(excinfo.value)

    def test_empty_run_raises(self):
        """空运行 → 抛错（"不能对比空运行"），而不是静默给出 0 差值."""
        with pytest.raises(FinetuneEvalError) as excinfo:
            compare_runs(EvalRun(name="before"), EvalRun(name="after"))
        assert "空运行" in str(excinfo.value)

    def test_empty_versus_non_empty_raises(self):
        """一臂为空、另一臂非空：先被"条数不一致"拦下."""
        before = make_run("before", [])
        after = make_run("after", [("a", "citation", "normal", True)])
        with pytest.raises(FinetuneEvalError):
            compare_runs(before, after)

    @pytest.mark.parametrize("max_regression", [-0.01, -1.0, -1e-12])
    def test_negative_max_regression_raises(self, max_regression):
        """``max_regression`` 不能为负（负数等于"必须回退"）."""
        before, after = make_regressing_pair()
        with pytest.raises(FinetuneEvalError):
            compare_runs(before, after, max_regression=max_regression)


class TestCompareRunsGate:
    """门禁与分桶比对：本文件的核心——**失败路径必须被真实构造出来**."""

    def test_bucket_regression_fails_the_gate(self):
        """总体上升 + refusal 桶回退 → ``passed is False`` 且桶名进 ``regressions``.

        缺省 ``max_regression = 0.0`` 的意思是"一处都不许掉"，所以这条断言
        就是门禁存在的全部理由。
        """
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=200)

        assert report.before_pass_rate == pytest.approx(2 / 5)
        assert report.after_pass_rate == pytest.approx(3 / 5)
        assert report.pass_rate_delta == pytest.approx(0.2)
        assert report.passed is False
        assert report.regressions == ["refusal"]

    def test_relaxing_max_regression_turns_the_same_data_green(self):
        """把 ``max_regression`` 放宽到足够大，同一份数据变成通过（门禁是可配置的）.

        两处对照一起断言：这是"门禁拦的是幅度，不是方向"的证据。
        """
        before, after = make_regressing_pair()
        strict = compare_runs(before, after, samples=200)
        relaxed = compare_runs(before, after, samples=200, max_regression=1.0)
        assert strict.passed is False
        assert relaxed.passed is True
        assert relaxed.regressions == []
        assert relaxed.max_regression == 1.0

    def test_partially_relaxed_max_regression_still_fails(self):
        """放宽到 0.5 时 refusal 掉 100% 仍然超限 → 依然不通过."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=200, max_regression=0.5)
        assert report.passed is False
        assert report.regressions == ["refusal"]

    def test_improving_arms_pass_with_no_regressions(self):
        """正常上升的两臂：``passed is True``、``regressions`` 为空."""
        before = make_run(
            "before",
            [("a", "citation", "normal", False), ("b", "tool_use", "normal", False)],
        )
        after = make_run(
            "after",
            [("a", "citation", "normal", True), ("b", "tool_use", "normal", True)],
        )
        report = compare_runs(before, after, samples=200)
        assert report.passed is True
        assert report.regressions == []
        assert report.pass_rate_delta == pytest.approx(1.0)

    def test_mcnemar_and_bootstrap_are_always_present(self):
        """``mcnemar`` / ``bootstrap`` 不为 None（报告里两行证据必须齐）."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=200)
        assert report.mcnemar is not None
        assert report.bootstrap is not None
        assert report.mcnemar.discordant == 5
        assert report.bootstrap.samples == 200
        assert report.bootstrap.alpha == DEFAULT_ALPHA

    def test_total_equals_pair_count(self):
        """``total`` 等于对齐后的对数（也就是两臂各自的条数）."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100)
        assert report.total == 5
        assert report.total == before.total == after.total

    def test_buckets_are_sorted_and_regression_flag_is_per_bucket(self):
        """桶表按名字排序；只有真正回退的那个桶被标 ``regressed``."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100)
        assert [bucket.name for bucket in report.buckets] == ["citation", "refusal"]
        flags = {bucket.name: bucket.regressed for bucket in report.buckets}
        assert flags == {"citation": False, "refusal": True}

    def test_regressed_bucket_summary_line_marks_regression(self):
        """回退桶的 ``summary_line`` 含 ``⚠回退``；未回退的桶不含."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100)
        lines = {bucket.name: bucket.summary_line() for bucket in report.buckets}
        assert "⚠回退" in lines["refusal"]
        assert "⚠回退" not in lines["citation"]
        assert "合格率 100.0% → 0.0%" in lines["refusal"]

    def test_difficulty_table_follows_the_fixed_order(self):
        """难度表顺序固定为 easy / normal / hard（与报告的表格一致）."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100)
        assert [bucket.name for bucket in report.difficulties] == ["easy", "normal", "hard"]
        assert [bucket.total for bucket in report.difficulties] == [2, 2, 1]

    def test_max_regression_tolerance_swallows_float_noise(self):
        """落差在 ``RATE_TOLERANCE`` 之内不算回退——一个总是报警的门禁会被忽略.

        构造前后合格率"名义上相等"的两臂：差值来自浮点表示，必须被判为通过。
        """
        before = make_run("before", [("a", "citation", "normal", True)])
        after = make_run("after", [("a", "citation", "normal", True)])
        report = compare_runs(before, after, samples=50)
        assert RATE_TOLERANCE > 0
        assert report.regressions == []
        assert report.passed is True

    def test_alpha_is_passed_into_both_statistics(self):
        """``alpha`` 同时进 McNemar 与自助法（两处证据用同一个显著性水平）."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100, alpha=0.2)
        assert report.mcnemar.alpha == 0.2
        assert report.bootstrap.alpha == 0.2

    def test_seed_makes_bootstrap_reproducible(self):
        """同 seed 两次比较得到**逐位相同**的自助法区间."""
        before, after = make_regressing_pair()
        first = compare_runs(before, after, samples=200, seed=11)
        second = compare_runs(before, after, samples=200, seed=11)
        assert first.bootstrap == second.bootstrap

    def test_symmetric_arms_give_zero_delta(self):
        """两臂完全相同 → 差值为 0、无回归、McNemar 的 p 值为 1.0."""
        rows = [
            ("a", "citation", "normal", True),
            ("b", "citation", "normal", False),
        ]
        report = compare_runs(make_run("x", rows), make_run("y", rows), samples=100)
        assert report.pass_rate_delta == 0.0
        assert report.score_delta == 0.0
        assert report.mcnemar.discordant == 0
        assert report.mcnemar.p_value == 1.0
        assert report.passed is True


class TestComparisonReportRendering:
    """``ComparisonReport`` / ``BucketDelta`` 的投影与渲染."""

    def test_to_dict_has_all_keys(self):
        """``to_dict`` 的键集合完整（含派生字段 ``passed`` 与嵌套的两张表）."""
        before, after = make_regressing_pair()
        payload = compare_runs(before, after, samples=100).to_dict()
        assert set(payload) == {
            "before_name",
            "after_name",
            "total",
            "before_pass_rate",
            "after_pass_rate",
            "pass_rate_delta",
            "before_mean_score",
            "after_mean_score",
            "score_delta",
            "buckets",
            "difficulties",
            "mcnemar",
            "bootstrap",
            "regressions",
            "max_regression",
            "passed",
        }
        assert payload["passed"] is False
        assert payload["regressions"] == ["refusal"]
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_format_lines_count_equals_bucket_count(self):
        """``format_lines`` 的行数等于桶数（不是条目数，也不是难度数）."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100)
        lines = report.format_lines()
        assert len(lines) == len(report.buckets) == 2
        assert all(line.startswith("| ") and line.endswith(" |") for line in lines)

    def test_format_lines_marks_regressed_bucket(self):
        """表格行的最后一列是"是/否"，与 ``regressed`` 一致."""
        before, after = make_regressing_pair()
        report = compare_runs(before, after, samples=100)
        rows = {
            line.split("|")[1].strip(): line for line in report.format_lines()
        }
        assert rows["refusal"].strip().endswith("| 是 |")
        assert rows["citation"].strip().endswith("| 否 |")

    def test_summary_line_reports_gate_state(self):
        """``summary_line`` 同时给出合格率差值与门禁状态."""
        before, after = make_regressing_pair()
        line = compare_runs(before, after, samples=100).summary_line()
        assert "before → after" in line
        assert "门禁 未通过" in line
        assert "回退分组：refusal" in line

    def test_result_frozen_dataclass_holds_shape(self):
        """``BucketDelta`` 的派生属性（合格率差 / 总分差）与输入一致."""
        bucket = BucketDelta(
            name="refusal",
            total=2,
            before_pass_rate=1.0,
            after_pass_rate=0.0,
            before_mean_score=0.85,
            after_mean_score=0.15,
        )
        assert bucket.pass_rate_delta == pytest.approx(-1.0)
        assert bucket.score_delta == pytest.approx(-0.7)
        assert set(bucket.to_dict()) == {
            "name",
            "total",
            "before_pass_rate",
            "after_pass_rate",
            "pass_rate_delta",
            "before_mean_score",
            "after_mean_score",
            "score_delta",
            "regressed",
        }
        assert bucket.regressed is False

    def test_comparison_report_passed_property_tracks_regressions(self):
        """``passed`` 就是"``regressions`` 为空"，没有第二条判据."""
        report = ComparisonReport(
            before_name="a",
            after_name="b",
            total=1,
            before_pass_rate=0.0,
            after_pass_rate=1.0,
            before_mean_score=0.0,
            after_mean_score=1.0,
        )
        assert report.passed is True
        report.regressions = ["refusal"]
        assert report.passed is False
