"""对齐策略测试（M5-D6 A 组之二）：五个维度、样本量算术、标注一致性与风险表（strategy.py）.

本文件与 ``tests/test_alignment_preference.py`` 成对：那个文件盯"手上的数据能不能
用"，这个文件盯"要买多少、怎么配"。四条承担"证明结论"角色的用例：

1. :meth:`TestPairsForMargin.test_pinned_values` —— 四档目标准确率的样本量必须
   等于 ``778 / 189 / 80 / 42``。这四个数字本身就是结论：**把偏好准确率从抛硬币
   提升到 55% 需要的标注量，比提升到 70% 高一个数量级**，所以小团队的现实选择
   是"先对齐一个明显的维度"，而不是一上来追 5% 的提升。
2. :meth:`TestPairsForMargin.test_effect_size_drives_the_count` —— 上一条的
   推论被单独钉住：55% 那档比 70% 那档多一个数量级（断言 10 倍以上），
   用一条不等式把"样本量随效应量平方反比增长"这件事放进测试。
3. :meth:`TestAnnotatorAgreement.test_degenerate_agreement_is_flagged` —— 两位
   标注者都只用了同一个标签时 ``p_e = 1``、κ 的分母为 0。此时函数返回
   ``kappa = 0.0`` 并标 ``degenerate = True``：**"完全一致但无信息"不该被报告成
   κ = 1.0**。而 :meth:`TestAnnotatorAgreement.test_fully_opposite_is_negative`
   给出另一端：两份完全相反的标注必须落到 κ = -1（"比随机还差"）。
4. :meth:`TestAlignmentPlan.test_default_plan` /
   :meth:`TestAlignmentPlan.test_single_dimension_ready` —— ``summary.ready`` 的
   口径：默认（没走 ``collected``）时每维 189 条、总计 945 条、``ready = False``；
   只有 citation 收满 200 条时它 ``ready = True``、``gap = 0``，其余四维仍
   ``not ready``、总判据仍为 ``False``。**"数据够不够开工"必须有明确答案**，
   否则最常见的做法是"先随便标几条跑起来再说"。

所有断言都与 ``strategy.py`` 的实现逐字对应。一处实现细节值得写在这里：
``annotator_agreement`` 的 κ 是浮点算出来的，规格里那个例子的 κ 落在
``0.3999999999999999``（比 0.4 小一个 ULP），因此 ``interpret_kappa`` 给出的是
"一般一致"而不是"中等一致"——本文件用 ``pytest.approx`` 断言 0.4，并把
解释档位一起钉住，避免"改一位浮点算法就悄悄换档"。
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from smart_research_agent.alignment.preference import (
    ALIGNMENT_DIMENSIONS,
    DIMENSION_GOALS,
)
from smart_research_agent.alignment.strategy import (
    ALIGNMENT_RISKS,
    DEFAULT_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TARGET_ACCURACY,
    Z_BY_ALPHA,
    Z_BY_POWER,
    SampleBudget,
    alignment_plan,
    annotator_agreement,
    dimension_table,
    interpret_kappa,
    pairs_for_margin,
    strategy_notes,
    z_for_confidence,
    z_for_power,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError

#: 四档目标准确率的冻结样本量（``alpha=0.05`` / ``power=0.80`` / ``baseline=0.5``）
PINNED_PAIRS_BY_TARGET = {0.55: 778, 0.6: 189, 0.65: 80, 0.7: 42}

#: 规格里那两批标注的冻结一致度（``kappa`` 是浮点值，落在 0.4 的下一个 ULP）
LABELS_A = ["A", "A", "B", "A", "B", "B", "A", "A", "B", "A"]
LABELS_B = ["A", "B", "B", "A", "B", "A", "A", "A", "B", "B"]

#: 风险表的声明顺序（报告顺序，改了就会让两份报告没法逐行比对）
PINNED_RISK_NAMES = ("长度偏置", "过优化", "维度混淆", "参考模型漂移")


def budget(collected: int, required: int = 189, dimension: str = "citation") -> SampleBudget:
    """构造一个 ``SampleBudget``（默认复现 ``alignment_plan`` 里的默认配额）."""
    return SampleBudget(dimension=dimension, collected=collected, required=required)


def repeated_labels(label: str, count: int) -> list[str]:
    """同一批标注里全是同一个标签——用来构造退化情形（``p_e = 1``）."""
    return [label] * count


class TestZTables:
    """``z_for_confidence`` / ``z_for_power``：刻意查表而不是引统计库，因此**只认表里的键**."""

    def test_confidence_table_is_pinned(self):
        assert Z_BY_ALPHA == {0.10: 1.6449, 0.05: 1.9600, 0.02: 2.3263, 0.01: 2.5758}

    def test_power_table_is_pinned(self):
        assert Z_BY_POWER == {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}

    @pytest.mark.parametrize(
        "alpha,expected",
        [(0.10, 1.6449), (0.05, 1.9600), (0.02, 2.3263), (0.01, 2.5758)],
    )
    def test_confidence_lookups(self, alpha, expected):
        assert z_for_confidence(alpha) == expected

    @pytest.mark.parametrize("power,expected", [(0.80, 0.8416), (0.90, 1.2816), (0.95, 1.6449)])
    def test_power_lookups(self, power, expected):
        assert z_for_power(power) == expected

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 0.99, -0.05])
    def test_unsupported_confidence_rejected(self, alpha):
        with pytest.raises(FinetuneEvalError, match="不支持的 alpha"):
            z_for_confidence(alpha)

    @pytest.mark.parametrize("power", [0.0, 0.5, 0.99, 1.0])
    def test_unsupported_power_rejected(self, power):
        with pytest.raises(FinetuneEvalError, match="不支持的 power"):
            z_for_power(power)

    def test_lookup_is_exact_not_approximate(self):
        """表查的是键的**精确**相等：``0.1 + 1e-9`` 不是 0.10，必须报错."""
        with pytest.raises(FinetuneEvalError, match="不支持的 alpha"):
            z_for_confidence(0.10 + 1e-9)
        with pytest.raises(FinetuneEvalError, match="不支持的 power"):
            z_for_power(0.80 + 1e-9)

    def test_error_message_says_it_is_a_table_lookup(self):
        """错误信息要解释"为什么不支持"：这是刻意查表，不是忘了实现."""
        with pytest.raises(FinetuneEvalError, match="刻意查表"):
            z_for_confidence(0.5)

    def test_confidence_is_monotonic(self):
        """置信水平越高（alpha 越小）→ z 越大；表里四个值必须严格递增."""
        values = [z_for_confidence(alpha) for alpha in (0.10, 0.05, 0.02, 0.01)]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_power_is_monotonic(self):
        values = [z_for_power(power) for power in (0.80, 0.90, 0.95)]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_power_table_reuses_the_confidence_value_at_95_percent(self):
        """0.95 功效与 0.10 显著水平共用 ``1.6449``——同一个单侧分位数的两个用法."""
        assert z_for_power(0.95) == z_for_confidence(0.10) == 1.6449

    def test_default_alpha_and_power_are_in_the_tables(self):
        """默认参数必须落在表里，否则 ``pairs_for_margin`` 的默认路径会直接炸."""
        assert 0.05 in Z_BY_ALPHA
        assert 0.80 in Z_BY_POWER


class TestPairsForMargin:
    """``pairs_for_margin``：要多少条偏好对，才能分辨出目标准确率与基线."""

    @pytest.mark.parametrize("target,expected", sorted(PINNED_PAIRS_BY_TARGET.items()))
    def test_pinned_values(self, target, expected):
        assert pairs_for_margin(target_accuracy=target) == expected

    def test_default_target_is_sixty_percent(self):
        assert DEFAULT_TARGET_ACCURACY == 0.6
        assert pairs_for_margin() == 189

    @pytest.mark.parametrize("target", [0.55, 0.6, 0.65, 0.7])
    def test_closed_form_is_reproduced(self, target):
        """逐字复算 ``n = ⌈(z_α + z_β)² · p(1−p) / (p − baseline)²⌉``."""
        expected = math.ceil(
            (z_for_confidence(0.05) + z_for_power(0.80)) ** 2
            * target
            * (1 - target)
            / (target - 0.5) ** 2
        )
        assert pairs_for_margin(target_accuracy=target) == expected

    def test_effect_size_drives_the_count(self):
        """效应量小一个数量级，标注量多一个数量级：这是本模块最想传达的结论."""
        assert pairs_for_margin(target_accuracy=0.55) > 10 * pairs_for_margin(target_accuracy=0.7)

    def test_higher_target_needs_fewer_pairs(self):
        """目标越高 → 越好分辨 → 需要的样本越少（四档严格单调递减）."""
        counts = [pairs_for_margin(target_accuracy=target) for target in (0.55, 0.6, 0.65, 0.7)]
        assert counts == sorted(counts, reverse=True)
        assert len(set(counts)) == len(counts)

    @pytest.mark.parametrize("baseline", [0.0, 1.0, -0.1, 1.5])
    def test_baseline_out_of_range_rejected(self, baseline):
        with pytest.raises(FinetuneEvalError, match="baseline 必须落在"):
            pairs_for_margin(target_accuracy=0.8, baseline=baseline)

    @pytest.mark.parametrize(
        "baseline,target",
        [(0.5, 0.5), (0.5, 0.4), (0.6, 0.55), (0.7, 0.6)],
    )
    def test_target_not_above_baseline_rejected(self, baseline, target):
        """目标准确率必须**严格大于**基线：等于基线时效应量为 0，样本量无意义."""
        with pytest.raises(FinetuneEvalError, match="target_accuracy 必须落在"):
            pairs_for_margin(target_accuracy=target, baseline=baseline)

    @pytest.mark.parametrize("target", [1.0, 1.2])
    def test_target_of_one_rejected(self, target):
        """``p = 1`` 时方差为 0：公式会给出"不需要样本"这种荒谬答案，必须拦住."""
        with pytest.raises(FinetuneEvalError, match="target_accuracy 必须落在"):
            pairs_for_margin(target_accuracy=target)

    def test_just_above_baseline_needs_the_most_pairs(self):
        """贴着基线的目标最难分辨：0.505 的样本量比 0.55 还大."""
        assert pairs_for_margin(target_accuracy=0.505) > pairs_for_margin(target_accuracy=0.55)

    def test_higher_baseline_needs_more_pairs(self):
        """同样的目标准确率，基线越高越难证明"不是碰巧"."""
        assert pairs_for_margin(target_accuracy=0.7, baseline=0.6) == 165
        assert pairs_for_margin(target_accuracy=0.7, baseline=0.6) > pairs_for_margin(
            target_accuracy=0.7, baseline=0.5
        )

    def test_higher_confidence_and_power_need_more_pairs(self):
        assert pairs_for_margin(target_accuracy=0.6, alpha=0.01, power=0.95) == 428
        assert pairs_for_margin(target_accuracy=0.6, alpha=0.01, power=0.95) > pairs_for_margin(
            target_accuracy=0.6
        )

    def test_looser_confidence_with_higher_power(self):
        """0.10 / 0.90 这组是"更松的显著水平 + 更高的功效"，净效果仍然更大."""
        assert pairs_for_margin(target_accuracy=0.6, alpha=0.10, power=0.90) == 206
        assert pairs_for_margin(target_accuracy=0.6, alpha=0.10, power=0.90) > pairs_for_margin(
            target_accuracy=0.6
        )

    def test_unsupported_alpha_is_propagated(self):
        with pytest.raises(FinetuneEvalError, match="不支持的 alpha"):
            pairs_for_margin(target_accuracy=0.6, alpha=0.5)

    def test_unsupported_power_is_propagated(self):
        with pytest.raises(FinetuneEvalError, match="不支持的 power"):
            pairs_for_margin(target_accuracy=0.6, power=0.5)

    def test_returns_a_plain_integer(self):
        """返回值是"条数"：必须是 int（``math.ceil`` 的产物），且为正."""
        for target in (0.55, 0.6, 0.65, 0.7):
            count = pairs_for_margin(target_accuracy=target)
            assert isinstance(count, int)
            assert count > 0


class TestAnnotatorAgreement:
    """``annotator_agreement``：观察一致度、期望一致度与 Cohen's κ."""

    def test_length_mismatch_rejected(self):
        with pytest.raises(FinetuneEvalError, match="两份标注长度必须一致"):
            annotator_agreement(["A"], ["A", "B"])

    def test_empty_labels_rejected(self):
        with pytest.raises(FinetuneEvalError, match="至少需要 2 条标注才能算一致性"):
            annotator_agreement([], [])

    @pytest.mark.parametrize("count", [0, 1])
    def test_too_few_labels_rejected(self, count):
        with pytest.raises(FinetuneEvalError, match="至少需要 2 条标注才能算一致性"):
            annotator_agreement(["A"] * count, ["A"] * count)

    def test_pinned_example(self):
        """规格里那两批标注的冻结数字：观察 0.7 / 期望 0.5 / κ ≈ 0.4."""
        report = annotator_agreement(LABELS_A, LABELS_B)
        assert report["total"] == 10
        assert report["observed_agreement"] == pytest.approx(0.7)
        assert report["expected_agreement"] == pytest.approx(0.5)
        assert report["kappa"] == pytest.approx(0.4)
        assert report["degenerate"] is False
        assert report["categories"] == ["A", "B"]

    def test_pinned_example_interpretation(self):
        """κ ≈ 0.4 落在"一般一致"档（浮点误差让它比 0.4 小一个 ULP）."""
        assert annotator_agreement(LABELS_A, LABELS_B)["interpretation"] == "一般一致"

    def test_result_keys_are_pinned(self):
        report = annotator_agreement(LABELS_A, LABELS_B)
        assert set(report) == {
            "total",
            "observed_agreement",
            "expected_agreement",
            "kappa",
            "degenerate",
            "interpretation",
            "categories",
        }

    def test_observed_agreement_is_the_share_of_equal_positions(self):
        report = annotator_agreement(["A", "B", "A", "B"], ["A", "B", "A", "A"])
        assert report["observed_agreement"] == pytest.approx(0.75)
        assert report["total"] == 4

    def test_perfect_agreement_with_two_labels(self):
        report = annotator_agreement(["A", "B", "A", "B"], ["A", "B", "A", "B"])
        assert report["observed_agreement"] == 1.0
        assert report["expected_agreement"] == pytest.approx(0.5)
        assert report["kappa"] == pytest.approx(1.0)
        assert report["interpretation"] == "几乎完全一致"

    @pytest.mark.parametrize("count", [2, 3, 5, 10])
    @pytest.mark.parametrize("label", ["A", "B"])
    def test_degenerate_agreement_is_flagged(self, count, label):
        """双方都只用同一个标签：``p_e = 1`` → κ 无定义，取 0.0 并标 degenerate.

        这**不是** κ = 1.0：在一个答案 100% 都合格的标注任务里，100% 的观察
        一致度等于零信息，报告成"几乎完全一致"会直接误导放量决策。
        """
        report = annotator_agreement(repeated_labels(label, count), repeated_labels(label, count))
        assert report["total"] == count
        assert report["observed_agreement"] == 1.0
        assert report["expected_agreement"] == 1.0
        assert report["kappa"] == 0.0
        assert report["degenerate"] is True
        assert "无信息" in report["interpretation"]
        assert report["categories"] == [label]

    def test_degenerate_wording_explains_the_zero(self):
        report = annotator_agreement(repeated_labels("A", 3), repeated_labels("A", 3))
        assert report["interpretation"] == (
            "完全一致但无信息（双方只用了同一个标签），κ 无定义，取 0.0"
        )

    def test_degenerate_detection_needs_both_sides_to_be_single_label(self):
        """只要有一方用了两个标签，``p_e`` 就不为 1，计算照常进行（这里 κ 恰为 0）."""
        report = annotator_agreement(["A", "A", "A", "A"], ["A", "A", "A", "B"])
        assert report["degenerate"] is False
        assert report["observed_agreement"] == pytest.approx(0.75)
        assert report["expected_agreement"] == pytest.approx(0.75)
        assert report["kappa"] == pytest.approx(0.0)
        assert report["interpretation"] == "轻微一致"

    def test_fully_opposite_is_negative(self):
        """两份完全相反（且边际分布相同）的标注 → κ = -1：比随机还差."""
        report = annotator_agreement(["A", "A", "B", "B"], ["B", "B", "A", "A"])
        assert report["observed_agreement"] == 0.0
        assert report["expected_agreement"] == pytest.approx(0.5)
        assert report["kappa"] == pytest.approx(-1.0)
        assert report["kappa"] < 0
        assert report["interpretation"] == "差（比随机还差）"

    def test_three_category_case(self):
        report = annotator_agreement(["A", "B", "C", "A"], ["A", "B", "C", "B"])
        assert report["observed_agreement"] == pytest.approx(0.75)
        assert report["expected_agreement"] == pytest.approx(0.3125)
        assert report["kappa"] == pytest.approx(0.636364, abs=1e-6)
        assert report["categories"] == ["A", "B", "C"]
        assert report["interpretation"] == "显著一致"

    def test_majority_agreement_case(self):
        labels_a = ["A"] * 8 + ["B"] * 2
        labels_b = ["A"] * 7 + ["B"] * 3
        report = annotator_agreement(labels_a, labels_b)
        assert report["observed_agreement"] == pytest.approx(0.9)
        assert report["expected_agreement"] == pytest.approx(0.62)
        assert report["kappa"] == pytest.approx(0.736842, abs=1e-6)
        assert report["interpretation"] == "显著一致"

    def test_categories_are_the_sorted_union(self):
        report = annotator_agreement(["B", "A", "B", "A"], ["B", "A", "B", "B"])
        assert report["categories"] == ["A", "B"]

    def test_labels_can_be_arbitrary_strings(self):
        report = annotator_agreement(["好", "差", "好", "差"], ["好", "差", "好", "好"])
        assert report["categories"] == ["好", "差"]
        assert report["observed_agreement"] == pytest.approx(0.75)

    @pytest.mark.parametrize(
        "labels_a,labels_b",
        [
            (["A", "B", "A", "B"], ["A", "B", "A", "B"]),
            (["A", "A", "B", "B"], ["B", "B", "A", "A"]),
            (["A", "B", "C", "A"], ["A", "B", "C", "B"]),
            (["A"] * 8 + ["B"] * 2, ["A"] * 7 + ["B"] * 3),
        ],
    )
    def test_kappa_is_bounded(self, labels_a, labels_b):
        """κ 落在 [-1, 1]：报告里的"一致度"必须是个能解释的相关系数."""
        report = annotator_agreement(labels_a, labels_b)
        assert -1.0 <= report["kappa"] <= 1.0

    def test_symmetry(self):
        """交换两位标注者不改变任何数字——κ 是对称的相关系数."""
        forward = annotator_agreement(LABELS_A, LABELS_B)
        backward = annotator_agreement(LABELS_B, LABELS_A)
        assert forward["kappa"] == pytest.approx(backward["kappa"])
        assert forward["observed_agreement"] == pytest.approx(backward["observed_agreement"])
        assert forward["expected_agreement"] == pytest.approx(backward["expected_agreement"])


class TestInterpretKappa:
    """``interpret_kappa``：Landis & Koch 的六个文字档，边界是**半开区间**."""

    @pytest.mark.parametrize("kappa", [-1.0, -0.5, -0.001])
    def test_poor_band(self, kappa):
        assert interpret_kappa(kappa) == "差（比随机还差）"

    @pytest.mark.parametrize("kappa", [0.0, 0.19, 0.19999])
    def test_slight_band(self, kappa):
        assert interpret_kappa(kappa) == "轻微一致"

    @pytest.mark.parametrize("kappa", [0.2, 0.3, 0.39])
    def test_fair_band(self, kappa):
        assert interpret_kappa(kappa) == "一般一致"

    @pytest.mark.parametrize("kappa", [0.4, 0.5, 0.59])
    def test_moderate_band(self, kappa):
        assert interpret_kappa(kappa) == "中等一致"

    @pytest.mark.parametrize("kappa", [0.6, 0.7, 0.79])
    def test_substantial_band(self, kappa):
        assert interpret_kappa(kappa) == "显著一致"

    @pytest.mark.parametrize("kappa", [0.8, 0.9, 1.0, 2.0])
    def test_almost_perfect_band(self, kappa):
        assert interpret_kappa(kappa) == "几乎完全一致"

    @pytest.mark.parametrize(
        "kappa,expected",
        [(0.19999, "轻微一致"), (0.2, "一般一致"), (0.4, "中等一致"), (0.6, "显著一致")],
    )
    def test_band_boundaries_are_half_open(self, kappa, expected):
        """每档都从下界开始算：0.4 属于"中等一致"，而 0.3999999999999999 属于"一般一致"."""
        assert interpret_kappa(kappa) == expected

    def test_the_pinned_example_kappa_lands_in_the_fair_band(self):
        """规格例子的 κ 是 0.3999999999999999（浮点），所以档位是"一般一致".

        顺带记录一个真实存在的边界效应：把它四舍五入到 6 位小数得到 0.4，
        而 0.4 已经落进"中等一致"档——**同一个 κ 报成 0.4 与报成
        0.3999999999999999 会得到两个不同的结论**。这正是分档边界必须写成
        半开区间、并且要被测试钉住的原因。
        """
        kappa = annotator_agreement(LABELS_A, LABELS_B)["kappa"]
        assert kappa < 0.4
        assert interpret_kappa(kappa) == "一般一致"
        assert round(kappa, 6) == 0.4
        assert interpret_kappa(round(kappa, 6)) == "中等一致"

    def test_labels_advance_in_the_declared_order(self):
        """六档的标签互不相同，且随 κ 单调前进（不会出现"回退"的档位）."""
        labels = [interpret_kappa(kappa) for kappa in (-1.0, 0.0, 0.2, 0.4, 0.6, 0.8)]
        assert labels == [
            "差（比随机还差）",
            "轻微一致",
            "一般一致",
            "中等一致",
            "显著一致",
            "几乎完全一致",
        ]

    def test_returns_a_non_empty_string(self):
        for kappa in (-2.0, 0.0, 0.25, 0.55, 0.75, 5.0):
            assert isinstance(interpret_kappa(kappa), str)
            assert interpret_kappa(kappa).strip()


class TestSampleBudget:
    """``SampleBudget``：一个维度上的目标与实际收集量（``gap`` 不返回负数）."""

    def test_gap_is_the_shortfall(self):
        assert budget(100).gap == 89

    @pytest.mark.parametrize("collected", [189, 200, 1000])
    def test_gap_is_clamped_at_zero(self, collected):
        """"超出目标"不是缺口：``gap`` 必须夹到 0，不能出现负数."""
        assert budget(collected).gap == 0

    @pytest.mark.parametrize("collected,expected", [(0, False), (188, False), (189, True), (400, True)])
    def test_ready_flag(self, collected, expected):
        assert budget(collected).ready is expected

    def test_summary_line_when_ready(self):
        assert budget(200).summary_line() == "citation: 200/189（可开工）"

    def test_summary_line_when_not_ready(self):
        assert budget(3).summary_line() == "citation: 3/189（还差 186 条）"

    def test_to_dict_keys_and_values(self):
        payload = budget(200).to_dict()
        assert list(payload) == ["dimension", "collected", "required", "gap", "ready", "goal"]
        assert payload["dimension"] == "citation"
        assert payload["collected"] == 200
        assert payload["required"] == 189
        assert payload["gap"] == 0
        assert payload["ready"] is True
        assert payload["goal"] == ""

    def test_goal_defaults_to_empty_string(self):
        assert budget(1, dimension="refusal").goal == ""

    def test_budget_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            budget(1).collected = 5  # type: ignore[misc]


class TestDimensionTable:
    """``dimension_table``：五个对齐维度的目标说明（进 API 响应与文档）."""

    def test_has_five_dimensions(self):
        assert len(dimension_table()) == len(ALIGNMENT_DIMENSIONS) == 5

    def test_order_matches_the_declared_dimensions(self):
        assert [item["dimension"] for item in dimension_table()] == list(ALIGNMENT_DIMENSIONS)

    def test_each_entry_has_dimension_and_goal(self):
        for item in dimension_table():
            assert set(item) == {"dimension", "goal"}
            assert isinstance(item["goal"], str)

    def test_goals_match_the_declared_map(self):
        for item in dimension_table():
            assert item["goal"] == DIMENSION_GOALS[item["dimension"]]

    def test_goals_are_non_empty(self):
        for item in dimension_table():
            assert item["goal"].strip()

    def test_returns_a_fresh_list_each_call(self):
        """每次调用都新建：调用方就地改一改不该污染下一次的响应."""
        first = dimension_table()
        first[0]["goal"] = "被改坏"
        assert dimension_table()[0]["goal"] == DIMENSION_GOALS["citation"]

    def test_two_calls_are_equal(self):
        assert dimension_table() == dimension_table()


class TestAlignmentPlan:
    """``alignment_plan``：把"要买什么、要多少、怎么配"打成一份可执行的计划."""

    def test_default_plan(self):
        plan = alignment_plan()
        assert plan["per_dimension_required"] == 189
        assert plan["target_accuracy"] == DEFAULT_TARGET_ACCURACY == 0.6
        assert plan["alpha"] == 0.05
        assert plan["power"] == 0.80

    def test_default_summary(self):
        summary = alignment_plan()["summary"]
        assert summary == {
            "collected": 0,
            "required": 945,
            "gap": 945,
            "ready_dimensions": 0,
            "dimensions": 5,
            "ready": False,
        }

    def test_default_budgets(self):
        budgets = alignment_plan()["budgets"]
        assert len(budgets) == 5
        assert [item["dimension"] for item in budgets] == list(ALIGNMENT_DIMENSIONS)
        for item in budgets:
            assert item["collected"] == 0
            assert item["required"] == 189
            assert item["gap"] == 189
            assert item["ready"] is False

    def test_budgets_are_plain_dicts_carrying_the_goal(self):
        for item in alignment_plan()["budgets"]:
            assert isinstance(item, dict)
            assert item["goal"] == DIMENSION_GOALS[item["dimension"]]

    def test_single_dimension_ready(self):
        """只把 citation 收满：它开工，其余四维仍差 189 条，总判据仍为 False."""
        plan = alignment_plan(collected={"citation": 200})
        budgets = {item["dimension"]: item for item in plan["budgets"]}
        assert budgets["citation"]["ready"] is True
        assert budgets["citation"]["gap"] == 0
        assert budgets["citation"]["collected"] == 200
        for dimension in ("format", "conciseness", "refusal", "honesty"):
            assert budgets[dimension]["ready"] is False
            assert budgets[dimension]["gap"] == 189
        assert plan["summary"]["ready"] is False
        assert plan["summary"]["ready_dimensions"] == 1
        assert plan["summary"]["gap"] == 745

    def test_all_dimensions_ready(self):
        plan = alignment_plan(collected={dimension: 189 for dimension in ALIGNMENT_DIMENSIONS})
        assert plan["summary"]["ready"] is True
        assert plan["summary"]["ready_dimensions"] == 5
        assert plan["summary"]["gap"] == 0
        assert plan["summary"]["collected"] == 945
        assert all(item["ready"] for item in plan["budgets"])

    def test_surplus_is_ready_and_gap_stays_zero(self):
        """收多了不该出现负缺口：``gap`` 是"还差多少"，不是"多了多少"."""
        plan = alignment_plan(collected={dimension: 1000 for dimension in ALIGNMENT_DIMENSIONS})
        assert plan["summary"]["ready"] is True
        assert plan["summary"]["gap"] == 0
        assert plan["summary"]["collected"] == 5000
        assert all(item["gap"] == 0 for item in plan["budgets"])

    def test_one_dimension_short_blocks_the_whole_plan(self):
        """四个维度收满、一个差一条：总判据必须是 False（``all`` 的口径）."""
        collected = {dimension: 200 for dimension in ALIGNMENT_DIMENSIONS}
        collected["refusal"] = 188
        plan = alignment_plan(collected=collected)
        assert plan["summary"]["ready"] is False
        assert plan["summary"]["ready_dimensions"] == 4

    def test_none_and_empty_collected_are_equivalent(self):
        assert alignment_plan(collected=None)["summary"] == alignment_plan(collected={})["summary"]

    @pytest.mark.parametrize("dimension", ["unknown", "Citation", "工具", "citation "])
    def test_unknown_dimension_rejected(self, dimension):
        """维度名不容错：拼错会让"某个维度的量"悄悄变成另一个维度的量."""
        with pytest.raises(FinetuneEvalError, match="未知的对齐维度"):
            alignment_plan(collected={dimension: 10})

    def test_unknown_dimensions_are_listed_sorted(self):
        with pytest.raises(FinetuneEvalError) as excinfo:
            alignment_plan(collected={"nope": 1, "bad": 2})
        message = str(excinfo.value)
        assert "bad, nope" in message
        assert "citation" in message

    @pytest.mark.parametrize("value", [-1, -189])
    def test_negative_collected_rejected(self, value):
        with pytest.raises(FinetuneEvalError, match="不能为负数"):
            alignment_plan(collected={"citation": value})

    def test_zero_collected_is_allowed(self):
        assert alignment_plan(collected={"citation": 0})["summary"]["collected"] == 0

    def test_custom_target_accuracy_changes_the_quota(self):
        plan = alignment_plan(target_accuracy=0.65)
        assert plan["per_dimension_required"] == 80
        assert plan["summary"]["required"] == 400
        assert plan["summary"]["gap"] == 400

    def test_custom_alpha_and_power(self):
        plan = alignment_plan(alpha=0.01, power=0.95)
        assert plan["per_dimension_required"] == 428
        assert plan["summary"]["required"] == 2140
        assert plan["alpha"] == 0.01
        assert plan["power"] == 0.95

    def test_unsupported_alpha_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不支持的 alpha"):
            alignment_plan(alpha=0.5)

    def test_unsupported_power_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不支持的 power"):
            alignment_plan(power=0.5)

    def test_invalid_target_accuracy_rejected(self):
        with pytest.raises(FinetuneEvalError, match="target_accuracy 必须落在"):
            alignment_plan(target_accuracy=0.5)

    def test_dpo_config_defaults(self):
        config = alignment_plan()["dpo_config"]
        assert config["beta"] == DEFAULT_BETA == 0.1
        assert config["learning_rate"] == DEFAULT_LEARNING_RATE == 0.5
        assert config["kl_budget"] == 0.5
        assert config["valid_ratio"] == 0.25
        assert "β" in config["note"]
        assert "不要照抄" in config["note"]

    def test_dpo_config_can_be_overridden(self):
        config = alignment_plan(beta=0.3, learning_rate=1e-5, kl_budget=2.0)["dpo_config"]
        assert config["beta"] == 0.3
        assert config["learning_rate"] == 1e-5
        assert config["kl_budget"] == 2.0

    def test_dpo_config_keys_are_pinned(self):
        """β / 学习率 / KL 预算三者缺一不可：只看训练 loss 的 DPO 报告是不完整的."""
        assert set(alignment_plan()["dpo_config"]) == {
            "beta",
            "learning_rate",
            "valid_ratio",
            "kl_budget",
            "note",
        }

    def test_dimensions_section_equals_dimension_table(self):
        assert alignment_plan()["dimensions"] == dimension_table()

    def test_risks_section_has_four_entries(self):
        risks = alignment_plan()["risks"]
        assert len(risks) == len(ALIGNMENT_RISKS) == 4
        assert tuple(risk["name"] for risk in risks) == PINNED_RISK_NAMES

    def test_every_risk_has_the_four_declared_fields(self):
        """每条风险都要"症状 + 缓解 + 怎么发现"三件套齐全，否则它只是口号."""
        for risk in ALIGNMENT_RISKS:
            assert set(risk) == {"name", "symptom", "mitigation", "detect"}
            assert all(value.strip() for value in risk.values())

    def test_risk_detect_fields_point_at_real_entry_points(self):
        detects = {risk["name"]: risk["detect"] for risk in ALIGNMENT_RISKS}
        assert "length_bias_report" in detects["长度偏置"]
        assert "over_optimization_flags" in detects["过优化"]
        assert "annotator_agreement" in detects["维度混淆"]
        assert "指纹" in detects["参考模型漂移"]

    def test_risks_are_copies_of_the_module_constant(self):
        """计划里改一条风险不该污染模块常量（下一次请求还得是原文）."""
        plan = alignment_plan()
        plan["risks"][0]["name"] = "被改坏"
        plan["risks"][0]["symptom"] = "被改坏"
        assert ALIGNMENT_RISKS[0]["name"] == "长度偏置"
        assert ALIGNMENT_RISKS[0]["symptom"] != "被改坏"
        assert alignment_plan()["risks"][0]["name"] == "长度偏置"

    def test_plan_keys_are_pinned(self):
        assert set(alignment_plan()) == {
            "target_accuracy",
            "alpha",
            "power",
            "per_dimension_required",
            "budgets",
            "summary",
            "dpo_config",
            "dimensions",
            "risks",
        }

    def test_plan_is_json_serializable(self):
        """计划是要被 API 返回的：整个返回结构必须能过 ``json.dumps``."""
        text = json.dumps(alignment_plan(), ensure_ascii=False)
        payload: dict[str, Any] = json.loads(text)
        assert payload["per_dimension_required"] == 189
        assert payload["budgets"][0]["dimension"] == "citation"

    def test_required_is_quota_times_dimensions(self):
        for target in (0.6, 0.65):
            plan = alignment_plan(target_accuracy=target)
            assert plan["summary"]["required"] == plan["per_dimension_required"] * 5
            assert plan["summary"]["dimensions"] == 5

    def test_collected_mapping_is_not_mutated(self):
        collected = {"citation": 10}
        alignment_plan(collected=collected)
        assert collected == {"citation": 10}

    def test_extra_dimensions_beyond_the_table_are_rejected(self):
        """``collected`` 里混进未知维度会直接报错，而不是被默默忽略."""
        with pytest.raises(FinetuneEvalError, match="未知的对齐维度"):
            alignment_plan(collected={"citation": 10, "safety": 10})


class TestStrategyNotes:
    """``strategy_notes``：开工前先读一遍的流程建议（与风险表互补）."""

    def test_has_five_notes(self):
        assert len(strategy_notes()) == 5

    def test_all_notes_are_non_empty_strings(self):
        for note in strategy_notes():
            assert isinstance(note, str)
            assert note.strip()

    def test_notes_cover_the_key_practices(self):
        """五条建议分别对应：样本量、参考模型、κ、验证集、KL+验证准确率."""
        notes = strategy_notes()
        assert any("pairs_for_margin" in note for note in notes)
        assert any("参考模型" in note for note in notes)
        assert any("κ" in note for note in notes)
        assert any("验证集" in note for note in notes)
        assert any("KL" in note for note in notes)

    def test_notes_mention_the_kappa_threshold(self):
        """κ < 0.4 时"放量等于放大噪声"——这条阈值必须写进建议里."""
        assert any("0.4" in note for note in strategy_notes())

    def test_returns_a_fresh_list_each_call(self):
        notes = strategy_notes()
        notes.append("被追加的一条")
        assert len(strategy_notes()) == 5

    def test_two_calls_are_equal(self):
        assert strategy_notes() == strategy_notes()
