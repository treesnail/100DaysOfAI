"""day055 DPO 损失函数测试：三种损失的取值、梯度与起点自检（全部离线）.

本课最重要的两条纪律都在这里被钉住：

1. **解析梯度必须等于数值梯度**——中心差分是独立于实现的第二意见；
2. **起点（margin=0）的损失是可闭式预期的**——它是训练前的自检值，
   期望值写错（比如换了 loss_type 却沿用 ln2）必须立刻红。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.dpo import (
    DEFAULT_LOSS_TYPE,
    GRADIENT_CHECK_STEP,
    LOSS_TYPES,
    DPOError,
    check_loss_type,
    describe_loss,
    expected_zero_margin_tolerance,
    hinge_dpo_gradient,
    hinge_dpo_loss,
    ipo_dpo_gradient,
    ipo_dpo_loss,
    loss_gradient,
    loss_table,
    loss_value,
    numeric_gradient,
    sigmoid_dpo_gradient,
    sigmoid_dpo_loss,
    zero_margin_loss,
    zero_margin_losses,
)

BETA = 0.1


class TestCheckLossType:
    def test_all_three_names_accepted(self):
        for name in LOSS_TYPES:
            assert check_loss_type(name) == name

    def test_default_is_sigmoid(self):
        assert DEFAULT_LOSS_TYPE == "sigmoid"
        assert LOSS_TYPES == ("sigmoid", "hinge", "ipo")

    def test_unknown_name_rejected_with_choices(self):
        with pytest.raises(DPOError, match="未知的损失函数"):
            check_loss_type("kto_pair")

    def test_error_message_lists_alternatives(self):
        with pytest.raises(DPOError) as excinfo:
            check_loss_type("nope")
        assert "sigmoid" in str(excinfo.value)


class TestSigmoidLoss:
    def test_zero_margin_is_ln2(self):
        """起点自检值 ln2：唯一能与 day054 的 ZERO_MARGIN_LOSS 直接对照的一档."""
        assert sigmoid_dpo_loss(0.0, beta=BETA) == pytest.approx(math.log(2), abs=1e-12)

    def test_loss_decreases_as_margin_grows(self):
        assert sigmoid_dpo_loss(5.0, beta=BETA) < sigmoid_dpo_loss(0.0, beta=BETA)

    def test_gradient_at_zero_is_minus_beta_over_two(self):
        assert sigmoid_dpo_gradient(0.0, beta=BETA) == pytest.approx(-BETA / 2)

    def test_gradient_saturates_for_large_margin(self):
        """分对了就不再使劲：margin 很大时梯度趋近 0。"""
        assert abs(sigmoid_dpo_gradient(200.0, beta=BETA)) < 1e-8

    def test_gradient_is_always_negative(self):
        for margin in (-3.0, 0.0, 3.0, 10.0):
            assert sigmoid_dpo_gradient(margin, beta=BETA) < 0


class TestHingeLoss:
    def test_zero_loss_beyond_margin(self):
        """β·m >= 1 时合页完全不再产生梯度（margin 超过 1/β）。"""
        assert hinge_dpo_loss(10.0, beta=BETA) == 0.0
        assert hinge_dpo_gradient(10.0, beta=BETA) == 0.0

    def test_loss_within_margin(self):
        assert hinge_dpo_loss(0.0, beta=BETA) == pytest.approx(1.0)
        assert hinge_dpo_gradient(0.0, beta=BETA) == pytest.approx(-BETA)

    def test_gradient_discontinuous_at_threshold(self):
        """间断点取次梯度 0（与 TRL 的 relu 一致），两侧不连续。"""
        assert hinge_dpo_gradient(9.999, beta=BETA) == pytest.approx(-BETA)
        assert hinge_dpo_gradient(10.0, beta=BETA) == 0.0


class TestIPOLoss:
    def test_target_margin_is_reciprocal_of_two_beta(self):
        """IPO 的目标 margin 是 1/(2β)：到点即零损失、零梯度。"""
        target = 1.0 / (2.0 * BETA)
        assert ipo_dpo_loss(target, beta=BETA) == pytest.approx(0.0)
        assert ipo_dpo_gradient(target, beta=BETA) == pytest.approx(0.0)

    def test_gradient_sign_flips_across_target(self):
        target = 1.0 / (2.0 * BETA)
        assert ipo_dpo_gradient(target - 1.0, beta=BETA) < 0
        assert ipo_dpo_gradient(target + 1.0, beta=BETA) > 0

    def test_zero_margin_value_is_one_over_four_beta_squared(self):
        """起点值 1/(4β²) 不是 ln2——换档必须同时换自检期望值。"""
        assert ipo_dpo_loss(0.0, beta=BETA) == pytest.approx(1.0 / (4 * BETA**2))


class TestDispatch:
    @pytest.mark.parametrize("name", LOSS_TYPES)
    def test_value_matches_direct_call(self, name):
        direct = {
            "sigmoid": sigmoid_dpo_loss,
            "hinge": hinge_dpo_loss,
            "ipo": ipo_dpo_loss,
        }[name]
        assert loss_value(name, 2.5, beta=BETA) == pytest.approx(direct(2.5, beta=BETA))

    @pytest.mark.parametrize("name", LOSS_TYPES)
    def test_gradient_matches_direct_call(self, name):
        direct = {
            "sigmoid": sigmoid_dpo_gradient,
            "hinge": hinge_dpo_gradient,
            "ipo": ipo_dpo_gradient,
        }[name]
        assert loss_gradient(name, 2.5, beta=BETA) == pytest.approx(
            direct(2.5, beta=BETA)
        )

    def test_unknown_name_rejected(self):
        with pytest.raises(DPOError):
            loss_value("bco_pair", 0.0, beta=BETA)


class TestZeroMargin:
    @pytest.mark.parametrize("name", LOSS_TYPES)
    def test_zero_margin_matches_loss_value(self, name):
        assert zero_margin_loss(name, beta=BETA) == pytest.approx(
            loss_value(name, 0.0, beta=BETA)
        )

    def test_zero_margin_losses_covers_all_types(self):
        values = zero_margin_losses(beta=BETA)
        assert set(values) == set(LOSS_TYPES)
        assert values["sigmoid"] == pytest.approx(math.log(2), abs=1e-12)


class TestNumericGradient:
    @pytest.mark.parametrize("name", LOSS_TYPES)
    @pytest.mark.parametrize("margin", [-2.0, 0.0, 1.5, 6.0])
    def test_matches_analytic_gradient(self, name, margin):
        """中心差分核对解析梯度：这是"梯度确实是我写下的那个式子"的独立证据。"""
        numeric = numeric_gradient(name, margin, beta=BETA)
        analytic = loss_gradient(name, margin, beta=BETA)
        assert numeric == pytest.approx(analytic, abs=1e-5)

    def test_non_positive_step_rejected(self):
        with pytest.raises(DPOError, match="差分步长必须为正数"):
            numeric_gradient("sigmoid", 0.0, beta=BETA, step=0.0)

    def test_default_step_is_documented_constant(self):
        assert GRADIENT_CHECK_STEP == 1e-6


class TestBetaValidation:
    @pytest.mark.parametrize("bad_beta", [0.0, -0.1])
    def test_non_positive_beta_rejected(self, bad_beta):
        with pytest.raises(DPOError, match="beta 必须为正数"):
            sigmoid_dpo_loss(0.0, beta=bad_beta)

    def test_zero_margin_losses_validates_beta(self):
        with pytest.raises(DPOError):
            zero_margin_losses(beta=0.0)


class TestLossTable:
    def test_table_has_all_three_rows(self):
        table = loss_table(beta=BETA)
        assert [row["loss_type"] for row in table] == list(LOSS_TYPES)

    def test_table_numbers_come_from_the_functions(self):
        """表格里的数字必须是算出来的，不是抄进文档的常量。"""
        for row in loss_table(beta=BETA):
            expected = zero_margin_loss(row["loss_type"], beta=BETA)
            assert row["zero_margin_loss"] == pytest.approx(round(expected, 6))

    def test_each_row_explains_beta_role(self):
        for row in loss_table(beta=BETA):
            assert row["formula"] and row["gradient"] and row["beta_role"]

    def test_describe_loss_returns_single_row(self):
        row = describe_loss("ipo", beta=BETA)
        assert row["loss_type"] == "ipo"
        assert "1/(2β)" in row["beta_role"]


class TestTolerance:
    def test_tolerance_scales_with_expected_magnitude(self):
        assert expected_zero_margin_tolerance(0.0) == pytest.approx(1e-9)
        assert expected_zero_margin_tolerance(25.0) == pytest.approx(25.0 * 1e-9)

    def test_tolerance_rejects_real_wiring_mistake(self):
        """把 ipo 的起点值写成 ln2 的话，误差远大于容差。"""
        tolerance = expected_zero_margin_tolerance(zero_margin_loss("ipo", beta=0.01))
        assert abs(zero_margin_loss("ipo", beta=0.01) - math.log(2)) > tolerance
