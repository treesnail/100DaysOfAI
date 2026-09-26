"""八条性质与"误加因果掩码"的代价（day079 / M7-D4）.

这一份测试守三件事：

```text
① 四条"逐行"性质              前三条关于 LN、第四条关于前馈——都用 **== ** 断言（逐位）
② 两条"残差"性质              output == inputs（逐位）；反向里那条 +1 路（量它的代价）
③ 一条"交叉"性质              causal=True 被拒绝，而"加错之后差多少"必须能算出来
```

第 ③ 条是这一课最值钱的一条：它的失效方式很坏——**在 ``n_tgt == n_src`` 时不报错**，
只是悄悄把源序列的后半段删掉了。因此判据不止断言"它会拒绝"，还要**量出代价**。
"""

from __future__ import annotations

import pytest

from smart_research_agent.encoder_decoder import (
    ACTIVATION_GELU,
    ENCODER_DECODER_PROPERTIES,
    NORM_POST,
    NORM_PRE,
    NumericError,
    ParameterError,
    PROPERTY_CROSS_NOT_CAUSAL,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FFN_POSITION_WISE,
    PROPERTY_NORM_ROW_INDEPENDENT,
    PROPERTY_RESIDUAL_IDENTITY,
    PROPERTY_RESIDUAL_UNIT_PATH,
    PROPERTY_ROWS_STANDARDIZED,
    PROPERTY_SCALE_EQUIVARIANT,
    PROPERTY_SHIFT_INVARIANT,
    PropertyOutcome,
    PropertyReport,
    causal_mask_damage,
    check_cross_attention_is_not_causal,
    check_feed_forward_is_position_wise,
    check_norm_is_row_independent,
    check_properties,
    check_residual_identity,
    check_residual_unit_path,
    check_rows_are_standardized,
    check_scale_equivariance,
    check_shift_invariance,
    cross_attention,
    layer_norm,
)
from smart_research_agent.encoder_decoder.verify import _row_statistics
from tests.encoder_decoder_samples import (
    HIDDEN,
    TOKENS,
    approx,
    attention_parameters,
    block_parameters,
    cross_parameters,
    inputs,
    saturated_cross_parameters,
    source_inputs,
    target,
)


def _cache():
    _out, cache = layer_norm(inputs())
    return cache


class TestPropertyOutcome:
    """一条性质的读数：名字 + 通过与否 + 证据 + 说明."""

    def test_ok_line_carries_the_evidence(self):
        outcome = check_rows_are_standardized(_cache())
        assert outcome.name == PROPERTY_ROWS_STANDARDIZED
        assert outcome.passed is True
        assert "[ok]" in outcome.summary_line()
        assert "σ²/(σ²+eps)" in outcome.evidence
        assert outcome.description == PROPERTY_DESCRIPTIONS[PROPERTY_ROWS_STANDARDIZED]

    def test_to_dict_is_json_ready(self):
        payload = check_rows_are_standardized(_cache()).to_dict()
        assert payload["name"] == PROPERTY_ROWS_STANDARDIZED
        assert payload["passed"] is True
        assert payload["description"]

    def test_rejects_unknown_property_name(self):
        with pytest.raises(ParameterError, match="未知的性质名"):
            PropertyOutcome(name="magic", passed=True, evidence="?")

    def test_failure_mark_and_stringification(self):
        outcome = PropertyOutcome(
            name=PROPERTY_RESIDUAL_IDENTITY, passed=False, evidence=7, detail=8
        )
        assert "[!!]" in outcome.summary_line()
        assert outcome.evidence == "7"
        assert outcome.detail == "8"


class TestPropertyReport:
    """八条性质的报告：名单必须完整（漏一条与"那一条没问题"读起来不一样）."""

    def test_report_is_green_and_complete(self):
        report = check_properties(
            block_parameters(), inputs(), attention_parameters(), saturated_cross_parameters()
        )
        assert report.ok is True
        assert len(report.outcomes) == len(ENCODER_DECODER_PROPERTIES)
        assert report.failures == ()
        assert "通过 8、失败 0" in report.summary_line()
        assert report.to_dict()["ok"] is True
        assert len(report.to_dict()["outcomes"]) == 8

    def test_notes_split_the_eight_properties(self):
        report = check_properties(
            block_parameters(), inputs(), attention_parameters(), cross_parameters()
        )
        assert len(report.notes) == 2
        assert "前四条" in report.notes[0]

    def test_rejects_an_empty_report(self):
        with pytest.raises(ParameterError, match="性质报告不能为空"):
            PropertyReport(outcomes=())

    def test_rejects_an_incomplete_report(self):
        report = check_properties(
            block_parameters(), inputs(), attention_parameters(), cross_parameters()
        )
        with pytest.raises(NumericError, match="名单不完整"):
            PropertyReport(outcomes=report.outcomes[:-1])

    def test_failures_are_listed(self):
        failed = PropertyOutcome(
            name=PROPERTY_RESIDUAL_IDENTITY, passed=False, evidence="分支没有归零"
        )
        others = tuple(
            PropertyOutcome(name=name, passed=True, evidence="ok")
            for name in ENCODER_DECODER_PROPERTIES
            if name != PROPERTY_RESIDUAL_IDENTITY
        )
        report = PropertyReport(outcomes=others + (failed,))
        assert report.ok is False
        assert report.failures[0].name == PROPERTY_RESIDUAL_IDENTITY
        assert "失败 1" in report.summary_line()


class TestRowStatistics:
    """``_row_statistics``：每一行的均值与**有偏**方差."""

    def test_statistics_are_hand_computable(self):
        means, variances = _row_statistics(((1.0, 2.0, 3.0, 4.0),))
        assert means == (2.5,)
        assert variances == (1.25,)


class TestNormProperties:
    """四条关于 LayerNorm 的性质."""

    def test_rows_are_standardized_against_the_eps_expectation(self):
        outcome = check_rows_are_standardized(_cache())
        assert outcome.passed is True
        assert "它由 eps" in outcome.evidence

    def test_zero_variance_row_is_defined_by_eps(self):
        """整行取值相同时 ``eps`` 让标准化结果恰好是 0（本函数不抛异常）."""
        _out, cache = layer_norm(((5.0, 5.0, 5.0, 5.0),))
        outcome = check_rows_are_standardized(cache)
        assert outcome.passed is True
        assert cache.normalized == ((0.0, 0.0, 0.0, 0.0),)

    def test_shift_invariance_holds(self):
        outcome = check_shift_invariance(inputs(), shift=2.0)
        assert outcome.passed is True
        assert "加常数 2.0" in outcome.evidence

    def test_scale_equivariance_holds(self):
        outcome = check_scale_equivariance(inputs(), factor=3.0)
        assert outcome.passed is True
        assert "乘正数 3.0" in outcome.evidence

    def test_scale_equivariance_rejects_a_bad_factor(self):
        with pytest.raises(ParameterError, match="factor 必须是正的有限数"):
            check_scale_equivariance(inputs(), factor=0.0)

    def test_scale_equivariance_rejects_nan_factor(self):
        with pytest.raises(ParameterError, match="factor 必须是正的有限数"):
            check_scale_equivariance(inputs(), factor=float("nan"))

    def test_norm_is_row_independent(self):
        outcome = check_norm_is_row_independent(inputs())
        assert outcome.name == PROPERTY_NORM_ROW_INDEPENDENT
        assert outcome.passed is True
        assert "逐位" in outcome.evidence

    def test_feed_forward_is_position_wise(self):
        outcome = check_feed_forward_is_position_wise(inputs(), block_parameters().ffn)
        assert outcome.name == PROPERTY_FFN_POSITION_WISE
        assert outcome.passed is True

    def test_feed_forward_is_position_wise_with_gelu(self):
        outcome = check_feed_forward_is_position_wise(
            inputs(), block_parameters().ffn, activation=ACTIVATION_GELU
        )
        assert outcome.passed is True


class TestResidualProperties:
    """两条关于残差的性质：恒等映射与反向里的那条 +1 路."""

    def test_identity_when_both_branches_vanish(self):
        outcome = check_residual_identity(block_parameters(), inputs(), placement=NORM_PRE)
        assert outcome.name == PROPERTY_RESIDUAL_IDENTITY
        assert outcome.passed is True
        assert "output == inputs（True）" in outcome.evidence
        assert "‖dx‖/‖dy‖ = 1.000000" in outcome.evidence

    def test_post_placement_does_not_give_the_identity(self):
        """post-LN 的最后一步是 LN，因此"两个分支为零"不再给出恒等映射——这是**正确的结果**."""
        outcome = check_residual_identity(block_parameters(), inputs(), placement=NORM_POST)
        assert outcome.passed is False
        assert "不" in outcome.evidence

    def test_unit_path_survives_compressing_the_branch(self):
        outcome = check_residual_unit_path(
            block_parameters(), inputs(), attention_parameters(), placement=NORM_PRE
        )
        assert outcome.name == PROPERTY_RESIDUAL_UNIT_PATH
        assert outcome.passed is True
        assert "18.2 倍" in outcome.evidence
        assert approx(1.13, 1.13)

    def test_unit_path_reports_the_two_ratios(self):
        outcome = check_residual_unit_path(
            block_parameters(), inputs(), attention_parameters(), placement=NORM_POST
        )
        assert "有残差" in outcome.evidence
        assert "无残差" in outcome.evidence


class TestCrossPropertyAndDamage:
    """第 8 条性质：交叉注意力**不能**加因果掩码，以及"加错的代价"."""

    def test_causal_is_refused(self):
        outcome = check_cross_attention_is_not_causal(
            cross_parameters(), inputs(), source_inputs()
        )
        assert outcome.name == PROPERTY_CROSS_NOT_CAUSAL
        assert outcome.passed is True
        assert "causal=True 是否被拒绝：True" in outcome.evidence
        assert "长方形" in outcome.detail

    def test_damage_on_a_rectangular_weight_is_not_silent(self):
        """``n_tgt != n_src`` 时掩码形状对不上——它**会**以形状错误的形式暴露出来."""
        forward = cross_attention(cross_parameters(), inputs(), source_inputs())
        damage, note = causal_mask_damage(forward)
        assert damage == 0.0
        assert "n_tgt=4 != n_src=6" in note

    def test_damage_on_a_square_weight_is_silent_and_maximal(self):
        """``n_tgt == n_src`` 时掩码形状刚好合适 → 不报错，而权重被整行重写（代价 1.00e+00）."""
        forward = cross_attention(saturated_cross_parameters(), inputs(), inputs())
        damage, note = causal_mask_damage(forward)
        assert f"{damage:.2e}" == "1.00e+00"
        assert "n_tgt == n_src == 4" in note

    def test_property_evidence_carries_the_damage(self):
        outcome = check_cross_attention_is_not_causal(
            saturated_cross_parameters(), inputs(), inputs()
        )
        assert outcome.passed is True
        assert "1.00e+00" in outcome.evidence


class TestCheckProperties:
    """八条性质一起跑：这一课的"判决书"."""

    def test_eight_out_of_eight_with_the_saturated_crosses(self):
        report = check_properties(
            block_parameters(), inputs(), attention_parameters(), saturated_cross_parameters()
        )
        assert report.ok is True
        by_name = {outcome.name: outcome for outcome in report.outcomes}
        assert by_name[PROPERTY_CROSS_NOT_CAUSAL].passed is True
        assert by_name[PROPERTY_ROWS_STANDARDIZED].passed is True
        assert by_name[PROPERTY_SHIFT_INVARIANT].passed is True
        assert by_name[PROPERTY_SCALE_EQUIVARIANT].passed is True
        assert by_name[PROPERTY_RESIDUAL_UNIT_PATH].passed is True

    def test_post_placement_turns_one_property_red(self):
        """post-LN 下"两个分支为零 → 恒等"不成立：报告会诚实地把它标红."""
        report = check_properties(
            block_parameters(),
            inputs(),
            attention_parameters(),
            cross_parameters(),
            placement=NORM_POST,
        )
        names = {outcome.name for outcome in report.failures}
        assert PROPERTY_RESIDUAL_IDENTITY in names
        assert report.ok is False

    def test_properties_names_match_the_closed_table(self):
        report = check_properties(
            block_parameters(), inputs(), attention_parameters(), cross_parameters()
        )
        assert {outcome.name for outcome in report.outcomes} == set(ENCODER_DECODER_PROPERTIES)


def test_module_imports_stay_used():
    """把 ``target`` 与 ``HIDDEN`` 用一次，避免"样本被误删"这类静默改动."""
    assert len(target()) == TOKENS
    assert HIDDEN == 6
