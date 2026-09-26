"""九项块级 / 六项交叉 / 两项解码器梯度校验与报告记录（day079 / M7-D4）.

这一份测试守三件事：

```text
① 报告的名单是**闭合**的      少一项的报告在"全绿"时看起来与完整报告一样
② 四组对照口径必须两边一致    pre/post × 残差开/关 都要对到 ~1e-10（容差 1e-6）
③ source 那一侧是两条链之和   少一条不报错，只会让梯度偏小——而它会被抓出来
```

第 ① 条是这一份里最容易被忽略的一处：``GradientReport`` 校验"校验项的集合 == 名单"，
所以"漏了一项"与"这一项没问题"在报告层面**不可能**长得一样。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.encoder_decoder import (
    ACTIVATION_GELU,
    BLOCK_GRADIENT_TARGETS,
    CROSS_GRADIENT_TARGETS,
    NORM_POST,
    NORM_PRE,
    AssemblyError,
    BlockGradients,
    CrossGradients,
    GradientError,
    GradientOutcome,
    GradientReport,
    NumericError,
    ParameterError,
    ShapeError,
    block_loss,
    check_block_gradients,
    check_cross_gradients,
    check_decoder_input_gradients,
    cross_loss,
    numerical_block_gradients,
    numerical_cross_gradients,
)
from smart_research_agent.encoder_decoder.verify import (
    GRADIENT_TOLERANCE,
    KIND_BLOCK,
    KIND_CROSS,
    KIND_DECODER,
    _checked_tolerance,
    _compare,
    _report,
    _scaled_error,
    permute_rows,
    reverse_permutation,
)
from smart_research_agent.transformer_core.errors import ShapeError as CoreShapeError
from smart_research_agent.transformer_core.layers import self_attention
from tests.encoder_decoder_samples import (
    HIDDEN,
    SOURCE_TOKENS,
    TOKENS,
    attention_parameters,
    block_parameters,
    cross_parameters,
    decoder_betas,
    decoder_gammas,
    decoder_inputs,
    decoder_target,
    encoder_outputs,
    inputs,
    source_inputs,
    target,
)


def _cross_case():
    return cross_parameters(), inputs(), source_inputs(), target()


class TestGradientOutcome:
    """一项梯度校验的读数：两个非负量、一个容差、一个逐点数."""

    def _outcome(self, **overrides):
        payload = {
            "target": "inputs",
            "max_absolute_error": 1e-11,
            "max_scaled_error": 1e-11,
            "tolerance": GRADIENT_TOLERANCE,
            "compared_points": 24,
        }
        payload.update(overrides)
        return GradientOutcome(**payload)

    def test_passed_is_the_scaled_error_against_the_tolerance(self):
        assert self._outcome().passed is True
        assert self._outcome(max_scaled_error=1e-3).passed is False

    def test_summary_line_marks_failures(self):
        assert "[ok]" in self._outcome().summary_line()
        assert "[!!]" in self._outcome(max_scaled_error=1.0).summary_line()

    def test_to_dict_is_json_ready(self):
        payload = self._outcome().to_dict()
        assert payload["target"] == "inputs"
        assert payload["passed"] is True
        assert payload["compared_points"] == 24

    def test_rejects_negative_error(self):
        with pytest.raises(NumericError, match="必须是非负的数"):
            self._outcome(max_absolute_error=-1.0)

    def test_rejects_nan_error(self):
        with pytest.raises(NumericError, match="必须是非负的数"):
            self._outcome(max_scaled_error=math.nan)

    def test_rejects_bad_tolerance(self):
        with pytest.raises(ParameterError, match="正的有限数"):
            self._outcome(tolerance=0.0)

    def test_rejects_non_integer_points(self):
        with pytest.raises(NumericError, match="必须是非负整数"):
            self._outcome(compared_points=1.5)

    def test_stringifies_the_message(self):
        assert self._outcome(message=7).message == "7"


class TestGradientReport:
    """一份按名单校验的报告：它属于控制流（不通过就该停下来）."""

    def _passing(self):
        params = block_parameters()
        return check_block_gradients(params, inputs(), attention_parameters(), target())

    def test_block_report_is_closed_and_green(self):
        report = self._passing()
        assert report.kind == KIND_BLOCK
        assert report.ok is True
        assert report.failures == ()
        assert set(report.targets) == set(BLOCK_GRADIENT_TARGETS)
        assert report.worst_scaled_error <= GRADIENT_TOLERANCE
        assert report.worst_scaled_error < 1e-8
        assert len(report.formulas) == len(BLOCK_GRADIENT_TARGETS)
        assert report.to_dict()["ok"] is True
        assert "通过 9、失败 0" in report.summary_line()

    def test_raise_if_failed_is_silent_when_green(self):
        self._passing().raise_if_failed()

    def test_report_rejects_unknown_kind(self):
        with pytest.raises(ParameterError, match="未知的报告种类"):
            GradientReport(kind="magic", outcomes=(self._passing().outcomes[0],))

    def test_report_rejects_an_empty_outcome_list(self):
        with pytest.raises(ParameterError, match="梯度报告不能为空"):
            GradientReport(kind=KIND_BLOCK, outcomes=())

    def test_report_rejects_a_missing_item(self):
        report = self._passing()
        with pytest.raises(NumericError, match="报告的校验项与名单不一致"):
            GradientReport(kind=KIND_BLOCK, outcomes=report.outcomes[:-1])

    def test_report_rejects_a_bad_tolerance(self):
        report = self._passing()
        with pytest.raises(ParameterError, match="正的有限数"):
            GradientReport(kind=KIND_BLOCK, outcomes=report.outcomes, tolerance=-1.0)

    def test_decoder_report_has_no_formulas(self):
        report = _report(
            KIND_DECODER,
            {"decoder_inputs": ((1.0,),), "encoder_outputs": ((1.0,),)},
            {"decoder_inputs": ((1.0,),), "encoder_outputs": ((1.0,),)},
            tolerance=1e-6,
            step=1e-6,
            notes=("手搭",),
        )
        assert report.formulas == {}
        assert report.targets == ("decoder_inputs", "encoder_outputs")

    def test_cross_report_carries_the_source_formula(self):
        report = _report(
            KIND_CROSS,
            {name: ((0.0,),) for name in CROSS_GRADIENT_TARGETS},
            {name: ((0.0,),) for name in CROSS_GRADIENT_TARGETS},
            tolerance=1e-6,
            step=1e-6,
            notes=(),
        )
        assert "两条链之和" in report.formulas["source_inputs"]

    def test_a_wrong_gradient_turns_the_report_red(self):
        analytic = {name: ((0.0,),) for name in BLOCK_GRADIENT_TARGETS}
        numeric = {name: ((0.0,),) for name in BLOCK_GRADIENT_TARGETS}
        numeric["inputs"] = ((2.0,),)
        report = _report(
            KIND_BLOCK, analytic, numeric, tolerance=1e-6, step=1e-6, notes=()
        )
        assert report.ok is False
        assert len(report.failures) == 1
        assert report.failures[0].target == "inputs"
        assert report.failures[0].message
        with pytest.raises(GradientError, match="inputs"):
            report.raise_if_failed()


class TestVerifyHelpers:
    """三张名单之外的小工具：置换、逐点相对误差、容差与逐点对照."""

    def test_permute_rows_reorders(self):
        matrix = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        assert permute_rows(matrix, (2, 0, 1)) == ((5.0, 6.0), (1.0, 2.0), (3.0, 4.0))

    def test_permute_rows_rejects_a_non_permutation(self):
        with pytest.raises(ParameterError, match="不是 0"):
            permute_rows(((1.0,), (2.0,)), (0, 0))

    def test_reverse_permutation_is_the_least_identity_like(self):
        assert reverse_permutation(4) == (3, 2, 1, 0)

    @pytest.mark.parametrize("rows", [True, 1.5, 0])
    def test_reverse_permutation_rejects_bad_rows(self, rows):
        with pytest.raises(ParameterError, match="必须是 >= 1 的整数"):
            reverse_permutation(rows)

    def test_scaled_error_is_bounded(self):
        assert _scaled_error(1.0, 1.0) == 0.0
        assert _scaled_error(0.0, 0.5) == 0.5
        assert _scaled_error(math.inf, 1.0) == math.inf

    def test_checked_tolerance_rejects_a_string(self):
        with pytest.raises(ParameterError, match="必须是数"):
            _checked_tolerance("tiny")

    def test_compare_checks_the_shape(self):
        with pytest.raises(ShapeError, match="解析梯度"):
            _compare("inputs", ((1.0,),), ((1.0, 2.0),))

    def test_compare_counts_the_points(self):
        _absolute, _scaled, points = _compare("inputs", ((1.0, 2.0),), ((1.0, 2.0),))
        assert points == 2


class TestBlockGradientChecks:
    """九项块级校验：pre/post × 残差开/关 四组，以及三个旋钮."""

    def test_four_groups_all_pass_and_match_the_list(self):
        params = block_parameters()
        for placement in (NORM_PRE, NORM_POST):
            for use_residual in (True, False):
                report = check_block_gradients(
                    params,
                    inputs(),
                    attention_parameters(),
                    target(),
                    placement=placement,
                    use_residual=use_residual,
                )
                assert report.ok is True
                assert [item.target for item in report.outcomes] == list(BLOCK_GRADIENT_TARGETS)
                assert report.worst_scaled_error <= GRADIENT_TOLERANCE
                assert f"placement={placement}" in report.notes[0]

    def test_causal_attention_is_a_separate_switch(self):
        params = block_parameters()
        report = check_block_gradients(
            params, inputs(), attention_parameters(), target(), causal=True
        )
        assert report.ok is True
        assert "因果 开" in report.notes[0]

    def test_gelu_activation_is_checked_too(self):
        params = block_parameters()
        report = check_block_gradients(
            params, inputs(), attention_parameters(), target(), activation=ACTIVATION_GELU
        )
        assert report.ok is True
        assert ACTIVATION_GELU in report.notes[0]

    def test_notes_count_the_rows_of_the_bias_items(self):
        params = block_parameters()
        report = check_block_gradients(params, inputs(), attention_parameters(), target())
        by_target = {item.target: item for item in report.outcomes}
        assert by_target["norm1_beta"].compared_points == HIDDEN
        assert by_target["inputs"].compared_points == TOKENS * HIDDEN
        assert by_target["ffn_w_in"].compared_points == 24 * HIDDEN

    def test_numerical_gradients_are_a_block_gradients_record(self):
        params = block_parameters()
        numeric = numerical_block_gradients(params, inputs(), attention_parameters(), target())
        assert isinstance(numeric, BlockGradients)
        assert len(numeric.grad_inputs) == TOKENS

    def test_block_loss_matches_the_plain_mse(self):
        params = block_parameters()
        value = block_loss(params, inputs(), attention_parameters(), target())
        assert value > 0.0
        assert isinstance(value, float)

    def test_a_custom_tolerance_is_echoed(self):
        params = block_parameters()
        report = check_block_gradients(
            params, inputs(), attention_parameters(), target(), tolerance=1e-3
        )
        assert report.tolerance == 1e-3
        assert "0.001" in report.summary_line()


class TestCrossGradientChecks:
    """六项交叉注意力校验：四个投影 + 两路输入（source 是两条链之和）."""

    def test_six_items_all_pass(self):
        params, tgt, src, tgt_target = _cross_case()
        report = check_cross_gradients(params, tgt, src, tgt_target)
        assert report.ok is True
        assert [item.target for item in report.outcomes] == list(CROSS_GRADIENT_TARGETS)
        assert report.worst_scaled_error <= GRADIENT_TOLERANCE
        assert report.worst_scaled_error < 1e-8

    def test_source_has_more_points_than_target(self):
        params, tgt, src, tgt_target = _cross_case()
        report = check_cross_gradients(params, tgt, src, tgt_target)
        by_target = {item.target: item for item in report.outcomes}
        assert by_target["target_inputs"].compared_points == TOKENS * HIDDEN
        assert by_target["source_inputs"].compared_points == SOURCE_TOKENS * HIDDEN

    def test_cross_loss_matches_the_forward(self):
        params, tgt, src, tgt_target = _cross_case()
        value = cross_loss(params, tgt, src, tgt_target)
        assert value > 0.0
        assert isinstance(value, float)

    def test_numerical_cross_gradients_are_a_cross_gradients_record(self):
        params, tgt, src, tgt_target = _cross_case()
        numeric = numerical_cross_gradients(params, tgt, src, tgt_target)
        assert isinstance(numeric, CrossGradients)
        assert len(numeric.grad_source_inputs) == SOURCE_TOKENS

    def test_a_loose_tolerance_is_recorded(self):
        params, tgt, src, tgt_target = _cross_case()
        report = check_cross_gradients(params, tgt, src, tgt_target, tolerance=1e-2)
        assert report.tolerance == 1e-2
        assert "source" in "".join(report.notes)


class TestDecoderGradientChecks:
    """两项解码器校验：两路输入（其余八块走的是同一套 LN/FFN 反向）."""

    def _build(self):
        params = block_parameters()
        gammas = decoder_gammas()
        betas = decoder_betas()
        dec = decoder_inputs()
        enc = encoder_outputs()
        attention = self_attention(
            attention_parameters(), _normed(dec, gammas[0], betas[0]), causal=True
        )
        return params.ffn, gammas, betas, dec, enc, attention

    def test_two_inputs_are_checked_and_green(self):
        ffn, gammas, betas, dec, enc, attention = self._build()
        report = check_decoder_input_gradients(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc, decoder_target()
        )
        assert report.ok is True
        assert [item.target for item in report.outcomes] == ["decoder_inputs", "encoder_outputs"]
        assert report.worst_scaled_error <= GRADIENT_TOLERANCE
        assert report.worst_scaled_error < 1e-8

    def test_encoder_outputs_receive_a_gradient(self):
        """``dEncoder`` 不为 0：'解码器只读编码器的输出'这句话在反向里不成立."""
        ffn, gammas, betas, dec, enc, attention = self._build()
        report = check_decoder_input_gradients(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc, decoder_target()
        )
        by_target = {item.target: item for item in report.outcomes}
        assert by_target["encoder_outputs"].max_absolute_error > 0.0

    def test_notes_record_the_pre_ln_contract(self):
        ffn, gammas, betas, dec, enc, attention = self._build()
        report = check_decoder_input_gradients(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc, decoder_target()
        )
        assert "重建" in "".join(report.notes)

    def test_rejects_a_non_attention_forward(self):
        _ffn, gammas, betas, dec, enc, _attention = self._build()
        with pytest.raises(ParameterError, match="AttentionForward"):
            check_decoder_input_gradients(
                "attention",
                cross_parameters(),
                gammas,
                betas,
                block_parameters().ffn,
                dec,
                enc,
                decoder_target(),
            )

    def test_non_causal_self_attention_is_refused(self):
        _ffn, gammas, betas, dec, enc, _attention = self._build()
        loose = self_attention(attention_parameters(), dec, causal=False)
        with pytest.raises(AssemblyError, match="是因果的"):
            check_decoder_input_gradients(
                loose,
                cross_parameters(),
                gammas,
                betas,
                block_parameters().ffn,
                dec,
                enc,
                decoder_target(),
            )

    def test_target_rows_must_equal_the_decoder_rows(self):
        ffn, gammas, betas, dec, enc, attention = self._build()
        with pytest.raises(CoreShapeError, match="不一致"):
            check_decoder_input_gradients(
                attention,
                cross_parameters(),
                gammas,
                betas,
                ffn,
                dec,
                enc,
                decoder_target(rows=TOKENS),
            )

    def test_a_loose_tolerance_is_recorded(self):
        ffn, gammas, betas, dec, enc, attention = self._build()
        report = check_decoder_input_gradients(
            attention,
            cross_parameters(),
            gammas,
            betas,
            ffn,
            dec,
            enc,
            decoder_target(),
            tolerance=1e-2,
        )
        assert report.tolerance == 1e-2


def _normed(matrix, gamma, beta):
    """把一行输入标准化（解码器块的自注意力作用在 ``LN(x)`` 上）."""
    from smart_research_agent.encoder_decoder import layer_norm

    return layer_norm(matrix, gamma=gamma, beta=beta)[0]
