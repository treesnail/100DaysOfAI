"""``transformer_stack.verify`` 的两处梯度校验（day080）."""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.calculus import DEFAULT_STEP
from smart_research_agent.transformer_stack import (
    GRADIENT_TOLERANCE,
    GRAD_STACK_INPUTS,
    KIND_LAYER,
    KIND_STACK,
    LAYER_GRADIENT_TARGETS,
    NumericError,
    ParameterError,
    check_layer_parameter_gradients,
    check_stack_input_gradient,
    gradient_summary,
    layer_norm_sequence,
    numerical_layer_parameter_gradients,
    numerical_stack_input_gradient,
    stack_loss_gradient,
)
from smart_research_agent.transformer_stack.verify import (
    GradientOutcome,
    GradientReport,
)
from tests.stack_samples import HIDDEN, LAYERS, TOKENS, inputs, parameters, shape_of, target


class TestNumericalStackInputGradient:
    """整条链的输入梯度（数值侧）."""

    def test_it_returns_a_matrix_of_the_input_shape(self):
        """数值梯度的形状与输入一致（``∇f`` 与 ``x`` 同形）."""
        numeric = numerical_stack_input_gradient(parameters(), inputs(), target())
        assert shape_of(numeric) == (TOKENS, HIDDEN)

    def test_it_is_finite_and_non_zero(self):
        """样本上的数值梯度不是零矩阵（否则“对上了”这句话没有意义）."""
        numeric = numerical_stack_input_gradient(parameters(), inputs(), target())
        assert all(math.isfinite(value) for row in numeric for value in row)
        assert any(abs(value) > 0.0 for row in numeric for value in row)

    def test_it_does_not_depend_on_the_chain_length_by_accident(self):
        """换一个深度会得到不同的输入梯度（链真的被穿过了）."""
        shallow = numerical_stack_input_gradient(parameters(layers=1), inputs(), target())
        deep = numerical_stack_input_gradient(parameters(layers=3), inputs(), target())
        assert shallow != deep

    def test_a_smaller_step_gives_a_similar_answer(self):
        """把差分步长从 ``1e-6`` 换到 ``1e-5``，结果仍在 ``1e-4`` 量级以内."""
        coarse = numerical_stack_input_gradient(parameters(), inputs(), target())
        finer = numerical_stack_input_gradient(
            parameters(), inputs(), target(), step=DEFAULT_STEP * 10
        )
        worst = max(
            abs(left - right)
            for row_left, row_right in zip(coarse, finer, strict=True)
            for left, right in zip(row_left, row_right, strict=True)
        )
        assert worst < 1e-4


class TestCheckStackInputGradient:
    """``stack`` 那一份报告：一项，解析 vs 数值."""

    def test_the_single_outcome_passes(self):
        """样本上这一项通过，且误差在 ``1e-9`` 量级以内（容差是 ``1e-6``）."""
        report = check_stack_input_gradient(parameters(), inputs(), target())
        assert report.kind == KIND_STACK
        assert report.ok
        assert not report.failures
        assert report.worst_scaled_error < 1e-8
        assert report.passed_targets == (GRAD_STACK_INPUTS,)

    def test_the_outcome_records_the_sample_points(self):
        """读数里带着“比较了多少个逐点”：``4 × 6 = 24``."""
        outcome = check_stack_input_gradient(parameters(), inputs(), target()).outcomes[0]
        assert outcome.target == GRAD_STACK_INPUTS
        assert outcome.compared_points == TOKENS * HIDDEN
        assert outcome.tolerance == GRADIENT_TOLERANCE
        assert outcome.message == ""

    @pytest.mark.parametrize("placement", ["pre", "post"])
    def test_both_placements_pass(self, placement: str):
        """两种摆放位置都要过（口径必须两边一致）."""
        report = check_stack_input_gradient(
            parameters(), inputs(), target(), placement=placement
        )
        assert report.ok

    def test_without_residual_it_still_passes(self):
        """把残差关掉之后梯度链更短，但校验仍然要过."""
        report = check_stack_input_gradient(
            parameters(), inputs(), target(), use_residual=False
        )
        assert report.ok

    def test_a_tight_tolerance_still_passes_at_the_resolution_limit(self):
        """把容差压到 ``1e-9``：样本上的解析—数值差仍然过得去."""
        report = check_stack_input_gradient(
            parameters(), inputs(), target(), tolerance=1e-9
        )
        assert report.worst_scaled_error <= 1e-9

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, "1e-6"])
    def test_tolerance_must_be_positive(self, tolerance: object):
        """容差必须是正的有限数（``0`` 意味着“必须逐位相等”）."""
        with pytest.raises(ParameterError):
            check_stack_input_gradient(
                parameters(), inputs(), target(), tolerance=tolerance  # type: ignore[arg-type]
            )

    def test_summary_line_and_detail_lines(self):
        """两种输出的形状（演示脚本按节打印它们）."""
        report = check_stack_input_gradient(parameters(), inputs(), target())
        assert "通过 1、失败 0" in report.summary_line()
        assert len(report.detail_lines()) == 1
        assert GRAD_STACK_INPUTS in report.detail_lines()[0]
        assert gradient_summary(report) == report.summary_line()

    def test_to_dict_is_json_serialisable(self):
        """报告能 json.dumps（它要被写进演示输出与文档）."""
        payload = check_stack_input_gradient(parameters(), inputs(), target()).to_dict()
        assert payload["kind"] == KIND_STACK
        assert payload["ok"] is True
        assert isinstance(json.dumps(payload), str)


class TestCheckLayerParameterGradients:
    """``layer`` 那一份报告：八项，第 k 层的参数梯度."""

    def test_eight_outcomes_and_they_all_pass(self):
        """八块参数（不含 ``inputs``）逐项对照，样本上全过."""
        report = check_layer_parameter_gradients(parameters(), inputs(), target(), 0)
        assert report.kind == KIND_LAYER
        assert len(report.outcomes) == 8
        assert report.passed_targets == LAYER_GRADIENT_TARGETS
        assert report.ok

    def test_it_covers_every_middle_layer(self):
        """第 0、1、2、3 层各查一遍都要过（链上每一环都接到正确的上游）."""
        for index in range(LAYERS):
            report = check_layer_parameter_gradients(
                parameters(), inputs(), target(), index
            )
            assert report.ok, index

    def test_a_middle_layer_is_more_expensive_to_be_right_about(self):
        """第 0 与第 3 层都能过——**这两份报告并不相同**（链在起作用）."""
        bottom = check_layer_parameter_gradients(parameters(), inputs(), target(), 0)
        top = check_layer_parameter_gradients(parameters(), inputs(), target(), 3)
        assert bottom.ok and top.ok
        assert bottom.outcomes[0].max_scaled_error != top.outcomes[0].max_scaled_error

    def test_the_wrong_chaining_would_be_caught_by_this_check(self):
        """**反证**：把每一层都从 ``dLoss/dy_N`` 起步的那种错法会被这一项抓住.

        做法是把错误的那一份解析梯度也算出来，然后与数值差分比：
        正确的那一份对得上（``< 1e-6``），错法那一份对不上（``> 1e-3``）——
        而两者的**形状完全一样**。
        """
        from smart_research_agent.encoder_decoder.layers import encoder_block_backward
        from smart_research_agent.transformer_core.layers import mse_gradient
        from smart_research_agent.transformer_stack import (
            relative_matrix_error,
            stack_forward,
        )

        params = parameters()
        forward, grads = stack_loss_gradient(params, inputs(), target())
        top = mse_gradient(forward.output, target())
        layer0 = forward.layer_at(0)
        wrong = encoder_block_backward(layer0.block, layer0.params, top)
        numeric = numerical_layer_parameter_gradients(params, inputs(), target(), 0)
        correct = grads.layer_at(0)
        assert relative_matrix_error(numeric[2], correct.grad_ffn_w_in) < 1e-6
        assert relative_matrix_error(numeric[2], wrong.grad_ffn_w_in) > 1e-3

    def test_numerical_layer_gradients_have_the_eight_matrices(self):
        """数值侧返回八块，顺序与名单一致；**向量块是 (1, n) 的矩阵**.

        八块里有两块原本是**向量**（γ / β 与两个偏置），而 ``unflatten_matrices``
        只会给出矩阵——于是它们以 ``(1, n)`` 的形式出现，``BlockParameters.unflatten``
        再取第一行还原成向量。这个约定必须在测试里写下来，否则“少了一维”看起来像 bug。
        """
        parts = numerical_layer_parameter_gradients(parameters(), inputs(), target(), 1)
        assert len(parts) == 8
        assert shape_of(parts[0]) == (1, HIDDEN)  # norm1_gamma（向量，以 (1, d) 出现）
        assert shape_of(parts[2]) == (24, HIDDEN)  # ffn_w_in
        assert shape_of(parts[4]) == (HIDDEN, 24)  # ffn_w_out
        assert shape_of(parts[7]) == (1, HIDDEN)  # norm2_beta

    @pytest.mark.parametrize("index", [-1, LAYERS])
    def test_layer_index_is_checked(self, index: int):
        """层号越界当场报错."""
        with pytest.raises(ParameterError):
            check_layer_parameter_gradients(parameters(), inputs(), target(), index)

    def test_notes_explain_what_is_not_in_the_list(self):
        """报告的说明里必须写明“为什么名单里没有 inputs”."""
        report = check_layer_parameter_gradients(parameters(), inputs(), target(), 2)
        assert any("inputs" in item for item in report.notes)
        assert any("第 2 层" in item for item in report.notes)


class TestGradientOutcomeAndReport:
    """两个报告类的构造检查（缺项报告在“全绿”时看起来一样）."""

    def test_outcome_passed_uses_the_scaled_error(self):
        """``passed`` 判的是缩放误差，不是绝对误差."""
        outcome = GradientOutcome(
            target=GRAD_STACK_INPUTS,
            max_absolute_error=1.0,
            max_scaled_error=1e-9,
            tolerance=1e-6,
            compared_points=24,
        )
        assert outcome.passed
        assert "[ok]" in outcome.summary_line()

    def test_outcome_failure_line_is_marked(self):
        """没过的那一项在行首是 ``!!``（一眼能看出来）."""
        outcome = GradientOutcome(
            target=GRAD_STACK_INPUTS,
            max_absolute_error=1.0,
            max_scaled_error=1.0,
            tolerance=1e-6,
            compared_points=24,
        )
        assert not outcome.passed
        assert "[!!]" in outcome.summary_line()

    @pytest.mark.parametrize("field", ["max_absolute_error", "max_scaled_error"])
    def test_outcome_rejects_nan(self, field: str):
        """``nan`` 不能进读数（它会静默污染整张表）."""
        payload = {
            "target": GRAD_STACK_INPUTS,
            "max_absolute_error": 0.0,
            "max_scaled_error": 0.0,
            "tolerance": 1e-6,
            "compared_points": 1,
        }
        payload[field] = float("nan")
        with pytest.raises(NumericError):
            GradientOutcome(**payload)  # type: ignore[arg-type]

    def test_outcome_rejects_a_negative_point_count(self):
        """比较点数不能是负数."""
        with pytest.raises(NumericError):
            GradientOutcome(
                target=GRAD_STACK_INPUTS,
                max_absolute_error=0.0,
                max_scaled_error=0.0,
                tolerance=1e-6,
                compared_points=-1,
            )

    def test_report_rejects_a_missing_target(self):
        """**缺项的报告**在'全绿'时看起来与完整的一样——因此逐项对名单."""
        with pytest.raises(NumericError):
            GradientReport(kind=KIND_LAYER, outcomes=())

    def test_report_rejects_an_unknown_kind(self):
        """不认识的报告种类当场报错."""
        with pytest.raises(ParameterError):
            GradientReport(kind="everything", outcomes=())

    def test_report_rejects_a_mismatched_tolerance(self):
        """第一项的容差与报告的不一致时当场报错（口径只能有一个）."""
        outcome = GradientOutcome(
            target=GRAD_STACK_INPUTS,
            max_absolute_error=0.0,
            max_scaled_error=0.0,
            tolerance=1e-9,
            compared_points=1,
        )
        with pytest.raises(NumericError):
            GradientReport(kind=KIND_STACK, outcomes=(outcome,), tolerance=1e-6)

    def test_report_failures_and_worst(self):
        """``failures`` / ``worst_scaled_error`` 在混色报告上的读数."""
        good = GradientOutcome(
            target=LAYER_GRADIENT_TARGETS[0],
            max_absolute_error=0.0,
            max_scaled_error=1e-9,
            tolerance=1e-6,
            compared_points=6,
        )
        bad = GradientOutcome(
            target=LAYER_GRADIENT_TARGETS[1],
            max_absolute_error=1.0,
            max_scaled_error=1.0,
            tolerance=1e-6,
            compared_points=6,
            message="差太多",
        )
        rest = tuple(
            GradientOutcome(
                target=name,
                max_absolute_error=0.0,
                max_scaled_error=0.0,
                tolerance=1e-6,
                compared_points=6,
            )
            for name in LAYER_GRADIENT_TARGETS[2:]
        )
        report = GradientReport(kind=KIND_LAYER, outcomes=(good, bad, *rest))
        assert not report.ok
        assert report.failures == (bad,)
        assert report.worst_scaled_error == 1.0
        assert "通过 7、失败 1" in report.summary_line()
        assert len(report.detail_lines()) == 8

    def test_the_real_layer_report_is_serialisable(self):
        """真实的八项报告能 json.dumps."""
        payload = check_layer_parameter_gradients(parameters(), inputs(), target(), 0).to_dict()
        assert len(payload["outcomes"]) == 8
        assert isinstance(json.dumps(payload), str)


class TestLayerNormSequence:
    """一个小转发：逐层梯度范数序列."""

    def test_it_matches_the_gradient_record(self):
        """``layer_norm_sequence`` 与 ``StackGradients.norms`` 一致."""
        _forward, grads = stack_loss_gradient(parameters(), inputs(), target())
        assert layer_norm_sequence(grads) == grads.norms()
        assert len(layer_norm_sequence(grads)) == LAYERS

    def test_it_rejects_a_non_gradient_record(self):
        """传进来不是 ``StackGradients`` 时当场报错."""
        with pytest.raises(ParameterError):
            layer_norm_sequence("not-a-record")  # type: ignore[arg-type]
