"""``arch_variants.stacks`` 与 ``types`` 的测试：接线、数参数、反向（day082）."""

from __future__ import annotations

from dataclasses import replace

import pytest

from smart_research_agent.arch_variants import masks, stacks
from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.arch_variants.types import (
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    BlockStackParameters,
    DecoderStackParameters,
    VariantCensus,
    VariantForward,
    VariantParameters,
    VariantShape,
    ones_vector,
    relative_matrix_error,
    validate_variant,
    zeros_vector,
)
from smart_research_agent.arch_variants.stacks import VariantGradients as StackGradients
from tests import arch_variants_samples as samples


class TestShape:
    """形状的校验与两个派生量."""

    def test_ffn_ratio_follows_the_paper(self) -> None:
        shape = stacks.make_variant_shape(tokens=4, hidden=6, layers=3)
        assert shape.ffn == 24

    def test_source_length_defaults_to_tokens(self) -> None:
        shape = stacks.make_variant_shape(tokens=4, hidden=6, layers=2)
        assert shape.source_length == 4

    def test_explicit_source_length(self) -> None:
        shape = samples.sample_shape()
        assert shape.source_length == 5

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
    def test_bad_tokens_are_rejected(self, value: object) -> None:
        with pytest.raises(ParameterError):
            VariantShape(tokens=value, hidden=6, ffn=24, layers=2)  # type: ignore[arg-type]

    def test_bad_ffn_ratio_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            stacks.make_variant_shape(tokens=4, hidden=6, layers=2, ffn_ratio=0)

    def test_summary_line_reports_every_number(self) -> None:
        line = samples.sample_shape().summary_line()
        assert "tokens=4" in line and "sources=5" in line and "layers=3" in line

    def test_hidden_dimension_matches_hidden(self) -> None:
        assert samples.sample_shape().hidden_dimension == 6


class TestVariantName:
    """变体名：**不做默认值兜底**."""

    @pytest.mark.parametrize("name", VARIANTS)
    def test_known_names_pass(self, name: str) -> None:
        assert validate_variant(name) == name

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            validate_variant("bert")

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            validate_variant(1)  # type: ignore[arg-type]


class TestParameterConstruction:
    """参数容器的校验（变体名与部件的搭配在类型这一层被挡住）."""

    def test_every_variant_builds(self) -> None:
        for variant in VARIANTS:
            params = samples.sample_parameters(variant)
            assert params.variant == variant
            assert params.layers == 3

    def test_only_encoder_decoder_has_a_decoder_stack(self) -> None:
        assert samples.sample_parameters(VARIANT_ENCODER_ONLY).decoder is None
        assert samples.sample_parameters(VARIANT_DECODER_ONLY).decoder is None
        assert samples.sample_parameters(VARIANT_ENCODER_DECODER).decoder is not None

    def test_encoder_decoder_without_decoder_is_rejected(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(AssemblyError):
            VariantParameters(
                shape=params.shape,
                variant=VARIANT_ENCODER_DECODER,
                blocks=params.blocks,
            )

    def test_single_stream_with_decoder_is_rejected(self) -> None:
        both = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(AssemblyError):
            VariantParameters(
                shape=both.shape,
                variant=VARIANT_ENCODER_ONLY,
                blocks=both.blocks,
                decoder=both.decoder,
            )

    def test_decoder_layer_mismatch_is_rejected(self) -> None:
        both = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            VariantParameters(
                shape=VariantShape(tokens=4, hidden=6, ffn=24, layers=2, sources=5),
                variant=VARIANT_ENCODER_DECODER,
                blocks=both.blocks,
                decoder=both.decoder,
            )

    def test_block_layer_mismatch_is_rejected(self) -> None:
        both = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            VariantParameters(
                shape=VariantShape(tokens=4, hidden=6, ffn=24, layers=2, sources=5),
                variant=VARIANT_ENCODER_DECODER,
                blocks=both.blocks,
                decoder=both.decoder,
            )

    def test_shape_type_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ParameterError):
            VariantParameters(shape="shape", variant=VARIANT_ENCODER_ONLY, blocks=params.blocks)  # type: ignore[arg-type]

    def test_blocks_type_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ParameterError):
            VariantParameters(shape=params.shape, variant=VARIANT_ENCODER_ONLY, blocks="blocks")  # type: ignore[arg-type]

    def test_empty_block_stack_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            BlockStackParameters(blocks=(), attentions=())

    def test_block_count_must_match_attention_count(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            BlockStackParameters(blocks=params.blocks.blocks, attentions=())

    def test_attention_type_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ParameterError):
            BlockStackParameters(blocks=params.blocks.blocks, attentions=(1, 2, 3))  # type: ignore[arg-type]

    def test_empty_decoder_stack_is_rejected(self) -> None:
        decoder = samples.sample_parameters(VARIANT_ENCODER_DECODER).decoder
        assert decoder is not None
        with pytest.raises(ParameterError):
            DecoderStackParameters(
                attentions=(),
                crosses=decoder.crosses,
                ffns=decoder.ffns,
                gammas=decoder.gammas,
                betas=decoder.betas,
            )

    def test_decoder_group_lengths_must_match(self) -> None:
        decoder = samples.sample_parameters(VARIANT_ENCODER_DECODER).decoder
        assert decoder is not None
        with pytest.raises(ShapeError):
            DecoderStackParameters(
                attentions=decoder.attentions,
                crosses=decoder.crosses[:1],
                ffns=decoder.ffns,
                gammas=decoder.gammas,
                betas=decoder.betas,
            )

    def test_decoder_needs_three_norms(self) -> None:
        decoder = samples.sample_parameters(VARIANT_ENCODER_DECODER).decoder
        assert decoder is not None
        with pytest.raises(AssemblyError):
            DecoderStackParameters(
                attentions=decoder.attentions,
                crosses=decoder.crosses,
                ffns=decoder.ffns,
                gammas=tuple(item[:2] for item in decoder.gammas),
                betas=decoder.betas,
            )

    def test_ones_and_zeros_vectors(self) -> None:
        assert ones_vector(3) == (1.0, 1.0, 1.0)
        assert zeros_vector(3) == (0.0, 0.0, 0.0)
        with pytest.raises(ParameterError):
            ones_vector(0)

    def test_cross_parameters_are_square(self) -> None:
        cross = stacks.make_cross_parameters(samples.sample_shape())
        assert cross.parameter_count == 4 * 6 * 6

    def test_sample_matrix_rejects_bad_scale(self) -> None:
        with pytest.raises(ParameterError):
            stacks.sample_matrix(2, 2, scale=0.0)

    def test_random_matrix_rejects_bad_rows(self) -> None:
        with pytest.raises(ParameterError):
            stacks.sample_matrix(0, 2)

    def test_random_matrix_rejects_bad_columns(self) -> None:
        with pytest.raises(ParameterError):
            stacks.sample_matrix(2, 0)

    def test_block_shape_of_checks_the_type(self) -> None:
        with pytest.raises(ParameterError):
            stacks.block_shape_of("shape")  # type: ignore[arg-type]


class TestForwardWiring:
    """接线：三个变体各自的前向、掩码分配与两条流."""

    def test_encoder_only_preserves_shape(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        assert len(forward.output) == 4 and len(forward.output[0]) == 6
        assert forward.encoder_output is None
        assert not forward.has_cross

    def test_decoder_only_is_causal_by_default(self) -> None:
        forward = samples.sample_forward(VARIANT_DECODER_ONLY)
        assert masks.mask_is_causal(forward.mask)

    def test_encoder_only_is_bidirectional_by_default(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        assert not masks.mask_is_causal(forward.mask)

    def test_encoder_decoder_keeps_both_streams(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        assert forward.has_cross
        assert forward.encoder_output is not None
        assert len(forward.encoder_output) == 5
        assert len(forward.cross_weights) == 3
        assert len(forward.decoder_self_weights) == 3

    def test_block_weights_shape_follows_the_stream(self) -> None:
        """``block_weights`` 是**编码器（或唯一那条）流**的权重：

        ```text
        encoder_only / decoder_only    (4, 4)  —— 目标流就是唯一那条流
        encoder_decoder                (5, 5)  —— 编码器吃的是**源序列**（5 个位置）
        ```
        """
        assert len(samples.sample_forward(VARIANT_ENCODER_ONLY).block_weights[0]) == 4
        assert len(samples.sample_forward(VARIANT_DECODER_ONLY).block_weights[0]) == 4
        assert len(samples.sample_forward(VARIANT_ENCODER_DECODER).block_weights[0]) == 5

    def test_masked_weights_are_exactly_zero(self) -> None:
        """前向层面的第 1 条性质：被掩码挡掉的位置**恰好** 0.0."""
        forward = samples.sample_forward(VARIANT_DECODER_ONLY)
        for weights in forward.block_weights:
            for row in range(4):
                for column in range(4):
                    if not forward.mask[row][column]:
                        assert weights[row][column] == 0.0

    def test_weights_rows_sum_to_one(self) -> None:
        import math

        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        for weights in (*forward.block_weights, *forward.decoder_self_weights):
            for row in weights:
                assert math.fsum(row) == pytest.approx(1.0, abs=1e-12)

    def test_cross_weights_are_rectangular(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        for weights in forward.cross_weights:
            assert len(weights) == 4
            assert len(weights[0]) == 5

    def test_mask_kind_can_be_overridden(self) -> None:
        """反证用得到的那条接口：把掩码换成另一张，形状还是对的."""
        forward = samples.sample_forward(VARIANT_DECODER_ONLY, mask_kind=masks.MASK_FULL)
        assert not masks.mask_is_causal(forward.mask)

    def test_padding_mask_for_a_single_stream(self) -> None:
        """第 1 个位置是填充：**非填充行**在那一列上的权重恰好是 0.0."""
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY, pads=(False, True, False, False))
        for weights in forward.block_weights:
            assert weights[0][1] == 0.0
            assert weights[2][1] == 0.0
            assert weights[3][1] == 0.0
            # 而那一行自己（对角线）仍然是一条合法的分布
            assert weights[1][1] > 0.0
            assert sum(weights[1]) == pytest.approx(1.0, abs=1e-12)

    def test_padding_mask_combines_with_causality(self) -> None:
        """因果 ∧ 填充：第 1 行的填充位被挡掉，而第 2 行本来就看不到它之后的第 3 行."""
        forward = samples.sample_forward(VARIANT_DECODER_ONLY, pads=(False, True, False, False))
        assert masks.mask_is_causal(forward.mask)
        assert masks.mask_allowed_counts(forward.mask) == (1, 2, 2, 3)

    def test_padding_on_a_two_stream_variant_is_rejected(self) -> None:
        """**一条被写下来的边界**：T5 的解码器读不到填充信息."""
        with pytest.raises(AssemblyError):
            samples.sample_forward(VARIANT_ENCODER_DECODER, pads=(False, True, False, False))

    def test_padding_length_must_match(self) -> None:
        with pytest.raises(ShapeError):
            samples.sample_forward(VARIANT_ENCODER_ONLY, pads=(False, True))

    def test_empty_pads_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            samples.sample_forward(VARIANT_ENCODER_ONLY, pads=())

    def test_non_bool_pads_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            samples.sample_forward(VARIANT_ENCODER_ONLY, pads=(0, 1, 0, 0))

    def test_unknown_mask_kind_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            samples.sample_forward(VARIANT_ENCODER_ONLY, mask_kind="triangular")

    def test_source_is_rejected_for_single_stream(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(AssemblyError):
            stacks.variant_forward(
                params, samples.sample_inputs(), source=samples.sample_source()
            )

    def test_source_shape_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            stacks.variant_forward(
                params, samples.sample_inputs(), source=samples.sample_source()[:3]
            )

    def test_generated_source_is_reproducible(self) -> None:
        """不给源序列时造一段确定性样本：**同一颗种子 ⇒ 逐位相同**."""
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        first = stacks.variant_forward(params, samples.sample_inputs())
        second = stacks.variant_forward(params, samples.sample_inputs())
        assert first.source == second.source
        assert len(first.source or ()) == 5

    def test_generated_source_changes_with_the_seed(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        first = stacks.variant_forward(params, samples.sample_inputs(), source_seed=13)
        second = stacks.variant_forward(params, samples.sample_inputs(), source_seed=99)
        assert first.source != second.source

    def test_params_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            stacks.variant_forward("params", samples.sample_inputs())  # type: ignore[arg-type]

    def test_input_shape_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            stacks.variant_forward(params, samples.sample_inputs()[:3])

    def test_notes_mention_both_masks(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        joined = " ".join(forward.notes)
        assert "主流" in joined and "源流" in joined and "decoder_block" in joined

    def test_block_weights_accessor(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        assert forward.weights_of(1) == forward.block_weights[1]
        with pytest.raises(ParameterError):
            forward.weights_of(3)
        with pytest.raises(ParameterError):
            forward.weights_of(True)

    def test_decoder_weights_accessor(self) -> None:
        both = samples.sample_forward(VARIANT_ENCODER_DECODER)
        assert both.decoder_weights_of(0) == both.decoder_self_weights[0]
        with pytest.raises(ParameterError):
            both.decoder_weights_of(9)
        with pytest.raises(AssemblyError):
            samples.sample_forward(VARIANT_ENCODER_ONLY).decoder_weights_of(0)

    def test_cross_weights_accessor(self) -> None:
        both = samples.sample_forward(VARIANT_ENCODER_DECODER)
        assert both.cross_weights_of(2) == both.cross_weights[2]
        with pytest.raises(ParameterError):
            both.cross_weights_of(9)
        with pytest.raises(AssemblyError):
            samples.sample_forward(VARIANT_DECODER_ONLY).cross_weights_of(0)

    def test_summary_line_of_a_forward(self) -> None:
        line = samples.sample_forward(VARIANT_ENCODER_DECODER).summary_line()
        assert "encoder_decoder" in line and "有交叉注意力" in line

    def test_forward_record_validates_shapes(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs[:3],
                output=forward.output,
                mask=forward.mask,
            )

    def test_forward_record_rejects_cross_without_encoder_output(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        with pytest.raises(NumericError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output,
                mask=forward.mask,
                block_weights=forward.block_weights,
                cross_weights=forward.cross_weights,
            )

    def test_forward_record_rejects_wrong_block_count(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output,
                mask=forward.mask,
                block_weights=forward.block_weights[:2],
            )

    def test_forward_record_rejects_a_bad_output_shape(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output[:3],
                mask=forward.mask,
                block_weights=forward.block_weights,
            )

    def test_forward_record_rejects_a_bad_source_shape(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output,
                mask=forward.mask,
                block_weights=forward.block_weights,
                decoder_self_weights=forward.decoder_self_weights,
                cross_weights=forward.cross_weights,
                source=(forward.source or ())[:2],
                encoder_output=forward.encoder_output,
            )

    def test_forward_record_rejects_a_bad_encoder_output_shape(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output,
                mask=forward.mask,
                block_weights=forward.block_weights,
                decoder_self_weights=forward.decoder_self_weights,
                cross_weights=forward.cross_weights,
                encoder_output=(forward.encoder_output or ())[:2],
            )

    def test_forward_record_rejects_a_wrong_cross_count(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output,
                mask=forward.mask,
                block_weights=forward.block_weights,
                decoder_self_weights=forward.decoder_self_weights,
                cross_weights=forward.cross_weights[:2],
                encoder_output=forward.encoder_output,
            )

    def test_forward_record_rejects_a_wrong_decoder_self_count(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            VariantForward(
                variant=forward.variant,
                shape=forward.shape,
                inputs=forward.inputs,
                output=forward.output,
                mask=forward.mask,
                block_weights=forward.block_weights,
                decoder_self_weights=forward.decoder_self_weights[:1],
                cross_weights=forward.cross_weights,
                encoder_output=forward.encoder_output,
            )

    def test_accessors_reject_non_integer_layers(self) -> None:
        both = samples.sample_forward(VARIANT_ENCODER_DECODER)
        with pytest.raises(ParameterError):
            both.decoder_weights_of("0")  # type: ignore[arg-type]
        with pytest.raises(ParameterError):
            both.cross_weights_of("0")  # type: ignore[arg-type]

    def test_frobenius_of_two_by_two(self) -> None:
        from smart_research_agent.arch_variants.types import frobenius

        assert frobenius(((3.0, 4.0), (0.0, 0.0))) == pytest.approx(5.0)

    def test_frobenius_can_be_read_from_a_forward(self) -> None:
        """行随机矩阵的 Frobenius 范数落在 ``[1, √n]``.

        ```text
        每一行都均匀（1/n）时    Σa² = n·(1/n) = 1        ⇒ √1 = 1
        每一行都独热时           Σa² = n·1²   = n        ⇒ √n = 2（本课的 n = 4）
        ```

        于是“注意力有多尖”可以被**一个标量**看到：越接近 2 越尖。
        """
        from smart_research_agent.arch_variants.types import frobenius

        forward = samples.sample_forward(VARIANT_DECODER_ONLY)
        norm = frobenius(forward.block_weights[0])
        assert 1.0 <= norm <= 2.0


class TestAttentionSeesTheNormedInput:
    """**踩到的第一个坑**：pre-LN 下注意力必须作用在 ``LN(x)`` 上.

    判据不读代码：改 ``γ₁`` 之后注意力权重会不会变。
    ``pre`` 下会变（注意力吃的是 ``LN(x)``），``post`` 下不会变（它吃的是 ``x``）。
    """

    def test_gamma1_changes_the_weights_under_pre(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        changed = replace(
            params, blocks=replace(params.blocks, blocks=(
                replace(params.blocks.blocks[0], norm1_gamma=(2.0,) * 6),
                *params.blocks.blocks[1:],
            ))
        )
        before = samples.sample_forward(VARIANT_ENCODER_ONLY)
        after = stacks.variant_forward(changed, before.inputs)
        assert after.block_weights[0] != before.block_weights[0]

    def test_gamma1_does_not_change_the_weights_under_post(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        changed = replace(
            params, blocks=replace(params.blocks, blocks=(
                replace(params.blocks.blocks[0], norm1_gamma=(2.0,) * 6),
                *params.blocks.blocks[1:],
            ))
        )
        before = stacks.variant_forward(params, samples.sample_inputs(), placement="post")
        after = stacks.variant_forward(changed, samples.sample_inputs(), placement="post")
        assert after.block_weights[0] == before.block_weights[0]


class TestLossAndBackward:
    """损失、损失梯度与整条链的反向."""

    def test_loss_is_non_negative(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        value = stacks.variant_loss(params, samples.sample_inputs(), samples.sample_target())
        assert value >= 0.0

    def test_loss_of_the_target_itself_is_zero(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        assert stacks.variant_loss(params, forward.inputs, forward.output) == 0.0

    def test_loss_gradient_is_the_residual(self) -> None:
        output = ((1.0, 2.0),)
        target = ((0.5, 3.0),)
        assert stacks.loss_gradient(output, target) == ((0.5, -1.0),)

    def test_loss_gradient_shape_is_checked(self) -> None:
        with pytest.raises(ShapeError):
            stacks.loss_gradient(((1.0,),), ((1.0, 2.0),))

    def test_loss_target_shape_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            stacks.variant_loss(params, samples.sample_inputs(), ((1.0,),))

    def test_backward_returns_the_input_gradient(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        grads = stacks.variant_backward(
            forward, params, stacks.loss_gradient(forward.output, samples.sample_target())
        )
        assert len(grads.grad_inputs) == 4
        assert grads.grad_source_inputs is None
        assert not grads.decoder_gradients

    def test_backward_records_every_layer_entry(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        grads = stacks.variant_backward(
            forward, params, stacks.loss_gradient(forward.output, samples.sample_target())
        )
        assert len(grads.grad_block_inputs) == 3
        assert grads.grad_block_inputs[2] == stacks.loss_gradient(
            forward.output, samples.sample_target()
        )

    def test_two_stream_backward_returns_both_gradients(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        grads = stacks.variant_backward(
            forward, params, stacks.loss_gradient(forward.output, samples.sample_target())
        )
        assert grads.grad_source_inputs is not None
        assert len(grads.grad_source_inputs) == 5
        assert len(grads.decoder_gradients) == 3

    def test_backward_replays_the_same_configuration(self) -> None:
        """重放要求记录里留着三个旋钮——``post`` 与 ``pre`` 的结果必须不同."""
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        target = samples.sample_target()
        pre = stacks.variant_forward(params, samples.sample_inputs(), placement="pre")
        post = stacks.variant_forward(params, samples.sample_inputs(), placement="post")
        grad_pre = stacks.variant_backward(
            pre, params, stacks.loss_gradient(pre.output, target)
        )
        grad_post = stacks.variant_backward(
            post, params, stacks.loss_gradient(post.output, target)
        )
        assert grad_pre.grad_inputs != grad_post.grad_inputs

    def test_zero_gradient_when_the_output_matches(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        grads = stacks.variant_backward(forward, params, stacks.loss_gradient(forward.output, forward.output))
        assert all(value == 0.0 for row in grads.grad_inputs for value in row)

    def test_variant_mismatch_is_rejected(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        other = samples.sample_parameters(VARIANT_DECODER_ONLY)
        with pytest.raises(AssemblyError):
            stacks.variant_backward(forward, other, forward.output)

    def test_grad_output_shape_is_checked(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            stacks.variant_backward(forward, params, ((1.0,),))

    def test_forward_type_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ParameterError):
            stacks.variant_backward("forward", params, ((1.0,),))  # type: ignore[arg-type]

    def test_params_type_is_checked_in_backward(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        with pytest.raises(ParameterError):
            stacks.variant_backward(forward, "params", forward.output)  # type: ignore[arg-type]

    def test_activation_can_be_overridden_in_backward(self) -> None:
        """激活的白名单校验发生在 **day079 那一层**，因此它抛的是那一边的 ParameterError.

        两个包的失败族是**兄弟**（都继承 day075 的三族），不是父子——
        这正是 day079 文档里那张继承图的样子。
        """
        from smart_research_agent.encoder_decoder.errors import ParameterError as EdError

        forward = samples.sample_forward(VARIANT_ENCODER_ONLY, activation="gelu")
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        grads = stacks.variant_backward(
            forward,
            params,
            stacks.loss_gradient(forward.output, samples.sample_target()),
            activation="gelu",
        )
        assert grads.grad_block_inputs
        with pytest.raises(EdError):
            stacks.variant_backward(forward, params, forward.output, activation="swish")

    def test_readings_record_validation(self) -> None:
        with pytest.raises(ParameterError):
            StackGradients(variant="bert", grad_inputs=((1.0,),))

    def test_flatten_concatenates_every_block(self) -> None:
        """压平顺序由 :data:`PARAMETER_BLOCK_ORDER` 定死：长度必须等于参数量."""
        for variant in VARIANTS:
            params = samples.sample_parameters(variant)
            flat = params.flatten()
            assert len(flat) == params.parameter_count()
            assert len(flat) == len(params.flatten())  # 两次调用逐位一致

    def test_flatten_of_two_streams_is_longer(self) -> None:
        single = samples.sample_parameters(VARIANT_ENCODER_ONLY).flatten()
        both = samples.sample_parameters(VARIANT_ENCODER_DECODER).flatten()
        assert len(both) > len(single)

    def test_block_stack_layer_mismatch_is_rejected(self) -> None:
        """单流变体上“块链层数与 shape 不一致”这条分支（与解码器那条分开）."""
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            VariantParameters(
                shape=VariantShape(tokens=4, hidden=6, ffn=24, layers=2, sources=5),
                variant=VARIANT_ENCODER_ONLY,
                blocks=params.blocks,
            )

    def test_decoder_gradients_require_a_source_gradient(self) -> None:
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        grads = stacks.variant_backward(
            forward, params, stacks.loss_gradient(forward.output, samples.sample_target())
        )
        with pytest.raises(NumericError):
            StackGradients(
                variant=grads.variant,
                grad_inputs=grads.grad_inputs,
                decoder_gradients=grads.decoder_gradients,
            )


class TestCensus:
    """家底：逐个数 vs 公式算（**两条路径必须相等**）."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_counts_match_the_formula(self, variant: str) -> None:
        assert stacks.census_of(samples.sample_parameters(variant)).matches_analytic

    def test_encoder_only_carries_two_sublayers_per_block(self) -> None:
        census = stacks.census_of(samples.sample_parameters(VARIANT_ENCODER_ONLY))
        assert census.sub_layers == 6
        assert census.layers_per_block == 2.0
        assert census.cross_attention_layers == 0

    def test_encoder_decoder_carries_three_sublayers_more(self) -> None:
        census = stacks.census_of(samples.sample_parameters(VARIANT_ENCODER_DECODER))
        assert census.sub_layers == 15
        assert census.cross_attention_layers == 3
        assert census.self_attention_layers == 6

    def test_encoder_and_decoder_only_have_the_same_parameter_count(self) -> None:
        """**因果与否不改变参数量**（只改变能看到什么）——这是一条值钱的对照."""
        first = stacks.census_of(samples.sample_parameters(VARIANT_ENCODER_ONLY))
        second = stacks.census_of(samples.sample_parameters(VARIANT_DECODER_ONLY))
        assert first.parameters == second.parameters

    def test_census_scales_with_the_shape(self) -> None:
        small = stacks.census_of(samples.sample_parameters(VARIANT_ENCODER_ONLY))
        big_shape = samples.sample_shape(layers=6)
        big = stacks.census_of(samples.sample_parameters(VARIANT_ENCODER_ONLY, big_shape))
        assert big.parameters == 2 * small.parameters

    def test_census_of_checks_the_type(self) -> None:
        with pytest.raises(ParameterError):
            stacks.census_of("params")  # type: ignore[arg-type]

    def test_census_record_validation(self) -> None:
        with pytest.raises(ParameterError):
            VariantCensus(
                variant="bert",
                layers=1,
                block_layers=1,
                decoder_layers=0,
                self_attention_layers=1,
                cross_attention_layers=0,
                sub_layers=2,
                parameters=10,
                analytic_parameters=10,
            )

    def test_census_requires_positive_parameters(self) -> None:
        with pytest.raises(NumericError):
            VariantCensus(
                variant=VARIANT_ENCODER_ONLY,
                layers=1,
                block_layers=1,
                decoder_layers=0,
                self_attention_layers=1,
                cross_attention_layers=0,
                sub_layers=2,
                parameters=0,
                analytic_parameters=10,
            )

    def test_census_requires_at_least_two_sublayers_per_layer(self) -> None:
        with pytest.raises(NumericError):
            VariantCensus(
                variant=VARIANT_ENCODER_ONLY,
                layers=2,
                block_layers=2,
                decoder_layers=0,
                self_attention_layers=2,
                cross_attention_layers=0,
                sub_layers=1,
                parameters=10,
                analytic_parameters=10,
            )

    def test_census_to_dict(self) -> None:
        payload = stacks.census_of(samples.sample_parameters(VARIANT_ENCODER_ONLY)).to_dict()
        assert payload["variant"] == VARIANT_ENCODER_ONLY
        assert payload["matches_analytic"] is True


class TestMaskHelpers:
    """掩码分配的辅助读数."""

    def test_visible_positions_per_variant(self) -> None:
        assert stacks.visible_positions(
            samples.sample_parameters(VARIANT_ENCODER_ONLY)
        ) == (4, 4, 4, 4)
        assert stacks.visible_positions(
            samples.sample_parameters(VARIANT_DECODER_ONLY)
        ) == (1, 2, 3, 4)

    def test_stream_causality_per_variant(self) -> None:
        assert not stacks.stream_is_causal(samples.sample_parameters(VARIANT_ENCODER_ONLY))
        assert stacks.stream_is_causal(samples.sample_parameters(VARIANT_DECODER_ONLY))
        assert stacks.stream_is_causal(
            samples.sample_parameters(VARIANT_ENCODER_DECODER)
        )

    def test_encoder_stream_causality(self) -> None:
        assert not stacks.encoder_stream_is_causal(
            samples.sample_parameters(VARIANT_ENCODER_DECODER)
        )
        assert not stacks.encoder_stream_is_causal(
            samples.sample_parameters(VARIANT_ENCODER_ONLY)
        )
        assert stacks.encoder_stream_is_causal(
            samples.sample_parameters(VARIANT_DECODER_ONLY)
        )

    def test_mask_for_streams_shape(self) -> None:
        shape = samples.sample_shape()
        main, source = stacks.mask_for_streams(VARIANT_ENCODER_DECODER, shape)
        assert masks.shape_of_mask(main) == (4, 4)
        assert source is not None and masks.shape_of_mask(source) == (5, 5)

    def test_check_mask_fits(self) -> None:
        mask = masks.mask_of(masks.MASK_FULL, 4)
        assert stacks.check_mask_fits(mask, 4) == mask
        with pytest.raises(ShapeError):
            stacks.check_mask_fits(mask, 5)

    def test_params_mask_of(self) -> None:
        assert masks.mask_is_causal(
            stacks.params_mask_of(samples.sample_parameters(VARIANT_DECODER_ONLY))
        )


class TestReadings:
    """两个读数工具（相对误差与最大绝对值）."""

    def test_relative_error_of_identical_matrices_is_zero(self) -> None:
        matrix = ((1.0, 2.0), (3.0, 4.0))
        assert relative_matrix_error(matrix, matrix) == 0.0

    def test_relative_error_uses_a_floor_of_one(self) -> None:
        """分母带 ``max(1, ·)``：两边都接近 0 时纯相对误差会放大到无意义的量级."""
        assert relative_matrix_error(((0.0, 0.0),), ((0.0, 0.0),)) == 0.0
        assert relative_matrix_error(((0.5, 0.0),), ((0.0, 0.0),)) == pytest.approx(0.5)

    def test_relative_error_checks_shapes(self) -> None:
        with pytest.raises(ShapeError):
            relative_matrix_error(((1.0,),), ((1.0, 2.0),))

    def test_max_absolute_of_an_empty_matrix(self) -> None:
        from smart_research_agent.arch_variants.types import max_absolute

        assert max_absolute(((0.0,),)) == 0.0
