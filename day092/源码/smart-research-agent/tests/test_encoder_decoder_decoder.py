"""交叉注意力与解码器块：三角形权重、双路输入与三项组装层面的拒绝（day079 / M7-D4）.

这一份测试守三件事：

```text
① 交叉注意力的权重是 (n_tgt, n_src) 的**长方形**    ——不是方阵，因此没有掩码
② 解码器块比编码器块多一个子层                      ——九个阶段，三段残差，三个 LN
③ 三处组装层面的拒绝都是 AssemblyError             ——自注意力必须因果、两路同宽、三个 LN
```

第 ① 条是这一课最值钱的一条：`causal=True` 在交叉注意力上**没有意义**，
而"给它加一份方阵掩码"这件事只在 `n_tgt == n_src` 时才不报错——因此本包把它做成显式拒绝。
"""

from __future__ import annotations

import pytest

from smart_research_agent.encoder_decoder import (
    ACTIVATION_GELU,
    DECODER_BLOCK_STAGES,
    DECODER_STAGE_DESCRIPTIONS,
    DECODER_STAGE_SHAPES,
    AssemblyError,
    CrossParameters,
    ParameterError,
    check_decoder_input_gradients,
    cross_attention,
    cross_attention_backward,
    decoder_block,
    decoder_block_backward,
    layer_norm,
)
from smart_research_agent.encoder_decoder.errors import ShapeError
from smart_research_agent.transformer_core.errors import ShapeError as CoreShapeError
from smart_research_agent.transformer_core.layers import mse_gradient, self_attention
from tests.encoder_decoder_samples import (
    DECODER_TOKENS,
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
)


def _zeros(rows: int, columns: int) -> tuple:
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def _decode_case():
    """一套自洽的解码器用例：自注意力作用在 ``LN(decoder_inputs)`` 上（pre-LN 契约）."""
    params = block_parameters()
    gammas = decoder_gammas()
    betas = decoder_betas()
    dec = decoder_inputs()
    enc = encoder_outputs()
    attention = self_attention(
        attention_parameters(), layer_norm(dec, gamma=gammas[0], beta=betas[0])[0], causal=True
    )
    return params.ffn, gammas, betas, dec, enc, attention


class TestDecoderStages:
    """九个阶段的两张表必须逐键对齐（少一条不会让测试变红，只会让报告缺一节）."""

    def test_nine_stages_in_data_order(self):
        assert DECODER_BLOCK_STAGES == (
            "norm1",
            "self_branch",
            "add1",
            "norm2",
            "cross_branch",
            "add2",
            "norm3",
            "ffn_branch",
            "add3",
        )
        assert len(DECODER_BLOCK_STAGES) == 9

    def test_every_stage_has_a_description_and_a_shape(self):
        assert set(DECODER_BLOCK_STAGES) == set(DECODER_STAGE_DESCRIPTIONS)
        assert set(DECODER_BLOCK_STAGES) == set(DECODER_STAGE_SHAPES)
        for stage in DECODER_BLOCK_STAGES:
            assert DECODER_STAGE_DESCRIPTIONS[stage]
            assert DECODER_STAGE_SHAPES[stage]

    def test_cross_stage_says_the_weight_is_rectangular(self):
        assert "权重是 (n_tgt, n_src)" in DECODER_STAGE_SHAPES["cross_branch"]

    def test_self_branch_is_the_only_causal_one(self):
        assert "因果掩码" in DECODER_STAGE_DESCRIPTIONS["self_branch"]
        assert "绝不能加因果掩码" in DECODER_STAGE_DESCRIPTIONS["cross_branch"]


class TestCrossAttention:
    """交叉注意力：两路、长方形权重、没有掩码."""

    def test_weights_are_rectangular_when_the_lengths_differ(self):
        forward = cross_attention(cross_parameters(), inputs(), source_inputs())
        assert len(inputs()) != len(source_inputs())
        assert len(forward.weights) == TOKENS
        assert len(forward.weights[0]) == SOURCE_TOKENS
        assert all(abs(value - 1.0) < 1e-12 for value in forward.row_sums)
        assert all(value >= 0.0 for row in forward.weights for value in row)
        assert any("长方形" in note for note in forward.notes)
        assert any("没有掩码" in note for note in forward.notes)

    def test_rejects_a_non_cross_parameters(self):
        with pytest.raises(ParameterError, match="CrossParameters"):
            cross_attention("params", inputs(), source_inputs())

    def test_rejects_a_causal_mask(self):
        with pytest.raises(AssemblyError, match="交叉注意力不能被赋予因果掩码"):
            cross_attention(cross_parameters(), inputs(), source_inputs(), causal=True)

    def test_rejects_a_wrong_query_width(self):
        params = CrossParameters(
            w_query=_zeros(HIDDEN, 5),
            w_key=_zeros(HIDDEN, HIDDEN),
            w_value=_zeros(HIDDEN, HIDDEN),
            w_output=_zeros(HIDDEN, HIDDEN),
        )
        with pytest.raises(ShapeError, match="W_q 的列数"):
            cross_attention(params, inputs(), source_inputs())

    def test_rejects_a_wrong_key_width(self):
        params = CrossParameters(
            w_query=_zeros(HIDDEN, HIDDEN),
            w_key=_zeros(HIDDEN, 5),
            w_value=_zeros(HIDDEN, 5),
            w_output=_zeros(HIDDEN, HIDDEN),
        )
        with pytest.raises(ShapeError, match="W_k 的列数"):
            cross_attention(params, inputs(), source_inputs())

    def test_backward_checks_the_grad_shape(self):
        forward = cross_attention(cross_parameters(), inputs(), source_inputs())
        with pytest.raises(ShapeError, match="与输出"):
            cross_attention_backward(forward, ((1.0,),))

    def test_backward_gives_both_inputs(self):
        forward = cross_attention(cross_parameters(), inputs(), source_inputs())
        grads = cross_attention_backward(
            forward, mse_gradient(forward.output, tuple(tuple(0.5 for _ in range(HIDDEN)) for _ in range(TOKENS)))
        )
        assert len(grads.grad_target_inputs) == TOKENS
        assert len(grads.grad_source_inputs) == SOURCE_TOKENS


class TestDecoderBlockForward:
    """解码器块前向：九个阶段、三个 LN、两路输入."""

    def test_forward_keeps_the_decoder_shape(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        output, cross_forward, caches, ffn_cache = decoder_block(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc
        )
        assert len(output) == DECODER_TOKENS
        assert len(output[0]) == HIDDEN
        assert len(caches) == 3
        assert len(cross_forward.weights) == DECODER_TOKENS
        assert len(cross_forward.weights[0]) == SOURCE_TOKENS
        assert len(ffn_cache.inputs) == DECODER_TOKENS

    def test_cross_attention_reads_the_encoder_outputs(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        _output, cross_forward, _caches, _ffn = decoder_block(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc
        )
        assert cross_forward.source_inputs == enc

    def test_rejects_a_non_attention_forward(self):
        ffn, gammas, betas, dec, enc, _attention = _decode_case()
        with pytest.raises(ParameterError, match="AttentionForward"):
            decoder_block(
                "attention", cross_parameters(), gammas, betas, ffn, dec, enc
            )

    def test_rejects_a_non_causal_self_attention(self):
        ffn, gammas, betas, dec, enc, _attention = _decode_case()
        loose = self_attention(attention_parameters(), dec, causal=False)
        with pytest.raises(AssemblyError, match="是因果的"):
            decoder_block(loose, cross_parameters(), gammas, betas, ffn, dec, enc)

    def test_rejects_a_wrong_number_of_norms(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        with pytest.raises(AssemblyError, match="三个 LN"):
            decoder_block(
                attention, cross_parameters(), gammas[:2], betas, ffn, dec, enc
            )

    def test_rejects_a_width_mismatch_between_the_two_paths(self):
        ffn, gammas, betas, dec, _enc, attention = _decode_case()
        narrow = tuple(tuple(0.3 for _ in range(HIDDEN - 1)) for _ in range(SOURCE_TOKENS))
        with pytest.raises(AssemblyError, match="编码器输出的列数"):
            decoder_block(attention, cross_parameters(), gammas, betas, ffn, dec, narrow)

    def test_without_residual_runs(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        output, _cross, _caches, _ffn = decoder_block(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc, use_residual=False
        )
        assert len(output) == DECODER_TOKENS

    def test_gelu_activation_runs(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        _output, _cross, _caches, ffn_cache = decoder_block(
            attention,
            cross_parameters(),
            gammas,
            betas,
            ffn,
            dec,
            enc,
            activation=ACTIVATION_GELU,
        )
        assert ffn_cache.activation == ACTIVATION_GELU


class TestDecoderBlockBackward:
    """解码器块反向：三段残差、两路输入、三个 LN 的 γ/β."""

    def _backward(self, *, use_residual=True):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        output, cross_forward, caches, ffn_cache = decoder_block(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc, use_residual=use_residual
        )
        grads = decoder_block_backward(
            attention,
            cross_forward,
            caches,
            ffn_cache,
            mse_gradient(output, decoder_target()),
            use_residual=use_residual,
        )
        return grads

    def test_encoder_outputs_receive_a_gradient(self):
        grads = self._backward()
        worst = max(abs(value) for row in grads.grad_encoder_outputs for value in row)
        assert worst > 0.0
        assert len(grads.grad_encoder_outputs) == SOURCE_TOKENS

    def test_three_norms_produce_gammas_and_betas(self):
        grads = self._backward()
        assert len(grads.grad_norm_gammas) == 3
        assert len(grads.grad_norm_betas) == 3
        assert all(len(vector) == HIDDEN for vector in grads.grad_norm_gammas)

    def test_decoder_input_gradient_has_the_decoder_shape(self):
        grads = self._backward()
        assert len(grads.grad_decoder_inputs) == DECODER_TOKENS
        assert len(grads.grad_decoder_inputs[0]) == HIDDEN

    def test_without_residual_still_traces_the_ln_chains(self):
        grads = self._backward(use_residual=False)
        assert len(grads.grad_decoder_inputs) == DECODER_TOKENS

    def test_rejects_a_wrong_number_of_caches(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        output, cross_forward, caches, ffn_cache = decoder_block(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc
        )
        with pytest.raises(AssemblyError, match="三个 LN 的账"):
            decoder_block_backward(
                attention,
                cross_forward,
                caches[:2],
                ffn_cache,
                mse_gradient(output, decoder_target()),
            )

    def test_rejects_a_wrong_grad_shape(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        output, cross_forward, caches, ffn_cache = decoder_block(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc
        )
        with pytest.raises(ShapeError, match="与解码器输入"):
            decoder_block_backward(
                attention, cross_forward, caches, ffn_cache, ((1.0,),)
            )


class TestDecoderGradientCheck:
    """两路输入梯度校验：解析侧把自注意力按 pre-LN 契约**重建**在 LN(x) 上."""

    def test_two_inputs_are_green(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
        report = check_decoder_input_gradients(
            attention, cross_parameters(), gammas, betas, ffn, dec, enc, decoder_target()
        )
        assert report.ok is True
        assert report.worst_scaled_error < 1e-8
        by_target = {item.target: item for item in report.outcomes}
        assert by_target["decoder_inputs"].passed is True
        assert by_target["encoder_outputs"].passed is True

    def test_an_attention_built_on_the_raw_input_is_still_rebuilt(self):
        """检查只用到传入 forward 的 ``params``——它会把注意力重建在 ``LN(x)`` 上."""
        ffn, gammas, betas, dec, enc, _attention = _decode_case()
        on_raw = self_attention(attention_parameters(), dec, causal=True)
        report = check_decoder_input_gradients(
            on_raw, cross_parameters(), gammas, betas, ffn, dec, enc, decoder_target()
        )
        assert report.ok is True

    def test_rejects_a_non_attention_forward(self):
        ffn, gammas, betas, dec, enc, _attention = _decode_case()
        with pytest.raises(ParameterError, match="AttentionForward"):
            check_decoder_input_gradients(
                "attention",
                cross_parameters(),
                gammas,
                betas,
                ffn,
                dec,
                enc,
                decoder_target(),
            )

    def test_rejects_a_non_causal_forward(self):
        ffn, gammas, betas, dec, enc, _attention = _decode_case()
        loose = self_attention(attention_parameters(), dec, causal=False)
        with pytest.raises(AssemblyError, match="是因果的"):
            check_decoder_input_gradients(
                loose, cross_parameters(), gammas, betas, ffn, dec, enc, decoder_target()
            )

    def test_target_rows_must_match_the_decoder_rows(self):
        ffn, gammas, betas, dec, enc, attention = _decode_case()
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
