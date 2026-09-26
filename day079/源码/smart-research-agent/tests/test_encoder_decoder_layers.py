"""形状契约、六类记录、三个子层（LN / 前馈 / 残差）与块的组装（day079 / M7-D4）.

这一份测试守三件事：

```text
① 六张记录的形状契约      每一个 dataclass 的 __post_init__ 都在入口处拒绝坏输入
② 三个子层的逐行性        LN 与前馈用 **== ** 断言（逐位），残差那条 +1 路用容差
③ 块的组装与摆放位置      pre 时注意力作用在 LN(x) 上、post 时作用在 x 上（block_attention）
```

第 ② 条里的"逐位"是刻意的：逐行算子不产生任何求和顺序的变化，
因此"换行序 → 输出逐位跟着换"这条断言比"误差很小"难伪造得多。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.encoder_decoder import (
    ACTIVATION_GELU,
    ACTIVATION_RELU,
    ENCODER_BLOCK_STAGES,
    NORM_POST,
    NORM_PRE,
    BlockForward,
    BlockGradients,
    BlockParameters,
    BlockShape,
    CrossForward,
    CrossGradients,
    CrossParameters,
    CrossShape,
    DecoderGradients,
    FFNCache,
    FFNGradients,
    FFNWeights,
    NormCache,
    NormGradients,
    activate,
    activation_backward,
    add_matrices,
    add_residual,
    block_attention,
    cross_attention,
    cross_attention_backward,
    decoder_block,
    decoder_block_backward,
    encoder_block,
    encoder_block_backward,
    feed_forward,
    feed_forward_backward,
    layer_norm,
    layer_norm_backward,
    matrix_max_absolute,
    relative_matrix_error,
    stage_order,
    validate_epsilon,
    zero_matrix,
)
from smart_research_agent.encoder_decoder.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    EncoderDecoderError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.encoder_decoder.types import (
    _checked_activation,
    _checked_placement,
    _checked_positive_int,
    _checked_tolerance,
)
from smart_research_agent.transformer_core.errors import ShapeError as CoreShapeError
from smart_research_agent.transformer_core.layers import self_attention
from smart_research_agent.transformer_core.types import AttentionParams
from tests.encoder_decoder_samples import (
    HIDDEN,
    TOKENS,
    approx,
    approx_matrix,
    attention_parameters,
    block_parameters,
    block_shape,
    inputs,
    target,
)


def _zeros(rows: int, columns: int) -> tuple:
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def _pre_forward():
    """一个 pre-LN 的前向账（供记录类的用例复用）."""
    params = block_parameters()
    attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
    return params, attention, encoder_block(params, inputs(), attention, placement=NORM_PRE)


class TestFamilies:
    """五族失败：分族的依据是"谁的错、该谁去修"，不是"哪一行抛的"."""

    def test_family_table_names_are_closed(self):
        assert set(FAMILY_OUTCOMES) == {
            "ShapeError",
            "ParameterError",
            "AssemblyError",
            "NumericError",
            "GradientError",
        }

    def test_assembly_error_is_a_parameter_error(self):
        assert issubclass(AssemblyError, ParameterError)
        assert issubclass(AssemblyError, EncoderDecoderError)

    def test_families_stay_inside_the_parent_families(self):
        assert issubclass(ShapeError, CoreShapeError)
        assert issubclass(ShapeError, EncoderDecoderError)
        assert issubclass(EncoderDecoderError, ValueError)
        assert issubclass(GradientError, EncoderDecoderError)
        assert issubclass(NumericError, EncoderDecoderError)

    def test_every_family_says_who_should_fix_it(self):
        for name, outcome in FAMILY_OUTCOMES.items():
            assert outcome.startswith(("改调用", "改数据", "改推导"))
            assert name in {"ShapeError", "ParameterError", "AssemblyError", "NumericError", "GradientError"}


class TestShapeContracts:
    """``BlockShape`` 与 ``CrossShape`` 的派生量与三条拒绝."""

    def test_block_shape_derived_quantities(self):
        shape = block_shape(TOKENS)
        assert shape.hidden == HIDDEN
        assert shape.ffn == 24
        assert shape.tokens == TOKENS
        assert shape.ffn_ratio == 4.0
        assert shape.parameter_count == 4 * HIDDEN + 2 * HIDDEN * 24 + 24 + HIDDEN
        assert shape.parameter_count == 342
        assert shape.to_dict()["ffn_ratio"] == 4.0
        assert "本块新增参数 342 个" in shape.summary_line()

    @pytest.mark.parametrize("name", ["hidden", "ffn", "tokens"])
    def test_block_shape_rejects_zero(self, name):
        payload = {"hidden": HIDDEN, "ffn": 24, "tokens": TOKENS}
        payload[name] = 0
        with pytest.raises(ParameterError, match="必须 >= 1"):
            BlockShape(**payload)

    def test_block_shape_rejects_non_integer(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            BlockShape(hidden=6.0, ffn=24, tokens=4)

    def test_block_shape_rejects_boolean(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            BlockShape(hidden=True, ffn=24, tokens=4)

    def test_cross_shape_weights_are_rectangular(self):
        shape = CrossShape(targets=TOKENS, sources=6, dimension=HIDDEN)
        assert shape.weights_shape == (TOKENS, 6)
        assert "权重 4×6" in shape.summary_line()

    def test_cross_shape_rejects_zero(self):
        with pytest.raises(ParameterError, match="必须 >= 1"):
            CrossShape(targets=0, sources=6, dimension=HIDDEN)


class TestNormRecords:
    """``NormCache`` / ``NormGradients``：四条形状拒绝与两条读数."""

    def _cache(self):
        _out, cache = layer_norm(inputs())
        return cache

    def test_cache_records_the_eps_and_the_normalized(self):
        cache = self._cache()
        assert cache.epsilon == 1e-5
        assert len(cache.mean) == TOKENS
        assert len(cache.variance) == TOKENS
        assert cache.gamma == (1.0,) * HIDDEN
        assert "eps=1e-05" in cache.summary_line()
        assert cache.to_dict()["epsilon"] == 1e-5

    def test_cache_resolves_a_missing_gamma_to_ones(self):
        _out, cache = layer_norm(inputs(), gamma=None, beta=None)
        assert cache.gamma == (1.0,) * HIDDEN
        assert cache.to_dict()["gamma"] == [1.0] * HIDDEN

    def test_cache_records_none_when_gamma_is_absent(self):
        """直接构造一份不带 gamma 的账：``to_dict`` 必须把它写成 ``None``（而不是一行 1）."""
        cache = NormCache(
            inputs=((1.0, 2.0),),
            mean=(1.5,),
            variance=(0.25,),
            normalized=((-1.0, 1.0),),
            epsilon=1e-5,
        )
        assert cache.to_dict()["gamma"] is None

    def test_cache_rejects_mismatched_normalized(self):
        with pytest.raises(ShapeError, match="形状不一致"):
            NormCache(
                inputs=((1.0, 2.0),),
                mean=(0.0,),
                variance=(1.0,),
                normalized=((1.0, 2.0, 3.0),),
                epsilon=1e-5,
            )

    def test_cache_rejects_wrong_mean_length(self):
        with pytest.raises(ShapeError, match="均值/方差各应有"):
            NormCache(
                inputs=((1.0, 2.0),),
                mean=(0.0, 0.0),
                variance=(1.0,),
                normalized=((0.0, 0.0),),
                epsilon=1e-5,
            )

    def test_cache_rejects_wrong_gamma_length(self):
        with pytest.raises(ShapeError, match="与列数"):
            NormCache(
                inputs=((1.0, 2.0),),
                mean=(0.0,),
                variance=(1.0,),
                normalized=((0.0, 0.0),),
                epsilon=1e-5,
                gamma=(1.0, 2.0, 3.0),
            )

    def test_cache_rejects_non_positive_epsilon(self):
        with pytest.raises(ParameterError, match="epsilon"):
            NormCache(
                inputs=((1.0, 2.0),),
                mean=(0.0,),
                variance=(1.0,),
                normalized=((0.0, 0.0),),
                epsilon=0.0,
            )

    def test_norm_gradients_validates_and_summarizes(self):
        cache = self._cache()
        grads = layer_norm_backward(cache, target())
        assert isinstance(grads, NormGradients)
        assert len(grads.grad_gamma) == HIDDEN
        assert len(grads.grad_beta) == HIDDEN
        assert "dGamma" in grads.summary_line()


class TestFFNRecords:
    """``FFNWeights`` / ``FFNCache`` / ``FFNGradients`` 的四条形状拒绝."""

    def _weights(self):
        return block_parameters().ffn

    def test_weights_derived_quantities(self):
        weights = self._weights()
        assert weights.parameter_count == 6 * 24 + 24 + 24 * 6 + 6
        assert len(weights.matrices()) == 4
        assert "w_in (24, 6)" in weights.summary_line()

    def test_weights_reject_wrong_in_columns(self):
        with pytest.raises(ShapeError, match="与 w_out 的行数"):
            FFNWeights(
                w_in=_zeros(24, 5),
                b_in=(0.0,) * 24,
                w_out=_zeros(6, 24),
                b_out=(0.0,) * 6,
            )

    def test_weights_reject_wrong_out_columns(self):
        with pytest.raises(ShapeError, match="与 w_in 的行数"):
            FFNWeights(
                w_in=_zeros(24, HIDDEN),
                b_in=(0.0,) * 24,
                w_out=_zeros(6, 12),
                b_out=(0.0,) * 6,
            )

    def test_weights_reject_wrong_bias_lengths(self):
        with pytest.raises(ShapeError, match="偏置长度不匹配"):
            FFNWeights(
                w_in=_zeros(24, HIDDEN),
                b_in=(0.0,) * 5,
                w_out=_zeros(6, 24),
                b_out=(0.0,) * 6,
            )

    def test_cache_rejects_mismatched_pre_activation(self):
        weights = self._weights()
        with pytest.raises(ShapeError, match="形状必须一致"):
            FFNCache(
                inputs=inputs(),
                pre_activation=_zeros(TOKENS, 5),
                hidden=_zeros(TOKENS, 24),
                weights=weights,
            )

    def test_cache_rejects_unknown_activation(self):
        weights = self._weights()
        with pytest.raises(ParameterError, match="未知的激活函数"):
            FFNCache(
                inputs=inputs(),
                pre_activation=_zeros(TOKENS, 24),
                hidden=_zeros(TOKENS, 24),
                weights=weights,
                activation="tanh",
            )

    def test_cache_summary_counts_the_zeros(self):
        _out, cache = feed_forward(inputs(), self._weights())
        assert "零点占比" in cache.summary_line()
        assert cache.activation == ACTIVATION_RELU

    def test_ffn_gradients_summary_names_five_blocks(self):
        _out, cache = feed_forward(inputs(), self._weights())
        grads = feed_forward_backward(cache, target())
        assert isinstance(grads, FFNGradients)
        for label in ("dW_in", "db_in", "dW_out", "db_out", "dx"):
            assert label in grads.summary_line()


class TestBlockRecords:
    """``BlockParameters`` / ``BlockForward`` / ``BlockGradients`` 的账."""

    def test_parameters_derived_quantities(self):
        params = block_parameters()
        assert params.hidden == HIDDEN
        assert params.parameter_count == 342
        assert len(params.matrices()) == 8
        assert "γ/β 各 2 组" in params.summary_line()
        assert params.ffn.parameter_count == 318

    def test_parameters_flatten_and_unflatten_round_trip(self):
        params = block_parameters()
        flat, shapes = params.flatten()
        assert len(flat) == params.parameter_count
        restored = BlockParameters.unflatten(flat, shapes)
        assert restored == params

    def test_parameters_unflatten_checks_the_block_count(self):
        with pytest.raises(ShapeError, match="与预期的 8 个不一致"):
            BlockParameters.unflatten((1.0,), ((1, 1),))

    def test_parameters_reject_mismatched_gamma_lengths(self):
        params = block_parameters()
        with pytest.raises(ShapeError, match="gamma 长度必须相同"):
            BlockParameters(
                norm1_gamma=(1.0,) * 5,
                norm1_beta=params.norm1_beta,
                ffn_w_in=params.ffn_w_in,
                ffn_b_in=params.ffn_b_in,
                ffn_w_out=params.ffn_w_out,
                ffn_b_out=params.ffn_b_out,
                norm2_gamma=(1.0,) * 6,
                norm2_beta=params.norm2_beta,
            )

    def test_parameters_borrow_the_ffn_shape_check(self):
        params = block_parameters()
        with pytest.raises(ShapeError, match="不一致"):
            BlockParameters(
                norm1_gamma=params.norm1_gamma,
                norm1_beta=params.norm1_beta,
                ffn_w_in=_zeros(24, 5),
                ffn_b_in=params.ffn_b_in,
                ffn_w_out=params.ffn_w_out,
                ffn_b_out=params.ffn_b_out,
                norm2_gamma=params.norm2_gamma,
                norm2_beta=params.norm2_beta,
            )

    def test_parameters_with_ffn_keeps_the_norms(self):
        params = block_parameters()
        swapped = params.with_ffn(params.ffn)
        assert swapped == params
        assert swapped.hidden == params.hidden

    def test_forward_records_six_stages(self):
        params, _attention, forward = _pre_forward()
        assert isinstance(forward, BlockForward)
        assert forward.placement == NORM_PRE
        assert forward.tokens == TOKENS
        assert forward.shape.hidden == HIDDEN
        assert "pre-LN" in forward.summary_line()
        assert forward.to_dict()["placement"] == NORM_PRE
        assert any("阶段顺序" in note for note in forward.notes)

    def test_forward_rejects_a_shape_change(self):
        _params, attention, forward = _pre_forward()
        with pytest.raises(ShapeError, match="形状必须一致"):
            BlockForward(
                shape=forward.shape,
                placement=NORM_PRE,
                use_residual=True,
                inputs=forward.inputs,
                attention=attention,
                norm1=forward.norm1,
                branch1=forward.branch1,
                residual1=forward.residual1,
                norm2=forward.norm2,
                branch2=forward.branch2,
                output=tuple(tuple(1.0 for _ in range(HIDDEN)) for _ in range(TOKENS + 1)),
            )

    def test_forward_rejects_unknown_placement(self):
        _params, attention, forward = _pre_forward()
        with pytest.raises(AssemblyError, match="未知的 LN 摆放位置"):
            BlockForward(
                shape=forward.shape,
                placement="middle",
                use_residual=True,
                inputs=forward.inputs,
                attention=attention,
                norm1=forward.norm1,
                branch1=forward.branch1,
                residual1=forward.residual1,
                norm2=forward.norm2,
                branch2=forward.branch2,
                output=forward.output,
            )

    def test_forward_stringifies_notes(self):
        _params, attention, forward = _pre_forward()
        assert BlockForward(
            shape=forward.shape,
            placement=NORM_PRE,
            use_residual=True,
            inputs=forward.inputs,
            attention=attention,
            norm1=forward.norm1,
            branch1=forward.branch1,
            residual1=forward.residual1,
            norm2=forward.norm2,
            branch2=forward.branch2,
            output=forward.output,
            notes=(1,),
        ).notes == ("1",)

    def test_block_gradients_flatten_and_dict(self):
        params, _attention, forward = _pre_forward()
        grads = encoder_block_backward(forward, params, target())
        assert isinstance(grads, BlockGradients)
        assert len(grads.matrices()) == 9
        assert len(grads.flatten()) == params.parameter_count + TOKENS * HIDDEN
        assert set(grads.as_dict()) == {
            "norm1_gamma",
            "norm1_beta",
            "ffn_w_in",
            "ffn_b_in",
            "ffn_w_out",
            "ffn_b_out",
            "norm2_gamma",
            "norm2_beta",
            "inputs",
        }
        for label in ("dγ1", "dβ1", "dW_in", "db_in", "dW_out", "db_out", "dγ2", "dβ2", "dx"):
            assert label in grads.summary_line()


class TestCrossRecords:
    """``CrossParameters`` / ``CrossForward`` / ``CrossGradients`` 的两路形状."""

    def _params(self):
        return CrossParameters(
            w_query=tuple(tuple(0.02 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
            w_key=tuple(tuple(0.03 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
            w_value=tuple(tuple(0.04 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
            w_output=tuple(tuple(0.05 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
        )

    def test_parameters_summary_and_count(self):
        params = self._params()
        assert params.parameter_count == 4 * HIDDEN * HIDDEN
        assert len(params.matrices()) == 4
        assert "W_q (6, 6)" in params.summary_line()

    def test_parameters_reject_query_key_row_mismatch(self):
        with pytest.raises(ShapeError, match="必须相同（打分是同一个空间里的点积）"):
            CrossParameters(
                w_query=_zeros(6, HIDDEN),
                w_key=_zeros(5, HIDDEN),
                w_value=_zeros(6, HIDDEN),
                w_output=_zeros(6, 6),
            )

    def test_parameters_reject_value_output_column_mismatch(self):
        with pytest.raises(ShapeError, match="必须相同"):
            CrossParameters(
                w_query=_zeros(6, HIDDEN),
                w_key=_zeros(6, HIDDEN),
                w_value=_zeros(6, HIDDEN),
                w_output=_zeros(6, 5),
            )

    def test_parameters_reject_key_value_column_mismatch(self):
        """K 与 V 都来自 source——一个作用在 target 上的 K/V 会被**组装层**拒绝."""
        with pytest.raises(AssemblyError, match="都来自 source 那一侧"):
            CrossParameters(
                w_query=_zeros(6, HIDDEN),
                w_key=_zeros(6, HIDDEN),
                w_value=_zeros(6, 5),
                w_output=_zeros(6, 6),
            )

    def test_forward_weights_are_rectangular(self):
        params = self._params()
        source = tuple(tuple(0.2 * (i + 1) for _ in range(HIDDEN)) for i in range(6))
        forward = cross_attention(params, inputs(), source)
        assert forward.shape.weights_shape == (TOKENS, 6)
        assert len(forward.weights) == TOKENS
        assert len(forward.weights[0]) == 6
        assert approx(max(abs(v - 1.0) for v in forward.row_sums), 0.0)
        assert "权重 4×6" in forward.summary_line()
        assert forward.to_dict()["row_sums"][0] == pytest.approx(1.0, abs=1e-12)

    def test_forward_rejects_wrong_weight_shape(self):
        params = self._params()
        source = tuple(tuple(0.2 * (i + 1) for _ in range(HIDDEN)) for i in range(6))
        forward = cross_attention(params, inputs(), source)
        with pytest.raises(ShapeError, match="权重是 \\(n_tgt, n_src\\)"):
            CrossForward(
                shape=forward.shape,
                params=params,
                target_inputs=forward.target_inputs,
                source_inputs=forward.source_inputs,
                queries=forward.queries,
                keys=forward.keys,
                values=forward.values,
                scores=forward.scores,
                weights=forward.weights[:-1],
                context=forward.context,
                output=forward.output,
                scale=forward.scale,
            )

    def test_forward_stringifies_notes(self):
        params = self._params()
        source = tuple(tuple(0.2 * (i + 1) for _ in range(HIDDEN)) for i in range(6))
        forward = cross_attention(params, inputs(), source)
        rebuilt = CrossForward(
            shape=forward.shape,
            params=params,
            target_inputs=forward.target_inputs,
            source_inputs=forward.source_inputs,
            queries=forward.queries,
            keys=forward.keys,
            values=forward.values,
            scores=forward.scores,
            weights=forward.weights,
            context=forward.context,
            output=forward.output,
            scale=forward.scale,
            notes=(1,),
        )
        assert rebuilt.notes == ("1",)

    def test_cross_gradients_flatten_and_dict(self):
        params = self._params()
        source = tuple(tuple(0.2 * (i + 1) for _ in range(HIDDEN)) for i in range(6))
        forward = cross_attention(params, inputs(), source)
        grads = cross_attention_backward(
            forward, tuple(tuple(0.5 for _ in range(HIDDEN)) for _ in range(TOKENS))
        )
        assert isinstance(grads, CrossGradients)
        assert len(grads.matrices()) == 6
        assert len(grads.flatten()) == 4 * HIDDEN * HIDDEN + TOKENS * HIDDEN + len(source) * HIDDEN
        assert set(grads.as_dict()) == {
            "w_query",
            "w_key",
            "w_value",
            "w_output",
            "target_inputs",
            "source_inputs",
        }
        for label in ("dW_q", "dW_k", "dW_v", "dW_o", "dTarget", "dSource"):
            assert label in grads.summary_line()


class TestDecoderRecord:
    """``DecoderGradients`` 的两路输入与三段子层的账."""

    def _build(self):
        params = block_parameters()
        gammas = tuple((1.0,) * HIDDEN for _ in range(3))
        betas = tuple((0.0,) * HIDDEN for _ in range(3))
        dec = tuple(tuple(0.1 * (i + 1) + 0.02 * (j + 1) for j in range(HIDDEN)) for i in range(3))
        enc = tuple(tuple(0.3 * (i + 1) + 0.02 * (j + 1) for j in range(HIDDEN)) for i in range(6))
        attention = self_attention(
            attention_parameters(), layer_norm(dec, gamma=gammas[0], beta=betas[0])[0], causal=True
        )
        return params, gammas, betas, dec, enc, attention

    def test_decoder_gradients_summary(self):
        params, gammas, betas, dec, enc, attention = self._build()
        cross = CrossParameters(
            w_query=tuple(tuple(0.02 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
            w_key=tuple(tuple(0.03 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
            w_value=tuple(tuple(0.04 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
            w_output=tuple(tuple(0.05 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
        )
        output, cross_forward, caches, ffn_cache = decoder_block(
            attention, cross, gammas, betas, params.ffn, dec, enc
        )
        grads = decoder_block_backward(
            attention,
            cross_forward,
            caches,
            ffn_cache,
            tuple(tuple(0.5 for _ in range(HIDDEN)) for _ in range(3)),
        )
        assert isinstance(grads, DecoderGradients)
        assert len(grads.grad_decoder_inputs) == 3
        assert len(grads.grad_encoder_outputs) == 6
        assert len(grads.grad_norm_gammas) == 3
        assert "dDecoder" in grads.summary_line()

    def test_decoder_gradients_require_three_norms(self):
        _params, _gammas, _betas, dec, enc, _attention = self._build()
        with pytest.raises(ShapeError, match="必须有 3 个"):
            DecoderGradients(
                grad_decoder_inputs=dec,
                grad_encoder_outputs=enc,
                grad_self_params=None,
                grad_cross=None,
                grad_ffn=None,
                grad_norm_gammas=((1.0,), (1.0,)),
                grad_norm_betas=((0.0,), (0.0,), (0.0,)),
            )


class TestLayerNorm:
    """LayerNorm 前向/反向与两条逐元素原语（激活与其导数）."""

    def test_layer_norm_defaults_are_gamma_one_beta_zero(self):
        out, cache = layer_norm(((1.0, 2.0, 3.0, 4.0),))
        assert approx_matrix(out, ((-1.341641, -0.447214, 0.447214, 1.341641),), tolerance=1e-5)
        assert cache.mean == (2.5,)
        assert cache.variance == (1.25,)

    def test_layer_norm_matches_the_eps_free_closed_form(self):
        out, _cache = layer_norm(((1.0, 2.0, 3.0, 4.0),), epsilon=1e-12)
        assert out[0][0] == pytest.approx(-1.341641, abs=1e-6)

    def test_layer_norm_rejects_wrong_gamma_length(self):
        with pytest.raises(ShapeError, match="γ/β 的长度必须等于列数"):
            layer_norm(inputs(), gamma=(1.0, 1.0))

    def test_layer_norm_rejects_non_positive_epsilon(self):
        with pytest.raises(ParameterError, match="epsilon"):
            layer_norm(inputs(), epsilon=-1.0)

    def test_layer_norm_constant_row_gets_zero(self):
        out, cache = layer_norm(((3.0, 3.0, 3.0),))
        assert out[0] == (0.0, 0.0, 0.0)
        assert cache.variance == (0.0,)

    def test_layer_norm_backward_checks_the_grad_shape(self):
        _out, cache = layer_norm(inputs())
        with pytest.raises(ShapeError, match="与输入"):
            layer_norm_backward(cache, ((1.0,),))

    def test_layer_norm_backward_checks_beta_length(self):
        _out, cache = layer_norm(inputs())
        with pytest.raises(ShapeError, match="beta 的长度"):
            layer_norm_backward(cache, target(), beta=(1.0, 2.0))

    def test_layer_norm_backward_accepts_a_valid_beta(self):
        _out, cache = layer_norm(inputs())
        grads = layer_norm_backward(cache, target(), beta=(0.0,) * HIDDEN)
        assert len(grads.grad_inputs) == TOKENS

    def test_layer_norm_backward_runs_without_gamma(self):
        _out, cache = layer_norm(inputs())
        grads = layer_norm_backward(cache, target())
        assert len(grads.grad_inputs) == TOKENS

    def test_activate_relu(self):
        assert activate(((-1.0, 0.0, 2.0),)) == ((0.0, 0.0, 2.0),)

    def test_activate_gelu_at_zero_is_zero(self):
        value = activate(((0.0,),), activation=ACTIVATION_GELU)[0][0]
        assert value == 0.0

    def test_activate_rejects_unknown_activation(self):
        with pytest.raises(ParameterError, match="未知的激活函数"):
            activate(((1.0,),), activation="tanh")

    def test_activation_backward_masks_negative_entries(self):
        pre = ((-1.0, 2.0),)
        grad = ((1.0, 1.0),)
        assert activation_backward(pre, grad) == ((0.0, 1.0),)

    def test_activation_backward_gelu_uses_the_derivative(self):
        pre = ((0.0,),)
        grad = ((1.0,),)
        assert activation_backward(pre, grad, activation=ACTIVATION_GELU)[0][0] == pytest.approx(0.5)

    def test_activation_backward_checks_the_grad_shape(self):
        with pytest.raises(ShapeError, match="形状必须一致（激活是逐元素的）"):
            activation_backward(((1.0, 2.0),), ((1.0,),))

    def test_activation_backward_rejects_unknown_activation(self):
        with pytest.raises(ParameterError, match="未知的激活函数"):
            activation_backward(((1.0,),), ((1.0,),), activation="tanh")


class TestFeedForward:
    """前馈：逐位置的两层网络（第一层扩张 4 倍），以及两条形状拒绝."""

    def test_feed_forward_expands_then_projects(self):
        out, cache = feed_forward(inputs(), block_parameters().ffn)
        assert len(out) == TOKENS
        assert len(out[0]) == HIDDEN
        assert len(cache.hidden[0]) == 24

    def test_feed_forward_rejects_wrong_input_width(self):
        with pytest.raises(ShapeError, match="与输入列数"):
            feed_forward(tuple(tuple(1.0 for _ in range(5)) for _ in range(TOKENS)), block_parameters().ffn)

    def test_feed_forward_backward_checks_the_grad_shape(self):
        _out, cache = feed_forward(inputs(), block_parameters().ffn)
        with pytest.raises(ShapeError, match="回传梯度"):
            feed_forward_backward(cache, ((1.0,),))

    def test_feed_forward_gelu_path_is_covered(self):
        out, cache = feed_forward(inputs(), block_parameters().ffn, activation=ACTIVATION_GELU)
        assert cache.activation == ACTIVATION_GELU
        grads = feed_forward_backward(cache, target())
        assert len(out) == TOKENS
        assert approx(grads.grad_inputs[0][0], grads.grad_inputs[0][0])


class TestResidual:
    """残差：逐位相加，以及"关掉残差是一个开关，而不是另一份实现"."""

    def test_add_residual_is_elementwise(self):
        left = ((1.0, 2.0),)
        right = ((3.0, 4.0),)
        assert add_residual(left, right) == ((4.0, 6.0),)
        assert add_matrices(left, right) == ((4.0, 6.0),)

    def test_add_residual_can_be_switched_off(self):
        left = ((1.0, 2.0),)
        right = ((3.0, 4.0),)
        assert add_residual(left, right, use_residual=False) == right

    def test_add_residual_checks_the_shape(self):
        with pytest.raises(ShapeError, match="残差相加的两边形状不同"):
            add_residual(((1.0, 2.0),), ((1.0, 2.0, 3.0),))

    def test_zero_matrix_is_zero_and_checks_its_shape(self):
        assert zero_matrix(2, 3) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with pytest.raises(ShapeError, match="零矩阵的形状必须为正"):
            zero_matrix(0, 3)


class TestBlockAssembly:
    """块：``block_attention`` 决定注意力作用在什么上面，``stage_order`` 决定阶段顺序."""

    def test_stage_order_moves_the_norm_by_one_slot(self):
        assert stage_order(NORM_PRE) == ENCODER_BLOCK_STAGES
        assert stage_order(NORM_POST) == ("branch1", "add1", "norm1", "branch2", "add2", "norm2")

    def test_stage_order_rejects_unknown_placement(self):
        with pytest.raises(AssemblyError, match="未知的 LN 摆放位置"):
            stage_order("middle")

    def test_block_attention_pre_acts_on_the_normed_input(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
        normed, _cache = layer_norm(inputs(), gamma=params.norm1_gamma, beta=params.norm1_beta)
        reference = self_attention(attention_parameters(), normed, causal=False)
        assert attention.output == reference.output

    def test_block_attention_post_acts_on_the_raw_input(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_POST)
        reference = self_attention(attention_parameters(), inputs(), causal=False)
        assert attention.output == reference.output

    def test_encoder_block_pre_and_post_keep_the_shape(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
        pre = encoder_block(params, inputs(), attention, placement=NORM_PRE)
        attention_post = block_attention(params, inputs(), attention_parameters(), placement=NORM_POST)
        post = encoder_block(params, inputs(), attention_post, placement=NORM_POST)
        assert len(pre.output) == len(post.output) == TOKENS
        assert pre.output != post.output

    def test_encoder_block_rejects_a_non_block_parameters(self):
        with pytest.raises(ParameterError, match="BlockParameters"):
            encoder_block("params", inputs(), None)

    def test_encoder_block_rejects_a_non_attention_forward(self):
        with pytest.raises(ParameterError, match="AttentionForward"):
            encoder_block(block_parameters(), inputs(), "attention")

    def test_encoder_block_checks_the_input_width(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
        with pytest.raises(ShapeError, match="与块的隐藏维"):
            encoder_block(params, tuple(tuple(1.0 for _ in range(5)) for _ in range(TOKENS)), attention)

    def test_encoder_block_checks_the_attention_input_shape(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
        shorter = inputs()[:-1]
        with pytest.raises(ShapeError, match="本块的输入"):
            encoder_block(params, shorter, attention)

    def test_encoder_block_backward_checks_the_grad_shape(self):
        params, _attention, forward = _pre_forward()
        with pytest.raises(ShapeError, match="与块输出"):
            encoder_block_backward(forward, params, ((1.0,),))

    def test_encoder_block_backward_post_path(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_POST)
        forward = encoder_block(params, inputs(), attention, placement=NORM_POST)
        grads = encoder_block_backward(forward, params, target())
        assert len(grads.grad_inputs) == TOKENS

    def test_encoder_block_backward_without_residual(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
        forward = encoder_block(params, inputs(), attention, placement=NORM_PRE, use_residual=False)
        grads = encoder_block_backward(forward, params, target())
        assert len(grads.grad_inputs) == TOKENS

    def test_encoder_block_backward_post_without_residual(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_POST)
        forward = encoder_block(params, inputs(), attention, placement=NORM_POST, use_residual=False)
        grads = encoder_block_backward(forward, params, target(), activation=ACTIVATION_GELU)
        assert len(grads.grad_inputs) == TOKENS

    def test_encoder_block_gelu_path(self):
        params = block_parameters()
        attention = block_attention(params, inputs(), attention_parameters(), placement=NORM_PRE)
        forward = encoder_block(params, inputs(), attention, placement=NORM_PRE, activation=ACTIVATION_GELU)
        assert forward.notes[1].endswith(ACTIVATION_GELU)


class TestTypeHelpers:
    """转发型小工具：最大绝对值、相对误差、eps 与三个私有校验."""

    def test_matrix_max_absolute_is_forwarded(self):
        assert matrix_max_absolute(((1.0, -3.0),)) == 3.0
        assert matrix_max_absolute(((-1.0,),)) == 1.0

    def test_relative_matrix_error_is_forwarded(self):
        assert relative_matrix_error(((1.0,),), ((1.0,),)) == 0.0
        assert approx(relative_matrix_error(((1.0,),), ((2.0,),)), 0.5)

    def test_validate_epsilon_accepts_positive(self):
        assert validate_epsilon(1e-5) == 1e-5

    def test_validate_epsilon_rejects_zero(self):
        with pytest.raises(ParameterError, match="正的有限数"):
            validate_epsilon(0.0)

    def test_checked_positive_int_rejects_boolean(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            _checked_positive_int(True, name="hidden")

    def test_checked_positive_int_rejects_below_one(self):
        with pytest.raises(ParameterError, match="必须 >= 1"):
            _checked_positive_int(0, name="hidden")

    def test_checked_tolerance_rejects_non_number(self):
        with pytest.raises(ParameterError, match="必须是数"):
            _checked_tolerance("tiny")

    def test_checked_tolerance_rejects_nan(self):
        with pytest.raises(ParameterError, match="正的有限数"):
            _checked_tolerance(math.nan)

    def test_checked_placement_rejects_unknown(self):
        with pytest.raises(AssemblyError, match="未知的 LN 摆放位置"):
            _checked_placement("middle")

    def test_checked_activation_rejects_unknown(self):
        with pytest.raises(ParameterError, match="未知的激活函数"):
            _checked_activation("tanh")
