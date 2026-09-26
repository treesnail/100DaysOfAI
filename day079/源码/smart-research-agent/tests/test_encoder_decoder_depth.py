"""深度实验：残差是梯度的那条高速路（day079 / M7-D4）.

这一份测试守三件事：

```text
① 三个变体齐备               缺一个变体，"残差有用"这句话就失去参照物
② 一次只改一个旋钮           同一批初始参数、同一个目标、同一个深度序列
③ 判决落在"最深那一层"        bare 的比值必须比 pre 小一个数量级以上（不因换种子翻面）
```

第 ③ 条是一条**判决**而不是一条读数：它的判据写成了"bare 的比值 < pre 的 1/10"，
这个距离足够远，因此它不会因为换一组随机种子而翻面。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.encoder_decoder import (
    DEFAULT_DEPTHS,
    DEFAULT_INIT_SCALE,
    NORM_POST,
    NORM_PRE,
    NumericError,
    ParameterError,
    VARIANTS,
    VARIANT_BARE,
    VARIANT_DESCRIPTIONS,
    VARIANT_POST_RESIDUAL,
    VARIANT_PRE_RESIDUAL,
    BlockShape,
    DepthRow,
    DepthStudy,
    block_parameter_count,
    depth_study,
    ffn_output_scale,
    make_block_parameters,
    make_stack_parameters,
    matrix_frobenius,
    stack_forward,
    stack_input_gradient_norm,
    stack_loss,
    target_matrix,
)
from smart_research_agent.encoder_decoder.depth import _random_matrix, _variant_settings
from tests.encoder_decoder_samples import HIDDEN, block_parameters, block_shape


class TestVariantTables:
    """三个变体与它们的设置：名字、说明、``(placement, use_residual)``."""

    def test_three_variants_are_described(self):
        assert set(VARIANTS) == {VARIANT_PRE_RESIDUAL, VARIANT_POST_RESIDUAL, VARIANT_BARE}
        assert set(VARIANT_DESCRIPTIONS) == set(VARIANTS)
        for description in VARIANT_DESCRIPTIONS.values():
            assert description

    def test_variant_settings_are_one_knob_apart(self):
        assert _variant_settings(VARIANT_PRE_RESIDUAL) == (NORM_PRE, True)
        assert _variant_settings(VARIANT_POST_RESIDUAL) == (NORM_POST, True)
        assert _variant_settings(VARIANT_BARE) == (NORM_PRE, False)

    def test_unknown_variant_is_refused(self):
        with pytest.raises(ParameterError, match="未知的变体名"):
            _variant_settings("magic")

    def test_default_depths_are_strictly_increasing(self):
        assert list(DEFAULT_DEPTHS) == sorted(set(DEFAULT_DEPTHS))
        assert DEFAULT_INIT_SCALE == 0.25


class TestRandomPrimitives:
    """确定性随机矩阵与参数工厂（同一个种子必须给出同一批数）."""

    def test_random_matrix_is_bounded_and_deterministic(self):
        first = _random_matrix(4, 3, scale=0.25, seed=7)
        second = _random_matrix(4, 3, scale=0.25, seed=7)
        assert first == second
        assert len(first) == 4 and len(first[0]) == 3
        assert all(abs(value) < 0.25 for row in first for value in row)

    def test_random_matrix_rejects_a_non_positive_shape(self):
        with pytest.raises(ParameterError, match="矩阵形状必须为正"):
            _random_matrix(0, 3, scale=0.25, seed=7)

    def test_random_matrix_rejects_a_bad_scale(self):
        with pytest.raises(ParameterError, match="正的有限数"):
            _random_matrix(4, 3, scale=0.0, seed=7)

    def test_block_parameters_start_at_the_identity_norm(self):
        params = make_block_parameters(block_shape(), seed=7, scale=0.25)
        assert params.norm1_gamma == (1.0,) * HIDDEN
        assert params.norm1_beta == (0.0,) * HIDDEN
        assert params.norm2_gamma == (1.0,) * HIDDEN
        assert params.ffn_b_in == (0.0,) * 24
        assert params.ffn_b_out == (0.0,) * HIDDEN

    def test_block_parameters_are_deterministic(self):
        assert make_block_parameters(block_shape()) == make_block_parameters(block_shape())
        assert make_block_parameters(block_shape()) == block_parameters()

    def test_stack_parameters_give_one_block_per_layer(self):
        blocks, attentions = make_stack_parameters(block_shape(), 3)
        assert len(blocks) == len(attentions) == 3
        assert blocks[0] != blocks[1]

    def test_stack_parameters_reject_a_bad_layer_count(self):
        with pytest.raises(ParameterError, match="必须是 >= 1 的整数"):
            make_stack_parameters(block_shape(), 0)

    def test_target_matrix_has_the_block_shape(self):
        target = target_matrix(block_shape())
        assert len(target) == 4
        assert len(target[0]) == HIDDEN


class TestStackForward:
    """整摞块的前向与损失：长度必须一致，摆放位置决定注意力作用在什么上面."""

    def test_forward_returns_one_record_per_layer(self):
        blocks, attentions = make_stack_parameters(block_shape(), 2)
        output, forwards = stack_forward(blocks, attentions, _inputs())
        assert len(output) == 4
        assert len(forwards) == 2

    def test_forward_checks_the_layer_counts(self):
        blocks, attentions = make_stack_parameters(block_shape(), 3)
        with pytest.raises(ParameterError, match="不一致"):
            stack_forward(blocks, attentions[:-1], _inputs())

    def test_post_placement_runs(self):
        blocks, attentions = make_stack_parameters(block_shape(), 2)
        output, _forwards = stack_forward(
            blocks, attentions, _inputs(), placement=NORM_POST
        )
        assert len(output) == 4

    def test_stack_loss_is_a_positive_float(self):
        blocks, attentions = make_stack_parameters(block_shape(), 2)
        value = stack_loss(blocks, attentions, _inputs(), target_matrix(block_shape()))
        assert isinstance(value, float)
        assert value > 0.0

    def test_input_gradient_norm_is_positive_and_shaped(self):
        blocks, attentions = make_stack_parameters(block_shape(), 2)
        norm, grad = stack_input_gradient_norm(
            blocks, attentions, _inputs(), target_matrix(block_shape())
        )
        assert norm > 0.0
        assert len(grad) == 4 and len(grad[0]) == HIDDEN
        assert norm == pytest.approx(matrix_frobenius(grad))

    def test_input_gradient_norm_without_residual(self):
        blocks, attentions = make_stack_parameters(block_shape(), 2)
        norm, _grad = stack_input_gradient_norm(
            blocks,
            attentions,
            _inputs(),
            target_matrix(block_shape()),
            use_residual=False,
        )
        assert norm >= 0.0


class TestDepthRow:
    """一行读数：变体名、层数、‖dx‖、相对第 1 层的比值、损失."""

    def test_row_exposes_its_readings(self):
        row = DepthRow(
            variant=VARIANT_PRE_RESIDUAL,
            layers=8,
            input_gradient_norm=1.35,
            ratio_to_first=9.27,
            loss=0.69,
        )
        assert row.to_dict()["variant"] == VARIANT_PRE_RESIDUAL
        assert "pre_residual" in row.summary_line()
        assert "8 层" in row.summary_line()

    def test_row_rejects_an_unknown_variant(self):
        with pytest.raises(ParameterError, match="未知的变体名"):
            DepthRow(
                variant="magic",
                layers=1,
                input_gradient_norm=1.0,
                ratio_to_first=1.0,
                loss=1.0,
            )

    def test_row_rejects_a_non_finite_reading(self):
        with pytest.raises(NumericError, match="必须是有限数"):
            DepthRow(
                variant=VARIANT_BARE,
                layers=1,
                input_gradient_norm=math.inf,
                ratio_to_first=1.0,
                loss=1.0,
            )


class TestDepthStudy:
    """一条深度序列上的对照（这一课的值钱结论落在这里）."""

    def test_small_study_is_complete(self):
        study = depth_study(block_shape(), depths=(1, 2, 3))
        assert isinstance(study, DepthStudy)
        assert len(study.ratios(VARIANT_PRE_RESIDUAL)) == 3
        assert len(study.ratios(VARIANT_BARE)) == 3
        assert study.deepest == 3
        assert study.to_dict()["depths"] == [1, 2, 3]
        assert "深度实验" in study.summary_line()
        assert len(study.table_lines()) == 3 * 3 + 2

    def test_ratios_are_relative_to_the_first_layer(self):
        study = depth_study(block_shape(), depths=(1, 2))
        assert study.ratios(VARIANT_PRE_RESIDUAL)[0] == 1.0
        assert study.ratios(VARIANT_POST_RESIDUAL)[0] == 1.0

    def test_study_rejects_sparse_variants(self):
        rows = (
            DepthRow(
                variant=VARIANT_PRE_RESIDUAL,
                layers=1,
                input_gradient_norm=1.0,
                ratio_to_first=1.0,
                loss=1.0,
            ),
        )
        with pytest.raises(NumericError, match="三个变体必须齐备"):
            DepthStudy(shape=block_shape(), rows=rows, depths=(1,), seed=7)

    def test_study_rejects_an_empty_row_list(self):
        with pytest.raises(ParameterError, match="至少要有一行读数"):
            DepthStudy(shape=block_shape(), rows=(), depths=(1,), seed=7)

    def test_study_rejects_an_empty_depth_sequence(self):
        with pytest.raises(ParameterError, match="深度序列不能为空"):
            depth_study(block_shape(), depths=())

    def test_study_rejects_unsorted_depths(self):
        with pytest.raises(ParameterError, match="必须严格递增"):
            depth_study(block_shape(), depths=(1, 3, 2))

    def test_default_study_gives_the_verdict(self):
        """默认深度序列（1,2,3,4,6,8）：bare 衰减得**明显**更快——'残差有用'的判决."""
        study = depth_study(block_shape())
        assert study.verdict_ok is True
        assert study.ratios(VARIANT_BARE)[-1] < study.ratios(VARIANT_PRE_RESIDUAL)[-1] / 10.0
        assert study.ratios(VARIANT_BARE)[-1] < 1e-2
        assert study.ratios(VARIANT_POST_RESIDUAL)[-1] < 1.0
        assert study.ratios(VARIANT_PRE_RESIDUAL)[-1] > 1.0


class TestDepthHelpers:
    """两个小工具：每层参数量与前馈输出层的缩放."""

    def test_block_parameter_count_scales_with_layers(self):
        shape = block_shape()
        assert block_parameter_count(shape, 1) == shape.parameter_count
        assert block_parameter_count(shape, 8) == 8 * shape.parameter_count

    def test_block_parameter_count_rejects_a_bad_layer_count(self):
        with pytest.raises(ParameterError, match="必须是 >= 1 的整数"):
            block_parameter_count(block_shape(), 0)

    def test_ffn_output_scale_only_touches_the_output_layer(self):
        weights = block_parameters().ffn
        scaled = ffn_output_scale(weights, 0.5)
        assert scaled.w_in == weights.w_in
        assert scaled.b_in == weights.b_in
        assert scaled.w_out[0][0] == pytest.approx(weights.w_out[0][0] * 0.5)
        assert scaled.b_out[0] == pytest.approx(weights.b_out[0] * 0.5)

    def test_ffn_output_scale_rejects_a_non_finite_factor(self):
        with pytest.raises(ParameterError, match="scale 必须是有限数"):
            ffn_output_scale(block_parameters().ffn, math.inf)

    def test_matrix_frobenius_is_the_euclidean_norm(self):
        assert matrix_frobenius(((3.0, 4.0),)) == 5.0


def _inputs():
    """写死的非平凡输入（每一行方差非零）."""
    return tuple(
        tuple(0.2 * (i + 1) + 0.1 * (j + 1) for j in range(HIDDEN)) for i in range(4)
    )
