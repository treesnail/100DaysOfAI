"""``transformer_stack.layers``：链式前向、链式反向与几个开关（day080）."""

from __future__ import annotations

import pytest

from smart_research_agent.encoder_decoder.types import NORM_POST, NORM_PRE
from smart_research_agent.math_foundations.types import matrix_shape
from smart_research_agent.transformer_core.layers import mse_gradient
from smart_research_agent.transformer_stack import (
    DEFAULT_FFN_RATIO,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_TOKENS,
    LAYER_SEED_STRIDE,
    AssemblyError,
    ParameterError,
    make_shape,
    make_stack_parameters,
    sequence_slice,
    stack_backward,
    stack_forward,
    stack_gradient_profile,
    stack_loss,
    stack_loss_gradient,
)
from smart_research_agent.transformer_stack.layers import (
    ATTENTION_SEED_STRIDE,
    flatten_layer_parameter_norms,
    layer_census_of,
    zero_attention,
    zero_branches,
)
from tests.stack_samples import (
    HIDDEN,
    LAYERS,
    PLACEMENTS,
    TOKENS,
    inputs,
    matrix_norm,
    parameters,
    shape,
    shape_of,
    target,
)


class TestMakeShape:
    """形状工厂：``ffn = d × ratio``，默认值与 day073~079 同口径."""

    def test_defaults_match_the_sample(self):
        """默认形状就是样本形状（4 层 / d=6 / d_ff=24 / n=4）."""
        item = make_shape()
        assert (item.layers, item.hidden, item.ffn, item.tokens) == (
            DEFAULT_LAYERS,
            DEFAULT_HIDDEN,
            DEFAULT_HIDDEN * DEFAULT_FFN_RATIO,
            DEFAULT_TOKENS,
        )

    def test_ratio_is_applied_to_the_hidden_width(self):
        """``ffn_ratio=2`` 时 d_ff = 2d."""
        assert make_shape(ffn_ratio=2).ffn == 2 * DEFAULT_HIDDEN

    @pytest.mark.parametrize("ratio", [0, -1, True, 1.5])
    def test_ratio_must_be_a_positive_integer(self, ratio: object):
        """前馈倍数必须是 >= 1 的整数."""
        with pytest.raises(ParameterError):
            make_shape(ffn_ratio=ratio)  # type: ignore[arg-type]


class TestSeedDiscipline:
    """两份参数各走各的步长：第 i 层用 ``seed + i*13`` 与 ``seed + i*17``."""

    def test_layer_seeds_differ(self):
        """相邻两层的块参数**不相等**（否则逐层曲线会变成周期 1 的序列）."""
        params = parameters(layers=3)
        assert params.blocks[0] != params.blocks[1]
        assert params.attentions[0] != params.attentions[1]

    def test_the_first_layer_is_the_same_across_depths(self):
        """同一个 ``seed`` 下，第 0 层在任何深度上都相同（深度实验靠这一条）."""
        shallow = parameters(layers=1)
        deep = parameters(layers=8)
        assert shallow.blocks[0] == deep.blocks[0]
        assert shallow.attentions[0] == deep.attentions[0]

    def test_strides_are_13_and_17(self):
        """两个步长被写成常量，而不是散在公式里."""
        assert LAYER_SEED_STRIDE == 13
        assert ATTENTION_SEED_STRIDE == 17

    @pytest.mark.parametrize("scale", [0.0, 1.0, -0.5, True, "0.25"])
    def test_scale_must_live_in_the_open_unit_interval(self, scale: object):
        """初始化幅度必须落在 ``(0, 1)``（与 day075 的 ``default_parameters`` 同一条纪律）."""
        with pytest.raises(ParameterError):
            make_stack_parameters(shape(), scale=scale)  # type: ignore[arg-type]


class TestStackForward:
    """五阶段里的前三段（进入 / 块 / 记账）."""

    def test_shape_is_preserved_through_the_whole_chain(self):
        """逐层保形 + 入口与出口同形（``(4, 6)`` 进、``(4, 6)`` 出）."""
        forward = stack_forward(parameters(), inputs())
        assert shape_of(forward.inputs) == (TOKENS, HIDDEN)
        assert shape_of(forward.output) == (TOKENS, HIDDEN)
        for index in range(LAYERS):
            layer = forward.layer_at(index)
            assert shape_of(layer.block.inputs) == shape_of(layer.block.output)

    def test_the_chain_carries_the_same_value(self):
        """第 i 层的输出与第 i+1 层的输入**逐位相等**（不是重算一遍的近似）."""
        forward = stack_forward(parameters(), inputs())
        assert forward.layers[0].block.output == forward.layers[1].block.inputs
        assert forward.layers[1].block.output == forward.layers[2].block.inputs
        assert forward.layers[-1].block.output == forward.output

    def test_census_is_recorded_for_every_layer(self):
        """每一层都有一行读数，且第 i 行的出口就是第 i+1 行的入口."""
        forward = stack_forward(parameters(), inputs())
        assert [row.index for row in forward.censuses] == list(range(LAYERS))
        for index in range(LAYERS - 1):
            assert forward.censuses[index].output_norm == forward.censuses[index + 1].input_norm

    def test_carry_share_is_strictly_inside_the_unit_interval(self):
        """残差开时直通占比严格落在 ``(0, 1)``:两份都真的在贡献."""
        forward = stack_forward(parameters(), inputs(), use_residual=True)
        assert all(0.0 < value < 1.0 for value in forward.carry_profile())

    def test_carry_share_is_exactly_zero_without_residual(self):
        """残差关时直通占比**恰好**是 0：入口那一份没有被传下去."""
        forward = stack_forward(parameters(), inputs(), use_residual=False)
        assert forward.carry_profile() == tuple(0.0 for _ in range(LAYERS))

    def test_gain_profile_matches_the_readings(self):
        """增益序列与逐行读数一致（同一个口径，没有第二个实现）."""
        forward = stack_forward(parameters(), inputs())
        assert forward.gain_profile() == tuple(row.gain for row in forward.censuses)
        assert all(abs(value) < 100.0 for value in forward.gain_profile())

    def test_max_abs_input_matches_manual_scan(self):
        """``max_abs_input`` 是一个**单点**读数（可手算核对）."""
        sample = inputs()
        forward = stack_forward(parameters(), inputs())
        assert forward.census_at(0).max_abs_input == pytest.approx(
            max(abs(value) for row in sample for value in row)
        )

    @pytest.mark.parametrize("placement", PLACEMENTS)
    def test_both_placements_run_and_preserve_shape(self, placement: str):
        """pre 与 post 都能跑通，且输出同形."""
        forward = stack_forward(parameters(), inputs(), placement=placement)
        assert forward.placement == placement
        assert shape_of(forward.output) == (TOKENS, HIDDEN)

    @pytest.mark.parametrize("placement", PLACEMENTS)
    def test_both_placements_agree_on_the_census_count(self, placement: str):
        """两种摆放位置留下的账行数一致（形状一样，这是 day079 的结论）."""
        forward = stack_forward(parameters(), inputs(), placement=placement)
        assert len(forward.censuses) == LAYERS

    def test_gelu_runs_too(self):
        """``gelu`` 也能跑（激活是函数参数，不是写死的）。"""
        forward = stack_forward(parameters(), inputs(), activation="gelu")
        assert forward.activation == "gelu"

    @pytest.mark.parametrize("activation", ["silu", "", None])
    def test_unknown_activation_is_rejected(self, activation: object):
        """不认识的激活当场报错（不给它挑一个默认值）."""
        with pytest.raises(ValueError):
            stack_forward(parameters(), inputs(), activation=activation)  # type: ignore[arg-type]

    def test_unknown_placement_is_rejected(self):
        """不认识的摆放位置当场报错."""
        with pytest.raises(ValueError):
            stack_forward(parameters(), inputs(), placement="middle")

    def test_input_width_must_match_the_hidden_width(self):
        """输入宽度与隐藏维不一致时当场报错（而不是在注意力那一步炸）."""
        with pytest.raises(ParameterError):
            stack_forward(parameters(), inputs(hidden=8))

    def test_single_layer_stack(self):
        """一层也是合法的堆叠（性质里的“与手写循环一致”要在 1 层上也成立）."""
        forward = stack_forward(parameters(layers=1), inputs())
        assert forward.depth == 1
        assert forward.censuses[0].carry_share > 0.0

    def test_eight_layer_stack(self):
        """八层也能跑（深度实验最深的那一档）."""
        forward = stack_forward(parameters(layers=8), inputs())
        assert forward.depth == 8


class TestStackLossAndGradient:
    """损失与链式反向."""

    def test_loss_matches_the_hand_written_forward(self):
        """损失与“先跑前向、再算 MSE”一致（两边用同一个损失函数）."""
        params = parameters()
        forward = stack_forward(params, inputs())
        from smart_research_agent.transformer_core.layers import mean_squared_error

        assert stack_loss(params, inputs(), target()) == pytest.approx(
            mean_squared_error(forward.output, target())
        )

    def test_loss_gradient_returns_both_records(self):
        """一条龙返回前向账与反向账，且两者层数一致."""
        forward, grads = stack_loss_gradient(parameters(), inputs(), target())
        assert forward.depth == grads.depth == LAYERS

    def test_gradient_norms_are_all_finite_and_positive(self):
        """逐层 ‖dx‖ 都是有限正数（残差开时不该有 0）."""
        norms = stack_gradient_profile(parameters(), inputs(), target())
        assert len(norms) == LAYERS
        assert all(0.0 < value < 1e6 for value in norms)

    def test_gradient_reaches_the_bottom_and_shrinks(self):
        """残差开时梯度是**一路传下来的**：逐层范数随下标增大而递减（本样本）. """
        norms = stack_gradient_profile(parameters(), inputs(), target())
        assert norms[0] > norms[-1]
        assert all(norms[index] > norms[index + 1] for index in range(LAYERS - 1))

    def test_gradients_are_recorded_in_layer_order(self):
        """梯度账与层号同序：第 i 环的输入梯度与该层入口同形."""
        forward, grads = stack_loss_gradient(parameters(), inputs(), target())
        for index in range(LAYERS):
            assert shape_of(grads.layer_at(index).grad_inputs) == shape_of(
                forward.layer_at(index).block.inputs
            )
        assert grads.grad_inputs == grads.layer_at(0).grad_inputs

    def test_each_layer_sees_a_different_upstream_gradient(self):
        """**链式反向的那一行**：把“每层都从 ``dLoss/dy_N`` 起步”写出来，结果不同.

        这是本课最有价值的一条反证：错法（每层用同一份 ``grad_output``）
        形状全对、也能跑，而它给出的第 0 层梯度与正确结果**不同**——
        因此“换手”那一行是必须的，而不是风格问题。
        """
        from smart_research_agent.encoder_decoder.layers import encoder_block_backward

        params = parameters()
        forward, grads = stack_loss_gradient(params, inputs(), target())
        top = mse_gradient(forward.output, target())
        layer0 = forward.layer_at(0)
        wrong = encoder_block_backward(layer0.block, layer0.params, top).grad_inputs
        assert wrong != grads.grad_inputs

    def test_manual_chaining_reproduces_the_analytic_gradient(self):
        """手写一趟“换手”的循环，结果与 ``stack_backward`` 逐位一致."""
        from smart_research_agent.encoder_decoder.layers import encoder_block_backward

        params = parameters()
        forward = stack_forward(params, inputs())
        current = mse_gradient(forward.output, target())
        for index in reversed(range(LAYERS)):
            layer = forward.layer_at(index)
            current = encoder_block_backward(layer.block, layer.params, current).grad_inputs
        analytic = stack_backward(forward, params, mse_gradient(forward.output, target()))
        assert current == analytic.grad_inputs

    def test_backward_rejects_a_mismatched_parameter_bundle(self):
        """账与参数份数不一致 ⇒ ``AssemblyError``（反向必须用产生这份账的那一摞参数）."""
        forward = stack_forward(parameters(), inputs())
        with pytest.raises(AssemblyError):
            stack_backward(forward, parameters(layers=2), mse_gradient(forward.output, target()))

    def test_backward_rejects_a_non_matrix_gradient(self):
        """``grad_output`` 必须是矩阵（形状口径只有一处实现）."""
        forward = stack_forward(parameters(), inputs())
        with pytest.raises(ValueError):
            stack_backward(forward, parameters(), [[0.0]])  # type: ignore[arg-type]

    def test_without_residual_the_gradient_is_much_smaller_at_the_bottom(self):
        """把残差关掉，最底层的梯度明显变小（day079 的深度结论在这里重现）."""
        params = parameters(layers=8)
        with_residual = stack_gradient_profile(params, inputs(), target(), use_residual=True)
        without = stack_gradient_profile(params, inputs(), target(), use_residual=False)
        assert without[0] < with_residual[0] / 10.0


class TestZeroSwitches:
    """两个置零开关（“分支全零 ⇒ 恒等”这条性质靠它们）."""

    def test_zero_branches_zeroes_the_ffn_output_layer(self):
        """``w_out`` 与 ``b_out`` 变成全零，其余参数**逐位不动**."""
        params = parameters(layers=1).blocks[0]
        deflated = zero_branches(params)
        assert all(value == 0.0 for row in deflated.ffn_w_out for value in row)
        assert all(value == 0.0 for value in deflated.ffn_b_out)
        assert deflated.ffn_w_in == params.ffn_w_in
        assert deflated.norm1_gamma == params.norm1_gamma

    def test_zero_attention_zeroes_the_four_projections(self):
        """四个投影都变成零矩阵."""
        item = parameters(layers=1).attentions[0]
        deflated = zero_attention(item)
        for matrix in (deflated.w_query, deflated.w_key, deflated.w_value, deflated.w_output):
            assert all(value == 0.0 for row in matrix for value in row)

    def test_zero_branches_requires_a_block_record(self):
        """传进来不是 ``BlockParameters`` 时当场报错."""
        with pytest.raises(ParameterError):
            zero_branches("not-a-record")  # type: ignore[arg-type]

    def test_both_switches_together_give_the_identity(self):
        """两个开关一起关 ⇒ 输出与输入逐位相等（**逐位**，不是近似）."""
        from smart_research_agent.transformer_stack import StackParameters

        params = parameters()
        deflated = StackParameters(
            blocks=tuple(zero_branches(item) for item in params.blocks),
            attentions=tuple(zero_attention(item) for item in params.attentions),
        )
        forward = stack_forward(deflated, inputs())
        assert forward.output == inputs()

    def test_only_the_ffn_out_zero_is_not_enough(self):
        """只关前馈的输出层**不够**：注意力那一支还在（day079 5.2 节的坑）."""
        from smart_research_agent.transformer_stack import StackParameters

        params = parameters()
        half = StackParameters(
            blocks=tuple(zero_branches(item) for item in params.blocks),
            attentions=params.attentions,
        )
        forward = stack_forward(half, inputs())
        assert forward.output != inputs()


class TestSmallHelpers:
    """几个便宜的小函数（它们存在的理由是“让演示脚本少写循环”）."""

    def test_parameter_norms_are_positive_per_layer(self):
        """逐层参数范数是正数，且层与层不同（每一层有自己的种子）."""
        norms = flatten_layer_parameter_norms(parameters())
        assert len(norms) == LAYERS
        assert all(value > 0.0 for value in norms)
        assert norms[0] != norms[1]

    def test_layer_census_of_returns_the_row(self):
        """按层号取读数，越界当场报错."""
        forward = stack_forward(parameters(), inputs())
        assert layer_census_of(forward, 2).index == 2
        with pytest.raises(ParameterError):
            layer_census_of(forward, LAYERS)

    def test_sequence_slice(self):
        """按层号取序列里的读数，越界与非整数下标都当场报错."""
        values = (1.0, 2.0, 3.0)
        assert sequence_slice(values, 1) == 2.0
        with pytest.raises(ParameterError):
            sequence_slice(values, 3)
        with pytest.raises(ParameterError):
            sequence_slice(values, 1.5)  # type: ignore[arg-type]

    def test_norm_helper_matches_the_package_reading(self):
        """样本里的范数与包里的读数一致（口径只有一处）."""
        from smart_research_agent.transformer_stack import frobenius

        matrix = ((3.0, 4.0), (0.0, 0.0))
        assert matrix_norm(matrix) == pytest.approx(frobenius(matrix))

    def test_gelu_activation_changes_the_output(self):
        """换激活会真的改变输出（否则“激活是一个参数”这句话就是空的）."""
        relu = stack_forward(parameters(), inputs(), activation="relu")
        gelu = stack_forward(parameters(), inputs(), activation="gelu")
        assert relu.output != gelu.output

    def test_post_placement_changes_the_output(self):
        """换摆放位置会真的改变输出."""
        pre = stack_forward(parameters(), inputs(), placement=NORM_PRE)
        post = stack_forward(parameters(), inputs(), placement=NORM_POST)
        assert pre.output != post.output

    def test_depth_can_be_read_off_the_record(self):
        """层数从账里读，不从参数里猜."""
        assert stack_forward(parameters(layers=6), inputs()).depth == 6

    def test_matrix_shape_is_used_consistently(self):
        """形状读数与 ``matrix_shape`` 一致（没有第二套形状口径）."""
        forward = stack_forward(parameters(), inputs())
        assert matrix_shape(forward.output) == (TOKENS, HIDDEN)
