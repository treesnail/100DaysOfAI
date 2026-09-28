"""``transformer_stack.types``：形状、读数、两种账与三张表（day080）."""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.transformer_stack import (
    DEFAULT_INIT_SCALE,
    GRAD_STACK_INPUTS,
    KINDS,
    KIND_LAYER,
    KIND_STACK,
    LAYER_GRADIENT_TARGETS,
    MINIMUM_LAYERS,
    PROPERTY_DESCRIPTIONS,
    STACK_GRADIENT_TARGETS,
    STACK_NOTES,
    STACK_PROPERTIES,
    STACK_STAGES,
    STAGE_BLOCK,
    STAGE_CARRY,
    STAGE_CENSUS,
    STAGE_DESCRIPTIONS,
    STAGE_ENTER,
    STAGE_EXIT,
    STAGE_SHAPES,
    LayerCensus,
    ShapeError,
    StackForward,
    StackGradients,
    StackParameters,
    StackShape,
    StackedLayer,
    frobenius,
    max_absolute,
    relative_matrix_error,
)
from smart_research_agent.transformer_stack.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
)
from tests.stack_samples import HIDDEN, LAYERS, TOKENS, inputs, parameters, shape


class TestStackShape:
    """形状的四个维度、三个参数量与两种输出."""

    def test_sample_numbers_are_hand_checkable(self):
        """4 层 / d=6 / d_ff=24：每层 486、合计 1944——三个数都能手算."""
        item = shape()
        assert item.block_parameter_count == 5 * 6 + 2 * 6 * 24 + 24 == 342
        assert item.attention_parameter_count == 4 * 6 * 6 == 144
        assert item.layer_parameter_count == 486
        assert item.total_parameter_count == 4 * 486 == 1944

    def test_ffn_ratio_is_derived(self):
        """``ffn_ratio`` 是派生量（d=6、d_ff=24 → 4.0），而不是写死的常数."""
        assert shape().ffn_ratio == 4.0
        assert shape(ffn_ratio=2).ffn_ratio == 2.0

    def test_block_shape_drops_the_layer_count(self):
        """``block_shape`` 交给 day079 的 ``encoder_block``，因此**不含层数**."""
        item = shape()
        assert (item.block_shape.hidden, item.block_shape.ffn, item.block_shape.tokens) == (
            6,
            24,
            4,
        )

    @pytest.mark.parametrize("name", ["hidden", "ffn", "tokens"])
    def test_positive_dimensions_are_required(self, name: str):
        """三个维度都必须是正整数."""
        payload = {"hidden": HIDDEN, "ffn": HIDDEN * 4, "tokens": TOKENS, "layers": LAYERS}
        payload[name] = 0
        with pytest.raises(ParameterError):
            StackShape(**payload)  # type: ignore[arg-type]

    @pytest.mark.parametrize("layers", [0, -1, True, 1.5, "4"])
    def test_layers_must_be_a_positive_integer(self, layers: object):
        """层数必须是 >= 1 的整数（``0`` 层在数学上是恒等，在工程上是一次笔误）."""
        with pytest.raises(ParameterError):
            StackShape(hidden=HIDDEN, ffn=24, tokens=TOKENS, layers=layers)  # type: ignore[arg-type]

    def test_minimum_layers_constant(self):
        """``MINIMUM_LAYERS`` 就是 1：它被写在错误信息里."""
        assert MINIMUM_LAYERS == 1

    def test_to_dict_and_summary_line(self):
        """两种输出都不含“只有内部才知道”的东西."""
        item = shape()
        payload = item.to_dict()
        assert payload["layers"] == LAYERS
        assert payload["total_parameter_count"] == 1944
        assert isinstance(json.dumps(payload), str)
        assert "1944" in item.summary_line()
        assert "4.0×" in item.summary_line()


class TestLayerCensus:
    """逐层读数的四个断言与两个派生量."""

    def test_gain_is_the_output_over_input(self):
        """增益 = ‖y‖ / ‖x‖（样本 1.0 → 2.0 时增益恰好 2.0）."""
        row = LayerCensus(
            index=0,
            input_norm=1.0,
            output_norm=2.0,
            branch1_norm=1.0,
            branch2_norm=0.5,
            max_abs_input=0.5,
        )
        assert row.gain == 2.0

    def test_gain_of_a_zero_input_is_zero_not_one(self):
        """入口范数为 0 时增益返回 ``0.0``：它在一张增益表里是**显眼的**."""
        row = LayerCensus(
            index=1,
            input_norm=0.0,
            output_norm=3.0,
            branch1_norm=0.0,
            branch2_norm=0.0,
            max_abs_input=0.0,
        )
        assert row.gain == 0.0
        assert row.carry_share == 0.0

    def test_carry_share_is_a_share(self):
        """直通占比落在一个类里：``‖x‖ / (‖x‖ + ‖b1‖ + ‖b2‖)``."""
        row = LayerCensus(
            index=2,
            input_norm=2.0,
            output_norm=1.0,
            branch1_norm=1.0,
            branch2_norm=1.0,
            max_abs_input=1.0,
        )
        assert row.carry_share == pytest.approx(0.5)

    def test_carry_share_of_all_zero_is_zero(self):
        """三个范数全为 0 时返回 0（而不是 nan 或 ZeroDivisionError）."""
        row = LayerCensus(
            index=0,
            input_norm=0.0,
            output_norm=0.0,
            branch1_norm=0.0,
            branch2_norm=0.0,
            max_abs_input=0.0,
        )
        assert row.carry_share == 0.0

    def test_carry_share_is_zero_when_the_residual_is_off(self):
        """残差关时直通占比**恰好**是 0：入口那一份确实没有被传下去."""
        row = LayerCensus(
            index=0,
            input_norm=2.0,
            output_norm=1.0,
            branch1_norm=1.0,
            branch2_norm=1.0,
            max_abs_input=1.0,
            use_residual=False,
        )
        assert row.carry_share == 0.0
        assert row.gain == 0.5

    def test_use_residual_must_be_a_boolean(self):
        """``use_residual`` 必须是布尔量（写成 ``"no"`` 时当场报错）."""
        with pytest.raises(ParameterError):
            LayerCensus(
                index=0,
                input_norm=1.0,
                output_norm=1.0,
                branch1_norm=0.0,
                branch2_norm=0.0,
                max_abs_input=1.0,
                use_residual="no",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("index", [-1, 1.5, True])
    def test_index_must_be_a_non_negative_integer(self, index: object):
        """层号必须是非负整数."""
        with pytest.raises(ParameterError):
            LayerCensus(
                index=index,  # type: ignore[arg-type]
                input_norm=1.0,
                output_norm=1.0,
                branch1_norm=0.0,
                branch2_norm=0.0,
                max_abs_input=1.0,
            )

    @pytest.mark.parametrize("name", ["input_norm", "output_norm", "branch1_norm", "branch2_norm", "max_abs_input"])
    def test_norms_must_be_finite_and_non_negative(self, name: str):
        """范数是“非负的有限数”——``-1`` 与 ``inf`` 都要被挡住."""
        payload = {
            "index": 0,
            "input_norm": 1.0,
            "output_norm": 1.0,
            "branch1_norm": 0.0,
            "branch2_norm": 0.0,
            "max_abs_input": 1.0,
        }
        payload[name] = -1.0
        with pytest.raises(NumericError):
            LayerCensus(**payload)  # type: ignore[arg-type]

    def test_to_dict_carries_the_two_derived_readings(self):
        """读数进字典时要连派生量一起给（否则读表的人要自己算）."""
        row = LayerCensus(
            index=0,
            input_norm=1.0,
            output_norm=1.0,
            branch1_norm=0.0,
            branch2_norm=0.0,
            max_abs_input=1.0,
        )
        payload = row.to_dict()
        assert set(payload) == {
            "index",
            "input_norm",
            "output_norm",
            "branch1_norm",
            "branch2_norm",
            "max_abs_input",
            "use_residual",
            "gain",
            "carry_share",
        }
        assert "层 0" in row.summary_line()


class TestStackedLayer:
    """链上的一环：账与参数必须来自同一个宽度."""

    def test_layer_matches_its_parameters(self):
        """样本的第 0 层：账里的形状与参数记录的宽度一致."""
        from smart_research_agent.transformer_stack import stack_forward

        params = parameters()
        forward = stack_forward(params, inputs())
        layer = forward.layer_at(0)
        assert isinstance(layer, StackedLayer)
        assert layer.hidden == HIDDEN == layer.params.hidden
        assert f"第 0 层" in layer.summary_line()

    def test_width_mismatch_is_rejected(self):
        """账的宽度与参数的宽度不一致时当场报错（而不是在反向里才炸）."""
        from smart_research_agent.transformer_stack import stack_forward

        width6 = stack_forward(parameters(hidden=6), inputs(hidden=6))
        with pytest.raises(ShapeError):
            StackedLayer(
                index=0,
                block=width6.layer_at(0).block,
                attention=width6.layer_at(0).attention,
                params=parameters(hidden=8, tokens=TOKENS).blocks[0],
            )

    @pytest.mark.parametrize("index", [-1, 2.5])
    def test_index_must_be_a_non_negative_integer(self, index: object):
        """层号必须是非负整数."""
        from smart_research_agent.transformer_stack import stack_forward

        layer = stack_forward(parameters(), inputs()).layer_at(0)
        with pytest.raises(ParameterError):
            StackedLayer(
                index=index,  # type: ignore[arg-type]
                block=layer.block,
                attention=layer.attention,
                params=layer.params,
            )


class TestStackParameters:
    """一摞参数：两份逐层对齐、压平可还原、可换掉某一层."""

    def test_sample_count_is_1944(self):
        """逐块数出来的参数量与解析式一致（这一步在性质 5 里也被查）."""
        params = parameters()
        assert params.layers == LAYERS
        assert params.parameter_count == 1944

    def test_shape_for_needs_tokens_from_the_caller(self):
        """形状只能由调用方补上 ``tokens``：参数里没有“序列长度”这个东西."""
        params = parameters()
        assert params.shape_for(TOKENS).total_parameter_count == 1944
        assert params.shape_for(9).tokens == 9

    def test_length_mismatch_is_a_shape_error(self):
        """两份参数份数不一致 ⇒ ``ShapeError``（少一份会在前向跑到那层时才炸）."""
        blocks = parameters().blocks
        with pytest.raises(ShapeError):
            StackParameters(blocks=blocks, attentions=parameters().attentions[:-1])

    def test_width_mismatch_is_an_assembly_error(self):
        """第 i 层的注意力投影列数与块宽度不一致 ⇒ ``AssemblyError``."""
        params = parameters()
        wrong = parameters(hidden=8, tokens=TOKENS)
        with pytest.raises(AssemblyError):
            StackParameters(blocks=params.blocks, attentions=wrong.attentions)

    def test_different_widths_between_layers_are_rejected(self):
        """链上每一层都必须同宽（第 0 层与第 1 层不同宽时当场报错）."""
        wide = parameters(hidden=8, tokens=TOKENS)
        narrow = parameters()
        with pytest.raises(AssemblyError):
            StackParameters(
                blocks=(narrow.blocks[0], wide.blocks[1]),
                attentions=(narrow.attentions[0], wide.attentions[1]),
            )

    def test_empty_is_rejected(self):
        """一摞参数至少要有一层."""
        with pytest.raises(ParameterError):
            StackParameters(blocks=(), attentions=())

    def test_flatten_round_trip(self):
        """压平再还原：整条链逐位回到原来（形状表是逐层的，因此不会串层）."""
        params = parameters()
        flat, shapes = params.flatten()
        assert params.block_parameter_count == 4 * 342 == 1368
        assert len(flat) == params.block_parameter_count
        assert len(shapes) == LAYERS
        rebuilt = StackParameters.unflatten(flat, shapes, params.attentions)
        assert rebuilt == params

    def test_flatten_leaves_the_attention_out(self):
        """注意力那四个投影不进 ``flatten``：它由 ``unflatten`` 原样带回."""
        params = parameters()
        flat, _shapes = params.flatten()
        assert len(flat) == params.parameter_count - 4 * 144
        assert params.parameter_count == 1944

    def test_unflatten_rejects_short_input(self):
        """少给数时报错（而不是把后面的层错位填上）."""
        params = parameters()
        flat, shapes = params.flatten()
        with pytest.raises(ShapeError):
            StackParameters.unflatten(flat[:-1], shapes, params.attentions)

    def test_unflatten_rejects_leftovers(self):
        """多给数时也报错（否则“剩下的没人要”会静默丢掉）."""
        params = parameters()
        flat, shapes = params.flatten()
        with pytest.raises(ShapeError):
            StackParameters.unflatten(flat + (0.0,), shapes, params.attentions)

    def test_unflatten_needs_a_shape_table(self):
        """空形状表没有意义."""
        with pytest.raises(ParameterError):
            StackParameters.unflatten((0.0,), (), parameters().attentions)

    def test_replace_layer_keeps_the_rest(self):
        """换掉第 2 层：其余层的参数逐位不动."""
        params = parameters()
        swapped = parameters(seed=99)
        updated = params.replace_layer(2, swapped.blocks[2])
        assert updated.blocks[2] == swapped.blocks[2]
        assert updated.blocks[0] == params.blocks[0]
        assert updated.blocks[3] == params.blocks[3]

    @pytest.mark.parametrize("index", [-1, LAYERS, 1.5])
    def test_replace_layer_checks_the_index(self, index: object):
        """层号越界当场报错（不许静默地改最后一层）."""
        params = parameters()
        with pytest.raises(ParameterError):
            params.replace_layer(index, params.blocks[0])  # type: ignore[arg-type]

    def test_replace_layer_checks_the_width(self):
        """换进来的参数宽度必须一致."""
        params = parameters()
        other = parameters(hidden=8, tokens=TOKENS)
        with pytest.raises(ShapeError):
            params.replace_layer(0, other.blocks[0])

    def test_summary_line_mentions_the_total(self):
        """一行说明里带着“每层 d / d_ff”与总量."""
        text = parameters().summary_line()
        assert "4 层" in text
        assert "1944" in text


class TestStackForwardRecord:
    """前向账的三条构造检查."""

    def test_sample_record_is_well_formed(self):
        """样本账：4 层、4 行读数、入口与出口同形."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        assert forward.depth == LAYERS
        assert forward.tokens == TOKENS
        assert len(forward.censuses) == LAYERS
        assert len(forward.gain_profile()) == LAYERS
        assert len(forward.carry_profile()) == LAYERS

    def test_input_output_shape_must_match(self):
        """链的输入与输出不同形 ⇒ ``ShapeError``（块保形 ⇒ 链保形）."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        with pytest.raises(ShapeError):
            StackForward(
                shape=shape(tokens=6),
                placement=forward.placement,
                use_residual=forward.use_residual,
                activation=forward.activation,
                inputs=forward.inputs,
                output=forward.output,
                layers=forward.layers,
                censuses=forward.censuses,
            )

    def test_layer_count_must_match_the_shape(self):
        """账的环数与形状说的层数不一致 ⇒ ``ShapeError``."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        with pytest.raises(ShapeError):
            StackForward(
                shape=shape(layers=2),
                placement=forward.placement,
                use_residual=forward.use_residual,
                activation=forward.activation,
                inputs=forward.inputs,
                output=forward.output,
                layers=forward.layers,
                censuses=forward.censuses,
            )

    def test_census_count_must_match(self):
        """读数行数缺一行 ⇒ ``ShapeError``（“第几层开始塌”就没有依据了）."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        with pytest.raises(ShapeError):
            StackForward(
                shape=forward.shape,
                placement=forward.placement,
                use_residual=forward.use_residual,
                activation=forward.activation,
                inputs=forward.inputs,
                output=forward.output,
                layers=forward.layers,
                censuses=forward.censuses[:-1],
            )

    def test_layer_index_must_equal_its_position(self):
        """账必须按层号顺序存放（把第 1 环的号写成 0 时当场报错）."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        broken = [forward.layers[0], forward.layers[1]]
        broken.append(
            StackedLayer(
                index=9,
                block=forward.layers[2].block,
                attention=forward.layers[2].attention,
                params=forward.layers[2].params,
            )
        )
        with pytest.raises(ShapeError):
            StackForward(
                shape=forward.shape,
                placement=forward.placement,
                use_residual=forward.use_residual,
                activation=forward.activation,
                inputs=forward.inputs,
                output=forward.output,
                layers=tuple(broken),
                censuses=forward.censuses,
            )

    def test_output_must_be_the_last_layers_output(self):
        """``output`` 必须是最后一层的输出（换一份张量时当场报错）."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        with pytest.raises(ShapeError):
            StackForward(
                shape=forward.shape,
                placement=forward.placement,
                use_residual=forward.use_residual,
                activation=forward.activation,
                inputs=forward.inputs,
                output=tuple(tuple(0.0 for _ in row) for row in forward.output),
                layers=forward.layers,
                censuses=forward.censuses,
            )

    def test_census_at_and_layer_at_check_bounds(self):
        """``census_at`` / ``layer_at`` 越界当场报错."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        assert forward.census_at(0).index == 0
        with pytest.raises(ParameterError):
            forward.census_at(LAYERS)
        with pytest.raises(ParameterError):
            forward.layer_at(-1)

    def test_notes_and_to_dict(self):
        """账进字典时**不含**每一环的九个中间矩阵（太大且无意义）."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        payload = forward.to_dict()
        assert set(payload) == {
            "shape",
            "placement",
            "use_residual",
            "activation",
            "inputs",
            "output",
            "censuses",
            "notes",
        }
        assert len(forward.notes) == 2
        assert "4 层" in forward.summary_line()

    def test_census_table_has_a_header_and_a_row_per_layer(self):
        """读数表：一行表头 + 分隔线 + 每层一行."""
        from smart_research_agent.transformer_stack import stack_forward

        forward = stack_forward(parameters(), inputs())
        table = forward.census_table()
        assert len(table) == LAYERS + 2
        assert "直通占比" in table[0]
        assert table[1].strip().startswith("-")


class TestStackGradientsRecord:
    """反向账的两条构造检查."""

    def test_sample_record(self):
        """样本梯度账：4 环、``grad_inputs`` 就是第 0 层的输入梯度."""
        from smart_research_agent.transformer_core.layers import mse_gradient
        from smart_research_agent.transformer_stack import stack_backward, stack_forward

        params = parameters()
        sample = inputs()
        goal = inputs()
        forward = stack_forward(params, sample)
        grads = stack_backward(forward, params, mse_gradient(forward.output, goal))
        assert grads.depth == LAYERS
        assert grads.layer_at(0).grad_inputs == grads.grad_inputs
        assert len(grads.norms()) == LAYERS
        assert isinstance(json.dumps(grads.to_dict()), str)

    def test_layer_count_must_match(self):
        """环数对不上 ⇒ ``ShapeError``."""
        from smart_research_agent.transformer_core.layers import mse_gradient
        from smart_research_agent.transformer_stack import stack_backward, stack_forward

        params = parameters()
        forward = stack_forward(params, inputs())
        grads = stack_backward(forward, params, mse_gradient(forward.output, inputs()))
        with pytest.raises(ShapeError):
            StackGradients(shape=shape(layers=2), grad_inputs=grads.grad_inputs, layers=grads.layers)

    def test_grad_inputs_must_be_the_first_layers_gradient(self):
        """``grad_inputs`` 换成别的矩阵 ⇒ ``ShapeError``（它会指两个不同的东西）."""
        from smart_research_agent.transformer_core.layers import mse_gradient
        from smart_research_agent.transformer_stack import stack_backward, stack_forward

        params = parameters()
        forward = stack_forward(params, inputs())
        grads = stack_backward(forward, params, mse_gradient(forward.output, inputs()))
        shifted = tuple(row[1:] + (0.0,) for row in grads.grad_inputs)
        with pytest.raises(ShapeError):
            StackGradients(shape=grads.shape, grad_inputs=shifted, layers=grads.layers)

    def test_layer_at_checks_bounds(self):
        """``layer_at`` 越界当场报错."""
        from smart_research_agent.transformer_core.layers import mse_gradient
        from smart_research_agent.transformer_stack import stack_backward, stack_forward

        params = parameters()
        forward = stack_forward(params, inputs())
        grads = stack_backward(forward, params, mse_gradient(forward.output, inputs()))
        with pytest.raises(ParameterError):
            grads.layer_at(LAYERS)

    def test_summary_line(self):
        """一行说明里有“逐层 ‖dx‖ 从…到…”."""
        from smart_research_agent.transformer_core.layers import mse_gradient
        from smart_research_agent.transformer_stack import stack_backward, stack_forward

        params = parameters()
        forward = stack_forward(params, inputs())
        grads = stack_backward(forward, params, mse_gradient(forward.output, inputs()))
        assert "逐层 ‖dx‖" in grads.summary_line()


class TestReadingsAndTables:
    """三个读数与三张表."""

    def test_frobenius_matches_hand_computation(self):
        """``(3, 4)`` 的范数是 5（毕达哥拉斯三元组，便于手算）."""
        matrix: Matrix = ((0.0, 3.0), (4.0, 0.0))
        assert frobenius(matrix) == pytest.approx(5.0)

    def test_frobenius_of_a_zero_matrix_is_zero(self):
        """零矩阵的范数是 0."""
        assert frobenius(((0.0, 0.0), (0.0, 0.0))) == 0.0

    def test_max_absolute_is_the_single_point_reading(self):
        """``max_absolute`` 是**单点**读数（与“平方和的根”不是一回事）."""
        assert max_absolute(((1.0, -3.0), (2.0, 0.0))) == 3.0

    def test_relative_matrix_error_is_zero_for_identical(self):
        """两个相同矩阵的相对误差是 0."""
        same: Matrix = ((1.0, 2.0), (3.0, 4.0))
        assert relative_matrix_error(same, same) == 0.0

    def test_relative_matrix_error_matches_the_shared_definition(self):
        """口径与 day075~079 共享实现：分母是参考的范数."""
        reference: Matrix = ((3.0, 4.0),)
        approximate: Matrix = ((0.0, 0.0),)
        assert relative_matrix_error(approximate, reference) == pytest.approx(1.0)

    def test_stage_table_is_closed(self):
        """五个阶段的名字与说明逐键对应（少一条说明就是一张有洞的表）."""
        assert set(STACK_STAGES) == set(STAGE_DESCRIPTIONS) == set(STAGE_SHAPES)
        assert STACK_STAGES == (
            STAGE_ENTER,
            STAGE_BLOCK,
            STAGE_CENSUS,
            STAGE_CARRY,
            STAGE_EXIT,
        )
        assert len(STACK_STAGES) == 5

    def test_property_table_is_closed(self):
        """六条性质的名字与说明逐键对应."""
        assert len(STACK_PROPERTIES) == 6
        assert set(STACK_PROPERTIES) == set(PROPERTY_DESCRIPTIONS)

    def test_gradient_target_table_is_closed(self):
        """两类报告的名单与 ``KINDS`` 逐键对应，且**不含**中间层的 'inputs'."""
        assert set(KINDS) == set(STACK_GRADIENT_TARGETS)
        assert STACK_GRADIENT_TARGETS[KIND_STACK] == (GRAD_STACK_INPUTS,)
        assert len(STACK_GRADIENT_TARGETS[KIND_LAYER]) == 8
        assert "inputs" not in LAYER_GRADIENT_TARGETS
        assert len(LAYER_GRADIENT_TARGETS) == 8

    def test_notes_are_three_boundaries(self):
        """三条边界（回答什么 / 不回答什么 / 读数不承诺什么）."""
        assert len(STACK_NOTES) == 3
        assert any("day081" in item for item in STACK_NOTES)

    def test_default_init_scale_is_shared_with_earlier_days(self):
        """初始化幅度与 day075~079 同值（0.25）."""
        assert DEFAULT_INIT_SCALE == 0.25
        assert math.isfinite(DEFAULT_INIT_SCALE)
