"""形状、前向九步与反向九步（day076 / M7-D2）.

这一份测试守三件事：

```text
① 两张表对得上         九阶段 / 五项梯度 / 六条性质 三张口径表逐键对齐
② 手算的期望值         单位矩阵配置下每一头的 weights 与 context 都能算出来
③ heads=1 的退化关系   **逐位**等于 day075 的自注意力（前向与反向都要）
```

第 ③ 条用的是 ``==`` 而不是容差：连续块划分在 ``heads=1`` 时的唯一划分就是整张矩阵，
因此两个实现给出的是同一串数。一条"逐位相等"的断言比"误差很小"难伪造得多。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.attention import scaling_factor
from smart_research_agent.math_foundations.errors import (
    NumericError as MathNumericError,
)
from smart_research_agent.math_foundations.errors import (
    ShapeError as MathShapeError,
)
from smart_research_agent.multi_head.errors import (
    FAMILY_OUTCOMES,
    GradientError,
    MultiHeadError,
    NumericError,
    ParameterError,
    PartitionError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import (
    head_parameter_gradients,
    head_scale,
    head_sequence,
    head_value_of,
    multi_head_attention,
    multi_head_backward,
    project_heads_separately,
)
from smart_research_agent.multi_head.types import (
    GRADIENT_TARGETS,
    MULTIHEAD_GRADIENT_DESCRIPTIONS,
    MULTIHEAD_GRADIENT_FORMULAS,
    MULTIHEAD_GRADIENT_TARGETS,
    MULTIHEAD_PROPERTIES,
    MULTIHEAD_PROPERTY_DESCRIPTIONS,
    MULTIHEAD_STAGE_DESCRIPTIONS,
    MULTIHEAD_STAGE_SHAPES,
    MULTIHEAD_STAGES,
    HeadGradients,
    HeadPartition,
    MultiHeadForward,
    MultiHeadShape,
    head_gradient_norms,
    head_gradient_shares,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    full_mask,
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.types import (
    AttentionParams,
    AttentionShape,
    ParameterGradients,
    relative_matrix_error,
    softmax_shape_scale,
)
from tests.multihead_samples import (
    HEAD_COUNTS,
    IDENTITY_INPUTS,
    IDENTITY_OUTPUT,
    IDENTITY_OUTPUT_FOUR_HEADS_ROW_ONE,
    P_SQRT2,
    P_THREE_SQRT2_HALF,
    approx,
    approx_matrix,
    head_shape,
    identity_parameters,
    induction_tasks,
    repeated_block_parameters,
    toy_parameters,
    witness_sample,
)


def _forward(
    heads: int = 2,
    *,
    causal: bool = True,
    params: AttentionParams | None = None,
    inputs: tuple[tuple[float, ...], ...] = IDENTITY_INPUTS,
) -> MultiHeadForward:
    """手算配置下的一次多头前向（测试里反复用到的那一个）."""
    return multi_head_attention(
        identity_parameters(4) if params is None else params,
        inputs,
        heads=heads,
        causal=causal,
    )


class TestTables:
    """三张口径表必须逐键对齐（少一个键不会让测试变红，只会让报告缺一节）."""

    def test_stage_tables_align(self):
        assert set(MULTIHEAD_STAGES) == set(MULTIHEAD_STAGE_DESCRIPTIONS)
        assert set(MULTIHEAD_STAGES) == set(MULTIHEAD_STAGE_SHAPES)

    def test_nine_stages_with_split_and_merge(self):
        assert MULTIHEAD_STAGES == (
            "project",
            "split",
            "score",
            "scale",
            "mask",
            "softmax",
            "mix",
            "merge",
            "output",
        )
        assert len(MULTIHEAD_STAGES) == 9

    def test_every_stage_has_a_description_and_a_shape(self):
        for stage in MULTIHEAD_STAGES:
            assert MULTIHEAD_STAGE_DESCRIPTIONS[stage]
            assert MULTIHEAD_STAGE_SHAPES[stage]

    def test_gradient_tables_align_and_share_day075_names(self):
        assert set(MULTIHEAD_GRADIENT_TARGETS) == set(MULTIHEAD_GRADIENT_FORMULAS)
        assert set(MULTIHEAD_GRADIENT_TARGETS) == set(MULTIHEAD_GRADIENT_DESCRIPTIONS)
        # 五项的名字**刻意与 day075 一样**（它们是同一个参数矩阵的梯度）
        assert MULTIHEAD_GRADIENT_TARGETS == GRADIENT_TARGETS

    def test_gradient_formulas_carry_the_head_sum(self):
        """公式里必须出现 Σ_h 或 split——否则它可能只是 day075 那一份被抄了过来."""
        joined = " ".join(MULTIHEAD_GRADIENT_FORMULAS.values())
        assert "Σ_h" in joined
        assert "split" in joined

    def test_property_tables_align(self):
        assert set(MULTIHEAD_PROPERTIES) == set(MULTIHEAD_PROPERTY_DESCRIPTIONS)
        assert len(MULTIHEAD_PROPERTIES) == 6

    def test_family_table_covers_every_error_class(self):
        for name in FAMILY_OUTCOMES:
            assert name in {
                "ShapeError",
                "ParameterError",
                "PartitionError",
                "NumericError",
                "GradientError",
            }

    def test_partition_error_is_a_parameter_error(self):
        """划分失败是**参数失败的特例**：`except ParameterError` 必须能兜住它."""
        assert issubclass(PartitionError, ParameterError)
        assert issubclass(PartitionError, MultiHeadError)

    def test_families_stay_inside_the_parent_families(self):
        """上游的 `except ShapeError`（day075 那一层）必须能兜住本层的形状错误."""
        assert issubclass(ShapeError, MultiHeadError)
        assert issubclass(NumericError, MultiHeadError)
        assert issubclass(GradientError, MultiHeadError)
        assert issubclass(MultiHeadError, ValueError)


class TestHeadPartition:
    """头划分：连续块、可往返、不可整除当场拒绝."""

    def test_width_and_bounds(self):
        partition = HeadPartition(6, 3)
        assert partition.width == 2
        assert partition.bounds == ((0, 2), (2, 4), (4, 6))

    def test_rows_of(self):
        assert HeadPartition(6, 3).rows_of(1) == (2, 3)

    def test_describe_is_printable(self):
        assert HeadPartition(6, 2).describe() == "6 维 → 2 头 × 3 维 | 边界 (0,3)、(3,6)"

    @pytest.mark.parametrize("total,heads", [(6, 4), (5, 2), (7, 3)])
    def test_indivisible_is_rejected(self, total, heads):
        with pytest.raises(PartitionError):
            HeadPartition(total, heads)

    @pytest.mark.parametrize("total,heads", [(0, 1), (6, 0), (-4, 2)])
    def test_non_positive_is_rejected(self, total, heads):
        with pytest.raises(ParameterError):
            HeadPartition(total, heads)

    @pytest.mark.parametrize("value", [True, 2.0, "3"])
    def test_non_integer_is_rejected(self, value):
        with pytest.raises(ParameterError):
            HeadPartition(value, 1)
        with pytest.raises(ParameterError):
            HeadPartition(4, value)

    def test_rows_round_trip(self):
        partition = HeadPartition(6, 3)
        matrix = tuple((float(row), float(row + 1)) for row in range(6))
        assert partition.merge_rows(partition.split_rows(matrix)) == matrix

    def test_columns_round_trip(self):
        partition = HeadPartition(6, 2)
        matrix = (tuple(float(column) for column in range(6)),)
        assert partition.merge_columns(partition.split_columns(matrix)) == matrix

    def test_vector_round_trip(self):
        partition = HeadPartition(6, 3)
        vector = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert partition.merge_vector(partition.split_vector(vector)) == vector

    def test_split_rows_checks_width(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).split_rows(((1.0, 2.0),))

    def test_split_columns_checks_width(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).split_columns(((1.0, 2.0),))

    def test_merge_rows_counts_blocks(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).merge_rows([((1.0,),), ((2.0,),)])

    def test_merge_rows_checks_every_block_shape(self):
        """第一块合法、后面某块的**行数**不对——第 0 块之后的那条校验."""
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).merge_rows(
                [((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0),), ((7.0, 8.0), (9.0, 10.0))]
            )

    def test_merge_rows_checks_block_shape(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).merge_rows([((1.0,),), ((2.0,),), ((3.0, 4.0), 5.0)])

    def test_merge_columns_checks_block_shape(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 2).merge_columns([((1.0,),), ((2.0, 3.0, 4.0),)])

    def test_split_vector_checks_length(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).split_vector((1.0, 2.0))

    def test_merge_vector_checks_width(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).merge_vector([(1.0, 2.0), (3.0, 4.0), (5.0,)])

    def test_merge_vector_counts_segments(self):
        with pytest.raises(ShapeError):
            HeadPartition(6, 3).merge_vector([(1.0, 2.0), (3.0, 4.0)])

    @pytest.mark.parametrize("head", [-1, 2])
    def test_head_index_is_checked(self, head):
        with pytest.raises(ParameterError):
            HeadPartition(6, 2).rows_of(head)

    def test_head_index_must_be_an_integer(self):
        with pytest.raises(ParameterError):
            HeadPartition(6, 2).rows_of(True)


class TestMultiHeadShape:
    """形状：四个维度 + heads，五个派生量与它们各自的来源."""

    def test_derived_dimensions(self):
        shape = head_shape(6, 2)
        assert shape.head_dim == 3
        assert shape.head_value == 3
        assert shape.attention.keys == 6

    def test_scale_uses_head_dim(self):
        shape = head_shape(6, 2)
        assert approx(shape.scale, 1.0 / math.sqrt(3.0))
        assert approx(shape.single_head_scale, 1.0 / math.sqrt(6.0))
        assert not approx(shape.scale, shape.single_head_scale)

    def test_single_head_scale_equals_classic(self):
        shape = head_shape(6, 1)
        assert shape.scale == shape.single_head_scale

    @pytest.mark.parametrize("heads", HEAD_COUNTS + (6,))
    def test_scale_ratio_is_the_square_root_of_heads(self, heads):
        """**派生量必须等于 √heads**：它是这一课的代数结论，不是一句口号."""
        shape = head_shape(6, heads)
        assert approx(shape.scale_ratio, math.sqrt(heads))

    def test_to_dict_contains_the_ratio(self):
        payload = head_shape(6, 3).to_dict()
        assert payload["heads"] == 3
        assert payload["head_dim"] == 2
        assert approx(payload["scale_ratio"], math.sqrt(3.0))
        assert payload["attention"]["keys"] == 6

    def test_summary_line_is_readable(self):
        line = head_shape(6, 2).summary_line()
        assert "2 头" in line
        assert "0.577350" in line

    def test_attention_must_be_a_shape(self):
        with pytest.raises(ShapeError):
            MultiHeadShape(attention=(1, 2), heads=1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("heads", [0, -2])
    def test_non_positive_heads_is_rejected(self, heads):
        with pytest.raises(ParameterError):
            MultiHeadShape(AttentionShape(6, 6, 6, 6), heads)

    def test_heads_must_be_an_integer(self):
        with pytest.raises(ParameterError):
            MultiHeadShape(AttentionShape(6, 6, 6, 6), 2.0)  # type: ignore[arg-type]

    def test_keys_must_be_divisible(self):
        with pytest.raises(PartitionError):
            MultiHeadShape(AttentionShape(6, 5, 6, 6), 2)

    def test_values_must_be_divisible(self):
        with pytest.raises(PartitionError):
            MultiHeadShape(AttentionShape(6, 6, 5, 6), 2)

    def test_partitions_carry_the_widths(self):
        shape = head_shape(6, 3)
        assert shape.keys_partition.bounds == ((0, 2), (2, 4), (4, 6))
        assert shape.values_partition.width == 2


class TestHeadScale:
    """缩放系数：1/√head_dim，且与 day075 的同名实现的**同一个输入**给出同一个数."""

    def test_value(self):
        assert approx(head_scale(4), 0.5)

    @pytest.mark.parametrize("head_dim", [1, 2, 3, 6, 12])
    def test_matches_day075_helper_bit_for_bit(self, head_dim):
        assert head_scale(head_dim) == softmax_shape_scale(head_dim)
        assert head_scale(head_dim) == scaling_factor(head_dim)

    @pytest.mark.parametrize("head_dim", [0, -1])
    def test_non_positive_is_rejected(self, head_dim):
        with pytest.raises(ParameterError):
            head_scale(head_dim)

    @pytest.mark.parametrize("value", [True, 2.0, None])
    def test_non_integer_is_rejected(self, value):
        with pytest.raises(ParameterError):
            head_scale(value)  # type: ignore[arg-type]


class TestHeadSequence:
    """头号规范：一个整数展开成 `0..n−1`，一个列表去重后保序."""

    def test_integer_expands(self):
        assert head_sequence(3) == (0, 1, 2)

    def test_list_is_preserved(self):
        assert head_sequence([2, 0]) == (2, 0)

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_count_is_rejected(self, value):
        with pytest.raises(ParameterError):
            head_sequence(value)

    def test_duplicates_are_rejected(self):
        with pytest.raises(ParameterError):
            head_sequence([0, 0])

    def test_empty_list_is_rejected(self):
        with pytest.raises(ParameterError):
            head_sequence([])

    def test_negative_head_is_rejected(self):
        with pytest.raises(ParameterError):
            head_sequence([-1])

    def test_non_integer_head_is_rejected(self):
        with pytest.raises(ParameterError):
            head_sequence([1.5])  # type: ignore[list-item]


class TestForwardHandComputed:
    """前向九步：单位矩阵配置下每一头、每一行都能手算."""

    def test_heads_two_output_matches_hand_computation(self):
        forward = _forward(2)
        assert approx_matrix(forward.output, IDENTITY_OUTPUT)
        assert approx_matrix(forward.merged_context, IDENTITY_OUTPUT)

    def test_heads_two_head_zero_weights(self):
        forward = _forward(2)
        assert approx_matrix(
            forward.head_weights[0],
            ((1.0, 0.0), (P_SQRT2, 1.0 - P_SQRT2)),
        )

    def test_heads_two_head_one_weights(self):
        forward = _forward(2)
        assert approx_matrix(
            forward.head_weights[1],
            ((1.0, 0.0), (P_THREE_SQRT2_HALF, 1.0 - P_THREE_SQRT2_HALF)),
        )

    def test_heads_two_head_contexts(self):
        forward = _forward(2)
        assert approx_matrix(
            forward.head_contexts[0],
            ((1.0, 0.0), (P_SQRT2, 1.0 - P_SQRT2)),
        )
        assert approx_matrix(
            forward.head_contexts[1],
            ((1.0, 0.0), (2.0 - P_THREE_SQRT2_HALF, 1.0 - P_THREE_SQRT2_HALF)),
        )

    def test_heads_four_hand_computed_row(self):
        """``head_dim = 1`` 是边界情形（缩放系数恰好 1），同样可手算."""
        forward = _forward(4)
        assert approx_matrix(
            (forward.output[1],),
            (IDENTITY_OUTPUT_FOUR_HEADS_ROW_ONE,),
        )

    def test_heads_one_reproduces_day075_bit_for_bit(self):
        """**逐位相等**：连续块在 heads=1 时的唯一划分就是整张矩阵."""
        params = identity_parameters(4)
        forward = _forward(1, params=params)
        classic = self_attention(params, IDENTITY_INPUTS, causal=True)
        assert forward.output == classic.output
        assert forward.head_weights[0] == classic.weights
        assert forward.merged_context == classic.context
        assert forward.head_contexts[0] == classic.context

    def test_weight_blocks_are_columns_of_the_activations(self):
        """**权重按行切、激活按列切**：两张表必须各自往返回去."""
        params = toy_parameters(6)
        forward = multi_head_attention(params, induction_tasks(1)[0].inputs, heads=3)
        keys_partition = forward.shape.keys_partition
        values_partition = forward.shape.values_partition
        assert keys_partition.merge_columns(forward.head_queries) == forward.queries
        assert keys_partition.merge_columns(forward.head_keys) == forward.keys
        assert values_partition.merge_columns(forward.head_values) == forward.values
        assert keys_partition.merge_rows(keys_partition.split_rows(params.w_query)) == (
            params.w_query
        )

    def test_activation_split_by_rows_would_be_a_shape_error(self):
        """反过来说：把激活按**行**切会立刻报错——这就是两条口径的差别."""
        params = toy_parameters(4)
        forward = multi_head_attention(params, IDENTITY_INPUTS, heads=2)
        with pytest.raises(ShapeError):
            forward.shape.keys_partition.split_rows(forward.queries)

    def test_merged_context_is_the_column_merge_of_head_contexts(self):
        forward = _forward(2)
        merged = forward.shape.values_partition.merge_columns(forward.head_contexts)
        assert merged == forward.merged_context

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_every_head_row_is_a_distribution(self, heads):
        params = toy_parameters(6)
        forward = multi_head_attention(params, induction_tasks(1)[0].inputs, heads=heads)
        assert forward.distributions == heads * forward.tokens
        for head in forward.head_weights:
            for row in head:
                assert approx(math.fsum(row), 1.0)

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_causal_no_leak_in_every_head(self, heads):
        params = toy_parameters(6)
        forward = multi_head_attention(
            params, induction_tasks(1)[0].inputs, heads=heads, causal=True
        )
        for head in forward.head_weights:
            for row in range(forward.tokens):
                for column in range(row + 1, forward.tokens):
                    assert head[row][column] == 0.0

    def test_full_mask_matches_day075(self):
        """不开掩码时（``causal=False``）heads=1 **逐位**等于 day075."""
        params = toy_parameters(6)
        inputs = induction_tasks(1)[0].inputs
        forward = multi_head_attention(params, inputs, heads=1, causal=False)
        classic = self_attention(params, inputs, causal=False)
        assert forward.mask == full_mask(forward.tokens)
        assert forward.head_weights[0] == classic.weights
        assert forward.output == classic.output

    def test_full_mask_is_not_all_zero(self):
        """全开掩码下**每一行都能看到所有位置**（峰值权重因此更小）."""
        params = toy_parameters(6)
        inputs = induction_tasks(1)[0].inputs
        causal = multi_head_attention(params, inputs, heads=2, causal=True)
        full = multi_head_attention(params, inputs, heads=2, causal=False)
        assert all(all(row) for row in full.mask)
        assert full.mean_peak_weight < causal.mean_peak_weight

    def test_explicit_mask_is_used(self):
        params = toy_parameters(4)
        mask = (
            (True, False, False, False),
            (True, True, False, False),
            (True, False, True, False),
            (True, True, True, True),
        )
        four_rows = IDENTITY_INPUTS + IDENTITY_INPUTS
        forward = multi_head_attention(params, four_rows, heads=2, mask=mask)
        assert forward.mask == mask

    def test_causal_with_explicit_mask_is_rejected(self):
        """这一条拒绝属于 **day075 那一层**（掩码的解析在那里），因此抛的是它的族."""
        params = toy_parameters(4)
        mask = ((True, True), (True, True))
        with pytest.raises(MathShapeError, match="不能同时给"):
            multi_head_attention(
                params, IDENTITY_INPUTS, heads=2, causal=True, mask=mask
            )

    def test_zero_row_is_rejected_like_day075(self):
        """两层对**同一批数据**必须给出同一族的拒绝（一个改数据、一个改训练）."""
        params = toy_parameters(4)
        bad_inputs = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
        with pytest.raises(MathNumericError):
            multi_head_attention(params, bad_inputs, heads=2)
        with pytest.raises(MathNumericError):
            self_attention(params, bad_inputs)

    def test_parameters_must_be_attention_params(self):
        with pytest.raises(ShapeError):
            multi_head_attention("not-params", IDENTITY_INPUTS, heads=2)  # type: ignore[arg-type]

    def test_zero_rows_in_the_weights_are_allowed(self):
        """参数里的全零行是初始化的自由——只有**数据**里的全零行才拒绝."""
        witness = witness_sample()
        forward = multi_head_attention(witness.params, witness.inputs, heads=2)
        assert forward.tokens == 4


class TestForwardReadouts:
    """前向报告的派生读数（它们必须能被独立读出，而不是只能看整张表）."""

    def test_counts(self):
        forward = _forward(2)
        assert forward.tokens == 2
        assert forward.heads == 2
        assert forward.distributions == 4

    def test_head_weight_row(self):
        forward = _forward(2)
        assert forward.head_weight_row(0, 1) == forward.head_weights[0][1]

    @pytest.mark.parametrize("head,row", [(2, 0), (0, 2), (0, -1)])
    def test_head_weight_row_range(self, head, row):
        with pytest.raises(ParameterError):
            _forward(2).head_weight_row(head, row)

    def test_peak_weights_of_head(self):
        forward = _forward(2)
        assert forward.peak_weights_of_head(1) == forward.head_peaks[1]

    def test_peak_weights_of_head_range(self):
        with pytest.raises(ParameterError):
            _forward(2).peak_weights_of_head(5)

    def test_head_summary_line(self):
        line = _forward(2).head_summary_line(1)
        assert "head 1" in line
        assert "argmax" in line

    def test_weights_tensor(self):
        forward = _forward(2)
        assert forward.weights_tensor() == forward.head_weights

    def test_entropy_readouts(self):
        forward = _forward(2)
        assert forward.max_entropy() == math.log(2.0)
        assert 0.0 <= forward.mean_entropy <= forward.max_entropy()
        assert 0.0 <= forward.mean_peak_weight <= 1.0
        assert 0.0 <= forward.focus_ratio() <= 1.0

    def test_to_dict_is_json_ready(self):
        payload = _forward(2).to_dict()
        assert payload["heads"] == 2
        assert len(payload["head_weights"]) == 2
        assert payload["distributions"] == 4
        assert payload["scale"] == 1.0 / math.sqrt(2.0)

    def test_summary_line(self):
        line = _forward(2).summary_line()
        assert "causal" in line
        assert "2 头" in line


class TestForwardValidation:
    """前向记录的构造校验：少一项、形状不符都要当场拒绝."""

    def _forward_kwargs(self) -> dict[str, object]:
        forward = _forward(2)
        return {
            "params": forward.params,
            "shape": forward.shape,
            "inputs": forward.inputs,
            "queries": forward.queries,
            "keys": forward.keys,
            "values": forward.values,
            "head_queries": forward.head_queries,
            "head_keys": forward.head_keys,
            "head_values": forward.head_values,
            "head_raw": forward.head_raw,
            "head_scores": forward.head_scores,
            "head_weights": forward.head_weights,
            "head_contexts": forward.head_contexts,
            "merged_context": forward.merged_context,
            "output": forward.output,
            "causal": forward.causal,
            "mask": forward.mask,
            "scale": forward.scale,
            "head_entropies": forward.head_entropies,
            "head_peaks": forward.head_peaks,
            "head_indices": forward.head_indices,
        }

    def test_params_must_be_attention_params(self):
        kwargs = self._forward_kwargs()
        kwargs["params"] = "no"
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_params_shape_must_match(self):
        kwargs = self._forward_kwargs()
        kwargs["params"] = identity_parameters(6)
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        [
            "head_queries",
            "head_keys",
            "head_values",
            "head_raw",
            "head_scores",
            "head_weights",
            "head_contexts",
            "head_entropies",
            "head_peaks",
            "head_indices",
        ],
    )
    def test_every_head_group_must_be_complete(self, field):
        kwargs = self._forward_kwargs()
        kwargs[field] = tuple(kwargs[field])[:1]  # type: ignore[arg-type]
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_weights_must_be_square(self):
        kwargs = self._forward_kwargs()
        kwargs["head_weights"] = (((1.0, 0.0, 0.0),),) * 2
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_per_head_vectors_must_match_the_row_count(self):
        """每个头内部的**派生量表**长度也要等于行数（不是只看头的个数）."""
        kwargs = self._forward_kwargs()
        kwargs["head_entropies"] = (
            tuple(kwargs["head_entropies"])[0][:1],
            tuple(kwargs["head_entropies"])[1],
        )
        with pytest.raises(ShapeError, match="head_entropies"):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_head_shape_must_match_the_partition(self):
        kwargs = self._forward_kwargs()
        kwargs["head_queries"] = (kwargs["head_queries"][0], ((1.0,), (2.0,)))
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_merged_context_shape_is_checked(self):
        kwargs = self._forward_kwargs()
        kwargs["merged_context"] = ((1.0, 2.0), (3.0, 4.0))
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_output_row_count_is_checked(self):
        kwargs = self._forward_kwargs()
        kwargs["output"] = ((1.0, 2.0, 3.0, 4.0),)
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_mask_shape_is_checked(self):
        kwargs = self._forward_kwargs()
        kwargs["mask"] = ((True,), (True,))
        with pytest.raises(ShapeError):
            MultiHeadForward(**kwargs)  # type: ignore[arg-type]

    def test_notes_are_stringified(self):
        kwargs = self._forward_kwargs()
        kwargs["notes"] = (1, 2)
        assert MultiHeadForward(**kwargs).notes == ("1", "2")  # type: ignore[arg-type]


class TestBackward:
    """反向九步：heads=1 逐位退化、逐头梯度按行拼得回融合梯度."""

    def test_heads_one_reproduces_day075_bit_for_bit(self):
        params = identity_parameters(4)
        forward = _forward(1, params=params)
        classic = self_attention(params, IDENTITY_INPUTS, causal=True)
        target = ((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0))
        mine = multi_head_backward(forward, mse_gradient(forward.output, target))
        theirs = attention_backward(classic, mse_gradient(classic.output, target))
        assert mine.flatten() == theirs.flatten()
        assert mine.grad_inputs == theirs.grad_inputs

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_parameter_gradient_shapes_match_parameters(self, heads):
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=heads)
        gradients = multi_head_backward(
            forward, mse_gradient(forward.output, task.target)
        )
        for matrix, weight in zip(
            gradients.matrices(), params.matrices(), strict=True
        ):
            assert len(matrix) == len(weight)
            assert len(matrix[0]) == len(weight[0])
        assert len(gradients.grad_inputs) == forward.tokens

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_head_gradients_assemble_into_the_merged_gradients(self, heads):
        """**每一行只由一头贡献**——因此拼接必须是逐位相等，而不是"误差很小"."""
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=heads)
        gradients = multi_head_backward(
            forward, mse_gradient(forward.output, task.target)
        )
        per_head = head_parameter_gradients(
            forward, mse_gradient(forward.output, task.target)
        )
        keys_partition = forward.shape.keys_partition
        values_partition = forward.shape.values_partition
        assert len(per_head) == heads
        assert keys_partition.merge_rows(
            [item.grad_w_query for item in per_head]
        ) == gradients.grad_w_query
        assert keys_partition.merge_rows([item.grad_w_key for item in per_head]) == (
            gradients.grad_w_key
        )
        assert values_partition.merge_rows(
            [item.grad_w_value for item in per_head]
        ) == gradients.grad_w_value

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_input_gradient_is_the_sum_of_all_heads(self, heads):
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=heads)
        gradient_matrix = mse_gradient(forward.output, task.target)
        per_head = head_parameter_gradients(forward, gradient_matrix)
        merged = multi_head_backward(forward, gradient_matrix)
        summed = tuple(
            tuple(
                math.fsum(part.grad_inputs[row][column] for part in per_head)
                for column in range(len(merged.grad_inputs[0]))
            )
            for row in range(len(merged.grad_inputs))
        )
        assert summed == merged.grad_inputs

    def test_heads_are_not_identical_when_they_should_not_be(self):
        """两个头拿到的梯度**一般不同**——否则"分头"这件事就没有发生."""
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=2)
        per_head = head_parameter_gradients(
            forward, mse_gradient(forward.output, task.target)
        )
        assert per_head[0].grad_w_query != per_head[1].grad_w_query

    def test_grad_output_shape_is_checked(self):
        forward = _forward(2)
        with pytest.raises(ShapeError):
            multi_head_backward(forward, ((1.0, 2.0, 3.0, 4.0),))

    def test_per_head_gradients_check_the_same_shape(self):
        """逐头入口与融合入口必须**各自**校验形状（两条入口都不能少）."""
        forward = _forward(2)
        with pytest.raises(ShapeError):
            head_parameter_gradients(forward, ((1.0, 2.0, 3.0, 4.0),))

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_project_heads_separately_matches_the_output(self, heads):
        """`merge → project` 与"逐头投影再相加"是**同一个式子**（浮点下差 1e-16）."""
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=heads)
        gap = relative_matrix_error(project_heads_separately(forward), forward.output)
        assert gap <= 1e-12

    def test_project_heads_separately_is_exact_for_a_single_head(self):
        """单头时两种写法**一模一样**（没有求和顺序可换）——因此是逐位相等."""
        forward = _forward(1)
        assert project_heads_separately(forward) == forward.output

    def test_head_value_of(self):
        forward = _forward(2)
        assert head_value_of(forward, 1, 1) == forward.head_values[1][1]

    @pytest.mark.parametrize("head,position", [(2, 0), (0, 2), (-1, 0)])
    def test_head_value_of_range(self, head, position):
        with pytest.raises(ParameterError):
            head_value_of(_forward(2), head, position)

    def test_head_value_of_head_must_be_integer(self):
        with pytest.raises(ParameterError):
            head_value_of(_forward(2), True, 0)


class TestHeadGradientsRecord:
    """逐头梯度记录：范数、打印与形状校验."""

    def _per_head(self) -> tuple[HeadGradients, ...]:
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=3)
        return head_parameter_gradients(
            forward, mse_gradient(forward.output, task.target)
        )

    def test_parameter_matrices_are_three(self):
        item = self._per_head()[0]
        assert len(item.parameter_matrices()) == 3

    def test_norm_is_positive_and_finite(self):
        for item in self._per_head():
            assert item.norm() > 0.0
            assert math.isfinite(item.norm())

    def test_summary_line(self):
        line = self._per_head()[2].summary_line()
        assert "head 2" in line
        assert "范数" in line

    def test_ragged_matrix_is_rejected(self):
        """形状校验来自 ``math_foundations``，因此抛的是**上一层那一族**."""
        with pytest.raises(MathShapeError):
            HeadGradients(
                head=0,
                grad_w_query=((1.0, 2.0), (3.0,)),
                grad_w_key=((1.0,),),
                grad_w_value=((1.0,),),
                grad_inputs=((1.0,),),
            )


class TestHeadGradientAccounting:
    """"哪一头在学"这个读数：范数、占比，以及它们的边界情形."""

    def _gradients(self, heads: int = 2) -> tuple[ParameterGradients, MultiHeadShape]:
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=heads)
        gradients = multi_head_backward(
            forward, mse_gradient(forward.output, task.target)
        )
        return gradients, forward.shape

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_norms_have_one_entry_per_head(self, heads):
        gradients, shape = self._gradients(heads)
        norms = head_gradient_norms(gradients, shape)
        assert len(norms) == heads
        assert all(value > 0.0 for value in norms)

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_shares_sum_to_one(self, heads):
        gradients, shape = self._gradients(heads)
        shares = head_gradient_shares(gradients, shape)
        assert approx(math.fsum(shares), 1.0)

    def test_zero_gradients_give_zero_shares(self):
        """全零梯度是一个**真实的观测**（没有头在动），不是一个需要捕获的错误."""
        gradients, shape = self._gradients(2)
        zeroed = ParameterGradients(
            grad_w_query=tuple(tuple(0.0 for _ in row) for row in gradients.grad_w_query),
            grad_w_key=gradients.grad_w_key,
            grad_w_value=gradients.grad_w_value,
            grad_w_output=gradients.grad_w_output,
            grad_inputs=gradients.grad_inputs,
        )
        zeros = ParameterGradients(
            grad_w_query=tuple(tuple(0.0 for _ in row) for row in gradients.grad_w_query),
            grad_w_key=tuple(tuple(0.0 for _ in row) for row in gradients.grad_w_key),
            grad_w_value=tuple(tuple(0.0 for _ in row) for row in gradients.grad_w_value),
            grad_w_output=gradients.grad_w_output,
            grad_inputs=gradients.grad_inputs,
        )
        assert head_gradient_shares(zeros, shape) == (0.0, 0.0)
        assert any(value > 0.0 for value in head_gradient_shares(zeroed, shape))

    def test_shape_type_is_checked(self):
        gradients, _shape = self._gradients(2)
        with pytest.raises(ShapeError):
            head_gradient_norms(gradients, "shape")  # type: ignore[arg-type]

    def test_non_finite_gradients_are_rejected(self):
        """NaN 在**更早一层**就被拒绝（分块要先把矩阵读一遍）——因此抛的是那一族.

        本地再写一遍有限性检查是多余的：``HeadPartition.split_rows`` 里的
        ``validate_matrix`` 已经先拒了，而**一个永远走不到的分支**在覆盖率上
        看起来与"防御写得全"一模一样。
        """
        gradients, shape = self._gradients(2)
        broken = ParameterGradients(
            grad_w_query=(
                (float("nan"),) + gradients.grad_w_query[0][1:],
            )
            + gradients.grad_w_query[1:],
            grad_w_key=gradients.grad_w_key,
            grad_w_value=gradients.grad_w_value,
            grad_w_output=gradients.grad_w_output,
            grad_inputs=gradients.grad_inputs,
        )
        with pytest.raises(MathNumericError):
            head_gradient_norms(broken, shape)

    def test_norms_match_the_per_head_records(self):
        """两处算法必须对账：逐头记录的范数之和应等于融合梯度的分块范数（同口径）."""
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        forward = multi_head_attention(params, task.inputs, heads=2)
        gradient_matrix = mse_gradient(forward.output, task.target)
        per_head = head_parameter_gradients(forward, gradient_matrix)
        merged = multi_head_backward(forward, gradient_matrix)
        merged_norms = head_gradient_norms(merged, forward.shape)
        for index, item in enumerate(per_head):
            assert approx(item.norm(), merged_norms[index], tolerance=1e-12)


class TestDegenerateConfiguration:
    """"每一头完全相同"的参数：头间差异必须是**恰好 0**，而前向仍然正确."""

    def test_repeated_blocks_are_accepted(self):
        params = repeated_block_parameters(6, 2)
        assert params.shape.summary_line().startswith("d_in=6 d_k=6 d_v=6 d_out=6")

    def test_repeated_blocks_give_identical_heads(self):
        params = repeated_block_parameters(6, 2)
        forward = multi_head_attention(params, induction_tasks(1)[0].inputs, heads=2)
        assert forward.head_weights[0] == forward.head_weights[1]
        assert forward.head_contexts[0] == forward.head_contexts[1]
        assert forward.head_queries[0] == forward.head_queries[1]

    def test_two_heads_cost_the_same_as_one(self):
        """参数量与 heads **无关**——这是"同参数量对照"的物理基础."""
        params = toy_parameters(6)
        counts = {MultiHeadShape(params.shape, heads).heads for heads in HEAD_COUNTS}
        assert counts == {1, 2, 3}
        assert params.parameter_count() == 4 * 6 * 6
