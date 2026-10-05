"""造表、形状、注入、前向六阶段与反向（day078 / M7-D3）.

这一份测试守三件事：

```text
① 三张口径表逐键对齐     六个阶段 / 六项梯度 / 七条性质，少一个键不会让报告变红
② 手算的期望值            d = 4、base = 10000 时 PE(0)/PE(1) 与每行范数都能在纸上算出来
③ 一行加法的两条护栏      注入是逐位相加，而表的梯度必须**按位置累加**（不是按行覆盖）
```

第 ③ 条用的是 ``==`` 而不是容差：加法注入的偏导数恰好是 ``1``，
因此 ``dx`` 与 ``dInjected`` **逐位相等**——一条"逐位相等"的断言比"误差很小"难伪造得多。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.attention import POSITIONAL_BASE as DAY073_BASE
from smart_research_agent.math_foundations.errors import ShapeError as MathShapeError
from smart_research_agent.positional_encoding import (
    ENCODING_LEARNABLE,
    ENCODING_SINUSOIDAL,
    FAMILY_OUTCOMES,
    GRAD_INPUTS,
    GRAD_TABLE,
    GRAD_W_KEY,
    GRAD_W_OUTPUT,
    GRAD_W_QUERY,
    GRAD_W_VALUE,
    GRADIENT_TOLERANCE,
    INIT_RANDOM,
    OFFSET_TOLERANCE,
    PAIRING_ALIGNED,
    PAIRING_DESCRIPTIONS,
    PAIRING_STAGGERED,
    POSITIONAL_BASE,
    POSITIONAL_GRADIENT_DESCRIPTIONS,
    POSITIONAL_GRADIENT_FORMULAS,
    POSITIONAL_GRADIENT_TARGETS,
    POSITIONAL_PROPERTIES,
    POSITIONAL_PROPERTY_DESCRIPTIONS,
    POSITIONAL_STAGE_DESCRIPTIONS,
    POSITIONAL_STAGE_SHAPES,
    POSITIONAL_STAGES,
    ROW_NORM_TOLERANCE,
    STAGE_ADD,
    STAGE_TABLE,
    EncodingTable,
    GradientError,
    NumericError,
    ParameterError,
    PositionalError,
    PositionalForward,
    PositionalGradients,
    PositionalShape,
    RangeError,
    ShapeError,
    add_matrices,
    batch_accuracy,
    batch_loss,
    batch_mean_injection_ratio,
    check_rows_are_aligned,
    closed_form_offset_inner,
    default_positions,
    flatten_parameters,
    frequency_of,
    gather_rows,
    inject,
    learnable_table,
    loss_gradient,
    matrix_row_norms,
    positional_backward,
    positional_forward,
    rotate,
    rotation_factor,
    scatter_add_rows,
    sinusoidal_row,
    sinusoidal_table,
    unflatten_parameters,
    validate_positions,
    validate_settings,
)
from smart_research_agent.positional_encoding.types import relative_matrix_error
from smart_research_agent.transformer_core.errors import ShapeError as CoreShapeError
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    masked_mean_squared_error,
    mean_squared_error,
    self_attention,
)
from tests.position_samples import (
    DIMENSION,
    FREQUENCY_PAIR_ONE,
    FREQUENCY_PAIR_ZERO,
    LENGTH,
    OFFSET_INNER_ONE,
    ONES_INJECTED_ROW_ZERO,
    ONES_INPUTS,
    ONES_MEAN_INJECTION_RATIO,
    PE_ROW_ONE,
    PE_ROW_ZERO,
    POSITIONAL_BASE_VALUE,
    ROW_NORM_TWO,
    STAGGERED_ROW_ONE,
    VOCABULARY,
    approx,
    approx_matrix,
    learnable,
    position_parameters,
    readout_orbit,
    sinusoidal,
    staggered,
    zeros,
)


class TestTables:
    """三张口径表必须逐键对齐（少一个键不会让测试变红，只会让报告缺一节）."""

    def test_stage_tables_align(self):
        assert set(POSITIONAL_STAGES) == set(POSITIONAL_STAGE_DESCRIPTIONS)
        assert set(POSITIONAL_STAGES) == set(POSITIONAL_STAGE_SHAPES)

    def test_six_stages_in_data_order(self):
        """顺序就是数据流的顺序：造表与相加**在最前面**，因此反向时它们在最后面."""
        assert POSITIONAL_STAGES == (
            "positions",
            "table",
            "add",
            "attend",
            "project",
            "loss",
        )
        assert len(POSITIONAL_STAGES) == 6

    def test_every_stage_has_a_description_and_a_shape(self):
        for stage in POSITIONAL_STAGES:
            assert POSITIONAL_STAGE_DESCRIPTIONS[stage]
            assert POSITIONAL_STAGE_SHAPES[stage]

    def test_two_new_stages_are_described_as_such(self):
        assert "注入" in POSITIONAL_STAGE_DESCRIPTIONS[STAGE_ADD]
        assert "位置 → 一行" in POSITIONAL_STAGE_DESCRIPTIONS[STAGE_TABLE]

    def test_gradient_tables_align_and_add_the_table(self):
        assert set(POSITIONAL_GRADIENT_TARGETS) == set(POSITIONAL_GRADIENT_FORMULAS)
        assert set(POSITIONAL_GRADIENT_TARGETS) == set(POSITIONAL_GRADIENT_DESCRIPTIONS)
        assert POSITIONAL_GRADIENT_TARGETS == (
            GRAD_W_OUTPUT,
            GRAD_W_VALUE,
            GRAD_W_QUERY,
            GRAD_W_KEY,
            GRAD_TABLE,
            GRAD_INPUTS,
        )
        assert len(POSITIONAL_GRADIENT_TARGETS) == 6

    def test_table_gradient_formula_says_accumulate(self):
        assert "Σ" in POSITIONAL_GRADIENT_FORMULAS[GRAD_TABLE]
        assert "逐位" in POSITIONAL_GRADIENT_FORMULAS[GRAD_INPUTS]

    def test_property_tables_align(self):
        assert set(POSITIONAL_PROPERTIES) == set(POSITIONAL_PROPERTY_DESCRIPTIONS)
        assert len(POSITIONAL_PROPERTIES) == 7

    def test_pairing_table_aligns(self):
        assert set(PAIRING_DESCRIPTIONS) == {PAIRING_ALIGNED, PAIRING_STAGGERED}
        assert "旋转" in PAIRING_DESCRIPTIONS[PAIRING_ALIGNED]
        assert "隔壁" in PAIRING_DESCRIPTIONS[PAIRING_STAGGERED]

    def test_base_matches_day073(self):
        """同一个公式在两处必须逐位相同（day073 的常量在这里再钉一次）."""
        assert POSITIONAL_BASE == DAY073_BASE
        assert POSITIONAL_BASE == POSITIONAL_BASE_VALUE
        assert POSITIONAL_BASE == 10000.0


class TestFamilies:
    """五族失败：分族的依据是"谁的错、该谁去修"，不是"哪一行抛的"."""

    def test_family_table_names_are_closed(self):
        assert set(FAMILY_OUTCOMES) == {
            "ShapeError",
            "ParameterError",
            "RangeError",
            "NumericError",
            "GradientError",
        }

    def test_range_error_is_a_parameter_error(self):
        """位置越界是**参数失败的特例**：`except ParameterError` 必须能兜住它."""
        assert issubclass(RangeError, ParameterError)
        assert issubclass(RangeError, PositionalError)

    def test_families_stay_inside_the_parent_families(self):
        """上游的 `except ShapeError`（day075 那一层）必须能兜住本层的形状错误."""
        assert issubclass(ShapeError, CoreShapeError)
        assert issubclass(ShapeError, PositionalError)
        assert issubclass(PositionalError, ValueError)

    def test_seven_properties_come_from_the_types_table(self):
        from smart_research_agent.positional_encoding.types import POSITIONAL_PROPERTIES as TYPES

        assert POSITIONAL_PROPERTIES == TYPES


class TestFrequencyAndRow:
    """造表的两个原语：频率 ``f_i = base^(2i/d)`` 与单行 ``sin/cos``."""

    def test_frequency_periods_are_hand_computed(self):
        """``d = 4``、``base = 10000``：两个周期恰好是 ``1`` 与 ``100``（``√10000 = 100``）."""
        assert frequency_of(0, DIMENSION) == FREQUENCY_PAIR_ZERO
        assert frequency_of(1, DIMENSION) == FREQUENCY_PAIR_ONE

    def test_frequency_at_the_smallest_dimension_is_one(self):
        """``d = 2`` 只有一对频率，且 ``2·0/d = 0`` → ``f = base^0 = 1``."""
        assert frequency_of(0, 2) == 1.0

    @pytest.mark.parametrize("value", [True, 2.0, "0"])
    def test_frequency_pair_must_be_an_integer(self, value):
        with pytest.raises(ParameterError, match="必须是整数"):
            frequency_of(value, DIMENSION)

    @pytest.mark.parametrize("pair", [-1, 2])
    def test_frequency_pair_range(self, pair):
        with pytest.raises(ParameterError, match="频率对下标"):
            frequency_of(pair, DIMENSION)

    def test_frequency_dimension_must_be_at_least_two(self):
        with pytest.raises(ParameterError, match="必须 >= 2"):
            frequency_of(0, 1)

    def test_frequency_base_must_exceed_one(self):
        with pytest.raises(ParameterError, match="频率基数必须 > 1"):
            frequency_of(0, DIMENSION, base=1.0)

    def test_sinusoidal_row_is_the_anchor(self):
        assert sinusoidal_row(0, DIMENSION) == PE_ROW_ZERO
        assert sinusoidal_row(1, DIMENSION) == PE_ROW_ONE

    def test_sinusoidal_row_staggered_differs_from_aligned(self):
        staggered_row = sinusoidal_row(1, DIMENSION, pairing=PAIRING_STAGGERED)
        assert staggered_row == STAGGERED_ROW_ONE
        assert staggered_row != PE_ROW_ONE

    @pytest.mark.parametrize("value", [True, 1.5, "1"])
    def test_row_position_must_be_an_integer(self, value):
        with pytest.raises(ParameterError, match="必须是整数"):
            sinusoidal_row(value, DIMENSION)

    def test_row_position_must_be_non_negative(self):
        with pytest.raises(ParameterError, match="必须 >= 0"):
            sinusoidal_row(-1, DIMENSION)

    def test_row_requires_an_even_dimension(self):
        """奇数维最后一维只有一个 sin——两条正弦表性质都以偶数维为前提."""
        with pytest.raises(ShapeError, match="偶数"):
            sinusoidal_row(0, 3)

    def test_row_rejects_unknown_pairing(self):
        with pytest.raises(ParameterError, match="未知的频率配对方式"):
            sinusoidal_row(0, DIMENSION, pairing="weird")


class TestSinusoidalTable:
    """正弦表：**没有任何参数**，表长之外仍然有定义."""

    def test_rows_are_hand_computed(self):
        table = sinusoidal(LENGTH, DIMENSION)
        assert table.table[0] == PE_ROW_ZERO
        assert table.table[1] == PE_ROW_ONE
        assert approx_matrix(table.table[:2], (PE_ROW_ZERO, PE_ROW_ONE))

    def test_row_norms_are_exactly_sqrt_d_over_two(self):
        table = sinusoidal(LENGTH, DIMENSION)
        for value in matrix_row_norms(table.table):
            assert approx(value, ROW_NORM_TWO, tolerance=ROW_NORM_TOLERANCE)
        assert table.aligned_within is True

    def test_parameter_count_is_zero_and_pairing_is_aligned(self):
        table = sinusoidal(LENGTH, DIMENSION)
        assert table.kind == ENCODING_SINUSOIDAL
        assert table.parameter_count == 0
        assert table.learnable is False
        assert table.pairing_is_aligned is True
        assert table.base == POSITIONAL_BASE

    def test_staggered_table_row_is_hand_computed(self):
        table = staggered(LENGTH, DIMENSION)
        assert table.table[1] == STAGGERED_ROW_ONE
        assert table.pairing == PAIRING_STAGGERED
        assert table.aligned_within is False

    def test_notes_record_the_periods(self):
        table = sinusoidal(LENGTH, DIMENSION)
        assert any("频率对的周期" in note for note in table.notes)
        assert any("表长之外" in note for note in table.notes)

    def test_long_table_truncates_the_period_list(self):
        """``d = 12`` 有 6 对频率：提示串只列前 4 个，其余用省略号."""
        table = sinusoidal_table(2, 12)
        assert "…" in table.notes[1]

    def test_table_length_outside_the_table_is_still_defined(self):
        """正弦表越界**算得出来**：第 999 位的行由公式直接给出（表长不是边界）."""
        table = sinusoidal(LENGTH, DIMENSION)
        assert table.extends_to(999) is True
        assert len(sinusoidal_row(999, DIMENSION)) == DIMENSION
        assert sinusoidal_row(999, DIMENSION) == sinusoidal_table(1000, DIMENSION).row(999)

    @pytest.mark.parametrize("positions", [0, -1])
    def test_non_positive_position_count_is_rejected(self, positions):
        with pytest.raises(ParameterError, match="position"):
            sinusoidal_table(positions, DIMENSION)

    def test_odd_dimension_is_rejected(self):
        with pytest.raises(ShapeError, match="偶数"):
            sinusoidal_table(LENGTH, 3)

    def test_dimension_below_two_is_rejected(self):
        with pytest.raises(ParameterError, match="必须 >= 2"):
            sinusoidal_table(LENGTH, 1)

    def test_unknown_pairing_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的频率配对方式"):
            sinusoidal_table(LENGTH, DIMENSION, pairing="nope")


class TestLearnableTable:
    """可学习表：每一行都是一个参数，表长是一条**硬边界**."""

    def test_from_sine_starts_at_the_sinusoidal_table(self):
        table = learnable(LENGTH, DIMENSION)
        assert table.kind == ENCODING_LEARNABLE
        assert table.table == sinusoidal(LENGTH, DIMENSION).table
        assert table.parameter_count == LENGTH * DIMENSION
        assert table.learnable is True

    def test_random_initialisation_is_bounded_and_deterministic(self):
        table = learnable_table(LENGTH, DIMENSION, initializer=INIT_RANDOM, scale=0.25, seed=7)
        again = learnable_table(LENGTH, DIMENSION, initializer=INIT_RANDOM, scale=0.25, seed=7)
        assert table.table == again.table
        assert all(abs(value) < 0.25 for row in table.table for value in row)

    def test_explicit_values_are_used_verbatim(self):
        values = ((1.0, 2.0), (3.0, 4.0))
        table = learnable_table(2, 2, values=values)
        assert table.table == values

    def test_explicit_values_wrong_shape_is_rejected(self):
        with pytest.raises(ShapeError, match="不一致"):
            learnable_table(2, 2, values=((1.0, 2.0, 3.0),))

    def test_unknown_initialiser_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的初始化方式"):
            learnable_table(LENGTH, DIMENSION, initializer="magic")

    def test_scale_must_be_positive(self):
        with pytest.raises(ParameterError, match="scale"):
            learnable_table(LENGTH, DIMENSION, initializer=INIT_RANDOM, scale=0.0)

    def test_seed_must_be_an_integer(self):
        with pytest.raises(ParameterError, match="seed"):
            learnable_table(LENGTH, DIMENSION, initializer=INIT_RANDOM, seed=1.5)

    def test_learnable_table_does_not_extend(self):
        table = learnable(LENGTH, DIMENSION)
        assert table.extends_to(LENGTH - 1) is True
        assert table.extends_to(LENGTH) is False
        with pytest.raises(RangeError, match="范围"):
            table.row(LENGTH)

    def test_zero_table_is_learnable_with_zero_parameters_norm(self):
        table = zeros(LENGTH, DIMENSION)
        assert table.learnable is True
        assert table.parameter_count == LENGTH * DIMENSION
        assert all(value == 0.0 for row in table.table for value in row)
        assert table.aligned_within is False


class TestEncodingTableRecord:
    """两种编码共用一条记录：注入那一步对两者一无所知."""

    def test_inner_product_uses_the_anchor(self):
        table = sinusoidal(LENGTH, DIMENSION)
        assert table.inner_product(0, 1) == OFFSET_INNER_ONE

    def test_flatten_is_row_major(self):
        table = sinusoidal(LENGTH, DIMENSION)
        flat, shapes = table.flatten()
        assert len(flat) == LENGTH * DIMENSION
        assert shapes == ((LENGTH, DIMENSION),)
        assert flat[:DIMENSION] == PE_ROW_ZERO

    def test_with_table_keeps_kind_and_base(self):
        table = sinusoidal(LENGTH, DIMENSION)
        replacement = zeros(LENGTH, DIMENSION).table
        swapped = table.with_table(replacement)
        assert swapped.kind == table.kind
        assert swapped.base == table.base
        assert swapped.pairing == table.pairing
        assert swapped.table == replacement

    def test_with_table_checks_the_shape(self):
        with pytest.raises(ShapeError, match="不能改变形状"):
            sinusoidal(LENGTH, DIMENSION).with_table(((1.0,),))

    def test_describe_mentions_the_origin(self):
        table = sinusoidal(LENGTH, DIMENSION)
        assert "公式生成" in table.describe()
        assert "行参数" in learnable(LENGTH, DIMENSION).describe()
        assert table.summary_line() == table.describe()

    def test_to_dict_is_json_ready(self):
        payload = sinusoidal(LENGTH, DIMENSION).to_dict()
        assert payload["kind"] == ENCODING_SINUSOIDAL
        assert payload["pairing"] == PAIRING_ALIGNED
        assert payload["aligned_within"] is True
        assert payload["parameter_count"] == 0

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的编码种类"):
            EncodingTable(kind="magic", table=((1.0, 2.0),))

    def test_empty_table_is_rejected(self):
        """空矩阵由 day073 的 ``validate_matrix`` 拦下（抛数学族的那一个）."""
        with pytest.raises(MathShapeError, match="不能为空"):
            EncodingTable(kind=ENCODING_LEARNABLE, table=())

    def test_sinusoidal_requires_a_base(self):
        with pytest.raises(ParameterError, match="频率基数 base"):
            EncodingTable(
                kind=ENCODING_SINUSOIDAL, table=((1.0, 2.0),), pairing=PAIRING_ALIGNED
            )

    def test_sinusoidal_base_must_exceed_one(self):
        with pytest.raises(ParameterError, match="频率基数必须 > 1"):
            EncodingTable(
                kind=ENCODING_SINUSOIDAL, table=((1.0, 2.0),), base=1.0, pairing=PAIRING_ALIGNED
            )

    def test_sinusoidal_requires_a_known_pairing(self):
        with pytest.raises(ParameterError, match="未知的频率配对方式"):
            EncodingTable(
                kind=ENCODING_SINUSOIDAL, table=((1.0, 2.0),), base=10000.0, pairing="x"
            )

    def test_learnable_rejects_base_and_pairing(self):
        with pytest.raises(ParameterError, match="没有频率基数与配对方式"):
            EncodingTable(kind=ENCODING_LEARNABLE, table=((1.0, 2.0),), base=1.0)
        with pytest.raises(ParameterError, match="没有频率基数与配对方式"):
            EncodingTable(kind=ENCODING_LEARNABLE, table=((1.0, 2.0),), pairing=PAIRING_ALIGNED)

    def test_row_index_must_be_an_integer(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            sinusoidal(LENGTH, DIMENSION).row(True)

    def test_row_index_out_of_range_never_extrapolates(self):
        with pytest.raises(RangeError, match="范围"):
            learnable(LENGTH, DIMENSION).row(LENGTH)
        with pytest.raises(RangeError, match="范围"):
            learnable(LENGTH, DIMENSION).row(-1)

    def test_extends_to_validates_the_position(self):
        with pytest.raises(ParameterError, match="必须 >= 0"):
            sinusoidal(LENGTH, DIMENSION).extends_to(-1)

    def test_inner_product_validates_the_indices(self):
        with pytest.raises(ParameterError, match="必须 >= 0"):
            sinusoidal(LENGTH, DIMENSION).inner_product(-1, 0)


class TestShapeAndPositions:
    """形状契约（四个维度、两个必须相等）与位置序列的三种校验."""

    def test_derived_quantities(self):
        shape = PositionalShape(inputs=4, positions=4, dimension=4, outputs=4)
        assert shape.table_shape == (4, 4)
        assert shape.pairs == 2
        assert shape.even_dimension is True
        assert approx(shape.row_norm, ROW_NORM_TWO)
        assert shape.to_dict()["pairs"] == 2
        assert "偶数维" in shape.summary_line()

    def test_odd_dimension_is_flagged(self):
        shape = PositionalShape(inputs=3, positions=2, dimension=3, outputs=3)
        assert shape.even_dimension is False
        assert "奇数维" in shape.summary_line()

    def test_width_must_equal_the_input_columns(self):
        with pytest.raises(ShapeError, match="不一致"):
            PositionalShape(inputs=4, positions=4, dimension=5, outputs=4)

    @pytest.mark.parametrize("name", ["inputs", "positions", "dimension", "outputs"])
    def test_every_dimension_must_be_positive(self, name):
        payload = {"inputs": 4, "positions": 4, "dimension": 4, "outputs": 4}
        payload[name] = 0
        with pytest.raises(ParameterError, match=name):
            PositionalShape(**payload)

    def test_non_integer_dimension_is_rejected(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            PositionalShape(inputs=4, positions=4, dimension=4.0, outputs=4)

    def test_default_positions_are_contiguous(self):
        assert default_positions(4) == (0, 1, 2, 3)

    def test_default_positions_rejects_zero(self):
        with pytest.raises(ParameterError, match="必须 >= 1"):
            default_positions(0)

    def test_validate_positions_accepts_a_valid_sequence(self):
        assert validate_positions((0, 3), expected=2, table_positions=4) == (0, 3)

    def test_validate_positions_rejects_non_sequences(self):
        with pytest.raises(ShapeError, match="必须是序列"):
            validate_positions("01", expected=2, table_positions=4)

    def test_validate_positions_rejects_boolean_entries(self):
        with pytest.raises(ShapeError, match="必须是整数"):
            validate_positions((True,), expected=1, table_positions=4)

    def test_validate_positions_rejects_negative_positions(self):
        """负下标在 Python 里会**从尾部取**——那是一个静默的错误取值."""
        with pytest.raises(RangeError, match="负位置"):
            validate_positions((-1,), expected=1, table_positions=4)

    def test_validate_positions_rejects_out_of_range(self):
        with pytest.raises(RangeError, match="超出"):
            validate_positions((4,), expected=1, table_positions=4)

    def test_validate_positions_checks_the_count(self):
        with pytest.raises(ShapeError, match="个数"):
            validate_positions((0, 1), expected=3, table_positions=4)

    def test_gather_rows_takes_rows_verbatim(self):
        table = learnable_table(2, 2, values=((1.0, 2.0), (3.0, 4.0)))
        assert gather_rows(table.table, (1, 0)) == ((3.0, 4.0), (1.0, 2.0))

    def test_gather_rows_rejects_out_of_range(self):
        with pytest.raises(RangeError, match="越界"):
            gather_rows(((1.0,), (2.0,)), (2,))

    def test_gather_rows_rejects_an_empty_sequence(self):
        with pytest.raises(ShapeError, match="为空"):
            gather_rows(((1.0,), (2.0,)), ())

    def test_scatter_add_accumulates_repeated_positions(self):
        """**按位置累加**：位置重复时两份梯度落在同一行，必须是 6 而不是 5."""
        row_gradients = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        assert scatter_add_rows(row_gradients, (0, 1, 0), table_positions=2) == (
            (6.0, 8.0),
            (3.0, 4.0),
        )

    def test_scatter_add_without_duplicates_is_a_plain_copy(self):
        row_gradients = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        assert scatter_add_rows(row_gradients, (0, 1, 2), table_positions=3) == row_gradients

    @pytest.mark.parametrize("position", [-1, 9])
    def test_scatter_add_rejects_out_of_range(self, position):
        with pytest.raises(RangeError, match="越界"):
            scatter_add_rows(((1.0, 2.0),), (position,), table_positions=3)

    def test_scatter_add_requires_a_positive_table_size(self):
        with pytest.raises(ParameterError, match="必须 >= 1"):
            scatter_add_rows(((1.0, 2.0),), (0,), table_positions=0)

    def test_add_matrices_is_elementwise(self):
        assert add_matrices(((1.0, 2.0),), ((3.0, 4.0),)) == ((4.0, 6.0),)

    def test_add_matrices_checks_the_shape(self):
        with pytest.raises(ShapeError, match="形状不同"):
            add_matrices(((1.0, 2.0),), ((1.0, 2.0, 3.0),))

    def test_matrix_row_norms_is_the_l2_norm(self):
        assert matrix_row_norms(((3.0, 4.0),)) == (5.0,)


class TestInject:
    """注入：返回 ``(取出的表行, inputs + 取出的表行)``——两个返回值都必须可见."""

    def test_hand_computed_injection(self):
        table = sinusoidal(LENGTH, DIMENSION)
        gathered, injected = inject(ONES_INPUTS, table)
        assert gathered == table.table
        assert injected[0] == ONES_INJECTED_ROW_ZERO
        assert injected == add_matrices(ONES_INPUTS, table.table)

    def test_inject_can_use_an_explicit_position_sequence(self):
        table = sinusoidal(LENGTH, DIMENSION)
        gathered, injected = inject(ONES_INPUTS, table, positions=(3, 2, 1, 0))
        assert gathered == tuple(reversed(table.table))
        assert injected[0] == add_matrices(ONES_INPUTS, gathered)[0]

    def test_inject_checks_the_position_count(self):
        with pytest.raises(ShapeError, match="个数"):
            inject(ONES_INPUTS, sinusoidal(LENGTH, DIMENSION), positions=(0, 1))

    def test_inject_checks_the_row_width(self):
        with pytest.raises(ShapeError, match="宽度"):
            inject(ONES_INPUTS, zeros(LENGTH, 5))


class TestForward:
    """前向六阶段：``injected`` 进 day075 那层，本课**一行未改**那一层."""

    def _forward(self, **kwargs):
        return positional_forward(
            position_parameters(DIMENSION),
            sinusoidal(LENGTH, DIMENSION),
            ONES_INPUTS,
            **kwargs,
        )

    def test_shape_positions_and_gathered(self):
        forward = self._forward()
        assert forward.shape == PositionalShape(
            inputs=DIMENSION, positions=LENGTH, dimension=DIMENSION, outputs=DIMENSION
        )
        assert forward.positions == (0, 1, 2, 3)
        assert forward.tokens == LENGTH
        assert forward.learnable is False

    def test_injected_is_the_sum_and_the_ratio_is_hand_computed(self):
        forward = self._forward()
        assert forward.inputs == ONES_INPUTS
        assert forward.gathered == sinusoidal(LENGTH, DIMENSION).table
        assert forward.injected[0] == ONES_INJECTED_ROW_ZERO
        assert approx(forward.mean_injection_ratio, ONES_MEAN_INJECTION_RATIO)
        assert all(approx(value, ONES_MEAN_INJECTION_RATIO) for value in forward.injection_ratios)

    def test_attention_is_day075_on_the_injected_input(self):
        forward = self._forward()
        reference = self_attention(
            position_parameters(DIMENSION), forward.injected, causal=False
        )
        assert forward.attention.output == reference.output
        assert forward.attention.weights == reference.weights

    def test_no_target_means_no_loss(self):
        forward = self._forward()
        assert forward.loss is None
        assert forward.supervised_rows == ()
        assert "损失 —" in forward.summary_line()

    def test_full_supervision_uses_the_plain_mse(self):
        target = ((1.0, 0.0, 0.0, 0.0),) * LENGTH
        forward = self._forward(target=target)
        assert forward.loss == mean_squared_error(forward.attention.output, target)
        assert forward.supervised_rows == (0, 1, 2, 3)

    def test_partial_supervision_uses_the_masked_mse(self):
        target = ((1.0, 0.0, 0.0, 0.0),) * LENGTH
        forward = self._forward(target=target, supervised=(0, 1))
        assert forward.loss == masked_mean_squared_error(
            forward.attention.output, target, (0, 1)
        )

    def test_causal_flag_is_recorded_in_the_notes(self):
        forward = self._forward(causal=True)
        assert any("跳过" in note for note in forward.notes)

    def test_output_row_reads_one_row(self):
        forward = self._forward()
        assert forward.output_row(0) == forward.attention.output[0]
        with pytest.raises(RangeError, match="越界"):
            forward.output_row(9)

    def test_long_sequences_truncate_the_notes(self):
        """8 行时位置列表只列前 6 个——一行说明不能无限长."""
        inputs = tuple(
            tuple(1.0 if index == column else 0.0 for column in range(8)) for index in range(8)
        )
        forward = positional_forward(
            position_parameters(8), sinusoidal_table(8, 8), inputs
        )
        assert "…" in forward.notes[0]
        assert "…" in forward.summary_line()

    def test_params_must_be_attention_params(self):
        with pytest.raises(ParameterError, match="AttentionParams"):
            positional_forward("params", sinusoidal(LENGTH, DIMENSION), ONES_INPUTS)

    def test_table_must_be_an_encoding_table(self):
        with pytest.raises(ParameterError, match="EncodingTable"):
            positional_forward(position_parameters(DIMENSION), "table", ONES_INPUTS)

    def test_width_must_match_the_attention_layer(self):
        with pytest.raises(ShapeError, match="不一致"):
            positional_forward(position_parameters(VOCABULARY), zeros(LENGTH, 4), ONES_INPUTS)

    def test_positions_must_match_the_row_count(self):
        with pytest.raises(ShapeError, match="个数"):
            positional_forward(
                position_parameters(DIMENSION),
                sinusoidal(LENGTH, DIMENSION),
                ONES_INPUTS,
                positions=(0, 1),
            )

    def test_target_shape_is_checked(self):
        with pytest.raises(ShapeError, match="目标形状"):
            self._forward(target=((1.0,),))

    def test_supervision_errors(self):
        target = ((1.0, 0.0, 0.0, 0.0),) * LENGTH
        with pytest.raises(ShapeError, match="重复"):
            self._forward(target=target, supervised=(0, 0))
        with pytest.raises(ShapeError, match="超出"):
            self._forward(target=target, supervised=(9,))
        with pytest.raises(ShapeError, match="整数"):
            self._forward(target=target, supervised=(True,))
        with pytest.raises(ParameterError, match="不能为空"):
            self._forward(target=target, supervised=[])


class TestForwardRecord:
    """前向记录的构造校验：少一项、形状不符都要当场拒绝."""

    def _kwargs(self):
        target = ((1.0, 0.0, 0.0, 0.0),) * LENGTH
        forward = positional_forward(
            position_parameters(DIMENSION),
            sinusoidal(LENGTH, DIMENSION),
            ONES_INPUTS,
            target=target,
            supervised=(0, 1),
        )
        return dict(
            shape=forward.shape,
            table=forward.table,
            positions=forward.positions,
            inputs=forward.inputs,
            gathered=forward.gathered,
            injected=forward.injected,
            attention=forward.attention,
            target=forward.target,
            supervised=forward.supervised,
            loss=forward.loss,
        )

    def test_gathered_shape_must_match(self):
        with pytest.raises(ShapeError, match="逐行对齐"):
            PositionalForward(**{**self._kwargs(), "gathered": ((1.0,),) * LENGTH})

    def test_injected_shape_must_match(self):
        with pytest.raises(ShapeError, match="注入前后"):
            PositionalForward(**{**self._kwargs(), "injected": ((1.0,),) * LENGTH})

    def test_position_count_must_match(self):
        with pytest.raises(ShapeError, match="位置个数"):
            PositionalForward(**{**self._kwargs(), "positions": (0, 1)})

    def test_loss_without_target_is_rejected(self):
        with pytest.raises(NumericError, match="没有给 target"):
            PositionalForward(**{**self._kwargs(), "target": None, "supervised": ()})

    def test_supervision_without_target_is_rejected(self):
        with pytest.raises(NumericError, match="却有监督行"):
            PositionalForward(**{**self._kwargs(), "target": None, "loss": None})

    def test_target_without_loss_is_rejected(self):
        with pytest.raises(NumericError, match="没有 loss"):
            PositionalForward(**{**self._kwargs(), "loss": None})

    def test_notes_are_stringified(self):
        assert PositionalForward(**{**self._kwargs(), "notes": (1,)}).notes == ("1",)


class TestBackward:
    """反向：``dx`` 逐位传回、``dTable`` 按位置累加，前四块与 day075 逐字相同."""

    def _forward(self, *, positions=None):
        target = ((1.0, 0.0, 0.0, 0.0),) * LENGTH
        return positional_forward(
            position_parameters(DIMENSION),
            sinusoidal(LENGTH, DIMENSION),
            ONES_INPUTS,
            positions=positions,
            target=target,
            supervised=tuple(range(LENGTH)),
        )

    def test_gradient_inputs_equal_dinjected_bit_for_bit(self):
        """加法注入的偏导数恰好是 ``1``：``dx == dInjected``（不是"接近"）."""
        forward = self._forward()
        grad = loss_gradient(forward)
        gradients = positional_backward(forward, grad)
        inner = attention_backward(forward.attention, grad)
        assert gradients.grad_inputs == inner.grad_inputs
        assert gradients.grad_w_output == inner.grad_w_output
        assert gradients.grad_w_value == inner.grad_w_value

    def test_table_gradient_is_scatter_add_of_the_input_gradient(self):
        forward = self._forward(positions=(0, 1, 0, 1))
        grad = loss_gradient(forward)
        gradients = positional_backward(forward, grad)
        assert gradients.grad_table == scatter_add_rows(
            gradients.grad_inputs, forward.positions, table_positions=forward.table.positions
        )

    def test_loss_gradient_needs_a_target(self):
        forward = positional_forward(
            position_parameters(DIMENSION), sinusoidal(LENGTH, DIMENSION), ONES_INPUTS
        )
        with pytest.raises(ParameterError, match="没有 target"):
            loss_gradient(forward)

    def test_grad_output_shape_is_checked(self):
        with pytest.raises(ShapeError, match="grad_output"):
            positional_backward(self._forward(), ((1.0,),))


class TestGradientsRecord:
    """六块梯度的账：压平、量级、按名字取用."""

    def _gradients(self):
        target = ((1.0, 0.0, 0.0, 0.0),) * LENGTH
        forward = positional_forward(
            position_parameters(DIMENSION),
            sinusoidal(LENGTH, DIMENSION),
            ONES_INPUTS,
            target=target,
            supervised=tuple(range(LENGTH)),
        )
        return positional_backward(forward, loss_gradient(forward))

    def test_five_matrices_flatten_to_the_parameter_blocks(self):
        gradients = self._gradients()
        assert len(gradients.matrices()) == 5
        assert len(gradients.flatten()) == 4 * 16 + 16
        assert gradients.max_absolute() > 0.0

    def test_as_dict_is_keyed_by_the_gradient_names(self):
        gradients = self._gradients()
        assert set(gradients.as_dict()) == {
            GRAD_W_QUERY,
            GRAD_W_KEY,
            GRAD_W_VALUE,
            GRAD_W_OUTPUT,
            GRAD_TABLE,
            GRAD_INPUTS,
        }
        assert gradients.as_dict()[GRAD_INPUTS] == gradients.grad_inputs

    def test_summary_line_names_all_six(self):
        line = self._gradients().summary_line()
        for label in ("dW_q", "dW_k", "dW_v", "dW_o", "dTable", "dx"):
            assert label in line

    def test_record_validates_its_matrices(self):
        """一张"每行不等长"的梯度表由 day073 的 ``validate_matrix`` 拦下."""
        with pytest.raises(MathShapeError, match="等长"):
            PositionalGradients(
                grad_table=((1.0,),),
                grad_inputs=((1.0,),),
                grad_w_query=((1.0, 2.0), (3.0,)),
                grad_w_key=((1.0,),),
                grad_w_value=((1.0,),),
                grad_w_output=((1.0,),),
            )


class TestBatchReadouts:
    """批量读数：一批序列（位置会重复）上的平均损失 / 命中率 / 位置占比."""

    def _sequences(self):
        return tuple((item.inputs, item.target) for item in readout_orbit())

    def test_batch_loss_is_the_mean_of_the_per_sequence_loss(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        sequences = self._sequences()
        expected = 0.0
        for inputs, target in sequences:
            expected += positional_forward(params, table, inputs, target=target).loss
        expected /= len(sequences)
        assert batch_loss(params, table, sequences) == expected

    def test_batch_accuracy_is_a_fraction(self):
        value = batch_accuracy(
            position_parameters(VOCABULARY), sinusoidal(LENGTH, VOCABULARY), self._sequences()
        )
        assert 0.0 <= value <= 1.0

    def test_mean_injection_ratio_is_sqrt_three(self):
        """正弦表行范数 ``√3``、one-hot 输入行范数 ``1`` → 比值恰好 ``√3``."""
        value = batch_mean_injection_ratio(
            position_parameters(VOCABULARY), sinusoidal(LENGTH, VOCABULARY), self._sequences()
        )
        assert approx(value, math.sqrt(3.0))

    def test_empty_batches_are_rejected(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        with pytest.raises(ParameterError, match="至少要给一条序列"):
            batch_loss(params, table, [])
        with pytest.raises(ParameterError, match="至少要给一条序列"):
            batch_accuracy(params, table, [])
        with pytest.raises(ParameterError, match="至少要给一条序列"):
            batch_mean_injection_ratio(params, table, [])


class TestHelpers:
    """报告用的小工具：逐位比较、相对误差、容差校验、压平/还原."""

    def test_rows_are_aligned(self):
        assert check_rows_are_aligned(((1.0, 2.0),), ((1.0, 2.0),)) is True
        assert check_rows_are_aligned(((1.0,),), ((1.0, 2.0),)) is False
        assert check_rows_are_aligned(((1.0,),), ((2.0,),)) is False

    def test_relative_matrix_error_is_forwarded(self):
        assert relative_matrix_error(((1.0,),), ((1.0,),)) == 0.0
        assert approx(relative_matrix_error(((1.0,),), ((2.0,),)), 0.5)

    def test_validate_settings(self):
        assert validate_settings(OFFSET_TOLERANCE) == OFFSET_TOLERANCE
        with pytest.raises(ParameterError, match="正的有限数"):
            validate_settings(-1.0)
        with pytest.raises(ParameterError, match="正的有限数"):
            validate_settings(float("nan"))

    def test_flatten_parameters_appends_the_table_last(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        flat, shapes = flatten_parameters(params, table)
        assert shapes == ((VOCABULARY, VOCABULARY),) * 4 + ((LENGTH, VOCABULARY),)
        assert len(flat) == params.parameter_count() + LENGTH * VOCABULARY
        restored_params, restored_table = unflatten_parameters(flat, shapes, template=table)
        assert restored_params == params
        assert restored_table == table

    def test_unflatten_checks_the_block_count(self):
        with pytest.raises(ShapeError, match="5 个"):
            unflatten_parameters(
                (1.0,), ((1, 1),), template=sinusoidal(LENGTH, DIMENSION)
            )

    def test_rotation_factor_at_zero_is_the_identity(self):
        identity = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(DIMENSION))
            for row in range(DIMENSION)
        )
        assert rotation_factor(DIMENSION, 0) == identity

    def test_rotate_by_zero_is_the_table_itself(self):
        table = sinusoidal(LENGTH, DIMENSION)
        assert rotate(table, 0) == table.table

    def test_rotate_requires_an_aligned_table(self):
        with pytest.raises(ParameterError, match="只有对齐频率"):
            rotate(learnable(LENGTH, DIMENSION), 1)
        with pytest.raises(ParameterError, match="只有对齐频率"):
            rotate(staggered(LENGTH, DIMENSION), 1)

    def test_rotate_requires_a_non_negative_offset(self):
        with pytest.raises(ParameterError, match="必须 >= 0"):
            rotate(sinusoidal(LENGTH, DIMENSION), -1)

    def test_rotation_tolerance_is_pinned(self):
        from smart_research_agent.positional_encoding.verify import ROTATION_TOLERANCE

        assert ROTATION_TOLERANCE == 1e-12
        assert GRADIENT_TOLERANCE > ROTATION_TOLERANCE

    def test_closed_form_requires_an_even_dimension(self):
        with pytest.raises(ShapeError, match="偶数"):
            closed_form_offset_inner(3, 1)

    def test_closed_form_requires_a_non_negative_offset(self):
        with pytest.raises(ParameterError, match="必须 >= 0"):
            closed_form_offset_inner(DIMENSION, -1)

    def test_rotation_factor_requires_an_even_dimension(self):
        with pytest.raises(ShapeError, match="偶数"):
            rotation_factor(3, 1)

    def test_gradient_error_is_a_positional_error(self):
        assert issubclass(GradientError, PositionalError)
