"""``sequence_models``：RNN / LSTM 的前向、BPTT 与训练的测试（day094 / M8-D5）.

本文件覆盖新包的九个功能模块与包入口：口径表、六个失败族、算子层
（向量 / 矩阵 / 门切块 / 参数量）、层（可复现初始化 / 一步 / 整段）、
反向（BPTT 的累加与两条 carry）、序列分类器的前向反向与参数压平、
合成数据集上的训练与梯度爆炸护栏、七条性质与七张表。

样本全部来自本包自己的确定性构造；**跨天对账真的调用既有包**：
数值差分用 ``math_foundations.calculus``（day074），激活导数用 ``backprop.gradients``（day090），
交叉熵用 ``neural_basics.losses``（day089），优化器用 ``optimizers``（day092）。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.sequence_models import (
    errors,
    gradients,
    layers,
    network,
    ops,
    study,
    train,
    types,
    verify,
)


@pytest.fixture(scope="module")
def reports() -> dict[str, train.TrainReport]:
    """模块级缓存：``rnn`` 与 ``lstm`` 各训一次（供训练相关与表格相关用例共用）."""
    return study.train_reports()


# --------------------------------------------------------------------------- 口径表


def test_cell_tables_are_closed() -> None:
    """两种循环单元的三张表逐键对齐。"""
    assert types.CELL_TYPES == ("rnn", "lstm")
    assert set(types.CELL_TYPES) == set(types.CELL_DESCRIPTIONS) == set(types.CELL_STATE_SHAPES)
    assert all(types.CELL_DESCRIPTIONS.values())
    assert all(types.CELL_STATE_SHAPES.values())


def test_gate_tables_are_closed() -> None:
    """四个门的三张表逐键对齐，且门激活分别是 sigmoid / tanh。"""
    assert types.GATES == ("input", "forget", "output", "candidate")
    assert set(types.GATES) == set(types.GATE_DESCRIPTIONS)
    assert set(types.GATES) == set(types.GATE_ACTIVATIONS)
    assert set(types.GATES) == set(types.GATE_ROW_BLOCKS)
    assert [types.GATE_ACTIVATIONS[g] for g in types.GATES] == [
        "sigmoid",
        "sigmoid",
        "sigmoid",
        "tanh",
    ]


def test_rnn_activations_are_a_subset_of_day089() -> None:
    """循环激活是 day089 ACTIVATIONS 的子集，且不含 softmax。"""
    from smart_research_agent.neural_basics.types import ACTIVATIONS

    assert set(types.RNN_ACTIVATIONS) <= set(ACTIVATIONS)
    assert "softmax" not in types.RNN_ACTIVATIONS
    assert types.RNN_ACTIVATIONS == ("tanh", "relu")


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单 / 说明 / "失败意味着什么"三张表逐键对齐。"""
    assert len(types.RECURRENT_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.RECURRENT_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.RECURRENT_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())
    assert types.PROPERTIES == types.RECURRENT_PROPERTIES


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.RECURRENT_NOTES) == 10
    assert types.NOTES_ORDER == tuple(types.RECURRENT_NOTES)
    assert all(types.RECURRENT_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）。"""
    assert len(types.RECURRENT_BOUNDARIES) == 5
    assert all(types.RECURRENT_BOUNDARIES)


def test_torch_counterparts_cover_core_ops_and_no_version() -> None:
    """PyTorch 对照表覆盖单元 / 整段 / BPTT / 裁剪，且**不出现**版本号。"""
    required = {"rnn_cell", "lstm_cell", "rnn_sequence", "lstm_sequence", "bptt", "grad_clipping"}
    assert required <= set(types.TORCH_COUNTERPARTS)
    assert all(types.TORCH_COUNTERPARTS.values())
    assert "torch==" not in " ".join(types.TORCH_COUNTERPARTS.values())


def test_formulas_are_non_empty() -> None:
    """五张公式都写下了一行可读的式子。"""
    assert "h_t" in types.RNN_RECURRENCE_FORMULA
    assert "c_t" in types.LSTM_GATE_FORMULA
    assert "H·D" in types.RNN_PARAM_FORMULA
    assert "4H" in types.LSTM_PARAM_FORMULA
    assert "Σ" in types.BPTT_FORMULA
    assert types.DEFAULT_FORGET_BIAS == 1.0


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_aligned() -> None:
    """六个族的"该怎么办"表与类表逐键对齐。"""
    assert len(errors.FAMILY_OUTCOMES) == 6
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "TimeStepError",
        "BackwardError",
        "GradientError",
    }


def test_gradient_error_returned_and_none_absent() -> None:
    """``GradientError`` 今天**回来了**，且缺席名单为空（可断言的事实）。"""
    assert errors.RETURNED_FAMILY == "GradientError"
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY is None
    assert errors.ABSENT_FAMILY_REASON


def test_errors_inherit_both_families() -> None:
    """本族错误既是 ``RecurrentError``、也是 day075 的对应族。"""
    from smart_research_agent.transformer_core import errors as core_errors

    assert issubclass(errors.ShapeError, errors.RecurrentError)
    assert issubclass(errors.ShapeError, core_errors.ShapeError)
    assert issubclass(errors.ParameterError, core_errors.ParameterError)
    assert issubclass(errors.NumericError, core_errors.NumericError)
    assert issubclass(errors.GradientError, core_errors.GradientError)
    assert issubclass(errors.TimeStepError, errors.RecurrentError)
    assert issubclass(errors.BackwardError, errors.RecurrentError)


# --------------------------------------------------------------------------- 算子


def test_checked_positive_int() -> None:
    """整数校验：非整数 / 过小都拒绝。"""
    assert ops.checked_positive_int(3, name="k") == 3
    with pytest.raises(errors.ParameterError):
        ops.checked_positive_int(3.0, name="k")  # type: ignore[arg-type]
    with pytest.raises(errors.ParameterError):
        ops.checked_positive_int(0, name="k")
    with pytest.raises(errors.ParameterError):
        ops.checked_positive_int(True, name="k")


def test_as_vector_rejects_bad_inputs() -> None:
    """空 / 非实数 / 非有限都拒绝。"""
    assert ops.as_vector((1, 2.5), name="x") == (1.0, 2.5)
    with pytest.raises(errors.ShapeError):
        ops.as_vector((), name="x")
    with pytest.raises(errors.ShapeError):
        ops.as_vector(("a",), name="x")  # type: ignore[arg-type]
    with pytest.raises(errors.NumericError):
        ops.as_vector((float("inf"),), name="x")


def test_as_matrix_rejects_bad_inputs() -> None:
    """非二维 / 空 / 行宽不齐 / 非有限数都拒绝。"""
    with pytest.raises(errors.ShapeError):
        ops.as_matrix((), name="x")
    with pytest.raises(errors.ShapeError):
        ops.as_matrix(((1.0, 2.0), (3.0,)), name="x")
    with pytest.raises(errors.NumericError):
        ops.as_matrix(((1.0, float("nan")),), name="x")
    assert ops.as_matrix([[1, 2], [3, 4]], name="x") == ((1.0, 2.0), (3.0, 4.0))


def test_as_sequence_rejects_empty_and_ragged() -> None:
    """空序列抛 ``TimeStepError``；每步宽度不齐抛 ``ShapeError``。"""
    assert ops.as_sequence(((1.0,), (2.0,)), name="s") == ((1.0,), (2.0,))
    with pytest.raises(errors.TimeStepError):
        ops.as_sequence((), name="s")
    with pytest.raises(errors.ShapeError):
        ops.as_sequence(((1.0,), (2.0, 3.0)), name="s")


def test_as_finite_vector_allows_empty() -> None:
    """``as_finite_vector`` 允许为空（给"整体范数"一类的读数用）。"""
    assert ops.as_finite_vector((), name="g") == ()
    assert ops.as_finite_vector((1, 2), name="g") == (1.0, 2.0)
    with pytest.raises(errors.NumericError):
        ops.as_finite_vector((float("inf"),), name="g")
    with pytest.raises(errors.ShapeError):
        ops.as_finite_vector(("a",), name="g")  # type: ignore[arg-type]


def test_vector_helpers() -> None:
    """逐分量三件套 + 三向量相加。"""
    assert ops.zeros(3) == (0.0, 0.0, 0.0)
    assert ops.vec_add((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)
    assert ops.vec_scale((1.0, -2.0), 2.0) == (2.0, -4.0)
    assert ops.vec_hadamard((1.0, 2.0), (3.0, 4.0)) == (3.0, 8.0)
    assert ops.add_triple((1.0,), (2.0,), (3.0,)) == (6.0,)
    with pytest.raises(errors.ShapeError):
        ops.vec_add((1.0,), (1.0, 2.0))
    with pytest.raises(errors.ShapeError):
        ops.vec_hadamard((1.0,), (1.0, 2.0))
    with pytest.raises(errors.ShapeError):
        ops.add_triple((1.0,), (1.0, 2.0), (1.0,))
    with pytest.raises(errors.NumericError):
        ops.vec_scale((1.0,), float("inf"))


def test_matvec_and_transpose() -> None:
    """``W·x`` 与 ``Wᵀ·g`` 的内维检查。"""
    weight = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert ops.matvec(weight, (1.0, 0.0, -1.0)) == (-2.0, -2.0)
    assert ops.matvec_transpose(weight, (1.0, 1.0)) == (5.0, 7.0, 9.0)
    with pytest.raises(errors.ShapeError):
        ops.matvec(weight, (1.0, 2.0))
    with pytest.raises(errors.ShapeError):
        ops.matvec_transpose(weight, (1.0, 2.0, 3.0))


def test_outer_and_affine() -> None:
    """外积的形状与取值；仿射的偏置长度检查。"""
    assert ops.outer((1.0, 2.0), (3.0, 4.0, 5.0)) == ((3.0, 4.0, 5.0), (6.0, 8.0, 10.0))
    assert ops.affine((1.0, 1.0), ((1.0, 0.0), (0.0, 1.0)), (0.5, -0.5)) == (1.5, 0.5)
    with pytest.raises(errors.ShapeError):
        ops.affine((1.0, 1.0), ((1.0, 0.0),), (0.0, 0.0))


def test_gate_blocks_split_and_validate() -> None:
    """``4H`` 的打分按 i / f / o / g 切成四块；长度不符当场拒绝。"""
    assert ops.gate_blocks((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), 2) == (
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
    )
    assert ops.gate_blocks((1.0, 2.0, 3.0, 4.0), 1) == ((1.0,), (2.0,), (3.0,), (4.0,))
    with pytest.raises(errors.ShapeError):
        ops.gate_blocks((1.0, 2.0, 3.0), 2)


def test_parameter_count_formulas() -> None:
    """参数量公式：rnn = H(D+H+1)，lstm = 4×；与 T 无关。"""
    assert ops.parameter_count(types.CELL_RNN, 2, 3) == 3 * (2 + 3 + 1)
    assert ops.parameter_count(types.CELL_LSTM, 2, 3) == 4 * 3 * (2 + 3 + 1)
    with pytest.raises(errors.ParameterError):
        ops.parameter_count("gru", 2, 3)
    with pytest.raises(errors.ParameterError):
        ops.parameter_count(types.CELL_RNN, 0, 3)


def test_sequence_norms() -> None:
    """逐步范数。"""
    assert ops.sequence_norms(((3.0, 4.0), (0.0, 0.0))) == (5.0, 0.0)


# --------------------------------------------------------------------------- 层


def test_rnn_spec_validation_and_line() -> None:
    """RNNSpec 的每个字段都有护栏；一行说明含维度与参数量。"""
    spec = layers.RNNSpec(2, 3, seed=1)
    assert spec.cell == types.CELL_RNN
    assert spec.fan_in == 5
    assert spec.parameter_count == 18
    assert "rnn(D=2→H=3" in spec.line()
    assert "与 T 无关" in spec.line()
    with pytest.raises(errors.ParameterError):
        layers.RNNSpec(0, 3)
    with pytest.raises(errors.ParameterError):
        layers.RNNSpec(2, 0)
    with pytest.raises(errors.ParameterError):
        layers.RNNSpec(2, 3, activation="softmax")
    with pytest.raises(errors.ParameterError):
        layers.RNNSpec(2, 3, seed=-1)


def test_rnn_cell_validation() -> None:
    """RNNCell 拒绝 wh 行数 / 列数 / bias 长度不符。"""
    with pytest.raises(errors.ShapeError):
        layers.RNNCell(wx=((1.0, 0.0),), wh=((1.0,), (1.0,)), bias=(0.0,))
    with pytest.raises(errors.ShapeError):
        layers.RNNCell(wx=((1.0, 0.0),), wh=((1.0, 1.0, 1.0),), bias=(0.0,))
    with pytest.raises(errors.ShapeError):
        layers.RNNCell(wx=((1.0, 0.0),), wh=((1.0,),), bias=(0.0, 0.0))


def test_manual_rnn_step_covers_relu() -> None:
    """手写递推也覆盖 relu 分支，且与实现逐位一致。"""
    cell = layers.RNNCell(wx=((1.0,),), wh=((0.0,),), bias=(0.0,))
    assert verify._manual_rnn_step(cell, (2.0,), (0.0,), "relu") == (2.0,)
    assert verify._manual_rnn_step(cell, (-2.0,), (0.0,), "relu") == (0.0,)
    assert layers.rnn_cell_forward(cell, (2.0,), (0.0,), activation="relu") == (2.0,)
    assert layers.rnn_cell_forward(cell, (-2.0,), (0.0,), activation="relu") == (0.0,)


def test_initialize_rnn_is_reproducible() -> None:
    """同一份 spec 两次初始化逐位相同；不同种子不同；非 spec 拒绝。"""
    spec = layers.RNNSpec(3, 2, seed=7)
    first = layers.initialize_rnn(spec)
    second = layers.initialize_rnn(spec)
    assert first.wx == second.wx and first.wh == second.wh and first.bias == second.bias
    other = layers.initialize_rnn(layers.RNNSpec(3, 2, seed=8))
    assert other.wx != first.wx
    assert first.input_size == 3 and first.hidden_size == 2
    assert first.parameter_count == 2 * (3 + 2 + 1)
    assert first.initial_state() == (0.0, 0.0)
    assert "参数" in first.line()
    with pytest.raises(errors.ParameterError):
        layers.initialize_rnn("spec")  # type: ignore[arg-type]


def test_rnn_cell_forward_and_errors() -> None:
    """一步前向的取值 + 三种护栏（输入宽 / 状态宽 / 未知激活）。"""
    cell = layers.RNNCell(wx=((1.0,), (0.0,)), wh=((0.0, 0.0), (0.0, 0.0)), bias=(0.0, 0.0))
    assert layers.rnn_cell_forward(cell, (2.0,), (0.0, 0.0)) == (math.tanh(2.0), 0.0)
    assert layers.rnn_cell_forward(cell, (2.0,), (0.0, 0.0), activation="relu") == (2.0, 0.0)
    with pytest.raises(errors.ShapeError):
        layers.rnn_cell_forward(cell, (1.0, 2.0), (0.0, 0.0))
    with pytest.raises(errors.ShapeError):
        layers.rnn_cell_forward(cell, (1.0,), (0.0, 0.0, 0.0))
    with pytest.raises(errors.ParameterError):
        layers.rnn_cell_forward(cell, (1.0,), (0.0, 0.0), activation="gelu")


def test_rnn_pre_activation_and_forward() -> None:
    """激活之前的值；整段前向的状态数与初值护栏。"""
    cell = layers.initialize_rnn(layers.RNNSpec(2, 2, seed=4))
    pre = layers.rnn_pre_activation(cell, (1.0, -1.0), (0.0, 0.0))
    assert len(pre) == 2
    states = layers.rnn_forward(cell, study.SAMPLE_SEQUENCE)
    assert len(states) == len(study.SAMPLE_SEQUENCE)
    zero_start = layers.rnn_forward(cell, study.SAMPLE_SEQUENCE, state0=(0.0, 0.0))
    assert zero_start == states
    other_start = layers.rnn_forward(cell, study.SAMPLE_SEQUENCE, state0=(1.0, 1.0))
    assert other_start != states
    with pytest.raises(errors.ShapeError):
        layers.rnn_forward(cell, study.SAMPLE_SEQUENCE, state0=(0.0,))
    with pytest.raises(errors.TimeStepError):
        layers.rnn_forward(cell, ())


def test_rnn_pre_activation_guard() -> None:
    """``rnn_pre_activation`` 的两条形状护栏。"""
    cell = layers.initialize_rnn(layers.RNNSpec(2, 2, seed=4))
    with pytest.raises(errors.ShapeError):
        layers.rnn_pre_activation(cell, (1.0,), (0.0, 0.0))
    with pytest.raises(errors.ShapeError):
        layers.rnn_pre_activation(cell, (1.0, 1.0), (0.0,))


def test_lstm_spec_and_cell_validation() -> None:
    """LSTMSpec / LSTMCell 的护栏与属性。"""
    spec = layers.LSTMSpec(2, 3, seed=1)
    assert spec.cell == types.CELL_LSTM
    assert spec.fan_in == 5
    assert spec.parameter_count == 72
    assert "lstm(D=2→H=3" in spec.line()
    with pytest.raises(errors.ParameterError):
        layers.LSTMSpec(2, 3, seed=-1)
    with pytest.raises(errors.ParameterError):
        layers.LSTMSpec(0, 3)
    with pytest.raises(errors.ShapeError):
        layers.LSTMCell(wx=((1.0, 0.0),) * 3, wh=((1.0,),) * 3, bias=(0.0,) * 3)
    with pytest.raises(errors.ShapeError):
        layers.LSTMCell(wx=((1.0, 0.0),) * 4, wh=((1.0,),) * 3, bias=(0.0,) * 4)
    with pytest.raises(errors.ShapeError):
        layers.LSTMCell(wx=((1.0, 0.0),) * 4, wh=((1.0, 1.0),) * 4, bias=(0.0,) * 4)
    with pytest.raises(errors.ShapeError):
        layers.LSTMCell(wx=((1.0, 0.0),) * 4, wh=((1.0,),) * 4, bias=(0.0,) * 3)


def test_initialize_lstm_reproducible_and_forget_bias() -> None:
    """同一份 spec 两次初始化逐位相同；**遗忘门偏置是 1**（刻意的例外）。"""
    spec = layers.LSTMSpec(2, 3, seed=5)
    first = layers.initialize_lstm(spec)
    second = layers.initialize_lstm(spec)
    assert first.wx == second.wx and first.wh == second.wh and first.bias == second.bias
    hidden = spec.hidden_size
    assert first.bias[hidden : 2 * hidden] == (types.DEFAULT_FORGET_BIAS,) * hidden
    assert first.hidden_size == 3 and first.input_size == 2
    assert first.parameter_count == 4 * 3 * (2 + 3 + 1)
    assert first.initial_state() == ((0.0,) * 3, (0.0,) * 3)
    assert "lstm cell" in first.line()
    with pytest.raises(errors.ParameterError):
        layers.initialize_lstm("spec")  # type: ignore[arg-type]


def test_lstm_pre_activation_and_gates() -> None:
    """门的打分与四个门的取值范围；形状护栏。"""
    cell = layers.initialize_lstm(layers.LSTMSpec(2, 2, seed=6))
    state = cell.initial_state()
    pre = layers.lstm_pre_activation(cell, (1.0, -1.0), state)
    assert len(pre) == 8
    input_gate, forget_gate, output_gate, candidate = layers.lstm_gates(
        cell, (1.0, -1.0), state
    )
    for gate in (input_gate, forget_gate, output_gate):
        assert all(0.0 < value < 1.0 for value in gate)
    assert all(-1.0 < value < 1.0 for value in candidate)
    with pytest.raises(errors.ShapeError):
        layers.lstm_pre_activation(cell, (1.0,), state)
    with pytest.raises(errors.ShapeError):
        layers.lstm_pre_activation(cell, (1.0, -1.0), ((0.0,), (0.0, 0.0)))


def test_lstm_cell_forward_and_forward() -> None:
    """一步与整段：T 个输入 ⇒ T 个 h 与 T 个 c；初值护栏。"""
    cell = layers.initialize_lstm(layers.LSTMSpec(2, 3, seed=9))
    hidden, state_cell = layers.lstm_cell_forward(cell, (0.5, -0.5), cell.initial_state())
    assert len(hidden) == 3 and len(state_cell) == 3
    hidden_states, cell_states = layers.lstm_forward(cell, study.SAMPLE_SEQUENCE)
    assert len(hidden_states) == len(study.SAMPLE_SEQUENCE)
    assert len(cell_states) == len(study.SAMPLE_SEQUENCE)
    with pytest.raises(errors.ShapeError):
        layers.lstm_forward(cell, study.SAMPLE_SEQUENCE, state0=((0.0,), (0.0,) * 3))
    with pytest.raises(errors.TimeStepError):
        layers.lstm_forward(cell, ())


def test_sequence_length_and_cell_types() -> None:
    """序列长度与单元名单。"""
    assert layers.sequence_length(study.SAMPLE_SEQUENCE) == 4
    assert layers.cell_types() == types.CELL_TYPES
    with pytest.raises(errors.TimeStepError):
        layers.sequence_length(())


# --------------------------------------------------------------------------- 反向


def test_rnn_forward_with_cache_matches_layers() -> None:
    """带缓存的前向与 ``layers.rnn_forward`` 逐位一致（前向只写一遍）。"""
    cell = layers.initialize_rnn(layers.RNNSpec(2, 3, seed=4))
    states, caches = gradients.rnn_forward_with_cache(cell, study.SAMPLE_SEQUENCE)
    assert states == layers.rnn_forward(cell, study.SAMPLE_SEQUENCE)
    assert len(caches) == len(study.SAMPLE_SEQUENCE)
    assert caches[0].state_prev == cell.initial_state()


def test_rnn_forward_with_cache_guards() -> None:
    """带缓存前向的三条护栏（激活 / 初值宽度 / 空序列）。"""
    cell = layers.initialize_rnn(layers.RNNSpec(2, 3, seed=4))
    with pytest.raises(errors.ParameterError):
        gradients.rnn_forward_with_cache(cell, study.SAMPLE_SEQUENCE, activation="gelu")
    with pytest.raises(errors.ShapeError):
        gradients.rnn_forward_with_cache(cell, study.SAMPLE_SEQUENCE, state0=(0.0,))
    with pytest.raises(errors.TimeStepError):
        gradients.rnn_forward_with_cache(cell, ())


def test_rnn_bptt_known_value_and_guards() -> None:
    """BPTT 在一层已知权重上的取值；四条护栏。"""
    cell = layers.RNNCell(wx=((1.0,),), wh=((0.0,),), bias=(0.0,))
    inputs = ((1.0,), (1.0,))
    _states, caches = gradients.rnn_forward_with_cache(cell, inputs)
    result = gradients.rnn_bptt(cell, caches, ((1.0,), (1.0,)))
    # dW_x = Σ_t dz_t·x_t = tanh'(1)·1 + tanh'(1)·1
    expected = 2.0 * (1.0 - math.tanh(1.0) ** 2)
    assert result.cell.wx[0][0] == pytest.approx(expected)
    # dW_h = Σ_t dz_t·h_{t−1} = tanh'(1)·h₁ + tanh'(1)·h₀，而 h₀ = 0、h₁ = tanh(1)
    expected_wh = (1.0 - math.tanh(1.0) ** 2) * math.tanh(1.0)
    assert result.cell.wh[0][0] == pytest.approx(expected_wh)
    assert result.dx[0][0] == pytest.approx(1.0 - math.tanh(1.0) ** 2)
    assert result.norm() > 0.0
    assert result.carry() == 0.0
    assert result.notes
    with pytest.raises(errors.TimeStepError):
        gradients.rnn_bptt(cell, caches, ((1.0,),))
    with pytest.raises(errors.BackwardError):
        gradients.rnn_bptt(cell, caches, ((1.0,), (1.0, 2.0)))
    with pytest.raises(errors.ParameterError):
        gradients.rnn_bptt(cell, caches, ((1.0,), (1.0,)), activation="gelu")
    with pytest.raises(errors.TimeStepError):
        gradients.rnn_bptt(cell, (), ())


def test_rnn_bptt_accumulates_across_time() -> None:
    """**累加**的纪律：T 步的 dW 严格大于任何单步的 dW。"""
    cell = layers.initialize_rnn(layers.RNNSpec(2, 2, seed=3))
    inputs = ((1.0, 0.5), (0.5, -1.0), (-1.0, 0.0))
    _states, caches = gradients.rnn_forward_with_cache(cell, inputs)
    grads = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    full = gradients.rnn_bptt(cell, caches, grads)
    last_only = gradients.rnn_bptt(
        cell, caches, ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    )
    assert full.norm() > last_only.norm()


def test_lstm_forward_with_cache_matches_layers() -> None:
    """带缓存的前向与 ``layers.lstm_forward`` 逐位一致。"""
    cell = layers.initialize_lstm(layers.LSTMSpec(2, 3, seed=4))
    hidden, cell_states, caches = gradients.lstm_forward_with_cache(cell, study.SAMPLE_SEQUENCE)
    expected_hidden, expected_cells = layers.lstm_forward(cell, study.SAMPLE_SEQUENCE)
    assert hidden == expected_hidden
    assert cell_states == expected_cells
    assert len(caches) == len(study.SAMPLE_SEQUENCE)
    assert caches[0].tanh_cell() == tuple(math.tanh(value) for value in caches[0].cell)


def test_lstm_forward_with_cache_guards() -> None:
    """LSTM 带缓存前向的初值护栏。"""
    cell = layers.initialize_lstm(layers.LSTMSpec(2, 3, seed=4))
    with pytest.raises(errors.ShapeError):
        gradients.lstm_forward_with_cache(cell, study.SAMPLE_SEQUENCE, state0=((0.0,), (0.0,) * 3))
    with pytest.raises(errors.TimeStepError):
        gradients.lstm_forward_with_cache(cell, ())


def test_lstm_bptt_shapes_and_values() -> None:
    """LSTM 的 BPTT：产物形状、范数、carry 与四条护栏。"""
    cell = layers.initialize_lstm(layers.LSTMSpec(2, 2, seed=6))
    _h, _c, caches = gradients.lstm_forward_with_cache(cell, study.SAMPLE_SEQUENCE)
    grads = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, -1.0))
    result = gradients.lstm_bptt(cell, caches, grads)
    assert len(result.cell.wx) == 8
    assert len(result.cell.bias) == 8
    assert len(result.dx) == len(study.SAMPLE_SEQUENCE)
    assert result.norm() > 0.0
    assert result.cell_carry() >= 0.0
    assert result.notes
    with pytest.raises(errors.TimeStepError):
        gradients.lstm_bptt(cell, caches, grads, grad_cell=((0.0, 0.0),))
    with pytest.raises(errors.TimeStepError):
        gradients.lstm_bptt(cell, caches, ((0.0, 0.0),))
    with pytest.raises(errors.BackwardError):
        gradients.lstm_bptt(cell, caches, ((0.0, 0.0, 1.0),) * 4)
    with pytest.raises(errors.TimeStepError):
        gradients.lstm_bptt(cell, (), ())


def test_add_gate_blocks_and_gradient_norm() -> None:
    """门梯度的拼接顺序与整体范数；越界与形状护栏。"""
    stacked = gradients.add_gate_blocks((1.0,), (2.0,), (3.0,), (4.0,))
    assert stacked == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(errors.ShapeError):
        gradients.add_gate_blocks((1.0,), (2.0, 3.0), (3.0,), (4.0,))
    assert gradients.gradient_norm((3.0, 4.0)) == pytest.approx(5.0)
    assert gradients.gradient_norm(()) == 0.0
    with pytest.raises(errors.NumericError):
        gradients.gradient_norm((float("nan"),))


def test_carry_norms() -> None:
    """两条 carry 读数：RNN 的 ∂h_T/∂h_0 与 LSTM 的 ∂L/∂c_0。"""
    rnn_cell = layers.RNNCell(wx=((0.0,),), wh=((0.5,),), bias=(0.0,))
    inputs = tuple((0.0,) for _ in range(6))
    carry = gradients.rnn_carry_norm(rnn_cell, inputs, state0=(1.0,))
    assert 0.0 < carry <= 0.5**6
    lstm_cell = layers.LSTMCell(
        wx=((0.0,), (0.0,), (0.0,), (0.0,)),
        wh=((0.0,), (0.0,), (0.0,), (0.0,)),
        bias=(-40.0, 40.0, 40.0, 0.0),
    )
    lstm_carry = gradients.lstm_cell_carry_norm(lstm_cell, inputs, state0=((0.0,), (1.0,)))
    assert lstm_carry == pytest.approx(1.0)


def test_mat_add_guard() -> None:
    """内部累加函数的形状护栏（它保证 `+=` 不会静默地加错形状）。"""
    with pytest.raises(errors.ShapeError):
        gradients._mat_add(((1.0,),), ((1.0, 2.0),))


# --------------------------------------------------------------------------- 网络


def test_build_sequence_net_is_reproducible_and_shaped() -> None:
    """同一配置两次建网逐位相同，形状与参数量与预期一致。"""
    for cell_type in (types.CELL_RNN, types.CELL_LSTM):
        first = network.build_sequence_net(cell_type, input_size=2, hidden_size=3, seed=100)
        second = network.build_sequence_net(cell_type, input_size=2, hidden_size=3, seed=100)
        assert first.cell.wx == second.cell.wx
        assert first.head_weight == second.head_weight
        assert first.cell_type == cell_type
        assert first.classes == 2 and first.hidden_size == 3 and first.input_size == 2
        assert first.parameter_count == first.cell.parameter_count + 2 * 3 + 2
        assert cell_type in first.line()


def test_build_sequence_net_guards() -> None:
    """未知单元 / 类别数过小都拒绝。"""
    with pytest.raises(errors.ParameterError):
        network.build_sequence_net("gru", input_size=2, hidden_size=3)
    with pytest.raises(errors.ParameterError):
        network.build_sequence_net(types.CELL_RNN, input_size=2, hidden_size=3, classes=1)


def test_sequence_net_params_validation() -> None:
    """``SequenceNetParams`` 的列数与偏置长度护栏，以及非单元对象。"""
    params = network.build_sequence_net(types.CELL_RNN, input_size=2, hidden_size=3)
    with pytest.raises(errors.ParameterError):
        network.SequenceNetParams(cell="cell", head_weight=((1.0, 1.0, 1.0),), head_bias=(0.0,))  # type: ignore[arg-type]
    with pytest.raises(errors.ShapeError):
        network.SequenceNetParams(cell=params.cell, head_weight=((1.0, 1.0),), head_bias=(0.0,))
    with pytest.raises(errors.ShapeError):
        network.SequenceNetParams(
            cell=params.cell, head_weight=params.head_weight, head_bias=(0.0,)
        )


def test_sequence_cache_validation() -> None:
    """``SequenceCache`` 的四条护栏（未知单元 / 缺缓存 / 步数不符）。"""
    params = network.build_sequence_net(types.CELL_RNN, input_size=2, hidden_size=3)
    _logits, cache = network.sequence_forward_cached(params, study.SAMPLE_SEQUENCE)
    assert cache.cell_type == types.CELL_RNN
    assert cache.length == 4
    assert cache.rnn_steps is not None and cache.lstm_steps is None
    with pytest.raises(errors.ParameterError):
        network.SequenceCache(cell_type="gru", length=1, last_hidden=(0.0,))
    with pytest.raises(errors.BackwardError):
        network.SequenceCache(cell_type=types.CELL_RNN, length=1, last_hidden=(0.0,))
    with pytest.raises(errors.BackwardError):
        network.SequenceCache(
            cell_type=types.CELL_LSTM, length=1, last_hidden=(0.0,), lstm_steps=None
        )
    with pytest.raises(errors.BackwardError):
        network.SequenceCache(
            cell_type=types.CELL_RNN, length=2, last_hidden=(0.0,), rnn_steps=cache.rnn_steps
        )


def test_sequence_forward_and_predict() -> None:
    """前向给出两个 logits；predict 取最大值下标；输入宽度护栏。"""
    params = network.build_sequence_net(types.CELL_LSTM, input_size=2, hidden_size=3)
    logits = network.sequence_forward(params, study.SAMPLE_SEQUENCE)
    assert len(logits) == 2
    assert network.predict(params, study.SAMPLE_SEQUENCE) in (0, 1)
    with pytest.raises(errors.ShapeError):
        network.sequence_forward_cached(params, ((1.0,), (1.0,)))


def test_loss_and_grad_and_backward_guards() -> None:
    """损失为正、梯度压平长度与参数一致；四条护栏。"""
    params = network.build_sequence_net(types.CELL_LSTM, input_size=2, hidden_size=3)
    loss, grads = network.loss_and_grad(params, study.SAMPLE_SEQUENCE, 0)
    assert loss > 0.0
    assert len(network.flatten_gradients(grads)) == len(network.flatten_params(params))
    assert grads.total_norm() > 0.0
    assert grads.length == 4
    with pytest.raises(errors.BackwardError):
        network.sequence_backward(params, study.SAMPLE_SEQUENCE, (1.0,))
    with pytest.raises(errors.BackwardError):
        _logits, cache = network.sequence_forward_cached(params, study.SAMPLE_SEQUENCE)
        network.sequence_backward(params, ((1.0, 1.0),) * 3, (1.0, 0.0), cache=cache)


def test_sequence_backward_without_cache_matches_with_cache() -> None:
    """不带缓存时重跑一次前向，结果与带缓存逐位相同。"""
    params = network.build_sequence_net(types.CELL_RNN, input_size=2, hidden_size=3)
    logits, cache = network.sequence_forward_cached(params, study.SAMPLE_SEQUENCE)
    probabilities = math.fsum(math.exp(v - max(logits)) for v in logits)
    grad = tuple(math.exp(v - max(logits)) / probabilities for v in logits)
    first = network.sequence_backward(params, study.SAMPLE_SEQUENCE, grad)
    second = network.sequence_backward(params, study.SAMPLE_SEQUENCE, grad, cache=cache)
    assert network.flatten_gradients(first) == network.flatten_gradients(second)


def test_step_norms() -> None:
    """逐步隐状态范数：两种单元都给 T 个数。"""
    for cell_type in (types.CELL_RNN, types.CELL_LSTM):
        params = network.build_sequence_net(cell_type, input_size=2, hidden_size=3)
        norms = network.step_norms(params, study.SAMPLE_SEQUENCE)
        assert len(norms) == len(study.SAMPLE_SEQUENCE)
        assert all(value >= 0.0 for value in norms)


def test_flatten_unflatten_roundtrip() -> None:
    """压平再还原逐位相同（两种单元）。"""
    for cell_type in (types.CELL_RNN, types.CELL_LSTM):
        params = network.build_sequence_net(cell_type, input_size=2, hidden_size=3)
        flat = network.flatten_params(params)
        restored = network.unflatten_params(flat, template=params)
        assert restored.cell.wx == params.cell.wx
        assert restored.cell.wh == params.cell.wh
        assert restored.cell.bias == params.cell.bias
        assert restored.head_weight == params.head_weight
        assert restored.head_bias == params.head_bias


def test_unflatten_rejects_wrong_length_and_non_finite() -> None:
    """长度对不上 / 非有限数都抛 ``ShapeError``。"""
    params = network.build_sequence_net(types.CELL_RNN, input_size=2, hidden_size=3)
    with pytest.raises(errors.ShapeError):
        network.unflatten_params((1.0, 2.0), template=params)
    flat = list(network.flatten_params(params))
    flat[0] = float("inf")
    with pytest.raises(errors.ShapeError):
        network.unflatten_params(tuple(flat), template=params)


# --------------------------------------------------------------------------- 数据集与训练


def test_make_sign_sample_shapes_and_labels() -> None:
    """样本的形状、标签与编码；空 / 非 ±1 / 零和都拒绝。"""
    inputs, label = train.make_sign_sample((1, 1, -1))
    assert label == 1
    assert len(inputs) == 3
    assert all(len(step) == 1 for step in inputs)
    assert train.make_sign_sample((-1, -1, 1))[1] == 0
    with pytest.raises(errors.ParameterError):
        train.make_sign_sample(())
    with pytest.raises(errors.ParameterError):
        train.make_sign_sample((1, 0))
    with pytest.raises(errors.ParameterError):
        train.make_sign_sample((1, -1))


def test_make_sign_dataset_is_balanced_and_rejects_even() -> None:
    """数据集枚举 2^T 条、两类均衡；偶数长度拒绝。"""
    dataset = train.make_sign_dataset(5)
    assert len(dataset) == 32
    assert sum(1 for _inputs, label in dataset if label == 1) == 16
    with pytest.raises(errors.ParameterError):
        train.make_sign_dataset(4)
    with pytest.raises(errors.ParameterError):
        train.make_sign_dataset(0)


def test_accuracy_and_mean_loss_reject_empty() -> None:
    """空数据集没有准确率 / 损失。"""
    params = network.build_sequence_net(types.CELL_RNN, input_size=1, hidden_size=4)
    with pytest.raises(errors.ParameterError):
        train.accuracy(params, ())
    with pytest.raises(errors.ParameterError):
        train.mean_loss(params, ())


def test_check_gradient_norm_guard() -> None:
    """``check_gradient_norm``：正常读数、越界、非有限、非法上界。"""
    assert train.check_gradient_norm((3.0, 4.0), max_norm=10.0) == pytest.approx(5.0)
    with pytest.raises(errors.GradientError):
        train.check_gradient_norm((3.0, 4.0), max_norm=1.0, step=2)
    with pytest.raises(errors.GradientError):
        train.check_gradient_norm((float("inf"),), max_norm=10.0)
    with pytest.raises(errors.ParameterError):
        train.check_gradient_norm((1.0,), max_norm=0.0)
    assert train.MAX_GRADIENT_NORM > 0.0
    assert isinstance(train.check_gradient_norm((3.0, 4.0)), float)


def test_gradient_and_loss_composite() -> None:
    """复合读数：平均损失 + 最后一条样本的梯度 + 平均梯度。"""
    dataset = train.make_sign_dataset(3)
    params = network.build_sequence_net(types.CELL_RNN, input_size=1, hidden_size=4)
    loss, grads, flat = train.gradient_and_loss(params, dataset)
    assert loss > 0.0
    assert isinstance(grads.total_norm(), float)
    assert len(flat) == len(network.flatten_params(params))


def test_train_sequence_learns_the_sign_task(reports) -> None:
    """两种单元都真的学会"和的正负"：准确率 100%、损失显著下降（读数来自共享 fixture）."""
    for cell_type in (types.CELL_RNN, types.CELL_LSTM):
        report = reports[cell_type]
        assert report.cell_type == cell_type
        assert report.final_loss < report.initial_loss
        assert report.accuracy == 1.0
        assert report.improvement > 0.9
        assert report.max_gradient_norm <= train.MAX_GRADIENT_NORM
        assert cell_type in report.summary_line()
        assert report.best_loss <= report.final_loss
        assert len(report.gradient_norms) == report.steps


def test_train_sequence_is_reproducible() -> None:
    """同一配置两次训练给出同一条损失曲线（LCG 决定一切）。"""
    dataset = train.make_sign_dataset(3)
    first = train.train_sequence(types.CELL_RNN, dataset, steps=12, seed=100)[1]
    second = train.train_sequence(types.CELL_RNN, dataset, steps=12, seed=100)[1]
    assert first.losses == second.losses
    assert first.gradient_norms == second.gradient_norms


def test_train_sequence_rejects_bad_inputs() -> None:
    """未知单元 / 空数据集 / 步数为 0 都拒绝。"""
    dataset = train.make_sign_dataset(3)
    with pytest.raises(errors.ParameterError):
        train.train_sequence("gru", dataset, steps=1)
    with pytest.raises(errors.ParameterError):
        train.train_sequence(types.CELL_RNN, (), steps=1)
    with pytest.raises(errors.ParameterError):
        train.train_sequence(types.CELL_RNN, dataset, steps=0)


def test_train_report_improvement_zero_when_initial_zero() -> None:
    """初始损失为 0 时改善率记 0.0（不除零）；无梯度范数时最大范数记 0.0。"""
    report = train.TrainReport(
        cell_type=types.CELL_RNN,
        steps=1,
        losses=(0.0,),
        gradient_norms=(),
        initial_loss=0.0,
        final_loss=0.0,
        best_loss=0.0,
        accuracy=1.0,
        optimizer="adam",
    )
    assert report.improvement == 0.0
    assert report.max_gradient_norm == 0.0


# --------------------------------------------------------------------------- 性质校验


def test_check_three_judgement_types() -> None:
    """``Check`` 的三类判据：相等 / 上界 / 下界。"""
    assert verify.Check(reading=0.0, upper_bound=0.0).passed() is True
    assert verify.Check(reading=1e-13, upper_bound=0.0).passed() is False
    assert verify.Check(reading=2.0, upper_bound=1.0).passed() is False
    assert verify.Check(reading=0.5, upper_bound=1.0).passed() is True
    assert verify.Check(reading=0.9, lower_bound=0.5).passed() is True
    assert verify.Check(reading=0.1, lower_bound=0.5).passed() is False
    assert verify.Check(reading=0.0).bound_text() == "== 逐位"
    assert verify.Check(reading=0.0, upper_bound=0.0).bound_text() == "== 0"
    assert verify.Check(reading=0.0, upper_bound=1e-9).bound_text() == "<= 1.0e-09"
    assert verify.Check(reading=0.0, lower_bound=0.5).bound_text() == ">= 5.0e-01"
    assert (
        verify.Check(reading=0.0, lower_bound=0.1, upper_bound=0.9).bound_text()
        == "∈ [1.0e-01, 9.0e-01]"
    )


def test_check_all_is_seven_and_all_pass() -> None:
    """七条性质全部通过（这就是"这条链被真的接上了"）。"""
    report = verify.check_all()
    assert report.total == 7
    assert report.all_passed()
    assert len(report.lines()) == 7
    assert report.passed == 7


def test_each_check_returns_passing_outcome() -> None:
    """逐条跑：每条都返回通过的 ``PropertyOutcome`` 且读数有限。"""
    for name, function in verify.CHECKS.items():
        outcome = function()
        assert outcome.name == name
        assert outcome.passed is True
        assert math.isfinite(outcome.check.reading)
        assert outcome.line().startswith("通过")


def test_saturated_lstm_helper() -> None:
    """"门被钉死"的 LSTM：f = 1、i = 0、g = 0（逐位）。"""
    cell = verify.saturated_lstm(2, 2)
    assert cell.hidden_size == 2
    assert cell.bias[2:4] == (verify.SATURATED_GATE_BIAS, verify.SATURATED_GATE_BIAS)
    assert all(value == 0.0 for row in cell.wx for value in row)
    gates = layers.lstm_gates(cell, (1.0, 1.0), cell.initial_state())
    assert gates[1] == (1.0, 1.0)
    assert gates[3] == (0.0, 0.0)


def test_check_bptt_matches_numerical_uses_numerical_gradient() -> None:
    """本课命门那条：解析 BPTT 与 day074 的数值差分一致。"""
    outcome = verify.check_bptt_matches_numerical()
    assert outcome.passed is True
    assert outcome.check.reading <= verify.GRAD_TOLERANCE


# --------------------------------------------------------------------------- 七张表


def test_cell_rows_two_units() -> None:
    """单元表两行，且参数量与公式一致。"""
    rows = study.cell_rows()
    assert [row.cell for row in rows] == [types.CELL_RNN, types.CELL_LSTM]
    assert rows[0].parameters == 18
    assert rows[1].parameters == 72
    assert "rnn" in rows[0].line()


def test_gate_rows_cover_four_gates() -> None:
    """门表四行，激活正确，读数落在各自的值域里。"""
    rows = study.gate_rows()
    assert [row.gate for row in rows] == list(types.GATES)
    assert [row.activation for row in rows] == ["sigmoid", "sigmoid", "sigmoid", "tanh"]
    for row in rows[:3]:
        assert 0.0 < row.minimum <= row.mean <= row.maximum < 1.0
    assert -1.0 < rows[3].minimum <= rows[3].maximum < 1.0


def test_state_rows_cover_both_units() -> None:
    """状态表覆盖两种单元的每一步。"""
    rows = study.state_rows()
    assert len(rows) == 2 * len(study.SAMPLE_SEQUENCE)
    assert all(row.norm >= 0.0 for row in rows)
    assert any(row.cell == types.CELL_LSTM for row in rows)


def test_parameter_rows_are_constant_across_lengths() -> None:
    """参数量表：三个 T 下的读数**完全相同**（时间轴权重共享的直接证据）。"""
    rows = study.parameter_rows()
    assert len(rows) == len(study.LENGTH_CASES)
    assert all(row.rnn_parameters == 18 for row in rows)
    assert all(row.lstm_parameters == 72 for row in rows)


def test_bptt_rows_two_units() -> None:
    """反向表两行：解析范数为正，与数值的差落在容差内。"""
    rows = study.bptt_rows()
    assert [row.cell for row in rows] == [types.CELL_RNN, types.CELL_LSTM]
    assert all(row.analytic_norm > 0.0 for row in rows)
    assert all(row.numerical_gap <= study.BPTT_TOLERANCE for row in rows)


def test_train_rows_have_summary_and_points(reports) -> None:
    """训练表：两种单元各一行总结 + 损失点。"""
    lines = study.train_rows(reports)
    assert sum(1 for line in lines if "准确率" in line) == 2
    assert len(lines) >= 6
    assert any(types.CELL_LSTM in line for line in lines)


def test_property_rows_are_seven_and_pass() -> None:
    """性质表七行，全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert all("vs" in row.cross_check for row in rows)


def test_study_lines_has_seven_sections(reports) -> None:
    """一次跑完七张表：七个小节标题都在。"""
    lines = study.study_lines(reports)
    text = "\n".join(lines)
    for index in range(1, 8):
        assert f"== {index}." in text


def test_property_names_alias() -> None:
    """``PROPERTY_NAMES`` 就是性质名单。"""
    assert study.PROPERTY_NAMES == types.RECURRENT_PROPERTIES


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_non_empty_and_has_no_submodules() -> None:
    """包入口的 ``__all__`` 非空、且不含子模块名。"""
    import smart_research_agent.sequence_models as package

    assert len(package.__all__) > 100
    assert not (
        {
            "errors",
            "types",
            "ops",
            "layers",
            "gradients",
            "network",
            "train",
            "verify",
            "study",
        }
        & set(package.__all__)
    )


def test_package_star_import_resolves_every_name() -> None:
    """每个 ``__all__`` 里的名字都能在包命名空间里取到。"""
    import smart_research_agent.sequence_models as package

    assert [name for name in package.__all__ if not hasattr(package, name)] == []


def test_package_exposes_key_objects() -> None:
    """几个关键对象从包入口就能取到。"""
    import smart_research_agent.sequence_models as package

    assert package.rnn_cell_forward is layers.rnn_cell_forward
    assert package.lstm_bptt is gradients.lstm_bptt
    assert package.build_sequence_net is network.build_sequence_net
    assert package.Check is verify.Check
    assert package.train_sequence is train.train_sequence
