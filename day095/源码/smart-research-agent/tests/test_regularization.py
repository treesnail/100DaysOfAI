"""``regularization``：训练技巧与正则化的测试（day095 / M8-D6）.

本文件覆盖新包的九个功能模块与包入口：口径表、四个失败族、BatchNorm
（统计量 / 两相 / running / 三项与一项反向 / 批大小 1 的塌缩）、四个旋钮的转发层、
文本可视化、装上旋钮之后的整条链（**批版本**）、分批训练与消融、七条性质与七张表。

样本全部来自本包自己的确定性构造；**跨天对账真的调用既有包**：
数值差分用 ``math_foundations.calculus``（day074），LayerNorm 用
``encoder_decoder.layers``（day079），dropout 与早停用 ``training_optim``（day081），
BPTT 与梯度守卫用 ``sequence_models``（day094）。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.regularization import (
    errors,
    network,
    normalization,
    study,
    techniques,
    train,
    types,
    verify,
    visualize,
)

BATCH = verify.SAMPLE_BATCH


@pytest.fixture(scope="module")
def ablation() -> dict[str, dict[str, train.TrainReport]]:
    """模块级缓存：四个变体 × 两种单元各训一次（供消融与日志相关用例共用）."""
    return study.ablation_reports()


# --------------------------------------------------------------------------- 口径表


def test_axis_tables_are_closed() -> None:
    """两条归一化轴的四张表逐键对齐。"""
    assert types.NORM_AXES == ("batch", "feature")
    assert set(types.NORM_AXES) == set(types.AXIS_DESCRIPTIONS)
    assert set(types.NORM_AXES) == set(types.AXIS_FORMULAS)
    assert set(types.NORM_AXES) == set(types.AXIS_SIGNATURES)
    assert "批" in types.AXIS_SIGNATURES[types.AXIS_BATCH]
    assert "样本" in types.AXIS_SIGNATURES[types.AXIS_FEATURE]


def test_phase_tables_are_closed() -> None:
    """两个相的三张表逐键对齐，且与 day081 的 dropout 同名。"""
    from smart_research_agent.training_optim.dropout import PHASES as DROPOUT_PHASES

    assert types.PHASES == ("train", "eval")
    assert types.PHASES == DROPOUT_PHASES
    assert set(types.PHASES) == set(types.PHASE_DESCRIPTIONS)
    assert set(types.PHASES) == set(types.PHASE_STATISTIC_SOURCES)


def test_technique_tables_are_closed() -> None:
    """四个技巧的三张表逐键对齐，且来源指向真实的实现者。"""
    assert types.TECHNIQUES == ("batchnorm", "dropout", "lr_decay", "early_stop")
    assert set(types.TECHNIQUES) == set(types.TECHNIQUE_DESCRIPTIONS)
    assert set(types.TECHNIQUES) == set(types.TECHNIQUE_SOURCES)
    assert "本包新建" in types.TECHNIQUE_SOURCES[types.TECHNIQUE_BATCHNORM]
    assert "day081" in types.TECHNIQUE_SOURCES[types.TECHNIQUE_DROPOUT]
    assert "day074" in types.TECHNIQUE_SOURCES[types.TECHNIQUE_LR_DECAY]
    assert "day081" in types.TECHNIQUE_SOURCES[types.TECHNIQUE_EARLY_STOP]


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单 / 说明 / "失败意味着什么"三张表逐键对齐。"""
    assert len(types.REGULARIZATION_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.REGULARIZATION_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.REGULARIZATION_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())
    assert types.PROPERTIES == types.REGULARIZATION_PROPERTIES


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.REGULARIZATION_NOTES) == 10
    assert types.NOTES_ORDER == tuple(types.REGULARIZATION_NOTES)
    assert all(types.REGULARIZATION_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）。"""
    assert len(types.REGULARIZATION_BOUNDARIES) == 5
    assert all(types.REGULARIZATION_BOUNDARIES)


def test_torch_counterparts_cover_core_ops_and_no_version() -> None:
    """PyTorch 对照表覆盖 BatchNorm / 两相 / dropout / 调度，且**不出现**版本号。"""
    required = {"batchnorm", "running_stats", "phases", "dropout", "lr_schedule", "early_stop"}
    assert required <= set(types.TORCH_COUNTERPARTS)
    assert all(types.TORCH_COUNTERPARTS.values())
    assert "torch==" not in " ".join(types.TORCH_COUNTERPARTS.values())


def test_formulas_are_non_empty() -> None:
    """四张公式都写下了一行可读的式子。"""
    assert "x̂" in types.BN_FORMULA
    assert "μ_run" in types.RUNNING_UPDATE_FORMULA
    assert "mean_batch" in types.BN_BACKWARD_FORMULA
    assert "min_delta" in types.EARLY_STOP_FORMULA
    assert types.DEFAULT_EPSILON == 1e-5
    assert types.DEFAULT_MOMENTUM == 0.1


def test_tolerance_is_same_as_day079_epsilon() -> None:
    """本包的 ε 与 day079 的 LayerNorm **同值**（第 ② 条对账的前提）。"""
    from smart_research_agent.encoder_decoder.types import DEFAULT_EPSILON as DAY079_EPSILON

    assert types.DEFAULT_EPSILON == DAY079_EPSILON


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_aligned() -> None:
    """四个族的"该怎么办"表与类表逐键对齐。"""
    assert len(errors.FAMILY_OUTCOMES) == 4
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "PhaseError",
    }


def test_returned_and_absent_families_are_constants() -> None:
    """本课没有"回来"的族，且 ``GradientError`` 再次缺席（可断言的事实）。"""
    assert errors.RETURNED_FAMILY is None
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.ABSENT_FAMILY_REASON


def test_errors_inherit_both_families() -> None:
    """本族错误既是 ``RegularizationError``、也是 day075 的对应族。"""
    from smart_research_agent.transformer_core import errors as core_errors

    assert issubclass(errors.ShapeError, errors.RegularizationError)
    assert issubclass(errors.ShapeError, core_errors.ShapeError)
    assert issubclass(errors.ParameterError, core_errors.ParameterError)
    assert issubclass(errors.NumericError, core_errors.NumericError)
    assert issubclass(errors.PhaseError, errors.RegularizationError)


# --------------------------------------------------------------------------- BatchNorm


def test_epsilon_and_momentum_guards() -> None:
    """ε 必须为正、momentum 必须落在 [0, 1]、相必须是两个名字之一。"""
    assert normalization._checked_epsilon(1e-5) == 1e-5
    with pytest.raises(errors.ParameterError):
        normalization._checked_epsilon(0.0)
    with pytest.raises(errors.ParameterError):
        normalization._checked_epsilon("x")
    assert normalization._checked_momentum(0.1) == 0.1
    with pytest.raises(errors.ParameterError):
        normalization._checked_momentum(1.5)
    with pytest.raises(errors.ParameterError):
        normalization._checked_momentum("x")
    assert normalization._checked_phase("eval") == "eval"
    with pytest.raises(errors.PhaseError):
        normalization._checked_phase("infer")


def test_batch_shape_helpers() -> None:
    """``as_matrix`` / ``as_vector`` / ``matrix_shape`` / ``transpose``. """
    assert normalization.as_matrix(BATCH) == BATCH
    assert normalization.as_vector((1, 2.5), name="v") == (1.0, 2.5)
    with pytest.raises(errors.ShapeError):
        normalization.as_matrix(())
    with pytest.raises(errors.ShapeError):
        normalization.as_matrix(((1.0,), (2.0, 3.0)))
    with pytest.raises(errors.ShapeError):
        normalization.as_matrix(((),))
    with pytest.raises(errors.NumericError):
        normalization.as_matrix(((float("inf"),),))
    with pytest.raises(errors.ShapeError):
        normalization.as_matrix("batch")
    with pytest.raises(errors.ShapeError):
        normalization.as_matrix(("x",))
    with pytest.raises(errors.ShapeError):
        normalization.as_vector((), name="v")
    with pytest.raises(errors.ShapeError):
        normalization.as_vector(("a",), name="v")
    with pytest.raises(errors.NumericError):
        normalization.as_vector((float("nan"),), name="v")
    assert normalization.matrix_shape(BATCH) == (4, 3)
    assert normalization.transpose(((1.0, 2.0), (3.0, 4.0))) == ((1.0, 3.0), (2.0, 4.0))


def test_batch_statistics_known_value() -> None:
    """逐特征均值与**有偏**方差（分母是 N）。"""
    stats = normalization.batch_statistics(((1.0, 2.0), (3.0, 4.0)))
    assert stats.mean == (2.0, 3.0)
    assert stats.variance == (1.0, 1.0)
    assert stats.features == 2
    assert "σ²" in stats.line()
    assert normalization.BatchStatistics(mean=(), variance=()).line() == "空统计量"


def test_running_update_and_initial() -> None:
    """running 的初值与折入；momentum = 0 / 1 的两个端点。"""
    running = normalization.initial_running(2)
    assert running.mean == (0.0, 0.0)
    assert running.variance == (1.0, 1.0)
    assert running.batches == 0
    assert "running" in running.line()
    assert normalization.RunningStatistics(mean=(), variance=()).line() == "空 running 统计量"
    stats = normalization.BatchStatistics(mean=(2.0, 4.0), variance=(0.5, 0.5))
    frozen = normalization.running_update(running, stats, momentum=0.0)
    assert frozen.mean == (0.0, 0.0)
    fresh = normalization.running_update(running, stats, momentum=1.0)
    assert fresh.mean == (2.0, 4.0)
    assert fresh.batches == 1
    with pytest.raises(errors.ShapeError):
        normalization.running_update(normalization.initial_running(3), stats)
    with pytest.raises(errors.ParameterError):
        normalization.initial_running(0)


def test_batch_norm_forward_default_affine_and_running() -> None:
    """γ=1、β=0 时输出就是纯粹的标准化；训练相会把 running 往这批靠一步。"""
    output, cache, running = normalization.batch_norm_forward(BATCH)
    assert cache.phase == types.PHASE_TRAIN
    assert cache.gamma == (1.0,) * 3
    assert cache.features == 3
    assert running.batches == 1
    # 每一列的均值在浮点误差内是 0（γ=1、β=0 时输出的列和应当约掉）
    for column in range(3):
        total = math.fsum(row[column] for row in output)
        assert total == pytest.approx(0.0, abs=1e-12)
    for row in output:
        assert len(row) == 3


def test_batch_norm_forward_phase_and_shape_guards() -> None:
    """推理相必须给出 running；γ/β 的长度必须等于特征数。"""
    with pytest.raises(errors.PhaseError):
        normalization.batch_norm_forward(BATCH, phase=types.PHASE_EVAL)
    running = normalization.RunningStatistics(mean=(0.0,) * 3, variance=(1.0,) * 3, batches=1)
    output, cache, returned = normalization.batch_norm_forward(
        BATCH, phase=types.PHASE_EVAL, running=running
    )
    assert cache.phase == types.PHASE_EVAL
    assert returned is running
    assert len(output) == len(BATCH)
    with pytest.raises(errors.PhaseError):
        normalization.batch_norm_forward(BATCH, phase="infer")
    with pytest.raises(errors.ShapeError):
        normalization.batch_norm_forward(BATCH, gamma=(1.0, 1.0))
    with pytest.raises(errors.ShapeError):
        normalization.batch_norm_forward(BATCH, beta=(0.0, 0.0))
    with pytest.raises(errors.ShapeError):
        normalization.batch_norm_forward(BATCH, phase=types.PHASE_EVAL, running=normalization.initial_running(2))


def test_batch_norm_backward_train_three_terms_known_value() -> None:
    """训练相的反向在"均匀回传"的批上给出一个可手算的读数：**两项正好抵消 ⇒ dx = 0**.

    ```text
    x̂ 每一列的和是 0（标准化的定义）
    dy 全 1、γ 全 2  ⇒  dŷ = 2、mean(dŷ) = 2、mean(dŷ⊙x̂) = 0
    dx = (1/σ)(2 − 2 − x̂·0) = 0
    ```
    训练相的三项在这里**恰好**抵消——而推理相（只有一项）会给出一个非零的 dx，
    两条路径因此从同一个缓存上分家（第 ④ / ⑤ 条性质各钉一条）。
    """
    batch = ((1.0, -1.0), (3.0, 1.0))
    output, cache, _running = normalization.batch_norm_forward(batch, gamma=(2.0, 2.0))
    grads = normalization.batch_norm_backward(cache, ((1.0, 1.0), (1.0, 1.0)))
    assert grads.d_gamma == pytest.approx((0.0, 0.0), abs=1e-12)
    assert grads.d_beta == (2.0, 2.0)
    assert all(abs(value) <= 1e-12 for row in grads.d_inputs for value in row)
    assert grads.norm() == pytest.approx(math.sqrt(8.0))
    assert len(output) == 2


def test_batch_norm_backward_eval_single_term() -> None:
    """推理相的反向只剩一项：``dx = γ⊙dy/σ``（与训练相的读数**不同**）."""
    running = normalization.RunningStatistics(mean=(0.0,) * 3, variance=(0.25,) * 3, batches=5)
    _output, cache, _running = normalization.batch_norm_forward(
        BATCH, gamma=(1.0, 1.0, 1.0), phase=types.PHASE_EVAL, running=running
    )
    grads = normalization.batch_norm_backward(cache, ((1.0,) * 3,) * 4)
    sigma = math.sqrt(0.25 + 1e-5)
    for row in grads.d_inputs:
        for value in row:
            assert value == pytest.approx(1.0 / sigma, rel=1e-12)


def test_batch_norm_backward_guards() -> None:
    """反向的四种护栏（非 cache / 列数 / 行数 / β 长度）."""
    _output, cache, _running = normalization.batch_norm_forward(BATCH)
    with pytest.raises(errors.ParameterError):
        normalization.batch_norm_backward("cache", BATCH)  # type: ignore[arg-type]
    with pytest.raises(errors.ShapeError):
        normalization.batch_norm_backward(cache, ((1.0, 1.0),) * 4)
    with pytest.raises(errors.ShapeError):
        normalization.batch_norm_backward(cache, ((1.0, 1.0, 1.0),))
    with pytest.raises(errors.ShapeError):
        normalization.batch_norm_backward(cache, BATCH, beta=(0.0, 0.0))


def test_batch_size_one_collapses_to_beta() -> None:
    """批大小为 1 时方差恒为 0 ⇒ 输出恒为 β（**这是 BatchNorm 的签名**）."""
    report = normalization.batch_size_one_report(((0.4, -1.2, 2.0),))
    assert report["rows"] == 1.0
    assert report["max_variance"] == 0.0
    assert report["max_abs_output"] == 0.0
    beta_probe, _cache, _running = normalization.batch_norm_forward(
        ((0.4, -1.2, 2.0),), beta=(5.0, -3.0, 1.0)
    )
    assert beta_probe == ((5.0, -3.0, 1.0),)


def test_batch_norm_backward_matches_numerical_standalone() -> None:
    """裸 BatchNorm 的两相反向都与数值差分一致（本课命门的直接形态）."""
    for phase in (types.PHASE_TRAIN, types.PHASE_EVAL):
        outcome = (
            verify.check_batchnorm_backward_matches_numerical()
            if phase == types.PHASE_TRAIN
            else verify.check_eval_backward_matches_numerical()
        )
        assert outcome.passed is True
        assert outcome.check.reading <= verify.GRAD_TOLERANCE


# --------------------------------------------------------------------------- 四个旋钮的转发


def test_dropout_vector_train_and_eval() -> None:
    """训练相置零 + 放大；推理相恒等（与 rate=0 的训练相逐位相同）。"""
    sample = (1.0, 2.0, 3.0, 4.0)
    trained, mask, scale = techniques.dropout_vector(sample, rate=0.5, seed=9)
    assert scale == 2.0
    assert len(mask) == 1 and len(mask[0]) == 4
    for index, (value, flag) in enumerate(zip(trained, mask[0], strict=True)):
        assert value == (sample[index] * scale if flag else 0.0)
    evaluation, eval_mask, eval_scale = techniques.dropout_vector(
        sample, rate=0.5, seed=9, phase=types.PHASE_EVAL
    )
    assert evaluation == sample
    assert eval_scale == 1.0
    assert eval_mask == ((1.0,) * 4,)
    zero, _zero_mask, _zero_scale = techniques.dropout_vector(sample, rate=0.0, seed=9)
    assert zero == sample
    assert techniques.dropout_vector_backward(sample, mask, scale)[0] in (0.0, 2.0, 4.0, 6.0, 8.0)
    matrix = techniques.dropout_vector_backward_matrix((sample,), mask, scale)
    assert matrix[0] == techniques.dropout_vector_backward(sample, mask, scale)


def test_dropout_vector_guards() -> None:
    """rate 越界与相不成立都拒绝。"""
    with pytest.raises(errors.ParameterError):
        techniques.dropout_vector((1.0,), rate=1.0, seed=1)
    with pytest.raises(errors.ParameterError):
        techniques.dropout_vector((1.0,), rate=-0.1, seed=1)
    with pytest.raises(errors.PhaseError):
        techniques.dropout_vector((1.0,), rate=0.5, seed=1, phase="infer")


def test_learning_rate_at_forwards_day074() -> None:
    """四种调度各给一条曲线；步号与非法参数都有护栏。"""
    constant = techniques.learning_rate_at(3, base_lr=0.1, schedule="constant")
    assert constant == pytest.approx(0.1)
    cosine_start = techniques.learning_rate_at(
        1, base_lr=0.1, schedule="cosine", total_steps=11, min_lr=0.0
    )
    cosine_end = techniques.learning_rate_at(
        11, base_lr=0.1, schedule="cosine", total_steps=11, min_lr=0.0
    )
    assert cosine_start == pytest.approx(0.1)
    assert cosine_end == pytest.approx(0.0, abs=1e-12)
    assert techniques.learning_rate_at(
        3, base_lr=0.1, schedule="step_decay", drop_every=2, gamma=0.5
    ) == pytest.approx(0.05)
    assert techniques.learning_rate_at(
        2, base_lr=0.1, schedule="warmup_cosine", warmup_steps=4, total_steps=12
    ) == pytest.approx(0.05)
    with pytest.raises(errors.ParameterError):
        techniques.learning_rate_at(0, base_lr=0.1, schedule="constant")
    with pytest.raises(errors.ParameterError):
        techniques.learning_rate_at(1, base_lr=0.1, schedule="nope")
    with pytest.raises(errors.ParameterError):
        techniques.learning_rate_at(1, base_lr="x", schedule="constant")  # type: ignore[arg-type]
    with pytest.raises(errors.ParameterError):
        techniques.learning_rate_at(1, base_lr=0.1, schedule="cosine")


def test_early_stopping_watch() -> None:
    """把一串损失交给 day081 的早停；单调下降时永不触发。"""
    monotone = techniques.early_stopping_watch((1.0, 0.9, 0.8, 0.7), patience=2)
    assert monotone.triggered is False
    assert monotone.best_step == 4
    flat = techniques.early_stopping_watch((1.0, 0.9, 0.9, 0.9, 0.9), patience=3)
    assert flat.triggered is True
    assert flat.best_step == 2
    with pytest.raises(errors.ParameterError):
        techniques.early_stopping_watch((), patience=1)


def test_regularization_config_validation_and_line() -> None:
    """配置的每个字段都有护栏；enabled 与 line 可读。"""
    config = techniques.RegularizationConfig()
    assert config.norm is True
    assert config.enabled() == ("batchnorm", "dropout", "lr_decay")
    assert "norm=on" in config.line()
    bare = techniques.RegularizationConfig(
        norm=False, dropout_rate=0.0, schedule="constant", patience=0
    )
    assert bare.enabled() == ()
    assert "norm=off" in bare.line()
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(dropout_rate=1.0)
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(bn_momentum=1.5)
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(bn_momentum="x")  # type: ignore[arg-type]
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(dropout_seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(patience=-1)
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(base_lr=float("inf"))
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(min_lr="x")  # type: ignore[arg-type]


def test_schedule_kwargs_for_four_schedules() -> None:
    """四种调度各给出一份精确的参数表（多传一个就会 TypeError）。"""
    assert techniques.RegularizationConfig(schedule="constant").schedule_kwargs(10) == {
        "base_lr": 0.05
    }
    cosine = techniques.RegularizationConfig(schedule="cosine").schedule_kwargs(10)
    assert cosine == {"base_lr": 0.05, "total_steps": 10, "min_lr": 0.0}
    warmup = techniques.RegularizationConfig(schedule="warmup_cosine").schedule_kwargs(10)
    assert warmup["warmup_steps"] == 1
    decay = techniques.RegularizationConfig(schedule="step_decay").schedule_kwargs(10)
    assert decay == {"base_lr": 0.05, "drop_every": 2, "gamma": 0.5}
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig(schedule="nope").schedule_kwargs(10)
    with pytest.raises(errors.ParameterError):
        techniques.RegularizationConfig().schedule_kwargs(0)


# --------------------------------------------------------------------------- 可视化


def test_sparkline_shape_and_extremes() -> None:
    """sparkline 的长度等于序列长度，极值位置与 argmin / argmax 一致。"""
    values = (0.9, 0.7, 0.5, 0.3, 0.1, 0.55)
    drawn = visualize.sparkline(values, width=len(values))
    assert len(drawn) == len(values)
    assert drawn[4] == visualize.GLYPHS[0]
    assert drawn[0] == visualize.GLYPHS[-1]
    flat = visualize.sparkline((1.0, 1.0, 1.0), width=3)
    assert flat == visualize.GLYPHS[(len(visualize.GLYPHS) - 1) // 2] * 3
    downsampled = visualize.sparkline(tuple(float(i) for i in range(100)), width=10)
    assert len(downsampled) == 10
    assert len(visualize.sparkline((1.0, 2.0), width=1)) == 1
    with pytest.raises(errors.ParameterError):
        visualize.sparkline((1.0,), width=0)


def test_loss_curve_shape_and_labels() -> None:
    """折线图含最大值与最小值两个标注，且行数 = 高度 + 两行说明。"""
    values = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
    lines = visualize.loss_curve(values, height=4, width=9)
    assert len(lines) == 4 + 2
    assert "0.900000" in lines[0]
    assert "0.100000" in lines[3]
    assert any("*" in line for line in lines)
    with pytest.raises(errors.ParameterError):
        visualize.loss_curve(values, height=2)
    with pytest.raises(errors.ParameterError):
        visualize.loss_curve(values, width=1)


def test_bar_chart_and_guards() -> None:
    """条形图的长度与数值成正比；空表 / 负值 / 坏项都拒绝。"""
    lines = visualize.bar_chart((("a", 1.0), ("bb", 0.5)), width=4)
    assert len(lines) == 2
    assert lines[0].count("█") == 4
    assert lines[1].count("█") == 2
    assert "bb" in lines[1]
    zeros = visualize.bar_chart((("a", 0.0), ("b", 0.0)), width=3)
    assert all("█" not in line for line in zeros)
    with pytest.raises(errors.ShapeError):
        visualize.bar_chart(())
    with pytest.raises(errors.ShapeError):
        visualize.bar_chart((("a",),))  # type: ignore[arg-type]
    with pytest.raises(errors.ShapeError):
        visualize.bar_chart((("", 1.0),))
    with pytest.raises(errors.ParameterError):
        visualize.bar_chart((("a", -1.0),))
    with pytest.raises(errors.ParameterError):
        visualize.bar_chart((("a", 1.0),), width=0)


def test_summary_block_and_curve_summary() -> None:
    """两列表与一行汇总裁剪。"""
    block = visualize.summary_block("标题", (("a", "1"), ("bb", "2")))
    assert block[0] == "标题"
    assert "  a : 1" in block[1]
    assert " bb : 2" in block[2]
    with pytest.raises(errors.ParameterError):
        visualize.summary_block("", ())
    with pytest.raises(errors.ShapeError):
        visualize.summary_block("t", (("a",),))  # type: ignore[arg-type]
    summary = visualize.curve_summary((0.9, 0.5, 0.1, 0.4))
    assert "4 步" in summary
    assert "最好第 3 步" in summary
    assert "最坏第 1 步" in summary


# --------------------------------------------------------------------------- 装上旋钮的链


def test_as_batch_guards() -> None:
    """空批 / 坏项 / 非整数标签都拒绝。"""
    good = verify.SAMPLE_BATCH
    assert network.as_batch(((good, 0),)) is not None
    with pytest.raises(errors.ShapeError):
        network.as_batch(())
    with pytest.raises(errors.ShapeError):
        network.as_batch("batch")
    with pytest.raises(errors.ShapeError):
        network.as_batch((((1.0,),),))  # type: ignore[arg-type]
    with pytest.raises(errors.ShapeError):
        network.as_batch(((good, "0"),))  # type: ignore[arg-type]


def test_regularized_params_validation() -> None:
    """γ/β 的长度必须等于隐藏宽；sequence 必须是 day094 的参数本体。"""
    params = network.build_regularized("rnn", input_size=1, hidden_size=3, classes=2)
    assert params.features == 3
    assert params.parameter_count == params.sequence.parameter_count + 6
    assert "bn(3)" in params.line()
    with pytest.raises(errors.ParameterError):
        network.RegularizedParams(sequence="net", gamma=(1.0,), beta=(0.0,))  # type: ignore[arg-type]
    with pytest.raises(errors.ShapeError):
        network.RegularizedParams(sequence=params.sequence, gamma=(1.0, 1.0), beta=(0.0,) * 3)
    with pytest.raises(errors.ShapeError):
        network.RegularizedParams(sequence=params.sequence, gamma=(1.0,) * 3, beta=(0.0, 0.0))


def test_forward_cached_shapes_and_two_phases() -> None:
    """前向给出 (N×C) 的 logits；BN 的输出是 N×H；两相在同一个批上不同。"""
    batch = (verify.SAMPLE_BATCH, 0)
    for cell_type in ("rnn", "lstm"):
        params = network.build_regularized(cell_type, input_size=3, hidden_size=3)
        logits, cache = network.regularized_forward_cached(
            params, (batch,), dropout_rate=0.0
        )
        assert len(logits) == 1 and len(logits[0]) == 2
        assert cache.rows == 1
        assert cache.features == 3
        assert cache.normalized == cache.bn.normalized if cache.bn else True
        assert cache.phase == types.PHASE_TRAIN
        assert cache.running.batches == 1
        # 批大小为 1 时 BN 塌成 β = 0，因此 logits 就是 head 的偏置
        assert logits[0] == pytest.approx(tuple(params.sequence.head_bias))


def test_forward_eval_requires_running() -> None:
    """推理相的前向必须给出 running（否则抛 PhaseError）."""
    params = network.build_regularized("lstm", input_size=3, hidden_size=3)
    batch = (verify.SAMPLE_BATCH, 0)
    with pytest.raises(errors.PhaseError):
        network.regularized_forward_cached(params, (batch,), phase=types.PHASE_EVAL)
    _logits, cache = network.regularized_forward_cached(params, (batch,))
    evaluation = network.regularized_forward(
        params, (batch,), phase=types.PHASE_EVAL, running=cache.running
    )
    assert len(evaluation) == 1
    with pytest.raises(errors.PhaseError):
        network.regularized_forward_cached(params, (batch,), phase="infer")


def test_forward_without_norm_is_identity_bn() -> None:
    """``use_norm=False`` 时 BN 整层跳过（γ/β 的梯度为 0）."""
    params = network.build_regularized("rnn", input_size=3, hidden_size=3)
    batch = (verify.SAMPLE_BATCH, 0)
    _logits, cache = network.regularized_forward_cached(params, (batch,), use_norm=False)
    assert cache.bn is None
    assert cache.normalized == cache.last_hidden
    loss, grads, _same = network.loss_and_grad(params, (batch,), use_norm=False)
    assert loss > 0.0
    assert grads.d_gamma == (0.0, 0.0, 0.0)
    assert grads.d_beta == (0.0, 0.0, 0.0)
    assert grads.total_norm() > 0.0


def test_backward_guards() -> None:
    """反向的四条护栏（非 cache / 行数 / 列数）。"""
    params = network.build_regularized("rnn", input_size=1, hidden_size=3)
    dataset = train.shard_batches(_sign_samples(4), 4)[0]
    _logits, cache = network.regularized_forward_cached(params, dataset)
    with pytest.raises(errors.ParameterError):
        network.regularized_backward(params, "cache", ((1.0, 0.0),) * 4)  # type: ignore[arg-type]
    with pytest.raises(errors.ShapeError):
        network.regularized_backward(params, cache, ((1.0, 0.0),))
    with pytest.raises(errors.ShapeError):
        network.regularized_backward(params, cache, ((1.0,),) * 4)


def test_backward_rejects_ragged_batch() -> None:
    """一批里序列长度不一致时，反向当场拒绝（BN 不关心长度，**BPTT 关心**）."""
    params = network.build_regularized("rnn", input_size=1, hidden_size=3)
    short = (((1.0,), (1.0,)), 1)
    long = (((1.0,), (1.0,), (1.0,)), 0)
    _logits, cache = network.regularized_forward_cached(params, (short, long))
    with pytest.raises(errors.ShapeError):
        network.regularized_backward(params, cache, ((1.0, 0.0), (1.0, 0.0)))


def _sign_samples(count: int):
    """取 day094 的符号任务里的前 ``count`` 条样本（供若干用例共用）."""
    from smart_research_agent.sequence_models.train import make_sign_dataset

    return make_sign_dataset(3)[:count]


def test_loss_and_grad_are_consistent() -> None:
    """损失为正、梯度压平长度与参数一致、两种单元的整链梯度都能算。"""
    batch = _sign_samples(6)
    for cell_type in ("rnn", "lstm"):
        params = network.build_regularized(cell_type, input_size=1, hidden_size=3)
        loss, grads, cache = network.loss_and_grad(params, batch, dropout_rate=0.2, dropout_seed=3)
        assert loss > 0.0
        assert cache.rows == 6
        assert len(network.flatten_gradients(grads)) == len(network.flatten_params(params))
        assert grads.total_norm() > 0.0


def test_accuracy_train_and_eval() -> None:
    """准确率在两个相上都能算（推理相用 running）。"""
    batch = _sign_samples(8)
    params = network.build_regularized("lstm", input_size=1, hidden_size=4)
    _logits, cache = network.regularized_forward_cached(params, batch)
    evaluation = network.accuracy(params, batch, phase=types.PHASE_EVAL, running=cache.running)
    training = network.accuracy(params, batch, phase=types.PHASE_TRAIN)
    assert 0.0 <= evaluation <= 1.0
    assert 0.0 <= training <= 1.0
    assert network.accuracy(params, batch, phase=types.PHASE_EVAL, running=cache.running, use_norm=False) >= 0.0


def test_flatten_unflatten_roundtrip() -> None:
    """压平再还原逐位相同（两种单元）。"""
    for cell_type in ("rnn", "lstm"):
        params = network.build_regularized(cell_type, input_size=2, hidden_size=3)
        flat = network.flatten_params(params)
        restored = network.unflatten_params(flat, template=params)
        assert network.flatten_params(restored) == flat
        assert restored.gamma == params.gamma
        assert restored.beta == params.beta
        assert restored.sequence.cell.bias == params.sequence.cell.bias


def test_unflatten_rejects_wrong_length_and_non_finite() -> None:
    """长度对不上 / 非有限数都抛 ``ShapeError``。"""
    params = network.build_regularized("rnn", input_size=1, hidden_size=3)
    with pytest.raises(errors.ShapeError):
        network.unflatten_params((1.0, 2.0), template=params)
    flat = list(network.flatten_params(params))
    flat[0] = float("nan")
    with pytest.raises(errors.ShapeError):
        network.unflatten_params(tuple(flat), template=params)


# --------------------------------------------------------------------------- 训练


def test_shard_batches_guards_and_split() -> None:
    """分批：顺序切、最后一批可不满；空数据集与批大小 1 都拒绝。"""
    dataset = _sign_samples(8)
    batches = train.shard_batches(dataset, 3)
    assert len(batches) == 3
    assert len(batches[0]) == 3 and len(batches[-1]) == 2
    with pytest.raises(errors.ParameterError):
        train.shard_batches((), 2)
    with pytest.raises(errors.ParameterError):
        train.shard_batches(dataset, 1)
    with pytest.raises(errors.ParameterError):
        train.shard_batches(dataset, 2.0)  # type: ignore[arg-type]


def test_epoch_record_line() -> None:
    """一个 epoch 的一行读数。"""
    record = train.EpochRecord(epoch=3, train_loss=0.5, eval_loss=0.25, eval_accuracy=1.0, learning_rate=0.01)
    assert "epoch   3" in record.line()
    assert "准确率 100.0%" in record.line()


def test_train_regularized_learns_lstm(ablation) -> None:
    """lstm 上四个变体都学会（读数来自共享 fixture）。"""
    reports = ablation["lstm"]
    assert len(reports) == len(study.ABLATION_VARIANTS)
    for report in reports.values():
        assert report.accuracy == 1.0
        assert report.final_loss < report.initial_loss
        assert report.steps == len(report.losses)
        assert report.max_gradient_norm <= train.GRADIENT_NORM_GUARD
        assert "准确率 100.0%" in report.summary_line()
        # running 统计量只在**开着 BN** 时才会被更新（关掉 BN 时它保持初值）
        if report.config.norm:
            assert report.running.batches == report.steps
        else:
            assert report.running.batches == 0


def test_train_regularized_reports_gap_and_early_stop(ablation) -> None:
    """消融表的核心读数：训练 / 推理之间的**间隔**与早停是否触发。"""
    rnn_reports = ablation["rnn"]
    assert rnn_reports["无归一化"].loss_gap < 0.1
    assert rnn_reports["无丢弃"].loss_gap > 0.5
    assert rnn_reports["无丢弃"].early_stop is not None
    assert rnn_reports["无丢弃"].early_stop.triggered is True
    assert ablation["lstm"]["全开"].early_stop is not None
    assert ablation["lstm"]["全开"].early_stop.triggered is False


def test_report_derived_readings() -> None:
    """报告的派生读数：improvement / max_gradient_norm / loss_gap（零初始损失不除零）。"""
    report = train.TrainReport(
        cell_type="rnn",
        config=techniques.RegularizationConfig(),
        steps=1,
        batch_size=2,
        losses=(0.0,),
        learning_rates=(0.05,),
        gradient_norms=(),
        epochs=(),
        initial_loss=0.0,
        final_loss=0.0,
        eval_loss=0.0,
        accuracy=1.0,
        early_stop=None,
        running=normalization.initial_running(3),
    )
    assert report.improvement == 0.0
    assert report.max_gradient_norm == 0.0
    assert report.loss_gap == 0.0


def test_train_regularized_guards() -> None:
    """未知单元 / 空数据集 / 非法 epoch 都拒绝。"""
    dataset = _sign_samples(4)
    with pytest.raises(errors.ParameterError):
        train.train_regularized("gru", dataset, epochs=1)
    with pytest.raises(errors.ParameterError):
        train.train_regularized("rnn", (), epochs=1)
    with pytest.raises(errors.ParameterError):
        train.train_regularized("rnn", dataset, epochs=0)


def test_train_is_reproducible() -> None:
    """同一配置两次训练给出同一批损失（LCG 与顺序切批决定一切）。"""
    dataset = _sign_samples(4)
    first = train.train_regularized("rnn", dataset, epochs=2, batch_size=2)[1]
    second = train.train_regularized("rnn", dataset, epochs=2, batch_size=2)[1]
    assert first.losses == second.losses
    assert first.eval_loss == second.eval_loss


def test_compare_configs_guards_and_evaluate() -> None:
    """空变体表拒绝；``evaluate`` 给出一对读数。"""
    dataset = _sign_samples(4)
    with pytest.raises(errors.ParameterError):
        train.compare_configs("rnn", dataset, {})
    params = network.build_regularized("rnn", input_size=1, hidden_size=3)
    loss, accuracy = train.evaluate(
        params, dataset, running=normalization.initial_running(3), batch_size=2
    )
    assert loss > 0.0
    assert 0.0 <= accuracy <= 1.0


# --------------------------------------------------------------------------- 性质校验


def test_check_three_judgement_types() -> None:
    """``Check`` 的三类判据：相等 / 上界 / 下界。"""
    assert verify.Check(reading=0.0, upper_bound=0.0).passed() is True
    assert verify.Check(reading=1e-13, upper_bound=0.0).passed() is False
    assert verify.Check(reading=0.9, lower_bound=0.5).passed() is True
    assert verify.Check(reading=0.1, lower_bound=0.5).passed() is False
    assert verify.Check(reading=0.0).bound_text() == "== 逐位"
    assert verify.Check(reading=0.0, upper_bound=0.0).bound_text() == "== 0"
    assert verify.Check(reading=0.0, upper_bound=1e-7).bound_text() == "<= 1.0e-07"
    assert verify.Check(reading=0.0, lower_bound=0.5).bound_text() == ">= 5.0e-01"
    assert (
        verify.Check(reading=0.0, lower_bound=0.1, upper_bound=0.9).bound_text()
        == "∈ [1.0e-01, 9.0e-01]"
    )


def test_check_all_is_seven_and_all_pass() -> None:
    """七条性质全部通过（这就是"这个链被真的接上了"）。"""
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


def test_manual_standardize_matches_forward() -> None:
    """手写标准化是第二条独立实现（与 ① 的读数一致）。"""
    manual = verify._manual_standardize(
        verify.SAMPLE_BATCH, gamma=verify.SAMPLE_GAMMA, beta=verify.SAMPLE_BETA
    )
    produced, _cache, _running = normalization.batch_norm_forward(
        verify.SAMPLE_BATCH, gamma=verify.SAMPLE_GAMMA, beta=verify.SAMPLE_BETA
    )
    assert manual == produced


def test_check_batchnorm_equals_transposed_layernorm_calls_day079() -> None:
    """本课唯一一条跨天对账：BatchNorm(训练相) = transpose(LayerNorm(批ᵀ))。"""
    outcome = verify.check_batchnorm_equals_transposed_layernorm()
    assert outcome.passed is True
    assert outcome.check.reading <= verify.LAYERNORM_TOLERANCE


# --------------------------------------------------------------------------- 七张表


def test_axis_rows_cover_both_axes() -> None:
    """轴表两行，公式与签名都在。"""
    rows = study.axis_rows()
    assert [row.axis for row in rows] == list(types.NORM_AXES)
    assert "Σ_n" in rows[0].formula
    assert "Σ_j" in rows[1].formula
    assert all(row.line() for row in rows)


def test_phase_rows_come_from_different_statistics() -> None:
    """相表两行，且两个相的读数**不同**（μ 的来源不同）。"""
    rows = study.phase_rows()
    assert [row.phase for row in rows] == list(types.PHASES)
    assert rows[0].statistic != rows[1].statistic
    assert all(math.isfinite(row.max_abs) for row in rows)


def test_technique_and_parameter_rows() -> None:
    """技巧表四行；参数表三行且是 2H。"""
    techniques_rows = study.technique_rows()
    assert [row.technique for row in techniques_rows] == list(types.TECHNIQUES)
    parameter_rows = study.parameter_rows()
    assert [row.hidden for row in parameter_rows] == list(study.HIDDEN_CASES)
    assert all(row.batch_norm_parameters == 2 * row.hidden for row in parameter_rows)


def test_ablation_rows_two_cells_four_variants(ablation) -> None:
    """消融表：两种单元 × 四个变体，每行都带训练与推理两个损失。"""
    rows = study.ablation_rows(ablation)
    assert len(rows) == len(study.ABLATION_CELLS) * len(study.ABLATION_VARIANTS)
    assert all(row.train_loss >= 0.0 for row in rows)
    assert all(math.isfinite(row.gap) for row in rows)
    assert all("训练" in row.line() and "推理" in row.line() for row in rows)


def test_log_lines_contain_four_views(ablation) -> None:
    """日志表：sparkline / 折线 / 条形图 / 汇总裁剪四张脸都在。"""
    lines = study.log_lines(ablation)
    text = "\n".join(lines)
    assert "sparkline" in text
    assert "ASCII 折线" in text
    assert "条形图" in text
    assert "汇总裁剪" in text
    assert any(character in visualize.GLYPHS for character in text)


def test_property_rows_are_seven_and_pass() -> None:
    """性质表七行，全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert all("vs" in row.cross_check for row in rows)


def test_study_lines_has_seven_sections(ablation) -> None:
    """一次跑完七张表：七个小节标题都在。"""
    lines = study.study_lines(ablation)
    text = "\n".join(lines)
    for index in range(1, 8):
        assert f"== {index}." in text


def test_property_names_alias_and_phase_notes() -> None:
    """``PROPERTY_NAMES`` 就是性质名单；两个相的解释可读。"""
    assert study.PROPERTY_NAMES == types.REGULARIZATION_PROPERTIES
    assert set(study.PHASE_NOTES) == set(types.PHASES)
    assert study.FRESH_RUNNING.mean == (0.0,) * 3


def test_edge_guard_branches_are_covered() -> None:
    """四个"边界上的第一种失败"各走一次（非可迭代 / 非数 / 非整数 / 早停开着）."""
    with pytest.raises(errors.ShapeError):
        normalization.as_vector(5, name="v")  # 不可迭代
    with pytest.raises(errors.ParameterError):
        techniques.dropout_vector((1.0,), rate="x", seed=1)  # type: ignore[arg-type]
    with pytest.raises(errors.ParameterError):
        visualize.sparkline((1.0, 2.0), width=2.5)  # type: ignore[arg-type]
    watcher = techniques.RegularizationConfig(patience=3).enabled()
    assert "early_stop" in watcher


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_non_empty_and_has_no_submodules() -> None:
    """包入口的 ``__all__`` 非空、且不含子模块名。"""
    import smart_research_agent.regularization as package

    assert len(package.__all__) > 100
    assert not (
        {
            "errors",
            "types",
            "normalization",
            "techniques",
            "visualize",
            "network",
            "train",
            "verify",
            "study",
        }
        & set(package.__all__)
    )


def test_package_star_import_resolves_every_name() -> None:
    """每个 ``__all__`` 里的名字都能在包命名空间里取到。"""
    import smart_research_agent.regularization as package

    assert [name for name in package.__all__ if not hasattr(package, name)] == []


def test_package_exposes_key_objects() -> None:
    """几个关键对象从包入口就能取到。"""
    import smart_research_agent.regularization as package

    assert package.batch_norm_forward is normalization.batch_norm_forward
    assert package.dropout_vector is techniques.dropout_vector
    assert package.sparkline is visualize.sparkline
    assert package.train_regularized is train.train_regularized
    assert package.Check is verify.Check
