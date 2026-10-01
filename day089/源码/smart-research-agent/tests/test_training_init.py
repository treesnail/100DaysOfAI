"""``training_optim.init`` 与 ``controls``：初始化与三个控制器（day081）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.optim import global_norm
from smart_research_agent.training_optim import (
    INIT_SCHEMES,
    SCHEME_DESCRIPTIONS,
    SCHEME_KAIMING,
    SCHEME_NORMAL,
    SCHEME_UNIFORM,
    SCHEME_XAVIER,
    STD_RATIO_HIGH,
    STD_RATIO_LOW,
    EarlyStopping,
    ParameterError,
    TrainingConfig,
    build_optimizer,
    direction_cosine,
    expected_std,
    init_matrix,
    initialization_is_consistent,
    initialization_rows,
    initialize_block,
    initialize_parameters,
    learning_rate_at,
    matrix_std_ratio,
    measure_std,
)
from smart_research_agent.training_optim.controls import clip_gradients
from smart_research_agent.training_optim.errors import NumericError
from smart_research_agent.training_optim.types import ClipReport, EarlyStopReport
from tests.training_samples import (
    HIDDEN,
    LAYERS,
    day080_params,
    matrix_shape_of,
    params,
    shape,
)


class TestExpectedStd:
    """三条理论公式（每一项都能手算）."""

    def test_uniform_formula(self):
        """``U(−s, s)`` 的标准差是 ``s/√3``（s=0.25 → 0.144338）."""
        assert expected_std(SCHEME_UNIFORM, fan_in=6, fan_out=24) == pytest.approx(
            0.25 / math.sqrt(3.0)
        )

    def test_xavier_formula(self):
        """``√(2/(fan_in+fan_out))``：fan 6/24 → ``√(2/30) = 0.258199``."""
        assert expected_std(SCHEME_XAVIER, fan_in=6, fan_out=24) == pytest.approx(
            math.sqrt(2.0 / 30.0)
        )

    def test_kaiming_formula(self):
        """``√(2/fan_in)``：fan_in=6 → ``√(1/3) = 0.577350``（三种里最大）."""
        assert expected_std(SCHEME_KAIMING, fan_in=6, fan_out=24) == pytest.approx(
            math.sqrt(2.0 / 6.0)
        )
        assert expected_std(SCHEME_KAIMING, fan_in=6, fan_out=24) > expected_std(
            SCHEME_XAVIER, fan_in=6, fan_out=24
        )

    def test_normal_formula(self):
        """``1/√fan_in``：fan_in=6 → ``0.408248``."""
        assert expected_std(SCHEME_NORMAL, fan_in=6, fan_out=24) == pytest.approx(
            1.0 / math.sqrt(6.0)
        )

    def test_kaiming_is_root_two_times_xavier_when_fans_are_equal(self):
        """``fan_in == fan_out`` 时 kaiming 恰好是 xavier 的 ``√2`` 倍.

        ```text
        xavier   √(2/(fan+fan)) = √(2/16) = 0.353553
        kaiming  √(2/fan)       = √(2/8)  = 0.500000
        比值     0.5 / 0.353553 = √2
        ```

        这个 ``√2`` 就是"ReLU 关掉一半单元"那件事在公式里的样子：
        kaiming 把那一半补回来，因此幅度大 ``√2`` 倍。
        第一版把它写成"两者相等"，而它在 ``fan_in == fan_out`` 时也**不**相等。
        """
        xavier = expected_std(SCHEME_XAVIER, fan_in=8, fan_out=8)
        kaiming = expected_std(SCHEME_KAIMING, fan_in=8, fan_out=8)
        assert kaiming / xavier == pytest.approx(math.sqrt(2.0))

    def test_unknown_scheme_is_rejected(self):
        """不认识的方案当场报错（**不给它挑一个默认值**）。"""
        with pytest.raises(ParameterError):
            expected_std("he_normal", fan_in=6, fan_out=24)

    @pytest.mark.parametrize("fan_in,fan_out", [(0, 6), (6, 0), (1.5, 6)])
    def test_fans_must_be_positive_integers(self, fan_in: object, fan_out: object):
        """``fan_in`` / ``fan_out`` 必须是正整数。"""
        with pytest.raises(ParameterError):
            expected_std(SCHEME_XAVIER, fan_in=fan_in, fan_out=fan_out)  # type: ignore[arg-type]

    def test_uniform_scheme_checks_the_scale(self):
        """``uniform`` 方案要额外检查 ``scale``（它只在那里用得到）。"""
        with pytest.raises(ParameterError):
            expected_std(SCHEME_UNIFORM, fan_in=6, fan_out=24, scale=1.5)


class TestInitMatrix:
    """四个方案造出来的矩阵：形状、确定性、实测标准差."""

    def test_shapes_are_exact(self):
        """每一个方案的形状都正好是 ``(rows, columns)``."""
        for scheme in INIT_SCHEMES:
            matrix = init_matrix(24, 6, scheme=scheme, seed=3)
            assert matrix_shape_of(matrix) == (24, 6)

    def test_deterministic(self):
        """同一颗种子给出同一个矩阵（实验可复现的前提）。"""
        for scheme in INIT_SCHEMES:
            assert init_matrix(6, 6, scheme=scheme, seed=5) == init_matrix(
                6, 6, scheme=scheme, seed=5
            )

    def test_measured_std_is_close_to_the_theory(self):
        """**这一条的护栏**：``24 × 6 = 144`` 个元素上，实测与理论之比落在 ``[0.5, 2]``."""
        for scheme in INIT_SCHEMES:
            matrix = init_matrix(24, 6, scheme=scheme, seed=7)
            ratio = matrix_std_ratio(matrix, scheme=scheme)
            assert STD_RATIO_LOW <= ratio <= STD_RATIO_HIGH

    def test_uniform_entries_stay_inside_the_bound(self):
        """``uniform`` 的每个元素都落在 ``[−s, s]`` 内（Box–Muller 那一路才可能越界）。"""
        matrix = init_matrix(24, 6, scheme=SCHEME_UNIFORM, seed=2)
        assert all(abs(value) <= 0.25 + 1e-12 for row in matrix for value in row)

    def test_kaiming_entries_are_wider_than_xavier(self):
        """同形状下 kaiming 的实测标准差更大（它的半宽 ``√(6/fan_in)`` 更大）。"""
        kaiming = measure_std(init_matrix(24, 6, scheme=SCHEME_KAIMING, seed=4))
        xavier = measure_std(init_matrix(24, 6, scheme=SCHEME_XAVIER, seed=4))
        assert kaiming > xavier

    def test_normal_scheme_has_both_signs(self):
        """正态方案会给出正负两侧的数（它不是"只往一边偏"的分布）。"""
        matrix = init_matrix(24, 6, scheme=SCHEME_NORMAL, seed=6)
        values = [value for row in matrix for value in row]
        assert any(value > 0 for value in values) and any(value < 0 for value in values)

    def test_measure_std_of_a_constant_matrix_is_zero(self):
        """常值矩阵的标准差是 0（它没有"散布"）。"""
        assert measure_std(((2.0, 2.0), (2.0, 2.0))) == 0.0

    def test_measure_std_matches_a_two_point_hand_computation(self):
        """``(1, 3)`` 的有偏标准差是 1（均值 2、方差 1）。"""
        assert measure_std(((1.0, 3.0),)) == pytest.approx(1.0)


class TestInitializeParameters:
    """整条链的参数：与 day080 的口径对齐、LN 保持恒等初值."""

    def test_uniform_scheme_reproduces_day080(self):
        """**uniform 方案与 day080 的 ``make_stack_parameters`` 逐位一致**.

        这条等式让"换初始化"这件事有一个干净的零点：
        基线不是"另一套实现"，而是昨天那一摞参数。
        """
        assert initialize_parameters(shape(), scheme=SCHEME_UNIFORM, seed=7) == day080_params()

    def test_other_schemes_differ_from_the_baseline(self):
        """另外三种方案与基线不同（否则"换方案"这件事没有发生）。"""
        baseline = day080_params()
        for scheme in (SCHEME_XAVIER, SCHEME_KAIMING, SCHEME_NORMAL):
            assert initialize_parameters(shape(), scheme=scheme, seed=7) != baseline

    def test_layers_get_different_seeds(self):
        """层与层用不同的种子（否则逐层曲线会变成周期 1 的序列）。"""
        stack = initialize_parameters(shape(), scheme=SCHEME_XAVIER)
        assert stack.blocks[0] != stack.blocks[1]
        assert stack.blocks[0].ffn_w_in != stack.blocks[1].ffn_w_in

    def test_layernorm_starts_at_identity(self):
        """两个 LN 保持 ``γ = 1、β = 0``（day079 的口径，今天不改）。"""
        block = initialize_parameters(shape(), scheme=SCHEME_KAIMING).blocks[0]
        assert all(value == 1.0 for value in block.norm1_gamma)
        assert all(value == 0.0 for value in block.norm1_beta)
        assert all(value == 1.0 for value in block.norm2_gamma)
        assert all(value == 0.0 for value in block.norm2_beta)

    def test_initialize_block_shapes(self):
        """一个块的八块参数形状与 ``BlockShape`` 一致。"""
        block = initialize_block(shape().block_shape, scheme=SCHEME_XAVIER)
        assert matrix_shape_of(block.ffn_w_in) == (24, 6)
        assert matrix_shape_of(block.ffn_w_out) == (6, 24)
        assert len(block.ffn_b_in) == 24 and len(block.ffn_b_out) == 6

    def test_layer_count_matches_the_shape(self):
        """层数与形状一致（参数份数不能多也不能少）。"""
        stack = initialize_parameters(shape(layers=2), scheme=SCHEME_NORMAL)
        assert stack.layers == 2 == len(stack.blocks)


class TestInitializationRows:
    """逐矩阵的``(名字, 理论, 实测, 比值)``表."""

    def test_rows_cover_the_two_weight_matrices(self):
        """两个权重矩阵各一行（偏置是 0，因此没有"幅度"可测）。"""
        rows = initialization_rows(shape(), scheme=SCHEME_XAVIER, seed=7)
        assert [name for name, *_rest in rows] == ["ffn_w_in", "ffn_w_out"]

    def test_ratios_are_consistent(self):
        """``initialization_is_consistent`` 在这四组上都为真。"""
        for scheme in INIT_SCHEMES:
            rows = initialization_rows(shape(), scheme=scheme, seed=7)
            assert initialization_is_consistent(rows)

    def test_a_bad_scheme_is_caught_by_the_ratio(self):
        """**反证**：把理论值故意算成另一种方案的值，比值就会越界.

        这一条证明"比值"这个读数**真的能失败**——否则它只是一张好看的纸。
        """
        rows = initialization_rows(shape(), scheme=SCHEME_XAVIER, seed=7)
        _name, theory, measured, _ratio = rows[0]
        wrong_ratio = measured / (theory * 10.0)
        assert not initialization_is_consistent((("ffn_w_in", theory, measured, wrong_ratio),))

    def test_empty_rows_are_rejected(self):
        """空表没有可判的东西。"""
        with pytest.raises(ParameterError):
            initialization_is_consistent(())


class TestLearningRateSchedule:
    """调度：逐位对齐 day074，而且整数参数要还原成 ``int``."""

    def test_constant_schedule(self):
        """常数调度每一步都是 ``base_lr``."""
        config = TrainingConfig(learning_rate=0.1, steps=5)
        assert [learning_rate_at(config, step) for step in range(1, 6)] == [0.1] * 5

    def test_warmup_rises_then_falls(self):
        """热身段的斜率是 ``base_lr / warmup``：第 1 步是 ``0.02``、第 5 步到峰值。"""
        config = TrainingConfig(
            learning_rate=0.1,
            steps=40,
            schedule="warmup_cosine",
            schedule_params=(("warmup_steps", 5.0), ("total_steps", 40.0)),
        )
        values = [learning_rate_at(config, step) for step in range(1, 41)]
        assert values[0] == pytest.approx(0.02)
        assert values[4] == pytest.approx(0.1)
        assert max(values) == pytest.approx(0.1)
        assert values[-1] == pytest.approx(0.0)
        assert values[-1] < values[-2] < values[-3]

    def test_first_step_is_not_zero(self):
        """第 1 步**不是** 0——"第 1 步 lr = 0"会让第一次更新不发生（day074 的边界）。"""
        config = TrainingConfig(
            learning_rate=0.1,
            steps=40,
            schedule="warmup_cosine",
            schedule_params=(("warmup_steps", 5.0), ("total_steps", 40.0)),
        )
        assert learning_rate_at(config, 1) > 0.0

    def test_it_matches_the_backbone_directly(self):
        """与 day074 的 ``warmup_cosine_schedule`` **逐位**一致（口径只有一处）。"""
        from smart_research_agent.math_foundations.optim import warmup_cosine_schedule

        config = TrainingConfig(
            learning_rate=0.3,
            steps=9,
            schedule="warmup_cosine",
            schedule_params=(("warmup_steps", 3.0), ("total_steps", 9.0)),
        )
        for step in range(1, 10):
            assert learning_rate_at(config, step) == warmup_cosine_schedule(
                step, base_lr=0.3, warmup_steps=3, total_steps=9
            )

    def test_missing_parameters_give_a_readable_error(self):
        """调度缺参数时给出可读的错（而不是一个 TypeError 栈）。"""
        config = TrainingConfig(learning_rate=0.1, steps=5, schedule="warmup_cosine")
        with pytest.raises(ParameterError):
            learning_rate_at(config, 1)

    @pytest.mark.parametrize("step", [0, -1, 1.5])
    def test_step_must_be_one_based(self, step: object):
        """步号是 1-based（day074 的纪律）。"""
        with pytest.raises(ParameterError):
            learning_rate_at(TrainingConfig(), step)  # type: ignore[arg-type]

    def test_a_non_config_is_rejected(self):
        """传进来不是 ``TrainingConfig`` 时当场报错。"""
        with pytest.raises(ParameterError):
            learning_rate_at("config", 1)  # type: ignore[arg-type]

    def test_build_optimizer_reads_the_config(self):
        """优化器从配置里造（名字与额外参数都进）。"""
        optimizer = build_optimizer(TrainingConfig(learning_rate=0.2, optimizer="momentum"))
        assert optimizer.learning_rate == 0.2
        assert optimizer.name == "momentum"


class TestClipping:
    """裁剪：只改长度、不改方向，并把读数留下来."""

    def test_no_threshold_gives_scale_one(self):
        """没设阈值时缩放是 1.0，但**读数仍然给出**（"没裁剪"必须可读）。"""
        grads = (3.0, 4.0)
        clipped, report = clip_gradients(grads, TrainingConfig(max_norm=None))
        assert clipped == grads
        assert report.scale == 1.0
        assert report.applied is False
        assert report.max_norm is None
        assert report.clipped_norm == pytest.approx(5.0)

    def test_threshold_above_the_norm_does_not_change_anything(self):
        """阈值比范数大时**裁剪了但没生效**——它与"没裁剪"在两个读数上都可区分。"""
        grads = (3.0, 4.0)
        clipped, report = clip_gradients(grads, TrainingConfig(max_norm=10.0))
        assert clipped == grads
        assert report.applied is False
        assert report.max_norm == 10.0

    def test_threshold_below_the_norm_scales_the_step(self):
        """阈值比范数小时整体缩放：``‖g‖`` 落到阈值上，方向余弦仍是 1。"""
        grads = (3.0, 4.0)
        clipped, report = clip_gradients(grads, TrainingConfig(max_norm=1.0))
        assert report.applied is True
        assert report.clipped_norm == pytest.approx(1.0)
        assert global_norm(clipped) == pytest.approx(1.0)
        assert direction_cosine(grads, clipped) == pytest.approx(1.0)

    def test_direction_cosine_is_one_for_a_zero_vector(self):
        """零向量的方向没有定义——本包返回 1.0（与 day080 的 ``gain`` 同一个约定）。"""
        assert direction_cosine((0.0, 0.0), (0.0, 0.0)) == 1.0

    def test_direction_cosine_detects_a_length_change(self):
        """换了长度但方向不变时余弦仍是 1（它测的是方向，不是长度）。"""
        assert direction_cosine((1.0, 1.0), (2.0, 2.0)) == pytest.approx(1.0)
        assert direction_cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_direction_cosine_checks_lengths(self):
        """两个向量长度不同时当场报错。"""
        with pytest.raises(ValueError):
            direction_cosine((1.0, 2.0), (1.0,))

    @pytest.mark.parametrize("max_norm", [0.0, -1.0, "0.5"])
    def test_invalid_thresholds_are_rejected_by_the_report(self, max_norm: object):
        """阈值必须是正的有限数（``ClipReport`` 会挡住它）。"""
        with pytest.raises(ParameterError):
            ClipReport(original_norm=1.0, max_norm=max_norm, scale=1.0)  # type: ignore[arg-type]

    def test_report_rejects_bad_numbers(self):
        """范数为 inf、缩放为 0 都当场报错（发散在读数这一层就要被看见）。"""
        with pytest.raises(NumericError):
            ClipReport(original_norm=float("inf"), max_norm=None, scale=1.0)
        with pytest.raises(NumericError):
            ClipReport(original_norm=1.0, max_norm=None, scale=0.0)

    def test_report_is_serialisable(self):
        """裁剪读数能 json.dumps。"""
        import json

        _clipped, report = clip_gradients((3.0, 4.0), TrainingConfig(max_norm=1.0))
        payload = report.to_dict()
        assert payload["applied"] is True
        assert isinstance(json.dumps(payload), str)
        assert "已裁剪" in report.summary_line()


class TestEarlyStopping:
    """早停：一条判据 + 一份报告."""

    def test_it_fires_after_patience_steps(self):
        """连续 ``patience`` 步没变好就触发（本样本 6 步）。"""
        stopper = EarlyStopping(patience=3, min_delta=0.001)
        curve = (1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7)
        decisions = [stopper.update(step, loss) for step, loss in enumerate(curve, start=1)]
        assert decisions == [False, False, False, False, False, True, True]
        report = stopper.report()
        assert report.triggered is True
        assert report.best_step == 3
        assert report.waited >= 3

    def test_it_never_fires_on_a_monotone_curve(self):
        """单调下降的曲线上永不触发（性质 6）。"""
        stopper = EarlyStopping(patience=2, min_delta=0.0)
        losses = [1.0 - 0.05 * step for step in range(1, 21)]
        decisions = [stopper.update(step, loss) for step, loss in enumerate(losses, start=1)]
        assert not any(decisions)
        assert stopper.report().triggered is False

    def test_min_delta_filters_out_float_noise(self):
        """``min_delta`` 把"浮点噪声级的变好"挡在外面（否则早停永不触发）。"""
        noisy = EarlyStopping(patience=2, min_delta=0.01)
        decisions = [noisy.update(step, loss) for step, loss in enumerate((1.0, 0.999, 0.998, 0.997), start=1)]
        assert decisions[-1] is True
        assert noisy.best_step == 1

    def test_report_before_any_update_is_rejected(self):
        """还没喂过任何一步就报告 ⇒ 报错（``best_step`` 会是一个凭空的数）。"""
        with pytest.raises(ParameterError):
            EarlyStopping(patience=2).report()

    def test_describe_before_and_after(self):
        """``describe`` 在两个时刻都有话可说。"""
        stopper = EarlyStopping(patience=2)
        assert "还没有喂过" in stopper.describe()
        stopper.update(1, 1.0)
        assert "最好第 1 步" in stopper.describe()

    def test_state_is_serialisable(self):
        """状态能 json.dumps（只含跨步保留的东西）。"""
        import json

        stopper = EarlyStopping(patience=2)
        stopper.update(1, 0.5)
        assert isinstance(json.dumps(stopper.state()), str)

    @pytest.mark.parametrize("patience", [0, -1, 1.5])
    def test_patience_must_be_positive(self, patience: object):
        """耐心必须是正整数。"""
        with pytest.raises(ParameterError):
            EarlyStopping(patience)  # type: ignore[arg-type]

    def test_bad_steps_and_losses_are_rejected(self):
        """步号与损失都要合法（损失不能是 nan）。"""
        stopper = EarlyStopping(patience=2)
        with pytest.raises(ParameterError):
            stopper.update(0, 1.0)
        with pytest.raises(NumericError):
            stopper.update(1, float("nan"))

    def test_report_validates_its_fields(self):
        """报告自己也要校验（``waited`` 不能是负数）。"""
        with pytest.raises(ParameterError):
            EarlyStopReport(patience=2, best_step=1, best_loss=1.0, waited=-1, triggered=False)

    def test_report_lines(self):
        """两种输出都带着"为什么停在这里"。"""
        report = EarlyStopReport(
            patience=3, best_step=2, best_loss=0.5, waited=3, triggered=True, min_delta=0.001
        )
        assert "已触发" in report.summary_line()
        assert "连续 3 步" in report.reason
        assert report.to_dict()["best_step"] == 2


class TestConstants:
    """两张表的闭合与默认值。"""

    def test_every_scheme_has_a_description(self):
        """四种方案各有一条说明。"""
        assert set(SCHEME_DESCRIPTIONS) == set(INIT_SCHEMES)
        assert all(SCHEME_DESCRIPTIONS[scheme] for scheme in INIT_SCHEMES)

    def test_ratio_bounds_are_sane(self):
        """比值区间是 ``[0.5, 2]``（一个宽到不会被浮点噪声翻面、又窄到能抓住写错的区间）。"""
        assert STD_RATIO_LOW == 0.5
        assert STD_RATIO_HIGH == 2.0

    def test_uniform_matrix_uses_the_sample_scale(self):
        """``uniform`` 的实测标准差贴近 ``0.25/√3 = 0.1443``."""
        matrix = init_matrix(24, 6, scheme=SCHEME_UNIFORM, seed=7)
        assert measure_std(matrix) == pytest.approx(0.25 / math.sqrt(3.0), rel=0.2)

    def test_parameters_helper_covers_all_layers(self):
        """样本参数的层数与形状一致（``params`` 是测试共用的那一处）。"""
        assert params().layers == LAYERS
        assert params().blocks[0].hidden == HIDDEN
