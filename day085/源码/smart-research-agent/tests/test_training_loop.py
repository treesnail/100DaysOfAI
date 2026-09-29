"""``training_optim.train`` 与记录：训练循环与每一步的读数（day081）."""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.encoder_decoder.types import BlockGradients
from smart_research_agent.math_foundations.types import matrix_shape
from smart_research_agent.transformer_stack import (
    StackGradients,
    StackParameters,
    StackedLayer,
    stack_backward,
    stack_forward,
)
from smart_research_agent.training_optim import (
    CONTROLS,
    CONTROL_DESCRIPTIONS,
    DEFAULT_DIVERGENCE_FACTOR,
    LAYER_MASK_STRIDE,
    STEP_MASK_STRIDE,
    DivergenceError,
    EpochRecord,
    ParameterError,
    ShapeError,
    StepCache,
    TrainingConfig,
    TrainingCurve,
    check_no_divergence,
    curve_summary,
    describe_controls,
    flatten_block_gradients,
    forward_output_at,
    initialize_parameters,
    mask_seed,
    plain_stack_output,
    record_count,
    shape_size,
    step_loss,
    steps_of,
    train,
    training_backward,
    training_forward,
)
from smart_research_agent.transformer_core.layers import mse_gradient
from smart_research_agent.training_optim.errors import NumericError
from tests.training_samples import (
    HIDDEN,
    LAYERS,
    LEARNING_RATE,
    PLACEMENTS,
    SEED,
    STEPS,
    TOKENS,
    config,
    inputs,
    params,
    shape,
    target,
)


class TestTrainingConfig:
    """一份配置：冻结、可比、五个旋钮."""

    def test_defaults_are_the_baseline(self):
        """默认值就是基线（无 dropout、无裁剪、无早停、SGD）。"""
        item = TrainingConfig()
        assert item.dropout == 0.0
        assert item.clipping_on is False
        assert item.early_stop_on is False
        assert item.optimizer == "sgd"
        assert item.schedule == "constant"
        assert item.divergence_factor == DEFAULT_DIVERGENCE_FACTOR

    def test_it_is_hashable(self):
        """配置可 hash（可以被放进集合、也可以直接比对）。"""
        assert len({TrainingConfig(), TrainingConfig()}) == 1

    def test_replaced_changes_one_knob(self):
        """``replaced`` 一次只改一个旋钮（实验的纪律靠它）。"""
        base = TrainingConfig(learning_rate=0.1, steps=5)
        bumped = base.replaced(dropout=0.2)
        assert bumped.dropout == 0.2
        assert bumped.learning_rate == base.learning_rate
        assert bumped.steps == base.steps

    def test_replaced_rejects_unknown_fields(self):
        """不认识的配置项当场报错（而不是被静默忽略）。"""
        with pytest.raises(ParameterError):
            TrainingConfig().replaced(learning_rat=0.1)

    def test_schedule_kwargs_restore_integers(self):
        """``warmup_steps`` 这类参数在出门时还原成 ``int``（day074 要求整数）。"""
        item = TrainingConfig(
            schedule="warmup_cosine",
            schedule_params=(("warmup_steps", 5.0), ("total_steps", 40.0)),
        )
        assert item.schedule_kwargs == {"warmup_steps": 5, "total_steps": 40}
        assert isinstance(item.schedule_kwargs["warmup_steps"], int)

    def test_fractional_parameters_stay_floats(self):
        """非整数的参数保持浮点（``gamma=0.1`` 这类）。"""
        item = TrainingConfig(optimizer_params=(("momentum", 0.9),))
        assert item.optimizer_kwargs == {"momentum": 0.9}

    @pytest.mark.parametrize("field,value", [
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("steps", 0),
        ("patience", 0),
        ("min_delta", -0.1),
        ("divergence_factor", 0.0),
        ("init_scale", 1.5),
    ])
    def test_bad_fields_are_rejected(self, field: str, value: object):
        """越界的字段当场报错。"""
        with pytest.raises(ParameterError):
            TrainingConfig(**{field: value})  # type: ignore[arg-type]

    def test_unknown_schedule_and_optimizer_are_rejected(self):
        """不认识的名字当场报错（可选值写进消息里）。"""
        with pytest.raises(ParameterError):
            TrainingConfig(schedule="one_cycle")
        with pytest.raises(ParameterError):
            TrainingConfig(optimizer="lamb")

    def test_summary_line_and_dict(self):
        """两种输出都带着"这一次是怎么训的"。"""
        item = config(max_norm=0.5, patience=3)
        assert "裁剪 0.5" in item.summary_line()
        assert "早停 3" in item.summary_line()
        assert isinstance(json.dumps(item.to_dict()), str)


class TestMasks:
    """掩码种子：由 ``(seed, step, layer)`` 唯一决定."""

    def test_seed_depends_on_all_three_numbers(self):
        """三个数各改一个，种子都不同。"""
        base = mask_seed(config(), 1, 0)
        assert mask_seed(config(), 2, 0) != base
        assert mask_seed(config(), 1, 1) != base
        assert mask_seed(config(seed=SEED + 1), 1, 0) != base

    def test_the_formula_is_the_two_strides(self):
        """公式就是 ``seed + step × 1009 + layer × 31``（两个步长都是常量）。"""
        assert mask_seed(config(), 3, 2) == SEED + 3 * STEP_MASK_STRIDE + 2 * LAYER_MASK_STRIDE
        assert (STEP_MASK_STRIDE, LAYER_MASK_STRIDE) == (1009, 31)

    @pytest.mark.parametrize("step,layer", [(0, 0), (1, -1), (1.5, 0)])
    def test_bad_indices_are_rejected(self, step: object, layer: object):
        """步号与层号都要合法。"""
        with pytest.raises(ParameterError):
            mask_seed(config(), step, layer)  # type: ignore[arg-type]


class TestTrainingForward:
    """带 dropout 的链式前向."""

    def test_rate_zero_matches_the_plain_stack(self):
        """**护栏一**：``dropout = 0`` 时与 day080 的 ``stack_forward`` **逐位**一致.

        没有这条等式，这一课训练的就不是那个模型，两条损失曲线也就无从比较。
        """
        stack = params()
        ours = training_forward(stack, inputs(), config=config()).output
        plain = stack_forward(stack, inputs()).output
        assert ours == plain

    @pytest.mark.parametrize("placement", PLACEMENTS)
    def test_rate_zero_matches_the_plain_stack_in_both_placements(self, placement: str):
        """两种摆放位置都要逐位一致。"""
        ours = training_forward(stack := params(), inputs(), config=config(), placement=placement)
        plain = stack_forward(stack, inputs(), placement=placement).output
        assert ours.output == plain

    def test_eval_phase_matches_the_plain_stack_even_with_dropout(self):
        """**推理相**在任何概率下都与 day080 的链逐位一致（inverted dropout 的好处）。"""
        stack = params()
        ours = forward_output_at(stack, inputs(), config=config(dropout=0.5), phase="eval")
        assert ours == plain_stack_output(stack, inputs())

    def test_train_phase_changes_the_output(self):
        """训练相**会**改变输出（否则 dropout 没有生效）。

        注意 ``forward_output_at`` 的默认相是**推理相**（它要在任何概率下都与
        day080 的链逐位一致）——因此这一条必须显式地说 ``phase="train"``。
        """
        stack = params()
        trained = forward_output_at(
            stack, inputs(), config=config(dropout=0.3), phase="train"
        )
        assert trained != plain_stack_output(stack, inputs())

    def test_shape_is_preserved(self):
        """保形：``(4, 6)`` 进、``(4, 6)`` 出。"""
        cache = training_forward(params(), inputs(), config=config(dropout=0.3))
        assert matrix_shape(cache.output) == (TOKENS, HIDDEN)

    def test_cache_records_one_mask_per_layer(self):
        """每一层各有一份掩码与缩放，``kept`` 在 ``[0, 1]``。"""
        cache = training_forward(params(), inputs(), config=config(dropout=0.3))
        assert cache.depth == LAYERS == len(cache.masks) == len(cache.scales)
        assert all(scale == pytest.approx(1.0 / 0.7) for scale in cache.scales)
        assert 0.0 <= cache.kept <= 1.0

    def test_rate_zero_keeps_everything(self):
        """``rate = 0`` 时保留比例是 1.0（掩码全 1）。"""
        cache = training_forward(params(), inputs(), config=config())
        assert cache.kept == 1.0

    def test_steps_use_different_masks(self):
        """**每步换一个掩码**（否则那不是 dropout，而是"固定丢掉同一批单元"）。"""
        first = training_forward(params(), inputs(), config=config(dropout=0.3), step=1).masks
        second = training_forward(params(), inputs(), config=config(dropout=0.3), step=2).masks
        assert first != second

    def test_cache_validates_itself(self):
        """三个序列必须逐层对齐、``kept`` 必须落在 ``[0, 1]``。"""
        layer = StackedLayer(
            index=0,
            block=stack_forward(params(), inputs()).layer_at(0).block,
            attention=stack_forward(params(), inputs()).layer_at(0).attention,
            params=params().blocks[0],
        )
        with pytest.raises(ShapeError):
            StepCache(output=inputs(), layers=(layer,), masks=(), scales=(), kept=1.0)
        with pytest.raises(ParameterError):
            StepCache(output=inputs(), layers=(), masks=(), scales=(), kept=2.0)

    def test_summary_line(self):
        """一行说明。"""
        cache = training_forward(params(), inputs(), config=config())
        assert "4 层" in cache.summary_line()


class TestTrainingBackward:
    """链式反向 + 每一层穿过自己的掩码."""

    def test_rate_zero_matches_the_plain_backward(self):
        """**护栏二**：``dropout = 0`` 时与 day080 的 ``stack_backward`` 逐位一致。"""
        stack = params()
        forward = stack_forward(stack, inputs())
        cache = training_forward(stack, inputs(), config=config())
        goal = target()
        ours = training_backward(cache, stack, mse_gradient(forward.output, goal))
        plain = stack_backward(forward, stack, mse_gradient(forward.output, goal))
        assert ours.grad_inputs == plain.grad_inputs
        for index in range(LAYERS):
            assert ours.layer_at(index).flatten() == plain.layer_at(index).flatten()

    def test_it_records_the_same_depth(self):
        """账的层数与参数份数一致。"""
        stack = params()
        cache = training_forward(stack, inputs(), config=config(dropout=0.2))
        grads = training_backward(cache, stack, mse_gradient(cache.output, target()))
        assert grads.depth == LAYERS
        assert grads.grad_inputs == grads.layer_at(0).grad_inputs

    def test_a_mismatched_bundle_is_rejected(self):
        """账与参数份数不一致 ⇒ ``ShapeError``（反向必须与产生这份账的那次前向配套）。"""
        stack = params()
        cache = training_forward(stack, inputs(), config=config())
        shallow = initialize_parameters(shape(layers=2), scheme="uniform", seed=SEED)
        with pytest.raises(ShapeError):
            training_backward(cache, shallow, mse_gradient(cache.output, target()))

    def test_the_mask_changes_the_gradient(self):
        """开着 dropout 时梯度**与关掉时不同**（掩码真的穿过了反向）。"""
        stack = params()
        goal = target()
        with_mask = training_forward(stack, inputs(), config=config(dropout=0.4), step=3)
        without = training_forward(stack, inputs(), config=config())
        first = training_backward(with_mask, stack, mse_gradient(with_mask.output, goal))
        second = training_backward(without, stack, mse_gradient(without.output, goal))
        assert first.grad_inputs != second.grad_inputs


class TestFlattenGradients:
    """参数梯度的压平：与 ``params.flatten()`` 同序、长度一致."""

    def test_length_matches_the_block_parameter_count(self):
        """长度是 ``N × 342 = 1368``（只含块参数，不含注意力）。"""
        stack = params()
        cache = training_forward(stack, inputs(), config=config())
        grads = training_backward(cache, stack, mse_gradient(cache.output, target()))
        flat = flatten_block_gradients(grads, stack)
        assert len(flat) == stack.block_parameter_count == 4 * 342
        assert len(stack.flatten()[0]) == len(flat)

    def test_it_takes_the_first_block_parameter_count_numbers(self):
        """它取的是每一环 ``flatten()`` 的**前 342 个**（八块参数，输入梯度排在最后）。"""
        stack = params()
        cache = training_forward(stack, inputs(), config=config())
        grads = training_backward(cache, stack, mse_gradient(cache.output, target()))
        flat = flatten_block_gradients(grads, stack)
        per_layer = stack.block_parameter_count // LAYERS
        expected = tuple(
            grads.layer_at(0).flatten()[:per_layer] + grads.layer_at(1).flatten()[:per_layer]
            + grads.layer_at(2).flatten()[:per_layer] + grads.layer_at(3).flatten()[:per_layer]
        )
        assert flat == expected

    def test_it_rejects_foreign_inputs(self):
        """类型不符与份数不符都要当场报错。"""
        stack = params()
        with pytest.raises(ParameterError):
            flatten_block_gradients("grads", stack)  # type: ignore[arg-type]
        cache = training_forward(stack, inputs(), config=config())
        grads = training_backward(cache, stack, mse_gradient(cache.output, target()))
        shallow = initialize_parameters(shape(layers=2), scheme="uniform", seed=SEED)
        with pytest.raises(ShapeError):
            flatten_block_gradients(grads, shallow)

    def test_the_gradient_matches_a_hand_written_block(self):
        """第一条链上的一环梯度与独立调用的 ``encoder_block_backward`` 逐位一致。"""
        stack = params()
        cache = training_forward(stack, inputs(), config=config())
        grads = training_backward(cache, stack, mse_gradient(cache.output, target()))
        assert isinstance(grads.layer_at(0), BlockGradients)
        assert isinstance(grads, StackGradients)


class TestStepLoss:
    """一步的损失：与 day079/080 用同一个函数."""

    def test_it_matches_a_manual_mse(self):
        """损失就是 ``mean_squared_error(输出, 目标)``."""
        from smart_research_agent.transformer_core.layers import mean_squared_error

        stack = params()
        loss, cache = step_loss(stack, inputs(), target(), config=config(), step=1, phase="train")
        assert loss == pytest.approx(mean_squared_error(cache.output, target()))

    def test_placement_is_threaded_through(self):
        """**一处真实踩过的坑**：``placement`` 必须一路传下来.

        第一版漏了它，于是"post-LN"那几行实际跑的是 pre-LN——
        而两条本该不同的曲线**重合成一条**，看起来完全正常。
        """
        stack = params()
        pre = step_loss(stack, inputs(), target(), config=config(), step=1, phase="train", placement="pre")[0]
        post = step_loss(stack, inputs(), target(), config=config(), step=1, phase="train", placement="post")[0]
        assert pre != post


class TestTrain:
    """训练循环：损失真的会下降、每一步都有读数."""

    def test_it_improves_the_loss(self):
        """样本上 40 步之后损失明显下降（改善 > 2 倍）。"""
        curve = train(shape(), inputs(), target(), config=config())
        assert curve.decreased
        assert curve.improvement_ratio > 2.0
        assert curve.last_loss < curve.first_loss
        assert not curve.diverged

    def test_it_records_every_step(self):
        """每一步一行读数（曲线的全部信息在这里）。"""
        curve = train(shape(), inputs(), target(), config=config(steps=7))
        assert record_count(curve) == 7
        assert steps_of(curve) == tuple(range(1, 8))

    def test_each_record_carries_five_readings(self):
        """每一步都有损失、推理损失、学习率、梯度范数与缩放。"""
        record = train(shape(), inputs(), target(), config=config(steps=3)).records[0]
        assert isinstance(record, EpochRecord)
        assert record.step == 1
        assert record.learning_rate == pytest.approx(LEARNING_RATE)
        assert record.grad_norm > 0.0
        assert record.clip_scale == 1.0
        assert record.kept_fraction == 1.0

    def test_it_does_not_mutate_the_incoming_parameters(self):
        """传进来的那一摞参数**不被改动**（元组是不可变的，这正是它可复用的原因）。"""
        stack = params()
        before = stack.blocks[0].ffn_w_in
        train(shape(), inputs(), target(), config=config(steps=3), params=stack)
        assert stack.blocks[0].ffn_w_in == before

    def test_it_is_deterministic(self):
        """**性质 1**：同一份配置跑两次，每一步的读数逐位相同。"""
        first = train(shape(), inputs(), target(), config=config(steps=5))
        second = train(shape(), inputs(), target(), config=config(steps=5))
        assert [item.to_dict() for item in first.records] == [
            item.to_dict() for item in second.records
        ]

    def test_both_placements_run(self):
        """两种摆放位置都能跑完，而它们的曲线不同（``placement`` 真的生效了）。"""
        pre = train(shape(), inputs(), target(), config=config(steps=5), placement="pre")
        post = train(shape(), inputs(), target(), config=config(steps=5), placement="post")
        assert pre.last_loss != post.last_loss

    def test_early_stopping_can_cut_it_short(self):
        """开了早停时步数可能少于计划步数（并留下报告）。"""
        curve = train(shape(), inputs(), target(), config=config(steps=40, patience=2))
        assert curve.early_stop is not None
        assert len(curve.records) <= STEPS

    def test_divergence_is_raised_not_swallowed(self):
        """学习率太大时抛 ``DivergenceError``（**不许把它当结果交出去**）。"""
        with pytest.raises(DivergenceError):
            train(shape(), inputs(), target(), config=config(learning_rate=1.5))

    def test_clipping_rescues_a_big_learning_rate(self):
        """同一个学习率下，裁剪把它从"发散"拉回"跑完"。"""
        with pytest.raises(DivergenceError):
            train(shape(), inputs(), target(), config=config(learning_rate=1.5))
        rescued = train(shape(), inputs(), target(), config=config(learning_rate=1.5, max_norm=0.5))
        assert rescued.decreased

    def test_a_bad_config_type_is_rejected(self):
        """``config`` 必须是 ``TrainingConfig``."""
        with pytest.raises(ParameterError):
            train(shape(), inputs(), target(), config="config")  # type: ignore[arg-type]

    def test_dropout_raises_the_training_loss_but_not_the_eval_loss(self):
        """**性质的口径**：开了 dropout 之后训练损失更高，而推理损失更低。

        这就是"只看训练损失会让 dropout 看起来有害"那件事的量化形式。
        """
        plain = train(shape(), inputs(), target(), config=config(dropout=0.0))
        dropped = train(shape(), inputs(), target(), config=config(dropout=0.2))
        assert dropped.first_loss > plain.first_loss
        assert dropped.mean_gap() < 0.0


class TestCurve:
    """曲线记录：三个数、两条序列与几条性质."""

    def test_curve_rejects_non_increasing_steps(self):
        """步号必须递增且不重复（重复会污染 ``best_step``）。"""
        good = train(shape(), inputs(), target(), config=config(steps=3)).records
        with pytest.raises(ShapeError):
            TrainingCurve(config=config(), shape=shape(), records=tuple(reversed(good)))
        with pytest.raises(NumericError):
            TrainingCurve(config=config(), shape=shape(), records=(good[0], good[0]))

    def test_curve_needs_at_least_one_record(self):
        """空曲线没有意义。"""
        with pytest.raises(ParameterError):
            TrainingCurve(config=config(), shape=shape(), records=())

    def test_curve_properties(self):
        """``first / best / last / improvement / decreased / eval_decreased``."""
        curve = train(shape(), inputs(), target(), config=config(steps=10))
        assert curve.first_loss == curve.records[0].loss
        assert curve.last_loss == curve.records[-1].loss
        assert curve.best_loss == min(item.loss for item in curve.records)
        assert curve.best_step == curve.best.step
        assert curve.best_is_before_the_end in (True, False)
        assert curve.improvement_ratio > 1.0
        assert curve.eval_decreased

    def test_curve_is_serialisable(self):
        """曲线能 json.dumps（配置、每一步、早停报告都在里面）。"""
        payload = train(shape(), inputs(), target(), config=config(steps=2)).to_dict()
        assert payload["config"]["learning_rate"] == LEARNING_RATE
        assert len(payload["records"]) == 2
        assert isinstance(json.dumps(payload), str)

    def test_table_lines(self):
        """一张表：表头 + 分隔线 + 每一步一行（``every`` 控制抽稀）。"""
        curve = train(shape(), inputs(), target(), config=config(steps=4))
        assert len(curve.table_lines()) == 4 + 2
        assert len(curve.table_lines(every=2)) == 2 + 2
        assert "训练损失" in curve.table_lines()[0]

    def test_summary_and_gap(self):
        """摘要与"训练/推理之差"两个读数。"""
        curve = train(shape(), inputs(), target(), config=config(steps=4))
        assert "损失" in curve.summary_line()
        assert math.isfinite(curve.mean_gap())
        assert curve.records[0].gap == pytest.approx(
            curve.records[0].eval_loss - curve.records[0].loss
        )

    def test_record_validation(self):
        """每一步的读数都要合法（缩放落在 ``(0, 1]``、保留比例落在 ``[0, 1]``）。"""
        with pytest.raises(ParameterError):
            EpochRecord(
                step=1, loss=1.0, eval_loss=1.0, learning_rate=0.1, grad_norm=1.0,
                clip_scale=1.5, kept_fraction=1.0,
            )
        with pytest.raises(ParameterError):
            EpochRecord(
                step=1, loss=1.0, eval_loss=1.0, learning_rate=0.1, grad_norm=1.0,
                clip_scale=1.0, kept_fraction=1.5,
            )
        with pytest.raises(NumericError):
            EpochRecord(
                step=1, loss=float("nan"), eval_loss=1.0, learning_rate=0.1, grad_norm=1.0,
                clip_scale=1.0, kept_fraction=1.0,
            )

    def test_record_learning_rate_may_be_zero(self):
        """学习率允许是 0（退火的端点：那一步不做更新）。"""
        record = EpochRecord(
            step=1, loss=1.0, eval_loss=1.0, learning_rate=0.0, grad_norm=1.0,
            clip_scale=1.0, kept_fraction=1.0,
        )
        assert record.learning_rate == 0.0
        assert "第   1 步" in record.summary_line()

    def test_check_no_divergence_rejects_foreign_records(self):
        """传进来不是曲线时当场报错。"""
        with pytest.raises(ParameterError):
            check_no_divergence("curve")  # type: ignore[arg-type]


class TestHelpers:
    """几个便宜的小工具（演示脚本与报告共用）."""

    def test_describe_controls(self):
        """六个旋钮各一行。"""
        lines = describe_controls()
        assert len(lines) == len(CONTROLS) == 6
        assert set(CONTROL_DESCRIPTIONS) == set(CONTROLS)

    def test_shape_size(self):
        """形状声明的参数量（转发 ``StackShape.total_parameter_count``）。"""
        assert shape_size(shape()) == 1944
        with pytest.raises(ParameterError):
            shape_size("shape")  # type: ignore[arg-type]

    def test_record_count_and_steps_reject_foreign_records(self):
        """两个小工具都要挡类型。"""
        with pytest.raises(ParameterError):
            record_count("curve")  # type: ignore[arg-type]
        with pytest.raises(ParameterError):
            steps_of(3)  # type: ignore[arg-type]

    def test_curve_summary(self):
        """把一条曲线压成一个字典。"""
        payload = curve_summary(train(shape(), inputs(), target(), config=config(steps=3)))
        assert payload["steps"] == 3
        assert payload["diverged"] is False
        assert "improvement_ratio" in payload
        with pytest.raises(ParameterError):
            curve_summary("curve")  # type: ignore[arg-type]

    def test_stack_parameters_are_immutable(self):
        """传进来的参数是 ``StackParameters``（可复用、可冻结）。"""
        assert isinstance(params(), StackParameters)
