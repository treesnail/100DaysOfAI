"""``optimizers``：一次训练期更新的测试（day092 / M8-D3）.

本文件覆盖新包的八个功能模块与包入口：口径表、五个失败族、两个新增更新规则、
两种衰减、训练期优化器的组装顺序、六类场景的选型建议、六变体的收敛对比、
七条性质与六张表。

样本全部来自本包自己的确定性构造（写死的参数 / 梯度 / 起点）；**跨天对账真的调用
既有包**（``math_foundations`` 的 ``optim``：三个优化器、四种调度、两个裁剪原语），
因此这一课考的就是"这条更新被真的接上了没有"。
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.optim import make_optimizer as math_make_optimizer
from smart_research_agent.optimizers import (
    advisor,
    compare,
    errors,
    optimizer,
    rules,
    study,
    types,
    verify,
)

# --------------------------------------------------------------------------- 口径表


def test_six_optimizers_tables_are_closed() -> None:
    """六个优化器的五张表逐键对齐，且总数是 6。"""
    assert len(types.TRAIN_OPTIMIZERS) == 6
    assert set(types.TRAIN_OPTIMIZERS) == set(types.OPTIMIZER_DESCRIPTIONS)
    assert set(types.TRAIN_OPTIMIZERS) == set(types.OPTIMIZER_FORMULAS)
    assert set(types.TRAIN_OPTIMIZERS) == set(types.OPTIMIZER_STATE_KEYS)
    assert set(types.TRAIN_OPTIMIZERS) == set(types.UPDATE_RULE_FAMILIES)


def test_shared_and_new_rules_are_both_non_empty() -> None:
    """复用 day074 的三条与新增的三条都非空（否则有一档是空的）。"""
    families = set(types.UPDATE_RULE_FAMILIES.values())
    assert families == {"shared", "new"}
    assert sum(1 for value in types.UPDATE_RULE_FAMILIES.values() if value == "shared") == 3
    assert sum(1 for value in types.UPDATE_RULE_FAMILIES.values() if value == "new") == 3


def test_shared_rules_are_the_day074_names() -> None:
    """共享的三条就是 day074 的 sgd / momentum / adam（名字逐字相同）。"""
    shared = {name for name, family in types.UPDATE_RULE_FAMILIES.items() if family == "shared"}
    assert shared == {"sgd", "momentum", "adam"}


def test_state_keys_start_empty_for_sgd() -> None:
    """sgd 无状态；三个规则各自的状态变量数正确。"""
    assert types.OPTIMIZER_STATE_KEYS["sgd"] == ()
    assert len(types.OPTIMIZER_STATE_KEYS["momentum"]) == 1
    assert len(types.OPTIMIZER_STATE_KEYS["rmsprop"]) == 1
    assert len(types.OPTIMIZER_STATE_KEYS["adam"]) == 2
    assert len(types.OPTIMIZER_STATE_KEYS["adamw"]) == 2


def test_three_decay_tables_are_closed() -> None:
    """三种衰减的三张表逐键对齐。"""
    assert types.DECAY_MODES == ("none", "l2", "decoupled")
    assert set(types.DECAY_MODES) == set(types.DECAY_DESCRIPTIONS)
    assert set(types.DECAY_MODES) == set(types.DECAY_FORMULAS)


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单 / 说明 / "失败意味着什么"三张表逐键对齐。"""
    assert len(types.OPTIMIZER_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.OPTIMIZER_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.OPTIMIZER_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.OPTIMIZER_NOTES) == 10
    assert types.OPTIMIZER_NOTES_ORDER == tuple(types.OPTIMIZER_NOTES)
    assert all(types.OPTIMIZER_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）。"""
    assert len(types.OPTIMIZER_BOUNDARIES) == 5
    assert all(types.OPTIMIZER_BOUNDARIES)


def test_torch_counterparts_cover_every_rule_and_no_version() -> None:
    """PyTorch 对照表覆盖六个规则与三个工具，且**不出现**版本号。"""
    required = set(types.TRAIN_OPTIMIZERS) | {
        "decoupled_weight_decay",
        "global_norm_clip",
        "lr_scheduler",
        "zero_grad",
    }
    assert required <= set(types.TORCH_COUNTERPARTS)
    assert all(types.TORCH_COUNTERPARTS[name].startswith("torch") for name in types.TRAIN_OPTIMIZERS)
    assert all(types.TORCH_COUNTERPARTS.values())
    assert "torch==" not in " ".join(types.TORCH_COUNTERPARTS.values())


def test_properties_alias_matches() -> None:
    """``PROPERTIES`` 是性质名单的别名（同一份，不是抄一份）。"""
    assert types.PROPERTIES == types.OPTIMIZER_PROPERTIES


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_aligned() -> None:
    """五个族的"该怎么办"表与类表逐键对齐。"""
    assert len(errors.FAMILY_OUTCOMES) == 5
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "StateError",
        "StepError",
    }


def test_returned_and_absent_families_are_constants() -> None:
    """今天没有"回来"的族，且 ``GradientError`` 再次缺席（可断言的事实）。"""
    assert errors.RETURNED_FAMILY is None
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.ABSENT_FAMILY_REASON


def test_errors_inherit_both_families() -> None:
    """本族错误既是 ``OptimizerError``、也是 day075 的对应族。"""
    from smart_research_agent.transformer_core import errors as core_errors

    assert issubclass(errors.ShapeError, errors.OptimizerError)
    assert issubclass(errors.ShapeError, core_errors.ShapeError)
    assert issubclass(errors.ParameterError, core_errors.ParameterError)
    assert issubclass(errors.NumericError, core_errors.NumericError)
    assert issubclass(errors.StateError, errors.OptimizerError)
    assert issubclass(errors.StepError, errors.OptimizerError)


def test_errors_are_catchable_by_value_error() -> None:
    """所有族都继承自 ``ValueError``（与前面各天一致）。"""
    assert issubclass(errors.OptimizerError, ValueError)
    with pytest.raises(ValueError):
        raise errors.StepError("boom")


# --------------------------------------------------------------------------- 规则：校验


def test_as_vector_rejects_non_sequence() -> None:
    """非序列（例如一个整数）当场拒绝。"""
    with pytest.raises(errors.ShapeError):
        rules.as_vector(5, name="x")  # type: ignore[arg-type]


def test_as_vector_rejects_non_number_element() -> None:
    """元素不是数（例如字符串）当场拒绝。"""
    with pytest.raises(errors.ShapeError):
        rules.as_vector(("a", "b"), name="x")  # type: ignore[arg-type]


def test_as_vector_rejects_nan() -> None:
    """非有限数（nan）当场拒绝。"""
    with pytest.raises(errors.NumericError):
        rules.as_vector((1.0, float("nan")), name="x")


def test_as_vector_accepts_tuple_and_list() -> None:
    """正常输入被收敛成 ``tuple[float, ...]``。"""
    assert rules.as_vector([1, 2.5], name="x") == (1.0, 2.5)


def test_checked_positive_rejects_zero_and_negative() -> None:
    """学习率 / ε / 阈值必须为正。"""
    with pytest.raises(errors.ParameterError):
        rules.checked_positive(0.0, name="learning_rate")
    with pytest.raises(errors.ParameterError):
        rules.checked_positive(-1.0, name="learning_rate")


def test_checked_positive_rejects_non_number() -> None:
    """非数也被拒绝。"""
    with pytest.raises(errors.ParameterError):
        rules.checked_positive("0.1", name="learning_rate")  # type: ignore[arg-type]


def test_checked_fraction_bounds() -> None:
    """动量 / 衰减率必须落在 [0, 1)。"""
    assert rules.checked_fraction(0.0, name="momentum") == 0.0
    assert rules.checked_fraction(0.9, name="momentum") == 0.9
    with pytest.raises(errors.ParameterError):
        rules.checked_fraction(1.0, name="momentum")
    with pytest.raises(errors.ParameterError):
        rules.checked_fraction(-0.1, name="momentum")
    with pytest.raises(errors.ParameterError):
        rules.checked_fraction("x", name="momentum")  # type: ignore[arg-type]


def test_checked_pair_shape_mismatch() -> None:
    """参数与梯度长度不一致时抛 ``ShapeError``。"""
    with pytest.raises(errors.ShapeError):
        rules.checked_pair((1.0, 2.0), (1.0,))


def test_checked_decay_mode_rejects_unknown() -> None:
    """不认识的衰减模式当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        rules.checked_decay_mode("cosine")
    assert rules.checked_decay_mode("l2") == "l2"


# --------------------------------------------------------------------------- 规则：衰减


def test_l2_gradient_zero_weight_decay_is_identity() -> None:
    """衰减为 0 时梯度原样返回。"""
    grads = (1.0, -2.0)
    assert rules.l2_gradient(grads, (10.0, 20.0), 0.0) == grads


def test_l2_gradient_adds_lambda_theta() -> None:
    """耦合衰减把 λθ 加进梯度。"""
    result = rules.l2_gradient((1.0, 1.0), (2.0, 4.0), 0.5)
    assert result == (2.0, 3.0)


def test_l2_gradient_shape_and_negative() -> None:
    """形状不符与负衰减都当场拒绝。"""
    with pytest.raises(errors.ShapeError):
        rules.l2_gradient((1.0, 2.0), (1.0,), 0.1)
    with pytest.raises(errors.ParameterError):
        rules.l2_gradient((1.0,), (1.0,), -0.1)


def test_decoupled_decay_scales_params() -> None:
    """解耦衰减把参数乘 (1 − lr·λ)。"""
    result = rules.apply_decoupled_decay((10.0, 20.0), 0.1, 0.5)
    assert result == pytest.approx((9.5, 19.0))


def test_decoupled_decay_zero_is_identity() -> None:
    """衰减为 0 时参数原样返回（且不校验 lr）。"""
    params = (1.0, 2.0)
    assert rules.apply_decoupled_decay(params, 0.0, 0.0) == params


def test_decoupled_decay_rejects_negative() -> None:
    """负衰减当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        rules.apply_decoupled_decay((1.0,), 0.1, -0.5)


# --------------------------------------------------------------------------- 规则：新更新规则


def test_nesterov_first_step_direction() -> None:
    """第一步：方向 = g + β·g（速度由上一步的 0 变成 g）。"""
    updated, velocity = rules.nesterov_step(
        (1.0,), (2.0,), (0.0,), learning_rate=0.1, momentum=0.5
    )
    assert velocity == pytest.approx((2.0,))
    # 方向 = g + β·v = 2 + 0.5·2 = 3 ⇒ 1 − 0.1·3 = 0.7
    assert updated == pytest.approx((0.7,))


def test_nesterov_direction_differs_from_momentum() -> None:
    """同一组输入下，Nesterov 的方向比普通动量多一项 β·v。"""
    params = (0.0,)
    grads = (1.0,)
    velocity = (4.0,)
    nesterov, _ = rules.nesterov_step(params, grads, velocity, learning_rate=0.1, momentum=0.5)
    momentum_velocity = tuple(0.5 * 4.0 + 1.0 for _ in params)
    momentum = tuple(p - 0.1 * v for p, v in zip(params, momentum_velocity))
    assert nesterov != momentum


def test_nesterov_state_length_mismatch() -> None:
    """速度长度与参数不符时抛 ``ShapeError``。"""
    with pytest.raises(errors.ShapeError):
        rules.nesterov_step((1.0, 2.0), (1.0, 2.0), (0.0,), learning_rate=0.1)


def test_checked_non_negative_allows_zero() -> None:
    """非负校验：0 允许（末段不再更新），负值与非数拒绝。"""
    assert rules.checked_non_negative(0.0, name="lr") == 0.0
    with pytest.raises(errors.ParameterError):
        rules.checked_non_negative(-0.1, name="lr")
    with pytest.raises(errors.ParameterError):
        rules.checked_non_negative("x", name="lr")  # type: ignore[arg-type]


def test_rmsprop_first_step_normalizes_sign() -> None:
    """第一步二阶动量只有 (1−ρ)g²；更新量 = lr·g/√((1−ρ)g²)（多步后趋近 lr·sign(g)）。"""
    updated, square = rules.rmsprop_step(
        (0.0,), (5.0,), (0.0,), learning_rate=0.1, alpha=0.9, epsilon=1e-12
    )
    assert square == pytest.approx((0.1 * 25.0,))
    assert updated[0] == pytest.approx(-0.1 * 5.0 / math.sqrt(2.5))


def test_rmsprop_state_length_mismatch() -> None:
    """二阶动量长度不符时抛 ``ShapeError``。"""
    with pytest.raises(errors.ShapeError):
        rules.rmsprop_step((1.0, 2.0), (1.0, 2.0), (0.0,), learning_rate=0.1)


def test_rmsprop_equalizes_two_scales() -> None:
    """相差 1e4 的梯度在稳态下给出几乎相同的每步位移。"""
    def steady(gradient: float) -> float:
        params = (0.0,)
        square = (0.0,)
        step = 0.0
        for _ in range(200):
            params, square = rules.rmsprop_step(
                (0.0,), (gradient,), square, learning_rate=0.05, alpha=0.99, epsilon=1e-8
            )
            step = abs(params[0])
        return step

    assert abs(steady(100.0) - steady(0.01)) < 1e-5


# --------------------------------------------------------------------------- 规则：裁剪与几何


def test_clip_gradients_under_limit_is_identity() -> None:
    """范数未超阈值时原样返回、系数 1.0。"""
    grads = (0.3, 0.4)
    clipped, factor = rules.clip_gradients(grads, 1.0)
    assert clipped == pytest.approx(grads)
    assert factor == 1.0


def test_clip_gradients_over_limit_scales() -> None:
    """范数超阈值时整体缩放，系数 = 阈值/范数。"""
    grads = (3.0, 4.0)
    clipped, factor = rules.clip_gradients(grads, 1.0)
    assert factor == pytest.approx(0.2)
    assert rules.global_norm(clipped) == pytest.approx(1.0)


def test_clip_gradients_matches_day074() -> None:
    """裁剪与 day074 的 ``clip_by_global_norm`` 逐位一致。"""
    from smart_research_agent.math_foundations.optim import clip_by_global_norm

    ours, our_factor = rules.clip_gradients(verify.CLIP_SAMPLE_GRADS, 1.0)
    theirs, their_factor = clip_by_global_norm(verify.CLIP_SAMPLE_GRADS, 1.0)
    assert ours == pytest.approx(theirs)
    assert our_factor == pytest.approx(their_factor)


def test_clip_gradients_rejects_bad_threshold() -> None:
    """裁剪阈值必须为正。"""
    with pytest.raises(errors.ParameterError):
        rules.clip_gradients((1.0,), 0.0)


def test_cosine_between_parallel_is_one() -> None:
    """同向向量的余弦为 1。"""
    assert rules.cosine_between((1.0, 2.0), (3.0, 6.0)) == pytest.approx(1.0)


def test_cosine_between_zero_vector_raises() -> None:
    """零向量没有方向。"""
    with pytest.raises(errors.NumericError):
        rules.cosine_between((0.0, 0.0), (1.0, 0.0))


def test_cosine_between_length_mismatch() -> None:
    """长度不一致无法比较方向。"""
    with pytest.raises(errors.ShapeError):
        rules.cosine_between((1.0,), (1.0, 0.0))


# --------------------------------------------------------------------------- 训练期优化器：构造与校验


def test_unknown_optimizer_name_rejected() -> None:
    """不认识的优化器名当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        optimizer.TrainingOptimizer("rms", 0.1)
    with pytest.raises(errors.ParameterError):
        optimizer.make_train_optimizer("rms", 0.1)


def test_bad_learning_rate_rejected() -> None:
    """非正学习率当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        optimizer.TrainingOptimizer("sgd", 0.0)


def test_bad_weight_decay_rejected() -> None:
    """负衰减当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        optimizer.TrainingOptimizer("sgd", 0.1, weight_decay=-1.0)


def test_bad_decay_mode_rejected() -> None:
    """不认识的衰减模式当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        optimizer.TrainingOptimizer("sgd", 0.1, decay_mode="cosine")


def test_bad_max_grad_norm_rejected() -> None:
    """非正裁剪阈值当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        optimizer.TrainingOptimizer("sgd", 0.1, max_grad_norm=0.0)


def test_core_rule_mapping() -> None:
    """六个规则的核心规则映射正确（AdamW 的核心是 Adam）。"""
    assert optimizer.CORE_RULE_FOR["adamw"] == "adam"
    assert optimizer.CORE_RULE_FOR["nesterov"] == "nesterov"
    assert optimizer.TrainingOptimizer("adamw", 0.1).core_rule == "adam"


def test_adamw_defaults_to_decoupled_decay() -> None:
    """AdamW 的默认衰减模式是解耦（这是它存在的理由）。"""
    assert optimizer.TrainingOptimizer("adamw", 0.1).decay_mode == "decoupled"
    assert optimizer.TrainingOptimizer("adam", 0.1).decay_mode == "none"


def test_all_six_are_supported() -> None:
    """六个优化器都能被工厂造出来。"""
    assert set(optimizer.SUPPORTED_OPTIMIZERS) == set(types.TRAIN_OPTIMIZERS)
    for name in types.TRAIN_OPTIMIZERS:
        assert optimizer.make_train_optimizer(name, 0.01).name == name


def test_state_is_json_serializable() -> None:
    """``state()`` 可 ``json.dumps``（跨步状态只有那些常量与列表）。"""
    opt = optimizer.TrainingOptimizer("adam", 0.1)
    opt.step((1.0, 2.0), (0.1, -0.2))
    payload = opt.state()
    assert json.loads(json.dumps(payload))["name"] == "adam"
    assert payload["step_count"] == 1


def test_state_omits_empty_state() -> None:
    """没有状态变量的规则，``state()`` 里不含 velocity / square_avg 键。"""
    payload = optimizer.TrainingOptimizer("sgd", 0.1).state()
    assert "velocity" not in payload
    assert payload["base"] == {"name": "sgd", "learning_rate": 0.1, "step_count": 0}


def test_state_includes_new_rule_state() -> None:
    """新增规则的状态会被写进 ``state()``。"""
    rms = optimizer.TrainingOptimizer("rmsprop", 0.1)
    rms.step((1.0,), (1.0,))
    assert "square_avg" in rms.state()
    nest = optimizer.TrainingOptimizer("nesterov", 0.1)
    nest.step((1.0,), (1.0,))
    assert "velocity" in nest.state()


def test_reset_clears_state() -> None:
    """``reset()`` 清空步数与跨步状态，但参数不动。"""
    opt = optimizer.TrainingOptimizer("momentum", 0.1)
    opt.step((1.0,), (1.0,))
    opt.reset()
    assert opt.step_count == 0
    assert opt.velocity == ()
    assert opt.state()["step_count"] == 0


def test_reset_without_base_optimizer() -> None:
    """新增规则的 ``reset()`` 同样有效（它们没有 day074 的基座对象）。"""
    opt = optimizer.TrainingOptimizer("rmsprop", 0.1)
    opt.step((1.0,), (1.0,))
    opt.reset()
    assert opt.square_avg == ()
    assert opt._base is None


def test_describe_contains_name_and_config() -> None:
    """``describe()`` 一行里能读出名字、学习率、衰减与裁剪。"""
    text = optimizer.TrainingOptimizer("adamw", 0.01, weight_decay=0.1).describe()
    assert "adamw" in text
    assert "0.01" in text
    assert "decoupled" in text


def test_as_shape_check() -> None:
    """显式形状检查：一致通过、不一致抛错。"""
    optimizer.as_shape_check((1.0, 2.0), (1.0, 2.0))
    with pytest.raises(errors.ShapeError):
        optimizer.as_shape_check((1.0, 2.0), (1.0,))


# --------------------------------------------------------------------------- 训练期优化器：一步的行为


def test_sgd_step_formula() -> None:
    """SGD 一步：θ − lr·g。"""
    opt = optimizer.TrainingOptimizer("sgd", 0.1)
    assert opt.step((1.0,), (2.0,)) == pytest.approx((0.8,))


def test_momentum_first_step_equals_sgd() -> None:
    """动量第一步与 SGD 相同（速度从 0 变成 g，方向还是 g）。"""
    sgd = optimizer.TrainingOptimizer("sgd", 0.1).step((1.0,), (2.0,))
    momentum = optimizer.TrainingOptimizer("momentum", 0.1).step((1.0,), (2.0,))
    assert sgd == pytest.approx(momentum)


def test_nesterov_first_step_is_larger_than_momentum() -> None:
    """Nesterov 第一步的步长比普通动量大（多一项 β·g）。"""
    params, grads = (1.0,), (2.0,)
    momentum = optimizer.TrainingOptimizer("momentum", 0.1).step(params, grads)
    nesterov = optimizer.TrainingOptimizer("nesterov", 0.1).step(params, grads)
    assert abs(nesterov[0] - 1.0) > abs(momentum[0] - 1.0)


def test_six_optimizers_all_reduce_a_bowl() -> None:
    """六个优化器在良态碗上都能把损失降下来（不要求收敛到 0）。"""
    def bowl(theta):
        return sum(v * v for v in theta)

    for name in types.TRAIN_OPTIMIZERS:
        row = compare.run_optimizer(
            optimizer.make_train_optimizer(name, 0.05),
            bowl,
            (3.0, -2.0),
            objective_name="bowl",
            steps=30,
        )
        assert row.final_loss < row.initial_loss, name


def test_l2_decay_mode_path() -> None:
    """耦合衰减：梯度先被加上 λθ，因此第一步与无衰减不同。"""
    plain = optimizer.TrainingOptimizer("sgd", 0.1).step((1.0,), (0.0,))
    decayed = optimizer.TrainingOptimizer("sgd", 0.1, weight_decay=0.5, decay_mode="l2").step(
        (1.0,), (0.0,)
    )
    assert plain == pytest.approx((1.0,))
    assert decayed == pytest.approx((0.95,))


def test_decoupled_decay_mode_path() -> None:
    """解耦衰减：更新之后再缩参数，因此第一步也会略小。"""
    decayed = optimizer.TrainingOptimizer(
        "sgd", 0.1, weight_decay=0.5, decay_mode="decoupled"
    ).step((1.0,), (0.0,))
    assert decayed == pytest.approx((1.0 * (1.0 - 0.1 * 0.5),))


def test_clip_sets_fields() -> None:
    """开启裁剪后，``last_grad_norm`` 与 ``last_clip_factor`` 被记录。"""
    opt = optimizer.TrainingOptimizer("sgd", 0.1, max_grad_norm=1.0)
    opt.step((0.0, 0.0), (3.0, 4.0))
    assert opt.last_grad_norm == pytest.approx(5.0)
    assert opt.last_clip_factor == pytest.approx(0.2)


def test_no_clip_records_norm_with_factor_one() -> None:
    """未开启裁剪时系数恒为 1.0，但仍记录范数。"""
    opt = optimizer.TrainingOptimizer("sgd", 0.1)
    opt.step((0.0, 0.0), (3.0, 4.0))
    assert opt.last_grad_norm == pytest.approx(5.0)
    assert opt.last_clip_factor == 1.0


def test_schedule_changes_learning_rate_per_step() -> None:
    """装上调度的优化器每一步的学习率按曲线走（先升后降）。"""
    opt = optimizer.TrainingOptimizer("sgd", 0.1)
    opt.set_schedule("warmup_cosine", base_lr=0.1, warmup_steps=3, total_steps=6, min_lr=0.0)
    rates = []
    for _ in range(6):
        opt.step((1.0,), (0.0,))
        rates.append(opt.learning_rate)
    assert rates[0] < rates[2] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.0)  # 余弦退火的末步合法地取到 min_lr=0


def test_step_error_on_overflow() -> None:
    """学习率大到把参数推到 inf 时抛 ``StepError``。"""
    opt = optimizer.TrainingOptimizer("sgd", 1e308)
    with pytest.raises(errors.StepError):
        opt.step((0.0,), (1e308,))


def test_state_error_on_wrong_velocity_length() -> None:
    """跨步状态与参数形状不符时抛 ``StateError``。"""
    opt = optimizer.TrainingOptimizer("nesterov", 0.1)
    opt.velocity = (0.0, 0.0)  # 模拟"上一段任务留下的状态"
    with pytest.raises(errors.StateError):
        opt.step((1.0, 2.0, 3.0), (1.0, 1.0, 1.0))


def test_rmsprop_state_length_checked_via_optimizer() -> None:
    """RMSProp 的二阶动量同样受状态检查保护。"""
    opt = optimizer.TrainingOptimizer("rmsprop", 0.1)
    opt.square_avg = (0.0,)
    with pytest.raises(errors.StateError):
        opt.step((1.0, 2.0), (1.0, 1.0))


def test_shape_error_on_mismatched_step() -> None:
    """直接迈步时参数与梯度长度不符抛 ``ShapeError``。"""
    with pytest.raises(errors.ShapeError):
        optimizer.TrainingOptimizer("sgd", 0.1).step((1.0, 2.0), (1.0,))


# --------------------------------------------------------------------------- 选型建议


def test_scenarios_and_recommendations_aligned() -> None:
    """场景表与推荐表逐键对齐，且每个场景都有非空的 why。"""
    assert set(advisor.SCENARIOS) == set(advisor.RECOMMENDATIONS)
    assert all(rec.why for rec in advisor.RECOMMENDATIONS.values())


def test_recommend_unknown_rejected() -> None:
    """不认识的场景当场拒绝，而不是给个默认值。"""
    with pytest.raises(errors.ParameterError):
        advisor.recommend("sft-magic")


def test_recommended_optimizers_are_known() -> None:
    """推荐里出现的优化器都属于六个规则。"""
    assert set(advisor.RECOMMENDED_OPTIMIZERS) <= set(types.TRAIN_OPTIMIZERS)


def test_recommendation_learning_rates_are_positive() -> None:
    """每条推荐的起点学习率都为正（可执行）。"""
    for rec in advisor.RECOMMENDATIONS.values():
        assert rec.learning_rate > 0
        assert rec.weight_decay >= 0


def test_recommendation_builds_usable_optimizer() -> None:
    """按推荐造出的优化器能在良态碗上真的把损失降下来。"""
    def bowl(theta):
        return sum(v * v for v in theta)

    rec = advisor.recommend("scratch-mlp")
    row = compare.run_optimizer(rec.build(), bowl, (3.0, -2.0), objective_name="bowl", steps=40)
    assert row.final_loss < row.initial_loss


def test_recommendation_line_and_scenario_line() -> None:
    """两行说明都能印出场景名。"""
    rec = advisor.recommend("cpu-tiny")
    assert "cpu-tiny" in rec.line()
    assert "cpu-tiny" in advisor.SCENARIOS["cpu-tiny"].line()


def test_recommendation_lines_non_empty() -> None:
    """六条推荐逐行印出（每个场景两行 + 每条 why 若干行）。"""
    lines = advisor.recommendation_lines()
    assert len(lines) >= 6 * 2


def test_adamw_is_recommended_for_finetuning() -> None:
    """三个大模型微调场景都推荐 AdamW（解耦衰减的理由可读）。"""
    for scenario in ("sft-lora", "sft-full", "dpo"):
        assert advisor.recommend(scenario).optimizer == "adamw"
        assert advisor.recommend(scenario).decay_mode == "decoupled"


# --------------------------------------------------------------------------- 收敛对比


def test_objectives_have_expected_keys() -> None:
    """三个目标函数都有各自的起点、步数与学习率。"""
    assert set(compare.OBJECTIVES) == {"quadratic_bowl", "anisotropic_valley", "rosenbrock"}
    for _function, _initial, steps, lr in compare.OBJECTIVES.values():
        assert steps > 0
        assert lr > 0


def test_compare_unknown_objective_rejected() -> None:
    """不认识的目标函数当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        compare.compare_optimizers(objective_name="sphere")


def test_compare_returns_one_row_per_name() -> None:
    """每个名字一行，且顺序与传入一致。"""
    rows = compare.compare_optimizers(("sgd", "adam"), objective_name="quadratic_bowl")
    assert [row.name for row in rows] == ["sgd", "adam"]
    assert all(row.objective == "quadratic_bowl" for row in rows)


def test_compare_all_six_improve_on_bowl() -> None:
    """六个变体在良态碗上都把损失降低了。"""
    rows = compare.compare_optimizers(objective_name="quadratic_bowl")
    assert len(rows) == 6
    assert all(row.final_loss < row.initial_loss for row in rows)


def test_best_row_picks_smallest_final_loss() -> None:
    """``best_row`` 按最终损失取最小（并列时名字序）。"""
    rows = compare.compare_optimizers(("sgd", "adam"), objective_name="quadratic_bowl")
    best = compare.best_row(rows)
    assert best.final_loss == min(row.final_loss for row in rows)


def test_best_row_empty_rejected() -> None:
    """空结果没有"最好的一行"。"""
    with pytest.raises(errors.ParameterError):
        compare.best_row([])


def test_divergence_is_flagged_not_formatted_absurdly() -> None:
    """发散的行被标记，``line()`` 里印的是倍数而不是巨大百分比。"""
    row = compare.run_optimizer(
        optimizer.make_train_optimizer("sgd", 0.5),
        compare.rosenbrock,
        (5.0, 5.0),
        objective_name="rosenbrock",
        steps=20,
    )
    assert row.diverged is True
    assert "发散" in row.line()


def test_rosenbrock_default_lr_keeps_rows_readable() -> None:
    """Rosenbrock 用自己的小学习率，因此不是每一行都发散。"""
    rows = compare.compare_optimizers(objective_name="rosenbrock")
    assert any(not row.diverged for row in rows)


def test_compare_accepts_extra_kwargs() -> None:
    """``optimizer_kwargs`` 会透传给每个优化器（这里给动量系数）。"""
    rows = compare.compare_optimizers(("momentum",), objective_name="quadratic_bowl", momentum=0.5)
    assert len(rows) == 1


def test_row_line_reports_tolerance_and_monotone() -> None:
    """一行读数里含到容差的步数或横杠、以及单调性。"""
    row = compare.run_optimizer(
        optimizer.make_train_optimizer("sgd", 0.05),
        compare.quadratic_bowl,
        (3.0, -2.0),
        objective_name="quadratic_bowl",
        steps=40,
        tolerance=1e-2,
    )
    assert "到容差" in row.line()
    assert "单调" in row.line()


def test_run_optimizer_default_objective_name() -> None:
    """不传目标名时用 ``custom``（默认参数可用）。"""
    row = compare.run_optimizer(
        optimizer.make_train_optimizer("adam", 0.05), compare.quadratic_bowl, (1.0, 1.0), steps=5
    )
    assert row.objective == "custom"


# --------------------------------------------------------------------------- 性质校验


def test_check_semantics_upper_and_lower() -> None:
    """``Check`` 的上界 / 下界 / 同时给三种判据。"""
    assert verify.Check(reading=0.5, upper_bound=1.0).passed() is True
    assert verify.Check(reading=1.5, upper_bound=1.0).passed() is False
    assert verify.Check(reading=0.5, lower_bound=1.0).passed() is False
    assert verify.Check(reading=1.5, lower_bound=1.0).passed() is True
    assert verify.Check(reading=1.0, lower_bound=1.0, upper_bound=1.0).passed() is True


def test_check_relation_and_bound_text() -> None:
    """判据的文本形态随上/下界变化。"""
    assert verify.Check(reading=0.0).relation() == "=="
    assert verify.Check(reading=0.0, upper_bound=1.0).relation() == "<="
    assert verify.Check(reading=0.0, lower_bound=1.0).relation() == ">="
    assert "∈" in verify.Check(reading=0.0, lower_bound=0.0, upper_bound=1.0).bound_text()
    assert verify.Check(reading=0.0).bound_text() == "== 逐位"


def test_check_all_is_seven_and_all_pass() -> None:
    """七条性质全部通过（这就是"这条更新被真的接上了"）。"""
    report = verify.check_all()
    assert report.total == 7
    assert report.all_passed()
    assert len(report.lines()) == 7


def test_each_check_returns_passing_outcome() -> None:
    """逐条跑：每条都返回通过的 ``PropertyOutcome`` 且读数有限。"""
    for name, function in verify.CHECKS.items():
        outcome = function()
        assert outcome.name == name
        assert outcome.passed is True
        assert math.isfinite(outcome.check.reading)
        assert outcome.line().startswith("通过")


def test_shared_rules_check_is_bitwise() -> None:
    """共享规则对账的读数是 0（逐位一致，不是"接近"）。"""
    assert verify.check_shared_rules_match_day074().check.reading == 0.0


def test_nesterov_check_has_lower_bound() -> None:
    """Nesterov 那条用的是下界判据（它们必须不同）。"""
    outcome = verify.check_nesterov_is_not_momentum()
    assert outcome.check.lower_bound is not None
    assert outcome.check.upper_bound is None


def test_decoupled_check_has_lower_bound() -> None:
    """解耦衰减那条用下界（AdamW 与 Adam+L2 必须不同）。"""
    outcome = verify.check_decoupled_decay_differs_under_varying_gradients()
    assert outcome.check.lower_bound is not None


def test_report_passed_and_total() -> None:
    """报告的计数与条数接口。"""
    report = verify.check_all()
    assert report.passed == report.total == 7


# --------------------------------------------------------------------------- 六张表


def test_rule_rows_cover_six() -> None:
    """规则表六行，且每行都带公式与归属。"""
    rows = study.rule_rows()
    assert len(rows) == 6
    assert all(row.formula for row in rows)
    assert {row.family for row in rows} == {"shared", "new"}


def test_one_step_rows_cover_six() -> None:
    """一步表六行，且每行都有读数。"""
    rows = study.one_step_rows()
    assert len(rows) == 6
    assert all(math.isfinite(row.first_after) for row in rows)
    assert all(row.step_norm >= 0 for row in rows)


def test_clip_row_reading() -> None:
    """裁剪表：阈值 1.0 下裁剪后范数恰好是 1、方向余弦为 1。"""
    row = study.clip_row()
    assert row.norm_before > 1.0
    assert row.norm_after == pytest.approx(1.0)
    assert row.cosine == pytest.approx(1.0)


def test_schedule_rows_rise_then_fall() -> None:
    """调度表：热身段升、退火段降。"""
    rows = study.schedule_rows()
    rates = {row.step: row.learning_rate for row in rows}
    assert rates[1] < rates[2]
    assert rates[5] == pytest.approx(0.1)  # 热身结束点 = base_lr
    assert rates[20] < rates[13]


def test_compare_rows_for_each_objective() -> None:
    """对比表：三个目标各给出六行。"""
    for objective_name in compare.OBJECTIVES:
        rows = study.compare_rows(objective_name)
        assert len(rows) == 6


def test_property_rows_are_seven_and_pass() -> None:
    """性质表七行，全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert all("vs" in row.cross_check for row in rows)


def test_study_lines_has_six_sections() -> None:
    """一次跑完六张表：六个小节标题都在。"""
    lines = study.study_lines()
    text = "\n".join(lines)
    for index in range(1, 7):
        assert f"== {index}." in text


def test_property_names_alias() -> None:
    """``PROPERTY_NAMES`` 就是性质名单。"""
    assert study.PROPERTY_NAMES == types.OPTIMIZER_PROPERTIES


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_non_empty_and_has_no_submodules() -> None:
    """包入口的 ``__all__`` 非空、且不含子模块名。"""
    import smart_research_agent.optimizers as package

    assert len(package.__all__) > 50
    assert not ({"errors", "types", "rules", "optimizer", "advisor", "compare", "verify", "study"} & set(package.__all__))


def test_package_star_import_resolves_every_name() -> None:
    """每个 ``__all__`` 里的名字都能在包命名空间里取到（否则 star import 会少名字）。"""
    import smart_research_agent.optimizers as package

    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


def test_package_exposes_key_objects() -> None:
    """几个关键对象从包入口就能取到。"""
    import smart_research_agent.optimizers as package

    assert package.TrainingOptimizer is optimizer.TrainingOptimizer
    assert package.Check is verify.Check
    assert package.recommend is advisor.recommend
    assert package.compare_optimizers is compare.compare_optimizers


def test_math_foundations_still_reachable_for_cross_check() -> None:
    """跨天对账的对方（day074）仍然可造对象——本课没有改动它。"""
    assert math_make_optimizer("sgd", 0.1).name == "sgd"
