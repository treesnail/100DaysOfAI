"""day058 重训触发测试（M5-D9）：触发器与否决项是两类东西.

这个文件守的是本模块唯一需要记住的区分：

```text
trigger：有一个理由支持训练      veto：有一个理由禁止训练
should_retrain = any(trigger.fired) and not vetoed_by
```

把 veto 写成触发器的反向条件（例如再加一个 ``cooldown_passed`` 要求它 fired），
报告里就分不清这两种状态：

```text
fired = []                            → 没有理由训练        （去看数据与线上评估）
fired = [数据涨了] vetoed_by=[冷却]   → 有理由但时机不允许   （只需要等）
```

它们的运维动作完全不同，因此断言也分开写。

另一组断言围绕"**缺失 ≠ 0**"：``online_pass_rate=None`` 不触发质量下降、
``hours_since_last_train=None`` 不否决冷却期。把缺失当 0 会让"从未训练过"
的仓库被冷却期永久拦住，而它恰恰最该训练一次。
"""

from __future__ import annotations

import pytest

from smart_research_agent.registry import RegistryError
from smart_research_agent.registry.triggers import (
    KIND_TRIGGER,
    KIND_VETO,
    TRIGGER_DATASET_CHANGE,
    TRIGGER_DATA_GROWTH,
    TRIGGER_QUALITY_DROP,
    VETO_ACTIVE_RUN,
    VETO_COOLDOWN,
    RetrainDecision,
    TriggerEvaluation,
    TriggerPolicy,
    TriggerState,
    evaluate_triggers,
    trigger_table,
)

DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"


# --------------------------------------------------------------------------- #
# TriggerState
# --------------------------------------------------------------------------- #


def test_trigger_state_defaults_are_all_missing() -> None:
    """缺省状态是"什么都不知道"：没有增量、没有观测、从未训练。"""
    state = TriggerState()
    assert (state.new_examples, state.online_pass_rate) == (0, None)
    assert state.hours_since_last_train is None
    assert state.dataset_changed() is False


def test_trigger_state_dataset_changed_requires_a_baseline() -> None:
    """没有基线指纹不是"变了"——首次上线不该被误判成数据漂移。"""
    assert TriggerState(dataset_fingerprint=DATA_A).dataset_changed() is False
    assert (
        TriggerState(dataset_fingerprint=DATA_A, stable_dataset_fingerprint=DATA_A)
        .dataset_changed()
        is False
    )
    assert (
        TriggerState(dataset_fingerprint=DATA_B, stable_dataset_fingerprint=DATA_A)
        .dataset_changed()
        is True
    )


def test_trigger_state_to_dict_includes_derived_flag() -> None:
    """投影里带上派生量 ``dataset_changed``，报告不必自己再算一次。"""
    payload = TriggerState(
        new_examples=30, dataset_fingerprint=DATA_B, stable_dataset_fingerprint=DATA_A
    ).to_dict()
    assert payload["dataset_changed"] is True
    assert payload["new_examples"] == 30


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"new_examples": -1}, "new_examples"),
        ({"active_runs": -1}, "active_runs"),
        ({"online_pass_rate": 1.5}, "online_pass_rate"),
        ({"online_pass_rate": -0.1}, "online_pass_rate"),
        ({"hours_since_last_train": -1.0}, "hours_since_last_train"),
    ],
)
def test_trigger_state_validates_at_construction(kwargs: dict, match: str) -> None:
    """非法状态在构造期拦下，而不是在判定里变成一个无声的边界值。"""
    with pytest.raises(RegistryError, match=match):
        TriggerState(**kwargs)


def test_trigger_state_allows_boundary_values() -> None:
    """边界值合法：0 条新增、0 小时、合格率正好 0 或 1。"""
    assert TriggerState(new_examples=0, active_runs=0).new_examples == 0
    assert TriggerState(hours_since_last_train=0.0).hours_since_last_train == 0.0
    assert TriggerState(online_pass_rate=0.0).online_pass_rate == 0.0
    assert TriggerState(online_pass_rate=1.0).online_pass_rate == 1.0


# --------------------------------------------------------------------------- #
# TriggerPolicy
# --------------------------------------------------------------------------- #


def test_trigger_policy_defaults() -> None:
    """五个缺省值（进文档与端点，因此逐项钉住）。"""
    policy = TriggerPolicy()
    assert policy.to_dict() == {
        "min_new_examples": 8,
        "min_pass_rate": 0.5,
        "cooldown_hours": 24.0,
        "max_parallel_runs": 1,
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_new_examples": 0}, "min_new_examples"),
        ({"min_pass_rate": 1.5}, "min_pass_rate"),
        ({"min_pass_rate": -0.1}, "min_pass_rate"),
        ({"cooldown_hours": -1.0}, "cooldown_hours"),
        ({"max_parallel_runs": 0}, "max_parallel_runs"),
    ],
)
def test_trigger_policy_validates_at_construction(kwargs: dict, match: str) -> None:
    """``min_new_examples=0`` 会让"一条没多"也触发训练，必须拦下。"""
    with pytest.raises(RegistryError, match=match):
        TriggerPolicy(**kwargs)


# --------------------------------------------------------------------------- #
# TriggerEvaluation / RetrainDecision
# --------------------------------------------------------------------------- #


def test_trigger_evaluation_rejects_unknown_kind() -> None:
    """类别只能是 trigger / veto——第三种类别会让判定规则失去定义。"""
    with pytest.raises(RegistryError, match="未知类别"):
        TriggerEvaluation(
            name="x", kind="hint", fired=True, actual=1, threshold=1, reason=""
        )


def test_trigger_evaluation_projection_and_summary() -> None:
    """一条评估的六个字段都要能投影与打印（含 None 的合法语义）。"""
    item = TriggerEvaluation(
        name=TRIGGER_QUALITY_DROP,
        kind=KIND_TRIGGER,
        fired=False,
        actual=None,
        threshold=0.5,
        reason="未观测到线上合格率",
    )
    assert item.to_dict()["actual"] is None
    line = item.summary_line()
    assert "未命中" in line and TRIGGER_QUALITY_DROP in line


def test_trigger_evaluation_summary_marks_fired() -> None:
    """命中时摘要写"命中"，避免"命中/未命中"两个词互换。"""
    item = TriggerEvaluation(
        name=VETO_COOLDOWN, kind=KIND_VETO, fired=True, actual=2.0, threshold=24.0, reason=""
    )
    assert "命中" in item.summary_line()


def test_retrain_decision_splits_triggers_and_vetoes() -> None:
    """``triggers()`` 与 ``vetoes()`` 各返回自己那一类。"""
    decision = evaluate_triggers(
        TriggerState(
            new_examples=30,
            dataset_fingerprint=DATA_B,
            stable_dataset_fingerprint=DATA_A,
            hours_since_last_train=2.0,
        )
    )
    assert [item.name for item in decision.triggers()] == [
        TRIGGER_DATA_GROWTH,
        TRIGGER_DATASET_CHANGE,
        TRIGGER_QUALITY_DROP,
    ]
    assert [item.name for item in decision.vetoes()] == [VETO_COOLDOWN, VETO_ACTIVE_RUN]


def test_retrain_decision_summary_lines_cover_three_states() -> None:
    """三种状态的摘要各不相同（"该训 / 该等等 / 不用训"）。"""
    train = evaluate_triggers(
        TriggerState(
            new_examples=30,
            dataset_fingerprint=DATA_B,
            stable_dataset_fingerprint=DATA_A,
            hours_since_last_train=48.0,
        )
    )
    wait = evaluate_triggers(
        TriggerState(
            new_examples=30,
            dataset_fingerprint=DATA_B,
            stable_dataset_fingerprint=DATA_A,
            hours_since_last_train=1.0,
        )
    )
    idle = evaluate_triggers(TriggerState(dataset_fingerprint=DATA_A))
    assert "应该重训" in train.summary_line()
    assert "暂不重训" in wait.summary_line() and VETO_COOLDOWN in wait.summary_line()
    assert "无需重训" in idle.summary_line()


def test_retrain_decision_to_dict_and_markdown() -> None:
    """投影与 markdown 渲染（评审看的就是后者）。"""
    decision = evaluate_triggers(TriggerState(new_examples=30, dataset_fingerprint=DATA_A))
    payload = decision.to_dict()
    assert payload["should_retrain"] is True
    assert payload["fired"] == [TRIGGER_DATA_GROWTH]
    assert payload["vetoed_by"] == []
    assert len(payload["evaluations"]) == 5
    markdown = decision.render_markdown()
    assert "# 重训触发判定" in markdown
    assert TRIGGER_DATA_GROWTH in markdown
    assert "**应该重训**" in markdown


def test_retrain_decision_is_frozen() -> None:
    """判定是**结论**，不可就地改写（改写会让报告与判定不一致）。"""
    decision = evaluate_triggers(TriggerState(new_examples=1))
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        decision.should_retrain = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# evaluate_triggers：六种状态
# --------------------------------------------------------------------------- #


def test_first_training_is_not_blocked_by_cooldown() -> None:
    """从未训练（``hours_since_last_train=None``）时冷却期**不否决**.

    这是"缺失 ≠ 0"最要紧的一处：把 None 当 0 会让新仓库被冷却期永久拦住，
    而它恰恰最该训练一次。
    """
    decision = evaluate_triggers(
        TriggerState(
            new_examples=37,
            dataset_fingerprint=DATA_A,
            stable_dataset_fingerprint="",
            hours_since_last_train=None,
        )
    )
    assert decision.should_retrain is True
    assert decision.vetoed_by == ()
    cooldown = next(item for item in decision.vetoes() if item.name == VETO_COOLDOWN)
    assert cooldown.fired is False
    assert "从未训练过" in cooldown.reason


def test_data_growth_fires_on_and_above_threshold() -> None:
    """增量门槛是闭区间：正好 8 条触发，7 条不触发。"""
    assert evaluate_triggers(TriggerState(new_examples=8)).fired == (TRIGGER_DATA_GROWTH,)
    assert evaluate_triggers(TriggerState(new_examples=7)).fired == ()


def test_dataset_change_trigger_requires_different_fingerprint() -> None:
    """指纹不同才触发；没有基线指纹时不触发。"""
    changed = evaluate_triggers(
        TriggerState(dataset_fingerprint=DATA_B, stable_dataset_fingerprint=DATA_A)
    )
    assert TRIGGER_DATASET_CHANGE in changed.fired
    baseline_missing = evaluate_triggers(TriggerState(dataset_fingerprint=DATA_B))
    assert TRIGGER_DATASET_CHANGE not in baseline_missing.fired


def test_quality_drop_fires_below_lower_bound() -> None:
    """线上合格率低于下限触发；等于下限不触发。"""
    assert TRIGGER_QUALITY_DROP in evaluate_triggers(
        TriggerState(online_pass_rate=0.42)
    ).fired
    assert TRIGGER_QUALITY_DROP not in evaluate_triggers(
        TriggerState(online_pass_rate=0.5)
    ).fired


def test_missing_online_rate_does_not_trigger_or_veto() -> None:
    """没有观测值就不触发、也不否决，且理由里说明"缺失"这个语义。"""
    decision = evaluate_triggers(TriggerState(new_examples=1))
    quality = next(item for item in decision.triggers() if item.name == TRIGGER_QUALITY_DROP)
    assert quality.fired is False
    assert quality.actual is None
    assert "缺失不触发" in quality.reason


def test_cooldown_vetoes_within_the_window() -> None:
    """冷却期内的重训请求被否决（这是"再等等"，不是"不需要"）。"""
    decision = evaluate_triggers(TriggerState(new_examples=30, hours_since_last_train=2.0))
    assert decision.should_retrain is False
    assert decision.fired == (TRIGGER_DATA_GROWTH,)
    assert decision.vetoed_by == (VETO_COOLDOWN,)
    assert "再等等" in decision.notes


def test_cooldown_passes_exactly_at_the_boundary() -> None:
    """正好等于冷却期 → 不算"不足"，可以训练（闭区间边界逐条钉住）。"""
    decision = evaluate_triggers(
        TriggerState(new_examples=30, hours_since_last_train=24.0)
    )
    assert decision.vetoed_by == ()
    assert decision.should_retrain is True


def test_active_run_vetoes_concurrent_training() -> None:
    """已有训练在跑 → 否决（两个任务写同一份产物目录会互相覆盖）。"""
    decision = evaluate_triggers(
        TriggerState(new_examples=30, hours_since_last_train=48.0, active_runs=1)
    )
    assert decision.vetoed_by == (VETO_ACTIVE_RUN,)
    assert decision.should_retrain is False


def test_max_parallel_runs_two_allows_one_active_run() -> None:
    """把并发上限调到 2 之后，1 个在跑就不再否决。"""
    decision = evaluate_triggers(
        TriggerState(new_examples=30, hours_since_last_train=48.0, active_runs=1),
        TriggerPolicy(max_parallel_runs=2),
    )
    assert decision.should_retrain is True


def test_no_trigger_means_idle_with_a_pointer_to_what_to_look_at() -> None:
    """一条都没触发时，备注指向"该去看什么"，而不是建议调阈值。"""
    decision = evaluate_triggers(TriggerState(dataset_fingerprint=DATA_A))
    assert decision.should_retrain is False
    assert decision.fired == ()
    assert decision.vetoed_by == ()
    assert "调冷却期" in decision.notes


def test_evaluation_order_is_triggers_then_vetoes() -> None:
    """判定顺序固定为三触发器 → 两否决，且与 markdown 展示顺序一致."""
    decision = evaluate_triggers(TriggerState())
    assert [item.name for item in decision.evaluations] == [
        TRIGGER_DATA_GROWTH,
        TRIGGER_DATASET_CHANGE,
        TRIGGER_QUALITY_DROP,
        VETO_COOLDOWN,
        VETO_ACTIVE_RUN,
    ]
    kinds = [item.kind for item in decision.evaluations]
    assert kinds == [KIND_TRIGGER] * 3 + [KIND_VETO] * 2


def test_policy_thresholds_are_echoed_in_each_evaluation() -> None:
    """自定义策略的阈值必须出现在对应项的 ``threshold`` 里（报告可复核）。"""
    policy = TriggerPolicy(min_new_examples=3, cooldown_hours=1.0, min_pass_rate=0.7)
    decision = evaluate_triggers(
        TriggerState(new_examples=5, online_pass_rate=0.65, hours_since_last_train=0.5),
        policy,
    )
    thresholds = {item.name: item.threshold for item in decision.evaluations}
    assert thresholds[TRIGGER_DATA_GROWTH] == 3
    assert thresholds[TRIGGER_QUALITY_DROP] == 0.7
    assert thresholds[VETO_COOLDOWN] == 1.0
    assert decision.fired == (TRIGGER_DATA_GROWTH, TRIGGER_QUALITY_DROP)
    assert decision.vetoed_by == (VETO_COOLDOWN,)


# --------------------------------------------------------------------------- #
# trigger_table：文档与实现同源
# --------------------------------------------------------------------------- #


def test_trigger_table_has_five_rows_in_fixed_order() -> None:
    """条件表五行，顺序与判定顺序一致（否则读报告的人与改代码的人对不上）。"""
    table = trigger_table()
    assert [row["name"] for row in table] == [
        TRIGGER_DATA_GROWTH,
        TRIGGER_DATASET_CHANGE,
        TRIGGER_QUALITY_DROP,
        VETO_COOLDOWN,
        VETO_ACTIVE_RUN,
    ]
    assert [row["kind"] for row in table] == ["trigger"] * 3 + ["veto"] * 2


def test_trigger_table_thresholds_come_from_the_policy_object() -> None:
    """阈值列必须等于策略对象的值——"文档说 8、代码是 50"这类漂移会立刻变红。"""
    policy = TriggerPolicy(min_new_examples=50, min_pass_rate=0.9, cooldown_hours=6.0)
    table = {row["name"]: row for row in trigger_table(policy)}
    assert table[TRIGGER_DATA_GROWTH]["threshold"] == 50
    assert table[TRIGGER_QUALITY_DROP]["threshold"] == 0.9
    assert table[VETO_COOLDOWN]["threshold"] == 6.0
    assert table[VETO_ACTIVE_RUN]["threshold"] == policy.max_parallel_runs


def test_trigger_table_rows_are_descriptive() -> None:
    """每行都要有含义与复核时机（否则表只是一堆阈值）。"""
    for row in trigger_table():
        assert row["meaning"]
        assert row["reviewed_when"]
