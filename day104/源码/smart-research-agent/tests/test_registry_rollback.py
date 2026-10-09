"""day058 采纳与回滚测试（M5-D9）：拒绝候选、采纳候选、回滚三件事分开.

这个文件守的是本课最重要的一条纪律：**"拒绝候选"与"回滚"是两个动作**
（触发者、对象、后果都不同）。混在一起会让"候选没通过评估"变成
"把线上版本也退掉"，后者的代价远大于前者。

四组断言尤其值得说明：

1. **首次上线必须走单独分支**：没有任何 stable 版本时增益无从计算，
   若与"有当前版本"共用一条路径，第一版**永远上不了线**；
2. **不可比时拒绝算差值**：把"换了数据的候选"与当前版本的分数相减，
   得到的数字会被下游当成提升量使用——实测那个差值是 -0.1600，
   而它既不是提升也不是退步；
3. **回滚第一步必须是 ``verify``**：先冻结当前版本、再发现目标不可部署，
   结果是"生产上没有任何可服务版本"。这条顺序错误只在最坏情况下暴露，
   所以它被钉成一条对 ``steps`` 顺序的断言；
4. **``hold`` 计划不可执行**：`apply_rollback` 在唯一有副作用的入口拦一次
   ——一个 hold 计划被误执行，等于在没有目标的情况下把生产版本退掉。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smart_research_agent.registry import (
    CHECK_ABSOLUTE,
    CHECK_AGE,
    CHECK_COMPARABLE,
    CHECK_DEPLOYABLE,
    CHECK_GAIN,
    CHECK_METRICS,
    STAGE_CANDIDATE,
    STAGE_ROLLED_BACK,
    STAGE_STABLE,
    ACTION_HOLD,
    ACTION_PROMOTE,
    ACTION_ROLLBACK,
    CheckResult,
    ModelRegistry,
    ModelVersion,
    PromotionDecision,
    PromotionPolicy,
    RegistryError,
    RollbackPlan,
    RollbackStep,
    VersionTriple,
    apply_rollback,
    candidate_age_hours,
    evaluate_candidate,
    plan_rollback,
)
from smart_research_agent.registry.rollback import (
    DEFAULT_OBSERVE_WINDOW_HOURS,
    STEP_FREEZE,
    STEP_OBSERVE,
    STEP_RECORD,
    STEP_VERIFY,
)

BASE = "Qwen3-8B"
BIG = "Qwen3-14B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"
NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)


def hexdigest64(seed: str) -> str:
    """由任意字符串派生 64 位十六进制串（确定性适配器哈希）."""
    raw = "".join(f"{byte:02x}" for byte in seed.encode("utf-8"))
    return (raw * 8)[:64]


def make(
    version: str,
    *,
    adapter: str | None = None,
    dataset: str = DATA_A,
    base: str = BASE,
    parent: str = "",
    stage: str = STAGE_CANDIDATE,
    rating: float | None = 0.6,
    age_hours: float = 0.0,
    artifacts: dict[str, str] | None = None,
    created_at: str | None = None,
) -> ModelVersion:
    """构造一条版本记录（``age_hours`` 用来把创建时间往前推）."""
    return ModelVersion(
        triple=VersionTriple(
            base_model=base,
            adapter_sha256=adapter or hexdigest64(version),
            dataset_fingerprint=dataset,
        ),
        version=version,
        parent_version=parent,
        stage=stage,
        created_at=(
            created_at
            if created_at is not None
            else (NOW - timedelta(hours=age_hours)).isoformat()
        ),
        metrics={} if rating is None else {"eval_pass_rate": rating},
        artifacts=(
            {"adapter": f"a/{version}", "merged": f"m/{version}"}
            if artifacts is None
            else artifacts
        ),
    )


CURRENT = make("1.0.0", stage=STAGE_STABLE, rating=0.60)


# --------------------------------------------------------------------------- #
# PromotionPolicy：阈值本身就是"什么算更好"的定义
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_gain": -0.1}, "min_gain"),
        ({"regression_tolerance": -0.1}, "regression_tolerance"),
        ({"absolute_min_pass_rate": 1.5}, "absolute_min_pass_rate"),
        ({"absolute_min_pass_rate": -0.1}, "absolute_min_pass_rate"),
        ({"max_candidate_age_hours": 0.0}, "max_candidate_age_hours"),
        ({"max_candidate_age_hours": -1.0}, "max_candidate_age_hours"),
    ],
)
def test_promotion_policy_validates_at_construction(kwargs: dict, match: str) -> None:
    """非法阈值在构造期拦下：负的 min_gain 会让"任何候选都值得采纳"。"""
    with pytest.raises(RegistryError, match=match):
        PromotionPolicy(**kwargs)


def test_promotion_policy_allows_none_age_limit() -> None:
    """``max_candidate_age_hours=None`` 表示不做时效检查（显式关闭）。"""
    assert PromotionPolicy(max_candidate_age_hours=None).max_candidate_age_hours is None


def test_promotion_policy_to_dict_has_all_six_fields() -> None:
    """策略要能被完整落进报告——一个没有写出阈值的判定无法被复核。"""
    payload = PromotionPolicy().to_dict()
    assert set(payload) == {
        "min_gain",
        "regression_tolerance",
        "absolute_min_pass_rate",
        "max_candidate_age_hours",
        "require_comparable",
        "require_deployable",
    }


# --------------------------------------------------------------------------- #
# CheckResult / PromotionDecision 投影
# --------------------------------------------------------------------------- #


def test_check_result_projects_and_prints() -> None:
    """一条检查的四个字段都要出现在报告里（只有 passed 的报告不可用）。"""
    check = CheckResult(
        name=CHECK_GAIN, passed=True, actual=0.12, threshold=0.02, reason="增益达标"
    )
    assert check.to_dict() == {
        "name": CHECK_GAIN,
        "passed": True,
        "actual": 0.12,
        "threshold": 0.02,
        "reason": "增益达标",
    }
    line = check.summary_line()
    assert "通过" in line and CHECK_GAIN in line and "0.12" in line


def test_promotion_decision_rejects_unknown_action() -> None:
    """动作名写错立刻报错（三个动作是封闭集合）。"""
    with pytest.raises(RegistryError, match="未知动作"):
        PromotionDecision(
            action="switch",
            reason="",
            current_version="1.0.0",
            candidate_version="1.0.1",
            target_version="1.0.1",
        )


def test_promotion_decision_properties_and_projection() -> None:
    """``promoted`` / ``failed_checks`` / ``to_dict`` 三个读取口都要对得上。"""
    checks = (
        CheckResult(name=CHECK_DEPLOYABLE, passed=True, actual="齐全", threshold="", reason="ok"),
        CheckResult(name=CHECK_GAIN, passed=False, actual=0.01, threshold=0.02, reason="死区"),
    )
    decision = PromotionDecision(
        action=ACTION_HOLD,
        reason="增益不足",
        current_version="1.0.0",
        candidate_version="1.0.1",
        target_version="1.0.0",
        gain=0.01,
        checks=checks,
    )
    assert not decision.promoted
    assert [item.name for item in decision.failed_checks()] == [CHECK_GAIN]
    payload = decision.to_dict()
    assert payload["action"] == ACTION_HOLD
    assert payload["promoted"] is False
    assert payload["gain"] == 0.01
    assert len(payload["checks"]) == 2
    assert "增益不足" in decision.summary_line()


def test_promotion_decision_summary_and_markdown() -> None:
    """摘要里"不可比"不写成数字；markdown 表逐项列出检查。"""
    decision = PromotionDecision(
        action=ACTION_PROMOTE,
        reason="首次上线",
        current_version="",
        candidate_version="1.0.0",
        target_version="1.0.0",
        checks=(
            CheckResult(
                name=CHECK_ABSOLUTE, passed=True, actual=0.58, threshold=0.5, reason="达标"
            ),
        ),
    )
    assert "不可比" in decision.summary_line()
    markdown = decision.render_markdown()
    assert "首次上线" in markdown
    assert CHECK_ABSOLUTE in markdown


# --------------------------------------------------------------------------- #
# candidate_age_hours
# --------------------------------------------------------------------------- #


def test_candidate_age_hours_computes_positive_age() -> None:
    """正常时间戳：年龄为正。"""
    age = candidate_age_hours(make("1.0.0", age_hours=48.0), now=NOW)
    assert age == pytest.approx(48.0)


def test_candidate_age_hours_returns_none_for_missing_or_broken_timestamp() -> None:
    """缺失/不可解析返回 None（"时间戳解析失败"与"刚好 0 小时"必须能区分）."""
    assert candidate_age_hours(make("1.0.0", created_at=""), now=NOW) is None
    assert candidate_age_hours(make("1.0.0", created_at="not-a-date"), now=NOW) is None


def test_candidate_age_hours_normalizes_zulu_and_naive_timestamps() -> None:
    """``Z`` 结尾与无时区的时间戳都按 UTC 处理。"""
    assert candidate_age_hours(make("1.0.0", created_at="2026-09-15T12:00:00Z"), now=NOW) == (
        pytest.approx(24.0)
    )
    assert candidate_age_hours(
        make("1.0.0", created_at="2026-09-15T12:00:00"), now=NOW
    ) == pytest.approx(24.0)


def test_candidate_age_hours_accepts_naive_reference() -> None:
    """参照时间无时区时按 UTC 补齐（否则会抛 TypeError）。"""
    naive = datetime(2026, 9, 16, 12, 0, 0)
    assert candidate_age_hours(make("1.0.0", age_hours=6.0), now=naive) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# evaluate_candidate：首次上线分支
# --------------------------------------------------------------------------- #


def test_first_launch_promotes_when_above_absolute_threshold() -> None:
    """还有没 stable 版本时只做绝对门槛判定：0.58 >= 0.5 → 采纳。"""
    decision = evaluate_candidate(None, make("1.0.0", rating=0.58), now=NOW)
    assert decision.action == ACTION_PROMOTE
    assert decision.gain is None
    assert decision.comparable is False
    assert decision.target_version == "1.0.0"
    assert CHECK_ABSOLUTE in [item.name for item in decision.checks]


def test_first_launch_holds_below_absolute_threshold() -> None:
    """首次上线也必须有门槛：0.42 < 0.5 → 不采纳。"""
    decision = evaluate_candidate(None, make("1.0.0", rating=0.42), now=NOW)
    assert decision.action == ACTION_HOLD
    assert decision.target_version == ""
    assert "首次上线被拦下" in decision.reason


def test_first_launch_holds_without_metrics() -> None:
    """没有评估分数就不能上线（缺失与 0 是两件事）。"""
    decision = evaluate_candidate(None, make("1.0.0", rating=None), now=NOW)
    assert decision.action == ACTION_HOLD
    assert [item.name for item in decision.failed_checks()] == [CHECK_ABSOLUTE]


# --------------------------------------------------------------------------- #
# evaluate_candidate：可比分支
# --------------------------------------------------------------------------- #


def test_comparable_candidate_with_enough_gain_is_promoted() -> None:
    """增益 +0.12 >= 0.02 → 采纳，target 指向候选。"""
    decision = evaluate_candidate(CURRENT, make("1.0.1", parent="1.0.0", rating=0.72), now=NOW)
    assert decision.action == ACTION_PROMOTE
    assert decision.gain == pytest.approx(0.12)
    assert decision.comparable is True
    assert decision.target_version == "1.0.1"
    assert [item.name for item in decision.checks] == [
        CHECK_DEPLOYABLE,
        CHECK_AGE,
        CHECK_COMPARABLE,
        CHECK_METRICS,
        CHECK_GAIN,
    ]


def test_gain_inside_dead_band_is_held_with_explanation() -> None:
    """+0.01 落在死区：评估方差的量级内不值得切换生产版本。"""
    decision = evaluate_candidate(CURRENT, make("1.0.1", parent="1.0.0", rating=0.61), now=NOW)
    assert decision.action == ACTION_HOLD
    assert decision.gain == pytest.approx(0.01)
    assert "死区" in decision.reason


def test_regression_is_held_by_the_gain_check() -> None:
    """退步 -0.08：``CHECK_GAIN`` 直接不通过（缺省不允许任何退步）。"""
    decision = evaluate_candidate(CURRENT, make("1.0.1", parent="1.0.0", rating=0.52), now=NOW)
    assert decision.action == ACTION_HOLD
    assert [item.name for item in decision.failed_checks()] == [CHECK_GAIN]


def test_regression_tolerance_allows_small_drop_but_not_switch() -> None:
    """容忍 -0.03 时差检查通过，但增益仍不达门槛 → 仍然 hold。"""
    decision = evaluate_candidate(
        CURRENT,
        make("1.0.1", parent="1.0.0", rating=0.57),
        policy=PromotionPolicy(regression_tolerance=0.05),
        now=NOW,
    )
    assert decision.action == ACTION_HOLD
    assert decision.failed_checks() == []
    assert "死区" in decision.reason


def test_missing_metrics_on_either_side_holds() -> None:
    """两边都要有分数：候选缺 → hold，当前缺 → 也 hold。"""
    candidate_missing = evaluate_candidate(
        CURRENT, make("1.0.1", parent="1.0.0", rating=None), now=NOW
    )
    assert candidate_missing.action == ACTION_HOLD
    assert CHECK_METRICS in [item.name for item in candidate_missing.failed_checks()]

    current_missing = evaluate_candidate(
        make("1.0.0", stage=STAGE_STABLE, rating=None),
        make("1.0.1", parent="1.0.0", rating=0.72),
        now=NOW,
    )
    assert current_missing.action == ACTION_HOLD
    assert current_missing.gain is None


def test_stale_candidate_is_held_but_gain_still_reported() -> None:
    """超龄候选不采纳；增益照算照报（报告要能解释"为什么好却没换"）."""
    decision = evaluate_candidate(
        CURRENT, make("1.0.1", parent="1.0.0", rating=0.80, age_hours=504.0), now=NOW
    )
    assert decision.action == ACTION_HOLD
    assert decision.gain == pytest.approx(0.20)
    assert [item.name for item in decision.failed_checks()] == [CHECK_AGE]
    assert "504" in str(decision.failed_checks()[0].actual)


def test_unparsable_timestamp_skips_age_check() -> None:
    """创建时间不可解析 → 跳过时效检查（actual 为 None，而不是伪造一个 0）。"""
    decision = evaluate_candidate(
        CURRENT, make("1.0.1", parent="1.0.0", rating=0.72, created_at="bad"), now=NOW
    )
    age = next(item for item in decision.checks if item.name == CHECK_AGE)
    assert age.passed is True
    assert age.actual is None
    assert decision.action == ACTION_PROMOTE


def test_age_check_can_be_disabled() -> None:
    """``max_candidate_age_hours=None`` 时不做时效检查（阈值字段也如实写出来）。"""
    decision = evaluate_candidate(
        CURRENT,
        make("1.0.1", parent="1.0.0", rating=0.72, age_hours=9999.0),
        policy=PromotionPolicy(max_candidate_age_hours=None),
        now=NOW,
    )
    age = next(item for item in decision.checks if item.name == CHECK_AGE)
    assert age.threshold is None
    assert decision.action == ACTION_PROMOTE


def test_missing_required_artifact_blocks_promotion() -> None:
    """缺合并产物的候选不能上线——哪怕它的分数最高。"""
    decision = evaluate_candidate(
        CURRENT,
        make("1.0.1", parent="1.0.0", rating=0.90, artifacts={"adapter": "a/1.0.1"}),
        now=NOW,
    )
    assert decision.action == ACTION_HOLD
    check = next(item for item in decision.checks if item.name == CHECK_DEPLOYABLE)
    assert check.actual == "merged"
    assert "缺少必需产物 merged" in check.reason


def test_require_deployable_false_skips_the_artifact_check() -> None:
    """显式关闭可部署性检查后，缺产物也能采纳（阈值由调用方负责）。"""
    decision = evaluate_candidate(
        CURRENT,
        make("1.0.1", parent="1.0.0", rating=0.90, artifacts={}),
        policy=PromotionPolicy(require_deployable=False),
        now=NOW,
    )
    assert decision.action == ACTION_PROMOTE
    check = next(item for item in decision.checks if item.name == CHECK_DEPLOYABLE)
    assert check.threshold == "不检查"


# --------------------------------------------------------------------------- #
# evaluate_candidate：不可比分支
# --------------------------------------------------------------------------- #


def test_incomparable_candidate_uses_absolute_threshold_and_reports_no_gain() -> None:
    """换了数据集：不算差值，改用绝对门槛（0.72 >= 0.5 → 采纳，gain 为 None）。"""
    decision = evaluate_candidate(
        CURRENT, make("1.1.0", parent="1.0.0", dataset=DATA_B, rating=0.72), now=NOW
    )
    assert decision.action == ACTION_PROMOTE
    assert decision.gain is None
    assert decision.comparable is False
    assert CHECK_ABSOLUTE in [item.name for item in decision.checks]
    assert CHECK_GAIN not in [item.name for item in decision.checks]


def test_incomparable_candidate_below_absolute_threshold_is_held() -> None:
    """不可比且未达绝对门槛 → hold，理由写明"不算差值"。"""
    decision = evaluate_candidate(
        CURRENT, make("1.1.0", parent="1.0.0", dataset=DATA_B, rating=0.44), now=NOW
    )
    assert decision.action == ACTION_HOLD
    assert decision.gain is None
    assert "不可比且未达绝对门槛" in decision.reason


def test_incomparable_with_base_change_is_also_absolute() -> None:
    """换基座同样不可比（旧适配器挂不上新基座，分数不在一个尺度上）。"""
    decision = evaluate_candidate(
        CURRENT, make("2.0.0", parent="1.0.0", base=BIG, rating=0.66), now=NOW
    )
    assert decision.action == ACTION_PROMOTE
    assert decision.gain is None
    assert decision.comparable is False


def test_disabling_comparability_computes_a_useless_difference() -> None:
    """关掉可比性要求后，那个跨分布的差值真的会被算出来——本课的反面教材.

    实测：当前 0.6000、候选 0.4400，得到的 -0.1600 既不是提升也不是退步，
    而是两种评估分布之差。它被算出来之后会被下游当成提升量使用。
    """
    decision = evaluate_candidate(
        CURRENT,
        make("1.1.0", parent="1.0.0", dataset=DATA_B, rating=0.44),
        policy=PromotionPolicy(require_comparable=False),
        now=NOW,
    )
    assert decision.gain == pytest.approx(-0.16)
    assert decision.comparable is False
    assert decision.action == ACTION_HOLD
    gain_check = next(item for item in decision.checks if item.name == CHECK_GAIN)
    assert "不可比" in str(gain_check.threshold)
    assert "不可作为提升证据" in gain_check.reason


# --------------------------------------------------------------------------- #
# RollbackStep / RollbackPlan 投影
# --------------------------------------------------------------------------- #


def test_rollback_step_projects_and_prints() -> None:
    """一步的四个字段都要可投影（含"是否阻塞"）。"""
    step = RollbackStep(order=1, action=STEP_VERIFY, detail="校验目标产物", blocking=True)
    assert step.to_dict() == {
        "order": 1,
        "action": STEP_VERIFY,
        "detail": "校验目标产物",
        "blocking": True,
    }
    assert "阻塞" in step.summary_line()


def test_rollback_plan_rejects_promote_action() -> None:
    """回滚计划的动作只能是 rollback / hold（promote 走采纳路径）。"""
    with pytest.raises(RegistryError, match="回滚计划的动作"):
        RollbackPlan(action=ACTION_PROMOTE, reason="", version="1.0.0")


def test_rollback_plan_without_steps_is_not_executable() -> None:
    """没有步的计划不可执行——"没有退路"是一个正式结论。"""
    plan = RollbackPlan(action=ACTION_HOLD, reason="无目标", version="1.0.0")
    assert plan.should_execute is False
    assert plan.blocking_steps() == []
    assert "无可执行动作" in plan.render_markdown()


def test_rollback_plan_projection_counts_blocking_steps() -> None:
    """投影里带上阻塞步数（执行器据此判断"哪一步失败必须中止"）。"""
    plan = RollbackPlan(
        action=ACTION_ROLLBACK,
        reason="事故",
        version="1.0.2",
        target_version="1.0.1",
        target_key="abc",
        steps=(
            RollbackStep(order=1, action=STEP_VERIFY, detail="v", blocking=True),
            RollbackStep(order=2, action=STEP_RECORD, detail="r", blocking=False),
        ),
    )
    assert plan.should_execute is True
    assert len(plan.blocking_steps()) == 1
    payload = plan.to_dict()
    assert payload["should_execute"] is True
    assert payload["target_key"] == "abc"
    assert len(payload["steps"]) == 2
    assert "1.0.2" in plan.summary_line()
    markdown = plan.render_markdown()
    assert "回滚计划" in markdown and "1.0.1" in markdown


# --------------------------------------------------------------------------- #
# plan_rollback
# --------------------------------------------------------------------------- #


def build(*specs: dict) -> ModelRegistry:
    """按顺序登记一批版本记录（父在前，避免悬空父指针）."""
    registry = ModelRegistry()
    for spec in specs:
        registry.register(make(**spec))
    return registry


V1 = dict(version="1.0.0", stage=STAGE_STABLE, rating=0.60)
V2 = dict(version="1.0.1", parent="1.0.0", stage=STAGE_STABLE, rating=0.68)
V3 = dict(version="1.0.2", parent="1.0.1", stage=STAGE_STABLE, rating=0.71)


def test_plan_rollback_builds_four_steps_with_verify_first() -> None:
    """回滚四步，且 ``verify`` 必须排在 ``freeze`` 之前.

    顺序反过来也能通过绝大多数测试，代价只在最坏情况下暴露：
    先冻结当前版本、再发现目标不可部署，结果是**生产上没有任何可服务版本**。
    """
    registry = build(V1, V2, V3)
    plan = plan_rollback(registry, "1.0.2", reason="上线后拒答率异常升高")
    assert plan.action == ACTION_ROLLBACK
    assert plan.target_version == "1.0.1"
    assert [step.action for step in plan.steps] == [STEP_VERIFY, STEP_FREEZE, STEP_RECORD, STEP_OBSERVE]
    assert [step.blocking for step in plan.steps] == [True, True, False, False]
    assert plan.observe_window_hours == DEFAULT_OBSERVE_WINDOW_HOURS
    assert "上线后拒答率异常升高" in plan.reason


def test_plan_rollback_rejects_non_stable_version() -> None:
    """候选退回候选不需要任何动作——回滚只针对正在服务的版本。"""
    registry = build(V1, dict(version="1.0.1", parent="1.0.0"))
    plan = plan_rollback(registry, "1.0.1")
    assert plan.action == ACTION_HOLD
    assert "不是 stable" in plan.reason


def test_plan_rollback_holds_when_no_ancestor_exists() -> None:
    """首个 stable 版本没有退路（这是真实存在的状态，必须显式输出）。"""
    registry = build(V1)
    plan = plan_rollback(registry, "1.0.0")
    assert plan.action == ACTION_HOLD
    assert plan.steps == ()
    assert "没有可部署的 stable 祖先" in plan.reason


def test_plan_rollback_skips_undeployable_ancestor() -> None:
    """最近的祖先不可部署时自动往前找（stable 记录可能只是"当年提升过"）。"""
    registry = build(
        V1,
        dict(
            version="1.0.1",
            parent="1.0.0",
            stage=STAGE_STABLE,
            rating=0.68,
            artifacts={"adapter": "a/1.0.1"},
        ),
        V3,
    )
    plan = plan_rollback(registry, "1.0.2", reason="产物被清理后的回滚演练")
    assert plan.action == ACTION_ROLLBACK
    assert plan.target_version == "1.0.0"
    assert len(plan.target_key) == 16


def test_plan_rollback_holds_when_all_ancestors_undeployable() -> None:
    """祖先一个都不可部署 → 明确报告"没有退路"，而不是给一个必然失败的计划。"""
    registry = build(
        dict(version="1.0.0", stage=STAGE_STABLE, rating=0.60, artifacts={}),
        dict(
            version="1.0.1",
            parent="1.0.0",
            stage=STAGE_STABLE,
            rating=0.68,
            artifacts={"merged": "m/1.0.1"},
        ),
        V3,
    )
    plan = plan_rollback(registry, "1.0.2")
    assert plan.action == ACTION_HOLD
    assert "可部署 0 个" in plan.reason


def test_plan_rollback_validates_observe_window() -> None:
    """观察窗口必须为正数（0 会让"回滚完成"没有时间点）。"""
    registry = build(V1, V2, V3)
    with pytest.raises(RegistryError, match="observe_window_hours"):
        plan_rollback(registry, "1.0.2", observe_window_hours=0.0)


def test_plan_rollback_propagates_unknown_version() -> None:
    """版本不在表里 → ``RegistryError`` 原样抛出（路由层转成 400）。"""
    registry = build(V1)
    with pytest.raises(RegistryError, match="不存在"):
        plan_rollback(registry, "9.9.9")


# --------------------------------------------------------------------------- #
# apply_rollback
# --------------------------------------------------------------------------- #


def test_apply_rollback_freezes_version_and_head_falls_back() -> None:
    """执行回滚：出问题的版本进 rolled_back，``head()`` 自动落回上一个 stable."""
    registry = build(V1, V2, V3)
    plan = plan_rollback(registry, "1.0.2")
    updated = apply_rollback(registry, plan)
    assert updated.stage == STAGE_ROLLED_BACK
    assert registry.head().version == "1.0.1"
    assert "回滚到 v1.0.1" in updated.notes


def test_apply_rollback_records_a_stage_event(tmp_path: Path) -> None:
    """回滚必须进审计：事件流里要多出一条 stage 事件。"""
    index = tmp_path / "versions.jsonl"
    registry = ModelRegistry(index)
    for spec in (V1, V2, V3):
        registry.register(make(**spec))
    before = len(registry.events())
    apply_rollback(registry, plan_rollback(registry, "1.0.2"))
    assert len(registry.events()) == before + 1
    assert registry.events()[-1]["stage"] == STAGE_ROLLED_BACK
    assert len(ModelRegistry(index).events()) == before + 1


def test_apply_rollback_rejects_hold_plan() -> None:
    """``hold`` 计划被误执行等于"在没有目标的情况下把生产版本退掉"——必须拦下。"""
    registry = build(V1)
    plan = plan_rollback(registry, "1.0.0")
    with pytest.raises(RegistryError, match="计划不可执行"):
        apply_rollback(registry, plan)
