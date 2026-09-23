"""day059 发布门禁测试（M5-D10）：绝对判定、方向语义、缺指标的两侧.

这个文件守的是三件最容易写反的事：

1. **退步判定的方向**。``delta = 候选 − 基线``，**退步是负值**，
   因此 ``max_regression=0.0`` 的含义是 ``delta >= 0``。
   实现时写反过一次（写成 ``delta <= 0``），表现是**所有改进都被拦下**——
   而"所有候选都被拒绝"看起来很像"门禁很严格"，不会有人立刻怀疑方向写反；
2. **缺指标落在哪一侧**。绝对判定里，合格率/体积/产物三项缺证据即
   **不通过**；退步与成本两项缺指标则**降级为告警**。把它们混在一起读，
   就会出现"因为没测延迟所以延迟门禁通过"这种事故；
3. **阻塞与告警分开**。``passed`` 只由阻塞项决定，而两类都要出现在报告里。
"""

from __future__ import annotations

import pytest

from smart_research_agent.mlops import (
    GATE_ADAPTER_SIZE,
    GATE_ARTIFACT_COMPLETENESS,
    GATE_COST,
    GATE_DATASET_TRACEABILITY,
    GATE_METRIC_NAMES,
    GATE_PASS_RATE,
    GATE_REGRESSION,
    MLOpsError,
    ReleaseGates,
    evaluate_gates,
    gate_table,
)

BASE_METRICS: dict = {
    "eval_pass_rate": 0.65,
    "eval_pass_rate_delta": 0.05,
    "adapter_mebibytes": 0.05,
    "dataset_fingerprint": "3f1b0c9d7e5a2468",
    "base_model": "Qwen3-8B",
}
ARTIFACTS = {"adapter": "a/adapter", "merged": "m/merged"}


def evaluate(metrics: dict | None = None, **kwargs):  # type: ignore[no-untyped-def]
    """用缺省策略跑一次门禁（``metrics=None`` 时用全项达标的那份）."""
    return evaluate_gates(
        BASE_METRICS if metrics is None else metrics,
        policy=kwargs.pop("policy", None),
        artifacts=kwargs.pop("artifacts", ARTIFACTS),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# ReleaseGates：构造期校验
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_pass_rate": 1.5}, "min_pass_rate"),
        ({"min_pass_rate": -0.1}, "min_pass_rate"),
        ({"max_regression": -0.01}, "max_regression"),
        ({"max_adapter_mebibytes": 0}, "max_adapter_mebibytes"),
        ({"max_adapter_mebibytes": -1.0}, "max_adapter_mebibytes"),
        ({"max_cost_per_1k_tokens": 0}, "max_cost_per_1k_tokens"),
    ],
)
def test_release_gates_validates_at_construction(kwargs: dict, match: str) -> None:
    """非法阈值在构造期拦下（负的 max_regression 会让"任何退步都放行"）。"""
    with pytest.raises(MLOpsError, match=match):
        ReleaseGates(**kwargs)


def test_release_gates_defaults_and_projection() -> None:
    """六个缺省值逐项钉住（它们进文档与端点，是"文档与实现同源"的一半）。"""
    payload = ReleaseGates().to_dict()
    assert payload == {
        "min_pass_rate": 0.5,
        "max_regression": 0.0,
        "max_adapter_mebibytes": 64.0,
        "max_cost_per_1k_tokens": 0.05,
        "require_traceability": True,
        "require_artifacts": True,
    }


# --------------------------------------------------------------------------- #
# 全项达标与单因素失败
# --------------------------------------------------------------------------- #


def test_all_checks_pass_on_a_clean_candidate() -> None:
    """全项达标 → ``passed=True``，六项都在报告里。"""
    report = evaluate()
    assert report.passed is True
    assert [item.name for item in report.checks] == [
        GATE_PASS_RATE,
        GATE_REGRESSION,
        GATE_ADAPTER_SIZE,
        GATE_DATASET_TRACEABILITY,
        GATE_ARTIFACT_COMPLETENESS,
        GATE_COST,
    ]
    assert report.blocking_failures == []


def test_positive_delta_passes_the_regression_gate() -> None:
    """**方向回归测试**：``+0.05`` 的提升必须通过 ``max_regression=0.0``.

    这一条是给一次真实的方向写反写的。反过来的话，本用例会失败，
    而其余用例全部照过——正是那种"只在某些输入下出错"的缺陷。
    """
    report = evaluate({**BASE_METRICS, "eval_pass_rate_delta": 0.05})
    check = report.check(GATE_REGRESSION)
    assert check.passed is True
    assert check.threshold == ">= -0.0"
    assert report.passed is True


def test_negative_delta_fails_the_regression_gate() -> None:
    """任何负的 delta 都被拦下（缺省 ``max_regression=0.0``）。"""
    report = evaluate({**BASE_METRICS, "eval_pass_rate_delta": -0.01})
    check = report.check(GATE_REGRESSION)
    assert check.passed is False
    assert report.passed is False
    assert [item.name for item in report.blocking_failures] == [GATE_REGRESSION]


def test_regression_tolerance_allows_a_known_small_drop() -> None:
    """显式给容忍度后，恰好落在边界上的退步放行（闭区间）。"""
    report = evaluate(
        {**BASE_METRICS, "eval_pass_rate_delta": -0.02},
        policy=ReleaseGates(max_regression=0.02),
    )
    assert report.check(GATE_REGRESSION).passed is True
    assert report.passed is True


def test_pass_rate_boundary_is_inclusive() -> None:
    """合格率正好等于下限 → 通过（闭区间逐条钉住）。"""
    assert evaluate({**BASE_METRICS, "eval_pass_rate": 0.5}).passed is True
    assert evaluate({**BASE_METRICS, "eval_pass_rate": 0.4999}).passed is False


def test_adapter_size_boundary_is_inclusive() -> None:
    """体积正好等于上限 → 通过。"""
    assert evaluate({**BASE_METRICS, "adapter_mebibytes": 64.0}).passed is True
    assert evaluate({**BASE_METRICS, "adapter_mebibytes": 64.01}).passed is False


def test_cost_gate_fires_when_the_metric_is_present_and_too_high() -> None:
    """有成本指标且超限时**是阻塞项**（缺指标才是告警）。"""
    report = evaluate({**BASE_METRICS, "cost_usd_per_1k_tokens": 0.09})
    check = report.check(GATE_COST)
    assert check.passed is False
    assert check.blocking is True
    assert report.passed is False


# --------------------------------------------------------------------------- #
# 缺指标：落在不通过那一侧还是告警那一侧
# --------------------------------------------------------------------------- #


def test_missing_pass_rate_is_a_blocking_failure() -> None:
    """绝对判定里「缺证据」就是不通过——与相对判定的"不知道"相反。"""
    metrics = {k: v for k, v in BASE_METRICS.items() if k != "eval_pass_rate"}
    report = evaluate(metrics)
    check = report.check(GATE_PASS_RATE)
    assert check.passed is False
    assert check.actual is None
    assert check.blocking is True
    assert report.passed is False


def test_missing_adapter_size_is_a_blocking_failure() -> None:
    """体积不可知就决定不了部署成本，因此不通过（而不是"跳过"）。"""
    metrics = {k: v for k, v in BASE_METRICS.items() if k != "adapter_mebibytes"}
    report = evaluate(metrics)
    assert report.check(GATE_ADAPTER_SIZE).passed is False


def test_missing_delta_is_only_a_warning() -> None:
    """没有基线（首次训练）时退步检查降级为"跳过"，不拦发布.

    它落在 ``skipped_checks`` 而不是 ``warnings``：**"没测"与"没达标"
    是两件事**——前者要回答"我们漏了一个证据"，后者要回答"这个值能不能接受"。
    """
    metrics = {k: v for k, v in BASE_METRICS.items() if k != "eval_pass_rate_delta"}
    report = evaluate(metrics)
    check = report.check(GATE_REGRESSION)
    assert check.passed is True
    assert check.blocking is False
    assert report.passed is True
    # 本轮指标里既没有 delta 也没有成本，因此两项都进 skipped（缺证据）
    assert [item.name for item in report.skipped_checks] == [GATE_REGRESSION, GATE_COST]
    assert report.warnings == []


def test_missing_cost_is_only_a_skip() -> None:
    """离线训练不产生推理成本，因此成本门禁降级为"跳过"。"""
    report = evaluate()
    check = report.check(GATE_COST)
    assert check.actual is None
    assert check.blocking is False
    assert [item.name for item in report.skipped_checks] == [GATE_COST]
    assert report.passed is True


def test_traceability_lists_every_missing_field() -> None:
    """溯源门禁把缺的字段逐个列出来（而不是只说"不通过"）。"""
    metrics = {k: v for k, v in BASE_METRICS.items() if k != "base_model"}
    report = evaluate(metrics)
    check = report.check(GATE_DATASET_TRACEABILITY)
    assert check.actual == "base_model"
    assert "缺少 base_model" in check.reason


def test_artifact_completeness_reports_the_missing_slot() -> None:
    """产物门禁报出缺哪一样（只有适配器、没有合并模型不算交付完成）。"""
    report = evaluate(artifacts={"adapter": "a/adapter"})
    check = report.check(GATE_ARTIFACT_COMPLETENESS)
    assert check.passed is False
    assert check.actual == "merged"
    assert report.passed is False


def test_traceability_and_artifacts_can_be_switched_off() -> None:
    """两个布尔开关关掉后，对应检查变成**告警**（非阻塞且未通过）.

    刻意不是"通过"：**关掉一项检查是一个值得被看见的决定**，
    而不是一个隐形的默认值。它落进 ``warnings``，因此会出现在
    CI 摘要与 markdown 报告的"告警（策略关闭）"一行。
    """
    metrics = {"eval_pass_rate": 0.6, "adapter_mebibytes": 1.0}
    report = evaluate(
        metrics,
        artifacts={},
        policy=ReleaseGates(require_traceability=False, require_artifacts=False),
    )
    assert report.check(GATE_DATASET_TRACEABILITY).blocking is False
    assert report.check(GATE_ARTIFACT_COMPLETENESS).blocking is False
    assert report.check(GATE_DATASET_TRACEABILITY).actual == "未检查"
    assert report.check(GATE_DATASET_TRACEABILITY).passed is False
    assert report.passed is True
    assert [item.name for item in report.warnings] == [
        GATE_DATASET_TRACEABILITY,
        GATE_ARTIFACT_COMPLETENESS,
    ]
    # 被"关闭"的两项进 warnings；缺证据的那两项（delta / cost）进 skipped
    assert [item.name for item in report.skipped_checks] == [GATE_REGRESSION, GATE_COST]


# --------------------------------------------------------------------------- #
# 报告投影
# --------------------------------------------------------------------------- #


def test_report_projection_and_markdown() -> None:
    """投影里带上阻塞失败与告警的名单，markdown 逐项列表。"""
    report = evaluate({**BASE_METRICS, "eval_pass_rate": 0.3}, commit="deadbeef")
    payload = report.to_dict()
    assert payload["passed"] is False
    assert payload["commit"] == "deadbeef"
    assert payload["blocking_failures"] == [GATE_PASS_RATE]
    assert len(payload["checks"]) == 6
    markdown = report.render_markdown()
    assert "发布门禁：不通过" in markdown
    assert "deadbeef" in markdown
    assert GATE_PASS_RATE in markdown


def test_report_summary_counts_checks() -> None:
    """摘要里给出"N/6 项通过"与阻塞失败名单。"""
    report = evaluate({**BASE_METRICS, "eval_pass_rate": 0.3})
    line = report.summary_line()
    assert "5/6 项通过" in line
    assert GATE_PASS_RATE in line


def test_report_check_rejects_unknown_name() -> None:
    """按名字取不存在的检查项 → 报错（而不是返回 None）。"""
    with pytest.raises(MLOpsError, match="没有门禁项"):
        evaluate().check("latency")


def test_gate_check_summary_marks_blocking() -> None:
    """一条检查的摘要写出"通过/不通过"与"阻塞/告警"。"""
    check = evaluate().check(GATE_COST)
    assert "告警" in check.summary_line()
    assert evaluate().check(GATE_PASS_RATE).summary_line().startswith("[通过/阻塞]")


# --------------------------------------------------------------------------- #
# gate_table：文档与实现同源
# --------------------------------------------------------------------------- #


def test_gate_table_matches_the_policy_object() -> None:
    """阈值列必须来自 ``ReleaseGates`` 实例——"文档说 64、代码是 16"会立刻变红。"""
    policy = ReleaseGates(
        min_pass_rate=0.7, max_regression=0.05, max_adapter_mebibytes=16.0,
        max_cost_per_1k_tokens=0.01,
    )
    table = {row["name"]: row for row in gate_table(policy)}
    assert table[GATE_PASS_RATE]["threshold"] == 0.7
    assert table[GATE_REGRESSION]["threshold"] == 0.05
    assert table[GATE_ADAPTER_SIZE]["threshold"] == 16.0
    assert table[GATE_COST]["threshold"] == 0.01
    assert table[GATE_PASS_RATE]["metric"] == GATE_METRIC_NAMES[GATE_PASS_RATE]


def test_gate_table_covers_all_six_checks_in_order() -> None:
    """六行、顺序与 ``evaluate_gates`` 的判定顺序一致。"""
    rows = gate_table()
    assert [row["name"] for row in rows] == [
        GATE_PASS_RATE,
        GATE_REGRESSION,
        GATE_ADAPTER_SIZE,
        GATE_DATASET_TRACEABILITY,
        GATE_ARTIFACT_COMPLETENESS,
        GATE_COST,
    ]


def test_gate_table_documents_missing_behaviour() -> None:
    """``when_missing`` 一列是这张表里最容易被忽略、也最容易出事的一列."""
    table = {row["name"]: row for row in gate_table()}
    assert "不通过" in table[GATE_PASS_RATE]["when_missing"]
    assert "告警" in table[GATE_REGRESSION]["when_missing"]
    assert "告警" in table[GATE_COST]["when_missing"]
    assert all(row["when_missing"] and row["meaning"] for row in gate_table())


def test_gate_table_reflects_disabled_switches() -> None:
    """关掉溯源与产物检查后，表里的阈值列写成"不检查"且阻塞列变 False."""
    table = {
        row["name"]: row
        for row in gate_table(ReleaseGates(require_traceability=False, require_artifacts=False))
    }
    assert table[GATE_DATASET_TRACEABILITY]["threshold"] == "不检查"
    assert table[GATE_DATASET_TRACEABILITY]["blocking"] is False
    assert table[GATE_ARTIFACT_COMPLETENESS]["threshold"] == "不检查"


def test_gate_metric_names_are_frozen_contract() -> None:
    """门禁**只认被点名的指标**——新增指标不会悄悄改变门禁行为。"""
    assert GATE_METRIC_NAMES == {
        GATE_PASS_RATE: "eval_pass_rate",
        GATE_REGRESSION: "eval_pass_rate_delta",
        GATE_ADAPTER_SIZE: "adapter_mebibytes",
        GATE_COST: "cost_usd_per_1k_tokens",
    }


def test_extra_metrics_are_ignored_by_the_gate() -> None:
    """无关指标原样忽略（它们进追踪器，不进判定）。"""
    report = evaluate({**BASE_METRICS, "latency_ms": 99999.0, "notes": "x"})
    assert report.passed is True
    assert len(report.checks) == 6
