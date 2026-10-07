"""day058 版本管理端点测试（M5-D9）：/registry/* 五个端点.

全部走 TestClient（进程内 ASGI 调用）：不起真实服务、不联网、**不写盘**、
不用随机数。五个端点都是只读或纯计算，因此同一个请求体的响应逐字节可复现。

这个文件挑的是四件"会静默失效"的事来钉：

1. **对照表必须与代码同源**：状态迁移表 == ``ALLOWED_TRANSITIONS``、
   重训六步 == ``PLAN_STEPS``、条件表 == ``trigger_table()``。
   文档与代码各写一份，漂移时没人会发现；
2. **``head`` 无 stable 时为 null**：不退回 candidate——"还没有生产版本"
   与"生产版本是某个候选"是两件事，后者被静默当成前者会让未验证的候选
   出现在部署脚本的视野里；
3. **非法输入必须响亮**：非法版本号/哈希/阶段/悬空父版本 → 400，
   非法策略值 → 400，Pydantic 层的负增量 → 422（"请求形状错了"与
   "业务规则不允许"是两种错误码）；
4. **版本表的写入不进 HTTP**：请求体带全量版本记录，端点只做折叠与判定，
   因而可以在任何机器上重放（多副本部署下"谁先写"无法保证）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.registry import (
    ALLOWED_TRANSITIONS,
    PLAN_STEPS,
    STAGES,
    ModelRegistry,
    ModelVersion,
    VersionTriple,
)

BASE = "Qwen3-8B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"
ADAPTER_A = "a" * 64
ADAPTER_B = "b" * 64

#: 候选版本的"新鲜"时间戳：**相对当前时刻**算，而不是写死一个日期（day072 修正）.
#: 写死日期的测试有一个**保质期**：``retrain_max_candidate_age_hours`` 缺省 168 小时，
#: 于是从那个日期起第 8 天开始，所有"应当 promote"的用例都会变成 hold——
#: 而它们报的是"候选年龄 185.56h 超过 168.0h 上限"，看起来像产品代码出了问题，
#: 实际是这条常量的保质期到了。改成"一小时前"之后，它与运行日期无关。
FRESH = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


@pytest.fixture
def client() -> TestClient:
    """注入 MockLLM 的离线测试客户端（端点本身不碰 LLM，注入只是为了不起真实客户端）."""
    return TestClient(create_app(llm=MockLLM(default="offline")))


def version_payload(
    version: str,
    *,
    adapter: str = ADAPTER_A,
    dataset: str = DATA_A,
    stage: str = "candidate",
    parent: str = "",
    rating: float | None = 0.6,
    created_at: str = FRESH,
    artifacts: dict | None = None,
) -> dict:
    """构造一条版本记录的请求体（默认产物齐全）."""
    payload = {
        "version": version,
        "base_model": BASE,
        "adapter_sha256": adapter,
        "dataset_fingerprint": dataset,
        "stage": stage,
        "created_at": created_at,
        "artifacts": (
            {"adapter": f"a/{version}", "merged": f"m/{version}"}
            if artifacts is None
            else artifacts
        ),
        "metrics": {} if rating is None else {"eval_pass_rate": rating},
    }
    if parent:
        payload["parent_version"] = parent
    return payload


STABLE = version_payload("1.0.0", stage="stable", rating=0.60)
CANDIDATE = version_payload("1.0.1", adapter=ADAPTER_B, parent="1.0.0", rating=0.72)


# --------------------------------------------------------------------------- #
# GET /registry/layout
# --------------------------------------------------------------------------- #


def test_layout_reports_self_description(client: TestClient) -> None:
    """布局端点回答四个开工前的问题：身份、递增、流转、步骤."""
    response = client.get("/registry/layout")
    assert response.status_code == 200
    payload = response.json()
    assert payload["version_key_length"] == 16
    assert payload["artifact_names"] == ["adapter", "merged", "dataset"]
    assert payload["deploy_required_artifacts"] == ["adapter", "merged"]
    assert payload["bump_kinds"] == ["major", "minor", "patch"]


def test_layout_transitions_come_from_the_code_table(client: TestClient) -> None:
    """迁移表必须等于 ``ALLOWED_TRANSITIONS``——"文档说能退回候选、代码里不行"会立刻变红."""
    payload = client.get("/registry/layout").json()
    assert payload["transitions"] == {
        name: list(targets) for name, targets in ALLOWED_TRANSITIONS.items()
    }
    assert payload["transitions"]["stable"] == ["archived", "rolled_back"]
    assert payload["transitions"]["archived"] == []


def test_layout_stages_and_plan_steps_match_constants(client: TestClient) -> None:
    """阶段表与六步表都从代码常量读出（顺序也逐项核对）。"""
    payload = client.get("/registry/layout").json()
    assert [row["stage"] for row in payload["stages"]] == list(STAGES)
    assert [row["name"] for row in payload["plan_steps"]] == list(PLAN_STEPS)
    assert [row["order"] for row in payload["plan_steps"]] == [1, 2, 3, 4, 5, 6]
    assert all(row["meaning"] for row in payload["plan_steps"])


def test_layout_policies_come_from_settings(client: TestClient) -> None:
    """策略缺省值来自 ``settings.retrain_*``：端点可被部署配置覆盖。"""
    payload = client.get("/registry/layout").json()
    assert payload["promotion_policy"]["min_gain"] == settings.retrain_min_gain
    assert (
        payload["promotion_policy"]["absolute_min_pass_rate"]
        == settings.retrain_absolute_min_pass_rate
    )
    assert payload["trigger_policy"]["min_new_examples"] == settings.retrain_min_new_examples
    assert payload["trigger_policy"]["cooldown_hours"] == settings.retrain_cooldown_hours


def test_layout_trigger_conditions_have_five_rows(client: TestClient) -> None:
    """条件表五行，触发器在前、否决项在后。"""
    rows = client.get("/registry/layout").json()["trigger_conditions"]
    assert [row["name"] for row in rows] == [
        "data_growth",
        "dataset_change",
        "quality_drop",
        "cooldown",
        "active_run",
    ]
    assert [row["kind"] for row in rows] == ["trigger"] * 3 + ["veto"] * 2


# --------------------------------------------------------------------------- #
# POST /registry/versions
# --------------------------------------------------------------------------- #


def test_versions_empty_list_is_a_valid_state(client: TestClient) -> None:
    """空列表合法：仓库里还没有任何版本正是首次上线前的真实状态。"""
    response = client.post("/registry/versions", json={"versions": []})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 0
    assert payload["head"] is None
    assert payload["head_key"] is None
    assert payload["counts"] == {
        "candidate": 0,
        "stable": 0,
        "rolled_back": 0,
        "archived": 0,
        "total": 0,
    }


def test_versions_reports_head_and_counts(client: TestClient) -> None:
    """折叠两个版本：head 指向 stable，counts 四个阶段键齐全。"""
    response = client.post("/registry/versions", json={"versions": [STABLE, CANDIDATE]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["head"] == "1.0.0"
    assert payload["head_key"] == VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    ).key
    assert payload["counts"]["stable"] == 1 and payload["counts"]["candidate"] == 1
    assert [item["version"] for item in payload["versions"]] == ["1.0.0", "1.0.1"]


def test_versions_head_is_null_without_stable(client: TestClient) -> None:
    """没有 stable → head 为 null（不退回候选）。

    这里刻意用一条**没有父版本**的候选：``CANDIDATE`` 的父版本是 1.0.0，
    单独提交会因为"悬空父指针"被判 400——那个行为另有一条用例守着。
    """
    lone_candidate = version_payload("1.0.1", adapter=ADAPTER_B, rating=0.72)
    payload = client.post("/registry/versions", json={"versions": [lone_candidate]}).json()
    assert payload["head"] is None
    assert payload["head_key"] is None
    assert payload["total"] == 1


def test_versions_filters_by_stage(client: TestClient) -> None:
    """阶段过滤只返回该阶段的记录。"""
    payload = client.post(
        "/registry/versions", json={"versions": [STABLE, CANDIDATE], "stage": "candidate"}
    ).json()
    assert [item["version"] for item in payload["versions"]] == ["1.0.1"]


def test_versions_accepts_unsorted_payload(client: TestClient) -> None:
    """请求顺序不影响结果：服务端按版本号升序登记（父版本必须先出现）。"""
    response = client.post("/registry/versions", json={"versions": [CANDIDATE, STABLE]})
    assert response.status_code == 200
    assert response.json()["head"] == "1.0.0"


def test_versions_markdown_is_rendered(client: TestClient) -> None:
    """响应里同时给 markdown：接口输出与人读报告是同一份数据。"""
    markdown = client.post(
        "/registry/versions", json={"versions": [STABLE, CANDIDATE]}
    ).json()["markdown"]
    assert "模型版本注册表" in markdown
    assert "| v1.0.0 | stable" in markdown
    assert "当前生产版本" in markdown


def test_versions_records_deployable_flag(client: TestClient) -> None:
    """每条记录带 ``deployable`` 与 ``missing_artifacts``（证据完整性一眼可见）."""
    partial = version_payload("1.0.1", adapter=ADAPTER_B, artifacts={"adapter": "a/1.0.1"})
    payload = client.post("/registry/versions", json={"versions": [partial]}).json()
    assert payload["versions"][0]["deployable"] is False
    assert payload["versions"][0]["missing_artifacts"] == ["merged"]


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"versions": [{"version": "1.0", "base_model": BASE,
                        "adapter_sha256": ADAPTER_A, "dataset_fingerprint": DATA_A}]}, "X.Y.Z"),
        ({"versions": [{"version": "1.0.0", "base_model": BASE,
                        "adapter_sha256": "xyz", "dataset_fingerprint": DATA_A}]}, "十六进制"),
        ({"versions": [{"version": "1.0.0", "base_model": BASE,
                        "adapter_sha256": "abc", "dataset_fingerprint": DATA_A}]}, "至少"),
        ({"versions": [{"version": "1.0.0", "base_model": "  ",
                        "adapter_sha256": ADAPTER_A, "dataset_fingerprint": DATA_A}]},
         "base_model"),
        ({"versions": [dict(STABLE, stage="production")]}, "未知阶段"),
        ({"versions": [dict(CANDIDATE, parent_version="9.9.9")]}, "父版本"),
    ],
)
def test_versions_rejects_invalid_records_with_400(
    client: TestClient, payload: dict, match: str
) -> None:
    """六类非法记录全部 400，详情来自 ``RegistryError`` 原文（可操作）。"""
    response = client.post("/registry/versions", json=payload)
    assert response.status_code == 400
    assert match in response.json()["detail"]


def test_versions_rejects_conflicting_key(client: TestClient) -> None:
    """同一个三元组两个版本号 → 400，且详情里写明"哪个版本占了哪个键"。"""
    clash = version_payload("1.0.9", adapter=ADAPTER_A, dataset=DATA_A)
    response = client.post("/registry/versions", json={"versions": [STABLE, clash]})
    assert response.status_code == 400
    assert "已被 1.0.0 占用" in response.json()["detail"]


def test_versions_rejects_unknown_stage_filter(client: TestClient) -> None:
    """``stage`` 过滤器写错也报 400（而不是静默返回空列表）。"""
    response = client.post(
        "/registry/versions", json={"versions": [STABLE], "stage": "production"}
    )
    assert response.status_code == 400
    assert "未知阶段" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /registry/candidate/evaluate
# --------------------------------------------------------------------------- #


def test_candidate_first_launch_promotes(client: TestClient) -> None:
    """``current`` 留空 → 首次上线分支：只做绝对门槛判定。"""
    payload = client.post(
        "/registry/candidate/evaluate",
        json={"candidate": version_payload("1.0.0", rating=0.58)},
    ).json()
    assert payload["decision"]["action"] == "promote"
    assert payload["decision"]["gain"] is None
    assert payload["decision"]["comparable"] is False
    assert "首次上线" in payload["decision"]["reason"]


def test_candidate_gain_above_threshold_promotes(client: TestClient) -> None:
    """可比 + 增益 +0.12 → promote，并给出 target 版本。"""
    payload = client.post(
        "/registry/candidate/evaluate", json={"current": STABLE, "candidate": CANDIDATE}
    ).json()
    assert payload["decision"]["action"] == "promote"
    assert payload["decision"]["gain"] == pytest.approx(0.12)
    assert payload["decision"]["target_version"] == "1.0.1"
    assert "采纳判定" in payload["markdown"]


def test_candidate_gain_in_dead_band_holds(client: TestClient) -> None:
    """增益落在死区 → hold，理由里点名死区。"""
    half_step = version_payload("1.0.1", adapter=ADAPTER_B, parent="1.0.0", rating=0.61)
    payload = client.post(
        "/registry/candidate/evaluate", json={"current": STABLE, "candidate": half_step}
    ).json()
    assert payload["decision"]["action"] == "hold"
    assert payload["decision"]["gain"] == pytest.approx(0.01)
    assert "死区" in payload["decision"]["reason"]


def test_candidate_incomparable_uses_absolute_threshold(client: TestClient) -> None:
    """换了数据集的候选 → 不算差值，改用绝对门槛（响应里的 checks 能看出来）。"""
    other_dataset = version_payload("1.1.0", adapter=ADAPTER_B, dataset=DATA_B, rating=0.72)
    payload = client.post(
        "/registry/candidate/evaluate", json={"current": STABLE, "candidate": other_dataset}
    ).json()
    decision = payload["decision"]
    assert decision["action"] == "promote"
    assert decision["gain"] is None
    assert "absolute_pass_rate" in [item["name"] for item in decision["checks"]]
    assert "gain" not in [item["name"] for item in decision["checks"]]


def test_candidate_policy_override_is_honoured(client: TestClient) -> None:
    """请求里的 ``min_gain`` 覆盖配置值（把门槛抬到 0.5 后同一候选被拒）。"""
    payload = client.post(
        "/registry/candidate/evaluate",
        json={"current": STABLE, "candidate": CANDIDATE, "policy": {"min_gain": 0.5}},
    ).json()
    assert payload["decision"]["action"] == "hold"
    assert payload["decision"]["gain"] == pytest.approx(0.12)


def test_candidate_unknown_policy_keys_are_ignored(client: TestClient) -> None:
    """未知策略键被忽略（旧客户端带上已删除的键不该 500）."""
    response = client.post(
        "/registry/candidate/evaluate",
        json={
            "current": STABLE,
            "candidate": CANDIDATE,
            "policy": {"min_gain": 0.02, "legacy_key": 1},
        },
    )
    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "promote"


@pytest.mark.parametrize(
    "policy, match",
    [
        ({"min_gain": -0.1}, "min_gain"),
        ({"regression_tolerance": -1.0}, "regression_tolerance"),
        ({"absolute_min_pass_rate": 2.0}, "absolute_min_pass_rate"),
        ({"max_candidate_age_hours": 0}, "max_candidate_age_hours"),
    ],
)
def test_candidate_rejects_invalid_policy_with_400(
    client: TestClient, policy: dict, match: str
) -> None:
    """非法策略值一律 400（不是 422）——与 day050~day057 的错误码语言一致。"""
    response = client.post(
        "/registry/candidate/evaluate",
        json={"current": STABLE, "candidate": CANDIDATE, "policy": policy},
    )
    assert response.status_code == 400
    assert match in response.json()["detail"]


def test_candidate_rejects_invalid_triple_with_400(client: TestClient) -> None:
    """候选的三元组非法 → 400（校验不复制到 Pydantic，错误信息来自 registry 包）。"""
    response = client.post(
        "/registry/candidate/evaluate",
        json={"candidate": version_payload("1.0.1", adapter="xyz")},
    )
    assert response.status_code == 400
    assert "十六进制" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /registry/triggers/evaluate
# --------------------------------------------------------------------------- #


def test_triggers_idle_state_does_not_train(client: TestClient) -> None:
    """什么都不报 → 不训练，且备注指向"该去看什么"。"""
    payload = client.post("/registry/triggers/evaluate", json={}).json()
    assert payload["decision"]["should_retrain"] is False
    assert payload["decision"]["fired"] == []
    assert payload["decision"]["vetoed_by"] == []
    assert "调冷却期" in payload["decision"]["notes"]


def test_triggers_data_growth_fires_and_reports_conditions(client: TestClient) -> None:
    """数据涨 30 条、冷却已过 → 训练，并带上五行条件表。"""
    payload = client.post(
        "/registry/triggers/evaluate",
        json={
            "new_examples": 30,
            "dataset_fingerprint": DATA_B,
            "stable_dataset_fingerprint": DATA_A,
            "hours_since_last_train": 30.0,
        },
    ).json()
    assert payload["decision"]["should_retrain"] is True
    assert payload["decision"]["fired"] == ["data_growth", "dataset_change"]
    assert len(payload["conditions"]) == 5
    assert "**应该重训**" in payload["markdown"]


def test_triggers_veto_is_reported_separately_from_fired(client: TestClient) -> None:
    """冷却期否决时，``fired`` 与 ``vetoed_by`` **同时非空**——这正是两类信号要分开的原因."""
    payload = client.post(
        "/registry/triggers/evaluate",
        json={"new_examples": 30, "hours_since_last_train": 2.0},
    ).json()
    decision = payload["decision"]
    assert decision["should_retrain"] is False
    assert decision["fired"] == ["data_growth"]
    assert decision["vetoed_by"] == ["cooldown"]
    assert "再等等" in decision["notes"]


def test_triggers_missing_values_are_not_zero(client: TestClient) -> None:
    """``online_pass_rate`` / ``hours_since_last_train`` 留空 = 缺失.

    若把缺失当 0，从未训练的仓库会被冷却期永久拦住——这里正相反：
    从未训练时冷却期不否决，合格率缺失不触发质量下降。
    """
    payload = client.post(
        "/registry/triggers/evaluate",
        json={"new_examples": 37, "dataset_fingerprint": DATA_A},
    ).json()
    decision = payload["decision"]
    assert decision["should_retrain"] is True
    assert decision["vetoed_by"] == []
    quality = next(item for item in decision["evaluations"] if item["name"] == "quality_drop")
    assert quality["actual"] is None
    assert "缺失不触发" in quality["reason"]


def test_triggers_policy_override(client: TestClient) -> None:
    """覆盖 ``min_new_examples`` 与 ``cooldown_hours``。"""
    payload = client.post(
        "/registry/triggers/evaluate",
        json={
            "new_examples": 5,
            "hours_since_last_train": 1.0,
            "policy": {"min_new_examples": 3, "cooldown_hours": 0.5},
        },
    ).json()
    assert payload["decision"]["should_retrain"] is True
    thresholds = {item["name"]: item["threshold"] for item in payload["decision"]["evaluations"]}
    assert thresholds["data_growth"] == 3
    assert thresholds["cooldown"] == 0.5


def test_triggers_negative_increment_is_a_422(client: TestClient) -> None:
    """负数增量是**请求形状错误** → 422（与业务规则不允许的 400 区分开）."""
    response = client.post("/registry/triggers/evaluate", json={"new_examples": -1})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "policy, match",
    [
        ({"min_new_examples": 0}, "min_new_examples"),
        ({"min_pass_rate": 2.0}, "min_pass_rate"),
        ({"cooldown_hours": -1}, "cooldown_hours"),
        ({"max_parallel_runs": 0}, "max_parallel_runs"),
    ],
)
def test_triggers_rejects_invalid_policy_with_400(
    client: TestClient, policy: dict, match: str
) -> None:
    """非法触发策略 → 400（来自 ``TriggerPolicy`` 的构造期校验）。"""
    response = client.post("/registry/triggers/evaluate", json={"policy": policy})
    assert response.status_code == 400
    assert match in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /registry/rollback/plan
# --------------------------------------------------------------------------- #


ROLLBACK_CHAIN = [
    version_payload("1.0.0", stage="stable", rating=0.60),
    version_payload("1.0.1", adapter=ADAPTER_B, parent="1.0.0", stage="stable", rating=0.68),
    version_payload("1.0.2", adapter="c" * 64, parent="1.0.1", stage="stable", rating=0.71),
]


def test_rollback_plan_orders_verify_before_freeze(client: TestClient) -> None:
    """回滚计划四步，且 ``verify`` 必须排在 ``freeze`` 之前.

    顺序反过来也能通过绝大多数测试，代价只在最坏情况下暴露：
    先冻结当前版本、再发现目标不可部署，结果是"生产上没有任何可服务版本"。
    """
    payload = client.post(
        "/registry/rollback/plan",
        json={"versions": ROLLBACK_CHAIN, "version": "1.0.2", "reason": "拒答率异常"},
    ).json()
    plan = payload["plan"]
    assert plan["action"] == "rollback"
    assert plan["target_version"] == "1.0.1"
    assert [step["action"] for step in plan["steps"]] == [
        "verify",
        "freeze",
        "record",
        "observe",
    ]
    assert [step["blocking"] for step in plan["steps"]] == [True, True, False, False]
    assert plan["should_execute"] is True
    assert "拒答率异常" in plan["reason"]
    assert "回滚计划" in payload["markdown"]


def test_rollback_plan_returns_lineage(client: TestClient) -> None:
    """响应里带上版本链（回滚目标就是链上第一个可部署的 stable 祖先）."""
    payload = client.post(
        "/registry/rollback/plan", json={"versions": ROLLBACK_CHAIN, "version": "1.0.2"}
    ).json()
    assert [item["version"] for item in payload["lineage"]] == ["1.0.2", "1.0.1", "1.0.0"]


def test_rollback_plan_holds_when_no_ancestor(client: TestClient) -> None:
    """首个 stable 版本没有退路 → ``should_execute=False`` 且零步."""
    payload = client.post(
        "/registry/rollback/plan",
        json={"versions": [ROLLBACK_CHAIN[0]], "version": "1.0.0"},
    ).json()
    plan = payload["plan"]
    assert plan["action"] == "hold"
    assert plan["should_execute"] is False
    assert plan["steps"] == []
    assert "没有可部署的 stable 祖先" in plan["reason"]


def test_rollback_plan_holds_for_candidate(client: TestClient) -> None:
    """候选退回候选不需要任何动作（回滚只针对正在服务的版本）."""
    payload = client.post(
        "/registry/rollback/plan",
        json={"versions": [STABLE, CANDIDATE], "version": "1.0.1"},
    ).json()
    assert payload["plan"]["action"] == "hold"
    assert "不是 stable" in payload["plan"]["reason"]


def test_rollback_plan_skips_undeployable_ancestor(client: TestClient) -> None:
    """最近的祖先不可部署时自动往前找（stable 记录可能只是"当年提升过"）."""
    chain = [
        version_payload("1.0.0", stage="stable", rating=0.60),
        version_payload(
            "1.0.1",
            adapter=ADAPTER_B,
            parent="1.0.0",
            stage="stable",
            rating=0.68,
            artifacts={"adapter": "a/1.0.1"},
        ),
        version_payload("1.0.2", adapter="c" * 64, parent="1.0.1", stage="stable", rating=0.71),
    ]
    payload = client.post(
        "/registry/rollback/plan", json={"versions": chain, "version": "1.0.2"}
    ).json()
    assert payload["plan"]["target_version"] == "1.0.0"


def test_rollback_plan_rejects_unknown_version(client: TestClient) -> None:
    """版本不在请求携带的表里 → 400，详情说明它既不是版本号也不是版本键."""
    response = client.post(
        "/registry/rollback/plan",
        json={"versions": ROLLBACK_CHAIN, "version": "9.9.9"},
    )
    assert response.status_code == 400
    assert "不存在" in response.json()["detail"]


def test_rollback_plan_requires_at_least_one_version(client: TestClient) -> None:
    """空版本表 → 422（没有版本链可走，这个问题在请求形状层就能回答）."""
    response = client.post(
        "/registry/rollback/plan", json={"versions": [], "version": "1.0.0"}
    )
    assert response.status_code == 422


def test_rollback_plan_rejects_non_positive_observe_window(client: TestClient) -> None:
    """观察窗口必须为正（0 会让"回滚完成"没有时间点）→ 422 由 Pydantic 拦下."""
    response = client.post(
        "/registry/rollback/plan",
        json={"versions": ROLLBACK_CHAIN, "version": "1.0.2", "observe_window_hours": 0},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 端到端：一次完整的"重训 → 采纳 → 回滚"链路（全部通过 HTTP）
# --------------------------------------------------------------------------- #


def test_full_lifecycle_over_http(client: TestClient) -> None:
    """四个端点串起来走一遍：触发 → 采纳 → 列版本 → 回滚.

    这条用例的价值在于**它验证的是端点之间的契约**：触发判定给出的
    "应该重训"、采纳判定给出的"promote"、版本表给出的 head、回滚计划
    给出的目标版本，四者在同一条时间线上必须自洽。
    """
    # 1) 触发判定：数据涨了 30 条，冷却已过
    decision = client.post(
        "/registry/triggers/evaluate",
        json={
            "new_examples": 30,
            "dataset_fingerprint": DATA_B,
            "stable_dataset_fingerprint": DATA_A,
            "hours_since_last_train": 30.0,
        },
    ).json()["decision"]
    assert decision["should_retrain"] is True

    # 2) 采纳判定：新数据集上训练出的候选，绝对门槛达标
    promoted = client.post(
        "/registry/candidate/evaluate",
        json={
            "current": STABLE,
            "candidate": version_payload(
                "1.1.0", adapter=ADAPTER_B, dataset=DATA_B, rating=0.72
            ),
        },
    ).json()["decision"]
    assert promoted["action"] == "promote"

    # 3) 版本表：两个 stable（rollback 目标与当前版本）
    chain = [
        STABLE,
        version_payload("1.1.0", adapter=ADAPTER_B, dataset=DATA_B, parent="1.0.0",
                        stage="stable", rating=0.72),
    ]
    listing = client.post("/registry/versions", json={"versions": chain}).json()
    assert listing["head"] == "1.1.0"
    assert listing["counts"]["stable"] == 2

    # 4) 回滚计划：从 1.1.0 退到 1.0.0
    plan = client.post(
        "/registry/rollback/plan",
        json={"versions": chain, "version": "1.1.0", "reason": "上线后质量劣化"},
    ).json()["plan"]
    assert plan["action"] == "rollback"
    assert plan["target_version"] == "1.0.0"
    assert plan["should_execute"] is True

    # 5) 端点的折叠结果与本地注册表一致（同一个三元组 → 同一个键）
    registry = ModelRegistry()
    for item in chain:
        registry.register(ModelVersion.from_dict(item))
    assert registry.head().version == listing["head"]
    assert registry.counts()["stable"] == listing["counts"]["stable"]
