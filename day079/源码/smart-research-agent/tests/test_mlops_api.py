"""day059 MLOps 端点测试（M5-D10）：/mlops/* 四个端点.

全部走 TestClient（进程内 ASGI 调用）：不起真实服务、不联网、**不做真实训练**、
**不写盘**。dry-run 端点用确定性的参考回调驱动六阶段，因此同一个请求体
永远得到同一个响应。

这个文件挑的是四件"会静默失效"的事来钉：

1. **对照表必须与代码同源**：阶段表的依赖/产出 == ``STAGE_SPECS``、
   门禁表的阈值 == ``settings.mlops_*``、CI 命令 == ``workflow_commands()``；
2. **``blocked`` 不是 ``failed``**：门禁不通过时 ``publish`` 是 ``blocked``
   而 ``failed`` 为 ``false``——把它报成失败会让 CI 的红灯掩盖门禁报告；
3. **错误码语言**：非法策略 → 400，越界的请求形状 → 422（与 day050~day058 一致）；
4. **注册表始终在内存里**：端点会出现 ``version`` 与 ``published``，
   但磁盘上什么都不写（HTTP 层不做版本表写入）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.mlops import (
    LIMITATIONS,
    PIPELINE_STAGES,
    STAGE_SPECS,
    WORKFLOW_DIRECTORY,
    WORKFLOW_FILENAME,
    CIConfig,
    ReleaseGates,
    critical_path,
    gate_table,
    render_github_actions,
    stage_table,
    workflow_commands,
    workflow_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = PROJECT_ROOT / WORKFLOW_DIRECTORY / WORKFLOW_FILENAME

GOOD_METRICS = {
    "eval_pass_rate": 0.65,
    "eval_pass_rate_delta": 0.05,
    "adapter_mebibytes": 0.05,
    "dataset_fingerprint": "3f1b0c9d7e5a2468",
    "base_model": "Qwen3-8B",
}
GOOD_ARTIFACTS = {"adapter": "a/adapter", "merged": "m/merged"}


@pytest.fixture
def client() -> TestClient:
    """注入 MockLLM 的离线测试客户端（端点本身不碰 LLM，注入只是为不起真实客户端）."""
    return TestClient(create_app(llm=MockLLM(default="offline")))


# --------------------------------------------------------------------------- #
# GET /mlops/stages
# --------------------------------------------------------------------------- #


def test_stages_endpoint_is_a_valid_self_description(client: TestClient) -> None:
    """六阶段、六门禁、CI 摘要、三份用途/限制清单都齐全。"""
    response = client.get("/mlops/stages")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "stages",
        "critical_path",
        "gates",
        "ci",
        "intended_use",
        "out_of_scope",
        "limitations",
    }
    assert len(payload["stages"]) == 6
    assert len(payload["gates"]) == 6
    assert len(payload["limitations"]) == len(LIMITATIONS)


def test_stages_match_the_code_tables(client: TestClient) -> None:
    """阶段表与主链必须等于代码常量——"文档说 64 MiB、代码是 16 MiB"会立刻变红。"""
    payload = client.get("/mlops/stages").json()
    assert payload["stages"] == stage_table()
    assert payload["critical_path"] == critical_path()
    assert [row["name"] for row in payload["stages"]] == list(PIPELINE_STAGES)


def test_stages_dependencies_match_the_spec_objects(client: TestClient) -> None:
    """依赖与产出逐项等于 ``STAGE_SPECS``（含顺序）。"""
    payload = client.get("/mlops/stages").json()
    for row, spec in zip(payload["stages"], STAGE_SPECS, strict=True):
        assert row["name"] == spec.name
        assert row["requires"] == list(spec.requires)
        assert row["produces"] == list(spec.produces)


def test_stages_gates_come_from_settings(client: TestClient) -> None:
    """门禁阈值来自 ``settings.mlops_*``（部署层面可调）。"""
    payload = client.get("/mlops/stages").json()
    table = {row["name"]: row for row in payload["gates"]}
    assert table["pass_rate"]["threshold"] == settings.mlops_min_pass_rate
    assert table["regression"]["threshold"] == settings.mlops_max_regression
    assert table["adapter_size"]["threshold"] == settings.mlops_max_adapter_mebibytes
    assert table["cost"]["threshold"] == settings.mlops_max_cost_per_1k_tokens
    assert payload["gates"] == gate_table(ReleaseGates(
        min_pass_rate=settings.mlops_min_pass_rate,
        max_regression=settings.mlops_max_regression,
        max_adapter_mebibytes=settings.mlops_max_adapter_mebibytes,
        max_cost_per_1k_tokens=settings.mlops_max_cost_per_1k_tokens,
    ))


def test_stages_ci_summary_matches_the_renderer(client: TestClient) -> None:
    """CI 摘要里的 cron、命令与阈值都来自 ``CIConfig`` 与 ``workflow_commands``."""
    payload = client.get("/mlops/stages").json()
    expected = workflow_summary(
        CIConfig(
            python_version=settings.mlops_ci_python_version,
            schedule=settings.mlops_ci_schedule,
            timeout_minutes=settings.mlops_ci_timeout_minutes,
            gates=ReleaseGates(
                min_pass_rate=settings.mlops_min_pass_rate,
                max_regression=settings.mlops_max_regression,
                max_adapter_mebibytes=settings.mlops_max_adapter_mebibytes,
                max_cost_per_1k_tokens=settings.mlops_max_cost_per_1k_tokens,
            ),
        )
    )
    assert payload["ci"] == expected


def test_stages_limitations_are_specific(client: TestClient) -> None:
    """限制里带着可核对的数字（"可能表现不佳"这种话对判断没有帮助）。"""
    payload = client.get("/mlops/stages").json()
    assert any("1/18" in item for item in payload["limitations"])
    assert all(len(item) > 10 for item in payload["out_of_scope"])


# --------------------------------------------------------------------------- #
# POST /mlops/gates/evaluate
# --------------------------------------------------------------------------- #


def test_gates_endpoint_passes_a_clean_candidate(client: TestClient) -> None:
    """全项达标 → passed 为真，六项都在报告里。"""
    response = client.post(
        "/mlops/gates/evaluate", json={"metrics": GOOD_METRICS, "artifacts": GOOD_ARTIFACTS}
    )
    assert response.status_code == 200
    report = response.json()["report"]
    assert report["passed"] is True
    assert len(report["checks"]) == 6
    assert "发布门禁：通过" in response.json()["markdown"]


@pytest.mark.parametrize(
    "patch, expected_failure",
    [
        ({"metrics": {**GOOD_METRICS, "eval_pass_rate": 0.2}}, "pass_rate"),
        ({"metrics": {**GOOD_METRICS, "eval_pass_rate_delta": -0.05}}, "regression"),
        ({"metrics": {**GOOD_METRICS, "adapter_mebibytes": 80.0}}, "adapter_size"),
        ({"metrics": {k: v for k, v in GOOD_METRICS.items() if k != "dataset_fingerprint"}},
         "dataset_traceability"),
        ({"artifacts": {"adapter": "a/adapter"}}, "artifact_completeness"),
        ({"metrics": {k: v for k, v in GOOD_METRICS.items() if k != "eval_pass_rate"}},
         "pass_rate"),
    ],
)
def test_gates_endpoint_single_factor_failures(
    client: TestClient, patch: dict, expected_failure: str
) -> None:
    """每次只改一项，看是哪一项把发布拦下来的（**先看拦在哪，再决定改什么**）。"""
    body = {"metrics": GOOD_METRICS, "artifacts": GOOD_ARTIFACTS, **patch}
    report = client.post("/mlops/gates/evaluate", json=body).json()["report"]
    assert report["passed"] is False
    assert report["blocking_failures"] == [expected_failure]


def test_gates_endpoint_reports_missing_evidence_as_skipped(client: TestClient) -> None:
    """缺基线/缺成本 → 进 ``skipped``（缺证据），而不是 ``warnings``（策略关闭）."""
    report = client.post(
        "/mlops/gates/evaluate", json={"metrics": GOOD_METRICS, "artifacts": GOOD_ARTIFACTS}
    ).json()["report"]
    assert report["skipped"] == ["cost"]
    assert report["warnings"] == []


def test_gates_endpoint_reports_disabled_checks_as_warnings(client: TestClient) -> None:
    """关掉一项检查是**告警**（一个值得被看见的决定），不是"通过"。"""
    report = client.post(
        "/mlops/gates/evaluate",
        json={
            "metrics": {"eval_pass_rate": 0.6, "adapter_mebibytes": 1.0},
            "policy": {"require_traceability": False, "require_artifacts": False},
        },
    ).json()["report"]
    assert report["passed"] is True
    assert report["warnings"] == ["dataset_traceability", "artifact_completeness"]


def test_gates_endpoint_honours_policy_overrides(client: TestClient) -> None:
    """请求里的策略覆盖配置值（把门槛抬到 0.9 后同一份指标被拦）。"""
    report = client.post(
        "/mlops/gates/evaluate",
        json={
            "metrics": GOOD_METRICS,
            "artifacts": GOOD_ARTIFACTS,
            "policy": {"min_pass_rate": 0.9},
        },
    ).json()["report"]
    assert report["blocking_failures"] == ["pass_rate"]


def test_gates_endpoint_ignores_unknown_policy_keys(client: TestClient) -> None:
    """未知策略键被忽略（旧客户端带上已删除的键不该 500）。"""
    response = client.post(
        "/mlops/gates/evaluate",
        json={"metrics": GOOD_METRICS, "artifacts": GOOD_ARTIFACTS, "policy": {"legacy": 1}},
    )
    assert response.status_code == 200
    assert response.json()["report"]["passed"] is True


@pytest.mark.parametrize(
    "policy, match",
    [
        ({"min_pass_rate": 2.0}, "min_pass_rate"),
        ({"max_regression": -1.0}, "max_regression"),
        ({"max_adapter_mebibytes": 0}, "max_adapter_mebibytes"),
        ({"max_cost_per_1k_tokens": 0}, "max_cost_per_1k_tokens"),
    ],
)
def test_gates_endpoint_rejects_invalid_policy_with_400(
    client: TestClient, policy: dict, match: str
) -> None:
    """非法策略 → **400**（不是 422）——与 day050~day058 的错误码语言一致。"""
    response = client.post("/mlops/gates/evaluate", json={"metrics": {}, "policy": policy})
    assert response.status_code == 400
    assert match in response.json()["detail"]


def test_gates_endpoint_includes_commit(client: TestClient) -> None:
    """提交号进报告（门禁结论要能指回一次提交）。"""
    report = client.post(
        "/mlops/gates/evaluate",
        json={"metrics": GOOD_METRICS, "artifacts": GOOD_ARTIFACTS, "commit": "deadbeef"},
    ).json()["report"]
    assert report["commit"] == "deadbeef"


# --------------------------------------------------------------------------- #
# POST /mlops/pipeline/dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_passes_the_gate_but_does_not_publish(client: TestClient) -> None:
    """缺省 ``dry_run=true``：六阶段照走，门禁通过，``publish`` 是 blocked."""
    payload = client.post("/mlops/pipeline/dry-run", json={}).json()
    statuses = {stage["name"]: stage["status"] for stage in payload["stages"]}
    assert statuses == {
        "ingest": "ok",
        "train": "ok",
        "evaluate": "ok",
        "gate": "ok",
        "package": "ok",
        "publish": "blocked",
    }
    assert payload["gate"]["passed"] is True
    assert payload["published"] is False
    assert payload["failed"] is False
    assert payload["version"] is None
    assert "dry_run" in payload["summary"] or "未发布" in payload["summary"]


def test_dry_run_with_publish_true_registers_in_memory_only(client: TestClient) -> None:
    """``dry_run=false`` 时会登记并置为 stable——但注册表是内存的，磁盘不变。"""
    payload = client.post("/mlops/pipeline/dry-run", json={"dry_run": False}).json()
    assert payload["published"] is True
    assert payload["failed"] is False
    assert payload["version"]["version"] == "1.1.0"
    assert payload["version"]["stage"] == "stable"
    assert payload["version"]["parent_version"] == "1.0.0"


def test_dry_run_default_request_is_reproducible(client: TestClient) -> None:
    """确定性：同一个请求体两次的响应完全一致（适配器哈希由哈希派生，不用随机数）."""
    first = client.post("/mlops/pipeline/dry-run", json={}).json()
    second = client.post("/mlops/pipeline/dry-run", json={}).json()
    assert first["run"]["run_id"] == second["run"]["run_id"]
    assert first["card"]["adapter_sha256"] == second["card"]["adapter_sha256"]
    assert first["summary"] == second["summary"]


def test_dry_run_low_pass_rate_blocks_on_pass_rate(client: TestClient) -> None:
    """合格率不达标 → ``publish`` 阻塞，但 ``failed`` 仍为 false."""
    payload = client.post("/mlops/pipeline/dry-run", json={"pass_rate": 0.30}).json()
    assert payload["gate"]["passed"] is False
    assert "pass_rate" in payload["gate"]["blocking_failures"]
    assert payload["published"] is False
    assert payload["failed"] is False
    stages = {stage["name"]: stage["status"] for stage in payload["stages"]}
    assert stages["gate"] == "blocked" and stages["publish"] == "blocked"


def test_dry_run_missing_merged_artifact_blocks_on_completeness(client: TestClient) -> None:
    """缺合并产物 → 阻塞在 ``artifact_completeness``（门禁报告里指得出是哪一项）."""
    payload = client.post(
        "/mlops/pipeline/dry-run", json={"with_merged_artifact": False}
    ).json()
    assert payload["gate"]["blocking_failures"] == ["artifact_completeness"]
    assert payload["failed"] is False


def test_dry_run_oversized_adapter_blocks(client: TestClient) -> None:
    """适配器 80 MiB > 64 MiB 上限 → 阻塞。"""
    payload = client.post(
        "/mlops/pipeline/dry-run", json={"adapter_mebibytes": 80.0}
    ).json()
    assert payload["gate"]["blocking_failures"] == ["adapter_size"]


def test_dry_run_gate_can_be_relaxed(client: TestClient) -> None:
    """放宽产物检查后同一个候选放行（策略可被请求覆盖），但 dry_run 仍不发。"""
    payload = client.post(
        "/mlops/pipeline/dry-run",
        json={"with_merged_artifact": False, "policy": {"require_artifacts": False}},
    ).json()
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["warnings"] == ["artifact_completeness"]
    assert payload["published"] is False
    assert payload["failed"] is False


def test_dry_run_explicit_adapter_hash_is_used(client: TestClient) -> None:
    """显式给出适配器哈希时用它（便于把演练对齐到一次真实训练）."""
    adapter = "f" * 64
    payload = client.post(
        "/mlops/pipeline/dry-run", json={"adapter_sha256": adapter, "dry_run": False}
    ).json()
    assert payload["card"]["adapter_sha256"] == adapter
    assert payload["version"]["version_key"]


def test_dry_run_response_carries_the_manifest_and_markdown(client: TestClient) -> None:
    """响应带发布清单与整次运行的 markdown（报告与机器可读两份都在）。"""
    payload = client.post("/mlops/pipeline/dry-run", json={}).json()
    assert payload["manifest"]["gate_passed"] is True
    assert payload["manifest"]["adapter_sha256"] == payload["card"]["adapter_sha256"]
    assert "微调流水线运行" in payload["markdown"]
    assert "发布门禁：通过" in payload["markdown"]


def test_dry_run_records_params_in_the_run_id(client: TestClient) -> None:
    """参数进 run_id：改一个参数就是另一次实验。"""
    first = client.post("/mlops/pipeline/dry-run", json={"params": {"lora_r": 8}}).json()
    second = client.post("/mlops/pipeline/dry-run", json={"params": {"lora_r": 16}}).json()
    assert first["run"]["run_id"] != second["run"]["run_id"]
    assert first["run"]["params"]["lora_r"] == 8


def test_dry_run_rejects_blank_fingerprint_with_400(client: TestClient) -> None:
    """空数据集指纹 → 400（版本三元组需要它）。"""
    response = client.post("/mlops/pipeline/dry-run", json={"dataset_fingerprint": ""})
    assert response.status_code == 400
    assert "dataset_fingerprint" in response.json()["detail"]


def test_dry_run_rejects_invalid_policy_with_400(client: TestClient) -> None:
    """非法门禁策略 → 400。"""
    response = client.post(
        "/mlops/pipeline/dry-run", json={"policy": {"min_pass_rate": 5.0}}
    )
    assert response.status_code == 400
    assert "min_pass_rate" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"pass_rate": 1.5},
        {"adapter_mebibytes": 0},
        {"adapter_mebibytes": -1.0},
        {"baseline_pass_rate": -0.1},
    ],
)
def test_dry_run_request_shape_errors_are_422(client: TestClient, body: dict) -> None:
    """请求**形状**错误 → 422（与业务规则不允许的 400 区分开）."""
    response = client.post("/mlops/pipeline/dry-run", json=body)
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# GET /mlops/ci/workflow
# --------------------------------------------------------------------------- #


def test_workflow_endpoint_returns_the_rendered_yaml(client: TestClient) -> None:
    """端点返回渲染结果、路径与命令（"手动复现 CI"不需要去 YAML 里抄）。"""
    response = client.get("/mlops/ci/workflow")
    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == f"{WORKFLOW_DIRECTORY}/{WORKFLOW_FILENAME}"
    assert payload["yaml"] == render_github_actions(
        CIConfig(
            python_version=settings.mlops_ci_python_version,
            schedule=settings.mlops_ci_schedule,
            timeout_minutes=settings.mlops_ci_timeout_minutes,
            gates=ReleaseGates(
                min_pass_rate=settings.mlops_min_pass_rate,
                max_regression=settings.mlops_max_regression,
                max_adapter_mebibytes=settings.mlops_max_adapter_mebibytes,
                max_cost_per_1k_tokens=settings.mlops_max_cost_per_1k_tokens,
            ),
        )
    )
    assert payload["commands"][1] == "scripts/mlops_demo.py"


def test_workflow_endpoint_matches_the_committed_file(client: TestClient) -> None:
    """**接桥断言**：端点返回的 YAML 与仓库里那份文件逐字相同.

    三者（``settings`` → 渲染器 → 仓库文件）之间只有一份权威；
    这条测试让"改了阈值却忘了重新生成"立刻变红。
    """
    payload = client.get("/mlops/ci/workflow").json()
    assert WORKFLOW_PATH.exists()
    assert payload["yaml"] == WORKFLOW_PATH.read_text(encoding="utf-8")


def test_workflow_endpoint_summary_mentions_utc(client: TestClient) -> None:
    """CI 摘要显式提醒 schedule 按 UTC 解释。"""
    payload = client.get("/mlops/ci/workflow").json()
    assert "UTC" in payload["summary"]["schedule_note"]
    assert payload["summary"]["timeout_minutes"] == settings.mlops_ci_timeout_minutes


def test_workflow_endpoint_commands_match_the_helper(client: TestClient) -> None:
    """命令列表与 ``workflow_commands()`` 完全一致（只有一处权威）。"""
    payload = client.get("/mlops/ci/workflow").json()
    expected = workflow_commands(
        CIConfig(
            python_version=settings.mlops_ci_python_version,
            schedule=settings.mlops_ci_schedule,
            timeout_minutes=settings.mlops_ci_timeout_minutes,
            gates=ReleaseGates(
                min_pass_rate=settings.mlops_min_pass_rate,
                max_regression=settings.mlops_max_regression,
                max_adapter_mebibytes=settings.mlops_max_adapter_mebibytes,
                max_cost_per_1k_tokens=settings.mlops_max_cost_per_1k_tokens,
            ),
        )
    )
    assert payload["commands"] == expected
