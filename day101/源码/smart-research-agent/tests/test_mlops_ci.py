"""day059 CI 集成测试（M5-D10）：生成 workflow、与仓库文件逐字一致、YAML 的 ``on`` 陷阱.

这个文件里最重要的一条断言是
``test_committed_workflow_matches_the_renderer``：**仓库里那份
``.github/workflows/finetune-nightly.yml`` 必须与渲染器的输出逐字相同**。

它把"改门禁阈值"变成一次**可追溯的 diff**：改了 ``ReleaseGates`` 的字段，
workflow 里的 ``--min-pass-rate`` 会跟着变；忘了重新生成，测试立刻变红。
手写 YAML 的问题是反过来的——阈值会在 YAML 里悄悄漂移，
而 YAML 里那个数字没有任何东西在守护它。

另一个值得单独说的考点是 **YAML 1.1 会把裸 ``on`` 解析成布尔 ``True``**：
``payload["on"]`` 会 KeyError，``payload[True]`` 才对。这不是本课的 bug，
而是所有解析 GitHub Actions workflow 的脚本都会撞上的一件事——
所以它被写成一条用例，而不是留在注释里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_research_agent.mlops import (
    GATE_SCRIPT,
    WORKFLOW_DIRECTORY,
    WORKFLOW_FILENAME,
    CIConfig,
    MLOpsError,
    ReleaseGates,
    parse_workflow,
    render_github_actions,
    workflow_commands,
    workflow_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = PROJECT_ROOT / WORKFLOW_DIRECTORY / WORKFLOW_FILENAME


# --------------------------------------------------------------------------- #
# CIConfig 校验
# --------------------------------------------------------------------------- #


def test_ci_config_defaults() -> None:
    """六个缺省值逐项钉住（它们进文档与端点）。"""
    payload = CIConfig().to_dict()
    assert payload["workflow_name"] == "finetune-nightly"
    assert payload["runs_on"] == "ubuntu-latest"
    assert payload["python_version"] == "3.11"
    assert payload["schedule"] == "0 20 * * *"
    assert payload["timeout_minutes"] == 30
    assert payload["gate_script"] == GATE_SCRIPT
    assert payload["gates"]["min_pass_rate"] == 0.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"workflow_name": "  "}, "workflow_name"),
        ({"schedule": "0 20 * *"}, "五段式 cron"),
        ({"schedule": "每小时"}, "五段式 cron"),
        ({"timeout_minutes": 0}, "timeout_minutes"),
        ({"artifact_paths": ()}, "artifact_paths"),
    ],
)
def test_ci_config_validates_at_construction(kwargs: dict, match: str) -> None:
    """非法配置在构造期拦下（cron 段数写错会让 workflow 静默不触发）。"""
    with pytest.raises(MLOpsError, match=match):
        CIConfig(**kwargs)


def test_schedule_is_documented_as_utc() -> None:
    """摘要里显式提醒"GitHub Actions 的 schedule 一律按 UTC 解释"。"""
    summary = workflow_summary()
    assert "UTC" in summary["schedule_note"]
    assert summary["schedule"] == "0 20 * * *"


# --------------------------------------------------------------------------- #
# workflow_commands：一处权威
# --------------------------------------------------------------------------- #


def test_commands_carry_every_gate_threshold() -> None:
    """门禁脚本命令行带上四个阈值（它同时出现在 YAML、教程与终端里）。"""
    commands = workflow_commands(CIConfig())
    assert commands[0] == "python"
    assert commands[1] == GATE_SCRIPT
    joined = " ".join(commands)
    for flag in (
        "--min-pass-rate",
        "--max-adapter-mebibytes",
        "--max-cost-per-1k-tokens",
        "--max-regression",
    ):
        assert flag in joined


def test_commands_track_the_policy_object() -> None:
    """阈值来自 ``ReleaseGates``：改策略 → 命令跟着变（**只有一处权威**）。"""
    config = CIConfig(
        gates=ReleaseGates(
            min_pass_rate=0.75, max_adapter_mebibytes=16.0, max_cost_per_1k_tokens=0.01,
            max_regression=0.05,
        )
    )
    joined = " ".join(workflow_commands(config))
    assert "--min-pass-rate 0.75" in joined
    assert "--max-adapter-mebibytes 16.0" in joined
    assert "--max-cost-per-1k-tokens 0.01" in joined
    assert "--max-regression 0.05" in joined


# --------------------------------------------------------------------------- #
# render_github_actions
# --------------------------------------------------------------------------- #


def test_render_is_deterministic() -> None:
    """渲染是纯函数：同样的配置得到同样的文本。"""
    assert render_github_actions() == render_github_actions(CIConfig())


def test_render_marks_the_file_as_generated() -> None:
    """文件头明确写"由渲染器生成，请勿手工编辑"——否则下一个人会去改 YAML。"""
    text = render_github_actions()
    assert "render_github_actions()" in text
    assert "请勿手工编辑" in text


def test_render_yaml_is_parseable_and_structured() -> None:
    """YAML 可解析，且结构齐全：触发器、作业、六个步骤.

    六步的顺序与理由：checkout → setup-python → 装依赖 →
    **测试（代码没坏）** → **门禁（这一版该不该发）** → 上传产物（``always()``）。
    """
    payload = parse_workflow(render_github_actions())
    assert payload["name"] == "finetune-nightly"
    triggers = payload[True]
    assert triggers["schedule"][0]["cron"] == "0 20 * * *"
    job = payload["jobs"]["finetune-gate"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 30
    steps = job["steps"]
    assert len(steps) == 6
    assert steps[0]["uses"].startswith("actions/checkout")
    assert "setup-python" in steps[1]["uses"]
    assert "pip install" in steps[2]["run"]
    assert "pytest" in steps[3]["run"]
    assert steps[4]["name"] == "Finetune pipeline release gate"
    assert steps[5]["if"] == "always()"


def test_parse_workflow_handles_the_boolean_on_key() -> None:
    """**YAML 1.1 的 ``on`` 陷阱**：裸 ``on`` 会被解析成布尔 ``True``.

    因此 ``payload["on"]`` 会 KeyError，``payload[True]`` 才对。
    这不是本课的 bug——所有解析 GitHub Actions workflow 的脚本都会撞上它，
    所以它被写成一条可运行的证据。
    """
    payload = parse_workflow(render_github_actions())
    assert "on" not in payload
    triggers = payload[True]
    assert triggers["schedule"][0]["cron"] == "0 20 * * *"
    assert "workflow_dispatch" in triggers


def test_render_contains_the_schedule_and_manual_trigger() -> None:
    """两种触发方式都在：定时（持续微调）+ 手动（临时演练）。"""
    text = render_github_actions()
    assert 'cron: "0 20 * * *"' in text
    assert "workflow_dispatch:" in text


def test_render_uploads_artifacts_even_on_failure() -> None:
    """产物上传必须是 ``if: always()``：**失败时那份报告才是最需要看的**."""
    text = render_github_actions()
    assert "if: always()" in text
    assert "outputs/mlops/runs.jsonl" in text
    assert "outputs/registry/versions.jsonl" in text


def test_render_reflects_custom_config() -> None:
    """换一套配置 → 换一份 YAML（语言版本、超时、cron 都跟着走）。"""
    config = CIConfig(
        workflow_name="custom-nightly",
        python_version="3.12",
        schedule="30 1 * * 1",
        timeout_minutes=90,
    )
    text = render_github_actions(config)
    assert "name: custom-nightly" in text
    assert 'python-version: "3.12"' in text
    assert 'cron: "30 1 * * 1"' in text
    assert "timeout-minutes: 90" in text


def test_parse_workflow_rejects_non_mapping() -> None:
    """顶层不是映射 → 报错（而不是让下游在 ``None`` 上继续走）。"""
    with pytest.raises(MLOpsError, match="顶层必须是映射"):
        parse_workflow("- just\n- a\n- list\n")


# --------------------------------------------------------------------------- #
# 与仓库文件的一致性
# --------------------------------------------------------------------------- #


def test_committed_workflow_exists() -> None:
    """仓库里必须有那份 workflow（生成之后要提交，否则 CI 不会跑）。"""
    assert WORKFLOW_PATH.exists(), f"缺少 {WORKFLOW_PATH}"


def test_committed_workflow_matches_the_renderer() -> None:
    """**本文件最重要的一条**：仓库里的 workflow 与渲染器输出逐字相同.

    它让"改门禁阈值"变成一次可追溯的 diff：忘了重新生成，测试立刻变红。
    手写 YAML 的阈值没有任何东西在守护它——它与
    ``settings.mlops_min_pass_rate`` 谁是权威，取决于读的人先看到哪一份。
    """
    committed = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert committed == render_github_actions(CIConfig())


def test_committed_workflow_is_valid_yaml() -> None:
    """提交进仓库的那一份必须是合法 YAML（渲染器的输出不合法就会在这里暴露）."""
    payload = parse_workflow(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert payload["name"] == "finetune-nightly"


def test_repository_already_has_the_day032_workflow() -> None:
    """day032 的 ``ci.yml`` 仍然在：两条流水线分工不同（事件驱动 vs 定时）。"""
    ci_path = PROJECT_ROOT / WORKFLOW_DIRECTORY / "ci.yml"
    assert ci_path.exists()
    payload = parse_workflow(ci_path.read_text(encoding="utf-8"))
    assert payload["name"] == "ci"
    assert "pull_request" in payload[True]
