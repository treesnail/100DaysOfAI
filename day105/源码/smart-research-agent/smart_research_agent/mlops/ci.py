"""CI 集成：把这条流水线接进 GitHub Actions（M5-D10）.

day032 已经给仓库装上了第一条 CI（``.github/workflows/ci.yml``）：单元测试
+ Agent 评估 + 红队评估三级门禁。那条流水线管的是"**代码改动有没有破坏
现有能力**"，触发条件是 push / pull_request。

今天要接的是另一种流水线：**领域数据的持续微调**。它与 CI 的四个差别
决定了它不能塞进原来的 workflow：

| | CI（day032） | 持续微调（今天） |
|---|---|---|
| 触发 | push / PR（事件驱动） | 定时（cron）+ 手动 |
| 耗时 | 分钟级 | 十分钟级以上（真实训练更久） |
| 输入 | 代码 | 代码 + 数据 + 上一版模型 |
| 失败意味着 | "这次提交不能合" | "这一版不值得上线" |

最后一行是关键的语义差异：**微调流水线失败不是"代码坏了"，而是"这一版
产物不该上线"**——所以它的门禁结论要能被下载与评审，而不是只留一个红灯。

## 为什么 workflow 是**生成**的而不是手写的

`render_github_actions()` 生成的 YAML 会被直接写进仓库
（``.github/workflows/finetune-nightly.yml``），并有一条测试断言
**"仓库里的那份文件 == 渲染器的输出"**。

这条断言让"改门禁阈值"变成一次**可追溯的 diff**：
改了 ``ReleaseGates.min_pass_rate``，workflow 里的 ``--min-pass-rate``
会跟着变；忘了重新生成，测试立刻变红。

手写 YAML 的问题是反过来的：阈值会**在 YAML 里悄悄漂移**，
而 YAML 里那个数字没有任何东西在守护它——它与
``settings.mlops_min_pass_rate`` 谁是权威，取决于读的人先看到哪一份。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.mlops.errors import MLOpsError
from smart_research_agent.mlops.gates import ReleaseGates

#: 生成出来的 workflow 文件名与路径（相对仓库根）。
WORKFLOW_FILENAME = "finetune-nightly.yml"
WORKFLOW_DIRECTORY = ".github/workflows"

#: 门禁脚本名（workflow 会调用它）。
GATE_SCRIPT = "scripts/mlops_demo.py"


@dataclass(frozen=True)
class CIConfig:
    """生成 workflow 所需的全部参数.

    ``schedule`` 用的是 **UTC cron**：GitHub Actions 的 ``schedule`` 一律按
    UTC 解释，写成本地时间会出现"明明配的凌晨 2 点，跑在下午 2 点"这种事。
    缺省 ``0 20 * * *`` = UTC 20:00 = 北京时间次日 04:00（避开业务高峰）。
    """

    workflow_name: str = "finetune-nightly"
    runs_on: str = "ubuntu-latest"
    python_version: str = "3.11"
    schedule: str = "0 20 * * *"
    timeout_minutes: int = 30
    gates: ReleaseGates = field(default_factory=ReleaseGates)
    artifact_paths: tuple[str, ...] = (
        "outputs/mlops/runs.jsonl",
        "outputs/registry/versions.jsonl",
    )
    gate_script: str = GATE_SCRIPT

    def __post_init__(self) -> None:
        if not self.workflow_name.strip():
            raise MLOpsError("workflow_name 不能为空")
        parts = self.schedule.split()
        if len(parts) != 5:
            raise MLOpsError(
                f"schedule 必须是五段式 cron（分 时 日 月 周），收到 {self.schedule!r}"
                "；注意 GitHub Actions 一律按 UTC 解释它"
            )
        if self.timeout_minutes <= 0:
            raise MLOpsError(f"timeout_minutes 必须为正数，收到 {self.timeout_minutes}")
        if not self.artifact_paths:
            raise MLOpsError("artifact_paths 不能为空：失败时的报告必须能下载复盘")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "workflow_name": self.workflow_name,
            "runs_on": self.runs_on,
            "python_version": self.python_version,
            "schedule": self.schedule,
            "timeout_minutes": self.timeout_minutes,
            "gates": self.gates.to_dict(),
            "artifact_paths": list(self.artifact_paths),
            "gate_script": self.gate_script,
        }


def workflow_commands(config: CIConfig | None = None) -> list[str]:
    """门禁脚本命令行（workflow 里那一行，也是手动复现的那一行）.

    刻意把命令**单独抽出来**：它同时出现在三个地方——生成的 YAML、
    教程、以及出问题时人在终端里敲的那一行。三处各写一遍，
    就会有三份会漂移的阈值。
    """
    resolved = config or CIConfig()
    gates = resolved.gates
    return [
        "python",
        resolved.gate_script,
        "--min-pass-rate",
        str(gates.min_pass_rate),
        "--max-adapter-mebibytes",
        str(gates.max_adapter_mebibytes),
        "--max-cost-per-1k-tokens",
        str(gates.max_cost_per_1k_tokens),
        "--max-regression",
        str(gates.max_regression),
    ]


def render_github_actions(config: CIConfig | None = None) -> str:
    """渲染一份可提交的 GitHub Actions workflow（YAML 文本）.

    五个步骤，顺序即策略（第五次复用同一种推理）：

    1. **checkout + setup-python**：标准动作；
    2. **安装依赖**：``pip install -e ".[dev]"``；
    3. **单元测试 + 覆盖率门禁**：`fail_under=90` 在 ``pyproject.toml`` 里，
       覆盖率不达标时 pytest 直接非零退出——**这一步保护的是"代码没坏"**；
    4. **运行门禁脚本**：生成模型卡与发布清单，并按阈值判定，
       **这一步保护的是"这一版产物不该上线"**；
    5. **上传产物（``if: always()``）**：失败时也要上传，因为
       **失败时那份报告才是最需要看的东西**。
    """
    resolved = config or CIConfig()
    command = " ".join(workflow_commands(resolved))
    paths = "\n".join(f"            {path}" for path in resolved.artifact_paths)
    return f"""# 领域微调持续运行（M5-D10 / day059）
#
# 本文件由 smart_research_agent.mlops.ci.render_github_actions() 生成，
# 请勿手工编辑：改门禁阈值请改 ReleaseGates，然后重新生成并提交。
# 测试 test_mlops_ci.py::test_committed_workflow_matches_the_renderer
# 会断言仓库里的这一份与渲染器的输出逐字相同。
#
# 触发：定时（cron 按 UTC 解释）+ 手动。
# 与 ci.yml 的分工：ci.yml 管"代码改动有没有破坏现有能力"（事件驱动），
# 本文件管"这一版产物值不值得上线"（定时驱动）。
# 失败的含义也因此不同：后者失败不是"代码坏了"，而是"这一版不该发布"。

name: {resolved.workflow_name}

on:
  schedule:
    - cron: "{resolved.schedule}"
  workflow_dispatch:

jobs:
  finetune-gate:
    runs-on: {resolved.runs_on}
    timeout-minutes: {resolved.timeout_minutes}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "{resolved.python_version}"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install -r requirements.txt

      # 第一步门禁：单元测试 + 分支覆盖率（fail_under 配置在 pyproject.toml，
      # 覆盖率不达标时 pytest 以非零退出码结束，本步直接失败）
      - name: Unit tests with coverage gate
        run: python -m pytest tests/ -q

      # 第二步门禁：跑一遍微调流水线并按发布门禁判定。
      # 阈值全部来自 ReleaseGates，因此它们与代码里的缺省值永远是同一份。
      - name: Finetune pipeline release gate
        run: {command}

      # 报告作为构建产物留存：失败时的那份报告才是最需要看的东西
      - name: Upload run and registry artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: finetune-artifacts
          path: |
{paths}
"""


def parse_workflow(text: str) -> dict[str, Any]:
    """解析 workflow YAML（测试用它做结构断言）.

    ``pyyaml`` 从 day052 起已被显式声明为 dev 依赖（原因见
    ``pyproject.toml`` 的注释：**依赖树一变，基于它的断言就会以
    ImportError 的形式失效**）。今天再用一次，理由相同。
    """
    import yaml

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - 渲染器输出不会产生非法 YAML
        raise MLOpsError(f"workflow 不是合法 YAML：{exc}") from exc
    if not isinstance(payload, dict):
        raise MLOpsError("workflow 的顶层必须是映射")
    return payload


def workflow_summary(config: CIConfig | None = None) -> dict[str, Any]:
    """给出一份"这份 workflow 会在什么时候、按什么阈值做什么"的摘要.

    它是 API 端点与文档共用的自我描述，因此阈值同样从
    ``ReleaseGates`` 现场读出。
    """
    resolved = config or CIConfig()
    return {
        "workflow_name": resolved.workflow_name,
        "path": f"{WORKFLOW_DIRECTORY}/{WORKFLOW_FILENAME}",
        "schedule": resolved.schedule,
        "schedule_note": "GitHub Actions 的 schedule 一律按 UTC 解释",
        "python_version": resolved.python_version,
        "timeout_minutes": resolved.timeout_minutes,
        "gate_command": workflow_commands(resolved),
        "gates": resolved.gates.to_dict(),
        "artifact_paths": list(resolved.artifact_paths),
    }


__all__ = [
    "GATE_SCRIPT",
    "WORKFLOW_DIRECTORY",
    "WORKFLOW_FILENAME",
    "CIConfig",
    "parse_workflow",
    "render_github_actions",
    "workflow_commands",
    "workflow_summary",
]
