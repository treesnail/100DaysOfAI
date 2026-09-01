"""安全红队评估演示脚本：MockLLM 驱动，全程离线、确定性输出.

运行方式（在项目根目录）：

    python scripts/run_redteam.py

产出：在 data/eval/ 下生成 redteam_report.md（Markdown 红队评估报告）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 保证 import 到的是本目录快照内的 smart_research_agent，而非环境里已装的版本
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation.redteam import RedTeamEvaluator
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.security import PromptInjectionDetector, ToolPermissionPolicy
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

CASES_PATH = PROJECT_ROOT / "data" / "eval" / "redteam_cases.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "redteam_report.md"


class FakeSearchTool(CalculatorTool):
    """离线演示用的假搜索工具：不联网，返回固定文本."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "联网搜索，返回与查询相关的资料摘要"

    def execute(self, expression: str = "", **kwargs) -> str:  # noqa: ARG002
        return "模拟搜索结果：这是离线评估使用的固定回复"


class ShellExecTool(CalculatorTool):
    """高危 shell 工具（演示用桩）：注册在册但不在权限白名单内.

    它存在的意义是让红队用例 abuse-001 走到"权限拒绝"分支：
    工具存在、模型选择调用它，但权限策略在调用前拦截。
    execute 永远不会被执行到。
    """

    @property
    def name(self) -> str:
        return "shell_exec"

    @property
    def description(self) -> str:
        return "在服务器上执行 shell 命令（高危操作）"

    def execute(self, expression: str = "", **kwargs) -> str:  # pragma: no cover
        raise RuntimeError("shell_exec 被权限策略禁用，不应到达这里")


def make_agent(responses: list[str]) -> ReactAgent:
    """装配被测 Agent：全套防线（注入检测 + 白名单权限）."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FakeSearchTool())
    registry.register(ShellExecTool())
    return ReactAgent(
        llm=MockLLM(responses=responses),
        registry=registry,
        max_steps=6,
        injection_detector=PromptInjectionDetector(),
        # 白名单模式：调研助手只需要这两个工具，shell_exec 默认拒绝（失败安全）
        permission_policy=ToolPermissionPolicy(whitelist=["calculator", "web_search"]),
    )


def main() -> int:
    evaluator = RedTeamEvaluator(agent_factory=make_agent)
    report = evaluator.evaluate(cases_path=CASES_PATH)

    print(f"攻击用例总数: {report.total}")
    print(f"被拦截: {report.blocked}，漏网: {len(report.leaked)}")
    print(f"整体拦截率: {report.block_rate:.1%}")
    print("分类拦截率:")
    for category, rate in report.category_block_rates().items():
        print(f"  {category}: {rate:.1%}")
    if report.leaked:
        print("漏网用例:")
        for r in report.leaked:
            print(f"  [{r.case_id}] {r.description}")
    else:
        print("漏网用例: 无")

    REPORT_PATH.write_text(report.render_markdown(), encoding="utf-8")
    print(f"报告已写入: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
