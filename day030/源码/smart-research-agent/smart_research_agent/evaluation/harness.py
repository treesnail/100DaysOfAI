"""EvaluationHarness：评估流水线的最小骨架（简版）.

职责切分：
  - 数据层：load_jsonl 把 JSONL 评估数据集读成 dict 列表；
  - 执行层：Harness.run 遍历数据集，逐条调用 runner 并捕获异常；
  - 汇总层：把每条用例的结果收集成 HarnessResult，交给上层计算指标。

Harness 本身不认识 Agent——它只要求 runner 是 "dict -> dict" 的可调用对象。
Agent 评估（agent_eval.py）就是在这个骨架上把 runner 具体化为"跑一条 Agent 任务"。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# runner 契约：输入一条数据集中的用例（dict），输出该用例的评估结果（dict）
CaseRunner = Callable[[dict], dict]


@dataclass
class HarnessResult:
    """一次评估运行的汇总."""

    total: int = 0
    case_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 数据集：每行一个 JSON 对象，跳过空行，行号计入错误信息."""
    cases: list[dict] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"数据集第 {lineno} 行不是合法 JSON: {exc}") from exc
    return cases


class Harness:
    """通用评估执行器：遍历数据集 -> 逐条运行 -> 收集结果与错误."""

    def run(self, cases: list[dict], runner: CaseRunner) -> HarnessResult:
        result = HarnessResult(total=len(cases))
        for index, case in enumerate(cases, start=1):
            case_id = str(case.get("id", f"case-{index}"))
            try:
                case_result = runner(case)
            except Exception as exc:  # noqa: BLE001
                # 单条用例崩溃不应拖垮整场评估：记录错误，继续跑后面的用例
                logger.warning("用例 %s 评估执行异常: %s", case_id, exc)
                result.errors.append(f"{case_id}: {exc}")
                continue
            case_result.setdefault("id", case_id)
            result.case_results.append(case_result)
        return result
