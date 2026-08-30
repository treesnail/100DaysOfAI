"""EvaluationHarness：评估框架简版基类（day025 产物）.

职责：加载评估数据集（jsonl）、逐条执行被测函数、调用打分器、汇总结果。
本模块是通用骨架，day026/day027 的具体评估器都在其上构建。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """一条评估用例.

    input: 喂给被测对象的输入（任务/问题文本）
    expected: 期望输出（可选，用于有标准答案的评估）
    metadata: 任意附加信息（来源、难度、标签等）
    """

    input: str
    expected: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """一条用例的评估结果."""

    case: EvalCase
    output: str
    score: float
    passed: bool


class EvaluationHarness:
    """评估框架基类：数据集加载 -> 批量执行 -> 结果汇总.

    参数：
        fn: 被测函数，签名为 fn(input: str) -> str（通常是 agent.run）
        scorer: 打分函数，签名为 scorer(output: str, expected: str | None) -> float（0~1）
        pass_threshold: 判定通过的分数线
    """

    def __init__(
        self,
        fn: Callable[[str], str],
        scorer: Callable[[str, str | None], float],
        pass_threshold: float = 0.6,
    ):
        self.fn = fn
        self.scorer = scorer
        self.pass_threshold = pass_threshold

    @staticmethod
    def load_cases(path: str | Path) -> list[EvalCase]:
        """从 jsonl 文件加载评估数据集，每行一个 {"input": ..., "expected": ...}."""
        cases: list[EvalCase] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    input=data["input"],
                    expected=data.get("expected"),
                    metadata=data.get("metadata", {}),
                )
            )
        logger.info("加载评估数据集: %s，共 %d 条", path, len(cases))
        return cases

    def run(self, cases: list[EvalCase]) -> dict:
        """批量执行用例并汇总.

        返回 {"total": n, "passed": m, "pass_rate": ..., "avg_score": ..., "results": [...]}
        """
        results: list[EvalResult] = []
        for case in cases:
            output = self.fn(case.input)
            score = self.scorer(output, case.expected)
            results.append(
                EvalResult(case=case, output=output, score=score, passed=score >= self.pass_threshold)
            )
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        summary = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_score": sum(r.score for r in results) / total if total else 0.0,
            "results": results,
        }
        logger.info("评估完成: %d/%d 通过，平均分 %.2f", passed, total, summary["avg_score"])
        return summary
