"""评估基座：EvalCase / CaseResult / EvaluationHarness.

所有评估器（Prompt 评估、答案质量评估、回归评测等）共享同一套抽象：
用例（EvalCase）进，结果（CaseResult）出，汇总（summary）给出通过率。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """一条评估用例.

    expected 的语义由具体 Harness 定义（期望值、最低分数线等），基类不约束。
    """

    name: str
    input: str
    expected: str | None = None


@dataclass
class CaseResult:
    """一条用例的评估结果."""

    case: EvalCase
    output: str
    passed: bool
    detail: str = ""


class EvaluationHarness(ABC):
    """评估执行器基类：子类只需实现 run_case，批量执行与汇总由基类提供."""

    @abstractmethod
    def run_case(self, case: EvalCase) -> CaseResult:
        """执行单条用例，返回评估结果."""

    def run(self, cases: list[EvalCase]) -> list[CaseResult]:
        """依次执行全部用例."""
        results = [self.run_case(case) for case in cases]
        passed = sum(1 for r in results if r.passed)
        logger.info("评估完成: %d/%d 通过", passed, len(results))
        return results

    @staticmethod
    def summary(results: list[CaseResult]) -> dict:
        """汇总评估结果：总数、通过数、通过率."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        }
