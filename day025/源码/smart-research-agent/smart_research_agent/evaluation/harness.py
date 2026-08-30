"""EvaluationHarness 基类：数据集加载、批量评估与结果汇总."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """一条评估用例：输入 + 期望输出 + 标签."""

    id: str
    input: str
    expected: str
    tags: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """单条用例的评估结果."""

    case_id: str
    output: str
    score: float
    passed: bool
    latency_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


class EvaluationHarness(ABC):
    """评估框架基类：定义"加载数据集 → 逐条评估 → 汇总报告"的固定骨架.

    子类只需实现 evaluate()，说明"如何把一条用例跑出一个 CaseResult"，
    数据加载、批量执行、统计汇总的通用流程由基类负责。
    """

    def __init__(self, pass_threshold: float = 1.0) -> None:
        self.pass_threshold = pass_threshold
        self.cases: list[EvalCase] = []
        self.results: list[CaseResult] = []

    @staticmethod
    def load_dataset(path: str | Path) -> list[EvalCase]:
        """从 JSONL 文件加载评估数据集，每行一个用例."""
        cases: list[EvalCase] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc
                try:
                    cases.append(
                        EvalCase(
                            id=str(record["id"]),
                            input=str(record["input"]),
                            expected=str(record["expected"]),
                            tags=list(record.get("tags", [])),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"第 {line_no} 行缺少必需字段: {exc}") from exc
        logger.info("加载评估数据集 %s，共 %d 条用例", path, len(cases))
        return cases

    @abstractmethod
    def evaluate(self, case: EvalCase) -> CaseResult:
        """评估单条用例，返回 CaseResult。子类必须实现."""

    def run(self, dataset_path: str | Path) -> list[CaseResult]:
        """加载数据集并逐条评估，返回全部结果."""
        self.cases = self.load_dataset(dataset_path)
        self.results = []
        for case in self.cases:
            result = self.evaluate(case)
            self.results.append(result)
            logger.info(
                "用例 %s: score=%.2f passed=%s", case.id, result.score, result.passed
            )
        return self.results

    def summary(self) -> dict[str, Any]:
        """汇总评估结果：总数、通过率、平均分、按 tag 分组统计、平均延迟."""
        total = len(self.results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "avg_latency_seconds": 0.0,
                "by_tag": {},
            }

        passed = sum(1 for r in self.results if r.passed)
        by_tag: dict[str, list[CaseResult]] = {}
        for case, result in zip(self.cases, self.results, strict=True):
            for tag in case.tags:
                by_tag.setdefault(tag, []).append(result)

        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4),
            "avg_score": round(sum(r.score for r in self.results) / total, 4),
            "avg_latency_seconds": round(
                sum(r.latency_seconds for r in self.results) / total, 4
            ),
            "by_tag": {
                tag: {
                    "total": len(group),
                    "passed": sum(1 for r in group if r.passed),
                    "pass_rate": round(
                        sum(1 for r in group if r.passed) / len(group), 4
                    ),
                    "avg_score": round(
                        sum(r.score for r in group) / len(group), 4
                    ),
                }
                for tag, group in sorted(by_tag.items())
            },
        }

    def write_report(self, summary: dict[str, Any], path: str | Path) -> Path:
        """把汇总结果与逐条明细写入 JSON 报告文件."""
        report = {
            "summary": summary,
            "results": [asdict(r) for r in self.results],
        }
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("评估报告已写入 %s", report_path)
        return report_path
