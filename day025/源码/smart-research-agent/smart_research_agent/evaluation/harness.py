"""评估 Harness 合并基准版：以 day025 完整版为骨架，兼容 day026/027/029/030 简版 API.

各天 harness.py 的 API 出入较大，本模块取并集：

- 主体是 day025 的 ``EvaluationHarness``（load_dataset / evaluate / run(数据集路径) /
  summary() 含 by_tag / write_report）；
- 兼容 day026 风格：子类实现 ``run_case`` 后可 ``run(list[EvalCase])``，
  ``summary(results)`` 对给定结果列表计数；
- 兼容 day027 风格：``EvaluationHarness(fn=..., scorer=...)`` 函数式构造，
  ``load_cases()`` 宽松加载，``run(cases)`` 返回 dict 汇总；
- 兼容 day029 的 ``EvalHarness`` / ``EvalReport``（用例自带 check 函数的套件式执行器）；
- 兼容 day030 的 ``Harness`` / ``HarnessResult`` / ``load_jsonl``（dict 数据集 + runner）。

``EvalCase`` / ``CaseResult`` / ``EvalResult`` 的字段取各天版本的并集，全部带默认值，
因此各天测试里的任何一种关键字构造方式都能成立。
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """一条评估用例（字段为各天版本的并集，全部可选）.

    - id / tags：day025 风格（数据集按 id 标识、按 tag 分组统计）；
    - name：day026 / day029 风格的用例名；
    - input / expected：day025 / day026 / day027 共有，expected 语义由具体 Harness 定义；
    - metadata：day027 风格的附加信息；
    - data / check：day029 风格（check 接收 data，返回 (passed, detail)）。
    """

    id: str = ""
    name: str = ""
    input: str = ""
    expected: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    check: Callable[[dict[str, Any]], tuple[bool, str]] | None = None


@dataclass
class CaseResult:
    """单条用例的评估结果（字段为 day025 与 day026 版本的并集）."""

    case_id: str = ""  # day025
    case: EvalCase | None = None  # day026
    output: str = ""
    score: float = 0.0
    passed: bool = False
    latency_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    detail: str = ""  # day026 的单条问题说明


@dataclass
class EvalResult:
    """简版评估结果（字段为 day027 与 day029 版本的并集）."""

    case: EvalCase | None = None  # day027
    name: str = ""  # day029
    output: str = ""
    score: float = 0.0
    passed: bool = False
    detail: str = ""  # day029


class EvaluationHarness:
    """评估框架基类：定义"加载数据集 → 逐条评估 → 汇总报告"的固定骨架.

    三种用法（对应各天版本）：
      1. day025 风格：子类实现 evaluate()，调用 run(数据集路径)，summary()/write_report() 汇总；
      2. day026 风格：子类实现 run_case()，调用 run(list[EvalCase])，summary(results) 计数；
      3. day027 风格：直接以 fn/scorer 构造，run(list[EvalCase]) 返回 dict 汇总。
    """

    def __init__(
        self,
        fn: Callable[[str], str] | None = None,
        scorer: Callable[[str, str | None], float] | None = None,
        pass_threshold: float = 1.0,
    ) -> None:
        self.fn = fn
        self.scorer = scorer
        self.pass_threshold = pass_threshold
        self.cases: list[EvalCase] = []
        self.results: list[CaseResult] = []

    # ------------------------------------------------------------------
    # 数据集加载
    # ------------------------------------------------------------------

    @staticmethod
    def load_dataset(path: str | Path) -> list[EvalCase]:
        """从 JSONL 文件加载评估数据集（day025 严格版：id/input/expected 必填）."""
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

    @staticmethod
    def load_cases(path: str | Path) -> list[EvalCase]:
        """从 JSONL 文件加载评估数据集（day027 宽松版：只有 input 必填）."""
        cases: list[EvalCase] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    id=str(data.get("id", "")),
                    input=data["input"],
                    expected=data.get("expected"),
                    metadata=data.get("metadata", {}),
                )
            )
        logger.info("加载评估数据集: %s，共 %d 条", path, len(cases))
        return cases

    # ------------------------------------------------------------------
    # 单条评估：子类扩展点
    # ------------------------------------------------------------------

    def evaluate(self, case: EvalCase) -> CaseResult:
        """评估单条用例（day025 风格扩展点），返回 CaseResult。子类必须实现."""
        raise NotImplementedError

    def run_case(self, case: EvalCase) -> CaseResult:
        """执行单条用例（day026 风格扩展点）。默认委托给 evaluate()."""
        return self.evaluate(case)

    # ------------------------------------------------------------------
    # 批量执行
    # ------------------------------------------------------------------

    def run(self, source: str | Path | list[EvalCase]):
        """批量执行用例.

        - fn/scorer 风格（day027）：source 为 list[EvalCase]，返回 dict 汇总；
        - 数据集路径（day025）：加载后逐条 evaluate()，返回 list[CaseResult]；
        - 用例列表（day026）：逐条 run_case()，返回 list[CaseResult]。
        """
        if getattr(self, "fn", None) is not None and isinstance(source, list):
            return self._run_functional(source)
        if isinstance(source, (str, Path)):
            self.cases = self.load_dataset(source)
            self.results = []
            for case in self.cases:
                result = self.evaluate(case)
                self.results.append(result)
                logger.info(
                    "用例 %s: score=%.2f passed=%s", case.id, result.score, result.passed
                )
            return self.results
        results = [self.run_case(case) for case in source]
        self.cases = list(source)
        self.results = results
        passed = sum(1 for r in results if r.passed)
        logger.info("评估完成: %d/%d 通过", passed, len(results))
        return results

    def _run_functional(self, cases: list[EvalCase]) -> dict:
        """day027 风格：fn 执行 + scorer 打分，返回 dict 汇总."""
        results: list[EvalResult] = []
        for case in cases:
            output = self.fn(case.input)
            score = self.scorer(output, case.expected)
            results.append(
                EvalResult(
                    case=case,
                    output=output,
                    score=score,
                    passed=score >= self.pass_threshold,
                )
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

    # ------------------------------------------------------------------
    # 汇总与报告
    # ------------------------------------------------------------------

    def summary(self, results: list | None = None) -> dict[str, Any]:
        """汇总评估结果.

        - 传入 results（day026 风格）：对给定结果列表计数，
          返回 {"total", "passed", "failed", "pass_rate"}；
        - 不传（day025 风格）：基于 self.results 全量汇总，
          含通过率、平均分、按 tag 分组统计、平均延迟。
        """
        if results is not None:
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            return {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(passed / total, 4) if total else 0.0,
            }

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
        """把汇总结果与逐条明细写入 JSON 报告文件（day025 风格）."""
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


# ---------------------------------------------------------------------------
# day029 兼容：套件式简版执行器（用例自带 check 函数）
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """整个套件的汇总报告（day029 风格）."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class EvalHarness:
    """评估执行器（day029 风格）：注册用例、批量运行、汇总报告."""

    def __init__(self) -> None:
        self._cases: list[EvalCase] = []

    def add(self, case: EvalCase) -> None:
        self._cases.append(case)

    def run(self) -> EvalReport:
        report = EvalReport()
        for case in self._cases:
            try:
                passed, detail = case.check(case.data)
            except Exception:  # noqa: BLE001 - 评估器必须隔离用例异常
                passed, detail = False, traceback.format_exc(limit=3)
            report.results.append(EvalResult(name=case.name, passed=passed, detail=detail))
        return report


# ---------------------------------------------------------------------------
# day030 兼容：dict 数据集 + runner 的最小骨架
# ---------------------------------------------------------------------------

# runner 契约：输入一条数据集中的用例（dict），输出该用例的评估结果（dict）
CaseRunner = Callable[[dict], dict]


@dataclass
class HarnessResult:
    """一次评估运行的汇总（day030 风格）."""

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
    """通用评估执行器（day030 风格）：遍历数据集 -> 逐条运行 -> 收集结果与错误."""

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
