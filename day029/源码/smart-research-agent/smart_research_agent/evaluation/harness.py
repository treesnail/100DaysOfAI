"""简版评估执行器：把"用例 + 检查函数"组织成可重复运行的评估套件.

harness 是评估系统的基础设施（harness 一词本义是"挽具"，在软件领域指
"把被测对象固定住、驱动它跑起来的那套脚手架"）。它本身不包含任何
具体指标，只负责三件事：

1. 登记用例（EvalCase：名字 + 输入数据 + 检查函数）；
2. 逐个执行并捕获异常（一个用例失败不拖垮整个套件）；
3. 汇总结果（EvalResult 列表 + 通过率）。
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """一条评估用例.

    check 接收 case.data，返回 (passed, detail)；
    抛出异常视为 failed，detail 记录异常信息。
    """

    name: str
    data: dict[str, Any]
    check: Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class EvalResult:
    """一条用例的执行结果."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalReport:
    """整个套件的汇总报告."""

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
    """评估执行器：注册用例、批量运行、汇总报告."""

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
