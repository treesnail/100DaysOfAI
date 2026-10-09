"""``summary``：把清单折成一份**可核对的课程总结**（day100 / G2-D1）.

"学习总结"最容易写成一段散文。本模块把它变成一份**结论表**：

```text
100 天        MILESTONE_TOTAL_DAYS（一个可被核对的常数）
53 个子包     inventory.discover_subpackages()（数出来的）
456 个文件    sum(SubpackageInfo.module_count)（数出来的）
1234 个名字   sum(SubpackageInfo.symbol_count)（数出来的）
8 项能力 / 9 个阶段 / 7 条性质 / 4 份交付物（各自从对应包的常量折出来）
```

## 一、今天最值钱的一句话

> **总结与清单必须给出同一个计数：两个数各说各的时，
> 读者只会相信看起来更合理的那一个，而正确的那一个已经无从判断。**

因此 :func:`summary_matches_inventory` 返回的是"两个口径差了几项"（一个可被反驳的整数），
而性质 ``summary_matches_inventory`` 就是"它 == 0"。这不是多余的重复：
总结是**给人读的**、清单是**给程序读的**，两条独立路径给出同一个数，才叫"总结被核对过"
（与 day099 第 ⑥ 条性质"记账 vs 手算"同一套纪律）。

## 二、为什么天数是一个常数，而不是数出来的

```text
子包可以数   仓库里真的有这么多目录
天数不能数   这个仓库里没有"日期"，100 天是一个**声明**（里程碑的定义）
```

因此 :data:`graduation.types.MILESTONE_TOTAL_DAYS` 是常量，并在这里被**复述一次**：
总结里"100 天"与常量相等这条检查，是让"声明"与"交付"对上的唯一办法。

## 三、与既有包的接缝

- **上游**：:mod:`graduation.inventory`（子包名册）、
  ``capstone.types``（能力 / 阶段名单）、:mod:`graduation.types`（性质 / 交付物名单）；
- **下游**：:mod:`graduation.verify` 检查"总结与清单一致"，
  :mod:`graduation.study` 用它打印总结表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.capstone.types import ASSEMBLY_STAGES, CAPABILITY_ORDER
from smart_research_agent.graduation.errors import MilestoneError, NumericError
from smart_research_agent.graduation.inventory import InventoryReport, build_inventory
from smart_research_agent.graduation.types import (
    DELIVERABLE_ORDER,
    GRADUATION_PROPERTIES,
    MILESTONE_TOTAL_DAYS,
)


@dataclass(frozen=True)
class CourseSummary:
    """一份课程总结：八个计数（前四个来自清单，后四个来自各包的常量）.

    ``days`` 是唯一的"声明性"计数（见模块 docstring 第二节）；
    其余七个都是**数出来 / 折出来**的，因此它们都能被一条独立路径复核。
    """

    days: int
    subpackages: int
    modules: int
    symbols: int
    capabilities: int
    stages: int
    properties: int
    deliverables: int

    def __post_init__(self) -> None:
        if self.days != MILESTONE_TOTAL_DAYS:
            raise MilestoneError(
                f"总结里的天数 {self.days} 与里程碑常量 {MILESTONE_TOTAL_DAYS} 不一致："
                "100 天是一个声明，总结必须与它逐位相同——否则'我们走完了 100 天'"
                "与'我们走了 98 天'在报告里读起来一样。"
            )
        for label in ("subpackages", "modules", "symbols", "capabilities", "stages", "properties", "deliverables"):
            value = getattr(self, label)
            if value < 1:
                raise NumericError(
                    f"总结的 {label}={value} 必须 >= 1：一个为 0 的计数说明这一次交付是空的。"
                )

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "days": self.days,
            "subpackages": self.subpackages,
            "modules": self.modules,
            "symbols": self.symbols,
            "capabilities": self.capabilities,
            "stages": self.stages,
            "properties": self.properties,
            "deliverables": self.deliverables,
        }

    def line(self) -> str:
        """一行读数：``总结：100 天 | 子包 53 | 文件 456 | 名字 1234 | 能力 8 | 阶段 9 | 性质 7 | 交付物 4``."""
        return (
            f"总结：{self.days} 天 | 子包 {self.subpackages} | 文件 {self.modules}"
            f" | 名字 {self.symbols} | 能力 {self.capabilities} | 阶段 {self.stages}"
            f" | 性质 {self.properties} | 交付物 {self.deliverables}"
        )


def build_summary(inventory: InventoryReport | None = None) -> CourseSummary:
    """把清单折成一份总结（前四个计数来自清单，后四个来自各包的常量）."""
    resolved = build_inventory() if inventory is None else inventory
    return CourseSummary(
        days=MILESTONE_TOTAL_DAYS,
        subpackages=len(resolved.rows),
        modules=resolved.total_modules,
        symbols=resolved.total_symbols,
        capabilities=len(CAPABILITY_ORDER),
        stages=len(ASSEMBLY_STAGES),
        properties=len(GRADUATION_PROPERTIES),
        deliverables=len(DELIVERABLE_ORDER),
    )


def summary_matches_inventory(summary: CourseSummary, inventory: InventoryReport) -> int:
    """总结与清单在**三个可数计数**上差了几项（0 = 一致）.

    只比"子包数 / 文件数 / 名字数"三个数：它们是两条路径都会算的量。
    ``days`` 不在其中（它是声明，不是数出来的），
    ``capabilities`` / ``stages`` / ``properties`` / ``deliverables`` 也不在其中
    （它们来自常量，与清单无关）。
    """
    pairs = (
        (summary.subpackages, len(inventory.rows)),
        (summary.modules, inventory.total_modules),
        (summary.symbols, inventory.total_symbols),
    )
    return sum(1 for left, right in pairs if left != right)


__all__ = [
    "CourseSummary",
    "build_summary",
    "summary_matches_inventory",
]
