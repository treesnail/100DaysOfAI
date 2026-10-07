"""``inventory``：用 ``pkgutil`` + ``importlib`` 把**全部**子包数一遍（day100 / G2-D1）.

day099 的 ``manifest`` 只数**12 个候选池**里的子包，因为结业整合要回答的是
"这 8 项能力由哪个包负责"。今天要回答的是另一个问题：

```text
这门课到底交付了多少个子包？它们各自公开了多少个名字？
```

于是本模块做了两件事，都是"数出来"而不是"抄出来"：

```text
① 发现    pkgutil.iter_modules(smart_research_agent.__path__) 遍历目录（不 import）
② 解析    对每个子包 importlib.import_module 一次，数公开符号 + 数子模块
```

## 一、今天最值钱的一句话

> **"数出来"与"抄出来"在报告里长得一样，区别只在代码变化之后：
> 抄一份名册不会随着仓库变化而变化，数一遍的会。**

因此 :class:`InventoryReport` 在 ``__post_init__`` 里做一条闭合检查：
**发现集合 == 记录集合**。少一个子包（漏项）与多一个（重复）都会当场变红。

## 二、为什么"解析失败"要被记录，而不是抛出去

```text
一个 import 失败的子包  →  记录成 exists=False + 第一行错误
抛出去                  →  整份清单崩掉，"还有哪些包"这个问题反而答不出来
```

这与 day099 的 ``manifest.resolve_subpackage`` 是同一条纪律。
真想"宁可报错不可漏算"时调 :func:`require_inventory_clean`。

## 三、与既有包的接缝

- **上游**：``smart_research_agent``（被扫描的包本体）；:mod:`graduation.types`
  （``MILESTONE_TOTAL_DAYS`` 与四份交付物的口径）；
- **下游**：:mod:`graduation.verify` 用它检查"清单既不重复也不漏项"，
  :mod:`graduation.study` 用它打印清单表与总结表。
"""

from __future__ import annotations

import importlib
import pkgutil
import types
from dataclasses import dataclass
from typing import Any

from smart_research_agent.graduation.errors import InventoryError, ParameterError
from smart_research_agent.graduation.types import MILESTONE_TOTAL_DAYS

#: 被扫描的包名（本模块只扫它下面的**一级**子包）.
PACKAGE_NAME = "smart_research_agent"

#: 不在清单里的名字（扫描时跳过）：``__init__`` 与私有模块从来不是"一个交付物".
_IGNORED_PREFIX = "_"


@dataclass(frozen=True)
class SubpackageInfo:
    """一个子包的解析结论：在不在 + 公开了多少个名字 + 有几个子模块（**不抛，只记录**）.

    ``symbol_count`` 与 ``module_count`` 分开报，是因为它们回答两个不同的问题：

    ```text
    symbol_count   这个包对外面**公开了几个名字**（有 __all__ 就信 __all__）
    module_count   这个包里面有**几个文件**（子模块与子包都算）
    ```
    """

    name: str
    exists: bool
    has_all: bool
    symbol_count: int
    module_count: int
    error: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith(_IGNORED_PREFIX):
            raise ParameterError(
                f"子包名 {self.name!r} 不是一个可交付的名字（空或以 _ 开头）。"
            )
        if self.exists and (self.symbol_count < 0 or self.module_count < 0):
            raise ParameterError(f"子包 {self.name!r} 的计数不能为负。")
        if not self.exists and not self.error:
            raise InventoryError(
                f"子包 {self.name!r} 标了'不存在'却没有留错误信息："
                "一份只写'不存在'的清单没有任何下一步动作——要点名它为什么不在。"
            )

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "exists": self.exists,
            "has_all": self.has_all,
            "symbol_count": self.symbol_count,
            "module_count": self.module_count,
            "error": self.error,
        }

    def line(self) -> str:
        """一行读数：``capstone     | 在场 | 公开  52 个名字 | 文件  9 | __all__ 有``."""
        state = "在场" if self.exists else "缺失"
        if not self.exists:
            return f"{self.name:<20} | {state} | {self.error}"
        exported = "有" if self.has_all else "无"
        return (
            f"{self.name:<20} | {state} | 公开 {self.symbol_count:>4} 个名字"
            f" | 文件 {self.module_count:>3} | __all__ {exported}"
        )


def count_public_symbols(module: types.ModuleType) -> int:
    """数一个模块**公开导出的名字**（有 ``__all__`` 就信它，否则按目录数子模块）.

    与 day099 的 ``manifest._public_symbol_count`` 同口径，理由也同一条：
    **确定性、与导入顺序无关**。没有 ``__all__`` 的包改用文件系统口径，
    于是"两个子包谁公开得多"这个问题在任何一次运行里都给出同一个答案。
    """
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return len(set(exported))
    package_paths = getattr(module, "__path__", None)
    if package_paths:
        return sum(1 for _ in pkgutil.iter_modules(package_paths))
    return sum(1 for name in vars(module) if not name.startswith(_IGNORED_PREFIX))


def count_submodules(name: str) -> int:
    """数一个子包目录下有多少个子模块 / 子包（按文件系统，不 import）."""
    module = importlib.import_module(f"{PACKAGE_NAME}.{name}")
    paths = getattr(module, "__path__", None)
    if not paths:
        return 0
    return sum(1 for _ in pkgutil.iter_modules(paths))


def discover_subpackages() -> tuple[str, ...]:
    """发现 ``smart_research_agent`` 下的**全部一级子包**（不 import，按名字排序）.

    排序让结果与文件系统的顺序无关——一个未固定的顺序会让"清单两次相同"
    这条检查变成一次运气。
    """
    package = importlib.import_module(PACKAGE_NAME)
    paths = getattr(package, "__path__", None)
    if not paths:
        raise InventoryError(f"{PACKAGE_NAME} 不是一个包（没有 __path__），无法扫描子包。")
    names = [
        info.name
        for info in pkgutil.iter_modules(paths)
        if info.ispkg and not info.name.startswith(_IGNORED_PREFIX)
    ]
    return tuple(sorted(names))


def resolve_subpackage(name: str) -> SubpackageInfo:
    """解析一个子包：``importlib`` 真的去 import 一次，返回 :class:`SubpackageInfo`.

    "名字对但这个包 import 失败"会被**记录**进 ``error`` 而不抛出——
    这样清单仍然给得出来（见模块 docstring 第二节）。
    """
    if not name or name.startswith(_IGNORED_PREFIX):
        raise ParameterError(f"子包名 {name!r} 不是一个可交付的名字（空或以 _ 开头）。")
    qualified = f"{PACKAGE_NAME}.{name}"
    try:
        module = importlib.import_module(qualified)
    except Exception as error:  # noqa: BLE001 —— 清单里只印第一行
        message = str(error).splitlines()[0] if str(error) else type(error).__name__
        return SubpackageInfo(
            name=name, exists=False, has_all=False, symbol_count=0, module_count=0, error=message
        )
    return SubpackageInfo(
        name=name,
        exists=True,
        has_all=getattr(module, "__all__", None) is not None,
        symbol_count=count_public_symbols(module),
        module_count=count_submodules(name),
    )


@dataclass(frozen=True)
class InventoryReport:
    """一份清单：全部子包逐行 + 一条"发现集合 == 记录集合"的闭合检查.

    ``clean`` 是"既不重复也不漏项"这一个条件——它是性质
    ``inventory_accounts_for_all`` 的读数的来源。
    """

    rows: tuple[SubpackageInfo, ...]
    discovered: tuple[str, ...]

    def __post_init__(self) -> None:
        names = [row.name for row in self.rows]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise InventoryError(
                f"清单里有重复的子包：{duplicated}——"
                "同一个包被数两遍会让汇总计数虚高，而名册读起来毫无破绽。"
            )
        if set(names) != set(self.discovered):
            missing = sorted(set(self.discovered) - set(names))
            extra = sorted(set(names) - set(self.discovered))
            raise InventoryError(
                f"记录集合与发现集合不一致：漏项 {missing}、多出 {extra}——"
                "漏掉的子包既不会出现在名册里，也不会被任何人负责。"
            )

    @property
    def present(self) -> tuple[SubpackageInfo, ...]:
        """在场的子包（``exists=True``）."""
        return tuple(row for row in self.rows if row.exists)

    @property
    def missing(self) -> tuple[str, ...]:
        """解析失败的子包名（应当为空）."""
        return tuple(row.name for row in self.rows if not row.exists)

    @property
    def clean(self) -> bool:
        """是否既不重复也不漏项."""
        return not self.missing and len({row.name for row in self.rows}) == len(self.rows)

    @property
    def total_symbols(self) -> int:
        """全部子包公开名字之和."""
        return sum(row.symbol_count for row in self.rows)

    @property
    def total_modules(self) -> int:
        """全部子包内部文件数之和."""
        return sum(row.module_count for row in self.rows)

    def row_of(self, name: str) -> SubpackageInfo:
        """取一个子包的解析结论（不在名册里当场拒绝）."""
        for row in self.rows:
            if row.name == name:
                return row
        raise ParameterError(f"清单里没有子包 {name!r}。")

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "clean": self.clean,
            "discovered": list(self.discovered),
            "missing": list(self.missing),
            "subpackages": len(self.rows),
            "symbols": self.total_symbols,
            "modules": self.total_modules,
            "rows": [row.to_dict() for row in self.rows],
        }

    def line(self) -> str:
        """一行读数：``子包 53 个 | 在场 53 | 公开名字 1234 | 内部文件 456``."""
        return (
            f"子包 {len(self.rows)} 个 | 在场 {len(self.present)}"
            f" | 公开名字 {self.total_symbols} | 内部文件 {self.total_modules}"
        )


def build_inventory(names: tuple[str, ...] | None = None) -> InventoryReport:
    """扫描并解析全部子包，产出一份 :class:`InventoryReport`（失败**被记录**）.

    ``names`` 可注入（测试用它构造"漏了一个子包"的反例），默认现场发现一遍。
    """
    resolved = discover_subpackages() if names is None else names
    rows = tuple(resolve_subpackage(name) for name in resolved)
    return InventoryReport(rows=rows, discovered=tuple(resolved))


def require_inventory_clean(report: InventoryReport | None = None) -> InventoryReport:
    """清单不干净时抛 :class:`InventoryError`（"宁可报错不可漏算"的那条路）.

    它与 :func:`build_inventory` 的分工是：build 只记录，require 才抛。
    """
    resolved = build_inventory() if report is None else report
    if resolved.clean:
        return resolved
    failures = [row.line() for row in resolved.rows if not row.exists]
    raise InventoryError("清单不干净（有子包解析不到）：" + "；".join(failures))


def inventory_lines(report: InventoryReport | None = None) -> tuple[str, ...]:
    """把清单逐行印出来（每个子包一行 + 一行汇总）."""
    resolved = build_inventory() if report is None else report
    lines = ["清单表（全部一级子包：在场 / 公开名字 / 文件 / __all__）："]
    lines.extend("  " + row.line() for row in resolved.rows)
    lines.append("  " + resolved.line())
    return tuple(lines)


__all__ = [
    "PACKAGE_NAME",
    "InventoryReport",
    "SubpackageInfo",
    "build_inventory",
    "count_public_symbols",
    "count_submodules",
    "discover_subpackages",
    "inventory_lines",
    "require_inventory_clean",
    "resolve_subpackage",
]
