"""``symbol_catalog``：把 100 天的**对外承诺**编成一份可核对的清单（day104）.

day101 把 100 天的**字**编成索引，day102 把**错误名字**编成台账，day103 把**模块关系**编成图。
今天换第四个资产：把 **几百份 ``.py`` 里写的 ``__all__``（对外承诺）** 编成一份清单。

```text
不是再写一份 API 文档，而是把已经写在源码里的 __all__ 收成一份可核对的表——
让"这个模块承诺了什么""这些承诺落到了哪里""哪个名字被很多模块同时承诺"变成可以被核对的问题。
```

## 一、今天最值钱的一句话

> **一个模块的 ``__all__`` 是它对外的承诺；一个"承诺了却找不到落点"的名字，
> 比一个报错更难被发现——因为 ``__all__`` 读起来永远像一份已经兑现的清单。**

## 二、七个模块

```text
errors.py    六个失败族（**回来的 CoverageError** + **继续缺席的 GradientError**）
types.py     1 种扫描目标 / 3 种承诺来源 / 4 类绑定 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
parse.py     从 ``sm/**/*.py`` 读承诺与模块级绑定（只用 ast，**不做 import**）
catalog.py   编清单：每个模块一行 + 重名表 + 幽灵表；可复算口径与摘要
verify.py    七条性质与三类判据（相等 / 上界 / 下界）
study.py     四张表（模块 / 幽灵 / 重名 / 性质）
__init__.py  本文件
```

## 三、四条纪律

1. **不重写任何子系统**：本包只新增代码；被扫描的 ``.py`` 是被**读取**的，
   既有模块、既有测试、pyproject 与 docs 下的既有文件**一行未改**。
2. **不做 import**：整个解析只用标准库 ``ast`` 读文本——因此语法错误会被当场点名，
   而一句写在 ``try`` 里的 ``from x import Y`` 也会被如实记成一次绑定。
3. **确定性**：模块排序、名字排序、重名表按（出现次数, 名字）排序，三处显式排序，
   因此"两次构建逐位相同"能被一条 ``==`` 判定。
4. **不联网、不读密钥**：本包只读文件系统与 ``ast``。

## 四、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/**/*.py``）；day102 的 ``failure_ledger``
  教会我们"用 ast 读源码、不做 import"，day103 的 ``import_graph`` 教会我们"把并列显式排序"，
  本课把它们用在**承诺**而不是**名字 / 关系**上；
- **脚下**：``config`` **没有**新增配置项——扫描目标、三种来源与四类绑定都是常量；
- **下游**：``docs/symbol_catalog.md`` 与 ``scripts/symbol_catalog_demo.py`` 是它的两份外围产物。
"""

from __future__ import annotations

from smart_research_agent.symbol_catalog import catalog as _catalog
from smart_research_agent.symbol_catalog import errors as _errors
from smart_research_agent.symbol_catalog import parse as _parse
from smart_research_agent.symbol_catalog import study as _study
from smart_research_agent.symbol_catalog import types as _types
from smart_research_agent.symbol_catalog import verify as _verify
from smart_research_agent.symbol_catalog.errors import SymbolError

#: 本包的六个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _parse,
    _catalog,
    _verify,
    _study,
)

#: 把六个模块 ``__all__`` 里的名字逐个搬进包命名空间。
#:
#: 故意不手写两份名单（一份 import、一份 ``__all__``）：手写一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: ——正是本课要抓的那种"幽灵导出"——在报告里长得和"它不存在"一模一样。
#: 下面那条导入期不变式代替人来核对这件事。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**六个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise SymbolError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from symbol_catalog import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "parse",
    "catalog",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise SymbolError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from symbol_catalog import catalog` 会拿到函数还是模块取决于导入顺序。"
    )
