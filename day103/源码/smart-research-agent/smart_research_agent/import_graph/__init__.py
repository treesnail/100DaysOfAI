"""``import_graph``：把 100 天的模块编成一张**依赖图**（day103）.

day101 把 100 天的**字**编成索引，day102 把**错误名字**编成台账。
今天换第三个资产：把 **450 个模块之间的 import 关系**编成一张有向图。

```text
不是再画一张架构图，而是把已经写在 450 份 .py 里的 import 语句收成一张图——
让"我依赖谁""动了我谁受影响""哪两个包互相咬住"变成可以被核对的问题。
```

## 一、今天最值钱的一句话

> **依赖图天生可能不是 DAG——所以"线性顺序"这句话必须先说清"对谁而言"。
> 450 个模块里就有 4 个环；谁都不说清，报告里那两个数字就会互相矛盾。**

## 二、七个模块

```text
errors.py   六个失败族（**回来的 AssemblyError** + **继续缺席的 GradientError**）
types.py    1 种扫描目标 / 2 类边 / 2 个方向 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
parse.py    从 ``sm/**/*.py`` 读节点与 import 目标（只用 ast，**不做 import**）
graph.py    建图 + 强连通分量（Kosaraju）+ 凝缩 + 线性序（Kahn）+ 闭包
verify.py   七条性质与三类判据（相等 / 上界 / 下界）
study.py    四张表（模块 / 环 / 边 / 性质）
__init__.py 本文件
```

## 三、四条纪律

1. **不重写任何子系统**：本包只新增代码；被扫描的 ``.py`` 是被**读取**的，
   既有模块、既有测试、pyproject 与 docs 下的既有文件**一行未改**。
2. **不做 import**：整个解析只用标准库 ``ast`` 读文本——因此语法错误会被当场点名，
   而一个**函数体里的延迟 import** 也会被如实记成一条边（这正是 ``api ↔ tools`` 那个环的来处）。
3. **确定性**：节点排序、边排序、分量排序、Kahn 取点用最小堆，四处显式排序，
   因此"两次构建逐位相同"能被一条 ``==`` 判定。
4. **不联网、不读密钥**：本包只读文件系统与 ``ast``。

## 四、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/**/*.py``）；day102 的 ``failure_ledger``
  教会我们"用 ast 读源码、不做 import"，本课把它用在**关系**而不是**名字**上；
- **脚下**：``config`` **没有**新增配置项——扫描目标、两类边与两个方向都是常量；
- **下游**：``docs/import_graph.md`` 与 ``scripts/import_graph_demo.py`` 是它的两份外围产物。
"""

from __future__ import annotations

from smart_research_agent.import_graph import errors as _errors
from smart_research_agent.import_graph import graph as _graph
from smart_research_agent.import_graph import parse as _parse
from smart_research_agent.import_graph import study as _study
from smart_research_agent.import_graph import types as _types
from smart_research_agent.import_graph import verify as _verify
from smart_research_agent.import_graph.errors import GraphError

#: 本包的六个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _parse,
    _graph,
    _verify,
    _study,
)

#: 把六个模块 ``__all__`` 里的名字逐个搬进包命名空间。
#:
#: 故意不手写两份名单（一份 import、一份 ``__all__``）：手写一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: 在报告里长得和"它不存在"一模一样。下面那条导入期不变式代替人来核对这件事。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**六个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise GraphError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from import_graph import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "parse",
    "graph",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise GraphError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from import_graph import graph` 会拿到函数还是模块取决于导入顺序。"
    )
