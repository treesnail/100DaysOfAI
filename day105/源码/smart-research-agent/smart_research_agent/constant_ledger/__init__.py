"""``constant_ledger``：把 100 天钉死的常量编成一份可核对的台账（day105）.

day101 把 100 天的**字**编成索引，day102 把**错误名字**编成台账，day103 把**模块关系**编成图，
day104 把**对外承诺**编成清单。今天换第五个资产：把 **几百份 ``.py`` 里模块级写的
``UPPER_CASE = 值``（被钉死的常量）** 编成一份台账。

```text
不是再列一张"配置表"，而是把已经写在源码里的模块级常量收成一份可核对的表——
让"这个仓库把哪些名字钉成了什么值""同名常量在不同包里是不是同一个值"变成可以被核对的问题。
```

## 一、今天最值钱的一句话

> **"同名不同值"本身不是错误，它是一个读数——
> 但它是这一课唯一一条能让人立刻去翻两个包的读数。**

## 二、七个模块

```text
errors.py    六个失败族（**回来的 ShapeError** + **继续缺席的 GradientError**）
types.py     1 种扫描目标 / 2 类取值形态 / 4 类同名关系 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
parse.py     从 ``sm/**/*.py`` 读模块级常量与取值指纹（只用 ast + literal_eval，**不做 import**）
ledger.py    编台账：每个模块一行 + 同名表 + 冲突表；可复算口径与摘要
verify.py    七条性质与三类判据（相等 / 上界 / 下界）
study.py     四张表（模块 / 同名 / 冲突 / 性质）
__init__.py  本文件
```

## 三、四条纪律

1. **不重写任何子系统**：本包只新增代码；被扫描的 ``.py`` 是被**读取**的，
   既有模块、既有测试、pyproject 与 docs 下的既有文件**一行未改**。
2. **不做 import、不执行被扫描的表达式**：取值指纹只走 ``ast.literal_eval``，
   因此 ``Path(__file__)`` 这类右端被记成 opaque，而不是被求值。
3. **确定性**：模块排序、常量按（名字, 行号）排序、同名组按（出现模块数, 名字）排序，
   三处显式排序；**集合字面量先排序再拼接**（`set` 的 `repr` 顺序随进程变化），
   因此"两次构建逐位相同"能被一条 ``==`` 判定。
4. **不联网、不读密钥**：本包只读文件系统与 ``ast``。

## 四、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/**/*.py``）；day103 的 ``import_graph``
  教会我们"把并列显式排序"，day104 的 ``symbol_catalog`` 教会我们"给每个名字一个去向"，
  本课把它们用在**取值**而不是**关系 / 承诺**上；
- **脚下**：``config`` **没有**新增配置项——扫描目标、两类形态与四类关系都是常量；
- **下游**：``docs/constant_ledger.md`` 与 ``scripts/constant_ledger_demo.py`` 是它的两份外围产物。
"""

from __future__ import annotations

from smart_research_agent.constant_ledger import errors as _errors
from smart_research_agent.constant_ledger import ledger as _ledger
from smart_research_agent.constant_ledger import parse as _parse
from smart_research_agent.constant_ledger import study as _study
from smart_research_agent.constant_ledger import types as _types
from smart_research_agent.constant_ledger import verify as _verify
from smart_research_agent.constant_ledger.errors import ConstantError

#: 本包的六个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _parse,
    _ledger,
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
    raise ConstantError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from constant_ledger import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "parse",
    "ledger",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise ConstantError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from constant_ledger import ledger` 会拿到函数还是模块取决于导入顺序。"
    )
