"""``failure_ledger``：把 100 天的失败族编成一份**跨包的台账**（day102）.

day101 把 100 天的**字**编成了一份索引。今天换一个资产：把 100 天的**错误名字**编成一张表。

```text
不是再写一页错误清单，而是把已经散在 36 份 errors.py（35 份历史 + 本课这一份）里的
194 个族收成一张台账——
让"这个失败该谁去修""这一族继承自谁""哪个包没有失败族"变成可以被核对的问题。
```

## 一、今天最值钱的一句话

> **一份没被扫到的 ``errors.py``，与"这个子包没有失败族"在台账里读起来完全一样。
> 一个"看起来像基类名"的字符串，如果没有人能说清它属于哪一类，它就不该被静默地当成类名。**

## 二、七个模块

```text
errors.py   六个失败族（**回来的 ShapeError** + **继续缺席的 GradientError**）
types.py    1 种扫描目标 / 4 类基类归属 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
scan.py     从 ``sm/**/errors.py`` 读原始族与别名表（只用 ast，**不做 import**）
resolve.py  把基类名归属到 local / builtin / imported / unresolved 四类之一
ledger.py   编台账：定身份（root / sub）+ 算继承深度 + 复算摘要
verify.py   七条性质与三类判据（相等 / 上界 / 下界）
study.py    四张表（模块 / 族 / 归属 / 性质）
__init__.py 本文件
```

## 三、四条纪律

1. **不重写任何子系统**：本包只新增代码；被扫描的 ``errors.py`` 是被**读取**的，
   既有模块、既有测试、pyproject 与 docs 下的既有文件**一行未改**。
2. **不做 import**：整个扫描只用标准库 ``ast`` 读文本——因此语法错误会被当场点名，
   而不是被"import 失败"掩盖（见 :mod:`failure_ledger.scan` 的第二节）。
3. **确定性**：族按"包名升序 × 源码顺序"排、归属表按四类固定顺序排，
   因此"两次构建逐位相同"能被一条 ``==`` 判定。
4. **不联网、不读密钥**：本包只读文件系统与 ``ast``。

## 四、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/*/errors.py``）；day101 的 ``course_index``
  教会我们"先收齐材料、再谈算法"，本课把它用在**源码**而不是**文字**上；
- **脚下**：``config`` **没有**新增配置项——扫描目标、白名单与四类归属都是常量；
- **下游**：``docs/failure_ledger.md`` 与 ``scripts/failure_ledger_demo.py`` 是它的两份外围产物。
"""

from __future__ import annotations

from smart_research_agent.failure_ledger import errors as _errors
from smart_research_agent.failure_ledger import ledger as _ledger
from smart_research_agent.failure_ledger import resolve as _resolve
from smart_research_agent.failure_ledger import scan as _scan
from smart_research_agent.failure_ledger import study as _study
from smart_research_agent.failure_ledger import types as _types
from smart_research_agent.failure_ledger import verify as _verify
from smart_research_agent.failure_ledger.errors import LedgerError

#: 本包的七个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _scan,
    _resolve,
    _ledger,
    _verify,
    _study,
)

#: 把七个模块 ``__all__`` 里的名字逐个搬进包命名空间。
#:
#: 故意不手写两份名单（一份 import、一份 ``__all__``）：手写一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: 在报告里长得和"它不存在"一模一样。下面那条导入期不变式代替人来核对这件事。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**七个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise LedgerError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from failure_ledger import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "scan",
    "resolve",
    "ledger",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise LedgerError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from failure_ledger import scan` 会拿到函数还是模块取决于导入顺序。"
    )
