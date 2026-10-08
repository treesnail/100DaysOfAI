"""``course_index``：把 100 天的材料编成一份**可检索、可复算**的索引（day101）.

走到第 100 天之后，仓库里已经躺着大量**已经写好但不再被读**的文字：
``docs/`` 下 20 份手册、52 个子包的 docstring。今天把它们的最后一点价值榨出来：

```text
不是再写一篇总结，而是把已经写好的总结**编成一份索引**——
让"哪一天讲过 XXX"变成一个可以被检索的问题。
```

## 一、今天最值钱的一句话

> **把材料编成索引，第一步不是"选一个搜索算法"，而是先说清"哪些材料必须进得来"——
> 一份没进语料的文档，与"这门课没有那份文档"在检索结果里读起来完全一样。**

## 二、六个模块

```text
errors.py   六个失败族（**回来的 TokenError** + **继续缺席的 GradientError**）
types.py    2 种语料 / 1 套分词口径 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
corpus.py   从 ``docs/*.md`` 与子包 docstring 收一份语料（名字带种类，绝不撞车）
index.py    分词（ASCII 词 + 中文二元组）+ 倒排索引 + 复算摘要
query.py    一次确定性检索：score = 命中词数 / 查询词数，并列按名字排序
verify.py   七条性质与三类判据（相等 / 上界 / 下界）
study.py    四张表（语料 / 索引 / 检索 / 性质）
__init__.py 本文件
```

## 三、四条纪律

1. **不重写任何子系统**：本包只新增代码；``docs/`` 下的手册是被**读取**的，
   既有模块、既有测试、pyproject 与 docs 下的既有文件**一行未改**。
2. **读数必须现场算出**：``study`` 里不存数字；每张表的每行都来自函数调用。
3. **确定性**：分词零依赖、倒排表显式排序、并列命中按名字兜顺序，
   因此"两次构建逐位相同"与"两次检索逐位相同"都能被一条 `==` 判定。
4. **不联网、不读密钥**：本包只读文件系统与 ``importlib``。

## 四、与既有包的接缝

- **上游**：文件系统（``docs/*.md``）、``importlib``（52 个子包的 docstring）、
  day100 的 ``graduation``（它告诉我们这个仓库里有多少个子包）；
- **脚下**：``config`` **没有**新增配置项——分词窗口、``top_k`` 与金标准都是常量；
- **下游**：``docs/course_index.md`` 与 ``scripts/course_index_demo.py`` 是它的两份外围产物。
"""

from __future__ import annotations

from smart_research_agent.course_index import corpus as _corpus
from smart_research_agent.course_index import errors as _errors
from smart_research_agent.course_index import index as _index
from smart_research_agent.course_index import query as _query
from smart_research_agent.course_index import study as _study
from smart_research_agent.course_index import types as _types
from smart_research_agent.course_index import verify as _verify
from smart_research_agent.course_index.errors import CourseIndexError

#: 本包的七个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _corpus,
    _index,
    _query,
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
    raise CourseIndexError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from course_index import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "corpus",
    "index",
    "query",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise CourseIndexError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from course_index import index` 会拿到函数还是模块取决于导入顺序。"
    )
