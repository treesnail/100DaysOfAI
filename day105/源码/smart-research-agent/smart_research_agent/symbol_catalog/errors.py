"""``symbol_catalog`` 的失败族：把 100 天的**对外承诺**编成清单时，每一种失败该谁去修（day104）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批都出现在同一个地方：**把 ``smart_research_agent`` 下几百份 ``.py``
里写的 ``__all__``（对外承诺）收成一份可核对的清单**。

```text
ParseError         解析不成立     某份 .py 读不到 / 语法错                            → 改解析
DeclareError       声明不成立     字面量 __all__ 里混进了非字符串（读不成一份名字表）  → 改声明
CatalogBuildError  清单不成立     同一批文件两次编出的清单不同 / 同一模块里重名       → 改清单实现
BindingError       承诺没有落点   声明的名字在本模块里既没定义、也没赋值、也没导入    → 改源码或改声明
NumericError       数值不可用     非有限读数 / 非法上下界                              → 改数据或改实现
ParameterError     参数越界       未知承诺来源 / 未知绑定类别 / 未知性质 / 非正上限    → 改调用
SymbolError        本族基类
```

## 一、``DeclareError`` 与 ``BindingError`` 为什么不是同一族

```text
DeclareError  "这份声明我读不下来"  __all__ 里写的不是一个字符串（例如写成了名字） ⇒ 改声明
BindingError  "这份声明没有落点"    名字读下来了，但本模块里找不到它的定义 / 赋值 / 导入 ⇒ 改源码或改声明
```

前者管**读不读得进来**，后者管**读进来之后有没有落点**。混成一族会让
"``__all__`` 写错了一个字面量"的人去翻整个模块的定义，而真正"名声在外、家里没人"
的那个模块反而没人去看。

## 二、``BindingError`` 是本课唯一"真的会被触发"的族

其余五族都是护栏（读不到 / 写法不认识 / 两次不一致 / 数值非法 / 参数越界）；
只有 ``BindingError`` 回答的是这 100 天里**真实存在**的一件事：

```text
transformer_stack/verify.py 的 __all__ 里写着 "STAGE_ITEMS"
而它在这个模块里既没有 def / class，也没有赋值，也没有 import
```

这不是一处分号笔误：它是一个**声明了却找不到落点的承诺**——
谁写 `from ... import *`，谁就会在那里拿到一个 ``AttributeError``。
本课把这类名字叫**幽灵导出**（phantom export），并把它做成一条**读数**印出来
（像 day103 对"环"的处理一样），而不是让一条"必须为 0"的性质把它遮掉。

## 三、``GradientError`` 今天继续缺席

```text
day090 / day094   抛 GradientError
day096 起          缺席：交付 / 索引 / 台账 / 关系图都是**纯静态**的
day104            缺席：本课同样是纯静态——它读 .py 里的 __all__ 与模块级绑定，一行都没跑
```

## 四、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前面那些包）
├── CourseIndexError（day101） / LedgerError（day102） / GraphError（day103）
└── SymbolError（day104，本模块）
    ├── ParseError / DeclareError / CatalogBuildError / BindingError
    └── NumericError / ParameterError
```

本层**不继承 day103 的六族**：``GraphBuildError`` 是"同一批文件编出两张图"，
而 :class:`CatalogBuildError` 是"同一批文件编出两份清单"——它们处置方向相同，
但**对象不同**（图是关系表，清单是承诺表）。混成一家会让 ``except GraphError``
在承诺层悄悄多兜住几族。
"""

from __future__ import annotations


class SymbolError(ValueError):
    """``symbol_catalog`` 这一族错误的基类（与 ``GraphError`` / ``LedgerError`` 同源）."""


class ParseError(SymbolError):
    """解析不成立：文件读不到、ast 语法错.

    它的处置动作最具体：**改解析**。一份漏掉的 ``.py``
    与"这个模块什么都没承诺"在清单里读起来完全一样。
    """


class DeclareError(SymbolError):
    """声明不成立：字面量 ``__all__`` 里混进了非字符串（读不成一份名字表）.

    出路是**改声明**：一份写不下来的 ``__all__`` 与"这个模块没有 ``__all__``"
    在结果里读起来一样，但它其实**想**说点什么。
    """


class CatalogBuildError(SymbolError):
    """清单不成立：同一批文件两次编出的清单不同，或同一模块里出现重名承诺.

    它是本课唯一一族"内部中间物"的失败。重复声明这一点很具体：
    一个名字在同一个 ``__all__`` 里出现两次，读的人就不知道这一条到底算几条。
    """


class BindingError(SymbolError):
    """承诺没有落点：声明的名字在本模块里既没定义、也没赋值、也没导入（见模块 docstring 第二节）.

    出路是**改源码**（把这个名字真的定义出来）或**改声明**（把它从 ``__all__`` 里删掉）。
    这是本课唯一"真的会被触发"的族。
    """


class NumericError(SymbolError):
    """数值不可用：非有限读数、上界 / 下界非法（负数）.

    读数非有限时任何比较都会给出一个**静默为假**的结论，因此本包在入口就拒绝。
    """


class ParameterError(SymbolError):
    """参数越界：未知承诺来源 / 未知绑定类别 / 未知性质 / 非正上限.

    出路是**改调用**。本包不接受"来源写错就取个默认值"这种兜底——
    一个被兜住的 ``source`` 会让"显式承诺"与"隐式承诺"在报告里长得一模一样。
    """


#: 六个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ParseError": "改解析：某份 .py 读不到或语法错时，先把它补进扫描范围或修好它",
    "DeclareError": "改声明：字面量 __all__ 里混进了非字符串时，把它改成一份名字表",
    "CatalogBuildError": "改清单实现：两次构建不一致或同一模块重名时，去掉没有固定的量 / 去掉重复项",
    "BindingError": "改源码或改声明：承诺没有落点时，要么把它定义出来，要么把它从 __all__ 里删掉",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：承诺来源 / 绑定类别 / 性质名 / 上限都是调用点的一次决定",
}

#: 本模块真正导出的六个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ParseError": ParseError,
    "DeclareError": DeclareError,
    "CatalogBuildError": CatalogBuildError,
    "BindingError": BindingError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise SymbolError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的族：``CoverageError``（day102 用它记"某个子包没有 errors.py"）.
RETURNED_FAMILY: str | None = "CoverageError"

#: 回来的理由：**与前面各课"请回旧名字"的理由同源**.
RETURNED_FAMILY_REASON = (
    "``CoverageError`` 由 day102（失败族台账）第一次命名，它当时记的是"
    "\"某个子包没有 errors.py\"——一句关于**覆盖**的话。"
    "day103（依赖图）没有它。day104 换了一个对象，问的却是同一个问题："
    "\"一个模块在 ``__all__`` 里承诺了 N 个名字，这 N 个名字**落到了哪里**？\""
    "——一个承诺了却没有落点的名字，正是一种覆盖缺口。于是这个名字**回来**了，"
    "但基类从 ``LedgerError`` 换成 ``SymbolError``：今天它要能被同一条 "
    "``except ValueError`` 兜住，与其余五族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是纯静态分析，一行都没跑）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day100 ~ day103 的理由同源（都是"没有一次反向"）**.
ABSENT_FAMILY_REASON = (
    "本课用 ``ast`` 读几百份 ``.py`` 的 ``__all__`` 与模块级绑定：一个**纯静态**的解析层，"
    "不但没有反向，连一次 import 都不执行（它只读 import **语句**，不真的导入）。"
    "因此这一族今天不属于本包——它属于被扫描的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "BindingError",
    "CatalogBuildError",
    "DeclareError",
    "NumericError",
    "ParameterError",
    "ParseError",
    "SymbolError",
]
