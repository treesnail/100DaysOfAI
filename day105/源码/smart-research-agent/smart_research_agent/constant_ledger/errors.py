"""``constant_ledger`` 的失败族：把 100 天钉死的常量编成台账时，每一种失败该谁去修（day105）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批都出现在同一个地方：**把 ``smart_research_agent`` 下几百份 ``.py``
里写的模块级常量（``UPPER_CASE = 值``）收成一份可核对的台账**。

```text
ParseError       解析不成立     某份 .py 读不到 / 语法错                              → 改解析
AssignError      赋值读不下来   常量的右端既不是字面量、也拿不到源码片段（无法定值）    → 改赋值
LedgerBuildError 台账不成立     同一批文件两次编出的台账不同 / 同一模块里同名列两次     → 改台账实现
ConflictError    取值不一致     同一个名字在两个模块里**字面值不同**，却被要求"必须一致" → 改数据或改调用
NumericError     数值不可用     非有限读数 / 非法上下界                                → 改数据或改实现
ParameterError   参数越界       未知值形态 / 未知同名关系 / 未知性质 / 非正上限          → 改调用
ConstantError    本族基类
```

## 一、``AssignError`` 与 ``ConflictError`` 为什么不是同一族

```text
AssignError    "这一条我读不下来"   右端拿不到任何可比较的形态 ⇒ 改赋值（或改成字面量）
ConflictError  "两条对不上"          两边都读下来了，但字面值不同，而调用点要求一致 ⇒ 改数据或改调用
```

前者管**读不读得出值**，后者管**读出来的值是不是一致**。混成一族会让
"有一处写法太复杂"的人去翻整个仓库的常量，而真正**同名不同值**的那一组反而没人去看。

## 二、``ConflictError`` 是本课唯一"按调用点决定"的族

"同名不同值"本身**不是错误**——它是这一课最值钱的一条**读数**（本仓库实测有几十组：

同一类东西在两个包里各取了一个数）。因此本课把它印出来（``conflicts()``），
只在调用点明确要求"这些同名常量必须一致"时才抛 :class:`ConflictError`：

```text
require_consistent(ledger)    ⇒ 有冲突就抛；调用点的**一次决定**，不是本包的默认行为
```

这与 day103 对"环"的处理、day104 对"幽灵"的处理是同一个立场：
**一个读数不该被一条性质静默地删掉，但它可以成为一次显式拒绝交付的理由。**

## 三、``GradientError`` 今天继续缺席

```text
day090 / day094   抛 GradientError
day096 起          缺席：交付 / 索引 / 台账 / 图 / 承诺清单都是**纯静态**的
day105            缺席：本课同样是纯静态——它读模块级常量的赋值语句，一行都没跑
```

## 四、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前面那些包）
├── CourseIndexError（day101） / LedgerError（day102） / GraphError（day103） / SymbolError（day104）
└── ConstantError（day105，本模块）
    ├── ParseError / AssignError / LedgerBuildError / ConflictError
    └── NumericError / ParameterError
```

本层**不继承 day104 的六族**：``CatalogBuildError`` 是"同一批文件编出两份清单"，
而 :class:`LedgerBuildError` 是"同一批文件编出两份台账"——它们处置方向相同，
但**对象不同**（清单是承诺表，台账是取值表）。混成一家会让 ``except SymbolError``
在取值层悄悄多兜住几族。
"""

from __future__ import annotations


class ConstantError(ValueError):
    """``constant_ledger`` 这一族错误的基类（与 ``SymbolError`` / ``GraphError`` 同源）."""


class ParseError(ConstantError):
    """解析不成立：文件读不到、ast 语法错.

    它的处置动作最具体：**改解析**。一份漏掉的 ``.py``
    与"这个模块一个常量都没定义"在台账里读起来完全一样。
    """


class AssignError(ConstantError):
    """赋值读不下来：常量的右端拿不到任何可比较的形态.

    出路是**改赋值**（把它写成字面量，或把它排除在台账之外）。
    一条读不下来的赋值与"这里没有常量"在台账里读起来一样——但它确实写在那里。
    """


class LedgerBuildError(ConstantError):
    """台账不成立：同一批文件两次编出的台账不同，或同一模块里同一个常量名出现两次.

    它是本课唯一一族"内部中间物"的失败。同名列两次这一点很具体：
    一个名字在同一个模块里被赋了两次，读的人就不知道台账该记哪一个数。
    """


class ConflictError(ConstantError):
    """取值不一致：同一个名字在两个模块里**字面值不同**，而调用点要求"必须一致"（见模块 docstring 第二节）.

    它与 :class:`AssignError` 的分界是"读得下来但不同" vs "读不下来"。
    """


class NumericError(ConstantError):
    """数值不可用：非有限读数、上界 / 下界非法（负数）.

    读数非有限时任何比较都会给出一个**静默为假**的结论，因此本包在入口就拒绝。
    """


class ParameterError(ConstantError):
    """参数越界：未知值形态 / 未知同名关系 / 未知性质 / 非正上限.

    出路是**改调用**。本包不接受"关系写错就取个默认值"这种兜底——
    一个被兜住的 ``relation`` 会让"取值一致"与"取值冲突"在报告里长得一模一样。
    """


#: 六个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ParseError": "改解析：某份 .py 读不到或语法错时，先把它补进扫描范围或修好它",
    "AssignError": "改赋值：常量右端读不下来时，把它写成字面量，或显式把它排除在台账之外",
    "LedgerBuildError": "改台账实现：两次构建不一致或同一模块同名列两次时，去掉没有固定的量 / 去掉重复项",
    "ConflictError": "改数据或改调用：同名常量取值不一致时，要么统一取值，要么别要求它们一致",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：值形态 / 同名关系 / 性质名 / 上限都是调用点的一次决定",
}

#: 本模块真正导出的六个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ParseError": ParseError,
    "AssignError": AssignError,
    "LedgerBuildError": LedgerBuildError,
    "ConflictError": ConflictError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise ConstantError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的族：``ShapeError``（day075 第一次命名，day102 曾请它回来过一次）.
RETURNED_FAMILY: str | None = "ShapeError"

#: 回来的理由：**与前面各课"请回旧名字"的理由同源**.
RETURNED_FAMILY_REASON = (
    "``ShapeError`` 由 day075（``transformer_core``）第一次命名，此后被十来个包各写一遍，"
    "是 day102 那份台账里跨包重名之最（34 个模块同时导出它）。"
    "day103（编图）与 day104（编承诺清单）都没有它；day105 换了一个对象，问的却是"
    "同一个问题：\"同一个名字在不同包里**长成了什么形状**？\""
    "——一个 ``EPSILON`` 在 A 包是 ``1e-5``、在 B 包是 ``1e-8``，正是一次形状对不上。"
    "于是这个名字**回来**了，但基类从 ``LedgerError``（day102）换成 ``ConstantError``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余五族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是纯静态分析，一行都没跑）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day100 ~ day104 的理由同源（都是"没有一次反向"）**.
ABSENT_FAMILY_REASON = (
    "本课用 ``ast`` 读几百份 ``.py`` 里模块级常量的赋值语句：一个**纯静态**的解析层，"
    "不但没有反向，连一次 import 都不执行。因此这一族今天不属于本包——它属于被扫描的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "AssignError",
    "ConflictError",
    "ConstantError",
    "LedgerBuildError",
    "NumericError",
    "ParameterError",
    "ParseError",
]
