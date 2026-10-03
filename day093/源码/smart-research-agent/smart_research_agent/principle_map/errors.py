"""``principle_map`` 的失败族：**串联里每一种失败该谁去修**（day088 / M7-D12）.

分族的依据与前十三天逐字相同——**按"谁的错、该谁去修"分**。今天这一批是四族全新的，
而它们都出现在同一个地方：**把那十二块拼图拼成一张图**的时候。

```text
ShapeError       形状不符     原理字段不齐、应用与原理的数量对不上、
                              覆盖表的行列数对不上                                → 改调用
ParameterError   参数越界     层名未知、应用未知、top_k 非正、时长非正              → 改调用
NumericError     数值不可用   探针读数非有限、置信上界为负、容差为负                → 改数据或改实现
ClaimError       命题不成立   命题缺证据（没有探针）、或证据不成立（实测越界）      → 改命题或改探针
ReferenceError   引用不成立   命题指向的实现或包不存在（artifact 解析失败）        → 改引用
CoverageError    覆盖不成立   某个应用没有任何原理支撑，或某条原理没有实现落点    → 改图或改提纲
OrderError       次序不成立   分享提纲违反了前置依赖（数学 → 注意力 → 表征 → 推理）→ 改提纲
BridgeError      本族基类
```

## 一、本课**连续第四天缺席**的那一族：``GradientError``

```text
day083   不改任何算术            ⇒ 这一族缺席
day085   只读别人的推理路径      ⇒ 这一族缺席
day086   装进来的层都是别人写好的 ⇒ 这一族缺席
day087   量化本身不可微          ⇒ 这一族缺席
day088   本日一个新式子都没有    ⇒ 这一族缺席（**第四条理由**）
```

今天的理由与那三条都**不同**，而且比它们更日常：

```text
本日不写任何新算法：全部动作是跨包对账与串联——
把既有读数对齐成一张"原理 → 实现 → 应用"的图。
没有手写反向，也没有一个新的前向算术；连一个可微的式子都没有新增，
因此这一族缺席的理由既不是"不改算术"（day083）、也不是"读别人的推理路径"（day085）、
更不是"不可微"（day087）——而是"**本日一个新式子都没有**"。
```

一条纪律在这里第五次兑现：**能"改推导"的地方必须是有人真写了推导的地方。**

## 二、``ClaimError`` 与 ``CoverageError`` 为什么不是同一族

两者的出路完全不同：

```text
ClaimError     "这一条命题本身错了"   命题缺证据 ⇒ 补探针；证据越界 ⇒ 改命题或改实现
CoverageError  "这张图缺一块"         某个应用没有原理支撑 / 某条原理没有落点 ⇒ 改图或改提纲
```

前者是**一条命题的问题**（它自己站不住），后者是**整张图的问题**（某处是空的）。
混成一族会让"补一条探针"的人去改提纲，而"补一节提纲"的人去查探针。

## 三、``ReferenceError`` 为什么值得单独一族

一条命题指向"哪个包的哪个函数"，而这个指向是**可以被检查**的：

```text
artifact 能解析出来 ⇒ 这条原理有实现落点（真的落在某个函数上）
artifact 解析不到   ⇒ 要么名字写错了，要么那个包根本不存在
```

把它归进 ``ClaimError`` 的后果是——"命题错了"与"引用的函数不存在"读起来一样，
而前者要重读论文、后者只要把名字改对。**"找不到"与"不成立"必须分开报。**

## 四、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── ...
├── OptimError（day087）
└── BridgeError（day088，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    └── ClaimError / ReferenceError / CoverageError / OrderError
```

本层同样**继承 day075 的三族而不是彼此**：``hf_source``、``hf_integration``、
``inference_optim`` 与本包是四个**兄弟**。因此
``except transformer_core.errors.ShapeError`` 能同时兜住八层，
而 ``except inference_optim.errors.ShapeError`` **兜不住本层**——
这一条有它自己的用处：四天的读数在报告里必须能分开统计。
"""

from __future__ import annotations

from smart_research_agent.transformer_core.errors import (
    NumericError as CoreNumericError,
)
from smart_research_agent.transformer_core.errors import (
    ParameterError as CoreParameterError,
)
from smart_research_agent.transformer_core.errors import (
    ShapeError as CoreShapeError,
)


class BridgeError(ValueError):
    """``principle_map`` 这一族错误的基类（判据与 ``SourceError`` / ``OptimError`` 同源）."""


class ShapeError(CoreShapeError, BridgeError):
    """形状不符：原理字段不齐、应用与原理的数量对不上、覆盖表的行列数对不上.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前十三天写的 ``except ShapeError`` 代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, BridgeError):
    """参数越界：层名未知、应用未知、``top_k`` 非正、时长非正.

    出路是**改调用**。本包不接受"层名写错就取个默认层"这种兜底——
    一个被兜住的 ``layer="atenntion"`` 会被就近取成某一层，
    而它与"我确实写了那一层"在读图上**完全一样**。
    """


class NumericError(CoreNumericError, BridgeError):
    """数值不可用：探针读数非有限、置信上界为负、容差为负.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝，而不是让它以"这条性质不通过"的形式出现在报告里。
    """


class ClaimError(BridgeError):
    """命题不成立：命题缺证据（没有对应探针）、或证据不成立（实测越界）.

    它与 :class:`CoverageError` 的差别在**尺度层**：
    前者是"这一条命题自己站不住"，后者是"整张图缺一块"。
    出路是**改命题或改探针**——一条站不住的命题不该被删掉，而该被改写。
    """


class ReferenceError(BridgeError):
    """引用不成立：命题指向的实现或包不存在（``artifact`` 解析失败）.

    它值得独立成族，因为它的出路最轻：**改引用**。把它归进 :class:`ClaimError`
    的后果是——"命题错了"与"引用的函数名不存在"读起来一样，
    而前者要重读论文、后者只要把名字改对。
    """


class CoverageError(BridgeError):
    """覆盖不成立：某个应用没有任何原理支撑，或某条原理没有实现落点.

    消息里必须指出**缺的是哪个应用或哪条原理**——"覆盖不全"这句话
    没有任何下一步动作，而"``serving`` 这条应用没有原理支撑"有。
    """


class OrderError(BridgeError):
    """次序不成立：分享提纲违反了前置依赖（数学 → 注意力 → 表征 → 推理）.

    它与 :class:`CoverageError` 的差别在**顺序**：图可以是对的（每一块都在），
    而提纲把它们讲反了——先讲部署再讲数学，听众会在第一页就掉队。
    """


#: 七个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：原理字段、应用与原理的数量、覆盖表的行列都要先对齐",
    "ParameterError": "改调用：层名 / 应用名 / top_k / 时长都是调用点的一次决定",
    "NumericError": "改数据或改实现：非有限读数、负上界、负容差都属于'数值不可用'",
    "ClaimError": "改命题或改探针：一条站不住的命题要改写，而不是删掉",
    "ReferenceError": "改引用：命题指向的模块或函数不存在时，先把名字改对",
    "CoverageError": "改图或改提纲：某个应用没有原理支撑、或某条原理没有落点时补上那一块",
    "OrderError": "改提纲：前置依赖（数学 → 注意力 → 表征 → 推理）不能被讲反",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "ClaimError": ClaimError,
    "ReferenceError": ReferenceError,
    "CoverageError": CoverageError,
    "OrderError": OrderError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise BridgeError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 本课**连续第四天缺席**的那一族（写进常量，让"缺席"是一个可断言的事实而不是一段散文）.
ABSENT_FAMILY = "GradientError"

#: 缺席的理由（与 day085/day086/day087 的那三条**不同**，这一条要写清楚）.
ABSENT_FAMILY_REASON = (
    "本日不写任何新算法：全部动作是跨包对账与串联——把既有读数对齐成一张"
    "'原理 → 实现 → 应用'的图。没有手写反向，也没有一个新的前向算术；"
    "连一个可微的式子都没有新增，因此这一族缺席的理由既不是'不改算术'（day083）、"
    "也不是'只读别人的推理路径'（day085）、更不是'不可微'（day087）——"
    "而是'本日一个新式子都没有'。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "BridgeError",
    "ClaimError",
    "CoverageError",
    "NumericError",
    "OrderError",
    "ParameterError",
    "ReferenceError",
    "ShapeError",
]
