"""``import_graph`` 的失败族：把 100 天的模块编成一张依赖图时，每一种失败该谁去修（day103）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批都出现在同一个地方：**把 ``smart_research_agent`` 下 450 份 ``.py``
编成一张有向图**。

```text
ParseError        解析不成立     某份 .py 读不到 / 语法错                               → 改解析
ImportTargetError 目标不成立     一条包内 import 的目标匹配不到任何模块 / 包             → 改解析或改源码
GraphBuildError   图不成立       边端点不在节点集里 / 同一批文件两次构建给出不同的图      → 改图实现
CycleError        线性序不成立   图里有环，却被要求给出一个线性顺序                      → 改调用或改层
NumericError      数值不可用     非有限读数 / 非法上下界                                → 改数据或改实现
ParameterError    参数越界       未知边种类 / 未知方向 / 未知性质 / 非正上限              → 改调用
GraphError        本族基类
```

## 一、``ParseError`` 与 ``ImportTargetError`` 为什么不是同一族

```text
ParseError         "材料读不进来"   文件读不到 / ast 语法错 ⇒ 改解析（把那一份补进来或修好它）
ImportTargetError  "读进来对不上号" 一条 import 的包内目标谁也对不上 ⇒ 改解析规则或改源码
```

前者管**输入**，后者管**归属**。混成一族会让"少扫了一份文件"的人去查相对导入的解析规则，
而"相对导入少减了一层点"的人去查文件系统。

## 二、``CycleError`` 为什么必须独立成族

依赖图**天生可能不是 DAG**。本仓库里就真的有 4 个环（其中一个是跨包的 ``api ↔ tools``）。
把"你要一个线性序、但图里有环"记成 ``GraphBuildError`` 会让读的人以为图画错了，
而真相是**调用点问错了问题**：有环的图只能给**凝缩后的**线性序。
因此 :func:`graph.require_acyclic` 单独抛这一族，并在消息里把环点名。

## 三、``GradientError`` 今天继续缺席

```text
day090 / day094   抛 GradientError
day096 / day099   缺席：本日没有一次反向
day100 / day101   缺席：交付与归档 / 编索引与检索
day102            缺席：纯静态扫描（ast）
day103            缺席：**同样是纯静态**——它读 450 份 .py 的 import 语句，一行都没跑
```

## 四、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前面那些包）
├── CapstoneError（day099） / GraduationError（day100）
├── CourseIndexError（day101） / LedgerError（day102）
└── GraphError（day103，本模块）
    ├── ParseError / ImportTargetError / GraphBuildError / CycleError
    └── NumericError / ParameterError
```

本层**不继承 day102 的六族**：``LedgerBuildError`` 是"同一批文件编出两份台账"，
而 :class:`GraphBuildError` 是"同一批文件编出两张图"——它们处置方向相同，
但**对象不同**（台账是名字表，图是关系表）。混成一家会让 ``except LedgerError``
在图层悄悄多兜住几族。
"""

from __future__ import annotations


class GraphError(ValueError):
    """``import_graph`` 这一族错误的基类（与 ``LedgerError`` / ``CourseIndexError`` 同源）."""


class ParseError(GraphError):
    """解析不成立：文件读不到、ast 语法错.

    它的处置动作最具体：**改解析**。一份漏掉的 ``.py``
    与"这个模块一条依赖都没有"在依赖图里读起来完全一样。
    """


class ImportTargetError(GraphError):
    """目标不成立：一条包内 import 的目标匹配不到任何模块或包.

    出路是**改解析规则**（相对导入少减了一层点？）或**改源码**。
    """


class GraphBuildError(GraphError):
    """图不成立：边端点不在节点集里，或同一批文件两次构建给出不同的图.

    它是本课唯一一族"内部中间物"的失败（见模块 docstring 第二节）。
    """


class CycleError(GraphError):
    """线性序不成立：图里有环，却被要求给出一个线性顺序（见模块 docstring 第二节）.

    它的处置动作是**改调用**：有环的图只能给**凝缩后**的线性序
    （先承认那些环是一个整体），或者**改层**去真正拆掉这个环。
    """


class NumericError(GraphError):
    """数值不可用：非有限读数、上界 / 下界非法（负数）.

    读数非有限时任何比较都会给出一个**静默为假**的结论，因此本包在入口就拒绝。
    """


class ParameterError(GraphError):
    """参数越界：未知边种类 / 未知方向 / 未知性质 / 非正上限.

    出路是**改调用**。本包不接受"方向写错就取个默认值"这种兜底——
    一个被兜住的 ``direction`` 会让"下游闭包"与"上游闭包"在报告里长得一模一样。
    """


#: 六个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ParseError": "改解析：某份 .py 读不到或语法错时，先把它补进扫描范围或修好它",
    "ImportTargetError": "改解析或改源码：包内 import 目标对不上时，先回到相对导入的解析规则",
    "GraphBuildError": "改图实现：两次构建不一致时，去掉那个没有固定的量（集合序 / 字典序）",
    "CycleError": "改调用或改层：要线性序就先凝缩；要真正拆环就改模块分层",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：边种类 / 方向 / 性质名 / 上限都是调用点的一次决定",
}

#: 本模块真正导出的六个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ParseError": ParseError,
    "ImportTargetError": ImportTargetError,
    "GraphBuildError": GraphBuildError,
    "CycleError": CycleError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise GraphError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的族：``AssemblyError``（"把零件装成一条链"的失败，在 day102 缺席后回来）.
RETURNED_FAMILY: str | None = "AssemblyError"

#: 回来的理由：**与前面各课"请回旧名字"的理由同源**.
RETURNED_FAMILY_REASON = (
    "``AssemblyError`` 由 day080（从零实现 Transformer 块）第一次命名，"
    "此后被 ``arch_variants`` / ``encoder_decoder`` / ``hf_source`` / ``transformer_stack`` / "
    "``capstone`` 各写一遍——它是本仓库里被跨包重写最多的族名之一。"
    "day102（编台账）没有它。day103 要把 450 个模块装成**一张图**，"
    "于是'零件装不成一个整体'重新需要一次控制流事件——这个名字**回来**了，"
    "但基类从 ``ValueError`` 换成 ``GraphError(ValueError)``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余五族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是纯静态分析，一行都没跑）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day100 / day101 / day102 的理由同源（都是"没有一次反向"）**.
ABSENT_FAMILY_REASON = (
    "本课用 ``ast`` 把 450 份 ``.py`` 读成一张图：一个**纯静态**的解析层，"
    "不但没有反向，连一次 import 都不执行（它只读 import 语句，不真的导入）。"
    "因此这一族今天不属于本包——它属于被扫描的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "CycleError",
    "GraphBuildError",
    "GraphError",
    "ImportTargetError",
    "NumericError",
    "ParameterError",
    "ParseError",
]
