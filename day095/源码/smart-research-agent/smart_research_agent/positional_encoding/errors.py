"""``positional_encoding`` 的失败族：**位置编码里每一种失败该谁去修**（day078 / M7-D3）.

分族的依据与 day075/076 逐字相同——**按“谁的错、该谁去修”分**，
而不是按“哪一行抛的”：

```text
ShapeError       形状不符   相加的两边行宽不一致、位置表的宽度与输入列数不一致   → 改调用
ParameterError   参数越界   维度 <= 0、位置数 < 1、频率基数 <= 1、容差 <= 0       → 改调用
RangeError       位置越界   查询的位置 >= 表的行数（可学习表是**定长**的）        → 改调用（换表/换编码）
NumericError     数值不可用 非有限数、表里出现非有限数、行范数不是常数            → 改数据
GradientError    梯度对不上 解析梯度与数值差分的误差超过容差                     → 改推导
PositionalError  本族基类：其余“这一层自己的拒绝”
```

## 为什么比 ``transformer_core`` 多了一族 ``RangeError``

day075 的四族判据是“形状 / 参数 / 数值 / 梯度”；day076 多了一族 ``PartitionError``。
本课多出来的是**位置越界**，而它有一个和“划分不成立”一模一样的性质——
**两个都合法的数放在一起才不成立**：

```text
表的行数 = 8（训练时的最大长度）      8 完全合法
查询位置 = 20                       20 完全合法（“第 20 个 token”没有任何问题）
查不到                               → 表里没有第 20 行
```

它**不是形状不符**（`20` 不是“某个矩阵的行数对不上”），也**不是普通的参数越界**
（`20` 本身完全合法，越界是相对表长而言的）。而这正是它值得单独命名的理由：

```text
ParameterError    dimension = 0        → 改一个数（维度）
RangeError        position = 20 而表长 = 8 → 改表长 或 换编码（**改哪一个都行**）
```

把一个“第 20 位”的查询悄悄截断成“第 7 位”，是这一层最危险的静默失败：
它不报错、不产生非有限数，只是让模型看到**另一个位置**——
而报告里“我查了第 20 位”这句话仍然原样写着。

## 与可学习表的关系（这一族存在的全部理由）

```text
正弦表      位置 p 的行是**公式**给出的 → 表长之外仍然有定义 → 越界可以算
可学习表    位置 p 的行是**参数**       → 表长之外没有参数     → 越界必须拒绝
```

“正弦表能外推”与“可学习表只能拒绝”是本课两条性质的直接对照，
因此这条拒绝不是防御性代码，而是**两种编码的语义差别**在一个异常类上的落点。

## 跨包的继承关系（**这张图必须写下来**）

```text
ValueError
├── MathError（day073）
│   └── ShapeError / NumericError / ParameterError
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError   ← day075 的三族
│   └── GradientError
├── MultiHeadError（day076）
│   ├── ShapeError / NumericError / ParameterError   （与 day075 的多继承）
│   ├── GradientError
│   └── PartitionError → ParameterError
└── PositionalError（day078，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是上一层族、也是本层族）
    ├── GradientError                                （只在控制流里出现）
    └── RangeError → ParameterError                  （**特例**：自己是“参数失败”的一种）
```

**一处必须写下来的不对称**：day076 与本层都继承 day075 的三族，而不是彼此——
它们是**兄弟**关系。因此

```text
except transformer_core.errors.ShapeError   同时兜住 day076 与本层（因为两者都是它的子类）
except multi_head.errors.ShapeError         只兜住 day076 —— **兜不住本层**
```

如果哪天有人把本层的 ``ShapeError`` 改成继承 day076 的，那么“本层抛出的失败”
会被归类成“多头那一层的失败”——报告里那一族的名字会变成一句假话。
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


class PositionalError(ValueError):
    """``positional_encoding`` 这一族错误的基类（判据与 ``TransformerError`` 同源）."""


class ShapeError(CoreShapeError, PositionalError):
    """形状不符：相加的两边行宽不一致、位置表的宽度与输入列数不一致、位置数与输入行数不一致.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    day075 那一层写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, PositionalError):
    """参数越界：维度 <= 0、位置数 < 1、频率基数 <= 1、容差 <= 0.

    出路是**改调用**。本包不接受“参数不合法就取个默认值”这种兜底。
    """


class NumericError(CoreNumericError, PositionalError):
    """数值不可用：非有限数、表里出现非有限数、行范数不是常数.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(PositionalError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**。它属于控制流而不是报告：一条对不上的梯度意味着这次训练不可信。
    """


class RangeError(ParameterError):
    """位置越界：查询的位置 ``>=`` 表的行数，而表不会替调用方外推.

    **它同时是一种“参数失败”**（继承 :class:`ParameterError`，因此
    ``except ParameterError`` 能兜住它），但它值得有自己的名字：
    “第 20 位查不到”要求调用方**同时**看两个数（查询的位置与表的行数），
    而“维度 <= 0”只需要改一个数——修法不同，族就不同。

    正弦表**不会**抛它（公式对任意位置都有定义）；只有可学习表会——
    因为它的每一行都是一个参数，而“第 20 行的参数”根本不存在。
    """


#: 五个族各自“该谁去修”（**这张表要被测试逐键检查**）.
#:
#: 它的用处与 day073 的四张口径表完全一致：报告里只写一个异常类名
#: （``RangeError``）时，读报告的人仍然需要知道“看到它该做什么”。
#: 少一个键不会让任何测试变红，只会让那一族在报告里失去“该怎么办”的部分。
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：相加要求两边的行宽逐位对齐，补零还是截断是调用方的决定",
    "ParameterError": "改调用：参数不是运行期数据，它是被写进调用点的一次决定",
    "RangeError": "改调用：位置与表长要一起看——本包绝不把'第 20 位'悄悄截成'第 7 位'",
    "NumericError": "改数据或改实现：形状对、式子有定义，但算出来的东西没有意义",
    "GradientError": "改推导（或显式承认容差低于分辨率并把它调大）：梯度对不上时训练不可信",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "RangeError": RangeError,
    "NumericError": NumericError,
    "GradientError": GradientError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):
    raise PositionalError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "GradientError",
    "NumericError",
    "ParameterError",
    "PositionalError",
    "RangeError",
    "ShapeError",
]
