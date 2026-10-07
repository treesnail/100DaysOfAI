"""``transformer_stack`` 的失败族：**把块堆起来时每一种失败该谁去修**（day080 / M7-D5）.

分族的依据与 day075~079 逐字相同——**按“谁的错、该谁去修”分**：

```text
ShapeError        形状不符    账的层数与参数份数不一致、某一层的输出与下一层的输入不同形 → 改调用
ParameterError    参数越界    层数 <= 0、维度 <= 0、容差 <= 0、初始化幅度 <= 0          → 改调用
AssemblyError     组装不成立  注意力参数与块参数的宽度不一致 / 层参数给少了 / 堆叠深度
                              与参数份数对不上                                      → 改调用（换配置）
NumericError      数值不可用  非有限数、范数为负、增益算不出来
GradientError     梯度对不上  解析梯度与数值差分的误差超过容差                        → 改推导
StackError        本族基类
```

## 为什么这一族又多了一个名字

day079 的 ``AssemblyError`` 说的是“两个各自合法的**部件**拼不成一个整体”。
今天的问题换了一层：**堆叠是同一个部件被复制 N 份，而 N 份之间要逐层对齐**。

```text
day079   块由子层拼起来        两个子层的宽度/掩码对不上
day080   块被复制成一条链      第 i 层的输出必须正好是第 i+1 层的输入（同形）
                              而参数是 N 份——少一份在**运行到第 N 层时**才炸
```

“少一份参数”这件事值得单独命名，因为它的失效方式很坏：**它不是一开始就报错**，
而是前向跑到最后一层才发现没参数可用。本包把它挡在入口（``StackParameters``
构造那一刻），因此报错位置与出错原因在同一处。

## 跨包的继承关系（**这张图必须写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── MultiHeadError（day076）
├── PositionalError（day078）
├── EncoderDecoderError（day079）
└── StackError（day080，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    ├── GradientError                                （只在控制流里出现）
    └── AssemblyError → ParameterError               （**特例**：自己是“参数失败”的一种）
```

本层与 day076~079 一样**继承 day075 的三族而不是彼此**：它们是**兄弟**关系。
因此 ``except transformer_core.errors.ShapeError`` 能同时兜住五层，
而 ``except encoder_decoder.errors.ShapeError`` **兜不住本层**。
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


class StackError(ValueError):
    """``transformer_stack`` 这一族错误的基类（判据与 ``TransformerError`` 同源）."""


class ShapeError(CoreShapeError, StackError):
    """形状不符：账的层数与参数份数不一致、某一层的输出与下一层不同形.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    day075 那一层写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, StackError):
    """参数越界：层数 <= 0、隐藏维 <= 0、容差 <= 0、初始化幅度不在 (0, 1).

    出路是**改调用**。本包不接受“参数不合法就取个默认值”这种兜底。
    """


class NumericError(CoreNumericError, StackError):
    """数值不可用：非有限数、范数为负、增益的分母为零而调用方又要一个增益.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(StackError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**。它属于控制流而不是报告：一条对不上的梯度意味着这摞块的
    链式反向不可信——而链式反向正是今天唯一新增的那段代码。
    """


class AssemblyError(ParameterError):
    """组装不成立：同一个部件被复制 N 份时，N 份之间对不上.

    三种典型情形：

    ```text
    块参数的宽度与注意力参数的宽度不一致   → 第 i 层的 LN 输出接不进第 i 层的注意力
    参数份数少于层数                       → 前向跑到最后一层才发现没参数可用
    层数给成 0 或负数                      → “零层堆叠”在数学上是恒等，在工程上是一次笔误
    ```

    **它同时是一种“参数失败”**（继承 :class:`ParameterError`），但值得有自己的名字：
    它要求调用方**同时**看“要多少层”与“给了几份”，而“容差 <= 0”只需要改一个数。
    """


#: 五个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：第 i 层的输出与第 i+1 层的输入不同形时那一步传递根本没有定义",
    "ParameterError": "改调用：参数不是运行期数据，它是被写进调用点的一次决定",
    "AssemblyError": "改调用：要同时看'要多少层'与'给了几份参数'——本包不在运行期补一份默认参数",
    "NumericError": "改数据或改实现：形状对、式子有定义，但算出来的东西没有意义",
    "GradientError": "改推导（或显式承认容差低于分辨率并把它调大）：梯度对不上时堆叠不可信",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "AssemblyError": AssemblyError,
    "NumericError": NumericError,
    "GradientError": GradientError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):
    raise StackError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "StackError",
]
