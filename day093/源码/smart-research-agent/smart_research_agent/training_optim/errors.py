"""``training_optim`` 的失败族：**训练时每一种失败该谁去修**（day081 / M7-D6）.

分族的依据与 day075~080 逐字相同——**按“谁的错、该谁去修”分**：

```text
ShapeError        形状不符      梯度与参数不同形、掩码与张量不同形              → 改调用
ParameterError    参数越界      dropout 概率不在 [0,1)、热身步数 >= 总步数、
                               耐心 <= 0、max_norm <= 0、初始化方案不认识        → 改调用
NumericError      数值不可用    损失/梯度出现非有限数、范数为负                  → 改数据或改实现
GradientError     梯度对不上    解析梯度与数值差分的误差超过容差                  → 改推导
DivergenceError   训练发散      损失在几步之内涨到初始值的若干倍以上（**它自己一族**）→ 改超参
TrainingError     本族基类
```

## 为什么“训练发散”值得单独成族

前几天的失败都是“一个函数拿了坏参数”或“两个部件拼不起来”。
今天第一次出现**整段过程**的失败：每一步都合法、每一步都算得出来，
而连起来之后损失一路变大。

```text
NumericError      某一步的梯度里出现了 inf        → 这一步就算错了
DivergenceError   每一步都算了，而损失涨了 1e3 倍  → 每一步都对，**这一次训练**不对
```

修法完全不同：前者改数据或改实现，后者**改超参**（学习率、裁剪阈值、热身）。
族不同，是因为“该谁去修”不同——这条判据从 day075 起就没有变过。

## 跨包的继承关系（**这张图必须写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── MultiHeadError（day076）
├── PositionalError（day078）
├── EncoderDecoderError（day079）
├── StackError（day080）
└── TrainingError（day081，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既属 day073 一族、也属本族）
    ├── GradientError                                （只在控制流里出现）
    └── DivergenceError → NumericError               （**特例**：数值没坏，而是数值**在变大**）
```

本层与 day076~080 一样**继承 day073 的三族而不是彼此**：它们是**兄弟**关系。
因此 ``except math_foundations.errors.ParameterError`` 能同时兜住六层，
而 ``except transformer_stack.errors.ShapeError`` **兜不住本层**。
"""

from __future__ import annotations

from smart_research_agent.math_foundations.errors import (
    NumericError as MathNumericError,
)
from smart_research_agent.math_foundations.errors import (
    ParameterError as MathParameterError,
)
from smart_research_agent.math_foundations.errors import (
    ShapeError as MathShapeError,
)


class TrainingError(ValueError):
    """``training_optim`` 这一族错误的基类（判据与 ``MathError`` 同源）."""


class ShapeError(MathShapeError, TrainingError):
    """形状不符：梯度与参数不同形、掩码与张量不同形.

    出路是**改调用**。同时继承 ``math_foundations.errors.ShapeError``：
    day073 那一层写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(MathParameterError, TrainingError):
    """参数越界：dropout 概率不在 ``[0, 1)``、热身步数 >= 总步数、耐心 <= 0、初始化方案不认识.

    出路是**改调用**。本包不接受“参数不合法就取个默认值”这种兜底。
    """


class NumericError(MathNumericError, TrainingError):
    """数值不可用：损失或梯度出现非有限数、范数为负.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(TrainingError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**。它属于控制流而不是报告：一条对不上的梯度意味着这一次更新不可信。
    """


class DivergenceError(NumericError):
    """训练发散：损失在某一步之后涨到初始值的若干倍以上.

    **它同时是一种“数值不可用”**（继承 :class:`NumericError`），但值得有自己的名字：
    它的每一**步**都是合法的，问题出在把那些步连起来这件事上——
    修法是改超参（学习率、裁剪阈值、热身步数），而不是改数据或改一行公式。

    ```text
    ParameterError     lr = -0.1        → 改一个数（它一开始就错了）
    DivergenceError    lr = 0.5         → 改超参（数本身合法，是**它太大**了）
    ```
    """


#: 五个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：梯度与参数不同形时那一步更新根本没有定义",
    "ParameterError": "改调用：超参不是运行期数据，它是被写进调用点的一次决定",
    "NumericError": "改数据或改实现：形状对、式子有定义，但算出来的东西没有意义",
    "GradientError": "改推导（或显式承认容差低于分辨率并把它调大）：梯度对不上时这一步不可信",
    "DivergenceError": "改超参：每一步都合法，是**这一步的步长**让整段过程不收敛",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "DivergenceError": DivergenceError,
    "NumericError": NumericError,
    "GradientError": GradientError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):
    raise TrainingError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "DivergenceError",
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "TrainingError",
]
