"""``transformer_core`` 的失败族：**训练一层注意力时，每一种失败该谁去修**（day075 / M7-D1）.

分族的依据与 ``math_foundations.errors`` 逐字相同——**按"谁的错、该谁去修"分**，
而不是按"哪一行抛的"：

```text
ShapeError       形状不符      四个投影矩阵的行列数对不上、输入行宽不匹配  → 改调用
ParameterError   参数越界      头数、维度、步长、容差、top_k 越界            → 改调用
NumericError     数值不可用    非有限数、权重行和不为 1、掩码形状不符        → 改数据
GradientError    梯度对不上    解析梯度与数值差分的误差超过容差              → 改推导（或改容差）
TransformerError 本族基类：其余"这一层自己的拒绝"
```

四族都继承 ``ValueError``（与 ``MathError`` / ``RetrievalError`` / ``RagOpsError`` 同一句话）。

## 为什么多了一族 ``GradientError``

day073 的 ``math_foundations`` 里"梯度对不上"只是一个**对照结论**
（``CheckOutcome.status == "differs"``），因为它不改变任何控制流。
今天不一样：一条对不上的梯度意味着**这次训练从头到尾都不可信**，
它必须能让调用方当场停下来，而不是在报告里留一行"分歧 1"。

```text
MathError       梯度对照的结论属于**报告**（人可以读完再决定）
GradientError   梯度校验的结论属于**控制流**（这一步就不该继续）
```

这条区分是全课的骨架：**同一种失败，"谁去修"不同，就该落在不同的族里。**

## 一条贯穿全包的纪律：**不替调用方猜**

零向量在这里出现了第三次（day073 的 ``normalize`` 拒绝它、``cosine`` 记 0.0、
``vectorstore.metrics`` 记 0.0）。本包的处理是**拒绝把它当输入**：

```text
输入行全零        → ShapeError/NumericError（一行全零的 token 没有任何信息，
                   它会让这一行的打分全是 0，softmax 给出均匀分布，
                   而"均匀分布'看起来像'模型还没学到"）
投影矩阵出现全零行 → 不拒绝（那是初始化的自由，训练会自己决定）
```

前者是**数据问题**，后者是**参数问题**——同一个"全是 0"在两个位置上含义完全不同。
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

#: 四族与本层基类的关系（**这张图必须写下来**，否则"该 catch 谁"只能靠读实现）。
#:
#: ```text
#: ValueError
#: ├── MathError（day073 的族）
#: │   ├── ShapeError / NumericError / ParameterError   ← 本层的三族**继承自它们**
#: │   │                                                  （所以 catch 数学族也能抓到本层）
#: │   └── TableError
#: └── TransformerError（本层的基类）
#:     ├── ShapeError / NumericError / ParameterError   （多继承：既是数学族、也是本层族）
#:     └── GradientError                                （**只在控制流里出现**）
#: ```
#:
#: ## 一处必须说清的边界
#:
#: ``AttentionParams`` 里那些形状校验调用的仍是 day073 的
#: ``validate_matrix``——因此**当输入是一张"每行不等长"的表时，
#: 抛出来的是 ``math_foundations.errors.ShapeError``，而不是本层的 ``ShapeError``**。
#:
#: 这不是疏漏，而是一次取舍：那一条校验的"修复人"就是 day073 那一层的调用方
#: （矩阵本身不合法），把它重新包装一遍只会让"错误发生在哪一层"更难读。
#: 本层的 ``ShapeError`` 只负责**本层自己新加的拒绝**
#: （例如"四个投影的列数不一致"——那只有在这一层才有意义）。
#:
#: 若调用方只想笼统地兜住"参数写错了"，catch ``ValueError`` 即可——
#: 两个族都继承它，这也正是 ``MathError`` 与 ``TransformerError`` 当初选择
#: 继承 ``ValueError`` 的原因。


class TransformerError(ValueError):
    """``transformer_core`` 这一族错误的基类（判据与 ``MathError`` 同源）."""


class ShapeError(MathShapeError, TransformerError):
    """形状不符：四个投影矩阵的行列数对不上、输入行宽与权重列数不匹配、掩码形状不符.

    出路是**改调用**：形状不对时那个乘法根本没有定义，
    而"补零还是截断"是调用方要做的决定，不是这一层能替它拍的。

    同时继承 ``math_foundations.errors.ShapeError``：本层大量复用 day073 的
    ``validate_matrix``，于是"catch 数学族"也能兜住本层的形状错误——
    两个族的取舍见模块开头那张图。
    """


class ParameterError(MathParameterError, TransformerError):
    """参数越界：维度 <= 0、步长 <= 0、容差 <= 0、top_k 越界、步数为负.

    出路是**改调用**。这类参数不是运行期数据，它们是被写进调用点的一次决定——
    因此本包不接受"参数不合法就取个默认值"这种兜底。
    """


class NumericError(MathNumericError, TransformerError):
    """数值不可用：非有限数、权重行和不为 1、权重出现负数、损失是 NaN.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(TransformerError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**（或者显式承认"这一次的容差低于分辨率"并把它调大）。
    它属于控制流而不是报告：一条对不上的梯度意味着这次训练不可信。
    """


__all__ = [
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "TransformerError",
]
