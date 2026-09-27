"""``multi_head`` 的失败族：**多头注意力里每一种失败该谁去修**（day076 / M7-D2）.

分族的依据与 ``transformer_core.errors`` 逐字相同——**按"谁的错、该谁去修"分**，
而不是按"哪一行抛的"：

```text
ShapeError       形状不符    拼接后的行宽对不上、掩码与每头的权重形状不符   → 改调用
ParameterError   参数越界    头数 <= 0、步长 <= 0、容差 <= 0、top_k 越界    → 改调用
PartitionError   头划分不成立 d_k % heads != 0、d_v % heads != 0           → 改调用（换头数/换维度）
NumericError     数值不可用  非有限数、某一头的权重行和不为 1、出现负权重   → 改数据
GradientError    梯度对不上  解析梯度与数值差分的误差超过容差              → 改推导（或改容差）
MultiHeadError   本族基类：其余"这一层自己的拒绝"
```

## 为什么比 ``transformer_core`` 多了一族 ``PartitionError``

day075 的四族判据是"形状 / 参数 / 数值 / 梯度"。多头注意力多出来的第一件事
就是**划分**：``d_k`` 要被切成 ``heads`` 段。而"切不了"这件事有一个很特别的性质——

```text
d_k = 6, heads = 4     6 不能被 4 整除
```

它**不是形状不符**（6 与 4 都是合法维度，``AttentionShape`` 一个都不会拒绝），
也**不是普通的参数越界**（``heads = 4`` 本身完全合法）：
它是**两个合法参数放在一起才不成立**的一种失败。而这正是它值得单独命名的理由：

```text
ParameterError    heads = 0     → 改 heads
PartitionError    heads = 4 而 d_k = 6 → 改 heads 或改 d_k（**改哪一个都行**）
```

把两者混在一个族里，调用方拿到的错误信息只会说"参数不对"，
而它真正需要知道的是"**这两个数得一起换**"。

## 跨包的继承关系（**这张图必须写下来**）

```text
ValueError
├── MathError（day073）
│   └── ShapeError / NumericError / ParameterError
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError   ← day075 的三族
│   └── GradientError
└── MultiHeadError（day076，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是上一层族、也是本层族）
    ├── GradientError                                （只在控制流里出现）
    └── PartitionError → ParameterError              （**特例**：自己是"参数失败"的一种）
```

多继承的用处很具体：**上游的调用方不需要改一行代码就能兜住本层的失败**。
day075 的某段代码写 ``except ShapeError``（那里的 ShapeError 来自 ``transformer_core``）
时，它同时兜住了 day076 的形状错误——因为本层的 ``ShapeError`` 是它的子类。

## 一条从 day075 继承下来的纪律：**不替调用方猜**

```text
输入行全零          → NumericError（一行全零的 token 让 softmax 给出均匀分布，
                      而均匀分布看起来像"模型还没学到"——它是数据问题）
层数/头数越界        → ParameterError（改调用）
d_k 不能被 heads 整除 → PartitionError（**不替调用方选一个"最近的合法头数"**：
                      悄悄把 heads=4 改成 heads=3 会让"我设了 4 个头"这件事
                      在报告里消失，而报告看起来完全正常）
```

最后一条是这一课最重要的拒绝：**自动修正参数是一种"看起来友好"的静默失败**。
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


class MultiHeadError(ValueError):
    """``multi_head`` 这一族错误的基类（判据与 ``TransformerError`` 同源）."""


class ShapeError(CoreShapeError, MultiHeadError):
    """形状不符：头块拼接后的行宽对不上、每头的权重不是方阵、掩码形状不符.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    day075 那一层写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, MultiHeadError):
    """参数越界：头数 <= 0、步长为负、容差 <= 0、top_k 越界.

    出路是**改调用**。本包不接受"参数不合法就取个默认值"这种兜底。
    """


class NumericError(CoreNumericError, MultiHeadError):
    """数值不可用：非有限数、某一头的权重行和不为 1、权重出现负数.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(MultiHeadError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**。它属于控制流而不是报告：一条对不上的梯度意味着这次训练不可信。
    """


class PartitionError(ParameterError):
    """头划分不成立：``d_k % heads != 0``、``d_v % heads != 0``、头块边界对不上.

    **它同时是一种"参数失败"**（继承 :class:`ParameterError`，因此
    ``except ParameterError`` 能兜住它），但它值得有自己的名字：
    "不能整除"要求调用方**同时**考虑两个数（头数与维度），
    而"维度 <= 0"只需要改一个数——修法不同，族就不同。
    """


#: 五个族各自"该谁去修"（**这张表要被测试逐键检查**）.
#:
#: 它的用处与 day073 的四张口径表完全一致：报告里只写一个异常类名
#: （``PartitionError``）时，读报告的人仍然需要知道"看到它该做什么"。
#: 少一个键不会让任何测试变红，只会让那一族在报告里失去"该怎么办"的部分。
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：形状不符时那个乘法根本没有定义，补零还是截断是调用方的决定",
    "ParameterError": "改调用：参数不是运行期数据，它是被写进调用点的一次决定",
    "PartitionError": "改调用：头数与维度要一起换——本包绝不替你挑一个「最接近的合法头数」",
    "NumericError": "改数据或改实现：形状对、式子有定义，但算出来的东西没有意义",
    "GradientError": "改推导（或显式承认容差低于分辨率并把它调大）：梯度对不上时训练不可信",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "PartitionError": PartitionError,
    "NumericError": NumericError,
    "GradientError": GradientError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):
    raise MultiHeadError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与_FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "GradientError",
    "MultiHeadError",
    "NumericError",
    "ParameterError",
    "PartitionError",
    "ShapeError",
]
