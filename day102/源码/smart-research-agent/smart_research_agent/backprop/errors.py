"""``backprop`` 的失败族：**一次反向传播里每一种失败该谁去修**（day090 / M8-D2）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。今天这一批里有一族是
**回来了**，而不是新来的：``GradientError`` 在连续缺席八天之后（day082~089），
今天第一次被真正用上。

```text
ShapeError       形状不符      回传梯度与缓存形状不一致、参数与梯度长度不符      → 改调用
ParameterError   参数越界      激活名未知、扰动步长 <= 0、容差 <= 0            → 改调用
NumericError     数值不可用    非有限读数、nan / inf 混进梯度                  → 改数据或改实现
GradientError    梯度对不上    解析梯度与数值差分的相对误差超过容差            → 改推导（或改容差）
BackwardError    反向不成立    cache 缺失、cache 与 inputs 不是同一次前向的产物  → 改调用
ChainError       接线不对      把一个与其上游无关的值接进图、或忘记清零而重复反向 → 改推导
StepError        更新不成立    一次参数更新给出非有限参数、或把参数改小了宽度     → 改调用
BackpropError    本族基类
```

## 一、本课**把缺席八天的那一族请了回来**

```text
day075~081   有手写反向            ⇒ 有 GradientError 那一族
day082~089   没有手写反向          ⇒ 这一族连续缺席八天
day090       有手写反向 + 自动微分  ⇒ **它回来了**（今天是第九次改口径，第一次是"加回来"）
```

前面八天每一次都给出一条"为什么今天没有它"的理由（读别人的推理路径、装进来的层、
不可微、没有新式子……）。今天不再需要理由——因为今天的**全部内容**就是这件事：

```text
手写反向    六个激活的导数、三个损失的梯度、Dense / MLP / FFN 的逐层回传
自动微分    一张向量级计算图，把同一批梯度再算一遍
对账        手写 = 自动微分 = 数值差分（math_foundations.calculus）
```

因此本模块把 ``RETURNED_FAMILY`` 与 ``ABSENT_FAMILY`` 两个常量都写下来：
``ABSENT_FAMILY`` 今天是 ``None``——**连续八天缺席的记录到此为止**。

## 二、``GradientError`` 与 ``BackwardError`` 为什么不是同一族

```text
GradientError   "这一条推导算错了"         解析式与数值差分对不上 ⇒ 改**推导**
BackwardError   "这次反向的输入不成立"     cache 缺失 / cache 与输入不是同一次前向 ⇒ 改**调用**
```

前者是**数学**问题（式子抄错了、漏了一项），后者是**接线**问题（把上一次前向的缓存
喂给了这一次的反向）。混成一族会让"查公式"的人去查调用点，而"查接线"的人去推公式。

## 三、``ChainError``：一个"不会报错、只会算错"的族

链式法则只有一条规则——**上游梯度 × 局部导数**，然后**累加**。写错它的方式有三种，
三种都不报错：

```text
用 = 代替 +=            一个值被用了 k 次时只保留最后一条路径 ⇒ 梯度偏小
把常量节点写进梯度       常量不该接收梯度 ⇒ 一个永远为 0 的梯度被当成"这个参数不学"
重复反向而不清零         两次 backward 把梯度加起来 ⇒ "这一步"变成"这两步之和"
```

三条都能通过"形状对、数值有限"的检查，因此它们值得一个**独立成族**的名字：
出现 ``ChainError`` 时，要改的是接线方式，而不是某个公式。

## 四、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError
│   └── GradientError（day075 定义，day090 第一次真正用上）
├── ...
└── BackpropError（day090，本模块）
    ├── ShapeError / NumericError / ParameterError / GradientError
    │                             （多继承：既是 day075 族、也是本族）
    └── BackwardError / ChainError / StepError
```

本层**继承 day075 的四族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map``、``neural_basics`` 与本包是五个**兄弟**。因此
``except transformer_core.errors.GradientError`` 能同时兜住本层——
这正是今天"请回那一族"的意义：**它在 day075 就被命名，一直到今天才有人真的抛它。**
"""

from __future__ import annotations

from smart_research_agent.transformer_core.errors import (
    GradientError as CoreGradientError,
)
from smart_research_agent.transformer_core.errors import (
    NumericError as CoreNumericError,
)
from smart_research_agent.transformer_core.errors import (
    ParameterError as CoreParameterError,
)
from smart_research_agent.transformer_core.errors import (
    ShapeError as CoreShapeError,
)


class BackpropError(ValueError):
    """``backprop`` 这一族错误的基类（判据与 ``NeuralError`` / ``BridgeError`` 同源）."""


class ShapeError(CoreShapeError, BackpropError):
    """形状不符：回传梯度与缓存形状不一致、参数压平后的长度与形状表不符.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前面各天写的 ``except ShapeError`` 代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, BackpropError):
    """参数越界：激活名未知、数值差分的步长 <= 0、容差 <= 0.

    出路是**改调用**。本包不接受"步长不合法就取个默认值"这种兜底——
    一个被静默替换的步长会让"我调了精度"这件事变成假的。
    """


class NumericError(CoreNumericError, BackpropError):
    """数值不可用：读数非有限、``nan`` / ``inf`` 混进梯度.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝，而不是让它以"这条性质不通过"的形式出现在报告里。
    """


class GradientError(CoreGradientError, BackpropError):
    """梯度对不上：解析梯度与数值差分的相对误差超过容差.

    **这一族在连续缺席八天之后回到了课堂。** day075 定义它时写的理由是
    "一条对不上的梯度意味着这次训练从头到尾都不可信"，但那一天以后
    一直没有人真的抛它——直到今天，因为今天有了**两条独立算出梯度的路径**
    （手写反向与数值差分），而"两条路径不一致"到这里第一次成为一个控制流事件。

    出路是**改推导**，或者显式承认"这一次的容差低于测量手段的分辨率"并把它调大
    （见 :func:`backprop.verify.difference_resolution` 的同一口径）。
    """


class BackwardError(BackpropError):
    """反向不成立：``cache`` 缺失，或 ``cache`` 与当前输入不是同一次前向的产物.

    它值得独立成族，因为它与 :class:`GradientError` 的处置方向相反：

    ```text
    GradientError   公式抄错了         ⇒ 去改推导
    BackwardError   把上一次前向的缓存喂给了这一次反向 ⇒ 去改调用点
    ```

    把后者报成前者，会让人去推公式——而公式一直是对的。
    """


class ChainError(BackpropError):
    """接线不对：把一个与其上游无关的值接进图，或忘记清零而重复反向.

    链式法则只有一条规则（上游梯度 × 局部导数）与一条纪律（**累加**，也就是 ``+=``）。
    写错它的后果非常具体：**不报错，只是梯度偏小或偏大**，
    而"梯度偏小"看起来像"学习率设小了"——于是有人去调学习率。

    这一族在实现里对应三处具体的位置（见 :mod:`backprop.graph` 的 ``Node.zero_grad``
    与 :func:`backprop.graph.backward` 的累加循环）。把它们的失败放在一族里，
    是为了让"这一次要改的是接线"这句话可以被直接读出来。
    """


class StepError(BackpropError):
    """更新不成立：一次参数更新给出非有限参数，或改动了参数的形状.

    它与 :class:`backprop.errors.ShapeError` 的差别在**时机**：形状错误在
    "取梯度"时就该被发现，而这一族说的是"梯度是合法的，但**这一步走完之后**参数不再合法"
    （最典型的是学习率大到把参数推到 ``inf``）。出路是**改调用**：
    调小学习率、加梯度裁剪，或先把梯度里的非有限数查出来。
    """


#: 八个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：回传梯度与缓存的形状、参数压平后的长度都要先对齐",
    "ParameterError": "改调用：激活名 / 扰动步长 / 容差都是调用点的一次决定",
    "NumericError": "改数据或改实现：nan / inf 混进梯度时，先查更早的那一次 log 或除法",
    "GradientError": "改推导：解析式与数值差分对不上时，先怀疑漏了一项或符号写反",
    "BackwardError": "改调用：cache 必须是**同一次前向**的产物，缺了就重跑前向",
    "ChainError": "改推导：梯度一律用 += 累加、常量节点不接收梯度、重复反向前先清零",
    "StepError": "改调用：调小学习率或加裁剪——梯度合法不代表这一步走完还合法",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）
#: 加上本族基类 ``BackpropError``，因此表里是七个、类表里是七个。
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "GradientError": GradientError,
    "BackwardError": BackwardError,
    "ChainError": ChainError,
    "StepError": StepError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise BackpropError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 连续缺席八天之后**回来了**的那一族（写进常量，让"它回来了"是一个可断言的事实）.
RETURNED_FAMILY = "GradientError"

#: 它回来的理由（与前八天那八条**都不同**：这次不是"没有它"，而是"有两条独立路径"）.
RETURNED_FAMILY_REASON = (
    "day075 定义了这一族，给出的理由是'一条对不上的梯度意味着这次训练不可信'；"
    "day082~089 连续八天没有手写反向，因此它一直缺席。"
    "day090 第一次同时拥有两条**独立**算出梯度的路径——手写反向（本包的 gradients / layers / "
    "network）与数值差分（math_foundations.calculus）——于是'两条路径不一致'第一次成为一个"
    "控制流事件：它不是报告里的'分歧 1'，而是必须当场停下的 GradientError。"
)

#: 本课**没有**缺席的那一族：写进常量，让"这一天不再有缺席者"可以被逐字钉住.
ABSENT_FAMILY: str | None = None

#: 缺席理由：今天是空的（这是它连续八天非空以来的第一次）.
ABSENT_FAMILY_REASON = (
    "本课不缺席任何一族：连续缺席八天的 GradientError 今天回来了（见 RETURNED_FAMILY_REASON）。"
    "前面八天每天都有一条'为什么今天没有它'的理由，今天那条理由第一次没有被写下——"
    "因为今天全部的内容就是它。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "BackpropError",
    "BackwardError",
    "ChainError",
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "StepError",
]
