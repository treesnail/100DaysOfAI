"""``optimizers`` 的失败族：一次参数更新里每一种失败该谁去修（day092 / M8-D3）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。

```text
ShapeError       形状不符      参数与梯度长度不符、裁剪后向量变了长度        → 改调用
ParameterError   参数越界      学习率 / 衰减系数 / 动量系数 / 调度参数越界    → 改调用
NumericError     数值不可用    非有限读数、nan / inf 混进参数或状态          → 改数据或改实现
StateError       状态不匹配    动量 / 二阶动量的长度与参数不一致             → 改调用
StepError        更新不成立    一步走完参数不再有限、或参数被改了宽度        → 改调用
OptimizerError   本族基类
```

## 一、本课**没有**"回来"的那一族，却**再次缺席**了一族

```text
day075~081   有手写反向            ⇒ 有 GradientError 那一族
day082~089   没有手写反向          ⇒ 这一族连续缺席八天
day090       有手写反向 + 自动微分  ⇒ GradientError **回来了**
day091       复盘日，不新增代码      ⇒ 不适用
day092       消费梯度、不计算梯度    ⇒ GradientError **再次缺席**
```

day092 的主题是 day090 那句话的下一半——**"往哪走"之后是"走多远"**。方向（梯度）是
day090 交出来的输入，本课一行都不再算它。因此 ``ABSENT_FAMILY = "GradientError"``：
**它不是被忘记了，而是被交给了上一步**。把这条写进常量，是为了让"今天没有它"
成为一个可断言的事实，而不是一段散文。

## 二、``StateError`` 为什么要独立成族

```text
ShapeError   "这一次的输入形状不对"     参数与梯度当场就不匹配
StateError   "上一次留下的状态不对"      动量 / 二阶动量的长度与这一次的参数不一致
```

前者是**这一次调用**的问题（改调用点即可），后者是**跨步状态**的问题——
最典型的是"换了任务、换了参数形状，却忘了 ``reset()``"，于是上一段的动量被
悄悄带进新任务的前几步。把两者报成同一族，会让"查本次调用"的人去查状态，
而"查状态"的人去查本次调用。出路都是**改调用**，但改的是不同的那一行。

## 三、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError
│   └── GradientError（day075）
├── ...
└── OptimizerError（day092，本模块）
    ├── ShapeError / NumericError / ParameterError
    │                             （多继承：既是 day075 族、也是本族）
    └── StateError / StepError
```

本层**继承 day075 的三族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map``、``neural_basics``、``backprop`` 与本包是六个**兄弟**。因此
``except transformer_core.errors.ShapeError`` 能同时兜住本层——这正是"新写的一族
天然接进既有体系"的含义。
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


class OptimizerError(ValueError):
    """``optimizers`` 这一族错误的基类（判据与 ``BackpropError`` / ``NeuralError`` 同源）."""


class ShapeError(CoreShapeError, OptimizerError):
    """形状不符：参数与梯度长度不一致、裁剪后向量长度改变.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前面各天写的 ``except ShapeError`` 代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, OptimizerError):
    """参数越界：学习率 / 衰减系数 / 动量系数 / 调度参数落在定义域之外.

    出路是**改调用**。本包不接受"越界就取个默认值"这种兜底——
    一个被静默替换的学习率会让"我调过它"这件事变成假的。
    """


class NumericError(CoreNumericError, OptimizerError):
    """数值不可用：读数非有限、``nan`` / ``inf`` 混进参数或优化器状态.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝，而不是让它以"这条性质不通过"的形式出现在报告里。
    """


class StateError(OptimizerError):
    """状态不匹配：动量 / 二阶动量的长度与当前参数不一致.

    它值得独立成族，因为它与 :class:`ShapeError` 处置的**时机**不同：

    ```text
    ShapeError   这一次的参数与梯度不符           ⇒ 看这一次调用
    StateError   上一次留下的状态与这一次的参数不符 ⇒ 看 reset() 有没有被漏掉
    ```

    混成一族会让"换任务后前几步莫名地大"这种症状被误诊成"梯度算错了"。
    """


class StepError(OptimizerError):
    """更新不成立：一步走完参数不再有限，或参数被改了宽度.

    它与 :class:`StateError` 的差别在**时机**：状态错误在"迈步之前"就该被发现，
    而这一族说的是"这一步的输入都合法，但**走完之后**参数不再合法"
    （最典型的是学习率大到把参数推到 ``inf``）。出路是**改调用**：
    调小学习率、把整体范数裁剪打开，或先把梯度里的非有限数查出来。
    """


#: 五个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：参数与梯度长度、裁剪前后的向量长度都要先对齐",
    "ParameterError": "改调用：学习率 / 衰减 / 动量 / 调度参数都是调用点的一次决定",
    "NumericError": "改数据或改实现：nan / inf 混进状态时，先查更早的那一次除法或 log",
    "StateError": "改调用：换任务换形状时先 reset()，别让上一段的动量进来",
    "StepError": "改调用：调小学习率或加整体范数裁剪——梯度合法不代表这一步走完还合法",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
#: 加上本族基类 ``OptimizerError``，因此表里是五个、类表里是五个。
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "StateError": StateError,
    "StepError": StepError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise OptimizerError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 今天**没有回来**的那一族：本课不新增"回来"的族（它是由 day074 与 day090 会合而成的一天）.
RETURNED_FAMILY: str | None = None

#: 没有回来的理由（与 day090 那条"回来了"的理由刻意对称）.
RETURNED_FAMILY_REASON = (
    "本课不把任何缺席族'请回来'：day092 是 day074（三个更新公式）与 day090（梯度）"
    "的会合点，它既没有重新定义梯度、也没有重新引入某个旧族——"
    "它把已有的原语**组装成一次真正的训练期更新**：衰减 + 裁剪 + 调度 + 三个更新规则。"
)

#: 今天缺席的那一族：``GradientError`` **再次**缺席（这次的理由与 day082~089 那八条都不同）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：不是"没有反向"，而是"反向被交给了上一步".
ABSENT_FAMILY_REASON = (
    "day090 刚刚把 GradientError 请回来，本课又让它缺席——但理由完全不同："
    "day082~089 是'没有手写反向'，今天是'**有梯度，但不在这里算**'。"
    "本课的全部输入就是一个已经算好的梯度向量 g：它只回答 day090 那句话的下一半——"
    "'往哪个方向走'已经确定，剩下的只是'一次走多远'（学习率 × 更新规则）。"
    "因此'两条独立路径的梯度对不上'在今天不可能发生：本课一行都不算梯度。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "NumericError",
    "OptimizerError",
    "ParameterError",
    "ShapeError",
    "StateError",
    "StepError",
]
