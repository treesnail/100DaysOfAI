"""``sequence_models`` 的失败族：一次 RNN / LSTM 前向与 BPTT 里每一种失败该谁去修（day094 / M8-D5）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。

```text
ShapeError       形状不符      输入维 / 隐层宽 / 权重行列对不上          → 改调用
ParameterError   参数越界      hidden_size / 序列长度 / 步数取值非法      → 改调用
NumericError     数值不可用    nan / inf 混进状态或梯度                 → 改数据或改实现
TimeStepError    时间步不成立  返回 / 缓存的状态数与输入步数不是同一个 T   → 改调用
BackwardError    反向不成立    BPTT 拿到的缓存与回传梯度不是同一次前向     → 改调用
GradientError    梯度爆炸      一次 BPTT 的梯度范数超过上界（非有限或过大）→ 改超参（或改实现）
RecurrentError   本族基类
```

## 一、``TimeStepError`` 为什么要独立成族（**这是本课真的会踩的坑**）

```text
ShapeError      "张量的形状不对"          输入维 / 隐层宽当场就不匹配
TimeStepError   "形状全对，但步数错位了"   返回 / 缓存的状态数与输入步数不是同一个 T
```

循环网络最容易踩、而且**永远不会报错**的一个坑是"状态数比输入数多一个"：

```text
rnn_forward 的约定：输入 T 步（x₁..x_T）⇒ 返回 T 个状态（h₁..h_T），h₀ 是**初值**不是输出
若实现里顺手把 h₀ 也塞进输出 ⇒ 得到 T+1 个状态
  它不会报错：下游 zip 会静默截断，于是"第 t 个状态"与"第 t 个输入"整体错位一格。
  症状是训练仍然在跑、损失仍然在降（降得更慢），而"模型记不记得住"这个结论是假的。
```

把它与 ``ShapeError`` 混成一族，会让"错位一格"这种症状被误诊成"张量喂错了"。

## 二、``GradientError`` 回来了——理由与 day090 那一次**不同**

```text
day090  抛 GradientError   因为"解析梯度与数值差分对不上" ⇒ 那次训练不可信
day092  缺席               因为它消费梯度、不计算梯度
day093  缺席               因为对账是离线校验，不是运行期守卫
day094  回来               因为**梯度真的会爆**：BPTT 把同一个权重乘了 T 次，
                           一次爆炸就足以让参数变成 nan，必须当场拦下
```

三个"回来/缺席"的理由互不相同，这正是这一族值得被认真对待的地方：
**同一个名字，在不同的一天指的是不同的控制流事件。**

## 三、跨包的继承关系

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError
│   └── GradientError（day075）
└── RecurrentError（day094，本模块）
    ├── ShapeError / NumericError / ParameterError / GradientError  （多继承：既是 day075 族、也是本族）
    └── TimeStepError / BackwardError
```

本层**继承 day075 的族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map``、``neural_basics``、``backprop``、``optimizers``、``conv_net``
与本包是八个**兄弟**。
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


class RecurrentError(ValueError):
    """``sequence_models`` 这一族错误的基类（判据与 ``ConvError`` / ``OptimizerError`` 同源）."""


class ShapeError(CoreShapeError, RecurrentError):
    """形状不符：输入向量的宽度、隐层宽、权重行列数对不上.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前面各天写的 ``except ShapeError`` 不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, RecurrentError):
    """参数越界：``hidden_size`` / 输入维 / 序列长度 / 步数 / 范数上界取值非法.

    出路是**改调用**。本包不接受"越界就取个默认值"这种兜底——
    一个被静默替换的隐层宽会让"我有多少个可学的量"这件事变成假的。
    """


class NumericError(CoreNumericError, RecurrentError):
    """数值不可用：读数非有限、``nan`` / ``inf`` 混进状态或梯度.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝。
    """


class TimeStepError(RecurrentError):
    """时间步不成立：返回 / 缓存的状态数与输入步数不是同一个 ``T``.

    它值得独立成族，因为它与 :class:`ShapeError` 处置的**时机**不同：

    ```text
    ShapeError     每一维当场就不匹配        ⇒ 看这次调用的入参
    TimeStepError  每一维都对，只是数量差一个 ⇒ 看"初值有没有被算进输出"
    ```

    混成一族会让"状态与输入整体错位一格"这种症状被误诊成"张量喂错了"。
    """


class BackwardError(RecurrentError):
    """反向不成立：BPTT 拿到的前向缓存与回传梯度**不是同一次**前向的产物.

    它与 :class:`ShapeError` 的处置方向相反：形状错误在"取梯度"时就能被发现，
    而这一族说的是"形状都对，但这两个东西不是同一次前向的"——
    出路是**改调用**（重跑前向、把缓存与梯度配对）。
    """


class GradientError(CoreGradientError, RecurrentError):
    """梯度爆炸：一次 BPTT 的梯度范数**非有限**或超过给定的上界.

    出路是**改超参**（降学习率、缩短截断长度、加梯度裁剪），或者改实现。
    它属于控制流而不是报告：一次爆炸之后参数会变成 ``nan``，
    而 ``nan`` 会一路污染到很久以后——那时再查已经失去现场。

    **本族今天回来了**：day090 抛它的理由是"解析梯度与数值差分对不上"，
    本课抛它的理由是"梯度真的爆了"。同名、不同事，见模块开头那张表。
    """


#: 六个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：输入维 / 隐层宽 / 权重行列要逐维对齐",
    "ParameterError": "改调用：hidden_size / 输入维 / 序列长度 / 步数 都是调用点的一次决定",
    "NumericError": "改数据或改实现：nan / inf 混进状态时，先查更早的那一次除法或激活",
    "TimeStepError": "改调用：输入 T 步就要返回 T 个状态——初值 h₀ 不是输出",
    "BackwardError": "改调用：BPTT 的梯度必须与**同一次**前向的缓存配对",
    "GradientError": "改超参或改实现：一次 BPTT 的范数越界时，先降学习率 / 缩短展开长度",
}

#: 本模块真正导出的六个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "TimeStepError": TimeStepError,
    "BackwardError": BackwardError,
    "GradientError": GradientError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise RecurrentError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的那一族：``GradientError``（它由 day090 命名，day092 / day093 两次缺席）.
RETURNED_FAMILY: str | None = "GradientError"

#: 回来的理由：与 day090 那一次**不是同一件事**.
RETURNED_FAMILY_REASON = (
    "day090 抛 GradientError 是因为'解析梯度与数值差分对不上'——那是一次离线对账的结论；"
    "day094 抛它是因为**梯度真的爆了**：BPTT 把同一个循环权重乘了 T 次，"
    "范数随 T 指数增长，一次爆炸就足以把参数变成 nan。"
    "前者是'这次训练不可信'，后者是'这一步就不该继续'——同名、不同事。"
    "本包把它做成一个运行期守卫（``train.check_gradient_norm``），因此它回来了。"
)

#: 今天**缺席**的那一族：没有（六个族全部在场，``GradientError`` 是回来的那一个）.
ABSENT_FAMILY: str | None = None

#: 缺席理由：本课六族齐备，缺席名单是空的——但"空"本身要能被断言.
ABSENT_FAMILY_REASON = (
    "本课没有缺席的族：形状 / 参数 / 数值 三族描述'喂进去的东西'，"
    "TimeStepError / BackwardError 描述'时间轴上的两处接缝'，"
    "GradientError 描述'一次 BPTT 走得太远'。"
    "前五个覆盖了前向与反向的每一种'当场就该停下'，第六个覆盖了梯度本身——"
    "因此这一天没有任何一族需要被记成'缺席'。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "BackwardError",
    "GradientError",
    "NumericError",
    "ParameterError",
    "RecurrentError",
    "ShapeError",
    "TimeStepError",
]
