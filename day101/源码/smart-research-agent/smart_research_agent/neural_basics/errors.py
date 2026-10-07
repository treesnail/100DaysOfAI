"""``neural_basics`` 的失败族：**神经网络基础里每一种失败该谁去修**（day089 / M8-D1）.

分族的依据与前面各天逐字相同——**按“谁的错、该谁去修”分**。今天这一批是四族全新的，
而它们都出现在同一条链上：**神经元 → 一层 → 一个网络 → 一个损失**。

```text
ShapeError           形状不符     张量宽度不齐、权重形状不是 (out, in)、
                                 样本与标签的数量对不上                        → 改调用
ParameterError       参数越界     激活名未知、初始化名未知、种子越界、层数非正    → 改调用
NumericError         数值不可用   读数非有限（nan / inf）、越界读数              → 改数据或改实现
ActivationError      激活不成立   未知激活名，或激活的定义域 / 数值越界          → 改激活或改定义域
LossError            损失不成立   空样本、标签越界、概率为 0 取 log              → 改数据或改损失
ForwardError         前向不成立   层与层宽度接不上                              → 改结构
InitializationError  初始化不成立 初始化尺度非法（例如 xavier 的分母为 0）        → 改初始化尺度
NeuralError          本族基类
```

## 一、本课**连续第八天缺席**的那一族：``GradientError``

```text
day075~081   有手写反向            ⇒ 有 GradientError 那一族
day082~089   没有手写反向          ⇒ 这一族连续缺席八天
```

今天的理由与前四条**都不同**：

```text
day085   只读别人的推理路径（HF 的反向由 autograd 推出来）
day086   装进来的层都是别人写好的，全部落在前向之外
day087   量化不可微（round 的导数几乎处处为 0）
day088   本日一个新式子都没有：全部动作是跨包对账与串联
day089   今天**有新式子**（前向与损失），但**一行反向都没有**——反向传播是 day090 的主题
```

这一条最容易被误判：今天明明写了 sigmoid、tanh、gelu、softmax 与交叉熵，
它们每一个都**可微**，看上去"该有梯度"了。但本课做的是：

```text
手写反向      一行都没有
自动微分      没有调用（本仓库不依赖 torch，也没有 autograd 参与本包）
```

因此 ``GradientError`` 缺席的理由既不是"不改算术"（day083）、也不是"读别人的路径"（day085）、
不是"不可微"（day087）、更不是"一个式子都没有"（day088）——而是
"**有前向、有损失、没有反向，而且没有交给自动微分**"。
一条纪律在这里第六次兑现：**能"改推导"的地方必须是有人真写了推导的地方。**

## 二、``ActivationError`` 与 ``ParameterError`` 为什么不是同一族

```text
ParameterError    "调用点把名字写错了"       unknown="rellu" 是一个**拼写**问题
ActivationError   "这个名字对应的定义域不成立" logit = 1e6 让 sigmoid 饱和到 1.0，
                                              再取 log 得到 -inf——数值上是**定义域**问题
```

前者改一个字符串，后者要么换激活、要么先缩放输入。混成一族会让"把名字改对"的人
去调数据分布，而"该换激活"的人去查拼写。

## 三、``LossError`` 为什么不能静默兜成 0

损失是训练里**唯一**会把"配置错了"伪装成"效果很好"的地方：

```text
一批样本全被屏蔽      ⇒ 若返回 0.0，报告里是一行漂亮的 "loss = 0"
标签越界              ⇒ 若返回 0.0，模型只是"没有在学"，而看不出是标签错了
概率为 0 取 log       ⇒ 若返回 0.0，交叉熵在数值上"完美"，其实那一步没有定义
```

因此本包在入口就抛 :class:`LossError`，把"这一批没有可用的监督信号"当作一次
**必须被发现的配置错误**，而不是一个可以被平均掉的数。

## 四、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── ...
├── OptimError（day087）
├── BridgeError（day088）
└── NeuralError（day089，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    └── ActivationError / LossError / ForwardError / InitializationError
```

本层同样**继承 day075 的三族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map`` 与本包是四个**兄弟**。因此
``except transformer_core.errors.ShapeError`` 能同时兜住九层，
而 ``except principle_map.errors.ShapeError`` **兜不住本层**——
这一条有它自己的用处：每一天的读数在报告里必须能分开统计。
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


class NeuralError(ValueError):
    """``neural_basics`` 这一族错误的基类（判据与 ``BridgeError`` / ``OptimError`` 同源）."""


class ShapeError(CoreShapeError, NeuralError):
    """形状不符：张量宽度不齐、权重形状不是 ``(out, in)``、样本与标签数量对不上.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前十六天写的 ``except ShapeError`` 代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, NeuralError):
    """参数越界：激活名未知、初始化名未知、``seed`` 越界、层数非正.

    出路是**改调用**。本包不接受"激活名写错就取个默认激活"这种兜底——
    一个被兜住的 ``activation="rellu"`` 会被就近取成 ``relu``，
    而它与"我确实写了 relu"在读表时**完全一样**，差别只在读数悄悄变了。
    """


class NumericError(CoreNumericError, NeuralError):
    """数值不可用：读数非有限（``nan`` / ``inf``）、越界读数.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝，而不是让它以"这条性质不通过"的形式出现在报告里。
    """


class ActivationError(NeuralError):
    """激活不成立：未知的激活名，或激活的定义域 / 数值越界.

    它与 :class:`ParameterError` 的差别在**定义域**而不是拼写：
    ``sigmoid(1e6)`` 会饱和到 ``1.0``、再取 ``log`` 得到 ``-inf``——
    这不是"名字写错了"，而是"这一组数配这个激活不成立"。
    出路是**改激活或改定义域**（换一个不饱和的激活，或先把输入缩放回安全区间）。
    """


class LossError(NeuralError):
    """损失不成立：空样本、标签越界、概率为 0 取 log.

    它值得独立成族，因为它是最容易被**静默兜成 0.0** 的一类失败：
    "这一批没有可用的监督信号"与"这一批学得完美"在报告里都写成 ``loss = 0``。
    出路是**改数据或改损失**——要么补样本、要么改标签、要么在定义域外先拒绝。
    """


class ForwardError(NeuralError):
    """前向不成立：层与层宽度接不上.

    它与 :class:`ShapeError` 的差别在**尺度层**：前者是"这一摞层的接口对不上"
    （改的是网络结构），后者是"某一次调用的张量形状不对"（改的是这一次调用）。
    把宽度接不上的问题报成单层形状错误，会让人去改某一层的**内部**，
    而真正该改的是它和下一层之间的**接口**。
    """


class InitializationError(NeuralError):
    """初始化不成立：初始化尺度非法（例如 xavier 的分母 ``fan_in + fan_out`` 为 0）.

    它值得独立成族，因为它的出路既不是改调用（宽度是合法的）也不是改数据，
    而是**先决定这一层的宽度**：``xavier`` 的尺度是 ``sqrt(6 / (fan_in + fan_out))``，
    当两者之和为 0 时这个式子没有定义——本包当场拒绝，而不是让它变成一次
    ``ZeroDivisionError``（那会被读成"随机崩了"而不是"这一层没有宽度"）。
    """


#: 七个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：张量宽度、权重形状 (out, in)、样本与标签的数量都要先对齐",
    "ParameterError": "改调用：激活名 / 初始化名 / 种子 / 层数都是调用点的一次决定",
    "NumericError": "改数据或改实现：nan / inf / 越界读数都属于'数值不可用'",
    "ActivationError": "改激活或改定义域：未知激活名要改名，定义域越界要么换激活、要么先缩放输入",
    "LossError": "改数据或改损失：空样本、标签越界、概率为 0 取 log 都不该被静默兜成 0",
    "ForwardError": "改结构：层与层宽度接不上时，要改的是这一摞层的接口，而不是某一层的内部",
    "InitializationError": "改初始化尺度：xavier 的分母（fan_in + fan_out）为 0 时，先决定这一层的宽度",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "ActivationError": ActivationError,
    "LossError": LossError,
    "ForwardError": ForwardError,
    "InitializationError": InitializationError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise NeuralError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 本课**连续第八天缺席**的那一族（写进常量，让"缺席"是一个可断言的事实而不是一段散文）.
ABSENT_FAMILY = "GradientError"

#: 缺席的理由（与 day085~day088 的那四条**都不同**，这一条要写清楚）.
ABSENT_FAMILY_REASON = (
    "今天有新式子（六个激活的前向、三个损失），它们每一个都可微，看上去'该有梯度'了；"
    "但本课手写反向一行都没有，也没有调用自动微分（本仓库不依赖 torch，autograd 不参与本包）。"
    "反向传播是 day090 的主题。因此这一族缺席的理由既不是'不改算术'（day083）、"
    "也不是'只读别人的推理路径'（day085）、不是'不可微'（day087）、"
    "更不是'一个式子都没有'（day088）——而是'有前向、有损失、没有反向，而且没有交给自动微分'。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "ActivationError",
    "ForwardError",
    "InitializationError",
    "LossError",
    "NeuralError",
    "NumericError",
    "ParameterError",
    "ShapeError",
]
