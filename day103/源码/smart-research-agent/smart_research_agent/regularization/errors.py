"""``regularization`` 的失败族：一次"带正则化的训练"里每一种失败该谁去修（day095 / M8-D6）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。

```text
ShapeError       形状不符      特征数 / 批大小 / γ β 长度对不上            → 改调用
ParameterError   参数越界      epsilon / momentum / rate / patience 取值非法 → 改调用
NumericError     数值不可用    非有限读数、方差为负、统计量出现 nan          → 改数据或改实现
PhaseError       相不成立      训练相与推理相被用反了 / 推理相缺 running 统计 → 改调用
RegularizationError   本族基类
```

## 一、``PhaseError`` 为什么要独立成族

day081 的 dropout 已经引入了"两个相"（``train`` / ``eval``），
day087 的量化也用过它。本课第一次遇到一个**相错了不会报错**的地方：

```text
BatchNorm 在训练相用**这一批自己的**统计量，在推理相用**running 统计量**
  把两相写反 ⇒ 推理时用的是"这一批"的均值方差（批与批之间抖动）
              ⇒ 单条样本推理时方差为 0、整层输出被压成常数
              ⇒ loss 突然变差，而"形状 / 有限性 / 覆盖"全都正常
```

它与 :class:`ParameterError` 的处置方向不同：参数越界是"调用点的某个数写错了"，
而这一族是"**两个都对的东西被配错了对**"（statistics 与 phase 不匹配）。
把它混进参数族，会让"推理时结果抖动"被误诊成"某个超参没调好"。

## 二、``GradientError`` 今天缺席——理由与 day092 / day093 都不同

```text
day090   抛它：解析梯度与数值差分对不上（离线对账的结论）
day092   缺席：它消费梯度、不计算梯度
day093   缺席：算了梯度，但对账是离线校验
day094   抛它：梯度真的会爆（BPTT 把同一个权重乘了 T 次）⇒ 做成运行期守卫
day095   缺席：**梯度爆炸的守卫由 day094 装好、本课只是调用它**
```

本课手写了 BatchNorm 的反向，也确实与数值差分对了一次账（第 ④ ⑤ 条性质）——
但那次对账同样是一段**离线校验**，不在训练回路里。而训练回路里唯一与梯度有关的
控制流事件是 day094 的 ``sequence_models.train.check_gradient_norm``，
本课**调用**它、不重新抛。因此这一族今天不属于本包。

## 三、跨包的继承关系

```text
ValueError
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError
│   └── GradientError
└── RegularizationError（day095，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    └── PhaseError
```

本层**继承 day075 的族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map``、``neural_basics``、``backprop``、``optimizers``、``conv_net``、
``sequence_models`` 与本包是九个**兄弟**。
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


class RegularizationError(ValueError):
    """``regularization`` 这一族错误的基类（判据与 ``RecurrentError`` / ``ConvError`` 同源）."""


class ShapeError(CoreShapeError, RegularizationError):
    """形状不符：批的列数、γ/β 的长度、梯度的形状对不上.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前面各天写的 ``except ShapeError`` 不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, RegularizationError):
    """参数越界：``epsilon`` / ``momentum`` / ``rate`` / ``patience`` 取值非法.

    出路是**改调用**。本包不接受"越界就取个默认值"这种兜底——
    一个被静默替换的 ``momentum`` 会让"running 统计跟得有多快"这件事变成假的。
    """


class NumericError(CoreNumericError, RegularizationError):
    """数值不可用：非有限读数、方差为负、统计量里混进 ``nan`` / ``inf``.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝。
    """


class PhaseError(RegularizationError):
    """相不成立：训练相与推理相被用反了，或推理相缺少 running 统计量.

    它值得独立成族，因为它与 :class:`ParameterError` 处置的**对象**不同：

    ```text
    ParameterError  调用点的**某一个数**越界了      ⇒ 去改那个数
    PhaseError      两个都合法的东西被**配错了对**   ⇒ 去看"这一步是谁在跑"
    ```

    混成一族会让"推理时输出抖动"这种症状被误诊成"某个超参没调好"。
    """


#: 四个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：批的列数（特征数）与 γ/β 的长度必须逐维对齐",
    "ParameterError": "改调用：epsilon / momentum / rate / patience 都是调用点的一次决定",
    "NumericError": "改数据或改实现：统计量出现 nan 时，先查更早的那一次除法或 sqrt",
    "PhaseError": "改调用：训练相用本批统计、推理相用 running 统计——两者不能互换",
}

#: 本模块真正导出的四个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "PhaseError": PhaseError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise RegularizationError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**没有**"回来"的族（本课的新族 ``PhaseError`` 是新命名的，不是回来的）.
RETURNED_FAMILY: str | None = None

#: 没有回来的理由.
RETURNED_FAMILY_REASON = (
    "本课不把任何缺席族'请回来'：day095 第一次引入『训练相与推理相必须配对』这件事，"
    "对应的失败族 ``PhaseError`` 是**新命名**的；"
    "而 day094 刚请回来的 ``GradientError`` 在本课是**被调用**的（不是被重新抛出的）。"
)

#: 今天缺席的那一族：``GradientError``（它由 day090 命名、day094 回来，本课再次让它退场）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day092 / day093 / day094 都不同**.
ABSENT_FAMILY_REASON = (
    "本课真的手写了 BatchNorm 的反向，也真的与数值差分对了一次账（第 ④ / ⑤ 条性质），"
    "但**没有**把'对不上'做成一个控制流事件：对账发生在确定性样本上，结论写进性质表。"
    "而训练回路里唯一与梯度有关的控制流事件——梯度爆炸守卫——"
    "由 day094 的 ``sequence_models.train.check_gradient_norm`` 装好，本课只是**调用**它"
    "（见 ``train.train_regularized`` 的 notes）。"
    "同一个族在四天里有四种不同的理由在场或不在场，这正是'缺席要可断言'的意义。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "NumericError",
    "ParameterError",
    "PhaseError",
    "RegularizationError",
    "ShapeError",
]
