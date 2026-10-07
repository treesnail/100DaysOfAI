"""``conv_net`` 的失败族：一次卷积前向 / 反向里每一种失败该谁去修（day093 / M8-D4）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。

```text
ShapeError       形状不符      输入 / 卷积核 / 步长 与输出尺寸对不上        → 改调用
ParameterError   参数越界      核尺寸 / 步长 / 填充 / 膨胀取值非法           → 改调用
NumericError     数值不可用    非有限读数、nan / inf 混进特征图             → 改数据或改实现
WindowError      窗口不成立    池化窗口盖不满输入、或通道数与核数不一致      → 改调用
BackwardError    反向不成立    梯度形状与缓存的前向不是同一次               → 改调用
ConvError        本族基类
```

## 一、``WindowError`` 为什么要独立成族

```text
ShapeError     "张量的形状不对"         输入与核当场就不匹配
WindowError    "这一次的窗口盖不满"     池化窗口 / 步长组合让最后一块越界或缺角
```

池化最容易踩的坑是"6×6 的图按 2×2、步长 3 下采样"——`(6−2)/3+1 = 2.33`，
不是一个整数。"取前 2 个窗口、把 0.33 丢掉"这种兜底**不会报错**，
只会让下采样率与你以为的不同。因此本包不接受它，直接拒绝。

## 二、``BackwardError`` 与 day090 同名不同义（**这一点要写下来**）

day090 的 ``backprop.errors.BackwardError`` 指的是"注意力 / MLP 层的 cache 缺失"；
本包的 ``BackwardError`` 指的是"卷积反向拿到的梯度形状与缓存的前向不是同一次"。
两者**都指向"改调用"**，但分属两个兄弟包，因此各自成族、互不继承——
它们共同继承 day075 的失败族体系。

## 三、跨包的继承关系

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError
│   └── GradientError（day075）
├── ...
└── ConvError（day093，本模块）
    ├── ShapeError / NumericError / ParameterError     （多继承：既是 day075 族、也是本族）
    └── WindowError / BackwardError
```

本层**继承 day075 的三族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map``、``neural_basics``、``backprop``、``optimizers`` 与本包是七个**兄弟**。
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


class ConvError(ValueError):
    """``conv_net`` 这一族错误的基类（判据与 ``BackpropError`` / ``OptimizerError`` 同源）."""


class ShapeError(CoreShapeError, ConvError):
    """形状不符：输入 / 核 / 输出尺寸对不上、特征图行宽不齐.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前面各天写的 ``except ShapeError`` 不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, ConvError):
    """参数越界：核尺寸 / 步长 / 填充 / 膨胀 / 池化窗口取值非法.

    出路是**改调用**。本包不接受"越界就取个默认值"这种兜底——
    一个被静默替换的步长会让"我下采样了几倍"这件事变成假的。
    """


class NumericError(CoreNumericError, ConvError):
    """数值不可用：读数非有限、``nan`` / ``inf`` 混进特征图.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝。
    """


class WindowError(ConvError):
    """窗口不成立：池化窗口盖不满输入，或卷积核数与输入通道数不一致.

    它值得独立成族，因为它与 :class:`ShapeError` 处置的**时机**不同：

    ```text
    ShapeError   张量形状当场就不匹配        ⇒ 看这次调用的入参
    WindowError  形状合法，但窗口与步长组合让最后一块越界或缺角 ⇒ 改窗口 / 步长
    ```

    混成一族会让"下采样率与预期不同"这种症状被误诊成"图喂错了"。
    """


class BackwardError(ConvError):
    """反向不成立：反向拿到的前向缓存与当前梯度**不是同一次**前向的产物.

    它与 :class:`ShapeError` 的处置方向相反：形状错误在"取梯度"时就能被发现，
    而这一族说的是"形状都对，但这两个东西不是同一次前向的"——
    出路是**改调用**（重跑前向、把缓存与梯度配对）。
    """


#: 五个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：输入 / 核 / 输出的通道与空间尺寸要逐维对齐",
    "ParameterError": "改调用：核尺寸 / 步长 / 填充 / 膨胀 / 窗口都是调用点的一次决定",
    "NumericError": "改数据或改实现：nan / inf 混进特征图时，先查更早的那一次除法",
    "WindowError": "改调用：窗口与步长必须能把输入**整除地**盖满",
    "BackwardError": "改调用：反向的梯度必须与**同一次**前向的缓存配对",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "WindowError": WindowError,
    "BackwardError": BackwardError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise ConvError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**没有**"回来"的族（本课是全新的一个包，不涉及旧族回归）.
RETURNED_FAMILY: str | None = None

#: 没有回来的理由.
RETURNED_FAMILY_REASON = (
    "本课不把任何缺席族'请回来'：day093 第一次引入『空间上的权重共享』这件事，"
    "对应的失败族（WindowError / BackwardError）都是**新命名**的，"
    "没有哪一族是'回来了'。"
)

#: 今天缺席的那一族：``GradientError``（它由 day090 命名、day092 缺席，今天继续缺席）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：本课确实算了梯度，但"两条独立路径对不上"这件事被放在性质校验里，
#: 而不是作为一个控制流事件抛出。
ABSENT_FAMILY_REASON = (
    "本课真的算了卷积的梯度，也真的与数值差分对了一次账（见 verify 的第 ⑤ 条性质），"
    "但**没有**把'对不上'做成一个控制流事件：对账是一段离线校验，不是一个运行期守卫。"
    "day090 抛 GradientError 是因为'训练不可信'必须当场停下；本课的对账发生在"
    "确定性样本上，它的结论写进性质表，而不是改变控制流——因此这一族今天缺席。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "BackwardError",
    "ConvError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "WindowError",
]
