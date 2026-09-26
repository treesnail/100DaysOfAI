"""``encoder_decoder`` 的失败族：**组装一个 Transformer 块时每一种失败该谁去修**（day079 / M7-D4）.

分族的依据与 day075~078 逐字相同——**按“谁的错、该谁去修”分**：

```text
ShapeError        形状不符    两路形状不匹配、隐藏维不一致、gamma/beta 长度不符   → 改调用
ParameterError    参数越界    层数 <= 0、eps <= 0、维度 <= 0、容差 <= 0           → 改调用
AssemblyError     组装不成立  encoder 与 decoder 的隐藏维不一致 / **把因果掩码
                              用在交叉注意力上** / 子层顺序给错                 → 改调用（换配置）
NumericError      数值不可用  非有限数、某一行的方差为 0 而 eps 救不回来
GradientError     梯度对不上  解析梯度与数值差分的误差超过容差                   → 改推导
EncoderDecoderError  本族基类
```

## 为什么比 ``transformer_core`` 多了一族 ``AssemblyError``

前几天的失败都是“一个函数拿了坏参数”。而这一课第一次出现**组装**：
块由若干子层拼起来，而**两个各自合法的部件可以拼不成一个合法的整体**。

```text
encoder 的隐藏维 = 6，decoder 的隐藏维 = 8     ← 两个数各自都合法
交叉注意力要求 Q 与 K/V 落在同一个空间里        ← 于是“拼起来”这件事不成立
```

它**不是形状不符**（两个 6 与 8 都是合法维度），也**不是普通的参数越界**
（把哪个改成另一个都行）。它值得单独命名的理由与 day076 的 ``PartitionError``
完全一样：**修法不同**。

```text
ParameterError    layers = 0            → 改一个数
AssemblyError     n_src != n_tgt 的因果掩码 → 改“这两个部件怎么接”（**两处一起看**）
```

## 这一族里最值钱的一条：把因果掩码用到交叉注意力上

```text
自注意力（解码器）   Q/K/V 都来自同一路 → 必须加因果掩码（不许看未来）
交叉注意力           Q 来自解码器、K/V 来自编码器 → **绝不能加因果掩码**
```

而“加错了”这件事有一个很坏的性质：**只有在 ``n_tgt == n_src`` 时才不报错**。
形状刚好对得上、输出也仍然是一张合法的权重表，而它悄悄地把“源序列的后半段”
从注意范围里删掉了。因此本包把这一条**做成一条显式的拒绝**，
而不是留给调用方去小心（第 8 条性质专门量它）。

## 跨包的继承关系（**这张图必须写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── MultiHeadError（day076）
├── PositionalError（day078）
└── EncoderDecoderError（day079，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    ├── GradientError                                （只在控制流里出现）
    └── AssemblyError → ParameterError               （**特例**：自己是“参数失败”的一种）
```

本层与 day076/078 一样**继承 day075 的三族而不是彼此**：它们是**兄弟**关系。
因此 ``except transformer_core.errors.ShapeError`` 能同时兜住四层，
而 ``except positional_encoding.errors.ShapeError`` **兜不住本层**。
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


class EncoderDecoderError(ValueError):
    """``encoder_decoder`` 这一族错误的基类（判据与 ``TransformerError`` 同源）."""


class ShapeError(CoreShapeError, EncoderDecoderError):
    """形状不符：两路形状不匹配、隐藏维与权重列数不一致、gamma/beta 长度不符.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    day075 那一层写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, EncoderDecoderError):
    """参数越界：层数 <= 0、eps <= 0、隐藏维 <= 0、容差 <= 0.

    出路是**改调用**。本包不接受“参数不合法就取个默认值”这种兜底。
    """


class NumericError(CoreNumericError, EncoderDecoderError):
    """数值不可用：非有限数、某一行的方差为 0 而 ``eps`` 救不回来.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(EncoderDecoderError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**。它属于控制流而不是报告：一条对不上的梯度意味着这次堆叠不可信。
    """


class AssemblyError(ParameterError):
    """组装不成立：两个各自合法的部件拼不成一个合法的整体.

    三种典型情形：

    ```text
    编码器与解码器的隐藏维不一致   → 解码器的初始化无法从编码器的输出接下去
    交叉注意力被加了因果掩码       → **n_tgt == n_src 时不报错**，只是悄悄删掉了半段源序列
    子层顺序给错                   → pre/post 只差 LN 的位置，给错时形状完全合法
    ```

    **它同时是一种“参数失败”**（继承 :class:`ParameterError`），但值得有自己的名字：
    它要求调用方**同时**看两个部件，而“层数 <= 0”只需要改一个数——修法不同，族就不同。
    """


#: 五个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：两路形状不符时那个加法/乘法根本没有定义",
    "ParameterError": "改调用：参数不是运行期数据，它是被写进调用点的一次决定",
    "AssemblyError": "改调用：两个部件要一起看——本包绝不把跨路掩码当成一个默认值",
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
    raise EncoderDecoderError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "EncoderDecoderError",
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
]
