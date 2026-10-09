"""``arch_variants`` 的失败族：**换一个变体之后，哪一种失败该谁去修**（day082 / M7-D7）.

分族的依据与前六天逐字相同——**按“谁的错、该谁去修”分**：

```text
ShapeError        形状不符    两路形状不匹配、隐藏维与权重列数不一致、γ/β 长度不符   → 改调用
ParameterError    参数越界    variant 名字不认识、层数 <= 0、维度 <= 0、种子非法     → 改调用
AssemblyError     组装不成立  **把填充掩码交给解码器**、自注意力不是因果的、
                              **给编码器装交叉注意力**、掩码与序列长度不符         → 改调用（换配置）
NumericError      数值不可用  非有限数、某一行的掩码全为 False、损失里出现 nan      → 改数据或改实现
GradientError     梯度对不上  解析梯度与数值差分的误差超过容差                     → 改推导
VariantError      本族基类
```

## 这一天新增的那一条拒绝：把填充掩码交给解码器

day079 的 ``AssemblyError`` 里最值钱的一条是“交叉注意力不能被赋因果掩码”。今天
出现的是它的**镜像**：

```text
因果性是**解码器自己那一流**的性质   → 自注意力必须因果
填充是**输入数据**的性质            → 谁读输入，谁才需要对填充视而不见
```

而本课的 T5 解码器**不能**同时满足两者——理由是 day079 的一条硬接口：
``decoder_block`` 要求 ``self_attention_forward.causal`` 为 ``True``，
而 day075 的 ``resolve_mask`` **拒绝** ``causal=True`` 与显式掩码同时给
（“两个来源同时生效时到底用哪张要读代码才知道”）。

于是本包把这件事**变成一条显式的拒绝**，而不是悄悄忽略填充：

```text
encoder_decoder + pads 非空   → AssemblyError（一句话说清为什么，以及该怎么绕）
```

这条拒绝的代价是真实的（T5 的解码器读不到填充信息），因此它必须被写下来——
与 day081 第 10.2 节那条“漏传 ``placement``”是同一类：**沉默的忽略会让两条
本该不同的曲线重合成一条**。

## 跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── EncoderDecoderError（day079）
└── VariantError（day082，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    ├── GradientError                                （只在控制流里出现）
    └── AssemblyError → ParameterError               （特例：自己是“参数失败”的一种）
```

本层与 day079 一样**继承 day075 的三族而不是彼此**：``encoder_decoder`` 与本包是
**兄弟**。因此 ``except transformer_core.errors.ShapeError`` 能同时兜住三层，
而 ``except encoder_decoder.errors.ShapeError`` **兜不住本层**。
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


class VariantError(ValueError):
    """``arch_variants`` 这一族错误的基类（判据与 ``EncoderDecoderError`` 同源）."""


class ShapeError(CoreShapeError, VariantError):
    """形状不符：两路形状不匹配、隐藏维与权重列数不一致、γ/β 长度不符.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    day075~079 里写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, VariantError):
    """参数越界：变体名不认识、层数 <= 0、维度 <= 0、容差 <= 0.

    出路是**改调用**。本包不接受“参数不合法就取个默认值”这种兜底——
    一个拼错的变体名如果被兜成默认值，那么“跑的是哪个变体”就只能靠读代码才知道。
    """


class NumericError(CoreNumericError, VariantError):
    """数值不可用：非有限数、某一行的掩码全为 ``False``、损失里出现 ``nan``.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    """


class GradientError(VariantError):
    """梯度对不上：解析梯度与数值差分的误差超过容差.

    出路是**改推导**。它属于控制流而不是报告：一条对不上的梯度意味着这次堆叠不可信。
    本课把它放在**整条变体**上（而不是单个块上）——测的就是“掩码有没有被穿进反向”。
    """


class AssemblyError(ParameterError):
    """组装不成立：两个各自合法的部件拼不成一个合法的整体.

    四种典型情形：

    ```text
    把填充掩码交给解码器        → T5 那一路的接口不许（见模块说明）
    给编码器装交叉注意力        → 编码器没有第二路 K/V，交叉注意力没有定义
    解码器的自注意力不是因果的   → 它在训练时直接看到要预测的下一个 token
    掩码与序列长度不符          → 一张 (3, 5) 的掩码配一段 4 个 token 的序列
    ```

    **它同时是一种“参数失败”**（继承 :class:`ParameterError`），但值得有自己的名字：
    它要求调用方**同时**看两个部件（掩码与流、两路流的宽度），
    而“层数 <= 0”只需要改一个数——修法不同，族就不同。
    """


#: 五个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：两路形状不符时那个加法/乘法根本没有定义",
    "ParameterError": "改调用：变体名与维度是调用点的一次决定，本包不做默认值兜底",
    "AssemblyError": "改调用：掩码与流的搭配要一起看——本包绝不把跨流掩码当成一个默认值",
    "NumericError": "改数据或改实现：形状对、式子有定义，但算出来的东西没有意义",
    "GradientError": "改推导（或显式承认容差低于分辨率并把它调大）：梯度对不上时变体不可信",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "AssemblyError": AssemblyError,
    "NumericError": NumericError,
    "GradientError": GradientError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise VariantError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "VariantError",
]
