"""``inference_optim`` 的失败族：**推理优化里每一种失败该谁去修**（day087 / M7-D11）.

分族的依据与前十一天逐字相同——**按“谁的错、该谁去修”分**。今天这一批是三族全新的，
而它们都出现在同一个地方：**推理路径上的那次装箱**。

```text
ShapeError       形状不符     缓存行数与位置对不上、量化块大小与矩阵宽度不整除、
                              激活的字节数与批的形状对不上                       → 改调用
ParameterError   参数越界     bits 不在 {4, 8, 16, 32}、max_batch 非正、
                              block_size 非正、预算为负                          → 改调用
CacheError       缓存不成立   追加超出容量、层数不匹配、位置越界、重复 prefill      → 改调用（换策略）
QuantError       量化不成立   int4 的打包长度是奇数、per-channel 的通道数不匹配、
                              scale 为 0（整块被量化成同一个值）                  → 改数据或改方案
BudgetError      预算不成立   权重 + 缓存 + 激活 > 预算 ⇒ 这一次放不下            → 改部署或改精度
NumericError     数值不可用   非有限数、反量化后出现非有限值、误差上界失效         → 改数据或改实现
OptimError       本族基类
```

## 一、本课**连续第三天缺席**的那一族：``GradientError``

```text
day075~081   有手写反向            ⇒ 有 GradientError 那一族
day082~087   没有手写反向          ⇒ 这一族连续缺席六天
```

今天的理由比前两天更硬：**量化本身是不可微的**。
`round(x / scale)` 的导数几乎处处为 0、而在跳变点不存在——
真实世界的做法是"前向用量化权重、反向用直通估计（STE）"，
而那是**训练**的事。本课只做推理，因此连"该不该用 STE"都不需要回答。

一条纪律在这里第四次兑现：**能"改推导"的地方必须是有人真写了推导的地方。**

## 二、``CacheError`` 与 ``QuantError`` 为什么不是同一族

两者的出路完全不同：

```text
CacheError   "这一次的生成策略不成立"   缓存满了 ⇒ 换个容量、或换更短的上下文
QuantError   "这一份数据配这个方案不成立" 整块权重都是同一个值 ⇒ scale = 0，除了 0 什么都没有
```

前者是**运行期的一次决定**（生成多长、缓存开多大），后者是**数据侧的性质**
（这一层的权重是什么样子）。混成一族会让"该去调容量"的人去看权重，
而"该去换方案"的人去调容量。

## 三、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── EncoderDecoderError（day079）
├── VariantError（day082）
├── ExplainError（day083）
├── SourceError（day085）
├── IntegrationError（day086）
└── OptimError（day087，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    └── CacheError / QuantError / BudgetError / AssemblyError → ParameterError
```

本层同样**继承 day075 的三族而不是彼此**：``hf_source``、``hf_integration`` 与本包是
三个**兄弟**。因此 ``except transformer_core.errors.ShapeError`` 能同时兜住七层，
而 ``except hf_integration.errors.ShapeError`` **兜不住本层**——
这一条有它自己的用处：三天的读数在报告里必须能分开统计。
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


class OptimError(ValueError):
    """``inference_optim`` 这一族错误的基类（判据与 ``SourceError`` / ``IntegrationError`` 同源）."""


class ShapeError(CoreShapeError, OptimError):
    """形状不符：缓存的行数与位置对不上、量化块大小与矩阵宽度不整除.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前两天写的 ``except ShapeError`` 代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, OptimError):
    """参数越界：``bits`` 不在 ``{4, 8, 16, 32}``、``max_batch`` 非正、预算为负.

    出路是**改调用**。本包不接受"参数不合法就取个默认值"这种兜底——
    一个被兜住的 ``bits=3`` 会被就近取成 4，而它与"我说了要 4"在读数上**完全一样**，
    差别只在权重变得不准了（而那是"模型效果不好"，不是"我写错了参数"）。
    """


class NumericError(CoreNumericError, OptimError):
    """数值不可用：非有限数、反量化后出现非有限值、误差上界失效.

    第三条值得单独说：量化误差的上界 ``scale/2`` 是一条**可推导**的性质，
    而它成立的前提是"每个元素都落在 ``[-max_abs, max_abs]`` 内"。
    若输入里出现了 ``inf``，那个前提就不成立了，上界自然也不成立——
    本包在入口就把非有限数拒绝，而不是等到误差统计给出一个漂亮的谎话。
    """


class CacheError(ParameterError):
    """缓存不成立：追加超出容量、层数不匹配、位置越界、重复 prefill.

    它与 :class:`ParameterError` 的差别在**策略层**：
    ``max_new_tokens=10000`` 是一个合法的数，只是在现在的容量下放不下。
    出路是**改调用（换策略）**——把容量开大、把上下文缩短、或者换一条路线。
    """


class QuantError(ParameterError):
    """量化不成立：``int4`` 的打包长度是奇数、per-channel 的通道数不匹配、``scale`` 为 0.

    本族里最值钱的一条是 ``scale = 0``：它出现在**整块权重都是同一个值**的时候
    （例如一个刚初始化、或者被剪枝剪空的层）。那时反量化会把整块还原成 0，
    而"误差很大"与"这一层本来就没东西"读起来不一样——因此本包把它当场拒绝，
    而不是让它以"效果变差"的形式出现在下游。
    """


class BudgetError(ParameterError):
    """预算不成立：权重 + 缓存 + 激活的字节数超过预算 ⇒ **这一次放不下**.

    它值得独立成族，因为它的出路既不是"改一个参数"也不是"改数据"，
    而是**改部署或改精度**：换一张卡、换更小的精度、或者换一条能分页的路线。
    把它归进 ``ParameterError`` 的后果是——有人会去把 ``context_length`` 偷偷调小，
    而"上下文被截断"与"上下文够用"在生成结果上**不一定看得出来**。
    """


class AssemblyError(ParameterError):
    """组装不成立：两份各自合法的东西拼不成一个合法整体.

    ```text
    缓存与卡片不是同一个模型   层数对得上、隐藏维对得上，而它们本来是两个模型
    批里长度不齐              一次前向要求等宽；长度不齐时"补到最长"与"补到预算"是两个不同的数
    ```
    """


#: 七个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：缓存行数与位置、量化块与宽度、激活与批形状必须先对齐",
    "ParameterError": "改调用：bits / max_batch / block_size / 预算都是调用点的一次决定",
    "CacheError": "改调用（换策略）：容量、上下文长度、prefill 次数都属于'这一次怎么生成'",
    "QuantError": "改数据或改方案：打包长度、通道数、scale=0 都属于'这份数据配不上这个方案'",
    "BudgetError": "改部署或改精度：放不下时该换卡、换精度或换路线，而不是偷偷截断上下文",
    "NumericError": "改数据或改实现：非有限数、误差上界失效都属于'数值不可用'",
    "AssemblyError": "改组装：两份各自合法的东西拼不成立时，两边要**同时**看",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "CacheError": CacheError,
    "QuantError": QuantError,
    "BudgetError": BudgetError,
    "NumericError": NumericError,
    "AssemblyError": AssemblyError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise OptimError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 本课**连续缺席**的那一族（写进常量，让"缺席"是一个可断言的事实而不是一段散文）.
ABSENT_FAMILY = "GradientError"

#: 缺席的理由（与 day085/day086 的那两条**不同**，这一条要写清楚）.
ABSENT_FAMILY_REASON = (
    "量化是不可微的：round(x/scale) 的导数几乎处处为 0、在跳变点不存在。"
    "真实世界的做法是前向用量化权重、反向用直通估计（STE）——而那是**训练**的事。"
    "本课只做推理，因此连'该不该用 STE'都不需要回答。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "BudgetError",
    "CacheError",
    "NumericError",
    "OptimError",
    "ParameterError",
    "QuantError",
    "ShapeError",
]
