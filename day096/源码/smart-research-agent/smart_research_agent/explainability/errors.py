"""``explainability`` 的失败族：**做可解释性时每一种失败该谁去修**（day083 / M7-D8）.

分族的依据与前八天逐字相同——**按“谁的错、该谁去修”分**：

```text
ShapeError        形状不符    权重不是方阵、掩码与权重形状不一致、标签个数不符   → 改调用
ParameterError    参数越界    层号/头号越界、阈值非法、级数或宽度 <= 0          → 改调用
AssemblyError     组装不成立  把交叉权重当自注意力、层数与记录不符、
                              多头拼接后与单头对不上                          → 改调用（换配置）
NumericError      数值不可用  非有限数、一行之和不是 1、熵超过天花板            → 改数据或改实现
ExplainError      本族基类
```

## 一、本课**缺席**的那一族：`GradientError`

前八天的五族里有 `GradientError`（“改推导”）。今天它不在，理由值得写下来：

```text
day075~082   每一课都交出新公式 ⇒ 新公式可能写错 ⇒ 需要"解析 vs 数值"那一族
day083       本课不改任何算术：它读的是**已经算好**的权重
             ⇒ 没有新公式可错，因此没有"改推导"这一族
```

这条缺席不是省略，而是一条**边界**：可解释性不能替训练兜底。
它读到的权重是不是"好"的，取决于模型怎么被训（day081 的四个旋钮）——
本课只负责**把权重读出来、画出来、并给出可被断言的读数**。

## 二、新增的那一条拒绝：把交叉权重当成自注意力

这是本课最像 bug 的一处，而它不会报错：

```text
自注意力权重 (n, n)     方阵：每一行是"第 i 个位置怎么看第 j 个位置"
交叉权重     (n_tgt, n_src)  长方形：每一行是"第 i 个解码位置怎么看第 j 个源位置"
```

当 `n_tgt == n_src` 时两者的**形状完全一样**，而它们的含义不同：
一张对角线上有质量、另一张可能一行集中在某个源位置。
把它们混在一起时，热力图照样画得出来、熵照样算得出来——
只是"这张图在说什么"已经变了。本包因此在
:class:`~smart_research_agent.explainability.types.AttentionRecord` 里留下
``stream`` 字段（``self`` / ``cross``），并在需要区分的地方**显式拒绝**。

## 三、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── EncoderDecoderError（day079）
├── VariantError（day082）
└── ExplainError（day083，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    └── AssemblyError → ParameterError               （特例：自己是“参数失败”的一种）
```

本层与前两层一样**继承 day075 的三族而不是彼此**：``arch_variants`` 与本包是
**兄弟**。因此 ``except transformer_core.errors.ShapeError`` 能同时兜住四层，
而 ``except arch_variants.errors.ShapeError`` **兜不住本层**。
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


class ExplainError(ValueError):
    """``explainability`` 这一族错误的基类（判据与 ``VariantError`` 同源）."""


class ShapeError(CoreShapeError, ExplainError):
    """形状不符：权重不是方阵、掩码与权重形状不一致、标签个数与行数不符.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前八天里写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, ExplainError):
    """参数越界：层号/头号越界、阈值与 α 非法、级数或条形宽度 <= 0.

    出路是**改调用**。本包不接受“参数不合法就取个默认值”这种兜底——
    一个被兜住的 ``heads=0`` 会画出一张"看起来正常"的单头热力图。
    """


class NumericError(CoreNumericError, ExplainError):
    """数值不可用：非有限数、一行之和不是 1、某行的熵超过它的天花板.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    第三条（熵超过天花板）值得单独说：**熵不可能超过它能看到的位置数的对数**，
    因此它一旦越界，说明"权重"与"掩码"来自两次不同的前向。
    """


class AssemblyError(ParameterError):
    """组装不成立：两个各自合法的部件拼不成一个合法的整体.

    三种典型情形：

    ```text
    把交叉权重当自注意力       → n_tgt == n_src 时**形状完全一样**，只是含义变了
    层数与记录不符             → 3 层的记录配 4 层的模型
    多头拼接后与单头对不上      → 两处口径走散（本包把"heads=1 必须逐位等于 day082"写成一条判据）
    ```

    **它同时是一种“参数失败”**（继承 :class:`ParameterError`），但值得有自己的名字：
    它要求调用方**同时**看两个部件（权重与流、记录与模型）——
    而“层号越界”只需要改一个数：修法不同，族就不同。
    """


#: 四个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：权重与掩码/标签的形状必须先对齐，否则那张图在说什么都不确定",
    "ParameterError": "改调用：层号、头号、阈值与本包的级数都是调用点的一次决定",
    "AssemblyError": "改调用：权重与流要一起看——本包绝不把交叉权重当自注意力",
    "NumericError": "改数据或改实现：非有限数、和不为 1、熵超过天花板都属于'数值不可用'",
}

#: 本模块真正导出的四个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "AssemblyError": AssemblyError,
    "NumericError": NumericError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise ExplainError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "ExplainError",
    "NumericError",
    "ParameterError",
    "ShapeError",
]
