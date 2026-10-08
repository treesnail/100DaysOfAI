"""``hf_source`` 的失败族：**读源码时每一种失败该谁去修**（day085 / M7-D9）.

分族的依据与前九天逐字相同——**按“谁的错、该谁去修”分**：

```text
ShapeError        形状不符    头数不能整除隐藏维、掩码与打分形状不一致、位置数超表长  → 改调用
ParameterError    参数越界    temperature/top_k/top_p/penalty/max_new_tokens 非法    → 改调用
AssemblyError     组装不成立  把加性掩码当权重、把交叉权重当自注意力、
                              fused 投影的列数与 heads 对不上                      → 改调用（换配置）
GenerationError   生成策略失败 核为空、beam 宽度与词表冲突、长度惩罚为负            → 改调用（换策略）
NumericError      数值不可用  非有限数、一行之和不是 1、logits 全为 -inf            → 改数据或改实现
SourceError       本族基类
```

## 一、本课**仍然缺席**的那一族：`GradientError`

day083 缺席 `GradientError` 的理由是"本课不改任何算术"。**今天这个理由不成立了**——
本课真的写了新算术（融合投影、分头、加性掩码、四种生成策略）。但它仍然缺席，
理由换成了另一条，而这条边界更值得写下来：

```text
源码精读读的是**推理路径**。
  Hugging Face 的 ``modeling_gpt2.py`` 与 ``modeling_bert.py`` 里，
  反向不是被"写"出来的——它由 ``autograd`` 从这张计算图上**推**出来。
  因此"解析 vs 数值"那一族在本课没有对应物：**没有手写的反向公式可以错**。
```

这条缺席是一条**纪律的兑现**：能"改推导"的地方必须是有人真写了推导的地方。
本课对 HF 的全部断言都落在**前向与生成**上，因此也只能在这一侧被反驳。

## 二、本课**新增**的那一族：`GenerationError`

前九天没有这一族，因为前九天没有"从分布里选一个 token"这件事。
它之所以值得单独一族，是因为它的失败方式与 `ParameterError` 不同：

```text
ParameterError    "这个数不合法"        temperature = 0、top_k = 0、层号越界
GenerationError   "这个策略不成立"      top_p = 0.02 在 3 个候选上把核清空；
                                        num_beams = 1 却要求 beam 搜索；
                                        长度惩罚为负 ⇒ "越长越差"是反的
```

两者的修法都写"改调用"，但**要改的那个东西不同**：前者改一个数，
后者改**这一次生成用哪套策略**。day070 起的那条纪律在这里再兑现一次：
**"没写该怎么办"与"这一族不需要处理"读起来是一样的**，因此两族的出路必须分别写出来。

## 三、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── EncoderDecoderError（day079）
├── VariantError（day082）
├── ExplainError（day083）
└── SourceError（day085，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    └── AssemblyError → ParameterError               （特例：自己是“参数失败”的一种）
        GenerationError → ParameterError             （同上：策略不成立也是“参数失败”的一种）
```

与前两层一样，本层**继承 day075 的三族而不是彼此**：``explainability`` 与本包是
**兄弟**。因此 ``except transformer_core.errors.ShapeError`` 能同时兜住五层，
而 ``except explainability.errors.ShapeError`` **兜不住本层**。
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


class SourceError(ValueError):
    """``hf_source`` 这一族错误的基类（判据与 ``ExplainError`` 同源）."""


class ShapeError(CoreShapeError, SourceError):
    """形状不符：头数不能整除隐藏维、掩码与打分形状不一致、位置数超过位置表长.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前九天里写 ``except ShapeError`` 的代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, SourceError):
    """参数越界：``temperature`` 非正、``top_k`` 非正、``penalty`` 非正、层号越界.

    出路是**改调用**。本包不接受"参数不合法就取个默认值"这种兜底——
    一个被兜住的 ``top_k = 0`` 会退化成"全词表采样"，而它与"按预算采样"
    在前几步的输出上**看起来一样**。
    """


class NumericError(CoreNumericError, SourceError):
    """数值不可用：非有限数、一行之和不是 1、全是 ``-inf`` 的 logits.

    出路是**改数据或改实现**：形状是对的、式子有定义，但算出来的东西没有意义。
    第三条值得单独说：一行全为 ``-inf`` 时 softmax 的分母是 0——
    Hugging Face 在 ``top_p`` 里用"至少留一个"来避免它（见 :mod:`generation`），
    而本包把"真的空了"当场拒绝，而不是交给 ``nan`` 去污染下游。
    """


class AssemblyError(ParameterError):
    """组装不成立：两个各自合法的部件拼不成一个合法的整体.

    三种典型情形：

    ```text
    把加性掩码当权重     一次减了 1e9 的"打分"与一张 (n, n) 的权重表形状一样
    fused 投影切错列     c_attn 的 3·hidden 维按 hidden 切——切错**不报错**，
                         只是 q/k/v 各自混进了别人的维度
    把交叉权重当自注意力  n_tgt == n_src 时形状完全一样（day083 记下的同一条）
    ```

    **它同时是一种“参数失败”**（继承 :class:`ParameterError`），但值得有自己的名字：
    它要求调用方**同时**看两个部件（投影与切法、权重与掩码）——
    而"层号越界"只需要改一个数：修法不同，族就不同。
    """


class GenerationError(ParameterError):
    """生成策略不成立：核被清空、beam 宽度与词表冲突、长度惩罚为负.

    它与 :class:`ParameterError` 的差别不在"合法性"，而在**策略层面**：
    ``top_p = 0.02`` 是一个合法的概率，只是在一个 3 个候选的分布上它把核清空了；
    ``top_k = 5`` 在词表只有 3 个 token 时也不是"参数错"，而是"策略写得太宽"。

    出路是**改调用**——但要改的是"这一次生成用哪套策略"，
    而不是把那个数字改成另一个随便的数字。
    """


#: 五个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：头数与隐藏维、掩码与打分、位置数与表长必须先对齐",
    "ParameterError": "改调用：temperature / top_k / top_p / penalty / 层号都是调用点的一次决定",
    "AssemblyError": "改调用：投影与切法、权重与掩码要一起看——本包绝不静默截断",
    "GenerationError": "改调用（换策略）：核被清空、beam 宽度、长度惩罚都属于'这次怎么生成'",
    "NumericError": "改数据或改实现：非有限数、和不为 1、全 -inf 的 logits 都属于'数值不可用'",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "AssemblyError": AssemblyError,
    "GenerationError": GenerationError,
    "NumericError": NumericError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise SourceError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

__all__ = [
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "GenerationError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "SourceError",
]
