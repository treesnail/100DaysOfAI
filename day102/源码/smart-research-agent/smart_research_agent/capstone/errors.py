"""``capstone`` 的失败族：把八项能力装成一条链时，每一种失败该谁去修（day099 / G1-D1）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批依旧全新，而它们都出现在同一个地方：**把分散的能力装成一条端到端调用链**。

```text
ManifestError    清单不成立   子包里解析不到 / 清单与常量表对不上 / 承担者不在池子里 → 改清单或改包
CapabilityError  能力不成立   未知能力 / 某能力没有承担子包 / 某能力没有阶段落点      → 改能力表
StageError       阶段不成立   阶段次序或条数与 ASSEMBLY_STAGES 不符                  → 改实现或改阶段表
AssemblyError    装配不成立   同一输入两次运行不一致（不确定），或某个阶段失败        → 改实现（去掉随机/时间）
DocumentError    文档不成立   渲染出的文档漏掉了某项能力                             → 改渲染器
NumericError     数值不可用   非有限读数、上界/下界非法                              → 改数据或改实现
ParameterError   参数越界     未知子包名 / 未知性质 / 序号错位                       → 改调用
CapstoneError    本族基类
```

## 一、``AssemblyError`` 为什么要独立成族

它是本课唯一一族"看起来什么都没错，但结果不一样"的失败：

```text
同一个问题跑两次 ⇒ 读数不同 ⇒ 依赖它的一切（评估、对账、回归）全部作废
```

这类失败的病因**不在某一行**，而在"整条链里混进了一个没有固定的量"——
时间戳、字典序、全局游标、LLM 采样。把它归进 :class:`NumericError`
会得到"某个数不对"的诊断，而真正要改的是"去掉那个未固定的量"。
因此本族在**整条链**上做一次对照（两次运行逐位比），而不是逐个字段校验。

## 二、``ManifestError`` 与 ``CapabilityError`` 为什么不是同一族

两者的出路完全不同：

```text
ManifestError    "清单指到了不存在的东西"   子包名写错 / 包被改没了 ⇒ 改清单或改包
CapabilityError  "能力表本身站不住"         未知能力 / 没有承担子包 ⇒ 改能力表
```

前者是**清单与包**之间的问题（可以被 ``importlib`` 直接验证），
后者是**能力表内部**的问题（表自己就不完整）。混成一族会让"把名字改对"
的人去重写能力表，而"补一项能力"的人去查子包。

## 三、``DocumentError`` 今天**回来**了

```text
day0xx  documents.errors.DocumentError   第一次给"文档解析失败"命名，此后一直缺席
day099  capstone.DocumentError           本课把"渲染出的文档"变成一等公民——
                                         漏掉一项能力的文档会以"看起来完整"的形式交付
```

回来的理由与 day096 请回 ``CheckpointError`` 的理由同源：**那个名字背后的纪律
（"做不到要被报成一个可读的结论"）在新场景里重新需要一次控制流事件**。
但基类从 ``Exception`` 换成 :class:`CapstoneError`（``ValueError`` 的子类），
好让它能被同一条 ``except ValueError`` 兜住，与其余六族同源。

## 四、``GradientError`` 今天继续缺席

```text
day090   抛 GradientError：解析梯度与数值差分对不上
day094   抛 GradientError：梯度真的会爆 ⇒ 做成运行期守卫
day096   缺席：训练回路里唯一的梯度事件是被调用的那一行
day099   缺席：本日是**装配与渲染**，一个新式子都没有写
```

本课装配的链里唯一的"梯度事件"仍是被调用的那一行（``finetune`` 那一侧），
本课连一次反向都没有触发。因此这一族今天不属于本包——它属于被调用的代码。

## 五、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前十三天那些包）
├── BridgeError（day088）
├── TorchPipelineError（day096）
└── CapstoneError（day099，本模块）
    ├── ManifestError / CapabilityError / StageError / DocumentError
    └── AssemblyError / NumericError / ParameterError   （本族独立命名，不继承别人）
```

本层**不继承 day075 的三族**（与 ``principle_map`` 不同）：今天要报的是"装配"
这一层的失败，而它们与"形状 / 数值 / 参数"在处置方向上并不同源——
``ShapeError`` 是"维数对不上"，而 :class:`StageError` 是"链的形状对不上"。
把它们混成一家，会让 ``except ShapeError`` 这种既有写法在结业项目里
悄悄多兜住几族，而报告里的分类统计会因此失真。
"""

from __future__ import annotations


class CapstoneError(ValueError):
    """``capstone`` 这一族错误的基类（判据与 ``BridgeError`` / ``TorchPipelineError`` 同源）."""


class ManifestError(CapstoneError):
    """清单不成立：子包解析不到、清单与常量表对不上、承担者不在候选池里.

    它值得独立成族，因为它的处置动作最具体：**改清单或改包**。
    一个解析不到的子包名是一条指向空气的边——它既不会被计数、也不会被报警，
    只会在覆盖报告里以"已覆盖"的形式出现。
    """


class CapabilityError(CapstoneError):
    """能力不成立：未知能力、某能力没有承担子包、或某项能力没有任何阶段落点.

    它与 :class:`ManifestError` 的分工是"表内 vs 表间"（见模块 docstring 第二节）。
    """


class StageError(CapstoneError):
    """阶段不成立：阶段次序或条数与 ``ASSEMBLY_STAGES`` 不符.

    它与 :class:`ManifestError` 的差别在**对象**：前者管"链跑了几段、按什么次序"，
    后者管"链上每一段的落点是不是真的存在"。
    """


class AssemblyError(CapstoneError):
    """装配不成立：同一输入两次运行不一致（不确定），或某个阶段失败.

    它是本课唯一一族"看起来什么都没错，但结果不一样"的失败（见模块 docstring 第一节）。
    """


class DocumentError(CapstoneError):
    """文档不成立：渲染出的文档漏掉了某项能力，或覆盖检查发现缺口.

    这个名字**今天回来**（见模块 docstring 第三节）：基类从 ``Exception``
    换成 ``CapstoneError``，好让它与其余各族的兜底口径一致。
    """


class NumericError(CapstoneError):
    """数值不可用：非有限读数、上界 / 下界非法（负数），或容差为负.

    读数非有限时，任何比较（相等 / 上界 / 下界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝，而不是让它以"这条性质不通过"的形式出现在报告里。
    """


class ParameterError(CapstoneError):
    """参数越界：未知子包名 / 未知能力 / 未知性质 / 序号错位 / 预算非正.

    出路是**改调用**。本包不接受"名字写错就取个默认值"这种兜底——
    一个被兜住的 ``owner="retreival"`` 会被计进覆盖报告，
    而它与"我确实接上了检索包"在报告里**完全一样**。
    """


#: 七个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ManifestError": "改清单或改包：子包名写错 / 包被改没了，先把名字改对再重建清单",
    "CapabilityError": "改能力表：未知能力或没有承担子包时，先把能力表补完整",
    "StageError": "改实现或改阶段表：链的形状（次序 / 条数）对不上时，先让两者对齐",
    "AssemblyError": "改实现：两次运行不一致时，去掉那个没有固定的量（时间 / 哈希序 / 采样）",
    "DocumentError": "改渲染器：文档漏掉某项能力时，先把那一项渲染进正文",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：子包名 / 能力名 / 性质名 / 序号都是调用点的一次决定",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ManifestError": ManifestError,
    "CapabilityError": CapabilityError,
    "StageError": StageError,
    "AssemblyError": AssemblyError,
    "DocumentError": DocumentError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise CapstoneError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 今天**回来**的族：``DocumentError``（day0xx 命名、此后缺席，本课把它收进本族）.
RETURNED_FAMILY: str | None = "DocumentError"

#: 回来的理由：**与 day096 请回 ``CheckpointError`` 的理由同源**.
RETURNED_FAMILY_REASON = (
    "day0xx 的 ``documents.errors.DocumentError`` 第一次给'文档解析失败'命名了，"
    "此后各课都把它当工具函数、不把它当一次控制流事件。"
    "day099 把'渲染最终版 README 与架构文档'变成结业链路的最后一个交付物"
    "（``document.render_readme`` / ``render_architecture``），于是这个名字**回来**了——"
    "但基类从 ``Exception`` 换成 ``CapstoneError(ValueError)``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余六族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是装配与渲染，一个新式子都没有写）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day096 的理由不同**.
ABSENT_FAMILY_REASON = (
    "本课装配的链里唯一的'梯度事件'仍然是被调用的那一行（``finetune`` 那一侧），"
    "本课连一次反向都没有触发；它是一个纯装配与渲染的包——"
    "检索器、生成器、指标、工具全是既有的，本课只把它们接起来、把结果渲染成文档。"
    "因此这一族今天不属于本包——它属于被调用的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "AssemblyError",
    "CapabilityError",
    "CapstoneError",
    "DocumentError",
    "ManifestError",
    "NumericError",
    "ParameterError",
    "StageError",
]
