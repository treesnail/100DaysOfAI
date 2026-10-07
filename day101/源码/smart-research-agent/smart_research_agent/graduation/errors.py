"""``graduation`` 的失败族：把一次运行交付成四份产物时，每一种失败该谁去修（day100 / G2-D1）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批依旧全新，而它们都出现在同一个地方：**把"一次能跑的运行"变成"一份能被别人复算的交付"**。

```text
MilestoneError    里程碑不成立   天数 / 归档对不上，或里程碑被记成一个不可复算的数   → 改里程碑定义或改数据
DemoError         剧本不成立     重放不一致 / 剧本缺段 / 重放没跑起来               → 改剧本生成
InventoryError    清单不成立     重复发现 / 漏项 / 解析失败却没有被记录              → 改扫描
AssessmentError   自评不成立     未知能力 / 某项能力没有证据 / 评级越界               → 改自评规则
RoadmapError      规划不成立     有缺口却没有对应规划项 / 规划项指向不存在的缺口       → 改规划
NumericError      数值不可用     非有限读数 / 非法上下界                             → 改数据或改实现
ParameterError    参数越界       未知交付物名 / 未知缺口类型 / 非法阈值               → 改调用
GraduationError   本族基类
```

## 一、``DemoError`` 为什么要独立成族

它是本课唯一一族"**东西都在，但这一次跑不起来**"的失败：

```text
剧本要能重放：再跑一次必须给出同一份剧本
   重放不一致  →  交付物是"这一台机器这一次的输出"，不是一份产物
   剧本缺段    →  读者看到的是删过的剧本，而它读起来依然通顺
```

这两件事的处置动作是同一个：**回头改剧本生成**。因此它们同族，
而它们与 :class:`InventoryError`（"清单本身错了"）不同源——
后者要改的是"怎么数"，前者要改的是"怎么生成剧本"。

## 二、``AssessmentError`` 与 ``RoadmapError`` 为什么不是同一族

```text
AssessmentError  "自评站不住"   未知能力 / 没有证据 / 评级越界 ⇒ 改自评规则
RoadmapError     "规划对不上"   有缺口没规划 / 规划指向不存在的缺口 ⇒ 改规划
```

前者管"分数从哪里来"，后者管"缺口与下一步对不对得上"。
混成一族会让"给某一项能力补一条证据"的人去改规划表，
而"给一个缺口补一条下一步"的人去调评分规则。

## 三、``VersionError`` 今天**回来**了

```text
day065   indexing.errors.VersionError   第一次给"索引版本对不上"命名，此后各课都只把它当工具函数
day100   graduation.VersionError        本课把"100 天里程碑"归档成一个**可复算的版本事件**——
                                        "这一次交付"给出一份摘要，摘要相同即同一份产物
```

回来的理由与 day099 请回 ``DocumentError`` 的理由同源：**那个名字背后的纪律
（"对不上要被报成一个可读的结论"）在新场景里重新需要一次控制流事件**。
但基类从 ``Exception`` 换成 :class:`GraduationError`（``ValueError`` 的子类），
好让它能被同一条 ``except ValueError`` 兜住，与其余六族同源。

## 四、``GradientError`` 今天继续缺席

```text
day090   抛 GradientError：解析梯度与数值差分对不上
day094   抛 GradientError：梯度真的会爆 ⇒ 做成运行期守卫
day096   缺席：训练回路里唯一的梯度事件是被调用的那一行
day099   缺席：本日是**装配与渲染**，一个新式子都没有写
day100   缺席：本日是**交付与归档**，同样一个新式子都没有写
```

本课装配的四份产物里唯一的"梯度事件"仍是被调用的那一行（``finetune`` 那一侧），
本课连一次反向都没有触发。因此这一族今天不属于本包——它属于被调用的代码。

## 五、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前十四天那些包）
├── BridgeError（day088）
├── TorchPipelineError（day096）
├── CapstoneError（day099）
└── GraduationError（day100，本模块）
    ├── MilestoneError / DemoError / InventoryError
    └── AssessmentError / RoadmapError / NumericError / ParameterError
```

本层**不继承 day099 的七族**：今天要报的是"交付"这一层的失败，
而它们与"装配"那七族在处置方向上并不同源——``StageError`` 是"链的形状对不上"，
而 :class:`DemoError` 是"同一份剧本重放不出来"。把它们混成一家，
会让 ``except CapstoneError`` 这种既有写法在结业交付里悄悄多兜住几族，
而报告里的分类统计会因此失真。
"""

from __future__ import annotations


class GraduationError(ValueError):
    """``graduation`` 这一族错误的基类（与 ``CapstoneError`` / ``TorchPipelineError`` 同源）."""


class MilestoneError(GraduationError):
    """里程碑不成立：天数 / 归档对不上，或里程碑被记成一个不可复算的数.

    它值得独立成族，因为它的处置动作最具体：**改里程碑定义或改数据**。
    一个"100 天"如果来自手写的一个 100，而不是从一份可核对的读数推出，
    那么它在报告里与"我们真的学了 100 天"读起来一模一样。
    """


class DemoError(GraduationError):
    """剧本不成立：重放不一致、剧本缺段、或重放没跑起来（见模块 docstring 第一节）.

    它与 :class:`InventoryError` 的分工是"生成 vs 清点"：
    前者要改"怎么生成剧本"，后者要改"怎么数子包"。
    """


class InventoryError(GraduationError):
    """清单不成立：重复发现、漏项、解析失败却没有被记录.

    它与 day099 的 ``ManifestError`` 不同源：那里管的是"12 个候选池里的名字能不能解析"，
    本课管的是"**全部**子包有没有被数清、有没有被数重"。
    """


class AssessmentError(GraduationError):
    """自评不成立：未知能力、某项能力没有任何证据、或评级越界（见模块 docstring 第二节）.

    本包不接受"没有证据就给一个默认分"这种兜底——一个被兜住的默认分
    在报告里与"我确实检查过这一项"完全一样。
    """


class RoadmapError(GraduationError):
    """规划不成立：有缺口没有对应规划项，或规划项指向一个不存在的缺口（见模块 docstring 第二节）.

    出路是**改规划**：让"缺口集合"与"规划项集合"逐键对上，
    否则一份"下一步"读起来很完整，却可能整段漏掉了最该补的那一项。
    """


class NumericError(GraduationError):
    """数值不可用：非有限读数、上界 / 下界非法（负数），或阈值非正.

    读数非有限时，任何比较（相等 / 上界 / 下界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝，而不是让它以"这条性质不通过"的形式出现在报告里。
    """


class ParameterError(GraduationError):
    """参数越界：未知交付物名 / 未知缺口类型 / 非法阈值（<= 0）/ 序号错位.

    出路是**改调用**。本包不接受"名字写错就取个默认值"这种兜底——
    一个被兜住的 ``kind="unclaime"`` 会被计进规划覆盖报告，
    而它与"我确实给这一类缺口配了规划"在报告里**完全一样**。
    """


#: 七个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "MilestoneError": "改里程碑定义或改数据：天数与归档对不上时，先把那份可复算的读数补上",
    "DemoError": "改剧本生成：重放不一致或剧本缺段时，先让两次重放给出同一份剧本",
    "InventoryError": "改扫描：重复发现或漏项时，先让'发现'与'记录'是同一个集合",
    "AssessmentError": "改自评规则：未知能力、没有证据或评级越界时，先把证据补齐",
    "RoadmapError": "改规划：有缺口没有下一步时，先让缺口集合与规划项逐键对上",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：交付物名 / 缺口类型 / 阈值 / 序号都是调用点的一次决定",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "MilestoneError": MilestoneError,
    "DemoError": DemoError,
    "InventoryError": InventoryError,
    "AssessmentError": AssessmentError,
    "RoadmapError": RoadmapError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise GraduationError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 今天**回来**的族：``VersionError``（day065 命名、此后缺席，本课把里程碑归档成一次版本事件）.
RETURNED_FAMILY: str | None = "VersionError"

#: 回来的理由：**与 day099 请回 ``DocumentError`` 的理由同源**.
RETURNED_FAMILY_REASON = (
    "day065 的 ``indexing.errors.VersionError`` 第一次给'索引版本对不上'命名了，"
    "此后各课都把它当工具函数、不把它当一次控制流事件。"
    "day100 把'100 天里程碑'归档成一个**可复算的版本事件**"
    "（``demo.build_transcript`` 给出的摘要即这一份产物的指纹），于是这个名字**回来**了——"
    "但基类从 ``Exception`` 换成 ``GraduationError(ValueError)``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余六族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是交付与归档，一个新式子都没有写）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day099 的理由同源（都是"没有一次反向"），但场景不同**.
ABSENT_FAMILY_REASON = (
    "本课装配的四份产物里唯一的'梯度事件'仍然是被调用的那一行（``finetune`` 那一侧），"
    "本课连一次反向都没有触发；它是一个纯交付与归档的包——"
    "剧本、清单、自评、规划全是把既有读数折成产物，本课一个式子都没有新增。"
    "因此这一族今天不属于本包——它属于被调用的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "AssessmentError",
    "DemoError",
    "GraduationError",
    "InventoryError",
    "MilestoneError",
    "NumericError",
    "ParameterError",
    "RoadmapError",
]
