"""``import_graph`` 的口径表（day103）.

一次性把这一课的名词表写全：**1 种扫描目标 / 2 类边 / 2 个方向 / 7 条性质（判据分三类）/
10 条笔记 / 5 条边界**。全部是常量，因此可以被测试逐键检查。

```text
1 种扫描目标   ``smart_research_agent/**/*.py``（每个 .py 是一个节点，__init__.py 是包节点）
2 类边         绝对导入（from smart_research_agent.a.b import X） / 相对导入（from .b import X）
2 个方向       下游（我依赖谁） / 上游（谁依赖我）
7 条性质       判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **依赖图天生可能不是 DAG——所以"线性顺序"这句话必须先说清"对谁而言"。
> 450 个模块里就有 4 个环；谁都不说清，报告里那两个数字就会互相矛盾。**

因此本课的第一、二条性质是**覆盖与可复算**，然后才是**线性序**（③）与**环是不是真的**（④），
最后是三条结构性质（未解析目标数为 0、闭包至少含自己、每个子包至少一个模块）。

## 二、为什么"边"要分两类

```text
绝对导入   from smart_research_agent.tools.base import BaseTool   ⇒ 目标写全了，谁都能读
相对导入   from .base import BaseTool                            ⇒ 目标要**减掉 level-1 层点**才算得出来
```

相对导入是本课唯一"会算错而且不会报错"的地方：少减一层点，边就指到隔壁模块去了，
而依赖图照样能画出来、照样能排序——**只有"未解析目标数为 0"这条性质会把它抓住**。

## 三、一条纪律：方向要写出来

```text
下游闭包 closure(x, "down")   从 x 出发沿边能到谁   ⇒ "我依赖谁"（少一个就跑不起来）
上游闭包 closure(x, "up")     谁能沿边到 x          ⇒ "动了我，谁受影响"
```

两个方向的答案**极不对称**（``utils`` 的上游很大、下游很小）。
不写方向就取一个默认值，会让报告里两个完全不同的数长得一模一样。
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# 1. 扫描目标
# --------------------------------------------------------------------------------------

#: 被扫描的包名.
PACKAGE_NAME = "smart_research_agent"

#: 被扫描的后缀与包入口名.
MODULE_SUFFIX = ".py"
INIT_STEM = "__init__"

# --------------------------------------------------------------------------------------
# 2. 两类边与两个方向
# --------------------------------------------------------------------------------------

EDGE_KIND_ABSOLUTE = "absolute"
EDGE_KIND_RELATIVE = "relative"

#: 两类边（顺序 = 解析时的判定顺序）.
EDGE_KINDS: tuple[str, ...] = (EDGE_KIND_ABSOLUTE, EDGE_KIND_RELATIVE)

EDGE_KIND_DESCRIPTIONS: dict[str, str] = {
    EDGE_KIND_ABSOLUTE: "absolute：目标写全了（from smart_research_agent.a.b import X）",
    EDGE_KIND_RELATIVE: "relative：目标是相对写法（from .b import X），要按 level 减层算出来",
}

DIRECTION_DOWN = "down"
DIRECTION_UP = "up"

#: 两个方向（顺序 = 渲染顺序：先"我依赖谁"）。
DIRECTIONS: tuple[str, ...] = (DIRECTION_DOWN, DIRECTION_UP)

DIRECTION_DESCRIPTIONS: dict[str, str] = {
    DIRECTION_DOWN: "down（下游）：从 x 出发沿边能到谁——'我依赖谁'",
    DIRECTION_UP: "up（上游）：谁能沿边到 x——'动了我，谁受影响'",
}

# --------------------------------------------------------------------------------------
# 3. 七条性质（判据分三类）
# --------------------------------------------------------------------------------------

CRITERION_EQUALITY = "equality"
CRITERION_UPPER_BOUND = "upper_bound"
CRITERION_LOWER_BOUND = "lower_bound"

#: 三类判据（顺序 = 从"逐位相同"到"有方向的界"）.
CRITERIA: tuple[str, ...] = (CRITERION_EQUALITY, CRITERION_UPPER_BOUND, CRITERION_LOWER_BOUND)

CRITERION_DESCRIPTIONS: dict[str, str] = {
    CRITERION_EQUALITY: "相等（==）：读数与期望逐位 / 整数相同（没有容差空间）",
    CRITERION_UPPER_BOUND: "上界（<=）：读数不超过某个界（越少越好）",
    CRITERION_LOWER_BOUND: "下界（>=）：读数不低于某个底（越多越好）",
}

PROPERTY_GRAPH_COVERS_ALL_MODULES = "graph_covers_all_modules"
PROPERTY_GRAPH_IS_REPRODUCIBLE = "graph_is_reproducible"
PROPERTY_TOPOLOGICAL_ORDER_IS_VALID = "topological_order_is_valid"
PROPERTY_CYCLES_ARE_SOUND = "cycles_are_sound"
PROPERTY_IMPORT_TARGETS_RESOLVE = "import_targets_resolve"
PROPERTY_CLOSURES_INCLUDE_SELF = "closures_include_self"
PROPERTY_DEPENDENCY_DEPTH_IS_POSITIVE = "dependency_depth_is_positive"


@dataclass(frozen=True)
class PropertySpec:
    """一条性质：id + 说明 + 判据类别 + "失败意味着什么"."""

    id: str
    description: str
    criterion: str
    failure: str

    def __post_init__(self) -> None:
        from smart_research_agent.import_graph.errors import ParameterError

        if not self.id or not self.description or not self.failure:
            raise ParameterError(f"性质 {self.id!r} 的说明与失败语义都不能为空。")
        if self.criterion not in CRITERIA:
            raise ParameterError(
                f"性质 {self.id!r} 的判据 {self.criterion!r} 未知：可选 {list(CRITERIA)}。"
            )

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "id": self.id,
            "description": self.description,
            "criterion": self.criterion,
            "failure": self.failure,
        }

    def line(self) -> str:
        """一行说明：``[equality] graph_covers_all_modules | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 检查的顺序：先覆盖、再图、最后结构）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_GRAPH_COVERS_ALL_MODULES: PropertySpec(
        id=PROPERTY_GRAPH_COVERS_ALL_MODULES,
        description="每一个 ``.py`` 都是图里的一个节点（缺失 0）",
        criterion=CRITERION_EQUALITY,
        failure="某个模块没有进图——它在依赖图里与'这个模块不存在'一样",
    ),
    PROPERTY_GRAPH_IS_REPRODUCIBLE: PropertySpec(
        id=PROPERTY_GRAPH_IS_REPRODUCIBLE,
        description="同一批文件两次构建图**逐位相同**（节点与边都排序过）",
        criterion=CRITERION_EQUALITY,
        failure="图里混进了未固定的迭代序（集合序 / 字典序），两次构建不同",
    ),
    PROPERTY_TOPOLOGICAL_ORDER_IS_VALID: PropertySpec(
        id=PROPERTY_TOPOLOGICAL_ORDER_IS_VALID,
        description="凝缩后的线性序合法（跨分量的边一律指向后面）",
        criterion=CRITERION_EQUALITY,
        failure="线性序里出现一条向后指的边——那个顺序不是这个图的顺序",
    ),
    PROPERTY_CYCLES_ARE_SOUND: PropertySpec(
        id=PROPERTY_CYCLES_ARE_SOUND,
        description="报出来的每一个环都是**真环**（分量内任意两点互达）",
        criterion=CRITERION_EQUALITY,
        failure="报了一个假的环——把两个单向可达的模块说成互达",
    ),
    PROPERTY_IMPORT_TARGETS_RESOLVE: PropertySpec(
        id=PROPERTY_IMPORT_TARGETS_RESOLVE,
        description="每一条包内 import 的目标都能匹配到模块或包（未解析数 <= 0）",
        criterion=CRITERION_UPPER_BOUND,
        failure="有 import 目标对不上——多半是相对导入的层数算错了",
    ),
    PROPERTY_CLOSURES_INCLUDE_SELF: PropertySpec(
        id=PROPERTY_CLOSURES_INCLUDE_SELF,
        description="每个模块的下游闭包至少含它自己（最小闭包 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="某个模块连自己都不在自己的闭包里——闭包的起点算错了",
    ),
    PROPERTY_DEPENDENCY_DEPTH_IS_POSITIVE: PropertySpec(
        id=PROPERTY_DEPENDENCY_DEPTH_IS_POSITIVE,
        description="图里真的存在一条依赖链（最长下游闭包 >= 2，即至少有一条边）",
        criterion=CRITERION_LOWER_BOUND,
        failure="所有模块都不依赖任何模块——这张图退化成了 450 个孤点",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐）.
IMPORT_GRAPH_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.import_graph.errors import ParameterError

        raise ParameterError(f"未知的性质 {property_id!r}：可选 {list(IMPORT_GRAPH_PROPERTIES)}。")
    return property_id


def require_edge_kind(kind: str) -> str:
    """校验一类边（未知类别当场拒绝）."""
    if kind not in EDGE_KIND_DESCRIPTIONS:
        from smart_research_agent.import_graph.errors import ParameterError

        raise ParameterError(
            f"未知的边种类 {kind!r}：可选 {list(EDGE_KINDS)}——"
            "报告按这两类归拢边；类别外的名字既不会被渲染、也不会被覆盖检查发现。"
        )
    return kind


def require_direction(direction: str) -> str:
    """校验一个闭包方向（未知方向当场拒绝）."""
    if direction not in DIRECTION_DESCRIPTIONS:
        from smart_research_agent.import_graph.errors import ParameterError

        raise ParameterError(
            f"未知的闭包方向 {direction!r}：可选 {list(DIRECTIONS)}——"
            "不写方向会让'我依赖谁'与'谁依赖我'在报告里长得一模一样。"
        )
    return direction


def require_positive_int(label: str, value: int) -> int:
    """校验一个 >= 1 的整数（``limit`` 之类的参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        from smart_research_agent.import_graph.errors import ParameterError

        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的 limit 意味着报告永远空着，这条链路等于不存在。"
        )
    return value


if set(EDGE_KINDS) != set(EDGE_KIND_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "两类边的两张表不一致：EDGE_KINDS 与 EDGE_KIND_DESCRIPTIONS 必须逐键对齐——"
        "少一个键的类别在报告里只有名字、没有它从哪里来。"
    )

if set(DIRECTIONS) != set(DIRECTION_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError("两个方向的两张表不一致：DIRECTIONS 与 DIRECTION_DESCRIPTIONS 必须逐键对齐。")

_CRITERION_COVERAGE: dict[str, int] = {
    criterion: sum(1 for spec in PROPERTY_SPECS.values() if spec.criterion == criterion)
    for criterion in CRITERIA
}
if any(count == 0 for count in _CRITERION_COVERAGE.values()):  # pragma: no cover - 导入期不变式
    raise ValueError(
        f"三类判据至少要各有一条性质，当前分布 {_CRITERION_COVERAGE}——"
        "缺一类判据的后果是：'只能从下面兜住'的性质被写成了相等判据。"
    )

# --------------------------------------------------------------------------------------
# 4. 十条笔记
# --------------------------------------------------------------------------------------

IMPORT_GRAPH_NOTES: dict[str, str] = {
    "coverage_before_graph": (
        "画图的第一步不是说清'怎么解析 import'，而是说清'哪些文件必须进得来'："
        "一份没进图的模块，与'这个模块不存在'在依赖图里读起来一样。"
    ),
    "relative_imports_are_the_trap": (
        "相对导入是本课唯一'会算错而且不会报错'的地方："
        "少减一层点，边就指到隔壁——图照样画得出来，只有'未解析数为 0'能抓住它。"
    ),
    "graph_need_not_be_a_dag": (
        "依赖图天生可能不是 DAG：450 个模块里就有 4 个环。"
        "谁都不说清'线性序对谁而言'，报告里那两个数字就会互相矛盾。"
    ),
    "cycles_are_a_finding_not_a_bug": (
        "一个环是一个**读数**，不是一次失败：本课把它印出来（是 4 个），"
        "而不是让一条'图必须无环'的性质替我们把它删掉。"
    ),
    "lazy_import_breaks_the_loop": (
        "``api ↔ tools`` 这个环是**真的**：``tools/image_analysis.py`` 在函数体里写了 "
        "``from smart_research_agent.api.app import default_llm``——一个延迟 import，"
        "把环留在了图里、挡在了运行期之外。"
    ),
    "condensation_is_the_order_that_exists": (
        "有环的图没有线性序，但它的**凝缩图**有：先把环当成一个整体，再排序。"
        "这就是本课 topological_order 的长度是'分量数'而不是'模块数'的原因。"
    ),
    "direction_must_be_stated": (
        "下游闭包与上游闭包极不对称（utils 的上游很大、下游很小）："
        "不写方向就取默认值，会让两个完全不同的数长得一模一样。"
    ),
    "same_package_is_not_the_same_module": (
        "同一包内的模块也是两个节点：``from .base import X`` 是一条边，"
        "它和 ``from smart_research_agent.pkg import X`` 指的不是同一份东西。"
    ),
    "package_rollup_is_a_coarser_read": (
        "把模块折成子包会得到一张更粗的图（节点从几百降到几十）："
        "它更容易看出'哪两个包互相咬住'，但会丢掉'到底哪两个文件在互相咬'。"
    ),
    "reproducible_graph_is_the_deliverable": (
        "本课真正的交付物不是'一张好看的依赖图'，而是一份**可被重算出同一结果**的图。"
    ),
}

#: 十条笔记的键（顺序即写入顺序）.
IMPORT_GRAPH_NOTES_ORDER: tuple[str, ...] = tuple(IMPORT_GRAPH_NOTES)

# --------------------------------------------------------------------------------------
# 5. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

IMPORT_GRAPH_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：解析只用标准库 ``ast``，不 import 被扫描的模块",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件；被扫描的 ``.py`` 是被**读取**的，不是被改写的",
    "本包**不做 import**：它读的是 import **语句**，不是真的导入——"
    "因此一个函数体里的延迟 import 也会被记成一条边（这正是 ``api ↔ tools`` 那个环的来处）",
    "本包的全部读数**离线、确定性**：只读文件系统与 ``ast``，不联网、不读任何环境变量密钥",
    "本包**不判断分层好坏**：它只如实记录「谁 import 了谁」，"
    "不评价「这个环该不该存在」——那是读图的人要做的事",
)


def docs_of_kind(kind: str) -> str:
    """返回一类边的一句话解释（未知类别当场拒绝）."""
    return EDGE_KIND_DESCRIPTIONS[require_edge_kind(kind)]


def docs_of_direction(direction: str) -> str:
    """返回一个方向的一句话解释（未知方向当场拒绝）."""
    return DIRECTION_DESCRIPTIONS[require_direction(direction)]


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in IMPORT_GRAPH_PROPERTIES)


__all__ = [
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "DIRECTIONS",
    "DIRECTION_DESCRIPTIONS",
    "DIRECTION_DOWN",
    "DIRECTION_UP",
    "EDGE_KINDS",
    "EDGE_KIND_ABSOLUTE",
    "EDGE_KIND_DESCRIPTIONS",
    "EDGE_KIND_RELATIVE",
    "IMPORT_GRAPH_BOUNDARIES",
    "IMPORT_GRAPH_NOTES",
    "IMPORT_GRAPH_NOTES_ORDER",
    "IMPORT_GRAPH_PROPERTIES",
    "INIT_STEM",
    "MODULE_SUFFIX",
    "PACKAGE_NAME",
    "PROPERTY_CLOSURES_INCLUDE_SELF",
    "PROPERTY_CYCLES_ARE_SOUND",
    "PROPERTY_DEPENDENCY_DEPTH_IS_POSITIVE",
    "PROPERTY_GRAPH_COVERS_ALL_MODULES",
    "PROPERTY_GRAPH_IS_REPRODUCIBLE",
    "PROPERTY_IMPORT_TARGETS_RESOLVE",
    "PROPERTY_TOPOLOGICAL_ORDER_IS_VALID",
    "PROPERTY_SPECS",
    "PropertySpec",
    "docs_of_direction",
    "docs_of_kind",
    "property_specs",
    "require_direction",
    "require_edge_kind",
    "require_positive_int",
    "require_property",
]
