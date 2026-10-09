"""``symbol_catalog`` 的口径表（day104）.

一次性把这一课的名词表写全：**1 种扫描目标 / 3 种承诺来源 / 4 类绑定 / 7 条性质
（判据分三类）/ 10 条笔记 / 5 条边界**。全部是常量，因此可以被测试逐键检查。

```text
1 种扫描目标   ``smart_research_agent/**/*.py``（每个 .py 一行）
3 种承诺来源   显式 declared（字面 __all__） / 无法静态读 computed / 没有 __all__ implicit
4 类绑定       每个**声明的名字**在本模块里的去向：defined / assigned / imported / unresolved
7 条性质       判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **一个模块的 ``__all__`` 是它对外的承诺；一个"承诺了却找不到落点"的名字，
> 比一个报错更难被发现——因为 ``__all__`` 读起来永远像一份已经兑现的清单。**

因此本课的第一、二条性质是**覆盖与可复算**，然后才问**每一个承诺有没有落点**（③），
最后是三条结构性质（没有重名承诺、每个承诺的模块至少承诺一个名字、跨模块重名确实存在）。

## 二、为什么"承诺来源"要分三种

```text
declared    __all__ = ["A", "B"]            ⇒ 一份可以逐字核对的清单（346 份）
computed    __all__ = sorted({...})          ⇒ 它也算承诺，但**这一课读不出来**（11 份）
implicit    根本没有人写过 __all__           ⇒ 只能退而求其次：顶层公开 def / class（100 份）
```

第三种与前面两种的性质完全不同：**没有 ``__all__`` 不等于没有承诺**，
只等于"这份承诺没有被写下来"。本课把三种来源分开记，因为它们回答的不是同一个问题。

## 三、一条纪律：绑定类别里必须有一类是"我承认我不知道"

```text
defined     这个名字在本模块里有 def / class                       ⇒ 落点最硬
assigned    这个名字在本模块里被赋过值（模块级）                    ⇒ 落点是变量
imported    这个名字是本模块某条 import 的别名                      ⇒ 落点在外面
unresolved  以上三类都不是                                        ⇒ 当场点名（幽灵导出）
```

这与 day102 的"四类归属"是同一条纪律的**第五次**应用：一个没人说得清归属的名字，
比一个报错更难被发现。因此 ``unresolved`` 不是解析器的失败，而是**纪律**。
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

#: ``__all__`` 这个名字（本课要读的那个特殊绑定）.
ALL_NAME = "__all__"

# --------------------------------------------------------------------------------------
# 2. 三种承诺来源
# --------------------------------------------------------------------------------------

SOURCE_DECLARED = "declared"
SOURCE_COMPUTED = "computed"
SOURCE_IMPLICIT = "implicit"

#: 三种承诺来源（顺序 = 从"最硬"到"最软"）.
PROMISE_SOURCES: tuple[str, ...] = (SOURCE_DECLARED, SOURCE_COMPUTED, SOURCE_IMPLICIT)

SOURCE_DESCRIPTIONS: dict[str, str] = {
    SOURCE_DECLARED: "declared：写了字面量 __all__，可以逐字核对",
    SOURCE_COMPUTED: "computed：写了 __all__ 但不是字面量（例如 sorted({...})），**这一课读不出来**",
    SOURCE_IMPLICIT: "implicit：没有 __all__，只能退而取顶层公开 def / class",
}

# --------------------------------------------------------------------------------------
# 3. 四类绑定
# --------------------------------------------------------------------------------------

BINDING_DEFINED = "defined"
BINDING_ASSIGNED = "assigned"
BINDING_IMPORTED = "imported"
BINDING_UNRESOLVED = "unresolved"

#: 四类绑定（顺序 = 判定顺序，**不能换**：先"本模块的定义"、再赋值、再导入）。
BINDING_KINDS: tuple[str, ...] = (
    BINDING_DEFINED,
    BINDING_ASSIGNED,
    BINDING_IMPORTED,
    BINDING_UNRESOLVED,
)

BINDING_DESCRIPTIONS: dict[str, str] = {
    BINDING_DEFINED: "defined：本模块里有 def / class（落点最硬）",
    BINDING_ASSIGNED: "assigned：本模块里被赋过值（落点是一个变量）",
    BINDING_IMPORTED: "imported：是本模块某条 import 的别名（落点在外面）",
    BINDING_UNRESOLVED: "unresolved：三类都不是——**幽灵导出**，当场点名",
}

# --------------------------------------------------------------------------------------
# 4. 七条性质（判据分三类）
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

PROPERTY_CATALOG_COVERS_ALL_MODULES = "catalog_covers_all_modules"
PROPERTY_CATALOG_IS_REPRODUCIBLE = "catalog_is_reproducible"
PROPERTY_PHANTOMS_ARE_SOUND = "phantom_exports_are_sound"
PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE = "qualified_export_names_are_unique"
PROPERTY_DUPLICATE_DECLARATIONS_WITHIN_MODULE = "duplicate_declarations_within_module"
PROPERTY_EVERY_DECLARING_MODULE_PROMISES = "every_declaring_module_promises_something"
PROPERTY_EXPORT_NAMES_ARE_REUSED = "export_names_are_reused_across_modules"


@dataclass(frozen=True)
class PropertySpec:
    """一条性质：id + 说明 + 判据类别 + "失败意味着什么"."""

    id: str
    description: str
    criterion: str
    failure: str

    def __post_init__(self) -> None:
        from smart_research_agent.symbol_catalog.errors import ParameterError

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
        """一行说明：``[equality] catalog_covers_all_modules | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 检查的顺序：先覆盖、再可复算、最后结构）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_CATALOG_COVERS_ALL_MODULES: PropertySpec(
        id=PROPERTY_CATALOG_COVERS_ALL_MODULES,
        description="每一个 ``.py`` 都是清单里的一行（缺失 0）",
        criterion=CRITERION_EQUALITY,
        failure="某个模块没有进清单——它在清单里与'这个模块什么都没承诺'一样",
    ),
    PROPERTY_CATALOG_IS_REPRODUCIBLE: PropertySpec(
        id=PROPERTY_CATALOG_IS_REPRODUCIBLE,
        description="同一批文件两次编出的清单**逐位相同**（模块与名字都排序过）",
        criterion=CRITERION_EQUALITY,
        failure="清单里混进了未固定的迭代序（集合序 / 字典序），两次构建不同",
    ),
    PROPERTY_PHANTOMS_ARE_SOUND: PropertySpec(
        id=PROPERTY_PHANTOMS_ARE_SOUND,
        description="报出来的每一个『幽灵导出』都**是真的**（名字在 ``__all__`` 之外一次都不出现）",
        criterion=CRITERION_EQUALITY,
        failure="报了一个假幽灵——把一个其实有落点的名字说成没落点（多半是解析漏了一类绑定）",
    ),
    PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE: PropertySpec(
        id=PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE,
        description="``(模块, 名字)`` 在清单里不重复（重复 0 个）",
        criterion=CRITERION_EQUALITY,
        failure="同一个模块里同一个名字被数了两遍——清单的总数就不可信了",
    ),
    PROPERTY_DUPLICATE_DECLARATIONS_WITHIN_MODULE: PropertySpec(
        id=PROPERTY_DUPLICATE_DECLARATIONS_WITHIN_MODULE,
        description="同一个 ``__all__`` 里不重复声明同一个名字（重复模块数 <= 0）",
        criterion=CRITERION_UPPER_BOUND,
        failure="一个名字在同一个 __all__ 里出现了两次——读的人不知道这一条到底算几条",
    ),
    PROPERTY_EVERY_DECLARING_MODULE_PROMISES: PropertySpec(
        id=PROPERTY_EVERY_DECLARING_MODULE_PROMISES,
        description="每一个写了字面量 ``__all__`` 的模块至少承诺一个名字（最小承诺数 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="有空 __all__——它写了一份承诺，却什么都没承诺",
    ),
    PROPERTY_EXPORT_NAMES_ARE_REUSED: PropertySpec(
        id=PROPERTY_EXPORT_NAMES_ARE_REUSED,
        description="至少有一个名字被两个以上的模块导出（跨模块重名数 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="所有名字都只出现在一个模块里——这份清单退化成了几百份互不相干的名单",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐）.
SYMBOL_CATALOG_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.symbol_catalog.errors import ParameterError

        raise ParameterError(f"未知的性质 {property_id!r}：可选 {list(SYMBOL_CATALOG_PROPERTIES)}。")
    return property_id


def require_promise_source(source: str) -> str:
    """校验一种承诺来源（未知来源当场拒绝）."""
    if source not in SOURCE_DESCRIPTIONS:
        from smart_research_agent.symbol_catalog.errors import ParameterError

        raise ParameterError(
            f"未知的承诺来源 {source!r}：可选 {list(PROMISE_SOURCES)}——"
            "报告按这三类归拢承诺；类别外的名字既不会被渲染、也不会被覆盖检查发现。"
        )
    return source


def require_binding_kind(kind: str) -> str:
    """校验一类绑定（未知类别当场拒绝）."""
    if kind not in BINDING_DESCRIPTIONS:
        from smart_research_agent.symbol_catalog.errors import ParameterError

        raise ParameterError(
            f"未知的绑定类别 {kind!r}：可选 {list(BINDING_KINDS)}——"
            "不写类别会让'本模块定义的'与'从别处导入的'在报告里长得一模一样。"
        )
    return kind


def require_positive_int(label: str, value: int) -> int:
    """校验一个 >= 1 的整数（``limit`` 之类的参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        from smart_research_agent.symbol_catalog.errors import ParameterError

        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的 limit 意味着报告永远空着，这条链路等于不存在。"
        )
    return value


if set(PROMISE_SOURCES) != set(SOURCE_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError("三种承诺来源的两张表不一致：PROMISE_SOURCES 与 SOURCE_DESCRIPTIONS 必须逐键对齐。")

if set(BINDING_KINDS) != set(BINDING_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError("四类绑定的两张表不一致：BINDING_KINDS 与 BINDING_DESCRIPTIONS 必须逐键对齐。")

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
# 5. 十条笔记
# --------------------------------------------------------------------------------------

SYMBOL_CATALOG_NOTES: dict[str, str] = {
    "coverage_before_reading_all": (
        "编清单的第一步不是说清'怎么读 __all__'，而是说清'哪些文件必须进得来'："
        "一份没进清单的模块，与'这个模块什么都没承诺'在清单里读起来一样。"
    ),
    "a_promise_without_a_landing_is_invisible": (
        "一个承诺了却找不到落点的名字，比一个报错更难被发现——"
        "因为 ``__all__`` 读起来永远像一份已经兑现的清单。"
    ),
    "three_sources_are_three_questions": (
        "declared / computed / implicit 是三个不同的问题："
        "写了字面量、写了但读不出来、根本没写。混成一类会让'没写'看起来像'写错'。"
    ),
    "unresolved_is_a_discipline": (
        "四类绑定里必须有一类是'我承认我不知道'：``unresolved`` 不是解析器的失败，而是纪律。"
    ),
    "phantom_is_a_reading_not_a_bug": (
        "本仓库里真的有一个幽灵导出（``transformer_stack/verify.py`` 的 ``STAGE_ITEMS``）："
        "本课把它印出来，而不是让一条'必须为 0'的性质把它遮掉。"
    ),
    "sort_is_the_third_discipline": (
        "模块按名字排序、声明的名字按名字排序、跨模块重名按出现次数与名字排序——"
        "三处显式排序，'两次构建逐位相同'才从'希望如此'变成'不可能不如此'。"
    ),
    "reuse_across_modules_is_the_reading": (
        "一个名字被几十个模块同时导出（如 ``ShapeError`` / ``NumericError``）："
        "这既是'同一件事在很多层各写了一遍'的证据，也是跨包契约的入口。"
    ),
    "implicit_promise_is_a_silent_gap": (
        "没有 ``__all__`` 不等于没有承诺，只等于'这份承诺没有被写下来'："
        "读者只能去猜哪些名字是对外的。"
    ),
    "declared_count_and_bound_count_differ": (
        "一个模块'承诺了几个名字'与'它一共绑定了几个名字'是两个数："
        "前者是它说出去的，后者是它手里有的。"
    ),
    "reproducible_catalog_is_the_deliverable": (
        "本课真正的交付物不是'一份好看的 API 清单'，而是一份**可被重算出同一结果**的清单。"
    ),
}

#: 十条笔记的键（顺序即写入顺序）.
SYMBOL_CATALOG_NOTES_ORDER: tuple[str, ...] = tuple(SYMBOL_CATALOG_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

SYMBOL_CATALOG_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：解析只用标准库 ``ast``，不 import 被扫描的模块",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件；被扫描的 ``.py`` 是被**读取**的，不是被改写的",
    "本包**不做 import**：它读的是 ``__all__`` 与 import **语句**，"
    "因此一个函数体里的延迟 import 也算一次绑定",
    "本包的全部读数**离线、确定性**：只读文件系统与 ``ast``，不联网、不读任何环境变量密钥",
    "本包**不判断承诺的好坏**：它只如实记录'谁承诺了什么、落到哪里'，"
    "不评价'这个模块该不该导出这个概念'——那是读清单的人要做的事",
)


def docs_of_source(source: str) -> str:
    """返回一种承诺来源的一句话解释（未知来源当场拒绝）."""
    return SOURCE_DESCRIPTIONS[require_promise_source(source)]


def docs_of_binding(kind: str) -> str:
    """返回一类绑定的一句话解释（未知类别当场拒绝）."""
    return BINDING_DESCRIPTIONS[require_binding_kind(kind)]


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in SYMBOL_CATALOG_PROPERTIES)


__all__ = [
    "ALL_NAME",
    "BINDING_ASSIGNED",
    "BINDING_DEFINED",
    "BINDING_DESCRIPTIONS",
    "BINDING_IMPORTED",
    "BINDING_KINDS",
    "BINDING_UNRESOLVED",
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "INIT_STEM",
    "MODULE_SUFFIX",
    "PACKAGE_NAME",
    "PROMISE_SOURCES",
    "PROPERTY_CATALOG_COVERS_ALL_MODULES",
    "PROPERTY_CATALOG_IS_REPRODUCIBLE",
    "PROPERTY_DUPLICATE_DECLARATIONS_WITHIN_MODULE",
    "PROPERTY_EVERY_DECLARING_MODULE_PROMISES",
    "PROPERTY_EXPORT_NAMES_ARE_REUSED",
    "PROPERTY_PHANTOMS_ARE_SOUND",
    "PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE",
    "PROPERTY_SPECS",
    "PropertySpec",
    "SOURCE_COMPUTED",
    "SOURCE_DECLARED",
    "SOURCE_DESCRIPTIONS",
    "SOURCE_IMPLICIT",
    "SYMBOL_CATALOG_BOUNDARIES",
    "SYMBOL_CATALOG_NOTES",
    "SYMBOL_CATALOG_NOTES_ORDER",
    "SYMBOL_CATALOG_PROPERTIES",
    "docs_of_binding",
    "docs_of_source",
    "property_specs",
    "require_binding_kind",
    "require_positive_int",
    "require_promise_source",
    "require_property",
]
