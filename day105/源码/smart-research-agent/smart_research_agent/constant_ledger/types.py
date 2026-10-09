"""``constant_ledger`` 的口径表（day105）.

一次性把这一课的名词表写全：**1 种扫描目标 / 2 类取值形态 / 4 类同名关系 / 7 条性质
（判据分三类）/ 10 条笔记 / 5 条边界**。全部是常量，因此可以被测试逐键检查。

```text
1 种扫描目标   ``smart_research_agent/**/*.py`` 里**模块级**的 UPPER_CASE 赋值
2 类取值形态   literal（可以用 ast.literal_eval 求出值） / opaque（求不出来）
4 类同名关系   unique / consistent / conflict / incomparable
7 条性质       判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **"同名不同值"本身不是错误，它是一个读数——
> 但它是这一课唯一一条能让人立刻去翻两个包的读数。**

因此本课把它整张印出来（``conflicts()``），只对它做两条判据：
"报出来的冲突是不是真的"（③），以及"同名表是不是把成员算全了"（④）。
只有调用点**显式**要求"这些同名常量必须一致"时，:func:`ledger.require_consistent`
才把它变成一次拒绝交付（:class:`ConflictError`）。

## 二、为什么"取值形态"只分两类

```text
literal   EPSILON = 1e-5 / TOLERANCE = 1e-9 / NAMES = ("a", "b")   ⇒ 能求出值，可以比较
opaque    ROOT = Path(__file__).resolve().parents[2]               ⇒ 求不出来，只能记"读不出来"
```

两类必须分开记：**把 opaque 当成"空值"去和别的模块比，会让两个根本无关的常量"看起来一致"**
——这正是 day102 那条纪律（"必须有一类是我承认我不知道"）在本课的形态。

## 三、一条纪律：四类同名关系里，"冲突"与"无法比较"不能混

```text
unique          只在一个模块里出现
consistent      跨模块，且能比较的那些**字面值全都相同**
conflict        跨模块，且能比较的那些字面值**不止一个**
incomparable    跨模块，但至少一侧是 opaque ⇒ 这一课**拒绝下结论**
```

`incomparable` 与 `conflict` 混成一类，会让"我算不出来"被读成"它们对不上"——
一个没人说得清归属的名字，比一个报错更难被发现；这句话在 day102 / day104 说过两次了。
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

#: 常量名的下限：**至少要两个字符**（`A = 1` 这种单字母不算"钉死的常量"）.
MIN_CONSTANT_NAME_LENGTH = 2

# --------------------------------------------------------------------------------------
# 2. 两类取值形态
# --------------------------------------------------------------------------------------

VALUE_LITERAL = "literal"
VALUE_OPAQUE = "opaque"

#: 两类取值形态（顺序 = 从"能比较"到"读不出来"）.
VALUE_KINDS: tuple[str, ...] = (VALUE_LITERAL, VALUE_OPAQUE)

VALUE_DESCRIPTIONS: dict[str, str] = {
    VALUE_LITERAL: "literal：可以用 ast.literal_eval 求出值，因而可以逐位比较",
    VALUE_OPAQUE: "opaque：求不出值（如 Path(__file__)、f-string、函数调用），只能记'读不出来'",
}

# --------------------------------------------------------------------------------------
# 3. 四类同名关系
# --------------------------------------------------------------------------------------

RELATION_UNIQUE = "unique"
RELATION_CONSISTENT = "consistent"
RELATION_CONFLICT = "conflict"
RELATION_INCOMPARABLE = "incomparable"

#: 四类同名关系（顺序 = 判定顺序，**不能换**）。
RELATIONS: tuple[str, ...] = (
    RELATION_UNIQUE,
    RELATION_CONSISTENT,
    RELATION_CONFLICT,
    RELATION_INCOMPARABLE,
)

RELATION_DESCRIPTIONS: dict[str, str] = {
    RELATION_UNIQUE: "unique：只在一个模块里出现",
    RELATION_CONSISTENT: "consistent：跨模块，且能比较的字面值全都相同",
    RELATION_CONFLICT: "conflict：跨模块，且能比较的字面值不止一个（**本课最值钱的读数**）",
    RELATION_INCOMPARABLE: "incomparable：跨模块，但至少一侧是 opaque ⇒ 拒绝下结论",
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

PROPERTY_LEDGER_COVERS_ALL_MODULES = "ledger_covers_all_modules"
PROPERTY_LEDGER_IS_REPRODUCIBLE = "ledger_is_reproducible"
PROPERTY_CONFLICTS_ARE_SOUND = "conflicts_are_sound"
PROPERTY_SHARED_CONSTANTS_ARE_SOUND = "shared_constants_are_sound"
PROPERTY_DUPLICATE_ASSIGNMENTS_WITHIN_MODULE = "duplicate_assignments_within_module"
PROPERTY_CONSTANTS_ARE_NUMEROUS = "constants_are_numerous"
PROPERTY_SHARED_CONSTANTS_EXIST = "shared_constants_exist"


@dataclass(frozen=True)
class PropertySpec:
    """一条性质：id + 说明 + 判据类别 + "失败意味着什么"."""

    id: str
    description: str
    criterion: str
    failure: str

    def __post_init__(self) -> None:
        from smart_research_agent.constant_ledger.errors import ParameterError

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
        """一行说明：``[equality] ledger_covers_all_modules | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 检查的顺序：先覆盖、再可复算、最后结构）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_LEDGER_COVERS_ALL_MODULES: PropertySpec(
        id=PROPERTY_LEDGER_COVERS_ALL_MODULES,
        description="每一个 ``.py`` 都是台账里的一行（缺失 0）",
        criterion=CRITERION_EQUALITY,
        failure="某个模块没有进台账——它在台账里与'这个模块一个常量都没定义'一样",
    ),
    PROPERTY_LEDGER_IS_REPRODUCIBLE: PropertySpec(
        id=PROPERTY_LEDGER_IS_REPRODUCIBLE,
        description="同一批文件两次编出的台账**逐位相同**（模块与常量名都排序过）",
        criterion=CRITERION_EQUALITY,
        failure="台账里混进了未固定的迭代序（集合序 / 字典序 / 绝对路径），两次构建不同",
    ),
    PROPERTY_CONFLICTS_ARE_SOUND: PropertySpec(
        id=PROPERTY_CONFLICTS_ARE_SOUND,
        description="报出来的每一组『同名不同值』都**是真的**（独立复核源码片段确实不同）",
        criterion=CRITERION_EQUALITY,
        failure="报了一组假冲突——把两个其实相同的赋值说成不同（多半是取值指纹算错了）",
    ),
    PROPERTY_SHARED_CONSTANTS_ARE_SOUND: PropertySpec(
        id=PROPERTY_SHARED_CONSTANTS_ARE_SOUND,
        description="同名表里每一项的成员集合与独立重数一致（不成立 0 项）",
        criterion=CRITERION_EQUALITY,
        failure="同名表把某个模块算漏了 / 算重了——它的成员集合就不是这个仓库的真成员",
    ),
    PROPERTY_DUPLICATE_ASSIGNMENTS_WITHIN_MODULE: PropertySpec(
        id=PROPERTY_DUPLICATE_ASSIGNMENTS_WITHIN_MODULE,
        description="同一个模块里不重复给同一个常量名赋值（重复模块数 <= 0）",
        criterion=CRITERION_UPPER_BOUND,
        failure="一个名字在同一个模块里被赋了两次——读的人不知道台账该记哪一个数",
    ),
    PROPERTY_CONSTANTS_ARE_NUMEROUS: PropertySpec(
        id=PROPERTY_CONSTANTS_ARE_NUMEROUS,
        description="台账里至少有一批常量（常量赋值总数 >= 500）",
        criterion=CRITERION_LOWER_BOUND,
        failure="一份几乎没有常量的台账——这说明扫描口径写错了，而不是仓库真的没有常量",
    ),
    PROPERTY_SHARED_CONSTANTS_EXIST: PropertySpec(
        id=PROPERTY_SHARED_CONSTANTS_EXIST,
        description="至少有一个常量名出现在两个以上的模块里（跨模块同名数 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="所有常量名都只出现在一个模块里——这份台账退化成了几百份互不相干的清单",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐）.
CONSTANT_LEDGER_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.constant_ledger.errors import ParameterError

        raise ParameterError(f"未知的性质 {property_id!r}：可选 {list(CONSTANT_LEDGER_PROPERTIES)}。")
    return property_id


def require_value_kind(kind: str) -> str:
    """校验一类取值形态（未知类别当场拒绝）."""
    if kind not in VALUE_DESCRIPTIONS:
        from smart_research_agent.constant_ledger.errors import ParameterError

        raise ParameterError(
            f"未知的取值形态 {kind!r}：可选 {list(VALUE_KINDS)}——"
            "不写形态会让'能比较'与'读不出来'在报告里长得一模一样。"
        )
    return kind


def require_relation(relation: str) -> str:
    """校验一类同名关系（未知类别当场拒绝）."""
    if relation not in RELATION_DESCRIPTIONS:
        from smart_research_agent.constant_ledger.errors import ParameterError

        raise ParameterError(
            f"未知的同名关系 {relation!r}：可选 {list(RELATIONS)}——"
            "不写关系会让'取值一致'与'取值冲突'在报告里长得一模一样。"
        )
    return relation


def require_positive_int(label: str, value: int) -> int:
    """校验一个 >= 1 的整数（``limit`` 之类的参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        from smart_research_agent.constant_ledger.errors import ParameterError

        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的 limit 意味着报告永远空着，这条链路等于不存在。"
        )
    return value


if set(VALUE_KINDS) != set(VALUE_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError("两类取值形态的两张表不一致：VALUE_KINDS 与 VALUE_DESCRIPTIONS 必须逐键对齐。")

if set(RELATIONS) != set(RELATION_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError("四类同名关系的两张表不一致：RELATIONS 与 RELATION_DESCRIPTIONS 必须逐键对齐。")

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

CONSTANT_LEDGER_NOTES: dict[str, str] = {
    "coverage_before_values": (
        "编台账的第一步不是说清'怎么读一个常量'，而是说清'哪些文件必须进得来'："
        "一份没进台账的模块，与'这个模块一个常量都没定义'读起来一样。"
    ),
    "same_name_is_not_same_thing": (
        "同一个名字在两百多份 .py 里各写一遍：**同名不等于同一件事**——"
        "台账要做的是把它们的**取值**摆在一起，而不是假设它们一定相等。"
    ),
    "conflict_is_a_reading_not_a_bug": (
        "本仓库里真的有一批'同名不同值'（本课把它整张印出来）："
        "它是一处**读数**，不是一次失败——只有调用点要求'必须一致'时它才成为失败。"
    ),
    "opaque_must_be_its_own_kind": (
        "把 opaque 当成'空值'去比较，会让两个根本无关的常量'看起来一致'："
        "四类关系里必须有一类是'我承认我算不出来'。"
    ),
    "literal_eval_is_the_ruler": (
        "取值指纹只用标准库 `ast.literal_eval`：它不执行代码、不 import、不联网，"
        "因此'这个值是什么'这件事有一个离线、确定的答案。"
    ),
    "sort_is_the_fifth_discipline": (
        "模块按名字排序、常量按（名字, 行号）排序、同名表按（-出现模块数, 名字）排序——"
        "三处显式排序，'两次构建逐位相同'才从'希望如此'变成'不可能不如此'。"
    ),
    "module_level_only": (
        "常量必须是**模块级**的：写在 class 体里或函数体里的 UPPER_CASE 不算——"
        "它们是那一层的实现细节，不是被钉死在外面的数。"
    ),
    "duplicate_assignment_is_ambiguous": (
        "同一个模块里给同一个常量名赋两次值，读的人就不知道台账该记哪一个数："
        "因此本课把它做成一条上界性质（⑤），而不是随便取一个。"
    ),
    "threshold_must_be_winning": (
        "第 ⑥ 条性质的阈值定在'这批文件必然成立'的地方：它是**存在性**断言，"
        "写成 == 某个精确数会在仓库合理地变化时误报失败。"
    ),
    "reproducible_ledger_is_the_deliverable": (
        "本课真正的交付物不是'一份常量表'，而是一份**可被重算出同一结果**的台账。"
    ),
}

#: 十条笔记的键（顺序即写入顺序）.
CONSTANT_LEDGER_NOTES_ORDER: tuple[str, ...] = tuple(CONSTANT_LEDGER_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

CONSTANT_LEDGER_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：解析只用标准库 ``ast`` 与 ``ast.literal_eval``，不 import 被扫描的模块",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件；被扫描的 ``.py`` 是被**读取**的，不是被改写的",
    "本包**不做 import、不执行被扫描的表达式**：取值指纹只走 ``literal_eval``，"
    "因此 ``Path(__file__)`` 这类右端被记成 opaque，而不是被求值",
    "本包的全部读数**离线、确定性**：只读文件系统与 ``ast``，不联网、不读任何环境变量密钥",
    "本包**不判断常量取值的好坏**：它只如实记录'谁把哪个名字钉成了什么值'，"
    "不评价'这个 γ 该不该是 0.99'——那是读台账的人要做的事",
)


def docs_of_value_kind(kind: str) -> str:
    """返回一类取值形态的一句话解释（未知类别当场拒绝）."""
    return VALUE_DESCRIPTIONS[require_value_kind(kind)]


def docs_of_relation(relation: str) -> str:
    """返回一类同名关系的一句话解释（未知类别当场拒绝）."""
    return RELATION_DESCRIPTIONS[require_relation(relation)]


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in CONSTANT_LEDGER_PROPERTIES)


__all__ = [
    "CONSTANT_LEDGER_BOUNDARIES",
    "CONSTANT_LEDGER_NOTES",
    "CONSTANT_LEDGER_NOTES_ORDER",
    "CONSTANT_LEDGER_PROPERTIES",
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "INIT_STEM",
    "MIN_CONSTANT_NAME_LENGTH",
    "MODULE_SUFFIX",
    "PACKAGE_NAME",
    "PROPERTY_CONFLICTS_ARE_SOUND",
    "PROPERTY_CONSTANTS_ARE_NUMEROUS",
    "PROPERTY_DUPLICATE_ASSIGNMENTS_WITHIN_MODULE",
    "PROPERTY_LEDGER_COVERS_ALL_MODULES",
    "PROPERTY_LEDGER_IS_REPRODUCIBLE",
    "PROPERTY_SHARED_CONSTANTS_ARE_SOUND",
    "PROPERTY_SHARED_CONSTANTS_EXIST",
    "PROPERTY_SPECS",
    "PropertySpec",
    "RELATIONS",
    "RELATION_DESCRIPTIONS",
    "RELATION_CONFLICT",
    "RELATION_CONSISTENT",
    "RELATION_INCOMPARABLE",
    "RELATION_UNIQUE",
    "VALUE_DESCRIPTIONS",
    "VALUE_KINDS",
    "VALUE_LITERAL",
    "VALUE_OPAQUE",
    "docs_of_relation",
    "docs_of_value_kind",
    "property_specs",
    "require_positive_int",
    "require_property",
    "require_relation",
    "require_value_kind",
]
