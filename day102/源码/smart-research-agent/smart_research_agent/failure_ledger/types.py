"""``failure_ledger`` 的口径表（day102）.

一次性把这一课的名词表写全：**1 种扫描目标 / 4 类基类归属 / 7 条性质（判据分三类）/
10 条笔记 / 5 条边界**。全部是常量，因此可以被测试逐键检查。

```text
1 种扫描目标   ``smart_research_agent/*/errors.py``（36 份：35 份历史 + 本课这一份）
4 类基类归属   local（本模块） / builtin（内置异常） / imported（导入别名） / unresolved
7 条性质       判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **一份没有被扫到的 ``errors.py``，与"这个子包没有失败族"在台账里读起来完全一样。**
> 因此本课的第一条性质是**覆盖**（36 份都要进来），然后才是台账本身的
> 三条（可复算 / 名字唯一 / 每份恰好一个根），最后是三条**结构性**性质
> （未解析基类数为 0、每个族至少一个基类、最大继承深度 >= 1）。

## 二、为什么"归属"只有四类

```text
local       基类名就在同一个文件里（``class ShapeError(CoreShapeError, StackError)`` 的 StackError）
builtin     Python 内置异常名（``ValueError`` / ``Exception`` / ``RuntimeError`` ……）
imported    基类名来自另一个 ``errors.py``，以别名出现（``ShapeError as CoreShapeError``）
unresolved  以上三类都不是 ⇒ 台账作者也不知道它是什么，必须当场点名
```

把 ``unresolved`` 单独列出来是本课的核心纪律：**一个"看起来像基类名"的字符串，
如果没有人能说清它属于哪一类，它就不该被静默地当成类名。**
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# 1. 扫描目标
# --------------------------------------------------------------------------------------

#: 被扫描的包名.
PACKAGE_NAME = "smart_research_agent"

#: 被扫描的文件名（每个子包一份）.
SCAN_FILENAME = "errors.py"

# --------------------------------------------------------------------------------------
# 2. 四类基类归属
# --------------------------------------------------------------------------------------

BASE_KIND_LOCAL = "local"
BASE_KIND_BUILTIN = "builtin"
BASE_KIND_IMPORTED = "imported"
BASE_KIND_UNRESOLVED = "unresolved"

#: 四类归属（顺序 = 解析时的判定顺序）.
BASE_KINDS: tuple[str, ...] = (
    BASE_KIND_LOCAL,
    BASE_KIND_BUILTIN,
    BASE_KIND_IMPORTED,
    BASE_KIND_UNRESOLVED,
)

#: 每类归属的一句话解释（"它从哪里来"）.
BASE_KIND_DESCRIPTIONS: dict[str, str] = {
    BASE_KIND_LOCAL: "local：基类名就在同一个 errors.py 里定义（本模块内部的继承）",
    BASE_KIND_BUILTIN: "builtin：Python 内置异常名（ValueError / Exception / RuntimeError ……）",
    BASE_KIND_IMPORTED: "imported：基类名来自另一个 errors.py，以 import 别名出现",
    BASE_KIND_UNRESOLVED: "unresolved：以上三类都不是——台账作者也不知道它是什么",
}

#: 被认作"内置异常"的名字（根族应当且只应当继承这些）.
#:
#: 这是一份**白名单**而不是"任何 import 得到的东西"：只有在这张表里的名字，
#: 才允许作为"根"的基类出现——否则一个手滑的别名就会被当成内置异常。
BUILTIN_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "BaseException",
        "Exception",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BufferError",
        "EOFError",
        "FileNotFoundError",
        "ImportError",
        "IndexError",
        "KeyError",
        "LookupError",
        "MemoryError",
        "NameError",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "PermissionError",
        "ReferenceError",
        "RuntimeError",
        "StopIteration",
        "SyntaxError",
        "SystemError",
        "TimeoutError",
        "TypeError",
        "UnicodeError",
        "ValueError",
        "Warning",
        "ZeroDivisionError",
    }
)

# --------------------------------------------------------------------------------------
# 3. 族的两种身份与七条性质（判据分三类）
# --------------------------------------------------------------------------------------

FAMILY_KIND_ROOT = "root"
FAMILY_KIND_SUB = "sub"

#: 一个族的两种身份（没有 local 基类的是 root，否则是 sub）.
FAMILY_KINDS: tuple[str, ...] = (FAMILY_KIND_ROOT, FAMILY_KIND_SUB)

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

PROPERTY_LEDGER_COVERS_ALL_ERROR_MODULES = "ledger_covers_all_error_modules"
PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE = "qualified_names_are_unique"
PROPERTY_LEDGER_IS_REPRODUCIBLE = "ledger_is_reproducible"
PROPERTY_EVERY_MODULE_HAS_EXACTLY_ONE_ROOT = "every_module_has_exactly_one_root"
PROPERTY_BASE_REFERENCES_RESOLVE = "base_references_resolve"
PROPERTY_EVERY_FAMILY_HAS_A_PARENT = "every_family_has_a_parent"
PROPERTY_INHERITANCE_DEPTH_IS_POSITIVE = "inheritance_depth_is_positive"


@dataclass(frozen=True)
class PropertySpec:
    """一条性质：id + 说明 + 判据类别 + "失败意味着什么"."""

    id: str
    description: str
    criterion: str
    failure: str

    def __post_init__(self) -> None:
        from smart_research_agent.failure_ledger.errors import ParameterError

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
        """一行说明：``[equality] ledger_covers_all_error_modules | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 检查的顺序：先覆盖、再台账、最后结构）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_LEDGER_COVERS_ALL_ERROR_MODULES: PropertySpec(
        id=PROPERTY_LEDGER_COVERS_ALL_ERROR_MODULES,
        description="每一份 ``errors.py`` 都进了台账（缺失 0）",
        criterion=CRITERION_EQUALITY,
        failure="某一份 errors.py 没有进台账——它在报告里与'这个子包没有失败族'一样",
    ),
    PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE: PropertySpec(
        id=PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE,
        description="``包.名`` 在台账里唯一（冲突 0；裸名字允许跨包重名）",
        criterion=CRITERION_EQUALITY,
        failure="两个族共用一个 ``包.名``——它们在台账里就没法被区分",
    ),
    PROPERTY_LEDGER_IS_REPRODUCIBLE: PropertySpec(
        id=PROPERTY_LEDGER_IS_REPRODUCIBLE,
        description="同一批文件两次构建台账**逐位相同**（没有未固定的迭代序）",
        criterion=CRITERION_EQUALITY,
        failure="台账里混进了未固定的量（集合序 / 字典序），两次构建不同",
    ),
    PROPERTY_EVERY_MODULE_HAS_EXACTLY_ONE_ROOT: PropertySpec(
        id=PROPERTY_EVERY_MODULE_HAS_EXACTLY_ONE_ROOT,
        description="每一份 ``errors.py`` 恰好有一个根族（既不是 0 个、也不是 2 个）",
        criterion=CRITERION_EQUALITY,
        failure="某个模块有 0 个或 2 个根族——'这一族的基类'这句话在它身上说不清",
    ),
    PROPERTY_BASE_REFERENCES_RESOLVE: PropertySpec(
        id=PROPERTY_BASE_REFERENCES_RESOLVE,
        description="每一个基类名都能归属到 local / builtin / imported（未解析数 <= 0）",
        criterion=CRITERION_UPPER_BOUND,
        failure="有基类名无法归属——台账作者也说不清它属于哪一类",
    ),
    PROPERTY_EVERY_FAMILY_HAS_A_PARENT: PropertySpec(
        id=PROPERTY_EVERY_FAMILY_HAS_A_PARENT,
        description="每一个族至少有 1 个基类（最小基类数 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="某个族一个基类都没有——它不会成为任何一条 ``except`` 的目标",
    ),
    PROPERTY_INHERITANCE_DEPTH_IS_POSITIVE: PropertySpec(
        id=PROPERTY_INHERITANCE_DEPTH_IS_POSITIVE,
        description="本模块内部真的存在一条继承链（最大深度 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="所有族都直接继承内置异常——本课的'继承森林'退化成了散点",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐）.
FAILURE_LEDGER_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.failure_ledger.errors import ParameterError

        raise ParameterError(f"未知的性质 {property_id!r}：可选 {list(FAILURE_LEDGER_PROPERTIES)}。")
    return property_id


def require_base_kind(kind: str) -> str:
    """校验一类基类归属（未知类别当场拒绝）."""
    if kind not in BASE_KIND_DESCRIPTIONS:
        from smart_research_agent.failure_ledger.errors import ParameterError

        raise ParameterError(
            f"未知的基类归属 {kind!r}：可选 {list(BASE_KINDS)}——"
            "报告按这四类归拢基类；类别外的名字既不会被渲染、也不会被覆盖检查发现。"
        )
    return kind


def require_positive_int(label: str, value: int) -> int:
    """校验一个 >= 1 的整数（``limit`` 之类的参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        from smart_research_agent.failure_ledger.errors import ParameterError

        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的 limit 意味着报告永远空着，这条链路等于不存在。"
        )
    return value


# 导入期不变式：四类归属、两种身份、三类判据都必须闭合。
if set(BASE_KINDS) != set(BASE_KIND_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "四类归属的两张表不一致：BASE_KINDS 与 BASE_KIND_DESCRIPTIONS 必须逐键对齐——"
        "少一个键的类别在报告里只有名字、没有它从哪里来。"
    )

if not BUILTIN_EXCEPTIONS:  # pragma: no cover - 导入期不变式
    raise ValueError("内置异常白名单不能为空：否则任何名字都会变成 unresolved。")

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

FAILURE_LEDGER_NOTES: dict[str, str] = {
    "coverage_before_algorithm": (
        "编台账的第一步不是说清'怎么解析基类'，而是说清'哪些文件必须进得来'："
        "一份没进台账的 errors.py，与'这个子包没有失败族'在结果里读起来一样。"
    ),
    "unresolved_must_be_named": (
        "归属只有四类，其中 ``unresolved`` 是**要被点名的**那一类："
        "一个没人说得清归属的名字，比一个报错更难被发现。"
    ),
    "qualified_name_vs_bare_name": (
        "族的裸名字允许跨包重名（十来个包都有 ``ShapeError``），"
        "但 ``包.名`` 必须唯一——这正是 day101 给文档名加种类前缀的同一条纪律。"
    ),
    "ledger_is_an_intermediate": (
        "台账是中间物，它的错（同名冲突 / 两次不同）在被印出来之前是看不见的——"
        "所以它必须自己能被人单独复算。"
    ),
    "root_must_inherit_a_builtin": (
        "每个模块的根族都必须继承一个**内置异常**："
        "否则 ``except ValueError`` 这种朴素写法兜不住它，而这件事在报告里看不出来。"
    ),
    "cross_package_base_is_a_contract": (
        "``ShapeError as CoreShapeError`` 这种跨包继承是一份**契约**："
        "day075 那一层写 ``except ShapeError`` 的代码，不需要改动就能兜住后来各层的失败。"
    ),
    "depth_is_not_quality": (
        "继承深度不是越深越好：本课量它只是为了证明'链真的存在'，"
        "而不是'链越长设计越好'。"
    ),
    "exactly_one_root_per_module": (
        "一个模块恰好一个根族，是「每一族都能被一条 except 兜住」的前提；"
        "两个根意味着读者要先猜「我该 except 哪一个」。"
    ),
    "same_name_is_meant_to_be_shared": (
        "``ShapeError`` / ``ParameterError`` / ``NumericError`` 天天重名——"
        "这不是命名偷懒，而是**同一类失败在每一层都要能被同一句话命名**。"
    ),
    "reproducible_ledger_is_the_deliverable": (
        "本课真正的交付物不是'一张好看的继承图'，而是一份**可被重算出同一结果**的台账。"
    ),
}

#: 十条笔记的键（顺序即写入顺序）.
FAILURE_LEDGER_NOTES_ORDER: tuple[str, ...] = tuple(FAILURE_LEDGER_NOTES)

# --------------------------------------------------------------------------------------
# 5. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

FAILURE_LEDGER_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：扫描只用标准库 ``ast``，不 import 被扫描的模块",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件；被扫描的 ``errors.py`` 是被**读取**的，不是被改写的",
    "本包**不做导入**：它读源码文本并解析 AST，因此一个语法错误会被当场点名，"
    "而不是被「import 失败」掩盖",
    "本包的全部读数**离线、确定性**：只读文件系统与 ``ast``，不联网、不读任何环境变量密钥",
    "本包**不判断设计好坏**：它只如实记录「谁继承了谁」，"
    "不评价「这个继承关系该不该存在」——那是读台账的人要做的事",
)


def docs_of_kind(kind: str) -> str:
    """返回一类归属的一句话解释（未知类别当场拒绝）."""
    return BASE_KIND_DESCRIPTIONS[require_base_kind(kind)]


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in FAILURE_LEDGER_PROPERTIES)


__all__ = [
    "BASE_KINDS",
    "BASE_KIND_BUILTIN",
    "BASE_KIND_DESCRIPTIONS",
    "BASE_KIND_IMPORTED",
    "BASE_KIND_LOCAL",
    "BASE_KIND_UNRESOLVED",
    "BUILTIN_EXCEPTIONS",
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "FAILURE_LEDGER_BOUNDARIES",
    "FAILURE_LEDGER_NOTES",
    "FAILURE_LEDGER_NOTES_ORDER",
    "FAILURE_LEDGER_PROPERTIES",
    "FAMILY_KINDS",
    "FAMILY_KIND_ROOT",
    "FAMILY_KIND_SUB",
    "PACKAGE_NAME",
    "PROPERTY_BASE_REFERENCES_RESOLVE",
    "PROPERTY_EVERY_FAMILY_HAS_A_PARENT",
    "PROPERTY_EVERY_MODULE_HAS_EXACTLY_ONE_ROOT",
    "PROPERTY_INHERITANCE_DEPTH_IS_POSITIVE",
    "PROPERTY_LEDGER_COVERS_ALL_ERROR_MODULES",
    "PROPERTY_LEDGER_IS_REPRODUCIBLE",
    "PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE",
    "PROPERTY_SPECS",
    "PropertySpec",
    "SCAN_FILENAME",
    "docs_of_kind",
    "property_specs",
    "require_base_kind",
    "require_positive_int",
    "require_property",
]
