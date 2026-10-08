"""``course_index`` 的口径表（day101）.

一次性把这一课的名词表写全：**2 种语料 / 1 套分词口径 / 7 条性质（判据分三类）/
10 条笔记 / 5 条边界**。全部是常量，因此可以被测试逐键检查。

```text
2 种语料     ``docs/*.md``（文档）与每个子包的 docstring（子包）
1 套分词     ASCII 词 + 中文二元组（bigram）——**确定性、零依赖**
7 条性质     判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **把材料编成索引，第一步不是"选一个搜索算法"，而是先说清"哪些材料必须进得来"——
> 一份没进语料的文档，与"这门课没有那份文档"在检索结果里读起来完全一样。**

因此本课的第一、二条性质是**覆盖**（``docs/*.md`` 全覆盖 + 全部子包全覆盖），
然后才是索引本身的三条（可复算 / 确定 / 有界），最后是两条**金标准**性质
（每个文档都进了索引、一个写死的查询必须命中它该命中的那一份）。

## 二、为什么分词要"确定性、零依赖"

```text
中文没有空格   按空白切词 ⇒ 一整段中文变成"一个词"，索引等于没建
引入分词库     多一个依赖，且不同版本会给出不同切法 ⇒ 索引不再是可复算的
中文二元组     把"反向传播"切成 {反向, 向传, 传播}——零依赖、版本无关、逐位可复算
```

代价是"会多召回一些"（bigram 比词粗），但本课要的不是最好的检索质量，
而是**一份能被别人重算出同一份结果的索引**——这与 day087 的"位置必须由缓存长度决定"、
day100 的"剧本必须能被重放"是同一条纪律。

## 三、一条纪律：分数必须有界，且界要写出来

```text
score = 该文档命中的查询词数 / 查询词总数   ⇒  数学上落在 [0, 1]
```

第 ⑤ 条性质就检查这个上界。**为什么值得做成判据**：一个越界的分数
（例如实现里多减了一次）在报告里只是"这一条分数看起来偏大"，
而在 `[0, 1]` 之外的值会让"排序对不对"这件事彻底失去意义。
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# 1. 两种语料
# --------------------------------------------------------------------------------------

DOC_KIND_DOC = "doc"
DOC_KIND_SUBPACKAGE = "subpackage"

#: 两种语料（顺序 = 语料排序里的顺序：先文档、后子包）.
DOC_KINDS: tuple[str, ...] = (DOC_KIND_DOC, DOC_KIND_SUBPACKAGE)

#: 每种语料的一句话解释（"它是从哪里来的"）.
DOC_KIND_DESCRIPTIONS: dict[str, str] = {
    DOC_KIND_DOC: "文档：仓库 ``docs/`` 下的 ``*.md`` 全文（课程手册）",
    DOC_KIND_SUBPACKAGE: "子包：每个子包的 ``__init__.py`` docstring（包自述）",
}

if set(DOC_KINDS) != set(DOC_KIND_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "两种语料的两张表不一致：DOC_KINDS 与 DOC_KIND_DESCRIPTIONS 必须逐键对齐——"
        "少一个键的语料在报告里只有名字、没有它从哪里来。"
    )


def require_doc_kind(kind: str) -> str:
    """校验一种语料种类（未知种类当场拒绝）."""
    if kind not in DOC_KIND_DESCRIPTIONS:
        from smart_research_agent.course_index.errors import ParameterError

        raise ParameterError(
            f"未知的语料种类 {kind!r}：可选 {list(DOC_KINDS)}——"
            "报告按这两种归拢语料；种类外的名字既不会被渲染、也不会被覆盖检查发现。"
        )
    return kind


# --------------------------------------------------------------------------------------
# 2. 分词口径与检索参数
# --------------------------------------------------------------------------------------

#: 中文的 n-gram 长度（2 = 二元组）。取 2 的理由见模块 docstring 第二节。
NGRAM = 2

#: ASCII 词的最小长度（长度为 1 的词直接丢掉：``a`` / ``x`` 这类没有分辨力）。
MIN_WORD_LEN = 2

#: 检索的默认返回条数.
TOP_K_DEFAULT = 5

#: 金标准查询与它必须命中的那一份材料（第 ⑦ 条性质读它）.
#:
#: 选 ``course_index`` 自己的包名，是因为它在自己的 docstring 里必然出现——
#: 金标准必须"由构造保证成立"，否则一条性质会因为语料变化而随机变红。
#: 文档名带种类前缀（见 :mod:`course_index.corpus`），因此金标准材料名也带前缀。
GOLD_QUERY = "course_index"
GOLD_DOC = f"{DOC_KIND_SUBPACKAGE}:{GOLD_QUERY}"

if NGRAM < 1 or MIN_WORD_LEN < 1 or TOP_K_DEFAULT < 1:  # pragma: no cover - 导入期不变式
    raise ValueError("分词与检索参数都必须是 >= 1 的整数。")

if GOLD_DOC != f"{DOC_KIND_SUBPACKAGE}:{GOLD_QUERY.lower()}":  # pragma: no cover - 导入期不变式
    raise ValueError(
        f"金标准查询 {GOLD_QUERY!r} 与它要命中的材料名 {GOLD_DOC!r} 对不上——"
        "本课的金标准是'查询包名、命中该包'，两者必须一致，否则第 ⑦ 条性质永远失败。"
    )


def require_positive_int(label: str, value: int) -> int:
    """校验一个 >= 1 的整数（``top_k`` 之类的参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        from smart_research_agent.course_index.errors import ParameterError

        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的 top_k 意味着检索永远返回空，这条链路等于不存在。"
        )
    return value


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

PROPERTY_CORPUS_COVERS_ALL_DOCS = "corpus_covers_all_docs"
PROPERTY_CORPUS_COVERS_ALL_SUBPACKAGES = "corpus_covers_all_subpackages"
PROPERTY_INDEX_IS_REPRODUCIBLE = "index_is_reproducible"
PROPERTY_SEARCH_IS_DETERMINISTIC = "search_is_deterministic"
PROPERTY_SCORES_WITHIN_BOUNDS = "scores_within_bounds"
PROPERTY_EVERY_DOCUMENT_IS_INDEXED = "every_document_is_indexed"
PROPERTY_GOLD_QUERY_FINDS_GOLD_DOCUMENT = "gold_query_finds_gold_document"


@dataclass(frozen=True)
class PropertySpec:
    """一条性质：id + 说明 + 判据类别 + "失败意味着什么"."""

    id: str
    description: str
    criterion: str
    failure: str

    def __post_init__(self) -> None:
        from smart_research_agent.course_index.errors import ParameterError

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
        """一行说明：``[equality] corpus_covers_all_docs | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 检查的顺序：先覆盖、再索引、最后金标准）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_CORPUS_COVERS_ALL_DOCS: PropertySpec(
        id=PROPERTY_CORPUS_COVERS_ALL_DOCS,
        description="``docs/`` 下的每一份 ``*.md`` 都进了语料（缺失 0）",
        criterion=CRITERION_EQUALITY,
        failure="某一份手册没有进语料——它在检索结果里与'这份手册不存在'一样",
    ),
    PROPERTY_CORPUS_COVERS_ALL_SUBPACKAGES: PropertySpec(
        id=PROPERTY_CORPUS_COVERS_ALL_SUBPACKAGES,
        description="每一个子包都有一条语料条目（缺失 0）",
        criterion=CRITERION_EQUALITY,
        failure="某个子包没有 docstring 条目——它的自述永远不会被检索到",
    ),
    PROPERTY_INDEX_IS_REPRODUCIBLE: PropertySpec(
        id=PROPERTY_INDEX_IS_REPRODUCIBLE,
        description="同一份语料两次构建索引**逐位相同**（没有未固定的迭代序）",
        criterion=CRITERION_EQUALITY,
        failure="索引里混进了未固定的量（集合序 / 字典序），两次构建不同",
    ),
    PROPERTY_SEARCH_IS_DETERMINISTIC: PropertySpec(
        id=PROPERTY_SEARCH_IS_DETERMINISTIC,
        description="同一个查询两次检索给出同一份命中（顺序也相同）",
        criterion=CRITERION_EQUALITY,
        failure="并列命中的排序不稳定——同一个问题两次得到两份名单",
    ),
    PROPERTY_SCORES_WITHIN_BOUNDS: PropertySpec(
        id=PROPERTY_SCORES_WITHIN_BOUNDS,
        description="每一个命中分数都落在 ``[0, 1]``（读数 <= 1）",
        criterion=CRITERION_UPPER_BOUND,
        failure="打分越界——越界的分数会让'排序对不对'这件事失去意义",
    ),
    PROPERTY_EVERY_DOCUMENT_IS_INDEXED: PropertySpec(
        id=PROPERTY_EVERY_DOCUMENT_IS_INDEXED,
        description="每一份语料至少被索引了一个词（最小词数 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="某一份语料一个词都没切出来——它进了语料但永远检索不到",
    ),
    PROPERTY_GOLD_QUERY_FINDS_GOLD_DOCUMENT: PropertySpec(
        id=PROPERTY_GOLD_QUERY_FINDS_GOLD_DOCUMENT,
        description="金标准查询命中它该命中的那一份材料（命中 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="一次确定的检索没找到确定的东西——要么语料漏了，要么索引错了",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐）.
COURSE_INDEX_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.course_index.errors import ParameterError

        raise ParameterError(f"未知的性质 {property_id!r}：可选 {list(COURSE_INDEX_PROPERTIES)}。")
    return property_id


#: 三类判据各自至少要有一条性质（否则报告里那一类判据是空的）.
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

COURSE_INDEX_NOTES: dict[str, str] = {
    "coverage_before_algorithm": (
        "编索引的第一步不是选算法，而是说清'哪些材料必须进得来'："
        "一份没进语料的文档，与'这份文档不存在'在结果里读起来一样。"
    ),
    "tokenizer_must_be_deterministic": (
        "分词必须**确定性、零依赖**：换一个分词库版本就会换一份索引，"
        "而'索引可复算'是这份产物的全部价值。"
    ),
    "index_is_an_intermediate": (
        "索引是中间物，它的错只在检索时才显形——所以它必须自己能被人单独复算。"
    ),
    "score_needs_a_ceiling": (
        "分数必须有界，而且界要写出来：越界的分数会让排序失去意义，"
        "而报告里它只表现为'这一条看起来偏大'。"
    ),
    "empty_query_is_not_a_miss": (
        "'空查询'与'查了没命中'是两件事：前者是调用点的问题，后者是语料的问题。"
    ),
    "bigram_trades_precision_for_reproducibility": (
        "中文二元组用一点点精度换了版本无关的可复算性——"
        "本课要的是'别人重算出同一份索引'，不是最好的检索质量。"
    ),
    "subpackage_docstring_is_its_self_description": (
        "每个子包的 docstring 就是它的自述：把它编进语料，"
        "'这个包是干什么的'才成为一个可以被检索的问题。"
    ),
    "ties_must_be_ordered_by_name": (
        "并列的命中必须用**名字**兜一个确定的顺序："
        "否则同一个问题两次得到两份名单，而两份都'对'。"
    ),
    "gold_query_must_be_true_by_construction": (
        "金标准要由构造保证成立（查询包名、命中该包），"
        "否则一条性质会因为语料变化而随机变红，最后被人关掉。"
    ),
    "reproducible_index_is_the_deliverable": (
        "本课真正的交付物不是'一个搜索'，而是一份**可被重算出同一结果**的索引。"
    ),
}

#: 十条笔记的键（顺序即写入顺序）.
COURSE_INDEX_NOTES_ORDER: tuple[str, ...] = tuple(COURSE_INDEX_NOTES)

# --------------------------------------------------------------------------------------
# 5. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

COURSE_INDEX_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：分词是自写的 bigram + ASCII 词切分，"
    "不引入任何分词库",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件；``docs/`` 下的手册是被**读取**的，不是被改写的",
    "本包**不承诺检索质量**：它承诺的是'同一份语料编出同一份索引、"
    "同一查询给出同一份命中'，而不是'排在最前面的总是最好的'",
    "本包的全部读数**离线、确定性**：只读文件系统与 ``importlib``，"
    "不联网、不读任何环境变量密钥",
    "本包**不含语义检索**：bigram 索引认得的是'字面共现'，"
    "不是'同义'——把'怎么训练'与'训练方法'联系起来不是它要解决的问题",
)


def docs_of_kind(kind: str) -> str:
    """返回一种语料的一句话解释（未知种类当场拒绝）."""
    return DOC_KIND_DESCRIPTIONS[require_doc_kind(kind)]


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in COURSE_INDEX_PROPERTIES)


__all__ = [
    "COURSE_INDEX_BOUNDARIES",
    "COURSE_INDEX_NOTES",
    "COURSE_INDEX_NOTES_ORDER",
    "COURSE_INDEX_PROPERTIES",
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "DOC_KINDS",
    "DOC_KIND_DESCRIPTIONS",
    "DOC_KIND_DOC",
    "DOC_KIND_SUBPACKAGE",
    "GOLD_DOC",
    "GOLD_QUERY",
    "MIN_WORD_LEN",
    "NGRAM",
    "PROPERTY_CORPUS_COVERS_ALL_DOCS",
    "PROPERTY_CORPUS_COVERS_ALL_SUBPACKAGES",
    "PROPERTY_EVERY_DOCUMENT_IS_INDEXED",
    "PROPERTY_GOLD_QUERY_FINDS_GOLD_DOCUMENT",
    "PROPERTY_INDEX_IS_REPRODUCIBLE",
    "PROPERTY_SCORES_WITHIN_BOUNDS",
    "PROPERTY_SEARCH_IS_DETERMINISTIC",
    "PROPERTY_SPECS",
    "TOP_K_DEFAULT",
    "PropertySpec",
    "docs_of_kind",
    "property_specs",
    "require_doc_kind",
    "require_positive_int",
    "require_property",
]
