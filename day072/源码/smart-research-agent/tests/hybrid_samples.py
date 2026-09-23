"""day067 混合检索测试用的确定性语料（两路互补 + 术语精确匹配）.

单独一个模块而不是把样本复制进六个测试文件，理由与 ``retrieval_samples`` 相同：
**同一份语料要被 lexical / fusion / hybrid / filters / api 六个文件同时用到**，
复制多份会让"样本里加一条记录"变成要改许多处，而漏改的那一处只表现为
"某个测试不再覆盖那条语义"。

## 这批样本要造出的两件事（缺一个这门课就讲不清）

```text
1. 语义相近但词面完全不同   向量路能召回、关键词路一条都召不回
                           → "午饭吃点什么好"（语料里一个相关字都没有）
                           → 以及与 h-t-02 同向的那句话（"重试预算"那一类问法）
2. 术语/编号类词面精确匹配  关键词路一条命中、向量路完全找不到方向
                           → "ERR-2043" / "401" / "retry_budget"
```

第 2 条是本课的**主案例**。它靠两处人为的设计实现，两处都写在下面：

```text
向量的方向与"这条记录讲什么"**刻意错开**：h-t-01（讲 ERR-2043）的向量被放在
第 8 根轴上，而查询 "ERR-2043" 的向量指向第 5 根轴（h-c-01 的方向）。
于是"语义上最像"的那条（在语料里恰好是"把一句话映射到低维空间的向量"）
压过了真正包含这个编号的那条 —— 而这正是向量路在编号、函数名、报错码
面前"差不多就行"的真实表现。
h-t-01 的词面是唯一的：只有它同时含有 err 与 2043 两个词元。
```

## 期望值为什么能手算

向量维度取 8、分量只用 0 / 0.6 / 0.8 / 1.0（都已归一），因此余弦分数
是 1.0 或 0.0——**"0 分并列"这件事因此是一个精确的事实**：向量路在
"没有一条真的像"时会退化成"按 id 排序"，而本课正好要讲"这时关键词路在做什么"。

BM25 的分数由 ``tests/test_hybrid_bm25.py`` 里**独立实现的公式**交叉核对
（那份实现只用 ``math.log`` 与手写的 tf/df，不引用 ``lexical`` 的任何私函数），
因此下面那几个 ``LEXICAL_SCORES`` 不是"跑一遍看看"，而是两处独立算出的同一个数。

RRF 的分数是纯算术：``1/(k + rank + 1)``，k=60 时第 1 名是 ``1/61``——
它不需要任何交叉核对，因为**它就是那个分数的定义**。

全部离线、确定性、零网络、不写仓库外文件。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.retrieval.hybrid import HybridRetriever
from smart_research_agent.retrieval.lexical import (
    DEFAULT_B,
    DEFAULT_K1,
    BM25Params,
    LexicalDocument,
    LexicalIndex,
)
from smart_research_agent.retrieval.retriever import Retriever, build_retriever
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import VectorRecord, make_record
from tests.retrieval_samples import TableEmbedding

#: 样本向量维度。取 8（与 ``retrieval_samples`` 同一个量级）：余弦分数只有
#: 1.0 与 0.0 两种取值，手算与断言都省事，而"0 分并列"正是本课要用的素材。
VECTOR_DIMENSION = 8

#: 三份文档（day062 的 ``parent_doc_id``；多样性裁剪按它分组）.
DOC_MANUAL = "doc-manual"
DOC_TROUBLE = "doc-trouble"
DOC_CONCEPT = "doc-concept"

#: 十条记录：``(id, 向量, 正文, 文档, strategy, index, topic, created_at)``.
#:
#: 摊平成元组而不是字典，是为了让"向量与正文的对应关系"在一屏里看得完——
#: 这个文件里最要紧的一件事就是那两列对不对得上（见模块 docstring）。
#:
#: ``created_at`` 为 ``None`` 的那条（``h-t-03``）是刻意的：它让"时间过滤把
#: 缺字段的记录排除掉"这条 day066 的约定在**两路**上分别可核对。
RECORD_SPECS: tuple[tuple[Any, ...], ...] = (
    (
        "h-m-01",
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "语义缓存用分位数标定阈值，换编码器必须重新标定。",
        DOC_MANUAL,
        "structural",
        0,
        "cache",
        "2026-09-01",
    ),
    (
        "h-m-02",
        (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "召回深度按三倍过取，阈值在检索层落刀。",
        DOC_MANUAL,
        "structural",
        1,
        "retrieval",
        "2026-09-05",
    ),
    (
        "h-m-03",
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "一次全量重建索引大约 12 万条记录，按每条 320 token 计。",
        DOC_MANUAL,
        "fixed",
        2,
        "cost",
        "2026-09-10",
    ),
    (
        # 主案例：讲 ERR-2043，但向量被刻意放在第 8 根轴上（见模块 docstring）。
        "h-t-01",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "ERR-2043 表示向量维度不一致，先重建索引再重试。",
        DOC_TROUBLE,
        "semantic",
        0,
        "errors",
        "2026-09-12",
    ),
    (
        "h-t-02",
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "重试预算 retry_budget 缺省 3 次，超时按指数退避。",
        DOC_TROUBLE,
        "semantic",
        1,
        "errors",
        "2026-09-15",
    ),
    (
        # **唯一一条没有 created_at 的记录**：任何时间范围都会把它排除。
        "h-t-03",
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "报错码 401 与 403 的处置方式完全不同。",
        DOC_TROUBLE,
        "fixed",
        2,
        "errors",
        None,
    ),
    (
        "h-c-01",
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        "把一句话映射到低维空间的向量，越近表示意思越像。",
        DOC_CONCEPT,
        "structural",
        0,
        "embedding",
        "2026-09-18",
    ),
    (
        "h-c-02",
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "同义词与改写会让词面匹配失效，这时要靠语义相似度。",
        DOC_CONCEPT,
        "structural",
        1,
        "embedding",
        "2026-09-20",
    ),
    (
        "h-c-03",
        (0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0),
        "术语表给出缩写与英文全称的对照。",
        DOC_CONCEPT,
        "semantic",
        2,
        "glossary",
        "2026-09-22",
    ),
    (
        "h-c-04",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "关键词检索擅长精确匹配：编号、函数名、报错码。",
        DOC_CONCEPT,
        "semantic",
        3,
        "keyword",
        "2026-09-25",
    ),
)

#: 十条记录的 id（按插入顺序）。注意它与**升序不同**：
#: 插入序里 ``h-t-*`` 排在 ``h-c-*`` 之前，而字典序上 ``h-c-…`` 更小——
#: "同分按 id 升序"这条规则的用例正是靠这个差别才有区分度。
RECORD_IDS: tuple[str, ...] = tuple(str(spec[0]) for spec in RECORD_SPECS)

#: id → 正文.
RECORD_TEXTS: dict[str, str] = {str(spec[0]): str(spec[2]) for spec in RECORD_SPECS}

#: 唯一缺少 ``created_at`` 的记录：任何时间范围都会把它排除（两路都要如此）.
MISSING_CREATED_AT_ID = "h-t-03"

#: 主案例的两条记录（一个讲编号、一个讲"向量是什么"）.
EXACT_MATCH_ID = "h-t-01"
VECTOR_DECOY_ID = "h-c-01"


# --------------------------------------------------------------------------- #
# 四个查询：两句话各砸一路，一句话两路都砸中
# --------------------------------------------------------------------------- #

#: 词面精确：查询里是一个编号，只有 ``h-t-01`` 含有它的两个词元（err / 2043）.
QUERY_EXACT = "ERR-2043"

#: 词面精确：三位报错码，只有 ``h-t-03`` 含有它.
QUERY_CODE = "401"

#: 语义改写：这句话的每一个字都不在语料里出现（连单字都不出现），
#: 因此关键词路**一条都召不回**，而向量路指向 ``h-t-02``（"重试预算…"）。
#:
#: "每一个字都不出现"是刻意构造的：中文按单字切之后，只要有一个字撞上
#: （比如"试"），关键词路就会返回一条 0 分以上的记录——那时这个用例
#: 就不再是"关键词路的边界"，而是"关键词路命中了几个字"。
QUERY_SEMANTIC = "午饭吃点什么好"

#: 两路都砸中：拉丁词元（词表里唯一出现）＋ 向量方向恰好也指向同一条.
QUERY_BOTH = "retry_budget"

#: 查询文本 → 查询向量（``TableEmbedding`` 的查表键）。
#:
#: 四个方向各自指向一条记录，而指向谁**不等于**哪条在语义上最相关——
#: 这正是模块 docstring 里那处人为设计。（``QUERY_EXACT`` 指向 h-c-01，
#: 而真正含 ERR-2043 的是 h-t-01。）
QUERY_VECTORS: dict[str, tuple[float, ...]] = {
    QUERY_EXACT: (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    QUERY_CODE: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    QUERY_SEMANTIC: (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    QUERY_BOTH: (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

#: ``TableEmbedding`` 的兜底向量（查询文本不在表里时用它）：
#: 第 1 根轴 = ``h-m-01`` 的方向，与 ``retrieval_samples`` 的做法一致。
DEFAULT_QUERY_VECTOR: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# 关键词路的期望值（与 test_hybrid_bm25.py 的独立实现交叉核对）
# --------------------------------------------------------------------------- #

#: 查询 → 关键词路的期望命中（**只有真的命中至少一个词元的那几条**）.
#:
#: ``QUERY_SEMANTIC`` 是空元组：它的 13 个词元全不在词表里（见下一条常量）。
EXPECTED_LEXICAL_IDS: dict[str, tuple[str, ...]] = {
    QUERY_EXACT: ("h-t-01",),
    QUERY_CODE: ("h-t-03",),
    QUERY_SEMANTIC: (),
    QUERY_BOTH: ("h-t-02",),
    "预算": ("h-t-02",),
}

#: 查询 → 关键词路的期望分数（保留 6 位）。
#:
#: 这几个数不是"跑一遍抄下来的"：``test_hybrid_bm25.py::brute_force_bm25``
#: 用一份**独立实现**（只用 ``math.log`` 与手写的 tf/df，不引用 ``lexical``
#: 的任何函数）算出同一批数，两边必须逐位相同。测试用 ``rel=1e-9`` 比较，
#: 因此这里写 6 位足够。
EXPECTED_LEXICAL_SCORES: dict[str, float] = {
    QUERY_EXACT: 4.000606,
    QUERY_CODE: 2.269313,
    QUERY_BOTH: 2.235902,
    "预算": 6.707707,
}

#: ``QUERY_SEMANTIC`` 的词元数（13 个：7 个单字 + 6 个 2-gram）。
#: 它是"关键词路一条都没召回"这个结论的**分母**——报告里必须能说出
#: "13 个词元里 0 个在词表里"，而不是只说"没找到"。
QUERY_SEMANTIC_TERM_COUNT = 13

#: 词表大小与平均文档长度（10 条样本上的确定性数字）。
EXPECTED_VOCABULARY_SIZE = 274
EXPECTED_AVGDL = 34.3
EXPECTED_TOTAL_TERMS = 343


# --------------------------------------------------------------------------- #
# 两路各自的期望名次（手算依据见模块 docstring）
# --------------------------------------------------------------------------- #

#: 向量路在 ``top_k=3, fetch_k=3`` 下的期望命中。
#:
#: 每个查询的向量与一条记录完全同向（余弦 1.0），其余记录全是 0.0——
#: 于是"第 2、3 名"由 id 升序决定（h-c-01、h-c-02、h-c-03 是字典序最小的三条）。
VECTOR_ONLY_SHALLOW: dict[str, tuple[str, ...]] = {
    QUERY_EXACT: ("h-c-01", "h-c-02", "h-c-03"),
    QUERY_CODE: ("h-c-02", "h-c-01", "h-c-03"),
    QUERY_SEMANTIC: ("h-t-02", "h-c-01", "h-c-02"),
    QUERY_BOTH: ("h-t-02", "h-c-01", "h-c-02"),
}

#: 默认深度（``top_k=5`` → ``fetch_k=15``，库只有 10 条）下混合检索的期望名次。
#:
#: 两个"词面精确"的查询里，**含编号的那条排第 1**，而它在向量路里要么
#: 排在很后面（0 分并列里按 id 靠后）、要么根本不在向量的 top-3 里——
#: 这就是本课要展示的互补。
HYBRID_DEFAULT: dict[str, tuple[str, ...]] = {
    QUERY_EXACT: ("h-t-01", "h-c-01", "h-c-02", "h-c-03", "h-c-04"),
    QUERY_CODE: ("h-t-03", "h-c-02", "h-c-01", "h-c-03", "h-c-04"),
    QUERY_SEMANTIC: ("h-t-02", "h-c-01", "h-c-02", "h-c-03", "h-c-04"),
    QUERY_BOTH: ("h-t-02", "h-c-01", "h-c-02", "h-c-03", "h-c-04"),
}

#: ``top_k=3, fetch_k=3`` 下混合检索的期望名次（**第三排序键的用例**）.
#:
#: ``QUERY_EXACT`` 的这一行是最值得看的一条：``h-c-01`` 与 ``h-t-01``
#: 的融合分数**精确相等**（都是 1/61，各自只被一路的 rank 0 命中），
#: 于是次序由第三键 ``record_id`` 决定——``h-c-01`` < ``h-t-01``，
#: 向量路的那条排在前面。这不是"向量路赢了"，而是"分数并列时要有确定性的第三键"。
HYBRID_SHALLOW: dict[str, tuple[str, ...]] = {
    QUERY_EXACT: ("h-c-01", "h-t-01", "h-c-02"),
    QUERY_CODE: ("h-c-02", "h-t-03", "h-c-01"),
    QUERY_SEMANTIC: ("h-t-02", "h-c-01", "h-c-02"),
    QUERY_BOTH: ("h-t-02", "h-c-01", "h-c-02"),
}

#: RRF 的两个常数（与 ``fusion.DEFAULT_*`` 必须一致，这里是给断言用的字面量）.
RRF_K = 60

#: RRF 下"某一路的第 1 名"的贡献（k=60 → 1/61）。测试里到处要用它，
#: 写成一个不依赖实现的字面量：它**就是** RRF 的定义，不是实现的产物。
RRF_FIRST = 1.0 / 61.0


# --------------------------------------------------------------------------- #
# 构造器
# --------------------------------------------------------------------------- #


def query_vectors() -> dict[str, tuple[float, ...]]:
    """查询文本 → 查询向量（``TableEmbedding`` 的查表参数）."""
    return dict(QUERY_VECTORS)


def sample_metadata() -> dict[str, dict[str, Any]]:
    """id → 元数据（**逐层拷贝**：数组值与嵌套字典都复制，避免用例之间互相污染）."""
    return {str(spec[0]): _metadata_of(spec) for spec in RECORD_SPECS}


def metadata_field_names() -> list[str]:
    """样本库里出现过的**全部**元数据字段名（升序）——字段拼写检查的已知全集."""
    names: set[str] = set()
    for spec in RECORD_SPECS:
        names.update(_metadata_of(spec))
    return sorted(names)


def doc_ids_by_parent() -> dict[str, list[str]]:
    """``parent_doc_id → 该文档下的记录 id``（升序）——多样性用例的素材."""
    grouped: dict[str, list[str]] = {}
    for spec in RECORD_SPECS:
        grouped.setdefault(str(spec[3]), []).append(str(spec[0]))
    return {name: sorted(ids) for name, ids in grouped.items()}


def documents() -> list[LexicalDocument]:
    """十条 ``LexicalDocument``（关键词索引的输入）."""
    return [
        LexicalDocument(
            record_id=str(spec[0]),
            text=str(spec[2]),
            metadata=_metadata_of(spec),
        )
        for spec in RECORD_SPECS
    ]


def sample_records(*, metric: str = "cosine", reverse: bool = False) -> list[VectorRecord]:
    """十条 ``VectorRecord``（``reverse=True`` 时插入顺序相反，用于验证可复现性）."""
    specs = list(RECORD_SPECS)
    if reverse:
        specs.reverse()
    return [
        make_record(
            record_id=str(spec[0]),
            vector=tuple(float(value) for value in spec[1]),
            text=str(spec[2]),
            metadata=_metadata_of(spec),
            metric=metric,
        )
        for spec in specs
    ]


def hybrid_store(*, metric: str = "cosine", reverse: bool = False) -> FlatVectorStore:
    """装好十条样本记录的 flat 库（混合检索用例的默认起点）."""
    store = FlatVectorStore(metric=metric, dimension=VECTOR_DIMENSION)
    store.upsert(sample_records(metric=metric, reverse=reverse))
    return store


def empty_store(*, metric: str = "cosine") -> FlatVectorStore:
    """一个**空**库（维度固定为 8，与 ``hybrid_store`` 同规格）.

    空库是合法状态：混合检索为它返回 ``empty_reason="no_data"`` 而不报错
    （与 day066 同一口径），而且 ``fetch_k`` 照样是算过的。
    """
    return FlatVectorStore(metric=metric, dimension=VECTOR_DIMENSION)


def hybrid_lexical(
    store: FlatVectorStore | None = None,
    *,
    params: BM25Params | None = None,
    name: str = "lexical",
) -> LexicalIndex:
    """从样本库现建一份关键词索引（``store`` 缺省用 ``hybrid_store()``）."""
    source = store if store is not None else hybrid_store()
    return LexicalIndex.from_backend(source, params=params, name=name)


def hybrid_embedding() -> TableEmbedding:
    """按 ``QUERY_VECTORS`` 查表的确定性编码器（8 维，与样本库同规格）."""
    return TableEmbedding(
        table=query_vectors(),
        dimension=VECTOR_DIMENSION,
        default=DEFAULT_QUERY_VECTOR,
    )


def hybrid_vector_retriever(
    store: FlatVectorStore | None = None,
    *,
    embedding: TableEmbedding | None = None,
    **overrides: Any,
) -> Retriever:
    """只有向量一路的检索器（``build_retriever`` 装配，``**overrides`` 逐个覆盖）."""
    return build_retriever(
        store if store is not None else hybrid_store(),
        embedding if embedding is not None else hybrid_embedding(),
        **overrides,
    )


def hybrid_retriever(
    store: FlatVectorStore | None = None,
    *,
    embedding: TableEmbedding | None = None,
    lexical: LexicalIndex | None = None,
    **overrides: Any,
) -> HybridRetriever:
    """两路齐全的混合检索器（这一层用例的默认起点）."""
    resolved_store = store if store is not None else hybrid_store()
    return HybridRetriever(
        hybrid_vector_retriever(resolved_store, embedding=embedding),
        lexical if lexical is not None else hybrid_lexical(resolved_store),
        **overrides,
    )


def _metadata_of(spec: tuple[Any, ...]) -> dict[str, Any]:
    """把一条记录规格折成元数据（``created_at`` 为 ``None`` 时不写这个键）."""
    metadata: dict[str, Any] = {
        "parent_doc_id": str(spec[3]),
        "source": f"docs/{spec[3]}.md",
        "strategy": str(spec[4]),
        "index": int(spec[5]),
        "topic": str(spec[6]),
    }
    if spec[7] is not None:
        metadata["created_at"] = str(spec[7])
    return metadata


__all__ = [
    "DEFAULT_B",
    "DEFAULT_K1",
    "DEFAULT_QUERY_VECTOR",
    "DOC_CONCEPT",
    "DOC_MANUAL",
    "DOC_TROUBLE",
    "EXPECTED_AVGDL",
    "EXPECTED_LEXICAL_IDS",
    "EXPECTED_LEXICAL_SCORES",
    "EXPECTED_TOTAL_TERMS",
    "EXPECTED_VOCABULARY_SIZE",
    "EXACT_MATCH_ID",
    "HYBRID_DEFAULT",
    "HYBRID_SHALLOW",
    "MISSING_CREATED_AT_ID",
    "QUERY_BOTH",
    "QUERY_CODE",
    "QUERY_EXACT",
    "QUERY_SEMANTIC",
    "QUERY_SEMANTIC_TERM_COUNT",
    "QUERY_VECTORS",
    "RECORD_IDS",
    "RECORD_SPECS",
    "RECORD_TEXTS",
    "RRF_FIRST",
    "RRF_K",
    "VECTOR_DECOY_ID",
    "VECTOR_DIMENSION",
    "VECTOR_ONLY_SHALLOW",
    "BM25Params",
    "LexicalDocument",
    "LexicalIndex",
    "TableEmbedding",
    "doc_ids_by_parent",
    "documents",
    "empty_store",
    "hybrid_embedding",
    "hybrid_lexical",
    "hybrid_retriever",
    "hybrid_store",
    "hybrid_vector_retriever",
    "make_record",
    "metadata_field_names",
    "query_vectors",
    "sample_metadata",
    "sample_records",
]
