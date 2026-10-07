"""day066 检索测试用的确定性小库构造器（记录表 + 三个编码器 + 清单助手）.

单独一个模块而不是复制进四个测试文件，理由与 ``vectorstore_samples`` /
``chunking_samples`` 一样：**同一份样本要被 types / filters / retriever / routing
四个测试文件同时用到**，复制多份会让"样本里加一条记录"变成要改许多处，
而漏改的那一处只表现为"某个测试不再覆盖那条语义"。

## 这批样本刻意埋了四个差异点

```text
多条同 doc             doc-alpha 有 3 条、doc-gamma 有 3 条
                       → max_per_doc 的裁剪、doc_ids 去重才有东西可剪
缺 created_at          c-c-01 的 metadata 里**没有**这个键
                       → "字段缺失的记录被排除"变成一个可核对的数字
逐位相同的同分对       c-a-01 与 c-b-01 的文本与向量逐位相同
                       → 同分必须由 id 升序决定（否则两次运行的 top-1 会不同）
可过滤的元数据字段     strategy / heading_path / tags / topic / created_at
                       → where、时间范围、字段拼写检查三类用例都有素材
```

## 期望值为什么能手算

向量维度取 8（与 ``vectorstore_samples`` 同一个量级），分量只用 0 / 0.5 / 0.6 /
0.8 / 1.0，于是余弦分数都是可以直接写下来的数：``(1,0,…)`` 与 ``(0.8,0.6,…)``
的内积是 0.8，而两个 ``(1,0,…)`` 的分数**精确相等**（不是近似相等）——
后者正是 ``>=`` 阈值语义与"同分按 id"两条用例的立足点。

三个查询向量给出三种不同的分数分布，这样"阈值切掉几条"才能写成一个精确的数字：

```text
axis    1.0 / 1.0 / 0.8 / 0.6 / 0.5 / 0 / 0 / 0   前两名精确并列（同分排序的素材）
tilt    1.0 / 0.96 / 0.8 / 0.7 / 0.6 / 0.6 / 0 / 0  分数彼此拉开（阈值可以与某条精确相等）
ortho   八条全是 0.0                                全库并列（"阈值把命中全切了"的素材）
```

## 三种编码器各自干什么

```text
TableEmbedding         按文本查表 → 分数由人给定（断言写的是期望值，不是实现）
CharNgramEmbedding     真实但离线的字符 n-gram，只用来验证"维度对得上"的端到端路径
FixedVectorEmbedding   永远返回同一个向量 → 零向量 / nan / 维度不符三类"编码器坏了"
NotASequenceEmbedding  返回非序列 → 模拟提供方把 ndarray 或字符串塞进了结果
```

``TableEmbedding`` 记录每次 ``embed`` 收到的文本（``calls``）：这样"查询被
strip 过再编码"、"一次检索只编码一次"这两件事都能被断言，而不是只能相信代码。

## id 为什么是可读形状

``c-a-01`` 表示"doc-alpha 的第 1 条"，而不是 day062 的 16 位十六进制。
理由：这一层的测试要逐条核对"同分按 id 升序"的期望值，而 id 的形状本身
已经在 day062 / day064 的用例里覆盖过——此处可读性比形态一致性更重要。

全部离线、确定性、零网络、不写仓库外文件。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.indexing.manifest import manifest_from_store
from smart_research_agent.indexing.types import EmbeddingIdentity, IndexManifest
from smart_research_agent.llm.embedding import CharNgramEmbedding, EmbeddingProvider
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import VectorRecord, make_record

#: 样本向量维度。取 8 而不是 384：期望值能手算，且占用极小。
VECTOR_DIMENSION = 8

#: 三个文档 id（day062 的 ``parent_doc_id``；多样性裁剪就是按它分组）.
DOC_ALPHA = "doc-alpha"
DOC_BETA = "doc-beta"
DOC_GAMMA = "doc-gamma"

#: 三个来源路径（``RetrievalHit.citation`` 优先用它渲染引用）.
SOURCE_ALPHA = "docs/检索手册.md"
SOURCE_BETA = "docs/语义缓存.md"
SOURCE_GAMMA = "docs/附录.md"

#: **被两条记录共用的正文**：``c-a-01`` 与 ``c-b-01`` 因此得到逐位相同的向量。
#: 这条相同不是巧合，而是"同分排序"用例的数据来源（用同一个常量写两遍最可靠）。
TEXT_SHARED = "相似度阈值用分位数标定，换 embedding 模型必须重新标定。"

#: 八条样本记录：``(id, 向量, 正文, 元数据)`` 摊平成字典，便于逐键引用。
#:
#: 向量与"哪个查询最近"的对应关系是**人给的**（见 ``TableEmbedding``），
#: 因此下面每条断言的期望值都能在纸上算出来：
#:
#: ```text
#: query "axis" (1,0,0,…)   分数 1.0 / 1.0 / 0.8 / 0.6 / 0.5 / 0 / 0 / 0
#: query "tilt" (0.6,0.8,0) 分数 1.0 / 0.96 / 0.7 / 0.6 / 0.6 / 0 / 0 / 0
#: query "ortho" (0,…,0,1)  分数 全是 0.0（八条并列，只能靠 id 决定次序）
#: ```
RECORD_SPECS: tuple[dict[str, Any], ...] = (
    {
        "record_id": "c-a-01",
        "vector": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "text": TEXT_SHARED,
        "metadata": {
            "parent_doc_id": DOC_ALPHA,
            "source": SOURCE_ALPHA,
            "strategy": "structural",
            "index": 0,
            "token_count": 120,
            "created_at": "2026-09-01",
            "heading_path": "检索手册 > 分块",
            "topic": "chunking",
            "tags": ["guide", "chunk"],
        },
    },
    {
        "record_id": "c-a-02",
        "vector": (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "text": "默认返回前 5 条，召回深度按三倍过取，阈值在检索层落刀。",
        "metadata": {
            "parent_doc_id": DOC_ALPHA,
            "source": SOURCE_ALPHA,
            "strategy": "structural",
            "index": 1,
            "token_count": 160,
            "created_at": "2026-09-05",
            "heading_path": "检索手册 > 检索",
            "topic": "retrieval",
            "tags": ["guide", "search"],
        },
    },
    {
        "record_id": "c-a-03",
        "vector": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "text": "一次全量重建索引大约 12 万条记录，按每条 320 token 计。",
        "metadata": {
            "parent_doc_id": DOC_ALPHA,
            "source": SOURCE_ALPHA,
            "strategy": "fixed",
            "index": 2,
            "token_count": 90,
            "created_at": "2026-09-10",
            "heading_path": "检索手册 > 成本",
            "topic": "cost",
            "tags": ["guide", "cost"],
        },
    },
    {
        "record_id": "c-b-01",
        "vector": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "text": TEXT_SHARED,
        "metadata": {
            "parent_doc_id": DOC_BETA,
            "source": SOURCE_BETA,
            "strategy": "structural",
            "index": 0,
            "token_count": 120,
            "created_at": "2026-09-15",
            "heading_path": "语义缓存 > 阈值",
            "topic": "cache",
            "tags": ["guide", "cache"],
        },
    },
    {
        "record_id": "c-b-02",
        "vector": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "text": "术语表列出本手册用到的全部缩写与它们的英文全称。",
        "metadata": {
            "parent_doc_id": DOC_BETA,
            "source": SOURCE_BETA,
            "strategy": "semantic",
            "index": 1,
            "token_count": 64,
            "created_at": "2026-09-20",
            "heading_path": "附录 > 术语表",
            "topic": "glossary",
            "tags": ["appendix"],
        },
    },
    {
        # **唯一一条没有 created_at 的记录**：时间过滤会把它排除，这是显式约定。
        "record_id": "c-c-01",
        "vector": (0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        "text": "参数表把默认值与取值范围并排列出，便于逐项核对配置。",
        "metadata": {
            "parent_doc_id": DOC_GAMMA,
            "source": SOURCE_GAMMA,
            "strategy": "fixed",
            "index": 0,
            "token_count": 200,
            "heading_path": "附录 > 参数表",
            "topic": "glossary",
            "tags": ["appendix", "cost"],
        },
    },
    {
        "record_id": "c-c-02",
        "vector": (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "text": "阈值用分位数标定，语义缓存与检索共用同一套标定流程。",
        "metadata": {
            "parent_doc_id": DOC_GAMMA,
            "source": SOURCE_GAMMA,
            "strategy": "structural",
            "index": 1,
            "token_count": 140,
            "created_at": "2026-08-31",
            "heading_path": "检索手册 > 阈值",
            "topic": "cache",
            "tags": ["guide", "cache"],
        },
    },
    {
        "record_id": "c-c-03",
        "vector": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "text": "术语表与参数表都放在附录里，方便一次查完。",
        "metadata": {
            "parent_doc_id": DOC_GAMMA,
            "source": SOURCE_GAMMA,
            "strategy": "semantic",
            "index": 2,
            "token_count": 48,
            "created_at": "2026-09-25",
            "heading_path": "附录 > 术语表",
            "topic": "glossary",
            "tags": ["appendix"],
        },
    },
)

#: 八条记录的 id（按 ``RECORD_SPECS`` 顺序，也就是"插入顺序"）.
#:
#: 注意它与"升序"**不同**：``c-c-03`` 排在 ``c-b-*`` 之后，但字典序上
#: ``c-b-…`` 更小。同分排序的用例正是靠这个差别才有区分度。
RECORD_IDS: tuple[str, ...] = tuple(spec["record_id"] for spec in RECORD_SPECS)

#: id → 正文（编码器与引用渲染都要用）.
RECORD_TEXTS: dict[str, str] = {
    spec["record_id"]: str(spec["text"]) for spec in RECORD_SPECS
}

#: 那对**文本与向量逐位相同**的记录：``axis`` 查询下它们的分数精确相等.
TIE_IDS: tuple[str, str] = ("c-a-01", "c-b-01")

#: 唯一缺少 ``created_at`` 的记录：任何时间范围都会把它排除.
MISSING_CREATED_AT_ID = "c-c-01"

#: 两个查询文本（它们是 ``TableEmbedding`` 查表的键，本身没有语义）.
QUERY_TEXTS: dict[str, str] = {
    "axis": "检索手册里怎么配置时间范围过滤",
    "tilt": "语义缓存的阈值应该怎么标定",
    "ortho": "今天午饭吃什么",
}

#: 查询名 → 查询向量。三个向量刻意给出三种不同的分数分布（见 ``RECORD_SPECS``）.
QUERY_VECTORS: dict[str, tuple[float, ...]] = {
    "axis": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "tilt": (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "ortho": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}

#: ``axis`` 查询下期望的完整次序（分数降序 + 同分按 id 升序）.
#:
#: 写成一个常量而不是在断言里现算：一旦样本被改动，这个列表会**同时**被
#: 四个测试文件用到，于是"改样本"这件事会在多处一起响，而不是只坏一个断言。
EXPECTED_AXIS_ORDER: tuple[str, ...] = (
    "c-a-01",
    "c-b-01",
    "c-a-02",
    "c-c-02",
    "c-c-01",
    "c-a-03",
    "c-b-02",
    "c-c-03",
)

#: ``axis`` 查询下期望的分数（与 ``EXPECTED_AXIS_ORDER`` 逐项对应）.
EXPECTED_AXIS_SCORES: tuple[float, ...] = (1.0, 1.0, 0.8, 0.6, 0.5, 0.0, 0.0, 0.0)

#: ``tilt`` 查询下期望的完整次序（分数 1.0 / 0.96 / 0.8 / 0.7 / 0.6 / 0.6 / 0 / 0）.
EXPECTED_TILT_ORDER: tuple[str, ...] = (
    "c-c-02",
    "c-a-02",
    "c-a-03",
    "c-c-01",
    "c-a-01",
    "c-b-01",
    "c-b-02",
    "c-c-03",
)

#: ``ortho`` 查询下的次序：八条分数**全部为 0.0**，因此完全由 id 决定.
EXPECTED_ORTHO_ORDER: tuple[str, ...] = (
    "c-a-01",
    "c-a-02",
    "c-a-03",
    "c-b-01",
    "c-b-02",
    "c-c-01",
    "c-c-02",
    "c-c-03",
)


def query_text(name: str) -> str:
    """取一个查询文本（名字拼错时 ``KeyError`` 会指出可用名）."""
    return QUERY_TEXTS[name]


def sample_metadata() -> dict[str, dict[str, Any]]:
    """id → 元数据（**深一层拷贝**：数组值也复制，避免用例之间互相污染）."""
    return {
        spec["record_id"]: _copy_metadata(spec["metadata"]) for spec in RECORD_SPECS
    }


def metadata_field_names() -> list[str]:
    """样本库里出现过的**全部**元数据字段名（升序）.

    它是 ``unknown_filter_fields`` 的"已知字段全集"：拼写检查能发现
    "整个库里都没有这个键"，而它的输入正是这一份列表。
    """
    names: set[str] = set()
    for spec in RECORD_SPECS:
        names.update(str(key) for key in spec["metadata"])
    return sorted(names)


def doc_ids_by_parent() -> dict[str, list[str]]:
    """parent_doc_id → 该文档下的记录 id（升序）——多样性用例的素材."""
    grouped: dict[str, list[str]] = {}
    for spec in RECORD_SPECS:
        parent = str(spec["metadata"]["parent_doc_id"])
        grouped.setdefault(parent, []).append(str(spec["record_id"]))
    return {name: sorted(ids) for name, ids in grouped.items()}


def sample_vectors() -> dict[str, tuple[float, ...]]:
    """id → 原始向量（cosine 下 ``make_record`` 会归一化，这里的分量已经归一）. """
    return {spec["record_id"]: tuple(spec["vector"]) for spec in RECORD_SPECS}


def sample_records(*, metric: str = "cosine", reverse: bool = False) -> list[VectorRecord]:
    """八条 ``VectorRecord``（``reverse=True`` 时插入顺序相反，用于验证可复现性）."""
    specs = list(RECORD_SPECS)
    if reverse:
        specs.reverse()
    return [
        make_record(
            record_id=str(spec["record_id"]),
            vector=tuple(spec["vector"]),
            text=str(spec["text"]),
            metadata=_copy_metadata(spec["metadata"]),
            metric=metric,
        )
        for spec in specs
    ]


def flat_store(
    records: list[VectorRecord] | None = None,
    *,
    metric: str = "cosine",
    dimension: int | None = None,
) -> FlatVectorStore:
    """用一个给定的记录表建库（``dimension`` 缺省时随第一条记录学到）."""
    store = FlatVectorStore(metric=metric, dimension=dimension)
    if records:
        store.upsert(records)
    return store


def sample_store(*, metric: str = "cosine", reverse: bool = False) -> FlatVectorStore:
    """装好八条样本记录的 flat 库（检索器用例的默认起点）."""
    return flat_store(
        sample_records(metric=metric, reverse=reverse),
        metric=metric,
    )


def empty_store(*, metric: str = "cosine") -> FlatVectorStore:
    """一个**空**库（维度固定为 8，与 ``sample_store`` 同规格）.

    空库是合法状态：检索器为它返回 ``empty_reason="no_data"`` 而不报错
    （见 ``Retriever`` 第 2 步）。
    """
    return FlatVectorStore(metric=metric, dimension=VECTOR_DIMENSION)


def plain_records(*, count: int = 2, text_prefix: str = "无 parent 的片段") -> list[VectorRecord]:
    """若干条**没有 ``parent_doc_id``** 的记录（多样性裁剪的边界素材）.

    它们的分组键都是空串（"自成一类"），因此 ``max_per_doc=1`` 时
    这些记录会**互相挤**——这条规则对所有人一致，不能因为它们没有 parent
    就把它们全留下（见 ``retriever._apply_diversity`` 的取舍）。
    """
    return [
        make_record(
            record_id=f"plain-{index:02d}",
            vector=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            text=f"{text_prefix} {index}",
            metadata={"source": "docs/未归属.md", "strategy": "fixed"},
        )
        for index in range(count)
    ]


def sample_manifest(
    store: FlatVectorStore,
    *,
    provider: str = "TableEmbedding",
    model: str = "table-v1",
    dimension: int = VECTOR_DIMENSION,
    metric: str | None = None,
) -> IndexManifest:
    """按库的现状重建一份清单（漂移用例的"上一版"）.

    默认身份与 ``TableEmbedding`` 对上（provider 名 = 类名、维度 = 8），
    于是"编码器与清单身份不符"那两条注记不会平白出现；
    要构造它们只需把 ``provider`` / ``dimension`` 换掉。
    """
    return manifest_from_store(
        store,
        identity=EmbeddingIdentity(provider=provider, model=model, dimension=dimension),
        metric=metric,
    )


class TableEmbedding(EmbeddingProvider):
    """按文本查表的确定性编码器：向量由样本写死（"语义"由人给定）.

    为什么不用 ``MockEmbedding``：它把文本哈希成向量，于是"哪条最近"只能靠
    实际跑一遍才知道——那样写出来的断言只是在复述实现。这里把文本与向量的
    对应关系写死在 ``QUERY_TEXTS`` 里，**期望值是手算出来的**。

    ``calls`` 记录每次 ``embed`` 收到的文本：查询是否被 strip 过、一次检索
    是否只编码一次，这两件事因此可以被断言。``embed_batch`` 也记进 ``calls``
    （它逐条转发），所以"批量与逐条的入参一致"同样可查。
    """

    def __init__(
        self,
        table: dict[str, tuple[float, ...]] | None = None,
        *,
        dimension: int = VECTOR_DIMENSION,
        default: tuple[float, ...] | None = None,
    ) -> None:
        resolved = dict(table or {QUERY_TEXTS[name]: QUERY_VECTORS[name] for name in QUERY_TEXTS})
        self._table = {key: tuple(float(x) for x in value) for key, value in resolved.items()}
        if default is None:
            default = QUERY_VECTORS["axis"]
        self._default = tuple(float(x) for x in default)
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        """向量维度（与库的维度必须一致，否则检索器会在第 4 步报错）."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """按文本查表；查不到用 ``axis`` 的向量（并记一次调用）."""
        self.calls.append(text)
        return list(self._table.get(text, self._default))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码：逐条转发（``calls`` 会把整批文本都记下来）."""
        return [self.embed(text) for text in texts]


class FixedVectorEmbedding(EmbeddingProvider):
    """永远返回同一个向量的编码器（造三类"编码器坏了"的输入）.

    ```text
    [0.0] * 8   零向量 → 余弦 0/0，会"平等地命中或不命中一切"
    [nan] * 8   非有限数 → 参与的比较恒为 False，排序不可复现
    4 个分量     维度不符 → 编码器与建库的不是同一套
    ```
    """

    def __init__(self, vector: list[float], *, dimension: int | None = None) -> None:
        self._vector = [float(value) for value in vector]
        self._dimension = dimension if dimension is not None else len(self._vector)

    @property
    def dimension(self) -> int:
        """声明出来的维度（可以与向量的长度不一致——这正是被检出的错误）."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """忽略入参，返回那个固定向量."""
        return list(self._vector)


class NotASequenceEmbedding(EmbeddingProvider):
    """返回**非序列**的编码器：模拟提供方把 ndarray / 字符串塞进结果.

    它是"提供方实现违反了 ``EmbeddingProvider.embed`` 的返回约定"这条路径的
    唯一触发方式——正常的编码器不会这样，但第三方实现会。
    """

    def __init__(self, dimension: int = VECTOR_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """向量维度（这里只是为了满足抽象属性）."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """故意返回一个字符串（类型注解说的是 list[float]）."""
        return "not-a-vector"  # type: ignore[return-value]


def lexical_embedding(dimension: int = 64) -> CharNgramEmbedding:
    """离线的字符 n-gram 编码器（真实实现，用来验证"维度对得上"的端到端路径）.

    维度取 64 而不是默认的 256：样本只有 8 条记录，64 维已经足够避免桶碰撞，
    而它在报告里更容易一眼看清"库是几维的"。
    """
    return CharNgramEmbedding(dimension=dimension)


def encoded_store(
    embedding: EmbeddingProvider,
    *,
    metric: str = "cosine",
) -> FlatVectorStore:
    """用**给定编码器**把八条样本记录的正文编码后建库（维度随编码器而定）.

    用途有两个，都是"编码器与库维度"这一对的用例：
    ``encoded_store(lexical_embedding())`` 配同一个编码器 → 端到端能跑通；
    配 ``TableEmbedding()``（8 维）→ 当场触发 ``IndexStateError``。
    """
    records = [
        make_record(
            record_id=str(spec["record_id"]),
            vector=embedding.embed(str(spec["text"])),
            text=str(spec["text"]),
            metadata=_copy_metadata(spec["metadata"]),
            metric=metric,
        )
        for spec in RECORD_SPECS
    ]
    return flat_store(records, metric=metric)


def _copy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """浅拷贝一层、数组值再复制一份（样本之间不共享可变对象）."""
    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        copied[str(key)] = list(value) if isinstance(value, list) else value
    return copied


__all__ = [
    "DOC_ALPHA",
    "DOC_BETA",
    "DOC_GAMMA",
    "EXPECTED_AXIS_ORDER",
    "EXPECTED_AXIS_SCORES",
    "EXPECTED_ORTHO_ORDER",
    "EXPECTED_TILT_ORDER",
    "MISSING_CREATED_AT_ID",
    "QUERY_TEXTS",
    "QUERY_VECTORS",
    "RECORD_IDS",
    "RECORD_SPECS",
    "RECORD_TEXTS",
    "SOURCE_ALPHA",
    "SOURCE_BETA",
    "SOURCE_GAMMA",
    "TEXT_SHARED",
    "TIE_IDS",
    "VECTOR_DIMENSION",
    "FixedVectorEmbedding",
    "NotASequenceEmbedding",
    "TableEmbedding",
    "doc_ids_by_parent",
    "empty_store",
    "encoded_store",
    "flat_store",
    "lexical_embedding",
    "make_record",
    "metadata_field_names",
    "plain_records",
    "query_text",
    "sample_manifest",
    "sample_metadata",
    "sample_records",
    "sample_store",
    "sample_vectors",
]
