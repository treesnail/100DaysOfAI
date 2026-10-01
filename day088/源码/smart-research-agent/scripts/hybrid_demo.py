#!/usr/bin/env python
"""day067 演示脚本：混合检索（M6-D6）——两路一起答，并说清每一条是哪一路顶上来的.

九节（第 1 节建库，后八节一一对应教程里的八个演示），全部**离线、确定性、零网络**，
只依赖本包与标准库：

```text
1. 建一个小库      10 条记录 / 3 份文档 / 一条缺 created_at；两路的索引都现建
2. top-k 对照     同一句话：只有向量路 / 只有关键词路 / 两路融合，三份名单并排看
3. 互补案例       词面精确（ERR-2043 / 401）砸中关键词路，语义改写砸中向量路
4. RRF 手算核对   1/(k+rank+1) 手算一遍，与引擎逐位比（本课最硬的一条证据）
5. weighted 扫描  归一化 + alpha 扫描表：权重真的在起作用
6. k1 / b 扫描    两个 BM25 参数各自的**方向**（在受控语料上）
7. 去重与证据     同一条被两路召回 = 合并证据（channels 两条、贡献相加）
8. 阈值只作用向量路  min_score 落刀的位置，以及它为什么进 notes
9. 空结果诊断     过滤在**两路**都落刀；反面对照说明"只过滤一路"会发生什么
```

## 这一课要证明的那句话

```text
向量路擅长的     同义改写、语义相近但词面完全不同的问法
关键词路擅长的   编号、函数名、报错码、专有名词（字面精确）
```

第 3 节是它的正面陈述，第 9 节是它的反面（"过滤只在一路落刀"会发生什么）。

## 为什么这里的向量要写死成查表（与 day066 的演示脚本刻意不同）

day066 的演示用真实的 ``CharNgramEmbedding``，理由是"那一课要打印的是分数与命中
关系，写死的向量只证明我按表查了一下"。这一课要打印的是**一个关于映射关系的断言**：

> 向量路对编号（``ERR-2043``）**没有方向**，它会去命中"讲向量是什么"的那一条。

这条断言只能靠一份写死的映射来构造（真实编码器的行为要看训练分布，无法在这里
稳定复现）。因此本脚本用一张 ``查询文本 → 向量`` 的表，而**分数仍然是真算的**：
余弦相似度在两个已知向量之间，1.0 与 0.0 都是可以手算出来的数。

作为对照，第 3 节的最后一小段会**真的**用 ``CharNgramEmbedding`` 跑一遍同样的
那句话——它会命中含编号的那一条（因为字符 n-gram 共享了 ``err`` / ``2043``
里的字符）。这条对照要留着，免得上面那句话被读成"向量检索天生不行"：
**它只在"编码器没见过这个编号"时成立**，而那是最常见也最危险的情形。

## 样本与测试样本刻意分开

``tests/`` 里也有一份语料，但**脚本不 import tests**：脚本要能被别人复制走
单独运行，一旦依赖测试目录，复制到别处就跑不起来。因此这里自带一份
10 条记录的样本（正文与测试样本同源，便于对着文档核对）。

## 清单是现算的，不落盘

脚本用 day065 的 ``build_manifest`` **现算**一份索引清单：它不进
``data/index``，因此不会在仓库里留下一个看起来像「线上索引」的目录。
脚本结果同时打印到 stdout 并写入 ``outputs/hybrid_demo.txt``。

运行（cwd 为 ``day067/源码/smart-research-agent``）::

    python scripts/hybrid_demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# 本脚本要能直接 ``python scripts/hybrid_demo.py`` 跑起来（cwd 是仓库根）。
# 以脚本方式启动时 ``sys.path[0]`` 是 ``scripts/``，仓库根不在其中，
# 而本快照的包**没有 pip 安装**（``pip show smart-research-agent`` 为空）——
# 因此这里显式把仓库根塞进 ``sys.path``，让脚本不依赖 ``PYTHONPATH=.``。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.indexing.manifest import manifest_from_store  # noqa: E402
from smart_research_agent.indexing.types import EmbeddingIdentity  # noqa: E402
from smart_research_agent.llm.embedding import (  # noqa: E402
    CharNgramEmbedding,
    EmbeddingProvider,
)
from smart_research_agent.retrieval import (  # noqa: E402
    DEFAULT_B,
    DEFAULT_K1,
    BM25Params,
    HybridRetriever,
    LexicalDocument,
    LexicalIndex,
    RetrievalQuery,
    Retriever,
    build_retriever,
    fuse,
    tokenize,
    weighted_score_fusion,
)
from smart_research_agent.retrieval.types import (  # noqa: E402
    CHANNEL_BM25,
    CHANNEL_VECTOR,
)
from smart_research_agent.vectorstore import FlatVectorStore  # noqa: E402
from smart_research_agent.vectorstore.types import VectorRecord, make_record  # noqa: E402

#: 本层的边界（原文供教程引用，见 ``retrieval/hybrid.py`` 的模块 docstring）.
LAYER_BOUNDARY = (
    "向量路回答'意思像不像'，关键词路回答'这个词在不在'——"
    "把两路合起来不是为了更强，而是为了**两种失败不再是同一种失败**。"
)

#: 输出目录与文件名（``outputs/`` 在仓库 ``.gitignore`` 里，不污染仓库）.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "hybrid_demo.txt"

#: 样本向量维度。取 8：余弦分数只有 1.0 与 0.0 两种取值，
#: 于是"0 分并列"这件事是一个**精确的事实**（手算与断言都省事）。
DEMO_DIMENSION = 8

#: 三份文档（day062 的 ``parent_doc_id``；多样性裁剪按它分组）.
DOC_MANUAL = "doc-manual"
DOC_TROUBLE = "doc-trouble"
DOC_CONCEPT = "doc-concept"

#: 十条记录：``(id, 向量, 正文, 文档, strategy, index, topic, created_at)``.
#:
#: 两条设计刻意为之（理由见模块 docstring）：
#:
#: ```text
#: h-t-01 讲 ERR-2043，但向量被放在第 8 根轴上（与"讲向量是什么"的那条错开）
#: h-t-03 没有 created_at        → 时间过滤会把两路的产品都排除掉（约定三）
#: ```
DEMO_SPECS: tuple[tuple[Any, ...], ...] = (
    (
        "h-m-01", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "语义缓存用分位数标定阈值，换编码器必须重新标定。",
        DOC_MANUAL, "structural", 0, "cache", "2026-09-01",
    ),
    (
        "h-m-02", (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "召回深度按三倍过取，阈值在检索层落刀。",
        DOC_MANUAL, "structural", 1, "retrieval", "2026-09-05",
    ),
    (
        "h-m-03", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "一次全量重建索引大约 12 万条记录，按每条 320 token 计。",
        DOC_MANUAL, "fixed", 2, "cost", "2026-09-10",
    ),
    (
        "h-t-01", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "ERR-2043 表示向量维度不一致，先重建索引再重试。",
        DOC_TROUBLE, "semantic", 0, "errors", "2026-09-12",
    ),
    (
        "h-t-02", (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "重试预算 retry_budget 缺省 3 次，超时按指数退避。",
        DOC_TROUBLE, "semantic", 1, "errors", "2026-09-15",
    ),
    (
        "h-t-03", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "报错码 401 与 403 的处置方式完全不同。",
        DOC_TROUBLE, "fixed", 2, "errors", None,
    ),
    (
        "h-c-01", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        "把一句话映射到低维空间的向量，越近表示意思越像。",
        DOC_CONCEPT, "structural", 0, "embedding", "2026-09-18",
    ),
    (
        "h-c-02", (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "同义词与改写会让词面匹配失效，这时要靠语义相似度。",
        DOC_CONCEPT, "structural", 1, "embedding", "2026-09-20",
    ),
    (
        "h-c-03", (0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0),
        "术语表给出缩写与英文全称的对照。",
        DOC_CONCEPT, "semantic", 2, "glossary", "2026-09-22",
    ),
    (
        "h-c-04", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "关键词检索擅长精确匹配：编号、函数名、报错码。",
        DOC_CONCEPT, "semantic", 3, "keyword", "2026-09-25",
    ),
)

#: 查询文本 → 查询向量（见模块 docstring 的说明：这张表是**断言**的一部分）.
#:
#: ``QUERY_EXACT`` 指向 h-c-01（讲"向量是什么"），而真正含 ``ERR-2043`` 的是 h-t-01
#: ——这正是向量路在编号面前"差不多就行"的复现。
QUERY_VECTORS: dict[str, tuple[float, ...]] = {
    "ERR-2043": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "401": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "午饭吃点什么好": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "retry_budget": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

#: 四个演示查询（名字与语义见上面那张表）.
QUERY_EXACT = "ERR-2043"
QUERY_CODE = "401"
QUERY_SEMANTIC = "午饭吃点什么好"
QUERY_BOTH = "retry_budget"

#: 受控梯度语料（第 6 节的 k1 / b 扫描）：同一个词元的词频与长度都可控.
GRADED_DOCS: tuple[tuple[str, str], ...] = (
    ("heavy", "缓存缓存缓存缓存"),
    ("middle", "缓存缓存检索"),
    ("light", "缓存"),
)


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# 样本构造
# --------------------------------------------------------------------------- #


class TableEmbedding(EmbeddingProvider):
    """按文本查表的确定性编码器（本脚本的向量表，理由见模块 docstring）.

    ``calls`` 记录每次 ``embed`` 收到的文本：这样"一次混合检索只编码一次"
    这件事可以被打印出来核对（两路里只有一路需要编码器）。
    """

    def __init__(
        self,
        table: dict[str, tuple[float, ...]],
        *,
        dimension: int = DEMO_DIMENSION,
    ) -> None:
        self._table = {key: tuple(float(value) for value in row) for key, row in table.items()}
        self._default = (1.0,) + (0.0,) * (dimension - 1)
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        """向量维度（与库的维度必须一致，否则检索器会在编码那一步报错）."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """按文本查表；查不到用兜底向量（并记一次调用）."""
        self.calls.append(text)
        return list(self._table.get(text, self._default))


def demo_records(*, metric: str = "cosine") -> list[VectorRecord]:
    """十条 ``VectorRecord``（元数据按 day062 的 ``knowledge_records()`` 形状给）."""
    records: list[VectorRecord] = []
    for spec in DEMO_SPECS:
        metadata: dict[str, Any] = {
            "parent_doc_id": spec[3],
            "source": f"docs/{spec[3]}.md",
            "strategy": spec[4],
            "index": spec[5],
            "topic": spec[6],
        }
        if spec[7] is not None:
            metadata["created_at"] = spec[7]
        records.append(
            make_record(
                record_id=str(spec[0]),
                vector=tuple(float(value) for value in spec[1]),
                text=str(spec[2]),
                metadata=metadata,
                metric=metric,
            )
        )
    return records


def demo_store(*, metric: str = "cosine") -> FlatVectorStore:
    """装好十条样本记录的 flat 库."""
    store = FlatVectorStore(metric=metric, dimension=DEMO_DIMENSION)
    store.upsert(demo_records(metric=metric))
    return store


def demo_lexical(store: FlatVectorStore, *, params: BM25Params | None = None) -> LexicalIndex:
    """从**同一个库**现建一份关键词索引（两路看到的是同一批文档）."""
    return LexicalIndex.from_backend(store, params=params)


def demo_vector(
    store: FlatVectorStore,
    embedding: EmbeddingProvider,
    **overrides: Any,
) -> Retriever:
    """只有向量一路的检索器（对照组）."""
    return build_retriever(store, embedding, **overrides)


def demo_hybrid(
    store: FlatVectorStore,
    embedding: EmbeddingProvider,
    lexical: LexicalIndex,
    **overrides: Any,
) -> HybridRetriever:
    """两路齐全的混合检索器（第 1 节的装配有个总览）."""
    return HybridRetriever(demo_vector(store, embedding), lexical, **overrides)


def ids_of(result: Any) -> str:
    """``record_id`` 列表渲染成一行（打印表格用）."""
    return "、".join(result.ids()) if result.ids() else "（空）"


def show_hits(label: str, result: Any, *, limit: int = 5) -> None:
    """打印一份名单（前一列是名次，通道与分数一起给）."""
    print(f"  {label}")
    if not result.hits:
        print("      （一条都没有）")
        return
    for hit in result.hits[:limit]:
        channels = "/".join(getattr(hit, "channels", ()) or (getattr(hit, "channel", ""),))
        preview = hit.text.replace("\n", " ")[:30]
        print(
            f"      #{hit.rank} {hit.record_id:<7} [{channels:<13}] "
            f"score={hit.score:+.6f} | {preview}"
        )


# --------------------------------------------------------------------------- #
# 第 1 节：建库与两路的现状
# --------------------------------------------------------------------------- #


def section_1_corpus() -> None:
    """建一个小库，并把两路各自的现状打印出来（后面的每一节都从它出发）."""
    title("第 1 节：建一个小库（10 条记录 / 3 份文档 / 一条缺 created_at）")
    print(f"边界（本层的原文）：{LAYER_BOUNDARY}")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    print(f"\n向量库：{store.info().count} 条 / {store.dimension} 维 / 度量 {store.metric}")
    print(
        f"编码器：{type(embedding).__name__}（{embedding.dimension} 维，按一张写死的向量表查表"
        "——为什么写死见模块 docstring）"
    )
    print(f"关键词索引：{lexical.summary_line()}")
    print(f"BM25 参数：k1={lexical.params.k1}（词频饱和）、b={lexical.params.b}（长度归一化）")
    print("\n十条样本记录（正文 / 文档 / 有没有 created_at）：")
    for spec in DEMO_SPECS:
        created = spec[7] if spec[7] is not None else "（缺）"
        print(f"  {spec[0]:<7} {spec[3]:<12} {created:<12} {str(spec[2])[:34]}")
    print(
        f"\n检索器的默认值（来自 config）：top_k={settings.retrieval_top_k}、"
        f"过取倍率={settings.retrieval_fetch_multiplier}、"
        f"混合检索开关={settings.retrieval_hybrid_enabled}（缺省关闭）"
    )
    print(
        "\n注意那条没有 created_at 的记录（h-t-03）：它会被**两路**同时排除——"
        "过滤语义同一份，缺失字段的处置也同一份。"
    )
    print(
        "\n分词示例（零依赖、CJK 感知）："
        f"tokenize('语义缓存') = {tokenize('语义缓存')}"
    )
    print(f"tokenize('ERR-2043') = {tokenize('ERR-2043')}（连字符是分隔符）")


# --------------------------------------------------------------------------- #
# 第 2 节：三份名单并排看
# --------------------------------------------------------------------------- #


def section_2_topk() -> None:
    """同一句话：只有向量路 / 只有关键词路 / 两路融合，三份名单对照."""
    title("第 2 节：同一句话，三份名单（vector / bm25 / hybrid）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    vector = demo_vector(store, embedding)
    hybrid = demo_hybrid(store, embedding, lexical)
    for query in (QUERY_EXACT, QUERY_CODE, QUERY_SEMANTIC, QUERY_BOTH):
        depth = settings.retrieval_top_k * settings.retrieval_fetch_multiplier
        print(f"\n查询 {query!r}（top_k={settings.retrieval_top_k} → 深度 {depth}）")
        show_hits("向量路（只有语义）", vector.retrieve(query))
        show_hits("关键词路（只有字面）", _lexical_view(lexical.search(query)))
        result = hybrid.retrieve(query)
        show_hits("混合（融合两路）", result)
        print(
            f"      两路召回：vector {result.channel_candidates[CHANNEL_VECTOR]} 条、"
            f"bm25 {result.channel_candidates[CHANNEL_BM25]} 条 | "
            f"并集 {result.candidates} 条 | 融合 {result.fusion['fused']} 条 | "
            f"去重合并 {result.fusion['deduped']} 条"
        )
    print(
        "\n读法：前两行各有一处「瞎」，第三行把两处的长处合起来——"
        "而它凭什么这么合，就是第 4 节要算的账。"
    )


class _LexicalView:
    """把 ``LexicalSearchResult`` 包成一个有 ``hits`` / ``ids()`` 的视图（打印用）."""

    def __init__(self, result: Any) -> None:
        self.hits = result.hits

    def ids(self) -> list[str]:
        """命中的 id（按名次）."""
        return [hit.record_id for hit in self.hits]


def _lexical_view(result: Any) -> _LexicalView:
    """``LexicalSearchResult`` → 打印视图（``show_hits`` 只要求 ``hits`` 与 ``ids()``）."""
    return _LexicalView(result)


# --------------------------------------------------------------------------- #
# 第 3 节：互补案例
# --------------------------------------------------------------------------- #


def section_3_complementary() -> None:
    """词面精确 vs 语义改写：两路各有一个"只有它能做"的案例."""
    title("第 3 节：互补的两个真实案例（以及一条必须留下的反面说明）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    vector = demo_vector(store, embedding)
    hybrid = demo_hybrid(store, embedding, lexical)

    print("[案例 A] 词面精确：查询 " + repr(QUERY_EXACT))
    shallow = dict(top_k=3, fetch_k=3)
    vector_result = vector.retrieve(RetrievalQuery(text=QUERY_EXACT, **shallow))
    hybrid_result = hybrid.retrieve(QUERY_EXACT)
    print(f"  向量路的 top-3：{ids_of(vector_result)}")
    print(
        "  其中含这个编号的记录数："
        f"{sum(1 for hit in vector_result.hits if QUERY_EXACT in hit.text)}"
        "（三条分数都是 0.0——「向量路对编号没有方向」）"
    )
    recalled = _lexical_view(lexical.search(QUERY_EXACT))
    print(f"  关键词路只召回：{ids_of(recalled)}（唯一含 err 与 2043 的一条）")
    top = hybrid_result.hits[0]
    print(
        f"  混合检索的第 1 名：{top.record_id} —— 与它是 "
        f"'{'、'.join(top.channels)}' 两路的证据"
    )

    print("\n[案例 B] 语义改写：查询 " + repr(QUERY_SEMANTIC))
    semantic = hybrid.retrieve(QUERY_SEMANTIC)
    lexical_result = lexical.search(QUERY_SEMANTIC)
    missing = list(lexical_result.missing_terms)[:4]
    print(f"  关键词路：{lexical_result.count} 条（13 个词元全不在词表里：{missing} …）")
    print(f"  向量路：{ids_of(vector.retrieve(QUERY_SEMANTIC))}（它照常有答案）")
    print(f"  混合检索：{ids_of(semantic)}（全部来自向量路）")
    note = next((item for item in semantic.notes if "关键词路一条都没召回" in item), "")
    print(f"  报告里的那句话：{note[:80]}…")

    print("\n[反面说明] 换一个**真实**的编码器会怎样")
    char_store = FlatVectorStore(metric="cosine", dimension=64)
    char_records = [
        make_record(
            record_id=str(spec[0]),
            vector=CharNgramEmbedding(dimension=64).embed(str(spec[2])),
            text=str(spec[2]),
            metadata={"parent_doc_id": spec[3], "source": f"docs/{spec[3]}.md"},
        )
        for spec in DEMO_SPECS
    ]
    char_store.upsert(char_records)
    char_vector = demo_vector(char_store, CharNgramEmbedding(dimension=64))
    char_result = char_vector.retrieve(RetrievalQuery(text=QUERY_EXACT, **shallow))
    print(f"  CharNgramEmbedding(64) 的向量路 top-3：{ids_of(char_result)}")
    print(
        "  它**也**命中了含编号的那一条（字符 n-gram 共享了 err / 2043 里的字符）。"
        "\n  因此「向量路对编号没有方向」只在**编码器没见过这个编号**时成立——"
        "\n  那是最常见的情形，但把它当成普遍规律就错了（关键词路的边际价值会随之改变）。"
    )


# --------------------------------------------------------------------------- #
# 第 4 节：RRF 手算核对
# --------------------------------------------------------------------------- #


def section_4_rrf_handcheck() -> None:
    """把 RRF 的贡献手算一遍，与引擎逐位比（本课最硬的一条证据）."""
    title("第 4 节：RRF 手算核对（贡献 = 1/(k + rank + 1)）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    vector = demo_vector(store, embedding)

    k = settings.retrieval_hybrid_rrf_k
    query = QUERY_EXACT
    shallow = dict(top_k=3, fetch_k=3)
    vector_result = vector.retrieve(RetrievalQuery(text=query, **shallow))
    lexical_result = lexical.search(query, top_k=3)
    print(f"查询 {query!r}，深度 fetch_k=3，k={k}（RRF 原论文取值）")
    vector_rows = [(hit.record_id, hit.rank, round(hit.score, 4)) for hit in vector_result.hits]
    lexical_rows = [(hit.record_id, hit.rank, round(hit.score, 4)) for hit in lexical_result.hits]
    print(f"  向量路三名：{vector_rows}")
    print(f"  关键词路命中：{lexical_rows}")

    hand: dict[str, float] = {}
    for hit in vector_result.hits:
        hand[hit.record_id] = hand.get(hit.record_id, 0.0) + 1.0 / (k + hit.rank + 1)
    for hit in lexical_result.hits:
        hand[hit.record_id] = hand.get(hit.record_id, 0.0) + 1.0 / (k + hit.rank + 1)
    print("\n  手算（纸笔公式，不从引擎拿数）：")
    for record_id, value in sorted(hand.items(), key=lambda item: (-item[1], item[0])):
        print(f"      {record_id:<7} 1/({k}+rank+1) 求和 = {value:.9f}")

    engine = fuse(
        {CHANNEL_VECTOR: vector_result.hits, CHANNEL_BM25: lexical_result.hits},
        strategy="rrf",
        k_rrf=k,
    )
    print("\n  引擎给的（同一批输入）：")
    ok = True
    for hit in engine:
        expected = hand.get(hit.record_id, 0.0)
        same = math.isclose(hit.score, expected, rel_tol=1e-12)
        ok = ok and same
        print(
            f"      #{hit.rank} {hit.record_id:<7} score={hit.score:.9f} "
            f"| 通道 {'+'.join(hit.channels):<14} | 手算一致：{same}"
        )
    print(f"\n  逐位一致：{ok}（手算与引擎用同一个定义，差一位就说明实现错了）")

    print("\n  一条容易忽略的事实：并列时由**第三排序键**决定次序")
    tie = [hit.record_id for hit in engine if abs(hit.score - 1.0 / 61.0) < 1e-12]
    print(
        f"      {'、'.join(tie)} 的分数都是 1/61 = {1.0 / 61.0:.9f}"
        f"（各自只被一路的名次 0 命中）→ 按 record_id 升序：{'、'.join(sorted(tie))}"
    )
    print(
        "      这不是「向量路赢了」，而是「分数并列时必须有确定的第三键」——"
        "没有它，同一份数据两次运行会给出不同的 top-1。"
    )


# --------------------------------------------------------------------------- #
# 第 5 节：weighted + alpha 扫描
# --------------------------------------------------------------------------- #


def section_5_weighted() -> None:
    """归一化 + alpha 扫描：权重真的在起作用（而不是被量纲吃掉）."""
    title("第 5 节：weighted 策略——先归一化，再加权（alpha 扫描表）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    vector = demo_vector(store, embedding)
    query = QUERY_EXACT
    shallow = dict(top_k=3, fetch_k=3)
    vector_result = vector.retrieve(RetrievalQuery(text=query, **shallow))
    lexical_result = lexical.search(query, top_k=3)
    print("为什么不能直接相加：两路的分数**量纲不可比**")
    vector_scores = [round(hit.score, 4) for hit in vector_result.hits]
    lexical_scores = [round(hit.score, 4) for hit in lexical_result.hits]
    print(f"  向量路分数：{vector_scores}（余弦，∈ [-1, 1]）")
    print(f"  关键词路分数：{lexical_scores}（BM25，∈ [0, ∞)，没有上界）")
    print(
        "  直接相加时 BM25 会单方面决定名次——把量纲差当成相关性差。"
        "归一化（min-max 到 [0, 1]）把「谁的分数大」换成「谁在这一路里更靠前」。"
    )

    print("\n  min-max 的分段归一化（同一批输入的实测值）：")
    normalized = weighted_score_fusion(
        {CHANNEL_VECTOR: vector_result.hits, CHANNEL_BM25: lexical_result.hits},
        {CHANNEL_VECTOR: 1.0, CHANNEL_BM25: 0.0},
    )
    for hit in normalized:
        print(f"      {hit.record_id:<7} 归一化后 {hit.score:.4f}（权重 1.0 时可直接读出归一化值）")

    print("\n  alpha 扫描（alpha = 向量通道的权重，关键词通道是 1 - alpha）：")
    print(f"      {'alpha':>6} | {'第 1 名':<8} | {'h-c-01':>10} | {'h-t-01':>10} | 解释")
    for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        fused = fuse(
            {CHANNEL_VECTOR: vector_result.hits, CHANNEL_BM25: lexical_result.hits},
            strategy="weighted",
            alpha=alpha,
        )
        scores = {hit.record_id: hit.score for hit in fused}
        reason = "看关键词路" if alpha < 0.5 else ("并列，按 id" if alpha == 0.5 else "看向量路")
        print(
            f"      {alpha:>6} | {fused[0].record_id:<8} | {scores.get('h-c-01', 0.0):>10.4f} | "
            f"{scores.get('h-t-01', 0.0):>10.4f} | {reason}"
        )
    print(
        "\n  读法：两条「各自的第一名」的分数恰好是 alpha 与 1 - alpha"
        "（向量路 top-1 归一化到 1.0，关键词路 top-1 同样归一化到 1.0）。"
        "\n  因此 alpha 越过 0.5 时第一名翻转——权重**确实**在起作用，"
        "而它在 rrf 下根本不存在（RRF 只看名次）。"
    )


# --------------------------------------------------------------------------- #
# 第 6 节：k1 / b 扫描
# --------------------------------------------------------------------------- #


def section_6_bm25_params() -> None:
    """BM25 的两个参数各自的**方向**（在受控语料上，方向不依赖具体数字）."""
    title("第 6 节：BM25 的两个参数（k1 词频饱和 / b 长度归一化）")
    print(f"受控语料：{[(name, text) for name, text in GRADED_DOCS]}")
    print(
        "  heavy 的词频最高、也最长；light 最短。两个参数各自管一件事，"
        "因此**必须分开扫描**。"
    )
    print(f"\n  k1 扫描（b 固定 {DEFAULT_B}）：heavy / light 的分数比")
    print(f"      {'k1':>6} | {'heavy':>10} | {'light':>10} | {'比值':>8} | 说明")
    for k1 in (0.0, 0.5, DEFAULT_K1, 3.0, 10.0):
        index = LexicalIndex.build(
            [LexicalDocument(record_id=name, text=text) for name, text in GRADED_DOCS],
            params=BM25Params(k1=k1, b=DEFAULT_B),
        )
        scores = {hit.record_id: hit.score for hit in index.search("缓存", top_k=10).hits}
        heavy = scores.get("heavy", 0.0)
        light = scores.get("light", 0.0)
        ratio = heavy / light if light else float("nan")
        note = "只看出现过没有" if k1 == 0.0 else "词频收益随 k1 增大"
        print(f"      {k1:>6} | {heavy:>10.6f} | {light:>10.6f} | {ratio:>8.4f} | {note}")
    print(
        "      k1=0 时两条**同分**（词频完全不参与）；k1 越大，"
        "\n      「同一个词出现更多次」的收益越大——这条链是单调的。"
    )

    print(f"\n  b 扫描（k1 固定 {DEFAULT_K1}）：heavy / light 的分数比")
    print(f"      {'b':>6} | {'heavy':>10} | {'light':>10} | {'比值':>8} | 说明")
    for b in (0.0, 0.25, DEFAULT_B, 1.0):
        index = LexicalIndex.build(
            [LexicalDocument(record_id=name, text=text) for name, text in GRADED_DOCS],
            params=BM25Params(k1=DEFAULT_K1, b=b),
        )
        scores = {hit.record_id: hit.score for hit in index.search("缓存", top_k=10).hits}
        heavy = scores.get("heavy", 0.0)
        light = scores.get("light", 0.0)
        ratio = heavy / light if light else float("nan")
        note = "完全不归一化" if b == 0.0 else ("完全归一化" if b == 1.0 else "长文档打个折")
        print(f"      {b:>6} | {heavy:>10.6f} | {light:>10.6f} | {ratio:>8.4f} | {note}")
    print(
        "      b 越大，长文档里的同一个词越不值钱（比值单调下降）——"
        "\n      这一条是 BM25 与「纯词频排序」的分水岭：没有它，长文档永远占优。"
    )
    print(
        "\n  两个参数的缺省（1.5 / 0.75）是 Robertson & Zaragoza 的经典取值，"
        "**不是本课标定出来的**："
        f"\n  settings 里是 retrieval_bm25_k1={settings.retrieval_bm25_k1}、"
        f"retrieval_bm25_b={settings.retrieval_bm25_b}——引用一个公开取值，"
        "\n  别人才分得清「结果不同」是语料造成的还是参数造成的。"
    )


# --------------------------------------------------------------------------- #
# 第 7 节：去重与通道证据
# --------------------------------------------------------------------------- #


def section_7_dedupe() -> None:
    """同一条被两路召回 = **合并证据**（不是"丢掉一条"）."""
    title("第 7 节：去重与 channels 证据（同一条被两路召回是最强的证据）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    hybrid = demo_hybrid(store, embedding, lexical)
    query = QUERY_BOTH
    result = hybrid.retrieve(query)
    print(f"查询 {query!r}（它的词元是词表里唯一的拉丁词，且向量方向恰好也指向同一条）")
    recalled = result.channel_candidates
    print(
        f"  两路召回：vector {recalled[CHANNEL_VECTOR]} 条、"
        f"bm25 {recalled[CHANNEL_BM25]} 条"
    )
    print(f"  融合名单：{result.fusion['fused']} 条；并集 {result.candidates} 条；"
          f"被两路同时命中的 {result.fusion['deduped']} 条")
    print("\n  融合后的前三条（通道与分数一起看）：")
    for hit in result.hits[:3]:
        print(
            f"      #{hit.rank} {hit.record_id:<7} score={hit.score:.6f} "
            f"| channel={hit.channel:<7} | channels={hit.channels} "
            f"| {'、'.join(hit.text.replace(chr(10), ' ')[:24].split())}"
        )
    print(
        f"\n  {result.hits[0].record_id} 的分数是两个 1/61 之和（1/61 = {1.0 / 61.0:.6f}）："
        f"\n  {result.hits[0].score:.6f} ≈ 2 × {1.0 / 61.0:.6f} —— 两路各给了它一个名次 0。"
        "\n  这就是「去重」在这里的真实含义：**不是丢掉一条，而是把两条证据合并到一条记录上**。"
    )
    print(
        "\n  下一名的分数只剩 1/62（只有一路的名次 1）："
        "\n  因此「两路都提到它」比「某一路排第一」更有分量——这是融合给出的新信息，"
        "\n  单路检索永远给不出它（它连「被几路提到」这个概念都没有）。"
    )


# --------------------------------------------------------------------------- #
# 第 8 节：阈值只作用于向量路
# --------------------------------------------------------------------------- #


def section_8_threshold() -> None:
    """``min_score`` 落刀的位置：只对向量通道、只在融合前，而且必须进 notes."""
    title("第 8 节：阈值只作用于向量通道（而且它必须进 notes）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    vector = demo_vector(store, embedding)
    hybrid = demo_hybrid(store, embedding, lexical)
    query = QUERY_EXACT
    print(f"查询 {query!r}：向量路的分数只有 1.0 与 0.0（十条里只有一条与查询同向）")
    vector_result = vector.retrieve(query)
    print(
        f"  向量路（不设阈值时的原始分数）："
        f"{[round(hit.score, 4) for hit in vector_result.hits]}"
    )
    print(f"\n  {'min_score':>10} | {'向量路候选':>10} | {'被阈值切掉':>10} | {'命中':>4} | 名单")
    for min_score in (None, 1.0, 1.5):
        result = hybrid.retrieve(RetrievalQuery(text=query, min_score=min_score))
        print(
            f"  {str(min_score):>10} | {result.channel_candidates[CHANNEL_VECTOR]:>10} | "
            f"{result.dropped_below_threshold:>10} | {result.count:>4} | {ids_of(result)}"
        )
    print(
        "\n  关键一行是 min_score=1.5：它把向量路的 10 条**全部**切掉（余弦最高只有 1.0），"
        "\n  而 h-t-01 照样在名单里——它只有关键词路召回它（那条路的分数是 2.27，不受影响）。"
    )
    result = hybrid.retrieve(RetrievalQuery(text=query, min_score=1.5))
    print(f"  它的通道证据：{[(hit.record_id, hit.channels) for hit in result.hits]}")
    note = next((item for item in result.notes if "只作用于向量通道" in item), "")
    print(f"\n  notes 里那句必须出现的话（**即使一条都没切也要出现**）：\n      {note}")
    print(
        "\n  为什么关键词路不设阈值：BM25 的分数没有绝对标度（同一份文档在两个查询上的"
        "\n  7.31 与 2.27 不可比），给它一个数字是**假的安全感**——"
        "\n  出问题的人会去调它、发现没有稳定效果、于是转去调别的参数。"
        "\n  这与 config.retrieval_min_score 缺省 None 的理由是同一句话："
        "\n  「标定之前给一个数字是假的安全感」。"
    )


# --------------------------------------------------------------------------- #
# 第 9 节：空结果诊断与"过滤只在两路落刀"
# --------------------------------------------------------------------------- #


def section_9_empty_and_filter() -> None:
    """空结果诊断 + 本课最值得讲的那个坑：过滤必须在两路都落刀."""
    title("第 9 节：过滤要在两路都落刀（以及空结果的唯一原因）")
    store = demo_store()
    embedding = TableEmbedding(QUERY_VECTORS)
    lexical = demo_lexical(store)
    vector = demo_vector(store, embedding)
    hybrid = demo_hybrid(store, embedding, lexical)
    query = QUERY_EXACT
    empty_store = FlatVectorStore(metric="cosine", dimension=DEMO_DIMENSION)
    empty_hybrid = demo_hybrid(empty_store, embedding, demo_lexical(empty_store))

    print("空结果只有一个原因（封闭清单里按优先级取第一个成立的）——处置动作完全不同：")
    cases: tuple[tuple[str, Any, RetrievalQuery], ...] = (
        ("库是空的（合法状态）", empty_hybrid, RetrievalQuery(text=query)),
        (
            "过滤把两路候选都筛成 0 条",
            hybrid,
            RetrievalQuery(text=query, where={"strategy": "nope"}),
        ),
        (
            "阈值切光向量路 + 关键词路本来就没召回",
            hybrid,
            RetrievalQuery(text=QUERY_SEMANTIC, min_score=1.5),
        ),
    )
    print(f"\n  {'处境':<34} | {'命中':>4} | {'empty_reason':<16} | 动作")
    reasons: list[str] = []
    for label, engine, retrieval_query in cases:
        result = engine.retrieve(retrieval_query)
        reasons.append(result.empty_reason)
        action = {
            "no_data": "先建索引",
            "filtered_out": "放宽 where / 时间范围",
            "below_threshold": "降低 min_score 或不设阈值",
        }.get(result.empty_reason, "（未见过）")
        print(f"  {label:<34} | {result.count:>4} | {result.empty_reason:<16} | {action}")
    print(
        f"\n  三种成因互不相同：{sorted(set(reasons))} —— 报告里只给一个，"
        "\n  因为它们要调的是不同的参数（索引 / 过滤 / 阈值）。"
    )
    print(
        "\n  第三条值得单独看一眼：阈值**不可能**清空关键词路（那一侧不设阈值），"
        "\n  因此「阈值导致的空结果」在混合模式下必须**同时**满足「关键词路本来就没召回」——"
        "\n  单路检索里那条规则（min_score 切光即是 below_threshold）在这里不再充分。"
        "\n  （第四种成因 diversity_trimmed 在端到端路径上不可达：每组至少留一条，"
        "\n   它靠直接构造诊断输入才能触发——见 retrieval/types.py 的说明。）"
    )

    print("\n  【本课最值得讲的一个坑】过滤只在**一路**落刀会怎样")
    where = {"parent_doc_id": "doc-concept"}
    both = hybrid.retrieve(RetrievalQuery(text=query, where=where))
    lexical_filtered = lexical.search(query, where=where)
    lexical_unfiltered = lexical.search(query)
    print(f"      条件 where={where}（只想看 concept 那份文档）")
    print(
        f"      正确做法（两路都过滤）：向量 {both.channel_candidates[CHANNEL_VECTOR]} 条、"
        f"关键词 {both.channel_candidates[CHANNEL_BM25]} 条 → 名单 {ids_of(both)}"
    )
    print(
        f"      错误做法（只过滤向量路）：向量照旧 "
        f"{vector.retrieve(RetrievalQuery(text=query, where=where)).count} 条，"
        f"而关键词路**不过滤**时会返回 {lexical_unfiltered.count} 条"
        f"（{ids_of(_lexical_view(lexical_unfiltered))}）"
    )
    metadata_by_id = {str(record.record_id): dict(record.metadata) for record in demo_records()}
    leaked = fuse(
        {
            CHANNEL_VECTOR: list(vector.retrieve(RetrievalQuery(text=query, where=where)).hits),
            CHANNEL_BM25: list(lexical_unfiltered.hits),
        }
    )
    leaked_out = [
        (hit.record_id, metadata_by_id[hit.record_id]["parent_doc_id"])
        for hit in leaked
        if metadata_by_id[hit.record_id]["parent_doc_id"] != where["parent_doc_id"]
    ]
    print(
        f"      把这些合起来（用真实的融合函数算一遍）：融合后的名单里出现 "
        f"{len(leaked_out)} 条**不满足过滤条件**的记录 {leaked_out}"
    )
    print(
        "      ——而报告里 filter_applied=True（「过滤生效了」），没有任何异常："
        "\n      **多而不报**，比「少而不报」更难发现（没人会逐条核对命中的 strategy）。"
    )
    print(
        "\n      本层的做法：``where`` 只合成一次（``combine_where``），"
        "\n      同一个子句交给两个调用点——这是**代码结构**上的保证，不是注释里的约定。"
        f"\n      证据（关键词路自己的候选数与命中）：{lexical_filtered.candidates} 条 → "
        f"{lexical_filtered.ids() or '（不含编号的那条）'}"
    )
    manifest = manifest_from_store(
        store,
        identity=EmbeddingIdentity(
            provider="TableEmbedding", model="table-v1", dimension=DEMO_DIMENSION
        ),
        metric=store.metric,
    )
    print(
        f"\n  顺带：本节的索引清单是**现算**的（版本 {manifest.version_id[:16]}…），"
        "\n  它不落盘——仓库里不会多出一个看起来像「线上索引」的目录。"
    )


# --------------------------------------------------------------------------- #
# 输出：同时写终端与文件
# --------------------------------------------------------------------------- #


class _Tee:
    """把写往 stdout 的内容**同时**送到终端与文件（模块 docstring 的硬要求）."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> int:
    """按节运行演示，并把完整输出同时写入 ``outputs/hybrid_demo.txt``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    original = sys.stdout
    with output_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            for section in (
                section_1_corpus,
                section_2_topk,
                section_3_complementary,
                section_4_rrf_handcheck,
                section_5_weighted,
                section_6_bm25_params,
                section_7_dedupe,
                section_8_threshold,
                section_9_empty_and_filter,
            ):
                section()
            print(f"\n演示完成（工作目录 {Path.cwd()}）")
            print(f"本节输出已同时写入 {output_path}")
            print("九节全部离线：零网络、零新增依赖；两路的分数、融合与手算都是真算出来的。")
        finally:
            sys.stdout = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
