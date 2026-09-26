#!/usr/bin/env python
"""day066 演示脚本：检索器与 RAG 生成链路（M6-D5）.

八节，全部**离线、确定性、零网络**，只依赖本包与标准库：

```text
1. 建一个小库       8 条记录 / 3 份文档 / day062 的元数据形状（parent_doc_id 等）
2. Top-K 与深度     fetch_k = max(top_k, ceil(top_k × 倍率))、封顶 MAX_FETCH_K、两次调用方错误
3. where 过滤       strategy / heading_path / $and 各一次，以及"字段名写错"的诊断
4. 时间范围         闭区间按 ISO 文本比较；缺 created_at 的记录被排除的对照
5. 阈值             min_score 在本层落刀：切掉了几条 vs 全切开（below_threshold）
6. 多样性           max_per_doc=1 前后：命中条数没变，文档分布变了
7. 索引漂移         手工删一条 → drift 非空 + notes；strict_index=True → 拒绝服务
8. RAG 生成链路     命中（带 [1] 的答案）/ 收紧预算的降级注记 / 未命中时 llm_called=False
```

本层最重要的那句边界（教程会引用它）：

    向量库回答'给一个向量，谁最近'；检索器回答'给一句话，取回哪几条、
    为什么是这几条、为什么只有这几条'——第 7 节与第 8 节是后半句的兑现。

## 演示样本与测试样本刻意分开

``tests/`` 里也有一份小库，但**脚本不 import tests**：脚本要能被别人复制走
单独运行，一旦依赖测试目录，复制到别处就跑不起来。因此这里自带一份
8 条记录的样本，编码器用 ``CharNgramEmbedding`` 的确定性派生类
（离线、逐位可复现），因此本脚本打印的每一个分数都是**真算出来的**。

样本刻意留了两个"不整齐"的地方，因为它们是这一课要讲的东西：

```text
两条记录没有 created_at   让第 4 节能演示"字段缺失被排除"（不是 bug）
三份文档块数不均是 4/3/1   让第 6 节的多样性裁剪一定会发生（不是特例）
```

## 清单是现算的，不落盘

第 7 节需要一份"上一版索引"来对照，脚本用 day065 的
``entry_from_record`` + ``build_manifest`` **现算**一份清单：
它不进 ``data/index``，因此不会在仓库里留下一个看起来像"线上索引"的目录。
脚本结果同时打印到 stdout 并写入 ``outputs/retrieval_demo.txt``。

运行（cwd 为 ``day066/源码/smart-research-agent``）::

    python scripts/retrieval_demo.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# 本脚本要能直接 ``python scripts/retrieval_demo.py`` 跑起来（cwd 是仓库根）。
# 以脚本方式启动时 ``sys.path[0]`` 是 ``scripts/``，仓库根不在其中，
# 而本快照的包**没有 pip 安装**（``pip show smart-research-agent`` 为空）——
# 因此这里显式把仓库根塞进 ``sys.path``，让脚本不依赖 ``PYTHONPATH=.``。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.documents.types import content_id  # noqa: E402
from smart_research_agent.indexing import (  # noqa: E402
    IndexManifest,
    build_manifest,
    describe_embedding,
    entry_from_record,
)
from smart_research_agent.llm.embedding import CharNgramEmbedding, EmbeddingProvider  # noqa: E402
from smart_research_agent.llm.mock import MockLLM  # noqa: E402
from smart_research_agent.retrieval import (  # noqa: E402
    MAX_FETCH_K,
    RAG_ANSWER_PROMPT_VERSION,
    TRUNCATION_MARKER,
    ContextError,
    IndexStateError,
    QueryError,
    RagPipeline,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    TimeRange,
    build_retriever,
    describe_conditions,
    pack_context,
)
from smart_research_agent.vectorstore import FlatVectorStore  # noqa: E402
from smart_research_agent.vectorstore.pipeline import (  # noqa: E402
    embedding_text,
    record_from_knowledge,
)
from smart_research_agent.vectorstore.types import VectorRecord  # noqa: E402

#: 本层的边界（原文供教程引用，见模块 docstring）.
LAYER_BOUNDARY = (
    "向量库回答'给一个向量，谁最近'；检索器回答'给一句话，取回哪几条、"
    "为什么是这几条、为什么只有这几条'。"
)

#: 输出目录与文件名（``outputs/`` 在仓库 ``.gitignore`` 里，不污染仓库）.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "retrieval_demo.txt"

#: 样本维度。取 256 而不是 8：``CharNgramEmbedding`` 的真实默认维度，
#: 本脚本不依赖维度手算期望值（分数是算出来的，不是写死的）。
DEMO_DIMENSION = 256

#: 三份文档：``parent_doc_id → 来源路径``（day062 的 ``source`` 就是这个形状）.
DEMO_SOURCES: dict[str, str] = {
    "9f2c47b3a05e1d86": "docs/retrieval.md",
    "7b1e5c90d43a826f": "docs/embedding_index.md",
    "c05a3d18f7b2946e": "docs/vector_store.md",
}

#: 八条块：``(chunk_id, parent_doc_id, strategy, heading_path, text, created_at)``.
#:
#: ``created_at`` 为 ``None`` 的两条是刻意的（见模块 docstring）：
#: 它们让第 4 节能演示"时间过滤把缺字段的记录排除掉"这条约定。
DEMO_CHUNKS: tuple[tuple[str, str, str, str, str, str | None], ...] = (
    (
        "3a7c1e95b4d2086f",
        "9f2c47b3a05e1d86",
        "recursive",
        "检索手册 > Top-K 与深度",
        "检索深度必须大于要返回的条数：过取倍率 3 是三道减法（阈值、多样性、截断）的下界。",
        "2026-03-02T09:15:00",
    ),
    (
        "c41d8b27f6e0359a",
        "9f2c47b3a05e1d86",
        "structural",
        "检索手册 > 阈值",
        "阈值在本层落刀，切掉了几条这个数字必须能被报告出来，否则融合判断不了哪一路被清零。",
        "2026-03-02T09:20:00",
    ),
    (
        "5e90a36c1b7f4d28",
        "9f2c47b3a05e1d86",
        "structural",
        "检索手册 > 阈值",
        "阈值用同一个大于等于号，与库侧的口径完全一致，而且只落一次刀。",
        "2026-03-02T09:25:00",
    ),
    (
        "b28f5d04a9c761e3",
        "9f2c47b3a05e1d86",
        "fixed",
        "检索手册 > 过滤",
        "过滤条件里的字段名写错不会报错，只会让候选变成 0 条，所以只能靠诊断说出来。",
        "2026-04-11T14:05:00",
    ),
    (
        "d6a03f81c257b94e",
        "7b1e5c90d43a826f",
        "structural",
        "索引手册 > 清单漂移",
        "清单里有、库里没有，是上一次构建半途失败最典型的样子，请先重建清单。",
        "2026-04-11T14:40:00",
    ),
    (
        "0f47c92b5d18e3a6",
        "7b1e5c90d43a826f",
        "recursive",
        "索引手册 > 清单漂移",
        "漂移只观测不阻断：检索仍然能跑，只是可能少召回那几条被删掉的记录。",
        "2026-04-12T08:00:00",
    ),
    (
        "e9352d70b84a1c6f",
        "7b1e5c90d43a826f",
        "semantic",
        "索引手册 > 版本号",
        "版本号由编码器身份、度量、后端与内容摘要算出，时间戳不进版本号。",
        None,
    ),
    (
        "1c8b4a95e370d26f",
        "c05a3d18f7b2946e",
        "fixed",
        "缓存手册 > 键",
        "缓存键里必须含编码器身份，否则换模型之后会命中旧模型的向量。",
        None,
    ),
)

#: 查询文本。它同时含"阈值""过滤""检索"三个词，因此既有明确的第一名，
#: 又会让同一份文档占据多个名次（第 6 节的多样性裁剪才有东西可裁）。
QUERY = "检索的阈值与过滤条件怎么落刀"

#: 第 7 节被删掉的那一条（它的 id 只在这里出现一次）。
DRIFT_VICTIM = DEMO_CHUNKS[5][0]

#: MockLLM 的固定回复：句子里带 [1] [2]，用来验证"引用编号真的进了提示词"。
MOCK_ANSWER = (
    "阈值在本层落刀，切掉了几条这个数字要能被报告出来 [1]；"
    "同一个大于等于号只落一次刀 [2]。资料中没有说阈值该取多少。"
)

#: 预算收紧之后 MockLLM 的回复：只提到 [1] [2]，因为更靠后的片段已经被丢掉了。
MOCK_ANSWER_TIGHT = "按 [1] 与 [2] 回答：阈值在本层落刀，且只落一次。后面几条片段没看到。"


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# 样本构造：八条记录、一个确定性编码器、一份现算的清单
# --------------------------------------------------------------------------- #


class DemoEmbedding(CharNgramEmbedding):
    """确定性字符 n-gram 编码器（离线、逐位可复现），并记下被调用的次数.

    为什么**不**把向量写死成查表：这一课要打印的是分数与命中关系，
    而写死的向量只证明"我按表查了一下"。``CharNgramEmbedding`` 至少让
    "字面重叠多的文本向量更近"这件事是真的——第 2~6 节的排序因此是**算出来的**。

    ``embed_calls`` 是本脚本自己的台账：第 8 节能用它证明
    "检索为空时一次编码都没发生"（编码器与 LLM 都不该被叫醒）。
    """

    def __init__(self, dimension: int = DEMO_DIMENSION) -> None:
        super().__init__(dimension=dimension)
        self.embed_calls = 0

    def embed(self, text: str) -> list[float]:
        """逐条编码，并记一次调用."""
        self.embed_calls += 1
        return super().embed(text)


def knowledge_records() -> list[dict[str, Any]]:
    """八条 day062 ``knowledge_records()`` 形状的记录（四个键）.

    形状与 day062 的产物逐键一致（``doc_id`` / ``source`` / ``text`` /
    ``metadata``），因为它是本层的**真实**输入：用别的形状造样本，
    "两个模块能接上"这件事就永远没被验证过。四个派生字段各有出处：

    ```text
    metadata["retrieval_text"]   面包屑 + 正文  → 被编码的那份文本
    metadata["fingerprint"]      content_id(原文) → 内容身份（清单用它）
    metadata["parent_doc_id"]    day062 的文档归属 → 多样性裁剪按它分组
    metadata["tags"]             数组型元数据       → $contains 只对数组有意义
    ```

    落库时走的是 ``vectorstore.pipeline.record_from_knowledge``（不是 ``make_record``）：
    它会把记录层的 ``source`` 折进 metadata（``VectorRecord`` 没有独立的来源字段），
    因此第 8 节的引用才能打出"docs/retrieval.md › 检索手册 > 阈值"这样的出处。
    """
    records: list[dict[str, Any]] = []
    for index, (chunk_id, parent, strategy, heading, text, created_at) in enumerate(DEMO_CHUNKS):
        metadata: dict[str, Any] = {
            "parent_doc_id": parent,
            "strategy": strategy,
            "heading_path": heading,
            "index": index,
            "token_count": len(text),
            "tags": ["guide", strategy],
            "fingerprint": content_id(text),
            "retrieval_text": f"{heading}\n{text}",
        }
        if created_at is not None:
            metadata["created_at"] = created_at
        records.append(
            {
                "doc_id": chunk_id,
                "source": DEMO_SOURCES[parent],
                "text": text,
                "metadata": metadata,
            }
        )
    return records


def demo_records(embedding: EmbeddingProvider | None = None) -> list[VectorRecord]:
    """八条 ``VectorRecord``（向量由确定性编码器算出，不是写死的）.

    向量由 ``embedding_text``（= ``retrieval_text``）算出，落库的 ``text``
    却是原文——这正是 day062 留下的那条分工，本脚本不改写它。
    """
    embedder = embedding or DemoEmbedding()
    return [
        record_from_knowledge(record, embedder.embed(embedding_text(record)), metric="cosine")
        for record in knowledge_records()
    ]


def build_demo_store() -> FlatVectorStore:
    """建一个小库（8 条，cosine）——每节都从**全新的库**开始.

    逐节重建而不是共用一份：第 7 节会删记录、第 8 节会改预算，
    共用一个库会让"这一节看到的数字"取决于上一节做过什么。
    """
    store = FlatVectorStore(metric="cosine")
    store.upsert(demo_records())
    return store


def build_demo_manifest() -> IndexManifest:
    """按 day065 的方式**现算**一份清单（不落盘，见模块 docstring）.

    清单派两个用场：检索器用它报告"这次查的是哪一版索引"，
    以及用它发现漂移（第 7 节）。
    """
    identity = describe_embedding(DemoEmbedding())
    entries = [entry_from_record(record, identity) for record in knowledge_records()]
    return build_manifest(
        entries,
        identity=identity,
        metric="cosine",
        backend="flat",
        created_at="",
    )


def demo_embedding() -> DemoEmbedding:
    """一个新的确定性编码器（每节一个，调用次数才有意义）."""
    return DemoEmbedding()


def _source_counts(result: RetrievalResult) -> dict[str, int]:
    """命中按**来源**计数（``metadata["source"]``，缺失回落 ``record_id``）.

    与 ``result.doc_ids`` 分工不同：``doc_ids`` 回答"命中共来自几份文档"，
    这个计数回答"每份文档各占几条"——多样性裁剪的效果只有后者看得出来。
    """
    counts: dict[str, int] = {}
    for hit in result.hits:
        source = str(hit.metadata.get("source") or hit.record_id)
        counts[source] = counts.get(source, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 1. 建一个小库
# --------------------------------------------------------------------------- #


def section_1_corpus() -> None:
    """第 1 节：库里有什么、清单是什么版本、被编码的是哪份文本."""
    title("1. 建一个小库：8 条记录 / 3 份文档 / 256 维（cosine）")
    print("  本层的边界（原文，供教程引用）：")
    print(f"    {LAYER_BOUNDARY}")
    print()
    store = build_demo_store()
    print(f"  库状态：{store.info().summary_line()}")
    print()
    print(f"  {'chunk_id':<18}{'文档':<24}{'策略':<12}{'标题':<22}{'字数':>5}  创建时间")
    for record in sorted(store.get_many(store.ids()), key=lambda item: item.record_id):
        metadata = record.metadata
        created = str(metadata.get("created_at", "（缺失）"))
        print(
            f"  {record.record_id:<18}"
            f"{DEMO_SOURCES[str(metadata['parent_doc_id'])]:<24}"
            f"{str(metadata['strategy']):<12}"
            f"{str(metadata['heading_path']):<22}"
            f"{record.char_count:>5}  {created}"
        )
    print()
    chunks_per_doc: dict[str, int] = {}
    for record in store.get_many(store.ids()):
        source = DEMO_SOURCES[str(record.metadata["parent_doc_id"])]
        chunks_per_doc[source] = chunks_per_doc.get(source, 0) + 1
    print(f"  三份文档的块数（第 6 节的多样性裁剪要用的分布）：{chunks_per_doc}")
    without_time = [
        record.record_id
        for record in store.get_many(store.ids())
        if "created_at" not in record.metadata
    ]
    print(f"  没有 created_at 的记录：{without_time}（第 4 节的对照就是它们）")
    print()
    manifest = build_demo_manifest()
    print(f"  清单（day065 的 entry_from_record + build_manifest 现算）：{manifest.summary_line()}")
    print(f"    索引版本 version_id = {manifest.version_id}")
    print("    这个版本号回答了'我查的是哪一版索引'——第 7 节的漂移判定就基于它。")
    print()
    sample = knowledge_records()[1]
    print("  被编码的文本**不是原文**，而是 retrieval_text（面包屑 + 正文）：")
    print(f"    {embedding_text(sample)!r}")
    print(f"  原文（落库、给人看的那份）：{sample['text']!r}")
    print("  两者分开的理由：检索要的是'哪一节里的这句话'，展示要的只是那句话。")


# --------------------------------------------------------------------------- #
# 2. Top-K 与深度
# --------------------------------------------------------------------------- #


def section_2_depth() -> None:
    """第 2 节：top_k / fetch_k / 封顶，以及两次当场报错的调用方参数."""
    title("2. Top-K 与深度：fetch_k = max(top_k, ceil(top_k × 倍率))，封顶 1000")
    store = build_demo_store()
    retriever = build_retriever(store, demo_embedding())
    print(
        f"  构造参数缺省时读 settings：retrieval_top_k={settings.retrieval_top_k}、"
        f"retrieval_fetch_multiplier={settings.retrieval_fetch_multiplier}、"
        f"MAX_FETCH_K={MAX_FETCH_K}"
    )
    described = retriever.describe()
    print(
        f"  describe()：name={described['name']!r} top_k={described['top_k']} "
        f"fetch_multiplier={described['fetch_multiplier']} min_score={described['min_score']} "
        f"max_per_doc={described['max_per_doc']}"
    )
    print()
    print(f"  查询：{QUERY!r}")
    print(f"  {'top_k':>6} | {'fetch_k':>7} | {'候选':>4} | {'命中':>4} | {'top_k 截断丢掉':>14}")
    for top_k in (3, 5, 8, MAX_FETCH_K):
        result = retriever.retrieve(RetrievalQuery(text=QUERY, top_k=top_k))
        print(
            f"  {top_k:>6} | {result.fetch_k:>7} | {result.candidates:>4} | "
            f"{result.count:>4} | {result.dropped_by_top_k:>14}"
        )
        for note in result.notes:
            print(f"           注记：{note}")
    print()
    print("  四行的读法：top_k 决定'最终返回几条'（3 / 5 / 8 / 8），fetch_k 决定'取多深'")
    print("  （9 / 15 / 24 / 1000）——深度永远 >= 条数，这正是三道减法能有东西可减的前提。")
    print(f"  最后一行是本节的封顶：库只有 {store.count()} 条，深度却被封在 {MAX_FETCH_K} 上——")
    print("  封顶防的是'一次把整库搬进响应体'（与库规模无关），因此它必须有一个绝对上限。")
    print()
    print("  显式 fetch_k 小于默认 top_k 时，检索器不报错而是**收敛**（并记进 notes）：")
    converged = retriever.retrieve(RetrievalQuery(text=QUERY, fetch_k=2))
    print(f"    请求 fetch_k=2 → 本次 top_k={converged.query.top_k}、fetch_k={converged.fetch_k}")
    for note in converged.notes:
        print(f"      注记：{note}")
    print()
    print("  而**同一个请求里**把两者写反了，就是调用方的错误，当场报 QueryError：")
    try:
        RetrievalQuery(text=QUERY, top_k=5, fetch_k=2)
    except QueryError as exc:
        print(f"    {exc}")
    try:
        RetrievalQuery(text="   ")
    except QueryError as exc:
        print(f"    空查询：{exc}")


# --------------------------------------------------------------------------- #
# 3. where 过滤
# --------------------------------------------------------------------------- #


def section_3_filters() -> None:
    """第 3 节：三种 where + 一次"字段名写错"的诊断."""
    title("3. where 过滤：候选数、命中数与'少而不报'的诊断")
    store = build_demo_store()
    retriever = Retriever(store, demo_embedding())
    cases: tuple[tuple[str, dict[str, Any]], ...] = (
        ("按策略 strategy=structural", {"strategy": "structural"}),
        ("按数组元数据 tags 含 'structural'", {"tags": {"$contains": "structural"}}),
        ("按标题 heading_path 等值", {"heading_path": "索引手册 > 清单漂移"}),
        ("组合 $and（structural 且 index=2）",
         {"$and": [{"strategy": "structural"}, {"index": 2}]}),
        ("$contains 用在字符串字段上", {"heading_path": {"$contains": "检索手册"}}),
        ("筛空（库里没有的策略 bm25）", {"strategy": {"$in": ["bm25"]}}),
        ("字段名写错 stratgey", {"stratgey": "structural"}),
    )
    for label, where in cases:
        result = retriever.retrieve(RetrievalQuery(text=QUERY, where=where, top_k=5))
        print(f"  [{label}] where = {describe_conditions(where, None)}")
        print(
            f"    candidates={result.candidates} count={result.count} "
            f"filter_applied={result.filter_applied} empty_reason={result.empty_reason}"
        )
        print(f"    ids={result.ids()}")
        if result.is_empty and result.notes:
            print(f"    注记：{result.notes[-1]}")
        print()
    print("  第五种情况是'运算符用对了、字段类型不对'：$contains 只对**数组**元数据有意义")
    print("  （vectorstore 的既定语义：字段不是数组就判为不命中），用在字符串字段上会被判没命中。")
    print("  它同样不报错，而 explain() 的字段拼写检查也救不了它——字段名本身是对的。")
    print()
    print("  最后一种情况是本课最想消灭的那类失败：**字段名写错不报错**，")
    print("  过滤之后候选变成 0 条，而报告里只有'没命中'这一个事实。")
    print("  检索器的 explain() 会把它指出来（它要读一次库，因此只能由检索器给）：")
    typo = retriever.retrieve(RetrievalQuery(text=QUERY, where={"stratgey": "structural"}))
    for line in retriever.explain(typo):
        print(f"    {line}")


# --------------------------------------------------------------------------- #
# 4. 时间范围
# --------------------------------------------------------------------------- #


def section_4_time_range() -> None:
    """第 4 节：闭区间 + 缺字段被排除 + 与 where 撞车时当场报错."""
    title("4. 时间范围：闭区间按 ISO 文本比较，缺字段的记录被排除")
    store = build_demo_store()
    retriever = Retriever(store, demo_embedding())
    print(f"  库里 {store.count()} 条；其中 2 条没有 created_at（见第 1 节）。")
    print()
    ranges: tuple[TimeRange, ...] = (
        TimeRange(start="2026-03-01", end="2026-03-31"),
        TimeRange(start="2026-04-01", end="2026-04-30"),
        TimeRange(start="2026-01-01", end="2026-12-31"),
        TimeRange(start="2026-04-12"),
    )
    print(f"  {'时间范围':<42}{'候选':>4} | 命中的 id")
    for time_range in ranges:
        result = retriever.retrieve(
            RetrievalQuery(text=QUERY, time_range=time_range, top_k=store.count())
        )
        print(f"  {time_range.describe():<42}{result.candidates:>4} | {result.ids()}")
    print("  （这里的 top_k 取 8 = 库的大小，因此打印出来的 id 就是**全部候选**）")
    print()
    print("  第三条（整年区间）是本节的对照：区间覆盖了全部数据，候选却只有 6 条——")
    print("  缺 created_at 的那两条**既不在区间内、也不在区间外**，它们被排除了。")
    print("  $gte 对'字段不存在'返回 False，这是 vectorstore 的既定语义：")
    print("  宁可少召回，也不替一条记录猜一个创建时间（猜错不会报错，只会少几条）。")
    print()
    print("  与 where 撞在同一个字段时当场报 QueryError（不许替调用方取交集）：")
    try:
        retriever.retrieve(
            RetrievalQuery(
                text=QUERY,
                where={"created_at": {"$gte": "2026-04-01"}},
                time_range=TimeRange(start="2026-03-01"),
            )
        )
    except QueryError as exc:
        print(f"    {exc}")


# --------------------------------------------------------------------------- #
# 5. 阈值
# --------------------------------------------------------------------------- #


def section_5_threshold() -> None:
    """第 5 节：阈值在本层落刀，'切掉了几条'是它唯一的产出."""
    title("5. 阈值：切掉了几条 vs 全切开（below_threshold）")
    store = build_demo_store()
    retriever = Retriever(store, demo_embedding())
    baseline = retriever.retrieve(RetrievalQuery(text=QUERY, top_k=5))
    print(f"  不设阈值（min_score=None 表示**不设**，不是 0）：{baseline.summary_line()}")
    for hit in baseline.hits:
        print(
            f"    #{hit.rank} {hit.record_id} score={hit.score:+.6f} "
            f"doc={hit.doc_id} 标题={hit.heading_path!r}"
        )
    pivot = baseline.hits[2].score
    print()
    print(f"  取第 3 名的分数当阈值（min_score={pivot:.6f}，>= 才保留）：")
    cut = retriever.retrieve(RetrievalQuery(text=QUERY, top_k=5, min_score=pivot))
    print(f"    {cut.summary_line()}")
    print(
        f"    dropped_below_threshold={cut.dropped_below_threshold} "
        f"count={cut.count} → 阈值切掉了 {cut.dropped_below_threshold} 条"
    )
    print(
        f"    这笔账要对上：候选 {cut.candidates} 条 - 阈值切掉 "
        f"{cut.dropped_below_threshold} 条 = 剩下 {cut.count} 条"
    )
    print("    （注意 8 是**召回深度取回来的候选**，不是要返回的条数：")
    print("      不设阈值时那 8 条里有 5 条活到 top_k，另外 3 条被 top_k 截断丢掉。）")
    print()
    print("  阈值高到一条不剩（余弦的分数上界是 1.0，这里给 1.5）：")
    empty = retriever.retrieve(RetrievalQuery(text=QUERY, top_k=5, min_score=1.5))
    print(f"    {empty.summary_line()}")
    print(
        f"    candidates={empty.candidates}（候选还在）count={empty.count} "
        f"empty_reason={empty.empty_reason}"
    )
    print("    候选非 0 而命中为 0 → 数据在，只是不够像；原因被定成 below_threshold。")
    print()
    print("  为什么阈值必须在本层落刀：交给库侧过滤时，'切掉了几条'这个数字拿不到——")
    print("  而 day067 的融合正是靠它判断'这一路是不是一条都没活下来'。")


# --------------------------------------------------------------------------- #
# 6. 多样性
# --------------------------------------------------------------------------- #


def section_6_diversity() -> None:
    """第 6 节：max_per_doc 取三个值时，命中条数与文档分布各自的变化."""
    title("6. 多样性：调的不是'取几条'，而是'这 5 条从哪来'")
    store = build_demo_store()
    retriever = Retriever(store, demo_embedding())
    wide = retriever.retrieve(RetrievalQuery(text=QUERY, top_k=store.count()))
    distribution: dict[str, int] = {}
    for hit in wide.hits:
        distribution[hit.doc_id] = distribution.get(hit.doc_id, 0) + 1
    print(f"  settings.retrieval_max_per_doc = {settings.retrieval_max_per_doc}（0 = 不限）")
    print(f"  召回深度取回来的那批候选按文档分布：{distribution}")
    print(f"    → 名次靠前的这批里只有 {len(distribution)} 份文档，因此 max_per_doc=1 "
          f"至多凑出 {len(distribution)} 条。")
    print()
    rows: tuple[tuple[str, RetrievalResult], ...] = (
        ("不限（max_per_doc=0）", retriever.retrieve(RetrievalQuery(text=QUERY, top_k=5))),
        ("max_per_doc=2", retriever.retrieve(
            RetrievalQuery(text=QUERY, top_k=5, max_per_doc=2))),
        ("max_per_doc=1", retriever.retrieve(
            RetrievalQuery(text=QUERY, top_k=5, max_per_doc=1))),
    )
    for label, result in rows:
        print(f"  [{label}] {result.summary_line()}")
        print(f"    ids={result.ids()}")
        print(
            f"    doc_ids={result.doc_ids}（{len(result.doc_ids)} 份文档）"
            f" | 按来源计数={_source_counts(result)}"
        )
        print(f"    dropped_by_diversity={result.dropped_by_diversity}")
        print()
    print("  三行的 top_k 上限都是 5，实际条数却是 5 / 5 / 3：")
    print("    max_per_doc=2 → 同文档最多留两条，被挤掉的名额由**别的文档**补上了；")
    print("    max_per_doc=1 → 只能凑出 3 条，因为候选里一共只有 3 份文档，没有第四份可补。")
    print("  因此多样性**不改变要取几条**（top_k 还是 5），但它可能让实际条数少于 top_k；")
    print("  而 dropped_by_diversity 这个数字让人能算出'挤掉了多少、补回来多少'。")
    print("  分组的键是 metadata['parent_doc_id']，不是 record_id：")
    print("  record_id 永远唯一，用它分组等于没分组（治不了'同一份文档刷屏'）。")


# --------------------------------------------------------------------------- #
# 7. 索引漂移
# --------------------------------------------------------------------------- #


def section_7_drift() -> None:
    """第 7 节：手工删一条 → drift 非空；strict_index=True → 拒绝服务."""
    title("7. 索引漂移：清单与库互相矛盾时，它必须是可观测的")
    store = build_demo_store()
    manifest = build_demo_manifest()
    retriever = Retriever(store, demo_embedding(), manifest=manifest)
    print(f"  带清单的检索器：{manifest.summary_line()}")
    state = retriever.index_state
    print(f"  index_state：{state.summary_line()}")
    print(f"    version_id={state.version_id!r} has_drift={state.has_drift}")
    print()
    removed = store.delete(ids=[DRIFT_VICTIM])
    print(
        f"  现在绕过索引流程删掉一条（id={DRIFT_VICTIM!r}）："
        f"delete 返回**真正删掉**的条数 = {removed}，库剩 {store.count()} 条"
    )
    result = retriever.retrieve(QUERY)
    print(f"  再检索一次：{result.summary_line()}")
    print(f"    命中 {result.count} 条——**检索照常返回**，没有抛异常、没有空结果。")
    print(f"    index_state.drift 共 {len(result.index_state.drift)} 项：")
    for problem in result.index_state.drift:
        print(f"      {problem}")
    for note in result.notes:
        print(f"    注记：{note}")
    print()
    print("  漂移的处置有两条路，脚本把两条都走一遍：")
    strict = Retriever(store, demo_embedding(), manifest=manifest, strict_index=True)
    try:
        strict.retrieve(QUERY)
    except IndexStateError as exc:
        print("    strict_index=True → 拒绝服务，完整消息：")
        for line in str(exc).splitlines():
            print(f"      {line}")
    else:
        print("    （本机没有报错，说明漂移没被检出——这是不对的）")
    print("    默认（strict_index=False）→ 服务照常，但漂移进 result.index_state 与 notes。")
    print("  两种处置对应两种取舍：宁可少召回（默认）还是宁可不可用（严格）。")
    print("  唯一不可接受的是**静默降级**：那会让'这次少召回'看起来像'库里就只有这些'。")


# --------------------------------------------------------------------------- #
# 8. RAG 生成链路
# --------------------------------------------------------------------------- #


def section_8_rag() -> None:
    """第 8 节：命中路径 / 收紧预算的降级 / 未命中时一次 LLM 都不调."""
    title("8. RAG 生成链路：检索 → 打包 → 生成 → 引用（含空结果护栏）")
    store = build_demo_store()
    embedding = demo_embedding()
    retriever = Retriever(store, embedding, manifest=build_demo_manifest())
    scripted = [MOCK_ANSWER, MOCK_ANSWER_TIGHT, MOCK_ANSWER_TIGHT, MOCK_ANSWER_TIGHT]
    llm = MockLLM(responses=list(scripted))
    pipeline = RagPipeline(retriever, llm)
    print(f"  提示词版本：{RAG_ANSWER_PROMPT_VERSION}（模板占位符 {{context}} / {{question}}）")
    print(f"  describe()：{pipeline.describe()}")
    print(f"  MockLLM 按脚本逐条弹出回复（用尽后回落到默认串），本节备了 {len(scripted)} 条脚本。")
    print()

    # 8.1 命中路径
    answer = pipeline.answer(QUERY)
    print("  [8.1 命中] 检索有结果 → 打包 → 渲染提示词 → 调用 LLM")
    print(f"    {answer.summary_line()}")
    print(f"    llm_called={answer.llm_called} | LLM 被调用次数={len(llm.calls)}")
    context = answer.context
    assert context is not None  # 命中路径必然有上下文（这里给类型检查器看）
    print(f"    {context.summary_line()}")
    print(f"    引用编号：{[citation.marker for citation in answer.citations]}")
    for citation in answer.citations:
        print(f"      {citation.summary_line()}")
    print(f"    marker_for({answer.citations[0].record_id!r}) = "
          f"{context.marker_for(answer.citations[0].record_id)}")
    print(f"    marker_for(不在上下文里的 id) = {context.marker_for('nope')}（None，不是异常）")
    print("    模型看到的资料（= context.text，形状是 [n] 来源 › 标题路径 + 正文）：")
    for line in context.text.splitlines():
        print(f"      {line}")
    prompt = llm.calls[0][0].content
    used_markers = re.findall(r"\[(\d+)\]", answer.answer)
    print(f"    提示词 {len(prompt)} 字 | 含 [1]：{'[1]' in prompt} | "
          f"含问题：{QUERY in prompt} | 含'不要编造'类约束：{'不要编造' in prompt}")
    print(f"    答案用到的编号={used_markers}，最大合法编号={len(answer.citations)} → "
          f"全部可用：{all(int(item) <= len(answer.citations) for item in used_markers)}")
    print(f"    notes：{list(answer.notes)}")
    print()

    # 8.2 收紧预算
    hits = answer.retrieval.hits if answer.retrieval is not None else ()
    print("  [8.2 降级] 把预算压到 max_context_chars=220 / per_hit_chars=60：")
    packed = pack_context(hits, max_chars=220, per_hit_chars=60)
    print(f"    直接调用 pack_context → {packed.summary_line()}")
    print(
        f"      进上下文 {len(packed.used_hits)} 条 / 丢掉 {len(packed.dropped_hits)} 条 / "
        f"截断 {packed.truncated_hits} 条"
    )
    markers = "、".join(f"[{citation.marker}]" for citation in packed.citations)
    blocks = packed.text.split("\n\n")
    numbered = all(block.startswith(f"[{index}]") for index, block in enumerate(blocks, 1))
    print(f"      进上下文的编号：{markers}（与文本里的 [n] 一一对应：{numbered}）")
    print(f"      最后一块的结尾（截断标记 {TRUNCATION_MARKER!r} 就在这里）：{blocks[-1][-16:]!r}")
    squeezed = pack_context(hits, max_chars=60, per_hit_chars=60)
    print(
        f"      预算压到 60 字（放不下两条）→ {squeezed.summary_line()}："
        "**至少保住一条**，返回的不是空上下文"
    )
    try:
        pack_context(hits, max_chars=20)
    except ContextError as exc:
        print("      预算 20 字（连单条下限 32 字都放不下）→ ContextError：")
        for line in str(exc).splitlines():
            print(f"        {line}")
    print()
    tight = RagPipeline(retriever, llm, max_context_chars=220, per_hit_chars=60)
    tight_answer = tight.answer(QUERY)
    tight_context = tight_answer.context
    assert tight_context is not None
    print(f"    经过 pipeline：{tight_answer.summary_line()}")
    print(f"      {tight_context.summary_line()}")
    print(f"      进上下文的 id={[hit.record_id for hit in tight_context.used_hits]}")
    print(f"      被丢掉的 id={[hit.record_id for hit in tight_context.dropped_hits]}")
    print("    降级注记（每种降级的处置各不相同，因此各是一句话）：")
    for note in tight_answer.notes:
        print(f"      {note}")
    overrides = tight.answer(QUERY, max_context_chars=600)
    wide = overrides.context
    dropped = len(wide.dropped_hits) if wide is not None else -1
    truncated = wide.truncated_hits if wide is not None else -1
    print(
        f"    逐次覆盖 max_context_chars=600（单块上限仍是 60）→ "
        f"丢掉 {dropped} 条、截断 {truncated} 条：预算一放宽，丢尾就消失了"
    )
    try:
        tight.answer(QUERY, max_context_char=600)
    except QueryError as exc:
        print(f"    覆盖参数拼错时当场报错：{exc}")
    print()

    # 8.3 漂移注记
    drift_store = build_demo_store()
    drift_store.delete(ids=[DRIFT_VICTIM])
    drift_pipeline = RagPipeline(
        Retriever(drift_store, demo_embedding(), manifest=build_demo_manifest()),
        llm,
    )
    drift_answer = drift_pipeline.answer(QUERY)
    print("  [8.3 漂移] 清单没变、库被人删过一条（第 7 节的那种情形，这次走完整链路）：")
    print(f"    {drift_answer.summary_line()}")
    print(f"    llm_called={drift_answer.llm_called}（照常调用——漂移不阻断，只告警）")
    for note in drift_answer.notes:
        print(f"    注记：{note}")
    print()

    # 8.4 未命中（护栏）
    calls_before = len(llm.calls)
    print("  [8.4 未命中] 用一个库里不可能命中的过滤条件：")
    empty = pipeline.answer(
        RetrievalQuery(text=QUERY, where={"strategy": {"$in": ["bm25"]}})
    )
    print(f"    {empty.summary_line()}")
    print(
        f"    llm_called={empty.llm_called} | citations={empty.citations} | "
        f"context={empty.context}"
    )
    print(f"    调用前 LLM 次数={calls_before}，调用后={len(llm.calls)} → "
          f"这次一次 LLM 都没调：{len(llm.calls) == calls_before}")
    print(f"    empty_reason={empty.retrieval.empty_reason if empty.retrieval else None}")
    print("    answer 原文：")
    print(f"      {empty.answer}")
    for note in empty.notes:
        print(f"    注记：{note}")
    print()
    print("  这条护栏是本课最重要的东西：检索不到时，模型手里没有片段，")
    print("  它却仍然会写出一段通顺的答案——那种答案看起来最像成功。")
    print("  因此判空发生在**渲染提示词之前**（见 pipeline.answer 的第 2 步）。")


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
    """按节运行演示，并把完整输出同时写入 ``outputs/retrieval_demo.txt``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    original = sys.stdout
    with output_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            for section in (
                section_1_corpus,
                section_2_depth,
                section_3_filters,
                section_4_time_range,
                section_5_threshold,
                section_6_diversity,
                section_7_drift,
                section_8_rag,
            ):
                section()
            print(f"\n演示完成（工作目录 {Path.cwd()}）")
            print(f"本节输出已同时写入 {output_path}")
            print("八节全部离线：零网络、零新增依赖；分数与命中都是真算出来的。")
        finally:
            sys.stdout = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
