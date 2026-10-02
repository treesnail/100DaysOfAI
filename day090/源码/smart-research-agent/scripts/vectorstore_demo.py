#!/usr/bin/env python
"""day064 演示脚本：向量库（M6-D3）.

八节，全部**离线、确定性、零网络**，只依赖本包与标准库：

```text
1. 后端能力表        三个后端的 available / requires / install_hint（本机真实结果）
2. 写入与重放        6 条记录 → written 6 / unchanged 0；再重放一次 → unchanged 6
3. 三种度量的 top-3  同一批记录、同一个查询，换度量就换答案
4. 并列的断法        cosine 下两条记录分数精确相同 → 名次由 id 决定
5. where 过滤        $in / $and / $contains 各一次 + 一次"过滤器把结果筛空"
6. 与参照实现对账    ParityRow 的 agreed / overlap / first_divergence
7. 快照落盘与读回    JSON 快照的字节数 + 新实例的 ids() 是否逐条相同
8. 缺依赖的报错      三段式原文：缺什么 → 怎么装 → 还能用什么
```

本层最重要的那句边界（教程会引用它）：

    本层只回答'给一个向量、返回最像的 K 条'，再往前一步的问题
    （向量从哪来、命中之后怎么用）都属于别的模块

## 演示样本与测试样本刻意分开

``tests/vectorstore_samples.py`` 里也有一份 6 条样本（外加 ``TableEmbedding``
与独立的 ``brute_force_top``），但**脚本不 import tests**：
本脚本要被别人复制走单独运行，一旦依赖测试目录，复制到别处就跑不起来。
因此这里自带一份更小、但仍自洽的样本：

```text
向量与文本一一对应   每条记录的向量写死在样本里，期望值能手算
有一条并列           q_axis 与两条记录完全同向，余弦都是 1.0
有一条模长差异       其中一条的模长是 10 倍 → 余弦相同、内积分家
```

FAISS / ChromaDB 缺席时脚本不报错：第 1、6、8 节照常运行并打印"为什么不可用"。
结果同时打印到 stdout 并写入 ``outputs/vectorstore_demo.txt``。

运行（cwd 为 ``day064/源码/smart-research-agent``）::

    python scripts/vectorstore_demo.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# 本脚本要能直接 ``python scripts/vectorstore_demo.py`` 跑起来（cwd 是仓库根）。
# 以脚本方式启动时 ``sys.path[0]`` 是 ``scripts/``，仓库根不在其中，
# 而本快照的包**没有 pip 安装**（``pip show smart-research-agent`` 为空）——
# 因此这里显式把仓库根塞进 ``sys.path``，让脚本不依赖 ``PYTHONPATH=.``。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.llm.embedding import EmbeddingProvider  # noqa: E402
from smart_research_agent.vectorstore import (  # noqa: E402
    BackendUnavailable,
    FlatVectorStore,
    VectorIngestPipeline,
    backend_names,
    compare_backends,
    compare_metrics,
    describe_backends,
    describe_filter,
    describe_metrics,
    make_record,
    resolve_backend,
    verify_parity,
)

#: 本层的边界（原文供教程引用，见模块 docstring）.
LAYER_BOUNDARY = (
    "本层只回答'给一个向量、返回最像的 K 条'，"
    "再往前一步的问题（向量从哪来、命中之后怎么用）都属于别的模块"
)

#: 输出目录与文件名（``outputs/`` 在仓库 ``.gitignore`` 里，不污染仓库）.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "vectorstore_demo.txt"
SNAPSHOT_NAME = "vectorstore_demo.snapshot.json"

#: 样本维度。取 8 而不是 384：**期望值能手算**是这份样本的立足点。
DEMO_DIMENSION = 8

#: 查询向量：与两条记录完全同向（制造余弦并列）。
AXIS_VECTOR: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#: 第二个查询向量：偏向 (0.6, 0.8)，让 r_c 成为最近的那条。
TILT_VECTOR: tuple[float, ...] = (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#: 查询文本（编码器查表的键）.
QUERY_AXIS_TEXT = "q:并列查询"
QUERY_TILT_TEXT = "q:偏向查询"

#: 六条演示记录：``(id, 原始向量, 正文, 元数据)``.
#:
#: ``原始向量`` 一律写**未归一化**的那一份：``make_record`` 只在 ``cosine``
#: 下做 L2 归一化，因此同一份样本在 ``ip`` / ``l2`` 下保留模长差异——
#: 这正是第 3 节"换度量换答案"的来源。
#:
#: 写入顺序刻意把**模长 10 倍**的那条放在最前、把并列的另一条放在最后：
#: 于是第 4 节里"名次由 id 决定"不能被误读成"名次由插入顺序决定"。
DEMO_RECORDS: tuple[tuple[str, tuple[float, ...], str, dict[str, Any]], ...] = (
    (
        "a1f34c8b62e0d975",
        (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "阈值用分位数标定，换 embedding 模型必须重新标定。",
        {
            "parent_doc_id": "b41ad29c8f70e613",
            "strategy": "structural",
            "index": 0,
            "token_count": 120,
            "heading_path": "语义缓存 > 阈值",
            "topic": "cache",
            "tags": ["guide", "cache"],
            "oversized": False,
        },
    ),
    (
        "b7e2d40a9c13f586",
        (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "默认返回前 5 条，相似度低于 0.35 的直接丢掉。",
        {
            "parent_doc_id": "7feafd1c95ab3577",
            "strategy": "recursive",
            "index": 1,
            "token_count": 210,
            "heading_path": "检索手册 > 检索",
            "topic": "retrieval",
            "tags": ["guide", "search"],
            "oversized": False,
        },
    ),
    (
        "e08b7d25c4a91f36",
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "术语表列出本手册用到的全部缩写。",
        {
            "parent_doc_id": "b41ad29c8f70e613",
            "strategy": "semantic",
            "index": 0,
            "token_count": 90,
            "heading_path": "附录 > 术语表",
            "topic": "glossary",
            "tags": ["appendix"],
            "oversized": False,
        },
    ),
    (
        "5c9d0e13ab7f2468",
        (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "一次全量重建索引大约 12 万条记录，按每条 320 token 计。",
        {
            "parent_doc_id": "7feafd1c95ab3577",
            "strategy": "fixed",
            "index": 2,
            "token_count": 160,
            "heading_path": "检索手册 > 成本",
            "topic": "cost",
            "tags": ["guide", "cost"],
            "oversized": False,
        },
    ),
    (
        "2d6b9a50f31c7e84",
        (0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        "参数表把默认值与取值范围并排列出。",
        {
            "parent_doc_id": "b41ad29c8f70e613",
            "strategy": "structural",
            "index": 1,
            "token_count": 64,
            "heading_path": "附录 > 参数表",
            "topic": "glossary",
            "tags": ["appendix", "cost"],
            "oversized": True,
        },
    ),
    (
        "3f5a91c0d27be684",
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "固定长度分块每 320 个字符切一刀，重叠 48 个字符。",
        {
            "parent_doc_id": "7feafd1c95ab3577",
            "strategy": "recursive",
            "index": 0,
            "token_count": 319,
            "heading_path": "检索手册 > 分块",
            "topic": "chunking",
            "tags": ["guide", "chunk"],
            "oversized": False,
        },
    ),
)


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# 样本构造：记录、编码器、独立参照
# --------------------------------------------------------------------------- #


def retrieval_text(heading_path: str, text: str) -> str:
    """day062 约定的检索视图：面包屑 + 正文（**它才是被编码的那份文本**）."""
    return f"{heading_path}\n{text}"


def demo_records(*, metric: str = "cosine") -> list[Any]:
    """六条 ``VectorRecord``（按 ``DEMO_RECORDS`` 的顺序）.

    ``metric`` 决定要不要先归一化：``cosine`` 会归一化、``ip`` / ``l2`` 不会。
    """
    return [
        make_record(record_id, vector, text, dict(metadata), metric=metric)
        for record_id, vector, text, metadata in DEMO_RECORDS
    ]


def knowledge_records() -> list[dict[str, Any]]:
    """day062 ``ChunkSet.knowledge_records()`` 形状的样本（四个键）.

    形状与 day062 的产物逐键一致（``doc_id`` / ``source`` / ``text`` /
    ``metadata``），因为它就是本层的**真实**输入——用别的形状造样本，
    会让"两个模块能接上"这件事永远没被验证过。
    """
    records: list[dict[str, Any]] = []
    for record_id, _vector, text, metadata in DEMO_RECORDS:
        records.append(
            {
                "doc_id": record_id,
                "source": f"docs/{metadata['parent_doc_id']}.md",
                "text": text,
                "metadata": {
                    "parent_doc_id": metadata["parent_doc_id"],
                    "strategy": metadata["strategy"],
                    "index": str(metadata["index"]),
                    "token_count": str(metadata["token_count"]),
                    "token_measurer": "chars",
                    "heading_path": metadata["heading_path"],
                    "fingerprint": record_id[::-1],
                    "oversized": "true" if metadata["oversized"] else "false",
                    # 检索视图与正文不同：pipeline 必须优先用这一个（day062 的约定）
                    "retrieval_text": retrieval_text(metadata["heading_path"], text),
                },
            }
        )
    return records


def embedding_table() -> dict[str, tuple[float, ...]]:
    """文本 → 向量的查表（**把"语义"由人给定**，与测试样本同一思路）."""
    table = {
        retrieval_text(metadata["heading_path"], text): vector
        for _record_id, vector, text, metadata in DEMO_RECORDS
    }
    table[QUERY_AXIS_TEXT] = AXIS_VECTOR
    table[QUERY_TILT_TEXT] = TILT_VECTOR
    return table


class DemoEmbedding(EmbeddingProvider):
    """按文本查表的确定性编码器：向量写死，期望值因此能手算.

    为什么不用 ``MockEmbedding``：它把文本哈希成向量，于是"哪条最近"
    只能靠实际跑一遍才知道——那样打印出来的数字无法被核对。
    """

    def __init__(self, table: dict[str, tuple[float, ...]] | None = None) -> None:
        self._table = dict(table or embedding_table())
        self._default = AXIS_VECTOR

    @property
    def dimension(self) -> int:
        """向量维度."""
        return DEMO_DIMENSION

    def embed(self, text: str) -> list[float]:
        """按文本查表；查不到落到 ``q_axis``（本脚本不该有查不到的情况）."""
        return list(self._table.get(text, self._default))


def reference_ranking(
    records: list[Any], query: list[float], metric: str, top_k: int = 3
) -> list[tuple[str, float]]:
    """**独立的**暴力检索参照实现：只用 ``math``，不复用 ``metrics.py``.

    判定的依据必须与被判定的实现相互独立，否则"两者一致"只证明
    "这段代码等于它自己"。排序规则仍与本包一致：分数降序、id 升序。
    """
    scored = [
        (record.record_id, _reference_score(query, record.vector, metric))
        for record in records
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


def _reference_score(query: list[float], vector: tuple[float, ...], metric: str) -> float:
    """三种度量的朴素公式（``l2`` 取负号，让"越大越近"统一）."""
    products = math.fsum(float(a) * float(b) for a, b in zip(query, vector))
    if metric == "cosine":
        query_norm = math.sqrt(math.fsum(float(a) * float(a) for a in query))
        vector_norm = math.sqrt(math.fsum(float(b) * float(b) for b in vector))
        if query_norm == 0.0 or vector_norm == 0.0:
            return 0.0
        return products / (query_norm * vector_norm)
    if metric == "ip":
        return products
    return -math.fsum((float(a) - float(b)) ** 2 for a, b in zip(query, vector))


# --------------------------------------------------------------------------- #
# 1. 后端能力表
# --------------------------------------------------------------------------- #


def section_1_backends() -> None:
    """第 1 节：后端能力表与度量说明（本机真实结果）."""
    title("1. 后端能力表：available / requires / install_hint（本机实测）")
    print("  这一层的边界（原文，供教程引用）：")
    print(f"    {LAYER_BOUNDARY}")
    print()
    print("  三个后端按「依赖递增 / 可控性递减」排列：")
    print(f"    {'后端':<8}{'可用':<7}{'需要':<17}{'安装命令':<24}它适合什么")
    for row in describe_backends():
        requires = ", ".join(row["requires"]) or "（无）"
        install = row["install_hint"] or "（无需安装）"
        available = "是" if row["available"] else "否"
        print(
            f"    {row['name']:<8}{available:<7}{requires:<17}{install:<24}"
            f"{row['description']}"
        )
    print()
    print("  不可用的后端：缺的是哪一个包（``missing`` 与 ``available`` 一起给，")
    print("  因为只给一个布尔值时，看到 False 还得自己去猜缺什么）：")
    for row in describe_backends():
        if not row["available"]:
            print(f"    {row['name']:<8}缺 {', '.join(row['missing'])}")
    print()
    print("  三个度量（统一口径 **越大越近**；distance 只在对外对账时出现）：")
    print(f"    {'度量':<8}{'越大越近':<10}{'需归一化':<10}说明")
    for row in describe_metrics():
        normalized = row["needs_normalization"]
        print(f"    {row['metric']:<8}{row['better']:<10}{normalized:<10}{row['note']}")
    print()
    print(
        f"  当前默认（config.Settings）：backend={settings.vector_backend} "
        f"metric={settings.vector_metric} top_k={settings.vector_default_top_k} "
        f"min_score={settings.vector_min_score}"
    )
    print(
        f"  持久化默认 vector_persist_path={settings.vector_persist_path!r}"
        "（空串 = 明确不落盘：默认绝不静默写盘）"
    )


# --------------------------------------------------------------------------- #
# 2. 写入与重放
# --------------------------------------------------------------------------- #


def section_2_ingest() -> None:
    """第 2 节：写入与重放，written / unchanged 是两个必须分开的数字."""
    title("2. 写入与重放：written 与 unchanged 是两份证据")
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, DemoEmbedding())
    records = knowledge_records()

    first = pipeline.ingest(records)
    print(f"  第一次 ingest：{first.summary_line()}")
    print(
        f"    seen={first.seen} written={first.written} "
        f"unchanged={first.unchanged} skipped={first.skipped} "
        f"failed={first.failed} embedding_calls={first.embedding_calls}"
    )
    print(f"    库状态：{store.info().summary_line()}")

    second = pipeline.ingest(records)
    print(f"\n  重放同一批：{second.summary_line()}")
    print(
        f"    seen={second.seen} written={second.written} "
        f"unchanged={second.unchanged} embedding_calls={second.embedding_calls}"
    )
    print(f"    库状态仍然：{store.info().summary_line()}")
    print()
    print("  两个数字的分工：")
    print("    written=6  → 这次有 6 条真的写进了库（added+updated）")
    print("    unchanged=6 → 这次**没有改动库里的任何一条**（逐位相同的记录不会被写回）")
    print(
        f"    embedding_calls 两次都是 {second.embedding_calls}：本课是**逐条编码、没有向量缓存**，"
    )
    print("    所以重放仍然编码了 6 次（库没变，但钱照花）。")
    print("    day065 的增量对账就靠 unchanged 这个数来消掉那 6 次编码——")
    print("    把 unchanged 并进 updated 的话，'这次省了多少'这个问题就永远答不出来。")


# --------------------------------------------------------------------------- #
# 3. 三种度量的 top-3
# --------------------------------------------------------------------------- #


def section_3_metrics() -> None:
    """第 3 节：三种度量的 top-3（同一批记录、同一个查询）."""
    title("3. 三种度量的 top-3：换度量就是换答案")
    print(f"  查询 q_axis = {AXIS_VECTOR}")
    print("  （它与两条记录完全同向；其中一条的模长是 10 倍）")
    for metric in ("cosine", "ip", "l2"):
        store = FlatVectorStore(metric=metric)
        store.upsert(demo_records(metric=metric))
        result = store.query(list(AXIS_VECTOR), 3)
        print(f"\n  metric={metric}：{result.summary_line()}")
        for hit in result.hits:
            print(f"    {hit.summary_line()}")
    print()
    print("  cosine 与 ip 的 top-1 不是同一条：")
    print("    余弦先归一化再比方向 → 模长 10 倍的那条不占优（两条都是 1.0，并列）；")
    print("    内积不做归一化      → 同一个方向下，模长大的那条分数更大（10.0 vs 1.0）。")
    print("  l2 的 top-1 又是另一条：它按'距离'排，离得越远分数越负。")
    print("  （l2 那一行的 score=-0.000000 是**负零**：0.0 取负的结果，不是错误。）")


# --------------------------------------------------------------------------- #
# 4. 并列的断法
# --------------------------------------------------------------------------- #


def section_4_tie() -> None:
    """第 4 节：分数精确相同时，名次由 id 决定."""
    title("4. 并列的断法：分数精确相同时，名次由 id 决定")
    store = FlatVectorStore(metric="cosine")
    store.upsert(demo_records(metric="cosine"))
    insertion = "、".join(record_id for record_id, *_ in DEMO_RECORDS)
    print(f"  写入顺序：{insertion}")
    result = store.query(list(AXIS_VECTOR), 3)
    print(f"  查询结果：{result.summary_line()}")
    for hit in result.hits:
        print(f"    {hit.summary_line()}")
    top_score = result.hits[0].score
    tied = [hit for hit in result.hits if hit.score == top_score]
    print(f"\n  分数精确等于 {top_score!r} 的有 {len(tied)} 条：")
    for hit in tied:
        print(f"    #{hit.rank} {hit.record.record_id} score={hit.score!r}")
    print()
    print("  断法只写在一处（types.sort_hits）：**分数降序、id 升序**。")
    print("  第二条键不是装饰：同一段文本用确定性编码器会得到逐位相同的向量，")
    print("  于是分数**精确相等**；若只看分数，顺序会取决于 dict 的插入顺序，")
    print("  而插入顺序在分批重放之后会变 —— 那会让同一次查询两次运行给出不同的 top-1。")
    print(f"  这里插入顺序是 {DEMO_RECORDS[0][0]} 在前，名次却是并列里 id 较小的那条在前。")


# --------------------------------------------------------------------------- #
# 5. where 过滤
# --------------------------------------------------------------------------- #


def _filtered_store() -> FlatVectorStore:
    """给第 5 节用的余弦库（含六条样本）."""
    store = FlatVectorStore(metric="cosine")
    store.upsert(demo_records(metric="cosine"))
    return store


def section_5_filters() -> None:
    """第 5 节：where 子句与"候选数 / 命中数"两个字段."""
    title("5. where 过滤：candidates 与 count 回答的是两个问题")
    store = _filtered_store()
    cases = [
        ("$in", {"strategy": {"$in": ["recursive", "fixed"]}}),
        ("$and", {"$and": [{"strategy": "structural"}, {"oversized": True}]}),
        ("$contains", {"tags": {"$contains": "cost"}}),
        ("筛空", {"strategy": {"$in": ["hybrid"]}}),
    ]
    print("  库里六条记录的 strategy：recursive×2、fixed×1、structural×2、semantic×1")
    for label, where in cases:
        result = store.query(list(AXIS_VECTOR), 3, where=where)
        print(f"\n  [{label}] {describe_filter(where)}")
        print(f"    {result.summary_line()}")
        print(
            f"    candidates={result.candidates} count={result.count} "
            f"filter_applied={result.filter_applied}"
        )
        for hit in result.hits:
            print(
                f"      {hit.record.record_id} "
                f"strategy={hit.record.metadata['strategy']} "
                f"tags={hit.record.metadata['tags']}"
            )
    print()
    print("  顺带一组「没有过滤、但阈值把命中切空」的对照（同样是真实数字）：")
    threshold = store.query(list(AXIS_VECTOR), 3, min_score=1.5)
    print(f"    min_score=1.5 → {threshold.summary_line()}")
    print(
        f"    candidates={threshold.candidates} count={threshold.count} "
        f"filter_applied={threshold.filter_applied}"
    )
    print()
    print("  这就是 candidates 与 filter_applied 存在的理由——三种情形在返回体里")
    print("  长得一模一样（hits 很短或为空），只有这两个字段能把它们分开：")
    print("    candidates=0 且 filter_applied=True  → 过滤器把所有记录都排除了")
    print("    candidates=0 且 filter_applied=False → 库是空的")
    print("    candidates=6 而 count=0             → 阈值把命中切掉了（数据在，只是不够像）")
    print("  另外 $in 给一个库里没有的值不会报错：它是「合法但没有命中」的查询。")


# --------------------------------------------------------------------------- #
# 6. 与参照实现对账
# --------------------------------------------------------------------------- #


def _optional_backends() -> list[tuple[str, Any]]:
    """对账要用的后端规格；两个可选后端**取用时才加载**（包的延迟导出）."""
    specs: list[tuple[str, Any]] = [("flat", FlatVectorStore)]
    try:
        from smart_research_agent.vectorstore import FaissVectorStore

        specs.append(("faiss", FaissVectorStore))
    except ImportError as exc:  # pragma: no cover - 本机 numpy 在，导入不会失败
        print(f"    faiss 后端无法加载：{exc}")
    try:
        from smart_research_agent.vectorstore import ChromaVectorStore

        specs.append(("chroma", ChromaVectorStore))
    except ImportError as exc:  # pragma: no cover - 本机 chromadb 缺失也不影响导入
        print(f"    chroma 后端无法加载：{exc}")
    return specs


def section_6_parity() -> None:
    """第 6 节：与独立参照实现对账（ParityRow 的三个结论字段）."""
    title("6. 与参照实现对账：agreed / overlap / first_divergence")
    # 刻意用 metric="ip" 构造（不归一化）：这样 cosine 与 ip 才会分家。
    records = demo_records(metric="ip")
    query = list(AXIS_VECTOR)
    reference = {
        metric: reference_ranking(records, query, metric, 3)
        for metric in ("cosine", "ip", "l2")
    }
    print("  独立的参照实现（脚本自带，只用 math，不复用 metrics.py）：")
    for metric in ("cosine", "ip", "l2"):
        ids = [record_id for record_id, _ in reference[metric]]
        print(f"    {metric:<7}期望：{ids}")
    print()
    rows = compare_backends(
        records,
        query,
        metrics=("cosine", "ip"),
        top_k=3,
        backends=_optional_backends(),
        reference=reference,
    )
    print("  compare_backends（flat + faiss + chroma × cosine + ip）：")
    for row in rows:
        if row.available:
            print(f"    {row.summary_line()}")
        else:
            reason = (row.note or "缺少可选依赖").splitlines()[0]
            print(f"    {row.backend:<8} | {row.metric:<6} | 不可用 | {reason[:56]}…")
        print(
            f"        agreed={row.agreed} overlap={row.overlap:.6f} "
            f"first_divergence={row.first_divergence!r}"
        )
    verdict = verify_parity(rows)
    print()
    print(
        f"  verify_parity：ok={verdict['ok']} checked={verdict['checked']} "
        f"available={verdict['available']}"
    )
    for failure in verdict["failures"]:
        print(f"    失败：{failure[:92]}")
    print()
    print("  注意 ok=False 不等于「对账失败」：不可用的后端也会被记成一条 failure。")
    print("  这正是「装了 faiss 的机器与没装的机器，报告必须长得一样」的实现方式——")
    print("  差异只表现为一个 available 布尔与一句原因，而不是少了几行报告。")
    print()
    print("  同一份数据、只换度量（compare_metrics，基线是 cosine）：")
    metric_rows = compare_metrics(records, query, metrics=("cosine", "ip", "l2"), top_k=3)
    for row in metric_rows:
        print(f"    {row.summary_line()}")
        print(
            f"        agreed={row.agreed} overlap={row.overlap:.6f} "
            f"first_divergence={row.first_divergence!r}"
        )
    print()
    print("  first_divergence 的措辞里同时有「期望」与「实际」：对账的价值")
    print("  在于指名道姓，一句「结果不一致」对读报告的人没有任何帮助。")


# --------------------------------------------------------------------------- #
# 7. 快照落盘与读回
# --------------------------------------------------------------------------- #


def section_7_snapshot() -> None:
    """第 7 节：写 JSON 快照 → 打印字节数 → 新实例读回 → 比较 ids()."""
    title("7. 快照落盘与读回：JSON 是一份能被人打开的证据")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = OUTPUT_DIR / SNAPSHOT_NAME
    store = FlatVectorStore(metric="cosine")
    store.upsert(demo_records(metric="cosine"))
    written = store.persist(str(snapshot_path))
    size = Path(written).stat().st_size
    print(f"  persist → {written}")
    print(f"  文件字节数：{size}")
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    print(
        f"  快照头部：version={payload['version']} backend={payload['backend']!r} "
        f"metric={payload['metric']!r} dimension={payload['dimension']} "
        f"count={payload['count']}"
    )
    replayed = FlatVectorStore(metric="cosine", path=str(snapshot_path))
    print(f"  新实例（构造时给 path，自动 load）：{replayed.info().summary_line()}")
    same = replayed.ids() == store.ids()
    print(f"  ids() 逐条相同：{same}")
    print(f"    原实例 {len(store.ids())} 条 / 新实例 {len(replayed.ids())} 条")
    print(f"    ids() = {store.ids()}")
    print()
    print("  快照而不是 pickle：它的价值在于**能被人打开看一眼**——")
    print("  一份读不懂的快照，在'为什么重启之后少了两条'这种问题前面毫无帮助。")
    print("  记录按 id 升序写出，因此同一次状态的两次快照逐字节相同（便于 diff）。")


# --------------------------------------------------------------------------- #
# 8. 缺依赖时的三段式报错
# --------------------------------------------------------------------------- #


def section_8_unavailable() -> None:
    """第 8 节：缺依赖时的三段式报错原文（缺什么 → 怎么装 → 还能用什么）."""
    title("8. 缺依赖时的三段式报错：缺什么 → 怎么装 → 还能用什么")
    print(f"  backend_names() = {list(backend_names())}")
    for name in ("faiss", "chroma"):
        print(f"\n  resolve_backend({name!r}) 抛出的完整消息：")
        try:
            resolve_backend(name)
        except BackendUnavailable as exc:
            for line in str(exc).splitlines():
                print(f"    {line}")
        else:
            print("    （本机已装，本次没有报错）")
    print()
    backend = resolve_backend("flat")
    print(f"  现场可用的兜底后端：{backend.info().summary_line()}")
    print("  第三句（还能用什么）是三句里最容易被省掉、也最有用的一句：")
    print("  flat 的 requires 是空元组，因此它在任何机器上都给得出一个真答案——")
    print("  进度不该被一个可选依赖卡死。")


def _tee_streams() -> Any:
    """返回把 stdout 同时写进文件的包装器（见 ``_Tee``）."""
    return OUTPUT_DIR / OUTPUT_NAME


class _Tee:
    """把写往 stdout 的内容**同时**送到终端与文件（第 1 行的硬要求）."""

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
    """按节运行演示，并把完整输出同时写入 ``outputs/vectorstore_demo.txt``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _tee_streams()
    original = sys.stdout
    with output_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            for section in (
                section_1_backends,
                section_2_ingest,
                section_3_metrics,
                section_4_tie,
                section_5_filters,
                section_6_parity,
                section_7_snapshot,
                section_8_unavailable,
            ):
                section()
            print(f"\n演示完成（工作目录 {Path.cwd()}）")
            print(f"本节输出已同时写入 {output_path}")
        finally:
            sys.stdout = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
