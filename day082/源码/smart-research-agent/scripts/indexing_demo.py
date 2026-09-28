#!/usr/bin/env python
"""day065 演示脚本：Embedding 索引构建（M6-D4）.

八节，全部**离线、确定性、零网络**，只依赖本包与标准库：

```text
1. 编码器身份与向量键   身份三字段进键 → 换 model / dimension 之后同一段文本的键全变
2. 批量编码            batch_size 32 / 4 / 1 的 encoded / batches / cache_hits，含一次全命中
3. 差集的四条判据       未变 / 内容变了 / 删了一条 / 换了身份（IndexPlan + plan_reason）
4. 增量构建的省钱证据    首次构建 8 条都编码 → 重放 encoded=0、batches=0、版本号不变
5. 阈值换挡            变更比例 50% 超过阈值 0.10 → 从 incremental 切到 full（note 原文）
6. 版本表与血缘         三版 + 采纳第二版 + rollback(steps=1) → 回到第一版（沿血缘）
7. 备份与保留           keep=2 建三份 → 被 prune 的 backup_id、账本与可用份数、restore
8. 一致性体检           人为从库里删一条 → missing_in_store 与 problems 原文
```

本层最重要的那句边界（教程会引用它）：

    本层不回答'这段文本该不该进库'，只回答'它变了没有、要不要重算向量'——
    前者是分块与过滤的事，后者才是记账的事。

## 演示样本与测试样本刻意分开

``tests/`` 里也有一份"3 条记录、4 维向量"的样本，但**脚本不 import tests**：
脚本要能被别人复制走单独运行，一旦依赖测试目录，复制到别处就跑不起来。
因此这里自带一份 8 条记录的样本（day062 ``knowledge_records()`` 的四个键：
``doc_id`` / ``source`` / ``text`` / ``metadata``），编码器用
``CharNgramEmbedding`` 的确定性派生类——它是**真编码器**（离线、逐位可复现），
不是把向量写死的查表，因此"提供方被调用了几次"这件事是真的，
而本课要打印的每一个数字（encoded / batches / cache_hits）都是由它数出来的。

## 落盘产物写在 outputs/ 下，不碰 data/index

清单、快照、备份与恢复都发生在 ``outputs/indexing_demo_work/``
（``outputs/`` 在仓库 ``.gitignore`` 里）。演示脚本不该在仓库里留下一个
看起来像"线上索引"的 ``data/index`` 目录——那会让"这份索引是哪来的"
变成一个必须回答的问题。脚本结果同时打印到 stdout 并写入
``outputs/indexing_demo.txt``。

运行（cwd 为 ``day065/源码/smart-research-agent``）::

    python scripts/indexing_demo.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

# 本脚本要能直接 ``python scripts/indexing_demo.py`` 跑起来（cwd 是仓库根）。
# 以脚本方式启动时 ``sys.path[0]`` 是 ``scripts/``，仓库根不在其中，
# 而本快照的包**没有 pip 安装**（``pip show smart-research-agent`` 为空）——
# 因此这里显式把仓库根塞进 ``sys.path``，让脚本不依赖 ``PYTHONPATH=.``。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.documents.types import content_id  # noqa: E402
from smart_research_agent.indexing import (  # noqa: E402
    BACKUP_INDEX_FILE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_KEEP,
    BatchEncoder,
    EmbeddingCache,
    EmbeddingIdentity,
    IndexBackupStore,
    IndexBuilder,
    IndexVersionStore,
    build_manifest,
    compare_with_store,
    describe_embedding,
    entry_from_record,
    plan_index,
    vector_key,
    verify_index,
)
from smart_research_agent.llm.embedding import CharNgramEmbedding  # noqa: E402
from smart_research_agent.vectorstore import (  # noqa: E402
    FlatVectorStore,
    embedding_text,
)

#: 本层的边界（原文供教程引用，见模块 docstring）.
LAYER_BOUNDARY = (
    "本层不回答'这段文本该不该进库'，只回答'它变了没有、要不要重算向量'——"
    "前者是分块与过滤的事，后者才是记账的事。"
)

#: 输出目录与文件名（``outputs/`` 在仓库 ``.gitignore`` 里，不污染仓库）.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "indexing_demo.txt"
#: 落盘产物（快照 / 备份 / 恢复）的临时位置。每次运行先删掉它，
#: 否则上一次留下的备份目录会让"这次 create 出了几份"变得不可复现。
WORK_DIR = OUTPUT_DIR / "indexing_demo_work"

#: 样本维度。取 256 而不是 8：这一节要演示的是**计数与身份**，
#: 用 ``CharNgramEmbedding`` 的真实默认维度，脚本里没有依赖维度的手算期望值。
DEMO_DIMENSION = 256

#: 样本来源（记录里的 ``source`` 字段，只是元数据）。
DEMO_SOURCE = "docs/embedding_index.md"

#: 八条演示块：``(doc_id, heading_path, text, token_count)``.
#:
#: ``doc_id`` 一律是 16 位十六进制（day062 ``chunk_id`` 的形状），
#: ``text`` 是原文片段（落库的那份），``heading_path`` 是面包屑——
#: 它与 ``text`` 拼起来才是**被编码**的检索视图（``retrieval_text``）。
DEMO_CHUNKS: tuple[tuple[str, str, str, int], ...] = (
    (
        "4f1a7c20b93e5d68",
        "索引手册 > 缓存",
        "缓存键里必须含编码器身份，否则换模型之后会命中旧向量。",
        28,
    ),
    (
        "8c02e5b71d4a396f",
        "索引手册 > 缓存",
        "重放同一批时 encoded 等于 0，这一次一次提供方调用都没有发生。",
        30,
    ),
    (
        "1d9b4e60a37c8251",
        "索引手册 > 批量编码",
        "batch_size 太小会退化成逐条调用，太大则一次失败要重试一整批。",
        29,
    ),
    (
        "a3e81f5c20d97b46",
        "索引手册 > 批量编码",
        "同一批里的重复文本只编码一次，它省下的调用与缓存命中是同一个数字。",
        31,
    ),
    (
        "7b5d0a13c8f24e96",
        "索引手册 > 增量与全量",
        "增量不是永远更省，变更比例超过阈值时改走全量重建。",
        24,
    ),
    (
        "c60f92a4b1e83d57",
        "索引手册 > 版本与血缘",
        "回滚必须沿血缘走，不能按登记顺序倒着数。",
        20,
    ),
    (
        "2e74b8c905a1f63d",
        "索引手册 > 备份",
        "备份不是日志，不设保留上限会一直长到吃满磁盘。",
        22,
    ),
    (
        "9a0c3e57d28b4f16",
        "索引手册 > 体检",
        "清单里有、库里没有，是上一次构建半途失败最典型的样子。",
        25,
    ),
)

#: 第一条块的 id（第 3、5、6、7 节都拿它当"被改的那一条"）.
ID_FIRST = DEMO_CHUNKS[0][0]

#: 注入的固定时钟。备份记录的 ``created_at`` 若取真实时间，脚本每跑一次的
#: 输出都会不同（第 7 节会逐行打印它）。因此这里像测试一样把它钉死：
#: **``created_at`` 是给人看的备注，它不进版本号**（见 types.index_version_id），
#: 钉死它不会让任何身份变假，只是让这份输出可以被逐字节复现。
DEMO_CLOCK = "2026-01-02T03:04:05+00:00"


def demo_clock() -> str:
    """固定时钟（``IndexBackupStore(clock=...)`` 的注入点）."""
    return DEMO_CLOCK


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def rel(path: str | Path) -> str:
    """把路径写成相对仓库根的短形式（逐行打印时短路径读起来省力得多）."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:  # pragma: no cover - 本脚本的路径都在仓库内
        return str(path)


# --------------------------------------------------------------------------- #
# 样本构造：八条记录与一个确定性编码器
# --------------------------------------------------------------------------- #


class DemoEmbedding(CharNgramEmbedding):
    """确定性字符 n-gram 编码器 + 一份"提供方真的被调用了几次"的台账.

    为什么**不用**把向量写死的查表实现：本课的每一个数字都是计数
    （``encoded`` / ``batches`` / ``cache_hits``），而计数只有在
    "提供方真的被调用"时才有意义——查表不会告诉我们"这一批发了几次调用"。

    为什么**不用** ``MockEmbedding``：它把文本哈希成互不相关的向量，
    "同一段文本两次得到同一个向量"仍然成立，但向量之间没有任何距离含义；
    ``CharNgramEmbedding`` 至少让"字面重叠多的文本向量更近"这件事是真的。

    两个计数器刻意放在**编码器**上而不是编排层里：``batch_calls``
    就是"向提供方发起了几批"，它是 ``EncodeReport.batches`` 的独立对照——
    两个数字相等，才说明报告里的 ``batches`` 不是估算。
    """

    def __init__(self, dimension: int = DEMO_DIMENSION) -> None:
        super().__init__(dimension=dimension)
        #: ``embed_batch`` 被调用了几次（= 向提供方发起了几批）
        self.batch_calls = 0
        #: ``embed_batch`` 一共收到几段文本
        self.texts_requested = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """一次前向编码整批，并记下这次调用."""
        self.batch_calls += 1
        self.texts_requested += len(texts)
        return super().embed_batch(texts)


def _record(doc_id: str, heading_path: str, text: str, token_count: int) -> dict[str, Any]:
    """造一条 day062 ``knowledge_records()`` 形状的记录（四个键）.

    三个字段各有出处，**它们的差别就是本课两个指纹的差别**：

    ```text
    text                       原文片段         → 落库；fingerprint 由它算出（内容身份）
    metadata["retrieval_text"] 面包屑 + 正文     → 送进编码器；vector_key 由它算出
    metadata["fingerprint"]    content_id(text) → day062 算好的内容身份，planner 优先用它
    ```
    """
    return {
        "doc_id": doc_id,
        "source": DEMO_SOURCE,
        "text": text,
        "metadata": {
            "heading_path": heading_path,
            "strategy": "recursive",
            "token_count": token_count,
            "fingerprint": content_id(text),
            "retrieval_text": f"{heading_path}\n{text}",
        },
    }


def demo_records() -> list[dict[str, Any]]:
    """八条记录（每次调用返回一份**新的**列表，因此改动不会串到别的节）."""
    return [_record(*chunk) for chunk in DEMO_CHUNKS]


def with_text(records: list[dict[str, Any]], doc_id: str, text: str) -> list[dict[str, Any]]:
    """把某一条的原文换掉，并**同时**更新它的两个派生字段.

    这一步必须一起做，否则本节的演示就是个假动作：
    ``fingerprint`` 不跟着改 → planner 会认为"内容没变"；
    ``retrieval_text`` 不跟着改 → ``vector_key`` 也没变（编码的是检索视图）。
    真实的 day062 产出永远是一致的，脚本改样本时也必须一致。
    """
    updated: list[dict[str, Any]] = []
    for record in records:
        payload: dict[str, Any] = {**record, "metadata": dict(record["metadata"])}
        if record["doc_id"] == doc_id:
            heading = str(payload["metadata"]["heading_path"])
            payload["text"] = text
            payload["metadata"]["fingerprint"] = content_id(text)
            payload["metadata"]["retrieval_text"] = f"{heading}\n{text}"
        updated.append(payload)
    return updated


def demo_texts(records: list[dict[str, Any]]) -> list[str]:
    """这一批记录**实际会送给编码器**的文本（``retrieval_text`` 优先）."""
    return [embedding_text(record) for record in records]


# --------------------------------------------------------------------------- #
# 1. 编码器身份与向量键
# --------------------------------------------------------------------------- #


def section_1_identity() -> None:
    """第 1 节：身份三字段、身份指纹，以及"换配置键全变"的实测."""
    title("1. 编码器身份与向量键：身份进键，换配置就必须整库重算")
    print("  本层的边界（原文，供教程引用）：")
    print(f"    {LAYER_BOUNDARY}")
    print()
    identity = describe_embedding(DemoEmbedding())
    print("  describe_embedding(编码器) 只读**公开属性**：")
    print("    provider  = type(embedding).__name__（永远拿得到）")
    print(f"    model     = model → model_name → 回落常量，实测取到 {identity.model!r}")
    print("                n-gram 阶数是私有字段，所以它只能落到回落值——")
    print("                这不是 bug，是'建议值而不是判决'的兑现（见 encoder.py）")
    print(f"    dimension = embedding.dimension = {identity.dimension}")
    print(f"    identity.key = {identity.key}   ← 身份指纹，进向量键与版本号")
    print(f"    identity.summary_line() = {identity.summary_line()}")
    print()

    sample_text = demo_texts(demo_records())[0]
    print("  同一段**编码文本**（vectorstore.pipeline.embedding_text，取 retrieval_text）：")
    print(f"    {sample_text!r}")
    print()
    variants: tuple[tuple[str, EmbeddingIdentity], ...] = (
        ("现状", identity),
        (
            "只换 model",
            EmbeddingIdentity(
                provider=identity.provider,
                model="char-ngram-2-3",
                dimension=identity.dimension,
            ),
        ),
        (
            "只换 dimension",
            EmbeddingIdentity(
                provider=identity.provider,
                model=identity.model,
                dimension=384,
            ),
        ),
    )
    print("  三种身份下同一段文本的键：")
    keys: list[str] = []
    for label, item in variants:
        key = vector_key(item.provider, item.model, item.dimension, sample_text)
        keys.append(key)
        print(
            f"    {label}：model={item.model!r} / {item.dimension}d"
            f" | 身份 {item.key} | 向量键 {key}"
        )
    print()
    print(f"  三个向量键两两不同：{len(set(keys)) == len(keys)}（{len(set(keys))} 个不同的键）")
    print("  也就是说：**换 model 或换 dimension 之后，同一段文本的键全变了**——")
    print("  planner 会把全部条目判成 updated，旧向量一条都复用不了。")
    print("  这不是靠人记得重建索引，而是结构上漏不掉（见 types.vector_key 的取舍）。")
    print()
    cache = EmbeddingCache(
        provider=identity.provider, model=identity.model, dimension=identity.dimension
    )
    print("  缓存的键与向量键是**同一个函数**（cache.EmbeddingCache.key → types.vector_key）：")
    print(f"    cache.key(同一段文本) == 上面那个向量键 → {cache.key(sample_text) == keys[0]}")
    print(
        f"    cache.identity_key == identity.key → {cache.identity_key == identity.key}"
        f"（{cache.identity_key}）"
    )
    print("  身份进了键，所以换编码器之后**旧缓存条目永远不会命中**——")
    print("  而不是'命中了旧模型的向量，检索结果只是变得不太对'。")


# --------------------------------------------------------------------------- #
# 2. 批量编码
# --------------------------------------------------------------------------- #


def section_2_batching() -> None:
    """第 2 节：三种 batch_size 与一次"全命中"对照."""
    title("2. 批量编码：batch_size 是成本参数，不是正确性参数")
    texts = demo_texts(demo_records())
    print(f"  样本：{len(texts)} 条记录，编码文本取 retrieval_text（面包屑 + 正文）。")
    print(
        f"  默认批大小：settings.indexing_batch_size = {settings.indexing_batch_size}，"
        f"encoder.DEFAULT_BATCH_SIZE = {DEFAULT_BATCH_SIZE}"
    )
    print()
    print("  同一个 8 条文本、三种 batch_size（每次都用**全新编码器 + 全新缓存**）：")
    print()
    print("    batch_size | encoded | batches | cache_hits | 提供方调用 | 命中率")
    for batch_size in (32, 4, 1):
        embedding = DemoEmbedding()
        encoder = BatchEncoder(embedding, batch_size=batch_size)
        _vectors, report = encoder.encode(texts)
        print(
            f"    {batch_size:>10} | {report.encoded:>7} | {report.batches:>7} | "
            f"{report.cache_hits:>10} | {embedding.batch_calls:>10} | {report.hit_ratio:>5.0%}"
        )
    print()
    print("  encoded 恒等于 8：batch_size 只改变'分几次送'，不改变'送了几条'。")
    print("  batches 才是成本：32 → 1 批、4 → 2 批、1 → 8 批（退化成逐条调用，")
    print("  也就是 day064 那条基线：库没变、钱照花）。")
    print("  '提供方调用'一列与 batches 逐位相同——它不是估算，每一次")
    print("  embed_batch 都被本脚本数过；两个数字相等，batches 才是一份证据。")
    print()
    print("  同一份缓存上再编码一次（全命中对照）：")
    embedding = DemoEmbedding()
    encoder = BatchEncoder(embedding, batch_size=4)
    _first_vectors, first = encoder.encode(texts)
    calls_after_first = embedding.batch_calls
    _second_vectors, second = encoder.encode(texts)
    print(f"    第一次：{first.summary_line()}")
    print(f"    第二次：{second.summary_line()}")
    print(
        f"    提供方调用次数：第一次之后 {calls_after_first}，第二次之后 "
        f"{embedding.batch_calls}（本次新增 {embedding.batch_calls - calls_after_first}）"
    )
    print(f"    缓存状态：{encoder.cache.describe()}")
    print()
    print("  encoded=0 且 batches=0 是怎么做到的（两件事一起）：")
    print("    1. 缓存键 = vector_key(provider, model, dimension, text)，**身份进了键**，")
    print("       所以'同一个编码器 + 同一段文本'必然命中；")
    print("    2. 顺序是**先查缓存、再决定要不要发调用**。反过来（先发一批空调用）")
    print("       在真实 API 上也是一次真实账单——空批不是免费的。")


# --------------------------------------------------------------------------- #
# 3. 差集的四条判据
# --------------------------------------------------------------------------- #


def section_3_plan_criteria() -> None:
    """第 3 节：同一个 parent 清单下的四种差集."""
    title("3. 差集的四条判据：两个身份分别判，任一不同即 updated")
    identity = describe_embedding(DemoEmbedding())
    records = demo_records()
    parent = build_manifest(
        [entry_from_record(record, identity) for record in records],
        identity=identity,
        metric="cosine",
        backend="flat",
        created_at="",
    )
    print("  parent 清单（由 entry_from_record + build_manifest 造出）：")
    print(f"    {parent.summary_line()}")
    print()
    print("  四条判据（顺序不能换：先判'谁不在了'，再分别判两个身份）：")
    print("    previous 里没有这个 id → added")
    print("    fingerprint 变了        → updated（内容变了，向量必须重算）")
    print("    vector_key 变了         → updated（编码器身份或编码文本变了）")
    print("    两者都没变              → unchanged（复用向量，省钱的来源）")
    print()
    changed = with_text(records, ID_FIRST, "缓存键里必须含编码器身份，否则换模型会命中旧向量。")
    other_identity = EmbeddingIdentity(
        provider=identity.provider, model="char-ngram-2-3", dimension=identity.dimension
    )
    cases: tuple[tuple[str, Any], ...] = (
        ("① 未变（原样重放同一批）", plan_index(parent, records, identity)),
        ("② 内容变了（改 1 条原文）", plan_index(parent, changed, identity)),
        ("③ 删了一条（少给最后 1 条）", plan_index(parent, records[:-1], identity)),
        ("④ 换了身份（只改 model）", plan_index(parent, records, other_identity)),
    )
    for label, plan in cases:
        print(f"  {label}")
        print(f"    {plan.summary_line()}")
        print(f"    plan_reason() → {plan.reason}")
        print(
            f"    计数：to_encode={len(plan.to_encode)} / changed={plan.changed} / "
            f"total={plan.total} / reuse_ratio={plan.reuse_ratio}"
        )
        print()
    print("  ④ 那一行是本模块最该被记住的地方：报告上写着'更新 8 条'，")
    print("  但**一条数据都没改**——是编码器身份变了。数字相同、下一步动作完全不同")
    print("  （换模型要重新标定检索阈值，改一条原文不用），所以 plan_reason 必须")
    print("  把这句话说出来：IndexPlan 里只剩四组 id，已经看不出原因了。")
    print()
    print("  另外 ② 与 ③ 的计数是可以重合的：一个是 updated=1/unchanged=7，")
    print("  另一个是 removed=1/unchanged=7——只比总数看不出差异，")
    print("  差别只在**哪些 id 落在哪一组**，这也是 IndexPlan 存 id 而不只存计数的理由。")


# --------------------------------------------------------------------------- #
# 4. 增量构建的省钱证据
# --------------------------------------------------------------------------- #


def section_4_incremental_saving() -> None:
    """第 4 节：首次构建 → 重放同一批（四个数字两两成对）."""
    title("4. 增量构建的省钱证据：重放同一批，一次编码都不发生")
    versions = IndexVersionStore(path="")
    backend = FlatVectorStore(metric="cosine")
    embedding = DemoEmbedding()
    builder = IndexBuilder(backend, embedding, versions=versions)
    records = demo_records()

    first_manifest, first = builder.build(records)
    print(f"  首次构建：{first.summary_line()}")
    print(f"    version_id={first.version_id} / parent={first.parent_version or '（首版）'}")
    print(f"    库 {backend.count()} 条 | {builder.cache.describe()}")
    calls_after_first = embedding.batch_calls
    print()

    second_manifest, second = builder.build(records)
    print(f"  重放同一批：{second.summary_line()}")
    print(f"    version_id={second.version_id} / parent={second.parent_version}")
    print(f"    版本号与首次逐位相同：{second.version_id == first.version_id}")
    print(f"    库仍 {backend.count()} 条 | {builder.cache.describe()}")
    print(
        f"    提供方调用次数：两次构建之间增加了 "
        f"{embedding.batch_calls - calls_after_first} 次"
    )
    print()
    print("  四个数字两两成对，缺一个都会让结论说不完：")
    print(f"    encoded    {first.encoded} → {second.encoded}（第二次一条都没算）")
    print(f"    batches    {first.batches} → {second.batches}（第二次一批都没发）")
    print(f"    written    {first.written} → {second.written}（第二次一条都没写库）")
    print(f"    unchanged  {first.unchanged} → {second.unchanged}（第二次 8 条全都没碰库）")
    print(f"    cache_hits {first.cache_hits} → {second.cache_hits}")
    print("               第二次连缓存都没问：**增量是'不问'，缓存是'问了才省'**——")
    print("               所以'缓存累计命中率 0%'与'这次省下 8 次编码'可以同时为真。")
    print()
    print(f"  版本表：{len(versions)} 版（重放没有产生新版本——版本号由内容算出），")
    print("  current 与两次构建的 version_id 完全相同：")
    print(f"    {versions.current == first.version_id == second.version_id}")
    print()
    print("  '重放不产生新版本'这件事值得单独说一句：版本号由")
    print("  (编码器身份 + 度量 + 后端 + 内容摘要) 算出，四条输入在重放里逐位相同，")
    print("  于是算出来的键也相同；若哪一天有人把时间戳塞进版本号，")
    print("  这一节立刻会多出一个'凭空多出来的版本'。")


# --------------------------------------------------------------------------- #
# 5. 阈值换挡
# --------------------------------------------------------------------------- #


def section_5_threshold() -> None:
    """第 5 节：变更比例 50% 超过阈值 0.1 → 自动从 incremental 切到 full."""
    title("5. 阈值换挡：增量不是永远更省")
    original = settings.indexing_full_rebuild_threshold
    try:
        settings.indexing_full_rebuild_threshold = 0.1
        versions = IndexVersionStore(path="")
        backend = FlatVectorStore(metric="cosine")
        embedding = DemoEmbedding()
        builder = IndexBuilder(backend, embedding, versions=versions)
        records = demo_records()

        first_manifest, first = builder.build(records)
        print(f"  第一次构建（8 条全新，走 {first.mode}）：{first.summary_line()}")
        print()

        changed = records
        for index in range(4):
            changed = with_text(
                changed, DEMO_CHUNKS[index][0], f"第 {index + 1} 条被改写了：阈值换挡的样本。"
            )
        print("  现在改掉其中 4 条原文 → 变更比例 4 / 8 = 50%。")
        print("  请求的模式：mode=None → settings.indexing_mode =")
        print(f"    {settings.indexing_mode!r}（缺省就是增量）")
        print(
            f"  阈值 settings.indexing_full_rebuild_threshold = "
            f"{settings.indexing_full_rebuild_threshold}"
        )
        print()
        plan = builder.plan(changed)
        print(f"  先看计划（**只读**：不编码、不写库、不动版本表）：{plan.summary_line()}")
        print(
            f"    changed={plan.changed} / total={plan.total} → "
            f"比例 {plan.changed / plan.total:.1%} > 阈值 "
            f"{settings.indexing_full_rebuild_threshold:.2f}"
        )
        print(f"    plan 调用后库仍是 {backend.count()} 条、版本表仍是 {len(versions)} 版")
        print()
        manifest, report = builder.build(changed)
        print(f"  实际构建：{report.summary_line()}")
        print(f"    mode = {report.mode!r}（请求的是 incremental，实际走的是 full）")
        print("    report.note 原文：")
        print(f"      {report.note}")
        print()
        print(
            f"  换挡后的四个数字：written={report.written} unchanged={report.unchanged} "
            f"encoded={report.encoded} cache_hits={report.cache_hits} batches={report.batches}"
        )
        print(
            f"  复用率 report.reuse_ratio = {report.reuse_ratio}"
            "（全量下报的是缓存命中率，不是 plan 的复用率）"
        )
        print()
        print("  注意 encoded=4 而不是 8：**换挡换的是'怎么写库'，不是'要不要编码'**——")
        print("  没改的那 4 条仍然靠缓存免掉了编码。全量重建 ≠ 重新编码。")
        print()
        print("  这句话只有装配层说得了：planner 看不到'写库要花多少'，")
        print("  后端也不知道'这批数据相对上一版变了多少'——两边各缺一半。")
        print("  阈值是**严格大于**才换挡：比例正好等于阈值时不换。")
    finally:
        settings.indexing_full_rebuild_threshold = original


# --------------------------------------------------------------------------- #
# 6. 版本表与血缘
# --------------------------------------------------------------------------- #


def section_6_lineage() -> None:
    """第 6 节：三版 + adopt + rollback（沿血缘，不按登记顺序）."""
    title("6. 版本表与血缘：回滚沿血缘，不按登记顺序倒着数")
    versions = IndexVersionStore(path="")
    backend = FlatVectorStore(metric="cosine")
    builder = IndexBuilder(backend, DemoEmbedding(), versions=versions)
    records = demo_records()

    second_text = "第一版之后改写的第一条：回滚要回到的是生效过的版本。"
    third_text = "第二版之后又改写了一次第一条。"
    v1, _ = builder.build(records)
    v2, _ = builder.build(with_text(records, ID_FIRST, second_text))
    v3, _ = builder.build(with_text(records, ID_FIRST, third_text))

    print("  连续构建三版（每版只改一条原文），history() 按**登记顺序**（旧 → 新）：")
    for index, manifest in enumerate(versions.history(), 1):
        print(f"    #{index} {manifest.summary_line()}")
    print(f"    len(versions)={len(versions)} | current={versions.current}（构建成功即采纳）")
    print(f"    v2.lineage(v1) → {v2.lineage(v1)}")
    print(f"    v3.lineage(v1) → {v3.lineage(v1)}")
    print(
        "    血缘链（从 current 回溯，旧 → 新）："
        + " → ".join(manifest.version_id for manifest in versions.lineage(versions.current))
    )
    print()
    print("  现在采纳**第二版**（模拟'第二版被标为生效'，第三版仍然登记着）：")
    adopted = versions.adopt(v2.version_id)
    print(f"    versions.adopt(v2) → current={adopted}")
    print(
        "    血缘链（从 current 回溯）："
        + " → ".join(manifest.version_id for manifest in versions.lineage(versions.current))
    )
    print()
    print("  再回滚一步：")
    rolled = versions.rollback(steps=1)
    print(f"    versions.rollback(steps=1) → current={rolled}")
    print(f"    rolled == v1.version_id → {rolled == v1.version_id}")
    print()
    print("  如果按'登记顺序倒着数'：从最新登记的 v3 往回退一步会得到 v2。")
    print("  而回滚问的是'从**当前生效**的那一版往回看'——current 是 v2，")
    print("  它的 parent_version 是 v1，所以答案是 v1。两个答案都只是一个合法的")
    print("  版本号：没有异常、没有告警，只有下一次检索开始给出另一批结果。")
    print()
    print("  更普遍的反例是'v3 构建失败、从未被采纳'（见 versioning 的 docstring）：")
    print("  那时按时间倒序会回到一个**从没生效过**的版本，而它之所以没被采纳，")
    print("  通常就是因为失败了。所以 lineage() 一步步走 parent_version，")
    print("  遇到断裂或成环都直接报 VersionError，而不是'能走多远走多远'。")
    print()
    print(f"  最后：回滚只改**版本指针**，不动库与数据——库仍然 {backend.count()} 条。")
    print("  要真正重建某一版，请再调用一次 build（回滚是运维动作，不自动重建索引）。")


# --------------------------------------------------------------------------- #
# 7. 备份与保留
# --------------------------------------------------------------------------- #


def section_7_backups() -> None:
    """第 7 节：keep=2 建三份、prune、账本与可用份数、read_manifest 与 restore."""
    title("7. 备份与保留：备份不是日志，保留份数有上限")
    base = WORK_DIR / "backups"
    versions = IndexVersionStore(path="")
    backend = FlatVectorStore(metric="cosine")
    builder = IndexBuilder(backend, DemoEmbedding(), versions=versions)
    backups = IndexBackupStore(path=str(base), keep=2, clock=demo_clock)
    records = demo_records()

    manifests = []
    snapshots: list[str] = []
    current = records
    for index in range(3):
        if index:
            current = with_text(
                current, DEMO_CHUNKS[index][0], f"第 {index + 1} 版改写的原文。"
            )
        manifest, _report = builder.build(current)
        path = backend.persist(str(base / f"store_v{index + 1}.json"))
        manifests.append(manifest)
        snapshots.append(path)
    print(f"  先构建三版，并把每一版的库分别落盘（目录 {rel(base)}）：")
    for index, manifest in enumerate(manifests, 1):
        print(f"    v{index} {manifest.summary_line()}")
        print(f"         快照 {rel(snapshots[index - 1])}")
    print()
    print(
        f"  备份表 keep={backups.keep}"
        f"（配置 settings.indexing_backups_keep={settings.indexing_backups_keep}，"
        f"常量 backup.DEFAULT_KEEP={DEFAULT_KEEP}）"
    )
    print(
        f"  备份记录的 created_at 用**注入的固定时钟** {DEMO_CLOCK}："
        "created_at 不进版本号，钉死它只是让这份输出能逐字节复现"
    )
    print()

    created = [
        backups.create(
            source_files=[snapshots[index]],
            manifest=manifests[index],
            reason=f"第 {index + 1} 版快照",
        )
        for index in range(3)
    ]
    print("  三次 create 返回的 backup_id（version_id + backend + 四位序号）：")
    for record in created:
        print(f"    {record.summary_line()}")
    print()
    print(f"  第三次 create 之后自动 prune，被删掉的是最旧那一份：{created[0].backup_id}")
    print(f"    backups.get(被裁的那份) → {backups.get(created[0].backup_id)}（内存表里没有了）")
    print(f"    (base / backup_id).is_dir() → {(base / created[0].backup_id).is_dir()}")
    print(f"  list() 现在剩 {len(backups.list())} 条（**新 → 旧**）：")
    for record in backups.list():
        print(f"    {record.backup_id}")
    latest = backups.latest()
    print(f"  latest() → {latest.backup_id if latest else None}（目录仍在的那一份）")
    print()
    print("  注意 backup_id 里**没有时间**：它只由 (version_id, backend, 序号) 决定，")
    print("  所以同一份内容备份两次会得到 …-0001 与 …-0002：id 不同、version_id 相同——")
    print("  前者是存储位置，后者才是身份。时间戳只作为记录字段（created_at）给人看。")
    print()

    ledger = base / BACKUP_INDEX_FILE
    appended = [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"  create 是**追加写**账本（{rel(ledger)}）：三份各写一行，")
    print(
        f"  账本文件里现在有 {len(appended)} 行，而内存里的 list() 只有 "
        f"{len(backups.list())} 条——"
    )
    print("  prune 只删目录与内存里的记录，账本那一行留着：账本记的是'曾经发生过什么'，")
    print("  裁剪决定的是'现在还留了什么'，两者不是一回事。")
    available = backups.load()
    print(f"  backups.load() 返回 {available}——**实际可用**份数（目录还在的），不是记录条数")
    print(f"    读回之后 list() 有 {len(backups.list())} 条（账本里那三行都回来了）：")
    print(
        f"      被裁掉的那份 get() 仍取得到 → "
        f"{backups.get(created[0].backup_id) is not None}，"
        "但它的目录不在了，restore 会报 BackupError"
    )
    print("      （'账本里有记录'与'现在还能恢复'是两个问题，latest() 会跳过目录已丢的）")
    print()

    second = created[1]
    count_before = backend.count()
    listed_before = len(backups.list())
    restored = backups.read_manifest(second.backup_id)
    print(f"  read_manifest({second.backup_id}) —— 只看不动：")
    print(f"    {restored.summary_line()}")
    print(f"    version_id 与账本记录一致：{restored.version_id == second.version_id}")
    print(
        f"    只看不动：库仍是 {backend.count()} 条（读之前 {count_before}）、"
        f"账本仍是 {len(backups.list())} 条（读之前 {listed_before}）"
    )
    print()

    target = WORK_DIR / "restored"
    backups.restore(second.backup_id, target_dir=str(target))
    print(f"  restore({second.backup_id}, target_dir=...) → {rel(target)}")
    print("    恢复出来的文件（清单也跟着回去——没有账的库，下一次增量会当它没有上一版）：")
    for item in sorted(target.iterdir()):
        print(f"      {item.name:<18}{item.stat().st_size:>9} B")
    print()
    print("  备份与'持久化快照'不是一回事，这条边界值得写下来：")
    print("    persist() 是**库自己**的落盘，它写出来的永远是'现在这一版'（一份）")
    print("    backup    是**额外的**拷贝，它把库文件与清单**成对**固化下来（若干份）")
    print("  因此'重建把库写坏了'能从一次事故变成一次 restore；而没有保留上限的")
    print("  备份目录会一直长到吃满磁盘——那时正在写库的那次构建会失败，")
    print("  备份把主流程搞挂了，这是备份机制最讽刺的失败方式。")


# --------------------------------------------------------------------------- #
# 8. 一致性体检
# --------------------------------------------------------------------------- #


def section_8_consistency() -> None:
    """第 8 节：人为删掉一条 → compare_with_store 与 verify_index 的结论."""
    title("8. 一致性体检：清单与库不一致时，症状必须当场被说出来")
    backend = FlatVectorStore(metric="cosine")
    builder = IndexBuilder(backend, DemoEmbedding())
    records = demo_records()
    manifest, report = builder.build(records)
    print(f"  构建一版：{report.summary_line()}")
    victim = demo_records()[3]["doc_id"]
    removed = backend.delete(ids=[victim])
    print(
        f"  现在绕过索引流程，直接从库里删掉一条（id={victim!r}），"
        f"delete 返回**真正删掉**的条数 = {removed}"
    )
    print()
    comparison = compare_with_store(manifest, backend)
    print("  compare_with_store(manifest, backend)：")
    for key in (
        "missing_in_store",
        "orphan_in_store",
        "count_manifest",
        "count_store",
        "dimension_match",
        "metric_match",
        "backend_match",
    ):
        print(f"    {key:<18}= {comparison[key]}")
    print(f"    extra_tokens      = {comparison['extra_tokens']}")
    print(f"    tokens_match      = {comparison['tokens_match']}")
    print()
    verdict = verify_index(manifest, backend)
    print(f"  verify_index(manifest, backend) → ok={verdict['ok']}")
    print(f"    checks = {verdict['checks']}")
    print(f"    problems 原文（共 {len(verdict['problems'])} 条）：")
    for problem in verdict["problems"]:
        print(f"      {problem}")
    print()
    print("  拿'没有清单'的 verify() 对照一次（builder.verify() 无参 → 从库现算一份）：")
    fresh = builder.verify()
    print(f"    ok={fresh['ok']}、problems={fresh['problems']}")
    print("    它必然为真：清单是从库现算的，两者不可能对不上。这正是")
    print("    manifest_from_store 的说明里那句——重建出来的清单是'从现在起'的基线，")
    print("    不是历史的那一版。所以'库里被人删了一条'只有拿**原来那份清单**")
    print("    来对账才看得出来。")
    print()
    print("  四类不一致各自的表现与处置（problems 是一句能直接照做的话）：")
    print("    清单有库没有 → 点名 id：上一次构建可能半途失败，先重建清单，别做增量")
    print("    库有清单没有 → 点名 id：它们不会被增量更新覆盖，同样先重建清单")
    print("    维度不符     → 这份清单描述的是**另一个编码器**建的库：")
    print("                   所有 vector_key 都会变，'对不上'会被伪装成'数据全变了'")
    print("    度量不符     → 用清单的度量新建实例（metric=清单里那个），")
    print("                   而不是让一个已有实例改度量")
    print("    后端不符     → 换后端可以重建，但不能与旧版共用一份清单")
    print()
    print("  token_count 单独看：它不进内容摘要（见 types.entries_digest），")
    print("  改一个统计字段不需要重建索引，所以 tokens_match 为假不会让 ok 变假——")
    print("  一个过度敏感的检查项比一个不准的更麻烦。")


# --------------------------------------------------------------------------- #
# 输出：同时写终端与文件
# --------------------------------------------------------------------------- #


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
    """按节运行演示，并把完整输出同时写入 ``outputs/indexing_demo.txt``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 空仓起步：上一次运行留下的备份目录会让"这次 create 出了几份"不可复现。
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    original = sys.stdout
    with output_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            for section in (
                section_1_identity,
                section_2_batching,
                section_3_plan_criteria,
                section_4_incremental_saving,
                section_5_threshold,
                section_6_lineage,
                section_7_backups,
                section_8_consistency,
            ):
                section()
            print(f"\n演示完成（工作目录 {Path.cwd()}）")
            print(f"本节输出已同时写入 {rel(output_path)}")
            print(f"落盘产物（快照 / 备份 / 恢复）在 {rel(WORK_DIR)}")
        finally:
            sys.stdout = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
