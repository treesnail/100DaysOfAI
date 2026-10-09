#!/usr/bin/env python
"""day062 演示脚本：分块策略（M6-D2）.

八节，全部**离线、确定性、零第三方依赖、零网络**：

```text
1. 预算单位是谁               chars 与 tiktoken 的差别，以及"确定性"为什么要写下来
2. 固定长度                   窗口算术（块数、放大量）、重叠的代价
3. 递归                       分隔符直方图、降级路径、合并回预算
4. 结构                       标题栈、breadcrumb、原子块保护、退化路径
5. 语义                       自然段相似度、分位数阈值、它唯一的额外成本
6. 四个策略并排               同一份文档的报告对照（规模 / 覆盖 / 重复 / 超预算）
7. 用探针集打分                hit@k / MRR / 命中块均长，以及"检索深度"的影响
8. 四个 HTTP 端点              进程内 ASGI 调用，不写盘、不遍历目录
```

运行::

    PYTHONPATH=. python scripts/chunking_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from smart_research_agent.chunking import (
    CHUNKING_LIMITATIONS,
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_SEMANTIC,
    STRATEGY_STRUCTURAL,
    ChunkPipeline,
    ChunkPolicy,
    RetrievalProbe,
    build_chunker,
    chunking_boundaries,
    default_policy,
    evaluate_strategies,
    resolve_measurer,
    separator_histogram,
    window_plan,
)
from smart_research_agent.documents import BLOCK_CODE, default_registry
from smart_research_agent.llm.embedding import MockEmbedding

DOCUMENT = """# 检索手册

本文给出一套可复现的检索参数，所有数字都在同一台机器上实测。

## 分块

固定长度分块每 320 个字符切一刀，重叠 48 个字符。它不认识标点，因此一定会切在句子中间，
而重叠只能保证被切断的句子在至少一块里是完整的。

递归分块优先在段落边界切开，其次换行，然后是句号与逗号。中文标点必须在分隔符表里，
否则一份中文文档会一路退到按字符硬切，而那种退化不会有任何报错。

```python
def plan(doc_chars: int, max_tokens: int, overlap_tokens: int) -> int:
    step = max_tokens - overlap_tokens
    if doc_chars <= max_tokens:
        return 1
    return -(-(doc_chars - max_tokens) // step) + 1
```

## 检索

默认返回前 5 条，相似度低于 0.35 的直接丢掉。阈值用分位数标定，换 embedding 模型必须重新标定。

混合检索把向量分数与 BM25 分数做归一化后加权，权重 0.6 / 0.4。

## 成本

一次全量重建索引大约 12 万条记录，按每条 320 token 计，embedding 花费约 3.8 美元；
而重叠 15% 意味着其中约 1.8 万条是重复内容，这笔钱换来的只是边界处不丢句子。
"""

PROBES = [
    RetrievalProbe(question="分块预算怎么定", expect="固定长度分块每 320 个字符切一刀"),
    RetrievalProbe(question="相似度阈值设多少", expect="相似度低于 0.35 的直接丢掉"),
    RetrievalProbe(question="重叠多花多少钱", expect="重叠 15% 意味着其中约 1.8 万条是重复内容"),
]


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load_document():  # type: ignore[no-untyped-def]
    """把演示用的 Markdown 交给 day061 的真实加载器（而不是手工造 Document）."""
    return default_registry().load_bytes(
        DOCUMENT.encode("utf-8"), source="docs/guide.md", media_type="text/markdown"
    )


def pipeline() -> ChunkPipeline:
    """离线编排器：字符度量器 + MockEmbedding（完全确定性）."""
    return ChunkPipeline(measurer=resolve_measurer("chars"), embedding=MockEmbedding())


# --------------------------------------------------------------------------- #
# 1. 预算单位
# --------------------------------------------------------------------------- #


def section_1_measurers() -> None:
    """第 1 节：预算单位是谁."""
    title("1. 预算单位：max_tokens = 320 到底是多少？")
    text = "分块预算决定索引规模。"
    for name in ("chars", "tiktoken"):
        try:
            measurer = resolve_measurer(name)
        except Exception as exc:  # noqa: BLE001 - 演示脚本要把"不可用"打出来
            print(f"  {name:<9} 不可用：{exc}")
            continue
        print(
            f"  {name:<9} 这段话 = {measurer.count(text):>3} 个预算单位"
            f"（{len(text)} 个字符）"
        )
    print()
    print("  默认是 chars：**分块结果的可复现性比'与计费单位一致'更重要**——")
    print("  tiktoken 依赖本机词表缓存，换台机器可能切出另一套 chunk_id，")
    print("  而 chunk_id 是知识库的去重键。")


# --------------------------------------------------------------------------- #
# 2. 固定长度
# --------------------------------------------------------------------------- #


def section_2_fixed() -> None:
    """第 2 节：固定长度与窗口算术."""
    title("2. 固定长度：参数一确定，块数与成本就能算出来")
    document = load_document()
    policy = default_policy(STRATEGY_FIXED)
    plan = window_plan(len(document.text), policy)
    print(
        f"  参数：max={policy.max_tokens} overlap={policy.overlap_tokens} "
        f"step={plan['step']} | 原文 {plan['doc_units']} 字符"
    )
    print(
        f"  预估：{plan['chunks']} 块 / 重叠 {plan['overlap_units']} 字符 / "
        f"放大量 {plan['amplification']:.2%}"
    )
    result = build_chunker(STRATEGY_FIXED, policy, measurer=resolve_measurer("chars")).split(
        document
    )
    print(f"  实测：{result.summary_line()}")
    print(f"  覆盖率：{result.coverage:.2%}（差额来自块首尾被修剪的空白）")
    print()
    print("  每一块的前 24 个字符（注意它**不认识标点**）：")
    for chunk in result.chunks:
        print(f"    #{chunk.index} [{chunk.start_char}:{chunk.end_char}] {chunk.text[:24]!r}")
    print()
    print("  把预算调小一半，代价按比例变化：")
    for max_tokens in (320, 160, 80):
        scaled = policy.replace(max_tokens=max_tokens, overlap_tokens=max_tokens * 3 // 20)
        scaled_plan = window_plan(len(document.text), scaled)
        print(
            f"    max={max_tokens:>3} overlap={scaled.overlap_tokens:>2} → "
            f"{scaled_plan['chunks']:>2} 块，放大量 {scaled_plan['amplification']:.1%}"
        )


# --------------------------------------------------------------------------- #
# 3. 递归
# --------------------------------------------------------------------------- #


def section_3_recursive() -> None:
    """第 3 节：递归与分隔符表."""
    title("3. 递归：分隔符优先级表 + 合并回预算")
    document = load_document()
    print("  这份文档里每种分隔符出现多少次（count=0 的那一档就是从这层掉下去的原因）：")
    rows = separator_histogram(document.text)
    for row in rows[:8]:
        print(f"    {row['label']:<16} {row['label']!r:<10} → {row['count']}")
    result = build_chunker(
        STRATEGY_RECURSIVE,
        default_policy(STRATEGY_RECURSIVE),
        measurer=resolve_measurer("chars"),
    ).split(document)
    print(f"\n  默认参数：{result.summary_line()}")
    print("  逐块（reason 里写着这一刀切在哪一档分隔符上）：")
    for chunk in result.chunks:
        print(
            f"    #{chunk.index} {chunk.reason:<16} {chunk.token_count:>4} 字 "
            f"{chunk.text[:28]!r}"
        )
    print()
    print("  一份没有任何分隔符的日志（退化到字符硬切）：")
    log = default_registry().load_bytes(
        ("x" * 200).encode("utf-8"), source="logs/plain.log", media_type="text/plain"
    )
    degraded = build_chunker(
        STRATEGY_RECURSIVE,
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=64),
        measurer=resolve_measurer("chars"),
    ).split(log)
    print(f"    {degraded.summary_line()}")
    print(f"    reason：{[chunk.reason for chunk in degraded.chunks]}")


# --------------------------------------------------------------------------- #
# 4. 结构
# --------------------------------------------------------------------------- #


def section_4_structural() -> None:
    """第 4 节：标题栈与原子块保护."""
    title("4. 结构：标题层级 + breadcrumb + 原子块保护")
    document = load_document()
    result = build_chunker(
        STRATEGY_STRUCTURAL,
        default_policy(STRATEGY_STRUCTURAL),
        measurer=resolve_measurer("chars"),
    ).split(document)
    print(f"  {result.summary_line()}")
    print(
        f"  小节数 {result.metadata['sections']} / "
        f"定位失败的 {result.metadata['unlocated_sections']}"
    )
    for chunk in result.chunks:
        path = chunk.heading_text or "（前言）"
        print(
            f"    #{chunk.index} [{chunk.start_char:>3}:{chunk.end_char:>3}] "
            f"{chunk.token_count:>4} 字 | 块 {chunk.start_block}~{chunk.end_block} | {path}"
        )
    print()
    print("  breadcrumb 进的是检索视图而不是正文：")
    # 刻意挑一个"正文不是自己标题"的块（代码块）：结构策略里每一块都以
    # 所属小节的标题行开头，只有代码块/表格这类块看不到那一行——
    # 而它的检索视图里仍然带着完整的路径。
    target = next(
        chunk
        for chunk in result.chunks
        if chunk.heading_path and not chunk.text.startswith(chunk.heading_path[-1])
    )
    print(f"    heading_path   : {target.heading_text!r}")
    print(f"    text           : {target.text.splitlines()[0]!r}")
    print(f"    retrieval_text : {target.retrieval_text.splitlines()[0]!r}")
    print()
    code = next(block for block in document.blocks if block.kind == BLOCK_CODE)
    print(f"  代码块 {len(code.text)} 字符。把预算压到 40 之后：")
    tight = build_chunker(
        STRATEGY_STRUCTURAL,
        ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=40),
        measurer=resolve_measurer("chars"),
    ).split(document)
    for chunk in tight.chunks:
        if chunk.oversized:
            print(
                f"    ! #{chunk.index} {chunk.token_count} 字（reason={chunk.reason}）"
                f"—— 整块保留，**不切出残码**"
            )
    print(f"    超预算块数：{tight.oversized_count}")
    print()
    print("  同一条边界写在三个地方（元数据 / 块的标记 / 报告），排查时都能撞到它。")


# --------------------------------------------------------------------------- #
# 5. 语义
# --------------------------------------------------------------------------- #


def section_5_semantic() -> None:
    """第 5 节：语义切分与分位数阈值."""
    title("5. 语义：相邻自然段相似度最低处切一刀")
    document = load_document()
    result = build_chunker(
        STRATEGY_SEMANTIC,
        default_policy(STRATEGY_SEMANTIC),
        measurer=resolve_measurer("chars"),
        embedding=MockEmbedding(),
    ).split(document)
    print(f"  {result.summary_line()}")
    print(f"  这一次是怎么决定的：{result.metadata}")
    print()
    print("  分位数是一个**相对**判据：同一个阈值在任何提供方上都表示'最低的 25%'")
    for fraction in (0.1, 0.25, 0.5, 0.75):
        scaled = build_chunker(
            STRATEGY_SEMANTIC,
            default_policy(STRATEGY_SEMANTIC, similarity_percentile=fraction),
            measurer=resolve_measurer("chars"),
            embedding=MockEmbedding(),
        ).split(document)
        print(
            f"    percentile={fraction:<5} → {scaled.count:>2} 块 | "
            f"阈值 {scaled.metadata['threshold']:>8} | 断层 {scaled.metadata['breaks']}"
        )
    print()
    print("  它的额外成本：每切一次要多付 N 次 embedding（N = 自然段数）。")
    print("  它的软肋：只认识自然段，会把含空行的代码块切开。")


# --------------------------------------------------------------------------- #
# 6. 四个策略并排
# --------------------------------------------------------------------------- #


def section_6_compare() -> None:
    """第 6 节：四个策略同一份文档的报告对照."""
    title("6. 四个策略并排：规模 / 覆盖 / 重复 / 超预算")
    document = load_document()
    report = pipeline().chunk_all(document)
    print(f"  {report.summary_line()}")
    print()
    print(
        f"  {'策略':<12}{'块数':>5}{'平均':>7}{'覆盖率':>9}{'重复率':>9}{'超预算':>7}"
    )
    for row in report.strategy_summary():
        print(
            f"  {row['strategy']:<12}{row['chunks']:>5}{row['avg_tokens']:>7.1f}"
            f"{row['coverage']:>9.2%}{row['duplication_ratio']:>9.2%}"
            f"{row['oversized_count']:>7}"
        )
    print()
    print("  三列各答一个问题：")
    print("    - 块数 / 平均长度 → 索引有多大、embedding 要花多少")
    print("    - 重复率          → 重叠白花了多少（结构策略恒为 0）")
    print("    - 超预算          → 有多少块注定被模型截断（结构策略保护原子块的代价）")
    print()
    print("  用结构策略 + 40 的预算看一眼超预算的块：")
    tight = pipeline().chunk_all(
        document,
        strategies=[STRATEGY_STRUCTURAL],
        policies={STRATEGY_STRUCTURAL: ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=40)},
    )
    for chunk in tight.oversized():
        print(f"    ! {chunk.source} #{chunk.index} {chunk.token_count} 字（{chunk.reason}）")


# --------------------------------------------------------------------------- #
# 7. 探针集打分
# --------------------------------------------------------------------------- #


def section_7_evaluate() -> None:
    """第 7 节：用探针集给四个策略打分."""
    title("7. 用你自己的查询集打分：hit@k / MRR / 命中块均长")
    document = load_document()
    print("  探针集（期望片段必须是原文里逐字存在的片段）：")
    for probe in PROBES:
        print(f"    {probe.question:<16} → {probe.expect!r}")
    print()
    print("  top-3（真实链路里大多数内容会被提示词长度稀释掉）：")
    narrow = evaluate_strategies(
        document, PROBES, top_k=3, embedding=MockEmbedding()
    )
    print(f"    {narrow.summary_line()}")
    for score in narrow.scores:
        print(f"    {score.summary_line()}")
    print()
    print("  top-8（覆盖全部块）：分数只反映'期望片段有没有被完整保留'")
    wide = evaluate_strategies(document, PROBES, top_k=8, embedding=MockEmbedding())
    for score in wide.scores:
        print(f"    {score.summary_line()}")
    print()
    print("  两条纪律：探针要从**你自己的查询分布**里取；期望片段在原文里必须唯一")
    print("  （出现多次的探针会让分数虚高，而且毫不显眼——用 ambiguous_probes 自查）。")
    print()
    print("  注意：这里用的是 MockEmbedding（**没有真实语义**），因此 top-3 的差距里")
    print("  混进了检索的运气。要比较语义召回能力，必须注入真实的 embedding 提供方。")


# --------------------------------------------------------------------------- #
# 8. 端点
# --------------------------------------------------------------------------- #


def section_8_api() -> None:
    """第 8 节：四个端点."""
    title("8. 四个端点：只读或纯计算，不写盘、不遍历目录")
    from fastapi.testclient import TestClient

    from smart_research_agent.api.app import create_app

    client = TestClient(create_app(embedding=MockEmbedding()))
    described = client.get("/chunking/strategies").json()
    print(
        f"  GET  /chunking/strategies → 策略 {len(described['strategies'])} 个 / "
        f"度量器 {len(described['measurers'])} 个 / "
        f"分隔符 {len(described['separators'])} 档 / 限制 {len(described['limitations'])} 条"
    )
    split = client.post(
        "/chunking/split",
        json={"filename": "guide.md", "text": DOCUMENT, "strategy": STRATEGY_FIXED},
    ).json()
    print(f"  POST /chunking/split      → {split['summary']}")
    plan = split["plan"]
    print(
        f"       计划：{plan['chunks']} 块（精确={plan['estimated_chunks_exact']}）"
        f" | 实测覆盖率 {split['stats']['coverage']:.2%}"
    )
    records = client.post(
        "/chunking/records",
        json={"filename": "guide.md", "text": DOCUMENT, "strategy": STRATEGY_STRUCTURAL},
    ).json()
    print(
        f"  POST /chunking/records    → {records['summary']} | "
        f"{len(records['records'])} 条记录 | 向量化字段 {records['embedding_input']}"
    )
    evaluation = client.post(
        "/chunking/evaluate",
        json={
            "filename": "guide.md",
            "text": DOCUMENT,
            "probes": [probe.__dict__ for probe in PROBES],
            "top_k": 8,
        },
    ).json()
    print(f"  POST /chunking/evaluate   → {evaluation['summary']}")
    print()
    boundaries = chunking_boundaries()
    print("  覆盖不变量：")
    print(f"    {boundaries['coverage_invariant']}")
    print("  三条最该带走的限制：")
    for item in CHUNKING_LIMITATIONS[:3]:
        print(f"    - {item[:66]}…")
    print()
    print(f"  四种策略：{', '.join(STRATEGIES)}")
    print(f"  下一步：{boundaries['next_step']}")


def main() -> int:
    """按节运行演示（任何一节失败都返回非零退出码）."""
    for section in (
        section_1_measurers,
        section_2_fixed,
        section_3_recursive,
        section_4_structural,
        section_5_semantic,
        section_6_compare,
        section_7_evaluate,
        section_8_api,
    ):
        section()
    print(f"\n演示完成（工作目录 {Path.cwd()}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
