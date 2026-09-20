"""分块策略的检索效果评估：用你自己的查询集给四种策略打分（M6-D2）.

课程目标里写着"评估不同分块对检索效果的影响"。这句话很容易被做成
"跑一遍看一眼块长"，但那样得出的结论没有约束力——**块长好看不等于检索得准**。
本模块把它做成一个可复现的小型评估：

```text
1. 准备探针集        (question, expect) 的列表，expect 是期望命中的原文片段
2. 建索引            按某策略把文档切块 → 每块用 retrieval_text 编码 → 入内存向量库
3. 检索              question 编码后取 top_k 块
4. 判定              某个块的**原文**里含 expect 吗？含 → 命中，记下它的排名
5. 汇总              hit@k、MRR、命中块的平均字符数、索引规模
```

## 为什么这条指标能反映"分块质量"

判定用的是**块文本里是否含期望片段**，于是它同时测了两件事：

```text
检索到没有              → 向量召回能力
期望片段有没有被切开    → 块边界质量
```

第二件是本模块最有价值的发现通道。假设用户问"阈值该设多少"，期望片段是
"相似度低于 0.35 的直接丢掉"。如果某个策略**恰好把这句话从中间切开**，
那么它在任何块里都不完整，**该探针在该策略下必然 miss**——
无论向量检索多准。这类 miss 用"平均块长"是永远看不出来的。

## 探针集的三条纪律（比指标公式重要）

| 纪律 | 违反的后果 |
|------|-----------|
| 从**你自己的查询分布**里取，不能抄别人的 | 打分出来的最优策略只对别人有效 |
| 期望片段在原文里**唯一**（用 ``ambiguous_probes`` 自查） | 多次出现时命中判定失去意义 |
| 覆盖不同长度与类型（疑问句、命令、名词短语） | 只测长问句会系统性偏向"块要大" |

## 与 day071 的分工

本模块评估的是**检索层**（分块 → 召回），指标是 hit@k / MRR / 块粒度；
day071 的 RAG 评估要接上生成（召回质量 → 答案质量 → 引用准确率）。
**分层的理由**：生成环节引入 LLM 的不确定性与成本，先把检索层测准，
才谈得上判断"是检索没找到还是模型没用上"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.chunking.base import ChunkPolicy
from smart_research_agent.chunking.pipeline import ChunkPipeline
from smart_research_agent.chunking.tokens import TokenMeasurer
from smart_research_agent.chunking.types import STRATEGIES, ChunkSet
from smart_research_agent.documents.types import Document
from smart_research_agent.llm.embedding import EmbeddingProvider, default_embedding
from smart_research_agent.memory.vector_store import InMemoryVectorStore, VectorRecord
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 默认的检索深度。3 是刻意的：RAG 的实际链路里，top-3 之外的内容
#: 大概率会被提示词长度或模型的注意力稀释掉，**只看 top-3 相当于
#: 用"真实会被用上的块"来打分**。
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RetrievalProbe:
    """一条探针：一个问题 + 期望命中的原文片段."""

    question: str
    expect: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("探针必须有 question：它就是检索时输入的那句话")
        if not self.expect.strip():
            raise ValueError(
                "探针必须有 expect：没有期望片段就无法判定命中，"
                "而「人工看一眼觉得还行」不是一条可复现的判据"
            )


@dataclass(frozen=True)
class ProbeHit:
    """一条探针在某策略下的判定结果."""

    question: str
    expect: str
    hit: bool
    rank: int = 0
    score: float = 0.0
    chunk_id: str = ""
    chunk_chars: int = 0

    @property
    def reciprocal_rank(self) -> float:
        """RR = 1/排名（未命中记 0）."""
        return 1.0 / self.rank if self.hit and self.rank else 0.0

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "question": self.question,
            "expect": self.expect,
            "hit": self.hit,
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "chunk_chars": self.chunk_chars,
            "reciprocal_rank": round(self.reciprocal_rank, 4),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（miss 与低排名各有一个前缀）."""
        mark = "命中" if self.hit else "缺失"
        if self.hit and self.rank > 1:
            mark = f"第{self.rank}位"
        return (
            f"[{mark}] {self.question} | 期望 {len(self.expect)} 字 | "
            f"相似度 {self.score:.4f} | 命中块 {self.chunk_chars} 字"
        )


@dataclass(frozen=True)
class StrategyScore:
    """一个策略在探针集上的得分（**检索质量与索引代价并列**）."""

    strategy: str
    hits: tuple[ProbeHit, ...] = ()
    chunks: int = 0
    total_tokens: int = 0
    duplication_ratio: float = 0.0
    oversized_count: int = 0
    top_k: int = DEFAULT_TOP_K

    @property
    def probes(self) -> int:
        """探针总数."""
        return len(self.hits)

    @property
    def hit_count(self) -> int:
        """命中的条数."""
        return sum(1 for hit in self.hits if hit.hit)

    @property
    def hit_rate(self) -> float:
        """hit@k：命中的探针占比."""
        return round(self.hit_count / self.probes, 4) if self.probes else 0.0

    @property
    def mrr(self) -> float:
        """MRR：平均倒数排名（**比 hit@k 更细**：命中在第 1 位与第 3 位都算命中，
        但 RAG 链路里第 1 位命中更省 token、更不容易被噪声带偏）."""
        if not self.hits:
            return 0.0
        return round(sum(hit.reciprocal_rank for hit in self.hits) / len(self.hits), 4)

    @property
    def avg_hit_chars(self) -> float:
        """命中块的平均字符数（**越小越"精确"，但太小意味着上下文不足**）."""
        sizes = [hit.chunk_chars for hit in self.hits if hit.hit]
        return round(sum(sizes) / len(sizes), 1) if sizes else 0.0

    def missed(self) -> list[ProbeHit]:
        """未命中的探针（bad case 的入口：先看是没检索到，还是片段被切开了）."""
        return [hit for hit in self.hits if not hit.hit]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "strategy": self.strategy,
            "probes": self.probes,
            "hit_count": self.hit_count,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "avg_hit_chars": self.avg_hit_chars,
            "chunks": self.chunks,
            "total_tokens": self.total_tokens,
            "duplication_ratio": self.duplication_ratio,
            "oversized_count": self.oversized_count,
            "top_k": self.top_k,
            "misses": [hit.to_dict() for hit in self.missed()],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.strategy:<10} | hit@{self.top_k} {self.hit_rate:.1%} | "
            f"MRR {self.mrr:.3f} | 命中块 {self.avg_hit_chars:.0f} 字 | "
            f"{self.chunks} 块 / {self.total_tokens} tokens | "
            f"重复 {self.duplication_ratio:.1%}"
        )


@dataclass(frozen=True)
class ChunkingEvaluation:
    """一次分块策略对照评估的完整结果."""

    scores: tuple[StrategyScore, ...] = ()
    top_k: int = DEFAULT_TOP_K
    embedding: str = ""
    ambiguous_probes: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def strategies(self) -> tuple[str, ...]:
        """被评估的策略（按 ``STRATEGIES`` 的顺序）."""
        present = {score.strategy for score in self.scores}
        return tuple(name for name in STRATEGIES if name in present)

    def best(self, metric: str = "mrr") -> StrategyScore:
        """按某个指标取最优策略（并列时按 ``STRATEGIES`` 的顺序，保证确定）."""
        if not self.scores:
            raise ValueError("没有可比较的策略：至少要有一次评估结果")
        if metric not in {"mrr", "hit_rate", "avg_hit_chars", "total_tokens"}:
            raise ValueError(
                f"未知指标 {metric!r}，可选 mrr / hit_rate / avg_hit_chars / total_tokens"
            )
        # avg_hit_chars 与 total_tokens 是"越小越好"，其余是"越大越好"；
        # 方向必须在这里分开写：**同一个排序函数处理两个方向的指标，
        # 一定会有一天把成本指标当成质量指标排。**
        lower_is_better = metric in {"avg_hit_chars", "total_tokens"}
        ordered = sorted(
            self.scores,
            key=lambda score: (
                getattr(score, metric) if lower_is_better else -getattr(score, metric),
                STRATEGIES.index(score.strategy),
            ),
        )
        return ordered[0]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "top_k": self.top_k,
            "embedding": self.embedding,
            "ambiguous_probes": list(self.ambiguous_probes),
            "metadata": dict(self.metadata),
            "strategies": [score.to_dict() for score in self.scores],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if not self.scores:
            return "未评估任何策略"
        best = self.best()
        return (
            f"{len(self.scores)} 个策略 × {best.probes} 条探针（top-{self.top_k}）| "
            f"MRR 最优：{best.strategy}（{best.mrr:.3f}）| "
            f"embedding {self.embedding}"
            + (f" | 歧义探针 {len(self.ambiguous_probes)}" if self.ambiguous_probes else "")
        )

    def render_markdown(self) -> str:
        """把对照表渲染成 markdown（可直接贴进选型记录）."""
        lines = [
            "# 分块策略检索评估",
            "",
            f"- 探针数：{self.scores[0].probes if self.scores else 0}",
            f"- 检索深度：top-{self.top_k}",
            f"- embedding：{self.embedding}",
            "",
            "| 策略 | hit@k | MRR | 命中块均长 | 块数 | 预算单位 | 重复率 | 超预算 |",
            "|------|-------|-----|-----------|------|---------|--------|--------|",
        ]
        lines.extend(
            f"| `{score.strategy}` | {score.hit_rate:.1%} | {score.mrr:.3f} | "
            f"{score.avg_hit_chars:.0f} | {score.chunks} | {score.total_tokens} | "
            f"{score.duplication_ratio:.1%} | {score.oversized_count} |"
            for score in self.scores
        )
        if self.ambiguous_probes:
            lines.extend(
                ["", "## 歧义探针（期望片段在原文里出现多次，判定不可靠）", ""]
            )
            lines.extend(f"- {item}" for item in self.ambiguous_probes)
        misses = [
            (score.strategy, hit) for score in self.scores for hit in score.missed()
        ]
        if misses:
            lines.extend(["", "## 未命中明细（先看片段是不是被切开了）", ""])
            lines.extend(
                f"- `{strategy}` {hit.question} → 期望「{hit.expect[:30]}…」"
                for strategy, hit in misses
            )
        lines.append("")
        return "\n".join(lines)


def ambiguous_probes(document: Document, probes: Sequence[RetrievalProbe]) -> list[str]:
    """找出"期望片段在原文里出现多次"的探针（**探针自查**）.

    一个出现多次的期望片段仍然会让"某个块含它"成立，但命中判定再也
    回答不了"命中的是不是我们想问的那一处"——**评分会虚高，而且虚高得
    毫不显眼**。因此它必须被单独列出来，而不是混进总分。
    """
    findings: list[str] = []
    for probe in probes:
        occurrences = document.text.count(probe.expect)
        if occurrences > 1:
            findings.append(
                f"{probe.question} → 期望片段在原文中出现 {occurrences} 次"
            )
    return findings


def build_index(
    chunk_set: ChunkSet, embedding: EmbeddingProvider
) -> InMemoryVectorStore:
    """把一个块集建成内存向量库（**编码的是 ``retrieval_text``**）.

    编码带面包屑的检索视图而不是原始片段：查询"缓存相似度怎么配"里
    没有"缓存"这个词出现在片段本体时，标题路径是唯一的桥。这一条与
    ``ChunkSet.knowledge_records()`` 里放进 ``metadata.retrieval_text``
    是同一个决定，**两处必须一致**，否则评估出来的分数对应的不是
    线上真正会用的那份文本。
    """
    store = InMemoryVectorStore()
    texts = [chunk.retrieval_text for chunk in chunk_set.chunks]
    vectors = embedding.embed_batch(texts)
    for chunk, vector in zip(chunk_set.chunks, vectors):
        store.add(
            VectorRecord(
                id=chunk.chunk_id,
                text=chunk.text,
                vector=vector,
                metadata={"index": chunk.index, "heading_path": chunk.heading_text},
            )
        )
    return store


def score_probes(
    chunk_set: ChunkSet,
    probes: Sequence[RetrievalProbe],
    embedding: EmbeddingProvider,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> StrategyScore:
    """一个块集 + 一组探针 → 一份得分（含逐条明细）."""
    if top_k < 1:
        raise ValueError(f"top_k 必须至少为 1，收到 {top_k}")
    store = build_index(chunk_set, embedding)
    query_vectors = embedding.embed_batch([probe.question for probe in probes])
    hits: list[ProbeHit] = []
    for probe, query_vector in zip(probes, query_vectors):
        ranked = store.search(query_vector, top_k=top_k)
        hit = ProbeHit(question=probe.question, expect=probe.expect, hit=False)
        for rank, (record, score) in enumerate(ranked, start=1):
            if probe.expect in record.text:
                hit = ProbeHit(
                    question=probe.question,
                    expect=probe.expect,
                    hit=True,
                    rank=rank,
                    score=score,
                    chunk_id=record.id,
                    chunk_chars=len(record.text),
                )
                break
        hits.append(hit)
    return StrategyScore(
        strategy=chunk_set.strategy,
        hits=tuple(hits),
        chunks=chunk_set.count,
        total_tokens=chunk_set.total_tokens,
        duplication_ratio=chunk_set.duplication_ratio,
        oversized_count=chunk_set.oversized_count,
        top_k=top_k,
    )


def evaluate_strategies(
    document: Document,
    probes: Sequence[RetrievalProbe],
    *,
    strategies: Sequence[str] | None = None,
    policies: dict[str, ChunkPolicy] | None = None,
    top_k: int = DEFAULT_TOP_K,
    embedding: EmbeddingProvider | None = None,
    measurer: TokenMeasurer | None = None,
) -> ChunkingEvaluation:
    """对同一份文档跑多个分块策略并给出检索得分（**一遍出表**）.

    确定性来自两个注入点：``measurer``（默认 chars）与 ``embedding``
    （默认 MockEmbedding，**没有真实语义**）。默认用 Mock 是刻意的：
    评估脚本不该因为"这台机器没配 embedding 密钥"就跑不起来；
    但要记住一条——**MockEmbedding 的相似度接近随机**，
    此时策略之间的差异主要来自"片段是否被切开"与"标题面包屑"，
    而不是语义召回。想比较语义召回能力，必须注入真实提供方。
    """
    provider = embedding or default_embedding()
    pipeline = ChunkPipeline(measurer=measurer, embedding=provider)
    report = pipeline.chunk_all(
        document, strategies=strategies, policies=policies
    )
    scores = tuple(
        score_probes(chunk_set, probes, provider, top_k=top_k)
        for chunk_set in report.sets
    )
    evaluation = ChunkingEvaluation(
        scores=scores,
        top_k=top_k,
        embedding=type(provider).__name__,
        ambiguous_probes=tuple(ambiguous_probes(document, probes)),
        metadata={"document": document.source, "doc_id": document.fingerprint},
    )
    logger.info("分块评估完成：%s", evaluation.summary_line())
    return evaluation


__all__ = [
    "DEFAULT_TOP_K",
    "ChunkingEvaluation",
    "ProbeHit",
    "RetrievalProbe",
    "StrategyScore",
    "ambiguous_probes",
    "build_index",
    "evaluate_strategies",
    "score_probes",
]
