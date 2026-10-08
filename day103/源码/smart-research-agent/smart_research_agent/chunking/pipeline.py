"""分块编排：把"文档 + 策略"变成"块 + 可复核的报告"（M6-D2）.

前面的模块各自负责一件事：``base`` 管收尾、四个 ``*_chunker`` 管"在哪切"。
本模块是它们的**编排层**，回答三个批量才出现的问题：

```text
1. 一次切多份文档      每份文档各自成集，报告按文档 × 策略汇总
2. 同一份文档比多个策略 四个策略切同一份文档，产出可直接对照的表
3. 交出去的是什么       knowledge_records()：day009 的 VectorStore 形状
```

## 与 day061 的接缝在这一天收口

day061 的 ``IngestReport.knowledge_records()`` 交出去的是**整份文档**，
今天交出去的是**文档切出来的片段**，而**字段形状不变**：

```text
day061   {"doc_id": <doc_id>,   "source": …, "text": <全文>,   "metadata": {…}}
day062   {"doc_id": <chunk_id>, "source": …, "text": <片段>,   "metadata": {…}}
```

这不是巧合，是 day061 特意留下的：正文里写着"day062 会在这里切出片段，
接口形状不变"。**一个接口形状能扛过一次语义升级，才说明它当初是照
"下游需要什么"设计的，而不是照"当天有什么"设计的。**

## 报告里必须有的三列

| 列 | 回答的问题 | 没有它会发生什么 |
|----|-----------|-----------------|
| ``count`` / ``total_tokens`` | 索引有多大、embedding 要花多少 | 上线后才发现账单翻倍 |
| ``duplication_ratio`` | 重叠白花了多少 | 把 ``overlap`` 调大"为了效果"，成本悄悄涨 |
| ``oversized_count`` | 有多少块注定被模型截断 | 检索命中率莫名偏低，查不出原因 |

三列都来自 ``ChunkSet`` 的属性，因此**报告与单份文档的统计永远同源**——
这正是"先定形状、再写策略"的回报。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.chunking.base import (
    ChunkPolicy,
    build_chunker,
    default_registry,
)
from smart_research_agent.chunking.tokens import TokenMeasurer
from smart_research_agent.chunking.types import STRATEGIES, Chunk, ChunkSet
from smart_research_agent.documents.pipeline import DocumentIngestor, IngestReport
from smart_research_agent.documents.types import Document
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChunkReport:
    """一批分块结果的报告（逐文档 × 逐策略 + 汇总）."""

    sets: tuple[ChunkSet, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def documents(self) -> int:
        """报告里出现过多少份**不同的文档**（同一份文档跑四个策略只算一份）."""
        return len({chunk_set.doc_id for chunk_set in self.sets})

    @property
    def count(self) -> int:
        """块总数（跨文档、跨策略合计）."""
        return sum(chunk_set.count for chunk_set in self.sets)

    @property
    def total_tokens(self) -> int:
        """预算单位总数（按各集自己的度量器口径）."""
        return sum(chunk_set.total_tokens for chunk_set in self.sets)

    @property
    def strategies(self) -> tuple[str, ...]:
        """报告覆盖的策略（按 ``STRATEGIES`` 的顺序，不按传入顺序）."""
        present = {chunk_set.strategy for chunk_set in self.sets}
        return tuple(name for name in STRATEGIES if name in present)

    def by_strategy(self, strategy: str) -> list[ChunkSet]:
        """某个策略下的全部块集（按文档来源排序，便于逐次运行对比）."""
        return sorted(
            (chunk_set for chunk_set in self.sets if chunk_set.strategy == strategy),
            key=lambda item: item.source,
        )

    def chunks(self) -> list[Chunk]:
        """全部块（按文档、策略、序号展开）."""
        return [chunk for chunk_set in self.sets for chunk in chunk_set.chunks]

    def oversized(self) -> list[Chunk]:
        """全部超预算的块（跨文档跨策略）."""
        return [chunk for chunk in self.chunks() if chunk.oversized]

    def strategy_summary(self) -> list[dict[str, Any]]:
        """逐策略汇总（对照表的数据源，端点 ``/chunking/split`` 的 ``summary``）."""
        rows: list[dict[str, Any]] = []
        for strategy in self.strategies:
            sets = self.by_strategy(strategy)
            chunks = sum(item.count for item in sets)
            tokens = sum(item.total_tokens for item in sets)
            chars = sum(item.total_chars for item in sets)
            doc_chars = sum(item.doc_chars for item in sets)
            rows.append(
                {
                    "strategy": strategy,
                    "documents": len(sets),
                    "chunks": chunks,
                    "total_tokens": tokens,
                    "total_chars": chars,
                    "avg_tokens": round(tokens / max(1, chunks), 1),
                    "coverage": round(chars / max(1, doc_chars), 4),
                    # 与 ChunkSet.duplication_ratio 同一个口径：**下界裁到 0**。
                    # 块首尾被修剪的空白会让差额略小于 0，而那与重叠无关。
                    "duplication_ratio": max(0.0, round(chars / max(1, doc_chars) - 1, 4)),
                    "oversized_count": sum(item.oversized_count for item in sets),
                }
            )
        return rows

    def duplicate_chunks(self) -> dict[str, list[str]]:
        """跨文档/跨块重复的内容（``fingerprint`` → 出现的 ``chunk_id`` 列表）.

        这是 ``chunk_id`` 与 ``fingerprint`` 分工的落地：**只有指纹能回答
        "同一段话存了几份"**。一份 3000 页的规范里，同一段"注意"出现
        20 次是一笔真实的存储与检索噪声成本，而按 ``chunk_id`` 是看不出来的
        （它们各有各的位置 id）。只保留出现次数大于 1 的项。
        """
        buckets: dict[str, list[str]] = {}
        for chunk in self.chunks():
            buckets.setdefault(chunk.fingerprint, []).append(chunk.chunk_id)
        return {
            fingerprint: ids for fingerprint, ids in buckets.items() if len(ids) > 1
        }

    def knowledge_records(self) -> list[dict[str, Any]]:
        """全部块转成知识库记录（形状见模块 docstring）."""
        records: list[dict[str, Any]] = []
        for chunk_set in self.sets:
            records.extend(chunk_set.knowledge_records())
        return records

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "documents": self.documents,
            "chunks": self.count,
            "total_tokens": self.total_tokens,
            "strategies": list(self.strategies),
            "oversized_count": len(self.oversized()),
            "duplicate_groups": len(self.duplicate_chunks()),
            "summary": self.strategy_summary(),
            "metadata": dict(self.metadata),
            "sets": [
                chunk_set.to_dict(include_text=include_text) for chunk_set in self.sets
            ],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.documents} 份文档 × {len(self.strategies)} 个策略 → "
            f"{self.count} 块 / {self.total_tokens} tokens | "
            f"超预算 {len(self.oversized())} | 重复内容组 {len(self.duplicate_chunks())}"
        )

    def render_markdown(self) -> str:
        """把报告渲染成 markdown（可直接贴进选型记录）."""
        rows = self.strategy_summary()
        lines = [
            "# 分块报告",
            "",
            f"- 文档数：{self.documents}",
            f"- 策略数：{len(self.strategies)}（{', '.join(self.strategies)}）",
            f"- 块总数：{self.count}",
            f"- 预算单位合计：{self.total_tokens}",
            "",
            "| 策略 | 文档 | 块数 | 平均长度 | 覆盖率 | 重复率 | 超预算 |",
            "|------|------|------|---------|--------|--------|--------|",
        ]
        lines.extend(
            f"| `{row['strategy']}` | {row['documents']} | {row['chunks']} | "
            f"{row['avg_tokens']} | {row['coverage']:.2%} | "
            f"{row['duplication_ratio']:.2%} | {row['oversized_count']} |"
            for row in rows
        )
        oversized = self.oversized()
        if oversized:
            lines.extend(
                ["", "## 超预算的块（原子块保护的结果，需人工看一眼）", ""]
            )
            lines.extend(
                f"- `{chunk.source}` #{chunk.index} {chunk.token_count} "
                f"{chunk.token_measurer}（{chunk.reason}）"
                for chunk in oversized
            )
        lines.append("")
        return "\n".join(lines)


class ChunkPipeline:
    """分块编排器：一个注册表 + 一个度量器 + 一个 embedding 提供方.

    三个依赖都是**可注入**的，这一点决定了本模块能不能被离线测试覆盖：

    ```text
    measurer   默认 chars（确定性）；注入 tiktoken 才按真实 token 计
    embedding  只有语义策略用；注入 MockEmbedding 即可完全离线
    registry   默认四种策略各一份；测试可只注册其中一种，断言"没注册的策略会报错"
    ```
    """

    def __init__(
        self,
        *,
        measurer: TokenMeasurer | None = None,
        embedding: EmbeddingProvider | None = None,
    ) -> None:
        self.measurer = measurer
        self.embedding = embedding
        self.registry = default_registry(measurer=measurer, embedding=embedding)

    def chunk(
        self,
        document: Document,
        strategy: str,
        policy: ChunkPolicy | None = None,
    ) -> ChunkSet:
        """按策略切一份文档（每次都用策略的默认参数，除非显式传入 ``policy``）."""
        # 无论走哪条路都先在注册表里查一次：注册表是"**哪些策略存在**"的
        # 唯一答案。少了这一步，一个拼错的策略名会绕过注册表被直接建出来，
        # 于是"未知策略应当报错"这条保证就只剩一半。
        self.registry.get(strategy)
        if policy is None:
            return self.registry.get(strategy).split(document)
        # 显式给了参数就现建一个：注册表里那份是"默认参数"的实例，
        # **改它会让下一次调用的默认值悄悄变掉**。
        chunker = build_chunker(
            strategy, policy, measurer=self.measurer, embedding=self.embedding
        )
        return chunker.split(document)

    def chunk_all(
        self,
        document: Document,
        *,
        strategies: Sequence[str] | None = None,
        policies: dict[str, ChunkPolicy] | None = None,
    ) -> ChunkReport:
        """同一份文档跑多个策略（选型对照的入口）."""
        wanted = list(strategies or STRATEGIES)
        overrides = policies or {}
        sets = tuple(
            self.chunk(document, strategy, overrides.get(strategy))
            for strategy in wanted
        )
        report = ChunkReport(
            sets=sets,
            metadata={"document": document.source, "doc_id": document.fingerprint},
        )
        logger.info("分块对照完成：%s", report.summary_line())
        return report

    def chunk_documents(
        self,
        documents: Sequence[Document],
        strategy: str,
        policy: ChunkPolicy | None = None,
    ) -> ChunkReport:
        """一批文档用同一个策略切（真实入库时的形状）."""
        sets = tuple(
            self.chunk(document, strategy, policy) for document in documents
        )
        report = ChunkReport(
            sets=sets,
            metadata={"strategy": strategy, "documents": str(len(sets))},
        )
        logger.info("批量分块完成：%s", report.summary_line())
        return report

    def ingest_and_chunk(
        self,
        items: Sequence[tuple[str, bytes]],
        strategy: str,
        policy: ChunkPolicy | None = None,
    ) -> tuple[IngestReport, ChunkReport]:
        """一次跑通 day061 → day062：入库（含去重与三类状态）→ 分块.

        返回**两个报告**而不是一个合并的：入库回答"哪些文件没进来"，
        分块回答"进来的被切成了什么样"。合并成一个会让"3 个文件失败"
        与"3 个块超预算"混在同一张表里。
        """
        ingest = DocumentIngestor().ingest_bytes(items)
        chunked = self.chunk_documents(list(ingest.documents), strategy, policy)
        return ingest, chunked


def chunking_plan(doc_chars: int, policy: ChunkPolicy) -> dict[str, Any]:
    """参数 → 代价的**事前**估算（先算再切；实际块数由内容决定）.

    它回答的是调参时最常问的两个问题：**预算调大一倍，块数会少多少？
    重叠调大一倍，embedding 会多花多少？** 这两个数在切之前就能算出来，
    因此不必"改参数 → 跑一遍 → 看账单"。

    ``estimated_chunks`` 是按固定窗口推算的量级，必须**说清它的适用范围**：

    - ``fixed``：精确一致（窗口就是它的定义）；
    - ``recursive`` / ``structural`` / ``semantic``：是**上界**——
      这三种策略会把相邻片段合并到预算，因此实际块数只会更少。
    """
    from smart_research_agent.chunking.fixed import window_plan

    plan = window_plan(doc_chars, policy)
    plan["measurer"] = policy.measurer
    plan["similarity_percentile"] = policy.similarity_percentile
    plan["estimated_chunks_exact"] = policy.strategy == "fixed"
    plan["note"] = (
        "estimated_chunks 按固定窗口推算：fixed 下与实测一致；"
        "recursive / structural / semantic 的实际块数由内容边界决定，通常更少"
    )
    return plan


def knowledge_record_shape() -> dict[str, Any]:
    """知识库记录的字段说明（端点与文档同源）."""
    return {
        "fields": {
            "doc_id": "块 id（chunk_id）：知识库里的唯一键就是片段",
            "source": "来源标签（继承自 Document.source）",
            "text": "原文片段：能在 document.text 里按 start_char/end_char 逐字核对",
            "metadata": "parent_doc_id / strategy / index / token_count / "
            "heading_path / fingerprint / oversized / retrieval_text",
        },
        "embedding_input": "metadata.retrieval_text（含标题面包屑），不是 text",
        "next_step": "day064/065 的向量库按 chunk_id 入库；day066 的检索器按 retrieval_text 召回",
        "previous": "day061 交的是整份文档，今天交的是片段，字段形状不变",
    }


__all__ = ["ChunkPipeline", "ChunkReport", "chunking_plan", "knowledge_record_shape"]
