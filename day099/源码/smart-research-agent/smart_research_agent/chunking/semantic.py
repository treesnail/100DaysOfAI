"""语义分块：在"话题变了"的地方切（M6-D2）.

前三种策略切在哪里，都由**写在代码里的规则**决定：窗口长度、分隔符表、
标题层级。语义分块换了一个判据——**让向量告诉我话题在哪里变了**：

```text
1. 把原文按自然段切开            （用空行来定"自然段"）
2. 每个自然段编码成一个向量       （day041 的 EmbeddingProvider）
3. 算相邻两段的余弦相似度          （day009 的 cosine_similarity）
4. 相似度最低的那些缝切一刀        （阈值取相似度分布的分位数）
5. 按预算合并，再回带重叠
```

它的直觉很干净：**相邻两段讲同一件事时，向量方向接近；换了话题，方向就岔开。**

## 三处必须说清楚的设计

### 1. 阈值用"分位数"，不用绝对值

每天都会有人问"相似度低于 0.7 就切，对不对"。答案是**这个数没有意义**：

```text
同一个 0.7：
  在 text-embedding-3-small 上可能是"非常相似"
  在 char-ngram 上可能是"完全不相干"
  在另一种语言上又是另一个含义
```

余弦相似度的绝对值依赖模型的训练目标、维度、是否归一化以及语料的语言。
换一个 embedding 提供方，同一个绝对阈值就换了一个含义。

分位数把它变成**相对判据**：``similarity_percentile=0.25`` 表示
"在相似度最低的那 25% 的缝上切"。它自动适配任何提供方的尺度——
这是"参数应当表达意图（切得碎还是切得粗），而不是表达一个在别处
标定出来的数字"。

### 2. 它是四种策略里唯一会调用外部能力的

其余三种是纯字符串运算：给定文本与参数，结果**完全确定**。
语义分块要 embedding，于是它的确定性取决于提供方：

```text
MockEmbedding / CharNgramEmbedding     离线确定（本课默认与测试都用它们）
SentenceTransformer / OpenAI / Ollama  依赖权重或网络
```

因此 ``self.stats`` 里记着 embedding 的类名与维度，产物里也带着它——
**"这份块是怎么切出来的"必须能被回答**，否则一个"这次块变少了"的
现象无法归因（换提供方、换阈值、换文档，三种原因长得一模一样）。

### 3. 它只认识自然段，不认识块类型

语义分块在**纯文本**上工作（四种策略里只有结构策略消费 ``blocks``）。
因此它**会把一个含空行的代码块从中间切开**——这与结构策略的
"原子块保护"正好相反。这不是缺陷，而是选型依据：代码密集的知识库
该用结构策略，问答对/散文密集的语料才轮到语义策略。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.chunking.base import (
    Chunker,
    ChunkPolicy,
    Span,
    default_policy,
    heading_positions,
    paragraph_spans,
    path_for_offset,
)
from smart_research_agent.chunking.errors import ChunkingError
from smart_research_agent.chunking.types import STRATEGY_SEMANTIC
from smart_research_agent.documents.types import Document
from smart_research_agent.llm.embedding import EmbeddingProvider, default_embedding
from smart_research_agent.memory.vector_store import cosine_similarity


def percentile(values: list[float], fraction: float) -> float:
    """线性插值分位数（``values`` 不必有序，函数内部排序）.

    与 ``types._percentile`` 是同一个口径（``rank = fraction * (n - 1)``，
    相邻两点线性插值），只是这里处理浮点而不是整数。**两处口径必须一致**：
    报告里"块长 p90"与"相似度阈值"会被并排比较，
    若分位数在两处用两种算法，那种比较就没有意义。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


class SemanticChunker(Chunker):
    """语义分块器（自然段向量 → 相邻相似度 → 分位数阈值 → 预算合并）."""

    name = STRATEGY_SEMANTIC
    #: 重叠由收尾器统一施加（``finalize`` 的顺序：修剪 → 重叠 → 合并 → 校验）.
    apply_overlap = True

    def __init__(
        self,
        policy: ChunkPolicy | None = None,
        *,
        embedding: EmbeddingProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy or default_policy(STRATEGY_SEMANTIC), embedding=embedding, **kwargs
        )
        # 构造期不探活（与 day045 LocalModel 同一条）：没注入提供方就取
        # settings 决定的默认实现——离线环境是 char-ngram，配了密钥才是云端。
        self.embedding = self.embedding or default_embedding()
        # 与递归策略同一口径：先把重叠的额度留出来，最后统一回带（见 base）。
        self.window_budget = self.policy.max_tokens - self.policy.overlap_tokens
        self.window_overlap = 0
        if self.policy.overlap_tokens * 2 >= self.policy.max_tokens:
            # 与递归策略同一条护栏（见 recursive.py 的说明）：重叠过大时
            # 每一块都会变成上一块的复制品，而参数看起来完全合法。
            raise ChunkingError(
                f"重叠 {self.policy.overlap_tokens} 相对预算 "
                f"{self.policy.max_tokens} 过大（要求 2 × overlap < max_tokens）："
                "核心窗口只有 max-overlap 个单位，回带之后每一块几乎都是上一块的复制品"
            )

    # ------------------------------------------------------------------ #
    # 切分
    # ------------------------------------------------------------------ #

    def _spans(self, document: Document) -> list[Span]:
        """五步流水线：分段 → 编码 → 相似度 → 阈值切缝 → 预算合并."""
        text = document.text
        if not text.strip():
            return []
        units = paragraph_spans(text)
        if len(units) < 2:
            # 只有一个自然段（或干脆没有）：语义判据没有施展空间。
            # **不报错**——这是一个结论，报告里会看到 units=1 / breaks=0。
            self.stats = {"units": str(len(units)), "breaks": "0", "threshold": "n/a"}
            return self._group_spans(text, units, boundaries=set(), positions=[])
        vectors = self.embedding.embed_batch(
            [text[left:right] for left, right in units]
        )
        similarities = [
            cosine_similarity(left, right) for left, right in zip(vectors, vectors[1:])
        ]
        threshold = percentile(similarities, self.policy.similarity_percentile)
        positions = heading_positions(document)
        heading_starts = {position for position, _ in positions}
        breaks = {
            index
            for index, similarity in enumerate(similarities)
            if similarity < threshold
        }
        # 标题处**额外**断开：heading_path 是块级属性，"一个块有两条路径"
        # 是表达不了的。因此标题丰富的文档上，语义策略会自然地接近
        # "结构策略 + 相似度细分"。
        heading_breaks = {
            index
            for index, (left, _) in enumerate(units)
            if left in heading_starts and index > 0
        }
        self.stats = {
            "units": str(len(units)),
            "breaks": str(len(breaks | heading_breaks)),
            "threshold": f"{threshold:.4f}",
            "similarity_percentile": str(self.policy.similarity_percentile),
            "heading_breaks": str(len(heading_breaks)),
            "embedding": type(self.embedding).__name__,
            "embedding_dimension": str(self.embedding.dimension),
        }
        return self._group_spans(
            text, units, boundaries=breaks | heading_breaks, positions=positions
        )

    def _group_spans(
        self,
        text: str,
        units: list[tuple[int, int]],
        boundaries: set[int],
        positions: list[tuple[int, tuple[str, ...]]],
    ) -> list[Span]:
        """按边界把自然段分组，组内按预算合并，装不下的组再按预算下切.

        两处与结构策略同源的做法：

        - **先合并再判断**：一个组有 5 个自然段但都很短时，它们应当合成一块，
          而不是各成一块——语义的判据是"话题"，而话题可以跨自然段；
        - **装不下的组回切**：一份没有空行的长文（如日志）在这里退化，
          与其它策略的退化路径一致（``_back_off``）。

        每块只带**一个** ``heading_path``，取的是块起点所在的小节——
        块内部若跨了标题，路径不再细分（标题处已额外断开，因此这种情况
        只在"标题与正文挤在同一段"时出现）。
        """
        spans: list[Span] = []
        group_start: int | None = None
        group_end = 0
        group_path: tuple[str, ...] = ()
        for index, (left, right) in enumerate(units):
            must_break = index in boundaries
            if group_start is not None and (
                must_break or not self._fits(text, group_start, right)
            ):
                self._flush_group(spans, text, group_start, group_end, group_path)
                group_start = None
            if group_start is None:
                group_start, group_end, group_path = left, right, path_for_offset(
                    left, positions
                )
            else:
                group_end = right
        if group_start is not None:
            self._flush_group(spans, text, group_start, group_end, group_path)
        return spans

    def _flush_group(
        self,
        spans: list[Span],
        text: str,
        start: int,
        end: int,
        heading_path: tuple[str, ...],
    ) -> None:
        """收束一个组：装得下就是一个区间，装不下就按预算回切."""
        if start >= end:
            return
        if self._fits(text, start, end):
            spans.append(
                Span(
                    start_char=start,
                    end_char=end,
                    reason="topic",
                    heading_path=heading_path,
                )
            )
            return
        spans.extend(self._back_off(text, start, end, "topic-overflow"))

    def describe(self) -> dict[str, Any]:
        """策略自述."""
        return {
            "name": self.name,
            "unit": "自然段向量 + 相邻相似度",
            "where_to_cut": "相邻段落相似度低于分位数阈值处切一刀",
            "parameters": [
                "max_tokens",
                "overlap_tokens",
                "min_tokens",
                "measurer",
                "similarity_percentile",
            ],
            "uses_similarity_percentile": True,
            "supports_overlap": True,
            "embedding": type(self.embedding).__name__,
            "deterministic": "取决于 embedding 提供方（离线实现确定，云端不确定）",
            "strengths": [
                "唯一按内容判据切分的策略，话题转折处断得比标点更准",
                "阈值是相对分位数，换 embedding 提供方不用重新标定",
                "不依赖结构元数据，任何纯文本都能跑",
            ],
            "weaknesses": [
                "每切一次要多付一次 embedding（N 个自然段 = N 次编码）",
                "只认识自然段，会把含空行的代码块切开（与结构策略相反）",
                "边界不可解释：说不出「为什么切在这」，只能说出「这里相似度最低」",
            ],
            "cost": "N 段 × 1 次 embedding + 重叠放大量",
        }


__all__ = ["SemanticChunker", "percentile"]
