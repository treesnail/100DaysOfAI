"""分块策略包（M6-D2）：把文档切成"能被检索到的最小语境单元".

day061 把八种格式收敛成了 ``Document``（全文 + 块序列 + 指纹）；
今天在它上面切出 ``Chunk``，交给向量库。

```text
Document（全文 + 块序列）
    │  四种策略，同一个形状
    ├─ fixed       每 max_tokens 个单位一刀，不看内容
    ├─ recursive   按分隔符表从强到弱下切，再合并到预算
    ├─ structural  按 day061 抽出的标题层级切小节，保护代码块与表格
    └─ semantic    相邻自然段相似度最低处切（用 embedding）
    ▼
ChunkSet（块序列 + 度量口径 + 覆盖率/重复率/超预算）
    │
    ▼
knowledge_records()  →  day009 的 VectorStore / day064 的向量库
```

九个模块，每个回答一个问题：

```text
errors.py     分块会怎么失败（参数矛盾 / 未知策略）
tokens.py     预算单位是谁（chars 还是真实 token）、为什么默认 chars
types.py      产物长什么样（Chunk / ChunkSet / chunk_id 与 fingerprint 的分工）
base.py       在哪里切之外的**全部事情**（收尾六步 + 覆盖率不变量 + 注册表）
fixed.py      不看内容的基线策略（参数一确定，成本就算得出来）
recursive.py  分隔符优先级表（中文标点必须在表里，否则会静默退化）
structural.py 按标题切（breadcrumb、原子块保护、定位失败如何退化）
semantic.py   按话题切（分位数阈值、唯一会调用外部能力的策略）
pipeline.py   批量与对照（报告三列：规模、重复、超预算）
evaluate.py   用你自己的探针集给四种策略打分（hit@k / MRR / 命中块均长）
```

## 三条贯穿全包的纪律

1. **块文本是原文的连续子串，且非空白字符全覆盖**。
   ``document.text[start_char:end_char] == chunk.text`` 由构造保证
   （``Span`` 只带区间、不带文本），非空白字符全覆盖由
   ``Chunker._assert_spans`` 强制——**违反即报错，而不是给一个
   coverage=0.94 的报告**。这是 day061「只分类不丢弃」在切分层的对应物；
2. **确定性要能被回答**。``ChunkSet.token_measurer`` 与
   ``metadata.policy`` 记录了"这批块是按什么单位、什么参数切的"，
   语义策略还额外记下 embedding 的类名——**没有这一层，
   "这次比上次少了 40 个块"永远说不清原因**；
3. **同一个字段只表达一件事**。``chunk_id`` 是位置身份、
   ``fingerprint`` 是内容身份、``text`` 是原文片段、
   ``retrieval_text`` 是检索视图、``oversized`` 是"不可切且超预算"。
   任何两个混在一起，报告里就会出现无法区分的情形（见 types.py）。

## 与既有包的接缝

- **上游**：``documents.Document``（day061）、``llm.embedding``（day041）、
  ``llm.tokenizer``（day034）；
- **下游**：``memory.vector_store``（day009）、``evaluation``（day025 ~ day031）
  ——day071 的 RAG 评估会直接消费 ``evaluate.py`` 的探针与得分结构；
- **端点**：``api.routes`` 的 ``/chunking/*`` 四个端点全部**只读或纯计算**，
  目录遍历与批量入库由 ``scripts/chunking_demo.py`` 承担。
"""

from __future__ import annotations

from smart_research_agent.chunking.base import (
    BOUNDARY_CHARS,
    CHUNKING_LIMITATIONS,
    CHUNKING_OUT_OF_SCOPE,
    DEFAULT_POLICIES,
    PARAGRAPH_GAP,
    Chunker,
    ChunkerRegistry,
    ChunkPolicy,
    Span,
    build_chunker,
    chunking_boundaries,
    default_policy,
    default_registry,
    heading_positions,
    paragraph_spans,
    path_for_offset,
    snap_to_boundary,
)
from smart_research_agent.chunking.errors import ChunkingError, UnsupportedStrategy
from smart_research_agent.chunking.evaluate import (
    DEFAULT_TOP_K,
    ChunkingEvaluation,
    ProbeHit,
    RetrievalProbe,
    StrategyScore,
    ambiguous_probes,
    build_index,
    evaluate_strategies,
    score_probes,
)
from smart_research_agent.chunking.fixed import FixedSizeChunker, window_plan
from smart_research_agent.chunking.pipeline import (
    ChunkPipeline,
    ChunkReport,
    chunking_plan,
    knowledge_record_shape,
)
from smart_research_agent.chunking.recursive import (
    DEFAULT_SEPARATORS,
    SEPARATOR_LABELS,
    SEPARATOR_NOTES,
    RecursiveChunker,
    separator_histogram,
    split_keep_separator,
)
from smart_research_agent.chunking.semantic import SemanticChunker, percentile
from smart_research_agent.chunking.structural import (
    ATOMIC_KINDS,
    Section,
    StructuralChunker,
    build_sections,
)
from smart_research_agent.chunking.tokens import (
    MEASURER_CHARS,
    MEASURER_TIKTOKEN,
    MEASURERS,
    CharMeasurer,
    TiktokenMeasurer,
    TokenMeasurer,
    count_tokens,
    describe_measurers,
    longest_prefix_within,
    resolve_measurer,
)
from smart_research_agent.chunking.types import (
    CHUNK_ID_LENGTH,
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_SEMANTIC,
    STRATEGY_STRUCTURAL,
    Chunk,
    ChunkSet,
    chunk_id_for,
)

__all__ = [
    "ATOMIC_KINDS",
    "BOUNDARY_CHARS",
    "CHUNKING_LIMITATIONS",
    "CHUNKING_OUT_OF_SCOPE",
    "CHUNK_ID_LENGTH",
    "DEFAULT_POLICIES",
    "DEFAULT_SEPARATORS",
    "DEFAULT_TOP_K",
    "MEASURERS",
    "MEASURER_CHARS",
    "MEASURER_TIKTOKEN",
    "PARAGRAPH_GAP",
    "SEPARATOR_LABELS",
    "SEPARATOR_NOTES",
    "STRATEGIES",
    "STRATEGY_FIXED",
    "STRATEGY_RECURSIVE",
    "STRATEGY_SEMANTIC",
    "STRATEGY_STRUCTURAL",
    "CharMeasurer",
    "Chunk",
    "ChunkPipeline",
    "ChunkPolicy",
    "ChunkReport",
    "ChunkSet",
    "Chunker",
    "ChunkerRegistry",
    "ChunkingError",
    "ChunkingEvaluation",
    "FixedSizeChunker",
    "ProbeHit",
    "RecursiveChunker",
    "RetrievalProbe",
    "Section",
    "SemanticChunker",
    "Span",
    "StrategyScore",
    "StructuralChunker",
    "TiktokenMeasurer",
    "TokenMeasurer",
    "UnsupportedStrategy",
    "ambiguous_probes",
    "build_chunker",
    "build_index",
    "build_sections",
    "chunk_id_for",
    "chunking_boundaries",
    "chunking_plan",
    "count_tokens",
    "default_policy",
    "default_registry",
    "describe_measurers",
    "evaluate_strategies",
    "heading_positions",
    "knowledge_record_shape",
    "longest_prefix_within",
    "paragraph_spans",
    "path_for_offset",
    "percentile",
    "resolve_measurer",
    "score_probes",
    "separator_histogram",
    "snap_to_boundary",
    "split_keep_separator",
    "window_plan",
]
