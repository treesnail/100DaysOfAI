"""检索包（M6-D5）：把"库里有向量"变成"**问一句话能拿回可解释的几条**".

day064 交了库（存向量、按相似度查），day065 交了账（清单、版本、漂移判定）。
今天这一层要回答两件它们都答不了的事：

```text
1. 给一句话，取回哪几条？        → 编码 + Top-K + 深度 + 过滤
2. 为什么是这几条、为什么只有这几条？ → 四副形状里的 dropped_* 与 empty_reason
```

第 2 条才是这一课的主题。**一个只会 `backend.query()` 的检索器，
在"结果变少了"这个问题前面是哑的**：它交出一批命中，然后就没有话可说了。
今天的每一层都在为"把话说清楚"服务——深度可以复述、过滤条件可以复述、
三道减法各自留了数字、空结果只有**一个**原因、索引版本与漂移跟着结果一起走。

## 模块地图（每个模块回答一个问题）

```text
errors.py     这一层会怎么失败（调用方 / 索引运维 / 打包配置 三族分开，修复人不同）
types.py      四副形状：查询、命中、索引状态、结果；时间范围的三条约定
filters.py    时间范围怎么变成 where；where 与 time_range 撞车为什么必须报错
retriever.py  九步流水线：Top-K + 深度 + 过滤 + 阈值 + 多样性 + 空结果诊断 + 漂移告警
routing.py    多索引选路：一句查询该去哪个库，写成显式的一层而不是隐式的 if
context.py    上下文打包与引用标记：预算、单条截断、整包丢尾、[n] 编号
pipeline.py   检索 → 打包 → 生成 → 引用（检索为空时绝不调 LLM）
```

## 三条贯穿全包的纪律

1. **排序与阈值只有一个口径**。``score`` 越大越近（与 ``vectorstore.metrics``
   同一条），阈值用同一个 ``>=``、在本层**只落一次刀**。把阈值同时交给库
   与本层会让"切掉了几条"这个数字变成两处各一份，而它们迟早会不一样——
   那时报告里那个数字就不再是事实。
2. **"条数少于期望"必须是可回答的**。``dropped_below_threshold`` /
   ``dropped_by_diversity`` / ``dropped_by_top_k`` 三个数字各自对应一个可调参数，
   ``empty_reason`` 只给**一个**原因（按 ``EMPTY_REASONS`` 的优先级定）。
   合成一个 ``dropped`` 会让"该调哪个参数"这个问题永远无法回答。
3. **检索器不认识语料，只认识索引状态**。它不知道文档写的是什么，
   但它必须能说出"我查的是哪一版索引、清单有没有漂"——版本号与漂移跟着
   每一条结果走，**不静默降级**（漂移要么进 notes、要么在严格模式下直接报错）。

## 与既有包的接缝

- **上游**：``vectorstore``（day064）提供库与六个原语、``indexing``（day065）
  提供清单与 ``compare_with_store``、``llm.embedding``（day041）提供编码器、
  ``chunking``（day062）的 ``heading_path`` / ``parent_doc_id`` 元数据在这里
  终于被用上（过滤、分组、引用）；
- **脚下**：``config`` 的 ``retrieval_*`` 组（这一组**不改变任何产物字节**，
  只改变"怎么问"——与 ``indexing_*`` 的分工见 ``config``）；
- **下游**：day067 的混合检索用 ``RetrievalHit.channel`` / ``RetrievalResult.channels``
  做融合，用 ``RetrievalQuery.extra`` 挂融合参数，用 ``dropped_*`` 判断
  "这一路是不是一条都没活下来"；day068 的重排消费命中的分数与文本；
  day071 的 RAG 评估用 ``aggregate_results`` 与 ``IndexState.drift``；
- **端点**：``api.routes`` 的 ``/retrieval/*``；手册在 ``docs/retrieval.md``；
  离线演示 ``scripts/retrieval_demo.py``。

## 导出范围（七个模块全部落地）

```text
已落地    errors / types / filters / retriever / routing / context / pipeline
```

因此本文件的导入**覆盖全部七个模块**：一个包在 ``__init__`` 里
导入还没写出来的模块，会让"只想用检索器"的人连 ``import`` 都过不去——
而那种失败与检索本身毫无关系（与 ``vectorstore.__init__`` 把两个可选后端
做成延迟导入是同一条理由）。
"""

from __future__ import annotations

from smart_research_agent.retrieval.context import (
    BLOCK_SEPARATOR,
    TRUNCATION_MARKER,
    Citation,
    PackedContext,
    pack_context,
)
from smart_research_agent.retrieval.errors import (
    ContextError,
    IndexStateError,
    QueryError,
    RetrievalError,
)
from smart_research_agent.retrieval.filters import (
    DEFAULT_TIME_FIELD,
    TIME_FIELD_SUFFIXES,
    combine_where,
    describe_conditions,
    time_field_of,
    time_range_clause,
    unknown_filter_fields,
    validate_where,
)
from smart_research_agent.retrieval.pipeline import (
    FALLBACK_NO_CONTEXT,
    RAG_ANSWER_PROMPT,
    RAG_ANSWER_PROMPT_V1,
    RAG_ANSWER_PROMPT_VERSION,
    RagAnswer,
    RagPipeline,
)
from smart_research_agent.retrieval.retriever import Retriever, build_retriever
from smart_research_agent.retrieval.routing import RouteDecision, StoreRouter
from smart_research_agent.retrieval.types import (
    CHANNEL_VECTOR,
    DEFAULT_DOC_ID_FIELD,
    DEFAULT_FETCH_MULTIPLIER,
    DEFAULT_MIN_HIT_CHARS,
    EMPTY_REASON_BELOW_THRESHOLD,
    EMPTY_REASON_DESCRIPTIONS,
    EMPTY_REASON_DIVERSITY,
    EMPTY_REASON_FILTERED_OUT,
    EMPTY_REASON_NO_DATA,
    EMPTY_REASON_NONE,
    EMPTY_REASONS,
    MAX_FETCH_K,
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
    IndexState,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
    TimeRange,
    aggregate_results,
)

__all__ = [
    "BLOCK_SEPARATOR",
    "CHANNEL_VECTOR",
    "DEFAULT_DOC_ID_FIELD",
    "DEFAULT_FETCH_MULTIPLIER",
    "DEFAULT_MIN_HIT_CHARS",
    "DEFAULT_TIME_FIELD",
    "EMPTY_REASON_BELOW_THRESHOLD",
    "EMPTY_REASON_DESCRIPTIONS",
    "EMPTY_REASON_DIVERSITY",
    "EMPTY_REASON_FILTERED_OUT",
    "EMPTY_REASON_NONE",
    "EMPTY_REASON_NO_DATA",
    "EMPTY_REASONS",
    "FALLBACK_NO_CONTEXT",
    "MAX_FETCH_K",
    "RAG_ANSWER_PROMPT",
    "RAG_ANSWER_PROMPT_V1",
    "RAG_ANSWER_PROMPT_VERSION",
    "RETRIEVAL_LIMITATIONS",
    "RETRIEVAL_OUT_OF_SCOPE",
    "TIME_FIELD_SUFFIXES",
    "TRUNCATION_MARKER",
    "Citation",
    "ContextError",
    "IndexState",
    "IndexStateError",
    "PackedContext",
    "QueryError",
    "RagAnswer",
    "RagPipeline",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "RouteDecision",
    "StoreRouter",
    "TimeRange",
    "aggregate_results",
    "build_retriever",
    "combine_where",
    "describe_conditions",
    "pack_context",
    "time_field_of",
    "time_range_clause",
    "unknown_filter_fields",
    "validate_where",
]
