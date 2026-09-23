"""检索包（M6-D5 / M6-D6 / M6-D7）：把"库里有向量"变成"**问一句话能拿回可解释的几条**".

day064 交了库（存向量、按相似度查），day065 交了账（清单、版本、漂移判定）。
day066 这一层回答了那两件事都答不了的问题：

```text
1. 给一句话，取回哪几条？        → 编码 + Top-K + 深度 + 过滤
2. 为什么是这几条、为什么只有这几条？ → 四副形状里的 dropped_* 与 empty_reason
```

第 2 条才是这一课的主题。**一个只会 `backend.query()` 的检索器，
在"结果变少了"这个问题前面是哑的**：它交出一批命中，然后就没有话可说了。
day066 的每一层都在为"把话说清楚"服务——深度可以复述、过滤条件可以复述、
三道减法各自留了数字、空结果只有**一个**原因、索引版本与漂移跟着结果一起走。

day067 加了第二个失败面：**单路检索的错法只有一种，而两路的错法各不相同**。

```text
向量路擅长的     同义改写、语义相近但词面完全不同的问法
关键词路擅长的   编号、函数名、报错码、专有名词（字面精确）
```

因此 day067 多出来的三件事，全都围着"两路各自的对错"转：

```text
lexical.py  关键词一路（零依赖 BM25）：它**会干脆地一条都不返回**，
            因此它必须能说出"为什么"（missing_terms 对账）
fusion.py   两路怎么合：跨通道的分数量纲不可比（cos ∈[-1,1]、BM25 ∈[0,∞)），
            于是要么只看名次（rrf），要么先归一化再加权（weighted）
hybrid.py   一条十三步的流水线：过滤在两路都落刀、阈值只在向量路落刀、
            融合后的每一条都留着"哪几路召回了它、各贡献了多少"
```

day068 回答的是第三个问题：**两路合出来的那份名单，次序对不对**。

```text
召回与排序是两件事   双塔把查询与文档分开编码（所以文档向量能离线算），
                     代价是两两不相见 → 它擅长"别漏"，不擅长"排准"
交叉编码器           查询与文档拼起来过一遍模型 → 更准，但每对一次前向推理
因此的姿势           两段式：召回取几十~几百条，重排只对**前 N 条**重打分
```

这一层加进来的三件事：

```text
rerank.py    交叉编码器重排（教学级确定性替身）+ **窗口纪律**：
             只对前 N 条打分，因此"第 N+1 条永远没有机会证明自己"必须进 notes；
             两种模式对应两种量纲处理（replace 只信重排分 / blend 各自归一再加权）；
             外加三条指标（recall@k / RR / nDCG@k）与 measure_lift 的对照报告
retriever.py 第 6.5 步：阈值落刀之后、多样性裁剪之前插一步重排
hybrid.py    第 10.5 步：融合装配之后、多样性裁剪之前插同一步
```

**为什么重排必须在多样性之前**（两处都是这一步，理由只有一条）：
多样性会按 ``doc_id`` 丢掉同一文档的其余记录，若先裁后重排，
被丢掉的那些候选**再也回不到重排器眼前**——"同一文档的第 2 段其实更贴题"
这个事实就无法被发现。顺序是"先让所有候选上桌，再做取舍"。

day069 回答的是第五个问题：**模型给的那段话，能不能被片段核对**。

```text
引用编号对账   答案里的 [n] 与那次提示词的编号一一对上（有效 / 幻觉 / 未引用）
接地定义       grounded = valid 非空 且 invalid 为空（**不是**字符串比较）
六种回退       检索/打包为空 / 调用失败 / 空回复 / 模型自己拒答 /
               没有合法引用（开了开关）/ 都没有（NONE）
提示词版本化   v1（day066 首版）→ v2（四段式 + 固定拒答句式），历史版本永不删
```

这一层加进来的三件事：

```text
generation.py  生成器与提示词：多版本模板 + 约束 + 引用溯源 + 接地校验 + 回退族
pipeline.py    第 4 步换成 RAGGenerator（空检索护栏**仍留在**它这里）；
               RagAnswer 尾部新增 check / fallback_reason 两个字段（只增不改）
types.py       两份边界声明第三次改写："不做引用溯源校验"从限制里删除，
               换成"只做编号层面核对"与"生成器只做推理侧"
```

**为什么护栏要写在两处**：本包的空检索护栏在 ``pipeline``（打包与渲染之前），
day069 的生成器又在自己门口加了一道（``PackedContext.is_empty``）。
两处守的是同一件事——不许把"没有片段"变成"让模型自由发挥"，
因为那正是 RAG 最贵的一种失败：产物看起来最像成功。

## 模块地图（每个模块回答一个问题）

```text
errors.py     这一层会怎么失败（调用方 / 索引运维 / 打包配置 / 融合 / 重排 / 生成）
types.py      四副形状：查询、命中、索引状态、结果；时间范围的三条约定；两个通道名
filters.py    时间范围怎么变成 where；where 与 time_range 撞车为什么必须报错
retriever.py  九步 + 一步重排：Top-K + 深度 + 过滤 + 阈值 + 重排 + 多样性 + 诊断
routing.py    多索引选路：一句查询该去哪个库，写成显式的一层而不是隐式的 if
context.py    上下文打包与引用标记：预算、单条截断、整包丢尾、[n] 编号
pipeline.py   检索 → 打包 → 生成 → 引用（检索为空时绝不调 LLM）
generation.py 生成器与提示词：多版本模板 + 约束 + 引用溯源 + 接地校验 + 回退族
lexical.py    关键词一路：零依赖 BM25 + CJK 2-gram 分词 + "为什么一条都没召回"的证据
fusion.py     融合与去重：RRF（只看名次）与归一化加权（先归一再加）；去重 = 合并证据
hybrid.py     混合检索器：两路共用一份过滤子句与深度，融合后仍能逐条解释
rerank.py     重排：交叉编码器替身 + 窗口纪律 + replace/blend + 三条提升指标
```

## 六条贯穿全包的纪律

1. **排序与阈值只有一个口径**。``score`` 越大越近（与 ``vectorstore.metrics``
   同一条），阈值用同一个 ``>=``、在本层**只落一次刀**。把阈值同时交给库
   与本层会让"切掉了几条"这个数字变成两处各一份，而它们迟早会不一样——
   那时报告里那个数字就不再是事实。
2. **"条数少于期望"必须是可回答的**。``dropped_below_threshold`` /
   ``dropped_by_diversity`` / ``dropped_by_top_k`` 三个数字各自对应一个可调参数，
   ``empty_reason`` 只给**一个**原因（按 ``EMPTY_REASONS`` 的优先级定）。
   合成一个 ``dropped`` 会让"该调哪个参数"这个问题永远无法回答。
   day067 把这条纪律又用了两次：**每一路**召回了几条（``channel_candidates``）
   与**每一路**贡献了几条（``fusion.contributions``）都各自成数。
   day068 加了**第四个数** ``dropped_by_rerank``（重排阈值那个旋钮）——
   它同样不许并进 ``dropped_by_top_k``，理由与前三道减法一样。
3. **检索器不认识语料，只认识索引状态**。它不知道文档写的是什么，
   但它必须能说出"我查的是哪一版索引、清单有没有漂"——版本号与漂移跟着
   每一条结果走，**不静默降级**（漂移要么进 notes、要么在严格模式下直接报错）。
4. **被忽略的参数必须被说出来**（day067 新增）。混合模式下 ``min_score``
   只作用于向量通道，而这件事**无条件进 notes**——一个"看起来能调、
   实际上不生效"的旋钮比没有旋钮更贵（见 ``hybrid`` 的阈值语义一小节）。
5. **被限制的能力也必须被说出来**（day068 新增）。重排只对前 N 条打分，
   而这条**无条件进 notes**：只报"重排生效了"却不报"有一批条连分都没拿到"，
   等于把一次有边界的改进说成了一次全面的改进（见 ``rerank`` 的窗口纪律）。
6. **判定不靠字符串比较**（day069 新增）。"这段答案有没有依据"由**编号**判定
   （``GroundingReport`` 的三个集合），"这次调没调模型"由 ``llm_called`` 判定；
   不许"看 answer 等不等兜底串""看答案里有没有'没有相关内容'"。
   兜底文案改一个字、模型换个说法，字符串判定就会静默失效——
   而它失效的样子恰好是"看起来一切正常"。

## 与既有包的接缝

- **上游**：``vectorstore``（day064）提供库与六个原语、``indexing``（day065）
  提供清单与 ``compare_with_store``、``llm.embedding``（day041）提供编码器、
  ``chunking``（day062）的 ``heading_path`` / ``parent_doc_id`` 元数据在这里
  终于被用上（过滤、分组、引用）；``llm.base.BaseLLM``（day02x）提供模型接口，
  day069 的生成器只依赖它（不认识任何具体提供方）；
- **脚下**：``config`` 的 ``retrieval_*`` 组（这一组**不改变任何产物字节**，
  只改变"怎么问"——与 ``indexing_*`` 的分工见 ``config``）。
  day067 补的六项（``retrieval_hybrid_*`` / ``retrieval_bm25_*``）同属这一类，
  但多了一条："新能力默认不生效"（``retrieval_hybrid_enabled=False``）；
  day068 补的六项（``retrieval_rerank_*``）同样是这一类、同样默认不生效——
  而且同样符合"不改变任何产物字节"：重排只改**次序**，
  它不写库、不重算向量、不改清单（"改次序"不会让任何文件变一个字节）；
  day069 补的四项（``retrieval_prompt_version`` / ``retrieval_answer_*`` /
  ``retrieval_require_citation``）仍是这一类，但它们比前两组**重**一档：
  换提示词版本会**改变答案文本本身**，因此它是一次需要留痕的实验；
- **下游**：day068 的重排消费**融合之后**那份名单（``RetrievalHit.channels``
  与 ``RetrievalResult.fusion`` 让它知道"这一条是几路证据支持的"，
  而重排的收益正是靠"融合已经把证据摆出来了"才量得准）；
  它自己又给 day071 的 RAG 评估留下 ``measure_lift``：三条指标只依赖
  "名单 + 金标准"，因此**替身也能跑出一份前后对照表**
  （``aggregate_results`` 与 ``IndexState.drift`` 仍在评估里被用）；
  day069 的生成器给 day071 留下 ``GroundingReport``（三组编号 + 覆盖率）
  与 ``Generation.fallback_reason``：**"换一版提示词之后可核对率变了吗"
  这件事从此有一份可统计的账**；
- **端点**：``api.routes`` 的 ``/retrieval/*``（day066 五个 + day067 两个）；
  手册在 ``docs/retrieval.md`` 与 ``docs/hybrid_retrieval.md``；
  离线演示 ``scripts/retrieval_demo.py`` 与 ``scripts/hybrid_demo.py``。

## 导出范围（十二个模块全部落地）

```text
已落地    errors / types / filters / retriever / routing / context / pipeline
          lexical / fusion / hybrid          ← day067 新增的三个
          rerank                             ← day068 新增的这个
          generation                         ← day069 新增的这个
```

因此本文件的导入**覆盖全部十二个模块**：一个包在 ``__init__`` 里
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
    FusionError,
    GenerationError,
    IndexStateError,
    LexicalError,
    QueryError,
    RerankError,
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
from smart_research_agent.retrieval.fusion import (
    DEFAULT_ALPHA,
    DEFAULT_RRF_K,
    FUSION_OVERRIDE_KEYS,
    FUSION_RRF,
    FUSION_STRATEGIES,
    FUSION_WEIGHTED,
    FusedHit,
    contribution_counts,
    fuse,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from smart_research_agent.retrieval.generation import (
    CURRENT_PROMPT,
    CURRENT_PROMPT_VERSION,
    CUSTOM_PROMPT_VERSION,
    DECLINE_MARKERS,
    DEFAULT_PROMPT_VERSION,
    FALLBACK_NO_CONTEXT,
    FALLBACK_REASON_DESCRIPTIONS,
    FALLBACK_REASON_EMPTY_REPLY,
    FALLBACK_REASON_LLM_ERROR,
    FALLBACK_REASON_MODEL_DECLINED,
    FALLBACK_REASON_NO_CONTEXT,
    FALLBACK_REASON_NONE,
    FALLBACK_REASON_UNUSABLE_CITATIONS,
    FALLBACK_REASONS,
    GENERATION_OVERRIDE_KEYS,
    PROMPT_CHANGELOG,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSIONS,
    RAG_PROMPT_V1,
    RAG_PROMPT_V2,
    RAG_PROMPTS,
    CitationCheck,
    Generation,
    GroundingReport,
    RAGGenerator,
    prompt_template_of,
)
from smart_research_agent.retrieval.hybrid import (
    DEFAULT_HYBRID_NAME,
    HybridRetriever,
    build_hybrid_retriever,
    hybrid_enabled,
)
from smart_research_agent.retrieval.lexical import (
    DEFAULT_B,
    DEFAULT_K1,
    BM25Params,
    LexicalDocument,
    LexicalHit,
    LexicalIndex,
    LexicalSearchResult,
    tokenize,
)
from smart_research_agent.retrieval.pipeline import (
    RAG_ANSWER_PROMPT,
    RAG_ANSWER_PROMPT_V1,
    RAG_ANSWER_PROMPT_VERSION,
    RagAnswer,
    RagPipeline,
)
from smart_research_agent.retrieval.rerank import (
    DEFAULT_RERANK_BATCH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RERANK_WEIGHT,
    IDEAL_LEN,
    RERANK_FEATURE_NAMES,
    RERANK_MODE_BLEND,
    RERANK_MODE_REPLACE,
    RERANK_MODES,
    RERANK_OVERRIDE_KEYS,
    BaseReranker,
    CrossEncoderReranker,
    LiftProbe,
    LiftReport,
    RerankFeatures,
    RerankHit,
    RerankResult,
    measure_lift,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    rerank_hits,
)
from smart_research_agent.retrieval.retriever import Retriever, build_retriever
from smart_research_agent.retrieval.routing import RouteDecision, StoreRouter
from smart_research_agent.retrieval.types import (
    CHANNEL_BM25,
    CHANNEL_VECTOR,
    CHANNELS,
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
    "CHANNELS",
    "CHANNEL_BM25",
    "CHANNEL_VECTOR",
    "CURRENT_PROMPT",
    "CURRENT_PROMPT_VERSION",
    "CUSTOM_PROMPT_VERSION",
    "DECLINE_MARKERS",
    "DEFAULT_ALPHA",
    "DEFAULT_B",
    "DEFAULT_DOC_ID_FIELD",
    "DEFAULT_FETCH_MULTIPLIER",
    "DEFAULT_HYBRID_NAME",
    "DEFAULT_K1",
    "DEFAULT_MIN_HIT_CHARS",
    "DEFAULT_PROMPT_VERSION",
    "DEFAULT_RERANK_BATCH",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_RERANK_WEIGHT",
    "DEFAULT_RRF_K",
    "DEFAULT_TIME_FIELD",
    "EMPTY_REASON_BELOW_THRESHOLD",
    "EMPTY_REASON_DESCRIPTIONS",
    "EMPTY_REASON_DIVERSITY",
    "EMPTY_REASON_FILTERED_OUT",
    "EMPTY_REASON_NONE",
    "EMPTY_REASON_NO_DATA",
    "EMPTY_REASONS",
    "FALLBACK_NO_CONTEXT",
    "FALLBACK_REASON_DESCRIPTIONS",
    "FALLBACK_REASON_EMPTY_REPLY",
    "FALLBACK_REASON_LLM_ERROR",
    "FALLBACK_REASON_MODEL_DECLINED",
    "FALLBACK_REASON_NO_CONTEXT",
    "FALLBACK_REASON_NONE",
    "FALLBACK_REASON_UNUSABLE_CITATIONS",
    "FALLBACK_REASONS",
    "FUSION_OVERRIDE_KEYS",
    "FUSION_RRF",
    "FUSION_STRATEGIES",
    "FUSION_WEIGHTED",
    "GENERATION_OVERRIDE_KEYS",
    "IDEAL_LEN",
    "MAX_FETCH_K",
    "PROMPT_CHANGELOG",
    "PROMPT_VERSION_V1",
    "PROMPT_VERSION_V2",
    "PROMPT_VERSIONS",
    "RAG_ANSWER_PROMPT",
    "RAG_ANSWER_PROMPT_V1",
    "RAG_ANSWER_PROMPT_VERSION",
    "RAG_PROMPT_V1",
    "RAG_PROMPT_V2",
    "RAG_PROMPTS",
    "RERANK_FEATURE_NAMES",
    "RERANK_MODES",
    "RERANK_MODE_BLEND",
    "RERANK_MODE_REPLACE",
    "RERANK_OVERRIDE_KEYS",
    "RETRIEVAL_LIMITATIONS",
    "RETRIEVAL_OUT_OF_SCOPE",
    "TIME_FIELD_SUFFIXES",
    "TRUNCATION_MARKER",
    "BM25Params",
    "BaseReranker",
    "Citation",
    "CitationCheck",
    "ContextError",
    "CrossEncoderReranker",
    "FusedHit",
    "FusionError",
    "Generation",
    "GenerationError",
    "GroundingReport",
    "HybridRetriever",
    "IndexState",
    "IndexStateError",
    "LexicalDocument",
    "LexicalError",
    "LexicalHit",
    "LexicalIndex",
    "LexicalSearchResult",
    "LiftProbe",
    "LiftReport",
    "PackedContext",
    "QueryError",
    "RAGGenerator",
    "RagAnswer",
    "RagPipeline",
    "RerankError",
    "RerankFeatures",
    "RerankHit",
    "RerankResult",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "RouteDecision",
    "StoreRouter",
    "TimeRange",
    "aggregate_results",
    "build_hybrid_retriever",
    "build_retriever",
    "combine_where",
    "contribution_counts",
    "describe_conditions",
    "fuse",
    "hybrid_enabled",
    "measure_lift",
    "ndcg_at_k",
    "pack_context",
    "prompt_template_of",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_fusion",
    "rerank_hits",
    "time_field_of",
    "time_range_clause",
    "tokenize",
    "unknown_filter_fields",
    "validate_where",
    "weighted_score_fusion",
]
