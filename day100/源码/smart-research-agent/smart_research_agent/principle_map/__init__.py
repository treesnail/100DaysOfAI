"""``principle_map``：把底层原理串成一张可校验的图（M7-D12 / day088）.

day087 把推理路径上的三笔账算清了；今天是 M7 阶段的**最后一天**，
动作只有一个：**把十七天里散落的底层原理串起来**。

```text
原理（四层共 12 条命题）
  → 实现（项目里的包与函数）
    → 应用（Agent/RAG 的六个能力）
```

## 一、今天最值钱的一句话

> **原理只有落到一个能被解析的函数上、并且能被现场量出读数，才算真的"串起来"了。**

这句话有一个立刻可用的推论：一张图里如果某条原理指不到任何函数、或者某个应用
没有任何原理支撑，那它就不是"知识"，而是一句口号。因此本课把"图是否完整"
写成一条可断言的性质。

## 二、七条性质（判据分四类）

```text
存在性      每条原理都有实现落点 / 每个应用都至少被一条原理支撑
可复现性    同一个探针连续两次调用得到的证据**逐位相同**（单次计算、可重复）
跨天对账    注意力每一行是分布（调 transformer_core）、缓存公式与 day087 一致（调 inference_optim）
次序/完整性 提纲满足前置依赖（数学 → 注意力 → 表征 → 推理）、文档覆盖全部十二条
```

第二类是本课唯一一处"两个数相等"与"一个数不超过另一个数"混在一起的地方——
因此 ``CrossCheck`` 带一个 ``upper_bound`` 字段（与 day087 同源）：有它时判据是"≤"，
没有时才是"=="。把两类混成一个判据，就会出现"实测误差恰好等于 0（因为输入全是 0）
被当成通过"这种事。

## 三、九个模块

```text
errors.py     七个失败族（**缺席的仍是 GradientError**，理由第四次换了一条：本日一个新式子都没有）
types.py      四个层 / 六个应用 / 十二条命题 / 七条性质 / 十条笔记 / 五条边界
claims.py     十二块拼图的定义与查询：principles / by_layer / by_application / principle
reconcile.py  十二个探针：真的 import 并调用既有包，把读数算出来
graph.py      原理 → 实现 → 应用：build_graph / coverage_report / resolve_artifact
outline.py    分享提纲生成器 + 原理文档渲染器（10 节）
verify.py     七条性质与两类判据（含两条跨天对账）
study.py      五张表（原理 / 分层覆盖 / 应用覆盖 / 证据 / 提纲）
__init__.py   本文件
```

## 四、四条纪律

1. **原理必须能被解析**：``artifact`` 是"模块.函数"的字符串，能被 ``importlib`` 解析出来——
   指不到东西的箭头是装饰。
2. **读数必须现场算出**：``claims`` 里只有命题、没有数字；数字全部来自 ``reconcile`` 的探针。
   某一天底层实现改了，读数会跟着改，而不会"悄悄地不再是真的"。
3. **图必须单次计算、可重复**：默认那份图用 ``lru_cache`` 缓存，但每一个读数都确定——
   同一个探针两次调用逐位相同。
4. **次序必须能被检查**：提纲的层次序由 ``outline.check_order`` 做拓扑检验，
   讲反了抛 ``OrderError``——因为"讲反了"与"讲对了"在文本上都是合法的 markdown。

## 五、与既有包的接缝（day075~day087 的传承）

- **上游（十二块拼图的落点）**：
  ``math_foundations``（day073：``scaled_dot_product_attention`` / ``cosine``）、
  ``transformer_core``（day075：``self_attention`` / ``resolve_mask`` /
  ``assert_rows_are_distributions`` / ``compare_with_retrieval``）、
  ``hf_source``（day085：``split_heads`` / ``merge_heads``）、
  ``positional_encoding``（day078：``inject``）、
  ``vectorstore``（day064：``cosine_similarity``）、
  ``inference_optim``（day087：``compare_with_recompute`` / ``cache_bytes`` /
  ``error_bound`` / ``plan``）；
- **脚下**：``config`` **没有**新增配置项——层名、应用名、时长、容差都是函数参数；
- **下游**：day099（结业项目）会把这张图当成"部署前的第一张检查单"：
  任何一条没有落点的能力，都会在那一天以"跑不起来"的形式出现。
"""

from __future__ import annotations

from smart_research_agent.principle_map.claims import (
    PRINCIPLE_DEFS,
    application_of,
    applications,
    by_application,
    by_layer,
    layer_of,
    principle,
    principles,
)
from smart_research_agent.principle_map.errors import (
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    BridgeError,
    ClaimError,
    CoverageError,
    NumericError,
    OrderError,
    ParameterError,
    ReferenceError,
    ShapeError,
)
from smart_research_agent.principle_map.graph import (
    PrincipleGraph,
    application_coverage,
    build_graph,
    coverage_report,
    graph_lines,
    layer_coverage,
    orphan_principles,
    resolve_artifact,
    supported_applications,
    unsupported_applications,
)
from smart_research_agent.principle_map.outline import (
    DOCUMENT_SECTIONS,
    PROJECT_ROOT,
    SECTION_DEMOS,
    SECTION_MINUTES,
    SECTION_TITLES,
    build_outline,
    check_order,
    demo_scripts,
    document_lines,
    ensure_document_covers_all,
    missing_demos,
    missing_principles,
    outline_lines,
    outline_minutes,
    render_document,
)
from smart_research_agent.principle_map.reconcile import (
    EVIDENCE_TOLERANCE,
    PROBES,
    PROBE_NEW_TOKEN,
    PROBE_PROMPT,
    evidence_for,
    evidence_sources,
    probe_all,
    probe_attention_is_differentiable_retrieval,
    probe_attention_rows_are_distributions,
    probe_cache_bytes_is_a_formula,
    probe_cache_reuse_is_bitwise_exact,
    probe_causal_mask_blocks_future,
    probe_cosine_is_normalized_dot,
    probe_embedding_similarity_is_direction,
    probe_generation_respects_budget_breakdown,
    probe_multi_head_splits_inside_projection,
    probe_position_encoding_breaks_permutation,
    probe_quantization_error_bounded_by_half_step,
    probe_rank_ordering_matches_attention_peaks,
    probe_source,
)
from smart_research_agent.principle_map.study import (
    ApplicationRow,
    EvidenceRow,
    LayerRow,
    OutlineRow,
    PrincipleRow,
    application_rows,
    evidence_rows,
    layer_rows,
    note_lines,
    outline_rows,
    principle_rows,
    study_lines,
)
from smart_research_agent.principle_map.types import (
    APPLICATIONS,
    APPLICATION_DESCRIPTIONS,
    APP_AGENT_REASONING,
    APP_EXPLAINABILITY,
    APP_GENERATION,
    APP_RAG_RERANK,
    APP_RAG_RETRIEVAL,
    APP_SERVING,
    LAYERS,
    LAYER_ATTENTION,
    LAYER_DESCRIPTIONS,
    LAYER_INFERENCE,
    LAYER_MATH,
    LAYER_ORDER,
    LAYER_REPRESENTATION,
    PRINCIPLES,
    PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL,
    PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PRINCIPLE_BOUNDARIES,
    PRINCIPLE_CACHE_BYTES_IS_A_FORMULA,
    PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT,
    PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE,
    PRINCIPLE_COSINE_IS_NORMALIZED_DOT,
    PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION,
    PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN,
    PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION,
    PRINCIPLE_NOTES,
    PRINCIPLE_NOTES_ORDER,
    PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION,
    PRINCIPLE_PROPERTIES,
    PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP,
    PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS,
    PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_CACHE_FORMULA_MATCHES_DAY087,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES,
    PROPERTY_EVIDENCE_IS_REPRODUCIBLE,
    PROPERTY_EVERY_APPLICATION_IS_SUPPORTED,
    PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT,
    PROPERTY_FAILURE,
    PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES,
    Application,
    CoverageReport,
    Evidence,
    Principle,
    TalkSection,
    require_application,
    require_layer,
    require_principle_id,
)
from smart_research_agent.principle_map.verify import (
    ROW_TOLERANCE,
    CrossCheck,
    PropertyOutcome,
    PropertyReport,
    check_all,
    check_attention_rows_are_distributions,
    check_cache_formula_matches_day087,
    check_document_covers_all_principles,
    check_evidence_is_reproducible,
    check_every_application_is_supported,
    check_every_principle_has_artifact,
    check_outline_respects_dependencies,
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "APPLICATIONS",
    "APPLICATION_DESCRIPTIONS",
    "APP_AGENT_REASONING",
    "APP_EXPLAINABILITY",
    "APP_GENERATION",
    "APP_RAG_RERANK",
    "APP_RAG_RETRIEVAL",
    "APP_SERVING",
    "Application",
    "ApplicationRow",
    "BridgeError",
    "ClaimError",
    "CoverageError",
    "CoverageReport",
    "CrossCheck",
    "DOCUMENT_SECTIONS",
    "EVIDENCE_TOLERANCE",
    "Evidence",
    "EvidenceRow",
    "FAMILY_OUTCOMES",
    "LAYERS",
    "LAYER_ATTENTION",
    "LAYER_DESCRIPTIONS",
    "LAYER_INFERENCE",
    "LAYER_MATH",
    "LAYER_ORDER",
    "LAYER_REPRESENTATION",
    "LayerRow",
    "NumericError",
    "OrderError",
    "OutlineRow",
    "PROBES",
    "PROBE_NEW_TOKEN",
    "PROBE_PROMPT",
    "PROJECT_ROOT",
    "PRINCIPLES",
    "PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL",
    "PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS",
    "PRINCIPLE_BOUNDARIES",
    "PRINCIPLE_CACHE_BYTES_IS_A_FORMULA",
    "PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT",
    "PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE",
    "PRINCIPLE_COSINE_IS_NORMALIZED_DOT",
    "PRINCIPLE_DEFS",
    "PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION",
    "PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN",
    "PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION",
    "PRINCIPLE_NOTES",
    "PRINCIPLE_NOTES_ORDER",
    "PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION",
    "PRINCIPLE_PROPERTIES",
    "PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP",
    "PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS",
    "PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS",
    "PROPERTY_CACHE_FORMULA_MATCHES_DAY087",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES",
    "PROPERTY_EVIDENCE_IS_REPRODUCIBLE",
    "PROPERTY_EVERY_APPLICATION_IS_SUPPORTED",
    "PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT",
    "PROPERTY_FAILURE",
    "PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES",
    "ParameterError",
    "Principle",
    "PrincipleGraph",
    "PrincipleRow",
    "PropertyOutcome",
    "PropertyReport",
    "ROW_TOLERANCE",
    "ReferenceError",
    "SECTION_DEMOS",
    "SECTION_MINUTES",
    "SECTION_TITLES",
    "ShapeError",
    "TalkSection",
    "application_coverage",
    "application_of",
    "application_rows",
    "applications",
    "build_graph",
    "build_outline",
    "by_application",
    "by_layer",
    "check_all",
    "check_attention_rows_are_distributions",
    "check_cache_formula_matches_day087",
    "check_document_covers_all_principles",
    "check_evidence_is_reproducible",
    "check_every_application_is_supported",
    "check_every_principle_has_artifact",
    "check_order",
    "check_outline_respects_dependencies",
    "coverage_report",
    "demo_scripts",
    "document_lines",
    "ensure_document_covers_all",
    "evidence_for",
    "evidence_rows",
    "evidence_sources",
    "graph_lines",
    "layer_coverage",
    "layer_of",
    "layer_rows",
    "missing_demos",
    "missing_principles",
    "note_lines",
    "orphan_principles",
    "outline_lines",
    "outline_minutes",
    "outline_rows",
    "principle",
    "principle_rows",
    "principles",
    "probe_all",
    "probe_attention_is_differentiable_retrieval",
    "probe_attention_rows_are_distributions",
    "probe_cache_bytes_is_a_formula",
    "probe_cache_reuse_is_bitwise_exact",
    "probe_causal_mask_blocks_future",
    "probe_cosine_is_normalized_dot",
    "probe_embedding_similarity_is_direction",
    "probe_generation_respects_budget_breakdown",
    "probe_multi_head_splits_inside_projection",
    "probe_position_encoding_breaks_permutation",
    "probe_quantization_error_bounded_by_half_step",
    "probe_rank_ordering_matches_attention_peaks",
    "probe_source",
    "render_document",
    "require_application",
    "require_layer",
    "require_principle_id",
    "resolve_artifact",
    "study_lines",
    "supported_applications",
    "unsupported_applications",
]
