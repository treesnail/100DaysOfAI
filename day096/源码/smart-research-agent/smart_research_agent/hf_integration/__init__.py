"""``hf_integration``：把 Hugging Face 生态**接进来**（M7-D10 / day086）.

day085 把 ``modeling_gpt2.py`` 与 ``modeling_bert.py`` 的那几行读成了**行为**；
今天要做的不是"再读一遍"，而是**把那份东西装进自己的程序里**。
装进来这件事比读它多出四样东西，而它们**一样都不在模型里**：

```text
① Hub 与缓存      from_pretrained 的第一步不是"加载权重"，而是"把文件找齐"：
                  repo_id → models--org--name/snapshots/<commit>/<相对路径>
② 配置即形状      config.json 里那几个整数决定了每一层有多宽，而参数量可以由它**算出来**
③ 分词器          两个文件（vocab.json + merges.txt）与一段纯 Python 算术
④ 管线与池化      从 token 到向量之间还差一次池化——而**池化是被掩码约束的**
```

## 一、今天最值钱的一句话

> **"接进来"要接的不是代码，是接口面。**

day045 的 ``LocalModel`` 走的是一条协议路线（Ollama / vLLM 的 OpenAI 兼容端点）；
今天出现第三条路线：**模型就在这个进程里**。三条路线并存时，上层代码凭什么一行不改？
答案是**方法面一致**——``chat`` / ``stream`` / ``chat_with_tools`` / ``is_available`` /
``describe`` / ``model_name`` / ``backend`` / ``context_length``，八项逐名对齐。
本包把这张清单写成一个常量（:data:`bridge.PROTOCOL_SURFACE`），
并让 :func:`verify.check_protocol_surface_matches` 逐名核对——**一条能被失败的**对账。

## 二、十条性质（判据分三类，三类用三种比法）

```text
逐位（==）       同一次算术里的两个量：两种填充宽度下的池化、批量与逐条、
                 encode/decode 往返、本包的一层 vs day085 的那个块
整数相等         参数量是两个整数：本包公式 vs 库读数、权重 vs 配置口径
逐名对齐         接口面是一串名字：在进程模型 vs LocalModel
```

## 三、十一个模块

```text
errors.py    七个失败族（**缺席的仍是 GradientError**，理由换了一条）
types.py     缓存布局 / 两个架构的键名映射 / 池化三法 / 两条管线 / 十条性质 / 十二条生态笔记 / 五条边界
hub.py       名字 → commit → 文件 → 实体：四段结构与**逐文件命中**
config.py    config.json → ModelCard，并由配置**算出**参数量（与真实库整数相等）
tokenizer.py byte-level BPE：两个文件 + 一个循环 + 一个被复刻的怪癖
forward.py   按卡片把 day079~085 的零件串成一次真前向（填充何时惰性、样本边界何时会串味）
features.py  池化三法 + 那个**只差一个分母**的 bug
pipeline.py  两条管线（零数学的编排层）
bridge.py    接进 local_model：八项方法面 + 三条路线的客观事实
verify.py    十条性质与三条跨天对账
study.py     六张表
```

## 四、五条纪律

1. **每一行都要带两个数**：命中/下载、配置/库、真实长度/总宽度、safe/naive。
   一行只有"通过"的表没法被反驳（day082 的探针那条纪律）。
2. **"不适用" ≠ "通过"**：``applicable=False`` 时 ``passed`` 也必须是 ``False``
   （构造期就拒绝）。今天有两条真的会走到"不适用"——
   参数量对账只对**两个记录在案的模型**适用，块对账只在 ``L=1 且 heads=1`` 时适用。
3. **不可能失败的判据不算判据**：逐文件缓存那条性质必须**从空缓存开始跑**，
   否则第一次解析本身就是一次命中，"下载 0 个"这句话永远成立。
4. **差值必须被逐项解释**：本包权重与配置口径的参数量之差**恰好**是
   ``4·hidden·layers``（注意力偏置），而两侧的拆法不同——
   GPT-2 是 ``3h + h``，BERT 是 ``h + h + h + h``。
5. **"因果 ⇒ 填充惰性"只在单条样本内成立**：把几条样本拼成一条长序列之后，
   那张 ``(n, n)`` 的下三角会让第 2 条看到第 1 条的全部 token。
   正确做法是把掩码升级成**段内下三角**（``block_causal_mask``）——
   也就是真实实现里那张 ``(batch, 1, seq, seq)``。

## 五、与既有包的接缝

- **上游**：``hf_source``（day085：块、注意力、生成四策略、`SourceShape`/`GenerationSettings`）、
  ``encoder_decoder``（day079：`BlockParameters` / `layer_norm`）、
  ``transformer_core``（day075：`AttentionParams`）、``math_foundations``（day073：LCG）；
- **脚下**：``config.py`` **没有**新增配置项——卡片、种子、策略都是函数参数；
- **下游**：day087（高效推理与量化）会从今天这条**在进程推理**的路径上接下去
  （KV Cache 与量化都只作用于推理，而今天的 `hidden_states` 正是那条推理路径）；
  day088（项目底层原理串联）用今天这六张表写"原理 → 应用"的提纲。
"""

from __future__ import annotations

from smart_research_agent.hf_integration.bridge import (
    ALL_BACKENDS,
    BACKEND_TRAITS,
    IN_PROCESS,
    PLACEHOLDER_KEYS,
    PROTOCOL_SURFACE,
    BackendChoice,
    InProcessModel,
    InProcessSpec,
    choose_backend,
    create_in_process_model,
    render_messages,
    trait_line,
)
from smart_research_agent.hf_integration.config import (
    DEFAULT_ACTIVATIONS,
    DEFAULT_LN_EPS,
    DEFAULT_TIE,
    DEFAULT_TYPE_VOCAB,
    KNOWN_ACTIVATIONS,
    architecture_of,
    card_from_snapshot,
    check_snapshot_is_a_model,
    describe,
    ffn_of,
    missing_keys,
    parameter_count,
    parse_config,
    profile_agreement,
    reference_cards,
    tie_delta,
    tiny_card,
    untied_card,
)
from smart_research_agent.hf_integration.errors import (
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    AssemblyError,
    ConfigError,
    HubError,
    IntegrationError,
    NumericError,
    ParameterError,
    ShapeError,
    TokenError,
)
from smart_research_agent.hf_integration.features import (
    POOLING_RULES,
    PoolingComparison,
    compare_pooling,
    cosine_similarity,
    l2_normalize,
    max_gap,
    pool_forward,
    pool_masked,
    pool_rows,
    pooling_line,
    tolerance_of,
)
from smart_research_agent.hf_integration.forward import (
    BIAS_BREAKDOWN,
    BIAS_ITEMS_PER_LAYER,
    LAYER_CONFIG_SCALE,
    LAYER_WEIGHTS_SCALE,
    POLICIES,
    POLICY_NAIVE,
    POLICY_SAFE,
    HiddenBatch,
    ModelWeights,
    bias_breakdown_line,
    block_causal_mask,
    embed_tokens,
    embedding_norm_line,
    hidden_states,
    logits,
    make_logits_fn,
    make_weights,
    pooler_output,
    run_one_block,
)
from smart_research_agent.hf_integration.hub import (
    DEFAULT_CACHE_DIR,
    REQUIRED_FILES,
    CacheStore,
    HubClient,
    HubResolver,
    InMemoryHub,
    blob_path,
    check_readable,
    flat_paths,
    known_offered,
    repo_folder,
    select_files,
    sha256_of,
    snapshot_root,
)
from smart_research_agent.hf_integration.pipeline import (
    HEAD_OF_TASK,
    FeatureOutput,
    GenerationOutput,
    check_head_available,
    feature_extraction,
    head_line,
    run_both,
    text_generation,
)
from smart_research_agent.hf_integration.study import (
    BATCH_TEXTS,
    GENERATION_CASES,
    TOKEN_SAMPLES,
    BatchRow,
    CacheRow,
    GenerationRow,
    ParamRow,
    PoolingRow,
    TokenRow,
    batch_rows,
    cache_rows,
    generation_rows,
    note_lines,
    param_rows,
    pooling_rows,
    study_lines,
    token_rows,
)
from smart_research_agent.hf_integration.tokenizer import (
    CONTRACTION_SUFFIXES,
    MERGES_HEADER,
    PADDING_LONGEST,
    PADDING_MAX_LENGTH,
    PADDING_STRATEGIES,
    ByteBPETokenizer,
    build_tiny_tokenizer,
    bytes_to_unicode,
    get_pairs,
    merge_ranks,
    parse_merges,
    pretokenize,
    split_on_specials,
)
from smart_research_agent.hf_integration.types import (
    ARCHITECTURES,
    ARCHITECTURE_BERT,
    ARCHITECTURE_DESCRIPTIONS,
    ARCHITECTURE_GPT2,
    BERT_BASE_PARAMETERS,
    BERT_DEFAULT_INTERMEDIATE,
    BLOB_DIR,
    CACHE_LAYOUT,
    CACHE_PREFIX,
    CONFIG_FILE,
    DEFAULT_REVISION,
    FEATURE_STAGES,
    FEATURE_STAGE_SHAPES,
    FFN_RATIO,
    GPT2_LEGACY_VOCAB,
    GPT2_SMALL_PARAMETERS,
    HUB_LIBRARY,
    HUB_VERSION,
    INTEGRATION_BOUNDARIES,
    INTEGRATION_NOTES,
    INTEGRATION_NOTES_ORDER,
    INTEGRATION_PROPERTIES,
    KNOWN_FILES,
    MERGES_FILE,
    POOLING_DESCRIPTIONS,
    POOLING_LAST_TOKEN,
    POOLING_MAX,
    POOLING_MEAN,
    POOLING_STRATEGIES,
    POOLING_TOLERANCE,
    PROPERTY_BATCH_MATCHES_SINGLE,
    PROPERTY_BLOCK_MATCHES_DAY085,
    PROPERTY_CACHE_IS_PER_FILE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FAILURE,
    PROPERTY_GENERATION_MATCHES_DAY085,
    PROPERTY_PARAMS_MATCH_LIBRARY,
    PROPERTY_POOLING_RESPECTS_MASK,
    PROPERTY_PROTOCOL_SURFACE_MATCHES,
    PROPERTY_ROUND_TRIP_IS_EXACT,
    PROPERTY_SNAPSHOT_IS_FLAT,
    PROPERTY_WEIGHTS_GAP_IS_EXPLAINED,
    REF_DIR,
    REQUIRED_CONFIG_KEYS,
    SIZE_KEYS,
    SNAPSHOT_DIR,
    TASK_DESCRIPTIONS,
    TASK_FEATURE_EXTRACTION,
    TASK_HEADS,
    TASK_KINDS,
    TASK_TEXT_GENERATION,
    TRANSFORMERS_LIBRARY,
    TRANSFORMERS_VERSION,
    TRANSFORMERS_VERSION_SAMPLE,
    VOCAB_FILE,
    WEIGHTS_FILE,
    CacheEntry,
    Encoding,
    IntegrationRecord,
    ModelCard,
    PooledBatch,
    Snapshot,
    TextBatch,
)
from smart_research_agent.hf_integration.verify import (
    LIBRARY_RECORDS,
    LIBRARY_RECORD_SOURCE,
    ROUND_TRIP_SAMPLES,
    CrossCheck,
    PropertyOutcome,
    PropertyReport,
    check_all,
    check_batch_matches_single,
    check_block_matches_day085,
    check_cache_is_per_file,
    check_generation_matches_day085,
    check_params_match_library,
    check_pooling_respects_mask,
    check_protocol_surface_matches,
    check_round_trip_is_exact,
    check_snapshot_is_flat,
    check_weights_gap_is_explained,
    finite_or_raise,
    reference_agreement_lines,
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "ALL_BACKENDS",
    "ARCHITECTURES",
    "ARCHITECTURE_BERT",
    "ARCHITECTURE_DESCRIPTIONS",
    "ARCHITECTURE_GPT2",
    "BACKEND_TRAITS",
    "BATCH_TEXTS",
    "BERT_BASE_PARAMETERS",
    "BERT_DEFAULT_INTERMEDIATE",
    "BIAS_BREAKDOWN",
    "BIAS_ITEMS_PER_LAYER",
    "BLOB_DIR",
    "CACHE_LAYOUT",
    "CACHE_PREFIX",
    "CONFIG_FILE",
    "CONTRACTION_SUFFIXES",
    "DEFAULT_ACTIVATIONS",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LN_EPS",
    "DEFAULT_REVISION",
    "DEFAULT_TIE",
    "DEFAULT_TYPE_VOCAB",
    "FAMILY_OUTCOMES",
    "FEATURE_STAGES",
    "FEATURE_STAGE_SHAPES",
    "FFN_RATIO",
    "GENERATION_CASES",
    "GPT2_LEGACY_VOCAB",
    "GPT2_SMALL_PARAMETERS",
    "HEAD_OF_TASK",
    "HUB_LIBRARY",
    "HUB_VERSION",
    "IN_PROCESS",
    "INTEGRATION_BOUNDARIES",
    "INTEGRATION_NOTES",
    "INTEGRATION_NOTES_ORDER",
    "INTEGRATION_PROPERTIES",
    "KNOWN_ACTIVATIONS",
    "KNOWN_FILES",
    "LAYER_CONFIG_SCALE",
    "LAYER_WEIGHTS_SCALE",
    "LIBRARY_RECORDS",
    "LIBRARY_RECORD_SOURCE",
    "MERGES_FILE",
    "MERGES_HEADER",
    "PADDING_LONGEST",
    "PADDING_MAX_LENGTH",
    "PADDING_STRATEGIES",
    "PLACEHOLDER_KEYS",
    "POLICIES",
    "POLICY_NAIVE",
    "POLICY_SAFE",
    "POOLING_DESCRIPTIONS",
    "POOLING_LAST_TOKEN",
    "POOLING_MAX",
    "POOLING_MEAN",
    "POOLING_RULES",
    "POOLING_STRATEGIES",
    "POOLING_TOLERANCE",
    "PROPERTY_BATCH_MATCHES_SINGLE",
    "PROPERTY_BLOCK_MATCHES_DAY085",
    "PROPERTY_CACHE_IS_PER_FILE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_GENERATION_MATCHES_DAY085",
    "PROPERTY_PARAMS_MATCH_LIBRARY",
    "PROPERTY_POOLING_RESPECTS_MASK",
    "PROPERTY_PROTOCOL_SURFACE_MATCHES",
    "PROPERTY_ROUND_TRIP_IS_EXACT",
    "PROPERTY_SNAPSHOT_IS_FLAT",
    "PROPERTY_WEIGHTS_GAP_IS_EXPLAINED",
    "PROTOCOL_SURFACE",
    "REF_DIR",
    "REQUIRED_CONFIG_KEYS",
    "REQUIRED_FILES",
    "ROUND_TRIP_SAMPLES",
    "SIZE_KEYS",
    "SNAPSHOT_DIR",
    "TASK_DESCRIPTIONS",
    "TASK_FEATURE_EXTRACTION",
    "TASK_HEADS",
    "TASK_KINDS",
    "TASK_TEXT_GENERATION",
    "TOKEN_SAMPLES",
    "TRANSFORMERS_LIBRARY",
    "TRANSFORMERS_VERSION",
    "TRANSFORMERS_VERSION_SAMPLE",
    "VOCAB_FILE",
    "WEIGHTS_FILE",
    "AssemblyError",
    "BackendChoice",
    "BatchRow",
    "ByteBPETokenizer",
    "CacheEntry",
    "CacheRow",
    "CacheStore",
    "ConfigError",
    "CrossCheck",
    "Encoding",
    "FeatureOutput",
    "GenerationOutput",
    "GenerationRow",
    "HiddenBatch",
    "HubClient",
    "HubError",
    "HubResolver",
    "InMemoryHub",
    "InProcessModel",
    "InProcessSpec",
    "IntegrationError",
    "IntegrationRecord",
    "ModelCard",
    "ModelWeights",
    "NumericError",
    "ParamRow",
    "ParameterError",
    "PooledBatch",
    "PoolingComparison",
    "PoolingRow",
    "PropertyOutcome",
    "PropertyReport",
    "ShapeError",
    "Snapshot",
    "TextBatch",
    "TokenError",
    "TokenRow",
    "architecture_of",
    "batch_rows",
    "bias_breakdown_line",
    "blob_path",
    "block_causal_mask",
    "build_tiny_tokenizer",
    "bytes_to_unicode",
    "cache_rows",
    "card_from_snapshot",
    "check_all",
    "check_batch_matches_single",
    "check_block_matches_day085",
    "check_cache_is_per_file",
    "check_generation_matches_day085",
    "check_head_available",
    "check_params_match_library",
    "check_pooling_respects_mask",
    "check_protocol_surface_matches",
    "check_readable",
    "check_round_trip_is_exact",
    "check_snapshot_is_a_model",
    "check_snapshot_is_flat",
    "check_weights_gap_is_explained",
    "choose_backend",
    "compare_pooling",
    "cosine_similarity",
    "create_in_process_model",
    "describe",
    "embed_tokens",
    "embedding_norm_line",
    "feature_extraction",
    "ffn_of",
    "finite_or_raise",
    "flat_paths",
    "generation_rows",
    "get_pairs",
    "head_line",
    "hidden_states",
    "known_offered",
    "l2_normalize",
    "logits",
    "make_logits_fn",
    "make_weights",
    "max_gap",
    "merge_ranks",
    "missing_keys",
    "note_lines",
    "param_rows",
    "parse_config",
    "parse_merges",
    "parameter_count",
    "pool_forward",
    "pool_masked",
    "pool_rows",
    "pooler_output",
    "pooling_line",
    "pooling_rows",
    "pretokenize",
    "profile_agreement",
    "reference_agreement_lines",
    "reference_cards",
    "render_messages",
    "repo_folder",
    "run_both",
    "run_one_block",
    "select_files",
    "sha256_of",
    "snapshot_root",
    "split_on_specials",
    "study_lines",
    "text_generation",
    "tie_delta",
    "tiny_card",
    "tolerance_of",
    "token_rows",
    "trait_line",
    "untied_card",
]
