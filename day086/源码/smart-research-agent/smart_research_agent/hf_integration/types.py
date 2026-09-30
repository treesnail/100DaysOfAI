"""``hf_integration`` 的形状、口径表与生态笔记（day086 / M7-D10）.

day085 把 Hugging Face 的那几行读成了**行为**；今天要做的是把它**接进来**。
接进来这件事比读它多出四样东西，而它们都**不在模型里**：

```text
① Hub 与缓存      from_pretrained 的第一步不是"加载权重"，而是"把文件找齐"：
                  repo_id → models--org--name/snapshots/<commit>/<相对路径>
② 配置即形状      config.json 里那几个整数决定了模型的每一层有多宽；
                  而"参数量"可以由这几个整数**算出来**（本包给出两条精确公式）
③ 分词器          byte-level BPE 的两个文件（vocab.json + merges.txt）与一段纯 Python 算术
④ 管线与池化      从 token 到向量之间还差一次池化——而**池化是被掩码约束的**
```

本模块是这一天的"判据总表"：它把上面四样拆成可被断言的东西。

## 一条纪律（与 day085 同源，但落点不同）

```text
承诺      给定同样的文件与输入，本包复现"装载与调用"那几层的**计算与解析语义**
不承诺    某个文件在第几行、某个内部变量的名字、某个版本的缓存目录布局细节
```

day085 承诺的是"那几个类怎么算"，今天承诺的是"那几个文件怎么被找齐、被解释、
被喂进模型"。两者的**反证方式**也不同：day085 用扰动量（改一个数、看谁不变），
今天用**换一个来源**（换一个 revision、换一种 padding、换一次批量大小，看读数是否一致）。

## 三个刻意的"逐位"判据

本课的判据里，有三条用 ``==`` 而不是容差，因为它们的两个量来自**同一次算术**：

```text
encode → decode 往返        字节级 BPE 是无损的：往返必须逐位还原
池化不受 padding 影响        只对真实位置求和 ⇒ 追加填充**不改变任何一位**
参数量与真实库一致            整数相等 ⇒ 这就是"逐位"（没有舍入可言）
```

第二条值得单独说：它成立的前提是**因果掩码**。对双向模型，填充会真的改变
真实位置那一行的读数——本包因此把"填充是否惰性"写成一张表（:mod:`study`），
而不是一句"padding 无所谓"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.math_foundations.types import Vector

# --------------------------------------------------------------------------------------
# 0. 阅读与对接对象（版本号写进常量，避免"当时接的是哪一版"变成一个传说）
# --------------------------------------------------------------------------------------

#: 两个生态库（跨版本稳定）.
HUB_LIBRARY = "huggingface_hub"
TRANSFORMERS_LIBRARY = "huggingface/transformers"

#: ``huggingface_hub`` 的大版本.
HUB_VERSION = "1.x"

#: ``transformers`` 的版本（本机 venv 实测 ``5.17.0``，写进常量供文档与习题引用）.
TRANSFORMERS_VERSION = "5.17.0"

#: 写作时最新的一版（用于文档与习题里的"当时读数"标注）.
TRANSFORMERS_VERSION_SAMPLE = "v5.17.0（2026-09-09 发布）"

# --------------------------------------------------------------------------------------
# 1. 缓存布局（**这是"文件怎么被找齐"的答案**）
# --------------------------------------------------------------------------------------

#: 仓库目录前缀。``org/name`` 里的斜杠会被换成一个双横线。
CACHE_PREFIX = "models--"

#: 快照目录名（每个 commit 一个）。
SNAPSHOT_DIR = "snapshots"

#: 内容寻址目录名（按 sha256 存实体文件）。
BLOB_DIR = "blobs"

#: 引用目录名（``refs/main`` 里存的是那串 commit hash）。
REF_DIR = "refs"

#: 默认版本名（与 HF 的默认分支同名）。
DEFAULT_REVISION = "main"

#: 缓存布局的五个角色（**这张表要被测试逐键检查**）。
CACHE_LAYOUT: dict[str, str] = {
    "root": "缓存根目录（默认 $HF_HOME/hub 或 ~/.cache/huggingface/hub）",
    "refs": f"{REF_DIR}/<revision> —— 一个文本文件，内容是一串 commit hash",
    "snapshots": f"{SNAPSHOT_DIR}/<commit>/<相对路径> —— 与仓库里的目录结构一致（**平铺**）",
    "blobs": f"{BLOB_DIR}/<sha256> —— 实体文件；快照里的每个文件都指向其中一个",
    "folder": f"{CACHE_PREFIX}<org>--<name> —— 一个仓库一个目录",
}

#: 一次成功快照需要的四类文件（这是一份**最小**清单，不是完整清单）。
CONFIG_FILE = "config.json"
VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"
WEIGHTS_FILE = "model.safetensors"

#: 本包认得的四个文件名（``tokenizer.json`` 这类"另一种 tokenizer 格式"不在其中）。
KNOWN_FILES: tuple[str, ...] = (CONFIG_FILE, VOCAB_FILE, MERGES_FILE, WEIGHTS_FILE)

# --------------------------------------------------------------------------------------
# 2. 两个架构（同一个量，两个键名）
# --------------------------------------------------------------------------------------

ARCHITECTURE_GPT2 = "gpt2"
ARCHITECTURE_BERT = "bert"

ARCHITECTURES: tuple[str, ...] = (ARCHITECTURE_GPT2, ARCHITECTURE_BERT)

ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    ARCHITECTURE_GPT2: "gpt2：解码器（pre-LN + 因果注意力），位置表 n_positions，融合 QKV，无 token_type",
    ARCHITECTURE_BERT: "bert：编码器（post-LN + 双向注意力），位置表 max_position_embeddings，三次投影，有 token_type",
}

#: **同一个量在两个架构里的两个键名**——这是本课最便宜的一批判据。
SIZE_KEYS: dict[str, dict[str, str]] = {
    ARCHITECTURE_GPT2: {
        "hidden": "n_embd",
        "heads": "n_head",
        "layers": "n_layer",
        "positions": "n_positions",
        "ffn": "n_inner",
        "activation": "activation_function",
        "ln_eps": "layer_norm_epsilon",
    },
    ARCHITECTURE_BERT: {
        "hidden": "hidden_size",
        "heads": "num_attention_heads",
        "layers": "num_hidden_layers",
        "positions": "max_position_embeddings",
        "ffn": "intermediate_size",
        "activation": "hidden_act",
        "ln_eps": "layer_norm_eps",
    },
}

#: 每个架构**必须**在 ``config.json`` 里出现的键（缺一个就装不起来）。
REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    ARCHITECTURE_GPT2: (
        "model_type",
        "vocab_size",
        "n_embd",
        "n_head",
        "n_layer",
        "n_positions",
    ),
    ARCHITECTURE_BERT: (
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "max_position_embeddings",
    ),
}

#: GPT-2 的前馈扩张比：``n_inner`` **缺省时**取 ``4·hidden``（这是 GPT-2 的规矩）.
FFN_RATIO = 4

#: BERT 的前馈中间维默认值：``intermediate_size = 3072``，**与 hidden 无关**.
#:
#: 这是本课与真库对账时冒出来的一处**看起来一样、其实不一样**：
#:
#: ```text
#: bert-base-uncased   hidden=768  intermediate=3072  ⇒ 比例恰好 4.0（于是"都是 4 倍"看起来成立）
#: 一个 hidden=16 的小 BERT                                   ⇒ 比例 192.0（"4 倍"这句话立刻不成立）
#: ```
#:
#: 因此"前馈是 4 倍"对 BERT 只是**一个模型碰巧**，而对 GPT-2 才是**缺省规则**。
#: 这一条不是推理出来的，是拿真 ``BertConfig()`` 读出来的（见 ``tests/test_hf_real_bridge.py``）。
BERT_DEFAULT_INTERMEDIATE = 3072

#: GPT-2 的 ``vocab_size`` 必须覆盖 ``eos_token_id`` 这一条在真实库里的告警口径
#: （本机实测：``vocab_size=100`` 而 ``eos_token_id=50256`` 会打印一行告警，
#: 而 it **不报错**——因此本包把它写成一条可校验的约束）。
GPT2_LEGACY_VOCAB = 50257

#: bert-base-uncased 的参数量（**真实库读数**，本包用它做跨实现对账）.
BERT_BASE_PARAMETERS = 109_482_240

#: gpt2（124M）的参数量（同上）.
GPT2_SMALL_PARAMETERS = 124_439_808

# --------------------------------------------------------------------------------------
# 3. 池化三法（从 token 到向量之间的那一步）
# --------------------------------------------------------------------------------------

POOLING_LAST_TOKEN = "last_token"
POOLING_MEAN = "mean"
POOLING_MAX = "max"

POOLING_STRATEGIES: tuple[str, ...] = (POOLING_LAST_TOKEN, POOLING_MEAN, POOLING_MAX)

POOLING_DESCRIPTIONS: dict[str, str] = {
    POOLING_LAST_TOKEN: "取**最后一个真实位置**（因果模型的经典取法：它见过前面所有 token）",
    POOLING_MEAN: "对真实位置取平均（**必须用 mask 求和与计数**：把填充也算进去会稀释向量）",
    POOLING_MAX: "对真实位置逐维取最大（对离群值最敏感，因此它最容易被填充污染）",
}

#: 池化对账的容差：三个策略里有两个用 `math.fsum`，求和顺序不同 ⇒ 用容差而不是逐位。
POOLING_TOLERANCE = 1e-12

# --------------------------------------------------------------------------------------
# 4. 两条管线
# --------------------------------------------------------------------------------------

TASK_TEXT_GENERATION = "text-generation"
TASK_FEATURE_EXTRACTION = "feature-extraction"

TASK_KINDS: tuple[str, ...] = (TASK_TEXT_GENERATION, TASK_FEATURE_EXTRACTION)

TASK_DESCRIPTIONS: dict[str, str] = {
    TASK_TEXT_GENERATION: "文本生成：自回归地往后接 token（消费 day085 的 generate 与四种策略）",
    TASK_FEATURE_EXTRACTION: "特征抽取：一次前向拿到 hidden states，再池化成一个定长向量",
}

#: 两条管线各自需要的模型头：生成要 ``lm_head``（可与词嵌入共享），
#: 特征抽取只要主干。**"要什么头"决定了同一份主干能跑哪条管线**。
TASK_HEADS: dict[str, str] = {
    TASK_TEXT_GENERATION: "lm_head（词表大小的输出投影；tie_word_embeddings=True 时与 wte 共享）",
    TASK_FEATURE_EXTRACTION: "无（只用主干与最后的 ln_f）",
}

# --------------------------------------------------------------------------------------
# 5. 矩阵的两类形状（报告里要能读出"这一步是什么形状的"）
# --------------------------------------------------------------------------------------

#: 一次特征抽取的四个形状（**顺序 = 数据的流动顺序**）.
FEATURE_STAGES: tuple[str, ...] = (
    "input_ids",
    "embedding",
    "blocks",
    "final_norm",
    "pooled",
)

FEATURE_STAGE_SHAPES: dict[str, str] = {
    "input_ids": "(batch, seq) 整数 id",
    "embedding": "(batch, seq, hidden) = wte[id] + wpe[position]（BERT 侧再加 token_type 与一次 LN）",
    "blocks": "(batch, seq, hidden) —— 每一层**不改变形状**",
    "final_norm": "(batch, seq, hidden) —— GPT-2 的 ln_f / BERT 的最后一层 LN",
    "pooled": "(batch, hidden) —— 三法之一",
}

# --------------------------------------------------------------------------------------
# 6. 七条性质
# --------------------------------------------------------------------------------------

PROPERTY_CACHE_IS_PER_FILE = "cache_is_per_file"
PROPERTY_SNAPSHOT_IS_FLAT = "snapshot_is_flat"
PROPERTY_PARAMS_MATCH_LIBRARY = "params_match_library"
PROPERTY_ROUND_TRIP_IS_EXACT = "round_trip_is_exact"
PROPERTY_POOLING_RESPECTS_MASK = "pooling_respects_mask"
PROPERTY_BATCH_MATCHES_SINGLE = "batch_matches_single"
PROPERTY_GENERATION_MATCHES_DAY085 = "generation_matches_day085"
PROPERTY_WEIGHTS_GAP_IS_EXPLAINED = "weights_gap_is_explained"
PROPERTY_PROTOCOL_SURFACE_MATCHES = "protocol_surface_matches_local_model"
PROPERTY_BLOCK_MATCHES_DAY085 = "block_matches_day085"

INTEGRATION_PROPERTIES: tuple[str, ...] = (
    PROPERTY_CACHE_IS_PER_FILE,
    PROPERTY_SNAPSHOT_IS_FLAT,
    PROPERTY_PARAMS_MATCH_LIBRARY,
    PROPERTY_ROUND_TRIP_IS_EXACT,
    PROPERTY_POOLING_RESPECTS_MASK,
    PROPERTY_BATCH_MATCHES_SINGLE,
    PROPERTY_GENERATION_MATCHES_DAY085,
    PROPERTY_WEIGHTS_GAP_IS_EXPLAINED,
    PROPERTY_PROTOCOL_SURFACE_MATCHES,
    PROPERTY_BLOCK_MATCHES_DAY085,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_CACHE_IS_PER_FILE: "缓存命中是**逐文件**的：第二次解析同一个 commit 时下载字节数恰好为 0",
    PROPERTY_SNAPSHOT_IS_FLAT: "快照目录里的相对路径与仓库一致（**不带** blobs 前缀），且文件数与清单一致",
    PROPERTY_PARAMS_MATCH_LIBRARY: "按 config.json 算出的参数量与真实库**整数相等**（没有舍入可言）",
    PROPERTY_ROUND_TRIP_IS_EXACT: "encode 之后 decode 必须逐位还原原文（字节级 BPE 是无损的）",
    PROPERTY_POOLING_RESPECTS_MASK: "把填充长度从 3 改成 7，池化结果**逐位不变**（只对真实位置求和）",
    PROPERTY_BATCH_MATCHES_SINGLE: "批量结果与逐条结果逐位一致（因果侧靠掩码、双向侧靠先切开）",
    PROPERTY_GENERATION_MATCHES_DAY085: "本包的生成与 day085 的 ``generate`` 落在同一个 token 序列上（同一份策略、同一个种子）",
    PROPERTY_WEIGHTS_GAP_IS_EXPLAINED: "本包权重与配置口径的参数量之差**恰好**是注意力偏置 ``4·hidden·layers``",
    PROPERTY_PROTOCOL_SURFACE_MATCHES: "在进程模型与 ``LocalModel`` 的**方法面**逐名对齐（接进既有模块靠的是接口一致）",
    PROPERTY_BLOCK_MATCHES_DAY085: "单条样本、因果、同一份参数下，本包的一层与 day085 的 ``gpt2_block`` **逐位**相同",
}

PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_CACHE_IS_PER_FILE: "把'缓存命中'写成了'all-or-nothing'：一个文件缺失就整仓重下",
    PROPERTY_SNAPSHOT_IS_FLAT: "把 blobs 的哈希名当成了快照里的路径（于是 config.json 找不到）",
    PROPERTY_PARAMS_MATCH_LIBRARY: "漏算了偏置、或把 tying 当成'再算一遍词表'（两者都是整块的差）",
    PROPERTY_ROUND_TRIP_IS_EXACT: "分词时丢掉了字节级映射（中文与 emoji 会先坏）",
    PROPERTY_POOLING_RESPECTS_MASK: "均值的分母用了**总长度**而不是真实长度（填充越多、向量越短）",
    PROPERTY_BATCH_MATCHES_SINGLE: "双向模型上把填充一起喂进注意力（真实位置那一行被改写了）",
    PROPERTY_GENERATION_MATCHES_DAY085: "把 logits 的裁剪做在了策略之前（于是 top_k 与 top_p 的顺序换了）",
    PROPERTY_WEIGHTS_GAP_IS_EXPLAINED: "偏置被漏算或错算：那个差值就不再是 4h·L 这样一个可拆的数",
    PROPERTY_PROTOCOL_SURFACE_MATCHES: "少一个方法（例如没有 describe）时，上层的监控会安静地少一行读数",
    PROPERTY_BLOCK_MATCHES_DAY085: "接线顺序走散（LN 放错一侧、残差少加一次），而形状完全合法",
}

# --------------------------------------------------------------------------------------
# 7. 生态笔记（十二条——这是"把生态接进来"这一课的成品）
# --------------------------------------------------------------------------------------

INTEGRATION_NOTES: dict[str, str] = {
    "cache_layout": (
        "``from_pretrained`` 的第一步不是加载权重，而是**把文件找齐**："
        "``models--org--name/snapshots/<commit>/<相对路径>``，而 ``refs/main`` "
        "只是把 'main' 这个名字翻译成一串 commit hash。"
    ),
    "per_file_cache": (
        "缓存命中是**逐文件**的：第二次解析同一个 commit 时，每一个文件都已经在快照里，"
        "于是下载字节数恰好为 0。把它写成 'all-or-nothing' 会让'少了一个文件'变成'整仓重下'。"
    ),
    "flat_snapshot": (
        "快照目录里的相对路径与仓库里**一模一样**（平铺）；实体文件另存在 ``blobs/<sha256>``。"
        "这条'两处名字'正是最容易读错的地方：把哈希名当路径，``config.json`` 就找不到了。"
    ),
    "config_is_shape": (
        "``config.json`` 里的几个整数决定了每一层的宽度。两个架构给的是**同一个量的两个键名**："
        "``n_embd`` 与 ``hidden_size`` 是同一个数，``n_layer`` 与 ``num_hidden_layers`` 也是。"
    ),
    "params_are_computable": (
        "参数量不是'读出来的'，而是**算出来的**：给定 config 里那几个整数，"
        "本包的两条公式能给出与真实库**整数相等**的结果。"
        "这条对账的价值在于：它把'我理解了这个架构'变成了一条能失败的判据。"
    ),
    "tying": (
        "``tie_word_embeddings=True`` 让输出投影与词嵌入**共享同一份参数**："
        "于是参数量少了一份 ``vocab × hidden``，而模型的行为不变。"
        "本机的真实读数：tiny GPT-2（vocab=277, hidden=16, layers=2）在共享时是 **11 536** 个参数，"
        "不共享时多出 277×16 = **4 432** 个。"
        "而真实的 ``bert-base-uncased`` **也**共享——它的配置文件里根本没有这个键。"
    ),
    "byte_level_bpe": (
        "GPT-2 的 BPE 是**字节级**的：先把文本编码成字节，再做合并。"
        "因此它没有 'unknown token' 这件事——任何字节串都能被切出来。"
        "代价则取决于词表：本包的玩具词表只有 256 个基元，一个汉字**恰好 3 个 token**"
        "（UTF-8 三字节），而真实 GPT-2 的 50257 词表会把常见汉字合并成 1~2 个。"
        "**这个数是词表的性质，不是分词的铁律。**"
    ),
    "vocab_and_merges": (
        "``vocab.json`` 与 ``merges.txt`` 必须**同源**：两个不同来源的文件能拼出一个"
        "可用的分词器，而它切出来的 token 与训练时不一致——模型不报错，只是答得不对。"
    ),
    "pooling_needs_mask": (
        "均值池化的分母是**真实长度**，不是总长度。用总长度会让'填充越多、向量越短'，"
        "而这件事在相似度排序上表现为'短句总是更靠前'——看起来像一条数据结论。"
    ),
    "padding_when_causal": (
        "因果模型上填充是**惰性**的：被填出来的那些位置排在真实位置**之后**，"
        "而因果掩码让它们一格都传不回来。**但这句话只在单条样本内成立**——"
        "几条样本拼成一条长序列之后，那张 (n, n) 的下三角会让第 2 条看到第 1 条的全部 token，"
        "因此批量之后的掩码必须升级成**段内下三角**。双向模型上则连填充都不可忽略。"
    ),
    "batching_semantics": (
        "两条管线的批量语义不同：生成要逐条跑（长度不齐且要停在各自的 eos），"
        "特征抽取可以合批（一次前向，但要按上面的规则处理填充）。"
    ),
    "pipeline_is_orchestration": (
        "``pipeline`` 这一层**不含任何数学**：它只做'分词 → 前向 → 池化 → 后处理'的编排。"
        "因此换管线不改模型，换模型也不改管线——这正是'把它接进来'最省力的地方。"
    ),
}

INTEGRATION_NOTES_ORDER: tuple[str, ...] = tuple(INTEGRATION_NOTES)

# --------------------------------------------------------------------------------------
# 8. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

INTEGRATION_BOUNDARIES: tuple[str, ...] = (
    "本包复现的是**解析与装箱**语义，不是 ``huggingface_hub`` 的内部实现（并发下载、"
    "断点续传、符号链接与 xet 存储都不在其中）",
    "本包不下载任何东西：网络侧由一个**可注入的 Hub 客户端**承担，测试里它是内存里的假仓库；"
    "因此'离线也能跑'是设计目标，而不是一次运气",
    "本包的参数量公式覆盖 gpt2 与 bert 两个架构，**不**覆盖权重共享之外的高级配置"
    "（专家混合、跨层参数共享、量化权重都不在其中）",
    "本包的 byte-level BPE 用两个文件实现，**不**包含 ``tokenizer.json`` 里的正规化与预分词规则；"
    "因此它对中文的切法只是'可复现'，不承诺与某个线上 tokenizer 逐 token 相同",
    "本包的块只支持**因果 / 全双向**两种掩码，不支持 padding 掩码；"
    "双向侧处理填充的办法是**先按真实长度切开**，而不是给注意力塞一张 mask",
)

# --------------------------------------------------------------------------------------
# 9. 记录（一次"装载 + 调用"的账）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """快照里的一个文件：它叫什么、多大、这次是命中还是新下的."""

    path: str
    size: int
    cached: bool

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {"path": self.path, "size": self.size, "cached": self.cached}


@dataclass(frozen=True)
class Snapshot:
    """一次解析的结果：仓库、版本、commit、根目录与每一个文件的账.

    ``root`` 是一个**字符串**（不是 ``Path``）——本包刻意不碰文件系统，
    这样"缓存目录长什么样"这件事可以在内存里被完整断言。
    """

    repo_id: str
    revision: str
    commit: str
    root: str
    files: tuple[CacheEntry, ...]
    local_files_only: bool = True

    @property
    def file_count(self) -> int:
        """这次快照提供了几个文件."""
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        """这些文件的总字节数."""
        return sum(entry.size for entry in self.files)

    @property
    def cached_count(self) -> int:
        """命中缓存的文件数（第二次解析时它应当等于 ``file_count``）."""
        return sum(1 for entry in self.files if entry.cached)

    @property
    def downloaded_count(self) -> int:
        """这次新下的文件数（第二次解析时它应当**恰好为 0**）."""
        return self.file_count - self.cached_count

    @property
    def downloaded_bytes(self) -> int:
        """这次新下的字节数（逐文件记账的直接后果）."""
        return sum(entry.size for entry in self.files if not entry.cached)

    def names(self) -> tuple[str, ...]:
        """文件名的有序清单（顺序 = 清单顺序，测试与演示共用）."""
        return tuple(entry.path for entry in self.files)

    def entry(self, path: str) -> CacheEntry:
        """按路径取一个文件（缺失时由 :mod:`hub` 抛 ``HubError``）."""
        for item in self.files:
            if item.path == path:
                return item
        raise KeyError(path)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段（含三个派生读数）."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "commit": self.commit,
            "root": self.root,
            "local_files_only": self.local_files_only,
            "files": [item.to_dict() for item in self.files],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "cached_count": self.cached_count,
            "downloaded_count": self.downloaded_count,
            "downloaded_bytes": self.downloaded_bytes,
        }


@dataclass(frozen=True)
class ModelCard:
    """一份被解析过的 ``config.json``：**形状、口径与参数量都在这里**.

    字段名刻意与 :class:`~smart_research_agent.hf_source.types.ModelProfile` 对齐
    （``hidden`` / ``heads`` / ``layers`` / ``positions`` / ``activation`` / ``ln_eps``），
    因为它们是同一件事：前者来自**源码**，后者来自**配置文件**。
    两条来源都对上，才说明"我读的那份源码"与"我装的那个模型"是同一个东西。
    """

    name: str
    model_type: str
    hidden: int
    heads: int
    layers: int
    vocab: int
    positions: int
    activation: str
    ln_eps: float
    tie_word_embeddings: bool = False
    uses_token_type: bool = False
    type_vocab: int = 0
    architectures: tuple[str, ...] = ()
    intermediate: int | None = None

    @property
    def head_dim(self) -> int:
        """每一头的宽度 ``hidden / heads``（不整除时在 :mod:`config` 里被拒绝）."""
        return self.hidden // self.heads

    @property
    def ffn(self) -> int:
        """前馈的中间维：**配置里有就用配置里的，没有才按架构的规矩推**.

        ```text
        GPT-2   n_inner 缺省        ⇒ 4·hidden（这是它的规矩）
        BERT    intermediate_size   ⇒ 缺省 3072，与 hidden 无关
        ```

        于是"两个架构的前馈都是 4 倍"这句话只对 bert-base-uncased 那种尺寸成立——
        它是一条**被真库读出来的**事实，而不是一条可以推出来的规律。
        """
        if self.intermediate is not None:
            return self.intermediate
        if self.model_type == ARCHITECTURE_BERT:
            return BERT_DEFAULT_INTERMEDIATE
        return FFN_RATIO * self.hidden

    @property
    def causal(self) -> bool:
        """这个架构**默认**要不要因果掩码（GPT-2 要、BERT 不要）."""
        return self.model_type == ARCHITECTURE_GPT2

    @property
    def fused_qkv(self) -> bool:
        """注意力投影是不是融合的（GPT-2 一个 ``c_attn``、BERT 三个独立投影）."""
        return self.model_type == ARCHITECTURE_GPT2

    @property
    def parameter_count(self) -> int:
        """按配置算出的参数量（**与真实库整数相等**，见 :data:`PROPERTY_PARAMS_MATCH_LIBRARY`）.

        两条公式都把**偏置**算进去，因为两个架构的 ``Linear`` 默认都带偏置。
        GPT-2 与 BERT 的差别集中在三处：

        ```text
        LN 的个数     GPT-2 每层 2 个 + 栈尾 1 个；BERT 每层 2 个 + 嵌入后 1 个 + 池化前 0 个
        嵌入的三张表  BERT 多一张 token_type（type_vocab × hidden）与一次嵌入后的 LN
        池化          BERT 多一个 pooler（hidden² + hidden）
        ```
        """
        hidden = self.hidden
        shared = hidden * self.vocab  # 词嵌入（也是被共享的那一份输出投影）
        position = hidden * self.positions
        if self.model_type == ARCHITECTURE_GPT2:
            attention = (hidden + 1) * 3 * hidden  # c_attn：权重 + 偏置
            projection = (hidden + 1) * hidden  # c_proj
            feed_forward = (hidden + 1) * self.ffn + (self.ffn + 1) * hidden
            norms = 4 * hidden  # ln_1 与 ln_2 各一组 γ/β
            per_layer = attention + projection + feed_forward + norms
            final_norm = 2 * hidden  # ln_f
            head = 0 if self.tie_word_embeddings else shared
            return shared + position + self.layers * per_layer + final_norm + head
        if self.model_type == ARCHITECTURE_BERT:
            token_type = self.type_vocab * hidden
            embedding_norm = 2 * hidden
            attention = 4 * (hidden * hidden + hidden)  # 四次投影（q/k/v/dense），各带偏置
            projection = (hidden + 1) * self.ffn  # intermediate.dense
            output = (self.ffn + 1) * hidden  # output.dense
            norms = 4 * hidden  # attention.output.LN 与 output.LN
            per_layer = attention + projection + output + norms
            pooler = hidden * hidden + hidden
            head = 0 if self.tie_word_embeddings else shared
            return (
                shared
                + position
                + token_type
                + embedding_norm
                + self.layers * per_layer
                + pooler
                + head
            )
        from smart_research_agent.hf_integration.errors import ConfigError

        raise ConfigError(
            f"未知的架构 {self.model_type!r}：本包只给了两种参数量公式"
            f"（{list(ARCHITECTURES)}）。回落到某一条公式的后果是——"
            "一份 BERT 的读数会被印成 GPT-2 的（day085 第 2 章那条'回退到默认画像'的同族）。"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（含五个派生量与参数量）."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "hidden": self.hidden,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "layers": self.layers,
            "ffn": self.ffn,
            "ffn_ratio": self.ffn / self.hidden,
            "vocab": self.vocab,
            "positions": self.positions,
            "activation": self.activation,
            "ln_eps": self.ln_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "uses_token_type": self.uses_token_type,
            "type_vocab": self.type_vocab,
            "causal": self.causal,
            "fused_qkv": self.fused_qkv,
            "parameters": self.parameter_count,
            "architectures": list(self.architectures),
        }


@dataclass(frozen=True)
class Encoding:
    """一条文本被编码之后的样子：id、掩码、以及**切出来的子词**."""

    tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    truncated: bool = False

    def __post_init__(self) -> None:
        if not (len(self.tokens) == len(self.input_ids) == len(self.attention_mask)):
            from smart_research_agent.hf_integration.errors import ShapeError

            raise ShapeError(
                f"tokens / input_ids / attention_mask 的长度必须一致，收到 "
                f"{len(self.tokens)} / {len(self.input_ids)} / {len(self.attention_mask)}："
                "长度不齐会被 zip 静默截断（少看几个 token），而形状看起来是合法的。"
            )

    @property
    def length(self) -> int:
        """这条例子的长度（含填充）."""
        return len(self.input_ids)

    @property
    def real_length(self) -> int:
        """**真实长度**（掩码里 1 的个数）——池化的分母是它，不是 :attr:`length`."""
        return sum(self.attention_mask)

    @property
    def padding(self) -> int:
        """填充了几个位置（``length - real_length``）."""
        return self.length - self.real_length

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（``tokens`` 也印出来：它是"切得对不对"的唯一证据）."""
        return {
            "tokens": list(self.tokens),
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "length": self.length,
            "real_length": self.real_length,
            "padding": self.padding,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class TextBatch:
    """一批编码结果：形状被**对齐**过，因此它带着对齐方式与是否截断."""

    rows: tuple[Encoding, ...]
    padding: str
    pad_token_id: int
    max_length: int

    @property
    def batch_size(self) -> int:
        """这一批有几条."""
        return len(self.rows)

    @property
    def width(self) -> int:
        """对齐之后的宽度（= 最长的那条；``max_length`` 是它的上界）."""
        return max((row.length for row in self.rows), default=0)

    @property
    def real_width(self) -> int:
        """对齐之前的真实宽度（= 最长的那条的**真实**长度）."""
        return max((row.real_length for row in self.rows), default=0)

    @property
    def padding_ratio(self) -> float:
        """填充占比（0 = 一点没浪费）——它是"该不该分批"的唯一依据."""
        total = self.batch_size * self.width
        if total == 0:
            return 0.0
        return 1.0 - sum(row.real_length for row in self.rows) / total

    def mask_matrix(self) -> tuple[tuple[int, ...], ...]:
        """掩码矩阵（对齐后的 ``(batch, seq)`` 表）——池化与批量前向都吃它."""
        return tuple(row.attention_mask for row in self.rows)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "batch_size": self.batch_size,
            "width": self.width,
            "real_width": self.real_width,
            "padding": self.padding,
            "max_length": self.max_length,
            "padding_ratio": self.padding_ratio,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class PooledBatch:
    """一次池化的结果：``(batch, hidden)`` 的向量组，外加它用的是哪一法."""

    vectors: tuple[Vector, ...]
    strategy: str
    normalized: bool = False

    @property
    def batch_size(self) -> int:
        """几条向量."""
        return len(self.vectors)

    @property
    def dim(self) -> int:
        """向量宽度（= 模型的 hidden）."""
        return len(self.vectors[0]) if self.vectors else 0

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**向量本体，只印它的范数）."""
        return {
            "batch_size": self.batch_size,
            "dim": self.dim,
            "strategy": self.strategy,
            "normalized": self.normalized,
            "norms": [_norm(vector) for vector in self.vectors],
        }


@dataclass(frozen=True)
class IntegrationRecord:
    """一次"装载 + 调用"的账：从仓库名一直到池化向量."""

    repo_id: str
    revision: str
    commit: str
    model_type: str
    parameters: int
    files: int
    bytes_total: int
    tokens: int
    pooled_dim: int
    strategy: str
    lines: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "commit": self.commit,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "files": self.files,
            "bytes_total": self.bytes_total,
            "tokens": self.tokens,
            "pooled_dim": self.pooled_dim,
            "strategy": self.strategy,
            "lines": list(self.lines),
        }


def _norm(vector: Vector) -> float:
    """向量的欧氏范数（报告里用一个数概括"这条向量有多长"）."""
    return math.sqrt(math.fsum(value * value for value in vector))


__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_BERT",
    "ARCHITECTURE_DESCRIPTIONS",
    "ARCHITECTURE_GPT2",
    "BERT_BASE_PARAMETERS",
    "BERT_DEFAULT_INTERMEDIATE",
    "BLOB_DIR",
    "CACHE_LAYOUT",
    "CACHE_PREFIX",
    "CONFIG_FILE",
    "DEFAULT_REVISION",
    "FEATURE_STAGES",
    "FEATURE_STAGE_SHAPES",
    "FFN_RATIO",
    "GPT2_LEGACY_VOCAB",
    "GPT2_SMALL_PARAMETERS",
    "HUB_LIBRARY",
    "HUB_VERSION",
    "INTEGRATION_BOUNDARIES",
    "INTEGRATION_NOTES",
    "INTEGRATION_NOTES_ORDER",
    "INTEGRATION_PROPERTIES",
    "KNOWN_FILES",
    "MERGES_FILE",
    "POOLING_DESCRIPTIONS",
    "POOLING_LAST_TOKEN",
    "POOLING_MAX",
    "POOLING_MEAN",
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
    "REF_DIR",
    "REQUIRED_CONFIG_KEYS",
    "SIZE_KEYS",
    "SNAPSHOT_DIR",
    "TASK_DESCRIPTIONS",
    "TASK_FEATURE_EXTRACTION",
    "TASK_HEADS",
    "TASK_KINDS",
    "TASK_TEXT_GENERATION",
    "TRANSFORMERS_LIBRARY",
    "TRANSFORMERS_VERSION",
    "TRANSFORMERS_VERSION_SAMPLE",
    "VOCAB_FILE",
    "WEIGHTS_FILE",
    "CacheEntry",
    "Encoding",
    "IntegrationRecord",
    "ModelCard",
    "PooledBatch",
    "Snapshot",
    "TextBatch",
]
