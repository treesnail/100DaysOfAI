"""``hf_source`` 的形状、口径表与源码阅读笔记（day085 / M7-D9）.

本模块是这一天的"判据总表"。它把**读 Hugging Face 源码**这件事拆成三样可以被断言的东西：

```text
① 两个模型画像（ModelProfile）
   同一个 ``LayerNorm``、同一个"块"，在 BERT 与 GPT-2 里**默认参数不同**：
     BERT     layer_norm_eps = 1e-12   hidden_act = "gelu"       LN **在子层之后**（post）
     GPT-2    layer_norm_epsilon = 1e-5 activation_function = "gelu_new"  LN **在子层之前**（pre）
   这四条都是配置文件里的默认值，因此它们是**事实**，不是"一般认为"。
② 一条推理链路的十个阶段（ATTENTION_STAGES）
   从融合投影到最后一次输出投影。"分头"发生在**投影内部**（一次 view + transpose），
   而不是靠换一个块类型——这正是 day079/day083 记下的那条接口边界在 HF 里被打破的地方。
③ 五种生成策略（GENERATION_STRATEGIES）
   greedy / sample / top_k / top_p / beam。"预算是过滤器"这件事在 HF 里
   由四个 ``*LogitsWarper`` / ``*LogitsProcessor`` 实现，顺序是
   **重复惩罚 → 温度 → top_k → top_p → 采样**，而顺序会改变结果。

## 一条纪律：本包只承诺"行为"，不承诺"行号"

```text
承诺      给定同样的权重与输入，本包复现 HF 那几个类的**计算语义**
不承诺    某个符号在第几行、某个内部变量的名字、某个版本的中间张量形状
```

理由与 day080 的"参数量差值必须被逐项解释"同源：
HF 的源码会随版本重构（本课程写作时最新为 v5.17.0），
但"``ln_1`` 在自注意力**之前**"这类**结构**事实是跨版本稳定的。
因此本包的每一条断言都写成**行为**，并且在测试里都有一条**反证**。
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# 0. 阅读对象（版本号写进常量，避免"当时读的是哪一版"变成一个传说）
# --------------------------------------------------------------------------------------

#: 阅读对象所在的仓库（跨版本稳定）.
SOURCE_LIBRARY = "huggingface/transformers"

#: 阅读对象的大版本（本课程写作时是 v5.x）.
SOURCE_VERSION = "5.x"

#: 写作时最新的一版（用于文档与习题里的"当时读数"标注）.
SOURCE_VERSION_SAMPLE = "v5.17.0（2026-09-09 发布）"

#: 两个模型各自的实现文件（**这两行是"读什么"的答案**）.
SOURCE_FILES: dict[str, str] = {
    "gpt2": "src/transformers/models/gpt2/modeling_gpt2.py",
    "bert": "src/transformers/models/bert/modeling_bert.py",
}

# --------------------------------------------------------------------------------------
# 1. 两个模型画像
# --------------------------------------------------------------------------------------

#: LayerNorm 的默认 eps：**同一个算子、三个默认值**（T5 的一并记下，作为对照）。
LN_EPS_GPT2 = 1e-5
LN_EPS_BERT = 1e-12
LN_EPS_T5 = 1e-6

LN_EPS_DEFAULTS: dict[str, float] = {
    "gpt2": LN_EPS_GPT2,
    "bert": LN_EPS_BERT,
    "t5": LN_EPS_T5,
}

#: 两个模型的激活函数默认值（**不同**，这常被当成"都一样"）。
ACTIVATION_BERT = "gelu"
ACTIVATION_GPT2 = "gelu_new"

#: 本包支持的三种逐元素激活（前两种是 HF 的两个默认值，第三种用来与 day079 对账）。
ACTIVATION_RELU = "relu"
ACTIVATIONS: tuple[str, ...] = (ACTIVATION_BERT, ACTIVATION_GPT2, ACTIVATION_RELU)

ACTIVATION_DESCRIPTIONS: dict[str, str] = {
    ACTIVATION_BERT: "gelu：0.5x(1+erf(x/√2))——**精确式**（erf），BERT 的默认值",
    ACTIVATION_GPT2: "gelu_new：0.5x(1+tanh(√(2/π)(x+0.044715x³)))——**tanh 近似**，GPT-2 的默认值",
    ACTIVATION_RELU: "relu：max(0, x)——day079 的默认值，用来与既有实现逐点对账",
}

#: 两个摆放（与 day079 的 ``NORM_PRE`` / ``NORM_POST`` 同名同义）。
NORM_PRE = "pre"
NORM_POST = "post"

NORM_PLACEMENTS: tuple[str, ...] = (NORM_PRE, NORM_POST)

NORM_PLACEMENT_DESCRIPTIONS: dict[str, str] = {
    NORM_PRE: "pre-LN：y = x + F(LN(x))——LayerNorm 在**子层之前**（GPT-2 与所有现代实现）",
    NORM_POST: "post-LN：y = LN(x + F(x))——LayerNorm 在**子层之后**（BERT 与原论文）",
}


@dataclass(frozen=True)
class ModelProfile:
    """一个模型的**可被断言**的源码事实集合.

    每一条都对应配置文件里的一个默认值或实现里的一处结构，因此每一条都能被复核。
    """

    name: str
    norm_placement: str
    ln_eps: float
    activation: str
    fused_qkv: bool
    normalize_embeddings: bool
    uses_token_type: bool
    source_file: str

    def to_dict(self) -> dict[str, object]:
        """把画像摊平成一行可读的字段（端点与演示脚本共用）."""
        return {
            "name": self.name,
            "norm_placement": self.norm_placement,
            "ln_eps": self.ln_eps,
            "activation": self.activation,
            "fused_qkv": self.fused_qkv,
            "normalize_embeddings": self.normalize_embeddings,
            "uses_token_type": self.uses_token_type,
            "source_file": self.source_file,
        }


#: 两个画像。``fused_qkv`` 与 ``normalize_embeddings`` 是最容易被忽略的两条差别。
PROFILES: dict[str, ModelProfile] = {
    "gpt2": ModelProfile(
        name="gpt2",
        norm_placement=NORM_PRE,
        ln_eps=LN_EPS_GPT2,
        activation=ACTIVATION_GPT2,
        fused_qkv=True,
        normalize_embeddings=False,
        uses_token_type=False,
        source_file=SOURCE_FILES["gpt2"],
    ),
    "bert": ModelProfile(
        name="bert",
        norm_placement=NORM_POST,
        ln_eps=LN_EPS_BERT,
        activation=ACTIVATION_BERT,
        fused_qkv=False,
        normalize_embeddings=True,
        uses_token_type=True,
        source_file=SOURCE_FILES["bert"],
    ),
}


def profile_of(name: str) -> ModelProfile:
    """取一个模型画像（未知名字当场拒绝，绝不回退到某个默认画像）."""
    from smart_research_agent.hf_source.errors import ParameterError

    if name not in PROFILES:
        raise ParameterError(
            f"未知的模型名 {name!r}：本包只认 {sorted(PROFILES)}。"
            "回退到某个默认画像的后果是——一次 BERT 的读数会被印成 GPT-2 的。"
        )
    return PROFILES[name]


# --------------------------------------------------------------------------------------
# 2. 形状
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceShape:
    """一次前向的形状：``(tokens, hidden)``，加上头数与词表大小.

    ``causal_default`` 是"这个模型**默认**要不要因果掩码"——它是画像带来的，
    而不是形状本身的性质（BERT 双向、GPT-2 因果）。把这一条放进形状里，
    是为了让 :func:`verify.check_all` 能按"这一次该不该因果"取默认值，
    而不是把"因果"当成一个必须每次手传的开关（手传会漏，而漏了不报错）。
    """

    tokens: int
    hidden: int
    heads: int
    vocab: int
    causal_default: bool = False

    @property
    def head_dim(self) -> int:
        """每一头的宽度 ``hidden / heads``（不能整除时在 :func:`attention.resolve_heads` 里拒绝）."""
        return self.hidden // self.heads

    @property
    def fused_width(self) -> int:
        """融合投影 ``c_attn`` 的输出宽度——**恰好是 3·hidden**，这是切法的依据."""
        return 3 * self.hidden

    def to_dict(self) -> dict[str, object]:
        """把形状摊平成一行字段（含两个派生量与因果默认值）."""
        return {
            "tokens": self.tokens,
            "hidden": self.hidden,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "vocab": self.vocab,
            "fused_width": self.fused_width,
            "causal_default": self.causal_default,
        }


# --------------------------------------------------------------------------------------
# 3. 十个阶段（一条推理链路的"地图"）
# --------------------------------------------------------------------------------------

STAGE_QKV = "qkv"
STAGE_SPLIT = "split_heads"
STAGE_SCORE = "score"
STAGE_SCALE = "scale"
STAGE_BIAS = "bias"
STAGE_SOFTMAX = "softmax"
STAGE_DROP = "attention_dropout"
STAGE_MIX = "mix"
STAGE_MERGE = "merge_heads"
STAGE_OUT = "out_proj"

ATTENTION_STAGES: tuple[str, ...] = (
    STAGE_QKV,
    STAGE_SPLIT,
    STAGE_SCORE,
    STAGE_SCALE,
    STAGE_BIAS,
    STAGE_SOFTMAX,
    STAGE_DROP,
    STAGE_MIX,
    STAGE_MERGE,
    STAGE_OUT,
)

ATTENTION_STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_QKV: "融合投影：GPT-2 用一个 c_attn 一次算出 3·hidden 维（BERT 用三个独立 Linear）",
    STAGE_SPLIT: "**分头发生在投影内部**：view(n, heads, head_dim).transpose(0, 1)",
    STAGE_SCORE: "打分 Q·Kᵀ（逐头）",
    STAGE_SCALE: "乘 1/√head_dim（**不是** 1/√hidden）",
    STAGE_BIAS: "加**加性掩码**（-inf 或一个大负数），而不是把被挡位置的权重置 0",
    STAGE_SOFTMAX: "在最后一维做 softmax（被加性掩码挡掉的格子**恰好 0.0**）",
    STAGE_DROP: "注意力 dropout：**只在训练相**；推理相恒等",
    STAGE_MIX: "加权求和 weights·V（逐头）",
    STAGE_MERGE: "transpose + view 拼回 (n, hidden)",
    STAGE_OUT: "输出投影（GPT-2 的 c_proj / BERT 的 dense）",
}

ATTENTION_STAGE_SHAPES: dict[str, str] = {
    STAGE_QKV: "(n, hidden) → (n, 3·hidden) 或三个 (n, hidden)",
    STAGE_SPLIT: "(n, hidden) → heads × (n, head_dim)",
    STAGE_SCORE: "heads × (n, n)",
    STAGE_SCALE: "heads × (n, n)",
    STAGE_BIAS: "heads × (n, n) ⊕ (n, n)",
    STAGE_SOFTMAX: "heads × (n, n)",
    STAGE_DROP: "heads × (n, n)",
    STAGE_MIX: "heads × (n, head_dim)",
    STAGE_MERGE: "(n, hidden)",
    STAGE_OUT: "(n, hidden) → (n, hidden)",
}

# --------------------------------------------------------------------------------------
# 4. 五种生成策略
# --------------------------------------------------------------------------------------

STRATEGY_GREEDY = "greedy"
STRATEGY_SAMPLE = "sample"
STRATEGY_TOP_K = "top_k"
STRATEGY_TOP_P = "top_p"
STRATEGY_BEAM = "beam"

GENERATION_STRATEGIES: tuple[str, ...] = (
    STRATEGY_GREEDY,
    STRATEGY_SAMPLE,
    STRATEGY_TOP_K,
    STRATEGY_TOP_P,
    STRATEGY_BEAM,
)

GENERATION_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    STRATEGY_GREEDY: "贪心：每步取 argmax——**完全确定**，并列时取最小下标（与 torch.argmax 一致）",
    STRATEGY_SAMPLE: "纯采样：温度缩放后按整条分布采样（需要一串可注入的均匀数）",
    STRATEGY_TOP_K: "top-k：只保留概率最大的 k 个，其余置 -inf，再采样",
    STRATEGY_TOP_P: "top-p（核采样）：按**累积概率**留最小的一个集合，至少留 1 个",
    STRATEGY_BEAM: "beam search：同时维护 num_beams 条前缀，按长度惩罚后的分数选最终序列",
}

#: HF 里四个 warper / processor 的**应用顺序**（顺序会改变结果，因此它是一个常量）.
WARPER_ORDER: tuple[str, ...] = ("repetition_penalty", "temperature", "top_k", "top_p")

WARPER_ORDER_NOTE = (
    "顺序即语义：repeat → temperature → top_k → top_p。"
    "把 top_k 放到 temperature 之前，会在**未缩放**的 logits 上排名次——"
    "而温度并不改变排名，所以这一处顺序其实安全；"
    "真正的陷阱是 top_k 与 top_p 之间：top_p 的核是在**已经截断过的**分布上算的。"
)

# --------------------------------------------------------------------------------------
# 5. 六条性质
# --------------------------------------------------------------------------------------

PROPERTY_ROWS_ARE_DISTRIBUTIONS = "rows_are_distributions"
PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO = "masked_entries_are_exact_zero"
PROPERTY_CAUSAL_PREFIX_IS_STABLE = "causal_prefix_is_stable"
PROPERTY_SPLIT_MERGE_ROUND_TRIP = "split_merge_round_trip"
PROPERTY_FUSED_MATCHES_SEPARATE = "fused_matches_separate"
PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD = "single_head_matches_multi_head"
PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079 = "pre_norm_block_matches_day079"

SOURCE_PROPERTIES: tuple[str, ...] = (
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_CAUSAL_PREFIX_IS_STABLE,
    PROPERTY_SPLIT_MERGE_ROUND_TRIP,
    PROPERTY_FUSED_MATCHES_SEPARATE,
    PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD,
    PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_ROWS_ARE_DISTRIBUTIONS: "每一行非负、和为 1、全部有限（注意力权重是一组分布）",
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO: "被加性掩码挡掉的格子**逐位为 0.0**（不是'很小'）",
    PROPERTY_CAUSAL_PREFIX_IS_STABLE: "扰动第 j 个输入后，第 i < j 行的输出**逐位不变**",
    PROPERTY_SPLIT_MERGE_ROUND_TRIP: "split_heads 之后 merge_heads 必须逐位还原（分头只是记账）",
    PROPERTY_FUSED_MATCHES_SEPARATE: "融合投影按 hidden 切回去后，与三次独立投影**逐位相同**",
    PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD: "heads=1 时本包与 day076 的 multi_head_attention 一致",
    PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079: "pre 摆放的本包块与 day079 的 encoder_block 一致",
}

PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_ROWS_ARE_DISTRIBUTIONS: "softmax 的分母算错，或被挡位置没有真的被排除",
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO: "掩码加法与 softmax 之间漏了一次，或掩码选错了变体",
    PROPERTY_CAUSAL_PREFIX_IS_STABLE: "加性掩码的行列搞反了（上三角与被挡集合不是同一个集合）",
    PROPERTY_SPLIT_MERGE_ROUND_TRIP: "分头时的轴与拼回时的轴不是同一对（形状合法、语义已变）",
    PROPERTY_FUSED_MATCHES_SEPARATE: "拼接顺序与切开顺序不一致（q/k/v 三个角色互换而不报错）",
    PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD: "缩放口径或掩码口径走散（差 √heads 倍那一类）",
    PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079: "第一个子层的输入取错了（day079 第 5.3 节那处坑的同族）",
}

# --------------------------------------------------------------------------------------
# 6. 源码阅读笔记（十二条——这是"记录源码阅读笔记"这句学习目标的成品）
# --------------------------------------------------------------------------------------

SOURCE_NOTES: dict[str, str] = {
    "proj_inside": (
        "GPT-2 的注意力把分头放进**投影内部**：c_attn 的输出被 view 成 (n, heads, head_dim) "
        "再 transpose——因此同一个类**同时**支持 heads>1 与因果掩码，"
        "而不需要像 day079 那样'换一个块类型'。"
    ),
    "fused_three": (
        "c_attn 一次算出 3·hidden 维，然后按 hidden 切成 q/k/v 三块（split(hidden, dim=-1)）。"
        "切法错了**不会报错**：三个 (n, hidden) 的块永远凑得出来，只是各自的列混了。"
    ),
    "scale_head_dim": (
        "缩放用 1/√head_dim（不是 1/√hidden）。这与 day076 的 head_scale 是同一个量——"
        "它是**派生量**：head_dim = hidden / heads。"
    ),
    "additive_bias": (
        "掩码是**加性**的：先生成下三角的 0/1 矩阵，取反后乘一个很大的负数（或直接填 -inf），"
        "再与打分相加。因此'被挡住'在权重上表现为恰好 0（exp(-inf) = 0），"
        "而'很小'表现为一个正数——两者在热力图上是同一个空白格。"
    ),
    "ln_eps_two": (
        "同一个 LayerNorm 有两个常用默认值：GPT-2 的 1e-5 与 BERT 的 1e-12。"
        "day079 取的 1e-5 与 GPT-2 同源——而 BERT 的 1e-12 会让'方差离 1'这件事"
        "小得多（eps 越小，x̂ 的方差越接近 1）。"
    ),
    "norm_placement": (
        "GPT-2 是 pre-LN（ln_1 / ln_2 在子层**之前**）并在栈尾加一次 ln_f；"
        "BERT 是 post-LN（每个子层之后都跟一次 LayerNorm）。"
        "因此'norm 放在哪两个 Module 之间'就是 day079 那张 pre/post 对照表在真实源码里的样子。"
    ),
    "embedding_ln": (
        "BERT 的嵌入层里有一个 LayerNorm（word + token_type + position 之后再归一化），"
        "GPT-2 的嵌入层里**没有**——它只在栈尾做一次 ln_f。"
        "这一条差别会改变第一层的输入尺度，因此它不是'实现的自由'。"
    ),
    "activation_two": (
        "BERT 默认 hidden_act = 'gelu'（erf 精确式），GPT-2 默认 activation_function = 'gelu_new'"
        "（tanh 近似）。两者差多少可以被量出来——本包的量法是逐点最大绝对差。"
    ),
    "generation_warpers": (
        "生成策略在 HF 里是四个 warper/processor 的**流水线**："
        "RepetitionPenalty → Temperature → TopK → TopP，每一步只改 logits，不改模型。"
        "因此'换策略'与'换模型'是两件完全独立的事。"
    ),
    "top_p_semantics": (
        "top-p 的实现是**升序**排序 + 累积 softmax 概率 + `cumulative_probs <= (1 - top_p)` "
        "的删除条件 + '至少保留一个'。它保住的是'累积概率刚好超过 p 的那个最小集合'，"
        "因此 top_p 的**保留个数是数据决定的**，不是参数决定的。"
    ),
    "repetition_penalty": (
        "重复惩罚是**除**而不是减：正 logits 除以 penalty、负 logits 乘以 penalty。"
        "写成'减一个常数'在 logits 全为正时看起来一样，一遇到负 logits 就变成放大。"
    ),
    "beam_length_penalty": (
        "beam search 给每条候选打 `logprob / length**length_penalty` 的分。"
        "注意那是个**非正**的数：lp = 0 时'越长的候选总分越负'（系统性偏短），"
        "lp > 0 把这个'越负'摊薄（偏好长序列，越大越强），lp < 0 反过来放大它"
        "（偏好短序列）。**参数名是'长度惩罚'而 lp > 0 其实在偏好长序列**——"
        "这正是它容易被读错的地方。"
    ),
}

SOURCE_NOTES_ORDER: tuple[str, ...] = tuple(SOURCE_NOTES)

# --------------------------------------------------------------------------------------
# 7. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

SOURCE_BOUNDARIES: tuple[str, ...] = (
    "本包复现的是**计算语义**，不是 HF 的内部实现细节（变量名、缓存类、张量布局的中间态）",
    "本包只读**推理**路径：反向由 autograd 从计算图推出，因此本课没有'手写反向'可以校验",
    "本包的采样用一串可注入的均匀数（确定性），因此'同样的种子得到同样的文本'是设计目标，"
    "而不是对 torch.multinomial 的逐位复现",
    "beam search 复现的是'长度惩罚 + 早停'的**打分与选择**语义，"
    "不承诺与 HF 在 beam 间的并列打破顺序上逐位一致（那是实现细节）",
    "两个模型的画像来自**配置文件默认值**；挂上真实权重后，'哪个更适合你的任务'"
    "仍然要回到任务指标（day088 的串联会用到这一条）",
)

# --------------------------------------------------------------------------------------
# 8. 生成记录
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationSettings:
    """一次生成的配置（字段名与 ``transformers.GenerationConfig`` 对齐）."""

    max_new_tokens: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    eos_token: int | None = None
    seed: int = 7

    def to_dict(self) -> dict[str, object]:
        """把配置摊平成一行字段（演示脚本与报告共用）."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "eos_token": self.eos_token,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GenerationStep:
    """生成的一步：选了什么、候选剩几个、这一行分布有多尖."""

    step: int
    chosen: int
    kept: int
    top_probability: float
    entropy: float
    strategy: str

    def to_dict(self) -> dict[str, object]:
        """把一步摊平成一行字段."""
        return {
            "step": self.step,
            "chosen": self.chosen,
            "kept": self.kept,
            "top_probability": self.top_probability,
            "entropy": self.entropy,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class GenerationResult:
    """一次生成的账：token 序列 + 逐步读数 + 收尾原因."""

    token_ids: tuple[int, ...]
    prompt_length: int
    steps: tuple[GenerationStep, ...]
    strategy: str
    stopped_early: bool = False
    notes: tuple[str, ...] = ()

    @property
    def generated(self) -> tuple[int, ...]:
        """"新生成的那一段"（不含 prompt）——报告里最常被引用的字段."""
        return self.token_ids[self.prompt_length :]

    def to_dict(self) -> dict[str, object]:
        """把一次生成摊平成可 JSON 化的字段（**不含** prompt 的逐 token 内容）."""
        return {
            "token_ids": list(self.token_ids),
            "generated": list(self.generated),
            "prompt_length": self.prompt_length,
            "strategy": self.strategy,
            "stopped_early": self.stopped_early,
            "steps": [step.to_dict() for step in self.steps],
            "notes": list(self.notes),
        }


__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_BERT",
    "ACTIVATION_DESCRIPTIONS",
    "ACTIVATION_GPT2",
    "ACTIVATION_RELU",
    "ATTENTION_STAGES",
    "ATTENTION_STAGE_DESCRIPTIONS",
    "ATTENTION_STAGE_SHAPES",
    "GENERATION_STRATEGIES",
    "GENERATION_STRATEGY_DESCRIPTIONS",
    "LN_EPS_BERT",
    "LN_EPS_DEFAULTS",
    "LN_EPS_GPT2",
    "LN_EPS_T5",
    "NORM_PLACEMENTS",
    "NORM_PLACEMENT_DESCRIPTIONS",
    "NORM_POST",
    "NORM_PRE",
    "PROFILES",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_CAUSAL_PREFIX_IS_STABLE",
    "PROPERTY_FUSED_MATCHES_SEPARATE",
    "PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO",
    "PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079",
    "PROPERTY_ROWS_ARE_DISTRIBUTIONS",
    "PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD",
    "PROPERTY_SPLIT_MERGE_ROUND_TRIP",
    "SOURCE_BOUNDARIES",
    "SOURCE_FILES",
    "SOURCE_LIBRARY",
    "SOURCE_NOTES",
    "SOURCE_NOTES_ORDER",
    "SOURCE_PROPERTIES",
    "SOURCE_VERSION",
    "SOURCE_VERSION_SAMPLE",
    "STAGE_BIAS",
    "STAGE_DROP",
    "STAGE_MERGE",
    "STAGE_MIX",
    "STAGE_OUT",
    "STAGE_QKV",
    "STAGE_SCALE",
    "STAGE_SCORE",
    "STAGE_SOFTMAX",
    "STAGE_SPLIT",
    "STRATEGY_BEAM",
    "STRATEGY_GREEDY",
    "STRATEGY_SAMPLE",
    "STRATEGY_TOP_K",
    "STRATEGY_TOP_P",
    "WARPER_ORDER",
    "WARPER_ORDER_NOTE",
    "GenerationResult",
    "GenerationSettings",
    "GenerationStep",
    "ModelProfile",
    "SourceShape",
    "profile_of",
]
