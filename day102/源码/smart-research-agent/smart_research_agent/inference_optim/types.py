"""``inference_optim`` 的形状、口径表与笔记（day087 / M7-D11）.

day086 把模型**接进来**了；今天的全部动作都发生在那之后、而且只发生在一侧：

```text
推理路径上可以动三样东西，而它们互不相干：
① 缓存        别再重算已经算过的 K/V —— 换来的是**显存**（时间换空间）
② 精度        把权重与缓存从 32 位压到 8 位 / 4 位 —— 换来的是**误差**（空间换精度）
③ 批          把若干条样本绑在一起跑 —— 换来的是**吞吐**（吞吐换延迟）
```

三样东西各有一个"账"，而本模块把三本账都写成**可断言**的形式：

```text
缓存账   每一步新增 2·L·h·bytes 字节 ⇒ 长度 T 的缓存恰好是 2·L·T·h·bytes
精度账   反量化误差的上界是 scale/2（它由"每个元素都落在 [-max_abs, max_abs] 内"推出）
批账     一次前向处理 B 条、总 padding 与总有效 token 都是整数 ⇒ 填充占比可以被算出来
```

## 一条纪律：**三个账必须能被逐项相加**

今天的报告里每个数都要能被"加起来对得上"：
权重 + 缓存 + 激活 = 总量；每一步的增量 × 步数 = 总缓存；B 条的长度之和 = 有效 token。
这与 day080 的"参数量差值必须被逐项解释"同源——
一个不能被逐项拆开的数，就无法被反驳。

## 另一条纪律：**方向必须写清楚**

三样东西都是"换来"而不是"变好"：

```text
缓存    省时间，花显存          ⇒ 收益是"这一步只算一个 token"
精度    省显存，花精度          ⇒ 收益是"同样显存能放更大模型"，代价是误差
批      省显存/时间（单位吞吐）  ⇒ 收益是吞吐，代价是**首字延迟**（要等价最长的那个）
```

因此本课每一条读数都带方向（`higher` / `lower` / `neutral`），
而"越大越好"这种话一句都不写。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.math_foundations.types import Matrix, Vector

# --------------------------------------------------------------------------------------
# 0. 对接对象（版本号写进常量）
# --------------------------------------------------------------------------------------

#: 对接的量化库（真实世界里这两个名字最常见；本课只用它们的事实，不调用它们）。
QUANT_LIBRARY = "bitsandbytes"
ACCELERATE_LIBRARY = "huggingface/accelerate"

#: 本机实测版本（用于文档与习题里的"当时读数"标注）。
QUANT_LIBRARY_VERSION = "0.4x（本机未安装，本课只用它的事实：NF4 / LLM.int8() 两类方案）"

#: 本课程写作时的 transformers 版本（与 day086 同一份记录）。
TRANSFORMERS_VERSION = "5.17.0"

# --------------------------------------------------------------------------------------
# 1. 四种精度（**一个级别 = 一个字节数**）
# --------------------------------------------------------------------------------------

LEVEL_FP32 = "fp32"
LEVEL_FP16 = "fp16"
LEVEL_INT8 = "int8"
LEVEL_INT4 = "int4"

LEVELS: tuple[str, ...] = (LEVEL_FP32, LEVEL_FP16, LEVEL_INT8, LEVEL_INT4)

#: 每个级别占多少**字节**（int4 是打包之后的等效值：两个数一个字节）。
BYTES_PER_ELEMENT: dict[str, float] = {
    LEVEL_FP32: 4.0,
    LEVEL_FP16: 2.0,
    LEVEL_INT8: 1.0,
    LEVEL_INT4: 0.5,
}

#: 每个级别的**位宽**（float32 是 32 位；int4 也是 4 位，只是要打包）。
BIT_WIDTHS: dict[str, int] = {
    LEVEL_FP32: 32,
    LEVEL_FP16: 16,
    LEVEL_INT8: 8,
    LEVEL_INT4: 4,
}

LEVEL_DESCRIPTIONS: dict[str, str] = {
    LEVEL_FP32: "fp32：4 字节/元素——训练与数值实验的基准，也是'不小'的那一档",
    LEVEL_FP16: "fp16：2 字节/元素——推理的常见默认；相对 fp32 的误差来自**有效位**而不是量程",
    LEVEL_INT8: "int8：1 字节/元素——整数量化，误差上界是 scale/2（本课的主角之一）",
    LEVEL_INT4: "int4：0.5 字节/元素——**两个数挤进一个字节**，因此它必须打包与解包",
}

#: 只有这两个级别是"整数量化"（本课的量化模块只处理它们）。
INTEGER_LEVELS: tuple[str, ...] = (LEVEL_INT8, LEVEL_INT4)

#: 每一个整数级别的位宽（``2**(bits-1) - 1`` 是它的最大量化值）。
INTEGER_BITS: dict[str, int] = {LEVEL_INT8: 8, LEVEL_INT4: 4}

# --------------------------------------------------------------------------------------
# 2. 两种量化方案 × 两种粒度
# --------------------------------------------------------------------------------------

SCHEME_ABSMAX = "absmax"
SCHEME_ZERO_POINT = "zero_point"

SCHEMES: tuple[str, ...] = (SCHEME_ABSMAX, SCHEME_ZERO_POINT)

SCHEME_DESCRIPTIONS: dict[str, str] = {
    SCHEME_ABSMAX: "absmax（对称）：q = round(x / scale)，scale = max|x| / (2^(b-1) - 1)；"
    "它把 0 精确地映射到 0（**没有零点偏移**）",
    SCHEME_ZERO_POINT: "zero_point（非对称）：q = round(x / scale) + zero，"
    "scale = (max - min) / (2^b - 1)；它把动态范围用满（**0 要靠 zero 找回来**）",
}

GRANULARITY_TENSOR = "per_tensor"
GRANULARITY_CHANNEL = "per_channel"

GRANULARITIES: tuple[str, ...] = (GRANULARITY_TENSOR, GRANULARITY_CHANNEL)

GRANULARITY_DESCRIPTIONS: dict[str, str] = {
    GRANULARITY_TENSOR: "per_tensor：整块共用一个 scale——省元数据，代价是**离群值会抬高所有人的台阶**",
    GRANULARITY_CHANNEL: "per_channel：每一行一个 scale——行内动态范围小得多，代价是每行都要存一个 scale",
}

# --------------------------------------------------------------------------------------
# 3. 缓存的两种布局与两个阶段
# --------------------------------------------------------------------------------------

LAYOUT_GROW = "grow"
LAYOUT_PREALLOCATED = "preallocated"

LAYOUTS: tuple[str, ...] = (LAYOUT_GROW, LAYOUT_PREALLOCATED)

LAYOUT_DESCRIPTIONS: dict[str, str] = {
    LAYOUT_GROW: "grow：每步追加一行——实现简单，代价是**每次追加都可能重新分配**（真实实现里是扩容拷贝）",
    LAYOUT_PREALLOCATED: "preallocated：开局按最大长度开好——没有扩容，代价是**长度必须提前知道**",
}

STAGE_PREFILL = "prefill"
STAGE_DECODE = "decode"

STAGES: tuple[str, ...] = (STAGE_PREFILL, STAGE_DECODE)

STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_PREFILL: "prefill：把提示词一次算完（**它是一次普通的整段前向**，只是顺手把 K/V 存下来）",
    STAGE_DECODE: "decode：每次一个 token（**每层只算一个新位置**，K/V 从缓存里读）",
}

#: 缓存里存的是**两**张表：K 与 V。这个 2 出现在每一条缓存公式里。
CACHE_TABLES = 2

#: 每步 decode 的查询行数（这就是"省时间"的全部来源）。
DECODE_TOKENS_PER_STEP = 1

# --------------------------------------------------------------------------------------
# 4. 两个与"位数"无关的常数（写下来，免得散落成字面量）
# --------------------------------------------------------------------------------------

#: 一个字节的位数（int4 的打包按它算）。
BITS_PER_BYTE = 8

#: ``int4`` 一个字节装几个数（``BITS_PER_BYTE / BIT_WIDTHS[int4]``）。
INT4_PER_BYTE = 2

# --------------------------------------------------------------------------------------
# 5. 七条性质
# --------------------------------------------------------------------------------------

PROPERTY_CACHED_EQUALS_RECOMPUTE = "cached_equals_recompute"
PROPERTY_PREFILL_MATCHES_FULL = "prefill_matches_full"
PROPERTY_CACHE_GROWTH_IS_A_FORMULA = "cache_growth_is_a_formula"
PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE = "dequant_error_within_half_scale"
PROPERTY_INT4_PACK_IS_LOSSLESS = "int4_pack_is_lossless"
PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED = "schedule_occupancy_is_counted"
PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED = "budget_is_exactly_accounted"

OPTIM_PROPERTIES: tuple[str, ...] = (
    PROPERTY_CACHED_EQUALS_RECOMPUTE,
    PROPERTY_PREFILL_MATCHES_FULL,
    PROPERTY_CACHE_GROWTH_IS_A_FORMULA,
    PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE,
    PROPERTY_INT4_PACK_IS_LOSSLESS,
    PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED,
    PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_CACHED_EQUALS_RECOMPUTE: "用缓存的逐步 decode，最后一行与 day086 的整段重算**逐位**相同",
    PROPERTY_PREFILL_MATCHES_FULL: "prefill 的输出与整段前向**逐位**相同，且缓存长度恰好等于提示长度",
    PROPERTY_CACHE_GROWTH_IS_A_FORMULA: "每步新增字节数 == ``2·L·h·bytes``（整数相等，没有取整空间）",
    PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE: "反量化的每个元素误差 <= ``scale/2``（两种方案、两种粒度都要满足）",
    PROPERTY_INT4_PACK_IS_LOSSLESS: "两个 4 位数打包成一个字节再解回来必须**逐位**还原（含边界值 -8 与 7）",
    PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED: "处理表的占用率 == 占位槽位 / 总槽位，且静态批步数 == Σ 每组最长",
    PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED: "权重 + 缓存 + 激活的字节数之和 == 总量，且 ``T_max`` 的两侧都对得上",
}

PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_CACHED_EQUALS_RECOMPUTE: "缓存的 K/V 是在**归一化之后**还是之前取的搞错了（数值全变、形状不变）",
    PROPERTY_PREFILL_MATCHES_FULL: "prefill 用了逐 token 的方式、而比较对象是整段——两者的掩码并不相同",
    PROPERTY_CACHE_GROWTH_IS_A_FORMULA: "把 K 与 V 只算了一份，或者把层数漏乘（差值是一个整数倍）",
    PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE: "量化值的取整方向错了（round 写成 floor 后误差上界会变成 scale）",
    PROPERTY_INT4_PACK_IS_LOSSLESS: "打包时按无符号切、解包时按有符号拼（负数的低 4 位会串到高位）",
    PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED: "把空位算成了占用（占用率虚高，而它的形状完全合法）",
    PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED: "把两次方一次算（例如把 K 与 V 的字节数当成了缓存的全部）",
}

# --------------------------------------------------------------------------------------
# 6. 十条笔记（这是"评估加速手段"这件事的成品）
# --------------------------------------------------------------------------------------

OPTIM_NOTES: dict[str, str] = {
    "cache_is_a_trade": (
        "缓存不是**变快**，而是**换**：它把每一步的计算量从 O(T) 降到 O(1)，"
        "代价是 O(T) 的显存。因此它的方向是'时间 → 空间'，"
        "而'缓存一定更好'这句话在显存紧的时候是不成立的。"
    ),
    "two_tables": (
        "缓存里存的是**两**张表（K 与 V），因此每一条缓存公式前面都有那个 2。"
        "漏掉它读起来像'缓存小了一半'——而它其实是'少存了一半'。"
    ),
    "prefill_is_a_normal_forward": (
        "prefill 就是一次普通的整段前向，只是顺手把每一层的 K/V 存下来。"
        "因此在 prefill 上跑出的输出必须与不缓存时的整段前向**逐位相同**——"
        "这条等式是'我的缓存接对了'的最强证据。"
    ),
    "norm_then_cache": (
        "K/V 是从**归一化之后**的张量投影出来的（pre-LN 块里注意力吃的是 ``ln_1(x)``）。"
        "在归一化之前缓存，形状完全一样、数值全变——而它只在多层堆叠后变得明显。"
    ),
    "absmax_vs_zero_point": (
        "对称量化把 0 精确地映射到 0（没有零点偏移），非对称量化把动态范围用满"
        "（0 要靠 zero 找回来）。对权重来说两者差别不大；对**激活**来说"
        "（它的分布常常是单边、且带离群值）非对称通常更省台阶。"
    ),
    "granularity_is_the_real_win": (
        "把粒度从 per-tensor 换成 per-channel 会明显减小误差（本课在**带离群值**的样本上"
        "量到约 2 倍），而它**不改变位宽**。两件事不能互换：**粒度买精度、位宽买空间**——"
        "'从 8 位降到 4 位'换来一半的显存，代价是台阶数少 16 倍。"
        "本课先把两者的读数分开量出来，再谈该调哪一个。"
    ),
    "scale_is_half_a_step": (
        "对称量化的误差上界是 ``scale/2``，而 scale 是'一格'的宽度。"
        "因此'误差不超过半格'这句话不是经验，是推导；"
        "它成立的前提是每个元素都落在 ``[-max_abs, max_abs]`` 内——"
        "出现 ``inf`` 时前提就破了。"
    ),
    "int4_must_pack": (
        "int4 的 4 位装不满一个字节，因此它**必须**打包（两个数一个字节）。"
        "打包是无损的位运算：它省的是**空间**，不是精度——"
        "精度损失发生在 round 那一步，打包那一步一位都不丢。"
    ),
    "batching_costs_first_token": (
        "批处理的收益是吞吐，代价是**首字延迟**：一次前向要等价最长的那个，"
        "因此最短的那条要陪着等。连续批处理（continuous batching）做的就是"
        "'不等'——一条结束就把新的一条塞进空位。"
    ),
    "quantize_is_not_free": (
        "量化的收益是'同样显存能放更大的模型'，代价是误差；"
        "而误差对**不同层**的影响并不相同（真实实现会给注意力投影更高的精度）。"
        "本课的读数是逐层的误差统计——它不是'整体误差'一个数。"
    ),
}

OPTIM_NOTES_ORDER: tuple[str, ...] = tuple(OPTIM_NOTES)

# --------------------------------------------------------------------------------------
# 7. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

OPTIM_BOUNDARIES: tuple[str, ...] = (
    "本包复现的是**账**（字节数、误差上界、吞吐计数），不是真实推理引擎的性能"
    "（显存分配器、kernel 融合、页式注意力都不在其中）",
    "本包不做真正的 8 位/4 位算术：量化是'量化 → 反量化'，用浮点算出**误差**，"
    "因此它能量误差，不能量真实的加速比",
    "本包不安装、也不调用 bitsandbytes / accelerate：它们只出现在'对接对象'与笔记里",
    "本包的批处理是**静态**批（一次一组），连续批处理只在预算模块里被算成一串计数",
    "本包的缓存不跨请求复用（没有前缀共享），也没有分页——"
    "因此'缓存能不能共享'这个问题本课不回答",
)

# --------------------------------------------------------------------------------------
# 8. 记录
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSpec:
    """一份量化方案：级别 + 方案 + 粒度（三个旋钮，各有一个可读的名字）."""

    level: str = LEVEL_INT8
    scheme: str = SCHEME_ABSMAX
    granularity: str = GRANULARITY_TENSOR

    def __post_init__(self) -> None:
        from smart_research_agent.inference_optim.errors import QuantError

        if self.level not in INTEGER_LEVELS:
            raise QuantError(
                f"量化级别 {self.level!r} 不是整数量化：本包只做 {list(INTEGER_LEVELS)}"
                "（fp16/fp32 没有'台阶'，因此没有 scale 与误差上界）"
            )
        if self.scheme not in SCHEMES:
            raise QuantError(f"未知方案 {self.scheme!r}：可选 {list(SCHEMES)}")
        if self.granularity not in GRANULARITIES:
            raise QuantError(f"未知粒度 {self.granularity!r}：可选 {list(GRANULARITIES)}")

    @property
    def bits(self) -> int:
        """这个级别是几位（每个级别由 :data:`INTEGER_BITS` 唯一决定）."""
        return INTEGER_BITS[self.level]

    @property
    def levels_count(self) -> int:
        """可用的量化台阶数：对称是 ``2^(b-1)``、非对称是 ``2^b``（含零点）."""
        if self.scheme == SCHEME_ABSMAX:
            return 2 ** (self.bits - 1)
        return 2**self.bits

    @property
    def max_quantized(self) -> int:
        """量化值的上界（对称是 ``2^(b-1) - 1``，非对称是 ``2^b - 1``）."""
        if self.scheme == SCHEME_ABSMAX:
            return 2 ** (self.bits - 1) - 1
        return 2**self.bits - 1

    @property
    def bytes_per_element(self) -> float:
        """每个元素占多少字节（**int4 是 0.5**：它是打包之后的等效值）."""
        return BYTES_PER_ELEMENT[self.level]

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（含四个派生量）."""
        return {
            "level": self.level,
            "scheme": self.scheme,
            "granularity": self.granularity,
            "bits": self.bits,
            "levels_count": self.levels_count,
            "max_quantized": self.max_quantized,
            "bytes_per_element": self.bytes_per_element,
        }

    def line(self) -> str:
        """一行说明：``int8 absmax per_tensor | 8 位、256 格、1.00 B/元素``."""
        return (
            f"{self.level} {self.scheme} {self.granularity} | {self.bits} 位、"
            f"{self.levels_count} 格、{self.bytes_per_element:.2f} B/元素"
        )


@dataclass(frozen=True)
class QuantizedMatrix:
    """一块被量化过的矩阵：整数值 + scale（+ 可能的 zero），外加原始形状.

    它刻意**不丢失形状**：量化是逐元素的操作，但反量化要知道"按哪一行还原"，
    而 per-channel 粒度下那个"行"就是 ``rows``。
    """

    values: tuple[int, ...]
    rows: int
    columns: int
    scale: float
    zero: int = 0
    scheme: str = SCHEME_ABSMAX
    granularity: str = GRANULARITY_TENSOR
    scales: tuple[float, ...] = field(default=())
    zeros: tuple[int, ...] = field(default=())
    bits: int = 8

    @property
    def count(self) -> int:
        """元素个数."""
        return len(self.values)

    @property
    def packed(self) -> bool:
        """是否打包（只有 int4 打包：4 位装不满一个字节）."""
        return self.bits == BIT_WIDTHS[LEVEL_INT4]

    @property
    def per_byte(self) -> int:
        """一个字节装几个数（就是打包比）."""
        return INT4_PER_BYTE if self.packed else 1

    @property
    def packed_bytes(self) -> int:
        """**打包之后**的字节数（int4 两个数一个字节；其余一个数一个字节）.

        它回答的是"这块权重在显存里占多大"——因此它必须按打包后的算，
        否则 int4 的收益会凭空少一半（那正是最容易算错的一格）。
        """
        return math.ceil(self.count / self.per_byte)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**整数值本身，只印形状与两个字节读数）."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "count": self.count,
            "scale": self.scale,
            "zero": self.zero,
            "scheme": self.scheme,
            "granularity": self.granularity,
            "bits": self.bits,
            "packed": self.packed,
            "packed_bytes": self.packed_bytes,
            "scales": list(self.scales),
            "zeros": list(self.zeros),
        }


@dataclass(frozen=True)
class ErrorStats:
    """一次量化的误差统计（**四个数**，而不是"误差很小"）."""

    level: str
    scheme: str
    granularity: str
    max_abs_error: float
    mean_abs_error: float
    snr_db: float
    elements: int

    @property
    def direction(self) -> str:
        """这条读数的方向：误差类永远是 ``lower``（越小越好）."""
        return "lower"

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（含方向）."""
        return {
            "level": self.level,
            "scheme": self.scheme,
            "granularity": self.granularity,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "snr_db": self.snr_db,
            "elements": self.elements,
            "direction": self.direction,
        }

    def line(self) -> str:
        """一行说明：``int8 absmax per_tensor | 最大误差 1.9e-03 | 均值 5.1e-04 | SNR 52.4 dB``."""
        return (
            f"{self.level:<5} {self.scheme:<10} {self.granularity:<11} | "
            f"最大误差 {self.max_abs_error:.3e} | 均值 {self.mean_abs_error:.3e} | "
            f"SNR {self.snr_db:6.2f} dB | {self.elements} 个元素"
        )


@dataclass(frozen=True)
class CacheLayer:
    """一层缓存的账：长度 + 两个维度 + 两个字节读数."""

    layer: int
    length: int
    capacity: int
    hidden: int
    bytes_per_element: float

    @property
    def bytes(self) -> int:
        """这一层缓存占多少字节（**K 与 V 各一份**，因此那个 2 在这里）."""
        return int(CACHE_TABLES * self.length * self.hidden * self.bytes_per_element)

    @property
    def free(self) -> int:
        """还能装多少个位置."""
        return max(self.capacity - self.length, 0)

    @property
    def used_ratio(self) -> float:
        """占用率（0 = 空、1 = 满）——它是"还能生成多长"的唯一依据."""
        if self.capacity == 0:
            return 0.0
        return self.length / self.capacity

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "layer": self.layer,
            "length": self.length,
            "capacity": self.capacity,
            "hidden": self.hidden,
            "bytes_per_element": self.bytes_per_element,
            "bytes": self.bytes,
            "free": self.free,
            "used_ratio": self.used_ratio,
        }

    def line(self) -> str:
        """一行说明：``层 0：12/32 位置、16 维、1.00 B/元素 ⇒ 384 字节``."""
        return (
            f"层 {self.layer}：{self.length}/{self.capacity} 位置、{self.hidden} 维、"
            f"{self.bytes_per_element:.2f} B/元素 ⇒ {self.bytes} 字节"
            f"（占用 {self.used_ratio:.1%}）"
        )


@dataclass(frozen=True)
class CacheState:
    """整个缓存的账：每一层一行 + 一个总量."""

    layers: tuple[CacheLayer, ...]
    level: str = LEVEL_FP32

    @property
    def depth(self) -> int:
        """层数."""
        return len(self.layers)

    @property
    def length(self) -> int:
        """当前长度（各层长度相同；不同时下面那条断言会拦下来）."""
        if not self.layers:
            return 0
        first = self.layers[0].length
        for layer in self.layers[1:]:
            if layer.length != first:  # pragma: no cover - 由 append 保证
                from smart_research_agent.inference_optim.errors import ShapeError

                raise ShapeError(
                    f"第 {layer.layer} 层缓存长度 {layer.length} 与第 0 层的 {first} 不一致："
                    "各层长度不齐时'当前长度'这个数就没有意义了。"
                )
        return first

    @property
    def total_bytes(self) -> int:
        """缓存总量（把每一层加起来——这就是"能被逐项相加"的那件事）."""
        return sum(layer.bytes for layer in self.layers)

    @property
    def capacity(self) -> int:
        """容量（各层相同）."""
        return self.layers[0].capacity if self.layers else 0

    @property
    def free(self) -> int:
        """剩余位置."""
        return self.layers[0].free if self.layers else 0

    def lines(self) -> tuple[str, ...]:
        """每一层一行（报告里读它）."""
        return tuple(layer.line() for layer in self.layers)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段（含总量与四个上下文读数）."""
        return {
            "depth": self.depth,
            "length": self.length,
            "capacity": self.capacity,
            "free": self.free,
            "level": self.level,
            "total_bytes": self.total_bytes,
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class BudgetBreakdown:
    """一次部署的预算账：三项 + 一个总量 + 一个"放得下吗".

    三个字段的名字与它们的来源必须一一对应，否则"放不下"这句话无法被追到具体哪一项：

    ```text
    weights       参数量 × 每元素字节（与序列长度无关）
    cache         ``2·L·T·h·bytes``（与序列长度**线性相关**）
    activations   一次前向的中间张量（与批大小线性相关，本课按最保守的一层估）
    ```
    """

    level: str
    parameters: int
    tokens: int
    hidden: int
    layers: int
    batch: int
    weights_bytes: int
    cache_bytes: int
    activation_bytes: int
    budget_bytes: int

    @property
    def total_bytes(self) -> int:
        """三项之和（**它就是"逐项相加"这句话本身**）."""
        return self.weights_bytes + self.cache_bytes + self.activation_bytes

    @property
    def fits(self) -> bool:
        """放得下吗（``<=``，边界算放得下）."""
        return self.total_bytes <= self.budget_bytes

    @property
    def headroom_bytes(self) -> int:
        """余量（放不下时是负数——那是一个能被看见的负数，而不是"失败"两个字）."""
        return self.budget_bytes - self.total_bytes

    @property
    def share(self) -> dict[str, float]:
        """三项各占多少（分母是**总量**而不是预算：这样三项之和恒为 1）."""
        total = self.total_bytes
        if total == 0:  # pragma: no cover - 只有零参数量才会到这里
            return {"weights": 0.0, "cache": 0.0, "activations": 0.0}
        return {
            "weights": self.weights_bytes / total,
            "cache": self.cache_bytes / total,
            "activations": self.activation_bytes / total,
        }

    def require_fits(self) -> None:
        """放不下时抛 :class:`errors.BudgetError`（消息里带上三项与余量）."""
        if self.fits:
            return
        from smart_research_agent.inference_optim.errors import BudgetError

        raise BudgetError(
            f"{self.level} 下放不下：权重 {self.weights_bytes} + 缓存 {self.cache_bytes} + "
            f"激活 {self.activation_bytes} = {self.total_bytes} 字节 > 预算 {self.budget_bytes} "
            f"（差 {abs(self.headroom_bytes)} 字节）。出路是改部署或改精度——"
            "而不是把上下文偷偷截短。"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段（含三项占比与余量）."""
        return {
            "level": self.level,
            "parameters": self.parameters,
            "tokens": self.tokens,
            "hidden": self.hidden,
            "layers": self.layers,
            "batch": self.batch,
            "weights_bytes": self.weights_bytes,
            "cache_bytes": self.cache_bytes,
            "activation_bytes": self.activation_bytes,
            "total_bytes": self.total_bytes,
            "budget_bytes": self.budget_bytes,
            "headroom_bytes": self.headroom_bytes,
            "fits": self.fits,
            "share": self.share,
        }

    def line(self) -> str:
        """一行说明：``fp16 | 权重 23 072 + 缓存 8 192 + 激活 4 096 = 35 360 / 65 536 ✓``."""
        mark = "✓" if self.fits else "✗"
        return (
            f"{self.level:<5} | 权重 {self.weights_bytes:>8} + 缓存 {self.cache_bytes:>8} + "
            f"激活 {self.activation_bytes:>8} = {self.total_bytes:>9} / {self.budget_bytes:>9} {mark}"
            f" | 余量 {self.headroom_bytes}"
        )


@dataclass(frozen=True)
class StepRow:
    """一次生成的一步（缓存账里的一行）."""

    step: int
    position: int
    cache_bytes: int
    delta_bytes: int
    stage: str

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "step": self.step,
            "position": self.position,
            "cache_bytes": self.cache_bytes,
            "delta_bytes": self.delta_bytes,
            "stage": self.stage,
        }

    def line(self) -> str:
        """一行说明：``第 3 步（decode，位置 8）：缓存 4 608 字节（+512）``."""
        return (
            f"第 {self.step} 步（{self.stage}，位置 {self.position}）："
            f"缓存 {self.cache_bytes} 字节（{self.delta_bytes:+d}）"
        )


def element_bytes(level: str) -> float:
    """一个级别下每个元素占多少字节（未知级别当场拒绝）."""
    if level not in BYTES_PER_ELEMENT:
        from smart_research_agent.inference_optim.errors import ParameterError

        raise ParameterError(
            f"未知的精度级别 {level!r}：本包只认 {list(LEVELS)}。"
            "回退到某个默认级别的后果是——一份 fp32 的预算读数被印成 int8 的。"
        )
    return BYTES_PER_ELEMENT[level]


def cache_bytes(layers: int, tokens: int, hidden: int, level: str = LEVEL_FP32) -> int:
    """**缓存的唯一公式**：``2 · L · T · h · bytes``.

    这个函数存在的理由是"一个量只被写一遍"：报告、测试、预算三处都调它，
    因此"缓存公式"这件事不可能分家。
    """
    if layers < 1 or tokens < 0 or hidden < 1:
        from smart_research_agent.inference_optim.errors import ParameterError

        raise ParameterError(
            f"缓存公式的参数不合法：layers={layers}、tokens={tokens}、hidden={hidden}。"
            "（tokens 可以为 0——那是 prefill 之前的空缓存。）"
        )
    return int(CACHE_TABLES * layers * tokens * hidden * element_bytes(level))


def norm_or_raise(vector: Vector) -> float:
    """向量的 L2 范数（报告里用一个数概括"这一行的量级"）."""
    return math.sqrt(math.fsum(value * value for value in vector))


def flatten(matrix: Matrix) -> tuple[float, ...]:
    """把矩阵摊平成一串元素（量化按元素走，因此它需要一个顺序）.

    顺序是**逐行**的：因此 per-channel 粒度下的"通道"就是"行"，
    而这个顺序必须与 :meth:`QuantizedMatrix.rows` 一致。
    """
    return tuple(value for row in matrix for value in row)


__all__ = [
    "ACCELERATE_LIBRARY",
    "BITS_PER_BYTE",
    "BIT_WIDTHS",
    "BYTES_PER_ELEMENT",
    "CACHE_TABLES",
    "DECODE_TOKENS_PER_STEP",
    "GRANULARITIES",
    "GRANULARITY_CHANNEL",
    "GRANULARITY_DESCRIPTIONS",
    "GRANULARITY_TENSOR",
    "INT4_PER_BYTE",
    "INTEGER_BITS",
    "INTEGER_LEVELS",
    "LAYOUTS",
    "LAYOUT_DESCRIPTIONS",
    "LAYOUT_GROW",
    "LAYOUT_PREALLOCATED",
    "LEVELS",
    "LEVEL_DESCRIPTIONS",
    "LEVEL_FP16",
    "LEVEL_FP32",
    "LEVEL_INT4",
    "LEVEL_INT8",
    "OPTIM_BOUNDARIES",
    "OPTIM_NOTES",
    "OPTIM_NOTES_ORDER",
    "OPTIM_PROPERTIES",
    "PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED",
    "PROPERTY_CACHED_EQUALS_RECOMPUTE",
    "PROPERTY_CACHE_GROWTH_IS_A_FORMULA",
    "PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_INT4_PACK_IS_LOSSLESS",
    "PROPERTY_PREFILL_MATCHES_FULL",
    "PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED",
    "QUANT_LIBRARY",
    "QUANT_LIBRARY_VERSION",
    "SCHEMES",
    "SCHEME_ABSMAX",
    "SCHEME_DESCRIPTIONS",
    "SCHEME_ZERO_POINT",
    "STAGES",
    "STAGE_DECODE",
    "STAGE_DESCRIPTIONS",
    "STAGE_PREFILL",
    "TRANSFORMERS_VERSION",
    "BudgetBreakdown",
    "CacheLayer",
    "CacheState",
    "ErrorStats",
    "QuantSpec",
    "QuantizedMatrix",
    "StepRow",
    "cache_bytes",
    "element_bytes",
    "flatten",
    "norm_or_raise",
]
