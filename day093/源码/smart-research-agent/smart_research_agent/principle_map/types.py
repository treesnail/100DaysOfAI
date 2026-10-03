"""``principle_map`` 的四层、六个应用、十二块拼图与七条性质（day088 / M7-D12）.

day087 结束时，这条链路上已经有了四本"账"：数学地基（day073）、注意力与结构
（day075~day079）、表征与检索（day041/064）、高效推理（day087）。今天的动作只有一件：

```text
把那十二块拼图拼成一张**可校验**的图：
    原理（底层）→ 实现（项目里的包与函数）→ 应用（Agent/RAG 的六个能力）
```

## 一、四层与六个应用

```text
LAYER_MATH            数学地基        向量 / 余弦 / 可微检索
LAYER_ATTENTION       注意力与结构    行分布 / 因果掩码 / 分头 / 位置
LAYER_REPRESENTATION  表征与检索      方向相似 / 排序对照 / 缓存复用
LAYER_INFERENCE       推理与部署      缓存公式 / 量化上界 / 预算拆分
```

六个应用是"用户能看见的能力"：``agent_reasoning`` / ``rag_retrieval`` / ``rag_rerank`` /
``generation`` / ``serving`` / ``explainability``。图必须保证**每个应用至少被一条原理支撑**。

## 二、一条纪律：**读数必须能被指向一个探针**

每一块拼图都带三个坐标：``artifact``（项目里的哪个函数）、``probe``（哪个探针去量它）、
``application``（它支撑哪个能力）。三者都用**字符串名字**写下来，而不是对象引用——
因为名字可以被解析、被检查、被报告；一个对象引用只能"看起来对"。

## 三、另一条纪律：**方向必须写清楚**

今天的读数分两类：

```text
相等（==）      证据读数与期望**逐位相同**（缓存路径 vs 整段重算、行和 == 1）
上界（<=）      证据读数**不超过**某个界（量化误差 <= scale/2）
```

因此 :class:`Evidence` 与 day087 的 ``CrossCheck`` 同源：有 ``upper_bound`` 时判据是"≤"，
没有时才是"=="。把两类混成一个判据，就会出现"实测误差恰好等于 0（因为输入全是 0）
被当成通过"这种事。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# 1. 四个层（**一个层 = 一串前置依赖里的一个阶段**）
# --------------------------------------------------------------------------------------

LAYER_MATH = "math"
LAYER_ATTENTION = "attention"
LAYER_REPRESENTATION = "representation"
LAYER_INFERENCE = "inference"

LAYERS: tuple[str, ...] = (LAYER_MATH, LAYER_ATTENTION, LAYER_REPRESENTATION, LAYER_INFERENCE)

LAYER_DESCRIPTIONS: dict[str, str] = {
    LAYER_MATH: "数学地基：点积 / 余弦 / 可微的检索——注意力打分用的全部零件（day073）",
    LAYER_ATTENTION: "注意力与结构：行分布 / 因果掩码 / 投影内部分头 / 位置注入（day075~079）",
    LAYER_REPRESENTATION: "表征与检索：方向相似 / 排序对照 / 缓存复用逐位相同（day041/064/087）",
    LAYER_INFERENCE: "推理与部署：缓存公式 / 量化上界 / 预算三项拆分（day087）",
}

#: 提纲与图里层的**前置依赖次序**（先有数学，再有注意力，再有表征，最后是推理）.
LAYER_ORDER: tuple[str, ...] = (LAYER_MATH, LAYER_ATTENTION, LAYER_REPRESENTATION, LAYER_INFERENCE)

# --------------------------------------------------------------------------------------
# 2. 六个应用（**用户能看见的六种能力**）
# --------------------------------------------------------------------------------------

APP_AGENT_REASONING = "agent_reasoning"
APP_RAG_RETRIEVAL = "rag_retrieval"
APP_RAG_RERANK = "rag_rerank"
APP_GENERATION = "generation"
APP_SERVING = "serving"
APP_EXPLAINABILITY = "explainability"

APPLICATIONS: tuple[str, ...] = (
    APP_AGENT_REASONING,
    APP_RAG_RETRIEVAL,
    APP_RAG_RERANK,
    APP_GENERATION,
    APP_SERVING,
    APP_EXPLAINABILITY,
)

APPLICATION_DESCRIPTIONS: dict[str, str] = {
    APP_AGENT_REASONING: "Agent 推理：把'哪些内容相关'变成可微的混合，模型才能学该看哪里",
    APP_RAG_RETRIEVAL: "RAG 检索：用方向（余弦）而不是长度来排序，并把向量归一到单位长度",
    APP_RAG_RERANK: "RAG 重排：注意力峰值与检索排序逐行对照，两把尺子互相验证",
    APP_GENERATION: "文本生成：因果掩码挡住未来，位置编码打破置换，预算决定最长能生成多少",
    APP_SERVING: "在线服务：投影内部分头、缓存复用逐位相同、字节公式与量化上界",
    APP_EXPLAINABILITY: "可解释性：注意力的每一行是一个分布，因此它是一张可读的热力图",
}

# --------------------------------------------------------------------------------------
# 3. 十二块拼图（**四层，条数 2 / 4 / 3 / 3**）
# --------------------------------------------------------------------------------------

PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL = "attention_is_differentiable_retrieval"
PRINCIPLE_COSINE_IS_NORMALIZED_DOT = "cosine_is_normalized_dot"
PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS = "attention_rows_are_distributions"
PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE = "causal_mask_blocks_future"
PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION = "multi_head_splits_inside_projection"
PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION = "position_encoding_breaks_permutation"
PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION = "embedding_similarity_is_direction"
PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS = "rank_ordering_matches_attention_peaks"
PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT = "cache_reuse_is_bitwise_exact"
PRINCIPLE_CACHE_BYTES_IS_A_FORMULA = "cache_bytes_is_a_formula"
PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP = "quantization_error_bounded_by_half_step"
PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN = "generation_respects_budget_breakdown"

PRINCIPLES: tuple[str, ...] = (
    PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL,
    PRINCIPLE_COSINE_IS_NORMALIZED_DOT,
    PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE,
    PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION,
    PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION,
    PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION,
    PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS,
    PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT,
    PRINCIPLE_CACHE_BYTES_IS_A_FORMULA,
    PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP,
    PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN,
)

# --------------------------------------------------------------------------------------
# 4. 七条性质（判据分四类）
# --------------------------------------------------------------------------------------

PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT = "every_principle_has_artifact"
PROPERTY_EVERY_APPLICATION_IS_SUPPORTED = "every_application_is_supported"
PROPERTY_EVIDENCE_IS_REPRODUCIBLE = "evidence_is_reproducible"
PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS = "attention_rows_are_distributions"
PROPERTY_CACHE_FORMULA_MATCHES_DAY087 = "cache_formula_matches_day087"
PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES = "outline_respects_dependencies"
PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES = "document_covers_all_principles"

PRINCIPLE_PROPERTIES: tuple[str, ...] = (
    PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT,
    PROPERTY_EVERY_APPLICATION_IS_SUPPORTED,
    PROPERTY_EVIDENCE_IS_REPRODUCIBLE,
    PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_CACHE_FORMULA_MATCHES_DAY087,
    PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES,
    PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT: "十二条命题指向的实现都能被解析出来（``artifact`` 是一个真实的函数）",
    PROPERTY_EVERY_APPLICATION_IS_SUPPORTED: "六个应用每一个都至少被一条原理支撑（没有一座孤岛）",
    PROPERTY_EVIDENCE_IS_REPRODUCIBLE: "同一个探针连续两次调用得到的证据**逐位相同**（图是单次计算、可重复的）",
    PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS: "跨天对账：调 ``transformer_core``，判据是'行和与 1 的偏差 <= 容差'",
    PROPERTY_CACHE_FORMULA_MATCHES_DAY087: "跨天对账：调 ``inference_optim.cache_bytes``，整数相等（没有容差空间）",
    PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES: "提纲的层次序满足前置依赖（数学 → 注意力 → 表征 → 推理）",
    PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES: "原理文档覆盖全部十二条命题（每一条的 id 都出现在正文里）",
}

PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT: "命题指向的函数被改名或搬走了——图里有一个指向空气的箭头",
    PROPERTY_EVERY_APPLICATION_IS_SUPPORTED: "某个应用在图上没有任何入边——它是一句没有依据的话",
    PROPERTY_EVIDENCE_IS_REPRODUCIBLE: "探针吃到了未固定的随机性（时间 / 哈希序 / 全局状态），两次读数不同",
    PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS: "softmax 的分母漏了一部分（被掩码的位置参与了归一化）",
    PROPERTY_CACHE_FORMULA_MATCHES_DAY087: "把 K 与 V 只算了一份，或把层数漏乘（差值是一个整数倍）",
    PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES: "先讲部署再讲数学——听众在第一页就掉队了",
    PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES: "文档漏掉了某一条命题，而'没写'与'不成立'读起来一样",
}

# --------------------------------------------------------------------------------------
# 5. 十条笔记（这是"串起来"这件事的成品）
# --------------------------------------------------------------------------------------

PRINCIPLE_NOTES: dict[str, str] = {
    "attention_is_retrieval_made_differentiable": (
        "检索与注意力都在回答'哪些内容与当前问题相关'。差别在于前者是**离散的选择**"
        "（取或不取，梯度传不回去），后者是**可微的混合**（想要什么是一个概率分布）。"
        "这就是'能学'这句话的技术前提。"
    ),
    "cosine_ignores_length": (
        "余弦相似度只看方向、不看长度：把任一侧放大 k 倍，余弦值不变。"
        "因此向量入库前要**归一化**——否则'长文档'会因为模长大而占便宜，"
        "而这件事在结果上只表现为'排序有点怪'。"
    ),
    "rows_are_distributions": (
        "注意力权重矩阵的每一行是一个和为 1 的分布。这一条是'热力图可读'的前提："
        "一个不是分布的行仍然能画出来，而它背后多半是 softmax 的分母错了。"
    ),
    "causal_mask_is_a_table": (
        "因果掩码是一张**可以打印出来看的布尔表**（``mask[i][j] = (j <= i)``），"
        "而不是一个需要读代码才知道含义的 ``None``。被挡住的位置权重**恰好是 0.0**。"
    ),
    "split_heads_is_inside_projection": (
        "'分头'不是换一个块类型，而是一次 ``view`` + ``transpose``——它发生在**投影内部**。"
        "因此 ``merge(split(x)) == x`` 是一条**逐位**性质，而不是'形状一致'性质。"
    ),
    "position_breaks_permutation": (
        "没有位置编码时，注意力是**置换等变**的：把输入的行换一个顺序，输出只是跟着换。"
        "注入位置编码（或开因果掩码）之后这条性质**失效**——而'失效'正是模型知道顺序的证据。"
    ),
    "rank_correlation_is_the_continuous_view": (
        "top-k 重合是一个**离散量**（k=1 时只取 0 或 1），秩相关是**连续量**。"
        "两者一起看才能区分'整体排序像'与'只是最相关的那几个恰好相同'。"
    ),
    "cache_is_a_trade": (
        "缓存不是**变快**，而是**换**：它把每步的计算量从 O(T) 降到 O(1)，"
        "代价是 O(T) 的显存。因此'缓存一定更好'在显存紧的时候是不成立的。"
    ),
    "scale_is_half_a_step": (
        "对称量化的误差上界是 ``scale/2``，而 scale 是'一格'的宽度。"
        "'误差不超过半格'不是经验，是推导；它的前提是每个元素都落在量程内。"
    ),
    "budget_is_a_sum_of_three": (
        "放不下的时候要能回答'**是哪一项放不下**'：权重、缓存、激活的出路完全不同。"
        "把总量当成一个数（而不是三项之和）的后果是——'显存不够'这句话没有任何下一步动作。"
    ),
}

#: 十条笔记的键（顺序即写入顺序，报告里读它）.
PRINCIPLE_NOTES_ORDER: tuple[str, ...] = tuple(PRINCIPLE_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

PRINCIPLE_BOUNDARIES: tuple[str, ...] = (
    "本包复现的是**图与对账**，不是新的模型能力——它不新增任何算术，只把既有读数对齐",
    "本包的十二块拼图是**课程口径**下的划分，不是'Transformer 原理的全部'；"
    "它只覆盖本项目真实实现过的那几条",
    "本包不安装、也不调用 transformers / torch / numpy：跨天对账走的是本项目自己的实现",
    "本包的探针输入全部写死（必要时用确定性 LCG 造数），因此它复现的是**读数**、不是统计规律",
    "本包不承诺那张分享提纲的'效果'——提纲的时长与次序是可断言的，"
    "而'听众听懂了没有'不是本包能度量的东西",
)

# --------------------------------------------------------------------------------------
# 7. 记录
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Principle:
    """一块拼图：底层命题 + 它在项目里的落点 + 它支撑的应用.

    ``artifact`` 与 ``probe`` 都是**字符串名字**（``模块.函数``），而不是函数对象：
    名字可以被解析、被检查、被打印，因此"这条原理有实现落点"这件事是可断言的。
    """

    id: str
    layer: str
    statement: str
    artifact: str
    probe: str
    application: str
    source_day: str

    def __post_init__(self) -> None:
        from smart_research_agent.principle_map.errors import ParameterError

        if not self.id:
            raise ParameterError("原理的 id 不能为空：它是图上唯一的键。")
        if self.layer not in LAYERS:
            raise ParameterError(
                f"未知的层 {self.layer!r}：可选 {list(LAYERS)}。"
                "回退到某一层的后果是——一张图里有两块拼图被算成了同一个阶段。"
            )
        if self.application not in APPLICATIONS:
            raise ParameterError(
                f"未知的应用 {self.application!r}：可选 {list(APPLICATIONS)}。"
            )
        if "." not in self.artifact:
            raise ParameterError(
                f"artifact 必须是'模块.函数'的形式，收到 {self.artifact!r}："
                "没有点号的引用无法被解析成一条可检查的落点。"
            )
        if not self.probe:
            raise ParameterError(f"原理 {self.id!r} 没有探针：缺证据的命题不算命题。")
        if not self.source_day:
            raise ParameterError(f"原理 {self.id!r} 没有来源天：它必须能追溯到哪一天。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "id": self.id,
            "layer": self.layer,
            "statement": self.statement,
            "artifact": self.artifact,
            "probe": self.probe,
            "application": self.application,
            "source_day": self.source_day,
        }

    def line(self) -> str:
        """一行说明：``[math] attention_is_differentiable_retrieval -> agent_reasoning``."""
        return f"[{self.layer}] {self.id} → {self.application}（{self.source_day}）"


@dataclass(frozen=True)
class Application:
    """一个应用：它的说明 + 支撑它的原理 id（**可以为空——那正是要被检查出来的**）."""

    id: str
    description: str
    principles: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        from smart_research_agent.principle_map.errors import ParameterError

        if self.id not in APPLICATIONS:
            raise ParameterError(f"未知的应用 {self.id!r}：可选 {list(APPLICATIONS)}。")
        if not self.description:
            raise ParameterError(f"应用 {self.id!r} 必须有说明。")

    @property
    def supported(self) -> bool:
        """是否至少被一条原理支撑."""
        return len(self.principles) > 0

    def with_principles(self, principles: tuple[str, ...]) -> Application:
        """按给定的原理集合返回一份新记录（图在构建时用它填边）."""
        return Application(id=self.id, description=self.description, principles=principles)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "id": self.id,
            "description": self.description,
            "principles": list(self.principles),
            "supported": self.supported,
        }

    def line(self) -> str:
        """一行说明：``serving | 4 条原理 | ...``."""
        return f"{self.id} | {len(self.principles)} 条原理 | {self.description}"


@dataclass(frozen=True)
class Evidence:
    """一个探针的一次读数：命题 + 读数 + 期望 + 判据（**含方向与上界**）.

    与 day087 的 ``CrossCheck`` 同源：有 ``upper_bound`` 时判据是"实测 <= 上界"，
    没有时才是"两个数相等"。``exact=True`` 要求逐位/整数相等，否则走容差。
    """

    principle: str
    reading: float
    expected: float
    exact: bool = True
    upper_bound: float | None = None
    source: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        import math

        from smart_research_agent.principle_map.errors import NumericError, ParameterError

        if not self.principle:
            raise ParameterError("证据必须指向一条命题（principle 不能为空）。")
        if not math.isfinite(self.reading) or not math.isfinite(self.expected):
            raise NumericError(
                f"证据读数必须有限：reading={self.reading!r}、expected={self.expected!r}。"
                "非有限数会让任何比较静默为假。"
            )
        if self.upper_bound is not None:
            if not math.isfinite(self.upper_bound) or self.upper_bound < 0:
                raise NumericError(
                    f"上界必须是有限非负数，收到 {self.upper_bound!r}："
                    "一个为负的界会让'误差不超过它'永远失败。"
                )

    @property
    def passed(self) -> bool:
        """相等（逐位 / 整数 / 容差）或"不超过上界"两种判据."""
        if self.upper_bound is not None:
            return float(self.reading) <= float(self.upper_bound) + 1e-12
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= 1e-12

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "principle": self.principle,
            "reading": self.reading,
            "expected": self.expected,
            "exact": self.exact,
            "upper_bound": self.upper_bound,
            "source": self.source,
            "note": self.note,
            "passed": self.passed,
        }

    def line(self) -> str:
        """一行可读的读数（带上来源与那个上界）."""
        verdict = "满足" if self.passed else "不满足"
        if self.upper_bound is not None:
            return (
                f"[{verdict}] {self.principle}: 读数 {self.reading:.6e} ≤ 上界 "
                f"{self.upper_bound:.6e}（{self.source}）"
            )
        return (
            f"[{verdict}] {self.principle}: 读数 {self.reading} / 期望 {self.expected}"
            f"（{self.source}）"
        )


@dataclass(frozen=True)
class TalkSection:
    """一场分享的一节：标题 + 三条要点 + 一条可运行的 demo + 时长（分钟）.

    ``layer`` 是这一节对应的层，用来做**前置依赖**的拓扑检验（见 ``outline.check_order``）。
    """

    index: int
    title: str
    bullets: tuple[str, ...]
    demo: str
    minutes: int
    layer: str

    def __post_init__(self) -> None:
        from smart_research_agent.principle_map.errors import ParameterError, ShapeError

        if self.index < 1:
            raise ParameterError(f"节号必须 >= 1，收到 {self.index}。")
        if self.layer not in LAYERS:
            raise ParameterError(f"未知的层 {self.layer!r}：可选 {list(LAYERS)}。")
        if len(self.bullets) != 3:
            raise ShapeError(
                f"第 {self.index} 节有 {len(self.bullets)} 条要点：本提纲每节恰好 3 条——"
                "少了讲不完整、多了听众记不住。"
            )
        if not self.title or not self.demo:
            raise ParameterError(f"第 {self.index} 节必须有标题与 demo 命令。")
        if self.minutes <= 0:
            raise ParameterError(f"第 {self.index} 节的时长必须为正，收到 {self.minutes}。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "title": self.title,
            "bullets": list(self.bullets),
            "demo": self.demo,
            "minutes": self.minutes,
            "layer": self.layer,
        }

    def line(self) -> str:
        """一行说明：``第 1 节 [math] ...（8 分钟）``."""
        return f"第 {self.index} 节 [{self.layer}] {self.title}（{self.minutes} 分钟）"


@dataclass(frozen=True)
class CoverageReport:
    """一张图的覆盖报告：按层计数 + 按应用计数 + 是否完整.

    ``complete`` 是**两个条件同时成立**：每个应用至少一条原理、每条原理都有实现落点。
    把两个条件写成一个布尔量是刻意的——报告里只印一个'完整/不完整'，
    而明细（哪一层几条、哪个应用几条）留在两个字典里。
    """

    per_layer: dict[str, int]
    per_application: dict[str, int]
    complete: bool

    def __post_init__(self) -> None:
        from smart_research_agent.principle_map.errors import ShapeError

        if set(self.per_layer) != set(LAYERS):
            raise ShapeError(
                f"按层计数的键必须与 LAYERS 一致：多 {sorted(set(self.per_layer) - set(LAYERS))}、"
                f"缺 {sorted(set(LAYERS) - set(self.per_layer))}。"
            )
        if set(self.per_application) != set(APPLICATIONS):
            raise ShapeError(
                f"按应用计数的键必须与 APPLICATIONS 一致："
                f"多 {sorted(set(self.per_application) - set(APPLICATIONS))}、"
                f"缺 {sorted(set(APPLICATIONS) - set(self.per_application))}。"
            )

    @property
    def unsupported(self) -> tuple[str, ...]:
        """一条原理都没有支撑的应用（应当为空）."""
        return tuple(app for app, count in self.per_application.items() if count == 0)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "per_layer": dict(self.per_layer),
            "per_application": dict(self.per_application),
            "complete": self.complete,
            "unsupported": list(self.unsupported),
        }

    def line(self) -> str:
        """一行读数：``覆盖 12 条原理 / 4 层 / 6 应用 | 完整 True``."""
        return (
            f"覆盖 {sum(self.per_layer.values())} 条原理 / {len(self.per_layer)} 层 / "
            f"{len(self.per_application)} 应用 | 完整 {self.complete}"
        )


def require_layer(layer: str) -> str:
    """校验一个层名（未知层当场拒绝）."""
    if layer not in LAYERS:
        from smart_research_agent.principle_map.errors import ParameterError

        raise ParameterError(f"未知的层 {layer!r}：可选 {list(LAYERS)}。")
    return layer


def require_application(application: str) -> str:
    """校验一个应用名（未知应用当场拒绝）."""
    if application not in APPLICATIONS:
        from smart_research_agent.principle_map.errors import ParameterError

        raise ParameterError(f"未知的应用 {application!r}：可选 {list(APPLICATIONS)}。")
    return application


def require_principle_id(principle: str) -> str:
    """校验一个原理 id（未知 id 当场拒绝）."""
    if principle not in PRINCIPLES:
        from smart_research_agent.principle_map.errors import ReferenceError

        raise ReferenceError(f"未知的原理 id {principle!r}：可选 {list(PRINCIPLES)}。")
    return principle


__all__ = [
    "APPLICATIONS",
    "APPLICATION_DESCRIPTIONS",
    "APP_AGENT_REASONING",
    "APP_EXPLAINABILITY",
    "APP_GENERATION",
    "APP_RAG_RERANK",
    "APP_RAG_RETRIEVAL",
    "APP_SERVING",
    "LAYERS",
    "LAYER_ATTENTION",
    "LAYER_DESCRIPTIONS",
    "LAYER_INFERENCE",
    "LAYER_MATH",
    "LAYER_ORDER",
    "LAYER_REPRESENTATION",
    "PRINCIPLES",
    "PRINCIPLE_BOUNDARIES",
    "PRINCIPLE_CACHE_BYTES_IS_A_FORMULA",
    "PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT",
    "PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE",
    "PRINCIPLE_COSINE_IS_NORMALIZED_DOT",
    "PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL",
    "PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS",
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
    "Application",
    "CoverageReport",
    "Evidence",
    "Principle",
    "TalkSection",
    "require_application",
    "require_layer",
    "require_principle_id",
]
