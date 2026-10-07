"""``adapters``：薄适配层——把既有子系统的真实 API 收敛成**统一形状**（day099 / G1-D1）.

本模块**不重写任何子系统**：它只做三件事——

```text
1) 用固定语料 + 固定编码器 + MockLLM 装配一个"可离线复算的底座"（build_substrate）；
2) 把每个子系统的**原始返回对象**折成一份统一的读数 dataclass
   （*Reading：只带确定性的标量，不带时间戳 / uuid / 对象引用）；
3) 让 capstone 的其余模块只认识这些读数，而**不认识**底层那些形状各异的返回类型。
```

## 一、今天最值钱的一句话

> **适配层的价值是"把二十种返回形状收敛成九种读数"——
> 收敛之后，报告与性质检查都只面对一种东西：带方向的小数。**

``RetrievalResult`` / ``PackedContext`` / ``Generation`` / ``GroundingReport`` /
``ModerationResult`` / ``InjectionReport`` / ``Plan`` / ``CostTracker.report()``
/Tracer 各自是**八套不同的形状**；本模块把它们折成九份 ``*Reading``，
每一份都有一个 :meth:`reading` 属性（一个可比较的浮点数）与一个 :meth:`line`
（一行可读文本）。于是 :mod:`capstone.verify` 只需要比较小数，
而 :mod:`capstone.study` 只需要打印行。

## 二、底座为什么必须"固定"

```text
语料        6 条写死的记录（正文 / 来源都写死）
编码器      llm.embedding.CharNgramEmbedding（字符 n-gram，纯确定性、零依赖）
模型        llm.mock.MockLLM（按脚本回复：先给规划 JSON、再给带引用的一行答案）
问题        一条写死的问题（向量维度不一致该怎么处理）
金标准      GOLD = ("k-3",)（含 ERR-2043 的那一条）
```

因此 :func:`capstone.assembly.run` 的两次调用**逐位相同**——
这不是"碰巧一致"，而是因为这条链里没有任何一个未固定的量。

## 三、与既有包的接缝

- **安全**：``security.injection_detector`` / ``security.content_moderator``；
- **规划**：``agent.planner.Planner``（配 ``llm.mock.MockLLM``）；
- **检索**：``retrieval.retriever.build_retriever`` + ``retrieval.hybrid.HybridRetriever``
  + ``retrieval.lexical.BM25Params``；
- **打包**：``retrieval.context.pack_context``（真正的预算 / 截断 / 编号）；
- **生成**：``retrieval.generation.RAGGenerator``（开口要求引用）；
- **评估**：``evaluation.rag_metrics`` 的四条纯函数；
- **观测**：``observability.cost_tracker.CostTracker`` / ``observability.tracing.Tracer``；
- **工具**：``tools.calculator.CalculatorTool``；
- **库与编码**：``vectorstore.FlatVectorStore`` / ``vectorstore.types.make_record``
  / ``llm.embedding.CharNgramEmbedding``。

**没有任何一处改动了上面的包**：本模块是它们的使用者。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.capstone.errors import ParameterError
from smart_research_agent.llm.embedding import CharNgramEmbedding, EmbeddingProvider

# 说明：下面这些导入是**子系统级的**——本包是它们的使用者，不修改它们。
from smart_research_agent.agent.planner import Planner
from smart_research_agent.evaluation.rag_metrics import (
    mrr,
    ndcg,
    retrieval_precision,
    retrieval_recall,
)
from smart_research_agent.llm.base import Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.observability.cost_tracker import DEFAULT_PRICE_TABLE, CostTracker
from smart_research_agent.observability.tracing import Tracer
from smart_research_agent.retrieval.context import PackedContext, pack_context
from smart_research_agent.retrieval.generation import Generation, RAGGenerator
from smart_research_agent.retrieval.hybrid import HybridRetriever
from smart_research_agent.retrieval.lexical import BM25Params, LexicalIndex
from smart_research_agent.retrieval.retriever import build_retriever
from smart_research_agent.retrieval.types import RetrievalQuery, RetrievalResult
from smart_research_agent.security.content_moderator import ContentModerator
from smart_research_agent.security.injection_detector import PromptInjectionDetector
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.vectorstore import FlatVectorStore
from smart_research_agent.vectorstore.types import make_record

# --------------------------------------------------------------------------- #
# 固定底座：语料 / 问题 / 脚本 / 预算
# --------------------------------------------------------------------------- #

#: 向量维度（与编码器一致；取 64 是为了让 n-gram 有足够的分辨力）.
DIMENSION = 64

#: 固定语料：``(record_id, 正文, 元数据)``——六条记录、两份来源.
CORPUS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("k-1", "语义缓存用分位数标定阈值，换编码器必须重新标定。", {"source": "docs/rag.md"}),
    ("k-2", "混合检索把向量路与关键词路合起来，两路各留一份证据。", {"source": "docs/hybrid.md"}),
    ("k-3", "ERR-2043 表示向量维度不一致，先重建索引再重试。", {"source": "docs/ops.md"}),
    ("k-4", "上下文打包按预算截断单条，引用编号从 1 起连续。", {"source": "docs/rag.md"}),
    ("k-5", "接地核对只做编号层面：有效引用、幻觉引用与未引用。", {"source": "docs/rag.md"}),
    ("k-6", "成本追踪按模型与接口归因 token，trace 把一次请求记成 span。", {"source": "docs/ops.md"}),
)

#: 固定问题（与金标准同源：它应当命中含 ``ERR-2043`` 的那一条）.
QUESTION = "向量维度不一致该怎么处理"

#: 金标准相关文档（召回的分母）.
GOLD: tuple[str, ...] = ("k-3",)

#: 分级相关（NDCG 用；含一条"部分相关"的文档）.
GRADES: dict[str, float] = {"k-3": 2.0, "k-2": 1.0}

#: MockLLM 的脚本：第一条给规划 JSON，第二条给带一条引用的答案.
PLAN_SCRIPT = '["拆解问题", "检索语料", "核对引用"]'
ANSWER_SCRIPT = "先重建索引再重试：ERR-2043 表示向量维度不一致 [1]。"

#: 记账用的模型名（价格表里有非零单价，因此费用是一个**有意义的**小数）.
PRICE_MODEL = "gpt-4o-mini"

#: 检索与打包的预算（top_k / 整包上限 / 单条上限）.
TOP_K = 3
MAX_CHARS = 600
PER_HIT_CHARS = 240

#: 工具自检用的确定性表达式与期望结果（"联调"证据里最轻的一条）.
TOOL_EXPRESSION = "2 + 3 * (4 - 1)"
TOOL_EXPECTED = "11"

#: 本层的边界（原文供报告引用）.
ADAPTER_BOUNDARY = (
    "适配层不生产信号，它只把信号搬进一个统一的信封——"
    "信封里只有带方向的小数，因此'这次到底怎么样'可以用一条不等式回答。"
)


def _require_positive_int(label: str, value: int) -> int:
    """校验一个 >= 1 的整数（预算类参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的预算意味着这一层根本不该被调用——"
            "放不下任何一条命中的预算会让一次正常检索退化成空上下文。"
        )
    return value


# --------------------------------------------------------------------------- #
# 底座
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Substrate:
    """一套**固定、离线、确定性**的子系统集合（本课所有读数的来源）.

    它是 frozen 的，但它持有的子系统对象是可变的——这没有矛盾：
    "装配参数不该在运行期被换掉"与"子系统内部有自己的状态"是两件事。
    每个 :func:`capstone.assembly.run` 都会新建一份底座（于是两次运行互不干扰）。
    """

    question: str
    gold: tuple[str, ...]
    grades: dict[str, float]
    store: FlatVectorStore
    embedding: EmbeddingProvider
    lexical: LexicalIndex
    vector: Any
    hybrid: HybridRetriever
    generator: RAGGenerator
    detector: PromptInjectionDetector
    moderator: ContentModerator
    planner: Planner
    calculator: CalculatorTool
    llm: MockLLM

    def describe(self) -> dict[str, Any]:
        """底座的一份自述（报告里回显它，证明"读数来自哪一套存量"）."""
        return {
            "question": self.question,
            "corpus": len(CORPUS),
            "dimension": self.embedding.dimension,
            "lexical": self.lexical.summary_line(),
            "store": self.store.info().count,
            "model": type(self.llm).__name__,
            "gold": list(self.gold),
        }


def build_substrate(
    *,
    question: str = QUESTION,
    top_k: int = TOP_K,
) -> Substrate:
    """装配一份固定底座（**每个 run 调一次，互不干扰**）.

    顺序是固定的，因为它复现的是一条真实的装配链：
    库 → 编码器 → 关键词索引 → 向量检索器 → 混合检索器 → 生成器 → 护栏 / 规划 / 工具。
    """
    _require_positive_int("top_k", top_k)
    embedding = CharNgramEmbedding(dimension=DIMENSION)
    store = FlatVectorStore(metric="cosine", dimension=DIMENSION)
    store.upsert(
        [
            make_record(
                record_id=record_id,
                vector=embedding.embed(text),
                text=text,
                metadata=dict(metadata),
            )
            for record_id, text, metadata in CORPUS
        ]
    )
    lexical = LexicalIndex.from_backend(store, params=BM25Params(k1=1.5, b=0.75))
    vector = build_retriever(store, embedding)
    hybrid = HybridRetriever(vector, lexical)
    # 一份 MockLLM 被规划与生成**共用**：脚本按调用顺序消费（先规划、再生成）。
    llm = MockLLM(responses=[PLAN_SCRIPT, ANSWER_SCRIPT])
    generator = RAGGenerator(llm, require_citation=True, model=PRICE_MODEL)
    return Substrate(
        question=question,
        gold=GOLD,
        grades=dict(GRADES),
        store=store,
        embedding=embedding,
        lexical=lexical,
        vector=vector,
        hybrid=hybrid,
        generator=generator,
        detector=PromptInjectionDetector(),
        moderator=ContentModerator(),
        planner=Planner(llm),
        calculator=CalculatorTool(),
        llm=llm,
    )


# --------------------------------------------------------------------------- #
# 九种统一读数（每一份都有一个可比较的 .reading 与一行 .line）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GuardReading:
    """护栏的一次读数：三类命中各自列出来，读数 = 命中总数（越少越好）."""

    injections: tuple[str, ...] = ()
    flagged_words: tuple[str, ...] = ()
    pii_types: tuple[str, ...] = ()
    sanitized: str = ""
    checked_chars: int = 0

    def __post_init__(self) -> None:
        if self.checked_chars < 0:
            raise ParameterError(f"checked_chars 不能为负，收到 {self.checked_chars}")

    @property
    def total(self) -> int:
        """三类命中的总数（护栏那一阶段的读数）."""
        return len(self.injections) + len(self.flagged_words) + len(self.pii_types)

    @property
    def safe(self) -> bool:
        """是否三类都没有命中（决定这次请求能不能往下走）."""
        return self.total == 0

    @property
    def reading(self) -> float:
        """护栏读数 = 命中总数（0 = 干净）."""
        return float(self.total)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "injections": list(self.injections),
            "flagged_words": list(self.flagged_words),
            "pii_types": list(self.pii_types),
            "checked_chars": self.checked_chars,
            "total": self.total,
            "safe": self.safe,
        }

    def line(self) -> str:
        """一行可读读数."""
        mark = "放行" if self.safe else "拦截"
        return (
            f"护栏：{mark} | 注入 {len(self.injections)} / 敏感 {len(self.flagged_words)}"
            f" / PII {len(self.pii_types)} | 检查 {self.checked_chars} 字"
        )


@dataclass(frozen=True)
class PlanReading:
    """规划的一次读数：子任务清单 + 工具自检结果，读数 = 子任务条数."""

    goal: str
    steps: tuple[str, ...]
    tool_expression: str
    tool_result: str
    tool_ok: bool

    @property
    def reading(self) -> float:
        """规划读数 = 子任务条数（本课写死 3 条）."""
        return float(len(self.steps))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "goal": self.goal,
            "steps": list(self.steps),
            "tool_expression": self.tool_expression,
            "tool_result": self.tool_result,
            "tool_ok": self.tool_ok,
        }

    def line(self) -> str:
        """一行可读读数."""
        mark = "✓" if self.tool_ok else "✗"
        return (
            f"规划：{len(self.steps)} 条子任务 {'、'.join(self.steps)} | "
            f"工具自检 {self.tool_expression} = {self.tool_result} {mark}"
        )


@dataclass(frozen=True)
class RetrievalReading:
    """检索的一次读数：名单 + 分数 + 空结果原因，读数 = 取回的条数."""

    ids: tuple[str, ...]
    scores: tuple[float, ...]
    empty_reason: str
    channel_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.scores):
            raise ParameterError(
                f"检索读数里 id 与分数条数不一致（{len(self.ids)} vs {len(self.scores)}）："
                "两者必须一一对应，否则'这一条多少分'会读成邻居的分。"
            )

    @property
    def reading(self) -> float:
        """检索读数 = 取回的命中条数."""
        return float(len(self.ids))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "ids": list(self.ids),
            "scores": [round(score, 6) for score in self.scores],
            "empty_reason": self.empty_reason,
            "channel_counts": dict(self.channel_counts),
        }

    def line(self) -> str:
        """一行可读读数."""
        pairs = "、".join(
            f"{rid}={score:+.4f}" for rid, score in zip(self.ids, self.scores)
        )
        return f"检索：{len(self.ids)} 条（{pairs or '（空）'}）| empty_reason={self.empty_reason}"


@dataclass(frozen=True)
class PackReading:
    """打包的一次读数：编号 + 占用，读数 = 进了上下文的引用条数."""

    citations: tuple[int, ...]
    used_ids: tuple[str, ...]
    char_count: int
    max_chars: int
    truncated_hits: int
    dropped_hits: int

    @property
    def reading(self) -> float:
        """打包读数 = 进了上下文的引用条数."""
        return float(len(self.citations))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "citations": list(self.citations),
            "used_ids": list(self.used_ids),
            "char_count": self.char_count,
            "max_chars": self.max_chars,
            "truncated_hits": self.truncated_hits,
            "dropped_hits": self.dropped_hits,
        }

    def line(self) -> str:
        """一行可读读数."""
        return (
            f"打包：{self.char_count}/{self.max_chars} 字 | 引用 {len(self.citations)} 条"
            f" | 截断 {self.truncated_hits} / 丢尾 {self.dropped_hits}"
        )


@dataclass(frozen=True)
class GenerationReading:
    """生成的一次读数：答案 + 有没有调模型，读数 = 调没调（1/0）."""

    answer: str
    llm_called: bool
    fallback_reason: str
    model: str

    @property
    def reading(self) -> float:
        """生成读数 = 这次调没调模型（1 = 调了，0 = 被护栏拦下）."""
        return 1.0 if self.llm_called else 0.0

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "answer": self.answer,
            "llm_called": self.llm_called,
            "fallback_reason": self.fallback_reason,
            "model": self.model,
        }

    def line(self) -> str:
        """一行可读读数."""
        state = "已调用" if self.llm_called else "未调用（被护栏拦下）"
        return f"生成：模型{state} | 回退 {self.fallback_reason or '（无）'} | {len(self.answer)} 字"


@dataclass(frozen=True)
class GroundReading:
    """接地的一次读数：三组编号，读数 = 幻觉引用条数（越少越好）."""

    valid: tuple[int, ...]
    invalid: tuple[int, ...]
    unused: tuple[int, ...]
    coverage: float
    grounded: bool

    @property
    def reading(self) -> float:
        """接地读数 = 幻觉引用条数（0 = 全部编号都能对上）."""
        return float(len(self.invalid))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "valid": list(self.valid),
            "invalid": list(self.invalid),
            "unused": list(self.unused),
            "coverage": round(self.coverage, 4),
            "grounded": self.grounded,
        }

    def line(self) -> str:
        """一行可读读数."""
        mark = "通过" if self.grounded else "未通过"
        return (
            f"接地：{mark} | 有效 {len(self.valid)} / 幻觉 {len(self.invalid)}"
            f" / 未引用 {len(self.unused)} | 覆盖 {self.coverage:.1%}"
        )


@dataclass(frozen=True)
class EvalReading:
    """评估的一次读数：四条指标，读数 = 召回（**它是一条下界**）."""

    recall: float
    precision: float
    mrr: float
    ndcg: float
    gold: tuple[str, ...]
    retrieved: tuple[str, ...]

    def __post_init__(self) -> None:
        for label in ("recall", "precision", "mrr", "ndcg"):
            value = getattr(self, label)
            if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
                raise ParameterError(f"评估指标 {label} 必须有限，收到 {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ParameterError(
                    f"评估指标 {label}={value!r} 必须落在 [0, 1]："
                    "越界说明分子或分母算错了（例如把幻觉引用也算进了命中）。"
                )

    @property
    def reading(self) -> float:
        """评估读数 = 召回（性质里以"下界"判据检查它）."""
        return self.recall

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "mrr": round(self.mrr, 6),
            "ndcg": round(self.ndcg, 6),
            "gold": list(self.gold),
            "retrieved": list(self.retrieved),
        }

    def line(self) -> str:
        """一行可读读数."""
        return (
            f"评估：召回 {self.recall:.4f} | 精确 {self.precision:.4f}"
            f" | MRR {self.mrr:.4f} | NDCG {self.ndcg:.4f} | 金标准 {list(self.gold)}"
        )


@dataclass(frozen=True)
class CostReading:
    """记账的一次读数：token 与费用，读数 = 总费用（美元）."""

    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 之和."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def reading(self) -> float:
        """记账读数 = 总费用（美元）."""
        return self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "model": self.model,
        }

    def line(self) -> str:
        """一行可读读数."""
        return (
            f"记账：{self.calls} 次调用 | token {self.total_tokens}"
            f"（{self.prompt_tokens}+{self.completion_tokens}）"
            f" | 费用 ${self.cost_usd:.8f}（{self.model}）"
        )


@dataclass(frozen=True)
class TraceReading:
    """追踪的一次读数：span 条数与名字，读数 = span 条数."""

    span_count: int
    span_names: tuple[str, ...]
    root_name: str

    def __post_init__(self) -> None:
        if self.span_count < 0:
            raise ParameterError(f"span_count 不能为负，收到 {self.span_count}：它是报告里的读数。")
        if self.span_count and not self.span_names:
            raise ParameterError(
                "span_count > 0 时 span_names 不能为空：条数与名字必须一起给出，"
                "否则'drive 了几段'与'分别是哪几段'这两件事只剩一件。"
            )

    @property
    def reading(self) -> float:
        """追踪读数 = span 条数（root + 每个阶段一个）."""
        return float(self.span_count)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "span_count": self.span_count,
            "span_names": list(self.span_names),
            "root_name": self.root_name,
        }

    def line(self) -> str:
        """一行可读读数."""
        return f"追踪：{self.span_count} 个 span（root={self.root_name}）"


# --------------------------------------------------------------------------- #
# 适配函数：真实子系统 → 统一读数
# --------------------------------------------------------------------------- #


def guard_input(substrate: Substrate, text: str) -> GuardReading:
    """第 1 阶段：调两个既有护栏，把结果折成一份 :class:`GuardReading`."""
    report = substrate.detector.scan(text)
    moderated = substrate.moderator.moderate(text)
    return GuardReading(
        injections=tuple(report.matched_patterns),
        flagged_words=tuple(moderated.flagged_words),
        pii_types=tuple(moderated.pii_types),
        sanitized=moderated.sanitized_text,
        checked_chars=len(text),
    )


def plan_goal(substrate: Substrate, goal: str) -> PlanReading:
    """第 2 阶段：调 ``Planner`` 拆解目标，并跑一次**确定性的工具自检**."""
    plan = substrate.planner.plan(goal)
    tool_result = substrate.calculator.execute(TOOL_EXPRESSION)
    return PlanReading(
        goal=plan.goal,
        steps=tuple(plan.steps),
        tool_expression=TOOL_EXPRESSION,
        tool_result=tool_result,
        tool_ok=tool_result == TOOL_EXPECTED,
    )


def retrieve(
    substrate: Substrate,
    *,
    question: str | None = None,
    top_k: int = TOP_K,
) -> tuple[RetrievalResult, RetrievalReading]:
    """第 3 阶段：调混合检索器，返回**原始结果**与它的一份读数（打包要用原始结果）."""
    _require_positive_int("top_k", top_k)
    text = substrate.question if question is None else question
    result = substrate.hybrid.retrieve(RetrievalQuery(text=text, top_k=top_k))
    hits = result.hits
    reading = RetrievalReading(
        ids=tuple(hit.record_id for hit in hits),
        scores=tuple(float(hit.score) for hit in hits),
        empty_reason=result.empty_reason,
        channel_counts={name: int(count) for name, count in sorted(result.channel_candidates.items())},
    )
    return result, reading


def pack_hits(
    result: RetrievalResult,
    *,
    max_chars: int = MAX_CHARS,
    per_hit_chars: int = PER_HIT_CHARS,
) -> tuple[PackedContext, PackReading]:
    """第 4 阶段：调 ``pack_context``，返回**原始上下文**与它的一份读数."""
    _require_positive_int("max_chars", max_chars)
    _require_positive_int("per_hit_chars", per_hit_chars)
    packed = pack_context(result.hits, max_chars=max_chars, per_hit_chars=per_hit_chars)
    reading = PackReading(
        citations=tuple(citation.marker for citation in packed.citations),
        used_ids=tuple(hit.record_id for hit in packed.used_hits),
        char_count=packed.char_count,
        max_chars=packed.max_chars,
        truncated_hits=packed.truncated_hits,
        dropped_hits=len(packed.dropped_hits),
    )
    return packed, reading


def generate(
    substrate: Substrate,
    packed: PackedContext,
    *,
    question: str | None = None,
) -> Generation:
    """第 5 阶段：调 ``RAGGenerator`` 生成一条**可核对的**答案."""
    text = substrate.question if question is None else question
    return substrate.generator.generate(text, packed)


def generation_reading(generation: Generation) -> GenerationReading:
    """把 ``Generation`` 折成 :class:`GenerationReading`."""
    return GenerationReading(
        answer=generation.answer,
        llm_called=bool(generation.llm_called),
        fallback_reason=generation.fallback_reason,
        model=generation.model,
    )


def ground_reading(generation: Generation) -> GroundReading:
    """第 6 阶段：把 ``GroundingReport`` 折成 :class:`GroundReading`."""
    check = generation.check
    return GroundReading(
        valid=tuple(int(marker) for marker in check.valid),
        invalid=tuple(int(marker) for marker in check.invalid),
        unused=tuple(int(marker) for marker in check.unused),
        coverage=float(check.coverage),
        grounded=bool(check.grounded),
    )


def evaluate(reading: RetrievalReading, *, gold: tuple[str, ...], grades: dict[str, float]) -> EvalReading:
    """第 7 阶段：调 ``evaluation.rag_metrics`` 的四条纯函数，现场算四条指标."""
    retrieved = list(reading.ids)
    relevant = list(gold)
    return EvalReading(
        recall=retrieval_recall(retrieved, relevant),
        precision=retrieval_precision(retrieved, relevant),
        mrr=mrr(retrieved, relevant),
        ndcg=ndcg(retrieved, dict(grades)),
        gold=tuple(gold),
        retrieved=tuple(retrieved),
    )


def account(llm: MockLLM, *, model: str = PRICE_MODEL) -> CostReading:
    """第 8 阶段：用 ``CostTracker`` 消费这一次底座上**全部**的模型调用."""
    tracker = CostTracker()
    calls = tracker.record_from_llm(llm, model=model)
    breakdown = tracker.breakdown("model").get(model, {})
    return CostReading(
        calls=calls,
        prompt_tokens=int(breakdown.get("prompt_tokens", 0)),
        completion_tokens=int(breakdown.get("completion_tokens", 0)),
        # 用未经四舍五入的 total_cost（report() 里的 total_cost_usd 被 round 到 6 位，
        # 那会让"与手算公式对账"这条性质在 3.8e-05 这种量级上误报失败——见 verify 第 ⑥ 条）。
        cost_usd=float(tracker.total_cost),
        model=model,
    )


def trace_reading(tracer: Tracer) -> TraceReading:
    """第 9 阶段：把一次 trace 的 span 折成 :class:`TraceReading`（只数条数，不带时间）."""
    spans = list(tracer.spans)
    # 根 span 靠 ``parent_id is None`` 认出来：它**最后落盘**（子 span 先结束先写），
    # 因此不能拿 ``spans[0]`` 当根——那会拿到第一个子 span。
    root = next((span for span in spans if span.parent_id is None), None)
    return TraceReading(
        span_count=len(spans),
        span_names=tuple(span.name for span in spans),
        root_name=root.name if root is not None else "",
    )


def hand_cost_formula(prompt_tokens: int, completion_tokens: int, *, model: str = PRICE_MODEL) -> float:
    """手算费用公式（**与 CostTracker 各写一遍**，用于跨实现对账）:

    ``cost = (prompt × input_price + completion × output_price) / 1000``.
    """
    if model not in DEFAULT_PRICE_TABLE:
        raise ParameterError(f"价格表里没有模型 {model!r}：可选 {sorted(DEFAULT_PRICE_TABLE)}。")
    price = DEFAULT_PRICE_TABLE[model]
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1000.0


def tool_self_check(substrate: Substrate) -> str:
    """跑一次工具自检并返回结果字符串（报告里印它，证明工具链真的可调用）."""
    return substrate.calculator.execute(TOOL_EXPRESSION)


def message_count(llm: MockLLM) -> int:
    """这次底座上模型**被调用**了几次（规划 + 生成 = 2）."""
    return len(llm.calls)


def messages_of(llm: MockLLM) -> tuple[Message, ...]:
    """把 ``MockLLM.calls`` 摊平成一串消息（只为报告里数一数条数）."""
    return tuple(message for batch in llm.calls for message in batch)


__all__ = [
    "ADAPTER_BOUNDARY",
    "ANSWER_SCRIPT",
    "CORPUS",
    "DIMENSION",
    "GOLD",
    "GRADES",
    "MAX_CHARS",
    "PER_HIT_CHARS",
    "PLAN_SCRIPT",
    "PRICE_MODEL",
    "QUESTION",
    "TOOL_EXPECTED",
    "TOOL_EXPRESSION",
    "TOP_K",
    "CostReading",
    "EvalReading",
    "GenerationReading",
    "GroundReading",
    "GuardReading",
    "PackReading",
    "PlanReading",
    "RetrievalReading",
    "Substrate",
    "TraceReading",
    "account",
    "build_substrate",
    "evaluate",
    "generate",
    "generation_reading",
    "ground_reading",
    "guard_input",
    "hand_cost_formula",
    "message_count",
    "messages_of",
    "pack_hits",
    "plan_goal",
    "retrieve",
    "tool_self_check",
    "trace_reading",
]
