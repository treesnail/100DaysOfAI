"""``assembly``：把八项能力**真实装配**成一条端到端可复算的调用链（day099 / G1-D1）.

day088 把底层原理拼成一张**可校验的图**；今天把已经各自成立的能力装成一条
**能一次跑完的链**——九个阶段，每一步都调一个**既有子系统**，并留下一份读数：

```text
guard     security.injection_detector / security.content_moderator
plan      agent.planner.Planner（+ tools.calculator.CalculatorTool 自检）
retrieve  retrieval.hybrid.HybridRetriever（build_retriever + LexicalIndex）
pack      retrieval.context.pack_context
generate  retrieval.generation.RAGGenerator（配 llm.mock.MockLLM）
ground    retrieval.generation.GroundingReport（同一份 Generation 的 check）
evaluate  evaluation.rag_metrics（recall / precision / mrr / ndcg）
account   observability.cost_tracker.CostTracker
trace     observability.tracing.Tracer
```

## 一、今天最值钱的一句话

> **"接上了"的定义是：同一输入跑两次，九个阶段的读数**逐位相同**——
> 否则后面所有的评估、对账与回归都建立在一堆会漂移的数上。**

因此 :class:`SystemRun` 带一个 :meth:`SystemRun.comparable`（只含确定性字段的元组）
与 :meth:`SystemRun.diff_count`（两次运行差了几项）。
:mod:`capstone.verify` 的第 ③ 条性质就是"diff_count == 0"。

## 二、一条纪律：两次运行必须**互不干扰**

底座（:class:`~capstone.adapters.Substrate`）里的 ``MockLLM`` 会**消费脚本**
——跑过一次之后它的 ``responses`` 就空了。因此:

```text
run() 每次**新建一份底座**（build_substrate）⇒ 两次运行的脚本从同一条起跑线出发
```

这不是"为了测试才这样"：真实的结业链路里，"这一次请求"本来就不该复用一个
被上一次请求改过状态的模型对象——那正是"两次运行读数不同"最常见的病因。

## 三、为什么读数里**没有时间戳**

trace 的 span 自带 ``uuid`` 与 ``time.time()``，它们**逐次都不同**。因此
:class:`~capstone.adapters.TraceReading` 只数 **span 条数**与名字，
不碰时间与 uuid。把耗时写进读数会让第 ③ 条性质永远失败——
而"耗时不该进复算口径"是本课刻意留下的一条边界（见 types 的边界表）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from smart_research_agent.capstone import adapters
from smart_research_agent.capstone.errors import AssemblyError, NumericError, ParameterError, StageError
from smart_research_agent.capstone.types import (
    ASSEMBLY_STAGES,
    STAGE_ACCOUNT,
    STAGE_CAPABILITIES,
    STAGE_EVALUATE,
    STAGE_GENERATE,
    STAGE_GROUND,
    STAGE_GUARD,
    STAGE_PACK,
    STAGE_PLAN,
    STAGE_RETRIEVE,
    STAGE_SPECS,
    STAGE_TRACE,
)

#: trace 根的 span 名字（报告里回显它）.
TRACE_ROOT_NAME = "capstone.run"

#: 每个阶段的 span 名字的前缀（``stage.guard`` 这样）.
SPAN_PREFIX = "stage."

#: 本课期望的 span 条数 = 根 1 个 + 每个**真的执行了**的阶段 1 个.
#:
#: 九个阶段里有八个在 trace 上下文内执行（trace 自己那一段是"读 trace"，
#: 它在 root 落盘之后才算得出来），因此期望值恰好等于九个阶段的下标数。
EXPECTED_SPANS = len(ASSEMBLY_STAGES)

#: 复算口径里保留的小数位（浮点在两次运行间逐位相同，这里只是去掉尾零噪声）.
DIGEST_FLOAT_DIGITS = 12

#: 摘要取前多少位十六进制（够短、够可读）.
DIGEST_LENGTH = 16


@dataclass(frozen=True)
class StageRecord:
    """一个阶段的一次记录：阶段名 + 承担能力 + 成没成 + 读数 + 一行可读证据.

    ``ok`` 与 ``reading`` 分开而不是合成一个字段：前者回答"这一步算不算成"，
    后者回答"它读出了多少"——合成之后，"跑完了但读数是 0"与"没跑"会长得一样。
    """

    stage: str
    capability: str
    ok: bool
    reading: float
    detail: str

    def __post_init__(self) -> None:
        if self.stage not in STAGE_SPECS:
            raise StageError(f"未知的阶段 {self.stage!r}：可选 {list(ASSEMBLY_STAGES)}。")
        expected = STAGE_CAPABILITIES[self.stage]
        if self.capability != expected:
            raise StageError(
                f"阶段 {self.stage!r} 的承担能力应当是 {expected!r}，收到 {self.capability!r}："
                "阶段与能力的映射只有一处定义（STAGE_CAPABILITIES），两处各写一遍迟早会分家。"
            )
        if self.reading != self.reading or self.reading in (float("inf"), float("-inf")):
            raise NumericError(
                f"阶段 {self.stage!r} 的读数必须有限，收到 {self.reading!r}："
                "非有限数会让任何比较静默为假。"
            )
        if not self.detail:
            raise ParameterError(f"阶段 {self.stage!r} 必须留一行可读证据（detail 不能为空）。")

    @property
    def index(self) -> int:
        """这一阶段在链里的序号（1 起）."""
        return STAGE_SPECS[self.stage].index

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "stage": self.stage,
            "index": self.index,
            "capability": self.capability,
            "ok": self.ok,
            "reading": round(self.reading, DIGEST_FLOAT_DIGITS),
            "detail": self.detail,
        }

    def line(self) -> str:
        """一行读数：``1. [guard] ✓ | 读数 0 | 护栏：……``."""
        mark = "✓" if self.ok else "✗"
        return f"{self.index}. [{self.stage:<8}] {mark} | 读数 {self.reading:.6g} | {self.detail}"


@dataclass(frozen=True)
class SystemRun:
    """一次端到端运行的结果：九个阶段记录 + 关键指标 + 成本 + trace 条数.

    ``comparable`` 只含**确定性字段**（无时间戳、无 uuid、无对象引用），
    因此"同一输入两次运行逐位相同"可以被一条元组比较直接判定。
    """

    question: str
    records: tuple[StageRecord, ...]
    answer: str
    recall: float
    precision: float
    mrr: float
    ndcg: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    spans: int

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ParameterError("SystemRun.question 不能为空：一份不知在回答什么问题的运行无法复核。")
        if len(self.records) != len(ASSEMBLY_STAGES):
            raise StageError(
                f"运行记录有 {len(self.records)} 段，应当是 {len(ASSEMBLY_STAGES)} 段："
                "段数与 ASSEMBLY_STAGES 对不上，报告里的阶段表就会缺行或多行。"
            )
        actual = tuple(record.stage for record in self.records)
        if actual != ASSEMBLY_STAGES:
            raise StageError(
                f"运行的阶段序列 {list(actual)} 与 ASSEMBLY_STAGES {list(ASSEMBLY_STAGES)} 逐位不同："
                "次序是链的形状；形状错了，后面所有读数都指错了地方。"
            )
        for label in ("recall", "precision", "mrr", "ndcg"):
            value = getattr(self, label)
            if value != value or value in (float("inf"), float("-inf")):
                raise NumericError(f"SystemRun.{label} 必须有限，收到 {value!r}。")
            if not 0.0 <= value <= 1.0:
                raise NumericError(f"SystemRun.{label}={value!r} 必须落在 [0, 1]。")
        if self.cost_usd < 0.0:
            raise NumericError(f"SystemRun.cost_usd 不能为负，收到 {self.cost_usd!r}。")
        if self.prompt_tokens < 0 or self.completion_tokens < 0 or self.spans < 1:
            raise ParameterError(
                f"计数字段非法：prompt={self.prompt_tokens}、completion={self.completion_tokens}、"
                f"spans={self.spans}。"
            )

    # ------------------------------------------------------------------ 只读视图

    @property
    def stages(self) -> tuple[str, ...]:
        """运行的阶段序列（应等于 ASSEMBLY_STAGES）."""
        return tuple(record.stage for record in self.records)

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 之和."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        """九个阶段是否全部为 :attr:`StageRecord.ok`."""
        return all(record.ok for record in self.records)

    def record_of(self, stage: str) -> StageRecord:
        """取某一阶段的记录（未知阶段当场拒绝）."""
        for record in self.records:
            if record.stage == stage:
                return record
        raise StageError(f"运行记录里没有阶段 {stage!r}。")

    def reading_of(self, stage: str) -> float:
        """取某一阶段的读数."""
        return self.record_of(stage).reading

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两次运行逐位比较它）.

        刻意不包含：时间戳、uuid、对象引用、耗时、日志文本。
        包含它们的后果是"两次运行必然不同"——而那是口径问题，不是实现问题。
        """

        def round_(value: float) -> float:
            """把浮点收敛到复算口径的小数位（去掉尾零噪声）."""
            return round(float(value), DIGEST_FLOAT_DIGITS)

        return (
            ("question", self.question),
            (
                "records",
                tuple(
                    (record.stage, record.ok, round_(record.reading), record.detail)
                    for record in self.records
                ),
            ),
            ("answer", self.answer),
            ("recall", round_(self.recall)),
            ("precision", round_(self.precision)),
            ("mrr", round_(self.mrr)),
            ("ndcg", round_(self.ndcg)),
            ("cost_usd", round_(self.cost_usd)),
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("spans", self.spans),
        )

    def digest(self) -> str:
        """复算口径的摘要（两次运行摘要相同 ⇔ 逐位相同）."""
        return hashlib.sha256(repr(self.comparable()).encode("utf-8")).hexdigest()[:DIGEST_LENGTH]

    def diff_count(self, other: SystemRun) -> int:
        """与另一次运行在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def is_identical_to(self, other: SystemRun) -> bool:
        """是否与另一次运行逐位相同."""
        return self.comparable() == other.comparable()

    # ------------------------------------------------------------------ 打印

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "question": self.question,
            "digest": self.digest(),
            "ok": self.ok,
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "mrr": round(self.mrr, 6),
            "ndcg": round(self.ndcg, 6),
            "cost_usd": round(self.cost_usd, 8),
            "total_tokens": self.total_tokens,
            "spans": self.spans,
            "records": [record.to_dict() for record in self.records],
        }

    def lines(self) -> tuple[str, ...]:
        """逐行文本（九个阶段 + 一行汇总）."""
        lines = [record.line() for record in self.records]
        lines.append(
            f"汇总：阶段 {len(self.records)} 段全通过={self.ok} | 召回 {self.recall:.4f}"
            f" | 费用 ${self.cost_usd:.8f} | token {self.total_tokens} | span {self.spans}"
            f" | 摘要 {self.digest()}"
        )
        return tuple(lines)


class SystemAssembly:
    """把一份固定底座装成一条端到端链的装配器（``run`` 是唯一入口）.

    它只持有一份 :class:`~capstone.adapters.Substrate` 与三个预算
    （``top_k`` / ``max_chars`` / ``per_hit_chars``）。**每次 run 之前请新建一份底座**
    ——见模块 docstring 第二节：复用一份被跑过的底座会让第二次运行的脚本空掉。
    本类的 :meth:`run` 会在需要时**自己新建**底座（``substrate`` 传 ``None`` 时）。
    """

    def __init__(
        self,
        substrate: adapters.Substrate | None = None,
        *,
        top_k: int = adapters.TOP_K,
        max_chars: int = adapters.MAX_CHARS,
        per_hit_chars: int = adapters.PER_HIT_CHARS,
    ) -> None:
        self._substrate = adapters.build_substrate() if substrate is None else substrate
        self._top_k = adapters._require_positive_int("top_k", top_k)
        self._max_chars = adapters._require_positive_int("max_chars", max_chars)
        self._per_hit_chars = adapters._require_positive_int("per_hit_chars", per_hit_chars)

    @property
    def substrate(self) -> adapters.Substrate:
        """这次装配用的底座（**可变**：它的 MockLLM 会消费脚本）."""
        return self._substrate

    def describe(self) -> dict[str, Any]:
        """装配器与底座的配置自述（报告里回显它）."""
        return {
            "top_k": self._top_k,
            "max_chars": self._max_chars,
            "per_hit_chars": self._per_hit_chars,
            "substrate": self._substrate.describe(),
        }

    def run(self, question: str | None = None) -> SystemRun:
        """跑一次端到端链路，返回一份 :class:`SystemRun`（九段，次序固定）.

        每一次 run 都用 :meth:`trace_one` 把八个**真的执行了**的阶段各记一个 span，
        因此报告里的 trace 阶段读数是一个被真实记下来的条数，而不是写死的 9。
        """
        substrate = self._substrate
        resolved_question = substrate.question if question is None else question
        records: list[StageRecord] = []
        tracer = adapters.Tracer()
        with tracer.start_trace(TRACE_ROOT_NAME, question=resolved_question) as ctx:
            with ctx.span(SPAN_PREFIX + STAGE_GUARD) as span:
                guard = adapters.guard_input(substrate, resolved_question)
                span.attributes["reading"] = guard.reading
            records.append(
                StageRecord(
                    stage=STAGE_GUARD,
                    capability=STAGE_CAPABILITIES[STAGE_GUARD],
                    ok=guard.safe,
                    reading=guard.reading,
                    detail=guard.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_PLAN) as span:
                plan = adapters.plan_goal(substrate, resolved_question)
                span.attributes["reading"] = plan.reading
            records.append(
                StageRecord(
                    stage=STAGE_PLAN,
                    capability=STAGE_CAPABILITIES[STAGE_PLAN],
                    ok=bool(plan.steps) and plan.tool_ok,
                    reading=plan.reading,
                    detail=plan.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_RETRIEVE) as span:
                result, retrieval = adapters.retrieve(
                    substrate, question=resolved_question, top_k=self._top_k
                )
                span.attributes["reading"] = retrieval.reading
            records.append(
                StageRecord(
                    stage=STAGE_RETRIEVE,
                    capability=STAGE_CAPABILITIES[STAGE_RETRIEVE],
                    ok=retrieval.reading > 0.0,
                    reading=retrieval.reading,
                    detail=retrieval.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_PACK) as span:
                packed, pack = adapters.pack_hits(
                    result, max_chars=self._max_chars, per_hit_chars=self._per_hit_chars
                )
                span.attributes["reading"] = pack.reading
            records.append(
                StageRecord(
                    stage=STAGE_PACK,
                    capability=STAGE_CAPABILITIES[STAGE_PACK],
                    ok=pack.reading > 0.0,
                    reading=pack.reading,
                    detail=pack.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_GENERATE) as span:
                generation = adapters.generate(substrate, packed, question=resolved_question)
                generated = adapters.generation_reading(generation)
                span.attributes["reading"] = generated.reading
            records.append(
                StageRecord(
                    stage=STAGE_GENERATE,
                    capability=STAGE_CAPABILITIES[STAGE_GENERATE],
                    ok=generated.llm_called,
                    reading=generated.reading,
                    detail=generated.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_GROUND) as span:
                ground = adapters.ground_reading(generation)
                span.attributes["reading"] = ground.reading
            records.append(
                StageRecord(
                    stage=STAGE_GROUND,
                    capability=STAGE_CAPABILITIES[STAGE_GROUND],
                    ok=ground.reading == 0.0,
                    reading=ground.reading,
                    detail=ground.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_EVALUATE) as span:
                evaluated = adapters.evaluate(retrieval, gold=substrate.gold, grades=substrate.grades)
                span.attributes["reading"] = evaluated.reading
            records.append(
                StageRecord(
                    stage=STAGE_EVALUATE,
                    capability=STAGE_CAPABILITIES[STAGE_EVALUATE],
                    ok=evaluated.reading >= 1.0,
                    reading=evaluated.reading,
                    detail=evaluated.line(),
                )
            )
            with ctx.span(SPAN_PREFIX + STAGE_ACCOUNT) as span:
                cost = adapters.account(substrate.llm)
                span.attributes["reading"] = cost.reading
            records.append(
                StageRecord(
                    stage=STAGE_ACCOUNT,
                    capability=STAGE_CAPABILITIES[STAGE_ACCOUNT],
                    ok=cost.calls > 0,
                    reading=cost.reading,
                    detail=cost.line(),
                )
            )
        # trace 阶段在 root 落盘之后才算得出来（见 EXPECTED_SPANS 的说明）。
        traced = adapters.trace_reading(tracer)
        records.append(
            StageRecord(
                stage=STAGE_TRACE,
                capability=STAGE_CAPABILITIES[STAGE_TRACE],
                ok=traced.span_count == EXPECTED_SPANS,
                reading=traced.reading,
                detail=traced.line(),
            )
        )
        return SystemRun(
            question=resolved_question,
            records=tuple(records),
            answer=generation.answer,
            recall=evaluated.recall,
            precision=evaluated.precision,
            mrr=evaluated.mrr,
            ndcg=evaluated.ndcg,
            cost_usd=cost.cost_usd,
            prompt_tokens=cost.prompt_tokens,
            completion_tokens=cost.completion_tokens,
            spans=traced.span_count,
        )


def run(question: str | None = None) -> SystemRun:
    """跑一次端到端链路（**每次新建底座**，因此两次调用互不干扰且逐位相同）."""
    return SystemAssembly().run(question)


def run_lines(run_result: SystemRun) -> tuple[str, ...]:
    """把一次运行逐行印出来（:mod:`capstone.study` 与演示脚本共用）."""
    return run_result.lines()


def require_ok(run_result: SystemRun) -> SystemRun:
    """运行的九段没有全部通过时抛 :class:`AssemblyError`（"拒绝交付"的那条路）."""
    if run_result.ok:
        return run_result
    failures = [record.line() for record in run_result.records if not record.ok]
    raise AssemblyError("端到端链有阶段没通过：" + "；".join(failures))


__all__ = [
    "DIGEST_FLOAT_DIGITS",
    "DIGEST_LENGTH",
    "EXPECTED_SPANS",
    "SPAN_PREFIX",
    "TRACE_ROOT_NAME",
    "StageRecord",
    "SystemAssembly",
    "SystemRun",
    "require_ok",
    "run",
    "run_lines",
]
