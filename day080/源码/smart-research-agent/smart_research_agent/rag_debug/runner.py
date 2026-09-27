"""运行器：把一条 ``RagEvalCase`` 真的跑一遍，交出 ``CaseOutcome``（day071）.

```text
RagEvalCase  →  RagPipeline.answer()  →  RagAnswer  →  CaseOutcome
                （检索 → 打包 → 生成 → 核对）
```

本模块只做一件容易做错的事：**把 ``RagAnswer`` 上的字段原样搬进一份扁平、
可序列化、可比较的账**。搬的过程有两个容易出错的地方，它们也是本模块存在
（而不是把这段代码抄进报告里）的理由：

```text
两段名单必须来自同一次回答   retrieved = answer.retrieval.ids
                            packed    = [c.record_id for c in answer.context.citations]
                            —— 若"检索"与"打包"分别跑一次，packed_away 这个指标
                            就会把两次运行的随机性记成一次预算损失
四个指标的两段要同源        measure_ids 复用 retrieval.rerank / evaluation.rag_metrics
                            的纯函数；本层不重写任何一个指标（重写 = 两个口径）
```

## 计时为什么由本层给

``RagAnswer`` 上有 ``retrieval.latency_ms``（检索那一段）与
``Generation.latency_ms``（生成那一段），但**没有"整条链路"那个数**，
而评估要的恰恰是端到端的延迟（用户等的是整个答案）。因此本层用注入的
``clock`` 在 ``answer()`` 外面包一层——**采样点只有一处**，而不是把两个
阶段耗时相加：相加会漏掉打包、序列化与两次调用之间的那段真实时间。

## 忠实度为什么是可选的

``judge`` 为 ``None`` 时一次模型都不多调（评测的成本由调用方决定）；
给了一个 ``BaseLLM`` 时，本层调用的是 **day029 那份**答案对比评估
（``evaluation.rag_eval`` 的提示词与解析），而不是另写一份——"核对只有
一份实现"这条纪律在 day071 的新场景里仍然成立。

评审失败（输出不是合法 JSON、分数越界）时**降级而不抛异常**：评审是
可观测性设施，它的故障不配打断一次已经跑完的评估；降级的事实写进
``CaseOutcome.notes``，于是"这一批的忠实度均值"里不会多出一个看不见的空洞。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from smart_research_agent.evaluation.rag_eval import (
    FAITHFULNESS_PROMPT,
    FaithfulnessParseError,
    FaithfulnessResult,
    parse_faithfulness,
)
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.rag_debug.errors import DiagnosisError
from smart_research_agent.rag_debug.types import (
    DEFAULT_MIN_COVERAGE,
    STAGE_PACKED,
    STAGE_RETRIEVED,
    CaseOutcome,
    RagEvalCase,
    StageMetrics,
    measure_ids,
)
from smart_research_agent.retrieval.pipeline import RagAnswer, RagPipeline

#: 缺省的 k（四个检索指标共用的那个 k）。与 ``retrieval_top_k`` 的缺省值无关——
#: 评估要的是"前 k 条里有没有金标准"，k 由评估方决定（见 ``evaluate`` 的 k 参数）。
DEFAULT_EVAL_K = 3


def score_faithfulness(
    answer: str, reference: str, llm: BaseLLM
) -> FaithfulnessResult:
    """用 LLM-as-a-judge 评一次忠实度（提示词与解析都来自 ``evaluation.rag_eval``）.

    LLM 在这里是一个**评分函数**：提示词锁定 JSON 输出，解析端用与
    ``parse_plan`` 相同的防御式策略逐层校验（定位 ``{}`` 边界 → JSON →
    结构 → 值域）。本函数一行都不自己写解析——两份解析一定会分家，
    而分家的表现是"同一段评审输出在旧报告里是 1.0、在新报告里是报错"。
    """
    raw = llm.chat(
        [
            Message(
                role="user",
                content=FAITHFULNESS_PROMPT.format(reference=reference, answer=answer),
            )
        ]
    )
    return parse_faithfulness(raw)


class RagEvalRunner:
    """在一条 ``RagPipeline`` 上跑评测集，产出 ``CaseOutcome``（逐条、顺序与入参一致）.

    构造参数分三组，**每组各自决定一件可以被单独讨论的事**：

```text
度量类   k                  "前几条算命中"（两段名单共用同一个 k）
判据类   min_coverage       "覆盖率到什么程度算低"（归因用，本层只透传）
                             它与 min_reciprocal_rank / min_faithfulness 一样，
                             都不在本层消费——归因是 diagnose 的事
评审类   judge              "评不评忠实度"（None = 不评，一次模型都不多调）
计时类   clock              "怎么读时间"（测试注入一个假时钟 → 结果逐位可复现）
```

    ``clock`` 之所以要能注入：延迟是报告里唯一的**非确定性**来源，
    而一个"同样的输入跑两次得到不同报告"的评估设施无法进 CI
    （与 ``perf_baseline.measure`` 把耗时交给流水线自报是同一条纪律）。
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        *,
        k: int = DEFAULT_EVAL_K,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        require_grounded: bool = True,
        judge: BaseLLM | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(pipeline, RagPipeline):
            raise DiagnosisError(
                f"pipeline 必须是 RagPipeline，收到 {type(pipeline).__name__}："
                "评估要跑的是**整条链路**（检索 → 打包 → 生成 → 核对），"
                "只给一个检索器会得到一份'没有答案可核对'的报告。"
            )
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise DiagnosisError(
                f"k 必须是 >= 1 的整数，收到 {k!r}："
                "四个检索指标都带着这个 k（recall@k / precision@k / nDCG@k）。"
            )
        if (
            isinstance(min_coverage, bool)
            or not isinstance(min_coverage, (int, float))
            or not 0.0 <= float(min_coverage) <= 1.0
        ):
            raise DiagnosisError(
                f"min_coverage 必须落在 [0, 1]，收到 {min_coverage!r}"
            )
        if not isinstance(require_grounded, bool):
            raise DiagnosisError(
                f"require_grounded 必须是布尔值，收到 {type(require_grounded).__name__}"
            )
        if judge is not None and not isinstance(judge, BaseLLM):
            raise DiagnosisError(
                f"judge 必须是 BaseLLM 或 None，收到 {type(judge).__name__}："
                "None 表示'这次不评忠实度'（一次模型都不多调），"
                "给了别的形状只会在第一条用例上炸。"
            )
        self._pipeline = pipeline
        self._k = k
        self.min_coverage = float(min_coverage)
        self.require_grounded = require_grounded
        self._judge = judge
        self._clock = clock or time.perf_counter

    # ------------------------------------------------------------------ 只读视图

    @property
    def pipeline(self) -> RagPipeline:
        """本次评估跑的那条链路（报告里的 ``prompt_version`` 来自它）."""
        return self._pipeline

    @property
    def k(self) -> int:
        """四个检索指标共用的那个 k."""
        return self._k

    @property
    def judge(self) -> BaseLLM | None:
        """评审模型（``None`` = 不评忠实度）."""
        return self._judge

    def describe(self) -> dict[str, object]:
        """这一层的配置（端点与演示脚本直接展示它）.

        ``judge`` 只报**有/没有**，不报模型名：本层不假设 ``BaseLLM`` 有名字
        （``MockLLM`` 就没有），把一个拿不到的东西写进报告只会在运行期炸。
        """
        return {
            "k": self._k,
            "min_coverage": self.min_coverage,
            "require_grounded": self.require_grounded,
            "judge": self._judge is not None,
            "prompt_version": self._pipeline.prompt_version,
        }

    # ------------------------------------------------------------------ 运行入口

    def run_case(self, case: RagEvalCase) -> CaseOutcome:
        """跑一条用例：调一次 ``answer``，把 ``RagAnswer`` 搬成一份账.

        四个搬运动作各自对应一处真实存在的形状：

```text
retrieved_ids   answer.retrieval.ids          （名单 + 次序，重排之后的）
packed_ids      answer.context.citations      （**进了提示词**的那些编号对应的记录）
空结果原因       answer.retrieval.empty_reason  （空时非空串，封闭清单）
生成侧四个数     answer.check                 （None = 没走到生成那一步）
```
        """
        if not isinstance(case, RagEvalCase):
            raise DiagnosisError(
                f"case 必须是 RagEvalCase，收到 {type(case).__name__}"
            )
        started = self._clock()
        answer = self._pipeline.answer(case.query)
        elapsed_ms = (self._clock() - started) * 1000.0
        return self.to_outcome(case, answer, elapsed_ms=elapsed_ms)

    def to_outcome(
        self, case: RagEvalCase, answer: RagAnswer, *, elapsed_ms: float
    ) -> CaseOutcome:
        """把一次回答搬成一份账（**与运行解耦**，因此可以拿历史回答重放）.

        "重放一份历史回答"这件事在排查时真实存在：``RagAnswer`` 是可以被
        ``to_dict()`` 落盘的，于是"昨天那次评估为什么判它没接地"可以在今天
        逐字段复核，而不必重跑一次（重跑一次会拿到另一段模型输出）。
        """
        retrieved_ids = tuple(answer.retrieval.ids() if answer.retrieval else ())
        packed_ids = tuple(
            citation.record_id for citation in (answer.context.citations if answer.context else ())
        )
        check = answer.check
        notes: list[str] = []
        faithfulness = self._faithfulness(answer, case, notes=notes)
        return CaseOutcome(
            query=case.query,
            relevant=case.relevant,
            retrieved=StageMetrics(
                stage=STAGE_RETRIEVED,
                ids=retrieved_ids,
                **measure_ids(retrieved_ids, case.relevant, self._k, grades=case.grades),
            ),
            packed=StageMetrics(
                stage=STAGE_PACKED,
                ids=packed_ids,
                **measure_ids(packed_ids, case.relevant, self._k, grades=case.grades),
            ),
            empty_reason=_empty_reason(answer),
            llm_called=answer.llm_called,
            fallback_reason=answer.fallback_reason,
            grounded=answer.grounded,
            coverage=check.coverage if check is not None else 0.0,
            given=check.given if check is not None else 0,
            cited=len(check.cited) if check is not None else 0,
            valid_count=len(check.valid) if check is not None else 0,
            unused_count=len(check.unused) if check is not None else 0,
            hallucinated=check.hallucinated if check is not None else 0,
            truncated_hits=answer.context.truncated_hits if answer.context else 0,
            dropped_hits=len(answer.context.dropped_hits) if answer.context else 0,
            dropped_by_rerank=(
                answer.retrieval.dropped_by_rerank if answer.retrieval else 0
            ),
            dropped_by_top_k=answer.retrieval.dropped_by_top_k if answer.retrieval else 0,
            answer_chars=len(answer.answer),
            latency_ms=elapsed_ms,
            faithfulness=faithfulness,
            notes=tuple(notes),
        )

    def run(self, cases: Sequence[RagEvalCase]) -> tuple[CaseOutcome, ...]:
        """批量跑（**逐条**，顺序与入参一致）.

        与 ``Retriever.retrieve_many`` / ``RagPipeline.answer_many`` 同样是
        刻意的逐条：每次问答的 ``llm_called`` / ``check`` / 延迟都是独立的证据，
        而"这一批总共花了多少"与"其中一次为什么没接地"是两个问题。
        """
        return tuple(self.run_case(case) for case in cases)

    # ------------------------------------------------------------------ 内部

    def _faithfulness(
        self, answer: RagAnswer, case: RagEvalCase, *, notes: list[str]
    ) -> float | None:
        """评一次忠实度（没有评审模型、没有参考答案、或评审失败时返回 ``None``）.

        三种"不评"的情形必须区分开，因此它们各自留一句 ``notes``：

```text
judge is None        没配评审模型      → 注记（这次的忠实度是"没量"，不是 0）
reference 为空        这份用例没有参考答案 → 注记（数据问题，不是失败）
评审输出无法解析      评审模型不守格式   → 注记（保留异常类型，便于定位）
```
        """
        if self._judge is None:
            notes.append("未配置评审模型：本次不评忠实度（faithfulness=None，不是 0）")
            return None
        if not case.reference:
            notes.append(
                "这条用例没有参考答案：忠实度无从评（faithfulness=None）——"
                "要评忠实度请给 reference 补一段答案"
            )
            return None
        try:
            result = score_faithfulness(answer.answer, case.reference, self._judge)
        except FaithfulnessParseError as exc:
            notes.append(
                f"评审模型输出无法解析（已跳过这一条的忠实度）: {exc}"
            )
            return None
        return result.score


def _empty_reason(answer: RagAnswer) -> str:
    """取空结果原因（**有命中时给空串**，而不是 ``empty_reason="hits"``）.

    为什么把 ``"hits"`` 归一成空串：``CaseOutcome.empty_reason`` 的语义是
    "这次为什么是空的"，而"有命中"不是一种空法。留一个 ``"hits"`` 在字段里
    会让归因函数必须记住"这个值要当空串看"，而那种约定迟早会有人漏掉
    （与 ``FALLBACK_REASON_NONE`` 用空串表示"没有回退"是同一条纪律：
    **哨兵值越少越好，最好只有一个**）。
    """
    if answer.retrieval is None or not answer.retrieval.is_empty:
        return ""
    reason = answer.retrieval.empty_reason
    return "" if reason == "hits" else reason


__all__ = [
    "DEFAULT_EVAL_K",
    "RagEvalRunner",
    "score_faithfulness",
]
