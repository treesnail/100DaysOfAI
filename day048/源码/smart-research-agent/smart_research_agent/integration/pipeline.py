"""M4 一体化流水线（day046）：把六层能力串成一条可观测的调用链.

M4 阶段（day037~day045）我们分别学会了六件独立的事：Function Calling、
FastAPI 服务化、流式与多模型路由、多模态、Embedding、成本与安全。它们
此前各自躺在自己的模块里，由不同的调用方分别使用——``/chat`` 走审核、
``CachedLLM`` 走缓存、``ModelRouter`` 走路由，**但没有任何一条真实请求
把所有能力都走一遍**。本模块补的就是这条主线：

    输入侧护栏 → 语义缓存 → 模型路由与生成 → 成本归因 → 输出侧审核 → 缓存回写

流水线的价值不在"多写一层封装"，而在**顺序**与**可观测**：

1. **顺序即策略**。六个阶段的先后不是随意的——
   - 护栏必须最先：恶意输入不该花掉一次昂贵的 LLM 调用；
   - 缓存必须在生成之前、且在护栏之后：缓存命中的是"安全前提下的重复问题"；
   - 审核必须在回写缓存之前：**缓存里存的必须是脱敏后的文本**，否则一次
     未脱敏的写入会让后续所有命中都泄露同一份 PII（缓存把一次性泄漏
     变成了持续性泄漏，这是最容易被忽略的安全陷阱）；
   - 成本归因紧跟在生成之后：只有真正发生了调用才有账单。

2. **阶段可观测**。每一步都产出 ``StageRecord``（阶段名 + 毫秒耗时 +
   人类可读细节），随响应一起返回。这是 day043 追踪、day044 审计的延续：
   当一次请求变慢或变贵，可以直接回答"是缓存没命中，还是路由选贵了，
   还是审核正则在长文本上回溯"——而不是对着一个总耗时数字猜。

3. **失败方向明确**。护栏拒答时**不调用模型**（省一次钱，且不给注入内容
   任何被模型看到的机会）；探活失败、缓存 miss 都退化为"继续往下走"，
   只有真正的 LLM 异常才向上抛。

与 day043 ``CachedLLM`` 的关系：``CachedLLM`` 是**透明包装器**路线——把缓存
藏在一个 ``BaseLLM`` 里，上层无感知但也无法区分"命中还是真调用"。流水线
需要把命中显式暴露为阶段事实（``cached=True``、``model="cache"``、成本 0），
所以这里直接组合 ``SemanticCache`` 而不是套一层 ``CachedLLM``。两条路线并存：
只要对上层透明就用 ``CachedLLM``，需要可观测就用流水线。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.cache import SemanticCache
from smart_research_agent.llm.router import CostFirstStrategy, ModelRouter, ModelSpec
from smart_research_agent.observability.cost_tracker import CostTracker
from smart_research_agent.security.content_moderator import ContentModerator, ModerationResult
from smart_research_agent.security.injection_detector import (
    InjectionReport,
    PromptInjectionDetector,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 输入侧护栏命中注入时的固定拒答文本.
#: 刻意不把用户的原文回显，也不请求模型"解释为什么拒绝"——拒答本身
#: 若由模型生成，就等于把攻击串送进了模型上下文，护栏形同虚设。
REFUSAL_REPLY = "抱歉，本次请求包含疑似 Prompt 注入的内容，已被输入侧护栏拦截。"

#: 缓存命中时的伪模型名：让调用方一眼看出"这次回答没有花模型的钱".
CACHE_MODEL_LABEL = "cache"

#: 阶段名常量（顺序即执行顺序，见 STAGE_ORDER）
STAGE_GUARD = "guard"
STAGE_CACHE = "cache"
STAGE_GENERATE = "generate"
STAGE_ACCOUNT = "account"
STAGE_MODERATE = "moderate"
STAGE_STORE = "store"

#: 正常路径的阶段顺序（拒答时只有 guard 一个阶段，缓存命中时止于 cache）
STAGE_ORDER: tuple[str, ...] = (
    STAGE_GUARD,
    STAGE_CACHE,
    STAGE_GENERATE,
    STAGE_ACCOUNT,
    STAGE_MODERATE,
    STAGE_STORE,
)

#: 未在价格表中出现的模型的默认单价：0 成本.
#: 本地/自建推理服务没有"每 token 单价"，按 0 计入是正确的会计口径；
#: 但它必须**显式**发生（写入 price_table 并告警），而不是被静默忽略。
ZERO_PRICE: dict[str, float] = {"input": 0.0, "output": 0.0}


@dataclass
class StageRecord:
    """流水线中一个阶段的执行记录（可观测性的最小单元）."""

    name: str
    duration_ms: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "duration_ms": self.duration_ms, "detail": self.detail}


@dataclass
class PipelineResult:
    """一次流水线请求的完整结果：答复 + 每个阶段的账.

    ``cost_usd`` 是**本次请求新增**的费用（不是累计值）：流水线在生成前后
    各读一次 ``CostTracker.total_cost``，差值就是这次调用花的钱——归因到
    单次请求才可能回答"哪类问题在烧钱"。
    """

    reply: str
    model: str
    cached: bool
    blocked: bool
    moderation: ModerationResult
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    injection: InjectionReport | None = None
    stages: list[StageRecord] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def stage(self, name: str) -> StageRecord | None:
        """按名取阶段记录（不存在返回 None，调用方无需先判断）."""
        return next((s for s in self.stages if s.name == name), None)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 json.dumps 的字典（供日志、API 与基线采集复用）."""
        return {
            "reply": self.reply,
            "model": self.model,
            "cached": self.cached,
            "blocked": self.blocked,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_ms": self.total_ms,
            "stages": [s.to_dict() for s in self.stages],
            "moderation": {
                "is_safe": self.moderation.is_safe,
                "flagged_words": list(self.moderation.flagged_words),
                "pii_types": list(self.moderation.pii_types),
            },
            "injection": (
                None
                if self.injection is None
                else {
                    "is_injection": self.injection.is_injection,
                    "matched_patterns": list(self.injection.matched_patterns),
                }
            ),
        }


class IntegratedPipeline:
    """M4 能力的一体化装配：一次 ``run`` 走完六个阶段.

    所有协作者都可注入（``None`` 表示"这一层不启用"），因此同一条流水线
    既能装配成生产形态（路由 + 缓存 + 计费 + 审核 + 护栏），也能退化成
    裸调用（全部为 None）——后者对测试与排障极有价值：怀疑某层有副作用时
    逐层关掉即可二分定位。

    ``clock`` 可注入是为了让**耗时统计本身可测**：测试传入受控的假时钟，
    就能断言阶段耗时与百分位数的计算逻辑，而不用真的 sleep。
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        cache: SemanticCache | None = None,
        tracker: CostTracker | None = None,
        moderator: ContentModerator | None = None,
        detector: PromptInjectionDetector | None = None,
        block_on_injection: bool = True,
        system_prompt: str | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.llm = llm
        self.cache = cache
        self.tracker = tracker
        self.moderator = moderator
        self.detector = detector
        self.block_on_injection = block_on_injection
        self.system_prompt = system_prompt
        self._clock = clock or time.perf_counter
        #: 累计计数器：轻量、可断言，供 /pipeline/baseline 与运维排查
        self.requests = 0
        self.blocked_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    # -- 内部工具 ---------------------------------------------------------

    def _elapsed_ms(self, start: float) -> float:
        return round((self._clock() - start) * 1000, 4)

    def cache_key(self, task: str, system_prompt: str | None = None) -> str:
        """缓存键 = 系统提示词 + 用户任务.

        之所以把系统提示词并入键：同一句"总结一下"在不同人格/指令下应当
        得到不同回答，若只按用户文本做键，换个 system prompt 就会命中旧
        答案——这类"张冠李戴"比缓存不命中危险得多。
        """
        prefix = (system_prompt if system_prompt is not None else self.system_prompt) or ""
        return f"{prefix}\n{task}" if prefix else task

    def _resolve_model(self) -> str:
        """解析本次调用实际生效的模型名.

        优先级：路由决策日志（``ModelRouter.decision_log`` 的最后一条，
        降级时取 ``fallback_used``）→ 客户端自述名（``model_name``/``_model``）
        → 类名兜底。成本归因必须落在**真实生效**的模型上，否则"是谁花的钱"
        会被记到首选模型头上。
        """
        log = getattr(self.llm, "decision_log", None)
        if log:
            last = log[-1]
            return str(last.fallback_used or last.chosen)
        name = getattr(self.llm, "model_name", None) or getattr(self.llm, "_model", None)
        return str(name or type(self.llm).__name__)

    def _moderate(self, text: str) -> ModerationResult:
        """输出侧审核；未装配审核器时视为干净（不脱敏、不改写）."""
        if self.moderator is None:
            return ModerationResult(is_safe=True, sanitized_text=text)
        return self.moderator.moderate(text)

    @staticmethod
    def _moderate_detail(moderation: ModerationResult) -> str:
        """审核阶段的细节描述（命中与通过两条文案共用一处，避免两处漂移）."""
        if moderation.is_safe:
            return "通过"
        return f"flagged={moderation.flagged_words} pii={moderation.pii_types}"

    def _account(self, model: str) -> tuple[float, int, int]:
        """成本归因：返回 (本次新增费用, 本次新增 prompt/completion token).

        归因对象是**真正产生用量的叶子模型**：若 ``llm`` 是 ``ModelRouter``
        这类包装器，就取它最后一次实际使用的叶子（``last_used_llm``）交给
        计费器——否则 ``CostTracker.record_from_llm`` 会因为"包装器没有
        ``usage_log``"而断链（这正是 day046 集成时暴露出的真实缺陷）。

        叶子不支持用量上报时退回"记 0 并告警"：服务不该因为一个可观测性
        协作者没接好就整体失败，但也绝不假装有数字。
        """
        if self.tracker is None:
            return 0.0, 0, 0
        target = getattr(self.llm, "last_used_llm", None) or self.llm
        if not hasattr(target, "usage_log"):
            logger.warning(
                "模型 %s（%s）不提供 usage_log，本次用量记 0；"
                "如需计费请让该客户端按约定上报用量",
                model,
                type(target).__name__,
            )
            return 0.0, 0, 0
        if model not in self.tracker.price_table:
            # 本地/自建模型不在价格表里是常态：显式按 0 记价并告警，
            # 让"成本为零"是一个被记录的决定，而不是一个被忽略的异常。
            self.tracker.price_table[model] = dict(ZERO_PRICE)
            logger.warning(
                "模型 %s 不在价格表中，本次按 0 成本计入（本地/自建推理的常见情形）", model
            )
        before_cost = self.tracker.total_cost
        before_records = len(self.tracker.records)
        self.tracker.record_from_llm(target, model=model)
        new_records = self.tracker.records[before_records:]
        prompt_tokens = sum(r.prompt_tokens for r in new_records)
        completion_tokens = sum(r.completion_tokens for r in new_records)
        return round(self.tracker.total_cost - before_cost, 8), prompt_tokens, completion_tokens

    # -- 主流程 -----------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> PipelineResult:
        """执行一次完整流水线.

        正常路径阶段序列为 ``STAGE_ORDER``；两条捷径各有意涵：
          - 注入命中且 ``block_on_injection=True``：只有 ``guard`` 一个阶段，
            模型完全不被调用（``blocked=True``）；
          - 缓存命中：``guard → cache → moderate``（跳过 generate 与 account），
            即 ``cached=True``、``model="cache"``、``cost_usd=0.0``；仍然复核
            输出审核是为了契约一致——缓存里存的已是脱敏文本，重复脱敏幂等。
        """
        self.requests += 1
        started = self._clock()
        stages: list[StageRecord] = []

        # 阶段 1：输入侧护栏（最早、最便宜、省一次调用）
        injection: InjectionReport | None = None
        guard_start = self._clock()
        if self.detector is not None:
            injection = self.detector.scan(task)
        stages.append(
            StageRecord(
                name=STAGE_GUARD,
                duration_ms=self._elapsed_ms(guard_start),
                detail=(
                    "未装配输入侧护栏"
                    if injection is None
                    else (
                        f"命中注入规则: {', '.join(injection.matched_patterns)}"
                        if injection.is_injection
                        else "通过"
                    )
                ),
            )
        )
        if injection is not None and injection.is_injection and self.block_on_injection:
            self.blocked_requests += 1
            moderation = self._moderate(REFUSAL_REPLY)
            logger.warning("输入侧护栏拦截请求: %s", injection.matched_patterns)
            return PipelineResult(
                reply=moderation.sanitized_text,
                model="blocked",
                cached=False,
                blocked=True,
                moderation=moderation,
                injection=injection,
                cost_usd=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                stages=stages,
                total_ms=self._elapsed_ms(started),
            )

        # 阶段 2：语义缓存
        key = self.cache_key(task, system_prompt)
        cache_start = self._clock()
        hit: str | None = None
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is None:
                self.cache_misses += 1
            else:
                self.cache_hits += 1
        stages.append(
            StageRecord(
                name=STAGE_CACHE,
                duration_ms=self._elapsed_ms(cache_start),
                detail=(
                    "未装配缓存"
                    if self.cache is None
                    else ("命中，跳过模型调用" if hit is not None else "未命中")
                ),
            )
        )
        if hit is not None:
            # 命中也要复核并记账：缓存里存的是脱敏文本，重复审核是幂等的，
            # 但把这一步记进阶段表，才能让调用方看懂"答得快是因为没调模型"。
            moderate_start = self._clock()
            moderation = self._moderate(hit)
            stages.append(
                StageRecord(
                    name=STAGE_MODERATE,
                    duration_ms=self._elapsed_ms(moderate_start),
                    detail=self._moderate_detail(moderation),
                )
            )
            return PipelineResult(
                reply=moderation.sanitized_text,
                model=CACHE_MODEL_LABEL,
                cached=True,
                blocked=False,
                moderation=moderation,
                injection=injection,
                cost_usd=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                stages=stages,
                total_ms=self._elapsed_ms(started),
            )

        # 阶段 3：组装提示词并调用（可能是 ModelRouter，路由决策在内部完成）
        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages: list[Message] = []
        if effective_prompt:
            messages.append(Message(role="system", content=effective_prompt))
        messages.append(Message(role="user", content=task))

        generate_start = self._clock()
        reply = self.llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
        model = self._resolve_model()
        stages.append(
            StageRecord(
                name=STAGE_GENERATE,
                duration_ms=self._elapsed_ms(generate_start),
                detail=f"model={model}",
            )
        )

        # 阶段 4：成本归因（差值为本次请求的费用）
        account_start = self._clock()
        cost_usd, prompt_tokens, completion_tokens = self._account(model)
        stages.append(
            StageRecord(
                name=STAGE_ACCOUNT,
                duration_ms=self._elapsed_ms(account_start),
                detail=(
                    "未装配成本追踪器"
                    if self.tracker is None
                    else f"tokens={prompt_tokens + completion_tokens} cost=${cost_usd:.6f}"
                ),
            )
        )

        # 阶段 5：输出侧审核（脱敏在这里发生）
        moderate_start = self._clock()
        moderation = self._moderate(reply)
        stages.append(
            StageRecord(
                name=STAGE_MODERATE,
                duration_ms=self._elapsed_ms(moderate_start),
                detail=self._moderate_detail(moderation),
            )
        )

        # 阶段 6：缓存回写——存**脱敏后**文本，避免把 PII 固化进缓存
        store_start = self._clock()
        if self.cache is not None:
            self.cache.put(key, moderation.sanitized_text)
        stages.append(
            StageRecord(
                name=STAGE_STORE,
                duration_ms=self._elapsed_ms(store_start),
                detail="已写入缓存（脱敏后文本）" if self.cache is not None else "未装配缓存",
            )
        )

        return PipelineResult(
            reply=moderation.sanitized_text,
            model=model,
            cached=False,
            blocked=False,
            moderation=moderation,
            injection=injection,
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stages=stages,
            total_ms=self._elapsed_ms(started),
        )

    # -- 自述与统计 -------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """自述装配情况（供 /health、/docs 与排障时确认"这层到底开没开"）."""
        return {
            "llm": type(self.llm).__name__,
            "stages": list(STAGE_ORDER),
            "cache": self.cache is not None,
            "tracker": self.tracker is not None,
            "moderator": self.moderator is not None,
            "injection_detector": self.detector is not None,
            "block_on_injection": self.block_on_injection,
        }

    def stats(self) -> dict[str, Any]:
        """运行统计：请求数、拦截数、缓存命中与命中率."""
        lookups = self.cache_hits + self.cache_misses
        return {
            "requests": self.requests,
            "blocked": self.blocked_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / lookups, 4) if lookups else 0.0,
        }


def build_hybrid_router(
    local: BaseLLM,
    cloud: BaseLLM,
    *,
    local_name: str | None = None,
    cloud_name: str | None = None,
    local_capability: int = 3,
    cloud_capability: int = 5,
    local_cost_per_1k: float = 0.0,
    cloud_cost_per_1k: float = 0.01,
) -> ModelRouter:
    """装配"本地 + 云端"混合路由器（day045 × day039 的合流点）.

    选型逻辑直接复用 day033 的 ``CostFirstStrategy``：在满足能力阈值的候选里
    挑最便宜的。把本地模型标成 **capability=3、cost=0**，云端标成
    **capability=5、cost>0**，于是免费的本地模型自然覆盖了简单任务，
    而复杂任务（``estimate_complexity`` 打出 4~5 分）会因能力不足被路由到
    云端——**不需要为"省钱"写任何 if 判断**，策略已经把成本与能力都算进去了。

    这正是 day045 结尾留下的问题（"本地模型值不值得用"）在架构上的答案：
    不搞二选一，而是让路由器按题目难度自动分流；本地扛不住的题自动升级到
    云端，用户无感知，账单却小了一截。

    capability 必须落在 1~5（``estimate_complexity`` 的值域）；当本地档位
    不低于云端时给出告警——这种配置下云端永远不会被选中，等于花了两份钱
    只用一份（可能是有意的，但更可能是配错了）。
    """
    if not 1 <= local_capability <= 5:
        raise ValueError("local_capability 必须在 1~5 之间")
    if not 1 <= cloud_capability <= 5:
        raise ValueError("cloud_capability 必须在 1~5 之间")
    if local_cost_per_1k < 0 or cloud_cost_per_1k < 0:
        raise ValueError("成本单价不能为负数")

    resolved_local = local_name or getattr(local, "model_name", None) or "local"
    resolved_cloud = cloud_name or getattr(cloud, "_model", None) or "cloud"
    if resolved_local == resolved_cloud:
        raise ValueError(f"本地与云端模型名相同（{resolved_local}），路由器无法区分候选")
    if local_capability >= cloud_capability:
        logger.warning(
            "本地档位 %d 不低于云端 %d：云端模型将不会被路由选中",
            local_capability,
            cloud_capability,
        )

    router = ModelRouter(
        [
            ModelSpec(
                name=str(resolved_local),
                llm=local,
                capability=local_capability,
                cost_per_1k=local_cost_per_1k,
            ),
            ModelSpec(
                name=str(resolved_cloud),
                llm=cloud,
                capability=cloud_capability,
                cost_per_1k=cloud_cost_per_1k,
            ),
        ],
        strategy=CostFirstStrategy(),
    )
    logger.info("混合路由已装配: 本地=%s(免费) 云端=%s(付费)", resolved_local, resolved_cloud)
    return router


def default_pipeline(
    llm: BaseLLM,
    *,
    cache: SemanticCache | None = None,
    tracker: CostTracker | None = None,
    moderator: ContentModerator | None = None,
    detector: PromptInjectionDetector | None = None,
    block_on_injection: bool | None = None,
    system_prompt: str | None = None,
    clock: Callable[[], float] | None = None,
) -> IntegratedPipeline:
    """按 ``settings`` 装配生产默认流水线（六层能力全开）.

    与 ``default_embedding()``/``default_cache()``/``create_local_model()``
    同一套路：**"从配置到实例"的映射集中在工厂里**。缓存是否启用读
    ``settings.pipeline_cache_enabled``，护栏是否拦截读
    ``settings.pipeline_block_on_injection``——两个开关都可在 .env 里改，
    无需碰代码（灰度开关的标准做法）。

    ``llm`` 由调用方传入而非在本函数内构造：生产端点要用 ``app.state`` 上
    那个已经被注入/替换过的 LLM（测试可换血），流水线不该绕过它自己造一个。
    """
    from smart_research_agent.config import settings

    resolved_cache = cache
    if resolved_cache is None and settings.pipeline_cache_enabled:
        from smart_research_agent.llm.cache import default_cache

        resolved_cache = default_cache()

    return IntegratedPipeline(
        llm,
        cache=resolved_cache,
        tracker=tracker if tracker is not None else CostTracker(),
        moderator=moderator if moderator is not None else ContentModerator(),
        detector=detector if detector is not None else PromptInjectionDetector(),
        block_on_injection=(
            settings.pipeline_block_on_injection
            if block_on_injection is None
            else block_on_injection
        ),
        system_prompt=system_prompt,
        clock=clock,
    )
