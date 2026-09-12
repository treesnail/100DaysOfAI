"""本地模型 vs 云端模型的对比评估（day045）：换模型之前先量出来.

本地部署最大的诱惑是"免费"，最大的坑是"悄悄变笨"。所以 day045 的最后
一步不是把模型跑起来，而是**给换模型这件事一个可量化的决策依据**：

1. **延迟**（``latency_s``）：本地推理没有网络往返，但受显存带宽与量化
   精度影响；云端相反。用注入的 ``clock`` 计时，离线可测且不含真实等待；
2. **成本**（``estimated_cost``）：本地按"每 1K token 的电费摊销"折算
   （默认 0，即只算机时不计电费），云端按公开单价折算——同一个函数口径下
   对比才公平；
3. **一致性**（``agreement``）：**这是最容易被忽略、也最容易致命的指标**。
   延迟与成本可以算，但"回答还是不是原来那个意思"必须量出来：用 day041
   的 embedding 把两个模型的回复向量化，取余弦相似度的平均值。相似度低于
   阈值意味着"换了模型，Agent 的行为会变"，这不是省钱，是换产品。

token 计数统一走 ``llm.mock.estimate_tokens``：本地后端与云端后端对 usage
的上报口径未必一致（有的返回真实 usage、有的不返回），用同一个估算函数
才能保证对比的是模型而不是统计口径。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import CharNgramEmbedding, EmbeddingProvider
from smart_research_agent.llm.mock import estimate_tokens
from smart_research_agent.memory.vector_store import cosine_similarity
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 默认一致性阈值：平均余弦相似度 ≥ 0.8 视为"行为基本一致"
DEFAULT_AGREEMENT_THRESHOLD = 0.8


@dataclass
class CaseResult:
    """单条任务的执行结果与开销."""

    task: str
    reply: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelRun:
    """一个模型在一组任务上的完整跑分."""

    label: str
    results: list[CaseResult] = field(default_factory=list)
    price_per_1k: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)

    @property
    def total_latency_s(self) -> float:
        return round(sum(r.latency_s for r in self.results), 4)

    @property
    def mean_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return round(self.total_latency_s / len(self.results), 4)

    def estimated_cost(self) -> float:
        """按混合单价（输入输出同价）估算成本，单位与单价一致（美元）."""
        return round(self.total_tokens / 1000 * self.price_per_1k, 6)


@dataclass
class ComparisonReport:
    """一次本地/云端对比的完整结论."""

    local: ModelRun
    cloud: ModelRun
    similarities: list[float] = field(default_factory=list)
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD

    @property
    def agreement(self) -> float:
        """平均余弦相似度；无样本时为 0.0（而不是 1.0，避免"没测过就算达标"）."""
        if not self.similarities:
            return 0.0
        return round(sum(self.similarities) / len(self.similarities), 4)

    @property
    def latency_ratio(self) -> float:
        """本地/云端延迟比：>1 表示本地更慢（小于 1 才是本地的优势）."""
        cloud_mean = self.cloud.mean_latency_s
        if cloud_mean <= 0:
            return 0.0
        return round(self.local.mean_latency_s / cloud_mean, 4)

    @property
    def cost_saved(self) -> float:
        """相对云端省下的成本（云端成本 - 本地成本）."""
        return round(self.cloud.estimated_cost() - self.local.estimated_cost(), 6)

    def worst_case(self) -> CaseResult | None:
        """相似度最低的那条任务——排查"哪里变了"的入口."""
        if not self.similarities or len(self.similarities) != len(self.local.results):
            return None
        index = min(range(len(self.similarities)), key=lambda i: self.similarities[i])
        return self.local.results[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": self.local.label,
            "cloud": self.cloud.label,
            "cases": len(self.local.results),
            "agreement": self.agreement,
            "agreement_threshold": self.agreement_threshold,
            "mean_latency_s": {
                "local": self.local.mean_latency_s,
                "cloud": self.cloud.mean_latency_s,
            },
            "latency_ratio": self.latency_ratio,
            "cost": {
                "local": self.local.estimated_cost(),
                "cloud": self.cloud.estimated_cost(),
                "saved": self.cost_saved,
            },
        }

    def verdict(self) -> str:
        """一句话结论：能不能用本地模型替代云端."""
        if not self.local.results:
            return "无评估样本，无法给出结论"
        if self.agreement >= self.agreement_threshold:
            return (
                f"可替代：一致性 {self.agreement} ≥ 阈值 {self.agreement_threshold}，"
                f"延迟比 {self.latency_ratio}，每条任务节省 ${self.cost_saved:.4f}（共 "
                f"{len(self.local.results)} 条）"
            )
        worst = self.worst_case()
        hint = f"，最差样本：{worst.task[:30]}" if worst else ""
        return (
            f"需谨慎：一致性 {self.agreement} < 阈值 {self.agreement_threshold}，"
            f"换模型会改变 Agent 行为{hint}"
        )


class LocalModelEvaluator:
    """在固定任务集上对比两个模型，输出可决策的报告.

    ``embedder`` 缺省是离线可用的 ``CharNgramEmbedding``（day041）：评估
    不该依赖网络，也不该因为"没配 embedding 密钥"就跑不起来。
    """

    def __init__(
        self,
        local: BaseLLM,
        cloud: BaseLLM,
        *,
        embedder: EmbeddingProvider | None = None,
        clock: Callable[[], float] | None = None,
        agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
        cloud_price_per_1k: float = 0.01,
        local_price_per_1k: float = 0.0,
        system_prompt: str | None = None,
    ):
        if not 0.0 <= agreement_threshold <= 1.0:
            raise ValueError("agreement_threshold 必须在 [0, 1] 区间内")
        self.local = local
        self.cloud = cloud
        self.embedder = embedder or CharNgramEmbedding()
        self.clock = clock or time.perf_counter
        self.agreement_threshold = agreement_threshold
        self.cloud_price_per_1k = cloud_price_per_1k
        self.local_price_per_1k = local_price_per_1k
        self.system_prompt = system_prompt

    def _run_one(
        self, llm: BaseLLM, label: str, tasks: list[str], price_per_1k: float
    ) -> ModelRun:
        run = ModelRun(label=label, price_per_1k=price_per_1k)
        for task in tasks:
            messages: list[Message] = []
            if self.system_prompt:
                messages.append(Message(role="system", content=self.system_prompt))
            messages.append(Message(role="user", content=task))
            start = self.clock()
            reply = llm.chat(messages)
            elapsed = self.clock() - start
            run.results.append(
                CaseResult(
                    task=task,
                    reply=reply,
                    latency_s=round(elapsed, 6),
                    prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
                    completion_tokens=estimate_tokens(reply),
                )
            )
        return run

    def run(self, tasks: list[str]) -> ComparisonReport:
        """跑完两个模型的全部任务并计算一致性（空任务集返回空报告）."""
        local_run = self._run_one(self.local, "local", tasks, self.local_price_per_1k)
        cloud_run = self._run_one(self.cloud, "cloud", tasks, self.cloud_price_per_1k)
        similarities: list[float] = []
        for local_case, cloud_case in zip(local_run.results, cloud_run.results, strict=True):
            vec_local = self.embedder.embed(local_case.reply)
            vec_cloud = self.embedder.embed(cloud_case.reply)
            similarities.append(round(cosine_similarity(vec_local, vec_cloud), 4))
        report = ComparisonReport(
            local=local_run,
            cloud=cloud_run,
            similarities=similarities,
            agreement_threshold=self.agreement_threshold,
        )
        logger.info(
            "本地/云端对比完成：%d 条任务，一致性 %s，延迟比 %s",
            len(tasks),
            report.agreement,
            report.latency_ratio,
        )
        return report
