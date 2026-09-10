"""成本追踪：按模型/按接口归因 token 用量与费用.

归因原理：LLM 计费的粒度是"一次 API 调用"，但账单要回答的问题是
"哪个模型、哪个接口、哪类任务花了多少钱"。因此每次调用产生一条
``UsageRecord``（模型 + 接口 + 输入/输出 token 数），费用由可配置的
价格表折算，聚合维度（模型、接口）在记录时打上标签而非事后猜测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 默认可配置价格表：模型 -> 每 1K token 的输入/输出单价（美元）.
#: 数值仅为演示量级，接入真实模型时应以厂商最新定价为准。
DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    "mock-model": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}


@dataclass
class UsageRecord:
    """一次 LLM 调用的用量记录（归因的最小单元）."""

    model: str
    endpoint: str  # 接口名，如 "chat" / "embedding"，便于按接口聚合
    prompt_tokens: int
    completion_tokens: int


@dataclass
class CostTracker:
    """成本追踪器：收集 UsageRecord，按价格表折算费用并多维聚合.

    用法::

        tracker = CostTracker()
        llm = MockLLM(responses=["回答"])
        llm.chat([Message(role="user", content="hi")])
        tracker.record_from_llm(llm, model="mock-model")  # 消费自上次以来的增量
        tracker.report()  # -> 总费用 + 按模型/接口的分解
    """

    price_table: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_PRICE_TABLE.items()}
    )
    records: list[UsageRecord] = field(default_factory=list)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        endpoint: str = "chat",
    ) -> UsageRecord:
        """登记一条用量记录；模型不在价格表中时抛 KeyError（宁可报错不可漏算）."""
        if model not in self.price_table:
            raise KeyError(f"价格表中没有模型 {model!r}，请先在 price_table 中配置价格")
        usage = UsageRecord(
            model=model,
            endpoint=endpoint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.records.append(usage)
        return usage

    def record_from_llm(self, llm, model: str, endpoint: str = "chat") -> int:
        """把 LLM 实例自上次调用本方法以来新增的 usage_log 条目登记进来.

        依赖 day032 给 MockLLM 加的 ``usage_log``；真实 OpenAI 兼容客户端
        每次响应自带 usage 字段，接线方式相同。返回本次登记的条数。
        """
        consumed = getattr(llm, "_cost_consumed", 0)
        new_entries = llm.usage_log[consumed:]
        for entry in new_entries:
            self.record(
                model=model,
                prompt_tokens=entry["prompt_tokens"],
                completion_tokens=entry["completion_tokens"],
                endpoint=endpoint,
            )
        llm._cost_consumed = len(llm.usage_log)
        return len(new_entries)

    def record_cost(self, usage: UsageRecord) -> float:
        """单条记录的费用（美元）= (输入token*输入单价 + 输出token*输出单价) / 1000."""
        price = self.price_table[usage.model]
        return (
            usage.prompt_tokens * price["input"] + usage.completion_tokens * price["output"]
        ) / 1000.0

    @property
    def total_cost(self) -> float:
        return sum(self.record_cost(r) for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.records)

    def breakdown(self, key: str) -> dict[str, dict[str, float]]:
        """按维度（"model" 或 "endpoint"）聚合 token 数与费用."""
        result: dict[str, dict[str, float]] = {}
        for r in self.records:
            group = getattr(r, key)
            bucket = result.setdefault(group, {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
            bucket["prompt_tokens"] += r.prompt_tokens
            bucket["completion_tokens"] += r.completion_tokens
            bucket["cost"] += self.record_cost(r)
        return result

    def report(self) -> dict:
        """导出聚合报告：总览 + 按模型 + 按接口分解（可直接 json.dumps）."""
        return {
            "total_calls": len(self.records),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "by_model": {
                k: {kk: round(vv, 6) if kk == "cost" else vv for kk, vv in v.items()}
                for k, v in self.breakdown("model").items()
            },
            "by_endpoint": {
                k: {kk: round(vv, 6) if kk == "cost" else vv for kk, vv in v.items()}
                for k, v in self.breakdown("endpoint").items()
            },
        }
