"""day033 习题 10 参考答案：路由收益实验 —— 成本优先 vs 质量优先.

运行方式（在项目根目录）：

    python scripts/route_cost_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.router import (
    CostFirstStrategy,
    ModelRouter,
    ModelSpec,
    QualityFirstStrategy,
)
from smart_research_agent.observability.cost_tracker import CostTracker

TASKS = [
    "2+3 等于几",
    "法国的国鸟是什么",
    "把 1024 除以 8",
    "请分析和对比 RAG 与微调两种技术路线的取舍",
    "对比集中式与分布式记忆架构，并推理各自适用的场景",
]

PRICE_TABLE = {
    name: {"input": price, "output": price}
    for name, price in [("cheap", 0.001), ("mid", 0.01), ("strong", 0.1)]
}


class FailingLLM(BaseLLM):
    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        raise ConnectionError("模拟故障：连接被拒绝")


def make_models(fail_cheap: bool = False) -> list[ModelSpec]:
    return [
        ModelSpec(
            "cheap",
            FailingLLM() if fail_cheap else MockLLM(default="便宜模型的回答"),
            capability=2,
            cost_per_1k=0.001,
        ),
        ModelSpec("mid", MockLLM(default="中档模型的回答"), capability=3, cost_per_1k=0.01),
        ModelSpec("strong", MockLLM(default="强模型的回答"), capability=5, cost_per_1k=0.1),
    ]


def run_strategy(strategy, fail_cheap: bool = False) -> tuple[CostTracker, ModelRouter]:
    router = ModelRouter(make_models(fail_cheap), strategy=strategy)
    tracker = CostTracker(price_table={k: dict(v) for k, v in PRICE_TABLE.items()})
    by_name = {m.name: m.llm for m in router.models}
    for task in TASKS:
        router.chat([Message(role="user", content=task)])
        decision = router.decision_log[-1]
        effective = decision.fallback_used or decision.chosen
        consumed = tracker.record_from_llm(by_name[effective], model=effective)
        assert consumed == 1, "每个任务应恰好产生一次用量记录"
        print(f"  [{effective:>6}] 复杂度{decision.complexity}  {task[:20]}…  ({decision.reason})")
    return tracker, router


def main() -> int:
    print("== 成本优先策略 ==")
    cost_tracker, cost_router = run_strategy(CostFirstStrategy())
    print("== 质量优先策略 ==")
    quality_tracker, _ = run_strategy(QualityFirstStrategy())

    c, q = cost_tracker.total_cost, quality_tracker.total_cost
    print(f"成本优先总费用: ${c:.6f}")
    print(f"质量优先总费用: ${q:.6f}")
    print(f"路由节省: {(1 - c / q):.1%}")
    assert c < q, "成本优先必须严格省钱"
    simple_decisions = cost_router.decision_log[:3]
    assert all(d.chosen == "cheap" for d in simple_decisions), "简单任务应全部路由到 cheap"

    print("== 降级验证（cheap 故障） ==")
    fb_tracker, fb_router = run_strategy(CostFirstStrategy(), fail_cheap=True)
    fb_simple = fb_router.decision_log[:3]
    assert all(d.fallback_used is not None for d in fb_simple), "降级记录必须非空"
    print(f"cheap 故障时简单任务降级到: {fb_simple[0].fallback_used}，仍全部得到回复")
    print("全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
