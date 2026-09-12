"""day043 演示：语义缓存降低重复调用成本 + 高成本路径识别.

运行方式（在项目根目录）：

    python scripts/cache_demo.py

演示三件事：
  1. 语义缓存：措辞不同的重复问题命中缓存，底层 LLM 只被调用一次；
  2. 省成本归因：命中缓存把本应发生的调用折算成美元累计；
  3. CostTracker 的 top_cost_paths 识别最烧钱的「模型 × 接口」路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.cache import CachedLLM, SemanticCache
from smart_research_agent.llm.embedding import CharNgramEmbedding
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.observability.cost_tracker import CostTracker


def main() -> int:
    print("== 语义缓存：重复问题命中缓存 ==")
    llm = MockLLM(responses=["RAG 是检索增强生成，先检索再生成。"])
    cache = SemanticCache(CharNgramEmbedding(), similarity_threshold=0.5)
    cached = CachedLLM(llm, cache, model="gpt-4o-mini")

    first = cached.chat([Message(role="user", content="机器学习是什么")])
    second = cached.chat([Message(role="user", content="什么是机器学习")])  # 语义命中
    third = cached.chat([Message(role="user", content="机器学习是什么")])  # 精确命中

    assert first == second == third, "语义/精确命中的回复应与首次调用一致"
    assert len(llm.calls) == 1, "底层 LLM 只应被真实调用一次"
    print(f"  底层真实调用次数: {len(llm.calls)}（3 次提问只调 1 次模型）")
    stats = f"精确命中 {cache.exact_hits}，语义命中 {cache.semantic_hits}，未命中 {cache.misses}"
    print(f"  缓存统计: {stats}")
    print(f"  节省成本: ${cached.savings_usd:.8f}（两次命中省掉的调用折价）")
    assert cached.cache_hits == 2 and cached.savings_usd > 0

    print("== 高成本调用路径识别 ==")
    tracker = CostTracker()
    # 模拟一个模型走不同接口的混合账单
    tracker.record("gpt-4o-mini", 1200, 600, endpoint="chat")
    tracker.record("gpt-4o-mini", 3000, 0, endpoint="embedding")
    tracker.record("gpt-4o", 2000, 2000, endpoint="chat")
    for row in tracker.top_cost_paths():
        line = f"  {row['model']:>12} × {row['endpoint']:>10}: {row['calls']} 次, ${row['cost_usd']:.6f}"
        print(line)
    top = tracker.top_cost_paths()[0]
    assert top["model"] == "gpt-4o", "gpt-4o 单价最高，应为最烧钱路径"
    assert "top_paths" in tracker.report()

    print("全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
