"""固定长度分块：把文本当成一串没有结构的字符（M6-D2）.

它是四种策略里唯一**完全不看内容**的一种：

```text
从第 0 个字符开始，每 max_tokens 个单位切一刀，刀口往回挪 overlap_tokens 个单位再切下一刀
```

它的价值不在于"切得好"，而在于它是**基线**：任何"更聪明的策略"都必须
先证明自己比它好。同时它也最容易推断——参数一确定，块数、放大量、
内存占用全都能**算出来**（见 ``window_plan``），不用先跑一遍。

## 必须说清楚的两件事

### 1. 它一定会切在句子中间

这不是缺陷，是定义。一份中文文档按 320 字切，平均每 320 字里必然
包含若干句号，而固定窗口不认识句号。表现是：

```text
块 #2 结尾："...因此我们建议把阈值设为 0."
块 #3 开头："85，然后观察两周的命中率。"
```

两侧各自读到半句话。**这正是重叠（overlap）存在的唯一理由**：它不解决
"切断了"，只保证"被切断的句子在至少一块里是完整的"。

### 2. "按 token 预算、按字符切"这两句话必须对齐

固定窗口的工作单位是**度量器的单位**：``measurer="chars"`` 时窗口是
"320 个字符"，``measurer="tiktoken"`` 时窗口是"320 个 token（约 250 个汉字）"。
实现上不做"320 token ≈ 某个字符数"的换算——那种换算表一旦写进代码，
它就同时失去了两个精度（既不是 token 也不是字符）。

参数与代价的关系（``window_plan`` 就是这段算术）：

```text
step            = max_tokens - overlap_tokens          每刀前进的量
chunks          = ceil((units - max_tokens) / step) + 1 切出的块数
overlap_units   = (chunks - 1) * overlap_tokens         白算的字符数
amplification   = overlap_tokens / step                 相对原文的放大量
```

一个必须记住的数：``max_tokens=320 / overlap=48`` 时 ``step=272``，
放大量 ``48/272 = 17.6%``。**重叠不是免费的**——它按比例推高
embedding 调用量与向量库容量，而它换来的只是"边界处不丢句子"。
"""

from __future__ import annotations

from math import ceil
from typing import Any

from smart_research_agent.chunking.base import Chunker, ChunkPolicy, Span, default_policy
from smart_research_agent.chunking.types import STRATEGY_FIXED
from smart_research_agent.documents.types import Document


def window_plan(doc_units: int, policy: ChunkPolicy) -> dict[str, Any]:
    """由"原文长度 + 参数"算出窗口切分的理论代价（**不切，只算**）.

    它是端点 ``POST /chunking/split`` 里 ``plan`` 字段的来源，也是
    "先算成本再决定参数"这条纪律的落地：把 ``max_tokens`` 从 320 调到 800
    会少切多少块、少花多少重叠，这里一行代码就能回答，
    而不用等 embedding 跑完一遍再看账单。

    ``doc_units`` 是原文按**度量器口径**的长度（chars 时就是字符数）。
    """
    step = policy.step
    if doc_units <= 0:
        chunks = 0
    elif doc_units <= policy.max_tokens:
        chunks = 1
    else:
        chunks = ceil((doc_units - policy.max_tokens) / step) + 1
    overlap_units = max(0, chunks - 1) * policy.overlap_tokens
    return {
        "strategy": policy.strategy,
        "max_tokens": policy.max_tokens,
        "overlap_tokens": policy.overlap_tokens,
        "step": step,
        "doc_units": doc_units,
        "chunks": chunks,
        "overlap_units": overlap_units,
        "total_units": doc_units + overlap_units,
        "amplification": round(policy.overlap_tokens / step, 4),
    }


class FixedSizeChunker(Chunker):
    """固定长度分块器（窗口 + 重叠）."""

    name = STRATEGY_FIXED

    def __init__(self, policy: ChunkPolicy | None = None, **kwargs: Any) -> None:
        super().__init__(policy or default_policy(STRATEGY_FIXED), **kwargs)
        # 固定窗口的重叠**在窗口内部**实现（下一刀往前挪），因此窗口
        # 本身就是总预算，不需要给重叠另留额度（见 base 里 window_budget 的说明）。
        self.window_budget = self.policy.max_tokens
        self.window_overlap = self.policy.overlap_tokens

    def _spans(self, document: Document) -> list[Span]:
        """整篇文本一个窗口一个窗口地切过去（**不看 blocks**）."""
        text = document.text
        if not text.strip():
            return []
        return self._back_off(text, 0, len(text), reason="budget")

    def describe(self) -> dict[str, Any]:
        """策略自述（端点与教程表格同源）."""
        return {
            "name": self.name,
            "unit": "固定窗口",
            "where_to_cut": "每 max_tokens 个单位一刀，与内容无关",
            "parameters": ["max_tokens", "overlap_tokens", "measurer"],
            "uses_similarity_percentile": False,
            "supports_overlap": True,
            "deterministic": "yes（chars 度量器下完全确定）",
            "strengths": [
                "参数一确定，块数与成本就能算出来（window_plan）",
                "不依赖结构信息：一份没有标题、没有标点的日志也能切",
                "块长度分布最窄，最容易做批处理与显存预算",
            ],
            "weaknesses": [
                "必然切在句子中间，靠 overlap 只能补救一部分",
                "把不同话题的相邻文本压进同一块（标题边界对它不存在）",
                "对代码与表格无保护：可能把一个函数从中间切开",
            ],
            "cost": "重叠放大量 = overlap/step",
        }


__all__ = ["FixedSizeChunker", "window_plan"]
