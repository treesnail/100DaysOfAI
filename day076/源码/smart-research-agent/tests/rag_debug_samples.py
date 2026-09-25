"""``rag_debug`` 的测试样本：**手算得出来**的三条用例与两条链路.

三条语料的向量两两正交（维度 3，各占一个坐标轴），因此"哪条最近"是算术题
而不是"跑一遍看看"：

```text
语料      d-1 "甲" → [1,0,0]     d-2 "乙" → [0,1,0]     d-3 "丙" → [0,0,1]
查询"甲" → [1,0,0]               余弦：d-1 = 1.0，d-2 = d-3 = 0.0
top_k=3 时名单恒为 ["d-1", "d-2", "d-3"]（0.0 的两条按 id 升序断）
```

于是四个检索指标全部可以手算，例如用例 1（金标准 ``d-1``、等级
``{"d-1": 2, "d-2": 1}``、k=3、名单 ``["d-1","d-2","d-3"]``）：

```text
recall            1/1 = 1.0
precision         1/3（三条里一条相关）
reciprocal_rank   1/1 = 1.0（第一条就命中）
ndcg（分级口径）   DCG  = (2²-1)/log2(2) + (2¹-1)/log2(3) = 3 + 1/1.585 = 3.6309
                  IDCG = 同上一行（理想排序就是 d-1 在前） → NDCG = 1.0
```

本模块还提供两个离线替身：

```text
VocabEmbedding   查表编码器：三个关键词各占一个坐标轴，其余字符忽略
ScriptedLLM      按脚本回复的模型，并把每次收到的提示词记下来
```
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.rag_debug.types import CaseOutcome, RagEvalCase, StageMetrics
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import make_record

#: 编码器的维度（= 关键词个数：三个关键词各占一个坐标轴）.
VECTOR_DIMENSION = 3

#: 查表：关键词 → 坐标轴序号（不在表里的字符被忽略）.
KEYWORDS: dict[str, int] = {"甲": 0, "乙": 1, "丙": 2}

#: 语料（id → 文本）。向量由 ``VocabEmbedding`` 现算。
CORPUS: tuple[dict[str, str], ...] = (
    {"id": "d-1", "text": "甲"},
    {"id": "d-2", "text": "乙"},
    {"id": "d-3", "text": "丙"},
)

#: 三条用例（第一条带"部分相关"的分级标注：``d-2`` 的等级是 1，
#: 但它**不在** relevant 里——这正是项目自带评测集的标注形态）。
CASES: tuple[RagEvalCase, ...] = (
    RagEvalCase(
        query="甲",
        relevant=("d-1",),
        reference="甲是第一份资料",
        grades={"d-1": 2, "d-2": 1},
    ),
    RagEvalCase(
        query="乙",
        relevant=("d-2",),
        reference="乙是第二份资料",
        grades={"d-2": 2},
    ),
    RagEvalCase(
        query="丙",
        relevant=("d-3",),
        reference="",
        grades={"d-3": 2},
    ),
)

#: 三个查询在 ``top_k = 3`` 上的名单（手算依据见模块 docstring）.
EXPECTED_IDS: tuple[str, ...] = ("d-1", "d-2", "d-3")

#: 用例 1 在 ``k = 3`` 上的四个指标（手算见模块 docstring）.
EXPECTED_CASE1_METRICS: dict[str, float] = {
    "recall": 1.0,
    "precision": 1 / 3,
    "reciprocal_rank": 1.0,
    "ndcg": 1.0,
}

#: 一条"引用了两个编号"的答案（前两个片段都用上了）.
ANSWER_TWO_CITATIONS = "依据 [1]，甲是第一份资料；依据 [2]，乙是第二份资料。"

#: 一条**幻觉引用**的答案（编号 9 不在那次提示词里）.
ANSWER_HALLUCINATED = "依据 [1] 与 [9]，两者一致。"


class VocabEmbedding(EmbeddingProvider):
    """查表编码器：三个关键词各占一个坐标轴（确定性、离线、可手算）."""

    @property
    def dimension(self) -> int:
        """向量维度（= 关键词个数）."""
        return VECTOR_DIMENSION

    def embed(self, text: str) -> list[float]:
        """按关键词出现次数填坐标轴（不在词表里的字符被忽略）."""
        vector = [0.0] * VECTOR_DIMENSION
        for char, axis in KEYWORDS.items():
            if char in text:
                vector[axis] += 1.0
        return vector


class ScriptedLLM(BaseLLM):
    """按脚本回复的模型：**记下每次的提示词**，并支持"用完之后回落到另一段".

    比 ``MockLLM`` 多一件事：``prompts`` 把每次收到的完整提示词留了下来。
    评估报告里那几句"模型看到的是哪几条片段"要靠它来核对——
    而"看答案像不像"是无法核对这件事的。
    """

    def __init__(self, replies: list[str] | None = None, fallback: str = "") -> None:
        self._replies = list(replies or [])
        self._fallback = fallback
        self.prompts: list[str] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """记下提示词并返回下一条脚本回复（脚本用尽时回落到 ``fallback``）."""
        self.prompts.append(messages[0].content if messages else "")
        if self._replies:
            return self._replies.pop(0)
        return self._fallback


class BoomLLM(BaseLLM):
    """一旦被调用就抛异常（用来证明"护栏生效"与"没走到生成"这两件事）."""

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """永远抛异常：它把"没调"变成一个可判定的证据."""
        raise AssertionError("这次不该调用模型")


def build_store(*, metric: str = "cosine") -> FlatVectorStore:
    """建一个灌好三条语料的库（向量由 ``VocabEmbedding`` 现算）."""
    store = FlatVectorStore(metric=metric)
    embedding = VocabEmbedding()
    store.upsert(
        [
            make_record(
                record_id=item["id"],
                vector=embedding.embed(item["text"]),
                text=item["text"],
            )
            for item in CORPUS
        ]
    )
    return store


def outcome(
    *,
    query: str = "甲",
    relevant: tuple[str, ...] = ("d-1",),
    retrieved_ids: tuple[str, ...] = ("d-1", "d-2"),
    packed_ids: tuple[str, ...] = ("d-1", "d-2"),
    empty_reason: str = "",
    llm_called: bool = True,
    fallback_reason: str = "",
    grounded: bool = True,
    coverage: float = 1.0,
    given: int = 2,
    cited: int = 2,
    valid_count: int = 2,
    unused_count: int = 0,
    hallucinated: int = 0,
    truncated_hits: int = 0,
    dropped_hits: int = 0,
    dropped_by_rerank: int = 0,
    dropped_by_top_k: int = 0,
    answer_chars: int = 12,
    latency_ms: float = 1.0,
    faithfulness: float | None = None,
    notes: tuple[str, ...] = (),
    k: int = 3,
) -> CaseOutcome:
    """手工造一条 ``CaseOutcome``（**归因是纯函数，因此可以逐字段构造**）.

    归因函数的每一条判据都只读几个字段，用显式构造的账来测它比"跑一条真链路
    再碰运气"精确得多：想测"重排切掉了金标准"就只改 ``dropped_by_rerank``，
    其余字段一动不动——**一个测试只验证一件事**。
    """
    from smart_research_agent.rag_debug.types import measure_ids

    grades = {"d-1": 2.0} if "d-1" in relevant else {}
    return CaseOutcome(
        query=query,
        relevant=relevant,
        retrieved=StageMetrics(
            stage="retrieved",
            ids=retrieved_ids,
            **measure_ids(retrieved_ids, relevant, k, grades=grades),
        ),
        packed=StageMetrics(
            stage="packed",
            ids=packed_ids,
            **measure_ids(packed_ids, relevant, k, grades=grades),
        ),
        empty_reason=empty_reason,
        llm_called=llm_called,
        fallback_reason=fallback_reason,
        grounded=grounded,
        coverage=coverage,
        given=given,
        cited=cited,
        valid_count=valid_count,
        unused_count=unused_count,
        hallucinated=hallucinated,
        truncated_hits=truncated_hits,
        dropped_hits=dropped_hits,
        dropped_by_rerank=dropped_by_rerank,
        dropped_by_top_k=dropped_by_top_k,
        answer_chars=answer_chars,
        latency_ms=latency_ms,
        faithfulness=faithfulness,
        notes=notes,
    )


def metrics_row(**overrides: Any) -> dict[str, float]:
    """造一行 11 个质量指标（缺省全 0.5，可逐项覆盖——用于基线对比的测试）."""
    from smart_research_agent.rag_debug.baseline import QUALITY_METRICS

    row = {name: 0.5 for name in QUALITY_METRICS}
    row.update({str(key): float(value) for key, value in overrides.items()})
    return row


__all__ = [
    "ANSWER_HALLUCINATED",
    "ANSWER_TWO_CITATIONS",
    "BoomLLM",
    "CASES",
    "CORPUS",
    "EXPECTED_CASE1_METRICS",
    "EXPECTED_IDS",
    "KEYWORDS",
    "ScriptedLLM",
    "VECTOR_DIMENSION",
    "VocabEmbedding",
    "build_store",
    "metrics_row",
    "outcome",
]
