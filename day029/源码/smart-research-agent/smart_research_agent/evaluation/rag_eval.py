"""RAG 评估器：把检索评估指标与生成答案对比评估接入知识库模块.

两个职责：

1. **检索评估**：对带标注的评测集逐条跑 Top-K 检索，计算
   recall / precision / MRR / NDCG 的单条值与均值。
2. **生成评估**：用 LLM-as-a-judge 对比生成答案与参考答案，
   给出忠实度（faithfulness）评分。

设计上有意把检索评估与生成评估拆成两条独立路径：检索错了答案再好
也是"蒙对"，检索对了答案错了则是生成环节的问题——分开评估才能
定位病灶（详见教程第五章）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.evaluation.rag_metrics import (
    mrr,
    ndcg,
    retrieval_precision,
    retrieval_recall,
)
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.memory.vector_store import InMemoryVectorStore, VectorRecord

FAITHFULNESS_PROMPT = """你是一个答案评审员。请对比"生成答案"与"参考答案"，评估生成答案的忠实度。

评分标准：
- 1.0：关键事实与参考答案完全一致（措辞可不同）
- 0.5：部分一致，有遗漏或轻微偏差
- 0.0：关键事实错误或答非所问

严格输出 JSON 对象，不要输出其他内容，例如：
{{"score": 1.0, "reason": "关键事实完全一致"}}

参考答案：{reference}
生成答案：{answer}
"""


class FaithfulnessParseError(ValueError):
    """忠实度评审输出解析失败."""


@dataclass
class FaithfulnessResult:
    """一次答案对比评估的结果."""

    score: float
    reason: str = ""


@dataclass
class QueryEvalResult:
    """单个查询的检索评估明细."""

    query: str
    retrieved_ids: list[str]
    recall: float
    precision: float
    mrr: float
    ndcg: float


@dataclass
class RetrievalEvalReport:
    """整个评测集的检索评估报告：逐条明细 + 四个指标的均值."""

    top_k: int
    per_query: list[QueryEvalResult] = field(default_factory=list)
    mean_recall: float = 0.0
    mean_precision: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "num_queries": len(self.per_query),
            "mean_recall": self.mean_recall,
            "mean_precision": self.mean_precision,
            "mean_mrr": self.mean_mrr,
            "mean_ndcg": self.mean_ndcg,
            "per_query": [
                {
                    "query": r.query,
                    "retrieved_ids": r.retrieved_ids,
                    "recall": r.recall,
                    "precision": r.precision,
                    "mrr": r.mrr,
                    "ndcg": r.ndcg,
                }
                for r in self.per_query
            ],
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，每行一个 JSON 对象，跳过空行."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_faithfulness(text: str) -> FaithfulnessResult:
    """防御式解析 LLM 的评审输出：定位 {} 边界 → JSON → 结构 → 值域."""
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FaithfulnessParseError("评审输出中未找到 JSON 对象")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise FaithfulnessParseError(f"评审输出 JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict) or "score" not in data:
        raise FaithfulnessParseError("评审输出必须包含 score 字段")
    try:
        score = float(data["score"])
    except (TypeError, ValueError) as exc:
        raise FaithfulnessParseError("score 必须是数值") from exc
    if not 0.0 <= score <= 1.0:
        raise FaithfulnessParseError(f"score 越界 [0, 1]: {score}")
    return FaithfulnessResult(score=score, reason=str(data.get("reason", "")))


class RagEvaluator:
    """RAG 评估器：对接向量库与 embedding，跑检索评估与答案对比评估.

    依赖全部通过构造函数注入：测试时注入 InMemoryVectorStore +
    MockEmbedding + MockLLM，完全离线；生产环境换成真实实现即可。
    """

    def __init__(self, vector_store: InMemoryVectorStore, embedding: EmbeddingProvider) -> None:
        self.vector_store = vector_store
        self.embedding = embedding

    def index_documents(self, docs: list[dict[str, str]]) -> None:
        """把语料灌入向量库。docs 元素形如 {"id": ..., "text": ...}."""
        for doc in docs:
            self.vector_store.add(
                VectorRecord(
                    id=doc["id"],
                    text=doc["text"],
                    vector=self.embedding.embed(doc["text"]),
                )
            )

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """对单个查询做 Top-K 检索，返回按相关度排序的文档 id 列表."""
        hits = self.vector_store.search(self.embedding.embed(query), top_k=top_k)
        return [record.id for record, _score in hits]

    def evaluate_retrieval(
        self, eval_set: list[dict[str, Any]], top_k: int = 3
    ) -> RetrievalEvalReport:
        """对评测集逐条检索并计算四个指标.

        eval_set 元素形如::

            {"query": "...", "relevant_ids": ["d1"], "relevance_grades": {"d1": 2}}

        relevance_grades 可省略，此时 NDCG 退化为"二值相关"版本
        （由 relevant_ids 自动生成等级 1）。
        """
        report = RetrievalEvalReport(top_k=top_k)
        for item in eval_set:
            query = item["query"]
            relevant_ids = list(item["relevant_ids"])
            grades = dict(item.get("relevance_grades") or {d: 1.0 for d in relevant_ids})
            retrieved = self.retrieve(query, top_k=top_k)
            report.per_query.append(
                QueryEvalResult(
                    query=query,
                    retrieved_ids=retrieved,
                    recall=retrieval_recall(retrieved, relevant_ids),
                    precision=retrieval_precision(retrieved, relevant_ids),
                    mrr=mrr(retrieved, relevant_ids),
                    ndcg=ndcg(retrieved, grades),
                )
            )
        n = len(report.per_query)
        if n:
            report.mean_recall = sum(r.recall for r in report.per_query) / n
            report.mean_precision = sum(r.precision for r in report.per_query) / n
            report.mean_mrr = sum(r.mrr for r in report.per_query) / n
            report.mean_ndcg = sum(r.ndcg for r in report.per_query) / n
        return report

    def answer_faithfulness(
        self, answer: str, reference: str, llm: BaseLLM
    ) -> FaithfulnessResult:
        """用 LLM-as-a-judge 对比生成答案与参考答案.

        LLM 在此被当作一个"评分函数"：提示词锁定 JSON 输出结构，
        解析端用与 parse_plan 相同的防御式策略逐层校验。
        """
        raw = llm.chat(
            [Message(role="user", content=FAITHFULNESS_PROMPT.format(reference=reference, answer=answer))]
        )
        return _parse_faithfulness(raw)
