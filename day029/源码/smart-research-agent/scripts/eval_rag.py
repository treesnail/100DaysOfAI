"""RAG 评估演示脚本：灌语料 → 跑评测集 → 打印指标.

用法::

    python scripts/eval_rag.py

embedding 说明：脚本内置一个确定性的"字符袋"embedding（按字符哈希分桶
计数），离线可复现，且对中文有朴素的字面相关性——共享字符越多的文本
向量越接近。它没有真正的语义理解能力，仅用于演示评估流水线；
生产环境应替换为真实 embedding 服务。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_research_agent.evaluation.rag_eval import RagEvaluator, load_jsonl
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.memory.vector_store import InMemoryVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


class CharBagEmbedding(EmbeddingProvider):
    """字符袋 embedding：每个字符哈希到一个桶并计数，最后做 L2 归一化."""

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for ch in text:
            bucket = int.from_bytes(hashlib.md5(ch.encode("utf-8")).digest()[:4]) % self._dimension
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def main() -> None:
    corpus = load_jsonl(DATA_DIR / "rag_corpus.jsonl")
    eval_set = load_jsonl(DATA_DIR / "rag_eval.jsonl")

    evaluator = RagEvaluator(vector_store=InMemoryVectorStore(), embedding=CharBagEmbedding())
    evaluator.index_documents(corpus)

    report = evaluator.evaluate_retrieval(eval_set, top_k=3)
    print("=== 检索评估 ===")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    print("\n=== 生成评估（MockLLM 扮演评审员） ===")
    judge = MockLLM(responses=['{"score": 1.0, "reason": "关键事实完全一致"}'])
    first = eval_set[0]
    result = evaluator.answer_faithfulness(
        answer="余弦相似度取值范围是 -1 到 1。",
        reference=first["reference_answer"],
        llm=judge,
    )
    print(f"faithfulness score = {result.score}, reason = {result.reason}")


if __name__ == "__main__":
    main()
