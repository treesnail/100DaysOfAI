"""day041 演示：哈希伪向量 vs 字符 n-gram 向量的语义差异.

运行方式（在项目根目录）：

    python scripts/embedding_demo.py

演示三件事：
  1. MockEmbedding（SHA-256）对语义相近的文本给不出高相似度；
  2. CharNgramEmbedding 因共享字符 n-gram 而"字面越近、向量越近"；
  3. 长期记忆批量写入（remember_many）+ 阈值召回（min_score）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.llm.embedding import CharNgramEmbedding, MockEmbedding
from smart_research_agent.memory.long_term import LongTermMemory
from smart_research_agent.memory.vector_store import cosine_similarity

ANCHOR = "我喜欢学习机器学习"
SIMILAR = "我爱学习机器学习"
UNRELATED = "今天晚饭吃什么"


def compare(name: str, emb) -> tuple[float, float]:
    anchor, similar, unrelated = emb.embed_batch([ANCHOR, SIMILAR, UNRELATED])
    sim_close = cosine_similarity(anchor, similar)
    sim_far = cosine_similarity(anchor, unrelated)
    print(f"  {name:>16}: 近义句 {sim_close:+.4f} | 无关句 {sim_far:+.4f}")
    return sim_close, sim_far


def main() -> int:
    print("== 语义区分度对比（锚点句：我喜欢学习机器学习） ==")
    mock_close, mock_far = compare("MockEmbedding", MockEmbedding(dimension=256))
    ngram_close, ngram_far = compare("CharNgramEmbedding", CharNgramEmbedding())

    # 哈希伪向量的相似度在 0 附近随机波动，近义句与无关句没有稳定差距
    assert abs(mock_close) < 0.5 and abs(mock_far) < 0.5, "哈希伪向量不应出现强相似"
    # 字符 n-gram 向量：近义句显著高于无关句，且自身接近 1
    assert ngram_close > ngram_far, "共享 n-gram 更多的文本必须更相似"
    assert ngram_close > 0.5, "近义句相似度应明显高于随机水平"

    print("== 长期记忆：批量写入 + 阈值召回 ==")
    mem = LongTermMemory(embedding=CharNgramEmbedding())
    ids = mem.remember_many(
        [
            "机器学习是人工智能的一个分支",
            "深度学习使用多层神经网络",
            "巴黎是法国的首都",
        ],
        metadatas=[{"topic": "ai"}, {"topic": "ai"}, {"topic": "geo"}],
    )
    assert len(ids) == 3
    hits = mem.recall("机器学习是什么", top_k=3, min_score=0.2)
    print(f"  min_score=0.2 召回 {len(hits)} 条: {hits}")
    assert hits, "阈值召回至少应命中机器学习相关记忆"
    print("全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
