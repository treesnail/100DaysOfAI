"""RAG 检索评估指标：召回率、精确率、MRR、NDCG.

本模块的四个函数全部是**纯函数**：不读文件、不碰网络、不依赖任何
全局状态，输入两个列表（或一个字典）就输出一个 [0, 1] 之间的小数。
这保证它们可以完全离线、逐位确定地被单元测试验证。

记号约定
--------
- retrieved_ids: 检索系统实际返回的文档 id 列表，**按相关度从高到低排序**
  （排名信息对 MRR / NDCG 至关重要，对 recall / precision 无影响）。
- relevant_ids: 人工标注的"该查询真正相关"的文档 id 集合（二值相关）。
- relevance_grades: {文档 id: 相关度等级}，等级为非负数值，越大越相关
  （分级相关，用于 NDCG）。
"""

from __future__ import annotations

import math


def retrieval_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """召回率（Recall）：相关文档中被检索出来的比例.

    公式::

        Recall = |retrieved ∩ relevant| / |relevant|

    直觉：相关文档一共就这么多，你找回了多大一块？召回率衡量系统的
    "查全"能力，完全不关心混进来了多少无关文档。

    - relevant 为空（该查询没有相关文档）时约定返回 1.0：
      没有该召回的东西，系统没有欠账。
    - retrieved 去重后参与计算，重复返回同一文档不算重复召回。
    """
    if not relevant_ids:
        return 1.0
    hits = set(retrieved_ids) & set(relevant_ids)
    return len(hits) / len(relevant_ids)


def retrieval_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """精确率（Precision）：检索结果中真正相关的比例.

    公式::

        Precision = |retrieved ∩ relevant| / |retrieved|

    直觉：你端上来的这一盘结果里，有多大比例是用户真正要的？
    精确率衡量系统的"查准"能力，完全不关心漏掉了多少相关文档。

    retrieved 为空时无法定义比例，约定返回 0.0（检索系统什么都没返回，
    对"结果可信度"而言记为零）。
    """
    if not retrieved_ids:
        return 0.0
    hits = set(retrieved_ids) & set(relevant_ids)
    return len(hits) / len(retrieved_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """倒数排名（Reciprocal Rank）：只看第一个命中出现在第几位.

    公式（单个查询）::

        RR = 1 / rank_of_first_relevant      （有命中）
        RR = 0                                （无命中）

    其中 rank 从 1 开始计数：第一个命中排第 1 位 → RR = 1.0；
    排第 2 位 → 0.5；第 3 位 → 0.333……

    直觉：对"导航型"查询（用户心里有一个明确答案，比如"X 的官方文档"），
    用户只关心正确答案出现得够不够靠前。第一个命中之后还有没有其他相关
    文档、以及它们排第几，本指标一概不看。

    严格地说 MRR（Mean Reciprocal Rank）是对多个查询的 RR 取平均::

        MRR = (1 / |Q|) * Σ_q RR_q

    本函数计算的是单查询 RR，跨查询的平均由 RagEvaluator 完成。
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: list[str], relevance_grades: dict[str, float]) -> float:
    """归一化折损累计增益（NDCG）：分级相关 + 位置折损的排序质量度量.

    分三步::

        DCG  = Σ_i (2^rel_i - 1) / log2(i + 1)     i 为排名，从 1 开始
        IDCG = 同样的公式，但按 rel 从大到小理想排序后计算
        NDCG = DCG / IDCG

    其中 rel_i 是第 i 位文档的相关度等级（未出现在 relevance_grades
    里的文档视为 rel = 0）。

    三个设计要点：

    1. **分级相关（graded relevance）**：recall / precision / MRR 都只有
       "相关 / 不相关"两档；NDCG 允许 0/1/2/3 这样的等级，
       "核心文档"排在前面比"边缘相关文档"排在前面得分高得多。
    2. **位置折损（discount）**：除以 log2(i + 1)，排名越靠后，
       同样的相关度贡献越小——模拟真实用户很少翻页的行为。
    3. **归一化（normalize）**：不同查询的相关文档数量天差地别，
       裸 DCG 无法跨查询比较；除以该查询的"理想排序 DCG"后，
       任何查询的 NDCG 都落在 [0, 1]，1.0 表示排序已经完美。

    relevance_grades 为空或全为 0 时 IDCG 为 0，NDCG 无定义，
    约定返回 0.0。
    """
    if not relevance_grades:
        return 0.0

    def _dcg(gains: list[float]) -> float:
        return sum(
            (2.0**rel - 1.0) / math.log2(rank + 1)
            for rank, rel in enumerate(gains, start=1)
        )

    actual = [relevance_grades.get(doc_id, 0.0) for doc_id in retrieved_ids]
    ideal = sorted(relevance_grades.values(), reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(actual) / idcg
