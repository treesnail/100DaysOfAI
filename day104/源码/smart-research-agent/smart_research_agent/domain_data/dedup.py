"""两级去重器：精确指纹先挡，MinHash 再挡近重复（M5-D8）.

day048 只有一级（精确指纹），day057 补上第二级。两级的分工是**成本**与
**确定性**的取舍，不是"更严 / 更松"：

| 级别 | 判据 | 成本 | 结论的性质 |
|------|------|------|-----------|
| 精确 | 归一化后 prompt 的 SHA-1 前 16 位 | O(1) 查表 | 确定：键相同必然重复 |
| 近重复 | MinHash 签名的 Jaccard 估计 | O(已入库条数) 逐条比较 | 估计：有方差，阈值需留裕度 |

**顺序不能反**：先精确后近似，才能保证"完全相同"这一档永远由确定性判据
裁决；若先跑近似，一条与已有样本逐字相同的样本也可能因为签名恰好差一位
而被放行，而它本该是 100% 确定的重复项。

``NearDuplicateIndex`` 支持**增量**：先把历史数据集 ``add`` 进索引，再
``filter`` 新批次，就得到了"跨批次去重"——这正是 M5-D8 要解决的核心问题，
"可持续更新的领域数据集"不能每来一批就把历史重读一遍。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.fingerprint import (
    DEFAULT_NUM_PERM,
    DEFAULT_SHINGLE_K,
    MINHASH_SEED,
    Fingerprint,
    fingerprint_example,
)
from smart_research_agent.finetune.schema import TrainingExample

#: 一条样本进索引后的判定结论（用常量而不是裸字符串，避免报表键拼错）
STATUS_KEPT = "kept"
STATUS_EXACT = "exact_duplicate"
STATUS_NEAR = "near_duplicate"

#: 近重复阈值。**这个值是本课在本课程语料上标定出来的，不是抄来的。**
#:
#: 很多资料给的近重复经验值是 0.8~0.9，但那对应的是**较长文档**的 shingle
#: 统计。字符级 shingle 在短中文提问上并不稳定：``k=3`` 时，
#: "如何评估 RAG 的检索质量？" 与 "请帮我看看：如何评估 RAG 的检索质量？"
#: （加 6 字前缀，明显是同一条语料）的**精确** Jaccard 只有 0.6842，
#: MinHash 估计 0.5469。阈值取 0.8 的实测后果是：在一份混入 16 条
#: 逐字复制 + 16 条前缀复制的语料上，**近重复一条都检不出来**（只有精确
#: 那 16 条被挡下），整条近重复链路静默空转。
#:
#: 本课实测的阈值敏感性（69 条 = 37 条真实样本 + 16 条逐字复制 + 16 条前缀复制）：
#:
#: ==========  ==========  ==========  ==========  ======================
#: threshold   保留条数    精确重复    近重复      说明
#: ==========  ==========  ==========  ==========  ======================
#: 0.6         40          16          13          抓到 13/16 前缀复制
#: **0.7**     **44**      **16**      **9**       37 条原样本零误判
#: 0.8         53          16          0           近重复链路空转
#: 0.9         53          16          0           同上
#: ==========  ==========  ==========  ==========  ======================
#:
#: 取 0.7 的理由：它在**不误伤 37 条真实样本**的前提下抓出 9/16 条前缀复制；
#: 再往下调（0.6）收益只多 4 条，而误判风险开始上升——数据工程里
#: "少删一条真重复"的代价远小于"删错一条真数据"。
DEFAULT_NEAR_DUP_THRESHOLD = 0.7


@dataclass
class DedupeDecision:
    """一条样本的去重决策：结论、与谁重复、相似度是多少."""

    example: TrainingExample
    status: str
    similarity: float
    partner_index: int | None = None

    @property
    def is_duplicate(self) -> bool:
        """是否被判为重复（精确或近重复）."""
        return self.status != STATUS_KEPT

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "status": self.status,
            "similarity": round(self.similarity, 4),
            "partner_index": self.partner_index,
            "source": self.example.source,
            "preview": self.example.prompt_text[:60],
        }


@dataclass
class DedupeReport:
    """去重报告：总量、两级各自的重复数、阈值与逐条决策.

    ``exact_duplicates + near_duplicates + kept == total`` 是一条**可断言的
    恒等式**（测试里钉住它）：三个数对不上，说明有样本在统计里丢失了，
    而"样本静默消失"是数据流水线里最难查的一类故障。
    """

    total: int
    kept: int
    exact_duplicates: int
    near_duplicates: int
    threshold: float
    decisions: list[DedupeDecision] = field(default_factory=list)

    @property
    def duplicates(self) -> int:
        """两级重复数之和."""
        return self.exact_duplicates + self.near_duplicates

    @property
    def keep_rate(self) -> float:
        """保留率（空输入按 0.0 计，与 day048 的 ``FilterReport`` 同一约定）."""
        return self.kept / self.total if self.total else 0.0

    def drop_reasons(self) -> dict[str, int]:
        """按去重级别汇总的丢弃原因计数（**只保留非零项**）."""
        counts = {
            STATUS_EXACT: self.exact_duplicates,
            STATUS_NEAR: self.near_duplicates,
        }
        return {name: count for name, count in counts.items() if count}

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典（含逐条决策的摘要）."""
        return {
            "total": self.total,
            "kept": self.kept,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "duplicates": self.duplicates,
            "threshold": self.threshold,
            "keep_rate": round(self.keep_rate, 4),
            "drop_reasons": self.drop_reasons(),
            "decisions": [
                decision.to_dict() for decision in self.decisions if decision.is_duplicate
            ],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"去重：{self.total} 条进 / {self.kept} 条留 | "
            f"精确重复 {self.exact_duplicates} / 近重复 {self.near_duplicates}"
            f"（阈值 {self.threshold}）"
        )


class NearDuplicateIndex:
    """两级去重索引：可增量 add，也可一次性 filter 一批样本.

    用法::

        index = NearDuplicateIndex(threshold=0.8)
        index.add_many(history_examples)          # 历史数据集先进索引
        kept, report = index.filter(new_batch)    # 新批次跨批次去重

    ``filter`` 会把保留的样本一并 ``add`` 进索引，因此**批内重复也顺手解决了**
    ——否则"同一批次里出现两次的样本"会双双通过。
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        k: int = DEFAULT_SHINGLE_K,
        num_perm: int = DEFAULT_NUM_PERM,
        seed: int = MINHASH_SEED,
    ):
        # 阈值必须落在 (0, 1]：0 等于"任何两条都算重复"，会把整批清空；
        # 大于 1 则没有任何样本能满足，近重复检测静默失效。
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"近重复阈值必须落在 (0, 1] 区间，收到 {threshold}")
        # ``k`` 与 ``num_perm`` 也在**构造期**校验（本包 ``errors.py`` 立的纪律）。
        # 若留给第一次 ``add`` 去报错，调用方拿到的异常会晚一步发生，
        # 而且堆栈里出现的是 ``shingles`` / ``permutation_params``——
        # 一个"参数写错了"的问题被伪装成"运行时出错"。
        if k < 1:
            raise DomainDataError(f"shingle 窗口 k 必须 >= 1，收到 {k}")
        if num_perm < 1:
            raise DomainDataError(f"置换个数 num_perm 必须 >= 1，收到 {num_perm}")
        self.threshold = threshold
        self.k = k
        self.num_perm = num_perm
        self.seed = seed
        self._fingerprints: list[Fingerprint] = []
        self._exact_keys: dict[str, int] = {}

    def __len__(self) -> int:
        """已入库的样本数."""
        return len(self._fingerprints)

    @property
    def fingerprints(self) -> list[Fingerprint]:
        """已入库指纹的只读副本（报告与诊断用）."""
        return list(self._fingerprints)

    def fingerprint(self, example: TrainingExample) -> Fingerprint:
        """按本索引的参数为样本生成指纹."""
        return fingerprint_example(
            example, k=self.k, num_perm=self.num_perm, seed=self.seed
        )

    def classify(self, example: TrainingExample) -> DedupeDecision:
        """判定样本与**已入库样本**的关系，但不入库（只读查询）."""
        candidate = self.fingerprint(example)
        if candidate.key in self._exact_keys:
            return DedupeDecision(
                example=example,
                status=STATUS_EXACT,
                similarity=1.0,
                partner_index=self._exact_keys[candidate.key],
            )
        best_index: int | None = None
        best_similarity = 0.0
        for index, known in enumerate(self._fingerprints):
            similarity = candidate.similarity(known)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index is not None and best_similarity >= self.threshold:
            return DedupeDecision(
                example=example,
                status=STATUS_NEAR,
                similarity=best_similarity,
                partner_index=best_index,
            )
        return DedupeDecision(example=example, status=STATUS_KEPT, similarity=best_similarity)

    def add(self, example: TrainingExample) -> DedupeDecision:
        """判定并入库：返回判定结论（重复样本**不入库**）.

        重复样本不入库是刻意的：入库后它会成为后续样本的比较对象，
        于是"重复的重复"也会被同一套判据挡掉——索引里只保留**已接受**的
        样本，是"保留首次出现"这条语义在增量场景下的自然延伸。
        """
        decision = self.classify(example)
        if not decision.is_duplicate:
            candidate = self.fingerprint(example)
            self._exact_keys[candidate.key] = len(self._fingerprints)
            self._fingerprints.append(candidate)
        return decision

    def add_many(self, examples: Sequence[TrainingExample]) -> int:
        """批量入库，返回其中被接受的条数."""
        accepted = 0
        for example in examples:
            if not self.add(example).is_duplicate:
                accepted += 1
        return accepted

    def filter(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[TrainingExample], DedupeReport]:
        """对一批样本做两级去重，返回（保留样本, 报告）.

        入参样本按**给定顺序**逐个判定：先到者优先保留（"保留首次出现"），
        因此同一批数据换个顺序会得到不同的保留集——这是确定性的代价，
        换来的是"不清空历史索引就能做跨批次去重"。要让结果稳定，
        批次内的顺序必须稳定（本课程的采集顺序由``DataCollector``固定）。
        """
        kept: list[TrainingExample] = []
        decisions: list[DedupeDecision] = []
        exact = 0
        near = 0
        for example in examples:
            decision = self.add(example)
            decisions.append(decision)
            if decision.status == STATUS_EXACT:
                exact += 1
            elif decision.status == STATUS_NEAR:
                near += 1
            else:
                kept.append(example)

        report = DedupeReport(
            total=len(examples),
            kept=len(kept),
            exact_duplicates=exact,
            near_duplicates=near,
            threshold=self.threshold,
            decisions=decisions,
        )
        return kept, report


def dedupe_examples(
    examples: Sequence[TrainingExample],
    *,
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    k: int = DEFAULT_SHINGLE_K,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = MINHASH_SEED,
) -> tuple[list[TrainingExample], DedupeReport]:
    """便捷入口：一次性的两级去重（内部新建空索引）.

    需要跨批次时请用 ``NearDuplicateIndex`` 显式管理索引——把历史样本
    重新 ``add`` 一遍虽然结果相同，但那是 O(历史) 的重复劳动，
    与"增量更新"的初衷相反。
    """
    index = NearDuplicateIndex(threshold=threshold, k=k, num_perm=num_perm, seed=seed)
    return index.filter(examples)
