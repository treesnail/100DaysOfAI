"""``rag_debug`` 的形状族：一次端到端评估里流动的四样东西（day071）.

```text
RagEvalCase    一条用例：问题 + 金标准（相关记录 id）+ 参考答案
StageMetrics   一段名单上的四个检索指标（"retrieved" / "packed" 两段各一份）
CaseOutcome    一次运行的结果：两段指标 + 空结果原因 + 生成侧的三组编号与回退
BadCase        一次归因：**一个**标签 + 一句人话解释 + **一个**下一步动作
```

四样东西的顺序就是这一课的逻辑顺序：**用例 → 运行 → 指标 → 归因**；
而本模块里最该被记住的是三张**封闭表**（``BAD_CASE_TAGS`` /
``BAD_CASE_DESCRIPTIONS`` / ``BAD_CASE_ACTIONS``），它们把 day070 教程第三章
那张"人工诊断表"变成可以统计的代码：

```text
先出现的先怀疑        BAD_CASE_PRIORITY 的排列 = 链路的顺序
每一类坏例对应一个动作  BAD_CASE_ACTIONS   = day070 第三章每一行右边那句话
```

## 为什么指标要分"两段"

同一个查询在链路上有两个**不同的名单**：

```text
retrieved   检索器交出的那份（day066~day068 的全部努力都在这里）
packed      真正进了提示词的那份（day069 的 given——预算丢尾之后剩下的）
```

金标准"在 retrieved 里但不在 packed 里"是一门**独立的失败**：
它既不是检索的错（检索抓到了），也不是生成的错（生成没机会看到），
而是**预算的错**。合成一个召回率之后，这门失败会消失在均值里——
而它的处置动作（调 `retrieval_max_context_chars`）与另两种完全不同。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.evaluation.rag_metrics import ndcg, retrieval_precision
from smart_research_agent.rag_debug.errors import CaseDataError, DiagnosisError
from smart_research_agent.retrieval.generation import (
    FALLBACK_REASON_DESCRIPTIONS,
    FALLBACK_REASON_EMPTY_REPLY,
    FALLBACK_REASON_LLM_ERROR,
    FALLBACK_REASON_MODEL_DECLINED,
    FALLBACK_REASON_NO_CONTEXT,
    FALLBACK_REASON_NONE,
    FALLBACK_REASON_UNUSABLE_CITATIONS,
)
from smart_research_agent.retrieval.rerank import ndcg_at_k, recall_at_k, reciprocal_rank
from smart_research_agent.retrieval.types import EMPTY_REASONS

# --------------------------------------------------------------------------- #
# 常量：两段名单、四个指标、一个覆盖率下限
# --------------------------------------------------------------------------- #

#: 检索器交出的那份名单（day066~day068 的账）.
STAGE_RETRIEVED = "retrieved"

#: 真正进了提示词的那份名单（day069 的账：预算丢尾之后剩下的）.
STAGE_PACKED = "packed"

#: 两段名单（顺序 = 链路的顺序，报告里按它排列）.
STAGES: tuple[str, ...] = (STAGE_RETRIEVED, STAGE_PACKED)

#: 每一段名单上算的四个指标（名字与 ``retrieval.rerank`` 那三个函数同源，
#: 只有 ``precision`` 来自 ``evaluation.rag_metrics``——它是 day029 就在的纯函数）。
RETRIEVAL_METRICS: tuple[str, ...] = (
    "recall",
    "precision",
    "reciprocal_rank",
    "ndcg",
)

#: 覆盖率的下限（低于它算"低覆盖"坏例）——它是召回之外的第二个质量闸门.
DEFAULT_MIN_COVERAGE = 0.5

#: "金标准进了提示词但排得太靠后"的判据：packed 的倒数排名低于它即算（0.5 = 前 2 名之外）.
DEFAULT_MIN_RECIPROCAL_RANK = 0.5

#: 与参考答案对比的忠实度下限（低于它算弱忠实度坏例）.
DEFAULT_MIN_FAITHFULNESS = 0.5

# --------------------------------------------------------------------------- #
# 坏例标签：一张封闭清单 + 两张对齐的表
# --------------------------------------------------------------------------- #

#: 库本身是空的（评测跑在了一个没灌语料的库上）——它不是模型问题.
TAG_NO_DATA = "no_data"

#: 检索为空（其余三种 empty_reason）——处置动作看那一个原因本身.
TAG_RETRIEVAL_EMPTY = "retrieval_empty"

#: 金标准一条都没进检索名单（抓错了）.
TAG_RETRIEVAL_MISS = "retrieval_miss"

#: 有金标准被**重排阈值**切掉（第四道减法）.
TAG_RERANK_CUT = "rerank_cut"

#: 金标准在名单里，但**没进提示词**（预算丢尾）——这一课独有的那门失败.
TAG_PACKED_AWAY = "packed_away"

#: 金标准进了提示词，但排在很靠后（排序问题）.
TAG_RANK_BAD = "rank_bad"

#: 片段被单块上限截断，且覆盖率偏低（"少半句结论"型失败）.
TAG_TRUNCATED_CHUNK = "truncated_chunk"

#: 检索非空、但答案走了回退（六种原因各自一个动作）.
TAG_GENERATION_FALLBACK = "generation_fallback"

#: 答案引用了**不存在的编号**（幻觉引用）.
TAG_HALLUCINATED_CITATION = "hallucinated_citation"

#: 答案一个有效引用都没有（模型自由发挥了，或者提示词没逼出引用）.
TAG_UNGROUNDED = "ungrounded"

#: 覆盖率低于下限（给了很多片段，用上的很少）.
TAG_LOW_COVERAGE = "low_coverage"

#: 与参考答案的忠实度低于下限.
TAG_WEAK_FAITHFULNESS = "weak_faithfulness"

#: 坏例标签的**封闭清单**，排列 = 归因优先级 = **链路的顺序**（先出现的先怀疑）.
#: 归因函数逐条判、命中即返回，因此这张表的顺序就是"先查检索、再查打包、
#: 最后查生成"这条纪律在代码里的位置（见 ``diagnose.diagnose``）。
BAD_CASE_TAGS: tuple[str, ...] = (
    TAG_NO_DATA,
    TAG_RETRIEVAL_EMPTY,
    TAG_RETRIEVAL_MISS,
    TAG_RERANK_CUT,
    TAG_PACKED_AWAY,
    TAG_RANK_BAD,
    TAG_TRUNCATED_CHUNK,
    TAG_GENERATION_FALLBACK,
    TAG_HALLUCINATED_CITATION,
    TAG_UNGROUNDED,
    TAG_LOW_COVERAGE,
    TAG_WEAK_FAITHFULNESS,
)

#: 归因优先级（与 ``BAD_CASE_TAGS`` 同一份排列，单列出来是为了让"顺序有含义"
#: 这件事在导入本模块时就能被读到；两者不一致会在导入期报错）。
BAD_CASE_PRIORITY: tuple[str, ...] = BAD_CASE_TAGS

#: 每一类坏例"是什么问题"（一句话，进报告与 `--explain`）.
BAD_CASE_DESCRIPTIONS: dict[str, str] = {
    TAG_NO_DATA: "库是空的：这次评估根本没有可检索的语料",
    TAG_RETRIEVAL_EMPTY: "检索为空：过滤/阈值/多样性把候选清空了",
    TAG_RETRIEVAL_MISS: "检索没抓到：金标准一条都没进名单",
    TAG_RERANK_CUT: "重排阈值把金标准切掉了（第四道减法）",
    TAG_PACKED_AWAY: "抓到了但没进提示词：被上下文预算从尾部丢掉",
    TAG_RANK_BAD: "进了提示词但排得太靠后（排序问题）",
    TAG_TRUNCATED_CHUNK: "片段被单块上限截断，依据是半截的",
    TAG_GENERATION_FALLBACK: "有片段，但这次答案走了回退",
    TAG_HALLUCINATED_CITATION: "答案引用了不存在的编号（幻觉引用）",
    TAG_UNGROUNDED: "答案没有任何有效引用（无从核对）",
    TAG_LOW_COVERAGE: "覆盖率低：给了不少片段，用上的很少",
    TAG_WEAK_FAITHFULNESS: "与参考答案对比，忠实度偏低",
}

#: 每一类坏例"下一步做什么"（可执行动作，且**只动一个旋钮**）.
BAD_CASE_ACTIONS: dict[str, str] = {
    TAG_NO_DATA: "先灌语料 / 建索引：库为空是评测的前置条件，不是模型问题",
    TAG_RETRIEVAL_EMPTY: "看 empty_reason 那一个原因：filtered_out 放宽过滤、"
    "below_threshold 调 retrieval_min_score、diversity 调 retrieval_max_per_doc、"
    "no_data 先建索引",
    TAG_RETRIEVAL_MISS: "查检索侧第 ③④ 层：换编码器必须重建索引、看分块粒度是否把答案切碎、"
    "确认索引版本是否包含这批资料",
    TAG_RERANK_CUT: "调 retrieval_rerank_min_score（或先把重排阈值关掉再量一次）",
    TAG_PACKED_AWAY: "调 retrieval_max_context_chars / retrieval_per_hit_chars："
    "这条是预算的错，不是检索或生成的错",
    TAG_RANK_BAD: "看排序侧：换/关重排（retrieval_rerank_enabled）、"
    "或换融合策略（retrieval_hybrid_strategy）",
    TAG_TRUNCATED_CHUNK: "调大 retrieval_per_hit_chars（截断是单块上限的账）",
    TAG_GENERATION_FALLBACK: "按 fallback_reason 选动作（见 FALLBACK_ACTIONS 那张表）",
    TAG_HALLUCINATED_CITATION: "收紧提示词里的引用要求，或换更强/别家的模型："
    "幻觉引用指向一份不存在的依据",
    TAG_UNGROUNDED: "先看提示词是否明确要求 `[n]` 引用；确认无误再考虑打开 "
    "retrieval_require_citation（它是闸门，打开后不可核对的答案会被整条丢掉）",
    TAG_LOW_COVERAGE: "先看检索给的片段贴不贴题（unused 是覆盖率的分母）；"
    "再用 retrieval_max_context_chars 减掉尾部噪音",
    TAG_WEAK_FAITHFULNESS: "逐句对照参考答案：生成侧（提示词/模型）与检索侧各核一次",
}

#: 六种回退各自的处置动作（第三种回退 ``no_context`` 属于检索层，不是生成层）.
FALLBACK_ACTIONS: dict[str, str] = {
    FALLBACK_REASON_NO_CONTEXT: "回到检索层：这次连片段都没有（本层不该出现这个原因）",
    FALLBACK_REASON_LLM_ERROR: "查提供方（超时/限流/密钥）；装配层可加退避重试，"
    "但不要在评估里静默重试——那会让这一条用例的失败被抹掉",
    FALLBACK_REASON_EMPTY_REPLY: "模型返回了空文本：查 max_tokens 是否被截到 0 附近，"
    "或换一个模型",
    FALLBACK_REASON_MODEL_DECLINED: "模型按固定句式自己拒答：可能是提示词对"
    "'有片段就必须作答'的要求不够硬，也可能是片段确实不贴题（先看 recall）",
    FALLBACK_REASON_UNUSABLE_CITATIONS: "开了 require_citation 而模型没写出可用编号："
    "先看提示词是否明确要求编号格式",
    FALLBACK_REASON_NONE: "没有回退：这一条不该出现在这里",
}

# 三张表必须逐键对齐：少一个键时"这一类的下一步"就无人回答，而那种缺失
# 在报告里表现为一片空白（最不该发生的一类沉默）。
if set(BAD_CASE_DESCRIPTIONS) != set(BAD_CASE_TAGS) or set(BAD_CASE_ACTIONS) != set(
    BAD_CASE_TAGS
):
    raise DiagnosisError(
        "坏例标签的三张表不一致："
        f"tags={sorted(BAD_CASE_TAGS)}、"
        f"descriptions={sorted(BAD_CASE_DESCRIPTIONS)}、"
        f"actions={sorted(BAD_CASE_ACTIONS)}。"
        "三张表必须逐键对齐（同一份封闭清单的三种投影），否则某一类坏例"
        "会在报告里只有名字、没有解释、也没有下一步。"
    )
if set(FALLBACK_ACTIONS) != {*FALLBACK_REASON_DESCRIPTIONS}:
    raise DiagnosisError(
        "FALLBACK_ACTIONS 必须覆盖 FALLBACK_REASON_DESCRIPTIONS 的全部键"
        "（含表示'没有回退'的空串）："
        f"actions={sorted(FALLBACK_ACTIONS)}、"
        f"reasons={sorted(FALLBACK_REASON_DESCRIPTIONS)}。"
        "回退原因是**同一份封闭清单**，少一个键就意味着那种回退没有处置动作。"
    )


def described_tags() -> dict[str, dict[str, str]]:
    """坏例标签的完整三件套（标签 → {description, action}），供端点与演示脚本展示."""
    return {
        tag: {
            "description": BAD_CASE_DESCRIPTIONS[tag],
            "action": BAD_CASE_ACTIONS[tag],
        }
        for tag in BAD_CASE_TAGS
    }


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RagEvalCase:
    """一条评测用例：**问题 + 金标准 + 参考答案**（金标准不可为空）.

    ```text
    query       被问的那句话（它会被真的送进检索器）
    relevant    金标准：这次问题**真正相关**的记录 id（tuple，不能为空）
    reference   参考答案（可空：只评检索时不需要它）
    grades      {记录 id: 相关度等级}（可空 → NDCG 退化为二值版本）
    ```

    ``relevant`` 必须是 **tuple** 且非空，理由与 ``LiftProbe.relevant`` 逐字相同：

```text
必须是 tuple   frozen 形状里塞一个 list，"这份金标准不曾被改过"就不再成立
不能为空       四个指标的分母都由它给出，空标注会让 recall@k 变成 0/0——
              而它算出来的 0.0 看起来像是"一条都没召回到"（一个假阴性）
```

    ``grades`` 与 ``relevant`` 是**两张不同的表**，服务不同的指标：

```text
relevant   二值相关集合 → recall / precision / MRR 的判据
grades     分级相关表   → NDCG 的判据（理想排序 IDCG 也由它算）
```

    因此 ``grades`` 的键**可以**超出 ``relevant``：项目自带的评测集里就有这种
    标注（``"relevant_ids": ["rag-001"]`` 而 ``"relevance_grades"`` 里还带着
    ``{"rag-002": 1}``）——等级 1 表示"部分相关"，它不计入 recall 的二值口径，
    却参与 NDCG 的位置折损。硬把它们对齐会**毁掉一份真实的标注**，
    而正确读法是把两张表分别喂给各自的指标（day029 的四个纯函数就是这么分工的）。

    唯一的硬约束是"等级表不能全是 0"：IDCG 为 0 时 NDCG 恒为 0，而那个 0
    读起来像"排序很差"，实际是一张标错的等级表。``grades`` 留空时用
    ``relevant`` 自动生成等级 1（二值版本，见 ``measure_ids`` 复用的
    ``retrieval.rerank.ndcg_at_k``）。
    """

    query: str
    relevant: tuple[str, ...]
    reference: str = ""
    grades: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise CaseDataError(
                f"RagEvalCase.query 必须是非空字符串，收到 {self.query!r}："
                "用例要被真的送进检索器，空查询在检索那一层本来就会被拒。"
            )
        if not isinstance(self.relevant, tuple):
            raise CaseDataError(
                f"RagEvalCase.relevant 必须是 tuple，收到 {type(self.relevant).__name__}。"
                "写法：relevant=('rag-001',)；frozen 形状里塞一个 list 会让"
                "'这份金标准不曾被改过'不再成立。"
            )
        if not self.relevant:
            raise CaseDataError(
                f"RagEvalCase.relevant 不能为空（query={self.query!r}）："
                "四个检索指标的分母都由它给出，空标注会让 recall@k 变成 0/0，"
                "而 0.0 看起来像是'一条都没召回到'。"
                "出路：给这条用例标出至少一条相关记录，或把它从评测集里删掉。"
            )
        for record_id in self.relevant:
            if not isinstance(record_id, str) or not record_id.strip():
                raise CaseDataError(
                    f"RagEvalCase.relevant 里出现了非字符串或空串：{record_id!r}。"
                    "金标准里存的是记录 id，空 id 永远匹配不上任何命中。"
                )
        if not isinstance(self.reference, str):
            raise CaseDataError(
                f"RagEvalCase.reference 必须是字符串，收到 "
                f"{type(self.reference).__name__}（空串 = 这次不评忠实度）。"
            )
        if not isinstance(self.grades, dict):
            raise CaseDataError(
                f"RagEvalCase.grades 必须是字典，收到 {type(self.grades).__name__}。"
            )
        for record_id in self.grades:
            if not isinstance(record_id, str) or not record_id.strip():
                raise CaseDataError(
                    f"RagEvalCase.grades 里出现了非字符串或空串键：{record_id!r}。"
                    "等级表的键是记录 id，空 id 永远匹配不上任何命中。"
                )
        for record_id, grade in self.grades.items():
            if isinstance(grade, bool) or not isinstance(grade, (int, float)):
                raise CaseDataError(
                    f"RagEvalCase.grades[{record_id!r}] 必须是数字，"
                    f"收到 {type(grade).__name__}（相关度等级越高越相关）。"
                )
            number = float(grade)
            if number != number or number < 0.0 or number == float("inf"):
                raise CaseDataError(
                    f"RagEvalCase.grades[{record_id!r}]={grade!r} 必须是非负有限数："
                    "nan 会让 NDCG 的分子变成 nan，而 nan 的比较永远为假——"
                    "排序错误的记录会静默地不被计入任何一类。"
                )
        if self.grades and max(self.grades.values()) <= 0.0:
            raise CaseDataError(
                f"RagEvalCase.grades 全是 0（query={self.query!r}）："
                "IDCG 会变成 0，NDCG 恒为 0——而那个 0 读起来像'排序很差'，"
                "实际是一张标错的等级表。出路：给核心文档标上正等级，"
                "或把 grades 整个留空（留空时 NDCG 退化为二值版本，用 relevant 生成等级 1）。"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "query": self.query,
            "relevant": list(self.relevant),
            "reference": self.reference,
            "grades": {key: self.grades[key] for key in sorted(self.grades)},
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"用例 {self.query!r} | 金标准 {len(self.relevant)} 条：{list(self.relevant)}"
            f" | 参考答案 {'有' if self.reference else '无'}"
        )


def measure_ids(
    ids: tuple[str, ...] | list[str],
    relevant: tuple[str, ...],
    k: int,
    *,
    grades: dict[str, float] | None = None,
) -> dict[str, float]:
    """在一段名单上算四个检索指标（纯函数，入参里没有一个来自全局状态）.

    ``recall@k`` / ``RR`` / ``nDCG`` 三个直接复用 ``retrieval.rerank`` 的纯函数
    （day068 的成果），``precision`` 用 ``evaluation.rag_metrics.retrieval_precision``
    （day029 的成果）——它只吃**前 k 条**：不然一个 20 条的名单会因为"分母大"
    而拿到更低的精确率，而它其实只是"给得多"。

    ``k`` 是两份名单共用的那个 k（与 ``LiftReport.k`` 同一条纪律：
    三条指标共用一个 k，因此它只写一次）。

    ## NDCG 有两个口径，按**标注形态**选一个

```text
有分级标注（grades 非空）  → evaluation.rag_metrics.ndcg(前 k 条, grades)
                            分级增益 + 位置折损，理想排序由 grades 自己算
没有分级标注               → retrieval.rerank.ndcg_at_k(ids, relevant, k)
                            二值增益（命中记 1），理想排序由金标准条数算
```

    两个函数都是既有代码里的实现（day029 / day068），本层不重写任何一个。
    但**同一批用例的标注形态必须一致**——否则 ``mean_ndcg`` 会变成两个口径的
    平均，而报告里看不出这件事（这一条由 ``RagEvalSuite`` 在装配时校验）。
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise CaseDataError(
            f"k 必须是 >= 1 的整数，收到 {k!r}："
            "四个指标都带着这个 k（recall@k / precision@k / nDCG@k），k=0 会让它们恒为 0。"
        )
    head = list(ids)[:k]
    graded = dict(grades or {})
    return {
        "recall": recall_at_k(ids, relevant, k),
        "precision": retrieval_precision(head, relevant),
        "reciprocal_rank": reciprocal_rank(ids, relevant),
        "ndcg": ndcg(head, graded) if graded else ndcg_at_k(ids, relevant, k),
    }


@dataclass(frozen=True)
class StageMetrics:
    """一段名单上的四个指标（``stage`` 只取 ``STAGES`` 里的两个值之一）.

    **名单本身也要存下来**（``ids``）：报告里"这一段的召回是 0.5"这句话
    在被追问时只有一个可核对的答案——把 id 列表摆出来。
    只存分数的报告无法回答"缺的是哪一条"，而那正是排查的起点。
    """

    stage: str
    ids: tuple[str, ...]
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise DiagnosisError(
                f"StageMetrics.stage 只能是 {list(STAGES)} 之一，收到 {self.stage!r}："
                "两段名单各有各的名字（retrieved / packed），自由文本会让"
                "'进了提示词的召回率'与'检索召回率'混成一行。"
            )
        if not isinstance(self.ids, tuple):
            raise DiagnosisError(
                f"StageMetrics.ids 必须是 tuple，收到 {type(self.ids).__name__}"
                "（frozen 形状里塞一个 list，'这份名单不曾被改过'就不再成立）。"
            )
        for record_id in self.ids:
            if not isinstance(record_id, str) or not record_id.strip():
                raise DiagnosisError(
                    f"StageMetrics.ids 里出现了非字符串或空串：{record_id!r}"
                )
        for name in RETRIEVAL_METRICS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DiagnosisError(
                    f"StageMetrics.{name} 必须是数字，收到 {type(value).__name__}"
                )
            number = float(value)
            if number != number or not 0.0 <= number <= 1.0:
                raise DiagnosisError(
                    f"StageMetrics.{name}={value!r} 必须落在 [0, 1]："
                    "四个指标都是比例，越界说明分母算错了"
                    "（例如把检索深度当成了金标准条数）。"
                )

    @property
    def count(self) -> int:
        """这一段名单有几条."""
        return len(self.ids)

    def value(self, name: str) -> float:
        """按名字取一个指标（名字必须是 ``RETRIEVAL_METRICS`` 里的）."""
        if name not in RETRIEVAL_METRICS:
            raise DiagnosisError(
                f"不认识的指标名 {name!r}：可用的是 {list(RETRIEVAL_METRICS)}。"
            )
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "stage": self.stage,
            "count": self.count,
            "ids": list(self.ids),
            **{name: round(self.value(name), 4) for name in RETRIEVAL_METRICS},
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        rendered = "、".join(
            f"{name} {self.value(name):.4f}" for name in RETRIEVAL_METRICS
        )
        return f"{self.stage}（{self.count} 条）：{rendered}"


@dataclass(frozen=True)
class CaseOutcome:
    """一次端到端运行的结果：**两段指标 + 生成侧的全部可核对数字**（day071）.

    ```text
    query             被问的那句话（与 RetrievalQuery.text 同一份）
    relevant          这一条的金标准（从用例搬来，报告不必再回查用例表）
    retrieved         检索名单那一段的四个指标
    packed            进了提示词那一段的四个指标
    empty_reason      检索的空结果原因（非空时是 EMPTY_REASONS 之一）
    llm_called        这一次到底调没调模型（护栏的唯一证据，day069 的字段）
    fallback_reason   六取一（或空串 = 没走回退）
    grounded          答案有没有可核对的依据（day069 的判定，不看字符串）
    coverage          有效引用 / 给出去的片段数
    given / cited     给出去几条、答案引用了几条（覆盖率的分母与分子）
    valid_count       有效引用的条数（覆盖率真正的分子，day069 的 len(valid)）
    unused_count      给了却没被引用的条数（day069 的 len(unused)）
    hallucinated      幻觉引用的条数（invalid 的条数）
    truncated_hits     被单块上限截断的条数（"少半句结论"的账）
    dropped_hits      被预算从尾部丢掉的条数（"少几条依据"的账）
    dropped_by_rerank 被**重排阈值**切掉的条数（第四道减法，day068）
    dropped_by_top_k  被 top_k 截断丢掉的条数（"排序对了但名额不够"的账）
    answer_chars      答案长度（只留长度不留正文：报告是账本不是档案）
    latency_ms        这一次花了多久
    faithfulness      与参考答案的对比评分（None = 没评）
    notes             降级注记（例如"评审模型输出无法解析，已跳过"）
    ```

    ``answer_chars`` 而不是 ``answer``：本层是**账本**，不是存档。
    答案正文属于那一次运行的产物（它会进 ``outputs/`` 或调用方的日志），
    而报告里存一段几百字的答案会把"这一批坏在哪"这个问题淹掉
    （与 ``RetrievalResult.to_dict`` 默认不含正文是同一条纪律）。

    ``notes`` 的用途只有一个：**降级必须留痕**。今天只有一种降级
    （评审模型输出无法解析 → ``faithfulness=None``），但它必须被写出来——
    否则"这一批的忠实度均值"里会多出一个看不见的空洞。
    """

    query: str
    relevant: tuple[str, ...]
    retrieved: StageMetrics
    packed: StageMetrics
    empty_reason: str = ""
    llm_called: bool = False
    fallback_reason: str = FALLBACK_REASON_NONE
    grounded: bool = False
    coverage: float = 0.0
    given: int = 0
    cited: int = 0
    valid_count: int = 0
    unused_count: int = 0
    hallucinated: int = 0
    truncated_hits: int = 0
    dropped_hits: int = 0
    dropped_by_rerank: int = 0
    dropped_by_top_k: int = 0
    answer_chars: int = 0
    latency_ms: float = 0.0
    faithfulness: float | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise CaseDataError(
                f"CaseOutcome.query 必须是非空字符串，收到 {self.query!r}"
            )
        if not isinstance(self.retrieved, StageMetrics) or not isinstance(
            self.packed, StageMetrics
        ):
            raise DiagnosisError(
                "CaseOutcome 的两段指标必须是 StageMetrics"
                f"（收到 {type(self.retrieved).__name__} / {type(self.packed).__name__}）："
                "它们是「两段名单各一份」的那件事本身。"
            )
        if self.retrieved.stage != STAGE_RETRIEVED or self.packed.stage != STAGE_PACKED:
            raise DiagnosisError(
                "CaseOutcome 的两段指标装反了：retrieved 位上是 "
                f"{self.retrieved.stage!r}、packed 位上是 {self.packed.stage!r}。"
                "装反之后 packed_away 这类归因会指向相反的层——"
                "而报告里的数字全都'看起来正常'。"
            )
        if self.empty_reason and self.empty_reason not in EMPTY_REASONS:
            raise CaseDataError(
                f"CaseOutcome.empty_reason 只能是空串或 {list(EMPTY_REASONS)} 之一，"
                f"收到 {self.empty_reason!r}：它来自 RetrievalResult.empty_reason，"
                "自由文本会让'这一批里空结果的成因分布'无法统计。"
            )
        if not isinstance(self.fallback_reason, str) or (
            self.fallback_reason not in FALLBACK_REASON_DESCRIPTIONS
        ):
            raise CaseDataError(
                f"CaseOutcome.fallback_reason 只能是 "
                f"{list(FALLBACK_REASON_DESCRIPTIONS)} 之一，收到 "
                f"{self.fallback_reason!r}：它与 generation.FALLBACK_REASONS 是"
                "**同一份封闭清单**（空串表示没有回退）。"
            )
        for name in ("llm_called", "grounded"):
            if not isinstance(getattr(self, name), bool):
                raise CaseDataError(
                    f"CaseOutcome.{name} 必须是布尔值，收到 "
                    f"{type(getattr(self, name)).__name__}"
                )
        if isinstance(self.coverage, bool) or not isinstance(self.coverage, (int, float)):
            raise CaseDataError(
                f"CaseOutcome.coverage 必须是数字，收到 {type(self.coverage).__name__}"
            )
        number = float(self.coverage)
        if number != number or not 0.0 <= number <= 1.0:
            raise CaseDataError(
                f"CaseOutcome.coverage={self.coverage!r} 必须落在 [0, 1]："
                "它是比例（有效引用 / 给出去的片段数）。"
            )
        for name in (
            "given",
            "cited",
            "valid_count",
            "unused_count",
            "hallucinated",
            "truncated_hits",
            "dropped_hits",
            "dropped_by_rerank",
            "dropped_by_top_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CaseDataError(
                    f"CaseOutcome.{name} 必须是非负整数，收到 {value!r}："
                    "这四个数字是账本（给了几条 / 引了几条 / 幻觉几条 / 截断几条），"
                    "负数说明上游的账算错了。"
                )
        if self.valid_count + self.unused_count != self.given:
            raise CaseDataError(
                f"CaseOutcome 的账不自洽：valid_count({self.valid_count}) + "
                f"unused_count({self.unused_count}) 应当等于 given({self.given})。"
                "这三条来自同一份 GroundingReport：'落在提示词里的'与"
                "'给了没引用的'恰好把 1..n 分完，两者相加才是给出去的片段数。"
            )
        if self.valid_count > self.cited:
            raise CaseDataError(
                f"CaseOutcome.valid_count({self.valid_count}) 不可能大于 "
                f"cited({self.cited})：有效引用是答案引用集合的子集。"
            )
        if self.cited > self.given + self.hallucinated:
            raise CaseDataError(
                f"CaseOutcome 的账不自洽：cited={self.cited} 大于 "
                f"given({self.given}) + hallucinated({self.hallucinated})。"
                "答案里的每个编号要么落在提示词里（given 之内）、"
                "要么落不到（幻觉引用），两者之和就是 cited 的上界。"
            )
        for name in ("answer_chars",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CaseDataError(
                    f"CaseOutcome.{name} 必须是非负整数，收到 {value!r}"
                )
        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise CaseDataError(
                f"CaseOutcome.latency_ms 必须是数字，收到 {type(self.latency_ms).__name__}"
            )
        elapsed = float(self.latency_ms)
        if elapsed != elapsed or elapsed < 0.0 or elapsed == float("inf"):
            raise CaseDataError(
                f"CaseOutcome.latency_ms 必须是非负有限数，收到 {self.latency_ms!r}"
            )
        if self.faithfulness is not None:
            if isinstance(self.faithfulness, bool) or not isinstance(
                self.faithfulness, (int, float)
            ):
                raise CaseDataError(
                    f"CaseOutcome.faithfulness 必须是数字或 None，收到 "
                    f"{type(self.faithfulness).__name__}（None 表示这次没评）"
                )
            score = float(self.faithfulness)
            if score != score or not 0.0 <= score <= 1.0:
                raise CaseDataError(
                    f"CaseOutcome.faithfulness={self.faithfulness!r} 必须落在 [0, 1]"
                    "（它是与参考答案的对比评分，越界说明评审那一层的值域校验漏了）。"
                )
        if not isinstance(self.notes, tuple):
            raise CaseDataError(
                f"CaseOutcome.notes 必须是 tuple，收到 {type(self.notes).__name__}"
            )
        for note in self.notes:
            if not isinstance(note, str) or not note.strip():
                raise CaseDataError(
                    f"CaseOutcome.notes 里出现了空条目 {note!r}："
                    "注记是给人读的，空串只会让报告里多一行空白。"
                )

    @property
    def packed_away(self) -> int:
        """金标准里"被预算丢掉"的条数（**这一课的核心新指标**）.

        定义：``packed.recall`` 与 ``retrieved.recall`` 的差额 × 金标准条数。
        用**条数**而不是比例，是因为它要与人对话："这一批里有 12 条依据
        被预算丢在了尾部"比"packed recall 低了 0.15"更像一句待办事项。
        """
        lost = (self.retrieved.recall - self.packed.recall) * len(self.relevant)
        return int(round(lost))

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "query": self.query,
            "relevant": list(self.relevant),
            "retrieved": self.retrieved.to_dict(),
            "packed": self.packed.to_dict(),
            "packed_away": self.packed_away,
            "empty_reason": self.empty_reason,
            "llm_called": self.llm_called,
            "fallback_reason": self.fallback_reason,
            "grounded": self.grounded,
            "coverage": round(self.coverage, 4),
            "given": self.given,
            "cited": self.cited,
            "valid_count": self.valid_count,
            "unused_count": self.unused_count,
            "hallucinated": self.hallucinated,
            "truncated_hits": self.truncated_hits,
            "dropped_hits": self.dropped_hits,
            "dropped_by_rerank": self.dropped_by_rerank,
            "dropped_by_top_k": self.dropped_by_top_k,
            "answer_chars": self.answer_chars,
            "latency_ms": round(float(self.latency_ms), 4),
            "faithfulness": (
                None if self.faithfulness is None else round(float(self.faithfulness), 4)
            ),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（把十个数字压在一行里）."""
        state = "接地" if self.grounded else "未接地"
        fallback = self.fallback_reason or "无回退"
        tail = "" if self.faithfulness is None else f" | 忠实度 {self.faithfulness:.2f}"
        return (
            f"{self.query[:24]} | 召回 {self.retrieved.recall:.2f}→"
            f"{self.packed.recall:.2f} | {state}（覆盖 {self.coverage:.0%}）"
            f" | 幻觉 {self.hallucinated} | {fallback} | {self.latency_ms:.1f}ms{tail}"
        )


@dataclass(frozen=True)
class BadCase:
    """一次归因：**一个**标签 + 一句解释 + **一个**动作（day071）.

    ```text
    query      哪条用例
    tag        坏在哪一类（BAD_CASE_TAGS 里的一个）
    reason     人话解释（这一类是什么问题，含这一条的具体数字）
    action     下一步动作（**只动一个旋钮**）
    evidence   支撑这次归因的那几个数字（字段名 → 值）
    ```

    为什么只给**一个**标签：一条坏例往往同时满足多个判据（既"召回为 0"
    又"没有引用"），但它们的**因果顺序是固定的**——检索没抓到的时候，
    "答案没有引用"只是它的后果。一次给出全部标签会把因果关系摊平成
    一张清单，而读的人只会去修最后一个（最显眼的那个）。

    ``evidence`` 存的都是**已经存在于 CaseOutcome 上的数字**
    （`retrieved.recall`、`dropped_hits`、`hallucinated` ...），不新增任何
    需要重新计算的量：归因是"读账"，不是"再算一次账"。
    """

    query: str
    tag: str
    reason: str
    action: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise CaseDataError(
                f"BadCase.query 必须是非空字符串，收到 {self.query!r}："
                "一条不知属于哪次提问的坏例无法被复核。"
            )
        if self.tag not in BAD_CASE_TAGS:
            raise DiagnosisError(
                f"不认识的坏例标签 {self.tag!r}：可用的是 {list(BAD_CASE_TAGS)}。"
                "标签是一张**封闭清单**——自由文本会让'这一批坏例里各类各有多少'"
                "这个问题无法统计（与 FALLBACK_REASONS 同一条纪律）。"
            )
        for name in ("reason", "action"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DiagnosisError(
                    f"BadCase.{name} 必须是非空字符串："
                    "归因的全部价值在于'是什么问题'与'下一步做什么'这两句话，"
                    "缺任何一句，这条记录都只剩一个标签。"
                )
        if not isinstance(self.evidence, dict):
            raise DiagnosisError(
                f"BadCase.evidence 必须是字典，收到 {type(self.evidence).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "query": self.query,
            "tag": self.tag,
            "description": BAD_CASE_DESCRIPTIONS[self.tag],
            "reason": self.reason,
            "action": self.action,
            "evidence": {key: self.evidence[key] for key in sorted(self.evidence)},
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return f"[{self.tag}] {self.query[:24]} —— {self.reason} → {self.action}"


def bad_case_rate(bad_cases: int, cases: int) -> float:
    """坏例占比（``cases`` 为 0 时返回 0.0，而不是除零异常）.

    与 ``aggregate_results`` 的空批次、``mean()`` 的"没有数据应得 0 分"
    是同一条纪律；但"评了 0 条"这件事会如实写在报告的 ``cases`` 字段里——
    **0 分与"评了 0 条"必须一起被看见**。
    """
    if isinstance(bad_cases, bool) or isinstance(cases, bool):
        raise CaseDataError("bad_case_rate 的入参必须是整数")
    if not isinstance(bad_cases, int) or not isinstance(cases, int) or bad_cases < 0 or cases < 0:
        raise CaseDataError("bad_case_rate 的入参必须是非负整数")
    if bad_cases > cases:
        raise CaseDataError(
            f"bad_case_rate：坏例条数 {bad_cases} 大于用例总数 {cases}——"
            "这个比例会大于 1，而报告里的'坏例占比'从此不可读。"
        )
    return 0.0 if cases == 0 else round(bad_cases / cases, 4)


def mean_of(values: list[float]) -> float:
    """一组数的均值（空列表返回 0.0，而不是崩溃）."""
    if not values:
        return 0.0
    return round(math.fsum(values) / len(values), 4)


__all__ = [
    "BAD_CASE_ACTIONS",
    "BAD_CASE_DESCRIPTIONS",
    "BAD_CASE_PRIORITY",
    "BAD_CASE_TAGS",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_MIN_FAITHFULNESS",
    "DEFAULT_MIN_RECIPROCAL_RANK",
    "FALLBACK_ACTIONS",
    "RETRIEVAL_METRICS",
    "STAGES",
    "STAGE_PACKED",
    "STAGE_RETRIEVED",
    "TAG_GENERATION_FALLBACK",
    "TAG_HALLUCINATED_CITATION",
    "TAG_LOW_COVERAGE",
    "TAG_NO_DATA",
    "TAG_PACKED_AWAY",
    "TAG_RANK_BAD",
    "TAG_RERANK_CUT",
    "TAG_RETRIEVAL_EMPTY",
    "TAG_RETRIEVAL_MISS",
    "TAG_TRUNCATED_CHUNK",
    "TAG_UNGROUNDED",
    "TAG_WEAK_FAITHFULNESS",
    "BadCase",
    "CaseOutcome",
    "RagEvalCase",
    "StageMetrics",
    "bad_case_rate",
    "described_tags",
    "mean_of",
    "measure_ids",
]
