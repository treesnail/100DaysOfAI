"""检索包的四副形状：查询、命中、索引状态、结果（M6-D5）.

day065 交出来的是一本**账**：清单、版本号、备份、漂移判定。
账回答的是"上一版长什么样"，而今天这一层要回答另一类问题：

```text
给一句话，取回哪几条？        → RetrievalQuery → RetrievalHit
为什么是这几条？              → RetrievalHit.score / rank（统一口径：越大越近）
为什么**只有**这几条？        → dropped_* 四个数字 + empty_reason（唯一原因）
这一版索引还作数吗？          → IndexState.version_id / drift
```

day068 给这套形状补了两处**只增不改**的字段（重排的账）：
``RetrievalHit.rerank_score`` / ``stage1_rank`` 回答"重排把它从第几名挪到了第几名"，
``RetrievalResult.rerank`` / ``dropped_by_rerank`` 回答"这次重排做了什么、
被重排阈值切掉几条"。空字典 / ``None`` 的含义都是**"这次没开重排"**
（与 ``channels=()`` 的"未记录"是同一套写法）。

| 形状 | 对应的问题 | 缺了它，会怎样 |
|------|-----------|---------------|
| ``RetrievalQuery`` | 这次问的是什么 | 默认值四散在调用点，"同一次查询"无法被复述 |
| ``RetrievalHit`` | 命中了哪一条 | 只有 id，展示/引用/分组都要回库再查一次 |
| ``IndexState`` | 这次的答案基于哪一版索引 | "检索变差了"无法区分是数据变了还是用了旧清单 |
| ``RetrievalResult`` | 这一次检索的完整交代 | 条数少于 top_k 时，报告里只有"少了"这个事实 |

## 三个诊断数字，为什么必须是三个而不是一个

"这次只返回了 2 条（期望 5 条）"在过去只能靠猜。检索器里有三道独立的减法，
它们各自会吃掉命中数：

```text
dropped_below_threshold  阈值落刀切掉的       → 阈值定得太高
dropped_by_diversity     同文档挤掉的         → max_per_doc 定得太低
dropped_by_top_k         截断到 top_k 丢掉的  → 这只是"取够了"，不是问题
```

**把它们合成一个 ``dropped`` 是一个看起来无害、实际上致命的简化**：
"阈值切掉 3 条"要去调 ``retrieval_min_score``，"多样性挤掉 3 条"要去调
``retrieval_max_per_doc``，而合成之后这两个动作的判据就没了。
更糟的是 day067 的融合需要用这个数字判断"这一路是不是一条都没活下来"。

day068 加进来**第四个数**（``dropped_by_rerank``），它同样**不许**并进
``dropped_by_top_k``：重排阈值是第四个旋钮（``retrieval_rerank_min_score``），
而它切的条数与"top_k 截断"是两件完全不同的事——前者说明阈值定得太高，
后者只是"取够了"。四道减法各自的数字一起读，才回答得了"为什么只有这几条"。

## ``empty_reason`` 为什么只允许一个值

空结果有四种成因（库空 / 被过滤筛没 / 被阈值切完 / 被多样性挤完），
它们对应的动作完全不同。**报告里写四条可能的猜测等于什么都没说**，
所以这里按 ``EMPTY_REASONS`` 的固定优先级只给出**一个**结论——
优先级顺序把"更靠上游的成因"排在前面（库空 > 过滤 > 阈值 > 多样性），
因为上游成因会掩盖下游成因：库是空的，就谈不上"被阈值切掉"。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from smart_research_agent.retrieval import filters as filter_module
from smart_research_agent.retrieval.errors import QueryError
from smart_research_agent.vectorstore.base import MAX_TOP_K

#: 过取倍率：去重/阈值/多样性都要吃名额，因此**从库里取的条数必须多于要用的条数**。
#: 取 3 的理由是三道减法里最狠的一道（``max_per_doc=1`` 且同一文档占满 top-k）
#: 需要约 3 倍深度才能凑够结果；再大就只是白扫（见 ``Retriever._fetch_depth``）。
DEFAULT_FETCH_MULTIPLIER = 3

#: 单次检索允许的召回深度上限。**直接复用 ``vectorstore.MAX_TOP_K``**：
#: 两个数字一旦分开，就会出现"检索深度 2000、库侧上限 1000"这种自相矛盾的配置，
#: 而它的表现是"设置里写 2000、实际只取了 1000"——一个必须靠读代码才能发现的差值。
MAX_FETCH_K = MAX_TOP_K

#: 单条命中至少留这么多字符。低于它的片段进上下文只是噪声
#: （"阈值设为 0.85"这种行离开它上面的标题就毫无意义），
#: 而占用的却是与长片段同样的提示词预算。
DEFAULT_MIN_HIT_CHARS = 32

#: 取"文档身份"的元数据字段名（day062 的 ``knowledge_records()`` 放的就是它）。
#: 它是 ``Retriever(doc_id_field=...)`` 的默认值，也是 ``RetrievalHit.doc_id``
#: 的取值来源——两处默认值必须是同一个常量，否则"分组用的键"与"展示用的键"
#: 会悄悄分家。
#:
#: 两处的**职责**仍然不同，必须说清：``RetrievalHit.doc_id`` 固定读这个字段
#: （它回答"这一块属于哪篇文档"，是**展示与引用口径**）；而
#: ``Retriever(doc_id_field=...)`` 改的是**多样性分组键**（它回答"哪几条算来自
#: 同一篇"，是**检索策略**）。改了分组键时 ``hit.doc_id`` 不跟着变——
#: 展示口径要保持稳定，否则同一份结果在不同检索器下会打印出不同的文档名。
#: 想确认这次的分组键是什么，看 ``Retriever.describe()["doc_id_field"]``
#: 与 ``Retriever.explain()`` 的"分组口径"那一行。
DEFAULT_DOC_ID_FIELD = "parent_doc_id"

#: 本层的默认召回通道名。day067 的 BM25 会用 ``"bm25"`` 之类的另一个值，
#: 而 ``RetrievalHit.channel`` / ``RetrievalResult.channels`` 的形状现在就要留出来。
CHANNEL_VECTOR = "vector"

#: 关键词通道的名字（day067 的 BM25 一路）。
#:
#: 取 ``"bm25"`` 而不是 ``"lexical"`` / ``"keyword"``：这个字符串会出现在
#: **报告、端点响应与融合参数里**（``weights={"vector": 0.5, "bm25": 0.5}``），
#: 而 ``bm25`` 是这一路**唯一的算法身份**——"lexical" 只说了"这是词面的"，
#: 没说"用哪个打分函数"，而换打分函数会让同一批数据的排序整体改变
#: （与 ``vectorstore`` 里"度量必须跟着结果走"是同一条理由）。
CHANNEL_BM25 = "bm25"

#: 本层认识的全部通道名。**它是封闭清单**：融合时出现清单外的通道名，
#: 要么是调用方拼错了，要么是给某一路的权重漏了——两种都必须当场报错，
#: 而不是"没见过就当权重 0"（那会让一路的证据静默消失）。
CHANNELS: tuple[str, ...] = (CHANNEL_VECTOR, CHANNEL_BM25)

#: 有命中。它不是"失败原因"，而是"没有失败"的哨兵值——
#: 让 ``empty_reason`` 永远有值可比，调用方不需要先判空再读它。
EMPTY_REASON_NONE = "hits"

#: 库是空的。**这是合法状态，不是错误**：刚建好的库、刚清空重建的库都是它，
#: 因此这条路返回空结果而不是抛异常（见 ``Retriever`` 第 2 步）。
EMPTY_REASON_NO_DATA = "no_data"

#: 过滤器把候选筛没了：``filter_applied=True`` 且 ``candidates == 0``。
EMPTY_REASON_FILTERED_OUT = "filtered_out"

#: 阈值把命中全切了：``min_score`` 落刀后一条不剩。
EMPTY_REASON_BELOW_THRESHOLD = "below_threshold"

#: ``max_per_doc`` 把命中全挤掉了.
#: **注意**：按 ``Retriever`` 第 7 步的规则（每组至少留一条），端到端路径上
#: 走不到这一支——它是一条**防御性**分支，靠直接构造诊断输入才能触发
#: （理由写在 ``retriever._diagnose`` 的 docstring 里）。
EMPTY_REASON_DIVERSITY = "diversity_trimmed"

#: 全部取值，**顺序 = 诊断优先级顺序**（从最上游的成因到最下游的）。
#: 判定必须按这个顺序做：库是空的时候，"被阈值切掉"这句话没有意义。
EMPTY_REASONS: tuple[str, ...] = (
    EMPTY_REASON_NONE,
    EMPTY_REASON_NO_DATA,
    EMPTY_REASON_FILTERED_OUT,
    EMPTY_REASON_BELOW_THRESHOLD,
    EMPTY_REASON_DIVERSITY,
)

#: 每个取值的人话解释（``explain()`` 与端点直接引用它，避免同一句话写两遍）。
EMPTY_REASON_DESCRIPTIONS: dict[str, str] = {
    EMPTY_REASON_NONE: "有命中，无需诊断",
    EMPTY_REASON_NO_DATA: "库是空的（合法状态）：请先建索引",
    EMPTY_REASON_FILTERED_OUT: "过滤条件把候选筛成了 0 条：请放宽 where 或时间范围",
    EMPTY_REASON_BELOW_THRESHOLD: "min_score 把命中的全切了：请降低阈值或不设阈值",
    EMPTY_REASON_DIVERSITY: "max_per_doc 把命中的全挤掉了：请放宽每文档条数",
}

#: 本层的边界（与 ``vectorstore`` 各留一份，风格一致）。
#:
#: day067 改写过这一份：BM25 与融合**已经实现**（``lexical.py`` / ``fusion.py`` /
#: ``hybrid.py``），因此它们不再出现在这里；day068 又改写了一次——"不带重排序"
#: 这条边界已经**兑现**（``rerank.py``），于是它换成了**兑现之后仍然存在**的边界：
#: 重排是有窗口的（只对前 N 条打分），而教学级的交叉编码器没有真实语义。
#: 剩下的每一条都指着一个别的模块或另一天。
RETRIEVAL_LIMITATIONS: tuple[str, ...] = (
    "重排只作用于前 N 条窗口：窗口之外的命中一条都不打分，"
    "因此它的名次只由第一阶段的分数决定（day068 的窗口纪律，N = retrieval_rerank_top_n）",
    "重排用的是教学级交叉编码器替身（cross-encoder-teaching-v1）：它没有真实语义，"
    "分数只能用于**相对排序**，不能当成'相关性'读（day068；生产环境请替换重排模型）",
    "阈值只对向量通道有效：BM25 的分数没有绝对标度，给它一个数字是假的安全感"
    "（半监督地标定一个跨语料的 BM25 阈值要先有标注数据，见 hybrid）",
    "关键词一路是零依赖的 BM25：CJK 用相邻 2-gram 代替分词器，"
    "因此不做词形还原、不做同义词扩展、不认识未登录词（见 lexical）",
    "不做权限过滤：where 是元数据筛选，不是访问控制（那属于 day015 的工具权限层）",
)

#: 明确排除在范围外的能力（写下来，避免"这不算 bug"的争论）。
RETRIEVAL_OUT_OF_SCOPE: tuple[str, ...] = (
    "查询改写与扩展（同义词/子问题）：属于查询理解，不在检索器里",
    "学习型排序与学习型融合权重（LTR / 用点击日志调 alpha）：本课的权重是显式参数；"
    "重排模型的**训练与微调**同样不在本课内（day068 只给推理侧接口 BaseReranker）",
    "跨索引的分布式检索：本层的路由是**业务选路**，不是分片聚合",
    "在线索引更新：检索是读路径，写入仍由 day065 的 indexing 包负责",
)


def _parse_iso(value: str, *, label: str) -> datetime:
    """把 ISO-8601 文本解析成 ``datetime``（失败时给出可照做的 ``QueryError``）.

    三件事按顺序做，每一步都对应一种真实写法：

    ```text
    "Z" 结尾        → 换成 "+00:00"（Python 3.10 的 fromisoformat 不认 Z）
    只有日期        → datetime.fromisoformat 按当日 00:00:00 解释
    其余            → 交给标准库；失败就是调用方写错了
    ```

    时区混用（``2026-09-01`` 与 ``2026-09-02T00:00:00+08:00``）的报错**不在本函数**：
    那要等两个端点都解析出来才能比较（见 ``TimeRange.__post_init__``）。
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise QueryError(
            f"{label} 不是合法的 ISO-8601 时间：{value!r}（{exc}）。"
            "本包按 ISO 文本比较时间，因此只接受 '2026-09-20'、"
            "'2026-09-20T08:30:00'、'2026-09-20T08:30:00+08:00' 这类写法。"
        ) from exc


@dataclass(frozen=True)
class TimeRange:
    """时间范围：一个字段 + 一个**闭区间**（M6-D5）.

    三条约定写在这里，因为它们是这一层最容易被误解的地方：

    ```text
    1) 闭区间 [start, end]       两端可单独省略；两端都省略 = "没有时间条件"
    2) 按 ISO 文本比较           库里存什么就比什么（要求入库时统一成同一种 ISO 文本）
    3) 字段缺失的记录被排除      $gte 对缺失字段返回 False —— 这不是 bug
    ```

    **第 2 条是显式约定，不是实现细节。** 本包不做"把时间解析成 datetime 再比"，
    而是把 ISO 字符串直接交给 ``vectorstore`` 的 ``$gte`` / ``$lte``：
    对同一种 ISO 文本（``YYYY-MM-DD`` 或 ``YYYY-MM-DDTHH:MM:SS``），
    **字典序与时间序一致**，于是比较是精确的；而一旦库里混进
    ``"2026/09/20"`` 这种别的写法，字典序就不再等于时间序——
    那时更该修的是入库口径，而不是让检索器去猜每一条的格式。
    只看日期（``"2026-09-20"``）时按当日 00:00:00 解释，闭区间因此包含当天全天
    的起点；要包含整天请把上界写成下一天的 ``"2026-09-21"``（半开写法）。

    **第 3 条的理由**：``$gte`` 对"字段不存在"返回 False，于是没有
    ``created_at`` 的记录既不在区间内、也不在区间外——它被**排除**。
    替代方案（把缺失当 0 或当"现在"）会让"这条什么时候建的"这个问题
    得到一个编出来的答案，而错误的时间过滤**不报错**，
    只表现为"过滤之后少了几条"。因此宁可"少召回"，也不"猜一个时间"。
    """

    field: str = "created_at"
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise QueryError(
                f"TimeRange.field 必须是非空字符串，收到 {self.field!r}。"
                "时间范围要有字段名才有意义——空字段名会让 $gte/$lte 落到一个"
                "永远不存在的键上，于是**每一次时间过滤都返回空**。"
            )
        object.__setattr__(self, "field", self.field.strip())

        parsed: dict[str, datetime] = {}
        for label, value in (("start", self.start), ("end", self.end)):
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise QueryError(
                    f"TimeRange.{label} 必须是非空字符串或 None，收到 {value!r}。"
                    "空串不是'没有边界'——没有边界请写 None。"
                )
            object.__setattr__(self, label, value.strip())
            parsed[label] = _parse_iso(value, label=f"TimeRange.{label}")

        if "start" in parsed and "end" in parsed:
            try:
                inverted = parsed["start"] > parsed["end"]
            except TypeError as exc:
                raise QueryError(
                    f"TimeRange 的两个端点无法比较：start={self.start!r}、end={self.end!r}。"
                    "多半是一个带时区、一个不带（例如 '2026-09-01' 与 "
                    "'2026-09-02T00:00:00+08:00'）——请把两端写成同一种形式。"
                ) from exc
            if inverted:
                raise QueryError(
                    f"时间范围倒置：start={self.start!r} 晚于 end={self.end!r}。"
                    "闭区间要求 start <= end；倒置的区间**必然返回空结果**，"
                    "因此在这里当场拒掉，而不是让它去库里筛出一个空集。"
                )

    @property
    def is_empty(self) -> bool:
        """两端都没给 = 这次查询不涉及时间条件（与"区间宽度为 0"是两件事）."""
        return self.start is None and self.end is None

    def bounds(self) -> tuple[datetime | None, datetime | None]:
        """解析后的两端（``None`` 表示该端省略；解析失败已在构造期报过）."""
        low = None if self.start is None else _parse_iso(self.start, label="TimeRange.start")
        high = None if self.end is None else _parse_iso(self.end, label="TimeRange.end")
        return low, high

    def clause(self) -> dict[str, Any]:
        """翻译成 ``where`` 子句（闭区间 → ``$gte`` + ``$lte``，空区间 → ``{}``）."""
        return filter_module.time_range_clause(self)

    def describe(self) -> str:
        """一行人类可读描述（``"created_at ∈ [2026-09-01, 2026-09-30]"``）."""
        if self.is_empty:
            return f"{self.field}（未设区间）"
        low = self.start if self.start is not None else "-∞"
        high = self.end if self.end is not None else "+∞"
        return f"{self.field} ∈ [{low}, {high}]"

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "field": self.field,
            "start": self.start,
            "end": self.end,
            "is_empty": self.is_empty,
            "describe": self.describe(),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（与 ``describe`` 同义，供报告逐行打印）."""
        return self.describe()


@dataclass(frozen=True)
class RetrievalQuery:
    """一次检索请求：**一句话 + 一组可复述的条件**（M6-D5）.

    ``None`` 与 ``0`` 在这里的含义必须分清，否则"没提"与"提了不限"会撞在一起：

    ```text
    None     没有提这个参数 → 取 Retriever 的默认值（构造参数 → settings）
    >= 1     显式给定
    0        只对 Retriever 的 max_per_doc 有意义，意思是"不限"；
             查询里**不允许**写 0（"最多 0 条"不是一个请求，是一个空结果）
    ```

    ``extra`` 是**覆盖参数的槽**：day067 给它定下第一批消费者（融合层：
    ``strategy`` / ``alpha`` / ``k_rrf`` / ``weights``），day068 又加了第二批
    （重排层：``enabled`` / ``mode`` / ``model`` / ``top_n`` / ``weight``）。
    两份键名都是**封闭清单**，清单外的键一律报错（见 ``hybrid._resolve_fusion`` /
    ``hybrid._resolve_rerank``）。

    **只有混合检索器读它**（``HybridRetriever``）：单路 ``Retriever`` 一个键都不读，
    因此同一份带 ``extra`` 的查询交给两个检索器时，单路那边的表现与不带 ``extra``
    完全一样。这不是遗漏而是刻意的分工——单路那条流水线的口径全部来自构造参数与
    ``RetrievalQuery`` 自己的字段，多一个隐式覆盖层会让"这次用的是哪组参数"
    多出一条来源。

    ``where`` 的**语法**在构造期就用 ``vectorstore.compile_filter`` 校验
    （非法条件当场报 ``QueryError``）；而 ``where`` 与 ``time_range`` 的
    **字段冲突**要等到 ``filters.combine_where`` 才判——那里同时握着两者，
    规则只写一份（见该函数的 docstring）。
    """

    text: str
    top_k: int | None = None
    fetch_k: int | None = None
    where: dict[str, Any] | None = None
    time_range: TimeRange | None = None
    min_score: float | None = None
    max_per_doc: int | None = None
    route: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise QueryError(
                f"query.text 必须是字符串，收到 {type(self.text).__name__}。"
                "检索的入口是一段文本——其它形状（向量、id 列表）请直接用 "
                "backend.query() / get_many()。"
            )
        stripped = self.text.strip()
        if not stripped:
            raise QueryError(
                "空查询在向量检索里没有定义：请给一段非空文本。"
                "空白串会编码成一个**固定**的向量，于是每次查询都返回"
                "同一批'最像空白的记录'——一个看起来正常、实际上毫无意义的排序。"
            )
        # 收敛成 strip 之后的形式：'  语义缓存  ' 与 '语义缓存' 必须是同一次查询，
        # 否则同一句话会因为首尾空格而被当成两次不同的检索（缓存与评估都会算错）。
        object.__setattr__(self, "text", stripped)

        if self.top_k is not None:
            self._check_positive_int("top_k", self.top_k, maximum=MAX_FETCH_K)
        if self.fetch_k is not None:
            self._check_positive_int("fetch_k", self.fetch_k)
            if self.top_k is not None and self.fetch_k < self.top_k:
                raise QueryError(
                    f"fetch_k={self.fetch_k} 小于 top_k={self.top_k}。"
                    "召回深度必须 >= 要返回的条数：深度更小时，那些'本来能进 top_k'"
                    "的候选根本不会被取回来（少了多少条不会报错，只会体现为结果变少）。"
                )
        if self.min_score is not None:
            if not isinstance(self.min_score, (int, float)) or isinstance(self.min_score, bool):
                raise QueryError(
                    f"min_score 必须是数字或 None，收到 {type(self.min_score).__name__}"
                )
            if not math.isfinite(float(self.min_score)):
                raise QueryError(
                    f"min_score 必须是有限数，收到 {self.min_score!r}。"
                    "nan 与所有分数比较都返回 False（于是**一条都不会留下**），"
                    "而它在报告里看起来只是一个阈值。"
                )
        if self.max_per_doc is not None:
            self._check_positive_int("max_per_doc", self.max_per_doc)
        if self.where is not None and not isinstance(self.where, dict):
            raise QueryError(
                f"where 必须是字典或 None，收到 {type(self.where).__name__}。"
                "写法：{'strategy': 'structural'} 或 {'token_count': {'$gt': 100}}"
            )
        filter_module.validate_where(self.where)
        if not isinstance(self.extra, dict):
            raise QueryError(f"extra 必须是字典，收到 {type(self.extra).__name__}")

    def _check_positive_int(self, name: str, value: int, *, maximum: int | None = None) -> None:
        """三个整数参数共用一套校验（范围 + 上界），错误消息按名字定制出路."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise QueryError(f"{name} 必须是整数，收到 {type(value).__name__}")
        if value < 1:
            raise QueryError(
                f"{name} 必须 >= 1，收到 {value}。"
                "'要 0 条'不是一个有意义的检索请求；要探测'库里有没有数据'"
                "请用 backend.count()，要看空结果请读 RetrievalResult.empty_reason。"
            )
        if maximum is not None and value > maximum:
            raise QueryError(
                f"{name}={value} 超过上限 {maximum}（与 vectorstore.MAX_TOP_K 同一个上限）。"
                "一次把整库搬进响应体没有意义——要遍历请用 backend.ids() + get_many()。"
            )

    @property
    def has_filter(self) -> bool:
        """是否带了任何过滤条件（元数据或时间范围）."""
        return bool(self.where) or (self.time_range is not None and not self.time_range.is_empty)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload: dict[str, Any] = {
            "text": self.text,
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "where": dict(self.where) if self.where else None,
            "time_range": self.time_range.to_dict() if self.time_range is not None else None,
            "min_score": self.min_score,
            "max_per_doc": self.max_per_doc,
            "route": self.route,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        parts = [f"查询 {self.text!r}"]
        if self.top_k is not None:
            parts.append(f"top_k={self.top_k}")
        if self.fetch_k is not None:
            parts.append(f"fetch_k={self.fetch_k}")
        if self.has_filter:
            parts.append(filter_module.describe_conditions(self.where, self.time_range))
        return " | ".join(parts)


@dataclass(frozen=True)
class RetrievalHit:
    """一条命中：**检索器视角**的一条结果（M6-D5）.

    与 ``vectorstore.SearchHit`` 是两个东西，别混：

    ```text
    SearchHit       record（含向量）+ score + distance + rank   → 库的视角
    RetrievalHit    id + text + metadata + score + rank + channel → 检索器的视角
    ```

    差别不在"少了一个 distance"，而在**它不再携带向量**：
    检索器的下游是打包与生成（day066 的 context.py、day071 的 RAG 评估），
    它们一个字节的向量都不需要，而 384 维浮点数组进 ``to_dict()``
    会让一次"看一眼检索结果"变成 3 KB × N 的响应。

    ``text`` 与 ``metadata`` 冗余存一份（库里也有一份）是刻意的：
    命中之后要还给用户看的就是这两样，**再回库取一次**会让"检索完成"
    与"能展示"之间多出一个可能失败、可能被删除改变的窗口。

    ``channel`` 是本课为 day067 留的形状：混合检索会把 BM25 与向量的结果
    合成一个列表，那时**这一条是哪一路召回来的**必须跟着它走
    （融合要按通道调权重，去重要按通道留证据）。本课所有命中的它都是
    ``CHANNEL_VECTOR``。

    ``channels`` 是 day067 加上去的第二个通道字段，两者**不是重复**：

    ```text
    channel    哪一路把它顶上来的（贡献最大的那一路；并列时向量优先）
    channels   这一条被哪些路同时召回（按名次好坏排序，可能有多条）
    ```

    差别在"被两路同时命中"这种情形上：``channel`` 只能说出一个名字，
    于是"关键词路也召回了它"这个**最强证据**会消失——而它恰恰是融合
    最该展示的东西（同一条记录被两路独立证实）。
    默认值 ``()`` 的含义是"未记录多路证据"，也就是 day066 的单路语义：
    它不是"没有任何通道"，而是"这个形状不知道通道这件事"。
    单路向量检索**刻意不填**它（填 ``(CHANNEL_VECTOR,)`` 会把
    "单路"与"混合但只被一路召回"混成同一种形状，而这两件事在诊断里
    的结论不同）。

    day068 又加了两个字段，它们是**重排的账**（默认 ``None`` = 这次没开重排，
    与 ``channels=()`` 是同一套"未记录"写法）：

    ```text
    rerank_score   重排分（越大越相关；教学级替身落在 [0, 1]）
    stage1_rank    重排**前**它在输入序列里的位置（0 起）
    ```

    **``score`` 的口径没有变**：它仍然是第一阶段（向量 / 融合）的分数。
    这一点必须说清楚，因为重排之后名单的次序由 ``rerank_score`` 决定，
    于是"``score`` 越大越靠前"这条不变量在**重排过的名单里不再成立**——
    把 ``score`` 覆盖成重排分会更"好看"，但代价是丢掉"它原来得了多少分"，
    而"重排把它从第 4 名提到第 1 名，原来那两条的分数差多少"正是评估要看的东西。
    因此两个分数各占一个字段：**排序依据看 ``rerank_score``，跨版本对比看 ``score``**。

    ``stage1_rank`` 记的是**输入序列的位置**（不是上游 ``rank``）：阈值落刀之后
    ``rank`` 会带空洞（0、1、3、4），而"它原来在第几"这个问题要的是一份稠密的次序。
    """

    record_id: str
    score: float
    rank: int
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    channel: str = CHANNEL_VECTOR
    channels: tuple[str, ...] = ()
    rerank_score: float | None = None
    stage1_rank: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise QueryError(
                f"RetrievalHit.record_id 必须是非空字符串，收到 {self.record_id!r}："
                "命中没有 id 就无法被引用、无法被去重、无法回库取原文。"
            )
        if not math.isfinite(float(self.score)):
            raise QueryError(
                f"RetrievalHit.score 必须是有限数，收到 {self.score!r}："
                "排序依赖它，nan 会让这条命中落到任意位置。"
            )
        if not isinstance(self.rank, int) or self.rank < 0:
            raise QueryError(
                f"RetrievalHit.rank 必须是非负整数，收到 {self.rank!r}（从 0 起）"
            )
        if not isinstance(self.metadata, dict):
            raise QueryError(
                f"RetrievalHit.metadata 必须是字典，收到 {type(self.metadata).__name__}"
            )
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise QueryError(
                f"RetrievalHit.channel 必须是非空字符串，收到 {self.channel!r}："
                "day067 的融合要靠它区分'这一条是哪一路召回'。"
            )
        if not isinstance(self.channels, tuple):
            # 收敛成 tuple 而不是就地接受 list：frozen dataclass 里塞一个可变对象，
            # "这份结果不曾被改过"就不再成立（与 memory 包的 frozen 约定同源）。
            raise QueryError(
                f"RetrievalHit.channels 必须是 tuple，收到 {type(self.channels).__name__}。"
                "写法：channels=('vector', 'bm25')；空元组 () 表示'未记录多路证据'。"
            )
        for name in self.channels:
            if not isinstance(name, str) or not name.strip():
                raise QueryError(
                    f"RetrievalHit.channels 里出现了非字符串或空串：{name!r}。"
                    "通道名必须是可读的标识符，否则报告里的证据无法核对。"
                )
        if self.rerank_score is not None:
            if isinstance(self.rerank_score, bool) or not isinstance(
                self.rerank_score, (int, float)
            ):
                raise QueryError(
                    f"RetrievalHit.rerank_score 必须是数字或 None，"
                    f"收到 {type(self.rerank_score).__name__}。"
                    "None 的含义是'这次没开重排'（不是 0 分）——要表达'重排给了 0 分'"
                    "请写 0.0，两者在报告里是两件不同的事。"
                )
            if not math.isfinite(float(self.rerank_score)):
                raise QueryError(
                    f"RetrievalHit.rerank_score 必须是有限数，收到 {self.rerank_score!r}："
                    "重排后的次序由它给出，nan 会让这条落到任意位置。"
                )
            object.__setattr__(self, "rerank_score", float(self.rerank_score))
        if self.stage1_rank is not None:
            if (
                not isinstance(self.stage1_rank, int)
                or isinstance(self.stage1_rank, bool)
                or self.stage1_rank < 0
            ):
                raise QueryError(
                    f"RetrievalHit.stage1_rank 必须是非负整数或 None，"
                    f"收到 {self.stage1_rank!r}。"
                    "它是重排**前**的名次（从 0 起）；None 表示这次没开重排——"
                    "写 0 会把'没记录'说成'它原本排第一'。"
                )

    @property
    def doc_id(self) -> str:
        """所属文档 id（``metadata[DEFAULT_DOC_ID_FIELD]``，缺失回落空串）.

        缺失时回落 ``""`` 而不是 ``record_id``：``""`` 是"这条记录没有文档归属"
        这个**事实**，而回落到 ``record_id`` 会把它伪装成"它自成一个文档"——
        多样性裁剪随后就会把"同一批没有 parent 的记录"当成不同文档全留下。
        """
        return str(self.metadata.get(DEFAULT_DOC_ID_FIELD, "") or "")

    @property
    def heading_path(self) -> str:
        """标题路径（day062 的 ``heading_path``，缺失回落空串）.

        这一条终于把 day062 留下的那个元数据用上了：它是**引用与展示**要的那一段
        （"这句话出自《语义缓存》下的《阈值配置》"），也是检索视图
        ``retrieval_text`` 里那段面包屑的来源。
        """
        return str(self.metadata.get("heading_path", "") or "")

    @property
    def char_count(self) -> int:
        """``text`` 的字符数（打包预算按这个口径算，见 ``DEFAULT_MIN_HIT_CHARS``）."""
        return len(self.text)

    def citation(self, index: int) -> str:
        """渲染成一条引用标记：``[n] 来源 › 标题路径``（字段缺失时回落到 record_id）.

        ``index`` 从 1 起（进提示词的就是 ``[1]`` ``[2]``，与 day069 的引用溯源
        是同一个形状）。编号**由调用方给**而不是这里自增：编号是"在这一次
        提示词里的位置"，只有捧着整份上下文的人知道它。
        """
        if not isinstance(index, int) or index < 1:
            raise QueryError(f"引用编号从 1 起，收到 {index!r}")
        source = str(
            self.metadata.get("source") or self.metadata.get("doc_id") or self.record_id
        )
        body = f"{source} › {self.heading_path}" if self.heading_path else source
        return f"[{index}] {body}"

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_text=False`` 用于"只看排序不变量"的场景（例如逐条对比两次运行
        的 top-k 是否一致）——那时文本只会淹没 diff。
        """
        payload: dict[str, Any] = {
            "rank": self.rank,
            "record_id": self.record_id,
            "score": round(self.score, 6),
            "doc_id": self.doc_id,
            "heading_path": self.heading_path,
            "char_count": self.char_count,
            "channel": self.channel,
            "channels": list(self.channels),
            # day068 新增：重排的账（``None`` = 这次没开重排）。
            # 两个键**恒存在**（值可能是 null）：让下游不必先判断"有没有这个键"，
            # 而"有没有重排"用 `is None` 就回答得了（与 channels=[] 同一套写法）。
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "stage1_rank": self.stage1_rank,
            "metadata": dict(self.metadata),
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        preview = self.text.replace("\n", " ")[:36] if self.text else ""
        return (
            f"#{self.rank} {self.record_id} [{self.channel}] score={self.score:+.6f} "
            f"| {self.doc_id or '（无 parent）'} | {preview}"
        )


@dataclass(frozen=True)
class IndexState:
    """这次检索**基于哪一版索引**（M6-D5）.

    四个字段回答四件事，其中 ``drift`` 是本课兑现的第二个伏笔：

    | 字段 | 回答 | 谁在读它 |
    |------|------|---------|
    | ``version_id`` | 这一版是谁 | 报告、端点、出问题时对照清单 |
    | ``count_store`` | 库里现在有多少条 | 运维一眼看规模 |
    | ``count_manifest`` | 清单说应该有多少条 | 与上一列相减就是差集规模 |
    | ``drift`` | 清单与库哪里互相矛盾 | ``Retriever`` 决定要不要告警 / 拒绝服务 |

    **漂移是"清单与库相互矛盾"的统称**，包括四类（措辞与 ``indexing.verify_index``
    的 ``problems`` 对齐）：清单有库里没有、库里有清单没有、维度不符、口径不符
    （度量或后端）。

    **本课对漂移只观测、不阻断**（除非 ``strict_index=True``）。理由是一个取舍：

    ```text
    阻断（拒绝服务）  清单过期 → 整个检索不可用 → 用户什么都拿不到
    观测（起警告）    清单过期 → 检索仍然能跑，只是可能少召回那几条被删掉的记录
    ```

    "检索仍然能跑、只是可能少召回"比"因为一份 JSON 过期而拒绝服务"更可取；
    而**静默降级**是唯一不可接受的选项——所以漂移必须进 ``RetrievalResult``
    与 ``notes``，让"这次的结果可能不完整"成为一个能被读出来的事实。
    """

    version_id: str = ""
    count_store: int = 0
    count_manifest: int | None = None
    dimension: int = 0
    metric: str = ""
    drift: tuple[str, ...] = ()

    @property
    def has_drift(self) -> bool:
        """清单与库是否互相矛盾（``count_manifest`` 为 ``None`` = 没有清单可比）."""
        return bool(self.drift)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "version_id": self.version_id,
            "count_store": self.count_store,
            "count_manifest": self.count_manifest,
            "dimension": self.dimension,
            "metric": self.metric,
            "has_drift": self.has_drift,
            "drift": list(self.drift),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        version = self.version_id or "（未提供清单）"
        expected = self.count_manifest if self.count_manifest is not None else "（无）"
        drift = f"漂移 {len(self.drift)} 项" if self.drift else "无漂移"
        return (
            f"索引版本 {version} | 库 {self.count_store} 条 / 清单 {expected} 条 | "
            f"{self.dimension}d {self.metric or '（未知度量）'} | {drift}"
        )


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的完整交代：命中 + 三道减法的账 + 空结果诊断（M6-D5）.

    ``hits`` 之外的每个字段都在回答"为什么是这个样子"：

    ```text
    fetch_k                 我本来打算取多深（可能被 MAX_FETCH_K 封顶）
    candidates              库侧过滤之后还剩多少条可选（来自 SearchResult.candidates）
    filter_applied          这次到底有没有过滤（{} 与 None 是两件事）
    dropped_below_threshold 阈值切掉几条
    dropped_by_diversity    每文档上限挤掉几条
    dropped_by_top_k        截断到 top_k 丢掉几条
    empty_reason            没有命中时的**唯一**原因
    index_state             这次的答案基于哪一版索引、有没有漂移
    notes                   漂移告警、封顶、编码器身份不符等"必须被看见但不是错误"的事
    ```

    ``query`` 存的是**已解析默认值**的那一份（``Retriever`` 会把 ``None``
    填成构造参数/settings 里的值），因此一份 ``RetrievalResult`` 能独立回答
    "这次用的是哪个 top_k"，不需要回去查当时的构造参数。

    day067 加了两个字段，它们**只在混合检索时非空**，空字典的含义是
    "这次是单路检索，没有多路证据可记"：

    ```text
    channel_candidates  每路各自候选多少（{"vector": 15, "bm25": 3}）——
                        它回答"有一路是不是根本没捞到东西"，而单看并集看不出来
    fusion              融合摘要（策略 / 参数 / 每路召回条数 / 每路贡献数）——
                        "为什么这条排第一"的全部输入都在这里
    ```

    ``candidates`` 在混合模式下的口径也随之确定：它是**两路候选的并集大小**
    （同一条记录被两路召回只算一次）。这个数字本身就是"融合去重生效了没有"
    的直接证据——并集小于两路之和，差值就是被合并掉的重复。

    day068 又加了两个字段，它们是重排的账（空字典 / 0 = **这次没有重排**：
    没开，或者上游一条候选都没有——两种情况都不需要重排任何东西）：

    ```text
    rerank              重排摘要（模型 / 模式 / 参数 / 候选 / 打分 / 窗口 /
                        被重排阈值切掉几条 / 名次动了几条 / 空结果原因）——
                        "这次重排做了什么"的全部输入都在这里。
                        注意里面的 ``candidates`` 是**重排的输入条数**
                        （阈值/融合之后交给它的那批），与上面那个
                        ``RetrievalResult.candidates``（库侧候选）不是同一个口径
    dropped_by_rerank   重排阈值切掉几条（**第四个 dropped_* 数**，见模块 docstring）
    ```
    """

    query: RetrievalQuery
    hits: tuple[RetrievalHit, ...] = ()
    fetch_k: int = 0
    candidates: int = 0
    filter_applied: bool = False
    dropped_below_threshold: int = 0
    dropped_by_diversity: int = 0
    dropped_by_top_k: int = 0
    empty_reason: str = EMPTY_REASON_NONE
    metric: str = ""
    index_state: IndexState = field(default_factory=IndexState)
    latency_ms: float = 0.0
    notes: tuple[str, ...] = ()
    channel_candidates: dict[str, int] = field(default_factory=dict)
    fusion: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    dropped_by_rerank: int = 0

    def __post_init__(self) -> None:
        if self.empty_reason not in EMPTY_REASONS:
            raise QueryError(
                f"empty_reason 只能是 {', '.join(EMPTY_REASONS)} 之一，"
                f"收到 {self.empty_reason!r}：诊断结论必须来自一份**封闭**的清单，"
                "否则'空结果的唯一原因'会慢慢长成一句自由文本。"
            )
        if self.hits and self.empty_reason != EMPTY_REASON_NONE:
            raise QueryError(
                f"有 {len(self.hits)} 条命中却报告 empty_reason={self.empty_reason!r}："
                "两个字段互相矛盾的结果会让诊断失去意义。"
            )
        for name in (
            "fetch_k",
            "candidates",
            "dropped_below_threshold",
            "dropped_by_diversity",
            "dropped_by_top_k",
            # day068 的第四个数：它**不许**并进 dropped_by_top_k（见模块 docstring）。
            "dropped_by_rerank",
        ):
            value = getattr(self, name)
            if value < 0:
                raise QueryError(f"RetrievalResult.{name} 必须非负，收到 {value}")
        for channel, count in self.channel_candidates.items():
            if not isinstance(channel, str) or not channel.strip():
                raise QueryError(
                    f"channel_candidates 的键必须是非空通道名，收到 {channel!r}"
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise QueryError(
                    f"channel_candidates[{channel!r}] 必须是非负整数，收到 {count!r}。"
                    "每路候选数是一个计数，负数说明这一路的账算错了。"
                )

    @property
    def count(self) -> int:
        """命中条数."""
        return len(self.hits)

    @property
    def is_empty(self) -> bool:
        """是否没有命中（``empty_reason`` 一定不是 ``"hits"``）."""
        return not self.hits

    @property
    def top_k(self) -> int | None:
        """这次要返回的条数上限（来自已解析的 query，未给时为 ``None``）."""
        return self.query.top_k

    def ids(self) -> list[str]:
        """命中的记录 id，按名次排列（与 ``SearchResult.ids()`` 同一个用途）."""
        return [hit.record_id for hit in self.hits]

    def top(self) -> RetrievalHit | None:
        """第一名（没有命中时返回 ``None``，而不是抛异常）."""
        return self.hits[0] if self.hits else None

    @property
    def channels(self) -> tuple[str, ...]:
        """这次结果里出现过的召回通道（去重、按首次出现顺序）.

        单路向量检索时它永远是 ``("vector",)``；day067 的混合检索会让它变成
        ``("vector", "bm25")`` 这样的形状——**融合前必须知道"有哪些路"**，
        否则没法给每一路分配权重。
        """
        seen: list[str] = []
        for hit in self.hits:
            if hit.channel not in seen:
                seen.append(hit.channel)
        return tuple(seen)

    @property
    def doc_ids(self) -> list[str]:
        """命中的文档 id（去重、按首次出现顺序）——多样性裁剪效果的直接证据."""
        seen: list[str] = []
        for hit in self.hits:
            if hit.doc_id not in seen:
                seen.append(hit.doc_id)
        return seen

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_text`` 默认 **False**（与 ``VectorRecord.to_dict`` 的取向相反，
        这里刻意反过来）：检索结果最常见的消费者是报告与端点，
        而一段 800 字的原文 × 10 条会把"这次检索好不好"这个问题直接淹掉。
        要看正文请显式 ``include_text=True``。
        """
        return {
            "query": self.query.to_dict(),
            "count": self.count,
            "fetch_k": self.fetch_k,
            "candidates": self.candidates,
            "filter_applied": self.filter_applied,
            "dropped_below_threshold": self.dropped_below_threshold,
            "dropped_by_diversity": self.dropped_by_diversity,
            "dropped_by_top_k": self.dropped_by_top_k,
            # day068 的第四个数与重排摘要（空 = 这次没开重排，与 fusion 同一写法）。
            "dropped_by_rerank": self.dropped_by_rerank,
            "empty_reason": self.empty_reason,
            "metric": self.metric,
            "index": self.index_state.to_dict(),
            "latency_ms": self.latency_ms,
            "channels": list(self.channels),
            "channel_candidates": dict(self.channel_candidates),
            "fusion": dict(self.fusion),
            "rerank": dict(self.rerank),
            "doc_ids": self.doc_ids,
            "notes": list(self.notes),
            "hits": [hit.to_dict(include_text=include_text) for hit in self.hits],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        head = "、".join(
            f"{hit.record_id}:{hit.score:+.4f}" for hit in self.hits[:3]
        )
        return (
            f"{self.metric or '未知度量'} | 深度 {self.fetch_k} | 候选 {self.candidates} | "
            f"命中 {self.count} | {self.empty_reason} | {head or '（无）'}"
        )

    def explain(self) -> list[str]:
        """人类可读诊断，逐行回答四件事（本课的卖点之一）.

        四问与四行的对应关系是固定的，因为"读一份诊断"的人要找的就是这四句：

        ```text
        1) 这次查的是哪一版索引          → 第一行（版本号 + 库/清单条数 + 漂移项数）
        2) 过滤条件是什么                → 第二行（where + 时间范围 + 候选数）
        3) 为什么条数少于 top_k          → 第三行（四个 dropped_* 数字）
        4) 空结果的唯一原因              → 第四行（empty_reason + 人话解释）
        ```

        漂移明细与 ``notes`` 追加在后面：它们不属于四问，但"这次的结果可能不完整"
        这句话必须跟着结果一起被看见（见 ``IndexState`` 的取舍）。
        day067 的融合摘要与 day068 的重排摘要同样追加在后面，且**都是非空才渲染**
        （空字典 = 这次没开那一层）：它们回答的是第五个问题——"这次中间那几步
        各做了什么"，而它不该在没开的时候占一行。
        """
        query = self.query
        state = self.index_state
        expected = state.count_manifest if state.count_manifest is not None else "（无清单）"
        top_k = query.top_k if query.top_k is not None else "（默认）"
        lines = [
            f"索引版本：{state.version_id or '（未提供清单，只报告库的现状）'}"
            f" | 库 {state.count_store} 条 / 清单 {expected} 条 | "
            f"{state.dimension}d {state.metric or '（未知度量）'} | 漂移 {len(state.drift)} 项",
            f"过滤条件：{filter_module.describe_conditions(query.where, query.time_range)}"
            f"（filter_applied={self.filter_applied}，过滤后候选 {self.candidates} 条）",
            f"条数：{self.count} / top_k={top_k}（召回深度 {self.fetch_k}）"
            f" —— 阈值切掉 {self.dropped_below_threshold} 条、"
            f"每文档上限挤掉 {self.dropped_by_diversity} 条、"
            f"top_k 截断丢掉 {self.dropped_by_top_k} 条、"
            f"重排阈值切掉 {self.dropped_by_rerank} 条",
            f"空结果原因：{self.empty_reason}"
            f"（{EMPTY_REASON_DESCRIPTIONS.get(self.empty_reason, '未知原因')}）",
        ]
        if self.channel_candidates:
            recalled = "、".join(
                f"{name} {count} 条" for name, count in sorted(self.channel_candidates.items())
            )
            lines.append(
                f"两路明细：{recalled} —— 并集 {self.candidates} 条"
                f"（同一条被两路召回只算一次，因此并集 <= 各路之和）"
            )
        if self.fusion:
            params = self.fusion.get("params", {})
            rendered = "、".join(f"{key}={value}" for key, value in sorted(params.items()))
            contributions = self.fusion.get("contributions", {})
            contrib = "、".join(
                f"{name} {count} 条" for name, count in sorted(dict(contributions).items())
            )
            lines.append(
                f"融合：策略 {self.fusion.get('strategy', '（未知）')}"
                f"（{rendered or '无参数'}）—— 每路贡献 {contrib or '（无）'}"
            )
        if self.rerank:
            # 与上面那一行同构：**非空才渲染**（空字典 = 这次没开重排，
            # 而"没开重排"不该在诊断里占一行——那会让每一次单路检索多一句废话）。
            params = self.rerank.get("params", {})
            rendered = "、".join(f"{key}={value}" for key, value in sorted(params.items()))
            # 窗口那个数字在 ``params.top_n`` 里（``RerankResult.to_summary`` 的形状），
            # 不在摘要顶层：多一个同义的顶层键就是多一处迟早会与它分家的副本。
            # 回落键取 ``window``——它是**实际生效**的窗口（`top_n` 大于候选数时两者相等，
            # 而被显式改写时以 window 为准）。
            window = params.get("top_n", self.rerank.get("window", 0))
            lines.append(
                f"重排：{self.rerank.get('model', '（未知模型）')}"
                f"（{rendered or '无参数'}）—— 候选 {self.rerank.get('candidates', 0)} 条、"
                f"只对前 {window} 条打分（真正打分 "
                f"{self.rerank.get('scored', 0)} 条）、名次动了 "
                f"{self.rerank.get('moved', 0)} 条、重排阈值切掉 {self.dropped_by_rerank} 条"
            )
        if state.drift:
            lines.append(f"漂移明细：{state.drift[0]}")
        for note in self.notes:
            lines.append(f"注记：{note}")
        if self.hits:
            preview = "、".join(hit.record_id for hit in self.hits[:3])
            lines.append(f"前 {min(3, self.count)} 条：{preview}")
        return lines


def aggregate_results(results: Sequence[RetrievalResult]) -> dict[str, Any]:
    """批量检索的汇总（风格对齐 ``evaluation.metrics.MetricsTracker.report()``）.

    ```text
    count             批次里的检索次数
    hits / avg_hits   总命中数 / 平均每次命中数
    empty / empty_rate 空结果次数 / 占比
    empty_reasons     空结果的原因分布（只列**真的出现过**的原因）
    avg_fetch_k       平均召回深度
    depth_hit_rate    召回深度命中率 = 平均(命中数 / 召回深度)
    avg_latency_ms    平均延迟
    with_drift        结果里带漂移的次数（清单与库对不上的批次有多少次）
    channels          这批结果里出现过的召回通道
    ```

    ``depth_hit_rate`` 的定义值得写清楚，否则它会被误读成"召回率"：
    它是"取回来的 ``fetch_k`` 条里，最终有多少条活到了结果里"。
    **它衡量的是深度定得准不准**——接近 1 说明深度几乎没浪费（过滤/阈值
    很轻），明显小于 1 说明深度被三道减法吃掉大半（该调深度或调阈值）。
    真正的召回率需要标准答案，那是 day071 的 RAG 评估（本模块没有答案可对）。

    空批次返回全 0 的一行，而不是 ``ZeroDivisionError``：与 ``mean()``
    "没有数据应得 0 分而非崩溃"是同一条纪律。
    """
    total = len(results)
    if total == 0:
        return {
            "count": 0,
            "hits": 0,
            "avg_hits": 0.0,
            "empty": 0,
            "empty_rate": 0.0,
            "empty_reasons": {},
            "avg_fetch_k": 0.0,
            "depth_hit_rate": 0.0,
            "avg_latency_ms": 0.0,
            "with_drift": 0,
            "channels": [],
        }

    reason_counts: dict[str, int] = {}
    for result in results:
        if result.is_empty:
            reason_counts[result.empty_reason] = reason_counts.get(result.empty_reason, 0) + 1
    empty = sum(reason_counts.values())

    hits = [result.count for result in results]
    depths = [result.fetch_k for result in results]
    ratios = [
        result.count / result.fetch_k for result in results if result.fetch_k > 0
    ]
    channels: list[str] = []
    for result in results:
        for channel in result.channels:
            if channel not in channels:
                channels.append(channel)

    return {
        "count": total,
        "hits": sum(hits),
        "avg_hits": round(sum(hits) / total, 4),
        "empty": empty,
        "empty_rate": round(empty / total, 4),
        "empty_reasons": {key: reason_counts[key] for key in sorted(reason_counts)},
        "avg_fetch_k": round(sum(depths) / total, 4),
        "depth_hit_rate": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
        "avg_latency_ms": round(
            sum(result.latency_ms for result in results) / total, 4
        ),
        "with_drift": sum(1 for result in results if result.index_state.has_drift),
        "channels": channels,
    }


__all__ = [
    "CHANNELS",
    "CHANNEL_BM25",
    "CHANNEL_VECTOR",
    "DEFAULT_DOC_ID_FIELD",
    "DEFAULT_FETCH_MULTIPLIER",
    "DEFAULT_MIN_HIT_CHARS",
    "EMPTY_REASON_BELOW_THRESHOLD",
    "EMPTY_REASON_DESCRIPTIONS",
    "EMPTY_REASON_DIVERSITY",
    "EMPTY_REASON_FILTERED_OUT",
    "EMPTY_REASON_NONE",
    "EMPTY_REASON_NO_DATA",
    "EMPTY_REASONS",
    "MAX_FETCH_K",
    "RETRIEVAL_LIMITATIONS",
    "RETRIEVAL_OUT_OF_SCOPE",
    "IndexState",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResult",
    "TimeRange",
    "aggregate_results",
]
