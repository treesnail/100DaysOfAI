"""day069 RAG 生成测试用的确定性样本：小库 + 查询 + 打包上下文 + 答案脚本 + 手算表.

单独一个模块而不是把样本复制进三个测试文件，理由与 ``retrieval_samples`` /
``rerank_samples`` 完全相同：**同一份语料要被 core / pipeline / grounding
三个文件同时用到**，复制多份会让"样本里改一个字"变成要改许多处，而漏改的那一处
只表现为"某个测试不再覆盖那条语义"。

## 这批样本刻意埋了五个差异点（每一个都对应一条要钉住的纪律）

```text
1. 编号 [n] 的四种写法同时出现
   "结论 [1] 与 [2]"（正常）/ "[ 1 ]"（带空格）/ "[1]，[1]"（重复）
   / "[0]"（不是编号）/ "[9]"（越界）
   → 解析口径（升序去重、0 不进 cited、越界进 invalid）每一支都有素材

2. 五条片段的编号恰好是 1~5，而答案脚本的编号集合刻意跨过两端
   {3}（只引中间一条）/ {1,2}（引前两条）/ {1,2,3}（引三条）
   / {1,2,3,4,5}（全引）/ {9} 与 {12}（一条都不在提示词里）
   → 覆盖率 = len(valid) / 5 只有 0.0 / 0.2 / 0.4 / 0.6 / 1.0 五个取值

3. 同一条答案里既有有效引用又有幻觉引用（"[1] 与 [9]"）
   → "有引用"与"引用可信"是两件事：valid 非空但 invalid 也非空时
      grounded 必须是 False（覆盖率仍然是 0.2，两个数各说各的）

4. 三个固定拒答句式各有一条脚本，另有一条**反例**
   "这份资料里写明了默认预算 [1]" 含"资料"二字、也含合法引用，
   但它没有命中任何固定句式 → 必须是正常路径（否则误判一次拒答
   等于把一次正常回答丢掉）

5. 空串与纯空白串分开
   → empty_reply 那条路要同时钉住 "answer 就是模型给的那个空串"
     （不是兜底答复），以及"它不渲染核对那一段注记"
```

## 期望值为什么能手算

```text
1) 小库是 ``retrieval_samples`` 那八条（向量分量只有 0 / 0.5 / 0.6 / 0.8 / 1.0），
   查询 "axis" 的分数次序写死为 EXPECTED_AXIS_ORDER：
   c-a-01 / c-b-01 / c-a-02 / c-c-02 / c-c-01 / c-a-03 / c-b-02 / c-c-03
   → 打包前 5 条得到的编号表是 1→c-a-01、2→c-b-01、3→c-a-02、4→c-c-02、5→c-c-01
   → "答案里的 [3] 指谁"是一个**可以直接写在断言里的字符串**
2) 五条片段共 295 字（远小于 BUDGET_CHARS=4000），因此一条都不被截断、
   一条都不被丢掉：citations 的条数与命中条数逐条相等
3) 覆盖率的分母恒为 5（或 8，见 ``FULL_CITATION_IDS``），分子是手数的编号个数
   → 1/5=0.2、2/5=0.4、3/5=0.6、5/5=1.0，全是有限小数
   → 需要验证 round(..., 4) 的样本另写在 ROUNDING_CASES 里（1/3、2/7 之类）
```

全部离线、确定性、零网络、不写盘：库是 ``FlatVectorStore``，编码器是
``TableEmbedding``（按文本查表，向量由人给定），模型是 ``AnswerLLM``。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.retrieval.context import PackedContext, pack_context
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import RetrievalQuery, RetrievalResult
from tests.retrieval_samples import (
    EXPECTED_AXIS_ORDER,
    QUERY_TEXTS,
    TableEmbedding,
    sample_store,
)

# --------------------------------------------------------------------------- #
# 小库、查询与打包上下文
# --------------------------------------------------------------------------- #

#: 本次唯一的查询（它是 ``TableEmbedding`` 查表的键，本身没有语义）.
QUESTION = QUERY_TEXTS["axis"]

#: 打包预算与单条上限：刻意给得很宽，让"进了上下文"与"被检索给了"一一相等
#: （截断与丢尾是 ``context`` 那一层的主题，这里要的是干净的编号表）.
BUDGET_CHARS = 4000
PER_HIT_CHARS = 800

#: 缺省上下文里的片段条数（= 检索器的缺省 top_k，见 ``settings.retrieval_top_k``）.
CITATION_COUNT = 5

#: 编号 → 记录 id 的手算对照表（前 5 条，见模块 docstring 第 1 条）.
#: 它是本模块最常被引用的一行：``MARKER_TO_ID[3] == "c-a-02"``。
CITATION_IDS: tuple[str, ...] = EXPECTED_AXIS_ORDER[:CITATION_COUNT]

#: 全部 8 条（``packed_context(8)`` 用）：覆盖率的另一个分母.
FULL_CITATION_IDS: tuple[str, ...] = EXPECTED_AXIS_ORDER

#: 编号 → 记录 id（从 1 起）。越界编号不在这张表里 —— 那正是"幻觉引用"的定义.
MARKER_TO_ID: dict[int, str] = {
    marker: record_id for marker, record_id in enumerate(CITATION_IDS, start=1)
}

#: 编号 → 记录 id（**全部 8 条**）：``packed_context(6|7|8)`` 用的编号表.
#: 6~8 是 ``EXPECTED_AXIS_ORDER`` 尾部的三条（c-a-03 / c-b-02 / c-c-03）。
ALL_MARKER_TO_ID: dict[int, str] = {
    marker: record_id for marker, record_id in enumerate(FULL_CITATION_IDS, start=1)
}


def question() -> str:
    """本次的问题文本（与 ``RetrievalQuery.text`` 同一份）."""
    return QUESTION


def sample_retriever() -> Retriever:
    """八条样本记录的检索器（缺省 top_k=5，无过滤、无阈值、不重排）."""
    return Retriever(sample_store(), TableEmbedding())


def retrieval(top_k: int | None = None) -> RetrievalResult:
    """跑一次检索（``top_k`` 缺省时用检索器的缺省值 = 5）."""
    query = QUESTION if top_k is None else RetrievalQuery(text=QUESTION, top_k=top_k)
    return sample_retriever().retrieve(query)


@cache
def packed_context(count: int = CITATION_COUNT) -> PackedContext:
    """打包前 ``count`` 条命中（编号 1..count 与 ``CITATION_IDS[:count]`` 一一对应）.

    缓存起来是因为接地那一层要在"上下文条数 × 答案编号集合"的笛卡尔积上跑
    上百组用例，而打包结果是 frozen 的形状、共享是安全的（没人能改它）。
    """
    return pack_context(
        retrieval(count if count != CITATION_COUNT else None).hits,
        max_chars=BUDGET_CHARS,
        per_hit_chars=PER_HIT_CHARS,
    )


@cache
def empty_packed_context() -> PackedContext:
    """**空**打包上下文（``is_empty`` 为真）：生成器第 2 步护栏的输入.

    它不是异常，也不是"检索为空"——它是"这一次没有命中可打包"，
    在生成器门口被 ``PackedContext.is_empty`` 拦住。
    """
    return pack_context([], max_chars=BUDGET_CHARS, per_hit_chars=PER_HIT_CHARS)


def markers_of(context: PackedContext) -> tuple[int, ...]:
    """一份上下文里的编号（升序）—— 它是覆盖率的**分母**."""
    return tuple(citation.marker for citation in context.citations)


# --------------------------------------------------------------------------- #
# 手算常量：三组编号 + 覆盖率（分母都是 5，见模块 docstring 第 2 条）
# --------------------------------------------------------------------------- #

#: 答案里出现过的编号（升序去重）—— 来自答案.
CITED_NONE: tuple[int, ...] = ()
CITED_ONE: tuple[int, ...] = (3,)
CITED_TWO: tuple[int, ...] = (1, 2)
CITED_THREE: tuple[int, ...] = (1, 2, 3)
CITED_ALL_FIVE: tuple[int, ...] = (1, 2, 3, 4, 5)
CITED_HALLUCINATED: tuple[int, ...] = (9,)
CITED_MIXED: tuple[int, ...] = (1, 9)

#: 落在提示词里的编号（升序）—— 覆盖率的**分子**.
VALID_NONE: tuple[int, ...] = ()
VALID_ONE: tuple[int, ...] = (3,)
VALID_TWO: tuple[int, ...] = (1, 2)
VALID_THREE: tuple[int, ...] = (1, 2, 3)
VALID_ALL_FIVE: tuple[int, ...] = (1, 2, 3, 4, 5)

#: 落不到提示词里的编号（升序）—— 幻觉引用.
INVALID_NONE: tuple[int, ...] = ()
INVALID_HALLUCINATED: tuple[int, ...] = (9,)
INVALID_BIG_MARKER: tuple[int, ...] = (12,)

#: 给了它、答案没引用的编号（升序）—— 覆盖率的分母的另一半.
UNUSED_TWO: tuple[int, ...] = (4, 5)
UNUSED_THREE: tuple[int, ...] = (3, 4, 5)
UNUSED_FROM_TWO: tuple[int, ...] = (2, 3, 4, 5)
UNUSED_FROM_ONE: tuple[int, ...] = (1, 2, 4, 5)
UNUSED_ALL_FIVE: tuple[int, ...] = (1, 2, 3, 4, 5)
UNUSED_NONE: tuple[int, ...] = ()

#: 覆盖率 = len(valid) / 5（全是有限小数，可以逐位写下来）.
COVERAGE_ZERO = 0.0
COVERAGE_ONE = 0.2
COVERAGE_TWO = 0.4
COVERAGE_THREE = 0.6
COVERAGE_FULL = 1.0

#: 分母为 2 时的覆盖率（``packed_context(2)`` + 引到 1 条）.
COVERAGE_HALF = 0.5

#: ``(分母, 分子, 期望值)`` 三元的表：专门盯 ``round(..., 4)`` 的进位与截断.
#:
#: ```text
#: 1/3 = 0.33333…       → 0.3333（截断）
#: 2/3 = 0.66666…       → 0.6667（进位：第 5 位是 6）
#: 1/6 = 0.16666…       → 0.1667（进位）
#: 1/7 = 0.142857…      → 0.1429（第 5 位是 5，进位）
#: 2/7 = 0.285714…      → 0.2857（第 5 位是 1，不进位）
#: 3/8 = 0.375          → 0.375（精确值不受 round 影响）
#: 4/5 = 0.8            → 0.8（同上）
#: ```
ROUNDING_CASES: tuple[tuple[int, int, float], ...] = (
    (1, 1, 1.0),
    (2, 1, 0.5),
    (3, 1, 0.3333),
    (3, 2, 0.6667),
    (4, 3, 0.75),
    (5, 4, 0.8),
    (6, 1, 0.1667),
    (7, 1, 0.1429),
    (7, 2, 0.2857),
    (8, 1, 0.125),
    (8, 3, 0.375),
)

# --------------------------------------------------------------------------- #
# 答案脚本
# --------------------------------------------------------------------------- #

#: 正常答案：引到前两条（valid 2 / unused 3 / 覆盖 0.4 / 接地通过）.
ANSWER_TWO = "默认返回前 5 条 [1]，召回深度按三倍过取 [2]。"

#: 只引中间那一条（valid 1 / unused 4 / 覆盖 0.2）—— 覆盖率的"部分引用".
ANSWER_ONE = "召回深度按三倍过取 [3]。"

#: 引满五条（valid 5 / unused 空 / 覆盖 1.0）—— 覆盖率的另一头.
ANSWER_ALL_FIVE = "[1][2][3][4][5] 五条依据都用上了。"

#: **越界引用**：提示词里只有 [1]~[5]，[9] 指不到任何片段.
#: 它不是"多写了一个数字"，而是**指到了一份并不存在的依据**。
ANSWER_OUT_OF_RANGE = "这条结论来自 [9]。"

#: 更大的越界编号（用来钉住"越界编号本身不参与排序，只按升序接在后面"）.
ANSWER_BIG_MARKER = "见 [12]。"

#: 有效引用与幻觉引用同现：[1] 落在提示词里、[9] 不落.
ANSWER_MIXED = "结论 [1] 与 [9]。"

#: 一条引用都没有的答案（它**不是**拒答，只是一次没有依据的总结）.
ANSWER_NO_CITATION = "这个配置项在本次片段里没有出现。"

#: 空串与纯空白串（两条路都要走 empty_reply，但 answer 要被原样交付）.
ANSWER_EMPTY = ""
ANSWER_BLANK = "   \n\t  "

#: ``[ 1 ]`` 带空格：编号语法允许空格（正则里的 ``\s*``）.
ANSWER_SPACED = "结论见 [ 1 ] 与 [2]。"

#: 同一个编号写三遍：升序去重之后只有一条.
ANSWER_DUPLICATED = "先说 [1]，再说 [1]，最后还是 [1]。"

#: 一段里连写三个编号：三个都要被看见.
ANSWER_MULTI = "[1][2][3] 三条都支持这个结论。"

#: ``[0]``：**不是编号**，也不是幻觉引用 —— 它只进报告自己的注记.
ANSWER_ZERO_NOISE = "见 [0] 与 [1]。"

#: 三个固定拒答句式各一条（v2 第 2 条约束要求模型三选一，检测清单见 DECLINE_MARKERS）.
DECLINE_ANSWER_NO_CONTENT = "资料中没有相关内容，片段里没有提到时间范围的默认值。"
DECLINE_ANSWER_NOT_MENTIONED = "资料未提及该配置项，建议补充文档后再问。"
DECLINE_ANSWER_CANNOT = "无法依据资料回答：给出的片段都不涉及这个问题。"

#: 句式 → 答案脚本（期望命中的那一句，就是 ``_declined_marker`` 的返回值）.
DECLINE_ANSWER_BY_MARKER: dict[str, str] = {
    "资料中没有相关内容": DECLINE_ANSWER_NO_CONTENT,
    "资料未提及": DECLINE_ANSWER_NOT_MENTIONED,
    "无法依据资料回答": DECLINE_ANSWER_CANNOT,
}

#: **反例**：含"资料"二字、也含合法引用，但没有命中任何固定句式.
#: 它必须走正常路径 —— 误判一次拒答等于把一次正常回答丢掉。
ANSWER_MENTIONS_SOURCES = "这份资料里写明了默认预算 [1]，可以按它配置。"

#: 其余几条"看起来像拒答、其实不是"的反例（都不含三个句式里的任何一句）.
DECLINE_COUNTEREXAMPLES: tuple[str, ...] = (
    ANSWER_MENTIONS_SOURCES,
    "片段中没有直接答案，但可以按 [1] 的预算配置。",
    "该问题超出本知识库范围 [2]。",
    "无法确定这个数字 [1]。",
)


# --------------------------------------------------------------------------- #
# 手算表：一条答案脚本 → 六个输出值（+ 报告注记条数）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HandGrounding:
    """一条答案脚本在**五条片段**的上下文下的核对结果（全部手算）.

    ``report_notes`` 是 ``GroundingReport.notes`` 的条数：它只由两种情形产生
    （``[0]`` 这类格式噪声、以及"答案里一条 [n] 都没有"），因此是个可数的整数。
    """

    label: str
    answer: str
    cited: tuple[int, ...]
    valid: tuple[int, ...]
    invalid: tuple[int, ...]
    unused: tuple[int, ...]
    coverage: float
    grounded: bool
    report_notes: int = 0


GROUNDING_CASES: tuple[HandGrounding, ...] = (
    HandGrounding(
        label="two",
        answer=ANSWER_TWO,
        cited=CITED_TWO,
        valid=VALID_TWO,
        invalid=INVALID_NONE,
        unused=UNUSED_THREE,
        coverage=COVERAGE_TWO,
        grounded=True,
    ),
    HandGrounding(
        label="one",
        answer=ANSWER_ONE,
        cited=CITED_ONE,
        valid=VALID_ONE,
        invalid=INVALID_NONE,
        unused=UNUSED_FROM_ONE,
        coverage=COVERAGE_ONE,
        grounded=True,
    ),
    HandGrounding(
        label="all_five",
        answer=ANSWER_ALL_FIVE,
        cited=CITED_ALL_FIVE,
        valid=VALID_ALL_FIVE,
        invalid=INVALID_NONE,
        unused=UNUSED_NONE,
        coverage=COVERAGE_FULL,
        grounded=True,
    ),
    HandGrounding(
        label="out_of_range",
        answer=ANSWER_OUT_OF_RANGE,
        cited=CITED_HALLUCINATED,
        valid=VALID_NONE,
        invalid=INVALID_HALLUCINATED,
        unused=UNUSED_ALL_FIVE,
        coverage=COVERAGE_ZERO,
        grounded=False,
    ),
    HandGrounding(
        label="big_marker",
        answer=ANSWER_BIG_MARKER,
        cited=(12,),
        valid=VALID_NONE,
        invalid=INVALID_BIG_MARKER,
        unused=UNUSED_ALL_FIVE,
        coverage=COVERAGE_ZERO,
        grounded=False,
    ),
    HandGrounding(
        label="mixed",
        answer=ANSWER_MIXED,
        cited=CITED_MIXED,
        valid=(1,),
        invalid=INVALID_HALLUCINATED,
        unused=UNUSED_FROM_TWO,
        coverage=COVERAGE_ONE,
        grounded=False,
    ),
    HandGrounding(
        label="no_citation",
        answer=ANSWER_NO_CITATION,
        cited=CITED_NONE,
        valid=VALID_NONE,
        invalid=INVALID_NONE,
        unused=UNUSED_ALL_FIVE,
        coverage=COVERAGE_ZERO,
        grounded=False,
        report_notes=1,
    ),
    HandGrounding(
        label="empty",
        answer=ANSWER_EMPTY,
        cited=CITED_NONE,
        valid=VALID_NONE,
        invalid=INVALID_NONE,
        unused=UNUSED_ALL_FIVE,
        coverage=COVERAGE_ZERO,
        grounded=False,
        report_notes=1,
    ),
    HandGrounding(
        label="blank",
        answer=ANSWER_BLANK,
        cited=CITED_NONE,
        valid=VALID_NONE,
        invalid=INVALID_NONE,
        unused=UNUSED_ALL_FIVE,
        coverage=COVERAGE_ZERO,
        grounded=False,
        report_notes=1,
    ),
    HandGrounding(
        label="spaced",
        answer=ANSWER_SPACED,
        cited=CITED_TWO,
        valid=VALID_TWO,
        invalid=INVALID_NONE,
        unused=UNUSED_THREE,
        coverage=COVERAGE_TWO,
        grounded=True,
    ),
    HandGrounding(
        label="duplicated",
        answer=ANSWER_DUPLICATED,
        cited=(1,),
        valid=(1,),
        invalid=INVALID_NONE,
        unused=UNUSED_FROM_TWO,
        coverage=COVERAGE_ONE,
        grounded=True,
    ),
    HandGrounding(
        label="multi",
        answer=ANSWER_MULTI,
        cited=CITED_THREE,
        valid=VALID_THREE,
        invalid=INVALID_NONE,
        unused=UNUSED_TWO,
        coverage=COVERAGE_THREE,
        grounded=True,
    ),
    HandGrounding(
        label="zero_noise",
        answer=ANSWER_ZERO_NOISE,
        cited=(1,),
        valid=(1,),
        invalid=INVALID_NONE,
        unused=UNUSED_FROM_TWO,
        coverage=COVERAGE_ONE,
        grounded=True,
        report_notes=1,
    ),
    HandGrounding(
        label="declined",
        answer=DECLINE_ANSWER_NOT_MENTIONED,
        cited=CITED_NONE,
        valid=VALID_NONE,
        invalid=INVALID_NONE,
        unused=UNUSED_ALL_FIVE,
        coverage=COVERAGE_ZERO,
        grounded=False,
        report_notes=1,
    ),
)

def hand_checks(
    used: tuple[int, ...] = (1, 2),
    *,
    count: int = CITATION_COUNT,
    extra: tuple[int, ...] = (),
) -> tuple[tuple[int, str | None, bool], ...]:
    """手算的逐编号检查行：**先 1..count，再按升序接上越界编号**.

    每一行是 ``(marker, record_id, used)``：

    ```text
    1..count      record_id 取自 MARKER_TO_ID（编号 → 我们给出去的那条记录）
    越界编号       record_id 必须是 None（它指向一份并不存在的依据）
    used          这个编号在答案里被引到了没有（越界那条恒为 True：
                  它之所以出现在这张表里，正是因为模型写了它）
    ```
    """
    used_set = set(used)
    rows = [
        (marker, MARKER_TO_ID[marker], marker in used_set) for marker in range(1, count + 1)
    ]
    rows.extend((marker, None, True) for marker in sorted(extra))
    return tuple(rows)


#: 引到前两条时的检查行（1、2 是 True，3、4、5 是 False）.
HAND_CHECKS_TWO: tuple[tuple[int, str | None, bool], ...] = hand_checks((1, 2))

#: 五条都被引用时的检查行（全 True）.
HAND_CHECKS_ALL_FIVE: tuple[tuple[int, str | None, bool], ...] = hand_checks((1, 2, 3, 4, 5))

#: ``[1] 与 [9]`` 那一条的手算结果：先 1..5，再接上越界编号 9.
#: 越界那一条的 ``record_id`` 必须是 ``None``。
HAND_CHECKS_MIXED: tuple[tuple[int, str | None, bool], ...] = hand_checks((1,), extra=(9,))

#: 一条都没有被引用时的检查行（``used`` 全 False）.
HAND_CHECKS_NONE_USED: tuple[tuple[int, str | None, bool], ...] = hand_checks(())


# --------------------------------------------------------------------------- #
# 假 LLM：答案由脚本给定（每次调用都记下入参）
# --------------------------------------------------------------------------- #


class AnswerLLM(BaseLLM):
    """按脚本回复的假 LLM：**每次都返回同一段答案**，并把入参逐次记下来.

    与 ``MockLLM`` 的取舍差别只有一处：``MockLLM`` 的 ``responses`` 是**弹出**的
    （脚本用尽之后回落到 ``"mock response"``），而接地那一层要在同一个模型上
    跑上百次"同一个答案"，弹出式脚本会让第 2 次之后的答案悄悄变样——
    那会让期望值与实际值错位，而错位的样子恰好是"某一个用例红了"。
    """

    def __init__(self, answer: str = ANSWER_TWO) -> None:
        self._answer = answer
        self.calls: list[dict[str, Any]] = []

    @property
    def answer(self) -> str:
        """这段脚本要返回的答案."""
        return self._answer

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """记下这一次的入参，再返回脚本答案."""
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self._answer


def prompt_of(llm: AnswerLLM) -> str:
    """取最后一次调用真正发出去的那段提示词."""
    return str(llm.calls[-1]["messages"][0].content)


__all__ = [
    "ALL_MARKER_TO_ID",
    "ANSWER_ALL_FIVE",
    "ANSWER_BIG_MARKER",
    "ANSWER_BLANK",
    "ANSWER_DUPLICATED",
    "ANSWER_EMPTY",
    "ANSWER_MENTIONS_SOURCES",
    "ANSWER_MIXED",
    "ANSWER_MULTI",
    "ANSWER_NO_CITATION",
    "ANSWER_ONE",
    "ANSWER_OUT_OF_RANGE",
    "ANSWER_SPACED",
    "ANSWER_TWO",
    "ANSWER_ZERO_NOISE",
    "BUDGET_CHARS",
    "CITATION_COUNT",
    "CITATION_IDS",
    "CITED_ALL_FIVE",
    "CITED_HALLUCINATED",
    "CITED_MIXED",
    "CITED_NONE",
    "CITED_ONE",
    "CITED_THREE",
    "CITED_TWO",
    "COVERAGE_FULL",
    "COVERAGE_HALF",
    "COVERAGE_ONE",
    "COVERAGE_THREE",
    "COVERAGE_TWO",
    "COVERAGE_ZERO",
    "DECLINE_ANSWER_BY_MARKER",
    "DECLINE_ANSWER_CANNOT",
    "DECLINE_ANSWER_NO_CONTENT",
    "DECLINE_ANSWER_NOT_MENTIONED",
    "DECLINE_COUNTEREXAMPLES",
    "FULL_CITATION_IDS",
    "GROUNDING_CASES",
    "HAND_CHECKS_ALL_FIVE",
    "HAND_CHECKS_MIXED",
    "HAND_CHECKS_NONE_USED",
    "HAND_CHECKS_TWO",
    "INVALID_BIG_MARKER",
    "INVALID_HALLUCINATED",
    "INVALID_NONE",
    "MARKER_TO_ID",
    "PER_HIT_CHARS",
    "QUESTION",
    "ROUNDING_CASES",
    "UNUSED_ALL_FIVE",
    "UNUSED_FROM_ONE",
    "UNUSED_FROM_TWO",
    "UNUSED_NONE",
    "UNUSED_THREE",
    "UNUSED_TWO",
    "VALID_ALL_FIVE",
    "VALID_NONE",
    "VALID_ONE",
    "VALID_THREE",
    "VALID_TWO",
    "AnswerLLM",
    "HandGrounding",
    "empty_packed_context",
    "hand_checks",
    "markers_of",
    "packed_context",
    "prompt_of",
    "question",
    "retrieval",
    "sample_retriever",
]
