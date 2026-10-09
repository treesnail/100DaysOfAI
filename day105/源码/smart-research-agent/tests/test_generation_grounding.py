"""day069 引用核对与接地判定这一层的专项测试（编号层面，不含语义判断）.

这一层的产物是六个输出值（``cited / valid / invalid / unused / coverage / grounded``），
它们要进报告、进 day071 的实验分组，因此必须**另写一份直白实现**交叉核对——

```text
被测实现   _grounding_report(answer, citations)：正则解析 + 集合运算
直白实现   本文件里的 reference_markers / reference_report / reference_checks：
           逐字符扫描 "[" + 空格 + 数字 + 空格 + "]"（不用正则），
           再用最朴素的集合运算算出同样六个值
```

两份实现在 **19 种答案编号集合 × 8 种上下文条数 = 152 组**枚举输入上逐位比对
（照 ``test_rerank_lift.py`` 的"另写一份直白实现交叉核对"手法）。

其余四组：

```text
逐编号检查行   checks 的顺序（先 1..n 再升序接上越界编号）与每行的 record_id
覆盖率        round(..., 4) 的进位与截断；分母不存在时是 0.0 而不是异常
接地定义      valid 非空 且 invalid 为空（真值表 + 逐组合的交叉核对）
三类注记      "幻觉引用点了名" / "一条都没用上的条数" / "覆盖率为 0"
```

**刻意不写"软断言"**：所有断言都指向具体数值、具体次序或具体的注记话术。

全部离线、确定性、零网络：上下文来自 ``generation_samples`` 的打包结果，
模型是 ``AnswerLLM``（同一个答案反复返回，脚本不会用尽）。

关于 19 种"答案编号集合"与 8 种"上下文条数"：``PackedContext`` 的编号是
**1 起连续**的（构造期强制），因此上下文那一侧的自由度只有条数 ``n``；
而答案那一侧的自由度是**任意**编号集合（含 0、重复写、越界编号）。
两边相乘就是这一层全部的形状。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.retrieval.context import Citation
from smart_research_agent.retrieval.generation import (
    FALLBACK_REASON_NO_CONTEXT,
    RAGGenerator,
    _grounding_report,
)
from tests.generation_samples import (
    ALL_MARKER_TO_ID,
    ANSWER_MIXED,
    ANSWER_NO_CITATION,
    ANSWER_OUT_OF_RANGE,
    CITATION_IDS,
    FULL_CITATION_IDS,
    HAND_CHECKS_ALL_FIVE,
    HAND_CHECKS_MIXED,
    HAND_CHECKS_NONE_USED,
    HAND_CHECKS_TWO,
    QUESTION,
    ROUNDING_CASES,
    AnswerLLM,
    empty_packed_context,
    packed_context,
)

#: 枚举的上下文条数（1..8 = 样本库的全部记录）.
CONTEXT_SIZES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#: 枚举的答案编号集合：空集 / 单条 / 一对 / 连号 / 全引 / 越界 / 0 / 重复写 / 越界夹带.
ANSWER_MARKER_SETS: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (2,),
    (3,),
    (5,),
    (1, 2),
    (2, 3),
    (1, 2, 3),
    (1, 2, 3, 4, 5),
    (8,),
    (9,),
    (12,),
    (0,),
    (1, 9),
    (1, 9, 12),
    (0, 1),
    (2, 8),
    (5, 9),
    (1, 1),
)


# --------------------------------------------------------------------------- #
# 直白实现（与 generation 的正则/集合写法无关的第二份算术）
# --------------------------------------------------------------------------- #


def reference_markers(answer: str) -> list[int]:
    """直白版 ``[n]`` 扫描：**不用正则**，逐个字符找 ``[`` + 空格 + 数字 + 空格 + ``]``.

    只把 ASCII 空格当作括号内的空白：枚举出来的答案里没有制表符或全角空格，
    而"``\\s`` 到底算哪些字符"正是被交叉核对的那一处口径——把它写窄一点，
    真要分家时它才会响。
    """
    found: list[int] = []
    index = 0
    while index < len(answer):
        if answer[index] != "[":
            index += 1
            continue
        cursor = index + 1
        while cursor < len(answer) and answer[cursor] == " ":
            cursor += 1
        start = cursor
        while cursor < len(answer) and answer[cursor].isdigit():
            cursor += 1
        if cursor > start:
            tail = cursor
            while tail < len(answer) and answer[tail] == " ":
                tail += 1
            if tail < len(answer) and answer[tail] == "]":
                found.append(int(answer[start:cursor]))
                index = tail + 1
                continue
        index += 1
    return found


def reference_report(
    answer: str, markers: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], float, bool]:
    """直白版核对报告：六个输出值用最朴素的集合运算算出来.

    ```text
    cited    解析出的编号里 >= 1 的那些（升序去重）—— 0 与负数不是编号
    valid    cited ∩ 提示词编号
    invalid  cited - 提示词编号        → 幻觉引用
    unused   提示词编号 - cited        → 覆盖率的另一半
    coverage len(valid) / len(提示词编号)，分母为 0 时是 0.0
    grounded bool(valid) and not invalid
    ```
    """
    known = sorted(set(markers))
    inside = set(known)
    cited = tuple(sorted({marker for marker in reference_markers(answer) if marker >= 1}))
    valid = tuple(marker for marker in cited if marker in inside)
    invalid = tuple(marker for marker in cited if marker not in inside)
    unused = tuple(sorted(inside - set(cited)))
    coverage = round(len(valid) / len(known), 4) if known else 0.0
    grounded = bool(valid) and not invalid
    return cited, valid, invalid, unused, coverage, grounded


def reference_checks(
    answer: str, markers: tuple[int, ...]
) -> tuple[tuple[int, str | None, bool], ...]:
    """直白版逐编号检查行：**先 1..n，再按升序接上越界编号**（越界那条 id 是 None）."""
    cited = {marker for marker in reference_markers(answer) if marker >= 1}
    inside = sorted(set(markers))
    rows: list[tuple[int, str | None, bool]] = [
        (marker, ALL_MARKER_TO_ID[marker], marker in cited) for marker in inside
    ]
    rows.extend((marker, None, True) for marker in sorted(cited - set(inside)))
    return tuple(rows)


def reference_note_count(answer: str, markers: tuple[int, ...]) -> int:
    """直白版注记条数：格式噪声一条 + "一条 [n] 都没写"一条（两者可以同时成立）."""
    parsed = reference_markers(answer)
    cited = [marker for marker in parsed if marker >= 1]
    notes = 1 if any(marker < 1 for marker in parsed) else 0
    if not markers or not cited:
        notes += 1
    return notes


def answer_text(markers: tuple[int, ...]) -> str:
    """把一组编号写成一个确定的答案串（没有编号时是一段没有括号的普通句子）."""
    if not markers:
        return "这一段没有任何引用编号。"
    return "结论 " + " ".join(f"[{marker}]" for marker in markers) + "。"


def marker_set_id(markers: tuple[int, ...]) -> str:
    """枚举用例的可读 id（``[1, 9]`` 这种写法比下标好认）."""
    return "[" + ",".join(str(marker) for marker in markers) + "]" if markers else "[]"


def check_of(answer: str, size: int = 5) -> Any:
    """跑一次生成并取回核对报告（这是这一层的**公开入口**）."""
    return RAGGenerator(AnswerLLM(answer)).generate(QUESTION, packed_context(size)).check


# --------------------------------------------------------------------------- #
# 交叉核对：152 组枚举输入
# --------------------------------------------------------------------------- #


class TestCrossCheckedAgainstTheReference:
    """六个数、逐编号检查行与注记条数都要与直白实现逐位一致（152 组）."""

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    @pytest.mark.parametrize(
        "markers", ANSWER_MARKER_SETS, ids=[marker_set_id(m) for m in ANSWER_MARKER_SETS]
    )
    def test_the_six_outputs_match_the_reference(
        self, size: int, markers: tuple[int, ...]
    ) -> None:
        """六个输出值逐项一致（分母是这一次给出去的片段数）."""
        answer = answer_text(markers)
        known = tuple(range(1, size + 1))
        cited, valid, invalid, unused, coverage, grounded = reference_report(answer, known)

        check = check_of(answer, size)

        assert check.cited == cited
        assert check.valid == valid
        assert check.invalid == invalid
        assert check.unused == unused
        assert check.coverage == coverage
        assert check.grounded is grounded

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    @pytest.mark.parametrize(
        "markers", ANSWER_MARKER_SETS, ids=[marker_set_id(m) for m in ANSWER_MARKER_SETS]
    )
    def test_the_check_rows_match_the_reference(
        self, size: int, markers: tuple[int, ...]
    ) -> None:
        """逐编号的检查行（编号 / 记录 id / 有没有被引用）也要一致."""
        answer = answer_text(markers)
        expected = reference_checks(answer, tuple(range(1, size + 1)))

        check = check_of(answer, size)

        assert tuple((row.marker, row.record_id, row.used) for row in check.checks) == expected

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    @pytest.mark.parametrize(
        "markers", ANSWER_MARKER_SETS, ids=[marker_set_id(m) for m in ANSWER_MARKER_SETS]
    )
    def test_the_derived_numbers_match_the_reference(
        self, size: int, markers: tuple[int, ...]
    ) -> None:
        """两个**算出来的**数（given / hallucinated）与注记条数同样交叉核对."""
        answer = answer_text(markers)
        cited, valid, invalid, unused, _coverage, _grounded = reference_report(
            answer, tuple(range(1, size + 1))
        )

        check = check_of(answer, size)

        assert check.given == size
        assert check.given == len(valid) + len(unused)
        assert check.hallucinated == len(invalid)
        assert len(check.notes) == reference_note_count(answer, tuple(range(1, size + 1)))
        assert check.cited == cited


# --------------------------------------------------------------------------- #
# 逐编号检查行：顺序与 record_id
# --------------------------------------------------------------------------- #


class TestTheChecksTable:
    """``checks`` 是最细的一层：先 1..n（``used`` 各有真假），再接上越界编号."""

    @pytest.mark.parametrize(
        ("rows", "used"),
        [
            (HAND_CHECKS_TWO, (1, 2)),
            (HAND_CHECKS_ALL_FIVE, (1, 2, 3, 4, 5)),
            (HAND_CHECKS_NONE_USED, ()),
        ],
    )
    def test_the_rows_cover_the_prompt_table_first(
        self, rows: tuple[tuple[int, str | None, bool], ...], used: tuple[int, ...]
    ) -> None:
        """前 n 行一一对应提示词里的编号，``used`` 由答案是否引到它决定."""
        check = check_of("结论 " + " ".join(f"[{marker}]" for marker in used) + "。")

        assert tuple((row.marker, row.record_id, row.used) for row in check.checks) == rows
        assert len(check.checks) == 5

    def test_the_hallucinated_row_is_appended_last(self) -> None:
        """越界编号接在 1..n 之后（顺序 = 编号升序），且它的 ``record_id`` 必须是 ``None``."""
        check = check_of(ANSWER_MIXED)

        assert tuple((row.marker, row.record_id, row.used) for row in check.checks) == (
            HAND_CHECKS_MIXED
        )
        assert check.checks[-1].marker == 9
        assert check.checks[-1].record_id is None
        assert check.checks[-1].hallucinated is True
        assert check.checks[-1].used is True

    @pytest.mark.parametrize("extra", [(7,), (6,), (9,), (6, 9), (9, 12)])
    def test_out_of_range_rows_are_sorted_among_themselves(self, extra: tuple[int, ...]) -> None:
        """多个越界编号之间也按升序（它们不在 1..n 里，因此只能排在后面）."""
        answer = "结论 [1] " + " ".join(f"[{marker}]" for marker in extra) + "。"

        check = check_of(answer)

        assert [row.marker for row in check.checks] == [1, 2, 3, 4, 5, *sorted(extra)]
        assert [row.record_id for row in check.checks[:5]] == list(CITATION_IDS)
        assert all(row.record_id is None for row in check.checks[5:])

    def test_the_row_count_is_given_plus_hallucinated(self) -> None:
        """行数 = 给出去的片段数 + 越界编号的种类数（去重之后）."""
        check = check_of("结论 [1] [9] [9] [12]。")

        assert check.given == 5
        assert check.hallucinated == 2
        assert len(check.checks) == 7
        assert check.cited == (1, 9, 12)

    def test_every_row_of_a_clean_answer_is_marked_used(self) -> None:
        """引满五条时每一行的 ``used`` 都是 True（没有一条白给）."""
        check = check_of("结论 [1][2][3][4][5]。")

        assert tuple((row.marker, row.record_id, row.used) for row in check.checks) == (
            HAND_CHECKS_ALL_FIVE
        )
        assert all(row.used for row in check.checks)
        assert check.unused == ()

    def test_a_partially_used_table_marks_only_the_cited_rows(self) -> None:
        """只引第三条时，只有第三行的 ``used`` 是 True（其余四行是"白给了"）."""
        check = check_of("只引 [3]。")

        assert [row.marker for row in check.checks] == [1, 2, 3, 4, 5]
        assert [row.used for row in check.checks] == [False, False, True, False, False]
        assert check.checks[2].record_id == ALL_MARKER_TO_ID[3] == "c-a-02"

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    def test_the_record_ids_follow_the_documented_order(self, size: int) -> None:
        """编号 → 记录 id 的映射来自检索名次那份清单（1..8 = 库里的顺序）."""
        check = check_of("结论 [1]。", size)

        assert [row.record_id for row in check.checks] == [
            ALL_MARKER_TO_ID[marker] for marker in range(1, size + 1)
        ]
        assert check.checks[0].record_id == FULL_CITATION_IDS[0]
        assert check.checks[-1].record_id == FULL_CITATION_IDS[size - 1]

    def test_the_eight_clause_table_is_the_full_library(self) -> None:
        """8 条上下文的编号表就是库里的全部记录（id 与手算清单一致）."""
        check = check_of("结论 [1]。", 8)

        assert check.given == 8
        assert [row.record_id for row in check.checks] == list(FULL_CITATION_IDS)

    def test_the_rows_project_to_json(self) -> None:
        """每一行都要能直接 ``json.dumps``（四个键，越界那行的 id 是 null）."""
        check = check_of(ANSWER_MIXED)
        payload = [row.to_dict() for row in check.checks]

        assert set(payload[0]) == {"marker", "record_id", "hallucinated", "used"}
        assert payload[0] == {
            "marker": 1,
            "record_id": "c-a-01",
            "hallucinated": False,
            "used": True,
        }
        assert payload[-1]["record_id"] is None
        assert payload[-1]["hallucinated"] is True


# --------------------------------------------------------------------------- #
# 覆盖率：四舍五入与分母不存在
# --------------------------------------------------------------------------- #


class TestCoverageArithmetic:
    """覆盖率 = 有效引用 / 给出去的片段数，按 4 位小数收敛."""

    @pytest.mark.parametrize(("count", "used", "expected"), ROUNDING_CASES)
    def test_the_rounded_value_matches_the_hand_computation(
        self, count: int, used: int, expected: float
    ) -> None:
        """``round(used / count, 4)`` 的期望值全部写在 ``ROUNDING_CASES`` 的注释里."""
        answer = "结论 " + " ".join(f"[{marker}]" for marker in range(1, used + 1)) + "。"

        check = check_of(answer, count)

        assert check.coverage == expected
        assert check.coverage == round(used / count, 4)
        assert check.to_dict()["coverage"] == expected

    @pytest.mark.parametrize(
        ("count", "used", "expected"),
        [
            (3, 1, 0.3333),  # 1/3 = 0.33333… 截断
            (3, 2, 0.6667),  # 2/3 = 0.66666… 第 5 位是 6 → 进位
            (6, 1, 0.1667),  # 1/6 = 0.16666… 进位
            (7, 1, 0.1429),  # 1/7 = 0.142857… 第 5 位是 5 → 进位
            (7, 2, 0.2857),  # 2/7 = 0.285714… 第 5 位是 1 → 不进位
            (7, 3, 0.4286),  # 3/7 = 0.428571… 进位
            (8, 5, 0.625),  # 5/8 = 0.625 精确
            (8, 7, 0.875),  # 7/8 = 0.875 精确
        ],
    )
    def test_the_carry_and_the_truncation_both_happen(
        self, count: int, used: int, expected: float
    ) -> None:
        """进位（0.6667）与截断（0.3333）各自至少一条：口径写错会在这两组里响."""
        answer = "结论 " + " ".join(f"[{marker}]" for marker in range(1, used + 1)) + "。"

        assert check_of(answer, count).coverage == expected

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    def test_no_citation_gives_a_zero_coverage_without_a_division_error(
        self, size: int
    ) -> None:
        """分子为 0 时是 0.0（不是异常）：它是"一条都没落在提示词里"的那个数."""
        check = check_of(ANSWER_NO_CITATION, size)

        assert check.coverage == 0.0
        assert check.valid == ()
        assert check.grounded is False

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    def test_a_full_citation_gives_exactly_one(self, size: int) -> None:
        """分子等于分母时是 1.0（8/8 与 1/1 都不该出现浮点毛刺）."""
        answer = "结论 " + " ".join(f"[{marker}]" for marker in range(1, size + 1)) + "。"

        check = check_of(answer, size)

        assert check.coverage == 1.0
        assert check.unused == ()
        assert check.grounded is True

    def test_the_zero_denominator_branch_returns_zero(self) -> None:
        """**分母不存在时是 0.0**：这一支公开入口到不了（空上下文会先早退）.

        ``generate`` 的第 2 步在打包为空时直接走 ``no_context``，那条路的核对
        报告是另造的（没有任何片段可核对），因此 ``0 / 0`` 这个分支只能直接调
        ``_grounding_report`` 才走得到——而它必须返回 0.0 而不是抛
        ``ZeroDivisionError``（一份核对报告不该因为"这次什么都没给"而炸掉）。
        """
        report = _grounding_report("答案里写了 [1]。", ())

        assert report.coverage == 0.0
        assert report.cited == (1,)
        assert report.valid == ()
        assert report.invalid == (1,)
        assert report.unused == ()
        assert report.given == 0
        assert report.grounded is False

    def test_the_public_zero_denominator_path_is_the_empty_report(self) -> None:
        """公开入口上"分母为 0"的形状：``no_context`` 那份报告（全空的六个值）."""
        generation = RAGGenerator(AnswerLLM(ANSWER_NO_CITATION)).generate(
            QUESTION, empty_packed_context()
        )
        check = generation.check

        assert generation.fallback_reason == FALLBACK_REASON_NO_CONTEXT
        assert check.coverage == 0.0
        assert check.given == 0
        assert check.hallucinated == 0
        assert check.grounded is False
        assert check.summary_line() == (
            "核对：给了 0 条片段 | 答案引用 0 条（有效 0 / 幻觉 0）"
            " | 未引用 0 条 | 覆盖 0.0% | 接地未通过"
        )

    def test_the_zero_denominator_report_is_not_an_exception(self) -> None:
        """它是**合法状态**（与"检索为空"一样）：报出来，而不是抛出去."""
        report = _grounding_report("", ())

        assert report.given == 0
        assert report.coverage == 0.0
        assert len(report.notes) == 1

    def test_the_denominator_is_the_prompt_table_not_the_answer(self) -> None:
        """分母是**给出去的片段数**，不是答案里引用的条数（同一份答案两个分母两个数）."""
        assert check_of(ANSWER_OUT_OF_RANGE, 5).coverage == 0.0
        assert check_of("[1] 与 [9]。", 2).coverage == 0.5
        assert check_of("[1] 与 [9]。", 8).coverage == 0.125

    def test_the_hallucinated_clause_never_enters_the_numerator(self) -> None:
        """幻觉引用不进分子、也不进分母（它根本没有对应的片段）."""
        check = check_of("[1] [9] [12]。", 5)

        assert check.valid == (1,)
        assert check.invalid == (9, 12)
        assert check.coverage == 0.2
        assert check.given == 5


# --------------------------------------------------------------------------- #
# 接地定义
# --------------------------------------------------------------------------- #


class TestTheGroundingDefinition:
    """``grounded = valid 非空 且 invalid 为空``：它只说明"有可核对的依据"."""

    @pytest.mark.parametrize(
        ("answer", "markers", "expected"),
        [
            ("结论 [1]。", (1,), True),
            ("结论 [1] [2]。", (1, 2), True),
            ("结论 [1] [2]。", (1,), False),  # [2] 没给过 → 幻觉引用
            ("结论 [9]。", (1,), False),
            ("结论 [1] [9]。", (1,), False),
            ("结论 [9] [12]。", (1,), False),
            ("这一段没有编号。", (1,), False),
            ("结论 [0]。", (1,), False),
            ("结论 [1]。", (), False),
            ("结论 [9]。", (), False),
            ("结论 [0] [1]。", (1,), True),
        ],
    )
    def test_the_truth_table(
        self, answer: str, markers: tuple[int, ...], expected: bool
    ) -> None:
        """十一条真值表（含"分母为空"两格）：两个前提缺一个就是 False."""
        cited, valid, invalid, _unused, _coverage, grounded = reference_report(answer, markers)

        assert grounded is expected
        assert grounded is (bool(valid) and not invalid)
        expected_cited = tuple(
            sorted({marker for marker in reference_markers(answer) if marker >= 1})
        )
        assert cited == expected_cited

    @pytest.mark.parametrize(
        ("clause_count", "cited_markers"),
        [(1, (1,)), (2, (1, 2)), (3, (3,)), (5, (5,)), (8, (1, 8))],
    )
    def test_a_clean_answer_is_grounded_at_every_denominator(
        self, clause_count: int, cited_markers: tuple[int, ...]
    ) -> None:
        """只要引用的都是给过的编号，分母多大都算接地通过."""
        answer = "结论 " + " ".join(f"[{marker}]" for marker in cited_markers) + "。"

        check = check_of(answer, clause_count)

        assert check.valid == tuple(sorted(cited_markers))
        assert check.invalid == ()
        assert check.grounded is True

    @pytest.mark.parametrize("size", [2, 3, 5, 8])
    def test_one_out_of_range_marker_flips_the_flag(self, size: int) -> None:
        """一条越界编号就足以让接地不通过（哪怕其余引用全都合法）."""
        answer = "结论 [1] [2] [99]。"

        check = check_of(answer, size)

        assert check.valid == (1, 2)
        assert check.invalid == (99,)
        assert check.grounded is False
        assert check.hallucinated == 1

    def test_the_flag_is_recomputed_not_carried(self) -> None:
        """同一个答案在两个上下文下的接地结论可以不同（它是算出来的，不是抄的）."""
        answer = "[1] 与 [3]。"

        assert check_of(answer, 5).grounded is True
        assert check_of(answer, 2).grounded is False
        assert check_of(answer, 5).coverage == 0.4
        assert check_of(answer, 2).coverage == 0.5


# --------------------------------------------------------------------------- #
# 三类注记
# --------------------------------------------------------------------------- #


class TestTheGroundingReportsNotes:
    """核对结果里"必须被看见"的三件事，各自的痕迹就是一条注记."""

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 8])
    def test_the_hallucination_note_names_each_marker(self, size: int) -> None:
        """幻觉引用那条要**点名编号**，并写出那次提示词的范围（1~n）."""
        check = check_of(ANSWER_OUT_OF_RANGE, size)

        assert len(check.notes) == 0
        assert len(check.checks) == size + 1
        assert check.checks[-1].hallucinated is True

        notes = RAGGenerator(AnswerLLM(ANSWER_OUT_OF_RANGE)).generate(
            QUESTION, packed_context(size)
        ).notes

        assert any("幻觉引用 1 条：编号 [9] 不在那次提示词里" in note for note in notes)
        assert any(f"（那次只给了 [1]~[{size}]）" in note for note in notes)

    def test_the_hallucination_note_counts_the_markers(self) -> None:
        """两条越界编号时那个数字必须是 2，且编号列表按升序写出来."""
        generation = RAGGenerator(AnswerLLM("结论 [12] 与 [9]。")).generate(
            QUESTION, packed_context()
        )

        assert any(
            "幻觉引用 2 条：编号 [9, 12] 不在那次提示词里" in note
            for note in generation.notes
        )

    def test_the_unused_note_counts_the_clauses(self) -> None:
        """未引用那条要写出**条数**与编号（它是覆盖率的分母）."""
        generation = RAGGenerator(
            AnswerLLM("结论 [1] [2]。"), require_citation=True
        ).generate(QUESTION, packed_context())

        assert len(generation.notes) == 1
        assert generation.notes[0].startswith(
            "未引用的片段 3 条（编号 [3]、[4]、[5]）："
        )
        assert "这次给了 5 条、答案引用到 2 条，覆盖率 40.0%" in generation.notes[0]

    def test_the_unused_note_previews_long_lists(self) -> None:
        """超过 6 条时折成"前 6 条 + 共 N 条"（8 条上下文里只引了 1 条）."""
        generation = RAGGenerator(
            AnswerLLM("只引 [1]。"), require_citation=True
        ).generate(QUESTION, packed_context(8))

        assert "未引用的片段 7 条（编号 [2]、[3]、[4]、[5]、[6]、[7] 等 7 条）" in (
            generation.notes[0]
        )

    @pytest.mark.parametrize("answer", [ANSWER_NO_CITATION, ANSWER_OUT_OF_RANGE])
    def test_the_zero_coverage_note_separates_the_two_causes(self, answer: str) -> None:
        """覆盖率为 0 那条要写清两种成因（没写引用 vs 引了不存在的编号）."""
        generation = RAGGenerator(AnswerLLM(answer)).generate(QUESTION, packed_context())

        note = next(note for note in generation.notes if "覆盖率为 0" in note)
        assert "给了片段的那些引用一条都没落在提示词里" in note
        assert "invalid 非空（引用了不存在的编号）" in note
        assert "cited 为空（干脆没写引用）" in note

    def test_the_zero_coverage_note_is_not_rendered_without_clauses(self) -> None:
        """没有片段时那句话不渲染（"覆盖率为 0"那时只是空集的性质，不是结论）."""
        generation = RAGGenerator(AnswerLLM(ANSWER_NO_CITATION)).generate(
            QUESTION, empty_packed_context()
        )

        assert all("覆盖率为 0" not in note for note in generation.notes)
        assert any("没有任何片段可核对" in note for note in generation.check.notes)

    def test_the_noise_note_is_rendered_by_the_report_itself(self) -> None:
        """``[0]`` 那条注记属于**报告**（它不进 generation.notes：它不是一次失败）."""
        generation = RAGGenerator(AnswerLLM("见 [0] 与 [1]。")).generate(
            QUESTION, packed_context()
        )

        assert len(generation.check.notes) == 1
        assert "[0]" in generation.check.notes[0]
        assert "它既不算引用、也不算幻觉引用" in generation.check.notes[0]
        assert generation.notes == ()

    def test_the_no_citation_note_is_rendered_by_the_report_too(self) -> None:
        """"答案里没有出现任何 [n]"同样是报告的注记（与"有幻觉"是两件事）."""
        generation = RAGGenerator(AnswerLLM(ANSWER_NO_CITATION)).generate(
            QUESTION, packed_context()
        )

        assert len(generation.check.notes) == 1
        assert "答案里没有出现任何 [n]" in generation.check.notes[0]
        assert "这是**没有引用**" in generation.check.notes[0]

    def test_both_report_notes_can_appear_together(self) -> None:
        """一条 ``[0]`` 加一段没有编号的文字 → 两条注记（格式噪声 + 没有引用）."""
        check = check_of("这一段写了 [0]，但没有别的编号。")

        assert len(check.notes) == 2
        assert "[0]" in check.notes[0]
        assert "答案里没有出现任何 [n]" in check.notes[1]

    def test_the_notes_are_stable_across_context_sizes(self) -> None:
        """同一条答案的**报告注记**不随分母变化（它只说答案自己的形状）."""
        notes = {check_of("这一段没有任何引用编号。", size).notes for size in CONTEXT_SIZES}

        assert len(notes) == 1


# --------------------------------------------------------------------------- #
# 与样本手算表的接口核对
# --------------------------------------------------------------------------- #


class TestAgainstTheHandTable:
    """这一层与 ``generation_samples`` 的手算表必须逐格对上（它是全部用例的基准）."""

    @pytest.mark.parametrize("size", CONTEXT_SIZES)
    def test_the_context_citations_are_one_to_n(self, size: int) -> None:
        """上下文那一侧的输入形状：编号 1..n 连续、记录 id 与手算清单一致."""
        context = packed_context(size)

        assert tuple(citation.marker for citation in context.citations) == tuple(
            range(1, size + 1)
        )
        assert [citation.record_id for citation in context.citations] == [
            ALL_MARKER_TO_ID[marker] for marker in range(1, size + 1)
        ]

    def test_the_hand_checks_table_is_what_the_reference_builds(self) -> None:
        """样本里的三张手算检查行表就是直白实现算出来的那三张."""
        assert reference_checks("结论 [1] [2]。", (1, 2, 3, 4, 5)) == HAND_CHECKS_TWO
        assert reference_checks("结论 [1] [9]。", (1, 2, 3, 4, 5)) == HAND_CHECKS_MIXED
        assert reference_checks("没有引用。", (1, 2, 3, 4, 5)) == HAND_CHECKS_NONE_USED
        assert reference_checks("结论 [1][2][3][4][5]。", (1, 2, 3, 4, 5)) == HAND_CHECKS_ALL_FIVE

    def test_the_citation_dataclass_is_the_input_shape(self) -> None:
        """这一层的输入形状是 ``Citation``（``_grounding_report`` 只读它的 marker）."""
        citations = packed_context().citations

        assert all(isinstance(citation, Citation) for citation in citations)
        assert [citation.marker for citation in citations] == [1, 2, 3, 4, 5]
