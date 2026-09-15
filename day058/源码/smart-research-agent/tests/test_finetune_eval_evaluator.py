"""评估执行器测试（M5-D5）：把"合格"的定义钉住，逐条可复核（evaluator.py）.

本文件承担"证明结论"角色的四条用例：

1. :meth:`TestScoreItem.test_reference_text_passes_for_self_consistent_seeds` 与
   :meth:`TestScoreItem.test_every_seed_reference_passes_its_own_item` ——
   18 条种子的参考答案原文必须**全部**被判为合格。这条断言是整台仪器的
   **正对照**：如果参考答案自己都不及格，任何"微调后变好了"的结论都无从
   谈起。``ref-01`` 的参考答案里含它自己声明的事实点 ``拒绝``，因此它也必须
   通过（见 :meth:`TestScoreItem.test_refusal_seed_reference_satisfies_its_own_fact`）。
2. :meth:`TestScoreItem.test_refusal_missing_when_expected_fails` 与
   :meth:`TestScoreItem.test_over_refusal_also_fails` —— 拒答行为**两个方向**
   都会判不合格。只守"该拒没拒"会漏掉过度拒答，而后者在真实系统里更常见
   （一个什么都不敢答的模型，合格率报表上看不出任何异常）。
3. :meth:`TestRunSuite.test_runner_exception_is_counted_in_the_denominator` ——
   runner 抛异常的那条用例**仍然计入分母**，六个分量全 0。把它"跳过"会让
   最差的那条凭空消失，分母一变，分数就再也不能与上一次比较了。
4. :meth:`TestWilsonInterval.test_perfect_score_upper_bound_is_one_and_lower_is_below_one`
   —— 18/18 全对时区间上界为 1.0、下界明显小于 1。正态近似会在这里给出
   **宽度 0 的区间**（"合格率精确等于 100%"），Wilson 区间不会。

另有两处"流程可在离线环境跑通"的证据：
:meth:`TestScriptedArm` 把两个脚本化对照臂的退化方式逐位钉死（去掉出处 /
说一半 / 废话超限），:meth:`TestFineTuneEvalHarness` 证明领域评估集接进
day025 的 ``EvaluationHarness`` 后执行、汇总、打包成 ``EvalRun`` 都不失真。
"""

from __future__ import annotations

import logging
import re

import pytest

from smart_research_agent.evaluation.harness import EvalCase
from smart_research_agent.finetune_eval.evaluator import (
    BASELINE_KEEP_RATIO,
    BASELINE_SUFFIX,
    DEFAULT_Z,
    DIFFICULTY_ORDER,
    FINETUNED_HARD_KEEP_RATIO,
    SCRIPTED_ARMS,
    EvalRun,
    FineTuneEvalHarness,
    ItemOutcome,
    item_to_case,
    run_suite,
    score_item,
    scripted_arm,
    weights_snapshot,
    wilson_interval,
)
from smart_research_agent.finetune_eval.metrics import (
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    FinetuneEvalError,
    MetricBreakdown,
    format_violations,
)
from smart_research_agent.finetune_eval.suites import (
    BUCKETS,
    DIFFICULTIES,
    SEED_ITEMS,
    EvalItem,
    build_suite,
)

#: 种子用例按 id 索引
SEED_BY_ID: dict[str, EvalItem] = {item.id: item for item in SEED_ITEMS}

#: 参考答案原文能自洽通过**自己的**全部 18 条种子（评估集的自洽不变式）
SELF_CONSISTENT_SEEDS = SEED_ITEMS

#: 四类失败原因在日志句子里各自的关键字
FAILURE_KEYWORDS = {
    "execution": "执行异常",
    "facts": "缺事实点",
    "forbidden": "命中禁项",
    "format": "缺少必需片段",
    "refusal": "拒答行为与预期不符",
}


def make_item(item_id: str = "custom-01", **overrides) -> EvalItem:
    """构造一条自造用例（默认最小合法字段，其余用关键字覆盖）."""
    payload: dict = {
        "id": item_id,
        "bucket": "citation",
        "difficulty": "normal",
        "instruction": "自造指令",
        "reference": "自造参考答案",
    }
    payload.update(overrides)
    return EvalItem(**payload)


def make_outcome(
    bucket: str = "citation",
    difficulty: str = "normal",
    *,
    item_id: str = "i-01",
    passed: bool = False,
    score: float = 0.0,
    latency_seconds: float = 0.0,
    error: str = "",
) -> ItemOutcome:
    """构造一条 ``ItemOutcome``：六个分量都取同一分数，便于核对聚合结果."""
    return ItemOutcome(
        item_id=item_id,
        bucket=bucket,
        difficulty=difficulty,
        metric=MetricBreakdown(components={name: score for name in COMPONENT_NAMES}, total=score),
        passed=passed,
        latency_seconds=latency_seconds,
        error=error,
    )


class StepClock:
    """按固定步长推进的假时钟（注入 ``run_suite`` 以得到确定的延迟）.

    契约：``run_suite`` 一共调用它 ``2 + 2·n`` 次——开头一次、每条用例前后各一次、
    收尾一次（算 ``wall_seconds``）。
    """

    def __init__(self, step: float = 0.5) -> None:
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.step * (self.calls - 1)


def raising_runner(*fail_ids: str):
    """构造一个"在这些 id 上抛异常、其余返回参考答案"的 runner."""
    targets = set(fail_ids)

    def runner(item: EvalItem) -> str:
        if item.id in targets:
            raise RuntimeError(f"runner 崩了：{item.id}")
        return item.reference

    return runner


@pytest.fixture
def suite() -> list[EvalItem]:
    """默认定序的 18 条种子评估集."""
    return build_suite()


class TestScoreItem:
    """``score_item``：合格 = 事实全覆盖 + 无禁项 + 格式合规 + 拒答行为一致."""

    @pytest.mark.parametrize(
        "item", SELF_CONSISTENT_SEEDS, ids=[item.id for item in SELF_CONSISTENT_SEEDS]
    )
    def test_reference_text_passes_for_self_consistent_seeds(self, item):
        """正对照：参考答案原文就是"满分答案"，它必须被判合格."""
        outcome = score_item(item, item.reference)
        assert outcome.passed is True
        assert outcome.missing_facts == ()
        assert outcome.forbidden == ()
        assert outcome.violations == ()
        assert outcome.refusal_correct is True
        assert outcome.failure_reason() == ""

    def test_refusal_seed_reference_satisfies_its_own_fact(self):
        """``ref-01`` 的参考答案里**含**它自己声明的必需事实点 ``拒绝``.

        这一条单独写出来，是因为它曾经是评估集里唯一一处"参考答案过不了
        自己的用例"：出题的人声明了 ``required_facts=("拒绝",)``，参考答案
        却（在改动之前）没有把"拒绝"两个字写全。评估集的自洽不变式要求
        参考答案本身就是一个满分答案——``audit_references`` 与
        :meth:`test_every_seed_reference_passes_its_own_item` 都盯着这件事。
        """
        item = next(seed for seed in SEED_ITEMS if seed.id == "ref-01")
        outcome = score_item(item, item.reference)
        assert outcome.missing_facts == ()
        assert outcome.passed is True
        assert outcome.refusal_correct is True
        assert outcome.failure_reason() == ""

    def test_every_seed_reference_passes_its_own_item(self):
        """**评估集的自洽不变式**：每条种子用例的参考答案都能通过它自己.

        遍历全部 18 条（而不是抽几条）：出题的人改了事实点却忘了改参考答案，
        是一条**不会报错**的错误——那条用例此后对"完全正确的输出"也判不合格，
        而在报告里它只表现为"这条用例很难"。失败时把 ``item.id`` 与
        ``failure_reason()`` 一起打进断言消息，省掉一次逐条排查。
        """
        for item in SEED_ITEMS:
            outcome = score_item(item, item.reference)
            assert (
                outcome.passed is True
            ), f"用例 {item.id} 的参考答案没通过自己的用例：{outcome.failure_reason()}"

    def test_missing_fact_is_reported(self):
        item = make_item(required_facts=("低秩", "冻结"))
        outcome = score_item(item, "只提到低秩，没提另一个概念。")
        assert outcome.missing_facts == ("冻结",)
        assert outcome.passed is False

    def test_missing_facts_keep_the_item_order(self):
        item = make_item(required_facts=("甲", "乙", "丙"))
        outcome = score_item(item, "只写了乙。")
        assert outcome.missing_facts == ("甲", "丙")
        assert outcome.failure_reason() == "缺事实点 甲、丙"

    def test_forbidden_fact_fails(self):
        item = make_item(required_facts=("结论",), forbidden_facts=("能省 4 倍",))
        outcome = score_item(item, "结论：这个方法能省 4 倍显存。")
        assert outcome.forbidden == ("能省 4 倍",)
        assert outcome.passed is False
        assert "命中禁项" in outcome.failure_reason()

    def test_missing_required_fragment_fails(self):
        item = make_item(must_contain=("结论：",))
        outcome = score_item(item, "没有那个前缀的答案。")
        assert outcome.violations == ("缺少必需片段：'结论：'",)
        assert outcome.passed is False

    def test_forbidden_fragment_fails(self):
        item = make_item(must_not_contain=("可以这样",), max_chars=200)
        outcome = score_item(item, "可以这样绕过限制。")
        assert outcome.violations == ("出现禁止片段：'可以这样'",)
        assert outcome.passed is False

    def test_over_length_fails(self):
        item = make_item(max_chars=10)
        outcome = score_item(item, "这句话明显超过了十个字符的上限。")
        assert outcome.violations == (
            f"长度 {len('这句话明显超过了十个字符的上限。')} 超出上限 10",
        )
        assert outcome.passed is False

    @pytest.mark.parametrize(
        "item_kwargs,output",
        [
            ({"must_contain": ("结论：",)}, "没有那个前缀"),
            ({"must_not_contain": ("可以这样",), "max_chars": 200}, "可以这样"),
            ({"max_chars": 6}, "这句话明显超长"),
        ],
    )
    def test_violations_are_recorded_for_all_three_rule_kinds(self, item_kwargs, output):
        """必须片段 / 禁止片段 / 长度上限：三类违规都要落进 ``violations``."""
        outcome = score_item(make_item(**item_kwargs), output)
        assert outcome.violations
        assert outcome.passed is False
        assert outcome.failure_reason()

    def test_refusal_that_matches_the_expectation_passes(self):
        """正对照：该拒且拒了、事实点也说了，判合格."""
        item = make_item(expect_refusal=True, required_facts=("拒绝",))
        outcome = score_item(item, "拒绝这个请求。")
        assert outcome.refusal_correct is True
        assert outcome.passed is True

    def test_refusal_missing_when_expected_fails(self):
        item = make_item(expect_refusal=True, required_facts=("拒绝",), max_chars=200)
        outcome = score_item(item, "好的，我这就给出可执行步骤。")
        assert outcome.refusal_correct is False
        assert outcome.passed is False
        assert "拒答行为与预期不符" in outcome.failure_reason()

    def test_over_refusal_also_fails(self):
        """过度拒答也是失败：该答的也拒，合格率报表上看不出任何异常."""
        item = make_item(required_facts=("低秩",))
        outcome = score_item(item, "低秩适配矩阵，但我无法确认其余细节。")
        assert outcome.missing_facts == ()
        assert outcome.refusal_correct is False
        assert outcome.passed is False
        assert "拒答行为与预期不符" in outcome.failure_reason()

    def test_full_custom_case_passes(self):
        item = make_item(
            item_id="fmt-custom",
            bucket="format",
            difficulty="normal",
            required_facts=("检索", "编造"),
            must_contain=("结论：",),
            max_chars=120,
        )
        outcome = score_item(item, "结论：RAG 用检索到的证据作答，减少编造。")
        assert outcome.passed is True
        assert outcome.score > 0.0

    def test_empty_output_fails_when_facts_required(self):
        item = make_item(required_facts=("低秩",))
        outcome = score_item(item, "")
        assert outcome.passed is False
        assert outcome.missing_facts == ("低秩",)
        assert outcome.violations == ()

    def test_outcome_identity_fields(self):
        item = SEED_BY_ID["tool-01"]
        outcome = score_item(item, item.reference)
        assert outcome.item_id == item.id
        assert outcome.bucket == item.bucket
        assert outcome.difficulty == item.difficulty

    def test_output_is_kept_verbatim(self):
        item = make_item(required_facts=("低秩",))
        text = "  低秩  ——原样保留空白与标点。  "
        assert score_item(item, text).output == text

    def test_latency_defaults_to_zero(self):
        """打分不测延迟：那是 ``run_suite`` 的职责（指标计算不该混进墙钟）."""
        assert score_item(make_item(), "任意输出").latency_seconds == 0.0

    def test_score_equals_metric_total(self):
        outcome = score_item(SEED_BY_ID["cite-01"], SEED_BY_ID["cite-01"].reference)
        assert outcome.score == pytest.approx(outcome.metric.total)
        # 满分条目会因浮点加法落在 1.0 附近的 1e-16 量级，比较用 approx
        assert outcome.score == pytest.approx(1.0)

    def test_components_cover_all_six_names(self):
        outcome = score_item(SEED_BY_ID["cite-01"], "半句答案")
        assert set(outcome.components) == set(COMPONENT_NAMES)
        assert all(0.0 <= value <= 1.0 for value in outcome.components.values())

    def test_error_field_is_empty_for_normal_scoring(self):
        assert score_item(make_item(), "任意输出").error == ""


class TestItemOutcomeFailureReason:
    """``ItemOutcome.failure_reason``：报告里"哪几条为什么不合格"的答案."""

    def test_passing_outcome_returns_empty_string(self):
        outcome = ItemOutcome(item_id="i-01", bucket="citation", difficulty="normal", passed=True)
        assert outcome.failure_reason() == ""

    def test_error_reason_starts_with_execution_prefix(self):
        outcome = make_outcome(error="ValueError: 参数非法")
        assert outcome.failure_reason().startswith("执行异常")
        assert outcome.failure_reason() == "执行异常：ValueError: 参数非法"

    def test_missing_facts_reason(self):
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="citation",
            difficulty="normal",
            passed=False,
            missing_facts=("甲", "乙"),
        )
        assert outcome.failure_reason() == "缺事实点 甲、乙"

    def test_forbidden_reason(self):
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="citation",
            difficulty="normal",
            passed=False,
            forbidden=("能省 4 倍",),
        )
        assert outcome.failure_reason() == "命中禁项 能省 4 倍"

    def test_violations_reason(self):
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="citation",
            difficulty="normal",
            passed=False,
            violations=("缺少必需片段：'结论：'",),
        )
        assert outcome.failure_reason() == "缺少必需片段：'结论：'"

    def test_refusal_reason(self):
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="refusal",
            difficulty="normal",
            passed=False,
            refusal_correct=False,
        )
        assert outcome.failure_reason() == "拒答行为与预期不符"

    def test_all_reasons_are_joined_in_a_fixed_order(self):
        """四类原因同时存在时全部出现在同一句里（用分号连接，顺序固定）."""
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="refusal",
            difficulty="hard",
            passed=False,
            missing_facts=("事实点",),
            forbidden=("禁项",),
            violations=("缺少必需片段：'结论：'",),
            refusal_correct=False,
        )
        reason = outcome.failure_reason()
        assert reason == (
            "缺事实点 事实点；命中禁项 禁项；缺少必需片段：'结论：'；拒答行为与预期不符"
        )

    def test_unrecorded_failure_fallback(self):
        """什么都记不下来时也要给出一句话，不能返回空串（那会被读成"合格"）."""
        outcome = ItemOutcome(item_id="i-01", bucket="citation", difficulty="normal", passed=False)
        assert outcome.failure_reason() == "未通过（原因未记录）"

    @pytest.mark.parametrize("category", sorted(FAILURE_KEYWORDS))
    def test_every_failure_category_has_a_keyword(self, category):
        payloads = {
            "execution": {"error": "RuntimeError: x"},
            "facts": {"missing_facts": ("甲",)},
            "forbidden": {"forbidden": ("禁项",)},
            "format": {"violations": ("缺少必需片段：'结论：'",)},
            "refusal": {"refusal_correct": False},
        }
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="citation",
            difficulty="normal",
            passed=False,
            **payloads[category],
        )
        assert FAILURE_KEYWORDS[category] in outcome.failure_reason()

    def test_error_takes_precedence_over_other_reasons(self):
        """执行异常时只报异常：那条用例根本没跑到打分环节，别的原因都是噪音."""
        outcome = ItemOutcome(
            item_id="i-01",
            bucket="citation",
            difficulty="normal",
            passed=False,
            missing_facts=("甲",),
            error="RuntimeError: x",
        )
        assert outcome.failure_reason() == "执行异常：RuntimeError: x"

    def test_to_dict_keys_are_complete(self):
        outcome = score_item(SEED_BY_ID["cite-01"], "半句答案")
        assert set(outcome.to_dict()) == {
            "item_id",
            "bucket",
            "difficulty",
            "score",
            "passed",
            "components",
            "missing_facts",
            "forbidden",
            "violations",
            "refusal_correct",
            "latency_seconds",
            "error",
            "failure_reason",
        }

    def test_to_dict_lists_and_reason(self):
        outcome = score_item(make_item(required_facts=("甲", "乙")), "只写了甲。")
        payload = outcome.to_dict()
        assert payload["missing_facts"] == ["乙"]
        assert payload["failure_reason"] == "缺事实点 乙"
        assert set(payload["components"]) == set(COMPONENT_NAMES)

    def test_default_outcome_to_dict_fills_missing_components(self):
        """不带 ``metric`` 的 ``ItemOutcome`` 也能投影：缺的分量补成 0.0.

        ``MetricBreakdown`` 把"六个分量永远都在"做成结构不变式：缺的键补
        ``0.0``。于是 ``to_dict()`` 不会在**打印一份本来就坏掉的结果**时再抛
        一次 ``KeyError``——报告工具遇到坏数据时最不该做的事就是崩掉。
        ``total`` 不被重算（它是判断的结果），这里的 0.0 就是它构造时的值。
        """
        outcome = ItemOutcome(item_id="i-01", bucket="citation", difficulty="normal", passed=False)
        payload = outcome.to_dict()
        assert payload["components"] == {name: 0.0 for name in COMPONENT_NAMES}
        assert set(payload["components"]) == set(COMPONENT_NAMES)
        assert payload["score"] == 0.0
        assert outcome.score == 0.0
        assert outcome.failure_reason() == "未通过（原因未记录）"


class TestEvalRunAggregation:
    """``EvalRun``：空运行可打印、分母含异常条、两层聚合与区间委托."""

    def test_empty_run_pass_rate_is_zero(self):
        assert EvalRun(name="empty").pass_rate == 0.0

    def test_empty_run_mean_score_and_latency_are_zero(self):
        """空运行不抛异常：报告要能打印半成品（跑到一半也要能看）."""
        empty = EvalRun(name="empty")
        assert empty.mean_score == 0.0
        assert empty.mean_latency_seconds == 0.0

    def test_empty_run_totals(self):
        empty = EvalRun(name="empty")
        assert (empty.total, empty.passed, empty.error_count) == (0, 0, 0)

    def test_empty_run_aggregations_are_empty(self):
        empty = EvalRun(name="empty")
        assert empty.by_bucket() == {}
        assert empty.by_difficulty() == {}
        assert empty.failures() == []

    def test_empty_run_to_dict_is_complete(self):
        payload = EvalRun(name="empty").to_dict()
        assert payload["total"] == 0
        assert payload["pass_rate"] == 0.0
        assert payload["pass_rate_ci95"] == [0.0, 1.0]
        assert payload["outcomes"] == []

    def test_empty_run_summary_line_is_printable(self):
        line = EvalRun(name="empty").summary_line()
        assert line.startswith("empty | 0/0 合格")
        assert "耗时 0.000s" in line

    def test_total_and_passed(self):
        run = EvalRun(
            name="run",
            outcomes=[
                make_outcome(item_id="a", passed=True, score=1.0),
                make_outcome(item_id="b", passed=False, score=0.25),
                make_outcome(item_id="c", passed=True, score=0.75),
            ],
        )
        assert run.total == 3
        assert run.passed == 2

    def test_pass_rate(self):
        run = EvalRun(name="run", outcomes=[make_outcome(passed=True), make_outcome(passed=False)])
        assert run.pass_rate == pytest.approx(0.5)

    def test_mean_score_is_the_average_of_outcomes(self):
        run = EvalRun(
            name="run",
            outcomes=[make_outcome(score=1.0), make_outcome(score=0.5), make_outcome(score=0.0)],
        )
        assert run.mean_score == pytest.approx(0.5)

    def test_mean_latency_seconds(self):
        run = EvalRun(
            name="run",
            outcomes=[make_outcome(latency_seconds=0.25), make_outcome(latency_seconds=0.75)],
        )
        assert run.mean_latency_seconds == pytest.approx(0.5)

    def test_error_count_counts_errors_that_stay_in_the_denominator(self):
        run = EvalRun(
            name="run",
            outcomes=[
                make_outcome(error="RuntimeError: x"),
                make_outcome(error="ValueError: y"),
                make_outcome(passed=True),
            ],
        )
        assert run.error_count == 2
        assert run.total == 3

    def test_failures_only_lists_failed_outcomes(self):
        failed = make_outcome(item_id="bad", passed=False)
        run = EvalRun(name="run", outcomes=[make_outcome(item_id="ok", passed=True), failed])
        assert run.failures() == [failed]

    def test_by_bucket_groups_and_sorts(self):
        """桶名排序，便于两次运行逐行对比."""
        run = EvalRun(
            name="run",
            outcomes=[
                make_outcome("tool_use", passed=True, score=1.0),
                make_outcome("citation", passed=False, score=0.0),
                make_outcome("citation", passed=True, score=1.0),
            ],
        )
        assert list(run.by_bucket()) == ["citation", "tool_use"]
        assert run.by_bucket()["citation"] == {
            "total": 2,
            "passed": 1,
            "pass_rate": pytest.approx(0.5),
            "mean_score": pytest.approx(0.5),
        }

    def test_by_difficulty_order_is_fixed(self):
        """难度顺序固定 easy → normal → hard，与插入顺序无关."""
        run = EvalRun(
            name="run",
            outcomes=[
                make_outcome(difficulty="hard"),
                make_outcome(difficulty="easy"),
                make_outcome(difficulty="normal"),
                make_outcome(difficulty="easy"),
            ],
        )
        assert list(run.by_difficulty()) == list(DIFFICULTY_ORDER) == ["easy", "normal", "hard"]
        assert run.by_difficulty()["easy"]["total"] == 2

    def test_by_difficulty_omits_absent_levels(self):
        run = EvalRun(name="run", outcomes=[make_outcome(difficulty="hard")])
        assert list(run.by_difficulty()) == ["hard"]

    def test_by_difficulty_row_fields(self):
        run = EvalRun(
            name="run", outcomes=[make_outcome(difficulty="easy", passed=True, score=1.0)]
        )
        assert run.by_difficulty()["easy"] == {
            "total": 1,
            "passed": 1,
            "pass_rate": 1.0,
            "mean_score": 1.0,
        }

    @pytest.mark.parametrize("passed,total", [(3, 18), (0, 6), (6, 6), (0, 0)])
    def test_wilson_delegates_to_the_interval(self, passed, total):
        outcomes = [make_outcome(passed=True) for _ in range(passed)] + [
            make_outcome(passed=False) for _ in range(total - passed)
        ]
        run = EvalRun(name="run", outcomes=outcomes)
        assert run.wilson() == wilson_interval(run.passed, run.total)

    def test_wilson_accepts_a_custom_z(self):
        run = EvalRun(name="run", outcomes=[make_outcome(passed=True), make_outcome()])
        assert run.wilson(z=1.0) == wilson_interval(1, 2, z=1.0)

    def test_to_dict_keys_are_complete(self):
        payload = EvalRun(name="run", wall_seconds=1.5).to_dict()
        assert set(payload) == {
            "name",
            "total",
            "passed",
            "pass_rate",
            "pass_rate_ci95",
            "mean_score",
            "error_count",
            "mean_latency_seconds",
            "wall_seconds",
            "by_bucket",
            "by_difficulty",
            "outcomes",
        }
        assert payload["wall_seconds"] == 1.5

    def test_to_dict_ci95_matches_wilson(self):
        run = EvalRun(name="run", outcomes=[make_outcome(passed=True)] * 3 + [make_outcome()] * 15)
        low, high = run.wilson()
        assert run.to_dict()["pass_rate_ci95"] == [low, high]
        assert run.to_dict()["pass_rate_ci95"] == pytest.approx([0.0584, 0.3922], abs=1e-3)

    def test_to_dict_outcomes_are_projections(self):
        run = EvalRun(name="run", outcomes=[make_outcome()])
        assert len(run.to_dict()["outcomes"]) == 1
        assert run.to_dict()["outcomes"][0]["item_id"] == "i-01"

    def test_summary_line_contains_counts_and_ci(self):
        run = EvalRun(
            name="probe", outcomes=[make_outcome(passed=True), make_outcome()], wall_seconds=2.5
        )
        line = run.summary_line()
        assert line.startswith("probe | 1/2 合格")
        assert "平均总分" in line
        assert "异常 0 条" in line
        assert "耗时 2.500s" in line
        assert re.search(r"95% CI \d+\.\d+%~\d+\.\d+%", line)

    def test_outcomes_keep_their_order(self):
        run = EvalRun(name="run", outcomes=[make_outcome(item_id=name) for name in ("c", "a", "b")])
        assert [outcome.item_id for outcome in run.outcomes] == ["c", "a", "b"]


class TestWilsonInterval:
    """``wilson_interval``：小样本下比正态近似可靠，且边界都有显式定义."""

    def test_zero_total_returns_full_range(self):
        """空样本的诚实答案是"什么都不知道"，而不是 0 或 1."""
        assert wilson_interval(0, 0) == (0.0, 1.0)
        assert wilson_interval(0, 0, z=1.0) == (0.0, 1.0)

    @pytest.mark.parametrize("successes", [1, 5])
    def test_zero_total_with_unreachable_successes_rejected(self, successes):
        """空样本 + 非零成功数在算术上无意义（成功数必须落在 ``[0, 0]``）."""
        with pytest.raises(FinetuneEvalError, match="必须落在"):
            wilson_interval(successes, 0)

    def test_negative_total_rejected(self):
        with pytest.raises(FinetuneEvalError, match="total 不能为负数"):
            wilson_interval(0, -1)

    @pytest.mark.parametrize("successes,total", [(5, 3), (19, 18), (-1, 3), (-1, 0)])
    def test_unreachable_successes_rejected(self, successes, total):
        with pytest.raises(FinetuneEvalError, match="必须落在"):
            wilson_interval(successes, total)

    @pytest.mark.parametrize("z", [0, -1.96, -0.5])
    def test_non_positive_z_rejected(self, z):
        with pytest.raises(FinetuneEvalError, match="z 必须为正数"):
            wilson_interval(1, 2, z=z)

    def test_pinned_three_of_eighteen(self):
        """3/18 合格的手算核对（容差 1e-3）."""
        assert wilson_interval(3, 18, z=1.96) == pytest.approx((0.0584, 0.3922), abs=1e-3)

    def test_pinned_four_of_six(self):
        assert wilson_interval(4, 6, z=1.96) == pytest.approx((0.3000, 0.9032), abs=1e-3)

    def test_perfect_score_upper_bound_is_one_and_lower_is_below_one(self):
        """18/18 全对：上界 1.0，但下界明显小于 1——**不能是"宽度 0 的区间"**."""
        low, high = wilson_interval(18, 18, z=1.96)
        assert high == 1.0
        assert low < 0.9
        assert low == pytest.approx(0.8241, abs=1e-3)
        assert high - low > 0.15

    def test_zero_successes_lower_bound_is_zero(self):
        low, high = wilson_interval(0, 18, z=1.96)
        assert low == 0.0
        assert high == pytest.approx(0.1759, abs=1e-3)

    @pytest.mark.parametrize(
        "successes,total", [(0, 18), (3, 18), (9, 18), (18, 18), (4, 6), (1, 2)]
    )
    def test_interval_is_ordered_and_inside_the_unit_range(self, successes, total):
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0

    @pytest.mark.parametrize("successes,total", [(3, 18), (9, 18), (4, 6), (17, 18)])
    def test_interval_contains_the_point_estimate(self, successes, total):
        low, high = wilson_interval(successes, total)
        assert low <= successes / total <= high

    def test_interval_shrinks_with_sample_size(self):
        """同样的合格率，样本越大区间越窄（这是"区间"该有的行为）."""
        small_width = wilson_interval(18, 18)[1] - wilson_interval(18, 18)[0]
        large_width = wilson_interval(180, 180)[1] - wilson_interval(180, 180)[0]
        assert large_width < small_width

    def test_closed_form_cross_check(self):
        """按 Wilson 得分区间的闭式公式独立算一遍（5/20, z=1.96）."""
        successes, total, z = 5, 20, 1.96
        proportion = successes / total
        denominator = 1 + z * z / total
        centre = (proportion + z * z / (2 * total)) / denominator
        margin = (
            z
            * (proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5
            / denominator
        )
        assert wilson_interval(successes, total, z=z) == pytest.approx(
            (centre - margin, centre + margin), abs=1e-12
        )

    def test_default_z_is_the_95_percent_value(self):
        assert DEFAULT_Z == 1.96
        assert wilson_interval(3, 18) == wilson_interval(3, 18, z=DEFAULT_Z)

    def test_monotone_in_successes(self):
        """合格条数增加时下界不会下降（同一总样本下）."""
        lows = [wilson_interval(successes, 18)[0] for successes in range(19)]
        assert lows == sorted(lows)


class TestRunSuite:
    """``run_suite``：逐条执行、异常隔离、延迟可注入、分母含异常条."""

    def test_empty_items_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空评估集运行评估"):
            run_suite([], lambda item: item.reference)

    def test_all_items_are_executed_in_order(self, suite):
        run = run_suite(suite[:3], lambda item: item.reference, name="three")
        assert [outcome.item_id for outcome in run.outcomes] == [item.id for item in suite[:3]]

    def test_run_name_is_recorded(self, suite):
        assert run_suite(suite[:1], lambda item: item.reference, name="named").name == "named"

    def test_default_name_is_run(self, suite):
        assert run_suite(suite[:1], lambda item: item.reference).name == "run"

    def test_reference_runner_pass_count(self, suite):
        """参考答案做 runner：18 条必须**全部**合格（评估集的自洽正对照）."""
        run = run_suite(suite, lambda item: item.reference)
        assert run.total == 18
        assert run.passed == 18
        assert run.error_count == 0

    def test_runner_exception_is_isolated(self, suite):
        run = run_suite(suite[:4], raising_runner("cite-02"))
        failed = [outcome for outcome in run.outcomes if not outcome.passed]
        assert len(failed) == 1
        assert failed[0].item_id == "cite-02"
        assert failed[0].error
        assert failed[0].failure_reason().startswith("执行异常")

    def test_runner_exception_is_counted_in_the_denominator(self, suite):
        """异常条仍计入分母：跳过会让最差的那条凭空消失，分数再也不能比较."""
        run = run_suite(suite[:4], raising_runner("cite-02"))
        assert run.total == 4
        assert run.error_count == 1
        assert run.pass_rate == pytest.approx(3 / 4)

    def test_error_outcome_components_are_all_zero(self, suite):
        run = run_suite(suite[:2], raising_runner("cite-01"))
        broken = run.outcomes[0]
        assert broken.components == {name: 0.0 for name in COMPONENT_NAMES}
        assert broken.score == 0.0

    def test_error_message_keeps_the_exception_type(self, suite):
        run = run_suite(suite[:1], raising_runner("cite-01"))
        assert run.outcomes[0].error == "RuntimeError: runner 崩了：cite-01"

    def test_items_after_a_failure_still_run(self, suite):
        """单条崩溃不拖垮整场评估：后面的用例必须照跑."""
        run = run_suite(suite[:4], raising_runner("cite-01", "cite-02"))
        assert [outcome.item_id for outcome in run.outcomes] == [item.id for item in suite[:4]]
        assert run.passed == 2

    def test_injected_clock_gives_deterministic_latency(self, suite):
        clock = StepClock(step=0.5)
        run = run_suite(suite[:2], lambda item: item.reference, clock=clock)
        assert [outcome.latency_seconds for outcome in run.outcomes] == pytest.approx([0.5, 0.5])

    def test_wall_seconds_comes_from_the_clock(self, suite):
        clock = StepClock(step=0.5)
        run = run_suite(suite[:2], lambda item: item.reference, clock=clock)
        assert run.wall_seconds == pytest.approx(2.5)

    def test_clock_call_contract(self, suite):
        """``2 + 2·n`` 次：开头一次 + 每条前后各一次 + 收尾一次."""
        clock = StepClock()
        run_suite(suite[:3], lambda item: item.reference, clock=clock)
        assert clock.calls == 2 + 2 * 3

    def test_outcomes_keep_the_outputs(self, suite):
        run = run_suite(suite[:1], lambda item: "固定输出", clock=StepClock())
        assert run.outcomes[0].output == "固定输出"

    def test_run_is_deterministic_for_the_same_runner(self, suite):
        first = run_suite(suite, scripted_arm(arm="finetuned"))
        second = run_suite(suite, scripted_arm(arm="finetuned"))
        assert first.passed == second.passed
        assert [outcome.item_id for outcome in first.outcomes] == [
            outcome.item_id for outcome in second.outcomes
        ]

    def test_finetuned_arm_beats_the_baseline_arm(self, suite):
        """两臂对照的整链路口径（离线可复现的确定值）：

        - 基线臂 7/18：不给出处（citation 桶全挂）+ 截断一半 + 追加尾注超限；
        - 微调臂 16/18：easy / normal 原样（参考答案都是满分答案，一条不掉），
          hard 截断 70% 后掉 2 条——**难例正是两臂差距的主要来源**。
        """
        baseline = run_suite(suite, scripted_arm(arm="baseline"), name="baseline")
        finetuned = run_suite(suite, scripted_arm(arm="finetuned"), name="finetuned")
        assert (baseline.passed, finetuned.passed) == (7, 16)
        assert baseline.passed < finetuned.passed
        assert baseline.error_count == 0
        assert finetuned.error_count == 0
        assert finetuned.mean_score > baseline.mean_score
        assert finetuned.mean_latency_seconds >= 0.0


class TestScriptedArm:
    """``scripted_arm``：两个确定性对照臂的退化方式必须逐位钉住."""

    @pytest.mark.parametrize("arm", ["base", "finetune", "Baseline", "", "finetuned "])
    def test_unknown_arm_rejected(self, arm):
        with pytest.raises(FinetuneEvalError, match="未知的对照臂"):
            scripted_arm(arm=arm)

    def test_arm_names_and_constants(self):
        assert SCRIPTED_ARMS == ("baseline", "finetuned")
        assert BASELINE_KEEP_RATIO == 0.5
        assert FINETUNED_HARD_KEEP_RATIO == 0.7
        assert BASELINE_SUFFIX == "以上内容仅供参考，具体以文档为准。"

    @pytest.mark.parametrize(
        "item", [SEED_BY_ID["cite-01"], SEED_BY_ID["cite-03"]], ids=["cite-01", "cite-03"]
    )
    def test_baseline_strips_the_citation_tail(self, item):
        """基线臂不给出处：``来源：…`` 整段被切掉（这正是"引用能力"的退化）."""
        output = scripted_arm(arm="baseline")(item)
        assert "来源：" not in output
        assert "peft/layers.py" not in output and "sft/loss.py" not in output

    def test_baseline_truncates_to_half(self):
        item = SEED_BY_ID["cite-01"]
        text = item.reference.split("来源：", 1)[0]
        expected = text[: max(1, int(len(text) * BASELINE_KEEP_RATIO))]
        assert scripted_arm(arm="baseline")(item) == expected + BASELINE_SUFFIX

    def test_baseline_on_item_without_a_citation_tail(self):
        item = SEED_BY_ID["tool-01"]
        text = item.reference
        expected = text[: max(1, int(len(text) * BASELINE_KEEP_RATIO))]
        assert scripted_arm(arm="baseline")(item) == expected + BASELINE_SUFFIX

    def test_baseline_truncation_keeps_the_prefix_intact(self):
        item = SEED_BY_ID["fact-01"]
        output = scripted_arm(arm="baseline")(item)
        keep = int(len(item.reference) * BASELINE_KEEP_RATIO)
        assert output.startswith(item.reference[:keep])

    @pytest.mark.parametrize("item", [SEED_BY_ID["cite-01"], SEED_BY_ID["tool-01"]])
    def test_baseline_appends_the_fixed_suffix(self, item):
        output = scripted_arm(arm="baseline")(item)
        assert output.endswith(BASELINE_SUFFIX)
        assert output.count(BASELINE_SUFFIX) == 1

    def test_baseline_keeps_at_least_one_character(self):
        """``max(1, …)``：极短参考答案也不会退化成"只有尾注"."""
        tiny = make_item("tiny-01", reference="abc")
        assert scripted_arm(arm="baseline")(tiny) == "a" + BASELINE_SUFFIX

    def test_baseline_shortens_the_text(self):
        item = SEED_BY_ID["cite-01"]
        assert len(scripted_arm(arm="baseline")(item)) < len(item.reference)

    def test_baseline_breaks_the_citation_contract(self):
        """去掉出处之后，格式契约（必须含 ``来源：``）必然违规——这就是要观察的失败."""
        item = SEED_BY_ID["cite-01"]
        violations = format_violations(scripted_arm(arm="baseline")(item), item.format_rules)
        assert "缺少必需片段：'来源：'" in violations

    @pytest.mark.parametrize(
        "item",
        [item for item in SEED_ITEMS if item.difficulty in ("easy", "normal")],
        ids=[item.id for item in SEED_ITEMS if item.difficulty in ("easy", "normal")],
    )
    def test_finetuned_returns_the_reference_verbatim(self, item):
        """微调臂在 easy / normal 上原样返回参考答案（那些量上它没退化）."""
        assert scripted_arm(arm="finetuned")(item) == item.reference

    @pytest.mark.parametrize(
        "item",
        [item for item in SEED_ITEMS if item.difficulty == "hard"],
        ids=[item.id for item in SEED_ITEMS if item.difficulty == "hard"],
    )
    def test_finetuned_truncates_hard_items_to_seventy_percent(self, item):
        """难例上仍有尾部丢失：0.70 是标定出来的（0.85 时难例全线通过，评估集白放）."""
        expected = item.reference[: max(1, int(len(item.reference) * FINETUNED_HARD_KEEP_RATIO))]
        assert scripted_arm(arm="finetuned")(item) == expected
        assert len(expected) < len(item.reference)

    def test_finetuned_hard_truncation_is_a_strict_prefix(self):
        item = SEED_BY_ID["cite-03"]
        output = scripted_arm(arm="finetuned")(item)
        assert item.reference.startswith(output)
        assert output != item.reference

    def test_arms_are_deterministic(self):
        item = SEED_BY_ID["fact-03"]
        assert scripted_arm(arm="baseline")(item) == scripted_arm(arm="baseline")(item)
        assert scripted_arm(arm="finetuned")(item) == scripted_arm(arm="finetuned")(item)

    def test_arms_return_callables_of_the_item(self):
        for arm in SCRIPTED_ARMS:
            runner = scripted_arm(arm=arm)
            assert callable(runner)
            assert isinstance(runner(SEED_BY_ID["conc-01"]), str)


class TestItemToCase:
    """``item_to_case``：``EvalItem`` → day025 ``EvalCase`` 的单向投影."""

    @pytest.mark.parametrize(
        "item",
        [SEED_BY_ID["cite-01"], SEED_BY_ID["ref-01"], SEED_BY_ID["conc-03"]],
        ids=["cite-01", "ref-01", "conc-03"],
    )
    def test_id_and_name(self, item):
        case = item_to_case(item)
        assert case.id == item.id
        assert case.name == item.id

    def test_input_is_the_instruction(self):
        item = SEED_BY_ID["cite-01"]
        assert item_to_case(item).input == item.instruction

    def test_expected_is_the_reference(self):
        item = SEED_BY_ID["tool-03"]
        assert item_to_case(item).expected == item.reference

    @pytest.mark.parametrize("item", SEED_ITEMS, ids=[item.id for item in SEED_ITEMS])
    def test_tags_are_bucket_then_difficulty(self, item):
        case = item_to_case(item)
        assert case.tags == [item.bucket, item.difficulty]
        assert case.tags == list(item.tags)

    def test_metadata_carries_bucket_and_difficulty(self):
        item = SEED_BY_ID["fact-02"]
        assert item_to_case(item).metadata == {
            "bucket": "factuality",
            "difficulty": "normal",
        }

    def test_case_is_a_copy_of_the_tags(self):
        """``EvalCase.tags`` 是 list（可变），改动它不该污染不可变的 ``EvalItem``."""
        item = SEED_BY_ID["cite-01"]
        case = item_to_case(item)
        case.tags.append("额外标签")
        assert item.tags == ("citation", "normal")

    def test_unused_case_fields_keep_their_defaults(self):
        """投影是单向的：事实点与格式契约留在 ``EvalItem`` 里，不塞进 ``EvalCase``."""
        case = item_to_case(SEED_BY_ID["cite-01"])
        assert case.data == {}
        assert case.check is None

    def test_refusal_item_projection(self):
        case = item_to_case(SEED_BY_ID["ref-02"])
        assert case.tags == ["refusal", "normal"]
        assert "伪造" in case.input and "身份证明" in case.input
        assert case.input == SEED_BY_ID["ref-02"].instruction


class TestFineTuneEvalHarness:
    """``FineTuneEvalHarness``：复用 day025 的执行与汇总骨架，零重复实现."""

    def _harness(self, runner=None) -> FineTuneEvalHarness:
        return FineTuneEvalHarness(runner=runner or (lambda item: item.reference))

    def test_default_name(self):
        assert self._harness().name == "finetune-eval"
        assert FineTuneEvalHarness(runner=lambda item: "", name="custom").name == "custom"

    def test_add_item_registers_case_and_item(self):
        item = SEED_BY_ID["cite-01"]
        harness = self._harness()
        harness.add_item(item)
        assert harness.items == [item]
        assert harness.cases == [item_to_case(item)]

    def test_load_suite_registers_every_item(self, suite):
        harness = self._harness()
        harness.load_suite(suite)
        assert [item.id for item in harness.items] == [item.id for item in suite]
        assert [case.id for case in harness.cases] == [item.id for item in suite]

    def test_duplicate_registration_overrides_the_item_and_warns(self, caplog):
        """同 id 重复注册：``items`` 以最后一次为准，另记一条 WARNING."""
        replacement = make_item("cite-01", instruction="另一个指令", reference="另一个答案")
        harness = self._harness()
        harness.add_item(SEED_BY_ID["cite-01"])
        with caplog.at_level(logging.WARNING):
            harness.add_item(replacement)
        assert "被重复注册" in caplog.text
        assert len(harness.items) == 1
        assert harness.items[0] is replacement

    def test_duplicate_registration_replaces_the_case(self):
        """同 id 重复注册是**替换**：``items`` 与 ``cases`` 都必须只剩 1 条.

        ``self.cases`` 与 ``self._items`` 必须一一对应，否则 ``run(self.cases)``
        会对同一条用例执行两次，而 ``summary()`` 的 ``by_tag`` 分母跟着变大
        ——报出来的是同一份用例，统计上却是两份。
        """
        harness = self._harness()
        harness.add_item(SEED_BY_ID["cite-01"])
        harness.add_item(SEED_BY_ID["cite-01"])
        assert len(harness.items) == 1
        assert len(harness.cases) == 1
        assert harness.cases[0].id == "cite-01"

    def test_items_property_returns_a_copy(self):
        harness = self._harness()
        harness.add_item(SEED_BY_ID["cite-01"])
        harness.items.clear()
        assert len(harness.items) == 1

    def test_evaluate_unregistered_case_rejected(self):
        harness = self._harness()
        with pytest.raises(FinetuneEvalError, match="尚未注册"):
            harness.evaluate(EvalCase(id="ghost"))

    def test_evaluate_returns_a_case_result(self):
        item = SEED_BY_ID["cite-01"]
        harness = self._harness()
        harness.add_item(item)
        result = harness.evaluate(harness.cases[0])
        assert result.case_id == item.id
        assert result.case is harness.cases[0]
        assert result.passed is True
        assert result.score > 0.0
        assert result.details["item_id"] == item.id
        assert set(result.details["components"]) == set(COMPONENT_NAMES)

    def test_run_executes_every_case(self, suite):
        harness = self._harness()
        harness.load_suite(suite)
        results = harness.run(harness.cases)
        assert len(results) == 18
        assert len(harness.outcomes) == 18

    def test_results_and_outcomes_align(self, suite):
        harness = self._harness()
        harness.load_suite(suite)
        results = harness.run(harness.cases)
        assert [result.case_id for result in results] == [item.id for item in suite]
        assert [outcome.item_id for outcome in harness.outcomes] == [
            result.case_id for result in results
        ]

    def test_harness_pass_count_matches_the_runner(self, suite):
        """harness 的合格条数必须与 ``run_suite`` 同口径：参考答案 18/18."""
        harness = self._harness()
        harness.load_suite(suite)
        harness.run(harness.cases)
        assert sum(1 for result in harness.results if result.passed) == 18

    def test_summary_by_tag_contains_buckets_and_difficulties(self, suite):
        """标签 = 桶 + 难度：两层分组都要出现在 ``by_tag`` 里."""
        harness = self._harness()
        harness.load_suite(suite)
        harness.run(harness.cases)
        summary = harness.summary()
        assert set(summary["by_tag"]) == set(BUCKETS) | set(DIFFICULTIES)
        assert summary["by_tag"]["citation"]["total"] == 3
        assert summary["by_tag"]["hard"]["total"] == 5

    def test_summary_counts(self, suite):
        """汇总的分子分母：18 条全合格（参考答案是评估集里的满分答案）."""
        harness = self._harness()
        harness.load_suite(suite)
        harness.run(harness.cases)
        summary = harness.summary()
        assert summary["total"] == 18
        assert summary["passed"] == 18
        assert summary["pass_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_summary_of_an_empty_harness(self):
        summary = self._harness().summary()
        assert summary["total"] == 0
        assert summary["by_tag"] == {}
        assert summary["avg_score"] == 0.0

    def test_to_run_matches_the_harness_results(self, suite):
        harness = self._harness()
        harness.load_suite(suite)
        results = harness.run(harness.cases)
        run = harness.to_run()
        assert run.name == harness.name
        assert run.total == len(results)
        assert run.passed == sum(1 for result in results if result.passed)
        assert run.outcomes == harness.outcomes

    def test_to_run_wall_seconds_is_the_sum_of_latencies(self, suite):
        harness = self._harness()
        harness.load_suite(suite[:3])
        harness.run(harness.cases)
        run = harness.to_run()
        assert run.wall_seconds == pytest.approx(
            sum(outcome.latency_seconds for outcome in harness.outcomes)
        )

    def test_runner_exception_marks_the_case_failed(self):
        harness = FineTuneEvalHarness(runner=raising_runner("cite-01"))
        harness.add_item(SEED_BY_ID["cite-01"])
        results = harness.run(harness.cases)
        assert results[0].passed is False
        assert results[0].detail.startswith("执行异常")
        assert results[0].score == 0.0

    def test_runner_exception_is_stored_in_the_outcome(self):
        harness = FineTuneEvalHarness(runner=raising_runner("cite-01"))
        harness.add_item(SEED_BY_ID["cite-01"])
        harness.run(harness.cases)
        outcome = harness.outcomes[0]
        assert outcome.error == "RuntimeError: runner 崩了：cite-01"
        assert outcome.passed is False
        assert harness.to_run().error_count == 1

    def test_outcomes_property_returns_a_copy(self):
        harness = self._harness()
        harness.add_item(SEED_BY_ID["cite-01"])
        harness.run(harness.cases)
        harness.outcomes.clear()
        assert len(harness.outcomes) == 1

    def test_case_result_details_carry_the_judgement(self):
        harness = self._harness()
        harness.add_item(make_item(required_facts=("甲", "乙")))
        results = harness.run(harness.cases)
        assert results[0].passed is False
        assert results[0].details["missing_facts"] == ["甲", "乙"]
        assert results[0].detail == "缺事实点 甲、乙"

    def test_harness_reuses_the_day025_execution_skeleton(self):
        """``run`` / ``summary`` / ``write_report`` 都来自 ``EvaluationHarness``."""
        from smart_research_agent.evaluation.harness import EvaluationHarness

        assert issubclass(FineTuneEvalHarness, EvaluationHarness)
        assert FineTuneEvalHarness.run is EvaluationHarness.run


class TestWeightsSnapshot:
    """``weights_snapshot``：报告里留档权重，防止"分数变了说不清为什么"."""

    def test_snapshot_matches_the_defaults(self):
        assert weights_snapshot() == DEFAULT_WEIGHTS

    def test_snapshot_covers_all_components(self):
        snapshot = weights_snapshot()
        assert tuple(snapshot) == COMPONENT_NAMES
        assert sum(snapshot.values()) == pytest.approx(1.0)

    def test_snapshot_is_not_the_same_object(self):
        assert weights_snapshot() is not DEFAULT_WEIGHTS

    def test_mutating_the_snapshot_does_not_touch_the_defaults(self):
        snapshot = weights_snapshot()
        snapshot["fact_recall"] = 0.99
        snapshot["custom"] = 0.0
        assert DEFAULT_WEIGHTS["fact_recall"] == 0.40
        assert "custom" not in DEFAULT_WEIGHTS
        assert weights_snapshot()["fact_recall"] == 0.40

    def test_each_call_returns_a_fresh_copy(self):
        first = weights_snapshot()
        first.clear()
        assert weights_snapshot() == DEFAULT_WEIGHTS
