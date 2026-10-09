"""``math_foundations.bridge``：与项目既有实现的八项对照（day073）.

本文件是这一课"数学与工程必须对得上"的**证据**：

```text
七项逐位一致       cosine / softmax / log_softmax / cross_entropy /
                   perplexity / sigmoid / log_sigmoid
一项记录差异       normalize（零向量：本包拒绝、生产实现原样返回，两条都对）
零项分歧           分歧才是需要改代码的信号
```

另外两条手算性质：``σ(0) = 0.5``、``log σ(0) = −ln 2``、
以及 ``log σ(−800)`` 仍然是有限值（朴素写法会崩）。
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.bridge import (
    CHECK_FUNCTIONS,
    LOGIT_SAMPLES,
    SCALAR_SAMPLES,
    VECTOR_PAIRS,
    BridgeReport,
    CheckOutcome,
    _compare_points,
    check_cosine,
    check_cross_entropy,
    check_log_sigmoid,
    check_log_softmax,
    check_normalize,
    check_perplexity,
    check_sigmoid,
    check_softmax,
    cross_check_all,
    log_sigmoid,
    one_hot,
    sample_outputs,
    softplus,
)
from smart_research_agent.math_foundations.errors import MathError, NumericError
from smart_research_agent.math_foundations.types import BRIDGE_TARGETS
from tests.math_samples import approx


class TestSchedulingTable:
    """八项对照的调度表：键与 BRIDGE_TARGETS 逐键对齐，顺序一致."""

    def test_keys_match_the_target_table(self) -> None:
        assert set(CHECK_FUNCTIONS) == set(BRIDGE_TARGETS)
        assert tuple(CHECK_FUNCTIONS) == BRIDGE_TARGETS

    def test_every_target_returns_a_verdict(self) -> None:
        for name, function in CHECK_FUNCTIONS.items():
            outcome = function()
            assert outcome.target == name
            assert outcome.status in ("agrees", "differs", "rejected")


class TestIndividualChecks:
    """逐项对照的结论（每一项都要能独立跑、独立读）."""

    def test_cosine_agrees(self) -> None:
        outcome = check_cosine()
        assert outcome.status == "agrees"
        assert outcome.compared_points == len(VECTOR_PAIRS)
        assert outcome.max_absolute_error == 0.0

    def test_normalize_records_the_zero_vector_convention(self) -> None:
        outcome = check_normalize()
        assert outcome.status == "rejected"
        assert outcome.ok is True  # 记录差异不等于分歧
        assert "零向量" in outcome.message

    def test_softmax_agrees_with_both_production_implementations(self) -> None:
        outcome = check_softmax()
        assert outcome.status == "agrees"
        assert outcome.compared_points == 2 * sum(len(item) for item in LOGIT_SAMPLES)

    def test_log_softmax_agrees(self) -> None:
        assert check_log_softmax().status == "agrees"

    def test_cross_entropy_agrees_on_one_hot_truths(self) -> None:
        outcome = check_cross_entropy()
        assert outcome.status == "agrees"
        assert outcome.max_absolute_error < 1e-15  # 浮点求和顺序之差
        assert "logits" in outcome.message

    def test_perplexity_agrees(self) -> None:
        assert check_perplexity().status == "agrees"

    def test_sigmoid_agrees(self) -> None:
        outcome = check_sigmoid()
        assert outcome.status == "agrees"
        assert outcome.compared_points == len(SCALAR_SAMPLES)

    def test_log_sigmoid_agrees(self) -> None:
        assert check_log_sigmoid().status == "agrees"


class TestFullReport:
    """汇总报告：八项齐全、结论可读、JSON 友好."""

    def test_report_covers_all_targets_in_order(self) -> None:
        report = cross_check_all()
        assert tuple(item.target for item in report.outcomes) == BRIDGE_TARGETS

    def test_report_has_no_disagreement(self) -> None:
        report = cross_check_all()
        assert report.ok is True
        assert report.disagreements == ()
        assert len(report.recorded_differences) == 1
        assert report.recorded_differences[0].target == "normalize"

    def test_report_summary_counts(self) -> None:
        summary = cross_check_all().summary_line()
        assert "对照 8 项" in summary
        assert "一致 7" in summary
        assert "记录差异 1" in summary
        assert "分歧 0" in summary

    def test_report_is_json_friendly(self) -> None:
        payload = cross_check_all().to_dict()
        json.dumps(payload)
        assert payload["ok"] is True
        assert payload["disagreements"] == []
        assert len(payload["outcomes"]) == 8
        assert payload["notes"]

    def test_missing_target_is_rejected(self) -> None:
        with pytest.raises(MathError):
            BridgeReport(outcomes=(CheckOutcome(
                target="cosine",
                status="agrees",
                max_absolute_error=0.0,
                compared_points=1,
                message="x",
                source="s",
            ),))

    def test_custom_tolerance_is_validated(self) -> None:
        assert cross_check_all(tolerance=1e-6).tolerance == 1e-6
        with pytest.raises(NumericError):
            cross_check_all(tolerance=0.0)


class TestOutcomeShape:
    """结论形状：只允许三种状态、误差不能为负、JSON 可读."""

    def test_projection(self) -> None:
        outcome = CheckOutcome(
            target="cosine",
            status="differs",
            max_absolute_error=0.5,
            compared_points=3,
            message="差得有点多",
            source="smart_research_agent.x.y",
        )
        assert outcome.ok is False
        payload = outcome.to_dict()
        assert payload["ok"] is False
        assert payload["description"]
        assert "differs" in outcome.summary_line()

    def test_unknown_target_is_rejected(self) -> None:
        with pytest.raises(MathError):
            CheckOutcome(
                target="unknown",
                status="agrees",
                max_absolute_error=0.0,
                compared_points=0,
                message="x",
                source="s",
            )

    def test_unknown_status_is_rejected(self) -> None:
        with pytest.raises(MathError):
            CheckOutcome(
                target="cosine",
                status="maybe",
                max_absolute_error=0.0,
                compared_points=0,
                message="x",
                source="s",
            )

    def test_negative_error_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            CheckOutcome(
                target="cosine",
                status="agrees",
                max_absolute_error=-1.0,
                compared_points=0,
                message="x",
                source="s",
            )


class TestCompareHelper:
    """对照的比较器本身：三种结论各自的触发条件（**它不能被静默跳过**）."""

    def _raiser(self, message: str):
        def _fail() -> list[float]:
            raise MathError(message)

        return _fail

    def test_one_side_rejecting_is_recorded_not_ignored(self) -> None:
        outcome = _compare_points("cosine", self._raiser("本包拒绝"), lambda: [1.0])
        assert outcome.status == "rejected"
        assert outcome.compared_points == 0
        assert "本包拒绝" in outcome.message
        assert outcome.ok is True  # 记录差异不是分歧

    def test_both_sides_rejecting_agree(self) -> None:
        outcome = _compare_points(
            "cosine", self._raiser("本包拒绝"), self._raiser("生产拒绝")
        )
        assert outcome.status == "agrees"
        assert outcome.max_absolute_error == 0.0

    def test_different_lengths_differ(self) -> None:
        outcome = _compare_points("cosine", lambda: [1.0], lambda: [1.0, 2.0])
        assert outcome.status == "differs"
        assert math.isinf(outcome.max_absolute_error)

    def test_non_finite_results_differ(self) -> None:
        outcome = _compare_points("cosine", lambda: [float("nan")], lambda: [1.0])
        assert outcome.status == "differs"

    def test_close_values_agree(self) -> None:
        outcome = _compare_points("cosine", lambda: [1.0], lambda: [1.0 + 1e-15])
        assert outcome.status == "agrees"
        assert outcome.max_absolute_error < 1e-14


class TestNormalizeConventionIsGuarded:
    """零向量的约定差异必须**真的被检查到**，否则那条"记录"就是一句空话."""

    def test_changing_the_convention_turns_it_into_a_disagreement(self, monkeypatch) -> None:
        from smart_research_agent.math_foundations import bridge

        # 把本包的归一化改成"零向量原样返回"，此时两边不再有约定差异，
        # 而是**真正的分歧**（本包本该拒绝）——对照必须报 differs。
        monkeypatch.setattr(
            bridge, "normalize", lambda vector: tuple(float(value) for value in vector)
        )
        outcome = bridge.check_normalize()
        assert outcome.status == "differs"
        assert outcome.ok is False


class TestScalarFunctions:
    """三个标量函数的手算值与稳定性（sigmoid / log_sigmoid / softplus）."""

    def test_sigmoid_at_zero_and_large(self) -> None:
        assert approx(log_sigmoid(0.0), -math.log(2.0))
        # σ(0) 用 log_sigmoid 反推（避免重复实现）：exp(log σ(0)) = 0.5
        assert approx(math.exp(log_sigmoid(0.0)), 0.5)

    def test_log_sigmoid_stays_finite_far_left(self) -> None:
        assert log_sigmoid(-800.0) == -800.0
        assert math.isfinite(log_sigmoid(-1e6))

    def test_softplus_matches_definition(self) -> None:
        assert approx(softplus(0.0), math.log(2.0))
        # softplus(x) → x（x 很大时）与 → 0（x 很负时）
        assert approx(softplus(50.0), 50.0)
        assert softplus(-50.0) < 1e-20

    def test_log_sigmoid_is_minus_softplus_of_minus_x(self) -> None:
        for value in SCALAR_SAMPLES:
            assert approx(log_sigmoid(value), -softplus(-value))


class TestLogitsVersusProbabilityTrap:
    """这一课真踩到的一个坑：**同名不同物（logits 还是概率）**."""

    def test_passing_probabilities_as_logits_changes_the_loss(self) -> None:
        """把概率当 logits 传进去不会报错，只会得到一个**不同的** loss.

        ```text
        本包口径       cross_entropy(p, q) 吃**概率**：q = softmax(logits)
        生产口径       sft.loss.cross_entropy(logits, target) 吃**打分**
        ```

        把概率当 logits 的后果是**又做了一次 softmax**，把概率压向均匀：

        ```text
        target 是 argmax      该类的概率被压小 → loss **变大**（看起来像"模型变差了"）
        target 不是 argmax    该类的概率被推大 → loss **变小**（看起来像"模型变好了"）
        ```

        两个方向都出现了——所以"loss 不对劲"这件事**不能只看方向**，
        必须回到入参形态。这条断言就是那次排查留下的证据。
        """
        from smart_research_agent.math_foundations.linalg import softmax
        from smart_research_agent.sft.loss import cross_entropy as training_cross_entropy

        logits = (1.0, 2.0)
        as_probabilities = list(softmax(logits))
        # target = 1（argmax）：概率被压向均匀 → loss 变大
        assert training_cross_entropy(as_probabilities, 1) > training_cross_entropy(
            list(logits), 1
        )
        # target = 0（非 argmax）：概率被推大 → loss 变小
        assert training_cross_entropy(as_probabilities, 0) < training_cross_entropy(
            list(logits), 0
        )
        # 本包的通式（真值 one-hot、预测取 softmax）与生产口径一致
        from smart_research_agent.math_foundations.probability import cross_entropy

        assert approx(
            cross_entropy(one_hot(1, 2), softmax(logits)),
            training_cross_entropy(list(logits), 1),
        )

    def test_one_hot_validates_its_arguments(self) -> None:
        assert one_hot(1, 3) == (0.0, 1.0, 0.0)
        with pytest.raises(NumericError):
            one_hot(3, 3)
        with pytest.raises(NumericError):
            one_hot(0, 0)


class TestSamples:
    """演示与测试共用的注意力输入（写死、可复现）."""

    def test_sample_outputs_are_consistent(self) -> None:
        queries, keys, values = sample_outputs()
        assert len(queries) == 3 and len(keys) == 3 and len(values) == 3
        assert all(len(row) == 4 for row in queries)
        assert all(len(row) == 2 for row in values)
        assert sample_outputs() == sample_outputs()
