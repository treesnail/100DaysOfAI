"""day060 ``serving.verify`` 的单元测试：上线前的端到端验证.

延迟一律用**注入的确定性时钟**复现：真实计时在单元测试里不可复现，
而把它跳过等于让"延迟检查"这条分支永远测不到。
"""

from __future__ import annotations

import pytest

from smart_research_agent.serving.errors import ServingError
from smart_research_agent.serving.verify import (
    ARM_CLOUD,
    ARM_DEDICATED,
    ARMS,
    CHECK_CASES,
    CHECK_ERRORS,
    CHECK_LATENCY,
    CHECK_PASS_RATE,
    CHECK_REGRESSION,
    RATE_EPSILON,
    ProbeCase,
    ProbeOutcome,
    VerifyCheck,
    VerifyPolicy,
    judge,
    median_of,
    run_verification,
    verify_table,
)


class _SeqClock:
    """按"每个臂的每用例延迟"构造的确定性时钟（毫秒 → 秒）.

    ``run_verification`` 先跑完云端臂再跑专属臂，且每个用例调用两次
    ``clock()``（开始与结束），因此把两段延迟序列依次展开成
    ``[0, l1, l1, l1+l2, ...]`` 就能让报告里的延迟精确等于给定的毫秒数。
    """

    def __init__(self, arms: list[list[float]]) -> None:
        values = [0.0]
        running = 0.0
        for arm in arms:
            for milliseconds in arm:
                running += milliseconds / 1000.0
                values.extend([running, running])
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def _case(case_id: str = "c1", *, prompt: str = "什么是 DPO？", must=("DPO",)) -> ProbeCase:
    return ProbeCase(case_id=case_id, prompt=prompt, must_contain=must)


def _responder(replies: dict[str, str], *, error_cases: tuple[str, ...] = ()):
    def respond(case: ProbeCase) -> str:
        if case.case_id in error_cases:
            raise RuntimeError(f"部署不可用：{case.case_id}")
        return replies.get(case.case_id, "")

    return respond


# --------------------------------------------------------------------------- #
# 用例与判定
# --------------------------------------------------------------------------- #


def test_probe_case_validates_identity_fields() -> None:
    with pytest.raises(ServingError, match="用例必须有 case_id"):
        ProbeCase(case_id="", prompt="x")
    with pytest.raises(ServingError, match="prompt 不能为空"):
        ProbeCase(case_id="c", prompt="")


def test_probe_case_to_dict_serializes_must_contain() -> None:
    payload = _case(must=("a", "b")).to_dict()
    assert payload["must_contain"] == ["a", "b"]
    assert payload["case_id"] == "c1"


def test_probe_outcome_rejects_unknown_arms() -> None:
    with pytest.raises(ServingError, match="未知推理臂"):
        ProbeOutcome(case_id="c", arm="edge", reply="", passed=True, latency_ms=1.0)
    assert ARMS == (ARM_CLOUD, ARM_DEDICATED)


def test_probe_outcome_summary_reports_state() -> None:
    assert "错误" in ProbeOutcome(
        case_id="c", arm=ARM_CLOUD, reply="", passed=False, latency_ms=1.0, error="boom"
    ).summary_line()
    assert "通过" in ProbeOutcome(
        case_id="c", arm=ARM_CLOUD, reply="ok", passed=True, latency_ms=1.0
    ).summary_line()
    long_line = ProbeOutcome(
        case_id="c", arm=ARM_CLOUD, reply="x" * 60, passed=False, latency_ms=1.0
    ).summary_line()
    assert "…" in long_line


def test_probe_outcome_to_dict_is_json_ready() -> None:
    payload = ProbeOutcome(
        case_id="c1", arm=ARM_CLOUD, reply="ok", passed=True, latency_ms=1.5
    ).to_dict()
    assert payload == {
        "case_id": "c1",
        "arm": ARM_CLOUD,
        "reply": "ok",
        "passed": True,
        "latency_ms": 1.5,
        "error": "",
    }


def test_verify_check_summary_names_both_state_and_blocking() -> None:
    """一行摘要要同时给出"通过没通过"与"阻塞不阻塞"."""
    blocking = VerifyCheck(
        name=CHECK_REGRESSION, passed=False, actual=-0.5, threshold=">= -0.0", reason="退步"
    )
    assert "不通过/阻塞" in blocking.summary_line()
    assert "-0.5" in blocking.summary_line()
    warning = VerifyCheck(
        name=CHECK_LATENCY,
        passed=False,
        actual=None,
        threshold="未给出倍数",
        reason="未检查",
        blocking=False,
    )
    assert "不通过/告警" in warning.summary_line()
    ok = VerifyCheck(
        name=CHECK_CASES, passed=True, actual=2, threshold="> 0", reason="够用"
    )
    assert "通过/阻塞" in ok.summary_line()
    assert ok.to_dict()["actual"] == 2


def test_judge_requires_every_fragment_but_ignores_case_and_order() -> None:
    case = _case(must=("DPO", "beta"))
    assert judge(case, "dpo 的目标函数里有一个 beta 系数") is True
    assert judge(case, "beta 是温度，DPO 用它") is True
    assert judge(case, "只提到了 DPO") is False


def test_a_case_without_fragments_only_checks_connectivity() -> None:
    """片段为空的用例**永远通过**：它退化成一条"只测连通性"的探针."""
    assert judge(_case(must=()), "任何内容") is True


def test_median_of_handles_odd_even_and_empty() -> None:
    assert median_of([3.0, 1.0, 2.0]) == 2.0
    assert median_of([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median_of([]) == 0.0


def test_verify_policy_validates_its_numbers() -> None:
    with pytest.raises(ServingError, match="min_pass_rate 必须落在"):
        VerifyPolicy(min_pass_rate=1.5)
    with pytest.raises(ServingError, match="max_regression 不能为负数"):
        VerifyPolicy(max_regression=-0.1)
    with pytest.raises(ServingError, match="max_latency_ratio 必须为正数或 None"):
        VerifyPolicy(max_latency_ratio=0.0)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def test_zero_cases_is_a_failure_not_a_pass() -> None:
    """**没有证据本身就是不切换的理由**——零用例必须是"查不了"，不是"全过"."""
    report = run_verification(
        [], cloud_respond=lambda case: "x", dedicated_respond=lambda case: "x"
    )
    assert report.passed is False
    assert report.cases == 0
    assert [item.name for item in report.blocking_failures] == [CHECK_CASES]
    assert "没有证据本身就是不切换的理由" in report.check(CHECK_CASES).reason
    assert report.outcomes == ()
    assert report.latency_ratio is None


def test_identical_arms_pass_every_check() -> None:
    cases = [_case("c1"), _case("c2", must=("LoRA",))]
    replies = {"c1": "DPO 是直接偏好优化", "c2": "LoRA 是低秩适配"}
    report = run_verification(
        cases,
        cloud_respond=_responder(replies),
        dedicated_respond=_responder(replies),
        clock=_SeqClock([[10.0, 10.0], [10.0, 10.0]]),
    )
    assert report.passed is True
    assert report.cases == 2
    assert report.delta == 0.0
    assert report.cloud_latency_ms == 10.0
    assert report.dedicated_latency_ms == 10.0
    assert report.latency_ratio == 1.0
    assert report.blocking_failures == []
    assert len(report.checks) == 5
    assert report.summary_line().startswith("上线验证 通过")
    assert len(report.outcomes) == 4


def test_a_regression_on_a_single_case_is_blocking() -> None:
    cases = [_case("c1"), _case("c2", must=("LoRA",))]
    report = run_verification(
        cases,
        cloud_respond=_responder({"c1": "DPO", "c2": "LoRA"}),
        dedicated_respond=_responder({"c1": "DPO", "c2": "不知道"}),
    )
    assert report.passed is False
    assert [item.name for item in report.blocking_failures] == [CHECK_REGRESSION]
    assert report.delta == -0.5
    assert report.check(CHECK_REGRESSION).threshold == ">= -0.0"
    assert report.dedicated_pass_rate == 0.5
    assert [item.case_id for item in report.failing_cases()] == ["c2"]
    assert [item.case_id for item in report.failing_cases(ARM_CLOUD)] == []


def test_a_positive_delta_passes_the_regression_check() -> None:
    """方向回归断言：``delta = 专属 − 云端``，**改进是正值**，必须判通过.

    与 day059 的 ``test_positive_delta_passes_the_regression_gate`` 同一理由：
    写成 ``delta <= max_regression`` 会把所有改进都判成失败，
    而"所有候选都被拦下"看起来很像"门槛很严格"。
    """
    cases = [_case("c1"), _case("c2", must=("LoRA",))]
    report = run_verification(
        cases,
        cloud_respond=_responder({"c1": "DPO", "c2": "不知道"}),
        dedicated_respond=_responder({"c1": "DPO", "c2": "LoRA"}),
    )
    assert report.check(CHECK_REGRESSION).passed is True
    assert report.delta == 0.5
    assert report.check(CHECK_REGRESSION).threshold == ">= -0.0"


def test_a_regression_within_tolerance_is_allowed() -> None:
    cases = [_case(f"c{i}", must=("LoRA",)) for i in range(20)]
    cloud = {case.case_id: "LoRA" for case in cases}
    dedicated = dict(cloud)
    dedicated["c0"] = "不知道"  # 1/20 = 0.05 的退步
    report = run_verification(
        cases,
        cloud_respond=_responder(cloud),
        dedicated_respond=_responder(dedicated),
        policy=VerifyPolicy(max_regression=0.05),
    )
    assert report.check(CHECK_REGRESSION).passed is True
    assert report.delta == pytest.approx(-0.05)
    assert report.check(CHECK_REGRESSION).threshold == ">= -0.05"


def test_a_regression_exactly_at_the_tolerance_is_accepted() -> None:
    """**恰好等于容忍度必须判通过**，而浮点减法会让它偶尔落到阈值之下.

    实测：``19/20 - 20/20`` 的双精度值是 ``-0.050000000000000044``，
    它比 ``-0.05`` 小——没有容差的话，"退步 5 个百分点、容忍 5 个百分点"
    会被判成失败，而且结论**随用例数翻转**（40 条里差 2 条得到的是
    ``-0.049999999999999996``，判通过）。这条断言把容差钉住。
    """
    cases = [_case(f"c{i}", must=("LoRA",)) for i in range(20)]
    cloud = {case.case_id: "LoRA" for case in cases}
    dedicated = dict(cloud)
    dedicated["c0"] = "不知道"
    report = run_verification(
        cases,
        cloud_respond=_responder(cloud),
        dedicated_respond=_responder(dedicated),
        policy=VerifyPolicy(max_regression=0.05),
    )
    # 先确认这个差值确实"比阈值更小"（否则这条测试就测了个寂寞）
    assert report.delta < -0.05
    assert report.check(CHECK_REGRESSION).passed is True
    assert RATE_EPSILON < 0.0556  # 远小于 18 条评估集的最小分辨率


def test_just_over_the_tolerance_is_still_blocked() -> None:
    """容差只有 1e-9：**多退一条用例仍然必须被拦下**（容差不是放宽门槛）."""
    cases = [_case(f"c{i}", must=("LoRA",)) for i in range(20)]
    cloud = {case.case_id: "LoRA" for case in cases}
    dedicated = dict(cloud)
    dedicated["c0"] = "不知道"
    dedicated["c1"] = "不知道"  # 2/20 = 0.10 的退步
    report = run_verification(
        cases,
        cloud_respond=_responder(cloud),
        dedicated_respond=_responder(dedicated),
        policy=VerifyPolicy(max_regression=0.05),
    )
    assert report.check(CHECK_REGRESSION).passed is False
    assert CHECK_REGRESSION in [item.name for item in report.blocking_failures]


def test_calling_failures_are_counted_separately_and_blocking() -> None:
    """"调用失败"与"答得不对"是两类问题：前者是部署问题，后者是模型问题."""
    cases = [_case("c1"), _case("c2")]
    report = run_verification(
        cases,
        cloud_respond=_responder({"c1": "DPO", "c2": "LoRA"}),
        dedicated_respond=_responder({"c1": "DPO"}, error_cases=("c2",)),
    )
    assert report.dedicated_errors == 1
    assert report.cloud_errors == 0
    assert CHECK_ERRORS in [item.name for item in report.blocking_failures]
    outcome = next(
        item for item in report.outcomes if item.arm == ARM_DEDICATED and item.case_id == "c2"
    )
    assert "RuntimeError" in outcome.error
    assert outcome.reply == ""


def test_disabling_the_error_check_records_a_warning_instead() -> None:
    cases = [_case("c1")]
    report = run_verification(
        cases,
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}, error_cases=("c1",)),
        policy=VerifyPolicy(require_no_errors=False),
    )
    error_check = report.check(CHECK_ERRORS)
    assert error_check.passed is False
    assert error_check.blocking is False
    assert "有意的关闭，不是「通过」" in error_check.reason
    assert error_check.actual == 1


def test_latency_is_unchecked_by_default_and_lands_in_skipped() -> None:
    """缺省不检查延迟：**"多慢算慢"取决于部署形态**，全局缺省只会被无脑放宽."""
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
        clock=_SeqClock([[10.0], [50.0]]),
    )
    assert report.passed is True
    assert [item.name for item in report.skipped_checks] == [CHECK_LATENCY]
    latency = report.check(CHECK_LATENCY)
    assert latency.actual is None
    assert latency.blocking is False
    assert "本项记为「未检查」" in latency.reason


def test_latency_check_passes_within_the_requested_multiple() -> None:
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
        policy=VerifyPolicy(max_latency_ratio=3.0),
        clock=_SeqClock([[100.0], [250.0]]),
    )
    assert report.latency_ratio == 2.5
    assert report.check(CHECK_LATENCY).passed is True
    assert report.passed is True
    assert report.skipped_checks == []


def test_latency_check_blocks_when_the_multiple_is_exceeded() -> None:
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
        policy=VerifyPolicy(max_latency_ratio=2.0),
        clock=_SeqClock([[100.0], [250.0]]),
    )
    assert report.passed is False
    assert [item.name for item in report.blocking_failures] == [CHECK_LATENCY]
    assert "超过上限 2.0" in report.check(CHECK_LATENCY).reason


def test_zero_cloud_latency_cannot_be_used_to_compute_a_multiple() -> None:
    """云端中位数为 0 时返回 ``None`` 而不是一个大数：**假数字比空值更容易骗过人**."""
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
        policy=VerifyPolicy(max_latency_ratio=2.0),
        clock=_SeqClock([[0.0], [10.0]]),
    )
    assert report.latency_ratio is None
    assert report.passed is False
    assert "云端延迟中位数为 0" in report.check(CHECK_LATENCY).reason


def test_latency_uses_the_median_not_the_mean() -> None:
    """自建推理是长尾分布：中位数回答的是"典型用户感受到多慢"."""
    cases = [_case(f"c{i}") for i in range(5)]
    replies = {case.case_id: "DPO" for case in cases}
    report = run_verification(
        cases,
        cloud_respond=_responder(replies),
        dedicated_respond=_responder(replies),
        clock=_SeqClock([[1.0, 1.0, 1.0, 1.0, 1000.0], [1.0, 1.0, 1.0, 1.0, 1000.0]]),
    )
    assert report.cloud_latency_ms == 1.0
    assert report.dedicated_latency_ms == 1.0


def test_low_pass_rate_blocks_even_without_a_regression() -> None:
    """两条臂都差时，合格率自身也要拦住切换（绝对判定那一侧）."""
    cases = [_case("c1"), _case("c2")]
    report = run_verification(
        cases,
        cloud_respond=_responder({"c1": "nope", "c2": "nope"}),
        dedicated_respond=_responder({"c1": "nope", "c2": "nope"}),
    )
    assert report.delta == 0.0
    assert CHECK_PASS_RATE in [item.name for item in report.blocking_failures]
    assert report.check(CHECK_PASS_RATE).actual == 0.0


def test_report_to_dict_carries_every_counter() -> None:
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
        notes="灰度前验证",
        clock=_SeqClock([[5.0], [15.0]]),
    )
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["cases"] == 1
    assert payload["cloud_pass_rate"] == 1.0
    assert payload["dedicated_pass_rate"] == 1.0
    assert payload["delta"] == 0.0
    assert payload["latency_ratio"] == 3.0
    assert payload["notes"] == "灰度前验证"
    assert payload["skipped"] == [CHECK_LATENCY]
    assert payload["policy"]["min_pass_rate"] == 0.5
    assert len(payload["checks"]) == 5


def test_report_markdown_lists_checks_and_per_case_results() -> None:
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
    )
    text = report.render_markdown()
    assert "# 专属模型上线验证：通过" in text
    assert "| 检查 | 实际 | 阈值 | 通过 | 阻塞 | 理由 |" in text
    assert "## 逐条结果" in text
    assert "`cloud`" in text and "`dedicated`" in text


def test_check_lookup_reports_unknown_names() -> None:
    report = run_verification(
        [_case("c1")],
        cloud_respond=_responder({"c1": "DPO"}),
        dedicated_respond=_responder({"c1": "DPO"}),
    )
    with pytest.raises(ServingError, match="报告里没有检查项"):
        report.check("no_such_check")


def test_verify_table_documents_when_missing_behaviour() -> None:
    rows = verify_table()
    assert [row["name"] for row in rows] == [
        CHECK_CASES,
        CHECK_PASS_RATE,
        CHECK_REGRESSION,
        CHECK_ERRORS,
        CHECK_LATENCY,
    ]
    assert rows[4]["blocking"] is False
    assert rows[4]["threshold"] == "未给出倍数"
    assert "跳过（降级为告警" in rows[4]["when_missing"]
    assert rows[1]["when_missing"] == "不通过（缺证据不能切换）"
    assert verify_table(VerifyPolicy(max_latency_ratio=2.0))[4]["blocking"] is True
