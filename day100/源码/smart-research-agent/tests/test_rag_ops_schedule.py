"""``rag_ops.schedule`` 与 ``rag_ops.worker``：什么时候跑、跑完怎么等（day072）.

调度是这一课唯一能"被推到任意时刻"的部分，因此它承担了三条性质的证明：

```text
① 抖动是算出来的     同样的（锚点，运行序号）永远得到同一个等待时长
② 迟到只补跑一次     错过三个计划点时，下一个计划点**不往后挪**
③ 失败要退避         连续失败 n 次的等待是指数增长且封顶的，而不是"再也不跑"
```

``run_forever`` 用注入的 ``sleep`` 与假时钟在 0 秒内推演一整天：
真实的 24 小时循环不可能进 CI，而它的逻辑恰恰最需要被测。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from smart_research_agent.rag_ops.errors import ScheduleError
from smart_research_agent.rag_ops.schedule import decide, describe, jitter_seconds
from smart_research_agent.rag_ops.types import (
    MODE_BACKOFF,
    MODE_DUE,
    MODE_FIRST_RUN,
    MODE_FORCED,
    MODE_NOT_DUE,
    OpsReport,
    parse_iso,
    to_iso,
)
from smart_research_agent.rag_ops.worker import (
    MAXIMUM_WAIT_SECONDS,
    MINIMUM_WAIT_SECONDS,
    build_parser,
    main,
    run_forever,
    seconds_until_next,
)
from tests.rag_ops_samples import FIXED_NOW, build_pipeline, ledger, policy, ran_ledger, snapshot


class TestJitter:
    """抖动：确定性、落在区间内、可关闭."""

    def test_is_deterministic(self) -> None:
        first = jitter_seconds(policy(), anchor=to_iso(FIXED_NOW), run_index=3)
        second = jitter_seconds(policy(), anchor=to_iso(FIXED_NOW), run_index=3)
        assert first == second

    def test_differs_across_runs_but_stays_in_range(self) -> None:
        values = {
            jitter_seconds(policy(), anchor=to_iso(FIXED_NOW), run_index=index)
            for index in range(12)
        }
        assert len(values) > 1
        assert all(0 <= value <= 300 for value in values)

    def test_zero_jitter_is_zero(self) -> None:
        assert jitter_seconds(policy(jitter_seconds=0), anchor="x", run_index=1) == 0


class TestDecide:
    """五种模式各自的判据与理由（每条理由都要能照着做）."""

    def test_first_run_is_due(self) -> None:
        result = decide(policy(), ledger(), now=FIXED_NOW)
        assert result.mode == MODE_FIRST_RUN and result.due is True
        assert "首次运行" in result.reason
        assert parse_iso(result.next_run_at) > FIXED_NOW

    def test_not_due_before_the_planned_moment(self) -> None:
        state = ran_ledger(snapshot(), moment=FIXED_NOW)
        result = decide(policy(), state, now=FIXED_NOW + timedelta(minutes=30))
        assert result.mode == MODE_NOT_DUE and result.due is False
        assert "未到计划时间" in result.reason and "还差" in result.reason
        assert result.missed_runs == 0
        # 未到点时"下一次"就是那个计划点本身（不重算、不漂移）
        assert result.next_run_at == result.effective_at

    def test_due_once_the_interval_passed(self) -> None:
        state = ran_ledger(snapshot(), moment=FIXED_NOW)
        result = decide(policy(), state, now=FIXED_NOW + timedelta(hours=25))
        assert result.mode == MODE_DUE and result.due is True
        assert result.lag_minutes > 0
        assert "正常触发" in result.reason

    def test_missed_windows_only_catch_up_once(self) -> None:
        state = ran_ledger(snapshot(), moment=FIXED_NOW)
        moment = FIXED_NOW + timedelta(days=3, hours=1)
        result = decide(policy(), state, now=moment)
        assert result.mode == MODE_DUE and result.missed_runs >= 2
        assert "只补跑这一次" in result.reason
        # 下一个计划点按**计划**算（锚点 + (错过数 + 2) 个间隔），
        # 而不是从"本次实际运行时间"起算——否则每次迟到都会把调度整体往后推
        expected = parse_iso(to_iso(FIXED_NOW)) + timedelta(
            minutes=1440 * (result.missed_runs + 2)
        )
        assert abs((parse_iso(result.next_run_at) - expected).total_seconds()) <= 300

    def test_excessive_lag_adds_a_reason_but_still_runs(self) -> None:
        state = ran_ledger(snapshot(), moment=FIXED_NOW)
        result = decide(policy(max_lag_minutes=60), state, now=FIXED_NOW + timedelta(hours=30))
        assert result.due is True
        assert "max_lag_minutes" in result.reason

    def test_backoff_window_blocks_the_run(self) -> None:
        state = ledger(
            snapshot(),
            last_run_at=to_iso(FIXED_NOW),
            last_success_at=to_iso(FIXED_NOW - timedelta(days=1)),
            consecutive_failures=3,
            runs=3,
            failures=3,
            last_error="编码器挂了",
        )
        result = decide(policy(), state, now=FIXED_NOW + timedelta(seconds=10))
        assert result.mode == MODE_BACKOFF and result.due is False
        assert "正在退避" in result.reason and "编码器挂了" in result.reason
        assert parse_iso(result.effective_at) == FIXED_NOW + timedelta(seconds=240)

    def test_backoff_expiry_triggers_a_retry(self) -> None:
        state = ledger(
            snapshot(),
            last_run_at=to_iso(FIXED_NOW),
            consecutive_failures=1,
            runs=1,
            failures=1,
            last_error="编码器挂了",
        )
        result = decide(policy(), state, now=FIXED_NOW + timedelta(seconds=90))
        assert result.mode == MODE_DUE and result.due is True
        assert "退避窗口" in result.reason

    def test_failure_without_any_anchor_is_rejected(self) -> None:
        broken = ledger(consecutive_failures=2, last_error="挂了")
        with pytest.raises(ScheduleError):
            decide(policy(), broken, now=FIXED_NOW)

    def test_force_bypasses_everything(self) -> None:
        state = ledger(
            snapshot(),
            last_run_at=to_iso(FIXED_NOW),
            consecutive_failures=5,
            runs=5,
            failures=5,
            last_error="挂了",
        )
        result = decide(policy(), state, now=FIXED_NOW, force=True)
        assert result.mode == MODE_FORCED and result.due is True
        assert "force=True" in result.reason

    def test_describe_is_side_effect_free(self) -> None:
        payload = describe(policy(), ledger(), now=FIXED_NOW)
        assert set(payload) == {
            "policy",
            "policy_summary",
            "decision",
            "decision_summary",
            "ledger",
        }
        assert payload["decision"]["mode"] == MODE_FIRST_RUN
        assert payload["ledger"]["watermark"] is None


class TestSecondsUntilNext:
    """等待时长：夹在 [minimum, maximum] 之间，且空值优先保守."""

    def test_future_plan_is_clamped_by_maximum(self) -> None:
        decision = decide(policy(), ran_ledger(snapshot(), moment=FIXED_NOW), now=FIXED_NOW)
        assert seconds_until_next(decision, now=FIXED_NOW) == MAXIMUM_WAIT_SECONDS

    def test_past_plan_falls_back_to_minimum(self) -> None:
        decision = decide(policy(), ran_ledger(snapshot(), moment=FIXED_NOW), now=FIXED_NOW)
        later = FIXED_NOW + timedelta(days=2)
        assert seconds_until_next(decision, now=later) == MINIMUM_WAIT_SECONDS

    def test_short_gap_is_kept_between_the_bounds(self) -> None:
        decision = decide(
            policy(interval_minutes=1, jitter_seconds=0),
            ran_ledger(snapshot(), moment=FIXED_NOW),
            now=FIXED_NOW,
        )
        # 计划点 = FIXED_NOW + 60 秒，从 +10 秒看还剩 50 秒
        assert seconds_until_next(decision, now=FIXED_NOW + timedelta(seconds=10)) == 50.0

    def test_missing_next_run_at_is_conservative(self) -> None:
        decision = decide(policy(), ledger(), now=FIXED_NOW)
        blanked = type(decision)(
            mode=decision.mode,
            due=decision.due,
            reason=decision.reason,
            now=decision.now,
            next_run_at="",
        )
        assert seconds_until_next(blanked, now=FIXED_NOW) == MINIMUM_WAIT_SECONDS

    def test_illegal_range_is_rejected(self) -> None:
        decision = decide(policy(), ledger(), now=FIXED_NOW)
        with pytest.raises(ScheduleError):
            seconds_until_next(decision, now=FIXED_NOW, minimum=-1)
        with pytest.raises(ScheduleError):
            seconds_until_next(decision, now=FIXED_NOW, minimum=10, maximum=5)


class TestRunForever:
    """循环：跑几次、睡多久、每一趟的结论都要交出去."""

    def test_zero_iterations_runs_nothing(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        assert run_forever(pipeline, max_iterations=0, sleep=lambda _: None) == 0
        assert pipeline.buffer.describe() == []

    def test_two_iterations_share_one_pipeline(self, tmp_path: Path) -> None:
        clock = _SteppingClock()
        pipeline = build_pipeline(tmp_path, clock=clock)
        waited: list[float] = []
        reports: list[OpsReport] = []

        iterations = run_forever(
            pipeline,
            sleep=waited.append,
            now=clock,
            max_iterations=2,
            on_tick=reports.append,
        )
        assert iterations == 2
        assert [report.status for report in reports] == ["synced", "not_due"]
        assert waited and waited[0] == MAXIMUM_WAIT_SECONDS
        # 第一趟真的写了库，第二趟一行都没写
        assert reports[0].index is not None and reports[0].index["written"] > 0
        assert reports[1].index is None

    def test_negative_iterations_are_rejected(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with pytest.raises(ScheduleError):
            run_forever(pipeline, max_iterations=-1)


class _SteppingClock:
    """每次取时间都往前走一小时的时钟（让"到点"在第二轮自然发生）."""

    def __init__(self) -> None:
        self.moment: datetime = FIXED_NOW

    def __call__(self) -> datetime:
        """取当前时刻（**第一轮之后每小时走一格**）."""
        moment = self.moment
        self.moment = self.moment + timedelta(hours=1)
        return moment


class TestWorkerMain:
    """命令行入口：只跑一次、可指定语料目录、失败也返回 0（失败已记账）."""

    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        assert args.loop is False and args.force is False
        assert args.interval is None and args.max_iterations is None

    def test_once_runs_and_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "rag.md").write_text("# 检索增强生成\n\n正文。\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)  # 让 settings 里的相对路径全部落在 tmp 下
        assert main(["--source-dir", str(corpus), "--interval", "1440"]) == 0
        assert (tmp_path / "data" / "ops" / "sync_ledger.json").exists()
        assert (tmp_path / "outputs" / "rag_ops" / "ops_report.json").exists()

    def test_once_on_a_missing_corpus_still_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["--source-dir", str(tmp_path / "nope")]) == 0
        # 失败已被记进账本（水位不动、原因留下），退出码仍是 0：
        # 容器不该因为一次可重试的失败被编排层反复重启。
        ledger_file = tmp_path / "data" / "ops" / "sync_ledger.json"
        assert ledger_file.exists()
        assert "nope" in ledger_file.read_text(encoding="utf-8")
