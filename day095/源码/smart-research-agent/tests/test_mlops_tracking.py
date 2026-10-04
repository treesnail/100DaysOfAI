"""day059 实验追踪测试（M5-D10）：确定性 run_id、可空指标、三种比较关系.

这个文件守的是三件"少了它会静默出错"的事：

1. **``run_id`` 是确定性的**：同参数 → 同 id（``reuse=True`` 时命中已有 run，
   否则得到 ``-2`` 后缀），改一个参数 → 全新 id。它破了会让"这个数字是哪次
   跑出来的"变成一句空话，因为每一次重跑都会产生一个内容重复的新 run；
2. **同一个 ``(指标名, step)`` 不能重复记录**：覆盖会让 loss 曲线变成
   "最后一次写入的曲线"，历史被静默改写；
3. **缺指标时 ``relation`` 是 ``None`` 而不是 ``worse``**：把"没测过"
   报成"更差"，是这类比较函数最危险的默认行为。

``mode`` 方向（``max`` / ``min``）的意义也被逐条钉住：同一个 delta
在两种方向下的结论**必须相反**——如果它们相同，说明方向参数根本没被用上。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_research_agent.mlops import (
    METRIC_MODE_MAX,
    METRIC_MODE_MIN,
    RELATION_BETTER,
    RELATION_EQUAL,
    RELATION_WORSE,
    RUNS_FILENAME,
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_RUNNING,
    ExperimentTracker,
    MLOpsError,
    Run,
    compare_runs,
    run_id_for,
)

PARAMS = {"lora_r": 8, "learning_rate": 1e-4, "dataset_fingerprint": "3f1b0c9d7e5a2468"}


# --------------------------------------------------------------------------- #
# run_id_for：确定性身份
# --------------------------------------------------------------------------- #


def test_run_id_is_deterministic_and_twelve_hex() -> None:
    """同参数永远同一个 id，长度固定 12 位十六进制。"""
    first = run_id_for("exp", PARAMS)
    assert first == run_id_for("exp", PARAMS)
    assert len(first) == 12
    assert all(char in "0123456789abcdef" for char in first)


def test_run_id_changes_with_name_or_any_param() -> None:
    """名字或任一参数变化都会换 id（键顺序不影响）。"""
    reference = run_id_for("exp", PARAMS)
    assert run_id_for("other", PARAMS) != reference
    assert run_id_for("exp", {**PARAMS, "lora_r": 16}) != reference
    assert run_id_for("exp", {"learning_rate": 1e-4, "dataset_fingerprint": PARAMS["dataset_fingerprint"], "lora_r": 8}) == reference


def test_run_id_ignores_metrics_by_design() -> None:
    """id 只覆盖 params：指标是**结果**，算进去会让"同一次实验的两条记录"分裂。"""
    assert run_id_for("exp", PARAMS) == run_id_for("exp", dict(PARAMS))


def test_run_id_rejects_blank_name() -> None:
    """空名会让所有实验撞在同一个 id 上。"""
    with pytest.raises(MLOpsError, match="实验名不能为空"):
        run_id_for("   ", PARAMS)


# --------------------------------------------------------------------------- #
# Run：指标、历史与派生量
# --------------------------------------------------------------------------- #


def test_run_rejects_unknown_status() -> None:
    """状态是封闭集合。"""
    with pytest.raises(MLOpsError, match="未知状态"):
        Run(run_id="x", name="exp", status="done")


def test_run_flat_metrics_takes_the_latest_numeric_step() -> None:
    """压平指标取**数值最大 step** 的值（字符串排序会把 "10" 排在 "9" 前面）。"""
    run = Run(
        run_id="x",
        name="exp",
        metrics={"train_loss": {"0": 0.62, "9": 0.41, "10": 0.3187}},
    )
    assert run.metric("train_loss") == pytest.approx(0.3187)
    assert run.history("train_loss") == [(0, 0.62), (9, 0.41), (10, 0.3187)]


def test_run_metric_returns_none_for_missing() -> None:
    """缺指标返回 ``None``（缺失与 0 是两件事）。"""
    run = Run(run_id="x", name="exp")
    assert run.metric("nope") is None
    assert run.flat_metrics() == {}


def test_run_flat_metrics_skips_empty_series() -> None:
    """空序列被跳过（否则 ``max()`` 会对空序列抛 ValueError）。"""
    run = Run(run_id="x", name="exp", metrics={"a": {}})
    assert run.flat_metrics() == {}


def test_run_duration_is_zero_while_running() -> None:
    """未结束时时长是 0.0，但状态字段同时存在，因此不会被误读成"瞬间跑完"。"""
    run = Run(run_id="x", name="exp")
    assert run.finished is False
    assert run.duration_seconds == 0.0


def test_run_duration_is_computed_from_iso_timestamps() -> None:
    """结束后时长由两个 UTC 时间戳算出。"""
    run = Run(
        run_id="x",
        name="exp",
        started_at="2026-09-16T12:00:00+00:00",
        ended_at="2026-09-16T12:02:30+00:00",
        status=STATUS_FINISHED,
    )
    assert run.finished is True
    assert run.duration_seconds == pytest.approx(150.0)


def test_run_duration_tolerates_broken_timestamps() -> None:
    """时间戳不可解析时返回 0.0（不抛异常：报告要能打印出来）。"""
    run = Run(
        run_id="x",
        name="exp",
        started_at="bad",
        ended_at="worse",
        status=STATUS_FINISHED,
    )
    assert run.duration_seconds == 0.0


def test_run_roundtrip_projection() -> None:
    """``to_dict`` / ``from_dict`` 往返（派生键不进构造）。"""
    run = Run(
        run_id="x",
        name="exp",
        params={"a": 1},
        metrics={"train_loss": {"0": 0.5}},
        artifacts={"adapter": "p"},
        tags={"commit": "abc"},
        status=STATUS_FINISHED,
    )
    payload = run.to_dict()
    assert payload["finished"] is True
    assert payload["flat_metrics"] == {"train_loss": 0.5}
    restored = Run.from_dict(payload)
    assert restored.run_id == run.run_id
    assert restored.params == {"a": 1}
    assert restored.metrics == {"train_loss": {"0": 0.5}}
    assert restored.status == STATUS_FINISHED


def test_run_summary_line_lists_headline_metrics() -> None:
    """摘要里印出指标名与值（否则一眼看不出这次跑成什么样）。"""
    run = Run(
        run_id="x",
        name="exp",
        metrics={"eval_pass_rate": {"-1": 0.72}, "train_loss": {"-1": 0.31}},
    )
    line = run.summary_line()
    assert "eval_pass_rate=0.72" in line and "train_loss=0.31" in line


# --------------------------------------------------------------------------- #
# compare_runs：方向与缺失
# --------------------------------------------------------------------------- #


def make_run(run_id: str, value: float | None) -> Run:
    """构造一条只带一个指标的 run（``None`` 表示没记录）."""
    metrics = {} if value is None else {"eval_pass_rate": {"-1": value}}
    return Run(run_id=run_id, name="exp", metrics=metrics, status=STATUS_FINISHED)


@pytest.mark.parametrize(
    "left, right, mode, expected",
    [
        (0.60, 0.72, METRIC_MODE_MAX, RELATION_BETTER),
        (0.60, 0.72, METRIC_MODE_MIN, RELATION_WORSE),
        (0.72, 0.60, METRIC_MODE_MAX, RELATION_WORSE),
        (0.72, 0.60, METRIC_MODE_MIN, RELATION_BETTER),
        (0.66, 0.66, METRIC_MODE_MAX, RELATION_EQUAL),
        (0.66, 0.66, METRIC_MODE_MIN, RELATION_EQUAL),
    ],
)
def test_compare_runs_direction_flips_the_verdict(
    left: float, right: float, mode: str, expected: str
) -> None:
    """同一个 delta 在两种方向下的结论**必须相反**——相同就说明方向没被用上."""
    payload = compare_runs(make_run("a", left), make_run("b", right), metric="eval_pass_rate", mode=mode)
    assert payload["relation"] == expected
    assert payload["delta"] == pytest.approx(right - left)


def test_compare_runs_reports_missing_as_incomparable() -> None:
    """任一臂缺指标 → ``comparable=False`` 且 ``relation=None``.

    把"没测过"报成 ``worse``，是这类比较函数最危险的默认行为。
    """
    payload = compare_runs(make_run("a", None), make_run("b", 0.7), metric="eval_pass_rate")
    assert payload["comparable"] is False
    assert payload["relation"] is None
    assert payload["delta"] is None


def test_compare_runs_rejects_unknown_mode() -> None:
    """未知方向报错，而不是退回某个缺省。"""
    with pytest.raises(MLOpsError, match="未知指标方向"):
        compare_runs(make_run("a", 0.6), make_run("b", 0.7), metric="eval_pass_rate", mode="highest")


# --------------------------------------------------------------------------- #
# ExperimentTracker：内存模式
# --------------------------------------------------------------------------- #


def test_in_memory_tracker_writes_nothing(tmp_path: Path) -> None:
    """``path=None`` 时纯内存：不创建任何文件。"""
    tracker = ExperimentTracker()
    tracker.start_run("exp", params=PARAMS)
    assert tracker.path is None
    assert tracker.directory is None
    assert list(tmp_path.iterdir()) == []


def test_start_run_reuse_returns_the_same_object() -> None:
    """``reuse=True`` 命中已有 run（CI 重跑同一个任务）。"""
    tracker = ExperimentTracker()
    first = tracker.start_run("exp", params=PARAMS)
    again = tracker.start_run("exp", params=PARAMS, reuse=True)
    assert again is first
    assert len(tracker) == 1


def test_start_run_without_reuse_appends_a_suffix() -> None:
    """``reuse=False`` 时得到 ``-2`` / ``-3``（同参数重复运行）。"""
    tracker = ExperimentTracker()
    first = tracker.start_run("exp", params=PARAMS)
    second = tracker.start_run("exp", params=PARAMS)
    third = tracker.start_run("exp", params=PARAMS)
    assert second.run_id == f"{first.run_id}-2"
    assert third.run_id == f"{first.run_id}-3"
    assert len(tracker) == 3


def test_start_run_rejects_dangling_parent() -> None:
    """悬空父实验会让消融实验的树断掉。"""
    tracker = ExperimentTracker()
    with pytest.raises(MLOpsError, match="父实验"):
        tracker.start_run("exp", params=PARAMS, parent_run_id="nope")


def test_children_and_lineage() -> None:
    """子实验与实验谱系（自身 → 父 → 祖父）。"""
    tracker = ExperimentTracker()
    root = tracker.start_run("root", params={"a": 1})
    child = tracker.start_run("child", params={"a": 2}, parent_run_id=root.run_id)
    grandchild = tracker.start_run("gc", params={"a": 3}, parent_run_id=child.run_id)
    assert [run.run_id for run in tracker.children(root.run_id)] == [child.run_id]
    assert [run.run_id for run in tracker.lineage(grandchild.run_id)] == [
        grandchild.run_id,
        child.run_id,
        root.run_id,
    ]
    with pytest.raises(MLOpsError, match="不存在"):
        tracker.children("nope")


def test_lineage_detects_cycle() -> None:
    """环检测：带环的谱系会让 ``while`` 变成死循环（挂住而不是报错）。"""
    tracker = ExperimentTracker()
    root = tracker.start_run("root", params={})
    child = tracker.start_run("child", params={"x": 1}, parent_run_id=root.run_id)
    root.parent_run_id = child.run_id
    with pytest.raises(MLOpsError, match="环"):
        tracker.lineage(child.run_id)


def test_log_metrics_rejects_duplicate_step() -> None:
    """同一个 ``(指标名, step)`` 重复记录必须报错（覆盖会静默改写历史）。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    tracker.log_metrics(run, {"train_loss": 0.5}, step=10)
    with pytest.raises(MLOpsError, match="已记录过"):
        tracker.log_metrics(run, {"train_loss": 0.4}, step=10)


def test_log_metrics_allows_the_same_name_at_other_steps() -> None:
    """换一个 step 就可以再记一次（支持逐步曲线）。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    tracker.log_metrics(run, {"train_loss": 0.62}, step=0)
    tracker.log_metrics(run, {"train_loss": 0.3187}, step=40)
    assert run.history("train_loss") == [(0, 0.62), (40, 0.3187)]


def test_log_metrics_without_step_uses_the_scalar_slot() -> None:
    """无 step 的标量指标落在 ``-1`` 槽位（**不是 0**：0 是一个真实训练步）。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    tracker.log_metrics(run, {"eval_pass_rate": 0.72})
    assert run.metrics["eval_pass_rate"] == {"-1": 0.72}


def test_log_artifact_and_tags_merge() -> None:
    """产物与标签是"当前值"（同名覆盖），与指标的历史语义不同。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    tracker.log_artifact(run, "adapter", "p1")
    tracker.log_artifact(run, "adapter", "p2")
    tracker.set_tags(run, {"stage": "train"})
    tracker.set_tags(run, {"commit": "abc"})
    assert run.artifacts == {"adapter": "p2"}
    assert run.tags == {"stage": "train", "commit": "abc"}
    with pytest.raises(MLOpsError, match="产物名不能为空"):
        tracker.log_artifact(run, "  ", "p")


def test_finish_is_idempotent_but_appends_notes() -> None:
    """重复结束不改状态，只追加备注（第一次的结束时间才是真的结束时间）。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    first = tracker.finish(run, status=STATUS_FINISHED, notes="第一次")
    ended_at = first.ended_at
    second = tracker.finish(run, status=STATUS_FINISHED, notes="第二次")
    assert second.ended_at == ended_at
    assert "第一次" in second.notes and "第二次" in second.notes


def test_finish_rejects_running_status() -> None:
    """``finish`` 不能把状态设回 ``running``。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    with pytest.raises(MLOpsError, match="finish 只能把状态置为"):
        tracker.finish(run, status=STATUS_RUNNING)


def test_fail_marks_status_and_records_error() -> None:
    """失败也是要留档的事实：错误写进 tags（失败那次 run 往往最需要看）。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    tracker.fail(run, error="CUDA out of memory")
    assert run.status == STATUS_FAILED
    assert run.tags["error"] == "CUDA out of memory"


def test_runs_filtering_by_status_and_tag() -> None:
    """按状态与标签过滤；未知状态报错。"""
    tracker = ExperimentTracker()
    ok = tracker.start_run("ok", params={"a": 1})
    bad = tracker.start_run("bad", params={"a": 2})
    tracker.set_tags(bad, {"gpu": "a100"})
    tracker.finish(ok)
    tracker.fail(bad)
    assert [run.run_id for run in tracker.runs(status=STATUS_FINISHED)] == [ok.run_id]
    assert [run.run_id for run in tracker.runs(tag=("gpu", "a100"))] == [bad.run_id]
    assert len(tracker.runs()) == 2
    with pytest.raises(MLOpsError, match="未知状态"):
        tracker.runs(status="done")


def test_best_only_considers_finished_runs() -> None:
    """``best`` 只看已完成的 run（半截指标不该被选成"最好"）。"""
    tracker = ExperimentTracker()
    running = tracker.start_run("running", params={"a": 1})
    tracker.log_metrics(running, {"eval_pass_rate": 0.99})
    done = tracker.start_run("done", params={"a": 2})
    tracker.log_metrics(done, {"eval_pass_rate": 0.65})
    tracker.finish(done)
    assert tracker.best("eval_pass_rate").run_id == done.run_id
    assert tracker.best("eval_pass_rate", mode=METRIC_MODE_MIN).run_id == done.run_id


def test_best_returns_none_without_candidates() -> None:
    """没有候选时返回 ``None``（"没有候选"与"候选都很差"是两件事）。"""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params=PARAMS)
    tracker.finish(run)
    assert tracker.best("eval_pass_rate") is None
    with pytest.raises(MLOpsError, match="未知指标方向"):
        tracker.best("eval_pass_rate", mode="highest")


def test_get_find_and_iteration() -> None:
    """``get`` 抛错、``find`` 返回 None；迭代按开启顺序。"""
    tracker = ExperimentTracker()
    first = tracker.start_run("a", params={"a": 1})
    tracker.start_run("b", params={"a": 2})
    assert tracker.get(first.run_id) is first
    assert tracker.find("nope") is None
    with pytest.raises(MLOpsError, match="不存在"):
        tracker.get("nope")
    assert [run.name for run in tracker] == ["a", "b"]


# --------------------------------------------------------------------------- #
# 落盘与重载
# --------------------------------------------------------------------------- #

def test_persistence_writes_one_line_per_touch(tmp_path: Path) -> None:
    """每次变更追加整条 run（读最后一行即得终态）。"""
    index = tmp_path / RUNS_FILENAME
    tracker = ExperimentTracker(index)
    run = tracker.start_run("exp", params=PARAMS)
    tracker.log_metrics(run, {"train_loss": 0.5}, step=10)
    tracker.finish(run)
    lines = index.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["status"] == STATUS_FINISHED
    assert json.loads(lines[0])["status"] == STATUS_RUNNING


def test_reload_rebuilds_final_state(tmp_path: Path) -> None:
    """重载后取到的是**终态**（同一 id 的多行折叠成最后一行）。"""
    index = tmp_path / RUNS_FILENAME
    tracker = ExperimentTracker(index)
    run = tracker.start_run("exp", params=PARAMS)
    tracker.log_metrics(run, {"train_loss": 0.5}, step=10)
    tracker.log_metrics(run, {"eval_pass_rate": 0.72})
    tracker.finish(run)

    reloaded = ExperimentTracker(index)
    restored = reloaded.get(run.run_id)
    assert restored.status == STATUS_FINISHED
    assert restored.metric("train_loss") == pytest.approx(0.5)
    assert restored.metric("eval_pass_rate") == pytest.approx(0.72)
    assert len(reloaded) == 1


def test_reload_is_idempotent(tmp_path: Path) -> None:
    """``reload`` 先清空再折叠，可以重复调用。"""
    index = tmp_path / RUNS_FILENAME
    tracker = ExperimentTracker(index)
    tracker.start_run("exp", params=PARAMS)
    tracker.reload()
    tracker.reload()
    assert len(tracker) == 1


def test_missing_file_is_an_empty_tracker(tmp_path: Path) -> None:
    """文件不存在 = 空追踪器。"""
    tracker = ExperimentTracker(tmp_path / "nope" / RUNS_FILENAME)
    assert len(tracker) == 0
    assert tracker.to_dict()["counts"]["total"] == 0


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """空行被跳过（追加写被打断时可能出现）。"""
    index = tmp_path / RUNS_FILENAME
    run = Run(run_id="x", name="exp")
    index.write_text(
        "\n" + json.dumps(run.to_dict(), ensure_ascii=False) + "\n\n", encoding="utf-8"
    )
    assert len(ExperimentTracker(index)) == 1


@pytest.mark.parametrize(
    "content, match",
    [
        ('{"run_id": "x", broken\n', "不是合法 JSON"),
        ('["not", "an", "object"]\n', "不是一条 run 记录"),
        ('{"name": "exp"}\n', "不是一条 run 记录"),
    ],
)
def test_corrupt_lines_raise(tmp_path: Path, content: str, match: str) -> None:
    """三类损坏各自报错（静默跳过会丢实验记录）。"""
    index = tmp_path / RUNS_FILENAME
    index.write_text(content, encoding="utf-8")
    with pytest.raises(MLOpsError, match=match):
        ExperimentTracker(index)


def test_to_dict_and_markdown(tmp_path: Path) -> None:
    """投影与 markdown 渲染（CI 摘要看的就是后者）。"""
    tracker = ExperimentTracker(tmp_path / RUNS_FILENAME)
    run = tracker.start_run("exp", params=PARAMS, tags={"commit": "abc"})
    tracker.log_metrics(run, {"eval_pass_rate": 0.72})
    tracker.finish(run)
    payload = tracker.to_dict()
    assert payload["counts"] == {
        STATUS_RUNNING: 0,
        STATUS_FINISHED: 1,
        STATUS_FAILED: 0,
        "total": 1,
    }
    assert payload["runs"][0]["run_id"] == run.run_id
    markdown = tracker.render_markdown()
    assert "实验追踪" in markdown
    assert run.run_id in markdown
    assert "lora_r=8" in markdown
