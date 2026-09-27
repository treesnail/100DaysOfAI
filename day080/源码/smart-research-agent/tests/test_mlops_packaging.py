"""day059 打包与模型卡测试（M5-D10）：产物自描述与"不能做什么".

这个文件守的是模型卡存在的**主要原因**：第四段"已知限制"。

前三段（身份、指标、门禁）在 day058 的注册表里已经能回答，而"它不能做什么"
从来没有人写下来——于是每一次"模型答错了"的讨论都要从头争一遍
"这算不算预期内"。因此本文件的核心断言是：**四条限制各自带着一个可核对的数字**。

另外两条：

1. **``gate_`` 前缀的指标被过滤掉**：它们是门禁自己的结论，卡片另有专节；
   把同一份事实在同一份文档里写两遍且措辞不同，会让评审产生不信任；
2. **三元组三项缺一即拒绝构图**：缺了任何一项，卡片回答不了"凭什么可信"。
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.mlops import (
    GATE_METRIC_PREFIX,
    INTENDED_USE,
    LIMITATIONS,
    MODEL_CARD_FILENAME,
    OUT_OF_SCOPE,
    RELEASE_MANIFEST_FILENAME,
    ExperimentTracker,
    MLOpsError,
    ModelCard,
    ReleaseGates,
    build_model_card,
    build_release_manifest,
    evaluate_gates,
)

ADAPTER = "5c6d7e8f9012345678" + "b" * 46
DATA = "aa77c31e90f4b258"
METRICS = {
    "eval_pass_rate": 0.7222,
    "eval_pass_rate_delta": 0.1222,
    "adapter_mebibytes": 0.05,
    "dataset_fingerprint": DATA,
    "base_model": "Qwen3-8B",
}
ARTIFACTS = {"adapter": "a/adapter", "merged": "m/merged"}


def make_run(*, with_gate_flags: bool = False):  # type: ignore[no-untyped-def]
    """构造一条已完成的 run（可选带上 ``gate_`` 前缀的门禁结论标记）."""
    tracker = ExperimentTracker()
    run = tracker.start_run("exp", params={"lora_r": 8})
    tracker.log_metrics(run, {"train_loss": 0.3187}, step=42)
    tracker.log_metrics(run, {"eval_pass_rate": 0.7222})
    if with_gate_flags:
        tracker.log_metrics(run, {f"{GATE_METRIC_PREFIX}pass_rate": 1.0})
    tracker.finish(run)
    return run


def make_card(**overrides):  # type: ignore[no-untyped-def]
    """构造一份模型卡（缺省全部字段齐备）."""
    payload = {
        "version": "1.1.0",
        "base_model": "Qwen3-8B",
        "adapter_sha256": ADAPTER,
        "dataset_fingerprint": DATA,
    }
    payload.update(overrides)
    return build_model_card(**payload)


# --------------------------------------------------------------------------- #
# 常量与"不能做什么"
# --------------------------------------------------------------------------- #


def test_filenames_are_frozen() -> None:
    """两个产物名固定（脚本照着它们找）。"""
    assert MODEL_CARD_FILENAME == "MODEL_CARD.md"
    assert RELEASE_MANIFEST_FILENAME == "release_manifest.json"


def test_limitations_each_carry_a_checkable_number_or_source() -> None:
    """四条限制里必须出现"可核对的具体量"，否则它们只是免责声明.

    这条断言是刻意的：把限制写成"可能在某些情况下表现不佳"是没有用的，
    因为它**无法被用来否掉任何一个结论**。
    """
    joined = " ".join(LIMITATIONS)
    assert len(LIMITATIONS) == 4
    assert "1/18" in joined  # 评估集分辨率
    assert "557" in joined  # 参考模型规模
    assert "单一来源" in joined  # 数据画像
    assert "事实正确性" in joined  # 质量分的边界


def test_intended_use_and_out_of_scope_are_non_empty_and_distinct() -> None:
    """适用与不适用都要有内容，且不重叠（重叠意味着有一条写错了位置）。"""
    assert len(INTENDED_USE) >= 2
    assert len(OUT_OF_SCOPE) >= 3
    assert not set(INTENDED_USE) & set(OUT_OF_SCOPE)


def test_out_of_scope_is_specific_not_a_generic_disclaimer() -> None:
    """"明确不适用"要具体（"不适用于生产"这种话对判断没有帮助）。"""
    assert all(len(item) > 10 for item in OUT_OF_SCOPE)
    assert any("事实准确性" in item for item in OUT_OF_SCOPE)


# --------------------------------------------------------------------------- #
# ModelCard：构造期护栏
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["base_model", "adapter_sha256", "dataset_fingerprint"])
def test_card_requires_the_full_triple(field: str) -> None:
    """三元组缺一项即拒绝构图：卡片回答不了"凭什么可信"。"""
    with pytest.raises(MLOpsError, match=field):
        make_card(**{field: "   "})


def test_card_short_adapter_and_projection() -> None:
    """``short_adapter`` 取前 12 位；投影把三个元组字段转成列表。"""
    card = make_card()
    assert card.short_adapter == ADAPTER[:12]
    payload = card.to_dict()
    assert payload["short_adapter"] == ADAPTER[:12]
    assert isinstance(payload["limitations"], list)
    assert isinstance(payload["intended_use"], list)
    assert isinstance(payload["out_of_scope"], list)


# --------------------------------------------------------------------------- #
# build_model_card：指标来源与 gate_ 过滤
# --------------------------------------------------------------------------- #


def test_card_pulls_metrics_from_the_run() -> None:
    """指标取 ``run.flat_metrics()``（每个指标的最后一步值）。"""
    card = make_card(run=make_run())
    assert card.metrics == {"train_loss": 0.3187, "eval_pass_rate": 0.7222}


def test_card_filters_gate_prefixed_metrics() -> None:
    """``gate_`` 前缀的指标被过滤——卡片有专门的"发布门禁"一节，不该抄第二遍."""
    card = make_card(run=make_run(with_gate_flags=True))
    assert not any(name.startswith(GATE_METRIC_PREFIX) for name in card.metrics)
    assert "eval_pass_rate" in card.metrics


def test_card_without_run_has_no_metrics() -> None:
    """没有 run 就没有指标（而不是伪造一组 0）。"""
    assert make_card().metrics == {}
    assert "没有记录任何指标" in make_card().render_markdown()


def test_card_records_gate_verdict() -> None:
    """门禁结论进卡片（通过/不通过 + 三类名单）。"""
    passing = evaluate_gates(METRICS, policy=ReleaseGates(), artifacts=ARTIFACTS)
    card = make_card(gate=passing, run=make_run())
    assert card.gate_passed is True
    assert card.gate_summary["blocking_failures"] == []
    assert "发布门禁：通过" in card.render_markdown()

    failing = evaluate_gates(
        {**METRICS, "eval_pass_rate": 0.2}, policy=ReleaseGates(), artifacts=ARTIFACTS
    )
    blocked = make_card(gate=failing)
    assert blocked.gate_passed is False
    assert blocked.gate_summary["blocking_failures"] == ["pass_rate"]


def test_card_without_gate_says_not_passed() -> None:
    """缺门禁报告时卡片明确写"不通过"（不留空白；空白会被当成"还没来得及填"）."""
    card = make_card()
    assert card.gate_passed is False
    assert "发布门禁：不通过" in card.render_markdown()


def test_card_records_artifacts_and_commit() -> None:
    """产物路径与提交号进卡片（追溯的两条线索）。"""
    card = make_card(artifacts=ARTIFACTS, commit="deadbeef", parent_version="1.0.0")
    markdown = card.render_markdown()
    assert "deadbeef" in markdown
    assert "a/adapter" in markdown
    assert "父版本 1.0.0" in markdown
    assert "（首版）" in make_card().render_markdown()


def test_card_renders_all_four_sections() -> None:
    """卡片四段齐全：身份、指标、门禁、产物 + 三段用途/限制。"""
    markdown = make_card(run=make_run(), gate=evaluate_gates(METRICS, artifacts=ARTIFACTS)).render_markdown()
    for heading in ("## 身份", "## 指标", "## 发布门禁", "## 产物", "## 适用场景", "## 明确不适用", "## 已知限制"):
        assert heading in markdown


def test_card_without_artifacts_says_so() -> None:
    """没有产物时明确写出，而不是留一个空列表。"""
    assert "没有登记任何产物" in make_card().render_markdown()


def test_card_notes_are_rendered_when_present() -> None:
    """备注非空时才出现"备注"一节。"""
    assert "## 备注" not in make_card().render_markdown()
    assert "轮次 42" in make_card(notes="轮次 42").render_markdown()


# --------------------------------------------------------------------------- #
# build_release_manifest
# --------------------------------------------------------------------------- #


def test_manifest_shape_and_values() -> None:
    """清单是给机器读的那一份：字段平铺、可直接取用。"""
    run = make_run()
    gate = evaluate_gates(METRICS, policy=ReleaseGates(), artifacts=ARTIFACTS, commit="deadbeef")
    card = make_card(run=run, gate=gate, artifacts=ARTIFACTS, commit="deadbeef")
    manifest = build_release_manifest(card=card, run=run, gate=gate, extra={"bump_kind": "minor"})
    assert manifest["version"] == "1.1.0"
    assert manifest["adapter_sha256"] == ADAPTER
    assert manifest["dataset_fingerprint"] == DATA
    assert manifest["gate_passed"] is True
    assert manifest["run_status"] == "finished"
    assert manifest["run_duration_seconds"] is not None
    assert manifest["extra"] == {"bump_kind": "minor"}
    assert json.loads(json.dumps(manifest, ensure_ascii=False))["version"] == "1.1.0"


def test_manifest_without_run_and_gate() -> None:
    """缺 run / gate 时对应字段是 ``None`` 或空字典（不是伪造的默认值）。"""
    manifest = build_release_manifest(card=make_card())
    assert manifest["run_id"] == ""
    assert manifest["run_status"] is None
    assert manifest["gate"] == {}
    assert manifest["gate_passed"] is False
    assert manifest["limitations_count"] == len(LIMITATIONS)
    assert manifest["out_of_scope_count"] == len(OUT_OF_SCOPE)


def test_manifest_and_card_are_two_renderings_of_the_same_facts() -> None:
    """两份产物内容重叠但用途不同：任一方的版本/哈希/指纹必须与另一方一致."""
    run = make_run()
    gate = evaluate_gates(METRICS, policy=ReleaseGates(), artifacts=ARTIFACTS)
    card = make_card(run=run, gate=gate, artifacts=ARTIFACTS)
    manifest = build_release_manifest(card=card, run=run, gate=gate)
    assert manifest["version"] == card.version
    assert manifest["adapter_sha256"] == card.adapter_sha256
    assert manifest["dataset_fingerprint"] == card.dataset_fingerprint
    assert manifest["metrics"] == card.metrics
    assert manifest["gate_passed"] == card.gate_passed


def test_model_card_class_is_constructible_directly() -> None:
    """直接构造 ``ModelCard`` 也走同一套校验（build 只是语法糖）。"""
    card = ModelCard(
        title="t",
        version="1.0.0",
        base_model="Qwen3-8B",
        adapter_sha256=ADAPTER,
        dataset_fingerprint=DATA,
        run_id="abc",
    )
    assert card.gate_passed is False
    assert card.intended_use == INTENDED_USE
    assert card.limitations == LIMITATIONS
