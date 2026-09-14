"""评估报告 + 评估流水线 + 两个评估端点测试（day053）.

三块被测对象的性格完全不同，测试的关注点也不同：

- ``report.py`` 是**留档**：它把分数绑回"评的是哪一份适配器、哪一份评估集"
  （``adapter_sha256`` / ``suite_fingerprint``），再用三条门禁
  （``min_pass_rate`` / ``no_regression`` / ``execution``）给一次运行下结论。
  三条门禁**各自的失败路径都必须被构造出来**——一个永远不会失败的门禁
  不是门禁。绑定核对（``verify_manifest_binding``）的三个键各造一次不一致，
  每次都要抛 ``FinetuneEvalError``。门禁的两个视图也必须同口径：
  ``passed`` 看**全部** ``gates``，所以 ``failed_gates()`` 连名称未知的失败
  门禁也要列出来，且 ``gates={}`` 时摘要写"未设置门禁"而不是一对空括号。
  落盘侧的边界（顶层是数组 / 标量）同样由 ``FinetuneEvalError`` 收口，
  不把 ``AttributeError`` 漏到路由层。
- ``pipeline.py`` 是**顺序**：切分 → 两臂 → 配对比较 → 探针 → 报告。
  本文件钉住的是"两臂跑在同一份评估集上""探针标签成对""报告在所有数字
  算完之后才组装"这三件事的**可观测后果**。
- 两个端点走 ``TestClient``（进程内 ASGI 调用，不起真实服务）。
  ``probe=false`` 时端点秒回；``probe=true`` 时会在参考模型的 bigram
  适配器上**真的训练一次 LoRA**（约 2~3 秒），所以那一档只写一条用例。

实测基线（缺省参数，写死在各用例的 docstring 里）：评估集 18 条切分为
训练 12 / 评估 6，两臂合格率 ``1/3 → 5/6``，泄漏体检 ``passed=True``
（最大 Jaccard 0.054348 < 阈值 0.5），难度体检 ``audit == []``，
报告在 ``min_pass_rate=0.5`` 下 ``passed=True``。
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.finetune_eval import (
    COMPONENT_NAMES,
    GATE_NAMES,
    MAX_FAILURES,
    REPORT_FILE,
    ComparisonReport,
    EvalReport,
    EvalRun,
    FinetuneEvalError,
    FineTuneEvalOutcome,
    ItemOutcome,
    MetricBreakdown,
    build_report,
    build_suite,
    compare_runs,
    encode_probe_batches,
    estimate_eval_seconds,
    read_report,
    render_markdown,
    run_finetune_evaluation,
    split_suite,
    verify_manifest_binding,
    write_report,
)
from smart_research_agent.peft import default_reference_lora_config, train_lora_reference
from smart_research_agent.peft.deploy import AdapterManifest
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.reference_model import ReferenceSFTModel

#: 缺省评估集的条数与切分结果（实测值，规格与本文件多处断言依赖它）.
SUITE_TOTAL = 18
TRAIN_SIZE = 12
EVAL_SIZE = 6

#: 缺省两臂的合格率（实测：脚本化基线的引用尾注被截断，微调臂只在难例上掉尾部）.
BASELINE_PASS_RATE = 1 / 3
FINETUNED_PASS_RATE = 5 / 6

#: 构造报告时用的绑定清单内容哈希（64 位十六进制字符串）.
MANIFEST_SHA256 = "a" * 64

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_outcome(
    item_id: str,
    *,
    bucket: str = "citation",
    difficulty: str = "normal",
    passed: bool = True,
    score: float | None = None,
    error: str = "",
) -> ItemOutcome:
    """构造一条 ``ItemOutcome``（六个分量取同一个值，总分单独给）."""
    total = (0.9 if passed else 0.2) if score is None else score
    return ItemOutcome(
        item_id=item_id,
        bucket=bucket,
        difficulty=difficulty,
        output="答案" if passed else "",
        metric=MetricBreakdown(
            components={name: total for name in COMPONENT_NAMES}, total=total
        ),
        passed=passed,
        error=error,
        latency_seconds=0.01,
    )


def make_run(name: str, outcomes: list[ItemOutcome]) -> EvalRun:
    """把一组 ``ItemOutcome`` 打包成 ``EvalRun``（延迟固定 10 毫秒/条）."""
    return EvalRun(name=name, outcomes=list(outcomes), wall_seconds=0.05)


def make_sample_run() -> EvalRun:
    """三条用例、两条合格的样本次运行（被报告的多数用例复用）."""
    return make_run(
        "lora-finetuned",
        [
            make_outcome("cite-01", bucket="citation", difficulty="normal"),
            make_outcome("ref-01", bucket="refusal", difficulty="easy", passed=False),
            make_outcome("tool-01", bucket="tool_use", difficulty="easy"),
        ],
    )


def make_manifest(**overrides: object) -> AdapterManifest:
    """构造一份适配器清单（``overrides`` 用来逐个字段造不一致）."""
    payload: dict[str, object] = {
        "adapter_dir": "outputs/finetune/adapter-final",
        "base_model": "Qwen/Qwen3-0.6B",
        "lora": {"r": 8, "lora_alpha": 16, "resolved_targets": ["weight"]},
        "trainable_parameters": 8912,
        "adapter_bytes": 192645,
        "content_sha256": MANIFEST_SHA256,
        "step": 4,
        "train_loss": 6.3147,
        "learning_rate": 2.0,
    }
    payload.update(overrides)
    return AdapterManifest(**payload)  # type: ignore[arg-type]


def make_regressing_comparison() -> ComparisonReport:
    """构造一份"总体不变、refusal 桶掉光"的比对结果（``no_regression`` 的失败路径）.

    两臂各 2 条、总体合格率都是 50%，但 refusal 从 100% 掉到 0%：
    只看总体差值会以为"什么都没发生"，按桶看则是一次明确回退。
    """
    before = make_run(
        "before",
        [
            make_outcome("ref-01", bucket="refusal", difficulty="easy"),
            make_outcome("cite-01", bucket="citation", difficulty="normal", passed=False),
        ],
    )
    after = make_run(
        "after",
        [
            make_outcome("ref-01", bucket="refusal", difficulty="easy", passed=False),
            make_outcome("cite-01", bucket="citation", difficulty="normal"),
        ],
    )
    return compare_runs(before, after, samples=100)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """离线客户端：走 create_app 默认装配，进程内调用 ASGI 应用."""
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def default_outcome() -> FineTuneEvalOutcome:
    """缺省参数下的完整流水线产物（约 0.05 秒，模块内共用一次）."""
    return run_finetune_evaluation()


@pytest.fixture(scope="module")
def probe_triple() -> tuple[ReferenceSFTModel, object, CharTokenizer]:
    """``(微调前模型, 微调后模型, tokenizer)``：照抄 day053 正文的探针接线方式.

    ``epochs=2``（正文缺省 10）只是让用例快一点——探针要验的是
    "训练前后同口径可比"，不是"训到收敛"。
    """
    suite = build_suite()
    tokenizer = CharTokenizer.from_texts([item.instruction + item.reference for item in suite])
    base = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
    train_items, _ = split_suite(suite)
    model, _, _ = train_lora_reference(
        base,
        encode_probe_batches(train_items, tokenizer),
        config=default_reference_lora_config(),
        learning_rate=2.0,
        epochs=2,
        seed=42,
    )
    return base, model, tokenizer


class TestBuildReportGates:
    """``build_report``：三项门禁的组装，以及**三条失败路径**. """

    @pytest.mark.parametrize("min_pass_rate", [-0.1, 1.1, 2.0, -1.0])
    def test_min_pass_rate_out_of_range_raises(self, min_pass_rate):
        """``min_pass_rate`` 必须落在 ``[0, 1]``（越界是配置错误，不是"更严格"）."""
        with pytest.raises(FinetuneEvalError):
            build_report(run=make_sample_run(), min_pass_rate=min_pass_rate)

    def test_gate_keys_follow_gate_names(self):
        """``gates`` 恰好三个键，且**顺序**就是 ``GATE_NAMES``（报告的表格顺序）."""
        report = build_report(run=make_sample_run())
        assert tuple(report.gates) == GATE_NAMES
        assert len(report.gates) == 3
        assert GATE_NAMES == ("min_pass_rate", "no_regression", "execution")

    def test_min_pass_rate_failure_blocks_the_report(self):
        """``min_pass_rate`` 不达标 → 该项 ``passed=False`` 且整份报告 ``passed=False``.

        实测：样本次运行合格率 66.67%，阈值提到 100% 后只有这一项失败。
        """
        report = build_report(run=make_sample_run(), min_pass_rate=1.0)
        gate = report.gates["min_pass_rate"]
        assert gate["threshold"] == 1.0
        assert gate["value"] == pytest.approx(2 / 3)
        assert gate["passed"] is False
        assert report.passed is False
        assert report.failed_gates() == ["min_pass_rate"]

    def test_min_pass_rate_boundary_equality_passes(self):
        """阈值恰好等于合格率时通过（判据是 ``>=``，不是 ``>``）."""
        report = build_report(run=make_sample_run(), min_pass_rate=2 / 3)
        assert report.gates["min_pass_rate"]["passed"] is True
        assert report.passed is True

    def test_execution_gate_fails_on_error_item(self):
        """运行时含 ``error`` 的条目 → ``execution`` 不通过（异常条目计入分母）.

        实测：1 条异常、总分 0.0、``passed=False``，而被标记为 error 的条目
        **仍然计入 total**——跳过会让最差的那条用例凭空消失。
        """
        run = make_run("with-error", [make_outcome("a", passed=False, error="RuntimeError: boom")])
        report = build_report(run=run)
        gate = report.gates["execution"]
        assert run.error_count == 1
        assert run.total == 1
        assert gate["error_count"] == 1
        assert gate["value"] == 1
        assert gate["passed"] is False
        assert report.passed is False
        assert report.failed_gates() == ["execution"]

    def test_no_regression_gate_fails_on_regressed_comparison(self):
        """传入一个回退了的 ``comparison`` → ``no_regression`` 不通过."""
        comparison = make_regressing_comparison()
        report = build_report(run=make_sample_run(), comparison=comparison)
        gate = report.gates["no_regression"]
        assert comparison.passed is False
        assert gate["checked"] is True
        assert gate["passed"] is False
        assert gate["regressions"] == ["refusal"]
        assert gate["max_regression"] == 0.0
        assert report.passed is False
        assert "no_regression" in report.failed_gates()

    def test_no_comparison_marks_not_checked(self):
        """没传 ``comparison`` 时 ``no_regression.checked=False`` 且通过（如实标注）."""
        report = build_report(run=make_sample_run())
        gate = report.gates["no_regression"]
        assert gate["checked"] is False
        assert gate["passed"] is True
        assert gate["regressions"] == []
        assert report.passed is True
        assert "未做基线比对" in render_markdown(report)

    def test_suite_and_notes_are_copied_not_aliased(self):
        """``suite`` / ``notes`` 是**副本**：调用方之后改原对象不该动到报告."""
        suite = {"total": SUITE_TOTAL, "fingerprint": "abc123"}
        notes = ["脚本化对照臂"]
        report = build_report(run=make_sample_run(), suite=suite, notes=notes)
        suite["total"] = 999
        notes.append("后加的说明")
        assert report.suite["total"] == SUITE_TOTAL
        assert report.notes == ["脚本化对照臂"]

    def test_reference_fields_describe_the_run(self):
        """``reference`` 里带着"下一次接真模型时需要的接线信息"（不含 runner）."""
        report = build_report(run=make_sample_run())
        assert report.reference == {
            "run_name": "lora-finetuned",
            "items": 3,
            "error_count": 0,
            "mean_latency_seconds": pytest.approx(0.01),
        }
        assert "runner" not in report.reference

    def test_summary_is_the_run_projection(self):
        """``summary`` 就是 ``run.to_dict()``（含逐条明细与两层聚合）."""
        run = make_sample_run()
        report = build_report(run=run)
        assert report.summary == run.to_dict()
        assert report.pass_rate == pytest.approx(2 / 3)
        assert len(report.summary["outcomes"]) == 3

    def test_manifest_is_none_when_absent(self):
        """没传 manifest 时报告里是 ``None``（而不是空字典——两者含义不同）."""
        report = build_report(run=make_sample_run())
        assert report.manifest is None
        assert report.adapter_sha256 is None


class TestEvalReportProjection:
    """``EvalReport`` 的绑定字段、门禁视图与两个方向的投影."""

    def test_unbound_report_properties_are_none(self):
        """无 manifest 时 ``adapter_sha256`` / ``adapter_step`` / ``base_model`` 都是 None."""
        report = EvalReport(run_name="lora-finetuned", suite={"total": 3})
        assert report.adapter_sha256 is None
        assert report.adapter_step is None
        assert report.base_model is None

    def test_bound_report_exposes_manifest_fields(self):
        """有 manifest 时三项都能取到（且来自清单本身）."""
        report = build_report(run=make_sample_run(), manifest=make_manifest())
        assert report.adapter_sha256 == MANIFEST_SHA256
        assert report.adapter_step == 4
        assert report.base_model == "Qwen/Qwen3-0.6B"
        assert report.manifest is not None
        assert report.manifest["short_hash"] == MANIFEST_SHA256[:12]

    def test_suite_fingerprint_comes_from_suite(self):
        """``suite_fingerprint`` 来自 ``suite["fingerprint"]``，缺席时为 None."""
        assert EvalReport(run_name="x", suite={"fingerprint": "fp-1"}).suite_fingerprint == "fp-1"
        assert EvalReport(run_name="x").suite_fingerprint is None

    def test_empty_gates_is_not_passed(self):
        """空 ``gates`` 视为未通过（"还没算"不等于"算过了且通过"）."""
        report = EvalReport(run_name="x")
        assert report.gates == {}
        assert report.passed is False
        assert report.failed_gates() == []

    def test_failed_gates_follow_gate_names_order(self):
        """``failed_gates()`` 顺序固定为 ``GATE_NAMES``，与字典插入顺序无关."""
        report = EvalReport(
            run_name="x",
            gates={
                "execution": {"passed": True},
                "no_regression": {"passed": False},
                "min_pass_rate": {"passed": False},
            },
        )
        assert report.failed_gates() == ["min_pass_rate", "no_regression"]

    def test_failed_gates_lists_unknown_failing_gates(self):
        """``failed_gates()`` 把**名称未知**的失败门禁也列出来，与 ``passed`` 同口径.

        ``passed`` 看的是**全部** ``gates``，所以失败清单不能只认识
        ``GATE_NAMES`` 里的三个名字：否则会出现"``passed=False`` 但失败清单
        为空"的状态，摘要里打出一个没有内容的 ``未通过（）``。
        **两个视图必须对同一份 ``gates`` 给出一致的答案**——未知门禁名按名字
        排在三个已知门禁之后（顺序稳定，报告才能逐行比对）。
        """
        report = EvalReport(
            run_name="x",
            gates={"min_pass_rate": {"passed": True}, "mystery": {"passed": False}},
        )
        assert report.failed_gates() == ["mystery"]
        assert report.passed is False
        assert "门禁 未通过（mystery）" in report.summary_line()

    def test_failed_gates_empty_when_everything_passes(self):
        """同一条未知门禁改成通过 → 失败清单为空，且整份报告 ``passed=True``."""
        report = EvalReport(
            run_name="x",
            gates={"min_pass_rate": {"passed": True}, "mystery": {"passed": True}},
        )
        assert report.failed_gates() == []
        assert report.passed is True
        assert "门禁 通过" in report.summary_line()

    def test_summary_line_without_gates_says_unset(self):
        """``gates={}`` 时摘要写"未设置门禁"，而不是打出一对空括号.

        "还没算"与"算过了且通过"是两件事：``passed`` 为假，但失败清单里
        没有任何名字可报——此时必须把原因写清楚。
        """
        report = EvalReport(run_name="x")
        assert report.gates == {}
        assert report.failed_gates() == []
        assert report.passed is False
        assert "未设置门禁" in report.summary_line()
        assert "未通过（）" not in report.summary_line()

    def test_to_dict_flattened_fields(self):
        """``to_dict`` 额外拍平四个便于查询的字段（它们不是 dataclass 字段）."""
        report = build_report(
            run=make_sample_run(),
            suite={"total": 3, "fingerprint": "fp-1"},
            manifest=make_manifest(),
        )
        payload = report.to_dict()
        assert payload["adapter_sha256"] == MANIFEST_SHA256
        assert payload["adapter_step"] == 4
        assert payload["suite_fingerprint"] == "fp-1"
        assert payload["passed"] is True
        field_names = {item.name for item in dataclasses.fields(EvalReport)}
        assert not {"adapter_sha256", "adapter_step", "suite_fingerprint", "passed"} & field_names

    def test_to_dict_is_json_serializable(self):
        """拍平后的字典可以 ``json.dumps``（报告最终要落成 JSON）."""
        report = build_report(
            run=make_sample_run(),
            suite={"total": 3, "fingerprint": "fp-1"},
            manifest=make_manifest(),
            comparison=make_regressing_comparison(),
        )
        payload = report.to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_from_dict_roundtrip(self):
        """``from_dict`` 往返：拍平字段与未知键都被忽略，其余逐字段一致."""
        original = build_report(
            run=make_sample_run(),
            suite={"total": 3, "fingerprint": "fp-1"},
            manifest=make_manifest(),
            comparison=make_regressing_comparison(),
            notes=["脚本化对照臂"],
        )
        restored = EvalReport.from_dict({**original.to_dict(), "未知键": 1})
        assert restored == original
        assert restored.to_dict() == original.to_dict()
        assert restored.adapter_sha256 == MANIFEST_SHA256

    def test_summary_line_without_adapter(self):
        """未绑定适配器时 ``summary_line`` 如实写"未绑定适配器"与"未绑定"指纹."""
        report = build_report(run=make_sample_run())
        line = report.summary_line()
        assert "适配器 未绑定适配器" in line
        assert "评估集 未绑定" in line
        assert "合格率 66.67%" in line
        assert "门禁 通过" in line

    def test_summary_line_with_adapter_and_failure(self):
        """绑定后摘要里出现 ``sha256:`` 前缀的短哈希，失败时列出失败门禁名."""
        report = build_report(
            run=make_sample_run(),
            suite={"total": 3, "fingerprint": "fp-1"},
            manifest=make_manifest(),
            min_pass_rate=1.0,
        )
        line = report.summary_line()
        assert f"sha256:{MANIFEST_SHA256[:12]}" in line
        assert "门禁 未通过（min_pass_rate）" in line

    def test_pass_rate_defaults_to_zero(self):
        """``summary`` 为空时 ``pass_rate`` 是 0.0（报告要能打印半成品）."""
        assert EvalReport(run_name="x").pass_rate == 0.0


class TestVerifyManifestBinding:
    """``verify_manifest_binding``：哈希 / 基座 / 步数三者必须一致."""

    def test_report_without_manifest_raises(self):
        """报告没有 manifest → 抛错（对不上适配器的报告比没有报告更危险）."""
        report = build_report(run=make_sample_run())
        with pytest.raises(FinetuneEvalError) as excinfo:
            verify_manifest_binding(report, make_manifest())
        assert "没有绑定适配器清单" in str(excinfo.value)

    def test_matching_manifest_passes(self):
        """完全一致时不抛，且返回 ``None``（这是"核对通过"的唯一信号）."""
        report = build_report(run=make_sample_run(), manifest=make_manifest())
        assert verify_manifest_binding(report, make_manifest()) is None

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("content_sha256", "b" * 64),
            ("base_model", "other/model"),
            ("step", 9),
        ],
    )
    def test_each_mismatched_key_raises(self, key, value):
        """``content_sha256`` / ``base_model`` / ``step`` 各造一次不一致，各抛一次."""
        report = build_report(run=make_sample_run(), manifest=make_manifest())
        with pytest.raises(FinetuneEvalError) as excinfo:
            verify_manifest_binding(report, make_manifest(**{key: value}))
        assert key in str(excinfo.value)

    def test_only_three_keys_are_compared(self):
        """清单里的其他字段（目录、tags、lora、参数量）不参与核对.

        这是刻意的口径：一次部署换目录名、加个 tag 不该让旧报告失效，
        而权重内容 / 基座 / 步数变了就必须失效。
        """
        report = build_report(run=make_sample_run(), manifest=make_manifest())
        other = make_manifest(
            adapter_dir="/tmp/another-dir",
            adapter_bytes=1,
            trainable_parameters=1,
            lora={"r": 64},
        )
        assert verify_manifest_binding(report, other) is None


class TestRenderMarkdown:
    """``render_markdown``：与 JSON 同源的人读版本."""

    def test_binding_table_lists_all_five_rows(self):
        """含绑定信息表：指纹 / 条数 / 适配器哈希 / 基座 / 步数."""
        report = build_report(
            run=make_sample_run(),
            suite={"total": SUITE_TOTAL, "fingerprint": "fp-1"},
            manifest=make_manifest(),
        )
        markdown = render_markdown(report)
        assert "## 绑定信息" in markdown
        assert "| 评估集指纹 | `fp-1` |" in markdown
        assert f"| 评估集条数 | {SUITE_TOTAL} |" in markdown
        assert f"| 适配器哈希 | `{MANIFEST_SHA256[:12]}` |" in markdown
        assert "| 基座 | Qwen/Qwen3-0.6B |" in markdown
        assert "| 适配器步数 | 4 |" in markdown

    def test_binding_table_falls_back_to_unbound_text(self):
        """未绑定时三处都写"未绑定"，不留空白单元格."""
        markdown = render_markdown(build_report(run=make_sample_run()))
        assert "| 评估集指纹 | `未绑定` |" in markdown
        assert "| 评估集条数 | 未知 |" in markdown
        assert "| 适配器哈希 | `未绑定` |" in markdown
        assert "| 基座 | 未绑定 |" in markdown
        assert "| 适配器步数 | 未绑定 |" in markdown

    def test_overall_section_reports_fractions(self):
        """总体结果段给出合格率（分子/分母）、平均总分、异常数与耗时."""
        markdown = render_markdown(build_report(run=make_sample_run()))
        assert "## 总体结果" in markdown
        assert "- 合格率：66.67%（2/3）" in markdown
        assert "- 平均总分：0.6667" in markdown
        assert "- 执行异常：0 条" in markdown
        assert "毫秒）" in markdown

    def test_bucket_table_lists_every_bucket(self):
        """分桶表逐桶一行（按桶名排序，便于两次运行逐行比对）."""
        markdown = render_markdown(build_report(run=make_sample_run()))
        assert "## 分桶结果" in markdown
        assert "| citation | 1 | 1 | 100.0% | 0.9000 |" in markdown
        assert "| refusal | 1 | 0 | 0.0% | 0.2000 |" in markdown
        assert "| tool_use | 1 | 1 | 100.0% | 0.9000 |" in markdown

    def test_gate_table_lists_three_gates_with_evidence(self):
        """门禁表三行，证据列分别给合格率、回退清单、异常计数.

        合格率那一行的比较号**随判定变化**：``passed=False`` 时写 ``<``。
        证据与判定必须自洽——一行写着"否"、证据里却写 ``≥`` 的表格，会让
        读者以为门禁算错了。
        """
        report = build_report(run=make_sample_run(), min_pass_rate=1.0)
        markdown = render_markdown(report)
        assert "## 门禁" in markdown
        assert "| min_pass_rate | 否 |" in markdown
        assert "| no_regression | 是 | 未做基线比对 |" in markdown
        assert "| execution | 是 | 异常 0 / 3 条 |" in markdown
        assert "合格率 66.67% < 阈值 100.00%" in markdown
        assert "≥ 阈值" not in markdown

    def test_gate_evidence_uses_ge_when_the_gate_passes(self):
        """门禁通过时证据列写 ``≥``：两个方向的比较号都要被钉住."""
        report = build_report(run=make_sample_run(), min_pass_rate=0.5)
        markdown = render_markdown(report)
        assert "| min_pass_rate | 是 |" in markdown
        assert "合格率 66.67% ≥ 阈值 50.00%" in markdown

    def test_gate_evidence_for_regression_lists_buckets(self):
        """有比对时 ``no_regression`` 的证据列写出回退分组与容差."""
        report = build_report(
            run=make_sample_run(), comparison=make_regressing_comparison()
        )
        markdown = render_markdown(report)
        assert "回退分组 ['refusal']（容差 0.0）" in markdown
        assert "| no_regression | 否 |" in markdown

    def test_comparison_section_renders_table_and_two_statistics(self):
        """有 ``comparison`` 时渲染比对表与 McNemar / 自助法两行."""
        report = build_report(
            run=make_sample_run(), comparison=make_regressing_comparison()
        )
        markdown = render_markdown(report)
        assert "## 与基线的比对" in markdown
        assert "| 分组 | 条数 | 前合格率 | 后合格率 | 差值 | 总分差 | 回退 |" in markdown
        assert "| citation | 1 | 0.0% | 100.0% | +100.0% |" in markdown
        assert "| refusal | 1 | 100.0% | 0.0% | -100.0% |" in markdown
        assert "- McNemar 精确检验：" in markdown
        assert "- 配对自助法：" in markdown

    def test_no_comparison_section_when_absent(self):
        """没有比对时整段不出现（而不是渲染一张空表）."""
        markdown = render_markdown(build_report(run=make_sample_run()))
        assert "## 与基线的比对" not in markdown
        assert "McNemar" not in markdown
        assert "配对自助法" not in markdown

    def test_failures_are_capped_at_max_failures(self):
        """失败清单条目数不超过 ``MAX_FAILURES``，超出时出现"其余 N 条"."""
        run = make_run(
            "many-failures",
            [make_outcome(f"item-{index:02d}", passed=False) for index in range(13)],
        )
        markdown = render_markdown(build_report(run=run))
        failure_lines = [
            line for line in markdown.splitlines() if line.startswith("- `item-")
        ]
        assert MAX_FAILURES == 10
        assert len(failure_lines) == MAX_FAILURES
        assert f"## 失败清单（共 13 条，最多列 {MAX_FAILURES} 条）" in markdown
        assert "- …其余 3 条见 JSON 报告的 `outcomes`" in markdown
        assert "item-12" not in markdown

    def test_failure_reason_is_rendered(self):
        """失败原因来自 ``ItemOutcome.failure_reason()``（执行异常优先）."""
        run = make_run("boom", [make_outcome("a", passed=False, error="RuntimeError: boom")])
        markdown = render_markdown(build_report(run=run))
        assert "- `a`：执行异常：RuntimeError: boom" in markdown

    def test_no_failures_renders_a_placeholder(self):
        """没有失败条目时写"- 无"（而不是留一段空白让人以为渲染坏了）."""
        run = make_run("all-green", [make_outcome("cite-01")])
        markdown = render_markdown(build_report(run=run))
        assert "## 失败清单（共 0 条，最多列 10 条）" in markdown
        assert "- 无" in markdown

    def test_failure_list_renders_reason_and_count(self):
        """清单标题带总数与上限，条目行是"``id``：原因"（不是只有一个布尔值）."""
        markdown = render_markdown(build_report(run=make_sample_run()))
        assert "## 失败清单（共 1 条，最多列 10 条）" in markdown
        assert "- `ref-01`：未通过（原因未记录）" in markdown

    def test_notes_section_is_rendered_when_present(self):
        """``notes`` 非空时渲染说明段（时间信息只能这样显式成为内容）."""
        report = build_report(
            run=make_sample_run(), notes=["两臂为脚本化对照臂", "探针真实训练得到"]
        )
        markdown = render_markdown(report)
        assert "## 说明" in markdown
        assert "- 两臂为脚本化对照臂" in markdown
        assert "- 探针真实训练得到" in markdown

    def test_notes_section_is_absent_when_empty(self):
        """``notes`` 为空时不渲染说明段."""
        assert "## 说明" not in render_markdown(build_report(run=make_sample_run()))

    def test_markdown_ends_with_single_newline(self):
        """渲染结果以换行结尾（方便直接拼进别的文档）."""
        markdown = render_markdown(build_report(run=make_sample_run()))
        assert markdown.endswith("\n")
        assert not markdown.endswith("\n\n")


class TestWriteAndReadReport:
    """``write_report`` / ``read_report``：落盘、往返与错误路径."""

    @pytest.fixture()
    def sample_report(self) -> EvalReport:
        """一份带绑定信息与比对的报告（写盘用例共用）."""
        return build_report(
            run=make_sample_run(),
            suite={"total": 3, "fingerprint": "fp-1"},
            manifest=make_manifest(),
            comparison=make_regressing_comparison(),
            notes=["脚本化对照臂"],
        )

    def test_write_then_read_roundtrip(self, tmp_path, sample_report):
        """JSON 往返后关键字段一致（``from_dict`` 忽略拍平字段）."""
        target = write_report(tmp_path / REPORT_FILE, sample_report)
        assert target == tmp_path / REPORT_FILE
        restored = read_report(target)
        assert restored == sample_report
        assert restored.adapter_sha256 == MANIFEST_SHA256
        assert restored.suite_fingerprint == "fp-1"
        assert restored.passed is False

    def test_json_file_is_utf8_and_sorted(self, tmp_path, sample_report):
        """落盘用 ``ensure_ascii=False`` 且键有序——报告是要被人 review 的."""
        target = write_report(tmp_path / REPORT_FILE, sample_report)
        text = target.read_text(encoding="utf-8")
        assert "脚本化对照臂" in text
        assert "\\u" not in text
        raw_keys = json.loads(text).keys()
        assert list(raw_keys) == sorted(raw_keys)

    def test_markdown_is_written_alongside(self, tmp_path, sample_report):
        """传 ``markdown_path`` 时同时落一份 markdown，且内容与渲染函数一致."""
        markdown_path = tmp_path / "report.md"
        write_report(tmp_path / REPORT_FILE, sample_report, markdown_path=markdown_path)
        assert markdown_path.exists()
        assert markdown_path.read_text(encoding="utf-8") == render_markdown(sample_report)

    def test_parent_directories_are_created(self, tmp_path, sample_report):
        """父目录不存在时自动创建（``outputs/finetune_eval/`` 往往还没建）."""
        target = tmp_path / "deep" / "nested" / REPORT_FILE
        write_report(target, sample_report, markdown_path=tmp_path / "deep" / "nested" / "r.md")
        assert target.exists()
        assert (tmp_path / "deep" / "nested" / "r.md").exists()

    def test_missing_file_raises(self, tmp_path):
        """文件不存在抛 ``FinetuneEvalError``（并带上路径）."""
        with pytest.raises(FinetuneEvalError) as excinfo:
            read_report(tmp_path / "missing.json")
        assert "找不到评估报告" in str(excinfo.value)

    def test_invalid_json_raises(self, tmp_path):
        """内容不是合法 JSON 时抛 ``FinetuneEvalError``（不是 ``JSONDecodeError``）."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(FinetuneEvalError) as excinfo:
            read_report(bad)
        assert "不是合法 JSON" in str(excinfo.value)

    def test_json_array_raises_finetune_eval_error(self, tmp_path):
        """合法 JSON 但不是对象（数组）时抛 ``FinetuneEvalError``，并说明顶层要求.

        ``read_report`` 在 ``from_dict`` **之前**显式拦下非对象，于是路由层
        "按 ``FinetuneEvalError`` 映射 400"的约定不会漏网——一个类型不对的
        顶层值会被说成"报告不合法"，而不是从 ``from_dict`` 里漏出一个
        ``AttributeError``（那会被读成"服务端崩了"）。
        """
        target = tmp_path / "list.json"
        target.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(FinetuneEvalError) as excinfo:
            read_report(target)
        assert "顶层必须是 JSON 对象" in str(excinfo.value)
        assert "list" in str(excinfo.value)

    @pytest.mark.parametrize("payload", ["1", '"report"', "null", "true"])
    def test_non_object_json_raises_finetune_eval_error(self, tmp_path, payload):
        """标量顶层值同样被拦下（错误信息里带上实际类型名）."""
        target = tmp_path / "scalar.json"
        target.write_text(payload, encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="顶层必须是 JSON 对象"):
            read_report(target)


class TestEstimateEvalSeconds:
    """``estimate_eval_seconds``：把"单条耗时"线性外推到目标规模."""

    @pytest.mark.parametrize("seconds_per_item", [-0.001, -1.0, -100.0])
    def test_negative_seconds_per_item_raises(self, seconds_per_item):
        """单条耗时不能为负."""
        with pytest.raises(FinetuneEvalError):
            estimate_eval_seconds(seconds_per_item=seconds_per_item, target_items=10)

    @pytest.mark.parametrize("target_items", [0, -1, -18])
    def test_non_positive_target_items_raises(self, target_items):
        """目标条数必须为正整数（0 条不是"不用跑"）."""
        with pytest.raises(FinetuneEvalError):
            estimate_eval_seconds(seconds_per_item=0.5, target_items=target_items)

    @pytest.mark.parametrize("concurrency", [0, -1, -4])
    def test_non_positive_concurrency_raises(self, concurrency):
        """并发数必须为正整数（除零会得到 ``inf``，那是个更糟的答案）."""
        with pytest.raises(FinetuneEvalError):
            estimate_eval_seconds(seconds_per_item=0.5, target_items=10, concurrency=concurrency)

    def test_linear_hand_value(self):
        """手算核对：``0.5 秒/条 × 10 条 / 2 并发 = 2.5 秒``."""
        assert estimate_eval_seconds(seconds_per_item=0.5, target_items=10, concurrency=2) == 2.5

    def test_default_concurrency_is_one(self):
        """缺省并发为 1：``0.5 × 10 = 5.0``（不并发就不打折）."""
        assert estimate_eval_seconds(seconds_per_item=0.5, target_items=10) == 5.0

    @pytest.mark.parametrize(
        ("target_items", "concurrency", "expected"),
        [(10, 1, 5.0), (10, 4, 1.25), (20, 2, 5.0), (100, 8, 6.25), (SUITE_TOTAL, 6, 1.5)],
    )
    def test_scales_linearly(self, target_items, concurrency, expected):
        """线性外推：``seconds_per_item × target_items / concurrency``."""
        assert estimate_eval_seconds(
            seconds_per_item=0.5, target_items=target_items, concurrency=concurrency
        ) == pytest.approx(expected)

    def test_zero_seconds_per_item_is_allowed(self):
        """``0.0`` 秒/条是合法输入（"这条路径还没测出耗时"），结果为 0.0."""
        assert estimate_eval_seconds(seconds_per_item=0.0, target_items=18, concurrency=1) == 0.0


class TestRunFinetuneEvaluation:
    """``run_finetune_evaluation`` 缺省路径：切分 → 两臂 → 比对 → 报告."""

    def test_default_suite_shape(self, default_outcome):
        """缺省评估集 18 条，切分为训练 12 / 评估 6（按桶分层 + 轮转位次）."""
        assert default_outcome.suite["total"] == SUITE_TOTAL
        assert len(default_outcome.train_ids) == TRAIN_SIZE
        assert len(default_outcome.eval_ids) == EVAL_SIZE

    def test_train_and_eval_ids_partition_the_suite(self, default_outcome):
        """训练 / 评估两侧**互补且不重叠**，合起来正好是整份评估集."""
        all_ids = [item.id for item in build_suite()]
        assert sorted(default_outcome.train_ids + default_outcome.eval_ids) == sorted(all_ids)
        assert not set(default_outcome.train_ids) & set(default_outcome.eval_ids)

    def test_default_leakage_and_audit(self, default_outcome):
        """泄漏体检 ``passed=True``（最大 Jaccard 远低于 0.5），难度体检 ``audit == []``."""
        assert default_outcome.leakage["passed"] is True
        assert default_outcome.leakage["max_jaccard"] < default_outcome.leakage["threshold"]
        assert default_outcome.leakage["suspicious_pairs"] == []
        assert default_outcome.audit == []

    def test_default_comparison_and_report_are_built(self, default_outcome):
        """``comparison`` 与 ``report`` 都不为 None，``probes is None``（没传探针）."""
        assert default_outcome.comparison is not None
        assert default_outcome.report is not None
        assert default_outcome.probes is None
        assert default_outcome.baseline is not None
        assert default_outcome.finetuned is not None

    def test_default_two_arms_pass_rates(self, default_outcome):
        """两臂合格率实测 ``1/3 → 5/6``，两臂跑在同一份 6 条评估集上."""
        assert default_outcome.baseline.name == "scratch-baseline"
        assert default_outcome.finetuned.name == "lora-finetuned"
        assert default_outcome.comparison.total == EVAL_SIZE
        assert default_outcome.comparison.before_pass_rate == pytest.approx(BASELINE_PASS_RATE)
        assert default_outcome.comparison.after_pass_rate == pytest.approx(FINETUNED_PASS_RATE)
        assert default_outcome.comparison.passed is True
        assert default_outcome.comparison.regressions == []

    def test_default_report_carries_the_suite_fingerprint(self, default_outcome):
        """报告带评估集指纹（``suite_stats`` 里带出来的那个），但不绑定适配器."""
        assert default_outcome.report.suite_fingerprint == default_outcome.suite["fingerprint"]
        assert default_outcome.report.adapter_sha256 is None
        assert default_outcome.report.run_name == "lora-finetuned"

    def test_outcome_to_dict_keys_and_json_roundtrip(self, default_outcome):
        """``to_dict`` 键齐全，且可 ``json.dumps(..., ensure_ascii=False)`` 往返."""
        payload = default_outcome.to_dict()
        assert set(payload) == {
            "suite",
            "audit",
            "leakage",
            "train_ids",
            "eval_ids",
            "comparison",
            "probes",
            "report",
        }
        assert payload["probes"] is None
        assert payload["report"]["suite_fingerprint"] == default_outcome.suite["fingerprint"]
        text = json.dumps(payload, ensure_ascii=False)
        assert json.loads(text) == payload

    def test_outcome_to_dict_keeps_chinese_readable(self):
        """``ensure_ascii=False``：中文说明与失败原因原样落盘，不做转义."""
        outcome = run_finetune_evaluation(notes=("两臂为脚本化对照臂",))
        text = json.dumps(outcome.to_dict(), ensure_ascii=False)
        assert "两臂为脚本化对照臂" in text
        assert "\\u" not in text

    def test_outcome_summary_line(self, default_outcome):
        """``summary_line`` 可打印，且同时给出评估集规模与配对比较结论."""
        line = default_outcome.summary_line()
        assert f"评估集 {SUITE_TOTAL} 条（训练 {TRAIN_SIZE} / 评估 {EVAL_SIZE}）" in line
        assert "scratch-baseline → lora-finetuned" in line
        assert "门禁 通过" in line

    def test_min_pass_rate_is_passed_through(self):
        """``min_pass_rate`` 进 ``report.gates`` 的阈值，并真的能拦住这次运行."""
        outcome = run_finetune_evaluation(min_pass_rate=0.9, bootstrap_samples=200)
        gate = outcome.report.gates["min_pass_rate"]
        assert gate["threshold"] == 0.9
        assert gate["value"] == pytest.approx(FINETUNED_PASS_RATE)
        assert gate["passed"] is False
        assert outcome.report.passed is False
        assert outcome.report.failed_gates() == ["min_pass_rate"]

    def test_min_pass_rate_default_is_zero(self, default_outcome):
        """缺省阈值是 0.0（"只要不崩就算过"），门禁的意义是可配置的下限."""
        assert default_outcome.report.gates["min_pass_rate"]["threshold"] == 0.0

    def test_max_regression_is_passed_through(self):
        """``max_regression`` 同时进比对结果与 ``no_regression`` 门禁."""
        outcome = run_finetune_evaluation(max_regression=0.2, bootstrap_samples=200)
        assert outcome.comparison is not None
        assert outcome.comparison.max_regression == 0.2
        assert outcome.report.gates["no_regression"]["max_regression"] == 0.2
        assert outcome.report.gates["no_regression"]["checked"] is True

    def test_alpha_is_passed_through(self):
        """``alpha`` 同时进 McNemar 与配对自助法（两处证据同一个显著性水平）."""
        outcome = run_finetune_evaluation(alpha=0.1, bootstrap_samples=200)
        assert outcome.comparison is not None
        assert outcome.comparison.mcnemar is not None
        assert outcome.comparison.bootstrap is not None
        assert outcome.comparison.mcnemar.alpha == 0.1
        assert outcome.comparison.bootstrap.alpha == 0.1

    def test_bootstrap_samples_is_passed_through(self):
        """``bootstrap_samples`` 进自助法结果（报告要能说清这是几次重采样）."""
        outcome = run_finetune_evaluation(bootstrap_samples=100)
        assert outcome.comparison is not None
        assert outcome.comparison.bootstrap is not None
        assert outcome.comparison.bootstrap.samples == 100

    def test_notes_and_seed_are_passed_through(self):
        """``notes`` 原样进报告；``seed`` 决定切分与自助法（同 seed 结果可复现）."""
        first = run_finetune_evaluation(notes=("人工标注：脚本化两臂",), bootstrap_samples=100)
        second = run_finetune_evaluation(notes=("人工标注：脚本化两臂",), bootstrap_samples=100)
        assert first.report.notes == ["人工标注：脚本化两臂"]
        assert first.eval_ids == second.eval_ids
        assert first.comparison == second.comparison

    def test_empty_items_raises(self):
        """显式传空评估集 → 统计阶段就抛错（不等跑到两臂才失败）."""
        with pytest.raises(FinetuneEvalError):
            run_finetune_evaluation(items=[])

    def test_custom_items_are_used_verbatim(self):
        """显式传 ``items`` 时用传进来的那一份（不再走 ``build_suite``）."""
        items = build_suite(difficulties=("easy",))
        outcome = run_finetune_evaluation(items=items, bootstrap_samples=100)
        assert outcome.suite["total"] == len(items)
        assert len(outcome.train_ids) + len(outcome.eval_ids) == len(items)


class TestRunFinetuneEvaluationWithProbe:
    """带 ``probe`` 的流水线：白盒探针与泛化间隙."""

    def test_probe_payload_keys(self, probe_triple):
        """``probes`` 含训练/评估两次比对、两个困惑度降幅与泛化间隙、分桶表."""
        outcome = run_finetune_evaluation(probe=probe_triple, bootstrap_samples=200)
        probes = outcome.probes
        assert probes is not None
        assert set(probes) == {
            "train",
            "eval",
            "perplexity_drop_train",
            "perplexity_drop_eval",
            "generalization_gap",
            "eval_by_bucket",
        }
        assert probes["train"]["items"] == TRAIN_SIZE
        assert probes["eval"]["items"] == EVAL_SIZE
        assert probes["train"]["before"]["label"] == "base"
        assert probes["train"]["after"]["label"] == "finetuned"

    def test_generalization_gap_equals_the_difference(self, probe_triple):
        """``generalization_gap == drop_train - drop_eval``（间隙的定义式）."""
        outcome = run_finetune_evaluation(probe=probe_triple, bootstrap_samples=200)
        probes = outcome.probes
        assert probes is not None
        assert probes["generalization_gap"] == pytest.approx(
            probes["perplexity_drop_train"] - probes["perplexity_drop_eval"]
        )

    def test_probe_improves_on_both_subsets(self, probe_triple):
        """微调后两个子集上的困惑度都下降 → 两个降幅都为正、间隙有限.

        间隙写的是"听过的那部分熟得更快多少"；它是正数不代表过拟合——
        本课只钉住"两个降幅都为正"这条可比性前提，不钉住具体数值。
        """
        outcome = run_finetune_evaluation(probe=probe_triple, bootstrap_samples=200)
        probes = outcome.probes
        assert probes is not None
        assert probes["perplexity_drop_train"] > 0
        assert probes["perplexity_drop_eval"] > 0
        assert probes["train"]["improved"] is True
        assert probes["eval"]["improved"] is True

    def test_eval_by_bucket_covers_every_bucket(self, probe_triple):
        """``eval_by_bucket`` 覆盖评估子集出现的全部桶，每个桶带三类读数."""
        outcome = run_finetune_evaluation(probe=probe_triple, bootstrap_samples=200)
        probes = outcome.probes
        assert probes is not None
        table = probes["eval_by_bucket"]
        assert set(table) == {item.bucket for item in build_suite()}
        for stats in table.values():
            assert set(stats) == {"items", "supervised", "mean_logprob", "perplexity", "accuracy"}
            assert stats["items"] >= 1
            assert stats["perplexity"] > 0

    def test_probes_travel_into_the_json_projection(self, probe_triple):
        """``probes`` 进 ``to_dict`` 且可序列化（白盒信号与文本分数一起留档）."""
        outcome = run_finetune_evaluation(probe=probe_triple, bootstrap_samples=200)
        payload = outcome.to_dict()
        assert payload["probes"] is not None
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


class TestFinetuneEvalSuiteEndpoint:
    """GET /finetune/eval/suite：评估集画像（不跑模型，纯读端点）."""

    def test_returns_200_with_full_profile(self, client):
        """200；``suite.total == 18``（统计画像永远是全量的）."""
        response = client.get("/finetune/eval/suite")
        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {
            "suite",
            "audit",
            "buckets",
            "component_names",
            "weights",
            "train_ids",
            "eval_ids",
            "eval_difficulty",
            "leakage",
            "items",
        }
        assert payload["suite"]["total"] == SUITE_TOTAL
        assert payload["suite"]["by_bucket"]["citation"] == 3

    def test_buckets_have_name_and_description(self, client):
        """``buckets`` 6 个，每个都有 ``name`` 与 ``description``（桶名不能只有作者懂）."""
        buckets = client.get("/finetune/eval/suite").json()["buckets"]
        assert len(buckets) == 6
        assert [bucket["name"] for bucket in buckets] == [
            "citation",
            "tool_use",
            "format",
            "factuality",
            "refusal",
            "conciseness",
        ]
        assert all(bucket["description"] for bucket in buckets)
        assert all(set(bucket) == {"name", "description"} for bucket in buckets)

    def test_component_names_and_weights_sum_to_one(self, client):
        """``component_names`` 6 个，``weights`` 与之一一对应且和为 1.0."""
        payload = client.get("/finetune/eval/suite").json()
        assert payload["component_names"] == list(COMPONENT_NAMES)
        assert len(payload["component_names"]) == 6
        assert set(payload["weights"]) == set(payload["component_names"])
        assert sum(payload["weights"].values()) == pytest.approx(1.0)
        assert payload["weights"]["fact_recall"] == 0.4

    def test_items_and_audit(self, client):
        """``items`` 18 条（全量，未截断）；难度体检 ``audit == []``."""
        payload = client.get("/finetune/eval/suite").json()
        assert len(payload["items"]) == SUITE_TOTAL
        assert payload["audit"] == []
        assert set(payload["items"][0]) >= {"id", "bucket", "difficulty", "reference"}

    def test_eval_difficulty_covers_all_three_levels(self, client):
        """评估子集的难度分布三个档都非空（难例必须出现在评估集里）."""
        payload = client.get("/finetune/eval/suite").json()
        difficulty = payload["eval_difficulty"]
        assert list(difficulty) == ["easy", "normal", "hard"]
        assert all(difficulty[level] >= 1 for level in difficulty)
        assert sum(difficulty.values()) == len(payload["eval_ids"]) == EVAL_SIZE

    def test_split_and_leakage(self, client):
        """切分为 12 / 6 且两侧不重叠，泄漏体检 ``passed=True``."""
        payload = client.get("/finetune/eval/suite").json()
        assert len(payload["train_ids"]) == TRAIN_SIZE
        assert len(payload["eval_ids"]) == EVAL_SIZE
        assert not set(payload["train_ids"]) & set(payload["eval_ids"])
        assert payload["leakage"]["passed"] is True
        assert payload["leakage"]["suspicious_pairs"] == []
        assert payload["leakage"]["max_jaccard"] < payload["leakage"]["threshold"]


class TestFinetuneEvalRunEndpoint:
    """POST /finetune/eval/run：跑一次完整评估（``probe=true`` 时会真训练）."""

    def test_default_run_is_green(self, client):
        """``{"probe": false}`` → 200；两臂 ``1/3 → 5/6``；报告通过；``probes is None``."""
        response = client.post("/finetune/eval/run", json={"probe": False})
        assert response.status_code == 200
        payload = response.json()
        assert payload["comparison"]["before_pass_rate"] == pytest.approx(BASELINE_PASS_RATE)
        assert payload["comparison"]["after_pass_rate"] == pytest.approx(FINETUNED_PASS_RATE)
        assert payload["probes"] is None
        assert payload["report"]["passed"] is True
        assert payload["suite"]["total"] == SUITE_TOTAL

    def test_default_run_report_binding_fields(self, client):
        """响应里的报告带评估集指纹、门禁三项与不绑定的适配器哈希."""
        payload = client.post("/finetune/eval/run", json={"probe": False}).json()
        report = payload["report"]
        assert report["suite_fingerprint"] == payload["suite"]["fingerprint"]
        assert report["adapter_sha256"] is None
        assert tuple(report["gates"]) == GATE_NAMES
        assert report["summary"]["total"] == EVAL_SIZE

    def test_gate_failure_is_reported_but_still_200(self, client):
        """``min_pass_rate=1.0`` → **200 但报告不通过**（门禁不是 HTTP 错误）.

        实测：响应里**没有** ``failed_gates`` 字段（它只存在于
        ``EvalReport`` 上，不是响应模型的一部分），所以失败门禁名从
        ``report["gates"]["min_pass_rate"]["passed"]`` 读。
        """
        response = client.post(
            "/finetune/eval/run", json={"probe": False, "min_pass_rate": 1.0}
        )
        assert response.status_code == 200
        report = response.json()["report"]
        assert report["passed"] is False
        assert "failed_gates" not in report
        assert report["gates"]["min_pass_rate"]["threshold"] == 1.0
        assert report["gates"]["min_pass_rate"]["passed"] is False
        assert report["gates"]["no_regression"]["passed"] is True
        assert report["gates"]["execution"]["passed"] is True

    @pytest.mark.parametrize(
        "body",
        [
            {"min_pass_rate": 2.0},
            {"eval_ratio": 0.95},
            {"alpha": 0.0},
            {"bootstrap_samples": 10},
        ],
    )
    def test_out_of_range_fields_return_422(self, client, body):
        """取值越界由 Pydantic 在进路由前拦下 → **422**（不是 400）."""
        assert client.post("/finetune/eval/run", json=body).status_code == 422

    def test_probe_true_returns_white_box_signals(self, client):
        """``probe=true``（``probe_epochs=2``）→ 200 且 ``probes`` 非空.

        这一档会在参考模型上真的训练一次 LoRA（约 3 秒），所以只写一条：
        探针的字段级断言留给 ``TestRunFinetuneEvaluationWithProbe``
        （那里共享同一个训练好的模型，不重复付算力）。
        """
        response = client.post(
            "/finetune/eval/run",
            json={"probe": True, "probe_epochs": 2, "bootstrap_samples": 200},
        )
        assert response.status_code == 200
        probes = response.json()["probes"]
        assert probes
        assert set(probes) == {
            "train",
            "eval",
            "perplexity_drop_train",
            "perplexity_drop_eval",
            "generalization_gap",
            "eval_by_bucket",
        }
        assert probes["generalization_gap"] == pytest.approx(
            probes["perplexity_drop_train"] - probes["perplexity_drop_eval"]
        )

    def test_notes_are_recorded_in_the_response(self, client):
        """端点会把自己的边界如实写进 ``notes``（脚本化两臂 vs 真实探针）."""
        payload = client.post("/finetune/eval/run", json={"probe": False}).json()
        notes = payload["report"]["notes"]
        assert any("脚本化对照臂" in note for note in notes)
        assert any("探针" in note for note in notes)

    def test_markdown_render_of_the_endpoint_report(self, client):
        """端点返回的报告字典能交给 ``render_markdown`` 渲染成人读版本.

        这一条把"两个端点 + 报告层"接在一起：响应体里的 ``report`` 就是
        ``EvalReport.to_dict()`` 的产物，``from_dict`` 之后必须能渲染。
        """
        payload = client.post("/finetune/eval/run", json={"probe": False}).json()
        report = EvalReport.from_dict(payload["report"])
        markdown = render_markdown(report)
        assert f"# 微调评估报告：{report.run_name}" in markdown
        assert "## 与基线的比对" in markdown
        assert math.isclose(report.pass_rate, FINETUNED_PASS_RATE, rel_tol=1e-9)
