"""评估报告：把分数绑回"评的是哪一份适配器、哪一份评估集"（M5-D5）.

一份评估报告如果没有绑定信息，它的寿命只有一天：明天换一个适配器、
后天改一条用例，那些数字就再也没人能解释清楚了。所以本模块把
day052 的纪律照搬到评估侧：

.. code-block:: text

    适配器侧：adapter_manifest.json  → 内容哈希 + 基座 + 步数
    评估集侧：suite_fingerprint      → 18 条用例的内容指纹
    报告侧：  finetune_eval_report.json → 两边的哈希 + 分数 + 判定 + 门禁

``verify_manifest_binding`` 是这条纪律的执行者：拿一份报告和一份清单来
核对，**哈希对不上就报错**。它守的是一个很常见的运维事故——"这个
83.3% 是哪一版适配器跑出来的"，答案只有一个来源：报告里写的那串哈希。

三处刻意的取舍：

1. **报告不写时间戳**。时间戳让报告在两个时刻必然不同，于是"这两份报告
   是不是同一件事"只能靠人眼；要留时间信息就写进 ``notes``，
   让它显式地成为内容的一部分。
2. **门禁是复数的，且逐项列出**。``passed`` 只有在 ``min_pass_rate`` /
   ``no_regression`` / ``execution`` 三项都通过时才为真——把三条判据
   压成一个布尔值，会让失败时不知道该先修哪一条。
3. **失败清单进报告**。报告最后一段列出所有不合格条目的原因，
   因为"哪些用例失败了、为什么"才是下一步要动手的地方。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.finetune_eval.compare import ComparisonReport
from smart_research_agent.finetune_eval.evaluator import EvalRun
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.peft.deploy import AdapterManifest
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 报告文件名（固定名字，便于"一个适配器目录配一份评估报告"的做法）.
REPORT_FILE = "finetune_eval_report.json"

#: 报告里失败清单最多列几条（与 ``routes.MAX_DATASET_ISSUES`` 同一条纪律：
#: 一份 500 条失败的报告没人会读，关键是**首条失败已经暴露了失败模式**）.
MAX_FAILURES = 10

#: 门禁名（``gates`` 字典的键，顺序即报告顺序）.
GATE_NAMES: tuple[str, ...] = ("min_pass_rate", "no_regression", "execution")


def estimate_eval_seconds(
    *, seconds_per_item: float, target_items: int, concurrency: int = 1
) -> float:
    """把"单条耗时"外推到目标规模（**线性，不做非线性外推**）.

    与 day052 的 ``logistic_estimate_hours`` 是同一条取舍：**把不确定的
    东西乘进一个精确公式，得到的是精确的错**。真实评估里瓶颈会从
    模型推理换成指标计算、再换成写盘，但在本课规模上用一条乘法
    足够回答问题——"再加 200 条要多花多久"。
    """
    if seconds_per_item < 0:
        raise FinetuneEvalError(f"seconds_per_item 不能为负数，收到 {seconds_per_item}")
    if target_items <= 0:
        raise FinetuneEvalError(f"target_items 必须为正整数，收到 {target_items}")
    if concurrency <= 0:
        raise FinetuneEvalError(f"concurrency 必须为正整数，收到 {concurrency}")
    return seconds_per_item * target_items / concurrency


@dataclass
class EvalReport:
    """一次微调评估的完整留档（分数 + 证据 + 绑定 + 门禁）."""

    run_name: str
    suite: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] | None = None
    gates: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reference: dict[str, Any] | None = None

    @property
    def adapter_sha256(self) -> str | None:
        """被评估的适配器内容哈希（没有绑定清单时为 ``None``）."""
        return None if self.manifest is None else self.manifest.get("content_sha256")

    @property
    def adapter_step(self) -> int | None:
        """适配器的训练步数（没有绑定清单时为 ``None``）."""
        return None if self.manifest is None else self.manifest.get("step")

    @property
    def base_model(self) -> str | None:
        """适配器的基座名（没有绑定清单时为 ``None``）."""
        return None if self.manifest is None else self.manifest.get("base_model")

    @property
    def suite_fingerprint(self) -> str | None:
        """评估集指纹（``suite_stats`` 里带出来的那个）."""
        return self.suite.get("fingerprint")

    @property
    def pass_rate(self) -> float:
        """总体合格率."""
        return float(self.summary.get("pass_rate", 0.0))

    @property
    def passed(self) -> bool:
        """三项门禁全通过才为真（空 ``gates`` 视为未通过）."""
        if not self.gates:
            return False
        return all(bool(gate.get("passed", False)) for gate in self.gates.values())

    def failed_gates(self) -> list[str]:
        """未通过的门禁名（``GATE_NAMES`` 顺序在前，额外的门禁按名字排在后）.

        为什么要把"额外的门禁"也列出来：``passed`` 看的是**全部** ``gates``，
        而这里若只列 ``GATE_NAMES``，就会出现"``passed=False`` 但失败清单为空"
        的状态——摘要里打出 ``未通过（）`` 这种没有内容的括号。
        **两个方法必须对同一份 ``gates`` 给出一致的答案。**
        """
        known = [
            name
            for name in GATE_NAMES
            if name in self.gates and not self.gates[name].get("passed", False)
        ]
        extra = sorted(
            name
            for name in self.gates
            if name not in GATE_NAMES and not self.gates[name].get("passed", False)
        )
        return known + extra

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（额外带三个便于查询的拍平字段）."""
        payload = asdict(self)
        payload["adapter_sha256"] = self.adapter_sha256
        payload["adapter_step"] = self.adapter_step
        payload["suite_fingerprint"] = self.suite_fingerprint
        payload["passed"] = self.passed
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalReport:
        """从 ``to_dict`` 的产物还原（未知键忽略）."""
        known = {item.name for item in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {key: value for key, value in payload.items() if key in known}
        return cls(**data)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        fingerprint = self.suite_fingerprint or "未绑定"
        adapter = self.adapter_sha256
        adapter_text = f"sha256:{adapter[:12]}" if adapter else "未绑定适配器"
        if self.passed:
            gate_text = "通过"
        else:
            failed = self.failed_gates()
            # 空 ``gates``（或所有已知门禁都通过、但整体仍为 False）时不能打出
            # ``未通过（）``——那是一个没有内容的括号。
            gate_text = "未通过（" + ("、".join(failed) if failed else "未设置门禁") + "）"
        return (
            f"{self.run_name} | 评估集 {fingerprint} | 适配器 {adapter_text} | "
            f"合格率 {self.pass_rate:.2%} | 门禁 {gate_text}"
        )


def build_report(
    *,
    run: EvalRun,
    suite: dict[str, Any] | None = None,
    manifest: AdapterManifest | None = None,
    comparison: ComparisonReport | None = None,
    min_pass_rate: float = 0.0,
    notes: Sequence[str] = (),
) -> EvalReport:
    """组装报告：跑一次的汇总 + （可选）与基线的比对 + 三项门禁.

    ``min_pass_rate`` 缺省 0.0（"只要不崩就算过"）：这个门禁的意义不在于
    卡一个好看的数，而在于**让"合格率"有一个可配置的下限**——把课程里
    的通过率阈值写死在代码里，会让不同任务被迫共用一个不合适的标准。
    """
    if not 0.0 <= min_pass_rate <= 1.0:
        raise FinetuneEvalError(f"min_pass_rate 必须落在 [0, 1]，收到 {min_pass_rate}")
    gates: dict[str, Any] = {
        "min_pass_rate": {
            "threshold": min_pass_rate,
            "value": run.pass_rate,
            "passed": run.pass_rate >= min_pass_rate,
        },
        "no_regression": {
            "max_regression": 0.0 if comparison is None else comparison.max_regression,
            "regressions": [] if comparison is None else list(comparison.regressions),
            "passed": True if comparison is None else comparison.passed,
            "checked": comparison is not None,
        },
        "execution": {
            "error_count": run.error_count,
            "value": run.total,
            "passed": run.error_count == 0,
        },
    }
    return EvalReport(
        run_name=run.name,
        suite=dict(suite) if suite is not None else {},
        manifest=None if manifest is None else manifest.to_dict(),
        summary=run.to_dict(),
        comparison=None if comparison is None else comparison.to_dict(),
        gates=gates,
        notes=list(notes),
        reference=_run_reference(run),
    )


def _run_reference(run: EvalRun) -> dict[str, Any]:
    """从一次运行里抽出"下一次要接上真模型时需要的接线信息".

    ``runner`` 不在报告里（它可能是个不可序列化的闭包），但**两臂的名字与
    条数**必须在——否则报告读者无法判断"这两次运行是不是同一套流程"。
    """
    return {
        "run_name": run.name,
        "items": run.total,
        "error_count": run.error_count,
        "mean_latency_seconds": run.mean_latency_seconds,
    }


def verify_manifest_binding(report: EvalReport, manifest: AdapterManifest) -> None:
    """核对报告与适配器清单：哈希 / 基座 / 步数三者必须一致.

    任何一项不一致就抛 ``FinetuneEvalError``。这不是"过度严格"：一份
    对不上适配器的评估报告，比没有报告更危险——**它会被当成证据引用**。
    """
    if report.manifest is None:
        raise FinetuneEvalError("这份报告没有绑定适配器清单：无法核对它评的是哪一份权重")
    expected = manifest.to_dict()
    for key in ("content_sha256", "base_model", "step"):
        if report.manifest.get(key) != expected.get(key):
            raise FinetuneEvalError(
                f"报告与适配器清单不一致（{key}）：报告 {report.manifest.get(key)!r} "
                f"vs 清单 {expected.get(key)!r}"
            )


def render_markdown(report: EvalReport) -> str:
    """把报告渲染成 markdown（人读的那一份，与 JSON 同源）."""
    adapter = report.adapter_sha256
    lines: list[str] = [
        f"# 微调评估报告：{report.run_name}",
        "",
        "## 绑定信息",
        "",
        "| 项 | 值 |",
        "|----|----|",
        f"| 评估集指纹 | `{report.suite_fingerprint or '未绑定'}` |",
        f"| 评估集条数 | {report.suite.get('total', '未知')} |",
        f"| 适配器哈希 | `{adapter[:12] if adapter else '未绑定'}` |",
        f"| 基座 | {report.base_model or '未绑定'} |",
        f"| 适配器步数 | {report.adapter_step if report.adapter_step is not None else '未绑定'} |",
        "",
        "## 总体结果",
        "",
        f"- 合格率：{report.pass_rate:.2%}"
        f"（{report.summary.get('passed', 0)}/{report.summary.get('total', 0)}）",
        f"- 平均总分：{report.summary.get('mean_score', 0.0):.4f}",
        f"- 执行异常：{report.summary.get('error_count', 0)} 条",
        f"- 耗时：{report.summary.get('wall_seconds', 0.0):.3f} 秒"
        f"（单条平均 {report.summary.get('mean_latency_seconds', 0.0) * 1000:.3f} 毫秒）",
        "",
        "## 分桶结果",
        "",
        "| 能力桶 | 条数 | 合格 | 合格率 | 平均总分 |",
        "|--------|------|------|--------|----------|",
    ]
    for bucket, stats in (report.summary.get("by_bucket") or {}).items():
        lines.append(
            f"| {bucket} | {stats['total']} | {stats['passed']} | "
            f"{stats['pass_rate']:.1%} | {stats['mean_score']:.4f} |"
        )
    if report.comparison is not None:
        lines += [
            "",
            "## 与基线的比对",
            "",
            f"- 合格率：{report.comparison['before_pass_rate']:.1%} → "
            f"{report.comparison['after_pass_rate']:.1%}"
            f"（{report.comparison['pass_rate_delta']:+.1%}）",
            f"- 平均总分：{report.comparison['before_mean_score']:.4f} → "
            f"{report.comparison['after_mean_score']:.4f}"
            f"（{report.comparison['score_delta']:+.4f}）",
            "",
            "| 分组 | 条数 | 前合格率 | 后合格率 | 差值 | 总分差 | 回退 |",
            "|------|------|----------|----------|------|--------|------|",
        ]
        for bucket in report.comparison.get("buckets", []):
            lines.append(
                f"| {bucket['name']} | {bucket['total']} | {bucket['before_pass_rate']:.1%} | "
                f"{bucket['after_pass_rate']:.1%} | {bucket['pass_rate_delta']:+.1%} | "
                f"{bucket['score_delta']:+.4f} | {'是' if bucket['regressed'] else '否'} |"
            )
        mcnemar = report.comparison.get("mcnemar")
        if mcnemar:
            lines += [
                "",
                f"- McNemar 精确检验：过→不过 {mcnemar['before_pass_after_fail']} 条，"
                f"不过→过 {mcnemar['before_fail_after_pass']} 条，"
                f"p = {mcnemar['p_value']:.6f}（显著 {mcnemar['significant']}）",
            ]
        bootstrap = report.comparison.get("bootstrap")
        if bootstrap:
            lines.append(
                f"- 配对自助法：平均总分差 {bootstrap['mean_delta']:+.4f}，"
                f"区间 [{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}]，"
                f"不含 0 {bootstrap['excludes_zero']}"
            )
    lines += ["", "## 门禁", "", "| 门禁 | 通过 | 证据 |", "|------|------|------|"]
    for name in GATE_NAMES:
        gate = report.gates.get(name)
        if gate is None:
            continue
        if name == "min_pass_rate":
            # 证据里的比较号必须与判定一致：``passed=False`` 时还写 "≥"，
            # 读者会看到一句与"否"自相矛盾的证据。
            symbol = "≥" if gate["passed"] else "<"
            evidence = f"合格率 {gate['value']:.2%} {symbol} 阈值 {gate['threshold']:.2%}"
        elif name == "no_regression":
            evidence = (
                f"回退分组 {gate['regressions']}（容差 {gate['max_regression']}）"
                if gate.get("checked")
                else "未做基线比对"
            )
        else:
            evidence = f"异常 {gate['error_count']} / {gate['value']} 条"
        lines.append(f"| {name} | {'是' if gate['passed'] else '否'} | {evidence} |")
    failures = [
        outcome for outcome in report.summary.get("outcomes", []) if not outcome["passed"]
    ]
    lines += ["", f"## 失败清单（共 {len(failures)} 条，最多列 {MAX_FAILURES} 条）", ""]
    if not failures:
        lines.append("- 无")
    for outcome in failures[:MAX_FAILURES]:
        lines.append(f"- `{outcome['item_id']}`：{outcome['failure_reason']}")
    if len(failures) > MAX_FAILURES:
        lines.append(f"- …其余 {len(failures) - MAX_FAILURES} 条见 JSON 报告的 `outcomes`")
    if report.notes:
        lines += ["", "## 说明", ""] + [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def write_report(
    path: str | Path, report: EvalReport, *, markdown_path: str | Path | None = None
) -> Path:
    """把报告写成 JSON（可选同时落一份 markdown）."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if markdown_path is not None:
        markdown_target = Path(markdown_path)
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        markdown_target.write_text(render_markdown(report), encoding="utf-8")
    logger.info("评估报告已写入 %s（%s）", target, report.summary_line())
    return target


def read_report(path: str | Path) -> EvalReport:
    """读回 JSON 报告；文件缺失或内容不是 JSON 对象都抛 ``FinetuneEvalError``."""
    source = Path(path)
    if not source.exists():
        raise FinetuneEvalError(f"找不到评估报告：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinetuneEvalError(f"评估报告不是合法 JSON: {exc}") from exc
    # 合法 JSON 不一定是对象（顶层可能是数组或字符串）。这里显式拦下，
    # 否则会在 ``from_dict`` 里漏出 ``AttributeError``——**路由层按
    # ``FinetuneEvalError`` 映射 400 的约定会因此漏网**。
    if not isinstance(payload, dict):
        raise FinetuneEvalError(
            f"评估报告的顶层必须是 JSON 对象，收到 {type(payload).__name__}"
        )
    return EvalReport.from_dict(payload)


__all__ = [
    "GATE_NAMES",
    "MAX_FAILURES",
    "REPORT_FILE",
    "EvalReport",
    "build_report",
    "estimate_eval_seconds",
    "read_report",
    "render_markdown",
    "verify_manifest_binding",
    "write_report",
]
