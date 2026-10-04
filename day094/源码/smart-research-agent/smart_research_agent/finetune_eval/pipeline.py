"""评估流水线：一条调用串起"切分 → 跑两臂 → 配对比较 → 探针 → 报告"（M5-D5）.

这个模块存在的理由与 day046 的 ``integration/pipeline.py`` 一样：**每一步
单独跑通不等于整条串起来跑通**。day052 的第七章记过一次真实事故——
两个 helper 各自的用例全绿，而端点的组合方式 500。评估流水线同样有
组合风险，而且风险点很具体：

- 切分之后**评估臂必须跑在评估集上**，基线臂也是。两臂跑在不同子集上
  是最容易犯的错，而且它不会报错——``compare_runs`` 会拒绝不同条数的
  输入，但"同一份条数、不同内容"只能靠调用方自觉；
- 泄漏体检必须在**切分之后**做（切分之前没有"训练集"这个概念）；
- 探针的模型与 tokenizer 必须**成对**（``CharTokenizer`` 的词表是从语料
  现造的，换一份语料换一套 id），这一点由 ``model_probe`` 的越界检查兜底；
- 报告必须在**所有数字都算完之后**组装，否则门禁里会出现"还没算的项"。

流水线把顺序固定下来，并把这些约束写成代码里的一次调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.finetune_eval.compare import (
    DEFAULT_ALPHA,
    ComparisonReport,
    compare_runs,
)
from smart_research_agent.finetune_eval.evaluator import EvalRun, run_suite, scripted_arm
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.model_probe import (
    ContextModel,
    compare_probes,
    probe_items,
)
from smart_research_agent.finetune_eval.report import EvalReport, build_report
from smart_research_agent.finetune_eval.suites import (
    DEFAULT_EVAL_RATIO,
    EvalItem,
    audit_suite,
    build_suite,
    detect_leakage,
    split_suite,
    suite_stats,
)
from smart_research_agent.peft.deploy import AdapterManifest
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 脚本化两臂的固定名字（进报告；真实模型接进来时应改成模型名 + 步数）.
BASELINE_ARM_NAME = "scratch-baseline"
FINETUNED_ARM_NAME = "lora-finetuned"

#: 探针标签（进报告）.
PROBE_BEFORE_LABEL = "base"
PROBE_AFTER_LABEL = "finetuned"


@dataclass
class FineTuneEvalOutcome:
    """一次完整评估流水线的产物（所有中间证据都留着）."""

    suite: dict[str, Any]
    audit: list[dict[str, Any]] = field(default_factory=list)
    leakage: dict[str, Any] = field(default_factory=dict)
    train_ids: list[str] = field(default_factory=list)
    eval_ids: list[str] = field(default_factory=list)
    baseline: EvalRun | None = None
    finetuned: EvalRun | None = None
    comparison: ComparisonReport | None = None
    probes: dict[str, Any] | None = None
    report: EvalReport | None = None

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if self.comparison is None or self.report is None:  # pragma: no cover - 防御式
            raise FinetuneEvalError("流水线尚未完成：comparison / report 缺失")
        return (
            f"评估集 {self.suite['total']} 条（训练 {len(self.train_ids)} / "
            f"评估 {len(self.eval_ids)}）| {self.comparison.summary_line()}"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（不重复放两臂的逐条明细）."""
        if self.report is None:  # pragma: no cover - 防御式
            raise FinetuneEvalError("流水线尚未完成：report 缺失")
        return {
            "suite": self.suite,
            "audit": self.audit,
            "leakage": self.leakage,
            "train_ids": list(self.train_ids),
            "eval_ids": list(self.eval_ids),
            "comparison": None if self.comparison is None else self.comparison.to_dict(),
            "probes": self.probes,
            "report": self.report.to_dict(),
        }


def run_finetune_evaluation(
    *,
    items: list[EvalItem] | None = None,
    manifest: AdapterManifest | None = None,
    eval_ratio: float = DEFAULT_EVAL_RATIO,
    seed: int = 42,
    min_pass_rate: float = 0.0,
    max_regression: float = 0.0,
    bootstrap_samples: int = 2000,
    alpha: float = DEFAULT_ALPHA,
    probe: tuple[ContextModel, ContextModel, Any] | None = None,
    notes: tuple[str, ...] = (),
) -> FineTuneEvalOutcome:
    """跑完整条评估流水线（**评估臂跑在评估集上，基线臂跑在同一份评估集上**）.

    ``probe`` 传 ``(微调前模型, 微调后模型, tokenizer)`` 时会额外做一次
    模型侧探针，并给出**泛化间隙**：训练子集上的困惑度降幅减去评估子集上的
    降幅。间隙为正说明"在见过的文本上变熟得更快"，这正是过拟合的定义——
    而它在文本指标上不一定看得出来（脚本化两臂的输出根本不依赖模型）。
    """
    suite_items = build_suite() if items is None else list(items)
    stats = suite_stats(suite_items)
    audit = audit_suite(suite_items)
    train_items, eval_items = split_suite(suite_items, eval_ratio=eval_ratio, seed=seed)
    leakage = detect_leakage(train_items, eval_items)
    baseline_run = run_suite(
        eval_items, scripted_arm(arm="baseline"), name=BASELINE_ARM_NAME
    )
    finetuned_run = run_suite(
        eval_items, scripted_arm(arm="finetuned"), name=FINETUNED_ARM_NAME
    )
    comparison = compare_runs(
        baseline_run,
        finetuned_run,
        max_regression=max_regression,
        alpha=alpha,
        samples=bootstrap_samples,
        seed=seed,
    )
    probes = _run_probes(probe, train_items, eval_items) if probe is not None else None
    report = build_report(
        run=finetuned_run,
        suite=stats,
        manifest=manifest,
        comparison=comparison,
        min_pass_rate=min_pass_rate,
        notes=notes,
    )
    outcome = FineTuneEvalOutcome(
        suite=stats,
        audit=audit,
        leakage=leakage,
        train_ids=[item.id for item in train_items],
        eval_ids=[item.id for item in eval_items],
        baseline=baseline_run,
        finetuned=finetuned_run,
        comparison=comparison,
        probes=probes,
        report=report,
    )
    logger.info("评估流水线完成：%s", outcome.summary_line())
    return outcome


def _run_probes(
    probe: tuple[ContextModel, ContextModel, Any],
    train_items: list[EvalItem],
    eval_items: list[EvalItem],
) -> dict[str, Any]:
    """在训练子集与评估子集上各探一次，给出泛化间隙."""
    before_model, after_model, tokenizer = probe
    train_before = probe_items(before_model, tokenizer, train_items, label=PROBE_BEFORE_LABEL)
    train_after = probe_items(after_model, tokenizer, train_items, label=PROBE_AFTER_LABEL)
    eval_before = probe_items(before_model, tokenizer, eval_items, label=PROBE_BEFORE_LABEL)
    eval_after = probe_items(after_model, tokenizer, eval_items, label=PROBE_AFTER_LABEL)
    train_comparison = compare_probes(train_before, train_after)
    eval_comparison = compare_probes(eval_before, eval_after)
    return {
        "train": train_comparison,
        "eval": eval_comparison,
        "perplexity_drop_train": train_comparison["perplexity_delta"],
        "perplexity_drop_eval": eval_comparison["perplexity_delta"],
        "generalization_gap": (
            train_comparison["perplexity_delta"] - eval_comparison["perplexity_delta"]
        ),
        "eval_by_bucket": eval_after.by_bucket(),
    }


__all__ = [
    "BASELINE_ARM_NAME",
    "FINETUNED_ARM_NAME",
    "PROBE_AFTER_LABEL",
    "PROBE_BEFORE_LABEL",
    "FineTuneEvalOutcome",
    "run_finetune_evaluation",
]
