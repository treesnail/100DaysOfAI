"""发布门禁：这次产物能不能上线（M5-D10）.

day058 的 ``evaluate_candidate`` 与今天的 ``evaluate_gates`` 都在判"行不行"，
但它们**判的不是同一件事**，混用会让两个结论都失去意义：

| | `evaluate_candidate`（day058） | `evaluate_gates`（今天） |
|---|---|---|
| 问题 | 候选比当前**更好吗**？ | 这次产物**够格发布吗**？ |
| 判据 | 相对（与当前版本比差值） | **绝对**（与固定阈值比） |
| 输入 | 两个 ``ModelVersion`` | 一次 run 的指标 + 产物清单 |
| 缺指标时 | 无法判定（gain 无从计算） | **门禁不通过**（缺证据不能发布） |
| 结论 | promote / hold | passed / failed |

第二张表的最后一行是本模块最需要记住的一条：**相对判定在缺数据时只能说
"不知道"，绝对判定在缺数据时必须说"不行"**。把两者合并的那一天，
就会出现"因为没测延迟所以延迟门禁通过"这种事故。

## 六项门禁与它们各自挡住的故障

| 门禁 | 阈值（缺省） | 挡住的故障 |
|------|-------------|-----------|
| `pass_rate` | >= 0.5 | 训完之后效果反而变差（day053 的合格率） |
| `regression` | <= 0.0 | 相对基线**任何**退步（0.0 表示一处都不许掉） |
| `adapter_size` | <= 64 MiB | 适配器体积失控（LoRA 不该长到和基座同量级） |
| `dataset_traceability` | 必填 | 数据指纹缺失 → 这个产物三个月后说不清来路 |
| `artifact_completeness` | 必填 | 缺必需产物（只有适配器、没有合并模型） |
| `cost` | <= 0.05 美元/千 token | 单位成本失控（路由到了贵模型、prompt 太长） |

前两项从 ``metrics`` 读，后四项从 ``artifacts`` 与 ``metrics`` 联合判定。
**每一项都要把"实际值 / 阈值 / 结论 / 理由"四个字段写进报告**：
只有 `passed` 的报告在运维上是不可用的——看到"没通过"之后唯一的动作
是去读代码。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.mlops.errors import MLOpsError
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 六项门禁的名字（进报告，逐项可核对）。
GATE_PASS_RATE = "pass_rate"
GATE_REGRESSION = "regression"
GATE_ADAPTER_SIZE = "adapter_size"
GATE_DATASET_TRACEABILITY = "dataset_traceability"
GATE_ARTIFACT_COMPLETENESS = "artifact_completeness"
GATE_COST = "cost"

#: 门禁项与指标名的对应关系：门禁**只认被点名的指标**，
#: 因此"新增一个指标"不会悄悄改变门禁行为。
GATE_METRIC_NAMES: dict[str, str] = {
    GATE_PASS_RATE: "eval_pass_rate",
    GATE_REGRESSION: "eval_pass_rate_delta",
    GATE_ADAPTER_SIZE: "adapter_mebibytes",
    GATE_COST: "cost_usd_per_1k_tokens",
}

#: 必需产物槽位（与 day058 的 ``DEPLOY_REQUIRED_ARTIFACTS`` 同一集合，
#: 这里显式列出是为了让门禁的错误信息能说清"缺哪一样"）。
REQUIRED_ARTIFACTS: tuple[str, ...] = ("adapter", "merged")

#: 数据溯源要求的指标名：数据集指纹与基座名。它们是"三元组可重建"的充分条件。
TRACEABILITY_KEYS: tuple[str, ...] = ("dataset_fingerprint", "base_model")

#: 流水线会把六项门禁的结论也记进追踪器（``gate_pass_rate=1.0`` 这样），
#: 于是它们会出现在 ``Run.flat_metrics()`` 里——**但模型卡不该再抄一遍**：
#: 卡片有专门的"发布门禁"一节，把同一份结论当成"指标"再列一次，
#: 会让读者以为"门禁通过"是一个独立于门禁的观测值。
#: 因此打包层按这个前缀把它们过滤掉。
GATE_METRIC_PREFIX = "gate_"


@dataclass(frozen=True)
class ReleaseGates:
    """发布门禁策略：六个数字 + 两个布尔开关.

    ``require_traceability`` 与 ``require_artifacts`` 是**开关而不是阈值**：
    "要不要检查"和"检查的标准是多少"是两件事，用一个
    ``min_artifacts: int = 2`` 表达会同时失去两者（"至少两个产物"
    不能表达"必须含 merged"）。
    """

    min_pass_rate: float = 0.5
    max_regression: float = 0.0
    max_adapter_mebibytes: float = 64.0
    max_cost_per_1k_tokens: float = 0.05
    require_traceability: bool = True
    require_artifacts: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_pass_rate <= 1.0:
            raise MLOpsError(f"min_pass_rate 必须落在 [0, 1]，收到 {self.min_pass_rate}")
        if self.max_regression < 0:
            raise MLOpsError(f"max_regression 不能为负数，收到 {self.max_regression}")
        if self.max_adapter_mebibytes <= 0:
            raise MLOpsError(
                f"max_adapter_mebibytes 必须为正数，收到 {self.max_adapter_mebibytes}"
            )
        if self.max_cost_per_1k_tokens <= 0:
            raise MLOpsError(
                f"max_cost_per_1k_tokens 必须为正数，收到 {self.max_cost_per_1k_tokens}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)


@dataclass(frozen=True)
class GateCheck:
    """一条门禁的结果：实际值、阈值、结论、理由.

    四个字段缺一不可（与 day058 的 ``CheckResult`` 同一形状与同一理由）。
    ``actual`` 允许为 ``None``：它表示"这个量没有观测值"，
    而缺观测值在**绝对判定**里的结论是"不通过"。
    """

    name: str
    passed: bool
    actual: float | str | None
    threshold: float | str | None
    reason: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        flag = "阻塞" if self.blocking else "告警"
        return (
            f"[{'通过' if self.passed else '不通过'}/{flag}] {self.name}: "
            f"实际 {self.actual} / 阈值 {self.threshold} —— {self.reason}"
        )


@dataclass(frozen=True)
class GateReport:
    """一次发布判定的完整报告.

    六项检查的结果分成**三类**，分开命名是刻意的：

    | 类别 | 判据 | 含义 |
    |------|------|------|
    | ``blocking_failures`` | 阻塞且未通过 | 决定 ``passed`` —— 发布被拦下 |
    | ``warnings`` | 非阻塞且未通过 | 有人**显式关掉了**这项检查（一个值得注意的决定） |
    | ``skipped_checks`` | 非阻塞且没有观测值 | 缺证据（例如首次训练没有基线） |

    如果不分开，运维看到"有 2 项没通过"时无法判断该去**补数据**还是
    该去**把开关打开**——而这两件事的动作完全不同。
    """

    passed: bool
    checks: tuple[GateCheck, ...] = ()
    commit: str = ""
    notes: str = ""

    @property
    def blocking_failures(self) -> list[GateCheck]:
        """阻塞项里没通过的（决定 ``passed`` 的就是它们）."""
        return [item for item in self.checks if item.blocking and not item.passed]

    @property
    def warnings(self) -> list[GateCheck]:
        """非阻塞项里没通过的：记录但不拦发布（当前只有"策略显式关闭"落在这里）."""
        return [item for item in self.checks if not item.blocking and not item.passed]

    @property
    def skipped_checks(self) -> list[GateCheck]:
        """**没有观测值**、按策略降级为非阻塞的检查.

        与 ``warnings`` 的差别是"没测"与"没达标"：前者要回答的是
        "我们漏了一个证据"，后者要回答的是"这个值能不能接受"。
        """
        return [item for item in self.checks if not item.blocking and item.actual is None]

    def check(self, name: str) -> GateCheck:
        """按名字取一条检查；不存在抛 ``MLOpsError``."""
        for item in self.checks:
            if item.name == name:
                return item
        raise MLOpsError(f"报告里没有门禁项 {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "passed": self.passed,
            "commit": self.commit,
            "notes": self.notes,
            "blocking_failures": [item.name for item in self.blocking_failures],
            "warnings": [item.name for item in self.warnings],
            "skipped": [item.name for item in self.skipped_checks],
            "checks": [item.to_dict() for item in self.checks],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"发布门禁 {'通过' if self.passed else '不通过'} | "
            f"{sum(1 for item in self.checks if item.passed)}/{len(self.checks)} 项通过 | "
            f"阻塞失败 {[item.name for item in self.blocking_failures] or '无'}"
        )

    def render_markdown(self) -> str:
        """把报告渲染成 markdown（可直接贴进 CI 摘要与评审文档）."""
        lines = [
            f"# 发布门禁：{'通过' if self.passed else '不通过'}",
            "",
            f"- 提交：{self.commit or '（未提供）'}",
            f"- 阻塞失败：{', '.join(item.name for item in self.blocking_failures) or '（无）'}",
            f"- 告警（策略关闭）：{', '.join(item.name for item in self.warnings) or '（无）'}",
            f"- 跳过（缺证据）：{', '.join(item.name for item in self.skipped_checks) or '（无）'}",
            f"- 备注：{self.notes or '（无）'}",
            "",
            "| 门禁 | 实际 | 阈值 | 通过 | 阻塞 | 理由 |",
            "|------|------|------|------|------|------|",
        ]
        lines.extend(
            f"| `{item.name}` | {item.actual} | {item.threshold} | "
            f"{'是' if item.passed else '否'} | {'是' if item.blocking else '否'} | "
            f"{item.reason} |"
            for item in self.checks
        )
        lines.append("")
        return "\n".join(lines)


def evaluate_gates(
    metrics: dict[str, Any],
    *,
    policy: ReleaseGates | None = None,
    artifacts: dict[str, str] | None = None,
    commit: str = "",
    notes: str = "",
) -> GateReport:
    """按策略评估六项门禁，返回完整报告.

    ``metrics`` 是**平铺的** ``{指标名: 值}``；门禁只认 ``GATE_METRIC_NAMES``
    里点名的四个名字，其余指标原样忽略（它们进追踪器，不进判定）。

    绝对判定的黄金法则在这里体现为一句代码形态：
    **"缺失" 一律落在不通过的那一侧**。
    """
    resolved = policy or ReleaseGates()
    checks: list[GateCheck] = []
    artifacts = dict(artifacts or {})

    # ---- 门禁 1：合格率下限
    rate = metrics.get(GATE_METRIC_NAMES[GATE_PASS_RATE])
    if rate is None:
        checks.append(
            GateCheck(
                name=GATE_PASS_RATE,
                passed=False,
                actual=None,
                threshold=resolved.min_pass_rate,
                reason="没有 eval_pass_rate：绝对判定里「缺证据」就是不通过（与相对判定相反）",
            )
        )
    else:
        value = float(rate)
        passed = value >= resolved.min_pass_rate
        checks.append(
            GateCheck(
                name=GATE_PASS_RATE,
                passed=passed,
                actual=value,
                threshold=resolved.min_pass_rate,
                reason=(
                    f"合格率 {value:.4f}{'达到' if passed else '低于'}下限 "
                    f"{resolved.min_pass_rate}"
                ),
            )
        )

    # ---- 门禁 2：相对基线的退步
    delta = metrics.get(GATE_METRIC_NAMES[GATE_REGRESSION])
    if delta is None:
        checks.append(
            GateCheck(
                name=GATE_REGRESSION,
                passed=True,
                actual=None,
                threshold=f"<= {resolved.max_regression}",
                reason="没有 eval_pass_rate_delta：跳过退步检查（**首次训练没有基线**）",
                blocking=False,
            )
        )
    else:
        value = float(delta)
        # 语义必须写清：``delta = 候选 − 基线``，**退步是负值**。
        # 因此 ``max_regression=0.0`` 的含义是 ``delta >= 0``（一处都不许掉），
        # 而不是 ``delta <= 0``——后者会把"提升 5 个百分点"判成失败。
        # 这个方向在实现时写反过一次，而它的表现是"所有改进都被拦下"，
        # 所以现在由一条断言钉住：``test_positive_delta_passes_the_regression_gate``。
        passed = value >= -resolved.max_regression
        checks.append(
            GateCheck(
                name=GATE_REGRESSION,
                passed=passed,
                actual=round(value, 4),
                threshold=f">= {-resolved.max_regression}",
                reason=(
                    f"相对基线 {value:+.4f}"
                    f"{'未低于' if passed else '低于'}允许的下限 "
                    f"{-resolved.max_regression}"
                ),
            )
        )

    # ---- 门禁 3：适配器体积
    size = metrics.get(GATE_METRIC_NAMES[GATE_ADAPTER_SIZE])
    if size is None:
        checks.append(
            GateCheck(
                name=GATE_ADAPTER_SIZE,
                passed=False,
                actual=None,
                threshold=resolved.max_adapter_mebibytes,
                reason="没有 adapter_mebibytes：体积不可知就不能发布（它决定部署成本）",
            )
        )
    else:
        value = float(size)
        passed = value <= resolved.max_adapter_mebibytes
        checks.append(
            GateCheck(
                name=GATE_ADAPTER_SIZE,
                passed=passed,
                actual=round(value, 4),
                threshold=resolved.max_adapter_mebibytes,
                reason=(
                    f"适配器 {value:.2f} MiB"
                    f"{'未超过' if passed else '超过'}上限 "
                    f"{resolved.max_adapter_mebibytes} MiB"
                ),
            )
        )

    # ---- 门禁 4：数据溯源（三元组可重建）
    if not resolved.require_traceability:
        # 关掉一项检查**不是"通过"**：``passed=False`` 让它落进报告的
        # ``warnings``（"有人显式关掉了这项检查"是一个值得被看见的决定），
        # 而 ``blocking=False`` 保证它不拦发布。
        checks.append(
            GateCheck(
                name=GATE_DATASET_TRACEABILITY,
                passed=False,
                actual="未检查",
                threshold="不检查",
                reason="策略显式关闭了溯源检查：这是一次有意的关闭，不是「通过」",
                blocking=False,
            )
        )
    else:
        missing = [key for key in TRACEABILITY_KEYS if not str(metrics.get(key, "")).strip()]
        checks.append(
            GateCheck(
                name=GATE_DATASET_TRACEABILITY,
                passed=not missing,
                actual=", ".join(missing) if missing else "齐全",
                threshold=", ".join(TRACEABILITY_KEYS),
                reason=(
                    "三元组可重建（数据集指纹 + 基座名都在）"
                    if not missing
                    else f"缺少 {', '.join(missing)}：这个产物三个月后说不清来路"
                ),
            )
        )

    # ---- 门禁 5：产物完整性
    if not resolved.require_artifacts:
        checks.append(
            GateCheck(
                name=GATE_ARTIFACT_COMPLETENESS,
                passed=False,
                actual="未检查",
                threshold="不检查",
                reason="策略显式关闭了产物检查：这是一次有意的关闭，不是「通过」",
                blocking=False,
            )
        )
    else:
        missing = [
            name for name in REQUIRED_ARTIFACTS if not str(artifacts.get(name, "")).strip()
        ]
        checks.append(
            GateCheck(
                name=GATE_ARTIFACT_COMPLETENESS,
                passed=not missing,
                actual=", ".join(missing) if missing else "齐全",
                threshold=", ".join(REQUIRED_ARTIFACTS),
                reason=(
                    "必需产物齐全"
                    if not missing
                    else f"缺少必需产物 {', '.join(missing)}"
                ),
            )
        )

    # ---- 门禁 6：单位成本
    cost = metrics.get(GATE_METRIC_NAMES[GATE_COST])
    if cost is None:
        checks.append(
            GateCheck(
                name=GATE_COST,
                passed=True,
                actual=None,
                threshold=resolved.max_cost_per_1k_tokens,
                reason="没有成本指标：跳过成本门禁（**离线训练不产生推理成本**）",
                blocking=False,
            )
        )
    else:
        value = float(cost)
        passed = value <= resolved.max_cost_per_1k_tokens
        checks.append(
            GateCheck(
                name=GATE_COST,
                passed=passed,
                actual=round(value, 6),
                threshold=resolved.max_cost_per_1k_tokens,
                reason=(
                    f"单位成本 {value:.6f} 美元/千 token"
                    f"{'未超过' if passed else '超过'}上限 "
                    f"{resolved.max_cost_per_1k_tokens}"
                ),
            )
        )

    blocking_failures = [item for item in checks if item.blocking and not item.passed]
    report = GateReport(
        passed=not blocking_failures,
        checks=tuple(checks),
        commit=commit,
        notes=notes,
    )
    logger.info("发布门禁：%s", report.summary_line())
    return report


def gate_table(policy: ReleaseGates | None = None) -> list[dict[str, Any]]:
    """把门禁策略渲染成一张"条件表"（API 的自我描述端点与文档同源）.

    每一行都对应 ``evaluate_gates`` 里的一段判定，
    因此"文档说 64 MiB、代码是 16 MiB"这类漂移会立刻可见。

    每行带 ``when_missing`` 一列：**缺指标时这一项是什么行为**。
    它是这张表里最容易被忽略、也最容易出事的一列——六项门禁里有三项
    "缺指标即不通过"（合格率、适配器体积、产物完整性），另三项
    "缺指标则降级为告警"（退步、成本、以及被策略关掉的溯源检查）。
    把这两类混在一起读，就会出现"因为没测延迟所以延迟门禁通过"这种事故。
    """
    resolved = policy or ReleaseGates()
    return [
        {
            "name": GATE_PASS_RATE,
            "metric": GATE_METRIC_NAMES[GATE_PASS_RATE],
            "threshold": resolved.min_pass_rate,
            "blocking": True,
            "when_missing": "不通过（缺证据不能发布）",
            "meaning": "领域评估集的合格率下限：训完之后效果反而变差必须被拦下",
        },
        {
            "name": GATE_REGRESSION,
            "metric": GATE_METRIC_NAMES[GATE_REGRESSION],
            "threshold": resolved.max_regression,
            "blocking": True,
            "when_missing": "跳过（降级为告警：首次训练没有基线）",
            "meaning": (
                "相对基线的合格率**下降**上限（delta = 候选 − 基线，退步为负值，"
                "因此判定是 delta >= -上限）"
            ),
        },
        {
            "name": GATE_ADAPTER_SIZE,
            "metric": GATE_METRIC_NAMES[GATE_ADAPTER_SIZE],
            "threshold": resolved.max_adapter_mebibytes,
            "blocking": True,
            "when_missing": "不通过（体积不可知决定不了部署成本）",
            "meaning": "适配器体积上限：LoRA 不该长到与基座同量级",
        },
        {
            "name": GATE_DATASET_TRACEABILITY,
            "metric": ", ".join(TRACEABILITY_KEYS),
            "threshold": "必填" if resolved.require_traceability else "不检查",
            "blocking": bool(resolved.require_traceability),
            "when_missing": "不通过（缺字段就重建不出三元组）",
            "meaning": "数据集指纹与基座名都在：三元组可重建，产物可追溯",
        },
        {
            "name": GATE_ARTIFACT_COMPLETENESS,
            "metric": ", ".join(REQUIRED_ARTIFACTS),
            "threshold": "必填" if resolved.require_artifacts else "不检查",
            "blocking": bool(resolved.require_artifacts),
            "when_missing": "不通过（只有适配器、没有合并模型不算交付完成）",
            "meaning": "必需产物齐全",
        },
        {
            "name": GATE_COST,
            "metric": GATE_METRIC_NAMES[GATE_COST],
            "threshold": resolved.max_cost_per_1k_tokens,
            "blocking": True,
            "when_missing": "跳过（降级为告警：离线训练不产生推理成本）",
            "meaning": "单位推理成本上限",
        },
    ]


__all__ = [
    "GATE_ADAPTER_SIZE",
    "GATE_ARTIFACT_COMPLETENESS",
    "GATE_COST",
    "GATE_DATASET_TRACEABILITY",
    "GATE_METRIC_NAMES",
    "GATE_METRIC_PREFIX",
    "GATE_PASS_RATE",
    "GATE_REGRESSION",
    "REQUIRED_ARTIFACTS",
    "TRACEABILITY_KEYS",
    "GateCheck",
    "GateReport",
    "ReleaseGates",
    "evaluate_gates",
    "gate_table",
]
