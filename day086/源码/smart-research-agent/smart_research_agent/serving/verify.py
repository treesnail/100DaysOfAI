"""上线验证：在切流量之前，把"专属模型够不够好"变成一个可复核的判定（M5-D11）.

day053 已经有一套领域评估（18 条用例、六分量合取判定），day059 又把它接进了
发布门禁。今天的问题不是"再评估一次"，而是回答一个**部署特有的问题**：

> 同一批真实问题上，专属模型与现在正在服务的云端模型，**差在哪里**？

差在三处，而它们必须分开看：

| 维度 | 为什么必须单独看 |
|------|-----------------|
| **合格率** | 有没有变差。差值可能是噪声（day053 的配对检验就是为它准备的） |
| **错误数** | 抛异常（超时、连不上、显存不足）与"答得不对"是两件事：前者是部署问题，后者是模型问题 |
| **延迟** | 自建推理的延迟分布与云端完全不同（首 token 更快、长尾更长），它决定这次切换能不能过 |

``run_verification`` 把两个推理臂放在**同一批用例**上跑，然后给出五条检查。
其中三条是阻塞的，两条刻意不是——**"没测"与"没达标"必须分开报**：

```text
pass_rate   专属合格率 >= 下限            阻塞
regression  相对云端不出现超出容忍的退步   阻塞
errors      专属侧不能有调用失败           阻塞（可由策略关闭）
latency     专属延迟 / 云端延迟 <= 倍数    缺省**不检查**（策略里显式给出才查）
cases       用例数必须大于 0              阻塞
```

``latency`` 缺省不检查是刻意的：**"多慢算慢"取决于部署形态**，本地 CPU 推理与
A10G 上跑的差异是十倍量级，给一个全局阈值只会让它被无脑放宽。把它设成
``None``（不检查）让调用方**显式给出**倍数——这与 day033 的 ``compare_runs``
坚持 "``mode`` 必须显式给出" 是同一条纪律：**一个全局缺省会被忘记，而忘记的
后果是结论反向。**

## 本模块不调用任何模型

``cloud_respond`` 与 ``dedicated_respond`` 由调用方注入。这是为了让它能在
CI 与单元测试里**离线、确定性**地运行（与 day050 以来的所有训练/评估模块同一
纪律）；真实推理由 ``scripts/serving_demo.py`` 或调用方接上真实客户端。
本模块只负责"把两个臂放在一起算"，不负责"怎么把模型跑起来"。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.serving.errors import ServingError
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 五条检查的名字（进报告，逐项可核对）。
CHECK_CASES = "cases"
CHECK_PASS_RATE = "pass_rate"
CHECK_REGRESSION = "regression"
CHECK_ERRORS = "errors"
CHECK_LATENCY = "latency"

#: 两侧的名字（与 ``switch`` 模块的路径名同一口径）。
ARM_CLOUD = "cloud"
ARM_DEDICATED = "dedicated"
ARMS: tuple[str, ...] = (ARM_CLOUD, ARM_DEDICATED)

#: 比率比较的容差。**这不是"手抖加的 1e-9"**，它修的是一个真实缺陷：
#: 合格率是两个整数相除，而 ``19/20 - 20/20`` 在 IEEE 754 双精度下的值是
#: ``-0.050000000000000044``——它比阈值 ``-0.05`` 小，于是"退步恰好等于容忍度"
#: 会被判成"超过了容忍度"。这个结论还会**随分母变化**：同样的 5 个百分点，
#: 40 条用例里差 2 条可能是 ``-0.049999999999999996``（通过），
#: 20 条用例里差 1 条是上面那个值（不通过）。
#:
#: 阈值本身是十进制的（0.05 / 0.5），因此在比较时带上一个远小于任何有意义的
#: 评估粒度（18 条评估集的最小分辨率是 0.0556）的容差，
#: 才是与"阈值是怎么写出来的"一致的判法。
RATE_EPSILON = 1e-9


@dataclass(frozen=True)
class ProbeCase:
    """一条端到端验证用例.

    ``must_contain`` 是**确定性判定**所需的片段：回复里必须出现全部片段才算
    通过。用片段而不是"LLM 打分"，是为了让验证在 CI 里可复现——
    day026 的 LLM-as-a-judge 是**评估质量**的工具，而这里要判的是
    "这次切换有没有让行为退化"，前者的方差会让后者永远给不出稳定结论。

    片段为空时用例**永远通过**：它变成一条"只测连通性"的探针
    （回答内容不论，只要没抛异常）。这是有意的用法，不是漏洞。
    """

    case_id: str
    prompt: str
    must_contain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ServingError("用例必须有 case_id：报告要按它定位到具体哪一条")
        if not self.prompt:
            raise ServingError(f"用例 {self.case_id} 的 prompt 不能为空")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload = asdict(self)
        payload["must_contain"] = list(self.must_contain)
        return payload


@dataclass(frozen=True)
class ProbeOutcome:
    """一条用例在一个推理臂上的结果."""

    case_id: str
    arm: str
    reply: str
    passed: bool
    latency_ms: float
    error: str = ""

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ServingError(f"未知推理臂 {self.arm!r}，可选 {', '.join(ARMS)}")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        state = "错误" if self.error else ("通过" if self.passed else "不通过")
        detail = self.error or (self.reply[:40] + ("…" if len(self.reply) > 40 else ""))
        return f"[{state}] {self.arm}/{self.case_id} {self.latency_ms:.3f}ms —— {detail}"


@dataclass(frozen=True)
class VerifyPolicy:
    """上线验证策略：两个阈值 + 三个开关.

    ``max_latency_ratio=None``（缺省）表示**不检查延迟**。
    它不是"检查了但阈值很宽"，而是"这一项没有被纳入判定"——
    报告里会明确写成"未检查（策略未给出倍数）"，与 day059 门禁表的
    ``when_missing`` 一列是同一种表达。
    """

    min_pass_rate: float = 0.5
    max_regression: float = 0.0
    require_no_errors: bool = True
    max_latency_ratio: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_pass_rate <= 1.0:
            raise ServingError(f"min_pass_rate 必须落在 [0, 1]，收到 {self.min_pass_rate}")
        if self.max_regression < 0:
            raise ServingError(f"max_regression 不能为负数，收到 {self.max_regression}")
        if self.max_latency_ratio is not None and self.max_latency_ratio <= 0:
            raise ServingError(
                f"max_latency_ratio 必须为正数或 None，收到 {self.max_latency_ratio}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)


@dataclass(frozen=True)
class VerifyCheck:
    """一条上线检查的结果：实际值、阈值、结论、理由.

    与 day059 的 ``GateCheck`` 是**同一种东西**，但刻意是两个类型：
    ``serving`` 子包不该为了一个四字段的记录去加载整个 ``mlops`` 包
    （那里有 CI 渲染、模型卡、六阶段流水线）。**"部署验证"与"训练发布"
    是两条独立的生命周期**，它们的检查项将来也会分叉——现在合并，
    分叉时就要做一次破坏性重构。
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
class ServingVerification:
    """一次上线验证的完整报告."""

    cases: int
    cloud_passed: int
    dedicated_passed: int
    cloud_errors: int
    dedicated_errors: int
    cloud_pass_rate: float
    dedicated_pass_rate: float
    cloud_latency_ms: float
    dedicated_latency_ms: float
    passed: bool
    policy: VerifyPolicy
    checks: tuple[VerifyCheck, ...] = ()
    outcomes: tuple[ProbeOutcome, ...] = ()
    notes: str = ""

    @property
    def delta(self) -> float:
        """专属相对云端的合格率变化（**退步是负值**，与 day053/059 同一约定）."""
        return self.dedicated_pass_rate - self.cloud_pass_rate

    @property
    def latency_ratio(self) -> float | None:
        """专属延迟 / 云端延迟；云端中位数为 0 时返回 ``None``（**除零不是 0**）.

        返回 ``None`` 而不是一个大数：调用方看到 ``None`` 会去查"为什么云端
        延迟是 0"（通常是测试里的假时钟），而看到一个 9999 会直接把它当成
        "专属模型慢得离谱"——**一个假数字比一个空值更容易骗过人**。
        """
        if self.cloud_latency_ms <= 0:
            return None
        return self.dedicated_latency_ms / self.cloud_latency_ms

    @property
    def blocking_failures(self) -> list[VerifyCheck]:
        """阻塞项里没通过的（决定 ``passed`` 的就是它们）."""
        return [item for item in self.checks if item.blocking and not item.passed]

    @property
    def warnings(self) -> list[VerifyCheck]:
        """非阻塞项里没通过的：记录但不拦切换（策略未给出的检查落在这里）."""
        return [item for item in self.checks if not item.blocking and not item.passed]

    @property
    def skipped_checks(self) -> list[VerifyCheck]:
        """**没有观测值**、按策略降级为非阻塞的检查（当前只有"策略未给出倍数"的延迟项）."""
        return [item for item in self.checks if not item.blocking and item.actual is None]

    def check(self, name: str) -> VerifyCheck:
        """按名字取一条检查；不存在抛 ``ServingError``."""
        for item in self.checks:
            if item.name == name:
                return item
        raise ServingError(f"报告里没有检查项 {name!r}")

    def failing_cases(self, arm: str = ARM_DEDICATED) -> list[ProbeOutcome]:
        """某个臂上没通过的用例（用于定位"差在哪几条题上"）."""
        return [item for item in self.outcomes if item.arm == arm and not item.passed]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "passed": self.passed,
            "cases": self.cases,
            "notes": self.notes,
            "cloud_passed": self.cloud_passed,
            "dedicated_passed": self.dedicated_passed,
            "cloud_errors": self.cloud_errors,
            "dedicated_errors": self.dedicated_errors,
            "cloud_pass_rate": round(self.cloud_pass_rate, 6),
            "dedicated_pass_rate": round(self.dedicated_pass_rate, 6),
            "delta": round(self.delta, 6),
            "cloud_latency_ms": round(self.cloud_latency_ms, 6),
            "dedicated_latency_ms": round(self.dedicated_latency_ms, 6),
            "latency_ratio": None if self.latency_ratio is None else round(self.latency_ratio, 6),
            "policy": self.policy.to_dict(),
            "blocking_failures": [item.name for item in self.blocking_failures],
            "warnings": [item.name for item in self.warnings],
            "skipped": [item.name for item in self.skipped_checks],
            "checks": [item.to_dict() for item in self.checks],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        ratio = self.latency_ratio
        latency = "未可比" if ratio is None else f"{ratio:.3f}×"
        return (
            f"上线验证 {'通过' if self.passed else '不通过'} | "
            f"{self.cases} 条用例 | 云端 {self.cloud_pass_rate:.4f} → "
            f"专属 {self.dedicated_pass_rate:.4f}（{self.delta:+.4f}）| "
            f"延迟 {latency} | 阻塞失败 "
            f"{[item.name for item in self.blocking_failures] or '无'}"
        )

    def render_markdown(self) -> str:
        """把报告渲染成 markdown（可直接贴进变更单）."""
        lines = [
            f"# 专属模型上线验证：{'通过' if self.passed else '不通过'}",
            "",
            f"- 用例数：{self.cases}",
            f"- 云端：{self.cloud_passed}/{self.cases} 通过"
            f"（{self.cloud_pass_rate:.4f}），调用失败 {self.cloud_errors} 次",
            f"- 专属：{self.dedicated_passed}/{self.cases} 通过"
            f"（{self.dedicated_pass_rate:.4f}），调用失败 {self.dedicated_errors} 次",
            f"- 变化：{self.delta:+.4f}",
            f"- 延迟中位数：云端 {self.cloud_latency_ms:.3f}ms / "
            f"专属 {self.dedicated_latency_ms:.3f}ms"
            + ("" if self.latency_ratio is None else f"（{self.latency_ratio:.3f}×）"),
            f"- 阻塞失败：{', '.join(item.name for item in self.blocking_failures) or '（无）'}",
            f"- 告警：{', '.join(item.name for item in self.warnings) or '（无）'}",
            "",
            "| 检查 | 实际 | 阈值 | 通过 | 阻塞 | 理由 |",
            "|------|------|------|------|------|------|",
        ]
        lines.extend(
            f"| `{item.name}` | {item.actual} | {item.threshold} | "
            f"{'是' if item.passed else '否'} | {'是' if item.blocking else '否'} | {item.reason} |"
            for item in self.checks
        )
        if self.outcomes:
            lines.extend(
                [
                    "",
                    "## 逐条结果",
                    "",
                    "| 臂 | 用例 | 通过 | 延迟(ms) | 备注 |",
                    "|----|------|------|---------|------|",
                ]
            )
            lines.extend(
                f"| `{item.arm}` | `{item.case_id}` | {'是' if item.passed else '否'} | "
                f"{item.latency_ms:.3f} | {item.error or '—'} |"
                for item in self.outcomes
            )
        lines.append("")
        return "\n".join(lines)


def judge(case: ProbeCase, reply: str) -> bool:
    """确定性判定：回复里是否出现了全部必需片段.

    **大小写不敏感**是刻意的：片段通常是术语（``DPO`` / ``LoRA``），
    而模型把它们写成 ``dpo`` 是同一个答案，不是一次退化。
    但片段之间的**顺序不做要求**——要求顺序会把"表述顺序不同"判成失败，
    那是评估者在为难自己。
    """
    haystack = reply.lower()
    return all(fragment.lower() in haystack for fragment in case.must_contain)


def median_of(values: Sequence[float]) -> float:
    """一组数的中位数（偶数个时取中间两个的平均值）.

    用中位数而不是平均值：自建推理的延迟是**长尾分布**（首 token 之后
    每个 token 都可能因为 batching 抖动），平均值会被少数几个慢请求拉高，
    而"典型用户感受到多慢"正是中位数在回答的问题。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _run_arm(
    cases: Sequence[ProbeCase],
    *,
    arm: str,
    respond: Callable[[ProbeCase], str],
    clock: Callable[[], float],
) -> list[ProbeOutcome]:
    """在一条推理臂上跑完全部用例（异常被捕获成 ``error``，不向上抛）.

    异常必须被捕获：一次切换前的验证里，"专属模型在这个用例上超时了"
    恰好是最需要被看见的信息。让它冒出去会让整次验证以一个堆栈结束，
    而报告里什么都不会留下。
    """
    outcomes: list[ProbeOutcome] = []
    for case in cases:
        started = clock()
        try:
            reply = respond(case)
        except Exception as exc:  # noqa: BLE001 - 逐条捕获，错误进报告
            outcomes.append(
                ProbeOutcome(
                    case_id=case.case_id,
                    arm=arm,
                    reply="",
                    passed=False,
                    latency_ms=round((clock() - started) * 1000.0, 6),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        outcomes.append(
            ProbeOutcome(
                case_id=case.case_id,
                arm=arm,
                reply=reply,
                passed=judge(case, reply),
                latency_ms=round((clock() - started) * 1000.0, 6),
            )
        )
    return outcomes


def run_verification(
    cases: Sequence[ProbeCase],
    *,
    cloud_respond: Callable[[ProbeCase], str],
    dedicated_respond: Callable[[ProbeCase], str],
    policy: VerifyPolicy | None = None,
    clock: Callable[[], float] = time.perf_counter,
    notes: str = "",
) -> ServingVerification:
    """把两个推理臂放在同一批用例上跑一遍，给出五条检查的完整报告.

    ``clock`` 可注入（缺省 ``time.perf_counter``）：延迟是**真实测量**，
    而真实测量在单元测试里不可复现。注入一个确定性计数器让"延迟检查"
    这条分支可以被逐位复现地测到——**否则这一条只能靠"跳过"来通过**，
    那等于没有测。
    """
    resolved = policy or VerifyPolicy()
    if not cases:
        # 零用例时两条臂都不会被调用：报告必须是"查不了"，而不是"全过"
        checks = (
            VerifyCheck(
                name=CHECK_CASES,
                passed=False,
                actual=0,
                threshold="> 0",
                reason="没有任何验证用例：没有证据本身就是不切换的理由",
            ),
        )
        return ServingVerification(
            cases=0,
            cloud_passed=0,
            dedicated_passed=0,
            cloud_errors=0,
            dedicated_errors=0,
            cloud_pass_rate=0.0,
            dedicated_pass_rate=0.0,
            cloud_latency_ms=0.0,
            dedicated_latency_ms=0.0,
            passed=False,
            policy=resolved,
            checks=checks,
            notes=notes,
        )

    cloud_outcomes = _run_arm(cases, arm=ARM_CLOUD, respond=cloud_respond, clock=clock)
    dedicated_outcomes = _run_arm(
        cases, arm=ARM_DEDICATED, respond=dedicated_respond, clock=clock
    )
    total = len(cases)
    cloud_passed = sum(1 for item in cloud_outcomes if item.passed)
    dedicated_passed = sum(1 for item in dedicated_outcomes if item.passed)
    cloud_errors = sum(1 for item in cloud_outcomes if item.error)
    dedicated_errors = sum(1 for item in dedicated_outcomes if item.error)
    cloud_rate = cloud_passed / total
    dedicated_rate = dedicated_passed / total
    cloud_latency = median_of([item.latency_ms for item in cloud_outcomes])
    dedicated_latency = median_of([item.latency_ms for item in dedicated_outcomes])
    ratio = None if cloud_latency <= 0 else dedicated_latency / cloud_latency

    checks: list[VerifyCheck] = [
        VerifyCheck(
            name=CHECK_CASES,
            passed=True,
            actual=total,
            threshold="> 0",
            reason=f"{total} 条用例参与验证",
        ),
        VerifyCheck(
            name=CHECK_PASS_RATE,
            passed=dedicated_rate >= resolved.min_pass_rate - RATE_EPSILON,
            actual=round(dedicated_rate, 6),
            threshold=resolved.min_pass_rate,
            reason=(
                f"专属模型合格率 {dedicated_rate:.4f}"
                f"{'达到' if dedicated_rate >= resolved.min_pass_rate - RATE_EPSILON else '低于'}下限 "
                f"{resolved.min_pass_rate}"
            ),
        ),
    ]
    # 退步方向与 day059 的门禁一律：delta = 专属 − 云端，**退步是负值**，
    # 因此判定式是 ``delta >= -max_regression``（写成 ``<=`` 会把改进判成失败）。
    # 比较带上 ``RATE_EPSILON``：见该常量的说明——"恰好等于容忍度"必须判通过，
    # 而浮点减法会让它偶尔落到阈值之下。
    delta = dedicated_rate - cloud_rate
    regression_ok = delta >= -resolved.max_regression - RATE_EPSILON
    checks.append(
        VerifyCheck(
            name=CHECK_REGRESSION,
            passed=regression_ok,
            actual=round(delta, 6),
            threshold=f">= {-resolved.max_regression}",
            reason=(
                f"相对云端 {delta:+.4f}"
                f"{'未低于' if regression_ok else '低于'}允许的下限 "
                f"{-resolved.max_regression}"
            ),
        )
    )
    if resolved.require_no_errors:
        checks.append(
            VerifyCheck(
                name=CHECK_ERRORS,
                passed=dedicated_errors == 0,
                actual=dedicated_errors,
                threshold=0,
                reason=(
                    "专属模型没有调用失败"
                    if dedicated_errors == 0
                    else f"专属模型有 {dedicated_errors} 条用例调用失败："
                    "调用失败是部署问题，与「答得不对」必须分开处理"
                ),
            )
        )
    else:
        checks.append(
            VerifyCheck(
                name=CHECK_ERRORS,
                passed=False,
                actual=dedicated_errors,
                threshold="不检查",
                reason=(
                    "策略关闭了失败检查：这是一次有意的关闭，不是「通过」"
                    "（调用失败仍会计数，但不会拦住切换）"
                ),
                blocking=False,
            )
        )
    if resolved.max_latency_ratio is None:
        checks.append(
            VerifyCheck(
                name=CHECK_LATENCY,
                passed=False,
                actual=None,
                threshold="未给出倍数",
                reason=(
                    "策略未给出延迟倍数上限：本项记为「未检查」——"
                    "多慢算慢取决于部署形态，给一个全局缺省只会让它被无脑放宽"
                ),
                blocking=False,
            )
        )
    else:
        latency_ok = (
            ratio is not None and ratio <= resolved.max_latency_ratio + RATE_EPSILON
        )
        checks.append(
            VerifyCheck(
                name=CHECK_LATENCY,
                passed=latency_ok,
                actual=None if ratio is None else round(ratio, 6),
                threshold=resolved.max_latency_ratio,
                reason=(
                    "云端延迟中位数为 0，无法计算倍数（缺证据不能算通过）"
                    if ratio is None
                    else f"专属/云端延迟倍数 {ratio:.3f}"
                    f"{'未超过' if latency_ok else '超过'}上限 "
                    f"{resolved.max_latency_ratio}"
                ),
            )
        )

    report = ServingVerification(
        cases=total,
        cloud_passed=cloud_passed,
        dedicated_passed=dedicated_passed,
        cloud_errors=cloud_errors,
        dedicated_errors=dedicated_errors,
        cloud_pass_rate=cloud_rate,
        dedicated_pass_rate=dedicated_rate,
        cloud_latency_ms=cloud_latency,
        dedicated_latency_ms=dedicated_latency,
        passed=not [item for item in checks if item.blocking and not item.passed],
        policy=resolved,
        checks=tuple(checks),
        outcomes=tuple(cloud_outcomes + dedicated_outcomes),
        notes=notes,
    )
    logger.info("上线验证：%s", report.summary_line())
    return report


def verify_table(policy: VerifyPolicy | None = None) -> list[dict[str, Any]]:
    """五条检查的对照表（API 的自我描述端点与文档同源）.

    ``when_missing`` 一列与 day059 的门禁表同一用途：**它是这张表里最容易
    被忽略、也最容易出事的一列**——「没测」与「没达标」在这里给出的是
    不同的结论（未给出的延迟检查不阻塞，缺证据的合格率则必须判不通过）。
    """
    resolved = policy or VerifyPolicy()
    return [
        {
            "name": CHECK_CASES,
            "threshold": "> 0",
            "blocking": True,
            "when_missing": "不通过（没有用例就没有证据）",
            "meaning": "验证用例数必须大于 0",
        },
        {
            "name": CHECK_PASS_RATE,
            "threshold": resolved.min_pass_rate,
            "blocking": True,
            "when_missing": "不通过（缺证据不能切换）",
            "meaning": "专属模型的合格率下限",
        },
        {
            "name": CHECK_REGRESSION,
            "threshold": f">= {-resolved.max_regression}",
            "blocking": True,
            "when_missing": "不通过（云端臂缺席时算不出差值）",
            "meaning": "相对云端的合格率变化下限（delta = 专属 − 云端，退步为负）",
        },
        {
            "name": CHECK_ERRORS,
            "threshold": 0 if resolved.require_no_errors else "不检查",
            "blocking": bool(resolved.require_no_errors),
            "when_missing": "不通过（调用失败是部署问题，不是模型问题）",
            "meaning": "专属侧不允许出现调用失败",
        },
        {
            "name": CHECK_LATENCY,
            "threshold": resolved.max_latency_ratio
            if resolved.max_latency_ratio is not None
            else "未给出倍数",
            "blocking": resolved.max_latency_ratio is not None,
            "when_missing": "跳过（降级为告警：策略未给出倍数上限）",
            "meaning": "专属/云端的延迟中位数倍数上限",
        },
    ]


__all__ = [
    "ARMS",
    "ARM_CLOUD",
    "ARM_DEDICATED",
    "CHECK_CASES",
    "CHECK_ERRORS",
    "CHECK_LATENCY",
    "CHECK_PASS_RATE",
    "CHECK_REGRESSION",
    "RATE_EPSILON",
    "ProbeCase",
    "ProbeOutcome",
    "ServingVerification",
    "VerifyCheck",
    "VerifyPolicy",
    "judge",
    "median_of",
    "run_verification",
    "verify_table",
]
