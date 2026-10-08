"""打包与模型卡：让产物能自描述（M5-D10）.

day058 的 ``ModelVersion`` 已经带着"三元组 + 指标 + 产物路径 + 父版本"，
但那是一份**机器记录**：字段名是 `eval_pass_rate`、`dataset_fingerprint`，
而拿到它的人需要知道这些字段的含义才能读懂。

模型卡（model card）是这份记录的**人类读物**，它回答四个问题：

```text
它是什么？      用途与适用场景（intended use）
它好不好？      指标 + 发布门禁的逐项结论
它凭什么可信？  三元组（基座 / 适配器哈希 / 数据集指纹）+ 提交号
它不能做什么？  已知限制与明确排除的用途（limitations / out of scope）
```

第四行是本模块存在的**主要原因**。前三个问题在 day058 的注册表里已经能
回答，而"它不能做什么"从来没有人写下来——于是每一次"模型答错了"的讨论
都要从头争一遍"这算不算预期内"。

## 三条限制不是模板文字，而是本课程的真实边界

``LIMITATIONS`` 里的每一条都对应前九天里一个被实测过的数字：

| 限制 | 出处 |
|------|------|
| 领域评估集只有 18 条用例，合格率的分辨率是 `1/18 ≈ 0.0556` | day053 的评估套件 |
| 参考模型是 557×557 的双字符 bigram（约 31 万参数），不是真实 LLM | day050 的参考模型 |
| 单一数据来源与单一语言，跨领域表现未测 | day048/day057 的数据画像 |

**把限制写成"可能在某些情况下表现不佳"是没有用的**；写成"合格率的最小
可分辨变化是 0.0556，小于它的差异不可作为提升证据"才有用——因为后者
可以被用来**否掉一个结论**。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.mlops.errors import MLOpsError
from smart_research_agent.mlops.gates import GATE_METRIC_PREFIX, GateReport
from smart_research_agent.mlops.tracking import Run

#: 模型卡的文件名（随产物一起落盘）。
MODEL_CARD_FILENAME = "MODEL_CARD.md"

#: 发布清单的文件名。
RELEASE_MANIFEST_FILENAME = "release_manifest.json"

#: 本节四条限制（每条都能被前几天的数字支持，见模块 docstring）。
LIMITATIONS: tuple[str, ...] = (
    "领域评估集只有 18 条用例，合格率的最小可分辨变化是 1/18 ≈ 0.0556："
    "小于它的分数差异不可作为提升证据",
    "参考模型是 557×557 的双字符 bigram（约 31 万参数），"
    "它与真实 LLM 的差距远大于任何超参调整带来的差距",
    "训练数据来自单一来源、单一语言（中文技术问答），跨领域与跨语言表现未测",
    "质量门禁只覆盖格式与治理（day057 的五维质量分），"
    "不覆盖事实正确性——`drop_tail` 这类内容缺陷质量分察觉不到",
)

#: 明确的适用场景。
INTENDED_USE: tuple[str, ...] = (
    "作为领域 SFT / LoRA 微调的**流程验证与回归基线**："
    "确认数据 → 训练 → 评估 → 门禁 → 注册这一条链路可复现",
    "作为接入真实基座时的**接线检查**：三元组、内容哈希、报告字段都能对上",
)

#: 明确排除的用途（比"不适用于生产"具体得多）。
OUT_OF_SCOPE: tuple[str, ...] = (
    "直接面向终端用户提供服务：参考模型不具备可用的语言能力",
    "作为通用问答或通用 Agent 的能力依据",
    "用于任何需要事实准确性的场景（限制第 4 条）",
)


@dataclass
class ModelCard:
    """模型卡：机器记录的人类读物（结构固定，字段全部来自实测）."""

    title: str
    version: str
    base_model: str
    adapter_sha256: str
    dataset_fingerprint: str
    run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    gate_passed: bool = False
    gate_summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    parent_version: str = ""
    commit: str = ""
    intended_use: tuple[str, ...] = INTENDED_USE
    out_of_scope: tuple[str, ...] = OUT_OF_SCOPE
    limitations: tuple[str, ...] = LIMITATIONS
    notes: str = ""

    def __post_init__(self) -> None:
        # 三元组的三个字段缺一个，模型卡就回答不了"它凭什么可信"。
        for name, value in (
            ("base_model", self.base_model),
            ("adapter_sha256", self.adapter_sha256),
            ("dataset_fingerprint", self.dataset_fingerprint),
        ):
            if not str(value).strip():
                raise MLOpsError(f"模型卡缺少 {name}：三元组不完整就无法追溯")

    @property
    def short_adapter(self) -> str:
        """适配器哈希前 12 位（卡片正文里印它）."""
        return self.adapter_sha256[:12]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload = asdict(self)
        payload["short_adapter"] = self.short_adapter
        for name in ("intended_use", "out_of_scope", "limitations"):
            payload[name] = list(payload[name])
        return payload

    def render_markdown(self) -> str:
        """把模型卡渲染成 markdown（落盘成 ``MODEL_CARD.md``）."""
        lines = [
            f"# {self.title}",
            "",
            "## 身份",
            "",
            f"- 版本：**v{self.version}**" + (f"（父版本 {self.parent_version}）" if self.parent_version else "（首版）"),
            f"- 基座：`{self.base_model}`",
            f"- 适配器：`sha256:{self.short_adapter}`",
            f"- 数据集指纹：`{self.dataset_fingerprint}`",
            f"- 实验：`{self.run_id}`",
            f"- 提交：{self.commit or '（未提供）'}",
            "",
            "## 指标",
            "",
        ]
        if self.metrics:
            lines.extend(["| 指标 | 值 |", "|------|----|"])
            lines.extend(
                f"| `{name}` | {value:g} |" for name, value in sorted(self.metrics.items())
            )
        else:
            lines.append("（没有记录任何指标）")
        lines.extend(
            [
                "",
                f"## 发布门禁：{'通过' if self.gate_passed else '不通过'}",
                "",
                f"- 阻塞失败：{', '.join(self.gate_summary.get('blocking_failures', [])) or '（无）'}",
                f"- 告警：{', '.join(self.gate_summary.get('warnings', [])) or '（无）'}",
                "",
                "## 产物",
                "",
            ]
        )
        if self.artifacts:
            lines.extend(
                f"- `{name}`：`{path}`" for name, path in sorted(self.artifacts.items())
            )
        else:
            lines.append("（没有登记任何产物）")
        lines.extend(["", "## 适用场景", ""])
        lines.extend(f"- {item}" for item in self.intended_use)
        lines.extend(["", "## 明确不适用", ""])
        lines.extend(f"- {item}" for item in self.out_of_scope)
        lines.extend(["", "## 已知限制", ""])
        lines.extend(f"- {item}" for item in self.limitations)
        if self.notes:
            lines.extend(["", "## 备注", "", self.notes])
        lines.append("")
        return "\n".join(lines)


def build_model_card(
    *,
    version: str,
    base_model: str,
    adapter_sha256: str,
    dataset_fingerprint: str,
    run: Run | None = None,
    gate: GateReport | None = None,
    artifacts: dict[str, str] | None = None,
    parent_version: str = "",
    commit: str = "",
    title: str = "智研 AI 助手 —— 领域微调模型卡",
    notes: str = "",
) -> ModelCard:
    """组装一份模型卡.

    ``run`` 与 ``gate`` 都是可选的，但**缺了它们卡片会明确写出来**
    （"没有记录任何指标" / "发布门禁：不通过"），而不是留空白。
    空白会被读者当成"还没来得及填"，而"明确写出的缺失"才会被当成一个
    需要处理的状态。

    指标取 ``run.flat_metrics()``（每个指标的最后一步值）：
    卡片是**结论性文档**，逐步的 loss 曲线属于追踪器，不属于卡片。

    但 ``gate_`` 前缀的指标**被过滤掉**：它们是门禁自己的结论
    （``gate_pass_rate=1.0`` 这样），而卡片有专门的"发布门禁"一节。
    把同一份结论当"指标"再列一次，会让读者以为"门禁通过"是一个
    独立于门禁的观测值——**同一份事实在一份文档里出现两次且措辞不同，
    是让评审产生不信任的最快方式。**
    """
    metrics = (
        {
            name: value
            for name, value in run.flat_metrics().items()
            if not name.startswith(GATE_METRIC_PREFIX)
        }
        if run is not None
        else {}
    )
    gate_summary = gate.to_dict() if gate is not None else {}
    return ModelCard(
        title=title,
        version=version,
        base_model=base_model,
        adapter_sha256=adapter_sha256,
        dataset_fingerprint=dataset_fingerprint,
        run_id="" if run is None else run.run_id,
        metrics=metrics,
        gate_passed=bool(gate is not None and gate.passed),
        gate_summary=gate_summary,
        artifacts=dict(artifacts or {}),
        parent_version=parent_version,
        commit=commit,
        notes=notes,
    )


def build_release_manifest(
    *,
    card: ModelCard,
    run: Run | None = None,
    gate: GateReport | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装发布清单（JSON，供机器消费；模型卡供人阅读）.

    两份产物**内容重叠但用途不同**：清单要让另一个脚本能直接取字段，
    卡片要让一个评审能在五分钟内判断"能不能用"。只留一份的后果是
    总有一方要去解析对方的格式。
    """
    return {
        "version": card.version,
        "parent_version": card.parent_version,
        "base_model": card.base_model,
        "adapter_sha256": card.adapter_sha256,
        "dataset_fingerprint": card.dataset_fingerprint,
        "run_id": card.run_id,
        "commit": card.commit,
        "metrics": dict(card.metrics),
        "gate_passed": card.gate_passed,
        "gate": {} if gate is None else gate.to_dict(),
        "artifacts": dict(card.artifacts),
        "run_status": None if run is None else run.status,
        "run_duration_seconds": None if run is None else round(run.duration_seconds, 4),
        "limitations_count": len(card.limitations),
        "out_of_scope_count": len(card.out_of_scope),
        "extra": dict(extra or {}),
    }


__all__ = [
    "INTENDED_USE",
    "LIMITATIONS",
    "MODEL_CARD_FILENAME",
    "OUT_OF_SCOPE",
    "RELEASE_MANIFEST_FILENAME",
    "ModelCard",
    "build_model_card",
    "build_release_manifest",
]
