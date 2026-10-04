"""模型版本记录：一条版本"是什么状态、由谁产生、能部署吗"（M5-D9）.

``VersionTriple`` 回答"这是不是同一份产物"，本模块回答剩下三个运维问题：

1. **它是第几版、从哪一版来的**——``version`` 与 ``parent_version`` 构成版本链。
   链不是装饰：回滚要走的正是这条链（"退回到最近一个稳定祖先"），
   没有它就只能退回"最新的一版"，而最新的一版往往就是出问题的那一版；
2. **它现在处于什么状态**——``stage`` 四态（candidate / stable / rolled_back /
   archived）。**状态是版本记录的一部分，不是另一个文件**：把状态放进单独的
   "当前生产版本"配置文件里，就会出现"索引说 A 是 stable、配置文件说 B 是
   stable"的双份真相；
3. **它凭什么可以被部署**——``artifacts``（适配器目录、合并模型目录、数据集目录）
   与 ``metrics``。``is_deployable`` 只认**证据**：没有适配器路径的 ``stable``
   记录不能上线，哪怕它的指标再好看。

## 为什么记录是"不可变事实"

``ModelVersion`` 全程以**新对象**的形式流转（``with_stage`` / ``with_metrics``
返回新实例），索引文件也只在末尾追加。原因是版本记录的消费者很多
（CI 报告、回滚决策、审计、模型卡），任何一处"就地改一下"都会让另一处
读到半旧半新的对象——而"读到半旧半新的版本记录"这种缺陷不会报错，
只会让某个判断悄悄基于错误的前提成立。day057 的 ``StageRecord`` 与
``DedupeDecision`` 都是同一条思路。

## 四个状态的流转规则

```text
             register                 promote              prune
（无）──────────────→ candidate ──────────────→ stable ──────────→ archived
                          │                        │
                          │ 回滚失败/reject         │ 回滚（退位）
                          └──────────┬─────────────┘
                                     ▼
                                rolled_back
```

两条**只允许单向**的边，值得单独记住：

- ``stable → candidate`` 不允许：把已上线的版本"降级为候选"会让
  ``head()``（当前生产版本）突然返回 ``None``，而线上还在跑它。
  要下线一个 stable 版本，走 ``rolled_back`` 或 ``archived``；
- ``rolled_back → stable`` 不允许：回滚过的版本要么被修好重新登记
  （新的适配器哈希 → 新的版本键 → 新的记录），要么永久退出。
  允许它复活，等于承认"回滚"这个动作可以被撤销，那审计就失去意义了。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from smart_research_agent.registry.errors import RegistryError
from smart_research_agent.registry.version import (
    VersionTriple,
    parse_semver,
)

#: 四个阶段名（顺序即流转方向的大致次序，不是严格拓扑序）.
STAGE_CANDIDATE = "candidate"
STAGE_STABLE = "stable"
STAGE_ROLLED_BACK = "rolled_back"
STAGE_ARCHIVED = "archived"
STAGES: tuple[str, ...] = (STAGE_CANDIDATE, STAGE_STABLE, STAGE_ROLLED_BACK, STAGE_ARCHIVED)

#: 终止态：进入之后不再接受任何流转（``archived`` 与 ``rolled_back``）.
TERMINAL_STAGES: tuple[str, ...] = (STAGE_ARCHIVED, STAGE_ROLLED_BACK)

#: 允许的状态迁移（源 → 目标集合）。表驱动而不是散在 ``if`` 里：
#: 一张能打印出来的表才可以被核对，而"哪些迁移是合法的"这件事
#: 每次有人改代码都会被重新问一遍。
ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    STAGE_CANDIDATE: (STAGE_STABLE, STAGE_ROLLED_BACK, STAGE_ARCHIVED),
    STAGE_STABLE: (STAGE_ARCHIVED, STAGE_ROLLED_BACK),
    STAGE_ROLLED_BACK: (STAGE_ARCHIVED,),
    STAGE_ARCHIVED: (),
}

#: 产物槽位的固定名字。用固定槽位而不是自由字典，是为了让
#: ``is_deployable`` 能有一个**可断言的定义**（"部署需要哪几样东西"）。
ARTIFACT_ADAPTER = "adapter"
ARTIFACT_MERGED = "merged"
ARTIFACT_DATASET = "dataset"
ARTIFACT_NAMES: tuple[str, ...] = (ARTIFACT_ADAPTER, ARTIFACT_MERGED, ARTIFACT_DATASET)

#: 部署所必需的产物槽位。``dataset`` 不在其中：合并模型已经包含了数据的影响，
#: 数据集路径只用于追溯（追溯缺失不该阻止上线，但会上报为"追溯不完整"）。
DEPLOY_REQUIRED_ARTIFACTS: tuple[str, ...] = (ARTIFACT_ADAPTER, ARTIFACT_MERGED)

#: 进版本记录的核心指标名。其余指标原样保存，但不参与门禁判断——
#: **门禁只认被点名的指标**，否则"新增一个指标"就会悄悄改变门禁行为。
METRIC_PASS_RATE = "eval_pass_rate"
METRIC_TRAIN_LOSS = "train_loss"
METRIC_LATENCY_MS = "latency_ms"
CORE_METRICS: tuple[str, ...] = (METRIC_PASS_RATE, METRIC_TRAIN_LOSS, METRIC_LATENCY_MS)


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO 字符串（与 day057 ``quality_logger`` 同一口径）.

    统一 UTC 而不是本地时间：持续微调很可能跨时区跑（CI 在 UTC、
    开发机在 UTC+8），本地时间戳会让"哪个版本更新"这个问题在跨机器时答错。
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelVersion:
    """一条模型版本记录（M5-D9）.

    用法::

        record = ModelVersion(
            triple=VersionTriple(base_model, adapter_sha256, dataset_fingerprint),
            version="1.2.0",
            parent_version="1.1.0",
            metrics={"eval_pass_rate": 0.72, "train_loss": 0.41},
            artifacts={"adapter": "outputs/lora/adapters/adapter-final"},
        )
        registry.register(record)
    """

    triple: VersionTriple
    version: str
    parent_version: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    stage: str = STAGE_CANDIDATE
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 版本号与阶段名在构造期就校验：非法值写进索引之后，读回来的每一处
        # 都要重新怀疑它——**入口校验一次，胜过出口校验十次**。
        parse_semver(self.version)
        if self.stage not in STAGES:
            raise RegistryError(
                f"未知阶段 {self.stage!r}，可选 {', '.join(STAGES)}"
            )
        unknown = [name for name in self.artifacts if name not in ARTIFACT_NAMES]
        if unknown:
            raise RegistryError(
                f"未知产物槽位 {', '.join(sorted(unknown))}，"
                f"可选 {', '.join(ARTIFACT_NAMES)}"
            )

    # ------------------------------------------------------------------ 派生量
    @property
    def version_key(self) -> str:
        """内容寻址的版本键（来自三元组）."""
        return self.triple.key

    @property
    def base_model(self) -> str:
        """基座模型名（三元组字段的直通读取，免去处处 ``.triple.``）."""
        return self.triple.base_model

    @property
    def dataset_fingerprint(self) -> str:
        """数据集指纹（三元组字段的直通读取）."""
        return self.triple.dataset_fingerprint

    @property
    def adapter_sha256(self) -> str:
        """适配器内容哈希（三元组字段的直通读取）."""
        return self.triple.adapter_sha256

    @property
    def version_sort_key(self) -> tuple[int, int, int]:
        """版本号的整数段排序键（避免 ``1.10.0 < 1.9.0`` 这类字符串陷阱）."""
        return parse_semver(self.version)

    @property
    def pass_rate(self) -> float | None:
        """评估合格率（缺失返回 ``None``——**缺失与 0 是两件事**）.

        把缺失当 0 处理会让"这是第一版、还没评估"变成"这一版合格率为 0"，
        于是自动回滚逻辑会把每一版都退回。返回 ``None`` 强制调用方
        显式处理"没有分数"这个状态。
        """
        value = self.metrics.get(METRIC_PASS_RATE)
        return None if value is None else float(value)

    def missing_artifacts(self) -> list[str]:
        """部署所必需、但本条记录里缺失的产物槽位."""
        return [
            name
            for name in DEPLOY_REQUIRED_ARTIFACTS
            if not str(self.artifacts.get(name, "")).strip()
        ]

    def is_deployable(self) -> bool:
        """能否被部署：必需产物齐全（**只看证据，不看阶段与分数**）.

        刻意不把 ``stage == stable`` 写进来：``is_deployable`` 回答的是
        "这份产物能不能上线"，而"要不要上线"是阶段与门禁的事。两件事混在
        一个布尔量里，回滚计划就没法表达"目标版本可部署但需要先提升阶段"。
        """
        return not self.missing_artifacts()

    # ------------------------------------------------------------------ 流转
    def with_stage(self, stage: str, *, note: str = "") -> ModelVersion:
        """返回一个阶段被改写的新记录（并校验迁移合法性）.

        非法迁移抛 ``RegistryError`` 而不是静默接受：
        ``stable → candidate`` 会让 ``head()`` 返回 ``None``，而它的调用方
        （部署脚本、回滚决策）大多按"head 一定存在"写——静默接受等于
        把一次状态错误延迟成一次线上故障。
        """
        if stage not in STAGES:
            raise RegistryError(f"未知阶段 {stage!r}，可选 {', '.join(STAGES)}")
        if stage == self.stage:
            raise RegistryError(f"{self.version} 已经是 {stage} 阶段，无需重复流转")
        allowed = ALLOWED_TRANSITIONS[self.stage]
        if stage not in allowed:
            raise RegistryError(
                f"非法状态迁移 {self.stage} → {stage}（{self.version}）；"
                f"允许的目标：{', '.join(allowed) or '（终止态，不可迁移）'}"
            )
        merged_note = self.notes if not note else f"{self.notes} | {note}".lstrip(" |")
        return replace(self, stage=stage, notes=merged_note)

    def with_metrics(self, metrics: dict[str, float]) -> ModelVersion:
        """返回一个指标被合并的新记录（新值覆盖同名旧值）."""
        merged = dict(self.metrics)
        merged.update({name: float(value) for name, value in metrics.items()})
        return replace(self, metrics=merged)

    # ------------------------------------------------------------------ 投影
    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生量，便于报告直读）."""
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "version_key": self.version_key,
            "base_model": self.base_model,
            "adapter_sha256": self.adapter_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "created_at": self.created_at,
            "stage": self.stage,
            "artifacts": dict(self.artifacts),
            "metrics": {name: float(value) for name, value in self.metrics.items()},
            "notes": self.notes,
            "tags": dict(self.tags),
            "deployable": self.is_deployable(),
            "missing_artifacts": self.missing_artifacts(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelVersion:
        """从 ``to_dict`` 的产物还原（忽略派生键）."""
        return cls(
            triple=VersionTriple.from_dict(payload),
            version=str(payload["version"]),
            parent_version=str(payload.get("parent_version", "")),
            created_at=str(payload.get("created_at", "")) or utc_now_iso(),
            stage=str(payload.get("stage", STAGE_CANDIDATE)),
            artifacts={str(k): str(v) for k, v in dict(payload.get("artifacts", {})).items()},
            metrics={
                str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()
            },
            notes=str(payload.get("notes", "")),
            tags={str(k): str(v) for k, v in dict(payload.get("tags", {})).items()},
        )

    def summary_line(self) -> str:
        """人类可读的一行摘要（报告与日志直接打印）."""
        pass_rate = self.pass_rate
        score = "未评估" if pass_rate is None else f"合格率 {pass_rate:.4f}"
        parent = self.parent_version or "（首版）"
        return (
            f"v{self.version} {self.stage:<11} 键 {self.version_key} ← {parent} | "
            f"{score} | 产物 {len(self.artifacts)}/{len(ARTIFACT_NAMES)} | "
            f"可部署 {self.is_deployable()}"
        )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "ARTIFACT_ADAPTER",
    "ARTIFACT_DATASET",
    "ARTIFACT_MERGED",
    "ARTIFACT_NAMES",
    "CORE_METRICS",
    "DEPLOY_REQUIRED_ARTIFACTS",
    "METRIC_LATENCY_MS",
    "METRIC_PASS_RATE",
    "METRIC_TRAIN_LOSS",
    "ModelVersion",
    "STAGES",
    "STAGE_ARCHIVED",
    "STAGE_CANDIDATE",
    "STAGE_ROLLED_BACK",
    "STAGE_STABLE",
    "TERMINAL_STAGES",
    "utc_now_iso",
]
