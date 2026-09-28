"""流水线阶段：六步的定义、依赖与结构校验（M5-D10）.

day046 的 `integration/pipeline.py`、day057 的 `domain_data/pipeline.py`、
day058 的 `registry/retrain.py` 都讲过"顺序即策略"。今天这一课把它做成
**可被程序校验的结构**，而不是只写在注释里的约定：

```text
ingest → train → evaluate → gate → package → publish
取数据    训练     评估       门禁     打包      发布
```

每个阶段声明自己 ``requires``（需要哪些上游产物）与 ``produces``
（产出哪些产物），``validate_stage_order`` 逐条检查"每个 requires
都能在**它之前**的某个 produces 里找到"。这条校验能抓住两类真实错误：

1. **顺序写错**：把 ``publish`` 挪到 ``gate`` 之前，它的 ``requires``
   里那个 ``gate_report`` 就没人产出了；
2. **依赖漏写**：给 ``package`` 加一个对 ``metrics`` 的依赖，却忘了
   把它写进 ``requires``——这类错误平时不会报错，只会在某次运行时
   出现 ``KeyError``，而那时的堆栈指向的是阶段内部的某一行，
   不是"依赖表漏了一项"。

### 四个状态里为什么要有 ``blocked``

一个直觉上"够用"的设计是三个状态：``ok`` / ``failed`` / ``skipped``。
但门禁不通过时，``publish`` 属于哪一种？

- 说它是 ``failed``：把"评估没过"报成了"流水线崩了"，
  而 CI 的红灯会掩盖真正需要看的那份门禁报告；
- 说它是 ``skipped``：把"因为门禁没过所以没发"说成"这一步被配置跳过了"，
  而后者听起来像是**有意的选择**。

所以第四个状态 ``blocked`` 是必要的：**它表示"上游判定不允许我执行"**。
这与 day058 把回滚的"没有退路"输出成 ``hold`` 而不是抛异常是同一种纪律——
**把"做不到"和"出错了"分开报。**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.mlops.errors import MLOpsError

#: 六个阶段的名字（顺序即策略）。
STAGE_INGEST = "ingest"
STAGE_TRAIN = "train"
STAGE_EVALUATE = "evaluate"
STAGE_GATE = "gate"
STAGE_PACKAGE = "package"
STAGE_PUBLISH = "publish"
PIPELINE_STAGES: tuple[str, ...] = (
    STAGE_INGEST,
    STAGE_TRAIN,
    STAGE_EVALUATE,
    STAGE_GATE,
    STAGE_PACKAGE,
    STAGE_PUBLISH,
)

#: 四个阶段状态。
STAGE_OK = "ok"
STAGE_FAILED = "failed"
STAGE_SKIPPED = "skipped"
STAGE_BLOCKED = "blocked"
STAGE_STATUSES: tuple[str, ...] = (STAGE_OK, STAGE_FAILED, STAGE_SKIPPED, STAGE_BLOCKED)

#: 产物名（阶段之间传递的"东西"）。用固定名字而不是自由字符串，
#: 是为了让 ``validate_stage_order`` 能真正校验出"漏写依赖"。
ARTIFACT_DATASET = "dataset_fingerprint"
ARTIFACT_ADAPTER = "adapter_sha256"
ARTIFACT_METRICS = "metrics"
ARTIFACT_GATE_REPORT = "gate_report"
ARTIFACT_MODEL_CARD = "model_card"
ARTIFACT_VERSION = "version"

#: 失败时的默认行为：阻塞（下游不再执行）。``False`` 表示"这一步失败
#: 只记告警，流水线继续"——今天没有任何一步用得上它，保留字段是为了
#: 让"某一步失败怎么办"成为一个**显式声明的选择**而不是隐含约定。
DEFAULT_ON_FAILURE_BLOCKING = True


@dataclass(frozen=True)
class StageSpec:
    """一个阶段的声明：名字、说明、依赖与产出."""

    name: str
    description: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    blocking: bool = DEFAULT_ON_FAILURE_BLOCKING

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（列表字段转 list 便于序列化）."""
        payload = asdict(self)
        payload["requires"] = list(self.requires)
        payload["produces"] = list(self.produces)
        return payload


#: 六个阶段的声明（``stage_table()`` 与流水线共用这一份）。
STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        name=STAGE_INGEST,
        description="确认数据集指纹与基线：没有指纹的产物三个月后说不清来路",
        requires=(),
        produces=(ARTIFACT_DATASET,),
    ),
    StageSpec(
        name=STAGE_TRAIN,
        description="调用训练回调，产出适配器（内容哈希是版本三元组的一半）",
        requires=(ARTIFACT_DATASET,),
        produces=(ARTIFACT_ADAPTER,),
    ),
    StageSpec(
        name=STAGE_EVALUATE,
        description="在领域评估集上评估适配器，产出指标（门禁唯一的判据来源）",
        requires=(ARTIFACT_ADAPTER,),
        produces=(ARTIFACT_METRICS,),
    ),
    StageSpec(
        name=STAGE_GATE,
        description="按发布门禁策略做绝对判定：不通过则 publish 被阻塞",
        requires=(ARTIFACT_METRICS, ARTIFACT_ADAPTER),
        produces=(ARTIFACT_GATE_REPORT,),
    ),
    StageSpec(
        name=STAGE_PACKAGE,
        description="生成模型卡与发布清单（产物要能自描述）",
        requires=(ARTIFACT_METRICS, ARTIFACT_GATE_REPORT),
        produces=(ARTIFACT_MODEL_CARD,),
    ),
    StageSpec(
        name=STAGE_PUBLISH,
        description="把版本登记进注册表并提升为 stable（**上线是显式动作**）",
        requires=(ARTIFACT_ADAPTER, ARTIFACT_MODEL_CARD, ARTIFACT_GATE_REPORT),
        produces=(ARTIFACT_VERSION,),
    ),
)


@dataclass
class StageResult:
    """一个阶段的执行结果.

    ``duration_ms`` 用毫秒而不是秒：这六步里有四步是纯计算（毫秒级），
    用秒会得到一列 ``0.0``，而**一列全是 0 的耗时列等于没有耗时列**。
    """

    name: str
    status: str
    detail: str = ""
    duration_ms: float = 0.0
    produced: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATUSES:
            raise MLOpsError(
                f"未知阶段状态 {self.status!r}，可选 {', '.join(STAGE_STATUSES)}"
            )
        if self.name not in PIPELINE_STAGES:
            raise MLOpsError(f"未知阶段 {self.name!r}，可选 {', '.join(PIPELINE_STAGES)}")

    @property
    def ok(self) -> bool:
        """是否成功（``blocked`` 与 ``skipped`` 都不算成功，但**都不算失败**）."""
        return self.status == STAGE_OK

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"[{self.status:<7}] {self.name:<9} {self.duration_ms:7.2f} ms —— {self.detail}"
        )


def validate_stage_order(specs: tuple[StageSpec, ...] | None = None) -> None:
    """校验阶段依赖表的自洽性，不满足即抛 ``MLOpsError``.

    四条判据：

    1. 阶段名不重复；
    2. 每个 ``requires`` 都能在**它之前**的 ``produces`` 里找到
       （这条抓"顺序写错"）；
    3. 每个 ``produces`` 都不与前面重复
       （两个阶段产出同名的东西时，"谁负责它"就没有答案）；
    4. 每个阶段都产出至少一样东西。

    **刻意没有"至少有一个无依赖阶段"这条检查**：它由第 2 条自动保证——
    第一个阶段的 ``requires`` 必须落在空集里，因此只能为空。
    一条**恒为真**的断言不是保护，而是噪声：它会让人以为存在某种
    未被覆盖的情形，从而在别处写出多余的兜底代码。
    （实现时这里确实先写了一条，覆盖率报告指出它不可达，于是删掉。）
    """
    resolved = STAGE_SPECS if specs is None else specs
    seen_names: set[str] = set()
    available: set[str] = set()
    for spec in resolved:
        if spec.name in seen_names:
            raise MLOpsError(f"阶段名重复：{spec.name}")
        seen_names.add(spec.name)
        if not spec.produces:
            raise MLOpsError(
                f"阶段 {spec.name} 没有产出：一个不产出东西的阶段只是在消耗时间，"
                "而且它的下游无从声明依赖"
            )
        missing = [item for item in spec.requires if item not in available]
        if missing:
            raise MLOpsError(
                f"阶段 {spec.name} 依赖了尚未产出的 {', '.join(missing)}："
                "顺序即策略——把产出它的那一步挪到前面，或者把依赖写对"
            )
        duplicated = [item for item in spec.produces if item in available]
        if duplicated:
            raise MLOpsError(
                f"阶段 {spec.name} 重复产出了 {', '.join(duplicated)}："
                "两处产出同名的东西时，谁负责它就没有答案"
            )
        available.update(spec.produces)


def stage_table(specs: tuple[StageSpec, ...] | None = None) -> list[dict[str, Any]]:
    """把阶段表渲染成"文档与实现同源"的结构（API 与文档共用）."""
    resolved = STAGE_SPECS if specs is None else specs
    return [
        {
            "order": index,
            "name": spec.name,
            "description": spec.description,
            "requires": list(spec.requires),
            "produces": list(spec.produces),
            "blocking": spec.blocking,
        }
        for index, spec in enumerate(resolved, start=1)
    ]


def critical_path(specs: tuple[StageSpec, ...] | None = None) -> list[str]:
    """从"无依赖的起点"走到"产出最终版本的终点"的路径.

    返回阶段名列表。终点定义为**最后一个产出**满足"没有任何后续阶段依赖它"
    的阶段——在这里就是 ``publish``。它的用途不是排期，而是回答
    "这条流水线的主链是什么"：主链上任何一步失败都会让发布停下来，
    而主链之外的分支（今天没有）失败只影响自己。
    """
    resolved = STAGE_SPECS if specs is None else specs
    order = {spec.name: index for index, spec in enumerate(resolved)}
    produced_by: dict[str, str] = {}
    for spec in resolved:
        for item in spec.produces:
            produced_by[item] = spec.name
    depended_on = {item for spec in resolved for item in spec.requires}
    terminal = [spec for spec in reversed(resolved) if not set(spec.produces) & depended_on]
    if not terminal:  # pragma: no cover - 结构上不可能（最后一步的产出没人依赖）
        raise MLOpsError("找不到终点阶段：每个阶段的产出都被依赖，依赖图成环")
    path: list[str] = []
    current: str | None = terminal[0].name
    by_name = {spec.name: spec for spec in resolved}
    while current is not None:
        path.append(current)
        spec = by_name[current]
        # 在多个上游里取**最靠后**的那一个：`publish` 同时依赖 train（适配器）、
        # gate（门禁报告）与 package（模型卡），而主链显然要经过 gate 与 package，
        # 而不是从 train 直接跳到 publish。取 index 最大者即"离终点最近的上游"。
        upstream = [produced_by[item] for item in spec.requires if item in produced_by]
        current = max(upstream, key=lambda name: order[name]) if upstream else None
    return list(reversed(path))


__all__ = [
    "ARTIFACT_ADAPTER",
    "ARTIFACT_DATASET",
    "ARTIFACT_GATE_REPORT",
    "ARTIFACT_METRICS",
    "ARTIFACT_MODEL_CARD",
    "ARTIFACT_VERSION",
    "DEFAULT_ON_FAILURE_BLOCKING",
    "PIPELINE_STAGES",
    "STAGE_BLOCKED",
    "STAGE_EVALUATE",
    "STAGE_FAILED",
    "STAGE_GATE",
    "STAGE_INGEST",
    "STAGE_OK",
    "STAGE_PACKAGE",
    "STAGE_PUBLISH",
    "STAGE_SKIPPED",
    "STAGE_SPECS",
    "STAGE_STATUSES",
    "STAGE_TRAIN",
    "StageResult",
    "StageSpec",
    "critical_path",
    "stage_table",
    "validate_stage_order",
]
