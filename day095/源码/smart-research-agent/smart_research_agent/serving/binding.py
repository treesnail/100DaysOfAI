"""部署绑定：把"注册表里的哪一版"与"端点上跑的哪一份"绑在一起（M5-D11）.

day058 的注册表回答"我有哪些版本、哪一个是 stable"；day059 的流水线回答
"这一版够不够格发布"。**但这两件事都不保证线上跑的确实是它。**

"线上跑的不是我以为的那一版"有三条完全不同的发生路径，而且**每一条都不会报错**：

```text
1. adapter_dir 指到了旧目录    服务起得来、答得出来，只是答的是上一版
2. 基座被换过（8b → 14b）      适配器能挂上，输出"莫名其妙"（day052 已经写过一次）
3. 服务名/副本不一致           多副本部署里某个 Pod 还挂着旧配置
```

三条路径的共同点是：**证据都在，只是没有人把它们放在一起比对**。本模块就是
这一次比对——把三处"版本"（注册表记录 / 部署记录 / 端点自述）摊开逐项核对。

## 五条检查，以及每一条挡住的故障

| 检查 | 挡住的故障 | 缺省是否阻塞 |
|------|-----------|-------------|
| ``serving_name`` | 部署脚本改错了目标服务（改了 A 却以为改了 B） | 是 |
| ``base_model`` | 基座与适配器不匹配——**不报错，只是答错** | 是 |
| ``adapter_hash`` | 线上加载的是上一份适配器（唯一能证明"跑的是哪一份"的证据） | 是 |
| ``kind_shape`` | 形态与自述矛盾（合并模型却自述挂了适配器） | 是 |
| ``registry_head`` | 注册表 head 与绑定版本不同（有人提升了新版本却没部署，或反过来） | **否** |

最后一行是刻意的：**灰度发布期间"部署的不是 head"是完全正常的**（先部署
再切流量、或先给 5% 流量试跑）。把它设成阻塞会逼着团队为了过校验而先提升
阶段——那是"为了让门禁变绿而改数据"，比不做校验更糟。

## 与 day052 的 ``AdapterManifest`` 的分工

``AdapterManifest`` 回答"**这一份适配器**是什么、配哪个基座、怎么复现"（产物侧）。
本模块回答"**此刻服务的那个进程**用的是不是这一份"（运行侧）。
一份清单再好，也无法证明线上加载的是它——清单是文件，服务是进程。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.registry.record import ModelVersion, utc_now_iso
from smart_research_agent.serving.errors import ServingError
from smart_research_agent.serving.spec import KIND_ADAPTER, ServingSpec
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 部署记录的文件名（与 day052 的 ``adapter_manifest.json`` 分开：
#: 一个是产物清单、一个是部署记录，放在两个目录里才不会互相覆盖）。
BINDING_FILE = "deployment_binding.json"

#: 五条检查的名字（进报告，逐项可核对）。
CHECK_SERVING_NAME = "serving_name"
CHECK_BASE_MODEL = "base_model"
CHECK_ADAPTER_HASH = "adapter_hash"
CHECK_KIND_SHAPE = "kind_shape"
CHECK_REGISTRY_HEAD = "registry_head"

#: 自述哈希的比对长度：``sha256`` 的前 12 位（与 day052 清单的
#: ``short_hash``、day059 报告的适配器短哈希同一口径）。
SHORT_HASH_LENGTH = 12


@dataclass(frozen=True)
class ServedEndpoint:
    """推理端点**自述**的状态（"我现在加载的是什么"）.

    字段刻意与真实后端的自述能力对齐：

    - ``name``：Ollama ``GET /v1/models`` 的 id；
    - ``base_model`` / ``adapter_short_hash``：Ollama ``/api/show`` 的
      ``details.parent_model`` 与 ``modelfile`` 里的 ``ADAPTER`` 行，
      或 vLLM ``--served-model-name`` + 启动参数。

    三者都可能是空串——**空串表示"端点没有自述"**，而"没自述"在绝对判定里
    与"对不上"同等处理（缺证据不能判定一致）。这与 day059 门禁那一列
    ``when_missing`` 是同一条纪律。
    """

    name: str
    base_model: str = ""
    adapter_short_hash: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要（"没自述"要能看出来，而不是显示成空）."""
        base = self.base_model or "（未自述）"
        adapter = self.adapter_short_hash or "（未自述）"
        return f"{self.name} | 基座 {base} | 适配器 {adapter}"


@dataclass(frozen=True)
class ServingBinding:
    """一条部署记录：把"哪一版"钉到"哪个部署单元"上.

    ``serving_name`` 与 ``spec.name`` 是两个字段而不是一个，因为部署记录要能
    回答**两个不同的问题**：

    - ``spec.name``：我们**打算**把这一版部署到哪个服务名上；
    - ``serving_name``：**实际**部署时用的是哪个服务名（可能因为灰度而带了
      后缀，如 ``smart-research-qwen3-8b-canary``）。

    两者不同是可以接受的（灰度就是要不同），但**必须被记下来**——
    "灰度期间线上有两个服务名"这件事如果只存在于某个人的记忆里，
    下一次事故复盘就会从"线上到底有几个实例"开始吵。
    """

    spec: ServingSpec
    version: str
    version_key: str
    base_model: str
    adapter_sha256: str
    dataset_fingerprint: str
    serving_name: str
    deployed_at: str = field(default_factory=utc_now_iso)
    commit: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("version", self.version),
            ("version_key", self.version_key),
            ("base_model", self.base_model),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("serving_name", self.serving_name),
        ):
            if not str(value).strip():
                raise ServingError(f"部署记录缺少 {name}：缺了它就无法回答「线上跑的是哪一份」")
        # ``adapter`` 形态必须有适配器哈希，``base`` / ``merged`` 形态则**不该有**
        # 运行时适配器可指认——但它可以为空串（合并模型就是没有适配器）。
        if self.spec.kind == KIND_ADAPTER and not self.adapter_sha256.strip():
            raise ServingError(
                "adapter 形态的部署记录必须有 adapter_sha256："
                "没有它就无法证明线上挂的是哪一份适配器"
            )

    @property
    def short_adapter(self) -> str:
        """适配器内容哈希的前 12 位（端点自述与它比对）."""
        return self.adapter_sha256[:SHORT_HASH_LENGTH]

    @property
    def is_canary(self) -> bool:
        """服务名是否与部署单元名不同（灰度部署的显式标记）."""
        return self.serving_name != self.spec.name

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生量）."""
        return {
            "version": self.version,
            "version_key": self.version_key,
            "base_model": self.base_model,
            "adapter_sha256": self.adapter_sha256,
            "short_adapter": self.short_adapter,
            "dataset_fingerprint": self.dataset_fingerprint,
            "serving_name": self.serving_name,
            "is_canary": self.is_canary,
            "deployed_at": self.deployed_at,
            "commit": self.commit,
            "notes": self.notes,
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ServingBinding:
        """从 ``to_dict`` 的产物还原（``spec`` 缺失视为非法记录）."""
        spec_payload = payload.get("spec")
        if not isinstance(spec_payload, dict):
            raise ServingError("部署记录里没有 spec：没有它就无法还原部署形态")
        known = {item.name for item in ServingSpec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        spec = ServingSpec(**{k: v for k, v in spec_payload.items() if k in known})
        return cls(
            spec=spec,
            version=str(payload["version"]),
            version_key=str(payload["version_key"]),
            base_model=str(payload["base_model"]),
            adapter_sha256=str(payload.get("adapter_sha256", "")),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            serving_name=str(payload["serving_name"]),
            deployed_at=str(payload.get("deployed_at", "")) or utc_now_iso(),
            commit=str(payload.get("commit", "")),
            notes=str(payload.get("notes", "")),
        )

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        canary = "（灰度）" if self.is_canary else ""
        adapter = f"sha256:{self.short_adapter}" if self.short_adapter else "（无适配器）"
        return (
            f"v{self.version} → {self.serving_name}{canary} | 键 {self.version_key} | "
            f"基座 {self.base_model} | {adapter} | 部署于 {self.deployed_at}"
        )


@dataclass(frozen=True)
class BindingCheck:
    """一条一致性检查的结果：实际值、期望值、结论、理由.

    三个字段缺一不可（与 day058 的 ``CheckResult``、day059 的 ``GateCheck``
    同一形状与同一理由）：只有 ``passed`` 的报告在运维上不可用——
    看到"没通过"之后唯一的动作是去读代码。
    """

    name: str
    passed: bool
    actual: str
    expected: str
    reason: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        flag = "阻塞" if self.blocking else "告警"
        return (
            f"[{'一致' if self.passed else '不一致'}/{flag}] {self.name}: "
            f"实际 {self.actual} / 期望 {self.expected} —— {self.reason}"
        )


@dataclass(frozen=True)
class BindingVerification:
    """三处版本比对的结果.

    ``passed`` 只看**阻塞项**。非阻塞项落进 ``warnings``：
    "部署的不是注册表 head"这件事必须被看见，但不该拦住一次灰度。
    """

    passed: bool
    checks: tuple[BindingCheck, ...] = ()
    version: str = ""
    serving_name: str = ""
    notes: str = ""

    @property
    def blocking_failures(self) -> list[BindingCheck]:
        """阻塞项里没通过的（决定 ``passed`` 的就是它们）."""
        return [item for item in self.checks if item.blocking and not item.passed]

    @property
    def warnings(self) -> list[BindingCheck]:
        """非阻塞项里没通过的：记录但不拦部署（当前只有 head 不一致落在这里）."""
        return [item for item in self.checks if not item.blocking and not item.passed]

    @property
    def consistent(self) -> bool:
        """是否**全部**一致（含非阻塞项）——与 ``passed`` 分开命名.

        "可以放行"与"完全一致"是两件事：一个 ``passed=True`` 但
        ``consistent=False`` 的部署是灰度期间最正常的状态，
        把两者合成一个布尔量，就再也表达不出"我放行了，但我知道它不是 head"。
        """
        return not self.blocking_failures and not self.warnings

    def check(self, name: str) -> BindingCheck:
        """按名字取一条检查；不存在抛 ``ServingError``."""
        for item in self.checks:
            if item.name == name:
                return item
        raise ServingError(f"报告里没有检查项 {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "passed": self.passed,
            "consistent": self.consistent,
            "version": self.version,
            "serving_name": self.serving_name,
            "notes": self.notes,
            "blocking_failures": [item.name for item in self.blocking_failures],
            "warnings": [item.name for item in self.warnings],
            "checks": [item.to_dict() for item in self.checks],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"部署绑定 {'一致' if self.passed else '不一致'} | "
            f"{sum(1 for item in self.checks if item.passed)}/{len(self.checks)} 项通过 | "
            f"阻塞失败 {[item.name for item in self.blocking_failures] or '无'}"
        )

    def render_markdown(self) -> str:
        """把报告渲染成 markdown（可直接贴进变更单与事故工单）."""
        lines = [
            f"# 部署绑定校验：{'通过' if self.passed else '不通过'}",
            "",
            f"- 版本：v{self.version}",
            f"- 线上服务名：`{self.serving_name}`",
            f"- 阻塞失败：{', '.join(item.name for item in self.blocking_failures) or '（无）'}",
            f"- 告警：{', '.join(item.name for item in self.warnings) or '（无）'}",
            f"- 备注：{self.notes or '（无）'}",
            "",
            "| 检查 | 实际 | 期望 | 一致 | 阻塞 | 理由 |",
            "|------|------|------|------|------|------|",
        ]
        lines.extend(
            f"| `{item.name}` | {item.actual} | {item.expected} | "
            f"{'是' if item.passed else '否'} | {'是' if item.blocking else '否'} | {item.reason} |"
            for item in self.checks
        )
        lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True)
class BindingPolicy:
    """绑定校验策略：两个**开关**，不是一个阈值.

    ``require_self_report`` 与 ``require_registry_head`` 分开，因为它们是
    两个不同的问题："端点愿不愿意自述"（后端能力）与"能不能部署非 head 版本"
    （发布流程）。用一个 ``strict: bool`` 表达会同时失去两者。
    """

    require_self_report: bool = True
    require_registry_head: bool = False

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)


def _self_report_check(
    *,
    name: str,
    actual: str,
    expected: str,
    policy: BindingPolicy,
    missing_reason: str,
    mismatch_reason: str,
    match_reason: str,
) -> BindingCheck:
    """构造一条"自述 vs 期望"的检查（三条分支的理由各不相同）.

    抽成函数是因为五条检查里有三条是同一形状，而**它们的差别只在理由文本**——
    这正是外面那一层"报告里每一行都要能读懂"的代价，值得付。
    """
    if not actual.strip():
        if policy.require_self_report:
            return BindingCheck(
                name=name,
                passed=False,
                actual="（未自述）",
                expected=expected,
                reason=missing_reason,
            )
        return BindingCheck(
            name=name,
            passed=False,
            actual="（未自述）",
            expected=expected,
            reason=f"策略关闭了自述要求：这是一次有意的关闭，不是「一致」（{missing_reason}）",
            blocking=False,
        )
    if actual.strip() != expected.strip():
        return BindingCheck(
            name=name,
            passed=False,
            actual=actual,
            expected=expected,
            reason=mismatch_reason,
        )
    return BindingCheck(
        name=name,
        passed=True,
        actual=actual,
        expected=expected,
        reason=match_reason,
    )


def bind_version(
    spec: ServingSpec,
    version: ModelVersion,
    *,
    serving_name: str = "",
    deployed_at: str = "",
    commit: str = "",
    notes: str = "",
) -> ServingBinding:
    """把注册表里的一条版本绑成一条部署记录（部署的**前置条件**都在这里拦）.

    三个前置条件，每一个都对应一种真实事故：

    1. **产物齐全**（``version.is_deployable()``）——复用 day058 的判定，
       不重新实现一份"什么叫可部署"；
    2. **基座对齐**：``version.base_model`` 必须等于 ``spec.base_model``。
       基座不匹配的适配器**能挂上也会答错**（day052 的原话），因此它在
       构造部署记录时就被拦住，而不是等到推理结果不对时才被发现；
    3. **形态与证据对齐**：``adapter`` 形态的部署记录必须有适配器哈希。

    第 3 条**没有**在这里写成一个额外的判定：``VersionTriple`` 已经在入口
    保证 ``adapter_sha256`` 非空（长度下限 16 位十六进制），
    再判一次就是一条恒为真的检查——而 day059 已经记过这笔账：
    **一条永远为真的断言不是保护，而是噪声**，它会让人以为存在某种
    未被覆盖的情形，从而在别处写出多余的兜底代码。``ServingBinding``
    里那道判定守的是**另一条路径**（从 ``from_dict`` 还原的记录）。
    """
    if not version.is_deployable():
        missing = ", ".join(version.missing_artifacts())
        raise ServingError(
            f"v{version.version} 的产物不完整（缺少 {missing}）："
            "不可部署的版本不该进入部署记录"
        )
    if version.base_model != spec.base_model:
        raise ServingError(
            f"基座不匹配：版本记录的基座是 {version.base_model}，"
            f"部署单元声明的是 {spec.base_model}——"
            "基座不匹配的适配器能挂上也会答错，而且不会报错"
        )
    binding = ServingBinding(
        spec=spec,
        version=version.version,
        version_key=version.version_key,
        base_model=version.base_model,
        adapter_sha256=version.adapter_sha256,
        dataset_fingerprint=version.dataset_fingerprint,
        serving_name=serving_name or spec.name,
        deployed_at=deployed_at or utc_now_iso(),
        commit=commit,
        notes=notes,
    )
    logger.info("部署绑定：%s", binding.summary_line())
    return binding


def verify_binding(
    binding: ServingBinding,
    served: ServedEndpoint,
    *,
    head: ModelVersion | None = None,
    policy: BindingPolicy | None = None,
) -> BindingVerification:
    """把"部署记录"与"端点自述"（可选地加上"注册表 head"）逐项比对.

    ``head=None`` 表示调用方没有提供注册表状态，此时 ``registry_head``
    这一项记为"未检查"而不是"通过"——**缺证据不能算一致**，
    但它也**不阻塞**（本次部署没有声明要与 head 比对）。
    """
    resolved = policy or BindingPolicy()
    checks: list[BindingCheck] = []

    # ---- 检查 1：服务名（部署脚本改错了目标服务）
    checks.append(
        _self_report_check(
            name=CHECK_SERVING_NAME,
            actual=served.name,
            expected=binding.serving_name,
            policy=resolved,
            missing_reason="端点没有自述服务名：无法确认这次部署落在了目标服务上",
            mismatch_reason=(
                "服务名不一致：部署记录写的是这个名，而端点自述的是另一个——"
                "多副本部署里最常见的形态是「某一个 Pod 还是旧配置」"
            ),
            match_reason="部署记录与端点自述的服务名一致",
        )
    )

    # ---- 检查 2：基座（不报错、只答错的那一类）
    checks.append(
        _self_report_check(
            name=CHECK_BASE_MODEL,
            actual=served.base_model,
            expected=binding.base_model,
            policy=resolved,
            missing_reason="端点没有自述基座：基座不匹配不会报错，只会给出莫名其妙的结果",
            mismatch_reason=(
                "基座不一致：适配器挂在了别的基座上——它能加载成功，"
                "但输出与训练时的行为不再相关"
            ),
            match_reason="部署记录与端点自述的基座一致",
        )
    )

    # ---- 检查 3：适配器哈希（唯一能证明"跑的是哪一份"的证据）
    if binding.short_adapter:
        checks.append(
            _self_report_check(
                name=CHECK_ADAPTER_HASH,
                actual=served.adapter_short_hash,
                expected=binding.short_adapter,
                policy=resolved,
                missing_reason=(
                    "端点没有自述适配器哈希：这是唯一能证明"
                    "「线上跑的是哪一份」的证据，缺了它就只能相信部署脚本"
                ),
                mismatch_reason=(
                    "适配器哈希不一致：线上挂的是另一份适配器——"
                    "多数情况是 adapter_dir 指到了旧目录，而服务起得来、答得出来"
                ),
                match_reason="部署记录与端点自述的适配器短哈希一致",
            )
        )
    else:
        checks.append(
            BindingCheck(
                name=CHECK_ADAPTER_HASH,
                passed=True,
                actual="（本形态无适配器）",
                expected="（本形态无适配器）",
                reason=(
                    f"{binding.spec.kind} 形态不依赖运行时适配器，"
                    "本项按「不适用」通过（不是「没检查」）"
                ),
                blocking=False,
            )
        )

    # ---- 检查 4：形态与自述是否自洽
    #
    # 这一项检的是"形态"与"端点自述"之间的一致：合并形态不该自述适配器
    # （自述了说明端点加载的其实是别的目录）；适配器形态则必须自述。
    shape_ok: bool
    if binding.spec.uses_adapter:
        shape_ok = bool(served.adapter_short_hash.strip())
        shape_reason = (
            "adapter 形态且端点自述了适配器哈希"
            if shape_ok
            else "adapter 形态但端点没有自述适配器哈希：形态与自述矛盾"
        )
    else:
        shape_ok = not served.adapter_short_hash.strip()
        shape_reason = (
            f"{binding.spec.kind} 形态且端点未自述适配器，形态自洽"
            if shape_ok
            else f"{binding.spec.kind} 形态却自述了适配器哈希：端点加载的其实是别的目录"
        )
    checks.append(
        BindingCheck(
            name=CHECK_KIND_SHAPE,
            passed=shape_ok,
            actual=served.adapter_short_hash or "（未自述）",
            expected="有适配器" if binding.spec.uses_adapter else "无适配器",
            reason=shape_reason,
        )
    )

    # ---- 检查 5：注册表 head（灰度期间允许不一致，因此**不阻塞**）
    if head is None:
        checks.append(
            BindingCheck(
                name=CHECK_REGISTRY_HEAD,
                passed=False,
                actual="未检查",
                expected="与注册表 head 比对",
                reason="调用方没有提供注册表状态：本项记为「未检查」，缺证据不能算一致",
                blocking=False,
            )
        )
    else:
        same = head.version_key == binding.version_key
        checks.append(
            BindingCheck(
                name=CHECK_REGISTRY_HEAD,
                passed=same,
                actual=f"head=v{head.version}（键 {head.version_key}）",
                expected=f"绑定版本 v{binding.version}（键 {binding.version_key}）",
                reason=(
                    "绑定版本就是注册表当前的 head"
                    if same
                    else "绑定版本不是注册表 head："
                    "有人在它之后提升过新版本（或这次部署走在提升之前）——"
                    "灰度期间这是正常的，但它必须被看见"
                ),
                blocking=resolved.require_registry_head,
            )
        )

    verification = BindingVerification(
        passed=not [item for item in checks if item.blocking and not item.passed],
        checks=tuple(checks),
        version=binding.version,
        serving_name=binding.serving_name,
    )
    logger.info("部署绑定校验：%s", verification.summary_line())
    return verification


def binding_table(policy: BindingPolicy | None = None) -> list[dict[str, Any]]:
    """五条一致性检查的对照表（API 的自我描述端点与文档同源）.

    表里的 ``failure_mode`` 一列是这张表的重点：**每一条检查都对应一种
    "不会报错但会答错"的故障**。把它们写出来，是为了让"为什么要做这五项校验"
    变成一个可以被复核的决定，而不是一次形式主义的打卡。
    """
    resolved = policy or BindingPolicy()
    return [
        {
            "name": CHECK_SERVING_NAME,
            "blocking": True,
            "failure_mode": "部署脚本改错了目标服务（改了 A 却以为改了 B）",
            "evidence": "服务端 /v1/models 返回的 id",
        },
        {
            "name": CHECK_BASE_MODEL,
            "blocking": True,
            "failure_mode": "基座与适配器不匹配：能挂上，但输出与训练行为无关",
            "evidence": "端点自述的基座（Ollama /api/show 的 details.parent_model）",
        },
        {
            "name": CHECK_ADAPTER_HASH,
            "blocking": True,
            "failure_mode": "线上加载的是上一份适配器（adapter_dir 指到了旧目录）",
            "evidence": "端点自述的适配器短哈希（12 位）",
        },
        {
            "name": CHECK_KIND_SHAPE,
            "blocking": True,
            "failure_mode": "形态与自述矛盾：合并形态却自述挂了适配器",
            "evidence": "形态声明 vs 端点自述里有没有适配器",
        },
        {
            "name": CHECK_REGISTRY_HEAD,
            "blocking": bool(resolved.require_registry_head),
            "failure_mode": "绑定版本不是注册表 head：有人提升了新版本却没部署（或反过来）",
            "evidence": "注册表 head 的版本键",
        },
    ]


def write_binding(directory: str | Path, binding: ServingBinding) -> Path:
    """把部署记录写进目录（``deployment_binding.json``）.

    落盘的理由与 day052 把清单写进适配器目录是同一条：**产物与记录一起走**。
    只把"线上跑的是哪一版"记在 CI 日志里，会让三个月后的追溯从"翻日志"开始。
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / BINDING_FILE
    path.write_text(
        json.dumps(binding.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_binding(directory: str | Path) -> ServingBinding:
    """读回部署记录；文件缺失抛 ``ServingError``."""
    path = Path(directory) / BINDING_FILE
    if not path.exists():
        raise ServingError(f"找不到部署记录：{path}")
    return ServingBinding.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "BINDING_FILE",
    "CHECK_ADAPTER_HASH",
    "CHECK_BASE_MODEL",
    "CHECK_KIND_SHAPE",
    "CHECK_REGISTRY_HEAD",
    "CHECK_SERVING_NAME",
    "SHORT_HASH_LENGTH",
    "BindingCheck",
    "BindingPolicy",
    "BindingVerification",
    "ServedEndpoint",
    "ServingBinding",
    "bind_version",
    "binding_table",
    "read_binding",
    "verify_binding",
    "write_binding",
]
