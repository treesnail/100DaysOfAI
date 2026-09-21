"""版本注册表：一条追加写的事件日志 + 内存折叠（M5-D9）.

``registry`` 只有一个持久化产物：``versions.jsonl``。它存的**不是**"版本列表"，
而是**事件流**：

```text
{"type": "register", "version": "1.0.0", "version_key": "…", …}
{"type": "stage",    "version": "1.0.0", "stage": "stable", "reason": "…"}
{"type": "register", "version": "1.0.1", "version_key": "…", …}
```

读的时候把事件折叠成当前状态（``register`` 建记录、``stage`` 改阶段）。

### 为什么是事件流而不是"重写整个列表"

如果状态变更靠"读全量 → 改一条 → 写回全量"，会同时引进三个问题：

1. **写路径从 O(1) 变成 O(N)**，而版本数量只会单调增长；
2. **丢失了"什么时候变的、为什么变"**——``stable`` 记录里那个字段只能说明
   "现在是 stable"，说明不了"它是 3 天前因为候选胜出被提升的，还是刚刚
   因为回滚被设回来的"。而这两种情况在运维上完全不同；
3. **一次崩溃就可能截断整个文件**：day027 的 ``quality_logger`` 选 jsonl 的
   理由正是"单行损坏不传染"。重写全量的写入方式把这条保护还回去了。

代价是**读的时候要多做一次折叠**，以及索引文件不能只靠"看最后一行"来读。
这两条代价都很小，而且都能被测试钉住（``test_store_folds_events``）。

### 折叠的确定性

折叠必须**对事件顺序敏感、对重复登记幂等**：

- ``register`` 遇到已经存在的版本号：内容一致 → 幂等返回（CI 重跑同一个任务
  正是这种情形）；内容不一致 → 抛 ``RegistryError``；
- ``stage`` 遇到相同的目标阶段：抛错而不是静默通过（"重复流转"说明有两条
  代码路径在同时改状态，这是要查的信号）；
- 未知事件类型：抛错。**跳过不认识的事件会丢状态**，而丢状态的表现是
  "``head()`` 返回了 None"，去查的人根本不会想到是索引里有一行没读懂。

### 幂等 ≠ 宽松

``register`` 的幂等只覆盖"三元组键与版本号都相同"这一种情形。把幂等放宽到
"版本号相同就覆盖"是危险的：版本号是**人写的叙事**，两个人给两份不同的产物
写同一个 ``1.2.0`` 是常有的事，静默覆盖会让索引里少掉一份真实产物。
因此"版本号相同但键不同"必须报冲突，且冲突信息里要给出**具体哪些字段不同**
（``VersionConflict.fields``）——只报 "conflict" 的话，运维唯一的动作就是去翻文件。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from smart_research_agent.registry.errors import RegistryError
from smart_research_agent.registry.record import (
    STAGES,
    STAGE_ARCHIVED,
    STAGE_CANDIDATE,
    STAGE_STABLE,
    ModelVersion,
    utc_now_iso,
)
from smart_research_agent.registry.version import (
    VersionConflict,
    VersionTriple,
    bump_kind_for,
    next_version,
    semver_sort_key,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 索引文件名（相对 ``registry_dir``）。与 ``sft`` 的 ``metrics.json``、
#: ``domain_data`` 的 ``manifest.json`` 一样，产物名固定，脚本才能照着找。
INDEX_FILENAME = "versions.jsonl"

#: 事件类型（只有两种：登记与阶段流转。**没有"删除"事件**——
#: 版本记录的删除会破坏审计，下线走 ``archived``）。
EVENT_REGISTER = "register"
EVENT_STAGE = "stage"

#: ``register`` 事件的字段（读回时用它做必填校验）.
REGISTER_REQUIRED_FIELDS: tuple[str, ...] = ("version", "base_model", "adapter_sha256")

#: 缺省保留的候选版本数（超出的最旧候选被归档）。**这个数字不限制 stable
#: 版本的数量**——stable 是回滚目标，见 ``prune`` 的说明。
DEFAULT_KEEP_VERSIONS = 20


class ModelRegistry:
    """模型版本注册表（可选落盘；``path=None`` 时纯内存）.

    用法::

        registry = ModelRegistry("outputs/registry/versions.jsonl")
        registry.register(record)                       # 登记为 candidate
        registry.set_stage("1.0.0", STAGE_STABLE)       # 提升为生产版本
        registry.head()                                 # 当前生产版本

    ``path=None`` 的纯内存模式不是"降级"：测试、API 端点与"先算计划再落盘"
    的调用方都只需要折叠好的状态。**把"要不要落盘"做成构造参数**，
    比给每个方法加一个 ``persist=False`` 开关干净得多。
    """

    def __init__(
        self, path: str | Path | None = None, *, keep_versions: int = DEFAULT_KEEP_VERSIONS
    ):
        if keep_versions < 1:
            raise RegistryError(f"keep_versions 必须为正整数，收到 {keep_versions}")
        self.path: Path | None = Path(path) if path is not None else None
        self.keep_versions = keep_versions
        #: 版本号 → 记录（折叠后的当前状态）
        self._records: dict[str, ModelVersion] = {}
        #: 版本键 → 版本号（保证"一个产物一个版本号"）
        self._by_key: dict[str, str] = {}
        #: 已折叠的事件（保留下来供审计与诊断打印）
        self._events: list[dict[str, Any]] = []
        if self.path is not None:
            self.reload()

    # ------------------------------------------------------------------ 读盘
    @property
    def directory(self) -> Path | None:
        """索引所在目录（``path=None`` 时为 ``None``）."""
        return None if self.path is None else self.path.parent

    def reload(self) -> None:
        """清空内存状态并重新从索引文件折叠一次（幂等）."""
        self._records.clear()
        self._by_key.clear()
        self._events.clear()
        if self.path is None or not self.path.exists():
            return
        for lineno, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RegistryError(
                    f"{self.path} 第 {lineno} 行不是合法 JSON：{exc}"
                ) from exc
            if not isinstance(event, dict):
                raise RegistryError(f"{self.path} 第 {lineno} 行不是 JSON 对象")
            self._fold(event, lineno)
        logger.info(
            "版本注册表已加载：%s（%d 条事件 / %d 个版本）",
            self.path,
            len(self._events),
            len(self._records),
        )

    def _fold(self, event: dict[str, Any], lineno: int = 0) -> None:
        """把一个事件折进内存状态（读盘与写入共用同一条路径）."""
        kind = event.get("type")
        if kind == EVENT_REGISTER:
            missing = [name for name in REGISTER_REQUIRED_FIELDS if name not in event]
            if missing:
                raise RegistryError(
                    f"第 {lineno or '?'} 个 register 事件缺少字段 {', '.join(missing)}"
                )
            record = ModelVersion.from_dict(event)
            self._insert(record)
        elif kind == EVENT_STAGE:
            version = str(event.get("version", ""))
            record = self._records.get(version)
            if record is None:
                raise RegistryError(
                    f"第 {lineno or '?'} 个 stage 事件指向不存在的版本 {version!r}"
                )
            self._records[version] = record.with_stage(
                str(event.get("stage", "")), note=str(event.get("reason", ""))
            )
        else:
            raise RegistryError(
                f"第 {lineno or '?'} 行是未知事件类型 {kind!r}"
                "（跳过不认识的事件会丢状态，而丢状态的表现只是 head() 变 None）"
            )
        self._events.append(event)

    def _insert(self, record: ModelVersion) -> ModelVersion:
        """把一条记录放进内存表（幂等 / 冲突判定都在这里）."""
        key = record.version_key
        owner = self._by_key.get(key)
        if owner is not None:
            existing = self._records[owner]
            if existing.version == record.version:
                # 同键同版本号：CI 重跑同一个任务的情形，幂等返回已有记录
                return existing
            raise RegistryError(self._conflict(existing, record).summary_line())
        clashing = self._records.get(record.version)
        if clashing is not None:
            raise RegistryError(
                f"版本号 {record.version} 已属于键 {clashing.version_key}"
                f"（{clashing.triple.describe()}），不能再给键 {key} 使用；"
                "版本号是人写的叙事，两份不同产物写同一个号是常有的事，"
                "静默覆盖会让索引少掉一份真实产物"
            )
        if record.parent_version and record.parent_version not in self._records:
            raise RegistryError(
                f"{record.version} 声明的父版本 {record.parent_version} 不存在："
                "悬空的父指针会让回滚在走链时突然断掉"
            )
        self._records[record.version] = record
        self._by_key[key] = record.version
        return record

    @staticmethod
    def _conflict(existing: ModelVersion, incoming: ModelVersion) -> VersionConflict:
        """构造冲突报告：逐字段给出"已有 vs 传入"的差异."""
        pairs = {
            "version": (existing.version, incoming.version),
            "base_model": (existing.base_model, incoming.base_model),
            "adapter_sha256": (existing.adapter_sha256, incoming.adapter_sha256),
            "dataset_fingerprint": (
                existing.dataset_fingerprint,
                incoming.dataset_fingerprint,
            ),
        }
        changed = {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}
        return VersionConflict(
            key=existing.version_key,
            existing_version=existing.version,
            incoming_version=incoming.version,
            fields=changed,
        )

    def _append(self, event: dict[str, Any]) -> None:
        """把事件追加到索引文件，并同步进内存的事件镜像.

        **两条路径（读盘折叠 / 写入追加）共用 ``self._events``**：写盘时不更新
        镜像的话，一个正在写入的注册表 ``events()`` 会返回空列表——而这正是
        "审计流"唯一被消费的地方（``/registry/*`` 的报告与 demo 都读它）。
        这个不一致在测试里第一次跑就暴露了：同一份操作序列，**重载后的
        ``events()`` 有 3 条、写入方的 ``events()`` 有 0 条**。
        """
        if self.path is not None:
            assert self.directory is not None  # noqa: S101 - path 非空时 directory 必然非空
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._events.append(event)

    # ------------------------------------------------------------------ 写
    def register(self, record: ModelVersion, *, reason: str = "") -> ModelVersion:
        """登记一条版本记录，返回**折叠后**的记录（可能已存在）.

        三件事按顺序发生：内存折叠（含幂等与冲突判定）→ 落盘事件 → 返回。
        **先折叠再落盘**的理由：折叠会抛异常，抛出时不该在文件里留下
        一行"事件已经写了、但状态没建立"的记录。

        ``record.stage`` 允许是任意合法阶段（不只 ``candidate``）：``register``
        登记的是**状态快照**，"从落盘的索引重建注册表"正需要这一点。
        阶段**流转**是另一件事，只有 ``set_stage`` 会产生 ``stage`` 事件——
        两条路径的差别在于"这是初始状态"还是"发生过一次变更"，
        而后者要进审计。
        """
        existing = self._insert(record)
        if existing is not record:
            # 幂等命中：不重复写事件（重复写会让索引膨胀，而语义上什么都没发生）
            logger.info("版本 %s 已存在且内容一致，跳过重复登记", record.version)
            return existing
        self._append({"type": EVENT_REGISTER, **record.to_dict(), "reason": reason})
        logger.info("已登记版本 %s", record.summary_line())
        return record

    def set_stage(
        self, version: str, stage: str, *, reason: str = "", record: ModelVersion | None = None
    ) -> ModelVersion:
        """把某个版本的阶段改写为新值，返回改写后的记录.

        ``record`` 参数用于"带上评估结果一起提升阶段"的场景：调用方
        刚跑完评估、手上有新指标，提升阶段时顺手把它们并进记录，
        可以少一次写入。不传则原样保留旧指标。
        """
        current = self.get(version)
        if record is not None:
            if record.version != current.version or record.version_key != current.version_key:
                raise RegistryError(
                    f"传入的 record（v{record.version} / 键 {record.version_key}）"
                    f"与目标版本（v{current.version} / 键 {current.version_key}）不一致："
                    "提升阶段时顺手带上新指标是允许的，但换掉版本本身不行"
                )
            self._records[current.version] = record
            current = record
        updated = current.with_stage(stage, note=reason)
        self._records[current.version] = updated
        self._append(
            {
                "type": EVENT_STAGE,
                "version": updated.version,
                "stage": stage,
                "reason": reason,
                "at": utc_now_iso(),
            }
        )
        logger.info("版本 %s 阶段流转到 %s（%s）", updated.version, stage, reason or "未说明")
        return updated

    def prune(self, *, keep: int | None = None) -> list[str]:
        """归档最旧的候选版本，返回被归档的版本号列表.

        两条保护，都是"不这么做就会自断退路"的：

        1. **``stable`` 永不归档**。stable 是回滚目标，归档一个 stable 版本
           等于删掉一条退路。要下线它，请显式走 ``rolled_back`` 或 ``archived``
           ——那是一次有理由的运维动作，而不是磁盘清理的副作用；
        2. **``head()`` 永不归档**（它必然是 stable，这一条是第一条的推论，
           但仍然显式写在代码里：``prune`` 的调用方不该需要自己记住这个推论）。

        只清理 ``candidate`` 的理由同样具体：候选是"跑完还没通过门禁"的中间态，
        它们数量增长最快、生命周期最短，而且**每一个都可以由三元组重新算出键**
        ——真需要它们回来，重新登记一次即可，丢失的信息只有"当时分数多少"。
        """
        limit = self.keep_versions if keep is None else keep
        if limit < 1:
            raise RegistryError(f"prune 的 keep 必须为正整数，收到 {limit}")
        candidates = sorted(
            (item for item in self._records.values() if item.stage == STAGE_CANDIDATE),
            key=lambda item: item.version_sort_key,
        )
        stale = candidates[: max(0, len(candidates) - limit)]
        archived: list[str] = []
        for item in stale:
            self.set_stage(item.version, STAGE_ARCHIVED, reason="prune：候选版本超出保留上限")
            archived.append(item.version)
        return archived

    # ------------------------------------------------------------------ 读
    def get(self, version: str) -> ModelVersion:
        """按版本号或版本键取记录；不存在抛 ``RegistryError``.

        同时接受两种标识是刻意的：**版本键来自机器（CI 输出**、
        ``AdapterManifest``），**版本号来自人**（口头、issue、变更单）。
        只支持一种，就总有一方要自己去转换，而转换代码会散落在各处。
        """
        if version in self._records:
            return self._records[version]
        owner = self._by_key.get(version)
        if owner is not None:
            return self._records[owner]
        raise RegistryError(f"版本 {version!r} 不存在（既不是版本号也不是版本键）")

    def find(self, version: str) -> ModelVersion | None:
        """``get`` 的不抛异常版本（不存在返回 ``None``）."""
        try:
            return self.get(version)
        except RegistryError:
            return None

    def versions(self, *, stage: str | None = None) -> list[ModelVersion]:
        """按版本号升序列出记录（可按阶段过滤；空集合法，不报错）."""
        if stage is not None and stage not in STAGES:
            raise RegistryError(f"未知阶段 {stage!r}，可选 {', '.join(STAGES)}")
        items = [item for item in self._records.values() if stage is None or item.stage == stage]
        return sorted(items, key=lambda item: item.version_sort_key)

    def latest(self, *, stage: str | None = None) -> ModelVersion | None:
        """最新的记录（可按阶段过滤）；空集合返回 ``None``."""
        items = self.versions(stage=stage)
        return items[-1] if items else None

    def head(self) -> ModelVersion | None:
        """当前生产版本 = 最新的 ``stable``；没有 stable 时返回 ``None``.

        返回 ``None`` 而不是"退回 candidate"是刻意的：**"还没有生产版本"
        与"生产版本是某个候选"是两件事**。前者对应首次上线（部署脚本应当
        直接报错、要求先完成一次人工门禁），后者会静默把未验证的候选推上线。
        """
        return self.latest(stage=STAGE_STABLE)

    def lineage(self, version: str, *, limit: int | None = None) -> list[ModelVersion]:
        """版本链：自身 → 父 → 祖父 ……（按父指针走，遇到缺失即停）.

        遇到不存在的父版本**不抛异常**（索引里可能有更早的、已被清理的祖先），
        但会在它之前停下——一条"半截的链"是真实存在的状态，
        把它当成错误会让回滚在历史较长的仓库里彻底不可用。
        """
        chain: list[ModelVersion] = []
        current = self.get(version)
        seen: set[str] = set()
        while current is not None:
            if current.version in seen:
                # 环检测：手工登记时把父指向自己的后代是可能的，
                # 而带环的链会让 while 变成死循环——那是最糟的失败方式（挂住而不是报错）
                raise RegistryError(
                    f"版本链出现环：{current.version} 已在链上（{', '.join(seen)}）"
                )
            seen.add(current.version)
            chain.append(current)
            if limit is not None and len(chain) >= limit:
                break
            current = self._records.get(current.parent_version) if current.parent_version else None
        return chain

    def stable_ancestors(self, version: str) -> list[ModelVersion]:
        """版本链上（不含自身）的 ``stable`` 版本，**从近到远**排序.

        这是回滚的直接依据：回滚要退回的必然是一个**曾经稳定过**的版本，
        而不是"任意一个祖先"。祖先里可能夹着一堆 candidate（每次重训都会
        产生一个候选），把它们当回滚目标等于回滚到一个从未上线过的状态。
        """
        chain = self.lineage(version, limit=None)
        return [item for item in chain[1:] if item.stage == STAGE_STABLE]

    def counts(self) -> dict[str, int]:
        """各阶段的版本数（四个阶段键恒存在，另加 ``total``）.

        四个键**恒存在且缺省为 0**：报告里"rolled_back: 0"与"没有这个键"
        读起来完全不同，前者是"回滚过 0 次"，后者是"没人统计过回滚"。
        """
        result: dict[str, int] = {stage: 0 for stage in STAGES}
        for item in self._records.values():
            result[item.stage] += 1
        result["total"] = len(self._records)
        return result

    def events(self) -> list[dict[str, Any]]:
        """已折叠事件的副本（审计与 demo 打印用；返回副本避免外部改写）."""
        return [dict(event) for event in self._events]

    def propose_version(
        self, *, base_model: str, dataset_fingerprint: str
    ) -> tuple[str, str]:
        """算出"下一个版本号"与它对应的递增位，返回 ``(版本号, 递增位)``.

        递增位由**相对最新版本变化的那一项**推出（见 ``version.bump_kind_for``），
        因此调用方不需要自己判断"这次是换基座还是重训"——那个判断需要有
        "上一版的三元组"这个信息，而它就在注册表里。
        """
        previous = self.latest()
        kind = bump_kind_for(
            None if previous is None else previous.triple,
            base_model=base_model,
            dataset_fingerprint=dataset_fingerprint,
        )
        existing = [item.version for item in self.versions()]
        return next_version(existing, kind=kind), kind

    def next_version_for(
        self, *, base_model: str, adapter_sha256: str, dataset_fingerprint: str
    ) -> str:
        """便捷方法：直接给出一个 ``ModelVersion`` 该用的版本号.

        如果这个三元组已经在表里，就返回它**已有的**版本号（而不是新算一个）
        ——重复登记同一个产物不该消耗一个版本号。
        """
        key = VersionTriple(
            base_model=base_model,
            adapter_sha256=adapter_sha256,
            dataset_fingerprint=dataset_fingerprint,
        ).key
        owner = self._by_key.get(key)
        if owner is not None:
            return owner
        version, _ = self.propose_version(
            base_model=base_model, dataset_fingerprint=dataset_fingerprint
        )
        return version

    # ------------------------------------------------------------------ 投影
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ModelVersion]:
        return iter(self.versions())

    def __contains__(self, version: object) -> bool:
        return isinstance(version, str) and self.find(version) is not None

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（不含事件流的全量正文）."""
        head = self.head()
        return {
            "path": None if self.path is None else str(self.path),
            "total": len(self._records),
            "counts": self.counts(),
            "head": None if head is None else head.version,
            "head_key": None if head is None else head.version_key,
            "events": len(self._events),
            "versions": [item.to_dict() for item in self.versions()],
        }

    def render_markdown(self) -> str:
        """把注册表渲染成 markdown 表格（报告与 demo 直接打印）."""
        head = self.head()
        lines = [
            "# 模型版本注册表",
            "",
            f"- 版本总数：**{len(self._records)}**",
            f"- 阶段分布：{self.counts()}",
            f"- 当前生产版本：{('v' + head.version + '（键 ' + head.version_key + '）') if head else '**尚无 stable 版本**'}",
            f"- 事件条数：{len(self._events)}",
            "",
            "| 版本 | 阶段 | 父版本 | 版本键 | 基座 | 适配器 | 数据指纹 | 合格率 | 可部署 |",
            "|------|------|--------|--------|------|--------|---------|--------|--------|",
        ]
        for item in self.versions():
            rate = item.pass_rate
            lines.append(
                f"| v{item.version} | {item.stage} | {item.parent_version or '-'} | "
                f"`{item.version_key}` | {item.base_model} | "
                f"`{item.triple.short_adapter}` | `{item.dataset_fingerprint}` | "
                f"{'-' if rate is None else f'{rate:.4f}'} | "
                f"{'是' if item.is_deployable() else '否'} |"
            )
        lines.append("")
        return "\n".join(lines)


__all__ = [
    "DEFAULT_KEEP_VERSIONS",
    "EVENT_REGISTER",
    "EVENT_STAGE",
    "INDEX_FILENAME",
    "REGISTER_REQUIRED_FIELDS",
    "ModelRegistry",
]
