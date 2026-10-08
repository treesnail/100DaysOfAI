"""``rag_ops`` 的形状族：一次生产化同步里流动的六样东西（day072）.

```text
SourceEntry      一份来源在某一刻的内容身份（相对路径 + 指纹 + 字数 + 媒体类型）
CorpusSnapshot   一圈扫盘的结果：全部来源的身份 + 一个可比较的快照号
SyncPlan         两次快照的差集：新增 / 更新 / 删除 / 未变 四组 + 一份理由
SyncLedger       同步的**账本**：水位（上一份快照）+ 上次尝试/成功时间 + 连续失败数
ScheduleDecision 一次调度判定：该不该跑 + 为什么 + 下一次在哪（含错过窗口与退避）
HealthFinding    一条健康判据的结论：查的是哪一项 + 三态 + 一句人话 + 明细
MetricSample     一个监控采样：指标名 + 值 + 单位（由指标名决定）+ 一组标签
OpsReport        一次运行的账：状态 + 调度判定 + 差集 + 构建报告 + 健康 + 采样
```

## 两条贯穿本模块的纪律

**1. 四张"封闭表"必须在导入期逐键对齐。**

```text
同步动作   SYNC_ACTIONS / SYNC_ACTION_DESCRIPTIONS / SYNC_ACTION_IS_CHANGE
调度模式   SCHEDULE_MODES / SCHEDULE_MODE_DESCRIPTIONS
健康三态   HEALTH_STATES / HEALTH_STATE_DESCRIPTIONS / HEALTH_STATE_EXIT_CODES
健康检查   HEALTH_CHECKS / HEALTH_CHECK_DESCRIPTIONS
监控指标   OPS_METRICS / OPS_METRIC_UNITS / OPS_METRIC_DESCRIPTIONS
运行状态   OPS_STATUSES / OPS_STATUS_DESCRIPTIONS
```

它们少一个键**不会让任何测试变红**，只会让报告里某一项缺少解释、
某条指标进不了汇总、某个状态没有退出码——而"缺一行"与"没什么可报的"
在读的时候长得一样（与 day071 的坏例三张表是同一条理由）。
因此本模块在**导入期**就把这些表比一遍，不一致就当场拒绝导入。

**2. 时间戳一律是"秒级 UTC 字符串"，且**不进**任何身份。**

```text
快照号   snapshot_id   由（来源，指纹）两列算出，与扫描时间无关
清单版本 index_version 由（编码器身份，度量，后端，内容摘要）算出（day065 定的）
账本水位 ledger        记时间，但**判定**只比快照号
```

理由与 day065 的"能重算的才叫版本号"逐字相同：一旦时间进了身份，
"这份快照是哪台机器上扫的"就会变成必须回答的问题，而它其实不该被问。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from smart_research_agent.rag_ops.errors import (
    HealthError,
    MetricsError,
    ScheduleError,
    SyncError,
)

# --------------------------------------------------------------------------- #
# 格式版本与长度常量
# --------------------------------------------------------------------------- #

#: 快照的序列化格式版本（进 ``to_dict``；变了就必须拒读旧文件）.
SNAPSHOT_VERSION = 1

#: 账本的序列化格式版本.
LEDGER_VERSION = 1

#: 运行报告的序列化格式版本.
OPS_REPORT_VERSION = 1

#: 内容指纹的长度（与 ``documents.types.content_id`` 一致：sha256 前 16 位十六进制）.
FINGERPRINT_LENGTH = 16

#: 快照号的长度（与 ``indexing.types.index_version_id`` 一致）.
SNAPSHOT_ID_LENGTH = 16


# --------------------------------------------------------------------------- #
# 时间：秒级 UTC 字符串
# --------------------------------------------------------------------------- #


def utc_now() -> datetime:
    """当前 UTC 时间（**唯一**取时间的地方，方便测试注入时钟）."""
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    """把 ``datetime`` 写成秒级 UTC 字符串：``2026-09-24T03:00:00Z``.

    为什么砍掉微秒：秒级精度够回答运维层的所有问题（"多久没同步了"），
    而微秒进报告只会让两份本该相同的报告 diff 出一行噪声——
    与 day065 把 ``created_at`` 排除在版本号之外是同一种取舍。
    为什么统一成 ``Z`` 后缀：``+00:00`` 与 ``Z`` 是同一时刻的两种写法，
    两种写法混在一份报告里会让"这两条是同一秒吗"变成一道字符串题。
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    """把 :func:`to_iso` 写出的字符串读回 ``datetime``（UTC，秒级）.

    两种拒绝都是 :class:`SyncError`，且消息里带上原值：

    ```text
    空串            不是时间——"没有水位"要用 None / 空串字段表达，不要在这里兜
    解析不了        多半是手改过的账本或报告：当场报错，而不是拿现在时间顶上
    ```

    "拿现在时间顶上"是本模块最想消灭的一类兜底：它让一份坏账本看起来
    只是"刚同步过"，于是下一次增量会把全部来源当成没变（漏更新不报错）。
    """
    raw = str(text or "").strip()
    if not raw:
        raise SyncError(
            "空字符串不是一个时间戳：'没有这个时间'要用空串字段表达（例如 "
            "SyncLedger.last_success_at == ''），不要交给 parse_iso。"
        )
    normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        moment = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SyncError(
            f"时间戳 {raw!r} 不是 ISO 8601 格式（应形如 2026-09-24T03:00:00Z）："
            "账本与报告都是可以被手工编辑的 JSON，读到脏值时报错比猜一个时间安全。"
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def minutes_between(earlier: str, later: datetime) -> float:
    """从 ``earlier``（ISO 串）到 ``later`` 相隔多少分钟（``later`` 更早则为负）."""
    delta = later - parse_iso(earlier)
    return delta.total_seconds() / 60.0


def shift_iso(moment: datetime, **delta: float) -> str:
    """在 ``moment`` 上叠加一个时间差并写回 ISO 串（调度算"下一次"用它）."""
    return to_iso(moment + timedelta(**delta))


# --------------------------------------------------------------------------- #
# 来源与快照
# --------------------------------------------------------------------------- #


def _require_hex(text: str, length: int, *, field_name: str, error: type[ValueError]) -> None:
    """校验一个"定长十六进制"字段（指纹、快照号、版本号共用一条口径）.

    三条判据合在一起，是因为它们各自都对应一类真实事故：

    ```text
    长度不对   "看前 6 位就够了"的手改痕迹——它会让两个不同文件撞成同一个身份
    不是十六进制 大写、带前缀、带空格：同一个值在两次扫描里算出两个身份
    空的          "还没有"要用空串字段（例如 previous_id），而不是空值冒充身份
    ```
    """
    if text == "":
        raise error(f"{field_name} 不能是空串（'还没有'要用空串字段表达，不能拿空值冒充身份）。")
    if len(text) != length:
        raise error(f"{field_name} 必须是 {length} 位，收到 {len(text)} 位：{text!r}。")
    if any(ch not in "0123456789abcdef" for ch in text):
        raise error(
            f"{field_name} 必须是**小写**十六进制（收到 {text!r}）："
            "大小写或前缀会让同一个内容在两次扫描里算出两个身份。"
        )


def _validate_source(source: str) -> None:
    """校验一个来源路径是"相对的 posix 风格路径".

    三条判据对应三类真实问题：

    ```text
    绝对路径    同一份语料在两台机器上路径不同 → 快照号不同 → 每次都全量同步
    反斜杠      Windows 上扫出来的路径进了快照，换到 Linux 上就认不出来
    .. 段       指向语料目录之外：同步会把不该进库的文件也灌进去
    ```
    """
    if not source:
        raise SyncError("来源路径不能是空串。")
    if "\\" in source:
        raise SyncError(
            f"来源路径必须用 '/' 分隔（收到 {source!r}）："
            "反斜杠是 Windows 的写法，它会让同一份语料在 Linux 上算成另一个来源。"
        )
    if source.startswith("/") or (len(source) > 1 and source[1] == ":"):
        raise SyncError(
            f"来源路径必须是**相对**路径（收到 {source!r}）："
            "绝对路径会让快照随机器而变，而“变了”与“没变”正是本模块唯一要判的事。"
        )
    if ".." in source.split("/"):
        raise SyncError(
            f"来源路径不能含 '..' 段（收到 {source!r}）："
            "它指向语料目录之外，同步会把不该进库的文件也灌进去。"
        )


@dataclass(frozen=True)
class SourceEntry:
    """一份来源在某一刻的身份：路径 + 内容指纹 + 两个描述性数字.

    ``char_count`` 与 ``media_type`` **不参与**判定（快照号只由路径与指纹算出），
    它们进快照是为了让报告能回答"这次改了多少字""这份是什么格式"——
    把描述性字段拉进身份，会让"改了解析器"（字数变了、内容没变）被误判成"内容变了"。
    """

    source: str
    fingerprint: str
    char_count: int = 0
    media_type: str = ""

    def __post_init__(self) -> None:
        _validate_source(self.source)
        _require_hex(
            self.fingerprint,
            FINGERPRINT_LENGTH,
            field_name="fingerprint",
            error=SyncError,
        )
        if self.char_count < 0:
            raise SyncError(f"char_count 不能为负：来源 {self.source!r} 报 {self.char_count}。")

    def to_dict(self) -> dict[str, Any]:
        """一行 JSON 的形状（**顺序固定**，便于 diff 两份快照）."""
        return {
            "source": self.source,
            "fingerprint": self.fingerprint,
            "char_count": self.char_count,
            "media_type": self.media_type,
        }

    def summary_line(self) -> str:
        """一行说明：``path | 指纹 | 字数 | 媒体类型``."""
        kind = self.media_type or "未知"
        return f"{self.source} | {self.fingerprint} | {self.char_count} 字 | {kind}"


def snapshot_id(entries: tuple[SourceEntry, ...]) -> str:
    """由（来源，指纹）两列算出快照号（**16 位小写十六进制**）.

    与 ``indexing.types.index_version_id`` 同一个套路，且同样刻意**不含**时间：

    ```text
    含时间 → 同一个语料扫两次得到两个快照号 → 账本永远"和现在不一样"
             → 每一次调度都判定"要同步"（而且它看起来只是"勤快"）
    ```
    """
    body = "\n".join(f"{entry.source}|{entry.fingerprint}" for entry in entries)
    return hashlib.sha256(f"snapshot|{SNAPSHOT_VERSION}|{body}".encode()).hexdigest()[
        :SNAPSHOT_ID_LENGTH
    ]


@dataclass(frozen=True)
class CorpusSnapshot:
    """一圈扫盘的结果：全部来源的身份 + 一个可比较的快照号.

    ``skipped`` 是"读不进来但没让整次扫描失败"的那些来源（``strict=False`` 时才有），
    它的存在是为了让"漏了一份文档"这件事**留在账上**：

    ```text
    读不了的文件静默跳过 → 库里少一条 → 检索质量下降，没有任何一处报错
    读不了的文件进 skipped → 报告里多一行"有 1 份没读进来" → 有人会去看
    ```

    构造时会把 ``entries`` 与 ``skipped`` 规范化（按路径排序）、校验唯一性，
    并保证两边的来源**互不相交**：同一份文件不能既"读进来了"又"跳过了"。
    """

    entries: tuple[SourceEntry, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    created_at: str = ""
    version: int = SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != SNAPSHOT_VERSION:
            raise SyncError(
                f"快照格式版本 {self.version} 不认识（本版本只认 {SNAPSHOT_VERSION}）："
                "格式变了就必须拒读旧文件，而不是按新口径猜它的含义。"
            )
        entries = tuple(sorted(self.entries, key=lambda item: item.source))
        seen: dict[str, str] = {}
        for entry in entries:
            previous = seen.get(entry.source)
            if previous is not None:
                raise SyncError(
                    f"来源 {entry.source!r} 在快照里出现了两次（指纹 {previous} / "
                    f"{entry.fingerprint}）：一份来源只有一个身份，"
                    "同一路径两行会让“它变了吗”这个问题有两个答案。"
                )
            seen[entry.source] = entry.fingerprint
        skipped = tuple(sorted((str(src), str(err)) for src, err in self.skipped))
        for source, error in skipped:
            _validate_source(source)
            if not error:
                raise SyncError(f"跳过的来源 {source!r} 必须带一条失败原因。")
            if source in seen:
                raise SyncError(
                    f"来源 {source!r} 既在 entries 里又在 skipped 里："
                    "同一份文件不能既读进来了又跳过了。"
                )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "skipped", skipped)
        if self.created_at:
            parse_iso(self.created_at)

    # ------------------------------------------------------------------ 派生量

    @property
    def count(self) -> int:
        """读进来的来源数."""
        return len(self.entries)

    @property
    def digest(self) -> str:
        """快照号（**只由 entries 算出**，skipped 不进身份）.

        为什么 skipped 不进：它是一次尝试的**环境**结果（权限、编码），
        而不是语料的内容。把它算进身份会让"上一次有一份没读进来"变成
        "语料变了"，于是下一次调度又会全量重跑一遍。
        """
        return snapshot_id(self.entries)

    @property
    def sources(self) -> tuple[str, ...]:
        """全部来源路径（升序）."""
        return tuple(entry.source for entry in self.entries)

    @property
    def total_chars(self) -> int:
        """全部来源的字数之和."""
        return sum(entry.char_count for entry in self.entries)

    def by_source(self) -> dict[str, SourceEntry]:
        """``路径 → 条目`` 的查表（增量比对与报告都用它）."""
        return {entry.source: entry for entry in self.entries}

    def entry(self, source: str) -> SourceEntry | None:
        """取一份来源；不存在返回 ``None``（查不到是正常结果，不是异常）."""
        return self.by_source().get(source)

    def to_dict(self, *, include_entries: bool = True) -> dict[str, Any]:
        """可 json.dumps 的形状（``include_entries=False`` 时只报聚合数字）."""
        payload: dict[str, Any] = {
            "version": self.version,
            "snapshot_id": self.digest,
            "count": self.count,
            "total_chars": self.total_chars,
            "skipped": len(self.skipped),
            "created_at": self.created_at,
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
            payload["skipped_detail"] = [list(item) for item in self.skipped]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CorpusSnapshot:
        """从 :meth:`to_dict` 的产物读回一份快照（缺 ``entries`` 时读到空快照）."""
        entries = tuple(
            SourceEntry(
                source=str(item["source"]),
                fingerprint=str(item["fingerprint"]),
                char_count=int(item.get("char_count", 0)),
                media_type=str(item.get("media_type", "")),
            )
            for item in payload.get("entries") or ()
        )
        return cls(
            entries=entries,
            skipped=tuple((str(a), str(b)) for a, b in payload.get("skipped_detail") or ()),
            created_at=str(payload.get("created_at", "")),
            version=int(payload.get("version", SNAPSHOT_VERSION)),
        )

    def summary_line(self) -> str:
        """一行说明：``快照 3f2a… | 3 份来源 | 12 字 | 0 份跳过``."""
        return (
            f"快照 {self.digest} | {self.count} 份来源 | {self.total_chars} 字 | "
            f"{len(self.skipped)} 份跳过"
        )


def snapshot_from_entries(
    entries: tuple[SourceEntry, ...],
    *,
    skipped: tuple[tuple[str, str], ...] = (),
    created_at: str = "",
) -> CorpusSnapshot:
    """装一份快照（``CorpusSnapshot`` 的显式工厂，省去每次都写 ``version=``）."""
    return CorpusSnapshot(entries=entries, skipped=skipped, created_at=created_at)


# --------------------------------------------------------------------------- #
# 同步：四组动作 + 一份差集
# --------------------------------------------------------------------------- #

#: 新增：上一份快照里没有这份来源.
ACTION_ADDED = "added"

#: 更新：路径在，但内容指纹变了.
ACTION_UPDATED = "updated"

#: 删除：上一份快照里有，现在没有了.
ACTION_REMOVED = "removed"

#: 未变：路径在，指纹也一样.
ACTION_UNCHANGED = "unchanged"

#: 四种动作（**顺序 = 报告里的排列 = 处置的紧急度**）.
SYNC_ACTIONS: tuple[str, ...] = (ACTION_ADDED, ACTION_UPDATED, ACTION_REMOVED, ACTION_UNCHANGED)

#: 每一种动作的"一句话解释"（进报告，少一个键 = 某类变化没有解释）.
SYNC_ACTION_DESCRIPTIONS: dict[str, str] = {
    ACTION_ADDED: "新增来源：上一份快照里没有它，需要解析、分块并写入向量库",
    ACTION_UPDATED: "内容变了：路径还在、指纹不同，需要重新解析入库（旧块会被替换）",
    ACTION_REMOVED: "来源没了：语料目录里已不存在，它在库里的块必须被删掉",
    ACTION_UNCHANGED: "没有变化：指纹一致，本次不为它花任何编码成本",
}

#: 哪些动作算"变化"（决定要不要真的跑一次构建）.
SYNC_ACTION_IS_CHANGE: dict[str, bool] = {
    ACTION_ADDED: True,
    ACTION_UPDATED: True,
    ACTION_REMOVED: True,
    ACTION_UNCHANGED: False,
}

if not (
    set(SYNC_ACTIONS) == set(SYNC_ACTION_DESCRIPTIONS) == set(SYNC_ACTION_IS_CHANGE)
):
    raise SyncError(
        "同步动作的三张表不一致：SYNC_ACTIONS / SYNC_ACTION_DESCRIPTIONS / "
        "SYNC_ACTION_IS_CHANGE 必须逐键对齐，否则某一类变化会在报告里只有名字、"
        "没有解释，或者被静默地当成'没变化'。"
    )


@dataclass(frozen=True)
class SyncPlan:
    """两次快照的差集：四组来源 + 两侧快照号 + 两侧来源清单 + 一份理由.

    四组必须**互不相交**，且各自覆盖两边：

    ```text
    removed ∪ updated ∪ unchanged == previous_sources   （上一份快照被完全解释）
    added   ∪ updated ∪ unchanged == current_sources    （这一份快照被完全解释）
    ```

    这两条恒等式不是装饰：漏掉一份来源会让"库里少一条记录"，
    而它的表象只是"这次改得不多"——没有异常、没有告警，只有检索质量下降。
    因此它们在构造期就被校验（差集是**纯函数**算出来的，本来就应该满足）。

    ``removed`` 与 ``updated`` 都算"变化"，但**处置方向相反**：
    前者要删库里的东西，后者要覆盖写。合成一个 ``changed`` 之后，
    "这次要不要清点什么"就再也从计划里读不出来了。
    """

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    previous_id: str = ""
    current_id: str = ""
    previous_sources: tuple[str, ...] = ()
    current_sources: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("added", "updated", "removed", "unchanged"):
            values = tuple(sorted({str(item) for item in getattr(self, name)}))
            for source in values:
                _validate_source(source)
            object.__setattr__(self, name, values)
        if self.previous_id:
            _require_hex(
                self.previous_id,
                SNAPSHOT_ID_LENGTH,
                field_name="previous_id",
                error=SyncError,
            )
        if self.current_id:
            _require_hex(
                self.current_id,
                SNAPSHOT_ID_LENGTH,
                field_name="current_id",
                error=SyncError,
            )
        for name in ("previous_sources", "current_sources"):
            values = tuple(sorted({str(item) for item in getattr(self, name)}))
            object.__setattr__(self, name, values)
        groups = {
            "added": set(self.added),
            "updated": set(self.updated),
            "removed": set(self.removed),
            "unchanged": set(self.unchanged),
        }
        names = list(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise SyncError(
                        f"差集里 {left} 与 {right} 同时含有 {sorted(overlap)}："
                        "一份来源在一次同步里只能落在**一个**动作上——"
                        "落在两个上时，读的人不知道它到底会被覆盖还是被删除。"
                    )
        expected_previous = groups["removed"] | groups["updated"] | groups["unchanged"]
        if expected_previous != set(self.previous_sources):
            missing = sorted(set(self.previous_sources) - expected_previous)
            extra = sorted(expected_previous - set(self.previous_sources))
            raise SyncError(
                "差集没有完全解释上一份快照："
                f"漏掉的来源 {missing}、多出来的来源 {extra}。"
                "漏掉的来源会静默地留在库里（没人删它），"
                "而它的表象只是“这次改得不多”。"
            )
        expected_current = groups["added"] | groups["updated"] | groups["unchanged"]
        if expected_current != set(self.current_sources):
            missing = sorted(set(self.current_sources) - expected_current)
            extra = sorted(expected_current - set(self.current_sources))
            raise SyncError(
                "差集没有完全解释这一份快照："
                f"漏掉的来源 {missing}、多出来的来源 {extra}。"
                "漏掉的来源不会被写进库（没人写它），"
                "而它的表象同样是“这次改得不多”。"
            )

    # ------------------------------------------------------------------ 派生量

    def sources(self, action: str) -> tuple[str, ...]:
        """取某一类动作的来源清单（未知动作名当场报错）."""
        if action not in SYNC_ACTIONS:
            raise SyncError(
                f"不认识的同步动作 {action!r}：可用取值 {list(SYNC_ACTIONS)}。"
            )
        return tuple(getattr(self, action))

    @property
    def changed(self) -> tuple[str, ...]:
        """全部变化的来源（增 + 改 + 删），**升序去重**."""
        return tuple(sorted({*self.added, *self.updated, *self.removed}))

    @property
    def changed_count(self) -> int:
        """变化的来源数（"这次要动几份"）."""
        return len(self.changed)

    @property
    def empty(self) -> bool:
        """一份都没变（``unchanged`` 不算变化）."""
        return self.changed_count == 0

    @property
    def churn_ratio(self) -> float:
        """变更比例 = 变化来源数 / 两侧来源的**并集**大小（并集为空时 0.0）.

        分母取并集而不是"当前来源数"，是因为删除也算变化：
        一个"删掉了 5 份、一份没留"的目录，用当前数做分母会得到 5/0，
        而用并集做分母得到 1.0（"这一趟全变了"）——后者才是要传给
        "要不要整库重建"这个判断的那个数字。
        """
        union = {*self.previous_sources, *self.current_sources}
        if not union:
            return 0.0
        return round(self.changed_count / len(union), 6)

    def to_dict(self, *, include_sources: bool = True) -> dict[str, Any]:
        """可 json.dumps 的形状（默认把两边来源清单也带上）."""
        payload: dict[str, Any] = {
            "added": list(self.added),
            "updated": list(self.updated),
            "removed": list(self.removed),
            "unchanged": list(self.unchanged),
            "counts": {action: len(self.sources(action)) for action in SYNC_ACTIONS},
            "changed": self.changed_count,
            "churn_ratio": self.churn_ratio,
            "empty": self.empty,
            "previous_id": self.previous_id,
            "current_id": self.current_id,
            "reason": self.reason,
        }
        if include_sources:
            payload["previous_sources"] = list(self.previous_sources)
            payload["current_sources"] = list(self.current_sources)
        return payload

    def summary_line(self) -> str:
        """一行说明：``3 份来源（+1 ~1 -0）| 变更比例 33.3%``."""
        return (
            f"{len(self.current_sources)} 份来源"
            f"（+{len(self.added)} ~{len(self.updated)} -{len(self.removed)}）"
            f"| 未变 {len(self.unchanged)} | 变更比例 {self.churn_ratio:.1%}"
        )


# --------------------------------------------------------------------------- #
# 账本：水位 + 尝试史
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyncLedger:
    """同步的账本：水位（上一份**成功**的快照）+ 尝试史 + 连续失败数.

    三个时间字段的语义必须分清，否则"多久没同步了"会有三个答案：

    ```text
    last_run_at      上次**尝试**的时间（成功或失败都写）
    last_success_at  上次**成功**的时间（只有成功才前进）
    last_error       最近一次失败的原因（连续失败数 > 0 时必然非空）
    ```

    ``snapshot`` 只在成功时前进。这条纪律的代价是"失败之后同一批来源会被
    再处理一次"，收益是"失败不会造成漏更新"——而重跑是幂等的
    （day065 的增量对账会把没变的块判成 ``unchanged``），漏更新不是。
    """

    snapshot: CorpusSnapshot | None = None
    last_run_at: str = ""
    last_success_at: str = ""
    last_index_version: str = ""
    consecutive_failures: int = 0
    runs: int = 0
    failures: int = 0
    last_error: str = ""
    version: int = LEDGER_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != LEDGER_VERSION:
            raise SyncError(
                f"账本格式版本 {self.version} 不认识（本版本只认 {LEDGER_VERSION}）："
                "账本里存着水位，按新口径猜旧账本等于自己给自己写一份假水位。"
            )
        for name in ("last_run_at", "last_success_at"):
            value = getattr(self, name)
            if value:
                parse_iso(value)
        if self.consecutive_failures < 0:
            raise SyncError(f"consecutive_failures 不能为负：{self.consecutive_failures}。")
        if self.runs < 0 or self.failures < 0:
            raise SyncError(
                f"runs / failures 不能为负：runs={self.runs} failures={self.failures}。"
            )
        if self.failures > self.runs:
            raise SyncError(
                f"失败次数 {self.failures} 超过尝试次数 {self.runs}："
                "这两个数来自同一次自增，其中一个是手改过的。"
            )
        if self.consecutive_failures > 0 and not self.last_error:
            raise SyncError(
                "连续失败数大于 0，但 last_error 是空的："
                "报告里只会剩下“失败了”、没有原因，而那种消息没人能照着修。"
            )
        if self.last_index_version:
            _require_hex(
                self.last_index_version,
                SNAPSHOT_ID_LENGTH,
                field_name="last_index_version",
                error=SyncError,
            )

    # ------------------------------------------------------------------ 构造

    @classmethod
    def empty(cls) -> SyncLedger:
        """一份全新的账本（**没有水位**：下一次同步会按全量处理）."""
        return cls()

    @property
    def has_watermark(self) -> bool:
        """有没有水位（水位是"上一份成功快照"，不是"上次运行时间"）."""
        return self.snapshot is not None

    # ------------------------------------------------------------------ 前进

    def mark_success(
        self,
        *,
        at: str,
        snapshot: CorpusSnapshot,
        index_version: str = "",
    ) -> SyncLedger:
        """记一次成功：水位前进、连续失败清零、两个计数各自 +1.

        ``index_version`` 缺省保留旧值：一次"没变化所以没构建"的成功运行
        不该把"库是哪一版"擦掉——那一版仍然生效着。
        """
        return SyncLedger(
            snapshot=snapshot,
            last_run_at=at,
            last_success_at=at,
            last_index_version=index_version or self.last_index_version,
            consecutive_failures=0,
            runs=self.runs + 1,
            failures=self.failures,
            last_error="",
        )

    def mark_failure(self, *, at: str, error: str) -> SyncLedger:
        """记一次失败：**水位不动**、连续失败数 +1、原因留下.

        水位不动是这一层最关键的一条纪律：一次失败如果推进了水位，
        那么失败的那批来源会被永久跳过（"已经处理过了"），
        而库里少的那些记录没有任何地方会提醒你。
        """
        if not error:
            raise SyncError(
                "记录失败必须带一条原因：只记'失败了'的账本没有可执行的下一步。"
            )
        return SyncLedger(
            snapshot=self.snapshot,
            last_run_at=at,
            last_success_at=self.last_success_at,
            last_index_version=self.last_index_version,
            consecutive_failures=self.consecutive_failures + 1,
            runs=self.runs + 1,
            failures=self.failures + 1,
            last_error=error,
        )

    def age_minutes(self, now: datetime) -> float | None:
        """距上次成功同步多少分钟；从未成功时返回 ``None``."""
        if not self.last_success_at:
            return None
        return round(minutes_between(self.last_success_at, now), 4)

    def describe(self, *, now: datetime | None = None) -> dict[str, Any]:
        """账本的一段人话摘要（端点与报告共用，避免两处各拼一遍）."""
        moment = now if now is not None else utc_now()
        age = self.age_minutes(moment)
        return {
            "watermark": self.snapshot.digest if self.snapshot is not None else None,
            "sources": self.snapshot.count if self.snapshot is not None else 0,
            "last_run_at": self.last_run_at,
            "last_success_at": self.last_success_at,
            "last_index_version": self.last_index_version,
            "age_minutes": age,
            "runs": self.runs,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }

    def summary_line(self) -> str:
        """一行说明：``水位 3f2a… | 上次成功 … | 连续失败 0``."""
        watermark = self.snapshot.digest if self.snapshot is not None else "（无）"
        return (
            f"水位 {watermark} | 上次成功 {self.last_success_at or '从未'} | "
            f"连续失败 {self.consecutive_failures}"
        )

    # ------------------------------------------------------------------ 落盘

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（快照的 entries 也写进去——它是水位本体）."""
        return {
            "version": self.version,
            "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
            "last_run_at": self.last_run_at,
            "last_success_at": self.last_success_at,
            "last_index_version": self.last_index_version,
            "consecutive_failures": self.consecutive_failures,
            "runs": self.runs,
            "failures": self.failures,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SyncLedger:
        """从 :meth:`to_dict` 的产物读回一份账本."""
        raw = payload.get("snapshot")
        return cls(
            snapshot=CorpusSnapshot.from_dict(raw) if isinstance(raw, dict) else None,
            last_run_at=str(payload.get("last_run_at", "")),
            last_success_at=str(payload.get("last_success_at", "")),
            last_index_version=str(payload.get("last_index_version", "")),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            runs=int(payload.get("runs", 0)),
            failures=int(payload.get("failures", 0)),
            last_error=str(payload.get("last_error", "")),
            version=int(payload.get("version", LEDGER_VERSION)),
        )

    def save(self, path: str | Path) -> Path:
        """把账本写成一份 JSON（自动建父目录），返回落盘路径."""
        target = Path(path)
        if not str(target):
            raise SyncError("账本落盘路径不能是空串（空路径会写到进程的当前目录）。")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> SyncLedger:
        """读回账本；**文件不存在时返回空账本**（与 ``RagBaseline.load`` 刻意相反）.

        两处相反是刻意的，理由也正好相反：

        ```text
        RagBaseline.load   基线不存在 → **报错**
                           "没有基线"与"基线全是零"是两回事，后者会让门禁静默放行
        SyncLedger.load    账本不存在 → 返回空账本
                           "没有账本"就是**首次运行**，它本来就该按全量处理一遍；
                           报错只会让一次全新的部署起不来
        ```

        但**读得回来却读不懂**（JSON 坏了、格式版本不认识）一律照抛：
        那才是"有一份假账"的特征，而假账比没有账危险得多。
        """
        target = Path(path)
        if not target.exists():
            return cls.empty()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SyncError(
                f"账本 {str(target)!r} 不是合法 JSON（第 {exc.lineno} 行）："
                "坏账本比没有账本危险——没有账本会全量重跑，坏账本会被当成水位。"
            ) from exc
        if not isinstance(payload, dict):
            raise SyncError(
                f"账本 {str(target)!r} 的顶层必须是对象，收到 {type(payload).__name__}。"
            )
        return cls.from_dict(payload)


# --------------------------------------------------------------------------- #
# 调度：策略 + 判定
# --------------------------------------------------------------------------- #

#: 第一次运行（还没有账本）.
MODE_FIRST_RUN = "first_run"

#: 到点了，该跑.
MODE_DUE = "due"

#: 还没到点.
MODE_NOT_DUE = "not_due"

#: 上次失败过，正在退避窗口里.
MODE_BACKOFF = "backoff"

#: 被显式要求跑（``force=True``）——"我不管间隔，现在就跑".
MODE_FORCED = "forced"

#: 五种模式（``due`` / ``forced`` 才是"要跑"）.
SCHEDULE_MODES: tuple[str, ...] = (
    MODE_FIRST_RUN,
    MODE_DUE,
    MODE_NOT_DUE,
    MODE_BACKOFF,
    MODE_FORCED,
)

#: 每一种模式的一句话解释.
SCHEDULE_MODE_DESCRIPTIONS: dict[str, str] = {
    MODE_FIRST_RUN: "首次运行：没有账本，按全量处理一次",
    MODE_DUE: "已到计划时间：正常触发",
    MODE_NOT_DUE: "未到计划时间：本次不跑（不产生任何编码成本）",
    MODE_BACKOFF: "正在退避：上次失败过，等在重试窗口里",
    MODE_FORCED: "被显式要求：忽略间隔与退避，立即执行一次",
}

if set(SCHEDULE_MODES) != set(SCHEDULE_MODE_DESCRIPTIONS):
    raise ScheduleError(
        "调度模式的两张表不一致：SCHEDULE_MODES / SCHEDULE_MODE_DESCRIPTIONS 必须逐键对齐，"
        "否则某一种模式在报告里只有名字、没有解释。"
    )


@dataclass(frozen=True)
class SchedulePolicy:
    """调度策略：多久一次、抖多大、退避多久、错过多久算失效.

    五个参数的**量纲**都写在名字里，且都可由配置覆盖（``rag_ops_*`` 一组）：

    ```text
    interval_minutes      86400 秒的常识值：1440（每天一次，凌晨那趟调度的本意）
    jitter_seconds        抖动上限（秒）：把所有实例的调度点错开，避开整点争抢
    max_lag_minutes       滞后多久算"错过了一个窗口"（只写进理由，不吞掉本次运行）
    backoff_base_seconds  首次失败后的重试等待（秒）
    backoff_max_seconds   退避上限（秒）：指数退避封顶，避免永远不重试
    ```
    """

    interval_minutes: int = 1440
    jitter_seconds: int = 300
    max_lag_minutes: int = 720
    backoff_base_seconds: int = 60
    backoff_max_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.interval_minutes < 1:
            raise ScheduleError(
                f"interval_minutes 必须 >= 1，收到 {self.interval_minutes}："
                "间隔为 0 会让调度每次判定都'到点了'，于是同步退化成一个忙循环。"
            )
        if self.jitter_seconds < 0:
            raise ScheduleError(f"jitter_seconds 不能为负：{self.jitter_seconds}。")
        if self.jitter_seconds >= self.interval_minutes * 60:
            raise ScheduleError(
                f"抖动上限 {self.jitter_seconds}s 不小于调度间隔 "
                f"{self.interval_minutes} 分钟：抖动比间隔还大时，"
                "两次计划的先后顺序会被打乱，'下一次'这个答案就没有意义了。"
            )
        if self.max_lag_minutes < 0:
            raise ScheduleError(f"max_lag_minutes 不能为负：{self.max_lag_minutes}。")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ScheduleError("退避参数不能为负。")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ScheduleError(
                f"退避上限 {self.backoff_max_seconds}s 小于基数 {self.backoff_base_seconds}s："
                "上界小于下界时，'封顶'这个动作会把等待时间越封越小。"
            )

    def to_dict(self) -> dict[str, int]:
        """可 json.dumps 的形状（进报告与 ``/rag/ops/status``）."""
        return {
            "interval_minutes": self.interval_minutes,
            "jitter_seconds": self.jitter_seconds,
            "max_lag_minutes": self.max_lag_minutes,
            "backoff_base_seconds": self.backoff_base_seconds,
            "backoff_max_seconds": self.backoff_max_seconds,
        }

    def summary_line(self) -> str:
        """一行说明：``每 1440 分钟一次 | 抖动 ≤300s | 退避 60→3600s``."""
        return (
            f"每 {self.interval_minutes} 分钟一次 | 抖动 ≤{self.jitter_seconds}s | "
            f"退避 {self.backoff_base_seconds}→{self.backoff_max_seconds}s"
        )

    def backoff_seconds(self, consecutive_failures: int) -> int:
        """连续失败 ``n`` 次后的重试等待（秒）：**指数退避 + 封顶**.

        ``n <= 0`` 时为 0（没有失败就没有等待）。指数取 ``2 ** min(n-1, 20)``
        而不是无脑左移：无脑左移在 n 稍大时就会算出一个天文数字，
        而那个数字会被 ceil 成一个"永远不重试"的等待——比退避本身更糟。
        """
        if consecutive_failures <= 0:
            return 0
        exponent = min(consecutive_failures - 1, 20)
        return min(self.backoff_base_seconds * (2**exponent), self.backoff_max_seconds)


@dataclass(frozen=True)
class ScheduleDecision:
    """一次调度判定：该不该跑 + 为什么 + 下一次在哪 + 错过了几次.

    ``due`` 为 ``False`` 时**不是失败**：它是一条明确的结论
    （"还没到点，本次不产生任何编码成本"），因此报告里照样有它一席之地。
    """

    mode: str
    due: bool
    reason: str
    now: str
    effective_at: str = ""
    next_run_at: str = ""
    lag_minutes: float = 0.0
    missed_runs: int = 0
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        if self.mode not in SCHEDULE_MODES:
            raise ScheduleError(
                f"不认识的调度模式 {self.mode!r}：可用取值 {list(SCHEDULE_MODES)}。"
            )
        if self.due and self.mode in (MODE_NOT_DUE, MODE_BACKOFF):
            raise ScheduleError(
                f"模式 {self.mode!r} 与 due=True 矛盾："
                "'还没到点'与'该跑了'是同一次判定里唯一互斥的两个结论。"
            )
        if not self.due and self.mode in (MODE_DUE, MODE_FORCED, MODE_FIRST_RUN):
            raise ScheduleError(
                f"模式 {self.mode!r} 与 due=False 矛盾："
                "这个模式本身就意味着'本次要跑'，把 due 设成 False 会让端点少跑一次而不报警。"
            )
        for name in ("now", "effective_at", "next_run_at"):
            value = getattr(self, name)
            if value:
                parse_iso(value)
        if self.lag_minutes < 0:
            raise ScheduleError(f"lag_minutes 不能为负：{self.lag_minutes}。")
        if self.missed_runs < 0:
            raise ScheduleError(f"missed_runs 不能为负：{self.missed_runs}。")
        if self.consecutive_failures < 0:
            raise ScheduleError(f"consecutive_failures 不能为负：{self.consecutive_failures}。")
        if not self.reason:
            raise ScheduleError("调度判定必须带一条理由：没有理由的'不跑'与'忘了跑'长得一样。")

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含模式说明与策略无关的两个时间点）."""
        return {
            "mode": self.mode,
            "mode_description": SCHEDULE_MODE_DESCRIPTIONS[self.mode],
            "due": self.due,
            "reason": self.reason,
            "now": self.now,
            "effective_at": self.effective_at,
            "next_run_at": self.next_run_at,
            "lag_minutes": self.lag_minutes,
            "missed_runs": self.missed_runs,
            "consecutive_failures": self.consecutive_failures,
        }

    def summary_line(self) -> str:
        """一行说明：``未到点（mode=not_due）| 下次 …``."""
        flag = "该跑" if self.due else "不跑"
        return (
            f"{flag}（mode={self.mode}）| 下次 {self.next_run_at or '（未知）'} | "
            f"滞后 {self.lag_minutes:.1f} 分钟 | 错过 {self.missed_runs} 次"
        )


# --------------------------------------------------------------------------- #
# 健康：三态 + 四项检查
# --------------------------------------------------------------------------- #

#: 一切正常.
STATE_OK = "ok"

#: 有需要人看一眼的事情，但它不该让容器被判定为不健康.
STATE_WARN = "warn"

#: 这个实例**现在不能**回答知识库问题.
STATE_FAIL = "fail"

#: 三种状态（顺序 = 严重度递增，"取最坏"按它排）.
HEALTH_STATES: tuple[str, ...] = (STATE_OK, STATE_WARN, STATE_FAIL)

#: 每一种状态的一句话解释.
HEALTH_STATE_DESCRIPTIONS: dict[str, str] = {
    STATE_OK: "正常：这一项没查出问题",
    STATE_WARN: "警告：有问题但不该摘掉实例，需要人看一眼（或先跑一次同步）",
    STATE_FAIL: "失败：这个实例现在不能回答知识库问题，编排层应停止向它导流量",
}

#: 状态 → 进程退出码（容器探针与 CI 共用；``warn`` 仍然算"活着"）.
HEALTH_STATE_EXIT_CODES: dict[str, int] = {STATE_OK: 0, STATE_WARN: 0, STATE_FAIL: 1}

#: 严重度排序用的名次（取最坏时用）.
HEALTH_STATE_RANK: dict[str, int] = {state: index for index, state in enumerate(HEALTH_STATES)}

#: 四项检查（``vector_store`` 看库有没有东西、``index_integrity`` 看清单与库对不对得上、
#: ``corpus_freshness`` 看多久没同步、``quality_regression`` 看质量有没有掉）.
CHECK_VECTOR_STORE = "vector_store"
CHECK_INDEX_INTEGRITY = "index_integrity"
CHECK_CORPUS_FRESHNESS = "corpus_freshness"
CHECK_QUALITY_REGRESSION = "quality_regression"

#: 全部检查项（顺序 = 报告里的排列 = 从"能不能用"到"用得对不对"）.
HEALTH_CHECKS: tuple[str, ...] = (
    CHECK_VECTOR_STORE,
    CHECK_INDEX_INTEGRITY,
    CHECK_CORPUS_FRESHNESS,
    CHECK_QUALITY_REGRESSION,
)

#: 每一项检查的一句话解释（进报告与端点，少一个键 = 某一项查了什么没人知道）.
HEALTH_CHECK_DESCRIPTIONS: dict[str, str] = {
    CHECK_VECTOR_STORE: "向量库里有多少条记录（0 条意味着这个实例答不了任何知识库问题）",
    CHECK_INDEX_INTEGRITY: "清单与库是否对得上（少记录、多记录、维度或口径不符）",
    CHECK_CORPUS_FRESHNESS: "距上次**成功**同步有多久（超过阈值说明语料可能已经过期）",
    CHECK_QUALITY_REGRESSION: "最近一次 RAG 评估相对基线有没有退化（不跑评估，只读结论）",
}

if not (
    set(HEALTH_STATES) == set(HEALTH_STATE_DESCRIPTIONS) == set(HEALTH_STATE_EXIT_CODES)
):
    raise HealthError(
        "健康三态的三张表不一致：HEALTH_STATES / HEALTH_STATE_DESCRIPTIONS / "
        "HEALTH_STATE_EXIT_CODES 必须逐键对齐，否则某个状态既没有解释也没有退出码——"
        "而探针会把“没有退出码”当成 0（健康）。"
    )
if set(HEALTH_CHECKS) != set(HEALTH_CHECK_DESCRIPTIONS):
    raise HealthError(
        "健康检查的两张表不一致：HEALTH_CHECKS / HEALTH_CHECK_DESCRIPTIONS 必须逐键对齐，"
        "否则某一项检查在报告里只有名字、没有解释。"
    )


def worst_state(states: tuple[str, ...]) -> str:
    """取最坏的那个状态（空的 ``states`` 返回 ``ok``）.

    只给一个状态而不是把三项都列出来：报告里读的人需要的是
    "这个实例现在能不能服务"，而三行并列会让"有一项 fail"淹没在两行 ok 里。
    明细照旧保留在 ``findings`` 里。
    """
    if not states:
        return STATE_OK
    for state in reversed(HEALTH_STATES):
        if state in states:
            return state
    raise HealthError(f"状态取值不在闭集里：{list(states)}（可用 {list(HEALTH_STATES)}）。")


@dataclass(frozen=True)
class HealthFinding:
    """一条健康判据的结论：查的是哪一项 + 三态 + 一句人话 + 明细.

    ``message`` 必须非空，且**要能照着做**：一条写着"异常"的判据
    与一条写着"库里 0 条记录，先跑一次 POST /rag/ops/sync"的判据，
    差别不在于文笔，而在于后者能被值班的人执行。
    """

    check: str
    state: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.check not in HEALTH_CHECKS:
            raise HealthError(
                f"不认识的检查项 {self.check!r}：可用取值 {list(HEALTH_CHECKS)}。"
            )
        if self.state not in HEALTH_STATES:
            raise HealthError(
                f"不认识的状态 {self.state!r}：可用取值 {list(HEALTH_STATES)}。"
            )
        if not self.message:
            raise HealthError(
                f"检查 {self.check!r} 的结论不能没有说明："
                "一条只写着状态的判据没人能照着修。"
            )
        object.__setattr__(self, "detail", dict(self.detail))

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含这一项检查在查什么）."""
        return {
            "check": self.check,
            "description": HEALTH_CHECK_DESCRIPTIONS[self.check],
            "state": self.state,
            "state_description": HEALTH_STATE_DESCRIPTIONS[self.state],
            "message": self.message,
            "detail": dict(self.detail),
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] vector_store 向量库里有 3 条记录``."""
        return f"[{self.state}] {self.check} {self.message}"


@dataclass(frozen=True)
class HealthReport:
    """一次健康体检：若干条判据 + 一个总状态 + 一个退出码.

    **空报告会当场报错**，这是刻意的：一份"一项都没查"的报告与一份"全绿"的报告
    在 ``state`` 上长得一模一样（都是 ``ok``），而它们的含义完全相反。
    要表达"这次只查了两项"，就只放两项进去（每项自己会说话）；
    要表达"什么都查不了"，应该让那一项以 ``warn`` / ``fail`` 出现，
    而不是把报告空着交出去。
    """

    findings: tuple[HealthFinding, ...] = ()
    checked_at: str = ""

    def __post_init__(self) -> None:
        if not self.findings:
            raise HealthError(
                "健康报告不能是空的：'一项都没查'与'全绿'的 state 都是 ok，"
                "交出一份空报告等于宣称这个实例健康。"
            )
        seen: list[str] = []
        for finding in self.findings:
            if finding.check in seen:
                raise HealthError(
                    f"检查项 {finding.check!r} 在报告里出现了两次："
                    "同一项的两个结论会让'这个实例到底怎么样'有两个答案。"
                )
            seen.append(finding.check)
        if self.checked_at:
            parse_iso(self.checked_at)

    # ------------------------------------------------------------------ 派生量

    @property
    def state(self) -> str:
        """总状态（取最坏的一条）."""
        return worst_state(tuple(finding.state for finding in self.findings))

    @property
    def exit_code(self) -> int:
        """给容器探针 / CI 用的退出码（``warn`` 仍算活着）."""
        return HEALTH_STATE_EXIT_CODES[self.state]

    @property
    def ok(self) -> bool:
        """能不能继续服务（``fail`` 才算不能）."""
        return self.state != STATE_FAIL

    @property
    def failed(self) -> tuple[HealthFinding, ...]:
        """全部 ``fail`` 的判据."""
        return tuple(item for item in self.findings if item.state == STATE_FAIL)

    @property
    def warnings(self) -> tuple[HealthFinding, ...]:
        """全部 ``warn`` 的判据."""
        return tuple(item for item in self.findings if item.state == STATE_WARN)

    def by_check(self) -> dict[str, HealthFinding]:
        """``检查项 → 判据`` 的查表."""
        return {finding.check: finding for finding in self.findings}

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含退出码与三态计数）."""
        return {
            "state": self.state,
            "state_description": HEALTH_STATE_DESCRIPTIONS[self.state],
            "ok": self.ok,
            "exit_code": self.exit_code,
            "checked_at": self.checked_at,
            "counts": {
                state: sum(1 for item in self.findings if item.state == state)
                for state in HEALTH_STATES
            },
            "findings": [item.to_dict() for item in self.findings],
        }

    def summary_line(self) -> str:
        """一行说明：``健康 warn（ok 3 / warn 1 / fail 0）| 探针退出码 0``."""
        counts = " / ".join(
            f"{state} {sum(1 for item in self.findings if item.state == state)}"
            for state in HEALTH_STATES
        )
        return f"健康 {self.state}（{counts}）| 探针退出码 {self.exit_code}"


# --------------------------------------------------------------------------- #
# 监控：11 个指标 + 采样 + 序列
# --------------------------------------------------------------------------- #

#: 语料来源总数.
METRIC_SOURCES_TOTAL = "sources_total"

#: 本次变化的来源数（增 + 改 + 删）.
METRIC_SOURCES_CHANGED = "sources_changed"

#: 本次写进向量库的块数（**构建器的账**，与文件级的 sources_changed 不同粒度）.
METRIC_CHUNKS_WRITTEN = "chunks_written"

#: 一次同步的墙钟耗时.
METRIC_SYNC_SECONDS = "sync_seconds"

#: 距上次成功同步多久（分钟）.
METRIC_SYNC_AGE_MINUTES = "sync_age_minutes"

#: 向量库里的记录条数.
METRIC_INDEX_RECORDS = "index_records"

#: 连续失败次数.
METRIC_RUN_FAILURES = "run_failures"

#: 最近一次 RAG 评估的检索召回率（day071 的 11 个指标里最常看的一个）.
METRIC_QUALITY_RECALL = "quality_retrieval_recall"

#: 最近一次 RAG 评估的接地率.
METRIC_QUALITY_GROUNDED = "quality_grounded_rate"

#: 最近一次 RAG 评估的幻觉率.
METRIC_QUALITY_HALLUCINATION = "quality_hallucination_rate"

#: 最近一次 RAG 评估的坏例率.
METRIC_QUALITY_BAD_CASE = "quality_bad_case_rate"

#: 11 个监控指标（顺序 = 报告里的排列 = 从"语料"到"质量"）.
OPS_METRICS: tuple[str, ...] = (
    METRIC_SOURCES_TOTAL,
    METRIC_SOURCES_CHANGED,
    METRIC_CHUNKS_WRITTEN,
    METRIC_SYNC_SECONDS,
    METRIC_SYNC_AGE_MINUTES,
    METRIC_INDEX_RECORDS,
    METRIC_RUN_FAILURES,
    METRIC_QUALITY_RECALL,
    METRIC_QUALITY_GROUNDED,
    METRIC_QUALITY_HALLUCINATION,
    METRIC_QUALITY_BAD_CASE,
)

#: 单位（**由指标名决定，不由调用方填**）——单位写错的图表比没有图表更容易误导人.
METRIC_UNIT_COUNT = "count"
METRIC_UNIT_SECONDS = "seconds"
METRIC_UNIT_MINUTES = "minutes"
METRIC_UNIT_RATIO = "ratio"

#: 四种单位.
METRIC_UNITS: tuple[str, ...] = (
    METRIC_UNIT_COUNT,
    METRIC_UNIT_SECONDS,
    METRIC_UNIT_MINUTES,
    METRIC_UNIT_RATIO,
)

#: 每个指标的单位表.
OPS_METRIC_UNITS: dict[str, str] = {
    METRIC_SOURCES_TOTAL: METRIC_UNIT_COUNT,
    METRIC_SOURCES_CHANGED: METRIC_UNIT_COUNT,
    METRIC_CHUNKS_WRITTEN: METRIC_UNIT_COUNT,
    METRIC_SYNC_SECONDS: METRIC_UNIT_SECONDS,
    METRIC_SYNC_AGE_MINUTES: METRIC_UNIT_MINUTES,
    METRIC_INDEX_RECORDS: METRIC_UNIT_COUNT,
    METRIC_RUN_FAILURES: METRIC_UNIT_COUNT,
    METRIC_QUALITY_RECALL: METRIC_UNIT_RATIO,
    METRIC_QUALITY_GROUNDED: METRIC_UNIT_RATIO,
    METRIC_QUALITY_HALLUCINATION: METRIC_UNIT_RATIO,
    METRIC_QUALITY_BAD_CASE: METRIC_UNIT_RATIO,
}

#: 每个指标的一句话说明（进报告与端点）.
OPS_METRIC_DESCRIPTIONS: dict[str, str] = {
    METRIC_SOURCES_TOTAL: "语料来源总数（这次扫盘读进来的份数）",
    METRIC_SOURCES_CHANGED: "本次变化的来源数（增 + 改 + 删，文件级的账）",
    METRIC_CHUNKS_WRITTEN: "本次写进向量库的块数（块级的账，通常与上一项不同量级）",
    METRIC_SYNC_SECONDS: "一次同步的墙钟耗时（秒）",
    METRIC_SYNC_AGE_MINUTES: "距上次成功同步的分钟数（没有成功过时记 0 并留一条 note）",
    METRIC_INDEX_RECORDS: "向量库里的记录条数（= 可被检索的块数）",
    METRIC_RUN_FAILURES: "连续失败次数（退避策略的输入，也为告警提供输入）",
    METRIC_QUALITY_RECALL: "最近一次 RAG 评估的检索召回率（day071 的 retrieval_recall）",
    METRIC_QUALITY_GROUNDED: "最近一次 RAG 评估的接地率（答案里有可核对引用的比例）",
    METRIC_QUALITY_HALLUCINATION: "最近一次 RAG 评估的幻觉率（越低越好）",
    METRIC_QUALITY_BAD_CASE: "最近一次 RAG 评估的坏例率（越低越好）",
}

if not (
    set(OPS_METRICS)
    == set(OPS_METRIC_UNITS)
    == set(OPS_METRIC_DESCRIPTIONS)
):
    raise MetricsError(
        "监控指标的三张表不一致：OPS_METRICS / OPS_METRIC_UNITS / OPS_METRIC_DESCRIPTIONS "
        "必须逐键对齐——少一个键的指标会静默地不进汇总表，"
        "而'没进汇总'与'一直在正常范围'在读的时候长得一样。"
    )
if not set(OPS_METRIC_UNITS.values()) <= set(METRIC_UNITS):
    raise MetricsError(
        f"指标单位表里出现了不认识的值：{sorted(set(OPS_METRIC_UNITS.values()))}，"
        f"可用取值 {list(METRIC_UNITS)}。"
    )


@dataclass(frozen=True)
class MetricSample:
    """一个监控采样：指标名 + 值 + 时刻 + 一组标签.

    ``unit`` **不在这里**：它由指标名决定（``OPS_METRIC_UNITS``），
    因此同一个指标不可能在两次采样里带着两个单位——
    而那种事在图表上的表现是"这条线昨天是毫秒、今天是秒"。
    """

    name: str
    value: float
    at: str
    labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.name not in OPS_METRICS:
            raise MetricsError(
                f"不认识的指标名 {self.name!r}：可用取值 {list(OPS_METRICS)}。"
                "自定义名字不会进汇总表，而'没进汇总'与'一直正常'一样看不见。"
            )
        value = float(self.value)
        if not math.isfinite(value):
            raise MetricsError(
                f"指标 {self.name!r} 的值必须是有限数，收到 {self.value!r}："
                "NaN / inf 进了汇总会让 min / mean 一起变成 NaN，读的人只会看到一片空。"
            )
        parse_iso(self.at)
        labels = tuple(sorted((str(k), str(v)) for k, v in self.labels))
        for key, label in labels:
            if not key or "=" in key or "\n" in key:
                raise MetricsError(f"标签名非法：{key!r}（不能为空、不能含 '=' 或换行）。")
            if "\n" in label:
                raise MetricsError(f"标签值不能含换行：{label!r}。")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "value", value)

    @property
    def unit(self) -> str:
        """单位（由指标名查表得到）."""
        return OPS_METRIC_UNITS[self.name]

    @property
    def direction(self) -> str:
        """"越大越好 / 越小越好 / 只做描述"——**监控层不判方向，只标出来**.

        只有六个指标有方向，其余五个是 ``neutral``（描述性）：

        ```text
        lower_better   同步年龄 / 连续失败数 / 幻觉率 / 坏例率（越小越好）
        higher_better  检索召回率 / 接地率（越大越好）
        neutral        语料份数 / 变化数 / 写入块数 / 同步耗时 / 库记录数
        ```

        语料份数与库记录数刻意**不给方向**：它们多可能是内容更全，
        也可能是有人在灌噪声——把"多就是好"写进监控，等于给一条
        迟早会被人手动忽略的告警线。这一层不做阈值判定：
        阈值属于告警规则，而告警规则应该由运维显式设定（与 day071 的质量门禁同源）。
        """
        if self.name in (
            METRIC_SYNC_AGE_MINUTES,
            METRIC_RUN_FAILURES,
            METRIC_QUALITY_HALLUCINATION,
            METRIC_QUALITY_BAD_CASE,
        ):
            return "lower_better"
        if self.name in (METRIC_QUALITY_RECALL, METRIC_QUALITY_GROUNDED):
            return "higher_better"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含单位、说明与方向）."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "direction": self.direction,
            "description": OPS_METRIC_DESCRIPTIONS[self.name],
            "at": self.at,
            "labels": dict(self.labels),
        }

    def summary_line(self) -> str:
        """一行说明：``sources_total = 3 count``."""
        return f"{self.name} = {self.value} {self.unit}"


@dataclass(frozen=True)
class MetricSeries:
    """一条指标的时间序列（**同一指标名、按时间非降序**）.

    两条校验都不是形式主义：混了两个指标的序列会让 min / mean 变成
    两个东西的平均（day071 的 NDCG 混口径是同一种错），
    时间倒序会让"最新值"变成"最后一个元素"而不是"最新的那个"。
    """

    name: str
    samples: tuple[MetricSample, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in OPS_METRICS:
            raise MetricsError(f"不认识的指标名 {self.name!r}：可用取值 {list(OPS_METRICS)}。")
        previous = ""
        for sample in self.samples:
            if sample.name != self.name:
                raise MetricsError(
                    f"序列 {self.name!r} 里混进了指标 {sample.name!r} 的采样："
                    "两个指标混在一个序列里，均值会变成两个东西的平均。"
                )
            if previous and sample.at < previous:
                raise MetricsError(
                    f"序列 {self.name!r} 的时间倒序：{previous} 之后出现了 {sample.at}——"
                    "'最新值'必须真的是最新的那个。"
                )
            previous = sample.at

    @property
    def count(self) -> int:
        """采样个数."""
        return len(self.samples)

    @property
    def latest(self) -> MetricSample | None:
        """最新一个采样；一个都没有时返回 ``None``."""
        return self.samples[-1] if self.samples else None

    @property
    def values(self) -> tuple[float, ...]:
        """全部值（按时间顺序）."""
        return tuple(sample.value for sample in self.samples)

    def summary(self) -> dict[str, Any]:
        """一段汇总：``count / min / max / mean / latest``（空序列时全为 ``None``）."""
        values = self.values
        if not values:
            return {
                "name": self.name,
                "unit": OPS_METRIC_UNITS[self.name],
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "latest": None,
            }
        latest = self.samples[-1]
        return {
            "name": self.name,
            "unit": OPS_METRIC_UNITS[self.name],
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 6),
            "latest": latest.value,
            "latest_at": latest.at,
        }

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（汇总 + 全部采样）."""
        return {
            "summary": self.summary(),
            "samples": [sample.to_dict() for sample in self.samples],
        }


# --------------------------------------------------------------------------- #
# 运行状态与一次运行的账
# --------------------------------------------------------------------------- #

#: 没到点，本次没跑（**不是失败**）.
STATUS_NOT_DUE = "not_due"

#: 到点了、但语料一份都没变（因此一行都没写）.
STATUS_NO_CHANGE = "no_change"

#: 真的同步了（至少一份来源新增/更新/删除）.
STATUS_SYNCED = "synced"

#: 跑失败了（水位不动，原因留在账本与报告里）.
STATUS_FAILED = "failed"

#: 四种运行状态.
OPS_STATUSES: tuple[str, ...] = (
    STATUS_NOT_DUE,
    STATUS_NO_CHANGE,
    STATUS_SYNCED,
    STATUS_FAILED,
)

#: 每一种状态的一句话解释.
OPS_STATUS_DESCRIPTIONS: dict[str, str] = {
    STATUS_NOT_DUE: "未到点：调度判定为不跑，本次没有产生任何编码成本",
    STATUS_NO_CHANGE: "无变化：到点了但语料一份没变，本次不重建索引",
    STATUS_SYNCED: "已同步：至少一份来源发生了变化，索引已重建",
    STATUS_FAILED: "失败：本次没跑成，水位未前进（下一趟会重试同一批来源）",
}

if set(OPS_STATUSES) != set(OPS_STATUS_DESCRIPTIONS):
    raise SyncError(
        "运行状态的两张表不一致：OPS_STATUSES / OPS_STATUS_DESCRIPTIONS 必须逐键对齐，"
        "否则某种状态在报告里只有名字、没有解释。"
    )


@dataclass(frozen=True)
class OpsReport:
    """一次运行的账：状态 + 调度判定 + 差集 + 构建报告 + 健康 + 采样.

    三处"必须有 / 必须没有"的搭配关系在构造期就被校验：

    ```text
    status == synced       → index 必须有（"同步了"却拿不出构建报告）
    status == failed       → error 必须非空（只有"失败了"没有原因）
    status == not_due      → index 必须没有（没跑却带着一份构建报告）
    ```

    它们防的是同一类错误：**报告与事实不符**。一份写着"已同步"却没有任何
    构建记录的报告，与一份真的同步了的报告，在摘要行上长得一样。
    """

    started_at: str
    finished_at: str
    status: str
    schedule: ScheduleDecision
    plan: SyncPlan | None = None
    index: dict[str, Any] | None = None
    health: HealthReport | None = None
    metrics: tuple[MetricSample, ...] = ()
    notes: tuple[str, ...] = ()
    error: str = ""
    seconds: float = 0.0
    version: int = OPS_REPORT_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != OPS_REPORT_VERSION:
            raise SyncError(
                f"运行报告格式版本 {self.version} 不认识（本版本只认 {OPS_REPORT_VERSION}）。"
            )
        parse_iso(self.started_at)
        parse_iso(self.finished_at)
        if self.status not in OPS_STATUSES:
            raise SyncError(
                f"不认识的运行状态 {self.status!r}：可用取值 {list(OPS_STATUSES)}。"
            )
        if self.seconds < 0:
            raise SyncError(f"seconds 不能为负：{self.seconds}。")
        if self.status == STATUS_SYNCED and self.index is None:
            raise SyncError(
                "状态是 synced，但报告里没有构建报告："
                "'同步了'与'有构建记录'必须同时成立，否则这份报告无法被复核。"
            )
        if self.status == STATUS_FAILED and not self.error:
            raise SyncError("状态是 failed，但 error 是空的：报告里只剩'失败了'、没有原因。")
        if self.status == STATUS_NOT_DUE and self.index is not None:
            raise SyncError(
                "状态是 not_due，却带着一份构建报告：没跑就不该有构建产物——"
                "这种报告会让人以为索引刚被更新过。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        object.__setattr__(self, "metrics", tuple(self.metrics))

    @property
    def changed(self) -> int:
        """本次变化的来源数（没有差集时 0）."""
        return self.plan.changed_count if self.plan is not None else 0

    @property
    def healthy(self) -> bool:
        """健康结论（没有体检时**不算**健康）.

        "没体检"被记成不健康，与 day071 的"不下结论 ≠ 通过"是同一条纪律：
        报告摘要里不许出现一个"看起来没问题"的空洞。
        """
        return self.health is not None and self.health.ok

    def metric(self, name: str) -> float | None:
        """取某个采样值（没有这个指标时返回 ``None``）."""
        for sample in self.metrics:
            if sample.name == name:
                return sample.value
        return None

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含状态说明与全部四块）."""
        return {
            "version": self.version,
            "status": self.status,
            "status_description": OPS_STATUS_DESCRIPTIONS[self.status],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seconds": self.seconds,
            "schedule": self.schedule.to_dict(),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "index": dict(self.index) if self.index is not None else None,
            "health": self.health.to_dict() if self.health is not None else None,
            "metrics": [sample.to_dict() for sample in self.metrics],
            "notes": list(self.notes),
            "error": self.error,
        }

    def summary_line(self) -> str:
        """一行说明：``[synced] 3 份来源（+1 ~0 -0）| 健康 ok | 0.42s``."""
        plan = self.plan.summary_line() if self.plan is not None else "（无差集）"
        health = self.health.state if self.health is not None else "未体检"
        return f"[{self.status}] {plan} | 健康 {health} | {self.seconds:.2f}s"

    def save(self, path: str | Path) -> Path:
        """把报告写成一份 JSON（自动建父目录），返回落盘路径."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


__all__ = [
    "ACTION_ADDED",
    "ACTION_REMOVED",
    "ACTION_UNCHANGED",
    "ACTION_UPDATED",
    "CHECK_CORPUS_FRESHNESS",
    "CHECK_INDEX_INTEGRITY",
    "CHECK_QUALITY_REGRESSION",
    "CHECK_VECTOR_STORE",
    "FINGERPRINT_LENGTH",
    "HEALTH_CHECKS",
    "HEALTH_CHECK_DESCRIPTIONS",
    "HEALTH_STATES",
    "HEALTH_STATE_DESCRIPTIONS",
    "HEALTH_STATE_EXIT_CODES",
    "HEALTH_STATE_RANK",
    "LEDGER_VERSION",
    "METRIC_CHUNKS_WRITTEN",
    "METRIC_INDEX_RECORDS",
    "METRIC_QUALITY_BAD_CASE",
    "METRIC_QUALITY_GROUNDED",
    "METRIC_QUALITY_HALLUCINATION",
    "METRIC_QUALITY_RECALL",
    "METRIC_RUN_FAILURES",
    "METRIC_SOURCES_CHANGED",
    "METRIC_SOURCES_TOTAL",
    "METRIC_SYNC_AGE_MINUTES",
    "METRIC_SYNC_SECONDS",
    "METRIC_UNITS",
    "METRIC_UNIT_COUNT",
    "METRIC_UNIT_MINUTES",
    "METRIC_UNIT_RATIO",
    "METRIC_UNIT_SECONDS",
    "MODE_BACKOFF",
    "MODE_DUE",
    "MODE_FIRST_RUN",
    "MODE_FORCED",
    "MODE_NOT_DUE",
    "OPS_METRICS",
    "OPS_METRIC_DESCRIPTIONS",
    "OPS_METRIC_UNITS",
    "OPS_REPORT_VERSION",
    "OPS_STATUSES",
    "OPS_STATUS_DESCRIPTIONS",
    "SCHEDULE_MODES",
    "SCHEDULE_MODE_DESCRIPTIONS",
    "SNAPSHOT_ID_LENGTH",
    "SNAPSHOT_VERSION",
    "STATE_FAIL",
    "STATE_OK",
    "STATE_WARN",
    "STATUS_FAILED",
    "STATUS_NOT_DUE",
    "STATUS_NO_CHANGE",
    "STATUS_SYNCED",
    "SYNC_ACTIONS",
    "SYNC_ACTION_DESCRIPTIONS",
    "SYNC_ACTION_IS_CHANGE",
    "CorpusSnapshot",
    "HealthFinding",
    "HealthReport",
    "MetricSample",
    "MetricSeries",
    "OpsReport",
    "ScheduleDecision",
    "SchedulePolicy",
    "SourceEntry",
    "SyncLedger",
    "SyncPlan",
    "minutes_between",
    "parse_iso",
    "shift_iso",
    "snapshot_from_entries",
    "snapshot_id",
    "to_iso",
    "utc_now",
    "worst_state",
]
