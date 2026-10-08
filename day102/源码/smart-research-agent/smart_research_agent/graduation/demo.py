"""``demo``：把一次端到端运行变成一份**能被重放的剧本**（day100 / G2-D1）.

day099 给了这条链一条判据——"同一输入两次运行逐位相同"。今天把这条判据**用起来**：
把它包装成一份交付物，一个读者拿去就能自己复跑、自己对照的东西。

```text
build_transcript(run)   →  DemoTranscript（question / 九段读数 / 汇总 / 归档行 / 摘要）
replay()                →  ReplayReport（两次重放各一份剧本，逐位比对）
```

## 一、今天最值钱的一句话

> **剧本的价值是"展示"而不是"宣称"：它把九段读数原样贴出来，
> 读者可以自己跑一遍对照，而不必相信任何一句话。**

因此 :class:`DemoTranscript` 里**没有一个字是形容词**：每一行都是
``capstone.assembly.run()`` 真的算出来的读数（第几段、成没成、读数多少、摘要多少）。

## 二、为什么剧本必须能被重放

```text
剧本要能重放    ⇒  它是"这一份产物"，任何人都能得到同一份
剧本不能重放    ⇒  它是"这一台机器这一次的输出"，别人拿到也没用
```

:func:`replay` 的做法最省力也最严格：**跑两次、逐位比**。
凡是没有被固定的量（时间戳 / uuid / 字典序 / 采样），第二次重放都会把它们露出来。
这与 :mod:`capstone.verify` 的第 ③ 条性质同源，但今天它从"一条检查"
变成了"一份交付物"——读者可以自己调用 :func:`replay` 复核。

## 三、与既有包的接缝

- **上游**：``capstone.assembly.run``（端到端链）/ ``capstone.types.ASSEMBLY_STAGES``
  （九段规格）；:mod:`graduation.types`（里程碑常量）；
- **下游**：:mod:`graduation.verify` 用它检查"剧本重放两次逐位相同"，
  :mod:`graduation.study` 用它打印剧本表。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.capstone import assembly as capstone_assembly
from smart_research_agent.capstone.types import ASSEMBLY_STAGES
from smart_research_agent.graduation.errors import DemoError, NumericError, ParameterError
from smart_research_agent.graduation.types import (
    MILESTONE_TITLE,
    MILESTONE_TOTAL_DAYS,
)

#: 剧本摘要取前多少位十六进制（与 ``capstone`` 的复算摘要同一口径）.
TRANSCRIPT_DIGEST_LENGTH = 16

#: 演示剧本落盘的文件名（**新文件**，不覆盖仓库里既有的任何文件）.
TRANSCRIPT_FILENAME = "demo.transcript.txt"


@dataclass(frozen=True)
class DemoTranscript:
    """一份演示剧本：问题 + 九段读数 + 汇总 + 归档行 + 摘要.

    ``digest`` 只覆盖**确定性字段**（问题 / 九段读数 / 汇总 / 归档行），
    因此"两份剧本逐位相同"可以被一条字符串比较直接判定。
    """

    question: str
    stage_lines: tuple[str, ...]
    summary_line: str
    milestone_line: str
    digest: str

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ParameterError("剧本的问题不能为空：一份不写清在演什么的剧本无法复核。")
        if len(self.stage_lines) != len(ASSEMBLY_STAGES):
            raise DemoError(
                f"剧本有 {len(self.stage_lines)} 段，应当是 {len(ASSEMBLY_STAGES)} 段："
                "缺段或多段的剧本读起来依然通顺，因此它必须被当成一次失败。"
            )
        if not self.summary_line or not self.milestone_line:
            raise ParameterError("剧本的汇总行与归档行都不能为空。")
        if not self.digest:
            raise DemoError("剧本必须带摘要：没有摘要的剧本无法证明'两次重放是同一份'。")

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两次重放比较它）.

        刻意不包含：时间戳、uuid、耗时、日志文本——它们逐次都不同，
        放进来会让"两次重放逐位相同"这条性质永远失败。
        """
        return (
            ("question", self.question),
            ("stages", self.stage_lines),
            ("summary", self.summary_line),
            ("milestone", self.milestone_line),
        )

    def diff_count(self, other: DemoTranscript) -> int:
        """与另一份剧本在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def is_identical_to(self, other: DemoTranscript) -> bool:
        """是否与另一份剧本逐位相同."""
        return self.comparable() == other.comparable()

    # ------------------------------------------------------------------ 打印

    def lines(self) -> tuple[str, ...]:
        """逐行文本（抬头 + 九段 + 汇总 + 归档）."""
        lines = [
            f"# 演示剧本 · {MILESTONE_TITLE}",
            f"# 问题：{self.question}",
            *self.stage_lines,
            self.summary_line,
            self.milestone_line,
            f"# 剧本摘要：{self.digest}",
        ]
        return tuple(lines)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "question": self.question,
            "digest": self.digest,
            "stages": len(self.stage_lines),
            "milestone": self.milestone_line,
            "lines": list(self.lines()),
        }


def _digest_of(comparable: tuple[Any, ...]) -> str:
    """把复算口径摘要成一段十六进制（``sha256`` 前 :data:`TRANSCRIPT_DIGEST_LENGTH` 位）."""
    return hashlib.sha256(repr(comparable).encode("utf-8")).hexdigest()[:TRANSCRIPT_DIGEST_LENGTH]


def build_transcript(run: capstone_assembly.SystemRun | None = None) -> DemoTranscript:
    """把一次端到端运行变成一份 :class:`DemoTranscript`（**确定性**）.

    ``run`` 缺省时现场跑一次 :func:`capstone.assembly.run`——
    它每次新建底座，因此两次 :func:`build_transcript` 得到同一份剧本。
    """
    resolved = capstone_assembly.run() if run is None else run
    stage_lines = tuple(record.line() for record in resolved.records)
    # 汇总行是 ``SystemRun.lines()`` 的最后一行（九段之后的那一行）。
    summary_line = resolved.lines()[-1]
    milestone_line = (
        f"里程碑：{MILESTONE_TITLE} | 第 {MILESTONE_TOTAL_DAYS} 天"
        f" | 运行摘要 {resolved.digest()} | 阶段 {len(resolved.records)} 段全通过={resolved.ok}"
    )
    comparable = (
        ("question", resolved.question),
        ("stages", stage_lines),
        ("summary", summary_line),
        ("milestone", milestone_line),
    )
    return DemoTranscript(
        question=resolved.question,
        stage_lines=stage_lines,
        summary_line=summary_line,
        milestone_line=milestone_line,
        digest=_digest_of(comparable),
    )


@dataclass(frozen=True)
class ReplayReport:
    """一次重放的结论：两份剧本 + 差异项数.

    ``identical`` 与 ``diff_count`` 由 ``__post_init__`` 当场对齐：
    一份报告里"它说逐位相同"与"它挂着差异 3 项"是最糟的状态。
    """

    first: DemoTranscript
    second: DemoTranscript

    @property
    def diff_count(self) -> int:
        """两份剧本在复算口径上差了几项（0 = 逐位相同）."""
        return self.first.diff_count(self.second)

    @property
    def identical(self) -> bool:
        """两份剧本是否逐位相同（等价于 ``diff_count == 0``）."""
        return self.diff_count == 0

    def require_identical(self) -> None:
        """重放不一致时抛 :class:`DemoError`（"拒绝交付"的那条路）."""
        if self.identical:
            return
        raise DemoError(
            f"剧本重放不一致：差异项数 {self.diff_count}——"
            "剧本里混进了未固定的量（时间 / uuid / 采样），它还不是一份产物。"
        )

    def line(self) -> str:
        """一行读数：``重放：第一次摘要 xxx / 第二次摘要 xxx | 逐位相同=True | 差异 0 项``."""
        return (
            f"重放：第一次摘要 {self.first.digest} / 第二次摘要 {self.second.digest}"
            f" | 逐位相同={self.identical} | 差异 {self.diff_count} 项"
        )

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "first": self.first.digest,
            "second": self.second.digest,
            "identical": self.identical,
            "diff_count": self.diff_count,
        }


def replay() -> ReplayReport:
    """跑两次端到端链、各生成一份剧本，返回 :class:`ReplayReport`.

    两次都**现场新建底座**（``capstone.assembly.run`` 每次新建），
    因此两份剧本从同一条起跑线出发。
    """
    first = build_transcript(capstone_assembly.run())
    second = build_transcript(capstone_assembly.run())
    return ReplayReport(first=first, second=second)


def write_transcript(directory: str | Path, transcript: DemoTranscript | None = None) -> Path:
    """把剧本写到给定目录，返回落盘路径（**不覆盖仓库里既有的任何文件**）.

    目录不存在时创建它；文件名固定为 :data:`TRANSCRIPT_FILENAME`。
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    resolved = build_transcript() if transcript is None else transcript
    path = target / TRANSCRIPT_FILENAME
    path.write_text("\n".join(resolved.lines()) + "\n", encoding="utf-8")
    return path


def require_digest(transcript: DemoTranscript, *, expected: str) -> DemoTranscript:
    """剧本摘要与期望不符时抛 :class:`NumericError`（指纹对不上时拒绝交付）.

    ``expected`` 为空串时不做检查（"只想要这份剧本"的调用点）。
    """
    if not expected:
        return transcript
    if transcript.digest != expected:
        raise NumericError(
            f"剧本摘要 {transcript.digest!r} 与期望 {expected!r} 不一致："
            "同一次运行应当给出同一个指纹，对不上说明这不是同一份产物。"
        )
    return transcript


__all__ = [
    "TRANSCRIPT_DIGEST_LENGTH",
    "TRANSCRIPT_FILENAME",
    "DemoTranscript",
    "ReplayReport",
    "build_transcript",
    "replay",
    "require_digest",
    "write_transcript",
]
