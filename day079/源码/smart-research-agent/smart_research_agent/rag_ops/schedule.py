"""调度：什么时候该跑、为什么、下一次在哪（M6-D10 / day072）.

本模块只有一个公开的判定函数 :func:`decide`，它是**纯函数**：
不读盘、不写盘、不看 ``settings``、不 sleep。所有"现在几点"都从参数进来，
所有输入都来自账本与策略。因此它可以在测试里被推到任意一个时刻上——
包括"跨了三个计划点才醒过来"这种需要等三天才能真实发生的情况。

```text
账本（上次尝试/上次成功/连续失败）+ 策略（间隔/抖动/退避）+ 现在时刻
        ↓ decide
判定（该跑 / 不该跑）+ 理由 + 计划中的下一次 + 滞后多久 + 错过了几次
```

## 四条规定，每一条都对应一类真实的运维事故

**1. 抖动是**算出来的**，不是 ``random`` 出来的。**

```text
用 random        → 同一次判定两次调用给出两个"下一次"（报告与端点对不上）
                 测试会 flaky（偶尔落在 0 抖动那一侧）
用 sha256(锚点, 运行序号)  → 同样输入永远同样输出，而不同实例的抖动仍然不同
```

**2. 错过窗口只补跑一次。**

```text
计划点 03:00  进程 12:00 才醒（宕机 9 小时）
不补跑               → 这一天语料没更新，而报告里只有"未到点"
补跑 N 次             → 一次"补 3 趟"会连着重建三次索引，纯属浪费
补跑一次 + 写进理由   → 下一个计划点**不动**（锚点仍然按计划算，不按实际运行算）
```

第三条是这一课最容易被写成 bug 的地方：如果下一个计划点从"本次实际运行时间"
起算，那么每次迟到都会把调度整体往后推，几天之后计划时间就会漂到凌晨三点之外。

**3. 失败之后的等待由连续失败数决定（指数退避 + 封顶）。**

退避与"下一次计划点"是**两件事**，因此它们各有各的算法：
``next_run_at`` 说的是"计划"，退避说的是"重试"。报告里两者都给，
否则"下一次在 24 小时后"会让人以为"失败要等一天才重试"。

**4. "没到点"是一个结论，不是失败。**

它的理由是具体的（"还差 812 分钟"），并且它会进报告与监控
（``rag_ops`` 的运行状态里有 ``not_due`` 一档）。
一条"今天没跑"如果没有任何留痕，与"今天忘了跑"在同一天里没法区分。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from smart_research_agent.rag_ops.errors import ScheduleError
from smart_research_agent.rag_ops.types import (
    MODE_BACKOFF,
    MODE_DUE,
    MODE_FIRST_RUN,
    MODE_FORCED,
    MODE_NOT_DUE,
    ScheduleDecision,
    SchedulePolicy,
    SyncLedger,
    parse_iso,
    to_iso,
)


def jitter_seconds(policy: SchedulePolicy, *, anchor: str, run_index: int) -> int:
    """由（锚点，运行序号）算出一个确定性的抖动（**秒**，落在 ``[0, jitter_seconds]``）.

    为什么不是 ``random.randint``：同一次判定的两次调用必须给出同一个"下一次"——
    否则端点显示的时间与报告里记的时间会不一样，而那种不一致
    会被读成"调度自己改了主意"。哈希输入的第二个分量用 ``run_index``
    而不是机器名：**可重算**（换一台机器跑同一份账本得到同一个抖动），
    与 day065 的"能重算的才叫版本号"是同一条纪律。
    """
    if policy.jitter_seconds <= 0:
        return 0
    digest = hashlib.sha256(f"{anchor}|{int(run_index)}".encode()).hexdigest()
    return int(digest[:8], 16) % (policy.jitter_seconds + 1)


def _next_run_at(
    policy: SchedulePolicy,
    *,
    anchor: str,
    run_index: int,
    intervals: int = 1,
) -> str:
    """锚点 + ``intervals`` 个间隔 + 抖动，写成 ISO 串."""
    base = parse_iso(anchor) + timedelta(minutes=policy.interval_minutes * max(1, intervals))
    offset = jitter_seconds(policy, anchor=anchor, run_index=run_index)
    return to_iso(base + timedelta(seconds=offset))


def decide(
    policy: SchedulePolicy,
    ledger: SyncLedger,
    *,
    now: datetime,
    force: bool = False,
) -> ScheduleDecision:
    """判定"这一刻该不该跑"（纯函数，见模块 docstring 的四条规定）.

    分支顺序就是优先级顺序，每一步都在这里写清"为什么排在前面"：

    ```text
    ① force        调用方说了算（"我不管间隔"）——它是唯一能绕过退避的口子
    ② 退避         上次失败过就先退避：在窗口里判定"不跑"，出了窗口才恢复计划
    ③ 首次运行     没有任何尝试记录 → 按首次运行处理（全量）
    ④ 计划点       锚点 + 间隔 + 抖动；到点即 due，并把滞后与错过次数一起报出来
    ```

    ②排在③前面的理由：一次**失败**的首次运行也写 ``last_run_at``，
    此时它已经不是"首次"了——它应该退避重试，而不是无限重试。
    """
    now_iso = to_iso(now)

    if force:
        return ScheduleDecision(
            mode=MODE_FORCED,
            due=True,
            reason="被显式要求执行（force=True）：忽略间隔与退避，立即跑一次",
            now=now_iso,
            effective_at=now_iso,
            next_run_at=_next_run_at(policy, anchor=now_iso, run_index=ledger.runs + 1),
            lag_minutes=0.0,
            missed_runs=0,
            consecutive_failures=ledger.consecutive_failures,
        )

    if ledger.consecutive_failures > 0:
        wait = policy.backoff_seconds(ledger.consecutive_failures)
        anchor = ledger.last_run_at or ledger.last_success_at
        if not anchor:
            raise ScheduleError(
                "账本记着连续失败，但既没有 last_run_at 也没有 last_success_at："
                "退避需要一个起点，缺了它只能猜——而猜出来的等待时间会让重试"
                "要么立刻发生、要么永远不发生。"
            )
        ready = parse_iso(anchor) + timedelta(seconds=wait)
        if now < ready:
            remaining = (ready - now).total_seconds()
            return ScheduleDecision(
                mode=MODE_BACKOFF,
                due=False,
                reason=(
                    f"连续失败 {ledger.consecutive_failures} 次，正在退避："
                    f"还要等 {remaining:.0f} 秒（等满 {wait} 秒）才重试，"
                    f"最近一次失败：{ledger.last_error}"
                ),
                now=now_iso,
                effective_at=to_iso(ready),
                next_run_at=to_iso(ready),
                lag_minutes=0.0,
                missed_runs=0,
                consecutive_failures=ledger.consecutive_failures,
            )
        return ScheduleDecision(
            mode=MODE_DUE,
            due=True,
            reason=(
                f"连续失败 {ledger.consecutive_failures} 次，退避窗口（{wait} 秒）已过："
                "本次按重试执行（水位未前进，因此同一批来源会被重新处理一遍，"
                "而重跑是幂等的）"
            ),
            now=now_iso,
            effective_at=to_iso(ready),
            next_run_at=_next_run_at(policy, anchor=now_iso, run_index=ledger.runs + 1),
            lag_minutes=0.0,
            missed_runs=0,
            consecutive_failures=ledger.consecutive_failures,
        )

    if not ledger.last_run_at:
        return ScheduleDecision(
            mode=MODE_FIRST_RUN,
            due=True,
            reason="首次运行：账本里没有任何尝试记录，本次按全量处理（水位为空）",
            now=now_iso,
            effective_at=now_iso,
            next_run_at=_next_run_at(policy, anchor=now_iso, run_index=ledger.runs + 1),
            lag_minutes=0.0,
            missed_runs=0,
            consecutive_failures=0,
        )

    anchor = ledger.last_run_at
    planned = _next_run_at(policy, anchor=anchor, run_index=ledger.runs)
    planned_moment = parse_iso(planned)
    if now < planned_moment:
        remaining = (planned_moment - now).total_seconds() / 60.0
        return ScheduleDecision(
            mode=MODE_NOT_DUE,
            due=False,
            reason=(
                f"未到计划时间：计划点 {planned}，还差 {remaining:.1f} 分钟"
                f"（间隔 {policy.interval_minutes} 分钟，含 ≤{policy.jitter_seconds} 秒抖动）"
            ),
            now=now_iso,
            effective_at=planned,
            next_run_at=planned,
            lag_minutes=max(0.0, -remaining),
            missed_runs=0,
            consecutive_failures=0,
        )

    lag = (now - planned_moment).total_seconds() / 60.0
    missed = int(lag // policy.interval_minutes)
    notes = [
        f"已到计划时间：计划点 {planned}，滞后 {lag:.1f} 分钟，本次正常触发",
    ]
    if missed > 0:
        notes.append(
            f"其间错过了 {missed} 个计划点（滞后超过一个间隔）：**只补跑这一次**，"
            "不连续补跑——补跑 N 次会连着重建 N 次索引，而语料只需要最新那一份"
        )
    if lag > policy.max_lag_minutes:
        notes.append(
            f"滞后 {lag:.1f} 分钟已超过 max_lag_minutes="
            f"{policy.max_lag_minutes}：本次照常执行，但请检查调度进程为什么这么久没被唤醒"
            "（这条只写进理由，不吞掉本次运行）"
        )
    return ScheduleDecision(
        mode=MODE_DUE,
        due=True,
        reason="；".join(notes),
        now=now_iso,
        effective_at=planned,
        next_run_at=_next_run_at(
            policy, anchor=anchor, run_index=ledger.runs, intervals=missed + 2
        ),
        lag_minutes=round(lag, 4),
        missed_runs=missed,
        consecutive_failures=0,
    )


def describe(
    policy: SchedulePolicy,
    ledger: SyncLedger,
    *,
    now: datetime,
    force: bool = False,
) -> dict[str, object]:
    """给端点用的一段摘要：策略 + 判定 + 账本（**不产生任何副作用**）.

    端点为什么可以直接调它：:func:`decide` 是纯函数，因此"看一下什么时候跑"
    与"跑一次"是两件完全不同的事——这正是"只读状态"与"执行动作"
    能共用同一份判定的原因。
    """
    decision = decide(policy, ledger, now=now, force=force)
    return {
        "policy": policy.to_dict(),
        "policy_summary": policy.summary_line(),
        "decision": decision.to_dict(),
        "decision_summary": decision.summary_line(),
        "ledger": ledger.describe(now=now),
    }


__all__ = [
    "decide",
    "describe",
    "jitter_seconds",
]
