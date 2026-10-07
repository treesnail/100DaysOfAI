"""定时同步容器：把"每天凌晨跑一次"变成一段可以离线推演的循环（M6-D10 / day072）.

容器编排里那个 ``rag-ops-sync`` 服务跑的就是本模块：

```bash
python -m smart_research_agent.rag_ops.worker --loop --interval 1440
```

它**不重新实现调度**：该不该跑、下一次在哪，全部由 ``schedule.decide``
（纯函数）回答；本模块只做两件事——**跑一次**、**按判定睡到下一次**。

```text
run_forever(pipeline, ...)      循环：run_once → 算出要等多久 → sleep → 再来
seconds_until_next(decision)    "还有多少秒"（带下限与上限，见下）
main(argv)                      命令行入口：--once / --loop / --interval / --force
```

## 为什么"睡多久"要夹在两个界之间

```text
下限 minimum_seconds   判定说"下一次在 0.2 秒后"时不要忙等
                       （时钟抖动、或 next_run_at 恰好等于现在，都会造成这种情况）
上限 maximum_seconds   判定说"下一次在 24 小时后"时也不要睡 24 小时不醒
                       ——进程收不到停止信号、配置改了要等到明天才生效
```

因此循环的实际行为是"**每 ``maximum_seconds`` 醒来一次，问一遍该不该跑**"。
这不是退让，而是容器里唯一正确的姿势：``schedule.decide`` 是纯函数，
多问几遍没有任何代价（它不读语料、不碰网络），而"睡得久"才是真的代价。

## 为什么 ``sleep`` 与 ``now`` 都是参数

```text
测试要能在 0 秒内推演"跨了三天"：注入一个记账用的 sleep + 一个假时钟
生产要能 Ctrl-C 立刻停下：真 sleep 会被信号打断（而假 sleep 不会）
```

这也是本模块**唯一**能被自动化测试覆盖的原因：真实的 24 小时循环
不可能进 CI，而"循环的逻辑"（跑几次、等多久、失败之后怎么退避）
恰恰是最需要被测的部分。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime

from smart_research_agent.rag_ops.errors import ScheduleError
from smart_research_agent.rag_ops.pipeline import (
    RagOpsPipeline,
    default_rag_ops_pipeline,
    policy_from_settings,
)
from smart_research_agent.rag_ops.types import (
    OpsReport,
    ScheduleDecision,
    SchedulePolicy,
    parse_iso,
    utc_now,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 循环的下限等待（秒）：小于它就不叫"睡"，叫"忙等"。
MINIMUM_WAIT_SECONDS = 1.0

#: 循环的上限等待（秒）：900 = 15 分钟。容器里"醒着但不干活"几乎不花钱，
#: 而"睡过头"会让停止信号与配置变更都要等到下一个周期才生效。
MAXIMUM_WAIT_SECONDS = 900.0


def seconds_until_next(
    decision: ScheduleDecision,
    *,
    now: datetime,
    minimum: float = MINIMUM_WAIT_SECONDS,
    maximum: float = MAXIMUM_WAIT_SECONDS,
) -> float:
    """按判定算"还要等多少秒"（夹在 ``[minimum, maximum]`` 之间）.

    ``next_run_at`` 为空串（理论上不会发生：三个分支都会填它）时返回
    ``minimum``——**宁可多问一遍，也不要睡在一个猜出来的时长上**。
    """
    if minimum < 0 or maximum < minimum:
        raise ScheduleError(
            f"等待区间非法：minimum={minimum} maximum={maximum}（要求 0 <= minimum <= maximum）"
        )
    if not decision.next_run_at:
        return minimum
    delta = (parse_iso(decision.next_run_at) - now).total_seconds()
    return float(min(max(delta, minimum), maximum))


def run_forever(
    pipeline: RagOpsPipeline,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
    max_iterations: int | None = None,
    minimum_wait: float = MINIMUM_WAIT_SECONDS,
    maximum_wait: float = MAXIMUM_WAIT_SECONDS,
    on_tick: Callable[[OpsReport], None] | None = None,
) -> int:
    """循环跑同步，返回**实际循环了几次**（``max_iterations`` 为 ``None`` 时永不返回）.

    每一轮的顺序是固定的：

    ```text
    ① pipeline.run_once(now=now())   跑一次（内部会自己判"该不该跑"）
    ② on_tick(report)                把这一趟的结论交出去（日志 / 指标 / 测试断言）
    ③ 达到 max_iterations 就结束
    ④ sleep(seconds_until_next(...)) 睡到判定给出的下一个时刻（带上下限）
    ```

    为什么第 ① 步仍然要重新判定：循环可能睡过头（机器挂起）、
    也可能被人工 ``restart``。**每次醒来重新问一遍**，是让"错过一次运行"
    与"时钟跳变"都不会被无声吞掉的唯一办法。
    """
    if max_iterations is not None and max_iterations < 0:
        raise ScheduleError(f"max_iterations 不能为负：{max_iterations}")

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        report = pipeline.run_once(now=now())
        iterations += 1
        if on_tick is not None:
            on_tick(report)
        if max_iterations is not None and iterations >= max_iterations:
            break
        wait = seconds_until_next(
            report.schedule,
            now=now(),
            minimum=minimum_wait,
            maximum=maximum_wait,
        )
        sleep(wait)
    return iterations


def build_parser() -> argparse.ArgumentParser:
    """命令行参数（容器 CMD 与本地手动跑用的是同一份）.

    ``--interval`` 会覆盖 ``rag_ops_interval_minutes``：它是这一课里唯一
    "由编排层传给进程"的调度参数（其余参数都走环境变量）。理由很直接：
    ``docker-compose.rag.yml`` 里那句 ``--interval 1440`` 是**给人看的**，
    而"每天一次"这件事在编排文件里比在一个远端配置里更容易被发现。
    """
    parser = argparse.ArgumentParser(
        prog="python -m smart_research_agent.rag_ops.worker",
        description="RAG 生产化同步 worker（day072 / M6-D10）：扫语料、增量重建、落账本",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环执行（容器里用这个）；缺省只跑一次就退出",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="调度间隔（分钟），覆盖 rag_ops_interval_minutes；缺省读配置",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="最多循环几次（0 表示不跑）。用于冒烟验证与灰度：容器里不要传它",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略调度判定，强制跑一次（等价于 POST /rag/ops/sync 的 force）",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="语料根目录，覆盖 rag_ops_source_dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：跑一次或循环跑（返回进程退出码）.

    退出码只有 0 与 1：

    ```text
    0   本次运行**正常结束**（包括 no_change / not_due / 失败但已记账）
    1   进程没能建立起来（参数非法、语料目录不存在、配置读不出来）
    ```

    "运行失败"不返回 1，这不是纵容：失败已经进了账本与报告，
    而容器如果因为一次可重试的失败就退出，编排层会不断重启它——
    真正的重试节奏应该由 ``schedule`` 的退避决定，而不是由容器生命期决定。
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    policy: SchedulePolicy = policy_from_settings()
    if args.interval is not None:
        policy = SchedulePolicy(
            interval_minutes=int(args.interval),
            jitter_seconds=policy.jitter_seconds,
            max_lag_minutes=policy.max_lag_minutes,
            backoff_base_seconds=policy.backoff_base_seconds,
            backoff_max_seconds=policy.backoff_max_seconds,
        )
    pipeline = default_rag_ops_pipeline(source_dir=args.source_dir, policy=policy)

    def report_tick(report: OpsReport) -> None:
        logger.info("rag_ops 运行结果：%s", report.summary_line())
        for note in report.notes:
            logger.info("  · %s", note)

    if not args.loop:
        report_tick(pipeline.run_once(force=args.force))
        return 0
    logger.info("rag_ops worker 启动：%s | %s", policy.summary_line(), pipeline.source_dir)
    run_forever(pipeline, max_iterations=args.max_iterations, on_tick=report_tick)
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    sys.exit(main())
