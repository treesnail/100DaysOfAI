"""微调数据工程演示脚本（M5-D1）：采集 → 清洗 → 统计 → 切分落盘，全程离线.

用法（在项目根目录或任意位置均可）::

    python scripts/finetune_data_demo.py

产出：
  - ``data/finetune/out/train.jsonl`` 与 ``data/finetune/out/eval.jsonl``
    （由 ``dump_bundle`` 写出）；
  - 终端打印 ``FilterReport``（保留数、去重数、各拒绝原因计数）、
    ``DatasetStats``（含来源/标签分布）与 ``render_methods_table()``。

脚本只读 ``data/`` 下的既有语料，不联网、不调用任何模型：数据工程的每一步
都必须是确定的——否则"这批数据到底能不能用"就变成了掷骰子。
数据目录显式指向项目根，因此从任何工作目录启动都能得到同样的结果。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 保证 import 到的是本目录快照内的 smart_research_agent。
# 因此下面三处 import 必须写在本行之后，显式豁免 E402。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.finetune import (  # noqa: E402
    ALPACA,
    build_dataset,
    compute_stats,
    dump_bundle,
    render_methods_table,
)
from smart_research_agent.finetune.cleaner import DECISION_KEPT  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "finetune"
OUT_DIR = DATA_DIR / "out"

#: 报告里最多展开几条被拒样本（全量列出会把终端刷满，前几条已能说明问题）
MAX_SHOWN_REJECTIONS = 5


def main() -> None:
    # 1) 采集 + 清洗：默认采集器把 data/finetune 与 data/eval 下的语料一起收进来
    bundle = build_dataset(DATA_DIR)
    report = bundle.report

    print("=== 数据源（清洗后保留数） ===")
    for name, count in bundle.source_kept.items():
        print(f"{name:<20} {count:>4} 条")

    # 2) 清洗报告：丢弃了多少、为什么丢——必须是一个能印出来的数字
    print("\n=== 清洗报告 FilterReport ===")
    print(
        f"总计 {report.total} 条 | 保留 {report.kept} 条 | 丢弃 {report.rejected} 条 "
        f"| 其中重复 {report.duplicates} 条 | 保留率 {report.keep_rate:.2%}"
    )
    print("拒绝原因计数 drop_reasons:")
    for reason, count in sorted(report.drop_reasons.items()):
        print(f"  - {reason:<20} {count} 条")
    if not report.drop_reasons:
        print("  （无：本次没有样本被清洗规则拒绝）")

    rejected = [d for d in report.decisions if d.rule != DECISION_KEPT]
    if rejected:
        print(f"被拒样本示例（最多 {MAX_SHOWN_REJECTIONS} 条）:")
        for decision in rejected[:MAX_SHOWN_REJECTIONS]:
            print(f"  - [{decision.rule}] {decision.example.prompt_text[:30]} | {decision.reason}")

    # 3) 数据集画像：规模、长度分布、来源与标签分布
    stats = compute_stats(bundle.examples)
    print("\n=== 数据集画像 DatasetStats ===")
    print(stats.summary_line())
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))

    # 4) 切分并落盘：格式（alpaca）与切分参数在最后一步才决定
    paths = dump_bundle(
        bundle,
        OUT_DIR,
        fmt=ALPACA,
        eval_ratio=settings.finetune_eval_ratio,
        seed=settings.finetune_split_seed,
    )
    print("\n=== 落盘结果 ===")
    for split, path in paths.items():
        print(f"{split:<6} -> {path}")

    # 5) 方法总览：数据准备好之后，"用哪种方法训"才轮到被回答
    print("\n=== 微调方法总览 ===")
    print(render_methods_table())


if __name__ == "__main__":
    main()
