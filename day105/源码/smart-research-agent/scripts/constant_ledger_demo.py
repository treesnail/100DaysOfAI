"""``constant_ledger`` 的演示脚本（day105）.

跑法（在 ``smart-research-agent`` 目录下）：

```bash
python scripts/constant_ledger_demo.py
```

它把这一课的七节读数按节打印出来，并把同一份文本写进
``outputs/constant_ledger_demo.txt``。**所有读数都是现场算的**，
没有任何一行是手写的——因此它给出的每个数字都能被复算。

为避免打印出超长的字面量（如某些包里成对的 ``dict`` 常量），
同名表与冲突表只印**名字 / 关系 / 成员数**，冲突表再把每一行截到 120 个字符。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 这样 `python scripts/constant_ledger_demo.py` 也能 import 到本包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.constant_ledger import errors as ledger_errors  # noqa: E402
from smart_research_agent.constant_ledger import ledger as ledger_module  # noqa: E402
from smart_research_agent.constant_ledger import parse as parse_module  # noqa: E402
from smart_research_agent.constant_ledger import study  # noqa: E402
from smart_research_agent.constant_ledger import types as ledger_types  # noqa: E402
from smart_research_agent.constant_ledger import verify  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "constant_ledger_demo.txt"

#: 一行最多印多少字符（只影响演示脚本的排版，不影响任何读数）.
LINE_LIMIT = 120


def _collect() -> tuple[str, ...]:
    """把七节收成一份文本（与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("========== day105 · 常量台账 ==========")

    # 1. 扫描
    lines.append("")
    lines.append("[1] 扫描：``smart_research_agent/**/*.py`` 里模块级的 UPPER_CASE 赋值")
    scanned = parse_module.scan_all()
    lines.append(f"    模块 {len(scanned)} 个 | 重复赋值的模块 {len(parse_module.duplicate_assignments(scanned))} 个")

    # 2. 台账
    lines.append("")
    lines.append("[2] 台账：模块 / 常量 / 可取值 / 读不出 / 同名 / 冲突")
    ledger = ledger_module.build_ledger(scanned)
    lines.append("    " + ledger.line())
    for kind, count in ledger.value_kind_counts():
        lines.append(f"      {kind:<9}：{count} 条")

    # 3. 关系分布
    lines.append("")
    lines.append("[3] 四类同名关系（unique / consistent / conflict / incomparable）")
    for line in study.relation_lines(ledger):
        lines.append("      " + line)

    # 4. 同名表
    lines.append("")
    lines.append("[4] 同名表：被两个以上模块钉住的名字（前 8 行，只印名字 / 关系 / 成员数）")
    for row in study.shared_rows(ledger, limit=8):
        lines.append(f"      {row.name:<34} | {row.relation:<12} | {row.count:>3} 个模块")
    lines.append(f"    跨模块同名 {len(ledger.shared_groups())} 组 / 常量总数 {ledger.constant_count()}")

    # 5. 冲突表
    lines.append("")
    lines.append("[5] 冲突表：同名不同值——把每一组逐个点名（前 8 行，每行截到 120 字符）")
    for row in study.conflict_rows(ledger)[:8]:
        text = row.line()
        lines.append("      " + (text[:LINE_LIMIT] + "…" if len(text) > LINE_LIMIT else text))
    lines.append(f"    冲突 {len(ledger.conflicts())} 组 / 无法比较 {len(ledger.incomparables())} 组")

    # 6. 性质
    lines.append("")
    lines.append("[6] 性质：七条性质、三类判据")
    report = verify.check_all(ledger=ledger)
    for row in study.property_rows(report):
        lines.append("    " + row.line())
    lines.append(f"    合计：{sum(1 for o in report.outcomes if o.passed)}/{len(report.outcomes)} 条通过")

    # 7. 结论
    lines.append("")
    lines.append("[7] 结论：这份台账是一份**可复算**的中间物——")
    lines.append(f"    同一批文件两次构建，差异项 {ledger.diff_count(ledger)} 项（0 = 逐位相同）")
    lines.append(f"    摘要 {ledger.digest()}")
    lines.append(f"    回来/缺席的族：{ledger_errors.RETURNED_FAMILY} / {ledger_errors.ABSENT_FAMILY}")
    lines.append(f"    边界第 3 条：{ledger_types.CONSTANT_LEDGER_BOUNDARIES[2][:30]}……")
    return tuple(lines)


def main() -> None:
    """跑一遍并把文本写进 outputs/。"""
    text = "\n".join(_collect()) + "\n"
    print(text, end="")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT}）")


if __name__ == "__main__":  # pragma: no cover - 脚本入口
    main()
