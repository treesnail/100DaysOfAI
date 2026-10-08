"""``failure_ledger`` 的演示脚本（day102）.

跑法（在 ``smart-research-agent`` 目录下）：

```bash
python scripts/failure_ledger_demo.py
```

它把这一课的四张表按节打印出来，并把同一份文本写进
``outputs/failure_ledger_demo.txt``。**所有读数都是现场算的**，
没有任何一行是手写的——因此它给出的每个数字都能被复算。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 这样 `python scripts/failure_ledger_demo.py` 也能 import 到本包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.failure_ledger import errors as ledger_errors  # noqa: E402
from smart_research_agent.failure_ledger import ledger as ledger_module  # noqa: E402
from smart_research_agent.failure_ledger import scan as scan_module  # noqa: E402
from smart_research_agent.failure_ledger import study  # noqa: E402
from smart_research_agent.failure_ledger import verify  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "failure_ledger_demo.txt"


def _collect() -> tuple[str, ...]:
    """把四张表收成一份文本（与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("========== day102 · 失败族台账 ==========")

    # 1. 扫描
    lines.append("")
    lines.append("[1] 扫描：``smart_research_agent/*/errors.py``（只用 ast，不做 import）")
    modules = scan_module.scan_all()
    lines.append(f"    模块 {len(modules)} 份")
    lines.append(f"    没有 errors.py 的子包 {len(scan_module.packages_without_errors())} 个")
    lines.append("    " + modules[0].line())

    # 2. 归属
    lines.append("")
    lines.append("[2] 归属：把每个基类名归到四类之一（local / builtin / imported / unresolved）")
    ledger = ledger_module.build_ledger(modules)
    for row in study.base_kind_rows(ledger):
        lines.append("    " + row.line())

    # 3. 台账
    lines.append("")
    lines.append("[3] 台账：一个模块恰好一个根；``包.名`` 唯一")
    lines.append("    " + ledger.line())
    lines.append(f"    根 {len(ledger.roots)} 个 / 派生 {len(ledger.subs)} 个 / 最大深度 {ledger.max_depth()}")
    for family in ledger.families[:6]:
        lines.append("    " + family.line())

    # 4. 性质
    lines.append("")
    lines.append("[4] 性质：七条性质、三类判据")
    report = verify.check_all(ledger=ledger)
    for row in study.property_rows(report):
        lines.append("    " + row.line())
    lines.append(f"    合计：{sum(1 for o in report.outcomes if o.passed)}/{len(report.outcomes)} 条通过")

    # 5. 结论
    lines.append("")
    lines.append("[5] 结论：这份台账是一份**可复算**的中间物——")
    lines.append(f"    同一批文件两次构建，差异项 {ledger.diff_count(ledger)} 项（0 = 逐位相同）")
    lines.append(f"    摘要 {ledger.digest()}")
    lines.append(f"    回来/缺席的族：{ledger_errors.RETURNED_FAMILY} / {ledger_errors.ABSENT_FAMILY}")
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
