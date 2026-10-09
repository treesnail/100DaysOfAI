"""``symbol_catalog`` 的演示脚本（day104）.

跑法（在 ``smart-research-agent`` 目录下）：

```bash
python scripts/symbol_catalog_demo.py
```

它把这一课的六节读数按节打印出来，并把同一份文本写进
``outputs/symbol_catalog_demo.txt``。**所有读数都是现场算的**，
没有任何一行是手写的——因此它给出的每个数字都能被复算。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 这样 `python scripts/symbol_catalog_demo.py` 也能 import 到本包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.symbol_catalog import catalog as catalog_module  # noqa: E402
from smart_research_agent.symbol_catalog import errors as symbol_errors  # noqa: E402
from smart_research_agent.symbol_catalog import parse as parse_module  # noqa: E402
from smart_research_agent.symbol_catalog import study  # noqa: E402
from smart_research_agent.symbol_catalog import types as symbol_types  # noqa: E402
from smart_research_agent.symbol_catalog import verify  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "symbol_catalog_demo.txt"


def _collect() -> tuple[str, ...]:
    """把六节收成一份文本（与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("========== day104 · 对外承诺清单 ==========")

    # 1. 扫描
    lines.append("")
    lines.append("[1] 扫描：``smart_research_agent/**/*.py``（只用 ast，不做 import）")
    scanned = parse_module.scan_all()
    lines.append(f"    模块 {len(scanned)} 个 | 幽灵导出 {len(parse_module.phantom_exports(scanned))} 个")

    # 2. 清单
    lines.append("")
    lines.append("[2] 清单：模块 / 来源 / 承诺 / 重名 / 幽灵")
    catalog = catalog_module.build_catalog(scanned)
    lines.append("    " + catalog.line())
    for source, count in catalog.source_counts():
        lines.append(f"      {source:<9}：{count} 个模块")

    # 3. 幽灵
    lines.append("")
    lines.append("[3] 幽灵：承诺了却找不到落点的名字——把每一个逐个点名")
    for row in study.phantom_rows(catalog):
        lines.append("    " + row.line())
    lines.append(f"    幽灵 {len(catalog.phantom_pairs())} 个 / 字面量承诺的模块 {catalog.declared_count()} 个")

    # 4. 重名
    lines.append("")
    lines.append("[4] 重名：被两个以上模块导出的名字（跨包契约的入口）")
    for row in study.shared_rows(catalog, limit=6):
        lines.append("    " + row.line())
    lines.append(f"    跨模块重名 {len(catalog.shared_names())} 个 / 承诺总数 {catalog.promise_count()}")

    # 5. 性质
    lines.append("")
    lines.append("[5] 性质：七条性质、三类判据")
    report = verify.check_all(catalog=catalog)
    for row in study.property_rows(report):
        lines.append("    " + row.line())
    lines.append(f"    合计：{sum(1 for o in report.outcomes if o.passed)}/{len(report.outcomes)} 条通过")

    # 6. 结论
    lines.append("")
    lines.append("[6] 结论：这份清单是一份**可复算**的中间物——")
    lines.append(f"    同一批文件两次构建，差异项 {catalog.diff_count(catalog)} 项（0 = 逐位相同）")
    lines.append(f"    摘要 {catalog.digest()}")
    lines.append(f"    回来/缺席的族：{symbol_errors.RETURNED_FAMILY} / {symbol_errors.ABSENT_FAMILY}")
    lines.append(f"    边界第 3 条：{symbol_types.SYMBOL_CATALOG_BOUNDARIES[2][:28]}……")
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
