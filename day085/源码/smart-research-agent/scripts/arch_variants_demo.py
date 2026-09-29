"""``arch_variants`` 的离线演示（day082 / M7-D7）：十节，全部可复现.

用法::

    python scripts/arch_variants_demo.py

它**不联网、不需要 torch**（最后一节会在有 torch 的机器上额外跑一遍生成的脚本，
没有 torch 时那一节打印“跳过”）。十节依次是：

```text
一  三个变体的一句话      （名字 / 例子 / 掩码 / 训练目标）
二  两张结构性掩码        允许的位置对与熵天花板
三  家底                  块数、子层数、参数量（逐个数 vs 公式算）
四  实测依赖表            三个变体的 0/非 0 表（**这一课的判据**）
五  掩码的账              五个正常行 + 两个反证行
六  前缀稳定性            扰动最后一个 token，看哪几行动了
七  六条性质              通过 / 未通过 / 不适用三类读数
八  梯度校验              解析 vs 中心差分（两条流各一次）
九  生成的 PyTorch 脚本   类名 / nn 模块 / CONFIG，以及本机实测的参数量
十  一条总闸              所有判据的汇总
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.arch_variants import (  # noqa: E402
    assembly,
    probe,
    stacks,
    study,
    verify,
)
from smart_research_agent.arch_variants.types import (  # noqa: E402
    VARIANTS,
    VARIANT_ENCODER_DECODER,
)


def section(title: str) -> None:
    """打印一节的分隔行."""
    print()
    print(f"=== {title} ===")


def main() -> int:
    """跑完十节，返回 0 表示全部判据通过."""
    shape = stacks.make_variant_shape(tokens=4, hidden=6, layers=3, sources=5)
    inputs = stacks.sample_matrix(shape.tokens, shape.hidden, seed=21)
    source = stacks.sample_matrix(shape.source_length, shape.hidden, seed=17)
    target = stacks.sample_matrix(shape.tokens, shape.hidden, seed=31)
    failures: list[str] = []

    section("一 三个变体的一句话")
    print(study.study_summary(shape))
    for variant, example, mask_names, objective in study.objectives_table():
        print(f"  {variant:<16} | {example:<4} | 掩码 {mask_names:<14} | {objective}")

    section("二 两张结构性掩码")
    for kind, line in study.mask_catalogue(shape):
        ceiling = dict(study.entropy_ceiling_table(shape))[kind]
        print(f"  {kind:<7} | {line} | 熵天花板 {ceiling:.6f}")
    print(f"  因果/全开 的位置对比例：{study.mask_pair_ratio(shape):.6f}")

    section("三 家底（逐个数 vs 公式算）")
    census = study.census_study(shape)
    for line in census.table_lines():
        print(f"  {line}")
    print(f"  例子：{', '.join(census.examples)} | 两条路径全部一致：{census.ok}")

    section("四 实测依赖表（扰动每一列）")
    for variant in VARIANTS:
        params = stacks.make_variant_parameters(shape, variant)
        kwargs = {"source": source} if variant == VARIANT_ENCODER_DECODER else {}
        report = probe.dependency_matrix(params, inputs, **kwargs)
        print(f"  {variant:<16} | 可达 {report.reach_profile()} | 逐位为 0：{report.exact} | "
              f"与掩码一致：{report.matches_mask}")
        for row in report.matrix:
            print("      " + " ".join(f"{value:.6f}" for value in row))
    cross = probe.cross_dependency(
        stacks.make_variant_parameters(shape, VARIANT_ENCODER_DECODER), inputs, source
    )
    print(f"  交叉那一路（长方形 {cross.rows}×{cross.columns}）：")
    for row in cross.matrix:
        print("      " + " ".join(f"{value:.6f}" for value in row))

    section("五 掩码的账（含两个反证）")
    leaks = study.leak_study(shape)
    for line in leaks.table_lines():
        print(f"  {line}")
    print(f"  表格自评：{leaks.ok}（正常行一致、反证行如期不一致）| 逐位为 0 的行数 "
          f"{leaks.exact_rows}/{len(leaks.rows)}")
    if not leaks.ok:
        failures.append("掩码的账")

    section("六 前缀稳定性（扰动最后一个 token）")
    prefix = study.prefix_study(shape)
    for line in prefix.table_lines():
        print(f"  {line}")

    section("七 六条性质")
    for variant in VARIANTS:
        params = stacks.make_variant_parameters(shape, variant)
        kwargs = {"source": source} if variant == VARIANT_ENCODER_DECODER else {}
        report = verify.check_properties(params, inputs, **kwargs)
        print(f"  {variant} | 全绿 {report.ok} | 不适用 {report.skipped}")
        for line in report.summary_lines():
            print(f"      {line}")
        if not report.ok:
            failures.append(f"{variant} 的性质")

    section("八 梯度校验（解析 vs 中心差分）")
    for variant in VARIANTS:
        params = stacks.make_variant_parameters(shape, variant)
        kwargs = {"source": source} if variant == VARIANT_ENCODER_DECODER else {}
        report = verify.check_variant_gradients(params, inputs, target, **kwargs)
        print(f"  {variant} | 全绿 {report.ok}（容差 {report.tolerance}，步长 {report.step}）")
        for line in report.summary_lines():
            print(f"      {line}")
        if not report.ok:
            failures.append(f"{variant} 的梯度")

    section("九 生成的 PyTorch 脚本")
    for variant in VARIANTS:
        script = assembly.variant_script(shape, variant)
        facts = assembly.assembly_facts(script)
        print(f"  {variant:<16} | 类 {facts['classes'][0]} | 与形状一致 "
              f"{assembly.config_matches_shape(script, shape)}")
        print(f"      nn 模块：{', '.join(facts['modules'])}")
        measured = _run_generated(script)
        print(f"      本机实测：{measured}")

    section("十 一条总闸")
    for variant in VARIANTS:
        params = stacks.make_variant_parameters(shape, variant)
        census_row = stacks.census_of(params)
        print(f"  {variant:<16} | 参数 {census_row.parameters}（解析 "
              f"{census_row.analytic_parameters}）| 两条路径一致 "
              f"{census_row.matches_analytic}")
    print()
    if failures:
        print(f"有 {len(failures)} 项判据未通过：{', '.join(failures)}")
        return 1
    print("十节全部通过：三个变体的掩码、家底、实测依赖、性质与梯度都对上了。")
    return 0


def _run_generated(script: str) -> str:
    """在有 torch 的机器上真跑一遍生成的脚本，把参数量带回来（否则如实说明）."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return "跳过（本机没有 torch：生成侧与解析侧只做结构核对）"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "generated.py"
        path.write_text(script, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, check=False
        )
    for line in done.stdout.splitlines():
        if line.startswith("PARAM_COUNT"):
            return f"{line.strip()}（PyTorch 的真实参数量，与本包的解析式不同，差值见教程第 9 章）"
    return f"运行失败：{done.stderr.strip().splitlines()[-1] if done.stderr else '无输出'}"


if __name__ == "__main__":
    raise SystemExit(main())
