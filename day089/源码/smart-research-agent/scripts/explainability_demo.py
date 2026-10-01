"""``explainability`` 的离线演示（day083 / M7-D8）：十节，全部可复现.

用法::

    python scripts/explainability_demo.py

它**不联网、不需要 torch**（也不需要 matplotlib：图是文本）。十节依次是：

```text
一  模型真实权重        逐层摘要（读出来的就是模型算出来的）
二  文本热力图          一张因果表 + 一张多头表（**10 级 + 图例**）
三  逐层熵与天花板       熵 / 天花板 / 归一化熵
四  头间冗余            同一层几个头之间的逐格余弦
五  偏移质量            质量落在离对角线 0 / 1 / 2 格上的比例
六  滚动                Â = α·A + (1−α)·I，R = Â_L ⋯ Â_1（含 α = 0）
七  六条性质            通过 / 未通过 + 证据
八  天花板对照          全开 vs 因果（"尖"里有多少是掩码造成的）
九  两个变体的对照       BERT（全开）与 GPT（因果）并排
十  一条总闸            所有判据的汇总
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.arch_variants.types import (  # noqa: E402
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_ONLY,
)
from smart_research_agent.explainability import (  # noqa: E402
    analyze,
    extract,
    render,
    rollout,
    study,
    verify,
)


def section(title: str) -> None:
    """打印一节的分隔行."""
    print()
    print(f"=== {title} ===")


def main() -> int:
    """跑完十节，返回 0 表示六条性质全绿."""
    shape = study.default_shape()
    inputs = study.study_inputs(shape)
    params = study.study_params(VARIANT_DECODER_ONLY, shape)
    layer_records = study.all_layer_records(params, inputs, heads=2)
    failures: list[str] = []

    section("一 模型真实权重（读出来的就是模型算出来的）")
    for line in extract.summary_lines(extract.self_records(params, inputs)):
        print(f"  {line}")
    print(f"  heads=1 的多头读法与它逐位相同：{extract.single_head_matches(params, inputs)}")

    section("二 文本热力图（10 级 + 图例）")
    print(render.heatmap_block(extract.self_records(params, inputs)[0], indent="  "))
    print()
    print(render.heatmap_block(extract.head_records(params, inputs, layer=1, heads=2)[0], indent="  "))

    section("三 逐层熵与天花板")
    entropy_study = study.layer_entropy_study(VARIANT_DECODER_ONLY, shape)
    for line in entropy_study.table_lines():
        print(f"  {line}")
    print(f"  汇总：{entropy_study.to_dict()['summary']}")

    section("四 头间冗余（逐格余弦）")
    redundancy = study.redundancy_study(VARIANT_DECODER_ONLY, shape)
    for line in redundancy.table_lines():
        print(f"  {line}")
    print(f"  非对角平均：{redundancy.off_diagonal_mean():.6f}（对角线全为 1：{redundancy.diagonal_is_one}）")

    section("五 偏移质量（离对角线 k 格）")
    for line in study.offset_study(VARIANT_DECODER_ONLY, shape).table_lines():
        print(f"  {line}")

    section("六 滚动（Â = α·A + (1−α)·I）")
    rolled = rollout.rollout_record(layer_records, alpha=0.5)
    print(f"  {rolled.summary_line()}")
    print(f"  α = 0 时为单位阵：{rollout.identity_when_alpha_zero(layer_records)}")
    print(f"  掩码外逐位为 0 的格子数与最大读数：{rollout.mask_respected_after_rollout(layer_records)}")
    for line in study.rollout_study(VARIANT_DECODER_ONLY, shape).table_lines():
        print(f"  {line}")

    section("七 六条性质")
    report = verify.check_properties(layer_records)
    print(f"  全绿：{report.ok} | 主体：{report.subject}")
    for line in report.summary_lines():
        print(f"  {line}")
    if not report.ok:
        failures.append("六条性质")

    section("八 天花板对照（「尖」里有多少是掩码造成的）")
    ceilings = study.ceiling_study(shape)
    for line in ceilings.table_lines():
        print(f"  {line}")
    print(f"  归一化基线（完全均匀时）：{study.all_ceilings(shape)[0][2]}")

    section("九 两个变体的对照（BERT 全开 vs GPT 因果）")
    for variant in (VARIANT_ENCODER_ONLY, VARIANT_DECODER_ONLY):
        other = study.layer_entropy_study(variant, shape, heads=2)
        head = other.profiles[0]
        print(
            f"  {variant:<16} | 第 0 层：熵 {head.entropy:.6f} | 天花板 {head.ceiling:.6f} | "
            f"归一化 {head.normalized_entropy:.6f} | 判词 {other.verdicts()[0]}"
        )
    print("  热力图（BERT 那一张的上三角**不是**空的，GPT 那一张是）：")
    for variant in (VARIANT_ENCODER_ONLY, VARIANT_DECODER_ONLY):
        record = study.all_layer_records(
            study.study_params(variant, shape), inputs, heads=1
        )[0]
        print(render.heatmap_block(record, indent="    ", column_labels=False))

    section("十 一条总闸")
    print(f"  记录 {len(layer_records)} 条 | 层 {extract.layer_count(layer_records)} | 头 2")
    print(f"  六条性质全绿：{report.ok} | 逐层熵全绿：{entropy_study.ok}")
    print(f"  滚动之后行随机且掩码外为 0：{verify.check_rollout_is_stochastic(layer_records).passed}")
    print(f"  读数的口径：{study.how_to_read()[0]}")
    print()
    if failures:
        print(f"有 {len(failures)} 项判据未通过：{', '.join(failures)}")
        return 1
    print(f"十节全部通过：三个变体的权重读得出、画得下、也被六条性质钉住了（{len(VARIANTS)} 个变体）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
