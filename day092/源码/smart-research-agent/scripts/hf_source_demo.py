"""day085 离线演示：Transformer 源码精读 —— 把 Hugging Face 的那几个类读成可断言的行为.

十一节，全部离线、全部确定性（不需要 API Key，也不装 transformers）：

```text
1  两个模型画像：四个默认值的对照表
2  融合投影与"分头发生在投影内部"
3  加性掩码与显式掩码**逐位等价**（-1e9 与 -inf 在这里是同一个东西）
4  因果性用扰动量出来（前缀稳定）
5  七条性质与两条跨天对账（逐位 vs 容差）
6  LayerNorm 的三个 eps 默认值（同一个算子、三个默认值）
7  gelu 与 gelu_new（两个模型的默认激活）
8  四个 warper：温度 / top-k / top-p
9  四种策略在同一个玩具模型上（同一个种子）
10 贪心 vs beam（长度惩罚改变的是选哪一条前缀）
11 十二条源码阅读笔记与五条边界
```

运行方式::

    cd day085/源码/smart-research-agent
    python scripts/hf_source_demo.py

产出 ``outputs/hf_source_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.hf_source import (  # noqa: E402
    attention,
    blocks,
    generation,
    study,
    types,
    verify,
)
from smart_research_agent.hf_source.errors import FAMILY_OUTCOMES  # noqa: E402
from smart_research_agent.math_foundations.attention import masked_softmax_rows  # noqa: E402

#: 演示用的形状与样本（与测试样本同源，因此演示里的数能被测试复算）。
HIDDEN = 6
TOKENS = 4
VOCAB = 8
SEED = 7

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "hf_source_demo.txt"


def _inputs(tokens: int = TOKENS, hidden: int = HIDDEN):
    """写死的斜坡输入（**每一行非零方差**，否则 LN 与 softmax 会一起退化）."""
    return tuple(
        tuple(0.2 * (row + 1) + 0.1 * (column + 1) for column in range(hidden))
        for row in range(tokens)
    )


def _shape(name: str, heads: int = 1) -> types.SourceShape:
    """形状（因果默认值随画像走）."""
    return blocks.source_shape_of(name, hidden=HIDDEN, tokens=TOKENS, heads=heads, vocab=VOCAB)


class Report:
    """一个只负责"攒行 + 最后打印并落盘"的小工具（**不做任何计算**）."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, index: int, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"== {index}. {title}")

    def add(self, *texts: str) -> None:
        self.lines.extend(texts)

    def flush(self) -> None:
        text = "\n".join(self.lines)
        print(text)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text + "\n", encoding="utf-8")


def section_profiles(report: Report) -> None:
    """第 1 节：两个模型的四个默认值（**这就是"读源码"读出来的东西**）."""
    report.section(1, "两个模型画像：四个默认值的对照表")
    report.add(f"阅读对象：{types.SOURCE_LIBRARY} {types.SOURCE_VERSION}（{types.SOURCE_VERSION_SAMPLE}）")
    for name in ("gpt2", "bert"):
        profile = types.PROFILES[name]
        report.add(
            f"  {name:<5} | 摆放 {profile.norm_placement:<4} | eps {profile.ln_eps:g} | "
            f"激活 {profile.activation:<8} | 融合QKV {str(profile.fused_qkv):<5} | "
            f"嵌入后 LN {str(profile.normalize_embeddings):<5} | token_type {profile.uses_token_type}"
        )
        report.add(f"          {profile.source_file}")
    report.add(f"  三个 eps 默认值：{types.LN_EPS_DEFAULTS}")


def section_fused(report: Report) -> None:
    """第 2 节：融合投影切回去必须逐位相同（GPT-2 只写一次、BERT 写三次）."""
    report.section(2, "融合投影与「分头发生在投影内部」")
    _block, params, _shape_obj = blocks.make_hf_block("gpt2", hidden=HIDDEN, tokens=TOKENS)
    fused = attention.fused_weight(params.w_query, params.w_key, params.w_value)
    back = attention.split_fused(fused, HIDDEN)
    report.add(f"  c_attn.weight 的形状   {len(fused)} × {len(fused[0])} = 3·hidden × hidden")
    report.add(f"  切回后与三个独立投影逐位相同：{back == (params.w_query, params.w_key, params.w_value)}")
    for heads in (1, 2, 3):
        matrix = _inputs()
        restored = attention.merge_heads(attention.split_heads(matrix, heads))
        report.add(
            f"  heads={heads}：head_dim={HIDDEN // heads}，split→merge 逐位还原：{restored == matrix}"
        )


def section_additive_mask(report: Report) -> None:
    """第 3 节：加性掩码与显式掩码**逐位等价**（这是 -1e9 与 -inf 的等价证明）."""
    report.section(3, "加性掩码与显式掩码逐位等价（-1e9 与 -inf 在这里是同一个东西）")
    scores = _inputs()
    mask = tuple(
        tuple(column <= row for column in range(len(scores[0]))) for row in range(len(scores))
    )
    additive = attention.row_softmax(scores, attention.additive_bias_of(mask))
    explicit = masked_softmax_rows(scores, mask)
    gap = blocks.max_abs_gap(additive, explicit)
    report.add(f"  floor = {attention.BIAS_FLOOR:g}（HF 在 fp32 下用 -inf、fp16 下用 finfo.min）")
    report.add(f"  math.exp(-1e9) = {math.exp(-1e9)!r}（**精确下溢到 0**）")
    report.add(f"  两处权重的最大差：{gap:.6e}（逐位相同：{additive == explicit}）")
    report.add(f"  被挡格子共 {sum(1 for i, row in enumerate(mask) for j, ok in enumerate(row) if not ok)} 个，最大读数 {max(abs(additive[i][j]) for i, row in enumerate(mask) for j, ok in enumerate(row) if not ok):.6e}")


def section_causal_probe(report: Report) -> None:
    """第 4 节：因果性用扰动量出来（**不是读 `causal=True` 四个字**）."""
    report.section(4, "因果性用扰动量出来（前缀稳定）")
    _block, params, _shape_obj = blocks.make_hf_block("gpt2", hidden=HIDDEN, tokens=TOKENS)
    for name, causal in (("gpt2（因果）", True), ("bert（双向）", False)):
        shape = types.SourceShape(
            tokens=TOKENS, hidden=HIDDEN, heads=1, vocab=VOCAB, causal_default=causal
        )
        before = attention.hf_attention(params, _inputs(), shape, causal=causal)
        perturbed = tuple(_inputs()[:-1]) + (
            tuple(value * verify.PERTURBATION for value in _inputs()[-1]),
        )
        after = attention.hf_attention(params, perturbed, shape, causal=causal)
        stable = sum(
            1
            for a, b in zip(before.output[:-1], after.output[:-1])
            if a == b
        )
        report.add(
            f"  {name}：前 {TOKENS - 1} 行逐位不变 {stable}/{TOKENS - 1}；"
            f"最后一行变了 {before.output[-1] != after.output[-1]}"
        )


def section_properties(report: Report) -> None:
    """第 5 节：七条性质与两条跨天对账（**逐位与容差分开写**）."""
    report.section(5, "七条性质与两条跨天对账（逐位 vs 容差）")
    block_params, params, _shape_obj = blocks.make_hf_block("gpt2", hidden=HIDDEN, tokens=TOKENS)
    report_obj = verify.check_all(
        params, _inputs(), _shape("gpt2"), block_params=block_params, activation="relu"
    )
    report.add(*report_obj.lines())
    report.add(f"  全绿：{report_obj.ok} | 适用 {len(report_obj.applicable)}/{len(report_obj.outcomes)}")
    report.add("  " + verify.scale_agreement_line(HIDDEN // 2))


def section_layer_norm(report: Report) -> None:
    """第 6 节：三个 eps 默认值（**同一个算子、三个默认值**）."""
    report.section(6, "LayerNorm 的三个 eps 默认值（方差 = σ²/(σ²+eps)）")
    report.add(*[row.line() for row in study.layer_norm_study()])


def section_activation(report: Report) -> None:
    """第 7 节：gelu 与 gelu_new（两个模型的默认激活）."""
    report.section(7, "gelu 与 gelu_new：两个模型的默认激活")
    rows = study.activation_study()
    report.add(*[row.line() for row in rows])
    report.add(f"  逐点最大绝对差 {max(row.gap for row in rows):.3e}（x=0 处两式都为 0，导数都为 0.5）")


def section_warpers(report: Report) -> None:
    """第 8 节：四个 warper（温度 / top-k / top-p）."""
    report.section(8, "四个 warper：温度 / top-k / top-p")
    report.add(f"  顺序：{' → '.join(types.WARPER_ORDER)}")
    report.add(*[row.line() for row in study.filter_study()])
    report.add("")
    report.add(*[row.line() for row in study.top_k_study()])
    report.add("")
    report.add(*[row.line() for row in study.nucleus_study()])


def section_strategies(report: Report) -> None:
    """第 9 节：四种策略在同一个玩具模型上（**同一个种子**）."""
    report.section(9, "四种策略在同一个玩具模型上（同一个种子 seed=7）")
    report.add(*[row.line() for row in study.sampling_study()])
    report.add("")
    for row in study.block_study():
        report.add("  " + row.line())


def section_beam(report: Report) -> None:
    """第 10 节：贪心 vs beam（长度惩罚改变的是"选哪一条前缀"）."""
    report.section(10, "贪心 vs beam（长度惩罚改变的是「选哪一条前缀」）")
    report.add(*[row.line() for row in study.beam_study()])


def section_notes(report: Report) -> None:
    """第 11 节：十二条源码阅读笔记、五个失败族与五条边界."""
    report.section(11, "十二条源码阅读笔记、五个失败族与五条边界")
    for key in types.SOURCE_NOTES_ORDER:
        report.add(f"  [{key}] {types.SOURCE_NOTES[key]}")
    report.add("")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<16} → {outcome}")
    report.add("")
    report.add("  边界（这一课明确不承诺的事）：")
    for boundary in types.SOURCE_BOUNDARIES:
        report.add(f"    - {boundary}")


def main() -> int:
    """跑完十一节并把结果落盘（**任一节抛异常就直接失败，不留半份报告**）."""
    report = Report()
    for index, section in enumerate(
        (
            section_profiles,
            section_fused,
            section_additive_mask,
            section_causal_probe,
            section_properties,
            section_layer_norm,
            section_activation,
            section_warpers,
            section_strategies,
            section_beam,
            section_notes,
        ),
        start=1,
    ):
        before = len(report.lines)
        section(report)
        if len(report.lines) == before:
            raise AssertionError(f"第 {index} 节没有产出任何读数：一节空表是一种失败的沉默。")
    report.flush()
    print(f"\n已写出 {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
