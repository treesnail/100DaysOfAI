"""day089 离线演示：从神经元到 FFN —— 神经网络基础.

十一节，全部离线、全部确定性（不需要 API Key，也不装 transformers / torch / numpy）：

```text
1   一条链总览：神经元 → 一层 → 网络 → 损失（宽度链与参数量）
2   六个激活：口径表 + 有限性网格（含 ±1000）读数
3   值域与 softmax 行分布（开区间、行和为 1）
4   五种初始化：同一份 spec、同一个种子上的读数
5   一层与一网的账：Dense 前向、forward_trace、层次账
6   恒等塌缩：多层网络 == 合成后的单层仿射映射
7   三个损失与交叉熵的两条路径
8   跨天对账①：softmax / 交叉熵 / gelu 与既有实现逐位比
9   跨天对账②：ffn_block 与 encoder_decoder.feed_forward 逐位比
10  七条性质与五张表
11  失败族（缺席的 GradientError）与五条边界、Torch 对照表
```

运行方式::

    cd day089/源码/smart-research-agent
    python scripts/neural_basics_demo.py

产出 ``outputs/neural_basics_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.neural_basics import (  # noqa: E402
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    NEURAL_BOUNDARIES,
    TORCH_COUNTERPARTS,
    activations,
    layers,
    losses,
    network,
    study,
    types,
    verify,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "neural_basics_demo.txt"


class Report:
    """攒行 + 落盘（**不做任何计算**）."""

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


def section_chain(report: Report) -> None:
    """第 1 节：一条链总览."""
    report.section(1, "一条链总览：神经元 → 一层 → 网络 → 损失")
    neuron = types.NeuronSpec(inputs=3, activation="relu", init="zeros", seed=0)
    report.add(f"  神经元：{neuron.line()} | 前向 = {network.neuron_forward(neuron, (1.0, 2.0, 3.0))}")
    dense = types.DenseSpec(3, 4, activation="relu", init="xavier", seed=7)
    report.add(f"  一层：  {dense.line()}")
    report.add(f"         账：{layers.layer_accounting(dense)}")
    spec = study.LAYER_SAMPLE
    report.add(f"  网络：  {spec.line()}")
    report.add(f"         宽度链：{' → '.join(str(width) for width in spec.widths)}")
    report.add(f"         前向账：{network.forward_trace(spec).line()}")
    report.add(f"  损失：  mse(完美) = {losses.mse(verify.MSE_PRED, verify.MSE_PRED)}（恰好 0）")
    report.add(f"          mse(样本) = {losses.mse(verify.MSE_PRED, verify.MSE_TARGET):.6e}")


def section_activations(report: Report) -> None:
    """第 2 节：六个激活与有限性网格."""
    report.section(2, "六个激活：口径表 + 有限性网格（含 ±1000）读数")
    for row in study.activation_rows():
        report.add("  " + row.line())
    report.add(f"  有限性网格（{len(activations.FINITE_GRID)} 点）：{list(activations.FINITE_GRID)}")
    for name in types.ACTIVATIONS:
        output = activations.activate(name, activations.FINITE_GRID)
        bad = sum(1 for value in output if not _finite(value))
        report.add(f"    {name:<11} 非有限个数={bad}  min={min(output):+.6g}  max={max(output):+.6g}")
    report.add("  极端点：sigmoid(-1000)=0.0、tanh(-1000)=-1.0——这是**饱和**（定义域代价），不是 nan")


def section_ranges(report: Report) -> None:
    """第 3 节：值域与 softmax 行分布."""
    report.section(3, "值域与 softmax 行分布（开区间、行和为 1）")
    report.add(f"  值域网格（{len(activations.RANGE_GRID)} 点）：{list(activations.RANGE_GRID)}")
    for name in types.ACTIVATIONS:
        bound = activations.activation_range(name)
        output = activations.activate(name, activations.RANGE_GRID)
        violations = sum(1 for value in output if not bound.contains(value))
        report.add(f"    {name:<11} 值域 {bound.line():<12} 越界 {violations}")
    distribution = activations.softmax(activations.RANGE_GRID)
    report.add(f"  softmax 行和 = {sum(distribution):.16f}（与 1 的偏差 {abs(sum(distribution) - 1.0):.3e}）")
    report.add(f"  softmax 每项落在 (0, 1)：(min={min(distribution):.6e}, max={max(distribution):.6f})")


def section_initializations(report: Report) -> None:
    """第 4 节：五种初始化."""
    report.section(4, "五种初始化：同一份 spec、同一个种子上的读数")
    for row in study.initialization_rows():
        report.add("  " + row.line())
    report.add(f"  xavier 半宽 a = sqrt(6/({study.INIT_SAMPLE.in_features}+{study.INIT_SAMPLE.out_features}))"
               f" = {layers.xavier_limit(study.INIT_SAMPLE.in_features, study.INIT_SAMPLE.out_features):.6f}")
    report.add(f"  he 标准差 = sqrt(2/{study.INIT_SAMPLE.in_features}) = {layers.he_scale(study.INIT_SAMPLE.in_features):.6f}")


def section_layers(report: Report) -> None:
    """第 5 节：一层与一网的账."""
    report.section(5, "一层与一网的账：Dense 前向 / forward_trace / 层次账")
    dense = types.DenseSpec(3, 2, activation="gelu", init="xavier", seed=4)
    inputs = ((0.3, -0.7, 1.1),)
    linear = layers.dense_forward(
        types.DenseSpec(3, 2, activation=None, init="xavier", seed=4), inputs
    )
    activated = layers.dense_forward(dense, inputs)
    report.add(f"  输入 {inputs}")
    report.add(f"  affine 输出 {linear}")
    report.add(f"  gelu 输出   {activated}")
    report.add(f"  两步（affine → 逐行 gelu）与一步一致：{activated == activated_first(linear)}")
    spec = study.LAYER_SAMPLE
    report.add(f"  网络 {spec.line()}")
    for row in study.layer_rows(spec):
        report.add("  " + row.line())
    report.add(f"  参数总数 {spec.parameter_count}")


def section_collapse(report: Report) -> None:
    """第 6 节：恒等塌缩."""
    report.section(6, "恒等塌缩：多层网络 == 合成后的单层仿射映射")
    spec = verify.IDENTITY_SPEC
    multi = network.mlp_forward(spec, verify.IDENTITY_INPUTS)
    weight, bias = network.collapse_identity_mlp(spec)
    single = layers.affine_forward(weight, bias, verify.IDENTITY_INPUTS)
    worst = max(abs(a - b) for ra, rb in zip(multi, single) for a, b in zip(ra, rb))
    report.add(f"  网络 {spec.line()}（全恒等）")
    report.add(f"  多层输出 {multi}")
    report.add(f"  单层输出 {single}")
    report.add(f"  最大偏差 {worst:.3e} <= 容差 {verify.IDENTITY_TOLERANCE:.0e}：{worst <= verify.IDENTITY_TOLERANCE}")
    report.add("  这条事实的反证：拿掉激活，深度不增加表达力（再多层也只是一个仿射映射）")
    report.add("  任一非恒等激活都会让塌缩失败（抛 ParameterError）：本课测试逐条钉住")


def section_losses(report: Report) -> None:
    """第 7 节：三个损失与交叉熵的两条路径."""
    report.section(7, "三个损失与交叉熵的两条路径")
    for row in study.loss_rows():
        report.add("  " + row.line())
    worst = 0.0
    for logits, target in verify.CE_CASES:
        stable = losses.cross_entropy(logits, target)
        naive = losses.cross_entropy_via_probability(logits, target)
        worst = max(worst, abs(stable - naive))
        report.add(f"    logits={logits} target={target} | A={stable:.12f} B={naive:.12f}")
    report.add(f"  两条路径最大偏差 {worst:.3e}（容差 {verify.PATH_TOLERANCE:.0e}：一条走 sum、一条走 fsum）")
    report.add(f"  perplexity(mse 那一行的 loss) = {losses.perplexity(1.0):.6f}")


def section_crosscheck_one(report: Report) -> None:
    """第 8 节：跨天对账①（softmax / 交叉熵 / gelu）."""
    report.section(8, "跨天对账①：softmax / 交叉熵 / gelu 与既有实现逐位比")
    from smart_research_agent.hf_source.blocks import gelu_exact
    from smart_research_agent.math_foundations.linalg import softmax as reference_softmax
    from smart_research_agent.sft.loss import cross_entropy as reference_cross_entropy
    from smart_research_agent.sft.loss import log_softmax as reference_log_softmax

    softmax_mismatch = 0
    for logits in verify.SOFTMAX_CASES:
        mine = activations.softmax(logits)
        theirs = tuple(reference_softmax(list(logits)))
        softmax_mismatch += sum(1 for a, b in zip(mine, theirs) if a != b)
    report.add(f"  softmax  vs math_foundations.linalg.softmax：逐位不一致 {softmax_mismatch} 个")
    ce_mismatch = sum(
        1
        for logits, target in verify.CE_CASES
        if losses.cross_entropy(logits, target) != reference_cross_entropy(list(logits), target)
    )
    report.add(f"  ce       vs sft.loss.cross_entropy：逐位不一致 {ce_mismatch} 个")
    log_softmax_mismatch = sum(
        1
        for logits in verify.SOFTMAX_CASES
        if losses.log_softmax(logits) != tuple(reference_log_softmax(list(logits)))
    )
    report.add(f"  logsm    vs sft.loss.log_softmax：逐位不一致 {log_softmax_mismatch} 个")
    gelu_mismatch = sum(
        1 for value in activations.FINITE_GRID if activations.gelu(value) != gelu_exact(value)
    )
    report.add(f"  gelu     vs hf_source.blocks.gelu_exact：{len(activations.FINITE_GRID)} 点上不一致 {gelu_mismatch} 个")


def section_crosscheck_two(report: Report) -> None:
    """第 9 节：跨天对账②（FFN）."""
    report.section(9, "跨天对账②：ffn_block 与 encoder_decoder.feed_forward 逐位比")
    from smart_research_agent.encoder_decoder.layers import feed_forward
    from smart_research_agent.encoder_decoder.types import FFNWeights

    params = network.build_ffn_params(hidden=verify.FFN_HIDDEN, seed=verify.FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    report.add(f"  hidden={verify.FFN_HIDDEN}、d_ff={network.FFN_RATIO * verify.FFN_HIDDEN}、输入 {verify.FFN_INPUTS}")
    for activation in ("relu", "gelu"):
        mine = network.ffn_block(verify.FFN_INPUTS, real, activation=activation)
        theirs, _cache = feed_forward(verify.FFN_INPUTS, real, activation=activation)
        mismatch = sum(1 for ra, rb in zip(mine, theirs) for a, b in zip(ra, rb) if a != b)
        report.add(f"    [{activation}] 本包第 0 行 {mine[0]}")
        report.add(f"    [{activation}] 项目第 0 行 {theirs[0]}")
        report.add(f"    [{activation}] 逐位不一致 {mismatch} 个")


def section_properties_and_study(report: Report) -> None:
    """第 10 节：七条性质与五张表."""
    report.section(10, "七条性质与五张表")
    result = verify.check_all()
    for line in result.lines():
        report.add("  " + line)
    report.add(f"  全部通过：{result.ok}")
    report.add("")
    for line in study.study_lines():
        report.add("  " + line)


def section_families(report: Report) -> None:
    """第 11 节：失败族、边界与 Torch 对照表."""
    report.section(11, "失败族（缺席的 GradientError）、五条边界与 Torch 对照表")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<20} {outcome}")
    report.add("")
    report.add(f"  连续缺席的那一族：{ABSENT_FAMILY}")
    report.add(f"  理由：{ABSENT_FAMILY_REASON}")
    report.add("")
    for index, boundary in enumerate(NEURAL_BOUNDARIES, start=1):
        report.add(f"  {index}. {boundary}")
    report.add("")
    report.add("  纯 Python ↔ PyTorch 对照表（**只对照，不调用，不声明版本**）：")
    for key in ("neuron", "dense_forward", "relu", "gelu", "softmax", "mse", "cross_entropy", "xavier_init"):
        report.add(f"    {key:<16} → {TORCH_COUNTERPARTS[key]}")
    report.add("")
    report.add("  接缝：day073（数学）→ day075（注意力）→ day079（前馈）→ day085（源码）→")
    report.add("        day087（推理）→ day088（原理图）→ **day089（今天：从神经元重新搭起）**")
    report.add("  下游：day090 会给这条链补上唯一缺席的那一族 GradientError（反向传播）")


def _finite(value: float) -> bool:
    """有限性判定（本地小工具，避免在演示脚本里再 import 一次 math）."""
    return value == value and value not in (float("inf"), float("-inf"))


def activated_first(linear_rows: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """把 affine 输出逐行过一遍 gelu（用于演示"两步 == 一步"）."""
    return tuple(activations.activate("gelu", row) for row in linear_rows)


def main() -> None:
    """跑完十一节并落盘."""
    report = Report()
    section_chain(report)
    section_activations(report)
    section_ranges(report)
    section_initializations(report)
    section_layers(report)
    section_collapse(report)
    section_losses(report)
    section_crosscheck_one(report)
    section_crosscheck_two(report)
    section_properties_and_study(report)
    section_families(report)
    report.flush()


if __name__ == "__main__":
    main()
