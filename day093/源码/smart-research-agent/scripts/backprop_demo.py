"""day090 离线演示：反向传播 —— 从一条链上取回每一点的导数.

十一节，全部离线、全部确定性（不需要 API Key，也不装 transformers / torch / numpy）：

```text
1   三件事总览：前向 / 导数 / 回传（宽度链与参数量）
2   六个激活的导数：公式表 + 现场读数（softmax 那一行印"每行和为 0"）
3   两条独立路径：手写反向 vs 图自动微分（同一组权重）
4   一层的三块梯度：dW / db / dx 的形状与范数
5   整条 MLP 的逐层回传：三个范数 + 与数值差分的相对误差
6   softmax：显式雅可比 vs 那条 O(n) 的 JVP
7   两个损失的梯度：p − onehot 与 2(p − t)/N
8   跨天对账①：与 day074 的尺子（calculus / gradcheck）比
9   跨天对账②：ffn_backward 与 encoder_decoder.feed_forward_backward 逐位比
10  七条性质与五张表
11  一次解析反向训练 + 失败族（GradientError 回来了）与五条边界
```

运行方式::

    cd day090/源码/smart-research-agent
    python scripts/backprop_demo.py

产出 ``outputs/backprop_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.backprop import (  # noqa: E402
    ABSENT_FAMILY,
    FAMILY_OUTCOMES,
    RETURNED_FAMILY,
    RETURNED_FAMILY_REASON,
    BACKPROP_BOUNDARIES,
    TORCH_COUNTERPARTS,
    gradients,
    graph,
    layers,
    network,
    study,
    train,
    types,
    verify,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "backprop_demo.txt"


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


def section_overview(report: Report) -> None:
    """第 1 节：三件事总览."""
    report.section(1, "三件事总览：前向 / 导数 / 回传")
    spec = verify.MLP_SPEC
    report.add(f"  网络：{spec.line()}")
    report.add(f"  宽度链：{' → '.join(str(width) for width in spec.widths)}")
    cache = network.mlp_forward_with_cache(spec, verify.MLP_INPUTS)
    report.add(f"  前向：输出 {cache.output}")
    report.add(f"        缓存：{cache.depth} 层的输入 / 激活前值 / 输出都被留下（反向只读缓存）")
    trace = network.mlp_loss_gradients(cache, verify.MLP_TARGETS)
    report.add(f"  导数：{len(trace.flat_parameter_gradients())} 个偏导数（对参数）")
    report.add(f"        mse 损失 = {train.evaluate_loss(spec, verify.MLP_INPUTS, verify.MLP_TARGETS):.6e}")
    report.add("  回传：从 dL/dŷ 出发，逐层：激活的反向 → 仿射的反向（dW、db、dx）")


def section_derivatives(report: Report) -> None:
    """第 2 节：六个激活的导数."""
    report.section(2, "六个激活的导数：公式表 + 现场读数")
    for row in study.derivative_rows():
        report.add("  " + row.line())
    report.add("  softmax 那一行印的不是「某个数的导数」，而是**雅可比每一行之和**：")
    report.add("    Σ_j J[i][j] = p_i(1 − Σ_j p_j) = 0 —— 这就是「平移不变」在导数上的样子")
    report.add("  逐元素激活的导数与中心差分逐点一致（最大相对误差见第 8 节）")


def section_two_paths(report: Report) -> None:
    """第 3 节：手写反向 vs 图自动微分."""
    report.section(3, "两条独立路径：手写反向 vs 图自动微分（同一组权重）")
    spec = verify.MLP_SPEC
    params = network.build_parameters(spec)
    cache = network.chain_forward_with_cache(network.build_chain(spec), verify.MLP_INPUTS)
    hand = network.mlp_loss_gradients(cache, verify.MLP_TARGETS).flat_parameter_gradients()

    inputs_node = graph.constant(verify.MLP_INPUTS, label="x")
    weight_nodes: list[graph.Node] = []
    bias_nodes: list[graph.Node] = []
    current = inputs_node
    for (weight, bias), layer_spec in zip(params, spec.layers, strict=True):
        weight_node = graph.variable(weight, label="W")
        bias_node = graph.variable(bias, label="b")
        weight_nodes.append(weight_node)
        bias_nodes.append(bias_node)
        activated = graph.dense(current, weight_node, bias_node)
        current = _graph_activate(layer_spec.activation, activated)
    loss_node = graph.mse(current, verify.MLP_TARGETS)
    loss_node.backward()
    graph_flat = tuple(
        value
        for weight_node, bias_node in zip(weight_nodes, bias_nodes, strict=True)
        for block in (weight_node.grad, bias_node.grad)
        for value in _flat(block)
    )
    worst = max((abs(a - b) for a, b in zip(hand, graph_flat, strict=True)), default=0.0)
    report.add(f"  手写反向：{len(hand)} 个偏导数")
    report.add(f"  图自动微分：{len(graph_flat)} 个偏导数，损失 = {loss_node.value:.12f}")
    report.add(f"  两条路径最大偏差 {worst:.3e}（逐位一致：{worst == 0.0}）")
    report.add("  第三把尺子（数值差分）在 verify 的第 6 条性质里，相对误差 <= 1e-5")


def section_layer(report: Report) -> None:
    """第 4 节：一层的三块梯度."""
    report.section(4, "一层的三块梯度：dW / db / dx 的形状与范数")
    from smart_research_agent.neural_basics.layers import dense_linear, initialize

    spec = study.DENSE_SAMPLE
    weight, bias = initialize(spec.init, spec.out_features, spec.in_features, seed=spec.seed)
    pre_activation = dense_linear(weight, bias, study.DENSE_INPUTS)
    hidden = gradients.elementwise_backward(
        spec.activation, pre_activation, study.DENSE_GRAD_OUTPUT
    )
    grads = layers.dense_backward(weight, bias, study.DENSE_INPUTS, hidden)
    summary = layers.gradient_summary(grads, index=1)
    report.add(f"  层：dense({spec.in_features}→{spec.out_features}) | 激活 {spec.activation}")
    report.add(f"  输入 {study.DENSE_INPUTS}")
    report.add(f"  回传梯度 {study.DENSE_GRAD_OUTPUT}")
    report.add(f"  dW {grads.grad_weight}")
    report.add(f"  db {grads.grad_bias}")
    report.add(f"  dx {grads.grad_inputs}")
    report.add(f"  三块账：{summary.line()}")
    report.add("  三条式子：dW = dYᵀ·X、db = Σ_i dY[i]、dx = dY·W（按 W 的**列**收）")


def section_network(report: Report) -> None:
    """第 5 节：整条 MLP 的逐层回传."""
    report.section(5, "整条 MLP 的逐层回传：三个范数 + 与数值差分的相对误差")
    for row in study.network_layer_rows():
        report.add("  " + row.line())
    outcome = verify.check_mlp_backward_matches_numerical()
    report.add(f"  {outcome.line()}")
    report.add("  逐层账读的是**范数**：一层 (5, 3) 的权重有 15 个偏导数，")
    report.add("    全印出来读不出「这一层在学什么」，而三个范数能看出量级差")


def section_softmax(report: Report) -> None:
    """第 6 节：softmax 的雅可比与 JVP."""
    report.section(6, "softmax：显式雅可比 vs 那条 O(n) 的 JVP")
    probabilities = (0.2, 0.3, 0.5)
    vector = (0.5, -1.0, 2.0)
    jacobian = gradients.softmax_jacobian(probabilities)
    explicit = tuple(
        math.fsum(jacobian[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )
    implicit = gradients.softmax_jacobian_vector_product(probabilities, vector)
    worst = max(abs(a - b) for a, b in zip(explicit, implicit, strict=True))
    report.add(f"  概率 {probabilities}、回传梯度 {vector}")
    report.add(f"  显式雅可比 {jacobian}")
    report.add(f"  每一行之和 {[math.fsum(row) for row in jacobian]}（理论上全为 0）")
    report.add(f"  Jᵀv（显式）{explicit}")
    report.add(f"  p⊙(v − ⟨p,v⟩)  {implicit}")
    report.add(f"  最大偏差 {worst:.3e} <= 容差 {verify.IDENTITY_TOLERANCE:.0e}")
    report.add("  生产路径只走后者：中间量从 O(n²) 降到 O(n)")


def section_losses(report: Report) -> None:
    """第 7 节：两个损失的梯度."""
    report.section(7, "两个损失的梯度：p − onehot 与 2(p − t)/N")
    pred = ((1.0, 2.0), (3.0, 4.0))
    target = ((1.5, 1.5), (4.0, 3.0))
    report.add(f"  mse_grad({pred}, {target}) = {gradients.mse_grad(pred, target)}")
    report.add("    （2(p − t)/N，N = 全部元素个数 = 4）")
    for logits, index in verify.CE_CASES:
        grads = gradients.cross_entropy_grad(logits, index)
        report.add(f"  cross_entropy_grad({logits}, {index}) = {tuple(round(v, 6) for v in grads)}")
    report.add("  ∂(−log p_k)/∂z = p − onehot(k)：softmax 的雅可比在这里被**消掉**了")
    report.add("    因此「雅可比写错」这件事在交叉熵的梯度上**看不见**——它必须被第 3 条性质单独钉住")


def section_crosscheck_one(report: Report) -> None:
    """第 8 节：跨天对账①（与 day074 的尺子比）."""
    report.section(8, "跨天对账①：与 day074 的尺子（calculus / gradcheck）比")
    from smart_research_agent.math_foundations.gradcheck import (
        check_cross_entropy_gradient,
        check_softmax_jacobian,
        difference_resolution,
    )

    report.add(f"  分辨率下限（|f|=1、h=1e-6）：{difference_resolution(magnitude=1.0):.3e}")
    report.add(f"  分辨率下限（|f|=100、h=1e-6）：{difference_resolution(magnitude=100.0):.3e}")
    softmax_outcome = check_softmax_jacobian()
    cross_outcome = check_cross_entropy_gradient()
    report.add(f"  day074 的 softmax 雅可比对照：{softmax_outcome.summary_line()}")
    report.add(f"  day074 的交叉熵梯度对照：  {cross_outcome.summary_line()}")
    for outcome in (
        verify.evaluate_derivatives_vs_numerical(),
        verify.check_softmax_jacobian_matches_numerical(),
        verify.check_cross_entropy_gradient_is_p_minus_onehot(),
    ):
        report.add(f"  本包那一侧：{outcome.line()}")


def section_crosscheck_two(report: Report) -> None:
    """第 9 节：跨天对账②（与 encoder_decoder 的前馈反向逐位比）."""
    report.section(9, "跨天对账②：ffn_backward 与 feed_forward_backward 逐位比")
    from smart_research_agent.encoder_decoder.layers import feed_forward, feed_forward_backward
    from smart_research_agent.encoder_decoder.types import FFNWeights
    from smart_research_agent.neural_basics.network import build_ffn_params

    params = build_ffn_params(hidden=verify.FFN_HIDDEN, seed=verify.FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    report.add(f"  hidden={verify.FFN_HIDDEN}、d_ff={verify.FFN_HIDDEN * 4}、输入 {verify.FFN_INPUTS}")
    report.add(f"  回传梯度 {verify.FFN_GRAD_OUTPUT}")
    for activation in ("relu", "gelu"):
        ours = network.ffn_backward(
            verify.FFN_INPUTS, real, activation=activation, grad_output=verify.FFN_GRAD_OUTPUT
        )
        _output, cache = feed_forward(verify.FFN_INPUTS, real, activation=activation)
        theirs = feed_forward_backward(cache, verify.FFN_GRAD_OUTPUT)
        mismatch = 0
        for field in ("grad_w_in", "grad_b_in", "grad_w_out", "grad_b_out", "grad_inputs"):
            mine = _flat(getattr(ours, field))
            other = _flat(getattr(theirs, field))
            mismatch += sum(1 for a, b in zip(mine, other, strict=True) if a != b)
        report.add(f"    [{activation}] 五块梯度共 {len(_flat(ours.grad_w_in)) + len(_flat(ours.grad_w_out)) + len(ours.grad_b_in) + len(ours.grad_b_out) + len(_flat(ours.grad_inputs))} 个元素，逐位不一致 {mismatch} 个")
    report.add("  四条前提：同一组权重、同样的 (d_ff, d) 形状约定、同样的 fsum 累加、同样的 relu/gelu 导数")


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


def section_training_and_families(report: Report) -> None:
    """第 11 节：训练、失败族与边界."""
    report.section(11, "一次解析反向训练、失败族（GradientError 回来了）与五条边界")
    report.add("  sgd（20 步，lr=0.05）：")
    sgd = train.train_mlp(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS, steps=20, learning_rate=0.05)
    report.add(f"    {sgd.summary_line()}")
    report.add("  adam（20 步，lr=0.05）：")
    adam = train.train_mlp(
        verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS, steps=20, learning_rate=0.05, optimizer_name="adam"
    )
    report.add(f"    {adam.summary_line()}")
    report.add("")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<20} {outcome}")
    report.add("")
    report.add(f"  连续缺席八天之后**回来了**的那一族：{RETURNED_FAMILY}")
    report.add(f"  理由：{RETURNED_FAMILY_REASON}")
    report.add(f"  本课缺席的那一族：{ABSENT_FAMILY}（第一次为空——不是没有理由，而是没有缺席者）")
    report.add("")
    for index, boundary in enumerate(BACKPROP_BOUNDARIES, start=1):
        report.add(f"  {index}. {boundary}")
    report.add("")
    report.add("  纯 Python ↔ PyTorch 对照表（**只对照，不调用，不声明版本**）：")
    for key in ("backward", "zero_grad", "graph", "dense_backward", "gelu_backward", "softmax_backward", "cross_entropy_grad", "grad_check"):
        report.add(f"    {key:<20} → {TORCH_COUNTERPARTS[key]}")
    report.add("")
    report.add("  接缝：day074（尺子与优化器）→ day079（前馈反向）→ day089（前向）→")
    report.add("        **day090（今天：在同一个链上取回每一点的导数）**")
    report.add("  下游：day092 会沿着同一条链问「走多远」（优化器）——今天只回答「往哪走」")


def _graph_activate(name: str | None, node: graph.Node) -> graph.Node:
    """按名字把图上的一个节点过一遍激活（演示里只用到 relu 与 gelu）."""
    if name is None:
        return node
    if name == "relu":
        return graph.relu(node)
    if name == "gelu":
        return graph.gelu(node)
    raise ValueError(f"演示脚本只支持 relu / gelu（恒等用 None），收到 {name!r}")


def _flat(value: object) -> tuple[float, ...]:
    """把一个矩阵 / 向量压平成一串数."""
    items = tuple(value)  # type: ignore[call-overload]
    if items and isinstance(items[0], (tuple, list)):
        return tuple(float(item) for row in items for item in row)  # type: ignore[union-attr]
    return tuple(float(item) for item in items)


def main() -> None:
    """跑完十一节并落盘."""
    report = Report()
    section_overview(report)
    section_derivatives(report)
    section_two_paths(report)
    section_layer(report)
    section_network(report)
    section_softmax(report)
    section_losses(report)
    section_crosscheck_one(report)
    section_crosscheck_two(report)
    section_properties_and_study(report)
    section_training_and_families(report)
    report.flush()


if __name__ == "__main__":
    main()
