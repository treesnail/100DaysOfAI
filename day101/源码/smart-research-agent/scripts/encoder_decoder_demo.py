"""离线演示：把三个子层拼成一个块（day079 / M7-D4）.

跑法：

```bash
python scripts/encoder_decoder_demo.py      # 十一节，输出写到 outputs/encoder_decoder_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。
产物只写在 ``outputs/`` 下（幂等，可随时删）。

十一节里最值得看的是第 5、6、7、8、9、11 节：

```text
5    残差：两个分支都为 0 时 output == inputs（**逐位**），而 dx 只由两条 LN 链决定
6    残差那条 +1 路：把前馈压到 1/16，有残差 ‖dx‖ 几乎不变、无残差随分支一起塌（实测差 18.2 倍）
7    九项块级梯度校验：pre/post × 残差开/关 四组，实测 ~1e-10（容差 1e-6，隔着四个数量级）
8    六项交叉注意力校验：权重是 (n_tgt, n_src) 长方形；source 那一侧拿到**两条链之和**
9    八条性质（8/8）与"误加因果掩码"的代价：n_tgt == n_src 时**不报错**，权重被改掉 1.00e+00
11   深度实验：pre/post/bare 三个变体的 ‖∂loss/∂x‖，bare 衰减到第 1 层的 1.92e-03
```
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.encoder_decoder import (  # noqa: E402
    ACTIVATION_DESCRIPTIONS,
    ACTIVATION_GELU,
    ACTIVATION_RELU,
    BLOCK_GRADIENT_FORMULAS,
    BLOCK_GRADIENT_TARGETS,
    CROSS_GRADIENT_FORMULAS,
    CROSS_GRADIENT_TARGETS,
    DECODER_BLOCK_STAGES,
    DECODER_STAGE_DESCRIPTIONS,
    DECODER_STAGE_SHAPES,
    DEFAULT_DEPTHS,
    DEFAULT_EPSILON,
    DEFAULT_FFN_RATIO,
    ENCODER_BLOCK_STAGES,
    ENCODER_DECODER_NOTES,
    ENCODER_DECODER_PROPERTIES,
    FAMILY_OUTCOMES,
    NORM_PLACEMENTS,
    NORM_PLACEMENT_DESCRIPTIONS,
    NORM_POST,
    NORM_PRE,
    PROPERTY_DESCRIPTIONS,
    STAGE_DESCRIPTIONS,
    STAGE_SHAPES,
    VARIANTS,
    VARIANT_DESCRIPTIONS,
    BlockShape,
    CrossParameters,
    block_attention,
    causal_mask_damage,
    check_block_gradients,
    check_cross_attention_is_not_causal,
    check_cross_gradients,
    check_decoder_input_gradients,
    check_feed_forward_is_position_wise,
    check_norm_is_row_independent,
    check_properties,
    check_residual_identity,
    check_residual_unit_path,
    check_scale_equivariance,
    check_shift_invariance,
    cross_attention,
    cross_attention_backward,
    decoder_block,
    decoder_block_backward,
    depth_study,
    encoder_block,
    encoder_block_backward,
    feed_forward,
    layer_norm,
    make_block_parameters,
    matrix_frobenius,
    stage_order,
)
from smart_research_agent.transformer_core.layers import (  # noqa: E402
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.train import default_parameters  # noqa: E402
from smart_research_agent.transformer_core.types import AttentionParams  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "encoder_decoder_demo.txt"

#: 演示用的形状（与 ``tests/encoder_decoder_samples.py`` 同值）
HIDDEN = 6
FFN = 24
TOKENS = 4
SOURCE_TOKENS = 6
DECODER_TOKENS = 3

#: 手算 LayerNorm 的那一行
HAND_ROW = (1.0, 2.0, 3.0, 4.0)


class _Tee:
    """把 stdout 同时写终端与文件（与前面的演示脚本同一手法）."""

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        self._handle.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._handle.flush()
        self._stream.flush()


def section(title: str) -> None:
    """打印一节的分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def round_row(values, digits: int = 6) -> tuple:
    """把一行四舍五入到 ``digits`` 位（打印用）."""
    return tuple(round(value, digits) for value in values)


def round_matrix(matrix, digits: int = 6) -> tuple:
    """把矩阵四舍五入（打印用）."""
    return tuple(round_row(row, digits) for row in matrix)


def ramp(rows: int, columns: int = HIDDEN, *, step: float = 0.2, drift: float = 0.1) -> tuple:
    """写死的非平凡输入（每一行**方差非零**，否则 LN 会退化成全零行）."""
    return tuple(
        tuple(step * (i + 1) + drift * (j + 1) for j in range(columns)) for i in range(rows)
    )


def zero_matrix(rows: int, columns: int) -> tuple:
    """零矩阵（构造"两个分支都为 0"的残差用例）."""
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def identity(size: int, scale: float) -> tuple:
    """``scale`` 倍的单位矩阵."""
    return tuple(tuple(scale if i == j else 0.0 for j in range(size)) for i in range(size))


def main() -> None:
    """十一节演示."""
    shape = BlockShape(hidden=HIDDEN, ffn=FFN, tokens=TOKENS)
    params = make_block_parameters(shape)
    attention_params = default_parameters(HIDDEN)
    inputs = ramp(TOKENS)
    target = tuple(tuple(0.4 for _ in range(HIDDEN)) for _ in range(TOKENS))

    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：六个阶段与两种摆放位置——LN 站在哪一边")
    print(f"  块形状：{shape.summary_line()}（d_ff/d = {shape.ffn_ratio:.1f} = DEFAULT_FFN_RATIO）")
    print()
    for placement in NORM_PLACEMENTS:
        order = " → ".join(stage_order(placement))
        print(f"  {placement:>4}-LN  实际顺序：{order}")
        print(f"          {NORM_PLACEMENT_DESCRIPTIONS[placement]}")
    print()
    print("  同一批阶段名，只是 LN 转过一格——**形状完全一样**（这是第 11 节实验的全部意义）：")
    for stage in ENCODER_BLOCK_STAGES:
        print(f"    {stage:<8} {STAGE_SHAPES[stage]:<38} {STAGE_DESCRIPTIONS[stage]}")
    print()
    print(f"  解码器块多一个交叉注意力子层，因此是 **{len(DECODER_BLOCK_STAGES)}** 个阶段："
          f"{' → '.join(DECODER_BLOCK_STAGES)}")
    print(f"  五个失败族：{'、'.join(FAMILY_OUTCOMES)}")
    print("  （多出来的 AssemblyError 专管'两个各自合法的部件拼不成一个整体'）")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：LayerNorm 手算——x = (1, 2, 3, 4)")
    actual, cache = layer_norm((HAND_ROW,))
    mean = 2.5
    variance = 1.25
    closed = tuple(round((value - mean) / math.sqrt(variance), 6) for value in HAND_ROW)
    print(f"  μ = Σx/d = 10/4 = {mean}")
    print(f"  σ² = Σ(x−μ)²/d = 5/4 = {variance}（**有偏**方差，与 PyTorch 一致）")
    print(f"  x̂（eps → 0 的闭式）  = {closed}")
    print(f"  x̂（实测，eps=1e-5） = {round_row(actual[0])}")
    print("  → 两者只差第 6 位小数（eps 让分母从 √1.25 变成 √(1.25+1e-5)）：")
    print(f"     |Δ| ≈ {abs(closed[0] - actual[0][0]):.0e}，而'x̂ 的方差离 1 多远'是 "
          f"{abs(cache.variance[0] / (cache.variance[0] + cache.epsilon) - 1.0):.2e}")
    print(f"  实测 x̂ 的方差 = σ²/(σ² + eps) = {cache.variance[0] / (cache.variance[0] + cache.epsilon):.6f}"
          f"（**它是 eps 的定义，不是误差**）")
    print(f"  DEFAULT_EPSILON = {DEFAULT_EPSILON:g}；cache 记下的 eps = {cache.epsilon:g}")
    print(f"  γ/β 缺省是 1/0，因此这一节的输出就是 x̂ 本身：{cache.summary_line()}")

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：LN 的三条性质——平移不是信息、尺度不是信息、逐行独立")
    shift = check_shift_invariance(inputs)
    scale = check_scale_equivariance(inputs)
    row = check_norm_is_row_independent(inputs)
    for outcome in (shift, scale, row):
        print(f"  [{ 'ok' if outcome.passed else '!!' }] {outcome.name}")
        print(f"       {outcome.evidence}")
    print()
    print("  前两条是**精确**恒等式（eps → 0）：(x+c)−(μ+c) = x−μ，而 eps 在分子分母里是同一个数。")
    print("  第三条是 LayerNorm 与 **BatchNorm** 的分水岭：后者的统计量跨样本，换行序会改变每一行。")
    print(f"  它的判据是 `==` 而不是容差：{row.evidence.split('：')[-1]}")

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：前馈（d_ff = 4d）与逐位置")
    print(f"  前馈参数：{params.ffn.summary_line()}")
    hidden_out, ffn_cache = feed_forward(inputs, params.ffn, activation=ACTIVATION_RELU)
    print(f"  前向形状：{ffn_cache.summary_line()}")
    print(f"  激活：{ACTIVATION_DESCRIPTIONS[ACTIVATION_RELU]}")
    print(f"        {ACTIVATION_DESCRIPTIONS[ACTIVATION_GELU]}")
    print("  零点占比是 ReLU 的一个便宜读数：关掉的神经元在反向里梯度**恰好是 0**。")
    position_wise = check_feed_forward_is_position_wise(inputs, params.ffn)
    print(f"  [{ 'ok' if position_wise.passed else '!!' }] {position_wise.evidence}")
    print("  LN 与前馈都是**逐行**算子，因此它们合起来给出一个跨天的结论：")
    print("  **打破置换等变性的只有位置编码与掩码**（day078 的第 4 条性质）。")
    print(f"  输入 {len(inputs)}×{HIDDEN} × 四个投影 → {len(hidden_out)}×{HIDDEN}"
          f"（本块新增参数 {shape.parameter_count} 个）")

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：残差——两个分支都为 0 时 output == inputs（逐位）")
    identity_outcome = check_residual_identity(params, inputs, placement=NORM_PRE)
    print(f"  [{ 'ok' if identity_outcome.passed else '!!' }] {identity_outcome.evidence}")
    print(f"       {identity_outcome.detail}")
    print()
    print("  这里有一个值得记下的坑：**“分支为零”要两支都为零**——")
    print("    只把前馈的输出层置零 → residual2 = residual1 = inputs + 注意力输出 ≠ inputs")
    print("    两个分支都置零       → output = inputs（**逐位**，判据用 `==`）")
    blank = make_block_parameters(shape)
    zero_params = blank.with_ffn(_zero_ffn(blank.ffn))
    zero_attention_params = AttentionParams(
        w_query=zero_matrix(HIDDEN, HIDDEN),
        w_key=zero_matrix(HIDDEN, HIDDEN),
        w_value=zero_matrix(HIDDEN, HIDDEN),
        w_output=zero_matrix(HIDDEN, HIDDEN),
    )
    zero_attention = block_attention(zero_params, inputs, zero_attention_params, placement=NORM_PRE)
    zero_forward = encoder_block(
        zero_params, inputs, zero_attention, placement=NORM_PRE, use_residual=True
    )
    print(f"  直接构造一次：output == inputs ? {zero_forward.output == inputs}")
    grad_out = tuple(tuple(1.0 for _ in range(HIDDEN)) for _ in range(TOKENS))
    zero_grads = encoder_block_backward(zero_forward, zero_params, grad_out)
    dy = matrix_frobenius(grad_out)
    dx = matrix_frobenius(zero_grads.grad_inputs)
    print(f"  反向：上游取全 1 时 ‖dx‖/‖dy‖ = {dx / dy:.6f}"
          "（两个 LN 的链也在贡献，它恰好把比值抬回 1 附近——见性质报告）")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：残差那条 +1 的路——把前馈压到 1/16 看梯度还在不在")
    unit = check_residual_unit_path(params, inputs, attention_params, placement=NORM_PRE)
    gap = unit.evidence.split("相差")[-1].strip().rstrip("。") if "相差" in unit.evidence else "?"
    print(f"  [{ 'ok' if unit.passed else '!!' }] {unit.evidence}")
    print(f"       {unit.detail}")
    print()
    print("  读法：把前馈输出层整体乘 s（s = 1 → 0.25 → 0.0625）：")
    print(f"    带残差   dx ≈ dy + O(s)   → s 再小也有一条 dy 在那里（比值 1.13e+00）")
    print(f"    不带残差 dx = O(s)        → 分支被压小时梯度一起被压小（比值 6.19e-02）")
    print(f"  ⇒ 两者相差 {gap}——**残差是梯度的那条高速路**（第 11 节把它放大到 8 层）。")
    print("  没有残差时，这一项会变成 O(s) 并随深度指数衰减；这正是 bare 变体在深度实验里崩掉的原因。")

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：九项块级梯度校验——pre/post × 残差开/关 四组")
    print("  九项名单（顺序就是压平时的顺序）：")
    for name in BLOCK_GRADIENT_TARGETS:
        print(f"    {name:<12} {BLOCK_GRADIENT_FORMULAS[name]}")
    print()
    print(f"  {'摆放/残差':<16} | {'通过':>4} | {'最大相对误差':>12} | 逐项（最大相对误差）")
    for placement in (NORM_PRE, NORM_POST):
        for use_residual in (True, False):
            report = check_block_gradients(
                params,
                inputs,
                attention_params,
                target,
                placement=placement,
                use_residual=use_residual,
            )
            detail = "、".join(
                f"{item.target}={item.max_scaled_error:.1e}" for item in report.outcomes
            )
            passed = sum(1 for item in report.outcomes if item.passed)
            label = f"{placement}-LN 残差 {'开' if use_residual else '关'}"
            print(f"  {label:<16} | {passed:>4} | {report.worst_scaled_error:>12.2e} | {detail}")
    print()
    print("  容差 1e-6，实测 ~1e-10——隔着四个数量级；四组全部 9/9。")
    print("  为什么能四组都过：两侧都走**同一个** block_attention 与 encoder_block，")
    print("  而 placement / use_residual / activation / causal 都由同一组参数传给两侧（'忘了传'结构上不可能）。")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：六项交叉注意力校验——权重是长方形、source 是两条链之和")
    cross_params = CrossParameters(
        w_query=tuple(tuple(0.02 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
        w_key=tuple(tuple(0.03 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
        w_value=tuple(tuple(0.04 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
        w_output=tuple(tuple(0.05 * (j + 1) for j in range(HIDDEN)) for _ in range(HIDDEN)),
    )
    source = ramp(SOURCE_TOKENS, HIDDEN, step=0.2, drift=0.01)
    cross_forward = cross_attention(cross_params, inputs, source)
    print(f"  {cross_forward.summary_line()}")
    print(f"  Q 来自 target（{TOKENS} 行）、K/V 来自 source（{SOURCE_TOKENS} 行）"
          f"→ 权重是 ({TOKENS}, {SOURCE_TOKENS}) 的**长方形**，不是方阵。")
    print()
    for name in CROSS_GRADIENT_TARGETS:
        print(f"    {name:<14} {CROSS_GRADIENT_FORMULAS[name]}")
    print()
    cross_report = check_cross_gradients(cross_params, inputs, source, target)
    print(f"  {cross_report.summary_line()}")
    for outcome in cross_report.outcomes:
        print(f"    {outcome.summary_line()}")
    print()
    print("  最值钱的一行是 `source_inputs`：**dSource = dK·W_k + dV·W_v（两条链之和）**。")
    print("  少一条不会报错，只会让梯度偏小——而它在这一项里会被抓出来。")
    print("  `target_inputs` 只有一条链（dQ·W_q），因此它的比较点数比 source 少。")
    cross_grads = cross_attention_backward(cross_forward, mse_gradient(cross_forward.output, target))
    print(f"    dTarget {len(cross_grads.grad_target_inputs)}×{HIDDEN}（与 target 同行数）| "
          f"dSource {len(cross_grads.grad_source_inputs)}×{HIDDEN}（与 source 同行数）")
    print(f"    {cross_grads.summary_line()}")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：八条性质（8/8）与'误加因果掩码'的代价")
    saturated = CrossParameters(
        w_query=identity(HIDDEN, 10.0),
        w_key=identity(HIDDEN, 10.0),
        w_value=identity(HIDDEN, 10.0),
        w_output=identity(HIDDEN, 1.0),
    )
    property_report = check_properties(params, inputs, attention_params, saturated)
    print(f"  {property_report.summary_line()}")
    for outcome in property_report.outcomes:
        print(f"    {outcome.summary_line()}")
    print()
    for name in ENCODER_DECODER_PROPERTIES:
        print(f"    {name:<52} {PROPERTY_DESCRIPTIONS[name]}")
    print()
    print("  第 8 条是整个包里最值钱的一条——它的失效方式**很坏**：")
    print("    自注意力（解码器）  Q/K/V 来自同一路 → 必须加因果掩码（不许看未来）")
    print("    交叉注意力          Q 来自解码器、K/V 来自编码器 → **绝不能加因果掩码**")
    cross_forward_sat = cross_attention(saturated, inputs, inputs[:TOKENS])
    damage, note = causal_mask_damage(cross_forward_sat)
    print(f"    本包直接拒绝 causal=True（check_cross_attention_is_not_causal 通过："
          f"{check_cross_attention_is_not_causal(saturated, inputs, inputs[:TOKENS]).passed}）")
    print(f"    而'加错之后差多少'：{note}")
    print(f"    ⇒ 代价 {damage:.2e}：n_tgt == n_src 时**形状刚好合适、不报错**，")
    print("      只是把源序列的后半段从注意范围里删掉了。")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：解码器块——九个阶段与 dEncoder（编码器会收到梯度）")
    print(f"  解码器块比编码器块多一个子层，因此是 {len(DECODER_BLOCK_STAGES)} 个阶段：")
    for stage in DECODER_BLOCK_STAGES:
        print(f"    {stage:<12} {DECODER_STAGE_SHAPES[stage]:<44} {DECODER_STAGE_DESCRIPTIONS[stage]}")
    print()
    decoder_inputs = ramp(DECODER_TOKENS, HIDDEN, step=0.1, drift=0.02)
    encoder_outputs = ramp(SOURCE_TOKENS, HIDDEN, step=0.3, drift=0.02)
    gammas = tuple(tuple(1.0 for _ in range(HIDDEN)) for _ in range(3))
    betas = tuple(tuple(0.0 for _ in range(HIDDEN)) for _ in range(3))
    ffn = params.ffn
    causal_attention = self_attention(
        attention_params,
        layer_norm(decoder_inputs, gamma=gammas[0], beta=betas[0])[0],
        causal=True,
    )
    output, cross_forward2, caches, ffn_cache2 = decoder_block(
        causal_attention,
        cross_params,
        gammas,
        betas,
        ffn,
        decoder_inputs,
        encoder_outputs,
    )
    print(f"  前向：解码器输入 {len(decoder_inputs)}×{HIDDEN}、编码器输出 "
          f"{len(encoder_outputs)}×{HIDDEN} → 输出 {len(output)}×{len(output[0])}")
    print(f"        交叉注意力在这一层里的权重是 "
          f"({len(cross_forward2.weights)}, {len(cross_forward2.weights[0])})"
          f"——**K/V 是另一路的全部位置，没有掩码**")
    decoder_target = tuple(tuple(0.5 for _ in range(HIDDEN)) for _ in range(DECODER_TOKENS))
    decoder_grads = decoder_block_backward(
        causal_attention,
        cross_forward2,
        caches,
        ffn_cache2,
        mse_gradient(output, decoder_target),
    )
    d_decoder = max(abs(v) for row in decoder_grads.grad_decoder_inputs for v in row)
    d_encoder = max(abs(v) for row in decoder_grads.grad_encoder_outputs for v in row)
    print(f"  反向：{decoder_grads.summary_line()}")
    print(f"        dDecoder max = {d_decoder:.6f} | dEncoder max = {d_encoder:.6f}")
    print("        **dEncoder ≠ 0**：'解码器只读编码器的输出'这句话在反向里不成立——")
    print("        编码器的输出会收到一股梯度（它是否传回编码器参数取决于调用方怎么用它）。")
    decoder_report = check_decoder_input_gradients(
        causal_attention,
        cross_params,
        gammas,
        betas,
        ffn,
        decoder_inputs,
        encoder_outputs,
        decoder_target,
    )
    print(f"  两路输入梯度校验：{decoder_report.summary_line()}")
    for outcome in decoder_report.outcomes:
        print(f"    {outcome.summary_line()}")
    print("  这一份只查**接线**（三段子层的前向/反向都已被 9 项 + 6 项逐项验过）；")
    print("  一处必须写下来的契约：自注意力按 pre-LN 作用在 LN(解码器输入) 上，")
    print("  因此检查里也从它的 params **重建**注意力（与 block_attention 同一纪律）。")

    # ------------------------------------------------------------------ 第 11 节
    section("第 11 节：深度实验——残差是梯度的那条高速路")
    print(f"  三个变体（一次只改一个旋钮，共用同一个目标与同一批初始参数）：")
    for variant in VARIANTS:
        print(f"    {variant:<14} {VARIANT_DESCRIPTIONS[variant]}")
    study = depth_study(shape)
    print()
    print(f"  深度序列 = {DEFAULT_DEPTHS}（都是 2 的幂附近）")
    for line in study.table_lines():
        print(line)
    print()
    print(f"  {study.summary_line()}")
    print(f"  判决（bare 的比值 < pre 的 1/10）：{study.verdict_ok}")
    print()
    print("  读法（三个读数是同一条 §6 的读数在被堆叠放大之后的版本）：")
    print(f"    pre_residual   最深一层 ‖dx‖ 相对第 1 层 {study.ratios('pre_residual')[-1]:.2e}"
          "——**基本平稳**（现代实现的主流）")
    print(f"    post_residual  最深一层 {study.ratios('post_residual')[-1]:.2e}"
          "——衰减但不塌（原论文的写法）")
    print(f"    bare           最深一层 {study.ratios('bare')[-1]:.2e}"
          "——**崩掉**（把 +x 关掉之后只剩雅可比连乘）")
    bare_ratio = study.ratios('bare')[-1] / study.ratios('pre_residual')[-1]
    print(f"  ⇒ bare / pre = {bare_ratio:.2e}：一个只差'子层顺序 + 一个 +x'的改动，")
    print("     在 8 层之后让梯度差出一个数量级以上——**这一课的值钱结论就在这里**。")
    print()
    print("  边界（必须写下来）：这一条读数回答'梯度能不能传到最底层'，")
    print("  它**不回答**'这个堆叠能不能训好'——梯度大也可能是爆炸而不是好事。")

    # ------------------------------------------------------------------ 收尾
    section("收尾：这一课留下的四句话")
    for note in ENCODER_DECODER_NOTES:
        print(f"  · {note}")
    print()
    print("  速查：")
    print(f"    DEFAULT_EPSILON = {DEFAULT_EPSILON:g}（PyTorch nn.LayerNorm 的默认值）")
    print(f"    DEFAULT_FFN_RATIO = {DEFAULT_FFN_RATIO}（d_ff = 4d）")
    print(f"    九项梯度名单 = {'、'.join(BLOCK_GRADIENT_TARGETS)}")
    print(f"    六个阶段 = {' → '.join(ENCODER_BLOCK_STAGES)}")
    print(f"    残差那条 +1 路的代价 = {gap}（压到 1/16）；误加因果掩码的代价 = {damage:.2e}")
    print(f"    深度实验最深一层 ‖dx‖ 比："
          + "、".join(f"{v.split('_')[0]} {study.ratios(v)[-1]:.2e}" for v in VARIANTS))
    print()
    print(f"  提示：残差开关是 encoder_block(..., use_residual=False) 的一个参数——")
    print("  它与 pre/post 一起构成第 11 节那张表的两个旋钮。")


def _zero_ffn(weights):
    """把前馈的四块参数全部置零（构造"分支为 0"的残差用例）."""
    from smart_research_agent.encoder_decoder.types import FFNWeights

    return FFNWeights(
        w_in=tuple(tuple(0.0 for _ in row) for row in weights.w_in),
        b_in=tuple(0.0 for _ in weights.b_in),
        w_out=tuple(tuple(0.0 for _ in row) for row in weights.w_out),
        b_out=tuple(0.0 for _ in weights.b_out),
    )


def _run() -> None:
    """建目录、装 tee、跑演示."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handle = (OUTPUT_DIR / OUTPUT_NAME).open("w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, handle)
    try:
        main()
    finally:
        sys.stdout = original
        handle.close()
        print(f"\n输出已写入 {OUTPUT_DIR / OUTPUT_NAME}")


if __name__ == "__main__":
    _run()
