"""离线演示：把一个块复制成一条链（day080 / M7-D5）.

跑法：

```bash
python scripts/transformer_stack_demo.py      # 十一节，输出写到 outputs/transformer_stack_demo.txt
```

全部离线：纯 Python 算术（不用 numpy）、生成的那段 PyTorch 脚本也**只生成不运行**
（本包从不 import torch）。产物只写在 ``outputs/`` 下（幂等，可随时删）。

十一节里最值得看的是第 1、4、5、7、8、10、11 节：

```text
1     形状与参数量：4 层 / d=6 / d_ff=24 → 每层 486、合计 1944（三个数都能手算）
4     逐层 ‖dx‖：day079 只看过**一个数**，今天把它变成一条序列
5     两处梯度校验：整条链的输入（1 项）与第 0 层的八块参数（8 项），实测 ~1e-10
7     PyTorch 组装：把生成的脚本打出来，并把 PyTorch 与解析式的参数量差算清（4d × N）
8     堆叠实验：2 变体 × 6 档深度，"每层倍数"与总深度几乎无关（残差开时）
10    反证：把"每层都用 dLoss/dy_N"的错法写出来，它与正确结果的相对差有多大
11    边界：残差关 + 分支全零 + 两层 ⇒ 撞上 day075 的全零行守卫（两天的判据在这里接上）
```
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.encoder_decoder.types import NORM_PRE  # noqa: E402
from smart_research_agent.transformer_core.errors import (  # noqa: E402
    NumericError as CoreNumericError,
)
from smart_research_agent.transformer_stack import (  # noqa: E402
    ASSEMBLY_REQUIREMENT,
    ASSEMBLY_CLASSES,
    ASSEMBLY_MODULES,
    DEFAULT_INIT_SCALE,
    DEFAULT_STUDY_DEPTHS,
    FAMILY_OUTCOMES,
    GRAD_STACK_INPUTS,
    KIND_LAYER,
    KIND_STACK,
    LAYER_GRADIENT_TARGETS,
    STACK_NOTES,
    STACK_PROPERTIES,
    STUDY_VARIANTS,
    VARIANT_BARE,
    VARIANT_DESCRIPTIONS,
    VARIANT_RESIDUAL,
    AssemblyError,
    StackParameters,
    assembly_facts,
    assembly_script,
    check_layer_parameter_gradients,
    check_properties,
    check_stack_input_gradient,
    geometric_mean,
    make_shape,
    make_stack_parameters,
    parse_script_module_names,
    relative_gradient_change,
    relative_matrix_error,
    stack_forward,
    stack_gradient_profile,
    stack_loss,
    stack_loss_gradient,
    stack_stage_lines,
    stack_study,
    zero_attention,
    zero_branches,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "transformer_stack_demo.txt"

LAYERS = 4
HIDDEN = 6
FFN_RATIO = 4
TOKENS = 4
SEED = 7


class _Tee:
    """把 stdout 同时写到文件与终端（day079 的演示脚本用的是同一种写法）."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _section(title: str) -> None:
    """打印一节的小标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _sample_inputs() -> tuple:
    """样本输入（与 tests/stack_samples.py 同值，因此两处读数可以直接对照）."""
    return tuple(
        tuple(0.1 * (row + 1) - 0.05 * (column + 1) for column in range(HIDDEN))
        for row in range(TOKENS)
    )


def _sample_target() -> tuple:
    """样本目标（与 tests/stack_samples.py 同值）."""
    return tuple(
        tuple(0.2 - 0.03 * (row * HIDDEN + column) for column in range(HIDDEN))
        for row in range(TOKENS)
    )


def _loss_gradient_norm(forward) -> float:
    """``‖dy_N‖``：损失对最后一层输出的梯度范数（``overall_ratio`` 的分母）."""
    from smart_research_agent.transformer_core.layers import mse_gradient
    from smart_research_agent.transformer_stack import frobenius

    return frobenius(mse_gradient(forward.output, _sample_target()))


def section_1_shape() -> None:
    """第 1 节：形状与参数量（三个数都能手算）."""
    _section("1 形状与参数量：4 层 / d=6 / d_ff=24")
    shape = make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO)
    print(f"  {shape.summary_line()}")
    d, f = shape.hidden, shape.ffn
    print()
    print("  块参数  = 4d（两个 LN 的 γ/β）+ 2·d·d_ff（两层线性）+ d_ff + d")
    print(f"          = 4×{d} + 2×{d}×{f} + {f} + {d} = {shape.block_parameter_count}")
    print(f"  注意力  = 4d² = 4×{d}² = {shape.attention_parameter_count}（day075 起就在那里）")
    print(f"  一层合计 = {shape.layer_parameter_count}；{shape.layers} 层合计 = "
          f"{shape.total_parameter_count}")


def section_2_stages() -> None:
    """第 2 节：一次堆叠前向的五个阶段（教材第 2 章用的是同一张表）."""
    _section("2 五个阶段：进入 / 块 / 记账 / 传递 / 出口")
    for line in stack_stage_lines():
        print(f"  {line}")
    print()
    print(f"  FAMILY_OUTCOMES 里的一行（示例）：{FAMILY_OUTCOMES['AssemblyError'][:48]}…")


def section_3_forward() -> None:
    """第 3 节：逐层读数表（增益与直通占比）."""
    _section("3 逐层读数：‖x‖ / ‖y‖ / 增益 / 直通占比")
    params = make_stack_parameters(
        make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO), seed=SEED
    )
    forward = stack_forward(params, _sample_inputs())
    print(f"  {forward.summary_line()}")
    print()
    for line in forward.census_table():
        print(line)
    print()
    print(f"  逐层增益     {['%.6f' % value for value in forward.gain_profile()]}")
    print(f"  逐层直通占比 {['%.4f' % value for value in forward.carry_profile()]}")


def section_4_carry_share() -> None:
    """第 4 节：直通占比 —— 残差开是正数、残差关**恰好**是 0."""
    _section("4 直通占比：残差开的正数 vs 残差关的 0.0")
    params = make_stack_parameters(
        make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO), seed=SEED
    )
    for use_residual in (True, False):
        forward = stack_forward(params, _sample_inputs(), use_residual=use_residual)
        label = "残差 开" if use_residual else "残差 关"
        values = ["%.4f" % value for value in forward.carry_profile()]
        print(f"  {label} | 直通占比 {values}")
    print()
    print("  读数里带着 use_residual：否则同一个 0.30 会指两件不同的事")


def section_5_gradient_profile() -> None:
    """第 5 节：逐层 ‖dx‖ —— day079 的一个数变成今天的一条序列."""
    _section("5 逐层 ‖dx‖：从“一个数”到“一条序列”")
    params = make_stack_parameters(
        make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO), seed=SEED
    )
    norms = stack_gradient_profile(params, _sample_inputs(), _sample_target())
    print("  层号 |        ‖dx‖ | 与上一层的比")
    print("  " + "-" * 42)
    ratios = (None,) + relative_gradient_change(norms)
    for index, value in enumerate(norms):
        ratio = ratios[index]
        text = "—" if ratio is None else f"{ratio:.6f}"
        print(f"  {index:>4} | {value:>11.6f} | {text:>12}")
    print()
    forward = stack_forward(params, _sample_inputs())
    loss_norm = _loss_gradient_norm(forward)
    print(f"  链内首尾之比 ‖dx_0‖ / ‖dx_{len(norms) - 1}‖ = {norms[0] / norms[-1]:.6f}")
    print(f"  整体比       ‖dx_0‖ / ‖dy_N‖ = {norms[0] / loss_norm:.6f}"
          f"（分母 ‖dy_N‖ = {loss_norm:.6f}，它**不在** dx 序列里）")


def section_6_gradients() -> None:
    """第 6 节：两处梯度校验（整条链 + 第 0 层的八块参数）."""
    _section("6 两处梯度校验：整条链的输入 + 第 0 层的八块参数")
    params = make_stack_parameters(
        make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO), seed=SEED
    )
    sample, goal = _sample_inputs(), _sample_target()
    chain = check_stack_input_gradient(params, sample, goal)
    print(f"  kind={KIND_STACK}（名单 {[GRAD_STACK_INPUTS]}）")
    print(f"  {chain.summary_line()}")
    for line in chain.detail_lines():
        print(line)
    print()
    layer = check_layer_parameter_gradients(params, sample, goal, 0)
    print(f"  kind={KIND_LAYER}（名单 {len(LAYER_GRADIENT_TARGETS)} 项，不含 'inputs'）")
    print(f"  {layer.summary_line()}")
    for line in layer.detail_lines():
        print(line)
    print()
    print("  为什么名单里没有 'inputs'：中间层的输入不是自变量，它由前一层算出来")


def section_7_properties() -> None:
    """第 7 节：六条性质."""
    _section("7 六条性质：保形 / 与手写循环一致 / 确定性 / 恒等 / 参数量 / 组装")
    shape = make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO)
    params = make_stack_parameters(shape, seed=SEED)
    script = assembly_script(shape)
    report = check_properties(shape, params, _sample_inputs(), script=script)
    print(f"  {report.summary_line()}")
    for item in report.outcomes:
        print(f"    {item.summary_line()}")
    print()
    print(f"  六条性质的名单：{list(STACK_PROPERTIES)}")
    for note in STACK_NOTES:
        print(f"  边界：{note}")


def section_8_assembly() -> None:
    """第 8 节：PyTorch 组装脚本（生成不运行）."""
    _section("8 PyTorch 组装：生成一段可独立运行的脚本")
    shape = make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO)
    script = assembly_script(shape, num_heads=2)
    facts = assembly_facts(script)
    print(f"  依赖：{ASSEMBLY_REQUIREMENT}（本包本身不 import torch）")
    print(f"  类：{facts['classes']}")
    print(f"  nn 模块：{facts['modules']}")
    print(f"  CONFIG：{facts['config']}")
    print(f"  行数：{facts['line_count']}；有 main：{facts['has_main']}")
    print(f"  用到的名字（前 12 个）：{list(parse_script_module_names(script))[:12]}")
    print()
    print("  —— 脚本前 24 行 ——")
    for line in script.splitlines()[:24]:
        print(f"  | {line}")
    gap = shape.layers * 4 * shape.hidden
    print()
    print(f"  PyTorch 比解析式多 {gap} 个参数 = {shape.layers} 层 × 4d"
          f"（nn.MultiheadAttention 的四个投影**带**偏置）")
    print("  这个差异必须被看见，而不是被凑平：本课的解析式是 4d²，PyTorch 是 4d² + 4d")


def section_9_study() -> None:
    """第 9 节：堆叠实验（2 变体 × 6 档深度）."""
    _section("9 堆叠实验：2 个变体 × 6 档深度")
    for variant in STUDY_VARIANTS:
        print(f"  {variant:<9} {VARIANT_DESCRIPTIONS[variant]}")
    print()
    study = stack_study(
        depths=DEFAULT_STUDY_DEPTHS, hidden=HIDDEN, ffn_ratio=FFN_RATIO, tokens=TOKENS, seed=SEED
    )
    for line in study.table_lines():
        print(line)
    print()
    print(f"  {study.summary_line()}")
    print(f"  判决（bare 比 residual 差一个数量级以上，且每层倍数极差 < 2）：{study.verdict_ok}")


def section_10_independent_of_depth() -> None:
    """第 10 节：这一课的值钱结论 —— 每层倍数与总深度几乎无关."""
    _section("10 “每层倍数”与总深度几乎无关（残差开时）")
    study = stack_study(
        depths=DEFAULT_STUDY_DEPTHS, hidden=HIDDEN, ffn_ratio=FFN_RATIO, tokens=TOKENS, seed=SEED
    )
    print("  深度 | 残差开：每层倍数 | 残差关：每层倍数 | 残差开：整体比 | 残差关：整体比")
    print("  " + "-" * 74)
    residual_steps = study.step_ratios(VARIANT_RESIDUAL)
    bare_steps = study.step_ratios(VARIANT_BARE)
    residual_ratios = study.ratios(VARIANT_RESIDUAL)
    bare_ratios = study.ratios(VARIANT_BARE)
    for index, depth in enumerate(study.depths):
        print(
            f"  {depth:>4} | {residual_steps[index]:>18.6f} | {bare_steps[index]:>16.6f} | "
            f"{residual_ratios[index]:>13.6f} | {bare_ratios[index]:>13.2e}"
        )
    print()
    print(f"  残差开的每层倍数极差 max/min = {study.step_ratio_spread:.6f}"
          "（< 2 ⇒ 与深度无关）")
    print(f"  残差关的每层倍数极差 max/min = "
          f"{max(bare_steps) / min(bare_steps):.6f}（自己就开始乱跳）")
    print()
    print("  几何平均的理由：相邻比值是**相乘**的。一条 4 层的链上实测这条恒等式：")
    shape = make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO)
    params = make_stack_parameters(shape, seed=SEED)
    _forward, grads = stack_loss_gradient(params, _sample_inputs(), _sample_target())
    norms = grads.norms()
    steps = relative_gradient_change(norms)
    gm = geometric_mean(steps)
    print(f"    逐层 ‖dx‖ = {['%.6f' % value for value in norms]}")
    print(f"    相邻比值   = {['%.6f' % value for value in steps]}")
    print(f"    几何平均   = {gm:.6f}；它的 (N-1) 次方 = {gm ** (len(norms) - 1):.6f}")
    print(f"    而 ‖dx_last‖ / ‖dx_0‖ = {norms[-1] / norms[0]:.6f} —— 逐位对上")
    print(f"    （注意 overall_ratio 是另一个量：‖dx_0‖ / ‖dy_N‖ = "
          f"{norms[0] / _loss_gradient_norm(_forward):.6f}）")


def section_11_falsification() -> None:
    """第 11 节：反证与边界（错法差多少 / 两天的判据在哪里接上）."""
    _section("11 反证与边界：错法差多少、哪一条守卫会拦下来")
    from smart_research_agent.encoder_decoder.layers import encoder_block_backward
    from smart_research_agent.transformer_core.layers import mse_gradient

    shape = make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO)
    params = make_stack_parameters(shape, seed=SEED)
    forward, grads = stack_loss_gradient(params, _sample_inputs(), _sample_target())
    top = mse_gradient(forward.output, _sample_target())
    layer0 = forward.layer_at(0)
    wrong = encoder_block_backward(layer0.block, layer0.params, top)
    correct = grads.layer_at(0)
    print("  错法：每一层都用同一份 grad_output = dLoss/dy_N（形状全对、也能跑）")
    print(f"    第 0 层 dW_in 相对差（错法 vs 正确）= "
          f"{relative_matrix_error(wrong.grad_ffn_w_in, correct.grad_ffn_w_in):.6e}")
    print(f"    正确的那一份与数值差分比：< 1e-6（第 6 节已量）")
    print()
    print("  边界：残差关 + 分支全零 + 两层 —— 撞上 day075 的全零行守卫")
    deflated = StackParameters(
        blocks=tuple(zero_branches(item) for item in params.blocks),
        attentions=tuple(zero_attention(item) for item in params.attentions),
    )
    one_layer = StackParameters(blocks=(deflated.blocks[0],), attentions=(deflated.attentions[0],))
    single = stack_forward(one_layer, _sample_inputs(), placement=NORM_PRE, use_residual=False)
    print(f"    一层：算完了，输出是{'全零矩阵' if all(v == 0.0 for r in single.output for v in r) else '非零矩阵'}"
          "（零矩阵是一个合法的结果）")
    try:
        stack_forward(deflated, _sample_inputs(), placement=NORM_PRE, use_residual=False)
    except CoreNumericError as error:
        print(f"    两层：{type(error).__name__} —— 第二层的注意力拿到全零行，被显式拒绝")
        print(f"    守卫的原话：{str(error)[:64]}…")
    print()
    print("  这不是 bug：一个把输入压成全零的堆叠，在 day075 的眼里就是一次")
    print("  “不该交给这一层猜”的调用。两天的判据在这里接上了。")
    print()
    print(f"  初始化幅度（与本课同源）：scale = {DEFAULT_INIT_SCALE}")
    print(f"  缺一份参数时的失败族：{AssemblyError.__name__}"
          f"（继承 {AssemblyError.__mro__[1].__name__}）")
    loss = stack_loss(params, _sample_inputs(), _sample_target())
    print(f"  样本损失（前向的一个旁读）：{loss:.6f}")


def main() -> None:
    """十一节依次跑完."""
    print("day080 / M7-D5 —— 从一个块到一条链（transformer_stack）")
    print("全部离线：纯 Python 算术；生成的 PyTorch 脚本只生成、不在本进程里运行。")
    section_1_shape()
    section_2_stages()
    section_3_forward()
    section_4_carry_share()
    section_5_gradient_profile()
    section_6_gradients()
    section_7_properties()
    section_8_assembly()
    section_9_study()
    section_10_independent_of_depth()
    section_11_falsification()


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
