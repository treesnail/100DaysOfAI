"""离线演示：可训练的注意力（day075 / M7-D1）.

跑法：

```bash
python scripts/attention_demo.py      # 十节，输出写到 outputs/attention_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。
产物只写在 ``outputs/`` 下（幂等，可随时删）。

十节里最值得看的是第 3、5、7、9、10 节：

```text
3    前向七步：单位矩阵配置下每一个数都能手算（softmax([0, 1/√2]) = (0.330238, 0.669762)）
5    反向七步：解析梯度与数值差分的五项逐点对照（实测 1e-11 量级）
7    四条性质：行和为 1 / 非负 / 因果无泄漏 / 置换等变（**掩码会破坏最后一条**）
9    induction 训练：损失 → 0、命中率 → 100%，而峰值权重只到 0.44
10   冻结对照：只训 q/k 时注意力变尖而损失降不下去——"损失到 0'不等于'学到机制"
```
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.math_foundations.optim import (  # noqa: E402
    AdamOptimizer,
    SGDOptimizer,
    cosine_schedule,
)
from smart_research_agent.transformer_core import (  # noqa: E402
    ATTENTION_STAGES,
    ATTENTION_STAGE_DESCRIPTIONS,
    ATTENTION_STAGE_SHAPES,
    GRADIENT_TARGETS,
    GRADIENT_TARGET_FORMULAS,
    GRADIENT_SOURCE_NUMERIC,
    batch_accuracy,
    batch_gradients,
    batch_loss,
    batch_mean_peak_weight,
    check_attention_gradients,
    check_properties,
    compare_with_retrieval,
    default_parameters,
    make_induction_batch,
    make_induction_task,
    permutation_gap,
    self_attention,
    train_attention,
)
from smart_research_agent.transformer_core.types import AttentionParams  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "attention_demo.txt"

#: 两个正交的 one-hot token（``d = 2``）——所有手算的支点。
ONE_HOT = ((1.0, 0.0), (0.0, 1.0))

#: 三个一般的输入行（``d = 3``）。
SMALL_INPUTS = (
    (1.0, 0.0, 0.5),
    (0.0, 1.0, 0.25),
    (0.5, 0.5, 0.0),
)
SMALL_TARGET = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


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


def round_matrix(matrix) -> tuple:
    """把矩阵四舍五入到 6 位（打印用）."""
    return tuple(tuple(round(value, 6) for value in row) for row in matrix)


def identity(size: int):
    """单位矩阵."""
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size)) for row in range(size)
    )


def identity_params(size: int) -> AttentionParams:
    """四个投影都是单位矩阵的注意力层."""
    matrix = identity(size)
    return AttentionParams(
        w_query=matrix, w_key=matrix, w_value=matrix, w_output=matrix
    )


def small_params() -> AttentionParams:
    """一个手算友好的 3×3 配置：``W_q = 0.5I``、``W_k`` 带小扰动、``W_v = W_o = I``."""
    return AttentionParams(
        w_query=tuple(
            tuple(0.5 if row == column else 0.0 for column in range(3)) for row in range(3)
        ),
        w_key=((0.4, 0.1, 0.0), (0.0, 0.4, 0.1), (0.1, 0.0, 0.4)),
        w_value=identity(3),
        w_output=identity(3),
    )


def main() -> None:
    """十节演示."""
    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：从'公式'到'可训练'——七个阶段")
    for stage in ATTENTION_STAGES:
        print(f"  {stage:<9} {ATTENTION_STAGE_SHAPES[stage]}")
        print(f"            {ATTENTION_STAGE_DESCRIPTIONS[stage]}")
    print()
    print("  四个维度不合成一个 d：d_k 只影响打分尺度、d_v 只影响混合结果宽度、")
    print("  d_out 只影响最终输出宽度——而缩放系数只由 d_k 决定。")
    params = small_params()
    print(f"  {params.describe()}")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：乘法口径（``x·Wᵀ`` 与两条反向式子）")
    print("  权重形状 (d_out, d_in)、投影 = x·Wᵀ —— 与 nn.Linear 一致")
    weight = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    inputs = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    from smart_research_agent.transformer_core.types import project

    print(f"  x = {inputs}")
    print(f"  W = {weight}   （2×3：输出 2 维、输入 3 维）")
    print(f"  x·Wᵀ = {round_matrix(project(inputs, weight))}   ← 第一行取到 W 的第一行")
    print("  反向：dW = gradᵀ·x、dx = grad·W（两条式子各有一条手算测试）")

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：前向七步（单位矩阵配置下每一个数都能手算）")
    forward = self_attention(identity_params(2), ONE_HOT, causal=True)
    scale = 1.0 / math.sqrt(2.0)
    p = 1.0 / (1.0 + math.exp(scale))
    print(f"  Q = K = V = x（四个投影都是 I），缩放系数 1/√2 = {scale:.6f}")
    print(f"  raw = x·xᵀ = {round_matrix(forward.raw)}   （一行与自己的点积是 1、与别人的是 0）")
    print(f"  weights（因果）= {round_matrix(forward.weights)}")
    print(f"    第 0 行只看自己 → (1, 0)；第 1 行看 [0, 1/√2] → softmax = ({p:.6f}, {1 - p:.6f})")
    print(f"     手算 p = 1/(1+e^(1/√2)) = {p:.6f}")
    print(f"  context = {round_matrix(forward.context)}   （输入的凸组合）")
    print(f"  output  = {round_matrix(forward.output)}    （W_o = I，因此与 context 相同）")
    print(f"  {forward.summary_line()}")

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：掩码的三种来源（以及两种拒绝）")
    full = self_attention(identity_params(2), ONE_HOT, causal=False)
    print(f"  无掩码：weights = {round_matrix(full.weights)}")
    print("    第 0 行看到 [1/√2, 0]，权重反过来（它更看重第一个位置）")
    print(f"  因果掩码：上三角恰好 {forward.weights[0][1]}（**精确的 0.0**，不是 1e-17）")
    print("  causal=True 与显式掩码同时给 → 当场报错（'到底用哪张'要读代码才知道）")
    print("  输入行全零 → 当场报错（它让 softmax 给出均匀分布，看起来像'还没学到'）")

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：反向七步与五项梯度校验")
    report = check_attention_gradients(small_params(), SMALL_INPUTS, SMALL_TARGET)
    print(f"  {report.summary_line()}   容差 {report.tolerance:.1e}、步长 {report.step:g}")
    print(f"  {'项':<10}{'点数':>6}{'最大相对误差':>16}   解析梯度公式")
    for outcome in report.outcomes:
        print(
            f"  {outcome.target:<10}{outcome.compared_points:>6}"
            f"{outcome.max_scaled_error:>16.2e}   {GRADIENT_TARGET_FORMULAS[outcome.target]}"
        )
    print()
    print("  五项里最特别的是 inputs：它是 q/k/v **三条链之和**——")
    print("  只写两条链的实现不会报错，只会让'靠近输入的那几层学得慢'。")
    print("  方法：把四个矩阵压平成一串数（day074 的 flatten_matrices），")
    print("  再对每个分量做中心差分（day074 的 calculus.gradient）。")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：softmax 的反向（一行公式，三种读法）")
    from smart_research_agent.transformer_core.layers import softmax_backward_row

    for weights_row, gradient_row in (
        ((0.5, 0.5), (1.0, 0.0)),
        ((0.25, 0.25, 0.25, 0.25), (1.0, 1.0, 1.0, 1.0)),
        ((1.0, 0.0, 0.0), (5.0, 7.0, -3.0)),
        ((p, 1 - p), (1.0, 0.0)),
    ):
        result = softmax_backward_row(weights_row, gradient_row)
        print(
            f"  s = {tuple(round(v, 6) for v in weights_row)}  g = {gradient_row}"
            f"  →  dScores = {tuple(round(v, 6) for v in result)}"
        )
    print("  读法：均匀分布 + 均匀梯度给出全 0；全押一个位置时其它位置拿不到梯度；")
    print("        括号里那一项是**整行的加权平均**，不是逐元素的。")

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：四条性质（其中一条在两种配置下答案不同）")
    properties = check_properties(small_params(), SMALL_INPUTS, causal=True)
    print(f"  {properties.summary_line()}")
    for outcome in properties.outcomes:
        print(f"    {outcome.summary_line()}")
        print(f"        {outcome.detail}")
    gap_free = permutation_gap(small_params(), SMALL_INPUTS, causal=False)
    gap_causal = permutation_gap(small_params(), SMALL_INPUTS, causal=True)
    print()
    print(f"  置换等变（把输入行倒过来）：无掩码 {gap_free:.2e}、因果掩码 {gap_causal:.4f}")
    print("  **因果掩码破坏它，这不是 bug**：掩码把'谁在谁前面'写进了那张表。")
    print("  但掩码只提供一种很粗的顺序——要表达'位置 3 与位置 5 的距离'，")
    print("  仍然需要位置编码（day078）。")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：与检索的类比（同一批输入的两种排序）")
    for label, forward_item in (
        ("单位配置 / 因果", self_attention(identity_params(2), ONE_HOT, causal=True)),
        ("小参数 / 因果", self_attention(small_params(), SMALL_INPUTS, causal=True)),
        ("小参数 / 无掩码", self_attention(small_params(), SMALL_INPUTS, causal=False)),
    ):
        comparison = compare_with_retrieval(forward_item, top_k=2)
        print(f"  {label:<16} {comparison.summary_line()}")
    print()
    print("  为什么三个读数都要：top-k 是离散量（k=1 时只取值 0 或 1，")
    print("  很多差别会被它吞掉），而秩相关是连续量。")
    print("  注意力是**可微的混合**（权重是一组概率），检索是**离散的选择**（取前 k 条）。")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：induction 任务——把这一层训练起来")
    tasks = make_induction_batch(4, seed=42)
    for task in tasks:
        print(f"    {task.describe()}")
    initial = default_parameters(6, seed=7)
    print()
    print(f"  初始：损失 {batch_loss(initial, tasks):.6f}、命中率 {batch_accuracy(initial, tasks):.0%}、"
          f"峰值 {batch_mean_peak_weight(initial, tasks):.4f}")
    report = train_attention(initial, tasks, optimizer=AdamOptimizer(0.05), steps=200)
    print(f"  {report.summary_line()}")
    print()
    print(f"  {'步':>5}   {'损失':>10}   {'命中率':>7}   {'峰值权重':>9}")
    for index in (0, 1, 3, 5, 8, 10, 20, 50, 100, 200):
        print(
            f"  {index:>5}   {report.trace.losses[index]:>10.6f}   "
            f"{report.accuracies[index]:>7.0%}   {report.peak_weights[index]:>9.4f}"
        )
    print()
    print("  三个读数一起看才说明问题：损失在降、命中率在涨、峰值也在升——")
    print("  但峰值只到 0.44，**没有变成'每行只看一处'**。")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：冻结对照——'损失到 0'不等于'学到机制'")
    for label, trainable in (
        ("四块全训 ", None),
        ("只训 q/k ", ("w_query", "w_key")),
    ):
        frozen = train_attention(
            default_parameters(6, seed=7),
            tasks,
            optimizer=AdamOptimizer(0.05),
            steps=200,
            trainable=trainable,
        )
        print(
            f"  {label} 损失 {frozen.initial_loss:.4f} → {frozen.final_loss:.4f}   "
            f"命中率 {frozen.initial_accuracy:.0%} → {frozen.final_accuracy:.0%}   "
            f"峰值 {frozen.initial_peak_weight:.3f} → {frozen.final_peak_weight:.3f}"
            f"   冻结 = {list(frozen.frozen) or '无'}"
        )
    print()
    print("  两个方向都反直觉：")
    print("    只训 q/k   注意力变尖了（峰值 0.59），但损失降不下去——")
    print("               '看对了地方'只是必要条件，还要 value/output 把看到的东西映射成目标")
    print("    四块全训   损失到 0、命中率 100%，而峰值只有 0.44——")
    print("               模型在 value 路径上把不需要的分量抵消掉了")
    print("  结论：没有任何一个读数能单独说明'学到了什么'。")
    print()

    # 两种梯度源的对账
    print("  两种梯度源的对账（梯度实现对不对，靠它验证）：")
    for label, optimizer, steps in (
        ("SGD  0.05", SGDOptimizer(0.05), 6),
        ("Adam 0.05", AdamOptimizer(0.05), 6),
        ("Adam 0.10", AdamOptimizer(0.10), 30),
    ):
        first = train_attention(default_parameters(6, seed=7), tasks, optimizer=optimizer, steps=steps)
        second = train_attention(
            default_parameters(6, seed=7),
            tasks,
            optimizer=type(optimizer)(optimizer.learning_rate),
            steps=steps,
            source=GRADIENT_SOURCE_NUMERIC,
        )
        worst = max(
            abs(a - b) for a, b in zip(first.trace.losses, second.trace.losses, strict=True)
        )
        print(f"    {label:<10} {steps:>3} 步   损失轨迹最大差 {worst:.3e}")
    print("    差别来自数值差分自身（约 1e-11）；Adam 比 SGD 大约 1000 倍——")
    print("    它按 √v̂ 归一化，会放大梯度分量的**相对**误差。")
    print("    两支都收敛，因此要**验证梯度实现**，请优先用 SGD。")

    # 批量的两条读数（说明"梯度是有用的"）
    grads = batch_gradients(default_parameters(6, seed=7), tasks)
    print()
    print(f"  批量梯度：{len(grads)} 个分量、范数 "
          f"{math.sqrt(math.fsum(v * v for v in grads)):.6f}")
    print(f"  学习率调度示例（cosine，4 步）："
          f"{[round(cosine_schedule(step, base_lr=0.05, total_steps=10), 6) for step in (1, 2, 3, 4)]}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / OUTPUT_NAME).open("w", encoding="utf-8") as handle:
        original = sys.stdout
        sys.stdout = _Tee(original, handle)
        try:
            main()
        finally:
            sys.stdout = original
    print(f"\n输出已写入 {OUTPUT_DIR / OUTPUT_NAME}")
