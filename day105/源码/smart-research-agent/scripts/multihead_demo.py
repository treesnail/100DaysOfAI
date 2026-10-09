"""离线演示：多头注意力（day076 / M7-D2）.

跑法：

```bash
python scripts/multihead_demo.py      # 十节，输出写到 outputs/multihead_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。
产物只写在 ``outputs/`` 下（幂等，可随时删）。

十节里最值得看的是第 2、4、5、8、9 节：

```text
2    手算配置：heads=2 时每一头、每一行的分布与 context 都能在纸上算出来
4    heads=1 **逐位**等于 day075（前向与反向），而 heads=2 的输出明显不同
5    九步反向：解析梯度与数值差分的五项逐点对照（实测 1e-11 量级）
8    可达集合：**同一批参数**下单头只能到一条线段、双头能到一个正方形
9    同参数量对照：heads=1/2/3 的参数量一字不差，而头间差异只在前两者里非零
```
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.math_foundations.optim import (  # noqa: E402
    AdamOptimizer,
    SGDOptimizer,
)
from smart_research_agent.multi_head import (  # noqa: E402
    MULTIHEAD_GRADIENT_FORMULAS,
    MULTIHEAD_PROPERTIES,
    MULTIHEAD_STAGE_DESCRIPTIONS,
    MULTIHEAD_STAGE_SHAPES,
    MULTIHEAD_STAGES,
    WITNESS_ONE_HEAD_DISTANCE,
    check_multihead_gradients,
    compare_heads,
    designed_witness,
    head_disagreement,
    head_gradient_shares,
    head_parameter_gradients,
    head_scale,
    multi_head_attention,
    multi_head_backward,
    permute_head_blocks,
    project_heads_separately,
    realize_with_weights,
    train_multi_head,
    witness_report,
)
from smart_research_agent.multi_head.types import (  # noqa: E402
    HeadPartition,
    MultiHeadShape,
    head_gradient_norms,
)
from smart_research_agent.multi_head.verify import check_properties  # noqa: E402
from smart_research_agent.transformer_core.layers import (  # noqa: E402
    attention_backward,
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.train import (  # noqa: E402
    default_parameters,
    make_induction_batch,
)
from smart_research_agent.transformer_core.types import (  # noqa: E402
    AttentionParams,
    relative_matrix_error,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "multihead_demo.txt"

#: 两个 token 的 4 维输入：**两个头看到的打分差不同**（1/√2 与 3/√2）——
#: 因此"各头看到的东西不一样"在这条样本上是可手算的。
HAND_INPUTS = (
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 2.0, 1.0),
)

#: 头数序列（**全部整除 6**）：同参数量对照的三个配置。
HEAD_COUNTS = (1, 2, 3)


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


def identity(size: int) -> tuple:
    """单位矩阵."""
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size)) for row in range(size)
    )


def identity_params(size: int) -> AttentionParams:
    """四个投影都是单位矩阵的注意力层（最容易手算的配置）."""
    matrix = identity(size)
    return AttentionParams(w_query=matrix, w_key=matrix, w_value=matrix, w_output=matrix)


def repeated_block_params(size: int, heads: int) -> AttentionParams:
    """**每一头完全相同**的参数（用来把"多头退化"变成可看见的数）."""
    partition = HeadPartition(size, heads)
    block = tuple(
        tuple(1.0 if (index + column) % size in (0, 1) else 0.0 for column in range(size))
        for index in range(partition.width)
    )
    stacked = tuple(row for _ in range(partition.heads) for row in block)
    return AttentionParams(
        w_query=stacked,
        w_key=stacked,
        w_value=stacked,
        w_output=identity(size),
    )


def main() -> None:
    """十节演示."""
    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：九个阶段——多出来的两步正是这一课的全部内容")
    for stage in MULTIHEAD_STAGES:
        print(f"  {stage:<9} {MULTIHEAD_STAGE_SHAPES[stage]}")
        print(f"            {MULTIHEAD_STAGE_DESCRIPTIONS[stage]}")
    print()
    print("  七个阶段 + split + merge：前七个与 day075 逐字相同，")
    print("  多出来的两步只是'一次切片'与'一次拼接'——而难点不在这两步，在三件事：")
    print("    ① 每头缩放分母是 √d_h（不是 √d_k）")
    print("    ② 四个投影被所有头共用（反向要按行拼回）")
    print("    ③ 头与维度的对应只是一个记账约定")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：手算配置——每一头、每一行的数都能在纸上写出来")
    params = identity_params(4)
    shape = MultiHeadShape(params.shape, 2)
    print(f"  {shape.summary_line()}")
    print(f"  划分：{shape.keys_partition.describe()}")
    print(f"  缩放：head_dim={shape.head_dim} → 1/√{shape.head_dim} = {shape.scale:.6f}")
    print(
        f"        单头用的是 1/√{shape.attention.keys} = {shape.single_head_scale:.6f}，"
        f"两者之比 {shape.scale_ratio:.6f} = √heads"
    )
    forward = multi_head_attention(params, HAND_INPUTS, heads=2, causal=True)
    print()
    print(f"  输入 x = {HAND_INPUTS}")
    for head in range(forward.heads):
        print(f"  {forward.head_summary_line(head)}")
    print("  每一头的权重（上下两行是两个 token）：")
    for head in range(forward.heads):
        print(f"    head {head}  {round_matrix(forward.head_weights[head])}")
    print(f"  拼接后的 context = {round_matrix(forward.merged_context)}")
    print(f"  输出（W_o = I）= {round_matrix(forward.output)}")
    print()
    print("  第 0 头：softmax([0, 1/√2]) 的第一个分量 = 0.330238")
    print("  第 1 头：softmax([2/√2, 5/√2]) 的第一个分量 = 0.107042")
    print("  **两个头在同一行上给出了两个不同的分布**——这就是'多头'的全部意思。")
    print(f"  这一层一共产生了 {forward.distributions} 个条件分布（单头只有 2 个）")

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：划分——切与拼（往返恒等，且可以打印出来看）")
    partition = HeadPartition(6, 3)
    print(f"  {partition.describe()}")
    matrix = tuple(tuple(float(row * 10 + column) for column in range(4)) for row in range(6))
    parts = partition.split_rows(matrix)
    print(
        f"  6×4 的矩阵按行切成 {len(parts)} 块："
        f"{len(parts[0])} 行、{len(parts[1])} 行、{len(parts[2])} 行"
    )
    print(f"  merge_rows(split_rows(M)) == M ？ {partition.merge_rows(parts) == matrix}")
    print()
    print("  **权重按行切、激活按列切**：")
    print("    W_q 的形状是 (d_k, d_in)，d_k 在它的**行**上")
    print("    Q = x·W_qᵀ 的形状是 (n, d_k)，d_k 在它的**列**上")
    keys_partition = shape.keys_partition
    print(
        "  本层两条都验过："
        f"merge_columns(head_queries) == queries ？ "
        f"{keys_partition.merge_columns(forward.head_queries) == forward.queries}"
    )

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：heads=1 **逐位**等于 day075（前向、context 与反向）")
    single = multi_head_attention(params, HAND_INPUTS, heads=1, causal=True)
    classic = self_attention(params, HAND_INPUTS, causal=True)
    target = tuple(reversed(HAND_INPUTS))
    mine_grad = multi_head_backward(single, mse_gradient(single.output, target))
    classic_grad = attention_backward(classic, mse_gradient(classic.output, target))
    print(f"  heads=1 输出 == day075 输出 ？ {single.output == classic.output}")
    print(f"  heads=1 权重 == day075 权重 ？ {single.head_weights[0] == classic.weights}")
    print(f"  heads=1 梯度 == day075 梯度 ？ {mine_grad.flatten() == classic_grad.flatten()}")
    two = multi_head_attention(params, HAND_INPUTS, heads=2, causal=True)
    print(
        "  而 heads=2 的输出**明显不同**："
        f"与单头的最大相对误差 {relative_matrix_error(two.output, single.output):.4f}"
    )
    print("  → '多头是单头的严格超集'这句话，在 heads=1 时是**逐位相等**的。")

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：九步反向——五项梯度校验（解析 vs 数值）")
    for formula_target, formula in MULTIHEAD_GRADIENT_FORMULAS.items():
        print(f"  {formula_target:<9} {formula}")
    print()
    tasks = make_induction_batch(1)
    sample_params = default_parameters(6)
    print(f"  {'heads':>5} | {'通过':>4} | {'最大相对误差':>12} | 逐项")
    for heads in HEAD_COUNTS:
        report = check_multihead_gradients(
            sample_params, tasks[0].inputs, tasks[0].target, heads=heads
        )
        detail = "、".join(
            f"{item.target}={item.max_scaled_error:.1e}" for item in report.outcomes
        )
        passed = sum(1 for item in report.outcomes if item.passed)
        print(
            f"  {heads:>5} | {passed:>4} | {report.worst_scaled_error:>12.2e} | {detail}"
        )
    print()
    print("  容差 1e-6，实测 1e-11 量级——**隔着五个数量级**，")
    print("  而'真的写错了'通常是 1e-2 以上。这个距离就是这条护栏的区分度。")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：六条性质（其中两条是本层专有）")
    properties = check_properties(sample_params, tasks[0].inputs, heads=2, causal=True)
    print(f"  {properties.summary_line()}")
    for outcome in properties.outcomes:
        print(f"    {outcome.summary_line()}")
    print()
    print("  第 4 条（往返恒等）与第 5 条（heads=1 等于 day075）用 **==** 判据：")
    print("  连续块在 heads=1 时的唯一划分就是整张矩阵，因此两者是同一串数。")
    print("  第 6 条（头序只是记账）用 1e-12：重排会把浮点求和的**顺序**也换掉。")
    permuted = permute_head_blocks(sample_params, 2)
    permuted_gap = relative_matrix_error(
        multi_head_attention(permuted, tasks[0].inputs, heads=2).output,
        multi_head_attention(sample_params, tasks[0].inputs, heads=2).output,
    )
    print(f"  一致重排头块之后的输出差：{permuted_gap:.2e}")
    print(f"  六条性质的名单：{'、'.join(MULTIHEAD_PROPERTIES)}")

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：头间差异——多头最大的失败模式要能被看见")
    hand = head_disagreement(forward)
    print(f"  手算配置（两个头确实不同）：{hand.summary_line()}")
    print(f"    TV 的手算值（第 1 行 = |p − q|）：{hand.max_total_variation:.6f}")
    degenerate = multi_head_attention(
        repeated_block_params(6, 2), tasks[0].inputs, heads=2, causal=True
    )
    print(f"  每头完全相同（构造的退化配置）：{head_disagreement(degenerate).summary_line()}")
    print(
        f"    degenerate={head_disagreement(degenerate).degenerate}——"
        "'付了 2 份参数、只用了一份注意力'"
    )
    single_head = multi_head_attention(sample_params, tasks[0].inputs, heads=1)
    print(f"  单头（没有可比的头对）：{head_disagreement(single_head).summary_line()}")
    print("    单头**不是**'退化'：没有可以退化的对象——两者在数值上都是 0，含义完全不同。")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：可达集合——单头是一条线段，双头是一个正方形")
    witness = designed_witness()
    report = witness_report()
    print(f"  见证样本：{witness.to_dict()['heads']} 头、d_out=2、row={witness.row}")
    print(f"  head 0 的 pull = {witness.to_dict()['head_zero_pulls']}")
    print(f"  head 1 的 pull = {witness.to_dict()['head_one_pulls']}")
    print()
    print("  **同一批参数**下的两种结构：")
    print(f"    {report.summary_line()}")
    print(f"    单头凸包 = {report.one_head_hull}   ← 第三点是前两点的**中点**，共线")
    print(f"    多头凸包 = {report.multi_head_hull}   ← 张开了一整个面")
    print(f"    单头到目标的距离 = {report.one_head_distance:.6f}")
    print(f"    公式值 1/√2      = {WITNESS_ONE_HEAD_DISTANCE:.6f}")
    print(f"    多头到目标的距离 = {report.multi_head_distance:.6f}")
    print()
    print(f"  兑现目标 (1, 1) 的两份分布：{witness.witness_weights}")
    realized = realize_with_weights(
        multi_head_attention(witness.params, witness.inputs, heads=witness.heads, causal=True),
        witness.row,
        witness.witness_weights,
    )
    print(f"    代进前向核对：{tuple(round(value, 6) for value in realized)} == (1.0, 1.0)")
    print("    两份分布的 TV = 1.0（最大值）——'多头真的分了两路'的最强形态")
    print()
    print("  **注意区分两件事**：")
    print("    可达集合回答'输出空间里哪些点到得了'（存在性，一个几何问题）")
    print("    它**不回答**'训练能不能找到那组参数'（优化问题）")
    print("    上面那个 (1,1) 是被**构造**出来的，不是被训练出来的。")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：训练与同参数量对照")
    train_params = default_parameters(6)
    train_tasks = make_induction_batch(4)
    for heads in HEAD_COUNTS:
        trained = train_multi_head(
            train_params,
            train_tasks,
            heads=heads,
            optimizer=AdamOptimizer(0.05),
            steps=120,
        )
        print(f"  {trained.summary_line()}")
    print()
    print("  **参数量与 heads 无关**（四个投影的形状里根本没有 heads）：")
    print(f"    三行的参数量 = {train_params.parameter_count()}（同一批初始参数）")
    comparison = compare_heads(
        train_params,
        train_tasks,
        heads_list=HEAD_COUNTS,
        optimizer_factory=lambda: AdamOptimizer(0.05),
        steps=120,
    )
    print(f"  {comparison.summary_line()}")
    for line in comparison.table_lines():
        print(line)
    print()
    print("  读法（四个读数一起看）：")
    print("    heads=1 的头间差异**恒为 0**（没有第二个头），而它照样把损失压到 0")
    print("    heads=2/3 各头给出了**不同的分布**——差异是真实存在的，不是白买的")
    print("    但'哪个 heads 更好'这份表回答不了：它是一次观测，不是统计结论")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：两种梯度源的对账（以及为什么优先用 SGD）")
    trajectories: dict[str, list[float]] = {}
    for source in ("analytic", "numeric"):
        run = train_multi_head(
            train_params,
            train_tasks[:1],
            heads=2,
            optimizer=SGDOptimizer(0.05),
            steps=4,
            source=source,
        )
        trajectories[source] = list(run.trace.losses)
    gap = max(
        abs(a - b)
        for a, b in zip(trajectories["analytic"], trajectories["numeric"], strict=True)
    )
    print("  同一批参数、同一个优化器、同一批样本，只换梯度来源：")
    for source, losses in trajectories.items():
        print(f"    {source:<9} 损失轨迹 {[round(value, 8) for value in losses]}")
    print(f"    最大绝对差 {gap:.3e} —— 这是**数值差分自身**的误差量级，不是实现问题")
    print()
    print("  梯度分块（'哪一头在学'的读数，W_o 不在任何一头里）：")
    probe = multi_head_attention(train_params, train_tasks[0].inputs, heads=2, causal=True)
    gradients = multi_head_backward(
        probe, mse_gradient(probe.output, train_tasks[0].target)
    )
    norms = head_gradient_norms(gradients, probe.shape)
    shares = head_gradient_shares(gradients, probe.shape)
    for head in range(probe.heads):
        print(f"    head {head}  ‖dW‖ = {norms[head]:.6f}，占比 {shares[head]:.1%}")
    per_head = head_parameter_gradients(
        probe, mse_gradient(probe.output, train_tasks[0].target)
    )
    print(f"    逐头记录（两处算法对账）：{per_head[0].summary_line()}")
    identity_gap = relative_matrix_error(project_heads_separately(probe), probe.output)
    print(f"    merge→project 恒等式的实测差：{identity_gap:.2e}（浮点下是**近似**）")

    # ------------------------------------------------------------------ 收尾
    section("收尾：这一课留下的三句话")
    print("  ① 多头不是'更多参数'，而是'同一批参数买到 heads 份注意力'——")
    print("     参数量一字不差，因此 H=1/2/3 的对照是同参数量对照。")
    print("  ② 可达集合给出了结构性的差别：单头只能到一条线段、双头能到一个正方形，")
    print("     但那是**存在性**，它不承诺训练能找到。")
    print("  ③ 该逐位相等的地方就说逐位相等（heads=1、往返恒等），")
    print("     该说近似的地方就说近似（merge→project、头序重排）——")
    print("     把近似写成相等，会让一次无关的重构变成一次假失败。")
    print()
    print(f"  缩放系数速查：head_dim=2 → {head_scale(2):.6f}；head_dim=4 → {head_scale(4):.6f}")


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
