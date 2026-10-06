"""离线演示：位置编码（day078 / M7-D3）.

跑法：

```bash
python scripts/position_demo.py      # 十一节，输出写到 outputs/position_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。
产物只写在 ``outputs/`` 下（幂等，可随时删）。

十一节里最值得看的是第 3、5、8、9、10 节：

```text
3    位移律：闭式 Σ_i cos(δ/f_i) 与实测的内积**逐位**相等，且 PE(p+δ) 恰是 PE(p) 的旋转
5    六项梯度校验：多出来的那一项（位置表）与数值差分对到 1e-11
     —— 位置重复的那条样本才真正区分"按位置累加"与"按行覆盖"
8    位置选择任务：下界 (n−1)/(n·V) = 0.125，而袋子预测器**恰好**达到它（下界是紧的）
9    四个编码刻度：无位置编码停在 0.125，位置编码跌到 0.003 量级
10   下界判决书：等变变体**停在界上**（0.125003），破对称变体**越过界**（0.0032 / 0.0003）
```
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.math_foundations.attention import (  # noqa: E402
    positional_encoding as day073_positional_encoding,
)
from smart_research_agent.math_foundations.optim import AdamOptimizer, SGDOptimizer  # noqa: E402
from smart_research_agent.positional_encoding import (  # noqa: E402
    POSITIONAL_BASE,
    POSITIONAL_GRADIENT_FORMULAS,
    POSITIONAL_PROPERTIES,
    POSITIONAL_STAGE_DESCRIPTIONS,
    POSITIONAL_STAGE_SHAPES,
    POSITIONAL_STAGES,
    PositionalShape,
    batch_mean_injection_ratio,
    check_positional_gradients,
    check_properties,
    closed_form_offset_inner,
    compare_encodings,
    default_parameters,
    floor_witness,
    frequency_of,
    inject,
    learnable_table,
    make_readout_task,
    orbit_floor,
    positional_backward,
    positional_forward,
    rotation_factor,
    sinusoidal_table,
    staggered_table,
    symmetry_break,
    symmetry_floor_study,
    train_positions,
    zero_table,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "position_demo.txt"

#: 手算配置：四维表（于是两对频率的周期是 1 与 100）、两个位置
HAND_DIMENSION = 4
HAND_POSITIONS = 4

#: 演示用的词表大小与序列长度（与 ``default_parameters(6)`` 对齐）
VOCABULARY = 6
LENGTH = 4


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


def identity(size: int) -> tuple:
    """单位矩阵."""
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size)) for row in range(size)
    )


def main() -> None:
    """十一节演示."""
    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：六个阶段——多出来的两步在**最前面**")
    for stage in POSITIONAL_STAGES:
        print(f"  {stage:<10} {POSITIONAL_STAGE_SHAPES[stage]}")
        print(f"             {POSITIONAL_STAGE_DESCRIPTIONS[stage]}")
    print()
    print("  比 day075 多出来的两步是 `positions` 与 `table`，而它们在**最前面**——")
    print("  于是反向时它们也在最后面，而'最后一步'恰好是最容易写错的一步：")
    print("    dx = dInjected            （逐位：加法注入的偏导数恰好是 1）")
    print("    dTable[p] = Σ dInjected[i]（按位置**累加**，不是覆盖）")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：手算配置——d=4、base=10000，两对频率的周期是 1 与 100")
    table = sinusoidal_table(HAND_POSITIONS, HAND_DIMENSION)
    print(f"  频率对的周期：f_0 = {frequency_of(0, 4):g}、f_1 = {frequency_of(1, 4):g}")
    print(f"  {table.describe()}")
    print(f"  PE(0) = {round_row(table.row(0))}")
    print(f"  PE(1) = {round_row(table.row(1))}")
    print(f"  PE(2) = {round_row(table.row(2))}")
    print(f"  PE(3) = {round_row(table.row(3))}")
    print(f"  每一行的范数 = {round_row(table.norms, 12)}")
    print(f"  √(d/2) = √2 = {math.sqrt(2):.12f}（**恰好**相等）")
    print("  → 因为它由两对 sin² + cos² = 1 组成，与位置无关。")
    print()
    print("  对照 day073 的写法（cos 用**隔壁**那个频率）：")
    staggered = staggered_table(HAND_POSITIONS, HAND_DIMENSION)
    print(f"  staggered PE(1) = {round_row(staggered.row(1))}")
    print(f"  staggered 各行范数 = {round_row(staggered.norms)}")
    print(
        f"  行范数是否恒定：aligned {table.aligned_within}、"
        f"staggered {staggered.aligned_within}"
    )
    print(f"  day073 的 positional_encoding(2, 4) 第 1 行 = "
          f"{round_row(day073_positional_encoding(2, 4)[1])}")
    print("  → **同一个名字、两种配对**：day073 那一份的范数随位置变化。")

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：位移律与旋转——'相对距离'是算出来的")
    long_table = sinusoidal_table(8, HAND_DIMENSION)
    print(f"  为了量到 δ = 5，这里换一张 8 行的表（{long_table.describe()}）")
    print(f"  {'δ':>2} | {'实测 <PE(0),PE(δ)>':>22} | {'任意起点 p=2':>22} | {'闭式 Σcos(δ/f_i)':>22} | 差")
    for offset in (1, 2, 3, 5):
        closed = closed_form_offset_inner(HAND_DIMENSION, offset, base=POSITIONAL_BASE)
        first = long_table.inner_product(0, offset)
        shifted = long_table.inner_product(2, 2 + offset)
        worst = max(abs(first - closed), abs(shifted - closed))
        print(
            f"  {offset:>2} | {first:>22.15f} | {shifted:>22.15f} | "
            f"{closed:>22.15f} | {worst:.1e}"
        )
    print()
    print("  三条读法：")
    print("    ① 同一个 δ 在不同起点上给出**同一个数**——内积只依赖 p − q")
    print("    ② 它与闭式 Σ_i cos(δ/f_i) 逐位相等（两条不同的求和路径）")
    print("    ③ 因此模型可以从'两行编码的内积'里读出相对距离，而不是死记绝对位置")
    print()
    print("  为什么？因为位移恰好是一个**分块正交旋转**：")
    for offset in (1, 3):
        factor = rotation_factor(HAND_DIMENSION, offset)
        print(f"  R_{offset} 的非零块：")
        for row in range(HAND_DIMENSION):
            cells = "、".join(
                f"{factor[row][column]:+.6f}" for column in range(HAND_DIMENSION) if factor[row][column]
            )
            print(f"    {cells}")
        rotated = tuple(
            tuple(
                math.fsum(factor[row][column] * table.row(start)[column] for column in range(HAND_DIMENSION))
                for row in range(HAND_DIMENSION)
            )
            for start in range(HAND_POSITIONS - offset)
        )
        worst = max(
            abs(a - b)
            for rotated_row, expected_row in zip(
                rotated,
                [table.row(start + offset) for start in range(HAND_POSITIONS - offset)],
                strict=True,
            )
            for a, b in zip(rotated_row, expected_row, strict=True)
        )
        print(f"    |R_{offset}·PE(p) − PE(p+{offset})| 的最大值 = {worst:.2e}")

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：注入——一行加法，而三张表都要能看见")
    shape = PositionalShape(inputs=HAND_DIMENSION, positions=2, dimension=HAND_DIMENSION, outputs=HAND_DIMENSION)
    print(f"  {shape.summary_line()}")
    inputs = ((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0))
    for name, sample in (
        ("全零表（注入=恒等）", zero_table(2, HAND_DIMENSION)),
        ("正弦表", sinusoidal_table(2, HAND_DIMENSION)),
    ):
        gathered, injected = inject(inputs, sample)
        print(f"  {name}：")
        print(f"    x        = {round_matrix(inputs)}")
        print(f"    table    = {round_matrix(gathered)}")
        print(f"    x + table= {round_matrix(injected)}")
    print()
    print("  三个数都要打印出来：'模型看到了什么'与'位置那部分是哪些数'必须同时可见，")
    print("  否则'位置编码贡献了多少'只能靠反推。")
    print()
    print("  位置/内容比（‖PE‖ / ‖x‖）是一个便宜而有用的读数：")
    params = default_parameters(VOCABULARY)
    task = make_readout_task(seed=42, vocabulary=VOCABULARY, length=LENGTH)
    zero = zero_table(LENGTH, VOCABULARY)
    sine = sinusoidal_table(LENGTH, VOCABULARY)
    print(f"    全零表：{batch_mean_injection_ratio(params, zero, ((task.inputs, task.target),)):.6f}")
    print(f"    正弦表：{batch_mean_injection_ratio(params, sine, ((task.inputs, task.target),)):.6f}")
    print("  它回答的问题是'位置占了多少'——太小则位置信息被内容淹没，太大则内容被淹没。")

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：六项梯度校验——多出来的那一项就是这一课")
    for target, formula in POSITIONAL_GRADIENT_FORMULAS.items():
        print(f"  {target:<9} {formula}")
    print()
    print(f"  {'场景':<26} | {'通过':>4} | {'最大相对误差':>12} | 逐项")
    scenarios = (
        ("正弦表 / 位置不重复", task.inputs, task.target, None),
        ("正弦表 / 位置重复 (0,1,0,1)", task.inputs, task.target, (0, 1, 0, 1)),
        ("可学习表 / 位置不重复", task.inputs, task.target, None),
    )
    for label, sample_inputs, sample_target, positions in scenarios:
        sample_table = sine if label.startswith("正弦") else learnable_table(LENGTH, VOCABULARY)
        report = check_positional_gradients(
            params, sample_table, sample_inputs,
            target=sample_target, positions=positions,
        )
        detail = "、".join(
            f"{item.target}={item.max_scaled_error:.1e}" for item in report.outcomes
        )
        passed = sum(1 for item in report.outcomes if item.passed)
        print(
            f"  {label:<26} | {passed:>4} | {report.worst_scaled_error:>12.2e} | {detail}"
        )
    print()
    print("  容差 1e-6，实测 1e-11 量级——隔着五个数量级。")
    print("  而'位置重复'那一行是**唯一**能区分两种写法的样本：")
    repeat_forward = positional_forward(
        params, sine, task.inputs, positions=(0, 1, 0, 1), target=task.target
    )
    gradients = positional_backward(repeat_forward, _loss_gradient(repeat_forward))
    print(f"    dTable 第 0 行 = {round_row(gradients.grad_table[0], 8)}")
    print(f"    dTable 第 1 行 = {round_row(gradients.grad_table[1], 8)}")
    print("    第 0 行是**两行之和**、第 1 行是**一行**——覆盖写法只会留下后者。")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：七条性质——其中三条只对'对齐频率的正弦表'成立")
    for name in POSITIONAL_PROPERTIES:
        print(f"  {name}")
    print()
    hand_report = check_properties(params, sine, task.inputs, target=task.target)
    print(f"  正弦表（对齐）：{hand_report.summary_line()}")
    for outcome in hand_report.outcomes:
        print(f"    {outcome.summary_line()}")
    learnable_report = check_properties(
        params, learnable_table(LENGTH, VOCABULARY), task.inputs, target=task.target
    )
    print(f"  可学习表：{learnable_report.summary_line()}")
    for outcome in learnable_report.failures:
        print(f"    [失败] {outcome.name} {outcome.evidence}")
    causal_report = check_properties(params, sine, task.inputs, target=task.target, causal=True)
    print(f"  因果掩码：{causal_report.summary_line()}")
    for outcome in causal_report.skipped:
        print(f"    {outcome.summary_line()}")
    print()
    print("  可学习表的'失败 3 条'是**正确的结果**：它的行是自由参数，")
    print("  既不保证范数恒定、也不保证位移律。这正是两种编码的差别所在。")

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：置换缺口——等变性是被**打破**的，而这是目的")
    for label, sample in (("全零表（= 不加位置编码）", zero), ("正弦表", sine)):
        break_report = symmetry_break(params, sample, task.inputs)
        print(f"  {label}：{break_report.summary_line()}")
        print(f"    broken = {break_report.broken}")
    print()
    print("  基准层的缺口**恰好 0.0**（不是'很小'）：无掩码时 QKᵀ 只是行列跟着换，")
    print("  每一行 softmax 的求和项顺序完全相同 → 浮点上也逐位相等。")
    print("  注入之后缺口 > 0：位置那部分是按**行号**加进去的，它没有跟着换。")
    print("  → day075 用'置换等变'提出了问题，今天在这里回答它。")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：位置选择任务与可证的下界")
    print(f"  {task.describe()}")
    print(f"  基序列 {task.tokens} 的目标：每一行都输出第 0 位那个 token 的 one-hot")
    print("  轨道（4 个循环位移，每个位移的'第 0 位'是另一个 token）：")
    for item in task.orbit():
        print(f"    {item.tokens}  →  目标 token = {item.tokens[0]}")
    print()
    print(f"  闭式下界 (n−1)/(n·V) = ({LENGTH}−1)/({LENGTH}×{VOCABULARY}) = {orbit_floor(task):.6f}")
    print(f"  袋子预测器（每一行输出'这一袋里有哪几个 token'）的实测平均损失 = "
          f"{floor_witness(task):.6f}")
    print("  → 两者相等：下界是**紧的**，常数预测器已经达到它，而它本身也是等变的。")
    print()
    print("  一句话：**没有位置信息时，'第 0 位是什么'最多只能被读成'这一袋里有什么'**——")
    print(f"  两者之间的差 {orbit_floor(task):.6f} 就是位置知识的价格。")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：四个编码刻度（同一批初始参数、同一个任务、同一个优化器）")
    comparison = compare_encodings(
        params, task, optimizer_factory=lambda: AdamOptimizer(0.05), steps=60
    )
    print(f"  {comparison.summary_line()}")
    for line in comparison.table_lines():
        print(line)
    print()
    print("  读法：")
    print("    none               停在 0.125 附近（= 下界），命中率 1/n —— 它给不出位置")
    print("    sinusoidal         跌到 0.003 量级，命中率 100% —— 位置知识是公式给的（0 参数）")
    print("    learnable_random   也跌下去，但它的表动了 0.6 量级 —— 知识是从数据里学出来的")
    print("    learnable_from_sine 起点已经'会读位置'，因此收敛更快、更低")
    print("  但注意：**这一份表是一次观测**，不是'哪种编码更好'的结论。")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：下界判决书——'停在界上'与'越过界'")
    study = symmetry_floor_study(
        params, task, optimizer_factory=lambda: AdamOptimizer(0.05), steps=60
    )
    print(f"  {study.summary_line()}")
    for line in study.table_lines():
        print(line)
    print()
    for variant in study.variants:
        print(f"  {variant.summary_line()}")
        for note in variant.notes:
            print(f"      {note}")
    print()
    print(f"  下界是否被精确达到（tight）：{study.tight}")
    print(f"  判决（ok）：{study.ok}")
    print("  等变变体**不能**低于下界；破了对称性的两个变体**越过了**它——")
    print("  而因果掩码也能越过界，说明'位置编码不是唯一出路'，这也是必要的对照：")
    print("  没有它，'位置编码有用'这句话无法与'掩码已经提供了顺序'区分开。")

    # ------------------------------------------------------------------ 第 11 节
    section("第 11 节：两种梯度源的对账（为什么优先用 SGD）")
    trajectories: dict[str, list[float]] = {}
    for label, optimizer in (("Adam 0.05", AdamOptimizer(0.05)), ("SGD 0.05", SGDOptimizer(0.05))):
        run = train_positions(
            params, sine, task, optimizer=optimizer, steps=3,
            trainable=("w_query", "w_key", "w_value", "w_output"),
        )
        trajectories[label] = list(run.trajectory)
        print(f"  {label:<10} {run.summary_line()}")
    print()
    print("  两条轨迹的差（同一批参数、同一批样本，只换优化器）：")
    gap = max(
        abs(a - b)
        for a, b in zip(trajectories["Adam 0.05"], trajectories["SGD 0.05"], strict=True)
    )
    print(f"    最大绝对差 {gap:.3e}")
    print("  Adam 按 √v̂ 归一化，会放大梯度分量的**相对**误差；")
    print("  因此要验证梯度实现，优先用 SGD（day075/076 的同一条纪律）。")

    # ------------------------------------------------------------------ 收尾
    section("收尾：这一课留下的三句话")
    print("  ① 位置编码的全部算术是**一行加法**，而值钱的部分在它两边：")
    print("     造表（两种编码的语义差别）与反向（逐位传回 + 按位置累加）。")
    print("  ② 正弦表的两条性质只对**对齐频率**成立——")
    print("     day073 的错位写法会让'行范数恒定'与'位移律'一起失效，")
    print("     而判据必须能抓住它，否则'实现对了'与'判据太松'分不开。")
    print(f"  ③ 等变模型在位置选择任务上有下界 {orbit_floor(task):.6f}，")
    print("     而这个下界是**紧**的：常数预测器恰好达到它。")
    print("     '损失越过下界'因此是位置信息真的被用上的证据，而不是一句直觉。")
    print()
    print(f"  速查：d=4 的两对频率周期 = {frequency_of(0, 4):g}/{frequency_of(1, 4):g}；"
          f"每行范数 = √(d/2) = {math.sqrt(HAND_DIMENSION / 2):.6f}")
    print(f"  POSITIONAL_BASE = {POSITIONAL_BASE:g}（与 day073 逐位相同）")


def _loss_gradient(forward):
    """转发 ``layers.loss_gradient``（演示脚本里只用一个短名字）."""
    from smart_research_agent.positional_encoding.layers import loss_gradient

    return loss_gradient(forward)


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
