"""离线演示：微积分与优化基础（day074 / Math-D2）.

跑法：

```bash
python scripts/calculus_demo.py      # 十二节，输出写到 outputs/calculus_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。
产物只写在 ``outputs/`` 下（幂等，可随时删）。

十二节里最值得看的是第 2、5、8、11、12 节：

```text
2    步长实验：把「前向 O(h) / 中心 O(h²)」与「h 太小反而更差」**量出来**
5    计算图与反向传播：x·x → 6 是怎么从「两条边各加一份」里算出来的
8    三个优化器在两种地形上：碗形上三者几乎一样，峡谷上 SGD 明显最差
11   六项梯度对照：解析式 vs 数值差分（含 softmax 雅可比的单独对照）
12   Adam 的偏差修正：第一步恰好是 lr·sign(g)（不做修正会大 3.16 倍）
```
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.math_foundations import (  # noqa: E402
    AdamOptimizer,
    MomentumOptimizer,
    SGDOptimizer,
    Scalar,
    best_step_for_central,
    central_difference,
    chain_rule,
    check_gradients_all,
    clip_by_global_norm,
    clip_by_value,
    constant_schedule,
    cosine_schedule,
    directional_derivative,
    gradient,
    gradients_of,
    linear_approximation,
    make_schedule,
    minimize,
    step_decay_schedule,
    step_size_study,
    warmup_cosine_schedule,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "calculus_demo.txt"

#: 碗形目标（最小值在 (1, −2)，最小值 0）——所有优化器实验的公共地形。
BOWL_START = (3.0, 3.0)
#: 峡谷目标 ``x² + 20y²``（两个方向曲率差 20 倍）——用来放大优化器之间的差别。
RAVINE_START = (3.0, 0.5)


class _Tee:
    """把 stdout 同时写终端与文件（与 scripts/math_foundations_demo.py 同一手法）."""

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        """两边都写，返回写到终端的字符数."""
        self._handle.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        """两边都刷."""
        self._handle.flush()
        self._stream.flush()


def section(title: str) -> None:
    """打印一节的分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def bowl(params) -> float:
    """碗形目标 ``(x−1)² + 4(y+2)²``."""
    return (params[0] - 1.0) ** 2 + 4.0 * (params[1] + 2.0) ** 2


def ravine(params) -> float:
    """峡谷目标 ``x² + 20y²``."""
    return params[0] ** 2 + 20.0 * params[1] ** 2


def main() -> None:
    """十二节演示."""
    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：三种差分方法（同一组函数，手算真值）")
    cases = (
        ("exp", math.exp, 1.0, math.e),
        ("sin", math.sin, 1.0, math.cos(1.0)),
        ("log", math.log, 2.0, 0.5),
        ("x³", lambda value: value**3, 2.0, 12.0),
        ("1/x", lambda value: 1.0 / value, 4.0, -0.0625),
    )
    print(f"  步长 h = 1e-6（中心差分的实践最优量级是 eps^(1/3) ≈ 6.06e-6）")
    print(f"  {'函数':<6}{'点':>5}{'真导数':>14}{'前向':>20}{'后向':>20}{'中心':>20}")
    for name, function, point, exact in cases:
        forward = forward_error(function, point, exact)
        backward = backward_error(function, point, exact)
        central = central_error(function, point, exact)
        print(
            f"  {name:<6}{point:>5.1f}{exact:>14.8f}"
            f"{forward:>20.3e}{backward:>20.3e}{central:>20.3e}"
        )
    print("  后三列是**误差**（绝对值）：中心差分的误差比前向小 3 个数量级左右。")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：步长实验（前向 O(h)、中心 O(h²)，以及'触底反弹'）")
    study = step_size_study(math.exp, 1.0, math.e)
    print("  f = exp 在 x = 1 处（真导数 e = 2.718281828）")
    print(f"  {'步长':<10}{'前向误差':>14}{'后向误差':>14}{'中心误差':>14}"
          f"{'前向/h':>14}{'中心/h²':>14}")
    for record in study.records:
        print(
            f"  {record.step:<10.0e}{record.forward_error:>14.3e}"
            f"{record.backward_error:>14.3e}{record.central_error:>14.3e}"
            f"{record.forward_scaled:>14.3e}{record.central_scaled:>14.3e}"
        )
    print()
    print(f"  {study.summary_line()}")
    print(f"  理论最优步长（按曲率算）：{best_step_for_central(math.exp, 1.0):.3e}")
    print("  读数三条：")
    print("    ① 最后两列在截断误差主导区是**常数**——这证明误差阶数确实是 1 与 2")
    print("    ② 误差在 h ≈ 1e-6 附近触底（中心差分），之后**上升**")
    print("    ③ 上升的原因是舍入：f(x±h) 两个几乎相等的数相减丢了有效位")

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：梯度与方向导数（'最陡方向'是可验证的）")
    point = (0.7, -0.4)

    def surface(params) -> float:
        """一个各向异性的二次曲面（手算偏导数可得）。"""
        return 3.0 * params[0] ** 2 + params[1] ** 2 + params[0] * params[1]

    grads = gradient(surface, point)
    norm = math.sqrt(math.fsum(value * value for value in grads))
    print(f"  f(x, y) = 3x² + y² + xy 在 {point} 处")
    print(f"    手算偏导：∂f/∂x = 6x + y = {6.0 * point[0] + point[1]:+.6f}，"
          f"∂f/∂y = 2y + x = {2.0 * point[1] + point[0]:+.6f}")
    print(f"    ∇f = ({grads[0]:+.6f}, {grads[1]:+.6f})   ‖∇f‖ = {norm:.6f}")
    directions = (
        ("(1, 0)", (1.0, 0.0), "= ∂f/∂x"),
        ("(0, 1)", (0.0, 1.0), "= ∂f/∂y"),
        ("(1, 1)", (1.0, 1.0), "两个单位的合成"),
        ("∇f", grads, "= ‖∇f‖（柯西–施瓦茨取等）"),
    )
    for label, direction, note in directions:
        value = directional_derivative(surface, point, direction)
        print(f"    沿 {label:<6} 方向导数 = {value:+.6f}   {note}")
    grid = max(
        directional_derivative(
            surface,
            point,
            (math.cos(index * math.pi / 18.0), math.sin(index * math.pi / 18.0)),
        )
        for index in range(36)
    )
    print(f"  扫 36 个固定方向（步长 π/18）取最大值：{grid:+.6f}")
    print(f"    它 ≤ ‖∇f‖ = {norm:.6f}（网格是离散的，最大值点不一定正好落在网格上）")
    print(f"    网格最大值与 ‖∇f‖ 的距离 {norm - grid:.2e} 就是'网格分辨率'的代价。")
    print("  结论：沿梯度方向走，函数值变化最快——这就是'往哪走'的答案。")

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：链式法则的两条路（整体差分 vs 局部导数之积）")
    layers = [math.sin, math.exp, lambda value: 2.0 * value]
    overall, product = chain_rule(layers, 0.0)
    print("  d/dx sin(exp(2x)) 在 x = 0 处")
    print(f"    整体（对复合函数做差分）      {overall:.12f}")
    print(f"    局部导数之积（从外往内乘）    {product:.12f}")
    print(f"    手算   2·cos(1)·e⁰ = 2cos1 =  {2.0 * math.cos(1.0):.12f}")
    print("  '每条局部导数必须在**正确的点**上取值'——反向传播做的就是这件事。")

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：计算图与反向传播（x·x 的梯度是 2x，不是 x）")
    x = Scalar(3.0, label="x")
    node = x * x
    node.backward()
    print(f"  y = x·x，x = 3")
    print(f"  {'节点':<18}{'运算':<10}{'值':>10}{'梯度':>12}")
    for row in node.trace():
        print(f"  {row.label:<18}{row.operation:<10}{row.value:>+10.4f}{row.grad:>+12.4f}")
    print(f"  x 的梯度是 {x.grad:.1f} = 2×3 —— **两条边各贡献一份 3**。")
    print("  把累积写成 `=` 不会报错，只会给出一个偏小的梯度（看起来像 lr 设小了）。")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：梯度累积的三种形状")
    shapes = (
        ("x·x", lambda inputs: inputs[0] * inputs[0], 3.0, "2x = 6"),
        ("x + x", lambda inputs: inputs[0] + inputs[0], 3.0, "2"),
        ("exp(x) + log(x)", lambda inputs: inputs[0].exp() + inputs[0].log(), 2.0, "e² + 1/2"),
        ("x³（= x·x·x）", lambda inputs: inputs[0] * inputs[0] * inputs[0], 2.0, "3x² = 12"),
    )
    for expression, builder, value, hand in shapes:
        node = builder([Scalar(value, label="x")])
        node.backward()
        table = gradients_of(node)
        print(f"  {expression:<16} @x={value}  梯度 = {table['x']:+.6f}   手算 {hand}")

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：定义域拒绝（不是安静地给 inf/nan）")
    attempts = (
        ("log(0)", lambda: Scalar(0.0).log()),
        ("log(−1)", lambda: Scalar(-1.0).log()),
        ("sqrt(0)", lambda: Scalar(0.0).sqrt()),
        ("1/0", lambda: Scalar(1.0) / 0.0),
        ("0^0.5", lambda: Scalar(0.0) ** 0.5),
        ("(−2)^0.5", lambda: Scalar(-2.0) ** 0.5),
    )
    for label, action in attempts:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - 演示里只看"拒绝得清不清楚"
            print(f"  {label:<10} → {type(exc).__name__}: {str(exc).splitlines()[0][:58]}…")
    print("  每一次拒绝都给出'下一步该走哪'：这是本包所有错误消息的写法。")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：三个优化器，两种地形")
    print("  碗形（曲率相同）：")
    for label, optimizer in (
        ("SGD      lr=0.05", SGDOptimizer(0.05)),
        ("动量      lr=0.05", MomentumOptimizer(0.05, momentum=0.9)),
        ("Adam     lr=0.20", AdamOptimizer(0.20)),
    ):
        trace = minimize(bowl, BOWL_START, optimizer=optimizer, steps=100)
        print(f"    {label}  →  {trace.summary_line()}")
    print("  峡谷 x² + 20y²（曲率差 20 倍，60 步）：")
    for label, optimizer in (
        ("SGD      lr=0.01", SGDOptimizer(0.01)),
        ("动量      lr=0.01", MomentumOptimizer(0.01, momentum=0.9)),
        ("Adam     lr=0.10", AdamOptimizer(0.10)),
    ):
        trace = minimize(ravine, RAVINE_START, optimizer=optimizer, steps=60)
        print(f"    {label}  →  {trace.summary_line()}")
    print("  读数：碗形上三者差别很小（曲率相同，SGD 甚至最快——它没有额外状态）；")
    print("  峡谷上 SGD 明显最差（在陡方向反复横跳），动量与 Adam 都把它压住了。")
    print("  这就是三种优化器各自的适用场景，而不是'某个一定更好'。")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：四种学习率调度（每个数都手算可复核）")
    schedules = (
        ("constant      ", constant_schedule, {"base_lr": 0.1}),
        ("step_decay    ", step_decay_schedule, {"base_lr": 0.1, "drop_every": 3, "gamma": 0.5}),
        ("cosine        ", cosine_schedule, {"base_lr": 0.1, "total_steps": 10}),
        (
            "warmup_cosine ",
            warmup_cosine_schedule,
            {"base_lr": 0.1, "warmup_steps": 4, "total_steps": 10},
        ),
    )
    columns = (1, 2, 3, 4, 5, 10)
    header = "  " + "调度".ljust(16) + "".join(f"t={step}".ljust(9) for step in columns)
    print(header)
    for label, function, params in schedules:
        values = "".join(
            f"{function(step, **params):<10.6f}" for step in columns
        )
        print(f"  {label}{values}")
    print("  常数是**基线**：没有它，'调度有没有用'无法回答。")
    print("  热身 + 退火是唯一同时解决'前期别乱跑'与'后期别抖'的形状（先升后降）。")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：梯度裁剪（两个原语，行为不同）")
    grads = (100.0, 1.0)
    clipped, scale = clip_by_global_norm(grads, 10.0)
    print(f"  原始梯度 {grads}   ‖g‖ = {math.sqrt(math.fsum(v * v for v in grads)):.4f}")
    print(f"  整体范数裁剪到 10：{tuple(round(value, 6) for value in clipped)}  缩放系数 {scale:.6f}")
    print("    方向不变（整体缩放）——这是'这一步走多长'的约束。")
    print(f"  逐分量裁剪到 ±1：{clip_by_value(grads, 1.0)}")
    print("    方向**变了**：从几乎水平变成 45°——它只适合'个别分量爆炸'的场景。")
    print(f"  范数未超限时原样返回（缩放系数 {clip_by_global_norm((3.0, 4.0), 10.0)[1]:.1f}），")
    print("  因此'裁剪有没有生效'必须靠返回的缩放系数来读。")

    # ------------------------------------------------------------------ 第 11 节
    section("第 11 节：六项梯度对照（解析式 vs 数值差分）")
    report = check_gradients_all()
    print(f"  {report.summary_line()}")
    for outcome in report.outcomes:
        print(f"    {outcome.summary_line()}")
        print(f"        解析式：{outcome.formula}")
        print(f"        对照：  {outcome.source}")
    print()
    print("  最值得看的是 softmax：它的雅可比在交叉熵的梯度里**会被抵消**")
    print("  （∂(−log p_y)/∂z 化简成 p − onehot(y)），因此只有单独比一次才拦得住它写错。")

    # ------------------------------------------------------------------ 第 12 节
    section("第 12 节：Adam 的偏差修正（第一步恰好是 lr·sign(g)）")
    for gradient_value in (1.0, -1.0, 0.01):
        fresh = AdamOptimizer(0.1)
        updated = fresh.step((0.0,), (gradient_value,))
        print(
            f"  梯度 g = {gradient_value:>+7}  →  第一步位移 {updated[0]:+.9f}"
            f"   lr·sign(g) = {-0.1 * math.copysign(1.0, gradient_value):+.9f}"
        )
    print("  不做偏差修正时第一步是 lr·3.16（m 只有真值的 1/10、v 只有 1/1000），")
    print("  而它看起来只是'前期有点抖'——因此这一条必须被算出来、断言下来。")


def forward_error(function, point: float, exact: float) -> float:
    """前向差分的绝对误差（演示用的三个小包装）."""
    from smart_research_agent.math_foundations import forward_difference

    return abs(forward_difference(function, point) - exact)


def backward_error(function, point: float, exact: float) -> float:
    """后向差分的绝对误差."""
    from smart_research_agent.math_foundations import backward_difference

    return abs(backward_difference(function, point) - exact)


def central_error(function, point: float, exact: float) -> float:
    """中心差分的绝对误差."""
    return abs(central_difference(function, point) - exact)


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
