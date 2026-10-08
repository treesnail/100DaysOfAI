"""离线演示：把一条链更新 40 次（day081 / M7-D6）.

跑法：

```bash
python scripts/training_optim_demo.py      # 十节，输出写到 outputs/training_optim_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。产物只写在 ``outputs/`` 下。

十节里最值得看的是第 3、5、6、7、8、9 节：

```text
3   Dropout：掩码、两个相、以及"掩码固定之后梯度**可以**被数值差分逐项核对"
5   初始化：四种方案的实测/理论标准差之比；**kaiming 的初始增益 7.76 是唯一发散的那一组**
6   调度：day079 留下的问题——热身有没有救下 post-LN？（答案是：在这条链上它没有救下任何东西）
7   裁剪：同一个学习率下关/开裁剪；**裁剪在学习率合适时只是拖慢，在学习率过大时是救命的**
8   四组实验的总表（初始化 / 调度 / 裁剪 / Dropout 各一次只改一个旋钮）
9   "梯度小 ≠ 学不好"：post-LN 的损失起点更差、却收敛到更低的平台
10  学习率边界：0.5 能收敛、0.8 溢出——而**溢出发生在发散判据之前**
```
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.encoder_decoder.types import NORM_POST, NORM_PRE  # noqa: E402
from smart_research_agent.transformer_stack import make_shape  # noqa: E402
from smart_research_agent.training_optim import (  # noqa: E402
    INIT_SCHEMES,
    SCHEME_DESCRIPTIONS,
    TRAINING_NOTES,
    TRAINING_PROPERTIES,
    DivergenceError,
    EarlyStopping,
    TrainingConfig,
    check_no_divergence,
    clip_gradients,
    describe_controls,
    direction_cosine,
    dropout_forward,
    dropout_mask,
    expected_keep,
    expected_std,
    initialization_rows,
    initialize_parameters,
    kept_fraction,
    learning_rate_at,
    measure_std,
    train,
    training_study,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "training_optim_demo.txt"

LAYERS = 4
HIDDEN = 6
FFN_RATIO = 4
TOKENS = 4
SEED = 7
STEPS = 40
LEARNING_RATE = 0.1
WARMUP_STEPS = 5


class _Tee:
    """把 stdout 同时写到文件与终端（day079/080 的演示脚本用的是同一种写法）."""

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


def _samples() -> tuple[tuple, tuple]:
    """样本输入与目标（与 tests/training_samples.py 同值，因此两处读数可直接对照）."""
    inputs = tuple(
        tuple(0.1 * (row + 1) - 0.05 * (column + 1) for column in range(HIDDEN))
        for row in range(TOKENS)
    )
    target = tuple(
        tuple(0.2 - 0.03 * (row * HIDDEN + column) for column in range(HIDDEN))
        for row in range(TOKENS)
    )
    return inputs, target


def _shape():
    """样本形状（4 层 / d=6 / d_ff=24 / n=4）."""
    return make_shape(layers=LAYERS, hidden=HIDDEN, tokens=TOKENS, ffn_ratio=FFN_RATIO)


def _base_config(**changes) -> TrainingConfig:
    """样本配置（一次只改一个旋钮时用它 ``replaced``）."""
    base = TrainingConfig(
        learning_rate=LEARNING_RATE, steps=STEPS, seed=SEED, schedule_params=(), max_norm=None
    )
    return base if not changes else base.replaced(**changes)


def section_1_controls() -> None:
    """第 1 节：六个旋钮与六条性质（教材用的是同一张表）."""
    _section("1 六个旋钮与六条性质")
    for line in describe_controls():
        print(f"  {line}")
    print()
    print(f"  六条性质：{len(TRAINING_PROPERTIES)} 条 —— {list(TRAINING_PROPERTIES)}")
    for note in TRAINING_NOTES:
        print(f"  边界：{note}")


def section_2_dropout() -> None:
    """第 2 节：一个掩码、两个相、一行反向."""
    _section("2 Dropout：一个掩码、两个相、一行反向")
    for rate, phase in ((0.2, "train"), (0.2, "eval"), (0.0, "train")):
        output, mask, scale = dropout_forward(
            ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)), rate=rate, seed=SEED, phase=phase
        )
        print(f"  rate={rate:.1f} {phase:<5} | 缩放 {scale:.4f} | 保留比例 {kept_fraction(mask):.4f}")
        print(f"                 掩码 {[[int(v) for v in row] for row in mask]}")
        print(f"                 输出 {[[round(v, 4) for v in row] for row in output]}")
    print()
    print(f"  期望保留比例 1−p（rate=0.2）= {expected_keep(0.2):.4f}（实测值在它附近摆动）")
    print("  inverted dropout 的判据：E[掩码 × 缩放] = 1 ⇒ 推理路径与'没有 dropout 的模型'逐位相同")


def section_3_dropout_is_checkable() -> None:
    """第 3 节：掩码固定之后，dropout 的梯度**可以**被数值差分逐项核对."""
    _section("3 “掩码固定 ⇒ 梯度可校验”（本课最容易被说错的一句话）")
    inputs = ((0.3, -0.7, 1.1, -0.2), (0.9, 0.4, -1.3, 0.6))
    mask = dropout_mask(2, 4, rate=0.25, seed=SEED)
    scale = 1.0 / (1.0 - 0.25)
    grad = ((1.0, -1.0, 1.0, -1.0), (0.5, 0.5, -0.5, -0.5))
    from smart_research_agent.training_optim import dropout_backward

    analytic = dropout_backward(grad, mask, scale)
    print(f"  掩码 {[[int(v) for v in row] for row in mask]}；缩放 {scale:.6f}")
    print("  解析梯度（dy ⊙ 掩码 × 缩放）：")
    for row in analytic:
        print(f"    {[round(v, 6) for v in row]}")
    print()
    print("  掩码固定之后，这个算子就是'逐元素乘以一个已知常数'——一个**线性**、对角的东西，")
    print("  因此它的梯度完全能用中心差分逐项核对（测试里就是这么做的：test_training_dropout.py）")
    print("  换句话说：'dropout 是随机的所以梯度没法校验'这句话，只对'掩码也随机'那一半成立")


def section_4_schedule_and_clip() -> None:
    """第 4 节：两个控制器的读数（调度逐位对齐 day074、裁剪只改长度）."""
    _section("4 调度与裁剪：两个控制器各自留下什么读数")
    config = _base_config(
        schedule="warmup_cosine",
        schedule_params=(("warmup_steps", float(WARMUP_STEPS)), ("total_steps", float(STEPS))),
    )
    print("  热身 + 余弦的学习率（前 8 步与最后 3 步）：")
    values = [learning_rate_at(config, step) for step in range(1, STEPS + 1)]
    print(f"    1~8 步   {['%.6f' % v for v in values[:8]]}")
    print(f"    38~40 步 {['%.6f' % v for v in values[-3:]]}")
    print(f"    峰值 {max(values):.6f}（第 {values.index(max(values)) + 1} 步）")
    print()
    from smart_research_agent.math_foundations.optim import global_norm

    grads = tuple(0.1 * (index + 1) for index in range(12))
    for max_norm in (None, 0.5, 0.2):
        clipped, report = clip_gradients(grads, _base_config(max_norm=max_norm))
        cosine = direction_cosine(grads, clipped)
        print(f"  max_norm={str(max_norm):<5} | {report.summary_line()} | 方向余弦 {cosine:.6f}")
    print(f"  未裁剪时的 ‖g‖ = {global_norm(grads):.6f}（与读数里的 original_norm 一致）")
    print("  方向余弦恒为 1.0 ⇒ 整体范数裁剪**只改长度、不改方向**（性质 4）")


def section_5_init() -> None:
    """第 5 节：四种初始化方案的实测/理论标准差与逐层增益."""
    _section("5 初始化：实测/理论标准差之比，与 day080 的逐层增益")
    shape = _shape()
    print("  方案            | fan_in/out | 理论 std   | 实测 std   | 比值   | 第一层增益")
    print("  " + "-" * 84)
    for scheme in INIT_SCHEMES:
        rows = initialization_rows(shape, scheme=scheme, seed=SEED)
        name, theory, measured, ratio = rows[0]
        from smart_research_agent.training_optim import gain_profile_of

        gains = gain_profile_of(shape, _samples()[0], scheme=scheme, seed=SEED)
        print(
            f"  {scheme:<15} | 24/6       | {theory:>10.6f} | {measured:>10.6f} | "
            f"{ratio:>6.4f} | {gains[0]:>10.6f}"
        )
    print()
    for scheme in INIT_SCHEMES:
        print(f"  {scheme:<15} {SCHEME_DESCRIPTIONS[scheme]}")
    print()
    print("  kaiming 的第一层增益最大（≈7.76），而它正好是第 8 节里**唯一发散**的那一组")


def section_6_schedule_promise() -> None:
    """第 6 节：day079 留下的那个问题——热身有没有救下 post-LN？"""
    _section("6 day079 的悬案：post-LN 需要学习率预热吗？")
    shape = _shape()
    inputs, target = _samples()
    for lr in (0.1, 0.5, 0.8, 1.2):
        print(f"  lr={lr}")
        for placement in (NORM_PRE, NORM_POST):
            for schedule, params in (
                ("constant", ()),
                (
                    "warmup_cosine",
                    (("warmup_steps", float(WARMUP_STEPS)), ("total_steps", float(STEPS))),
                ),
            ):
                config = _base_config(learning_rate=lr, schedule=schedule, schedule_params=params)
                try:
                    curve = train(shape, inputs, target, config=config, placement=placement)
                    print(
                        f"    {placement}-LN {schedule:<14} | {curve.summary_line()}"
                    )
                except DivergenceError as error:
                    print(f"    {placement}-LN {schedule:<14} | 发散：{str(error)[:64]}…")
    print()
    print("  读法：在这条 4 层小链上，**热身没有救下任何东西**；")
    print("  而 post-LN 的起点更差（0.962255 vs 0.247506）、收敛到的平台更低（0.0405 vs 0.0980）。")


def section_7_clipping() -> None:
    """第 7 节：裁剪在学习率合适时只是拖慢，在学习率过大时是救命的."""
    _section("7 裁剪：什么时候它只是拖慢、什么时候它是救命的")
    shape = _shape()
    inputs, target = _samples()
    for lr in (0.1, 0.8, 1.5):
        for max_norm in (None, 0.5):
            config = _base_config(learning_rate=lr, max_norm=max_norm)
            try:
                curve = train(shape, inputs, target, config=config)
                scales = [record.clip_scale for record in curve.records]
                print(
                    f"  lr={lr:<4} max_norm={str(max_norm):<5} | {curve.summary_line()}"
                    f" | 平均缩放 {sum(scales) / len(scales):.6f}"
                )
            except DivergenceError as error:
                print(f"  lr={lr:<4} max_norm={str(max_norm):<5} | 发散：{str(error)[:56]}…")


def section_8_study() -> None:
    """第 8 节：四组实验的总表."""
    _section("8 四组实验：初始化 / 调度 / 裁剪 / Dropout（一次只改一个旋钮）")
    shape = _shape()
    inputs, target = _samples()
    study = training_study(shape, inputs, target, config=_base_config())
    for line in study.table_lines():
        print(line)
    print()
    print(f"  {study.summary_line()}")
    print(f"  发散的行：{list(study.diverged_variants)}")
    print()
    print("  逐行附加读数：")
    for row in study.rows:
        extras = "、".join(f"{key} {value:.6f}" for key, value in row.extras)
        print(f"    {row.control:<8} {row.variant:<24} | {extras}")


def section_9_counterintuitive() -> None:
    """第 9 节：这一课唯一"反直觉"的结论."""
    _section("9 “梯度小” ≠ “学不好”（对 day079 那句话的边界）")
    shape = _shape()
    inputs, target = _samples()
    rows = []
    for placement in (NORM_PRE, NORM_POST):
        curve = train(shape, inputs, target, config=_base_config(), placement=placement)
        rows.append((placement, curve))
    for placement, curve in rows:
        print(
            f"  {placement}-LN | 首个损失 {curve.first_loss:.6f} | "
            f"最好 {curve.best_loss:.6f} | 末损失 {curve.last_loss:.6f} | "
            f"改善 {curve.improvement_ratio:.4f} 倍"
        )
    print()
    print("  day079 量的是 ‖∂loss/∂x‖（**梯度能不能传到最底层**），结论是 pre-LN 更好；")
    print("  本节量的是**损失曲线**（同样的学习率、同样的步数），结论是 post-LN 更好。")
    print("  两件事都对，而它们回答的是两个不同的问题——day079 的 12.1 节早就写下了这条边界。")


def section_10_early_stop_and_edge() -> None:
    """第 10 节：早停与学习率边界（溢出发生在发散判据之前）."""
    _section("10 早停与学习率边界")
    stopper = EarlyStopping(patience=3, min_delta=0.001)
    curve_values = (1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7)
    for step, loss in enumerate(curve_values, start=1):
        stop = stopper.update(step, loss)
        print(f"  第 {step} 步：损失 {loss:.3f} | 该不该停：{stop} | {stopper.describe()}")
    print(f"  报告：{stopper.report().summary_line()}")
    print(f"  原因：{stopper.report().reason}")
    print()
    shape = _shape()
    inputs, target = _samples()
    for lr in (0.5, 0.8, 1.0):
        config = _base_config(learning_rate=lr)
        try:
            curve = train(shape, inputs, target, config=config)
            check_no_divergence(curve)
            print(f"  lr={lr:<4} | {curve.summary_line()}")
        except DivergenceError as error:
            print(f"  lr={lr:<4} | DivergenceError：{str(error)[:88]}…")
    print()
    print("  这三行是有顺序的：0.5 收敛；0.8 时**梯度范数先溢出**；1.0 时参数本身溢出。")
    print("  两种溢出都发生在'损失涨到初始值 1000 倍'这个判据之前——因此在 train() 里")
    print("  它们被显式归入同一个族（修法都是改超参，不是改数据）。")


def main() -> None:
    """十节依次跑完."""
    print("day081 / M7-D6 —— 把一条链更新 40 次（training_optim）")
    print("全部离线：纯 Python 算术；训练的是块参数，注意力那一层是给定函数。")
    section_1_controls()
    section_2_dropout()
    section_3_dropout_is_checkable()
    section_4_schedule_and_clip()
    section_5_init()
    section_6_schedule_promise()
    section_7_clipping()
    section_8_study()
    section_9_counterintuitive()
    section_10_early_stop_and_edge()
    print()
    print(f"  （另：scheme=normal 的 ffn_w_in 实测 std 与理论值的比值 = "
          f"{measure_std(initialize_parameters(_shape(), scheme='normal', seed=SEED).blocks[0].ffn_w_in) / expected_std('normal', fan_in=HIDDEN, fan_out=FFN_RATIO * HIDDEN):.6f}")


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
