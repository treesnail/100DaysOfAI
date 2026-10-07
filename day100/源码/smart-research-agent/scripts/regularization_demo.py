"""day095 演示脚本：训练技巧与正则化 —— 四个旋钮与训练日志（十一节）.

全部离线、全部确定性：不需要 API Key，不依赖 torch / numpy。
跑法::

    cd day095/源码/smart-research-agent
    python scripts/regularization_demo.py

十一节对应教程的十一章；打印的读数与 ``tests/test_regularization.py`` 断言的是同一批。
产出 ``outputs/regularization_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.regularization import study, types, verify  # noqa: E402
from smart_research_agent.regularization import visualize as vz  # noqa: E402
from smart_research_agent.regularization.normalization import (  # noqa: E402
    batch_norm_backward,
    batch_norm_forward,
    batch_size_one_report,
    batch_statistics,
)
from smart_research_agent.regularization.network import (  # noqa: E402
    build_regularized,
    flatten_params,
    loss_and_grad,
)
from smart_research_agent.regularization.techniques import (  # noqa: E402
    dropout_vector,
    early_stopping_watch,
    learning_rate_at,
    RegularizationConfig,
)
from smart_research_agent.sequence_models.train import make_sign_dataset  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "regularization_demo.txt"

SEP = "=" * 72


def section(index: int, title: str) -> None:
    """打印一节的小标题."""
    print()
    print(SEP)
    print(f"[{index}] {title}")
    print(SEP)


def main() -> None:
    batch = verify.SAMPLE_BATCH

    section(1, "两条归一化轴：BatchNorm 沿批、LayerNorm 沿特征")
    for row in study.axis_rows():
        print("  " + row.line())

    section(2, "BatchNorm 的前向：统计量 → 标准化 → γ/β")
    stats = batch_statistics(batch)
    output, cache, running = batch_norm_forward(
        batch, gamma=verify.SAMPLE_GAMMA, beta=verify.SAMPLE_BETA
    )
    print(f"  这一批 {len(batch)}×{len(batch[0])}：{stats.line()}")
    print(f"  输出第一行：({output[0][0]:+.6f}, {output[0][1]:+.6f}, {output[0][2]:+.6f})")
    print(f"  {running.line()}")
    print(f"  公式：{types.BN_FORMULA}")

    section(3, "批大小为 1：BN 的签名（方差恒为 0 ⇒ 输出被压成 β）")
    single = batch_size_one_report(((0.4, -1.2, 2.0),))
    print(f"  批大小 {single['rows']:.0f}：最大方差 {single['max_variance']:.6f}，"
          f"最大 |输出| {single['max_abs_output']:.6f}")
    print("  这条读数是'BN 必须分批'最直接的证据——本课的网络因此是**批版本**")

    section(4, "两个相在同一个批上的差别（μ/σ 的来源不同）")
    for row in study.phase_rows():
        print("  " + row.line())
    print(f"  训练相：{types.PHASE_DESCRIPTIONS[types.PHASE_TRAIN]}")
    print(f"  推理相：{types.PHASE_DESCRIPTIONS[types.PHASE_EVAL]}")

    section(5, "反向：训练相三项 vs 推理相一项")
    probe = ((1.0, -1.0), (3.0, 1.0))
    _out, probe_cache, _run = batch_norm_forward(probe, gamma=(2.0, 2.0))
    train_grads = batch_norm_backward(probe_cache, ((1.0, 1.0), (1.0, 1.0)))
    print(f"  训练相（dy 全 1、γ=2）：dx = {train_grads.d_inputs[0][0]:+.12f}"
          f"、dβ = {train_grads.d_beta[0]:.6f}（三项恰好抵消）")
    print(f"  公式（训练相）：{types.BN_BACKWARD_FORMULA}")
    print(f"  公式（推理相）：{types.BN_EVAL_BACKWARD_FORMULA}")

    section(6, "四个旋钮与它们的实现者")
    for row in study.technique_rows():
        print("  " + row.line())

    section(7, "转发层的三行读数（dropout / 调度 / 早停）")
    dropped, mask, scale = dropout_vector((1.0, 2.0, 3.0, 4.0), rate=0.5, seed=9)
    print(f"  dropout(rate=0.5, seed=9)：掩码 {mask[0]}，缩放 {scale:g}")
    print(f"                输出 ({dropped[0]:g}, {dropped[1]:g}, {dropped[2]:g}, {dropped[3]:g})")
    for step in (1, 5, 10):
        value = learning_rate_at(step, base_lr=0.1, schedule="cosine", total_steps=10, min_lr=0.0)
        print(f"  cosine 第 {step:>2} 步的学习率：{value:.6f}")
    report = early_stopping_watch((1.0, 0.9, 0.9, 0.9, 0.9), patience=3)
    print(f"  早停：最好第 {report.best_step} 步（{report.best_loss:.6f}），"
          f"已等 {report.waited} 步，触发={report.triggered}")

    section(8, "训练日志的四张脸")
    curve = (0.92, 0.81, 0.62, 0.44, 0.35, 0.31, 0.22, 0.17, 0.13, 0.11, 0.09, 0.08)
    print(f"  sparkline : {vz.sparkline(curve, width=len(curve))}")
    print("  ASCII 折线:")
    for line in vz.loss_curve(curve, height=4, width=len(curve)):
        print("    " + line)
    print("  条形图（最终准确率）:")
    for line in vz.bar_chart((("全开", 0.6875), ("无归一化", 1.0), ("无丢弃", 0.5))):
        print("    " + line)
    print("  " + vz.curve_summary(curve))

    section(9, "七条性质（是否通过 / 读数 / 两个来源）")
    for row in study.property_rows():
        print("  " + row.line())
    outcome = verify.check_all()
    print(f"  合计：{outcome.passed}/{outcome.total} 条通过（第 ③ 条是**下界**判据）")

    section(10, "装上旋钮之后的链（参数量与一次批前向）")
    dataset = make_sign_dataset(3)
    batch10 = dataset[:6]
    params = build_regularized("lstm", input_size=1, hidden_size=4, classes=2, seed=100)
    loss, grads, cache = loss_and_grad(params, batch10, dropout_rate=0.2, dropout_seed=5)
    print(f"  结构：{params.line()}")
    print(f"  参数量 {params.parameter_count}（day094 的网络 + γ/β 共 {2 * params.features} 个）")
    print(f"  压平后长度：{len(flatten_params(params))}")
    print(f"  这一批的损失：{loss:.6f}；梯度整体范数：{grads.total_norm():.6f}")

    section(11, "消融表：四个变体 × 两种单元")
    for row in study.ablation_rows():
        print("  " + row.line())
    print(f"  配置示例：{RegularizationConfig().line()}")
    print(f"  边界：{types.REGULARIZATION_BOUNDARIES[3]}")
    print("  注意：这一步真的训练了八次，读数来自现场，不是缓存")


if __name__ == "__main__":
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    rendered = buffer.getvalue()
    print(rendered)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT_FILE}）")
