"""day094 演示脚本：循环神经网络 —— 时间轴上的权重共享与 BPTT（十一节）.

全部离线、全部确定性：不需要 API Key，不依赖 torch / numpy。
跑法::

    cd day094/源码/smart-research-agent
    python scripts/sequence_models_demo.py

十一节对应教程的十一章；打印的读数与 ``tests/test_sequence_models.py`` 断言的是同一批。
产出 ``outputs/sequence_models_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.sequence_models import study, types, verify  # noqa: E402
from smart_research_agent.sequence_models.gradients import (  # noqa: E402
    rnn_bptt,
    rnn_forward_with_cache,
)
from smart_research_agent.sequence_models.layers import (  # noqa: E402
    LSTMSpec,
    RNNSpec,
    initialize_lstm,
    initialize_rnn,
    lstm_cell_forward,
    rnn_cell_forward,
)
from smart_research_agent.sequence_models.network import (  # noqa: E402
    build_sequence_net,
    flatten_params,
    sequence_forward,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "sequence_models_demo.txt"

SEP = "=" * 72


def section(index: int, title: str) -> None:
    """打印一节的小标题."""
    print()
    print(SEP)
    print(f"[{index}] {title}")
    print(SEP)


def main() -> None:
    section(1, "两种循环单元的画像（状态形状 / 参数量）")
    for row in study.cell_rows():
        print("  " + row.line())
    print(f"  递推：{types.RNN_RECURRENCE_FORMULA}")

    section(2, "时间轴上的权重共享到底省了多少")
    for row in study.parameter_rows():
        print("  " + row.line())
    print("  T 从 1 涨到 8，参数量一个也不多——因为循环体只有一个，它被调用了 T 次")

    section(3, "一步前向：实现与手写递推逐位一致")
    outcome = verify.check_rnn_cell_matches_manual()
    print("  " + outcome.line())
    cell = initialize_rnn(RNNSpec(2, 3, seed=study.SAMPLE_SPEC_RNN.seed))
    first = rnn_cell_forward(cell, study.SAMPLE_SEQUENCE[0], cell.initial_state())
    print(f"  h₁ = ({first[0]:+.6f}, {first[1]:+.6f}, {first[2]:+.6f})")

    section(4, "LSTM 的四个门：一次前向里的实际读数")
    for row in study.gate_rows():
        print("  " + row.line())
    print(f"  遗忘门能到 {study.gate_rows()[1].maximum:.6f}：因为它的偏置初始化成 "
          f"{types.DEFAULT_FORGET_BIAS}（σ({types.DEFAULT_FORGET_BIAS:.0f}) ≈ "
          f"{1.0 / (1.0 + math.exp(-types.DEFAULT_FORGET_BIAS)):.6f}）")

    section(5, "记忆通道：把三个门钉死之后 c 原样冻结")
    frozen = verify.check_lstm_gates_control_state()
    print("  " + frozen.line())
    pinned = verify.saturated_lstm(2, 2)
    pinned_state = ((0.0, 0.0), (0.75, -0.4))
    new_hidden, new_cell = lstm_cell_forward(pinned, study.SAMPLE_SEQUENCE[0], pinned_state)
    print(f"  c₁ = ({new_cell[0]:+.6f}, {new_cell[1]:+.6f}) 与 c₀ = (+0.750000, −0.400000) 逐位相同")
    print(f"  h₁ = ({new_hidden[0]:+.6f}, {new_hidden[1]:+.6f}) = tanh(c₁)（因为输出门恰好是 1）")

    section(6, "逐步的隐状态范数（信息在这条链上还剩多少）")
    for row in study.state_rows():
        print("  " + row.line())

    section(7, "BPTT：dW 是**累加**出来的")
    bptt_cell = initialize_rnn(RNNSpec(2, 2, seed=3))
    inputs7 = ((1.0, 0.5), (0.5, -1.0), (-1.0, 0.0))
    _states7, caches7 = rnn_forward_with_cache(bptt_cell, inputs7)
    full = rnn_bptt(bptt_cell, caches7, ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))
    last_only = rnn_bptt(bptt_cell, caches7, ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    print(f"  三步都回传：‖dW‖ = {full.norm():.6f}")
    print(f"  只留最后一步：‖dW‖ = {last_only.norm():.6f}")
    print(f"  写成 `=` 会让梯度只剩 {last_only.norm() / full.norm():.1%} 的量——而它不会报错")

    section(8, "反向：解析 BPTT 与数值差分对账")
    for row in study.bptt_rows():
        print("  " + row.line())
    print(f"  纪律：{types.BPTT_FORMULA}")
    print(f"  容差：{verify.GRAD_TOLERANCE:.1e}（与 day093 同一把尺子、同一个量级）")

    section(9, "七条性质（是否通过 / 读数 / 两个来源）")
    for row in study.property_rows():
        print("  " + row.line())
    report = verify.check_all()
    print(f"  合计：{report.passed}/{report.total} 条通过（第 ⑦ 条是**下界**判据）")

    section(10, "一个序列分类器的前向（参数怎么数）")
    params = build_sequence_net(types.CELL_LSTM, input_size=2, hidden_size=3, classes=2, seed=100)
    logits = sequence_forward(params, study.SAMPLE_SEQUENCE)
    print(f"  结构：{params.line()}")
    print(f"  参数量 {params.parameter_count}（循环 {params.cell.parameter_count} + 头 "
          f"{params.parameter_count - params.cell.parameter_count}）")
    print(f"  压平后长度：{len(flatten_params(params))}")
    print(f"  一次前向的 logits：({logits[0]:+.6f}, {logits[1]:+.6f})")

    section(11, "训练表：±1 序列的「和为正 vs 和为负」")
    for line in study.train_rows():
        print("  " + line if not line.startswith("  ") else line)
    print("  " + verify.check_lstm_outlasts_rnn_long_range().line())
    print("  边界：" + "；".join(types.RECURRENT_BOUNDARIES[:2]))


if __name__ == "__main__":
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    rendered = buffer.getvalue()
    print(rendered)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT_FILE}）")
