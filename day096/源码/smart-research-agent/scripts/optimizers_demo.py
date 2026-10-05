"""day092 演示脚本：优化器 —— 往哪走、走多远（十一节）.

全部离线、全部确定性：不需要 API Key，不依赖 torch / numpy。
跑法::

    cd day092/源码/smart-research-agent
    python scripts/optimizers_demo.py

十一节对应教程的十一章；打印的读数与 ``tests/test_optimizers.py`` 断言的是同一批。
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.optimizers import advisor, compare, study, types, verify  # noqa: E402
from smart_research_agent.optimizers.optimizer import TrainingOptimizer  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "optimizers_demo.txt"

SEP = "=" * 72


def section(index: int, title: str) -> None:
    """打印一节的小标题."""
    print()
    print(SEP)
    print(f"[{index}] {title}")
    print(SEP)


def main() -> None:
    section(1, "六个更新规则（公式 / 状态 / 归属）")
    for row in study.rule_rows():
        print("  " + row.line())

    section(2, "一步表：同一组写死输入各走一步")
    for row in study.one_step_rows():
        print("  " + row.line())

    section(3, "Nesterov 与普通动量的第一步为什么不同")
    params, grads = study.STEP_SAMPLE_PARAMS, study.STEP_SAMPLE_GRADS
    for name in ("momentum", "nesterov"):
        opt = TrainingOptimizer(name, study.STEP_SAMPLE_LR)
        updated = opt.step(params, grads)
        print(f"  {name:<9} θ₀ {params[0]:.6f} → {updated[0]:.6f}")

    section(4, "三种衰减：同一梯度、同一学习率，一步之后差多少")
    for mode in types.DECAY_MODES:
        opt = TrainingOptimizer("sgd", 0.1, weight_decay=0.5, decay_mode=mode)
        updated = opt.step((1.0, 2.0), (1.0, 0.0))
        print(f"  decay={mode:<10} θ=(1,2) → ({updated[0]:.6f}, {updated[1]:.6f})")
    print("  耦合（l2）把 λθ 加进梯度，解耦（decoupled）在更新之后再缩参数——两者不同。")

    section(5, "AdamW 与 Adam+L2：为什么解耦不是摆设")
    outcome = verify.check_decoupled_decay_differs_under_varying_gradients()
    print("  " + outcome.line())
    print("  解耦把 λθ 挡在梯度之外，因此它不会被 √v̂ 除成随尺度变化的强度。")

    section(6, "梯度裁剪：整体范数只改长度、不改方向")
    print("  " + study.clip_row().line())

    section(7, "学习率调度：热身 + 余弦（先升后降）")
    for row in study.schedule_rows():
        print("  " + row.line())
    print("  末步 lr = 0 是合法的（这一步不再更新）——训练回路用非负校验而不是正数校验。")

    section(8, "收敛对比：良态碗")
    for row in study.compare_rows("quadratic_bowl"):
        print("  " + row.line())

    section(9, "收敛对比：病态峡谷与 Rosenbrock")
    for objective_name in ("anisotropic_valley", "rosenbrock"):
        print(f"  -- {objective_name} --")
        for row in study.compare_rows(objective_name):
            print("    " + row.line())

    section(10, "七条性质（是否通过 / 读数 / 跨包对象）")
    for row in study.property_rows():
        print("  " + row.line())
    report = verify.check_all()
    print(f"  合计：{report.passed}/{report.total} 条通过")

    section(11, "六类场景的选型建议")
    for line in advisor.recommendation_lines():
        print(line)


if __name__ == "__main__":
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    rendered = buffer.getvalue()
    print(rendered)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT_FILE}）")
