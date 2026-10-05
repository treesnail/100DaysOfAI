"""day093 演示脚本：卷积神经网络 —— 权重共享与空间反向（十一节）.

全部离线、全部确定性：不需要 API Key，不依赖 torch / numpy。
跑法::

    cd day093/源码/smart-research-agent
    python scripts/conv_net_demo.py

十一节对应教程的十一章；打印的读数与 ``tests/test_conv_net.py`` 断言的是同一批。
产出 ``outputs/conv_net_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.conv_net import study, types, verify  # noqa: E402
from smart_research_agent.conv_net.gradients import conv2d_backward  # noqa: E402
from smart_research_agent.conv_net.layers import (  # noqa: E402
    ConvSpec,
    conv_layer_forward,
    initialize_conv,
)
from smart_research_agent.conv_net.network import (  # noqa: E402
    build_cnn,
    cnn_forward,
    flatten_params,
)
from smart_research_agent.conv_net.ops import conv2d, output_size, pad2d, receptive_field  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "conv_net_demo.txt"

SEP = "=" * 72


def section(index: int, title: str) -> None:
    """打印一节的小标题."""
    print()
    print(SEP)
    print(f"[{index}] {title}")
    print(SEP)


def main() -> None:
    section(1, "一层卷积的画像（通道 / 核 / 参数 / 输出）")
    for row in study.layer_rows():
        print("  " + row.line())
    print(f"  参数量公式：{types.CONV_PARAM_FORMULA}")

    section(2, "权重共享到底省了多少参数")
    dense = 36 * 18 + 18
    conv = ConvSpec(1, 2, 3).parameter_count
    print(f"  6×6 图直连 18 维全连接头：36×18 + 18 = {dense} 个权重")
    print(f"  一层 conv(1→2, k=3)：2×9 + 2 = {conv} 个权重")
    print(f"  少了 {dense / conv:.1f} 倍——因为同一组核被用到了全图的每一个位置")

    section(3, "尺寸公式：先算它，再谈别的")
    for row in study.size_rows():
        print("  " + row.line())
    print(f"  公式：{types.OUTPUT_SIZE_FORMULA}")

    section(4, "same 填充与补零")
    padded = pad2d(study.SAMPLE_IMAGE, 1)
    print(f"  6×6 图补 1 圈 → {len(padded)}×{len(padded[0])}")
    kernel = ((1.0, 0.0, -1.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 1.0))
    produced = conv2d(verify.SLIDING_IMAGE, kernel, padding="same")
    print(f"  4×4 图 + 3×3 核 + same → {len(produced)}×{len(produced[0])}")
    print("  约定：本包的 conv2d 是**互相关**（核不翻转）——与 torch 一致")

    section(5, "一次卷积之后每张特征图的画像")
    for row in study.feature_rows():
        print("  " + row.line())
    print("  通道 0 全零：那一张核恰好没被这根竖线激活（ReLU 把它压平了）")

    section(6, "池化：降采样率与保留了什么")
    for row in study.pool_rows():
        print("  " + row.line())
    print(f"  6×6 按窗口 2 下采样成 3×3（{types.POOL_FORMULAS[types.POOL_MAX]}）")

    section(7, "感受野：为什么两层 3×3 等于一层 5×5")
    for row in study.field_rows():
        print("  " + row.line())
    print(f"  递推公式：{types.RECEPTIVE_FIELD_FORMULA}")
    print(f"  两层 3×3：参数 2×9 = 18；一层 5×5：参数 25 —— 感受野相同，参数更少")

    section(8, "反向：dK / dB / dX 三块梯度")
    d_kernel, d_image = conv2d_backward(
        verify.SLIDING_IMAGE, verify.SLIDING_KERNEL, verify.GRAD_OUTPUT
    )
    print(f"  d_kernel 形状 {len(d_kernel)}×{len(d_kernel[0])}（与核同形）")
    print(f"  d_image  形状 {len(d_image)}×{len(d_image[0])}（与原图同形）")
    print("  这两块互为'转置'——索引写反了形状仍然正确，只有数值差分能发现")

    section(9, "七条性质（是否通过 / 读数 / 两个来源）")
    for row in study.property_rows():
        print("  " + row.line())
    report = verify.check_all()
    print(f"  合计：{report.passed}/{report.total} 条通过")

    section(10, "一个小 CNN 的前向（参数怎么数）")
    spec = study.TRAIN_SPEC
    params = build_cnn(spec, (6, 6), seed=100)
    logits = cnn_forward(params, spec, (study.SAMPLE_IMAGE,))
    print(f"  结构：{spec.line()} → maxpool(2) → flatten(18) → dense(18→2)")
    print(f"  参数量：{params.parameter_count}（卷积 {params.conv.parameter_count} + 头 {params.parameter_count - params.conv.parameter_count}）")
    print(f"  压平后长度：{len(flatten_params(params))}")
    print(f"  一次前向的 logits：({logits[0]:+.6f}, {logits[1]:+.6f})")

    section(11, "训练表：竖线 vs 横线")
    for line in study.train_rows():
        print("  " + line if not line.startswith("  ") else line)
    print(f"  感受野（四层示例）：{receptive_field(verify.RECEPTIVE_LAYERS)}")
    print("  边界：" + "；".join(types.CONV_BOUNDARIES[:2]))


if __name__ == "__main__":
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    rendered = buffer.getvalue()
    print(rendered)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT_FILE}）")
