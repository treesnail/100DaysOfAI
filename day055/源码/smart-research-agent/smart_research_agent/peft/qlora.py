"""QLoRA：把冻结的基座压到 4-bit，适配器仍然是 16-bit（M5-D3）.

QLoRA 常被描述成"4-bit 训练"，这句话不准确，本模块的存在就是为了把它说清楚：

    **QLoRA 训练的是 fp16/bf16 的适配器 ``A`` / ``B``；4-bit 只用于存储冻结的基座。**

所以"QLoRA 掉点"从来不是"用 4-bit 训练"造成的，而是**基座权重的量化误差**
造成的。误差有多大、能不能接受，是一个可以被度量的问题——本模块把度量
做出来（``QuantizationReport``），而不是留在"据说几乎无损"这个说法上。

本模块实现的四件事，都是可以在没有 GPU、没有 bitsandbytes 的环境里逐位
验证的：

1. **码本**（``nf4_levels`` / ``fp4_levels`` / ``int4_levels``）。NF4 的定义
   很直白：**标准正态分布的 16 个等概率分位数，归一化到 ``[-1, 1]``**。
   之所以用正态分位数，是因为预训练权重近似零均值正态分布——码点放在
   概率密度高的地方，量化误差自然更小。本课按定义生成码本，所以
   ``nf4_levels()[0] == -1.0``、``nf4_levels()[15] == 1.0`` 这两条断言成立；
2. **分块量化**（``quantize_blockwise``）：每 ``block_size`` 个权重共享一个
   ``absmax`` 常数，先归一化再取最近码点。块越小精度越高、常数开销越大，
   论文取 **64** 是收益与开销的折中点；
3. **常数量化**（double quantization）：把一级的 fp32 常数再量化一次
   （8-bit 常数 + 每 256 个常数一个 fp32 缩放）。每参数存储因此从
   ``4 + 32/64 = 4.5 bit`` 降到 ``4 + 8/64 + 32/(64×256) = 4.126953 bit``；
4. **真实的存储字节数**（而不是解析式）。解析式给的是下界：实际还要
   ``ceil`` 取整（最后一个不满的块、最后一个不满的常数分组）。两者都报出来，
   因为它们回答的是两个不同的问题——"理论上限"与"磁盘上占多少"。

一处必须说明的口径差异：bitsandbytes 的 4-bit 常数用的是 **FP8（E4M3）**，
本课用 **8-bit 对称线性量化**模拟。位宽相同，因此**存储字节数完全一致**，
只有量化误差的分布略有差异。之所以不实现 E4M3 的位级舍入：它的收益只是
让"常数误差"这一项的第三位小数更漂亮，而代价是把读者的注意力从
"每参数到底占多少字节"上引开。
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from smart_research_agent.peft.config import (
    CODE_BITS,
    FULL_PRECISION_CONSTANT_BITS,
    PEFTConfigError,
    QLoRAConfig,
)
from smart_research_agent.sft.reference_model import ModelState, ReferenceSFTModel

#: NF4 的码点数（4 bit → 16 个）
NF4_CODE_COUNT = 16

#: 4-bit 常数量化用的位宽（bitsandbytes 用 FP8 E4M3；本课用 8-bit 对称线性）
SCALE_BITS = 8

#: 模拟常数量化时使用的 8-bit 量化器名称（写进报告，避免"到底用的什么"含糊）
SCALE_CODEC = "int8-symmetric"


def nf4_levels() -> tuple[float, ...]:
    """NF4（4-bit NormalFloat）的 16 个码点：正态分位数归一化到 ``[-1, 1]``.

    构造方式就是定义本身：

    .. code-block:: text

        q_i = Φ⁻¹((i + 0.5) / 16)      i = 0..15        （等概率分位数）
        level_i = q_i / max|q|                           （归一化）

    三条件值得记住的结论（都在测试里被钉死）：

    1. ``levels[0] == -1.0``、``levels[-1] == 1.0``，且严格递增；
    2. **16 个码点里没有 0**：偶数个码点时，最接近 0 的两个码点是
       ``Φ⁻¹(0.46875)/1.8627 ≈ -0.0421`` 与 ``≈ +0.0421``——即 NF4 在
       0 附近的最小分辨率是 0.0842（相对满量程），而**均匀量化在 0 附近
       分辨率最高**（间距 2/15 ≈ 0.1333，但包含 0 附近的点）。
       NF4 用"牺牲 0 附近的稠密性"换来"中间概率区间的更细分辨率"；
    3. 码点在两端更稀疏：``-1.0`` 与 ``-0.7076`` 之间的距离是
       ``-0.0421`` 与 ``+0.0421`` 之间距离的 6 倍。
    """
    normal = NormalDist()
    raw = [normal.inv_cdf((index + 0.5) / NF4_CODE_COUNT) for index in range(NF4_CODE_COUNT)]
    peak = max(abs(value) for value in raw)
    return tuple(value / peak for value in raw)


def fp4_levels() -> tuple[float, ...]:
    """FP4（E2M1：1 位符号 + 2 位指数 + 1 位尾数）的码点，归一化到 ``[-1, 1]``.

    8 个幅值 ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` 各配正负号，共 16 个编码。
    注意 **16 个编码只有 15 个不同取值**：``0`` 有 ``+0`` 与 ``-0`` 两个
    编码（``-0.0 == 0.0``）。这一点在小节里会被用到——搜索最近码点时
    重复的 0 不影响结果（取第一个即可），但它解释了"为什么 FP4 的有效
    码点比 NF4 少"。

    FP4 的码点分布是**指数式**的：靠近 0 处密（0.5 与 1 之间还有 0.5 的
    间距），远离 0 处疏（4 与 6 之间差 2）。对近似正态分布的权重来说，
    这不如 NF4 的分位数布局合适——这正是 QLoRA 选 NF4 而不是 FP4 的原因。
    """
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    peak = max(magnitudes)
    values = [-value / peak for value in magnitudes] + [value / peak for value in magnitudes]
    return tuple(sorted(values))


def int4_levels() -> tuple[float, ...]:
    """对称均匀 4-bit 的 16 个码点（``[-1, 1]`` 上等间距，间距 ``2/15``）.

    这是"最朴素"的 4-bit 方案，作为 NF4 的对照基线：把同样 16 个码点均匀
    铺开，而不是按概率密度铺开。实测（本课程模型的 557×557 权重矩阵、
    ``block_size=64``）会在 ``compare_codecs`` 里给出三者的误差对照。
    """
    return tuple((index - 7.5) / 7.5 for index in range(NF4_CODE_COUNT))


#: 码本名 → 生成函数（键名与 ``bnb_4bit_quant_type`` 一致，另加 ``int4`` 对照）
LEVEL_BUILDERS: dict[str, Any] = {
    "nf4": nf4_levels,
    "fp4": fp4_levels,
    "int4": int4_levels,
}


def levels_for(quant_type: str) -> tuple[float, ...]:
    """按名字取码本（未知名字抛 ``PEFTConfigError``）."""
    builder = LEVEL_BUILDERS.get(quant_type)
    if builder is None:
        raise PEFTConfigError(
            f"未知的量化码本 {quant_type!r}，可选：{', '.join(sorted(LEVEL_BUILDERS))}"
        )
    return tuple(builder())


#: 二级量化的缺省块大小（论文取 256；``BlockwiseResult`` 与 ``quantize_scales`` 共用）
DEFAULT_DOUBLE_QUANT_BLOCK_SIZE = 256


@dataclass(frozen=True)
class BlockwiseResult:
    """一次分块量化的原始产物：码点下标 + 每块的 ``absmax`` 常数.

    ``codes`` 与 ``scales`` 分开存，正是 4-bit 存储的物理布局：**码点
    连续存放（每参数 0.5 字节），常数另行存放**。计算某一项时再按块号
    取回它的常数——这就是"反量化" `reconstruction = level[code] * absmax`。
    """

    codes: tuple[int, ...]
    scales: tuple[float, ...]
    block_size: int
    quant_type: str
    levels: tuple[float, ...]

    @property
    def parameters(self) -> int:
        """被量化的参数个数."""
        return len(self.codes)

    @property
    def blocks(self) -> int:
        """块数（最后一个块可能不满）."""
        return len(self.scales)

    def code_bytes(self) -> int:
        """码点占用的字节数：``ceil(参数数 × 4 / 8)``（按位打包后向上取整）."""
        return math.ceil(self.parameters * CODE_BITS / 8)

    def constant_bytes(self, *, double_quant: bool) -> int:
        """常数占用的字节数.

        - 单重量化：每块一个 fp32 → ``blocks × 4``；
        - 二级量化：每块一个 8-bit 常数 → ``blocks × 1``，另加每
          ``double_quant_block_size`` 个常数一个 fp32 缩放。
        """
        if not double_quant:
            return self.blocks * FULL_PRECISION_CONSTANT_BITS // 8
        groups = math.ceil(self.blocks / DEFAULT_DOUBLE_QUANT_BLOCK_SIZE)
        return self.blocks * (SCALE_BITS // 8) + groups * (FULL_PRECISION_CONSTANT_BITS // 8)

    def dequantize(self) -> tuple[float, ...]:
        """反量化回浮点（``level[code] * scale``）."""
        levels = self.levels
        block = self.block_size
        return tuple(
            levels[code] * self.scales[index // block]
            for index, code in enumerate(self.codes)
        )


#: 二级量化的缺省块大小（论文取 256；放在模块级供 ``constant_bytes`` 复用）


def quantize_blockwise(
    values: Sequence[float],
    *,
    quant_type: str = "nf4",
    block_size: int = 64,
    levels: Sequence[float] | None = None,
) -> BlockwiseResult:
    """把一串浮点做分块量化，返回码点下标与每块的 ``absmax``.

    步骤（每一块独立做一遍）：

    .. code-block:: text

        absmax = max|v|                    块的量程
        v_norm = v / absmax                归一化到 [-1, 1]
        code   = argmin |v_norm - level|    最近的码点
        v_hat  = level[code] * absmax       反量化

    用 ``bisect`` 在**已排序的码本**上做最近邻搜索，而不是线性扫 16 个码点：
    前者是 ``O(log 16)``。在 557×557 = 310 249 个权重上，这一点差别就是
    "秒级"与"十几秒级"的区别——**教学代码也应当避免无谓的规模失配**。
    """
    if block_size <= 0:
        raise PEFTConfigError(f"block_size 必须为正整数，收到 {block_size}")
    table = tuple(sorted(levels if levels is not None else levels_for(quant_type)))
    if not table:
        raise PEFTConfigError("码本不能为空")
    codes: list[int] = []
    scales: list[float] = []
    for start in range(0, len(values), block_size):
        chunk = values[start : start + block_size]
        absmax = max(abs(value) for value in chunk)
        scale = absmax if absmax > 0.0 else 1.0
        scales.append(scale)
        for value in chunk:
            normalized = value / scale
            position = bisect.bisect_left(table, normalized)
            if position == 0:
                codes.append(0)
            elif position >= len(table):
                codes.append(len(table) - 1)
            else:
                lower = table[position - 1]
                upper = table[position]
                codes.append(
                    position - 1
                    if normalized - lower <= upper - normalized
                    else position
                )
    return BlockwiseResult(
        codes=tuple(codes),
        scales=tuple(scales),
        block_size=block_size,
        quant_type=quant_type,
        levels=table,
    )


def quantize_scales(scales: Sequence[float], *, block_size: int) -> tuple[int, int]:
    """把一级常数再量化一次，返回 ``(常数个数, 常数占用的字节数)``.

    ``bitsandbytes`` 这一步用 FP8（E4M3）存常数，每 256 个常数再存一个
    fp32 缩放；本课用 8-bit 对称线性量化模拟。**位宽相同，所以字节数完全
    一致**，差别只在量化误差的分布（见模块文档的说明）。
    """
    if block_size <= 0:
        raise PEFTConfigError(f"常数分组大小必须为正整数，收到 {block_size}")
    groups = math.ceil(len(scales) / block_size) if scales else 0
    bytes_used = len(scales) * (SCALE_BITS // 8) + groups * (FULL_PRECISION_CONSTANT_BITS // 8)
    return len(scales), bytes_used


@dataclass
class QuantizationReport:
    """一次量化的完整记录（误差 + 存储）."""

    quant_type: str
    block_size: int
    double_quant: bool
    rows: int
    cols: int
    blocks: int
    code_bytes: int
    constant_bytes: int
    #: 解析式给出的每参数字节数（``QLoRAConfig.bytes_per_parameter``，是下界）
    analytic_bytes_per_parameter: float
    mse: float
    relative_error: float
    max_abs_error: float
    weight_absmax: float
    #: 对照基线：16-bit 存同一份权重需要多少字节
    fp16_bytes: int
    fp32_bytes: int

    @property
    def parameters(self) -> int:
        """参数量."""
        return self.rows * self.cols

    @property
    def total_bytes(self) -> int:
        """实际存储字节数 = 码点 + 常数（含 ``ceil`` 取整）."""
        return self.code_bytes + self.constant_bytes

    @property
    def measured_bytes_per_parameter(self) -> float:
        """实测每参数字节数（含尾部取整），与解析式的差就是取整开销."""
        return self.total_bytes / self.parameters if self.parameters else 0.0

    @property
    def savings_vs_fp16(self) -> float:
        """相对 16-bit 存储省下的比例."""
        return 1.0 - self.total_bytes / self.fp16_bytes if self.fp16_bytes else 0.0

    @property
    def rmse(self) -> float:
        """量化的均方根误差（与权重量级同单位，便于判断"误差是否可接受"）."""
        return math.sqrt(self.mse)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.quant_type} block={self.block_size} "
            f"double_quant={self.double_quant} | {self.parameters} 参数 "
            f"{self.total_bytes} B（{self.measured_bytes_per_parameter:.6f} B/参数，"
            f"解析式 {self.analytic_bytes_per_parameter:.6f}）| "
            f"RMSE {self.rmse:.6f}（权重 absmax {self.weight_absmax:.6f}）| "
            f"相对 16-bit 省 {self.savings_vs_fp16:.2%}"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（附派生量）."""
        payload = asdict(self)
        payload["parameters"] = self.parameters
        payload["total_bytes"] = self.total_bytes
        payload["measured_bytes_per_parameter"] = round(
            self.measured_bytes_per_parameter, 6
        )
        payload["savings_vs_fp16"] = round(self.savings_vs_fp16, 6)
        payload["rmse"] = round(self.rmse, 8)
        payload["analytic_bytes_per_parameter"] = round(
            self.analytic_bytes_per_parameter, 6
        )
        return payload


def _flatten(matrix: Sequence[Sequence[float]], *, name: str) -> tuple[list[float], int, int]:
    """把二维矩阵摊平成一维（量化的物理布局是"连续存放"，不分行）."""
    rows = len(matrix)
    if rows == 0:
        raise PEFTConfigError(f"{name} 不能为空矩阵")
    cols = len(matrix[0])
    if cols == 0:
        raise PEFTConfigError(f"{name} 的列数不能为 0")
    flat: list[float] = []
    for index, row in enumerate(matrix):
        if len(row) != cols:
            raise PEFTConfigError(
                f"{name} 不是矩形：第 {index} 行有 {len(row)} 列，第 0 行有 {cols} 列"
            )
        flat.extend(float(value) for value in row)
    return flat, rows, cols


def error_metrics(
    original: Sequence[float], reconstructed: Sequence[float]
) -> tuple[float, float, float]:
    """返回 ``(MSE, 相对 Frobenius 误差, 最大绝对误差)``.

    三个指标各有用途：MSE/RMSE 有量纲（与权重量级同单位，好判断"这算不算大"），
    相对误差无量纲（好跨层比较），最大绝对误差用于发现**个别离群值的失效**
    （平均误差很小但某个权重被量化到离谱的值，会让那一整列的贡献失真）。
    """
    if len(original) != len(reconstructed):
        raise PEFTConfigError("原始与重建序列长度必须一致")
    if not original:
        raise PEFTConfigError("误差度量需要至少一个元素")
    squared = 0.0
    reference = 0.0
    largest = 0.0
    for source, target in zip(original, reconstructed):
        difference = source - target
        squared += difference * difference
        reference += source * source
        largest = max(largest, abs(difference))
    count = len(original)
    mse = squared / count
    relative = math.sqrt(squared / reference) if reference > 0 else 0.0
    return mse, relative, largest


def quantize_weights(
    matrix: Sequence[Sequence[float]],
    *,
    quant_type: str = "nf4",
    block_size: int = 64,
    double_quant: bool = True,
    double_quant_block_size: int = DEFAULT_DOUBLE_QUANT_BLOCK_SIZE,
) -> tuple[list[list[float]], QuantizationReport]:
    """量化一个二维权重矩阵并反量化，返回 ``(重建矩阵, 报告)``.

    QLoRA 的推理过程正是"存 4-bit、算 16-bit"：权重以 4-bit 落盘，前向计算
    时按块反量化回 bf16。因此本函数返回的是**反量化后的浮点矩阵**——
    后续在它上面挂 LoRA 训练，与真实 QLoRA 的行为一致。
    """
    flat, rows, cols = _flatten(matrix, name="待量化矩阵")
    result = quantize_blockwise(flat, quant_type=quant_type, block_size=block_size)
    reconstructed = result.dequantize()
    mse, relative, largest = error_metrics(flat, reconstructed)
    config = QLoRAConfig(
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        block_size=block_size,
        double_quant_block_size=double_quant_block_size,
    )
    config.validate()
    if double_quant:
        _, constant_bytes = quantize_scales(
            result.scales, block_size=double_quant_block_size
        )
    else:
        constant_bytes = result.constant_bytes(double_quant=False)
    report = QuantizationReport(
        quant_type=quant_type,
        block_size=block_size,
        double_quant=double_quant,
        rows=rows,
        cols=cols,
        blocks=result.blocks,
        code_bytes=result.code_bytes(),
        constant_bytes=constant_bytes,
        analytic_bytes_per_parameter=config.bytes_per_parameter,
        mse=mse,
        relative_error=relative,
        max_abs_error=largest,
        weight_absmax=max(abs(value) for value in flat),
        fp16_bytes=len(flat) * 2,
        fp32_bytes=len(flat) * 4,
    )
    rebuilt = [
        reconstructed[index * cols : (index + 1) * cols] for index in range(rows)
    ]
    return rebuilt, report


def quantize_reference_model(
    model: ReferenceSFTModel,
    config: QLoRAConfig | None = None,
) -> tuple[ReferenceSFTModel, QuantizationReport]:
    """按 QLoRA 的方式量化一个参考模型的**权重矩阵**，返回新的冻结模型.

    只量化权重矩阵：偏置与（真实模型里的）LayerNorm 保持原精度。这不是
    简化，而是 QLoRA 的实际做法——**被量化的只有占据绝大多数参数量的
    线性层权重**，常数量级的小张量留在 16-bit 更划算。

    ``updates`` 沿用原状态：量化**不是**一次参数更新，它只是换了一种存储。
    """
    effective = config or QLoRAConfig()
    effective.validate()
    state = model.state_dict()
    rebuilt, report = quantize_weights(
        state.weights,
        quant_type=effective.bnb_4bit_quant_type,
        block_size=effective.block_size,
        double_quant=effective.bnb_4bit_use_double_quant,
        double_quant_block_size=effective.double_quant_block_size,
    )
    quantized_state = ModelState(
        vocab_size=state.vocab_size,
        weights=tuple(tuple(row) for row in rebuilt),
        bias=tuple(state.bias),
        updates=state.updates,
    )
    return ReferenceSFTModel.from_state(quantized_state), report


def compare_codecs(
    matrix: Sequence[Sequence[float]],
    *,
    block_size: int = 64,
    quant_types: Sequence[str] = ("nf4", "fp4", "int4"),
) -> list[dict[str, Any]]:
    """对同一份权重跑多种码本，返回误差对照表（教程与 API 直接渲染）.

    这张表回答的是"为什么是 NF4"：三种码本的**存储字节数完全相同**
    （都是 4 bit + 同样的常数开销），差别只在误差。如果 NF4 的 RMSE
    明显最小，那 QLoRA 的选择就不是"惯例"而是"有数据支持的"。
    """
    rows: list[dict[str, Any]] = []
    for quant_type in quant_types:
        _, report = quantize_weights(
            matrix, quant_type=quant_type, block_size=block_size
        )
        payload = report.to_dict()
        payload.pop("fp32_bytes", None)
        rows.append(payload)
    rows.sort(key=lambda item: item["rmse"])
    return rows


def quantization_error_table(
    codes: Sequence[int], levels: Sequence[float]
) -> list[dict[str, float]]:
    """给出"每个码点 → 它代表的归一化值"的对照表（供文档展示码本形状）.

    返回的三列是 ``code``（十进制编码）、``level``（该编码代表的归一化值）与
    ``binary``（4 位二进制写法）。三列一起给出，是因为 NF4 的码点**不是**
    等间距的：只有把 16 行列出来，"两端稀疏、中间较密"这句话才有依据。
    """
    if len(codes) != len(levels):
        raise PEFTConfigError("codes 与 levels 长度必须一致")
    return [
        {
            "code": float(code),
            "level": float(levels[code]),
            "binary": format(code, "04b"),
        }
        for code in codes
    ]


__all__ = [
    "DEFAULT_DOUBLE_QUANT_BLOCK_SIZE",
    "LEVEL_BUILDERS",
    "NF4_CODE_COUNT",
    "SCALE_BITS",
    "SCALE_CODEC",
    "BlockwiseResult",
    "QuantizationReport",
    "compare_codecs",
    "error_metrics",
    "fp4_levels",
    "int4_levels",
    "levels_for",
    "nf4_levels",
    "quantize_blockwise",
    "quantize_reference_model",
    "quantize_scales",
    "quantize_weights",
    "quantization_error_table",
]
