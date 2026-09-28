"""QLoRA 量化算术测试（M5-D3）：码本、分块量化、常数量化与误差报告.

本文件的断言把模块文档里那几句"可以被逐位验证的话"钉死：

- NF4 的 16 个码点**没有 0**、首尾是 ``±1.0``、严格递增；
- FP4 的 16 个编码**只有 15 个不同取值**（``+0`` 与 ``-0`` 同一个数）；
- 分块量化每块共享一个 ``absmax``，全零块的常数退化为 ``1.0``；
- 报告里的"实测字节数"（含 ``ceil`` 取整）**不小于**解析式给出的下界；
- ``compare_codecs`` 按 ``rmse`` 升序返回，正态权重上 NF4 优于均匀 4-bit。

所有数字都只依赖标准库，不需要 GPU、不需要 bitsandbytes。
"""

from __future__ import annotations

import math
import random

import pytest

from smart_research_agent.peft.config import PEFTConfigError, QLoRAConfig
from smart_research_agent.peft.qlora import (
    NF4_CODE_COUNT,
    BlockwiseResult,
    QuantizationReport,
    compare_codecs,
    error_metrics,
    fp4_levels,
    int4_levels,
    levels_for,
    nf4_levels,
    quantization_error_table,
    quantize_blockwise,
    quantize_reference_model,
    quantize_scales,
    quantize_weights,
)
from smart_research_agent.sft.reference_model import ModelState, ReferenceSFTModel

#: 一个 2×2 的手算矩阵：权重 absmax=4，4 个参数；``block_size=64`` 时只有 1 块
SMALL_MATRIX = [[1.0, 2.0], [3.0, 4.0]]


class TestCodebooks:
    """码本：NF4 / FP4 / int4 三套码点的定义性质."""

    def test_nf4_has_sixteen_levels(self):
        assert len(nf4_levels()) == NF4_CODE_COUNT == 16

    def test_nf4_is_strictly_increasing(self):
        levels = nf4_levels()
        assert all(low < high for low, high in zip(levels, levels[1:]))

    def test_nf4_endpoints_are_plus_minus_one(self):
        """归一化到 ``[-1, 1]``：首码点是 -1.0、末码点是 1.0."""
        levels = nf4_levels()
        assert levels[0] == -1.0
        assert levels[-1] == 1.0

    def test_nf4_never_hits_zero(self):
        """偶数个等概率分位数：0 落在中间两个码点之间，码本里没有 0."""
        assert 0.0 not in nf4_levels()

    def test_nf4_middle_pair_is_symmetric(self):
        levels = nf4_levels()
        assert levels[8] == pytest.approx(-levels[7])

    def test_fp4_has_sixteen_codes_but_fifteen_values(self):
        """``+0`` 与 ``-0`` 是同一个数：16 个编码只有 15 个不同取值."""
        levels = fp4_levels()
        assert len(levels) == 16
        assert len(set(levels)) == 15

    def test_fp4_endpoints_are_plus_minus_one(self):
        levels = fp4_levels()
        assert levels[0] == -1.0
        assert levels[-1] == 1.0

    def test_int4_is_uniformly_spaced(self):
        levels = int4_levels()
        gaps = {round(high - low, 12) for low, high in zip(levels, levels[1:])}
        assert gaps == {round(2 / 15, 12)}

    def test_int4_endpoints_are_plus_minus_one(self):
        levels = int4_levels()
        assert levels[0] == -1.0
        assert levels[-1] == 1.0

    def test_levels_for_known_names(self):
        assert levels_for("nf4") == nf4_levels()
        assert levels_for("fp4") == fp4_levels()
        assert levels_for("int4") == int4_levels()

    def test_levels_for_unknown_name_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的量化码本"):
            levels_for("int8")


class TestQuantizeBlockwise:
    """分块量化：分块数、absmax、全零块、最近码点、自定义码本与非法输入."""

    def test_blocks_are_ceil_of_length_over_block_size(self):
        result = quantize_blockwise([0.1] * 130, block_size=64)
        assert result.parameters == 130
        assert result.blocks == 3  # ceil(130 / 64)
        assert len(result.scales) == 3

    def test_scale_is_block_absmax(self):
        result = quantize_blockwise([-3.0, 1.0], block_size=64)
        assert result.scales == pytest.approx((3.0,))

    def test_all_zero_block_uses_unit_scale(self):
        """全零块没有量程可言，常数退化为 1.0（避免除零），码点仍然统一."""
        result = quantize_blockwise([0.0, 0.0, 0.0], block_size=2)
        assert result.scales == (1.0, 1.0)
        assert len(set(result.codes)) == 1

    def test_nearest_level_is_chosen(self):
        """0.4 更靠近 0、0.6 更靠近 1——按绝对值距离取最近码点."""
        result = quantize_blockwise([0.4, 0.6, 1.0], levels=(0.0, 1.0), block_size=64)
        assert result.codes == (0, 1, 1)

    def test_exact_levels_round_trip(self):
        levels = (0.0, 0.5, 1.0)
        result = quantize_blockwise([0.0, 0.5, 1.0], levels=levels, block_size=64)
        assert result.codes == (0, 1, 2)
        assert result.dequantize() == pytest.approx(levels)

    def test_custom_levels_are_sorted_before_search(self):
        result = quantize_blockwise([0.5], levels=(1.0, 0.0), block_size=64)
        assert result.levels == (0.0, 1.0)

    def test_non_positive_block_size_rejected(self):
        with pytest.raises(PEFTConfigError, match="block_size 必须为正整数"):
            quantize_blockwise([1.0], block_size=0)

    def test_empty_codebook_rejected(self):
        with pytest.raises(PEFTConfigError, match="码本不能为空"):
            quantize_blockwise([1.0], levels=())


class TestBlockwiseResult:
    """BlockwiseResult 的派生量：参数量、块数、字节数与反量化."""

    def test_parameters_and_blocks(self):
        result = BlockwiseResult(
            codes=(0, 1, 2, 3),
            scales=(1.0,),
            block_size=4,
            quant_type="nf4",
            levels=nf4_levels(),
        )
        assert result.parameters == 4
        assert result.blocks == 1

    def test_code_bytes_rounds_up(self):
        result = BlockwiseResult(
            codes=(0,) * 5,
            scales=(1.0,),
            block_size=64,
            quant_type="nf4",
            levels=nf4_levels(),
        )
        assert result.code_bytes() == 3  # ceil(5 × 4 / 8)

    def test_constant_bytes_single_quant(self):
        """单重量化：每块一个 fp32 常数."""
        result = BlockwiseResult(
            codes=tuple(range(64)),
            scales=(1.0,) * 3,
            block_size=64,
            quant_type="nf4",
            levels=nf4_levels(),
        )
        assert result.constant_bytes(double_quant=False) == 12  # 3 × 32 // 8

    def test_constant_bytes_double_quant(self):
        """二级量化：每块 1 字节 + 每 256 个常数一个 fp32 缩放."""
        result = BlockwiseResult(
            codes=tuple(range(64)),
            scales=(1.0,) * 3,
            block_size=64,
            quant_type="nf4",
            levels=nf4_levels(),
        )
        assert result.constant_bytes(double_quant=True) == 7  # 3 × 1 + 1 × 4

    def test_dequantize_multiplies_level_by_block_scale(self):
        result = BlockwiseResult(
            codes=(0, 15, 15, 0),
            scales=(2.0, 3.0),
            block_size=2,
            quant_type="nf4",
            levels=nf4_levels(),
        )
        levels = nf4_levels()
        assert result.dequantize() == pytest.approx(
            (levels[0] * 2.0, levels[15] * 2.0, levels[15] * 3.0, levels[0] * 3.0)
        )


class TestQuantizeScales:
    """常数量化 ``quantize_scales``：返回 ``(常数个数, 常数占用字节数)``."""

    def test_counts_and_bytes(self):
        assert quantize_scales((1.0,) * 300, block_size=256) == (300, 308)

    def test_empty_scales_give_zero(self):
        assert quantize_scales((), block_size=256) == (0, 0)

    def test_non_positive_block_size_rejected(self):
        with pytest.raises(PEFTConfigError, match="常数分组大小必须为正整数"):
            quantize_scales((1.0,), block_size=0)


class TestErrorMetrics:
    """误差度量：三个指标的定义，以及长度不一致与空输入."""

    def test_metrics_values(self):
        """手算：平方误差和 4 / 3 个元素；相对误差 sqrt(4/9)；最大误差 2."""
        mse, relative, largest = error_metrics([1.0, 2.0, 2.0], [1.0, 2.0, 0.0])
        assert mse == pytest.approx(4.0 / 3.0)
        assert relative == pytest.approx(2.0 / 3.0)
        assert largest == pytest.approx(2.0)

    def test_perfect_reconstruction_is_all_zero(self):
        assert error_metrics([1.0, -2.0], [1.0, -2.0]) == (0.0, 0.0, 0.0)

    def test_zero_reference_avoids_division_by_zero(self):
        """原序列全零时相对误差定义为 0，而不是抛 ZeroDivisionError."""
        mse, relative, largest = error_metrics([0.0], [1.0])
        assert mse == pytest.approx(1.0)
        assert relative == 0.0
        assert largest == pytest.approx(1.0)

    def test_length_mismatch_rejected(self):
        with pytest.raises(PEFTConfigError, match="长度必须一致"):
            error_metrics([1.0], [1.0, 2.0])

    def test_empty_input_rejected(self):
        with pytest.raises(PEFTConfigError, match="至少一个元素"):
            error_metrics([], [])


class TestQuantizeWeights:
    """``quantize_weights``：报告字段、double_quant 开关、取整差与 savings."""

    def test_report_fields(self):
        rebuilt, report = quantize_weights(SMALL_MATRIX, block_size=64)
        assert isinstance(report, QuantizationReport)
        assert (report.rows, report.cols) == (2, 2)
        assert report.parameters == 4
        assert report.blocks == 1
        assert report.code_bytes == 2  # ceil(4 × 4 / 8)
        assert report.quant_type == "nf4"
        assert report.double_quant is True
        assert report.weight_absmax == pytest.approx(4.0)
        assert report.fp16_bytes == 8
        assert report.fp32_bytes == 16
        assert report.mse > 0.0
        assert report.relative_error > 0.0
        assert report.max_abs_error > 0.0
        assert [len(row) for row in rebuilt] == [2, 2]

    def test_double_quant_switches_constant_storage(self):
        _, on = quantize_weights(SMALL_MATRIX, block_size=64, double_quant=True)
        _, off = quantize_weights(SMALL_MATRIX, block_size=64, double_quant=False)
        assert on.double_quant is True and off.double_quant is False
        assert off.constant_bytes == 4  # 1 × 32 // 8
        assert on.constant_bytes == 5  # 1 × 1 + 1 × 4

    def test_double_quant_saves_when_many_blocks(self):
        """块数一多，把 fp32 常数换成 8-bit 才开始省字节."""
        matrix = [[1.0] * 64 for _ in range(64)]
        _, on = quantize_weights(matrix, block_size=64, double_quant=True)
        _, off = quantize_weights(matrix, block_size=64, double_quant=False)
        assert on.blocks == off.blocks == 64
        assert on.constant_bytes == 68  # 64 + 1 × 4
        assert off.constant_bytes == 256  # 64 × 4
        assert on.total_bytes < off.total_bytes

    def test_measured_bytes_exceed_analytic_after_rounding(self):
        """解析式是下界；``ceil`` 取整让实测值严格更大."""
        _, report = quantize_weights(SMALL_MATRIX, block_size=64)
        assert report.analytic_bytes_per_parameter == pytest.approx(0.515869140625)
        assert report.measured_bytes_per_parameter > report.analytic_bytes_per_parameter

    def test_savings_vs_fp16(self):
        _, report = quantize_weights(SMALL_MATRIX, block_size=64)
        expected = 1.0 - report.total_bytes / report.fp16_bytes
        assert report.savings_vs_fp16 == pytest.approx(expected)
        assert report.savings_vs_fp16 > 0.0

    def test_rmse_is_sqrt_of_mse(self):
        _, report = quantize_weights(SMALL_MATRIX, block_size=64)
        assert report.rmse == pytest.approx(math.sqrt(report.mse))

    def test_invalid_block_size_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的 block_size"):
            quantize_weights(SMALL_MATRIX, block_size=100)


class TestQuantizeReferenceModel:
    """``quantize_reference_model``：只量化权重，bias 与 updates 原样保留."""

    @staticmethod
    def _reference(*, updates: int = 3) -> ReferenceSFTModel:
        """构造一个带非零偏置与更新计数的参考模型."""
        state = ModelState(
            vocab_size=8,
            weights=tuple(tuple(0.01 * (row + 1) for _ in range(8)) for row in range(8)),
            bias=tuple(0.5 for _ in range(8)),
            updates=updates,
        )
        return ReferenceSFTModel.from_state(state)

    def test_returns_a_new_model(self):
        base = self._reference()
        quantized, _ = quantize_reference_model(base)
        assert quantized is not base

    def test_bias_and_updates_are_preserved(self):
        """量化只换存储、不是一次参数更新：bias 与 updates 必须逐位不变."""
        base = self._reference(updates=3)
        quantized, _ = quantize_reference_model(base)
        assert quantized.state_dict().bias == base.state_dict().bias
        assert quantized.state_dict().updates == base.state_dict().updates

    def test_weights_are_quantized(self):
        base = self._reference()
        quantized, report = quantize_reference_model(base)
        assert quantized.state_dict().weights != base.state_dict().weights
        assert (report.rows, report.cols) == (8, 8)

    def test_accepts_explicit_config(self):
        base = self._reference()
        config = QLoRAConfig(bnb_4bit_use_double_quant=False, block_size=128)
        quantized, report = quantize_reference_model(base, config)
        assert report.double_quant is False
        assert report.block_size == 128
        assert quantized.state_dict().bias == base.state_dict().bias


class TestCompareCodecs:
    """``compare_codecs``：三种码本在同一份权重上的误差对照，按 rmse 升序."""

    @staticmethod
    def _matrix() -> list[list[float]]:
        """一张固定种子的正态权重矩阵（确定性，避免测试抖动）."""
        rng = random.Random(1234)
        return [[rng.gauss(0.0, 1.0) for _ in range(48)] for _ in range(48)]

    def test_rows_are_sorted_by_rmse(self):
        rows = compare_codecs(self._matrix(), block_size=64)
        rmses = [row["rmse"] for row in rows]
        assert rmses == sorted(rmses)

    def test_returns_three_codecs_with_nf4_best(self):
        rows = compare_codecs(self._matrix(), block_size=64)
        assert {row["quant_type"] for row in rows} == {"nf4", "fp4", "int4"}
        assert rows[0]["quant_type"] == "nf4"

    def test_nf4_beats_uniform_quantization(self):
        """正态权重上 NF4 的 RMSE 小于均匀 4-bit——这就是选 NF4 的数据依据."""
        rows = compare_codecs(self._matrix(), block_size=64)
        rmses = {row["quant_type"]: row["rmse"] for row in rows}
        assert rmses["nf4"] < rmses["int4"]

    def test_fp32_bytes_is_dropped_from_the_table(self):
        rows = compare_codecs(self._matrix(), block_size=64)
        assert all("fp32_bytes" not in row for row in rows)


class TestQuantizationErrorTable:
    """``quantization_error_table``：码点 → 归一化值 + 4 位二进制."""

    def test_row_shape(self):
        table = quantization_error_table(list(range(16)), nf4_levels())
        assert len(table) == 16
        assert table[0] == {"code": 0.0, "level": -1.0, "binary": "0000"}
        assert table[-1] == {"code": 15.0, "level": 1.0, "binary": "1111"}

    def test_levels_come_from_the_codebook(self):
        """``levels`` 按码点下标取值：16 个码点逐一对应 16 个 level."""
        levels = nf4_levels()
        table = quantization_error_table(list(range(16)), levels)
        assert table[3]["level"] == pytest.approx(levels[3])
        assert table[12]["level"] == pytest.approx(levels[12])

    def test_length_mismatch_rejected(self):
        with pytest.raises(PEFTConfigError, match="codes 与 levels 长度必须一致"):
            quantization_error_table([0, 1], nf4_levels())


class TestBlockwiseEdgeCases:
    """``quantize_blockwise`` 的边界：码本上界之外的值与自定义码本.

    ``bisect`` 的最近邻搜索有三个出口：取第 0 个码点、取最后一个码点、
    以及两端之间的比较。第三个出口在上界之外时才会命中——而**归一化之后
    ``|v_norm| ≤ 1``，所以用 NF4/INT4 这类满量程码本永远走不到它**。
    只有把码本换成"最大码点小于 1"的自定义表才能覆盖（测试里用 ``(0.0, 0.5)``）。
    """

    def test_value_above_max_level_takes_last_code(self):
        result = quantize_blockwise(
            [1.0, 0.4], quant_type="nf4", block_size=2, levels=(0.0, 0.5)
        )
        # 归一化后的 1.0 已经超出码本最大码点 0.5，只能落到最后一个码点
        assert result.codes == (1, 1)
        assert result.dequantize() == pytest.approx((0.5, 0.5))

    def test_value_below_min_level_takes_first_code(self):
        result = quantize_blockwise(
            [-1.0, -0.4], quant_type="nf4", block_size=2, levels=(-0.5, 0.0)
        )
        # -0.4 离 -0.5（距离 0.1）比离 0.0（距离 0.4）更近，因此取第 0 个码点
        assert result.codes == (0, 0)
        assert result.dequantize() == pytest.approx((-0.5, -0.5))

    def test_report_summary_line_mentions_quant_type(self):
        _, report = quantize_weights(
            [[0.1, -0.2], [0.3, 0.4]], quant_type="nf4", block_size=64, double_quant=False
        )
        line = report.summary_line()
        assert "nf4" in line
        assert "double_quant=False" in line
        assert f"{report.parameters} 参数" in line

    def test_flatten_rejects_empty_matrix(self):
        with pytest.raises(PEFTConfigError, match="待量化矩阵 不能为空矩阵"):
            quantize_weights([])

    def test_flatten_rejects_zero_columns(self):
        with pytest.raises(PEFTConfigError, match="列数不能为 0"):
            quantize_weights([[]])

    def test_flatten_rejects_ragged_matrix(self):
        with pytest.raises(PEFTConfigError, match="不是矩形"):
            quantize_weights([[0.1, 0.2], [0.3]])
