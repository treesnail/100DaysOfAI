"""LoRA / QLoRA 参数高效微调包（M5-D3）：把"训多少参数"算清，再让它真的跑一次.

day050 把 SFT 的机制压缩成一个整数（``-100``）；今天把微调的**规模**压缩成
一个比例：

.. code-block:: text

    全参微调：训 V² + V 个参数              （本课程参考模型：310 806 个）
    LoRA r=8：训 2·r·V 个参数               （参考模型：8 912 个 = 2.787%）
    真实模型上：一个 in×out 的投影层新增 r(in+out) 个参数，方阵时比例 = 2r/d

四层结构，与 day050 ``sft`` 包的分层纪律一致：

============================  ==========================================
``config``                    ``LoRAConfig`` / ``QLoRAConfig``：纯数据 + 纯校验
``layers`` / ``models``       矩阵算术与参考模型：真梯度、可手算、可对照
``targets`` / ``memory``      架构与显存算术：真实规格上的参数量与预算
``qlora`` / ``hf_script``     量化码本与脚本生成：从机制到可执行产物
============================  ==========================================

一条贯穿全包的边界：**LoRA 与训练循环正交**。``LoRAReferenceModel`` 与
day050 的 ``ReferenceSFTModel`` 接口完全一致，因此渲染、编码、label mask、
padding、步数算术、检查点落盘全都原样复用——今天新增的代码里没有一行
重新实现它们。把这条边界划清楚，"换微调方法"才是换一层皮，而不是重写脚本。
"""

from __future__ import annotations

from smart_research_agent.peft.config import (
    BASELINE_ONLY_QUANT_TYPES,
    LORA_TARGET_PRESETS,
    REFERENCE_MODULE_NAME,
    SUPPORTED_INIT_MODES,
    SUPPORTED_LORA_BIASES,
    SUPPORTED_QUANT_BLOCK_SIZES,
    SUPPORTED_QUANT_TYPES,
    LoRAConfig,
    PEFTConfigError,
    QLoRAConfig,
    default_peft_lr_range,
    quantization_table,
)
from smart_research_agent.peft.hf_script import (
    DTYPE_MAPPING_SNIPPET,
    PEFT_DEPENDENCIES,
    QLORA_DEPENDENCIES,
    peft_dependency_commands,
    render_lora_script,
    render_qlora_script,
    rendered_peft_summary,
)
from smart_research_agent.peft.layers import (
    ZERO_TOLERANCE,
    LoRALinear,
    adapter_state_size,
    add_matrices,
    context_delta,
    describe_delta,
    frobenius_norm,
    init_lora_weights,
    is_rank_one,
    lora_delta,
    matmul,
    matrix_rank,
    max_abs,
    merge_lora_weight,
)
from smart_research_agent.peft.memory import (
    COMPUTE_DTYPE_BYTES,
    DEVICE_BUDGETS,
    OPTIMIZER_STATES,
    STRATEGIES,
    MemoryBreakdown,
    adamw_state_bytes,
    compare_strategies,
    device_fit_table,
    format_bytes,
    logistic_estimate_hours,
    minimal_device,
    parse_size,
    plan_memory,
    quantization_memory_note,
    savings_table,
)
from smart_research_agent.peft.models import (
    REFERENCE_LORA_ALPHA,
    REFERENCE_LORA_RANK,
    LoRAReferenceModel,
    default_reference_lora_config,
    reference_lora_accounting,
    train_lora_reference,
)
from smart_research_agent.peft.qlora import (
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
from smart_research_agent.peft.targets import (
    MODEL_SPECS,
    TARGET_LAYER_DESCRIPTIONS,
    DecoderSpec,
    LoRAPlan,
    ModuleShape,
    adapter_footprint,
    describe_model,
    parameter_check,
    plan_lora,
    rank_comparison,
    ratio_curve,
    reference_matrix_ratios,
    single_matrix_ratio,
    square_ratio_shortcut,
    target_preset_table,
    theoretical_adapter_parameter_cap,
)

__all__ = [
    "BASELINE_ONLY_QUANT_TYPES",
    "COMPUTE_DTYPE_BYTES",
    "DEVICE_BUDGETS",
    "DTYPE_MAPPING_SNIPPET",
    "LORA_TARGET_PRESETS",
    "MODEL_SPECS",
    "OPTIMIZER_STATES",
    "PEFT_DEPENDENCIES",
    "QLORA_DEPENDENCIES",
    "REFERENCE_LORA_ALPHA",
    "REFERENCE_LORA_RANK",
    "REFERENCE_MODULE_NAME",
    "STRATEGIES",
    "SUPPORTED_INIT_MODES",
    "SUPPORTED_LORA_BIASES",
    "SUPPORTED_QUANT_BLOCK_SIZES",
    "SUPPORTED_QUANT_TYPES",
    "TARGET_LAYER_DESCRIPTIONS",
    "ZERO_TOLERANCE",
    "BlockwiseResult",
    "DecoderSpec",
    "LoRAConfig",
    "LoRALinear",
    "LoRAPlan",
    "LoRAReferenceModel",
    "MemoryBreakdown",
    "ModuleShape",
    "PEFTConfigError",
    "QLoRAConfig",
    "QuantizationReport",
    "adamw_state_bytes",
    "adapter_footprint",
    "adapter_state_size",
    "add_matrices",
    "compare_codecs",
    "compare_strategies",
    "default_peft_lr_range",
    "default_reference_lora_config",
    "describe_delta",
    "describe_model",
    "device_fit_table",
    "error_metrics",
    "format_bytes",
    "fp4_levels",
    "frobenius_norm",
    "init_lora_weights",
    "int4_levels",
    "is_rank_one",
    "levels_for",
    "logistic_estimate_hours",
    "lora_delta",
    "matmul",
    "matrix_rank",
    "max_abs",
    "merge_lora_weight",
    "minimal_device",
    "nf4_levels",
    "parameter_check",
    "parse_size",
    "peft_dependency_commands",
    "plan_lora",
    "plan_memory",
    "quantization_error_table",
    "quantization_memory_note",
    "quantization_table",
    "quantize_blockwise",
    "quantize_reference_model",
    "quantize_scales",
    "quantize_weights",
    "rank_comparison",
    "ratio_curve",
    "reference_lora_accounting",
    "reference_matrix_ratios",
    "render_lora_script",
    "render_qlora_script",
    "rendered_peft_summary",
    "context_delta",
    "savings_table",
    "single_matrix_ratio",
    "square_ratio_shortcut",
    "target_preset_table",
    "theoretical_adapter_parameter_cap",
    "train_lora_reference",
]
