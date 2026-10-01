"""``transformer_stack``：把一个块复制成一条链（M7-D5 / day080）.

day079 交出一个**块**，并留下一句话：*块是唯一被复制的单位*。
今天把那句话执行掉——整条链就是同一个块被复制 N 份：

```text
          ┌──────────── 一条链（N 份） ────────────┐
x ──► 块₀ ──► 块₁ ──► … ──► 块_{N-1} ──► y         块 = LN → 注意力 → ⊕ → LN → 前馈 → ⊕
          └────────────────────────────────────────┘
```

## 一、今天新增的代码只有三行要紧的

```text
前向     current = encoder_block(参数_i, current, block_attention(...)).output
反向     current = encoder_block_backward(账_i, 参数_i, current).grad_inputs
记账     每一层记一行（‖x_i‖ / ‖y_i‖ / ‖b1‖ / ‖b2‖ / 最大绝对值）
```

而这三行里各藏着一处**“漏掉不会报错”**的写法：

```text
前向   把 y_i **原样**交给下一层（重算一遍会差 1e-16，而它让"逐位一致"失效）
反向   第 i 层拿到的 grad_output 必须是第 i+1 层的 grad_inputs
       ——换个写法（每层都从 dLoss/dy_N 起步）形状全对，只是链被掐断了
记账   范数要用 math.fsum：跨层比较的两个数必须同口径
```

## 二、于是今天有三样东西可以量

```text
增益        ‖y_i‖/‖x_i‖        这一层把整体尺度放大还是缩小
直通占比     ‖x‖/(‖x‖+‖b1‖+‖b2‖)  输出里有多少来自残差直通（残差关时恰好为 0）
逐层 ‖dx‖    每一层入口的梯度范数     day079 只看过最底层那一个数，今天把它变成序列
```

## 三、六条性质里有两条用 ``==``

```text
保形                    逐层 + 传递 + 出口，全部逐位
与手写循环一致           两条路径做同一串浮点运算 ⇒ 必须逐位相等
确定性                  没有随机数 ⇒ 两次调用逐位相同
每一层分支全零 ⇒ 整条链是恒等
解析参数量 == 逐块数出来的参数量
生成的 PyTorch 脚本带着同一个 CONFIG
```

“与手写循环逐位一致”是本课最便宜也最值钱的一条：它把
“``stack_forward`` 只是 ``encoder_block`` 外面套了一个 ``for``”这句话
变成一条**可以失败**的断言。

## 四、PyTorch 组装：生成一份脚本，而不是在包里 import torch

```text
包里的代码     纯 Python（逐位可复核），**不 import torch**——与 day073~079 同一条纪律
生成的脚本     assembly_script(shape, ...) 产出一段**完整可运行**的 PyTorch 代码
结构核对       assembly_facts(script) 用 ast 把那段文本解析回类名 / nn 模块 / CONFIG
本机实测       在有 torch 的机器上真的跑一遍，把真实输出写进教程（第 7 章）
```

## 五、五个模块

```text
errors.py    五族失败：形状 / 参数 / **组装** / 数值 / 梯度
types.py     形状、五个阶段、六条性质、两份梯度名单、三种记录（读数 / 前向账 / 反向账）
layers.py    链式前向、链式反向、逐层读数、参数量、置零开关
verify.py    两处梯度校验（整条链的输入 / 第 k 层的八块参数）+ 六条性质
assembly.py  生成 PyTorch 组装脚本，并把那段文本解析回结构
```

## 六、四条纪律

1. **逐位可断言的绝不留容差**：保形、传递、确定性与“分支全零即恒等”都用 ``==``。
2. **链式反向只在“换手”那一行**：三行代码里最容易写错的是 ``current = grads.grad_inputs``。
3. **口径必须两边一致**：解析侧与数值侧用同一个 ``StackParameters``、同一个 ``placement``、
   同一个 ``use_residual``、同一个 ``activation``——因此“忘了传”在结构上不可能发生。
4. **差异必须被看见**：生成的脚本会打印 PyTorch 的参数量与本课解析式的差值
   （``4d × N``，来自 ``nn.MultiheadAttention`` 的偏置），而不是把它凑平。

## 七、与既有包的接缝

- **上游**：``encoder_decoder``（day079 的 ``encoder_block`` / ``encoder_block_backward`` /
  ``block_attention``）、``transformer_core``（day075 的 ``self_attention`` 与 ``default_parameters``）、
  ``math_foundations``（day073 的 ``flatten_matrices``、day074 的 ``calculus.gradient``）；
- **脚下**：``config`` **没有**新增配置项——层数、摆放位置、残差开关与激活
  都是“这一次调用或这一次对照的判据”，它们进的是函数参数（与 day073~079 同一条纪律）；
- **下游**：day081（训练优化与正则化）会拿这条链去真训——
  它要的是**逐层梯度序列**（今天交出来了）、以及“post-LN 需要预热吗”这个 day079 留下的问题；
  day082（变体架构）会看到 BERT/GPT 就是“取这摞块的哪几层”；
  day085（源码精读）会看到 Hugging Face 的 ``TransformerEncoder`` 就是 ``ModuleList`` + 一个 for。
"""

from __future__ import annotations

from smart_research_agent.transformer_stack.assembly import (
    ASSEMBLY_CLASSES,
    ASSEMBLY_CONFIG_KEYS,
    ASSEMBLY_MODULES,
    ASSEMBLY_REQUIREMENT,
    activation_is_supported,
    assembly_facts,
    assembly_script,
    placement_is_supported,
)
from smart_research_agent.transformer_stack.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
    StackError,
)
from smart_research_agent.transformer_stack.layers import (
    ATTENTION_SEED_STRIDE,
    DEFAULT_FFN_RATIO,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_TOKENS,
    LAYER_SEED_STRIDE,
    flatten_layer_parameter_norms,
    layer_census_of,
    make_shape,
    make_stack_parameters,
    sequence_slice,
    stack_backward,
    stack_forward,
    stack_gradient_profile,
    stack_loss,
    stack_loss_gradient,
    zero_attention,
    zero_branches,
)
from smart_research_agent.transformer_stack.study import (
    DEFAULT_STUDY_DEPTHS,
    STUDY_VARIANTS,
    VARIANT_BARE,
    VARIANT_DESCRIPTIONS,
    VARIANT_RESIDUAL,
    StackRow,
    StackStudy,
    geometric_mean,
    stack_study,
)
from smart_research_agent.transformer_stack.types import (
    DEFAULT_INIT_SCALE,
    GRAD_STACK_INPUTS,
    KINDS,
    KIND_LAYER,
    KIND_STACK,
    LAYER_GRADIENT_TARGETS,
    MINIMUM_LAYERS,
    PROPERTY_ASSEMBLY_CARRIES_SHAPE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH,
    PROPERTY_MATCHES_LOOP,
    PROPERTY_PARAMETER_COUNT_MATCHES,
    PROPERTY_SHAPE_PRESERVED,
    STACK_GRADIENT_TARGETS,
    STACK_NOTES,
    STACK_PROPERTIES,
    STACK_STAGES,
    STAGE_BLOCK,
    STAGE_CARRY,
    STAGE_CENSUS,
    STAGE_DESCRIPTIONS,
    STAGE_ENTER,
    STAGE_EXIT,
    STAGE_SHAPES,
    LayerCensus,
    StackForward,
    StackGradients,
    StackParameters,
    StackShape,
    StackedLayer,
    frobenius,
    max_absolute,
    relative_matrix_error,
)
from smart_research_agent.transformer_stack.verify import (
    GRADIENT_TOLERANCE,
    GradientOutcome,
    GradientReport,
    PropertyOutcome,
    PropertyReport,
    census_of_layer,
    check_assembly_carries_shape,
    check_layer_parameter_gradients,
    check_parameter_count_matches,
    check_properties,
    check_stack_identity_when_branches_vanishes,
    check_stack_input_gradient,
    check_stack_is_deterministic,
    check_stack_matches_loop,
    check_stack_preserves_shape,
    checked_vector,
    frobenius_profile,
    gradient_summary,
    layer_norm_sequence,
    make_samples,
    matrix_norm_of,
    numerical_layer_parameter_gradients,
    numerical_stack_input_gradient,
    parse_script_module_names,
    relative_gradient_change,
    stack_stage_lines,
)

__all__ = [
    "ASSEMBLY_CLASSES",
    "ASSEMBLY_CONFIG_KEYS",
    "ASSEMBLY_MODULES",
    "ASSEMBLY_REQUIREMENT",
    "ATTENTION_SEED_STRIDE",
    "DEFAULT_FFN_RATIO",
    "DEFAULT_HIDDEN",
    "DEFAULT_INIT_SCALE",
    "DEFAULT_LAYERS",
    "DEFAULT_STUDY_DEPTHS",
    "DEFAULT_TOKENS",
    "FAMILY_OUTCOMES",
    "GRADIENT_TOLERANCE",
    "GRAD_STACK_INPUTS",
    "KINDS",
    "KIND_LAYER",
    "KIND_STACK",
    "LAYER_GRADIENT_TARGETS",
    "LAYER_SEED_STRIDE",
    "MINIMUM_LAYERS",
    "PROPERTY_ASSEMBLY_CARRIES_SHAPE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH",
    "PROPERTY_MATCHES_LOOP",
    "PROPERTY_PARAMETER_COUNT_MATCHES",
    "PROPERTY_SHAPE_PRESERVED",
    "STACK_GRADIENT_TARGETS",
    "STACK_NOTES",
    "STACK_PROPERTIES",
    "STACK_STAGES",
    "STAGE_BLOCK",
    "STAGE_CARRY",
    "STAGE_CENSUS",
    "STAGE_DESCRIPTIONS",
    "STAGE_ENTER",
    "STAGE_EXIT",
    "STAGE_SHAPES",
    "STUDY_VARIANTS",
    "AssemblyError",
    "GradientError",
    "GradientOutcome",
    "GradientReport",
    "LayerCensus",
    "NumericError",
    "ParameterError",
    "PropertyOutcome",
    "PropertyReport",
    "ShapeError",
    "StackError",
    "StackForward",
    "StackGradients",
    "StackParameters",
    "StackShape",
    "StackedLayer",
    "StackRow",
    "StackStudy",
    "VARIANT_BARE",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_RESIDUAL",
    "activation_is_supported",
    "assembly_facts",
    "assembly_script",
    "census_of_layer",
    "check_assembly_carries_shape",
    "check_layer_parameter_gradients",
    "check_parameter_count_matches",
    "check_properties",
    "check_stack_identity_when_branches_vanishes",
    "check_stack_input_gradient",
    "check_stack_is_deterministic",
    "check_stack_matches_loop",
    "check_stack_preserves_shape",
    "checked_vector",
    "flatten_layer_parameter_norms",
    "frobenius",
    "frobenius_profile",
    "geometric_mean",
    "gradient_summary",
    "layer_census_of",
    "layer_norm_sequence",
    "make_samples",
    "make_shape",
    "make_stack_parameters",
    "matrix_norm_of",
    "max_absolute",
    "numerical_layer_parameter_gradients",
    "numerical_stack_input_gradient",
    "parse_script_module_names",
    "placement_is_supported",
    "relative_gradient_change",
    "relative_matrix_error",
    "sequence_slice",
    "stack_backward",
    "stack_forward",
    "stack_gradient_profile",
    "stack_loss",
    "stack_loss_gradient",
    "stack_stage_lines",
    "stack_study",
    "zero_attention",
    "zero_branches",
]
