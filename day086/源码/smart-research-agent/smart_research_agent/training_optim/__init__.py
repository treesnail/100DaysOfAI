"""``training_optim``：把一条链更新 T 次（M7-D6 / day081）.

day080 交出**一条链**，并留下两件东西：

```text
一份逐层读数（第 5 章）            它今天用来解释"初始化把信号放大了多少"
一份 day079 的悬案：post-LN 要不要学习率预热？   它今天用一段真训练回答
```

## 一、今天新增的代码：**每一层各一行**

```python
# 前向：账算完之后，把这一层的**输出**过一次 dropout
current, mask_i, scale_i = dropout_forward(forward.output, rate=..., seed=...)

# 反向：**先穿过掩码，再反传**
current = dropout_backward(current, mask_i, scale_i)
layer_grads = encoder_block_backward(layer.block, layer.params, current)
```

其余部分原封不动地调用 day079/080 的 ``block_attention`` / ``encoder_block`` /
``encoder_block_backward``——因为"加一个 dropout"不该顺手把九个梯度公式重写一遍。

## 二、六个旋钮，四组实验

```text
初始化    每一层权重的幅度        day080 的逐层增益就是它的读数
Dropout   训练时丢掉多少单元      训练损失与推理损失之差就是它的读数
学习率    每一步走多远            太大就发散（DivergenceError）
调度      学习率随步数怎么变      热身能不能救下 post-LN（day079 的问题）
裁剪      一步最长能有多长        一次坏梯度不该把参数甩出去
早停      什么时候该放弃          判据是"连续多少步没变好"
```

四组实验（初始化 / 调度 / 裁剪 / Dropout）**一次只改一个旋钮**，
共用同一颗种子、同一个目标与同一份配置。

## 三、六条性质里有两条是"护栏"

```text
训练的确定性                 同一份配置跑两次，每一步**逐位**相同
dropout = 0 ⇒ 与 day080 的链逐位一致   ——"我们训的是同一个模型"的唯一证据
E[掩码 × 缩放] = 1           inverted dropout 在期望上不改变这一层
裁剪只改长度、不改方向         整体范数裁剪的定义（余弦 = 1.0）
调度与 day074 逐位一致        口径只有一处
单调曲线上的早停永不触发       它一旦触发就说明 patience 太小
```

第二条最要紧：如果 ``dropout = 0`` 时两条路径不**逐位**相同，
那么这一课训练的根本不是 day080 那个模型，而两条损失曲线也就无从比较。

## 四、六个模块

```text
errors.py     五族失败：形状 / 参数 / 数值 / 梯度 / **发散**（新族：修法是改超参）
types.py      五个旋钮 + 两个记录口径（loss 训练相 / eval_loss 推理相）、六条性质、三条边界
dropout.py    掩码（确定性 LCG）+ 两个相 + 一行反向；**掩码固定之后梯度可校验**
init.py       四种方案 + 理论标准差 + **实测/理论比值**（这是"公式有没有写错"的唯一读数）
controls.py   调度（转发 day074）、裁剪（转发 day074）+ 读数、早停（新增）
train.py      带 dropout 的链式前向/反向、参数压平、训练循环
study.py      四组实验（初始化 / 调度 / 裁剪 / Dropout）
```

## 五、四条纪律

1. **转发而不是重写**：调度与裁剪的公式在 day074 已经推过、测过；重写一遍的代价是
   两处口径会慢慢分开。
2. **两个损失口径都要给**：``loss``（带 dropout）与 ``eval_loss``（不带）——
   只看训练损失会让 dropout 看起来有害。
3. **判据必须是事先写下来的数**：「发散」是"损失涨到初始值的 ``divergence_factor`` 倍"，
   而不是"看起来不太对"。
4. **一次只改一个旋钮**：四组实验共用同一颗种子、同一个目标与同一份配置。

## 六、与既有包的接缝

- **上游**：``transformer_stack``（day080 的链、逐层读数与两种账）、
  ``encoder_decoder``（day079 的 ``block_attention`` / ``encoder_block`` /
  ``encoder_block_backward``）、``math_foundations``（day073 的 ``uniforms`` 与
  ``float`` 口径、day074 的四种调度 / 两种裁剪 / 三个优化器）；
- **脚下**：``config`` **没有**新增配置项——五个旋钮全部进 ``TrainingConfig``，
  它是"这一次训练"的判据，而不是服务级的默认值；
- **下游**：day082（变体架构）会看到 BERT/GPT 就是"取这摞块的哪几层"，
  而它们的训练配方（学习率、热身、dropout）与今天这张表直接可比；
  day087（高效推理与量化）会用到今天"训练相与推理相"的区分——
  量化只作用于推理相，而 dropout 只在训练相；
  day052/058（微调与版本管理）会看到今天的"最好的一步"与"该不该回滚"是同一件事。
"""

from __future__ import annotations

from smart_research_agent.training_optim.controls import (
    EarlyStopping,
    build_optimizer,
    clip_gradients,
    direction_cosine,
    learning_rate_at,
)
from smart_research_agent.training_optim.dropout import (
    DEFAULT_DROPOUT_RATE,
    PHASES,
    PHASE_DESCRIPTIONS,
    PHASE_EVAL,
    PHASE_TRAIN,
    dropout_backward,
    dropout_forward,
    dropout_mask,
    expected_keep,
    kept_fraction,
    scale_of,
)
from smart_research_agent.training_optim.errors import (
    FAMILY_OUTCOMES,
    DivergenceError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
    TrainingError,
)
from smart_research_agent.training_optim.init import (
    INIT_SCHEMES,
    SCHEME_DESCRIPTIONS,
    SCHEME_KAIMING,
    SCHEME_NORMAL,
    SCHEME_UNIFORM,
    SCHEME_XAVIER,
    STD_RATIO_HIGH,
    STD_RATIO_LOW,
    expected_std,
    init_matrix,
    initialization_is_consistent,
    initialization_rows,
    initialize_block,
    initialize_parameters,
    matrix_std_ratio,
    measure_std,
)
from smart_research_agent.training_optim.study import (
    CONTROL_CLIP,
    CONTROL_DROPOUT,
    CONTROL_GROUPS,
    CONTROL_GROUP_DESCRIPTIONS,
    CONTROL_INIT,
    CONTROL_SCHEDULE,
    StudyRow,
    TrainingStudy,
    compare_clipping,
    compare_dropout,
    compare_initializations,
    compare_schedules,
    gain_profile_of,
    stack_of,
    training_study,
)
from smart_research_agent.training_optim.train import (
    LAYER_MASK_STRIDE,
    STEP_MASK_STRIDE,
    StepCache,
    curve_summary,
    flatten_block_gradients,
    forward_output_at,
    mask_seed,
    numerical_training_input_gradient,
    plain_stack_output,
    step_loss,
    train,
    training_backward,
    training_forward,
)
from smart_research_agent.training_optim.types import (
    CONTROLS,
    CONTROL_DESCRIPTIONS,
    DEFAULT_DIVERGENCE_FACTOR,
    PROPERTY_CLIP_PRESERVES_DIRECTION,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_DROPOUT_EXPECTATION,
    PROPERTY_EARLY_STOP_ON_MONOTONE,
    PROPERTY_RATE_ZERO_MATCHES_STACK,
    PROPERTY_SCHEDULE_MATCHES_BACKBONE,
    TRAINING_NOTES,
    TRAINING_PROPERTIES,
    ClipReport,
    EarlyStopReport,
    EpochRecord,
    TrainingConfig,
    TrainingCurve,
    check_no_divergence,
    describe_controls,
    record_count,
    shape_size,
    steps_of,
)

__all__ = [
    "CONTROLS",
    "CONTROL_CLIP",
    "CONTROL_DESCRIPTIONS",
    "CONTROL_DROPOUT",
    "CONTROL_GROUPS",
    "CONTROL_GROUP_DESCRIPTIONS",
    "CONTROL_INIT",
    "CONTROL_SCHEDULE",
    "DEFAULT_DIVERGENCE_FACTOR",
    "DEFAULT_DROPOUT_RATE",
    "FAMILY_OUTCOMES",
    "INIT_SCHEMES",
    "LAYER_MASK_STRIDE",
    "PHASES",
    "PHASE_DESCRIPTIONS",
    "PHASE_EVAL",
    "PHASE_TRAIN",
    "PROPERTY_CLIP_PRESERVES_DIRECTION",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_DROPOUT_EXPECTATION",
    "PROPERTY_EARLY_STOP_ON_MONOTONE",
    "PROPERTY_RATE_ZERO_MATCHES_STACK",
    "PROPERTY_SCHEDULE_MATCHES_BACKBONE",
    "SCHEME_DESCRIPTIONS",
    "SCHEME_KAIMING",
    "SCHEME_NORMAL",
    "SCHEME_UNIFORM",
    "SCHEME_XAVIER",
    "STD_RATIO_HIGH",
    "STD_RATIO_LOW",
    "STEP_MASK_STRIDE",
    "TRAINING_NOTES",
    "TRAINING_PROPERTIES",
    "ClipReport",
    "DivergenceError",
    "EarlyStopReport",
    "EarlyStopping",
    "EpochRecord",
    "GradientError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "StepCache",
    "StudyRow",
    "TrainingConfig",
    "TrainingCurve",
    "TrainingError",
    "TrainingStudy",
    "build_optimizer",
    "check_no_divergence",
    "clip_gradients",
    "compare_clipping",
    "compare_dropout",
    "compare_initializations",
    "compare_schedules",
    "curve_summary",
    "describe_controls",
    "direction_cosine",
    "dropout_backward",
    "dropout_forward",
    "dropout_mask",
    "expected_keep",
    "expected_std",
    "flatten_block_gradients",
    "forward_output_at",
    "gain_profile_of",
    "init_matrix",
    "initialization_is_consistent",
    "initialization_rows",
    "initialize_block",
    "initialize_parameters",
    "kept_fraction",
    "learning_rate_at",
    "mask_seed",
    "matrix_std_ratio",
    "measure_std",
    "numerical_training_input_gradient",
    "plain_stack_output",
    "record_count",
    "scale_of",
    "shape_size",
    "stack_of",
    "step_loss",
    "steps_of",
    "train",
    "training_backward",
    "training_forward",
    "training_study",
]
