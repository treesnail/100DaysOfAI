"""``regularization`` 的口径表（day095 / M8-D6）.

一次性把这一课的"名词表"写全：**两条归一化轴 / 两个相 / 四个技巧 / 七条性质 /
十条笔记 / 五条边界 / 四张公式 / 一张转发表 / 一张 PyTorch 对照表**。
全部是常量，因此可以被测试逐键检查。

```text
两条归一化轴  batch（沿批，BatchNorm）/ feature（沿特征，LayerNorm，day079）
两个相        train（用本批统计）/ eval（用 running 统计）
四个技巧      batchnorm（本包新写）/ dropout（转发 day081）/
              lr_decay（转发 day074/081）/ early_stop（转发 day081）
```

## 这一课的位置：把 day094 的链"装上旋钮"，而不是重写它

```text
day089  最小的零件（Dense / 激活 / 交叉熵）
day090  梯度怎么回传（链式法则 + 累加）
day092  拿到梯度后一步走多远（优化器）
day093  空间的权重共享（卷积）
day094  时间的权重共享（RNN / LSTM，BPTT）
day095  **训练技巧**：归一化 / 随机丢 / 学习率衰减 / 早停 + 训练日志可视化
```

因此本课的**不变量**是"不改 day094 的一行"：day094 的 ``sequence_models``
既是上游（提供循环单元、BPTT、压平契约），也是被实验的对象。
"""

from __future__ import annotations

from smart_research_agent.regularization.errors import RegularizationError

# --------------------------------------------------------------------------- #
# 闭合表 1：两条归一化轴
# --------------------------------------------------------------------------- #

AXIS_BATCH = "batch"
AXIS_FEATURE = "feature"

#: 两条归一化轴（顺序 = 沿批 → 沿特征）.
NORM_AXES: tuple[str, ...] = (AXIS_BATCH, AXIS_FEATURE)

#: 每条轴的一句话解释.
AXIS_DESCRIPTIONS: dict[str, str] = {
    AXIS_BATCH: "沿**批**归一化（BatchNorm）：固定一个特征，在 N 条样本上算均值与方差",
    AXIS_FEATURE: "沿**特征**归一化（LayerNorm，day079）：固定一条样本，在 F 个特征上算均值与方差",
}

#: 每条轴的一句话公式.
AXIS_FORMULAS: dict[str, str] = {
    AXIS_BATCH: "μ_j = (1/N)·Σ_n x_nj；σ²_j = (1/N)·Σ_n (x_nj − μ_j)²   （对每个特征 j）",
    AXIS_FEATURE: "μ_n = (1/F)·Σ_j x_nj；σ²_n = (1/F)·Σ_j (x_nj − μ_n)²   （对每条样本 n）",
}

#: 每条轴"最能暴露它的一个事实".
AXIS_SIGNATURES: dict[str, str] = {
    AXIS_BATCH: "批大小变成 1 时方差恒为 0 ⇒ 输出被压成常数（β）——这是它的签名",
    AXIS_FEATURE: "每条样本各自标准化 ⇒ 与批里有多少条样本完全无关",
}

if not (
    set(NORM_AXES) == set(AXIS_DESCRIPTIONS) == set(AXIS_FORMULAS) == set(AXIS_SIGNATURES)
):  # pragma: no cover - 导入期不变式
    raise RegularizationError(
        "两条归一化轴的四张表不一致：NORM_AXES / AXIS_DESCRIPTIONS / AXIS_FORMULAS / "
        "AXIS_SIGNATURES 必须逐键对齐，否则某条轴在报告里只有名字、没有它的签名。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 2：两个相
# --------------------------------------------------------------------------- #

PHASE_TRAIN = "train"
PHASE_EVAL = "eval"

#: 两个相（顺序 = 先跑训练、再跑推理）.
PHASES: tuple[str, ...] = (PHASE_TRAIN, PHASE_EVAL)

#: 每个相的一句话解释（**与 day081 的 dropout 逐字同源**）.
PHASE_DESCRIPTIONS: dict[str, str] = {
    PHASE_TRAIN: "训练相：BatchNorm 用**这一批自己的** μ/σ；Dropout 置零并放大",
    PHASE_EVAL: "推理相：BatchNorm 用 **running 统计量**；Dropout 恒等",
}

#: 每个相在 BatchNorm 里需要的统计量来自哪里.
PHASE_STATISTIC_SOURCES: dict[str, str] = {
    PHASE_TRAIN: "本批统计量（同时更新 running 统计量）",
    PHASE_EVAL: "running 统计量（**必须由调用方给出**，否则抛 PhaseError）",
}

if not (
    set(PHASES) == set(PHASE_DESCRIPTIONS) == set(PHASE_STATISTIC_SOURCES)
):  # pragma: no cover - 导入期不变式
    raise RegularizationError("两个相的三张表不一致：PHASES / 描述 / 统计量来源必须逐键对齐。")

# --------------------------------------------------------------------------- #
# 闭合表 3：四个技巧（含"谁实现了它"）
# --------------------------------------------------------------------------- #

TECHNIQUE_BATCHNORM = "batchnorm"
TECHNIQUE_DROPOUT = "dropout"
TECHNIQUE_LR_DECAY = "lr_decay"
TECHNIQUE_EARLY_STOP = "early_stop"

#: 四个技巧（顺序 = 本课的讲述顺序：先归一化、再随机丢、再调度、最后停）。
TECHNIQUES: tuple[str, ...] = (
    TECHNIQUE_BATCHNORM,
    TECHNIQUE_DROPOUT,
    TECHNIQUE_LR_DECAY,
    TECHNIQUE_EARLY_STOP,
)

#: 每个技巧的一句话解释.
TECHNIQUE_DESCRIPTIONS: dict[str, str] = {
    TECHNIQUE_BATCHNORM: "批量归一化：把每个特征在批内标准化，再把尺度交给可学的 γ/β",
    TECHNIQUE_DROPOUT: "随机丢：训练时按概率置零并放大 1/(1−p)，推理时恒等（inverted）",
    TECHNIQUE_LR_DECAY: "学习率衰减：让每一步比上一步小（本课用余弦退火）",
    TECHNIQUE_EARLY_STOP: "早停：连续 patience 步没有明显变好就建议停下",
}

#: 每个技巧**由谁实现**（**这张表是本课的"转发而不重写"纪律的可核对形式**）.
TECHNIQUE_SOURCES: dict[str, str] = {
    TECHNIQUE_BATCHNORM: "本包新建（normalization.batch_norm_forward / batch_norm_backward）",
    TECHNIQUE_DROPOUT: "day081 training_optim.dropout（dropout_forward / dropout_backward）",
    TECHNIQUE_LR_DECAY: "day074 math_foundations.optim.make_schedule（经 day081 controls 转发）",
    TECHNIQUE_EARLY_STOP: "day081 training_optim.controls.EarlyStopping",
}

if not (
    set(TECHNIQUES) == set(TECHNIQUE_DESCRIPTIONS) == set(TECHNIQUE_SOURCES)
):  # pragma: no cover - 导入期不变式
    raise RegularizationError(
        "四个技巧的三张表不一致：TECHNIQUES / TECHNIQUE_DESCRIPTIONS / TECHNIQUE_SOURCES "
        "必须逐键对齐——少一个键的技巧会静默地不知道'它由谁实现'。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 4：四张公式
# --------------------------------------------------------------------------- #

#: BatchNorm 的标准化那一行（**训练相**用本批统计量）.
BN_FORMULA = "x̂ = (x − μ_batch)/√(σ²_batch + ε)；y = γ ⊙ x̂ + β"

#: running 统计量的更新（滑窗平均；momentum 就是"跟得多快"）.
RUNNING_UPDATE_FORMULA = "μ_run ← (1 − m)·μ_run + m·μ_batch；σ²_run ← (1 − m)·σ²_run + m·σ²_batch"

#: BatchNorm 的反向（**训练相**：μ/σ 也是 x 的函数，因此有三项）.
BN_BACKWARD_FORMULA = (
    "dŷ = γ ⊙ dy；dx = (1/σ)·(dŷ − mean_batch(dŷ) − x̂ ⊙ mean_batch(dŷ ⊙ x̂))"
)

#: BatchNorm 的反向（**推理相**：μ/σ 是常数，因此只剩一项）.
BN_EVAL_BACKWARD_FORMULA = "dx = γ ⊙ dy / σ_run（推理相的统计量与 x 无关，没有那两项）"

#: 早停的判据（day081 的纪律：变好必须带一个 min_delta）.
EARLY_STOP_FORMULA = "变好 ⇔ loss < best_loss − min_delta；连续 patience 步没变好 ⇒ 停"

#: BatchNorm 的缺省 epsilon（**与 day079 的 ``encoder_decoder.DEFAULT_EPSILON`` 同值**——
#: 这是第 ② 条跨天对账能"逐位对得上"的前提）。
DEFAULT_EPSILON = 1e-5

#: running 统计量的缺省 momentum（0.1 = 每个 batch 占新统计量的 10%）。
DEFAULT_MOMENTUM = 0.1

# --------------------------------------------------------------------------- #
# 闭合表 5：七条性质
# --------------------------------------------------------------------------- #

PROPERTY_BATCHNORM_MATCHES_MANUAL = "batchnorm_matches_manual_standardization"
PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM = "batchnorm_equals_transposed_layer_norm"
PROPERTY_TWO_PHASES_DIFFER = "train_and_eval_phases_differ"
PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL = "batchnorm_backward_matches_numerical"
PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL = "eval_backward_matches_numerical"
PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO = "dropout_eval_matches_rate_zero"
PROPERTY_SPARKLINE_MATCHES_EXTREMES = "sparkline_matches_extremes"

#: 七条性质（顺序 = 从"前向对不对"到"日志画得对不对"）。
REGULARIZATION_PROPERTIES: tuple[str, ...] = (
    PROPERTY_BATCHNORM_MATCHES_MANUAL,
    PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM,
    PROPERTY_TWO_PHASES_DIFFER,
    PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO,
    PROPERTY_SPARKLINE_MATCHES_EXTREMES,
)

#: 每条性质在讲什么（一句话）。
PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_BATCHNORM_MATCHES_MANUAL: "batch_norm_forward 与手写的逐特征标准化逐位一致",
    PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM: "把批转置之后，BatchNorm 与 day079 的 LayerNorm 给出同一个矩阵",
    PROPERTY_TWO_PHASES_DIFFER: "同一个 batch 上，训练相与推理相的输出**必须不同**（否则两相是假的）",
    PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL: "训练相反向（三项公式）与数值差分一致",
    PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL: "推理相反向（一项公式）与数值差分一致",
    PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO: "dropout 的推理相与 rate=0 的训练相逐位相同（转发 day081）",
    PROPERTY_SPARKLINE_MATCHES_EXTREMES: "sparkline 的长度等于序列长度，且极值位置与 argmin / argmax 一致",
}

#: 每条性质"失败意味着什么"（**不通过时要去看哪里**）。
PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_BATCHNORM_MATCHES_MANUAL: "方差用了无偏（除以 N−1）而不是有偏（除以 N）——样本少时差得明显",
    PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM: "γ/β 的广播方向错了（按批广播写成了按特征广播）",
    PROPERTY_TWO_PHASES_DIFFER: "推理相也在用本批统计量（即两相被写成了同一件事）",
    PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL: "三项里漏了 (1/σ) 或第二个 mean_batch 项",
    PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL: "推理相的反向被写成了训练相的三项版本（多了两项）",
    PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO: "推理相也在置零（即把 inverted dropout 的缩放放错了侧）",
    PROPERTY_SPARKLINE_MATCHES_EXTREMES: "档位映射把最大值映成了最低档（方向写反）",
}

if not (
    set(REGULARIZATION_PROPERTIES) == set(PROPERTY_DESCRIPTIONS) == set(PROPERTY_FAILURE)
):  # pragma: no cover
    raise RegularizationError("七条性质的三张表不一致：名单 / 说明 / 失败意味着什么必须逐键对齐。")

# --------------------------------------------------------------------------- #
# 闭合表 6：十条笔记 / 五条边界 / 一张 PyTorch 对照表
# --------------------------------------------------------------------------- #

NOTES_ORDER: tuple[str, ...] = (
    "axis",
    "batch_statistics",
    "running_stats",
    "two_phases",
    "batch_size_one",
    "epsilon_role",
    "gamma_beta",
    "backward_three_terms",
    "inverted_dropout",
    "early_stop_delta",
)

#: 十条笔记（键 -> 一句话）.
REGULARIZATION_NOTES: dict[str, str] = {
    "axis": "归一化的第一条问题是**沿哪条轴取平均**：BatchNorm 沿批、LayerNorm 沿特征。",
    "batch_statistics": "训练相用**这一批自己的** μ/σ（有偏方差，分母是 N 不是 N−1）。",
    "running_stats": "推理相用 running 统计量；momentum 不是「遗忘率」，而是「每个批占多少」。",
    "two_phases": "两相写反**不会报错**：它只让推理时输出随批抖动，而所有形状检查都会通过。",
    "batch_size_one": "批大小为 1 时，每个特征的方差恒为 0 ⇒ 输出被压成常数 β（ε 只是不让它除零）。",
    "epsilon_role": "ε 在分母里，作用是「别让 0 方差炸掉」，不是「提高数值精度」。",
    "gamma_beta": "γ/β 是可学的：把它们设成 (σ, μ) 就能**撤销**标准化（恒等映射是它的一个特例）。",
    "backward_three_terms": "训练相的反向有三项（μ/σ 也依赖 x）；推理相只剩一项。",
    "inverted_dropout": "inverted dropout 把缩放放在**训练侧**，因此推理路径与「没有 dropout」逐位相同。",
    "early_stop_delta": "min_delta = 0 时浮点噪声也会被当成「进展」，早停因此永不触发。",
}

#: 五条边界（**这一课明确不承诺的事**）.
REGULARIZATION_BOUNDARIES: tuple[str, ...] = (
    "只做 BatchNorm 与「转发 day081 的三个旋钮」：不做 GroupNorm / InstanceNorm / 权重标准化。",
    "不做分布式训练：running 统计量不跨卡同步（那是工程问题，不是这一课的数学）。",
    "不做 GPU / 向量化：纯 Python 的逐元素实现，为的是可读与可对账，不是速度。",
    "只在 day094 的序列分类头上做实验：**不改动 sequence_models 的任何一行**。",
    "不新增第三方依赖：全部纯标准库实现，不 import torch / numpy。",
)

#: 纯 Python ↔ PyTorch 对照表（**只核对语义，本仓库不安装也不调用 torch**）.
TORCH_COUNTERPARTS: dict[str, str] = {
    "batchnorm": "torch.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1)",
    "running_stats": "BatchNorm 的 running_mean / running_var 两个 buffer",
    "track_running_stats": "track_running_stats=True 时才会维护 running 统计量",
    "phases": "model.train() / model.eval() 切换相（BatchNorm 与 Dropout 都看它）",
    "dropout": "torch.nn.Dropout(p)（inverted；p=0 时逐位恒等）",
    "lr_schedule": "torch.optim.lr_scheduler.CosineAnnealingLR / LambdaLR",
    "early_stop": "没有内置实现——它属于训练回路（本仓库 day081 的 EarlyStopping）",
    "grad_clip": "torch.nn.utils.clip_grad_norm_（本仓库 day094 的 check_gradient_norm 是它的守卫）",
    "batchnorm_backward": "loss.backward() 给定 y = γx̂+β 之后由 autograd 求出（本课手写同一条链）",
    "layer_norm": "torch.nn.LayerNorm(normalized_shape)（本仓库 day079 的 layer_norm 是它的最小实现）",
}

#: 这一课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTIES = REGULARIZATION_PROPERTIES

__all__ = [
    "AXIS_BATCH",
    "AXIS_DESCRIPTIONS",
    "AXIS_FEATURE",
    "AXIS_FORMULAS",
    "AXIS_SIGNATURES",
    "BN_BACKWARD_FORMULA",
    "BN_EVAL_BACKWARD_FORMULA",
    "BN_FORMULA",
    "DEFAULT_EPSILON",
    "DEFAULT_MOMENTUM",
    "EARLY_STOP_FORMULA",
    "NOTES_ORDER",
    "NORM_AXES",
    "PHASES",
    "PHASE_DESCRIPTIONS",
    "PHASE_EVAL",
    "PHASE_STATISTIC_SOURCES",
    "PHASE_TRAIN",
    "PROPERTIES",
    "PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL",
    "PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM",
    "PROPERTY_BATCHNORM_MATCHES_MANUAL",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO",
    "PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL",
    "PROPERTY_FAILURE",
    "PROPERTY_SPARKLINE_MATCHES_EXTREMES",
    "PROPERTY_TWO_PHASES_DIFFER",
    "REGULARIZATION_BOUNDARIES",
    "REGULARIZATION_NOTES",
    "REGULARIZATION_PROPERTIES",
    "RUNNING_UPDATE_FORMULA",
    "TECHNIQUES",
    "TECHNIQUE_BATCHNORM",
    "TECHNIQUE_DESCRIPTIONS",
    "TECHNIQUE_DROPOUT",
    "TECHNIQUE_EARLY_STOP",
    "TECHNIQUE_LR_DECAY",
    "TECHNIQUE_SOURCES",
    "TORCH_COUNTERPARTS",
]
