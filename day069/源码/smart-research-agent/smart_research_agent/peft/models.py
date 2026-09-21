"""LoRA 参考模型：把 day050 的训练循环**原封不动**地用在只训 2.787% 参数的场景上（M5-D3）.

day050 的结尾留了一句话：**"label mask 完全不变、``plan_training`` 的步数算术
完全不变、``SFTTrainingArgs`` 加一个 ``peft_config`` 就能接上 PEFT"**。本模块
把那句话变成可以运行的事实。

``LoRAReferenceModel`` 与 day050 的 ``ReferenceSFTModel`` **接口完全一致**：

===========================  ==========================================
``accumulate(batch)``        前向 + 反向，返回 ``(loss 之和, 监督 token 数)``
``apply_update(lr)``         按 ``1/N`` 缩放累加梯度并做一次 SGD
``zero_grad()``              丢弃未应用的梯度
``evaluate(batches)``        只前向、不污染梯度缓冲
``state_dict()``             可序列化状态（**合并后的**权重）
``vocab_size`` / ``updates`` 与基座一致的两条属性
===========================  ==========================================

因此 ``SFTTrainer(model=LoRAReferenceModel(...), tokenizer, args)`` **一行不改**
就能跑起来。这不是巧合，而是刻意的边界设计：**LoRA 改变的只有"哪些参数有
梯度"，训练循环里的渲染、编码、mask、padding、梯度累积、学习率调度、
评估、落盘全都与它无关。** 把这条边界划清楚，"换微调方法"就不再意味着
"重写训练脚本"。

与 day050 的第二个衔接点是**分母**。day050 花了整整一节说明"loss 的分母只数
监督 token"；在 LoRA 这里同一条纪律出现在梯度层面：``apply_update`` 的缩放
因子是 **自上次更新以来累积的位置数**，不是批数、也不是参数量。用错分母
会让有效学习率被悄悄缩放——而日志里的 ``learning_rate`` 完全看不出来。

三处刻意的取舍：

1. **基座在构造时冻结一份快照**（``base.state_dict()``），而不是持有基座对象
   的引用。这样"基座是否被改动过"不取决于调用方的自觉；
2. **要求 ``lora_dropout = 0``**：参考模型的输入是 one-hot（embedding 查表），
   逐元素 dropout 在这里退化为"以概率 p 整条适配器分支被置零"，会让 loss
   曲线不可复现。dropout 机制由 ``LoRALinear`` 的专门用例覆盖；
3. **要 ``target_modules`` 命中参考模型唯一的那个矩阵**（预设名 ``bigram``）。
   本课参考模型只有一个 ``V × V`` 的 bigram 矩阵，模块名 ``weight``；这一条
   校验保证"你以为适配了哪里"与"实际适配了哪里"一致——真实模型上漏配一个
   投影名只会表现为"效果不如预期"，不会报错。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from smart_research_agent.peft.config import (
    REFERENCE_MODULE_NAME,
    LoRAConfig,
    PEFTConfigError,
)
from smart_research_agent.peft.layers import LoRALinear
from smart_research_agent.sft.encoding import IGNORE_INDEX, Batch
from smart_research_agent.sft.loss import SFTLossError, log_softmax, softmax
from smart_research_agent.sft.reference_model import ModelState, ReferenceSFTModel

#: 参考模型的 LoRA 默认秩：``2·r·V / (V² + V)`` 在本课程数据集（V=557）上
#: 是 **2.787%**——足以说明"参数高效"这四个字的量级。
REFERENCE_LORA_RANK = 8

#: 参考模型的 LoRA 默认 alpha（``alpha = 2r`` 的社区惯例）.
REFERENCE_LORA_ALPHA = 16


def default_reference_lora_config(**overrides: Any) -> LoRAConfig:
    """参考模型上可直接使用的 LoRA 配置（``target_modules="bigram"``）.

    单独提供这个工厂而不是改 ``LoRAConfig`` 的缺省值：``attention`` 是真实
    模型的常见选择，把缺省改成 ``bigram`` 会让真实场景的配置变别扭；而
    "参考模型该用哪个预设"这件事应该显式说出来。
    """
    base = LoRAConfig(
        r=REFERENCE_LORA_RANK,
        lora_alpha=REFERENCE_LORA_ALPHA,
        lora_dropout=0.0,
        target_modules="bigram",
    )
    return base.with_overrides(**overrides) if overrides else base


def reference_lora_accounting(vocab_size: int, config: LoRAConfig) -> dict[str, Any]:
    """参考模型（``V × V`` bigram）上的参数量核算——**纯算术，不构造模型**。

    存在的理由是一个真实的接口断层：``peft.targets.plan_lora`` 面向的是
    **解码器规格**（``q_proj`` / ``gate_proj`` … 七个投影），而参考模型只有
    一个叫 ``weight`` 的 bigram 矩阵。把参考模型的规格硬塞进 ``plan_lora``
    会命中"目标模块未匹配"的校验并报错——本课实现时真的踩到过（``GET
    /finetune/lora/defaults`` 因此返回 500，而单元测试全绿：因为测试只
    覆盖了 helper，没有覆盖"端点的组合方式"）。

    所以参考模型的账单独算，而且只有两行：

    .. code-block:: text

        基座   = V² + V        （W 是 V×V，b 是长度 V）
        适配器 = r × (V + V)   （A 是 r×V，B 是 V×r；bias="none" 时没有别的）

    ``V=557`` / ``r=8`` 时是 8 912 / 319 718 = **2.7875%**。
    """
    if vocab_size < 2:
        raise PEFTConfigError(f"vocab_size 至少为 2，收到 {vocab_size}")
    config.validate()
    if config.resolved_targets != (REFERENCE_MODULE_NAME,):
        raise PEFTConfigError(
            f"参考模型只有一个 bigram 矩阵（模块名 {REFERENCE_MODULE_NAME!r}），"
            f"收到 target_modules={list(config.resolved_targets)}"
        )
    frozen = vocab_size * vocab_size + vocab_size
    trainable = config.adapter_parameter_count(vocab_size, vocab_size)
    total = frozen + trainable
    return {
        "vocab_size": vocab_size,
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "scaling": config.scaling,
        "targets": list(config.resolved_targets),
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "total_parameters": total,
        "trainable_ratio": trainable / total,
        "rank_upper_bound": config.rank_upper_bound(vocab_size, vocab_size),
    }


class LoRAReferenceModel:
    """在冻结的 ``ReferenceSFTModel`` 上挂一个 LoRA 适配器（真实梯度下降）.

    ``ΔW = (alpha/r) · B @ A`` 作用在基座唯一的 ``V × V`` 矩阵上。前向时
    基座的一行与适配器的一行相加（``O(r·V)``），反向时只累加 ``∂L/∂A`` 与
    ``∂L/∂B``——**基座没有任何梯度**。这是"参数高效"在代码上的全部体现。
    """

    def __init__(
        self,
        base: ReferenceSFTModel,
        config: LoRAConfig,
        *,
        seed: int = 42,
        std_scale: float = 1.0,
    ):
        config.validate()
        targets = config.resolved_targets
        if targets != (REFERENCE_MODULE_NAME,):
            raise PEFTConfigError(
                f"参考模型只有一个 bigram 矩阵（模块名 {REFERENCE_MODULE_NAME!r}），"
                f"收到 target_modules={list(targets)}。请使用 target_modules='bigram' "
                "或 default_reference_lora_config()。"
            )
        if config.lora_dropout != 0.0:
            raise PEFTConfigError(
                "参考模型要求 lora_dropout = 0：它的输入是 one-hot（embedding 查表），"
                "逐元素 dropout 在这里等价于「以概率 p 整条适配器分支被置零」，"
                "会让 loss 曲线不可复现。dropout 机制由 LoRALinear 的专门用例覆盖。"
            )
        if config.bias != "none":
            raise PEFTConfigError(
                f"参考模型不支持 bias={config.bias!r}：它只训练 A / B 两个矩阵，"
                "偏置视为冻结参数（与 peft 的 bias='none' 语义一致）。"
                "允许 bias！='none' 会让「可训练参数 = 2·r·V」这条手算公式失效。"
            )
        self._config = config
        self._base_state = base.state_dict()
        self._layer = LoRALinear(
            self._base_state.weights, config, seed=seed, std_scale=std_scale
        )
        self._bias = list(self._base_state.bias)
        self._vocab_size = self._base_state.vocab_size

    # ------------------------------------------------------------------ 属性
    @property
    def config(self) -> LoRAConfig:
        """本模型使用的 LoRA 配置."""
        return self._config

    @property
    def layer(self) -> LoRALinear:
        """适配器所在的那个层（测试与诊断直接用它）."""
        return self._layer

    @property
    def vocab_size(self) -> int:
        """词表大小（与基座一致）."""
        return self._vocab_size

    @property
    def updates(self) -> int:
        """已完成的参数更新次数（= 适配器的优化器步数）."""
        return self._layer.updates

    @property
    def base_updates(self) -> int:
        """基座在此前已经历过的更新次数（合并后的模型沿用它）."""
        return self._base_state.updates

    @property
    def uniform_loss(self) -> float:
        """完全瞎猜时的参考 loss：``ln(V)``（与 day050 同一条基线）."""
        return math.log(self._vocab_size)

    @property
    def trainable_parameters(self) -> int:
        """可训练参数个数（只有 ``A`` 与 ``B``）."""
        return self._layer.trainable_parameters

    @property
    def frozen_parameters(self) -> int:
        """冻结参数个数（基座权重矩阵）. 基座偏置在本课也是冻结的."""
        return self._layer.frozen_parameters + self._vocab_size

    @property
    def total_parameters(self) -> int:
        """模型总参数（冻结 + 可训练）."""
        return self.frozen_parameters + self.trainable_parameters

    @property
    def trainable_ratio(self) -> float:
        """可训练参数占比（= 本课要讲的那个"百分之几"）."""
        return self.trainable_parameters / self.total_parameters

    # -------------------------------------------------------------- 前向 / 梯度
    def logits(self, context_token: int) -> list[float]:
        """给定上下文 token 的 logits（基座一行 + 适配器一行 + 偏置）.

        用的是 ``LoRALinear.forward_context``（**行约定**），而不是 ``forward_onehot``
        （列约定）：基座模型的 ``logits(ctx)`` 定义为 ``W[ctx] + b``，即"每个
        上下文 token 用一行权重"，适配器必须叠加在同一行上。

        由此得到本课最重要的一条不变式：**``ΔW = 0`` 时（刚挂上适配器、
        还没训练），本方法与 ``base.logits(ctx)`` 必须逐位相同。** 这条断言
        在测试里对全部词表下标都跑一遍——它守的是"行/列约定没有混用"，
        而这类错误**不会让 loss 曲线变得可疑**（本课实现时真的踩到过，
        见 ``LoRALinear.forward_context`` 的说明）。
        """
        if not 0 <= context_token < self._vocab_size:
            raise SFTLossError(
                f"上下文 token {context_token} 落在词表范围 [0, {self._vocab_size}) 之外"
            )
        row = self._layer.forward_context(context_token)
        bias = self._bias
        return [row[v] + bias[v] for v in range(self._vocab_size)]

    def predict_next(self, context_token: int) -> list[float]:
        """给定上下文 token 的下一 token 概率分布."""
        return softmax(self.logits(context_token))

    def accumulate(self, batch: Batch) -> tuple[float, int]:
        """前向 + 反向，把 ``∂L/∂A`` / ``∂L/∂B`` 累加进缓冲.

        对每条样本、每个位置 ``t >= 1``：上下文是 ``input_ids[t-1]``、目标写在
        ``labels[t]``；``labels[t] == -100`` 的位置完全跳过。**这一段与 day050
        ``ReferenceSFTModel.accumulate`` 逐行同构**——LoRA 不进这里。
        """
        loss_sum = 0.0
        count = 0
        for ids, labels in zip(batch.input_ids, batch.labels):
            for position in range(1, len(ids)):
                target = labels[position]
                if target == IGNORE_INDEX:
                    continue
                if not 0 <= target < self._vocab_size:
                    raise SFTLossError(
                        f"标签 {target} 落在词表范围 [0, {self._vocab_size}) 之外"
                    )
                context = ids[position - 1]
                log_probs = log_softmax(self.logits(context))
                loss_sum += -log_probs[target]
                count += 1
                # ∂L/∂logits = softmax(logits) - onehot(target)，此处**不除 N**：
                # N 要等整个累积窗口数完才确定（见 LoRALinear.apply_update）
                gradient = [
                    math.exp(log_probs[v]) - (1.0 if v == target else 0.0)
                    for v in range(self._vocab_size)
                ]
                self._layer.backward_context(gradient)
        if count == 0:
            raise SFTLossError(
                "本批没有任何监督位置：请检查 labels 是否全被 ignore_index 屏蔽"
            )
        return loss_sum, count

    def apply_update(self, learning_rate: float) -> None:
        """按 ``1/N`` 缩放累加梯度并做一次 SGD（只更新 ``A`` 与 ``B``）."""
        self._layer.apply_update(learning_rate)

    def zero_grad(self) -> None:
        """丢弃未应用的累积梯度（warmup 第 0 步的学习率为 0 时用）."""
        self._layer.zero_grad()

    def step(self, batch: Batch, *, learning_rate: float) -> tuple[float, int]:
        """``accumulate`` + ``apply_update`` 的便捷组合（单批更新）."""
        loss_sum, count = self.accumulate(batch)
        self.apply_update(learning_rate)
        return loss_sum, count

    # ------------------------------------------------------------------ 评估
    def evaluate(self, batches: Iterable[Batch]) -> tuple[float, int]:
        """只前向、不更新，返回 ``(平均 loss, 监督 token 数)``.

        与 day050 一样，评估前记下梯度缓冲、评估后恢复——**评估不能污染
        梯度**。LoRA 让这件事更值得做：适配器的梯度缓冲小（``2·r·V`` 个数），
        但一次污染就足以让某个窗口的更新被评估数据带偏。
        """
        snapshot = self._layer.gradient_snapshot()
        loss_sum = 0.0
        count = 0
        for batch in batches:
            batch_loss, batch_count = self.accumulate(batch)
            loss_sum += batch_loss
            count += batch_count
        self._layer.restore_gradients(snapshot)
        if count == 0:
            raise SFTLossError("评估集没有任何监督位置，loss 无定义")
        return loss_sum / count, count

    # --------------------------------------------------------------- 结构信息
    def delta_is_zero(self) -> bool:
        """增量是否为 0（``gaussian`` 初始化下、第一次更新之前恒为 True）."""
        return self._layer.delta_is_zero()

    def describe(self) -> dict[str, float | int | str]:
        """本模型的规模画像（参数量、占比、缩放、秩上界）.

        ``trainable_ratio`` 是这一课的核心数字：参考模型上 ``r=8`` 时是
        **2.787%**（8912 / 319718）。
        """
        return {
            "vocab_size": self._vocab_size,
            "r": self._config.r,
            "lora_alpha": self._config.lora_alpha,
            "scaling": self._config.scaling,
            "targets": list(self._config.resolved_targets),
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "total_parameters": self.total_parameters,
            "trainable_ratio": self.trainable_ratio,
            "rank_upper_bound": self._config.rank_upper_bound(
                self._vocab_size, self._vocab_size
            ),
            "delta_max_abs": self._layer.describe()["max_abs"],
            "updates": self.updates,
        }

    # --------------------------------------------------------------- 序列化
    def merged_weights(self) -> list[list[float]]:
        """按上下文合并出的权重矩阵 ``M[i][j] = W[i][j] + ΔW[j][i]``.

        **它是密集合并矩阵 ``W + ΔW`` 的转置**，这不是笔误：基座模型的
        ``logits(ctx) = W[ctx]`` 等价于把 ``Wᵀ`` 当作这一层的权重，而
        ``(Wᵀ + s·B@A)·e_ctx = W[ctx] + s·B@A[:, ctx]``——"基座取行、
        增量取列"就来自这里。两条约定各自的合并矩阵互为转置，落盘时
        必须选对（落错了不会报错，只会让输出全错）。

        逐行合并而不是显式算 ``B @ A`` 再整体相加：``M`` 的第 ``i`` 行就是
        ``W[i] + delta_for_context(i)``，因此合并一次只要 ``O(V·r·V)``，
        而且**不需要把整个 ``ΔW`` 物化**——这与真实 LoRA 推理时的做法一致。
        """
        return self._layer.merged_by_context()

    def merge(self) -> ReferenceSFTModel:
        """把适配器合并进基座，返回一个**普通**的 ``ReferenceSFTModel``.

        合并后适配器消失、模型结构与基座完全一致——这是 LoRA 部署路径的
        第一步（day052 会把它做成落盘与验证流程）。``updates`` 沿用的是
        **基座**的更新次数，因为合并本身不是一次参数更新。
        """
        state = ModelState(
            vocab_size=self._vocab_size,
            weights=tuple(tuple(row) for row in self.merged_weights()),
            bias=tuple(self._bias),
            updates=self._base_state.updates,
        )
        return ReferenceSFTModel.from_state(state)

    def state_dict(self) -> ModelState:
        """**合并后**的状态（这样 day050 的通用检查点层可以原样复用）.

        代价是每次落盘都要物化一次 ``V × V`` 的合并权重。真实场景里不会
        这么干——adapter 的落盘只需要 ``A`` / ``B`` 两个小矩阵（``2·r·V`` 个数），
        这正是 day052 要实现的 adapter 检查点。本课在这里选择"兼容优先"，
        让 ``save_checkpoint`` 一行不改就能工作。
        """
        return self.merge().state_dict()

    def load_state(self, state: ModelState) -> None:
        """载入状态：**只接受与合并权重一致的状态**，否则抛错.

        这条限制是刻意的：把一个"随机训练过"的基座状态灌进一个带适配器的
        模型，会让"增量是相对哪个基座算的"失去意义。LoRA 的恢复路径是
        ``load_adapter_state``，不是 ``load_state``。
        """
        if state.vocab_size != self._vocab_size:
            raise PEFTConfigError(
                f"词表大小不一致：模型 {self._vocab_size} vs 状态 {state.vocab_size}"
            )
        if len(state.bias) != self._vocab_size:
            raise PEFTConfigError(
                f"偏置长度必须等于词表大小：期望 {self._vocab_size}，收到 {len(state.bias)}"
            )
        merged = tuple(tuple(row) for row in self.merged_weights())
        if tuple(tuple(row) for row in state.weights) != merged:
            raise PEFTConfigError(
                "load_state 只接受与当前合并权重一致的状态：LoRA 的恢复应当用 "
                "load_adapter_state（适配器状态），而不是灌入一份新的基座权重"
            )
        self._bias = list(state.bias)

    def adapter_state_dict(self) -> dict[str, Any]:
        """适配器状态（``A`` / ``B`` / 更新次数 / 配置）——真正该落盘的东西."""
        payload = dict(self._layer.adapter_state())
        payload["config"] = self._config.to_dict()
        payload["base_parameters"] = self.frozen_parameters
        payload["base_updates"] = self._base_state.updates
        return payload

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        """载入适配器状态（形状或缩放不一致直接拒绝）."""
        config_payload = state.get("config")
        if isinstance(config_payload, dict):
            incoming = LoRAConfig.from_dict(config_payload)
            if incoming.r != self._config.r or incoming.lora_alpha != self._config.lora_alpha:
                raise PEFTConfigError(
                    "适配器配置与模型不一致："
                    f"r={incoming.r}/alpha={incoming.lora_alpha} vs "
                    f"r={self._config.r}/alpha={self._config.lora_alpha}"
                )
        self._layer.load_adapter_state(state)

    @classmethod
    def from_base(
        cls,
        base: ReferenceSFTModel,
        config: LoRAConfig,
        *,
        seed: int = 42,
    ) -> LoRAReferenceModel:
        """便捷构造入口（语义与 ``__init__`` 相同）."""
        return cls(base, config, seed=seed)


def train_lora_reference(
    base: ReferenceSFTModel,
    batches: Sequence[Batch],
    *,
    config: LoRAConfig,
    learning_rate: float,
    epochs: int = 1,
    seed: int = 42,
    evaluate_batches: Sequence[Batch] = (),
) -> tuple[LoRAReferenceModel, list[tuple[int, float]], float | None]:
    """在给定批次上做一次最小 LoRA 训练，返回 ``(模型, 逐步 loss, 评估 loss)``.

    这个函数刻意做得很小——**它不是一个训练器**（完整的 LoRA 训练器在
    day052），只是一条让"LoRA 真的能学"可被观察的最短路径：每个 epoch
    顺序过一遍批次，每个批次做一次 ``accumulate + apply_update``。loss 的
    起点与 day050 的参考模型在同一位置（都接近 ``ln(V)``，因为
    ``ΔW = 0``），因此两条曲线可以直接对照。

    ``evaluate_batches`` 非空时在训练结束后额外返回一次评估 loss——它是
    "LoRA 与全参微调到底谁更好"这个问题的**同一把尺子**（day050 第六章
    已经把这条纪律立起来了）。
    """
    model = LoRAReferenceModel(base, config, seed=seed)
    history: list[tuple[int, float]] = []
    step = 0
    for _ in range(epochs):
        for batch in batches:
            loss_sum, count = model.accumulate(batch)
            model.apply_update(learning_rate)
            step += 1
            history.append((step, loss_sum / count))
    eval_loss: float | None = None
    if evaluate_batches:
        eval_loss, _ = model.evaluate(evaluate_batches)
    return model, history, eval_loss


__all__ = [
    "REFERENCE_LORA_ALPHA",
    "REFERENCE_LORA_RANK",
    "LoRAReferenceModel",
    "default_reference_lora_config",
    "reference_lora_accounting",
    "train_lora_reference",
]
