"""微调方法总览与选型建议（M5-D1）.

微调方法很多，但它们回答的是两类不同的问题：

1. **怎么省着改**（参数高效微调，parameter-efficient）：
   LoRA / QLoRA 不更新原权重，只训练一个小的低秩增量 ``ΔW = B @ A``；
2. **改多少**（全参微调，full-parameter）：
   SFT 更新全部权重，能力上限最高，代价是显存与存储；
3. **按什么信号改**（对齐，alignment）：
   DPO 直接吃「chosen/rejected」偏好对，RLHF-PPO 先训奖励模型再用强化
   学习优化——两者优化的都不是"下一个 token 像不像标准答案"，而是
   "人更喜欢哪个回答"。

本模块把"怎么选"固化成**确定性规则**（``recommend_method``）：同样的
输入永远给出同样的建议与理由。这不是要替代工程师判断，而是把判断依据
显式化——理由与告警都会随建议返回，评审时可以逐条反驳，而不是面对一个
黑盒结论。纯函数 + 纯数据也让 M5 的选型逻辑可被单元测试逐分支验证。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 全参监督微调
SFT = "sft"
#: 低秩适配（Low-Rank Adaptation）
LORA = "lora"
#: 量化 + LoRA（4-bit 基座 + 低秩增量）
QLORA = "qlora"
#: 直接偏好优化（Direct Preference Optimization）
DPO = "dpo"
#: 经典 RLHF：奖励模型 + PPO 强化学习
RLHF_PPO = "rlhf-ppo"


@dataclass(frozen=True)
class MethodProfile:
    """一个微调方法的结构化画像（供文档表格与 API 契约复用）.

    ``frozen=True`` 是刻意的：方法画像是一份**知识常量**，任何代码路径
    都不该就地修改它；要定制请新建 MethodProfile。
    """

    key: str
    full_name: str
    #: "parameter-efficient" / "full-parameter" / "alignment" 三选一
    family: str
    #: 经验数据需求量级（不是硬门槛，而是"低于此量级别指望见效"）
    data_need: str
    #: 可训练参数比例的定性描述（精确计算见 lora_trainable_ratio）
    trainable_ratio_hint: str
    #: 训练稳定性与主要风险
    stability: str
    best_for: str
    #: 训练产物形态：要保存/部署哪些文件
    artifacts: tuple[str, ...]
    notes: str


#: 五种方法的结构化画像；键即 ``key`` 字段（字典序仅用于渲染表格）
METHODS: dict[str, MethodProfile] = {
    SFT: MethodProfile(
        key=SFT,
        full_name="全参监督微调（Supervised Fine-Tuning）",
        family="full-parameter",
        data_need="中等：1k~100k 条高质量指令-答案对",
        trainable_ratio_hint="100%（全部权重参与更新）",
        stability="较低：显存占用大，小数据下易过拟合与灾难性遗忘",
        best_for="数据与显存都充足、且需要整体抬升领域能力时",
        artifacts=("full_model_weights",),
        notes="推理零额外开销；每份任务都要存一份完整权重，存储与迭代成本最高。",
    ),
    LORA: MethodProfile(
        key=LORA,
        full_name="低秩适配（Low-Rank Adaptation）",
        family="parameter-efficient",
        data_need="较小：几百~几万条高质量样本即可见效",
        trainable_ratio_hint="通常 0.01%~1%（仅低秩增量 B@A）",
        stability="较高：原权重冻结，超参不敏感，不易灾难性遗忘",
        best_for="显存有限、需要多任务多适配器并存、快速迭代试验",
        artifacts=("adapter_config.json", "adapter_model.safetensors"),
        notes="推理时可合并回基座（零延迟）或保持分离（多任务热插拔）。",
    ),
    QLORA: MethodProfile(
        key=QLORA,
        full_name="量化低秩适配（Quantized LoRA，4-bit）",
        family="parameter-efficient",
        data_need="较小：与 LoRA 同量级（数据质量比数量更重要）",
        trainable_ratio_hint="与 LoRA 同级（增量参数仍为 fp16/bf16）",
        stability="中等：4-bit 量化引入噪声，训练时间更长",
        best_for="单卡消费级显卡（< 16GB）微调 7B~13B 模型",
        artifacts=("adapter_config.json", "adapter_model.safetensors"),
        notes="基座以 4-bit 冻结加载，显存约为 fp16 全参的 1/4；训练速度更慢。",
    ),
    DPO: MethodProfile(
        key=DPO,
        full_name="直接偏好优化（Direct Preference Optimization）",
        family="alignment",
        data_need="偏好对：每条 prompt 至少一对 chosen/rejected，建议 1k 对以上",
        trainable_ratio_hint="通常接在 LoRA/SFT 之上（可训练比例取决于底座）",
        stability="中等：对偏好数据质量与参考模型极敏感",
        best_for="已有明确「哪个回答更好」的人工/模型偏好标注，想让风格对齐",
        artifacts=("policy_adapter", "reference_model_ref"),
        notes="不需要奖励模型与 RL 采样循环，工程复杂度远低于 PPO。",
    ),
    RLHF_PPO: MethodProfile(
        key=RLHF_PPO,
        full_name="人类反馈强化学习（RLHF，PPO）",
        family="alignment",
        data_need="偏好对 + 奖励模型训练集 + 在线采样预算",
        trainable_ratio_hint="奖励模型全参 + 策略模型（常用 LoRA）",
        stability="低：四个模型同场（策略/参考/奖励/价值），超参难调",
        best_for="有充足标注与算力预算、需要精细控制对齐目标的研究场景",
        artifacts=("reward_model", "policy_adapter", "ppo_config"),
        notes="流程最长（SFT → 奖励模型 → PPO），短期落地成本最高。",
    ),
}


@dataclass
class MethodRecommendation:
    """一条选型建议：方法 + 支撑理由 + 必须知道的告警.

    ``reasons`` 与 ``warnings`` 分开，是因为二者说话对象不同：理由写给
    "为什么是它"，告警写给"选了它之后还得注意什么"。把告警混进理由里，
    评审时很容易被当成理由的一部分而被忽略。
    """

    method: str
    reasons: list[str]
    warnings: list[str]


def recommend_method(
    *,
    examples: int,
    gpu_memory_gb: float,
    has_preference_pairs: bool = False,
    need_full_capability: bool = False,
) -> MethodRecommendation:
    """按确定性规则推荐微调方法（顺序判断，先命中先返回）.

    规则顺序（每一步的"为什么"都写进 reasons）：

    1. ``has_preference_pairs=True`` → **DPO**。手上已经有偏好对，
       说明"哪个回答更好"的信号是现成的；DPO 可以直接在偏好数据上
       优化策略，不必像 RLHF-PPO 那样先训奖励模型再跑强化学习循环。
       ``RLHF_PPO`` 只会在告警里作为"数据量不足时的别路"被提及——
       它需要的算力与工程复杂度都高得多，不该作为首选。
    2. ``need_full_capability`` 且 ``gpu_memory_gb >= 80`` → **SFT**。
       80GB 是 A100/H100 级别单卡的显存门槛，也是"全参训一个 7B 模型
       （fp16 权重 + 梯度 + 优化器状态 ≈ 16 倍参数量）"的现实下限。
    3. ``gpu_memory_gb < 16`` → **QLoRA**。低于 16GB 时 fp16 的基座都
       放不下，必须靠 4-bit 量化把权重压到 1/4，再用低秩增量训练。
    4. 其余 → **LoRA**。显存够载入 fp16 基座，就没有必要承受量化的
       精度损失与额外训练时间；低秩增量是这一档的性价比最优解。

    另外，``examples < 500`` 时**无论走哪条分支**都会追加一条告警：
    样本量不足时，换方法带来的收益远小于补数据，先把这条话说在前面。
    """
    reasons: list[str] = []
    warnings: list[str] = []

    if examples < 500:
        warnings.append(
            f"样本量偏少（{examples} 条）：500 条以下微调极易过拟合，"
            "先补数据比换方法更有效。"
        )

    if has_preference_pairs:
        reasons.append(
            "提供了偏好对（chosen/rejected）：DPO 可直接在偏好数据上优化策略，"
            "无需先训练奖励模型，工程链路比 RLHF-PPO 短得多。"
        )
        if examples < 1000:
            warnings.append(
                f"偏好对仅 {examples} 条：DPO 对偏好数据的规模与质量都很敏感，"
                "建议先扩到 1000 条以上；若受限于标注预算，RLHF-PPO 需要额外"
                "训练奖励模型与在线采样，成本只会更高。"
            )
        return MethodRecommendation(method=DPO, reasons=reasons, warnings=warnings)

    if need_full_capability and gpu_memory_gb >= 80:
        reasons.append(
            f"需要完整能力且显存充足（{gpu_memory_gb}GB >= 80GB）："
            "全参 SFT 更新全部权重，能力上限最高。"
        )
        return MethodRecommendation(method=SFT, reasons=reasons, warnings=warnings)

    if gpu_memory_gb < 16:
        reasons.append(
            f"显存受限（{gpu_memory_gb}GB < 16GB）：QLoRA 以 4-bit 量化加载基座，"
            "可在单张消费级显卡上完成训练。"
        )
        return MethodRecommendation(method=QLORA, reasons=reasons, warnings=warnings)

    reasons.append(
        f"显存条件（{gpu_memory_gb}GB）足以承载 fp16 基座：LoRA 只训练低秩增量，"
        "在效果与成本之间性价比最高。"
    )
    return MethodRecommendation(method=LORA, reasons=reasons, warnings=warnings)


def lora_trainable_ratio(in_features: int, out_features: int, rank: int) -> float:
    """LoRA 可训练参数占该线性层全参的比例.

    公式::

        全参参数量   = d_in * d_out
        LoRA 参数量  = r * (d_in + d_out)      # A: r×d_in，B: d_out×r
        ratio        = r * (d_in + d_out) / (d_in * d_out)

    直觉：LoRA 用一个"瘦长 × 胖短"的矩阵对逼近原权重矩阵，参数量从
    平方级降到线性级。以 4096×4096 的层、rank=8 为例::

        ratio = 8 * (4096 + 4096) / (4096 * 4096) = 16 / 4096 ≈ 0.0039

    也就是说这一层只有约 0.39% 的参数需要训练——这正是 LoRA 能在
    单卡上跑起来的算术基础。

    参数非法（任一参数非正，或 ``in_features * out_features == 0``）
    抛 ``ValueError``：比例无定义时宁可报错，也不要返回一个 0 或 inf
    被下游当成"不用训练"或"参数爆炸"。
    """
    if in_features <= 0 or out_features <= 0 or rank <= 0 or in_features * out_features == 0:
        raise ValueError(
            f"in_features/out_features/rank 必须为正整数，收到 "
            f"in_features={in_features}, out_features={out_features}, rank={rank}"
        )
    return rank * (in_features + out_features) / (in_features * out_features)


def render_methods_table() -> str:
    """把 ``METHODS`` 渲染成 Markdown 表格（供文档与教程直接引用）.

    表格由代码生成而非手写在 markdown 里：文档里最危险的错误是"代码改了
    文档没改"。把它做成函数，文档只要贴一次输出，漂移立刻可见。
    """
    header = "| 方法 | 全称 | 家族 | 数据需求 | 可训练比例 | 稳定性 | 适用场景 |"
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [
        f"| `{profile.key}` | {profile.full_name} | {profile.family} | {profile.data_need} | "
        f"{profile.trainable_ratio_hint} | {profile.stability} | {profile.best_for} |"
        for profile in METHODS.values()
    ]
    return "\n".join([header, separator, *rows])
