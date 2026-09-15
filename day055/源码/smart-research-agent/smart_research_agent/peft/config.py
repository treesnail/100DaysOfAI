"""LoRA / QLoRA 配置（M5-D3）：把"训哪些参数、用多少位存基座"写成一个对象.

day050 反复强调的一件事是**超参之间的算术关系必须先算清**。LoRA 把这件事
推到了一个新的层次：它引入了一组**只影响"参数量与显存"、不影响训练循环**
的配置项（``r`` / ``lora_alpha`` / ``target_modules`` / ``量化位数``）。这些
参数一旦写错，训练会照常跑、loss 会照常下降，只是**训出来的东西与你以为的
不是同一回事**：

- ``r`` 写大了（例如 ``r=512``）→ 可训练参数反而超过全参微调，"参数高效"
  这四个字就失效了；
- ``target_modules`` 只命中了 ``q_proj`` 而漏掉 ``v_proj`` → 适配器容量
  减半，属于静默的能力缺失；
- ``lora_alpha`` 与 ``r`` 的比例（即缩放 ``alpha/r``）改了 → **实际生效的
  学习率被整体缩放**，而日志里的 ``learning_rate`` 完全看不出这一点；
- QLoRA 的 ``block_size`` 从 64 改成 256 → 每参数存储从 **0.515869** 字节
  降到 **0.503967** 字节（块越大、常数开销越小），代价是每块 256 个权重
  共享一个 absmax，量化精度下降。**方向容易记反**：4-bit 的常数开销与块
  大小成反比。

所以本模块的设计与 day050 ``SFTTrainingArgs`` 完全一致：**字段名与
``peft.LoraConfig`` / ``transformers.BitsAndBytesConfig`` 逐字对齐**
（``to_peft_dict()`` / ``to_bnb_dict()`` 可以直接展开成构造参数，不需要
翻译表），**非法值一律在 ``validate()`` 里拒绝**（不留到远程 GPU 上才发现）。

依赖说明：``peft`` 与 ``bitsandbytes`` **不装进本课的日常环境**，本模块
只用标准库计算派生量。真实框架的落地由 ``peft/hf_script.py`` 生成脚本承担。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from smart_research_agent.sft.args import LR_SOFT_RANGE_PEFT


class PEFTConfigError(ValueError):
    """LoRA / QLoRA 配置非法（继承 ``ValueError``，便于按类型捕获）.

    与 ``SFTConfigError`` 同一种纪律：**非法配置在构造处就拒绝**。一个
    ``r = 0`` 或 ``target_modules = ()`` 的 LoRA 任务提交到 GPU 集群上，
    最轻的结果是白花几十分钟，最重的是悄悄训出一个"什么都没适配"的
    适配器——而这类产物在评估里往往只表现为"效果一般"，很难定位。
    """


#: ``target_modules`` 的预设（字符串）→ 实际模块名后缀元组.
#:
#: 为什么用预设而不是让调用方直接写模块名：**"只适配 q/v"与"适配全部线性层"
#: 的可训练参数量相差 5 倍以上**（7B 上 4.19M vs 19.99M），而两者的
#: ``r`` / ``lora_alpha`` 可以完全一样。把选择固化成有名字的预设，"这次
#: 到底适配了什么"就变成一句可核对的话。
#:
#: - ``attention``：只适配 ``q_proj`` / ``v_proj``，LoRA 原论文的选择
#:   （论文的消融实验结论是"在给定参数预算下，把这部分预算加在 q/v 上
#:   优于加在其他投影上"）；
#: - ``attention_all``：q/k/v/o 四个投影全上；
#: - ``mlp``：只适配 MLP 的三个投影；
#: - ``all_linear``：七类线性层全上，容量最大、参数量也最大；
#: - ``bigram``：本课参考模型（``ReferenceSFTModel``）里的唯一矩阵，
#:   名为 ``weight``——预设机制对参考模型同样适用。
LORA_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "attention": ("q_proj", "v_proj"),
    "attention_all": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "all_linear": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    "bigram": ("weight",),
}

#: ``bias`` 的取值（与 peft ``LoraConfig.bias`` 一致）.
#:
#: - ``none``：偏置也冻结（缺省）。**本课用它**：LoRA 的全部意义就是"只训
#:   一个低秩增量"，把偏置也放开会引入一批与秩无关的可训练参数，让
#:   "可训练参数 = 2·r·d"这条可以手算的公式不再成立；
#: - ``all``：所有偏置都训（含 LayerNorm 偏置）；
#: - ``lora_only``：只训被适配层自己的偏置。
SUPPORTED_LORA_BIASES: tuple[str, ...] = ("none", "all", "lora_only")

#: ``init_lora_weights`` 的取值.
#:
#: ``gaussian`` 是本课参考实现唯一支持的初始化：``A`` 取尺度匹配的高斯、
#: ``B`` 取全零，因此**第 0 步的增量 ``ΔW = scaling · B @ A`` 恒为 0**，
#: 适配器挂上去之后模型行为与基座逐位一致——这是"接上适配器不会让效果
#: 立刻变差"的保证。``random``（两个矩阵都随机）会破坏这条性质，只用于
#: 对照实验。其余取值（``loftq`` / ``eva`` / ``olora`` / ``orthogonal``）
#: 是 peft 提供的进阶初始化，需要真实运行时才能算，本课只做透传与显式拒绝。
SUPPORTED_INIT_MODES: tuple[str, ...] = (
    "gaussian",
    "random",
    "loftq",
    "eva",
    "olora",
    "orthogonal",
)

#: 需要 peft 运行时才能执行的初始化方式（参考实现会显式拒绝）.
RUNTIME_ONLY_INIT_MODES: tuple[str, ...] = ("loftq", "eva", "olora", "orthogonal")

#: 4-bit 量化码本的取值类型.
#:
#: 前两个与 ``BitsAndBytesConfig.bnb_4bit_quant_type`` 一致（bitsandbytes 只认
#: 这两个）；``int4`` 是**对照基线**——对称均匀 4-bit，用来回答"为什么是 NF4
#: 而不是随便一个 4-bit"。它只在本课的量化误差对照里使用，
#: ``to_bnb_dict()`` 会显式拒绝它（毕竟 bitsandbytes 不认这个值）。
SUPPORTED_QUANT_TYPES: tuple[str, ...] = ("nf4", "fp4", "int4")

#: 只用于本课对照、不能被 ``BitsAndBytesConfig`` 接受的口径
BASELINE_ONLY_QUANT_TYPES: tuple[str, ...] = ("int4",)

#: ``bnb_4bit_compute_dtype`` 的取值：计算时把 4-bit 权重反量化成什么精度.
SUPPORTED_COMPUTE_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")

#: ``bnb_4bit_quant_storage`` 的取值：4-bit 码点用什么类型承载.
SUPPORTED_QUANT_STORAGE: tuple[str, ...] = ("uint8", "bfloat16", "float16", "float32")

#: 4-bit 分块量化的块大小（每块共享一个 absmax 常数）.
SUPPORTED_QUANT_BLOCK_SIZES: tuple[int, ...] = (64, 128, 256, 512)

#: 二级（常数量化）块大小：每 256 个一级常数共享一个 fp32 缩放.
SUPPORTED_DOUBLE_QUANT_BLOCK_SIZES: tuple[int, ...] = (64, 128, 256, 512)

#: 各 dtype 的字节数（本课只用到这四种）.
DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "uint8": 1,
}

#: 量化码点位数（``nf4`` / ``fp4`` 都是 4 bit → 16 个码点）.
CODE_BITS = 4

#: 一级量化常数的存储位数（fp32）.
FULL_PRECISION_CONSTANT_BITS = 32

#: 二级（常数量化后）常数的存储位数（fp8）.
DOUBLE_QUANT_CONSTANT_BITS = 8

#: 参考实现（``ReferenceSFTModel`` 的 557×557 bigram 矩阵）对应的模块名.
REFERENCE_MODULE_NAME = "weight"


@dataclass
class LoRAConfig:
    """一份 LoRA 配置（字段名与 ``peft.LoraConfig`` 逐字对齐）.

    缺省值与 peft 官方文档给出的常见组合一致（``r=8`` / ``alpha=16`` /
    ``dropout=0.05`` / ``target_modules=["q_proj","v_proj"]`` / ``bias="none"``）。
    三个字段值得单独说明：

    - ``r``：低秩增量的秩。``ΔW = (alpha/r)·B @ A`` 的秩**上界**是
      ``min(r, in_features, out_features)``——这就是"参数高效"的来源。
      本课程数据集上 ``r=8`` 时可训练参数只有全参的 **2.787%**；
    - ``lora_alpha``：缩放分子。注意**单独看 ``lora_alpha`` 没有意义**，
      有意义的只有比例 ``alpha/r``（见 ``scaling``）。``alpha = 2r`` 是社区
      惯例，它让"r 翻倍"时缩放保持不变；
    - ``target_modules``：可以是预设名（见 ``LORA_TARGET_PRESETS``）或
      模块名后缀元组。预设在构造时展开成具体元组（``resolved``），
      因此后续所有计算——参数量、显存、生成脚本——看到的都是同一个集合。
    """

    #: 低秩维（秩）
    r: int = 8
    #: 缩放分子；实际生效的缩放是 ``alpha / r``（或 ``alpha / sqrt(r)``）
    lora_alpha: int = 16
    #: 施加在 LoRA 分支输入上的 dropout（正则化；0 表示关闭）
    lora_dropout: float = 0.05
    #: 目标模块：预设名（``attention`` / ``attention_all`` / ``mlp`` /
    #: ``all_linear`` / ``bigram``）或模块名后缀元组
    target_modules: str | tuple[str, ...] = "attention"
    #: 偏置训练策略：``none``（全冻结，缺省）/ ``all`` / ``lora_only``
    bias: str = "none"
    #: 是否使用 rank-stabilized LoRA（缩放改为 ``alpha / sqrt(r)``）
    use_rslora: bool = False
    #: 是否使用 DoRA（把增量分解为"方向 + 幅度"；本课只做配置透传）
    use_dora: bool = False
    #: 初始化方式（见 ``SUPPORTED_INIT_MODES``）
    init_lora_weights: str = "gaussian"
    #: 任务类型（peft ``TaskType`` 的字符串取值）
    task_type: str = "CAUSAL_LM"
    #: 基座权重是否以"输出维 × 输入维"的转置方式存储（GPT-2 系为 True）
    fan_in_fan_out: bool = False

    def __post_init__(self) -> None:
        # 构造处就把预设展开成元组：后续所有派生量都基于同一个集合计算
        if isinstance(self.target_modules, list):
            self.target_modules = tuple(self.target_modules)

    # ------------------------------------------------------------------ 派生量
    @property
    def resolved_targets(self) -> tuple[str, ...]:
        """展开后的目标模块名元组（预设名 → 具体后缀）."""
        if isinstance(self.target_modules, str):
            preset = LORA_TARGET_PRESETS.get(self.target_modules)
            if preset is None:
                raise PEFTConfigError(
                    f"未知的 target_modules 预设 {self.target_modules!r}，"
                    f"可选：{', '.join(sorted(LORA_TARGET_PRESETS))}；"
                    "也可以直接传模块名元组（如 ('q_proj', 'v_proj')）"
                )
            return preset
        return tuple(self.target_modules)

    @property
    def target_preset_name(self) -> str:
        """预设名；直接传元组时返回 ``"custom"``（供日志与 API 展示）."""
        if isinstance(self.target_modules, str):
            return self.target_modules
        return "custom"

    @property
    def scaling(self) -> float:
        """实际生效的缩放系数：``alpha / r``（rsLoRA 时为 ``alpha / sqrt(r)``）.

        ``sqrt(r)`` 不是随手改的：秩越大，``B @ A`` 中每个元素的量级随
        ``sqrt(r)`` 增长，用 ``alpha/r`` 会让大秩适配器的有效步长被压小，
        于是"加大秩"看起来毫无收益。rank-stabilized LoRA 把分母换成
        ``sqrt(r)`` 正是为了让不同秩之间的学习率可比。
        """
        if self.r <= 0:
            raise PEFTConfigError(f"r 必须为正整数，收到 {self.r}")
        denominator = math.sqrt(self.r) if self.use_rslora else float(self.r)
        return self.lora_alpha / denominator

    @property
    def scaling_formula(self) -> str:
        """缩放的可读公式（写进日志，避免"实际学习率被缩放却看不出来"）."""
        if self.use_rslora:
            return f"alpha/sqrt(r) = {self.lora_alpha}/sqrt({self.r}) = {self.scaling:.6f}"
        return f"alpha/r = {self.lora_alpha}/{self.r} = {self.scaling:.6f}"

    @property
    def requires_peft_runtime(self) -> bool:
        """该配置是否只能在真实 peft 运行时里执行（本课参考实现会拒绝）."""
        return self.init_lora_weights in RUNTIME_ONLY_INIT_MODES

    def adapter_parameter_count(self, in_features: int, out_features: int) -> int:
        """单个被适配层的**新增**参数数：``r × (in + out)``（含偏置时另加）.

        这是"LoRA 把参数量从平方级降到线性级"的全部算术：一个
        ``4096 × 4096`` 的投影层原本有 ``16 777 216`` 个参数，``r=8`` 时
        只新增 ``8 × (4096 + 4096) = 65 536`` 个，比例正好是
        ``2r/d = 16/4096 = 0.3906%``——与 day049 预先算出的四个比例之一
        逐位一致。

        偏置项按 ``bias`` 的取值叠加：``all`` / ``lora_only`` 时每层多出
        ``out_features`` 个（``lora_only`` 只训被适配层自己的偏置）。
        """
        if in_features <= 0 or out_features <= 0:
            raise PEFTConfigError(
                f"in_features / out_features 必须为正整数，收到 {in_features} / {out_features}"
            )
        count = self.r * (in_features + out_features)
        if self.bias in ("all", "lora_only"):
            count += out_features
            if self.bias == "all":
                count += in_features
        return count

    def rank_upper_bound(self, in_features: int, out_features: int) -> int:
        """增量矩阵的秩上界：``min(r, in_features, out_features)``.

        ``ΔW = (alpha/r)·B @ A`` 中 ``B`` 是 ``out × r``、``A`` 是
        ``r × in``，因此 ``rank(ΔW) ≤ min(out, r, in)``。这条上界是 LoRA
        可被证伪的性质之一：``r=1`` 时 ``ΔW`` 的**每一行都成比例**，用一条
        断言就能验证（见 ``tests/test_lora_layers.py``）。
        """
        return min(self.r, in_features, out_features)

    def trainable_ratio(self, base_parameters: int, adapter_parameters: int) -> float:
        """可训练参数占比 = 适配器参数 / (基座 + 适配器) 参数.

        分母用"模型总参数"而不是"基座参数"：报告里说的是"这次训练要更新
        模型的百分之几"，用总参数才是那个意思。
        """
        total = base_parameters + adapter_parameters
        if total <= 0:
            raise PEFTConfigError("总参数量必须为正数")
        return adapter_parameters / total

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """硬校验：非法配置在这里抛 ``PEFTConfigError``（不留到训练循环）."""
        if self.r < 1:
            raise PEFTConfigError(
                f"r 必须为正整数（r=0 等于不训任何参数），收到 {self.r}"
            )
        if self.lora_alpha <= 0:
            raise PEFTConfigError(f"lora_alpha 必须为正数，收到 {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise PEFTConfigError(
                f"lora_dropout 必须落在 [0, 1) 区间，收到 {self.lora_dropout}"
            )
        if self.bias not in SUPPORTED_LORA_BIASES:
            raise PEFTConfigError(
                f"未知的 bias {self.bias!r}，可选：{', '.join(SUPPORTED_LORA_BIASES)}"
            )
        if self.init_lora_weights not in SUPPORTED_INIT_MODES:
            raise PEFTConfigError(
                f"未知的 init_lora_weights {self.init_lora_weights!r}，"
                f"可选：{', '.join(SUPPORTED_INIT_MODES)}"
            )
        targets = self.resolved_targets
        if not targets:
            raise PEFTConfigError("target_modules 不能为空：没有目标模块就不会插入任何适配器")
        duplicates = len(set(targets)) != len(targets)
        if duplicates:
            raise PEFTConfigError(f"target_modules 存在重复项：{targets}")
        if self.use_dora and self.use_rslora:
            raise PEFTConfigError(
                "use_dora 与 use_rslora 不能同时启用：DoRA 把增量分解为"
                "「方向 × 幅度」，幅度项会改写缩放口径，与 rsLoRA 的 "
                "alpha/sqrt(r) 叠加后有效缩放不再有单一公式可表达"
            )
        if self.use_dora and self.init_lora_weights == "random":
            raise PEFTConfigError(
                "use_dora 时 init_lora_weights 不能为 random：DoRA 的幅度项"
                "从 base 权重列范数初始化，随机初始化会让第 0 步的增量非零"
            )
        if self.requires_peft_runtime and self.fan_in_fan_out:
            raise PEFTConfigError(
                f"init_lora_weights={self.init_lora_weights!r} 不支持 fan_in_fan_out"
            )

    # --------------------------------------------------------------- 投影与序列化
    def to_peft_dict(self) -> dict[str, Any]:
        """投影为 ``peft.LoraConfig`` 的构造参数（字段名逐字对齐）.

        对齐的收益与 day050 ``SFTTrainingArgs.to_hf_dict()`` 相同：
        ``LoraConfig(**cfg.to_peft_dict())`` 可以直接展开，不需要一张
        "我们的字段 → peft 字段"的映射表——而翻译表就是漂移的来源。
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.resolved_targets),
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "init_lora_weights": self.init_lora_weights,
            "task_type": self.task_type,
            "fan_in_fan_out": self.fan_in_fan_out,
        }

    def to_dict(self) -> dict[str, Any]:
        """完整配置（含本课自己的派生量与约束），用于日志与落盘.

        ``target_modules`` 保留**原始写法**（预设名或元组），而不是展开后的
        元组：这样 ``from_dict(to_dict())`` 是逐字段等值的往返，预设名不会
        在往返中丢失（丢失后 ``target_preset_name`` 会从 ``"mlp"`` 变成
        ``"custom"``，日志里就再也看不出这次用的是什么预设）。展开后的集合
        另用 ``resolved_targets`` 单独给出。
        """
        payload = asdict(self)
        payload["target_modules"] = (
            self.target_modules
            if isinstance(self.target_modules, str)
            else list(self.target_modules)
        )
        payload["resolved_targets"] = list(self.resolved_targets)
        payload["target_preset_name"] = self.target_preset_name
        payload["scaling"] = self.scaling
        payload["scaling_formula"] = self.scaling_formula
        payload["requires_peft_runtime"] = self.requires_peft_runtime
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoRAConfig:
        """从 ``to_dict`` 的产物还原（未知键忽略，便于向前兼容）."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in known:
                continue
            if key == "target_modules" and isinstance(value, list):
                value = tuple(value)
            data[key] = value
        return cls(**data)

    def with_overrides(self, **changes: Any) -> LoRAConfig:
        """返回改了若干字段的新对象（不改原对象，实验记录才对得上）."""
        return replace(self, **changes)

    def describe(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"LoRA r={self.r} alpha={self.lora_alpha} dropout={self.lora_dropout} "
            f"targets={self.target_preset_name}{list(self.resolved_targets)} "
            f"bias={self.bias} scaling[{self.scaling_formula}]"
        )


@dataclass
class QLoRAConfig:
    """一份 4-bit 量化配置（字段名与 ``BitsAndBytesConfig`` 逐字对齐）.

    QLoRA 只做一件事：**把冻结的基座权重压到 4-bit 存储，前向计算时再
    反量化回 16-bit**。适配器（LoRA）的精度不变——训练的是 fp16/bf16 的
    A/B，不是 4-bit 的基座。所以"QLoRA 掉点"从来不是"4-bit 训练"造成的，
    而是"4-bit 存储引入了量化误差"造成的。

    四个字段值得单独说明：

    - ``bnb_4bit_quant_type``：``nf4``（正态浮点，缺省）或 ``fp4``。
      NF4 的 16 个码点取自标准正态分布的等概率分位数，正因为**预训练权重
      近似零均值正态分布**，它才比均匀量化更适合；
    - ``bnb_4bit_use_double_quant``：把一级量化的 fp32 常数**再量化一次**
      （fp8 + 每 256 个常数一个 fp32 缩放）。每参数存储因此从
      ``4 + 32/64 = 4.5 bit`` 降到 ``4 + 8/64 + 32/(64·256) = 4.126953 bit``，
      在 7B 上省下约 0.33 GB；
    - ``bnb_4bit_compute_dtype``：反量化后的计算精度，缺省 ``bfloat16``；
    - ``block_size``：每多少个权重共享一个 absmax 常数。论文取 **64**：
      块越小精度越高、常数开销越大，64 是收益与开销的折中点。
    """

    #: 码本类型：``nf4`` / ``fp4``
    bnb_4bit_quant_type: str = "nf4"
    #: 反量化后的计算精度
    bnb_4bit_compute_dtype: str = "bfloat16"
    #: 是否启用二级（常数量化）
    bnb_4bit_use_double_quant: bool = True
    #: 4-bit 码点的承载类型
    bnb_4bit_quant_storage: str = "uint8"
    #: 一级量化的块大小（每块共享一个 absmax）
    block_size: int = 64
    #: 二级（常数量化）的块大小（每块共享一个 fp32 缩放）
    double_quant_block_size: int = 256

    # ------------------------------------------------------------------ 派生量
    @property
    def code_bits(self) -> int:
        """码点本身的位数（``nf4`` / ``fp4`` 都是 4 bit）."""
        return CODE_BITS

    @property
    def constant_bits_per_parameter(self) -> float:
        """**常数开销**的每参数位数（不含 4-bit 码点本身）.

        单重量化：每 ``block_size`` 个权重共享一个 fp32 常数 → ``32/block_size``；
        二级量化：该常数本身用 fp8 存、每 ``double_quant_block_size`` 个常数
        再共享一个 fp32 缩放 → ``8/block_size + 32/(block_size·dq_block)``。

        实测（``block_size=64`` / ``dq_block=256``）::

            单重：32/64                     = 0.5      bit/参数
            二级：8/64 + 32/(64×256)        = 0.126953 bit/参数
        """
        if self.bnb_4bit_use_double_quant:
            return (
                DOUBLE_QUANT_CONSTANT_BITS / self.block_size
                + FULL_PRECISION_CONSTANT_BITS / (self.block_size * self.double_quant_block_size)
            )
        return FULL_PRECISION_CONSTANT_BITS / self.block_size

    @property
    def bits_per_parameter(self) -> float:
        """每参数总位数 = 码点 + 常数开销."""
        return self.code_bits + self.constant_bits_per_parameter

    @property
    def bytes_per_parameter(self) -> float:
        """每参数字节数（= ``bits_per_parameter / 8``）.

        ``nf4`` + 二级量化 + ``block_size=64`` 时是 **0.515869** 字节；
        不启用量化常数时是 **0.5625** 字节；16-bit 基座则是 2 字节。
        """
        return self.bits_per_parameter / 8.0

    @property
    def compute_dtype_bytes(self) -> int:
        """计算精度每参数字节数（反量化后的临时量级，不是存储量）."""
        return DTYPE_BYTES[self.bnb_4bit_compute_dtype]

    def base_weight_bytes(self, base_parameters: int) -> int:
        """基座权重的存储字节数（这是 QLoRA 真正省下来的那一项）."""
        if base_parameters <= 0:
            raise PEFTConfigError(f"base_parameters 必须为正整数，收到 {base_parameters}")
        return int(round(base_parameters * self.bytes_per_parameter))

    def to_bnb_dict(self) -> dict[str, Any]:
        """投影为 ``transformers.BitsAndBytesConfig`` 的构造参数（字段名逐字对齐）.

        ``load_in_4bit`` 恒为 ``True``：这个配置类的定义就是 4-bit 基座，
        留一个"可以是 False"的开关只会让读者怀疑自己在配什么。

        ``int4`` 会在这里被拒绝：它是本课的对照基线，bitsandbytes 不认这个
        取值，把它写进生成脚本会让脚本一运行就报错——**与其让下游报一个
        看不懂的错，不如在这里说清楚**。
        """
        if self.bnb_4bit_quant_type in BASELINE_ONLY_QUANT_TYPES:
            raise PEFTConfigError(
                f"{self.bnb_4bit_quant_type!r} 是本课的量化对照基线，"
                "BitsAndBytesConfig 不接受这个取值（可选：nf4 / fp4）"
            )
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_quant_storage": self.bnb_4bit_quant_storage,
        }

    def to_dict(self) -> dict[str, Any]:
        """完整配置（含派生量）."""
        payload = asdict(self)
        payload["bits_per_parameter"] = round(self.bits_per_parameter, 6)
        payload["bytes_per_parameter"] = round(self.bytes_per_parameter, 6)
        payload["constant_bits_per_parameter"] = round(self.constant_bits_per_parameter, 6)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QLoRAConfig:
        """从 ``to_dict`` 的产物还原（未知键忽略）."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})

    def with_overrides(self, **changes: Any) -> QLoRAConfig:
        """返回改了若干字段的新对象（不改原对象）."""
        return replace(self, **changes)

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """硬校验：非法配置在这里抛 ``PEFTConfigError``."""
        if self.bnb_4bit_quant_type not in SUPPORTED_QUANT_TYPES:
            raise PEFTConfigError(
                f"未知的量化类型 {self.bnb_4bit_quant_type!r}，"
                f"可选：{', '.join(SUPPORTED_QUANT_TYPES)}"
            )
        if self.bnb_4bit_compute_dtype not in SUPPORTED_COMPUTE_DTYPES:
            raise PEFTConfigError(
                f"未知的 compute_dtype {self.bnb_4bit_compute_dtype!r}，"
                f"可选：{', '.join(SUPPORTED_COMPUTE_DTYPES)}"
            )
        if self.bnb_4bit_quant_storage not in SUPPORTED_QUANT_STORAGE:
            raise PEFTConfigError(
                f"未知的 quant_storage {self.bnb_4bit_quant_storage!r}，"
                f"可选：{', '.join(SUPPORTED_QUANT_STORAGE)}"
            )
        if self.block_size not in SUPPORTED_QUANT_BLOCK_SIZES:
            raise PEFTConfigError(
                f"未知的 block_size {self.block_size!r}，"
                f"可选：{', '.join(str(item) for item in SUPPORTED_QUANT_BLOCK_SIZES)}"
            )
        if self.double_quant_block_size not in SUPPORTED_DOUBLE_QUANT_BLOCK_SIZES:
            raise PEFTConfigError(
                f"未知的 double_quant_block_size {self.double_quant_block_size!r}，"
                f"可选：{', '.join(str(item) for item in SUPPORTED_DOUBLE_QUANT_BLOCK_SIZES)}"
            )

    def describe(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"QLoRA {self.bnb_4bit_quant_type} block={self.block_size} "
            f"double_quant={self.bnb_4bit_use_double_quant} "
            f"compute={self.bnb_4bit_compute_dtype} "
            f"{self.bits_per_parameter:.6f} bit/参数（{self.bytes_per_parameter:.6f} B）"
        )


def quantization_table(block_sizes: tuple[int, ...] = (64, 128, 256, 512)) -> list[dict[str, Any]]:
    """给出一张"块大小 → 每参数存储"的对照表（供教程与 API 直接渲染）.

    它回答的是一个容易被忽略的问题：**4-bit 并没有把权重压到 0.5 字节**。
    常数开销是真实存在的，块越小开销越大：``block_size=64`` 且二级量化时
    每参数 0.515869 字节，``block_size=512`` 时降到 0.501984 字节——
    代价是每块 512 个权重共享一个 absmax，精度下降。
    """
    rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        config = QLoRAConfig(block_size=block_size)
        no_double = config.with_overrides(bnb_4bit_use_double_quant=False)
        rows.append(
            {
                "block_size": block_size,
                "bytes_per_parameter": round(config.bytes_per_parameter, 6),
                "bytes_per_parameter_single_quant": round(no_double.bytes_per_parameter, 6),
                "bits_per_parameter": round(config.bits_per_parameter, 6),
                "constant_bits_per_parameter": round(
                    config.constant_bits_per_parameter, 6
                ),
            }
        )
    return rows


def default_peft_lr_range() -> tuple[float, float]:
    """LoRA / PEFT 的学习率经验区间（转发 ``sft.args.LR_SOFT_RANGE_PEFT``）.

    单独提供这个入口而不是让调用方直接 import 常量：**LoRA 的学习率区间
    与全参微调不同**（只训一小批新增参数时可以更大），而"区间必须与训练
    方式匹配"这件事在 day050 已经吃过一次亏（针对 7B 全参的区间用在参考
    模型上会永久误报）。
    """
    return LR_SOFT_RANGE_PEFT


__all__ = [
    "BASELINE_ONLY_QUANT_TYPES",
    "CODE_BITS",
    "DOUBLE_QUANT_CONSTANT_BITS",
    "DTYPE_BYTES",
    "FULL_PRECISION_CONSTANT_BITS",
    "LORA_TARGET_PRESETS",
    "REFERENCE_MODULE_NAME",
    "RUNTIME_ONLY_INIT_MODES",
    "SUPPORTED_COMPUTE_DTYPES",
    "SUPPORTED_DOUBLE_QUANT_BLOCK_SIZES",
    "SUPPORTED_INIT_MODES",
    "SUPPORTED_LORA_BIASES",
    "SUPPORTED_QUANT_BLOCK_SIZES",
    "SUPPORTED_QUANT_STORAGE",
    "SUPPORTED_QUANT_TYPES",
    "LoRAConfig",
    "PEFTConfigError",
    "QLoRAConfig",
    "default_peft_lr_range",
    "quantization_table",
]
