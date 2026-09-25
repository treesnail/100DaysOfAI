"""部署形态与端点档案：把"线上跑的到底是哪一种模型"写成一个可校验的对象（M5-D11）.

day052 的 ``peft/deploy.py`` 已经回答过"适配器怎么合并、两种形态各占多少体积"。
今天的问题更进一步：

> **线上那台机器上，此刻加载的到底是哪一种？**

三种形态在**协议层**完全一样——都是 OpenAI 兼容端点，day045 的 ``LocalModel``
一行都不用改。它们在**运维层**完全不同：

| 形态 | 线上是什么 | 换业务的代价 | 它带来的事故 |
|------|-----------|-------------|-------------|
| ``base`` | 未微调的基座 | 无（对照组 / 回退目标） | 无 |
| ``adapter`` | 基座 + 一个适配器（热插拔） | 换十几 MB 文件 | **基座与适配器不匹配时不会报错** |
| ``merged`` | 合并后的完整模型 | 换一份完整权重（数 GiB） | 忘了重新合并 → 线上还是旧权重 |

最后一行是本模块存在的理由。``ServingSpec.__post_init__`` 把"形态"与
"必填字段"用一张表绑在一起（``KIND_REQUIRED_FIELDS``）：**声明成 ``adapter``
却没给 ``adapter_dir`` 的部署会在构造期就失败**，而不是在某次推理时安静地
给出基座的答案——那是"专属模型上线了却没生效"最真实的一种发生方式。

## 内存算术：为什么要按"每 token 多少字节"来算

自建推理的显存里有两块完全不同的东西：

```text
权重      与上下文长度无关，开机就占满    参数数 × 每参数字节
KV 缓存   与上下文长度、并发数成正比      2 × 层数 × KV头数 × head_dim × 字节 × 长度 × 并发
```

第二行是自建部署最容易算错的一项，因为它的量级"看起来"应该很小。
实测（本模块 demo 第 3 节，Qwen3-8B 的公开配置 36 层 / 8 个 KV 头 / head_dim 128，
bf16 即每值 2 字节）：

```text
每 token 每序列：147456 字节 = 144.0000 KiB
4096 上下文 × 1 并发：576.0000 MiB
4096 上下文 × 4 并发：2304.0000 MiB（2.25 GiB）
```

也就是说，"把并发从 1 调到 4"会在**权重之外**多要 1.69 GiB，而且它**不会
提升任何质量**。Ollama 官方 FAQ 把这件事写在 ``OLLAMA_NUM_PARALLEL`` 的
说明里：并发请求会按并发数放大上下文与内存分配
（<https://docs.ollama.com/faq>）。本模块把这句话做成一个可复算的函数，
而不是留在文档里当一句提醒。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.llm.local_model import (
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    OLLAMA,
    SUPPORTED_BACKENDS,
    VLLM,
)
from smart_research_agent.serving.errors import ServingError
from smart_research_agent.sft.hf_script import DEFAULT_BASE_MODEL

#: 三种部署形态。
KIND_BASE = "base"
KIND_ADAPTER = "adapter"
KIND_MERGED = "merged"
KINDS: tuple[str, ...] = (KIND_BASE, KIND_ADAPTER, KIND_MERGED)

#: 形态 → 必填字段。**表驱动而不是散在 if 里**：这张表可以被打印出来核对，
#: 而"哪些字段是哪种形态的必需品"这件事每次加一种形态都会被重新问一遍。
KIND_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    KIND_BASE: (),
    KIND_ADAPTER: ("adapter_dir",),
    KIND_MERGED: ("merged_dir",),
}

#: 形态的一句话说明（API 的自我描述端点与文档同源）。
KIND_DESCRIPTIONS: dict[str, str] = {
    KIND_BASE: "未微调的基座：对照组与回退目标，A/B 比较里的基准臂",
    KIND_ADAPTER: "一个基座 + 一个 LoRA 适配器：换业务只换十几 MB，需要 peft 运行时",
    KIND_MERGED: "合并后的完整模型：部署链路与普通模型一致，不需要 peft 运行时",
}

#: 每参数字节数与量化位宽的换算：``bits / 8``。
#: 只允许 4 / 8 / 16 / 32 这几档（含 QLoRA 的 4-bit）——给一个任意浮点数
#: 会让"我到底按几位算的"变得不可核对。
SUPPORTED_QUANTIZATION_BITS: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)

#: Ollama 官方 FAQ 的并发缺省值：``OLLAMA_NUM_PARALLEL`` 默认 **1**
#: （<https://docs.ollama.com/faq>）。把它写成常量而不是散落的字面量 1，
#: 是因为"并发 1"与"没配置"在本课程里必须是同一件事。
DEFAULT_NUM_PARALLEL = 1

#: KV 缓存里每个值的字节数：bf16/fp16 = 2，fp32 = 4。
DEFAULT_KV_BYTES_PER_VALUE = 2

#: 部署这件事**做不到什么**。与 day059 模型卡的四条限制同一条纪律：
#: **每条限制都要带一个能被用来否掉结论的数字**，否则它只是免责声明。
SERVING_LIMITATIONS: tuple[str, ...] = (
    "本课程没有真实 GPU：成本数字用的是公开按需时价（AWS g5.xlarge $1.006/h），"
    "而吞吐 `tokens_per_second` 是**配置项而非实测值**——"
    "没在本机量过吞吐的盈亏平衡点不可信",
    "哈希分桶不是随机分流：`dedicated_ratio=0.05` 指的是"
    "「哈希落进前 5% 的那些 prompt」，同一句话被问一万次也只会走同一侧",
    "端到端验证用的是确定性片段判定（`must_contain`），它只能发现"
    "「该出现的东西没出现」，**发现不了「多说了不该说的」**——"
    "事实正确性与安全性仍要走 day031 的红队与 day044 的审核",
    "本课程参考模型是 557×557 的双字符 bigram（约 31 万参数），"
    "它与真实 LLM 的差距远大于任何部署方式带来的差距："
    "部署流程可复现，但结论不能外推到生产模型",
)

#: 明确排除的用途（比"不适用于生产"具体得多）。
SERVING_OUT_OF_SCOPE: tuple[str, ...] = (
    "直接按本课的价格数字做采购决策：时价随时变化，吞吐必须实测",
    "把哈希分流的切换器当 A/B 实验平台：它不是随机分流，"
    "两侧的用户构成不可比",
    "长期把 `ModelSwitcher` 留在调用链上：它的终点是 "
    "`dedicated_ratio=1.0` 之后被摘掉，永远留着的是一层没有收益的间接层",
)


@dataclass(frozen=True)
class ModelArchitecture:
    """推理时算显存需要的模型结构参数（只要四项，不要整个 config）.

    刻意**不**做成"支持全部 HF config"的容器：KV 缓存只与
    ``层数``、``KV 头数``、``head_dim`` 有关，权重只与``参数数``有关。
    多存的字段没人核对，而没人核对的字段迟早会写错。
    """

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    parameters: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ServingError("模型结构需要名字（用于报告里区分是哪一份配置）")
        if self.layers <= 0:
            raise ServingError(f"layers 必须为正整数，收到 {self.layers}")
        if self.kv_heads <= 0:
            raise ServingError(f"kv_heads 必须为正整数，收到 {self.kv_heads}")
        if self.head_dim <= 0:
            raise ServingError(f"head_dim 必须为正整数，收到 {self.head_dim}")
        if self.parameters <= 0:
            raise ServingError(f"parameters 必须为正整数，收到 {self.parameters}")

    @property
    def kv_bytes_per_token(self) -> int:
        """每 token、每序列的 KV 缓存字节数（K 与 V 各一份，因此先乘 2）."""
        return 2 * self.layers * self.kv_heads * self.head_dim * DEFAULT_KV_BYTES_PER_VALUE

    def kv_cache_bytes(self, context_length: int, max_parallel: int = DEFAULT_NUM_PARALLEL) -> int:
        """给定上下文长度与并发数时的 KV 缓存字节数.

        两个乘数都必须显式给出：``context_length`` 是因为它有强业务含义
        （4096 是 Ollama 的缺省窗口，但本课程的技术问答常常需要更长），
        ``max_parallel`` 是因为**它是最贵也最容易被忘掉的一个开关**——
        官方 FAQ 的原话是并发会把上下文与内存分配按并发数放大。
        """
        if context_length <= 0:
            raise ServingError(f"context_length 必须为正整数，收到 {context_length}")
        if max_parallel < 1:
            raise ServingError(f"max_parallel 至少为 1，收到 {max_parallel}")
        return self.kv_bytes_per_token * context_length * max_parallel

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生量）."""
        payload = asdict(self)
        payload["kv_bytes_per_token"] = self.kv_bytes_per_token
        payload["kv_kib_per_token"] = round(self.kv_bytes_per_token / 1024, 4)
        return payload


#: 本课程默认参照的模型结构。数字来自 Qwen3-8B 的公开配置：
#: 36 层、32 个注意力头 / **8 个 KV 头**（GQA）、head_dim 128、约 8.2B 参数。
#: 来源：NVIDIA 的 Qwen3-8B 模型卡（"Number of Layers: 36"、
#: "Number of Attention Heads (GQA): 32 for Q and 8 for KV"）与
#: Qwen3 技术报告（Qwen3 系列含 8B 稠密模型）。
#: **换成别的模型时必须重新填这四个数字**——KV 缓存的量级直接由层数与
#: KV 头数决定，照抄别人的值会得到一个差了数倍的显存估算。
ARCH_QWEN3_8B = ModelArchitecture(
    name="Qwen3-8B",
    layers=36,
    kv_heads=8,
    head_dim=128,
    parameters=8_200_000_000,
)


def weights_bytes(parameters: int, bits_per_parameter: float = 16.0) -> int:
    """权重体积：``参数数 × 每参数字节数``（未计入元数据与对齐）.

    只支持 ``SUPPORTED_QUANTIZATION_BITS`` 里的四档。原因很实际：
    4-bit 权重的真实占用**不是** ``参数数 / 2``（还要算量化常数、
    未量化的层与对齐），本模块刻意不算那一层——它给的是**下界**，
    而"下界"与"以为精确"是两件事，报告里会写清楚。
    """
    if parameters <= 0:
        raise ServingError(f"parameters 必须为正整数，收到 {parameters}")
    if bits_per_parameter not in SUPPORTED_QUANTIZATION_BITS:
        allowed = " / ".join(f"{item:g}" for item in SUPPORTED_QUANTIZATION_BITS)
        raise ServingError(
            f"不支持的量化位宽 {bits_per_parameter}，可选 {allowed}（qlora 的 4-bit 用 4.0）"
        )
    return int(parameters * bits_per_parameter / 8)


def memory_breakdown(
    arch: ModelArchitecture,
    *,
    context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH,
    max_parallel: int = DEFAULT_NUM_PARALLEL,
    bits_per_parameter: float = 16.0,
    adapter_bytes: int = 0,
) -> dict[str, Any]:
    """把一次部署的显存占用拆成"权重 / KV 缓存 / 适配器"三块.

    三块分开列的理由：它们**由不同的决策决定**——

    - 权重 ← 选哪个模型、量化到几位（decide once，改一次要重新下载/转换）；
    - KV 缓存 ← 上下文窗口与并发数（**每天都可以调，且调大不需要重新部署**）；
    - 适配器 ← 挂哪一个业务（换业务时唯一变动的部分）。

    合成一个总数之后，运维就只剩"显存不够"这一个信息；分开列才能回答
    "是该换更小的模型，还是该把并发从 4 调回 1"。
    """
    if adapter_bytes < 0:
        raise ServingError(f"adapter_bytes 不能为负数，收到 {adapter_bytes}")
    weights = weights_bytes(arch.parameters, bits_per_parameter)
    kv_cache = arch.kv_cache_bytes(context_length, max_parallel)
    total = weights + kv_cache + adapter_bytes
    return {
        "architecture": arch.name,
        "context_length": context_length,
        "max_parallel": max_parallel,
        "bits_per_parameter": bits_per_parameter,
        "weights_bytes": weights,
        "weights_gib": round(weights / 1024**3, 4),
        "kv_cache_bytes": kv_cache,
        "kv_cache_mib": round(kv_cache / 1024**2, 4),
        "kv_bytes_per_token": arch.kv_bytes_per_token,
        "kv_kib_per_token": round(arch.kv_bytes_per_token / 1024, 4),
        "adapter_bytes": adapter_bytes,
        "total_bytes": total,
        "total_gib": round(total / 1024**3, 4),
        # 权重的口径必须自述：它是**下界**（不含量化常数、未量化层与对齐）。
        "weights_is_lower_bound": bits_per_parameter < 16.0,
    }


@dataclass(frozen=True)
class ServingSpec:
    """一个部署单元的声明（"线上应该跑什么"）.

    ``name`` 是**部署单元名**，它必须与服务端 ``/v1/models`` 返回的 id 一致。
    这个字段不是装饰：day045 已经写过"模型名拼错是本地部署最高频的错误"，
    而多副本部署里"某一个 Pod 的服务名还是旧的"是同一类问题的运维版本——
    区别只在于拼错的不是名字，而是**你以为你部署的那一份**。
    """

    name: str
    kind: str = KIND_ADAPTER
    backend: str = OLLAMA
    base_model: str = DEFAULT_BASE_MODEL
    adapter_dir: str = ""
    merged_dir: str = ""
    context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH
    max_parallel: int = DEFAULT_NUM_PARALLEL
    bits_per_parameter: float = 16.0
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ServingError("部署单元名不能为空（它要与 /v1/models 的 id 一致）")
        if self.kind not in KINDS:
            raise ServingError(
                f"未知部署形态 {self.kind!r}，可选 {', '.join(KINDS)}"
            )
        if self.backend not in SUPPORTED_BACKENDS:
            raise ServingError(
                f"未知推理后端 {self.backend!r}，可选 {', '.join(SUPPORTED_BACKENDS)}"
            )
        if not self.base_model:
            raise ServingError("base_model 不能为空：基座不匹配不会报错，只会给出莫名其妙的结果")
        # 形态决定必填字段：**在构造期拦住"声明了形态但没给路径"的部署**。
        missing = [key for key in KIND_REQUIRED_FIELDS[self.kind] if not self._field(key)]
        if missing:
            raise ServingError(
                f"形态 {self.kind} 缺少必填字段 {', '.join(missing)}："
                f"声明了形态却没给路径的部署不会报错，它只会安静地服务别的东西"
            )
        if self.context_length <= 0:
            raise ServingError(f"context_length 必须为正整数，收到 {self.context_length}")
        if self.max_parallel < 1:
            raise ServingError(f"max_parallel 至少为 1，收到 {self.max_parallel}")
        if self.bits_per_parameter not in SUPPORTED_QUANTIZATION_BITS:
            allowed = " / ".join(f"{item:g}" for item in SUPPORTED_QUANTIZATION_BITS)
            raise ServingError(
                f"不支持的量化位宽 {self.bits_per_parameter}，可选 {allowed}"
            )

    def _field(self, key: str) -> str:
        """读取一个必填字段的字符串值（``__post_init__`` 里用，避免重复取值）."""
        return str(getattr(self, key, "") or "").strip()

    @property
    def uses_adapter(self) -> bool:
        """这个部署是否依赖运行时挂载适配器（只有 ``adapter`` 形态依赖）."""
        return self.kind == KIND_ADAPTER

    @property
    def artifact_path(self) -> str:
        """这个形态实际加载的产物路径（``base`` 形态为空串）."""
        if self.kind == KIND_ADAPTER:
            return self.adapter_dir
        if self.kind == KIND_MERGED:
            return self.merged_dir
        return ""

    @property
    def is_multi_tenant(self) -> bool:
        """是否属于"一个基座 + N 个适配器"的多业务部署（后端为 vLLM 时需要单独配置）."""
        return self.kind == KIND_ADAPTER and self.backend == VLLM

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生量）."""
        payload = asdict(self)
        payload["uses_adapter"] = self.uses_adapter
        payload["artifact_path"] = self.artifact_path
        payload["description"] = KIND_DESCRIPTIONS[self.kind]
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        artifact = self.artifact_path or "（无产物：未微调基座）"
        return (
            f"{self.name} [{self.kind}] {self.backend} | 基座 {self.base_model} | "
            f"产物 {artifact} | ctx {self.context_length} × 并发 {self.max_parallel}"
        )


def recommend_kind(*, variants: int, needs_hot_swap: bool = False) -> tuple[str, str]:
    """在 ``merged`` 与 ``adapter`` 之间给出一条**带理由**的建议.

    规则只有两条，而且两条都来自"代价在哪一侧"：

    - **只有一个业务、也不需要热插拔 → ``merged``**：部署链路与普通模型完全
      一致（不需要 peft 运行时、不需要管"哪个适配器配哪个基座"），代价是要多存
      一份完整权重——而只有一个业务时那份权重本来就只有一份；
    - **其他情况 → ``adapter``**：第二个业务开始，"一个基座 + N 个适配器"省下的
      是 N−1 份完整权重，而多出来的复杂度（运行时挂载 + 基座/适配器配对校验）
      由 day052 的清单与今天的 ``binding`` 模块承担。

    ``needs_hot_swap`` 单独列出来，是因为它会在**只有一个业务时**也推翻第一条：
    需要毫秒级切回基座做对照实验时，``merged`` 形态要重新加载一份完整权重。
    """
    if variants < 0:
        raise ServingError(f"variants 不能为负数，收到 {variants}")
    if variants <= 1 and not needs_hot_swap:
        return (
            KIND_MERGED,
            "只有一个业务且不需要热插拔：合并形态的部署链路与普通模型一致，"
            "而完整权重本来就只有一份",
        )
    if needs_hot_swap:
        return (
            KIND_ADAPTER,
            "需要热插拔：适配器形态换业务只换十几 MB，合并形态要重新加载完整权重",
        )
    return (
        KIND_ADAPTER,
        f"有 {variants} 个业务：一个基座 + N 个适配器省下 {variants - 1} 份完整权重，"
        "代价由清单与绑定校验承担",
    )


def spec_table() -> list[dict[str, Any]]:
    """三种形态的对照表（API 的自我描述端点与文档同源）.

    表里带 ``required_fields`` 一列：**它是这张表里最容易被漏掉、也最容易
    出事的一列**——"声明成 adapter 却没给路径"正是靠它才在构造期被拦住。
    """
    rows: list[dict[str, Any]] = []
    for kind in KINDS:
        required = KIND_REQUIRED_FIELDS[kind]
        rows.append(
            {
                "kind": kind,
                "description": KIND_DESCRIPTIONS[kind],
                "required_fields": list(required),
                "required_fields_text": ", ".join(required) or "（无）",
                "runtime_dependency": (
                    "需要 peft 在推理时挂载适配器" if kind == KIND_ADAPTER else "不需要额外运行时"
                ),
                "switch_cost": (
                    "换业务只换十几 MB 的适配器文件"
                    if kind == KIND_ADAPTER
                    else "换业务要换一份与基座同量级的完整权重"
                    if kind == KIND_MERGED
                    else "（无业务可换）"
                ),
            }
        )
    return rows


__all__ = [
    "ARCH_QWEN3_8B",
    "DEFAULT_KV_BYTES_PER_VALUE",
    "DEFAULT_NUM_PARALLEL",
    "KINDS",
    "KIND_ADAPTER",
    "KIND_BASE",
    "KIND_DESCRIPTIONS",
    "KIND_MERGED",
    "KIND_REQUIRED_FIELDS",
    "OLLAMA",
    "SERVING_LIMITATIONS",
    "SERVING_OUT_OF_SCOPE",
    "SUPPORTED_QUANTIZATION_BITS",
    "ModelArchitecture",
    "ServingSpec",
    "memory_breakdown",
    "recommend_kind",
    "spec_table",
    "weights_bytes",
]
