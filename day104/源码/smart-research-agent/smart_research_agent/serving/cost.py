"""成本对比：把"自建更便宜"这句话拆成两个不同的成本模型（M5-D11）.

"专属模型部署到本地能省钱"是一句**没有主语就不成立**的话。本模块要做的事情
只有一件：把两侧的成本写成各自的式子，然后让数字说话——

```text
云端 API   = 可变成本 × 请求数      不调用就不花钱，调用越多花越多
自建推理   = 固定成本 × 开机时长    开机就在花钱，与调用量无关（吞吐上限之内）
```

两行式子合起来就能解释一件反直觉的事：**空转的 GPU 是最贵的**。
一台 A10G 实例按需 $1.006/小时（AWS g5.xlarge，us-east-1），
无论你这小时处理了 10 个请求还是 10000 个请求，账单都是 $1.006。
利用率 10% 时，单位成本正好是满载时的 10 倍。

因此本模块输出的不是一句"更便宜"，而是四个数：

| 输出 | 回答的问题 |
|------|-----------|
| ``cloud_cost_usd`` / ``dedicated_cost_usd`` | 这批请求量下，两侧各花多少 |
| ``required_gpu_hours`` vs ``deployed_gpu_hours`` | **用掉的算力**与**为之付费的时长**差多少 |
| ``utilization_actual`` | 这台机器是"在工作"还是"在待机" |
| ``breakeven_requests`` | 调用量要到多少，自建才真正更省 |

## 价格数字的来源（2026-09 核对，随时可能变）

价格是**外部事实**，本模块把它们集中放在 ``GPU_HOURLY_USD`` 与
``CLOUD_PRICING`` 两个常量里，并注明出处——把价格散落在函数体里，
会让"这个 $1.006 是哪来的"变成一次考古。

```text
AWS g5.xlarge（A10G 24GB，us-east-1）按需 $1.006/h
AWS g4dn.xlarge（T4 16GB，us-east-1）按需 $0.526/h
GCP NVIDIA T4（us-central1）GPU 附加费 $0.35/GPU/h（**不含 VM 机器类型费用**）
RunPod Pods 的 L4 24GB $0.49/h
Lambda 的 A10 24GB $1.29/GPU/h
gpt-4o-mini 输入 $0.15/1M tokens、输出 $0.60/1M tokens
```

**吞吐（``tokens_per_second``）刻意没有缺省值**，必须由调用方给出。
理由很直接：吞吐取决于模型、量化、批大小与序列长度，公开资料给不出
"你这套配置"的数字。本模块的存在意义就是强迫把它填成一个**测出来的数**，
而不是从别人的博客里抄一个看起来合理的数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from smart_research_agent.serving.errors import ServingError
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 一个月的平均小时数：``365 × 24 / 12 = 730``。
#: 用 730 而不是 30×24=720：月账单按自然月折算，720 会让年化成本少算 1.4%。
HOURS_PER_MONTH = 730.0

SECONDS_PER_HOUR = 3600.0
TOKENS_PER_1K = 1000.0

#: GPU 按需小时价（美元）。每条都注明了机型与区域，**来源写在模块 docstring 里**。
GPU_HOURLY_USD: dict[str, float] = {
    "aws-g5-xlarge-a10g": 1.006,
    "aws-g4dn-xlarge-t4": 0.526,
    "gcp-t4-gpu-only": 0.35,
    "runpod-pod-l4": 0.49,
    "lambda-a10": 1.29,
}

#: 云端 API 的公开价（美元 / 千 token），2026-09 核对。
CLOUD_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}

#: 成本对比结果的两种取值（``cheaper_side``）.
SIDE_CLOUD = "cloud"
SIDE_DEDICATED = "dedicated"
SIDE_TIE = "tie"


@dataclass(frozen=True)
class CloudPricing:
    """云端 API 的计价：输入与输出**分开**报价.

    分开是必须的：两者的单价常差 4 倍（gpt-4o-mini 是 $0.15 vs $0.60 / 1M），
    而"一次请求平均多少输入 token、多少输出 token"也正是 RAG 与 Agent 场景里
    差异最大的一个参数（检索把输入撑到几千 token，输出可能只有几十 token）。
    合成一个"平均单价"会让这份对比失去可调性。
    """

    name: str
    usd_per_1k_input_tokens: float
    usd_per_1k_output_tokens: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ServingError("计价方案需要名字（报告里要写清按谁的价格算的）")
        for field_name in ("usd_per_1k_input_tokens", "usd_per_1k_output_tokens"):
            if getattr(self, field_name) < 0:
                raise ServingError(f"{field_name} 不能为负数，收到 {getattr(self, field_name)}")

    def cost_usd(self, input_tokens: float, output_tokens: float) -> float:
        """按 token 数算一次（或一批）调用的费用."""
        if input_tokens < 0 or output_tokens < 0:
            raise ServingError("token 数不能为负数（token 是计数，不是可正可负的量）")
        return (
            input_tokens / TOKENS_PER_1K * self.usd_per_1k_input_tokens
            + output_tokens / TOKENS_PER_1K * self.usd_per_1k_output_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)


@dataclass(frozen=True)
class GpuPricing:
    """自建推理的计价：时价 × 开机时长，**吞吐只决定"要开几台"**.

    ``utilization`` 是**你实际能用上的算力比例**：批处理调度、长尾序列、
    空闲间隙都会吃掉它。把它显式列出来，是因为它是最容易被默认成 1.0
    然后忘掉的一个字段——而 1.0 的含义恰好是"这台机器从不空转"。
    """

    name: str
    usd_per_hour: float
    tokens_per_second: float
    utilization: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ServingError("GPU 方案需要名字（报告里要写清按哪台的时价算的）")
        if self.usd_per_hour <= 0:
            raise ServingError(f"usd_per_hour 必须为正数，收到 {self.usd_per_hour}")
        if self.tokens_per_second <= 0:
            raise ServingError(
                f"tokens_per_second 必须为正数，收到 {self.tokens_per_second}："
                "吞吐没有缺省值——它取决于模型、量化、批大小与序列长度，必须实测"
            )
        if not 0.0 < self.utilization <= 1.0:
            raise ServingError(f"utilization 必须落在 (0, 1]，收到 {self.utilization}")

    @property
    def effective_tokens_per_second(self) -> float:
        """计入利用率后的有效吞吐（token/s）."""
        return self.tokens_per_second * self.utilization

    def hours_for(self, total_tokens: float) -> float:
        """处理这些 token **真正需要**的 GPU 小时数."""
        if total_tokens < 0:
            raise ServingError("总 token 数不能为负数")
        return total_tokens / (self.effective_tokens_per_second * SECONDS_PER_HOUR)

    def cost_usd(self, gpu_hours: float) -> float:
        """按 GPU 小时数算费用（``gpu_hours`` 是**付费时长**，不是用掉的算力时长）."""
        if gpu_hours < 0:
            raise ServingError("GPU 小时数不能为负数")
        return gpu_hours * self.usd_per_hour

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含有效吞吐）."""
        payload = asdict(self)
        payload["effective_tokens_per_second"] = self.effective_tokens_per_second
        return payload


@dataclass(frozen=True)
class CostComparison:
    """一次成本对比的完整结果（每一列都能被单独核对）."""

    requests: int
    input_tokens_per_request: float
    output_tokens_per_request: float
    monthly_hours: float
    cloud: CloudPricing
    gpu: GpuPricing
    total_tokens: float
    cloud_cost_usd: float
    cloud_cost_per_request: float
    dedicated_cost_usd: float
    required_gpu_hours: float
    deployed_gpu_hours: float
    machines: int
    utilization_actual: float
    breakeven_requests: float
    cheaper_side: str

    @property
    def saving_usd(self) -> float:
        """专属相对云端省下的钱（**为负表示更贵**）."""
        return self.cloud_cost_usd - self.dedicated_cost_usd

    @property
    def saving_ratio(self) -> float | None:
        """节省比例；云端成本为 0 时返回 ``None``（**除零不是 0**）."""
        if self.cloud_cost_usd <= 0:
            return None
        return self.saving_usd / self.cloud_cost_usd

    @property
    def dedicated_cost_per_1k_tokens(self) -> float | None:
        """专属侧的单位成本（美元/千 token）；无 token 时为 ``None``.

        这个数才是"自建到底便不便宜"的答案：它与云端单价可以**直接比大小**，
        而总费用不能（两者对应的请求量不同）。
        """
        if self.total_tokens <= 0:
            return None
        return self.dedicated_cost_usd / (self.total_tokens / TOKENS_PER_1K)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        ratio = self.saving_ratio
        per_1k = self.dedicated_cost_per_1k_tokens
        return {
            "requests": self.requests,
            "input_tokens_per_request": self.input_tokens_per_request,
            "output_tokens_per_request": self.output_tokens_per_request,
            "monthly_hours": self.monthly_hours,
            "total_tokens": round(self.total_tokens, 4),
            "cloud": self.cloud.to_dict(),
            "gpu": self.gpu.to_dict(),
            "cloud_cost_usd": round(self.cloud_cost_usd, 6),
            "cloud_cost_per_request": round(self.cloud_cost_per_request, 8),
            "dedicated_cost_usd": round(self.dedicated_cost_usd, 6),
            "dedicated_cost_per_1k_tokens": None if per_1k is None else round(per_1k, 8),
            "required_gpu_hours": round(self.required_gpu_hours, 6),
            "deployed_gpu_hours": round(self.deployed_gpu_hours, 6),
            "machines": self.machines,
            "utilization_actual": round(self.utilization_actual, 6),
            "saving_usd": round(self.saving_usd, 6),
            "saving_ratio": None if ratio is None else round(ratio, 6),
            "breakeven_requests": round(self.breakeven_requests, 4),
            "cheaper_side": self.cheaper_side,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        side = {
            SIDE_CLOUD: "云端更省",
            SIDE_DEDICATED: "自建更省",
            SIDE_TIE: "两侧持平",
        }[self.cheaper_side]
        return (
            f"{self.requests} 请求 | 云端 ${self.cloud_cost_usd:.4f} vs "
            f"自建 ${self.dedicated_cost_usd:.4f} → {side}"
            f"（省 ${self.saving_usd:+.4f}）| {self.machines} 台 × "
            f"{self.monthly_hours:g}h | 利用率 {self.utilization_actual:.2%} | "
            f"盈亏平衡 {self.breakeven_requests:.0f} 请求"
        )

    def render_markdown(self) -> str:
        """把对比渲染成 markdown（可直接贴进技术选型文档）."""
        ratio = self.saving_ratio
        per_1k = self.dedicated_cost_per_1k_tokens
        # 两边的"单位成本"必须用同一个口径：**总费用 ÷ 总 token**。
        # 云端侧刻意不写成"输入单价 + 输出单价"，因为那会把两种不同比例的
        # 报价相加——一个 4 倍价差的输入输出比会让那个和失去意义。
        cloud_per_1k = (
            None if self.total_tokens <= 0 else self.cloud_cost_usd / (self.total_tokens / 1000)
        )
        per_request = self.dedicated_cost_usd / self.requests if self.requests else None
        conclusion = {
            SIDE_CLOUD: "云端更省",
            SIDE_DEDICATED: "自建更省",
            SIDE_TIE: "两侧持平",
        }[self.cheaper_side]
        return "\n".join(
            [
                "# 专属模型部署成本对比",
                "",
                f"- 云端计价：`{self.cloud.name}`"
                f"（输入 ${self.cloud.usd_per_1k_input_tokens}/1K，"
                f"输出 ${self.cloud.usd_per_1k_output_tokens}/1K）",
                f"- 自建计价：`{self.gpu.name}`"
                f"（${self.gpu.usd_per_hour}/h，吞吐 "
                f"{self.gpu.effective_tokens_per_second:g} token/s，"
                f"利用率 {self.gpu.utilization:.0%}）",
                f"- 请求量：{self.requests}（{self.input_tokens_per_request:g} 入 + "
                f"{self.output_tokens_per_request:g} 出 = "
                f"{self.input_tokens_per_request + self.output_tokens_per_request:g} token/次）",
                f"- 月度开机时长：{self.monthly_hours:g}h",
                "",
                "| 项 | 云端 | 自建 |",
                "|----|------|------|",
                f"| 月度费用 | ${self.cloud_cost_usd:.4f} | ${self.dedicated_cost_usd:.4f} |",
                f"| 单次请求 | "
                f"{'—' if not self.requests else f'${self.cloud_cost_per_request:.8f}'} | "
                f"{'—' if per_request is None else f'${per_request:.8f}'} |",
                f"| 单位成本（$/1K token） | "
                f"{'—' if cloud_per_1k is None else f'${cloud_per_1k:.8f}'} | "
                f"{'—' if per_1k is None else f'${per_1k:.8f}'} |",
                "",
                f"- 用掉的算力：{self.required_gpu_hours:.4f} GPU·h",
                f"- 为之付费的时长：{self.deployed_gpu_hours:.4f} GPU·h"
                f"（{self.machines} 台 × {self.monthly_hours:g}h）",
                f"- 实际利用率：{self.utilization_actual:.2%}",
                f"- 盈亏平衡调用量：{self.breakeven_requests:.0f} 请求/月",
                f"- 结论：{conclusion}" + ("" if ratio is None else f"（节省 {ratio:.2%}）"),
                "",
            ]
        )


def gpu_pricing(
    key: str,
    *,
    tokens_per_second: float,
    utilization: float = 1.0,
) -> GpuPricing:
    """按 ``GPU_HOURLY_USD`` 里的机型键构造一个 GPU 计价方案.

    ``tokens_per_second`` 必须由调用方给出（见模块 docstring 的理由）。
    未知的键会抛错并列出全部可选项——**"你写的机型不在价格表里"必须响亮**，
    否则会用一个 0 元时价算出一份"自建完全免费"的报告。
    """
    if key not in GPU_HOURLY_USD:
        raise ServingError(
            f"未知的 GPU 机型 {key!r}，可选 {', '.join(sorted(GPU_HOURLY_USD))}"
        )
    return GpuPricing(
        name=key,
        usd_per_hour=GPU_HOURLY_USD[key],
        tokens_per_second=tokens_per_second,
        utilization=utilization,
    )


def cloud_pricing(key: str = "gpt-4o-mini") -> CloudPricing:
    """按 ``CLOUD_PRICING`` 里的模型键构造一个云端计价方案."""
    if key not in CLOUD_PRICING:
        raise ServingError(
            f"未知的云端计价方案 {key!r}，可选 {', '.join(sorted(CLOUD_PRICING))}"
        )
    input_price, output_price = CLOUD_PRICING[key]
    return CloudPricing(
        name=key,
        usd_per_1k_input_tokens=input_price,
        usd_per_1k_output_tokens=output_price,
    )


def breakeven_requests(
    cloud: CloudPricing,
    gpu: GpuPricing,
    *,
    input_tokens_per_request: float,
    output_tokens_per_request: float,
    monthly_hours: float = HOURS_PER_MONTH,
    machines: int = 1,
) -> float:
    """算出自建与云端**成本相等**时的月度请求量.

    推导只有两行：

    ```text
    自建月费 = machines × monthly_hours × usd_per_hour          （固定，与请求量无关）
    云端月费 = requests × cloud.cost_usd(in, out)               （可变，随请求量线性）
    相等时： requests = 自建月费 / 单次云端费用
    ```

    返回值可能大得离谱——那正是结论：**在请求量远低于它的时候，"自建省钱"
    是一个错觉，而那台机器的账单一分不少。**

    ## 它与"单位成本之比"是同一个数

    把 ``machines × monthly_hours × usd_per_hour`` 代进自建的单位成本
    （``dedicated_cost_per_1k_tokens``），可以证明一条很有用的性质：

    ```text
    breakeven_requests / 当前请求量 ≈ 自建单位成本 / 云端单位成本
    ```

    实测（本课缺省参数）：10 万请求/月时自建 $0.00367/1K vs 云端 $0.0002625/1K，
    比值 **13.99**；而盈亏平衡点 1398819 ÷ 100000 = **13.99**。
    于是"盈亏平衡点是我现在请求量的 14 倍"与"我的单位成本是云端的 14 倍"
    是同一句话的两种说法——**两个数字互相校验**，任何一个算错都会立刻暴露。
    """
    if machines < 1:
        raise ServingError(f"machines 至少为 1，收到 {machines}")
    if monthly_hours <= 0:
        raise ServingError(f"monthly_hours 必须为正数，收到 {monthly_hours}")
    per_request = cloud.cost_usd(input_tokens_per_request, output_tokens_per_request)
    if per_request <= 0:
        # 云端单价为 0 时"平衡点"在无穷远处：自建的固定成本无法被任何请求量摊平。
        return float("inf")
    return machines * monthly_hours * gpu.usd_per_hour / per_request


def compare_costs(
    *,
    requests: int,
    input_tokens_per_request: float,
    output_tokens_per_request: float,
    cloud: CloudPricing,
    gpu: GpuPricing,
    monthly_hours: float = HOURS_PER_MONTH,
) -> CostComparison:
    """把一批月度请求量放到两个成本模型上，给出四个可核对的数字.

    关键的一步是 ``machines``：**用掉的算力时长超过一台机器能提供的时长时，
    就得开第二台**——而第二台同样是按月付费。这一步把"自建 = 一台机器"
    这个隐性假设显式化了，否则会出现"用一台机器的时价，承接十台机器的负载"
    这种算得出来但部署不出来的方案。

    ``breakeven_requests`` 用的正是这批请求所需的机器数，因此它**与当前
    请求量同尺度可比**：``breakeven / requests`` 恰好等于
    ``自建单位成本 / 云端单位成本``（推导见 ``breakeven_requests``）。
    报告里两个数字同时给出，就是为了让它们互相校验——
    **一个能解释另一个的数字，才值得被相信。**
    """
    if requests < 0:
        raise ServingError(f"requests 不能为负数，收到 {requests}")
    if input_tokens_per_request < 0 or output_tokens_per_request < 0:
        raise ServingError("每请求 token 数不能为负数")
    if monthly_hours <= 0:
        raise ServingError(f"monthly_hours 必须为正数，收到 {monthly_hours}")

    total_tokens = requests * (input_tokens_per_request + output_tokens_per_request)
    cloud_cost = cloud.cost_usd(
        requests * input_tokens_per_request, requests * output_tokens_per_request
    )
    required_hours = gpu.hours_for(total_tokens)
    # 向上取整：开第二台的那一刻就按整月付费（按需实例按小时计费，而月是
    # "这个月要不要保有这些机器"的决策周期）。
    machines = max(1, int(-(-required_hours // monthly_hours)) if required_hours > 0 else 1)
    deployed_hours = machines * monthly_hours
    dedicated_cost = gpu.cost_usd(deployed_hours)
    utilization_actual = required_hours / deployed_hours if deployed_hours > 0 else 0.0
    difference = cloud_cost - dedicated_cost
    if abs(difference) < 1e-12:
        side = SIDE_TIE
    else:
        side = SIDE_CLOUD if difference < 0 else SIDE_DEDICATED

    comparison = CostComparison(
        requests=requests,
        input_tokens_per_request=input_tokens_per_request,
        output_tokens_per_request=output_tokens_per_request,
        monthly_hours=monthly_hours,
        cloud=cloud,
        gpu=gpu,
        total_tokens=total_tokens,
        cloud_cost_usd=cloud_cost,
        cloud_cost_per_request=cloud_cost / requests if requests else 0.0,
        dedicated_cost_usd=dedicated_cost,
        required_gpu_hours=required_hours,
        deployed_gpu_hours=deployed_hours,
        machines=machines,
        utilization_actual=utilization_actual,
        breakeven_requests=breakeven_requests(
            cloud,
            gpu,
            input_tokens_per_request=input_tokens_per_request,
            output_tokens_per_request=output_tokens_per_request,
            monthly_hours=monthly_hours,
            machines=machines,
        ),
        cheaper_side=side,
    )
    logger.info("成本对比：%s", comparison.summary_line())
    return comparison


def price_book() -> list[dict[str, Any]]:
    """把两张价格表摊开（API 的自我描述端点与文档同源）.

    价格是外部事实，因此它必须**只有一个出处**：``GPU_HOURLY_USD`` 与
    ``CLOUD_PRICING``。这个函数把它们渲染成可读的行，让"文档里的 $1.006"
    与"代码里的 1.006"永远是同一个数字。
    """
    rows: list[dict[str, Any]] = [
        {
            "kind": "gpu",
            "name": key,
            "unit": "USD/GPU·h",
            "value": value,
            "note": "按需实例时价；不含 VM 机型费用（gcp-t4-gpu-only 为纯 GPU 附加费）",
        }
        for key, value in sorted(GPU_HOURLY_USD.items())
    ]
    rows.extend(
        {
            "kind": "cloud",
            "name": key,
            "unit": "USD/1K tokens",
            # 四舍五入到 8 位是为了不让 ``0.00015 + 0.0006`` 打印成
            # ``0.0007499999999999999``——**一个看起来像 bug 的数字会让人
            # 去查代码，而它其实是浮点加法的正常结果**。
            "value": round(value[0] + value[1], 8),
            "note": f"输入 {value[0]} / 输出 {value[1]}（分开计价，不要用合计值算钱）",
        }
        for key, value in sorted(CLOUD_PRICING.items())
    )
    return rows


__all__ = [
    "CLOUD_PRICING",
    "GPU_HOURLY_USD",
    "HOURS_PER_MONTH",
    "SECONDS_PER_HOUR",
    "SIDE_CLOUD",
    "SIDE_DEDICATED",
    "SIDE_TIE",
    "TOKENS_PER_1K",
    "CloudPricing",
    "CostComparison",
    "GpuPricing",
    "breakeven_requests",
    "cloud_pricing",
    "compare_costs",
    "gpu_pricing",
    "price_book",
]
