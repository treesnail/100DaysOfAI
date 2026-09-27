"""本地部署的运维层（day045）：进程/模型生命周期管理与显存预算.

``LocalModel``（``llm/local_model.py``）解决的是"怎么调"，本模块解决的是
"**能不能跑起来**、跑起来之后怎么管"：

1. **显存预算**（``estimate_vram_gb`` / ``plan_deployment``）：本地部署的
   第一性问题不是代码而是硬件——7B 模型 FP16 要 14GB 显存，Q4 量化只要
   约 4GB，选错量化直接 OOM。估算公式来自通行的推理显存口径：

      权重显存 = 参数量 × 每参数字节数
      KV 缓存  = 2 × batch × 序列长度 × 层数 × KV 头数 × head_dim × KV 精度字节数

   其中"2"是 Key 与 Value 两份张量；``num_kv_heads`` 是 GQA/MQA 架构里的
   KV 头数（Llama/Qwen 系列远小于注意力头数，这也是长上下文能省显存的
   关键），``head_dim`` 是单头维度。再乘一个 overhead 系数覆盖推理框架的
   激活值、CUDA context 与碎片——所以估算值只作"能不能装下"的判断门槛，
   不作精确承诺。

2. **模型生命周期**（``OllamaRuntime``）：Ollama 的原生 REST API 提供了
   OpenAI 兼容端点没有的能力——列出本地已有模型（``/api/tags``）、列出
   正在运行的模型（``/api/ps``）、拉取模型（``/api/pull``）、以及控制模型
   在显存里的常驻时长（``keep_alive``）。生产上这一层是"冷启动优化"的
   抓手：定时用 ``keep_alive: -1`` 预热常驻模型，把首字延迟从"加载权重
   的十几秒"压到"纯推理的几百毫秒"。

HTTP 细节用标准库 ``urllib.request`` 实现（不引入新依赖），并通过
``transport`` 参数注入——测试传入桩函数即可完全离线地验证每一个请求的
URL 与 payload，无需真的装 Ollama。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from smart_research_agent.llm.local_model import (
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    default_base_url,
    normalize_base_url,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 1 GiB 的字节数（显存口径按 1024 进制）
BYTES_PER_GIB = 1024**3

#: 常见量化档位的**每参数位数**（bits per weight, bpw）.
#: fp16/bf16 = 16、int8 = 8、int4 = 4 是理论值；GGUF 的 k-quant 是混合
#: 量化（不同张量用不同位宽），工程上按平均值计：Q8_0 ≈ 8.5、Q5_K_M ≈ 5.5、
#: Q4_K_M ≈ 4.5 bpw——Q4_K_M 是"质量损失可接受 + 显存几乎减半"的常用默认档。
QUANTIZATION_BITS: dict[str, float] = {
    "fp32": 32.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "int8": 8.0,
    "q8_0": 8.5,
    "q5_k_m": 5.5,
    "q4_k_m": 4.5,
    "int4": 4.0,
}

#: 质量由高到低的量化顺序，用于"显存不够时降级到哪一档"的建议
QUANTIZATION_ORDER: tuple[str, ...] = ("fp16", "q8_0", "q5_k_m", "q4_k_m")

#: 传输函数签名：``(url, payload)``，payload 为 None 表示 GET，否则 POST JSON
Transport = Callable[[str, "dict[str, Any] | None"], dict[str, Any]]


def urllib_transport(url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """标准库实现的极简 JSON 传输：payload 为 None 走 GET，否则 POST.

    只做"把 JSON 发出去、把 JSON 收回来"这一件事——足以覆盖 Ollama 原生
    API 的全部用法，且不给项目增加 httpx/requests 依赖。生产上若要复用
    连接池与重试策略，替换本函数即可（``transport`` 参数就是为此留的口子）。
    """
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - 本地回环地址
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - 真实网络分支
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} 返回 {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - 真实网络分支
        raise RuntimeError(f"{method} {url} 连接失败: {exc.reason}") from exc
    return json.loads(body) if body else {}


def bytes_per_parameter(quant: str) -> float:
    """量化的每参数字节数（未知档位抛 ValueError，附带可用档位）."""
    key = (quant or "").strip().lower()
    if key not in QUANTIZATION_BITS:
        raise ValueError(
            f"未知量化档位 {quant}，可选值: {' / '.join(QUANTIZATION_BITS)}"
        )
    return QUANTIZATION_BITS[key] / 8.0


def estimate_vram_gb(
    params_b: float,
    *,
    quant: str = "q4_k_m",
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH,
    batch_size: int = 1,
    kv_dtype_bytes: int = 2,
    overhead_ratio: float = 0.1,
) -> tuple[float, float, float]:
    """估算部署所需显存，返回 ``(权重 GB, KV 缓存 GB, 含开销总 GB)``.

    参数 ``params_b`` 是参数量（单位：十亿 / B），例如 Qwen3-8B 传 8.0。
    ``num_layers`` / ``kv_heads`` / ``head_dim`` 取自模型 config.json
    （``num_hidden_layers`` / ``num_key_value_heads`` / ``head_dim``）——
    之所以要求显式传入而不是内置一张模型表，是因为架构参数随模型版本变化，
    猜错会让估算失去意义；宁可让调用方抄一次 config。
    """
    if params_b <= 0:
        raise ValueError("params_b 必须为正数")
    if num_layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("模型架构参数（层数/KV头数/head_dim）必须为正整数")
    if context_length <= 0 or batch_size <= 0:
        raise ValueError("context_length 与 batch_size 必须为正整数")
    if kv_dtype_bytes <= 0:
        raise ValueError("kv_dtype_bytes 必须为正整数")
    if overhead_ratio < 0:
        raise ValueError("overhead_ratio 不能为负数")

    weights_bytes = params_b * 1e9 * bytes_per_parameter(quant)
    kv_bytes = (
        2
        * batch_size
        * context_length
        * num_layers
        * kv_heads
        * head_dim
        * kv_dtype_bytes
    )
    weights_gb = weights_bytes / BYTES_PER_GIB
    kv_gb = kv_bytes / BYTES_PER_GIB
    total_gb = (weights_gb + kv_gb) * (1 + overhead_ratio)
    return weights_gb, kv_gb, total_gb


@dataclass
class HardwarePlan:
    """一次"模型 × 量化 × 显卡"组合的部署可行性结论."""

    params_b: float
    quant: str
    context_length: int
    weights_gb: float
    kv_cache_gb: float
    total_gb: float
    gpu_vram_gb: float

    @property
    def fits(self) -> bool:
        """总需求（含开销）是否落在可用显存内."""
        return self.total_gb <= self.gpu_vram_gb

    @property
    def headroom_gb(self) -> float:
        """余量：负数表示还差多少显存."""
        return round(self.gpu_vram_gb - self.total_gb, 3)

    def recommendation(self) -> str:
        """给出可直接执行的下一步建议（装得下 / 装不下各一条）."""
        if self.fits:
            return (
                f"可部署：{self.params_b}B 模型以 {self.quant} 运行，"
                f"预计占用 {self.total_gb:.2f} GB（权重 {self.weights_gb:.2f} + "
                f"KV {self.kv_cache_gb:.2f}），剩余 {self.headroom_gb:.2f} GB"
            )
        return (
            f"装不下：当前组合需 {self.total_gb:.2f} GB，超出显存 "
            f"{abs(self.headroom_gb):.2f} GB；建议降低量化档位或缩短上下文窗口"
            f"（当前 {self.context_length} token）"
        )


def plan_deployment(
    params_b: float,
    gpu_vram_gb: float,
    *,
    quant: str = "q4_k_m",
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH,
    batch_size: int = 1,
    kv_dtype_bytes: int = 2,
    overhead_ratio: float = 0.1,
) -> HardwarePlan:
    """生成一份部署可行性结论（教程里"先算再跑"的那一步）."""
    if gpu_vram_gb <= 0:
        raise ValueError("gpu_vram_gb 必须为正数")
    weights_gb, kv_gb, total_gb = estimate_vram_gb(
        params_b,
        quant=quant,
        num_layers=num_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        context_length=context_length,
        batch_size=batch_size,
        kv_dtype_bytes=kv_dtype_bytes,
        overhead_ratio=overhead_ratio,
    )
    return HardwarePlan(
        params_b=params_b,
        quant=quant,
        context_length=context_length,
        weights_gb=round(weights_gb, 3),
        kv_cache_gb=round(kv_gb, 3),
        total_gb=round(total_gb, 3),
        gpu_vram_gb=gpu_vram_gb,
    )


def fitting_quantizations(
    params_b: float,
    gpu_vram_gb: float,
    *,
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH,
    kv_dtype_bytes: int = 2,
) -> list[str]:
    """返回"在该显存下装得下"的量化档位，按质量从高到低排列.

    这是"降级路径"的依据：装不下 fp16 时，第一个能装下的档位就是建议目标。
    """
    fitted: list[str] = []
    for quant in QUANTIZATION_ORDER:
        plan = plan_deployment(
            params_b,
            gpu_vram_gb,
            quant=quant,
            num_layers=num_layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            context_length=context_length,
            kv_dtype_bytes=kv_dtype_bytes,
        )
        if plan.fits:
            fitted.append(quant)
    return fitted


def max_context_tokens(
    params_b: float,
    gpu_vram_gb: float,
    *,
    quant: str = "q4_k_m",
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    kv_dtype_bytes: int = 2,
    overhead_ratio: float = 0.1,
    step: int = 1024,
) -> int:
    """在给定显存下能开多长的上下文（按 ``step`` token 向下取整）.

    思路是把 KV 缓存的线性公式反解：先扣掉权重与开销预算，剩下的都给
    KV 缓存。返回 0 表示连权重都装不下（任何长度都跑不起来）。
    """
    if step <= 0:
        raise ValueError("step 必须为正整数")
    weights_bytes = params_b * 1e9 * bytes_per_parameter(quant)
    budget_bytes = gpu_vram_gb * BYTES_PER_GIB / (1 + overhead_ratio) - weights_bytes
    bytes_per_token = 2 * batch_size * num_layers * kv_heads * head_dim * kv_dtype_bytes
    if budget_bytes <= 0 or bytes_per_token <= 0:  # pragma: no cover - 防御式分支
        return 0
    tokens = int(budget_bytes // bytes_per_token)
    return (tokens // step) * step


def native_base_url(base_url: str | None = None) -> str:
    """把 OpenAI 兼容 base_url 还原成 Ollama 原生 API 的根地址.

    Ollama 的原生接口挂在 ``http://host:11434`` 上，而 OpenAI 兼容接口挂在
    ``http://host:11434/v1`` 上：同一进程两套路径。本函数负责在两者之间
    转换（去掉尾部 ``/v1``），避免运维层里散落字符串拼接。
    """
    normalized = normalize_base_url(base_url or default_base_url("ollama"), "ollama")
    parsed = urlparse(normalized)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


class OllamaRuntime:
    """Ollama 原生 API 的运维封装：模型清单、常驻控制、拉取.

    每个方法对应一个文档化的原生端点，且都通过 ``transport`` 发起请求——
    把"怎么发 HTTP"与"发什么"解耦，测试只需桩函数即可断言 URL 与 payload。

    ``keep_alive`` 语义（Ollama 文档）：
      - ``"5m"``：默认值，请求结束后模型在显存里再待 5 分钟；
      - ``"1h"`` / ``-1``：长常驻 / 永不卸载（预热常驻模型）；
      - ``0``：请求返回后立即卸载（用完即走的省显存策略）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: Transport | None = None,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
    ):
        self.base_url = native_base_url(base_url)
        self._transport = transport or urllib_transport
        self.keep_alive = keep_alive

    def _post(self, path: str, payload: dict[str, Any], failure: str) -> dict[str, Any]:
        """POST 并统一错误语义；``failure`` 是完整的中文失败描述（含模型名）."""
        url = f"{self.base_url}{path}"
        try:
            result = self._transport(url, payload)
        except Exception as exc:  # noqa: BLE001 - 统一转成运维语义的错误
            raise RuntimeError(f"{failure}（{url}）: {exc}") from exc
        if not isinstance(result, dict):  # pragma: no cover - 防御式分支
            raise RuntimeError(f"{failure}（{url}）: 响应不是 JSON 对象")
        return result

    def _get(self, path: str, failure: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            result = self._transport(url, None)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{failure}（{url}）: {exc}") from exc
        if not isinstance(result, dict):  # pragma: no cover - 防御式分支
            raise RuntimeError(f"{failure}（{url}）: 响应不是 JSON 对象")
        return result

    def list_models(self) -> list[str]:
        """列出本机已 pull 的模型（``GET /api/tags``）."""
        data = self._get("/api/tags", "列出本地模型失败")
        return [str(item.get("name", "")) for item in data.get("models") or []]

    def list_running(self) -> list[dict[str, Any]]:
        """列出当前常驻显存的模型（``GET /api/ps``），含 ``expires_at``."""
        data = self._get("/api/ps", "列出运行中模型失败")
        return [
            {"name": str(item.get("name", "")), "expires_at": item.get("expires_at")}
            for item in data.get("models") or []
        ]

    def is_running(self, name: str) -> bool:
        """指定模型当前是否常驻显存."""
        return any(item["name"] == name for item in self.list_running())

    def load_model(
        self,
        name: str,
        *,
        keep_alive: str | int | None = None,
        context_length: int | None = None,
    ) -> dict[str, Any]:
        """预热加载模型（``POST /api/generate``，只给 model、不给 prompt）.

        文档给出的预热方式就是"发一个只有 model 字段的 generate 请求"：
        Ollama 会加载权重并立即返回，不产生任何 token。``context_length``
        通过 ``options.num_ctx`` 下发——这是把 Ollama 默认的 4096 调大的
        唯一入口。
        """
        payload: dict[str, Any] = {
            "model": name,
            "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
        }
        if context_length is not None:
            if context_length <= 0:
                raise ValueError("context_length 必须为正整数")
            payload["options"] = {"num_ctx": context_length}
        return self._post("/api/generate", payload, f"加载模型 {name} 失败")

    def unload_model(self, name: str) -> dict[str, Any]:
        """立即卸载模型（``keep_alive: 0``），释放显存给别的服务."""
        return self._post(
            "/api/generate", {"model": name, "keep_alive": 0}, f"卸载模型 {name} 失败"
        )

    def pull_model(self, name: str, *, stream: bool = False) -> dict[str, Any]:
        """拉取模型权重（``POST /api/pull``），``stream=False`` 等下载完成再返回."""
        return self._post(
            "/api/pull", {"model": name, "stream": stream}, f"拉取模型 {name} 失败"
        )
