"""本地部署模型的调用层（day045）：让自建推理服务与云端模型共用一套接口.

为什么本地部署能"零改造"接入现有 Agent？因为 Ollama 与 vLLM 都提供
**OpenAI 兼容**端点：只要 base_url 指到本机、api_key 随便填一个非空
字符串，``openai`` SDK 的调用方式与访问云端完全一致。这正是 day045 的
核心结论——**协议兼容比 SDK 封装更重要**：我们不写一个"Ollama 客户端"，
而是让 ``LocalModel`` 复用 ``OpenAICompatibleLLM`` 的协议实现，只补齐
三件本地部署特有的事：

1. **端点规范化**：本地服务的 base_url 有强默认值（Ollama ``11434``、
   vLLM ``8000``），且用户常常只写 ``http://localhost:11434`` 而漏掉
   ``/v1`` 前缀——``normalize_base_url`` 负责把两种写法收敛成一种；
2. **占位密钥**：Ollama 不校验 API Key（文档原话是 "required but
   ignored"），vLLM 未加 ``--api-key`` 时也不校验，但 SDK 要求 api_key
   非空，所以本地路线用约定占位值而非真实密钥；
3. **能力自述**：本地模型是否支持视觉、上下文窗口多大、模型常驻多久，
   都是部署侧的配置事实（不是协议问题），由 ``LocalModelSpec`` 声明。

真实事实来源（day045 编写时核实）：
  - Ollama OpenAI 兼容端点 ``http://localhost:11434/v1/``，api_key
    传 ``"ollama"``（"required but ignored"）；
  - Ollama 默认上下文窗口 **4096** token（可用 ``OLLAMA_CONTEXT_LENGTH``
    覆盖），模型默认常驻 **5m**（可用 ``keep_alive`` 覆盖）；
  - vLLM 的 OpenAI 兼容服务默认端口 **8000**，``--api-key`` 默认未启用；
  - vLLM 不原生支持 Windows（需 WSL2 或社区构建），Ollama 原生支持
    Windows/macOS/Linux——本机是 Windows，因此本课以 Ollama 为主、
    vLLM 为进阶对照。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from smart_research_agent.config import settings
from smart_research_agent.llm.base import Message
from smart_research_agent.llm.openai_compatible import OpenAICompatibleLLM
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 支持的后端标识
OLLAMA = "ollama"
VLLM = "vllm"
SUPPORTED_BACKENDS: tuple[str, ...] = (OLLAMA, VLLM)

#: 两个后端的默认端口（官方文档默认值）
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_VLLM_PORT = 8000

#: 本地服务的占位密钥：SDK 要求非空，但本地服务端不校验
DEFAULT_OLLAMA_API_KEY = "ollama"
DEFAULT_VLLM_API_KEY = "EMPTY"

#: Ollama 默认上下文窗口（token）与模型常驻时长
DEFAULT_OLLAMA_CONTEXT_LENGTH = 4096
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"


def default_base_url(backend: str = OLLAMA) -> str:
    """返回后端的默认 OpenAI 兼容端点（含 ``/v1`` 前缀）."""
    if backend == VLLM:
        return f"http://localhost:{DEFAULT_VLLM_PORT}/v1"
    return f"http://localhost:{DEFAULT_OLLAMA_PORT}/v1"


def normalize_base_url(base_url: str, backend: str = OLLAMA) -> str:
    """把用户填写的本地端点规范化为可用的 OpenAI 兼容 base_url.

    处理三种常见写法：

    - ``http://localhost:11434``：只有主机与端口，补 ``/v1``（漏写前缀是
      最常见的配置错误，OpenAI SDK 会把 ``/chat/completions`` 直接拼在
      base_url 后，缺 ``/v1`` 就会 404）；
    - ``http://localhost:11434/v1/``：去掉尾部多余的斜杠，避免拼出
      ``//chat/completions``；
    - ``localhost:11434``：缺协议头，补 ``http://``。

    只在路径为空或 "/" 时自动补 ``/v1``；若用户显式写了别的前缀
    （如反向代理的 ``/proxy/llm``），原样保留，不做画蛇添足的改写。
    """
    text = (base_url or "").strip()
    if not text:
        raise ValueError("base_url 不能为空")

    if "://" not in text:
        text = f"http://{text}"

    parsed = urlparse(text)
    if not parsed.hostname:
        raise ValueError(f"无法解析的 base_url: {base_url}")

    path = parsed.path.rstrip("/")
    if path in ("", "/"):
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc
        return f"{scheme}://{netloc}/v1"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


@dataclass
class LocalModelSpec:
    """一个本地部署模型的部署侧档案.

    字段都是"部署事实"而非"协议参数"：同一个模型换个量化版本、换个
    上下文窗口就是另一份 spec，但调用协议完全不变。
    """

    name: str
    backend: str = OLLAMA
    base_url: str | None = None
    api_key: str | None = None
    supports_vision: bool = False
    #: 上下文窗口（token）。Ollama 默认 4096，vLLM 由 --max-model-len 决定
    context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH
    #: 模型空闲常驻时长（仅 Ollama 生效）："5m" / "1h" / "-1"（常驻不卸载）
    keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"未知的本地后端 {self.backend}，可选值: {' / '.join(SUPPORTED_BACKENDS)}"
            )
        if not self.name:
            raise ValueError("本地模型名不能为空（如 qwen3:8b）")
        if self.context_length <= 0:
            raise ValueError("context_length 必须为正整数")
        if self.base_url is None:
            self.base_url = default_base_url(self.backend)
        self.base_url = normalize_base_url(self.base_url, self.backend)
        if self.api_key is None:
            self.api_key = (
                DEFAULT_OLLAMA_API_KEY if self.backend == OLLAMA else DEFAULT_VLLM_API_KEY
            )

    @property
    def endpoint(self) -> str:
        """人类可读的端点描述，供日志与 /health 展示."""
        return f"{self.backend}@{self.base_url}"


class LocalModel(OpenAICompatibleLLM):
    """本地推理服务上的模型，协议层面与云端模型完全等价.

    继承 ``OpenAICompatibleLLM`` 是刻意的设计选择：**本地与云端的差异在
    部署与运维，不在调用协议**。因此 chat / stream / chat_vision 全部复用
    父类实现（Ollama 的 ``/v1/chat/completions`` 同样接受图像部件数组），
    本类只补父类没有的两件事：

    - ``chat_with_tools``：父类未实现 function calling，而 Ollama 的
      OpenAI 兼容端点已支持 tools 字段，本地路线不该因"部署在本地"就
      降级回文本协议；
    - ``is_available`` / ``list_served_models``：本地服务的可用性是**先于
      调用的运维事实**（进程没起来、模型没 pull），需要一次轻量探活
      （``GET /v1/models``）来回答，而不是等一次真实推理超时。

    ``client`` 参数用于测试注入：任何实现了 ``models.list`` 与
    ``chat.completions.create`` 的对象都可以充当客户端。
    """

    def __init__(
        self,
        spec: LocalModelSpec | None = None,
        *,
        name: str | None = None,
        backend: str = OLLAMA,
        base_url: str | None = None,
        api_key: str | None = None,
        supports_vision: bool = False,
        context_length: int = DEFAULT_OLLAMA_CONTEXT_LENGTH,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
        client: Any | None = None,
    ):
        resolved = spec or LocalModelSpec(
            name=name or settings.local_model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            supports_vision=supports_vision,
            context_length=context_length,
            keep_alive=keep_alive,
        )
        # 父类负责建立 OpenAI 客户端并把 model 名记在 _model 上
        super().__init__(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            model=resolved.name,
        )
        self.spec = resolved
        if client is not None:
            self._client = client

    @property
    def backend(self) -> str:
        return self.spec.backend

    @property
    def model_name(self) -> str:
        return self.spec.name

    @property
    def base_url(self) -> str:
        return self.spec.base_url  # type: ignore[return-value]

    @property
    def context_length(self) -> int:
        return self.spec.context_length

    @property
    def keep_alive(self) -> str:
        return self.spec.keep_alive

    @property
    def supports_vision(self) -> bool:
        """视觉能力来自部署侧声明（如 Ollama 的 qwen3-vl 系列）."""
        return self.spec.supports_vision

    def describe(self) -> dict[str, Any]:
        """输出可供日志/监控使用的端点档案."""
        return {
            "name": self.spec.name,
            "backend": self.spec.backend,
            "base_url": self.spec.base_url,
            "context_length": self.spec.context_length,
            "keep_alive": self.spec.keep_alive,
            "supports_vision": self.spec.supports_vision,
        }

    def is_available(self, timeout: float = 2.0) -> bool:
        """探活：本地服务与模型是否就绪（不发起推理）.

        在服务端覆盖 ``GET /v1/models``（Ollama 与 vLLM 都实现），一次
        轻量请求即可回答"现在能不能用"。失败不抛异常——探活的语义是
        返回事实，调用方据此决定是否降级到云端模型。
        """
        try:
            self._client.models.list(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - 探活要吞掉一切连接类异常
            logger.warning("本地模型 %s 不可用（%s）: %s", self.spec.name, self.spec.base_url, exc)
            return False
        return True

    def list_served_models(self, timeout: float = 2.0) -> list[str]:
        """列出本地服务已加载/已托管的模型名（``GET /v1/models``）.

        与 ``is_available`` 的区别：探活只关心"通不通"，本方法关心
        "有哪些"。模型名拼错（如 ``qwen3:8b`` 写成 ``qwen3-8b``）是本地
        部署最高频的错误，列出候选能让报错从"404 model not found"
        变成"你要的是不是其中之一"。
        """
        try:
            page = self._client.models.list(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"无法访问本地 {self.spec.backend} 服务（{self.spec.base_url}）: {exc}"
            ) from exc
        return [item.id for item in getattr(page, "data", []) or []]

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """本地模型的 function calling（与云端同协议、同返回结构）.

        返回结构与 ``MockLLM.chat_with_tools`` 及 OpenAI 兼容约定一致：
        ``{"content": str | None, "tool_calls": [...]}``，其中
        ``arguments`` 保持**JSON 字符串**形态（与线上协议一致，解析交给
        day037 实现的 ``parse_tool_calls``）。无工具调用时省略
        ``tool_calls`` 键——调用方用 ``response.get("tool_calls")`` 判断，
        不需要区分 None 与空列表。
        """
        resp = self._client.chat.completions.create(
            model=self.spec.name,
            messages=[m.to_dict() for m in messages],
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = resp.choices[0].message
        content = getattr(message, "content", None)
        calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            calls.append(
                {
                    "id": getattr(call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(function, "name", ""),
                        "arguments": getattr(function, "arguments", "{}"),
                    },
                }
            )
        result: dict[str, Any] = {"content": content}
        if calls:
            result["tool_calls"] = calls
        return result

    # 显式声明：本地模型同样支持流式（继承父类的 stream 实现）
    def stream(  # type: ignore[override]
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        yield from super().stream(messages, temperature=temperature, max_tokens=max_tokens)


def create_local_model(
    model: str | None = None,
    *,
    backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    supports_vision: bool | None = None,
    context_length: int | None = None,
    keep_alive: str | None = None,
    client: Any | None = None,
) -> LocalModel:
    """按 ``settings`` 构建本地模型（未传参的字段一律回落到配置）.

    与 day041 的 ``default_embedding()``、day043 的 ``default_llm()`` 同一
    套路：**工厂集中处理"从配置到实例"的映射**，业务代码只依赖接口。
    注意本工厂**不做连通性探测**——探活是运行期决策（``is_available``），
    构造期只保证配置被正确解析，这样离线开发机也能构造对象做单测。
    """
    backend = backend or settings.local_backend
    # 只有"配置里的那个后端"才继承 settings 的密钥；显式切换后端（如临时改用
    # vLLM）时必须回落到该后端自己的占位值，否则会把 Ollama 的占位密钥
    # 发给了 vLLM，日志里很难看出这种串台。
    settings_api_key = settings.local_api_key if backend == settings.local_backend else None
    settings_base_url = settings.local_base_url if backend == settings.local_backend else None
    spec = LocalModelSpec(
        name=model or settings.local_model,
        backend=backend,
        base_url=base_url or settings_base_url or None,
        api_key=api_key or settings_api_key or None,
        supports_vision=(
            settings.local_supports_vision if supports_vision is None else supports_vision
        ),
        context_length=(
            settings.local_context_length if context_length is None else context_length
        ),
        keep_alive=keep_alive or settings.local_keep_alive,
    )
    logger.info(
        "本地模型接入: name=%s backend=%s base_url=%s ctx=%d keep_alive=%s",
        spec.name,
        spec.backend,
        spec.base_url,
        spec.context_length,
        spec.keep_alive,
    )
    return LocalModel(spec, client=client)
