"""``bridge``：把它**接进** ``local_model`` 模块（day086 / M7-D10）.

day045 的 :class:`~smart_research_agent.llm.local_model.LocalModel` 走的是一条**协议**路线：
Ollama 与 vLLM 都提供 OpenAI 兼容端点，因此本地模型与云端模型共用一套调用方式。
今天出现第三条路线：**在进程内跑**（模型就在这个 Python 进程里，没有服务端）。
三条路线并存会立刻带来一个问题，而本模块的全部内容就是回答它：

```text
"接进来"到底要接什么？
```

答案不是"再写一个客户端"，而是**方法面一致**。`LocalModel` 暴露了八个东西：

```text
chat / stream / chat_with_tools / is_available / describe
model_name / backend / context_length
```

只要在进程模型也提供这八个（同样语义、同样返回结构），上层代码一行不改就能切换。
本模块因此把"接口面"写成一个**常量清单**（:data:`PROTOCOL_SURFACE`），
并让 :mod:`verify` 逐名核对两边——一条**能被失败的**对账。

## 三条路线各自的事实（不是偏好）

```text
                服务端   离线可用  视觉  工具调用  上下文窗口来自
in-process      不需要   是       由模型  不支持    config 里的位置表长度
ollama          需要     是       支持    支持      OLLAMA_CONTEXT_LENGTH（默认 4096）
vLLM            需要     是       部分    支持      --max-model-len
```

第四条差别最容易被忽略而后果最大：**上下文窗口的来源不同**。
在进程模型的窗口来自 ``n_positions`` / ``max_position_embeddings``（写死在配置里），
而 Ollama 的默认窗口是 **4096**（day045 核实过）——同一段长文本，
在两条路线上的"够不够用"是**两个不同的答案**。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.hf_integration.forward import ModelWeights, hidden_states
from smart_research_agent.hf_integration.tokenizer import ByteBPETokenizer
from smart_research_agent.hf_integration.types import ModelCard
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.local_model import (
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_PORT,
    OLLAMA,
    SUPPORTED_BACKENDS,
    VLLM,
)
from smart_research_agent.hf_source.types import GenerationSettings

#: 第三条后端的名字（前两条在 day045 就占好了：``ollama`` / ``vllm``）.
IN_PROCESS = "in-process"

#: 三条后端（**本包把第三条加进同一张表**，而不是另起一套命名）.
ALL_BACKENDS: tuple[str, ...] = (IN_PROCESS,) + SUPPORTED_BACKENDS

#: "接进来"要求的方法面（**这张清单是这一课最值钱的一行**）.
PROTOCOL_SURFACE: tuple[str, ...] = (
    "chat",
    "stream",
    "chat_with_tools",
    "is_available",
    "describe",
    "model_name",
    "backend",
    "context_length",
)

#: 三条路线的客观事实（**不是偏好**，因此可以逐格核对）.
BACKEND_TRAITS: dict[str, dict[str, object]] = {
    IN_PROCESS: {
        "server": False,
        "offline": True,
        "vision": False,
        "tools": False,
        "context_source": "config 里的位置表长度（n_positions / max_position_embeddings）",
        "placeholder_key": False,
        "startup": "随进程（没有冷启动）",
    },
    OLLAMA: {
        "server": True,
        "offline": True,
        "vision": True,
        "tools": True,
        "context_source": f"OLLAMA_CONTEXT_LENGTH（默认 {DEFAULT_OLLAMA_CONTEXT_LENGTH}）",
        "placeholder_key": True,
        "startup": f"模型常驻时长 {DEFAULT_OLLAMA_KEEP_ALIVE}（冷启动是它）",
    },
    VLLM: {
        "server": True,
        "offline": True,
        "vision": False,
        "tools": True,
        "context_source": "--max-model-len（部署时给）",
        "placeholder_key": True,
        "startup": f"默认端口 {DEFAULT_VLLM_PORT}；不原生支持 Windows",
    },
}

#: 三条路线各自的占位密钥（真实的两个在 day045 核实过；进程内不需要密钥）.
PLACEHOLDER_KEYS: dict[str, str | None] = {
    IN_PROCESS: None,
    OLLAMA: DEFAULT_OLLAMA_API_KEY,
    VLLM: DEFAULT_VLLM_API_KEY,
}


@dataclass
class InProcessSpec:
    """在进程模型的部署侧档案：与 ``LocalModelSpec`` **同字段语义**.

    ``context_length`` 在这里不是"部署方随手给的数"，而是**配置算出来的**——
    它等于位置表长度。这条差别值得写下来：同一个字段在两条路线上
    一个是"我们的选择"、一个是"模型的硬上界"。
    """

    name: str
    backend: str = IN_PROCESS
    model_type: str = "gpt2"
    parameters: int = 0
    vocab_size: int = 0
    context_length: int = 0
    supports_vision: bool = False
    pooling: str = "mean"
    normalize: bool = False
    note: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.backend not in ALL_BACKENDS:
            raise ValueError(
                f"未知的后端 {self.backend!r}，可选值: {' / '.join(ALL_BACKENDS)}"
            )
        if not self.name:
            raise ValueError("模型名不能为空（在进程模型也应当有一个可读的名字）")
        if self.context_length <= 0:
            raise ValueError("context_length 必须为正整数（它来自配置的位置表）")

    @classmethod
    def of(cls, card: ModelCard, *, pooling: str = "mean", normalize: bool = False) -> InProcessSpec:
        """从一张卡片造出档案（**上下文窗口由配置决定**）."""
        return cls(
            name=card.name,
            model_type=card.model_type,
            parameters=card.parameter_count,
            vocab_size=card.vocab,
            context_length=card.positions,
            supports_vision=False,
            pooling=pooling,
            normalize=normalize,
            note=(
                "在进程模型没有服务端，因此没有冷启动、也没有连接类失败",
                "它的上下文窗口来自位置表长度，而不是一个部署参数",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段（与 ``LocalModel.describe`` 的键**尽量对齐**）."""
        return {
            "name": self.name,
            "backend": self.backend,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "supports_vision": self.supports_vision,
            "pooling": self.pooling,
            "normalize": self.normalize,
        }


class InProcessModel(BaseLLM):
    """在进程里跑的模型：**与 ``LocalModel`` 同方法面**，不经过任何服务端.

    它继承 :class:`~smart_research_agent.llm.base.BaseLLM` 而不是 ``LocalModel``，
    因为父类那一侧是"OpenAI 客户端 + HTTP 协议"，而在进程模型没有 HTTP。
    两者是**兄弟**：共同点写在 :data:`PROTOCOL_SURFACE` 那张清单里，由测试逐名核对。
    """

    def __init__(
        self,
        card: ModelCard,
        weights: ModelWeights,
        tokenizer: ByteBPETokenizer,
        *,
        spec: InProcessSpec | None = None,
        settings: GenerationSettings | None = None,
    ):
        if weights.card.name != card.name:  # pragma: no cover - 防错配
            raise ValueError("权重与卡片不是同一个模型：两份东西必须来自同一次装载。")
        self.card = card
        self.weights = weights
        self.tokenizer = tokenizer
        self.spec = spec or InProcessSpec.of(
            card, pooling="mean", normalize=False
        )
        self.settings = settings or GenerationSettings(max_new_tokens=8, do_sample=False)
        self.calls: list[str] = []

    # -- 与 LocalModel 的八个方法面 -------------------------------------------------

    @property
    def model_name(self) -> str:
        """模型名（与 ``LocalModel.model_name`` 同义）."""
        return self.spec.name

    @property
    def backend(self) -> str:
        """后端标识（这里是 ``"in-process"``）."""
        return self.spec.backend

    @property
    def context_length(self) -> int:
        """上下文窗口（= 位置表长度，**这是一条硬上界**）."""
        return self.spec.context_length

    @property
    def supports_vision(self) -> bool:
        """在进程模型不支持图像（与 ``BaseLLM`` 的默认值一致：诚实地说不支持）."""
        return False

    def describe(self) -> dict[str, Any]:
        """输出可供日志/监控使用的档案（键与 ``LocalModel.describe`` 对齐）."""
        payload = self.spec.to_dict()
        payload.update(
            {
                "tie_word_embeddings": self.card.tie_word_embeddings,
                "heads": self.card.heads,
                "layers": self.card.layers,
                "positions": self.card.positions,
            }
        )
        return payload

    def is_available(self, timeout: float = 2.0) -> bool:
        """在进程模型的可用性：**进程活着它就可用**.

        保留 ``timeout`` 参数是刻意的——签名一致才能被同一段代码调用；
        而在进程路线**不会**因为"服务没起来"而失败，因此这个参数在这里不产生分支。
        """
        del timeout
        return True

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """把消息按 ChatML 风格拼成一段提示词，然后自回归生成.

        ``temperature`` 与 ``max_tokens`` 都**真的**作用在生成上
        （温度进 day085 的策略，``max_tokens`` 截到生成预算）。
        """
        prompt = render_messages(messages)
        settings = self._settings_for(temperature, max_tokens)
        from smart_research_agent.hf_integration.pipeline import text_generation

        output = text_generation(self.card, self.weights, self.tokenizer, (prompt,), settings)
        self.calls.append("chat")
        return output.texts[0]

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """逐 token 增量返回（**拼接之后必须等于一次 chat 的完整回复**）.

        这条契约来自 day039：流式与一次性在语义上必须等价。
        本包按"新生成的那一个 token 解出来的增量"逐段 yield——因此
        ``"".join(stream(...)) == chat(...)`` 是一条**可断言**的性质，
        而它也是"首字秒出"在两条路线上的同一个定义。
        """
        prompt = render_messages(messages)
        settings = self._settings_for(temperature, max_tokens)
        from smart_research_agent.hf_integration.pipeline import text_generation

        output = text_generation(self.card, self.weights, self.tokenizer, (prompt,), settings)
        generated = output.results[0].generated
        previous = ""
        for size in range(1, len(generated) + 1):
            piece = self.tokenizer.decode(generated[:size])
            if len(piece) > len(previous):
                yield piece[len(previous) :]
                previous = piece
        self.calls.append("stream")

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """在进程模型**不实现** function calling——因此这里明确抛错，而不是静默降级.

        day045 的 ``LocalModel`` 之所以实现它，是因为 Ollama 的兼容端点**真的支持**
        ``tools`` 字段。本包的玩具主干没有工具调用头，因此"假装支持"的后果是：
        调用方以为会拿到 ``tool_calls``，实际拿到一段自由文本，而 Agent 会
        带着这段文本继续跑（它看起来像"模型决定不调工具了"）。
        """
        del messages, tools, temperature, max_tokens
        raise NotImplementedError(
            "在进程模型不实现 function calling：本包的玩具主干没有工具调用头。"
            "要工具调用请走 ollama / vLLM 路线（/v1/chat/completions 的 tools 字段），"
            "或使用 day037 的文本协议解析器。"
        )

    # -- 两个在进程路线特有的读数 ---------------------------------------------------

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """特征抽取（**在进程路线特有**：HTTP 路线的 OpenAI 兼容端点是 /v1/embeddings）."""
        from smart_research_agent.hf_integration.pipeline import feature_extraction

        output = feature_extraction(
            self.card,
            self.weights,
            self.tokenizer,
            texts,
            pooling=self.spec.pooling,
            normalize=self.spec.normalize,
        )
        self.calls.append("embed")
        return output.pooled.vectors

    def hidden_for(self, text: str) -> tuple[tuple[float, ...], ...]:
        """一条文本的 hidden states（**每条只保留真实长度**的那几行）."""
        ids = self.tokenizer.encode(text)
        batch = hidden_states(self.card, self.weights, (tuple(ids),))
        self.calls.append("hidden_for")
        return batch.rows[0]

    def _settings_for(self, temperature: float, max_tokens: int) -> GenerationSettings:
        """把一次调用的两个参数折进生成配置（**两个都真的生效**，不静默忽略）."""
        from dataclasses import replace

        budget = max(1, min(int(max_tokens), 64))
        sampling = self.settings.do_sample or temperature != 1.0
        return replace(
            self.settings,
            max_new_tokens=budget,
            do_sample=sampling,
            temperature=temperature if sampling else 1.0,
        )


def render_messages(messages: list[Message]) -> str:
    """把消息列表渲染成一段提示词（**这是"协议"在纯文本路线上的样子**）.

    四行格式（``<|role|>``）与 day036 的提示词模板同一取向：可读、可断言、
    不依赖任何服务端的模板机制。系统消息在最前，工具结果与用户/助手按原顺序。
    """
    lines = [f"<|{message.role}|>{message.content}" for message in messages]
    return "\n".join(lines) + "\n<|assistant|>"


def create_in_process_model(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    *,
    pooling: str = "mean",
    normalize: bool = False,
    settings: GenerationSettings | None = None,
) -> InProcessModel:
    """按 day045 的工厂口径造一个在进程模型（**未传参的字段一律回落到卡片**）.

    与 ``create_local_model`` 同一套路：**工厂集中处理"从配置到实例"的映射**，
    业务代码只依赖接口。这里连"连通性探测"都不需要——在进程模型没有连接。
    """
    spec = InProcessSpec.of(card, pooling=pooling, normalize=normalize)
    return InProcessModel(card, weights, tokenizer, spec=spec, settings=settings)


@dataclass(frozen=True)
class BackendChoice:
    """一次路线选择：选了哪一个、依据是什么（**依据必须逐条写出来**）."""

    backend: str
    reasons: tuple[str, ...]

    def line(self) -> str:
        """一行可读的结论."""
        return f"[{self.backend}] " + "；".join(self.reasons)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {"backend": self.backend, "reasons": list(self.reasons)}


def choose_backend(
    *,
    offline_required: bool = True,
    needs_tools: bool = False,
    needs_vision: bool = False,
    needs_batching: bool = False,
    multi_user: bool = False,
    long_context: bool = False,
) -> BackendChoice:
    """按需求选一条路线（**规则写在理由里**，因此每次选择都可复核）.

    两条硬规则（先判）：

    ```text
    需要工具调用 ⇒ 不是 in-process（它没有工具头）
    需要视觉     ⇒ 只有 ollama（vLLM 取决于模型，本课按"不支持"记）
    ```

    一条软规则：多用户/高并发倾向 vLLM（它的卖点是吞吐），
    单机单用户且要长上下文倾向 ollama（窗口可调），
    要"零部署、随进程"倾向 in-process。
    """
    if needs_tools:
        if needs_vision:
            return BackendChoice(
                backend=OLLAMA,
                reasons=(
                    "需要工具调用 ⇒ 排除 in-process（玩具主干没有工具头）",
                    "需要视觉 ⇒ 只剩 ollama（day045 核实过它的兼容端点接受图像部件）",
                ),
            )
        return BackendChoice(
            backend=VLLM if multi_user else OLLAMA,
            reasons=(
                "需要工具调用 ⇒ 排除 in-process",
                "多用户/高并发 ⇒ vLLM（吞吐是它的卖点）"
                if multi_user
                else "单机 ⇒ ollama（一条命令起服务，占位密钥 required but ignored）",
            ),
        )
    if needs_vision:
        return BackendChoice(
            backend=OLLAMA,
            reasons=("需要视觉 ⇒ 只剩 ollama（本课按 vLLM 不支持记）",),
        )
    if multi_user or needs_batching:
        return BackendChoice(
            backend=VLLM,
            reasons=(
                "要批量/多用户吞吐 ⇒ vLLM（continuous batching）",
                "代价：需要服务端与 --max-model-len，且不原生支持 Windows",
            ),
        )
    if long_context and offline_required:
        return BackendChoice(
            backend=OLLAMA,
            reasons=(
                "要长上下文 ⇒ 需要能调窗口的那条路线"
                f"（OLLAMA_CONTEXT_LENGTH 默认 {DEFAULT_OLLAMA_CONTEXT_LENGTH}，可覆盖）",
                "而 in-process 的窗口被位置表**写死**",
            ),
        )
    return BackendChoice(
        backend=IN_PROCESS,
        reasons=(
            "不要工具、不要视觉、不要并发 ⇒ 在进程跑最省事",
            "没有服务端、没有冷启动、没有占位密钥，也没有连接类失败",
            "代价：窗口由位置表写死，且换模型要改代码而不是改部署参数",
        ),
    )


def trait_line(backend: str) -> str:
    """一行读数：一条路线的客观事实（三条并排看时最有用）."""
    if backend not in BACKEND_TRAITS:
        raise ValueError(f"未知后端 {backend!r}，可选 {list(ALL_BACKENDS)}")
    traits = BACKEND_TRAITS[backend]
    key = PLACEHOLDER_KEYS[backend]
    return (
        f"{backend:<10} 服务端={traits['server']!s:<5} 离线={traits['offline']!s:<5} "
        f"视觉={traits['vision']!s:<5} 工具={traits['tools']!s:<5} "
        f"占位密钥={key if key is not None else '（不需要）'}"
    )


__all__ = [
    "ALL_BACKENDS",
    "BACKEND_TRAITS",
    "IN_PROCESS",
    "PLACEHOLDER_KEYS",
    "PROTOCOL_SURFACE",
    "BackendChoice",
    "InProcessModel",
    "InProcessSpec",
    "choose_backend",
    "create_in_process_model",
    "render_messages",
    "trait_line",
]
