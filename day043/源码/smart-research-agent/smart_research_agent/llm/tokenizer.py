"""Token 计数与成本预估：tiktoken 优先，字符级估算兜底（day034）.

为什么 token 计数是工程问题而不是学术问题：
上下文窗口按 token 计费、按 token 限长，提示词超一个字不会报错，
超一个 token 就会被截断或拒绝。要在调用前知道"这段话值多少 token"，
就必须在本地复刻厂商的分词器——tiktoken 正是 OpenAI 开源的官方 BPE 实现。

本模块的设计：
- 优先使用 tiktoken（cl100k_base / o200k_base 等 encoding，词表文件
  首次加载后缓存在本地，之后完全离线可用）；
- tiktoken 不可用（未安装 / 词表缓存缺失且无网络）时，降级为
  **中英感知的字符级估算**，保证计数功能永远可用，只是精度下降；
- ``backend`` 属性显式暴露当前走的是哪条路径，教程与测试可断言。

与 day032 的衔接：``estimate_cost`` 直接消费
``observability.cost_tracker.DEFAULT_PRICE_TABLE`` 格式的价格表，
使得"写提示词 → 预估费用 → 决定是否压缩"可以在调用发生前完成。
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

from smart_research_agent.observability.cost_tracker import DEFAULT_PRICE_TABLE

#: 模型名 -> tiktoken encoding 名。未知模型回退到 cl100k_base（GPT-4 时代的通用词表）。
MODEL_TO_ENCODING: dict[str, str] = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
}

#: 兜底估算时，每个 CJK 字符按多少 token 计（cl100k_base 下常见汉字约 1~1.5 token/字）。
CJK_TOKEN_PER_CHAR = 1.0

#: 兜底估算时，每多少个 ASCII 字符按 1 token 计（英文经验值约 4 字符/token）。
ASCII_CHARS_PER_TOKEN = 4.0


def estimate_tokens_chars(text: str) -> int:
    """中英感知的字符级 token 估算（tiktoken 不可用时的降级方案）.

    规则：CJK（中日韩）字符按 1 token/字计，其余字符按 4 字符/token 计。
    这比 mock.py 里"全文 4 字符 1 token"的粗略估算更贴近中文场景——
    中文按字符算是严重低估，必须单独处理。
    """
    if not text:
        return 0
    cjk = sum(
        1
        for ch in text
        if unicodedata.category(ch) == "Lo"  # 表意文字（汉字等）
        or "一" <= ch <= "鿿"
    )
    other = len(text) - cjk
    return max(1, math.ceil(cjk * CJK_TOKEN_PER_CHAR + other / ASCII_CHARS_PER_TOKEN))


@dataclass
class TokenCounter:
    """Token 计数器：优先 tiktoken 精确计数，失败时降级为字符级估算.

    用法::

        counter = TokenCounter()
        counter.backend                       # "tiktoken" 或 "fallback"
        counter.count_tokens("你好，世界")     # 精确或估算的 token 数
        counter.estimate_cost("提示词", "gpt-4o-mini")  # 按价格表折算美元

    ``prefer_tiktoken=False`` 可强制走兜底路径，用于教学对比
    "精确计数与字符估算差多少"。
    """

    default_encoding: str = "cl100k_base"
    prefer_tiktoken: bool = True
    _encodings: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _tiktoken_checked: bool = field(default=False, init=False, repr=False)
    _tiktoken_ok: bool = field(default=False, init=False, repr=False)

    def _load_encoding(self, name: str):
        """加载（并缓存）一个 tiktoken encoding；任何失败都返回 None 走降级."""
        if not self.prefer_tiktoken:
            return None
        if name in self._encodings:
            return self._encodings[name]
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(name)
        except Exception:
            encoding = None
        self._encodings[name] = encoding
        return encoding

    @property
    def backend(self) -> str:
        """当前计数后端：``"tiktoken"``（精确）或 ``"fallback"``（字符估算）."""
        return "tiktoken" if self._load_encoding(self.default_encoding) is not None else "fallback"

    def encoding_name_for(self, model: str | None) -> str:
        """模型名 -> encoding 名；未知模型回退到 default_encoding."""
        if model is None:
            return self.default_encoding
        return MODEL_TO_ENCODING.get(model, self.default_encoding)

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """计算文本的 token 数.

        - tiktoken 可用：按模型对应的 encoding 精确计数；
        - 不可用：字符级估算（CJK 1 token/字，其余约 4 字符/token）。
        """
        encoding = self._load_encoding(self.encoding_name_for(model))
        if encoding is not None:
            return len(encoding.encode(text))
        return estimate_tokens_chars(text)

    def count_messages(self, messages: list[dict[str, str]], model: str | None = None) -> int:
        """估算一组 chat 消息的总 token 数（含每条消息的角色标记开销）.

        参考 OpenAI 官方 cookbook 的口径：每条消息除内容外还有少量
        元数据 token（角色标记、分隔符），这里按每条 4 token 的常数开销
        估算，足以用于上下文窗口的余量规划。
        """
        per_message_overhead = 4
        return sum(
            self.count_tokens(m.get("content", ""), model) + per_message_overhead
            for m in messages
        )

    def estimate_cost(
        self,
        prompt_text: str,
        model: str,
        price_table: dict[str, dict[str, float]] | None = None,
        expected_completion_tokens: int = 0,
    ) -> dict:
        """在调用发生前预估一次调用的费用（美元）.

        价格表沿用 day032 ``CostTracker`` 的格式（模型 -> input/output 每 1K
        token 单价）。返回明细字典而非单个数字，便于教程展示与测试断言：
        输入/输出各多少 token、各多少钱、合计多少。
        """
        table = price_table if price_table is not None else DEFAULT_PRICE_TABLE
        if model not in table:
            raise KeyError(f"价格表中没有模型 {model!r}，请先在 price_table 中配置价格")
        prompt_tokens = self.count_tokens(prompt_text, model)
        price = table[model]
        input_cost = prompt_tokens * price["input"] / 1000.0
        output_cost = expected_completion_tokens * price["output"] / 1000.0
        return {
            "model": model,
            "backend": self.backend,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": expected_completion_tokens,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "total_cost_usd": round(input_cost + output_cost, 8),
        }
