"""Function Calling 的结构化基础：工具规格、调用解析与参数校验（day037）.

本模块覆盖 function calling 链路中"不依赖具体 LLM 厂商"的三件事：

1. FunctionSpec：从 BaseTool 的元数据自动生成 OpenAI 兼容的函数描述
   （name / description / parameters JSON Schema），直接可放进请求的 tools 字段；
2. parse_tool_calls：把 OpenAI 兼容响应里的 tool_calls 数组解析成 ToolCall 列表；
3. validate_arguments：按 JSON Schema 校验模型给出的参数，失败抛 SchemaValidationError。

错误分类（ToolCallError 三子类）对应三种截然不同的处置策略：
  - SchemaValidationError：模型给错了参数——把错误反馈给 LLM 重试（fc_agent.py）；
  - UnknownToolError：模型编了不存在的工具——反馈重试，且提示词可能需加固；
  - ToolExecutionError：工具自身执行异常——参数没错，重试模型无意义，应反馈观察结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from smart_research_agent.tools.base import BaseTool

#: JSON Schema 类型到 Python 类型的映射（number 含 int 与 float，但 bool 不算 int）
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolCallError(Exception):
    """函数调用失败的基类，kind 字段用于分类处置."""

    kind = "error"


class SchemaValidationError(ToolCallError):
    """参数不符合工具的 JSON Schema（缺必填字段 / 类型不符 / 参数不是合法 JSON）."""

    kind = "schema"


class UnknownToolError(ToolCallError):
    """模型调用了注册表中不存在的工具."""

    kind = "unknown_tool"


class ToolExecutionError(ToolCallError):
    """工具执行过程抛出异常（参数合法，失败发生在工具内部）."""

    kind = "execution"


@dataclass
class ToolCall:
    """一次函数调用的结构化表示.

    id 用于把 tool role 消息与调用配对（OpenAI 协议要求）；arguments 已解析为 dict。
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class FunctionSpec:
    """一个可被 LLM 调用的函数描述（OpenAI tools 字段的 core 部分）."""

    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def from_tool(cls, tool: BaseTool) -> FunctionSpec:
        """从 BaseTool 元数据自动生成规格——schema 与工具实现同源，永不漂移."""
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )

    def to_openai_tool(self) -> dict[str, Any]:
        """导出 OpenAI 兼容的 tools 数组元素."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def specs_from_tools(tools: list[BaseTool]) -> list[FunctionSpec]:
    """把一组工具批量转成 FunctionSpec."""
    return [FunctionSpec.from_tool(t) for t in tools]


def parse_tool_calls(response: dict[str, Any]) -> list[ToolCall]:
    """解析 OpenAI 兼容响应中的 tool_calls 数组.

    response 形如::

        {"tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}}
        ]}

    arguments 兼容两种形态：JSON 字符串（OpenAI 线上格式）或已解析的 dict
    （MockLLM 与部分厂商 SDK 会代理解析）。JSON 解析失败抛 SchemaValidationError——
    坏 JSON 本质上是模型没按 schema 输出，归入 schema 类错误走重试回路。
    """
    calls: list[ToolCall] = []
    for i, raw in enumerate(response.get("tool_calls") or []):
        function = raw.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"第 {i + 1} 个 tool_call（{name}）的 arguments 不是合法 JSON: {exc}"
                ) from exc
        if not isinstance(arguments, dict):
            raise SchemaValidationError(
                f"第 {i + 1} 个 tool_call（{name}）的 arguments 必须是 JSON 对象，"
                f"实际为 {type(arguments).__name__}"
            )
        calls.append(
            ToolCall(id=str(raw.get("id") or f"call_{i}"), name=name, arguments=arguments)
        )
    return calls


def validate_arguments(spec: FunctionSpec, arguments: dict[str, Any]) -> None:
    """按 JSON Schema 校验参数，任一违例抛 SchemaValidationError.

    实现一个够用的 JSON Schema 子集：required 必填检查 + properties 中
    声明了 type 的字段做类型校验。不做完整 JSON Schema 是为了零依赖离线可跑；
    模型选参错误绝大多数就这两类，拦截收益/实现成本比最高。
    """
    schema = spec.parameters or {}
    required = schema.get("required") or []
    missing = [key for key in required if key not in arguments]
    if missing:
        raise SchemaValidationError(
            f"调用 {spec.name} 缺少必填参数: {', '.join(missing)}"
            f"（schema 要求: {', '.join(required)}）"
        )

    properties = schema.get("properties") or {}
    for key, value in arguments.items():
        prop = properties.get(key)
        if not prop:
            continue  # schema 未声明的字段不做限制（additionalProperties 默认为 True）
        expected = prop.get("type")
        if expected is None or expected not in _TYPE_MAP:
            continue
        # bool 是 int 的子类，但 JSON Schema 里 boolean 与 integer/number 互不相容
        if expected in ("integer", "number") and isinstance(value, bool):
            raise SchemaValidationError(
                f"调用 {spec.name} 参数 {key} 应为 {expected}，实际为 boolean"
            )
        if not isinstance(value, _TYPE_MAP[expected]):
            raise SchemaValidationError(
                f"调用 {spec.name} 参数 {key} 应为 {expected}，"
                f"实际为 {type(value).__name__}: {value!r}"
            )
