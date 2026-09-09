"""计算器工具：基于 AST 白名单的安全算术求值."""

from __future__ import annotations

import ast
import operator
from typing import Any

from smart_research_agent.tools.base import BaseTool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数字常量")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


class CalculatorTool(BaseTool):
    """算术计算工具."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "计算数学表达式，支持 + - * / ** % 与括号，例如 '2 + 3 * (4 - 1)'"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str = "", **kwargs: Any) -> str:
        try:
            result = _safe_eval(ast.parse(expression, mode="eval"))
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            return f"计算失败: {exc}"
        return str(result)
