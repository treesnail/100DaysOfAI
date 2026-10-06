"""工具注册表测试."""

from __future__ import annotations

import pytest

from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert registry.get("calculator") is not None
        assert registry.get("not_exist") is None

    def test_duplicate_register_raises(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="重复注册"):
            registry.register(CalculatorTool())

    def test_describe_contains_tool_info(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        text = registry.describe()
        assert "calculator" in text
        assert "数学表达式" in text

    def test_schemas_export(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = registry.schemas()
        assert schemas[0]["name"] == "calculator"
