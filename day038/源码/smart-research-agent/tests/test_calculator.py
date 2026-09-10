"""计算器工具测试."""

from __future__ import annotations

import pytest

from smart_research_agent.tools.calculator import CalculatorTool


class TestCalculator:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("2 + 3", "5"),
            ("2 * 3 + 4", "10"),
            ("(2 + 3) * 4", "20"),
            ("10 / 4", "2.5"),
            ("2 ** 10", "1024"),
            ("-5 + 3", "-2"),
            ("7 % 3", "1"),
        ],
    )
    def test_arithmetic(self, expr: str, expected: str):
        tool = CalculatorTool()
        assert tool.execute(expression=expr) == expected

    def test_rejects_code_injection(self):
        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os').system('echo hacked')")
        assert result.startswith("计算失败")

    def test_rejects_syntax_error(self):
        tool = CalculatorTool()
        assert tool.execute(expression="2 +").startswith("计算失败")

    def test_schema_export(self):
        schema = CalculatorTool().to_schema()
        assert schema["name"] == "calculator"
        assert "expression" in schema["parameters"]["properties"]
