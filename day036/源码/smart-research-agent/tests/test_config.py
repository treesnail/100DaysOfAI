"""配置模块单元测试：验证 Settings 的默认值、类型与环境变量覆盖."""

from __future__ import annotations

import pytest

from smart_research_agent.config import Settings


class TestDefaultValues:
    """默认配置应满足开箱即用."""

    def test_default_values(self, default_settings: Settings):
        assert default_settings.project_name == "智研 AI 助手"
        assert default_settings.debug is False
        assert default_settings.log_level == "INFO"
        assert default_settings.default_model == "gpt-4o-mini"

    def test_openai_api_key_defaults_to_none(self, default_settings: Settings):
        """API Key 默认为空，由 .env 或环境变量注入，绝不硬编码."""
        assert default_settings.openai_api_key is None


class TestEnvOverride:
    """环境变量应能覆盖默认值."""

    def test_debug_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEBUG", "true")
        assert Settings().debug is True

    def test_model_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")
        assert Settings().default_model == "gpt-4o"


class TestTypeCoercion:
    """Pydantic 的类型转换行为."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("0", False),
        ],
    )
    def test_debug_accepts_common_bool_strings(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ):
        monkeypatch.setenv("DEBUG", raw)
        assert Settings().debug is expected
