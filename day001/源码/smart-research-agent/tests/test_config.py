"""配置模块单元测试."""

from __future__ import annotations

from smart_research_agent.config import Settings


def test_settings_defaults() -> None:
    """默认配置应符合预期."""
    settings = Settings()
    assert settings.project_name == "智研 AI 助手"
    assert settings.debug is False
    assert settings.default_model == "gpt-4o-mini"
    assert settings.log_level.upper() == "INFO"


def test_settings_from_env(monkeypatch) -> None:
    """环境变量应能覆盖默认配置."""
    monkeypatch.setenv("PROJECT_NAME", "Test Agent")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings()
    assert settings.project_name == "Test Agent"
    assert settings.debug is True
    assert settings.log_level == "debug"
