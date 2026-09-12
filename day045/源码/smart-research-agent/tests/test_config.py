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

    def test_embedding_defaults(self, default_settings: Settings):
        """embedding 选型默认值（day041）：离线可用的 char-ngram."""
        assert default_settings.embedding_provider == "char-ngram"
        assert default_settings.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"
        assert default_settings.openai_embedding_model == "text-embedding-3-small"

    def test_local_deployment_defaults(self, default_settings: Settings):
        """本地部署默认值（day045）：默认仍是云端，本地参数对齐 Ollama 官方默认."""
        assert default_settings.llm_backend == "cloud"
        assert default_settings.local_backend == "ollama"
        assert default_settings.local_model == "qwen3:8b"
        assert default_settings.local_base_url == "http://localhost:11434/v1"
        assert default_settings.local_api_key == "ollama"
        assert default_settings.local_context_length == 4096  # Ollama 默认上下文窗口
        assert default_settings.local_keep_alive == "5m"  # Ollama 默认常驻时长
        assert default_settings.local_supports_vision is False
        assert default_settings.local_embedding_model == "nomic-embed-text"


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
