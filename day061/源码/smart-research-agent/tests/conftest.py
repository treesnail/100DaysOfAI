"""pytest 共享 fixtures：为测试提供可复用的初始化逻辑."""

from __future__ import annotations

import pytest

from smart_research_agent.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    """提供一个未受环境变量污染的默认 Settings 实例.

    注意：fixture 每次测试都会新建实例，避免测试间状态泄漏。
    """
    return Settings()


@pytest.fixture
def tmp_env(monkeypatch: pytest.MonkeyPatch):
    """提供环境变量隔离工具.

    用法::

        def test_xxx(tmp_env):
            tmp_env.setenv("LOG_LEVEL", "DEBUG")
            ...
    """
    return monkeypatch
