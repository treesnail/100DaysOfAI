"""PromptLibrary 与 PromptTemplate 的单元测试（day036）."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.prompt_library import (
    PromptLibrary,
    PromptNotFoundError,
    PromptRenderError,
    PromptTemplate,
)
from smart_research_agent.agent.prompts import (
    DEFAULT_PROMPT_LIBRARY,
    REACT_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT_V1,
    REACT_SYSTEM_PROMPT_VERSION,
    build_default_prompt_library,
)


def _make(name: str = "greet", version: str = "v1", template: str = "你好，{name}") -> PromptTemplate:
    return PromptTemplate(name=name, version=version, template=template)


class TestPromptTemplate:
    def test_variables_extracted_in_order_and_deduped(self):
        tpl = PromptTemplate(name="t", template="{a} 与 {b}，再说一次 {a}")
        assert tpl.variables == ["a", "b"]

    def test_render_fills_variables(self):
        tpl = PromptTemplate(name="t", template="角色：{role}；任务：{task}")
        assert tpl.render(role="助手", task="总结") == "角色：助手；任务：总结"

    def test_render_missing_variable_raises(self):
        tpl = PromptTemplate(name="t", template="{a} {b}")
        with pytest.raises(PromptRenderError, match="缺少变量"):
            tpl.render(a="1")

    def test_render_ignores_non_identifier_braces(self):
        """JSON 片段等花括号不构成合法占位符，渲染时不应被误伤."""
        tpl = PromptTemplate(name="t", template='输出 JSON：{"k": 1}，任务 {task}')
        assert tpl.render(task="x") == '输出 JSON：{"k": 1}，任务 x'

    def test_created_at_auto_filled(self):
        assert _make().created_at  # 默认自动填充 ISO 时间戳


class TestPromptLibrary:
    def test_register_and_get_latest_by_default(self):
        lib = PromptLibrary()
        lib.register(_make(version="v1", template="第一版"))
        lib.register(_make(version="v2", template="第二版"))
        assert lib.get("greet").template == "第二版"

    def test_get_specific_version(self):
        lib = PromptLibrary()
        lib.register(_make(version="v1", template="第一版"))
        lib.register(_make(version="v2", template="第二版"))
        assert lib.get("greet", "v1").template == "第一版"

    def test_duplicate_name_version_rejected(self):
        lib = PromptLibrary()
        lib.register(_make())
        with pytest.raises(ValueError, match="重复注册"):
            lib.register(_make())

    def test_get_unknown_name_raises(self):
        with pytest.raises(PromptNotFoundError):
            PromptLibrary().get("nothing")

    def test_get_unknown_version_raises(self):
        lib = PromptLibrary()
        lib.register(_make())
        with pytest.raises(PromptNotFoundError):
            lib.get("greet", "v99")

    def test_render_via_library(self):
        lib = PromptLibrary()
        lib.register(_make(template="任务：{task}"))
        assert lib.render("greet", task="调研") == "任务：调研"

    def test_history_returns_versions_old_to_new(self):
        lib = PromptLibrary()
        lib.register(_make(version="v1", template="一"))
        lib.register(_make(version="v2", template="二"))
        lib.register(_make(version="v3", template="三"))
        history = lib.history("greet")
        assert [t.version for t in history] == ["v1", "v2", "v3"]
        assert [t.template for t in history] == ["一", "二", "三"]

    def test_names(self):
        lib = PromptLibrary()
        lib.register(_make(name="a"))
        lib.register(_make(name="b"))
        assert lib.names() == ["a", "b"]


class TestPersistence:
    @pytest.fixture
    def lib(self) -> PromptLibrary:
        library = PromptLibrary()
        library.register(
            PromptTemplate(
                name="react", template="工具：{tools}", version="v1", changelog="初始"
            )
        )
        library.register(
            PromptTemplate(
                name="react", template="你是助手。工具：{tools}", version="v2", changelog="四段式"
            )
        )
        return library

    def test_jsonl_roundtrip(self, lib: PromptLibrary, tmp_path):
        path = tmp_path / "prompts.jsonl"
        lib.save_jsonl(path)
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

        loaded = PromptLibrary()
        assert loaded.load_jsonl(path) == 2
        assert loaded.get("react", "v1").template == "工具：{tools}"
        assert loaded.get("react").version == "v2"
        assert loaded.get("react", "v2").changelog == "四段式"

    def test_yaml_roundtrip(self, lib: PromptLibrary, tmp_path):
        path = tmp_path / "prompts.yaml"
        lib.save_yaml(path)

        loaded = PromptLibrary()
        assert loaded.load_yaml(path) == 2
        assert loaded.get("react", "v1").changelog == "初始"
        assert loaded.render("react", "v2", tools="calc") == "你是助手。工具：calc"

    def test_persistence_preserves_created_at(self, lib: PromptLibrary, tmp_path):
        path = tmp_path / "prompts.jsonl"
        lib.save_jsonl(path)
        loaded = PromptLibrary()
        loaded.load_jsonl(path)
        assert loaded.get("react", "v1").created_at == lib.get("react", "v1").created_at


class TestDefaultLibraryMigration:
    """day026 prompts.py 常量迁移进 PromptLibrary 的验证."""

    def test_react_system_registered_with_two_versions(self):
        lib = build_default_prompt_library()
        history = lib.history("react_system")
        assert [t.version for t in history] == ["v1", "v2"]

    def test_library_templates_match_compat_constants(self):
        """库内模板与兼容常量共享同一份文本，转发不发生漂移."""
        lib = build_default_prompt_library()
        assert lib.get("react_system", "v1").template == REACT_SYSTEM_PROMPT_V1
        assert lib.get("react_system", "v2").template == REACT_SYSTEM_PROMPT

    def test_default_library_current_version_matches_constant(self):
        assert (
            DEFAULT_PROMPT_LIBRARY.get("react_system").version
            == REACT_SYSTEM_PROMPT_VERSION
        )

    def test_default_library_renders_react_prompt(self):
        rendered = DEFAULT_PROMPT_LIBRARY.render("react_system", tools="- calc: 计算")
        assert "你是 SmartResearch 智能研究助手" in rendered
        assert "- calc: 计算" in rendered

    def test_migration_preserves_changelogs(self):
        history = build_default_prompt_library().history("react_system")
        assert all(t.changelog for t in history)
