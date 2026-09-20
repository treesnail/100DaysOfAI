"""day064 ``vectorstore.registry`` 的单元测试：声明、可用性探测、构造与三段式报错.

全部离线、确定性、零网络；落盘只用 pytest 的 ``tmp_path``。

这个文件里有四条断言是**故意不能靠本机环境满足**的（它们用
``monkeypatch`` 伪造 ``importlib.util.find_spec`` 的两侧）：

```text
缺依赖 / 依赖齐全   两侧都断言   → 覆盖了哪一半不取决于本机装没装 faiss
导入失败那一支      sys.modules 塞 None → 覆盖"探测说在、import 却失败"的兜底
三个前端名字未知    断言"根本不去探测依赖" → 锁住"先不变量、后环境"的校验顺序
顶层不 import 后端  读 registry.py 的 AST → 这条纪律不靠运行期副作用来证明
```

其中最重要的是**三段式报错**（缺什么 / 怎么装 / 还能用什么）那一条：
它是本模块存在的理由，所以断言写成"逐行相等"而不是"消息里含某几个词"——
后者在有人把三句话合并成一句时依然会通过。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smart_research_agent.config import Settings
from smart_research_agent.vectorstore import registry
from smart_research_agent.vectorstore.chroma_backend import ChromaVectorStore
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    FilterError,
    VectorStoreError,
)
from smart_research_agent.vectorstore.faiss_backend import FaissVectorStore
from smart_research_agent.vectorstore.flat import FlatVectorStore
from tests.chroma_fakes import FakeChromaModule
from tests.faiss_fakes import FakeFaissModule
from tests.vectorstore_samples import RECORD_IDS, sample_records

#: registry.py 自己的源码路径（两条"延迟导入"的断言要读它）.
REGISTRY_PATH = Path(registry.__file__)

#: 会被环境变量覆盖的六个配置字段。逐条删掉，让"默认 Settings()"是**真的默认**
#: ——否则某个开发者的 shell 里一次 export 就能让这些断言红掉，
#: 而那种红与代码质量无关。
_VECTOR_ENV_FIELDS: tuple[str, ...] = (
    "VECTOR_BACKEND",
    "VECTOR_METRIC",
    "VECTOR_COLLECTION",
    "VECTOR_PERSIST_PATH",
    "VECTOR_DEFAULT_TOP_K",
    "VECTOR_MIN_SCORE",
)

#: 两个可选后端的模块路径（"导入失败 → BackendUnavailable"用得到）.
FAISS_MODULE_PATH = "smart_research_agent.vectorstore.faiss_backend"
CHROMA_MODULE_PATH = "smart_research_agent.vectorstore.chroma_backend"


@pytest.fixture
def clean_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """一份确定是"默认值"的配置（不读环境变量，也不读 .env）.

    ``_env_file=None`` 是必须的：``conftest.py`` 里那个 ``default_settings``
    用 ``Settings()``，它会读当前目录的 ``.env``——而 ``.env`` 是本地文件，
    让一个断言取决于它等于把测试绑在某台机器上。
    """
    for field in _VECTOR_ENV_FIELDS:
        monkeypatch.delenv(field, raising=False)
    return Settings(_env_file=None)


def fake_find_spec(present: set[str]) -> Any:
    """伪造 ``importlib.util.find_spec``：只有 ``present`` 里的包"装上了".

    返回一个占位 spec 对象（只要不是 ``None`` 就表示可用）——registry 只判断
    "是不是 ``None``"，因此伪造物越简单越好：复杂的假 spec 会让测试开始
    依赖 ``find_spec`` 的返回结构，而那是 importlib 的事。
    """

    def find_spec(name: str) -> Any:
        return SimpleNamespace(name=name) if name in present else None

    return find_spec


def source_of_registry() -> str:
    """registry.py 的源码（``utf-8-sig``：文件若带 BOM，``ast.parse`` 会报错）."""
    return REGISTRY_PATH.read_text(encoding="utf-8-sig")


def top_level_imported_modules(source: str) -> set[str]:
    """**模块顶层** import 的模块名（取点号后的最后一段）."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {name.rsplit(".", 1)[-1] for name in names}


def all_imported_modules(source: str) -> set[str]:
    """任意层次（含函数体内）import 的模块名（取点号后的最后一段）."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {name.rsplit(".", 1)[-1] for name in names}


# --------------------------------------------------------------------------- #
# 声明表：三个 profile 的字段与顺序
# --------------------------------------------------------------------------- #


def test_backend_names_are_declared_in_reading_order() -> None:
    """顺序是契约：报告与端点按它回显（依赖递增 / 可控性递减）."""
    assert registry.backend_names() == ("flat", "faiss", "chroma")
    assert [profile.name for profile in registry.BACKEND_PROFILES] == [
        "flat",
        "faiss",
        "chroma",
    ]


def test_flat_profile_is_the_zero_dependency_reference() -> None:
    """flat 的 ``requires`` 为空元组 —— 这正是"或者改用"那一句总能成立的原因."""
    profile = registry.backend_profile("flat")

    assert profile.requires == ()
    assert profile.install_hint == ""
    assert profile.supports_persistence is True
    assert profile.native_metadata is True
    assert profile.default_metric == "cosine"
    assert profile.description != ""
    assert profile.notes != ""


def test_faiss_profile_declares_both_dependencies_and_the_install_command() -> None:
    """``requires`` 里必须有 numpy：faiss-cpu 的绑定在导入期就要它."""
    profile = registry.backend_profile("faiss")

    assert profile.requires == ("faiss", "numpy")
    assert profile.install_hint == "pip install faiss-cpu"
    assert profile.supports_persistence is True
    assert profile.native_metadata is False
    assert profile.default_metric == "cosine"


def test_chroma_profile_requires_only_chromadb() -> None:
    profile = registry.backend_profile("chroma")

    assert profile.requires == ("chromadb",)
    assert profile.install_hint == "pip install chromadb"
    assert profile.supports_persistence is True
    assert profile.native_metadata is True


@pytest.mark.parametrize(
    ("name", "backend_class"),
    [
        ("flat", FlatVectorStore),
        ("faiss", FaissVectorStore),
        ("chroma", ChromaVectorStore),
    ],
)
def test_profile_table_agrees_with_the_backend_class(name: str, backend_class: type) -> None:
    """声明表与三个后端的类属性必须逐字段一致（两份说法会分叉，所以钉住）."""
    profile = registry.backend_profile(name)

    assert profile.name == backend_class.name
    assert profile.requires == backend_class.requires
    assert profile.native_metadata == backend_class.native_metadata
    assert profile.supports_persistence == backend_class.supports_persistence


def test_factories_cover_every_profile() -> None:
    """工厂表与清单表必须一一对应（少一个就是"端点能列出但建不出来"）."""
    assert set(registry.BACKEND_FACTORIES) == set(registry.backend_names())
    assert all(callable(factory) for factory in registry.BACKEND_FACTORIES.values())


def test_unknown_backend_lists_the_three_names() -> None:
    with pytest.raises(VectorStoreError) as excinfo:
        registry.backend_profile("semantic")

    message = str(excinfo.value)
    assert "未知后端 'semantic'" in message
    assert "flat, faiss, chroma" in message


def test_backend_profile_ignores_case_and_surrounding_whitespace() -> None:
    """``" FAISS "`` 这种来自命令行/env 的写法不该被当成未知后端."""
    assert registry.backend_profile("  FAISS  ").name == "faiss"
    assert registry.backend_profile("Chroma").name == "chroma"


# --------------------------------------------------------------------------- #
# 可用性探测：find_spec 的两侧都要断言
# --------------------------------------------------------------------------- #


def test_flat_is_always_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """即使"什么都没装"，flat 也可用（它一个可选依赖都没有）."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec(set()))

    assert registry.is_available("flat") is True
    assert registry.missing_requirements("flat") == []


def test_missing_requirements_reports_only_the_absent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """numpy 在、faiss 不在 → 只缺 faiss（"（numpy 已满足）"就是从这里来的）."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"numpy"}))

    assert registry.missing_requirements("faiss") == ["faiss"]


def test_missing_requirements_is_empty_when_everything_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"faiss", "numpy"}))

    assert registry.missing_requirements("faiss") == []
    assert registry.is_available("faiss") is True


def test_missing_requirements_lists_every_missing_package_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec(set()))

    assert registry.missing_requirements("faiss") == ["faiss", "numpy"]
    assert registry.is_available("faiss") is False


def test_missing_requirements_asks_find_spec_and_never_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """探测走 ``find_spec``（按 ``requires`` 顺序问一遍），而不是真的 import."""
    probed: list[str] = []

    def find_spec(name: str) -> Any:
        probed.append(name)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert registry.missing_requirements("faiss") == ["faiss", "numpy"]
    assert probed == ["faiss", "numpy"]


def test_a_failing_probe_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``find_spec`` 对畸形名字会**抛异常**而不是返回 ``None``（空串抛 ``ValueError``）.

    探测的目的是给出一句能照着做的指引；"探测本身炸了"与"没装"对调用方是
    同一件事，所以一律记成不可用。这个分支必须有测试：少了它，
    一次线上请求会以 ``ValueError: empty module name`` 的形式失败，
    而那与"缺依赖"这个真正的问题毫无关系。
    """

    def find_spec(name: str) -> Any:
        raise ValueError(f"empty module name: {name!r}")

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert registry.missing_requirements("chroma") == ["chromadb"]
    assert registry.is_available("chroma") is False


# --------------------------------------------------------------------------- #
# describe_backends：端点直接回显的那份数据
# --------------------------------------------------------------------------- #


def test_describe_backends_is_json_serializable_with_an_available_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"numpy", "chromadb"}))

    rows = registry.describe_backends()
    rendered = json.dumps(rows, ensure_ascii=False)

    assert json.loads(rendered) == rows
    assert all("available" in row for row in rows)
    assert all("missing" in row for row in rows)
    assert all("install_hint" in row for row in rows)


def test_describe_backends_renders_availability_in_reading_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"numpy", "chromadb"}))

    rows = registry.describe_backends()

    assert [row["name"] for row in rows] == list(registry.backend_names())
    assert [row["available"] for row in rows] == [True, False, True]
    assert rows[1]["missing"] == ["faiss"]
    assert rows[1]["requires"] == ["faiss", "numpy"]
    assert rows[1]["install_hint"] == "pip install faiss-cpu"
    assert rows[0]["requires"] == []
    assert rows[0]["install_hint"] == ""


def test_describe_backends_available_agrees_with_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一个问题（"现在能不能用"）只能有一个答案（端点与代码读的是它）."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"chromadb"}))

    rows = {row["name"]: row for row in registry.describe_backends()}
    for name in registry.backend_names():
        assert rows[name]["available"] is registry.is_available(name)


# --------------------------------------------------------------------------- #
# 延迟导入：读源码的 AST，而不是靠运行期副作用
# --------------------------------------------------------------------------- #


def test_registry_never_imports_the_optional_backends_at_module_level() -> None:
    """只想用 flat 的人不该因为缺 faiss 而在 import 阶段崩掉.

    用 AST 而不是 ``sys.modules``：父包的 ``__init__.py`` 由装配阶段重写，
    它**可能**合法地导出 faiss/chroma 的类；那种改动不该把这条断言弄红。
    这里断言的是 registry.py 自己的顶层语句——**只关于本文件**。
    """
    top_level = top_level_imported_modules(source_of_registry())

    assert "faiss_backend" not in top_level
    assert "chroma_backend" not in top_level
    assert "flat" not in top_level


def test_the_optional_backends_are_imported_inside_the_factories() -> None:
    """上一条不能靠"永不 import"来满足：延迟导入必须真的存在（在函数体里）."""
    imported = all_imported_modules(source_of_registry())

    assert {"faiss_backend", "chroma_backend", "flat"} <= imported


# --------------------------------------------------------------------------- #
# resolve_backend：按 profile 传参 + 校验顺序
# --------------------------------------------------------------------------- #


def test_resolve_backend_honours_metric_and_dimension() -> None:
    store = registry.resolve_backend("flat", metric="l2", dimension=8, path="")

    assert isinstance(store, FlatVectorStore)
    assert store.name == "flat"
    assert store.metric == "l2"
    assert store.dimension == 8
    assert store.count() == 0


def test_resolve_backend_defaults_name_metric_and_path_to_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个 ``None`` 表示"按项目基线来"，而不是"没有配置"."""
    monkeypatch.setattr(
        registry,
        "app_settings",
        Settings(_env_file=None, vector_backend="flat", vector_metric="cos"),
    )

    store = registry.resolve_backend()

    assert store.name == "flat"
    assert store.metric == "cosine"


def test_resolve_backend_keeps_empty_path_and_none_path_apart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``path=None``（按配置）与 ``path=""``（这次明确不落盘）必须可区分.

    两者在配置里的字面值都是空串，但语义相反：前者是"用项目基线"，
    后者是"这次只在内存里跑"。合并它们会让"临时覆盖掉配置里的路径"
    变得**无法表达**，而那正是三级优先级里的第二级。
    """
    target = tmp_path / "snapshot.json"
    monkeypatch.setattr(
        registry, "app_settings", Settings(_env_file=None, vector_persist_path=str(target))
    )

    from_settings = registry.resolve_backend("flat", dimension=8)
    explicit_memory = registry.resolve_backend("flat", dimension=8, path="")

    assert from_settings.info().location == str(target)
    assert from_settings.info().persistent is True
    assert explicit_memory.info().location == ""
    assert explicit_memory.info().persistent is False


def test_resolve_backend_unknown_name_raises_vector_store_error() -> None:
    with pytest.raises(VectorStoreError) as excinfo:
        registry.resolve_backend("nope")

    assert "未知后端 'nope'" in str(excinfo.value)
    assert "flat, faiss, chroma" in str(excinfo.value)


def test_unknown_backend_is_rejected_before_any_dependency_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """名字都没对上，就不该去问"本机装了什么"（先不变量、后环境）."""
    probed: list[str] = []

    def find_spec(name: str) -> Any:
        probed.append(name)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    with pytest.raises(VectorStoreError):
        registry.resolve_backend("semantic")

    assert probed == []


def test_metric_error_is_reported_before_the_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """拼错的度量**装上依赖之后依然错**，所以它先报（否则要装完一轮才知道）."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec(set()))

    with pytest.raises(FilterError) as excinfo:
        registry.resolve_backend("faiss", metric="semantic")

    message = str(excinfo.value)
    assert "未知度量 'semantic'" in message
    assert "cosine, ip, l2" in message


def test_resolve_backend_does_not_pass_collection_to_flat_or_faiss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``collection`` 只有 chroma 有；多传一个键会变成 ``TypeError``（本包的取舍表）."""
    flat = registry.resolve_backend("flat", dimension=8, collection="whatever")
    assert flat.name == "flat"

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"faiss", "numpy"}))
    faiss = registry.resolve_backend(
        "faiss", dimension=8, collection="whatever", faiss_module=FakeFaissModule()
    )
    assert faiss.name == "faiss"


def test_resolve_backend_builds_a_faiss_store_when_dependencies_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """依赖齐全 + 注入假模块 → 真的把 FaissVectorStore 建出来（走通导入那一支）."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"faiss", "numpy"}))

    store = registry.resolve_backend(
        "faiss", metric="l2", dimension=8, path="", faiss_module=FakeFaissModule()
    )

    assert isinstance(store, FaissVectorStore)
    assert store.metric == "l2"
    assert store.dimension == 8
    assert store.describe_extra()["index_type"] == "IndexFlatL2"


def test_resolve_backend_keeps_the_required_dimension_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dimension=None`` 对 faiss 是必然失败——**本层不许把它吞成自己的话**."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"faiss", "numpy"}))

    with pytest.raises(VectorStoreError) as excinfo:
        registry.resolve_backend("faiss", faiss_module=FakeFaissModule())

    message = str(excinfo.value)
    assert "FAISS 在建索引之前就要知道维度" in message
    assert "flat" in message


def test_resolve_backend_passes_collection_through_to_chroma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"chromadb"}))

    store = registry.resolve_backend(
        "chroma", chromadb_module=FakeChromaModule(), collection="day064_registry"
    )

    assert isinstance(store, ChromaVectorStore)
    assert store.name == "chroma"
    assert store.describe_extra()["collection"] == "day064_registry"


# --------------------------------------------------------------------------- #
# 三段式报错："缺什么 / 怎么装 / 还能用什么"
# --------------------------------------------------------------------------- #


def test_unavailable_message_is_exactly_three_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逐行相等，而不是"含这几个词"——合并成一句时后者仍会通过."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"numpy"}))

    with pytest.raises(BackendUnavailable) as excinfo:
        registry.resolve_backend("faiss")

    assert str(excinfo.value).splitlines() == [
        "后端 faiss 不可用：缺少 faiss（numpy 已满足）。",
        "安装：pip install faiss-cpu",
        "或者改用：flat（零可选依赖，结果逐位可复现）",
    ]


def test_unavailable_message_omits_the_satisfied_clause_when_nothing_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个依赖都不在时不写"（… 已满足）"——那句话必须反映事实."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec(set()))

    with pytest.raises(BackendUnavailable) as excinfo:
        registry.resolve_backend("faiss")

    first_line = str(excinfo.value).splitlines()[0]
    assert first_line == "后端 faiss 不可用：缺少 faiss、numpy。"


def test_alternatives_only_mention_backends_that_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没装的 chroma 不该出现在"还能用什么"里（那种建议比没有更费时间）."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"numpy", "chromadb"}))

    with pytest.raises(BackendUnavailable) as excinfo:
        registry.resolve_backend("faiss")

    assert str(excinfo.value).splitlines()[2] == (
        "或者改用：flat（零可选依赖，结果逐位可复现）、"
        "chroma（向量与元数据在同一个集合里，原生支持 where）"
    )


def test_alternatives_skip_the_missing_backend_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chroma 不可用时，它的"还能用什么"里不能出现 chroma 自己."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec(set()))

    with pytest.raises(BackendUnavailable) as excinfo:
        registry.resolve_backend("chroma")

    third_line = str(excinfo.value).splitlines()[2]
    assert third_line == "或者改用：flat（零可选依赖，结果逐位可复现）"
    assert "chroma" not in third_line


@pytest.mark.parametrize(
    ("name", "module_path", "install_hint"),
    [
        ("faiss", FAISS_MODULE_PATH, "pip install faiss-cpu"),
        ("chroma", CHROMA_MODULE_PATH, "pip install chromadb"),
    ],
)
def test_import_failure_is_converted_into_the_same_three_sentences(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    module_path: str,
    install_hint: str,
) -> None:
    """探测说"在"、``import`` 却失败（模块残缺 / 平台不匹配的 wheel）时也要三句.

    ``sys.modules[module_path] = None`` 是让 ``import`` 必然失败的标准手法：
    Python 见到 ``None`` 会直接抛 ``ImportError``，与文件是否真的在盘上无关。
    这一支是"探测与使用不是一回事"的兜底，因此必须有测试把它逼出来——
    否则它只会以 ``ImportError`` 的形式冒到调用方脸上。
    """
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"faiss", "numpy", "chromadb"}))
    monkeypatch.setitem(sys.modules, module_path, None)

    with pytest.raises(BackendUnavailable) as excinfo:
        registry.resolve_backend(name, dimension=8)

    lines = str(excinfo.value).splitlines()
    assert lines[0].startswith(f"后端 {name} 不可用：缺少 ")
    assert "导入失败" in lines[0]
    assert lines[1] == f"安装：{install_hint}"
    assert lines[2].startswith("或者改用：flat（")


# --------------------------------------------------------------------------- #
# create_backend：settings 是基线，overrides 最高
# --------------------------------------------------------------------------- #


def test_create_backend_with_default_settings_is_an_in_memory_flat_store(
    clean_settings: Settings,
) -> None:
    """默认配置 = flat + cosine + 不落盘（见 config.py 里那六个字段的注释）."""
    store = registry.create_backend(clean_settings)

    assert isinstance(store, FlatVectorStore)
    assert store.name == "flat"
    assert store.metric == clean_settings.vector_metric == "cosine"
    assert store.count() == 0
    info = store.info()
    assert info.persistent is False
    assert info.location == ""
    assert info.dimension == 0


def test_create_backend_normalizes_metric_aliases(clean_settings: Settings) -> None:
    """别名收敛只走 ``metrics.normalize_metric`` 一处（契约文件里的那张表）.

    注意 ``dot``：``METRIC_ALIASES`` 把它归到 ``ip``（内积），**不是** cosine
    ——"点积"与"余弦"在未归一化时是不同的东西（见 ``metrics`` 模块 docstring）。
    第二段规格的 D.4 里把 ``dot`` 写成了 cosine 的别名，与契约表不符；
    本测试按代码事实（``dot → ip``）锁住行为，差异已写进交付报告。
    """
    assert registry.create_backend(clean_settings, metric="cos").metric == "cosine"
    assert registry.create_backend(clean_settings, metric="euclidean").metric == "l2"
    assert registry.create_backend(clean_settings, metric="dot").metric == "ip"


def test_create_backend_overrides_win_over_settings(clean_settings: Settings) -> None:
    """配置里写着 faiss+l2，但本次覆盖成 flat+cos → **不该碰 faiss**（本机没装）."""
    settings = clean_settings.model_copy(update={"vector_backend": "faiss", "vector_metric": "l2"})

    store = registry.create_backend(settings, backend="flat", metric="cos")

    assert store.name == "flat"
    assert store.metric == "cosine"


def test_create_backend_reads_the_persist_path_from_settings(tmp_path: Path) -> None:
    """``vector_persist_path`` 非空 → 落盘并可被下一次构造自动载入."""
    target = tmp_path / "nested" / "snapshot.json"
    settings = Settings(_env_file=None, vector_persist_path=str(target))

    first = registry.create_backend(settings)
    first.upsert(sample_records(metric="cosine"))
    written = first.persist()

    assert written == str(target)
    assert target.exists()

    second = registry.create_backend(settings)
    assert second.count() == 6
    assert sorted(second.ids()) == sorted(RECORD_IDS)
    assert second.info().persistent is True


@pytest.mark.parametrize(("key", "value"), [("top_k", 3), ("min_score", 0.5)])
def test_create_backend_rejects_query_time_overrides(
    clean_settings: Settings, key: str, value: Any
) -> None:
    """查询期参数必须被**明确拒收**：静默忽略会让它看起来生效了（见 docstring）."""
    with pytest.raises(VectorStoreError) as excinfo:
        registry.create_backend(clean_settings, **{key: value})

    message = str(excinfo.value)
    assert key in message
    assert "VectorIngestPipeline" in message


def test_create_backend_backend_override_reaches_resolve_backend(
    clean_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec(set()))

    with pytest.raises(BackendUnavailable) as excinfo:
        registry.create_backend(clean_settings, backend="faiss", dimension=8)

    assert "pip install faiss-cpu" in str(excinfo.value)
    assert "flat" in str(excinfo.value)


def test_create_backend_passes_backend_specific_kwargs_through(
    clean_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非配置项的关键字原样透传（测试注入 / 高级用法），由后端自己校验."""
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec({"faiss", "numpy"}))

    store = registry.create_backend(
        clean_settings, backend="faiss", dimension=8, faiss_module=FakeFaissModule()
    )

    assert isinstance(store, FaissVectorStore)
    assert store.metric == "cosine"


def test_create_backend_reports_a_settings_object_missing_the_field() -> None:
    """字段名写错时要报出缺的是哪一个，而不是一个裸的 AttributeError."""
    with pytest.raises(VectorStoreError) as excinfo:
        registry.create_backend(SimpleNamespace())

    message = str(excinfo.value)
    assert "vector_backend" in message
    assert "SimpleNamespace" in message
