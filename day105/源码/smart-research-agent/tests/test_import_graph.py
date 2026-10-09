"""``import_graph``：把 100 天的模块编成一张可复算的依赖图（day103）.

本文件覆盖新包的六个模块：口径表、失败族、解析、图、性质与表。
样本有两类：**真实源码**（``smart_research_agent/**/*.py``）与
**注入的小模块**（构造反例时用）。因此这一课考的就是
"同一批文件能不能两次编出同一张图、每一个 import 目标能不能被归属、
以及这张图到底有没有环"。

> 注意：本文件**不写死**节点数 / 边数 / 环数 / 摘要——它们会随着仓库新增子包而变化。
> 断言一律用"结构不变式"或下界；另有**一条子进程测试**专门钉住"跨进程可复算"。
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest

from smart_research_agent.import_graph import errors, graph as graph_module, parse as parse_module
from smart_research_agent.import_graph import study, types, verify

# --------------------------------------------------------------------------- 注入样本


def _ref(
    raw: str,
    target: str | None,
    *,
    kind: str = types.EDGE_KIND_ABSOLUTE,
    internal: bool = True,
) -> parse_module.ImportRef:
    """一条 import 记录（默认：内部、绝对导入）."""
    return parse_module.ImportRef(raw=raw, kind=kind, target=target, internal=internal)


def _module(
    module: str,
    *refs: parse_module.ImportRef,
    is_package: bool = False,
) -> parse_module.ScannedModule:
    """一份被解析的小模块（路径是造的，内容只用于建图）."""
    return parse_module.ScannedModule(
        module=module, is_package=is_package, path=f"/x/{module}.py", refs=tuple(refs)
    )


def _chain_modules() -> tuple[parse_module.ScannedModule, ...]:
    """一条链：pkg.a → pkg.b → pkg.c（外加一个包节点）。"""
    return (
        _module("pkg", is_package=True),
        _module("pkg.a", _ref("pkg.b", "pkg.b")),
        _module("pkg.b", _ref("pkg.c", "pkg.c")),
        _module("pkg.c"),
    )


def _cycle_modules() -> tuple[parse_module.ScannedModule, ...]:
    """一个环：pkg.a ↔ pkg.b。"""
    return (
        _module("pkg.a", _ref("pkg.b", "pkg.b")),
        _module("pkg.b", _ref("pkg.a", "pkg.a")),
    )


def _isolated_modules() -> tuple[parse_module.ScannedModule, ...]:
    """两个互不依赖的模块（用来把"存在依赖链"这条下界顶破）。"""
    return (_module("pkg.a"), _module("pkg.b"))


# --------------------------------------------------------------------------- 口径表


def test_edge_kinds_and_directions_are_closed() -> None:
    """两类边与两个方向：名单与说明逐键对齐。"""
    assert len(types.EDGE_KINDS) == 2
    assert set(types.EDGE_KIND_DESCRIPTIONS) == set(types.EDGE_KINDS)
    assert len(types.DIRECTIONS) == 2
    assert set(types.DIRECTION_DESCRIPTIONS) == set(types.DIRECTIONS)
    assert types.require_edge_kind(types.EDGE_KIND_RELATIVE) == types.EDGE_KIND_RELATIVE
    assert types.require_direction(types.DIRECTION_UP) == types.DIRECTION_UP
    with pytest.raises(errors.ParameterError, match="未知的边种类"):
        types.require_edge_kind("nope")
    with pytest.raises(errors.ParameterError, match="未知的闭包方向"):
        types.require_direction("sideways")


def test_seven_properties_and_three_criteria() -> None:
    """7 条性质：名单与规格表逐键对齐，且三类判据都有性质。"""
    assert len(types.IMPORT_GRAPH_PROPERTIES) == 7
    assert set(types.PROPERTY_SPECS) == set(types.IMPORT_GRAPH_PROPERTIES)
    criteria = {spec.criterion for spec in types.property_specs()}
    assert criteria == set(types.CRITERIA)
    assert criteria == {
        types.CRITERION_EQUALITY,
        types.CRITERION_UPPER_BOUND,
        types.CRITERION_LOWER_BOUND,
    }


def test_notes_are_ten_and_boundaries_five() -> None:
    """十条笔记（有序）与五条边界。"""
    assert len(types.IMPORT_GRAPH_NOTES) == 10
    assert types.IMPORT_GRAPH_NOTES_ORDER == tuple(types.IMPORT_GRAPH_NOTES)
    assert all(types.IMPORT_GRAPH_NOTES.values())
    assert len(types.IMPORT_GRAPH_BOUNDARIES) == 5
    assert all(types.IMPORT_GRAPH_BOUNDARIES)


def test_docs_of_kind_and_direction() -> None:
    """两个说明函数：合法值返回一句话，非法值当场拒绝。"""
    assert types.docs_of_kind(types.EDGE_KIND_ABSOLUTE)
    assert types.docs_of_direction(types.DIRECTION_DOWN)
    with pytest.raises(errors.ParameterError, match="未知的边种类"):
        types.docs_of_kind("nope")
    with pytest.raises(errors.ParameterError, match="未知的闭包方向"):
        types.docs_of_direction("nope")


def test_require_helpers() -> None:
    """两个 require 帮手：合法值原样返回，非法值当场拒绝。"""
    assert types.require_positive_int("limit", 3) == 3
    assert types.require_property(types.PROPERTY_GRAPH_IS_REPRODUCIBLE)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("limit", 0)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("limit", True)
    with pytest.raises(errors.ParameterError, match="未知的性质"):
        types.require_property("nope")


def _property_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 PropertySpec 关键字参数."""
    base: dict[str, object] = {
        "id": types.PROPERTY_GRAPH_IS_REPRODUCIBLE,
        "description": "说明",
        "criterion": types.CRITERION_EQUALITY,
        "failure": "失败意味着",
    }
    base.update(overrides)
    return base


def test_property_spec_validates_fields() -> None:
    """性质记录的两条护栏与两个渲染方法。"""
    spec = types.PropertySpec(**_property_kwargs())
    assert spec.to_dict()["criterion"] == types.CRITERION_EQUALITY
    assert spec.line().startswith("[equality]")
    with pytest.raises(errors.ParameterError, match="不能为空"):
        types.PropertySpec(**_property_kwargs(description=""))
    with pytest.raises(errors.ParameterError, match="判据"):
        types.PropertySpec(**_property_kwargs(criterion="nope"))


# --------------------------------------------------------------------------- 失败族


def test_error_families_are_closed() -> None:
    """六个失败族：类表与处置表逐键对齐，且都是 ValueError 的子类。"""
    assert len(errors.FAMILY_OUTCOMES) == 6
    assert set(errors.FAMILY_OUTCOMES) == set(errors._FAMILY_CLASSES)
    for name in errors.FAMILY_OUTCOMES:
        cls = errors._FAMILY_CLASSES[name]
        assert issubclass(cls, errors.GraphError)
        assert issubclass(cls, ValueError)


def test_returned_and_absent_families_are_declared() -> None:
    """回来的族与缺席的族都是常量，且各自带理由。"""
    assert errors.RETURNED_FAMILY == "AssemblyError"
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY_REASON


def test_each_family_is_raisable() -> None:
    """每个失败族都能被 raise / except。"""
    for cls in errors._FAMILY_CLASSES.values():
        with pytest.raises(cls):
            raise cls("boom")


# --------------------------------------------------------------------------- 解析


def test_module_files_are_sorted_python_files() -> None:
    """被扫描的文件：非空、按相对路径排序、都以 .py 结尾。"""
    files = parse_module.module_files()
    assert files
    relative = [path.relative_to(parse_module.package_dir()).as_posix() for path in files]
    assert relative == sorted(relative)
    assert all(path.suffix == ".py" for path in files)


def test_module_index_views() -> None:
    """模块索引：模块名单排序唯一，包是它的子集，前缀能被解析。"""
    index = parse_module.module_index()
    assert index.modules == tuple(sorted(index.modules))
    assert len(set(index.modules)) == len(index.modules)
    assert set(index.packages) <= set(index.modules)
    assert index.to_dict()["modules"] == len(index.modules)
    assert "索引" in index.line()


def test_module_index_guards() -> None:
    """模块索引的两条护栏：空名单与未排序名单。"""
    with pytest.raises(errors.ParseError, match="不能为空"):
        parse_module.ModuleIndex(modules=(), packages=(), prefixes=())
    with pytest.raises(errors.ParseError, match="排序且不重复"):
        parse_module.ModuleIndex(modules=("b", "a"), packages=(), prefixes=())


def test_resolve_candidates_prefers_module_then_package() -> None:
    """候选解析：先挑已知模块，再挑已知包 / 前缀，都不行返回 None。"""
    index = parse_module.ModuleIndex(
        modules=("pkg", "pkg.a", "pkg.a.b"),
        packages=("pkg", "zebra"),
        prefixes=("pkg", "pkg.a", "pkg.deep"),
    )
    assert index.resolve_candidates(("pkg.a", "pkg.a.b")) == "pkg.a.b"
    assert index.resolve_candidates(("pkg.a", "pkg.a.zzz")) == "pkg.a"
    assert index.resolve_candidates(("pkg.deep",)) == "pkg.deep"
    assert index.resolve_candidates(("zebra",)) == "zebra"
    assert index.resolve_candidates(("nope",)) is None
    assert index.resolve_candidates(()) is None


def test_resolve_candidates_is_tie_broken_by_name() -> None:
    """长度相同的候选按**名字**兜顺序（否则跨进程结果会漂移）。"""
    index = parse_module.ModuleIndex(
        modules=("pkg.b", "pkg.c"),
        packages=(),
        prefixes=(),
    )
    assert index.resolve_candidates(("pkg.c", "pkg.b")) == "pkg.b"
    assert index.resolve_candidates(("pkg.b", "pkg.c")) == "pkg.b"


def test_module_name_and_package_flag(tmp_path: pathlib.Path) -> None:
    """模块名：``__init__.py`` 折成包名，普通文件折成点分名。"""
    root = tmp_path / types.PACKAGE_NAME
    nested = root / "widget"
    nested.mkdir(parents=True)
    init = nested / "__init__.py"
    init.write_text("", encoding="utf-8")
    plain = nested / "gadget.py"
    plain.write_text("", encoding="utf-8")
    (root / "__init__.py").write_text("", encoding="utf-8")
    assert parse_module.module_name(init, tmp_path) == "smart_research_agent.widget"
    assert parse_module.module_name(plain, tmp_path) == "smart_research_agent.widget.gadget"
    assert parse_module.module_name(root / "__init__.py", tmp_path) == types.PACKAGE_NAME
    assert parse_module.is_package_file(init) is True
    assert parse_module.is_package_file(plain) is False


def test_module_name_rejects_outside_path(tmp_path: pathlib.Path) -> None:
    """一个不在被扫描目录下的路径当场被拒。"""
    root = tmp_path / types.PACKAGE_NAME
    root.mkdir(parents=True)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(errors.ParseError, match="不在被扫描的包目录"):
        parse_module.module_name(outside, tmp_path)


def test_package_dir_and_empty_scan(tmp_path: pathlib.Path) -> None:
    """包目录：不存在时抛；存在但没有 .py 也抛。"""
    assert parse_module.package_dir().is_dir()
    with pytest.raises(errors.ParseError, match="包目录不存在"):
        parse_module.package_dir(tmp_path)
    (tmp_path / types.PACKAGE_NAME).mkdir()
    with pytest.raises(errors.ParseError, match=r"一份 \.py 都没有"):
        parse_module.module_files(tmp_path)


def test_parse_source_rejects_syntax_error() -> None:
    """语法错当场抛 ParseError。"""
    with pytest.raises(errors.ParseError, match="语法错误"):
        parse_module.parse_source("def :(", origin="<mem>")


def test_read_source_accepts_bom(tmp_path: pathlib.Path) -> None:
    """读源码容忍 BOM。"""
    target = tmp_path / "m.py"
    target.write_text("import os\n", encoding="utf-8-sig")
    assert "import os" in parse_module.read_source(target)


def test_relative_base_rules() -> None:
    """相对导入的折层规则：level - 1 层点，越界返回空串。"""
    assert (
        parse_module.relative_base(
            module="smart_research_agent.foo.bar", is_package=False, level=1, imported="base"
        )
        == "smart_research_agent.foo.base"
    )
    assert (
        parse_module.relative_base(
            module="smart_research_agent.foo.bar", is_package=False, level=2, imported="other"
        )
        == "smart_research_agent.other"
    )
    assert (
        parse_module.relative_base(
            module="smart_research_agent.foo", is_package=True, level=1, imported=None
        )
        == "smart_research_agent.foo"
    )
    assert (
        parse_module.relative_base(
            module="smart_research_agent.foo", is_package=True, level=5, imported="x"
        )
        == ""
    )


def test_imports_of_covers_the_forms() -> None:
    """一条 AST 里的四种写法：绝对 import、from-子模块、相对导入、外部导入。"""
    index = parse_module.ModuleIndex(
        modules=(
            "smart_research_agent",
            "smart_research_agent.foo",
            "smart_research_agent.foo.base",
            "smart_research_agent.foo.types",
        ),
        packages=("smart_research_agent", "smart_research_agent.foo"),
        prefixes=("smart_research_agent", "smart_research_agent.foo"),
    )
    import ast

    tree = ast.parse(
        "import os\n"
        "import smart_research_agent.foo.types\n"
        "from smart_research_agent.foo import base\n"
        "from . import types\n"
    )
    refs = parse_module.imports_of(
        tree, module="smart_research_agent.foo.base", is_package=False, index=index
    )
    by_raw = {ref.raw: ref for ref in refs}
    assert by_raw["os"].internal is False
    assert by_raw["os"].target is None
    assert by_raw["smart_research_agent.foo.types"].target == "smart_research_agent.foo.types"
    assert by_raw["smart_research_agent.foo"].target == "smart_research_agent.foo.base"
    assert by_raw["."].kind == types.EDGE_KIND_RELATIVE
    assert by_raw["."].target == "smart_research_agent.foo.types"


def test_beyond_root_relative_import_is_unresolved() -> None:
    """减点减到包根之外的相对导入，被记成"解析不出来的包内 import"。"""
    import ast

    index = parse_module.ModuleIndex(
        modules=("smart_research_agent",),
        packages=("smart_research_agent",),
        prefixes=(),
    )
    tree = ast.parse("from ...baz import X\n")
    refs = parse_module.imports_of(
        tree, module="smart_research_agent.foo.bar", is_package=False, index=index
    )
    assert refs[0].kind == types.EDGE_KIND_RELATIVE
    assert refs[0].internal is True
    assert refs[0].target is None
    assert refs[0].unresolved is True


def test_import_ref_guards_and_rendering() -> None:
    """import 记录的三条护栏与两个渲染方法。"""
    ref = _ref("pkg.b", "pkg.b")
    assert ref.resolved is True
    assert ref.unresolved is False
    assert ref.to_dict()["target"] == "pkg.b"
    assert "→ absolute" in ref.line()
    missing = _ref(".gone", None, kind=types.EDGE_KIND_RELATIVE)
    assert missing.unresolved is True
    assert "未解析" in missing.line()
    with pytest.raises(errors.ParseError, match="不能为空"):
        _ref("", "pkg.b")
    with pytest.raises(errors.ParameterError, match="未知的边种类"):
        parse_module.ImportRef(raw="x", kind="nope", target="y", internal=True)


def test_scanned_module_views_and_guards() -> None:
    """被解析模块：目标去重、去掉自己、未解析可挑出。"""
    module = _module(
        "pkg.a",
        _ref("pkg.b", "pkg.b"),
        _ref("pkg.b", "pkg.b"),
        _ref("pkg.a", "pkg.a"),
        _ref(".gone", None, kind=types.EDGE_KIND_RELATIVE),
    )
    assert module.targets() == ("pkg.b",)
    assert len(module.unresolved()) == 1
    assert module.to_dict()["targets"] == 1
    assert "pkg.a" in module.line()
    with pytest.raises(errors.ParseError, match="模块名不能为空"):
        _module("")


def test_scan_module_reads_a_real_file() -> None:
    """扫一份真实文件：模块名对得上、目标非空。"""
    path = parse_module.PROJECT_ROOT / "smart_research_agent" / "import_graph" / "parse.py"
    scanned = parse_module.scan_module(path, parse_module.PROJECT_ROOT)
    assert scanned.module == "smart_research_agent.import_graph.parse"
    assert scanned.is_package is False
    assert scanned.targets()


def test_scan_all_is_reproducible_in_process() -> None:
    """两次扫描逐位相同（同一进程内）。"""
    first = parse_module.scan_all()
    second = parse_module.scan_all()
    assert [m.to_dict() for m in first] == [m.to_dict() for m in second]
    assert parse_module.expected_modules() == tuple(sorted(m.module for m in first))


def test_graph_digest_is_stable_across_processes() -> None:
    """**跨进程**可复算：两个不同的哈希种子必须给出同一个摘要.

    这是本课最硬的一条测试：它正是"并列必须按名字兜顺序"那条纪律的守卫
    （``resolve_candidates`` 曾因为按长度排序而让边在两次运行之间换目标）。
    """
    script = (
        "from smart_research_agent.import_graph.graph import build_graph;"
        "from smart_research_agent.import_graph.parse import scan_all;"
        "print(build_graph(scan_all()).digest())"
    )
    digests = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=parse_module.PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1]


def test_unresolved_and_require_resolved() -> None:
    """未解析清单：默认全解析；注入一条对不上的相对导入时抛 ImportTargetError。"""
    assert parse_module.unresolved_of() == ()
    assert parse_module.require_resolved() is not None
    broken = (_module("pkg.a", _ref(".gone", None, kind=types.EDGE_KIND_RELATIVE)),)
    assert parse_module.unresolved_of(broken) == (("pkg.a", ".gone"),)
    with pytest.raises(errors.ImportTargetError, match="目标对不上号"):
        parse_module.require_resolved(broken)


def test_scan_lines_has_summary() -> None:
    """解析逐行打印，末行是合计。"""
    lines = parse_module.scan_lines(_chain_modules())
    assert lines[0].startswith("解析表")
    assert "4 个模块" in lines[-1]


# --------------------------------------------------------------------------- 图


def test_edge_views_and_guards() -> None:
    """边：排序键、跨包判断、渲染与两条护栏。"""
    edge = graph_module.Edge("smart_research_agent.a.app", "smart_research_agent.b.core")
    assert edge.is_cross_package is True
    assert edge.key == ("smart_research_agent.a.app", "smart_research_agent.b.core", "absolute")
    assert edge.to_dict()["kind"] == types.EDGE_KIND_ABSOLUTE
    assert "→" in edge.line()
    same = graph_module.Edge("smart_research_agent.a.app", "smart_research_agent.a.core")
    assert same.is_cross_package is False
    with pytest.raises(errors.ParameterError, match="两个端点都不能为空"):
        graph_module.Edge("", "x")
    with pytest.raises(errors.ParameterError, match="未知的边种类"):
        graph_module.Edge("a", "b", "nope")


def test_build_graph_shape() -> None:
    """小图：节点排序、包节点、边排序去重。"""
    graph = graph_module.build_graph(_chain_modules())
    assert graph.nodes == tuple(sorted(graph.nodes))
    assert graph.packages == ("pkg",)
    assert graph.node_count == 4
    assert graph.edge_count == 2
    assert graph.successors("pkg.a") == ("pkg.b",)
    assert graph.predecessors("pkg.c") == ("pkg.b",)
    assert graph.package_of("pkg.a") == "a"
    assert {name for name, _members in graph.package_groups()} == {"a", "b", "c", "pkg"}
    assert graph.to_dict()["edges"] == 2
    assert "图：" in graph.line()


def test_graph_guards() -> None:
    """图的六条护栏：空、未排序、包不在节点里、边未排序、边重复、端点未知。"""
    with pytest.raises(errors.GraphBuildError, match="不能没有节点"):
        graph_module.ModuleGraph(nodes=(), packages=(), edges=())
    with pytest.raises(errors.GraphBuildError, match="排序且不重复"):
        graph_module.ModuleGraph(nodes=("b", "a"), packages=(), edges=())
    with pytest.raises(errors.GraphBuildError, match="包节点不在节点集里"):
        graph_module.ModuleGraph(nodes=("a",), packages=("z",), edges=())
    with pytest.raises(errors.GraphBuildError, match="边没有排序"):
        graph_module.ModuleGraph(
            nodes=("a", "b"),
            packages=(),
            edges=(graph_module.Edge("b", "a"), graph_module.Edge("a", "b")),
        )
    with pytest.raises(errors.GraphBuildError, match="边里有重复项"):
        graph_module.ModuleGraph(
            nodes=("a", "b"),
            packages=(),
            edges=(graph_module.Edge("a", "b"), graph_module.Edge("a", "b")),
        )
    with pytest.raises(errors.GraphBuildError, match="端点"):
        graph_module.ModuleGraph(nodes=("a",), packages=(), edges=(graph_module.Edge("a", "z"),))


def test_graph_lookups_reject_unknown_nodes() -> None:
    """四个按节点取值的入口在未知节点上当场拒绝。"""
    graph = graph_module.build_graph(_chain_modules())
    for call in (
        lambda: graph.package_of("nope"),
        lambda: graph.successors("nope"),
        lambda: graph.predecessors("nope"),
        lambda: graph.edges_of("nope"),
        lambda: graph.closure("nope"),
        lambda: graph.component_key("nope"),
    ):
        with pytest.raises(errors.ParameterError, match="图里没有节点"):
            call()


def test_graph_cycles_and_condensation() -> None:
    """环：强连通分量、分量键、凝缩边、线性序都在这个小环上验证。"""
    graph = graph_module.build_graph(_cycle_modules())
    assert graph.cycles() == (("pkg.a", "pkg.b"),)
    assert graph.sccs() == (("pkg.a", "pkg.b"),)
    assert graph.component_key("pkg.b") == "pkg.a"
    assert graph.condensation_edges() == ()
    assert graph.topological_order() == ("pkg.a",)
    assert dict(graph.order_index()) == {"pkg.a": 0, "pkg.b": 0}
    assert graph.cross_package_edges() == graph.edges
    assert graph.edges_of("pkg.a")


def test_graph_linear_order_on_a_chain() -> None:
    """链上的线性序：每个节点一个分量，顺序跟着边走。"""
    graph = graph_module.build_graph(_chain_modules())
    order = graph.topological_order()
    positions = {key: index for index, key in enumerate(order)}
    assert positions["pkg.a"] < positions["pkg.b"] < positions["pkg.c"]


def test_graph_closures_and_direction() -> None:
    """闭包：下游与上游不对称，方向非法当场拒绝。"""
    graph = graph_module.build_graph(_chain_modules())
    assert graph.closure("pkg.a", types.DIRECTION_DOWN) == ("pkg.a", "pkg.b", "pkg.c")
    assert graph.closure("pkg.c", types.DIRECTION_DOWN) == ("pkg.c",)
    assert graph.closure("pkg.c", types.DIRECTION_UP) == ("pkg.a", "pkg.b", "pkg.c")
    with pytest.raises(errors.ParameterError, match="未知的闭包方向"):
        graph.closure("pkg.a", "sideways")


def test_graph_rollup_to_packages() -> None:
    """折成包级：节点变成子包名，包内边被丢掉。"""
    modules = (
        _module("smart_research_agent.a.one", _ref("x", "smart_research_agent.b.two")),
        _module("smart_research_agent.a.two", _ref("y", "smart_research_agent.a.one")),
        _module("smart_research_agent.b.two"),
    )
    graph = graph_module.build_graph(modules)
    rolled = graph.rollup_to_packages()
    assert rolled.nodes == ("a", "b")
    assert [(edge.source, edge.target) for edge in rolled.edges] == [("a", "b")]
    assert rolled.rollup_to_packages().nodes == ("a", "b")


def test_graph_recomputable_and_acyclic_guards() -> None:
    """可复算与"拒绝有环"两条路。"""
    modules = _chain_modules()
    first = graph_module.build_graph(modules)
    second = graph_module.build_graph(modules)
    assert first.diff_count(second) == 0
    assert first.digest() == second.digest()
    assert graph_module.graph_digest(first) == first.digest()
    assert graph_module.require_reproducible(first, second) is first
    other = graph_module.build_graph(_cycle_modules())
    assert first.diff_count(other) > 0
    with pytest.raises(errors.GraphBuildError, match="两张图差"):
        graph_module.require_reproducible(first, other)
    assert graph_module.require_acyclic(first) is first
    cyclic = graph_module.build_graph(_cycle_modules())
    with pytest.raises(errors.CycleError, match="图里有 1 个环"):
        graph_module.require_acyclic(cyclic)


def test_expected_packages_and_graph_lines() -> None:
    """包的名单与逐行打印（可截断，非法 limit 被拒）。"""
    graph = graph_module.build_graph(_chain_modules())
    assert graph_module.expected_packages(graph) == ("a", "b", "c", "pkg")
    lines = graph_module.graph_lines(graph, limit=1)
    assert lines[0].startswith("依赖图")
    assert len(lines) == 3
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        graph_module.graph_lines(graph, limit=0)


# --------------------------------------------------------------------------- 性质


def test_crosscheck_three_criteria() -> None:
    """三类判据各自的判定与渲染。"""
    equal = verify.CrossCheck(name="相等", left="a", right="b", reading=2.0, expected=2.0)
    assert equal.criterion == types.CRITERION_EQUALITY
    assert equal.passed is True
    assert "读数" in equal.line()
    upper = verify.CrossCheck(name="上界", left="a", right="b", reading=0.0, expected=0.0, upper_bound=0.0)
    assert upper.criterion == types.CRITERION_UPPER_BOUND
    assert upper.passed is True
    assert "≤ 上界" in upper.line()
    lower = verify.CrossCheck(name="下界", left="a", right="b", reading=1.0, expected=1.0, lower_bound=1.0)
    assert lower.criterion == types.CRITERION_LOWER_BOUND
    assert lower.passed is True
    assert "≥ 下界" in lower.line()
    assert lower.to_dict()["passed"] is True


def test_crosscheck_failures_and_guards() -> None:
    """判据不满足时的读数 + 五条构造护栏。"""
    failed = verify.CrossCheck(name="下界", left="a", right="b", reading=0.0, expected=1.0, lower_bound=1.0)
    assert failed.passed is False
    assert "不满足" in failed.line()
    tolerant = verify.CrossCheck(
        name="容差相等", left="a", right="b", reading=1.0, expected=1.0 + 1e-15, exact=False
    )
    assert tolerant.criterion == types.CRITERION_EQUALITY
    assert tolerant.passed is True
    with pytest.raises(errors.ParameterError, match="不能为空"):
        verify.CrossCheck(name="", left="a", right="b", reading=0.0, expected=0.0)
    with pytest.raises(errors.NumericError, match="必须有限"):
        verify.CrossCheck(name="n", left="a", right="b", reading=float("nan"), expected=0.0)
    with pytest.raises(errors.ParameterError, match="同时给了上界与下界"):
        verify.CrossCheck(
            name="n", left="a", right="b", reading=0.0, expected=0.0, upper_bound=0.0, lower_bound=0.0
        )
    with pytest.raises(errors.NumericError, match="上界必须是有限非负数"):
        verify.CrossCheck(name="n", left="a", right="b", reading=0.0, expected=0.0, upper_bound=-1.0)
    with pytest.raises(errors.NumericError, match="下界必须是有限数"):
        verify.CrossCheck(name="n", left="a", right="b", reading=0.0, expected=0.0, lower_bound=float("inf"))


def test_property_outcome_guards() -> None:
    """PropertyOutcome 的两条护栏与 criterion / line / to_dict。"""
    ok = verify.PropertyOutcome(
        name="x", applicable=True, passed=True, evidence=("a",),
        cross_check=verify.CrossCheck(name="c", left="l", right="r", reading=1.0, expected=1.0),
    )
    assert ok.criterion == types.CRITERION_EQUALITY
    assert ok.line().startswith("[通过] x")
    assert ok.to_dict()["passed"] is True
    not_applicable = verify.PropertyOutcome(name="x", applicable=False, passed=False)
    assert "不适用" in not_applicable.line()
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome(name="x", applicable=False, passed=True)
    with pytest.raises(errors.NumericError, match="不一致"):
        verify.PropertyOutcome(
            name="x", applicable=True, passed=True,
            cross_check=verify.CrossCheck(name="c", left="l", right="r", reading=2.0, expected=1.0),
        )


def test_property_report_views_and_require_ok() -> None:
    """PropertyReport：适用集合 / ok / require_ok / lines 把不适用排在前面。"""
    report = verify.check_all()
    assert report.ok is True
    assert len(report.applicable) == 7
    assert report.require_ok() is None
    assert len(report.lines()) == 7
    assert report.to_dict()["counts"]["applicable"] == 7
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="x", applicable=True, passed=False),)
    )
    assert bad.ok is False
    with pytest.raises(errors.ParseError, match="未全部通过"):
        bad.require_ok()


def test_check_functions_default() -> None:
    """七条性质函数在默认输入上全部通过。"""
    assert verify.check_graph_covers_all_modules().passed is True
    assert verify.check_graph_is_reproducible().passed is True
    assert verify.check_topological_order_is_valid().passed is True
    assert verify.check_cycles_are_sound().passed is True
    assert verify.check_import_targets_resolve().passed is True
    assert verify.check_closures_include_self().passed is True
    assert verify.check_dependency_depth_is_positive().passed is True


def test_check_coverage_fails_when_missing() -> None:
    """覆盖检查：期望名单里多一项时失败并点名。"""
    graph = graph_module.build_graph(_chain_modules())
    outcome = verify.check_graph_covers_all_modules(graph, expected=("pkg", "pkg.ghost"))
    assert outcome.passed is False
    assert "pkg.ghost" in outcome.evidence[1]


def test_check_reproducible_fails_on_different_graphs() -> None:
    """可复算：两张不同的图差异大于 0。"""
    first = graph_module.build_graph(_chain_modules())
    second = graph_module.build_graph(_cycle_modules())
    assert verify.check_graph_is_reproducible(first, second).passed is False


def test_check_topo_and_cycles_fail_directions() -> None:
    """线性序与"环是否真"两条相等判据的失败方向。"""
    graph = graph_module.build_graph(_chain_modules())

    class _Reversed(graph_module.ModuleGraph):
        def topological_order(self) -> tuple[str, ...]:
            return tuple(reversed(super().topological_order()))

    reversed_graph = _Reversed(nodes=graph.nodes, packages=graph.packages, edges=graph.edges)
    topo = verify.check_topological_order_is_valid(reversed_graph)
    assert topo.passed is False
    assert "违反" in topo.evidence[0]

    class _FakeCycle(graph_module.ModuleGraph):
        def sccs(self) -> tuple[tuple[str, ...], ...]:
            return (("pkg.a", "pkg.z"), ("pkg.a", "pkg.b"))

        def cycles(self) -> tuple[tuple[str, ...], ...]:
            return (("pkg.a", "pkg.z"), ("pkg.a", "pkg.b"))

    fake = _FakeCycle(nodes=graph.nodes, packages=graph.packages, edges=graph.edges)
    sound = verify.check_cycles_are_sound(fake)
    assert sound.passed is False
    assert "不成立 2 个" in sound.evidence[0]


def test_check_targets_and_closures_fail_directions() -> None:
    """未解析目标（上界）、闭包含自己与依赖深度（下界）的失败方向。"""
    broken = (_module("pkg.a", _ref(".gone", None, kind=types.EDGE_KIND_RELATIVE)),)
    outcome = verify.check_import_targets_resolve(broken)
    assert outcome.passed is False
    assert outcome.criterion == types.CRITERION_UPPER_BOUND

    flat = graph_module.build_graph(_isolated_modules())
    depth = verify.check_dependency_depth_is_positive(flat)
    assert depth.passed is False
    assert depth.criterion == types.CRITERION_LOWER_BOUND

    class _NoSelf(graph_module.ModuleGraph):
        def closure(self, node: str, direction: str = types.DIRECTION_DOWN) -> tuple[str, ...]:
            return ()

    no_self = _NoSelf(nodes=flat.nodes, packages=flat.packages, edges=flat.edges)
    missing = verify.check_closures_include_self(no_self)
    assert missing.passed is False
    assert missing.criterion == types.CRITERION_LOWER_BOUND


def test_check_all_default_and_injected() -> None:
    """check_all：默认全通过；注入小图时给出同一份 7 行报告。"""
    report = verify.check_all()
    assert report.ok is True
    assert [outcome.name for outcome in report.outcomes] == list(types.IMPORT_GRAPH_PROPERTIES)
    modules = _chain_modules()
    injected = verify.check_all(modules=modules, expected=("pkg", "pkg.a", "pkg.b", "pkg.c"))
    assert [outcome.name for outcome in injected.outcomes] == list(types.IMPORT_GRAPH_PROPERTIES)
    assert injected.ok is True


def test_require_ok() -> None:
    """require_ok：默认放行，注入一份失败的报告时抛 ParseError。"""
    assert verify.require_ok() is not None
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="x", applicable=True, passed=False),)
    )
    with pytest.raises(errors.ParseError, match="未全部通过"):
        verify.require_ok(bad)


# --------------------------------------------------------------------------- 表


def test_module_rows() -> None:
    """模块表：行数与节点数一致，出度 / 入度都对。"""
    graph = graph_module.build_graph(_chain_modules())
    rows = study.module_rows(graph)
    assert len(rows) == graph.node_count
    degrees = {row.module: (row.out_degree, row.in_degree) for row in rows}
    assert degrees["pkg.a"] == (1, 0)
    assert degrees["pkg.b"] == (1, 1)
    assert "出" in rows[0].line()


def test_cycle_rows() -> None:
    """环表：全部分量与只看环两种取法。"""
    graph = graph_module.build_graph(_cycle_modules())
    every = study.cycle_rows(graph)
    assert len(every) == len(graph.sccs())
    assert every[0].is_cycle is True
    assert len(study.cycle_rows(graph, cycles_only=True)) == 1
    assert "环 " in every[0].line()


def test_cycle_rows_marks_singletons() -> None:
    """环表对单点分量标注"单点"。"""
    graph = graph_module.build_graph(_chain_modules())
    rows = study.cycle_rows(graph, cycles_only=False)
    assert all(row.is_cycle is False for row in rows)
    assert "单点" in rows[0].line()


def test_edge_rows() -> None:
    """边表：可截断。"""
    graph = graph_module.build_graph(_chain_modules())
    assert len(study.edge_rows(graph)) == graph.edge_count
    assert len(study.edge_rows(graph, limit=1)) == 1
    assert "→" in study.edge_rows(graph, limit=1)[0].line()


def test_property_rows() -> None:
    """性质表：7 行、判据类别齐全、全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert {row.criterion for row in rows} == set(types.CRITERIA)
    assert "读数" in rows[0].line()


def test_note_and_boundary_lines() -> None:
    """笔记 / 边界两类文本行。"""
    assert len(study.note_lines()) == 10
    assert len(study.note_lines(limit=4)) == 4
    assert len(study.boundary_lines()) == len(types.IMPORT_GRAPH_BOUNDARIES)


def test_study_lines_runs_all_four_tables() -> None:
    """四张表一次跑完：四个小标题。"""
    lines = study.study_lines()
    headings = [line for line in lines if line.startswith("== ")]
    assert len(headings) == 4
    assert any("模块表" in line for line in headings)
    assert any("性质表" in line for line in headings)


def test_study_lines_accepts_injected_inputs() -> None:
    """四张表接收注入的图。"""
    graph = graph_module.build_graph(_cycle_modules())
    lines = study.study_lines(graph=graph, module_limit=2, edge_limit=1)
    assert len([line for line in lines if line.startswith("== ")]) == 4


def test_to_dict_lines() -> None:
    """把带 to_dict 的行折成 JSON 化字段（四张表都能导出）。"""
    assert study.to_dict_lines(study.module_rows(limit=1))[0]["module"]
    assert study.to_dict_lines(study.cycle_rows())[0]["size"]
    assert study.to_dict_lines(study.edge_rows(limit=1))[0]["kind"]
    assert study.to_dict_lines(study.property_rows())[0]["criterion"]


# --------------------------------------------------------------------------- 包


def test_package_all_is_sorted_and_unique() -> None:
    """包的公开名单：字母序、无重复、不含子模块名。"""
    import smart_research_agent.import_graph as package

    assert package.__all__ == sorted(package.__all__)
    assert len(package.__all__) == len(set(package.__all__))
    assert set(package.__all__) & {"errors", "types", "parse", "graph", "verify", "study"} == set()


def test_package_exposes_key_names() -> None:
    """包级命名空间真的导入了关键名字（不是只在 __all__ 里）。"""
    import smart_research_agent.import_graph as package

    for name in (
        "build_graph",
        "scan_all",
        "ModuleGraph",
        "check_all",
        "study_lines",
        "GraphError",
    ):
        assert name in package.__all__
        assert hasattr(package, name)


def test_dataclasses_replace_keeps_graph_valid() -> None:
    """``dataclasses.replace`` 能造出另一张合法图（给"可复算"用）。"""
    graph = graph_module.build_graph(_chain_modules())
    replaced = dataclasses.replace(graph, packages=())
    assert replaced.package_count == 0
    assert replaced.comparable() != graph.comparable()
