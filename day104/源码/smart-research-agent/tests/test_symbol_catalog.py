"""``symbol_catalog``：把 100 天的对外承诺编成一份可复核的清单（day104）.

本文件覆盖新包的六个模块：口径表、失败族、解析、清单、性质与表。
样本有两类：**真实源码**（``smart_research_agent/**/*.py``）与
**注入的小模块**（构造反例时用）。因此这一课考的就是
"同一批文件能不能两次编出同一份清单、每一个承诺能不能被归属、
以及报出来的幽灵与重名到底是不是真的"。

> 注意：本文件**不写死**模块数 / 承诺数 / 重名数 / 幽灵数 / 摘要——它们会随着仓库新增子包而变化。
> 断言一律用"结构不变式"或"该仓库确实拥有的一处读数"；另有**一条子进程测试**专门钉住"跨进程可复算"。
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest

from smart_research_agent.symbol_catalog import catalog as catalog_module
from smart_research_agent.symbol_catalog import errors, parse as parse_module, study, types, verify

# --------------------------------------------------------------------------- 注入样本


def _module(
    module: str,
    *,
    source: str = types.SOURCE_IMPLICIT,
    declared: tuple[str, ...] = (),
    defined: tuple[str, ...] = (),
    assigned: tuple[str, ...] = (),
    imported: tuple[str, ...] = (),
    path: str = "/x/m.py",
    is_package: bool = False,
) -> parse_module.ModuleBindings:
    """一份被解析的小模块（路径可以指向真实文件，供独立复核用）."""
    return parse_module.ModuleBindings(
        module=module,
        is_package=is_package,
        path=path,
        source=source,
        declared=tuple(sorted(set(declared))),
        defined=tuple(sorted(set(defined))),
        assigned=tuple(sorted(set(assigned))),
        imported=tuple(sorted(set(imported))),
    )


def _declared_module(module: str, declared: tuple[str, ...], **bound: tuple[str, ...]) -> parse_module.ModuleBindings:
    """一份带字面量 ``__all__`` 的模块."""
    return _module(module, source=types.SOURCE_DECLARED, declared=declared, **bound)


def _reuse_pair() -> tuple[parse_module.ModuleBindings, ...]:
    """两个模块都导出一个同名名字（用来把"重名存在"这条下界撑住）."""
    return (
        _declared_module("pkg.a", ("Shared", "OnlyA"), defined=("Shared", "OnlyA")),
        _declared_module("pkg.b", ("Shared",), defined=("Shared",)),
    )


# --------------------------------------------------------------------------- 口径表


def test_promise_sources_and_bindings_are_closed() -> None:
    """三种承诺来源与四类绑定：名单与说明逐键对齐。"""
    assert len(types.PROMISE_SOURCES) == 3
    assert set(types.SOURCE_DESCRIPTIONS) == set(types.PROMISE_SOURCES)
    assert len(types.BINDING_KINDS) == 4
    assert set(types.BINDING_DESCRIPTIONS) == set(types.BINDING_KINDS)
    assert types.require_promise_source(types.SOURCE_DECLARED) == types.SOURCE_DECLARED
    assert types.require_binding_kind(types.BINDING_IMPORTED) == types.BINDING_IMPORTED
    with pytest.raises(errors.ParameterError, match="未知的承诺来源"):
        types.require_promise_source("nope")
    with pytest.raises(errors.ParameterError, match="未知的绑定类别"):
        types.require_binding_kind("nope")


def test_seven_properties_and_three_criteria() -> None:
    """7 条性质：名单与规格表逐键对齐，且三类判据都有性质。"""
    assert len(types.SYMBOL_CATALOG_PROPERTIES) == 7
    assert set(types.PROPERTY_SPECS) == set(types.SYMBOL_CATALOG_PROPERTIES)
    criteria = {spec.criterion for spec in types.property_specs()}
    assert criteria == set(types.CRITERIA)
    assert criteria == {
        types.CRITERION_EQUALITY,
        types.CRITERION_UPPER_BOUND,
        types.CRITERION_LOWER_BOUND,
    }


def test_notes_are_ten_and_boundaries_five() -> None:
    """十条笔记（有序）与五条边界。"""
    assert len(types.SYMBOL_CATALOG_NOTES) == 10
    assert types.SYMBOL_CATALOG_NOTES_ORDER == tuple(types.SYMBOL_CATALOG_NOTES)
    assert all(types.SYMBOL_CATALOG_NOTES.values())
    assert len(types.SYMBOL_CATALOG_BOUNDARIES) == 5
    assert all(types.SYMBOL_CATALOG_BOUNDARIES)


def test_docs_of_source_and_binding() -> None:
    """两个说明函数：合法值返回一句话，非法值当场拒绝。"""
    assert types.docs_of_source(types.SOURCE_COMPUTED)
    assert types.docs_of_binding(types.BINDING_DEFINED)
    with pytest.raises(errors.ParameterError, match="未知的承诺来源"):
        types.docs_of_source("nope")
    with pytest.raises(errors.ParameterError, match="未知的绑定类别"):
        types.docs_of_binding("nope")


def test_require_helpers() -> None:
    """两个 require 帮手：合法值原样返回，非法值当场拒绝。"""
    assert types.require_positive_int("limit", 3) == 3
    assert types.require_property(types.PROPERTY_CATALOG_IS_REPRODUCIBLE)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("limit", 0)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("limit", True)
    with pytest.raises(errors.ParameterError, match="未知的性质"):
        types.require_property("nope")


def _property_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 PropertySpec 关键字参数."""
    base: dict[str, object] = {
        "id": types.PROPERTY_CATALOG_IS_REPRODUCIBLE,
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
        assert issubclass(cls, errors.SymbolError)
        assert issubclass(cls, ValueError)


def test_returned_and_absent_families_are_declared() -> None:
    """回来的族与缺席的族都是常量，且各自带理由。"""
    assert errors.RETURNED_FAMILY == "CoverageError"
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


def test_package_dir_and_empty_scan(tmp_path: pathlib.Path) -> None:
    """包目录：不存在时抛；存在但没有 .py 也抛。"""
    assert parse_module.package_dir().is_dir()
    with pytest.raises(errors.ParseError, match="包目录不存在"):
        parse_module.package_dir(tmp_path)
    (tmp_path / types.PACKAGE_NAME).mkdir()
    with pytest.raises(errors.ParseError, match=r"一份 \.py 都没有"):
        parse_module.module_files(tmp_path)


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


def test_parse_source_and_read_source(tmp_path: pathlib.Path) -> None:
    """语法错当场抛 ParseError；读源码容忍 BOM。"""
    with pytest.raises(errors.ParseError, match="语法错误"):
        parse_module.parse_source("def :(", origin="<mem>")
    target = tmp_path / "m.py"
    target.write_text("import os\n", encoding="utf-8-sig")
    assert "import os" in parse_module.read_source(target)


def test_iter_module_scope_skips_def_and_class() -> None:
    """模块级作用域：进 if / try / with，不进 def / class。"""
    import ast

    tree = ast.parse(
        "A = 1\n"
        "if True:\n"
        "    B = 2\n"
        "try:\n"
        "    C = 3\n"
        "except Exception:\n"
        "    D = 4\n"
        "finally:\n"
        "    E = 5\n"
        "with open('x') as F:\n"
        "    pass\n"
        "def g():\n"
        "    INSIDE = 6\n"
        "class H:\n"
        "    INSIDE2 = 7\n"
    )
    names: set[str] = set()
    for node in parse_module.iter_module_scope(tree.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(parse_module.target_names(target))
        elif isinstance(node, ast.With):
            for item in node.items:
                names.update(parse_module.target_names(item.optional_vars))
    assert {"A", "B", "C", "D", "E", "F"} <= names
    assert "INSIDE" not in names and "INSIDE2" not in names


def test_target_names_shapes() -> None:
    """赋值目标：裸名 / 元组 / 列表 / 星号 / 其它表达式。"""
    import ast

    assert parse_module.target_names(ast.parse("a = 1").body[0].targets[0]) == ("a",)
    assert parse_module.target_names(ast.parse("a, b = 1, 2").body[0].targets[0]) == ("a", "b")
    assert parse_module.target_names(ast.parse("[a, *b] = x").body[0].targets[0]) == ("a", "b")
    assert parse_module.target_names(ast.parse("d[0] = 1").body[0].targets[0]) == ()


def test_read_all_literal_forms() -> None:
    """``__all__`` 右端：字面量列表/元组/集合可读，非容器返回 None，混入非字符串抛错。"""
    import ast

    def value(src: str) -> ast.expr:
        return ast.parse(src).body[0].value  # type: ignore[attr-defined]

    assert parse_module.read_all_literal(value("['A', 'B']")) == ("A", "B")
    assert parse_module.read_all_literal(value("('A',)")) == ("A",)
    assert parse_module.read_all_literal(value("{'A'}")) == ("A",)
    assert parse_module.read_all_literal(value("sorted({'A'})")) is None
    assert parse_module.read_all_literal(None) is None
    with pytest.raises(errors.DeclareError, match="混进了非字符串元素"):
        parse_module.read_all_literal(value("[A, 'B']"))


def test_collect_bindings_full_grammar() -> None:
    """一次把模块级的各种绑定都读出来（含 try 里的 import 与 ``for`` 目标）。"""
    import ast

    tree = ast.parse(
        'class Foo:\n    pass\n'
        'def bar():\n    import json\n'
        'X = 1\n'
        'X += 1\n'
        'Y: int = 2\n'
        'if True:\n    Z = 3\n'
        'try:\n    from pkg import thing as alias\n    import mod.sub\n    from y import *\n'
        'except Exception:\n    W = 4\n'
        'for item in []:\n    pass\n'
        'with open("f") as handle:\n    pass\n'
        'with open("g"):\n    pass\n'
        'global GLOB\n'
        '__all__ = ["Foo", "X"]\n'
    )
    defined, assigned, imported, source, declared = parse_module.collect_bindings(tree)
    assert defined == ("Foo", "bar")
    assert set(assigned) == {"X", "Y", "Z", "W", "item", "handle", "GLOB"}
    assert imported == ("alias", "mod")
    assert source == types.SOURCE_DECLARED
    assert declared == ("Foo", "X")


def test_collect_bindings_annassign_computed_all() -> None:
    """``__all__`` 写成带注解的非字面量时，来源记成 computed。"""
    import ast

    _, _, _, source, declared = parse_module.collect_bindings(
        ast.parse('__all__: list[str] = sorted({"A"})\n')
    )
    assert source == types.SOURCE_COMPUTED and declared == ()


def test_collect_bindings_sources() -> None:
    """三种来源：没写 / 写成非字面量 / 写了两次或 ``+=``。"""
    import ast

    _, _, _, source, declared = parse_module.collect_bindings(ast.parse("X = 1\n"))
    assert source == types.SOURCE_IMPLICIT and declared == ()
    _, _, _, source, declared = parse_module.collect_bindings(ast.parse("__all__ = sorted({'A'})\n"))
    assert source == types.SOURCE_COMPUTED and declared == ()
    _, _, _, source, declared = parse_module.collect_bindings(
        ast.parse('__all__ = ["A"]\n__all__ = ["B"]\n')
    )
    assert source == types.SOURCE_COMPUTED and declared == ()
    _, _, _, source, declared = parse_module.collect_bindings(
        ast.parse('__all__ = ["A"]\n__all__ += ["B"]\n')
    )
    assert source == types.SOURCE_COMPUTED and declared == ()
    _, _, _, source, declared = parse_module.collect_bindings(
        ast.parse('__all__: list[str] = ["A"]\n')
    )
    assert source == types.SOURCE_DECLARED and declared == ("A",)


def test_module_bindings_guards() -> None:
    """模块行的四条护栏：空名 / 未知来源 / declared 却空 / 未排序。"""
    with pytest.raises(errors.ParseError, match="模块名不能为空"):
        _module("")
    with pytest.raises(errors.DeclareError, match="未知的承诺来源"):
        _module("pkg.a", source="nope")
    with pytest.raises(errors.DeclareError, match="却没有一份声明名单"):
        _module("pkg.a", source=types.SOURCE_DECLARED)
    with pytest.raises(errors.ParseError, match="排序且不重复"):
        parse_module.ModuleBindings(
            module="pkg.a", is_package=False, path="/x.py", source=types.SOURCE_DECLARED,
            declared=("B", "A"),
        )


def test_module_bindings_views() -> None:
    """模块行：四类落点、绑定表、公开定义、导出、幽灵、渲染。"""
    module = _declared_module(
        "pkg.a",
        ("Alpha", "Beta", "Gamma", "Delta", "Gone"),
        defined=("Alpha",),
        assigned=("Beta",),
        imported=("Gamma",),
    )
    assert module.binding_of("Alpha") == types.BINDING_DEFINED
    assert module.binding_of("Beta") == types.BINDING_ASSIGNED
    assert module.binding_of("Gamma") == types.BINDING_IMPORTED
    assert module.binding_of("Delta") == types.BINDING_UNRESOLVED
    assert module.bindings()[0] == ("Alpha", types.BINDING_DEFINED)
    assert module.phantoms() == ("Delta", "Gone")
    assert module.exports() == ("Alpha", "Beta", "Delta", "Gamma", "Gone")
    assert module.to_dict()["phantoms"] == 2
    assert "承诺" in module.line()


def test_module_bindings_implicit_exports_from_public_definitions() -> None:
    """没有 ``__all__`` 的模块：导出退而取顶层公开定义（下划线的不算）。"""
    module = _module("pkg.a", defined=("Public", "_private", "Other"))
    assert module.exports() == ("Other", "Public")
    assert module.bindings() == ()
    assert module.phantoms() == ()
    computed = _module("pkg.a", source=types.SOURCE_COMPUTED, defined=("Public",))
    assert computed.exports() == ("Public",)
    assert computed.phantoms() == ()


def test_scan_module_reads_a_real_file() -> None:
    """扫一份真实文件：模块名对得上、承诺非空。"""
    path = parse_module.PROJECT_ROOT / "smart_research_agent" / "symbol_catalog" / "catalog.py"
    scanned = parse_module.scan_module(path, parse_module.PROJECT_ROOT)
    assert scanned.module == "smart_research_agent.symbol_catalog.catalog"
    assert scanned.is_package is False
    assert scanned.source == types.SOURCE_DECLARED
    assert "SymbolCatalog" in scanned.declared


def test_scan_all_is_reproducible_in_process() -> None:
    """两次扫描逐位相同（同一进程内）。"""
    first = parse_module.scan_all()
    second = parse_module.scan_all()
    assert [m.to_dict() for m in first] == [m.to_dict() for m in second]
    assert parse_module.expected_modules() == tuple(sorted(m.module for m in first))


def test_catalog_digest_is_stable_across_processes() -> None:
    """**跨进程**可复算：两个不同的哈希种子必须给出同一个摘要.

    这正是"并列必须按名字兜顺序"那条纪律的守卫：清单里若有一个集合 / 字典 /
    绝对路径进了 ``comparable``，两次运行就会给出两个数字。
    """
    script = (
        "from smart_research_agent.symbol_catalog.catalog import build_catalog;"
        "from smart_research_agent.symbol_catalog.parse import scan_all;"
        "print(build_catalog(scan_all()).digest())"
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


def test_outside_all_names_is_an_independent_check(tmp_path: pathlib.Path) -> None:
    """独立复核：``__all__`` 里的名字不算"在正文里出现过"，正文里的名字算。"""
    target = tmp_path / "m.py"
    target.write_text('Body = 1\n__all__ = ["Body", "Ghost"]\n', encoding="utf-8")
    outside = parse_module.outside_all_names(target)
    assert "Body" in outside
    assert "Ghost" not in outside


def test_phantom_exports_and_require() -> None:
    """幽灵清单：注入一个"声明了却没有落点"的名字，默认清单里另有真实读数。"""
    injected = (_declared_module("pkg.a", ("Ghost",)),)
    assert parse_module.phantom_exports(injected) == (("pkg.a", "Ghost"),)
    with pytest.raises(errors.BindingError, match="承诺找不到落点"):
        parse_module.require_no_phantoms(injected)
    good = _reuse_pair()
    assert parse_module.require_no_phantoms(good) is good


def test_phantom_exports_default_matches_catalog() -> None:
    """默认（真实仓库）的幽灵清单：与清单模块给出的一致。"""
    assert parse_module.phantom_exports() == catalog_module.build_catalog().phantom_pairs()


def test_binding_kind_of_and_scan_lines() -> None:
    """落点类别帮手 + 解析逐行打印（末行是合计）。"""
    module = _declared_module("pkg.a", ("Alpha",))
    assert parse_module.binding_kind_of(module, "Alpha") == types.BINDING_UNRESOLVED
    lines = parse_module.scan_lines(_reuse_pair())
    assert lines[0].startswith("解析表")
    assert "合计" in lines[-1] and "承诺" in lines[-1]
    assert len(parse_module.scan_lines(_reuse_pair(), limit=1)) == 3


# --------------------------------------------------------------------------- 清单


def test_catalog_views() -> None:
    """清单：模块名单、分组、来源分布、导出对、总数、渲染。"""
    catalog = catalog_module.build_catalog(_reuse_pair())
    assert catalog.modules == ("pkg.a", "pkg.b")
    assert catalog.module_count == 2
    assert catalog.module_set == frozenset({"pkg.a", "pkg.b"})
    assert catalog.entry_of("pkg.a").module == "pkg.a"
    assert catalog.package_of("pkg.a") == "a"
    assert [name for name, _members in catalog.package_groups()] == ["a", "b"]
    assert dict(catalog.source_counts()) == {
        types.SOURCE_DECLARED: 2,
        types.SOURCE_COMPUTED: 0,
        types.SOURCE_IMPLICIT: 0,
    }
    assert catalog.declared_count() == 2
    assert catalog.promise_count() == 3
    assert ("pkg.a", "Shared") in catalog.export_pairs()
    assert catalog.to_dict()["modules"] == 2
    assert "清单：" in catalog.line()


def test_catalog_shared_and_phantoms() -> None:
    """重名表与幽灵表：名字排序、出现模块数排序。"""
    modules = _reuse_pair() + (_declared_module("pkg.c", ("Ghost",)),)
    catalog = catalog_module.build_catalog(modules)
    shared = catalog.shared_names()
    assert shared == (("Shared", ("pkg.a", "pkg.b")),)
    assert catalog.phantom_pairs() == (("pkg.c", "Ghost"),)
    assert catalog.duplicate_declarations() == ()


def test_catalog_guards_and_lookups() -> None:
    """清单的两条护栏与两个未知模块入口。"""
    with pytest.raises(errors.CatalogBuildError, match="不能没有模块行"):
        catalog_module.SymbolCatalog(entries=())
    first = _module("pkg.b")
    second = _module("pkg.a")
    with pytest.raises(errors.CatalogBuildError, match="排序且不重复"):
        catalog_module.SymbolCatalog(entries=(first, second))
    catalog = catalog_module.build_catalog(_reuse_pair())
    with pytest.raises(errors.ParameterError, match="清单里没有模块"):
        catalog.entry_of("pkg.zzz")
    with pytest.raises(errors.ParameterError, match="清单里没有模块"):
        catalog.package_of("pkg.zzz")


def test_catalog_recomputable_guards() -> None:
    """可复算与"拒绝重名"两条路。"""
    modules = _reuse_pair()
    first = catalog_module.build_catalog(modules)
    second = catalog_module.build_catalog(modules)
    assert first.diff_count(second) == 0
    assert first.digest() == second.digest()
    assert catalog_module.catalog_digest(first) == first.digest()
    assert catalog_module.require_reproducible(first, second) is first
    other = catalog_module.build_catalog((_declared_module("pkg.z", ("Zed",), defined=("Zed",)),))
    assert first.diff_count(other) > 0
    with pytest.raises(errors.CatalogBuildError, match="两份清单差"):
        catalog_module.require_reproducible(first, other)
    assert catalog_module.require_unique_declarations(first) is first


def test_catalog_require_unique_declarations_failure() -> None:
    """重复声明的模块会在"拒绝交付"的那条路上被点名。"""

    class _Duplicated(catalog_module.SymbolCatalog):
        def duplicate_declarations(self) -> tuple[str, ...]:
            return ("pkg.a",)

    catalog = _Duplicated(entries=tuple(sorted(_reuse_pair(), key=lambda e: e.module)))
    with pytest.raises(errors.CatalogBuildError, match="重复声明了名字"):
        catalog_module.require_unique_declarations(catalog)


def test_expected_packages_and_catalog_lines() -> None:
    """包的名单与逐行打印（可截断，非法 limit 被拒）。"""
    catalog = catalog_module.build_catalog(_reuse_pair())
    assert catalog_module.expected_packages(catalog) == ("a", "b")
    lines = catalog_module.catalog_lines(catalog, limit=1)
    assert lines[0].startswith("符号清单")
    assert len(lines) == 3
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        catalog_module.catalog_lines(catalog, limit=0)


def test_catalog_docs_and_root() -> None:
    """两个小帮手：来源说明与根包名。"""
    assert catalog_module.docs_of_source(types.SOURCE_IMPLICIT)
    assert catalog_module.root_package() == types.PACKAGE_NAME
    with pytest.raises(errors.ParameterError, match="未知的承诺来源"):
        catalog_module.docs_of_source("nope")


def test_dataclasses_replace_keeps_catalog_valid() -> None:
    """``dataclasses.replace`` 能造出另一份合法清单（给"可复算"用）。"""
    catalog = catalog_module.build_catalog(_reuse_pair())
    replaced = dataclasses.replace(catalog, entries=(catalog.entries[0],))
    assert replaced.module_count == 1
    assert replaced.comparable() != catalog.comparable()


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
    assert verify.check_catalog_covers_all_modules().passed is True
    assert verify.check_catalog_is_reproducible().passed is True
    assert verify.check_phantom_exports_are_sound().passed is True
    assert verify.check_qualified_names_are_unique().passed is True
    assert verify.check_duplicate_declarations_within_module().passed is True
    assert verify.check_every_declaring_module_promises_something().passed is True
    assert verify.check_export_names_are_reused().passed is True


def test_check_coverage_fails_when_missing() -> None:
    """覆盖检查：期望名单里多一项时失败并点名。"""
    catalog = catalog_module.build_catalog(_reuse_pair())
    outcome = verify.check_catalog_covers_all_modules(catalog, expected=("pkg.a", "pkg.b", "pkg.ghost"))
    assert outcome.passed is False
    assert "pkg.ghost" in outcome.evidence[1]


def test_check_reproducible_fails_on_different_catalogs() -> None:
    """可复算：两份不同的清单差异大于 0。"""
    first = catalog_module.build_catalog(_reuse_pair())
    second = catalog_module.build_catalog((_declared_module("pkg.z", ("Zed",), defined=("Zed",)),))
    assert verify.check_catalog_is_reproducible(first, second).passed is False


def test_check_phantom_soundness_failure_direction() -> None:
    """幽灵是否真：注入一个名字其实出现在正文里的"幽灵"，判据变红。

    这里让那份小模块的 ``path`` 指向**真实文件**（``parse.py``），
    并声明一个确实写在那个文件正文里的名字——独立复核立刻发现它是假幽灵。
    """
    real = parse_module.PROJECT_ROOT / "smart_research_agent" / "symbol_catalog" / "parse.py"
    fake = _declared_module(
        "pkg.a", ("parse_source",), path=real.as_posix()
    )
    outcome = verify.check_phantom_exports_are_sound(catalog_module.build_catalog((fake,)))
    assert outcome.passed is False
    assert "不成立" in outcome.evidence[0]


def test_check_shared_names_soundness_failure_direction() -> None:
    """重名是否真：把重名表换成一份对不上的，判据变红。"""

    class _WrongShared(catalog_module.SymbolCatalog):
        def shared_names(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
            return (("Ghost", ("pkg.a", "pkg.b")),)

    catalog = _WrongShared(entries=tuple(sorted(_reuse_pair(), key=lambda e: e.module)))
    outcome = verify.check_qualified_names_are_unique(catalog)
    assert outcome.passed is False
    assert "不成立 1" in outcome.evidence[0]


def test_check_duplicate_failure_direction() -> None:
    """重复声明（上界）的失败方向。"""

    class _Duplicated(catalog_module.SymbolCatalog):
        def duplicate_declarations(self) -> tuple[str, ...]:
            return ("pkg.a",)

    catalog = _Duplicated(entries=tuple(sorted(_reuse_pair(), key=lambda e: e.module)))
    outcome = verify.check_duplicate_declarations_within_module(catalog)
    assert outcome.passed is False
    assert outcome.criterion == types.CRITERION_UPPER_BOUND


def test_check_promise_and_reuse_failure_directions() -> None:
    """承诺下界与重名下界的失败方向。"""
    computed_only = catalog_module.build_catalog(
        (_module("pkg.a", source=types.SOURCE_COMPUTED, defined=("A",)),)
    )
    promise = verify.check_every_declaring_module_promises_something(computed_only)
    assert promise.passed is False
    assert promise.criterion == types.CRITERION_LOWER_BOUND

    unique = catalog_module.build_catalog(
        (
            _declared_module("pkg.a", ("Alpha",), defined=("Alpha",)),
            _declared_module("pkg.b", ("Beta",), defined=("Beta",)),
        )
    )
    reuse = verify.check_export_names_are_reused(unique)
    assert reuse.passed is False
    assert reuse.criterion == types.CRITERION_LOWER_BOUND


def test_check_all_default_and_injected() -> None:
    """check_all：默认全通过；注入小清单时给出同一份 7 行报告。"""
    report = verify.check_all()
    assert report.ok is True
    assert [outcome.name for outcome in report.outcomes] == list(types.SYMBOL_CATALOG_PROPERTIES)
    injected = verify.check_all(
        modules=_reuse_pair(), expected=("pkg.a", "pkg.b")
    )
    assert [outcome.name for outcome in injected.outcomes] == list(types.SYMBOL_CATALOG_PROPERTIES)
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
    """模块表：行数与模块数一致，来源与计数都对。"""
    catalog = catalog_module.build_catalog(_reuse_pair())
    rows = study.module_rows(catalog)
    assert len(rows) == catalog.module_count
    table = {row.module: row for row in rows}
    assert table["pkg.a"].source == types.SOURCE_DECLARED
    assert table["pkg.a"].promises == 2
    assert "承诺" in rows[0].line()
    assert len(study.module_rows(catalog, limit=1)) == 1


def test_phantom_and_shared_rows() -> None:
    """幽灵表 / 重名表：行数与读数一致，可截断。"""
    modules = _reuse_pair() + (_declared_module("pkg.c", ("Ghost",)),)
    catalog = catalog_module.build_catalog(modules)
    phantom = study.phantom_rows(catalog)
    assert [row.name for row in phantom] == ["Ghost"]
    assert "pkg.c" in phantom[0].line()
    shared = study.shared_rows(catalog)
    assert [row.name for row in shared] == ["Shared"]
    assert shared[0].count == 2
    assert "个模块" in shared[0].line()
    assert len(study.shared_rows(catalog, limit=0 + 1)) == 1


def test_property_rows() -> None:
    """性质表：7 行、判据类别齐全、全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert {row.criterion for row in rows} == set(types.CRITERIA)
    assert "读数" in rows[0].line()


def test_note_boundary_and_source_lines() -> None:
    """笔记 / 边界 / 来源分布三类文本行。"""
    assert len(study.note_lines()) == 10
    assert len(study.note_lines(limit=4)) == 4
    assert len(study.boundary_lines()) == len(types.SYMBOL_CATALOG_BOUNDARIES)
    assert len(study.source_lines()) == len(types.PROMISE_SOURCES)


def test_study_lines_runs_all_four_tables() -> None:
    """四张表一次跑完：四个小标题。"""
    lines = study.study_lines()
    headings = [line for line in lines if line.startswith("== ")]
    assert len(headings) == 4
    assert any("模块表" in line for line in headings)
    assert any("幽灵表" in line for line in headings)
    assert any("性质表" in line for line in headings)


def test_study_lines_accepts_injected_inputs() -> None:
    """四张表接收注入的清单。"""
    catalog = catalog_module.build_catalog(_reuse_pair())
    lines = study.study_lines(catalog=catalog, module_limit=1, shared_limit=1)
    assert len([line for line in lines if line.startswith("== ")]) == 4


def test_to_dict_lines() -> None:
    """把带 to_dict 的行折成 JSON 化字段（四张表都能导出）。"""
    catalog = catalog_module.build_catalog(_reuse_pair() + (_declared_module("pkg.c", ("Ghost",)),))
    assert study.to_dict_lines(study.module_rows(catalog, limit=1))[0]["module"]
    assert study.to_dict_lines(study.phantom_rows(catalog))[0]["name"] == "Ghost"
    assert study.to_dict_lines(study.shared_rows(catalog))[0]["count"] == 2
    assert study.to_dict_lines(study.property_rows())[0]["criterion"]


# --------------------------------------------------------------------------- 包


def test_package_all_is_sorted_and_unique() -> None:
    """包的公开名单：字母序、无重复、不含子模块名。"""
    import smart_research_agent.symbol_catalog as package

    assert package.__all__ == sorted(package.__all__)
    assert len(package.__all__) == len(set(package.__all__))
    assert set(package.__all__) & {"errors", "types", "parse", "catalog", "verify", "study"} == set()


def test_package_exposes_key_names() -> None:
    """包级命名空间真的导入了关键名字（不是只在 __all__ 里）。"""
    import smart_research_agent.symbol_catalog as package

    for name in (
        "build_catalog",
        "scan_all",
        "SymbolCatalog",
        "check_all",
        "study_lines",
        "SymbolError",
    ):
        assert name in package.__all__
        assert hasattr(package, name)
