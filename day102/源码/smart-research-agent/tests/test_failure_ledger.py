"""``failure_ledger``：把 100 天的失败族编成一份可复算的台账（day102）.

本文件覆盖新包的七个模块：口径表、失败族、扫描、归属、台账、性质与表。
样本有两类：**真实源码**（``smart_research_agent/*/errors.py``）与
**注入的小模块**（构造反例时用）。因此这一课考的就是
"同一批文件能不能两次编出同一份台账、每一份 errors.py 能不能被说清归属"。

> 注意：本文件**不写死**模块数 / 族数 / 摘要——它们会随着仓库新增子包而变化。
> 断言一律用"结构不变式"（例如"每个包恰好一个根""未解析为 0"）或下界。
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from smart_research_agent.failure_ledger import (
    errors,
    ledger as ledger_module,
    resolve as resolve_module,
    scan as scan_module,
    study,
    types,
    verify,
)

# --------------------------------------------------------------------------- 注入样本


def _raw(name: str, *bases: str, doc: str = "") -> scan_module.RawFamily:
    """一个原始族（默认无自述）."""
    return scan_module.RawFamily(name=name, bases=tuple(bases), doc_first_line=doc)


def _module(
    package: str,
    *families: scan_module.RawFamily,
    aliases: tuple[tuple[str, str], ...] = (),
) -> scan_module.ScannedModule:
    """一个被扫描的小模块（路径是造的，内容只用于解析）."""
    return scan_module.ScannedModule(
        package=package,
        path=f"/x/{package}/errors.py",
        families=tuple(families),
        aliases=aliases,
    )


def _small_modules() -> tuple[scan_module.ScannedModule, ...]:
    """一组两包的小模块：一个根、一个派生、一条跨包基类。"""
    return (
        _module(
            "alpha",
            _raw("AlphaError", "ValueError"),
            _raw("AlphaShapeError", "CoreShapeError", "AlphaError"),
            aliases=(("CoreShapeError", "smart_research_agent.core.errors.ShapeError"),),
        ),
        _module("beta", _raw("BetaError", "Exception")),
    )


# --------------------------------------------------------------------------- 口径表


def test_base_kinds_are_closed() -> None:
    """四类归属：名单与说明逐键对齐。"""
    assert len(types.BASE_KINDS) == 4
    assert set(types.BASE_KIND_DESCRIPTIONS) == set(types.BASE_KINDS)
    assert types.require_base_kind(types.BASE_KIND_LOCAL) == types.BASE_KIND_LOCAL
    with pytest.raises(errors.ParameterError, match="未知的基类归属"):
        types.require_base_kind("nope")


def test_builtin_whitelist_contains_the_three_real_roots() -> None:
    """内置白名单含三个真实用到的根基类，且不含明显非异常的名字。"""
    for name in ("ValueError", "Exception", "RuntimeError"):
        assert name in types.BUILTIN_EXCEPTIONS
    assert "smart_research_agent" not in types.BUILTIN_EXCEPTIONS


def test_seven_properties_and_three_criteria() -> None:
    """7 条性质：名单与规格表逐键对齐，且三类判据都有性质。"""
    assert len(types.FAILURE_LEDGER_PROPERTIES) == 7
    assert set(types.PROPERTY_SPECS) == set(types.FAILURE_LEDGER_PROPERTIES)
    criteria = {spec.criterion for spec in types.property_specs()}
    assert criteria == set(types.CRITERIA)
    assert criteria == {
        types.CRITERION_EQUALITY,
        types.CRITERION_UPPER_BOUND,
        types.CRITERION_LOWER_BOUND,
    }


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.FAILURE_LEDGER_NOTES) == 10
    assert types.FAILURE_LEDGER_NOTES_ORDER == tuple(types.FAILURE_LEDGER_NOTES)
    assert all(types.FAILURE_LEDGER_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（这一课明确不承诺的事）。"""
    assert len(types.FAILURE_LEDGER_BOUNDARIES) == 5
    assert all(types.FAILURE_LEDGER_BOUNDARIES)


def test_family_kinds_and_docs_of_kind() -> None:
    """两种身份是一份二元组；docs_of_kind 返回一句话。"""
    assert types.FAMILY_KINDS == (types.FAMILY_KIND_ROOT, types.FAMILY_KIND_SUB)
    assert types.docs_of_kind(types.BASE_KIND_IMPORTED)
    with pytest.raises(errors.ParameterError, match="未知的基类归属"):
        types.docs_of_kind("nope")


def test_require_helpers() -> None:
    """两个 require 帮手：合法值原样返回，非法值当场拒绝。"""
    assert types.require_positive_int("limit", 3) == 3
    assert types.require_property(types.PROPERTY_LEDGER_IS_REPRODUCIBLE)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("limit", 0)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("limit", True)
    with pytest.raises(errors.ParameterError, match="未知的性质"):
        types.require_property("nope")


def _property_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 PropertySpec 关键字参数."""
    base: dict[str, object] = {
        "id": types.PROPERTY_LEDGER_IS_REPRODUCIBLE,
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
        assert issubclass(cls, errors.LedgerError)
        assert issubclass(cls, ValueError)


def test_returned_and_absent_families_are_declared() -> None:
    """回来的族与缺席的族都是常量，且各自带理由。"""
    assert errors.RETURNED_FAMILY == "ShapeError"
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY_REASON


def test_each_family_is_raisable() -> None:
    """每个失败族都能被 raise / except。"""
    for cls in errors._FAMILY_CLASSES.values():
        with pytest.raises(cls):
            raise cls("boom")


# --------------------------------------------------------------------------- 扫描


def test_error_module_paths_are_sorted_and_named() -> None:
    """被扫描的文件：非空、按路径排序、都叫 errors.py。"""
    paths = scan_module.error_module_paths()
    assert paths
    assert list(paths) == sorted(paths, key=lambda item: item.as_posix())
    assert all(path.name == scan_module.SCAN_FILENAME for path in paths)


def test_expected_modules_match_paths() -> None:
    """期望包名与路径一一对应，且含本课自己与 day101。"""
    names = scan_module.expected_modules()
    assert list(names) == sorted(names)
    assert "failure_ledger" in names
    assert "course_index" in names


def test_package_dir_and_missing_errors(tmp_path: pathlib.Path) -> None:
    """包目录：不存在时抛；存在但没有 errors.py 也抛。"""
    assert scan_module.package_dir().is_dir()
    with pytest.raises(errors.ScanError, match="包目录不存在"):
        scan_module.package_dir(tmp_path)
    empty = tmp_path / types.PACKAGE_NAME / "somepkg"
    empty.mkdir(parents=True)
    with pytest.raises(errors.ScanError, match=r"一份 errors\.py 都没有"):
        scan_module.error_module_paths(tmp_path)


def test_package_label_and_root_label(tmp_path: pathlib.Path) -> None:
    """包名取自父目录；直接放在包根下的 errors.py 记为 <root>。"""
    root = tmp_path / types.PACKAGE_NAME
    nested = root / "widget"
    nested.mkdir(parents=True)
    (nested / "errors.py").write_text("class W(ValueError):\n    pass\n", encoding="utf-8")
    assert scan_module.package_label(nested / "errors.py", tmp_path) == "widget"
    top = root / "errors.py"
    top.write_text("class T(ValueError):\n    pass\n", encoding="utf-8")
    assert scan_module.package_label(top, tmp_path) == "<root>"


def test_parse_source_rejects_syntax_error() -> None:
    """语法错当场抛 ScanError（而不是被 import 掩盖）。"""
    with pytest.raises(errors.ScanError, match="语法错误"):
        scan_module.parse_source("class :(", origin="<mem>")


def test_read_source_accepts_bom(tmp_path: pathlib.Path) -> None:
    """读源码容忍 BOM（仓库里确有一份带 BOM 的 errors.py）。"""
    target = tmp_path / "errors.py"
    target.write_text('class A(ValueError):\n    pass\n', encoding="utf-8-sig")
    assert "class A" in scan_module.read_source(target)


def test_class_defs_only_takes_top_level() -> None:
    """嵌套类不算失败族（只取顶层 ClassDef）。"""
    tree = ast.parse(
        """
class Outer(ValueError):
    class Inner(ValueError):
        pass

def helper():
    class Local(ValueError):
        pass
"""
    )
    families = scan_module.class_defs(tree)
    assert [family.name for family in families] == ["Outer"]


def test_class_defs_takes_doc_first_line() -> None:
    """族的自述只取第一行非空文本。"""
    tree = ast.parse('class E(ValueError):\n    """第一行。\n\n    第二行。\n    """\n')
    family = scan_module.class_defs(tree)[0]
    assert family.doc_first_line == "第一行。"


def test_import_aliases_handles_both_forms() -> None:
    """import 的两种写法都被收进别名表（ImportFrom 与 Import）。"""
    tree = ast.parse(
        "from a.b.errors import ShapeError as CoreShapeError\n"
        "import pkg.mod as D\n"
        "import other\n"
    )
    aliases = dict(scan_module.import_aliases(tree))
    assert aliases["CoreShapeError"] == "a.b.errors.ShapeError"
    assert aliases["D"] == "pkg.mod"
    assert aliases["other"] == "other"


def test_raw_family_guards_and_rendering() -> None:
    """原始族：空名字被拒；两个渲染方法可用。"""
    family = _raw("ShapeError", "CoreShapeError", "StackError")
    assert family.to_dict()["bases"] == ["CoreShapeError", "StackError"]
    assert "ShapeError(" in family.line()
    with pytest.raises(errors.ScanError, match="族名不能为空"):
        _raw("")


def test_scanned_module_guards_and_views() -> None:
    """被扫描模块的三条护栏与只读视图。"""
    module = _small_modules()[0]
    assert module.family_names == ("AlphaError", "AlphaShapeError")
    assert module.alias_of("CoreShapeError") == "smart_research_agent.core.errors.ShapeError"
    assert module.alias_of("nope") is None
    assert module.to_dict()["families"] == 2
    assert "alpha" in module.line()
    with pytest.raises(errors.ScanError, match="包名不能为空"):
        _module("", _raw("A", "ValueError"))
    with pytest.raises(errors.ScanError, match="一个 ClassDef 都没有"):
        _module("alpha")
    with pytest.raises(errors.ScanError, match="重名的族"):
        _module("alpha", _raw("A", "ValueError"), _raw("A", "Exception"))


def test_scan_module_reads_a_real_file() -> None:
    """扫一份真实文件：族非空，且含它自己的根族。"""
    path = next(
        item for item in scan_module.error_module_paths() if item.parent.name == "failure_ledger"
    )
    module = scan_module.scan_module(path)
    assert module.package == "failure_ledger"
    assert "LedgerError" in module.family_names


def test_scan_all_is_reproducible() -> None:
    """两次扫描逐位相同。"""
    first = scan_module.scan_all()
    second = scan_module.scan_all()
    assert [m.to_dict() for m in first] == [m.to_dict() for m in second]


def test_packages_without_errors_is_a_sorted_tuple() -> None:
    """列出没有 errors.py 的子包（读数，不是失败）。"""
    names = scan_module.packages_without_errors()
    assert isinstance(names, tuple)
    assert list(names) == sorted(names)
    assert "failure_ledger" not in names


def test_scan_lines_has_summary() -> None:
    """扫描逐行打印，末行是合计。"""
    modules = _small_modules()
    lines = scan_module.scan_lines(modules)
    assert lines[0].startswith("扫描表")
    assert "合计 2 份" in lines[-1]


# --------------------------------------------------------------------------- 归属


def test_resolve_base_four_kinds() -> None:
    """四种归属各走一遍（判定顺序：local → builtin → imported → unresolved）。"""
    module = _small_modules()[0]
    local = frozenset(module.family_names)
    assert resolve_module.resolve_base("AlphaError", local_names=local, module=module).kind == (
        types.BASE_KIND_LOCAL
    )
    assert resolve_module.resolve_base("ValueError", local_names=local, module=module).kind == (
        types.BASE_KIND_BUILTIN
    )
    imported = resolve_module.resolve_base("CoreShapeError", local_names=local, module=module)
    assert imported.kind == types.BASE_KIND_IMPORTED
    assert imported.target.endswith("ShapeError")
    assert resolve_module.resolve_base("Nope", local_names=local, module=module).kind == (
        types.BASE_KIND_UNRESOLVED
    )


def test_local_wins_over_builtin_and_imported() -> None:
    """同名的 local 优先于内置与导入（判定顺序不能换）。"""
    module = _module(
        "gamma",
        _raw("ValueError", "Exception"),
        aliases=(("ValueError", "somewhere.ValueError"),),
    )
    ref = resolve_module.resolve_base(
        "ValueError", local_names=frozenset(module.family_names), module=module
    )
    assert ref.kind == types.BASE_KIND_LOCAL


def test_base_ref_guards() -> None:
    """归属记录的四条护栏与两个渲染方法。"""
    ref = resolve_module.BaseRef(raw="X", kind=types.BASE_KIND_BUILTIN, target="X")
    assert ref.to_dict()["kind"] == types.BASE_KIND_BUILTIN
    assert "X → builtin" in ref.line()
    assert ref.is_local is False
    with pytest.raises(errors.ResolveError, match="基类名不能为空"):
        resolve_module.BaseRef(raw="", kind=types.BASE_KIND_BUILTIN, target="X")
    with pytest.raises(errors.ParameterError, match="未知的基类归属"):
        resolve_module.BaseRef(raw="X", kind="nope", target="X")
    with pytest.raises(errors.ResolveError, match="没有去向"):
        resolve_module.BaseRef(raw="X", kind=types.BASE_KIND_IMPORTED, target="")
    with pytest.raises(errors.ResolveError, match="原样保留"):
        resolve_module.BaseRef(raw="X", kind=types.BASE_KIND_UNRESOLVED, target="Y")


def test_resolve_family_and_unresolved_of() -> None:
    """归属一族：local 的 target 是本地族名，unresolved 被挑出来。"""
    module = _module("delta", _raw("DError", "ValueError", "Missing"))
    refs = resolve_module.resolve_family(module.families[0], module)
    assert refs[0].is_local is False
    assert resolve_module.unresolved_of(refs) == ("Missing",)
    assert resolve_module.local_names_of(module) == frozenset({"DError"})


def test_require_resolved_ok_and_raise() -> None:
    """require_resolved：全解析放行；有未解析时抛 ResolveError。"""
    good = _small_modules()[0]
    refs = resolve_module.resolve_family(good.families[1], good)
    assert resolve_module.require_resolved(good, refs) == refs
    bad = _module("epsilon", _raw("EError", "ValueError", "Mystery"))
    with pytest.raises(errors.ResolveError, match="无法归属的基类名"):
        resolve_module.require_resolved(bad, resolve_module.resolve_family(bad.families[0], bad))


def test_resolve_lines() -> None:
    """归属逐行打印。"""
    module = _small_modules()[0]
    refs = resolve_module.resolve_family(module.families[0], module)
    lines = resolve_module.resolve_lines(refs)
    assert lines[0].startswith("归属表")


# --------------------------------------------------------------------------- 台账


def test_family_views() -> None:
    """族：身份、父族、跨包基类与两个渲染方法。"""
    ledger = ledger_module.build_ledger(_small_modules())
    root = ledger.family_of("alpha.AlphaError")
    sub = ledger.family_of("alpha.AlphaShapeError")
    assert root.is_root and root.kind == types.FAMILY_KIND_ROOT
    assert sub.kind == types.FAMILY_KIND_SUB
    assert sub.local_parents == ("AlphaError",)
    assert [ref.raw for ref in sub.cross_package_bases] == ["CoreShapeError"]
    assert sub.to_dict()["qualified"] == "alpha.AlphaShapeError"
    assert "alpha.AlphaShapeError" in sub.line()
    empty = ledger_module.Family(package="p", name="E", bases=())
    assert "（无基类）" in empty.line()


def test_family_guards() -> None:
    """族的两条护栏。"""
    with pytest.raises(errors.ParameterError, match="包名与名字都不能为空"):
        ledger_module.Family(package="", name="E", bases=())
    with pytest.raises(errors.ParameterError, match="包名与名字都不能为空"):
        ledger_module.Family(package="p", name="", bases=())


def test_build_ledger_default_readings() -> None:
    """真实台账：模块 / 族非空、根与派生加起来等于总数、未解析为空。"""
    ledger = ledger_module.build_ledger()
    assert ledger.module_count >= 30
    assert ledger.family_count >= 100
    assert len(ledger.roots) == ledger.module_count
    assert len(ledger.roots) + len(ledger.subs) == ledger.family_count
    assert ledger.unresolved_bases() == ()
    assert ledger.min_bases() >= 1
    assert ledger.max_depth() >= 1
    assert len(ledger.digest()) == ledger_module.LEDGER_DIGEST_LENGTH
    assert "台账" in ledger.line()
    assert ledger.to_dict()["families"] == ledger.family_count


def test_ledger_lookups_and_errors() -> None:
    """按包 / 按族名取值，以及三个"取不到"的分支。"""
    ledger = ledger_module.build_ledger(_small_modules())
    assert [family.name for family in ledger.by_package("alpha")] == [
        "AlphaError",
        "AlphaShapeError",
    ]
    assert ledger.root_of("alpha").name == "AlphaError"
    assert ledger.depth_of("alpha.AlphaShapeError") == 1
    with pytest.raises(errors.ParameterError, match="台账里没有包"):
        ledger.by_package("nope")
    with pytest.raises(errors.ParameterError, match="台账里没有族"):
        ledger.family_of("nope.nope")
    with pytest.raises(errors.ParameterError, match="台账里没有族"):
        ledger.depth_of("nope.nope")


def test_ledger_guards() -> None:
    """台账的三条护栏：空、模块重名、包.名 重复。"""
    good = ledger_module.build_ledger(_small_modules())
    with pytest.raises(errors.LedgerBuildError, match="台账不能为空"):
        ledger_module.Ledger(modules=(), families=())
    with pytest.raises(errors.LedgerBuildError, match="模块名单里有重复"):
        dataclasses.replace(good, modules=("alpha", "alpha"))
    with pytest.raises(errors.LedgerBuildError, match="重复的 ``包.名``"):
        ledger_module.Ledger(modules=("alpha",), families=(good.families[0], good.families[0]))


def test_root_of_rejects_zero_or_two_roots() -> None:
    """root_of：0 个根与 2 个根都当场拒绝。"""
    two = ledger_module.build_ledger((_module("zeta", _raw("A", "ValueError"), _raw("B", "Exception")),))
    with pytest.raises(errors.ParameterError, match="有 2 个根族"):
        two.root_of("zeta")
    none = ledger_module.build_ledger((_module("eta", _raw("A", "B"), _raw("B", "A")),))
    with pytest.raises(errors.ParameterError, match="有 0 个根族"):
        none.root_of("eta")


def test_inheritance_cycle_is_rejected() -> None:
    """源码里真的出现继承环时，算深度当场抛 LedgerBuildError。"""
    cyclic = ledger_module.build_ledger((_module("theta", _raw("A", "B"), _raw("B", "A")),))
    with pytest.raises(errors.LedgerBuildError, match="继承里出现环"):
        cyclic.max_depth()


def test_local_parent_missing_is_rejected() -> None:
    """手工造一条"local 基类不在同一包"的坏归属，算深度时当场抛。"""
    bad = ledger_module.Family(
        package="p",
        name="Solo",
        bases=(resolve_module.BaseRef(raw="Ghost", kind=types.BASE_KIND_LOCAL, target="Ghost"),),
    )
    ledger = ledger_module.Ledger(modules=("p",), families=(bad,))
    with pytest.raises(errors.LedgerBuildError, match="不在同一包里"):
        ledger.max_depth()


def test_ledger_reproducible_and_digest() -> None:
    """同一批模块两次构建逐位相同；different 输入差异大于 0。"""
    modules = _small_modules()
    first = ledger_module.build_ledger(modules)
    second = ledger_module.build_ledger(modules)
    assert first.diff_count(second) == 0
    assert first.digest() == second.digest()
    assert ledger_module.ledger_digest(first) == first.digest()
    assert ledger_module.require_reproducible(first, second) is first
    other = ledger_module.build_ledger((_module("omega", _raw("OError", "Exception")),))
    assert first.diff_count(other) > 0
    with pytest.raises(errors.LedgerBuildError, match="两份台账差"):
        ledger_module.require_reproducible(first, other)


def test_require_coverage_ok_and_raise() -> None:
    """require_coverage：默认放行；期望里多一个包时抛 CoverageError。"""
    assert ledger_module.require_coverage() is not None
    ledger = ledger_module.build_ledger(_small_modules())
    with pytest.raises(errors.CoverageError, match="台账漏了"):
        ledger_module.require_coverage(ledger, expected=("alpha", "beta", "ghost"))


def test_ledger_lines_with_and_without_limit() -> None:
    """台账逐行打印：可截断，非法 limit 被拒。"""
    ledger = ledger_module.build_ledger(_small_modules())
    limited = ledger_module.ledger_lines(ledger, limit=1)
    assert limited[0].startswith("台账表")
    assert len(limited) == 1 + 1 + 1
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        ledger_module.ledger_lines(ledger, limit=0)


# --------------------------------------------------------------------------- 性质


def test_crosscheck_three_criteria() -> None:
    """三类判据各自的判定与渲染。"""
    equal = verify.CrossCheck(
        name="相等", left="a", right="b", reading=2.0, expected=2.0, exact=True
    )
    assert equal.criterion == types.CRITERION_EQUALITY
    assert equal.passed is True
    assert "读数" in equal.line()
    upper = verify.CrossCheck(
        name="上界", left="a", right="b", reading=0.0, expected=0.0, upper_bound=0.0
    )
    assert upper.criterion == types.CRITERION_UPPER_BOUND
    assert upper.passed is True
    assert "≤ 上界" in upper.line()
    lower = verify.CrossCheck(
        name="下界", left="a", right="b", reading=1.0, expected=1.0, lower_bound=1.0
    )
    assert lower.criterion == types.CRITERION_LOWER_BOUND
    assert lower.passed is True
    assert "≥ 下界" in lower.line()
    assert lower.to_dict()["passed"] is True


def test_crosscheck_failures_and_guards() -> None:
    """判据不满足时的读数 + 五条构造护栏。"""
    failed = verify.CrossCheck(
        name="下界", left="a", right="b", reading=0.0, expected=1.0, lower_bound=1.0
    )
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
    with pytest.raises(errors.ScanError, match="未全部通过"):
        bad.require_ok()


def test_check_functions_default() -> None:
    """七条性质函数在默认输入上全部通过。"""
    assert verify.check_ledger_covers_all_error_modules().passed is True
    assert verify.check_qualified_names_are_unique().passed is True
    assert verify.check_ledger_is_reproducible().passed is True
    assert verify.check_every_module_has_exactly_one_root().passed is True
    assert verify.check_base_references_resolve().passed is True
    assert verify.check_every_family_has_a_parent().passed is True
    assert verify.check_inheritance_depth_is_positive().passed is True


def test_check_coverage_fails_when_missing() -> None:
    """覆盖检查：期望名单里多一项时失败并点名。"""
    ledger = ledger_module.build_ledger(_small_modules())
    outcome = verify.check_ledger_covers_all_error_modules(
        ledger, expected=("alpha", "beta", "ghost")
    )
    assert outcome.passed is False
    assert "ghost" in outcome.evidence[1]


def test_check_unique_names_fails_on_duplicate() -> None:
    """名字唯一：手工塞两个同名族时失败。"""
    dup = ledger_module.Family(package="p", name="E", bases=())
    outcome = verify.check_qualified_names_are_unique((dup, dup))
    assert outcome.passed is False
    assert "p.E" in outcome.evidence[1]


def test_check_reproducible_fails_on_different_inputs() -> None:
    """可复算：两份不同的台账差异大于 0。"""
    first = ledger_module.build_ledger((_module("a", _raw("A", "ValueError")),))
    second = ledger_module.build_ledger((_module("b", _raw("B", "Exception")),))
    assert verify.check_ledger_is_reproducible(first, second).passed is False


def test_check_one_root_fails_on_two_roots() -> None:
    """每包一个根：两个根时失败。"""
    two = ledger_module.build_ledger((_module("zeta", _raw("A", "ValueError"), _raw("B", "Exception")),))
    outcome = verify.check_every_module_has_exactly_one_root(two)
    assert outcome.passed is False
    assert "zeta" in outcome.evidence[1]


def test_check_resolve_fails_on_unresolved() -> None:
    """基类归属：出现未解析名时（上界方向）失败。"""
    bad = ledger_module.build_ledger((_module("eps", _raw("E", "ValueError", "Mystery")),))
    outcome = verify.check_base_references_resolve(bad)
    assert outcome.passed is False
    assert outcome.criterion == types.CRITERION_UPPER_BOUND
    assert "Mystery" in outcome.evidence[1]


def test_check_parent_and_depth_fail_directions() -> None:
    """基类数（下界）与继承深度（下界）的失败方向。"""
    no_base = ledger_module.build_ledger((_module("nb", _raw("N")),))
    parent = verify.check_every_family_has_a_parent(no_base)
    assert parent.passed is False
    assert parent.criterion == types.CRITERION_LOWER_BOUND
    flat = ledger_module.build_ledger(
        (_module("flat", _raw("A", "ValueError"), _raw("B", "Exception")),)
    )
    depth = verify.check_inheritance_depth_is_positive(flat)
    assert depth.passed is False
    assert depth.criterion == types.CRITERION_LOWER_BOUND


def test_check_all_default_and_injected() -> None:
    """check_all：默认全通过；注入小台账时给出同一份 7 行报告。"""
    report = verify.check_all()
    assert report.ok is True
    assert [outcome.name for outcome in report.outcomes] == list(types.FAILURE_LEDGER_PROPERTIES)
    modules = _small_modules()
    injected = verify.check_all(modules=modules, expected=("alpha", "beta"))
    assert [outcome.name for outcome in injected.outcomes] == list(types.FAILURE_LEDGER_PROPERTIES)
    assert injected.ok is True


def test_require_ok() -> None:
    """require_ok：默认放行，注入一份失败的报告时抛 ScanError。"""
    assert verify.require_ok() is not None
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="x", applicable=True, passed=False),)
    )
    with pytest.raises(errors.ScanError, match="未全部通过"):
        verify.require_ok(bad)


# --------------------------------------------------------------------------- 表


def test_module_rows() -> None:
    """模块表：行数与模块数一致，每行带根族名。"""
    ledger = ledger_module.build_ledger(_small_modules())
    rows = study.module_rows(ledger)
    assert len(rows) == 2
    assert rows[0].root == "AlphaError"
    assert "alpha" in rows[0].line()


def test_module_rows_without_root() -> None:
    """模块表对"0 个根"的模块也有话说（不崩）。"""
    cyclic = ledger_module.build_ledger((_module("theta", _raw("A", "B"), _raw("B", "A")),))
    rows = study.module_rows(cyclic)
    assert rows[0].root == "（无）"


def test_family_rows() -> None:
    """族表：按 limit 截断。"""
    ledger = ledger_module.build_ledger()
    assert len(study.family_rows(ledger, limit=5)) == 5
    assert "|" in study.family_rows(ledger, limit=1)[0].line()


def test_base_kind_rows() -> None:
    """归属表：四行、顺序与 BASE_KINDS 一致、计数之和等于基类引用总数。"""
    ledger = ledger_module.build_ledger()
    rows = study.base_kind_rows(ledger)
    assert [row.kind for row in rows] == list(types.BASE_KINDS)
    assert all(row.count >= 0 for row in rows)
    total_refs = sum(len(family.bases) for family in ledger.families)
    assert sum(row.count for row in rows) == total_refs
    assert "local" in rows[0].line()


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
    assert len(study.boundary_lines()) == len(types.FAILURE_LEDGER_BOUNDARIES)


def test_study_lines_runs_all_four_tables() -> None:
    """四张表一次跑完：四个小标题。"""
    lines = study.study_lines()
    headings = [line for line in lines if line.startswith("== ")]
    assert len(headings) == 4
    assert any("模块表" in line for line in headings)
    assert any("性质表" in line for line in headings)


def test_study_lines_accepts_injected_inputs() -> None:
    """四张表接收注入的台账。"""
    ledger = ledger_module.build_ledger(_small_modules())
    lines = study.study_lines(ledger=ledger, family_limit=2)
    assert len([line for line in lines if line.startswith("== ")]) == 4


def test_to_dict_lines() -> None:
    """把带 to_dict 的行折成 JSON 化字段（四张表都能导出）。"""
    assert study.to_dict_lines(study.module_rows())[0]["package"]
    assert study.to_dict_lines(study.family_rows(limit=1))[0]["qualified"]
    assert study.to_dict_lines(study.base_kind_rows())[0]["kind"]
    assert study.to_dict_lines(study.property_rows())[0]["criterion"]


# --------------------------------------------------------------------------- 包


def test_package_all_is_sorted_and_unique() -> None:
    """包的公开名单：字母序、无重复、不含子模块名。"""
    import smart_research_agent.failure_ledger as package

    assert package.__all__ == sorted(package.__all__)
    assert len(package.__all__) == len(set(package.__all__))
    assert set(package.__all__) & {
        "errors",
        "types",
        "scan",
        "resolve",
        "ledger",
        "verify",
        "study",
    } == set()


def test_package_exposes_key_names() -> None:
    """包级命名空间真的导入了关键名字（不是只在 __all__ 里）。"""
    import smart_research_agent.failure_ledger as package

    for name in (
        "build_ledger",
        "scan_all",
        "resolve_base",
        "check_all",
        "study_lines",
        "LedgerError",
    ):
        assert name in package.__all__
        assert hasattr(package, name)
