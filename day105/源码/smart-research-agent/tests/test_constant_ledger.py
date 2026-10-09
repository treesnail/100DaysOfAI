"""``constant_ledger``：把 100 天钉死的常量编成一份可复算的台账（day105）.

本文件覆盖新包的六个模块：口径表、失败族、解析、台账、性质与表。
样本有两类：**真实源码**（``smart_research_agent/**/*.py``）与
**注入的小模块**（构造反例时用）。因此这一课考的就是
"同一批文件能不能两次编出同一份台账、每一个常量能不能被归属、
以及报出来的同名与冲突到底是不是真的"。

> 注意：本文件**不写死**模块数 / 常量数 / 同名数 / 冲突数 / 摘要——它们会随着仓库新增子包而变化。
> 断言一律用"结构不变式"或下界；另有**一条子进程测试**专门钉住"跨进程可复算"。
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest

from smart_research_agent.constant_ledger import errors, ledger as ledger_module, parse as parse_module
from smart_research_agent.constant_ledger import study, types, verify

# --------------------------------------------------------------------------- 注入样本


def _const(
    module: str,
    name: str,
    *,
    line: int = 1,
    annotated: bool = False,
    value_kind: str = types.VALUE_LITERAL,
    literal: str | None = "1",
    path: str = "/x/m.py",
) -> parse_module.ConstantDef:
    """一条常量定义（注意：它是"值"的载体，不携带路径）."""
    return parse_module.ConstantDef(
        module=module,
        name=name,
        line=line,
        annotated=annotated,
        value_kind=value_kind,
        literal=literal,
    )


def _module(
    module: str,
    *defs: parse_module.ConstantDef,
    is_package: bool = False,
    path: str = "/x/m.py",
) -> parse_module.ModuleConstants:
    """一份被解析的小模块（路径可以指向真实文件，供独立复核用）."""
    return parse_module.ModuleConstants(
        module=module,
        is_package=is_package,
        path=path,
        constants=tuple(sorted(defs, key=lambda item: item.key)),
    )


def _shared_pair() -> tuple[parse_module.ModuleConstants, ...]:
    """两个模块都钉了 `SHARED`，且取值**相同**（consistent）。"""
    return (
        _module("pkg.a", _const("pkg.a", "SHARED", literal="1"), _const("pkg.a", "ONLYA", literal="1")),
        _module("pkg.b", _const("pkg.b", "SHARED", literal="1")),
    )


def _conflict_pair() -> tuple[parse_module.ModuleConstants, ...]:
    """两个模块都把 `EPSILON` 钉住，但取值**不同**（conflict）。"""
    return (
        _module("pkg.a", _const("pkg.a", "EPSILON", literal="1e-05")),
        _module("pkg.b", _const("pkg.b", "EPSILON", literal="1e-08")),
    )


# --------------------------------------------------------------------------- 口径表


def test_value_kinds_and_relations_are_closed() -> None:
    """两类取值形态与四类同名关系：名单与说明逐键对齐。"""
    assert len(types.VALUE_KINDS) == 2
    assert set(types.VALUE_DESCRIPTIONS) == set(types.VALUE_KINDS)
    assert len(types.RELATIONS) == 4
    assert set(types.RELATION_DESCRIPTIONS) == set(types.RELATIONS)
    assert types.require_value_kind(types.VALUE_OPAQUE) == types.VALUE_OPAQUE
    assert types.require_relation(types.RELATION_CONFLICT) == types.RELATION_CONFLICT
    with pytest.raises(errors.ParameterError, match="未知的取值形态"):
        types.require_value_kind("nope")
    with pytest.raises(errors.ParameterError, match="未知的同名关系"):
        types.require_relation("nope")


def test_seven_properties_and_three_criteria() -> None:
    """7 条性质：名单与规格表逐键对齐，且三类判据都有性质。"""
    assert len(types.CONSTANT_LEDGER_PROPERTIES) == 7
    assert set(types.PROPERTY_SPECS) == set(types.CONSTANT_LEDGER_PROPERTIES)
    criteria = {spec.criterion for spec in types.property_specs()}
    assert criteria == set(types.CRITERIA)
    assert criteria == {
        types.CRITERION_EQUALITY,
        types.CRITERION_UPPER_BOUND,
        types.CRITERION_LOWER_BOUND,
    }


def test_notes_are_ten_and_boundaries_five() -> None:
    """十条笔记（有序）与五条边界。"""
    assert len(types.CONSTANT_LEDGER_NOTES) == 10
    assert types.CONSTANT_LEDGER_NOTES_ORDER == tuple(types.CONSTANT_LEDGER_NOTES)
    assert all(types.CONSTANT_LEDGER_NOTES.values())
    assert len(types.CONSTANT_LEDGER_BOUNDARIES) == 5
    assert all(types.CONSTANT_LEDGER_BOUNDARIES)


def test_docs_of_value_kind_and_relation() -> None:
    """两个说明函数：合法值返回一句话，非法值当场拒绝。"""
    assert types.docs_of_value_kind(types.VALUE_LITERAL)
    assert types.docs_of_relation(types.RELATION_UNIQUE)
    with pytest.raises(errors.ParameterError, match="未知的取值形态"):
        types.docs_of_value_kind("nope")
    with pytest.raises(errors.ParameterError, match="未知的同名关系"):
        types.docs_of_relation("nope")


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
        assert issubclass(cls, errors.ConstantError)
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


# --------------------------------------------------------------------------- 解析


def test_canonical_value_is_deterministic() -> None:
    """取值指纹：集合先排序再拼接，容器递归折，其它走 repr。"""
    assert parse_module.canonical_value({3, 1, 2}) == "{1, 2, 3}"
    assert parse_module.canonical_value(frozenset({"b", "a"})) == "{'a', 'b'}"
    assert parse_module.canonical_value((1,)) == "(1,)"
    assert parse_module.canonical_value((1, 2)) == "(1, 2)"
    assert parse_module.canonical_value([1, 2]) == "[1, 2]"
    assert parse_module.canonical_value({"a": 1}) == "{'a': 1}"
    assert parse_module.canonical_value("x") == "'x'"
    assert parse_module.canonical_value(1e-5) == "1e-05"


def test_is_constant_name_rules() -> None:
    """常量名规则：全大写标识符、长度 >= 2、不以 ``_`` 开头。"""
    assert parse_module.is_constant_name("ALPHA") is True
    assert parse_module.is_constant_name("A1") is True
    assert parse_module.is_constant_name("A") is False
    assert parse_module.is_constant_name("_ALPHA") is False
    assert parse_module.is_constant_name("Alpha") is False


def test_fingerprint_literal_and_opaque() -> None:
    """取值指纹：字面量求得出、函数调用求不出、右端缺失也记 opaque。"""
    import ast

    literal = ast.parse("ALPHA = 1e-5").body[0].value  # type: ignore[attr-defined]
    assert parse_module.fingerprint(literal) == (types.VALUE_LITERAL, "1e-05")
    opaque = ast.parse("ALPHA = Path(__file__)").body[0].value  # type: ignore[attr-defined]
    assert parse_module.fingerprint(opaque) == (types.VALUE_OPAQUE, None)
    assert parse_module.fingerprint(None) == (types.VALUE_OPAQUE, None)


def test_constant_def_views_and_guards() -> None:
    """常量定义：排序键、渲染与六条护栏。"""
    item = _const("pkg.a", "ALPHA", line=3)
    assert item.key == ("ALPHA", 3)
    assert item.to_dict()["name"] == "ALPHA"
    assert "ALPHA" in item.line_text()
    opaque = _const("pkg.a", "BETA", value_kind=types.VALUE_OPAQUE, literal=None, annotated=True)
    assert "读不出来" in opaque.line_text() and "带注解" in opaque.line_text()
    with pytest.raises(errors.AssignError, match="不能为空"):
        _const("", "ALPHA")
    with pytest.raises(errors.AssignError, match="常量名"):
        _const("pkg.a", "lower")
    with pytest.raises(errors.AssignError, match="常量名"):
        _const("pkg.a", "A")
    with pytest.raises(errors.AssignError, match="没有取值指纹"):
        _const("pkg.a", "ALPHA", literal=None)
    with pytest.raises(errors.AssignError, match="带着取值指纹"):
        _const("pkg.a", "ALPHA", value_kind=types.VALUE_OPAQUE, literal="1")
    with pytest.raises(errors.AssignError, match="行号必须是 >= 1"):
        _const("pkg.a", "ALPHA", line=0)


def test_module_constants_views_and_guards() -> None:
    """模块行：名字、取值、主定义、重名、渲染与三条护栏。"""
    module = _module(
        "pkg.a",
        _const("pkg.a", "ALPHA", line=1, literal="1"),
        _const("pkg.a", "ALPHA", line=5, literal="2"),
        _const("pkg.a", "BETA", line=3, value_kind=types.VALUE_OPAQUE, literal=None),
    )
    assert module.names() == ("ALPHA", "BETA")
    assert len(module.defs_of("ALPHA")) == 2
    assert module.primary_of("ALPHA").line == 5
    assert module.duplicate_names() == ("ALPHA",)
    assert module.to_dict()["duplicates"] == 1
    assert "重名" in module.line()
    assert [item.name for item in module.primary_defs()] == ["ALPHA", "BETA"]
    with pytest.raises(errors.ParseError, match="模块名不能为空"):
        _module("")
    with pytest.raises(errors.ParseError, match="排序且不重复"):
        parse_module.ModuleConstants(
            module="pkg.a", is_package=False, path="/x.py",
            constants=(
                _const("pkg.a", "BETA", line=2),
                _const("pkg.a", "ALPHA", line=1),
            ),
        )
    with pytest.raises(errors.ParseError, match="混进了别的模块"):
        _module("pkg.a", _const("pkg.b", "ALPHA"))
    with pytest.raises(errors.ParseError, match="没有常量"):
        _module("pkg.a").primary_of("NOPE")


def test_module_files_and_names(tmp_path: pathlib.Path) -> None:
    """被扫描的文件与模块命名：非空、排序、``__init__.py`` 折成包名。"""
    files = parse_module.module_files()
    assert files
    relative = [path.relative_to(parse_module.package_dir()).as_posix() for path in files]
    assert relative == sorted(relative)
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


def test_package_dir_and_scan_guards(tmp_path: pathlib.Path) -> None:
    """包目录 / 语法错 / 站外的路径：三处当场拒绝。"""
    assert parse_module.package_dir().is_dir()
    with pytest.raises(errors.ParseError, match="包目录不存在"):
        parse_module.package_dir(tmp_path)
    (tmp_path / types.PACKAGE_NAME).mkdir()
    with pytest.raises(errors.ParseError, match=r"一份 \.py 都没有"):
        parse_module.module_files(tmp_path)
    with pytest.raises(errors.ParseError, match="语法错误"):
        parse_module.parse_source("def :(", origin="<mem>")
    outside = tmp_path / "elsewhere.py"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(errors.ParseError, match="不在被扫描的包目录"):
        parse_module.module_name(outside, tmp_path)


def test_read_source_accepts_bom(tmp_path: pathlib.Path) -> None:
    """读源码容忍 BOM。"""
    target = tmp_path / "m.py"
    target.write_text("ALPHA = 1\n", encoding="utf-8-sig")
    assert "ALPHA = 1" in parse_module.read_source(target)


def test_iter_module_scope_and_constant_defs() -> None:
    """模块级作用域与常量收集：五种写法，两种不算常量。"""
    import ast

    tree = ast.parse(
        "A = 1\n"
        "ALPHA = 1\n"
        "BETA: int = 2\n"
        "GAMMA += 3\n"
        "lower += 4\n"
        "lower2: int = 5\n"
        'DCT["k"]: int = 6\n'
        "left = right = 7\n"
        "DELTA, EPS = 1, 2\n"
        "if True:\n    ZETA = 5\n"
        "try:\n    ETA = 6\nexcept Exception:\n    THETA = 7\n"
        "def f():\n    IOTA = 8\n"
        "class C:\n    KAPPA = 9\n"
    )
    defs = parse_module.constant_defs(tree, module="pkg.a")
    names = [item.name for item in defs]
    assert names == ["ALPHA", "BETA", "ETA", "GAMMA", "THETA", "ZETA"]
    by_name = {item.name: item for item in defs}
    assert by_name["BETA"].annotated is True
    assert by_name["GAMMA"].value_kind == types.VALUE_OPAQUE
    assert by_name["ALPHA"].literal == "1"
    scoped = list(parse_module.iter_module_scope(tree.body))
    assert scoped[0] is tree.body[0]


def test_value_segments_reads_source(tmp_path: pathlib.Path) -> None:
    """独立复核的源码片段：只认模块级常量，最后一条算数，空白被折平。"""
    target = tmp_path / "m.py"
    target.write_text(
        "ALPHA = 1\n"
        "BETA = Path(__file__)\n"
        "ALPHA =   2\n"
        "lower = 3\n",
        encoding="utf-8",
    )
    table = parse_module.value_segments(target)
    assert table["ALPHA"] == "2"
    assert table["BETA"] == "Path(__file__)"
    assert "lower" not in table


def test_scan_module_and_scan_all() -> None:
    """扫一份真实文件 + 两次扫描逐位相同。"""
    path = parse_module.PROJECT_ROOT / "smart_research_agent" / "constant_ledger" / "types.py"
    scanned = parse_module.scan_module(path, parse_module.PROJECT_ROOT)
    assert scanned.module == "smart_research_agent.constant_ledger.types"
    assert scanned.is_package is False
    assert "VALUE_KINDS" in scanned.names()
    first = parse_module.scan_all()
    second = parse_module.scan_all()
    assert [entry.to_dict() for entry in first] == [entry.to_dict() for entry in second]
    assert parse_module.expected_modules() == tuple(sorted(entry.module for entry in first))


def test_scan_lines() -> None:
    """解析逐行打印（末行是合计，可截断）。"""
    lines = parse_module.scan_lines(_shared_pair())
    assert lines[0].startswith("解析表")
    assert "合计" in lines[-1] and "常量" in lines[-1]
    assert len(parse_module.scan_lines(_shared_pair(), limit=1)) == 3


def test_duplicate_assignments_and_require() -> None:
    """重复赋值：默认没有；注入一个"同名两次"的模块时能被点出并拒绝交付。"""
    doubled = _module(
        "pkg.a",
        _const("pkg.a", "ALPHA", line=1),
        _const("pkg.a", "ALPHA", line=2, literal="2"),
    )
    assert parse_module.duplicate_assignments((doubled,)) == ("pkg.a",)
    with pytest.raises(errors.LedgerBuildError, match="赋了两次以上"):
        parse_module.require_unique_assignments((doubled,))
    good = _shared_pair()
    assert parse_module.require_unique_assignments(good) is good


def test_ledger_digest_is_stable_across_processes() -> None:
    """**跨进程**可复算：两个不同的哈希种子必须给出同一个摘要.

    这是本课最硬的一条测试：它正是"集合字面量必须先排序"那条纪律的守卫
    （`set` 的 `repr` 顺序随进程变化，直接 `repr` 会给出两个摘要）。
    """
    script = (
        "from smart_research_agent.constant_ledger.ledger import build_ledger;"
        "from smart_research_agent.constant_ledger.parse import scan_all;"
        "print(build_ledger(scan_all()).digest())"
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


# --------------------------------------------------------------------------- 台账


def test_classify_four_relations() -> None:
    """四类关系：unique / incomparable / conflict / consistent。"""
    only_one = (_const("pkg.a", "ALPHA"),)
    assert ledger_module.classify(only_one) == types.RELATION_UNIQUE
    opaque = (_const("pkg.a", "ALPHA"), _const("pkg.b", "ALPHA", value_kind=types.VALUE_OPAQUE, literal=None))
    assert ledger_module.classify(opaque) == types.RELATION_INCOMPARABLE
    conflict = (_const("pkg.a", "ALPHA", literal="1"), _const("pkg.b", "ALPHA", literal="2"))
    assert ledger_module.classify(conflict) == types.RELATION_CONFLICT
    consistent = (_const("pkg.a", "ALPHA", literal="1"), _const("pkg.b", "ALPHA", literal="1"))
    assert ledger_module.classify(consistent) == types.RELATION_CONSISTENT
    with pytest.raises(errors.ParameterError, match="不能是空的"):
        ledger_module.classify(())


def test_name_group_views_and_guards() -> None:
    """同名组：成员模块、取值、冲突标记、渲染与五条护栏。"""
    members = tuple(sorted(_conflict_pair(), key=lambda entry: entry.module))
    group = ledger_module.NameGroup(
        name="EPSILON",
        relation=types.RELATION_CONFLICT,
        members=(
            _const("pkg.a", "EPSILON", literal="1e-05"),
            _const("pkg.b", "EPSILON", literal="1e-08"),
        ),
    )
    assert group.module_names == ("pkg.a", "pkg.b")
    assert group.is_conflict is True
    assert group.literals() == ("1e-05", "1e-08")
    assert group.to_dict()["relation"] == types.RELATION_CONFLICT
    assert "conflict" in group.line()
    assert members  # 注入样本本身合法
    with pytest.raises(errors.ParameterError, match="名字不能为空"):
        ledger_module.NameGroup(
            name="", relation=types.RELATION_UNIQUE, members=(_const("pkg.a", "ALPHA"),)
        )
    with pytest.raises(errors.ParameterError, match="未知的同名关系"):
        ledger_module.NameGroup(name="ALPHA", relation="nope", members=(_const("pkg.a", "ALPHA"),))
    with pytest.raises(errors.ParameterError, match="不能没有成员"):
        ledger_module.NameGroup(name="ALPHA", relation=types.RELATION_UNIQUE, members=())
    with pytest.raises(errors.ParameterError, match="混进了别的名字"):
        ledger_module.NameGroup(
            name="ALPHA", relation=types.RELATION_UNIQUE, members=(_const("pkg.a", "BETA"),)
        )
    with pytest.raises(errors.LedgerBuildError, match="关系标成了"):
        ledger_module.NameGroup(
            name="ALPHA",
            relation=types.RELATION_UNIQUE,
            members=(_const("pkg.a", "ALPHA"), _const("pkg.b", "ALPHA")),
        )


def test_ledger_views() -> None:
    """台账：模块名单、分组、计数、同名组、渲染。"""
    ledger = ledger_module.build_ledger(_shared_pair())
    assert ledger.modules == ("pkg.a", "pkg.b")
    assert ledger.module_count == 2
    assert ledger.module_set == frozenset({"pkg.a", "pkg.b"})
    assert ledger.entry_of("pkg.a").module == "pkg.a"
    assert ledger.package_of("pkg.a") == "a"
    assert [name for name, _members in ledger.package_groups()] == ["a", "b"]
    assert ledger.constant_count() == 3
    assert ledger.assignment_count() == 3
    assert ledger.literal_count() == 3
    assert ledger.opaque_count() == 0
    assert dict(ledger.value_kind_counts()) == {types.VALUE_LITERAL: 3, types.VALUE_OPAQUE: 0}
    assert len(ledger.name_groups()) == 2
    assert [group.name for group in ledger.shared_groups()] == ["SHARED"]
    assert ledger.to_dict()["constants"] == 3
    assert "台账：" in ledger.line()


def test_ledger_conflicts_and_incomparables() -> None:
    """同名不同值与"无法比较"：各挑各的，且分类正确。"""
    ledger = ledger_module.build_ledger(_conflict_pair())
    assert [group.name for group in ledger.conflicts()] == ["EPSILON"]
    assert ledger.incomparables() == ()
    assert ledger.duplicate_modules() == ()
    opaque = (
        _module("pkg.a", _const("pkg.a", "ROOT", value_kind=types.VALUE_OPAQUE, literal=None)),
        _module("pkg.b", _const("pkg.b", "ROOT", literal="1")),
    )
    mixed = ledger_module.build_ledger(opaque)
    assert [group.name for group in mixed.incomparables()] == ["ROOT"]
    assert mixed.conflicts() == ()


def test_ledger_guards_and_lookups() -> None:
    """台账的两条护栏与两个未知模块入口。"""
    with pytest.raises(errors.LedgerBuildError, match="不能没有模块行"):
        ledger_module.ConstantLedger(entries=())
    with pytest.raises(errors.LedgerBuildError, match="排序且不重复"):
        ledger_module.ConstantLedger(entries=(_module("pkg.b"), _module("pkg.a")))
    ledger = ledger_module.build_ledger(_shared_pair())
    with pytest.raises(errors.ParameterError, match="台账里没有模块"):
        ledger.entry_of("pkg.zzz")
    with pytest.raises(errors.ParameterError, match="台账里没有模块"):
        ledger.package_of("pkg.zzz")


def test_ledger_recomputable_and_consistent_guards() -> None:
    """可复算与"要求同名同值"两条路。"""
    modules = _shared_pair()
    first = ledger_module.build_ledger(modules)
    second = ledger_module.build_ledger(modules)
    assert first.diff_count(second) == 0
    assert first.digest() == second.digest()
    assert ledger_module.ledger_digest(first) == first.digest()
    assert ledger_module.require_reproducible(first, second) is first
    assert ledger_module.require_consistent(first) is first
    other = ledger_module.build_ledger(_conflict_pair())
    assert first.diff_count(other) > 0
    with pytest.raises(errors.LedgerBuildError, match="两份台账差"):
        ledger_module.require_reproducible(first, other)
    with pytest.raises(errors.ConflictError, match="取值不一致"):
        ledger_module.require_consistent(other)


def test_expected_packages_and_ledger_lines() -> None:
    """包的名单与逐行打印（可截断，非法 limit 被拒）。"""
    ledger = ledger_module.build_ledger(_shared_pair())
    assert ledger_module.expected_packages(ledger) == ("a", "b")
    lines = ledger_module.ledger_lines(ledger, limit=1)
    assert lines[0].startswith("常量台账")
    assert len(lines) == 3
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        ledger_module.ledger_lines(ledger, limit=0)


def test_ledger_docs_and_root() -> None:
    """两个小帮手：关系说明与根包名。"""
    assert ledger_module.docs_of_relation(types.RELATION_INCOMPARABLE)
    assert ledger_module.root_package() == types.PACKAGE_NAME
    with pytest.raises(errors.ParameterError, match="未知的同名关系"):
        ledger_module.docs_of_relation("nope")


def test_dataclasses_replace_keeps_ledger_valid() -> None:
    """``dataclasses.replace`` 能造出另一份合法台账（给"可复算"用）。"""
    ledger = ledger_module.build_ledger(_shared_pair())
    replaced = dataclasses.replace(ledger, entries=(ledger.entries[0],))
    assert replaced.module_count == 1
    assert replaced.comparable() != ledger.comparable()


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
    assert verify.check_ledger_covers_all_modules().passed is True
    assert verify.check_ledger_is_reproducible().passed is True
    assert verify.check_conflicts_are_sound().passed is True
    assert verify.check_shared_constants_are_sound().passed is True
    assert verify.check_duplicate_assignments_within_module().passed is True
    assert verify.check_constants_are_numerous().passed is True
    assert verify.check_shared_constants_exist().passed is True


def test_check_coverage_fails_when_missing() -> None:
    """覆盖检查：期望名单里多一项时失败并点名。"""
    ledger = ledger_module.build_ledger(_shared_pair())
    outcome = verify.check_ledger_covers_all_modules(ledger, expected=("pkg.a", "pkg.b", "pkg.ghost"))
    assert outcome.passed is False
    assert "pkg.ghost" in outcome.evidence[1]


def test_check_reproducible_fails_on_different_ledgers() -> None:
    """可复算：两份不同的台账差异大于 0。"""
    first = ledger_module.build_ledger(_shared_pair())
    second = ledger_module.build_ledger(_conflict_pair())
    assert verify.check_ledger_is_reproducible(first, second).passed is False


def test_check_conflicts_soundness_failure_direction() -> None:
    """冲突是否真：注入一组"取值不同、源码片段却相同"的假冲突，判据变红.

    这里让两个小模块的 ``path`` 都指向**同一个真实文件**（该文件里没有 `EPSILON`），
    于是独立复核取到的两条片段都是空串——不足两种，第三条当场变红。
    """
    real = parse_module.PROJECT_ROOT / "smart_research_agent" / "constant_ledger" / "types.py"
    modules = (
        _module("pkg.a", _const("pkg.a", "EPSILON", literal="1e-05"), path=real.as_posix()),
        _module("pkg.b", _const("pkg.b", "EPSILON", literal="1e-08"), path=real.as_posix()),
    )
    outcome = verify.check_conflicts_are_sound(ledger_module.build_ledger(modules))
    assert outcome.passed is False
    assert "不成立" in outcome.evidence[0]


def test_check_shared_soundness_failure_direction() -> None:
    """同名表是否真：把同名表换成一份对不上的，判据变红。"""

    class _WrongShared(ledger_module.ConstantLedger):
        def shared_groups(self):
            return (ledger_module.NameGroup(
                name="SHARED",
                relation=types.RELATION_CONSISTENT,
                members=(_const("pkg.a", "SHARED", literal="1"), _const("pkg.zzz", "SHARED", literal="1")),
            ),)

    modules = _shared_pair()
    ledger = _WrongShared(entries=tuple(sorted(modules, key=lambda entry: entry.module)))
    outcome = verify.check_shared_constants_are_sound(ledger)
    assert outcome.passed is False
    assert "不成立 1" in outcome.evidence[0]


def test_check_duplicate_failure_direction() -> None:
    """重复赋值（上界）的失败方向。"""

    class _Duplicated(ledger_module.ConstantLedger):
        def duplicate_modules(self):
            return ("pkg.a",)

    modules = _shared_pair()
    ledger = _Duplicated(entries=tuple(sorted(modules, key=lambda entry: entry.module)))
    outcome = verify.check_duplicate_assignments_within_module(ledger)
    assert outcome.passed is False
    assert outcome.criterion == types.CRITERION_UPPER_BOUND


def test_check_numerous_and_shared_failure_directions() -> None:
    """常量数下界与同名下界的失败方向。"""
    tiny = ledger_module.build_ledger((_module("pkg.a", _const("pkg.a", "ALPHA")),))
    numerous = verify.check_constants_are_numerous(tiny)
    assert numerous.passed is False
    assert numerous.criterion == types.CRITERION_LOWER_BOUND

    unique_only = ledger_module.build_ledger(
        (
            _module("pkg.a", _const("pkg.a", "ALPHA")),
            _module("pkg.b", _const("pkg.b", "BETA")),
        )
    )
    shared = verify.check_shared_constants_exist(unique_only)
    assert shared.passed is False
    assert shared.criterion == types.CRITERION_LOWER_BOUND


def test_check_all_default_and_injected() -> None:
    """check_all：默认全通过；注入小台账时给出同一份 7 行报告.

    注入样本只有 3 条常量，因此第 ⑥ 条（常量数 >= 500）会**当场地、正确地**变红——
    它的阈值定在"这批文件必然成立"的地方，本来就**不是**为小样本准备的。
    """
    report = verify.check_all()
    assert report.ok is True
    assert [outcome.name for outcome in report.outcomes] == list(types.CONSTANT_LEDGER_PROPERTIES)
    injected = verify.check_all(modules=_shared_pair(), expected=("pkg.a", "pkg.b"))
    assert [outcome.name for outcome in injected.outcomes] == list(types.CONSTANT_LEDGER_PROPERTIES)
    failed = [outcome.name for outcome in injected.outcomes if not outcome.passed]
    assert failed == [types.PROPERTY_CONSTANTS_ARE_NUMEROUS]


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
    """模块表：行数与模块数一致，计数都对，可截断。"""
    ledger = ledger_module.build_ledger(_shared_pair())
    rows = study.module_rows(ledger)
    assert len(rows) == ledger.module_count
    table = {row.module: row for row in rows}
    assert table["pkg.a"].constants == 2
    assert table["pkg.a"].literal == 2
    assert "常量" in rows[0].line()
    assert len(study.module_rows(ledger, limit=1)) == 1


def test_shared_and_conflict_rows() -> None:
    """同名表 / 冲突表：行数与读数一致，可截断。"""
    ledger = ledger_module.build_ledger(_conflict_pair())
    assert [row.name for row in study.conflict_rows(ledger)] == ["EPSILON"]
    assert "个模块" in study.shared_rows(ledger)[0].line()
    assert len(study.shared_rows(ledger, limit=1)) == 1
    assert study.conflict_rows(ledger)[0].to_dict()["pairs"][0][0] == "pkg.a"


def test_property_rows() -> None:
    """性质表：7 行、判据类别齐全、全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert {row.criterion for row in rows} == set(types.CRITERIA)
    assert "读数" in rows[0].line()


def test_note_boundary_and_kind_lines() -> None:
    """笔记 / 边界 / 取值形态 / 关系四类文本行。"""
    assert len(study.note_lines()) == 10
    assert len(study.note_lines(limit=4)) == 4
    assert len(study.boundary_lines()) == len(types.CONSTANT_LEDGER_BOUNDARIES)
    assert len(study.value_kind_lines()) == len(types.VALUE_KINDS)
    assert len(study.relation_lines()) == len(types.RELATIONS)


def test_study_lines_runs_all_four_tables() -> None:
    """四张表一次跑完：四个小标题。"""
    lines = study.study_lines()
    headings = [line for line in lines if line.startswith("== ")]
    assert len(headings) == 4
    assert any("模块表" in line for line in headings)
    assert any("冲突表" in line for line in headings)
    assert any("性质表" in line for line in headings)


def test_study_lines_accepts_injected_inputs() -> None:
    """四张表接收注入的台账（注入一份**没有冲突**的小台账：冲突判据要读真实源码）. """
    ledger = ledger_module.build_ledger(_shared_pair())
    lines = study.study_lines(ledger=ledger, module_limit=1, shared_limit=1, conflict_limit=1)
    assert len([line for line in lines if line.startswith("== ")]) == 4


def test_to_dict_lines() -> None:
    """把带 to_dict 的行折成 JSON 化字段（四张表都能导出）。"""
    ledger = ledger_module.build_ledger(_conflict_pair())
    assert study.to_dict_lines(study.module_rows(ledger, limit=1))[0]["module"]
    assert study.to_dict_lines(study.shared_rows(ledger))[0]["name"] == "EPSILON"
    assert study.to_dict_lines(study.conflict_rows(ledger))[0]["name"] == "EPSILON"
    assert study.to_dict_lines(study.property_rows())[0]["criterion"]


# --------------------------------------------------------------------------- 包


def test_package_all_is_sorted_and_unique() -> None:
    """包的公开名单：字母序、无重复、不含子模块名。"""
    import smart_research_agent.constant_ledger as package

    assert package.__all__ == sorted(package.__all__)
    assert len(package.__all__) == len(set(package.__all__))
    assert set(package.__all__) & {"errors", "types", "parse", "ledger", "verify", "study"} == set()


def test_package_exposes_key_names() -> None:
    """包级命名空间真的导入了关键名字（不是只在 __all__ 里）。"""
    import smart_research_agent.constant_ledger as package

    for name in (
        "build_ledger",
        "scan_all",
        "ConstantLedger",
        "check_all",
        "study_lines",
        "ConstantError",
    ):
        assert name in package.__all__
        assert hasattr(package, name)
