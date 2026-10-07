"""``graduation``：把一次能跑的运行交付成四份能被别人复算的产物（day100 / G2-D1）.

本文件覆盖新包的九个模块：口径表、失败族、清单、剧本、自评、规划、总结、性质与表。
样本来自本包自己的确定性构造（复用 day099 的固定底座）——
**交付真的消费既有子系统**（``capstone`` → ``security`` / ``agent`` / ``retrieval`` /
``evaluation`` / ``observability`` / ``tools`` / ``vectorstore`` / ``llm``），
因此这一课考的就是"四份产物是不是真的从读数折出来的、而且能被复算"。
"""

from __future__ import annotations

import dataclasses
import types as _pytypes

import pytest

from smart_research_agent.capstone.types import (
    ASSEMBLY_STAGES,
    CAPABILITY_INPUT_GUARD,
    CAPABILITY_ORDER,
    CAPABILITY_TOOL_EXECUTION,
)
from smart_research_agent.graduation import (
    assessment,
    demo,
    errors,
    inventory,
    roadmap,
    study,
    summary,
    types,
    verify,
)

# --------------------------------------------------------------------------- 口径表


def test_four_deliverables_are_closed() -> None:
    """4 份交付物的名单、说明、规格三张表逐键对齐，且都有标题 / 说明 / 问题。"""
    assert len(types.DELIVERABLE_ORDER) == 4
    assert set(types.DELIVERABLE_DESCRIPTIONS) == set(types.DELIVERABLE_ORDER)
    assert set(types.DELIVERABLE_SPECS) == set(types.DELIVERABLE_ORDER)
    assert all(spec.title and spec.description and spec.question for spec in types.deliverables())


def test_four_levels_are_closed() -> None:
    """4 级量程：名单、说明、规格逐键对齐，且最高级等于量程最大值。"""
    assert len(types.LEVELS) == 4
    assert set(types.LEVEL_DESCRIPTIONS) == set(types.LEVELS)
    assert set(types.LEVEL_SPECS) == set(types.LEVELS)
    assert types.TOP_LEVEL == max(types.LEVELS)
    assert [spec.level for spec in types.level_specs()] == list(types.LEVELS)


def test_seven_properties_and_three_criteria() -> None:
    """7 条性质：名单与规格表逐键对齐，且三类判据都有性质。"""
    assert len(types.GRADUATION_PROPERTIES) == 7
    assert set(types.PROPERTY_SPECS) == set(types.GRADUATION_PROPERTIES)
    criteria = {spec.criterion for spec in types.property_specs()}
    assert criteria == set(types.CRITERIA)
    assert types.CRITERION_EQUALITY in criteria
    assert types.CRITERION_UPPER_BOUND in criteria
    assert types.CRITERION_LOWER_BOUND in criteria


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.GRADUATION_NOTES) == 10
    assert types.GRADUATION_NOTES_ORDER == tuple(types.GRADUATION_NOTES)
    assert all(types.GRADUATION_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（这一课明确不承诺的事）。"""
    assert len(types.GRADUATION_BOUNDARIES) == 5
    assert all(types.GRADUATION_BOUNDARIES)


def test_milestone_constants() -> None:
    """里程碑三件：100 天、标题、说明都不为空。"""
    assert types.MILESTONE_TOTAL_DAYS == 100
    assert types.MILESTONE_TITLE and types.MILESTONE_DESCRIPTION


def test_require_helpers() -> None:
    """四个 require 帮手：合法值原样返回，非法值当场拒绝。"""
    assert types.require_deliverable(types.DELIVERABLE_DEMO) == types.DELIVERABLE_DEMO
    assert types.require_level(types.LEVEL_DELIVERED) == types.LEVEL_DELIVERED
    assert types.require_positive_threshold("symbol_floor", 8) == 8
    assert types.require_property(types.PROPERTY_DEMO_REPLAYS_IDENTICALLY)
    with pytest.raises(errors.ParameterError, match="未知的交付物"):
        types.require_deliverable("nope")
    with pytest.raises(errors.AssessmentError, match="未知的自评评级"):
        types.require_level(9)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_threshold("symbol_floor", 0)
    with pytest.raises(errors.ParameterError, match="未知的性质"):
        types.require_property("nope")


# --------------------------------------------------------------------------- 记录


def _deliverable_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 DeliverableSpec 关键字参数."""
    base: dict[str, object] = {
        "id": types.DELIVERABLE_DEMO,
        "title": "剧本",
        "description": "说明",
        "question": "问什么",
    }
    base.update(overrides)
    return base


def test_deliverable_spec_validates_fields() -> None:
    """交付物记录的两条护栏与两个渲染方法。"""
    spec = types.DeliverableSpec(**_deliverable_kwargs())
    assert spec.to_dict()["id"] == types.DELIVERABLE_DEMO
    assert spec.line().startswith("demo")
    with pytest.raises(errors.ParameterError, match="未知的交付物"):
        types.DeliverableSpec(**_deliverable_kwargs(id="nope"))
    with pytest.raises(errors.ParameterError, match="都不能为空"):
        types.DeliverableSpec(**_deliverable_kwargs(question=""))


def _level_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 LevelSpec 关键字参数."""
    base: dict[str, object] = {"level": types.LEVEL_DELIVERED, "name": "已交付", "criterion": "三项全过"}
    base.update(overrides)
    return base


def test_level_spec_validates_fields() -> None:
    """量程记录的两条护栏与两个渲染方法。"""
    spec = types.LevelSpec(**_level_kwargs())
    assert spec.to_dict()["level"] == types.LEVEL_DELIVERED
    assert spec.line().startswith("3 已交付")
    with pytest.raises(errors.AssessmentError, match="未知的自评评级"):
        types.LevelSpec(**_level_kwargs(level=7))
    with pytest.raises(errors.AssessmentError, match="名字与证据口径"):
        types.LevelSpec(**_level_kwargs(name=""))


def _property_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 PropertySpec 关键字参数."""
    base: dict[str, object] = {
        "id": types.PROPERTY_DEMO_REPLAYS_IDENTICALLY,
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
    """七个失败族：类表与处置表逐键对齐，且都是 ValueError 的子类。"""
    assert len(errors.FAMILY_OUTCOMES) == 7
    assert set(errors.FAMILY_OUTCOMES) == set(errors._FAMILY_CLASSES)
    for name in errors.FAMILY_OUTCOMES:
        cls = errors._FAMILY_CLASSES[name]
        assert issubclass(cls, errors.GraduationError)
        assert issubclass(cls, ValueError)


def test_returned_and_absent_families_are_declared() -> None:
    """回来的族与缺席的族都是常量，且各自带理由。"""
    assert errors.RETURNED_FAMILY == "VersionError"
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY_REASON


def test_each_family_is_raisable() -> None:
    """每个失败族都能被 raise / except（可读的类名与消息）。"""
    for cls in errors._FAMILY_CLASSES.values():
        with pytest.raises(cls):
            raise cls("boom")


# --------------------------------------------------------------------------- 清单


def test_discover_subpackages_is_sorted_and_unique() -> None:
    """发现全部一级子包：排序、无重复，且含 capstone 与本包。"""
    names = inventory.discover_subpackages()
    assert list(names) == sorted(names)
    assert len(names) == len(set(names))
    assert "capstone" in names
    assert "graduation" in names


def test_resolve_subpackage_reports_presence() -> None:
    """解析一个真实子包：在场、公开名字 > 0、有两处渲染方法。"""
    info = inventory.resolve_subpackage("capstone")
    assert info.exists is True
    assert info.symbol_count > 0
    assert info.module_count > 0
    assert info.has_all is True
    assert "在场" in info.line()
    assert info.to_dict()["name"] == "capstone"


def test_count_public_symbols_has_fallback() -> None:
    """无 ``__all__`` 的包按目录数子模块，并有 vars 兜底。"""
    synthetic = _pytypes.ModuleType("synthetic")
    synthetic.public = 1
    synthetic._private = 2
    assert inventory.count_public_symbols(synthetic) == 1


def test_count_submodules_counts_files() -> None:
    """数一个子包内部的文件数（本包自己有 9 个模块文件）。"""
    assert inventory.count_submodules("graduation") == 9


def test_count_submodules_handles_plain_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """被数的名字其实是一个普通模块（没有 __path__）时返回 0。"""
    monkeypatch.setattr(
        inventory.importlib, "import_module", lambda name: _pytypes.ModuleType(name)
    )
    assert inventory.count_submodules("whatever") == 0


def test_resolve_subpackage_rejects_bad_name() -> None:
    """空名 / 以 _ 开头的名字当场拒绝。"""
    with pytest.raises(errors.ParameterError, match="可交付的名字"):
        inventory.resolve_subpackage("_private")
    with pytest.raises(errors.ParameterError, match="可交付的名字"):
        inventory.resolve_subpackage("")


def test_resolve_subpackage_records_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """子包 import 失败被**记录**成 exists=False，而不是把清单打断。"""
    real = inventory.importlib.import_module

    def fake(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name.endswith(".capstone"):
            raise ImportError("boom")
        return real(name, package)

    monkeypatch.setattr(inventory.importlib, "import_module", fake)
    info = inventory.resolve_subpackage("capstone")
    assert info.exists is False
    assert info.symbol_count == 0
    assert info.module_count == 0
    assert "boom" in info.error
    assert "缺失" in info.line()


def test_discover_requires_a_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """被扫描的对象不是包（没有 __path__）时抛 InventoryError。"""
    monkeypatch.setattr(
        inventory.importlib, "import_module", lambda name: _pytypes.ModuleType(name)
    )
    with pytest.raises(errors.InventoryError, match="不是一个包"):
        inventory.discover_subpackages()


def test_subpackage_info_validates_fields() -> None:
    """SubpackageInfo 的三条护栏。"""
    ok = inventory.SubpackageInfo(
        name="capstone", exists=True, has_all=True, symbol_count=3, module_count=2
    )
    assert ok.to_dict()["exists"] is True
    with pytest.raises(errors.ParameterError, match="可交付的名字"):
        inventory.SubpackageInfo(
            name="_x", exists=True, has_all=True, symbol_count=0, module_count=0
        )
    with pytest.raises(errors.ParameterError, match="不能为负"):
        inventory.SubpackageInfo(
            name="capstone", exists=True, has_all=True, symbol_count=-1, module_count=0
        )
    with pytest.raises(errors.InventoryError, match="没有留错误信息"):
        inventory.SubpackageInfo(
            name="capstone", exists=False, has_all=False, symbol_count=0, module_count=0
        )


def test_inventory_report_closure_checks() -> None:
    """清单的三条闭合检查：重复、漏项、多出。"""
    report = inventory.build_inventory()
    assert report.clean is True
    assert report.missing == ()
    assert set(report.to_dict()["discovered"]) == set(report.discovered)
    with pytest.raises(errors.InventoryError, match="重复的子包"):
        dataclasses.replace(report, rows=report.rows + (report.rows[0],))
    with pytest.raises(errors.InventoryError, match="记录集合与发现集合不一致"):
        dataclasses.replace(report, discovered=report.discovered + ("nope",))


def test_inventory_report_views_and_lines() -> None:
    """清单的只读视图与打印：计数 / 取单项 / 文本行。"""
    report = inventory.build_inventory()
    assert len(report.present) == len(report.rows)
    assert report.total_symbols > 0
    assert report.total_modules > 0
    assert report.row_of("capstone").name == "capstone"
    assert "子包" in report.line()
    lines = inventory.inventory_lines(report)
    assert any("清单表" in line for line in lines)
    with pytest.raises(errors.ParameterError, match="清单里没有子包"):
        report.row_of("nope")


def test_require_inventory_clean() -> None:
    """'宁可报错不可漏算'那条路：干净时返回，有缺失时抛。"""
    report = inventory.build_inventory()
    assert inventory.require_inventory_clean(report) is report
    broken = dataclasses.replace(
        report,
        rows=(
            inventory.SubpackageInfo(
                name=report.rows[0].name,
                exists=False,
                has_all=False,
                symbol_count=0,
                module_count=0,
                error="boom",
            ),
        )
        + report.rows[1:],
    )
    with pytest.raises(errors.InventoryError, match="清单不干净"):
        inventory.require_inventory_clean(broken)


# --------------------------------------------------------------------------- 剧本


def test_build_transcript_is_deterministic() -> None:
    """剧本来自一次真实运行：九段、14 行、摘要 16 位。"""
    transcript = demo.build_transcript()
    assert len(transcript.stage_lines) == len(ASSEMBLY_STAGES)
    assert len(transcript.lines()) == len(ASSEMBLY_STAGES) + 5
    assert len(transcript.digest) == demo.TRANSCRIPT_DIGEST_LENGTH
    assert transcript.question
    assert transcript.to_dict()["stages"] == len(ASSEMBLY_STAGES)


def test_transcript_diff_and_identity() -> None:
    """两份剧本：逐位相同 / 差异项数 / 判定与渲染。"""
    first = demo.build_transcript()
    second = demo.build_transcript()
    assert first.is_identical_to(second) is True
    assert first.diff_count(second) == 0
    changed = dataclasses.replace(first, summary_line="换个汇总")
    assert first.is_identical_to(changed) is False
    assert first.diff_count(changed) == 1


def test_replay_is_identical() -> None:
    """重放：两份剧本摘要相同、require_identical 通过、有渲染方法。"""
    report = demo.replay()
    assert report.identical is True
    assert report.diff_count == 0
    assert report.first.digest == report.second.digest
    assert "重放" in report.line()
    assert report.to_dict()["identical"] is True
    assert report.require_identical() is None


def test_replay_requires_identity() -> None:
    """重放不一致时 require_identical 抛 DemoError。"""
    first = demo.build_transcript()
    changed = dataclasses.replace(first, question="另一个问题")
    report = demo.ReplayReport(first=first, second=changed)
    assert report.identical is False
    with pytest.raises(errors.DemoError, match="重放不一致"):
        report.require_identical()


def test_demo_transcript_validates_fields() -> None:
    """剧本记录的三条护栏。"""
    good = demo.build_transcript()
    with pytest.raises(errors.ParameterError, match="不能为空"):
        dataclasses.replace(good, question="")
    with pytest.raises(errors.DemoError, match="应当是"):
        dataclasses.replace(good, stage_lines=good.stage_lines[:-1])
    with pytest.raises(errors.ParameterError, match="汇总行与归档行"):
        dataclasses.replace(good, summary_line="")
    with pytest.raises(errors.DemoError, match="必须带摘要"):
        dataclasses.replace(good, digest="")


def test_write_transcript(tmp_path: pytest.TempPathFactory) -> None:
    """剧本落盘到给定目录（不覆盖仓库里既有的任何文件）。"""
    path = demo.write_transcript(tmp_path)
    assert path.name == demo.TRANSCRIPT_FILENAME
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert types.MILESTONE_TITLE in text


def test_require_digest() -> None:
    """摘要校对：空期望放行，不符时抛 NumericError。"""
    transcript = demo.build_transcript()
    assert demo.require_digest(transcript, expected="") is transcript
    assert demo.require_digest(transcript, expected=transcript.digest) is transcript
    with pytest.raises(errors.NumericError, match="不一致"):
        demo.require_digest(transcript, expected="deadbeefdeadbeef")


# --------------------------------------------------------------------------- 自评


def test_level_for_four_branches() -> None:
    """评级函数四档：未建 / 已建 / 可用 / 已交付。"""
    assert (
        assessment.level_for(owners_present=1, owners_total=2, symbols=100, stage_ok=True)
        == types.LEVEL_NOT_BUILT
    )
    assert (
        assessment.level_for(owners_present=2, owners_total=2, symbols=3, stage_ok=True)
        == types.LEVEL_BUILT
    )
    assert (
        assessment.level_for(owners_present=2, owners_total=2, symbols=100, stage_ok=False)
        == types.LEVEL_USABLE
    )
    assert (
        assessment.level_for(owners_present=2, owners_total=2, symbols=100, stage_ok=True)
        == types.LEVEL_DELIVERED
    )


def test_level_for_guards() -> None:
    """评级函数的四条入口护栏。"""
    with pytest.raises(errors.ParameterError, match="至少要有一个承担子包"):
        assessment.level_for(owners_present=0, owners_total=0, symbols=1, stage_ok=True)
    with pytest.raises(errors.ParameterError, match="必须落在"):
        assessment.level_for(owners_present=3, owners_total=2, symbols=1, stage_ok=True)
    with pytest.raises(errors.ParameterError, match="不能为负"):
        assessment.level_for(owners_present=1, owners_total=1, symbols=-1, stage_ok=True)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        assessment.level_for(
            owners_present=1, owners_total=1, symbols=1, stage_ok=True, symbol_floor=0
        )


def test_assess_default_is_consistent() -> None:
    """默认自评：8 行、评级在量程内、每行都有证据、最高已交付。"""
    result = assessment.assess()
    assert len(result.rows) == len(CAPABILITY_ORDER)
    assert result.max_level == types.TOP_LEVEL
    assert result.min_level >= types.LEVEL_NOT_BUILT
    assert result.min_evidence >= 1
    assert result.score_of(CAPABILITY_TOOL_EXECUTION).level == types.LEVEL_BUILT
    assert CAPABILITY_TOOL_EXECUTION in result.below
    assert "自评" in result.line()
    assert result.to_dict()["rows"][0]["level_name"]


def test_assessment_views_and_lines() -> None:
    """自评的只读视图与渲染：已交付清单 / 取单项 / 未知能力被拒。"""
    result = assessment.assess()
    assert len(result.delivered) == len(CAPABILITY_ORDER) - len(result.below)
    assert all(row.evidence for row in result.rows)
    assert "L3" in result.rows[0].line()
    assert result.rows[0].to_dict()["capability"]
    with pytest.raises(errors.ParameterError, match="自评里没有能力"):
        result.score_of("nope")


def test_capability_score_guards() -> None:
    """自评记录的三条闭合检查：未知能力 / 没有证据 / 评级与证据不一致。"""
    good = assessment.assess().rows[0]
    assert good.delivered is True
    with pytest.raises(errors.AssessmentError, match="未知的能力"):
        dataclasses.replace(good, capability="nope")
    with pytest.raises(errors.AssessmentError, match="没有任何证据"):
        dataclasses.replace(good, evidence=())
    with pytest.raises(errors.AssessmentError, match="与证据重算出的"):
        dataclasses.replace(good, owners_present=0, level=types.LEVEL_DELIVERED)
    with pytest.raises(errors.AssessmentError, match="未知的自评评级"):
        dataclasses.replace(good, level=9)


def test_assessment_closure_checks() -> None:
    """自评的闭合检查：重复能力 / 能力集合对不上。"""
    good = assessment.assess()
    with pytest.raises(errors.AssessmentError, match="重复的能力"):
        dataclasses.replace(good, rows=good.rows + (good.rows[0],))
    with pytest.raises(errors.AssessmentError, match="能力集合与 CAPABILITY_ORDER"):
        dataclasses.replace(good, rows=good.rows[:-1])


def test_assess_accepts_injected_manifest() -> None:
    """注入清单：某一项能力的承担子包被记成"不在场" ⇒ 该项掉到 L0。"""
    from smart_research_agent.capstone import manifest as manifest_module

    base = manifest_module.build_manifest()
    broken = dataclasses.replace(
        base,
        coverages=tuple(
            manifest_module.CapabilityCoverage(
                capability=coverage.capability,
                modules=tuple(
                    manifest_module.ModuleInfo(
                        name=module.name,
                        exists=False,
                        symbol_count=0,
                        has_all=False,
                        error="boom",
                    )
                    for module in coverage.modules
                ),
            )
            if coverage.capability == CAPABILITY_INPUT_GUARD
            else coverage
            for coverage in base.coverages
        ),
    )
    result = assessment.assess(broken)
    score = result.score_of(CAPABILITY_INPUT_GUARD)
    assert score.level == types.LEVEL_NOT_BUILT
    assert score.owners_present == 0
    assert any("缺失的承担子包" in item for item in score.evidence)


def test_require_delivered() -> None:
    """严格那条路：默认状态有一项在 L1 ⇒ 抛；构造一份全 L3 的 ⇒ 放行。"""
    default = assessment.assess()
    with pytest.raises(errors.AssessmentError, match="没有到最高级"):
        assessment.require_delivered(default)
    all_top = dataclasses.replace(
        default,
        rows=tuple(
            dataclasses.replace(
                row,
                level=types.LEVEL_DELIVERED,
                owners_present=row.owners_total,
                symbols=max(row.symbols, types.SYMBOL_FLOOR),
                stage_ok=True,
            )
            for row in default.rows
        ),
    )
    assert assessment.require_delivered(all_top) is all_top


# --------------------------------------------------------------------------- 规划


def test_gap_kinds_are_closed() -> None:
    """两种缺口：名单与说明逐键对齐，未知类型当场拒绝。"""
    assert len(roadmap.GAP_KINDS) == 2
    assert set(roadmap.GAP_DESCRIPTIONS) == set(roadmap.GAP_KINDS)
    assert roadmap.require_gap_kind(roadmap.GAP_UNCLAIMED) == roadmap.GAP_UNCLAIMED
    with pytest.raises(errors.ParameterError, match="未知的缺口类型"):
        roadmap.require_gap_kind("nope")


def test_gaps_have_two_kinds() -> None:
    """缺口来自两处真实读数：无人认领的子包 + 没到最高级的能力。"""
    found = roadmap.gaps()
    keys = [key for key, _ in found]
    assert len(keys) == len(set(keys))
    assert "api" in keys
    assert CAPABILITY_TOOL_EXECUTION in keys
    kinds = {kind for _, kind in found}
    assert kinds == set(roadmap.GAP_KINDS)


def test_plan_covers_every_gap() -> None:
    """规划与缺口逐键对上（不重不漏），且两类缺口都在里面。"""
    result = roadmap.plan()
    gap_keys = tuple(key for key, _ in roadmap.gaps())
    assert result.keys == gap_keys
    assert result.missing(gap_keys) == ()
    assert result.ignored(gap_keys) == ()
    assert result.size == len(gap_keys)
    assert set(result.kinds) == set(roadmap.GAP_KINDS)
    assert "规划" in result.line()
    assert result.to_dict()["size"] == result.size


def test_roadmap_item_guards() -> None:
    """规划项的三条护栏与两个渲染方法。"""
    item = roadmap.RoadmapItem(
        key="api", kind=roadmap.GAP_UNCLAIMED, reason="理由", action="下一步"
    )
    assert item.to_dict()["kind"] == roadmap.GAP_UNCLAIMED
    assert item.line().startswith("[unclaimed_package")
    with pytest.raises(errors.ParameterError, match="未知的缺口类型"):
        roadmap.RoadmapItem(key="api", kind="nope", reason="r", action="a")
    with pytest.raises(errors.ParameterError, match="缺口键不能为空"):
        roadmap.RoadmapItem(key="", kind=roadmap.GAP_UNCLAIMED, reason="r", action="a")
    with pytest.raises(errors.RoadmapError, match="都不能为空"):
        roadmap.RoadmapItem(key="api", kind=roadmap.GAP_UNCLAIMED, reason="r", action="")


def test_roadmap_closure_and_complete() -> None:
    """规划的重复检查 + require_complete 的两个方向。"""
    good = roadmap.plan()
    assert roadmap.require_complete(good) is good
    with pytest.raises(errors.RoadmapError, match="重复的缺口键"):
        dataclasses.replace(good, items=good.items + (good.items[0],))
    empty = roadmap.Roadmap(items=())
    assert empty.missing(tuple(good.keys)) == good.keys
    with pytest.raises(errors.RoadmapError, match="有缺口没有规划"):
        roadmap.require_complete(empty)
    bogus = roadmap.Roadmap(
        items=(
            roadmap.RoadmapItem(
                key="nope", kind=roadmap.GAP_UNCLAIMED, reason="r", action="a"
            ),
        )
    )
    with pytest.raises(errors.RoadmapError, match="有规划没有对应缺口"):
        roadmap.require_complete(bogus)


# --------------------------------------------------------------------------- 总结


def test_build_summary_reads_inventory() -> None:
    """总结的前四个计数都来自清单，后四个来自常量。"""
    report = inventory.build_inventory()
    result = summary.build_summary(report)
    assert result.days == types.MILESTONE_TOTAL_DAYS
    assert result.subpackages == len(report.rows)
    assert result.modules == report.total_modules
    assert result.symbols == report.total_symbols
    assert result.capabilities == len(CAPABILITY_ORDER)
    assert result.stages == len(ASSEMBLY_STAGES)
    assert result.properties == len(types.GRADUATION_PROPERTIES)
    assert result.deliverables == len(types.DELIVERABLE_ORDER)


def test_summary_matches_inventory() -> None:
    """总结与清单的三个可数计数一致；跑偏时差异项数变大。"""
    report = inventory.build_inventory()
    result = summary.build_summary(report)
    assert summary.summary_matches_inventory(result, report) == 0
    skewed = dataclasses.replace(result, subpackages=result.subpackages + 1)
    assert summary.summary_matches_inventory(skewed, report) == 1
    assert "总结" in result.line()
    assert result.to_dict()["days"] == types.MILESTONE_TOTAL_DAYS


def test_course_summary_guards() -> None:
    """总结的两条护栏：天数与常量不符 / 计数为 0。"""
    report = inventory.build_inventory()
    result = summary.build_summary(report)
    with pytest.raises(errors.MilestoneError, match="与里程碑常量"):
        dataclasses.replace(result, days=98)
    with pytest.raises(errors.NumericError, match="必须 >= 1"):
        dataclasses.replace(result, symbols=0)


# --------------------------------------------------------------------------- 性质


def test_crosscheck_three_criteria() -> None:
    """三类判据各自的判定与渲染。"""
    equal = verify.CrossCheck(
        name="相等", left="a", right="b", reading=2.0, expected=2.0, exact=True
    )
    assert equal.criterion == types.CRITERION_EQUALITY
    assert equal.passed is True
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
    assert "读数" in equal.line()
    assert equal.to_dict()["criterion"] == types.CRITERION_EQUALITY


def test_crosscheck_failures_and_guards() -> None:
    """判据不满足时的读数 + 四条构造护栏。"""
    failed = verify.CrossCheck(
        name="下界", left="a", right="b", reading=0.0, expected=1.0, lower_bound=1.0
    )
    assert failed.passed is False
    assert "不满足" in failed.line()
    tolerant = verify.CrossCheck(
        name="容差相等", left="a", right="b", reading=1.0, expected=1.0 + 1e-15, exact=False
    )
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
        verify.CrossCheck(
            name="n", left="a", right="b", reading=0.0, expected=0.0, upper_bound=-1.0
        )
    with pytest.raises(errors.NumericError, match="下界必须是有限数"):
        verify.CrossCheck(
            name="n", left="a", right="b", reading=0.0, expected=0.0, lower_bound=float("inf")
        )


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
    assert not_applicable.criterion == types.CRITERION_EQUALITY
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
    with pytest.raises(errors.AssessmentError, match="未全部通过"):
        bad.require_ok()


def test_render_deliverables_and_missing() -> None:
    """四份交付物的正文：键与名单一致、行数非空、空白检查。"""
    bodies = verify.render_deliverables()
    assert set(bodies) == set(types.DELIVERABLE_ORDER)
    assert all(bodies[name] for name in types.DELIVERABLE_ORDER)
    assert verify.missing_deliverables(bodies) == ()
    trimmed = {**bodies, types.DELIVERABLE_DEMO: ()}
    assert verify.missing_deliverables(trimmed) == (types.DELIVERABLE_DEMO,)


def test_check_functions_default() -> None:
    """七条性质函数在默认输入上全部通过（读数方向正确）。"""
    assert verify.check_demo_replays_identically().passed is True
    assert verify.check_deliverables_are_complete().passed is True
    assert verify.check_inventory_accounts_for_all().passed is True
    assert verify.check_roadmap_covers_every_gap().passed is True
    assert verify.check_assessment_within_scale().passed is True
    assert verify.check_assessment_has_evidence().passed is True
    assert verify.check_summary_matches_inventory().passed is True


def test_check_functions_with_injected_inputs() -> None:
    """七条性质函数在注入输入上给出同一批结论。"""
    report = inventory.build_inventory()
    asm = assessment.assess()
    rm = roadmap.plan()
    sm = summary.build_summary(report)
    bodies = verify.render_deliverables(inventory=report, summary=sm, assessment=asm, roadmap=rm)
    assert verify.check_demo_replays_identically(demo.replay()).passed is True
    assert verify.check_deliverables_are_complete(bodies).passed is True
    assert verify.check_inventory_accounts_for_all(report).passed is True
    assert verify.check_roadmap_covers_every_gap(rm).passed is True
    assert verify.check_assessment_within_scale(asm).passed is True
    assert verify.check_assessment_has_evidence(asm).passed is True
    assert verify.check_summary_matches_inventory(sm, report).passed is True


def test_check_demo_fails_when_replay_differs() -> None:
    """相等判据的方向：重放不一致时失败。"""
    first = demo.build_transcript()
    changed = dataclasses.replace(first, summary_line="换个汇总")
    outcome = verify.check_demo_replays_identically(demo.ReplayReport(first=first, second=changed))
    assert outcome.passed is False


def test_check_deliverables_fails_when_blank() -> None:
    """交付物完整性：删掉一份后失败并点名。"""
    bodies = {**verify.render_deliverables(), types.DELIVERABLE_ROADMAP: ()}
    outcome = verify.check_deliverables_are_complete(bodies)
    assert outcome.passed is False
    assert types.DELIVERABLE_ROADMAP in outcome.evidence[1]


def test_check_inventory_fails_on_missing_row() -> None:
    """清单闭合：有子包解析不到时失败。"""
    report = inventory.build_inventory()
    broken = dataclasses.replace(
        report,
        rows=(
            inventory.SubpackageInfo(
                name=report.rows[0].name,
                exists=False,
                has_all=False,
                symbol_count=0,
                module_count=0,
                error="boom",
            ),
        )
        + report.rows[1:],
    )
    outcome = verify.check_inventory_accounts_for_all(broken)
    assert outcome.passed is False


def test_check_roadmap_fails_on_uncovered_gap() -> None:
    """规划覆盖：漏掉一个缺口后失败。"""
    outcome = verify.check_roadmap_covers_every_gap(roadmap.Roadmap(items=()))
    assert outcome.passed is False


def test_check_assessment_scale_and_evidence_directions() -> None:
    """上界 / 下界判据的方向：越界与没证据都失败（用鸭子类型的替身注入）。"""

    @dataclasses.dataclass(frozen=True)
    class _Stub:
        max_level: int
        min_evidence: int

        def line(self) -> str:
            return "stub"

    over = verify.check_assessment_within_scale(_Stub(max_level=9, min_evidence=3))  # type: ignore[arg-type]
    assert over.passed is False
    assert over.criterion == types.CRITERION_UPPER_BOUND
    empty = verify.check_assessment_has_evidence(_Stub(max_level=3, min_evidence=0))  # type: ignore[arg-type]
    assert empty.passed is False
    assert empty.criterion == types.CRITERION_LOWER_BOUND


def test_check_summary_fails_when_skewed() -> None:
    """总结对账：计数跑偏后失败。"""
    report = inventory.build_inventory()
    skewed = dataclasses.replace(summary.build_summary(report), symbols=1)
    outcome = verify.check_summary_matches_inventory(skewed, report)
    assert outcome.passed is False


def test_check_all_accepts_injected_inputs() -> None:
    """check_all 接收注入的输入并给出同一份 7 行报告。"""
    report = inventory.build_inventory()
    report_object = verify.check_all(inventory=report, summary=summary.build_summary(report))
    names = [outcome.name for outcome in report_object.outcomes]
    assert names == list(types.GRADUATION_PROPERTIES)
    assert report_object.ok is True


def test_require_ok() -> None:
    """require_ok：默认放行，注入一份失败的报告时抛 AssessmentError。"""
    assert verify.require_ok() is not None
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="x", applicable=True, passed=False),)
    )
    with pytest.raises(errors.AssessmentError, match="未全部通过"):
        verify.require_ok(bad)


# --------------------------------------------------------------------------- 表


def test_deliverable_rows() -> None:
    """交付物表：4 行、每行带正文行数。"""
    rows = study.deliverable_rows()
    assert len(rows) == len(types.DELIVERABLE_ORDER)
    assert all(row.line_count > 0 for row in rows)
    assert any("演示剧本" in row.line() for row in rows)


def test_subpackage_rows() -> None:
    """清单表：与发现数一致、全部在场。"""
    rows = study.subpackage_rows()
    assert len(rows) == len(inventory.discover_subpackages())
    assert all(row.exists for row in rows)
    assert "在场" in rows[0].line()


def test_score_rows() -> None:
    """自评表：8 行、评级都在量程内。"""
    rows = study.score_rows()
    assert len(rows) == len(CAPABILITY_ORDER)
    assert all(row.level in types.LEVELS for row in rows)
    assert "L" in rows[0].line()


def test_roadmap_rows() -> None:
    """规划表：行数与缺口数一致。"""
    rows = study.roadmap_rows()
    assert len(rows) == len(roadmap.gaps())
    assert any(row.kind == roadmap.GAP_UNCLAIMED for row in rows)
    assert "下一步" in rows[0].line()


def test_property_rows() -> None:
    """性质表：7 行、判据类别齐全、全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert {row.criterion for row in rows} == set(types.CRITERIA)
    assert "读数" in rows[0].line()


def test_note_boundary_and_level_lines() -> None:
    """笔记 / 边界 / 量程三类文本行。"""
    assert len(study.note_lines()) == 10
    assert len(study.note_lines(limit=3)) == 3
    assert len(study.boundary_lines()) == len(types.GRADUATION_BOUNDARIES)
    assert len(study.level_lines()) == len(types.LEVELS)


def test_study_lines_runs_all_five_tables() -> None:
    """五张表一次跑完：五个小标题、含重放与总结两行。"""
    lines = study.study_lines()
    headings = [line for line in lines if line.startswith("== ")]
    assert len(headings) == 5
    assert any("交付物表" in line for line in headings)
    assert any("性质表" in line for line in headings)
    assert any("重放：" in line for line in lines)
    assert any("总结：" in line for line in lines)


def test_study_lines_accepts_injected_inputs() -> None:
    """五张表接收注入的输入。"""
    report = inventory.build_inventory()
    lines = study.study_lines(
        inventory=report,
        summary=summary.build_summary(report),
    )
    assert len([line for line in lines if line.startswith("== ")]) == 5


def test_to_dict_lines() -> None:
    """把带 to_dict 的行折成 JSON 化字段（五张表都能导出）。"""
    payload = study.to_dict_lines(study.roadmap_rows())
    assert payload
    assert "key" in payload[0]
    assert study.to_dict_lines(study.deliverable_rows())[0]["deliverable"]
    assert study.to_dict_lines(study.subpackage_rows())[0]["name"]
    assert study.to_dict_lines(study.score_rows())[0]["capability"]
    assert study.to_dict_lines(study.property_rows())[0]["criterion"]


# --------------------------------------------------------------------------- 包


def test_package_all_is_sorted_and_unique() -> None:
    """包的公开名单：字母序、无重复、不含子模块名。"""
    import smart_research_agent.graduation as graduation_pkg

    assert graduation_pkg.__all__ == sorted(graduation_pkg.__all__)
    assert len(graduation_pkg.__all__) == len(set(graduation_pkg.__all__))
    assert set(graduation_pkg.__all__) & {
        "errors",
        "types",
        "inventory",
        "demo",
        "assessment",
        "roadmap",
        "summary",
        "verify",
        "study",
    } == set()


def test_package_exposes_key_names() -> None:
    """包级命名空间真的导入了关键名字（不是只在 __all__ 里）。"""
    import smart_research_agent.graduation as graduation_pkg

    for name in (
        "assess",
        "build_inventory",
        "build_transcript",
        "plan",
        "build_summary",
        "check_all",
        "study_lines",
        "GraduationError",
    ):
        assert name in graduation_pkg.__all__
        assert hasattr(graduation_pkg, name)
