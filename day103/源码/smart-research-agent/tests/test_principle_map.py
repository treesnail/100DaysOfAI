"""``principle_map``：把底层原理串成一张可校验的图（day088 / M7-D12）.

本文件覆盖新包的九个模块：口径表、失败族、命题、探针、图、提纲、性质与表。
样本来自本包自己的确定性构造（LCG + 写死输入）——**跨天对账真的调用既有包**
（``transformer_core`` / ``inference_optim`` / ``math_foundations`` / ``vectorstore``），
因此这一课考的就是"这些包被真的接上了"。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from smart_research_agent.principle_map import (
    claims,
    errors,
    graph,
    outline,
    reconcile,
    study,
    types,
    verify,
)

# --------------------------------------------------------------------------- 口径表


def test_four_layers_are_closed() -> None:
    """四个层的名单、说明、次序三张表逐键对齐."""
    assert types.LAYERS == ("math", "attention", "representation", "inference")
    assert set(types.LAYER_DESCRIPTIONS) == set(types.LAYERS)
    assert types.LAYER_ORDER == types.LAYERS
    assert all(types.LAYER_DESCRIPTIONS.values())


def test_six_applications_are_closed() -> None:
    """六个应用的名单与说明逐键对齐."""
    assert set(types.APPLICATIONS) == {
        "agent_reasoning",
        "rag_retrieval",
        "rag_rerank",
        "generation",
        "serving",
        "explainability",
    }
    assert set(types.APPLICATION_DESCRIPTIONS) == set(types.APPLICATIONS)
    assert all(types.APPLICATION_DESCRIPTIONS.values())


def test_twelve_principles_are_declared() -> None:
    """十二条原理的 id 集合恰好是定义表里的那十二个."""
    assert len(types.PRINCIPLES) == 12
    assert set(types.PRINCIPLES) == {item.id for item in claims.PRINCIPLE_DEFS}


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单、说明、"失败意味着什么"三张表逐键对齐."""
    assert len(types.PRINCIPLE_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.PRINCIPLE_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.PRINCIPLE_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致."""
    assert len(types.PRINCIPLE_NOTES) == 10
    assert types.PRINCIPLE_NOTES_ORDER == tuple(types.PRINCIPLE_NOTES)
    assert all(types.PRINCIPLE_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）."""
    assert len(types.PRINCIPLE_BOUNDARIES) == 5
    assert all(types.PRINCIPLE_BOUNDARIES)


def test_require_helpers_reject_unknown_names() -> None:
    """三个 require 帮手的护栏：未知层 / 未知应用 / 未知原理 id."""
    assert types.require_layer("math") == "math"
    assert types.require_application("serving") == "serving"
    assert types.require_principle_id(claims.PRINCIPLE_DEFS[0].id)
    with pytest.raises(errors.ParameterError, match="未知的层"):
        types.require_layer("atenntion")
    with pytest.raises(errors.ParameterError, match="未知的应用"):
        types.require_application("search")
    with pytest.raises(errors.ReferenceError, match="未知的原理"):
        types.require_principle_id("no_such_principle")


# --------------------------------------------------------------------------- 记录


def test_principle_validates_its_fields() -> None:
    """原理的六条护栏：空 id、未知层、未知应用、artifact 无点号、空探针、空来源."""
    good = dict(
        id="x",
        layer=types.LAYER_MATH,
        statement="s",
        artifact="pkg.func",
        probe="p",
        application=types.APP_SERVING,
        source_day="day001",
    )
    item = types.Principle(**good)
    assert item.to_dict()["layer"] == types.LAYER_MATH
    assert item.line().startswith("[math] x")
    with pytest.raises(errors.ParameterError, match="id 不能为空"):
        types.Principle(**{**good, "id": ""})
    with pytest.raises(errors.ParameterError, match="未知的层"):
        types.Principle(**{**good, "layer": "atenntion"})
    with pytest.raises(errors.ParameterError, match="未知的应用"):
        types.Principle(**{**good, "application": "search"})
    with pytest.raises(errors.ParameterError, match="模块.函数"):
        types.Principle(**{**good, "artifact": "justaname"})
    with pytest.raises(errors.ParameterError, match="没有探针"):
        types.Principle(**{**good, "probe": ""})
    with pytest.raises(errors.ParameterError, match="没有来源天"):
        types.Principle(**{**good, "source_day": ""})


def test_application_record_and_supported_flag() -> None:
    """应用的记录：有原理才叫被支撑，``with_principles`` 只换边不换别的."""
    empty = types.Application(id=types.APP_SERVING, description="d")
    assert empty.supported is False
    filled = empty.with_principles(("a", "b"))
    assert filled.supported is True
    assert filled.line().startswith("serving | 2 条原理")
    assert filled.to_dict()["principles"] == ["a", "b"]
    with pytest.raises(errors.ParameterError, match="未知的应用"):
        types.Application(id="search", description="d")
    with pytest.raises(errors.ParameterError, match="必须有说明"):
        types.Application(id=types.APP_SERVING, description="")


def test_evidence_validation_and_two_judges() -> None:
    """证据的两类判据（相等 / 上界）与三条护栏（空 id、非有限、负上界）."""
    exact = types.Evidence(principle="p", reading=1.0, expected=1.0)
    assert exact.passed is True
    assert "读数 1" in exact.line()
    tolerance = types.Evidence(principle="p", reading=1.0 + 1e-13, expected=1.0, exact=False)
    assert tolerance.passed is True
    bounded = types.Evidence(principle="p", reading=0.09, expected=0.1, upper_bound=0.1)
    assert bounded.passed is True
    assert "≤ 上界" in bounded.line()
    too_big = types.Evidence(principle="p", reading=0.11, expected=0.1, upper_bound=0.1)
    assert too_big.passed is False
    assert too_big.to_dict()["passed"] is False
    with pytest.raises(errors.ParameterError, match="不能为空"):
        types.Evidence(principle="", reading=0.0, expected=0.0)
    with pytest.raises(errors.NumericError, match="必须有限"):
        types.Evidence(principle="p", reading=float("nan"), expected=0.0)
    with pytest.raises(errors.NumericError, match="上界"):
        types.Evidence(principle="p", reading=0.0, expected=0.0, upper_bound=-1.0)


def test_talk_section_validation() -> None:
    """提纲一节的五条护栏：节号、层、要点数、空标题/demo、负时长."""
    good = dict(
        index=1,
        title="t",
        bullets=("a", "b", "c"),
        demo="python scripts/x.py",
        minutes=5,
        layer=types.LAYER_MATH,
    )
    section = types.TalkSection(**good)
    assert section.line().startswith("第 1 节 [math]")
    assert section.to_dict()["minutes"] == 5
    with pytest.raises(errors.ParameterError, match="节号"):
        types.TalkSection(**{**good, "index": 0})
    with pytest.raises(errors.ParameterError, match="未知的层"):
        types.TalkSection(**{**good, "layer": "atenntion"})
    with pytest.raises(errors.ShapeError, match="恰好 3 条"):
        types.TalkSection(**{**good, "bullets": ("a",)})
    with pytest.raises(errors.ParameterError, match="标题与 demo"):
        types.TalkSection(**{**good, "title": ""})
    with pytest.raises(errors.ParameterError, match="时长"):
        types.TalkSection(**{**good, "minutes": 0})


def test_coverage_report_validation_and_views() -> None:
    """覆盖报告：键必须与两层名单一致，``unsupported`` 与 ``line`` 可读."""
    per_layer = {layer: 3 for layer in types.LAYERS}
    per_app = {app: 2 for app in types.APPLICATIONS}
    per_app[types.APP_SERVING] = 0
    report = types.CoverageReport(per_layer=per_layer, per_application=per_app, complete=False)
    assert report.unsupported == (types.APP_SERVING,)
    assert "完整 False" in report.line()
    assert report.to_dict()["unsupported"] == [types.APP_SERVING]
    with pytest.raises(errors.ShapeError, match="按层计数"):
        types.CoverageReport(per_layer={"math": 1}, per_application=per_app, complete=False)
    with pytest.raises(errors.ShapeError, match="按应用计数"):
        types.CoverageReport(per_layer=per_layer, per_application={}, complete=False)


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_closed() -> None:
    """七个族的两张表逐键对齐."""
    assert len(errors.FAMILY_OUTCOMES) == 7
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "ClaimError",
        "ReferenceError",
        "CoverageError",
        "OrderError",
    }
    assert all(errors.FAMILY_OUTCOMES.values())


def test_family_inheritance_holds() -> None:
    """继承关系：本层继承 day075 的三族，四个新族都是 BridgeError 的一种."""
    from smart_research_agent.transformer_core.errors import (
        NumericError as CoreNumeric,
    )
    from smart_research_agent.transformer_core.errors import (
        ParameterError as CoreParameter,
    )
    from smart_research_agent.transformer_core.errors import ShapeError as CoreShape

    assert issubclass(errors.ShapeError, CoreShape)
    assert issubclass(errors.ParameterError, CoreParameter)
    assert issubclass(errors.NumericError, CoreNumeric)
    for name in ("ClaimError", "ReferenceError", "CoverageError", "OrderError"):
        family = getattr(errors, name)
        assert issubclass(family, errors.BridgeError)
        assert not issubclass(family, errors.ParameterError)
    assert issubclass(errors.BridgeError, ValueError)


def test_gradient_error_is_absent_with_a_fourth_reason() -> None:
    """连续缺席的那一族有名字也**有理由**（第四条，与前三天的都不同）."""
    assert errors.ABSENT_FAMILY == "GradientError"
    assert not hasattr(errors, "GradientError")
    assert "本日不写任何新算法" in errors.ABSENT_FAMILY_REASON
    assert "量化是不可微的" not in errors.ABSENT_FAMILY_REASON


@pytest.mark.parametrize(
    "family",
    [
        "ShapeError",
        "ParameterError",
        "NumericError",
        "ClaimError",
        "ReferenceError",
        "CoverageError",
        "OrderError",
    ],
)
def test_every_family_can_be_raised_and_caught(family: str) -> None:
    """每一个失败族都至少有一条用例：抛得出、也能被 BridgeError 兜住."""
    klass = getattr(errors, family)
    with pytest.raises(errors.BridgeError):
        raise klass(f"{family} 的一次失败")


# --------------------------------------------------------------------------- 命题


def test_principles_are_twelve_and_ordered() -> None:
    """十二块拼图：条数、顺序、每块都有三个坐标."""
    items = claims.principles()
    assert len(items) == 12
    assert tuple(item.id for item in items) == types.PRINCIPLES
    assert all("." in item.artifact for item in items)
    assert all(item.probe for item in items)


def test_layer_distribution_is_two_four_three_three() -> None:
    """四层的划分是 2 / 4 / 3 / 3（数学 2、注意力 4、表征 3、推理 3）."""
    expected = {
        types.LAYER_MATH: 2,
        types.LAYER_ATTENTION: 4,
        types.LAYER_REPRESENTATION: 3,
        types.LAYER_INFERENCE: 3,
    }
    for layer, count in expected.items():
        assert len(claims.by_layer(layer)) == count
    with pytest.raises(errors.ParameterError, match="未知的层"):
        claims.by_layer("atenntion")


def test_every_application_has_at_least_one_principle() -> None:
    """六个应用每一个都至少被一条原理支撑（这是图上最基础的一条性质）."""
    for app in types.APPLICATIONS:
        assert claims.by_application(app)
    with pytest.raises(errors.ParameterError, match="未知的应用"):
        claims.by_application("search")


def test_principle_lookup_and_helpers() -> None:
    """按 id 取一块拼图；未知 id 抛 ReferenceError；两个派生查询可用."""
    item = claims.principle("cache_bytes_is_a_formula")
    assert item.application == types.APP_SERVING
    assert claims.layer_of(item.id) == types.LAYER_INFERENCE
    assert claims.application_of(item.id) == types.APP_SERVING
    with pytest.raises(errors.ReferenceError, match="未知的原理"):
        claims.principle("no_such_principle")
    with pytest.raises(errors.ReferenceError, match="未知的原理"):
        claims.layer_of("no_such_principle")


def test_applications_factory_leaves_edges_empty() -> None:
    """``applications()`` 造出六个应用，但**留空边**（填边是 graph 的事）."""
    apps = claims.applications()
    assert len(apps) == 6
    assert all(app.principles == () for app in apps)
    assert [app.id for app in apps] == list(types.APPLICATIONS)


# --------------------------------------------------------------------------- 探针


@pytest.mark.parametrize("principle_id", list(types.PRINCIPLES))
def test_each_probe_returns_a_passing_evidence(principle_id: str) -> None:
    """十二条命题各有一个探针，且现场读数都满足判据."""
    evidence = reconcile.evidence_for(principle_id)
    assert evidence.principle == principle_id
    assert evidence.passed is True, evidence.line()
    assert evidence.source


def test_probe_all_is_ordered_and_reproducible() -> None:
    """``probe_all`` 按 PRINCIPLES 顺序返回，且两次调用逐位相同."""
    first = reconcile.probe_all()
    second = reconcile.probe_all()
    assert tuple(item.principle for item in first) == types.PRINCIPLES
    assert first == second
    assert all(item.passed for item in first)


def test_probe_registry_matches_principles() -> None:
    """探针注册表的键与 twelve 条命题逐键对齐."""
    assert set(reconcile.PROBES) == set(types.PRINCIPLES)
    assert len(reconcile.PROBES) == 12


def test_evidence_for_unknown_principle_raises() -> None:
    """未知 id 的探针查询当场拒绝."""
    with pytest.raises(errors.ReferenceError, match="未知的原理"):
        reconcile.evidence_for("no_such_principle")


def test_evidence_sources_cover_every_principle() -> None:
    """来源表覆盖十二条命题，且每一条都指向一个真实函数名."""
    sources = reconcile.evidence_sources()
    assert set(sources) == set(types.PRINCIPLES)
    assert all("." in value for value in sources.values())
    assert "$" not in reconcile.probe_source("cache_reuse_is_bitwise_exact")


def test_attention_probe_reads_a_convex_combination() -> None:
    """可微检索那一条：输出与 ``weights · V`` 的最大差为 0（容差内）."""
    evidence = reconcile.probe_attention_is_differentiable_retrieval()
    assert evidence.reading == pytest.approx(0.0, abs=1e-12)
    assert "weights·V" in evidence.note


def test_cosine_probe_matches_the_hand_value() -> None:
    """余弦那一条：手算 24/25 与实测一致，且同向放大不改变余弦."""
    evidence = reconcile.probe_cosine_is_normalized_dot()
    assert evidence.reading == pytest.approx(0.0, abs=1e-12)
    assert evidence.passed is True


def test_row_distribution_probe_uses_transformer_core() -> None:
    """行分布那一条：偏差 <= 1e-9（**跨天对账**，来源是 transformer_core）."""
    evidence = reconcile.probe_attention_rows_are_distributions()
    assert evidence.upper_bound == 1e-9
    assert evidence.reading < evidence.upper_bound
    assert "transformer_core" in evidence.source


def test_causal_mask_probe_blocks_every_future_position() -> None:
    """因果掩码那一条：严格上三角的最大权重**恰好是 0.0**."""
    evidence = reconcile.probe_causal_mask_blocks_future()
    assert evidence.reading == 0.0
    assert evidence.exact is True


def test_multi_head_probe_is_a_bitwise_round_trip() -> None:
    """分头那一条：merge(split(x)) == x 逐位."""
    evidence = reconcile.probe_multi_head_splits_inside_projection()
    assert evidence.reading == 0.0
    assert "往返逐位还原 True" in evidence.note


def test_position_probe_shows_the_two_gaps() -> None:
    """位置那一条：无编码缺口为 0、有编码缺口大于 0（读数是一个 0/1 判决量）."""
    evidence = reconcile.probe_position_encoding_breaks_permutation()
    assert evidence.reading == 0.0
    assert "无编码的置换缺口" in evidence.note


def test_direction_probe_reads_plus_and_minus_one() -> None:
    """方向相似那一条：cos(5a,a)=1.0、cos(-a,a)=-1.0（容差内）."""
    evidence = reconcile.probe_embedding_similarity_is_direction()
    assert evidence.reading == pytest.approx(0.0, abs=1e-12)
    assert "向量" in evidence.source or "vectorstore" in evidence.source


def test_rank_probe_reaches_full_correlation() -> None:
    """排序对照那一条：恒等投影 + 单位行时秩相关为 1.0."""
    evidence = reconcile.probe_rank_ordering_matches_attention_peaks()
    assert evidence.reading == pytest.approx(1.0, abs=1e-12)
    assert "秩相关" in evidence.note


def test_cache_reuse_probe_is_bitwise() -> None:
    """缓存复用那一条：两条路径逐位相同（0.0 = 相同）."""
    evidence = reconcile.probe_cache_reuse_is_bitwise_exact()
    assert evidence.reading == 0.0
    assert "逐位相同 True" in evidence.note


def test_cache_formula_probe_checks_five_lengths() -> None:
    """缓存公式那一条：五个长度上整数相等（不等个数 0），每步 256 字节."""
    evidence = reconcile.probe_cache_bytes_is_a_formula()
    assert evidence.reading == 0.0
    assert "每步 256 字节" in evidence.note
    assert "核对长度 [0, 1, 5, 16, 32]" in evidence.note


def test_quantization_probe_uses_an_upper_bound() -> None:
    """量化上界那一条：**上界判定**（读数 <= scale/2），两个操作数不该相等."""
    evidence = reconcile.probe_quantization_error_bounded_by_half_step()
    assert evidence.upper_bound is not None
    assert evidence.reading <= evidence.upper_bound
    assert evidence.reading != evidence.expected


def test_budget_probe_matches_day087_total() -> None:
    """预算那一条：三项之和 == 总量（本课的读数与 day087 的 62528 一致）."""
    evidence = reconcile.probe_generation_respects_budget_breakdown()
    assert evidence.reading == evidence.expected == 62528.0
    assert "T_max=37" in evidence.note


def test_lcg_matrix_and_helpers_are_deterministic() -> None:
    """确定性造数：同一个种子同一批数；置换与最大差两个帮手可用."""
    first = reconcile._lcg_matrix(3, 4, seed=9)
    second = reconcile._lcg_matrix(3, 4, seed=9)
    assert first == second
    assert any(value != 0.0 for row in first for value in row)
    swapped = reconcile._permute(first, (1, 0, 2))
    assert swapped[0] == first[1] and swapped[1] == first[0]
    assert reconcile._max_abs_diff(first, first) == 0.0
    assert reconcile._identity(3)[1][1] == 1.0
    assert reconcile._identity(3)[0][1] == 0.0


def test_max_abs_diff_rejects_shape_mismatch() -> None:
    """最大差的两条护栏：行数不同、列数不同."""
    with pytest.raises(errors.ShapeError, match="行数不同"):
        reconcile._max_abs_diff(((1.0,),), ())
    with pytest.raises(errors.ShapeError, match="列数不同"):
        reconcile._max_abs_diff(((1.0,),), ((1.0, 2.0),))


# --------------------------------------------------------------------------- 图


def test_build_graph_has_twelve_nodes_six_apps() -> None:
    """默认那张图：十二块拼图、六个应用、十二条读数、十二条边."""
    built = graph.build_graph()
    assert len(built.principles) == 12
    assert len(built.applications) == 6
    assert len(built.evidence) == 12
    assert len(built.edges) == 12
    assert built.complete is True


def test_supported_and_unsupported_applications() -> None:
    """六个应用都被支撑；孤儿原理与缺读数的都为空."""
    assert graph.supported_applications() == types.APPLICATIONS
    assert graph.unsupported_applications() == ()
    assert graph.orphan_principles() == ()
    assert graph.build_graph().missing_evidence == ()


def test_layer_and_application_coverage_counts() -> None:
    """分层计数是 2/4/3/3；按应用计数六项都大于 0."""
    layers = graph.layer_coverage()
    assert layers == {
        types.LAYER_MATH: 2,
        types.LAYER_ATTENTION: 4,
        types.LAYER_REPRESENTATION: 3,
        types.LAYER_INFERENCE: 3,
    }
    apps = graph.application_coverage()
    assert set(apps) == set(types.APPLICATIONS)
    assert all(count > 0 for count in apps.values())
    assert apps[types.APP_SERVING] == 4


def test_coverage_report_is_complete() -> None:
    """覆盖报告完整，且 ``line`` 里印出条数."""
    report = graph.coverage_report()
    assert report.complete is True
    assert report.line().startswith("覆盖 12 条原理")


def test_graph_lookup_and_lines() -> None:
    """图上的两次查找（读证据 / 读应用）与逐行读数."""
    built = graph.build_graph()
    assert built.evidence_of("cache_bytes_is_a_formula").reading == 0.0
    assert built.application(types.APP_SERVING).supported is True
    assert built.evidence_ids == types.PRINCIPLES
    assert len(built.lines()) == len(types.APPLICATIONS) + 1
    assert built.to_dict()["coverage"]["complete"] is True
    with pytest.raises(errors.CoverageError, match="没有对应证据"):
        built.evidence_of("no_such_principle")
    with pytest.raises(errors.CoverageError, match="没有应用"):
        built.application("search")


def test_graph_lines_reads_every_application() -> None:
    """``graph_lines()`` 每个应用一行、外加一行覆盖汇总."""
    lines = graph.graph_lines()
    assert len(lines) == len(types.APPLICATIONS) + 1
    assert lines[-1].startswith("覆盖 12 条原理")


def test_resolve_artifact_finds_real_functions() -> None:
    """解析落实：十二条 artifact 都指向真实函数."""
    for item in claims.principles():
        assert graph.resolve_artifact(item.artifact) is not None


def test_resolve_artifact_rejects_bad_references() -> None:
    """解析失败的三类：没有点号、模块不存在、属性不存在."""
    with pytest.raises(errors.ReferenceError, match="模块.函数"):
        graph.resolve_artifact("justaname")
    with pytest.raises(errors.ReferenceError, match="不存在"):
        graph.resolve_artifact("no_such_package.func")
    with pytest.raises(errors.ReferenceError, match="没有"):
        graph.resolve_artifact("math_foundations.linalg.no_such_function")


def _graph_without(app: str):
    """造一张"少了某个应用支撑"的破图（用来考 CoverageError）."""
    kept = tuple(item for item in claims.principles() if item.application != app)
    evidence = tuple(item for item in reconcile.probe_all() if item.principle in {k.id for k in kept})
    return graph.build_graph(principles=kept, evidence=evidence)


def test_coverage_error_names_the_missing_application() -> None:
    """少了支撑时抛 CoverageError，且消息里**指出是哪个应用**."""
    broken = _graph_without(types.APP_SERVING)
    assert broken.unsupported == (types.APP_SERVING,)
    assert broken.complete is False
    with pytest.raises(errors.CoverageError, match="serving"):
        broken.require_complete()
    with pytest.raises(errors.CoverageError, match="serving"):
        graph.coverage_report(broken)


def test_orphan_principles_are_reported() -> None:
    """原理没有被任何应用引用（悬在图上）时，消息里指出是哪些."""
    base = graph.build_graph()
    target = "quantization_error_bounded_by_half_step"
    apps = tuple(
        app.with_principles(tuple(pid for pid in app.principles if pid != target))
        for app in base.applications
    )
    built = graph.PrincipleGraph(
        principles=base.principles, applications=apps, evidence=base.evidence
    )
    assert built.unsupported == ()
    assert built.orphans == (target,)
    with pytest.raises(errors.CoverageError, match="没有被任何应用引用"):
        built.require_complete()


def test_missing_evidence_is_reported() -> None:
    """原理没有读数（没有实现落点）时，消息里指出是哪些."""
    baseline = graph.build_graph()
    built = graph.PrincipleGraph(
        principles=baseline.principles, applications=baseline.applications, evidence=()
    )
    assert len(built.missing_evidence) == 12
    with pytest.raises(errors.CoverageError, match="没有实现落点"):
        built.require_complete()


# --------------------------------------------------------------------------- 提纲


def test_outline_has_four_sections_and_thirty_six_minutes() -> None:
    """提纲四节：每节 3 条要点、层次序正确、总时长 36 分钟."""
    sections = outline.build_outline()
    assert len(sections) == 4
    assert [section.layer for section in sections] == list(types.LAYER_ORDER)
    assert all(len(section.bullets) == 3 for section in sections)
    assert outline.outline_minutes(sections) == 36
    assert [section.index for section in sections] == [1, 2, 3, 4]


def test_outline_check_order_rejects_reversed_layers() -> None:
    """讲反了要抛 OrderError（拓扑序检查）."""
    sections = outline.build_outline()
    reversed_sections = tuple(reversed(sections))
    with pytest.raises(errors.OrderError, match="前置依赖"):
        outline.check_order(reversed_sections)


def test_outline_check_order_allows_same_layer_twice() -> None:
    """同一个层用两节讲是允许的（判据是"非递减"，不是"严格递增"）."""
    sections = outline.build_outline()
    duplicated = (sections[0], sections[0], sections[1])
    outline.check_order(duplicated)


def test_outline_demos_exist_on_disk() -> None:
    """提纲里引用的每个 demo 脚本都真实存在，且去重后是四个."""
    sections = outline.build_outline()
    assert outline.missing_demos(sections) == ()
    assert outline.demo_scripts() == (
        "scripts/math_foundations_demo.py",
        "scripts/attention_demo.py",
        "scripts/retrieval_demo.py",
        "scripts/inference_optim_demo.py",
    )


def test_outline_missing_demos_detects_a_ghost_script() -> None:
    """一个指向不存在脚本的 demo 会被 missing_demos 抓出来."""
    ghost = types.TalkSection(
        index=9,
        title="t",
        bullets=("a", "b", "c"),
        demo="python scripts/no_such_demo.py",
        minutes=1,
        layer=types.LAYER_MATH,
    )
    assert outline.missing_demos((ghost,)) == ("python scripts/no_such_demo.py",)


def test_outline_lines_include_total() -> None:
    """提纲逐行打印里含总时长与缺口两个读数."""
    lines = outline.outline_lines()
    assert any(line.startswith("总时长 36 分钟") for line in lines)
    assert any("demo" in line for line in lines)


def test_document_covers_all_principles() -> None:
    """原理文档十节，覆盖全部十二条命题."""
    document = outline.render_document()
    assert len(outline.DOCUMENT_SECTIONS) == 10
    assert outline.missing_principles(document) == ()
    outline.ensure_document_covers_all()
    assert "底层模型原理解析" in document
    lines = outline.document_lines()
    assert any(line.startswith("## 十、") for line in lines)


def test_ensure_document_raises_when_something_is_missing(monkeypatch) -> None:
    """文档漏掉命题时抛 CoverageError（把渲染结果替换成空串来触发）."""
    monkeypatch.setattr(outline, "render_document", lambda graph=None: "")
    with pytest.raises(errors.CoverageError, match="漏掉了"):
        outline.ensure_document_covers_all()


def test_missing_principles_lists_every_id_for_empty_text() -> None:
    """空文本里十二条命题一个都没有（这是 missing_principles 的直接读数）."""
    assert outline.missing_principles("") == types.PRINCIPLES


# --------------------------------------------------------------------------- 性质


def test_cross_check_three_kinds() -> None:
    """三类判据：逐位、容差、上界（上界那一类两个操作数不该相等）."""
    exact = verify.CrossCheck(name="a", left="l", right="r", reading=1.0, expected=1.0)
    assert exact.passed is True
    off = verify.CrossCheck(name="a", left="l", right="r", reading=1.0, expected=2.0)
    assert off.passed is False and "不满足" in off.line()
    bounded = verify.CrossCheck(
        name="b", left="l", right="r", reading=0.09, expected=0.1, upper_bound=0.1
    )
    assert bounded.passed is True
    assert "≤ 上界" in bounded.line()
    too_big = verify.CrossCheck(
        name="b", left="l", right="r", reading=0.11, expected=0.1, upper_bound=0.1
    )
    assert too_big.passed is False


def test_cross_check_tolerance_branch() -> None:
    """``exact=False`` 且没有上界时走容差判据（卡在 1e-12 边界两侧）."""
    inside = verify.CrossCheck(
        name="t", left="l", right="r", reading=1.0 + 1e-13, expected=1.0, exact=False
    )
    assert inside.passed is True
    outside = verify.CrossCheck(
        name="t", left="l", right="r", reading=1.0 + 1e-9, expected=1.0, exact=False
    )
    assert outside.passed is False


def test_cross_check_guards() -> None:
    """对账记录的三条护栏：非有限读数、非有限期望、负上界."""
    with pytest.raises(errors.NumericError, match="必须有限"):
        verify.CrossCheck(name="a", left="l", right="r", reading=float("inf"), expected=0.0)
    with pytest.raises(errors.NumericError, match="必须有限"):
        verify.CrossCheck(name="a", left="l", right="r", reading=0.0, expected=float("nan"))
    with pytest.raises(errors.NumericError, match="上界"):
        verify.CrossCheck(
            name="a", left="l", right="r", reading=0.0, expected=0.0, upper_bound=-1.0
        )


def test_property_outcome_forbids_not_applicable_as_passed() -> None:
    """"不适用"与"通过"必须分开（构造期就拒绝）."""
    skipped = verify.PropertyOutcome(name="x", applicable=False, passed=False, evidence=("a",))
    assert skipped.line().startswith("[不适用]")
    passed = verify.PropertyOutcome(name="y", applicable=True, passed=True, evidence=("b",))
    assert passed.line().startswith("[通过]")
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome(name="x", applicable=False, passed=True)


def test_property_report_bookkeeping() -> None:
    """报告的三件事：适用的通过才算 ok、不通过时抛错、不适用先印."""
    passed = verify.PropertyOutcome(name="p", applicable=True, passed=True)
    failed = verify.PropertyOutcome(name="f", applicable=True, passed=False)
    skipped = verify.PropertyOutcome(name="s", applicable=False, passed=False)
    report = verify.PropertyReport(outcomes=(passed, skipped))
    assert report.ok is True
    assert report.lines()[0].startswith("[不适用]")
    assert report.to_dict()["counts"] == {"total": 2, "applicable": 1}
    broken = verify.PropertyReport(outcomes=(passed, failed))
    assert broken.ok is False
    with pytest.raises(errors.CoverageError, match="未全部通过"):
        broken.require_ok()
    verify.PropertyReport(outcomes=(passed,)).require_ok()


def test_every_single_check_passes() -> None:
    """七条性质逐条检查，全部通过."""
    outcomes = (
        verify.check_every_principle_has_artifact(),
        verify.check_every_application_is_supported(),
        verify.check_evidence_is_reproducible(),
        verify.check_attention_rows_are_distributions(),
        verify.check_cache_formula_matches_day087(),
        verify.check_outline_respects_dependencies(),
        verify.check_document_covers_all_principles(),
    )
    for outcome in outcomes:
        assert outcome.applicable is True
        assert outcome.passed is True, outcome.line()
        assert outcome.cross_check is not None


def test_artifact_check_detects_a_ghost_reference() -> None:
    """一条指向不存在函数的原理会让"实现落点"那条性质失败."""
    ghost = types.Principle(
        id="ghost_principle",
        layer=types.LAYER_MATH,
        statement="不存在",
        artifact="math_foundations.linalg.no_such_function",
        probe="p",
        application=types.APP_SERVING,
        source_day="day000",
    )
    built = graph.build_graph(
        principles=(ghost,) + tuple(claims.principles()[:1]),
        evidence=reconcile.probe_all(),
    )
    outcome = verify.check_every_principle_has_artifact(built)
    assert outcome.passed is False
    assert "ghost_principle" in " ".join(outcome.evidence)


def test_application_check_detects_an_island() -> None:
    """某个应用没有入边时，"应用被支撑"那条性质失败，且消息里点名."""
    broken = _graph_without(types.APP_RAG_RERANK)
    outcome = verify.check_every_application_is_supported(broken)
    assert outcome.passed is False
    assert types.APP_RAG_RERANK in " ".join(outcome.evidence)


def test_outline_check_detects_reversed_order() -> None:
    """提纲讲反时，"前置依赖"那条性质失败."""
    reversed_sections = tuple(reversed(outline.build_outline()))
    outcome = verify.check_outline_respects_dependencies(reversed_sections)
    assert outcome.passed is False
    assert "OrderError" in " ".join(outcome.evidence) or "前置依赖" in " ".join(outcome.evidence)


def test_check_all_returns_seven_green_outcomes() -> None:
    """七条性质在一个完整的图上全绿（本课的总验收）."""
    report = verify.check_all()
    assert len(report.outcomes) == len(types.PRINCIPLE_PROPERTIES)
    assert report.ok is True, "\n".join(report.lines())
    assert all(outcome.passed for outcome in report.applicable)
    names = {outcome.name for outcome in report.outcomes}
    assert names == set(types.PRINCIPLE_PROPERTIES)


# --------------------------------------------------------------------------- 表


def test_study_five_tables() -> None:
    """五张表一次跑完（行数可由各表的规模加出来）."""
    lines = study.study_lines()
    headers = [line for line in lines if line.startswith("== ")]
    assert len(headers) == 5
    assert len(study.principle_rows()) == 12
    assert len(study.layer_rows()) == 4
    assert len(study.application_rows()) == 6
    assert len(study.evidence_rows()) == 12
    assert len(study.outline_rows()) == 4


def test_study_rows_carry_two_numbers() -> None:
    """表里每一行都带两个数（读数与参照），且可用标志列可读."""
    principle = study.principle_rows()[0]
    assert principle.passed is True
    assert "读数" in principle.line()
    layer = study.layer_rows()[0]
    assert "2 条" in layer.line()
    application = study.application_rows()[0]
    assert application.supported is True
    assert "✓" in application.line()
    evidence = study.evidence_rows()[2]
    assert "≤ 上界" in evidence.line()
    section = study.outline_rows()[0]
    assert "demo" in section.line()


def test_study_note_lines_are_ten_and_truncatable() -> None:
    """笔记逐行印出（十条，且可以截断）."""
    assert len(study.note_lines()) == 10
    assert study.note_lines(limit=2) == (
        f" 1. {types.PRINCIPLE_NOTES[types.PRINCIPLE_NOTES_ORDER[0]]}",
        f" 2. {types.PRINCIPLE_NOTES[types.PRINCIPLE_NOTES_ORDER[1]]}",
    )


# --------------------------------------------------------------------------- 包出口


def test_package_exports_are_unique_and_importable() -> None:
    """``__all__`` 里的名字都真的存在，且没有重复."""
    import smart_research_agent.principle_map as package

    assert len(package.__all__) == len(set(package.__all__))
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


def test_module_export_lists_match_the_package() -> None:
    """每个子模块的 ``__all__`` 都是包导出集合的子集（不许有暗门）."""
    import smart_research_agent.principle_map as package

    exported = set(package.__all__)
    for module in (errors, types, claims, reconcile, graph, outline, verify, study):
        assert set(module.__all__) <= exported, module.__name__


def test_principle_map_has_no_third_party_imports() -> None:
    """**零外部依赖**：本包不 import transformers / torch / numpy / bitsandbytes."""
    import smart_research_agent.principle_map as package

    root = pathlib.Path(package.__file__).parent
    forbidden = {"transformers", "torch", "numpy", "bitsandbytes", "accelerate"}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (names & forbidden), (path.name, sorted(names & forbidden))
