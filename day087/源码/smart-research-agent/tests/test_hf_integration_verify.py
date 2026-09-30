"""``hf_integration`` 的性质、表与失败族（day086 / M7-D10）.

这一份文件覆盖四块彼此相关的"记账"：
九张判断表（``types``）、十条性质与三类对账（``verify``）、六张表（``study``）、
七个失败族（``errors``），以及包入口 ``__init__`` 的自我一致性。
"""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration import errors, study, types, verify
from smart_research_agent.hf_integration.config import reference_cards, tiny_card
from smart_research_agent.hf_integration.forward import make_weights
from smart_research_agent.hf_integration.types import (
    ARCHITECTURES,
    ARCHITECTURE_DESCRIPTIONS,
    FEATURE_STAGES,
    FEATURE_STAGE_SHAPES,
    INTEGRATION_BOUNDARIES,
    INTEGRATION_NOTES,
    INTEGRATION_NOTES_ORDER,
    INTEGRATION_PROPERTIES,
    KNOWN_FILES,
    POOLING_DESCRIPTIONS,
    POOLING_STRATEGIES,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FAILURE,
    REQUIRED_CONFIG_KEYS,
    SIZE_KEYS,
    TASK_DESCRIPTIONS,
    TASK_KINDS,
    TRANSFORMERS_LIBRARY,
    TRANSFORMERS_VERSION,
    TRANSFORMERS_VERSION_SAMPLE,
    Encoding,
    IntegrationRecord,
    PooledBatch,
    Snapshot,
    TextBatch,
)
from smart_research_agent.hf_source.types import GenerationSettings

from tests.hf_integration_samples import MAIN_COMMIT, build_case


# --------------------------------------------------------------------------- types


def test_architecture_tables_are_closed() -> None:
    """两个架构的名单、说明、必需键、键名映射四张表逐键对齐."""
    assert set(ARCHITECTURES) == {"gpt2", "bert"}
    assert set(ARCHITECTURE_DESCRIPTIONS) == set(ARCHITECTURES)
    assert set(REQUIRED_CONFIG_KEYS) == set(ARCHITECTURES)
    assert set(SIZE_KEYS) == set(ARCHITECTURES)
    for model_type in ARCHITECTURES:
        assert len(SIZE_KEYS[model_type]) == 7
        assert len(REQUIRED_CONFIG_KEYS[model_type]) >= 6


def test_pooling_and_task_tables_are_closed() -> None:
    """池化三法与任务两种：名单与说明逐键对齐."""
    assert len(POOLING_STRATEGIES) == 3
    assert set(POOLING_DESCRIPTIONS) == set(POOLING_STRATEGIES)
    assert set(TASK_KINDS) == {"text-generation", "feature-extraction"}
    assert set(TASK_DESCRIPTIONS) == set(TASK_KINDS)


def test_ten_properties_have_descriptions_and_failures() -> None:
    """十条性质：名单、说明、"失败意味着什么"三张表逐键对齐."""
    assert len(INTEGRATION_PROPERTIES) == 10
    assert set(PROPERTY_DESCRIPTIONS) == set(INTEGRATION_PROPERTIES)
    assert set(PROPERTY_FAILURE) == set(INTEGRATION_PROPERTIES)
    assert all(PROPERTY_DESCRIPTIONS.values())
    assert all(PROPERTY_FAILURE.values())


def test_feature_stage_table_matches_the_stage_list() -> None:
    """特征抽取的五个阶段：名单与形状表逐键对齐."""
    assert len(FEATURE_STAGES) == 5
    assert set(FEATURE_STAGE_SHAPES) == set(FEATURE_STAGES)
    assert "batch" in FEATURE_STAGE_SHAPES["input_ids"]


def test_notes_are_twelve_and_ordered() -> None:
    """十二条生态笔记，且顺序表与键集合一致."""
    assert len(INTEGRATION_NOTES) == 12
    assert INTEGRATION_NOTES_ORDER == tuple(INTEGRATION_NOTES)
    assert all(INTEGRATION_NOTES.values())


def test_boundaries_and_known_files() -> None:
    """五条边界是一份非空清单；四个认得的文件名写进常量."""
    assert len(INTEGRATION_BOUNDARIES) == 5
    assert all(INTEGRATION_BOUNDARIES)
    assert set(KNOWN_FILES) == {"config.json", "vocab.json", "merges.txt", "model.safetensors"}


def test_version_constants_are_recorded() -> None:
    """版本号写进常量（"当时接的是哪一版"不是传说）."""
    assert TRANSFORMERS_LIBRARY == "huggingface/transformers"
    assert TRANSFORMERS_VERSION.startswith("5")
    assert "v5." in TRANSFORMERS_VERSION_SAMPLE
    assert types.HUB_LIBRARY == "huggingface_hub"


def test_encoding_derived_quantities() -> None:
    """一条编码的三个派生量：总长、真实长、填充数（池化只认中间那个）."""
    encoding = Encoding(
        tokens=("a", "b", "<pad>"),
        input_ids=(1, 2, 0),
        attention_mask=(1, 1, 0),
        truncated=True,
    )
    assert encoding.length == 3
    assert encoding.real_length == 2
    assert encoding.padding == 1
    assert encoding.to_dict()["truncated"] is True


def test_encoding_rejects_mismatched_lengths() -> None:
    """三串长度不齐时当场拒绝（长度不齐会被 zip 静默截断）."""
    with pytest.raises(errors.ShapeError, match="长度必须一致"):
        Encoding(tokens=("a",), input_ids=(1, 2), attention_mask=(1, 1))


def test_text_batch_derived_quantities() -> None:
    """一批的三个派生量：宽度、真实宽度、填充占比."""
    rows = (
        Encoding(tokens=("a", "b"), input_ids=(1, 2), attention_mask=(1, 1)),
        Encoding(
            tokens=("c", "d", "e", "f"),
            input_ids=(3, 4, 5, 6),
            attention_mask=(1, 1, 0, 0),
        ),
    )
    batch = TextBatch(rows=rows, padding="longest", pad_token_id=0, max_length=4)
    assert batch.batch_size == 2
    assert batch.width == 4
    assert batch.real_width == 2
    assert batch.padding_ratio == pytest.approx(1 - 4 / 8)
    assert batch.mask_matrix() == ((1, 1), (1, 1, 0, 0))
    assert batch.to_dict()["padding"] == "longest"


def test_text_batch_of_an_empty_batch() -> None:
    """空批的宽度与占比都是 0（不是除零）."""
    batch = TextBatch(rows=(), padding="longest", pad_token_id=0, max_length=0)
    assert batch.batch_size == 0
    assert batch.width == 0
    assert batch.padding_ratio == 0.0


def test_pooled_batch_derived_quantities() -> None:
    """一组向量的三个读数：条数、宽度、各自的范数."""
    pooled = PooledBatch(vectors=((3.0, 4.0),), strategy="mean")
    assert pooled.batch_size == 1
    assert pooled.dim == 2
    assert pooled.to_dict()["norms"] == [5.0]
    empty = PooledBatch(vectors=(), strategy="mean")
    assert empty.batch_size == 0
    assert empty.dim == 0


def test_snapshot_derived_quantities() -> None:
    """快照的四个派生量（含"新下了多少字节"）."""
    case = build_case("gpt2")
    snapshot = case.snapshot
    assert snapshot.file_count == 4
    assert snapshot.cached_count + snapshot.downloaded_count == snapshot.file_count
    assert snapshot.total_bytes == sum(entry.size for entry in snapshot.files)
    payload = snapshot.to_dict()
    assert payload["commit"] == MAIN_COMMIT
    assert payload["local_files_only"] is False
    assert payload["file_count"] == 4


def test_integration_record_to_dict() -> None:
    """一次装载的账摊平之后含全部字段."""
    record = IntegrationRecord(
        repo_id="org/tiny",
        revision="main",
        commit=MAIN_COMMIT,
        model_type="gpt2",
        parameters=11536,
        files=4,
        bytes_total=3848,
        tokens=5,
        pooled_dim=16,
        strategy="mean",
        lines=("a",),
    )
    payload = record.to_dict()
    assert payload["parameters"] == 11536
    assert payload["pooled_dim"] == 16
    assert payload["lines"] == ["a"]


# --------------------------------------------------------------------------- errors


def test_family_tables_are_closed() -> None:
    """七个族的两张表逐键对齐（少一个键的那一族就只剩类名）."""
    assert len(errors.FAMILY_OUTCOMES) == 7
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "ConfigError",
        "HubError",
        "TokenError",
        "AssemblyError",
        "NumericError",
    }
    assert all(errors.FAMILY_OUTCOMES.values())


def test_family_inheritance_holds() -> None:
    """继承关系：本层继承 day075 的三族，而 HubError **不是**参数错."""
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
    assert issubclass(errors.ConfigError, errors.ParameterError)
    assert issubclass(errors.TokenError, errors.ParameterError)
    assert issubclass(errors.AssemblyError, errors.ParameterError)
    assert not issubclass(errors.HubError, errors.ParameterError)
    assert issubclass(errors.HubError, errors.IntegrationError)
    assert issubclass(errors.IntegrationError, ValueError)


def test_gradient_error_is_still_absent() -> None:
    """连续缺席的那一族有名字也**有理由**（理由与 day085 的那一条不同）."""
    assert errors.ABSENT_FAMILY == "GradientError"
    assert not hasattr(errors, "GradientError")
    assert "一行反向都不写" in errors.ABSENT_FAMILY_REASON


# --------------------------------------------------------------------------- verify


def test_property_outcome_forbids_not_applicable_as_passed() -> None:
    """"不适用"与"通过"必须分开（构造期就拒绝）."""
    ok = verify.PropertyOutcome(name="x", applicable=False, passed=False, evidence=("a",))
    assert ok.line().startswith("[不适用]")
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome(name="x", applicable=False, passed=True)


def test_cross_check_exact_and_tolerant() -> None:
    """逐位与容差两种比法各有用武之地."""
    exact = verify.CrossCheck(name="a", left="l", right="r", reading=0, expected=0)
    assert exact.passed is True
    assert "一致" in exact.line()
    off = verify.CrossCheck(name="a", left="l", right="r", reading=1, expected=0)
    assert off.passed is False
    assert "不一致" in off.line()
    tolerant = verify.CrossCheck(
        name="b", left="l", right="r", reading=1e-15, expected=0.0, exact=False
    )
    assert tolerant.passed is True
    assert "容差 1e-12" in tolerant.line()


def test_property_report_bookkeeping() -> None:
    """报告的三件事：适用的通过才算 ok、不通过时抛错、不适用先印."""
    passed = verify.PropertyOutcome(name="p", applicable=True, passed=True)
    failed = verify.PropertyOutcome(name="f", applicable=True, passed=False)
    skipped = verify.PropertyOutcome(name="s", applicable=False, passed=False)
    report = verify.PropertyReport(outcomes=(passed, skipped))
    assert report.ok is True
    assert len(report.applicable) == 1
    assert report.lines()[0].startswith("[不适用]")
    assert report.to_dict()["counts"] == {"total": 2, "applicable": 1}
    broken = verify.PropertyReport(outcomes=(passed, failed))
    assert broken.ok is False
    with pytest.raises(errors.AssemblyError, match="未全部通过"):
        broken.require_ok()
    verify.PropertyReport(outcomes=(passed,)).require_ok()


def test_all_ten_checks_pass_on_a_built_case() -> None:
    """十条性质在造好的 case 上全绿（这是本课的总验收）."""
    case = build_case("gpt2", n_layer=1, n_head=1)
    report = verify.check_all(
        case.card,
        case.weights,
        case.tokenizer,
        case.resolver,
        case.snapshot,
        case.model(),
        texts=case.texts,
        prompt=case.prompt,
    )
    assert len(report.outcomes) == len(INTEGRATION_PROPERTIES)
    assert report.ok is True
    assert all(outcome.passed for outcome in report.applicable)


def test_library_property_is_not_applicable_for_toy_cards() -> None:
    """玩具卡片没有库读数可对 ⇒ **不适用**（而不是"通过"）."""
    outcome = verify.check_params_match_library(tiny_card("gpt2"))
    assert outcome.applicable is False
    assert outcome.passed is False
    assert "没有库读数" in "；".join(outcome.evidence)
    real = verify.check_params_match_library(reference_cards()["gpt2"])
    assert real.applicable is True
    assert real.passed is True


def test_block_property_applies_only_for_single_layer_single_head() -> None:
    """块对账只在一个很窄的条件下适用（其余情况**不适用也要印出来**）."""
    bert = build_case("bert")
    outcome = verify.check_block_matches_day085(
        bert.card, bert.weights, bert.tokenizer, bert.prompt
    )
    assert outcome.applicable is False
    many = build_case("gpt2", n_layer=2)
    outcome = verify.check_block_matches_day085(
        many.card, many.weights, many.tokenizer, many.prompt
    )
    assert outcome.applicable is False
    single = build_case("gpt2", n_layer=1, n_head=1)
    ok = verify.check_block_matches_day085(
        single.card, single.weights, single.tokenizer, single.prompt
    )
    assert ok.applicable is True and ok.passed is True


def test_individual_checks_report_their_readings() -> None:
    """逐条检查：每一条都在自己的证据行里带上两个数."""
    case = build_case("gpt2")
    flat = verify.check_snapshot_is_flat(case.snapshot)
    assert flat.passed is True
    assert len(flat.evidence) == 3
    cache = verify.check_cache_is_per_file(case.resolver, case.snapshot.repo_id)
    assert cache.passed is True
    assert "首次：下 4 个" in "；".join(cache.evidence)
    round_trip = verify.check_round_trip_is_exact(case.tokenizer)
    assert round_trip.passed is True
    pooling = verify.check_pooling_respects_mask(
        case.card, case.weights, case.tokenizer, case.texts
    )
    assert pooling.passed is True
    batch = verify.check_batch_matches_single(
        case.card, case.weights, case.tokenizer, case.texts
    )
    assert batch.passed is True
    assert "归因：" in "；".join(batch.evidence)
    gap = verify.check_weights_gap_is_explained(case.card, case.weights)
    assert gap.passed is True
    surface = verify.check_protocol_surface_matches(case.model())
    assert surface.passed is True
    generation = verify.check_generation_matches_day085(
        case.card, case.weights, case.tokenizer, case.prompt
    )
    assert generation.passed is True
    assert "策略 greedy" in "；".join(generation.evidence)


def test_checks_require_at_least_two_texts() -> None:
    """池化与批量两条性质都需要至少两条样本（一条样本没有"批"这回事）."""
    case = build_case("gpt2")
    with pytest.raises(errors.NumericError, match="至少两条"):
        verify.check_pooling_respects_mask(case.card, case.weights, case.tokenizer, ("a",))
    with pytest.raises(errors.NumericError, match="至少两条"):
        verify.check_batch_matches_single(case.card, case.weights, case.tokenizer, ("a",))


def test_reference_agreement_lines_cover_both_models() -> None:
    """第三条跨天对账的读数：配置 ↔ 源码画像，两条都要一致."""
    lines = verify.reference_agreement_lines()
    assert len(lines) == 2
    assert all("两侧一致：True" in line for line in lines)


def test_finite_or_raise_guards_the_report_end() -> None:
    """报告端的最后一道闸：非有限数当场拒绝."""
    verify.finite_or_raise((1.0, 2.0), name="x")
    with pytest.raises(errors.NumericError, match="有限数"):
        verify.finite_or_raise((1.0, float("nan")), name="x")


def test_library_records_match_the_constants() -> None:
    """记录表里的两个整数与常量表逐位相同（来源写进了证据行）."""
    assert verify.LIBRARY_RECORDS == {
        "gpt2": types.GPT2_SMALL_PARAMETERS,
        "bert-base-uncased": types.BERT_BASE_PARAMETERS,
    }
    assert "5.17.0" in verify.LIBRARY_RECORD_SOURCE
    assert len(verify.ROUND_TRIP_SAMPLES) >= 6


# --------------------------------------------------------------------------- study


def test_cache_rows_show_two_passes() -> None:
    """缓存表：第一轮全"新下"、第二轮全"命中"——两行放在一起才看得出"逐文件"."""
    case = build_case("gpt2")
    rows = study.cache_rows(case.resolver, case.resolver.hub, case.snapshot.repo_id)
    assert len(rows) == 8
    first = [row for row in rows if row.pass_number == 1]
    second = [row for row in rows if row.pass_number == 2]
    assert all(row.cached is False for row in first)
    assert all(row.cached is True for row in second)
    assert "第1次" in first[0].line()


def test_param_rows_cover_four_cards() -> None:
    """参数量表四行：两张真实卡片 + 两张玩具卡片（含 tying 的代价）.

    真实卡片那一行**不造权重**（1.2 亿个数），因此 ``weights`` 与 ``gap`` 是 ``None``；
    而"实测差"这一列在玩具卡片上是真的被算出来的。
    """
    rows = study.param_rows()
    assert len(rows) == 4
    by_name = {row.name: row for row in rows}
    assert by_name["gpt2"].library == types.GPT2_SMALL_PARAMETERS
    assert by_name["gpt2"].config == by_name["gpt2"].library
    assert by_name["gpt2"].weights is None
    assert by_name["bert-base-uncased"].expected_gap == 4 * 768 * 12
    assert by_name["gpt2-tiny"].library is None
    assert by_name["gpt2-tiny"].gap == by_name["gpt2-tiny"].expected_gap
    assert by_name["gpt2-tiny"].weights is not None
    assert "（无库读数）" in by_name["gpt2-tiny"].line()
    assert by_name["gpt2"].line().count("—") == 2
    assert study.param_rows(weights_for_toy=False)[2].weights is None


def test_token_rows_show_the_ratio() -> None:
    """分词表：同一段文本在**这份词表**下的 token 数是可读的."""
    case = build_case("gpt2")
    rows = study.token_rows(case.tokenizer)
    assert len(rows) == len(study.TOKEN_SAMPLES)
    assert all(row.round_trip for row in rows)
    chinese = next(row for row in rows if row.text == "你好，世界")
    assert chinese.tokens == 3 * len(chinese.text)
    assert "token/字符" in chinese.line()


def test_pooling_rows_show_zero_and_nonzero() -> None:
    """池化表：正确版两种填充下的差恒为 0，违反版不为 0."""
    case = build_case("bert")
    rows = study.pooling_rows(case.card, case.weights, case.tokenizer)
    assert len(rows) == len(POOLING_STRATEGIES)
    for row in rows:
        assert row.gap_correct == 0.0
        assert row.gap_buggy_short > 0.0
        assert row.gap_buggy_long > 0.0
        assert 0.0 < row.padding_waste < 1.0
        assert "正确版" in row.line()


def test_batch_rows_show_two_kinds_of_naive_mistake() -> None:
    """批量表：两种架构的 naive 错法**不同**（跨样本串味 vs 填充污染）."""
    case = build_case("gpt2")
    rows = study.batch_rows(case.tokenizer)
    assert [row.model_type for row in rows] == ["gpt2", "bert"]
    assert all(row.safe_matches_single is True for row in rows)
    assert all(row.naive_gap > 0.0 for row in rows)
    assert "跨样本串味" in rows[0].line()
    assert "填充污染" in rows[1].line()


def test_generation_rows_cover_four_strategies() -> None:
    """生成表：四条配置各动一个旋钮，策略名与保留数都可读."""
    case = build_case("gpt2")
    rows = study.generation_rows(case.card, case.weights, case.tokenizer)
    assert len(rows) == len(study.GENERATION_CASES)
    assert {row.strategy for row in rows} == {"greedy", "top_k", "sample", "top_p"}
    assert all(len(row.generated) == 3 for row in rows)
    assert all(1 <= row.kept <= case.card.vocab for row in rows)


def test_note_lines_and_all_tables() -> None:
    """六张表一次跑完：行数可由各表的规模加出来."""
    case = build_case("gpt2", n_layer=1, n_head=1)
    lines = study.study_lines(
        case.card, case.weights, case.tokenizer, case.snapshot, case.resolver
    )
    headers = [line for line in lines if line.startswith("== ")]
    assert len(headers) == 6
    assert study.note_lines(limit=3) == tuple(
        f"{index:>2}. {value}"
        for index, value in enumerate(tuple(INTEGRATION_NOTES.values())[:3], start=1)
    )
    assert len(study.note_lines()) == 12


# --------------------------------------------------------------------------- 包入口


def test_package_exports_are_unique_and_importable() -> None:
    """``__all__`` 里的名字都真的存在，且没有重复."""
    import smart_research_agent.hf_integration as package

    assert len(package.__all__) == len(set(package.__all__))
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


def test_module_export_lists_match_the_package() -> None:
    """每个子模块的 ``__all__`` 都是包导出集合的子集（不许有暗门）."""
    import smart_research_agent.hf_integration as package

    exported = set(package.__all__)
    for module in (
        errors,
        study,
        types,
        verify,
        package.hub,
        package.config,
        package.tokenizer,
        package.forward,
        package.features,
        package.pipeline,
        package.bridge,
    ):
        assert set(module.__all__) <= exported, module.__name__


def test_integration_does_not_need_transformers_or_torch() -> None:
    """**零外部依赖**：本包的核心路径不 import transformers 或 torch.

    这是刻意的设计：把"能不能装"与"装了没装"分开——
    核心算术与本机是否装了那个库无关（真库的对账放在一个单独的、可跳过的文件里）。
    """
    import ast
    import pathlib

    import smart_research_agent.hf_integration as package

    root = pathlib.Path(package.__file__).parent
    forbidden = {"transformers", "torch", "numpy"}
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
