"""day060 ``serving.binding`` 的单元测试：三处版本比对.

本文件里出现的每一个"故障场景"都对应一种**不会报错但会答错**的部署事故：
``adapter_dir`` 指到旧目录、基座被换过、某个副本还是旧配置。
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.registry.record import STAGE_CANDIDATE, STAGE_STABLE, ModelVersion
from smart_research_agent.registry.version import VersionTriple
from smart_research_agent.serving.binding import (
    BINDING_FILE,
    CHECK_ADAPTER_HASH,
    CHECK_BASE_MODEL,
    CHECK_KIND_SHAPE,
    CHECK_REGISTRY_HEAD,
    CHECK_SERVING_NAME,
    SHORT_HASH_LENGTH,
    BindingCheck,
    BindingPolicy,
    ServedEndpoint,
    ServingBinding,
    bind_version,
    binding_table,
    read_binding,
    verify_binding,
    write_binding,
)
from smart_research_agent.serving.errors import ServingError
from smart_research_agent.serving.spec import KIND_ADAPTER, KIND_BASE, KIND_MERGED, ServingSpec

#: 一份可部署的适配器哈希（40 位十六进制，与 ``sha256`` 同形）。
ADAPTER_SHA = "5c6d7e8f9012345678901234567890abcdef0123"
#: 另一份适配器（"线上挂错了"场景用它）。
OTHER_SHA = "deadbeef0000000000000000000000000000beef"
DATASET_FP = "aa77c31e90f4b258"
BASE_MODEL = "Qwen/Qwen3-0.6B"


def _version(
    version: str = "1.1.0",
    *,
    stage: str = STAGE_STABLE,
    adapter_sha256: str = ADAPTER_SHA,
    base_model: str = BASE_MODEL,
    with_merged: bool = True,
    metrics: dict | None = None,
) -> ModelVersion:
    """造一条版本记录（缺省是可部署的 stable 1.1.0）."""
    artifacts = {"adapter": "outputs/lora/adapters/adapter-final"}
    if with_merged:
        artifacts["merged"] = "outputs/lora-merged"
    return ModelVersion(
        triple=VersionTriple(
            base_model=base_model,
            adapter_sha256=adapter_sha256,
            dataset_fingerprint=DATASET_FP,
        ),
        version=version,
        stage=stage,
        artifacts=artifacts,
        metrics={"eval_pass_rate": 0.72} if metrics is None else metrics,
    )


def _spec(kind: str = KIND_ADAPTER, **overrides) -> ServingSpec:
    payload: dict = {
        "name": "smart-research-qwen3-8b",
        "kind": kind,
        "base_model": BASE_MODEL,
        "adapter_dir": "outputs/lora/adapters/adapter-final",
        "merged_dir": "outputs/lora-merged",
    }
    payload.update(overrides)
    return ServingSpec(**payload)


# --------------------------------------------------------------------------- #
# 绑定：部署的前置条件
# --------------------------------------------------------------------------- #


def test_bind_version_happy_path() -> None:
    binding = bind_version(_spec(), _version(), serving_name="smart-research-qwen3-8b")
    assert binding.version == "1.1.0"
    assert binding.short_adapter == ADAPTER_SHA[:SHORT_HASH_LENGTH]
    assert binding.is_canary is False
    assert binding.spec.kind == KIND_ADAPTER
    assert "adapter-final" not in binding.summary_line()  # 摘要印的是哈希不是路径
    assert "sha256:" in binding.summary_line()


def test_canary_binding_is_flagged() -> None:
    """灰度期间服务名与部署单元名不同——这是允许的，但**必须被记下来**."""
    binding = bind_version(
        _spec(), _version(), serving_name="smart-research-qwen3-8b-canary"
    )
    assert binding.is_canary is True
    assert "（灰度）" in binding.summary_line()
    assert binding.to_dict()["is_canary"] is True


def test_bind_version_rejects_non_deployable_artifacts() -> None:
    """缺合并模型 = 交付没完成，不该进入部署记录（复用 day058 的 is_deployable）."""
    with pytest.raises(ServingError, match="产物不完整"):
        bind_version(_spec(), _version(with_merged=False))


def test_bind_version_rejects_a_base_model_mismatch() -> None:
    """基座不匹配的适配器能挂上也会答错——因此在**构造部署记录时**就拦住."""
    with pytest.raises(ServingError, match="基座不匹配"):
        bind_version(_spec(), _version(base_model="Qwen/Qwen3-8B"))


def test_binding_rejects_blank_identity_fields() -> None:
    spec = _spec()
    version = _version()
    with pytest.raises(ServingError, match="部署记录缺少 version_key"):
        ServingBinding(
            spec=spec,
            version="1.1.0",
            version_key="",
            base_model=BASE_MODEL,
            adapter_sha256=ADAPTER_SHA,
            dataset_fingerprint=DATASET_FP,
            serving_name=spec.name,
        )
    with pytest.raises(ServingError, match="部署记录缺少 dataset_fingerprint"):
        ServingBinding(
            spec=spec,
            version="1.1.0",
            version_key=version.version_key,
            base_model=BASE_MODEL,
            adapter_sha256=ADAPTER_SHA,
            dataset_fingerprint="",
            serving_name=spec.name,
        )
    with pytest.raises(ServingError, match="部署记录缺少 serving_name"):
        ServingBinding(
            spec=spec,
            version="1.1.0",
            version_key=version.version_key,
            base_model=BASE_MODEL,
            adapter_sha256=ADAPTER_SHA,
            dataset_fingerprint=DATASET_FP,
            serving_name="",
        )


def test_adapter_binding_requires_an_adapter_hash() -> None:
    """这条判定守的是 ``from_dict`` 那条路径：还原出来的记录可能没有哈希."""
    with pytest.raises(ServingError, match="adapter 形态的部署记录必须有 adapter_sha256"):
        ServingBinding(
            spec=_spec(KIND_ADAPTER),
            version="1.1.0",
            version_key="x" * 16,
            base_model=BASE_MODEL,
            adapter_sha256="",
            dataset_fingerprint=DATASET_FP,
            serving_name="smart-research-qwen3-8b",
        )


def test_merged_binding_allows_no_adapter_hash() -> None:
    """合并模型就是没有运行时适配器——空哈希在 ``merged`` 形态下是合法的."""
    binding = ServingBinding(
        spec=_spec(KIND_MERGED),
        version="1.1.0",
        version_key="x" * 16,
        base_model=BASE_MODEL,
        adapter_sha256="",
        dataset_fingerprint=DATASET_FP,
        serving_name="smart-research-qwen3-8b",
    )
    assert binding.short_adapter == ""
    assert "（无适配器）" in binding.summary_line()


def test_binding_round_trips_through_dict() -> None:
    binding = bind_version(_spec(), _version())
    restored = ServingBinding.from_dict(json.loads(json.dumps(binding.to_dict())))
    assert restored == binding


def test_binding_from_dict_requires_the_spec() -> None:
    with pytest.raises(ServingError, match="部署记录里没有 spec"):
        ServingBinding.from_dict({"version": "1.1.0"})


def test_write_and_read_binding_round_trip(tmp_path) -> None:
    binding = bind_version(_spec(), _version(), commit="cf7b918")
    path = write_binding(tmp_path, binding)
    assert path.name == BINDING_FILE
    assert read_binding(tmp_path) == binding


def test_read_binding_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(ServingError, match="找不到部署记录"):
        read_binding(tmp_path)


# --------------------------------------------------------------------------- #
# 校验：五条检查
# --------------------------------------------------------------------------- #


def _endpoint(
    *,
    name: str = "smart-research-qwen3-8b",
    base_model: str = BASE_MODEL,
    adapter: str = ADAPTER_SHA[:SHORT_HASH_LENGTH],
) -> ServedEndpoint:
    return ServedEndpoint(name=name, base_model=base_model, adapter_short_hash=adapter)


def test_fully_consistent_binding_passes_and_is_consistent() -> None:
    binding = bind_version(_spec(), _version())
    report = verify_binding(binding, _endpoint(), head=_version())
    assert report.passed is True
    assert report.consistent is True
    assert report.blocking_failures == []
    assert len(report.checks) == 5
    assert report.summary_line().startswith("部署绑定 一致")
    assert "| 检查 | 实际 | 期望 | 一致 | 阻塞 | 理由 |" in report.render_markdown()


def test_name_mismatch_is_a_blocking_failure() -> None:
    """部署脚本改错了目标服务：改了 A 却以为改了 B."""
    binding = bind_version(_spec(), _version())
    report = verify_binding(binding, _endpoint(name="smart-research-other"), head=_version())
    assert report.passed is False
    assert [item.name for item in report.blocking_failures] == [CHECK_SERVING_NAME]
    assert "某一个 Pod 还是旧配置" in report.check(CHECK_SERVING_NAME).reason


def test_missing_self_report_is_a_blocking_failure_by_default() -> None:
    """缺证据不能判定一致——这是**绝对判定**，与 day059 的门禁同一侧."""
    binding = bind_version(_spec(), _version())
    report = verify_binding(
        binding, ServedEndpoint(name="smart-research-qwen3-8b"), head=_version()
    )
    assert report.passed is False
    names = [item.name for item in report.blocking_failures]
    assert CHECK_BASE_MODEL in names
    assert CHECK_ADAPTER_HASH in names


def test_disabling_self_report_turns_missing_evidence_into_a_warning() -> None:
    """关掉自述要求**不是"通过"**：它落进 ``warnings``，报告里看得见.

    用 ``merged`` 形态举例：它的端点本来就没有适配器可自述，
    因此"没自述基座"成了唯一的缺证据项——正好用来观察策略开关的效果。
    """
    binding = bind_version(_spec(KIND_MERGED), _version())
    report = verify_binding(
        binding,
        ServedEndpoint(name="smart-research-qwen3-8b"),
        head=_version(),
        policy=BindingPolicy(require_self_report=False),
    )
    assert report.passed is True
    assert report.consistent is False
    # 两项"端点没自述"都从"不一致"降级成"告警"，但**没有一项变成通过**
    assert [item.name for item in report.warnings] == [CHECK_BASE_MODEL, CHECK_ADAPTER_HASH]
    assert "有意的关闭，不是「一致」" in report.check(CHECK_BASE_MODEL).reason
    # 形态自洽（合并模型本就不该自述适配器）与其他检查仍按原样通过
    assert report.check(CHECK_KIND_SHAPE).passed is True
    assert report.check(CHECK_REGISTRY_HEAD).passed is True


def test_base_model_mismatch_is_caught() -> None:
    """适配器挂在了别的基座上：加载成功，但输出与训练行为无关."""
    binding = bind_version(_spec(), _version())
    report = verify_binding(
        binding, _endpoint(base_model="Qwen/Qwen3-8B"), head=_version()
    )
    assert report.passed is False
    assert [item.name for item in report.blocking_failures] == [CHECK_BASE_MODEL]
    assert "输出与训练时的行为不再相关" in report.check(CHECK_BASE_MODEL).reason


def test_adapter_hash_mismatch_is_caught() -> None:
    """线上挂的是另一份适配器——多数情况是 adapter_dir 指到了旧目录."""
    binding = bind_version(_spec(), _version())
    report = verify_binding(
        binding, _endpoint(adapter=OTHER_SHA[:SHORT_HASH_LENGTH]), head=_version()
    )
    assert report.passed is False
    assert [item.name for item in report.blocking_failures] == [CHECK_ADAPTER_HASH]
    assert "旧目录" in report.check(CHECK_ADAPTER_HASH).reason


def test_kind_shape_catches_a_merged_deployment_with_an_adapter() -> None:
    """形态与自述矛盾：合并形态却自述挂了适配器 → 端点加载的其实是别的目录.

    注意这里**只有** ``kind_shape`` 一条阻塞失败：适配器哈希那一项是**通过**的
    ——版本记录里的适配器哈希是权重的出身（合并模型由它合并而来），
    因此它在合并形态下仍然要被比对，而不是被跳过。
    """
    binding = bind_version(_spec(KIND_MERGED), _version())
    report = verify_binding(binding, _endpoint(), head=_version())
    assert report.passed is False
    names = [item.name for item in report.blocking_failures]
    assert names == [CHECK_KIND_SHAPE]
    assert "端点加载的其实是别的目录" in report.check(CHECK_KIND_SHAPE).reason
    hash_check = report.check(CHECK_ADAPTER_HASH)
    assert hash_check.passed is True
    assert hash_check.expected == ADAPTER_SHA[:SHORT_HASH_LENGTH]


def test_adapter_hash_check_is_not_applicable_when_the_binding_has_none() -> None:
    """``base`` 形态的绑定没有适配器哈希：那一项按**不适用**通过，而不是"没检查"."""
    binding = ServingBinding(
        spec=_spec(KIND_BASE),
        version="1.0.0",
        version_key=_version("1.0.0").version_key,
        base_model=BASE_MODEL,
        adapter_sha256="",
        dataset_fingerprint=DATASET_FP,
        serving_name="smart-research-base",
    )
    report = verify_binding(
        binding, ServedEndpoint(name="smart-research-base", base_model=BASE_MODEL)
    )
    hash_check = report.check(CHECK_ADAPTER_HASH)
    assert hash_check.passed is True
    assert hash_check.blocking is False
    assert hash_check.actual == "（本形态无适配器）"


def test_adapter_deployment_without_self_reported_adapter_fails_both_checks() -> None:
    binding = bind_version(_spec(), _version())
    report = verify_binding(binding, _endpoint(adapter=""), head=_version())
    assert report.passed is False
    assert CHECK_KIND_SHAPE in [item.name for item in report.blocking_failures]
    assert "形态与自述矛盾" in report.check(CHECK_KIND_SHAPE).reason


def test_missing_head_is_recorded_as_unchecked_but_not_blocking() -> None:
    """调用方没给注册表状态：记「未检查」，**缺证据不能算一致，但也不阻塞**."""
    binding = bind_version(_spec(), _version())
    report = verify_binding(binding, _endpoint())
    assert report.passed is True
    assert report.consistent is False
    head_check = report.check(CHECK_REGISTRY_HEAD)
    assert head_check.passed is False
    assert head_check.blocking is False
    assert head_check.actual == "未检查"
    assert "缺证据不能算一致" in head_check.reason


def test_other_head_is_a_warning_by_default_but_blocking_when_required() -> None:
    """绑定版本不是 head：灰度期间正常（告警），策略要求时才是阻塞项."""
    binding = bind_version(_spec(), _version("1.1.0"))
    newer = _version("1.2.0", adapter_sha256=OTHER_SHA)
    warn = verify_binding(binding, _endpoint(), head=newer)
    assert warn.passed is True
    assert CHECK_REGISTRY_HEAD in [item.name for item in warn.warnings]
    assert "灰度期间这是正常的" in warn.check(CHECK_REGISTRY_HEAD).reason

    strict = verify_binding(
        binding, _endpoint(), head=newer, policy=BindingPolicy(require_registry_head=True)
    )
    assert strict.passed is False
    assert [item.name for item in strict.blocking_failures] == [CHECK_REGISTRY_HEAD]
    assert strict.check(CHECK_REGISTRY_HEAD).blocking is True


def test_check_lookup_reports_unknown_names() -> None:
    report = verify_binding(bind_version(_spec(), _version()), _endpoint())
    with pytest.raises(ServingError, match="报告里没有检查项"):
        report.check("no_such_check")


def test_report_to_dict_separates_failures_warnings_and_all_checks() -> None:
    report = verify_binding(
        bind_version(_spec(), _version()),
        _endpoint(name="wrong"),
        policy=BindingPolicy(require_self_report=False),
    )
    payload = report.to_dict()
    assert payload["passed"] is False
    assert payload["consistent"] is False
    assert payload["blocking_failures"] == [CHECK_SERVING_NAME]
    assert payload["version"] == "1.1.0"
    assert payload["serving_name"] == "smart-research-qwen3-8b"
    assert len(payload["checks"]) == 5
    assert payload["checks"][0]["name"] == CHECK_SERVING_NAME


def test_endpoint_summary_marks_missing_self_report() -> None:
    """「没自述」必须能看出来——不能显示成空."""
    assert "（未自述）" in ServedEndpoint(name="x").summary_line()
    assert "sha256" not in ServedEndpoint(name="x").summary_line()
    assert ServedEndpoint(name="x", base_model="m", adapter_short_hash="abc").to_dict()[
        "adapter_short_hash"
    ] == "abc"


def test_binding_table_lists_five_failure_modes() -> None:
    rows = binding_table()
    assert [row["name"] for row in rows] == [
        CHECK_SERVING_NAME,
        CHECK_BASE_MODEL,
        CHECK_ADAPTER_HASH,
        CHECK_KIND_SHAPE,
        CHECK_REGISTRY_HEAD,
    ]
    assert all(row["failure_mode"] for row in rows)
    assert rows[-1]["blocking"] is False
    assert binding_table(BindingPolicy(require_registry_head=True))[-1]["blocking"] is True


def test_policy_to_dict_is_json_serializable() -> None:
    payload = BindingPolicy(require_self_report=False).to_dict()
    assert payload == {"require_self_report": False, "require_registry_head": False}
    assert json.loads(json.dumps(payload)) == payload


def test_binding_check_summary_names_both_state_and_blocking() -> None:
    """一行摘要要同时给出"一致不一致"与"阻塞不阻塞"——四字段缺一不可."""
    blocking = BindingCheck(
        name=CHECK_BASE_MODEL,
        passed=False,
        actual="Qwen/Qwen3-8B",
        expected=BASE_MODEL,
        reason="基座不一致",
    )
    assert "不一致/阻塞" in blocking.summary_line()
    assert "Qwen/Qwen3-8B" in blocking.summary_line()
    warning = BindingCheck(
        name=CHECK_REGISTRY_HEAD,
        passed=False,
        actual="未检查",
        expected="与注册表 head 比对",
        reason="缺证据",
        blocking=False,
    )
    assert "不一致/告警" in warning.summary_line()
    ok = BindingCheck(
        name=CHECK_SERVING_NAME, passed=True, actual="a", expected="a", reason="一致"
    )
    assert "一致/阻塞" in ok.summary_line()
    assert ok.to_dict()["passed"] is True


def test_candidate_stage_version_is_not_treated_as_head() -> None:
    """候选版本也能被绑定（灰度就是要部署非 stable），但 head 是 None 时如实记录."""
    binding = bind_version(_spec(), _version(stage=STAGE_CANDIDATE))
    report = verify_binding(binding, _endpoint(), head=None)
    assert report.passed is True
    assert report.check(CHECK_REGISTRY_HEAD).actual == "未检查"


def test_base_kind_deployment_binds_without_adapter_evidence() -> None:
    """``base`` 形态是回退目标：它同样需要三处一致性（只是没有适配器哈希）."""
    binding = ServingBinding(
        spec=_spec(KIND_BASE),
        version="1.0.0",
        version_key=_version("1.0.0").version_key,
        base_model=BASE_MODEL,
        adapter_sha256="",
        dataset_fingerprint=DATASET_FP,
        serving_name="smart-research-base",
    )
    report = verify_binding(
        binding,
        ServedEndpoint(name="smart-research-base", base_model=BASE_MODEL),
        head=_version("1.0.0"),
    )
    assert report.passed is True
    assert report.consistent is True
