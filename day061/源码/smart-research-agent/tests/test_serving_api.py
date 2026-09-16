"""day060 部署与切换端点测试（M5-D11）：/serving/* 四个端点.

全部走 TestClient（进程内 ASGI 调用）：不起真实服务、不联网、**不加载任何模型**、
不写盘、不用随机数。延迟由请求里送来的观测值决定，因此同一个请求体永远得到
同一个响应。

这个文件挑的是四件"会静默失效"的事来钉：

1. **对照表必须与代码同源**：形态表 == ``spec_table()``、流量表 == ``route_table()``、
   验证表 == ``verify_table()``、绑定表 == ``binding_table()``、价格表 == ``price_book()``，
   而显存分解里的每一个数字都能用 ``memory_breakdown()`` 复算；
2. **``passed`` 与 ``consistent`` 是两个结论**：绑定校验里"不是注册表 head"
   是告警而不是阻塞——合成一个布尔量就表达不出灰度期间最正常的那种状态；
3. **错误码语言**：未知机型 / 版本不存在 / 产物不完整 → 400，
   越界的请求形状（``gpu_utilization=1.5``、缺 ``bound_version``）→ 422；
4. **端点不猜事实**：缺回复、被标记为失败的用例都走**报告路径**
   （缺数据是结论，不是 500）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.routes import (
    SERVING_DEFAULT_ADAPTER_DIR,
    SERVING_DEFAULT_MERGED_DIR,
    serving_binding_policy_from_settings,
    serving_clock,
    serving_spec_from_settings,
    serving_traffic_from_settings,
    serving_verify_policy_from_settings,
)
from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.serving import (
    ARCH_QWEN3_8B,
    CHECK_LATENCY,
    DEFAULT_NUM_PARALLEL,
    HOURS_PER_MONTH,
    KINDS,
    SERVING_LIMITATIONS,
    SERVING_OUT_OF_SCOPE,
    TrafficPolicy,
    binding_table,
    memory_breakdown,
    price_book,
    route_table,
    spec_table,
    verify_table,
)

ADAPTER_SHA = "5c6d7e8f9012345678901234567890abcdef0123"
DATASET_FP = "aa77c31e90f4b258"
BASE_MODEL = "Qwen/Qwen3-0.6B"

#: 三条**三元组互不相同**的版本记录。这是必须的：注册表按三元组内容寻址，
#: 同一个三元组换一个版本号会被判成"版本键已占用"——而这正是 day058
#: "同一个东西不该有两个编号"那条纪律的执行结果。
_TRIPLES: dict[str, tuple[str, str]] = {
    "1.0.0": ("aaaa111122223333444455556666777788889999", "0011223344556677"),
    "1.1.0": (ADAPTER_SHA, DATASET_FP),
    "1.2.0": ("bbbb111122223333444455556666777788889999", "8899aabbccddeeff"),
}


@pytest.fixture
def client() -> TestClient:
    """注入 MockLLM 的离线测试客户端（/serving/* 端点本身不碰 LLM）."""
    return TestClient(create_app(llm=MockLLM(default="offline")))


def _version_payload(version: str = "1.1.0", *, stage: str = "stable") -> dict:
    adapter_sha256, dataset_fingerprint = _TRIPLES[version]
    return {
        "version": version,
        "base_model": BASE_MODEL,
        "adapter_sha256": adapter_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "stage": stage,
        "artifacts": {
            "adapter": SERVING_DEFAULT_ADAPTER_DIR,
            "merged": SERVING_DEFAULT_MERGED_DIR,
        },
        "metrics": {"eval_pass_rate": 0.72},
    }


def _binding_request(**overrides) -> dict:
    payload = {
        "versions": [_version_payload(), _version_payload("1.0.0")],
        "bound_version": "1.1.0",
        "served": {
            "name": settings.serving_name,
            "base_model": BASE_MODEL,
            "adapter_short_hash": ADAPTER_SHA[:12],
        },
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# 1. GET /serving/targets
# --------------------------------------------------------------------------- #


def test_targets_endpoint_returns_the_self_description(client: TestClient) -> None:
    resp = client.get("/serving/targets")
    assert resp.status_code == 200
    payload = resp.json()
    assert [row["kind"] for row in payload["kinds"]] == list(KINDS)
    assert payload["traffic"]["dedicated_ratio"] == settings.serving_dedicated_ratio
    assert payload["spec"]["kind"] == settings.serving_kind
    assert payload["spec"]["backend"] == settings.local_backend
    assert len(payload["verify_table"]) == 5
    assert len(payload["binding_table"]) == 5
    assert payload["out_of_scope"] == list(SERVING_OUT_OF_SCOPE)
    assert payload["limitations"] == list(SERVING_LIMITATIONS)


def test_targets_tables_are_the_same_objects_the_code_exposes(client: TestClient) -> None:
    """**文档与实现同源**：端点返回的三张表必须逐字等于代码里的那三张."""
    payload = client.get("/serving/targets").json()
    assert payload["kinds"] == spec_table()
    assert payload["traffic_table"] == route_table(serving_traffic_from_settings(None))
    assert payload["verify_table"] == verify_table(serving_verify_policy_from_settings(None))
    assert payload["binding_table"] == binding_table(
        serving_binding_policy_from_settings(None)
    )
    assert payload["price_book"] == price_book()


def test_targets_memory_breakdown_is_recomputable_and_sourced(client: TestClient) -> None:
    """显存里的每个数字都能用 ``memory_breakdown()`` 复算，且注明用了哪份结构."""
    payload = client.get("/serving/targets").json()
    spec = serving_spec_from_settings(None)
    expected = memory_breakdown(
        ARCH_QWEN3_8B,
        context_length=spec.context_length,
        max_parallel=spec.max_parallel,
        bits_per_parameter=spec.bits_per_parameter,
    )
    memory = payload["memory"]
    assert memory["kv_cache_mib"] == expected["kv_cache_mib"] == 576.0
    assert memory["kv_kib_per_token"] == 144.0
    assert memory["weights_bytes"] == expected["weights_bytes"]
    assert "Qwen3-8B" in memory["architecture_note"]
    assert memory["total_bytes"] == (
        memory["weights_bytes"] + memory["kv_cache_bytes"] + memory["adapter_bytes"]
    )


def test_targets_recommendation_flags_a_mismatch_with_the_configuration(
    client: TestClient,
) -> None:
    """缺省的 adapter 形态与"只有一个业务"的建议不一致——**这件事必须写出来**."""
    payload = client.get("/serving/targets").json()
    recommendation = payload["recommendation"]
    assert recommendation["kind"] == "merged"
    assert recommendation["configured_kind"] == "adapter"
    assert recommendation["matches_configured"] is False
    assert "部署链路与普通模型一致" in recommendation["reason"]


def test_targets_default_parallel_follows_the_serving_default(client: TestClient) -> None:
    payload = client.get("/serving/targets").json()
    assert payload["spec"]["max_parallel"] == DEFAULT_NUM_PARALLEL
    assert payload["memory"]["max_parallel"] == DEFAULT_NUM_PARALLEL


def test_price_book_rows_are_exposed_with_units(client: TestClient) -> None:
    rows = client.get("/serving/targets").json()["price_book"]
    assert {row["unit"] for row in rows} == {"USD/GPU·h", "USD/1K tokens"}
    assert any(row["value"] == 1.006 for row in rows)


# --------------------------------------------------------------------------- #
# 2. POST /serving/binding/verify
# --------------------------------------------------------------------------- #


def test_binding_verify_passes_when_all_three_places_agree(client: TestClient) -> None:
    resp = client.post("/serving/binding/verify", json=_binding_request())
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["passed"] is True
    assert payload["consistent"] is True
    assert payload["report"]["blocking_failures"] == []
    assert payload["binding"]["version"] == "1.1.0"
    assert payload["binding"]["spec"]["kind"] == "adapter"
    assert "部署绑定校验：通过" in payload["markdown"]


def test_binding_verify_reports_a_stale_adapter_as_a_blocking_failure(
    client: TestClient,
) -> None:
    """线上挂的是上一份适配器：绑定校验必须拦下，且理由要指出"旧目录"."""
    request = _binding_request(
        served={
            "name": settings.serving_name,
            "base_model": BASE_MODEL,
            "adapter_short_hash": "deadbeef0000",
        }
    )
    payload = client.post("/serving/binding/verify", json=request).json()
    assert payload["passed"] is False
    assert payload["report"]["blocking_failures"] == ["adapter_hash"]
    assert "旧目录" in payload["markdown"]


def test_binding_verify_treats_a_non_head_version_as_a_warning(
    client: TestClient,
) -> None:
    """绑定版本不是 head：``passed`` 仍为真，但 ``consistent`` 为假."""
    request = _binding_request(
        versions=[_version_payload("1.2.0"), _version_payload("1.1.0")],
        bound_version="1.1.0",
    )
    payload = client.post("/serving/binding/verify", json=request).json()
    assert payload["passed"] is True
    assert payload["consistent"] is False
    assert payload["report"]["warnings"] == ["registry_head"]
    assert payload["report"]["blocking_failures"] == []


def test_binding_verify_can_require_the_head(client: TestClient) -> None:
    request = _binding_request(
        versions=[_version_payload("1.2.0"), _version_payload("1.1.0")],
        bound_version="1.1.0",
        policy={"require_registry_head": True},
    )
    payload = client.post("/serving/binding/verify", json=request).json()
    assert payload["passed"] is False
    assert payload["report"]["blocking_failures"] == ["registry_head"]


def test_binding_verify_accepts_a_canary_service_name(client: TestClient) -> None:
    """灰度服务名与部署单元名不同：这是允许的，但记录里必须能看出来."""
    canary = f"{settings.serving_name}-canary"
    request = _binding_request(
        serving_name=canary,
        served={"name": canary, "base_model": BASE_MODEL, "adapter_short_hash": ADAPTER_SHA[:12]},
    )
    payload = client.post("/serving/binding/verify", json=request).json()
    assert payload["passed"] is True
    assert payload["binding"]["is_canary"] is True


def test_binding_verify_rejects_an_unknown_version_with_400(client: TestClient) -> None:
    payload = client.post(
        "/serving/binding/verify", json=_binding_request(bound_version="9.9.9")
    )
    assert payload.status_code == 400
    assert "不存在" in payload.json()["detail"]


def test_binding_verify_rejects_incomplete_artifacts_with_400(client: TestClient) -> None:
    version = _version_payload()
    version["artifacts"] = {"adapter": SERVING_DEFAULT_ADAPTER_DIR}
    payload = client.post(
        "/serving/binding/verify", json=_binding_request(versions=[version])
    )
    assert payload.status_code == 400
    assert "产物不完整" in payload.json()["detail"]


def test_binding_verify_rejects_a_base_model_mismatch_with_400(client: TestClient) -> None:
    request = _binding_request(spec={"base_model": "Qwen/Qwen3-8B"})
    payload = client.post("/serving/binding/verify", json=request)
    assert payload.status_code == 400
    assert "基座不匹配" in payload.json()["detail"]


def test_binding_verify_rejects_an_invalid_kind_with_400(client: TestClient) -> None:
    payload = client.post("/serving/binding/verify", json=_binding_request(spec={"kind": "int8"}))
    assert payload.status_code == 400
    assert "未知部署形态" in payload.json()["detail"]


def test_binding_verify_requires_a_bound_version_with_422(client: TestClient) -> None:
    payload = client.post(
        "/serving/binding/verify", json={"versions": [_version_payload()]}
    )
    assert payload.status_code == 422


# --------------------------------------------------------------------------- #
# 3. POST /serving/verify
# --------------------------------------------------------------------------- #


def _verify_request(**overrides) -> dict:
    payload = {
        "probes": [
            {"case_id": "dpo", "prompt": "什么是 DPO？", "must_contain": ["DPO"]},
            {"case_id": "lora", "prompt": "什么是 LoRA？", "must_contain": ["LoRA"]},
        ],
        "cloud": {
            "replies": {"dpo": "DPO 是直接偏好优化", "lora": "LoRA 是低秩适配"},
            "latency_ms": {"dpo": 100.0, "lora": 120.0},
        },
        "dedicated": {
            "replies": {"dpo": "DPO 是直接偏好优化", "lora": "LoRA 是低秩适配"},
            "latency_ms": {"dpo": 250.0, "lora": 250.0},
        },
    }
    payload.update(overrides)
    return payload


def test_verify_endpoint_compares_the_two_arms(client: TestClient) -> None:
    payload = client.post("/serving/verify", json=_verify_request()).json()
    report = payload["report"]
    assert report["passed"] is True
    assert report["cases"] == 2
    assert report["cloud_pass_rate"] == 1.0
    assert report["dedicated_pass_rate"] == 1.0
    assert report["delta"] == 0.0
    assert report["skipped"] == [CHECK_LATENCY]
    assert "上线验证 通过" in payload["summary"]


def test_verify_endpoint_latency_comes_from_the_request(client: TestClient) -> None:
    """延迟是**请求里送来的观测值**：确定性时钟让报告里的数字精确可复现.

    云端中位数 110ms、专属 250ms → 倍数 2.2727…；策略要求 <= 3.0 时通过。
    """
    request = _verify_request(policy={"max_latency_ratio": 3.0})
    report = client.post("/serving/verify", json=request).json()["report"]
    assert report["cloud_latency_ms"] == 110.0
    assert report["dedicated_latency_ms"] == 250.0
    assert report["latency_ratio"] == round(250.0 / 110.0, 6)
    assert report["skipped"] == []
    assert report["passed"] is True


def test_verify_endpoint_blocks_a_slow_dedicated_model(client: TestClient) -> None:
    request = _verify_request(policy={"max_latency_ratio": 1.5})
    report = client.post("/serving/verify", json=request).json()["report"]
    assert report["passed"] is False
    assert report["blocking_failures"] == [CHECK_LATENCY]


def test_verify_endpoint_captures_calling_failures_in_the_report(client: TestClient) -> None:
    """被标记为失败的用例走**报告路径**（缺数据是结论，不是 500）."""
    request = _verify_request(
        dedicated={
            "replies": {"dpo": "DPO 是直接偏好优化"},
            "error_cases": ["lora"],
        }
    )
    resp = client.post("/serving/verify", json=request)
    assert resp.status_code == 200
    report = resp.json()["report"]
    assert report["dedicated_errors"] == 1
    assert "errors" in report["blocking_failures"]


def test_verify_endpoint_treats_a_missing_reply_as_a_failure(client: TestClient) -> None:
    """调用方忘了填某条回复时**必须响亮**：返回空串会把"漏填"伪装成"答错了"."""
    request = _verify_request(
        dedicated={"replies": {"dpo": "DPO 是直接偏好优化"}, "latency_ms": {}}
    )
    resp = client.post("/serving/verify", json=request)
    report = resp.json()["report"]
    assert report["dedicated_errors"] == 1
    assert "缺少 专属 臂的回复" in resp.json()["markdown"]


def test_verify_endpoint_rejects_a_regression(client: TestClient) -> None:
    request = _verify_request(
        dedicated={"replies": {"dpo": "不知道", "lora": "LoRA 是低秩适配"}}
    )
    report = client.post("/serving/verify", json=request).json()["report"]
    assert report["passed"] is False
    assert report["blocking_failures"] == ["regression"]
    assert report["delta"] == -0.5


def test_verify_endpoint_without_probes_is_a_failure_not_an_error(client: TestClient) -> None:
    report = client.post("/serving/verify", json={"probes": []}).json()["report"]
    assert report["passed"] is False
    assert report["blocking_failures"] == ["cases"]


def test_verify_endpoint_rejects_an_illegal_policy_with_400(client: TestClient) -> None:
    payload = client.post(
        "/serving/verify", json=_verify_request(policy={"min_pass_rate": 2.0})
    )
    assert payload.status_code == 400
    assert "min_pass_rate 必须落在" in payload.json()["detail"]


def test_verify_endpoint_rejects_an_empty_prompt_with_400(client: TestClient) -> None:
    payload = client.post(
        "/serving/verify",
        json=_verify_request(probes=[{"case_id": "x", "prompt": ""}]),
    )
    assert payload.status_code == 400
    assert "prompt 不能为空" in payload.json()["detail"]


def test_verify_endpoint_markdown_has_per_case_rows(client: TestClient) -> None:
    payload = client.post("/serving/verify", json=_verify_request()).json()
    assert "## 逐条结果" in payload["markdown"]
    assert "`dpo`" in payload["markdown"]


# --------------------------------------------------------------------------- #
# 4. POST /serving/cost/compare
# --------------------------------------------------------------------------- #


def test_cost_compare_uses_settings_by_default(client: TestClient) -> None:
    payload = client.post("/serving/cost/compare", json={}).json()
    comparison = payload["comparison"]
    assert comparison["requests"] == settings.serving_monthly_requests
    assert comparison["monthly_hours"] == HOURS_PER_MONTH
    assert comparison["gpu"]["name"] == settings.serving_gpu_key
    assert comparison["cloud"]["name"] == settings.serving_cloud_pricing_key
    assert comparison["machines"] >= 1
    assert "盈亏平衡" in payload["summary"]


def test_cost_compare_reports_cloud_cheaper_at_low_traffic(client: TestClient) -> None:
    """缺省场景的实测结论：10 万请求/月时云端 $52.50，自建 $734.38（差 14 倍）."""
    comparison = client.post("/serving/cost/compare", json={}).json()["comparison"]
    assert comparison["cloud_cost_usd"] == 52.5
    assert comparison["dedicated_cost_usd"] == 734.38
    assert comparison["cheaper_side"] == "cloud"
    assert comparison["utilization_actual"] < 0.5


def test_cost_compare_accepts_explicit_measurements(client: TestClient) -> None:
    payload = client.post(
        "/serving/cost/compare",
        json={
            "requests": 50_000_000,
            "input_tokens_per_request": 1500,
            "output_tokens_per_request": 500,
            "gpu_key": "runpod-pod-l4",
            "gpu_tokens_per_second": 100_000.0,
            "gpu_utilization": 1.0,
            "cloud_key": "gpt-4o-mini",
            "monthly_hours": 730.0,
        },
    ).json()
    comparison = payload["comparison"]
    assert comparison["cheaper_side"] == "dedicated"
    assert comparison["gpu"]["name"] == "runpod-pod-l4"
    assert comparison["dedicated_cost_per_1k_tokens"] is not None
    assert "自建更省" in payload["markdown"]


def test_cost_compare_rejects_an_unknown_gpu_key_with_400(client: TestClient) -> None:
    resp = client.post("/serving/cost/compare", json={"gpu_key": "my-laptop"})
    assert resp.status_code == 400
    assert "未知的 GPU 机型" in resp.json()["detail"]


def test_cost_compare_rejects_an_unknown_cloud_key_with_400(client: TestClient) -> None:
    resp = client.post("/serving/cost/compare", json={"cloud_key": "gpt-9"})
    assert resp.status_code == 400
    assert "未知的云端计价方案" in resp.json()["detail"]


def test_cost_compare_rejects_an_out_of_range_utilization_with_422(
    client: TestClient,
) -> None:
    assert client.post("/serving/cost/compare", json={"gpu_utilization": 1.5}).status_code == 422
    assert client.post("/serving/cost/compare", json={"requests": -1}).status_code == 422


def test_cost_compare_zero_requests_still_bills_the_month(client: TestClient) -> None:
    comparison = client.post(
        "/serving/cost/compare", json={"requests": 0}
    ).json()["comparison"]
    assert comparison["cloud_cost_usd"] == 0.0
    assert comparison["dedicated_cost_usd"] > 0


# --------------------------------------------------------------------------- #
# 辅助函数与配置装配（与端点同源）
# --------------------------------------------------------------------------- #


def test_spec_from_settings_fills_artifact_paths_for_every_kind() -> None:
    """三种形态都能从配置装配出来——**必填字段由 settings 提供，不靠调用方**."""
    for kind in KINDS:
        spec = serving_spec_from_settings({"kind": kind})
        assert spec.kind == kind
        if kind == "base":
            assert spec.artifact_path == ""
        elif kind == "adapter":
            assert spec.artifact_path == SERVING_DEFAULT_ADAPTER_DIR
        else:
            assert spec.artifact_path == SERVING_DEFAULT_MERGED_DIR


def test_spec_from_settings_ignores_unknown_keys() -> None:
    spec = serving_spec_from_settings({"whatever": 1, "name": "custom"})
    assert spec.name == "custom"


def test_traffic_policy_from_settings_applies_overrides() -> None:
    policy = serving_traffic_from_settings({"dedicated_ratio": 0.5, "unknown": 1})
    assert policy.dedicated_ratio == 0.5
    assert policy.shadow_ratio == settings.serving_shadow_ratio
    assert serving_traffic_from_settings(None) == TrafficPolicy(
        dedicated_ratio=settings.serving_dedicated_ratio,
        shadow_ratio=settings.serving_shadow_ratio,
        fail_open=settings.serving_fail_open,
    )


def test_verify_policy_from_settings_only_checks_latency_when_asked() -> None:
    """延迟倍数在设置里缺省为 ``None``（不检查）；请求里给出才生效."""
    assert serving_verify_policy_from_settings(None).max_latency_ratio is None
    assert serving_verify_policy_from_settings({"max_latency_ratio": 2.0}).max_latency_ratio == 2.0


def test_binding_policy_from_settings_has_no_deployment_level_defaults() -> None:
    """两个开关刻意没有部署级缺省值：只有请求里显式给出时才生效."""
    default = serving_binding_policy_from_settings(None)
    assert default.require_self_report is True
    assert default.require_registry_head is False
    strict = serving_binding_policy_from_settings({"require_registry_head": True})
    assert strict.require_registry_head is True


def test_serving_clock_reproduces_the_requested_latencies() -> None:
    """确定性时钟：每个用例消费两次 tick（开始/结束），余下的 tick 停在最后一个值."""
    clock = serving_clock([100.0, 200.0])
    values = [clock() for _ in range(6)]
    assert values == pytest.approx([0.0, 0.1, 0.1, 0.3, 0.3, 0.3])
