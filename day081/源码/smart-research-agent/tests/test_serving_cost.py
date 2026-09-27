"""day060 ``serving.cost`` 的单元测试：两个成本模型与盈亏平衡点."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.serving.cost import (
    CLOUD_PRICING,
    GPU_HOURLY_USD,
    HOURS_PER_MONTH,
    SECONDS_PER_HOUR,
    SIDE_CLOUD,
    SIDE_DEDICATED,
    SIDE_TIE,
    CloudPricing,
    GpuPricing,
    breakeven_requests,
    cloud_pricing,
    compare_costs,
    gpu_pricing,
    price_book,
)
from smart_research_agent.serving.errors import ServingError


def _cloud(input_price: float = 0.00015, output_price: float = 0.0006) -> CloudPricing:
    return CloudPricing(
        name="test-cloud",
        usd_per_1k_input_tokens=input_price,
        usd_per_1k_output_tokens=output_price,
    )


def _gpu(
    *, usd_per_hour: float = 1.006, tokens_per_second: float = 250.0, utilization: float = 1.0
) -> GpuPricing:
    return GpuPricing(
        name="test-gpu",
        usd_per_hour=usd_per_hour,
        tokens_per_second=tokens_per_second,
        utilization=utilization,
    )


# --------------------------------------------------------------------------- #
# 价格表：外部事实只有一个出处
# --------------------------------------------------------------------------- #


def test_hours_per_month_is_the_calendar_average() -> None:
    """``365 × 24 / 12 = 730``：用 720 会让年化成本少算 1.4%."""
    assert HOURS_PER_MONTH == 730.0
    assert SECONDS_PER_HOUR == 3600.0


def test_gpu_price_book_matches_the_verified_numbers() -> None:
    """价格是**外部事实**，因此它必须集中存放、可被核对（2026-09 核对）."""
    assert GPU_HOURLY_USD["aws-g5-xlarge-a10g"] == 1.006
    assert GPU_HOURLY_USD["aws-g4dn-xlarge-t4"] == 0.526
    assert GPU_HOURLY_USD["gcp-t4-gpu-only"] == 0.35
    assert GPU_HOURLY_USD["runpod-pod-l4"] == 0.49
    assert GPU_HOURLY_USD["lambda-a10"] == 1.29


def test_cloud_price_book_splits_input_and_output() -> None:
    """输入与输出分开计价：两者常差 4 倍，合成一个平均价会让对比失去可调性."""
    assert CLOUD_PRICING["gpt-4o-mini"] == (0.00015, 0.0006)
    assert CLOUD_PRICING["gpt-4o"] == (0.0025, 0.01)


def test_gpu_pricing_looks_up_the_price_book() -> None:
    pricing = gpu_pricing("runpod-pod-l4", tokens_per_second=400.0, utilization=0.8)
    assert pricing.usd_per_hour == 0.49
    assert pricing.tokens_per_second == 400.0
    assert pricing.effective_tokens_per_second == 320.0


def test_unknown_gpu_key_is_rejected_with_the_option_list() -> None:
    """"你写的机型不在价格表里"必须响亮：否则会用 0 元时价算出"自建免费"."""
    with pytest.raises(ServingError, match="未知的 GPU 机型"):
        gpu_pricing("my-laptop", tokens_per_second=10.0)


def test_unknown_cloud_key_is_rejected() -> None:
    with pytest.raises(ServingError, match="未知的云端计价方案"):
        cloud_pricing("gpt-9")


def test_cloud_pricing_looks_up_the_price_book() -> None:
    pricing = cloud_pricing("gpt-4o")
    assert pricing.usd_per_1k_input_tokens == 0.0025
    assert pricing.usd_per_1k_output_tokens == 0.01


def test_price_book_rows_document_their_own_caveats() -> None:
    rows = price_book()
    assert {row["kind"] for row in rows} == {"gpu", "cloud"}
    gpu_row = next(row for row in rows if row["name"] == "aws-g5-xlarge-a10g")
    assert gpu_row["unit"] == "USD/GPU·h"
    assert gpu_row["value"] == 1.006
    assert "不含 VM 机型费用" in gpu_row["note"]
    cloud_row = next(row for row in rows if row["name"] == "gpt-4o-mini")
    assert "分开计价" in cloud_row["note"]


# --------------------------------------------------------------------------- #
# 两个计价对象
# --------------------------------------------------------------------------- #


def test_cloud_pricing_validates_its_fields() -> None:
    with pytest.raises(ServingError, match="计价方案需要名字"):
        CloudPricing(name="", usd_per_1k_input_tokens=1.0, usd_per_1k_output_tokens=1.0)
    with pytest.raises(ServingError, match="usd_per_1k_input_tokens 不能为负数"):
        CloudPricing(name="c", usd_per_1k_input_tokens=-1.0, usd_per_1k_output_tokens=1.0)
    with pytest.raises(ServingError, match="usd_per_1k_output_tokens 不能为负数"):
        CloudPricing(name="c", usd_per_1k_input_tokens=1.0, usd_per_1k_output_tokens=-1.0)


def test_cloud_cost_splits_input_and_output() -> None:
    pricing = _cloud()
    # 1500 入 + 500 出：1500/1000*0.00015 + 500/1000*0.0006 = 0.000225 + 0.0003
    assert pricing.cost_usd(1500, 500) == pytest.approx(0.000525)
    assert pricing.cost_usd(0, 0) == 0.0


def test_cloud_cost_rejects_negative_tokens() -> None:
    with pytest.raises(ServingError, match="token 数不能为负数"):
        _cloud().cost_usd(-1, 0)


def test_gpu_pricing_requires_a_measured_throughput() -> None:
    """**吞吐没有缺省值**：它取决于模型、量化、批大小与序列长度，必须实测."""
    with pytest.raises(ServingError, match="tokens_per_second 必须为正数"):
        _gpu(tokens_per_second=0.0)
    with pytest.raises(ServingError, match="GPU 方案需要名字"):
        GpuPricing(name="", usd_per_hour=1.0, tokens_per_second=1.0)


def test_gpu_pricing_validates_hourly_price_and_utilization() -> None:
    with pytest.raises(ServingError, match="usd_per_hour 必须为正数"):
        _gpu(usd_per_hour=0.0)
    with pytest.raises(ServingError, match=r"utilization 必须落在 \(0, 1\]"):
        _gpu(utilization=0.0)
    with pytest.raises(ServingError, match=r"utilization 必须落在 \(0, 1\]"):
        _gpu(utilization=1.5)


def test_gpu_hours_scale_with_tokens_and_utilization() -> None:
    pricing = _gpu(tokens_per_second=100.0, utilization=0.5)
    # 有效吞吐 50 token/s，一小时能处理 180000 token
    assert pricing.hours_for(180_000) == pytest.approx(1.0)
    assert pricing.hours_for(0) == 0.0
    with pytest.raises(ServingError, match="总 token 数不能为负数"):
        pricing.hours_for(-1)


def test_gpu_cost_is_billed_by_hours_not_tokens() -> None:
    """**这条是本模块的核心**：账单按时长走，与这一小时处理了多少 token 无关."""
    pricing = _gpu(usd_per_hour=1.006)
    assert pricing.cost_usd(0.0) == 0.0
    assert pricing.cost_usd(730.0) == pytest.approx(734.38)
    with pytest.raises(ServingError, match="GPU 小时数不能为负数"):
        pricing.cost_usd(-1.0)


def test_pricing_to_dict_carries_derivations() -> None:
    assert _gpu(tokens_per_second=100.0, utilization=0.5).to_dict()[
        "effective_tokens_per_second"
    ] == 50.0
    assert _cloud().to_dict()["name"] == "test-cloud"


# --------------------------------------------------------------------------- #
# 盈亏平衡
# --------------------------------------------------------------------------- #


def test_breakeven_is_fixed_cost_divided_by_cloud_cost_per_request() -> None:
    cloud = _cloud(input_price=0.5, output_price=0.0)
    gpu = _gpu(usd_per_hour=1.0, tokens_per_second=1_000_000.0)
    # 自建月费 = 730 × $1.0 = $730；单次云端费用 = 1000/1000 × $0.5 = $0.5
    assert breakeven_requests(
        cloud, gpu, input_tokens_per_request=1000, output_tokens_per_request=0
    ) == pytest.approx(1460.0)


def test_breakeven_validates_its_inputs() -> None:
    cloud, gpu = _cloud(), _gpu()
    with pytest.raises(ServingError, match="machines 至少为 1"):
        breakeven_requests(
            cloud, gpu, input_tokens_per_request=1, output_tokens_per_request=1, machines=0
        )
    with pytest.raises(ServingError, match="monthly_hours 必须为正数"):
        breakeven_requests(
            cloud, gpu, input_tokens_per_request=1, output_tokens_per_request=1, monthly_hours=0
        )


def test_breakeven_is_infinite_when_the_cloud_is_free() -> None:
    """云端单价为 0 时，自建的固定成本**无法被任何请求量摊平**（不是 0，也不是负）."""
    free = CloudPricing(name="free", usd_per_1k_input_tokens=0.0, usd_per_1k_output_tokens=0.0)
    assert breakeven_requests(
        free, _gpu(), input_tokens_per_request=1000, output_tokens_per_request=100
    ) == float("inf")


# --------------------------------------------------------------------------- #
# 对比
# --------------------------------------------------------------------------- #


def test_low_traffic_makes_self_hosting_clearly_more_expensive() -> None:
    """**本课程最重要的一条结论**：请求量低时自建一定更贵，因为账单不随调用量下降.

    实测（本课缺省参数）：10 万请求/月，云端 $52.50，自建 $734.38——
    差 14 倍，而盈亏平衡点在 140 万请求/月。
    """
    comparison = compare_costs(
        requests=100_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0, utilization=1.0),
    )
    assert comparison.cloud_cost_usd == pytest.approx(52.5)
    assert comparison.dedicated_cost_usd == pytest.approx(734.38)
    assert comparison.cheaper_side == SIDE_CLOUD
    assert comparison.saving_usd < 0
    assert comparison.machines == 1
    assert comparison.utilization_actual < 0.5
    assert comparison.breakeven_requests > 1_000_000


def test_high_throughput_at_scale_makes_self_hosting_cheaper() -> None:
    """吞吐足够高、请求量足够大时结论反过来——**这两个数字必须一起看**."""
    comparison = compare_costs(
        requests=50_000_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=100_000.0, utilization=1.0),
    )
    assert comparison.cheaper_side == SIDE_DEDICATED
    assert comparison.saving_usd > 0
    assert comparison.saving_ratio is not None and comparison.saving_ratio > 0.5
    assert comparison.dedicated_cost_per_1k_tokens is not None
    assert comparison.dedicated_cost_per_1k_tokens < 0.000525


def test_capacity_is_met_by_opening_more_machines() -> None:
    """用掉的算力超过一台机器能提供的时长时**必须开第二台**，而第二台同样按月付费.

    这一步把"自建 = 一台机器"这个隐性假设显式化了：不这么做会算出
    "用一台机器的时价承接十台机器的负载"这种部署不出来的方案。
    """
    comparison = compare_costs(
        requests=10_000_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0, utilization=1.0),
    )
    assert comparison.machines > 1
    assert comparison.deployed_gpu_hours == pytest.approx(comparison.machines * HOURS_PER_MONTH)
    assert comparison.required_gpu_hours > HOURS_PER_MONTH
    assert comparison.dedicated_cost_usd == pytest.approx(
        comparison.machines * HOURS_PER_MONTH * 1.006
    )


def test_zero_requests_still_costs_a_full_month() -> None:
    """**空转的 GPU 是最贵的**：0 请求时云端不花钱，自建照付一个月."""
    comparison = compare_costs(
        requests=0,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0),
    )
    assert comparison.cloud_cost_usd == 0.0
    assert comparison.dedicated_cost_usd == pytest.approx(734.38)
    assert comparison.cheaper_side == SIDE_CLOUD
    assert comparison.saving_ratio is None
    assert comparison.dedicated_cost_per_1k_tokens is None
    assert comparison.cloud_cost_per_request == 0.0
    assert comparison.utilization_actual == 0.0
    assert comparison.machines == 1
    # 盈亏平衡点**与当前请求量无关**：它问的是"要到多少才划得来"，
    # 而 0 请求时的答案与 10 万请求时完全相同（都是 139.9 万）
    assert comparison.breakeven_requests == pytest.approx(1_398_819.05, rel=1e-6)


def test_exact_tie_is_reported_as_a_tie() -> None:
    """两侧完全相等时既不说"更省"也不说"更贵"——**平局是一个独立结论**."""
    comparison = compare_costs(
        requests=2000,
        input_tokens_per_request=1000,
        output_tokens_per_request=0,
        cloud=CloudPricing(
            name="flat", usd_per_1k_input_tokens=0.5, usd_per_1k_output_tokens=0.0
        ),
        gpu=GpuPricing(name="flat-gpu", usd_per_hour=1.0, tokens_per_second=1_000_000.0),
        monthly_hours=1000.0,
    )
    assert comparison.cloud_cost_usd == pytest.approx(1000.0)
    assert comparison.dedicated_cost_usd == pytest.approx(1000.0)
    assert comparison.cheaper_side == SIDE_TIE
    assert comparison.saving_usd == pytest.approx(0.0)


def test_compare_costs_validates_its_inputs() -> None:
    cloud, gpu = _cloud(), _gpu()
    with pytest.raises(ServingError, match="requests 不能为负数"):
        compare_costs(
            requests=-1, input_tokens_per_request=1, output_tokens_per_request=1,
            cloud=cloud, gpu=gpu,
        )
    with pytest.raises(ServingError, match="每请求 token 数不能为负数"):
        compare_costs(
            requests=1, input_tokens_per_request=-1, output_tokens_per_request=1,
            cloud=cloud, gpu=gpu,
        )
    with pytest.raises(ServingError, match="monthly_hours 必须为正数"):
        compare_costs(
            requests=1, input_tokens_per_request=1, output_tokens_per_request=1,
            cloud=cloud, gpu=gpu, monthly_hours=-1,
        )


def test_utilization_raises_the_effective_unit_cost() -> None:
    """利用率减半 → 单位成本翻倍：这就是"空转的 GPU 最贵"的算术形式."""
    full = compare_costs(
        requests=1_000_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=1000.0, utilization=1.0),
    )
    half = compare_costs(
        requests=1_000_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=1000.0, utilization=0.5),
    )
    assert full.dedicated_cost_per_1k_tokens is not None
    assert half.dedicated_cost_per_1k_tokens is not None
    assert half.dedicated_cost_per_1k_tokens == pytest.approx(
        full.dedicated_cost_per_1k_tokens * 2
    )


def test_breakeven_and_unit_cost_ratio_are_the_same_number() -> None:
    """**两个数字互相校验**：``盈亏平衡 / 当前请求量 == 自建单位成本 / 云端单位成本``.

    这条恒等式把报告里最容易出错的几个乘数（机器数、月时长、时价、每请求 token）
    绑在一起：任何一个算错，两个比值都会分叉。实测 100000 请求/月时两者
    都是 **13.99**。
    """
    comparison = compare_costs(
        requests=100_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0),
    )
    dedicated_per_1k = comparison.dedicated_cost_per_1k_tokens
    assert dedicated_per_1k is not None
    cloud_per_1k = comparison.cloud_cost_usd / (comparison.total_tokens / 1000)
    assert dedicated_per_1k / cloud_per_1k == pytest.approx(
        comparison.breakeven_requests / comparison.requests, rel=1e-9
    )
    assert dedicated_per_1k / cloud_per_1k == pytest.approx(13.99, abs=0.01)


def test_price_book_rounds_the_combined_cloud_number() -> None:
    """``0.00015 + 0.0006`` 打印成 ``0.0007499999999999999`` 会看起来像 bug."""
    row = next(
        item for item in price_book() if item["name"] == "gpt-4o-mini"
    )
    assert row["value"] == 0.00075


def test_comparison_to_dict_is_json_serializable_and_rounds() -> None:
    comparison = compare_costs(
        requests=100_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0),
    )
    payload = comparison.to_dict()
    assert json.loads(json.dumps(payload))["requests"] == 100_000
    assert payload["cheaper_side"] == "cloud"
    assert payload["cloud_cost_usd"] == 52.5
    assert payload["machines"] == 1
    assert payload["dedicated_cost_per_1k_tokens"] is not None
    assert payload["saving_ratio"] is not None


def test_comparison_markdown_documents_both_reports() -> None:
    comparison = compare_costs(
        requests=100_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0),
    )
    text = comparison.render_markdown()
    assert "# 专属模型部署成本对比" in text
    assert "- 用掉的算力：" in text
    assert "- 为之付费的时长：" in text
    assert "- 实际利用率：" in text
    assert "- 盈亏平衡调用量：" in text
    assert "云端更省" in text


def test_comparison_markdown_handles_the_empty_batch() -> None:
    """0 请求时表格里的"单次请求"与"单位成本"必须是 ``—``，而不是除零崩溃."""
    comparison = compare_costs(
        requests=0,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0),
    )
    text = comparison.render_markdown()
    assert "| 单次请求 | — | — |" in text
    assert "| 单位成本（$/1K token） | — | — |" in text


def test_comparison_summary_line_names_the_cheaper_side() -> None:
    cloud_cheaper = compare_costs(
        requests=1000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud_pricing("gpt-4o-mini"),
        gpu=gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0),
    )
    assert "云端更省" in cloud_cheaper.summary_line()
    assert "盈亏平衡" in cloud_cheaper.summary_line()
