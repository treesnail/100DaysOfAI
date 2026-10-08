#!/usr/bin/env python
"""day060 演示脚本：专属模型部署替换（M5-D11）.

八节，全部**离线、确定性、零 GPU、零网络、零随机**：

```text
1. 形态表与一个被拦下的部署      声明成 adapter 却没给路径 → 构造期失败
2. 部署绑定：三处版本比对        注册表 / 部署记录 / 端点自述
3. 显存算术                      权重 / KV 缓存 / 适配器三块分开列
4. 流量切换                      确定性分桶、影子流量、单向降级
5. 上线验证                      同一批用例上两个推理臂的对比
6. 成本对比                      固定成本 vs 可变成本、盈亏平衡点
7. 灰度的四步时间线              0 → 0.05 → 0.5 → 1.0
8. 四个 HTTP 端点                进程内 ASGI 调用，不加载任何模型
```

运行::

    PYTHONPATH=. python scripts/serving_demo.py
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.registry.record import STAGE_STABLE, ModelVersion
from smart_research_agent.registry.version import VersionTriple
from smart_research_agent.serving import (
    ARCH_QWEN3_8B,
    BindingPolicy,
    ModelSwitcher,
    ProbeCase,
    ServedEndpoint,
    ServingSpec,
    TrafficPolicy,
    VerifyPolicy,
    bind_version,
    binding_table,
    cloud_pricing,
    compare_costs,
    gpu_pricing,
    memory_breakdown,
    price_book,
    recommend_kind,
    route_table,
    run_verification,
    spec_table,
    split_summary,
    verify_binding,
    verify_table,
)

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "5c6d7e8f9012345678901234567890abcdef0123"
DATASET = "aa77c31e90f4b258"
SERVING_NAME = "smart-research-qwen3-8b"


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


class FlakyLLM(BaseLLM):
    """一个会按脚本失败的叶子模型（模拟"专属服务没起来"）."""

    def __init__(self, name: str, *, reply: str, fail_times: int = 0) -> None:
        self.name = name
        self.reply = reply
        self.fail_times = fail_times
        self.calls = 0

    def chat(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1024
    ) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"{self.name} 暂时不可用（第 {self.calls} 次调用）")
        return self.reply

    def stream(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1024
    ) -> Iterator[str]:
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens)


def _version(version: str = "1.1.0", *, adapter: str = ADAPTER, dataset: str = DATASET):
    return ModelVersion(
        triple=VersionTriple(
            base_model=BASE_MODEL, adapter_sha256=adapter, dataset_fingerprint=dataset
        ),
        version=version,
        stage=STAGE_STABLE,
        artifacts={
            "adapter": "outputs/lora/adapters/adapter-final",
            "merged": "outputs/lora-merged",
        },
        metrics={"eval_pass_rate": 0.72},
    )


def section_1_kinds() -> None:
    """第 1 节：三种形态，以及一个在构造期就被拦下的部署."""
    title("1. 部署形态：声明成 adapter 却没给路径 → 构造期就失败")
    for row in spec_table():
        print(
            f"  {row['kind']:<8} 必填 {row['required_fields_text']:<14} "
            f"{row['runtime_dependency']}"
        )
        print(f"           换业务：{row['switch_cost']}")
    print()
    for variants, hot in ((1, False), (1, True), (4, False)):
        kind, reason = recommend_kind(variants=variants, needs_hot_swap=hot)
        print(f"  业务数 {variants} / 需热插拔 {str(hot):<5} → {kind:<8} {reason}")
    print()
    try:
        ServingSpec(name=SERVING_NAME, kind="adapter")
    except Exception as exc:  # noqa: BLE001 - 演示就是要打印这条错误
        print(f"  声明 adapter 却没给目录被拒：{exc}")
    print("  （这类部署不会报错，它只会安静地服务基座——这正是它必须在构造期被拦住的原因）")


def section_2_binding() -> None:
    """第 2 节：三处版本比对."""
    title("2. 部署绑定：注册表 / 部署记录 / 端点自述，三处必须指向同一份产物")
    spec = ServingSpec(
        name=SERVING_NAME,
        kind="adapter",
        base_model=BASE_MODEL,
        adapter_dir="outputs/lora/adapters/adapter-final",
    )
    version = _version()
    binding = bind_version(spec, version, commit="cf7b918")
    print(f"  绑定记录：{binding.summary_line()}")
    print()
    print("  五条一致性检查挡住的故障：")
    for row in binding_table():
        flag = "阻塞" if row["blocking"] else "告警"
        print(f"    [{flag}] {row['name']:<20} {row['failure_mode']}")
    print()
    good = ServedEndpoint(
        name=SERVING_NAME, base_model=BASE_MODEL, adapter_short_hash=ADAPTER[:12]
    )
    stale = ServedEndpoint(
        name=SERVING_NAME, base_model=BASE_MODEL, adapter_short_hash="deadbeef0000"
    )
    old_head = _version("1.2.0", adapter="bbbb111122223333444455556666777788889999",
                        dataset="8899aabbccddeeff")
    cases = (
        ("端点自述与记录一致", good, version, BindingPolicy()),
        ("线上挂的是上一份适配器", stale, version, BindingPolicy()),
        ("绑定版本不是注册表 head", good, old_head, BindingPolicy()),
        ("严格模式：必须部署 head", good, old_head, BindingPolicy(require_registry_head=True)),
    )
    for label, served, head, policy in cases:
        report = verify_binding(binding, served, head=head, policy=policy)
        print(
            f"  {label:<22} passed={str(report.passed):<5} "
            f"consistent={str(report.consistent):<5} "
            f"阻塞失败={[item.name for item in report.blocking_failures] or '无'}"
        )
        for check in report.checks:
            if not check.passed:
                print(f"      └ {check.summary_line()}")


def section_3_memory() -> None:
    """第 3 节：显存算术."""
    title("3. 显存算术：权重 / KV 缓存 / 适配器三块分开列（Qwen3-8B 的公开结构）")
    for parallel in (1, 4):
        payload = memory_breakdown(
            ARCH_QWEN3_8B,
            context_length=4096,
            max_parallel=parallel,
            bits_per_parameter=16.0,
            adapter_bytes=48 * 1024,
        )
        print(
            f"  ctx {payload['context_length']} × 并发 {parallel}："
            f"权重 {payload['weights_gib']:.4f} GiB + "
            f"KV {payload['kv_cache_mib']:.2f} MiB + "
            f"适配器 {payload['adapter_bytes'] / 1024:.0f} KiB = "
            f"{payload['total_gib']:.4f} GiB"
        )
    print()
    print(f"  每 token 每序列的 KV 缓存：{ARCH_QWEN3_8B.kv_bytes_per_token} 字节 "
          f"= {ARCH_QWEN3_8B.kv_bytes_per_token / 1024:.0f} KiB")
    print("  算式：2 × 36 层 × 8 个 KV 头 × head_dim 128 × 2 字节（bf16）= 147456")
    print("  并发 1 → 4 会在权重之外多要 1.69 GiB，而它不提升任何质量。")
    quantized = memory_breakdown(ARCH_QWEN3_8B, bits_per_parameter=4.0)
    print(
        f"  4-bit 权重 {quantized['weights_gib']:.4f} GiB"
        f"（下界标记：{quantized['weights_is_lower_bound']}）"
    )


def section_4_switch() -> None:
    """第 4 节：流量切换."""
    title("4. 流量切换：确定性分桶、影子流量、只向一个方向降级")
    prompts = [
        "什么是 DPO？",
        "LoRA 与全参微调的显存差多少",
        "QLoRA 的 NF4 码本是怎么来的",
        "为什么需要 warmup",
        "如何评估一次微调是否成功",
    ]
    for ratio in (0.0, 0.2, 0.6, 1.0):
        payload = split_summary(prompts, TrafficPolicy(dedicated_ratio=ratio))
        print(
            f"  设定 ratio={ratio:<5} 实际分走 {payload['dedicated']}/{payload['requests']}"
            f"（{payload['dedicated_ratio_actual']:.4f}）—— 再算一次结果完全相同"
        )
    print()
    cloud = FlakyLLM("cloud", reply="[云端] 参考答案")
    dedicated = FlakyLLM("dedicated", reply="[专属] 参考答案", fail_times=1)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    print(f"  fail_open=True ：第 1 次 {switcher.chat([Message('user', '什么是 DPO？')])}")
    print(f"                  第 2 次 {switcher.chat([Message('user', '什么是 DPO？')])}")
    print(f"                  影子调用 {switcher.shadow_calls}／降级次数 {switcher.fallback_count}")
    print(f"                  服务计数：专属 {switcher.dedicated_served} / 云端 {switcher.cloud_served}"
          f"（合计 = 请求数 {len(switcher.switch_log)}）")
    print("                  ↑ 第 1 次其实是云端兜住的：**fail_open 会让「一切正常」变成一个假象**")
    print()
    strict_cloud = FlakyLLM("cloud", reply="[云端] 参考答案")
    strict_dedicated = FlakyLLM("dedicated", reply="[专属] 参考答案", fail_times=1)
    strict = ModelSwitcher(
        strict_cloud, strict_dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, fail_open=False)
    )
    try:
        strict.chat([Message("user", "什么是 DPO？")])
    except Exception as exc:  # noqa: BLE001
        print(f"  fail_open=False：{exc}")
        print(f"                  云端被调用 {strict_cloud.calls} 次（严格模式不兜底）")
    print()
    shadow_switcher = ModelSwitcher(
        FlakyLLM("cloud", reply="[云端] 参考答案"),
        FlakyLLM("dedicated", reply="[专属] 参考答案"),
        policy=TrafficPolicy(shadow_ratio=1.0),
    )
    print(f"  影子模式返回：{shadow_switcher.chat([Message('user', '什么是 DPO？')])}")
    print(
        f"  影子调用 {shadow_switcher.shadow_calls} 次（付两次成本），"
        f"用户看到的仍是云端答案"
    )
    print()
    print("  策略表里每一条都带着自己的代价：")
    for row in route_table():
        print(f"    {row['field']:<20} = {str(row['value']):<8} {row['cost']}")


class SeqClock:
    """确定性时钟（毫秒 → 秒），把"实测延迟"变成可注入的输入."""

    def __init__(self, arms: list[list[float]]) -> None:
        self._values = [0.0]
        running = 0.0
        for arm in arms:
            for milliseconds in arm:
                running += milliseconds / 1000.0
                self._values.extend([running, running])
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def section_5_verify() -> None:
    """第 5 节：上线前的端到端验证."""
    title("5. 上线验证：同一批用例上两个推理臂的对比（云端 = 已经验证过的路径）")
    probes = [
        ProbeCase(case_id="dpo", prompt="什么是 DPO？", must_contain=("DPO", "偏好")),
        ProbeCase(case_id="lora", prompt="LoRA 省显存的原理", must_contain=("LoRA", "低秩")),
        ProbeCase(case_id="qlora", prompt="NF4 为什么适合权重", must_contain=("NF4", "正态")),
        ProbeCase(case_id="warmup", prompt="为什么需要 warmup", must_contain=("warmup",)),
    ]
    cloud_replies = {
        "dpo": "DPO 是直接偏好优化，用偏好对直接优化策略",
        "lora": "LoRA 用低秩矩阵近似权重更新",
        "qlora": "NF4 按正态分位数取码本",
        "warmup": "warmup 让早期更新幅度小一些",
    }
    dedicated_replies = dict(cloud_replies)
    dedicated_replies["qlora"] = "分位数量化，细节略"  # 退步一条
    print("  策略未给出延迟倍数时：")
    plain = run_verification(
        probes,
        cloud_respond=lambda case: cloud_replies[case.case_id],
        dedicated_respond=lambda case: dedicated_replies[case.case_id],
        clock=SeqClock([[180.0] * 4, [420.0] * 4]),
    )
    print(f"    {plain.summary_line()}")
    print(f"    跳过的检查 {[item.name for item in plain.skipped_checks]}"
          f"（延迟中位数 云端 {plain.cloud_latency_ms:.0f}ms / 专属 {plain.dedicated_latency_ms:.0f}ms"
          f" = {plain.latency_ratio:.3f}×）")
    print(f"    逐项：")
    for check in plain.checks:
        flag = "阻塞" if check.blocking else "告警"
        print(
            f"      [{'通过' if check.passed else '不通过'}/{flag}] {check.name:<12} "
            f"实际 {check.actual} / 阈值 {check.threshold}"
        )
    print()
    print("  显式要求延迟倍数 <= 2.0 时（同一条退步仍然存在）：")
    strict = run_verification(
        probes,
        cloud_respond=lambda case: cloud_replies[case.case_id],
        dedicated_respond=lambda case: dedicated_replies[case.case_id],
        policy=VerifyPolicy(min_pass_rate=0.5, max_regression=0.0, max_latency_ratio=2.0),
        clock=SeqClock([[180.0] * 4, [420.0] * 4]),
    )
    print(f"    {strict.summary_line()}")
    print()
    print("  专属侧有一条调用失败时（**调用失败与答得不对是两类问题**）：")
    failed = run_verification(
        probes,
        cloud_respond=lambda case: cloud_replies[case.case_id],
        dedicated_respond=lambda case: _maybe_fail(case, dedicated_replies),
        clock=SeqClock([[100.0] * 4, [100.0] * 4]),
    )
    print(f"    {failed.summary_line()}")
    print(f"    失败用例：{[(item.case_id, item.error) for item in failed.outcomes if item.error]}")


def _maybe_fail(case: ProbeCase, replies: dict[str, str]) -> str:
    """让 warmup 这一条在专属侧失败（模拟一条超时）."""
    if case.case_id == "warmup":
        raise TimeoutError("专属服务在 warmup 用例上超时")
    return replies[case.case_id]


def section_6_cost() -> None:
    """第 6 节：成本对比."""
    title("6. 成本对比：固定成本 vs 可变成本（价格是公开时价，吞吐是待替换的占位值）")
    print("  价格表（唯一出处：serving/cost.py 的两个常量）：")
    for row in price_book():
        print(f"    [{row['kind']:<5}] {row['name']:<24} {row['value']:<8} {row['unit']}")
    print()
    cloud = cloud_pricing("gpt-4o-mini")
    gpu = gpu_pricing("aws-g5-xlarge-a10g", tokens_per_second=250.0, utilization=1.0)
    for requests in (10_000, 100_000, 1_000_000, 10_000_000):
        comparison = compare_costs(
            requests=requests,
            input_tokens_per_request=1500,
            output_tokens_per_request=500,
            cloud=cloud,
            gpu=gpu,
        )
        print(f"  {comparison.summary_line()}")
    print()
    low = compare_costs(
        requests=100_000,
        input_tokens_per_request=1500,
        output_tokens_per_request=500,
        cloud=cloud,
        gpu=gpu,
    )
    print("  100000 请求这一个月的四项数字：")
    print(f"    用掉的算力      {low.required_gpu_hours:.4f} GPU·h")
    print(f"    为之付费的时长  {low.deployed_gpu_hours:.4f} GPU·h"
          f"（{low.machines} 台 × 730h）")
    print(f"    实际利用率      {low.utilization_actual:.2%}")
    print(f"    盈亏平衡调用量  {low.breakeven_requests:.0f} 请求/月")
    print("  → 「自建更便宜」这句话没有主语就不成立：它取决于调用量、吞吐与利用率。")
    print("  → 而空转的 GPU 是最贵的：账单按时长走，与这一小时处理了多少 token 无关。")


def ded_by_prompt(policy: TrafficPolicy, count: int) -> int:
    """先用 ``split_summary`` 算一遍会分走多少条（**不调用任何模型**）.

    它与 ``ModelSwitcher`` 用的是同一个哈希键（最近一条 user 消息的原文），
    因此"先算一遍"与"真跑一遍"必须得到同一个数字——这正是确定性分桶的价值。
    """
    payload = split_summary([f"问题 {i}" for i in range(count)], policy)
    return int(payload["dedicated"])


def section_7_timeline() -> None:
    """第 7 节：灰度的四步时间线."""
    title("7. 灰度时间线：0 → 0.05 → 0.5 → 1.0，以及每一步该看什么")
    probes = [ProbeCase(case_id="c", prompt="什么是 LoRA？", must_contain=("LoRA",))]
    steps = (
        (
            "① 影子",
            TrafficPolicy(dedicated_ratio=0.0, shadow_ratio=0.05),
            "专属模型在真实请求上跑一遍，用户完全不受影响",
        ),
        (
            "② 小流量",
            TrafficPolicy(dedicated_ratio=0.05, shadow_ratio=0.0),
            "5% 的用户真的在用专属模型，盯合格率与延迟",
        ),
        (
            "③ 半量",
            TrafficPolicy(dedicated_ratio=0.5, shadow_ratio=0.0),
            "一半流量，重点看线上指标有没有分叉",
        ),
        (
            "④ 全量",
            TrafficPolicy(dedicated_ratio=1.0, shadow_ratio=0.0),
            "切换完成：此后 ModelSwitcher 应当被摘掉",
        ),
    )
    for label, policy, note in steps:
        switcher = ModelSwitcher(
            FlakyLLM("cloud", reply="[云端] 答案"),
            FlakyLLM("dedicated", reply="[专属] 答案"),
            policy=policy,
        )
        served = [
            switcher.chat([Message("user", f"问题 {i}")]) for i in range(200)
        ]
        dedicated_count = sum(1 for item in served if item.startswith("[专属]"))
        share = ded_by_prompt(policy, 200)
        print(
            f"  {label} ratio={policy.dedicated_ratio:<5} shadow={policy.shadow_ratio:<5} "
            f"200 个不同问题 → 专属服务 {dedicated_count:<4} "
            f"影子 {switcher.shadow_calls:<4} 已切完 {policy.is_fully_cut_over}"
        )
        print(
            f"       先算一遍会分走 {share} 条（不调用任何模型）；"
            f"再跑一遍结果相同：{ded_by_prompt(policy, 200) == share}"
        )
        print(f"       {note}")
    print()
    print("  五条上线检查，以及每一条「没测时算什么」：")
    for row in verify_table():
        flag = "阻塞" if row["blocking"] else "告警"
        print(f"    [{flag}] {row['name']:<12} {row['when_missing']}")


def section_8_api() -> None:
    """第 8 节：四个 HTTP 端点（进程内调用，不加载任何模型）."""
    title("8. 四个端点：/serving/targets 与三个只算不跑的动作")
    from fastapi.testclient import TestClient

    from smart_research_agent.api.app import create_app

    client = TestClient(create_app(llm=MockLLM(default="offline")))
    targets = client.get("/serving/targets").json()
    print(
        f"  GET  /serving/targets          → 形态 {len(targets['kinds'])} 种 / "
        f"流量项 {len(targets['traffic_table'])} 条 / 上线检查 {len(targets['verify_table'])} 条 / "
        f"绑定检查 {len(targets['binding_table'])} 条 / 价格 {len(targets['price_book'])} 行"
    )
    print(f"       显存 KV 缓存 {targets['memory']['kv_cache_mib']} MiB / "
          f"权重 {targets['memory']['weights_gib']} GiB")
    print(f"       形态建议 {targets['recommendation']['kind']}"
          f"（与配置一致：{targets['recommendation']['matches_configured']}）")
    binding_response = client.post(
        "/serving/binding/verify",
        json={
            "versions": [
                {
                    "version": "1.1.0",
                    "base_model": BASE_MODEL,
                    "adapter_sha256": ADAPTER,
                    "dataset_fingerprint": DATASET,
                    "stage": "stable",
                    "artifacts": {
                        "adapter": "outputs/lora/adapters/adapter-final",
                        "merged": "outputs/lora-merged",
                    },
                }
            ],
            "bound_version": "1.1.0",
            "served": {
                "name": SERVING_NAME,
                "base_model": BASE_MODEL,
                "adapter_short_hash": "deadbeef0000",
            },
        },
    ).json()
    print(f"  POST /serving/binding/verify   → passed={binding_response['passed']} "
          f"阻塞失败={binding_response['report']['blocking_failures']}")
    verify_response = client.post(
        "/serving/verify",
        json={
            "probes": [{"case_id": "dpo", "prompt": "什么是 DPO？", "must_contain": ["DPO"]}],
            "cloud": {"replies": {"dpo": "DPO 是直接偏好优化"}, "latency_ms": {"dpo": 100.0}},
            "dedicated": {"replies": {"dpo": "DPO 是直接偏好优化"}, "latency_ms": {"dpo": 300.0}},
            "policy": {"max_latency_ratio": 2.0},
        },
    ).json()
    print(f"  POST /serving/verify           → {verify_response['summary']}")
    cost_response = client.post("/serving/cost/compare", json={}).json()
    print(f"  POST /serving/cost/compare     → {cost_response['summary']}")
    print()
    print("  四个端点全部只读或纯计算：不写磁盘、不联网、不加载模型、不用随机数。")


def main() -> int:
    """按节运行演示（任何一节失败都返回非零退出码）."""
    for section in (
        section_1_kinds,
        section_2_binding,
        section_3_memory,
        section_4_switch,
        section_5_verify,
        section_6_cost,
        section_7_timeline,
        section_8_api,
    ):
        section()
    print(f"\n演示完成（工作目录 {Path.cwd()}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
