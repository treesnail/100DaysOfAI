"""专属模型部署与切换包（M5-D11）：把"微调好的模型"变成"线上跑着的模型".

M5 的十一天到这里收尾。前面十天的产物是一条**离线**的链路：

```text
数据 → 训练 → 评估 → 对齐 → 版本 → 流水线 → 可发布的模型版本
```

而"可发布"与"已上线"之间还差三步，今天是这三步：

```text
部署    把哪一版放到哪个服务名上、加载形态是什么（spec + binding）
切换    让新模型按比例接管流量，而不是一刀切（switch）
验证    切之前先证明它不比现在差（verify），切之后算清它到底省不省钱（cost）
```

五个模块，每个回答一个问题：

```text
spec.py      线上应该跑什么：形态（base / adapter / merged）、端点、显存怎么算
binding.py   线上跑的到底是不是它：注册表 / 部署记录 / 端点自述三处比对
switch.py    流量怎么分：确定性分桶的灰度、影子流量、单向降级
verify.py    它够不够好：同一批用例上两个推理臂的合格率、错误数、延迟
cost.py      它值不值：固定成本 vs 可变成本、利用率、盈亏平衡调用量
```

## 三条贯穿全包的纪律

1. **"哪个模型"必须能被证伪**。``binding`` 的每一条检查都指向一种
   "不会报错但会答错"的故障：路径指错、基座换了、副本没更新。
   它们共同的特征是**证据都在、只是没人比对**——所以本包把比对做成了代码；
2. **切换只向一个方向降级**。专属失败可以回落云端（已验证路径），
   云端失败**绝不**回落专属。反向降级等于"在云端故障时把流量交给一个
   还没通过验证的模型"，两个故障叠加且第二个的根因会被第一个掩盖；
3. **缺失就是缺失**。``verify`` 里"策略没给延迟倍数"是**未检查**而不是
   通过；``binding`` 里"端点没自述"是不一致而不是跳过；
   ``cost`` 里云端成本为 0 时节省比例返回 ``None`` 而不是 0。

## 与既有包的接缝

- **上游**：``registry``（版本三元组与 ``is_deployable``）、``sft.hf_script``
  的 ``DEFAULT_BASE_MODEL``、``llm.local_model`` 的后端常量与默认端口。
  它们一行都不用改；
- **下游**：``api.routes`` 的 ``/serving/*`` 端点只做**只读或纯计算**——
  真正加载模型、压测延迟、切流量的动作由 ``scripts/serving_demo.py``
  与部署脚本承担。**一个需要 GPU 的流程不该挂在一个 HTTP 请求上。**
"""

from __future__ import annotations

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
    BindingVerification,
    ServedEndpoint,
    ServingBinding,
    bind_version,
    binding_table,
    read_binding,
    verify_binding,
    write_binding,
)
from smart_research_agent.serving.cost import (
    CLOUD_PRICING,
    GPU_HOURLY_USD,
    HOURS_PER_MONTH,
    SIDE_CLOUD,
    SIDE_DEDICATED,
    SIDE_TIE,
    CloudPricing,
    CostComparison,
    GpuPricing,
    breakeven_requests,
    cloud_pricing,
    compare_costs,
    gpu_pricing,
    price_book,
)
from smart_research_agent.serving.errors import ServingError
from smart_research_agent.serving.spec import (
    ARCH_QWEN3_8B,
    DEFAULT_NUM_PARALLEL,
    KINDS,
    KIND_ADAPTER,
    KIND_BASE,
    KIND_DESCRIPTIONS,
    KIND_MERGED,
    KIND_REQUIRED_FIELDS,
    SERVING_LIMITATIONS,
    SERVING_OUT_OF_SCOPE,
    SUPPORTED_QUANTIZATION_BITS,
    ModelArchitecture,
    ServingSpec,
    memory_breakdown,
    recommend_kind,
    spec_table,
    weights_bytes,
)
from smart_research_agent.serving.switch import (
    BUCKET_SALT_PRIMARY,
    BUCKET_SALT_SHADOW,
    ROUTE_CLOUD,
    ROUTE_DEDICATED,
    ROUTES,
    ModelSwitcher,
    RouteRecord,
    TrafficPolicy,
    bucket_of,
    route_table,
    split_summary,
)
from smart_research_agent.serving.verify import (
    ARM_CLOUD,
    ARM_DEDICATED,
    ARMS,
    CHECK_CASES,
    CHECK_ERRORS,
    CHECK_LATENCY,
    CHECK_PASS_RATE,
    CHECK_REGRESSION,
    ProbeCase,
    ProbeOutcome,
    ServingVerification,
    VerifyCheck,
    VerifyPolicy,
    judge,
    median_of,
    run_verification,
    verify_table,
)

__all__ = [
    "ARCH_QWEN3_8B",
    "ARM_CLOUD",
    "ARM_DEDICATED",
    "ARMS",
    "BINDING_FILE",
    "BUCKET_SALT_PRIMARY",
    "BUCKET_SALT_SHADOW",
    "CHECK_ADAPTER_HASH",
    "CHECK_BASE_MODEL",
    "CHECK_CASES",
    "CHECK_ERRORS",
    "CHECK_KIND_SHAPE",
    "CHECK_LATENCY",
    "CHECK_PASS_RATE",
    "CHECK_REGISTRY_HEAD",
    "CHECK_REGRESSION",
    "CHECK_SERVING_NAME",
    "CLOUD_PRICING",
    "DEFAULT_NUM_PARALLEL",
    "GPU_HOURLY_USD",
    "HOURS_PER_MONTH",
    "KINDS",
    "KIND_ADAPTER",
    "KIND_BASE",
    "KIND_DESCRIPTIONS",
    "KIND_MERGED",
    "KIND_REQUIRED_FIELDS",
    "ROUTES",
    "ROUTE_CLOUD",
    "ROUTE_DEDICATED",
    "SERVING_LIMITATIONS",
    "SERVING_OUT_OF_SCOPE",
    "SHORT_HASH_LENGTH",
    "SIDE_CLOUD",
    "SIDE_DEDICATED",
    "SIDE_TIE",
    "SUPPORTED_QUANTIZATION_BITS",
    "BindingCheck",
    "BindingPolicy",
    "BindingVerification",
    "CloudPricing",
    "CostComparison",
    "GpuPricing",
    "ModelArchitecture",
    "ModelSwitcher",
    "ProbeCase",
    "ProbeOutcome",
    "RouteRecord",
    "ServedEndpoint",
    "ServingBinding",
    "ServingError",
    "ServingSpec",
    "ServingVerification",
    "TrafficPolicy",
    "VerifyCheck",
    "VerifyPolicy",
    "bind_version",
    "binding_table",
    "breakeven_requests",
    "bucket_of",
    "cloud_pricing",
    "compare_costs",
    "gpu_pricing",
    "judge",
    "median_of",
    "memory_breakdown",
    "price_book",
    "read_binding",
    "recommend_kind",
    "route_table",
    "run_verification",
    "spec_table",
    "split_summary",
    "verify_binding",
    "verify_table",
    "weights_bytes",
    "write_binding",
]
