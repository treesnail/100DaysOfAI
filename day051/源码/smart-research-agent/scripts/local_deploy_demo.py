"""day045 演示：本地部署的三件事——算显存、连端点、量效果.

无需安装 Ollama 也能跑：显存估算是纯计算，端点与对比评估用确定性桩
（Mock 客户端 / MockLLM）演示协议与口径。真实接入只要把 ``.env`` 里的
``LLM_BACKEND`` 改成 ``local`` 即可。

用法::

    python scripts/local_deploy_demo.py
"""

from __future__ import annotations

from smart_research_agent.evaluation.local_eval import LocalModelEvaluator
from smart_research_agent.llm.local_model import (
    LocalModelSpec,
    create_local_model,
    normalize_base_url,
)
from smart_research_agent.llm.local_runtime import (
    OllamaRuntime,
    fitting_quantizations,
    max_context_tokens,
    plan_deployment,
)
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: Qwen3-8B 的架构参数（取自模型 config.json 的公开字段）：
#: 36 层、8 个 KV 头（GQA）、head_dim 128。
QWEN3_8B = {"num_layers": 36, "kv_heads": 8, "head_dim": 128}


def demo_vram_plan() -> None:
    """① 先算再跑：同一模型在不同显卡/量化下的可行性."""
    print("\n=== ① 显存预算：8B 模型（ctx=4096, fp16 KV）===")
    for vram in (8.0, 12.0, 24.0):
        plan = plan_deployment(8.0, vram, quant="q4_k_m", **QWEN3_8B)
        print(f"  显存 {vram:>4.0f}GB → 需 {plan.total_gb:>5.2f}GB | {plan.recommendation()}")

    print("\n  可用的量化档位（24GB 显存）：", fitting_quantizations(8.0, 24.0, **QWEN3_8B))
    print("  12GB 显存下 FP16 权重需要：", f"{8.0 * 2:.1f}GB（不含 KV）")
    print("  24GB 显存下单序列最大上下文：", max_context_tokens(8.0, 24.0, **QWEN3_8B), "token")


def demo_endpoint() -> None:
    """② 端点规范化：漏写 /v1 是本机部署最常见的配置错误."""
    print("\n=== ② 端点规范化 ===")
    for raw in ("http://localhost:11434", "localhost:11434/v1/", "http://127.0.0.1:8000"):
        print(f"  {raw:<28} → {normalize_base_url(raw)}")

    spec = LocalModelSpec(name="qwen3:8b")
    print(
        f"  默认 spec: {spec.endpoint} | keep_alive={spec.keep_alive}"
        f" | ctx={spec.context_length}"
    )

    model = create_local_model()
    print(f"  工厂产出: {model.describe()}")


def demo_runtime() -> None:
    """③ 生命周期：keep_alive 控制的显存占用（用桩 transport 演示请求体）."""
    print("\n=== ③ 模型生命周期（桩 transport，不依赖真实 Ollama）===")
    recorded: list[tuple[str, dict | None]] = []

    def stub(url: str, payload: dict | None) -> dict:
        recorded.append((url, payload))
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}]}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3:8b", "expires_at": "2026-09-12T09:00:00Z"}]}
        return {"status": "success"}

    runtime = OllamaRuntime(transport=stub)
    print("  本机模型：", runtime.list_models())
    print("  常驻中：", runtime.is_running("qwen3:8b"))
    runtime.load_model("qwen3:8b", keep_alive=-1, context_length=8192)
    runtime.unload_model("qwen3:8b")
    runtime.pull_model("qwen3-vl:8b")
    for url, payload in recorded[-3:]:
        print(f"  POST {url}  payload={payload}")


def demo_compare() -> None:
    """④ 换模型前先量：本地 vs 云端的一致性/延迟/成本."""
    print("\n=== ④ 本地 vs 云端对比评估（Mock 桩，离线可跑）===")
    tasks = [
        "什么是向量数据库？",
        "解释一下 RAG 的检索步骤",
        "对比 FP16 与 int4 量化的显存差异",
    ]
    # 本地小模型的回答更短、更口语；云端模型回答完整——这正是换模型时
    # "延迟更低但一致性下降"的真实形态
    local = MockLLM(
        responses=[
            "向量数据库存向量。",
            "RAG 先检索再生成。",
            "int4 比 FP16 省显存。",
        ]
    )
    cloud = MockLLM(
        responses=[
            "向量数据库用于存储与检索向量，支持最近邻搜索。",
            "RAG 的检索步骤：文档切片、向量化、建索引、查询相似度检索、拼进提示词。",
            "FP16 每参数 2 字节，int4 约 0.5 字节，权重显存相差约 4 倍。",
        ]
    )

    # 计时刻度：本地单卡 8B 每条 1.5s、云端集群每条 0.4s——"本地不等于更快"
    # 正是要量出来的第一件事（3 条 × 2 模型 × 2 次计时 = 12 个刻度）
    ticks = [0.0, 1.5, 1.5, 3.0, 3.0, 4.5, 0.0, 0.4, 0.4, 0.8, 0.8, 1.2]
    tick_iter = iter(ticks)
    evaluator = LocalModelEvaluator(
        local,
        cloud,
        clock=lambda: next(tick_iter),
        cloud_price_per_1k=0.01,
        local_price_per_1k=0.0,
    )
    report = evaluator.run(tasks)
    for key, value in report.to_dict().items():
        print(f"  {key}: {value}")
    worst = report.worst_case()
    if worst is not None:
        print(f"  最差样本：{worst.task} → 本地回复「{worst.reply}」")
    print("  结论：", report.verdict())


def main() -> int:
    demo_vram_plan()
    demo_endpoint()
    demo_runtime()
    demo_compare()
    logger.info("day045 演示完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
