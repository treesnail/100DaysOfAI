"""day087 离线演示：推理路径上的三笔账 —— 缓存、精度、批.

十节，全部离线、全部确定性（不需要 API Key，也不装 bitsandbytes / accelerate）：

```text
1  缓存的两阶段：prefill 一步到位、decode 每步一个固定增量
2  缓存账的公式：长度 × 每步 == 总量（对五个长度各核一次）
3  缓存路径 vs 整段重算：**逐位**相同（本课最强的一条对账）
4  位置必须由缓存长度决定（不是从 0 重新数起）
5  量化的三种组合：位数 × 方案 × 粒度（实测误差与上界 scale/2）
6  粒度收益：只换粒度（样本带离群值，否则量不出收益）
7  int4 打包：无损的位运算（含两个端点）
8  批的账：静态 vs 连续、槽位占用与吞吐
9  预算的账：四项精度 × 三项之和 × T_max
10 十条笔记与五条边界
```

运行方式::

    cd day087/源码/smart-research-agent
    python scripts/inference_optim_demo.py

产出 ``outputs/inference_optim_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.hf_integration.forward import hidden_states, logits  # noqa: E402
from smart_research_agent.inference_optim import (  # noqa: E402
    batching,
    budget,
    cache as cache_module,
    errors,
    study,
    types,
    verify,
)
from smart_research_agent.inference_optim.quantize import (  # noqa: E402
    QuantSpec,
    error_bound,
    logits_of,
    measure,
    round_trip_line,
    unpack_int4,
)
from smart_research_agent.inference_optim.types import (  # noqa: E402
    LEVELS,
    OPTIM_NOTES,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "inference_optim_demo.txt"

#: 演示用的提示与步数（与测试样本同源，因此演示里的数能被测试复算）。
PROMPT = "hello world"
STEPS = 4
BUDGET_BYTES = 64 * 1024


class Report:
    """攒行 + 落盘（**不做任何计算**）."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, index: int, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"== {index}. {title}")

    def add(self, *texts: str) -> None:
        self.lines.extend(texts)

    def flush(self) -> None:
        text = "\n".join(self.lines)
        print(text)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text + "\n", encoding="utf-8")


def load_case(model_type: str):
    """装一份 case（与 day086 的快照同源：内存假仓库 + 确定性权重）."""
    import json

    from smart_research_agent.hf_integration import config, forward, hub
    from smart_research_agent.hf_integration.tokenizer import (
        ByteBPETokenizer,
        build_tiny_tokenizer,
    )

    built = build_tiny_tokenizer()
    tiny = {
        "gpt2": {
            "model_type": "gpt2",
            "vocab_size": len(built.vocab),
            "n_embd": 16,
            "n_head": 2,
            "n_layer": 2,
            "n_positions": 32,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
            "tie_word_embeddings": True,
        },
        "bert": {
            "model_type": "bert",
            "vocab_size": len(built.vocab),
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "max_position_embeddings": 32,
            "intermediate_size": 64,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "type_vocab_size": 2,
            "tie_word_embeddings": True,
        },
    }[model_type]
    commit = "1f2e3d4c5b6a7988071625344352617069859a0b"
    repo_id = f"org/tiny-{model_type}"
    files = {
        "config.json": json.dumps(tiny).encode("utf-8"),
        "vocab.json": json.dumps(built.vocab).encode("utf-8"),
        "merges.txt": (
            "#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in built.merges) + "\n"
        ).encode("utf-8"),
        "model.safetensors": b"\x00\x01",
    }
    fake = hub.InMemoryHub(commits={repo_id: {"main": commit}}, trees={repo_id: {commit: files}})
    resolver = hub.HubResolver(hub=fake, cache_dir="C:/cache/hub")
    snapshot = resolver.resolve(repo_id, local_files_only=False)
    card = config.parse_config(tiny, name=repo_id)
    weights = forward.make_weights(card, seed=7)
    tokenizer = ByteBPETokenizer.from_snapshot(resolver, snapshot, name=repo_id)
    return card, weights, tokenizer


def section_steps(report: Report, card, weights, tokenizer) -> None:
    """第 1 节：缓存的两阶段."""
    report.section(1, "缓存的两阶段：prefill 一步到位、decode 每步一个固定增量")
    prompt = tuple(tokenizer.encode(PROMPT))
    _logits, state, rows = cache_module.generate_with_cache(
        card, weights, prompt, steps=STEPS, level=types.LEVEL_FP32
    )
    report.add(f"  提示 {PROMPT!r} = {len(prompt)} 个 token，{card.layers} 层、{card.hidden} 维")
    for row in rows:
        report.add("  " + row.line())
    report.add("  " + verify.cache_state_line(state))
    report.add(f"  注意力打分次数（重算, 缓存）= {cache_module.attention_work(len(prompt) + STEPS)}")


def section_formula(report: Report, card) -> None:
    """第 2 节：缓存账的公式."""
    report.section(2, "缓存账的公式：长度 × 每步 == 总量")
    report.add("  " + cache_module.step_delta_is_a_formula(card))
    for line in study.cache_formula_rows(card):
        report.add("  " + line)


def section_recompute(report: Report, card, weights, tokenizer) -> None:
    """第 3 节：缓存路径 vs 整段重算."""
    report.section(3, "缓存路径 vs 整段重算：**逐位**相同")
    prompt = tuple(tokenizer.encode(PROMPT))
    outcome = verify.check_cached_equals_recompute(card, weights, prompt, 32)
    report.add("  " + outcome.line())
    cached, full = cache_module.compare_with_recompute(card, weights, prompt, 32)
    report.add(f"  两条路径的宽度都是 {len(cached)}（= 词表）| 逐位相同 {cached == full}")


def section_position(report: Report, card, weights, tokenizer) -> None:
    """第 4 节：位置必须由缓存长度决定."""
    report.section(4, "位置必须由缓存长度决定（不是从 0 重新数起）")
    prompt = tuple(tokenizer.encode(PROMPT))
    cache = cache_module.make_cache(card)
    states, _values = cache_module.prefill(card, weights, cache, prompt)
    report.add(f"  prefill 之后：位置表用到了 0..{len(prompt) - 1}，缓存长度 {cache.length}")
    _state, _logits2, position = cache_module.decode_step(card, weights, cache, 32)
    report.add(f"  下一步的位置号 = {position}（= 缓存长度，不是 0）")
    fresh = cache_module.make_cache(card)
    cache_module.prefill(card, weights, fresh, (prompt[0],))
    report.add(
        "  对照：只 prefill 一个 token 时，那一步的位置是 0；"
        f"而整段路径里第 5 个 token 的位置是 {len(prompt) - 1}"
    )
    del states
    try:
        cache_module._embed_at(card, weights, 32, card.positions)
    except Exception as error:  # noqa: BLE001 - 演示里只印类名与第一行
        report.add(f"  越界的位置：{type(error).__name__} —— {str(error).splitlines()[0]}")


def section_quantize(report: Report, matrices) -> None:
    """第 5 节：量化的三种组合."""
    report.section(5, "量化：位数 × 方案 × 粒度（实测误差与上界 scale/2）")
    for row in study.quant_rows(matrices["skewed"]):
        report.add("  " + row.line())
    report.add("  手算锚点（对称、一行 1.0 / −0.5 / 0.0）：")
    spec = QuantSpec(level=types.LEVEL_INT8, scheme=types.SCHEME_ABSMAX)
    quantized, stats = measure(((1.0, -0.5, 0.0),), spec)
    report.add(
        f"    scale = {quantized.scale:.6e} = 1/127 ⇒ 格号 {list(quantized.values)}"
        f" | 上界 {error_bound(quantized.scale):.3e} | 实测 {stats.max_abs_error:.3e}"
    )


def section_granularity(report: Report, matrices) -> None:
    """第 6 节：粒度收益."""
    report.section(6, "粒度收益：只换粒度（样本带离群值，否则量不出收益）")
    for row in study.granularity_rows():
        report.add("  " + row.line())
    uniform = {row.level: row.ratio for row in study.granularity_rows(matrices["uniform"])}
    report.add(
        f"  对照（每行分布相同的权重）：int8 的收益只有 {uniform[types.LEVEL_INT8]:.1f} 倍"
        " ——它只在**行与行的动态范围差得很大**时才值钱"
    )


def section_pack(report: Report) -> None:
    """第 7 节：int4 打包."""
    report.section(7, "int4 打包：无损的位运算（含两个端点）")
    report.add("  " + round_trip_line((-8, 7, 0, -1, 1, -7, 6)))
    report.add("  " + verify.check_int4_pack_is_lossless().line())
    report.add(
        "  同一个字节 [120] 在两种约定下的含义不同：有符号解出 "
        f"{unpack_int4((120,), 1)}、无符号解出 {unpack_int4((120,), 1, signed=False)}"
    )


def section_batching(report: Report) -> None:
    """第 8 节：批的账."""
    report.section(8, "批的账：静态 vs 连续、槽位占用与吞吐")
    for line in study.batch_rows():
        report.add("  " + line)
    report.add("  " + verify.check_schedule_occupancy_is_counted().line())
    scarce = batching.continuous_batching_schedule((5, 5, 5, 5), max_batch=2)
    report.add(
        f"  槽位不够时（4 条各 5 长、批大小 2）：实际 {len(scarce)} 步，"
        f"而'最长的那条'（{batching.plan_batches((5, 5, 5, 5), max_batch=2).continuous_steps} 步）只是下界"
    )


def section_budget(report: Report, card) -> None:
    """第 9 节：预算的账."""
    report.section(9, "预算的账：四种精度 × 三项之和 × T_max")
    for row in study.budget_rows(card):
        report.add("  " + row.line())
    for level in LEVELS:
        report.add("  " + budget.check_max_tokens(card, level=level, budget_bytes=BUDGET_BYTES))


def section_tables(report: Report) -> None:
    """第 10 节：笔记、边界与失败族."""
    report.section(10, "十条笔记、五条边界与七个失败族")
    for line in study.note_lines():
        report.add("  " + line)
    report.add("")
    report.add("  五条边界（**这一课明确不承诺的事**）：")
    for index, boundary in enumerate(types.OPTIM_BOUNDARIES, start=1):
        report.add(f"    {index}. {boundary}")
    report.add("")
    report.add("  七个失败族各自该谁去修：")
    for name, outcome in errors.FAMILY_OUTCOMES.items():
        report.add(f"    {name:<15} {outcome}")
    report.add("")
    report.add(f"  连续缺席的那一族：{errors.ABSENT_FAMILY} —— {errors.ABSENT_FAMILY_REASON}")
    report.add("")
    report.add(f"  十条笔记的键：{list(OPTIM_NOTES)}")


def main() -> None:
    """跑完十节并落盘."""
    card, weights, tokenizer = load_case("gpt2")
    bert_card, bert_weights, bert_tokenizer = load_case("bert")
    matrices = {
        "skewed": study.skewed_matrix(),
        "uniform": tuple(weights.word[10:13]),
    }
    report = Report()
    section_steps(report, card, weights, tokenizer)
    section_formula(report, card)
    section_recompute(report, card, weights, tokenizer)
    section_position(report, card, weights, tokenizer)
    section_quantize(report, matrices)
    section_granularity(report, matrices)
    section_pack(report)
    section_batching(report)
    section_budget(report, card)
    section_tables(report)
    report.section(11, "第十一节附：BERT 侧的缓存路径也要对")
    bert_prompt = tuple(bert_tokenizer.encode("hey"))
    bert_cache = cache_module.make_cache(bert_card)
    bert_states, _values = cache_module.prefill(bert_card, bert_weights, bert_cache, bert_prompt)
    report.add(
        "  BERT prefill 与整段前向逐位相同："
        f"{bert_states == hidden_states(bert_card, bert_weights, (bert_prompt,)).rows[0]}"
    )
    report.add(
        "  logits_of 与 day086 的 logits 逐位相同："
        f"{logits_of(bert_card, bert_weights, bert_states[-1]) == logits(bert_card, bert_weights, bert_prompt)}"
    )
    report.flush()


if __name__ == "__main__":
    main()
