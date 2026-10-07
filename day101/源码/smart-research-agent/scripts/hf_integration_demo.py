"""day086 离线演示：把 Hugging Face 生态**接进来** —— 从文件找齐到接进 local_model.

十一节，全部离线、全部确定性（不需要 API Key，也不需要装 transformers）：

```text
1  缓存与"逐文件命中"（两轮解析 + 一次离线解析）
2  配置即形状：两个架构的键名映射与四个默认值
3  参数量：配置 / 库 / 本包权重 / 那 4h·L 的差
4  分词器：byte-level BPE 的两个文件与一个循环
5  一次真前向：hidden states 与 logits
6  池化三法，与那个"只差一个分母"的 bug
7  批量：段内掩码 vs naive（两种错法各一个读数）
8  两条管线共用一份权重；生成四策略
9  接进 local_model：八项方法面与三条路线
10 十条性质与三条跨天对账
11 十二条生态笔记与五条边界
```

运行方式::

    cd day086/源码/smart-research-agent
    python scripts/hf_integration_demo.py

产出 ``outputs/hf_integration_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.hf_integration import (  # noqa: E402
    bridge,
    config,
    features,
    forward,
    hub,
    pipeline,
    study,
    tokenizer as tokenizer_module,
    types,
    verify,
)
from smart_research_agent.hf_integration.errors import (  # noqa: E402
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
)
from smart_research_agent.hf_source.types import GenerationSettings  # noqa: E402
from smart_research_agent.llm.base import Message  # noqa: E402

#: 演示用的两个仓库与它们的 commit（与测试样本同源，因此演示里的数能被测试复算）。
REPO_ID = "org/tiny-gpt2"
BERT_REPO_ID = "org/tiny-bert"
COMMIT = "1f2e3d4c5b6a7988071625344352617069859a0b"
SECOND_COMMIT = "0a1b2c3d4e5f60718293a4b5c6d7e8f901234567"

SEED = 7
TEXTS = ("hello world", "hey", "hello there world")
PROMPT = "hello world"
SETTINGS = GenerationSettings(max_new_tokens=3, do_sample=False, seed=SEED)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "hf_integration_demo.txt"


class Report:
    """一个只负责"攒行 + 最后打印并落盘"的小工具（**不做任何计算**）."""

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


# ---------------------------------------------------------------------------
# 一个内存仓库：演示与测试用的是同一份结构（**不碰网络、不碰文件系统**）
# ---------------------------------------------------------------------------


def _config_payload(model_type: str, vocab: int) -> dict[str, object]:
    if model_type == types.ARCHITECTURE_GPT2:
        return {
            "model_type": "gpt2",
            "vocab_size": vocab,
            "n_embd": 16,
            "n_head": 2,
            "n_layer": 2,
            "n_positions": 32,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
            "tie_word_embeddings": True,
            "architectures": ["GPT2LMHeadModel"],
        }
    return {
        "model_type": "bert",
        "vocab_size": vocab,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "max_position_embeddings": 32,
        "intermediate_size": 64,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "type_vocab_size": 2,
        "tie_word_embeddings": True,
        "architectures": ["BertModel"],
    }


def _repo_files(model_type: str, tokenizer: object) -> dict[str, bytes]:
    builder = tokenizer
    payload = _config_payload(model_type, len(builder.vocab))  # type: ignore[attr-defined]
    merges = "#version: 0.2\n" + "\n".join(
        f"{a} {b}" for a, b in builder.merges  # type: ignore[attr-defined]
    )
    return {
        types.CONFIG_FILE: json.dumps(payload).encode("utf-8"),
        types.VOCAB_FILE: json.dumps(builder.vocab).encode("utf-8"),  # type: ignore[attr-defined]
        types.MERGES_FILE: merges.encode("utf-8"),
        types.WEIGHTS_FILE: b"\x00\x01\x02\x03\x04\x05\x06\x07",
    }


class Case:
    """一份"装好的"模型：假仓库 + 解析器 + 快照 + 卡片 + 权重 + 分词器."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        built = tokenizer_module.build_tiny_tokenizer()
        repo_id = REPO_ID if model_type == types.ARCHITECTURE_GPT2 else BERT_REPO_ID
        files = _repo_files(model_type, built)
        self.hub = hub.InMemoryHub(
            commits={repo_id: {"main": COMMIT, "v1": SECOND_COMMIT}},
            trees={repo_id: {COMMIT: files, SECOND_COMMIT: dict(files)}},
        )
        self.resolver = hub.HubResolver(hub=self.hub, cache_dir="C:/cache/hub")
        self.snapshot = self.resolver.resolve(repo_id, local_files_only=False)
        self.card = config.card_from_snapshot(self.resolver, self.snapshot, name=repo_id)
        self.weights = forward.make_weights(self.card, seed=SEED)
        self.tokenizer = tokenizer_module.ByteBPETokenizer.from_snapshot(
            self.resolver, self.snapshot, name=repo_id
        )

    def model(self, **kwargs: object) -> bridge.InProcessModel:
        return bridge.create_in_process_model(
            self.card, self.weights, self.tokenizer, settings=SETTINGS, **kwargs  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 十一节
# ---------------------------------------------------------------------------


def section_cache(report: Report, case: Case) -> None:
    """第 1 节：缓存与"逐文件命中"."""
    report.section(1, "缓存：名字 → commit → 文件 → 实体（三段结构 + 逐文件命中）")
    report.add(f"  仓库 {case.snapshot.repo_id} | revision {case.snapshot.revision} "
               f"| commit {case.snapshot.commit[:12]}…")
    report.add(f"  快照根 {case.snapshot.root}")
    report.add(f"  布局常量：{types.CACHE_PREFIX!r} / {types.SNAPSHOT_DIR!r} / "
               f"{types.BLOB_DIR!r} 三条路径段")
    for row in study.cache_rows(case.resolver, case.resolver.hub, case.snapshot.repo_id):
        report.add("  " + row.line())
    offline = case.resolver.resolve(case.snapshot.repo_id)
    report.add(
        f"  离线解析（local_files_only=True，复用同一个缓存）："
        f"下 {offline.downloaded_count} 个 / 命中 {offline.cached_count} 个"
    )
    report.add(
        "  换个缓存根再离线解析同一个 revision："
        + _offline_failure(case)
    )
    cache = verify.check_cache_is_per_file(case.resolver, case.snapshot.repo_id)
    report.add("  " + cache.line())


def _offline_failure(case: Case) -> str:
    """空缓存 + 离线 ⇒ 那一族错误（**它不是参数错**：参数全对，只是环境里没有）."""
    empty = hub.HubResolver(hub=case.hub, cache_dir="C:/cache/hub")
    try:
        empty.resolve(case.snapshot.repo_id)
    except Exception as error:  # noqa: BLE001 - 演示里只印类名与第一行
        return f"{type(error).__name__} —— {str(error).splitlines()[0]}"
    return "（没有报错）"


def section_config(report: Report, case: Case) -> None:
    """第 2 节：配置即形状."""
    report.section(2, "配置即形状：同一个量的两个键名 + 四个默认值")
    for model_type in types.ARCHITECTURES:
        keys = types.SIZE_KEYS[model_type]
        report.add(
            f"  {model_type:<5} " + " ".join(f"{name}={key}" for name, key in keys.items())
        )
    for model_type in types.ARCHITECTURES:
        card = (
            case.card
            if case.card.model_type == model_type
            else config.parse_config(_config_payload(model_type, case.card.vocab), name=model_type)
        )
        report.add("  " + config.describe(card))
        report.add(
            f"        激活 {card.activation!r} | eps {card.ln_eps:g} | "
            f"前馈 {card.ffn}（比 {card.ffn / card.hidden:.1f}） | "
            f"因果 {card.causal} | 融合 QKV {card.fused_qkv} | "
            f"token_type {card.uses_token_type}"
        )
    report.add(f"  BERT 的 intermediate_size 缺省 = {types.BERT_DEFAULT_INTERMEDIATE}"
               "（与 hidden 无关——这是与真库对账读出来的）")
    report.add(f"  GPT-2 的 n_inner 缺省 = 4·hidden = {types.FFN_RATIO}·hidden")


def section_parameters(report: Report, case: Case) -> None:
    """第 3 节：参数量四个口径."""
    report.section(3, "参数量：配置 / 库 / 本包权重 / 那 4h·L 的差")
    for row in study.param_rows():
        report.add("  " + row.line())
    for model_type in types.ARCHITECTURES:
        built = case if case.card.model_type == model_type else Case(model_type)
        weights = built.weights
        report.add("  " + forward.bias_breakdown_line(built.card, weights))
    report.add("  两个库读数（真实模型）："
               f"gpt2 = {types.GPT2_SMALL_PARAMETERS}，"
               f"bert-base-uncased = {types.BERT_BASE_PARAMETERS}")


def section_tokenizer(report: Report, case: Case) -> None:
    """第 4 节：分词器."""
    report.section(4, "分词器：byte-level BPE（两个文件 + 一个循环）")
    report.add(f"  词表 {case.tokenizer.vocab_size} 个（基元 {case.tokenizer.base_size} + "
               f"合并 {len(case.tokenizer.merges)} + 特殊 1）")
    report.add(f"  特殊 token：eos={case.tokenizer.eos_id}、pad={case.tokenizer.pad_id}"
               f"（{case.tokenizer.pad_token}）")
    for row in study.token_rows(case.tokenizer):
        report.add("  " + row.line())
    report.add("  预分词（四条分支，词首空格与词绑在一起）：")
    for text in ("hello world", "don't stop", "I'M here", "a1b"):
        report.add(f"    {text!r:<16} → {tokenizer_module.pretokenize(text)}")
    batch = case.tokenizer.batch_encode(list(TEXTS))
    report.add(f"  批量编码：{batch.batch_size} 条 × 宽 {batch.width}"
               f"（真实宽 {batch.real_width}，填充占比 {batch.padding_ratio:.1%}）")
    report.add(f"  重复合并的名次由**最后一次**出现决定："
               f"{tokenizer_module.merge_ranks((('a', 'b'), ('c', 'd'), ('a', 'b')))}")


def section_forward(report: Report, case: Case) -> None:
    """第 5 节：一次真前向."""
    report.section(5, "一次真前向：hidden states 与 logits")
    ids = [case.tokenizer.encode(text) for text in TEXTS]
    batch = forward.hidden_states(case.card, case.weights, tuple(tuple(row) for row in ids))
    report.add(f"  输入长度 {batch.lengths()} → hidden {batch.hidden}（宽度不变）")
    report.add(f"  policy={batch.policy}、batched={batch.batched}、"
               f"真实行数 {len(batch.padded())}（**不含任何填充行**）")
    values = forward.logits(case.card, case.weights, tuple(case.tokenizer.encode(PROMPT)))
    report.add(f"  logits 宽度 {len(values)}（= 词表）| 前 4 个 "
               + " ".join(f"{value:+.4f}" for value in values[:4]))
    report.add("  " + forward.embedding_norm_line(case.card, case.weights))
    bert = Case(types.ARCHITECTURE_BERT)
    bert_batch = forward.hidden_states(bert.card, bert.weights, ((1, 2, 3),))
    pooled = forward.pooler_output(bert.weights, bert_batch)
    report.add("  " + forward.embedding_norm_line(bert.card, bert.weights))
    report.add(f"  BERT 的 pooler_output（tanh 之后）前 3 个 "
               + " ".join(f"{value:+.4f}" for value in pooled[0][:3]))


def section_pooling(report: Report, case: Case) -> None:
    """第 6 节：池化三法与那个分母."""
    report.section(6, "池化三法，与那个'只差一个分母'的 bug")
    for row in study.pooling_rows(case.card, case.weights, case.tokenizer):
        report.add("  " + row.line())
    comparison = features.compare_pooling(
        ((1.0, 1.0), (3.0, 3.0), (5.0, 5.0), (9.0, 9.0)), (1, 1, 1, 0), "mean", width=4
    )
    report.add(f"  手算锚点：{comparison.line()}")
    report.add(f"  正确版 {comparison.correct[0]}（真实 3 格之和 ÷ 3）")
    report.add(f"  违反版 {comparison.buggy[0]}（4 格之和 ÷ 4 —— 分母用了总宽度）")
    report.add("  注意：**没有填充时两版逐位相同**——这才是它最难被发现的原因")


def section_batch(report: Report, case: Case) -> None:
    """第 7 节：批量与样本边界."""
    report.section(7, "批量：段内掩码 vs naive（两种错法各一个读数）")
    for row in study.batch_rows(case.tokenizer):
        report.add("  " + row.line())
    mask = forward.block_causal_mask((2, 2, 2))
    report.add("  段内下三角（三条各 2 长的样本，允许集合逐行）：")
    for index, row in enumerate(mask):
        report.add(f"    {index}: {''.join('1' if flag else '.' for flag in row)}")
    report.add("  真实现里这张表是 (batch, 1, seq, seq)——本包用一张 (total, total) 表达同一件事")
    rows = study.batch_rows(case.tokenizer)
    report.add(f"  naive 在两条架构上的错法不同：{rows[0].naive_reason} / {rows[1].naive_reason}")


def section_pipelines(report: Report, case: Case) -> None:
    """第 8 节：两条管线."""
    report.section(8, "两条管线共用一份权重；生成四策略")
    for task in types.TASK_KINDS:
        report.add("  " + pipeline.head_line(case.card, task))
    generation, embeddings = pipeline.run_both(
        case.card, case.weights, case.tokenizer, PROMPT, SETTINGS
    )
    report.add(f"  生成：{templates_text(generation)}")
    report.add(f"  特征：向量 {embeddings.pooled.batch_size} 条 × {embeddings.pooled.dim} 维，"
               f"策略 {embeddings.pooled.strategy}")
    for row in study.generation_rows(case.card, case.weights, case.tokenizer):
        report.add("  " + row.line())
    bert = Case(types.ARCHITECTURE_BERT)
    try:
        pipeline.text_generation(bert.card, bert.weights, bert.tokenizer, (PROMPT,), SETTINGS)
    except Exception as error:  # noqa: BLE001 - 演示里只印那一族与那句话
        report.add(f"  对 BERT 卡片跑生成：{type(error).__name__} —— {str(error).splitlines()[0]}")
    report.add(f"  feature-extraction 与 text-generation 的分工："
               f"{types.TASK_HEADS[types.TASK_TEXT_GENERATION]}")


def templates_text(output: pipeline.GenerationOutput) -> str:
    """把一次生成压成一行（演示用）."""
    return (
        f"策略 {output.strategy} | 新生成 {output.generated_lengths()} 个 token | "
        f"文本 {output.texts[0]!r}"
    )


def section_bridge(report: Report, case: Case) -> None:
    """第 9 节：接进 local_model."""
    report.section(9, "接进 local_model：八项方法面与三条路线")
    report.add(f"  方法面（{len(bridge.PROTOCOL_SURFACE)} 项）：{', '.join(bridge.PROTOCOL_SURFACE)}")
    report.add("  " + verify.check_protocol_surface_matches(case.model()).line())
    for backend in bridge.ALL_BACKENDS:
        report.add("  " + bridge.trait_line(backend))
    for kwargs in (
        {},
        {"needs_tools": True},
        {"needs_vision": True},
        {"multi_user": True, "needs_batching": True},
    ):
        report.add("  " + bridge.choose_backend(**kwargs).line())
    model = case.model()
    messages = [Message(role="user", content="hello")]
    full = model.chat(messages, max_tokens=2)
    pieces = "".join(model.stream(messages, max_tokens=2))
    report.add(f"  chat {full!r} 与 stream 拼接 {pieces!r} 相等：{full == pieces}")
    report.add(f"  is_available() = {model.is_available()} | context_length = {model.context_length}"
               "（来自位置表，不是部署参数）")
    report.add(f"  describe() 的键：{sorted(model.describe())}")


def section_verify(report: Report, case: Case) -> None:
    """第 10 节：十条性质与三条跨天对账."""
    report.section(10, "十条性质与三条跨天对账")
    report_object = verify.check_all(
        case.card,
        case.weights,
        case.tokenizer,
        case.resolver,
        case.snapshot,
        case.model(),
        texts=TEXTS,
        prompt=PROMPT,
    )
    for line in report_object.lines():
        report.add("  " + line)
    report.add(f"  ok = {report_object.ok}（不适用不算通过、也不算失败）")
    report.add("")
    report.add("  三条跨天对账：")
    for line in verify.reference_agreement_lines():
        report.add("    " + line)
    report.add(f"    记录在案的库读数来源：{verify.LIBRARY_RECORD_SOURCE}")


def section_tables(report: Report) -> None:
    """第 11 节：笔记、边界与失败族."""
    report.section(11, "十二条生态笔记、五条边界与七个失败族")
    for line in study.note_lines():
        report.add("  " + line)
    report.add("")
    report.add("  五条边界（**这一课明确不承诺的事**）：")
    for index, boundary in enumerate(types.INTEGRATION_BOUNDARIES, start=1):
        report.add(f"    {index}. {boundary}")
    report.add("")
    report.add("  七个失败族各自该谁去修：")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"    {name:<15} {outcome}")
    report.add("")
    report.add(f"  连续缺席的那一族：{ABSENT_FAMILY} —— {ABSENT_FAMILY_REASON}")


def main() -> None:
    """跑完十一节并落盘."""
    report = Report()
    gpt2 = Case(types.ARCHITECTURE_GPT2)
    section_cache(report, gpt2)
    section_config(report, gpt2)
    section_parameters(report, gpt2)
    section_tokenizer(report, gpt2)
    section_forward(report, gpt2)
    section_pooling(report, gpt2)
    section_batch(report, gpt2)
    section_pipelines(report, gpt2)
    section_bridge(report, gpt2)
    section_verify(report, gpt2)
    section_tables(report)
    report.flush()


if __name__ == "__main__":
    main()
