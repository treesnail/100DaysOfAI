"""``hf_integration`` 的样本构造器（day086 / M7-D10）.

与 ``hf_samples.py`` / ``stack_samples.py`` 同一取向：**所有样本在这里造一次**，
测试文件只消费它们。两条纪律：

```text
① 一切都在内存里     仓库是 InMemoryHub、缓存是 CacheStore、权重由 LCG 定死
                     ⇒ 测试不碰网络、不碰文件系统、不依赖 transformers 是否安装
② 词表与卡片必须对齐 卡片的 vocab_size 从**分词器的词表大小**推出来
                     （反过来会让 tokenizer.encode 给出越界的 id，而那不是本课要考的东西）
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from smart_research_agent.hf_integration.bridge import InProcessModel, create_in_process_model
from smart_research_agent.hf_integration.config import ModelCard, parse_config
from smart_research_agent.hf_integration.forward import ModelWeights, make_weights
from smart_research_agent.hf_integration.hub import CacheStore, HubResolver, InMemoryHub
from smart_research_agent.hf_integration.tokenizer import ByteBPETokenizer, build_tiny_tokenizer
from smart_research_agent.hf_integration.types import (
    CONFIG_FILE,
    MERGES_FILE,
    VOCAB_FILE,
    WEIGHTS_FILE,
    Snapshot,
)
from smart_research_agent.hf_source.types import GenerationSettings

#: 主版本与它的 commit（40 位十六进制，与真实的 commit hash 同形）.
MAIN_COMMIT = "1f2e3d4c5b6a7988071625344352617069859a0b"

#: 第二个版本的 commit（用来考"换 revision 会换一个快照目录"）.
SECOND_COMMIT = "0a1b2c3d4e5f60718293a4b5c6d7e8f901234567"

#: 默认的三条文本（**长度两两不同**，否则这一批没有填充可查）.
DEFAULT_TEXTS: tuple[str, ...] = ("hello world", "hey", "hello there world")

#: 默认的提示词.
DEFAULT_PROMPT = "hello world"


def merges_text(tokenizer: ByteBPETokenizer) -> str:
    """把分词器的合并表写成 ``merges.txt`` 的样子（含真实的版本头）."""
    return "#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in tokenizer.merges) + "\n"


def config_payload(model_type: str, vocab: int, **overrides: object) -> dict[str, object]:
    """一份 ``config.json`` 的内容（**vocab 由调用方给，通常来自分词器**）."""
    if model_type == "gpt2":
        payload: dict[str, object] = {
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
            "_name_or_path": "org/tiny",
        }
    elif model_type == "bert":
        payload = {
            "model_type": "bert",
            "vocab_size": vocab,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "max_position_embeddings": 32,
            # BERT 的 ``intermediate_size`` 缺省是固定的 3072（与 hidden 无关），
            # 玩具卡片必须**显式**写成 4·hidden，否则前馈会宽得离谱（也慢得离谱）。
            "intermediate_size": 64,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "type_vocab_size": 2,
            "tie_word_embeddings": True,
            "architectures": ["BertModel"],
            "_name_or_path": "org/tiny",
        }
    else:  # pragma: no cover - 调用方只会给两个架构
        raise ValueError(f"未知架构 {model_type!r}")
    payload.update(overrides)
    return payload


def repo_files(
    model_type: str = "gpt2",
    *,
    tokenizer: ByteBPETokenizer | None = None,
    **overrides: object,
) -> dict[str, bytes]:
    """一个仓库的全部文件（**四个**：配置 + 词表 + 合并表 + 一个假的权重文件）."""
    built = tokenizer if tokenizer is not None else build_tiny_tokenizer()
    payload = config_payload(model_type, built.vocab_size, **overrides)
    return {
        CONFIG_FILE: json.dumps(payload).encode("utf-8"),
        VOCAB_FILE: json.dumps(built.vocab).encode("utf-8"),
        MERGES_FILE: merges_text(built).encode("utf-8"),
        WEIGHTS_FILE: b"\x00\x01\x02\x03\x04\x05\x06\x07",
    }


def make_hub(
    model_type: str = "gpt2",
    *,
    repo_id: str = "org/tiny",
    revisions: dict[str, str] | None = None,
    trees: dict[str, dict[str, bytes]] | None = None,
    tokenizer: ByteBPETokenizer | None = None,
) -> InMemoryHub:
    """造一个内存假仓库（默认一个 ``main`` 版本，也可传多个版本）."""
    mapping = revisions or {"main": MAIN_COMMIT}
    built = tokenizer if tokenizer is not None else build_tiny_tokenizer()
    manifest = trees or {commit: repo_files(model_type, tokenizer=built) for commit in set(mapping.values())}
    return InMemoryHub(
        commits={repo_id: dict(mapping)},
        trees={repo_id: {commit: dict(files) for commit, files in manifest.items()}},
    )


@dataclass
class IntegrationCase:
    """一次"装载 + 调用"的完整零件（测试文件从这里取一切）."""

    model_type: str
    hub: InMemoryHub
    resolver: HubResolver
    snapshot: Snapshot
    card: ModelCard
    weights: ModelWeights
    tokenizer: ByteBPETokenizer
    texts: tuple[str, ...] = DEFAULT_TEXTS
    prompt: str = DEFAULT_PROMPT
    settings: GenerationSettings = field(
        default_factory=lambda: GenerationSettings(max_new_tokens=3, do_sample=False, seed=7)
    )

    def model(self, **kwargs: object) -> InProcessModel:
        """按这个 case 造一个在进程模型（**同一份权重**，不重新初始化）."""
        return create_in_process_model(
            self.card, self.weights, self.tokenizer, settings=self.settings, **kwargs  # type: ignore[arg-type]
        )


def build_case(
    model_type: str = "gpt2",
    *,
    seed: int = 7,
    texts: tuple[str, ...] = DEFAULT_TEXTS,
    prompt: str = DEFAULT_PROMPT,
    settings: GenerationSettings | None = None,
    allow_patterns: tuple[str, ...] | None = None,
    revisions: dict[str, str] | None = None,
    **overrides: object,
) -> IntegrationCase:
    """造一个完整 case：假仓库 → 解析 → 配置 → 分词器 → 权重 → 卡片.

    ``allow_patterns`` 可以用来造"少了某个文件"的快照（例如只要配置与词表）——
    那是考"必需文件缺失"那一条用的。
    """
    tokenizer = build_tiny_tokenizer()
    hub = make_hub(model_type, revisions=revisions)
    resolver = HubResolver(hub=hub, store=CacheStore(), cache_dir="C:/cache/hub")
    repo_id = next(iter(hub.commits))
    revision = "main"
    snapshot = resolver.resolve(
        repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_files_only=False,
    )
    payload = json.loads(resolver.read_text(snapshot, CONFIG_FILE))
    payload.update(overrides)
    card = parse_config(payload, name=f"tiny-{model_type}")
    weights = make_weights(card, seed=seed)
    tokenizer = ByteBPETokenizer.from_snapshot(resolver, snapshot, name=f"tiny-{model_type}")
    return IntegrationCase(
        model_type=model_type,
        hub=hub,
        resolver=resolver,
        snapshot=snapshot,
        card=card,
        weights=weights,
        tokenizer=tokenizer,
        texts=texts,
        prompt=prompt,
        settings=settings or GenerationSettings(max_new_tokens=3, do_sample=False, seed=7),
    )


def both_cases(**kwargs: object) -> tuple[IntegrationCase, IntegrationCase]:
    """两个架构各造一个（**同一个种子**，因此差别只来自架构）."""
    return build_case("gpt2", **kwargs), build_case("bert", **kwargs)


__all__ = [
    "DEFAULT_PROMPT",
    "DEFAULT_TEXTS",
    "MAIN_COMMIT",
    "SECOND_COMMIT",
    "IntegrationCase",
    "both_cases",
    "build_case",
    "config_payload",
    "make_hub",
    "merges_text",
    "repo_files",
]
