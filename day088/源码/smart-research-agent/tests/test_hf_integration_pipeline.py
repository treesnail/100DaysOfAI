"""``hf_integration.pipeline`` 与 ``bridge``：两条管线与"接进来"（day086 / M7-D10）."""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration import bridge, errors, pipeline
from smart_research_agent.hf_integration.bridge import (
    ALL_BACKENDS,
    BACKEND_TRAITS,
    IN_PROCESS,
    PLACEHOLDER_KEYS,
    PROTOCOL_SURFACE,
    InProcessModel,
    InProcessSpec,
    choose_backend,
    create_in_process_model,
    render_messages,
    trait_line,
)
from smart_research_agent.hf_integration.pipeline import (
    HEAD_OF_TASK,
    check_head_available,
    feature_extraction,
    head_line,
    run_both,
    text_generation,
)
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_BERT,
    ARCHITECTURE_GPT2,
    TASK_FEATURE_EXTRACTION,
    TASK_HEADS,
    TASK_KINDS,
    TASK_TEXT_GENERATION,
)
from smart_research_agent.hf_source.types import GenerationSettings
from smart_research_agent.llm.base import Message
from smart_research_agent.llm.local_model import OLLAMA, VLLM

from tests.hf_integration_samples import build_case

SETTINGS = GenerationSettings(max_new_tokens=3, do_sample=False, seed=7)


# --------------------------------------------------------------------------- 生成管线


def test_text_generation_returns_decoded_new_segment() -> None:
    """生成只解码**新生成的那一段**（prompt 不拼回去：那会让"生成了什么"消失）."""
    case = build_case("gpt2")
    output = text_generation(case.card, case.weights, case.tokenizer, ("hello",), SETTINGS)
    assert output.batch_size == 1
    assert output.strategy == "greedy"
    assert len(output.results[0].generated) == 3
    assert output.prompts == ("hello",)
    assert isinstance(output.texts[0], str)
    assert output.to_dict()["generated_lengths"] == [3]


def test_text_generation_runs_each_sample_separately() -> None:
    """逐条跑：长度不齐是正常的（各自停在 eos）."""
    case = build_case("gpt2")
    output = text_generation(
        case.card, case.weights, case.tokenizer, ("hello", "hey", "hi"), SETTINGS
    )
    assert len(output.generated_lengths()) == 3
    assert all(length >= 1 for length in output.generated_lengths())


def test_text_generation_fills_eos_from_the_tokenizer() -> None:
    """tokenizer 的 eos 会被填进生成配置（**不覆盖**调用方显式给的那个）."""
    case = build_case("gpt2")
    output = text_generation(case.card, case.weights, case.tokenizer, ("hello",), SETTINGS)
    assert output.results[0].notes  # 跑满预算或命中 eos 都记了一笔
    explicit = GenerationSettings(max_new_tokens=2, do_sample=False, eos_token=None, seed=3)
    assert text_generation(case.card, case.weights, case.tokenizer, ("hello",), explicit)
    fixed = GenerationSettings(max_new_tokens=2, do_sample=False, eos_token=0, seed=3)
    out = text_generation(case.card, case.weights, case.tokenizer, ("hello",), fixed)
    assert out.results[0].token_ids[:1] == tuple(case.tokenizer.encode("hello"))


def test_text_generation_empty_prompt_is_rejected() -> None:
    """两处空输入都要拒绝：没有提示词、以及编出 0 个 token 的提示词."""
    case = build_case("gpt2")
    with pytest.raises(errors.ParameterError, match="0 条提示词"):
        text_generation(case.card, case.weights, case.tokenizer, (), SETTINGS)
    with pytest.raises(errors.ParameterError, match="0 个 token"):
        text_generation(case.card, case.weights, case.tokenizer, ("",), SETTINGS)


def test_feature_extraction_shapes_and_pooling() -> None:
    """一次编码 + 一次前向 + 一次池化，宽度必须等于配置里的 hidden."""
    case = build_case("bert")
    output = feature_extraction(case.card, case.weights, case.tokenizer, case.texts)
    assert output.pooled.batch_size == len(case.texts)
    assert output.pooled.dim == case.card.hidden
    assert output.lengths == tuple(
        len(case.tokenizer.encode(text)) for text in case.texts
    )
    assert output.batch.padding_ratio > 0.0
    assert output.to_dict()["pooling"]["strategy"] == "mean"


def test_feature_extraction_variants() -> None:
    """三个旋钮各改一次：池化策略、是否归一、是否截断."""
    case = build_case("gpt2")
    for strategy in ("last_token", "mean", "max"):
        output = feature_extraction(
            case.card, case.weights, case.tokenizer, case.texts, pooling=strategy
        )
        assert output.pooled.strategy == strategy
    normalized = feature_extraction(
        case.card, case.weights, case.tokenizer, case.texts, normalize=True
    )
    assert normalized.pooled.normalized is True
    truncated = feature_extraction(
        case.card, case.weights, case.tokenizer, case.texts, max_length=3
    )
    assert all(row.length == 3 for row in truncated.batch.rows)
    assert any(row.truncated for row in truncated.batch.rows)


def test_feature_extraction_rejects_empty_input() -> None:
    """空输入 ⇒ ParameterError（没有文本就没有向量）."""
    case = build_case("gpt2")
    with pytest.raises(errors.ParameterError, match="0 条文本"):
        feature_extraction(case.card, case.weights, case.tokenizer, ())


def test_head_check_blocks_generation_on_bert() -> None:
    """生成要一个"hidden → 词表"的输出投影，且**只在因果卡片上**才有意义."""
    bert = build_case("bert")
    with pytest.raises(errors.ConfigError, match="没有因果掩码"):
        check_head_available(bert.card, TASK_TEXT_GENERATION, bert.weights)
    gpt2 = build_case("gpt2")
    assert check_head_available(gpt2.card, TASK_TEXT_GENERATION, gpt2.weights) is None
    with pytest.raises(errors.ParameterError, match="未知的任务"):
        check_head_available(gpt2.card, "summarization", gpt2.weights)


def test_task_tables_are_closed() -> None:
    """两条管线的名单、说明、所需模型头三张表逐键对齐."""
    assert set(TASK_KINDS) == {TASK_TEXT_GENERATION, TASK_FEATURE_EXTRACTION}
    assert set(TASK_HEADS) == set(TASK_KINDS)
    assert HEAD_OF_TASK == TASK_HEADS
    assert "lm_head" in head_line(build_case("gpt2").card, TASK_TEXT_GENERATION)
    assert "无" in head_line(build_case("bert").card, TASK_FEATURE_EXTRACTION)


def test_run_both_uses_the_same_weights() -> None:
    """两条管线共用**同一份权重对象**（"换管线不改模型"的可断言形式）."""
    case = build_case("gpt2")
    generation, features = run_both(
        case.card, case.weights, case.tokenizer, case.prompt, SETTINGS
    )
    assert generation.results[0].strategy == "greedy"
    assert features.pooled.dim == case.card.hidden


# --------------------------------------------------------------------------- 在进程模型


def test_in_process_spec_from_card() -> None:
    """档案的上下文窗口来自**位置表**（不是部署参数）——这是它和 Ollama 的差别."""
    case = build_case("gpt2")
    spec = InProcessSpec.of(case.card)
    assert spec.backend == IN_PROCESS
    assert spec.context_length == case.card.positions
    assert spec.parameters == case.card.parameter_count
    assert spec.vocab_size == case.card.vocab
    assert spec.supports_vision is False
    assert spec.note
    assert spec.to_dict()["model_type"] == ARCHITECTURE_GPT2


def test_in_process_spec_validates_its_fields() -> None:
    """三条护栏：未知后端、空名字、非正窗口."""
    with pytest.raises(ValueError, match="未知的后端"):
        InProcessSpec(name="x", backend="llama.cpp", context_length=8)
    with pytest.raises(ValueError, match="名字"):
        InProcessSpec(name="", context_length=8)
    with pytest.raises(ValueError, match="context_length"):
        InProcessSpec(name="x", context_length=0)


def test_in_process_model_matches_the_protocol_surface() -> None:
    """八项方法面逐名对齐——这是"接进来"的可断言形式."""
    model = build_case("gpt2").model()
    for name in PROTOCOL_SURFACE:
        assert hasattr(model, name), name
    assert PROTOCOL_SURFACE == (
        "chat",
        "stream",
        "chat_with_tools",
        "is_available",
        "describe",
        "model_name",
        "backend",
        "context_length",
    )


def test_in_process_model_readings() -> None:
    """四个读数：名字、后端、窗口、可用性（在进程路线**永远可用**）."""
    case = build_case("gpt2")
    model = case.model()
    assert model.model_name == case.card.name
    assert model.backend == IN_PROCESS
    assert model.context_length == case.card.positions
    assert model.is_available() is True
    assert model.is_available(timeout=0.001) is True
    assert model.supports_vision is False
    described = model.describe()
    assert described["context_length"] == case.card.positions
    assert described["layers"] == case.card.layers
    assert described["tie_word_embeddings"] is True


def test_in_process_model_chat_and_stream_agree() -> None:
    """day039 的契约：流式片段拼接起来必须等于一次 chat 的完整回复."""
    model = build_case("gpt2").model()
    messages = [Message(role="user", content="hello")]
    full = model.chat(messages, temperature=0.7, max_tokens=3)
    pieces = tuple(model.stream(messages, temperature=0.7, max_tokens=3))
    assert "".join(pieces) == full
    assert model.calls.count("chat") == 1
    assert model.calls.count("stream") == 1


def test_in_process_model_temperature_switches_to_sampling() -> None:
    """``temperature != 1.0`` 会把生成切成采样（**不静默忽略**这个参数）."""
    model = build_case("gpt2").model()
    settings = model._settings_for(temperature=0.5, max_tokens=4)
    assert settings.do_sample is True
    assert settings.temperature == 0.5
    greedy = model._settings_for(temperature=1.0, max_tokens=4)
    assert greedy.do_sample is False
    assert greedy.max_new_tokens == 4
    clamped = model._settings_for(temperature=1.0, max_tokens=10_000)
    assert clamped.max_new_tokens == 64


def test_in_process_model_does_not_pretend_to_support_tools() -> None:
    """不实现 function calling 时**明确抛错**，而不是静默降级成一段自由文本."""
    model = build_case("gpt2").model()
    with pytest.raises(NotImplementedError, match="function calling"):
        model.chat_with_tools([Message(role="user", content="hi")], [])


def test_in_process_model_embed_and_hidden() -> None:
    """两个在进程路线特有的读数：向量与 hidden states（都只留真实长度）."""
    case = build_case("bert")
    model = case.model(pooling="mean")
    vectors = model.embed(case.texts)
    assert len(vectors) == len(case.texts)
    assert all(len(vector) == case.card.hidden for vector in vectors)
    hidden = model.hidden_for(case.texts[0])
    assert len(hidden) == len(case.tokenizer.encode(case.texts[0]))
    assert "embed" in model.calls and "hidden_for" in model.calls


def test_render_messages_is_a_plain_text_protocol() -> None:
    """消息渲染成四行标记（不依赖任何服务端的模板机制）."""
    rendered = render_messages(
        [Message(role="system", content="s"), Message(role="user", content="u")]
    )
    assert rendered == "<|system|>s\n<|user|>u\n<|assistant|>"


def test_create_in_process_model_is_a_factory() -> None:
    """工厂：未传的字段一律回落到卡片."""
    case = build_case("bert")
    model = create_in_process_model(case.card, case.weights, case.tokenizer)
    assert isinstance(model, InProcessModel)
    assert model.spec.pooling == "mean"
    assert model.spec.normalize is False
    assert model.spec.context_length == case.card.positions


def test_factory_rejects_mismatched_card_and_weights() -> None:
    """权重与卡片不是同一个模型时当场拒绝（两份东西必须来自同一次装载）."""
    gpt2 = build_case("gpt2")
    with pytest.raises(ValueError, match="同一次装载"):
        InProcessModel(
            build_case("bert").card, gpt2.weights, gpt2.tokenizer  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- 三条路线


def test_trait_table_covers_three_backends() -> None:
    """三条路线各自的事实：服务端、离线、视觉、工具、窗口来源."""
    assert set(ALL_BACKENDS) == {IN_PROCESS, OLLAMA, VLLM}
    assert set(BACKEND_TRAITS) == set(ALL_BACKENDS)
    assert BACKEND_TRAITS[IN_PROCESS]["server"] is False
    assert BACKEND_TRAITS[OLLAMA]["server"] is True
    assert BACKEND_TRAITS[OLLAMA]["tools"] is True
    assert BACKEND_TRAITS[IN_PROCESS]["tools"] is False
    assert PLACEHOLDER_KEYS[IN_PROCESS] is None
    assert PLACEHOLDER_KEYS[OLLAMA] == "ollama"
    assert PLACEHOLDER_KEYS[VLLM] == "EMPTY"
    assert "4096" in str(BACKEND_TRAITS[OLLAMA]["context_source"])


def test_trait_line_and_its_guard() -> None:
    """一行读数；未知后端当场拒绝."""
    line = trait_line(IN_PROCESS)
    assert IN_PROCESS in line and "（不需要）" in line
    assert OLLAMA in trait_line(OLLAMA)
    with pytest.raises(ValueError, match="未知后端"):
        trait_line("llama.cpp")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, IN_PROCESS),
        ({"needs_tools": True}, OLLAMA),
        ({"needs_tools": True, "multi_user": True}, VLLM),
        ({"needs_tools": True, "needs_vision": True}, OLLAMA),
        ({"needs_vision": True}, OLLAMA),
        ({"multi_user": True}, VLLM),
        ({"needs_batching": True}, VLLM),
        ({"long_context": True}, OLLAMA),
        ({"long_context": True, "multi_user": True}, VLLM),
    ],
)
def test_choose_backend_rules(kwargs: dict[str, bool], expected: str) -> None:
    """九种需求组合各选一条路线，而且**每一条都带理由**."""
    choice = choose_backend(**kwargs)
    assert choice.backend == expected
    assert choice.reasons
    assert choice.line().startswith(f"[{expected}]")
    assert choice.to_dict()["backend"] == expected
