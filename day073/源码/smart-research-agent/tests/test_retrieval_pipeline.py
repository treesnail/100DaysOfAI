"""day066 ``retrieval.pipeline`` 的单元测试：护栏、提示词、降级注记（M6-D5）.

这一层只有四步（检索 → 判空 → 打包 → 生成），但它的价值几乎全在**第二步的位置**：

```text
检索为空 → 一次 LLM 都不调（llm_called=False、citations=()、context=None）
```

因此本文件里最重要的一组用例不是"命中时提示词长什么样"，而是
**用两个"一旦被调用就抛异常"的假 LLM 证明护栏生效**（``ExplodingLLM``）：
覆盖 ``no_data`` 与 ``filtered_out`` 两种空原因，顺带覆盖 ``below_threshold``。
判定读的是 ``llm_called`` 与假 LLM 的 ``calls`` 计数，而不是"answer 是否等于兜底串"
——兜底串将来改一个字，字符串比较那种判定就会静默失效（见 ``RagAnswer`` 的说明）。

其余三组：

```text
提示词      命中时模型收到了什么（含 [1]、含问题原文、含四条约束句）
降级注记    截断 / 丢尾 / 索引漂移三条各自可读，且都不阻断链路
构造期校验  模板缺占位符、拼错覆盖键、预算与温度越界都必须当场报错
```

全部离线、确定性：库是 ``FlatVectorStore`` + ``TableEmbedding``（见
``tests/retrieval_samples.py``），模型是 ``MockLLM`` 或本文件里的两个假实现，
不联网、不写仓库外文件。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.retrieval.context import TRUNCATION_MARKER
from smart_research_agent.retrieval.errors import ContextError, IndexStateError, QueryError
from smart_research_agent.retrieval.generation import RAG_PROMPT_V2
from smart_research_agent.retrieval.pipeline import (
    FALLBACK_NO_CONTEXT,
    OVERRIDE_KEYS,
    RAG_ANSWER_PROMPT,
    RAG_ANSWER_PROMPT_V1,
    RAG_ANSWER_PROMPT_VERSION,
    REQUIRED_PROMPT_FIELDS,
    RagAnswer,
    RagPipeline,
)
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import (
    EMPTY_REASON_FILTERED_OUT,
    EMPTY_REASON_NO_DATA,
    EMPTY_REASON_NONE,
    RetrievalQuery,
)
from smart_research_agent.vectorstore.flat import FlatVectorStore
from tests.retrieval_samples import (
    QUERY_TEXTS,
    RECORD_IDS,
    TableEmbedding,
    empty_store,
    sample_manifest,
    sample_store,
)

#: 命中路径用的查询（它的向量由 ``TableEmbedding`` 给定，见样本模块）。
AXIS = QUERY_TEXTS["axis"]
TILT = QUERY_TEXTS["tilt"]

#: V1 提示词里的四条约束句（每条都在挡一种真实失败，见 ``pipeline`` 的注释）。
CONSTRAINT_PHRASES: tuple[str, ...] = (
    "只使用上面资料片段中出现的信息",
    "资料中没有相关内容",
    "[1]、[2]",
    "不要编造",
)


# --------------------------------------------------------------------------- #
# 假 LLM 与装配助手
# --------------------------------------------------------------------------- #


class RecordingLLM(BaseLLM):
    """记录入参与温度的假 LLM：用来核对"这次到底怎么问的"."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """把入参原样记下来，再按脚本返回一条回复."""
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self._replies.pop(0) if self._replies else "（无脚本回复）"


class ExplodingLLM(BaseLLM):
    """**一旦被调用就抛异常**的假 LLM：护栏是否生效由它来作证.

    用它而不是"看 answer 等不等于兜底串"：护栏失效时这里的异常会**立刻**炸出来，
    而字符串比较那种判定只会在兜底文案改动之后静默失效（见 ``RagAnswer``）。
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """被调用即失败：检索为空时不该走到这里."""
        self.calls += 1
        raise AssertionError("检索为空时不许调用 LLM：护栏被绕过了")


def sample_retriever(
    store: FlatVectorStore | None = None,
    embedding: EmbeddingProvider | None = None,
    **overrides: Any,
) -> Retriever:
    """样本库 + ``TableEmbedding`` 的检索器（构造参数逐个可覆盖）."""
    return Retriever(
        store if store is not None else sample_store(),
        embedding if embedding is not None else TableEmbedding(),
        **overrides,
    )


def make_pipeline(
    *,
    retriever: Retriever | None = None,
    store: FlatVectorStore | None = None,
    llm: BaseLLM | None = None,
    **kwargs: Any,
) -> RagPipeline:
    """装配一条 RAG 链路（默认：样本库 + 一句带 ``[1]`` 的脚本回复）."""
    resolved_retriever = (
        retriever if retriever is not None else sample_retriever(store=store)
    )
    resolved_llm = llm if llm is not None else MockLLM(responses=["答案 [1]"])
    return RagPipeline(resolved_retriever, resolved_llm, **kwargs)


def drift_retriever(*, strict: bool = False) -> Retriever:
    """**清单与库矛盾**的检索器：先按整库建清单，再手工删掉一条记录.

    这是"上一版清单过期"的最小复现（day065 的漂移判定在这里被检索器看见）。
    """
    store = sample_store()
    manifest = sample_manifest(store)
    store.delete([RECORD_IDS[0]])
    return Retriever(store, TableEmbedding(), manifest=manifest, strict_index=strict)


def prompt_of(llm: BaseLLM) -> str:
    """取最后一条提示词（两个假 LLM 的 ``calls`` 形状不同，这里收敛一下）."""
    last = llm.calls[-1]  # type: ignore[attr-defined]
    messages = last["messages"] if isinstance(last, dict) else last
    return str(messages[0].content)


# --------------------------------------------------------------------------- #
# 常量与提示词版本
# --------------------------------------------------------------------------- #


class TestPromptConstants:
    """提示词也要版本化（day026 的约定）：历史版本永不删 + 版本号可被报告引用."""

    def test_the_current_prompt_is_v2(self) -> None:
        """``RAG_ANSWER_PROMPT`` 必须**指向当前版本**（day069 起是 v2）.

        day069 把默认模板从 v1 换成 v2（四段式 + 固定拒答句式），v1 作为历史版本
        继续导出。这两句断言钉的正是"新版生效、旧版仍在"这一对事实
        （历史版本永不删的约定见 ``generation``，模板现在住在那边）。
        """
        assert RAG_ANSWER_PROMPT is RAG_PROMPT_V2
        assert RAG_ANSWER_PROMPT_VERSION == "v2"
        assert RAG_ANSWER_PROMPT is not RAG_ANSWER_PROMPT_V1

    def test_the_required_placeholders_are_context_and_question(self) -> None:
        """构造期校验的就是这两个名字."""
        assert REQUIRED_PROMPT_FIELDS == ("context", "question")
        assert "{context}" in RAG_ANSWER_PROMPT_V1
        assert "{question}" in RAG_ANSWER_PROMPT_V1

    def test_the_override_keys_are_the_three_documented_ones(self) -> None:
        """允许逐次覆盖的键只有三个（拼错的键必须被拒，见 ``TestOverrides``）."""
        assert OVERRIDE_KEYS == ("max_context_chars", "per_hit_chars", "temperature")

    def test_the_fallback_answer_offers_three_ways_out(self) -> None:
        """兜底答复必须给下一步动作，而不是只宣告一次失败."""
        assert "知识库中没有检索到" in FALLBACK_NO_CONTEXT
        assert "换一种说法" in FALLBACK_NO_CONTEXT
        assert "放宽过滤条件" in FALLBACK_NO_CONTEXT
        assert "索引版本" in FALLBACK_NO_CONTEXT

    def test_v1_prompt_states_the_four_constraints(self) -> None:
        """四条约束各自挡一种真实失败（见 ``pipeline`` 里那段注释）."""
        for phrase in CONSTRAINT_PHRASES:
            assert phrase in RAG_ANSWER_PROMPT_V1


# --------------------------------------------------------------------------- #
# 命中路径
# --------------------------------------------------------------------------- #


class TestHitPath:
    """命中时：打包 → 渲染 → 调模型 → 把引用与上下文一起交回去."""

    def test_the_llm_is_called_once_with_a_single_user_message(self) -> None:
        """一次问答只发一条 user 消息（模板里没有 system 段）."""
        llm = MockLLM(responses=["答案 [1]"])
        make_pipeline(llm=llm).answer(AXIS)

        assert len(llm.calls) == 1
        assert len(llm.calls[0]) == 1
        assert llm.calls[0][0].role == "user"

    def test_the_prompt_contains_the_packed_context(self) -> None:
        """送进模型的必须**逐字**是打包出来的那段文本（不是重新拼一遍）."""
        llm = RecordingLLM(["答案 [1]"])
        answer = make_pipeline(llm=llm).answer(AXIS)

        assert answer.context is not None
        assert answer.context.text in prompt_of(llm)
        assert answer.context.citations[0].summary_line().split(" score=")[0] in prompt_of(llm)

    def test_the_prompt_contains_the_numbered_blocks(self) -> None:
        """提示词里要出现 ``[1] 来源 › 标题`` 这样的块头（编号已就位）."""
        llm = RecordingLLM(["答案 [1]"])
        answer = make_pipeline(llm=llm).answer(AXIS)

        content = prompt_of(llm)
        assert answer.context is not None
        assert answer.context.text in content
        assert "[1] docs/检索手册.md" in content
        assert "[2]" in content

    def test_the_prompt_contains_the_question_verbatim(self) -> None:
        """问题原文必须逐字出现在提示词里（编号之外的另一半锚点）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm).answer(AXIS)

        assert AXIS in prompt_of(llm)

    def test_the_prompt_contains_the_constraints(self) -> None:
        """四条约束句要跟着模板一起进提示词."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm).answer(AXIS)

        content = prompt_of(llm)
        for phrase in CONSTRAINT_PHRASES:
            assert phrase in content

    def test_the_prompt_has_no_placeholder_left(self) -> None:
        """渲染之后不该再留下 ``{context}`` / ``{question}``（缺占位符在构造期已拦）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm).answer(AXIS)

        content = prompt_of(llm)
        assert "{context}" not in content
        assert "{question}" not in content

    def test_llm_called_is_true(self) -> None:
        """命中路径上护栏不该拦任何东西."""
        answer = make_pipeline().answer(AXIS)

        assert answer.llm_called is True
        assert answer.answer == "答案 [1]"

    def test_citations_match_the_hits_one_to_one(self) -> None:
        """引用与命中逐条对齐（编号 i 对应第 i 条命中）."""
        answer = make_pipeline().answer(AXIS)

        assert answer.retrieval is not None
        assert [citation.record_id for citation in answer.citations] == answer.retrieval.ids()
        assert [citation.marker for citation in answer.citations] == list(
            range(1, answer.retrieval.count + 1)
        )

    def test_citations_are_the_context_citations(self) -> None:
        """答案里的引用与上下文里的引用必须是同一份（不是各算一遍）."""
        answer = make_pipeline().answer(AXIS)

        assert answer.context is not None
        assert answer.citations == answer.context.citations

    def test_the_answer_carries_the_context_and_the_retrieval(self) -> None:
        """一次回答要能独立回答"模型看到了什么"与"这次检索给了什么"."""
        answer = make_pipeline().answer(AXIS)

        assert answer.context is not None and answer.retrieval is not None
        assert answer.retrieval.empty_reason == EMPTY_REASON_NONE
        assert answer.retrieval.query.text == AXIS

    def test_all_hits_fit_the_default_budget(self) -> None:
        """默认预算（2400 字 / 单条 800 字）下 5 条命中一条都不丢."""
        answer = make_pipeline().answer(AXIS)

        assert answer.context is not None
        assert answer.context.count == 5
        assert answer.context.dropped_hits == ()
        assert answer.notes == (), "没有降级就不该有注记"

    def test_temperature_defaults_to_zero(self) -> None:
        """默认温度 0.0：RAG 问答是有依据的复述，让它照抄依据."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm).answer(AXIS)

        assert llm.calls[0]["temperature"] == 0.0

    def test_a_configured_temperature_is_forwarded(self) -> None:
        """构造期的温度被原样转发给模型（这一层不改它）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm, temperature=0.2).answer(AXIS)

        assert llm.calls[0]["temperature"] == 0.2

    def test_the_question_is_normalized_before_the_prompt(self) -> None:
        """送进提示词的是**已规范化**的问题（与 ``retrieval.query.text`` 同一份）."""
        llm = RecordingLLM(["答案 [1]"])
        answer = make_pipeline(llm=llm).answer(f"  {AXIS}  ")

        assert answer.question == AXIS
        assert AXIS in prompt_of(llm)
        assert "  检索手册" not in prompt_of(llm)

    def test_a_query_object_limits_the_citations(self) -> None:
        """用 ``RetrievalQuery`` 提问时，条数与引用都跟着它走."""
        answer = make_pipeline().answer(RetrievalQuery(text=AXIS, top_k=2))

        assert answer.retrieval is not None and answer.retrieval.count == 2
        assert len(answer.citations) == 2

    def test_the_prompt_version_is_the_current_one(self) -> None:
        """报告与评估按版本号分组，因此它必须跟着每一次回答走."""
        answer = make_pipeline().answer(AXIS)

        assert answer.to_dict()["prompt_version"] == RAG_ANSWER_PROMPT_VERSION


# --------------------------------------------------------------------------- #
# 硬护栏：检索为空 → 一次 LLM 都不调
# --------------------------------------------------------------------------- #


class TestNoHitGuard:
    """本课最重要的护栏："检索不到"不许变成"模型自由发挥"."""

    def test_no_data_does_not_call_the_llm(self) -> None:
        """空库（``no_data``）时假 LLM 一次都没被调用."""
        llm = ExplodingLLM()
        pipeline = make_pipeline(store=empty_store(), llm=llm)

        answer = pipeline.answer(AXIS)

        assert llm.calls == 0, "库是空的就没有片段可给，模型不该被叫醒"
        assert answer.llm_called is False

    def test_no_data_returns_the_fallback_answer(self) -> None:
        """答案直接取 ``FALLBACK_NO_CONTEXT``（它带着三条出路）."""
        answer = make_pipeline(store=empty_store(), llm=ExplodingLLM()).answer(AXIS)

        assert answer.answer == FALLBACK_NO_CONTEXT
        assert "换一种说法" in answer.answer

    def test_no_data_returns_no_citations_and_no_context(self) -> None:
        """``citations=()`` 且 ``context is None``：没有片段就没有引用，也没有上下文."""
        answer = make_pipeline(store=empty_store(), llm=ExplodingLLM()).answer(AXIS)

        assert answer.citations == ()
        assert answer.context is None

    def test_no_data_notes_explain_the_reason(self) -> None:
        """注记要写清"为什么没调模型"（原因 + 护栏 + 出路三条）."""
        answer = make_pipeline(store=empty_store(), llm=ExplodingLLM()).answer(AXIS)

        assert len(answer.notes) == 3
        assert any(f"empty_reason={EMPTY_REASON_NO_DATA}" in note for note in answer.notes)
        assert any("没有调用 LLM" in note for note in answer.notes)

    def test_filtered_out_does_not_call_the_llm(self) -> None:
        """过滤器把候选筛没（``filtered_out``）时同样一次都不调."""
        llm = ExplodingLLM()
        pipeline = make_pipeline(llm=llm)
        query = RetrievalQuery(text=AXIS, where={"strategy": "不存在的策略"})

        answer = pipeline.answer(query)

        assert answer.retrieval is not None
        assert answer.retrieval.empty_reason == EMPTY_REASON_FILTERED_OUT
        assert llm.calls == 0
        assert answer.llm_called is False
        assert answer.answer == FALLBACK_NO_CONTEXT

    def test_filtered_out_notes_explain_the_reason(self) -> None:
        """注记里的原因必须是 ``filtered_out``（不是笼统的"没有找到"）."""
        answer = make_pipeline(llm=ExplodingLLM()).answer(
            RetrievalQuery(text=AXIS, where={"strategy": "不存在的策略"})
        )

        assert any(f"empty_reason={EMPTY_REASON_FILTERED_OUT}" in note for note in answer.notes)

    def test_below_threshold_also_skips_the_llm(self) -> None:
        """阈值把命中全切了（``below_threshold``）也走同一条空结果分支."""
        llm = ExplodingLLM()
        answer = make_pipeline(llm=llm).answer(RetrievalQuery(text=AXIS, min_score=1.5))

        assert answer.retrieval is not None
        assert answer.retrieval.empty_reason == "below_threshold"
        assert answer.retrieval.count == 0
        assert llm.calls == 0

    def test_an_empty_result_still_carries_the_retrieval_record(self) -> None:
        """空结果也要能回答"本来打算取多深、有没有过滤"."""
        answer = make_pipeline(store=empty_store(), llm=ExplodingLLM()).answer(AXIS)

        assert answer.retrieval is not None
        assert answer.retrieval.is_empty is True
        assert answer.retrieval.fetch_k == 15, "深度按过取倍率算过：5 × 3"

    def test_the_guard_applies_to_every_empty_reason(self) -> None:
        """三种空原因（no_data / filtered_out / below_threshold）都不调模型."""
        for query, pipeline in (
            (AXIS, make_pipeline(store=empty_store(), llm=ExplodingLLM())),
            (
                RetrievalQuery(text=AXIS, where={"strategy": "不存在的策略"}),
                make_pipeline(llm=ExplodingLLM()),
            ),
            (
                RetrievalQuery(text=AXIS, min_score=1.5),
                make_pipeline(llm=ExplodingLLM()),
            ),
        ):
            answer = pipeline.answer(query)
            assert answer.llm_called is False
            assert pipeline.llm.calls == 0  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 三类降级注记
# --------------------------------------------------------------------------- #


class TestDegradeNotes:
    """三条降级各记一句（处置动作不同，因此不能合成一个布尔标志）."""

    def test_per_hit_truncation_is_noted(self) -> None:
        """单条被上限截断 → 有一条"可能缺一句结论"的注记."""
        answer = make_pipeline(per_hit_chars=50).answer(AXIS)

        assert answer.context is not None
        assert answer.context.truncated_hits >= 1
        assert any("被单块上限 50 字截断" in note for note in answer.notes)

    def test_the_truncation_note_names_the_cap(self) -> None:
        """注记里要带上那个上限的数字（否则不知道该调哪个参数）."""
        answer = make_pipeline(per_hit_chars=50).answer(AXIS)

        assert any("settings.retrieval_per_hit_chars" in note for note in answer.notes)
        assert answer.context is not None
        assert answer.context.truncated_hits == answer.context.count, "这次五条都被切过"

    def test_truncation_is_visible_in_the_prompt(self) -> None:
        """被切过的片段在提示词里带着截断标记（模型要能看出这是半截的）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm, per_hit_chars=50).answer(AXIS)

        assert TRUNCATION_MARKER in prompt_of(llm)

    def test_a_dropped_tail_is_noted(self) -> None:
        """预算放不下 → 有一条"从尾部丢掉了 N 条"的注记."""
        answer = make_pipeline(max_context_chars=120).answer(AXIS)

        assert answer.context is not None
        assert len(answer.context.dropped_hits) >= 1
        assert any("从尾部丢掉" in note for note in answer.notes)

    def test_the_drop_note_names_the_dropped_ids(self) -> None:
        """注记要点名被丢掉的那几条（报告里能一眼看出少了谁）."""
        answer = make_pipeline(max_context_chars=120).answer(AXIS)

        assert answer.context is not None
        first_dropped = answer.context.dropped_hits[0].record_id
        assert any(first_dropped in note for note in answer.notes)
        assert any("settings.retrieval_max_context_chars" in note for note in answer.notes)

    def test_dropping_the_tail_still_calls_the_llm(self) -> None:
        """丢尾只是降级，**不是**空结果：模型照样被调用（与护栏形成对照）."""
        llm = RecordingLLM(["答案 [1]"])
        answer = make_pipeline(llm=llm, max_context_chars=120).answer(AXIS)

        assert answer.llm_called is True
        assert len(llm.calls) == 1
        assert any("从尾部丢掉" in note for note in answer.notes)

    def test_index_drift_is_noted(self) -> None:
        """清单与库矛盾（漂移）→ 有一条"可能少召回"的注记."""
        answer = make_pipeline(retriever=drift_retriever()).answer(AXIS)

        assert answer.retrieval is not None
        assert answer.retrieval.index_state.has_drift is True
        assert any("索引漂移" in note for note in answer.notes)

    def test_drift_does_not_block_the_answer(self) -> None:
        """漂移只观测不阻断：结果照给，但"这次可能不完整"必须跟着结果走."""
        answer = make_pipeline(retriever=drift_retriever()).answer(AXIS)

        assert answer.llm_called is True
        assert answer.context is not None and answer.context.count >= 1
        assert any("可能过期的清单" in note for note in answer.notes)
        assert any("重建清单" in note for note in answer.notes)

    def test_strict_index_raises_before_any_llm_call(self) -> None:
        """严格模式下漂移直接拒绝服务——而且是在**调到模型之前**."""
        llm = ExplodingLLM()
        pipeline = make_pipeline(retriever=drift_retriever(strict=True), llm=llm)

        with pytest.raises(IndexStateError):
            pipeline.answer(AXIS)

        assert llm.calls == 0

    def test_an_empty_reply_is_noted(self) -> None:
        """模型返回空串也要记一句（它不是"知识库里没有"，两件事别混）."""
        answer = make_pipeline(llm=MockLLM(responses=[""])).answer(AXIS)

        assert answer.llm_called is True
        assert answer.answer == ""
        assert any("空回复" in note for note in answer.notes)

    def test_the_happy_path_has_no_notes(self) -> None:
        """没有降级就没有注记（注记多了会被读的人忽略）."""
        answer = make_pipeline().answer(AXIS)

        assert answer.notes == ()


# --------------------------------------------------------------------------- #
# 提示词模板的构造期校验
# --------------------------------------------------------------------------- #


class TestPromptTemplateValidation:
    """缺占位符的模板渲染出来是一段**没有资料**的提示词，而它不会报错 —— 因此构造期拦."""

    def test_prompt_without_placeholders_is_rejected(self) -> None:
        """一个占位符都没有的模板当场被拒（消息里点名两个缺失项）."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt="只依据资料回答。")

        message = str(excinfo.value)
        assert "缺少占位符" in message
        assert "['context', 'question']" in message

    def test_prompt_missing_only_the_question_is_rejected(self) -> None:
        """只缺 ``{question}`` 也不行：模型会不知道在回答什么."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt="资料片段：{context}")

        assert "缺少占位符 ['question']" in str(excinfo.value)

    def test_prompt_missing_only_the_context_is_rejected(self) -> None:
        """只缺 ``{context}`` 更严重：模型看不到任何资料."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt="问题是：{question}")

        assert "缺少占位符 ['context']" in str(excinfo.value)

    def test_escaped_braces_are_not_placeholders(self) -> None:
        """``{{context}}`` 是字面量而不是占位符（因此那种模板照样被拒）."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt="{{context}} 与 {{question}}")

        assert "缺少占位符" in str(excinfo.value)

    def test_a_malformed_template_is_rejected(self) -> None:
        """单独的 ``{`` 不是合法格式串（不拦的话会在运行期炸，而那时检索已经跑完）."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt="资料 {")

        assert "不是合法的格式串" in str(excinfo.value)

    def test_a_blank_prompt_is_rejected(self) -> None:
        """空模板等于"让模型自由发挥"——恰恰是这条链路要拦的事."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt="   ")

        assert "不能是空白串" in str(excinfo.value)

    def test_a_non_string_prompt_is_rejected(self) -> None:
        """模板必须是字符串（``None`` 之类的形状在构造期就挡住）."""
        with pytest.raises(ContextError) as excinfo:
            make_pipeline(prompt=123)  # type: ignore[arg-type]

        assert "必须是字符串" in str(excinfo.value)

    def test_a_custom_template_with_both_placeholders_is_accepted(self) -> None:
        """两个占位符都在的模板可以替换掉默认模板（并照它的样子渲染）."""
        llm = RecordingLLM(["答案 [1]"])
        pipeline = make_pipeline(llm=llm, prompt="【资料】{context}【问题】{question}")

        answer = pipeline.answer(AXIS)

        content = prompt_of(llm)
        assert content.startswith("【资料】[1] ")
        assert content.endswith(AXIS)
        assert answer.llm_called is True


# --------------------------------------------------------------------------- #
# 逐次覆盖（overrides）
# --------------------------------------------------------------------------- #


class TestOverrides:
    """``answer(**overrides)`` 允许逐次改预算与温度，但不认识的键必须被拒."""

    def test_max_context_chars_override_is_applied(self) -> None:
        """这一次的预算可以比构造期的更紧（例如端点按请求体给值）."""
        answer = make_pipeline().answer(AXIS, max_context_chars=400)

        assert answer.context is not None
        assert answer.context.max_chars == 400

    def test_per_hit_chars_override_is_applied(self) -> None:
        """单条上限也能逐次覆盖（覆盖之后降级注记跟着出现）."""
        answer = make_pipeline().answer(AXIS, per_hit_chars=40)

        assert answer.context is not None
        assert answer.context.truncated_hits >= 1
        assert any("被单块上限 40 字截断" in note for note in answer.notes)

    def test_temperature_override_is_applied(self) -> None:
        """温度覆盖必须**真的**传给了模型（不是记在返回值里）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm, temperature=0.2).answer(AXIS, temperature=0.7)

        assert llm.calls[0]["temperature"] == 0.7

    def test_an_unknown_override_key_is_rejected(self) -> None:
        """不认识的键要报错并**列出合法取值**（静默忽略会让调用方以为它生效了）."""
        with pytest.raises(QueryError) as excinfo:
            make_pipeline().answer(AXIS, top_k=3)

        message = str(excinfo.value)
        assert "top_k" in message
        assert "可用的键只有" in message
        assert "max_context_chars" in message

    def test_a_misspelled_budget_key_is_rejected(self) -> None:
        """把 ``max_context_chars`` 拼少一个 s：预算其实没生效，答案只会"看起来变短了"."""
        with pytest.raises(QueryError) as excinfo:
            make_pipeline().answer(AXIS, max_context_char=100)

        assert "max_context_char" in str(excinfo.value)

    def test_the_misspelling_is_caught_before_any_llm_call(self) -> None:
        """覆盖参数的校验在**检索与调用之前**（错误不该花掉一次模型调用）."""
        llm = ExplodingLLM()
        pipeline = make_pipeline(llm=llm)

        with pytest.raises(QueryError):
            pipeline.answer(AXIS, top_k=3)

        assert llm.calls == 0

    def test_an_override_budget_of_zero_is_rejected(self) -> None:
        """逐次覆盖同样走那套校验：0 不是"不限"."""
        with pytest.raises(ContextError):
            make_pipeline().answer(AXIS, max_context_chars=0)

    def test_an_out_of_range_temperature_override_is_rejected(self) -> None:
        """温度越界同样在覆盖路径上被拒（同一个校验器两处共用）."""
        with pytest.raises(QueryError) as excinfo:
            make_pipeline().answer(AXIS, temperature=3.0)

        assert "超出 [0, 2]" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 批量问答
# --------------------------------------------------------------------------- #


class TestAnswerMany:
    """``answer_many`` 刻意逐条调用：每次的 ``llm_called`` 与消耗都是独立证据."""

    def test_returns_one_answer_per_question_in_order(self) -> None:
        """顺序与入参一致（否则报告里"第 2 个问题"就没法核对了）."""
        llm = MockLLM(responses=["第一 [1]", "第二 [1]"])
        answers = make_pipeline(llm=llm).answer_many([AXIS, TILT])

        assert [answer.answer for answer in answers] == ["第一 [1]", "第二 [1]"]
        assert all(answer.llm_called for answer in answers)
        assert [answer.question for answer in answers] == [AXIS, TILT]

    def test_every_question_gets_its_own_prompt(self) -> None:
        """两个问题 → 两条不同的提示词（逐条调用，不是批处理）."""
        llm = MockLLM(responses=["第一 [1]", "第二 [1]"])
        make_pipeline(llm=llm).answer_many([AXIS, TILT])

        first = llm.calls[0][0].content
        second = llm.calls[1][0].content

        assert len(llm.calls) == 2
        assert AXIS in first and TILT not in first
        assert TILT in second and AXIS not in second

    def test_an_empty_list_returns_an_empty_list(self) -> None:
        """空批次不报错（与 ``aggregate_results`` 对空批次的取向一致）."""
        assert make_pipeline().answer_many([]) == []

    def test_a_mixed_batch_only_calls_the_llm_for_the_hits(self) -> None:
        """一批里有一条检索为空：它不消耗模型，其余照常（脚本只够一次也能过）."""
        llm = MockLLM(responses=["只有一次 [1]"])
        questions = [AXIS, RetrievalQuery(text=AXIS, where={"strategy": "不存在的策略"})]

        answers = make_pipeline(llm=llm).answer_many(questions)

        assert len(llm.calls) == 1, "第二条问题走的是空结果分支，不调模型"
        assert answers[0].llm_called is True
        assert answers[1].llm_called is False
        assert answers[1].answer == FALLBACK_NO_CONTEXT


# --------------------------------------------------------------------------- #
# RagAnswer 的形状
# --------------------------------------------------------------------------- #


class TestRagAnswerShape:
    """``RagAnswer`` 是"一次问答的完整交代"，它自己也校验两条字段纪律."""

    def test_defaults_are_empty_but_carried(self) -> None:
        """只有问题与答案时，其余字段都是"没有发生"的默认值."""
        answer = RagAnswer(question="问题", answer="答案")

        assert answer.citations == ()
        assert answer.context is None
        assert answer.retrieval is None
        assert answer.llm_called is False
        assert answer.notes == ()

    def test_the_question_must_be_non_empty(self) -> None:
        """一份不知在回答什么问题的答案无法被复核."""
        for question in ("", "   "):
            with pytest.raises(QueryError):
                RagAnswer(question=question, answer="答案")

    def test_the_answer_must_be_a_string(self) -> None:
        """模型回复的形状在 ``RagPipeline`` 里已经被收敛成文本."""
        with pytest.raises(QueryError) as excinfo:
            RagAnswer(question="问题", answer=123)  # type: ignore[arg-type]

        assert "必须是字符串" in str(excinfo.value)

    def test_to_dict_reports_the_prompt_version(self) -> None:
        """报告要能按提示词版本分组（day071 的 RAG 评估会拿它做实验分组）."""
        payload = make_pipeline().answer(AXIS).to_dict()

        assert payload["prompt_version"] == RAG_ANSWER_PROMPT_VERSION
        assert payload["llm_called"] is True
        assert payload["context"] is not None
        assert len(payload["citations"]) == 5
        assert payload["retrieval"] is not None

    def test_to_dict_can_hide_the_context(self) -> None:
        """只要能排序与统计时显式传 ``False``（与 ``RetrievalResult`` 的取向相反）."""
        answer = make_pipeline().answer(AXIS)

        assert answer.to_dict(include_context=False)["context"] is None
        assert answer.to_dict()["context"]["text"] == answer.context.text  # type: ignore[union-attr]

    def test_to_dict_includes_the_retrieval_record(self) -> None:
        """检索的账（含 ``empty_reason`` 与索引状态）跟着回答一起走."""
        payload = make_pipeline(retriever=drift_retriever()).answer(AXIS).to_dict()

        assert payload["retrieval"]["empty_reason"] == EMPTY_REASON_NONE
        assert payload["retrieval"]["index"]["has_drift"] is True
        assert any("索引漂移" in note for note in payload["notes"])

    def test_summary_line_says_whether_the_llm_was_called(self) -> None:
        """一行摘要里"调没调模型"必须看得见（护栏的唯一可见证据）."""
        called = make_pipeline().answer(AXIS)
        skipped = make_pipeline(store=empty_store(), llm=ExplodingLLM()).answer(AXIS)

        assert "LLM 已调用" in called.summary_line()
        assert "未调用（检索为空）" in skipped.summary_line()


# --------------------------------------------------------------------------- #
# 构造期校验与只读视图
# --------------------------------------------------------------------------- #


class TestPipelineConstruction:
    """构造期的两类校验：注入对象的形状，以及三个数字参数的取值."""

    def test_the_retriever_must_be_a_retriever(self) -> None:
        """前半段必须是检索器：它负责"取哪几条"."""
        with pytest.raises(QueryError) as excinfo:
            RagPipeline(sample_store(), MockLLM())  # type: ignore[arg-type]

        assert "必须是 Retriever" in str(excinfo.value)

    def test_the_llm_must_be_a_base_llm(self) -> None:
        """任何实现 ``BaseLLM.chat`` 的对象都可以注入（假实现是刻意的用法）."""
        with pytest.raises(QueryError) as excinfo:
            RagPipeline(sample_retriever(), "不是模型")  # type: ignore[arg-type]

        assert "必须是 BaseLLM" in str(excinfo.value)

    def test_budgets_must_be_ints(self) -> None:
        """两个预算都必须显式是整数（``True`` 是 bool，同样被拦）."""
        for value in ("800", True):
            with pytest.raises(ContextError) as excinfo:
                make_pipeline(max_context_chars=value)

            assert "必须是整数" in str(excinfo.value)

    def test_budgets_must_be_positive(self) -> None:
        """0 会让一次正常的检索退化成一个"没有资料的提问"."""
        with pytest.raises(ContextError):
            make_pipeline(max_context_chars=0)
        with pytest.raises(ContextError):
            make_pipeline(per_hit_chars=-5)

    def test_the_fallback_answer_must_be_non_empty(self) -> None:
        """返回一句空话与"不回答"是两件事——前者让用户以为系统坏了."""
        with pytest.raises(QueryError) as excinfo:
            make_pipeline(fallback_answer="   ")

        assert "fallback_answer 必须是非空字符串" in str(excinfo.value)

    def test_temperature_must_be_a_number(self) -> None:
        """温度不是数字时当场报（不要等到调用模型）."""
        with pytest.raises(QueryError) as excinfo:
            make_pipeline(temperature="0.5")  # type: ignore[arg-type]

        assert "必须是数字" in str(excinfo.value)

    def test_temperature_must_be_finite(self) -> None:
        """``nan`` 与无穷都不是温度（比较恒为 False，后果不可复现）."""
        for value in (float("nan"), float("inf")):
            with pytest.raises(QueryError) as excinfo:
                make_pipeline(temperature=value)

            assert "有限数" in str(excinfo.value)

    def test_temperature_range_is_enforced(self) -> None:
        """RAG 问答是有依据的复述：温度越高，模型越会改写片段里的事实."""
        for value in (-0.1, 2.5):
            with pytest.raises(QueryError) as excinfo:
                make_pipeline(temperature=value)

            assert "超出 [0, 2]" in str(excinfo.value)

    def test_an_int_temperature_is_accepted(self) -> None:
        """整数温度是可以的（会被收敛成 float）."""
        assert make_pipeline(temperature=2).describe()["temperature"] == 2.0

    def test_describe_reports_the_configuration(self) -> None:
        """``describe`` 是端点与演示脚本的统一入口，键名要与规格一致."""
        described = make_pipeline().describe()

        assert set(described) == {
            "prompt_version",
            "max_context_chars",
            "per_hit_chars",
            "temperature",
            "retriever",
            # day069 新增的那一组：生成器的配置（含受管版本清单与模板长度）。
            "generator",
        }
        assert described["generator"]["prompt_versions"] == ["v1", "v2"]
        assert described["prompt_version"] == RAG_ANSWER_PROMPT_VERSION
        assert described["retriever"]["name"] == "default"
        assert described["retriever"]["top_k"] == 5

    def test_defaults_come_from_settings(self) -> None:
        """不传预算 → 读 settings（项目默认预算只有一处定义）."""
        described = make_pipeline().describe()

        assert described["max_context_chars"] == settings.retrieval_max_context_chars
        assert described["per_hit_chars"] == settings.retrieval_per_hit_chars

    def test_properties_expose_the_prompt_version_and_the_llm(self) -> None:
        """端点要在"未配置 LLM"时提前拦住，因此注入的对象必须能读回来."""
        llm = MockLLM(responses=["答案 [1]"])
        pipeline = make_pipeline(llm=llm)

        assert pipeline.prompt_version == RAG_ANSWER_PROMPT_VERSION
        assert pipeline.llm is llm
