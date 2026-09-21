"""RAG 生成链路：检索 → 打包 → 生成 → 引用（M6-D5）.

本模块是 day066 的后半段：它把 ``retriever`` 与 ``context`` 串成一条**能回答
问题**的链路，并在链路上钉死一条硬护栏：

```text
检索为空 → 一次 LLM 都不调，直接返回 fallback_answer（llm_called=False）
```

## 为什么这条护栏是本课最重要的东西

```text
检索为空 + 调用 LLM   模型手里没有片段，却仍然会写出一段通顺的答案
                      → 它们全来自预训练记忆，而读的人看不出区别
检索为空 + 不调用     答案明确写着"知识库中没有检索到相关内容"，并给出三条出路
```

第一行的产物**看起来最像成功**：语法正确、语气确定、引用格式也像模像样。
它是 RAG 系统最贵的一种失败——因为没有异常、没有告警，只有"用户问了一个
知识库里没有的问题，却拿到了一个编出来的答案"。因此在 ``answer`` 里，
空结果的判断在**渲染提示词之前**：护栏写在调用的上游，而不是调用的下游。

## 提示词为什么也要版本化

与 ``agent.prompts`` 的约定一致（day026）：历史版本永不删 + 注释写变更原因。
理由是**答案质量的变化必须能被归因**：同一份索引、同一批问题，
换一版提示词之后答案变好了还是变差了，只有把版本号固定下来才回答得了
（day071 的 RAG 评估会拿它做实验分组）。模板占位符固定为
``{context}`` 与 ``{question}``，缺任何一个都在**构造期**报 ``ContextError``——
缺占位符的模板渲染出来是一段没有资料的提示词，而它不会报错，
只会让模型自由发挥（正是护栏要拦的那件事）。

## 降级为什么要进 ``notes``

三条降级都不阻断链路，因此它们唯一的痕迹就是 ``notes``：

```text
单条被截断      这次回答可能"少半句"（truncated_hits > 0）
整包丢尾        这次回答可能"少几条依据"（dropped_hits 非空）
索引有漂移      这次的依据来自一份可能过期的清单（可能少召回）
```

前两条的处置是调预算，第三条的处置是重建清单——三种不同的动作，
所以在 ``notes`` 里是三条各自可读的话，而不是一个布尔标志。
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.retrieval.context import Citation, PackedContext, pack_context
from smart_research_agent.retrieval.errors import ContextError, QueryError
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import (
    EMPTY_REASON_DESCRIPTIONS,
    RetrievalQuery,
    RetrievalResult,
)

#: 提示词模板必须有的两个占位符（构造期校验的就是它们）。
REQUIRED_PROMPT_FIELDS: tuple[str, ...] = ("context", "question")

#: 允许在 ``answer(**overrides)`` 里覆盖的三个参数。
OVERRIDE_KEYS: tuple[str, ...] = ("max_context_chars", "per_hit_chars", "temperature")

# v1（day066）：RAG 答案提示词的首版。四条约束各自挡一种真实失败：
# 1) "只依据片段"       挡"模型拿预训练知识把答案补圆"（引用看着正常，内容与库无关）
# 2) "片段不足就说没有" 挡"编一个通顺的答案"（最贵的一种失败：它看起来最像成功）
# 3) "引用写 [n]"       挡"答案对但溯源不上"（day069 的引用溯源要拿它当锚点）
# 4) "不许编造"         挡"把两个相似概念合并成一个"（错得最像常识的那一类）
# 变更时新增 _V2 并把 RAG_ANSWER_PROMPT 指过去，**本常量与历史版本永不删**。
RAG_ANSWER_PROMPT_V1 = """你是知识库问答助手，请严格依据下面提供的资料片段回答问题。

资料片段（每条的编号写在方括号里）：
{context}

问题：{question}

回答要求：
1. 只使用上面资料片段中出现的信息；不要引入片段之外的知识，不要靠推测补全。
2. 如果片段不足以回答问题，就直接说"资料中没有相关内容"，并指出缺什么，不要编造。
3. 用到的每条资料都要标出它的编号，写法是 [1]、[2]；编号必须来自上面的片段。
4. 用中文回答，先给结论再给依据，不要整段复述原文。
"""

#: 当前生效的 RAG 答案提示词及其版本号.
RAG_ANSWER_PROMPT = RAG_ANSWER_PROMPT_V1
RAG_ANSWER_PROMPT_VERSION = "v1"

#: 检索为空时的兜底答复。它必须包含**三条出路**而不是一句"没有找到"：
#: 用户看到的应该是一个下一步动作，而不是一次失败宣告。
FALLBACK_NO_CONTEXT = (
    "知识库中没有检索到与该问题相关的内容，因此不作回答。"
    "建议：换一种说法、放宽过滤条件（时间范围 / 元数据），"
    "或确认索引版本是否包含这批资料。"
)


@dataclass(frozen=True)
class RagAnswer:
    """一次 RAG 问答的完整交代：问题 + 答案 + 依据 + 这次到底调没调模型（M6-D5）.

    ```text
    question    规范化之后的问题（与 RetrievalQuery.text 同一份）
    answer      模型给的答案，或 fallback_answer
    citations   答案里那些 [n] 的对照表（只有进了上下文的命中）
    context     打进提示词的那段文本及其账（未调 LLM 时为 None）
    retrieval   这次检索的完整交代（含 empty_reason 与索引状态）
    llm_called  这一次到底调没调模型——护栏是否生效的唯一证据
    notes       降级注记：截断 / 丢尾 / 索引漂移 / 空回复
    ```

    ``llm_called`` 单独成为一个字段而不是"看 answer 等不等于 fallback"：
    **判定不能依赖字符串比较**——兜底答复将来改一个字，那种判定就静默失效。
    它是护栏的**证据**：``llm_called=False`` 且 ``citations=()`` 表示
    "模型这次没有机会自由发挥"。
    """

    question: str
    answer: str
    citations: tuple[Citation, ...] = ()
    context: PackedContext | None = None
    retrieval: RetrievalResult | None = None
    llm_called: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise QueryError(
                f"RagAnswer.question 必须是非空字符串，收到 {self.question!r}："
                "一份不知在回答什么问题的答案无法被复核。"
            )
        if not isinstance(self.answer, str):
            raise QueryError(
                f"RagAnswer.answer 必须是字符串，收到 {type(self.answer).__name__}："
                "模型回复的形状在 RagPipeline 里已经被收敛成文本。"
            )

    def to_dict(self, *, include_context: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_context=True`` 是默认值（与 ``RetrievalResult.to_dict`` 相反）：
        这里的消费者主要是**排查一次回答**（"模型到底看到了什么"），
        而那正是上下文本身。只要排序与统计时请显式传 ``False``。
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "llm_called": self.llm_called,
            "prompt_version": RAG_ANSWER_PROMPT_VERSION,
            "citations": [citation.to_dict() for citation in self.citations],
            "context": (
                self.context.to_dict()
                if include_context and self.context is not None
                else None
            ),
            "retrieval": self.retrieval.to_dict() if self.retrieval is not None else None,
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        question = self.question.replace("\n", " ")[:24]
        answer = self.answer.replace("\n", " ")[:36]
        called = "已调用" if self.llm_called else "未调用（检索为空）"
        return f"问：{question} | 答：{answer} | 引用 {len(self.citations)} 条 | LLM {called}"


class RagPipeline:
    """检索 → 打包 → 生成 → 引用（构造期定预算与模板，运行期只做四步）.

    构造参数分两组，**每组各自决定一件可以被单独讨论的事**：

    ```text
    预算类    max_context_chars / per_hit_chars      "给模型多少空间"（传给 pack_context）
    生成类    prompt / fallback_answer / temperature "怎么问、问不到时说什么"
    ```

    ``None`` 在这里有明确含义：**"没指定，去读 settings"**
    （``retrieval_max_context_chars`` / ``retrieval_per_hit_chars``）。
    于是"项目默认预算"只有一处定义，而端点与演示脚本不必各抄一遍。

    检索器与 LLM 都是**注入**的：这一层不认识任何具体后端，
    因此测试可以用一个"一旦被调用就抛异常"的 LLM 来证明护栏生效。
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        *,
        prompt: str = RAG_ANSWER_PROMPT,
        max_context_chars: int | None = None,
        per_hit_chars: int | None = None,
        fallback_answer: str = FALLBACK_NO_CONTEXT,
        temperature: float = 0.0,
    ) -> None:
        if not isinstance(retriever, Retriever):
            raise QueryError(
                f"retriever 必须是 Retriever，收到 {type(retriever).__name__}。"
                "RAG 链路的前半段是检索器：它负责'取哪几条'，本类负责'怎么问'。"
            )
        if not isinstance(llm, BaseLLM):
            raise QueryError(
                f"llm 必须是 BaseLLM，收到 {type(llm).__name__}。"
                "任何实现 BaseLLM.chat 的对象都可以注入（测试用假实现是刻意的用法）。"
            )
        self._retriever = retriever
        self._llm = llm
        self._prompt = _validate_prompt(prompt)
        self._max_context_chars = _resolve_budget(
            "max_context_chars", max_context_chars, settings.retrieval_max_context_chars
        )
        self._per_hit_chars = _resolve_budget(
            "per_hit_chars", per_hit_chars, settings.retrieval_per_hit_chars
        )
        self._fallback_answer = _validate_text("fallback_answer", fallback_answer)
        self._temperature = _validate_temperature(temperature)

    # ------------------------------------------------------------------ 只读视图

    @property
    def prompt_version(self) -> str:
        """当前提示词版本（报告与评估按它分组）."""
        return RAG_ANSWER_PROMPT_VERSION

    @property
    def llm(self) -> BaseLLM:
        """注入的模型（端点在"未配置 LLM"时要在构造前就拦住）."""
        return self._llm

    def describe(self) -> dict[str, Any]:
        """这一层的配置 + 检索器的现状（端点直接返回它）."""
        return {
            "prompt_version": self.prompt_version,
            "max_context_chars": self._max_context_chars,
            "per_hit_chars": self._per_hit_chars,
            "temperature": self._temperature,
            "retriever": self._retriever.describe(),
        }

    # ------------------------------------------------------------------ 问答入口

    def answer(self, question: str | RetrievalQuery, **overrides: Any) -> RagAnswer:
        """回答一个问题（四步：检索 → 判空 → 打包 → 生成）.

        第 2 步（判空）**在打包与渲染之前**，这是护栏在代码里的位置：
        检索为空时 ``pack_context`` 与 ``llm.chat`` 都不会被调用。

        ``overrides`` 允许逐次覆盖 ``max_context_chars`` / ``per_hit_chars`` /
        ``temperature``（例如端点按请求体里的预算回答）。不认识的键会报
        ``QueryError`` 并列出合法取值——**静默忽略一个覆盖参数**会让调用方
        以为它的预算生效了。
        """
        max_chars, per_hit, temperature = self._resolve_overrides(overrides)
        retrieval = self._retriever.retrieve(question)
        # 用**已规范化**的问题文本：一次检索里的 question 与送进提示词的问题
        # 必须是同一份（与 RetrievalResult.query 存已解析默认值是同一个理由）。
        question_text = retrieval.query.text

        if retrieval.is_empty:
            return RagAnswer(
                question=question_text,
                answer=self._fallback_answer,
                citations=(),
                context=None,
                retrieval=retrieval,
                llm_called=False,
                notes=tuple(_empty_notes(retrieval)),
            )

        context = pack_context(
            retrieval.hits,
            max_chars=max_chars,
            per_hit_chars=per_hit,
        )
        reply = self._llm.chat(
            [Message(role="user", content=self._render(context, question_text))],
            temperature=temperature,
        )
        notes = _degrade_notes(context, retrieval, max_chars=max_chars, per_hit=per_hit)
        if not reply.strip():
            notes.append(
                "模型返回了空回复：这一次的 answer 是空串，而引用与上下文都在——"
                "请检查提供方（超时、内容过滤、max_tokens 太小都会给出空串），"
                "不要把它当成'知识库里没有相关内容'（那是 empty_reason 的事）。"
            )
        return RagAnswer(
            question=question_text,
            answer=reply,
            citations=context.citations,
            context=context,
            retrieval=retrieval,
            llm_called=True,
            notes=tuple(notes),
        )

    def answer_many(self, questions: Sequence[str | RetrievalQuery]) -> list[RagAnswer]:
        """批量问答（逐条调用 ``answer``，顺序与入参一致）.

        与 ``Retriever.retrieve_many`` 同样是**刻意的逐条**：每次问答的
        ``llm_called`` / ``notes`` / 上下文占用都是独立的证据，
        而"这一批总共花了多少"与"其中一次为什么没调模型"是两个问题。
        """
        return [self.answer(item) for item in questions]

    # ------------------------------------------------------------------ 内部

    def _render(self, context: PackedContext, question: str) -> str:
        """渲染提示词（占位符在构造期已经校验过，因此这里不会缺参数）."""
        return self._prompt.format(context=context.text, question=question)

    def _resolve_overrides(self, overrides: dict[str, Any]) -> tuple[int, int, float]:
        """解析逐次覆盖，并**明确拒绝**不认识的键（见 ``answer``）."""
        unknown = sorted(key for key in overrides if key not in OVERRIDE_KEYS)
        if unknown:
            raise QueryError(
                f"不认识的覆盖参数：{unknown}。可用的键只有 {list(OVERRIDE_KEYS)}。"
                "静默忽略一个覆盖参数会让调用方以为它生效了——"
                "例如把 max_context_chars 拼成 max_context_char 之后，"
                "预算仍然是默认值，而答案看起来只是'变短了'。"
            )
        max_chars = overrides.get("max_context_chars", self._max_context_chars)
        per_hit = overrides.get("per_hit_chars", self._per_hit_chars)
        temperature = overrides.get("temperature", self._temperature)
        return (
            _resolve_budget("max_context_chars", max_chars, self._max_context_chars),
            _resolve_budget("per_hit_chars", per_hit, self._per_hit_chars),
            _validate_temperature(temperature),
        )


# --------------------------------------------------------------------------- #
# 提示词与注记
# --------------------------------------------------------------------------- #


def _validate_prompt(prompt: str) -> str:
    """校验模板含 ``{context}`` 与 ``{question}``（缺任何一个 → ``ContextError``）.

    用 ``string.Formatter().parse`` 而不是"字符串里搜 ``"{context}"``"：
    后者会把 ``{{context}}``（转义之后的字面量）也当成占位符，
    而那种模板渲染出来是一段谁也读不懂的文本。前者读的是**真正的语法**。

    为什么必须在构造期报错：缺占位符的模板``format`` 不会失败
    （多余的参数被忽略），于是模型收到的是一段**没有资料的提示词**——
    它照着模板回答，看起来一切正常，实际上一次检索都没用上。
    这正是本模块最初那条护栏要拦的事，所以它在构造期就被拦掉。
    """
    if not isinstance(prompt, str):
        raise ContextError(f"prompt 必须是字符串，收到 {type(prompt).__name__}")
    if not prompt.strip():
        raise ContextError(
            "prompt 不能是空白串：空模板渲染出来的提示词等于'让模型自由发挥'，"
            "而这条链路存在的意义恰恰是**不让它自由发挥**。"
        )
    fields = _template_fields(prompt)
    missing = [name for name in REQUIRED_PROMPT_FIELDS if name not in fields]
    if missing:
        expected = " / ".join("{" + name + "}" for name in REQUIRED_PROMPT_FIELDS)
        raise ContextError(
            f"提示词模板缺少占位符 {missing}：模板里必须同时出现 {expected}。"
            f"当前模板用到的占位符是 {sorted(fields) or '（一个都没有）'}。"
            "缺 {context} 会让模型看不到任何资料，缺 {question} 会让它不知道"
            "在回答什么——两者都不会让 format 失败，只会让答案悄悄变差。"
        )
    return prompt


def _template_fields(prompt: str) -> set[str]:
    """模板里的占位符名字集合（语法错误 → ``ContextError``）.

    ``Formatter.parse`` 只做语法解析，不要求每个占位符都能被填上；
    因此它能同时回答"'有哪些占位符'与'这个模板写得合法吗'"。
    """
    try:
        return {
            field_name
            for _literal, field_name, _spec, _conversion in string.Formatter().parse(prompt)
            if field_name
        }
    except ValueError as exc:
        raise ContextError(
            f"提示词模板不是合法的格式串：{exc}。"
            "单独的 { 或 } 必须写成 {{ 与 }}——否则 format 会在运行期抛异常，"
            "而那时检索已经跑完、预算已经花掉了。"
        ) from exc


def _empty_notes(retrieval: RetrievalResult) -> list[str]:
    """空结果时记的三条注记（**为什么没调模型**必须被写出来）.

    第一句给原因，第二句给护栏本身，第三句给出路——与
    ``FALLBACK_NO_CONTEXT`` 的三条出路是同一条纪律：
    "没有找到"必须跟一个下一步动作，否则它只是一次失败宣告。
    """
    reason = EMPTY_REASON_DESCRIPTIONS.get(retrieval.empty_reason, "未知原因")
    return [
        f"检索为空（empty_reason={retrieval.empty_reason}）：{reason}",
        "按硬护栏没有调用 LLM：检索为空时不许让模型自由发挥——"
        "answer 直接取 fallback_answer，llm_called=False、citations=()",
        "建议：换一种说法、放宽过滤条件（时间范围 / 元数据），"
        "或确认索引版本是否包含这批资料",
    ]


def _degrade_notes(
    context: PackedContext,
    retrieval: RetrievalResult,
    *,
    max_chars: int,
    per_hit: int,
) -> list[str]:
    """三类降级各记一条（截断 / 丢尾 / 索引漂移，见模块 docstring）."""
    notes: list[str] = []
    if context.truncated_hits:
        notes.append(
            f"有 {context.truncated_hits} 条命中被单块上限 {per_hit} 字截断："
            "这些片段是半截的，答案可能缺一句结论——要完整片段请调大 "
            "settings.retrieval_per_hit_chars。"
        )
    if context.dropped_hits:
        notes.append(
            f"预算 {max_chars} 字放不下，从尾部丢掉了 {len(context.dropped_hits)} 条命中"
            f"（{', '.join(hit.record_id for hit in context.dropped_hits[:3])}）："
            "被丢的是名次最靠后的那几条，答案可能少一部分依据——"
            "要更多依据请调大 settings.retrieval_max_context_chars。"
        )
    if retrieval.index_state.has_drift:
        notes.append(
            f"索引漂移 {len(retrieval.index_state.drift)} 项："
            "本次检索基于可能过期的清单，结果可能少召回"
            f"（{retrieval.index_state.drift[0]}）——"
            "请重建清单，或确认这次少召回是否可接受。"
        )
    return notes


# --------------------------------------------------------------------------- #
# 参数解析（构造期与逐次覆盖共用同一套校验）
# --------------------------------------------------------------------------- #


def _resolve_budget(label: str, value: int | None, default: int) -> int:
    """解析一个预算参数（``None`` → 默认值；必须是 >= 1 的整数）."""
    resolved = default if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        raise ContextError(
            f"{label} 必须是整数，收到 {type(resolved).__name__}（值为 {resolved!r}）"
        )
    if resolved < 1:
        raise ContextError(
            f"{label}={resolved}：预算为 0 意味着没有任何片段能进提示词，"
            "而那会让一次正常的检索退化成一个'没有资料的提问'。"
            "要'不限'请给一个足够大的正数，不要给 0。"
        )
    return resolved


def _validate_text(label: str, value: str) -> str:
    """``fallback_answer`` 一类的文本参数：必须是非空字符串."""
    if not isinstance(value, str) or not value.strip():
        raise QueryError(
            f"{label} 必须是非空字符串，收到 {value!r}："
            "检索为空时返回一句空话，与'不回答'是两件事——"
            "前者让用户以为系统坏了。"
        )
    return value


def _validate_temperature(value: float) -> float:
    """温度必须是 [0, 2] 内的有限数（0 是这一层的默认值）."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise QueryError(f"temperature 必须是数字，收到 {type(value).__name__}")
    resolved = float(value)
    if resolved != resolved or resolved in (float("inf"), float("-inf")):
        raise QueryError(f"temperature 必须是有限数，收到 {value!r}")
    if not 0.0 <= resolved <= 2.0:
        raise QueryError(
            f"temperature={resolved} 超出 [0, 2]：RAG 问答是**有依据的复述**，"
            "过高的温度会让模型改写片段里的事实（默认 0.0 就是为了让它照抄依据）。"
        )
    return resolved


__all__ = [
    "FALLBACK_NO_CONTEXT",
    "OVERRIDE_KEYS",
    "RAG_ANSWER_PROMPT",
    "RAG_ANSWER_PROMPT_V1",
    "RAG_ANSWER_PROMPT_VERSION",
    "REQUIRED_PROMPT_FIELDS",
    "RagAnswer",
    "RagPipeline",
]
