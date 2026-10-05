"""RAG 生成器：把"模型给的这段话"折成一份**可核对的账**（M6-D8）.

day066 的 ``pipeline`` 把"检索 → 打包 → 生成"串成了一条链路，并钉死了本课最
重要的一条护栏：检索为空时一次 LLM 都不调。但它回答不了第五个问题——而这个问题
恰恰是"RAG"与"搜索引擎 + 模型"的分界线：

```text
答案里的 [1] 指谁？               提示词里的位置已由 context.Citation 给出
[9] 呢？                          **指不到任何片段**，而它在报告里看不见
给了 5 条片段，模型引用了几条？      覆盖率 = valid / citations
模型自己说"资料里没有"了吗？         这句话以前**无法被检测**
```

## 为什么"引用核对"必须单独成层

```text
调用成功、答案通顺、引用格式像模像样   ← 这三件事都不能说明答案被核对过
引用编号全部落在那次提示词里           ← 只有这一条能说明
```

第一行是最容易骗过人的证据：模型可以在一段与库无关的文字后面规规矩矩地写上
``[1]``。因此本模块不把"答案"当成一件成品直接交出去，而是把它与**那次提示词
里的编号表**放在一起对账，并把结论折成三个数字：有效引用、幻觉引用、未被引用
的片段。核对的口径必须写清，否则它会被误读成一次事实核查：

```text
它做的事    答案里的 [n] 与那次提示词里的编号一一对账（**编号层面**）
它不做的事  判断这句话是否忠实于片段（**语义层面**，见 types 的边界声明）
```

``grounded`` 也一样，它的定义写在这里而不是靠字符串比较：

```text
grounded = valid 非空 且 invalid 为空
```

"valid 非空"挡的是"答案里一个引用都没有"（一次没有依据的总结），"invalid 为空"
挡的是"引用了不存在的编号"（幻觉引用）。两者都不成立时，这份答案**有可核对
的依据**——注意它仍然可能与原意相反，语义判断不在这一层（所以它叫 grounded
而不是 correct）。

## 提示词的 v1 → v2：四段式与三处新增

与 ``agent.prompts`` 的约定一致（day026）：历史版本永不删、变更原因写进注释、
当前版本用 ``CURRENT_PROMPT`` 指过去。v2（day069）改了三件事，每件都对应一个
能在报告里看见的字段：

```text
四段式          资料片段 / 问题 / 约束 / 输出格式 各自有小节标题
                → v1 把资料与要求混在一段里，模型只能靠位置猜
固定拒答句式    资料不足时必须以三个句式之一开头
                → "模型自己说没有"从此可检测（DECLINE_MARKERS）
引用格式约束    每条结论后至少一个 [n]，编号必须来自片段
                → 覆盖率与幻觉引用这两笔账才有意义
```

``RAG_PROMPT_V2`` 里那六条约束**逐条**注明了它挡哪一种真实失败（照 v1 的注释
写法）：一条挡不住的约束不该占提示词的预算。

## 六种回退：一次生成可以怎样"不算数"

"回退"不是"报错"，而是**这次答案不能按正常路径解读**。六种情形各自对应一个
处置动作，因此它们是六条话而不是一个布尔标志（与 ``EMPTY_REASONS`` 同一条
纪律）。判定顺序即优先级，写死在 ``FALLBACK_REASONS`` 里：

```text
no_context          检索 / 打包为空 → 一次 LLM 都不调（最靠上游，先判）
llm_error           调用抛异常（超时 / 鉴权 / 网络 / 额度）→ 答案取兜底
empty_reply         模型返回空串 → 它不是"资料里没有"，两件事的处置不同
model_declined      模型自己说"资料中没有相关内容"（命中固定句式）
unusable_citations  require_citation=True 而一条合法引用都没有
（NONE）            以上都没命中：答案来自模型，且核对出了有效引用
```

## 拒答为什么用固定句式而不是正则

``_declined`` 只做一件事：把答案归一化（去空白、折叠标点）之后，看它是否包含
``DECLINE_MARKERS`` 里的某一个字符串。不做正则、不做模糊匹配，理由是三条：

```text
可枚举    v2 的约束里写死了这三个句式，检测清单与要求清单是同一份
可测试    每个句式都能被一条用例钉住（"资料未提及"与"资料中没有相关内容"）
可解释    报告里能写出"命中了哪一句"，而不是一个没有依据的置信度
```

正则与编辑距离都会引入一个**阈值**：前者会把"没有相关内容"这半句也算命中
（误判一次拒答等于把一次正常回答丢掉），后者需要一个没有依据的相似度门限，
而同一个答案在两台机器上还可能得到不同结论。

## 与既有包的接缝

- **上游**：``context.PackedContext``（提示词文本 + 编号表 + 预算账）是本模块
  唯一的输入形状——提示词从它的 ``text`` 渲染，核对从它的 ``citations`` 对账；
- **脚下**：``pipeline`` 四步里"生成"这一步换成了本模块（``RAGGenerator``）；
  空检索的护栏**仍留在 pipeline**（它在打包与渲染之前，比本模块更上游）；
  ``retrieval_prompt_version`` / ``retrieval_answer_temperature`` /
  ``retrieval_answer_max_tokens`` / ``retrieval_require_citation`` 是它的四个旋钮；
- **下游**：day071 的 RAG 评估要拿 ``Generation.check`` 的三组编号与
  ``fallback_reason`` 做实验分组（"换一版提示词之后，可核对率变了吗"）；
- **端点**：``api.routes`` 的 ``/retrieval/answer``（提示词**只能在装配期**注入，
  请求体里没有提示词文本——理由见下）；
- **手册与脚本**：``docs/retrieval.md`` 与 ``scripts/retrieval_demo.py``。

## 一条端点纪律（由本模块的形状决定）

``generate(**overrides)`` 允许逐次覆盖 ``temperature`` / ``max_tokens`` /
``prompt_version`` 三个键，但**不允许**在请求体里传提示词文本：提示词是"这一次
模型到底被怎么问的"的唯一证据，一旦它能被请求方改写，``prompt_version`` 就不再
指着一份确定的模板，day071 的分组实验也随之失去可复现性。要换模板请在**装配期**
注入（``prompt=...``，版本号记为 ``custom``，它不在受管清单里、但是合法取值）。
"""

from __future__ import annotations

import re
import string
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.retrieval.context import Citation, PackedContext
from smart_research_agent.retrieval.errors import ContextError, GenerationError
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 提示词：多版本 + 封闭清单
# --------------------------------------------------------------------------- #

#: 提示词模板必须有的两个占位符（构造期校验的就是它们）。
#: 这份清单在 day066 由 ``pipeline`` 定义，day069 搬到这里：
#: **模板住在哪，清单就住在哪**——两者分家之后，"缺占位符"这件事就会在
#: 两个地方各有一套判据（pipeline 继续转发旧名，见它的模块 docstring）。
REQUIRED_PROMPT_FIELDS: tuple[str, ...] = ("context", "question")

#: 受管模板的版本号（顺序 = 报告里的顺序）。
PROMPT_VERSION_V1 = "v1"
PROMPT_VERSION_V2 = "v2"

#: 全部受管版本。**它是封闭清单**：``prompt_version`` 落不到里面就当场报
#: ``GenerationError`` 并列出合法取值——版本号是报告与评估的分组键，
#: "没见过就当默认"会让一次换模板的实验悄悄落进旧分组。
PROMPT_VERSIONS: tuple[str, ...] = (PROMPT_VERSION_V1, PROMPT_VERSION_V2)

#: 缺省的提示词版本（``settings.retrieval_prompt_version`` 的默认值就是它）。
#: 它与 ``CURRENT_PROMPT_VERSION`` 今天相同，但**服务两个问题**：
#: 前者回答"没指定时用哪一版"，后者回答"现在生效的模板是哪一份"。
#: 换版本时只改 ``CURRENT_*``；改缺省值是一次显式决定（它会让所有没传
#: ``prompt_version`` 的调用方一起换版本）。
DEFAULT_PROMPT_VERSION: str = PROMPT_VERSION_V2

#: 注入自定义模板时的版本号。它**不在** ``PROMPT_VERSIONS`` 里（那不是一版
#: 受管模板），但它是合法取值：``_validate_prompt_version`` 同时认这两份清单，
#: 因此"清单外的版本号一律报错"这条纪律没有被这道特例削弱——
#: 特例只有一个名字，且它精确地表示"这一份提示词来自调用方"。
CUSTOM_PROMPT_VERSION: str = "custom"

# v1（day066）：RAG 答案提示词的首版。四条约束各自挡一种真实失败：
# 1) "只依据片段"       挡"模型拿预训练知识把答案补圆"（引用看着正常，内容与库无关）
# 2) "片段不足就说没有" 挡"编一个通顺的答案"（最贵的一种失败：它看起来最像成功）
# 3) "引用写 [n]"       挡"答案对但溯源不上"（day069 的引用溯源要拿它当锚点）
# 4) "不许编造"         挡"把两个相似概念合并成一个"（错得最像常识的那一类）
# 变更时新增 _V2 并把 CURRENT_PROMPT 指过去，**本常量与历史版本永不删**。
RAG_PROMPT_V1 = """你是知识库问答助手，请严格依据下面提供的资料片段回答问题。

资料片段（每条的编号写在方括号里）：
{context}

问题：{question}

回答要求：
1. 只使用上面资料片段中出现的信息；不要引入片段之外的知识，不要靠推测补全。
2. 如果片段不足以回答问题，就直接说"资料中没有相关内容"，并指出缺什么，不要编造。
3. 用到的每条资料都要标出它的编号，写法是 [1]、[2]；编号必须来自上面的片段。
4. 用中文回答，先给结论再给依据，不要整段复述原文。
"""

# v2（day069）：按"资料片段 / 问题 / 约束 / 输出格式"四段式重写。
# 变更原因（照 agent.prompts.REACT_SYSTEM_PROMPT_V2 的写法）：
#   1. v1 把"资料 / 问题 / 要求"混在一段里，模型对"哪一段是资料、哪一段是要求"
#      只能靠位置猜；四段式让"喂进去的东西"与"要它做的事"各自有标题；
#   2. 约束从 4 条扩到 6 条，每条都注明它挡哪一种真实失败（见下）；
#   3. 新增"资料不足时必须以固定句式开头"：v1 只说"就直接说没有相关内容"，
#      句式不固定，于是"模型自己说没有"这件事在代码里**无法被检测**
#      （检测清单见 DECLINE_MARKERS，两者的字面值必须一起改）；
#   4. 新增"引用格式"约束，把 [n] 与"每条结论后至少一个引用"绑在一起：
#      day069 的接地校验是按 [n] 对账的，模型漏引一条就会让那一条变成 unused。
# 六条约束各自挡的失败：
# 1) 只使用上面片段里的信息（不要引入片段之外的知识、不要靠推测补全）
#    挡"拿预训练记忆把答案补圆"：答案通顺、引用格式也对，内容却与库无关
# 2) 资料不足时以固定句式开头（三个句式见下）
#    挡"编一个通顺的答案"，同时让"模型自己说没有"成为**可检测**的事实
# 3) 用到的每条资料都要标出编号 [1]、[2]，且编号必须来自片段
#    挡"答案对但溯源不上"，也挡"引用一个没给它的编号"（幻觉引用）
# 4) 每条结论后至少一个引用
#    挡"整段没有引用的总结"：没有引用就没有可核对的东西（coverage 会是 0）
# 5) 片段里没有的事实一律不要编造（数字、日期、代码、专有名词）
#    挡"把两个相似概念合并成一个"（错得最像常识的那一类）
# 6) 用中文回答、先结论再依据、不要整段复述原文
#    挡"复述原文"：它把一次问答变成一次拷贝，还把预算挤给了无关句子
# 另外新增 "## 输出格式" 一节：v1 只管"内容该怎样"，没管"这份答案长什么样"，
# 于是"有没有引用、引用写在哪里"全靠模型自行发挥。
RAG_PROMPT_V2 = """你是知识库问答助手，请严格依据下面提供的资料片段回答问题。

## 资料片段

每条的编号写在方括号里，编号从 1 起、连续：

{context}

## 问题

{question}

## 约束

1. 只使用上面资料片段中出现的信息；不要引入片段之外的知识，不要靠推测补全。
2. 如果资料不足以回答问题，必须以固定句式开头，三选一：
   "资料中没有相关内容" / "资料未提及" / "无法依据资料回答"；
   随后指出缺什么，不要编造。
3. 用到的每条资料都要标出它的编号，写法是 [1]、[2]；编号必须来自上面的片段，
   不许出现片段里没有的编号。
4. 每条结论后面至少跟一个引用；只给结论不给出处的那一段等于不可核对。
5. 片段里没有的事实一律不要编造，包括数字、日期、代码与专有名词。
6. 用中文回答，先给结论再给依据，不要整段复述原文。

## 输出格式

1. 第一行：一句话结论，句末给出引用（例如 [1]）。
2. 随后每行一条依据，写法是 "依据 [n]：…"；没有依据的结论不要写。
3. 资料不足时：第一行写成上面选定句式的原话，随后列出缺什么。
"""

#: 版本 → 模板。受管模板只有 ``PROMPT_VERSIONS`` 里那几个；
#: 自定义模板不在这张表里（它的版本号是 ``CUSTOM_PROMPT_VERSION``）。
RAG_PROMPTS: dict[str, str] = {
    PROMPT_VERSION_V1: RAG_PROMPT_V1,
    PROMPT_VERSION_V2: RAG_PROMPT_V2,
}

#: 版本 → 变更原因（口吻与 ``agent.prompt_library.PromptTemplate.changelog``
#: 一致：一句话说清"改了什么、为什么改"，让下一版的人知道上一版的取舍在哪）。
PROMPT_CHANGELOG: dict[str, str] = {
    PROMPT_VERSION_V1: (
        "day066 首版：资料、问题与四条回答要求写在同一段里，"
        "顺序即语义（模型只能靠位置区分'喂进去的'与'要它做的'）"
    ),
    PROMPT_VERSION_V2: (
        "day069 重写：按资料片段/问题/约束/输出格式四段式拆分；"
        "约束扩到 6 条并逐条注明挡哪种真实失败；"
        "新增固定拒答句式（让'模型自己说没有'可检测）与引用格式约束"
    ),
}

#: 当前生效的提示词模板及其版本号（照 ``agent.prompts`` 的三条约定：
#: 历史版本永不删、变更写进 changelog、当前版本用这三个名字指过去）。
CURRENT_PROMPT = RAG_PROMPT_V2
CURRENT_PROMPT_VERSION = PROMPT_VERSION_V2

# --------------------------------------------------------------------------- #
# 回退族：封闭清单 + 固定优先级 + 一个原因
# --------------------------------------------------------------------------- #

#: 没走回退：答案来自模型，且核对出了一条以上有效引用。
#: 它是"没有失败"的哨兵值（与 ``types.EMPTY_REASON_NONE`` 同一套写法），
#: 让 ``fallback_reason`` 永远有值可比，读的人不需要先判空。
FALLBACK_REASON_NONE = ""

#: 检索 / 打包为空：一次 LLM 都不调，答案直接取兜底答复。
#: 它排在最前面，因为它是**最靠上游**的成因：没有片段，就谈不上调用与否。
FALLBACK_REASON_NO_CONTEXT = "no_context"

#: 调用抛异常：超时 / 鉴权 / 网络 / 额度不足都会走到这里。
#: 它**不是**"资料里没有相关内容"——这次确实调用了模型（llm_called=True）。
FALLBACK_REASON_LLM_ERROR = "llm_error"

#: 模型返回空串。它不是"知识库里没有相关内容"（那是 no_context 的事），
#: 处置动作也不同：前者去查提供方，后者去改检索。
FALLBACK_REASON_EMPTY_REPLY = "empty_reply"

#: 模型自己说"资料中没有相关内容"（命中 ``DECLINE_MARKERS`` 里的固定句式）。
#: v1 时代这句话是**不可检测**的：句式不固定，任何字符串比较都会静默失效。
FALLBACK_REASON_MODEL_DECLINED = "model_declined"

#: ``require_citation=True`` 而答案里一条合法引用都没有：
#: 这条答案没有任何可核对的依据，因此不进正常路径（答案取兜底答复）。
FALLBACK_REASON_UNUSABLE_CITATIONS = "unusable_citations"

#: 全部回退原因，**顺序 = 判定优先级**，且**不含** NONE。
#: 判定必须按这个顺序做：检索为空时"调用失败了没有"这个问题没有意义。
FALLBACK_REASONS: tuple[str, ...] = (
    FALLBACK_REASON_NO_CONTEXT,
    FALLBACK_REASON_LLM_ERROR,
    FALLBACK_REASON_EMPTY_REPLY,
    FALLBACK_REASON_MODEL_DECLINED,
    FALLBACK_REASON_UNUSABLE_CITATIONS,
)

#: 每个取值的人话解释 + 出路（``explain()`` 与端点直接引用它，
#: 避免同一句话写两遍）。NONE 也在里面，理由与 ``EMPTY_REASON_DESCRIPTIONS``
#: 完全相同：字典的键集合要覆盖**每一个**可能出现的取值。
FALLBACK_REASON_DESCRIPTIONS: dict[str, str] = {
    FALLBACK_REASON_NONE: "没有回退：答案来自模型，且核对出了有效引用",
    FALLBACK_REASON_NO_CONTEXT: (
        "检索 / 打包为空：一次 LLM 都不调，答案取兜底答复——"
        "换一种说法、放宽过滤条件，或调大打包预算"
    ),
    FALLBACK_REASON_LLM_ERROR: (
        "调用模型时抛异常（超时 / 鉴权 / 网络 / 额度）：答案取兜底答复，"
        "请检查提供方配置与网络，或换一个可用模型重试"
    ),
    FALLBACK_REASON_EMPTY_REPLY: (
        "模型返回空串：answer 就是空串，请检查提供方（超时 / 内容过滤）"
        "并把 max_tokens 调大"
    ),
    FALLBACK_REASON_MODEL_DECLINED: (
        "模型自己说资料不足以回答（命中固定句式）：看 check 里给了几条片段、"
        "覆盖多少——多半是检索给的片段不相关"
    ),
    FALLBACK_REASON_UNUSABLE_CITATIONS: (
        "require_citation=True 而一条合法引用都没有：答案不可核对，"
        "请换一版模板 / 更强模型，或关掉这个开关"
    ),
}

#: 检索为空时的兜底答复（day066 的原文，day069 搬到这里，字面值不变）。
#: 它必须包含**三条出路**而不是一句"没有找到"：用户看到的应该是一个下一步
#: 动作，而不是一次失败宣告。
FALLBACK_NO_CONTEXT = (
    "知识库中没有检索到与该问题相关的内容，因此不作回答。"
    "建议：换一种说法、放宽过滤条件（时间范围 / 元数据），"
    "或确认索引版本是否包含这批资料。"
)

#: v2 要求模型在"资料不足"时使用的固定句式。**它是封闭清单**：
#: 检测（``_declined``）与要求（``RAG_PROMPT_V2`` 的第 2 条约束）必须是同一份，
#: 因此改这里就必须改那里——两边一旦分家，"模型自己说没有"会变成一件
#: 有时能检测、有时检测不到的事，而那种不一致在报告里看不出来。
DECLINE_MARKERS: tuple[str, ...] = (
    "资料中没有相关内容",
    "资料未提及",
    "无法依据资料回答",
)

#: 允许在 ``generate(**overrides)`` 里逐次覆盖的三个参数。**封闭清单**：
#: 多一个键就报错，理由与 ``pipeline.OVERRIDE_KEYS`` 逐字相同——
#: 静默忽略一个覆盖参数会让调用方以为它生效了（"我把 max_tokens 拼成
#: max_token，结果只是答案变短了一点"）。
GENERATION_OVERRIDE_KEYS: tuple[str, ...] = ("temperature", "max_tokens", "prompt_version")

#: 答案里引用编号的写法（``[1]`` / ``[ 1 ]`` / ``[12]``）。
#: 用正则是因为它读的就是**编号语法**：这里不存在"半句也算命中"的风险
#: （与 ``_declined`` 的固定句式判定刻意不同，理由见模块 docstring）。
_MARKER_PATTERN = re.compile(r"\[\s*(\d+)\s*\]")

#: 归一化时丢掉的字符：空白与标点。固定句式判定只关心"那串字在不在"，
#: 而模型有时会写成 "资料中没有相关内容，" 或 "资料中 没有相关内容"。
_DECLINE_NOISE = set(" \t\r\n\u3000，。、；：？！,.;:?!（）()【】[]<>《》“”\"'‘’…—-~")


# --------------------------------------------------------------------------- #
# 三个数据形状：一条检查行 / 一份核对报告 / 一次生成
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CitationCheck:
    """一条检查行：**某个编号到底有没有落在这次提示词里**（M6-D8）.

    ```text
    marker       答案里出现的那个编号（从 1 起）
    record_id    它对应的记录；None = 这个编号**不存在**（即幻觉引用）
    used         这条编号在答案里被引到了没有
    ```

    ``record_id is None`` 与 ``used is False`` 是**两件完全不同的事**，
    这张表把它们分成两种行而不是一种：

    ```text
    record_id=None  模型引用了不存在的编号（[9]）→ 幻觉引用，必须报出来
    used=False      片段给了它、模型没用          → 覆盖率的分母（unused）
    ```

    第一种的编号来自**答案**（模型编出来的），第二种来自**提示词**（我们给的）。
    处置动作因此不同：前者换模型或收紧提示词，后者先看检索给的片段贴不贴题。
    """

    marker: int
    record_id: str | None
    used: bool

    def __post_init__(self) -> None:
        if not isinstance(self.marker, int) or isinstance(self.marker, bool) or self.marker < 1:
            raise GenerationError(
                f"CitationCheck.marker 必须是从 1 起的整数，收到 {self.marker!r}。"
                "编号从 1 起是本层的硬约定（见 context.Citation.marker）："
                "0 不是编号、负数更不是——它们不该出现在这张表里。"
            )
        if self.record_id is not None:
            if not isinstance(self.record_id, str) or not self.record_id.strip():
                raise GenerationError(
                    f"CitationCheck.record_id 必须是非空字符串或 None，"
                    f"收到 {self.record_id!r}。"
                    "None 的含义是'这个编号在提示词里不存在'（幻觉引用）——"
                    "想表达'这条记录没有 id'请去修上游，不要用空串顶替 None。"
                )
        if not isinstance(self.used, bool):
            raise GenerationError(
                f"CitationCheck.used 必须是布尔值，收到 {type(self.used).__name__}："
                "它是'这条编号有没有被引用到'的唯一标记，"
                "非布尔的写法会被当成真值用。"
            )

    @property
    def hallucinated(self) -> bool:
        """这个编号是不是**幻觉引用**（``record_id is None``）."""
        return self.record_id is None

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "marker": self.marker,
            "record_id": self.record_id,
            "hallucinated": self.hallucinated,
            "used": self.used,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if self.hallucinated:
            return f"[{self.marker}] **幻觉引用**：这个编号不在那次提示词里"
        state = "已引用" if self.used else "未引用"
        return f"[{self.marker}] {self.record_id} {state}"


@dataclass(frozen=True)
class GroundingReport:
    """一次接地核对：**答案里的编号与那次提示词的编号对不对得上**（M6-D8）.

    ```text
    cited      答案里出现过的编号（升序去重）—— 来自答案
    valid      cited 里真的落在提示词里的那些（升序）
    invalid    cited 里落不到提示词里的那些（升序）→ 幻觉引用
    unused     给了它、但答案没引用的编号（升序）→ 覆盖率的分母
    coverage   len(valid) / 给出去的片段数（4 位小数，照 PackedContext.fill_ratio）
    grounded   valid 非空 且 invalid 为空（定义见模块 docstring）
    notes      关于**这份报告本身**的说明（空引用、非编号的 [0]、无从核对）
    checks     逐编号的检查行（先 1..n，再按升序接上越界编号）
    ```

    三条不变量在 ``__post_init__`` 里逐条校验，因为它们一旦破了，
    报告里的数字就会互相矛盾，而**读的人只会相信那个看起来最合理的**：

    ```text
    valid ∪ invalid == cited 且两者不交   引用只可能落在两类里
    unused ∩ valid == ∅                   '给了没引用'与'引用了'不可能同时成立
    coverage ∈ [0, 1]                     它是比例，越界说明分母算错了
    ```
    """

    cited: tuple[int, ...] = ()
    valid: tuple[int, ...] = ()
    invalid: tuple[int, ...] = ()
    unused: tuple[int, ...] = ()
    coverage: float = 0.0
    grounded: bool = False
    notes: tuple[str, ...] = ()
    checks: tuple[CitationCheck, ...] = ()

    def __post_init__(self) -> None:
        for label in ("cited", "valid", "invalid", "unused"):
            _validate_marker_tuple(label, getattr(self, label))
        if set(self.valid) & set(self.invalid):
            raise GenerationError(
                f"GroundingReport 的账不自洽：valid={list(self.valid)} 与 "
                f"invalid={list(self.invalid)} 有交集。同一个编号不可能既在提示词里、"
                "又不在提示词里——两类的并集必须恰好是 cited。"
                "出路：这两组编号只能由 _grounding_report 从一份 citations 算出来。"
            )
        if set(self.valid) | set(self.invalid) != set(self.cited):
            raise GenerationError(
                f"GroundingReport 的账不自洽：valid ∪ invalid = "
                f"{sorted(set(self.valid) | set(self.invalid))}，"
                f"而 cited = {list(self.cited)}。答案里出现的每个编号都只能落在"
                "这两类之一：漏掉的那个编号会让'有没有幻觉引用'这个问题答不出来。"
            )
        if set(self.unused) & set(self.valid):
            raise GenerationError(
                f"GroundingReport 的账不自洽：unused={list(self.unused)} 与 "
                f"valid={list(self.valid)} 有交集。'给了它、模型没引用'与"
                "'模型引用了它'不可能同时成立；两者相加才是给出去的片段数。"
            )
        if isinstance(self.coverage, bool) or not isinstance(self.coverage, (int, float)):
            raise GenerationError(
                f"GroundingReport.coverage 必须是数字，收到 {type(self.coverage).__name__}"
                "（比例是算出来的浮点数，字符串会让报告里的百分号变成一句乱码）。"
            )
        number = float(self.coverage)
        if number != number or number in (float("inf"), float("-inf")) or not 0.0 <= number <= 1.0:
            raise GenerationError(
                f"GroundingReport.coverage={self.coverage!r} 必须落在 [0, 1]："
                "它是'有效引用 / 给出去的片段数'，越界说明分母算错了"
                "（例如把幻觉引用也算进了分母）。"
            )
        expected = bool(self.valid) and not self.invalid
        if bool(self.grounded) != expected:
            raise GenerationError(
                f"GroundingReport.grounded={self.grounded!r} 与三组编号不一致："
                f"valid={list(self.valid)}、invalid={list(self.invalid)}，"
                f"按定义应当是 {expected}。"
                "接地判定**不看字符串**（写在这里的只有编号）："
                "'答案里有[没有]相关内容'这种比较会在文案一改之后静默失效。"
            )
        for note in self.notes:
            if not isinstance(note, str) or not note.strip():
                raise GenerationError(
                    f"GroundingReport.notes 里出现了空条目 {note!r}："
                    "注记是给人读的，空串只会让报告里多一行空白。"
                )
        for check in self.checks:
            if not isinstance(check, CitationCheck):
                raise GenerationError(
                    f"GroundingReport.checks 里出现了 {type(check).__name__}："
                    "这张表只收 CitationCheck（编号 / 记录 / 有没有被引用）。"
                    "它应当由 _grounding_report 一次算出来，不要手工拼。"
                )
        object.__setattr__(self, "coverage", round(number, 4))

    @property
    def given(self) -> int:
        """这次**给出去**的片段条数（= ``len(valid) + len(unused)``）.

        为什么可以这样算：提示词里的编号是 1 起连续的（``PackedContext``
        校验过），于是"落在提示词里的"（valid）与"给了没引用的"（unused）
        恰好把 1..n 分完。**它是算出来的而不是存下来的**——
        多存一个 ``given`` 就多一处可能与三组编号分家的副本。
        """
        return len(self.valid) + len(self.unused)

    @property
    def hallucinated(self) -> int:
        """幻觉引用的条数（``invalid`` 的条数，报告里最该被看见的那个数）."""
        return len(self.invalid)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "given": self.given,
            "cited": list(self.cited),
            "valid": list(self.valid),
            "invalid": list(self.invalid),
            "unused": list(self.unused),
            "coverage": round(self.coverage, 4),
            "grounded": self.grounded,
            "notes": list(self.notes),
            "checks": [check.to_dict() for check in self.checks],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        state = "通过" if self.grounded else "未通过"
        return (
            f"核对：给了 {self.given} 条片段 | 答案引用 {len(self.cited)} 条"
            f"（有效 {len(self.valid)} / 幻觉 {len(self.invalid)}）"
            f" | 未引用 {len(self.unused)} 条 | 覆盖 {self.coverage:.1%} | 接地{state}"
        )

    def explain(self) -> list[str]:
        """人类可读诊断：核对口径 / 覆盖率与接地的定义 / 幻觉 / 未引用 / 注记.

        四段与四个问题的对应关系是固定的（"读一份核对报告"的人要找的就是这几句）：

        ```text
        1) 给了几条、引用了几条、有效几条   → 核对的口径
        2) 覆盖率与接地是什么意思           → 别把它读成一次事实核查
        3) 有没有引用了不存在的编号         → 幻觉引用（编号会被点名）
        4) 哪些片段白给了                   → unused，覆盖率的分母
        ```
        """
        lines = [
            f"核对口径：答案里的 [n] 与**那一次提示词**的编号对账 —— "
            f"给了 {self.given} 条片段、答案引用到 {len(self.cited)} 条"
            f"（有效 {len(self.valid)} / 幻觉 {len(self.invalid)}）",
            f"覆盖率 {self.coverage:.1%}（有效引用 / 给出去的片段数），"
            f"接地{'通过' if self.grounded else '未通过'}"
            " —— 通过的定义是'valid 非空且 invalid 为空'，"
            "它只说明**有可核对的依据**，不说明这句话忠实于片段",
        ]
        if self.invalid:
            lines.append(
                f"幻觉引用：编号 {list(self.invalid)} 不在那次提示词里"
                "—— 它不是'多写了一个数字'，而是**指到了一份并不存在的依据**"
            )
        if self.unused:
            lines.append(
                f"未引用的片段：编号 {list(self.unused)}（共 {len(self.unused)} 条）"
                "—— 它们进了提示词却没被用上，是覆盖率的分母"
            )
        for note in self.notes:
            lines.append(f"注记：{note}")
        return lines


@dataclass(frozen=True)
class Generation:
    """一次生成：**答案 + 它是在什么条件下被问出来的 + 核对结果**（M6-D8）.

    ```text
    question        这一次问的是什么（与 RetrievalQuery.text 同一份）
    answer          交付给用户的答案（回退时可能是兜底答复或空串）
    prompt_version  用的是哪一版提示词（报告与评估的分组键）
    prompt_text     真正发出去的那段提示词（**模型看到了什么**要能被读出来）
    llm_called      这一次到底调没调模型（护栏是否生效的唯一证据）
    fallback_reason 六取一：为什么这次答案不是"模型给的可核对答案"
    citations       那次提示词里的编号表（与 PackedContext.citations 同一份）
    check           接地核对报告（三个编号集合 + 覆盖率 + 接地）
    notes           回退的出路 / 幻觉引用 / 未引用条数 / 被忽略的参数
    model           这次用的是哪个模型（装配期给的字符串，不按名字加载）
    latency_ms      这次生成花了多久（含渲染、调用与核对）
    ```

    ``llm_called`` 与 ``fallback_reason`` 分开而不是合成一个字段：前者回答
    "模型有没有机会自由发挥"（护栏的证据），后者回答"这份答案为什么不能按
    正常路径读"（处置动作的依据）。合成之后，"调了但答案没引用"与"根本没调"
    会变成同一个值——而这两件事的处置动作（换模型 / 调检索）完全不同。

    ``prompt_text`` 是这一层最容易被省掉的字段，也最不该省：一份"答案不对"的
    报告若没有它，读的人无法区分"检索给错了片段"与"提示词没把片段喂进去"。
    """

    question: str
    answer: str
    prompt_version: str
    prompt_text: str
    llm_called: bool
    fallback_reason: str
    citations: tuple[Citation, ...] = ()
    check: GroundingReport = field(default_factory=GroundingReport)
    notes: tuple[str, ...] = ()
    model: str = ""
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise GenerationError(
                f"Generation.question 必须是非空字符串，收到 {self.question!r}："
                "一份不知在回答什么问题的答案无法被复核。"
            )
        if not isinstance(self.answer, str):
            raise GenerationError(
                f"Generation.answer 必须是字符串，收到 {type(self.answer).__name__}："
                "模型回复的形状在 RAGGenerator 里已经被收敛成文本。"
            )
        if not isinstance(self.prompt_text, str):
            raise GenerationError(
                f"Generation.prompt_text 必须是字符串，收到 "
                f"{type(self.prompt_text).__name__}："
                "它是'模型看到了什么'的唯一证据，None 会让那次提问无从复现。"
            )
        _validate_prompt_version("Generation.prompt_version", self.prompt_version)
        if not isinstance(self.llm_called, bool):
            raise GenerationError(
                f"Generation.llm_called 必须是布尔值，收到 {type(self.llm_called).__name__}："
                "它是护栏的唯一证据（假 LLM 的 calls 计数与它一一对应）。"
            )
        if self.fallback_reason not in (FALLBACK_REASON_NONE, *FALLBACK_REASONS):
            raise GenerationError(
                f"未知的回退原因 {self.fallback_reason!r}：合法取值是 "
                f"{list(FALLBACK_REASONS)}，外加表示'没有回退'的空串 "
                f"{FALLBACK_REASON_NONE!r}。"
                "回退原因是**封闭清单**：自由文本会让'这一批里有几次不可交付'"
                "这个问题无法统计——出路：用 FALLBACK_REASONS 里的名字。"
            )
        if not isinstance(self.citations, tuple):
            raise GenerationError(
                f"Generation.citations 必须是 tuple，收到 {type(self.citations).__name__}。"
                "写法：citations=context.citations；frozen 形状里塞一个可变对象，"
                "'这份结果不曾被改过'就不再成立。"
            )
        if not isinstance(self.check, GroundingReport):
            raise GenerationError(
                f"Generation.check 必须是 GroundingReport，收到 "
                f"{type(self.check).__name__}："
                "核对结论要跟答案一起走，None 会让'这份答案有没有依据'无从回答。"
            )
        if self.check.given != len(self.citations):
            raise GenerationError(
                f"Generation 的账不自洽：核对报告覆盖 {self.check.given} 条片段，"
                f"而 citations 有 {len(self.citations)} 条。"
                "两者必须来自**同一份**上下文——否则报告里那句'给了 N 条'"
                "说的就是另一次提问的片段。"
            )
        if not isinstance(self.notes, tuple):
            raise GenerationError(
                f"Generation.notes 必须是 tuple，收到 {type(self.notes).__name__}"
            )
        if not isinstance(self.model, str):
            raise GenerationError(
                f"Generation.model 必须是字符串，收到 {type(self.model).__name__}："
                "空串表示'装配期没有报告模型名'（本模块不按名字加载模型）。"
            )
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise GenerationError(
                f"Generation.latency_ms 必须是数字，收到 {type(self.latency_ms).__name__}"
            )
        elapsed = float(self.latency_ms)
        if elapsed != elapsed or elapsed < 0.0 or elapsed == float("inf"):
            raise GenerationError(
                f"Generation.latency_ms 必须是非负有限数，收到 {self.latency_ms!r}："
                "耗时是报告里的一个数字，nan 与负数都只会让它失去意义。"
            )
        # 两条与回退联动的纪律：它们把"这次到底发生了什么"钉成一个**可判定的**
        # 事实，而不是一串需要读者自己拼起来的字段。
        if self.fallback_reason == FALLBACK_REASON_NO_CONTEXT:
            if self.llm_called or self.citations or self.prompt_text:
                raise GenerationError(
                    "no_context 的这次生成必须满足三条：llm_called=False、"
                    "citations=()、prompt_text=''——"
                    f"实际收到 llm_called={self.llm_called}、"
                    f"citations={len(self.citations)} 条、"
                    f"prompt_text {len(self.prompt_text)} 字。"
                    "护栏写在渲染提示词**之前**：没有片段就不该先把它渲染出来，"
                    "否则一次'没调模型'会被记成'问过了但没答案'。"
                )
        if self.fallback_reason == FALLBACK_REASON_NONE:
            if not self.answer.strip():
                raise GenerationError(
                    "报告'没有回退'却给了一份空答案：空串必须走 empty_reply 那条路"
                    "（它的处置动作是查提供方），"
                    "否则'这次答案可用'这句话会让下游拿到一份空交付物。"
                )
            if not self.llm_called:
                raise GenerationError(
                    "报告'没有回退'却写着 llm_called=False："
                    "一次正常的答案只可能来自模型——"
                    "没调模型时必须落到 no_context（或某个回退原因）上。"
                )
            if not self.citations:
                raise GenerationError(
                    "报告'没有回退'却没有任何引用："
                    "正常路径要求答案里至少有一条落在提示词里的引用"
                    "（这是 grounded 的前半条）。"
                    "一条引用都没有时要么走 unusable_citations（开了开关），"
                    "要么把这份答案当作不可核对的事实报出来。"
                )
        object.__setattr__(self, "latency_ms", round(elapsed, 3))

    @property
    def is_fallback(self) -> bool:
        """这次是不是走了回退（``fallback_reason`` 非空）."""
        return self.fallback_reason != FALLBACK_REASON_NONE

    @property
    def grounded(self) -> bool:
        """这份答案有没有可核对的依据（转发 ``check.grounded``）."""
        return self.check.grounded

    def to_dict(self, *, include_prompt: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_prompt=True`` 是默认值（与 ``RerankResult.to_dict`` 的取向一致，
        与 ``RetrievalResult.to_dict`` 相反）：这里最主要的消费者是**排查一次回答**
        （"模型到底被怎么问的"），而那段提示词正是它。只要排序与统计时请显式
        传 ``False``——一段两千字的提示词会把一次 diff 淹掉。
        """
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "prompt_version": self.prompt_version,
            "prompt_chars": len(self.prompt_text),
            "llm_called": self.llm_called,
            "fallback_reason": self.fallback_reason,
            "is_fallback": self.is_fallback,
            "grounded": self.grounded,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "citations": [citation.to_dict() for citation in self.citations],
            "check": self.check.to_dict(),
            "notes": list(self.notes),
        }
        if include_prompt:
            payload["prompt_text"] = self.prompt_text
        return payload

    def to_summary(self) -> dict[str, Any]:
        """进上游公共字段的那一小段摘要（端点与报告直接用它）.

        为什么要有它：端点、演示脚本与 day071 的评估都要把"这次生成做了什么"
        带进各自的结果里，而三份手写的摘要一定会分家（一个多了 ``model``、
        一个忘了 ``fallback_reason``）。形状只在这里定义一次。

        **它刻意不含 ``prompt_text``**（也没有 ``answer``）：摘要是公共字段，
        而一段两千字的提示词放在那里会让它变成第二份正文——
        要看细节的人手里本来就有完整的 ``Generation``（它的 ``to_dict()``
        默认带上提示词）。
        """
        return {
            "prompt_version": self.prompt_version,
            "model": self.model,
            "llm_called": self.llm_called,
            "fallback_reason": self.fallback_reason,
            "grounded": self.grounded,
            "given": self.check.given,
            "coverage": self.check.coverage,
            "valid": len(self.check.valid),
            "invalid": len(self.check.invalid),
            "unused": len(self.check.unused),
            "latency_ms": self.latency_ms,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        question = self.question.replace("\n", " ")[:24]
        answer = self.answer.replace("\n", " ")[:30]
        called = "已调用" if self.llm_called else "未调用"
        reason = f" | 回退 {self.fallback_reason}" if self.is_fallback else ""
        return (
            f"问：{question} | 答：{answer} | 提示词 {self.prompt_version}（{called}）"
            f" | {self.check.summary_line()}{reason}"
        )

    def explain(self) -> list[str]:
        """人类可读诊断：问法 / 核对 / 回退 / 注记.

        四段与四个问题的对应关系是固定的：

        ```text
        1) 这次怎么问的          → 版本 + 提示词长度 + 模型 + 耗时
        2) 答案能不能被核对      → 三组编号 + 覆盖率 + 接地的定义
        3) 为什么走了回退        → 原因 + 出路（没走回退时这一行不渲染）
        4) 还有什么必须被看见    → 注记（含被忽略的参数）
        ```
        """
        lines = [
            f"问法：提示词 {self.prompt_version}（{len(self.prompt_text)} 字）"
            f" | 模型 {self.model or '（未报告）'} | 耗时 {self.latency_ms}ms"
            f" | LLM {'已调用' if self.llm_called else '未调用'}",
        ]
        lines.extend(self.check.explain())
        if self.is_fallback:
            reason = FALLBACK_REASON_DESCRIPTIONS.get(self.fallback_reason, "未知原因")
            lines.append(f"回退 {self.fallback_reason}：{reason}")
        for note in self.notes:
            lines.append(f"注记：{note}")
        return lines


# --------------------------------------------------------------------------- #
# 生成器
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _CallParams:
    """一次 ``generate`` 的实际参数（构造参数 + 逐次覆盖的结果）.

    **私下形状、不进 ``__all__``**：它是"这一次调用"的临时载体，不是给调用方读的
    结果（要读结果请看 ``Generation``）。``notes`` 里装的是**解析覆盖时**发生的
    事情（覆盖了版本 / 覆盖被忽略），由 ``_resolve_overrides`` 一次写好——
    否则同一句话会在两个地方各写一遍，而它们迟早会分家。
    """

    temperature: float
    max_tokens: int
    require_citation: bool
    prompt_version: str
    prompt_template: str
    notes: tuple[str, ...] = ()


class RAGGenerator:
    """把打包好的上下文交给模型，并核对它给出的引用（M6-D8）.

    构造参数分三组，**每组各自决定一件可以被单独讨论的事**：

    ```text
    模板类    prompt / prompt_version       "怎么问"（两个都给时见下）
    采样类    temperature / max_tokens      "让它怎么答"
    交付类    fallback_answer / require_citation / model
                                            "不算数时说什么、算不算数怎么判"
    ```

    ``None`` 在这里有明确含义：**"没指定，去读 settings"**
    （``retrieval_prompt_version`` / ``retrieval_answer_temperature`` /
    ``retrieval_answer_max_tokens``）。于是"项目默认"只有一处定义，
    端点与演示脚本不必各抄一遍。

    ``prompt`` 与 ``prompt_version`` 的关系必须写清，否则会出现"我给的是 v1、
    实际用的是自定义模板"这种静默分歧：

    ```text
    只给 prompt_version   → 用受管模板（清单外的版本号当场报错）
    只给 prompt           → 用注入的模板，版本号记为 'custom'
    两个都给              → **以 prompt 为准**（它连内容都写好了），
                            prompt_version 会被记一句"被忽略的参数"进 notes
    ```

    模型是**注入**的：这一层不认识任何具体后端，因此测试可以用一个"一旦被调用
    就抛异常"的 LLM 证明空上下文护栏生效，也可以用一个返回空串的假模型走通
    ``empty_reply`` 那条路。
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        prompt_version: str | None = None,
        prompt: str | None = None,
        fallback_answer: str = FALLBACK_NO_CONTEXT,
        temperature: float | None = None,
        max_tokens: int | None = None,
        require_citation: bool = False,
        model: str = "",
    ) -> None:
        if not isinstance(llm, BaseLLM):
            raise GenerationError(
                f"llm 必须是 BaseLLM，收到 {type(llm).__name__}。"
                "这一层要核对的是'模型给了什么'：没有模型就没有答案可核对。"
                "任何实现 BaseLLM.chat 的对象都可以注入（假实现是刻意的用法）——"
                "出路：注入一个 BaseLLM，或只用 retrieval.context 做打包。"
            )
        self._llm = llm
        self._ignored_prompt_version: str | None = None
        if prompt is not None:
            self._prompt = _validate_prompt(prompt)
            self._prompt_version = CUSTOM_PROMPT_VERSION
            if prompt_version is not None:
                # 被忽略的版本号**也要过清单校验**：清单外的取值一律报错这条纪律
                # 不该被"反正它不生效"削弱——一个拼错的版本号被记成"已忽略"
                # 会让调用方以为自己指定对了。
                self._ignored_prompt_version = _validate_prompt_version(
                    "prompt_version", prompt_version
                )
        else:
            version = (
                settings.retrieval_prompt_version if prompt_version is None else prompt_version
            )
            self._prompt_version = _validate_prompt_version("prompt_version", version)
            self._prompt = prompt_template_of(self._prompt_version)
        self._fallback_answer = _validate_text("fallback_answer", fallback_answer)
        self._temperature = _resolve_temperature(temperature)
        self._max_tokens = _resolve_max_tokens(max_tokens)
        self._require_citation = _validate_flag("require_citation", require_citation)
        self._model = _validate_model(model)

    # ------------------------------------------------------------------ 只读视图

    @property
    def prompt_version(self) -> str:
        """当前提示词版本（报告与评估按它分组；自定义模板时是 ``'custom'``）."""
        return self._prompt_version

    @property
    def llm(self) -> BaseLLM:
        """注入的模型（端点在"未配置 LLM"时要在构造前就拦住）."""
        return self._llm

    @property
    def temperature(self) -> float:
        """这一次生成用的温度（已按 settings 解析过，不会是 ``None``）."""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """这一次生成要的上限（已按 settings 解析过）."""
        return self._max_tokens

    @property
    def require_citation(self) -> bool:
        """是否把"一条合法引用都没有"当作不可交付（见 ``FALLBACK_REASONS``）."""
        return self._require_citation

    @property
    def fallback_answer(self) -> str:
        """兜底答复（``pipeline`` 的空检索护栏也用它：一处定义、两处引用）."""
        return self._fallback_answer

    def describe(self) -> dict[str, Any]:
        """这一层的配置 + **受管版本清单**（端点直接返回它）.

        回显 ``prompt_versions`` 的理由：端点在收到一个版本号时要能当场列出合法
        取值，而"合法取值有哪些"只有这一个地方说了算（见 ``prompt_template_of``）。
        """
        return {
            "prompt_version": self._prompt_version,
            "prompt_chars": len(self._prompt),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "require_citation": self._require_citation,
            "model": self._model,
            "llm": type(self._llm).__name__,
            "prompt_versions": list(PROMPT_VERSIONS),
        }

    # ------------------------------------------------------------------ 生成入口

    def generate(self, question: str, context: PackedContext, **overrides: Any) -> Generation:
        """回答一个问题，并把答案折成一份可核对的账（九步，顺序即纪律）.

        ```text
        1. 解析本次调用     问题与 overrides 一起校验（清单外的键当场报错）
        2. 空上下文早退     is_empty → no_context，**一次 LLM 都不调**
        3. 渲染提示词       占位符在构造期校验过，因此这里不会缺参数
        4. 调用模型         **宽捕获 Exception** → llm_error（超时/鉴权/网络）
        5. 空回复判定       strip 之后是空串 → empty_reply
        6. 解析 [n]         折成 GroundingReport（valid / invalid / unused）
        7. 拒答判定         命中固定句式 → model_declined
        8. 引用开关         require_citation 且没有合法引用 → unusable_citations
        9. 正常路径         fallback_reason=NONE（答案来自模型且核对通过）
        ```

        第 2 步排在最前面，因为它是**本课最重要的护栏**（与 ``pipeline`` 那道
        同源）：没有片段时模型仍然会写出一段通顺的答案，而读的人看不出区别。

        第 4 步的宽捕获是刻意的：超时、鉴权失败、网络抖动都是**可预期的失败**，
        它们必须变成一份带原因的结果，而不是把整条链路打断——调用方拿到的应该是
        一句"这次没成，原因是这个"，而不是一个栈。

        第 6 步排在第 7、8 步之前：拒答与"一条合法引用都没有"都要读核对结果，
        而核对本身不花钱（纯字符串对账），先算出来就只写一次。
        """
        # 1. 解析本次调用：问题、上下文与 overrides（三者都必须在动手之前站得住）
        question_text = _check_question(question)
        packed = _check_context(context)
        call = self._resolve_overrides(overrides)
        started = time.perf_counter()
        # 2. 空上下文早退：一次 LLM 都不调（护栏写在渲染与调用的**上游**）
        if packed.is_empty:
            logger.warning(
                "生成回退 fallback_reason=%s：打包后的上下文是空的，一次 LLM 都没有调",
                FALLBACK_REASON_NO_CONTEXT,
            )
            return self._no_context(question_text, call, started)
        # 3. 渲染提示词：把 render 的结果**原样存进结果**（"模型看到了什么"）
        prompt_text = call.prompt_template.format(context=packed.text, question=question_text)
        # 4. 调用模型：宽捕获 Exception（超时 / 鉴权 / 网络 / 额度都是可预期的失败）
        try:
            reply = self._llm.chat(
                [Message(role="user", content=prompt_text)],
                temperature=call.temperature,
                max_tokens=call.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 —— 见 docstring 第 4 步的理由
            logger.warning(
                "生成回退 fallback_reason=%s：调用模型时抛出 %s",
                FALLBACK_REASON_LLM_ERROR,
                type(exc).__name__,
            )
            return self._fallback(
                FALLBACK_REASON_LLM_ERROR,
                question_text,
                packed,
                call,
                prompt_text,
                _grounding_report("", packed.citations),
                started,
                error=exc,
            )
        # 5. 空回复：它不是"知识库里没有相关内容"（那是 no_context 的事）
        if not reply.strip():
            logger.warning("生成回退 fallback_reason=%s", FALLBACK_REASON_EMPTY_REPLY)
            return self._fallback(
                FALLBACK_REASON_EMPTY_REPLY,
                question_text,
                packed,
                call,
                prompt_text,
                _grounding_report(reply, packed.citations),
                started,
                answer=reply,
            )
        # 6. 解析 [n]：核对只做**编号层面**的对账（纯字符串，不花一分钱）
        check = _grounding_report(reply, packed.citations)
        # 7. 拒答判定：模型自己说"资料不足"（命中 v2 要求的那几个固定句式）
        if _declined(reply):
            # 命中之后再取一次"命中了哪一句"：判定只要一个布尔，
            # 而报告里要写清是哪一句（否则"模型说资料不够"这个结论无法被复核）。
            declined_by = _declined_marker(reply)
            logger.warning(
                "生成回退 fallback_reason=%s：命中固定句式 %s",
                FALLBACK_REASON_MODEL_DECLINED,
                declined_by,
            )
            return self._fallback(
                FALLBACK_REASON_MODEL_DECLINED,
                question_text,
                packed,
                call,
                prompt_text,
                check,
                started,
                answer=reply,
                declined_by=declined_by,
            )
        # 8. 引用开关：开了它，'一条合法引用都没有'就不能交付
        if call.require_citation and not check.valid:
            logger.warning("生成回退 fallback_reason=%s", FALLBACK_REASON_UNUSABLE_CITATIONS)
            return self._fallback(
                FALLBACK_REASON_UNUSABLE_CITATIONS,
                question_text,
                packed,
                call,
                prompt_text,
                check,
                started,
                discarded=len(reply.strip()),
            )
        # 9. 正常路径：答案来自模型，且核对出了至少一条合法引用
        return Generation(
            question=question_text,
            answer=reply,
            prompt_version=call.prompt_version,
            prompt_text=prompt_text,
            llm_called=True,
            fallback_reason=FALLBACK_REASON_NONE,
            citations=packed.citations,
            check=check,
            notes=tuple(self._notes(FALLBACK_REASON_NONE, call, check, answered=True)),
            model=self._model,
            latency_ms=_elapsed_ms(started),
        )

    def generate_many(self, items: Sequence[tuple[str, PackedContext]]) -> list[Generation]:
        """批量生成（逐条调用 ``generate``，**顺序与入参一致**）.

        与 ``RagPipeline.answer_many`` / ``Retriever.retrieve_many`` 同样是刻意的
        逐条：每一次的 ``llm_called`` / ``fallback_reason`` / 覆盖率都是独立的
        证据，而"这一批里几次不可交付"与"其中一次为什么不可交付"是两个问题。
        把多个问题塞进一次调用会让这两件事混成一个数字。

        入参在**调用任何模型之前**逐条校验（第几条、错在哪、怎么改）：否则一个
        拼错的问题会在第 7 条才炸出来，而前 6 次调用已经花掉了。
        """
        checked = _check_items(items)
        return [self.generate(question, context) for question, context in checked]

    # ------------------------------------------------------------------ 内部

    def _resolve_overrides(self, overrides: dict[str, Any]) -> _CallParams:
        """解析逐次覆盖（**封闭清单**：清单外的键当场报错并列出合法取值）."""
        unknown = sorted(key for key in overrides if key not in GENERATION_OVERRIDE_KEYS)
        if unknown:
            raise GenerationError(
                f"不认识的覆盖参数：{unknown}。可用的键只有 "
                f"{list(GENERATION_OVERRIDE_KEYS)}。"
                "静默忽略一个覆盖参数会让调用方以为它生效了——"
                "例如把 max_tokens 拼成 max_token 之后，唯一的变化是答案莫名变短。"
                "出路：用上面那三个键，或在装配期构造一个新的 RAGGenerator。"
            )
        # None 在这三个键上的含义统一：'这一次不覆盖，用构造期的值'
        # （与构造参数里的 None 是同一个意思：没指定就往下读一层）。
        temperature = _resolve_temperature(
            overrides.get("temperature"), default=self._temperature
        )
        max_tokens = _resolve_max_tokens(overrides.get("max_tokens"), default=self._max_tokens)
        version = overrides.get("prompt_version")
        notes: list[str] = []
        if version is None:
            prompt_version = self._prompt_version
            prompt_template = self._prompt
        elif self._prompt_version == CUSTOM_PROMPT_VERSION:
            # 注入的是自定义模板：逐次覆盖版本号**换不了模板**，因此它无法生效。
            # 被忽略的参数必须进 notes（day067 的第四条纪律）：否则调用方会以为
            # "我指定了 v1，所以这次用的是 v1"，而报告里的 prompt_version 是 custom。
            notes.append(
                f"本次把 prompt_version 覆盖成 {version!r}，但本生成器注入的是自定义"
                f"模板（版本号 {CUSTOM_PROMPT_VERSION}）：覆盖**无法生效**，"
                "这次用的仍然是注入的那份模板——版本号是模板的标签，"
                "换标签不会换内容。出路：要按版本换模板请在装配期重建一个 "
                "RAGGenerator（或者在这次调用里传的是同一份模板才用覆盖）。"
            )
            prompt_version = self._prompt_version
            prompt_template = self._prompt
        else:
            prompt_version = _validate_prompt_version("prompt_version", version)
            prompt_template = prompt_template_of(prompt_version)
            notes.append(
                f"本次覆盖了提示词版本：{self._prompt_version} → {prompt_version}。"
                "逐次换版本是允许的（评估用它做 A/B），但它会**改变答案文本**，"
                "因此这次生成的版本号与构造期不同——按版本分组时请以结果里的 "
                "Generation.prompt_version 为准。"
            )
        if self._ignored_prompt_version is not None:
            notes.append(
                "构造期同时给了 prompt 与 prompt_version="
                f"{self._ignored_prompt_version!r}：**以 prompt 为准**，"
                f"版本号记为 {CUSTOM_PROMPT_VERSION!r}——"
                "被忽略的是那个版本号，不是那份模板。"
            )
        return _CallParams(
            temperature=temperature,
            max_tokens=max_tokens,
            require_citation=self._require_citation,
            prompt_version=prompt_version,
            prompt_template=prompt_template,
            notes=tuple(notes),
        )

    def _notes(
        self,
        reason: str,
        call: _CallParams,
        check: GroundingReport,
        *,
        answered: bool = False,
        error: BaseException | None = None,
        declined_by: str | None = None,
        discarded: int = 0,
    ) -> list[str]:
        """这次生成的全部注记（覆盖 → 回退原因与出路 → 核对结果）.

        三段的顺序与链路顺序一致：**怎么问的**、**为什么不算数**、
        **答案与片段对不上在哪**。``answered=False``（没有实测过一段答案）时
        第三段整段不渲染——那时候"覆盖率为 0"只是空集的性质，不是一次提问的结论。
        """
        notes = list(call.notes)
        notes.extend(
            _reason_notes(
                reason,
                call=call,
                check=check,
                fallback_answer=self._fallback_answer,
                error=error,
                declined_by=declined_by,
                discarded=discarded,
            )
        )
        if answered:
            notes.extend(_grounding_notes(check, require_citation=call.require_citation))
        return notes

    def _no_context(self, question: str, call: _CallParams, started: float) -> Generation:
        """``no_context`` 那条路（**唯一**一条 ``llm_called=False`` 的路）.

        它的三条字段都是"这次什么都没发生"的写法：``prompt_text=''``
        （没有渲染过提示词）、``citations=()``（没有片段可给）、
        ``llm_called=False``。``Generation`` 的 ``__post_init__`` 会把这三条
        钉住——它们是护栏在数据上的形状。
        """
        check = GroundingReport(
            notes=(
                "没有任何片段可核对：这一次检索 / 打包为空，"
                "因此三组编号都只能是空的——它不是'答案没引用'，"
                "而是'没有东西可引'。",
            )
        )
        return Generation(
            question=question,
            answer=self._fallback_answer,
            prompt_version=call.prompt_version,
            prompt_text="",
            llm_called=False,
            fallback_reason=FALLBACK_REASON_NO_CONTEXT,
            citations=(),
            check=check,
            notes=tuple(self._notes(FALLBACK_REASON_NO_CONTEXT, call, check)),
            model=self._model,
            latency_ms=_elapsed_ms(started),
        )

    def _fallback(
        self,
        reason: str,
        question: str,
        context: PackedContext,
        call: _CallParams,
        prompt_text: str,
        check: GroundingReport,
        started: float,
        *,
        answer: str | None = None,
        error: BaseException | None = None,
        declined_by: str | None = None,
        discarded: int = 0,
    ) -> Generation:
        """其余四条回退路径共用的装配（``no_context`` 另有一条，见 ``_no_context``）.

        ``llm_called=True`` 写在**装配这一层**而不是按原因分支：走到这里的手段
        都已经真的调过模型了（``llm_error`` 是调用抛了异常、``empty_reply`` 是
        调用返回空串……），那正是"护栏管的是该不该给片段"这句话的证据。

        ``answer=None`` 的哨兵值只表示"没传"，因此 ``empty_reply`` 传进来的空串
        不会被当成"没传"（两者在这次的回答里是两件不同的事）。
        """
        delivered = self._fallback_answer if answer is None else answer
        return Generation(
            question=question,
            answer=delivered,
            prompt_version=call.prompt_version,
            prompt_text=prompt_text,
            llm_called=True,
            fallback_reason=reason,
            citations=context.citations,
            check=check,
            notes=tuple(
                self._notes(
                    reason,
                    call,
                    check,
                    answered=bool(answer and answer.strip()),
                    error=error,
                    declined_by=declined_by,
                    discarded=discarded,
                )
            ),
            model=self._model,
            latency_ms=_elapsed_ms(started),
        )


# --------------------------------------------------------------------------- #
# 提示词与参数解析（构造期与逐次覆盖共用同一套校验）
# --------------------------------------------------------------------------- #


def prompt_template_of(version: str) -> str:
    """按版本号取**受管**模板（不在清单里 → ``GenerationError`` 并列出合法取值）.

    ``version=None``（没指定）请在外面用 ``settings.retrieval_prompt_version``
    解析完再调本函数：这里只管"这个版本号站不站得住"，不管"缺省是哪一个"——
    两件事分开之后，端点可以直接拿它校验请求体里的版本号。
    """
    if version not in RAG_PROMPTS:
        raise GenerationError(
            f"未知的提示词版本 {version!r}：受管版本只有 {list(PROMPT_VERSIONS)}"
            f"（另有 {CUSTOM_PROMPT_VERSION!r} 表示'调用方自己注入的模板'，"
            "它不是一版受管模板，因此不在 RAG_PROMPTS 里）。"
            "版本号是报告与评估的分组键，清单外的取值会让'这一版好不好'"
            "无法与任何一版模板对上。出路：用上面那几个版本号，"
            f"或在装配期注入 prompt=... 让版本号记为 {CUSTOM_PROMPT_VERSION!r}。"
        )
    return RAG_PROMPTS[version]


def _validate_prompt(prompt: str) -> str:
    """校验模板含 ``{context}`` 与 ``{question}``（缺任何一个 → ``ContextError``）.

    这段逻辑与文案是 day069 从 ``pipeline`` **搬**过来的（不是 import 那个私有名）：
    模块边界上不该依赖另一个模块的私有函数，而搬过来之后 ``pipeline`` 的公开行为
    逐字不变（错误类型与消息一字不改，见 ``tests/test_retrieval_pipeline.py``）。

    用 ``string.Formatter().parse`` 而不是"字符串里搜 ``"{context}"``"：
    后者会把 ``{{context}}``（转义之后的字面量）也当成占位符，
    而那种模板渲染出来是一段谁也读不懂的文本。前者读的是**真正的语法**。

    为什么必须在构造期报错：缺占位符的模板 ``format`` 不会失败
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


def _validate_prompt_version(label: str, value: Any) -> str:
    """版本号必须落在受管清单里（``custom`` 是唯一合法的清单外取值）."""
    if not isinstance(value, str) or not value.strip():
        raise GenerationError(
            f"{label} 必须是非空字符串，收到 {value!r}："
            "版本号是报告与评估的分组键（'换一版提示词之后答案变好了吗'靠它回答），"
            "空串会让这次生成落进一个谁也不认识的分组。"
            f"出路：写 {list(PROMPT_VERSIONS)} 里的一个，"
            f"或注入 prompt=... 让版本号记为 {CUSTOM_PROMPT_VERSION!r}。"
        )
    if value not in PROMPT_VERSIONS and value != CUSTOM_PROMPT_VERSION:
        raise GenerationError(
            f"{label}={value!r} 不在受管清单里：合法取值是 {list(PROMPT_VERSIONS)}，"
            f"外加 {CUSTOM_PROMPT_VERSION!r}（表示'调用方自己注入的模板'）。"
            "清单外的取值会让这次生成无法与任何一版模板对上，"
            f"而那正是 day071 分组实验的判据。出路：用 {list(PROMPT_VERSIONS)}，"
            "或注入 prompt=... 并把版本号交给装配期决定。"
        )
    return value


def _validate_text(label: str, value: Any) -> str:
    """``fallback_answer`` 一类的文本参数：必须是非空字符串."""
    if not isinstance(value, str) or not value.strip():
        raise GenerationError(
            f"{label} 必须是非空字符串，收到 {value!r}："
            "回退时返回一句空话与'不回答'是两件事——前者让用户以为系统坏了"
            "（`我们查到了但什么都没说`）。出路：给它一句带下一步动作的话。"
        )
    return value


def _validate_flag(label: str, value: Any) -> bool:
    """布尔开关：必须真的是布尔值（``"false"`` 这种字符串会被当成真值用）."""
    if not isinstance(value, bool):
        raise GenerationError(
            f"{label} 必须是布尔值，收到 {type(value).__name__}（值为 {value!r}）："
            "它是开关，非布尔的写法（例如字符串 'false'）在 if 里恒为真，"
            "于是'关掉了'这件事从来没有发生过。出路：写 True 或 False。"
        )
    return value


def _validate_model(value: Any) -> str:
    """模型名：只是报告里的一个字符串（本模块**不按名字加载模型**）."""
    if not isinstance(value, str):
        raise GenerationError(
            f"model 必须是字符串，收到 {type(value).__name__}："
            "它只进报告（'这次是谁答的'），不参与任何加载或路由——"
            "本模块不认识任何具体后端。空串表示'装配期没有报告模型名'。"
        )
    return value


def _validate_marker_tuple(label: str, value: Any) -> tuple[int, ...]:
    """一组编号：必须是**升序去重**的整数元组（照 ``Citation.marker`` 的口径）."""
    if not isinstance(value, tuple):
        raise GenerationError(
            f"GroundingReport.{label} 必须是 tuple，收到 {type(value).__name__}。"
            "写法：cited=(1, 2)；frozen 形状里塞一个可变对象，"
            "'这份核对不曾被改过'就不再成立。"
        )
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise GenerationError(
                f"GroundingReport.{label} 里出现了非法编号 {item!r}："
                "编号是从 1 起的整数（见 context.Citation.marker），"
                "0 / 负数 / 字符串都说明这份报告是手工拼出来的。"
            )
    if list(value) != sorted(set(value)):
        raise GenerationError(
            f"GroundingReport.{label}={list(value)} 必须是**升序去重**的："
            "三组编号会被放在一行里比较（valid / invalid / unused），"
            "顺序或重复会让'哪一条没被引用'这种问题读起来要先做一次心算。"
        )
    return value


def _resolve_temperature(value: float | None, *, default: float | None = None) -> float:
    """解析温度：``None`` → 给定缺省 → ``settings.retrieval_answer_temperature``.

    校验口径与 day066 的 ``pipeline._validate_temperature`` 逐字相同
    （[0, 2] 内的有限数，0 是这一层的默认值）：**0.0 才是这一层该有的缺省**，
    因为 RAG 问答是"有依据的复述"，温度越高模型越会改写片段里的事实。
    """
    resolved = default if value is None else value
    if resolved is None:
        resolved = settings.retrieval_answer_temperature
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise GenerationError(
            f"temperature 必须是数字，收到 {type(resolved).__name__}（值为 {resolved!r}）。"
            f"出路：给它一个 [0, 2] 里的数，或留 None 读 "
            "settings.retrieval_answer_temperature。"
        )
    number = float(resolved)
    if number != number or number in (float("inf"), float("-inf")):
        raise GenerationError(
            f"temperature 必须是有限数，收到 {resolved!r}："
            "nan 会让每一次采样都不可复现，而'这次为什么变了'就答不出来。"
        )
    if not 0.0 <= number <= 2.0:
        raise GenerationError(
            f"temperature={number} 超出 [0, 2]：RAG 问答是**有依据的复述**，"
            "过高的温度会让模型改写片段里的事实（默认 0.0 就是为了让它照抄依据）。"
        )
    return number


def _resolve_max_tokens(value: int | None, *, default: int | None = None) -> int:
    """解析 ``max_tokens``：``None`` → 给定缺省 → ``settings.retrieval_answer_max_tokens``.

    必须 >= 1：0 会让模型"没有空间说话"，而它的表现是一次**空回复**——
    那会被误读成"模型认为资料里没有相关内容"（两件事的处置动作完全不同）。
    """
    resolved = default if value is None else value
    if resolved is None:
        resolved = settings.retrieval_answer_max_tokens
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        raise GenerationError(
            f"max_tokens 必须是整数，收到 {type(resolved).__name__}（值为 {resolved!r}）"
        )
    if resolved < 1:
        raise GenerationError(
            f"max_tokens={resolved}：为 0 意味着模型没有空间说话，"
            "而它的表现是一次空回复——那会被误读成'资料里没有相关内容'。"
            "出路：给它一个正数（缺省见 settings.retrieval_answer_max_tokens）。"
        )
    return resolved


# --------------------------------------------------------------------------- #
# 核对与判定
# --------------------------------------------------------------------------- #


def _grounding_report(answer: str, citations: Sequence[Citation]) -> GroundingReport:
    """把一段答案与那次提示词的编号表对账，折成 ``GroundingReport``.

    三条规则，各自对应一种真实写法：

    ```text
    解析 [n]        \\[\\s*(\\d+)\\s*\\]  → 答案里出现过的编号（升序去重）
    编号从 1 起     [0] **不是编号**，因此它不进 cited（只进报告自己的注记）
    两类归属        cited 落在 1..n 里的进 valid，其余进 invalid（幻觉引用）
    ```

    ``[0]`` 为什么不进 ``invalid``：编号从 1 起是本层的硬约定，"0 号引用"不是
    一次幻觉引用（它根本没在指任何位置），而是一次格式噪声。把 0 算进 invalid
    会让"模型引用了不存在的片段"这句指控多出一批并不成立的样本，
    而幻觉引用的**条数**是要进报告、进评估、进结论的一个数字。
    """
    known = [citation.marker for citation in citations]
    markers = [int(match) for match in _MARKER_PATTERN.findall(answer)]
    cited = tuple(sorted({marker for marker in markers if marker >= 1}))
    inside = set(known)
    valid = tuple(marker for marker in cited if marker in inside)
    invalid = tuple(marker for marker in cited if marker not in inside)
    unused = tuple(sorted(inside - set(cited)))
    coverage = round(len(valid) / len(known), 4) if known else 0.0
    checks = tuple(
        CitationCheck(
            marker=citation.marker,
            record_id=citation.record_id,
            used=citation.marker in set(cited),
        )
        for citation in citations
    ) + tuple(
        CitationCheck(marker=marker, record_id=None, used=True) for marker in invalid
    )
    notes: list[str] = []
    noise = sorted({marker for marker in markers if marker < 1})
    if noise:
        rendered = "、".join(f"[{marker}]" for marker in noise)
        notes.append(
            f"答案里出现了 {rendered}：编号从 1 起（见 context.Citation.marker），"
            "0 不是编号——它既不算引用、也不算幻觉引用，因此不进 cited/valid/invalid "
            "任何一项，只在这里留一句提醒。"
        )
    if not known:
        notes.append(
            "没有任何片段可核对：这份报告的三组编号都只能是空的"
            "（不是'答案没引用'，而是'这次没有东西可引'）。"
        )
    elif not cited:
        notes.append(
            "答案里没有出现任何 [n]：这是**没有引用**，与 invalid 的"
            "'引用了不存在的编号'是两件事——前者改提示词或换模型，"
            "后者要查那段话是怎么被编出来的。"
        )
    return GroundingReport(
        cited=cited,
        valid=valid,
        invalid=invalid,
        unused=unused,
        coverage=coverage,
        grounded=bool(valid) and not invalid,
        notes=tuple(notes),
        checks=checks,
    )


def _normalize_for_decline(text: str) -> str:
    """拒答判定前的归一化：去掉空白与标点（**不做大小写折叠**）.

    折叠范围写在这里而不是藏着：模型会把固定句式写成 "资料中没有相关内容，"
    或 "资料中 没有相关内容"，因此空白与标点必须去掉。
    本课只认中文句式，所以这里**不做**大小写折叠——将来若要收英文句式，
    必须在这里补一次 ``casefold()``，否则 ``No relevant content`` 永远匹配不上
    （写下这一句是为了让下一个人知道它还没做，而不是以为它已经做了）。
    """
    return "".join(char for char in text if char not in _DECLINE_NOISE)


def _declined_marker(answer: str) -> str | None:
    """答案命中的那一个固定句式（没命中返回 ``None``）.

    返回**命中的那一句**而不是布尔值：报告里要能写出"命中哪一句"，
    否则"模型说资料不够"这个结论无法被复核（见 ``_reason_notes``）。
    """
    normalized = _normalize_for_decline(answer)
    for marker in DECLINE_MARKERS:
        if marker in normalized:
            return marker
    return None


def _declined(answer: str) -> bool:
    """模型是不是自己说了"资料不足"（固定句式的**包含判定**）.

    为什么不做正则、不做模糊匹配，写在模块 docstring 的"拒答为什么用固定句式"
    那一节：三条理由合起来只有一句——**可枚举、可测试、可解释**。
    """
    return _declined_marker(answer) is not None


# --------------------------------------------------------------------------- #
# 注记
# --------------------------------------------------------------------------- #


def _reason_notes(
    reason: str,
    *,
    call: _CallParams,
    check: GroundingReport,
    fallback_answer: str,
    error: BaseException | None = None,
    declined_by: str | None = None,
    discarded: int = 0,
) -> list[str]:
    """按回退原因写注记（**现象 + 为什么 + 出路**，没回退时返回空列表）.

    每一条都给出路，理由与 ``EMPTY_REASONS`` 同一句：一次失败宣告如果不带下一步
    动作，读的人只能猜该动哪个参数（而六种回退的处置动作是六件不同的事）。
    """
    notes: list[str] = []
    if reason == FALLBACK_REASON_NO_CONTEXT:
        notes.append(
            "回退原因 no_context（检索 / 打包为空）：这一次**一个 LLM 都没有调**"
            "——模型手里没有片段却仍然会写出一段通顺的答案，而读的人看不出区别，"
            "因此护栏写在调用的上游（与 pipeline 的空检索护栏同源）。"
        )
        notes.append(
            "答案取的是兜底答复（它带着三条出路）：要拿到模型给的答案，"
            "请先让检索给出片段——换一种说法、放宽过滤条件（时间范围 / 元数据），"
            "或调大 settings.retrieval_max_context_chars。"
        )
        notes.append(
            f"本次被忽略的参数：temperature={call.temperature}、"
            f"max_tokens={call.max_tokens}、"
            f"require_citation={call.require_citation} 都没有生效"
            "——一次都没调模型，它们没有作用对象。"
        )
    elif reason == FALLBACK_REASON_LLM_ERROR:
        error_name = type(error).__name__ if error is not None else "Exception"
        notes.append(
            f"回退原因 llm_error：调用模型时抛出 {error_name}: {error}"
            "——超时、鉴权失败、网络抖动与额度不足都会这样，"
            "它们都**不是**'资料里没有相关内容'（那种情况一次 LLM 都不会调）。"
        )
        notes.append(
            "这一次**确实**调用了模型（llm_called=True）：护栏管的是'该不该把片段"
            "交给模型'，管不了'给了片段之后调用失败'。出路：检查提供方的配置与"
            "网络（鉴权 / 超时 / 额度），或换一个可用模型重试。"
        )
        notes.append(
            "答案取的是兜底答复，而它的文案是为 no_context 写的"
            f"（{_preview_text(fallback_answer, limit=20)}…）：与本次原因不符——"
            "要一句贴切的兜底答复，请在装配期注入 fallback_answer=...。"
        )
    elif reason == FALLBACK_REASON_EMPTY_REPLY:
        notes.append(
            "回退原因 empty_reply：模型返回了空回复（strip 之后是空串）"
            "——这一次的 answer 就是空串，而引用与上下文都在。"
        )
        notes.append(
            "它**不是**'知识库里没有相关内容'（那是 no_context 的事，"
            "那种情况一次 LLM 都不会调）：超时、内容过滤与 max_tokens 太小"
            f"都会给出空串。出路：调大 max_tokens（当前 {call.max_tokens}）、"
            "检查提供方的内容过滤与超时，再跑一次。"
        )
    elif reason == FALLBACK_REASON_MODEL_DECLINED:
        notes.append(
            "回退原因 model_declined：模型自己说资料不足"
            f"（命中固定句式 {declined_by!r}）——这不是失败，而是 v2 明确要求"
            "它做的事：句式固定之后，'模型自己说没有'从一句无法检测的话"
            "变成了一件可检测的事实（DECLINE_MARKERS）。"
        )
        notes.append(
            f"这次给了 {check.given} 条片段、答案引用到 {len(check.valid)} 条："
            "若 valid 为空而 unused 非空，多半是检索给的片段不相关"
            "（该改查询或扩召回），而不是模型不肯答——两种原因对应两种动作，"
            "因此它必须被分开读。"
        )
    elif reason == FALLBACK_REASON_UNUSABLE_CITATIONS:
        notes.append(
            "回退原因 unusable_citations：require_citation=True 而这次一条**合法**"
            f"引用都没有（给了 {check.given} 条片段）——答案里没有任何指向片段的"
            "[n]，因此没有可核对的东西。"
        )
        notes.append(
            f"模型给的 {discarded} 字答案已被丢弃、答案取兜底答复：一条无法溯源的话"
            "交付出去，读的人会把它当成有依据的结论。出路：换一版模板或更强的模型"
            "（引用格式被忽略了），或把 require_citation 关掉——那样它会作为报告里"
            "的一项，而不是一道闸门。"
        )
    return notes


def _grounding_notes(check: GroundingReport, *, require_citation: bool) -> list[str]:
    """核对结果里"必须被看见"的三件事（幻觉引用 / 未引用条数 / 覆盖率为 0）.

    三条都不阻断链路，因此唯一的痕迹就是注记。它们都**按情况渲染**：
    没发生就不占一行（"注记多了会被读的人忽略"是这一层最该守的纪律，
    day066 的 ``test_the_happy_path_has_no_notes`` 钉的就是它）。

    ``unused`` 那条只在 ``require_citation=True`` 时渲染：默认关掉时"引了几条"
    是常态，每一次回答都念一遍会把它变成噪音——而开了这个开关之后，
    "给了的片段有几条没被用上"正是需要看的账。
    """
    notes: list[str] = []
    if check.hallucinated:
        notes.append(
            f"幻觉引用 {check.hallucinated} 条："
            f"编号 {list(check.invalid)} 不在那次提示词里"
            f"（那次只给了 [1]~[{check.given}]）——它不是'数字写多了'，"
            "而是**指到一份并不存在的依据**；要查这几行请看 check.checks 里"
            "record_id 为空的那几条。"
        )
    if require_citation and check.unused:
        notes.append(
            f"未引用的片段 {len(check.unused)} 条（编号 {_preview_ids(check.unused)}）："
            f"这次给了 {check.given} 条、答案引用到 {len(check.valid)} 条，"
            f"覆盖率 {check.coverage:.1%}。require_citation=True 时"
            "它们会被读成'这次资料没被用上'的证据。"
        )
    if check.given and not check.valid:
        notes.append(
            "覆盖率为 0：给了片段的那些引用一条都没落在提示词里——"
            "要分清两种情形：invalid 非空（引用了不存在的编号）与 cited 为空"
            "（干脆没写引用），后者的处置动作是改提示词或换模型。"
        )
    return notes


# --------------------------------------------------------------------------- #
# 入参与耗时的校验
# --------------------------------------------------------------------------- #


def _check_question(question: Any) -> str:
    """问题必须是非空字符串（收敛成 strip 之后的形式）."""
    if not isinstance(question, str):
        raise GenerationError(
            f"question 必须是字符串，收到 {type(question).__name__}："
            "这一次生成的入口就是一句话——其它形状（向量、id 列表）请直接用 "
            "backend.query() / RetrievalQuery。"
        )
    stripped = question.strip()
    if not stripped:
        raise GenerationError(
            "空问题在生成这一层没有定义：请给一段非空文本。"
            "空白串会让提示词里的'## 问题'一节变成空的，"
            "而模型会照着带引用的资料片段自由发挥——那正是本课要拦的事。"
        )
    return stripped


def _check_context(context: Any) -> PackedContext:
    """上下文必须是 ``PackedContext``（本模块唯一的输入形状）."""
    if not isinstance(context, PackedContext):
        raise GenerationError(
            f"context 必须是 PackedContext，收到 {type(context).__name__}。"
            "写法：context=pack_context(retrieval.hits, max_chars=...)。"
            "本模块只认这一种形状：提示词从它的 text 渲染、核对从它的 citations "
            "对账——少了编号表，'答案里的 [2] 指谁'就无从回答。"
        )
    return context


def _check_items(items: Sequence[Any]) -> list[tuple[str, PackedContext]]:
    """逐条校验批量入参（问题 + 上下文），**在任何模型调用之前**."""
    if not isinstance(items, (list, tuple)):
        raise GenerationError(
            f"generate_many 的入参必须是序列，收到 {type(items).__name__}。"
            "写法：generate_many([('一句话', context), ('另一句', context2)])；"
            "顺序与返回的列表逐条对应。"
        )
    checked: list[tuple[str, PackedContext]] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, tuple) or len(item) != 2:
            raise GenerationError(
                f"generate_many 的第 {position} 条不是 (问题, 上下文) 二元组，"
                f"收到 {item!r}。这里的每一对都会**逐条**送进 generate，"
                "形状不对会让它在半路才炸——而那时前面几次调用已经花掉了。"
            )
        question, context = item
        checked.append((_check_question(question), _check_context(context)))
    return checked


def _elapsed_ms(started: float) -> float:
    """这次生成花了多少毫秒（含渲染、调用、异常与核对）.

    量整段而不是只量那一次 ``chat``：``llm_error`` 那条路上"等了多久才失败"
    本身就是一个有用的数字（超时大多是从这里看出来的），
    而 ``no_context`` 那条路的耗时应当接近 0——它正是"什么都没发生"的证据。
    """
    return round((time.perf_counter() - started) * 1000, 3)


def _preview_ids(values: Sequence[int]) -> str:
    """一组编号的短预览（超过 6 条就折成"前 6 条 + 共 N 条"）."""
    head = "、".join(f"[{value}]" for value in values[:6])
    if len(values) <= 6:
        return head
    return f"{head} 等 {len(values)} 条"


def _preview_text(text: str, *, limit: int) -> str:
    """一段文本的短预览（换行折成空格，超长截断）."""
    flat = text.replace("\n", " ")
    return flat if len(flat) <= limit else flat[:limit]


__all__ = [
    "CURRENT_PROMPT",
    "CURRENT_PROMPT_VERSION",
    "CUSTOM_PROMPT_VERSION",
    "DECLINE_MARKERS",
    "DEFAULT_PROMPT_VERSION",
    "FALLBACK_NO_CONTEXT",
    "FALLBACK_REASON_DESCRIPTIONS",
    "FALLBACK_REASON_EMPTY_REPLY",
    "FALLBACK_REASON_LLM_ERROR",
    "FALLBACK_REASON_MODEL_DECLINED",
    "FALLBACK_REASON_NO_CONTEXT",
    "FALLBACK_REASON_NONE",
    "FALLBACK_REASON_UNUSABLE_CITATIONS",
    "FALLBACK_REASONS",
    "GENERATION_OVERRIDE_KEYS",
    "PROMPT_CHANGELOG",
    "PROMPT_VERSION_V1",
    "PROMPT_VERSION_V2",
    "PROMPT_VERSIONS",
    "RAG_PROMPT_V1",
    "RAG_PROMPT_V2",
    "RAG_PROMPTS",
    "REQUIRED_PROMPT_FIELDS",
    "CitationCheck",
    "Generation",
    "GroundingReport",
    "RAGGenerator",
    "prompt_template_of",
]

