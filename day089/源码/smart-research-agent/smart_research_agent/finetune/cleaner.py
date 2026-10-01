"""微调数据清洗与质量过滤（M5-D1）.

原始数据几乎不可能直接拿来训练：网页抓取的文本带零宽字符与 NBSP 变体，
人工复制的内容有空行与多余空格，模型生成的"答案"里躺着 ``TODO`` 占位符，
同一份语料被多个来源重复采集。本模块把这一步拆成三件可单独测试的事：

1. **归一化**（``clean_text``）：NFKC 统一全半角/相容字符 → 剔除控制字符
   → 折叠空白 → 去首尾。它保证"看起来一样的两条样本"在字符串层面也
   一样，是去重能被信任的前提。
2. **质量规则**（``QualityRule`` + ``default_rules``）：每条规则是一个
   纯函数，返回 True 表示**通过**。规则表可注入、可扩展，顺序即优先级
   （第一条不通过即拒），拒绝原因逐条记账——数据被丢掉多少、为什么丢，
   必须是一个能印出来的数字，而不是一句"清洗过了"。
3. **去重**（``dedupe_key``）：对"清洗 + 小写后的 prompt"取 SHA-1 前 16 位
   作为指纹，保留首次出现。用指纹而非原文，是为了不把长文本常驻内存；
   先清洗再取指纹，是为了让"仅空白/全半角不同"的重复也被识别。

清洗发生在过滤**之前**（``DatasetCleaner.run`` 先统一清洗再跑规则）：
规则要判断的是"最终会喂给模型的那些字符"，若拿原始文本判长度，
一条 ``"\\n\\n\\n"`` 的输出会以 3 个字符通过 ``min_output_chars=8`` 的
门槛，却在训练时变成空样本。顺序错了，规则就形同虚设。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from smart_research_agent.finetune.schema import TrainingExample

#: 占位符黑名单：命中即判定为"答案没写完"，绝不能进训练集.
#: 全部小写比较，因此英文占位符大小写不敏感。
PLACEHOLDER_MARKERS: tuple[str, ...] = ("todo", "待补充", "lorem", "tbd", "fixme")

#: 去重指纹保留的十六进制字符数：16 位 = 64 bit，重复碰撞概率对本课程
#: 规模（万级样本）可以忽略，同时比完整 40 位指纹更省日志空间。
DEDUPE_KEY_LENGTH = 16

#: 一条样本的处置结论：通过全部规则、被某条规则拒绝、或因重复被丢弃。
#: 用常量而不是裸字符串，是为了让 report 的键与决策记录不会拼错。
DECISION_KEPT = "kept"
DECISION_DUPLICATE = "duplicate"


def normalize_whitespace(text: str) -> str:
    """折叠空白：连续空格/制表符压成一个空格，多行结构保留一行.

    为什么**保留换行**而不是把 ``\\n`` 也压成空格：ReAct 轨迹、代码与
    步骤说明的语义都依赖换行（``Thought:`` / ``Action:`` 各占一行），
    压成一行会把结构化数据变成一坨无法阅读的长文本，训练时反而更难学。
    """
    if not text:
        return ""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def strip_control_chars(text: str) -> str:
    """剔除 Unicode 控制/格式字符（保留换行与制表符）.

    零宽空格（U+200B）、方向控制符（U+202E）、BOM（U+FEFF）这类字符肉眼
    不可见，却会改变分词结果、破坏字符串比较，是数据清洗里最典型的
    "隐形污染源"。``\\n``/``\\t`` 属于结构字符，必须留下。
    """
    if not text:
        return ""
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )


def clean_text(text: str) -> str:
    """文本清洗总入口：NFKC 归一化 → 控制字符剔除 → 空白折叠.

    NFKC 把全角字母数字、兼容字符、不间断空格（NBSP, U+00A0）等统一到
    标准形式：``"ＲＡＧ"`` 与 ``"RAG"`` 归一后相同，``"a\\u00a0b"`` 与
    ``"a b"`` 归一后相同。没有这一步，去重会把同一句话当成两条。
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return normalize_whitespace(strip_control_chars(normalized))


def dedupe_key(example: TrainingExample) -> str:
    """样本去重指纹：``clean_text(prompt_text).lower()`` 的 SHA-1 前 16 位.

    只用 prompt（不含 output）作为去重依据，是因为"同一个问题配两个不同
    答案"在监督微调里不是冗余而是**冲突**——两条都留着，模型学到的就是
    一个含糊的平均；去重把它们压成一条（保留首次出现），把问题暴露给
    数据评审，而不是把矛盾喂给训练。
    """
    normalized = clean_text(example.prompt_text).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:DEDUPE_KEY_LENGTH]


@dataclass(frozen=True)
class QualityRule:
    """一条质量规则：``check`` 返回 True 表示**通过**（不是"命中"）.

    用"通过"而非"命中"作为真值方向，可以让规则组合保持简单：任一规则
    返回 False 即拒绝。若反过来（True 表示命中），调用处就会出现
    ``if any(not r.check(...))`` 这种双重否定，评审时极易看错。
    """

    name: str
    reason: str
    check: Callable[[TrainingExample], bool]


def default_rules(
    *,
    min_output_chars: int,
    max_output_chars: int,
    max_instruction_chars: int,
    banned_patterns: Sequence[str] = (),
) -> list[QualityRule]:
    """构建默认质量规则表（顺序即优先级，第一条不通过即拒）.

    规则清单与阈值全部来自配置（``finetune_*`` 配置项），因此"多长算太短"
    这件事在部署层面可调，而不需要改代码。
    """
    patterns = tuple(pattern.lower() for pattern in banned_patterns if pattern)
    return [
        QualityRule(
            name="output_too_short",
            reason=f"输出不足 {min_output_chars} 字符（清洗后），信息量太低",
            check=lambda example: len(example.output) >= min_output_chars,
        ),
        QualityRule(
            name="output_too_long",
            reason=f"输出超过 {max_output_chars} 字符，多半是粘贴了长文而非答案",
            check=lambda example: len(example.output) <= max_output_chars,
        ),
        QualityRule(
            name="instruction_too_long",
            reason=f"指令超过 {max_instruction_chars} 字符，长上下文指令需单独评测",
            check=lambda example: len(example.instruction) <= max_instruction_chars,
        ),
        QualityRule(
            name="empty_instruction",
            reason="指令（或 prompt）为空，模型无从学起",
            check=lambda example: bool(example.instruction.strip()),
        ),
        QualityRule(
            name="placeholder_output",
            reason="输出含 TODO/待补充/lorem 之类占位符，答案并未真正完成",
            check=lambda example: not any(
                marker in example.output.lower() for marker in PLACEHOLDER_MARKERS
            ),
        ),
        QualityRule(
            name="banned_pattern",
            reason="指令或输出命中禁用词表（合规/版权风险）",
            check=lambda example: not any(
                pattern in f"{example.prompt_text}\n{example.output}".lower()
                for pattern in patterns
            ),
        ),
        QualityRule(
            name="unverified_source",
            reason="样本带 unverified 标签：答案未经核对，不进入训练集",
            check=lambda example: "unverified" not in example.tags,
        ),
    ]


@dataclass
class FilterDecision:
    """一条样本的处置记录：谁、因为什么被保留或丢弃."""

    example: TrainingExample
    rule: str
    reason: str


@dataclass
class FilterReport:
    """清洗报告：总量、保留量、各拒绝原因的计数与逐条决策.

    ``rejected`` 是 ``total - kept``（含重复），因此 ``rejected`` 与
    ``duplicates`` 有重叠：``drop_reasons["duplicate"]`` 记重复数，
    ``duplicates`` 字段是它的冗余快照，便于报表直接取用。
    """

    total: int
    kept: int
    rejected: int
    duplicates: int
    drop_reasons: dict[str, int] = field(default_factory=dict)
    decisions: list[FilterDecision] = field(default_factory=list)

    @property
    def keep_rate(self) -> float:
        """保留率（空输入按 0.0 计，而不是 1.0——没有数据不代表"全都合格"）."""
        return self.kept / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典（含逐条决策的摘要）."""
        return {
            "total": self.total,
            "kept": self.kept,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "keep_rate": round(self.keep_rate, 4),
            "drop_reasons": dict(self.drop_reasons),
            "decisions": [
                {
                    "index": index,
                    "rule": decision.rule,
                    "reason": decision.reason,
                    "source": decision.example.source,
                    "preview": decision.example.prompt_text[:60],
                }
                for index, decision in enumerate(self.decisions)
            ],
        }


class DatasetCleaner:
    """数据集清洗器：清洗 → 质量过滤 → 去重（保留首次出现）.

    用法::

        cleaner = DatasetCleaner(min_output_chars=8)
        kept, report = cleaner.run(examples)
        report.drop_reasons   # {"output_too_short": 1, "duplicate": 2, ...}

    规则表可整体注入（``rules=``）：给一条样本集定制"禁止出现某类表述"
    的规则时，不必去改 ``default_rules``——那会让所有数据集一起变严。
    """

    def __init__(
        self,
        *,
        min_output_chars: int = 8,
        max_output_chars: int = 4000,
        max_instruction_chars: int = 1000,
        banned_patterns: Sequence[str] = (),
        dedupe: bool = True,
        rules: Sequence[QualityRule] | None = None,
    ):
        self.min_output_chars = min_output_chars
        self.max_output_chars = max_output_chars
        self.max_instruction_chars = max_instruction_chars
        self.banned_patterns = tuple(banned_patterns)
        self.dedupe = dedupe
        self.rules: list[QualityRule] = (
            list(rules)
            if rules is not None
            else default_rules(
                min_output_chars=min_output_chars,
                max_output_chars=max_output_chars,
                max_instruction_chars=max_instruction_chars,
                banned_patterns=banned_patterns,
            )
        )

    @staticmethod
    def clean_example(example: TrainingExample) -> TrainingExample:
        """生成清洗后的新样本（不修改入参，原始数据永远可回溯）."""
        return replace(
            example,
            instruction=clean_text(example.instruction),
            output=clean_text(example.output),
        )

    def run(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[TrainingExample], FilterReport]:
        """执行清洗流水线，返回（保留样本, 清洗报告）.

        三个阶段严格按顺序：**清洗 → 规则 → 去重**。
        - 清洗必须最先：规则与指纹都要看"最终字符"；
        - 去重必须最后：先按规则把废样本丢掉，再对 survivors 取指纹，
          这样才能保证"保留的是首次出现的合法样本"，而不是被一个
          本该拒绝的样本抢先占位、把后面合法的重复项挤掉。
        """
        cleaned = [self.clean_example(example) for example in examples]
        kept: list[TrainingExample] = []
        decisions: list[FilterDecision] = []
        drop_reasons: dict[str, int] = {}
        duplicates = 0
        seen: set[str] = set()

        for example in cleaned:
            failed = next((rule for rule in self.rules if not rule.check(example)), None)
            if failed is not None:
                drop_reasons[failed.name] = drop_reasons.get(failed.name, 0) + 1
                decisions.append(
                    FilterDecision(example=example, rule=failed.name, reason=failed.reason)
                )
                continue

            if self.dedupe:
                key = dedupe_key(example)
                if key in seen:
                    duplicates += 1
                    drop_reasons[DECISION_DUPLICATE] = drop_reasons.get(DECISION_DUPLICATE, 0) + 1
                    decisions.append(
                        FilterDecision(
                            example=example,
                            rule=DECISION_DUPLICATE,
                            reason="与前面样本重复（清洗+小写后的 prompt 指纹相同）",
                        )
                    )
                    continue
                seen.add(key)

            kept.append(example)
            decisions.append(
                FilterDecision(example=example, rule=DECISION_KEPT, reason="通过全部质量规则")
            )

        total = len(cleaned)
        report = FilterReport(
            total=total,
            kept=len(kept),
            rejected=total - len(kept),
            duplicates=duplicates,
            drop_reasons=drop_reasons,
            decisions=decisions,
        )
        return kept, report
