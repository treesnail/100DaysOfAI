"""微调数据格式规范与统一样本模型（M5-D1）.

为什么微调的第一步是"数据工程"而不是"选模型"：一次微调的效果上限由
数据决定，方法只决定你能不能逼近这个上限。而数据工程的第一件事是
**统一表示**——ALpaca、ShareGPT（chat）、纯 completion 三种格式满天飞，
训练脚本各认一种，清洗规则又要各写一遍。本模块把三者收敛到同一个
内存对象 ``TrainingExample``：

    alpaca            {"instruction", "input", "output"}
    chat              {"messages": [{"role", "content"}, ...]}
    prompt-completion {"prompt", "completion"}

三者可以互相转换（``to_dict(fmt)``），因此"清洗一次、训练多种格式"
成为可能：清洗与去重永远作用在 ``TrainingExample`` 上，只在落盘那一刻
才投影回目标格式。

字段命名遵循既有约定：
- ``instruction`` 是"要做什么"，``input`` 是"要处理的材料"（可为空），
  ``output`` 是"标准答案"——这是 alpaca 的三元组语义，也是本课程
  所有指令样本的公共骨架；
- ``system`` 单独存放，因为 chat 模板里它是独立的一条消息，而 alpaca
  格式没有它的位置（投影时会被丢弃，这是格式本身的限制，不是 bug）；
- ``source``/``tags``/``license`` 是**数据治理元信息**：来源可追溯、
  标签可筛（``is_safety_example``）、许可证可审计。微调数据一旦混入
  不可再分发的语料，模型权重都不能安心发布，这三点必须随样本落盘。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: ALpaca 格式：instruction / input / output 三元组，最经典的指令微调格式
ALPACA = "alpaca"
#: ShareGPT 风格 chat 格式：messages 列表，支持 system/user/assistant 多轮
CHAT = "chat"
#: 纯续写格式：prompt / completion，适合把知识灌进模型（不要求"指令"语义）
PROMPT_COMPLETION = "prompt-completion"

#: 本模块支持的格式全集；新增格式时同步修改 SUPPORTED_FORMATS 与 to_dict/parse_example
SUPPORTED_FORMATS: tuple[str, ...] = (ALPACA, CHAT, PROMPT_COMPLETION)

#: 缺省格式：alpaca 字段最少、最直观，作为种子数据的默认写法
DEFAULT_FORMAT = ALPACA


class DatasetFormatError(ValueError):
    """数据格式或必填字段不合法（继承 ValueError，便于调用方按类型捕获）.

    刻意不复用 ``json.JSONDecodeError`` 之类的底层异常：调用方关心的是
    "这条样本不能用于训练"，而不是"第几个字符解析失败"。
    """


def _as_text(value: object) -> str:
    """把任意字段规整为文本：None → ""，其余走 str().

    数据工程里字段类型漂移是常态（数字被写成 ``123`` 而不是 ``"123"``），
    与其在每处调用都写 ``str(x or "")``，不如在此统一——也避免 ``None``
    被 ``str()`` 变成字符串 ``"None"`` 这种最隐蔽的脏数据。
    """
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_tags(value: object) -> tuple[str, ...]:
    """把 tags 字段规整为字符串元组（list/tuple/单字符串都接受）."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(_as_text(item) for item in value)
    return (_as_text(value),)


@dataclass
class TrainingExample:
    """统一的训练样本：所有格式在内存中的公共表示.

    用法::

        example = TrainingExample(instruction="什么是 RAG？", output="检索增强生成……")
        example.prompt_text          # "什么是 RAG？"
        example.to_dict("chat")      # {"messages": [{"role": "user", ...}, ...]}

    设计取舍：这是**可变 dataclass 而非 frozen**，因为采集器需要就地
    补齐缺失字段（例如 JSONL 源在样本未标注来源时用 ``SourceSpec.name``
    补 ``source``）；清洗器则用 ``dataclasses.replace`` 生成清洗后的新对象，
    不修改原始样本——原始数据永远可回溯，是数据工程的底线。
    """

    instruction: str
    output: str
    input: str = ""
    system: str | None = None
    source: str = ""
    tags: tuple[str, ...] = ()
    license: str = ""

    @property
    def prompt_text(self) -> str:
        """模型实际看到的"提问"部分：instruction 与 input 的拼接.

        两者都非空时用空行分隔——instruction 是"任务描述"、input 是
        "待处理材料"，视觉与语义上都需要一条边界；只有 instruction 时
        直接返回它，避免多余的前后空白污染 token 统计。
        """
        if self.instruction and self.input:
            return f"{self.instruction}\n\n{self.input}"
        return self.instruction

    def to_alpaca(self) -> dict:
        """投影为 alpaca 字典：``input`` 为空也保留该键.

        保留空键而非省略，是为了让下游（HF datasets / LLaMA-Factory 等）
        拿到**结构稳定**的记录：字段时有时无会让 schema 推断失败。
        """
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_chat(self) -> dict:
        """投影为 chat 字典：system（可选）→ user → assistant.

        system 为空时不写这条消息，而不是写一条空内容的消息——空 system
        在多数模板里会被渲染成 "system: " 这样的噪声前缀。
        """
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.prompt_text})
        messages.append({"role": "assistant", "content": self.output})
        return {"messages": messages}

    def to_prompt_completion(self) -> dict:
        """投影为续写字典：prompt（含 input）+ completion."""
        return {"prompt": self.prompt_text, "completion": self.output}

    def to_dict(self, fmt: str = DEFAULT_FORMAT) -> dict:
        """按目标格式投影为可 json.dumps 的字典（未知格式抛 DatasetFormatError）."""
        if fmt == ALPACA:
            return self.to_alpaca()
        if fmt == CHAT:
            return self.to_chat()
        if fmt == PROMPT_COMPLETION:
            return self.to_prompt_completion()
        raise DatasetFormatError(
            f"不支持的数据集格式 {fmt!r}，可选：{', '.join(SUPPORTED_FORMATS)}"
        )

    def is_safety_example(self) -> bool:
        """是否为安全对齐样本（``tags`` 含 "safety"）.

        安全拒答样本与普通指令样本在训练时常需要不同权重/不同阶段
        （对齐税：混得不好会损伤通用能力），因此需要一个显式判定，
        而不是让训练脚本去字符串匹配标签。
        """
        return "safety" in self.tags


def parse_example(raw: dict, fmt: str = DEFAULT_FORMAT) -> TrainingExample:
    """把一条原始字典解析为 ``TrainingExample``（格式不合法抛 DatasetFormatError）.

    三种格式的必填字段：
      - alpaca：``instruction`` 与 ``output`` 非空（``input`` 可空）；
      - chat：``messages`` 里至少各含一条 user 与 assistant，且内容非空
        （system 可选）；
      - prompt-completion：``prompt`` 与 ``completion`` 非空。

    除三元组外，``system``/``source``/``tags``/``license`` 若存在也会
    一并带出——治理元信息不该在解析这一步被丢掉。
    """
    if fmt not in SUPPORTED_FORMATS:
        raise DatasetFormatError(
            f"不支持的数据集格式 {fmt!r}，可选：{', '.join(SUPPORTED_FORMATS)}"
        )
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"样本必须是 JSON 对象（dict），实际是 {type(raw).__name__}")

    if fmt == CHAT:
        instruction, output, system = _parse_chat_fields(raw)
    else:
        instruction, output = _parse_text_fields(raw, fmt)
        system = _as_text(raw.get("system"))

    # 只有空白字符的字段等价于空：没有内容的"答案"是脏数据，不能放行
    if not instruction.strip():
        raise DatasetFormatError(f"{fmt} 格式的 instruction/prompt 不能为空")
    if not output.strip():
        raise DatasetFormatError(f"{fmt} 格式的 output/completion 不能为空")

    return TrainingExample(
        instruction=instruction,
        output=output,
        input=_as_text(raw.get("input")),
        system=system or None,
        source=_as_text(raw.get("source")),
        tags=_as_tags(raw.get("tags")),
        license=_as_text(raw.get("license")),
    )


def _parse_text_fields(raw: dict, fmt: str) -> tuple[str, str]:
    """alpaca / prompt-completion 的字段提取（两种格式只是键名不同）."""
    if fmt == PROMPT_COMPLETION:
        return _as_text(raw.get("prompt")), _as_text(raw.get("completion"))
    return _as_text(raw.get("instruction")), _as_text(raw.get("output"))


def _parse_chat_fields(raw: dict) -> tuple[str, str, str]:
    """chat 格式的字段提取：从 messages 中找 system/user/assistant.

    "各一条"的判定放在这里而不是交给下游，是因为多轮对话在微调中需要
    额外的 loss masking 策略；当前先只支持单轮（user → assistant），
    多轮样本应显式报错，而不是被静默截断成第一轮。
    """
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DatasetFormatError("chat 格式要求 messages 为非空列表")

    def _find(role: str) -> str:
        for message in messages:
            if isinstance(message, dict) and message.get("role") == role:
                return _as_text(message.get("content"))
        return ""

    user = _find("user")
    assistant = _find("assistant")
    if not user.strip() or not assistant.strip():
        raise DatasetFormatError("chat 格式要求 messages 中各含一条非空的 user 与 assistant 消息")
    return user, assistant, _find("system")


def load_jsonl(path: str | Path) -> list[dict]:
    """读取 JSONL 文件为字典列表（跳过空行，解析失败向上抛）.

    与 ``JSONLSource`` 的分工：本函数是"读一个确定合法的文件"的严格版本
    （失败即暴露），采集器则用宽松版本（坏行记 warning 后跳过）。两种
    策略各有适用面：离线固化数据集要严格，线上采集要容错。
    """
    records: list[dict] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def dump_jsonl(
    examples: Iterable[TrainingExample], path: str | Path, fmt: str = DEFAULT_FORMAT
) -> int:
    """把样本按指定格式写入 JSONL，返回写出的条数（父目录自动创建）.

    写入用 UTF-8 + ``ensure_ascii=False``：中文样本若被转义成 ``\\uXXXX``，
    文件体积会膨胀数倍，且人眼不可读——数据文件是要被 review 的。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(fmt), ensure_ascii=False) + "\n")
            count += 1
    return count
