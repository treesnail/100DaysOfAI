"""数据集统计、切分与落盘（M5-D1）.

数据工程做完"采集 → 清洗"之后，剩下三件必须做对的事：

1. **统计**（``compute_stats``）：在训练前就把数据的画像印出来——平均指令
   长度、输出长度分布、来源与标签分布。这些数字是超参选择的依据：平均
   输出 40 字却把 ``max_length`` 设成 2048，就是在浪费显存；来源分布里
   某个源占了 90%，就要警惕"训出来的模型只会说那一种话"。
   token 数复用 ``llm.tokenizer.TokenCounter``，且默认强制走字符级估算
   （``prefer_tiktoken=False``）：统计脚本必须离线、确定、零依赖。
2. **切分**（``split_dataset``）：训练/评估必须来自同一分布且**不重叠**。
   用 ``random.Random(seed)`` 打乱保证可复现——同一份数据、同一个 seed
   永远得到同一个切分，否则"这次评估涨了 2 个点"可能只是切分换了。
   同时保证 n>=2 时两侧都非空：空评估集会让评估静默失效。
3. **落盘**（``dump_bundle``）：写出 ``train.jsonl`` / ``eval.jsonl``。
   格式（alpaca/chat/prompt-completion）在落盘那一刻才决定，上游的清洗与
   统计永远作用在统一的 ``TrainingExample`` 上。

``build_dataset`` 是 ``default_collector().collect()`` 的便捷入口，
供脚本与 API 一行拿到数据集包。
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from smart_research_agent.finetune.collector import DatasetBundle, default_collector
from smart_research_agent.finetune.schema import ALPACA, TrainingExample, dump_jsonl
from smart_research_agent.llm.tokenizer import TokenCounter


@dataclass
class DatasetStats:
    """数据集画像：规模、长度分布、token 估算与来源/标签分布.

    平均值保留原始 float（不做四舍五入），只在 ``to_dict`` 输出时取两位
    小数——把"计算"与"展示"分开，测试断言才不会被展示格式绑架。
    """

    count: int
    avg_instruction_chars: float
    avg_output_chars: float
    min_output_chars: int
    max_output_chars: int
    avg_estimated_tokens: float
    source_distribution: dict[str, int] = field(default_factory=dict)
    tag_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典（平均值两位小数）."""
        return {
            "count": self.count,
            "avg_instruction_chars": round(self.avg_instruction_chars, 2),
            "avg_output_chars": round(self.avg_output_chars, 2),
            "min_output_chars": self.min_output_chars,
            "max_output_chars": self.max_output_chars,
            "avg_estimated_tokens": round(self.avg_estimated_tokens, 2),
            "source_distribution": dict(self.source_distribution),
            "tag_distribution": dict(self.tag_distribution),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"样本 {self.count} 条 | 平均指令 {self.avg_instruction_chars:.1f} 字 / "
            f"平均输出 {self.avg_output_chars:.1f} 字 | "
            f"输出长度 {self.min_output_chars}~{self.max_output_chars} 字 | "
            f"平均约 {self.avg_estimated_tokens:.1f} token/条"
        )


def compute_stats(
    examples: Sequence[TrainingExample], counter: TokenCounter | None = None
) -> DatasetStats:
    """计算数据集画像；空列表返回全零统计（不抛错）.

    空集不抛错是刻意的：清洗后一条不剩是一种**合法的中间状态**（说明数据
    质量门槛把所有样本都挡住了），此时调用方需要的是"0 条"这个事实，
    而不是一个异常——异常会让"为什么没数据"的排查被错误处理逻辑掩盖。
    """
    # 缺省强制字符级估算：离线、确定、不依赖 tiktoken 词表文件
    token_counter = counter if counter is not None else TokenCounter(prefer_tiktoken=False)
    count = len(examples)
    if count == 0:
        return DatasetStats(
            count=0,
            avg_instruction_chars=0.0,
            avg_output_chars=0.0,
            min_output_chars=0,
            max_output_chars=0,
            avg_estimated_tokens=0.0,
        )

    instruction_lengths = [len(example.instruction) for example in examples]
    output_lengths = [len(example.output) for example in examples]
    estimated_tokens = [
        token_counter.count_tokens(example.prompt_text)
        + token_counter.count_tokens(example.output)
        for example in examples
    ]
    source_distribution = Counter(example.source for example in examples)
    tag_distribution: Counter[str] = Counter()
    for example in examples:
        tag_distribution.update(example.tags)

    return DatasetStats(
        count=count,
        avg_instruction_chars=sum(instruction_lengths) / count,
        avg_output_chars=sum(output_lengths) / count,
        min_output_chars=min(output_lengths),
        max_output_chars=max(output_lengths),
        avg_estimated_tokens=sum(estimated_tokens) / count,
        source_distribution=dict(source_distribution),
        tag_distribution=dict(tag_distribution),
    )


def split_dataset(
    examples: Sequence[TrainingExample], *, eval_ratio: float = 0.2, seed: int = 42
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """切分为（训练集, 评估集）——可复现、且 n>=2 时两侧都非空.

    - 用 ``random.Random(seed)`` 就地打乱（不污染全局随机状态，测试
      与并发调用彼此隔离）；
    - 评估集条数 = ``max(1, round(n * eval_ratio))``，但**不超过 n-1**：
      把最后一条留给训练集，避免出现"评估集之外没有训练数据"的退化切分；
    - ``eval_ratio`` 必须落在 ``[0, 1)``：等于 1 会让训练集为空，
      小于 0 无意义，两者都在这里直接拒绝，而不是留到训练时报错。
    """
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError(f"eval_ratio 必须落在 [0, 1) 区间，收到 {eval_ratio}")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    eval_count = max(1, round(n * eval_ratio))
    eval_count = min(eval_count, n - 1)
    return shuffled[eval_count:], shuffled[:eval_count]


def dump_bundle(
    bundle: DatasetBundle,
    out_dir: str | Path,
    *,
    fmt: str = ALPACA,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Path]:
    """把数据集包按切分写成 ``train.jsonl`` / ``eval.jsonl``，返回路径字典.

    返回路径而非"成功"布尔值：调用方需要能直接打印/上传这两个文件，
    让函数把路径交出来，比让它自己写日志更有用。
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, eval_set = split_dataset(bundle.examples, eval_ratio=eval_ratio, seed=seed)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    dump_jsonl(train, train_path, fmt)
    dump_jsonl(eval_set, eval_path, fmt)
    return {"train": train_path, "eval": eval_path}


def build_dataset(data_dir: str | Path | None = None) -> DatasetBundle:
    """便捷入口：``default_collector(data_dir).collect()``.

    ``data_dir`` 缺省走 ``settings.finetune_data_dir``；传入临时目录即可
    在测试里完全离线地跑通整条流水线。
    """
    return default_collector(data_dir).collect()
