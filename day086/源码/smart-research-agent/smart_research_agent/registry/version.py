"""版本三元组：一份模型版本的身份是什么（M5-D9）.

day050~day057 让"训练"这件事变得可复现了：数据有指纹（``domain_data`` 的
``manifest.fingerprint``）、适配器有内容哈希（``peft.trainer`` 的
``content_sha256``）、检查点里有全量超参快照。但那是**三份各自独立**的
证据，谁也回答不了最常被问到的那一句话：

> "线上这个效果，是哪一份数据 + 哪一次训练 + 哪一版基座造出来的？"

本模块把这句话变成一个**内容寻址的键**：

```text
version_key = sha256(canonical(base_model, adapter_sha256, dataset_fingerprint))[:16]
```

三个字段缺一不可，理由都是具体的：

| 字段 | 缺了它会怎样 |
|------|-------------|
| ``base_model`` | 两个仓库里同名同格式的适配器，挂在不同的基座上会给出**完全不同**的输出，而且**不会报错**（day052 的推理脚本专门为此打警告） |
| ``adapter_sha256`` | 分不清"调了学习率重训"与"同一个权重" |
| ``dataset_fingerprint`` | 分不清"同一份数据多训了两轮"与"换了一批数据"——而这两件事的效果差异通常比调参大得多 |

## 为什么是三元组，而不是"一个自增号"

自增号（``v17``）能排序、能当主键，但它**不携带任何信息**：拿到 ``v17``
你无法判断它能不能替代 ``v16``，也无法判断两个仓库里的 ``v17`` 是不是
同一件东西。内容寻址的键牺牲了"可读的连续性"，换回来两件事：

1. **同一个键在任何机器上指向同一份产物**——CI 里重跑一次，键相同就意味着
   "什么都没变"（day057 ``dataset_fingerprint`` 是同一种设计）；
2. **键的相等是"可替代"的充分条件**：键相同 → 基座、数据、权重三者全同 →
   行为逐位相同。

代价是键无法排序，因此**三元组与语义化版本号是两件事，两个都要**：

- ``version_key``（16 位十六进制）：**身份**，用于比对与去重；
- ``version``（``1.2.0``）：**人类叙事**，用于"这是第几版、相对上一版动了什么"。

把两者混成一个字段，就会出现 day057 里已经出现过的那类缺陷——
报告里一个字段被两种口径消费，谁也说不清它到底在说什么。

## 版本号怎么递增：让"变化的那一项"决定

递增规则不是拍脑袋，而是**直接由三元组的差异推出**，这也是本模块
``bump_kind_for`` 存在的原因：

| 发生了什么 | 递增位 | 为什么 |
|-----------|--------|--------|
| ``base_model`` 变了 | ``major`` | 整条适配器链作废：旧适配器挂不上新基座，新旧版本的输出**没有可比性** |
| 只有 ``dataset_fingerprint`` 变了 | ``minor`` | 同一基座上的新一版领域数据，评估口径变了但模型谱系连续 |
| 只有 ``adapter_sha256`` 变了 | ``patch`` | 同数据同基座的重训（换种子、调步数），这是可比较的实验 |

第三条看起来最"小"，实际上最重要：**重训是持续微调里最高频的动作**，
把它标成 ``minor`` 会让版本号迅速失去区分度（v1.9 → v2.0 只因为多跑了一次
同样的数据）。而 ``major`` 的语义被保留给基座替换——这是唯一一个
让"回滚"都必须重新评估的动作。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.registry.errors import RegistryError

#: 版本键保留的十六进制位数。与 day057 的 ``DATASET_FINGERPRINT_LENGTH``
#: 同口径（64 bit）——在本项目"一百多个版本"的量级上碰撞概率可以忽略，
#: 而 16 位能在报告的一行里写完。
VERSION_KEY_LENGTH = 16

#: 适配器内容哈希的长度下限。真实的 ``peft.trainer.adapter_content_hash``
#: 给的是 64 位 sha256；这里只要求 >= 16，是为了让测试与"手工登记一份
#: 外部产物"的场景都能用短哈希。**下限而不是等值**：写死 64 会让
#: "从别的流水线拿来的 32 位哈希"无法登记，而那正是持续微调的常态。
MIN_ADAPTER_HASH_LENGTH = 16

#: 数据集指纹的长度下限（``domain_data.dataset_fingerprint`` 给 16 位）。
MIN_DATASET_FINGERPRINT_LENGTH = 8

#: 语义化版本号：只接受 ``X.Y.Z`` 三段数字。
#: 刻意**不接受** ``1.2`` 或 ``v1.2.0``：宽松解析会让 "1.2" 与 "1.2.0"
#: 同时存在，排序与去重立刻出问题——这与 day057 ``parse_example`` 拒绝
#: 模糊格式是同一条纪律。
SEMVER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

#: 递增位。``major`` = 基座更换 / ``minor`` = 数据集更换 / ``patch`` = 重训。
BUMP_MAJOR = "major"
BUMP_MINOR = "minor"
BUMP_PATCH = "patch"
BUMP_KINDS: tuple[str, ...] = (BUMP_MAJOR, BUMP_MINOR, BUMP_PATCH)

#: 首次登记的版本号。
INITIAL_VERSION = "1.0.0"

#: 十六进制校验（大小写都收，比较前统一转小写）。
_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


def normalize_digest(value: str, *, field_name: str) -> str:
    """把一份哈希/指纹归一化：去空白、去 ``sha256:`` 前缀、转小写.

    三个动作各自都踩过一次坑：

    - **去空白**：从终端复制哈希时尾部常带换行，而 ``"abc\\n" != "abc"``；
    - **去前缀**：``AdapterManifest`` 的日志里印的是 ``sha256:abc...``，
      人肉抄进配置时会把前缀一起抄进来，于是同一个适配器有两个键；
    - **转小写**：``hexdigest()`` 给小写，但很多人手写的大写——
      哈希比较必须是"值比较"，不是"字符串比较"。

    归一化之后**立刻做十六进制校验**：一个含 ``g`` 的"哈希"是抄错了，
    而它如果混进键里，只会表现为"版本总是不匹配"，不会有任何报错。
    """
    cleaned = value.strip()
    if cleaned.lower().startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :].strip()
    cleaned = cleaned.lower()
    if not cleaned:
        raise RegistryError(f"{field_name} 不能为空")
    if not _HEX_PATTERN.match(cleaned):
        raise RegistryError(
            f"{field_name} 必须是十六进制字符串，收到 {value!r}"
            f"（常见原因：把 'sha256:' 前缀或哈希以外的文本抄了进来）"
        )
    return cleaned


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """规范化序列化：排序键 + 紧凑分隔符.

    与 ``peft.trainer._canonical_bytes`` 逐字相同的一条理由：内容寻址只有在
    序列化规范固定之后才成立。同一份三元组换了键顺序就会得到不同的键，
    那么"这两个版本是不是同一份产物"就仍然答不出来。
    """
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_version_payload(
    *, base_model: str, adapter_sha256: str, dataset_fingerprint: str
) -> dict[str, str]:
    """三元组的规范字典（三个字段都已归一化与校验）."""
    base = base_model.strip()
    if not base:
        raise RegistryError("base_model 不能为空")
    adapter = normalize_digest(adapter_sha256, field_name="adapter_sha256")
    dataset = normalize_digest(dataset_fingerprint, field_name="dataset_fingerprint")
    if len(adapter) < MIN_ADAPTER_HASH_LENGTH:
        raise RegistryError(
            f"adapter_sha256 至少 {MIN_ADAPTER_HASH_LENGTH} 位十六进制，"
            f"收到 {len(adapter)} 位（{adapter}）"
        )
    if len(dataset) < MIN_DATASET_FINGERPRINT_LENGTH:
        raise RegistryError(
            f"dataset_fingerprint 至少 {MIN_DATASET_FINGERPRINT_LENGTH} 位十六进制，"
            f"收到 {len(dataset)} 位（{dataset}）"
        )
    return {
        "adapter_sha256": adapter,
        "base_model": base,
        "dataset_fingerprint": dataset,
    }


def version_key(
    *, base_model: str, adapter_sha256: str, dataset_fingerprint: str
) -> str:
    """由三元组算出内容寻址的版本键（sha256 前 16 位）.

    注意它**不是**对"模型文件"算哈希——那在 7B 上是十几 GB 的读盘，
    而三元组的三个字段本身就足以确定产物（适配器哈希已经覆盖了权重）。
    """
    payload = canonical_version_payload(
        base_model=base_model,
        adapter_sha256=adapter_sha256,
        dataset_fingerprint=dataset_fingerprint,
    )
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:VERSION_KEY_LENGTH]


@dataclass(frozen=True)
class VersionTriple:
    """模型版本的三元组：基座 + 适配器内容哈希 + 数据集指纹.

    ``frozen=True`` 是刻意的：三元组一旦确定，它就是这条版本的**身份**，
    任何一处改动都应当产出一个新的 ``VersionTriple``（进而一个新的版本键），
    而不是就地修改——就地修改会让索引里已经落盘的那一行与内存里的对象
    指向两件不同的事。
    """

    base_model: str
    adapter_sha256: str
    dataset_fingerprint: str

    def __post_init__(self) -> None:
        payload = canonical_version_payload(
            base_model=self.base_model,
            adapter_sha256=self.adapter_sha256,
            dataset_fingerprint=self.dataset_fingerprint,
        )
        # 归一化后的值写回字段：这样 ``VersionTriple("M", "AB", "CD")`` 与
        # ``VersionTriple(" M ", "ab", "cd")`` 相等且哈希相同。
        object.__setattr__(self, "base_model", payload["base_model"])
        object.__setattr__(self, "adapter_sha256", payload["adapter_sha256"])
        object.__setattr__(self, "dataset_fingerprint", payload["dataset_fingerprint"])

    @property
    def key(self) -> str:
        """内容寻址的版本键（恒为 16 位十六进制）."""
        return version_key(
            base_model=self.base_model,
            adapter_sha256=self.adapter_sha256,
            dataset_fingerprint=self.dataset_fingerprint,
        )

    @property
    def short_adapter(self) -> str:
        """适配器哈希前 12 位——报告里印它，完整哈希留在 ``adapter_sha256``."""
        return self.adapter_sha256[:12]

    def to_dict(self) -> dict[str, str]:
        """投影为可直接 ``json.dumps`` 的字典（含派生出的键）."""
        return {
            "base_model": self.base_model,
            "adapter_sha256": self.adapter_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VersionTriple:
        """从 ``to_dict`` 的产物还原（忽略派生键 ``key``）."""
        return cls(
            base_model=str(payload["base_model"]),
            adapter_sha256=str(payload["adapter_sha256"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
        )

    def describe(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"基座 {self.base_model} | 适配器 sha256:{self.short_adapter} | "
            f"数据 {self.dataset_fingerprint} | 键 {self.key}"
        )


def comparable(left: VersionTriple, right: VersionTriple) -> bool:
    """两个版本的效果是否**可直接比较**.

    只有基座相同、数据集指纹相同才可比。这条约束是在线运维里最容易
    被违反的一条，而违反它的后果是**静默**的：

    - 数据集换了，评估集的分布也换了（day057 的评估集是从同一份数据切出来的），
      于是"合格率从 0.62 掉到 0.55"既可能是模型变差，也可能是题目变难；
    - 基座换了，输出分布整个变了，两者的分数根本不在一个尺度上。

    这两种情况下 ``evaluate_candidate`` 都不该给出"提升/退步"的结论，而应
    回答"不可比"。**把不可比说出口，比给出一个看似精确的差值有用得多。**
    """
    return (
        left.base_model == right.base_model
        and left.dataset_fingerprint == right.dataset_fingerprint
    )


def parse_semver(text: str) -> tuple[int, int, int]:
    """解析 ``X.Y.Z``，返回三段整数元组；不合规抛 ``RegistryError``."""
    match = SEMVER_PATTERN.match(text.strip())
    if match is None:
        raise RegistryError(
            f"版本号必须是 X.Y.Z 形式，收到 {text!r}"
            "（不接受 '1.2' 或 'v1.2.0'：宽松解析会让 '1.2' 与 '1.2.0' 同时存在）"
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_semver(triple: tuple[int, int, int]) -> str:
    """把三段整数元组格式化为 ``X.Y.Z``."""
    return ".".join(str(part) for part in triple)


def semver_sort_key(text: str) -> tuple[int, int, int]:
    """排序键：**按整数段排序**而不是按字符串排序.

    字符串排序会把 ``1.10.0`` 排在 ``1.9.0`` 前面——版本号一旦涨到两位数，
    "最新版本"就会答错，而它恰恰是 ``head()`` 的实现基础。
    """
    return parse_semver(text)


def bump_version(version: str, *, kind: str = BUMP_PATCH) -> str:
    """按递增位生成下一个版本号.

    规则与语义化版本的通行约定一致：递增某一位时，它右边的位全部归零。
    缺少这一步（例如 ``major`` 只加一不归零）会让 ``1.9.3`` → ``2.9.3``，
    版本号看起来"跨了大版本却还带着小版本号"，失去叙事意义。
    """
    if kind not in BUMP_KINDS:
        raise RegistryError(f"未知的递增位 {kind!r}，可选 {', '.join(BUMP_KINDS)}")
    major, minor, patch = parse_semver(version)
    if kind == BUMP_MAJOR:
        return format_semver((major + 1, 0, 0))
    if kind == BUMP_MINOR:
        return format_semver((major, minor + 1, 0))
    return format_semver((major, minor, patch + 1))


def bump_kind_for(
    previous: VersionTriple | None,
    *,
    base_model: str,
    dataset_fingerprint: str,
) -> str:
    """由"相对上一版变化的那一项"推出递增位.

    ``previous=None``（首版）也返回 ``BUMP_MINOR``：首版拿 ``INITIAL_VERSION``
    就够，这里的返回值只在"已有历史"时才会被 ``next_version`` 使用。
    """
    if previous is None:
        return BUMP_MINOR
    if previous.base_model != base_model.strip():
        return BUMP_MAJOR
    if previous.dataset_fingerprint != normalize_digest(
        dataset_fingerprint, field_name="dataset_fingerprint"
    ):
        return BUMP_MINOR
    return BUMP_PATCH


def next_version(existing: list[str], *, kind: str = BUMP_PATCH) -> str:
    """在已有版本号集合上算出下一个版本号（空集合给 ``INITIAL_VERSION``）.

    取的是**最大的**版本号而不是最后一个登记的：手工登记与自动登记混用时，
    "最后一个"取决于登记顺序，而"最大的"才是版本号本身的语义。
    """
    if not existing:
        return INITIAL_VERSION
    latest = max(existing, key=semver_sort_key)
    return bump_version(latest, kind=kind)


@dataclass(frozen=True)
class VersionConflict:
    """一次"同一个键已经有主"的冲突记录（诊断用）.

    只在 ``ModelRegistry.register`` 拒绝重复登记时构造：冲突必须给出
    "键相同但版本号/数据集不同"的具体证据，否则运维看到 "conflict"
    三个字之后唯一能做的事就是去翻索引文件。
    """

    key: str
    existing_version: str
    incoming_version: str
    fields: dict[str, tuple[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "key": self.key,
            "existing_version": self.existing_version,
            "incoming_version": self.incoming_version,
            "fields": {name: list(pair) for name, pair in self.fields.items()},
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        changed = ", ".join(f"{name}: {old} -> {new}" for name, (old, new) in self.fields.items())
        return (
            f"版本键 {self.key} 已被 {self.existing_version} 占用，"
            f"无法登记 {self.incoming_version}（差异：{changed or '无'}）"
        )


__all__ = [
    "BUMP_KINDS",
    "BUMP_MAJOR",
    "BUMP_MINOR",
    "BUMP_PATCH",
    "INITIAL_VERSION",
    "MIN_ADAPTER_HASH_LENGTH",
    "MIN_DATASET_FINGERPRINT_LENGTH",
    "SEMVER_PATTERN",
    "VERSION_KEY_LENGTH",
    "VersionConflict",
    "VersionTriple",
    "bump_kind_for",
    "bump_version",
    "canonical_version_payload",
    "comparable",
    "format_semver",
    "next_version",
    "normalize_digest",
    "parse_semver",
    "semver_sort_key",
    "version_key",
]
