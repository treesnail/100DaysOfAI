"""向量缓存：把"这段文本的向量是多少"变成一次查表（M6-D4）.

day064 的 `VectorIngestPipeline` 是**逐条编码**的，实测留下了那个刺眼的数字：

```text
重放同一批：输入 6 → 写入 0（未变 6）| 编码调用 6 次
```

**库没变，钱照花。** 今天的第一个任务就是把它消掉——办法不是在 day064
的库里加判断，而是把"文本 → 向量"这件事本身做成一张表。

## 键就是全部设计

```python
key = vector_key(provider, model, dimension, text)   # sha256(...)[:16]
```

四个输入一个都不能少，其中**三个是编码器的身份**（day065 在 `types.py`
里已经论证过：身份不进键，换模型后缓存会命中旧模型的向量，
而那是不报错的一类错误）。

## 三条刻意的行为

**1. 命中与未命中都要计数。** `hits` / `misses` 是"缓存有没有在省钱"的
唯一证据；只留一个 `size` 的话，"缓存装满了但一次没命中"与
"缓存命中率 90%"看起来一样。

**2. 载入时身份不一致直接拒读。** 逻辑上键里已经含身份，
所以旧条目**永远不会命中**——载入它们是"无害但无用"的。
这里选择报错而不是默默接受，理由是**"无害"这件事本身很容易被误解**：
一个能载入别人模型缓存的接口，会让人以为"缓存可以跨模型共享"，
而真到换模型那天，命中的会是错向量（如果键里没有身份的话）。
宁可在载入时说清楚，也不要在某天靠"键里恰好有身份"兜住一个已经写错的假设。

**3. `prune` 必须显式调用，绝不自动清理。** 删缓存是不可逆的，
而"哪些条目还有用"这个判断只有调用方知道（它手里才有当前清单）。
自动清理的策略无论怎么写，都会在某些使用方式下删掉**马上还要用的**条目，
而那种损失的表现是"下一次构建突然变慢"，看不出是缓存的问题。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from smart_research_agent.indexing.errors import EncodingError, ManifestError
from smart_research_agent.indexing.types import VECTOR_KEY_LENGTH, vector_key

#: 缓存文件格式版本。与清单一样：**不匹配时拒读**。
CACHE_VERSION = 1


class EmbeddingCache:
    """一段文本 → 一个向量的内存表（可落盘、可修剪）.

    它**不关心**向量是怎么来的：只负责"这个键有没有值"。
    编码、批大小、提供方重试都在 ``encoder.py`` 里。
    这条边界让缓存能被单独测试：塞进去、取出来、落盘、读回，
    全程不需要一个真实的 embedding 提供方。
    """

    def __init__(
        self,
        *,
        path: str = "",
        provider: str = "",
        model: str = "",
        dimension: int = 0,
    ) -> None:
        self._path = path
        self._provider = provider
        self._model = model
        self._dimension = int(dimension)
        self._entries: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ 身份

    @property
    def provider(self) -> str:
        """编码器实现名（进键的一部分）."""
        return self._provider

    @property
    def model(self) -> str:
        """编码器模型/配置标识（进键的一部分）."""
        return self._model

    @property
    def dimension(self) -> int:
        """向量维度（进键的一部分；0 表示"不限"）."""
        return self._dimension

    @property
    def path(self) -> str:
        """落盘位置（空串表示只在内存里）."""
        return self._path

    @property
    def identity_key(self) -> str:
        """身份指纹（报告里用它说明"这份缓存属于谁"）."""
        return vector_key(self._provider, self._model, 0, f"dimension={self._dimension}")

    # ------------------------------------------------------------------ 读写

    def key(self, text: str) -> str:
        """算一段文本的缓存键（见模块 docstring）."""
        return vector_key(self._provider, self._model, self._dimension, text)

    def has(self, text: str) -> bool:
        """键是否存在——**不计入命中/未命中**.

        为什么与 ``get`` 分开：``has`` 是给"先探测再决策"的调用方用的
        （例如计划阶段想知道"这一批里有多少能直接复用"）。
        如果它会计数，那么一次探测就会把命中率污染成两倍。
        """
        return self.key(text) in self._entries

    def get(self, text: str) -> list[float] | None:
        """取向量；命中记 ``hits``、未命中记 ``misses``."""
        value = self._entries.get(self.key(text))
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        return list(value)

    def put(self, text: str, vector: Iterable[float]) -> str:
        """存一个向量并返回它的键.

        维度与本缓存声明的身份不符时抛 ``EncodingError``：
        那说明"正在写的向量"与"这份缓存的键所声明的身份"不是一回事，
        混进去之后同一次查询里会出现两种长度的向量
        ——**那是 day064 的维度护栏要拦的事，本层要更早拦住它。**
        """
        values = [float(x) for x in vector]
        if self._dimension and len(values) != self._dimension:
            raise EncodingError(
                f"写入缓存的向量是 {len(values)} 维，而这份缓存声明的是 "
                f"{self._dimension} 维（{self._provider}/{self._model}）。"
                "维度不同说明当前编码器与缓存的键所声明的身份不是同一个——"
                "请用当前的编码器身份新建一份缓存，不要复用旧的。"
            )
        stored = self.key(text)
        self._entries[stored] = values
        return stored

    def put_vectors(self, pairs: Iterable[tuple[str, Iterable[float]]]) -> int:
        """批量存（返回存进去的条数；任一条维度不符即整体拒绝）."""
        items = [(text, [float(x) for x in vector]) for text, vector in pairs]
        for text, values in items:
            if self._dimension and len(values) != self._dimension:
                raise EncodingError(
                    f"批量写入里有一条 {len(values)} 维的向量，与缓存声明的 "
                    f"{self._dimension} 维不符（文本前 20 字：{text[:20]!r}）。"
                    "整批拒绝而不是跳过：部分写入会让'缓存里有什么'变得不可预测。"
                )
        for text, values in items:
            self._entries[self.key(text)] = values
        return len(items)

    # ------------------------------------------------------------------ 统计

    @property
    def size(self) -> int:
        """条目数."""
        return len(self._entries)

    @property
    def hits(self) -> int:
        """累计命中次数."""
        return self._hits

    @property
    def misses(self) -> int:
        """累计未命中次数."""
        return self._misses

    def reset_counters(self) -> None:
        """把命中/未命中计数清零（条目保留）.

        为什么单独提供它：一次构建的命中率是"这次省了多少"的度量，
        而缓存实例的生命周期通常比一次构建长。**不清零，
        第二次构建的命中率会被第一次的命中累加进去**——那是两个数字混在一起。
        """
        self._hits = 0
        self._misses = 0

    def keys(self) -> list[str]:
        """全部键（**升序**：让"这批键"成为一个可比较的对象）."""
        return sorted(self._entries)

    def stats(self) -> dict[str, Any]:
        """给报告与端点用的统计块."""
        total = self._hits + self._misses
        return {
            "provider": self._provider,
            "model": self._model,
            "dimension": self._dimension,
            "identity_key": self.identity_key,
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "lookups": total,
            "hit_ratio": round(self._hits / total, 4) if total else 0.0,
            "path": self._path,
        }

    # ------------------------------------------------------------------ 修剪

    def prune(self, keep_keys: Iterable[str]) -> int:
        """删掉不在 ``keep_keys`` 里的条目，返回删掉的条数.

        调用方给的是**当前清单里还需要的键**。
        没有自动清理的设计理由见模块 docstring 第 3 条。
        """
        keep = set(keep_keys)
        stale = [key for key in self._entries if key not in keep]
        for key in stale:
            del self._entries[key]
        return len(stale)

    def clear(self) -> None:
        """清空条目（计数一并清零——它们描述的正是这些条目）."""
        self._entries.clear()
        self.reset_counters()

    # ------------------------------------------------------------------ 落盘

    def persist(self, path: str | None = None) -> str:
        """把缓存写到 JSON，返回实际路径.

        缓存**可以随时删**（它只是省下重算的时间，不承载任何唯一信息），
        因此这里的实现刻意简单：一次全量写、没有增量日志、没有校验和。
        需要更强保证的是清单与备份，那是另外两个模块的事。
        """
        target = Path(path or self._path)
        if not str(target):
            raise ManifestError(
                "缓存没有落盘位置：构造时没给 path，调用 persist() 时也没给。"
                "纯内存缓存是合法用法，但那时请直接不落盘。"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": CACHE_VERSION,
            "provider": self._provider,
            "model": self._model,
            "dimension": self._dimension,
            "identity_key": self.identity_key,
            "count": self.size,
            "entries": {key: self._entries[key] for key in self.keys()},
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        self._path = str(target)
        return self._path

    def load(self, path: str | None = None) -> int:
        """从 JSON 读回缓存，返回载入的条目数.

        三条硬校验（**任何一条不过都拒读**）：

        ```text
        cache_version 不符         → 格式不匹配，读进来一半比读不进来更糟
        identity 不符              → 见模块 docstring 第 2 条
        某条向量维度与声明不符      → 文件被改过或写到一半
        ```
        """
        target = Path(path or self._path)
        if not str(target):
            raise ManifestError("缓存没有可读位置：构造与调用都没有给 path")
        if not target.exists():
            raise ManifestError(
                f"缓存文件不存在：{target}。"
                "首次构建时缓存本来就是空的——请确认这不是'路径写错了'。"
            )
        raw = json.loads(target.read_text(encoding="utf-8"))
        version = int(raw.get("cache_version", 0))
        if version != CACHE_VERSION:
            raise ManifestError(
                f"缓存文件格式版本是 {version}，本代码只认识 {CACHE_VERSION}：拒读"
            )
        file_provider = str(raw.get("provider", ""))
        file_model = str(raw.get("model", ""))
        file_dimension = int(raw.get("dimension", 0))
        if (file_provider, file_model, file_dimension) != (
            self._provider,
            self._model,
            self._dimension,
        ):
            raise ManifestError(
                f"缓存身份不匹配：文件是 {file_provider}/{file_model}"
                f"（{file_dimension}d），本实例是 {self._provider}/{self._model}"
                f"（{self._dimension}d）。"
                "换了编码器就新建一份缓存——旧条目永远不会命中（键里含身份），"
                "留着只会让人误以为'缓存可以跨模型共享'。"
            )
        entries = raw.get("entries", {})
        if not isinstance(entries, dict):
            raise ManifestError("缓存文件的 entries 必须是对象（键 → 向量）")
        loaded: dict[str, list[float]] = {}
        for key, values in entries.items():
            if len(str(key)) != VECTOR_KEY_LENGTH:
                raise ManifestError(
                    f"缓存里有一个 {len(str(key))} 位的键（应为 {VECTOR_KEY_LENGTH} 位）"
                )
            vector = [float(x) for x in values]
            if self._dimension and len(vector) != self._dimension:
                raise ManifestError(
                    f"缓存条目 {key} 是 {len(vector)} 维，与声明的 {self._dimension} 维不符："
                    "文件可能被改过或写到一半。"
                )
            loaded[str(key)] = vector
        declared = raw.get("count")
        if declared is not None and int(declared) != len(loaded):
            raise ManifestError(
                f"缓存文件自相矛盾：声明的 count={declared}，实际有 {len(loaded)} 条"
            )
        self._entries = loaded
        self._path = str(target)
        return len(loaded)

    def describe(self) -> str:
        """人类可读的一行摘要."""
        where = self._path or "（内存）"
        return (
            f"缓存 {self._provider}/{self._model} ({self._dimension}d) | "
            f"{self.size} 条 | 命中 {self._hits} / 未命中 {self._misses} | {where}"
        )


__all__ = [
    "CACHE_VERSION",
    "EmbeddingCache",
]
