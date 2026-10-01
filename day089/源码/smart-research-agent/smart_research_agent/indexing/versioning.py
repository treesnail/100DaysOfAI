"""索引版本表：让"回滚到上一版"这句话里的"上一版"有一个确定的答案（M6-D4）.

day058 的模型版本管理论证过一条纪律：**没有血缘的版本表等于没有版本表**。
今天这句话在索引上有了一个更尖锐的形态——因为索引的版本比模型的版本
更容易被"时间顺序"这种看似自然的直觉带偏：

```text
v1 ─► v2（采纳） ─► v3（构建失败，从未采纳）
                    ▲
        按时间倒序，v3 是"最新的一版" → 回滚会回到 v3
        按血缘，从 current=v2 回溯      → 回滚回到 v1
```

**回滚到 v3 意味着切到一个从没生效过的版本**，而 v3 恰恰是那个失败的版本
（它之所以没被采纳，通常就是因为失败了）。更糟的是这个错误在报告里
完全看不出来：`current` 从 `v2` 变成一个合法的版本号，没有异常、
没有告警，只有下一次检索开始给出另一批结果。因此本模块把"回滚"实现成
**沿 ``parent_version`` 回溯**，而不是"按登记顺序取倒数第 N 个"，
并且把这件事写进这一条用例里（``test_rollback_follows_lineage_not_time_order``）。

## 为什么血缘是一等公民而不是装饰

``parent_version`` 是清单里唯一描述"关系"的字段。它有三个必须存在的理由：

```text
回滚       "上一版"必须只有一个答案（见上）
diff       两版的差集要回答"从哪一版到哪一版"，不是"两个随机的版本"
审计       版本表要能回答"这一版是从哪一版长出来的"，而不只是"它是什么时候来的"
```

## 放弃了什么（代价写在明面上）

| 放弃的东西 | 代价 | 为什么可以接受 |
|-----------|------|---------------|
| 按时间回滚 | "最新的一版"不再等于"回滚目标" | 那正是本模块要修的错（见上） |
| 自动去读盘 | 构造后 ``latest()`` 是 ``None`` | 悄悄用旧指针回滚是同一类错误 |
| 就地改写历史 | ``persist`` 覆盖整份文件 | 追加只失败最后一行，历史仍可读 |
| 树形血缘 | 只有单亲 ``parent_version`` | 索引是"一次构建接一次构建"，不是分支合并 |
| 历史过大 | 全部清单常驻内存 | 本课规模是几十版；上千版时先炸的是清单体积 |

## JSONL 与 ``current.txt``

```text
manifest_history.jsonl   一行一个清单（追加语义：加一版 = 写一行）
current.txt              一个版本号（被采纳的那一版；空文件表示还没采纳过）
```

为什么 ``current`` 单独放一个文件而不是写进 JSONL：它是**状态**而不是**事件**。
混进历史里就必须回答"最后一行是不是指针"，而任何一次写失败都会让这个问题
有两种答案。分开之后，两者的失败模式互不牵连：历史丢一行只是少一版，
指针丢了只是"还没采纳过"。

## 谁依赖它

```text
builder（装配阶段）   构建成功 → register；构建失败 → 什么都不登记（这就是 v3 的由来）
运维 / 端点           history() 打印版本表、diff() 说明两版之间变了什么、
                     rollback() 回到一个**生效过**的版本
tests/               全部断言在 tests/test_indexing_versioning.py
```
"""

from __future__ import annotations

import json
from pathlib import Path

from smart_research_agent.indexing.errors import ManifestError, VersionError
from smart_research_agent.indexing.manifest import MANIFEST_HISTORY_FILE
from smart_research_agent.indexing.types import IndexManifest, IndexPlan

#: 记录"被采纳的版本"的指针文件名。空内容表示还没有采纳过任何版本。
CURRENT_FILE = "current.txt"


class IndexVersionStore:
    """索引版本表：登记、查询、血缘回溯、差集与回滚.

    它**不构建任何东西**，也不碰向量库——只管理"清单的版本"。
    这条边界与 ``cache`` 不关心向量怎么来是同一个设计：
    版本表可以被单独测试（登记三版、回滚两步、diff 两版），
    全程不需要一个真实的 embedding 提供方或向量后端。

    ``path`` 是**目录**（``manifest_history.jsonl`` 与 ``current.txt`` 写在它下面）。
    构造时不会去读盘（见模块 docstring 的取舍表）：要用磁盘上的历史请显式 ``load()``。
    """

    def __init__(self, *, path: str = "") -> None:
        self._path = str(path)
        self._manifests: dict[str, IndexManifest] = {}
        self._current = ""

    # ------------------------------------------------------------------ 元信息

    @property
    def path(self) -> str:
        """历史所在目录（空串表示只在内存里）."""
        return self._path

    @property
    def current(self) -> str:
        """被"采纳"的版本号；空串表示还没采纳过任何版本.

        为什么"空串"而不是 ``None``：它是一个版本号的位置，
        而"没有版本号"与"版本号是空串"在本模块看来是同一件事
        （都可以直接进报告，不需要调用方分支处理）。
        """
        return self._current

    def __len__(self) -> int:
        """已登记的版本数."""
        return len(self._manifests)

    # ------------------------------------------------------------------ 登记

    def register(self, manifest: IndexManifest) -> str:
        """登记一版，返回它的 ``version_id``；同一个版本重复登记是**幂等**的.

        为什么先 ``verify_version_id()``：清单是一份可以被手工编辑的 JSON，
        而"版本号与内容不符"的清单一旦进表，``diff`` 与 ``rollback``
        都会基于一份假的"上一版"工作——错误会以"回滚后检索变差"的形式出现，
        没有任何异常指向"那一版是假的"。校验放在入口，版本表里就只有真清单。

        重复登记为什么不需要逐字段比较：``version_id`` 由
        (编码器身份 + 度量 + 后端 + 内容摘要) 算出（见 ``build_manifest``），
        因此**相同版本号必然同内容**。再比一遍字段只会让人误以为
        "版本号相同但内容可能不同"，而那件事在本模块里不存在。
        """
        version_id = manifest.verify_version_id()
        if version_id in self._manifests:
            return version_id
        self._manifests[version_id] = manifest
        return version_id

    def get(self, version_id: str) -> IndexManifest | None:
        """取一版；不存在返回 ``None``（查询不到是正常结果，不是异常）."""
        return self._manifests.get(str(version_id))

    def latest(self) -> IndexManifest | None:
        """最近**登记**的一版；一版都没有时返回 ``None``.

        注意它说的是"最近登记的"而不是"当前生效的"（后者是 ``current``）：
        这两者在本模块里刻意分开，因为"最新的"与"已采纳的"不是同一件事——
        两者合一的实现会让下面那个反例（v3 未采纳）无法被表达。

        ``None`` 而不是异常：**没有历史是一个合法状态**（第一次构建之前就是这样）。
        """
        if not self._manifests:
            return None
        return next(reversed(self._manifests.values()))

    def history(self) -> list[IndexManifest]:
        """全部版本，**按登记顺序**（旧 → 新）.

        登记顺序而不是版本号排序：版本号是内容摘要，它的字典序没有任何含义，
        按它排出来的"历史"不是时间线。
        """
        return list(self._manifests.values())

    # ------------------------------------------------------------------ 血缘

    def lineage(self, version_id: str | None = None) -> list[IndexManifest]:
        """沿 ``parent_version`` 回溯到首版，返回**旧 → 新**的一条血缘.

        ``version_id`` 缺省取 ``current``：回滚问的永远是"从**生效的**那一版
        往回看"，而不是"从最新登记的那一版往回看"（见模块 docstring 的反例）。

        血缘断裂（父版本不在表里）与成环都抛 ``VersionError`` 而不是"能走多远走多远"：
        一条被截断的血缘会让 ``rollback`` 在一个**错误的位置**停下来，
        而它的表现只是"回滚之后换了个版本"——正是本模块要消灭的那类静默错误。
        """
        start = str(version_id) if version_id is not None else self._current
        if not start:
            raise VersionError(
                "还没有采纳任何版本（current 为空），因此无法从'当前版本'回溯血缘："
                "请先 adopt() 一个版本，或显式传入 version_id。"
            )
        chain: list[IndexManifest] = []
        seen: set[str] = set()
        cursor = start
        while cursor:
            if cursor in seen:
                raise VersionError(
                    f"血缘里出现环：{cursor!r} 的父版本链回到了它自己"
                    f"（已走过 {len(chain)} 版）。"
                    "环会让'上一版'有无穷多个答案，请修掉其中一条 parent_version。"
                )
            seen.add(cursor)
            manifest = self._manifests.get(cursor)
            if manifest is None:
                raise VersionError(
                    f"血缘在 {cursor!r} 处断裂：它不在版本表里"
                    f"（已登记 {len(self._manifests)} 版）。"
                    "没有血缘的版本表等于没有版本表——'上一版'必须有一个确定的答案，"
                    "请先把缺失的那一版补登记，或清掉指向它的 parent_version。"
                )
            chain.append(manifest)
            cursor = manifest.parent_version
        chain.reverse()
        return chain

    # ------------------------------------------------------------------ 差集

    def diff(self, left_id: str, right_id: str) -> IndexPlan:
        """两版之间的差集（``left`` → ``right`` 要做什么），用 ``IndexPlan`` 表达.

        ```text
        added       right 有、left 没有
        removed     left 有、right 没有
        updated     两边都有，但 fingerprint 或 vector_key 变了
        unchanged   两边都有，且两个字段都没变   （**未变 = 两边都有的条目**）
        ```

        "更新"的判据与 ``planner.plan_index`` 逐条一致（``fingerprint`` 或
        ``vector_key`` 变），因此两份报告的"需要重算多少条"可以直接对照。
        本模块**刻意不 import planner**：两者同刻并行开发，而这条判据只有一行——
        为它建立一条导入边，会让"只想看版本差集"的调用方也被 planner 的依赖拖住。
        将来若要抽公共函数，`diff` 是应当改指过去的那个调用点。

        ``reason`` 写成 ``"<left> → <right>"``（两个版本号）而不是"第 2 版到第 3 版"：
        版本号才是唯一标识，序号会随"哪个表里数"而变化。
        """
        left = self._require(left_id)
        right = self._require(right_id)
        left_map = left.entry_map()
        right_map = right.entry_map()
        added = tuple(sorted(set(right_map) - set(left_map)))
        removed = tuple(sorted(set(left_map) - set(right_map)))
        updated: list[str] = []
        unchanged: list[str] = []
        for record_id in sorted(set(left_map) & set(right_map)):
            before = left_map[record_id]
            after = right_map[record_id]
            if (
                before.fingerprint != after.fingerprint
                or before.vector_key != after.vector_key
            ):
                updated.append(record_id)
            else:
                unchanged.append(record_id)
        return IndexPlan(
            added=added,
            updated=tuple(updated),
            removed=removed,
            unchanged=tuple(unchanged),
            reason=f"{left.version_id} → {right.version_id}",
        )

    # ------------------------------------------------------------------ 采纳与回滚

    def adopt(self, version_id: str) -> str:
        """把某一版标为"当前生效的"，返回它（不存在则 ``VersionError``）.

        "登记"与"采纳"分开是刻意的：登记的版本可能从未生效过
        （构建失败、验证不过、被人工否决）。把两者合成一个动作，
        就等于宣布"写进版本表的都生效过"——而那是错的。
        """
        manifest = self._require(version_id)
        self._current = manifest.version_id
        return self._current

    def rollback(self, *, steps: int = 1) -> str:
        """沿血缘回到 ``steps`` 步之前，返回新的 ``current``.

        ``steps`` 数的是**血缘上的步数**，不是登记顺序上的步数（见模块 docstring）：
        ``lineage(current)`` 的最后一版就是 ``current``，因此
        ``steps=1`` 取的是 ``chain[-2]``，也就是"紧接 current 之前的那个生效过的版本"。

        两种拒绝都是 ``VersionError``，且消息里必须带足定位信息：

        ```text
        steps < 1     "回退 0 步"不是一次回滚，负数没有意义 → 改调用
        血缘不够      消息里给出**当前版本与可用步数** → 让人知道能退几步
        ```

        血缘不够时**不**退而求其次地退回"血缘上能退到的最远那一版"：
        一次"想退 3 步但只退了 1 步"的回滚如果静默成功，调用方会以为
        自己在一个自己没打算去的版本上。
        """
        if steps < 1:
            raise VersionError(
                f"rollback 的 steps 必须 >= 1，收到 {steps}："
                "回退 0 步不是一次回滚，而回退负数没有意义。"
            )
        if not self._current:
            raise VersionError(
                "还没有采纳任何版本（current 为空），没有'上一版'可以回退——"
                "请先 adopt() 一个版本，或显式从历史里选一版。"
            )
        chain = self.lineage(self._current)
        if len(chain) <= steps:
            raise VersionError(
                f"血缘长度不够：当前版本 {self._current} 的血缘只有 {len(chain)} 版，"
                f"最多只能回退 {len(chain) - 1} 步，收到 steps={steps}。"
                "按血缘回溯是为了不回到一个从没生效过的版本——请改用更小的 steps。"
            )
        target = chain[-1 - steps]
        self._current = target.version_id
        return self._current

    # ------------------------------------------------------------------ 落盘

    def persist(self, path: str | None = None) -> str:
        """把版本表写成 JSONL + ``current.txt``，返回历史所在目录.

        写两份文件（见模块 docstring 的"JSONL 与 current.txt"）：
        一版一行的历史，与一个只有版本号的指针。两处都用覆盖式写入——
        历史文件的内容完全由内存里的表决定，因此"重写"与"追加"在结果上等价，
        而重写少一种失败模式（追加到一半的历史必须被读侧特殊处理）。
        """
        directory = _resolve_directory(path if path is not None else self._path)
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(manifest.to_dict(), ensure_ascii=False)
            for manifest in self._manifests.values()
        ]
        body = "".join(f"{line}\n" for line in lines)
        (directory / MANIFEST_HISTORY_FILE).write_text(body, encoding="utf-8")
        (directory / CURRENT_FILE).write_text(self._current, encoding="utf-8")
        self._path = str(directory)
        return self._path

    def load(self, path: str | None = None) -> int:
        """逐行读回版本表与 ``current`` 指针，返回**可用的版本数**.

        三道校验，任何一条不过都拒读：

        ```text
        文件不存在            → ManifestError（并说明"首次构建本来就没有历史"）
        某行不是合法 JSON      → ManifestError（并指出是第几行）
        manifest_version 不符 → ManifestError（由 IndexManifest.from_dict 抛）
        指针指向不存在的版本    → ManifestError（否则下一次回滚会回到一个幻影上）
        ```

        第二条值得说一句：JSONL 的常见卖点正是"一行坏掉不影响其它行"，
        这里**刻意放弃**那个特性。半份历史会让 ``rollback`` 在一个错误的
        位置停下来（中间少了一版，血缘就断了），而"读到一半的历史"
        与"完整历史"在返回值上只差一个数字。

        重复行（同一个 ``version_id`` 出现两次）按一版计：``register`` 是幂等的，
        磁盘上的重复只是"某次写入重试"留下的痕迹，不该让历史长度变化。

        ``load`` 会**整体替换**内存里的表（不是合并）：它的语义是
        "把版本表变成磁盘上那份状态"，合并会让内存里多出来的那些版本永远删不掉。
        """
        directory = _resolve_directory(path if path is not None else self._path)
        history_path = directory / MANIFEST_HISTORY_FILE
        if not history_path.exists():
            raise ManifestError(
                f"版本历史文件不存在：{history_path}。"
                "一版都没构建过时本来就没有历史文件——请确认这不是'路径写错了'，"
                "而不是把这份不存在的历史当成'没有任何版本'。"
            )
        loaded: dict[str, IndexManifest] = {}
        for number, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"版本历史第 {number} 行不是合法 JSON：{exc}。"
                    "拒读整份历史而不是跳过这一行：少一版会让血缘断裂，"
                    "而断裂的血缘会让回滚停在一个错误的位置。"
                ) from exc
            manifest = IndexManifest.from_dict(payload)
            manifest.verify_version_id()
            loaded.setdefault(manifest.version_id, manifest)

        pointer = directory / CURRENT_FILE
        current = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
        if current and current not in loaded:
            raise ManifestError(
                f"current 指针指向 {current!r}，但它不在这份历史里"
                f"（历史有 {len(loaded)} 版）。"
                "指针指向一个不存在的版本，会让下一次回滚以它为起点——"
                "请修复指针，或把那一版补回历史。"
            )
        self._manifests = loaded
        self._current = current
        self._path = str(directory)
        return len(loaded)

    # ------------------------------------------------------------------ 内部

    def _require(self, version_id: str) -> IndexManifest:
        """取一版，取不到就抛 ``VersionError``（并列出可用的版本号）."""
        manifest = self._manifests.get(str(version_id))
        if manifest is None:
            known = ", ".join(sorted(self._manifests)) or "（空表）"
            raise VersionError(
                f"版本 {version_id!r} 不在版本表里（已登记 {len(self._manifests)} 版："
                f"{known}）。可用的版本号只能从 history() 里拿——不要凭记忆拼一个 id。"
            )
        return manifest


def _resolve_directory(path: str) -> Path:
    """把 ``path`` 收敛成"历史所在目录"（也接受直接给历史文件路径）.

    接受 ``manifest_history.jsonl`` 这个写法，是因为调用方的手上常常只有
    "上次那个文件在哪"。其余一律按目录处理：本模块写的是两个文件，
    把一个目录当成文件、或反过来，都会让清单落到一个没人会去看的地方。
    """
    raw = str(path)
    if not raw.strip():
        raise ManifestError(
            "版本历史需要一个目录：persist / load 时路径是必需的——"
            "空路径会让历史落到进程的当前目录，而调用方以为它写在别处。"
        )
    target = Path(raw)
    if target.name == MANIFEST_HISTORY_FILE:
        return target.parent
    return target


__all__ = [
    "CURRENT_FILE",
    "IndexVersionStore",
]
