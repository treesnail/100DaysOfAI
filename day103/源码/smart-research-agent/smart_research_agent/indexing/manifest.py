"""索引清单：把"上一版索引长什么样"写成一份**能被任何人重算**的证据（M6-D4）.

day064 交出来的向量库只回答"现在有什么"，答不了"上一版有什么"。
今天补上的这一层最容易被低估——它的主体是一个 dataclass
（``types.IndexManifest``）与两次 JSON 读写，看起来不像有多少设计余地。

真正的设计全在三个问题上：

```text
1. 清单凭什么可信？       → 版本号由 (身份 + 度量 + 后端 + 内容摘要) 算出，
                            读回来时重算一遍（verify_version_id），对不上就拒读
2. 清单与库不一致怎么办？   → compare_with_store / verify_index 把不一致变成一个
                            当场就能看见的事件，而不是等增量更新算错差集
3. 库里没有原文怎么办？     → 用 metadata["fingerprint"]（day062 的内容身份）；
                            两者都没有就报错，**不许猜**
```

## 为什么"重算"比"签名"更适合这份文件

清单会被谁改？最常见的是人：手工修一个 id、把两天的构建结果拼在一起、
用编辑器打开 JSON 顺手格式化一下。这些改动都不会伪造签名，但一定会在重算时露出来。
因此本模块不引入任何密钥或签名机制——**版本号本身就是校验和**，
而"任何人拿着内容都能把版本号算出来"这件事在三处都要用到：
版本表要判断"这两份清单是不是同一版"、备份要判断"这份快照属于哪一版"、
报告要能被人独立核对。

## 放弃了什么（代价写在明面上）

| 放弃的东西 | 代价 | 为什么可以接受 |
|-----------|------|---------------|
| 增量读取 | ``manifest_from_store`` 要遍历整库 | 它的场景是"重建一次清单"，不是每次构建 |
| 从记录建清单 | 没有 ``manifest_from_records``（见下） | 那条路要先用 planner，规则只该有一份 |
| 存原文 | 清单里只有 id 与两个指纹 | 清单要进版本表；十万条原文会让它没法被打开 |
| 覆盖式写入 | 一次重写整个文件 | 只有几十 KB，且"要么完整要么不存在"更好解释 |

## 为什么这里没有 ``manifest_from_records``

"从一批知识库记录建清单"要先用 ``planner.entry_from_record`` 把记录变成
``IndexEntry``（"该编码哪段文本"与"内容身份从哪来"的完整规则在那里）。
把这件事放进本模块，等于让清单同时对"记录的形状"和"条目的形状"负责；
而装配阶段的 ``builder`` 本来就要同时用到 planner 与 manifest——
让它一次做完，规则就只有一份。

## 谁依赖它

```text
versioning.IndexVersionStore   登记 / 回滚 / diff 都建立在"版本号可重算"上
backup.IndexBackupStore        C 号按同一份 JSON 形状落盘备份里的 manifest.json
builder（装配阶段）             build → 写清单 → 登记 → 按清单算下一版差集
builder.verify / 端点           verify_index 的 problems 直接就是给运维看的结论
tests/                          全部断言在 tests/test_indexing_manifest.py
```
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smart_research_agent.documents.types import content_id
from smart_research_agent.indexing.errors import ManifestError
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexEntry,
    IndexManifest,
    entries_digest,
    index_version_id,
    vector_key,
)
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.pipeline import embedding_text, record_id_of
from smart_research_agent.vectorstore.types import VectorRecord

#: 清单文件名。取固定名而不是让调用方每次起一个：
#: "这个目录里哪份是清单"必须有一个不问自明的答案（备份与演示脚本都依赖它）。
MANIFEST_FILE = "manifest.json"

#: 版本历史的文件名（JSONL，一行一版）。由 ``versioning`` 使用，
#: 定义放在这里是为了让"清单相关的文件名"只有一处定义。
MANIFEST_HISTORY_FILE = "manifest_history.jsonl"


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #


def build_manifest(
    entries: Sequence[IndexEntry],
    *,
    identity: EmbeddingIdentity,
    metric: str,
    backend: str,
    parent_version: str = "",
    created_at: str = "",
    metadata: dict[str, str] | None = None,
) -> IndexManifest:
    """由一批条目造一份清单，**版本号在这里算**（调用方不传 ``version_id``）.

    为什么不让调用方传 ``version_id``：那是把"版本号等于内容"这条纪律
    交给每一个调用点去遵守。只要有一个调用点算错（或图省事传了个时间戳），
    版本表里就会出现**两个内容相同、版本号不同**的版本，
    而它在报告里只表现为"凭空多出一版"。这里是唯一的算出它的地方。

    ``parent_version`` / ``created_at`` / ``metadata`` 都不参与版本号：

    ```text
    parent_version   血缘是"谁接在谁之后"，与"这一版是什么"无关
    created_at       时间戳进版本号 → 同一份内容在两台机器上得到两个版本号
    metadata         给人看的备注（构建耗时、触发人等），不是身份
    ```

    条目顺序不影响结果：``entries_digest`` 内部排序，``IndexManifest`` 也会把
    条目收敛成"按 id 升序且不重复"。因此**同一份内容按任何顺序传进来
    都得到同一个版本号**（这条有专门用例）。
    """
    ordered = tuple(entries)
    version_id = index_version_id(
        identity_key=identity.key,
        metric=metric,
        backend=backend,
        digest=entries_digest(ordered),
    )
    return IndexManifest(
        version_id=version_id,
        identity=identity,
        metric=metric,
        backend=backend,
        entries=ordered,
        parent_version=parent_version,
        created_at=created_at,
        metadata=dict(metadata or {}),
    )


def manifest_from_store(
    backend: VectorBackend,
    *,
    identity: EmbeddingIdentity,
    metric: str | None = None,
    parent_version: str = "",
    created_at: str = "",
) -> IndexManifest:
    """从库里的记录**重建**一份清单（每条的 ``fingerprint`` 与 ``vector_key`` 现算）.

    使用场景是"清单丢了，但库还在"——这是真实会发生的事：清单是一份小文件，
    它可能被误删、被回滚到旧版本、或在迁移时没跟着走。此时唯一能做的补救是
    **按库里的现状重新算一份清单**，代价是这一版与真正的那一版可能不同
    （原文不在了、``retrieval_text`` 不在 metadata 里，都会让算出来的指纹与当初不同），
    而这一点必须让调用方知道：**重建出来的清单是"从现在起"的基线，不是历史的那一版。**

    两个字段各自从哪里来（这是本函数唯一需要判断的地方）：

    ```text
    fingerprint   metadata["fingerprint"]（day062 的内容身份，最可信）
                  否则 content_id(原文) 现算（原文在库里，也算得出来）
                  两者都没有（text 为空且没有 fingerprint）→ ManifestError，**不许猜**
    vector_key   vector_key(身份, 编码文本)：编码文本走 vectorstore.pipeline.embedding_text，
                  因此"该编码哪段文本"的规则与 day064 的摄取路径是同一条
    ```

    为什么 ``fingerprint`` 用**原文**而 ``vector_key`` 用**编码文本**（通常是
    ``retrieval_text`` = 面包屑 + 原文）：这两个字段回答的是两个问题——
    "内容变了没有"与"该不该复用向量"。用同一段文本去算它们，
    会让"只改了标题面包屑"被误判成内容变化（多花一次编码），
    或者让"内容变了但检索视图恰好没变"被漏判。

    ``token_count`` 从 ``metadata["token_count"]`` 尽力取，取不到或取不出整数时记 0：
    它只用于成本估算（见 ``types.entries_digest``：统计字段不进摘要），
    一个脏的统计值不该让整份清单建不出来。
    """
    entries: list[IndexEntry] = []
    for record_id in backend.ids():
        record = backend.get(record_id)
        if record is None:
            raise ManifestError(
                f"库自相矛盾：``ids()`` 里有 {record_id!r}，但 ``get()`` 取不到它。"
                "向量索引与记录表不同步（多见于删除只做了一半）——"
                "请先用 ``vectorstore.evaluate.index_health`` 体检，"
                "不要在这份不一致的库上重建清单："
                "重建出来的清单会把'索引里有、记录表里没有'当成正常。"
            )
        entries.append(_entry_from_record(record, identity))
    return build_manifest(
        entries,
        identity=identity,
        metric=metric or backend.metric,
        backend=backend.name,
        parent_version=parent_version,
        created_at=created_at,
    )


# --------------------------------------------------------------------------- #
# 与库对账
# --------------------------------------------------------------------------- #


def compare_with_store(manifest: IndexManifest, backend: VectorBackend) -> dict[str, Any]:
    """把清单与库逐项对照，返回**可读、可 json 序列化**的字典.

    这是本课的检查点（另一个是装配阶段的 ``builder.verify``）。它存在的理由与
    ``errors.ManifestError`` 的 docstring 是同一个：清单与库可以互相矛盾，
    而矛盾**不会在检索时抛异常**，只会让增量更新算出错误的差集。

    字段分成两组：

    ```text
    差集     missing_in_store / orphan_in_store        谁多谁少（都是升序 id 列表）
    口径     dimension_match / metric_match / backend_match   这份清单描述的是不是这个库
    对账     count_manifest / count_store / extra_tokens      两边的规模与 token 合计
    ```

    ``extra_tokens`` 返回三键（``manifest`` / ``store`` / ``diff``）而**不是一个合计值**：
    "两边的 token 合计"要对账就必须能看出差额是哪一边多出来的，
    一个合并后的数字只能在"对不上"之后再拆一次。
    ``tokens_match`` 单独给出，但**它不参与 ``verify_index`` 的判定**——
    ``token_count`` 不进内容摘要（见 ``types.entries_digest``），
    改一个统计字段不需要重建索引，把它当成问题会让检查结果过度敏感。

    为什么差集只比 id、不比 vector_key：那一层判断属于 ``planner.plan_index``
    （它要分"内容变了"与"该不该复用向量"）。本函数回答的是更粗也更硬的问题——
    **"清单与库说的是不是同一批记录"**。两者分开，报告里的结论才不会混在一起。
    """
    store_ids = set(backend.ids())
    manifest_ids = set(manifest.record_ids)
    missing = sorted(manifest_ids - store_ids)
    orphan = sorted(store_ids - manifest_ids)
    store_tokens = _store_token_total(backend, sorted(store_ids))
    count_manifest = manifest.count
    count_store = backend.count()
    return {
        "missing_in_store": missing,
        "orphan_in_store": orphan,
        "count_manifest": count_manifest,
        "count_store": count_store,
        "dimension_match": backend.dimension == manifest.identity.dimension,
        "metric_match": backend.metric == manifest.metric,
        "backend_match": backend.name == manifest.backend,
        "extra_tokens": {
            "manifest": manifest.total_tokens,
            "store": store_tokens,
            "diff": store_tokens - manifest.total_tokens,
        },
        "tokens_match": store_tokens == manifest.total_tokens,
    }


def verify_index(manifest: IndexManifest, backend: VectorBackend) -> dict[str, Any]:
    """把 ``compare_with_store`` 的结果收敛成一个"能不能用"的判断.

    返回 ``{"ok": bool, "checks": {...}, "problems": [...]}``：

    ```text
    checks    逐项结论（差集两项是 id 列表，口径三项是布尔）
    problems  每条都是一句能直接照做的中文结论（含具体 id 与出路）
    ok        四类问题一个都没有：清单有库里没有 / 库里有清单没有 /
              维度不符 / 口径不符（度量或后端）
    ```

    ``problems`` 必须是**句子的形式**而不是错误码：读它的人是运维，
    而"库里有 2 条记录不在清单里：['a1b2…', '0f2c…']——它们不会被增量
    更新覆盖，请先重建清单"这一句里已经包含了现象、数量与下一步。

    为什么 "清单说维度是 384、库是 768" 要让 ``ok`` 为假：它意味着这份清单
    描述的是**另一个编码器建的库**，拿它去算差集会得到一份看起来正常、
    实际上完全不相干的计划（所有 ``vector_key`` 都不同 → 全部重算，
    于是"清单与库对不上"这件事被伪装成"数据全变了"）。
    """
    comparison = compare_with_store(manifest, backend)
    missing = list(comparison["missing_in_store"])
    orphan = list(comparison["orphan_in_store"])
    dimension_match = bool(comparison["dimension_match"])
    metric_match = bool(comparison["metric_match"])
    backend_match = bool(comparison["backend_match"])

    problems: list[str] = []
    if missing:
        problems.append(
            f"清单里有 {len(missing)} 条记录在库里找不到：{_ids_preview(missing)}——"
            "上一次构建可能半途失败，或有人手工删过记录；"
            "请先重建清单（或用这些 id 补回向量），不要在这份清单上做增量。"
        )
    if orphan:
        problems.append(
            f"库里有 {len(orphan)} 条记录不在清单里：{_ids_preview(orphan)}——"
            "它们不会被增量更新覆盖，请先重建清单。"
        )
    if not dimension_match:
        problems.append(
            f"维度不一致：清单记的是 {manifest.identity.dimension} 维，"
            f"库里是 {backend.dimension} 维——这份清单描述的是另一个编码器建的库，"
            "不能用它驱动增量更新（所有 vector_key 都会变，"
            "'库与清单对不上'会被伪装成'数据全变了'）。"
        )
    if not metric_match:
        problems.append(
            f"度量不一致：清单记的是 {manifest.metric!r}，库是 {backend.metric!r}——"
            "同一批向量在两种度量下的最近邻不是同一批；"
            f"请用清单的度量新建实例（metric={manifest.metric!r}），"
            "而不是让一个已有实例改度量。"
        )
    if not backend_match:
        problems.append(
            f"后端不一致：清单记的是 {manifest.backend!r}，库是 {backend.name!r}——"
            "换后端可以重建，但不能与旧版共用一份清单，请为新后端单独记一份。"
        )

    return {
        "ok": not problems,
        "checks": {
            "missing_in_store": missing,
            "orphan_in_store": orphan,
            "dimension_match": dimension_match,
            "metric_match": metric_match,
            "backend_match": backend_match,
        },
        "problems": problems,
    }


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #


def write_manifest(manifest: IndexManifest, path: str) -> str:
    """把清单写成 JSON（UTF-8、``ensure_ascii=False``、``indent=2``），返回实际路径.

    三个刻意的决定：

    - **写之前先 ``verify_version_id()``**：一份内容被改过、版本号没跟着改的清单
      如果能被写进磁盘，它会在下一次构建时被当成上一版而当场炸掉——
      把校验放在写的一侧，磁盘上就永远不会有"读不回来的清单"；
    - ``indent=2`` + ``ensure_ascii=False``：清单是给人看与给人核对的东西，
      一行的 10 MB JSON 与一个 `\\uXXXX` 满天飞的文件都做不到这件事；
    - ``path`` **既接受目录也接受文件名**（见 ``_resolve_manifest_path``）：
      调用方写 ``write_manifest(m, settings.indexing_dir)`` 与
      ``write_manifest(m, "…/manifest.json")`` 时的意图都是明确的，
      让其中一种静默地写到别处（例如把清单写成一个叫 ``data/index`` 的文件）
      是这一层最不该有的故障。
    """
    manifest.verify_version_id()
    target = _resolve_manifest_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(target)


def read_manifest(path: str) -> IndexManifest:
    """读回清单并**重算版本号**（``verify_version_id`` 是这里唯一的校验器）.

    三道可能失败的关卡，各挡一种"安静地用错清单"：

    ```text
    文件不存在        → ManifestError（并说明"首次构建本来就没有上一版"）
    不是合法 JSON     → ManifestError（半截写入的文件必须当场拒读）
    版本号与内容不符   → ManifestError（由 verify_version_id 抛）
    ```

    第二道容易被省掉：``json.loads`` 抛出的 ``JSONDecodeError`` 是 ``ValueError``
    的子类，与 ``IndexingError`` 是同一个家族（``IndexingError`` 也继承 ``ValueError``），
    于是"忘了包一层"的代码在调用方那里看起来是**同一种失败**。
    这里显式转成 ``ManifestError``，是为了让"文件坏了"与"清单被改过"
    在报告里是两个可以分别处理的结论。
    """
    target = _resolve_manifest_path(path)
    if not target.exists():
        raise ManifestError(
            f"清单文件不存在：{target}。"
            "首次构建时本来就没有上一版清单——请确认这不是'路径写错了'，"
            "而不是把一个空库当成'上一版什么都没变'。"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"清单文件不是合法的 JSON：{target}（{exc}）。"
            "写到一半的文件必须当场拒读：把它当成'上一版'会让差集全部算错。"
        ) from exc
    if not isinstance(payload, dict):
        raise ManifestError(
            f"清单文件的顶层必须是对象，收到 {type(payload).__name__}：{target}"
        )
    manifest = IndexManifest.from_dict(payload)
    manifest.verify_version_id()
    return manifest


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _entry_from_record(record: VectorRecord, identity: EmbeddingIdentity) -> IndexEntry:
    """把一条库记录折成清单条目（``manifest_from_store`` 的单步）.

    ``embedding_text`` 需要一个"记录形状"的字典，因此这里把 ``VectorRecord``
    摊平成 ``pipeline`` 认识的三个键（``doc_id`` / ``text`` / ``metadata``）——
    **不是**为了转格式而转格式：这样"该编码哪段文本"这件事就只有
    ``vectorstore.pipeline`` 一处定义，索引层不会悄悄长出第二套规则。
    """
    payload: dict[str, Any] = {
        "doc_id": record.record_id,
        "text": record.text,
        "metadata": dict(record.metadata),
    }
    fingerprint = _fingerprint_of(record, payload)
    text = embedding_text(payload)
    return IndexEntry(
        record_id=record_id_of(payload),
        fingerprint=fingerprint,
        vector_key=vector_key(
            identity.provider, identity.model, identity.dimension, text
        ),
        token_count=_token_count(record.metadata),
    )


def _fingerprint_of(record: VectorRecord, payload: dict[str, Any]) -> str:
    """决定这一条的内容身份（见 ``manifest_from_store`` 的取舍说明）."""
    declared = record.metadata.get("fingerprint")
    if declared is not None and str(declared).strip():
        return str(declared).strip()
    if record.text.strip():
        return content_id(record.text)
    raise ManifestError(
        f"记录 {record.record_id!r} 无法重建内容身份：原文为空，"
        'metadata 里也没有 "fingerprint"。'
        "这条记录无法回答'它的内容变了没有'，因此不能进清单——"
        "请从 chunk 记录重建清单（day062 的 knowledge_records 里带着 fingerprint），"
        "而不是凭一个空字符串算一个指纹出来。"
    )


def _token_count(metadata: dict[str, Any]) -> int:
    """从 metadata 里尽力取 token 数；取不出整数时记 0（见 ``manifest_from_store``）."""
    raw = metadata.get("token_count", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _store_token_total(backend: VectorBackend, record_ids: Sequence[str]) -> int:
    """库里全部记录的 token 数之和（对账用；取不出的记录按 0 计）."""
    return sum(
        _token_count(record.metadata) for record in backend.get_many(list(record_ids))
    )


def _ids_preview(ids: Sequence[str], limit: int = 3) -> str:
    """把一批 id 写成一行可读的预览（``problems`` 里用的就是它）.

    只列前 ``limit`` 个并在超长时补 ``…``：``problems`` 是要被人读完的，
    一份 1200 条的差集把整列贴进去，真正的结论会被挤到看不见的地方。
    数量由调用方写在那句话的开头（"库里有 2 条记录不在清单里"），
    因此截断不会让人误以为"只有 3 条"。
    """
    shown = ", ".join(repr(str(item)) for item in list(ids)[:limit])
    suffix = ", …" if len(ids) > limit else ""
    return f"[{shown}{suffix}]"


def _resolve_manifest_path(path: str) -> Path:
    """把 ``path`` 收敛成"清单文件"的路径（目录写法与文件写法都接受）.

    判据按优先级排，且**每一档都有明确的理由**：

    ```text
    空路径             → ManifestError（落到进程当前目录是最难查的一类"写错地方"）
    以分隔符结尾       → 目录
    已存在且是目录     → 目录
    不存在且没有后缀   → 目录（``data/index`` 这种写法只有一个合理解释）
    其余               → 文件（``manifest.json`` / ``custom.json``）
    ```

    最后一条是"默认按文件处理"：一个**已经存在**的无后缀路径，
    说明调用方当初是拿它当文件建的，此时改按目录处理会凭空多出一层。
    """
    raw = str(path)
    if not raw.strip():
        raise ManifestError(
            "清单需要一个路径：读写清单时路径是必需的——"
            "空路径会让 write_manifest 落到进程的当前目录，而调用方以为它写在别处。"
        )
    target = Path(raw)
    if raw.endswith(("/", "\\")) or target.is_dir():
        return target / MANIFEST_FILE
    if not target.exists() and not target.suffix:
        return target / MANIFEST_FILE
    return target


__all__ = [
    "MANIFEST_FILE",
    "MANIFEST_HISTORY_FILE",
    "build_manifest",
    "compare_with_store",
    "manifest_from_store",
    "read_manifest",
    "verify_index",
    "write_manifest",
]
