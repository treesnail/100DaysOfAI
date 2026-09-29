"""day058 版本注册表测试（M5-D9）：追加写事件日志、折叠、版本链、清理.

这个文件守三件"看起来能用、实际已经错了"的事：

1. **折叠必须确定**：同一份事件流无论读几次、在哪台机器上读，都必须得到
   同一个状态。它破了的表现是"``head()`` 有时返回 None"，而查的人根本
   不会想到是索引里有一行没读懂；
2. **幂等 ≠ 宽松**：``register`` 只在"三元组键与版本号都相同"时幂等
   （CI 重跑同一个任务）。把幂等放宽到"版本号相同就覆盖"会让索引里
   少掉一份真实产物，而**少掉的那一份不会有任何提示**；
3. **``prune`` 不能自断退路**：``stable`` 永不归档（它是回滚目标），
   ``head()`` 永不归档。把这两条破掉，第一次回滚演练就会发现"没有目标"。

事件流的**每一类损坏**都单独有用例：非法 JSON、非对象、未知事件类型、
``register`` 缺字段、``stage`` 指向不存在的版本。它们全是"静默丢状态"
的入口，而静默丢状态正是本模块最想杜绝的失败模式。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from smart_research_agent.registry import (
    STAGE_ARCHIVED,
    STAGE_CANDIDATE,
    STAGE_ROLLED_BACK,
    STAGE_STABLE,
    ModelRegistry,
    ModelVersion,
    RegistryError,
    VersionTriple,
)
from smart_research_agent.registry.store import (
    DEFAULT_KEEP_VERSIONS,
    EVENT_REGISTER,
    EVENT_STAGE,
    INDEX_FILENAME,
)

BASE = "Qwen3-8B"
BIG = "Qwen3-14B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"


def hexdigest64(seed: str) -> str:
    """由任意字符串派生一个 64 位十六进制串（测试用的确定性适配器哈希）."""
    raw = "".join(f"{byte:02x}" for byte in seed.encode("utf-8"))
    return (raw * 8)[:64]


def make(
    version: str,
    *,
    adapter: str | None = None,
    dataset: str = DATA_A,
    base: str = BASE,
    parent: str = "",
    stage: str = STAGE_CANDIDATE,
    rating: float | None = 0.6,
    artifacts: dict[str, str] | None = None,
) -> ModelVersion:
    """构造一条版本记录（默认产物齐全、有一份可比的评估分数）."""
    return ModelVersion(
        triple=VersionTriple(
            base_model=base,
            adapter_sha256=adapter or hexdigest64(version),
            dataset_fingerprint=dataset,
        ),
        version=version,
        parent_version=parent,
        stage=stage,
        metrics={} if rating is None else {"eval_pass_rate": rating},
        artifacts=(
            {"adapter": f"a/{version}", "merged": f"m/{version}"}
            if artifacts is None
            else artifacts
        ),
    )


def seed_registry() -> ModelRegistry:
    """一条最小的版本链：v1.0.0(stable) ← v1.0.1(candidate)."""
    registry = ModelRegistry()
    registry.register(make("1.0.0", stage=STAGE_STABLE))
    registry.register(make("1.0.1", parent="1.0.0"))
    return registry


# --------------------------------------------------------------------------- #
# 内存模式：增删查改
# --------------------------------------------------------------------------- #


def test_register_returns_the_stored_record() -> None:
    """``register`` 返回折叠后的记录；内存模式下它就是存进去的那一个。"""
    registry = ModelRegistry()
    record = registry.register(make("1.0.0"))
    assert registry.get("1.0.0") is record
    assert len(registry) == 1


def test_register_is_idempotent_for_same_key_and_version() -> None:
    """同键同版本 → 幂等返回已有记录（"CI 重跑同一个任务"的真实情形）."""
    registry = ModelRegistry()
    first = registry.register(make("1.0.0"))
    second = registry.register(make("1.0.0"))
    assert second is first
    assert len(registry) == 1


def test_register_rejects_same_key_with_other_version() -> None:
    """同一个三元组不能有两个版本号——否则"这是不是同一份产物"就没有唯一答案。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0"))
    # 同一个适配器哈希（= 同一个键），但换了一个版本号
    with pytest.raises(RegistryError, match="已被 1.0.0 占用"):
        registry.register(make("1.0.1", adapter=hexdigest64("1.0.0")))


def test_register_rejects_same_version_with_other_key() -> None:
    """版本号被别的产物占用必须报错：静默覆盖会让索引少掉一份真实产物。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0"))
    with pytest.raises(RegistryError, match="版本号 1.0.0 已属于键"):
        registry.register(make("1.0.0", adapter=hexdigest64("other")))


def test_register_rejects_dangling_parent() -> None:
    """悬空父指针会让回滚在走链时突然断掉，因此登记期就拒绝。"""
    registry = ModelRegistry()
    with pytest.raises(RegistryError, match="父版本 9.9.9 不存在"):
        registry.register(make("1.0.0", parent="9.9.9"))


def test_get_accepts_version_and_version_key() -> None:
    """既接受人写的版本号，也接受机器输出的版本键。"""
    registry = seed_registry()
    record = registry.get("1.0.0")
    assert registry.get(record.version_key) is record


def test_get_unknown_version_raises_and_find_returns_none() -> None:
    """``get`` 抛错、``find`` 返回 None——两种调用风格各给一个入口。"""
    registry = seed_registry()
    with pytest.raises(RegistryError, match="不存在"):
        registry.get("9.9.9")
    assert registry.find("9.9.9") is None
    assert "9.9.9" not in registry


def test_versions_are_sorted_numerically() -> None:
    """版本表按整数段升序（1.10.0 排在 1.9.0 之后）。"""
    registry = ModelRegistry()
    for version in ("1.10.0", "1.9.0", "1.0.0"):
        registry.register(make(version))
    assert [item.version for item in registry.versions()] == ["1.0.0", "1.9.0", "1.10.0"]


def test_versions_filter_by_stage_and_reject_unknown_stage() -> None:
    """阶段过滤：空集合法；阶段名写错才报错。"""
    registry = seed_registry()
    assert [item.version for item in registry.versions(stage=STAGE_STABLE)] == ["1.0.0"]
    assert registry.versions(stage=STAGE_ARCHIVED) == []
    with pytest.raises(RegistryError, match="未知阶段"):
        registry.versions(stage="production")


def test_latest_and_head_differ_when_candidate_is_newer() -> None:
    """``latest`` 看全部、``head`` 只看 stable——两者的差别正是"待上线"的候选。"""
    registry = seed_registry()
    assert registry.latest().version == "1.0.1"
    assert registry.head().version == "1.0.0"


def test_head_is_none_without_stable() -> None:
    """没有 stable 时 head 为 None（**不退回 candidate**）。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0"))
    assert registry.head() is None


def test_counts_always_contains_all_four_stages() -> None:
    """四个阶段键恒存在：``rolled_back: 0`` 与"没有这个键"读起来完全不同。"""
    registry = seed_registry()
    counts = registry.counts()
    assert counts == {
        STAGE_CANDIDATE: 1,
        STAGE_STABLE: 1,
        STAGE_ROLLED_BACK: 0,
        STAGE_ARCHIVED: 0,
        "total": 2,
    }


def test_iteration_yields_versions_in_order() -> None:
    """``__iter__`` 与 ``versions()`` 同序。"""
    registry = seed_registry()
    assert [item.version for item in registry] == ["1.0.0", "1.0.1"]


# --------------------------------------------------------------------------- #
# 阶段流转
# --------------------------------------------------------------------------- #


def test_set_stage_promotes_and_is_readable() -> None:
    """candidate → stable 之后 head 跟上，并记下理由。"""
    registry = seed_registry()
    updated = registry.set_stage("1.0.1", STAGE_STABLE, reason="门禁通过")
    assert updated.stage == STAGE_STABLE
    assert "门禁通过" in updated.notes
    assert registry.head().version == "1.0.1"


def test_set_stage_rejects_same_stage() -> None:
    """重复流转说明有两条代码路径在同时改状态，是要查的信号。"""
    registry = seed_registry()
    with pytest.raises(RegistryError, match="已经是 stable"):
        registry.set_stage("1.0.0", STAGE_STABLE)


@pytest.mark.parametrize(
    "source, target",
    [
        (STAGE_STABLE, STAGE_CANDIDATE),
        (STAGE_ROLLED_BACK, STAGE_STABLE),
        (STAGE_ARCHIVED, STAGE_STABLE),
    ],
)
def test_set_stage_rejects_disallowed_transitions(source: str, target: str) -> None:
    """三条被明确禁止的边：stable→candidate、rolled_back→stable、archived→*。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0", stage=source))
    with pytest.raises(RegistryError, match="非法状态迁移"):
        registry.set_stage("1.0.0", target)


def test_set_stage_rejects_unknown_stage() -> None:
    """阶段名写错立刻报错，而不是把一个不认识的状态写进索引。"""
    registry = seed_registry()
    with pytest.raises(RegistryError, match="未知阶段"):
        registry.set_stage("1.0.1", "production")


def test_set_stage_accepts_record_to_carry_new_metrics() -> None:
    """提升阶段时顺手带上新指标：少一次写入，且不换掉版本本身。"""
    registry = seed_registry()
    candidate = registry.get("1.0.1").with_metrics({"eval_pass_rate": 0.75})
    updated = registry.set_stage("1.0.1", STAGE_STABLE, record=candidate)
    assert updated.pass_rate == 0.75
    assert updated.stage == STAGE_STABLE


def test_set_stage_rejects_record_for_another_version() -> None:
    """``record`` 只能带上"同一个版本"的新指标——换版本本身必须报错。"""
    registry = seed_registry()
    stranger = make("1.0.2", parent="1.0.1")
    with pytest.raises(RegistryError, match="与目标版本"):
        registry.set_stage("1.0.1", STAGE_STABLE, record=stranger)


def test_with_stage_returns_new_object_and_keeps_original() -> None:
    """记录不可变：``with_stage`` 返回新对象，原对象阶段不变。"""
    record = make("1.0.0")
    promoted = record.with_stage(STAGE_STABLE, note="人工门禁")
    assert record.stage == STAGE_CANDIDATE
    assert promoted.stage == STAGE_STABLE
    assert "人工门禁" in promoted.notes


def test_with_stage_entry_guard_rejects_unknown_name() -> None:
    """未知阶段在 ``with_stage`` 入口就被拦下（构造期已拦一次，这里是第二次）。"""
    with pytest.raises(RegistryError, match="未知阶段"):
        make("1.0.0").with_stage("prod")


# --------------------------------------------------------------------------- #
# 版本链
# --------------------------------------------------------------------------- #


def test_lineage_walks_parents_from_near_to_far() -> None:
    """版本链顺序：自身 → 父 → 祖父。"""
    registry = seed_registry()
    registry.register(make("1.0.2", parent="1.0.1"))
    chain = registry.lineage("1.0.2")
    assert [item.version for item in chain] == ["1.0.2", "1.0.1", "1.0.0"]


def test_lineage_honours_limit() -> None:
    """``limit`` 截断链（只要最近两代时不必走完整个历史）。"""
    registry = seed_registry()
    registry.register(make("1.0.2", parent="1.0.1"))
    assert [item.version for item in registry.lineage("1.0.2", limit=2)] == ["1.0.2", "1.0.1"]


def test_lineage_stops_at_missing_ancestor() -> None:
    """祖先不存在时**停在它之前**而不是抛错：历史更长的仓库里这是真实状态。"""
    registry = seed_registry()
    record = registry.get("1.0.0")
    registry._records["1.0.0"] = replace(record, parent_version="0.9.0")
    assert [item.version for item in registry.lineage("1.0.1")] == ["1.0.1", "1.0.0"]


def test_lineage_detects_cycle() -> None:
    """带环的链会让 ``while`` 变成死循环——那是最糟的失败方式（挂住而不是报错）。"""
    registry = seed_registry()
    registry._records["1.0.0"] = replace(registry.get("1.0.0"), parent_version="1.0.1")
    with pytest.raises(RegistryError, match="环"):
        registry.lineage("1.0.1")


def test_stable_ancestors_skips_candidates() -> None:
    """祖先里夹着的 candidate 不是回滚目标（它从未上线过）。"""
    registry = seed_registry()
    registry.register(make("1.0.2", parent="1.0.1"))
    assert [item.version for item in registry.stable_ancestors("1.0.2")] == ["1.0.0"]


# --------------------------------------------------------------------------- #
# prune：不自断退路
# --------------------------------------------------------------------------- #


def test_prune_archives_oldest_candidates_only() -> None:
    """只清理超出保留上限的最旧候选。"""
    registry = ModelRegistry(keep_versions=2)
    registry.register(make("1.0.0", stage=STAGE_STABLE))
    for patch in range(1, 5):
        registry.register(make(f"1.0.{patch}", parent="1.0.0"))
    archived = registry.prune()
    assert archived == ["1.0.1", "1.0.2"]
    assert registry.get("1.0.1").stage == STAGE_ARCHIVED
    assert [item.version for item in registry.versions(stage=STAGE_CANDIDATE)] == [
        "1.0.3",
        "1.0.4",
    ]


def test_prune_never_archives_stable_or_head() -> None:
    """stable 是回滚目标，head 是当前生产版本——两者都不能被"磁盘清理"顺手删掉。"""
    registry = ModelRegistry(keep_versions=1)
    registry.register(make("1.0.0", stage=STAGE_STABLE))
    registry.register(make("1.0.1", parent="1.0.0", stage=STAGE_STABLE))
    registry.register(make("1.0.2", parent="1.0.1"))
    registry.prune()
    assert registry.head().version == "1.0.1"
    assert registry.get("1.0.0").stage == STAGE_STABLE


def test_prune_rejects_non_positive_keep() -> None:
    """``keep=0`` 会被当成"全部归档"——那等于把注册表清空，必须拦下。"""
    registry = seed_registry()
    with pytest.raises(RegistryError, match="必须为正整数"):
        registry.prune(keep=0)


def test_registry_rejects_non_positive_keep_versions() -> None:
    """构造期校验 ``keep_versions``。"""
    with pytest.raises(RegistryError, match="keep_versions"):
        ModelRegistry(keep_versions=0)


# --------------------------------------------------------------------------- #
# 记录本身的构造期护栏
# --------------------------------------------------------------------------- #


def test_record_rejects_unknown_artifact_slot() -> None:
    """产物槽位是**固定集合**：自由字典会让"部署需要哪几样东西"失去定义."""
    with pytest.raises(RegistryError, match="未知产物槽位"):
        make("1.0.0", artifacts={"adapter": "a", "weights": "w"})


def test_record_rejects_invalid_version_and_stage() -> None:
    """版本号与阶段名在构造期校验（入口一次，胜过出口十次）。"""
    with pytest.raises(RegistryError, match="X.Y.Z"):
        make("v1.0")
    with pytest.raises(RegistryError, match="未知阶段"):
        make("1.0.0", stage="production")


def test_record_roundtrip_and_derived_readers() -> None:
    """``to_dict`` / ``from_dict`` 往返，且三个直通读取器与三元组一致."""
    record = make("1.0.0", rating=None, artifacts={"adapter": "a/1.0.0"})
    payload = record.to_dict()
    assert payload["deployable"] is False
    assert payload["missing_artifacts"] == ["merged"]
    restored = ModelVersion.from_dict(payload)
    assert restored.version_key == record.version_key
    assert restored.base_model == BASE
    assert restored.adapter_sha256 == record.adapter_sha256
    assert restored.dataset_fingerprint == DATA_A
    assert restored.version_sort_key == (1, 0, 0)


def test_record_with_metrics_merges_without_mutating() -> None:
    """``with_metrics`` 合并同名新值并返回新对象（记录不可变）。"""
    record = make("1.0.0", rating=0.6)
    updated = record.with_metrics({"eval_pass_rate": 0.8, "latency_ms": 120.0})
    assert updated.pass_rate == 0.8
    assert updated.metrics["latency_ms"] == 120.0
    assert record.pass_rate == 0.6
    assert updated.version_key == record.version_key


def test_record_is_deployable_requires_only_evidence() -> None:
    """``is_deployable`` 只看产物是否齐全，不看阶段与分数."""
    candidate = make("1.0.0")
    assert candidate.stage == STAGE_CANDIDATE
    assert candidate.is_deployable() is True
    assert make("1.0.0", artifacts={}).missing_artifacts() == ["adapter", "merged"]


def test_default_keep_versions_constant() -> None:
    """缺省保留上限是一个显式常量（进文档，不藏在函数体里）。"""
    assert DEFAULT_KEEP_VERSIONS == 20
    assert ModelRegistry().keep_versions == DEFAULT_KEEP_VERSIONS


# --------------------------------------------------------------------------- #
# 版本号提议
# --------------------------------------------------------------------------- #


def test_propose_version_on_empty_registry() -> None:
    """空注册表提议 1.0.0（递增位是占位语义）。"""
    version, kind = ModelRegistry().propose_version(
        base_model=BASE, dataset_fingerprint=DATA_A
    )
    assert (version, kind) == ("1.0.0", "minor")


def test_propose_version_tracks_changed_field() -> None:
    """三次提议分别命中 patch / minor / major。"""
    registry = seed_registry()
    registry.set_stage("1.0.1", STAGE_STABLE)
    assert registry.propose_version(base_model=BASE, dataset_fingerprint=DATA_A) == (
        "1.0.2",
        "patch",
    )
    assert registry.propose_version(base_model=BASE, dataset_fingerprint=DATA_B) == (
        "1.1.0",
        "minor",
    )
    assert registry.propose_version(base_model=BIG, dataset_fingerprint=DATA_B) == (
        "2.0.0",
        "major",
    )


def test_next_version_for_reuses_existing_version_number() -> None:
    """三元组已在表里 → 返回**已有的**版本号，不消耗一个新号。"""
    registry = seed_registry()
    record = registry.get("1.0.1")
    reused = registry.next_version_for(
        base_model=BASE,
        adapter_sha256=record.adapter_sha256,
        dataset_fingerprint=DATA_A,
    )
    assert reused == "1.0.1"


def test_next_version_for_allocates_for_new_triple() -> None:
    """三元组不在表里 → 按递增位算一个新号。"""
    registry = seed_registry()
    registry.set_stage("1.0.1", STAGE_STABLE)
    allocated = registry.next_version_for(
        base_model=BASE,
        adapter_sha256=hexdigest64("brand-new"),
        dataset_fingerprint=DATA_A,
    )
    assert allocated == "1.0.2"


# --------------------------------------------------------------------------- #
# 落盘与重载：事件流
# --------------------------------------------------------------------------- #


def test_index_file_name_constant() -> None:
    """索引文件名固定（脚本照着它找）。"""
    assert INDEX_FILENAME == "versions.jsonl"
    assert EVENT_REGISTER == "register"
    assert EVENT_STAGE == "stage"


def test_register_and_set_stage_write_append_only_events(tmp_path: Path) -> None:
    """落盘的每一行都是一个事件：登记与流转各自成行。"""
    index = tmp_path / INDEX_FILENAME
    registry = ModelRegistry(index)
    registry.register(make("1.0.0"))
    registry.set_stage("1.0.0", STAGE_STABLE, reason="人工门禁")
    lines = index.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["type"] == EVENT_REGISTER
    assert first["version"] == "1.0.0"
    assert second == {
        "type": EVENT_STAGE,
        "version": "1.0.0",
        "stage": STAGE_STABLE,
        "reason": "人工门禁",
        "at": second["at"],
    }


def test_idempotent_register_does_not_duplicate_events(tmp_path: Path) -> None:
    """幂等命中不写重复事件：重复写会让索引膨胀，而语义上什么都没发生。"""
    index = tmp_path / INDEX_FILENAME
    registry = ModelRegistry(index)
    registry.register(make("1.0.0"))
    registry.register(make("1.0.0"))
    assert len(index.read_text(encoding="utf-8").splitlines()) == 1


def test_reload_rebuilds_identical_state(tmp_path: Path) -> None:
    """重载后状态逐项一致（含 head 与版本键）。"""
    index = tmp_path / INDEX_FILENAME
    registry = ModelRegistry(index)
    registry.register(make("1.0.0"))
    registry.set_stage("1.0.0", STAGE_STABLE)
    registry.register(make("1.0.1", parent="1.0.0"))

    reloaded = ModelRegistry(index)
    assert reloaded.counts() == registry.counts()
    assert reloaded.head().version_key == registry.head().version_key
    assert len(reloaded.events()) == len(registry.events()) == 3


def test_reload_is_idempotent(tmp_path: Path) -> None:
    """``reload`` 先清空再折叠，因此可以重复调用。"""
    index = tmp_path / INDEX_FILENAME
    registry = ModelRegistry(index)
    registry.register(make("1.0.0"))
    registry.reload()
    registry.reload()
    assert len(registry) == 1
    assert len(registry.events()) == 1


def test_missing_index_file_is_not_an_error(tmp_path: Path) -> None:
    """索引文件不存在 = 空注册表（首次上线前的真实状态）。"""
    registry = ModelRegistry(tmp_path / "nope" / INDEX_FILENAME)
    assert len(registry) == 0
    assert registry.head() is None
    assert registry.directory == tmp_path / "nope"


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """空行被跳过（追加写被打断时可能出现）。"""
    index = tmp_path / INDEX_FILENAME
    payload = json.dumps({"type": EVENT_REGISTER, **make("1.0.0").to_dict()}, ensure_ascii=False)
    index.write_text(f"\n{payload}\n\n", encoding="utf-8")
    assert len(ModelRegistry(index)) == 1


@pytest.mark.parametrize(
    "content, match",
    [
        ('{"type": "register", "version": "1.0.0"}\n', "缺少字段"),
        ('{"type": "unknown", "version": "1.0.0"}\n', "未知事件类型"),
        ('["not", "an", "object"]\n', "不是 JSON 对象"),
        ('{"type": "register", broken\n', "不是合法 JSON"),
        ('{"type": "stage", "version": "1.0.0", "stage": "stable"}\n', "不存在的版本"),
    ],
)
def test_corrupt_index_lines_raise_instead_of_silently_dropping(
    tmp_path: Path, content: str, match: str
) -> None:
    """五类损坏各自报错：跳过不认识的事件会丢状态，而丢状态的表现只是 head() 变 None。"""
    index = tmp_path / INDEX_FILENAME
    index.write_text(content, encoding="utf-8")
    with pytest.raises(RegistryError, match=match):
        ModelRegistry(index)


def test_in_memory_registry_does_not_touch_disk(tmp_path: Path) -> None:
    """``path=None`` 时纯内存：不创建任何文件（测试与"先算计划"的调用方都靠它）。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0"))
    registry.set_stage("1.0.0", STAGE_STABLE)
    assert registry.path is None
    assert registry.directory is None
    assert list(tmp_path.iterdir()) == []


def test_events_returns_copies() -> None:
    """``events()`` 返回副本：外部改写不会污染内部审计流。"""
    registry = seed_registry()
    snapshot = registry.events()
    snapshot[0]["version"] = "tampered"
    assert registry.events()[0]["version"] == "1.0.0"


# --------------------------------------------------------------------------- #
# 投影
# --------------------------------------------------------------------------- #


def test_to_dict_reports_head_and_counts() -> None:
    """投影里带 head / head_key / counts，接口直接消费。"""
    registry = seed_registry()
    payload = registry.to_dict()
    assert payload["head"] == "1.0.0"
    assert payload["head_key"] == registry.get("1.0.0").version_key
    assert payload["total"] == 2
    assert payload["versions"][0]["version"] == "1.0.0"


def test_to_dict_without_head_is_null() -> None:
    """没有 stable → head 为 None（不是"退回最新候选"，那是两件事）。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0"))
    assert registry.to_dict()["head"] is None


def test_render_markdown_contains_every_version_row() -> None:
    """markdown 表格逐行包含每个版本（接口与报告同源的保证）。"""
    registry = seed_registry()
    markdown = registry.render_markdown()
    assert "模型版本注册表" in markdown
    assert "| v1.0.0 |" in markdown
    assert "| v1.0.1 |" in markdown
    assert registry.get("1.0.0").version_key in markdown


def test_render_markdown_marks_missing_production_version() -> None:
    """尚无 stable 时报告里明确写出来，而不是留一个空表格让人猜。"""
    registry = ModelRegistry()
    registry.register(make("1.0.0", rating=None))
    markdown = registry.render_markdown()
    assert "尚无 stable 版本" in markdown
    assert "未评估" in registry.get("1.0.0").summary_line()
