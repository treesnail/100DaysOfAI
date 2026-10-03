"""``rag_ops.snapshot`` 与 ``rag_ops.sync``：两级增量的上半段（day072）.

本文件钉住三件事：

```text
① 扫盘是确定的    同样的目录永远算出同一个快照号（不含时间、不含机器）
② 差集是完备的    四组动作必须把两侧来源各解释一遍（漏一份 = 库里少一条记录）
③ 换挡是显式的    变更比例超阈值才建议"整库重建"，且理由必须能被读出来
```

全部离线：语料写在 ``tmp_path`` 下，编码器与后端都不需要网络。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_research_agent.rag_ops.errors import SyncError
from smart_research_agent.rag_ops.snapshot import (
    entry_from_document,
    relative_source,
    resolve_source_root,
    scan_corpus,
    scan_sources,
)
from smart_research_agent.rag_ops.sync import (
    changed_lines,
    plan_sync,
    resolved_mode,
    suggest_mode,
)
from smart_research_agent.rag_ops.types import (
    CorpusSnapshot,
    SourceEntry,
    SyncPlan,
    to_iso,
)
from tests.rag_ops_samples import (
    CORPUS_FILES,
    FIXED_NOW,
    SINGLE_FILE,
    entry,
    fingerprint,
    plan,
    snapshot,
    write_corpus,
)


class TestScanCorpus:
    """扫盘：确定性、相对 posix 路径、limit 与"读不进来"的两种处理."""

    def test_returns_snapshot_and_documents_once(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        snap, documents = scan_corpus(root, clock=FIXED_NOW)
        assert snap.count == 1 and len(documents) == 1
        assert documents[0].fingerprint == snap.entries[0].fingerprint
        assert snap.created_at == to_iso(FIXED_NOW)
        assert snap.entries[0].char_count == documents[0].char_count
        assert snap.entries[0].media_type == "text/markdown"

    def test_sources_are_relative_posix_and_sorted(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", CORPUS_FILES)
        (root / "nested").mkdir()
        (root / "nested" / "deep.md").write_text("# 深一层\n\n正文。\n", encoding="utf-8")
        snap = scan_sources(root, clock=FIXED_NOW)
        assert snap.sources == ("nested/deep.md", "ops.md", "rag.md", "sched.md")
        assert all("\\" not in name for name in snap.sources)

    def test_same_content_gives_the_same_digest(self, tmp_path: Path) -> None:
        first = write_corpus(tmp_path / "one", CORPUS_FILES)
        second = write_corpus(tmp_path / "two", CORPUS_FILES)
        left = scan_sources(first, clock=FIXED_NOW)
        right = scan_sources(second, clock=FIXED_NOW)
        assert left.digest == right.digest

    def test_limit_keeps_the_sorted_head(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", CORPUS_FILES)
        snap = scan_sources(root, clock=FIXED_NOW, limit=2)
        assert snap.sources == ("ops.md", "rag.md")

    def test_unknown_suffixes_are_not_scanned(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        (root / "notes.txt").write_text("运维笔记。\n", encoding="utf-8")
        (root / "skip.bin").write_bytes(b"\x00\x01")  # 不在加载器注册表里的后缀：直接不扫
        snap = scan_sources(root, clock=FIXED_NOW)
        assert snap.sources == ("notes.txt", "rag.md")

    def test_strict_mode_fails_on_a_broken_file(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        (root / "broken.pdf").write_bytes(b"this is not a pdf")
        with pytest.raises(SyncError) as excinfo:
            scan_sources(root, clock=FIXED_NOW)
        assert "broken.pdf" in str(excinfo.value)
        assert "strict=False" in str(excinfo.value)

    def test_lenient_mode_records_the_skip(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        (root / "broken.pdf").write_bytes(b"this is not a pdf")
        snap = scan_sources(root, clock=FIXED_NOW, strict=False)
        assert snap.sources == ("rag.md",)
        assert snap.skipped[0][0] == "broken.pdf"
        assert snap.skipped[0][1]
        assert snap.to_dict(include_entries=False)["skipped"] == 1

    @pytest.mark.parametrize("bad", ["", ".", "no/such/dir"])
    def test_bad_roots_are_rejected(self, tmp_path: Path, bad: str) -> None:
        # "" 与 "." 都要拒绝：Path("") == Path(".")，否则"没配目录"会变成
        # "拿进程当前目录当语料"（在仓库根目录跑一次就会把整个仓库灌进库）。
        target = bad if bad in ("", ".") else tmp_path / bad
        with pytest.raises(SyncError):
            scan_sources(target)

    def test_file_instead_of_directory_is_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "corpus.md"
        target.write_text("# 不是目录\n", encoding="utf-8")
        with pytest.raises(SyncError):
            scan_sources(target)

    def test_empty_directory_gives_an_empty_snapshot(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        snap, documents = scan_corpus(root, clock=FIXED_NOW)
        assert snap.count == 0 and documents == ()
        # 空快照仍然有一个**稳定**的快照号（"空"也是一种内容，不是"算不出来"）
        assert snap.digest == scan_sources(root, clock=FIXED_NOW).digest


class TestSnapshotHelpers:
    """三个小助手：根目录解析、相对路径折算、文档折成条目."""

    def test_resolve_source_root(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        assert resolve_source_root(root) == root

    def test_relative_source(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        assert relative_source(root / "rag.md", root) == "rag.md"

    def test_entry_from_document(self, tmp_path: Path) -> None:
        root = write_corpus(tmp_path / "corpus", SINGLE_FILE)
        _, documents = scan_corpus(root, clock=FIXED_NOW)
        item = entry_from_document(documents[0], source="alias.md")
        assert item.source == "alias.md"
        assert item.fingerprint == documents[0].fingerprint


class TestPlanSync:
    """差集：四条判据、三种理由、以及"账本丢了会全量重跑一次"的实现."""

    def test_first_run_is_all_added(self) -> None:
        result = plan_sync(None, snapshot())
        assert result.added == ("docs/a.md", "docs/b.md")
        assert result.updated == () and result.removed == () and result.unchanged == ()
        assert result.previous_id == ""
        assert "没有水位" in result.reason
        assert result.changed_count == 2

    def test_no_change_keeps_everything_unchanged(self) -> None:
        before = snapshot()
        result = plan_sync(before, snapshot())
        assert result.unchanged == ("docs/a.md", "docs/b.md")
        assert result.empty is True
        assert result.previous_id == before.digest
        assert "一份来源都没变" in result.reason

    def test_added_updated_removed_are_all_detected(self) -> None:
        before = snapshot(
            (entry("docs/a.md"), entry("docs/b.md"), entry("docs/c.md"))
        )
        after = snapshot(
            (
                entry("docs/a.md", seed="changed"),  # 内容变了
                entry("docs/c.md"),  # 没变
                entry("docs/d.md"),  # 新增
            )
        )
        result = plan_sync(before, after)
        assert result.added == ("docs/d.md",)
        assert result.updated == ("docs/a.md",)
        assert result.removed == ("docs/b.md",)
        assert result.unchanged == ("docs/c.md",)
        assert result.changed == ("docs/a.md", "docs/b.md", "docs/d.md")
        assert "新增 1、更新 1、删除 1、未变 1" in result.reason

    def test_removing_everything_is_a_change(self) -> None:
        result = plan_sync(snapshot(), CorpusSnapshot())
        assert result.removed == ("docs/a.md", "docs/b.md")
        assert result.empty is False and result.churn_ratio == 1.0

    def test_plan_result_is_a_valid_sync_plan(self) -> None:
        result = plan_sync(snapshot((entry("docs/a.md"),)), snapshot((entry("docs/b.md"),)))
        assert isinstance(result, SyncPlan)
        assert result.to_dict()["current_sources"] == ["docs/b.md"]


class TestModeSuggestion:
    """换挡：只有"变更比例超过阈值"才建议全量，且理由必须能被读出来."""

    def test_empty_plan_suggests_nothing(self) -> None:
        assert suggest_mode(plan(unchanged=("docs/a.md",))) is None

    def test_high_churn_suggests_full_rebuild(self) -> None:
        heavy = plan(updated=("docs/a.md", "docs/b.md"), unchanged=("docs/c.md",))
        assert suggest_mode(heavy, threshold=0.5) == "full"

    def test_low_churn_suggests_nothing(self) -> None:
        light = plan(
            updated=("docs/a.md",),
            unchanged=("docs/b.md", "docs/c.md", "docs/d.md"),
        )
        assert suggest_mode(light, threshold=0.5) is None

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_out_of_range_threshold_is_rejected(self, threshold: float) -> None:
        with pytest.raises(SyncError):
            suggest_mode(plan(updated=("docs/a.md",)), threshold=threshold)

    def test_zero_threshold_triggers_on_any_change(self) -> None:
        one = plan(updated=("docs/a.md",), unchanged=("docs/b.md",))
        assert suggest_mode(one, threshold=0.0) == "full"

    def test_resolved_mode_explains_both_directions(self) -> None:
        heavy = plan(updated=("docs/a.md", "docs/b.md"), unchanged=("docs/c.md",))
        mode, note = resolved_mode(heavy, threshold=0.5)
        assert mode == "full" and "整库重建" in note and "变更比例" in note

        light = plan(updated=("docs/a.md",), unchanged=("docs/b.md", "docs/c.md"))
        mode, note = resolved_mode(light, threshold=0.5)
        assert mode == "incremental" and "增量" in note

    def test_resolved_mode_uses_the_config_threshold_by_default(self) -> None:
        # settings.rag_ops_churn_full_rebuild = 0.5：一份改、一份未变 → 50% 未超过
        assert resolved_mode(plan(updated=("docs/a.md",), unchanged=("docs/b.md",)))[0] == (
            "incremental"
        )


class TestChangedLines:
    """人话：按动作分组、带标记、超长只截断展示."""

    def test_groups_and_marks(self) -> None:
        lines = changed_lines(
            plan(added=("docs/new.md",), updated=("docs/a.md",), removed=("docs/gone.md",))
        )
        assert lines[0].startswith("+ docs/new.md")
        assert lines[1].startswith("~ docs/a.md")
        assert lines[2].startswith("- docs/gone.md")

    def test_limit_truncates_and_says_so(self) -> None:
        built = plan(
            added=("docs/a.md", "docs/b.md", "docs/c.md"), unchanged=("docs/d.md",)
        )
        lines = changed_lines(built, limit=2)
        assert len(lines) == 3
        assert "另有 2 条未列出" in lines[-1]

    def test_negative_limit_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            changed_lines(plan(), limit=-1)


class TestSnapshotWithSyntheticEntries:
    """手工快照与真实扫盘结果必须同形（否则样本与真实数据会分家）."""

    def test_manual_entries_are_valid_source_entries(self) -> None:
        manual = snapshot((entry("docs/a.md"),))
        assert isinstance(manual.entries[0], SourceEntry)
        assert manual.entries[0].fingerprint == fingerprint("docs/a.md")
        assert manual.digest != snapshot((entry("docs/b.md"),)).digest
