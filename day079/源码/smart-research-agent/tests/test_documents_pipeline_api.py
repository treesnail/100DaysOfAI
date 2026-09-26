"""day061 ``documents.pipeline`` 与 /documents/* 端点的单元测试.

全部离线、确定性：不联网、不遍历真实目录（用 ``tmp_path``）、不写盘。
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.documents import (
    BLOCK_KINDS,
    DOCUMENTS_LIMITATIONS,
    DOCUMENTS_OUT_OF_SCOPE,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    DocumentError,
    DocumentIngestor,
    IngestEntry,
    encoding_chain,
    iter_supported_files,
    knowledge_record_shape,
)
from smart_research_agent.llm.mock import MockLLM
from tests.document_samples import build_docx, build_pdf

MARKDOWN = "# 标题\n\n正文一段。\n"
MARKDOWN_OTHER = "# 另一个标题\n\n不同的正文。\n"


@pytest.fixture
def client() -> TestClient:
    """注入 MockLLM 的离线测试客户端（/documents/* 端点本身不碰 LLM）."""
    return TestClient(create_app(llm=MockLLM(default="offline")))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# --------------------------------------------------------------------------- #
# 入库
# --------------------------------------------------------------------------- #


def test_ingest_deduplicates_by_content_fingerprint() -> None:
    """**内容指纹**去重的两条性质：换个文件名也是同一份，改一个字节就是两份."""
    ingestor = DocumentIngestor()
    report = ingestor.ingest_bytes(
        [
            ("a.md", MARKDOWN.encode("utf-8")),
            ("copy-of-a.md", MARKDOWN.encode("utf-8")),      # 同内容不同名 → 重复
            ("b.md", MARKDOWN_OTHER.encode("utf-8")),
        ]
    )
    assert report.files == 3
    assert report.unique_documents == 2
    assert report.status_counts() == {"ok": 2, "duplicate": 1, "error": 0}
    duplicate = report.duplicates()[0]
    assert duplicate.source == "copy-of-a.md"
    assert duplicate.duplicate_of == "a.md"
    assert duplicate.doc_id == report.documents[0].fingerprint
    assert duplicate.status == "duplicate"


def test_ingest_keeps_the_first_seen_document() -> None:
    """顺序**就是**优先级：同一批里谁先出现谁留下，因此结果与遍历顺序无关."""
    first = DocumentIngestor().ingest_bytes(
        [("a.md", MARKDOWN.encode("utf-8")), ("b.md", MARKDOWN.encode("utf-8"))]
    )
    assert first.documents[0].source == "a.md"
    assert first.entries[1].duplicate_of == "a.md"


def test_ingest_records_unsupported_files_and_continues() -> None:
    """一份文件"不在支持范围"是**结论**，不是整批失败（与 day060 的 blocked/failed 同源）."""
    report = DocumentIngestor().ingest_bytes(
        [
            ("good.md", MARKDOWN.encode("utf-8")),
            ("binary.txt", b"\x00\x01\x02"),
            ("bad.pdf", b"not a pdf"),
        ]
    )
    assert report.status_counts() == {"ok": 1, "duplicate": 0, "error": 2}
    assert len(report.documents) == 1
    assert {entry.status for entry in report.errors()} == {"error"}
    assert any("看起来是二进制" in entry.error for entry in report.errors())
    assert any("不是 PDF" in entry.error for entry in report.errors())


def test_ingest_summary_and_markdown_are_readable() -> None:
    report = DocumentIngestor().ingest_bytes(
        [
            ("a.md", MARKDOWN.encode("utf-8")),
            ("b.html", b"<html><body><p>text</p></body></html>"),
            ("c.pdf", build_pdf(["pdf text"])),
            ("d.docx", build_docx([("Heading1", "标题")])),
            ("e.txt", "纯文本一段。\n".encode("utf-8")),
        ]
    )
    assert "5 个文件 → 5 份文档" in report.summary_line()
    markdown = report.render_markdown()
    assert "# 文档入库报告" in markdown
    assert "| 块类型 | 数量 |" in markdown
    counts = report.block_counts()
    assert set(counts) == set(BLOCK_KINDS)
    assert counts["table"] == 1  # docx 样本里的那张表
    assert sum(report.media_type_counts().values()) == 5


def test_ingest_report_to_dict_is_json_ready() -> None:
    report = DocumentIngestor().ingest_bytes([("a.md", MARKDOWN.encode("utf-8"))])
    payload = report.to_dict(include_documents=True)
    assert json.loads(json.dumps(payload))["files"] == 1
    assert payload["documents"][0]["char_count"] > 0
    assert "blocks" not in payload["documents"][0]
    assert payload["entries"][0]["status"] == "ok"


def test_knowledge_records_use_the_day009_shape() -> None:
    """接缝的形状只有四个字段，而且**刻意不做分块**——分块是 day062 的事."""
    report = DocumentIngestor().ingest_bytes([("a.md", MARKDOWN.encode("utf-8"))])
    record = report.knowledge_records()[0]
    assert set(record) == {"doc_id", "source", "text", "metadata"}
    assert record["doc_id"] == report.documents[0].fingerprint
    assert record["metadata"]["media_type"] == MEDIA_MARKDOWN
    assert record["metadata"]["block_count"] == "2"
    assert knowledge_record_shape()["fields"].keys() == set(record)


def test_iter_supported_files_is_sorted_and_skips_hidden_dirs(tmp_path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "c.md").write_text("c", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "d.txt").write_text("d", encoding="utf-8")

    names = [path.name for path in iter_supported_files(tmp_path)]
    assert names == ["a.md", "b.md", "d.txt"]

    only_markdown = iter_supported_files(tmp_path, suffixes=[".md"])
    assert [path.name for path in only_markdown] == ["a.md", "b.md"]


def test_iter_supported_files_rejects_a_non_directory(tmp_path) -> None:
    target = tmp_path / "file.md"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentError, match="不是一个目录"):
        iter_supported_files(target)


def test_ingest_dir_walks_and_reports_the_root(tmp_path) -> None:
    (tmp_path / "a.md").write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "b.md").write_text(MARKDOWN, encoding="utf-8")
    report = DocumentIngestor().ingest_dir(tmp_path)
    assert report.files == 2
    assert report.unique_documents == 1
    assert report.metadata["root"] == str(tmp_path)
    assert report.metadata["scanned"] == "2"


def test_ingest_dir_limit_applies_after_sorting(tmp_path) -> None:
    for name in ("c.md", "a.md", "b.md"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    report = DocumentIngestor().ingest_dir(tmp_path, limit=2)
    assert [entry.source.split("\\")[-1].split("/")[-1] for entry in report.entries] == [
        "a.md",
        "b.md",
    ]


def test_ingest_dir_rejects_a_negative_limit(tmp_path) -> None:
    with pytest.raises(DocumentError, match="limit 不能为负数"):
        DocumentIngestor().ingest_dir(tmp_path, limit=-1)


def test_ingest_entry_flags() -> None:
    entry = IngestEntry(source="a.md", media_type=MEDIA_MARKDOWN, doc_id="x", char_count=3)
    assert entry.ok is True
    assert entry.is_duplicate is False
    assert entry.status == "ok"
    assert "[ok]" in entry.summary_line()
    duplicate = IngestEntry(source="b.md", duplicate_of="a.md")
    assert duplicate.is_duplicate is True
    assert "[重复]" in duplicate.summary_line()
    failed = IngestEntry(source="c.txt", error="boom")
    assert failed.ok is False
    assert failed.status == "error"
    assert "[error]" in failed.summary_line()


def test_limitations_each_name_a_concrete_boundary() -> None:
    """每条限制都要能被用来否掉一个结论（day059/060 同一条纪律）."""
    assert len(DOCUMENTS_LIMITATIONS) == 5
    assert any("§9.10.2" in item for item in DOCUMENTS_LIMITATIONS)
    assert any("docx_parts_not_read" in item for item in DOCUMENTS_LIMITATIONS)
    assert any("encoding_confident=false" in item for item in DOCUMENTS_LIMITATIONS)
    assert len(DOCUMENTS_OUT_OF_SCOPE) == 3
    assert any("pypdf" in item for item in DOCUMENTS_OUT_OF_SCOPE)


# --------------------------------------------------------------------------- #
# 端点
# --------------------------------------------------------------------------- #


def test_loaders_endpoint_is_the_self_description(client: TestClient) -> None:
    payload = client.get("/documents/loaders").json()
    assert payload["media_types"]
    assert len(payload["loaders"]) == 5
    assert payload["encoding_chain"] == encoding_chain()
    assert payload["block_kinds"] == list(BLOCK_KINDS)
    assert payload["max_file_mib"] == settings.documents_max_file_mib
    assert payload["out_of_scope"] == list(DOCUMENTS_OUT_OF_SCOPE)
    assert payload["limitations"] == list(DOCUMENTS_LIMITATIONS)
    assert any("Identity-H" in item for item in payload["pdf"]["not_supported"])
    assert payload["docx"]["parts_read"]


def test_parse_endpoint_returns_a_document_and_a_detection(client: TestClient) -> None:
    resp = client.post("/documents/parse", json={"filename": "a.md", "text": MARKDOWN})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["document"]["media_type"] == MEDIA_MARKDOWN
    assert [block["kind"] for block in payload["document"]["blocks"]] == [
        "heading",
        "paragraph",
    ]
    assert payload["detection"]["decided_by"] == "suffix"
    assert payload["document"]["doc_id"] == client.post(
        "/documents/parse", json={"filename": "renamed.md", "text": MARKDOWN}
    ).json()["document"]["doc_id"]


def test_parse_endpoint_can_omit_blocks(client: TestClient) -> None:
    payload = client.post(
        "/documents/parse",
        json={"filename": "a.md", "text": MARKDOWN, "include_blocks": False},
    ).json()
    assert "blocks" not in payload["document"]
    assert payload["document"]["block_count"] == 2


def test_parse_endpoint_accepts_base64_for_binary_formats(client: TestClient) -> None:
    payload = client.post(
        "/documents/parse",
        json={"filename": "a.pdf", "content_base64": _b64(build_pdf(["from pdf"]))},
    ).json()
    assert "from pdf" in payload["document"]["preview"]
    assert payload["document"]["media_type"] == MEDIA_PDF


def test_parse_endpoint_honours_an_explicit_media_type(client: TestClient) -> None:
    """文件名不可信时给"我已经知道格式"留的入口（``decided_by=given``）."""
    payload = client.post(
        "/documents/parse",
        json={"filename": "noextension", "text": MARKDOWN, "media_type": MEDIA_MARKDOWN},
    ).json()
    assert payload["detection"]["media_type"] == MEDIA_MARKDOWN
    assert payload["detection"]["decided_by"] == "given"


def test_parse_endpoint_uses_an_inline_placeholder_when_no_name(client: TestClient) -> None:
    """``source`` 不允许为空，因此没给名字时用一个**显式的占位名**."""
    payload = client.post("/documents/parse", json={"text": MARKDOWN}).json()
    assert payload["document"]["source"] == "<inline>"


def test_parse_endpoint_rejects_bad_requests_with_400(client: TestClient) -> None:
    assert client.post("/documents/parse", json={"filename": "a.txt"}).status_code == 400
    assert (
        client.post(
            "/documents/parse", json={"filename": "a.txt", "content_base64": "@@@"}
        ).status_code
        == 400
    )
    unsupported = client.post(
        "/documents/parse",
        json={"filename": "a.zip", "content_base64": _b64(b"PK\x03\x04xxxx")},
    )
    assert unsupported.status_code == 400
    assert "application/zip" in unsupported.json()["detail"]


def test_detect_endpoint_reports_conflicts_and_encoding(client: TestClient) -> None:
    payload = client.post(
        "/documents/detect",
        json={"filename": "fake.txt", "content_base64": _b64(b"%PDF-1.7\n")},
    ).json()
    assert payload["media_type"] == MEDIA_PDF
    assert payload["conflict"] is True
    assert payload["magic_type"] == MEDIA_PDF
    assert payload["suffix_type"] == "text/plain"
    assert payload["supported"] is True
    assert payload["encoding"] == ""          # PDF 是二进制，不做编码探测
    assert payload["size_bytes"] == len(b"%PDF-1.7\n")


def test_detect_endpoint_reports_encoding_confidence(client: TestClient) -> None:
    trusted = client.post(
        "/documents/detect", json={"filename": "a.txt", "text": "中文"}
    ).json()
    assert trusted["encoding"] == "utf-8"
    assert trusted["encoding_confident"] is True
    untrusted = client.post(
        "/documents/detect",
        json={"filename": "a.txt", "content_base64": _b64(b"\x80\x80")},
    ).json()
    assert untrusted["encoding"] == "latin-1"
    assert untrusted["encoding_confident"] is False
    assert "编码" in untrusted["summary"]


def test_detect_endpoint_flags_an_unsupported_type_without_failing(client: TestClient) -> None:
    """「它是 ZIP」是一个结论，在 ``/documents/parse`` 里才会被拒收."""
    payload = client.post(
        "/documents/detect",
        json={"filename": "a.docx", "content_base64": _b64(build_docx([("", "x")]))},
    ).json()
    assert payload["supported"] is True
    assert payload["size_bytes"] > 0


def test_detect_endpoint_requires_content(client: TestClient) -> None:
    assert client.post("/documents/detect", json={"filename": "a.txt"}).status_code == 400


def test_ingest_endpoint_returns_a_full_report(client: TestClient) -> None:
    payload = client.post(
        "/documents/ingest",
        json={
            "files": [
                {"filename": "a.md", "text": MARKDOWN},
                {"filename": "b.md", "text": MARKDOWN},
                {"filename": "c.txt", "content_base64": _b64(b"\x00\x01")},
            ]
        },
    ).json()
    assert payload["report"]["status_counts"] == {"ok": 1, "duplicate": 1, "error": 1}
    assert payload["report"]["unique_documents"] == 1
    assert len(payload["report"]["documents"]) == 1
    assert "# 文档入库报告" in payload["markdown"]
    assert "1 份文档" in payload["summary"]


def test_ingest_endpoint_with_no_files_is_a_valid_empty_report(client: TestClient) -> None:
    """"仓库里还没有文档"是首次入库前的真实状态，不是错误."""
    payload = client.post("/documents/ingest", json={"files": []}).json()
    assert payload["report"]["files"] == 0
    assert payload["report"]["unique_documents"] == 0
    assert payload["report"]["status_counts"] == {"ok": 0, "duplicate": 0, "error": 0}


def test_ingest_endpoint_rejects_an_empty_file_with_400(client: TestClient) -> None:
    resp = client.post("/documents/ingest", json={"files": [{"filename": "a.md"}]})
    assert resp.status_code == 400
    assert "必须给出一个" in resp.json()["detail"]
