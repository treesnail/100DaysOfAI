"""day044 访问日志中间件测试：AccessLogStore 存取 + AccessLogMiddleware 记录（全部离线）."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.middleware import AccessLogStore


class TestAccessLogStore:
    def test_record_and_query(self):
        store = AccessLogStore()
        store.record("GET", "/health", 200, 1.5, "1.2.3.4")
        store.record("POST", "/chat", 500, 3.2, "5.6.7.8")
        records = store.query()
        assert len(records) == 2
        assert records[0].method == "GET"
        assert records[0].path == "/health"
        assert records[0].status_code == 200
        assert records[0].duration_ms == 1.5
        assert records[0].client_ip == "1.2.3.4"
        assert records[0].timestamp > 0
        assert records[1].status_code == 500

    def test_persist_to_jsonl(self, tmp_path):
        store = AccessLogStore(tmp_path / "access.jsonl")
        store.record("POST", "/chat", 200, 2.0, "testclient")
        lines = (tmp_path / "access.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["method"] == "POST"
        assert payload["path"] == "/chat"
        assert payload["status_code"] == 200

    def test_no_path_means_memory_only(self):
        """不给路径时纯内存，不产生文件."""
        store = AccessLogStore()
        store.record("GET", "/health", 200, 0.1, "x")
        assert store.query()  # 内存可查


class TestAccessLogMiddleware:
    def test_every_request_logged(self):
        store = AccessLogStore()
        client = TestClient(create_app(access_log_store=store))
        client.get("/health")
        client.post("/chat", json={"message": "你好"})
        records = store.query()
        assert len(records) == 2
        assert records[0].method == "GET"
        assert records[0].path == "/health"
        assert records[0].status_code == 200
        assert records[1].method == "POST"
        assert records[1].path == "/chat"

    def test_404_also_logged(self):
        """审计要覆盖失败：路由不存在的 404 也要留痕."""
        store = AccessLogStore()
        client = TestClient(create_app(access_log_store=store))
        resp = client.get("/no-such-route")
        assert resp.status_code == 404
        records = store.query()
        assert len(records) == 1
        assert records[0].status_code == 404

    def test_500_also_logged(self):
        """异常场景同样记录：全局异常处理器返回 500，中间件记下该状态码."""

        def exploding_factory():
            raise RuntimeError("boom")

        store = AccessLogStore()
        client = TestClient(
            create_app(access_log_store=store, agent_factory=exploding_factory),
            raise_server_exceptions=False,  # 让全局异常处理器把异常转为 500 响应
        )
        resp = client.post("/agent/run", json={"task": "会炸的任务"})
        assert resp.status_code == 500
        records = store.query()
        assert len(records) == 1
        assert records[0].status_code == 500
        assert records[0].method == "POST"
        assert records[0].path == "/agent/run"

    def test_duration_is_measured(self):
        store = AccessLogStore()
        client = TestClient(create_app(access_log_store=store))
        client.get("/health")
        assert store.query()[0].duration_ms >= 0

    def test_client_ip_recorded(self):
        store = AccessLogStore()
        client = TestClient(create_app(access_log_store=store))
        client.get("/health")
        # TestClient 的 client 元组 host 固定为 "testclient"
        assert store.query()[0].client_ip == "testclient"
