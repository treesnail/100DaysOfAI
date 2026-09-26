"""微调数据端点测试（day048）：/finetune/methods 与 /finetune/dataset/*.

全部走 TestClient（进程内 ASGI 调用），不起真实服务、不联网。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.routes import MAX_DATASET_ISSUES
from smart_research_agent.config import settings
from smart_research_agent.finetune.overview import METHODS
from smart_research_agent.llm.mock import MockLLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_ALPACA = {
    "instruction": "什么是 RAG？",
    "input": "",
    "output": "RAG 是检索增强生成：先检索相关片段，再交给大模型生成答案。",
}


@pytest.fixture
def client() -> TestClient:
    """离线客户端：注入 MockLLM，其余依赖走 create_app 默认装配."""
    return TestClient(create_app(llm=MockLLM()))


@pytest.fixture
def repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """把数据目录指向仓库绝对路径，使测试不依赖进程工作目录."""
    finetune_dir = PROJECT_ROOT / "data" / "finetune"
    monkeypatch.setattr(settings, "finetune_data_dir", str(finetune_dir))
    monkeypatch.setattr(settings, "finetune_seed_path", str(finetune_dir / "seed_examples.jsonl"))


class TestFinetuneMethodsEndpoint:
    """GET /finetune/methods：五种方法的画像."""

    def test_returns_all_methods(self, client):
        response = client.get("/finetune/methods")
        assert response.status_code == 200
        methods = response.json()["methods"]
        assert [m["key"] for m in methods] == list(METHODS)

    def test_method_contract_fields(self, client):
        first = client.get("/finetune/methods").json()["methods"][0]
        assert set(first) == {
            "key",
            "full_name",
            "family",
            "data_need",
            "trainable_ratio_hint",
            "stability",
            "best_for",
            "artifacts",
            "notes",
        }
        assert first["key"] == "sft"
        assert isinstance(first["artifacts"], list) and first["artifacts"]

    def test_paths_are_documented_in_openapi(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/finetune/methods" in paths
        assert "/finetune/dataset/validate" in paths
        assert "/finetune/dataset/stats" in paths


class TestDatasetValidateEndpoint:
    """POST /finetune/dataset/validate：逐条校验 + 画像."""

    def test_valid_batch(self, client):
        response = client.post(
            "/finetune/dataset/validate",
            json={"format": "alpaca", "examples": [VALID_ALPACA]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["valid"] == 1
        assert body["invalid"] == 0
        assert body["issues"] == []
        assert body["stats"]["count"] == 1

    def test_partial_invalid_batch_reports_index_and_reason(self, client):
        response = client.post(
            "/finetune/dataset/validate",
            json={
                "format": "alpaca",
                "examples": [
                    VALID_ALPACA,
                    {"instruction": "缺少答案"},
                    {"instruction": "另一个问题", "output": "合法答案"},
                ],
            },
        )
        body = response.json()
        assert body["total"] == 3
        assert body["valid"] == 2
        assert body["invalid"] == 1
        assert len(body["issues"]) == 1
        assert body["issues"][0]["index"] == 1
        assert "output/completion 不能为空" in body["issues"][0]["reason"]

    def test_unknown_format_returns_400(self, client):
        response = client.post(
            "/finetune/dataset/validate",
            json={"format": "sharegpt", "examples": [VALID_ALPACA]},
        )
        assert response.status_code == 400
        assert "未知数据集格式" in response.json()["detail"]

    def test_empty_examples_returns_422(self, client):
        response = client.post(
            "/finetune/dataset/validate", json={"format": "alpaca", "examples": []}
        )
        assert response.status_code == 422

    def test_default_format_is_alpaca(self, client):
        response = client.post("/finetune/dataset/validate", json={"examples": [VALID_ALPACA]})
        assert response.status_code == 200
        assert response.json()["valid"] == 1

    def test_chat_format(self, client):
        response = client.post(
            "/finetune/dataset/validate",
            json={
                "format": "chat",
                "examples": [
                    {
                        "messages": [
                            {"role": "user", "content": "问题"},
                            {"role": "assistant", "content": "答案"},
                        ]
                    }
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["valid"] == 1

    def test_chat_format_without_assistant_is_invalid(self, client):
        response = client.post(
            "/finetune/dataset/validate",
            json={
                "format": "chat",
                "examples": [{"messages": [{"role": "user", "content": "问题"}]}],
            },
        )
        body = response.json()
        assert body["invalid"] == 1
        assert "user 与 assistant" in body["issues"][0]["reason"]

    def test_prompt_completion_format(self, client):
        response = client.post(
            "/finetune/dataset/validate",
            json={
                "format": "prompt-completion",
                "examples": [{"prompt": "续写", "completion": "结果"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["valid"] == 1

    def test_issues_are_capped(self, client):
        bad = [{"instruction": "没有问题"}] * (MAX_DATASET_ISSUES + 5)
        response = client.post(
            "/finetune/dataset/validate", json={"format": "alpaca", "examples": bad}
        )
        body = response.json()
        assert body["invalid"] == MAX_DATASET_ISSUES + 5
        assert len(body["issues"]) == MAX_DATASET_ISSUES

    def test_stats_fields_are_complete(self, client):
        stats = client.post(
            "/finetune/dataset/validate",
            json={"format": "alpaca", "examples": [VALID_ALPACA, VALID_ALPACA]},
        ).json()["stats"]
        assert set(stats) == {
            "count",
            "avg_instruction_chars",
            "avg_output_chars",
            "min_output_chars",
            "max_output_chars",
            "avg_estimated_tokens",
            "source_distribution",
            "tag_distribution",
        }
        assert stats["count"] == 2


class TestDatasetStatsEndpoint:
    """GET /finetune/dataset/stats：默认采集器的数据集画像."""

    def test_returns_stats_from_repository_data(self, client, repo_data):
        response = client.get("/finetune/dataset/stats")
        assert response.status_code == 200
        stats = response.json()["stats"]
        assert stats["count"] > 0
        assert stats["source_distribution"]["seed"] >= 14
        assert stats["source_distribution"]["eval/agent_tasks"] == 5
        assert stats["min_output_chars"] > 0

    def test_stats_is_read_only_and_repeatable(self, client, repo_data):
        first = client.get("/finetune/dataset/stats").json()
        second = client.get("/finetune/dataset/stats").json()
        assert first == second
