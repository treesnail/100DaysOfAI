"""SFT 端点测试（day050）：/finetune/sft/defaults、/plan、/preview.

全部走 ``TestClient``（进程内 ASGI 调用），不起真实服务、不联网、不训练。
三个端点都是**只读**的：训练本身由 ``scripts/sft_demo.py`` 与生成的
``transformers`` 脚本承担。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.routes import SFT_SUGGEST_QUANTILE
from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.sft.args import LR_SOFT_RANGE_REFERENCE
from smart_research_agent.sft.encoding import IGNORE_INDEX

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


class TestSFTDefaultsEndpoint:
    """GET /finetune/sft/defaults：默认超参、专属项与依赖."""

    def test_returns_200(self, client):
        assert client.get("/finetune/sft/defaults").status_code == 200

    def test_training_arguments_use_hf_field_names(self, client):
        payload = client.get("/finetune/sft/defaults").json()["training_arguments"]
        for key in (
            "output_dir",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_train_epochs",
            "lr_scheduler_type",
            "warmup_ratio",
            "eval_strategy",
            "seed",
        ):
            assert key in payload
        # 废弃名不得出现（transformers v4.46 起已移除）
        assert "evaluation_strategy" not in payload

    def test_sft_specific_fields(self, client):
        payload = client.get("/finetune/sft/defaults").json()
        assert payload["sft_specific"]["ignore_index"] == IGNORE_INDEX
        assert payload["sft_specific"]["max_length"] == settings.sft_max_length
        assert payload["sft_specific"]["truncation"] == "keep-answer"

    def test_defaults_come_from_settings(self, client):
        payload = client.get("/finetune/sft/defaults").json()["training_arguments"]
        assert payload["learning_rate"] == settings.sft_learning_rate
        assert payload["num_train_epochs"] == settings.sft_num_train_epochs
        assert payload["output_dir"] == settings.sft_output_dir

    def test_dependency_lists(self, client):
        payload = client.get("/finetune/sft/defaults").json()
        assert any("transformers>=5.16" in item for item in payload["hf_dependencies"])
        assert any("trl>=1.5" in item for item in payload["trl_dependencies"])
        assert payload["base_model"] == "Qwen/Qwen3-0.6B"

    def test_install_commands(self, client):
        payload = client.get("/finetune/sft/defaults").json()["install"]
        assert payload["pip"].startswith("pip install ")
        assert payload["uv"].startswith("uv pip install ")


class TestSFTPlanEndpoint:
    """POST /finetune/sft/plan：派生量与风险告警."""

    def test_empty_body_uses_real_dataset(self, client, repo_data):
        response = client.post("/finetune/sft/plan", json={})
        assert response.status_code == 200
        payload = response.json()
        assert payload["plan"]["train_size"] == 30
        assert payload["plan"]["eval_size"] == 7

    def test_default_plan_step_count(self, client, repo_data):
        """配置缺省是 10 个 epoch（参考模型量级），因此步数是 3 × 10 = 30."""
        plan = client.post("/finetune/sft/plan", json={}).json()["plan"]
        assert plan["steps_per_epoch"] == 3
        assert plan["total_steps"] == 30

    def test_day049_numbers_reproduced(self, client, repo_data):
        """把 epoch 覆盖回 day049 用的 3.0，接口必须复现那六个数字."""
        plan = client.post(
            "/finetune/sft/plan", json={"overrides": {"num_train_epochs": 3.0}}
        ).json()["plan"]
        assert plan["micro_batches_per_epoch"] == 15
        assert plan["steps_per_epoch"] == 3
        assert plan["total_steps"] == 9
        assert plan["steps_per_epoch_flushing"] == 4
        assert plan["total_steps_flushing"] == 12
        assert plan["warmup_steps"] == 0
        assert plan["effective_batch_size"] == 8

    def test_length_distribution_reported(self, client, repo_data):
        payload = client.post("/finetune/sft/plan", json={}).json()
        length = payload["length"]
        assert length["count"] == 37
        assert length["min"] <= length["p50"] <= length["p95"] <= length["max"]
        assert length["max"] == 320

    def test_suggested_max_length(self, client, repo_data):
        payload = client.post("/finetune/sft/plan", json={}).json()
        assert payload["suggested_max_length"] == 320
        assert payload["suggested_quantile"] == SFT_SUGGEST_QUANTILE
        assert payload["suggested_max_length"] % 32 == 0

    def test_warnings_present(self, client, repo_data):
        warnings = client.post("/finetune/sft/plan", json={}).json()["warnings"]
        joined = " ".join(warnings)
        assert "训练样本仅 30 条" in joined
        assert "两种口径的步数不同" in joined

    def test_step_count_warning_under_three_epochs(self, client, repo_data):
        """3 个 epoch 时总步数只有 9，必须触发"步数过少"告警."""
        warnings = client.post(
            "/finetune/sft/plan", json={"overrides": {"num_train_epochs": 3.0}}
        ).json()["warnings"]
        joined = " ".join(warnings)
        assert "总优化器步数仅 9 步" in joined
        assert "warmup" in joined and "取整为 0" in joined

    def test_default_learning_rate_within_reference_range(self, client, repo_data):
        """缺省学习率按参考模型标定，因此不应触发区间告警."""
        warnings = client.post("/finetune/sft/plan", json={}).json()["warnings"]
        low, high = LR_SOFT_RANGE_REFERENCE
        assert low <= settings.sft_learning_rate <= high
        assert not any("learning_rate" in message for message in warnings)

    def test_overrides_are_applied(self, client, repo_data):
        payload = client.post(
            "/finetune/sft/plan",
            json={"overrides": {"num_train_epochs": 6.0, "gradient_accumulation_steps": 1}},
        ).json()
        assert payload["training_arguments"]["num_train_epochs"] == 6.0
        assert payload["training_arguments"]["gradient_accumulation_steps"] == 1
        assert payload["plan"]["total_steps"] == 90

    def test_unknown_override_keys_ignored(self, client, repo_data):
        payload = client.post(
            "/finetune/sft/plan", json={"overrides": {"future_field": 123}}
        ).json()
        assert "future_field" not in payload["training_arguments"]

    def test_explicit_sizes_respected(self, client, repo_data):
        payload = client.post(
            "/finetune/sft/plan", json={"train_size": 100, "eval_size": 20}
        ).json()
        assert payload["plan"]["train_size"] == 100
        assert payload["plan"]["eval_size"] == 20

    def test_invalid_override_returns_400(self, client, repo_data):
        response = client.post(
            "/finetune/sft/plan",
            json={"overrides": {"per_device_train_batch_size": 0}},
        )
        assert response.status_code == 400
        assert "per_device_train_batch_size" in response.json()["detail"]

    def test_conflicting_precision_returns_400(self, client, repo_data):
        response = client.post(
            "/finetune/sft/plan", json={"overrides": {"bf16": True, "fp16": True}}
        )
        assert response.status_code == 400

    def test_unknown_scheduler_returns_400(self, client, repo_data):
        response = client.post(
            "/finetune/sft/plan", json={"overrides": {"lr_scheduler_type": "warp"}}
        )
        assert response.status_code == 400

    def test_zero_train_size_returns_422(self, client, repo_data):
        response = client.post("/finetune/sft/plan", json={"train_size": 0})
        assert response.status_code == 422

    def test_negative_eval_size_returns_422(self, client, repo_data):
        response = client.post("/finetune/sft/plan", json={"eval_size": -1})
        assert response.status_code == 422


class TestSFTPreviewEndpoint:
    """POST /finetune/sft/preview：监督切分与 label mask 的可视化."""

    def test_returns_200(self, client):
        response = client.post("/finetune/sft/preview", json={"example": VALID_ALPACA})
        assert response.status_code == 200

    def test_offsets_sum_to_total(self, client):
        payload = client.post("/finetune/sft/preview", json={"example": VALID_ALPACA}).json()
        assert payload["prompt_chars"] + payload["supervised_chars"] == payload["total_chars"]

    def test_prompt_is_fully_masked(self, client):
        """本端点最有价值的字段：label mask 到底生效了没有."""
        payload = client.post("/finetune/sft/preview", json={"example": VALID_ALPACA}).json()
        assert payload["prompt_fully_masked"] is True

    def test_token_counts_are_consistent(self, client):
        payload = client.post("/finetune/sft/preview", json={"example": VALID_ALPACA}).json()
        assert payload["prompt_tokens"] + payload["supervised_tokens"] == payload["total_tokens"]
        assert payload["masked_tokens"] == payload["prompt_tokens"]
        assert 0.0 < payload["supervised_ratio"] < 1.0

    def test_not_truncated_when_max_length_is_large(self, client):
        payload = client.post(
            "/finetune/sft/preview", json={"example": VALID_ALPACA, "max_length": 512}
        ).json()
        assert payload["truncated"] is False

    def test_supervision_starts_with_the_answer(self, client):
        payload = client.post("/finetune/sft/preview", json={"example": VALID_ALPACA}).json()
        assert payload["supervised_preview"].startswith(VALID_ALPACA["output"][:10])

    def test_text_preview_contains_chatml_markers(self, client):
        payload = client.post("/finetune/sft/preview", json={"example": VALID_ALPACA}).json()
        assert payload["template"] == "chatml"
        assert "<|im_start|>user" in payload["text_preview"]

    @pytest.mark.parametrize("template", ["chatml", "llama3", "plain"])
    def test_templates_supported(self, client, template):
        response = client.post(
            "/finetune/sft/preview", json={"example": VALID_ALPACA, "template": template}
        )
        assert response.status_code == 200
        assert response.json()["template"] == template

    def test_input_field_is_joined_with_blank_line(self, client):
        example = {**VALID_ALPACA, "input": "请用一句话回答"}
        payload = client.post("/finetune/sft/preview", json={"example": example}).json()
        assert "请用一句话回答" in payload["text_preview"]

    def test_unknown_template_returns_400(self, client):
        response = client.post(
            "/finetune/sft/preview", json={"example": VALID_ALPACA, "template": "nope"}
        )
        assert response.status_code == 400
        assert "不支持的模板" in response.json()["detail"]

    def test_missing_output_returns_400(self, client):
        response = client.post(
            "/finetune/sft/preview", json={"example": {"instruction": "只有问题"}}
        )
        assert response.status_code == 400

    def test_empty_output_returns_400(self, client):
        example = {**VALID_ALPACA, "output": "   "}
        response = client.post("/finetune/sft/preview", json={"example": example})
        assert response.status_code == 400

    def test_answer_too_long_for_max_length_returns_400(self, client):
        response = client.post(
            "/finetune/sft/preview", json={"example": VALID_ALPACA, "max_length": 2}
        )
        assert response.status_code == 400

    def test_missing_example_returns_422(self, client):
        assert client.post("/finetune/sft/preview", json={}).status_code == 422

    def test_max_length_below_minimum_returns_422(self, client):
        response = client.post(
            "/finetune/sft/preview", json={"example": VALID_ALPACA, "max_length": 1}
        )
        assert response.status_code == 422
