"""LoRA / QLoRA 端点测试（day051）：/finetune/lora/defaults、/plan、/memory.

全部走 ``TestClient``（进程内 ASGI 调用），不起真实服务、不联网、不训练。
三个端点都是**只读**的：真正的训练由生成的 ``peft`` / ``bitsandbytes``
脚本承担，HTTP 端点只回答"这次微调训多少参数、占多少显存"。

与 ``test_sft_api.py`` 的差别只有一个：``client`` 用
``raise_server_exceptions=False``，这样万一某个端点抛异常，测试能观测到
**响应体**（``error_type`` + ``detail``）而不是让异常直接冒进测试里——
"500 长什么样"本身也是一条需要被钉住的契约。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.routes import lora_config_from_settings, qlora_config_from_settings
from smart_research_agent.config import settings
from smart_research_agent.finetune.dataset import build_dataset
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.peft.config import LORA_TARGET_PRESETS, quantization_table
from smart_research_agent.peft.hf_script import (
    PEFT_DEPENDENCIES,
    QLORA_DEPENDENCIES,
    peft_dependency_commands,
)
from smart_research_agent.peft.memory import STRATEGIES
from smart_research_agent.peft.models import (
    default_reference_lora_config,
    reference_lora_accounting,
)
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.hf_script import DEFAULT_BASE_MODEL
from smart_research_agent.sft.template import render_supervised_list

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 真实数据集（day048 落盘的 30 + 7 条样本渲染）的字符级词表大小
REFERENCE_VOCAB_SIZE = 557


@pytest.fixture
def client() -> TestClient:
    """离线客户端：注入 MockLLM，其余依赖走 create_app 默认装配."""
    return TestClient(create_app(llm=MockLLM()), raise_server_exceptions=False)


@pytest.fixture
def repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """把数据目录指向仓库绝对路径，使测试不依赖进程工作目录."""
    finetune_dir = PROJECT_ROOT / "data" / "finetune"
    monkeypatch.setattr(settings, "finetune_data_dir", str(finetune_dir))
    monkeypatch.setattr(settings, "finetune_seed_path", str(finetune_dir / "seed_examples.jsonl"))


def reference_bigram_vocab_size() -> int:
    """按 ``/finetune/lora/defaults`` 的口径复算参考模型的词表大小."""
    bundle = build_dataset()
    rendered = render_supervised_list(bundle.examples)
    return CharTokenizer.from_texts([item.text for item in rendered]).vocab_size


class TestLoRADefaultsEndpoint:
    """GET /finetune/lora/defaults：默认配置、量化派生量与参考模型计划.

    参考模型的参数量**不走** ``peft.targets.plan_lora``——后者面向解码器规格
    （``q_proj`` 等七个投影），而参考模型只有一个叫 ``weight`` 的 bigram
    矩阵。两者硬接上会命中"目标模块未匹配"的校验（本课实现时真的踩到过：
    端点返回 500，而单元测试全绿，因为测试只覆盖了 helper、没有覆盖端点的
    组合方式）。修好之后这里同时断言**端点的响应**与**helper 的直接输出**，
    两者必须一致——否则将来任何一侧改口径都会被立刻发现。

    本类的断言分两层：

    1. 端点的 HTTP 行为（200 + 响应体字段）；
    2. 端点自身调用的同一批 helper（``lora_config_from_settings`` /
       ``qlora_config_from_settings`` / ``quantization_table`` /
       ``LORA_TARGET_PRESETS`` / ``PEFT_DEPENDENCIES`` /
       ``peft_dependency_commands`` / ``DEFAULT_BASE_MODEL``）逐条锁定
       仪表盘上的字段值。
    """

    def test_returns_200_with_reference_accounting(self, client, repo_data):
        """端点必须返回 200，且参考模型的账由 ``reference_lora_accounting`` 给出."""
        response = client.get("/finetune/lora/defaults")
        assert response.status_code == 200
        payload = response.json()
        expected = reference_lora_accounting(
            REFERENCE_VOCAB_SIZE,
            default_reference_lora_config(
                r=settings.lora_r,
                lora_alpha=settings.lora_alpha,
                lora_dropout=0.0,
            ),
        )
        assert payload["reference_model"]["trainable_parameters"] == 8912
        assert payload["reference_model"]["frozen_parameters"] == 310806
        assert payload["reference_model"]["total_parameters"] == 319718
        assert payload["reference_model"] == expected
        assert payload["reference_model"]["targets"] == ["weight"]

    def test_lora_defaults_use_peft_field_names(self):
        """``lora`` 的键名与 ``peft.LoraConfig`` 逐字对齐，取值来自 settings."""
        payload = lora_config_from_settings().to_peft_dict()
        for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "use_rslora"):
            assert key in payload
        assert payload["r"] == settings.lora_r
        assert payload["lora_alpha"] == settings.lora_alpha
        assert payload["lora_dropout"] == settings.lora_dropout
        assert payload["target_modules"] == ["q_proj", "v_proj"]
        assert payload["bias"] == "none"
        assert payload["use_rslora"] is False
        assert payload["task_type"] == "CAUSAL_LM"

    def test_scaling_and_formula(self):
        """alpha=16 / r=8 → 缩放 2.0，公式串里同时给出分子与分母."""
        config = lora_config_from_settings()
        assert config.scaling == 2.0
        assert config.scaling_formula == "alpha/r = 16/8 = 2.000000"

    def test_quantization_config_matches_bitsandbytes(self):
        """``quantization_config``：4-bit 基座 / nf4 / 二级量化 / bf16 计算."""
        payload = qlora_config_from_settings().to_bnb_dict()
        assert payload["load_in_4bit"] is True
        assert payload["bnb_4bit_quant_type"] == "nf4"
        assert payload["bnb_4bit_use_double_quant"] is True
        assert payload["bnb_4bit_compute_dtype"] == "bfloat16"

    def test_quantization_bytes_per_parameter(self):
        """nf4 + 二级量化 + block=64 → 4.126953 bit = 0.515869 字节/参数."""
        payload = qlora_config_from_settings().to_dict()
        assert payload["block_size"] == 64
        assert payload["double_quant_block_size"] == 256
        assert payload["bits_per_parameter"] == pytest.approx(4.126953, abs=1e-6)
        assert payload["bytes_per_parameter"] == pytest.approx(0.515869, abs=1e-6)

    def test_quantization_block_table(self):
        """块大小对照表 4 行，``block_size=64`` 那行就是 0.515869 字节/参数."""
        rows = quantization_table()
        assert len(rows) == 4
        assert [row["block_size"] for row in rows] == [64, 128, 256, 512]
        by_size = {row["block_size"]: row for row in rows}
        assert by_size[64]["bytes_per_parameter"] == 0.515869
        assert by_size[64]["bytes_per_parameter_single_quant"] == 0.5625

    def test_target_presets_include_documented_names(self):
        """五种预设都要在表里：attention 是 q/v，bigram 是参考模型的 weight."""
        for name in ("attention", "attention_all", "mlp", "all_linear", "bigram"):
            assert name in LORA_TARGET_PRESETS
        assert LORA_TARGET_PRESETS["attention"] == ("q_proj", "v_proj")
        assert LORA_TARGET_PRESETS["bigram"] == ("weight",)
        assert LORA_TARGET_PRESETS["all_linear"] == (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    def test_reference_model_numbers_on_real_dataset(self, repo_data):
        """参考模型（真实数据集 V=557、r=8）：可训练 8912，占比约 2.7875%."""
        vocab_size = reference_bigram_vocab_size()
        lora = lora_config_from_settings()
        config = default_reference_lora_config(
            r=lora.r, lora_alpha=lora.lora_alpha, lora_dropout=0.0
        )
        accounting = reference_lora_accounting(vocab_size, config)
        assert vocab_size == REFERENCE_VOCAB_SIZE
        assert config.resolved_targets == ("weight",)
        assert accounting["trainable_parameters"] == 8912
        # 冻结部分 = bigram 权重矩阵（V×V）+ 冻结偏置（V）
        assert accounting["frozen_parameters"] == vocab_size * vocab_size + vocab_size
        assert accounting["trainable_ratio"] == pytest.approx(0.027875, abs=1e-6)

    def test_dependency_lists_and_install_commands(self):
        """依赖清单含 ``peft>=0.20`` / bitsandbytes，安装命令给出 pip 与 uv."""
        assert any("peft>=0.20" in item for item in PEFT_DEPENDENCIES)
        assert any("bitsandbytes" in item for item in QLORA_DEPENDENCIES)
        install = peft_dependency_commands(use_qlora=True)
        assert install["pip"].startswith("pip install ")
        assert install["uv"].startswith("uv pip install ")
        assert "peft>=0.20" in install["pip"]

    def test_base_model_is_not_empty(self):
        """``base_model`` 是生成脚本的缺省基座，不能为空."""
        assert DEFAULT_BASE_MODEL == "Qwen/Qwen3-0.6B"


class TestLoRAPlanEndpoint:
    """POST /finetune/lora/plan：参数量、预设/秩对照与风险告警."""

    def test_empty_body_defaults_to_llama_2_7b(self, client):
        """空请求体：model 缺省 llama-2-7b，规格自检值 6 738 415 616."""
        payload = client.post("/finetune/lora/plan", json={}).json()
        assert payload["model"]["name"] == "llama-2-7b"
        assert payload["model"]["total_parameters"] == 6738415616
        assert payload["plan"]["base_parameters"] == 6738415616

    def test_default_preset_is_attention(self, client):
        """实测：配置缺省 ``lora_target_preset="attention"``，可训练 4 194 304.

        正文里出现的 8 388 608 属于 ``attention_all``（q/k/v/o）预设，不是空
        请求体的缺省值——缺省取值必须实测，不能按正文猜。
        """
        plan = client.post("/finetune/lora/plan", json={}).json()["plan"]
        assert plan["targets"] == ["q_proj", "v_proj"]
        assert plan["trainable_parameters"] == 4194304
        assert plan["adapter_parameters_per_layer"] == 131072

    def test_attention_all_preset_trainable_parameters(self, client):
        """显式切到 attention_all 才是 8 388 608（q/v 预设的 2 倍）."""
        plan = client.post(
            "/finetune/lora/plan", json={"overrides": {"target_modules": "attention_all"}}
        ).json()["plan"]
        assert plan["targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert plan["trainable_parameters"] == 8388608

    def test_presets_are_monotonic(self, client):
        """四个预设按参数量严格递增，attention 最小、all_linear 最大."""
        presets = client.post("/finetune/lora/plan", json={}).json()["presets"]
        assert [row["preset"] for row in presets] == [
            "attention",
            "attention_all",
            "mlp",
            "all_linear",
        ]
        counts = [row["adapter_parameters"] for row in presets]
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)
        assert counts[0] == 4194304
        assert counts[-1] == 19988480

    def test_rank_table_is_linear_in_r(self, client):
        """秩对照表对 r 严格线性：``adapter_parameters == r × 1 048 576``.

        实测提醒：表里的 r 是按 **2 的幂**采样的（1/2/4/…/256），所以
        "相邻两项之差"并不相等（差值本身随 r 翻倍）；线性体现为"参数量 ÷ r"
        这个斜率恒定（32 层 × 32 768 个参数 = 1 048 576）。
        """
        rows = client.post("/finetune/lora/plan", json={}).json()["rank_table"]
        assert len(rows) == 9
        ranks = [row["r"] for row in rows]
        counts = [row["adapter_parameters"] for row in rows]
        assert ranks == [1, 2, 4, 8, 16, 32, 64, 128, 256]
        assert counts == sorted(counts)
        assert all(count == rank * 1048576 for rank, count in zip(ranks, counts))

    def test_overrides_are_applied(self, client):
        """r 与 target_modules 同时覆盖：16 + all_linear → 39 976 960 可训练."""
        payload = client.post(
            "/finetune/lora/plan",
            json={"overrides": {"r": 16, "target_modules": "all_linear"}},
        ).json()
        assert payload["plan"]["r"] == 16
        assert payload["plan"]["trainable_parameters"] == 39976960
        assert len(payload["plan"]["targets"]) == 7

    def test_unknown_model_returns_400(self, client):
        response = client.post("/finetune/lora/plan", json={"model": "nope"})
        assert response.status_code == 400
        assert "未知的模型规格" in response.json()["detail"]

    def test_zero_rank_returns_400(self, client):
        """r=0 等于不训任何参数：在 validate() 处被拒，而不是静默通过."""
        response = client.post("/finetune/lora/plan", json={"overrides": {"r": 0}})
        assert response.status_code == 400
        assert "r 必须为正整数" in response.json()["detail"]

    def test_unknown_target_preset_returns_400(self, client):
        response = client.post(
            "/finetune/lora/plan", json={"overrides": {"target_modules": "nope"}}
        )
        assert response.status_code == 400
        assert "未知的 target_modules 预设" in response.json()["detail"]

    def test_unknown_override_keys_are_ignored(self, client):
        """overrides 里的未知键被 ``from_dict`` 丢掉，不影响缺省计算."""
        plan = client.post(
            "/finetune/lora/plan", json={"overrides": {"future_field": 123}}
        ).json()["plan"]
        assert "future_field" not in plan
        assert plan["trainable_parameters"] == 4194304

    def test_extra_request_fields_do_not_break(self, client):
        """请求体的未知字段（如 train_size）被 Pydantic 忽略，不会 500."""
        response = client.post("/finetune/lora/plan", json={"train_size": 12})
        assert response.status_code == 200
        assert response.json()["plan"]["trainable_parameters"] == 4194304

    def test_model_type_error_returns_422(self, client):
        """model 传数字：Pydantic 先拦下（422），不进业务逻辑."""
        response = client.post("/finetune/lora/plan", json={"model": 123})
        assert response.status_code == 422


class TestLoRAMemoryEndpoint:
    """POST /finetune/lora/memory：全参 / LoRA / QLoRA 三种策略的显存预算."""

    def test_returns_200_with_three_strategies(self, client):
        """full / lora / qlora 三项齐全：full 是基准，qlora 省得最多."""
        response = client.post("/finetune/lora/memory", json={})
        assert response.status_code == 200
        payload = response.json()
        assert [row["strategy"] for row in payload["strategies"]] == list(STRATEGIES)
        savings = {row["strategy"]: row["savings_factor"] for row in payload["strategies"]}
        assert savings["full"] == 1.0
        assert savings["qlora"] == max(savings.values())
        assert savings["lora"] == pytest.approx(5.9777, abs=1e-4)
        assert savings["qlora"] == pytest.approx(22.9297, abs=1e-4)

    def test_details_total_equals_six_items(self, client):
        """每份明细的总字节数 = 六项分项之和（基座/适配器 × 权/梯度/优化器）."""
        details = client.post("/finetune/lora/memory", json={}).json()["details"]
        assert len(details) == 3
        for detail in details:
            items = detail["items"]
            assert len(items) == 6
            assert sum(items.values()) == detail["total_bytes"]

    def test_devices_report_fits_and_note(self, client):
        """单卡适配表每项都要有 fits，以及「不含激活显存」的说明."""
        devices = client.post("/finetune/lora/memory", json={}).json()["devices"]
        assert len(devices) == 4
        for row in devices:
            assert set(row["fits"]) == set(STRATEGIES)
            assert row["note"]
            assert all(isinstance(value, bool) for value in row["fits"].values())

    def test_unknown_optimizer_returns_400(self, client):
        response = client.post("/finetune/lora/memory", json={"optimizer": "nope"})
        assert response.status_code == 400
        assert "未知的优化器" in response.json()["detail"]

    def test_unknown_compute_dtype_returns_400(self, client):
        response = client.post("/finetune/lora/memory", json={"compute_dtype": "nope"})
        assert response.status_code == 400
        assert "未知的计算精度" in response.json()["detail"]

    def test_unknown_model_returns_400(self, client):
        response = client.post("/finetune/lora/memory", json={"model": "nope"})
        assert response.status_code == 400
        assert "未知的模型规格" in response.json()["detail"]

    def test_baseline_only_quant_type_returns_400(self, client):
        """``qlora_overrides`` 传本课的对照基线口径（int4）必须 **400**.

        显存预算只关心位宽，因此它会**静默接受** int4 并给出与 nf4 完全
        相同的数字——而"对照基线能进生产配置"是个陷阱：``int4`` 只是本课
        用来回答"为什么是 NF4"的对照码本，bitsandbytes 并不认这个取值，
        一旦写进 ``BitsAndBytesConfig`` 就会在运行时报错。所以端点在
        配置阶段就把它拦下（与 ``to_bnb_dict()`` 的口径一致）。
        """
        response = client.post(
            "/finetune/lora/memory",
            json={"qlora_overrides": {"bnb_4bit_quant_type": "int4"}},
        )
        assert response.status_code == 400
        assert "对照基线" in response.json()["detail"]

    def test_unknown_quant_type_returns_400(self, client):
        response = client.post(
            "/finetune/lora/memory",
            json={"qlora_overrides": {"bnb_4bit_quant_type": "int2"}},
        )
        assert response.status_code == 400
        assert "未知的量化类型" in response.json()["detail"]

    def test_model_type_error_returns_422(self, client):
        response = client.post("/finetune/lora/memory", json={"model": 123})
        assert response.status_code == 422
