"""LoRA 多卡与部署端点测试（day052）：/finetune/lora/distributed、/finetune/lora/deploy.

全部走 ``TestClient``（进程内 ASGI 调用），不起真实服务、不联网、不训练——
与 ``test_lora_api.py``（day051 的三个只读端点）同一套做法，``client`` 同样带
``raise_server_exceptions=False``：万一端点抛异常，测试能观测到**响应体**
（``error_type`` + ``detail``）而不是让异常直接冒进测试里。

两个端点的性格不同，测试的关注点也不同：

- ``/finetune/lora/distributed`` 是纯算术的只读端点：它算全局批被放大多少倍、
  每设备显存降了多少、通信量涨了多少，并生成 accelerate / DeepSpeed 配置。
  "省了几倍""通信量差多少"这类断言一律按**实测值**写（数字记在各用例的
  docstring 里），因为它们的口径来自 ``peft/accelerate.py`` 的分片规则；
- ``/finetune/lora/deploy`` 是本课唯一**会写磁盘**的端点，但它写的是
  ``tempfile`` 临时目录（请求结束即删除），所以测试仍然离线、不残留文件。
  它只支持参考模型的 ``bigram`` 适配器，而请求体的 LoRA 预设缺省是
  ``attention``——**任何不带 ``overrides`` 的请求都会 400**（本文件有专门用例
  钉住这条）。这不是 bug，而是"用参考模型演示部署流程"的显式边界。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.peft.trainer import ADAPTER_FILES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 参考模型（真实数据集 V=557、r=8）的适配器参数量：``r × (V + V)``
REFERENCE_TRAINABLE_AT_R8 = 8912

#: llama-2-7b 缺省 LoRA 预设（attention：q/v）下，适配器参数随秩的斜率.
#: 每层 ``2·r·4096 × 2`` 个投影 = ``r×16384``，32 层即 ``r×524288``。
ADAPTER_PARAMETERS_PER_RANK = 524288


@pytest.fixture
def client() -> TestClient:
    """离线客户端：注入 MockLLM，其余依赖走 create_app 默认装配."""
    return TestClient(create_app(llm=MockLLM()), raise_server_exceptions=False)


@pytest.fixture
def repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """把数据目录指向仓库绝对路径，使测试不依赖进程工作目录.

    两个端点都要 ``build_dataset()``（缺省 ``train_size`` 就是真实训练集条数，
    实测 30 条），所以数据目录必须是仓库里的那一份。
    """
    finetune_dir = PROJECT_ROOT / "data" / "finetune"
    monkeypatch.setattr(settings, "finetune_data_dir", str(finetune_dir))
    monkeypatch.setattr(settings, "finetune_seed_path", str(finetune_dir / "seed_examples.jsonl"))


class TestLoRADistributedEndpoint:
    """POST /finetune/lora/distributed：多卡步数、每设备显存、通信量与配置生成.

    实测基线（缺省请求体 = llama-2-7b / devices=4 / ddp / bf16、训练集 30 条）：
    ``per_device=2`` / ``accum=4`` → 全局批 32，每设备 12.5982 GiB，
    ``steps_per_epoch=0``（全局批超过训练集，端点会因此给出一条告警）。
    """

    def test_default_body_builds_the_ddp_plan(self, client, repo_data):
        """缺省请求体：全局批 = per_device × accum × devices，YAML 能解析回字典.

        实测 ``steps_per_epoch`` 是 **0**：``30 // (2×4) // 4 = 0``——全局批
        32 已超过训练集 30 条。这不是要修的东西（``warnings`` 里有一条专门的
        告警），所以断言按公式算，不写死数字。
        """
        response = client.post("/finetune/lora/distributed", json={})
        assert response.status_code == 200
        payload = response.json()
        plan = payload["plan"]
        assert payload["model"]["name"] == "llama-2-7b"
        assert plan["devices"] == 4
        assert plan["strategy"] == "ddp"
        assert plan["train_size"] == 30

        per_device = plan["per_device_train_batch_size"]
        accum = plan["gradient_accumulation_steps"]
        assert plan["global_batch_size"] == per_device * accum * plan["devices"]
        assert plan["steps_per_epoch"] == (
            plan["train_size"] // (per_device * plan["devices"]) // accum
        )

        assert yaml.safe_load(payload["config_yaml"]) == payload["accelerate_config"]
        assert payload["accelerate_config"]["num_processes"] == 4
        assert payload["launch_command"].startswith("accelerate launch")
        assert payload["deepspeed_config"] is None

        device_fit = payload["device_fit"]
        assert device_fit["fits"] is True
        assert device_fit["budget_gib"] == 24.0
        assert device_fit["note"] == "只比较权重 + 梯度 + 优化器状态，不含激活显存"

    def test_zero_strategies_emit_deepspeed_config(self, client, repo_data):
        """zero2 / zero3 带 ZeRO 配置、ddp 不带；stage 与 all-gather 开关逐项核对.

        实测：``ddp`` 的 ``deepspeed_config`` 为 None；``zero2`` / ``zero3`` 分别
        给出 ``zero_optimization.stage`` 2 / 3，且只有 stage 3 带
        ``stage3_gather_16bit_weights_on_model_save``（分片的权重落盘前必须聚合）。
        """
        ddp = client.post("/finetune/lora/distributed", json={"strategy": "ddp"}).json()
        zero2 = client.post("/finetune/lora/distributed", json={"strategy": "zero2"}).json()
        zero3 = client.post("/finetune/lora/distributed", json={"strategy": "zero3"}).json()

        assert ddp["deepspeed_config"] is None
        zero2_optimization = zero2["deepspeed_config"]["zero_optimization"]
        zero3_optimization = zero3["deepspeed_config"]["zero_optimization"]
        assert zero2_optimization["stage"] == 2
        assert zero3_optimization["stage"] == 3
        assert zero3_optimization["stage3_gather_16bit_weights_on_model_save"] is True
        assert "stage3_gather_16bit_weights_on_model_save" not in zero2_optimization

        # accelerate 那份配置与 DeepSpeed 那份同源：distributed_type 与 init 开关都跟着策略走
        assert zero2["accelerate_config"]["distributed_type"] == "DEEPSPEED"
        assert ddp["accelerate_config"]["distributed_type"] == "MULTI_GPU"
        assert zero2["accelerate_config"]["deepspeed_config"]["zero3_init_flag"] is False
        assert zero3["accelerate_config"]["deepspeed_config"]["zero3_init_flag"] is True

    def test_lora_zero2_barely_saves_memory(self, client, repo_data):
        """LoRA 场景下 zero2 几乎不省：实测 12.5982 → 12.5689 GiB（1.0023 倍）.

        原因在分片规则：zero2 只切优化器状态与梯度，而 LoRA 的优化器状态只有
        适配器那一百来 MB，占大头的**冻结基座权重**根本不切。
        """
        ddp = client.post("/finetune/lora/distributed", json={"strategy": "ddp"}).json()
        zero2 = client.post("/finetune/lora/distributed", json={"strategy": "zero2"}).json()
        ddp_gib = ddp["plan"]["per_device_gib"]
        zero2_gib = zero2["plan"]["per_device_gib"]

        assert zero2_gib == pytest.approx(ddp_gib, abs=0.05)
        assert ddp_gib / zero2_gib < 1.05
        assert zero2["plan"]["memory_saving_factor"] < 1.05
        assert zero2["plan"]["single_device_gib"] == pytest.approx(ddp_gib)

    def test_lora_zero3_quarters_per_device_memory(self, client, repo_data):
        """LoRA 场景下 zero3 约为 ddp 的 1/4：实测 12.5982 → 3.1495 GiB（正好 4.0 倍）.

        权重、梯度、优化器状态都按 4 个设备分片，所以倍数恰好等于设备数——
        这是"只有切参数才真正省 LoRA 的显存"的量化形式。
        """
        ddp = client.post("/finetune/lora/distributed", json={"strategy": "ddp"}).json()
        zero3 = client.post("/finetune/lora/distributed", json={"strategy": "zero3"}).json()

        assert zero3["plan"]["per_device_gib"] == pytest.approx(
            ddp["plan"]["per_device_gib"] / 4
        )
        assert zero3["plan"]["memory_saving_factor"] == pytest.approx(4.0)

    def test_full_finetune_zero2_shrinks_memory(self, client, repo_data):
        """全参微调时 zero2 明显省：实测 75.3077 → 28.2404 GiB（2.6667 倍）.

        同一个策略在两种场景下的结论相反，正是这个端点要暴露的事：
        "zero2 有用吗"取决于优化器状态在预算里占多大比重。
        """
        ddp = client.post(
            "/finetune/lora/distributed", json={"strategy": "ddp", "full_finetune": True}
        ).json()
        zero2 = client.post(
            "/finetune/lora/distributed", json={"strategy": "zero2", "full_finetune": True}
        ).json()
        ddp_gib = ddp["plan"]["per_device_gib"]
        zero2_gib = zero2["plan"]["per_device_gib"]

        assert zero2_gib < ddp_gib
        assert ddp_gib / zero2_gib > 2.5

    def test_lora_zero3_communication_dominates(self, client, repo_data):
        """LoRA + zero3 的每步通信量比 zero2 大三个数量级以上（实测约 2143 倍）.

        zero2 只 all-reduce 适配器的梯度（实测 12 582 912 B），zero3 还要在每层
        前向/反向 all-gather **整个基座**（67.38 亿参数，实测 26 966 245 376 B）
        ——"省下的显存是拿通信量换的"这句话在这里有具体倍数。
        """
        zero2_bytes = client.post(
            "/finetune/lora/distributed", json={"strategy": "zero2"}
        ).json()["plan"]["communication_bytes_per_step"]
        zero3_bytes = client.post(
            "/finetune/lora/distributed", json={"strategy": "zero3"}
        ).json()["plan"]["communication_bytes_per_step"]

        assert zero2_bytes == 12582912
        assert zero3_bytes == 26966245376
        assert zero3_bytes > 1000 * zero2_bytes

    def test_unknown_model_returns_400(self, client):
        response = client.post("/finetune/lora/distributed", json={"model": "nope"})
        assert response.status_code == 400
        assert "未知的模型规格" in response.json()["detail"]

    def test_unknown_strategy_returns_400(self, client, repo_data):
        response = client.post("/finetune/lora/distributed", json={"strategy": "zero4"})
        assert response.status_code == 400
        assert "未知的分片策略" in response.json()["detail"]

    def test_unknown_mixed_precision_returns_400(self, client, repo_data):
        """``int8`` 不是混合精度取值（可选 no / fp16 / bf16）→ 400."""
        response = client.post(
            "/finetune/lora/distributed", json={"mixed_precision": "int8"}
        )
        assert response.status_code == 400
        assert "未知的混合精度" in response.json()["detail"]

    def test_zero_devices_returns_422(self, client):
        """``devices=0`` 由 Pydantic 的 ``ge=1`` 在进路由前拦下 → **422**（实测）.

        端点 docstring 说"配置非法 → 400"，但 ``devices`` 是带 ``ge=1, le=1024``
        的 Pydantic 字段，根本走不到业务层的 400。这里以实测为准钉住，免得后来者
        照文档写成 400 的断言。
        """
        response = client.post("/finetune/lora/distributed", json={"devices": 0})
        assert response.status_code == 422

    def test_devices_string_returns_422(self, client):
        """``devices`` 传字符串：Pydantic 先拦下（422），不进业务逻辑."""
        response = client.post("/finetune/lora/distributed", json={"devices": "four"})
        assert response.status_code == 422

    def test_train_size_is_respected(self, client, repo_data):
        """显式 ``train_size`` 直接进 plan，步数随之变化（实测 128 → 4 步/epoch）."""
        plan = client.post(
            "/finetune/lora/distributed", json={"train_size": 128}
        ).json()["plan"]
        assert plan["train_size"] == 128
        assert plan["micro_batches_per_epoch"] == 16
        assert plan["steps_per_epoch"] == 4
        assert plan["total_steps"] == 40

    def test_rank_override_doubles_adapter_parameters(self, client, repo_data):
        """``overrides={"r": 16}``：适配器参数随 r 线性翻倍（实测 4 194 304 → 8 388 608）.

        实测口径与正文的一处差异：``plan`` 里**没有** ``trainable_parameters``
        字段，只有 ``checkpoint_bytes``——它是"可训练参数 × 每参数字节数"
        （bf16 = 2 字节），所以可训练参数要由它还原。本端点的 LoRA 参数走
        ``plan_lora`` 的 llama-2-7b 解码器规格（缺省 attention 预设、32 层），
        即 **r × 524 288**；正文里的 ``r × (V + V)``（V=557 的 bigram 口径）
        属于参考模型，只在 ``/finetune/lora/deploy`` 的 ``reference`` 里成立
        （见 ``TestLoRADeployEndpoint.test_reference_accounting_is_linear_in_rank``）。
        """
        narrow = client.post(
            "/finetune/lora/distributed", json={"overrides": {"r": 8}}
        ).json()["plan"]
        wide = client.post(
            "/finetune/lora/distributed", json={"overrides": {"r": 16}}
        ).json()["plan"]
        trained_narrow = narrow["checkpoint_bytes"] // 2
        trained_wide = wide["checkpoint_bytes"] // 2

        assert trained_narrow == 8 * ADAPTER_PARAMETERS_PER_RANK
        assert trained_wide == 16 * ADAPTER_PARAMETERS_PER_RANK
        assert trained_wide == 2 * trained_narrow
        # 通信量按"需要梯度的参数量"算，因此也随 r 线性翻倍
        assert wide["communication_bytes_per_step"] == 2 * narrow["communication_bytes_per_step"]


class TestLoRADeployEndpoint:
    """POST /finetune/lora/deploy：落盘适配器 → 清单 → 合并验证 → 体积对照.

    实测缺省（``train_steps=0``）：适配器 192 645 B、合并验证逐位一致、
    最大 logits 差 ``0.0``、参考模型 loss 6.322982；``train_steps=4`` 时
    loss 降到 6.314745。请求必须显式带 ``overrides={"target_modules": "bigram"}``
    ——缺省预设 attention 会被端点以 400 拒绝（本类有一条用例钉住）。
    """

    def test_untrained_adapter_passes_merge_verification(self, client, repo_data):
        """``train_steps=0``：ΔW=0，合并前后逐位一致，端点如实标注"未训练".

        ``base_model_bytes`` 是"规格参数量 × 16 bit / 8"（实测 13 476 831 232 B，
        即 12.55 GiB），与适配器那 188 KiB 放在一起才看得出部署形态的差别。
        """
        response = client.post(
            "/finetune/lora/deploy",
            json={"model": "llama-2-7b", "overrides": {"target_modules": "bigram"}},
        )
        assert response.status_code == 200
        payload = response.json()

        assert payload["adapter_trained"] is False
        verification = payload["verification"]
        assert verification["passed"] is True
        assert verification["bitwise_identical"] is True
        assert verification["max_logit_difference"] == 0.0

        assert len(payload["manifest"]["content_sha256"]) == 64
        assert payload["manifest"]["base_model"] == "Qwen/Qwen3-0.6B"
        assert payload["adapter_files"] == [
            "adapter_config.json",
            "adapter_model.json",
            "training_state.json",
        ]
        assert payload["adapter_files"] == list(ADAPTER_FILES)

        assert payload["deployment"]["base_model_bytes"] == 6738415616 * 16 // 8
        assert len(payload["registry"]) == 1
        assert payload["merge_script_lines"] > 30
        assert payload["inference_script_lines"] > 30

    def test_trained_adapter_reports_reference_loss(self, client, repo_data):
        """``train_steps=4``：适配器真的更新过，loss 落在参考模型的量级（实测 6.3147）.

        未训练的参考模型 loss 就在 6.32 附近（随机初始化的 bigram + 交叉熵），
        走 4 步只会把它压到 6.3147——这个量级本身就是"参考模型是玩具规模"的
        证据，所以断言写成区间而不是精确值。
        """
        response = client.post(
            "/finetune/lora/deploy",
            json={
                "model": "llama-2-7b",
                "train_steps": 4,
                "overrides": {"target_modules": "bigram"},
            },
        )
        assert response.status_code == 200
        payload = response.json()

        assert payload["adapter_trained"] is True
        assert payload["verification"]["passed"] is True
        assert 6.0 <= payload["verification"]["adapter_loss"] <= 6.4
        assert payload["registry"][0]["step"] == 4
        assert payload["manifest"]["step"] == 4

    def test_reference_accounting_is_linear_in_rank(self, client, repo_data):
        """参考模型的适配器参数 = ``r × (V + V)``：实测 V=557、r=8 → 8 912、r=16 → 17 824.

        这条公式是 ``reference_lora_accounting`` 的口径（``A`` 是 r×V、``B`` 是
        V×r），与 7B 解码器规格的 ``plan_lora`` 是两套账——端点把参考模型的账
        单独算，正是为了避免两套规格硬接（day051 为此踩过 500 的坑）。
        """
        narrow = client.post(
            "/finetune/lora/deploy",
            json={"overrides": {"target_modules": "bigram"}},
        ).json()
        wide = client.post(
            "/finetune/lora/deploy",
            json={"overrides": {"target_modules": "bigram", "r": 16}},
        ).json()

        assert narrow["reference"]["vocab_size"] == 557
        assert narrow["reference"]["trainable_parameters"] == REFERENCE_TRAINABLE_AT_R8
        assert narrow["reference"]["trainable_parameters"] == 8 * 2 * 557
        assert wide["reference"]["trainable_parameters"] == 16 * 2 * 557

    def test_unknown_model_returns_400(self, client):
        response = client.post("/finetune/lora/deploy", json={"model": "nope"})
        assert response.status_code == 400
        assert "未知的模型规格" in response.json()["detail"]

    def test_attention_target_returns_400(self, client, repo_data):
        """显式传 attention 预设 → 400；缺省请求体 → 200（预设被置为 bigram）.

        这条区分很关键：本端点用**参考模型的 bigram 适配器**演示部署流程，
        因此它在缺省时会把 ``target_modules`` 设为 ``bigram``——否则"不带
        overrides 的 deploy 请求"会因为全局缺省预设是 ``attention`` 而必然
        400，与"不传参数就能走完整流程"的意图自相矛盾。而**显式**传 attention
        仍然必须被拒绝：那是真实的配置错误，不该被悄悄改写。
        """
        response = client.post(
            "/finetune/lora/deploy", json={"overrides": {"target_modules": "attention"}}
        )
        assert response.status_code == 400
        assert "bigram" in response.json()["detail"]

        default_body = client.post("/finetune/lora/deploy", json={})
        assert default_body.status_code == 200
        assert default_body.json()["reference"]["targets"] == ["weight"]

    def test_train_steps_out_of_range_returns_422(self, client):
        """``train_steps`` 的 ``ge=0, le=200`` 由 Pydantic 拦下 → 422（实测）."""
        for steps in (-1, 1000):
            response = client.post("/finetune/lora/deploy", json={"train_steps": steps})
            assert response.status_code == 422

    def test_tags_are_written_into_manifest(self, client, repo_data):
        """``tags`` 原样进清单与注册表——"这个适配器是给哪个环境发布的"要能查到."""
        response = client.post(
            "/finetune/lora/deploy",
            json={
                "overrides": {"target_modules": "bigram"},
                "tags": {"env": "staging"},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["manifest"]["tags"] == {"env": "staging"}
        assert payload["registry"][0]["tags"] == {"env": "staging"}
