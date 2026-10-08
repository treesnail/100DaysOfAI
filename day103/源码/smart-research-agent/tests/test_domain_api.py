"""day057 领域数据端点测试（M5-D8）：/data/domain/dimensions、/augment、/run.

全部走 TestClient（进程内 ASGI 调用）：不起真实服务、不联网、**不写盘**、
不用随机数——三个端点都是只读或纯计算，因此同一个请求的响应逐字节可复现。

这个文件挑的是三件"会静默失效"的事来钉：

1. **对照表必须与代码同源**：维度表长度 = 5、五维权重之和 == 1.0、
   算子表与 ``augment_ops_table()`` 逐项一致、阶段顺序与 ``STAGE_ORDER``
   一致。文档与代码各写一份，漂移时没人会发现；
2. **``output`` 逐字不变**：增强只改 prompt 侧。这条不变式一旦破了，
   增强就从"换一种问法"变成"生成（可能错的）新答案"——而错答案会被
   训练当成金标准反复强化；
3. **参数非法必须响亮**：未知算子 → 400、非法 ``group_by`` → 400。
   "静默少生成一批样本"比报错危险得多。

真实数据只被 ``/data/domain/run`` 读取，且经 ``repo_data`` 把数据目录指向
仓库绝对路径，因此测试与"从哪个目录启动 pytest"无关。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.domain_data import (
    DEFAULT_OPS,
    STAGE_ORDER,
    augment_ops_table,
    default_weights,
    dimension_table,
)
from smart_research_agent.domain_data.quality import DIMENSIONS
from smart_research_agent.llm.mock import MockLLM

#: 项目根目录 = 本文件的上一级（因此测试与"从哪个目录启动 pytest"无关）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 两条**只差一点点**的原始样本（与 day048 的种子数据同一形状）。
#: 第二条的答案里刻意保留换行与全角标点：``output`` 逐字不变的断言
#: 必须能抓住"顺手把答案规范化了一下"这种改动。
ALPACA_EXAMPLES = [
    {
        "instruction": "什么是 RAG？",
        "input": "",
        "output": (
            "RAG 是检索增强生成：\n"
            "1) 先按问题检索相关片段；\n"
            "2) 再把片段拼进提示词交给模型生成答案。"
        ),
        "source": "seed",
        "tags": ["rag"],
        "license": "CC-BY-4.0",
    },
    {
        "instruction": "如何评估检索质量？",
        "input": "",
        "output": "看召回率与排序质量：先把标注好的相关问题集跑一遍，再看前 k 条里命中了多少。",
        "source": "eval/agent_tasks",
        "tags": ["rag", "eval"],
        "license": "CC-BY-4.0",
    },
]


@pytest.fixture
def client() -> TestClient:
    """离线客户端：注入 MockLLM，其余依赖走 create_app 默认装配."""
    return TestClient(create_app(llm=MockLLM()))


@pytest.fixture
def repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """把数据目录指向仓库绝对路径，使 /data/domain/run 不依赖进程工作目录.

    与 ``tests/test_finetune_api.py`` 的 ``repo_data`` 同一手法：
    ``default_collector`` 读的是 ``settings.finetune_data_dir``（相对路径），
    不固定住它，测试就会在"从别处启动 pytest"时以 500/空数据的形式失败。
    """
    finetune_dir = PROJECT_ROOT / "data" / "finetune"
    monkeypatch.setattr(settings, "finetune_data_dir", str(finetune_dir))
    monkeypatch.setattr(
        settings, "finetune_seed_path", str(finetune_dir / "seed_examples.jsonl")
    )


def augment_payload(client: TestClient, **overrides) -> dict:
    """POST /data/domain/augment 并断言 200，返回响应体."""
    body = {"examples": ALPACA_EXAMPLES, "ops": ["prefix", "constraint"], "max_per_example": 1}
    body.update(overrides)
    response = client.post("/data/domain/augment", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def run_payload(client: TestClient, **overrides) -> dict:
    """POST /data/domain/run 并断言 200，返回响应体."""
    response = client.post("/data/domain/run", json=overrides)
    assert response.status_code == 200, response.text
    return response.json()


def stage_of(body: dict, name: str) -> dict:
    """从清单的阶段账里按名字取一条阶段记录（拿不到就断言失败）."""
    stages = body["manifest"]["stages"]
    matched = [stage for stage in stages if stage["name"] == name]
    assert matched, f"清单里没有阶段 {name}：{[s['name'] for s in stages]}"
    return matched[0]


class TestOpenApiContract:
    """三个端点都要出现在 OpenAPI 文档里——接口存在但没进文档等于没交付."""

    def test_paths_are_documented(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/data/domain/dimensions" in paths
        assert "/data/domain/augment" in paths
        assert "/data/domain/run" in paths
        assert "get" in paths["/data/domain/dimensions"]
        assert "post" in paths["/data/domain/augment"]
        assert "post" in paths["/data/domain/run"]


class TestDomainDimensionsEndpoint:
    """GET /data/domain/dimensions：质量五维 + 增强算子 + 阶段顺序的对照表."""

    def test_returns_five_dimensions_in_fixed_order(self, client):
        """维度表长度 = 5，顺序与 ``DIMENSIONS`` 一致（顺序决定并列时的归因）."""
        body = client.get("/data/domain/dimensions").json()
        assert len(body["dimensions"]) == 5
        assert [row["dimension"] for row in body["dimensions"]] == list(DIMENSIONS)
        # 表里的权重列由 QualityWeights 现场读出，不是手写的数字
        assert body["dimensions"] == dimension_table()

    def test_weights_sum_to_one(self, client):
        """五维权重之和恒为 1.0：否则不同批次的分数不在同一尺度上，无法比较."""
        body = client.get("/data/domain/dimensions").json()
        weights = body["weights"]
        assert weights == default_weights().as_dict()
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_threshold_comes_from_settings(self, client):
        """门槛取自配置（部署层面可调），而不是写死在路由里."""
        body = client.get("/data/domain/dimensions").json()
        assert body["threshold"] == settings.domain_quality_threshold

    def test_augment_ops_table_matches_code(self, client):
        """算子表与 ``augment_ops_table()`` 一致，且 ``noise`` 在表但不在默认集里.

        ``noise`` 的这条断言是纪律三的可执行版本：它造成的质量下降恰好不在
        质量打分的五个维度内，因此只能靠"默认关闭 + 显式标签"治理。
        """
        body = client.get("/data/domain/dimensions").json()
        assert body["augment_ops"] == augment_ops_table()
        assert body["default_ops"] == list(DEFAULT_OPS)
        assert "noise" in [row["op"] for row in body["augment_ops"]]
        assert "noise" not in body["default_ops"]

    def test_stage_order_matches_code(self, client):
        """六阶段顺序即策略：端点交出来的顺序必须就是流水线执行的顺序."""
        body = client.get("/data/domain/dimensions").json()
        assert body["stage_order"] == list(STAGE_ORDER)
        assert body["stage_order"][0] == "clean"
        assert body["stage_order"][-1] == "freeze"

    def test_endpoint_is_pure_and_repeatable(self, client):
        """只读端点：两次请求的响应逐字节相同，且不产生任何文件."""
        first = client.get("/data/domain/dimensions")
        second = client.get("/data/domain/dimensions")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()


class TestDomainAugmentEndpoint:
    """POST /data/domain/augment：增强样本（output 逐字不动）+ 增强报告."""

    def test_returns_generated_examples(self, client):
        """两条原样本 × 每条 1 个 → 2 条增强样本，且都带 ``aug:*`` 标签."""
        body = augment_payload(client)
        assert body["inputs"] == 2
        assert body["generated"] == len(body["examples"])
        assert body["generated"] >= 1
        assert body["expanded"] == body["inputs"] + body["generated"]
        for example in body["examples"]:
            assert any(tag.startswith("aug:") for tag in example["tags"])

    def test_output_is_verbatim_identical(self, client):
        """**本端点最重要的一条不变式**：增强样本的 output 与输入逐字相同.

        换行与全角标点必须原样保留——"顺手规范化了一下答案"属于静默改写，
        它在报告里看不出来，却会让增强样本的参考答案与人工标注不一致。
        """
        body = augment_payload(client)
        assert len(body["examples"]) == 2
        assert {example["output"] for example in body["examples"]} == {
            example["output"] for example in ALPACA_EXAMPLES
        }
        assert any("\n" in example["output"] for example in body["examples"])

    def test_provenance_fields_are_inherited(self, client):
        """来源与许可证原样继承：增强不改变数据出处，否则治理链就断了."""
        body = augment_payload(client)
        assert {example["source"] for example in body["examples"]} == {
            "seed",
            "eval/agent_tasks",
        }
        assert all(example["license"] == "CC-BY-4.0" for example in body["examples"])

    def test_unknown_op_returns_400(self, client):
        """未知算子名 → 400，详情是 ``DomainDataError`` 的原文（不静默少产样本）."""
        response = client.post(
            "/data/domain/augment",
            json={"examples": ALPACA_EXAMPLES, "ops": ["prefix", "shuffle"]},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "未知的增强算子" in detail and "shuffle" in detail

    def test_ops_omitted_falls_back_to_default_ops(self, client):
        """不传 ``ops`` 即用 ``DEFAULT_OPS``（``noise`` 不在其中）."""
        body = augment_payload(client, ops=[])
        used = set(body["report"]["by_op"])
        assert used <= set(DEFAULT_OPS)
        assert "noise" not in used

    def test_report_accounts_for_every_input(self, client):
        """报告的四本账必须对得上：``inputs + generated == expanded``."""
        report = augment_payload(client)["report"]
        assert report["inputs"] == 2
        assert report["generated"] == report["applied"] - report["dropped_duplicates"]
        assert report["threshold"] == settings.domain_near_dup_threshold

    def test_max_per_example_zero_generates_nothing(self, client):
        """``max_per_example=0``：允许，且报告如实说"一条都没新增"而不是报错."""
        body = augment_payload(client, max_per_example=0)
        assert body["generated"] == 0
        assert body["examples"] == []
        assert body["report"]["applied"] == 0

    def test_empty_examples_is_rejected_by_pydantic(self, client):
        """空样本列表 → 422（Pydantic 拦在路由之前），与 day048 同一取舍."""
        response = client.post("/data/domain/augment", json={"examples": []})
        assert response.status_code == 422

    def test_unparseable_example_returns_400(self, client):
        """某条样本缺 ``output`` → 400，详情里带下标与 ``parse_example`` 的原文."""
        response = client.post(
            "/data/domain/augment",
            json={
                "examples": [
                    ALPACA_EXAMPLES[0],
                    {"instruction": "只有问题没有答案", "output": ""},
                ]
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "第 1 条样本无法解析" in detail and "output" in detail

    def test_unknown_format_returns_400(self, client):
        """未知 format → 400（与 day048 的 /finetune/dataset/validate 同一条语言）."""
        response = client.post(
            "/data/domain/augment",
            json={"format": "sharegpt", "examples": ALPACA_EXAMPLES},
        )
        assert response.status_code == 400
        assert "未知数据集格式" in response.json()["detail"]

    def test_endpoint_is_deterministic(self, client):
        """算子按"样本序号 + 第几次尝试"轮转、不用随机数：两次请求逐字节相同."""
        assert augment_payload(client) == augment_payload(client)


class TestDomainRunEndpoint:
    """POST /data/domain/run：用仓库内置数据源跑一次六阶段流水线."""

    def test_stages_match_the_fixed_order(self, client, repo_data):
        """清单里的阶段账与 ``STAGE_ORDER`` **逐项一致**（顺序即策略）."""
        body = run_payload(client)
        assert [stage["name"] for stage in body["manifest"]["stages"]] == list(
            STAGE_ORDER
        )

    def test_size_is_positive_and_matches_freeze(self, client, repo_data):
        """size > 0，且它就是 ``freeze`` 阶段的 kept（两个数字必须同源）."""
        body = run_payload(client)
        assert body["size"] > 0
        assert body["manifest"]["size"] == body["size"]
        freeze = stage_of(body, "freeze")
        assert freeze["kept"] == body["size"]
        assert freeze["dropped"] == 0

    def test_response_carries_no_example_text(self, client, repo_data):
        """只回摘要：响应里没有样本正文，清单里带齐阶段账与参数快照."""
        body = run_payload(client)
        assert set(body) == {
            "size",
            "manifest",
            "clean",
            "quality",
            "dedupe",
            "mixing",
            "augment",
        }
        manifest = body["manifest"]
        assert set(manifest) >= {
            "version",
            "fingerprint",
            "parent_fingerprint",
            "size",
            "stages",
            "counts",
            "groups",
            "group_ratios",
            "quality",
            "config",
        }
        assert "examples" not in body and "examples" not in manifest
        assert manifest["counts"], "来源分布不能为空"

    def test_stage_reports_are_chained(self, client, repo_data):
        """上一阶段的"出" == 下一阶段的"进"：样本不能在阶段之间静默消失."""
        body = run_payload(client)
        stages = body["manifest"]["stages"]
        for previous, following in zip(stages, stages[1:]):
            assert previous["kept"] == following["total_in"], previous["name"]

    def test_clean_report_keeps_its_drop_reasons(self, client, repo_data):
        """脏样本的归因必须留在报告里（清单里看不到"被哪条硬规则拒了"）."""
        body = run_payload(client)
        assert body["clean"]["drop_reasons"], "仓库数据里刻意保留了脏样本"
        assert body["clean"]["total"] > body["clean"]["kept"]

    def test_invalid_group_by_returns_400(self, client, repo_data):
        """``group_by`` 不在三选一 → 400，详情是 ``DomainDataError`` 的原文."""
        response = client.post("/data/domain/run", json={"group_by": "license"})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "未知的分组口径" in detail and "source, origin, safety" in detail

    def test_invalid_max_group_ratio_returns_400(self, client, repo_data):
        """``max_group_ratio`` 不在 (0, 1] → 400（刻意不复制到 Pydantic 里）."""
        response = client.post("/data/domain/run", json={"max_group_ratio": 1.5})
        assert response.status_code == 400
        assert "配比上限必须落在 (0, 1]" in response.json()["detail"]

    def test_invalid_near_dup_threshold_returns_400(self, client, repo_data):
        """近重复阈值不在 (0, 1] → 400，详情来自 ``NearDuplicateIndex`` 的原文."""
        response = client.post("/data/domain/run", json={"near_dup_threshold": 0.0})
        assert response.status_code == 400
        assert "近重复阈值必须落在 (0, 1]" in response.json()["detail"]

    def test_augment_disabled_adds_nothing(self, client, repo_data):
        """``augment=false`` 时增强阶段仍在账上，但 ``added == 0``.

        阶段**不消失**是刻意的：清单里的阶段形状永远一样，调用方不需要写
        "有没有增强"的分支；而"这一步没做"必须能从 ``added == 0`` 读出来。
        """
        body = run_payload(client, augment=False)
        stage = stage_of(body, "augment")
        assert stage["added"] == 0
        assert stage["net_change"] == 0
        assert stage["total_in"] == stage["kept"]
        assert stage["detail"]["enabled"] is False
        assert body["augment"]["generated"] == 0
        assert body["manifest"]["config"]["augment_enabled"] is False

    def test_mixing_disabled_drops_nothing(self, client, repo_data):
        """``mixing=false`` 时配比阶段一条不削（回到"先削峰"之前的分布）."""
        body = run_payload(client, mixing=False)
        stage = stage_of(body, "mixing")
        assert stage["dropped"] == 0
        assert stage["detail"]["enabled"] is False

    def test_overrides_land_in_the_config_snapshot(self, client, repo_data):
        """覆盖项必须进参数快照：否则同一个指纹在别人机器上复现不出来."""
        body = run_payload(
            client,
            quality_threshold=0.1,
            group_by="origin",
            max_group_ratio=0.9,
            version=7,
            parent_fingerprint="deadbeefdeadbeef",
        )
        manifest = body["manifest"]
        assert manifest["version"] == 7
        assert manifest["parent_fingerprint"] == "deadbeefdeadbeef"
        assert manifest["config"]["quality_threshold"] == 0.1
        assert manifest["config"]["group_by"] == "origin"
        assert manifest["config"]["max_group_ratio"] == 0.9
        # 门槛放到 0.1 之后清洗后的样本应当全部留下（质量阶段不丢样本）
        assert stage_of(body, "quality")["dropped"] == 0

    def test_run_is_deterministic(self, client, repo_data):
        """同参数两次运行 → 同指纹、同规模：可复现是清单存在的前提."""
        first = run_payload(client)
        second = run_payload(client)
        assert first["size"] == second["size"]
        assert first["manifest"]["fingerprint"] == second["manifest"]["fingerprint"]
        assert first["manifest"]["stages"] == second["manifest"]["stages"]
