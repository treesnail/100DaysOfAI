"""本地部署运维层测试（day045）：显存估算与 Ollama 生命周期 API（全离线）."""

from __future__ import annotations

import json
import urllib.request

import pytest

from smart_research_agent.llm.local_runtime import (
    QUANTIZATION_BITS,
    OllamaRuntime,
    bytes_per_parameter,
    estimate_vram_gb,
    fitting_quantizations,
    max_context_tokens,
    native_base_url,
    plan_deployment,
    urllib_transport,
)

#: Qwen3-8B 的公开架构参数（config.json）：36 层、8 个 KV 头、head_dim 128
QWEN3_8B = {"num_layers": 36, "kv_heads": 8, "head_dim": 128}

#: 1 GiB
GIB = 1024**3


class RecordingTransport:
    """桩 transport：记录 (url, payload)，按 path 返回预设响应."""

    def __init__(self, responses=None, error: Exception | None = None):
        self.responses = responses or {}
        self.error = error
        self.calls: list[tuple[str, dict | None]] = []

    def __call__(self, url: str, payload: dict | None):
        self.calls.append((url, payload))
        if self.error is not None:
            raise self.error
        for suffix, response in self.responses.items():
            if url.endswith(suffix):
                return response
        return {}


class TestBytesPerParameter:
    def test_theoretical_precisions(self):
        assert bytes_per_parameter("fp32") == 4.0
        assert bytes_per_parameter("fp16") == 2.0
        assert bytes_per_parameter("bf16") == 2.0
        assert bytes_per_parameter("int8") == 1.0
        assert bytes_per_parameter("int4") == 0.5

    def test_k_quant_are_mixed_precision(self):
        """GGUF 的 k-quant 是混合量化，按平均位宽计（Q4_K_M≈4.5bpw）."""
        assert bytes_per_parameter("q4_k_m") == QUANTIZATION_BITS["q4_k_m"] / 8
        assert bytes_per_parameter("Q4_K_M") == bytes_per_parameter("q4_k_m")

    def test_unknown_quant_lists_alternatives(self):
        with pytest.raises(ValueError, match="未知量化档位"):
            bytes_per_parameter("q3_k_s")


class TestEstimateVram:
    def test_weights_follow_params_times_bytes(self):
        weights_gb, _, _ = estimate_vram_gb(8.0, quant="fp16", **QWEN3_8B)
        assert weights_gb == pytest.approx(8e9 * 2 / GIB, rel=1e-6)

    def test_kv_cache_follows_documented_formula(self):
        """KV = 2 × batch × seq × layers × kv_heads × head_dim × 精度字节数."""
        _, kv_gb, _ = estimate_vram_gb(8.0, quant="q4_k_m", context_length=4096, **QWEN3_8B)
        expected = 2 * 1 * 4096 * 36 * 8 * 128 * 2 / GIB
        assert kv_gb == pytest.approx(expected, rel=1e-6)
        assert kv_gb == pytest.approx(0.5625, rel=1e-6)

    def test_total_applies_overhead_ratio(self):
        weights_gb, kv_gb, total_gb = estimate_vram_gb(
            8.0, quant="q4_k_m", context_length=4096, overhead_ratio=0.1, **QWEN3_8B
        )
        assert total_gb == pytest.approx((weights_gb + kv_gb) * 1.1, rel=1e-6)

    def test_longer_context_consumes_more_kv(self):
        _, short_kv, _ = estimate_vram_gb(8.0, context_length=4096, **QWEN3_8B)
        _, long_kv, _ = estimate_vram_gb(8.0, context_length=32768, **QWEN3_8B)
        assert long_kv == pytest.approx(short_kv * 8, rel=1e-6)

    def test_kv_dtype_bytes_scales_linearly(self):
        _, fp16_kv, _ = estimate_vram_gb(8.0, context_length=8192, kv_dtype_bytes=2, **QWEN3_8B)
        _, fp32_kv, _ = estimate_vram_gb(8.0, context_length=8192, kv_dtype_bytes=4, **QWEN3_8B)
        assert fp32_kv == pytest.approx(fp16_kv * 2, rel=1e-6)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"params_b": 0}, "params_b"),
            ({"num_layers": 0}, "架构参数"),
            ({"head_dim": -1}, "架构参数"),
            ({"context_length": 0}, "context_length"),
            ({"batch_size": 0}, "batch_size"),
            ({"kv_dtype_bytes": 0}, "kv_dtype_bytes"),
            ({"overhead_ratio": -0.1}, "overhead_ratio"),
        ],
    )
    def test_rejects_invalid_inputs(self, kwargs, match):
        base = {"params_b": 8.0, **QWEN3_8B}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            estimate_vram_gb(**base)


class TestPlanDeployment:
    def test_fits_on_24gb_with_q4(self):
        plan = plan_deployment(8.0, 24.0, quant="q4_k_m", **QWEN3_8B)
        assert plan.fits is True
        assert plan.total_gb < 24.0
        assert plan.headroom_gb > 0
        assert "可部署" in plan.recommendation()

    def test_does_not_fit_on_8gb_with_fp16(self):
        plan = plan_deployment(8.0, 8.0, quant="fp16", **QWEN3_8B)
        assert plan.fits is False
        assert plan.headroom_gb < 0
        assert "装不下" in plan.recommendation()

    def test_rejects_non_positive_vram(self):
        with pytest.raises(ValueError, match="gpu_vram_gb"):
            plan_deployment(8.0, 0.0, **QWEN3_8B)

    def test_fitting_quantizations_ranked_by_quality(self):
        """8GB 显存下 FP16/Q8 都装不下，只留下 Q5_K_M 与 Q4_K_M（质量从高到低）."""
        assert fitting_quantizations(8.0, 8.0, **QWEN3_8B) == ["q5_k_m", "q4_k_m"]

    def test_fitting_quantizations_all_on_large_gpu(self):
        assert fitting_quantizations(8.0, 24.0, **QWEN3_8B) == [
            "fp16",
            "q8_0",
            "q5_k_m",
            "q4_k_m",
        ]

    def test_max_context_tokens_in_range(self):
        tokens = max_context_tokens(8.0, 24.0, **QWEN3_8B)
        assert 120_000 <= tokens <= 140_000
        assert tokens % 1024 == 0

    def test_max_context_zero_when_weights_alone_overflow(self):
        """连权重都装不下时，任何上下文长度都跑不起来，返回 0."""
        assert max_context_tokens(8.0, 12.0, quant="fp16", **QWEN3_8B) == 0

    def test_max_context_rejects_bad_step(self):
        with pytest.raises(ValueError, match="step"):
            max_context_tokens(8.0, 24.0, step=0, **QWEN3_8B)


class TestNativeBaseUrl:
    def test_strips_v1_suffix(self):
        assert native_base_url("http://localhost:11434/v1") == "http://localhost:11434"

    def test_defaults_to_ollama(self):
        assert native_base_url() == "http://localhost:11434"

    def test_adds_v1_then_strips_it(self):
        assert native_base_url("localhost:11434") == "http://localhost:11434"


class TestOllamaRuntime:
    def test_list_models_parses_tags(self):
        transport = RecordingTransport(
            {"/api/tags": {"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}]}}
        )
        runtime = OllamaRuntime(transport=transport)
        assert runtime.list_models() == ["qwen3:8b", "nomic-embed-text:latest"]
        assert transport.calls[0] == ("http://localhost:11434/api/tags", None)

    def test_list_running_and_is_running(self):
        transport = RecordingTransport(
            {"/api/ps": {"models": [{"name": "qwen3:8b", "expires_at": "2026-09-12T09:00:00Z"}]}}
        )
        runtime = OllamaRuntime(transport=transport)
        assert runtime.list_running() == [
            {"name": "qwen3:8b", "expires_at": "2026-09-12T09:00:00Z"}
        ]
        assert runtime.is_running("qwen3:8b") is True
        assert runtime.is_running("llama3.2:3b") is False

    def test_list_models_tolerates_missing_key(self):
        runtime = OllamaRuntime(transport=RecordingTransport({"/api/tags": {}}))
        assert runtime.list_models() == []

    def test_base_url_normalized_to_native_root(self):
        runtime = OllamaRuntime("http://localhost:11434/v1", transport=RecordingTransport())
        assert runtime.base_url == "http://localhost:11434"

    def test_load_model_sends_default_keep_alive(self):
        transport = RecordingTransport()
        runtime = OllamaRuntime(transport=transport)
        runtime.load_model("qwen3:8b")
        url, payload = transport.calls[0]
        assert url.endswith("/api/generate")
        assert payload == {"model": "qwen3:8b", "keep_alive": "5m"}

    def test_load_model_with_keep_alive_and_context(self):
        """预热常驻 + 放大上下文：两个 Ollama 专有能力的组合."""
        transport = RecordingTransport()
        runtime = OllamaRuntime(transport=transport)
        runtime.load_model("qwen3:8b", keep_alive=-1, context_length=8192)
        assert transport.calls[0][1] == {
            "model": "qwen3:8b",
            "keep_alive": -1,
            "options": {"num_ctx": 8192},
        }

    def test_load_model_rejects_bad_context(self):
        runtime = OllamaRuntime(transport=RecordingTransport())
        with pytest.raises(ValueError, match="context_length"):
            runtime.load_model("qwen3:8b", context_length=0)

    def test_unload_model_sets_keep_alive_zero(self):
        transport = RecordingTransport()
        OllamaRuntime(transport=transport).unload_model("qwen3:8b")
        assert transport.calls[0][1] == {"model": "qwen3:8b", "keep_alive": 0}

    def test_pull_model_disables_stream(self):
        transport = RecordingTransport({"/api/pull": {"status": "success"}})
        result = OllamaRuntime(transport=transport).pull_model("qwen3-vl:8b")
        assert result == {"status": "success"}
        assert transport.calls[0][1] == {"model": "qwen3-vl:8b", "stream": False}

    def test_transport_error_becomes_runtime_error(self):
        runtime = OllamaRuntime(transport=RecordingTransport(error=OSError("连接被拒绝")))
        with pytest.raises(RuntimeError, match="列出本地模型失败"):
            runtime.list_models()

    def test_post_error_mentions_model_name(self):
        runtime = OllamaRuntime(transport=RecordingTransport(error=OSError("boom")))
        with pytest.raises(RuntimeError, match="加载模型 qwen3:8b 失败"):
            runtime.load_model("qwen3:8b")

    def test_non_dict_response_rejected(self):
        class BadTransport:
            def __call__(self, url, payload):
                return ["not", "a", "dict"]

        runtime = OllamaRuntime(transport=BadTransport())
        with pytest.raises(RuntimeError, match="响应不是 JSON 对象"):
            runtime.list_models()


class FakeHttpResponse:
    """urllib.urlopen 返回值的桩：只需支持 with 与 read()."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self._body


class TestUrllibTransport:
    """默认传输层的单元测试：monkeypatch urlopen，仍不发真实请求."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, body: bytes) -> dict:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["data"] = request.data
            captured["method"] = request.get_method()
            captured["headers"] = {k.lower(): v for k, v in request.header_items()}
            captured["timeout"] = timeout
            return FakeHttpResponse(body)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_post_serializes_json_payload(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._patch(monkeypatch, b'{"status": "success"}')
        result = urllib_transport("http://localhost:11434/api/generate", {"model": "qwen3:8b"})
        assert result == {"status": "success"}
        assert captured["method"] == "POST"
        assert json.loads(captured["data"]) == {"model": "qwen3:8b"}
        assert captured["headers"]["content-type"] == "application/json"

    def test_get_sends_no_body(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._patch(monkeypatch, b'{"models": []}')
        assert urllib_transport("http://localhost:11434/api/tags", None) == {"models": []}
        assert captured["method"] == "GET"
        assert captured["data"] is None

    def test_empty_body_returns_empty_dict(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch, b"")
        assert urllib_transport("http://localhost:11434/api/ps", None) == {}
