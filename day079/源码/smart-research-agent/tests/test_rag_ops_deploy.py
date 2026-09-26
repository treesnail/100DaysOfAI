"""``rag_ops.deploy``：编排规格、11 条静态判据与"仓库里那份 yml"（day072）.

这一课最"部署"的一处：规格是数据、判据是代码、yml 是渲染产物，
而**那份跟着仓库走的 yml 必须与规格逐字节一致**（本文件最后一组用例守着它）。

```text
ComposeSpec → validate_spec（R1~R11）→ render_compose（确定性文本）→ docker-compose.rag.yml
                                                    ↑
                              test_committed_compose_matches_the_spec：手改就变红
```

没有 Docker 也能全测：判据全是静态的（字符串与数字），渲染是纯函数。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from smart_research_agent.config import settings
from smart_research_agent.rag_ops.deploy import (
    API_CONTAINER_PORT,
    API_DOCKERFILE,
    API_HEALTH_PATH,
    CHROMA_HEARTBEAT_PATH,
    CHROMA_IMAGE,
    CHROMA_TAG,
    CONTAINER_CORPUS_DIR,
    CONTAINER_INDEX_DIR,
    CONTAINER_OPS_DIR,
    SERVICE_API,
    SERVICE_ROLES,
    SERVICE_STORE,
    SERVICE_WORKER,
    VALIDATION_RULES,
    ComposeSpec,
    HealthProbe,
    PortMapping,
    ServiceSpec,
    VolumeMount,
    check_committed_compose,
    compose_summary_lines,
    default_spec,
    describe,
    render_compose,
    require_valid,
    validate_spec,
    write_compose,
)
from smart_research_agent.rag_ops.errors import DeployError


class TestPortMapping:
    """端口：1~65535，宿主端口不能是 0（随机端口等于外面再也找不到它）."""

    def test_render(self) -> None:
        assert PortMapping(host=8080, container=8080).render() == "8080:8080"
        assert PortMapping(host=8080, container=8080).to_dict() == {
            "host": 8080,
            "container": 8080,
        }

    @pytest.mark.parametrize("host,container", [(0, 80), (65536, 80), (80, 0), (80, 70000)])
    def test_bad_ports_are_rejected(self, host: int, container: int) -> None:
        with pytest.raises(DeployError):
            PortMapping(host=host, container=container)


class TestVolumeMount:
    """卷：来源必须是**顶层声明的卷名**，目标必须是绝对路径."""

    def test_render_flags_read_only(self) -> None:
        mount = VolumeMount(source="rag-index", target="/app/data/index")
        assert mount.render() == "rag-index:/app/data/index"
        assert replace(mount, read_only=True).render() == "rag-index:/app/data/index:ro"
        assert mount.to_dict()["read_only"] is False

    @pytest.mark.parametrize(
        "source,target",
        [
            ("", "/data"),
            ("./data", "/data"),
            ("a/b", "/data"),
            ("rag-index", "data/index"),
            ("rag-index", "/"),
            ("rag-index", "/app/data/index/"),
        ],
    )
    def test_bad_mounts_are_rejected(self, source: str, target: str) -> None:
        with pytest.raises(DeployError):
            VolumeMount(source=source, target=target)


class TestHealthProbe:
    """探针：命令非空、首段合法、超时必须短于间隔、NONE 不算探针."""

    def test_render_lines(self) -> None:
        probe = HealthProbe(test=("CMD", "curl", "-f", "http://localhost:8080/x"))
        lines = probe.render()
        assert lines[0] == 'test: ["CMD", "curl", "-f", "http://localhost:8080/x"]'
        assert "interval: 30s" in lines and "start_period: 15s" in lines
        assert probe.to_dict()["test_line"].startswith("CMD curl")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"test": ()},
            {"test": ("curl", "-f", "http://x")},
            {"test": ("NONE",)},
            {"interval_seconds": 0},
            {"timeout_seconds": 0},
            {"start_period_seconds": 0},
            {"retries": 0},
            {"interval_seconds": 5, "timeout_seconds": 5},
        ],
    )
    def test_bad_probes_are_rejected(self, overrides: dict) -> None:
        base: dict = {"test": ("CMD", "true")}
        base.update(overrides)
        with pytest.raises(DeployError):
            HealthProbe(**base)


class TestServiceSpec:
    """服务：角色名封闭、环境变量不许重复、镜像 tag 可追溯."""

    def test_helpers_and_projection(self) -> None:
        service = ServiceSpec(
            name=SERVICE_API,
            image="repo/app:v1",
            ports=(PortMapping(8080, 8080),),
            volumes=(VolumeMount("rag-index", "/app/data/index"),),
            environment=(("B", "2"), ("A", "1")),
        )
        assert service.env() == {"A": "1", "B": "2"}
        assert service.volume_names() == ("rag-index",)
        assert service.port_pairs() == ((8080, 8080),)
        assert service.image_tag() == "v1"
        assert service.to_dict()["image_tag"] == "v1"

    @pytest.mark.parametrize(
        "image,tag",
        [
            ("chromadb/chroma:1.5.3", "1.5.3"),
            ("repo/app", ""),
            ("localhost:5000/app", ""),  # 端口不是 tag
            ("localhost:5000/app:v2", "v2"),
        ],
    )
    def test_image_tag_parsing(self, image: str, tag: str) -> None:
        assert ServiceSpec(name=SERVICE_API, image=image).image_tag() == tag

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ServiceSpec(name="worker-2", image="repo/app:v1")

    def test_empty_image_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ServiceSpec(name=SERVICE_API, image="")

    def test_duplicate_environment_key_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ServiceSpec(
                name=SERVICE_API,
                image="repo/app:v1",
                environment=(("A", "1"), ("A", "2")),
            )

    def test_build_without_context_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ServiceSpec(
                name=SERVICE_API,
                image="repo/app:v1",
                build_dockerfile="Dockerfile.api",
                build_context="  ",
            )


class TestComposeSpec:
    """规格：项目名非空、服务名唯一、卷名排序去重."""

    def _service(self, name: str, image: str = "repo/app:v1") -> ServiceSpec:
        return ServiceSpec(name=name, image=image)

    def test_services_are_sorted_and_unique(self) -> None:
        spec = ComposeSpec(
            project_name="demo",
            services=(self._service(SERVICE_WORKER), self._service(SERVICE_API)),
            volumes=("b", "a", "a"),
        )
        assert spec.service_names() == (SERVICE_API, SERVICE_WORKER)
        assert spec.volumes == ("a", "b")
        assert spec.service(SERVICE_STORE) is None
        assert spec.service(SERVICE_API) is not None

    def test_empty_project_name_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ComposeSpec(project_name="")

    def test_duplicate_service_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ComposeSpec(
                project_name="demo",
                services=(self._service(SERVICE_API), self._service(SERVICE_API)),
            )

    def test_empty_volume_name_is_rejected(self) -> None:
        with pytest.raises(DeployError):
            ComposeSpec(project_name="demo", volumes=("",))

    def test_projection(self) -> None:
        spec = default_spec()
        thin = spec.to_dict(include_services=False)
        assert thin["services"] == [SERVICE_API, SERVICE_WORKER, SERVICE_STORE]
        assert "detail" not in thin
        assert len(spec.to_dict()["detail"]) == 3


class TestRenderCompose:
    """渲染：确定性、键按名排序、必要的引号、结尾一个换行."""

    def test_is_deterministic(self) -> None:
        assert render_compose(default_spec()) == render_compose(default_spec())

    def test_header_and_ending(self) -> None:
        text = render_compose(default_spec())
        assert text.startswith("# 智研 AI 助手")
        assert "render_compose()" in text
        assert text.endswith("driver: bridge\n")

    def test_service_order_is_sorted(self) -> None:
        text = render_compose(default_spec())
        assert text.index("  rag-api:") < text.index("  rag-ops-sync:") < text.index(
            "  vector-store:"
        )

    def test_image_and_healthcheck_are_rendered(self) -> None:
        text = render_compose(default_spec())
        assert f'image: "{CHROMA_IMAGE}:{CHROMA_TAG}"' in text
        assert f"http://localhost:{API_CONTAINER_PORT}{API_HEALTH_PATH}" in text
        assert f"http://localhost:8000{CHROMA_HEARTBEAT_PATH}" in text
        assert 'command: ["python", "-m", "uvicorn"' in text
        assert "build:" in text and f"dockerfile: {API_DOCKERFILE}" in text

    def test_volumes_and_network_are_rendered(self) -> None:
        text = render_compose(default_spec())
        assert "volumes:" in text and "  rag-index:" in text
        assert f'"rag-index:{CONTAINER_INDEX_DIR}"' in text
        assert f'"rag-corpus:{CONTAINER_CORPUS_DIR}:ro"' in text
        assert f'"rag-corpus:{CONTAINER_CORPUS_DIR}"' in text
        assert "networks:" in text and "  rag-net:" in text

    def test_compose_summary_lines(self) -> None:
        lines = compose_summary_lines(default_spec())
        assert len(lines) == 3
        assert any("无探针（已写明豁免）" in line for line in lines)


class TestScalarRendering:
    """标量渲染：含 ':' 或 '#' 时加双引号，无法表达的当场拒绝."""

    def _render(self, value: str) -> str:
        spec = default_spec()
        worker = spec.service(SERVICE_WORKER)
        assert worker is not None
        mutated = replace(worker, environment=(("PROBE", value),))
        return render_compose(replace(spec, services=(mutated,)))

    def test_colon_and_hash_get_quoted(self) -> None:
        assert 'PROBE: "http://host:8000"' in self._render("http://host:8000")
        assert 'PROBE: "a#b"' in self._render("a#b")

    def test_plain_values_stay_unquoted(self) -> None:
        assert "PROBE: plain" in self._render("plain")

    @pytest.mark.parametrize("value", ['bad"quote', "line\nbreak", "  padded  ", "", "-lead"])
    def test_unrenderable_values_are_rejected(self, value: str) -> None:
        # "-lead" 会被加上引号（YAML 指示符），因此它不报错；其余五种必须报错
        if value == "-lead":
            assert 'PROBE: "-lead"' in self._render(value)
            return
        with pytest.raises(DeployError):
            self._render(value)


class TestValidationRules:
    """11 条判据：每条都要能被一次"手工改坏"触发."""

    def test_default_spec_is_valid(self) -> None:
        assert validate_spec(default_spec()) == ()
        assert len(VALIDATION_RULES) == 11

    def _problems(self, spec: ComposeSpec) -> tuple[str, ...]:
        return validate_spec(spec)

    def _with_service(self, spec: ComposeSpec, service: ServiceSpec) -> ComposeSpec:
        return replace(
            spec,
            services=tuple(
                service if item.name == service.name else item for item in spec.services
            ),
        )

    def test_r1_requires_all_three_roles(self) -> None:
        spec = default_spec()
        problems = self._problems(replace(spec, services=(spec.service(SERVICE_API),)))
        assert any(item.startswith("R1") for item in problems)

    def test_r2_requires_a_probe_or_a_reason(self) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        without_probe = replace(api, health_probe=None, probe_exempt="")
        assert any(
            item.startswith("R2") and SERVICE_API in item
            for item in self._problems(self._with_service(spec, without_probe))
        )
        both = replace(api, probe_exempt="我就是不想配")
        assert any(
            item.startswith("R2") and "同时" in item
            for item in self._problems(self._with_service(spec, both))
        )

    def test_r3_checks_the_persistent_volumes(self) -> None:
        spec = default_spec()
        stripped = replace(
            spec, services=tuple(replace(item, volumes=()) for item in spec.services)
        )
        assert any("没有任何服务挂载持久卷" in item for item in self._problems(stripped))

        undeclared = replace(spec, volumes=tuple(v for v in spec.volumes if v != "rag-index"))
        assert any("没有在顶层 volumes 里声明" in item for item in self._problems(undeclared))

        unused = replace(spec, volumes=(*spec.volumes, "leftover"))
        assert any("没有任何服务挂载它" in item for item in self._problems(unused))

    @pytest.mark.parametrize("image", ["repo/app:latest", "repo/app"])
    def test_r4_requires_a_traceable_tag(self, image: str) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        problems = self._problems(self._with_service(spec, replace(api, image=image)))
        assert any(item.startswith("R4") for item in problems)

    def test_r5_rejects_duplicate_host_ports(self) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        clash = replace(api, ports=(PortMapping(host=8000, container=8080),))
        problems = self._problems(self._with_service(spec, clash))
        assert any(item.startswith("R5") for item in problems)

    def test_r6_rejects_bad_environment_entries(self) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        lower = replace(api, environment=(("lower_case", "1"),))
        assert any(item.startswith("R6") for item in self._problems(self._with_service(spec, lower)))
        empty = replace(api, environment=(("EMPTY_VALUE", ""),))
        assert any(
            "空值" in item for item in self._problems(self._with_service(spec, empty))
        )
        broken = replace(api, environment=(("BAD_VALUE", 'has"quote'),))
        assert any(
            "无法渲染" in item for item in self._problems(self._with_service(spec, broken))
        )

    def test_r7_rejects_mounts_over_image_paths(self) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        for target in ("/usr/lib", "/app/smart_research_agent", "/bin"):
            broken = replace(
                api, volumes=(VolumeMount(source="rag-index", target=target),)
            )
            problems = self._problems(self._with_service(spec, broken))
            assert any(item.startswith("R7") for item in problems), target

    def test_mounting_the_container_root_is_rejected_by_the_shape(self) -> None:
        # 挂到 "/" 在**构造期**就被拒（比 R7 更早生效），因此 R7 不重复判它——
        # 同一条规则有两处实现时，两处迟早会对同一份输入给出不同结论。
        with pytest.raises(DeployError):
            VolumeMount(source="rag-index", target="/")

    def test_r8_requires_depends_on_for_the_chroma_backend(self) -> None:
        chroma = default_spec(backend="chroma")
        assert chroma.service(SERVICE_API).depends_on == (SERVICE_STORE,)  # type: ignore[union-attr]
        assert validate_spec(chroma) == ()
        api = chroma.service(SERVICE_API)
        assert api is not None
        broken = self._with_service(chroma, replace(api, depends_on=()))
        assert any(item.startswith("R8") for item in self._problems(broken))

    def test_r9_requires_real_dependencies(self) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        ghost = self._with_service(spec, replace(api, depends_on=("ghost",)))
        assert any("不存在的服务" in item for item in self._problems(ghost))
        itself = self._with_service(spec, replace(api, depends_on=(SERVICE_API,)))
        assert any("依赖自己" in item for item in self._problems(itself))

    def test_r10_rejects_blank_commands(self) -> None:
        spec = default_spec()
        worker = spec.service(SERVICE_WORKER)
        assert worker is not None
        self_blank = self._with_service(spec, replace(worker, command=(" ",)))
        assert any(item.startswith("R10") for item in self._problems(self_blank))

    def test_r11_rejects_padded_volume_names(self) -> None:
        spec = default_spec()
        padded = replace(spec, volumes=(" rag-index ",))
        assert any(item.startswith("R11") for item in self._problems(padded))

    def test_require_valid_lists_every_problem(self) -> None:
        spec = default_spec()
        api = spec.service(SERVICE_API)
        assert api is not None
        broken = replace(spec, services=(replace(api, health_probe=None, image="repo/app"),))
        with pytest.raises(DeployError) as excinfo:
            require_valid(broken)
        message = str(excinfo.value)
        assert "R2" in message and "R4" in message and "1." in message


class TestComposeFile:
    """仓库里那份 yml：写进去、比对得上、手改要能报出行号."""

    def test_write_and_check_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "docker-compose.rag.yml"
        spec = default_spec()
        assert write_compose(spec, target) == target
        assert check_committed_compose(spec, target) == ""

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        message = check_committed_compose(default_spec(), tmp_path / "absent.yml")
        assert "不存在" in message and "--write-compose" in message

    def test_drift_reports_the_first_differing_line(self, tmp_path: Path) -> None:
        target = tmp_path / "docker-compose.rag.yml"
        write_compose(default_spec(), target)
        lines = target.read_text(encoding="utf-8").splitlines()
        lines[5] = "# 我手改了这一行"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        message = check_committed_compose(default_spec(), target)
        assert "第 6 行" in message and "我手改了这一行" in message

    def test_truncated_file_is_reported(self, tmp_path: Path) -> None:
        target = tmp_path / "docker-compose.rag.yml"
        target.write_text("name: something-else\n", encoding="utf-8")
        message = check_committed_compose(default_spec(), target)
        assert "不一致" in message


class TestShippedArtifacts:
    """随仓库提交的两份文件：那个 yml 与那份 Dockerfile 都必须与规格一致."""

    def test_committed_compose_matches_the_spec(self) -> None:
        path = Path(settings.rag_ops_compose_path)
        assert path.exists(), f"{path} 应当随仓库提交（见 docs/rag_ops.md）"
        assert check_committed_compose(default_spec(), path) == ""

    def test_dockerfile_api_matches_the_spec(self) -> None:
        path = Path(API_DOCKERFILE)
        assert path.exists(), f"{path} 应当随仓库提交"
        text = path.read_text(encoding="utf-8")
        # 端口与探针路径必须与 deploy 的常量一致（否则容器"healthy"但没人能访问）
        assert f"EXPOSE {API_CONTAINER_PORT}" in text
        assert f"http://localhost:{API_CONTAINER_PORT}{API_HEALTH_PATH}" in text
        # 必须绑 0.0.0.0（`python -m ...api.app` 那个入口绑的是 127.0.0.1）
        assert "--host" in text and "0.0.0.0" in text
        # 三条容器内路径必须与卷挂载点逐字一致
        assert CONTAINER_INDEX_DIR in text
        assert CONTAINER_CORPUS_DIR in text
        assert CONTAINER_OPS_DIR in text
        # 启动命令与规格里的那一份逐段一致（factory 模式 + 端口）
        assert '"smart_research_agent.api.app:create_app"' in text
        assert '"--factory"' in text
        assert f'"--port", "{API_CONTAINER_PORT}"' in text


class TestDescribeSpec:
    """端点用的摘要：规格 + 判据结论 + 与仓库文件的一致性，一次性给出."""

    def test_describe_reports_rules_and_drift(self) -> None:
        payload = describe(default_spec())
        assert payload["valid"] is True and payload["problems"] == []
        assert payload["rules"] == len(VALIDATION_RULES)
        assert payload["compose_path"] == settings.rag_ops_compose_path
        assert payload["compose_committed"] is True
        assert payload["compose_drift"] == ""
        assert payload["checked_at"].endswith("Z")
        assert len(payload["summary_lines"]) == 3
