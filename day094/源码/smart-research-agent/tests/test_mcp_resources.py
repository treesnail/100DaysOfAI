"""MCP Resources 服务器的内存内客户端-服务器测试.

测试通过 ``create_connected_server_and_client_session`` 在进程内建立
真实的 MCP 客户端-服务器会话（内存传输），不经网络、不起子进程，
因此完全离线、确定可重复。
"""

from __future__ import annotations

import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcErrorBody,
    JsonRpcRequest,
    JsonRpcResponse,
    ResourceDescriptor,
)
from smart_research_agent.mcp_server.resources_server import (
    CONFIG_URI_TEMPLATE,
    KNOWLEDGE_URI_TEMPLATE,
    RESOURCE_DESCRIPTORS,
    KnowledgeNotFoundError,
    app,
    load_config_item,
    load_knowledge_doc,
)


class TestProtocolModels:
    """protocol.py 共享消息模型的契约测试."""

    def test_request_defaults_jsonrpc_2_0(self):
        req = JsonRpcRequest(id=1, method="resources/read", params={"uri": "research://a/b"})
        assert req.jsonrpc == "2.0"
        assert req.params["uri"] == "research://a/b"

    def test_response_roundtrip(self):
        resp = JsonRpcResponse(id="abc", result={"contents": []})
        dumped = resp.model_dump()
        assert dumped == {"jsonrpc": "2.0", "id": "abc", "result": {"contents": []}}

    def test_error_body_has_code_and_message(self):
        err = JsonRpcError(id=1, error=JsonRpcErrorBody(code=-32601, message="方法不存在"))
        assert err.error.code == -32601
        assert "不存在" in err.error.message

    def test_resource_descriptor_fields(self):
        desc = ResourceDescriptor(uri="research://x/{y}", name="x", mime_type="text/markdown")
        assert desc.uri == "research://x/{y}"
        assert desc.description == ""


class TestLoaders:
    """资源加载函数的单元测试（不经过 MCP 协议层）."""

    def test_load_existing_doc(self):
        text = load_knowledge_doc("rag")
        assert "RAG" in text
        assert "检索" in text

    def test_load_missing_doc_raises(self):
        with pytest.raises(KnowledgeNotFoundError, match="不存在"):
            load_knowledge_doc("no-such-doc")

    def test_path_traversal_rejected(self):
        with pytest.raises(KnowledgeNotFoundError, match="非法"):
            load_knowledge_doc("../config")

    def test_load_whitelisted_config(self):
        assert load_config_item("project_name") == "智研 AI 助手"

    def test_sensitive_config_blocked(self):
        with pytest.raises(KnowledgeNotFoundError, match="不可暴露"):
            load_config_item("openai_api_key")


class TestMcpResourcesInMemory:
    """内存传输下的端到端资源测试."""

    async def test_list_resource_templates(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.list_resource_templates()
        templates = {str(t.uriTemplate) for t in result.resourceTemplates}
        assert KNOWLEDGE_URI_TEMPLATE in templates
        assert CONFIG_URI_TEMPLATE in templates

    async def test_descriptors_match_registered_templates(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.list_resource_templates()
        server_uris = {str(t.uriTemplate) for t in result.resourceTemplates}
        assert server_uris == {d.uri for d in RESOURCE_DESCRIPTORS}

    async def test_read_knowledge_doc(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.read_resource("research://knowledge/rag")
        content = result.contents[0]
        assert content.mimeType == "text/markdown"
        assert "检索增强生成" in content.text

    async def test_read_config_item(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.read_resource("research://config/default_model")
        assert result.contents[0].text == "gpt-4o-mini"

    async def test_read_missing_doc_raises_mcp_error(self):
        async with create_connected_server_and_client_session(app) as session:
            with pytest.raises(McpError, match="不存在"):
                await session.read_resource("research://knowledge/nope")

    async def test_read_unknown_uri_raises_mcp_error(self):
        async with create_connected_server_and_client_session(app) as session:
            with pytest.raises(McpError, match="Unknown resource"):
                await session.read_resource("research://unknown/x")

    async def test_read_blocked_config_raises_mcp_error(self):
        async with create_connected_server_and_client_session(app) as session:
            with pytest.raises(McpError, match="不可暴露"):
                await session.read_resource("research://config/openai_api_key")
