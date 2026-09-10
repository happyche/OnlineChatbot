# -*- coding: utf-8 -*-
"""
对话模型提供方测试：云端 API 与自建 OpenAI 兼容服务两条路。

重点覆盖两类容易出错的地方：
  1. 端点解析 —— 两套配置并存时，必须能确定「哪一套在生效」，
     否则界面显示的和实际调用的会不一致
  2. 模型名取值 —— 自建服务用 local_llm_model，若代码仍固定读 llm_model，
     就会把云端模型名发到自建服务上，得到一个语义模糊的 404
"""
from __future__ import annotations

import dataclasses

import pytest

import config
from rag_engine import RAGEngine


def _settings(**overrides) -> dict:
    base = config.load_settings()
    base.update(overrides)
    return base


class TestResolveLocal:
    def test_local_uses_local_fields(self):
        endpoint = config.resolve_llm(
            _settings(
                llm_provider="local",
                local_llm_base_url="http://10.0.0.5:11434/v1",
                local_llm_model="qwen2.5:14b",
                # 云端字段存在也不应被采用
                llm_base_url="https://cloud.example.com/v1",
                llm_model="qwen-plus",
            )
        )

        assert endpoint.provider == "local"
        assert endpoint.base_url == "http://10.0.0.5:11434/v1"
        assert endpoint.model == "qwen2.5:14b"

    def test_local_needs_no_api_key_but_gets_placeholder(self):
        """自建服务多数不校验 Key，但 OpenAI 客户端要求非空，故填占位串。"""
        endpoint = config.resolve_llm(
            _settings(
                llm_provider="local",
                local_llm_base_url="http://10.0.0.5:11434/v1",
                local_llm_model="qwen2.5:7b",
                local_llm_api_key="",
                llm_api_key="",
            )
        )

        assert endpoint.api_key == "local"

    def test_local_api_key_used_when_provided(self):
        """vLLM 启用 --api-key 时需要真实令牌。"""
        endpoint = config.resolve_llm(
            _settings(
                llm_provider="local",
                local_llm_base_url="http://10.0.0.5:8000/v1",
                local_llm_model="Qwen/Qwen2.5-7B-Instruct",
                local_llm_api_key="server-token",
            )
        )

        assert endpoint.api_key == "server-token"

    def test_missing_base_url_rejected(self):
        with pytest.raises(ValueError, match="local_llm_base_url"):
            config.resolve_llm(_settings(llm_provider="local", local_llm_base_url=""))

    def test_missing_model_rejected(self):
        with pytest.raises(ValueError, match="local_llm_model"):
            config.resolve_llm(
                _settings(
                    llm_provider="local",
                    local_llm_base_url="http://10.0.0.5:11434/v1",
                    local_llm_model="",
                )
            )


class TestResolveCloud:
    def test_cloud_uses_cloud_fields(self):
        endpoint = config.resolve_llm(
            _settings(
                llm_provider="openai",
                llm_api_key="sk-real",
                llm_base_url="https://cloud.example.com/v1",
                llm_model="qwen-plus",
                local_llm_model="qwen2.5:7b",
            )
        )

        assert endpoint.provider == "openai"
        assert endpoint.model == "qwen-plus"
        assert endpoint.api_key == "sk-real"

    def test_cloud_without_key_rejected_with_hint(self):
        with pytest.raises(ValueError, match="llm_provider 设为 local"):
            config.resolve_llm(_settings(llm_provider="openai", llm_api_key=""))

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="llm_provider"):
            config.resolve_llm(_settings(llm_provider="魔法"))


class TestResolveEmbeddingEndpoint:
    def test_falls_back_to_chat_endpoint(self):
        chat = config.Endpoint("http://chat/v1", "m", "k", "local")

        embed = config.resolve_embedding_endpoint(
            _settings(embedding_base_url="", embedding_api_key=""), chat
        )

        assert embed.base_url == "http://chat/v1"
        assert embed.api_key == "k"

    def test_independent_url_and_key(self):
        """对话走内网自建、嵌入走云端时，两者的地址与 Key 都必须分开。"""
        chat = config.Endpoint("http://10.0.0.5:11434/v1", "qwen2.5:7b", "local", "local")

        embed = config.resolve_embedding_endpoint(
            _settings(
                embedding_base_url="https://dashscope.example.com/v1",
                embedding_api_key="sk-cloud",
                embedding_model="text-embedding-v3",
            ),
            chat,
        )

        assert embed.base_url == "https://dashscope.example.com/v1"
        assert embed.api_key == "sk-cloud"
        assert embed.model == "text-embedding-v3"


class TestInternalHostDetection:
    """
    企业环境普遍设置 HTTP_PROXY，而客户端默认读取该环境变量，
    导致访问内网自建服务的请求被发去代理并返回一段 HTML 错误页
    （表现为 InternalServerError + HTML 片段，极难定位）。
    因此需要按私有地址段自动绕过代理。
    """

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://10.67.34.44:11434/v1",
            "http://192.168.1.10:11434/v1",
            "http://172.16.5.20:8000/v1",
        ],
    )
    def test_internal_addresses_detected(self, url):
        assert config.is_internal_host(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://api.openai.com/v1",
            "http://8.8.8.8/v1",
        ],
    )
    def test_public_addresses_not_bypassed(self, url):
        assert config.is_internal_host(url) is False

    def test_hostname_treated_as_external(self):
        """主机名无法判断是否内网，按外网处理，交给用户显式配置 NO_PROXY。"""
        assert config.is_internal_host("http://llm-server.corp/v1") is False

    def test_malformed_url_is_safe(self):
        assert config.is_internal_host("") is False
        assert config.is_internal_host("not a url") is False

    def test_client_bypasses_proxy_for_internal_endpoint(self, make_engine):
        engine = make_engine(
            llm_provider="local",
            local_llm_base_url="http://10.67.34.44:11434/v1",
            local_llm_model="qwen3.8:27b",
            bypass_proxy_for_internal=True,
        )

        # trust_env=False 的客户端不会读取 HTTP_PROXY 环境变量
        assert engine._state.client._client.trust_env is False

    def test_bypass_can_be_disabled(self, make_engine):
        engine = make_engine(
            llm_provider="local",
            local_llm_base_url="http://10.67.34.44:11434/v1",
            local_llm_model="qwen3.8:27b",
            bypass_proxy_for_internal=False,
        )

        assert engine._state.client._client.trust_env is True

    def test_public_endpoint_keeps_proxy_settings(self, make_engine):
        """访问公网时必须仍然走代理，否则企业网络里根本连不出去。"""
        engine = make_engine(
            llm_provider="openai",
            llm_api_key="sk-real",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        assert engine._state.client._client.trust_env is True


class TestEngineWithLocalLLM:
    def test_engine_builds_without_cloud_key(self, make_engine):
        engine = make_engine(
            llm_provider="local",
            llm_api_key="",
            local_llm_base_url="http://10.0.0.5:11434/v1",
            local_llm_model="qwen2.5:7b",
        )

        assert engine._state.chat.provider == "local"
        assert str(engine._state.client.base_url).rstrip("/") == "http://10.0.0.5:11434/v1"

    def test_engine_degrades_gracefully_when_llm_unconfigured(self, make_engine):
        """
        只配好嵌入、还没配对话服务时，引擎仍应能构造。

        否则用户连配置接口都调不到，没法在界面里补填地址。
        """
        engine = make_engine(llm_provider="openai", llm_api_key="")

        assert engine._state.client is None
        assert "llm_api_key" in engine._state.chat_error

    def test_describe_endpoints_reports_active_target(self, make_engine):
        engine = make_engine(
            llm_provider="local",
            local_llm_base_url="http://10.0.0.5:11434/v1",
            local_llm_model="qwen2.5:7b",
        )

        info = engine.describe_endpoints()

        assert info["llm_provider"] == "local"
        assert info["llm_model"] == "qwen2.5:7b"
        assert info["llm_base_url"] == "http://10.0.0.5:11434/v1"
        assert info["llm_error"] is None

    def test_describe_endpoints_reports_error(self, make_engine):
        engine = make_engine(llm_provider="openai", llm_api_key="")

        info = engine.describe_endpoints()

        assert info["llm_model"] is None
        assert "llm_api_key" in info["llm_error"]


class TestModelNameSentToEndpoint:
    """
    回归防线：发给端点的模型名必须来自解析结果。

    固定读 settings["llm_model"] 会在自建服务上发出云端模型名，
    服务器返回的 404 很难让人联想到配置读错了字段。
    """

    async def _capture_model(self, engine) -> str:
        captured: dict = {}

        class FakeStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        engine._state = dataclasses.replace(engine._state, client=FakeClient())
        async for _ in engine.query("问题"):
            pass
        return captured["model"]

    async def test_local_provider_sends_local_model(self, make_engine):
        engine = make_engine(
            llm_provider="local",
            local_llm_base_url="http://10.0.0.5:11434/v1",
            local_llm_model="qwen2.5:7b",
            llm_model="qwen-plus",
        )

        assert await self._capture_model(engine) == "qwen2.5:7b"

    async def test_cloud_provider_sends_cloud_model(self, make_engine):
        engine = make_engine(
            llm_provider="openai",
            llm_api_key="sk-real",
            llm_model="qwen-max",
            local_llm_model="qwen2.5:7b",
        )

        assert await self._capture_model(engine) == "qwen-max"

    async def test_query_raises_clear_error_when_llm_missing(self, make_engine):
        engine = make_engine(llm_provider="openai", llm_api_key="")

        with pytest.raises(RuntimeError, match="对话模型不可用"):
            async for _ in engine.query("问题"):
                pass


class TestModelsEndpoint:
    def test_lists_models(self, client, monkeypatch):
        async def fake_list(self):
            return ["qwen2.5:7b", "bge-m3"]

        monkeypatch.setattr(RAGEngine, "list_chat_models", fake_list)

        resp = client.get("/api/models")

        assert resp.status_code == 200
        assert resp.json()["models"] == ["qwen2.5:7b", "bge-m3"]

    def test_connection_failure_reported_as_502(self, client, monkeypatch):
        async def fake_list(self):
            raise ConnectionError("Connection refused")

        monkeypatch.setattr(RAGEngine, "list_chat_models", fake_list)

        resp = client.get("/api/models")

        assert resp.status_code == 502
        assert "Connection refused" in resp.json()["detail"]


class TestSettingsExposure:
    def test_all_api_keys_masked(self, client):
        body = client.get("/api/settings").json()

        for field in ("llm_api_key", "local_llm_api_key", "embedding_api_key"):
            assert field not in body, f"{field} 明文出现在响应中"
            assert f"{field}_display" in body

    def test_active_endpoint_reported(self, client):
        body = client.get("/api/settings").json()

        assert "active" in body
        assert body["active"]["embedding_signature"]

    def test_provider_switch_persists(self, client):
        resp = client.post(
            "/api/settings",
            json={
                "llm_provider": "local",
                "local_llm_base_url": "http://10.0.0.5:11434/v1",
                "local_llm_model": "qwen2.5:7b",
            },
        )

        assert resp.status_code == 200
        body = client.get("/api/settings").json()
        assert body["llm_provider"] == "local"
        assert body["active"]["llm_provider"] == "local"
        assert body["active"]["llm_model"] == "qwen2.5:7b"

    def test_invalid_local_config_reported_on_save(self, client):
        """
        自建服务缺地址时必须在保存响应里就报出来，不能等用户提问才暴露。

        这里不用 400：设置仍需落盘，否则对话端点没配好时连检索参数都改不了。
        因此约定为「保存成功 + 显式 llm_error」，由前端弹错误提示。
        """
        resp = client.post(
            "/api/settings", json={"llm_provider": "local", "local_llm_base_url": ""}
        )

        assert resp.status_code == 200
        assert "local_llm_base_url" in resp.json()["llm_error"]

    def test_valid_config_has_no_llm_error(self, client):
        resp = client.post(
            "/api/settings",
            json={
                "llm_provider": "local",
                "local_llm_base_url": "http://10.0.0.5:11434/v1",
                "local_llm_model": "qwen2.5:7b",
            },
        )

        assert resp.json()["llm_error"] is None

    def test_unrelated_settings_still_saved_when_llm_broken(self, client):
        """对话端点没配好不应妨碍保存其他参数。"""
        client.post("/api/settings", json={"llm_provider": "local", "local_llm_base_url": ""})

        resp = client.post("/api/settings", json={"top_k": 9})

        assert resp.status_code == 200
        assert client.get("/api/settings").json()["top_k"] == 9
