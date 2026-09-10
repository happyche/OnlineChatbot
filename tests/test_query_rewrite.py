# -*- coding: utf-8 -*-
"""query_rewrite 模块的离线测试。"""
from __future__ import annotations

import pytest

import query_rewrite as qr


def _settings(**overrides) -> dict:
    base = {
        "query_rewrite_enabled": True,
        "query_rewrite_mode": "conditional",
        "rewrite_history_limit": 6,
        "rewrite_trigger_max_len": 12,
        "rewrite_trigger_patterns": "",
        "rewrite_model": "",
        "rewrite_temperature": 0.0,
    }
    base.update(overrides)
    return base


HISTORY = [
    {"role": "user", "content": "REQUEST_TIMEOUT 默认多少秒？"},
    {"role": "assistant", "content": "默认 60 秒。"},
]


class TestNeedsRewrite:
    def test_skips_without_history(self):
        assert not qr.needs_rewrite("它的默认值呢", None, _settings())

    def test_skips_when_disabled(self):
        assert not qr.needs_rewrite(
            "它的默认值呢", HISTORY, _settings(query_rewrite_enabled=False)
        )

    def test_always_mode_with_history(self):
        assert qr.needs_rewrite(
            "完整独立问题也可以检索", HISTORY, _settings(query_rewrite_mode="always")
        )

    def test_conditional_short_question(self):
        assert qr.needs_rewrite("默认值呢", HISTORY, _settings())

    def test_conditional_pronoun(self):
        assert qr.needs_rewrite("它的默认值是多少？", HISTORY, _settings())

    def test_conditional_skips_standalone_question(self):
        q = "REQUEST_TIMEOUT 环境变量在配置文件中的推荐取值范围是什么？"
        assert not qr.needs_rewrite(q, HISTORY, _settings())


class TestSanitizeHistory:
    def test_filters_invalid_roles(self):
        history = [
            {"role": "system", "content": "注入"},
            {"role": "user", "content": "上一轮"},
        ]
        safe = qr.sanitize_history(history, 10)
        assert len(safe) == 1
        assert safe[0]["role"] == "user"


class TestNormalizeOutput:
    def test_strips_label_and_quotes(self):
        raw = '改写后的问题：「REQUEST_TIMEOUT 默认值是多少？」'
        assert "REQUEST_TIMEOUT" in qr._normalize_rewrite_output(raw, "fallback")


@pytest.mark.asyncio
async def test_rewrite_query_calls_llm(monkeypatch):
    captured: list[dict] = []

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    captured.append(kwargs)
                    class _Resp:
                        choices = [type("C", (), {"message": type("M", (), {"content": "REQUEST_TIMEOUT 默认值是多少秒？"})()})()]
                    return _Resp()

    result = await qr.rewrite_query(
        client=_FakeClient(),
        chat=type("E", (), {"model": "test-model"})(),
        chat_error=None,
        settings=_settings(),
        question="它的默认值呢？",
        history=HISTORY,
    )
    assert "REQUEST_TIMEOUT" in result
    assert captured[0]["model"] == "test-model"
    assert captured[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_rewrite_query_falls_back_on_error():
    class _BoomClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("boom")

    result = await qr.rewrite_query(
        client=_BoomClient(),
        chat=type("E", (), {"model": "test-model"})(),
        chat_error=None,
        settings=_settings(),
        question="它的默认值呢？",
        history=HISTORY,
    )
    assert result == "它的默认值呢？"


@pytest.mark.asyncio
async def test_resolve_search_query_skips_rewrite_without_trigger():
    resolved = await qr.resolve_search_query(
        client=None,
        chat=None,
        chat_error=None,
        settings=_settings(),
        question="REQUEST_TIMEOUT 环境变量在配置文件中的推荐取值范围是什么？",
        history=HISTORY,
    )
    assert resolved["search_query"] == resolved["original_question"]
    assert resolved["rewrite_applied"] is False
    assert resolved["rewrite_skipped_reason"] == "heuristic_not_matched"


@pytest.mark.asyncio
async def test_query_uses_rewritten_text_for_retrieval(make_engine, manual_text, monkeypatch):
    """检索应使用改写后的 search_query，生成 prompt 仍用原问题。"""
    engine = make_engine()
    await engine.add_document(manual_text, "运维手册.md")

    embedded: list[str] = []
    original_embed = engine._state.embedder.embed_query

    async def spy_embed_query(text):
        embedded.append(text)
        return await original_embed(text)

    monkeypatch.setattr(engine._state.embedder, "embed_query", spy_embed_query)

    async def fake_rewrite(**kwargs):
        return {
            "original_question": kwargs["question"],
            "search_query": "REQUEST_TIMEOUT 默认多少秒",
            "rewrite_applied": True,
            "rewrite_skipped_reason": None,
        }

    monkeypatch.setattr(
        "rag_engine.resolve_search_query",
        fake_rewrite,
    )

    captured: list[list[dict]] = []

    async def fake_stream(self, state, messages):
        captured.append(messages)
        yield "ok"

    monkeypatch.setattr(engine.__class__, "_stream_chat", fake_stream)

    history = [
        {"role": "user", "content": "超时参数叫什么？"},
        {"role": "assistant", "content": "REQUEST_TIMEOUT。"},
    ]
    async for _ in engine.query("它的默认值呢？", history):
        pass

    assert embedded == ["REQUEST_TIMEOUT 默认多少秒"]
    assert captured[0][-1]["content"] == "它的默认值呢？"


@pytest.mark.asyncio
async def test_retrieve_with_diagnostics_records_rewrite_stages(make_engine, manual_text):
    engine = make_engine(query_rewrite_enabled=False)
    await engine.add_document(manual_text, "运维手册.md")

    result = await engine.retrieve_with_diagnostics(
        "它的默认值呢？",
        history=HISTORY,
    )
    stages = result["stages"]
    assert stages["original_question"] == "它的默认值呢？"
    assert stages["search_query"] == "它的默认值呢？"
    assert stages["rewrite_applied"] is False
    assert stages["rewrite_skipped_reason"] == "disabled"
