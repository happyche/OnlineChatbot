# -*- coding: utf-8 -*-
"""
接口端到端测试：上传 → 列表 → 问答 → 重建索引 → 删除。

LLM 调用被替换为假实现（见 conftest 的 client fixture），它会记录送入模型的
完整 messages，因此可以断言「检索到的文档内容确实进入了 prompt」，
而不是只验证接口返回了 200。
"""
from __future__ import annotations

import json

import config


def _sse_texts(raw: str) -> tuple[str, list[str]]:
    """解析 SSE 响应，返回 (拼接后的正文, 错误列表)。"""
    content, errors = [], []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        if "error" in obj:
            errors.append(obj["error"])
        elif "content" in obj:
            content.append(obj["content"])
    return "".join(content), errors


def _upload(client, name: str, text: str):
    return client.post(
        "/api/upload",
        files=[("files", (name, text.encode("utf-8"), "text/markdown"))],
    )


class TestHealth:
    def test_health_reports_ok(self, client):
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["engine_error"] is None

    def test_index_page_served(self, client):
        resp = client.get("/")

        assert resp.status_code == 200
        assert "智能文档助手" in resp.text


class TestUploadAndList:
    def test_upload_indexes_document(self, client, manual_text):
        resp = _upload(client, "运维手册.md", manual_text)

        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["status"] == "success"
        assert result["chunks"] > 5
        assert (config.UPLOADS_DIR / "运维手册.md").exists()

    def test_documents_listed_after_upload(self, client, manual_text):
        _upload(client, "运维手册.md", manual_text)

        docs = client.get("/api/documents").json()["documents"]

        assert len(docs) == 1
        assert docs[0]["filename"] == "运维手册.md"
        assert docs[0]["chunks"] > 5

    def test_reupload_does_not_duplicate(self, client, manual_text):
        first = _upload(client, "运维手册.md", manual_text).json()["results"][0]["chunks"]
        _upload(client, "运维手册.md", manual_text)

        docs = client.get("/api/documents").json()["documents"]

        assert len(docs) == 1
        assert docs[0]["chunks"] == first

    def test_non_utf8_upload_reports_error(self, client):
        resp = client.post(
            "/api/upload",
            files=[("files", ("gbk.md", "中文内容".encode("gbk"), "text/markdown"))],
        )

        assert "error" in resp.json()["results"][0]

    def test_multiple_files_in_one_request(self, client, manual_text):
        resp = client.post(
            "/api/upload",
            files=[
                ("files", ("甲.md", manual_text.encode("utf-8"), "text/markdown")),
                ("files", ("乙.md", "# 乙\n\n乙的内容。\n".encode("utf-8"), "text/markdown")),
            ],
        )

        results = resp.json()["results"]
        assert len(results) == 2
        assert all(r.get("status") == "success" for r in results)


class TestChat:
    def test_chat_streams_answer(self, client, manual_text):
        _upload(client, "运维手册.md", manual_text)

        resp = client.post("/api/chat", json={"question": "Python 版本要求是什么？"})

        assert resp.status_code == 200
        text, errors = _sse_texts(resp.text)
        assert not errors
        assert "模拟回答" in text
        assert "data: [DONE]" in resp.text

    def test_retrieved_content_reaches_the_prompt(self, client, manual_text):
        """RAG 的关键断言：检索到的文档内容必须真的出现在 system prompt 里。"""
        _upload(client, "运维手册.md", manual_text)

        client.post("/api/chat", json={"question": "Python 版本要求是什么？"})

        system_prompt = client.captured_messages[-1][0]["content"]
        assert "参考资料" in system_prompt
        assert "3.10" in system_prompt, "召回的正确内容没有进入 prompt"
        assert "运维手册.md" in system_prompt

    def test_citation_footer_appended(self, client, manual_text):
        _upload(client, "运维手册.md", manual_text)

        resp = client.post("/api/chat", json={"question": "依赖怎么安装？"})

        text, _ = _sse_texts(resp.text)
        assert "参考来源：" in text
        assert "运维手册.md" in text

    def test_empty_knowledge_base_tells_model_not_to_fabricate(self, client):
        resp = client.post("/api/chat", json={"question": "任何问题"})

        assert resp.status_code == 200
        system_prompt = client.captured_messages[-1][0]["content"]
        assert "不要编造" in system_prompt

    def test_history_is_forwarded(self, client, manual_text):
        _upload(client, "运维手册.md", manual_text)

        client.post(
            "/api/chat",
            json={
                "question": "那超时呢？",
                "history": [
                    {"role": "user", "content": "TOP_K 是什么"},
                    {"role": "assistant", "content": "检索返回的片段数"},
                ],
            },
        )

        messages = client.captured_messages[-1]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[-1]["content"] == "那超时呢？"

    def test_blank_question_rejected(self, client):
        assert client.post("/api/chat", json={"question": ""}).status_code == 422


class TestReindex:
    def test_reindex_rebuilds_from_uploads(self, client, manual_text):
        _upload(client, "运维手册.md", manual_text)
        original = client.get("/api/documents").json()["documents"][0]["chunks"]

        resp = client.post("/api/reindex")

        assert resp.status_code == 200
        assert resp.json()["results"][0]["status"] == "success"
        assert client.get("/api/documents").json()["documents"][0]["chunks"] == original

    def test_reindex_on_empty_uploads_is_noop(self, client):
        resp = client.post("/api/reindex")

        assert resp.status_code == 200
        assert resp.json()["results"] == []


class TestDelete:
    def test_delete_removes_vectors_and_file(self, client, manual_text):
        _upload(client, "运维手册.md", manual_text)

        resp = client.delete("/api/documents/运维手册.md")

        assert resp.status_code == 200
        assert client.get("/api/documents").json()["documents"] == []
        assert not (config.UPLOADS_DIR / "运维手册.md").exists()

    def test_delete_missing_document_is_idempotent(self, client):
        assert client.delete("/api/documents/不存在.md").status_code == 200


class TestSettings:
    def test_update_settings_persists_and_reloads(self, client):
        resp = client.post("/api/settings", json={"top_k": 7, "temperature": 0.3})

        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        body = client.get("/api/settings").json()
        assert body["top_k"] == 7
        assert body["temperature"] == 0.3

    def test_settings_take_effect_on_retrieval(self, client, manual_text):
        """改 top_k 后应立刻影响检索条数，验证热重载真的生效。"""
        _upload(client, "运维手册.md", manual_text)
        client.post("/api/settings", json={"top_k": 1})

        client.post("/api/chat", json={"question": "配置说明"})

        system_prompt = client.captured_messages[-1][0]["content"]
        assert system_prompt.count("｜出处:") == 1

    def test_out_of_range_values_rejected(self, client):
        assert client.post("/api/settings", json={"top_k": 0}).status_code == 422
        assert client.post("/api/settings", json={"temperature": 5}).status_code == 422
        assert client.post("/api/settings", json={"chunk_size": 10}).status_code == 422

    def test_unknown_fields_ignored(self, client):
        resp = client.post("/api/settings", json={"top_k": 4, "evil_field": "x"})

        assert resp.status_code == 200
        assert "evil_field" not in client.get("/api/settings").json()
