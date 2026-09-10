# -*- coding: utf-8 -*-
"""
接口安全测试。

覆盖两个原实现的漏洞：
  1. GET /api/settings 把完整 API Key 随响应返回（只加了脱敏字段，没移除原字段），
     配合 CORS 全开放与零鉴权，任何人都能取走密钥
  2. 上传与删除接口直接拼接用户提供的文件名，可通过 ../ 读写删除项目外文件
"""
from __future__ import annotations

import config
from main import masked_key, safe_md_filename


class TestApiKeyNeverLeaks:
    def test_settings_response_has_no_raw_key(self, client):
        resp = client.get("/api/settings")

        assert resp.status_code == 200
        body = resp.json()
        assert "llm_api_key" not in body, "响应体中仍包含完整 API Key 字段"

    def test_raw_key_value_absent_from_response_text(self, client):
        real_key = config.load_settings()["llm_api_key"]

        raw_text = client.get("/api/settings").text

        assert real_key
        assert real_key not in raw_text, "API Key 明文出现在响应中"

    def test_masked_display_is_present_and_partial(self, client):
        body = client.get("/api/settings").json()

        display = body["llm_api_key_display"]
        assert "***" in display
        assert display != config.load_settings()["llm_api_key"]

    def test_masking_helper_behaviour(self):
        assert masked_key("") == ""
        assert masked_key("short") == "***"
        assert masked_key("sk-1234567890abcdef") == "sk-12345***cdef"


class TestFilenameSanitisation:
    def test_rejects_parent_directory_traversal(self, client):
        resp = client.delete("/api/documents/..%2F..%2Fconfig.py")

        # 带路径分隔符的请求会在路由层就匹配不上（404）；
        # 关键的安全属性是：请求不成功，且项目文件没被删掉
        assert resp.status_code != 200
        assert (config.ROOT / "config.py").exists()

    def test_rejects_windows_style_traversal(self, client):
        resp = client.delete("/api/documents/..%5C..%5Cconfig.py")

        # 反斜杠不是 URL 路径分隔符，因此这次会进到处理函数，
        # 由 safe_md_filename 剥掉目录成分后因扩展名不合法被拒
        assert resp.status_code == 400
        assert (config.ROOT / "config.py").exists()

    def test_traversal_cannot_delete_files_outside_uploads(self, client, manual_text):
        """构造一个能进到处理函数的穿越名，确认它只作用于 uploads/ 内。"""
        outside = config.ROOT / "should-not-be-deleted.md"
        outside.write_text("# 不该被删除\n", encoding="utf-8")
        try:
            resp = client.delete("/api/documents/..%5Cshould-not-be-deleted.md")

            assert resp.status_code == 200  # 被规整成 uploads/ 下的同名文件
            assert outside.exists(), "路径穿越删除了 uploads/ 之外的文件"
        finally:
            outside.unlink(missing_ok=True)

    def test_upload_with_traversal_name_lands_in_uploads_dir(self, client):
        resp = client.post(
            "/api/upload",
            files=[("files", ("../../evil.md", b"# \xe6\x81\xb6\xe6\x84\x8f", "text/markdown"))],
        )

        assert resp.status_code == 200
        # 目录成分被剥离，文件只能落在 uploads/ 内
        assert not (config.ROOT.parent / "evil.md").exists()
        assert not (config.ROOT / "evil.md").exists()
        assert (config.UPLOADS_DIR / "evil.md").exists()

    def test_non_markdown_upload_rejected(self, client):
        resp = client.post(
            "/api/upload",
            files=[("files", ("payload.py", b"import os", "text/x-python"))],
        )

        result = resp.json()["results"][0]
        assert "error" in result
        assert not (config.UPLOADS_DIR / "payload.py").exists()

    def test_sanitiser_strips_directories(self):
        assert safe_md_filename("../../etc/passwd.md") == "passwd.md"
        assert safe_md_filename("..\\..\\windows\\a.md") == "a.md"
        assert safe_md_filename("normal name (2).md") == "normal name (2).md"

    def test_sanitiser_rejects_bad_names(self):
        import pytest
        from fastapi import HTTPException

        for bad in ["", "   ", ".hidden.md", "a.txt", "no-extension", 'quote".md', "a<b>.md"]:
            with pytest.raises(HTTPException):
                safe_md_filename(bad)


class TestAuthToken:
    def test_token_required_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(config, "APP_API_TOKEN", "secret-token")

        assert client.get("/api/documents").status_code == 401
        assert (
            client.get("/api/documents", headers={"X-API-Token": "secret-token"}).status_code
            == 200
        )

    def test_no_token_required_by_default(self, client):
        assert client.get("/api/documents").status_code == 200


class TestCorsDefaults:
    def test_wildcard_origin_not_used(self):
        """本服务持有密钥且能删文件，不应放行任意来源。"""
        assert "*" not in config.CORS_ORIGINS
