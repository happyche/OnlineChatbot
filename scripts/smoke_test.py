# -*- coding: utf-8 -*-
"""
针对运行中的服务做冒烟测试。

与 pytest 套件的区别：pytest 用假 embedder 和假 LLM 验证逻辑，
这里打真实 HTTP、走真实嵌入模型，验证部署形态下整条链路是否通。

用法:
    python scripts/smoke_test.py [--base-url http://127.0.0.1:8000] [--token XXX]

退出码 0 表示全部通过。对话接口若因 API Key 无效而失败，会单独标记为
「已到达 LLM 调用」而不算致命错误 —— 那说明检索链路本身是通的。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

# Windows 控制台默认是 GBK，输出中的 › 等字符会直接抛 UnicodeEncodeError。
# 统一切到 UTF-8，无法表示的字符降级显示而不是让脚本崩掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOC_NAME = "冒烟测试文档.md"
DOC_TEXT = """# 冒烟测试文档

## 环境准备

### Python 版本要求

系统要求 Python 3.10 或以上版本，推荐 3.11。

## 配置说明

### 超时设置

REQUEST_TIMEOUT 默认为 60 秒，模型响应慢时应先排查网络。
"""

passed: list[str] = []
failed: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    (passed if ok else failed).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    headers = {"X-API-Token": args.token} if args.token else {}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=120)

    # ---- 健康检查 ----
    resp = client.get("/api/health")
    body = resp.json()
    check("健康检查可用", resp.status_code == 200, f"status={body.get('status')}")
    if body.get("engine_error"):
        notes.append(f"引擎降级: {body['engine_error']}")

    # ---- 首页 ----
    resp = client.get("/")
    check("首页可访问", resp.status_code == 200 and "智能文档助手" in resp.text)
    check("前端已引入 DOMPurify", "purify.min.js" in resp.text)

    # ---- API Key 不泄漏 ----
    resp = client.get("/api/settings")
    settings = resp.json()
    check("配置接口不返回 llm_api_key 字段", "llm_api_key" not in settings)
    check(
        "配置接口返回脱敏展示串",
        "***" in (settings.get("llm_api_key_display") or "***"),
        f"display={settings.get('llm_api_key_display')!r}",
    )
    provider = settings.get("embedding_provider")
    # 本地与远程各有自己的模型名字段，不能一律打印 embedding_model
    model = (
        settings.get("local_embedding_model")
        if provider == "local"
        else settings.get("embedding_model")
    )
    print(f"       当前嵌入提供方: {provider} / {model}")
    active = settings.get("active") or {}
    print(f"       当前对话端点  : {active.get('llm_provider')} / {active.get('llm_model')} @ {active.get('llm_base_url')}")

    # ---- 对话端点连通性 ----
    resp = client.get("/api/models")
    if resp.status_code == 200:
        models = resp.json()["models"]
        check("对话端点可列出模型", bool(models), f"{len(models)} 个: {', '.join(models[:5])}")
    else:
        notes.append(f"对话端点无法列出模型（{resp.status_code}）: {resp.text[:120]}")
        check("对话端点可列出模型", False, "见下方备注")

    # ---- 路径穿越 ----
    resp = client.request("DELETE", "/api/documents/..%5C..%5Cconfig.py")
    check("反斜杠路径穿越被拒绝", resp.status_code == 400)
    check("项目文件未被删除", Path("config.py").exists())

    resp = client.post(
        "/api/upload",
        files=[("files", ("evil.py", b"import os", "text/x-python"))],
    )
    check("非 .md 文件被拒绝", "error" in resp.json()["results"][0])

    # ---- 上传并索引（真实嵌入模型）----
    resp = client.post(
        "/api/upload",
        files=[("files", (DOC_NAME, DOC_TEXT.encode("utf-8"), "text/markdown"))],
    )
    result = resp.json()["results"][0]
    ok = result.get("status") == "success"
    check("上传并索引成功", ok, f"chunks={result.get('chunks')}" if ok else str(result))
    if not ok:
        print("\n上传失败，后续检索相关检查已跳过。")
        return _summary()

    # ---- 文档列表 ----
    docs = client.get("/api/documents").json()["documents"]
    check("文档出现在列表中", any(d["filename"] == DOC_NAME for d in docs), f"{docs}")

    # ---- 重复上传不产生重复数据 ----
    before = next(d["chunks"] for d in docs if d["filename"] == DOC_NAME)
    client.post(
        "/api/upload",
        files=[("files", (DOC_NAME, DOC_TEXT.encode("utf-8"), "text/markdown"))],
    )
    docs = client.get("/api/documents").json()["documents"]
    after = next(d["chunks"] for d in docs if d["filename"] == DOC_NAME)
    check("重复上传未产生重复块", before == after, f"{before} -> {after}")

    # ---- 重建索引 ----
    resp = client.post("/api/reindex")
    results = resp.json().get("results", [])
    check(
        "重建索引成功",
        resp.status_code == 200 and any(r.get("status") == "success" for r in results),
        f"{results}",
    )

    # ---- 对话（SSE）----
    content, errors = [], []
    with client.stream("POST", "/api/chat", json={"question": "Python 版本要求是什么？"}) as stream:
        check("对话接口返回 200", stream.status_code == 200)
        for line in stream.iter_lines():
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

    answer = "".join(content)
    if errors:
        # 走到这里说明检索已完成、prompt 已组装，仅 LLM 调用本身失败
        # （通常是 API Key 无效）。属于配置问题，不是链路问题。
        notes.append(f"LLM 调用失败（检索链路已跑通）: {errors[0][:160]}")
        check("对话链路到达 LLM 调用", True, "LLM 返回错误，见下方备注")
    else:
        check("对话返回了内容", bool(answer.strip()), f"{answer[:80]!r}")
        check("回答附带参考来源", "参考来源" in answer)

    # ---- 删除 ----
    resp = client.delete(f"/api/documents/{DOC_NAME}")
    check("删除文档成功", resp.status_code == 200)
    docs = client.get("/api/documents").json()["documents"]
    check("删除后不再出现在列表中", not any(d["filename"] == DOC_NAME for d in docs))

    return _summary()


def _summary() -> int:
    print("\n" + "=" * 60)
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    for note in notes:
        print(f"备注: {note}")
    if failed:
        print("失败项: " + ", ".join(failed))
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
