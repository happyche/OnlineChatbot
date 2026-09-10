# -*- coding: utf-8 -*-
"""
Docker 部署后的冒烟验证。

验证项:
  1. /api/health 可用且引擎就绪
  2. 混合检索 + 重排开关已生效（stages 字段）
  3. 上传文档 → 检索命中（不依赖 LLM Key）

用法:
    python scripts/docker_verify.py
    python scripts/docker_verify.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOC_NAME = "docker-verify.md"
DOC_TEXT = """# Docker 验证文档

## 超时设置

REQUEST_TIMEOUT 默认为 60 秒。

## 配置项

MIN_SIMILARITY 用于过滤向量检索结果。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    headers = {"X-API-Token": args.token} if args.token else {}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=180)
    failed = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failed
        if not cond:
            failed += 1
        mark = "PASS" if cond else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))

    resp = client.get("/api/health")
    body = resp.json()
    ok("health", resp.status_code == 200)
    ok("engine ready", body.get("status") == "ok", body.get("engine_error") or "")

    settings = client.get("/api/settings").json()
    ok("hybrid enabled", settings.get("hybrid_search_enabled") is True)
    ok("rerank enabled", settings.get("rerank_enabled") is True)

    files = {"files": (DOC_NAME, BytesIO(DOC_TEXT.encode("utf-8")), "text/markdown")}
    resp = client.post("/api/upload", files=files)
    ok("upload", resp.status_code == 200, resp.text[:120])

    resp = client.post(
        "/api/retrieve",
        json={"question": "REQUEST_TIMEOUT 默认是多少秒？", "history": []},
    )
    data = resp.json()
    stages = data.get("stages") or {}
    ok("retrieve", resp.status_code == 200 and len(data.get("contexts", [])) > 0)
    ok("hybrid stage", stages.get("hybrid_search_enabled") is True, json.dumps(stages, ensure_ascii=False))
    ok("rerank stage", stages.get("rerank_enabled") is True)

    hits = data.get("contexts") or []
    if hits:
        ok("rerank score present", hits[0].get("rerank_score") is not None)

    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项失败'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
