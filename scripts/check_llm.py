# -*- coding: utf-8 -*-
"""
检查一个 OpenAI 兼容端点是否可用。

在把内网自建服务接进应用之前先用这个脚本验一遍，可以把「地址不对」、
「模型名不对」、「需要令牌」这几类问题区分开。

用法:
    # 直接指定端点
    python scripts/check_llm.py --base-url http://192.168.1.10:11434/v1 --model qwen2.5:7b

    # 或读取当前配置里生效的对话端点
    python scripts/check_llm.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI  # noqa: E402

import config  # noqa: E402

# Windows 控制台默认 GBK，模型返回的内容可能含无法编码的字符
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def run(base_url: str, model: str, api_key: str, timeout: float) -> int:
    print(f"端点   : {base_url}")
    print(f"模型   : {model or '(未指定)'}")
    print(f"令牌   : {'已提供' if api_key and api_key != 'local' else '未提供（多数自建服务无需）'}")
    print()

    # 内网地址绕过 HTTP_PROXY，否则企业代理会拦下请求并返回一段 HTML 错误页
    http_client = None
    if config.is_internal_host(base_url):
        import httpx

        print("（内网地址，已绕过 HTTP 代理）\n")
        http_client = httpx.AsyncClient(trust_env=False, timeout=timeout)

    client = AsyncOpenAI(
        api_key=api_key or "local",
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
        http_client=http_client,
    )

    # 第一步：列出模型。地址错误、服务未启动会在这里暴露。
    available: list[str] = []
    try:
        resp = await client.models.list()
        available = sorted(item.id for item in resp.data)
        print(f"[OK]   连接成功，可用模型 {len(available)} 个：{', '.join(available) or '(空)'}")
    except Exception as exc:
        print(f"[FAIL] 无法获取模型列表：{type(exc).__name__}: {str(exc)[:200]}")
        print("       请检查地址是否正确（通常需要以 /v1 结尾）、服务是否已启动、网络是否可达。")
        return 1

    if model and available and model not in available:
        print(f"[WARN] 模型名 {model!r} 不在服务端列表中，对话可能返回 404。")

    if not model:
        print("\n未指定模型名，跳过对话测试。")
        return 0

    # 第二步：发一次流式对话，验证生成协议。
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：可用"}],
            stream=True,
        )
        pieces = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                pieces.append(chunk.choices[0].delta.content)
        answer = "".join(pieces).strip()
        if not answer:
            print("[WARN] 对话连通但没有返回任何内容。")
            return 1
        print(f"[OK]   流式对话正常，返回：{answer[:60]!r}")
        return 0
    except Exception as exc:
        print(f"[FAIL] 对话请求失败：{type(exc).__name__}: {str(exc)[:200]}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    if args.base_url:
        base_url, model, api_key = args.base_url, args.model, args.api_key
    else:
        # 未指定则读当前配置中生效的对话端点
        try:
            endpoint = config.resolve_llm(config.load_settings())
        except ValueError as exc:
            print(f"当前配置无法解析出对话端点：{exc}")
            return 1
        base_url, model, api_key = endpoint.base_url, endpoint.model, endpoint.api_key
        print(f"(读取当前配置，provider={endpoint.provider})")

    return asyncio.run(run(base_url, model, api_key, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
