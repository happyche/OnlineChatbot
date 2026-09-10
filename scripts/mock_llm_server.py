# -*- coding: utf-8 -*-
"""
最小的 OpenAI 兼容对话服务（测试替身）。

用途：在没有真实自建推理服务可用时，验证 llm_provider=local 这条链路
是否真的打通——包括模型列表、流式协议、以及检索到的上下文有没有送进 prompt。
它不做任何推理，只把收到的内容回显出来。

用法:
    python scripts/mock_llm_server.py --port 9911
    # 然后把应用指向它：
    #   LLM_PROVIDER=local
    #   LOCAL_LLM_BASE_URL=http://127.0.0.1:9911/v1
    #   LOCAL_LLM_MODEL=mock-model

它实现了两个接口：
    GET  /v1/models
    POST /v1/chat/completions   （支持 stream=true / false）
"""
from __future__ import annotations

import argparse
import json
import re
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_IDS = ["mock-model", "mock-model-large"]

app = FastAPI(title="Mock OpenAI-compatible LLM")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float | None = None
    stream: bool = False


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "mock"}
            for mid in MODEL_IDS
        ],
    }


def build_reply(req: ChatRequest) -> str:
    """把收到的 prompt 回显成一段可断言的文本。"""
    system = next((m.get("content", "") for m in req.messages if m.get("role") == "system"), "")
    question = next(
        (m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"), ""
    )
    # 引擎组装上下文时每条资料都带 "[资料N｜出处: ...]" 标记，据此统计条数
    citations = re.findall(r"\[资料\d+｜出处: ([^\]]+)\]", system)
    return (
        f"[mock-llm model={req.model}] "
        f"收到参考资料 {len(citations)} 条"
        + (f"（出处: {'; '.join(citations)}）" if citations else "（无）")
        + f"；你的问题是「{question}」"
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    reply = build_reply(req)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    if not req.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def generate():
        def chunk(delta: dict, finish=None) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield chunk({"role": "assistant", "content": ""})
        # 切成小段输出，模拟逐 token 流式返回
        for i in range(0, len(reply), 12):
            yield chunk({"content": reply[i : i + 12]})
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9911)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
