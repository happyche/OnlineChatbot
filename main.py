# -*- coding: utf-8 -*-
"""
FastAPI 应用入口 — RAG 智能文档助手
====================================
路由一览：
    GET    /                       → 前端单页应用
    GET    /api/health             → 健康检查（含引擎状态）
    POST   /api/chat               → 流式问答（SSE）
    POST   /api/upload             → 上传 Markdown 文档并索引
    POST   /api/reindex            → 用当前配置重建全部索引
    GET    /api/documents          → 列出已入库文档
    DELETE /api/documents/{name}   → 删除文档（向量库 + 磁盘）
    GET    /api/settings           → 获取配置（不返回 API Key 明文）
    POST   /api/settings           → 更新配置并热重载引擎

安全相关：
  - API Key 绝不出现在任何响应体里，只返回脱敏后的展示串
  - 文件名一律取 basename 并做白名单校验，防止路径穿越读写项目外文件
  - CORS 默认只放行本机来源；设置 APP_API_TOKEN 后 /api/* 需带 X-API-Token
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")

STATIC_DIR = config.ROOT / "static"

#: 上传文件大小上限，防止超大文件把内存打满
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

#: 文件名白名单：禁止路径分隔符、盘符冒号、通配符等，且必须以 .md 结尾
_SAFE_NAME_RE = re.compile(r'^[^/\\:*?"<>|\x00-\x1f]+\.md$', re.IGNORECASE)


# ====================================================================
# 引擎持有者
# ====================================================================

class EngineHolder:
    """
    持有 RAGEngine 实例。

    引擎初始化会因缺少 API Key 等配置而失败。此时不能让整个进程起不来——
    否则用户没有任何途径通过 UI 补上配置。所以把失败信息记下来，
    业务接口返回 503 并说明原因，而 /api/settings 仍然可用。
    """

    def __init__(self):
        self._engine: Optional[RAGEngine] = None
        self._error: Optional[str] = None
        self.rebuild()

    def rebuild(self):
        try:
            self._engine = RAGEngine()
            self._error = None
            if self._engine.collection_reset_on_start:
                logger.warning("向量库因嵌入模型变更已重建，请调用 POST /api/reindex 重新索引。")
        except Exception as exc:
            self._engine = None
            self._error = str(exc)
            logger.error("RAG 引擎初始化失败: %s", exc)

    @property
    def error(self) -> Optional[str]:
        return self._error

    def get(self) -> RAGEngine:
        if self._engine is None:
            raise HTTPException(status_code=503, detail=f"RAG 引擎不可用: {self._error}")
        return self._engine

    def get_optional(self) -> Optional[RAGEngine]:
        """取引擎但不抛异常，供配置查询等即使引擎不可用也要能响应的接口使用。"""
        return self._engine


holder = EngineHolder()


def get_engine() -> RAGEngine:
    """FastAPI 依赖：取得可用的引擎，否则 503。"""
    return holder.get()


def require_token(x_api_token: Optional[str] = Header(default=None)):
    """FastAPI 依赖：配置了 APP_API_TOKEN 时校验请求头。"""
    if config.APP_API_TOKEN and x_api_token != config.APP_API_TOKEN:
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Token")


# ====================================================================
# 应用初始化
# ====================================================================

app = FastAPI(title="RAG 智能文档助手")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Token"],
)

config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


# ====================================================================
# 数据模型
# ====================================================================

class ChatRequest(BaseModel):
    """聊天请求：question 为当前提问，history 为前端维护的对话历史。"""

    question: str = Field(min_length=1, max_length=4000)
    history: Optional[list[dict]] = None


class SettingsRequest(BaseModel):
    """
    配置更新请求。字段全为 Optional，只提交需要修改的项，
    未提交的字段保持原值（通过 exclude_none 实现）。
    """

    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    local_llm_base_url: Optional[str] = None
    local_llm_model: Optional[str] = None
    local_llm_api_key: Optional[str] = None
    embedding_provider: Optional[str] = None
    local_embedding_model: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    chunk_size: Optional[int] = Field(default=None, ge=50, le=8000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=4000)
    min_similarity: Optional[float] = Field(default=None, ge=0, le=1)
    # 检索增强开关，可独立开闭以做消融对比
    hybrid_search_enabled: Optional[bool] = None
    bm25_tokenizer: Optional[str] = None
    rrf_k: Optional[int] = Field(default=None, ge=1, le=1000)
    rerank_enabled: Optional[bool] = None
    rerank_provider: Optional[str] = None
    rerank_model: Optional[str] = None
    min_rerank_score: Optional[str] = None
    candidate_pool_size: Optional[int] = Field(default=None, ge=1, le=200)


# ====================================================================
# 工具函数
# ====================================================================

def safe_md_filename(raw: str) -> str:
    """
    把用户提供的文件名规整为安全的 basename。

    跨平台地剥掉所有目录成分（Linux 上 Path().name 不会把反斜杠当分隔符，
    因此手工处理两种分隔符），再用白名单校验。
    这样 "../../etc/passwd.md"、"..%2Fconfig.py" 之类的输入都会被限制在 uploads/ 内。
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="非法文件名")
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="文件名不合法或不是 .md 文件")
    return name


def masked_key(key: str) -> str:
    """生成 API Key 的脱敏展示串。"""
    if not key:
        return ""
    return key[:8] + "***" + key[-4:] if len(key) > 12 else "***"


# ====================================================================
# 路由
# ====================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端单页应用。"""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@api.get("/health")
async def health():
    """健康检查：暴露引擎是否就绪，便于容器探针和排障。"""
    engine = holder.get_optional()
    return {
        "status": "ok" if holder.error is None else "degraded",
        "engine_error": holder.error,
        "active": engine.describe_endpoints() if engine else None,
    }


@api.post("/chat")
async def chat(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    """
    流式问答（Server-Sent Events）。

    engine.query 是异步生成器，用 async for 消费，
    网络等待期间事件循环可以调度其他请求，因此多用户并发不会相互阻塞。
    """
    async def generate():
        try:
            async for chunk in engine.query(req.question, req.history):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("问答失败")
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.post("/retrieve")
async def retrieve(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    """
    只做检索、不生成回答，返回命中片段与各阶段得分明细。

    存在的意义是把「检索质量」与「生成质量」分开评估：
    RAGAS 的 context_precision / context_recall 只需要 contexts，
    走这个接口既省掉生成的时间与费用，也避免生成环节的波动干扰归因。
    stages 字段记录本次生效的开关与各阶段候选数，便于消融实验对照。
    """
    result = await engine.retrieve_with_diagnostics(
        req.question, history=req.history
    )
    return {
        "question": req.question,
        "search_query": result["stages"].get("search_query", req.question),
        "stages": result["stages"],
        "contexts": [
            {
                "text": hit.text,
                "source": hit.source,
                "heading": hit.heading,
                "citation": hit.citation,
                "score": hit.score,
                "score_type": hit.score_type,
                "similarity": hit.similarity,
                "bm25_score": hit.bm25_score,
                "rrf_score": hit.rrf_score,
                "rerank_score": hit.rerank_score,
            }
            for hit in result["hits"]
        ],
    }


@api.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    engine: RAGEngine = Depends(get_engine),
):
    """上传 Markdown 文档：校验文件名与大小，落盘后切分入库。"""
    results = []
    for file in files:
        try:
            name = safe_md_filename(file.filename or "")
            raw = await file.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限",
                )
            text = raw.decode("utf-8")
            (config.UPLOADS_DIR / name).write_text(text, encoding="utf-8")
            chunks = await engine.add_document(text, name)
            results.append({"filename": name, "chunks": chunks, "status": "success"})
        except HTTPException as exc:
            results.append({"filename": file.filename, "error": exc.detail})
        except UnicodeDecodeError:
            results.append({"filename": file.filename, "error": "文件不是 UTF-8 编码的文本"})
        except Exception as exc:
            logger.exception("上传处理失败: %s", file.filename)
            results.append({"filename": file.filename, "error": str(exc)})
    return {"results": results}


@api.post("/reindex")
async def reindex(engine: RAGEngine = Depends(get_engine)):
    """
    用当前配置重建 uploads/ 下所有文档的索引。

    切换嵌入模型或调整 chunk 参数后必须执行：旧向量由旧模型/旧切分产出，
    与新的查询向量不可比。
    """
    return {"results": await engine.reindex()}


@api.get("/documents")
async def list_documents(engine: RAGEngine = Depends(get_engine)):
    """列出知识库中所有已入库的文档。"""
    return {"documents": engine.list_documents()}


@api.delete("/documents/{filename}")
async def delete_document(filename: str, engine: RAGEngine = Depends(get_engine)):
    """删除指定文档：先清向量库，再删磁盘文件。"""
    name = safe_md_filename(filename)
    engine.remove_document(name)
    path = config.UPLOADS_DIR / name
    if path.exists():
        path.unlink()
    return {"status": "success"}


@api.get("/settings")
async def get_settings():
    """
    获取当前配置。

    所有以 _api_key 结尾的字段都会被移除，只保留脱敏后的 *_display。
    这里按后缀统一处理而不是逐个点名，是为了将来新增密钥字段时不会漏掉——
    原实现只处理了 llm_api_key 且仅新增展示字段而没移除原字段，导致 Key 泄漏。

    另外回显当前实际生效的端点：配置分云端/自建两套，
    只看配置项很难判断哪套在生效。
    """
    settings = config.load_settings()
    safe = {}
    for key, value in settings.items():
        if key.endswith("_api_key"):
            safe[f"{key}_display"] = masked_key(value or "")
        else:
            safe[key] = value

    safe["active"] = (
        holder.get_optional().describe_endpoints()
        if holder.get_optional()
        else {"llm_error": holder.error}
    )
    return safe


@api.get("/models")
async def list_models(engine: RAGEngine = Depends(get_engine)):
    """
    列出对话端点上可用的模型，同时充当连通性检查。

    自建服务最常见的两类问题是地址写错和模型名写错，这个接口能同时暴露两者。
    """
    try:
        return {"models": await engine.list_chat_models()}
    except Exception as exc:
        # 连不通是配置问题而不是服务故障，用 502 并带上原始错误便于排障
        raise HTTPException(
            status_code=502, detail=f"无法从对话端点获取模型列表: {exc}"
        ) from exc


@api.post("/settings")
async def update_settings(req: SettingsRequest):
    """更新配置、落盘并热重载引擎。"""
    current = config.load_settings()
    current.update(req.model_dump(exclude_none=True))
    config.save_settings(current)

    holder.rebuild()
    if holder.error:
        # 引擎完全起不来（例如嵌入配置非法）才算保存失败
        raise HTTPException(status_code=400, detail=f"配置已保存但引擎初始化失败: {holder.error}")

    engine = holder.get()
    active = engine.describe_endpoints()
    return {
        "status": "success",
        # 嵌入模型变更会导致向量库重建，前端据此提示用户重新索引
        "collection_reset": engine.collection_reset_on_start,
        # 对话端点配置不全属于部分失败：设置照常保存（否则 LLM 没配好时
        # 连检索参数都改不了），但必须显式回报，不能等用户提问才暴露。
        "llm_error": active["llm_error"],
        "active": active,
    }


app.include_router(api)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
