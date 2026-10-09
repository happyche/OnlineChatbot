# -*- coding: utf-8 -*-
"""
FastAPI 应用入口 — RAG 智能文档助手
====================================
路由一览：
    GET    /                       → 前端单页应用
    GET    /api/health             → 健康检查（含引擎状态）
    POST   /api/chat               → 流式问答（SSE）
    POST   /api/agent              → 非流式跑一次 agent，返回事件轨迹（评测用）
    POST   /api/upload             → 上传 Markdown 文档并索引
    POST   /api/reindex            → 用当前配置重建全部索引
    GET    /api/documents          → 列出已入库文档
    DELETE /api/documents/{name}   → 删除文档（向量库 + 磁盘）
    GET    /api/settings           → 获取配置（不返回 API Key 明文）
    POST   /api/settings           → 更新配置并热重载引擎
    POST   /api/feedback           → 对某个 trace_id 提交 👍/👎
    GET    /api/feedback/stats     → 满意度、差评原因分布、分路径的成本与延迟
    GET    /api/feedback/badcases  → 导出差评现场，用于生成回归题集
    GET    /metrics                → Prometheus 指标

安全相关：
  - API Key 绝不出现在任何响应体里，只返回脱敏后的展示串
  - 文件名一律取 basename 并做白名单校验，防止路径穿越读写项目外文件
  - CORS 默认只放行本机来源；设置 APP_API_TOKEN 后 /api/* 需带 X-API-Token

可观测性：
  每个请求分配一个 trace_id，从响应头 X-Trace-Id 回传。它是把三件事串起来的
  唯一线索——一行结构化请求日志、一条用户反馈、和排查时 grep 出来的那几十行
  上下文。前端拿它提交反馈，因此它必须先于响应体产生（见 chat 里的 headers）。
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import config
import feedback as feedback_store
import observability
from rag_engine import RAGEngine

def _rotating_file(path_str: str, formatter: logging.Formatter) -> logging.Handler:
    """
    建一个滚动文件 handler。

    文件必须显式指定编码。Windows 的默认编码是 GBK，而日志里有「文档 › 章节」
    这类字符，用默认编码写文件会在运行中抛 UnicodeEncodeError——
    一条日志写不出去反而把请求搞挂，是典型的「排障手段自己成了故障源」。

    滚动而不是单文件：日志只增不减就是一条磁盘泄漏，和会话存储要上限同理。
    """
    from logging.handlers import RotatingFileHandler

    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=config.LOG_MAX_MB * 1024 * 1024,
        backupCount=config.LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def _configure_logging():
    """
    配置日志：控制台 + 可选的滚动文件，外加可选的独立结构化请求日志。

    每条记录都带 trace_id。它是把三件事串起来的唯一线索：一行结构化请求日志、
    一条用户反馈、和排查时在 app.log 里 grep 出来的那几十行上下文。
    没有它，「用户说昨天下午那一问答错了」就只能靠时间戳去猜。

    trace_id 靠装在 handler 上的过滤器补齐，而不是靠调用方传。format 里一旦
    写了 %(trace_id)s，任何缺这个字段的记录都会让日志系统自己抛异常——
    包括 httpx、chromadb 这些第三方库打的日志。
    """
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] [%(trace_id)s] %(message)s"
    )
    trace_filter = observability.TraceIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    handlers: list[logging.Handler] = [console]
    if config.LOG_FILE:
        handlers.append(_rotating_file(config.LOG_FILE, fmt))
    for handler in handlers:
        handler.addFilter(trace_filter)

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        handlers=handlers,
        force=True,
    )

    # 结构化请求日志：一行一个 JSON，直接可喂给 jq / pandas / Loki。
    # 给它一个独立文件是因为消费方式完全不同——人读日志带缩进、带多行栈回溯，
    # 混在一起会让两者都变难用。未配置则继续沿用上面的 handler。
    if config.REQUEST_LOG_FILE:
        request_logger = logging.getLogger("request")
        request_logger.handlers.clear()
        # message 本身就是完整的 JSON，再套一层时间戳前缀就不是合法 JSONL 了
        request_logger.addHandler(
            _rotating_file(config.REQUEST_LOG_FILE, logging.Formatter("%(message)s"))
        )
        request_logger.propagate = False
        request_logger.setLevel(logging.INFO)

    if config.AGENT_TRACE:
        # trace 自己是 DEBUG 级的，根级别若停在 INFO 就一条都看不到。
        # 只放开这一个 logger，不把 httpx / chromadb 的 DEBUG 一起灌进来。
        logging.getLogger("agent.trace").setLevel(logging.DEBUG)
        logging.getLogger("app").warning(
            "AGENT_TRACE 已开启：提问与文档原文会被完整写进日志，排查完请关掉。"
        )


_configure_logging()
logger = logging.getLogger("app")

STATIC_DIR = config.ROOT / "static"

#: 上传文件大小上限，防止超大文件把内存打满
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

#: 文件名白名单：禁止路径分隔符、盘符冒号、通配符等，且必须以 .md 结尾
_SAFE_NAME_RE = re.compile(r'^[^/\\:*?"<>|\x00-\x1f]+\.md$', re.IGNORECASE)

#: 客户端传入的 trace id 白名单。它会被写进日志文件，不校验就是一条日志注入：
#: 一个换行符足以在 app.log 里伪造出一整条不存在的记录。
_SAFE_TRACE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期：只负责会话库的开与关。

    引擎刻意不在这里初始化（见 EngineHolder）：它必须容忍失败，而 lifespan 里
    抛异常会让进程根本起不来，用户也就没有任何途径通过 UI 补上配置。
    会话库沿用同一条准则——打不开就记下原因、退回内存存储，
    由 /api/health/detail 回答「会话为什么没生效」，而不是让整个服务起不来。

    只在开关打开时才打开库：关着的时候不该在磁盘上留下一个会话文件。
    """
    import agent

    settings = config.load_settings()
    if not (_sessions_enabled(settings) and agent.is_available()):
        yield
        return

    # 用 ExitStack 而不是直接 `async with ... : yield`：后者会把应用运行期
    # 抛出的异常一并收进 except，然后在已经 yield 过一次的生成器里再 yield
    # 一次——ASGI 服务器只会得到一句 "generator didn't stop"。
    # try 必须只罩住「打开」这一步。
    async with AsyncExitStack() as stack:
        try:
            await stack.enter_async_context(
                agent.open_session_store(config.AGENT_SESSION_DB)
            )
        except Exception as exc:
            logger.error("会话库打开失败，会话退回内存存储（重启即清空）: %s", exc)
            agent.set_session_store_error(str(exc))
        yield


app = FastAPI(
    title="RAG 智能文档助手",
    lifespan=lifespan,
    # /docs 会把整个接口清单连同请求体结构一起摆出来。管理面关闭时
    # 它只剩对话那几个接口，但给使用者的部署本就不需要交互式文档，
    # 少一个可探测面总是好的。
    docs_url="/docs" if config.ADMIN_ENABLED else None,
    redoc_url="/redoc" if config.ADMIN_ENABLED else None,
    openapi_url="/openapi.json" if config.ADMIN_ENABLED else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Token", "X-Trace-Id"],
    # 跨域下响应头默认对 JS 不可见。不放行这一项，前端就拿不到 trace_id，
    # 也就没法提交反馈——而这个失败是静默的（header 读出来是 null）。
    expose_headers=["X-Trace-Id"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """
    给每个请求分配 trace_id，写进上下文并从响应头回传。

    接受客户端传来的 X-Trace-Id 以便串联上游调用，但要限长和过滤字符：
    它会被写进日志文件，放任任意内容进来就是一条日志注入
    （换行符可以伪造出一整条不存在的日志）。
    """
    incoming = (request.headers.get("X-Trace-Id") or "").strip()
    trace_id = (
        incoming[:64]
        if incoming and _SAFE_TRACE_RE.match(incoming)
        else observability.new_trace_id()
    )
    observability.set_trace_id(trace_id)

    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

#: 给使用者的接口：提问、健康探测、提交反馈。
api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])

#: 管理面接口：文档管理、模型配置、检索调试、反馈统计。
#:
#: ADMIN_ENABLED 为假时这个 router **不会被注册**，路径返回 404。
#: 用 404 而不是 403 是有意的：403 等于确认「这个接口存在，只是你没权限」，
#: 而给使用者的那份部署里，这些接口对他们来说本就不该存在。
admin = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


# ====================================================================
# 数据模型
# ====================================================================

class ChatRequest(BaseModel):
    """
    聊天请求：question 为当前提问，history 为前端维护的对话历史。

    session_id 存在时（需 agent_enabled + agent_session_enabled）改由服务端按该 id
    持久化对话，history 被忽略。它是 bearer 凭据而非身份认证——拿到 id 的人就是
    会话的主人，所以公网暴露时必须同时配置 APP_API_TOKEN。
    """

    question: str = Field(min_length=1, max_length=4000)
    history: Optional[list[dict]] = None
    session_id: Optional[str] = Field(default=None, max_length=128)


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
    # Agent 层。上限刻意收窄：每多一次模型调用就多几秒等待，
    # 放开到几十次只会让人在界面上误设出一个几分钟不出字的配置。
    agent_enabled: Optional[bool] = None
    agent_max_model_calls: Optional[int] = Field(default=None, ge=2, le=20)
    agent_recursion_limit: Optional[int] = Field(default=None, ge=4, le=100)
    agent_tool_payload_tokens: Optional[int] = Field(default=None, ge=100, le=50000)
    agent_expand_payload_tokens: Optional[int] = Field(default=None, ge=100, le=50000)
    # 交叉编码器的原始 logit，可正可负，所以不设上下限——范围随重排模型而变，
    # 卡一个数值区间只会在换模型后变成一个说不出理由的拒绝。
    agent_low_confidence_score: Optional[float] = None
    agent_summary_trigger: Optional[int] = Field(default=None, ge=4, le=200)
    agent_summary_keep: Optional[int] = Field(default=None, ge=2, le=100)
    agent_session_enabled: Optional[bool] = None
    llm_stream_usage: Optional[bool] = None


class FeedbackRequest(BaseModel):
    """
    一条用户评价。

    verdict 只有 up / down 两个取值，不做 1-5 星：星级在小样本上几乎不可用
    （所有人都点 4 星），而 badcase 库只需要「这条要不要进回归集」这一个比特。

    reason 是预置选项（见 feedback.REASONS），因为自由文本无法聚合——
    「本月 62% 的差评是答案不全」这种结论算不出来，也就指导不了下一步做什么。
    """

    trace_id: str = Field(min_length=1, max_length=64)
    verdict: str = Field(pattern="^(up|down)$")
    reason: Optional[str] = Field(default=None, max_length=32)
    comment: Optional[str] = Field(default=None, max_length=2000)


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

#: 页面里管理面区块的边界标记。HTML 注释与 JS 块注释两种写法都认，
#: 因为同一个页面里管理相关的 DOM 和脚本都要剔除。
_ADMIN_BLOCK = re.compile(
    r"(?:<!--|/\*)\s*ADMIN-BEGIN\s*(?:-->|\*/).*?(?:<!--|/\*)\s*ADMIN-END\s*(?:-->|\*/)",
    re.DOTALL,
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回前端单页应用；ADMIN_ENABLED 为假时剔除页面里的管理面区块。

    做成服务端剔除而不是前端 `display:none`，也不是两份 HTML：

      - 前端隐藏只是藏起入口，管理脚本和端点名仍然发给了每个使用者；
      - 两份 HTML 要复制那 270 行 CSS，样式改一处就得改两遍。

    标记区块覆盖侧栏、模型配置弹窗、上传遮罩三块 DOM，以及上传、文档列表、
    配置这三段脚本。剔除后剩下的脚本不引用任何被删掉的元素——
    聊天与反馈那两段本来就不碰它们。
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    if not config.ADMIN_ENABLED:
        html = _ADMIN_BLOCK.sub("", html)
    return html


@api.get("/health")
async def health():
    """
    健康检查：只报前端渲染真正需要的那几个开关。

    **刻意不含端点地址、模型名与引擎错误原文。** 这个接口对每个能打开页面的人
    都可见，而模型配置正是给使用者的部署里要藏起来的东西；`engine_error` 还会
    带出内部异常文本。要看这些走 /api/health/detail（属管理面）。
    """
    import agent

    settings = config.load_settings()
    agent_enabled = bool(settings.get("agent_enabled"))
    agent_available = agent.is_available()
    return {
        "status": "ok" if holder.error is None else "degraded",
        "agent": {
            "enabled": agent_enabled,
            "available": agent_available,
            "session_enabled": _sessions_enabled(settings) and agent_available,
        },
        # 让「反馈按钮为什么不出现」在提问之前就有答案：渲染一个点下去
        # 必然 503 的按钮，比没有按钮更糟。
        "observability": {
            "feedback_enabled": feedback_store.get_store() is not None,
        },
    }


@admin.get("/health/detail")
async def health_detail():
    """
    完整健康信息：生效端点、引擎错误、可观测性依赖状态。

    从 /api/health 里拆出来的，原因见那个接口的注释。这里的三个 error 字段
    都是同一个理由存在的：配置错误不预检，就只会在用户点下去那一刻
    表现成一次莫名的失败。
    """
    import agent

    engine = holder.get_optional()
    settings = config.load_settings()
    agent_enabled = bool(settings.get("agent_enabled"))
    agent_available = agent.is_available()
    return {
        "status": "ok" if holder.error is None else "degraded",
        "engine_error": holder.error,
        "active": engine.describe_endpoints() if engine else None,
        "agent": {
            "enabled": agent_enabled,
            "available": agent_available,
            "session_enabled": _sessions_enabled(settings) and agent_available,
            # 会话开着但库没打开时，对话仍然能用，只是重启即清空。
            # 不报出来的话，「昨天的会话怎么没了」就完全无从查起。
            "session_error": agent.session_store_error(),
            # 开了开关却没装依赖是最该被立刻看见的一种配置错误。
            "error": (
                agent.MISSING_DEPS
                if agent_enabled and not agent_available
                else None
            ),
        },
        "observability": {
            "feedback_enabled": feedback_store.get_store() is not None,
            "metrics_enabled": config.METRICS_ENABLED,
            "metrics_available": observability.METRICS_AVAILABLE,
            "metrics_error": (
                observability.MISSING_METRICS_DEPS
                if config.METRICS_ENABLED and not observability.METRICS_AVAILABLE
                else None
            ),
            "cost_currency": config.COST_CURRENCY,
            "pricing_configured": bool(config.LLM_PRICING),
        },
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sessions_enabled(settings: dict) -> bool:
    return bool(settings.get("agent_enabled")) and bool(
        settings.get("agent_session_enabled")
    )


def build_agent_runner(engine: RAGEngine, settings: dict, with_session: bool = False):
    """
    为本次请求装配 AgentRunner。

    每次请求重新装配而不是缓存：编译一张两节点的图是毫秒级开销，
    而缓存会引入「配置已热重载、agent 还绑着旧引擎」这种撕裂状态。
    取一次配置快照、用完即弃最简单。

    会话存储则相反，必须是进程级单例（见 agent.runner.default_checkpointer）：
    agent 可以重装，checkpointer 不能——它持有一条数据库连接，
    跟着请求重建就是每个请求开一次库、连接数随并发线性增长。
    单例由启动期的 lifespan 注入。
    """
    import agent

    return agent.build_runner(engine, settings, with_session=with_session)


def _persist(record: dict, answer: str, contexts: list[dict]) -> None:
    """
    把一次问答的现场落进反馈库，供后续的 👍/👎 关联与 badcase 导出。

    留痕失败不是问答失败，所以 record_interaction 内部吞掉异常。
    这里只负责「功能没开就什么也不做」。
    """
    store = feedback_store.get_store()
    if store is not None:
        store.record_interaction(record, answer=answer, contexts=contexts)


@api.post("/chat")
async def chat(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    """
    流式问答（Server-Sent Events）。

    两种模式共用这一个端点，由 agent_enabled 决定：
      - 固定管道（默认）：检索 → 生成，首 token 通常 1～2 秒
      - Agent 模式：可能先跑两三轮取材，因此额外推送工具级进度事件

    事件里始终保留 content 字段承载正文，进度事件不带 content，
    所以旧前端（只看 content / error）在两种模式下都能正常工作。

    engine.query 是异步生成器，用 async for 消费，
    网络等待期间事件循环可以调度其他请求，因此多用户并发不会相互阻塞。

    **观测在这一层收口。** 答案、token 用量、分阶段耗时这三样数据分散在
    三个地方（SSE 事件流、端点返回的 usage、引擎内部的计时），只有这里
    同时看得见全部。放到更深的层里做，就会变成每条路径各自统计一份，
    然后在某次重构后悄悄对不上。
    """
    settings = config.load_settings()
    agent_on = bool(settings.get("agent_enabled"))
    session_id = req.session_id if (agent_on and _sessions_enabled(settings)) else None

    active = engine.describe_endpoints()
    record = observability.RequestRecord(
        route="agent" if agent_on else "pipeline",
        question=req.question,
        trace_id=observability.current_trace_id() or observability.new_trace_id(),
        session_id=session_id,
        model=active.get("llm_model") or "",
        provider=active.get("llm_provider") or "",
    )

    async def generate():
        # 答案要服务端自己攒一份：反馈库里没有答案的 badcase 是无法复查的。
        # reset 事件必须同样处理（取材前的开场白不是答案的一部分），
        # 否则库里留下的答案会比用户看到的多一句「我查一下」。
        answer: list[str] = []
        contexts: list[dict] = []
        try:
            if agent_on:
                artifacts: list[dict] = []
                runner = build_agent_runner(
                    engine, settings, with_session=session_id is not None
                )
                async for event in runner.astream(
                    req.question, req.history, session_id, artifacts
                ):
                    kind = event.get("type")
                    if kind == "token":
                        record.mark_first_token()
                        answer.append(event.get("content", ""))
                    elif kind == "reset":
                        answer.clear()
                    elif kind == "done":
                        record.stages = event.get("stages") or {}
                    elif kind == "error":
                        record.error = event.get("error")
                        record.status = "error"
                    yield _sse(event)
                contexts = _dedupe_contexts(artifacts)
                _absorb_agent_usage(record, contexts, "".join(answer))
            else:
                collect: dict = {}
                async for chunk in engine.query(
                    req.question, req.history, collect=collect
                ):
                    record.mark_first_token()
                    answer.append(chunk)
                    yield _sse({"type": "token", "content": chunk})
                contexts = collect.get("contexts") or []
                _absorb_pipeline_usage(record, collect, "".join(answer))
        except Exception as exc:
            logger.exception("问答失败")
            record.status = "error"
            record.error = str(exc)
            yield _sse({"type": "error", "error": str(exc)})

        text = "".join(answer)
        record.answer_chars = len(text)
        _persist(record.finish(), text, contexts)
        # trace_id 也随流末尾发一次。响应头是主渠道，但反向代理、
        # EventSource（读不到头）和跨域漏配 expose_headers 都会让它丢失，
        # 而丢了就等于这一问无法被反馈。
        yield _sse({"type": "meta", "trace_id": record.trace_id})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": record.trace_id,
        },
    )


def _dedupe_contexts(artifacts: list[dict]) -> list[dict]:
    """
    按 (source, heading, text) 去重工具返回的资料。

    与 AgentRunner.run 用的是同一条规则：多轮取材几乎必然重复命中同一块，
    因为第二次的检索问句是基于第一次的材料改的。不去重会让落库的 contexts
    膨胀几倍，也会让基于它复算的指标失真。
    """
    out: list[dict] = []
    seen: set[tuple] = set()
    for artifact in artifacts:
        for ctx in artifact.get("contexts") or []:
            key = (ctx.get("source"), ctx.get("heading"), ctx.get("text"))
            if key in seen:
                continue
            seen.add(key)
            out.append(ctx)
    return out


def _absorb_agent_usage(record, contexts: list[dict], answer: str) -> None:
    """
    把 agent 的 token 用量搬进 record。

    stages 里的数字由 AgentRunner 逐次累加，只有每一次模型调用都如实报了
    用量才算精确（agent_tokens_exact）。缺了就退回估算：prompt 用
    「取回的资料」近似，因为那是 prompt 里体积最大的部分——
    只拿问题正文去估会把上千 token 整段漏掉，得到一个乐观到毫无意义的数。
    """
    stages = record.stages or {}
    prompt = int(stages.get("agent_prompt_tokens") or 0)
    completion = int(stages.get("agent_completion_tokens") or 0)
    if stages.get("agent_tokens_exact") and (prompt or completion):
        record.set_usage(prompt, completion)
    else:
        record.estimate_usage(
            [record.question] + [c.get("text", "") for c in contexts], answer
        )
    record.add_stage_seconds("model", float(stages.get("agent_model_seconds") or 0))
    record.add_stage_seconds("tools", float(stages.get("agent_tools_seconds") or 0))


def _absorb_pipeline_usage(record, collect: dict, answer: str) -> None:
    """把固定管道的用量与分阶段耗时搬进 record。"""
    usage = collect.get("usage") or {}
    if usage.get("prompt_tokens") or usage.get("completion_tokens"):
        record.set_usage(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
    else:
        # prompt_estimate 是引擎按真实 messages 算的，比在这一层重算准得多
        record.usage = observability.Usage(
            prompt_tokens=int(collect.get("prompt_estimate") or 0),
            completion_tokens=observability.estimate_tokens(answer),
            source="estimated",
        )
    record.add_stage_seconds("retrieve", collect.get("retrieve_seconds") or 0)
    record.add_stage_seconds("generate", collect.get("generate_seconds") or 0)


@admin.post("/agent")
async def agent_run(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    """
    非流式跑一次 agent，返回答案、取回的资料、完整事件轨迹与 stages。

    存在的意义与 /api/retrieve 相同，是把「可评估性」当架构需求。
    两类指标都靠它产出：

      - **路由指标**（工具选择正确率、调用次数、不检索比例）从 events 统计
      - **RAGAS 指标**（faithfulness / context_precision / context_recall）
        需要 contexts，也就是模型实际看到的那些资料

    contexts 的格式（text / source / heading）与 /api/retrieve 对齐，
    这样「agent 取材 vs 固定管道检索」可以用同一套指标横向比较。

    这个接口不受 agent_enabled 约束——评测要能在开关关闭时照样跑对照组。
    """
    settings = config.load_settings()
    try:
        runner = build_agent_runner(engine, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    result = await runner.run(req.question, req.history)

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return {
        "question": req.question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "stages": result["stages"],
        "events": [e for e in result["events"] if e["type"] != "token"],
    }


@admin.post("/retrieve")
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


@admin.post("/upload")
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


@admin.post("/reindex")
async def reindex(engine: RAGEngine = Depends(get_engine)):
    """
    用当前配置重建 uploads/ 下所有文档的索引。

    切换嵌入模型或调整 chunk 参数后必须执行：旧向量由旧模型/旧切分产出，
    与新的查询向量不可比。
    """
    return {"results": await engine.reindex()}


@admin.get("/documents")
async def list_documents(engine: RAGEngine = Depends(get_engine)):
    """列出知识库中所有已入库的文档。"""
    return {"documents": engine.list_documents()}


@admin.delete("/documents/{filename}")
async def delete_document(filename: str, engine: RAGEngine = Depends(get_engine)):
    """删除指定文档：先清向量库，再删磁盘文件。"""
    name = safe_md_filename(filename)
    engine.remove_document(name)
    path = config.UPLOADS_DIR / name
    if path.exists():
        path.unlink()
    return {"status": "success"}


@admin.get("/settings")
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


@admin.get("/models")
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


@admin.post("/settings")
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


# ====================================================================
# 在线反馈
# ====================================================================

def _require_store():
    """取反馈库，未启用或初始化失败时 503。"""
    store = feedback_store.get_store()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="反馈功能未启用（FEEDBACK_ENABLED=false）或反馈库初始化失败",
        )
    return store


@api.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    对某个 trace_id 提交 👍/👎。

    这是整个项目里唯一一条**用户驱动**的数据入口，也是评测闭环缺的那一环：
    在它之前，回归题集里的每一道题都靠人手写，于是「线上答错了」和
    「回归集里多一条用例」之间隔着一个人的记性。

    trace_id 未知也照常入库，不报 404：留痕是在流结束后才写的，而用户可能
    在那之前就点了评价。丢掉它等于惩罚手快的用户。
    """
    store = _require_store()
    try:
        result = store.record_feedback(
            req.trace_id, req.verdict, req.reason, req.comment
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    observability.observe_feedback(result["route"], req.verdict)
    logger.info(
        "收到反馈 trace=%s verdict=%s reason=%s", req.trace_id, req.verdict, req.reason
    )
    return {"status": "success"}


@api.get("/feedback/reasons")
async def feedback_reasons():
    """差评原因的可选项。由服务端给出，避免前后端各维护一份而对不上。"""
    return {"reasons": feedback_store.REASONS}


@admin.get("/feedback/stats")
async def feedback_stats():
    """满意度、差评原因分布，以及分路径的延迟与成本。"""
    return _require_store().stats()


@admin.get("/feedback/badcases")
async def feedback_badcases(limit: int = 200, verdict: str = "down"):
    """
    导出评价过的现场，默认取差评。

    带 contexts 一起出：一条 badcase 缺了「当时取回了什么资料」就无法区分
    这是检索问题还是生成问题，而两者的修法完全不同。
    """
    if verdict not in feedback_store.VERDICTS:
        raise HTTPException(status_code=400, detail=f"verdict 必须是 {feedback_store.VERDICTS} 之一")
    limit = max(1, min(int(limit), 2000))
    return {"badcases": _require_store().badcases(limit=limit, verdict=verdict)}


# ====================================================================
# 指标
# ====================================================================

@app.get("/metrics", dependencies=[Depends(require_token)])
async def metrics():
    """
    Prometheus 文本格式的指标。

    挂在 /metrics 而不是 /api/metrics 是抓取端的惯例，但仍然复用同一个
    X-API-Token 依赖：指标会暴露问答量、成本和错误率，是内部信息。
    Prometheus 的 scrape_configs 支持自定义请求头，所以这不构成障碍。

    prometheus_client 是可选依赖，缺它时返回 501 并给出安装命令——
    这一整层（trace_id、成本核算、结构化日志、反馈）在没有它时照常工作。
    """
    if not config.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="指标导出已关闭（METRICS_ENABLED=false）")
    try:
        payload = observability.render_metrics()
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return Response(content=payload, media_type=observability.CONTENT_TYPE_LATEST)


app.include_router(api)

# 关闭时整个 router 不注册，管理路径因此是 404 而不是 403。
if config.ADMIN_ENABLED:
    app.include_router(admin)
    logger.info("管理面已启用：文档管理、模型配置、检索调试接口可用")
else:
    logger.info("管理面已关闭（ADMIN_ENABLED=false）：仅暴露对话、健康检查与反馈接口")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
