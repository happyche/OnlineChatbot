# -*- coding: utf-8 -*-
"""
配置模块
========
配置来源分两层，后者覆盖前者：

  1. .env / 环境变量  —— 提供默认值，适合部署时注入
  2. settings.json    —— 运行时通过 UI 修改后落盘的覆盖值

注意 settings.json 会保存 API Key 明文，已在 .gitignore 中排除，切勿提交。

路径类配置支持用环境变量覆盖，目的是让自动化测试能指向临时目录，
避免污染开发者本地的向量库和上传文件。
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _flag(env_key: str, default: bool) -> bool:
    """读取布尔类配置，接受 1/true/yes/on 等常见写法。"""
    raw = os.getenv(env_key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _path_from_env(env_key: str, default_name: str) -> Path:
    """读取路径类配置：环境变量优先，否则取项目根目录下的默认名。"""
    raw = os.getenv(env_key, "").strip()
    return Path(raw).expanduser().resolve() if raw else ROOT / default_name


SETTINGS_FILE = _path_from_env("SETTINGS_FILE", "settings.json")
UPLOADS_DIR = _path_from_env("UPLOADS_DIR", "uploads")
CHROMA_DIR = _path_from_env("CHROMA_DIR", "chroma_db")

# 本地模型权重放用户级缓存目录，多个项目可共享，也不会被系统清理临时目录时删掉
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "fastembed"

# Windows 上非管理员账户无法建符号链接，huggingface_hub 会每次刷一条警告，
# 缓存本身仍可正常工作，这里直接静音
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULTS = {
    # 对话模型提供方：
    #   openai = 云端 API（需要 API Key，按量计费）
    #   local  = 自建的 OpenAI 兼容服务（Ollama / vLLM / LM Studio / llama.cpp 等），
    #            可以在本机，也可以在内网服务器上
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),

    # --- provider=openai 时生效 ---
    "llm_api_key": os.getenv("LLM_API_KEY", ""),
    "llm_base_url": os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "llm_model": os.getenv("LLM_MODEL", "qwen-plus"),

    # --- provider=local 时生效 ---
    # 必须是 OpenAI 兼容的地址，通常以 /v1 结尾。
    # Ollama: http://<host>:11434/v1 ｜ vLLM: http://<host>:8000/v1
    # LM Studio: http://<host>:1234/v1 ｜ llama.cpp server: http://<host>:8080/v1
    "local_llm_base_url": os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
    "local_llm_model": os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b"),
    # 多数自建服务不校验 Key（Ollama 直接忽略）；vLLM 若启用了 --api-key 则需填写
    "local_llm_api_key": os.getenv("LOCAL_LLM_API_KEY", ""),
    # 嵌入提供方：local=本地 ONNX（默认，离线免费、文本不出本机）
    #           | openai=远程接口 | hashing=确定性哈希（仅测试）
    "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local"),
    # 本地模型名（provider=local 时生效）。默认 bge-small-zh-v1.5：约 90MB、512 维、
    # 中文检索质量足够，输出已 L2 归一化。
    "local_embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
    # 本地模型权重的缓存目录。留空则用用户级缓存，避免落到系统临时目录被清理掉。
    "embedding_cache_dir": os.getenv("EMBEDDING_CACHE_DIR", str(DEFAULT_MODEL_CACHE)),
    # 部分嵌入模型要求查询与文档使用不同的指令前缀（如 e5 系列的 "query: " / "passage: "）。
    # bge-*-v1.5 不需要，故默认留空。
    "embedding_query_prefix": os.getenv("EMBEDDING_QUERY_PREFIX", ""),
    "embedding_doc_prefix": os.getenv("EMBEDDING_DOC_PREFIX", ""),
    # 远程模型名（provider=openai 时生效）
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    # 嵌入接口的独立地址。留空则复用当前生效的对话端点。
    # 需要它是因为对话模型和嵌入模型不一定在同一端点上：例如 DashScope 的
    # coding 专用端点只提供 chat，不提供 /embeddings（请求会返回 404）。
    "embedding_base_url": os.getenv("EMBEDDING_BASE_URL", ""),
    # 嵌入接口的独立 Key。留空则复用对话端点的 Key。
    # 当对话走自建服务（无需 Key）而嵌入走云端（需要 Key）时必须分开配置。
    "embedding_api_key": os.getenv("EMBEDDING_API_KEY", ""),
    # 远程嵌入接口对单次请求的文本条数有上限，分批发送
    "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
    "top_k": int(os.getenv("TOP_K", "5")),
    # 余弦相似度低于该阈值的向量检索结果视为不相关，直接丢弃。
    # 只作用于向量这一路：BM25 命中的文档即使余弦分低也保留，
    # 这正是混合检索的意义（捕捉语义相近度不高但字面精确匹配的内容）。
    "min_similarity": float(os.getenv("MIN_SIMILARITY", "0.2")),

    # === 检索增强开关（可独立开闭，便于用 RAGAS 做消融对比）===
    # 混合检索：在向量检索之外并行跑 BM25，再用 RRF 融合两路排名
    "hybrid_search_enabled": _flag("HYBRID_SEARCH_ENABLED", False),
    # BM25 分词器：jieba=词粒度｜bigram=字符二元组（不受词典覆盖率影响）
    "bm25_tokenizer": os.getenv("BM25_TOKENIZER", "jieba"),
    # RRF 平滑常数，越大则头部名次优势越小；60 为原论文取值
    "rrf_k": int(os.getenv("RRF_K", "60")),

    # 重排：用交叉编码器对候选做精排
    "rerank_enabled": _flag("RERANK_ENABLED", False),
    # local=本地 ONNX 交叉编码器｜lexical=字面重叠度（测试用，无需权重）
    "rerank_provider": os.getenv("RERANK_PROVIDER", "local"),
    # bge-reranker-base 约 1GB、支持中文；可用 download_model.py --reranker 预下载
    "rerank_model": os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
    # 重排分数低于该值的候选被丢弃。交叉编码器输出未归一化的 logit，
    # 不同模型的取值范围差异很大，留空表示不过滤。
    "min_rerank_score": os.getenv("MIN_RERANK_SCORE", ""),

    # 候选池大小：开启混合检索或重排时，每一路先取这么多候选，
    # 融合/精排后再截到 top_k。太小会让重排无从改进排序。
    "candidate_pool_size": int(os.getenv("CANDIDATE_POOL_SIZE", "20")),
    # 送入 LLM 的历史消息条数上限（2 条为一轮问答）
    "history_limit": int(os.getenv("HISTORY_LIMIT", "10")),

    # === 多轮 Query Rewrite（检索前改写，生成仍用原问题）===
    # 默认开启；首问或无指代时不调 LLM（见 query_rewrite_mode）
    "query_rewrite_enabled": _flag("QUERY_REWRITE_ENABLED", True),
    # conditional=启发式触发 | always=有 history 就改写 | off=关闭
    "query_rewrite_mode": os.getenv("QUERY_REWRITE_MODE", "conditional"),
    # 改写 prompt 使用的历史条数上限（默认 6 条 ≈ 3 轮，比生成侧更短）
    "rewrite_history_limit": int(os.getenv("REWRITE_HISTORY_LIMIT", "6")),
    # 条件触发：问句不超过该字数时视为可能省略（与指代词启发式取或）
    "rewrite_trigger_max_len": int(os.getenv("REWRITE_TRIGGER_MAX_LEN", "12")),
    # 逗号分隔的正则；留空用内置指代词表
    "rewrite_trigger_patterns": os.getenv("REWRITE_TRIGGER_PATTERNS", ""),
    # 留空则复用当前对话模型；可指定更小模型以降延迟
    "rewrite_model": os.getenv("REWRITE_MODEL", ""),
    "rewrite_temperature": float(os.getenv("REWRITE_TEMPERATURE", "0")),

    "request_timeout": float(os.getenv("REQUEST_TIMEOUT", "60")),

    # 访问本机/内网地址时绕过 HTTP_PROXY。企业环境几乎都配了代理，
    # 而客户端默认读取该环境变量，导致内网自建服务的请求被发去代理并失败。
    # 极少数需要经代理访问内网的场景可关掉此项，改用 NO_PROXY 手工配置。
    "bypass_proxy_for_internal": _flag("BYPASS_PROXY_FOR_INTERNAL", True),
}

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# 前端与后端同源（页面由本服务的 GET / 提供），因此默认只放行本机来源。
# 需要跨域调用时通过 CORS_ORIGINS 显式配置，不要图省事写成 "*"：
# 本服务持有 API Key 且可删除文件，全开放来源会让任意网页都能驱动它。
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()
]

# 可选的访问令牌：设置后所有 /api/* 请求必须带 X-API-Token 请求头。
# 留空表示不鉴权（仅适合本机开发）。
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "").strip()

#: 不允许通过 /api/settings 写入 settings.json 的字段（避免前端误改路径类配置）
_MUTABLE_KEYS = set(DEFAULTS.keys())


def load_settings() -> dict:
    """加载配置：DEFAULTS 打底，settings.json 覆盖。"""
    settings = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # 配置文件损坏时退回默认值，但必须让运维看到，不能静默
            raise RuntimeError(f"读取 {SETTINGS_FILE} 失败: {exc}") from exc
        settings.update({k: v for k, v in saved.items() if k in _MUTABLE_KEYS})
    return settings


def is_internal_host(url: str) -> bool:
    """
    判断地址是否指向本机或内网私有地址段。

    用途是自动绕过企业 HTTP 代理。企业环境普遍设置了 HTTP_PROXY，
    而 httpx / openai 客户端默认读取该环境变量，于是访问内网自建的
    推理服务时请求会被发到代理上，返回一个 HTML 错误页 ——
    表现为「InternalServerError + 一段 HTML」，很难联想到是代理问题。

    主机名（而非 IP）无法判断，按非内网处理，交由用户显式配置 NO_PROXY。
    """
    from ipaddress import ip_address
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    if not host:
        return False
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        addr = ip_address(host)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback


@dataclass(frozen=True)
class Endpoint:
    """一个 OpenAI 兼容端点的完整寻址信息。"""

    base_url: str
    model: str
    api_key: str
    #: 供界面与日志显示，如 "local" / "openai"
    provider: str


def resolve_llm(settings: dict) -> Endpoint:
    """
    解析当前生效的对话端点。

    把「provider 决定用哪组配置」这个判断收敛到一处，避免各处重复推断
    导致 UI 显示的和实际调用的不一致。

    provider=local 指任何自建的 OpenAI 兼容服务（Ollama / vLLM / LM Studio /
    llama.cpp），可以在本机也可以在内网服务器。多数这类服务不校验 Key，
    但 OpenAI 客户端要求 api_key 非空，因此填一个占位串。
    """
    provider = str(settings.get("llm_provider", "openai")).lower()

    if provider == "local":
        base_url = (settings.get("local_llm_base_url") or "").strip()
        if not base_url:
            raise ValueError("使用自建对话服务需要配置 local_llm_base_url（形如 http://主机:端口/v1）")
        model = (settings.get("local_llm_model") or "").strip()
        if not model:
            raise ValueError("使用自建对话服务需要配置 local_llm_model")
        return Endpoint(
            base_url=base_url,
            model=model,
            api_key=(settings.get("local_llm_api_key") or "").strip() or "local",
            provider="local",
        )

    if provider != "openai":
        raise ValueError(f"未知的 llm_provider: {provider!r}（可选 openai/local）")

    api_key = (settings.get("llm_api_key") or "").strip()
    if not api_key:
        raise ValueError(
            "使用云端对话模型需要配置 llm_api_key；"
            "若要用自建服务，请将 llm_provider 设为 local。"
        )
    return Endpoint(
        base_url=settings["llm_base_url"],
        model=settings["llm_model"],
        api_key=api_key,
        provider="openai",
    )


def resolve_embedding_endpoint(settings: dict, chat: Endpoint) -> Endpoint:
    """
    解析远程嵌入端点（仅 embedding_provider=openai 时使用）。

    地址与 Key 均可独立配置，留空则回落到对话端点——常见情况下两者同源，
    但「对话走内网自建服务、嵌入走云端」时必须分开，否则会拿占位 Key 去请求云端。
    """
    base_url = (settings.get("embedding_base_url") or "").strip() or chat.base_url
    api_key = (settings.get("embedding_api_key") or "").strip() or chat.api_key
    return Endpoint(
        base_url=base_url,
        model=settings["embedding_model"],
        api_key=api_key,
        provider="openai",
    )


def save_settings(settings: dict):
    """把运行时配置写入 settings.json（仅持久化已知字段）。"""
    payload = {k: v for k, v in settings.items() if k in _MUTABLE_KEYS}
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
