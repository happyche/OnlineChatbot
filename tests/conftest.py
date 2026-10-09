# -*- coding: utf-8 -*-
"""
pytest 公共配置与 fixture。

这里的环境变量必须在导入 config / main 之前设置好：config 在模块导入时
就读取环境变量确定路径和默认配置。pytest 会先导入 conftest 再导入测试模块，
因此把这些赋值放在模块顶层即可保证顺序。

测试全程使用 hashing 嵌入提供方，不访问网络、不需要真实 API Key，
因此结果完全可复现，也不会消耗额度。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="onlinechatbot-tests-"))

# 关掉 Chroma 匿名遥测，避免测试期间产生外部网络请求
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# 确定性嵌入：离线、可复现
os.environ["EMBEDDING_PROVIDER"] = "hashing"
os.environ["LLM_API_KEY"] = "test-key-not-a-real-credential"
# 固定对话提供方作为基线：否则开发者 .env 里的 LLM_PROVIDER 会渗进测试，
# 导致「本机跑通、别人机器上失败」。需要 local 的用例自行 override。
os.environ["LLM_PROVIDER"] = "openai"
# 同理固定 agent 开关：开着的话 /api/chat 会走 agent 路径去调真实端点，
# 表现成一个莫名其妙的 401。需要 agent 的用例自己注入假模型。
os.environ["AGENT_ENABLED"] = "false"
os.environ["AGENT_SESSION_ENABLED"] = "false"
# 会话库指向临时目录。开关是关的、正常不会建库，但一旦有用例把它打开，
# 落盘的位置必须是临时目录而不是开发者的 data/ 下
os.environ["AGENT_SESSION_DB"] = str(_TMP_ROOT / "sessions.db")
# 指向临时目录，避免污染开发者本地的向量库、上传文件和配置
os.environ["SETTINGS_FILE"] = str(_TMP_ROOT / "settings.json")
os.environ["UPLOADS_DIR"] = str(_TMP_ROOT / "uploads")
os.environ["CHROMA_DIR"] = str(_TMP_ROOT / "chroma")
os.environ["APP_API_TOKEN"] = ""
# 测试要覆盖文档上传、模型配置这些管理接口，而它们默认不注册（给使用者的
# 部署里是 404）。这个开关在 main 导入时就决定了 router 装不装，
# 所以必须在这里设、不能用 monkeypatch——「关闭时是 404」由专门的用例
# 另起一个 app 来验证（test_api_security.TestAdminSurface）。
os.environ["ADMIN_ENABLED"] = "true"
# 哈希嵌入的相似度分布与真实模型不同，默认不过滤；阈值行为由专门的用例覆盖
os.environ["MIN_SIMILARITY"] = "0.0"
# 反馈库指向临时目录；每个用例还会再拿一份独立的（见 isolated_feedback_db）
os.environ["FEEDBACK_DB"] = str(_TMP_ROOT / "feedback.db")
# 结构化请求日志不落盘：测试里它只会在临时目录外留一堆 jsonl
os.environ["REQUEST_LOG_FILE"] = ""
# 假的流式实现不返回 usage，索要 stream_options 只会让替身多背一个参数
os.environ["LLM_STREAM_USAGE"] = "false"

import pytest  # noqa: E402  必须在设置环境变量之后导入

import config  # noqa: E402
from embeddings import Embedder, HashingEmbedder  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MANUAL_PATH = FIXTURE_DIR / "运维手册.md"


def pytest_sessionfinish(session, exitstatus):
    """会话结束后清理临时目录。"""
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


class SpyEmbedder(Embedder):
    """
    包装真实 embedder 并记录调用次数。

    用于验证「引擎确实使用了注入/配置的 embedder」——原实现中 collection
    绑定了 Chroma 内置嵌入函数，配置的模型从未被调用，这个 spy 就是那条链路的回归防线。
    """

    def __init__(self, inner: Embedder | None = None):
        self._inner = inner or HashingEmbedder()
        self.signature = self._inner.signature
        self.doc_calls = 0
        self.query_calls = 0
        self.embedded_texts: list[str] = []

    async def embed_documents(self, texts):
        self.doc_calls += 1
        self.embedded_texts.extend(texts)
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text):
        self.query_calls += 1
        return await self._inner.embed_query(text)


@pytest.fixture(autouse=True)
def isolated_settings_file(tmp_path, monkeypatch):
    """
    每个用例使用独立的 settings.json。

    接口测试会通过 POST /api/settings 写盘，若共用一份文件，
    写入的配置会泄漏到后续用例，形成隐蔽的测试顺序依赖。
    """
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")


@pytest.fixture(autouse=True)
def isolated_feedback_db(tmp_path, monkeypatch):
    """
    每个用例使用独立的反馈库。

    存储是进程级单例（和 checkpointer 同理），不重置的话第一个用例建的连接会
    一直指向它自己的临时目录，后面所有用例的断言都在读别人的数据——
    而这种串扰只会表现成「单跑通过、全跑失败」。
    """
    import feedback

    monkeypatch.setattr(config, "FEEDBACK_DB", tmp_path / "feedback.db")
    feedback.reset_store()
    yield
    feedback.reset_store()


@pytest.fixture
def manual_text() -> str:
    """测试用的中文 Markdown 运维手册内容。"""
    return MANUAL_PATH.read_text(encoding="utf-8")


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    """
    构造使用独立向量库目录的 RAGEngine，保证用例之间互不干扰。

    用法：engine = make_engine()  或  make_engine(embedder=spy, chunk_size=200)
    """
    counter = {"n": 0}

    def _make(
        embedder: Embedder | None = None,
        use_config_embedder: bool = False,
        reranker=None,
        **overrides,
    ) -> RAGEngine:
        """
        use_config_embedder=True 时不注入 embedder，让引擎按配置自行构造，
        用于验证 embedding_provider / embedding_base_url 等配置的解析逻辑。
        reranker 可注入，避免测试依赖 1GB 的交叉编码器权重。
        """
        counter["n"] += 1
        chroma_dir = tmp_path / f"chroma-{counter['n']}"
        monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)
        settings = config.load_settings()
        settings.update(overrides)
        if use_config_embedder:
            return RAGEngine(settings=settings, embedder=None, reranker=reranker)
        return RAGEngine(
            settings=settings,
            embedder=embedder or HashingEmbedder(),
            reranker=reranker,
        )

    return _make


@pytest.fixture
def shared_dir_engine(tmp_path, monkeypatch):
    """
    构造共用同一个向量库目录的多个引擎。

    用于验证「更换嵌入模型后向量库被重建」这一行为，
    需要两个引擎先后打开同一份持久化数据。
    """
    chroma_dir = tmp_path / "chroma-shared"
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)

    def _make(embedder: Embedder, **overrides) -> RAGEngine:
        settings = config.load_settings()
        settings.update(overrides)
        return RAGEngine(settings=settings, embedder=embedder)

    return _make


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    FastAPI 测试客户端，使用隔离的向量库与上传目录，并把 LLM 调用替换为假实现。

    假实现会记录送入模型的完整 messages，便于断言检索到的内容
    是否真的进入了 prompt。
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma-api")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads-api")
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    import main

    captured: list[list[dict]] = []

    async def fake_stream(self, state, messages, **kwargs):
        captured.append(messages)
        yield "这是"
        yield "模拟回答。"

    monkeypatch.setattr(RAGEngine, "_stream_chat", fake_stream)
    main.holder.rebuild()
    assert main.holder.error is None, f"引擎初始化失败: {main.holder.error}"

    with TestClient(main.app) as test_client:
        test_client.captured_messages = captured
        yield test_client
