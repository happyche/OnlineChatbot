# -*- coding: utf-8 -*-
"""
RAG（检索增强生成）引擎
=======================
职责：
  1. 把 Markdown 文档按标题层级切分成带重叠的文本块
  2. 用配置指定的嵌入模型向量化，存入 ChromaDB，按余弦相似度检索
  3. 组装上下文并通过 OpenAI 兼容接口流式生成回答

几个关键设计：

**向量空间指纹**
  collection 元数据里记录产出这批向量的 embedder signature。换了嵌入模型后
  旧向量与新查询向量既不在同一语义空间、维度也不同，检索结果毫无意义，
  因此启动时检测到 signature 不匹配就重建 collection，并提示重新索引。

**状态快照**
  settings / client / embedder / collection 打包成不可变的 _EngineState。
  热重载时构造新对象再整体替换引用，读取方一次取到一致的快照，
  不会读到「新的 client 配旧的 collection」这种撕裂状态，也无需加锁。

**全异步**
  嵌入和生成都是网络 IO，同步调用会阻塞事件循环，导致并发请求被串行化。
  ChromaDB 本身是同步的本地库，其调用丢到线程池执行。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import chromadb
from openai import AsyncOpenAI

import config
from embeddings import Embedder, build_embedder
from query_rewrite import resolve_search_query
from retrieval import BM25Index, Reranker, build_reranker, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"

#: collection 元数据中记录向量空间指纹的键名
_SIGNATURE_KEY = "embedder_signature"

#: Markdown 标题行
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

#: 切分超长文本时优先使用的断点字符（含中英文标点，中文没有空格，不能只按空白切）
_BREAK_CHARS = "。！？；\n！?!;:：，,、 \t"

_PARAGRAPH_SEP = "\n\n"

SYSTEM_PROMPT_BASE = "你是一个专业的知识助手，负责根据公司内部文档回答用户的问题。"


@dataclass(frozen=True)
class _EngineState:
    """引擎的一致性快照，热重载时整体替换。"""

    settings: dict
    #: 对话端点解析失败时为 None（例如只配了嵌入、还没填对话服务地址）
    client: Optional[AsyncOpenAI]
    chat: Optional["config.Endpoint"]
    chat_error: Optional[str]
    embedder: Embedder
    #: 未启用重排时为 None
    reranker: Optional[Reranker]
    collection: object
    collection_reset: bool


class _ExplicitVectorCollection:
    """
    ChromaDB collection 的薄代理，强制所有写入与检索都显式提供向量。

    这一层不是洁癖，是防止 Bug 复发的硬约束。实测 chromadb 1.5.9 的行为：
    即使创建 collection 时传了 embedding_function=None，调用
    add(documents=...) 而不给 embeddings、或 query(query_texts=...) 时，
    Chroma 依然会静默回退到内置的 all-MiniLM-L6-v2（384 维、纯英文）自动向量化，
    既不报错也不打日志。

    这正是本项目原先「配置的嵌入模型从未生效、中文检索接近随机」的根因。
    把隐式路径直接封死，将来任何漏传向量的调用都会立即失败而不是悄悄降级。
    """

    def __init__(self, inner):
        self._inner = inner

    def add(self, *, ids, embeddings=None, documents=None, metadatas=None, **kwargs):
        if embeddings is None:
            raise ValueError(
                "写入向量库必须显式提供 embeddings；"
                "省略会导致 Chroma 静默使用内置英文模型向量化。"
            )
        return self._inner.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas, **kwargs
        )

    def query(self, *, query_embeddings=None, query_texts=None, **kwargs):
        if query_texts is not None:
            raise ValueError(
                "检索必须使用 query_embeddings；query_texts 会让 Chroma "
                "用内置英文模型编码查询，与库中向量不在同一语义空间。"
            )
        if query_embeddings is None:
            raise ValueError("检索必须显式提供 query_embeddings。")
        return self._inner.query(query_embeddings=query_embeddings, **kwargs)

    def __getattr__(self, name):
        # get / delete / count / metadata 等只读或无嵌入语义的成员直接透传
        return getattr(self._inner, name)


@dataclass
class _Candidate:
    """检索管道内部的可变候选，逐阶段累积各路得分。"""

    doc_id: str
    text: str
    source: str
    heading: str
    vector_similarity: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass(frozen=True)
class Retrieved:
    """
    一条检索结果，带各阶段得分明细。

    保留明细而不是只给一个最终分数，是为了做消融分析时能看清
    某条结果是靠语义召回的、靠字面命中的，还是靠重排捞上来的。

    similarity : 向量余弦相似度。仅被 BM25 召回的文档为 None。
    score      : 最终排序依据，其含义由 score_type 说明
                 （cosine=余弦｜rrf=融合排名分｜rerank=交叉编码器分）。
    """

    text: str
    source: str
    heading: str
    similarity: Optional[float]
    bm25_score: Optional[float]
    rrf_score: Optional[float]
    rerank_score: Optional[float]
    score: float
    score_type: str
    doc_id: str = ""

    @property
    def citation(self) -> str:
        """人类可读的出处，精确到章节。"""
        return f"{self.source} › {self.heading}" if self.heading else self.source


class RAGEngine:
    """RAG 检索增强生成引擎。"""

    def __init__(
        self,
        settings: Optional[dict] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ):
        """
        参数:
            settings : 覆盖配置，默认从 config 读取
            embedder : 注入的嵌入实现，默认按配置构造。测试时注入确定性 embedder，
                       即可完全离线运行，不依赖网络与 API Key。
            reranker : 注入的重排实现，默认按配置构造。注入时无需下载模型权重。
        """
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._injected_embedder = embedder
        self._injected_reranker = reranker
        # BM25 索引按需构建，文档或分词器变化时置空重建
        self._bm25: Optional[BM25Index] = None
        self._state = self._build_state(settings or config.load_settings())

    # ------------------------------------------------------------------
    # 状态构建与热重载
    # ------------------------------------------------------------------

    @staticmethod
    def _make_client(
        endpoint: "config.Endpoint", settings: dict, timeout: float
    ) -> AsyncOpenAI:
        """
        构造指向某个端点的 AsyncOpenAI 客户端。

        访问内网地址时用 trust_env=False 的 httpx 客户端绕过 HTTP_PROXY：
        企业环境普遍设置了代理，而客户端默认读取该环境变量，
        于是访问内网推理服务的请求被发去代理，返回一段 HTML 错误页 ——
        报错信息里只有 InternalServerError 和 HTML 片段，极难定位。
        """
        http_client = None
        if settings.get("bypass_proxy_for_internal") and config.is_internal_host(
            endpoint.base_url
        ):
            import httpx

            logger.info("端点 %s 属内网地址，已绕过 HTTP 代理", endpoint.base_url)
            http_client = httpx.AsyncClient(trust_env=False, timeout=timeout)

        return AsyncOpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            timeout=timeout,
            max_retries=2,
            http_client=http_client,
        )

    def _build_state(self, settings: dict) -> _EngineState:
        """由一份配置构造完整的引擎状态。"""
        timeout = settings.get("request_timeout", 60)

        # 注入了 embedder 的场景（测试）不需要对话端点也可用，
        # 因此对话端点解析失败时先记下来，等真正要生成回答时再报错。
        chat: config.Endpoint | None = None
        chat_error: str | None = None
        try:
            chat = config.resolve_llm(settings)
        except ValueError as exc:
            if self._injected_embedder is None and settings.get("embedding_provider") == "openai":
                # 远程嵌入要复用对话端点的寻址信息，此时必须立刻失败
                raise
            chat_error = str(exc)

        client = self._make_client(chat, settings, timeout) if chat is not None else None

        embedder = self._injected_embedder
        if embedder is None:
            embed_client = None
            if settings.get("embedding_provider") == "openai":
                embed = config.resolve_embedding_endpoint(settings, chat)
                # 嵌入与对话同源时复用同一个客户端，避免重复建连接池
                if chat is not None and embed.base_url == chat.base_url and embed.api_key == chat.api_key:
                    embed_client = client
                else:
                    embed_client = self._make_client(embed, settings, timeout)
            embedder = build_embedder(settings, embed_client)

        collection, was_reset = self._ensure_collection(embedder.signature)
        return _EngineState(
            settings=settings,
            client=client,
            chat=chat,
            chat_error=chat_error,
            embedder=embedder,
            reranker=self._injected_reranker or build_reranker(settings),
            collection=collection,
            collection_reset=was_reset,
        )

    def _ensure_collection(self, signature: str):
        """
        取得与当前嵌入模型匹配的 collection。

        若已存在的 collection 是别的嵌入模型建的（signature 不同或缺失），
        其向量无法与新查询向量比较，只能删除重建。

        返回 (collection, 是否发生了重建)。
        """
        existing = None
        try:
            # 显式传 None 可以去掉 Python 侧的默认嵌入函数（该参数默认值就是
            # DefaultEmbeddingFunction()）。但这还不够：Chroma 底层仍会在缺少
            # 显式向量时回退到内置英文模型，因此外面还要再套 _ExplicitVectorCollection。
            existing = self._chroma.get_collection(COLLECTION_NAME, embedding_function=None)
        except Exception:
            existing = None

        if existing is not None:
            current = (existing.metadata or {}).get(_SIGNATURE_KEY)
            if current == signature:
                return _ExplicitVectorCollection(existing), False
            logger.warning(
                "向量库由 %s 构建，当前嵌入模型为 %s，向量空间不兼容，已重建 collection。"
                "请调用 /api/reindex 重新索引 uploads/ 下的文档。",
                current or "未知模型(旧版本遗留)",
                signature,
            )
            self._chroma.delete_collection(COLLECTION_NAME)

        collection = self._chroma.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", _SIGNATURE_KEY: signature},
            embedding_function=None,
        )
        return _ExplicitVectorCollection(collection), existing is not None

    def reload_settings(self) -> dict:
        """
        热重载配置：构造新快照后原子替换。

        返回 {"collection_reset": bool}，供调用方提示用户是否需要重新索引。
        """
        new_state = self._build_state(config.load_settings())
        self._state = new_state
        # 分词器或向量库可能已变，BM25 索引一律重建
        self._bm25 = None
        return {"collection_reset": new_state.collection_reset}

    @property
    def collection_reset_on_start(self) -> bool:
        """启动时是否因嵌入模型变化重建过向量库。"""
        return self._state.collection_reset

    @property
    def embedder(self) -> Embedder:
        """
        [RAGAS] 当前生效的嵌入模型。

        为评测暴露：RAGAS 的 answer_relevancy 也要算向量相似度，
        必须和检索用的是同一个模型，否则比的是两个不同语义空间里的距离。
        直接读 _state 能拿到同一份快照，避免评测中途热重载导致前后不一致。
        """
        return self._state.embedder

    def describe_endpoints(self) -> dict:
        """
        报告当前实际生效的端点与模型。

        配置项有云端/自建两套，光看配置很难判断哪套在生效，
        这里直出结论供界面展示与排障，不含任何密钥。
        """
        state = self._state
        return {
            "llm_provider": state.chat.provider if state.chat else None,
            "llm_base_url": state.chat.base_url if state.chat else None,
            "llm_model": state.chat.model if state.chat else None,
            "llm_error": state.chat_error,
            "embedding_provider": state.settings.get("embedding_provider"),
            "embedding_signature": state.embedder.signature,
            "hybrid_search_enabled": bool(state.settings.get("hybrid_search_enabled")),
            "bm25_tokenizer": state.settings.get("bm25_tokenizer"),
            "rerank_enabled": state.reranker is not None,
            "reranker": state.reranker.signature if state.reranker else None,
        }

    async def list_chat_models(self) -> list[str]:
        """
        列出对话端点上可用的模型（OpenAI 兼容的 GET /models）。

        对自建服务尤其有用：既能确认网络与地址配对，也能直接看到服务器上
        实际加载了哪些模型，省去猜模型名。
        """
        state = self._state
        if state.client is None:
            raise RuntimeError(f"对话模型不可用: {state.chat_error}")
        resp = await state.client.models.list()
        return sorted(item.id for item in resp.data)

    # ------------------------------------------------------------------
    # Markdown 切分
    # ------------------------------------------------------------------

    def _split_markdown(self, content: str, filename: str) -> list[dict]:
        """
        把 Markdown 切分为文本块。

        策略：
          1. 按标题切成 section，同时维护标题栈，得到每段的层级路径
             （如「配置说明 › 环境变量 › API Key」），供引用溯源和检索增强使用
          2. section 内按空行分段，贪心装箱到 chunk_size
          3. 相邻块之间保留 chunk_overlap 个字符的重叠，避免跨块语义被切断
          4. 单个段落仍超长时按标点断点切分（中文无空格，不能按空白切）

        返回每项含 id / content / embed_text / metadata 的字典列表。
        """
        settings = self._state.settings
        chunk_size = max(50, int(settings["chunk_size"]))
        # 重叠不能超过块长的一半，否则相邻块内容高度冗余，检索结果会被同一段内容占满
        chunk_overlap = max(0, min(int(settings["chunk_overlap"]), chunk_size // 2))

        results: list[dict] = []
        for heading, body in self._iter_sections(content):
            units = self._split_oversized_units(body, chunk_size)
            for piece in self._pack_units(units, chunk_size, chunk_overlap):
                idx = len(results)
                doc_id = hashlib.md5(
                    f"{filename}:{idx}:{heading}:{piece[:50]}".encode("utf-8")
                ).hexdigest()
                results.append(
                    {
                        "id": doc_id,
                        "content": piece,
                        # 标题一起参与向量化：正文常用代词指代标题主体，
                        # 带上层级路径能明显提升这类块的召回率
                        "embed_text": f"{heading}\n{piece}" if heading else piece,
                        "metadata": {
                            "source": filename,
                            "chunk_index": idx,
                            "heading": heading,
                        },
                    }
                )
        return results

    @staticmethod
    def _iter_sections(content: str) -> list[tuple[str, str]]:
        """
        按 Markdown 标题切分，返回 [(标题层级路径, 正文), ...]。

        标题栈保证子章节能继承父章节的路径；正文里不再保留标题行，
        标题信息统一由 metadata 承载，渲染上下文时再补回去。
        """
        sections = re.split(r"\n(?=#{1,6}\s)", content)
        stack: list[tuple[int, str]] = []
        out: list[tuple[str, str]] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            match = _HEADING_RE.match(lines[0])
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                # 弹出同级及更深的标题，再压入当前标题
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                heading_path = " › ".join(t for _, t in stack)
                body = "\n".join(lines[1:]).strip()
            else:
                # 文档开头在第一个标题之前的内容
                heading_path = ""
                body = section

            if body:
                out.append((heading_path, body))
        return out

    @classmethod
    def _split_oversized_units(cls, body: str, chunk_size: int) -> list[str]:
        """按空行分段，并把超过 chunk_size 的段落进一步切开，保证每个单元都能装进一个块。"""
        units: list[str] = []
        for para in body.split(_PARAGRAPH_SEP):
            para = para.strip()
            if not para:
                continue
            if len(para) <= chunk_size:
                units.append(para)
            else:
                units.extend(cls._split_long_text(para, chunk_size))
        return units

    @staticmethod
    def _split_long_text(text: str, max_len: int) -> list[str]:
        """
        把超长文本切成不超过 max_len 的片段。

        优先在标点或空白处断开以保住句子完整性；若窗口内的断点过于靠前
        （会产出大量碎片），则按字符硬切。中文没有词间空格，
        原实现按 text.split() 切词对中文完全无效——整段会被当成一个「词」原样返回。
        """
        out: list[str] = []
        remaining = text.strip()

        while len(remaining) > max_len:
            window = remaining[:max_len]
            cut = max((window.rfind(ch) for ch in _BREAK_CHARS), default=-1)
            # 断点靠前意味着这一块会很短，宁可硬切以保持块大小均匀
            if cut < max_len // 2:
                cut = max_len - 1
            piece = remaining[: cut + 1].strip()
            if piece:
                out.append(piece)
            remaining = remaining[cut + 1 :].strip()

        if remaining:
            out.append(remaining)
        return out

    @staticmethod
    def _pack_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
        """
        把文本单元贪心装箱成块，并在相邻块之间加入重叠。

        重叠取上一块的尾部若干字符；仅在加上重叠后仍不超过 chunk_size 时才添加，
        以保证「任何块都不超过 chunk_size」这一不变量。
        """
        chunks: list[str] = []
        current = ""

        for unit in units:
            if not current:
                current = unit
            elif len(current) + len(_PARAGRAPH_SEP) + len(unit) <= chunk_size:
                current = f"{current}{_PARAGRAPH_SEP}{unit}"
            else:
                chunks.append(current)
                tail = current[-overlap:].strip() if overlap > 0 else ""
                if tail and len(tail) + len(_PARAGRAPH_SEP) + len(unit) <= chunk_size:
                    current = f"{tail}{_PARAGRAPH_SEP}{unit}"
                else:
                    current = unit

        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------
    # 文档增删查
    # ------------------------------------------------------------------

    async def add_document(self, content: str, filename: str) -> int:
        """
        添加（覆盖）一篇文档：先删同名旧数据，再切分、向量化、入库。

        返回写入的文本块数量。
        """
        state = self._state
        self.remove_document(filename)

        chunks = self._split_markdown(content, filename)
        if not chunks:
            return 0

        # 显式传入向量，不依赖 collection 的内置嵌入函数
        vectors = await state.embedder.embed_documents([c["embed_text"] for c in chunks])
        await asyncio.to_thread(
            state.collection.add,
            ids=[c["id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=vectors,
        )
        # 语料变了，BM25 索引失效
        self._bm25 = None
        logger.info("已索引 %s：%d 个文本块", filename, len(chunks))
        return len(chunks)

    def remove_document(self, filename: str):
        """
        删除指定文档的所有文本块。

        这里不能静默失败：add_document 依赖它实现「覆盖上传」，
        删除失败而新块照样写入会导致同一文档的新旧版本同时留在库里，
        检索时混杂返回。
        """
        collection = self._state.collection
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            self._bm25 = None
            logger.info("已删除 %s 的 %d 个文本块", filename, len(existing["ids"]))

    def list_documents(self) -> list[dict]:
        """列出已入库的文档及其文本块数量。"""
        results = self._state.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in results["metadatas"] or []:
            source = meta.get("source")
            if source:
                counts[source] = counts.get(source, 0) + 1
        return [{"filename": k, "chunks": v} for k, v in sorted(counts.items())]

    async def reindex(self) -> list[dict]:
        """
        用当前配置重建 uploads/ 下所有文档的索引。

        换嵌入模型或调整 chunk 参数后必须执行，否则库里仍是旧参数产出的向量。
        """
        report: list[dict] = []
        for path in sorted(config.UPLOADS_DIR.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                count = await self.add_document(text, path.name)
                report.append({"filename": path.name, "chunks": count, "status": "success"})
            except Exception as exc:
                logger.exception("重建索引失败: %s", path.name)
                report.append({"filename": path.name, "error": str(exc)})
        return report

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    # ---- 各检索阶段 ----

    async def _vector_search(
        self, state: _EngineState, question: str, limit: int
    ) -> list[_Candidate]:
        """
        向量检索。

        Chroma 在 cosine 空间返回的 distance = 1 - 余弦相似度，
        因此用 1 - distance 还原相似度，并按 min_similarity 过滤，
        避免把明显无关的内容塞进上下文污染生成。
        """
        query_vector = await state.embedder.embed_query(question)
        raw = await asyncio.to_thread(
            state.collection.query,
            query_embeddings=[query_vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        if not raw["documents"] or not raw["documents"][0]:
            return []

        threshold = float(state.settings.get("min_similarity", 0.0))
        out: list[_Candidate] = []
        for doc_id, text, meta, distance in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            out.append(
                _Candidate(
                    doc_id=doc_id,
                    text=text,
                    source=(meta or {}).get("source", "未知来源"),
                    heading=(meta or {}).get("heading", ""),
                    vector_similarity=similarity,
                )
            )
        return out

    @staticmethod
    def _bm25_document_text(doc: str, meta: dict) -> str:
        """BM25 索引的文本：标题一并纳入，与向量化时的处理保持一致。"""
        heading = (meta or {}).get("heading", "")
        return f"{heading}\n{doc}" if heading else doc

    async def _ensure_bm25(self, state: _EngineState) -> BM25Index:
        """
        取得与当前语料和分词器匹配的 BM25 索引，必要时重建。

        索引不做持久化：文本本身已存在 ChromaDB 中，随用随建可以彻底避免
        「两套数据不一致」的问题；分词是 CPU 密集操作，放线程池执行。
        """
        tokenizer_name = state.settings.get("bm25_tokenizer", "jieba")
        index = self._bm25
        if index is not None and index.ready and index.tokenizer_name == tokenizer_name:
            return index

        data = await asyncio.to_thread(
            state.collection.get, include=["documents", "metadatas"]
        )
        ids = data["ids"] or []
        texts = [
            self._bm25_document_text(doc, meta)
            for doc, meta in zip(data["documents"] or [], data["metadatas"] or [])
        ]

        def _build() -> BM25Index:
            fresh = BM25Index(tokenizer_name)
            fresh.build(ids, texts)
            return fresh

        index = await asyncio.to_thread(_build)
        self._bm25 = index
        return index

    async def _fetch_candidates(
        self, state: _EngineState, doc_ids: list[str]
    ) -> dict[str, _Candidate]:
        """按 id 补齐候选内容（用于 BM25 命中但向量没召回的文档）。"""
        if not doc_ids:
            return {}
        data = await asyncio.to_thread(
            state.collection.get, ids=doc_ids, include=["documents", "metadatas"]
        )
        out: dict[str, _Candidate] = {}
        for doc_id, text, meta in zip(
            data["ids"] or [], data["documents"] or [], data["metadatas"] or []
        ):
            out[doc_id] = _Candidate(
                doc_id=doc_id,
                text=text,
                source=(meta or {}).get("source", "未知来源"),
                heading=(meta or {}).get("heading", ""),
            )
        return out

    async def _apply_rerank(
        self, state: _EngineState, question: str, candidates: list[_Candidate]
    ) -> list[_Candidate]:
        """
        用交叉编码器对候选精排。

        向量检索是双塔结构，问题与文档分别编码，无法建模细粒度交互；
        交叉编码器把两者拼在一起打分，精度更高但成本也高，
        因此只对候选池里的少量文档执行。
        """
        reranker = state.reranker
        if reranker is None or not candidates:
            return candidates

        scores = await reranker.score(question, [c.text for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        raw_threshold = str(state.settings.get("min_rerank_score", "")).strip()
        if raw_threshold:
            threshold = float(raw_threshold)
            candidates = [c for c in candidates if (c.rerank_score or 0.0) >= threshold]

        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)

    async def retrieve(
        self, question: str, history: Optional[list[dict]] = None
    ) -> list[Retrieved]:
        """
        检索管道：向量检索 →（可选）BM25 + RRF 融合 →（可选）交叉编码器重排。

        两个增强环节各自由开关控制且互不依赖，可任意组合，
        便于做消融实验对比各环节的贡献。

        history 非空且开启 query rewrite 时，检索使用改写后的 search_query。
        """
        return (await self.retrieve_with_diagnostics(question, history=history))[
            "hits"
        ]

    async def _resolve_search_query(
        self,
        state: _EngineState,
        question: str,
        history: Optional[list[dict]],
    ) -> dict:
        """检索前解析 search_query（含可选的多轮 rewrite）。"""
        return await resolve_search_query(
            client=state.client,
            chat=state.chat,
            chat_error=state.chat_error,
            settings=state.settings,
            question=question,
            history=history,
        )

    async def retrieve_with_diagnostics(
        self, question: str, history: Optional[list[dict]] = None
    ) -> dict:
        """
        与 retrieve 相同，但额外返回各阶段的候选数量与生效开关。

        消融分析需要知道「这次结果是在什么配置下、经过哪些阶段产生的」，
        否则拿到一批 contexts 也无从归因。
        """
        state = self._state
        settings = state.settings
        resolved = await self._resolve_search_query(state, question, history)
        search_query = resolved["search_query"]
        top_k = max(1, int(settings["top_k"]))
        hybrid = bool(settings.get("hybrid_search_enabled"))
        rerank_on = state.reranker is not None

        stages: dict = {
            "original_question": resolved["original_question"],
            "search_query": search_query,
            "query_rewrite_enabled": bool(settings.get("query_rewrite_enabled", True)),
            "query_rewrite_mode": settings.get("query_rewrite_mode", "conditional"),
            "rewrite_applied": resolved["rewrite_applied"],
            "rewrite_skipped_reason": resolved["rewrite_skipped_reason"],
            "hybrid_search_enabled": hybrid,
            "rerank_enabled": rerank_on,
            "bm25_tokenizer": settings.get("bm25_tokenizer") if hybrid else None,
            "rrf_k": int(settings.get("rrf_k", 60)) if hybrid else None,
            "reranker": state.reranker.signature if rerank_on else None,
            "top_k": top_k,
        }

        total = await asyncio.to_thread(state.collection.count)
        stages["corpus_size"] = total
        if total == 0:
            return {"hits": [], "stages": stages}

        # 开了融合或重排就需要更大的候选池，否则重排无从改进排序
        pool = (
            min(max(top_k, int(settings.get("candidate_pool_size", 20))), total)
            if (hybrid or rerank_on)
            else min(top_k, total)
        )
        stages["candidate_pool_size"] = pool

        # ---- 阶段一：向量检索 ----
        vector_hits = await self._vector_search(state, search_query, pool)
        stages["vector_hits"] = len(vector_hits)
        candidates: dict[str, _Candidate] = {c.doc_id: c for c in vector_hits}

        # ---- 阶段二：BM25 + RRF ----
        bm25_ranked: list[str] = []
        if hybrid:
            index = await self._ensure_bm25(state)
            bm25_results = index.search(search_query, pool)
            stages["bm25_hits"] = len(bm25_results)
            stages["bm25_index_size"] = index.size

            # BM25 可能命中向量没召回的文档，需要补齐正文
            missing = [doc_id for doc_id, _ in bm25_results if doc_id not in candidates]
            candidates.update(await self._fetch_candidates(state, missing))

            for doc_id, score in bm25_results:
                if doc_id in candidates:
                    candidates[doc_id].bm25_score = score
                    bm25_ranked.append(doc_id)

            fused = reciprocal_rank_fusion(
                {
                    "vector": [c.doc_id for c in vector_hits],
                    "bm25": bm25_ranked,
                },
                k=int(settings.get("rrf_k", 60)),
            )
            for doc_id, score in fused.items():
                if doc_id in candidates:
                    candidates[doc_id].rrf_score = score
            ordered = sorted(
                candidates.values(), key=lambda c: c.rrf_score or 0.0, reverse=True
            )
            score_type = "rrf"
        else:
            ordered = vector_hits
            score_type = "cosine"

        stages["fused_candidates"] = len(ordered)

        # ---- 阶段三：重排 ----
        if rerank_on:
            ordered = await self._apply_rerank(state, search_query, ordered[:pool])
            stages["reranked_candidates"] = len(ordered)
            score_type = "rerank"

        final = ordered[:top_k]

        def final_score(c: _Candidate) -> float:
            if score_type == "rerank":
                return c.rerank_score or 0.0
            if score_type == "rrf":
                return c.rrf_score or 0.0
            return c.vector_similarity or 0.0

        hits = [
            Retrieved(
                text=c.text,
                source=c.source,
                heading=c.heading,
                similarity=c.vector_similarity,
                bm25_score=c.bm25_score,
                rrf_score=c.rrf_score,
                rerank_score=c.rerank_score,
                score=final_score(c),
                score_type=score_type,
                doc_id=c.doc_id,
            )
            for c in final
        ]
        stages["returned"] = len(hits)
        return {"hits": hits, "stages": stages}

    @staticmethod
    def _build_system_prompt(hits: list[Retrieved]) -> str:
        """把检索结果组装成 system prompt。"""
        if not hits:
            return (
                f"{SYSTEM_PROMPT_BASE}\n\n"
                "知识库中没有检索到与问题相关的内容。请明确告知用户未在文档中找到相关信息，"
                "不要编造文档内容。"
            )

        blocks = []
        for i, hit in enumerate(hits, 1):
            blocks.append(f"[资料{i}｜出处: {hit.citation}]\n{hit.text}")
        context = "\n\n---\n\n".join(blocks)
        return (
            f"{SYSTEM_PROMPT_BASE}\n\n"
            "请仅根据以下参考资料回答问题。资料中没有的信息不要推测或编造，"
            "如资料不足以回答，请如实说明。引用具体内容时请指明出处编号。\n\n"
            f"参考资料：\n{context}"
        )

    async def _stream_chat(self, state: _EngineState, messages: list[dict]) -> AsyncIterator[str]:
        """
        调用 LLM 流式补全，逐段产出文本。抽成独立方法便于测试替换。

        模型名取自解析后的端点，而不是固定读 settings["llm_model"]——
        自建服务用的是 local_llm_model，读错字段会把云端模型名发给本地服务。
        """
        if state.client is None or state.chat is None:
            raise RuntimeError(f"对话模型不可用: {state.chat_error}")

        stream = await state.client.chat.completions.create(
            model=state.chat.model,
            messages=messages,
            temperature=state.settings["temperature"],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(
        self,
        state: _EngineState,
        question: str,
        hits: list[Retrieved],
        history: Optional[list[dict]],
    ) -> list[dict]:
        """
        组装送给 LLM 的完整消息列表。

        [RAGAS] 原本内联在 query() 里，为让评测用的 answer() 复用同一条组装路径而抽出。
        其中的历史消息角色白名单是一道防注入措施，复制一份迟早会与主路径失配。
        """
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(hits)}
        ]

        if history:
            limit = max(0, int(state.settings.get("history_limit", 10)))
            for item in history[-limit:] if limit else []:
                # 只放行合法角色，防止前端传入 system 消息覆盖指令（prompt 注入）
                if item.get("role") in ("user", "assistant") and item.get("content"):
                    messages.append({"role": item["role"], "content": str(item["content"])})

        messages.append({"role": "user", "content": question})
        return messages

    async def answer(
        self, question: str, history: Optional[list[dict]] = None
    ) -> dict:
        """
        [RAGAS] 一次性问答，同时返回答案正文与本次实际用到的检索结果。

        为 RAGAS 评测新增，线上问答走的仍是 query()，本方法不参与请求链路。

        存在的理由是评测：RAGAS 的 faithfulness / answer_relevancy 要求
        「答案」与「生成该答案所依据的 contexts」严格来自同一次调用，
        而 query() 是异步生成器、只吐文本，contexts 留在了函数内部拿不到。
        分两次调用 retrieve 再调 query 看似等价，实则可能因热重载或检索
        非确定性而对不上，评出来的分数也就失去意义。

        返回的 answer 是模型原始输出，不含 query() 末尾追加的「参考来源」脚注——
        那段是给人看的 UI 元素，计入忠实度会被当作无出处的凭空断言。

        返回: {"answer": str, "hits": list[Retrieved], "stages": dict}
        """
        state = self._state

        result = await self.retrieve_with_diagnostics(question, history=history)
        hits = result["hits"]
        messages = self._build_messages(state, question, hits, history)

        pieces = [piece async for piece in self._stream_chat(state, messages)]
        return {
            "answer": "".join(pieces),
            "hits": hits,
            "stages": result["stages"],
        }

    async def query(
        self, question: str, history: Optional[list[dict]] = None
    ) -> AsyncIterator[str]:
        """
        RAG 问答主入口（异步生成器）。

        流程：检索 → 组装 system prompt → 拼接历史 → 流式生成 → 附加参考来源。
        """
        # 取一次快照，避免中途热重载导致前后使用不一致的配置
        state = self._state

        hits = await self.retrieve(question, history=history)
        messages = self._build_messages(state, question, hits, history)

        async for piece in self._stream_chat(state, messages):
            yield piece

        if hits:
            citations = []
            for hit in hits:
                if hit.citation not in citations:
                    citations.append(hit.citation)
            yield "\n\n---\n参考来源：" + "；".join(citations)
