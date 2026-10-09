# -*- coding: utf-8 -*-
"""
向量存储（Qdrant）
==================
RAG 引擎只通过这一层访问向量库，不直接碰 qdrant-client。

两种运行模式，由配置决定：
  QDRANT_URL 非空 → 连接独立的 Qdrant 服务（docker-compose 部署时的默认方式）
  QDRANT_URL 为空 → 嵌入式本地模式，数据落在 QDRANT_PATH 目录，无需起服务
                    （开发、测试、评测脚本用）

几个必须知道的约束：

**只接受显式向量**
  qdrant-client 自带 `add(documents=...)` / `query(query_text=...)` 这类便捷方法，
  会悄悄用内置的英文模型（fastembed 默认 bge-small-en）向量化——和当年
  Chroma 默认嵌入函数是同一个坑：配置的中文模型形同虚设，检索接近随机。
  这一层只暴露「带向量写入、按向量检索」两条路径，把隐式向量化彻底封死。

**集合在首次写入时才创建**
  Qdrant 建集合必须先给出向量维度，而维度只有真正编码过一次才知道。
  为了拿维度在启动时探测编码一次，对远程嵌入就是一次白花的网络调用，
  因此改为惰性创建：集合不存在时 count/检索一律视为空库。

**本地模式的两个限制**
  - 同一目录只能被一个客户端实例打开（文件锁），所以客户端按位置进程级复用；
    热重载、评测脚本反复构造引擎时不会撞锁。
  - 本地实现不是为多线程并发设计的，而引擎把调用丢进线程池执行，
    因此本地模式下所有操作串行化。服务端模式不加锁。

**块 ID 必须是 UUID 或无符号整数**
  Qdrant 不接受任意字符串 ID。调用方须传标准的带连字符 UUID：
  服务端会把其它写法（如 32 位纯十六进制）规范化成带连字符的形式返回，
  本地模式却原样保留，两种模式下 ID 会对不上。
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

import config

logger = logging.getLogger(__name__)

#: payload 中存放块正文的键
TEXT_KEY = "text"

#: 需要按值过滤的 payload 字段。服务端模式下为它们建 keyword 索引，
#: 否则按文档名删除、取整节这类过滤会退化成全量扫描
_INDEXED_FIELDS = ("source", "heading")

#: 写入与翻页的批大小。单次 upsert 过大可能超出服务端的请求体上限（默认 32MB）
_BATCH = 256


@dataclass(frozen=True)
class StoredChunk:
    """向量库中的一条记录。"""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    vector: Optional[list[float]] = None
    #: 仅向量检索结果有值：余弦相似度（Qdrant Cosine 距离直接返回相似度，不是 1-距离）
    similarity: Optional[float] = None


class _ClientHandle:
    """一个客户端连同它的并发约束。"""

    def __init__(self, client: QdrantClient, local: bool):
        self.client = client
        self.local = local
        self._lock = threading.RLock() if local else None

    def guard(self):
        return self._lock if self._lock is not None else contextlib.nullcontext()


_HANDLES: dict[str, _ClientHandle] = {}
_HANDLES_LOCK = threading.Lock()


def _should_bypass_proxy(url: str) -> bool:
    """
    内网 IP，或 docker-compose 里 `qdrant` 这种不带点的单段主机名，
    都不可能经公网代理到达，连接时应忽略 HTTP_PROXY。
    """
    if not config.DEFAULTS.get("bypass_proxy_for_internal"):
        return False
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    return config.is_internal_host(url) or (bool(host) and "." not in host and ":" not in host)


def _open_handle() -> _ClientHandle:
    """按当前配置取得（必要时创建）客户端。配置在调用时读取，便于测试切换目录。"""
    url = config.QDRANT_URL
    key = f"url:{url}" if url else f"path:{config.QDRANT_PATH}"
    with _HANDLES_LOCK:
        handle = _HANDLES.get(key)
        if handle is None:
            if url:
                extra = {}
                if _should_bypass_proxy(url):
                    # 与对话端点同一个坑：HTTP_PROXY 会把内网请求转给企业代理
                    extra["trust_env"] = False
                client = QdrantClient(
                    url=url,
                    api_key=config.QDRANT_API_KEY or None,
                    timeout=config.QDRANT_TIMEOUT,
                    **extra,
                )
                logger.info("向量库: Qdrant 服务 %s", url)
            else:
                config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(config.QDRANT_PATH))
                logger.info("向量库: Qdrant 本地模式 %s", config.QDRANT_PATH)
            handle = _ClientHandle(client, local=not url)
            _HANDLES[key] = handle
        return handle


@atexit.register
def close_all() -> None:
    """
    关闭所有客户端，释放本地模式的文件锁。

    不显式关闭的话，解释器退出时 qdrant-client 的析构会在模块已被回收后
    再去解锁，打出一串无害但吓人的 traceback。
    """
    with _HANDLES_LOCK:
        handles = list(_HANDLES.values())
        _HANDLES.clear()
    for handle in handles:
        try:
            handle.client.close()
        except Exception:  # noqa: BLE001  退出阶段尽力而为
            pass


def _to_filter(where: Optional[dict]) -> Optional[qm.Filter]:
    """{"source": "a.md", "heading": "x"} → 各字段精确匹配的 AND 过滤。"""
    if not where:
        return None
    return qm.Filter(
        must=[
            qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
            for k, v in where.items()
        ]
    )


def _to_chunk(point, similarity: Optional[float] = None) -> StoredChunk:
    payload = dict(point.payload or {})
    text = payload.pop(TEXT_KEY, "")
    vector = point.vector if isinstance(point.vector, list) else None
    return StoredChunk(
        doc_id=str(point.id),
        text=text,
        metadata=payload,
        vector=vector,
        similarity=similarity,
    )


class VectorStore:
    """
    单个 Qdrant 集合的访问入口，集合元数据里记录产出这批向量的 embedder signature。

    用 open() 构造：它负责判断已有集合与当前嵌入模型是否兼容，不兼容就删除。
    """

    def __init__(self, handle: _ClientHandle, name: str, signature: str, signature_key: str):
        self._handle = handle
        self._client = handle.client
        self.name = name
        self.signature = signature
        self._signature_key = signature_key
        self._create_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 打开与兼容性校验
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, name: str, signature: str, signature_key: str) -> tuple["VectorStore", bool]:
        """
        打开集合；若已存在的集合是别的嵌入模型建的，删除它（向量空间不兼容）。

        返回 (store, 是否发生了重建)。
        """
        handle = _open_handle()
        store = cls(handle, name, signature, signature_key)
        with handle.guard():
            if not handle.client.collection_exists(name):
                return store, False
            current = (handle.client.get_collection(name).config.metadata or {}).get(
                signature_key
            )
            if current == signature:
                return store, False
            logger.warning(
                "向量库由 %s 构建，当前嵌入模型为 %s，向量空间不兼容，已重建集合。"
                "请调用 /api/reindex 重新索引 uploads/ 下的文档。",
                current or "未知模型(旧版本遗留)",
                signature,
            )
            handle.client.delete_collection(name)
        return store, True

    def _exists(self) -> bool:
        return self._client.collection_exists(self.name)

    def _ensure_created(self, dim: int) -> None:
        """首次写入时按向量维度建集合。"""
        with self._create_lock:
            if self._exists():
                return
            try:
                self._client.create_collection(
                    collection_name=self.name,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                    metadata={self._signature_key: self.signature},
                )
            except Exception:
                # 服务端模式下可能被另一个进程抢先建好了
                if not self._exists():
                    raise
                return
            if not self._handle.local:
                for name in _INDEXED_FIELDS:
                    self._client.create_payload_index(
                        self.name, field_name=name, field_schema=qm.PayloadSchemaType.KEYWORD
                    )

    @property
    def metadata(self) -> dict:
        """集合元数据；集合尚未创建时为空。"""
        with self._handle.guard():
            if not self._exists():
                return {}
            return dict(self._client.get_collection(self.name).config.metadata or {})

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """写入一批块。向量必须显式提供，见模块说明。"""
        if embeddings is None:
            raise ValueError(
                "写入向量库必须显式提供 embeddings；"
                "不允许依赖向量库自带的嵌入模型隐式向量化。"
            )
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids / embeddings / documents / metadatas 长度不一致")
        if not ids:
            return

        points = [
            qm.PointStruct(
                id=doc_id,
                vector=[float(x) for x in vector],
                payload={**(meta or {}), TEXT_KEY: text},
            )
            for doc_id, vector, text, meta in zip(ids, embeddings, documents, metadatas)
        ]
        with self._handle.guard():
            self._ensure_created(len(points[0].vector))
            for start in range(0, len(points), _BATCH):
                self._client.upsert(
                    collection_name=self.name, points=points[start : start + _BATCH], wait=True
                )

    def delete(self, where: dict) -> int:
        """删除满足过滤条件的块，返回删除数量。"""
        if not where:
            raise ValueError("delete 必须带过滤条件，防止误删整个集合")
        with self._handle.guard():
            if not self._exists():
                return 0
            flt = _to_filter(where)
            n = self._client.count(self.name, count_filter=flt, exact=True).count
            if n:
                self._client.delete(
                    collection_name=self.name,
                    points_selector=qm.FilterSelector(filter=flt),
                    wait=True,
                )
            return n

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def count(self, where: Optional[dict] = None) -> int:
        with self._handle.guard():
            if not self._exists():
                return 0
            return self._client.count(
                self.name, count_filter=_to_filter(where), exact=True
            ).count

    def search(self, query_embedding: Sequence[float], limit: int) -> list[StoredChunk]:
        """按余弦相似度检索，结果按相似度降序。"""
        if isinstance(query_embedding, (str, bytes)) or not query_embedding:
            raise ValueError("检索必须显式提供查询向量，不接受文本查询。")
        with self._handle.guard():
            if not self._exists():
                return []
            resp = self._client.query_points(
                collection_name=self.name,
                query=[float(x) for x in query_embedding],
                limit=max(1, int(limit)),
                with_payload=True,
            )
        return [_to_chunk(p, similarity=float(p.score)) for p in resp.points]

    def get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        with_text: bool = True,
        with_vectors: bool = False,
    ) -> list[StoredChunk]:
        """
        按 id 或过滤条件取记录（不保证顺序）。

        with_text=False 时只取元数据，正文不出库——列文档清单这类场景
        几万个块的正文是纯粹的浪费。
        """
        if ids is not None:
            if not ids:
                return []
            payload = True if with_text else qm.PayloadSelectorExclude(exclude=[TEXT_KEY])
            with self._handle.guard():
                if not self._exists():
                    return []
                points = self._client.retrieve(
                    self.name, ids=list(ids), with_payload=payload, with_vectors=with_vectors
                )
            return [_to_chunk(p) for p in points]

        out: list[StoredChunk] = []
        for chunk in self._scroll(where, with_text=with_text, with_vectors=with_vectors):
            out.append(chunk)
            if limit is not None and len(out) >= limit:
                break
        return out

    def _scroll(
        self, where: Optional[dict], *, with_text: bool = True, with_vectors: bool = False
    ) -> Iterator[StoredChunk]:
        """分页遍历满足条件的记录。每页单独加锁，遍历期间不长时间占着本地模式的锁。"""
        payload = True if with_text else qm.PayloadSelectorExclude(exclude=[TEXT_KEY])
        flt = _to_filter(where)
        offset = None
        while True:
            with self._handle.guard():
                if not self._exists():
                    return
                points, offset = self._client.scroll(
                    self.name,
                    scroll_filter=flt,
                    limit=_BATCH,
                    offset=offset,
                    with_payload=payload,
                    with_vectors=with_vectors,
                )
            for p in points:
                yield _to_chunk(p)
            if offset is None:
                return

    def find_contains(
        self, text: str, where: Optional[dict] = None, limit: int = 40
    ) -> tuple[int, list[StoredChunk]]:
        """
        正文中包含 text（大小写敏感的子串）的块：返回 (命中总数, 前 limit 条)。

        不用 Qdrant 的 MatchText：它是分词匹配而非子串匹配，且会转小写。
        默认分词器按非字母数字切分，一整段中文会被当成一个词——查「超时」
        匹配不到「请求超时时间」，`QUEST_TIME` 也匹配不到 `REQUEST_TIMEOUT`。
        这里分页取正文在本进程内做子串判断，语义与原先 Chroma 的 $contains 一致；
        开销是一次按过滤条件的全量翻页，与构建 BM25 索引同一量级。
        """
        total = 0
        kept: list[StoredChunk] = []
        for chunk in self._scroll(where):
            if text in chunk.text:
                total += 1
                if len(kept) < limit:
                    kept.append(chunk)
        return total, kept
