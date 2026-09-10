# -*- coding: utf-8 -*-
"""
嵌入（Embedding）提供方抽象
==========================
把「文本 → 向量」这一步从 RAG 引擎里剥离出来，原因有三：

  1. 修复原实现的缺陷：原来 collection 绑定了 ChromaDB 的 DefaultEmbeddingFunction
     （纯英文的 all-MiniLM-L6-v2），配置里的 embedding_model 从未生效，
     中文文档实际上是用英文模型编码的，检索质量接近随机。
  2. 可注入：测试时传入一个确定性的假 embedder，不需要网络和 API Key。
  3. 可切换：本地与远程两种实现共存，可按数据合规要求和成本取舍。

三个提供方：
  local   : 本地 ONNX 推理（默认）。离线、免费、文本不出本机。
  openai  : 远程 OpenAI 兼容接口。省本地资源，但文本会离开本机且按量计费。
  hashing : 确定性哈希，仅供测试。

每个 Embedder 都有一个 signature，用于标记向量库里的向量是哪个模型产出的。
一旦 signature 变化（换了模型或提供方），旧向量与新查询向量不在同一空间、
甚至维度都不同，必须重建 collection —— 这个判断由 rag_engine 依据 signature 完成。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from hashlib import md5

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """
    嵌入模型接口。

    子类须提供：
        signature : 唯一标识「提供方 + 模型」的字符串，写入 collection 元数据用于兼容性校验
        embed_documents / embed_query : 分别用于入库和查询
            （部分模型要求查询和文档使用不同的指令前缀，故区分两个方法）
    """

    #: 形如 "local:BAAI/bge-small-zh-v1.5"，作为向量空间的指纹
    signature: str = "unknown"

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批待入库的文本编码为向量。"""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """把用户查询编码为向量。"""


class LocalEmbedder(Embedder):
    """
    本地嵌入模型，通过 fastembed 做 ONNX 推理。

    选 fastembed 而不是 sentence-transformers 的原因：后者依赖 PyTorch，
    在 Windows 上要额外装几百 MB 到 2GB；fastembed 只需 onnxruntime，
    而 ChromaDB 已经把它作为依赖装好了，等于零额外重量。

    默认模型 BAAI/bge-small-zh-v1.5 约 90MB、512 维，中文检索质量足够，
    且输出已 L2 归一化，可直接用余弦相似度比较。

    模型在首次编码时才加载（含首次下载），因此服务启动不会被阻塞——
    否则没配好模型的实例连配置接口都起不来。
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        query_prefix: str = "",
        doc_prefix: str = "",
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._model = None
        # 首次编码可能并发触发，加锁避免重复加载同一个模型
        self._load_lock = threading.Lock()
        self.signature = f"local:{model_name}"

    def _ensure_model(self):
        """惰性加载模型；首次调用会下载权重到缓存目录。"""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                from fastembed import TextEmbedding

                logger.info("正在加载本地嵌入模型 %s（首次运行需下载权重）", self._model_name)
                try:
                    self._model = TextEmbedding(
                        model_name=self._model_name, cache_dir=self._cache_dir
                    )
                except Exception as exc:
                    # 下载被限流或中断时，缓存里会留下不完整的快照，
                    # 之后加载会抛出难以理解的底层 ONNX「文件不存在」错误。
                    # 这里换成能指导下一步动作的提示。
                    raise RuntimeError(
                        f"本地嵌入模型 {self._model_name} 加载失败：{exc}\n"
                        f"若缓存不完整，请删除 {self._cache_dir} 后重新预下载：\n"
                        f"    python scripts/download_model.py"
                    ) from exc
                logger.info("本地嵌入模型 %s 已就绪", self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [vec.tolist() for vec in model.embed(texts)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{self._doc_prefix}{t}" for t in texts] if self._doc_prefix else texts
        # ONNX 推理是同步 CPU 计算，丢到线程池以免阻塞事件循环
        return await asyncio.to_thread(self._encode, prepared)

    async def embed_query(self, text: str) -> list[float]:
        prepared = f"{self._query_prefix}{text}" if self._query_prefix else text
        result = await asyncio.to_thread(self._encode, [prepared])
        return result[0]


class OpenAIEmbedder(Embedder):
    """
    通过 OpenAI 兼容接口调用远程嵌入模型（如 DashScope 的 text-embedding-v3）。

    注意远程接口对单次请求的文本条数有上限，因此按 batch_size 分批；
    重试与超时交给 AsyncOpenAI 客户端自身处理。
    """

    def __init__(self, client, model: str, batch_size: int = 10):
        self._client = client
        self._model = model
        # 批大小取 1 以上，避免配置为 0 时死循环
        self._batch_size = max(1, int(batch_size))
        self.signature = f"openai:{model}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await self._client.embeddings.create(input=batch, model=self._model)
            # 接口不保证返回顺序，按 index 排序后再取，避免向量与文本错位
            ordered = sorted(resp.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"嵌入结果数量({len(vectors)})与输入文本数量({len(texts)})不一致"
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed_documents([text])
        return result[0]


class HashingEmbedder(Embedder):
    """
    确定性的字符二元组（bigram）哈希嵌入，不依赖网络、模型文件和 API Key。

    用于自动化测试：它保留了基本的字面相似度（共享 bigram 越多向量越接近），
    因此可以真实地验证「问题能否检索到对应文本块」这一链路，
    同时保证测试结果完全可复现。不适用于生产。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.signature = f"hashing:bigram-{dim}"

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        cleaned = text.lower()
        for i in range(len(cleaned) - 1):
            bigram = cleaned[i : i + 2]
            if not bigram.strip():
                continue
            # 用 md5 而非内置 hash()，因为后者对字符串有随机化盐值，跨进程不稳定
            slot = int.from_bytes(md5(bigram.encode("utf-8")).digest()[:4], "little")
            vec[slot % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            # 全空文本给一个固定的单位向量，避免除零和零向量导致的相似度未定义
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def build_embedder(settings: dict, client) -> Embedder:
    """
    依据配置构造 embedder。

    参数:
        settings : 运行时配置
        client   : AsyncOpenAI 客户端，仅 openai 提供方使用

    本地与远程各自读取自己的模型名配置（local_embedding_model /
    embedding_model），这样切换提供方时不会把一个模型名喂给另一个提供方。
    """
    provider = str(settings.get("embedding_provider", "local")).lower()

    if provider == "local":
        return LocalEmbedder(
            model_name=settings["local_embedding_model"],
            cache_dir=settings.get("embedding_cache_dir") or None,
            # bge-*-v1.5 已改进相似度分布，实测不加指令前缀检索效果不逊于加前缀，
            # 故默认留空。换用 e5 系列等要求前缀的模型时再配置。
            query_prefix=settings.get("embedding_query_prefix", ""),
            doc_prefix=settings.get("embedding_doc_prefix", ""),
        )
    if provider == "hashing":
        return HashingEmbedder()
    if provider != "openai":
        raise ValueError(
            f"未知的 embedding_provider: {provider!r}（可选 local/openai/hashing）"
        )

    if not settings.get("llm_api_key"):
        raise ValueError(
            "使用远程嵌入模型需要配置 llm_api_key；"
            "若要离线运行，请将 embedding_provider 设为 local。"
        )
    return OpenAIEmbedder(
        client=client,
        model=settings["embedding_model"],
        batch_size=settings.get("embedding_batch_size", 10),
    )
