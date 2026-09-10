# -*- coding: utf-8 -*-
"""
检索增强组件：中文分词、BM25 稀疏检索、RRF 融合、交叉编码器重排
================================================================
向量检索之外的三个可独立开关的增强环节，拆到本模块以便单独测试与替换：

  1. BM25（稀疏检索）
     向量检索靠语义相似度，对精确串很弱 —— 技术文档里的 `kubectl exec`、
     `REQUEST_TIMEOUT`、错误码这类词，字面命中往往比语义相近更可靠。
     BM25 正好互补。

  2. RRF（Reciprocal Rank Fusion）
     融合两路结果。选 RRF 而不是加权求和，是因为余弦相似度与 BM25 分数
     量纲完全不同（前者有界 [-1,1]，后者无界且随语料规模漂移），
     直接加权需要为每个语料重新调参；RRF 只用排名，不需要归一化。

  3. 交叉编码器重排
     向量检索是双塔结构，问题与文档分别编码，无法建模两者的细粒度交互。
     交叉编码器把 (问题, 文档) 拼在一起打分，精度显著更高但成本也高，
     因此只用于对少量候选做精排。

中文注意事项：BM25 依赖分词，直接按空白切分对中文完全无效（整句变一个词）。
本模块提供 jieba 分词与字符二元组两种方案。
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

#: 匹配 CJK 统一汉字
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
#: 保留中日韩文字、字母、数字，其余一律视为分隔符
_KEEP_RE = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]+")


def _normalize(text: str) -> str:
    return _KEEP_RE.sub(" ", text.lower())


def tokenize_bigram(text: str) -> list[str]:
    """
    字符二元组分词，零依赖。

    对中文按相邻两字切分（「超时设置」→ 超时/时设/设置），
    对 ASCII 词保留整词。召回略宽于分词，但不受词典覆盖率影响，
    对新词和产品术语更稳。
    """
    tokens: list[str] = []
    for segment in _normalize(text).split():
        if _CJK_RE.search(segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


def tokenize_jieba(text: str) -> list[str]:
    """
    jieba 分词，中文 BM25 的常规做法。

    词典未覆盖的新词会被切碎，因此技术术语较多的语料可以对比 bigram 方案
    再决定用哪个 —— 这也是 bm25_tokenizer 做成配置项的原因。
    """
    import jieba

    return [t for t in jieba.lcut(_normalize(text)) if t.strip()]


TOKENIZERS = {"jieba": tokenize_jieba, "bigram": tokenize_bigram}


def get_tokenizer(name: str):
    """按名称取分词器；jieba 不可用时降级到 bigram 而不是直接失败。"""
    name = (name or "jieba").lower()
    if name not in TOKENIZERS:
        raise ValueError(f"未知的 bm25_tokenizer: {name!r}（可选 {'/'.join(TOKENIZERS)}）")
    if name == "jieba":
        try:
            import jieba  # noqa: F401
        except ImportError:
            logger.warning("未安装 jieba，BM25 分词降级为字符二元组")
            return tokenize_bigram
    return TOKENIZERS[name]


class BM25Index:
    """
    内存中的 BM25 索引。

    不额外引入存储系统：文本本身已经在 ChromaDB 里，索引随用随建、
    文档变更时置空重建。语料规模不大时重建成本可忽略，
    换来的是「不存在两套数据不一致」这个更重要的性质。
    """

    def __init__(self, tokenizer_name: str = "jieba"):
        self.tokenizer_name = tokenizer_name
        self._tokenize = get_tokenizer(tokenizer_name)
        self._ids: list[str] = []
        self._bm25 = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    @property
    def size(self) -> int:
        return len(self._ids)

    def build(self, ids: list[str], texts: list[str]):
        """用给定的文档集合重建索引。"""
        with self._lock:
            self._ids = list(ids)
            corpus = [self._tokenize(t) for t in texts]
            if not corpus:
                self._bm25 = None
                return
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(corpus)
            logger.info("BM25 索引已构建：%d 篇文档，分词=%s", len(corpus), self.tokenizer_name)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """
        返回 [(文档 id, BM25 分数), ...]，按分数降序。

        分数为 0 表示查询词在该文档中完全没出现，这类结果没有信息量，
        保留下来只会挤占候选池，因此直接丢弃。
        """
        if self._bm25 is None or limit <= 0:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores), key=lambda kv: kv[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:limit] if score > 0]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = 60
) -> dict[str, float]:
    """
    倒数排名融合：score(d) = Σ 1/(k + rank_i(d))，rank 从 1 开始。

    参数:
        rankings : {来源名: 该来源的有序文档 id 列表}
        k        : 平滑常数，越大则头部名次的优势越小。60 是原论文的取值。

    只依赖排名而不依赖原始分数，因此不需要对不同量纲的分数做归一化。
    """
    fused: dict[str, float] = {}
    for ids in rankings.values():
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class Reranker(ABC):
    """交叉编码器重排接口。"""

    signature: str = "unknown"

    @abstractmethod
    async def score(self, query: str, documents: list[str]) -> list[float]:
        """给出每个文档与查询的相关性分数，越大越相关，顺序与输入一致。"""


class LocalReranker(Reranker):
    """
    通过 fastembed 做本地 ONNX 交叉编码器重排。

    默认 BAAI/bge-reranker-base：约 1GB，支持中文。
    权重较大，惰性加载，首次使用时才下载，可用
    scripts/download_model.py --reranker 预下载。
    """

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._load_lock = threading.Lock()
        self.signature = f"local:{model_name}"

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("正在加载重排模型 %s（首次运行需下载权重）", self._model_name)
                try:
                    self._model = TextCrossEncoder(
                        model_name=self._model_name, cache_dir=self._cache_dir
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"重排模型 {self._model_name} 加载失败：{exc}\n"
                        f"可预下载：python scripts/download_model.py --reranker\n"
                        f"或关闭重排：将 rerank_enabled 设为 false"
                    ) from exc
                logger.info("重排模型 %s 已就绪", self._model_name)
        return self._model

    def _run(self, query: str, documents: list[str]) -> list[float]:
        model = self._ensure_model()
        return [float(s) for s in model.rerank(query, documents)]

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        # ONNX 推理是同步 CPU 计算，丢到线程池避免阻塞事件循环
        return await asyncio.to_thread(self._run, query, documents)


class LexicalOverlapReranker(Reranker):
    """
    基于字符二元组重叠度的确定性重排，不依赖模型权重。

    用于自动化测试：能真实改变候选顺序（因此可以断言重排确实生效），
    且结果完全可复现。质量远不及交叉编码器，不要用于生产。
    """

    def __init__(self):
        self.signature = "lexical:bigram-overlap"

    async def score(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = set(tokenize_bigram(query))
        if not query_tokens:
            return [0.0] * len(documents)
        scores = []
        for doc in documents:
            doc_tokens = set(tokenize_bigram(doc))
            hit = len(query_tokens & doc_tokens)
            scores.append(hit / len(query_tokens))
        return scores


def build_reranker(settings: dict) -> Reranker | None:
    """
    依据配置构造重排器；未启用时返回 None。

    provider 取值：
        local   : 本地 ONNX 交叉编码器（生产）
        lexical : 字面重叠度（测试用，无需权重）
    """
    if not settings.get("rerank_enabled"):
        return None

    provider = str(settings.get("rerank_provider", "local")).lower()
    if provider == "lexical":
        return LexicalOverlapReranker()
    if provider != "local":
        raise ValueError(f"未知的 rerank_provider: {provider!r}（可选 local/lexical）")

    return LocalReranker(
        model_name=settings["rerank_model"],
        cache_dir=settings.get("embedding_cache_dir") or None,
    )
