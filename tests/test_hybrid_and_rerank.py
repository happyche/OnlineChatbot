# -*- coding: utf-8 -*-
"""
混合检索（BM25 + 向量 + RRF）与交叉编码器重排的测试。

两个增强环节各自由开关控制，因此测试分三类：
  1. 组件本身（分词、BM25、RRF）的正确性
  2. 开关关闭时不改变原有行为（保证消融对比的基线是干净的）
  3. 开关打开时确实改变了检索结果，并在诊断信息中如实反映

重排一律用注入的确定性实现，不依赖 1GB 的交叉编码器权重。
"""
from __future__ import annotations

import pytest

from retrieval import (
    BM25Index,
    Reranker,
    build_reranker,
    get_tokenizer,
    reciprocal_rank_fusion,
    tokenize_bigram,
)


class ReverseReranker(Reranker):
    """把候选顺序完全反转的重排器，用于断言重排确实介入了排序。"""

    def __init__(self):
        self.signature = "test:reverse"
        self.calls = 0

    async def score(self, query, documents):
        self.calls += 1
        # 分数随位置递增，按分数降序排完即为输入的逆序
        return [float(i) for i in range(len(documents))]


class PickReranker(Reranker):
    """只把包含指定关键词的候选打高分。"""

    def __init__(self, keyword: str):
        self.signature = "test:pick"
        self._keyword = keyword

    async def score(self, query, documents):
        return [10.0 if self._keyword in doc else 0.0 for doc in documents]


# ======================================================================
# 组件：分词
# ======================================================================

class TestTokenizers:
    def test_bigram_splits_chinese(self):
        assert tokenize_bigram("超时设置") == ["超时", "时设", "设置"]

    def test_bigram_keeps_ascii_words(self):
        tokens = tokenize_bigram("设置 REQUEST_TIMEOUT 为 60")

        assert "request" in tokens and "timeout" in tokens and "60" in tokens

    def test_punctuation_becomes_separator(self):
        """标点必须当分隔符，否则会产出跨句的无意义二元组。"""
        assert "。默" not in tokenize_bigram("默认值。默认超时")

    def test_single_chinese_char_kept(self):
        assert tokenize_bigram("是") == ["是"]

    def test_jieba_produces_word_level_tokens(self):
        tokens = get_tokenizer("jieba")("检索增强生成很有用")

        # 词粒度分词的结果应比字符二元组少
        assert len(tokens) < len(tokenize_bigram("检索增强生成很有用"))
        assert all(t.strip() for t in tokens)

    def test_unknown_tokenizer_rejected(self):
        with pytest.raises(ValueError, match="bm25_tokenizer"):
            get_tokenizer("神秘分词器")


# ======================================================================
# 组件：BM25
# ======================================================================

class TestBM25Index:
    def _index(self, tokenizer="bigram") -> BM25Index:
        index = BM25Index(tokenizer)
        index.build(
            ["a", "b", "c"],
            [
                "执行 kubectl exec 进入容器排查问题",
                "系统要求 Python 3.10 或以上版本",
                "运维值班电话由值班表统一发布",
            ],
        )
        return index

    def test_exact_term_ranks_first(self):
        """BM25 的价值就在精确串匹配。"""
        index = self._index()

        results = index.search("kubectl exec 怎么用", limit=3)

        assert results
        assert results[0][0] == "a"

    def test_zero_score_results_dropped(self):
        """查询词完全没出现的文档没有信息量，不应占用候选池。"""
        index = self._index()

        results = index.search("kubectl", limit=3)

        assert all(score > 0 for _, score in results)
        assert len(results) < 3

    def test_limit_respected(self):
        index = self._index()

        assert len(index.search("版本 容器 值班", limit=2)) <= 2

    def test_empty_corpus_is_not_ready(self):
        index = BM25Index("bigram")
        index.build([], [])

        assert index.ready is False
        assert index.search("任何查询", limit=5) == []

    def test_unmatched_query_returns_empty(self):
        index = self._index()

        assert index.search("完全无关的外星语", limit=5) == []

    def test_size_reports_document_count(self):
        assert self._index().size == 3


# ======================================================================
# 组件：RRF
# ======================================================================

class TestReciprocalRankFusion:
    def test_document_in_both_lists_wins(self):
        """两路都认可的文档应排在只有一路认可的之前。"""
        fused = reciprocal_rank_fusion(
            {"vector": ["x", "y"], "bm25": ["y", "z"]}, k=60
        )

        assert fused["y"] > fused["x"]
        assert fused["y"] > fused["z"]

    def test_scores_follow_rank_not_raw_score(self):
        """RRF 只看排名，因此第一名的分数固定为 1/(k+1)。"""
        fused = reciprocal_rank_fusion({"only": ["first", "second"]}, k=60)

        assert fused["first"] == pytest.approx(1 / 61)
        assert fused["second"] == pytest.approx(1 / 62)

    def test_larger_k_flattens_ranking(self):
        small = reciprocal_rank_fusion({"s": ["a", "b"]}, k=1)
        large = reciprocal_rank_fusion({"s": ["a", "b"]}, k=1000)

        assert small["a"] / small["b"] > large["a"] / large["b"]

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == {}


# ======================================================================
# 组件：reranker 构造
# ======================================================================

class TestBuildReranker:
    def test_disabled_returns_none(self):
        assert build_reranker({"rerank_enabled": False}) is None

    def test_local_provider_is_lazy(self):
        """构造时不能加载 1GB 权重，否则关掉开关也要付下载代价。"""
        reranker = build_reranker(
            {
                "rerank_enabled": True,
                "rerank_provider": "local",
                "rerank_model": "BAAI/bge-reranker-base",
            }
        )

        assert reranker.signature == "local:BAAI/bge-reranker-base"
        assert reranker._model is None

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="rerank_provider"):
            build_reranker({"rerank_enabled": True, "rerank_provider": "魔法"})

    async def test_lexical_provider_scores_by_overlap(self):
        reranker = build_reranker(
            {"rerank_enabled": True, "rerank_provider": "lexical"}
        )

        scores = await reranker.score("超时设置", ["超时设置默认 60 秒", "无关内容"])

        assert scores[0] > scores[1]


# ======================================================================
# 引擎：开关关闭时的基线行为
# ======================================================================

class TestDefaultsOff:
    async def test_both_flags_off_by_default(self, make_engine, manual_text):
        engine = make_engine()
        await engine.add_document(manual_text, "运维手册.md")

        result = await engine.retrieve_with_diagnostics("Python 版本要求")

        assert result["stages"]["hybrid_search_enabled"] is False
        assert result["stages"]["rerank_enabled"] is False
        assert all(h.score_type == "cosine" for h in result["hits"])

    async def test_score_equals_similarity_when_off(self, make_engine, manual_text):
        engine = make_engine()
        await engine.add_document(manual_text, "运维手册.md")

        hits = await engine.retrieve("依赖怎么安装")

        assert hits
        assert all(h.score == h.similarity for h in hits)
        assert all(h.bm25_score is None and h.rrf_score is None for h in hits)

    async def test_pool_equals_top_k_when_off(self, make_engine, manual_text):
        """不做融合与重排时没必要多取候选，避免白付检索成本。"""
        engine = make_engine(top_k=3)
        await engine.add_document(manual_text, "运维手册.md")

        stages = (await engine.retrieve_with_diagnostics("配置"))["stages"]

        assert stages["candidate_pool_size"] == 3


# ======================================================================
# 引擎：混合检索
# ======================================================================

class TestHybridSearch:
    async def test_score_type_becomes_rrf(self, make_engine, manual_text):
        engine = make_engine(hybrid_search_enabled=True, bm25_tokenizer="bigram")
        await engine.add_document(manual_text, "运维手册.md")

        result = await engine.retrieve_with_diagnostics("Python 版本要求")

        assert result["stages"]["hybrid_search_enabled"] is True
        assert result["hits"]
        assert all(h.score_type == "rrf" for h in result["hits"])

    async def test_bm25_scores_recorded(self, make_engine, manual_text):
        engine = make_engine(hybrid_search_enabled=True, bm25_tokenizer="bigram")
        await engine.add_document(manual_text, "运维手册.md")

        result = await engine.retrieve_with_diagnostics("REQUEST_TIMEOUT 默认值")

        assert result["stages"]["bm25_hits"] > 0
        assert any(h.bm25_score for h in result["hits"])

    async def test_pool_enlarged_when_hybrid_on(self, make_engine, manual_text):
        engine = make_engine(
            hybrid_search_enabled=True, top_k=2, candidate_pool_size=15
        )
        await engine.add_document(manual_text, "运维手册.md")

        stages = (await engine.retrieve_with_diagnostics("配置"))["stages"]

        assert stages["candidate_pool_size"] > stages["top_k"]

    async def test_bm25_recalls_when_vector_path_filtered_out(
        self, make_engine, manual_text
    ):
        """
        min_similarity 只作用于向量这一路。

        把阈值拉到 1.0 让向量路全部被过滤，此时仍能出结果，
        说明 BM25 是一条独立的召回通道 —— 这正是混合检索的意义。
        """
        engine = make_engine(
            hybrid_search_enabled=True,
            bm25_tokenizer="bigram",
            min_similarity=1.0,
        )
        await engine.add_document(manual_text, "运维手册.md")

        result = await engine.retrieve_with_diagnostics("kubectl 容器 Python 版本")

        assert result["stages"]["vector_hits"] == 0
        assert result["hits"], "向量路被过滤后 BM25 未能召回任何内容"
        assert all(h.similarity is None for h in result["hits"])

    async def test_vector_only_path_returns_nothing_without_hybrid(
        self, make_engine, manual_text
    ):
        """同样的阈值下，不开混合检索就应该没有结果 —— 作为上一个用例的对照。"""
        engine = make_engine(hybrid_search_enabled=False, min_similarity=1.0)
        await engine.add_document(manual_text, "运维手册.md")

        assert await engine.retrieve("kubectl 容器 Python 版本") == []

    async def test_tokenizer_change_rebuilds_index(self, make_engine, manual_text):
        engine = make_engine(hybrid_search_enabled=True, bm25_tokenizer="bigram")
        await engine.add_document(manual_text, "运维手册.md")
        await engine.retrieve("配置")
        assert engine._bm25.tokenizer_name == "bigram"

        # 直接改快照里的配置，模拟热重载后分词器变化
        engine._state.settings["bm25_tokenizer"] = "jieba"
        await engine.retrieve("配置")

        assert engine._bm25.tokenizer_name == "jieba"

    async def test_index_invalidated_on_document_change(self, make_engine, manual_text):
        """新增文档后 BM25 索引必须重建，否则新内容永远检索不到。"""
        engine = make_engine(hybrid_search_enabled=True, bm25_tokenizer="bigram")
        await engine.add_document(manual_text, "运维手册.md")
        await engine.retrieve("配置")
        size_before = engine._bm25.size

        await engine.add_document("# 新增\n\n这是一段新增的内容。\n", "新增.md")
        assert engine._bm25 is None, "新增文档后索引未失效"

        await engine.retrieve("新增的内容")
        assert engine._bm25.size > size_before


# ======================================================================
# 引擎：重排
# ======================================================================

class TestRerank:
    async def test_rerank_reorders_results(self, make_engine, manual_text):
        """注入反转重排器，最终顺序应与未重排时相反。"""
        engine_plain = make_engine(top_k=3)
        await engine_plain.add_document(manual_text, "运维手册.md")
        baseline = [h.text for h in await engine_plain.retrieve("Python 版本要求")]

        reranker = ReverseReranker()
        engine = make_engine(top_k=3, candidate_pool_size=3, reranker=reranker)
        await engine.add_document(manual_text, "运维手册.md")
        reranked = [h.text for h in await engine.retrieve("Python 版本要求")]

        assert reranker.calls == 1
        assert reranked == list(reversed(baseline))

    async def test_score_type_becomes_rerank(self, make_engine, manual_text):
        engine = make_engine(reranker=ReverseReranker())
        await engine.add_document(manual_text, "运维手册.md")

        result = await engine.retrieve_with_diagnostics("配置")

        assert result["stages"]["rerank_enabled"] is True
        assert result["stages"]["reranker"] == "test:reverse"
        assert all(h.score_type == "rerank" for h in result["hits"])
        assert all(h.rerank_score is not None for h in result["hits"])

    async def test_rerank_can_promote_a_low_ranked_candidate(
        self, make_engine, manual_text
    ):
        """
        重排的实际价值：把向量检索排在候选池后段的正确答案提到首位。

        top_k=1 时若没有重排，返回的就是向量第一名；开启重排后，
        应返回重排器认定的那一条。
        """
        engine = make_engine(
            top_k=1, candidate_pool_size=20, reranker=PickReranker("值班表")
        )
        await engine.add_document(manual_text, "运维手册.md")

        hits = await engine.retrieve("配置说明")

        assert len(hits) == 1
        assert "值班表" in hits[0].text

    async def test_min_rerank_score_filters(self, make_engine, manual_text):
        engine = make_engine(
            reranker=PickReranker("值班表"), min_rerank_score="5.0", top_k=10
        )
        await engine.add_document(manual_text, "运维手册.md")

        hits = await engine.retrieve("配置说明")

        assert hits
        assert all(h.rerank_score >= 5.0 for h in hits)
        assert all("值班表" in h.text for h in hits)

    async def test_empty_threshold_means_no_filter(self, make_engine, manual_text):
        engine = make_engine(reranker=PickReranker("值班表"), min_rerank_score="", top_k=5)
        await engine.add_document(manual_text, "运维手册.md")

        hits = await engine.retrieve("配置说明")

        assert any((h.rerank_score or 0) == 0.0 for h in hits)

    async def test_pool_enlarged_when_rerank_on(self, make_engine, manual_text):
        engine = make_engine(top_k=2, candidate_pool_size=12, reranker=ReverseReranker())
        await engine.add_document(manual_text, "运维手册.md")

        stages = (await engine.retrieve_with_diagnostics("配置"))["stages"]

        # 候选池会被语料实际大小截断，因此只断言「大于 top_k 且不超过配置值」
        assert stages["top_k"] < stages["candidate_pool_size"] <= 12
        assert stages["candidate_pool_size"] <= stages["corpus_size"]
        assert stages["returned"] <= 2


# ======================================================================
# 引擎：两者叠加
# ======================================================================

class TestHybridPlusRerank:
    async def test_both_stages_run(self, make_engine, manual_text):
        engine = make_engine(
            hybrid_search_enabled=True,
            bm25_tokenizer="bigram",
            candidate_pool_size=15,
            top_k=3,
            reranker=ReverseReranker(),
        )
        await engine.add_document(manual_text, "运维手册.md")

        result = await engine.retrieve_with_diagnostics("REQUEST_TIMEOUT 默认值")
        stages = result["stages"]

        assert stages["hybrid_search_enabled"] is True
        assert stages["rerank_enabled"] is True
        assert stages["bm25_hits"] > 0
        # 融合分与重排分都应保留，便于归因
        assert all(h.rrf_score is not None for h in result["hits"])
        assert all(h.rerank_score is not None for h in result["hits"])
        # 最终排序依据是重排分
        assert all(h.score_type == "rerank" for h in result["hits"])


# ======================================================================
# 接口
# ======================================================================

class TestRetrieveEndpoint:
    def _upload(self, client, manual_text):
        return client.post(
            "/api/upload",
            files=[("files", ("运维手册.md", manual_text.encode("utf-8"), "text/markdown"))],
        )

    def test_returns_contexts_and_stages(self, client, manual_text):
        self._upload(client, manual_text)

        resp = client.post("/api/retrieve", json={"question": "Python 版本要求是什么？"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "Python 版本要求是什么？"
        assert body["contexts"]
        assert body["stages"]["corpus_size"] > 0
        assert all("citation" in ctx for ctx in body["contexts"])

    def test_no_generation_happens(self, client, manual_text):
        """检索接口不应触发 LLM 调用，否则评估检索时白付生成成本。"""
        self._upload(client, manual_text)
        before = len(client.captured_messages)

        client.post("/api/retrieve", json={"question": "依赖怎么安装？"})

        assert len(client.captured_messages) == before

    def test_flags_reflected_after_settings_change(self, client, manual_text):
        self._upload(client, manual_text)
        client.post(
            "/api/settings",
            json={"hybrid_search_enabled": True, "bm25_tokenizer": "bigram"},
        )

        body = client.post("/api/retrieve", json={"question": "超时设置"}).json()

        assert body["stages"]["hybrid_search_enabled"] is True
        assert body["stages"]["bm25_tokenizer"] == "bigram"
        assert all(ctx["score_type"] == "rrf" for ctx in body["contexts"])

    def test_empty_knowledge_base(self, client):
        body = client.post("/api/retrieve", json={"question": "任何问题"}).json()

        assert body["contexts"] == []
        assert body["stages"]["corpus_size"] == 0

    def test_health_reports_retrieval_flags(self, client):
        active = client.get("/api/health").json()["active"]

        assert "hybrid_search_enabled" in active
        assert "rerank_enabled" in active
