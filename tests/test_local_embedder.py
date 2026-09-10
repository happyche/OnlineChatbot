# -*- coding: utf-8 -*-
"""
本地嵌入模型测试。

分两层：
  - 默认执行的部分只验证装配逻辑（签名、惰性加载、模型名路由），
    不加载权重，因此无需下载、跑得很快。
  - 真正加载 bge 模型做中文检索的用例需要约 90MB 权重，默认跳过，
    设置 RUN_LOCAL_EMBED=1 后执行：

        RUN_LOCAL_EMBED=1 pytest tests/test_local_embedder.py
"""
from __future__ import annotations

import os

import pytest

from embeddings import LocalEmbedder, build_embedder

requires_model = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_EMBED") != "1",
    reason="需要下载本地嵌入模型权重，设置 RUN_LOCAL_EMBED=1 后执行",
)


class TestWiring:
    """不加载权重即可验证的装配逻辑。"""

    def test_signature_includes_model_name(self):
        embedder = LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")

        assert embedder.signature == "local:BAAI/bge-small-zh-v1.5"

    def test_model_is_not_loaded_on_construction(self):
        """
        构造时不能加载权重：否则没配好模型的实例连启动都过不去，
        用户也就没法通过配置接口补救。
        """
        embedder = LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")

        assert embedder._model is None

    def test_build_embedder_routes_to_local_model_name(self):
        settings = {
            "embedding_provider": "local",
            "local_embedding_model": "BAAI/bge-small-zh-v1.5",
            # 远程模型名不应被本地提供方误用
            "embedding_model": "text-embedding-v3",
            "embedding_cache_dir": "",
        }

        embedder = build_embedder(settings, client=None)

        assert isinstance(embedder, LocalEmbedder)
        assert embedder.signature == "local:BAAI/bge-small-zh-v1.5"

    def test_local_provider_needs_no_api_key(self):
        """本地路径不该因为缺少 API Key 而失败。"""
        settings = {
            "embedding_provider": "local",
            "local_embedding_model": "BAAI/bge-small-zh-v1.5",
            "llm_api_key": "",
            "embedding_cache_dir": "",
        }

        assert isinstance(build_embedder(settings, client=None), LocalEmbedder)

    def test_switching_provider_changes_signature(self, make_engine):
        """本地与远程的签名必须不同，否则换提供方后旧向量不会被重建。"""
        local = make_engine(
            use_config_embedder=True,
            embedding_provider="local",
            local_embedding_model="BAAI/bge-small-zh-v1.5",
        )
        remote = make_engine(
            use_config_embedder=True,
            embedding_provider="openai",
            embedding_model="text-embedding-v3",
        )

        assert local._state.embedder.signature != remote._state.embedder.signature

    def test_prefixes_are_applied_to_input(self, monkeypatch):
        """配置了指令前缀时（如 e5 系列要求的），前缀必须真的拼到文本上。"""
        embedder = LocalEmbedder(
            model_name="dummy", query_prefix="query: ", doc_prefix="passage: "
        )
        seen: list[list[str]] = []

        def fake_encode(texts):
            seen.append(list(texts))
            return [[0.0, 1.0] for _ in texts]

        monkeypatch.setattr(embedder, "_encode", fake_encode)

        import asyncio

        asyncio.run(embedder.embed_documents(["文档内容"]))
        asyncio.run(embedder.embed_query("用户问题"))

        assert seen[0] == ["passage: 文档内容"]
        assert seen[1] == ["query: 用户问题"]


@requires_model
class TestRealChineseRetrieval:
    """真正加载 bge-small-zh-v1.5，验证中文检索确实可用。"""

    async def test_embeds_chinese_with_expected_dimension(self):
        embedder = LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")

        vectors = await embedder.embed_documents(["这是一段中文测试内容。"])

        assert len(vectors) == 1
        assert len(vectors[0]) == 512

    async def test_vectors_are_l2_normalised(self):
        """bge 输出已归一化，因此可以直接按余弦相似度比较。"""
        embedder = LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")

        vector = await embedder.embed_query("中文查询")

        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-3

    async def test_relevant_chinese_chunk_ranks_first(self, make_engine, manual_text):
        engine = make_engine(
            use_config_embedder=True,
            embedding_provider="local",
            chunk_size=400,
            chunk_overlap=40,
            top_k=3,
            min_similarity=0.2,
        )
        await engine.add_document(manual_text, "运维手册.md")

        hits = await engine.retrieve("这个系统需要什么版本的 Python？")

        assert hits, "中文提问没有召回任何结果"
        assert "3.10" in hits[0].text, f"排在首位的不是正确片段: {hits[0].text[:60]}"
        assert hits[0].similarity > 0.4

    async def test_unrelated_question_scores_lower(self, make_engine, manual_text):
        """无关问题的最高相似度应明显低于相关问题，阈值过滤才有意义。"""
        engine = make_engine(
            use_config_embedder=True,
            embedding_provider="local",
            chunk_size=400,
            chunk_overlap=40,
            top_k=3,
        )
        await engine.add_document(manual_text, "运维手册.md")

        related = await engine.retrieve("依赖怎么安装？")
        unrelated = await engine.retrieve("请推荐一家附近的川菜馆")

        assert related[0].similarity > unrelated[0].similarity + 0.1
