# -*- coding: utf-8 -*-
"""
Markdown 切分逻辑测试。

覆盖三个原实现的缺陷：
  1. chunk_overlap 配置了但从未生效（相邻块零重叠）
  2. 超长文本按空白切词，对无空格的中文完全失效
  3. 按标题切分却丢弃了标题，引用无法精确到章节
"""
from __future__ import annotations

from rag_engine import RAGEngine


def _chunks(engine: RAGEngine, text: str, filename: str = "t.md") -> list[dict]:
    return engine._split_markdown(text, filename)


class TestHeadingPath:
    """标题层级路径应被保留在元数据中。"""

    def test_nested_heading_path_is_recorded(self, make_engine):
        engine = make_engine(chunk_size=500, chunk_overlap=0)
        text = "# 手册\n\n开篇说明\n\n## 配置说明\n\n### API Key 配置\n\n把密钥写进 .env 文件。\n"

        headings = [c["metadata"]["heading"] for c in _chunks(engine, text)]

        assert "手册" in headings
        assert "手册 › 配置说明 › API Key 配置" in headings

    def test_sibling_heading_pops_stack(self, make_engine):
        """同级标题应替换而非叠加，否则路径会越来越长。"""
        engine = make_engine(chunk_size=500, chunk_overlap=0)
        text = "# 根\n\n## 甲\n\n甲的内容\n\n## 乙\n\n乙的内容\n"

        headings = [c["metadata"]["heading"] for c in _chunks(engine, text)]

        assert "根 › 甲" in headings
        assert "根 › 乙" in headings
        assert not any("甲 › 乙" in h for h in headings)

    def test_heading_participates_in_embedding_text(self, make_engine):
        """标题需拼进待向量化文本，以提升用代词指代标题主体的段落的召回。"""
        engine = make_engine(chunk_size=500, chunk_overlap=0)
        text = "# 手册\n\n## 超时设置\n\n默认 60 秒，不要盲目调大。\n"

        target = next(c for c in _chunks(engine, text) if "60 秒" in c["content"])

        assert "超时设置" in target["embed_text"]
        assert "超时设置" not in target["content"]


class TestChunkOverlap:
    """相邻块之间应保留配置要求的重叠。"""

    def test_overlap_is_applied(self, make_engine):
        engine = make_engine(chunk_size=120, chunk_overlap=40)
        paragraphs = [f"第{i}段内容，" + "填充文字" * 12 for i in range(4)]
        text = "# 标题\n\n" + "\n\n".join(paragraphs)

        chunks = [c["content"] for c in _chunks(engine, text)]

        assert len(chunks) >= 2
        # 至少有一对相邻块共享一段非空前后缀
        overlaps = [
            any(chunks[i][-k:] == chunks[i + 1][:k] for k in range(8, 41))
            for i in range(len(chunks) - 1)
        ]
        assert any(overlaps), "相邻块之间没有任何重叠，chunk_overlap 未生效"

    def test_zero_overlap_produces_no_shared_tail(self, make_engine):
        engine = make_engine(chunk_size=120, chunk_overlap=0)
        paragraphs = [f"第{i}段内容，" + "填充文字" * 12 for i in range(4)]
        text = "# 标题\n\n" + "\n\n".join(paragraphs)

        chunks = [c["content"] for c in _chunks(engine, text)]

        for i in range(len(chunks) - 1):
            assert not any(
                chunks[i][-k:] == chunks[i + 1][:k] for k in range(8, 41)
            ), "overlap 为 0 时不应出现重叠"

    def test_overlap_clamped_to_half_chunk_size(self, make_engine):
        """重叠超过块长一半会让相邻块高度冗余，应被夹住。"""
        engine = make_engine(chunk_size=100, chunk_overlap=400)
        text = "# 标题\n\n" + "\n\n".join("段落" + "文字" * 30 for _ in range(3))

        chunks = _chunks(engine, text)

        assert all(len(c["content"]) <= 100 for c in chunks)


class TestLongChineseText:
    """无空格的中文长段落必须被真正切开。"""

    def test_long_cjk_paragraph_is_split(self, make_engine):
        engine = make_engine(chunk_size=200, chunk_overlap=20)
        # 连续中文，没有任何空格；原实现按 text.split() 切词会原样返回整段
        long_para = "这是一个非常长的中文段落用于验证切分行为" * 30
        text = f"# 标题\n\n{long_para}\n"

        chunks = _chunks(engine, text)

        assert len(chunks) > 1, "超长中文段落未被切分"
        assert all(len(c["content"]) <= 200 for c in chunks)

    def test_real_manual_respects_chunk_size(self, make_engine, manual_text):
        """真实中文文档（含超长段落）切分后不应有任何超限块。"""
        engine = make_engine(chunk_size=500, chunk_overlap=50)

        chunks = _chunks(engine, manual_text)

        assert len(chunks) > 5
        oversized = [c for c in chunks if len(c["content"]) > 500]
        assert not oversized, f"存在 {len(oversized)} 个超过 chunk_size 的块"

    def test_split_prefers_punctuation_boundary(self, make_engine):
        engine = make_engine()
        text = "第一句话结束。" + "第二句很长" * 5 + "。" + "第三句" * 5 + "。"

        pieces = RAGEngine._split_long_text(text, 40)

        assert all(len(p) <= 40 for p in pieces)
        # 在标点处断开时，多数片段应以句末标点收尾
        assert sum(1 for p in pieces if p.endswith("。")) >= 1

    def test_split_long_text_always_terminates(self, make_engine):
        """完全没有断点字符时也必须收敛，不能死循环。"""
        pieces = RAGEngine._split_long_text("啊" * 500, 30)

        assert all(len(p) <= 30 for p in pieces)
        assert "".join(pieces) == "啊" * 500


class TestChunkIdentity:
    """块 ID 需稳定且唯一。"""

    def test_ids_are_unique_and_deterministic(self, make_engine, manual_text):
        engine = make_engine(chunk_size=300, chunk_overlap=30)

        first = _chunks(engine, manual_text)
        second = _chunks(engine, manual_text)

        ids = [c["id"] for c in first]
        assert len(ids) == len(set(ids)), "块 ID 出现重复，入库时会互相覆盖"
        assert ids == [c["id"] for c in second], "同一输入两次切分的 ID 不一致"
