# -*- coding: utf-8 -*-
"""
取材工具集
==========
工具按**失败模式**划分，不按功能划分。每个工具存在的前提是
某一类问题在固定检索管道上必然失败：

  search_docs    —— 语义检索，把原本的「必经之路」变成「可选动作」
  expand_section —— 把命中片段还原成完整章节：「这一节的完整步骤是什么」
  find_literal   —— 精确串枚举：「所有出现 X 的地方」，top_k 会悄悄截断

三条贯穿所有工具的规则：

**一、不抛异常，返回可读的失败文本。** 模型读得懂才能改策略。
章节找不到时带上该文档里可用的章节列表，比一个 KeyError 有用得多。

**二、「执行失败」与「执行成功但无结果」严格分开。** 前者该重试或换工具，
后者是有效信息（「文档里确实没有」），应当直接支持拒答。
混成一句话的后果是模型在空结果上反复重检索，撞满调用上限之后给出更差的答案。

**三、每个工具自带 token 上限。** search_docs 老实返回 20 个候选的全文
会直接撑爆窗口。返回几条、每条截多长是工具本身的职责，不是调用方的事。
SummarizationMiddleware 折叠的是**历史消息**，管不到单条工具返回的大小。
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from observability import estimate_tokens
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)

#: 报错时最多列出多少个候选章节 / 文档名——提示要能读，不能是一屏 JSON
_MAX_HINT_ITEMS = 12

#: find_literal 单次返回的命中数上限。总数仍然如实回报，不会被这个数掩盖。
_MAX_LITERAL_HITS = 40

TOOL_NAMES = ("search_docs", "expand_section", "find_literal")


# ======================================================================
# token 估算与打包
# ======================================================================
#
# estimate_tokens 从 observability 导入（见本文件顶部），不在这里各算一套：
# 工具层的 token 预算和请求级的用量计费必须是同一个口径，否则
# 「预算说 1200、账单说 1800」这种对不上账的情况根本无从排查。


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    把文本截到预算内。按 estimate_tokens 的口径逐字累加，保证两者自洽。

    ⚠️ 这个逐字累加的循环必须与 estimate_tokens 的公式保持一致。
    改动任何一方都要同时改另一方，否则截断会在边界上超出上限一小截。
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    used = 0
    out: list[str] = []
    pending_ascii = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            cost = 1
        else:
            pending_ascii += 1
            cost = 1 if pending_ascii % 4 == 1 else 0
        if used + cost > max_tokens:
            break
        used += cost
        out.append(ch)
    return "".join(out)


#: 装配后的片段之间的分隔符
_SEP = "\n\n---\n\n"


def _pack_items(items: list[dict], budget_tokens: int) -> tuple[list[dict], bool]:
    """
    在 token 预算内尽量多装几个片段，装不下就整块丢弃而不是半截截断。

    只有最后一块允许被截断——半截的文本块仍然有参考价值，
    而中间丢一块会让后面的块错位。

    在**结构化片段**上做打包而不是在拼好的字符串上做，是为了让
    「模型实际看到了哪些内容」能被如实导出（见 build_tools 里的 artifact）。
    评测 faithfulness 时，contexts 必须是模型真正看过的那一份：
    拿截断前的原文去评分，等于在评一个不存在的输入。

    定位标注（「文档: X ｜ 章节: Y」）的开销一并计入预算，
    否则装到边界时会超出上限一小截。
    """
    kept: list[dict] = []
    used = 0
    truncated = False
    for item in items:
        label_cost = estimate_tokens(_label(item["source"], item["heading"])) + 1
        cost = estimate_tokens(item["text"]) + label_cost
        if used + cost <= budget_tokens:
            kept.append(dict(item))
            used += cost
            continue
        remaining = budget_tokens - used - label_cost
        if not kept and remaining > 0:
            kept.append({**item, "text": truncate_to_tokens(item["text"], remaining)})
        truncated = True
        break
    return kept, truncated


def _render(items: list[dict]) -> str:
    """把片段渲染成给模型看的文本，每段带上可原样抄进工具参数的定位标注。"""
    return _SEP.join(
        f"{_label(i['source'], i['heading'])}\n{i['text']}" for i in items
    )


# ======================================================================
# 参数 schema
# ======================================================================

class SearchDocsArgs(BaseModel):
    query: str = Field(description="检索用的完整问句，不要带指代词")
    top_k: Optional[int] = Field(default=None, description="返回条数，默认使用系统配置")


class ExpandSectionArgs(BaseModel):
    document: str = Field(description="文档名，抄自检索结果里的「文档」")
    section: str = Field(description="章节名，抄自检索结果里的「章节」")


class FindLiteralArgs(BaseModel):
    text: str = Field(description="要查找的字面串，如 REQUEST_TIMEOUT")
    document: Optional[str] = Field(default=None, description="限定某个文档名；留空则查全部")


# ======================================================================
# 辅助
# ======================================================================

def _artifact(
    tool: str,
    contexts: list[dict],
    truncated: bool = False,
    meta: Optional[dict] = None,
    *,
    ok: bool = True,
    empty: bool = False,
) -> dict:
    """
    工具返回的结构化副本，挂在 ToolMessage.artifact 上。

    **它不进 prompt**（`response_format="content_and_artifact"` 的语义），
    只留在图状态里供编排层读取。存在的理由是评测：RAGAS 的三个指标都要
    `retrieved_contexts`，而那份数据只有工具自己知道。从给模型看的文本里
    反向切分是可行但脆弱的——改一次渲染格式，评测就静默失真了。

    contexts 里是**截断之后**的文本，即模型真正看到的那一份。
    """
    return {
        "tool": tool,
        "ok": ok,
        "empty": empty,
        "truncated": truncated,
        "contexts": contexts,
        "meta": meta or {},
    }


def _fail(tool: str, error: str, hint: str = "") -> tuple[str, dict]:
    """渲染一次失败。hint 必须跟着 error 一起给，只说「失败了」不说怎么改等于没说。"""
    text = f"[{tool} 失败] {error}"
    if hint:
        text += f"\n提示：{hint}"
    return text, _artifact(tool, [], ok=False, meta={"error": error, "hint": hint})


def _empty(tool: str, message: str) -> tuple[str, dict]:
    """
    执行成功但没有内容。与失败严格分开：这不是错误，它恰恰是拒答的依据。

    混成一个错误码的后果是模型在空结果上反复重检索，
    撞满调用次数上限之后给出一个比直接拒答更差的答案。
    """
    return f"[{tool}] {message}", _artifact(tool, [], empty=True)


def _format_items(items: list[str]) -> str:
    """把候选列表渲染成一行提示，超量时截断并注明总数。"""
    shown = items[:_MAX_HINT_ITEMS]
    text = "、".join(shown)
    if len(items) > len(shown):
        text += f"…（共 {len(items)} 项）"
    return text


def _label(source: str, heading: str) -> str:
    """片段的定位标注。写成带字段名的形式，让模型能原样抄进工具参数。"""
    return f"文档: {source} ｜ 章节: {heading or '（无标题正文）'}"


def _overlap_len(prev: str, current: str, max_overlap: int) -> int:
    """求 prev 的后缀与 current 的前缀的最长公共长度，上限 max_overlap。"""
    limit = min(max_overlap, len(prev), len(current))
    for size in range(limit, 0, -1):
        if prev[-size:] == current[:size]:
            return size
    return 0


def _stitch(chunks: list[dict], max_overlap: int) -> str:
    """
    把同一章节的相邻片段拼成连续正文，去掉切分时的重叠部分。

    重叠是切分阶段刻意留的（避免跨块语义被切断），那是为检索服务的；
    一次性读整节时它就是重复内容，既占 token 又让模型看到两遍同样的话。
    """
    parts: list[str] = []
    previous = ""
    for chunk in chunks:
        text = chunk["text"]
        if previous and max_overlap > 0:
            cut = _overlap_len(previous, text, max_overlap)
            if cut:
                text = text[cut:]
        parts.append(text)
        previous = chunk["text"]
    return "\n".join(p for p in parts if p.strip())


def _confidence_note(hits: list, settings: dict) -> str:
    """
    命中质量存疑时，附一句**带方向**的提示；否则返回空串。

    为什么需要它：向量检索的语义是「返回最相似的 K 条」，不是「返回相关的条目」。
    知识库里没有的问题照样会拿到 top_k 个片段，而且往往主题沾边、读起来通顺，
    模型分辨不出来，于是挨个换工具重试直到撞上调用次数上限——几分钟只换来
    一句「不知道」。给它一个显式的质量信号，这一轮就能判。

    为什么看重排分而不看别的：四个分数里只有它有绝对含义。余弦受语料与模型
    分布影响（实测无关问题的余弦能比库内问题还高），BM25 是词频量纲，
    而 RRF 只用名次、完全丢弃幅度——任何查询的 top1 拿到的 RRF 分都一样，
    拿它当阈值是零信息量。所以重排关掉时这里直接不提示，而不是换个分数凑合。

    为什么只提示不过滤：见 config.agent_low_confidence_score 的注释。

    提示措辞上刻意做了两件事：给出**一条**具体的下一步（而不是泛泛的「再找找」，
    那只会让它把剩下的工具都试一遍），以及明说「不要硬凑」——低分片段最危险的
    用法不是被丢弃，而是被当成依据编出一个看着有出处的答案。
    """
    if not hits:
        return ""
    top = hits[0]
    if getattr(top, "score_type", "") != "rerank":
        return ""
    score = getattr(top, "rerank_score", None)
    if score is None:
        return ""

    threshold = float(settings.get("agent_low_confidence_score", 1.0))
    if score >= threshold:
        return ""
    return (
        f"\n\n注意：本次命中相关性偏低（重排最高分 {score:.1f}，低于 {threshold:g}），"
        "上面这些片段很可能与问题无关。"
        "若问题涉及精确串（错误码、命令、参数名、配置项），用 find_literal 再确认一次；"
        "若不属此类或已确认过，请直接说明知识库中没有相关内容，"
        "不要从上面的片段里硬凑答案。"
    )


def _payload_budget(settings: dict) -> int:
    return max(100, int(settings.get("agent_tool_payload_tokens", 1200)))


def _expand_budget(settings: dict) -> int:
    # 完整章节比检索片段长，给它独立且更宽的预算：
    # 截断一份「完整步骤」等于没有完成这个工具存在的目的
    return max(
        _payload_budget(settings),
        int(settings.get("agent_expand_payload_tokens", 2400)),
    )


# ======================================================================
# 工具实现
# ======================================================================

async def search_docs(
    engine: RAGEngine, settings: dict, query: str, top_k=None
) -> tuple[str, dict]:
    """语义检索。直接复用现有的三段式管道，不重新实现任何检索逻辑。"""
    query = (query or "").strip()
    if not query:
        return _fail("search_docs", "query 不能为空", "请给出完整的检索问句")

    # 不传 history：agent 模式下检索问句由模型自己写成不含指代的完整问句，
    # 再套一层基于历史的改写只会把它改回去。
    result = await engine.retrieve_with_diagnostics(query)
    hits = result["hits"]
    if top_k is not None and top_k > 0:
        hits = hits[:top_k]

    if not hits:
        # 成功执行、确实没有内容。这不是错误——它是拒答的依据。
        #
        # 但措辞不能替模型下「知识库里没有」这个结论：语义检索受措辞影响，
        # 没命中也可能只是问句用词和文档对不上。所以给的是下一步，不是判决。
        return _empty(
            "search_docs",
            f"检索「{query}」未命中任何内容。语义检索受措辞影响，这不一定代表"
            "知识库里没有——若问题涉及精确串（错误码、命令、参数名）可用 "
            "find_literal 再确认一次，否则可判定知识库中没有相关内容。",
        )

    items = [
        {"text": h.text, "source": h.source, "heading": h.heading} for h in hits
    ]
    kept, truncated = _pack_items(items, _payload_budget(settings))
    head = f"[search_docs] 检索「{query}」命中 {len(hits)} 个片段"
    tail = "\n…（结果已按长度上限截断）" if truncated else ""
    note = _confidence_note(hits, settings)
    top_score = getattr(hits[0], "rerank_score", None)
    return (
        f"{head}\n{_render(kept)}{tail}{note}",
        _artifact(
            "search_docs",
            kept,
            truncated,
            {
                "hits": len(hits),
                # 进 artifact 不进 prompt：评测要能统计「低置信度命中占多少」，
                # 而这个数字对模型没有额外价值——提示里已经把结论说了。
                "top_rerank_score": top_score,
                "low_confidence": bool(note),
            },
        ),
    )


async def expand_section(
    engine: RAGEngine, settings: dict, document: str, section: str
) -> tuple[str, dict]:
    """
    把某个章节的全部片段按原文顺序拼回完整正文。

    先按 (document, section) 精确匹配，下推给向量库；匹配不上再退回到
    在该文档内做章节名包含匹配——模型抄标题时丢掉层级前缀是常见情况
    （只给「超时设置」而完整路径是「配置说明 › 超时设置」）。
    兜底的扫描范围限定在单篇文档内，量级可控。
    """
    document = (document or "").strip()
    section = (section or "").strip()
    if not document:
        return _fail(
            "expand_section", "document 不能为空", "请抄写检索结果里标注的「文档」"
        )

    # 不预先校验文档名。校验要扫全部块的 metadata 才能算出文档清单，
    # 而正常路径（名字是从上一步检索结果里抄来的）根本用不上它。
    # 先查，查不到再解释——只有真出错时才付那份代价。
    chunks = await engine.fetch_section(document, section) if section else []
    if not chunks:
        # 章节名对不上：退回到该文档内做包含匹配，范围限定在单篇文档内
        whole = await engine.fetch_section(document)
        if not whole:
            available = [d["filename"] for d in engine.list_documents()]
            return _fail(
                "expand_section",
                f"没有名为 {document!r} 的文档",
                f"可用文档：{_format_items(available)}",
            )
        needle = section.lower()
        chunks = [c for c in whole if needle and needle in c["heading"].lower()]
        if not chunks:
            headings: list[str] = []
            for c in whole:
                if c["heading"] and c["heading"] not in headings:
                    headings.append(c["heading"])
            return _fail(
                "expand_section",
                f"在 {document!r} 里找不到章节 {section!r}",
                f"该文档的章节有：{_format_items(headings)}"
                if headings
                else "该文档没有任何章节标题",
            )

    body = _stitch(chunks, engine.chunk_overlap)
    heading = chunks[0]["heading"] or "（无标题正文）"
    items = [{"text": body, "source": document, "heading": heading}]
    kept, truncated = _pack_items(items, _expand_budget(settings))
    head = f"[expand_section] 章节「{document} › {heading}」共 {len(chunks)} 个片段，已拼接完整正文"
    tail = "\n…（结果已按长度上限截断）" if truncated else ""
    return (
        f"{head}\n{_render(kept)}{tail}",
        _artifact("expand_section", kept, truncated, {"chunks": len(chunks)}),
    )


async def find_literal(
    engine: RAGEngine, settings: dict, text: str, document=None
) -> tuple[str, dict]:
    """
    字面串查找。过滤下推给向量库，不在 Python 里扫全库。

    大小写敏感、不支持正则、只匹配正文——三条性质都来自下推实现，
    在工具描述里写明了，模型不会拿它去做模糊查找。
    """
    text = (text or "").strip()
    if not text:
        return _fail("find_literal", "text 不能为空", "请给出要查找的字面串")

    document = (str(document).strip() or None) if document else None
    result = await engine.find_literal(text, document, limit=_MAX_LITERAL_HITS)
    total = result["total"]
    chunks = result["chunks"]

    if not total:
        # 零命中有两种成因，对下一步完全不同：
        # 文档名写错了（该改参数重试）vs 确实没有这个串（该据实拒答）。
        # 文档存在性用 limit=1 的查询判断，不去算全量文档清单。
        if document and not await engine.document_exists(document):
            available = [d["filename"] for d in engine.list_documents()]
            return _fail(
                "find_literal",
                f"没有名为 {document!r} 的文档",
                f"可用文档：{_format_items(available)}",
            )
        return _empty(
            "find_literal",
            f"全库没有正文包含 {text!r}。"
            "注意大小写敏感；若要按语义或章节名查找请改用 search_docs。",
        )

    items = [
        {"text": c["text"], "source": c["source"], "heading": c["heading"]}
        for c in chunks
    ]
    kept, truncated = _pack_items(items, _payload_budget(settings))
    shown = len(chunks)
    head = f"[find_literal] {text!r} 在正文中命中 {total} 处"
    if shown < total:
        head += f"（展示前 {shown} 处）"
    tail = "\n…（结果已按长度上限截断）" if truncated or shown < total else ""
    return (
        f"{head}\n{_render(kept)}{tail}",
        _artifact(
            "find_literal", kept, truncated or shown < total,
            {"total": total, "shown": shown},
        ),
    )


# ======================================================================
# 装配
# ======================================================================

def build_tools(engine: RAGEngine, settings: dict) -> list[BaseTool]:
    """
    把三个取材函数包成 LangChain 工具。

    用工厂闭包而不是模块级 @tool：engine 与 settings 是**每次请求的快照**
    （配置可热重载），做成模块级全局就会出现「配置改了、工具还绑着旧引擎」
    这种撕裂状态。闭包让绑定关系和请求生命周期一致。

    协程包异常：工具内部的意外错误必须变成一句可读的失败文本交回模型，
    向上抛只会让整轮对话失败，而模型本可以换个工具再试。

    三个工具都声明 `response_format="content_and_artifact"`，因此返回
    `(文本, artifact)` 二元组：文本进 prompt，artifact 只留在 ToolMessage 上
    供编排层与评测读取（见 _artifact 的说明）。
    """

    def _guard(name: str, fn):
        async def run(**kwargs):
            try:
                return await fn(engine, settings, **kwargs)
            except Exception as exc:
                logger.exception("工具 %s 执行异常", name)
                return _fail(
                    name, f"工具执行异常：{exc}", "可以换一个工具或调整参数重试"
                )

        return run

    return [
        StructuredTool.from_function(
            coroutine=_guard("search_docs", search_docs),
            name="search_docs",
            args_schema=SearchDocsArgs,
            response_format="content_and_artifact",
            description=(
                "按语义检索知识库，返回最相关的若干文本片段。"
                "绝大多数问题都从这里开始。"
                "每个片段会标注它所属的「文档」与「章节」，"
                "需要完整章节时把这两个值原样传给 expand_section。"
            ),
        ),
        StructuredTool.from_function(
            coroutine=_guard("expand_section", expand_section),
            name="expand_section",
            args_schema=ExpandSectionArgs,
            response_format="content_and_artifact",
            description=(
                "取出某个章节的完整正文。"
                "**只用于扩展 search_docs 已经命中的片段**，"
                "document 与 section 必须抄自上一步检索结果里标注的「文档」和「章节」。"
                "适合「这一节的完整操作步骤是什么」「把上面那段的上下文补全」——"
                "检索只会返回最相似的一两个片段，而步骤是跨片段的，"
                "调大 top_k 也解决不了，因为相邻片段未必更相似。"
            ),
        ),
        StructuredTool.from_function(
            coroutine=_guard("find_literal", find_literal),
            name="find_literal",
            args_schema=FindLiteralArgs,
            response_format="content_and_artifact",
            description=(
                "在正文里做字面串查找，返回命中总数与前若干处位置。"
                "适合「所有出现 X 的地方」这类要求列全的问题——"
                "语义检索只返回最相关的几条且不会提示还有更多。"
                "大小写敏感，不支持通配符或正则；"
                "只匹配正文，按章节名查找请用 search_docs。"
            ),
        ),
    ]
