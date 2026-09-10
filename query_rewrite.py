# -*- coding: utf-8 -*-
"""
多轮对话 Query Rewrite
======================
在检索前把「带指代 / 省略的追问」改写成可独立检索的完整问句。

设计要点：
  - 检索用改写后的 search_query，生成仍用原始 question + history
  - 默认条件触发：首问不调 LLM；短句或含指代词时才改写
  - 改写失败时回退到原始 question，不阻断问答
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)

#: 长问句仅在这些指代词命中时触发
_PRONOUN_PATTERNS: tuple[str, ...] = (
    r"它",
    r"这个",
    r"那个",
    r"这[个些]",
    r"那[个些]",
    r"刚才",
    r"上面",
    r"前面",
    r"还有",
    r"继续",
    r"同样",
    r"也是",
)

#: 自定义模式覆盖时用于短问句 + 长问句
_DEFAULT_TRIGGER_PATTERNS: tuple[str, ...] = _PRONOUN_PATTERNS + (
    r"呢$",
    r"吗$",
    r"默认",
    r"多少",
    r"怎么",
)

_REWRITE_SYSTEM = (
    "你是检索查询改写助手。根据对话历史，把用户的最新问题改写成一条"
    "独立、可检索的完整问题。"
    "不要回答问题，不要解释，只输出改写后的问题文本。"
    "若最新问题本身已完整可检索，原样输出。"
)

_REWRITE_USER = """对话历史：
{history}

最新问题：{question}

改写后的问题："""


def _compile_trigger_patterns(raw: str) -> tuple[re.Pattern[str], ...]:
    text = (raw or "").strip()
    if not text:
        return tuple(re.compile(p) for p in _DEFAULT_TRIGGER_PATTERNS)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(re.compile(p) for p in parts)


def sanitize_history(
    history: Optional[list[dict]], limit: int
) -> list[dict]:
    """只保留合法角色的历史消息，与 rag_engine._build_messages 一致。"""
    if not history or limit <= 0:
        return []
    out: list[dict] = []
    for item in history[-limit:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            out.append({"role": item["role"], "content": str(item["content"])})
    return out


def format_history(history: list[dict]) -> str:
    """把历史格式化为 prompt 中的纯文本块。"""
    lines: list[str] = []
    for item in history:
        role = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{role}：{item['content']}")
    return "\n".join(lines)


def needs_rewrite(question: str, history: Optional[list[dict]], settings: dict) -> bool:
    """
    判断是否应调用 LLM 做 query rewrite。

    query_rewrite_enabled=false 或 mode=off 时恒为 False；
    mode=always 时只要有 history 就改写；默认 conditional 走启发式。
    """
    if not bool(settings.get("query_rewrite_enabled", True)):
        return False

    mode = str(settings.get("query_rewrite_mode", "conditional")).lower()
    if mode == "off":
        return False
    if not sanitize_history(history, 1):
        return False
    if mode == "always":
        return True

    q = question.strip()
    max_len = max(1, int(settings.get("rewrite_trigger_max_len", 12)))
    custom = (str(settings.get("rewrite_trigger_patterns", "")) or "").strip()
    if custom:
        patterns = _compile_trigger_patterns(custom)
        return any(p.search(q) for p in patterns)

    if len(q) <= max_len:
        return True

    pronoun = tuple(re.compile(p) for p in _PRONOUN_PATTERNS)
    return any(p.search(q) for p in pronoun)


def _normalize_rewrite_output(raw: str, fallback: str) -> str:
    text = (raw or "").strip()
    if not text:
        return fallback
    # 去掉常见引号包裹
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'「」":
        text = text[1:-1].strip()
    # 模型偶尔仍带标签前缀
    for prefix in ("改写后的问题：", "改写后的问题:", "问题：", "问题:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text or fallback


async def rewrite_query(
    *,
    client: Optional[AsyncOpenAI],
    chat: Optional[config.Endpoint],
    chat_error: Optional[str],
    settings: dict,
    question: str,
    history: Optional[list[dict]],
) -> str:
    """调用 LLM 改写检索 query；失败时返回原始 question。"""
    if client is None or chat is None:
        raise RuntimeError(f"对话模型不可用: {chat_error or '未配置'}")

    limit = max(0, int(settings.get("rewrite_history_limit", 6)))
    safe_history = sanitize_history(history, limit)
    prompt = _REWRITE_USER.format(
        history=format_history(safe_history),
        question=question.strip(),
    )

    rewrite_model = (settings.get("rewrite_model") or "").strip() or chat.model
    temperature = float(settings.get("rewrite_temperature", 0.0))

    try:
        response = await client.chat.completions.create(
            model=rewrite_model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else ""
        rewritten = _normalize_rewrite_output(content or "", question)
        if rewritten != question:
            logger.info("Query rewrite: %r -> %r", question, rewritten)
        return rewritten
    except Exception:
        logger.exception("Query rewrite 失败，回退到原始问题: %r", question)
        return question


async def resolve_search_query(
    *,
    client: Optional[AsyncOpenAI],
    chat: Optional[config.Endpoint],
    chat_error: Optional[str],
    settings: dict,
    question: str,
    history: Optional[list[dict]] = None,
) -> dict:
    """
    解析本次检索应使用的 query。

    返回:
        search_query: 实际送入 retrieve 的文本
        rewrite_applied: 是否调用了 LLM 改写
        rewrite_skipped_reason: 未改写时的原因（applied 时为 None）
        original_question: 用户原始问题
    """
    original = question.strip()
    if not needs_rewrite(original, history, settings):
        reason = "disabled"
        if bool(settings.get("query_rewrite_enabled", True)):
            mode = str(settings.get("query_rewrite_mode", "conditional")).lower()
            if mode != "off" and sanitize_history(history, 1):
                reason = "heuristic_not_matched"
            elif not sanitize_history(history, 1):
                reason = "no_history"
        return {
            "original_question": original,
            "search_query": original,
            "rewrite_applied": False,
            "rewrite_skipped_reason": reason,
        }

    rewritten = await rewrite_query(
        client=client,
        chat=chat,
        chat_error=chat_error,
        settings=settings,
        question=original,
        history=history,
    )
    return {
        "original_question": original,
        "search_query": rewritten,
        "rewrite_applied": True,
        "rewrite_skipped_reason": None,
    }
