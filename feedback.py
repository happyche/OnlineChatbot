# -*- coding: utf-8 -*-
"""
在线反馈与交互留痕
==================
存在的意义是补上这个项目唯一缺失的闭环：评测能力已经齐备（RAGAS + 确定性指标
+ 消融开关），但**输入评测的题目全部由人手写**。于是「线上答错了」和
「回归集里多一条用例」之间隔着一个人的记性。

这一层把两件事落盘：

  interactions  每一次问答的完整现场（问题、答案、取回的资料、用量、耗时）
  feedback      用户对某个 trace_id 的评价（👍 / 👎 + 原因）

两张表而不是一张：反馈是**后来**才到的，而且可能不到（绝大多数请求没人评价）。
合成一张表意味着每条请求都要预留一组空字段，还得靠 UPDATE 去改——
而 UPDATE 一条不存在的行是静默失败。

**为什么用 SQLite 而不是直接写 JSONL**：badcase 的主要用法是「把最近所有
👎 的、属于 enumerate 题型的、走了 agent 路径的记录捞出来」。这是查询，
不是追加。JSONL 要靠 grep 拼凑，而 P1 阶段迁到 Postgres 时这张表的
schema 可以整体搬过去。

**为什么不用 ORM**：两张表、五条语句。装一个 SQLAlchemy 只会让「这个项目
有几个依赖」这个问题更难回答。

⚠️ **这个文件会把提问、答案与文档原文写进磁盘。** 这和 AGENT_TRACE 是同一类
决定——留痕本身就是暴露面。它默认开启，因为没有反馈就没有闭环；但要留意
FEEDBACK_MAX_ROWS 是一道必须存在的闸（日志要滚动、会话要 TTL，同理）。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

#: 允许的评价取值。收窄成两个值而不是 1-5 星：星级评分在小样本上几乎不可用
#: （所有人都点 4 星），而 badcase 库只需要「这条要不要进回归集」这一个比特。
VERDICTS = ("up", "down")

#: 👎 的预置原因。做成固定选项而不是纯自由文本，是为了让它可以直接聚合——
#: 「本月 62% 的差评是答案不全」这种结论无法从自由文本里算出来。
#: 每一项都对应一类已知的失败模式（见 design_LangChainAgent.md 1.1）。
REASONS = {
    "incomplete": "答案不全（漏了内容）",
    "wrong": "答案错误",
    "no_answer": "明明有资料却说查不到",
    "bad_citation": "引用来源不对",
    "slow": "太慢",
    "other": "其他",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    trace_id         TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    route            TEXT NOT NULL,
    status           TEXT NOT NULL,
    session_id       TEXT,
    model            TEXT,
    provider         TEXT,
    question         TEXT NOT NULL,
    answer           TEXT,
    contexts_json    TEXT,
    stages_json      TEXT,
    prompt_tokens    INTEGER,
    completion_tokens INTEGER,
    tokens_source    TEXT,
    cost             REAL,
    currency         TEXT,
    latency_ms       REAL,
    first_token_ms   REAL,
    error            TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    reason     TEXT,
    comment    TEXT,
    UNIQUE(trace_id) ON CONFLICT REPLACE
);
CREATE INDEX IF NOT EXISTS idx_feedback_verdict ON feedback(verdict);
"""


def _now() -> str:
    """UTC ISO8601。带时区是硬要求：跨时区排查时裸的本地时间无法对齐。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FeedbackStore:
    """
    SQLite 存储。

    连接用 `check_same_thread=False` 并配一把互斥锁：FastAPI 的同步端点跑在
    线程池里，异步端点跑在事件循环上，同一个连接会被不同线程碰到。
    用锁而不是每次新建连接，是因为 WAL 下建连接的开销远大于这几条语句本身。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL：写入不阻塞读取。留痕绝对不能让问答变慢，
        # 否则这一层就从「观测手段」变成了「故障来源」。
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- 写入

    def record_interaction(
        self, record: dict, *, answer: str = "", contexts: Optional[list] = None
    ) -> None:
        """
        落一次问答的现场。record 来自 observability.RequestRecord.as_dict()。

        **失败只记日志，绝不上抛**：留痕失败不是问答失败。磁盘满了的时候
        用户应该照样能得到答案，而不是收到一个关于 SQLite 的报错。
        """
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO interactions (
                        trace_id, created_at, route, status, session_id,
                        model, provider, question, answer, contexts_json,
                        stages_json, prompt_tokens, completion_tokens,
                        tokens_source, cost, currency, latency_ms,
                        first_token_ms, error
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.get("trace_id"),
                        _now(),
                        record.get("route", ""),
                        record.get("status", ""),
                        record.get("session_id"),
                        record.get("model"),
                        record.get("provider"),
                        record.get("question", ""),
                        answer,
                        json.dumps(contexts or [], ensure_ascii=False),
                        json.dumps(
                            {
                                k: v
                                for k, v in record.items()
                                if k.startswith("agent_") or k.endswith("_ms")
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                        record.get("prompt_tokens"),
                        record.get("completion_tokens"),
                        record.get("tokens_source"),
                        record.get("cost"),
                        record.get("currency"),
                        record.get("latency_ms"),
                        record.get("first_token_ms"),
                        record.get("error"),
                    ),
                )
                self._conn.commit()
            self._prune()
        except Exception:
            logger.exception("写入交互记录失败（不影响问答）")

    def record_feedback(
        self,
        trace_id: str,
        verdict: str,
        reason: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> dict:
        """
        记录一条评价，返回 {"route": ...} 供调用方打指标。

        同一个 trace_id 反复提交按最后一次算（schema 里的 ON CONFLICT REPLACE）：
        用户点错了要能改，而不是攒出两条互相矛盾的记录。

        这里的异常**要**上抛：反馈接口的失败必须让用户看见，否则他会以为
        点评价生效了。这和 record_interaction 的取舍正好相反——
        前者是旁路留痕，后者是用户的主动动作。
        """
        if verdict not in VERDICTS:
            raise ValueError(f"verdict 必须是 {VERDICTS} 之一")
        if reason and reason not in REASONS:
            raise ValueError(f"未知的 reason: {reason}")

        with self._lock:
            row = self._conn.execute(
                "SELECT route FROM interactions WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            self._conn.execute(
                """
                INSERT INTO feedback (trace_id, created_at, verdict, reason, comment)
                VALUES (?,?,?,?,?)
                """,
                (trace_id, _now(), verdict, reason, (comment or "").strip()[:2000] or None),
            )
            self._conn.commit()
        # 未知 trace_id 照样入库而不是报 404：反馈可能比留痕先到（留痕在
        # 流结束后才写），把它丢掉等于惩罚点得快的用户。route 留空即可。
        return {"route": (row["route"] if row else "") or "unknown"}

    def _prune(self) -> None:
        """
        超出 FEEDBACK_MAX_ROWS 就删最老的、且没有人评价过的记录。

        「没有人评价过」是关键条件：被点过 👎 的记录正是这张表存在的理由，
        按时间无条件淘汰会把最有价值的数据先删掉。
        """
        limit = config.FEEDBACK_MAX_ROWS
        if limit <= 0:
            return
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()
            if total <= limit:
                return
            self._conn.execute(
                """
                DELETE FROM interactions WHERE trace_id IN (
                    SELECT i.trace_id FROM interactions i
                    LEFT JOIN feedback f ON f.trace_id = i.trace_id
                    WHERE f.trace_id IS NULL
                    ORDER BY i.created_at ASC
                    LIMIT ?
                )
                """,
                (total - limit,),
            )
            self._conn.commit()

    # ---------------------------------------------------------------- 读取

    def stats(self) -> dict:
        """
        汇总：用于 UI 上的一个小角标、也用于 README 里那张「上线后」的截图。

        差评按原因分组，因为总数没有行动含义——「38 条差评」不指向任何动作，
        「其中 24 条是答案不全」直接指向 expand_section 的路由准确率。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT verdict, COUNT(*) n FROM feedback GROUP BY verdict"
            ).fetchall()
            by_verdict = {r["verdict"]: r["n"] for r in rows}
            reasons = {
                r["reason"]: r["n"]
                for r in self._conn.execute(
                    """
                    SELECT reason, COUNT(*) n FROM feedback
                    WHERE verdict = 'down' AND reason IS NOT NULL
                    GROUP BY reason
                    """
                ).fetchall()
            }
            (interactions,) = self._conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()
            agg = self._conn.execute(
                """
                SELECT route,
                       COUNT(*) n,
                       AVG(latency_ms) avg_latency_ms,
                       AVG(total_cost) avg_cost,
                       SUM(total_cost) total_cost,
                       AVG(prompt_tokens + completion_tokens) avg_tokens
                FROM (SELECT route, latency_ms, prompt_tokens, completion_tokens,
                             COALESCE(cost, 0) total_cost
                      FROM interactions WHERE status = 'ok')
                GROUP BY route
                """
            ).fetchall()

        up = by_verdict.get("up", 0)
        down = by_verdict.get("down", 0)
        return {
            "interactions": interactions,
            "up": up,
            "down": down,
            # 没有反馈时满意度是「未知」而不是 0——两者的含义差别很大
            "satisfaction": round(up / (up + down), 3) if (up + down) else None,
            "down_reasons": reasons,
            "by_route": [
                {
                    "route": r["route"],
                    "requests": r["n"],
                    "avg_latency_ms": round(r["avg_latency_ms"] or 0, 1),
                    "avg_tokens": round(r["avg_tokens"] or 0, 1),
                    "avg_cost": round(r["avg_cost"] or 0, 6),
                    "total_cost": round(r["total_cost"] or 0, 6),
                    "currency": config.COST_CURRENCY,
                }
                for r in agg
            ],
        }

    def badcases(self, limit: int = 200, verdict: str = "down") -> list[dict]:
        """
        导出差评现场。供 scripts/badcase_export.py 转成回归题集。

        contexts 一并带出：一条 badcase 缺了「当时取回了什么资料」就无法判断
        这是检索问题还是生成问题，而这两者的修法完全不同。
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT i.*, f.verdict, f.reason, f.comment, f.created_at fb_at
                FROM feedback f JOIN interactions i ON i.trace_id = f.trace_id
                WHERE f.verdict = ?
                ORDER BY f.created_at DESC
                LIMIT ?
                """,
                (verdict, limit),
            ).fetchall()

        out = []
        for row in rows:
            item = dict(row)
            item["contexts"] = json.loads(item.pop("contexts_json") or "[]")
            item["stages"] = json.loads(item.pop("stages_json") or "{}")
            out.append(item)
        return out


# ======================================================================
# 进程级单例
# ======================================================================

_STORE: Optional[FeedbackStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> Optional[FeedbackStore]:
    """
    取存储；未启用时返回 None。

    惰性初始化而不是导入时建库：config.FEEDBACK_DB 在测试里会被 monkeypatch
    指向临时目录，导入时就建库会在开发者的项目根下留一个真实的 .db 文件。
    """
    global _STORE
    if not config.FEEDBACK_ENABLED:
        return None
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                try:
                    _STORE = FeedbackStore(config.FEEDBACK_DB)
                    logger.info("反馈库已就绪: %s", config.FEEDBACK_DB)
                except Exception:
                    logger.exception("反馈库初始化失败，反馈功能不可用")
                    return None
    return _STORE


def reset_store() -> None:
    """丢弃单例。供测试隔离，以及配置变更后重新指向新路径。"""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            try:
                _STORE.close()
            except Exception:
                pass
        _STORE = None
