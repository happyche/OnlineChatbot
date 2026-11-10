# -*- coding: utf-8 -*-
"""
决策模型（System One 协议）离线评测：不调 LLM，只比「判断」本身。

三组对照，每组都与项目里现有的做法比较：

  A. 能否作答：重排最高分阈值（agent.tools._confidence_note 的依据）
     vs 决策模型对「这些片段能否回答问题」的 noul 概率
  B. 是否改写：query_rewrite.needs_rewrite 的启发式
     vs 决策模型对「是否指代前文」的 noul 概率
  C. 首步取材方式：决策模型的 choice，对照题集标注的 expected_tool

检索部分完全复用 RAGEngine（本地嵌入 + 混合检索 + 本地重排），在临时目录
现建索引，不动现有向量库。

用法:
    python scripts/decision_eval.py                       # 三组全跑
    python scripts/decision_eval.py --only answerability
    python scripts/decision_eval.py --json eval/decision_eval_report.json

需要 .env 中的 DECISION_BASE_URL / DECISION_API_KEY / DECISION_MODEL。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx  # noqa: E402

import config  # noqa: E402
import vector_store  # noqa: E402
from query_rewrite import needs_rewrite  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ANSWERABILITY_SETS = [
    ("hss", ROOT / "eval/corpus", ROOT / "eval/questions.hss.jsonl"),
    ("example", ROOT / "eval/corpus.example", ROOT / "eval/questions.agent.example.jsonl"),
]

Q_ANSWERABLE = {
    "type": "noul",
    "instructions": "passages 中的内容是否包含回答 question 所需的具体信息？"
    "只是主题相关但没有给出答案的，判为否。",
}
REWRITE_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)

Q_NEEDS_HISTORY = {
    "type": "noul",
    "instructions": "question 里有没有指代前文的内容（如“它”“那个”“上一步”“还有别的吗”），"
    "必须结合 history 才能理解？",
}
Q_TOOL = {
    "type": "choice",
    "instructions": "回答这个问题，第一步应该用哪种方式从知识库取材？",
    "criteria": {
        "search_docs": "语义检索：用自然语言描述的一般问题",
        "expand_section": "读取某个章节的完整原文：要求完整步骤、整个流程、全部内容的问题",
        "find_literal": "按字面串精确查找：要找出某个命令、参数名、错误码出现在哪些地方",
        "none": "不需要查知识库：寒暄、闲聊、问助手自身",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


# ======================================================================
# 决策模型客户端
# ======================================================================

class DecisionClient:
    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int = 4):
        self.url = base_url.rstrip("/") + "/systemone"
        self.model = model
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30
        )
        self._sem = asyncio.Semaphore(concurrency)
        self.calls: list[dict] = []

    async def ask(self, state, questions: dict) -> dict:
        return (await self.ask_timed(state, questions))[0]

    async def ask_timed(self, state, questions: dict) -> tuple[dict, float]:
        """返回 (answers, 请求耗时 ms)。耗时不含并发排队，只算最后一次成功请求。"""
        body = {"model": self.model, "state": state, "questions": questions}
        async with self._sem:
            for attempt in range(3):
                t0 = time.perf_counter()
                try:
                    resp = await self._client.post(self.url, json=body)
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    if attempt == 2:
                        self.calls.append({"ok": False, "error": str(exc)[:200]})
                        raise
                    await asyncio.sleep(1 + attempt)
                    continue
                data = resp.json()
                wall_ms = (time.perf_counter() - t0) * 1000
                self.calls.append({
                    "ok": True,
                    "wall_ms": wall_ms,
                    "model_ms": data.get("latency_ms"),
                    "input_tokens": (data.get("usage") or {}).get("input_tokens", 0),
                })
                return data["answers"], wall_ms

    async def close(self):
        await self._client.aclose()


# ======================================================================
# 指标
# ======================================================================

def auroc(pos: list[float], neg: list[float]) -> float | None:
    """正类得分高于负类的概率；与阈值无关，用来公平比较两种不同量纲的分数。"""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def binary_metrics(labels: list[bool], preds: list[bool]) -> dict:
    tp = sum(l and p for l, p in zip(labels, preds))
    fp = sum((not l) and p for l, p in zip(labels, preds))
    fn = sum(l and (not p) for l, p in zip(labels, preds))
    tn = sum((not l) and (not p) for l, p in zip(labels, preds))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "precision": prec, "recall": rec, "f1": f1,
    }


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


# ======================================================================
# A. 能否作答
# ======================================================================

def _passages(hits) -> list[str]:
    return [f"【{h.source} › {h.heading}】{h.text}" if h.heading else f"【{h.source}】{h.text}" for h in hits]


async def eval_answerability(client: DecisionClient, rerank_threshold: float) -> dict:
    records: list[dict] = []
    for name, corpus, qpath in ANSWERABILITY_SETS:
        questions = [q for q in load_jsonl(qpath) if q.get("expected_tool", "x") is not None]
        tmp = Path(tempfile.mkdtemp(prefix=f"decision-eval-{name}-"))
        try:
            config.QDRANT_URL = ""
            config.QDRANT_PATH = tmp / "qdrant"
            settings = config.load_settings()
            settings.update({
                "hybrid_search_enabled": True, "rerank_enabled": True,
                "rerank_provider": "local", "top_k": 5, "candidate_pool_size": 20,
                "query_rewrite_enabled": False,
            })
            engine = RAGEngine(settings=settings)
            for path in sorted(corpus.glob("*.md")):
                await engine.add_document(path.read_text(encoding="utf-8"), path.name)
            print(f"[A] {name}: 已建临时索引，{len(questions)} 题")

            retrieved_hits = []
            for item in questions:
                retrieved_hits.append((await engine.retrieve_with_diagnostics(item["question"]))["hits"])

            async def one(item, hits):
                kws = item.get("expected_keywords") or []
                answerable = item.get("type") != "unanswerable"
                retrieved = answerable and (not kws or any(k in h.text for h in hits for k in kws))
                t0 = time.perf_counter()
                ans = await client.ask(
                    {"question": item["question"], "passages": _passages(hits)},
                    {"q": Q_ANSWERABLE},
                )
                return {
                    "set": name,
                    "question": item["question"],
                    "type": item.get("type"),
                    "answerable": answerable,
                    "answer_in_passages": retrieved,
                    "rerank_top": hits[0].rerank_score if hits else None,
                    "decision_p": ans["q"]["noul"],
                    "decision_wall_ms": (time.perf_counter() - t0) * 1000,
                }

            records += await asyncio.gather(*(one(q, h) for q, h in zip(questions, retrieved_hits)))
        finally:
            vector_store.close_all()
            shutil.rmtree(tmp, ignore_errors=True)

    def evaluate(label_key: str) -> dict:
        labels = [r[label_key] for r in records]
        rr = [r["rerank_top"] if r["rerank_top"] is not None else -99 for r in records]
        dp = [r["decision_p"] for r in records]
        out = {
            "n": len(records), "positives": sum(labels),
            "auroc_rerank": auroc([s for s, l in zip(rr, labels) if l], [s for s, l in zip(rr, labels) if not l]),
            "auroc_decision": auroc([s for s, l in zip(dp, labels) if l], [s for s, l in zip(dp, labels) if not l]),
            "no_gate": binary_metrics(labels, [True] * len(labels)),
            f"rerank>={rerank_threshold:g}": binary_metrics(labels, [s >= rerank_threshold for s in rr]),
        }
        for t in (0.3, 0.5, 0.7):
            out[f"decision>={t}"] = binary_metrics(labels, [p >= t for p in dp])
        return out

    return {
        "records": records,
        # 口径一：题目本身在库里有没有答案
        "by_question_label": evaluate("answerable"),
        # 口径二：取回的片段里实际有没有答案（检索漏召时拒答也算对）
        "by_retrieval_label": evaluate("answer_in_passages"),
    }


def print_answerability(res: dict, rerank_threshold: float) -> None:
    for title, key in (("题目口径（库内题=可答）", "by_question_label"),
                       ("片段口径（片段含答案=可答）", "by_retrieval_label")):
        m = res[key]
        print(f"\n  {title}: n={m['n']}，可答 {m['positives']}，不可答 {m['n'] - m['positives']}")
        print(f"    AUROC  重排分 {m['auroc_rerank']:.3f}  |  决策模型 {m['auroc_decision']:.3f}")
        print(f"    {'策略':<22}{'准确率':>8}{'拒答召回':>10}{'误拒率':>10}")
        for name, b in m.items():
            if not isinstance(b, dict):
                continue
            refuse_recall = b["tn"] / (b["tn"] + b["fp"]) if b["tn"] + b["fp"] else None
            false_reject = b["fn"] / (b["fn"] + b["tp"]) if b["fn"] + b["tp"] else None
            print(f"    {name:<22}{pct(b['accuracy']):>8}{pct(refuse_recall):>10}{pct(false_reject):>10}")
    print("\n  不可答题明细（重排分 / 决策概率）:")
    for r in res["records"]:
        if not r["answerable"]:
            print(f"    {r['rerank_top'] if r['rerank_top'] is not None else float('nan'):>7.2f} / {r['decision_p']:.2f}  {r['question'][:40]}")
    print("\n  可答但决策概率 < 0.5 的题（潜在误拒）:")
    for r in res["records"]:
        if r["answerable"] and r["decision_p"] < 0.5:
            mark = "片段含答案" if r["answer_in_passages"] else "检索已漏召"
            print(f"    {r['rerank_top']:>7.2f} / {r['decision_p']:.2f}  [{mark}] {r['question'][:40]}")


# ======================================================================
# B. 是否改写
# ======================================================================

async def eval_rewrite(client: DecisionClient) -> dict:
    settings = config.load_settings()
    settings.update({"query_rewrite_enabled": True, "query_rewrite_mode": "conditional"})
    samples: list[dict] = []
    for item in load_jsonl(ROOT / "eval/questions.multiturn.example.jsonl"):
        samples.append({"question": item["question"], "history": item["history"], "label": True})
    # 负例：本身完整的问题，配上一轮与之无关的历史，模拟「换了个话题继续问」
    standalone = [q for q in load_jsonl(ROOT / "eval/questions.hss.jsonl")]
    standalone += [q for q in load_jsonl(ROOT / "eval/questions.agent.example.jsonl")
                   if q.get("expected_tool", "x") is not None]
    for i, item in enumerate(standalone):
        prev = standalone[i - 1]
        history = [
            {"role": "user", "content": prev["question"]},
            {"role": "assistant", "content": prev.get("ground_truth") or "文档中未找到相关内容。"},
        ]
        samples.append({"question": item["question"], "history": history, "label": False})

    async def one(s):
        ans = await client.ask(
            {"history": s["history"], "question": s["question"]}, {"q": Q_NEEDS_HISTORY}
        )
        s = dict(s)
        s["heuristic"] = needs_rewrite(s["question"], s["history"], settings)
        s["decision_p"] = ans["q"]["noul"]
        return s

    records = await asyncio.gather(*(one(s) for s in samples))
    labels = [r["label"] for r in records]
    out = {
        "records": [{k: v for k, v in r.items() if k != "history"} for r in records],
        "n": len(records), "positives": sum(labels),
        "heuristic": binary_metrics(labels, [r["heuristic"] for r in records]),
    }
    for t in REWRITE_THRESHOLDS:
        out[f"decision>={t}"] = binary_metrics(labels, [r["decision_p"] >= t for r in records])
    return {
        **out,
        "auroc_decision": auroc([r["decision_p"] for r in records if r["label"]],
                                [r["decision_p"] for r in records if not r["label"]]),
    }


def print_rewrite(res: dict) -> None:
    print(f"\n  n={res['n']}，需改写 {res['positives']}，不需改写 {res['n'] - res['positives']}"
          f"（决策模型 AUROC {res['auroc_decision']:.3f}）")
    print(f"    {'策略':<16}{'召回':>8}{'精确率':>8}{'误触发率':>10}{'多出的改写调用':>14}")
    for name in ("heuristic", *(f"decision>={t}" for t in REWRITE_THRESHOLDS)):
        b = res[name]
        fpr = b["fp"] / (b["fp"] + b["tn"]) if b["fp"] + b["tn"] else None
        print(f"    {name:<16}{pct(b['recall']):>8}{pct(b['precision']):>8}{pct(fpr):>10}{b['fp']:>14}")
    print("\n  需改写样本的决策概率: " + "，".join(
        f"{r['decision_p']:.2f} {r['question']}" for r in res["records"] if r["label"]))
    print("  启发式与决策模型（门槛 0.8）判断不一致的样本:")
    for r in res["records"]:
        if r["heuristic"] != (r["decision_p"] >= 0.8):
            print(f"    标注={'改写' if r['label'] else '不改'} 启发式={'改写' if r['heuristic'] else '不改'} "
                  f"决策={r['decision_p']:.2f}  {r['question'][:40]}")


# ======================================================================
# C. 首步取材方式
# ======================================================================

async def eval_tool(client: DecisionClient) -> dict:
    items = [q for q in load_jsonl(ROOT / "eval/questions.agent.example.jsonl") if "expected_tool" in q]

    async def one(item):
        ans = (await client.ask({"question": item["question"]}, {"q": Q_TOOL}))["q"]
        expected = item["expected_tool"] or "none"
        return {
            "question": item["question"], "type": item.get("type"),
            "expected": expected, "choice": ans["choice"],
            "confidence": ans.get("confidence"), "probabilities": ans.get("probabilities"),
            "correct": ans["choice"] == expected,
        }

    records = await asyncio.gather(*(one(q) for q in items))
    per_type: dict[str, list[bool]] = {}
    for r in records:
        per_type.setdefault(r["type"], []).append(r["correct"])
    buckets = {}
    for t in (0.0, 0.3, 0.5, 0.7):
        sel = [r for r in records if (r["confidence"] or 0) >= t]
        buckets[f"conf>={t}"] = {
            "coverage": len(sel) / len(records),
            "accuracy": sum(r["correct"] for r in sel) / len(sel) if sel else None,
        }
    return {
        "records": records,
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "per_type": {k: sum(v) / len(v) for k, v in per_type.items()},
        "confidence_buckets": buckets,
    }


def print_tool(res: dict) -> None:
    print(f"\n  总体正确率 {pct(res['accuracy'])}（n={len(res['records'])}）")
    print("  按题型: " + "，".join(f"{k} {pct(v)}" for k, v in res["per_type"].items()))
    print("  按把握度分档（只采纳把握度 ≥ 门槛的判断，其余交给 LLM）:")
    for k, b in res["confidence_buckets"].items():
        print(f"    {k:<10} 覆盖 {pct(b['coverage']):>7}  正确率 {pct(b['accuracy']):>7}")
    print("  错例:")
    for r in res["records"]:
        if not r["correct"]:
            print(f"    期望 {r['expected']:<15} 选了 {r['choice']:<15} 把握 {r['confidence']:.2f}  {r['question'][:36]}")


# ======================================================================

def latency_summary(calls: list[dict]) -> dict:
    ok = [c for c in calls if c["ok"]]
    wall = sorted(c["wall_ms"] for c in ok)
    model = sorted(c["model_ms"] for c in ok if c["model_ms"] is not None)

    def q(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    return {
        "calls": len(calls), "failed": len(calls) - len(ok),
        "input_tokens": sum(c["input_tokens"] for c in ok),
        "model_ms_p50": q(model, 0.5), "model_ms_p95": q(model, 0.95), "model_ms_max": model[-1] if model else None,
        "wall_ms_p50": q(wall, 0.5), "wall_ms_p95": q(wall, 0.95),
        "model_ms_mean": statistics.mean(model) if model else None,
    }


async def main_async(args) -> int:
    base_url = os.getenv("DECISION_BASE_URL", "").strip()
    api_key = os.getenv("DECISION_API_KEY", "").strip()
    model = os.getenv("DECISION_MODEL", "decision-model-preview").strip()
    if not base_url or not api_key:
        print("需要在 .env 配置 DECISION_BASE_URL 与 DECISION_API_KEY")
        return 1

    client = DecisionClient(base_url, api_key, model, concurrency=args.concurrency)
    report: dict = {"model": model}
    only = set(args.only.split(",")) if args.only else {"answerability", "rewrite", "tool"}
    threshold = float(config.load_settings().get("agent_low_confidence_score", 1.0))
    try:
        if "answerability" in only:
            print("\n=== A. 能否作答 ===")
            report["answerability"] = await eval_answerability(client, threshold)
            print_answerability(report["answerability"], threshold)
        if "rewrite" in only:
            print("\n=== B. 是否改写 ===")
            report["rewrite"] = await eval_rewrite(client)
            print_rewrite(report["rewrite"])
        if "tool" in only:
            print("\n=== C. 首步取材方式 ===")
            report["tool"] = await eval_tool(client)
            print_tool(report["tool"])
    finally:
        await client.close()

    report["latency"] = latency_summary(client.calls)
    lat = report["latency"]
    print(f"\n=== 调用开销 ===\n  调用 {lat['calls']} 次，失败 {lat['failed']}，输入 token 合计 {lat['input_tokens']}")
    print(f"  模型耗时 p50 {lat['model_ms_p50']:.0f}ms / p95 {lat['model_ms_p95']:.0f}ms / 最大 {lat['model_ms_max']:.0f}ms")
    print(f"  端到端（含公司代理） p50 {lat['wall_ms_p50']:.0f}ms / p95 {lat['wall_ms_p95']:.0f}ms")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="决策模型离线评测（不调 LLM）")
    parser.add_argument("--only", default="", help="answerability,rewrite,tool 中的若干个")
    parser.add_argument("--json", default="", help="把逐题记录与汇总写到这个文件")
    parser.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
