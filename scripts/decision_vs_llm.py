# -*- coding: utf-8 -*-
"""
「用决策模型」vs「不用决策模型（全交给 LLM）」端到端对比。

两套方案在**同一份检索结果**上运行，差别只在判断由谁做：

  A. 能否作答
     方案一（现状）：检索 → LLM 直接作答，靠 prompt 里的「资料不足请如实说明」自行拒答
     方案二：检索 → 决策模型判断片段能否作答 → 否则直接拒答，是才调 LLM 生成
     另附：LLM 用与决策模型相同的问法当判断器，单比判断本身的质量与耗时
  B. 是否改写：LLM 判断器 vs 决策模型 vs 现有启发式
  C. 取材方式：现有 Agent（LLM 自选工具，跑完整循环）vs 决策模型首步选择

方案二的生成复用方案一在同一输入上的生成结果——prompt 与检索结果完全相同，
重跑一遍只会多出采样噪声。

用法:
    python scripts/decision_vs_llm.py --json eval/decision_vs_llm_report.json
    python scripts/decision_vs_llm.py --only rewrite --limit 10     # 冒烟
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

import config  # noqa: E402
import vector_store  # noqa: E402
from decision_eval import (  # noqa: E402
    ANSWERABILITY_SETS,
    Q_ANSWERABLE,
    Q_NEEDS_HISTORY,
    Q_TOOL,
    ROOT,
    DecisionClient,
    _passages,
    auroc,
    binary_metrics,
    load_jsonl,
    pct,
)
from query_rewrite import needs_rewrite  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402

GATE_THRESHOLD = 0.3

#: 生成结果里出现这些说法即视为拒答。只看开头一段：
#: 「资料中未提及 X，但……」这种部分作答的句式往往出现在中后部。
REFUSAL = re.compile(
    r"(没有|未|并未|无)(检索到|找到|提及|提到|涉及|包含|给出|说明|记载|相关|关于)"
    r"|无法(回答|根据|从|提供)|不足以回答|知识库中没有|资料中没有|文档中没有"
)
REFUSAL_WINDOW = 150

_THINK = re.compile(r"<think>.*?</think>", re.S)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(_THINK.sub("", answer).strip()[:REFUSAL_WINDOW]))


def q50(xs):
    return statistics.median(xs) if xs else None


def q95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else None


# ======================================================================
# LLM 判断器
# ======================================================================

class LLMJudge:
    def __init__(self, settings: dict, concurrency: int):
        ep = config.resolve_llm(settings)
        self.model = ep.model
        http = httpx.AsyncClient(trust_env=not config.is_internal_host(ep.base_url), timeout=600)
        self.client = AsyncOpenAI(
            base_url=ep.base_url, api_key=ep.api_key, http_client=http, max_retries=1, timeout=600
        )
        self._sem = asyncio.Semaphore(concurrency)
        self.calls: list[dict] = []

    async def yes_no(self, prompt: str) -> tuple[bool | None, float, str]:
        """返回 (判断, 耗时 ms, 原始输出)。解析不出是/否时判断为 None。"""
        async with self._sem:
            t0 = time.perf_counter()
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "你是一个判断器。只回答一个字：是 或 否，不要解释。"},
                    {"role": "user", "content": prompt},
                ],
            )
            ms = (time.perf_counter() - t0) * 1000
        text = _THINK.sub("", resp.choices[0].message.content or "").strip()
        usage = resp.usage
        self.calls.append({
            "ms": ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        })
        head = text[:4]
        verdict = True if "是" in head and "否" not in head else False if "否" in head else None
        return verdict, ms, text[:80]


def judge_summary(calls: list[dict]) -> dict:
    ms = [c["ms"] for c in calls]
    return {
        "calls": len(calls),
        "ms_p50": q50(ms), "ms_p95": q95(ms),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
    }


# ======================================================================
# A + C：需要建索引的部分
# ======================================================================

def _settings_for_eval() -> dict:
    settings = config.load_settings()
    settings.update({
        "hybrid_search_enabled": True, "rerank_enabled": True,
        "rerank_provider": "local", "top_k": 5, "candidate_pool_size": 20,
        "query_rewrite_enabled": False, "agent_session_enabled": False,
    })
    return settings


async def generate(engine: RAGEngine, question: str, hits, sem: asyncio.Semaphore) -> dict:
    """走固定管道的真实组装与生成路径。"""
    state = engine._state
    messages = engine._build_messages(state, question, hits, None)
    usage: dict = {}
    async with sem:
        t0 = time.perf_counter()
        first = None
        pieces = []
        async for piece in engine._stream_chat(state, messages, usage=usage):
            if first is None:
                first = time.perf_counter() - t0
            pieces.append(piece)
        total = time.perf_counter() - t0
    answer = "".join(pieces)
    return {
        "answer": answer, "gen_s": total, "first_token_s": first,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def run_indexed_sets(args, decision: DecisionClient, judge: LLMJudge) -> tuple[list, list]:
    answer_records: list[dict] = []
    tool_records: list[dict] = []
    gen_sem = asyncio.Semaphore(args.concurrency)

    for name, corpus, qpath in ANSWERABILITY_SETS:
        all_q = load_jsonl(qpath)
        questions = [q for q in all_q if q.get("expected_tool", "x") is not None][: args.limit or None]
        tmp = Path(tempfile.mkdtemp(prefix=f"decision-vs-llm-{name}-"))
        try:
            config.QDRANT_URL = ""
            config.QDRANT_PATH = tmp / "qdrant"
            settings = _settings_for_eval()
            engine = RAGEngine(settings=settings)
            for path in sorted(corpus.glob("*.md")):
                await engine.add_document(path.read_text(encoding="utf-8"), path.name)

            if "answerability" in args.only:
                hits_list = []
                for item in questions:
                    t0 = time.perf_counter()
                    hits = (await engine.retrieve_with_diagnostics(item["question"]))["hits"]
                    hits_list.append((hits, time.perf_counter() - t0))
                print(f"[A] {name}: {len(questions)} 题已检索，开始判断与生成")

                async def one(item, hits, retrieve_s):
                    passages = _passages(hits)
                    kws = item.get("expected_keywords") or []
                    answerable = item.get("type") != "unanswerable"
                    ans, d_ms = await decision.ask_timed(
                        {"question": item["question"], "passages": passages}, {"q": Q_ANSWERABLE}
                    )
                    d = ans["q"]["noul"]
                    prompt = (
                        f"question：{item['question']}\n\npassages：\n" + "\n\n".join(passages)
                        + f"\n\n{Q_ANSWERABLE['instructions']}"
                    )
                    j, j_ms, j_raw = await judge.yes_no(prompt)
                    g = await generate(engine, item["question"], hits, gen_sem)
                    rec = {
                        "set": name, "question": item["question"], "type": item.get("type"),
                        "answerable": answerable,
                        "answer_in_passages": answerable and (not kws or any(k in h.text for h in hits for k in kws)),
                        "keyword_in_answer": bool(kws) and any(k in g["answer"] for k in kws),
                        "retrieve_s": retrieve_s,
                        "decision_p": d, "decision_ms": d_ms,
                        "llm_judge": j, "llm_judge_ms": j_ms, "llm_judge_raw": j_raw,
                        "llm_refused": is_refusal(g["answer"]),
                        "answer_head": g["answer"][:200],
                        **{k: g[k] for k in ("gen_s", "first_token_s", "prompt_tokens", "completion_tokens")},
                    }
                    print(f"    · {name} {rec['type']:<12} 决策 {d:.2f} | LLM判 {j} | 生成{'拒答' if rec['llm_refused'] else '作答'} {g['gen_s']:.1f}s  {item['question'][:24]}")
                    return rec

                answer_records += await asyncio.gather(
                    *(one(q, h, s) for q, (h, s) in zip(questions, hits_list))
                )

            if "tool" in args.only and name == "example":
                from agent.runner import AgentRunner

                runner = AgentRunner(engine, settings)
                items = [q for q in all_q if "expected_tool" in q][: args.limit or None]
                print(f"[C] Agent 逐题运行 {len(items)} 题（串行，LLM 自选工具）")
                for item in items:
                    ans, d_ms = await decision.ask_timed({"question": item["question"]}, {"q": Q_TOOL})
                    d = ans["q"]
                    t0 = time.perf_counter()
                    out = await runner.run(item["question"])
                    total = time.perf_counter() - t0
                    st = out["stages"] or {}
                    used = st.get("agent_tools_used") or []
                    expected = item["expected_tool"] or "none"
                    rec = {
                        "question": item["question"], "type": item.get("type"), "expected": expected,
                        "agent_tools": used, "agent_error": out["error"],
                        "agent_correct": (not used) if expected == "none" else expected in used,
                        "agent_first_tool": used[0] if used else "none",
                        "agent_total_s": total,
                        "agent_model_calls": st.get("agent_model_calls"),
                        "agent_tool_calls": st.get("agent_tool_calls"),
                        "agent_model_seconds": st.get("agent_model_seconds"),
                        "agent_prompt_tokens": st.get("agent_prompt_tokens"),
                        "agent_completion_tokens": st.get("agent_completion_tokens"),
                        "agent_refused": is_refusal(out["answer"]),
                        "decision_choice": d["choice"], "decision_confidence": d.get("confidence"),
                        "decision_correct": d["choice"] == expected, "decision_ms": d_ms,
                    }
                    tool_records.append(rec)
                    print(f"    · {rec['type']:<12} 期望 {expected:<14} Agent {'+'.join(used) or 'none':<32} "
                          f"{total:5.1f}s | 决策 {d['choice']} {d.get('confidence', 0):.2f}")
        finally:
            vector_store.close_all()
            shutil.rmtree(tmp, ignore_errors=True)
    return answer_records, tool_records


def summarise_answerability(records: list[dict]) -> dict:
    labels = [r["answerable"] for r in records]
    strategies = {
        "方案一：LLM 直接作答（自行拒答）": [not r["llm_refused"] for r in records],
        "LLM 判断器（同问法）": [r["llm_judge"] is not False for r in records],
        f"方案二：决策模型拦截 p>={GATE_THRESHOLD}": [r["decision_p"] >= GATE_THRESHOLD for r in records],
        "方案二+：决策拦截 且 LLM 自行拒答": [
            r["decision_p"] >= GATE_THRESHOLD and not r["llm_refused"] for r in records
        ],
    }
    table = {k: binary_metrics(labels, v) for k, v in strategies.items()}

    # 端到端耗时：方案一 = 检索 + 生成；方案二 = 检索 + 决策 + （放行时）生成
    s1 = [r["retrieve_s"] + r["gen_s"] for r in records]
    s2 = [r["retrieve_s"] + r["decision_ms"] / 1000 + (r["gen_s"] if r["decision_p"] >= GATE_THRESHOLD else 0)
          for r in records]
    unans = [r for r in records if not r["answerable"]]
    passed = [r for r in records if r["decision_p"] >= GATE_THRESHOLD]
    return {
        "n": len(records), "positives": sum(labels), "table": table,
        "llm_judge_parse_failures": sum(r["llm_judge"] is None for r in records),
        "auroc_decision": auroc([r["decision_p"] for r in records if r["answerable"]],
                                [r["decision_p"] for r in records if not r["answerable"]]),
        "latency": {
            "scheme1_p50_s": q50(s1), "scheme1_p95_s": q95(s1),
            "scheme2_p50_s": q50(s2), "scheme2_p95_s": q95(s2),
            "scheme1_unanswerable_mean_s": statistics.mean(r["retrieve_s"] + r["gen_s"] for r in unans) if unans else None,
            "scheme2_unanswerable_mean_s": statistics.mean(
                r["retrieve_s"] + r["decision_ms"] / 1000 + (r["gen_s"] if r["decision_p"] >= GATE_THRESHOLD else 0)
                for r in unans) if unans else None,
            "decision_ms_p50": q50([r["decision_ms"] for r in records]),
            "decision_ms_p95": q95([r["decision_ms"] for r in records]),
            "llm_judge_ms_p50": q50([r["llm_judge_ms"] for r in records]),
            "llm_judge_ms_p95": q95([r["llm_judge_ms"] for r in records]),
        },
        "llm_calls": {"scheme1": len(records), "scheme2": len(passed)},
        "llm_tokens": {
            "scheme1": sum(r["prompt_tokens"] + r["completion_tokens"] for r in records),
            "scheme2": sum(r["prompt_tokens"] + r["completion_tokens"] for r in passed),
        },
    }


def print_answerability(s: dict) -> None:
    print(f"\n  n={s['n']}（库内 {s['positives']}，库外 {s['n'] - s['positives']}）；"
          f"LLM 判断器解析失败 {s['llm_judge_parse_failures']} 次")
    print(f"    {'策略':<30}{'准确率':>8}{'库外拒答率':>10}{'库内误拒率':>10}")
    for name, b in s["table"].items():
        refuse = b["tn"] / (b["tn"] + b["fp"]) if b["tn"] + b["fp"] else None
        false_rej = b["fn"] / (b["fn"] + b["tp"]) if b["fn"] + b["tp"] else None
        print(f"    {name:<30}{pct(b['accuracy']):>8}{pct(refuse):>10}{pct(false_rej):>10}")
    lat = s["latency"]
    print(f"  端到端耗时 p50/p95：方案一 {lat['scheme1_p50_s']:.1f}s/{lat['scheme1_p95_s']:.1f}s，"
          f"方案二 {lat['scheme2_p50_s']:.1f}s/{lat['scheme2_p95_s']:.1f}s")
    if lat["scheme1_unanswerable_mean_s"] is not None:
        print(f"  库外题平均耗时：方案一 {lat['scheme1_unanswerable_mean_s']:.1f}s，"
              f"方案二 {lat['scheme2_unanswerable_mean_s']:.1f}s")
    print(f"  单次判断耗时 p50/p95：决策模型 {lat['decision_ms_p50']:.0f}/{lat['decision_ms_p95']:.0f}ms，"
          f"LLM 判断器 {lat['llm_judge_ms_p50']:.0f}/{lat['llm_judge_ms_p95']:.0f}ms")
    print(f"  LLM 生成调用：方案一 {s['llm_calls']['scheme1']} 次，方案二 {s['llm_calls']['scheme2']} 次；"
          f"LLM token：{s['llm_tokens']['scheme1']} → {s['llm_tokens']['scheme2']}")


def summarise_tool(records: list[dict]) -> dict:
    def mean(key):
        xs = [r[key] for r in records if r.get(key) is not None]
        return statistics.mean(xs) if xs else None

    by_type: dict[str, dict] = {}
    for r in records:
        t = by_type.setdefault(r["type"], {"n": 0, "agent": 0, "decision": 0, "agent_s": []})
        t["n"] += 1
        t["agent"] += r["agent_correct"]
        t["decision"] += r["decision_correct"]
        t["agent_s"].append(r["agent_total_s"])
    return {
        "n": len(records),
        "agent_accuracy": sum(r["agent_correct"] for r in records) / len(records),
        "decision_accuracy": sum(r["decision_correct"] for r in records) / len(records),
        "agent_total_s_mean": mean("agent_total_s"),
        "agent_model_calls_mean": mean("agent_model_calls"),
        "agent_tool_calls_mean": mean("agent_tool_calls"),
        "agent_errors": sum(bool(r["agent_error"]) for r in records),
        "decision_ms_p50": q50([r["decision_ms"] for r in records]),
        "by_type": {k: {"n": v["n"], "agent": v["agent"] / v["n"], "decision": v["decision"] / v["n"],
                        "agent_s_mean": statistics.mean(v["agent_s"])} for k, v in by_type.items()},
    }


def print_tool(s: dict) -> None:
    print(f"\n  n={s['n']}：工具选择正确率 Agent(LLM) {pct(s['agent_accuracy'])} vs 决策模型 {pct(s['decision_accuracy'])}")
    print(f"  Agent 平均每题 {s['agent_total_s_mean']:.1f}s、模型调用 {s['agent_model_calls_mean']:.1f} 次、"
          f"工具调用 {s['agent_tool_calls_mean']:.1f} 次，运行错误 {s['agent_errors']} 题；"
          f"决策模型单次 p50 {s['decision_ms_p50']:.0f}ms")
    print(f"    {'题型':<14}{'n':>3}{'Agent':>9}{'决策':>9}{'Agent 平均耗时':>16}")
    for k, v in s["by_type"].items():
        print(f"    {k:<14}{v['n']:>3}{pct(v['agent']):>9}{pct(v['decision']):>9}{v['agent_s_mean']:>14.1f}s")


# ======================================================================
# B：是否改写
# ======================================================================

async def run_rewrite(args, decision: DecisionClient, judge: LLMJudge) -> dict:
    settings = config.load_settings()
    settings.update({"query_rewrite_enabled": True, "query_rewrite_mode": "conditional"})
    samples = [{"question": i["question"], "history": i["history"], "label": True}
               for i in load_jsonl(ROOT / "eval/questions.multiturn.example.jsonl")]
    standalone = load_jsonl(ROOT / "eval/questions.hss.jsonl") + [
        q for q in load_jsonl(ROOT / "eval/questions.agent.example.jsonl") if q.get("expected_tool", "x") is not None
    ]
    for i, item in enumerate(standalone):
        prev = standalone[i - 1]
        samples.append({"question": item["question"], "label": False, "history": [
            {"role": "user", "content": prev["question"]},
            {"role": "assistant", "content": prev.get("ground_truth") or "文档中未找到相关内容。"},
        ]})
    if args.limit:
        samples = samples[: args.limit]

    async def one(s):
        ans, d_ms = await decision.ask_timed(
            {"history": s["history"], "question": s["question"]}, {"q": Q_NEEDS_HISTORY}
        )
        d = ans["q"]["noul"]
        hist = "\n".join(f"{t['role']}: {t['content']}" for t in s["history"])
        j, j_ms, _ = await judge.yes_no(f"history：\n{hist}\n\nquestion：{s['question']}\n\n{Q_NEEDS_HISTORY['instructions']}")
        return {"question": s["question"], "label": s["label"],
                "heuristic": needs_rewrite(s["question"], s["history"], settings),
                "decision_p": d, "decision_ms": d_ms, "llm_judge": j, "llm_judge_ms": j_ms}

    records = await asyncio.gather(*(one(s) for s in samples))
    labels = [r["label"] for r in records]
    return {
        "records": records, "n": len(records), "positives": sum(labels),
        "table": {
            "现有启发式": binary_metrics(labels, [r["heuristic"] for r in records]),
            "LLM 判断器": binary_metrics(labels, [bool(r["llm_judge"]) for r in records]),
            "决策模型 p>=0.7": binary_metrics(labels, [r["decision_p"] >= 0.7 for r in records]),
        },
        "llm_judge_parse_failures": sum(r["llm_judge"] is None for r in records),
        "decision_ms_p50": q50([r["decision_ms"] for r in records]),
        "llm_judge_ms_p50": q50([r["llm_judge_ms"] for r in records]),
        "llm_judge_ms_p95": q95([r["llm_judge_ms"] for r in records]),
    }


def print_rewrite(s: dict) -> None:
    print(f"\n  n={s['n']}（需改写 {s['positives']}）；LLM 判断器解析失败 {s['llm_judge_parse_failures']} 次")
    print(f"    {'策略':<16}{'召回':>8}{'精确率':>8}{'误触发率':>10}{'误触发次数':>10}")
    for name, b in s["table"].items():
        fpr = b["fp"] / (b["fp"] + b["tn"]) if b["fp"] + b["tn"] else None
        print(f"    {name:<16}{pct(b['recall']):>8}{pct(b['precision']):>8}{pct(fpr):>10}{b['fp']:>10}")
    print(f"  单次判断耗时 p50：决策模型 {s['decision_ms_p50']:.0f}ms，LLM 判断器 {s['llm_judge_ms_p50']:.0f}ms"
          f"（p95 {s['llm_judge_ms_p95']:.0f}ms）")


# ======================================================================

async def main_async(args) -> int:
    base_url = os.getenv("DECISION_BASE_URL", "").strip()
    api_key = os.getenv("DECISION_API_KEY", "").strip()
    if not base_url or not api_key:
        print("需要在 .env 配置 DECISION_BASE_URL 与 DECISION_API_KEY")
        return 1
    args.only = set(args.only.split(",")) if args.only else {"answerability", "rewrite", "tool"}

    decision = DecisionClient(base_url, api_key, os.getenv("DECISION_MODEL", "decision-model-preview"))
    judge = LLMJudge(config.load_settings(), args.concurrency)
    print(f"LLM: {judge.model}；决策模型: {decision.model}")
    report: dict = {"llm": judge.model, "decision_model": decision.model, "gate_threshold": GATE_THRESHOLD}
    out = Path(args.json) if args.json else None

    def dump():
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    try:
        if "rewrite" in args.only:
            print("\n=== B. 是否改写 ===")
            report["rewrite"] = await run_rewrite(args, decision, judge)
            print_rewrite(report["rewrite"])
            dump()
        if args.only & {"answerability", "tool"}:
            print("\n=== A. 能否作答 / C. 取材方式 ===")
            answer_records, tool_records = await run_indexed_sets(args, decision, judge)
            if answer_records:
                report["answerability"] = {"records": answer_records, **summarise_answerability(answer_records)}
                print("\n--- A. 能否作答 ---")
                print_answerability(report["answerability"])
            if tool_records:
                report["tool"] = {"records": tool_records, **summarise_tool(tool_records)}
                print("\n--- C. 取材方式 ---")
                print_tool(report["tool"])
            dump()
    finally:
        await decision.close()

    report["llm_judge_calls"] = judge_summary(judge.calls)
    dump()
    if out:
        print(f"\n已写入 {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="决策模型 vs 纯 LLM 方案对比")
    parser.add_argument("--only", default="", help="answerability,rewrite,tool 中的若干个")
    parser.add_argument("--json", default="")
    parser.add_argument("--limit", type=int, default=0, help="每组只跑前 N 题，冒烟用")
    parser.add_argument("--concurrency", type=int, default=2, help="对 LLM 的并发数")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
