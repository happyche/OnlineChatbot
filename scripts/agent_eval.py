# -*- coding: utf-8 -*-
"""
评测 Agent 的**准确性**：路由对不对、取回的资料够不够、答案靠不靠谱。

与另外三个评测脚本的分工：

    retrieval_sweep.py  只测固定管道的检索，四种开关组合，不调生成模型
    ragas_eval.py       用 LLM 当裁判评质量；默认走**固定管道**（见下）
    agent_smoke.py      测 agent 的**代价**：耗时、首 token、调了几次模型
    agent_eval.py       测 agent 的**准确性**：路由指标 + 导出 RAGAS 样本

**为什么需要这个脚本**：`ragas_eval.py` 直接构造 RAGEngine 并调 `engine.answer()`，
那是固定管道。即使 `AGENT_ENABLED=true`，它测的也**不是** agent——没有报错、
没有警告，数字看起来完全正常，但评的是另一条路径。这是个静默陷阱。

这个脚本走 `POST /api/agent`，拿到答案 + agent 实际看到的 contexts + 事件轨迹，
然后做两件事：

**一、直接算路由指标**（不需要 LLM 评委，确定性的，秒出）：

    工具选择正确率    expected_tool 是否出现在本次的工具序列里
    资料来源命中率    expected_sources 是否出现在取回的 contexts 里
    关键词覆盖率      expected_keywords 是否出现在答案里
    不检索比例        闲聊类问题有没有省掉检索
    空手作答率        没取到任何资料却给出了肯定回答（幻觉高危信号）

**二、导出 RAGAS 样本**，字段直接对齐 ragas.SingleTurnSample，
因此可以原样喂给现有的 `ragas_eval.py --samples`，那边一行都不用改：

    python scripts/agent_eval.py  --questions eval/questions.agent.example.jsonl \
                                  --dump eval/agent_samples.jsonl
    python scripts/ragas_eval.py  --samples eval/agent_samples.jsonl

**关于工具选择正确率的判定口径**：看「该工具**出现在本次的工具序列里**」，
而不是「它是第一个被调用的」。`expand_section` 按设计不能当首工具——
它必须扩展 `search_docs` 已经命中的片段，正确行为本来就是「先检索、再扩展」。
按首工具判，这个指标对它结构性地永远是 0，量出来的是指标自己的缺陷。

用法:
    # 前提：服务已启动且 knowledge base 里有对应语料
    python scripts/agent_eval.py --questions eval/questions.agent.example.jsonl

    # 导出 RAGAS 样本，接着用现有评委打分
    python scripts/agent_eval.py --questions eval/questions.hss.jsonl \
        --dump eval/agent_samples.jsonl

    # 调参前后对照
    python scripts/agent_eval.py --json eval/agent_before.json
    python scripts/agent_eval.py --json eval/agent_after.json \
        --baseline eval/agent_before.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import httpx

# Windows 控制台默认是 GBK，输出中的 › 等字符会直接抛 UnicodeEncodeError。
#
# line_buffering 同样不能省：输出重定向到文件时 Python 默认块缓冲，
# 而这个脚本单题就要跑一到几分钟。没有逐行刷新的话，一次二十分钟的评测
# 在日志里是完全空白的，和卡死无从区分。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

#: 答案里出现这些说法，视为「模型明确表示查不到」。
#: 用于区分「没取到资料但老实拒答」（正确）和「没取到资料却硬答」（幻觉）。
_REFUSAL_MARKERS = (
    "没有相关",
    "未找到",
    "找不到",
    "知识库中没有",
    "没有提到",
    "无法回答",
    "没有查到",
    "不包含",
)


def load_questions(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def run_one(client: httpx.Client, item: dict) -> dict:
    """
    跑一道题，返回逐题记录。

    走 /api/agent 而不是 SSE：这里要的是答案与 contexts，不测首 token 延迟
    （那是 agent_smoke.py 的职责）。而且这个端点不受 agent_enabled 约束，
    评测要能在开关关闭时照样跑对照组。
    """
    question = item["question"]
    record: dict = {
        "question": question,
        "type": item.get("type", ""),
        # 「没标注」与「明确标成 null」是两回事：前者不参与判定，
        # 后者是一条断言——这道题**不该取材**。混在一起会把所有
        # 没标注的题都算成「不该取材」，正确率瞬间虚高。
        "has_expected_tool": "expected_tool" in item,
        "expected_tool": item.get("expected_tool"),
        "expected_sources": item.get("expected_sources") or [],
        "expected_keywords": item.get("expected_keywords") or [],
        "ground_truth": item.get("ground_truth", ""),
        "answer": "",
        "contexts": [],
        "sources": [],
        "tools_called": [],
        "stages": {},
        "error": None,
    }

    try:
        response = client.post("/api/agent", json={"question": question})
    except Exception as exc:
        record["error"] = f"请求失败: {exc}"
        return record

    if response.status_code != 200:
        record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
        return record

    body = response.json()
    contexts = body.get("contexts") or []
    record["answer"] = body.get("answer", "")
    record["contexts"] = contexts
    record["sources"] = sorted({c.get("source", "") for c in contexts if c.get("source")})
    record["stages"] = body.get("stages") or {}
    record["tools_called"] = list(record["stages"].get("agent_tools_used") or [])
    return record


def judge(record: dict) -> dict:
    """
    对一条记录做确定性判定。**不调 LLM**，所以可复现、零成本、秒出。

    这几项覆盖的是 RAGAS 看不到的失败模式。最典型的是「选错工具」：
    用 search_docs 去回答「列出所有 X」，答案里每句话都有出处，
    faithfulness 会给高分——它只是漏了一半内容。
    """
    verdict: dict = {}

    if record["has_expected_tool"]:
        want_tool = record["expected_tool"]
        if want_tool is None:
            # 明确标成 null = 这道题不该取材（闲聊、元问题）
            verdict["tool_ok"] = not record["tools_called"]
        else:
            # 口径见模块 docstring：看有没有用到，不看是不是第一个用的
            verdict["tool_ok"] = want_tool in record["tools_called"]

    if record["expected_sources"]:
        got = set(record["sources"])
        verdict["source_ok"] = bool(got & set(record["expected_sources"]))

    if record["expected_keywords"]:
        answer = record["answer"]
        hit = sum(1 for kw in record["expected_keywords"] if kw in answer)
        verdict["keyword_ratio"] = hit / len(record["expected_keywords"])

    # 空手作答：没取到任何资料，答案却不是拒答。这是幻觉的高危信号，
    # 但不能直接判错——闲聊类问题本来就该空手回答。
    if not record["contexts"] and record["tools_called"]:
        answer = record["answer"]
        verdict["answered_empty_handed"] = not any(
            marker in answer for marker in _REFUSAL_MARKERS
        )

    return verdict


def summarise(records: list[dict]) -> dict:
    ok = [r for r in records if not r["error"]]
    if not ok:
        return {}

    def ratio(key: str) -> tuple[int, int]:
        judged = [r for r in ok if key in r["verdict"]]
        passed = sum(1 for r in judged if r["verdict"][key])
        return passed, len(judged)

    tool_pass, tool_total = ratio("tool_ok")
    source_pass, source_total = ratio("source_ok")
    empty_hand_pass, empty_hand_total = ratio("answered_empty_handed")

    keyword_scores = [
        r["verdict"]["keyword_ratio"] for r in ok if "keyword_ratio" in r["verdict"]
    ]
    model_calls = [int(r["stages"].get("agent_model_calls", 0)) for r in ok]
    context_counts = [len(r["contexts"]) for r in ok]
    skipped = sum(1 for r in ok if r["stages"].get("agent_retrieval_skipped"))

    # 按题型分组：不同类别的路由难度差别很大，混在一起报均值会把它藏起来
    by_type: dict[str, dict] = {}
    for r in ok:
        bucket = by_type.setdefault(
            r.get("type") or "(未分类)", {"n": 0, "tool_pass": 0, "tool_total": 0}
        )
        bucket["n"] += 1
        if "tool_ok" in r["verdict"]:
            bucket["tool_total"] += 1
            bucket["tool_pass"] += int(r["verdict"]["tool_ok"])

    return {
        "n": len(ok),
        "failed": len(records) - len(ok),
        "tool_pass": tool_pass,
        "tool_total": tool_total,
        "source_pass": source_pass,
        "source_total": source_total,
        "keyword_coverage": (
            round(statistics.mean(keyword_scores), 3) if keyword_scores else None
        ),
        "answered_empty_handed": empty_hand_pass,
        "answered_empty_handed_total": empty_hand_total,
        "no_retrieval_ratio": round(skipped / len(ok), 2),
        "avg_model_calls": round(statistics.mean(model_calls), 2) if model_calls else 0,
        "avg_contexts": round(statistics.mean(context_counts), 1),
        "zero_context_runs": sum(1 for c in context_counts if c == 0),
        "tool_failures": sum(
            int(r["stages"].get("agent_tool_failures", 0)) for r in ok
        ),
        "per_type": {
            name: {
                "n": b["n"],
                "tool_accuracy": (
                    round(b["tool_pass"] / b["tool_total"], 2)
                    if b["tool_total"]
                    else None
                ),
            }
            for name, b in sorted(by_type.items())
        },
    }


def print_summary(summary: dict, baseline: dict | None):
    if not summary:
        print("没有成功的样本，无法汇总。")
        return

    def frac(passed_key: str, total_key: str) -> str:
        total = summary.get(total_key) or 0
        if not total:
            return "— 问题集未标注对应字段"
        passed = summary[passed_key]
        text = f"{passed}/{total}（{passed / total:.0%}）"
        if baseline and baseline.get(total_key):
            was = baseline[passed_key] / baseline[total_key]
            text += f"  (基线 {was:.0%})"
        return text

    def cell(key: str, fmt="{}") -> str:
        now = summary.get(key)
        if now is None:
            return "—"
        text = fmt.format(now)
        if baseline and baseline.get(key) is not None:
            was = baseline[key]
            if isinstance(now, (int, float)) and isinstance(was, (int, float)):
                delta = now - was
                if abs(delta) > 1e-9:
                    text += f"  ({fmt.format(was)} → {delta:+.2f})"
        return text

    print("\n" + "=" * 72)
    print(
        f"样本 {summary['n']} 条"
        + (f"，失败 {summary['failed']} 条" if summary["failed"] else "")
    )
    print("=" * 72)
    print("\n  【路由】agent 特有的失败模式，RAGAS 测不到")
    print(f"    {'工具选择正确率':<20} {frac('tool_pass', 'tool_total')}")
    print(f"    {'不检索比例':<20} {cell('no_retrieval_ratio', '{:.2f}')}")
    print(f"    {'平均模型调用次数':<20} {cell('avg_model_calls', '{:.2f}')}")
    print(f"    {'工具失败次数':<20} {cell('tool_failures', '{}')}")

    print("\n  【取材】资料对不对，不需要 LLM 评委")
    print(f"    {'来源命中率':<20} {frac('source_pass', 'source_total')}")
    print(f"    {'平均 contexts 条数':<20} {cell('avg_contexts', '{:.1f}')}")
    print(f"    {'零资料轮次':<20} {cell('zero_context_runs', '{}')}")

    print("\n  【答案】粗筛，细评交给 RAGAS")
    coverage = summary.get("keyword_coverage")
    print(
        f"    {'关键词覆盖率':<20} "
        + (f"{coverage:.0%}" if coverage is not None else "— 问题集未标注 expected_keywords")
    )
    total_eh = summary.get("answered_empty_handed_total") or 0
    print(
        f"    {'空手作答（幻觉高危）':<20} "
        + (
            f"{summary['answered_empty_handed']}/{total_eh}"
            if total_eh
            else "— 没有出现取材为空的轮次"
        )
    )

    per_type = summary.get("per_type") or {}
    if len(per_type) > 1:
        print("\n  按题型（均值会把类别之间的差距藏起来）")
        print(f"    {'类型':<18}{'条数':>5}{'工具选择正确率':>16}")
        for name, row in per_type.items():
            acc = row["tool_accuracy"]
            print(
                f"    {name:<18}{row['n']:>5}"
                + (f"{acc:>16.0%}" if acc is not None else f"{'—':>16}")
            )

    print(
        "\n  注：以上全部是确定性判定，不调 LLM 评委，可复现。"
        "\n      忠实度 / 答案相关性等需要评委的指标，用 --dump 导出后交给"
        "\n      ragas_eval.py --samples 打分。"
    )


def dump_ragas_samples(records: list[dict], path: Path) -> int:
    """
    导出成 ragas.SingleTurnSample 的字段名，可直接喂给 ragas_eval.py --samples。

    retrieved_contexts 只取正文，不带「文档: X ｜ 章节: Y」标注——
    这样和 ragas_eval.py 里固定管道那条路径（`[h.text for h in hits]`）
    格式一致，「agent vs 固定管道」才能用同一套指标横向比较。
    """
    lines: list[str] = []
    written = 0
    for r in records:
        if r["error"]:
            continue
        sample = {
            "combo": "agent",
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": [c["text"] for c in r["contexts"]],
            "citations": [
                f"{c.get('source', '')} › {c.get('heading', '') or '（无标题正文）'}"
                for c in r["contexts"]
            ],
            "stages": r["stages"],
        }
        if r["ground_truth"]:
            sample["reference"] = r["ground_truth"]
        lines.append(json.dumps(sample, ensure_ascii=False))
        written += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="评测 Agent 的准确性与路由决策")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--questions",
        default="eval/questions.agent.example.jsonl",
        help="JSONL 问题集。字段见 eval/questions.agent.example.jsonl 的头部注释",
    )
    parser.add_argument("--dump", default="", help="导出 RAGAS 样本到这个文件")
    parser.add_argument("--json", default="", help="把逐题记录与汇总写到这个文件")
    parser.add_argument("--baseline", default="", help="对照另一次 --json 的结果")
    parser.add_argument("--verbose", action="store_true", help="打印每题的判定明细")
    args = parser.parse_args()

    headers = {"X-API-Token": args.token} if args.token else {}
    client = httpx.Client(
        base_url=args.base_url,
        headers=headers,
        timeout=900,
        # 内网地址不能走企业代理，否则请求会被发去代理并返回 HTML 错误页
        trust_env=False,
    )

    # ---- 前置检查 ----
    # 走 /api/health/detail 而不是 /api/health：评测要的端点名与依赖错误都在那边
    # （面向使用者的 health 刻意不报这些）。它和 /api/agent 同属管理面，
    # 所以这一个请求同时验证了「服务在不在」和「管理面开没开」。
    try:
        response = client.get("/api/health/detail")
    except Exception as exc:
        print(f"无法连接 {args.base_url}: {exc}")
        return 2

    if response.status_code == 404:
        print(
            f"\n{args.base_url} 的管理面未启用（ADMIN_ENABLED=false），"
            "/api/agent 不存在。\n"
            "评测用的调试端点属管理面，请用 ADMIN_ENABLED=true 重启服务后再跑。"
        )
        return 2
    if response.status_code != 200:
        print(f"\n健康检查失败: HTTP {response.status_code}: {response.text[:200]}")
        return 2

    health = response.json()
    agent_info = health.get("agent") or {}
    print(f"服务    : {args.base_url}  status={health.get('status')}")
    print(f"对话端点: {(health.get('active') or {}).get('llm_model')}")
    print(f"agent   : {json.dumps(agent_info, ensure_ascii=False)}")
    if not agent_info.get("available"):
        print(f"\nagent 依赖不可用: {agent_info.get('error')}")
        return 2
    if not agent_info.get("enabled"):
        # /api/agent 不受 AGENT_ENABLED 约束，所以这里只是提醒，不中止
        print("\n提示：AGENT_ENABLED=false，但 /api/agent 仍会走 agent 路径。")

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"\n问题集不存在: {questions_path}")
        return 2
    questions = load_questions(questions_path)
    print(f"问题集  : {questions_path}（{len(questions)} 题）\n")

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")).get(
            "summary"
        )

    # ---- 跑 ----
    records: list[dict] = []
    for i, item in enumerate(questions, 1):
        record = run_one(client, item)
        record["verdict"] = judge(record)
        records.append(record)

        if record["error"]:
            print(f"!! [{i}/{len(questions)}] {item['question'][:36]}")
            print(f"     错误: {record['error'][:160]}")
            continue

        marks = []
        if "tool_ok" in record["verdict"]:
            marks.append("工具 " + ("✓" if record["verdict"]["tool_ok"] else "✗"))
        if "source_ok" in record["verdict"]:
            marks.append("来源 " + ("✓" if record["verdict"]["source_ok"] else "✗"))
        tools = "+".join(record["tools_called"]) or "（未检索）"
        print(
            f"   [{i}/{len(questions)}] {tools:<32} "
            f"ctx={len(record['contexts']):<3} {'  '.join(marks):<16} "
            f"{item['question'][:30]}"
        )
        if args.verbose:
            print(f"       期望工具={record['expected_tool']} 实际={record['tools_called']}")
            print(f"       期望来源={record['expected_sources']} 实际={record['sources']}")
            print(f"       答案: {record['answer'][:120]}")

    summary = summarise(records)
    print_summary(summary, baseline)

    if args.dump:
        written = dump_ragas_samples(records, Path(args.dump))
        print(f"\n已导出 {written} 条 RAGAS 样本 → {args.dump}")
        print("  下一步（现有评委脚本一行都不用改）：")
        print(f"    python scripts/ragas_eval.py --samples {args.dump}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"summary": summary, "records": records}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\n已写入 {out}")

    return 0 if summary and not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
