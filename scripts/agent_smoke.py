# -*- coding: utf-8 -*-
"""
针对运行中的服务，测量 Agent 模式的行为与代价。

与其它两个脚本的分工：

    retrieval_sweep.py  只测检索，四种开关组合，不调生成模型
    ragas_eval.py       用 LLM 当裁判评检索与生成质量
    agent_smoke.py      测 agent 的**取材行为与代价**：选了哪个工具、
                        跑了几轮、花了多久、上下文用掉多少

存在的理由是：agentic 每问多 1~3 次 LLM 调用，**收益和代价必须一起报**。
而代价只能对着真实模型测——pytest 里的替身是零延迟的，测不出任何耗时。

用法:
    # 用内置夹具（自带一篇覆盖三个工具的文档，跑完自动删除）
    python scripts/agent_smoke.py --base-url http://127.0.0.1:8001

    # 对着当前知识库跑自己的问题集（不上传任何东西）
    python scripts/agent_smoke.py --questions eval/questions.agent.jsonl

    # 调参前后对照：先存一份，改完 .env 重启再跑一次
    python scripts/agent_smoke.py --json eval/agent-rounds3.json
    python scripts/agent_smoke.py --json eval/agent-rounds1.json \
        --baseline eval/agent-rounds3.json

问题集格式（JSONL）：
    {"question": "...", "expected_tool": "search_docs", "type": "single-hop"}
    expected_tool 可省略，省略的题不计入「工具选择正确率」。

判定口径是「该工具**出现在本次的工具序列里**」，而不是「它是第一个被调用的」。
这一条是跑完第一版才改对的：`expand_section` 按设计不能当首工具——
它必须扩展 `search_docs` 已经命中的片段，所以正确行为本来就是
「先 search_docs，再 expand_section」。按首工具判，这个指标对它
结构性地永远是 0，量出来的是指标自己的缺陷而不是 agent 的行为。

**已知未测项**：每次问答实际消耗的 token。对话端点返回了 usage，
但服务端目前没有把它累计进 stages，所以这里报不出真实用量。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

# Windows 控制台默认是 GBK，输出中的 › 等字符会直接抛 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIXTURE_NAME = "agent冒烟夹具.md"

#: 夹具刻意设计成能把三个工具都逼出来：
#:   - 「故障处理流程」一节远超默认 chunk_size，被切成多块且不可能靠一次
#:     top_k 召回齐全 → 必须 expand_section
#:   - NSR_REFRESH_TIMER 散落在多个章节 → find_literal
#:   - 其余是普通的单跳内容 → search_docs
#:
#: **夹具必须足够大。** 第一版只有 4 个块，而默认 top_k=5——
#: 每次检索都召回了全库，于是 expand_section 结构性地永远用不上，
#: 量出来的「工具选择正确率」是夹具的缺陷而不是 agent 的行为。
#: 这就是 DESIGN 8.6 里「候选池比全库还大」那个教训的第二次发作，
#: 所以这里堆到二十几个块，让 top_k 覆盖不了整个语料。
FIXTURE_TEXT = """# Agent 冒烟夹具

本文档仅用于测量 agent 的取材行为，内容为虚构，不要作为任何操作依据。

## 超时设置

REQUEST_TIMEOUT 默认为 60 秒，控制单次上游调用的最长等待时间。若模型响应经常
超时，应先排查网络链路而不是盲目调大该值，因为超时往往意味着上游服务已经异常。
NSR_REFRESH_TIMER 默认为 2100 秒，决定缓存的刷新周期。两个参数互相独立，
调整任何一个之前都要先确认当前值与预期值是否一致。

## 连接池配置

连接池上限默认为 200，空闲连接的回收时间为 300 秒。连接数打满时的典型表现是
请求排队而不是报错，因此仅看错误率无法发现这类问题，必须同时观察排队时长。
把上限调大之前要先确认下游能承受对应的并发，否则只是把排队从本进程挪到了下游。

## 日志与采集

日志默认按天切分并保留十四天，超过保留期的文件由定时任务清理。采集侧会把
错误级别以上的日志实时推送到集中平台，信息级别的日志只在本地保留。
排查问题时如果只在集中平台上找不到线索，应当登录实例查看本地的完整日志，
因为大量上下文信息在信息级别里，不会被推送出去。

## 监控指标

核心指标包括请求量、错误率、P95 延迟、连接池占用率与缓存命中率五项。
告警阈值按各自的基线设定而不是取统一数值，因为不同服务的正常区间差异很大。
新服务上线后的前两周应当只观察不告警，等基线稳定之后再打开告警开关。

## 故障处理流程

处理线上故障时必须严格按顺序执行以下步骤，跳过任何一步都会导致定位方向偏离，
而定位方向一旦偏了，后续投入的时间基本都是浪费的。

第一步是确认告警的真实性。登录监控面板核对同一时间窗口内是否有其它关联告警
同时触发，单条孤立告警在多数情况下是采集抖动而不是真实故障，贸然操作反而会
引入新的变更风险。同时要确认告警对应的实例是否仍在服务中，已经摘除的实例
继续上报是常见的误报来源。

第二步是采集现场。包括进程状态、最近一小时的日志、当前连接数以及内存占用
快照，必要时还要抓一份线程栈。这一步必须在任何重启动作之前完成，否则现场
一旦丢失就只能等下一次复现，而下一次复现可能是几天之后，也可能就在最忙的
时候再来一遍。采集完成后要把文件归档到共享目录并在事件群里同步路径。

第三步是判断影响范围。确认是单实例问题还是整个集群的问题，方法是对照另外
两个可用区的同名服务指标。如果只有一个可用区异常，则优先考虑该可用区的
基础设施而不是应用本身；如果三个可用区同时异常，则多半是最近一次变更或者
共用的下游依赖出了问题，这时应当立刻去看变更记录。

第四步才是尝试恢复。优先采用流量摘除而不是重启，因为摘除是可逆的而重启会
丢掉现场。摘除之后先观察剩余实例能否承接全部流量，确认承接得住再处理故障
实例。如果剩余容量不足，应当先扩容再摘除，顺序反了会把单点故障放大成
全局不可用。

第五步是恢复之后的复盘。必须在当天内完成，趁记忆还清楚。把 NSR_REFRESH_TIMER
等关键配置的实际取值与预期取值逐项核对，并把差异记录到变更台账里，避免同一个
配置漂移反复引发故障。复盘结论要落到具体的改进项上并指定负责人，只写「加强
监控」这类没有主语的结论等于没有复盘。

## 变更管理

所有配置变更必须先在预发环境验证并留存验证记录。涉及 NSR_REFRESH_TIMER
这类影响缓存行为的参数时，验证时间要覆盖至少一个完整的刷新周期，否则看不出
真实影响。变更窗口避开业务高峰，并且同一时间只做一项变更——同时改两项的话，
出了问题无法判断是哪一项引起的。

## 容量评估

容量按峰值的一点五倍准备，评估依据取最近三个月的实际峰值而不是设计值。
新增功能上线前要单独评估其对连接池与缓存的额外占用，这两项最容易被忽略。

## 备份与恢复

数据每日全量备份一次，增量备份每小时一次，保留三十天。恢复演练每季度至少
一次，演练必须真的执行恢复流程而不是只检查备份文件是否存在——备份可读
和能恢复是两件事，只有真跑过才知道。

## 配置核对

发布前需要逐项核对下列配置：REQUEST_TIMEOUT、NSR_REFRESH_TIMER、连接池上限、
日志保留天数。其中 NSR_REFRESH_TIMER 一旦被改小，缓存刷新会显著变频繁，
下游压力随之上升，这个关联关系在核对时最容易被漏掉。

## 常见误区

最常见的误区是把重启当作首选手段。重启确实经常「有效」，但它同时销毁了
现场，使得同一个问题下次还会以同样的方式出现。第二个误区是只看错误率
不看延迟，很多故障在错误率上完全看不出来。第三个误区是改配置不记录，
台账缺失之后，配置漂移就成了无法追溯的问题。
"""

DEFAULT_QUESTIONS = [
    {"question": "REQUEST_TIMEOUT 默认多少秒？", "expected_tool": "search_docs", "type": "single-hop"},
    {"question": "把「故障处理流程」这一节的完整步骤给我", "expected_tool": "expand_section", "type": "full-section"},
    {"question": "文档里所有出现 NSR_REFRESH_TIMER 的地方都在哪里？", "expected_tool": "find_literal", "type": "enumerate"},
    {"question": "你好，你能做什么？", "expected_tool": None, "type": "chitchat"},
    {"question": "文档里有没有讲 Kubernetes 的部署方式？", "expected_tool": "search_docs", "type": "unanswerable"},
]


def load_questions(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def run_one(client: httpx.Client, question: str) -> dict:
    """
    跑一次问答，记录分段耗时与 stages。

    走 SSE 而不是 /api/agent，因为首 token 延迟只有流式才测得出来，
    而那正是 agent 模式最该被盯住的一个数：用户在看到第一个字之前
    要等模型决策 + 一到两轮取材。

    注意 reset 事件：模型在决定调工具那一轮可能先说一句「我查一下」，
    那段文本不是答案。首 token 时间照常从它算起（用户确实那时看到了字），
    但答案本身要从最后一次 reset 之后重新攒。
    """
    t0 = time.perf_counter()
    record: dict = {
        "question": question,
        "tools_called": [],
        "first_token_s": None,
        "total_s": None,
        "stages": {},
        "error": None,
        "timeline": [],
    }
    answer: list[str] = []

    with client.stream("POST", "/api/chat", json={"question": question}) as stream:
        if stream.status_code != 200:
            stream.read()
            record["error"] = f"HTTP {stream.status_code}"
            return record
        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            elapsed = round(time.perf_counter() - t0, 2)
            kind = obj.get("type")
            if kind == "token" or "content" in obj:
                if record["first_token_s"] is None:
                    record["first_token_s"] = elapsed
                answer.append(obj.get("content", ""))
            elif kind == "error" or "error" in obj:
                record["error"] = obj.get("error")
            else:
                if kind == "reset":
                    # 刚才那段是取材前的开场白，不是答案
                    answer.clear()
                if kind == "tool":
                    record["tools_called"].append(obj["tool"])
                if kind == "done":
                    record["stages"] = obj.get("stages") or {}
                record["timeline"].append((elapsed, kind, obj.get("tool") or ""))

    record["total_s"] = round(time.perf_counter() - t0, 2)
    record["answer"] = "".join(answer)
    return record


def summarise(records: list[dict], questions: list[dict]) -> dict:
    """把逐题记录汇总成 agent 专属的那几个指标（现在测得了的部分）。"""
    ok = [r for r in records if not r["error"]]
    if not ok:
        return {}

    expected = {q["question"]: q.get("expected_tool") for q in questions}
    judged = correct = 0
    for r in ok:
        want = expected.get(r["question"])
        if want is None:
            continue
        judged += 1
        # 看「有没有用到」而不是「是不是第一个用的」：expand_section
        # 按设计必须跟在一次 search_docs 之后，按首工具判它永远不可能命中
        if want in r["tools_called"]:
            correct += 1

    totals = sorted(r["total_s"] for r in ok)
    firsts = sorted(r["first_token_s"] for r in ok if r["first_token_s"] is not None)
    model_calls = [int(r["stages"].get("agent_model_calls", 0)) for r in ok]
    skipped = sum(1 for r in ok if r["stages"].get("agent_retrieval_skipped"))

    # 按题型分组：不同类别的代价可能差一个数量级，混在一起报均值会把它藏起来
    by_type: dict[str, dict] = {}
    for r in ok:
        bucket = by_type.setdefault(
            r.get("type") or "(未分类)", {"n": 0, "totals": [], "calls": []}
        )
        bucket["n"] += 1
        bucket["totals"].append(r["total_s"])
        bucket["calls"].append(int(r["stages"].get("agent_model_calls", 0)))
    per_type = {
        name: {
            "n": b["n"],
            "median_total_s": round(statistics.median(b["totals"]), 1),
            "max_total_s": round(max(b["totals"]), 1),
            "avg_model_calls": round(statistics.mean(b["calls"]), 2),
        }
        for name, b in sorted(by_type.items())
    }

    return {
        "per_type": per_type,
        "n": len(ok),
        "failed": len(records) - len(ok),
        "tool_choice_judged": judged,
        "tool_choice_correct": correct,
        # 模型调用次数就是 agent 相对固定管道的直接代价：固定管道恒为 1
        "avg_model_calls": round(statistics.mean(model_calls), 2) if model_calls else 0,
        "max_model_calls": max(model_calls) if model_calls else 0,
        "avg_tool_calls": round(
            statistics.mean(int(r["stages"].get("agent_tool_calls", 0)) for r in ok), 2
        ),
        "tool_failures": sum(
            int(r["stages"].get("agent_tool_failures", 0)) for r in ok
        ),
        "no_retrieval_ratio": round(skipped / len(ok), 2),
        "median_total_s": statistics.median(totals),
        "max_total_s": totals[-1],
        "median_first_token_s": statistics.median(firsts) if firsts else None,
        "max_first_token_s": firsts[-1] if firsts else None,
    }


def print_summary(summary: dict, baseline: dict | None):
    if not summary:
        print("没有成功的样本，无法汇总。")
        return

    def cell(key: str, fmt="{}"):
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

    judged = summary["tool_choice_judged"]
    accuracy = (
        f"{summary['tool_choice_correct']}/{judged}"
        f"（{summary['tool_choice_correct'] / judged:.0%}）"
        if judged
        else "— 问题集未标注 expected_tool"
    )

    print("\n" + "=" * 68)
    print(f"样本 {summary['n']} 条" + (f"，失败 {summary['failed']} 条" if summary["failed"] else ""))
    print("=" * 68)
    rows = [
        ("工具选择正确率", accuracy),
        (
            "平均模型调用次数",
            cell("avg_model_calls", "{:.2f}") + f"  (最大 {summary['max_model_calls']})",
        ),
        ("平均工具调用次数", cell("avg_tool_calls", "{:.2f}")),
        ("工具失败次数", cell("tool_failures", "{}")),
        ("不检索比例", cell("no_retrieval_ratio", "{:.2f}")),
        ("首 token 中位数 (s)", cell("median_first_token_s", "{:.1f}")),
        ("首 token 最大 (s)", cell("max_first_token_s", "{:.1f}")),
        ("总耗时中位数 (s)", cell("median_total_s", "{:.1f}")),
        ("总耗时最大 (s)", cell("max_total_s", "{:.1f}")),
    ]
    for name, value in rows:
        print(f"  {name:<22} {value}")

    per_type = summary.get("per_type") or {}
    if len(per_type) > 1:
        print("\n  按题型（均值会把类别之间的差距藏起来）")
        print(f"    {'类型':<16}{'条数':>4}{'耗时中位':>10}{'最大':>8}{'平均调用':>10}")
        for name, row in per_type.items():
            print(
                f"    {name:<16}{row['n']:>4}{row['median_total_s']:>10.1f}"
                f"{row['max_total_s']:>8.1f}{row['avg_model_calls']:>10.2f}"
            )

    print(
        "\n注：样本量小，只报中位数与最大值；P95 需要几十条以上才有意义。"
        "\n    模型调用次数是 agent 的直接代价，固定管道恒为 1。"
        "\n    LLM 实际消耗的 token 报不出来——需要服务端累计 usage，尚未埋点。"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", default="")
    parser.add_argument("--questions", default="", help="JSONL 问题集；给定时不上传夹具，直接对当前知识库跑")
    parser.add_argument("--repeat", type=int, default=1, help="每题重复次数，用于看稳态")
    parser.add_argument("--json", default="", help="把逐题记录与汇总写到这个文件")
    parser.add_argument("--baseline", default="", help="对照另一次 --json 的结果")
    parser.add_argument("--verbose", action="store_true", help="打印每题的分段时间线")
    args = parser.parse_args()

    headers = {"X-API-Token": args.token} if args.token else {}
    client = httpx.Client(
        base_url=args.base_url,
        headers=headers,
        timeout=600,
        # 内网地址不能走企业代理，否则请求会被发去代理并返回 HTML 错误页
        trust_env=False,
    )

    # ---- 前置检查：开关没开的话，下面测的全是固定管道 ----
    try:
        health = client.get("/api/health").json()
    except Exception as exc:
        print(f"无法连接 {args.base_url}: {exc}")
        return 2

    # 端点名与依赖错误只在管理面的 detail 里（面向使用者的 health 刻意不报）。
    # 这个脚本本身走 /api/chat，管理面关着也能跑，所以取不到就只少打印两行，
    # 不因此中止。
    try:
        response = client.get("/api/health/detail")
        detail = response.json() if response.status_code == 200 else {}
    except Exception:
        detail = {}

    agent = health.get("agent") or {}
    print(f"服务    : {args.base_url}  status={health.get('status')}")
    print(f"对话端点: {(detail.get('active') or {}).get('llm_model') or '（管理面未启用，不可见）'}")
    print(f"agent   : {json.dumps(agent, ensure_ascii=False)}")
    if not agent.get("enabled"):
        print("\nagent_enabled 为 false —— 现在测的是固定管道，不会有任何 agent 事件。")
        print("请在 .env 里设 AGENT_ENABLED=true 后重启服务。")
        return 2
    if not agent.get("available"):
        error = ((detail.get("agent") or {}).get("error")) or "详情见 /api/health/detail"
        print(f"\nagent 依赖不可用: {error}")
        return 2

    # ---- 问题集 ----
    uploaded = False
    if args.questions:
        questions = load_questions(Path(args.questions))
        print(f"问题集  : {args.questions}（{len(questions)} 题，对当前知识库运行）")
    else:
        questions = DEFAULT_QUESTIONS
        resp = client.post(
            "/api/upload",
            files=[("files", (FIXTURE_NAME, FIXTURE_TEXT.encode("utf-8"), "text/markdown"))],
        )
        result = resp.json()["results"][0]
        if result.get("status") != "success":
            print(f"夹具上传失败: {result}")
            return 2
        uploaded = True
        print(f"问题集  : 内置夹具（{len(questions)} 题，{result['chunks']} 个块）")

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")).get("summary")

    # ---- 跑 ----
    records: list[dict] = []
    try:
        for round_index in range(args.repeat):
            for item in questions:
                record = run_one(client, item["question"])
                record["type"] = item.get("type", "")
                record["expected_tool"] = item.get("expected_tool")
                record["repeat"] = round_index
                records.append(record)

                mark = "!!" if record["error"] else "  "
                tools = "+".join(record["tools_called"]) or "（未检索）"
                print(
                    f"{mark} [{record['total_s']:6.1f}s 首字 "
                    f"{record['first_token_s'] if record['first_token_s'] is not None else '—'}] "
                    f"{tools:<28} {item['question'][:34]}"
                )
                if record["error"]:
                    print(f"     错误: {record['error'][:160]}")
                if args.verbose:
                    for elapsed, kind, label in record["timeline"]:
                        print(f"       [{elapsed:6.1f}s] {kind}:{label}")
    finally:
        if uploaded:
            client.delete(f"/api/documents/{FIXTURE_NAME}")

    summary = summarise(records, questions)
    print_summary(summary, baseline)

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
