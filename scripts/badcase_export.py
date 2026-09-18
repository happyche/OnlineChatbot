# -*- coding: utf-8 -*-
"""
把线上差评导出成回归题集。

**这个脚本补的是整套评测设施里唯一缺的那一环。** 项目已经有 RAGAS 四指标、
确定性路由指标、消融开关和对照基线，但**输入这些评测的题目全部由人手写**。
于是「线上答错了」和「回归集里多一条用例」之间隔着一个人的记性。

    用户点 👎  →  feedback.db  →  本脚本  →  questions.jsonl  →  既有评测链路

产出的格式就是 `eval/questions.*.jsonl`（见 design_LangChainAgent.md 12.7），
因此 `agent_eval.py` / `ragas_eval.py` / `validate_questions.py` 一行都不用改。

**导出的题目是半成品，这是有意的。** `ground_truth` 一律留空，因为它只能由人
判断——如果能自动生成正确答案，那就不需要这个系统了。`validate_questions.py`
会把每一条都报成「缺 ground_truth」，那个列表就是待标注清单。

每条记录额外带一个 `_badcase` 块（当时的答案、取回的资料、用户填的原因、
耗时与用量）。标注的人需要它：光看问题无法判断该标什么，而「当时取回了什么
资料」直接决定这是检索问题还是生成问题——两者的修法完全不同。
下划线前缀表示它不参与评测，只给人看。

用法:
    # 导出全部差评
    python scripts/badcase_export.py -o eval/questions.badcase.jsonl

    # 只要「答案不全」这一类，并且只要走了 agent 的
    python scripts/badcase_export.py --reason incomplete --route agent

    # 追加进已有题集（去重按问题正文）
    python scripts/badcase_export.py -o eval/questions.regression.jsonl --append

    # 看看都攒了些什么，先不导
    python scripts/badcase_export.py --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
import feedback  # noqa: E402


def _to_question(row: dict) -> dict:
    """
    把一条 badcase 翻成题集里的一行。

    expected_sources 取自「当时实际召回的文档」，这是一个**猜测**而不是标注：
    答错的时候召回的很可能正是错的来源。留着它是因为大多数差评的问题在生成
    而不在检索，此时这个猜测是对的；标注的人看到 `_badcase.contexts_used`
    就能判断要不要改。宁可给一个待核对的初值，也不要让每条都从空白开始。
    """
    contexts = row.get("contexts") or []
    sources: list[str] = []
    for ctx in contexts:
        source = ctx.get("source")
        if source and source not in sources:
            sources.append(source)

    stages = row.get("stages") or {}
    reason = row.get("reason") or "unspecified"
    return {
        "question": row.get("question", ""),
        # 题型带上原因，这样 agent_eval.py 的「按题型分组」直接就能按失败模式看：
        # 均值会把类别之间的差距藏起来，而不同原因对应的修法完全不同。
        "type": f"badcase:{reason}",
        "expected_sources": sources,
        # ground_truth / expected_keywords / expected_tool 一律不写。
        # 尤其是 expected_tool：省略字段表示「不参与路由统计」，
        # 而写成 null 会被当成「不该取材」，把正确率算虚高（12.7 的那个坑）。
        "_badcase": {
            "trace_id": row.get("trace_id"),
            "created_at": row.get("created_at"),
            "route": row.get("route"),
            "model": row.get("model"),
            "verdict": row.get("verdict"),
            "reason": reason,
            "reason_label": feedback.REASONS.get(reason, reason),
            "comment": row.get("comment"),
            "answer": row.get("answer"),
            "latency_ms": row.get("latency_ms"),
            "total_tokens": (row.get("prompt_tokens") or 0)
            + (row.get("completion_tokens") or 0),
            "tokens_source": row.get("tokens_source"),
            "stages": stages,
            "contexts_used": contexts,
        },
    }


def _load_existing(path: Path) -> tuple[list[dict], set[str]]:
    """读已有题集，返回 (原有行, 已有问题正文集合)。"""
    if not path.exists():
        return [], set()
    rows: list[dict] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        rows.append(item)
        seen.add((item.get("question") or "").strip())
    return rows, seen


def main() -> int:
    parser = argparse.ArgumentParser(description="把线上差评导出成回归题集")
    parser.add_argument("-o", "--output", help="输出 jsonl 路径")
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加到已有题集（按问题正文去重），而不是覆盖",
    )
    parser.add_argument("--verdict", default="down", choices=list(feedback.VERDICTS))
    parser.add_argument(
        "--reason", help=f"只要这一类原因，可选: {', '.join(feedback.REASONS)}"
    )
    parser.add_argument("--route", help="只要这条路径的记录（agent / pipeline）")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument(
        "--summary", action="store_true", help="只打印统计，不写文件"
    )
    args = parser.parse_args()

    if args.reason and args.reason not in feedback.REASONS:
        print(f"未知的 reason: {args.reason}（可选 {', '.join(feedback.REASONS)}）")
        return 2
    if not args.summary and not args.output:
        print("需要 -o/--output，或者用 --summary 只看统计")
        return 2

    if not config.FEEDBACK_DB.exists():
        print(f"反馈库不存在: {config.FEEDBACK_DB}\n（FEEDBACK_ENABLED 是否开着？服务跑过吗？）")
        return 1

    store = feedback.FeedbackStore(config.FEEDBACK_DB)
    try:
        stats = store.stats()
        rows = store.badcases(limit=args.limit, verdict=args.verdict)
    finally:
        store.close()

    print(f"反馈库: {config.FEEDBACK_DB}")
    print(f"交互记录 {stats['interactions']} 条｜👍 {stats['up']}｜👎 {stats['down']}", end="")
    satisfaction = stats["satisfaction"]
    # 没有反馈时满意度是「未知」而不是 0，两者的含义差别很大
    print(f"｜满意度 {satisfaction:.1%}" if satisfaction is not None else "｜满意度 —")

    if stats["down_reasons"]:
        print("\n差评原因分布（总数没有行动含义，分布才有）:")
        for reason, count in sorted(
            stats["down_reasons"].items(), key=lambda kv: -kv[1]
        ):
            label = feedback.REASONS.get(reason, reason)
            print(f"  {label:<22} {count}")

    if stats["by_route"]:
        print("\n分路径的代价（收益和代价必须一起看）:")
        print(f"  {'路径':<10}{'请求':>8}{'平均延迟':>12}{'平均token':>12}{'累计费用':>14}")
        for item in stats["by_route"]:
            print(
                f"  {item['route']:<10}{item['requests']:>8}"
                f"{item['avg_latency_ms']:>10.0f}ms"
                f"{item['avg_tokens']:>12.0f}"
                f"{item['total_cost']:>12.4f} {item['currency']}"
            )

    if args.reason:
        rows = [r for r in rows if (r.get("reason") or "") == args.reason]
    if args.route:
        rows = [r for r in rows if (r.get("route") or "") == args.route]

    print(f"\n符合筛选条件的 badcase: {len(rows)} 条")
    if args.summary:
        return 0
    if not rows:
        print("没有可导出的记录，未写文件。")
        return 0

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing, seen = _load_existing(path) if args.append else ([], set())

    added = 0
    skipped = 0
    out = list(existing)
    for row in rows:
        question = (row.get("question") or "").strip()
        if not question or question in seen:
            skipped += 1
            continue
        seen.add(question)
        out.append(_to_question(row))
        added += 1

    with path.open("w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"写入 {path}：新增 {added} 条，跳过重复 {skipped} 条，共 {len(out)} 条")
    print(
        "\n下一步：\n"
        f"  1) 人工补 ground_truth（_badcase.answer 和 _badcase.contexts_used 是判断依据）\n"
        f"  2) python scripts/validate_questions.py {path} --corpus eval/corpus\n"
        f"     —— 它会把每条都报成「缺 ground_truth」，那个列表就是待标注清单\n"
        f"  3) python scripts/agent_eval.py --questions {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
