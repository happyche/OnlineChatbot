# -*- coding: utf-8 -*-
"""
[RAGAS] 对比两个评委在同一批样本上的打分。

整个脚本都是为 RAGAS 评测新增的，不参与服务运行。

为什么需要它：
    LLM-as-judge 的可信度取决于评委本身，而评委的水平没法直接测量。
    一个可行的间接办法是找两个**互相独立**的评委评同一批样本，看它们是否一致。
    一致不能证明两个都对（可能同时错），但不一致足以证明至少有一个不可信，
    这时就不该拿单一评委的绝对分数下结论。

    这里给三个层次的信息：
      1. 均值差   —— 两个评委的整体宽严差异（系统性偏移）
      2. 排序一致 —— Spearman 秩相关。评测真正关心的是「A 配置是否优于 B」，
                     即便绝对分不同，只要排序一致，用于横向比较的结论仍成立
      3. 分歧样本 —— 差得最多的几条，人工看一眼就知道是谁在乱评

用法:
    python scripts/ragas_compare.py eval/scored_qwen.jsonl eval/scored_cursor.jsonl

前提是两份文件来自**同一批样本**（用 ragas_eval.py 的 --samples 保证），
否则比出来的是答案差异而不是评委差异。脚本会校验这一点。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_scored(path: Path) -> tuple[str, dict[tuple[str, str], dict]]:
    """
    读入一份打过分的结果，按 (combo, question) 建索引。

    返回 (评委名, {键: 各指标分数})。用 combo+question 而不是行号做键，
    是为了在两份文件顺序不同时也能正确配对。
    """
    judge = path.stem
    scores: dict[tuple[str, str], dict] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "scores" not in row:
            raise ValueError(f"{path} 不是打过分的结果（缺 scores 字段）")
        judge = row.get("judge") or judge
        key = (row.get("combo", ""), row["user_input"])
        scores[key] = row["scores"]

    if not scores:
        raise ValueError(f"{path} 中没有任何记录")
    return judge, scores


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """
    Spearman 秩相关系数。

    不用 Pearson 是因为两个评委的分数尺度可能完全不同
    （一个偏严一个偏松），而我们关心的是名次是否一致而非数值是否接近。
    自己实现是为了不给评测脚本再加一个 scipy 依赖。

    样本不足 3 条、或某一侧分数全相同（秩无变化，方差为零）时返回 None——
    这两种情况下相关系数没有意义，硬算会得到误导性的数字。
    """
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        i = 0
        while i < n:
            # 并列值取平均秩，否则并列多时相关系数会被高估
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                result[order[k]] = avg
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def compare(
    left: tuple[str, dict], right: tuple[str, dict], top_n: int
) -> str:
    """产出对比报告。"""
    (name_a, scores_a), (name_b, scores_b) = left, right

    shared = sorted(set(scores_a) & set(scores_b))
    only_a = len(scores_a) - len(shared)
    only_b = len(scores_b) - len(shared)

    lines: list[str] = []
    lines.append(f"评委 A: {name_a}（{len(scores_a)} 条）")
    lines.append(f"评委 B: {name_b}（{len(scores_b)} 条）")
    lines.append(f"可配对样本: {len(shared)} 条")
    if only_a or only_b:
        # 配对不上通常意味着两份结果不是同一批样本，结论会失真
        lines.append(f"  ⚠ A 独有 {only_a} 条、B 独有 {only_b} 条 —— "
                     f"两份结果可能不是同一批样本，请确认都用了 --samples 同一个文件")
    if not shared:
        lines.append("\n没有可配对的样本，无法对比。")
        return "\n".join(lines)

    metrics = sorted(
        {m for key in shared for m in (*scores_a[key], *scores_b[key])}
    )

    lines.append("")
    lines.append("| 指标 | 有效样本 | A 均值 | B 均值 | 均值差(B−A) | 平均绝对差 | 秩相关 |")
    lines.append("|---|---|---|---|---|---|---|")

    disagreements: list[tuple[float, str, tuple[str, str], float, float]] = []

    for metric in metrics:
        pairs = []
        for key in shared:
            va = scores_a[key].get(metric, {}).get("value")
            vb = scores_b[key].get(metric, {}).get("value")
            if va is not None and vb is not None:
                pairs.append((key, float(va), float(vb)))

        if not pairs:
            lines.append(f"| {metric} | 0 | — | — | — | — | — |")
            continue

        xs = [p[1] for p in pairs]
        ys = [p[2] for p in pairs]
        mean_a = sum(xs) / len(xs)
        mean_b = sum(ys) / len(ys)
        mad = sum(abs(b - a) for a, b in zip(xs, ys)) / len(xs)
        rho = spearman(xs, ys)

        lines.append(
            f"| {metric} | {len(pairs)} | {mean_a:.3f} | {mean_b:.3f} | "
            f"{mean_b - mean_a:+.3f} | {mad:.3f} | "
            f"{'—' if rho is None else format(rho, '+.3f')} |"
        )

        for key, va, vb in pairs:
            disagreements.append((abs(vb - va), metric, key, va, vb))

    lines.append("")
    lines.append("怎么读：")
    lines.append("  均值差   两个评委的宽严差异。系统性偏移不影响横向比较，")
    lines.append("           只要你始终用同一个评委。")
    lines.append("  秩相关   接近 +1 表示两个评委对样本好坏的排序一致，")
    lines.append("           此时即使绝对分不同，用来比较配置优劣的结论仍然可信；")
    lines.append("           接近 0 说明至少有一个评委在乱评，别用单一评委下结论。")
    lines.append("           为 — 表示样本太少或分数全相同，算不出有意义的相关性。")

    disagreements.sort(reverse=True)
    top = [d for d in disagreements if d[0] > 0][:top_n]
    if top:
        lines.append("")
        lines.append(f"分歧最大的 {len(top)} 条（人工看一眼就知道是谁的问题）：")
        for diff, metric, (combo, question), va, vb in top:
            lines.append(f"  {metric:<20} A={va:.3f}  B={vb:.3f}  (差 {diff:.3f})")
            lines.append(f"    [{combo}] {question}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对比两个评委在同一批样本上的 RAGAS 打分"
    )
    parser.add_argument("left", help="第一份打分结果 JSONL")
    parser.add_argument("right", help="第二份打分结果 JSONL")
    parser.add_argument("--top", type=int, default=5, help="列出分歧最大的前几条")
    parser.add_argument("--output", default="", help="同时写入 Markdown 文件")
    args = parser.parse_args()

    report = compare(
        load_scored(Path(args.left)),
        load_scored(Path(args.right)),
        args.top,
    )
    print(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# 评委交叉对比\n\n```\n{report}\n```\n", encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
