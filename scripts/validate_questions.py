# -*- coding: utf-8 -*-
"""
校验问题集与语料是否自洽。

存在的理由：expected_keywords 是用「子串是否出现在召回片段里」来判定相关性的，
一旦某个关键词在语料里根本不存在，那道题的 Hit@K 会恒为 0。
这种错误不会报异常，只会让指标悄悄偏低，事后极难发现，
所以必须在跑评测之前先卡一道。

用法:
    python scripts/validate_questions.py eval/questions.hss.jsonl --corpus eval/corpus
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieval_sweep import load_questions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="校验问题集与语料自洽")
    parser.add_argument("questions")
    parser.add_argument("--corpus", required=True, help="语料目录")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    texts = {p.name: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.md")}
    if not texts:
        print(f"{corpus_dir} 下没有 .md 文件")
        return 1
    combined = "\n".join(texts.values())

    questions = load_questions(Path(args.questions))
    types = Counter(q.get("type", "untyped") for q in questions)
    problems: list[str] = []

    for i, item in enumerate(questions, 1):
        question = item["question"]
        kind = item.get("type", "untyped")

        if kind == "unanswerable":
            if item.get("ground_truth") or item.get("expected_keywords"):
                problems.append(f"[{i}] 拒答题不该带 ground_truth/expected_keywords: {question}")
            continue

        if not item.get("ground_truth"):
            problems.append(f"[{i}] 缺 ground_truth: {question}")

        for keyword in item.get("expected_keywords", []):
            if keyword not in combined:
                problems.append(f"[{i}] 关键词在语料中不存在: {keyword!r} —— {question}")

        for source in item.get("expected_sources", []):
            if source not in texts:
                problems.append(f"[{i}] 源文件不存在: {source}")

    print(f"语料: {len(texts)} 个文件，{len(combined)} 字符")
    print(f"问题: {len(questions)} 条")
    for kind, count in sorted(types.items()):
        print(f"    {kind:<14} {count}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for line in problems:
            print(f"  {line}")
        return 1

    print("\n校验通过：所有关键词都能在语料中找到，字段完整。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
