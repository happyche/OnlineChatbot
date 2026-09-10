# -*- coding: utf-8 -*-
"""
检索消融扫描：同一批问题跑遍各个开关组合，导出可直接喂给 RAGAS 的数据。

为什么需要它：混合检索与重排各有成本（BM25 需要建索引、重排要跑交叉编码器），
是否值得开启取决于具体语料与问题分布，只能靠量化对比来决定。
本脚本把「同一批问题 × 不同开关组合」的检索结果落成 JSONL，
每条记录含 question / contexts / stages，正好对上 RAGAS 的
context_precision、context_recall 所需的输入。

用法:
    # 用当前配置的向量库跑默认四种组合
    python scripts/retrieval_sweep.py --questions eval/questions.hss.jsonl

    # 用指定目录的 Markdown 现建一份临时索引（结果可复现，不动现有库）
    python scripts/retrieval_sweep.py --questions eval/questions.hss.jsonl \
        --corpus eval/corpus

    # 只跑部分组合；重排用无需权重的字面实现做冒烟
    python scripts/retrieval_sweep.py --questions q.jsonl \
        --combos baseline,hybrid --rerank-provider lexical

问题文件格式（JSONL，每行一条）:
    {"question": "...",
     "ground_truth": "...",
     "expected_keywords": ["3.10"],
     "expected_sources": ["a.md"]}
     "history": [{"role": "user", ...}, {"role": "assistant", ...}]  # 可选，多轮评测

  question 必填，其余可选。填了 expected_keywords 后脚本会直接算出
  Hit@K、MRR 与 Precision@K —— 判断重排是否有效关键要看 MRR，
  因为重排改变的是名次而不是召回集合，只看命中率往往看不出差别。
  也接受纯文本文件，每行一个问题。

  跑之前先用 scripts/validate_questions.py 校验一遍：expected_keywords
  若在语料中根本不存在，那道题的 Hit@K 恒为 0 且不会报错，
  只会让指标悄悄偏低。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402
from retrieval import LexicalOverlapReranker  # noqa: E402

#: 默认扫描的开关组合
COMBOS: dict[str, dict] = {
    "baseline": {"hybrid_search_enabled": False, "rerank_enabled": False},
    "hybrid": {"hybrid_search_enabled": True, "rerank_enabled": False},
    "rerank": {"hybrid_search_enabled": False, "rerank_enabled": True},
    "hybrid+rerank": {"hybrid_search_enabled": True, "rerank_enabled": True},
}


def load_questions(path: Path) -> list[dict]:
    """读取问题集，兼容 JSONL 与纯文本。"""
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            if "question" not in obj:
                raise ValueError(f"缺少 question 字段: {line[:80]}")
            items.append(obj)
        else:
            items.append({"question": line})
    if not items:
        raise ValueError(f"{path} 中没有任何问题")
    return items


async def build_temp_index(corpus_dir: Path, settings: dict, tmp_dir: Path) -> None:
    """把语料目录索引进一份临时向量库，避免污染现有数据。"""
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise ValueError(f"{corpus_dir} 下没有 .md 文件")
    engine = RAGEngine(settings=settings)
    total = 0
    for path in files:
        total += await engine.add_document(path.read_text(encoding="utf-8"), path.name)
    print(f"已在临时索引中写入 {len(files)} 个文档、{total} 个文本块 → {tmp_dir}")


async def run_combo(
    name: str, overrides: dict, questions: list[dict], base_settings: dict, rerank_provider: str
) -> list[dict]:
    """在一种开关组合下跑完整批问题。"""
    settings = dict(base_settings)
    settings.update(overrides)
    settings["rerank_provider"] = rerank_provider

    # 用字面重排时直接注入实例，省去模型权重
    injected = (
        LexicalOverlapReranker()
        if settings.get("rerank_enabled") and rerank_provider == "lexical"
        else None
    )
    engine = RAGEngine(settings=settings, reranker=injected)

    records: list[dict] = []
    for item in questions:
        result = await engine.retrieve_with_diagnostics(
            item["question"], history=item.get("history")
        )
        hits = result["hits"]
        record = {
            "combo": name,
            "question": item["question"],
            # RAGAS 的 contexts 就是这个字符串列表
            "contexts": [h.text for h in hits],
            "citations": [h.citation for h in hits],
            "scores": [round(h.score, 6) for h in hits],
            "score_type": hits[0].score_type if hits else None,
            "stages": result["stages"],
        }
        if "ground_truth" in item:
            record["ground_truth"] = item["ground_truth"]

        if "expected_sources" in item:
            expected = set(item["expected_sources"])
            record["expected_sources"] = sorted(expected)
            record["source_hit"] = bool(expected & {h.source for h in hits})

        if item.get("expected_keywords"):
            keywords = item["expected_keywords"]
            # 含任一关键词即视为相关片段
            ranks = [
                i
                for i, h in enumerate(hits, start=1)
                if any(kw in h.text for kw in keywords)
            ]
            record["expected_keywords"] = keywords
            record["relevant_ranks"] = ranks
            record["hit"] = bool(ranks)
            # MRR 的单条贡献：命中越靠前分值越高，正是重排应当改善的量
            record["reciprocal_rank"] = 1.0 / ranks[0] if ranks else 0.0
            record["precision_at_k"] = len(ranks) / len(hits) if hits else 0.0

        records.append(record)
    return records


def summarise(all_records: list[dict], top_k: int) -> None:
    """
    打印各组合的排名质量对比。

    重点看 MRR：重排改变的是名次而非召回集合，因此 Hit@K 常常四种组合
    都一样，只有 MRR 和 Precision@K 能反映出「正确片段被提到多前面」。
    """
    by_combo: dict[str, list[dict]] = {}
    for r in all_records:
        by_combo.setdefault(r["combo"], []).append(r)

    header = (
        f"{'组合':<16}{'问题数':>6}{'平均条数':>10}"
        f"{f'Hit@{top_k}':>9}{'MRR':>8}{f'P@{top_k}':>8}{'首位命中':>10}"
    )
    print("\n" + "=" * 78)
    print(header)
    print("-" * 78)

    baseline_mrr = None
    for name, records in by_combo.items():
        count = len(records)
        avg_ctx = sum(len(r["contexts"]) for r in records) / count
        judged = [r for r in records if "hit" in r]
        if judged:
            hit = sum(1 for r in judged if r["hit"]) / len(judged)
            mrr = sum(r["reciprocal_rank"] for r in judged) / len(judged)
            prec = sum(r["precision_at_k"] for r in judged) / len(judged)
            top1 = sum(1 for r in judged if r["relevant_ranks"][:1] == [1]) / len(judged)
            cells = f"{hit:>9.1%}{mrr:>8.3f}{prec:>8.3f}{top1:>10.1%}"
            if name == "baseline":
                baseline_mrr = mrr
        else:
            cells = f"{'—':>9}{'—':>8}{'—':>8}{'—':>10}"
        print(f"{name:<16}{count:>6}{avg_ctx:>10.2f}{cells}")

    print("=" * 78)

    if baseline_mrr:
        print("相对 baseline 的 MRR 变化：")
        for name, records in by_combo.items():
            judged = [r for r in records if "hit" in r]
            if name == "baseline" or not judged:
                continue
            mrr = sum(r["reciprocal_rank"] for r in judged) / len(judged)
            delta = mrr - baseline_mrr
            pct = delta / baseline_mrr * 100 if baseline_mrr else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<16}{sign}{delta:.3f} ({sign}{pct:.1f}%)")

    print("\n注：以上为关键词级的粗略指标，用于先定方向。")
    print("    忠实度、上下文精确率等请用导出的 JSONL 交给 RAGAS 计算。")


async def main_async(args) -> int:
    questions = load_questions(Path(args.questions))
    print(f"问题数: {len(questions)}")

    combos = {
        k: v
        for k, v in COMBOS.items()
        if not args.combos or k in {c.strip() for c in args.combos.split(",")}
    }
    if not combos:
        print(f"没有匹配的组合，可选: {', '.join(COMBOS)}")
        return 1
    print(f"开关组合: {', '.join(combos)}")

    if args.rerank_provider == "local" and any(
        v.get("rerank_enabled") for v in combos.values()
    ):
        print("提示: 重排使用本地交叉编码器，首次运行需下载约 1GB 权重。")
        print("      可加 --rerank-provider lexical 先做无权重冒烟。")

    tmp_dir = None
    try:
        if args.corpus:
            tmp_dir = Path(tempfile.mkdtemp(prefix="retrieval-sweep-"))
            config.CHROMA_DIR = tmp_dir / "chroma"
            base_settings = config.load_settings()
            await build_temp_index(Path(args.corpus), base_settings, tmp_dir)
        else:
            base_settings = config.load_settings()
            print(f"使用现有向量库: {config.CHROMA_DIR}")

        base_settings["top_k"] = args.top_k
        base_settings["candidate_pool_size"] = args.pool

        all_records: list[dict] = []
        for name, overrides in combos.items():
            print(f"\n--- {name} ---")
            records = await run_combo(
                name, overrides, questions, base_settings, args.rerank_provider
            )
            all_records.extend(records)
            for r in records[: args.preview]:
                head = r["citations"][0] if r["citations"] else "(无命中)"
                print(f"  {r['question'][:28]:<30} → {len(r['contexts'])} 条, 首位: {head}")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n已写入 {out_path}（{len(all_records)} 条记录）")

        summarise(all_records, args.top_k)
        return 0
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="问题集文件（JSONL 或纯文本）")
    parser.add_argument("--output", default="eval/sweep_results.jsonl")
    parser.add_argument("--corpus", default="", help="现建临时索引的 Markdown 目录")
    parser.add_argument("--combos", default="", help=f"逗号分隔，可选: {','.join(COMBOS)}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=20, help="融合/重排前的候选池大小")
    parser.add_argument(
        "--rerank-provider",
        default="local",
        choices=["local", "lexical"],
        help="local=交叉编码器（需权重）｜lexical=字面重叠（冒烟用）",
    )
    parser.add_argument("--preview", type=int, default=3, help="每个组合打印前几条")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
