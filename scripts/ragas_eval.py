# -*- coding: utf-8 -*-
"""
[RAGAS] 用 RAGAS 评测本项目的 RAG 效果。

整个脚本都是为 RAGAS 评测新增的，不参与服务运行。

与 retrieval_sweep.py 的分工：
    retrieval_sweep.py  只看检索，用关键词命中算 Hit@K / MRR / P@K。
                        不花 LLM 调用，适合快速定方向。
    本脚本              用 LLM 当评委，既评检索也评生成。
                        贵、慢，但能回答关键词指标回答不了的问题：
                        召回的片段到底相不相关、答案有没有编造。

为什么不用 ragas 的 evaluate() 或 @experiment：
    evaluate() 在 0.4 已废弃；@experiment 绑定了 ragas 自己的数据集后端与
    结果存储目录，与本仓库既有的「自己写 JSONL、自己打汇总表」重复。
    这里直接实例化 ragas.metrics.collections 里的指标逐条 ascore，
    产物格式与 retrieval_sweep.py 保持一致，两边结果可以并排看。

四个指标各自需要什么：
    context_precision  问题 + 参考答案 + contexts   —— 召回的片段是否按相关性排在前面
    context_recall     问题 + 参考答案 + contexts   —— 参考答案中的信息是否都被召回了
    faithfulness       问题 + 答案 + contexts       —— 答案里的每个断言是否都有出处
    answer_relevancy   问题 + 答案（+ 嵌入模型）    —— 答案是否答在点上

前两个只看检索，不需要生成答案；后两个必须先生成答案，
因此只选前两个时脚本会跳过 LLM 生成，整体调用量差不多减半。

用法:
    # 只导出样本，不算指标（不需要安装 ragas，用来先确认数据管道通了）
    python scripts/ragas_eval.py --questions eval/questions.hss.jsonl --dump-only

    # 完整评测，默认只跑 baseline 组合
    python scripts/ragas_eval.py --questions eval/questions.hss.jsonl

    # 消融对比：看开混合检索和重排之后各指标怎么变
    python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
        --combos baseline,hybrid,hybrid+rerank

    # 只跑检索类指标，省掉答案生成
    python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
        --metrics context_precision,context_recall

    # 被测系统跑本地小模型，评委走云端强模型（避免自评偏袒，也避开本地显存限制）
    python scripts/ragas_eval.py --questions ... \
        --judge-provider openai --judge-model qwen-plus

    # 端点不支持原生 json_schema 时降级
    python scripts/ragas_eval.py --questions ... --judge-json-mode

    # 交叉验证：同一批样本让两个独立评委各评一遍，再用 ragas_compare.py 对比。
    # 必须先 dump 样本再分别打分 —— 重跑 RAG 会生成不同的答案，
    # 那样比出来的是答案差异而不是评委差异。
    python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
        --corpus eval/corpus --dump-only --output eval/samples.jsonl
    python scripts/ragas_eval.py --samples eval/samples.jsonl \
        --judge-model qwen3:8b       --output eval/scored_qwen.jsonl
    python scripts/ragas_eval.py --samples eval/samples.jsonl \
        --judge-provider cursor      --output eval/scored_cursor.jsonl
    python scripts/ragas_compare.py eval/scored_qwen.jsonl eval/scored_cursor.jsonl

问题集格式与 retrieval_sweep.py 相同（eval/questions.hss.jsonl），
其中 ground_truth 会作为 RAGAS 的 reference 使用；
缺 ground_truth 的问题会自动跳过需要参考答案的两个指标 ——
测试集里的拒答类问题正是靠这一点，只参与生成侧指标的评估。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402
from retrieval import LexicalOverlapReranker  # noqa: E402

# 同在 scripts/ 下，直接按模块名导入。
# COMBOS 与 load_questions 复用 retrieval_sweep 的定义，
# 两个脚本必须用同一套开关组合与同一份问题集解析，否则结果没法并排比较。
import ragas_adapters  # noqa: E402
from retrieval_sweep import COMBOS, load_questions  # noqa: E402


class MetricSpec:
    """
    一个指标的元信息。

    needs_reference : 是否必须有 ground_truth，缺了就只能跳过这条
    needs_answer    : 是否必须先生成答案（决定要不要调 LLM 生成，代价差别很大）
    needs_embedding : 是否需要嵌入模型
    """

    def __init__(self, cls_name: str, needs_reference: bool, needs_answer: bool,
                 needs_embedding: bool = False):
        self.cls_name = cls_name
        self.needs_reference = needs_reference
        self.needs_answer = needs_answer
        self.needs_embedding = needs_embedding


#: 支持的指标。键名用 RAGAS 论文里的通用叫法，而不是 v0.4 的类名，
#: 这样和 DESIGN.md 里的表述以及别处的 RAGAS 资料对得上。
METRICS: dict[str, MetricSpec] = {
    "context_precision": MetricSpec("ContextPrecisionWithReference", True, False),
    "context_recall": MetricSpec("ContextRecall", True, False),
    "faithfulness": MetricSpec("Faithfulness", False, True),
    "answer_relevancy": MetricSpec("AnswerRelevancy", False, True, needs_embedding=True),
}

DEFAULT_METRICS = "context_precision,context_recall,faithfulness,answer_relevancy"


def build_metrics(names: list[str], llm, embeddings):
    """按名字实例化 RAGAS 指标对象。"""
    from ragas.metrics import collections as ragas_metrics

    built = {}
    for name in names:
        spec = METRICS[name]
        cls = getattr(ragas_metrics, spec.cls_name)
        if spec.needs_embedding:
            built[name] = cls(llm=llm, embeddings=embeddings)
        else:
            built[name] = cls(llm=llm)
    return built


async def collect_samples(
    combo: str,
    overrides: dict,
    questions: list[dict],
    base_settings: dict,
    rerank_provider: str,
    need_answer: bool,
) -> list[dict]:
    """
    在一种开关组合下跑完整批问题，产出 RAGAS 所需的样本。

    字段名直接对齐 ragas.SingleTurnSample（user_input / response /
    retrieved_contexts / reference），导出的 JSONL 可以原样喂给别的 RAGAS 工具。
    注意 v0.4 把 v0.3 的 ground_truths(list) 改成了 reference(str)。
    """
    settings = dict(base_settings)
    settings.update(overrides)
    settings["rerank_provider"] = rerank_provider

    injected = (
        LexicalOverlapReranker()
        if settings.get("rerank_enabled") and rerank_provider == "lexical"
        else None
    )
    engine = RAGEngine(settings=settings, reranker=injected)

    samples: list[dict] = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        if need_answer:
            # answer() 一次调用同时返回答案和它实际依据的 contexts。
            # 分开调 retrieve + query 看似等价，但两次检索未必给出同一批片段，
            # 那样评出来的忠实度就对不上号了。
            result = await engine.answer(question)
            response = result["answer"]
        else:
            result = await engine.retrieve_with_diagnostics(question)
            response = None

        hits = result["hits"]
        sample = {
            "combo": combo,
            "user_input": question,
            "retrieved_contexts": [h.text for h in hits],
            "citations": [h.citation for h in hits],
            "stages": result["stages"],
        }
        if response is not None:
            sample["response"] = response
        if item.get("ground_truth"):
            sample["reference"] = item["ground_truth"]

        samples.append(sample)
        print(f"  [{i}/{len(questions)}] {question[:30]:<32} "
              f"contexts={len(hits)}"
              + (f" answer={len(response)}字" if response else ""))

    return samples


def load_samples(path: Path) -> list[dict]:
    """
    读回之前 --dump-only 导出的样本。

    存在的意义是让「换个评委再评一遍」成为可能：生成是有随机性的，
    重跑一遍 RAG 会得到不同的答案，那时两次评分的差异就分不清
    是评委不同还是答案不同了。要比较评委，必须喂同一批样本。
    """
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if "user_input" not in obj:
            raise ValueError(f"样本缺少 user_input 字段: {line[:80]}")
        # 上一轮如果已经打过分，这里丢掉旧分数，避免和本轮结果混在一起
        obj.pop("scores", None)
        obj.pop("judge", None)
        samples.append(obj)
    if not samples:
        raise ValueError(f"{path} 中没有任何样本")
    return samples


async def score_samples(samples: list[dict], metrics: dict) -> list[dict]:
    """
    逐条逐指标打分。

    单条指标失败不中断整轮：评委 LLM 偶尔会返回解析不了的内容，
    十几分钟的评测不该因为其中一条就全部作废。失败的那格记 None 并留下错误信息。
    """
    records = []
    total = len(samples) * len(metrics)
    done = 0

    for sample in samples:
        record = dict(sample)
        scores: dict = {}

        for name, metric in metrics.items():
            spec = METRICS[name]
            done += 1

            if spec.needs_reference and not sample.get("reference"):
                scores[name] = {"value": None, "skipped": "缺少 ground_truth"}
                continue
            if spec.needs_answer and not sample.get("response"):
                scores[name] = {"value": None, "skipped": "未生成答案"}
                continue

            kwargs = {"user_input": sample["user_input"]}
            if spec.needs_answer:
                kwargs["response"] = sample["response"]
            if name != "answer_relevancy":
                # answer_relevancy 只看问题与答案，不吃 contexts
                kwargs["retrieved_contexts"] = sample["retrieved_contexts"]
            if spec.needs_reference:
                kwargs["reference"] = sample["reference"]

            started = time.monotonic()
            try:
                result = await metric.ascore(**kwargs)
                scores[name] = {"value": float(result.value)}
                # MetricResult 预留了评判理由字段，但 ragas 0.4.3 内置的这四个
                # 指标都不填。留着是为了将来上游补上或自定义指标用得着，
                # 恒为空时不写进 JSONL，免得每行都挂一串无意义的 null。
                reason = getattr(result, "reason", None)
                if reason:
                    scores[name]["reason"] = reason
            except Exception as exc:
                scores[name] = {"value": None, "error": f"{type(exc).__name__}: {exc}"}

            elapsed = time.monotonic() - started
            shown = scores[name].get("value")
            print(f"  [{done}/{total}] {name:<20} "
                  f"{'—' if shown is None else format(shown, '.3f')}  ({elapsed:.1f}s)")

        record["scores"] = scores
        records.append(record)

    return records


def report_failures(records: list[dict]) -> None:
    """
    汇总打分失败的原因。

    失败会让均值只按剩下的样本算，光看汇总表只能看到一个 n=7，
    看不出到底是模型不行还是评测本身出了岔子，所以要单独说清楚。
    """
    reasons: dict[str, int] = {}
    for r in records:
        for entry in r["scores"].values():
            if entry.get("error"):
                reasons[entry["error"].split(":")[0]] = reasons.get(
                    entry["error"].split(":")[0], 0
                ) + 1

    if not reasons:
        return

    print("\n打分失败统计：")
    for name, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count} 次")
        if name == "IncompleteOutputException":
            # 这条最常见也最容易误判：看起来像模型答不好，其实是评委被截断了
            print("    → 评委输出被 max_tokens 截断，未能产出完整 JSON。"
                  "会思考的模型尤其容易触发，请调大 --judge-max-tokens。")


def summarise(records: list[dict], metric_names: list[str]) -> str:
    """
    生成按组合汇总的 Markdown 对比表。

    均值只按打分成功的样本算，并单独标出有效样本数——
    如果某个指标大面积失败，光看均值会以为效果很好。
    """
    by_combo: dict[str, list[dict]] = {}
    for r in records:
        by_combo.setdefault(r["combo"], []).append(r)

    header = "| 组合 | 问题数 | " + " | ".join(metric_names) + " |"
    sep = "|---" * (len(metric_names) + 2) + "|"
    lines = [header, sep]

    for combo, rows in by_combo.items():
        cells = []
        for name in metric_names:
            values = [
                r["scores"][name]["value"]
                for r in rows
                if r["scores"].get(name, {}).get("value") is not None
            ]
            if values:
                mean = sum(values) / len(values)
                # n 与问题数不等时说明有跳过或失败，必须显式提示
                suffix = f" (n={len(values)})" if len(values) != len(rows) else ""
                cells.append(f"{mean:.3f}{suffix}")
            else:
                cells.append("—")
        lines.append(f"| {combo} | {len(rows)} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


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


async def main_async(args) -> int:
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        print(f"未知指标: {', '.join(unknown)}；可选: {', '.join(METRICS)}")
        return 1

    # 只选检索类指标时不必生成答案，可以省掉一半左右的 LLM 调用
    need_answer = any(METRICS[m].needs_answer for m in metric_names)
    need_embedding = any(METRICS[m].needs_embedding for m in metric_names)

    if not args.dump_only:
        try:
            ragas_adapters.ensure_ragas()
        except RuntimeError as exc:
            print(exc)
            print("\n若只想先确认数据管道，可加 --dump-only。")
            return 1

    questions: list[dict] = []
    combos: dict = {}
    if not args.samples:
        questions = load_questions(Path(args.questions))
        missing_gt = sum(1 for q in questions if not q.get("ground_truth"))
        print(f"问题数: {len(questions)}"
              + (f"（其中 {missing_gt} 条缺 ground_truth）" if missing_gt else ""))

        combos = {
            k: v
            for k, v in COMBOS.items()
            if not args.combos or k in {c.strip() for c in args.combos.split(",")}
        }
        if not combos:
            print(f"没有匹配的组合，可选: {', '.join(COMBOS)}")
            return 1
        print(f"开关组合: {', '.join(combos)}")

    print(f"评测指标: {', '.join(metric_names)}"
          + ("" if need_answer else "（均为检索类，跳过答案生成）"))

    tmp_dir = None
    judge_llm = None
    try:
        if args.corpus:
            tmp_dir = Path(tempfile.mkdtemp(prefix="ragas-eval-"))
            config.CHROMA_DIR = tmp_dir / "chroma"
            base_settings = config.load_settings()
            await build_temp_index(Path(args.corpus), base_settings, tmp_dir)
        else:
            base_settings = config.load_settings()
            if not args.samples:
                print(f"使用现有向量库: {config.CHROMA_DIR}")

        base_settings["top_k"] = args.top_k
        base_settings["candidate_pool_size"] = args.pool

        if args.llm_model:
            # 写回当前 provider 对应的那个键。写错键的话配置会被静默忽略，
            # 评的还是原来那个模型，而输出里却写着新模型名。
            provider = str(base_settings.get("llm_provider", "openai")).lower()
            base_settings["local_llm_model" if provider == "local" else "llm_model"] = args.llm_model

        # ---- 阶段一：取得样本 ----
        if args.samples:
            all_samples = load_samples(Path(args.samples))
            print(f"\n复用已有样本: {args.samples}（{len(all_samples)} 条）")
        else:
            if need_answer:
                sut = config.resolve_llm(base_settings)
                print(f"被测模型: {sut.provider} · {sut.base_url} · {sut.model}")

            all_samples = []
            for name, overrides in combos.items():
                print(f"\n--- 采样: {name} ---")
                all_samples.extend(
                    await collect_samples(
                        name, overrides, questions, base_settings,
                        args.rerank_provider, need_answer,
                    )
                )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.dump_only:
            with open(out_path, "w", encoding="utf-8") as f:
                for s in all_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"\n已写入 {out_path}（{len(all_samples)} 条样本，未计算指标）")
            return 0

        # ---- 阶段二：交给 RAGAS 打分 ----
        if args.judge_provider == "cursor":
            judge_model = args.judge_model or args.cursor_model
            print(f"\n评委: cursor · 本地 agent（已关闭工具）· {judge_model}")
            print("  注意：agent run 没有 temperature/seed，分数不可复现；"
                  "适合做交叉验证，不适合当回归基准线。")
            judge_llm = ragas_adapters.build_cursor_judge(model=judge_model)
            # 先用一个最小请求确认模型真的能用。Cursor 的模型目录列出的是
            # 账号「看得到」的模型，未开通的会返回 status=error 而不是鉴权错，
            # 不预检的话每一格都要重试三次才失败，白烧几十次 agent run。
            print("  预检评委模型…", end="", flush=True)
            await judge_llm.preflight()
            print(" 可用")
        else:
            judge = ragas_adapters.resolve_judge_endpoint(
                base_settings, provider=args.judge_provider, model=args.judge_model or None
            )
            judge_model = judge.model
            same_as_sut = judge.model == config.resolve_llm(base_settings).model
            print(f"\n评委: {judge.provider} · {judge.base_url} · {judge_model}"
                  + ("（与被测模型相同，存在自评偏袒风险）" if same_as_sut else ""))
            judge_llm = ragas_adapters.build_judge_llm(
                base_settings,
                model=args.judge_model or None,
                json_mode=args.judge_json_mode,
                provider=args.judge_provider,
                max_tokens=args.judge_max_tokens,
            )

        embeddings = None
        if need_embedding:
            # 复用引擎的 embedder，保证与检索处在同一向量空间。
            # 这一路始终走本项目自己的嵌入模型，与评委用哪个后端无关 ——
            # 换评委时 answer_relevancy 的可比性正是靠这一点保住的。
            probe = RAGEngine(settings=base_settings)
            embeddings = ragas_adapters.build_judge_embeddings(probe.embedder)
            print(f"评委嵌入: {probe.embedder.signature}")

        metrics = build_metrics(metric_names, judge_llm, embeddings)

        print(f"\n--- 打分（共 {len(all_samples) * len(metrics)} 次评判）---")
        records = await score_samples(all_samples, metrics)

        for r in records:
            r["judge"] = judge_model

        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n已写入 {out_path}（{len(records)} 条记录）")

        table = summarise(records, metric_names)
        print("\n" + table)
        report_failures(records)

        if getattr(judge_llm, "runs", None):
            print(f"\nCursor agent run 次数: {judge_llm.runs}，"
                  f"累计耗时 {judge_llm.total_ms / 1000:.1f}s")

        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"# RAGAS 评测报告\n\n"
            f"- 问题集: `{args.questions or args.samples}`（{len(all_samples)} 条样本）\n"
            f"- 评委模型: `{judge_model}`（provider: {args.judge_provider}）\n"
            f"- top_k: {args.top_k}｜候选池: {args.pool}\n\n"
            f"{table}\n\n"
            f"逐条明细见 `{out_path}`（每行含问题、检索到的 contexts、"
            f"答案原文与各指标分数，可据此排查低分样本）。\n",
            encoding="utf-8",
        )
        print(f"已写入 {report}")
        return 0
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # Cursor 评委持有一个 bridge 子进程与沙箱目录，不关会漏进程
        if judge_llm is not None and hasattr(judge_llm, "aclose"):
            await judge_llm.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="用 RAGAS 评测 RAG 效果")
    parser.add_argument("--questions", help="问题集文件（JSONL 或纯文本）")
    parser.add_argument("--samples", default="",
                        help="改为对已导出的样本打分，跳过 RAG 采样。"
                             "换评委做交叉验证时必须用它，否则比的不是评委而是答案")
    parser.add_argument("--output", default="eval/ragas_results.jsonl")
    parser.add_argument("--report", default="eval/ragas_report.md")
    parser.add_argument("--corpus", default="", help="现建临时索引的 Markdown 目录")
    parser.add_argument("--combos", default="baseline",
                        help=f"逗号分隔，可选: {','.join(COMBOS)}；留空跑全部")
    parser.add_argument("--metrics", default=DEFAULT_METRICS,
                        help=f"逗号分隔，可选: {','.join(METRICS)}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=20, help="融合/重排前的候选池大小")
    parser.add_argument("--rerank-provider", default="local", choices=["local", "lexical"],
                        help="local=交叉编码器（需权重）｜lexical=字面重叠（冒烟用）")
    parser.add_argument("--llm-model", default="",
                        help="覆盖被测的生成模型，用于横向比较不同模型在同一语料上的表现")
    parser.add_argument("--judge-model", default="",
                        help="评委模型名，默认与被测模型相同；下结论时建议换更强的模型")
    parser.add_argument("--judge-provider", default="auto",
                        choices=["auto", "openai", "local", "cursor"],
                        help="评委走哪个后端：auto=沿用 llm_provider｜openai=云端｜"
                             "local=自建｜cursor=Cursor agent（不可复现，仅供交叉验证）")
    parser.add_argument("--cursor-model", default="gpt-5.6-sol",
                        help="judge-provider=cursor 时用的模型 id。能列出不等于能用，"
                             "开跑前会做一次预检；不要用 auto/default，"
                             "它们会在请求间换模型，两次评测无从比较")
    parser.add_argument("--judge-json-mode", action="store_true",
                        help="评委改用 Markdown-JSON 结构化模式，端点不支持 json_schema 时用")
    parser.add_argument("--judge-max-tokens", type=int,
                        default=ragas_adapters.DEFAULT_JUDGE_MAX_TOKENS,
                        help="评委输出上限。会思考的模型若报 IncompleteOutput 需继续调大")
    parser.add_argument("--dump-only", action="store_true",
                        help="只导出样本不算指标，无需安装 ragas")
    args = parser.parse_args()
    if not args.questions and not args.samples:
        parser.error("需要 --questions（现跑一批）或 --samples（对已有样本打分）之一")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
