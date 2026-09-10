# -*- coding: utf-8 -*-
"""
预下载本地嵌入模型权重。

服务本身是惰性加载的，首次上传文档时才会下载模型（约 90MB）。
但 Hugging Face 对未认证请求有速率限制，下载可能很慢甚至中断，
中断后缓存里会留下不完整的快照，导致后续加载报「文件不存在」。

这个脚本带重试，并在开始前清掉不完整的缓存，适合部署前先跑一遍。

用法:
    python scripts/download_model.py
    python scripts/download_model.py --model BAAI/bge-small-zh-v1.5 --retries 5

提示：设置环境变量 HF_TOKEN 可显著提高下载速率上限；
国内网络可设置 HF_ENDPOINT=https://hf-mirror.com 走镜像。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# Windows 控制台默认 GBK，模型名与路径中可能含无法编码的字符
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def has_incomplete_blobs(cache_dir: Path) -> bool:
    """检查缓存中是否存在未下载完的 blob。"""
    return any(cache_dir.rglob("*.incomplete"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="留空则用配置中的默认模型")
    parser.add_argument("--cache-dir", default=config.DEFAULTS["embedding_cache_dir"])
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="下载重排用的交叉编码器（约 1GB）而不是嵌入模型",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="下载前先删除该模型的现有缓存（缓存损坏时使用）",
    )
    args = parser.parse_args()

    kind = "重排模型" if args.reranker else "嵌入模型"
    default_model = (
        config.DEFAULTS["rerank_model"] if args.reranker
        else config.DEFAULTS["local_embedding_model"]
    )
    model_name = args.model or default_model

    cache_dir = Path(args.cache_dir)
    print(f"{kind} : {model_name}")
    print(f"缓存目录 : {cache_dir}")
    if args.reranker:
        print("提示     : 交叉编码器权重约 1GB，下载可能较慢。")

    if args.clean and cache_dir.exists():
        print("正在清理现有缓存…")
        shutil.rmtree(cache_dir, ignore_errors=True)

    def _load_and_warm():
        """加载模型并跑一次推理，确认权重完整可用而不只是文件存在。"""
        if args.reranker:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))
            scores = list(model.rerank("超时设置", ["超时默认 60 秒", "无关内容"]))
            return f"重排分数样例 = {[round(s, 3) for s in scores]}"

        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        vector = next(iter(model.embed(["预热验证：这是一段中文文本。"])))
        return f"向量维度 = {len(vector)}"

    for attempt in range(1, args.retries + 1):
        print(f"\n第 {attempt}/{args.retries} 次尝试…")
        try:
            detail = _load_and_warm()
            print(f"完成。{detail}")
            if has_incomplete_blobs(cache_dir):
                print("警告：缓存中仍存在 .incomplete 文件，建议加 --clean 重跑。")
            return 0
        except Exception as exc:
            print(f"失败：{type(exc).__name__}: {str(exc)[:200]}")
            if attempt < args.retries:
                wait = attempt * 5
                print(f"{wait} 秒后重试…")
                time.sleep(wait)

    print(
        "\n多次尝试均失败。可尝试：\n"
        "  1. 设置 HF_TOKEN 提高速率上限\n"
        "  2. 设置 HF_ENDPOINT=https://hf-mirror.com 走镜像\n"
        "  3. 加 --clean 清掉损坏的缓存后重试\n"
        "  4. 改用远程嵌入：在 .env 中设 EMBEDDING_PROVIDER=openai"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
