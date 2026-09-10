# -*- coding: utf-8 -*-
"""
在会掐断长连接的网络环境下拉取重排模型权重（约 1GB）。

download_model.py 自带的重试只能应对「抛出异常」的失败。企业代理的表现不是这样：
它会静默丢弃长时间的大文件传输，连接断了但客户端收不到 RST，于是
huggingface_hub 永远阻塞在 socket read 上——既不报错，也不超时，进程就那么挂着。

这里在外层按下载进度做停滞检测：盯着缓存里的 .incomplete 分片，
一段时间不增长就判定卡死，杀掉子进程重来，靠 huggingface_hub 自身的
断点续传接着下。实测每轮能推进几百 MB，几轮即可下完。

用法:
    python scripts/fetch_reranker.py
    python scripts/fetch_reranker.py --stall-seconds 120 --max-attempts 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 10


def repo_cache_dir() -> Path:
    """重排模型在 HF 缓存中的目录。"""
    repo = config.DEFAULTS["rerank_model"].replace("/", "--")
    return Path(config.DEFAULTS["embedding_cache_dir"]) / f"models--{repo}"


def downloaded_bytes(cache_dir: Path) -> int:
    """当前未完成分片的大小；返回 0 表示没有分片（尚未开始或已完成）。"""
    sizes = [p.stat().st_size for p in cache_dir.rglob("*.incomplete")]
    return max(sizes, default=0)


def weights_ready(cache_dir: Path) -> bool:
    """权重是否已就位。下载完成后 hf 会把 .incomplete 转正。"""
    return any(cache_dir.rglob("model.onnx"))


def run_attempt(cache_dir: Path, stall_seconds: int) -> None:
    """跑一次下载子进程，停滞超时则终止它。"""
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "download_model.py"), "--reranker", "--retries", "1"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    last_size = downloaded_bytes(cache_dir)
    last_change = time.monotonic()

    while proc.poll() is None:
        time.sleep(POLL_SECONDS)
        current = downloaded_bytes(cache_dir)
        if current > last_size:
            print(f"    {current / 1024 / 1024:,.1f} MB", flush=True)
            last_size = current
            last_change = time.monotonic()
        elif time.monotonic() - last_change >= stall_seconds:
            print(f"    停滞 {stall_seconds} 秒，终止本次尝试", flush=True)
            proc.kill()
            break

    proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=90,
        help="进度多久不增长判定为卡死",
    )
    args = parser.parse_args()

    cache_dir = repo_cache_dir()
    print(f"模型     : {config.DEFAULTS['rerank_model']}")
    print(f"缓存目录 : {cache_dir}")

    for attempt in range(1, args.max_attempts + 1):
        if weights_ready(cache_dir):
            print("权重已就位。")
            return 0

        have = downloaded_bytes(cache_dir) / 1024 / 1024
        print(f"\n第 {attempt}/{args.max_attempts} 次尝试（已有 {have:,.1f} MB）…", flush=True)
        run_attempt(cache_dir, args.stall_seconds)

    if weights_ready(cache_dir):
        print("权重已就位。")
        return 0

    print("\n达到最大尝试次数仍未完成。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
