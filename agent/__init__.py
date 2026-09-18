# -*- coding: utf-8 -*-
"""
Agent 层（可选）
================
在既有 RAG 栈之上加一层取材决策，底座（rag_engine / retrieval / embeddings /
config）的既有逻辑一行不改，只新增了四个只读方法供工具取材。

    tools.py   三个取材工具 + token 预算  —— 只依赖 langchain_core
    runner.py  create_agent 装配与事件流  —— 需要 langchain / langchain-openai

分成两个模块的依据是「要不要完整的 langchain 栈」：未安装时工具层的单元测试
仍然可跑，只有 runner.py 的导入会给出一条可照做的安装提示。
"""
from __future__ import annotations

from typing import Optional

#: 未安装依赖时的提示。写成可直接照做的一行命令，
#: 而不是让调用方去猜缺了哪个包。
MISSING_DEPS = (
    "Agent 模式需要 langchain，请先安装：pip install -r requirements-agent.txt"
)


def is_available() -> bool:
    """依赖是否齐备。供健康检查与配置校验在开跑之前就给出结论。"""
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("langchain", "langchain_openai", "langgraph")
    )


def build_runner(
    engine,
    settings: Optional[dict] = None,
    model=None,
    with_session: bool = False,
):
    """
    构造 AgentRunner。惰性导入 runner.py，把「没装依赖」变成一条可读提示。

    model 可注入，用于离线测试。
    with_session 为真时接上进程级 InMemorySaver，按 thread_id 持久化对话；
    为假则每次运行都是全新状态，沿用「历史由前端维护」的原有行为。
    """
    try:
        from .runner import AgentRunner, default_checkpointer
    except ImportError as exc:  # pragma: no cover - 取决于环境是否装了 langchain
        raise RuntimeError(MISSING_DEPS) from exc
    return AgentRunner(
        engine,
        settings=settings,
        model=model,
        checkpointer=default_checkpointer() if with_session else None,
    )
