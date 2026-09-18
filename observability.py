# -*- coding: utf-8 -*-
"""
可观测性
========
回答三个横在「能跑」和「在线运行」之间的问题：

  1. 这一问是哪一问？          → trace_id，贯穿日志、指标与用户反馈
  2. 它花了多少钱、多少时间？  → token 计量 + 成本换算 + 分阶段耗时
  3. 整体趋势是什么？          → Prometheus 指标

三条设计约束：

**一、token 优先取端点如实返回的 usage，取不到才估算，并且如实标注是哪一种。**
把估算值当精确值上报比没有数字更糟：它会让「换了模型之后成本降三成」这类结论
建立在估算公式的偏差上。`tokens_source` 存在的意义就是让读数的人知道自己在看什么。

**二、结构化请求日志与人读日志分开落盘。** 一行一个 JSON 的文件可以直接喂给
jq / pandas / Loki；和带缩进换行的人读日志混在一个文件里，两者都变难用。

**三、`prometheus_client` 是可选依赖。** 缺它时 /metrics 返回 501，而 trace_id、
成本核算、结构化日志照常工作——可观测性不该是一个要么全有要么全无的开关。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

import config

#: 结构化请求日志。单独一个 logger 名字，才能在 _configure_logging 里
#: 把它接到独立文件上而不污染人读日志。
_REQUEST_LOGGER = logging.getLogger("request")

logger = logging.getLogger(__name__)


# ======================================================================
# trace id
# ======================================================================

#: 当前请求的 trace id。用 ContextVar 而不是参数层层传递：
#: 日志过滤器、工具内部、异常处理都要拿到它，而它们之间没有共同的调用签名。
#: asyncio 下每个 Task 继承一份独立的上下文副本，因此并发请求不会串号。
_TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """生成一个 trace id。取 uuid4 前 16 位——够长到不会撞，够短到能人工念。"""
    return uuid.uuid4().hex[:16]


def set_trace_id(trace_id: str) -> None:
    _TRACE_ID.set(trace_id)


def current_trace_id() -> str:
    """当前 trace id；不在请求上下文里（如后台任务）时返回空串。"""
    return _TRACE_ID.get()


class TraceIdFilter(logging.Filter):
    """
    给每条日志补上 trace_id 字段。

    装在 handler 上而不是某个 logger 上：format 里一旦写了 %(trace_id)s，
    **任何**缺这个字段的记录都会让日志系统自己抛异常——包括 httpx、chromadb
    这些第三方库打的日志。装在 handler 上可以覆盖所有流经它的记录。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = current_trace_id() or "-"
        return True


# ======================================================================
# token 估算
# ======================================================================

def estimate_tokens(text: str) -> int:
    """
    估算 token 数：CJK 字符按 1 个、其余按 4 字符 1 个。

    不用 tiktoken：它的分词器对应 OpenAI 系模型，而本项目的对话端点常常指向
    Ollama 上的 qwen，词表完全不同，算出来并不更准却要多背一个依赖。
    这里需要的只是一个**单调且可复现**的估计量，宁可略微高估。

    放在这里而不是 agent/tools.py：工具层的 token 预算和请求级的用量计费
    必须用同一个口径，否则「预算说 1200、账单说 1800」这种对不上账的情况
    根本无从排查。agent/tools.py 从这里导入。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(text) - cjk + 3) // 4


# ======================================================================
# 成本
# ======================================================================

@dataclass(frozen=True)
class Pricing:
    """每 1000 token 的单价，货币单位由 config.COST_CURRENCY 声明。"""

    prompt: float = 0.0
    completion: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.prompt == 0.0 and self.completion == 0.0


def pricing_for(model: str) -> Pricing:
    """
    取某个模型的单价。

    按模型名查而不是用一个全局单价：`llm_model` 可以在运行时热改，
    而云端模型与内网自建模型的成本差着好几个数量级。查不到就回落到
    `default`，默认 0——自建服务确实不按 token 计费，此时报一个 0
    比报一个瞎猜的数字诚实。
    """
    table = config.LLM_PRICING
    entry = table.get(model) or table.get("default") or {}
    return Pricing(
        prompt=float(entry.get("prompt", 0.0)),
        completion=float(entry.get("completion", 0.0)),
    )


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = pricing_for(model)
    return (
        prompt_tokens * price.prompt + completion_tokens * price.completion
    ) / 1000.0


# ======================================================================
# Prometheus 指标
# ======================================================================

try:  # pragma: no cover - 取决于环境是否装了 prometheus_client
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

    METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    METRICS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"

MISSING_METRICS_DEPS = (
    "指标导出需要 prometheus_client，请先安装：pip install prometheus-client"
)

#: 用独立 registry 而不是默认的全局 registry。默认 registry 是进程级单例，
#: 模块被重复导入（uvicorn --reload、测试里 importlib.reload）时重复注册
#: 同名指标会抛 Duplicated timeseries，把服务直接搞挂。
REGISTRY = CollectorRegistry() if METRICS_AVAILABLE else None

#: agent 模式端到端可能几十秒，固定管道通常几秒。用同一组桶覆盖两者，
#: 否则两条路径的分位数不可横向比较——而「该不该开 agent」恰恰要比这个。
_LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, float("inf"))
_FIRST_TOKEN_BUCKETS = (0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 30, 45, 60, float("inf"))


class _NoopMetric:
    """缺 prometheus_client 时的替身：所有调用都成功，什么也不做。"""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None


def _counter(name, doc, labels=()):
    if not METRICS_AVAILABLE:
        return _NoopMetric()
    return Counter(name, doc, labels, registry=REGISTRY)


def _histogram(name, doc, labels=(), buckets=_LATENCY_BUCKETS):
    if not METRICS_AVAILABLE:
        return _NoopMetric()
    return Histogram(name, doc, labels, buckets=buckets, registry=REGISTRY)


REQUESTS = _counter(
    "rag_requests_total", "问答请求数", ("route", "status")
)
LATENCY = _histogram(
    "rag_request_latency_seconds", "问答端到端耗时", ("route",)
)
FIRST_TOKEN = _histogram(
    "rag_first_token_latency_seconds",
    "首个 token 的延迟（用户唯一真正感知到的那个数）",
    ("route",),
    buckets=_FIRST_TOKEN_BUCKETS,
)
STAGE_LATENCY = _histogram(
    "rag_stage_latency_seconds",
    "分阶段耗时。用来定位瓶颈在检索还是在生成",
    ("route", "stage"),
)
TOKENS = _counter(
    "rag_llm_tokens_total", "LLM token 消耗", ("route", "kind", "source")
)
COST = _counter(
    "rag_llm_cost_total", f"LLM 费用（{config.COST_CURRENCY}）", ("route", "model")
)
MODEL_CALLS = _counter(
    "rag_model_calls_total", "LLM 调用次数。agent 模式下每问 2～6 次", ("route",)
)
TOOL_CALLS = _counter(
    "rag_tool_calls_total",
    "工具调用次数。status=fail 偏高说明工具描述或参数 schema 有歧义",
    ("tool", "status"),
)
RETRIEVAL_SKIPPED = _counter(
    "rag_retrieval_skipped_total", "agent 判定无需取材而直接作答的次数"
)
FEEDBACK = _counter(
    "rag_feedback_total", "用户反馈", ("route", "verdict")
)


def render_metrics() -> bytes:
    """导出文本格式的指标快照。"""
    if not METRICS_AVAILABLE:
        raise RuntimeError(MISSING_METRICS_DEPS)
    return generate_latest(REGISTRY)


# ======================================================================
# 请求记录
# ======================================================================

@dataclass
class Usage:
    """一次请求的 token 用量。source 标明是端点如实返回的还是估算的。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    source: str = "none"  # usage | estimated | none

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RequestRecord:
    """
    一次问答的全部可观测数据，从请求进来一直填到响应结束。

    做成一个可变累加器而不是在结束时一次性构造：流式响应下「首 token 什么时候
    到的」「中途出错前已经流了多少」这些事实只在过程中存在，事后无从补算。

    finish() 是唯一的出口，同时做三件事：写结构化日志、打指标、返回一份
    可以直接落库的 dict。三件事合在一处是为了保证它们看到的是同一组数字——
    分开写迟早会出现「日志里 2.1 秒、指标里 2.4 秒」这种无法解释的偏差。
    """

    route: str
    question: str
    trace_id: str = field(default_factory=new_trace_id)
    session_id: Optional[str] = None
    model: str = ""
    provider: str = ""

    usage: Usage = field(default_factory=Usage)
    stages: dict = field(default_factory=dict)
    stage_seconds: dict = field(default_factory=dict)

    status: str = "ok"
    error: Optional[str] = None
    answer_chars: int = 0

    _started: float = field(default_factory=time.perf_counter)
    _first_token: Optional[float] = None
    _finished: Optional[float] = None

    # ---------------------------------------------------------------- 过程

    def mark_first_token(self) -> None:
        """记录首 token 到达时刻。重复调用只认第一次。"""
        if self._first_token is None:
            self._first_token = time.perf_counter()

    def add_stage_seconds(self, stage: str, seconds: float) -> None:
        """累加某个阶段的耗时（同名阶段可能出现多次，如 agent 的多轮取材）。"""
        if seconds is None or seconds < 0:
            return
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + float(seconds)

    def set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """记录端点如实返回的用量。"""
        self.usage = Usage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            source="usage",
        )

    def estimate_usage(self, prompt_texts, completion_text: str) -> None:
        """
        端点没给 usage 时退而估算，**且只在没有精确值时生效**。

        精确值优先是硬规则：估算值覆盖真实值会让账单口径无声地退化，
        而 tokens_source 字段又会照旧显示 "usage"，等于骗自己。
        """
        if self.usage.source == "usage":
            return
        prompt = sum(estimate_tokens(t or "") for t in (prompt_texts or []))
        self.usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=estimate_tokens(completion_text or ""),
            source="estimated",
        )

    # ---------------------------------------------------------------- 结果

    @property
    def latency_seconds(self) -> float:
        end = self._finished if self._finished is not None else time.perf_counter()
        return end - self._started

    @property
    def first_token_seconds(self) -> Optional[float]:
        if self._first_token is None:
            return None
        return self._first_token - self._started

    @property
    def cost(self) -> float:
        return cost_of(
            self.model, self.usage.prompt_tokens, self.usage.completion_tokens
        )

    def finish(self, *, status: Optional[str] = None, error: Optional[str] = None) -> dict:
        """收尾：写结构化日志、打指标，返回可落库的 dict。"""
        self._finished = time.perf_counter()
        if error:
            self.error = error
            self.status = status or "error"
        elif status:
            self.status = status

        record = self.as_dict()
        self._emit_log(record)
        self._emit_metrics()
        return record

    def as_dict(self) -> dict:
        """
        扁平的一层结构，不做嵌套：这份数据的主要消费方式是
        `jq`、`pandas.read_json(lines=True)` 和时序库，它们都对嵌套不友好。
        """
        first = self.first_token_seconds
        data = {
            "trace_id": self.trace_id,
            "route": self.route,
            "status": self.status,
            "model": self.model,
            "provider": self.provider,
            "session_id": self.session_id,
            "question": self.question,
            "answer_chars": self.answer_chars,
            "latency_ms": round(self.latency_seconds * 1000, 1),
            "first_token_ms": round(first * 1000, 1) if first is not None else None,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "tokens_source": self.usage.source,
            "cost": round(self.cost, 6),
            "currency": config.COST_CURRENCY,
            "error": self.error,
        }
        for stage, seconds in self.stage_seconds.items():
            data[f"{stage}_ms"] = round(seconds * 1000, 1)
        # stages 是检索/agent 各阶段的既有诊断字段，原样带上，
        # 这样「走错了工具」和「延迟高」能在同一行日志里对上
        data.update(self.stages)
        return data

    def _emit_log(self, record: dict) -> None:
        _REQUEST_LOGGER.info(json.dumps(record, ensure_ascii=False, default=str))

    def _emit_metrics(self) -> None:
        route = self.route
        REQUESTS.labels(route=route, status=self.status).inc()
        LATENCY.labels(route=route).observe(self.latency_seconds)

        first = self.first_token_seconds
        if first is not None:
            FIRST_TOKEN.labels(route=route).observe(first)

        for stage, seconds in self.stage_seconds.items():
            STAGE_LATENCY.labels(route=route, stage=stage).observe(seconds)

        if self.usage.source != "none":
            source = self.usage.source
            if self.usage.prompt_tokens:
                TOKENS.labels(route=route, kind="prompt", source=source).inc(
                    self.usage.prompt_tokens
                )
            if self.usage.completion_tokens:
                TOKENS.labels(route=route, kind="completion", source=source).inc(
                    self.usage.completion_tokens
                )

        cost = self.cost
        if cost:
            COST.labels(route=route, model=self.model or "unknown").inc(cost)

        calls = int(self.stages.get("agent_model_calls") or 0) or 1
        MODEL_CALLS.labels(route=route).inc(calls)

        if self.stages.get("agent_retrieval_skipped"):
            RETRIEVAL_SKIPPED.inc()

    # ---------------------------------------------------------------- 工具事件

    @staticmethod
    def observe_tool(tool: str, ok: bool) -> None:
        TOOL_CALLS.labels(tool=tool or "unknown", status="ok" if ok else "fail").inc()


def observe_feedback(route: str, verdict: str) -> None:
    FEEDBACK.labels(route=route or "unknown", verdict=verdict).inc()
