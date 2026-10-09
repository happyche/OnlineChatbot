# -*- coding: utf-8 -*-
"""
Agent 装配与驱动
================
用 langchain 的 `create_agent` 直接组装「模型 ↔ 工具」循环，不再自己写状态图。

    create_agent(model, tools, system_prompt=..., middleware=[...], checkpointer=...)

它编译出的就是一张 model ⇄ tools 的图：模型自己决定调哪个工具、调几次、
什么时候停下来作答。原来手写的 plan / grade 两个节点在这个循环里是多余的——
「资料够不够」本来就是模型每一轮都在做的判断，额外再花一次 LLM 调用去问它，
只是把同一个判断做了两遍。

**三个必须自己接上的东西**（create_agent 不管这些）：

一、**LLM 接现有的端点解析**。项目的对话端点有云端/自建两套配置，
   「哪套在生效」只解析一处（config.resolve_llm），agent 直接读 settings 里的
   llm_model 就会在切到自建服务后把云端模型名发到内网。连带的内网代理绕过
   也要一起接过来，否则企业环境里请求会被发去代理并返回一段 HTML 错误页。

二、**上下文压缩用 SummarizationMiddleware**。多轮之后消息越堆越长，
   超过阈值就把早期消息折叠成摘要。这是 langchain 自带的能力，
   自己写一遍只会得到一个更差的版本。

三、**终止条件**。模型循环本身没有上限，写错 prompt 就能让它反复调工具。
   ModelCallLimitMiddleware 管调用次数，recursion_limit 管图层面的兜底，
   两者管的不是同一类失效，不能用一个推算出另一个。但**顺序是有要求的**：
   兜底必须比调用上限宽，否则用户拿到的是一句报错而不是答案。
   一次模型调用在图上要走好几步，这个换算见 AgentRunner.recursion_limit。

会话持久化按 thread_id 存取，存储落在本机的一个 SQLite 文件里
（AsyncSqliteSaver，位置见 config.AGENT_SESSION_DB）。选它而不是 Postgres 是
为了不给部署再添一个服务进程，与 feedback.py 同一个权衡；代价是写入串行、
库文件单进程独占，要多副本共享会话就得换 Postgres 版的 saver。

落盘本身是一个有代价的决定：「文档默认不出本机」是本项目的第一条设计约束，
而对话内容比文档更敏感。原先用内存存储，重启即清空，也就不必回答
「存多久、谁能读、怎么删」——现在必须回答了。因此开关默认关闭，
关着的时候连库文件都不会创建；打开它是一次显式的部署选择。
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional, Union

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

import config
import observability
from rag_engine import RAGEngine

from .tools import build_tools

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个基于知识库的文档助手。你有三个取材工具，请自己决定用哪个、用几次。

取材策略：
- 绝大多数问题从 search_docs 开始。
- 需要某一节的完整内容（如完整操作步骤）时，把检索结果里标注的「文档」「章节」
  原样传给 expand_section，不要指望调大 top_k 能拿全。
- 「所有出现 X 的地方」这类要求列全的问题用 find_literal，语义检索会悄悄截断。
- 闲聊、元问题（如「你能做什么」）不需要取材，直接回答。

回答要求：
- 只根据工具返回的资料回答，资料中没有的信息不要推测、不要用你自己的知识补全。
- 资料里确实没有时，直接说明知识库中没有相关内容，不要编造。
- 回答末尾另起一行写「参考来源：」，列出用到的「文档 › 章节」，多个用「；」分隔。
  没有用到任何资料时不要写这一行。
- 用中文回答，使用 Markdown 排版。"""


# ======================================================================
# 模型
# ======================================================================

#: 按端点参数缓存 chat model。
#:
#: 接入层每次请求都重新装配 agent（配置可热重载），而每个 ChatOpenAI 在内网场景下
#: 会带一个自己的 httpx.AsyncClient。不缓存的话，每次请求都新建一个从不关闭的
#: 连接池——这类泄漏不会立刻报错，只会在服务跑上几天之后表现成「连接数莫名涨满」。
#: 配置变了 key 就变，自然拿到新实例，所以缓存不会掩盖热重载。
_MODEL_CACHE: dict[tuple, ChatOpenAI] = {}


def build_model(settings: dict, *, temperature: Optional[float] = None) -> ChatOpenAI:
    """
    按当前生效的端点取 LangChain chat model。

    走 config.resolve_llm 而不是直接读 settings：provider 决定用云端还是自建的
    那一组配置，这个判断只应该存在一处。内网地址用 trust_env=False 的 httpx
    客户端绕过 HTTP_PROXY，与 RAGEngine._make_client 是同一个理由。
    """
    endpoint = config.resolve_llm(settings)
    timeout = float(settings.get("request_timeout", 60))
    if temperature is None:
        temperature = float(settings.get("temperature", 0.2))
    bypass = bool(settings.get("bypass_proxy_for_internal")) and config.is_internal_host(
        endpoint.base_url
    )
    # 流式响应默认不返回 token 用量，必须显式索要。不要它就只能靠估算，
    # 而估算值不能用来对账（见 observability.Usage.source）。
    stream_usage = bool(settings.get("llm_stream_usage", True))

    key = (
        endpoint.base_url,
        endpoint.model,
        endpoint.api_key,
        temperature,
        timeout,
        bypass,
        stream_usage,
    )
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    http_async_client = None
    if bypass:
        import httpx

        logger.info("端点 %s 属内网地址，已绕过 HTTP 代理", endpoint.base_url)
        http_async_client = httpx.AsyncClient(trust_env=False, timeout=timeout)

    model = ChatOpenAI(
        model=endpoint.model,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
        http_async_client=http_async_client,
        # langchain 会把它翻成 stream_options={"include_usage": True}，
        # 用量最终落在最后一条 AIMessage 的 usage_metadata 上。
        stream_usage=stream_usage,
    )
    _MODEL_CACHE[key] = model
    return model


# ======================================================================
# 会话存储
# ======================================================================

#: 未安装落盘 saver 时的提示。与 agent.MISSING_DEPS 一样写成可照做的命令。
MISSING_SESSION_DEPS = (
    "服务端会话需要 langgraph-checkpoint-sqlite，"
    "请重新安装：pip install -r requirements-agent.txt"
)

#: 进程级单例。接入层每次请求都会重新装配 agent（配置可热重载），
#: checkpointer 若跟着重建，会话就只能活一个请求。
#:
#: 它由应用启动期注入（见 open_checkpointer 与 main.lifespan），而不是像
#: feedback.get_store() 那样惰性自建：落盘的 saver 持有一条 aiosqlite 连接，
#: 必须在**运行中的事件循环**里创建、并在进程退出时关闭。这两件事模块内部
#: 都做不到——第一次用到它的时候已经在某个请求的中途了。
_CHECKPOINTER: Optional[BaseCheckpointSaver] = None

#: 打开会话库失败的原因，供 /api/health/detail 回答「会话为什么没生效」。
_SESSION_ERROR: Optional[str] = None


@asynccontextmanager
async def open_checkpointer(
    path: Union[str, Path],
) -> AsyncIterator[BaseCheckpointSaver]:
    """
    打开落盘的会话库，装成进程级单例，退出时关连接、清单例。

    显式调一次 setup()：建表本来是惰性自动做的，但那样一来「目录没有写权限」
    这类部署错误会推迟到用户第一次提问才暴露，表现成一次莫名的 500。
    启动期的错误要在启动期报。它是幂等的，重复调用无副作用。

    换 Postgres 只需改这里的两行（依赖与连接串）：
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
    调用方与 build_agent 都不受影响——两者都只认 BaseCheckpointSaver 接口。
    """
    global _CHECKPOINTER, _SESSION_ERROR

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:  # pragma: no cover - 取决于环境是否装了该包
        raise RuntimeError(MISSING_SESSION_DEPS) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await saver.setup()
        _CHECKPOINTER = saver
        _SESSION_ERROR = None
        logger.info("会话库已就绪: %s", path)
        try:
            yield saver
        finally:
            _CHECKPOINTER = None


def default_checkpointer() -> BaseCheckpointSaver:
    """
    取会话存储。

    启动期注入过落盘实例就用它；没有则退回内存。退回而不是抛错，是因为
    评测脚本、单元测试这类没有 ASGI 生命周期的入口也要能跑多轮会话，
    而它们本来就不需要「重启后还在」。日志留一条，避免线上静默降级。
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        logger.warning("会话库未初始化，本进程的会话退回内存存储（重启即清空）")
        _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


def session_store_error() -> Optional[str]:
    """打开会话库失败的原因；一切正常时为 None。"""
    return _SESSION_ERROR


def set_session_store_error(reason: Optional[str]) -> None:
    """由接入层在启动期记下失败原因（见 main.lifespan）。"""
    global _SESSION_ERROR
    _SESSION_ERROR = reason


def reset_default_checkpointer():
    """丢弃单例。供测试隔离用例之间的会话状态。"""
    global _CHECKPOINTER, _SESSION_ERROR
    _CHECKPOINTER = None
    _SESSION_ERROR = None


# ======================================================================
# 装配
# ======================================================================

#: 一次模型调用在图上占的 step 数。实测是 5：
#: ModelCallLimitMiddleware.before_model、SummarizationMiddleware.before_model、
#: model、ModelCallLimitMiddleware.after_model，模型要工具时再加一个 tools。
#:
#: 这个数字随 middleware 列表变化——每加一个带 before/after 钩子的中间件就多
#: 一步。改动后不必手算，test_递归上限自动宽到让模型调用上限先触发 会直接失败。
_STEPS_PER_MODEL_CALL = 5

#: 收尾那一轮：ModelCallLimitMiddleware.before_model 发现超限、跳到 end，
#: 这一跳本身也要占 step。
_STEPS_TO_WIND_DOWN = 2

#: 兜底被抬高时只警告一次。它每个请求都会算一遍，逐次打日志就是刷屏。
_FLOOR_WARNED = False


def max_model_calls(settings: dict) -> int:
    """模型调用次数上限。装配与兜底推算共用，避免两处各写一遍默认值。"""
    return max(2, int(settings.get("agent_max_model_calls", 6)))


#: 最后一轮追加给模型的交代。不改 SYSTEM_PROMPT 本体，只在这一次调用上叠加。
_FINAL_ROUND_NOTE = """
【重要】本次是最后一轮，你已经没有工具可用了。请仅依据上面已经取到的资料作答。
若资料不足以回答，就直接说明「已取到的资料不足以回答，缺的是……」，
明确指出还缺什么，不要猜测、不要用自己的知识补全。
"""


class FinalAnswerOnLimitMiddleware(AgentMiddleware):
    """
    在最后一次允许的模型调用上把工具摘掉，逼模型用现有材料作答。

    为什么需要它：ModelCallLimitMiddleware 的 exit_behavior="end" 并**不是**
    「带着现有材料去作答」——它只是注入一条固定的英文提示消息然后跳到 end。
    而那条消息是中间件节点的状态更新，astream 的 token 只取 model 节点
    （摘要中间件的输出不能推给用户，见 astream 里的过滤），于是用户拿到的是
    一个**空气泡**。对使用者来说，空回答比一句报错更难理解。

    做法是拦在模型调用外面，在第 limit 次调用时 override 掉 tools。模型没有
    工具可调，只能给出文本答案，于是：
      - 答案照常从 model 节点流出，用户看到的是正常的流式回答；
      - 不额外多花一次模型调用（这一次本来就在预算内）；
      - ModelCallLimitMiddleware 退回成纯兜底——正常情况下轮不到它跳转。

    只读 state 不写 state，所以不需要自己的 state_schema；
    run_model_call_count 由 ModelCallLimitMiddleware 的 schema 提供，
    它标了 UntrackedValue，不进 checkpoint，因此会话每一轮都是从 0 重新计数。
    """

    def __init__(self, limit: int, system_prompt: str) -> None:
        super().__init__()
        self._limit = limit
        self._final_system = SystemMessage(
            content=f"{system_prompt}\n{_FINAL_ROUND_NOTE}"
        )

    def _maybe_strip_tools(self, request):
        # 第 k 次调用之前，计数是 k-1。所以「这一次就是第 limit 次」等价于
        # 计数已经到了 limit-1。
        if request.state.get("run_model_call_count", 0) < self._limit - 1:
            return request
        logger.info("已达模型调用上限 %d，最后一轮摘掉工具强制作答", self._limit)
        return request.override(tools=[], system_message=self._final_system)

    def wrap_model_call(self, request, handler):
        return handler(self._maybe_strip_tools(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._maybe_strip_tools(request))


def build_agent(engine: RAGEngine, settings: dict, model=None, checkpointer=None):
    """
    组装 agent。settings 在装配时固定，与请求期间的热重载隔离。

    model 可注入，用于离线测试：替身只需是一个支持工具调用的 BaseChatModel。
    """
    limit = max_model_calls(settings)
    middleware = [
        # 终止条件之一：模型调用次数。没有它，一个写歪的 prompt 就能让模型
        # 反复调工具直到超时。
        #
        # exit_behavior 选 "end" 而不是 "error"：不阻断问答比硬失败更稳。
        # 但它自己的收场方式是注入一条固定提示就跳走，用户看到的是空回答，
        # 所以真正负责「带着现有材料作答」的是下面那个中间件，
        # 这一道退化成纯兜底——正常情况下轮不到它跳转。
        ModelCallLimitMiddleware(run_limit=limit, exit_behavior="end"),
        # 到了最后一轮就摘掉工具，逼模型用已取到的材料作答。理由见类文档。
        FinalAnswerOnLimitMiddleware(limit, SYSTEM_PROMPT),
        # 上下文压缩：消息数超过阈值就把早期消息折叠成摘要，保留最近若干条原文。
        # 摘要用同一个端点，但温度固定为 0——摘要要的是稳定复现，不是多样性。
        SummarizationMiddleware(
            model=model or build_model(settings, temperature=0.0),
            trigger=("messages", max(4, int(settings.get("agent_summary_trigger", 20)))),
            keep=("messages", max(2, int(settings.get("agent_summary_keep", 8)))),
        ),
    ]

    return create_agent(
        model or build_model(settings),
        tools=build_tools(engine, settings),
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer,
    )


# ======================================================================
# 全链路 trace
# ======================================================================

#: trace logger。用标准 logging 而不是 langchain 的 set_debug(True)：
#: 后者是直接 print 到 stdout 的，进不了日志文件也带不上时间戳，
#: 而 agent 要排查的恰恰是「昨天那一问为什么走错了工具」这类事后问题。
_TRACE_LOGGER = logging.getLogger("agent.trace")

#: langchain 的控制台 tracer 会往输出里塞 ANSI 颜色码。终端上好看，
#: 写进日志文件就是一堆 `ESC[32;1m`，还会让 grep 的模式匹配不上。
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_TRACE_HANDLER = None


def _build_trace_handler():
    from langchain_core.tracers.stdout import FunctionCallbackHandler

    class _TraceHandler(FunctionCallbackHandler):
        """
        把 langchain 的控制台 tracer 接到 logging 上。

        必须覆盖 _on_tool_start：上游实现写死了 `run.inputs["input"]`，
        而 LangGraph 的工具节点传进来的是结构化参数（{"query": ...}），
        于是每次工具调用都会抛一个被 callback manager 吞掉的 KeyError，
        在日志里表现成一条吓人的 WARNING、而工具入参恰恰没被记下来——
        正是排查时最想看的那一项。
        """

        name: str = "agent_trace_handler"

        def _on_tool_start(self, run) -> None:
            self.function_callback(
                f"[tool/start] [{self.get_breadcrumbs(run)}] "
                f"Entering Tool run with input:\n{run.inputs}"
            )

    return _TraceHandler(function=lambda text: _TRACE_LOGGER.debug(_ANSI.sub("", text)))


def trace_handler():
    """
    取全链路 trace 的回调处理器；未开启 AGENT_TRACE 时返回 None。

    它记录的是每次模型调用的完整 prompt、每个工具的入参与返回——
    也就是说**提问与文档原文会被完整写进日志**。这是它默认关闭的原因，
    不是因为性能。
    """
    global _TRACE_HANDLER
    if not getattr(config, "AGENT_TRACE", False):
        return None
    if _TRACE_HANDLER is None:
        _TRACE_HANDLER = _build_trace_handler()
    return _TRACE_HANDLER


# ======================================================================
# 运行器
# ======================================================================

class AgentRunner:
    """
    驱动 agent，把节点更新与 token 流合成一条事件序列。

    事件分型是必须的，不是锦上添花：固定管道下首 token 约 1.4 秒，
    agent 模式可能先跑两三轮取材，用户要盯着空白等十几秒——
    没有中间态的 agent 在用户看来和卡死没有区别。

    事件类型：
      tool      某个工具返回了        token 生成的文本片段
      plan      这一轮打算调的工具    reasoning 模型的思考过程（非答案正文）
      reset     刚才那段文本是取材前的开场白，客户端应当丢弃缓冲重新开始
      done      结束，带 stages       error     运行失败

    reasoning 单独分一型而不是混进 token：思考内容不是答案，落进反馈库、
    进 chatHistory 或被当成答案渲染都是错的。分开之后客户端可以把它收进
    一个折叠区，而调用方（评测、留痕）只认 token 就行，不必再去剥标签。

    reset 存在的理由：模型在决定调工具那一轮也可能先说一句「我查一下」，
    这段话会先于工具调用被流式推出去。不发 reset 的话它会粘在最终答案前面。
    """

    def __init__(
        self,
        engine: RAGEngine,
        settings: Optional[dict] = None,
        model=None,
        checkpointer=None,
    ):
        self._settings = settings or config.load_settings()
        self._agent = build_agent(
            engine, self._settings, model=model, checkpointer=checkpointer
        )

    @property
    def agent(self):
        """编译后的 agent 图。"""
        return self._agent

    @property
    def max_model_calls(self) -> int:
        return max_model_calls(self._settings)

    @property
    def recursion_limit(self) -> int:
        """
        图层面的兜底，与模型调用次数上限管的不是同一类失效——但**必须比它宽**。

        两者都是终止条件，区别在收场方式：模型调用上限是带着现有材料去作答，
        递归上限是直接报错中止。所以兜底一旦比它先触发，用户拿到的就不是答案
        而是一句「达到递归上限」，那条优雅收尾的路径等于永远走不到。

        它比想象中容易发生：一次模型调用在图上不是一步，而是
        _STEPS_PER_MODEL_CALL 步。默认的 6 次调用需要 32，而原先的默认值是 25,
        于是默认配置下每一次「模型停不下来」都报错收场——这正是本项目踩过的坑。

        因此这里取「配置值」与「算得出来的下限」的较大者。两个开关仍然各自独立
        可配，只是不允许配出一个让优雅收尾失效的组合。
        """
        global _FLOOR_WARNED

        configured = max(4, int(self._settings.get("agent_recursion_limit", 25)))
        floor = _STEPS_PER_MODEL_CALL * self.max_model_calls + _STEPS_TO_WIND_DOWN
        if configured < floor and not _FLOOR_WARNED:
            _FLOOR_WARNED = True
            logger.warning(
                "agent_recursion_limit=%d 太小，不足以让 agent_max_model_calls=%d "
                "优雅收尾，已按 %d 生效。调大模型调用上限时请一并调大它。",
                configured,
                self.max_model_calls,
                floor,
            )
        return max(configured, floor)

    def _config(self, session_id: Optional[str]) -> dict:
        cfg: dict = {"recursion_limit": self.recursion_limit}
        if session_id:
            cfg["configurable"] = {"thread_id": session_id}
        handler = trace_handler()
        if handler is not None:
            cfg["callbacks"] = [handler]
        return cfg

    @staticmethod
    def _inputs(question: str, history: Optional[list[dict]], session_id) -> dict:
        """
        构造本轮输入。

        有会话时只传本轮提问，历史由 checkpointer 按 thread_id 恢复；
        无会话时把前端维护的 history 一并带上（这是接 checkpointer 之前的原有行为）。
        角色白名单是一道防注入措施：只放行 user / assistant。
        """
        messages: list = []
        if not session_id:
            for turn in history or []:
                role = turn.get("role")
                content = str(turn.get("content", ""))
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))
        return {"messages": messages}

    async def astream(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        session_id: Optional[str] = None,
        collect: Optional[list[dict]] = None,
    ) -> AsyncIterator[dict]:
        """
        流式运行，逐个产出事件字典。

        collect 非空时，每次工具返回的 artifact 会被追加进去。
        做成显式的出参而不是塞进事件里，是因为 artifact 含文档原文：
        推给浏览器只会让进度区变成一屏正文，而评测又确实需要它。
        谁要谁自己递一个篮子来。
        """
        stages: dict = {
            "agent_tool_calls": 0,
            "agent_model_calls": 0,
            "agent_prompt_tokens": 0,
            "agent_completion_tokens": 0,
            # 只有每一次模型调用都如实报了用量，总数才能用来对账。
            # 有一次没报，整个总数就只是个下界——必须让读数的人知道。
            "agent_tokens_exact": True,
        }
        tools_used: list[str] = []

        # 节点耗时。LangGraph 的 updates 流在节点**完成时**推送，所以
        # 「上一次更新到这一次更新」的间隔就约等于刚结束那个节点的耗时。
        # 这是近似值（不含框架自身的调度开销），但足以回答唯一重要的那个
        # 问题：这几十秒是花在模型上还是花在检索上。
        last_update = time.perf_counter()
        # 思考内容有两种到达方式（取决于端点实现），两种都要接：
        # 独立字段走 _reasoning_of，混在正文里的 <think> 标签走这个切分器。
        thinking = _ThinkSplitter()

        try:
            async for mode, chunk in self._agent.astream(
                self._inputs(question, history, session_id),
                config=self._config(session_id),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message, meta = chunk
                    # 只推主模型节点的文本。摘要中间件也会调模型，
                    # 那段输出是内部产物，推给用户就是一段莫名其妙的英文摘要。
                    if meta.get("langgraph_node") != "model":
                        continue
                    reasoning = _reasoning_of(message)
                    if reasoning:
                        yield {"type": "reasoning", "content": reasoning}
                    for kind, piece in thinking.feed(_text_of(message)):
                        yield {"type": kind, "content": piece}
                elif mode == "updates":
                    now = time.perf_counter()
                    for node, update in (chunk or {}).items():
                        if node in ("model", "tools"):
                            key = f"agent_{node}_seconds"
                            stages[key] = round(
                                stages.get(key, 0.0) + (now - last_update), 3
                            )
                        for event in _update_events(
                            node, update, stages, tools_used, collect
                        ):
                            yield event
                    last_update = now
            # 标签没闭合时切分器手里还压着一段（它分不清那是正文还是半个标签），
            # 流结束了就没有后续能澄清，一律当正文吐出去——宁可多一段也不吞字。
            for kind, piece in thinking.flush():
                yield {"type": kind, "content": piece}
        except GraphRecursionError as exc:
            logger.error("Agent 触发递归上限: %s", exc)
            yield {
                "type": "error",
                "error": (
                    f"Agent 达到递归上限 {self.recursion_limit} 仍未结束，已中止。"
                ),
            }
            return
        except Exception as exc:
            logger.exception("Agent 运行失败")
            yield {"type": "error", "error": _explain(exc)}
            return

        stages["agent_tools_used"] = tools_used
        stages["agent_retrieval_skipped"] = not tools_used
        yield {"type": "done", "session_id": session_id, "stages": stages}

    async def run(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        非流式运行，返回 {"answer", "contexts", "stages", "events", "error"}。

        为评测而存在，两类指标都靠它产出：

          - **路由指标**（工具选择正确率、调用次数、不检索比例）从 events 统计，
            从最终答案反推是推不出来的。
          - **RAGAS 指标**（faithfulness / context_precision / context_recall）
            需要 contexts，也就是模型实际看到的那些资料。

        contexts 按 (source, heading, text) 去重：多轮取材几乎必然重复命中
        同一块——第二次的检索问句是基于第一次的材料改的，语义上更接近。
        不去重会让 context_precision 被同一段内容反复拉高或拉低。

        答案取「最后一次 reset 之后的 token」，而不是全部 token 拼起来——
        取材前的开场白不是答案的一部分。
        """
        answer: list[str] = []
        events: list[dict] = []
        artifacts: list[dict] = []
        stages: dict = {}
        error: Optional[str] = None
        async for event in self.astream(question, history, session_id, artifacts):
            events.append(event)
            kind = event["type"]
            if kind == "token":
                answer.append(event["content"])
            elif kind == "reset":
                answer.clear()
            elif kind == "done":
                stages = event["stages"]
            elif kind == "error":
                error = event["error"]

        contexts: list[dict] = []
        seen: set[tuple] = set()
        for artifact in artifacts:
            for ctx in artifact.get("contexts") or []:
                key = (ctx.get("source"), ctx.get("heading"), ctx.get("text"))
                if key in seen:
                    continue
                seen.add(key)
                contexts.append(ctx)

        return {
            "answer": "".join(answer),
            "contexts": contexts,
            "stages": stages,
            "events": events,
            "error": error,
        }


# ======================================================================
# 事件翻译
# ======================================================================

def _explain(exc: Exception) -> str:
    """
    把异常翻成一条用户能照做的提示。

    目前只特别处理一种：端点不接受 stream_options。我们为了拿到 token 用量
    默认带上了这个参数，而个别 OpenAI 兼容实现会直接返回 400——
    那个报错里只字不提「用量」，用户完全无从联想到该关哪个开关。

    固定管道能自己探测并回退（rag_engine._open_stream），agent 路径的请求
    由 langchain 构造，插不进重试，所以只能把话说明白。
    """
    text = str(exc)
    if "stream_options" in text or "include_usage" in text:
        return (
            f"{text}\n\n"
            "对话端点似乎不接受 stream_options（我们用它索取 token 用量）。"
            "请在 .env 中设置 LLM_STREAM_USAGE=false 并重启，"
            "之后 token 数会改用估算值。"
        )
    return text


def _accumulate_usage(message, stages: dict) -> None:
    """
    累计一次模型调用的 token 用量。

    usage_metadata 由 langchain 归一化过（各家端点的字段名不同），
    流式下要靠 ChatOpenAI(stream_usage=True) 才会被填上。取不到就把
    agent_tokens_exact 置假——继续累加一个残缺的总数并声称它精确，
    比直接没有这个数字更糟。
    """
    usage = getattr(message, "usage_metadata", None) or {}
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    if prompt is None and completion is None:
        stages["agent_tokens_exact"] = False
        return
    stages["agent_prompt_tokens"] = int(stages.get("agent_prompt_tokens", 0)) + int(
        prompt or 0
    )
    stages["agent_completion_tokens"] = int(
        stages.get("agent_completion_tokens", 0)
    ) + int(completion or 0)


def _text_of(message) -> str:
    """取消息里的纯文本。工具调用分片的 content 是空的，这里自然被跳过。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    # 部分模型返回分块结构（[{"type": "text", "text": ...}, ...]）
    parts = [
        b.get("text", "")
        for b in content or []
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "".join(parts)


#: 独立字段承载思考内容时用的键名。各家实现没有统一：DeepSeek 系叫
#: reasoning_content，Ollama 与部分 OpenAI 兼容网关叫 reasoning。
#: langchain 不认识这些非标准字段，原样堆在 additional_kwargs 里。
_REASONING_KEYS = ("reasoning_content", "reasoning")


def _reasoning_of(message) -> str:
    """取消息里的思考内容；端点不吐这个字段时返回空串。"""
    extra = getattr(message, "additional_kwargs", None) or {}
    for key in _REASONING_KEYS:
        value = extra.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class _ThinkSplitter:
    """
    把流式文本切成「思考」与「正文」两股。

    qwen3 这类模型在 OpenAI 兼容端点上常常不给独立字段，而是把思考直接写在
    正文里用 <think></think> 包起来。不切开的话，用户看到的答案前面会挂着
    一大段自言自语，落进反馈库和 chatHistory 的也是这段脏文本。

    难点只有一个：标签会被切片切断（"<thi" + "nk>"）。所以每次只吐出
    「确定不可能是半个标签」的那部分，剩下的尾巴留到下一片再判断。
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """喂一片文本，返回若干 (事件类型, 文本) —— 类型是 token 或 reasoning。"""
        out: list[tuple[str, str]] = []
        if not text:
            return out
        self._pending += text
        while self._pending:
            tag = self._CLOSE if self._inside else self._OPEN
            index = self._pending.find(tag)
            if index >= 0:
                self._emit(out, self._pending[:index])
                self._pending = self._pending[index + len(tag) :]
                self._inside = not self._inside
                continue
            hold = _tag_prefix_len(self._pending, tag)
            if hold:
                self._emit(out, self._pending[:-hold])
                self._pending = self._pending[-hold:]
            else:
                self._emit(out, self._pending)
                self._pending = ""
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        """流结束，把压着的尾巴吐干净。"""
        out: list[tuple[str, str]] = []
        self._emit(out, self._pending)
        self._pending = ""
        return out

    def _emit(self, out: list[tuple[str, str]], text: str) -> None:
        if text:
            out.append(("reasoning" if self._inside else "token", text))


def _tag_prefix_len(text: str, tag: str) -> int:
    """text 末尾有多少个字符可能是 tag 的开头（可能被下一片补全）。"""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def _update_events(
    node: str,
    update,
    stages: dict,
    tools_used: list[str],
    collect: Optional[list[dict]] = None,
):
    """把一次节点更新翻成对外事件，并顺带累计 stages 与 artifact。"""
    messages = (update or {}).get("messages") or [] if isinstance(update, dict) else []

    if node == "model":
        stages["agent_model_calls"] = int(stages.get("agent_model_calls", 0)) + 1
        calls = []
        for message in messages:
            _accumulate_usage(message, stages)
            calls.extend(getattr(message, "tool_calls", None) or [])
        if calls:
            # 这一轮要去取材，说明刚才流出去的文本是开场白而不是答案
            yield {"type": "reset"}
            yield {"type": "plan", "tools": [c.get("name", "") for c in calls]}
            stages["agent_tool_calls"] = int(stages.get("agent_tool_calls", 0)) + len(
                calls
            )

    elif node == "tools":
        for message in messages:
            name = getattr(message, "name", "") or "(未知工具)"
            tools_used.append(name)
            summary = _text_of(message).splitlines()

            # artifact 是工具挂在 ToolMessage 上的结构化副本（不进 prompt）。
            # ok / empty 从它读，比解析文本可靠——改一次渲染格式，
            # 基于前缀匹配的判定就会静默失效。
            artifact = getattr(message, "artifact", None)
            if isinstance(artifact, dict):
                ok = bool(artifact.get("ok", True))
                if collect is not None:
                    collect.append(artifact)
            else:  # 替身工具或旧格式：退回到按文本前缀判断
                ok = not summary or not summary[0].startswith(f"[{name} 失败]")

            if not ok:
                stages["agent_tool_failures"] = (
                    int(stages.get("agent_tool_failures", 0)) + 1
                )
            # 工具级指标在这里打而不是在接入层：接入层只收到 summary 一行字，
            # 而 ok 的判定依据（artifact）到不了那一层。
            observability.RequestRecord.observe_tool(name, ok)
            yield {
                "type": "tool",
                "tool": name,
                "ok": ok,
                # 工具返回的第一行就是设计好的一行摘要，正文不推给前端
                "summary": (summary[0] if summary else "")[:300],
            }
