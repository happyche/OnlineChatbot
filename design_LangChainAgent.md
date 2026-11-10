# LangChain Agent 层设计文档

> 本文档描述 `agent/` 这一层：它做什么、每一步怎么实现、为什么这么选。
> 目标是读完这一份就能独立维护和改造这层代码，因此每个关键机制都给出了
> 真实的调用链、真实的数据形态和真实的实测数字，而不是概念性描述。
>
> 阅读顺序建议：第 1～3 章建立整体印象，第 4 章理解 `create_agent` 生成了什么，
> 第 5 章逐模块细读，第 6 章（上下文压缩）和第 9 章（流式协议）是两个信息量最大
> 也最容易踩坑的地方。

---

## 目录

1. [这一层解决什么问题](#1-这一层解决什么问题)
2. [系统全景](#2-系统全景)
3. [一次提问的完整生命周期](#3-一次提问的完整生命周期)
4. [create_agent 到底生成了什么](#4-create_agent-到底生成了什么)
5. [模块逐个拆解](#5-模块逐个拆解)
6. [上下文压缩：完整机制](#6-上下文压缩完整机制)
7. [会话持久化](#7-会话持久化)
8. [终止条件：三道闸](#8-终止条件三道闸)
9. [流式事件协议](#9-流式事件协议)
10. [配置项全表](#10-配置项全表)
11. [可观测性与排障](#11-可观测性与排障)
12. [评测：怎么知道它准不准](#12-评测怎么知道它准不准)
13. [测试策略](#13-测试策略)
14. [设计取舍备忘](#14-设计取舍备忘)
15. [已知限制与演进方向](#15-已知限制与演进方向)

---

## 1. 这一层解决什么问题

### 1.1 固定管道的结构性缺陷

原有的 RAG 管道是一条直线，每个问题都走同样的三步：

```
问题 ──▶ 向量检索 + BM25 ──▶ RRF 融合 ──▶ 重排 ──▶ 取 top_k ──▶ 塞进 prompt ──▶ 生成
```

这条管道对「内容型单跳问题」非常有效（「REQUEST_TIMEOUT 默认多少秒」），
但有两类问题是它**结构性做不到**的——不是调参能解决的，是管道形状决定的：

| 问题类型 | 例子 | 为什么调 `top_k` 没用 |
|---------|------|---------------------|
| **要整节** | 「把故障处理流程的完整步骤给我」 | 步骤跨多个 chunk，而检索按**相似度**排序。第 3 步所在的 chunk 未必比第 8 步的更相似，调大 `top_k` 只会引入更多无关内容，仍然不保证拿全、更不保证有序 |
| **要列全** | 「影响POD启动时间的参数的默认值」 | 检索返回 `top_k` 条就停，而且**不会告诉你默认值**。一次检索只会把参数检索出来，默认值可能在其他文档 |

第二类尤其危险：它不是「回答得不好」，而是**回答得看起来很完整但其实是错的**。
这种静默失败没有异常、没有日志，用户没有任何线索知道答案不全。

### 1.2 解决思路：把「怎么取材」交给模型

与其在管道里加更多分支判断，不如把三种取材方式做成**工具**，让模型自己选：

```
                    ┌──────────────────────────────────────────┐
                    │  模型看到问题 + 三个工具的描述            │
                    │  自己决定：调哪个？调几次？还是直接答？   │
                    └──────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌─────────────┐            ┌──────────────┐           ┌──────────────┐
   │ search_docs │            │expand_section│           │ find_literal │
   │  语义检索    │            │  取整节正文   │           │  字面串枚举   │
   └─────────────┘            └──────────────┘           └──────────────┘
   复用现有三段式管道          按 (文档,章节) 直取         正文子串匹配
   解决：内容型问题            解决：要整节               解决：要列全
```

**关键设计原则：工具按「失败模式」划分，不按「功能」划分。**

这不是措辞讲究。按功能划分会得到「检索工具/浏览工具/统计工具」这种看起来整齐、
但每个工具都不知道自己为什么存在的集合。按失败模式划分意味着：
**每个工具的存在都对应一类在固定管道上必然失败的问题**，
删掉它就有一类问题回答不了——这让「要不要加第四个工具」有了明确的判断标准。

### 1.3 代价

这层不是免费的，代价必须和收益一起报：

| 维度 | 固定管道 | Agent 模式 |
|------|---------|-----------|
| LLM 调用次数 | 1 | 2 ～ 6 |
| 首 token 延迟 | ~1.4 秒 | 20 ～ 60 秒（取决于模型） |
| 实测总耗时（qwen3.8:27b，单请求） | ~15 秒 | ~48 秒 |

所以 **`AGENT_ENABLED` 默认为 `false`**，该不该开由评测数据决定。
这和项目里两个检索增强开关（混合检索、重排）是同一套逻辑：
有明确代价的能力必须可独立开闭，才能做消融对比。

---

## 2. 系统全景

### 2.1 分层架构

```
┌───────────────────────────────────────────────────────────────────────────┐
│  浏览器  static/index.html                                                 │
│  ├─ 发起 POST /api/chat（SSE）                                             │
│  ├─ 解析事件流：token 累积成答案 / plan+tool 渲染成进度轨迹                 │
│  └─ 收到 reset 就清空答案缓冲（详见第 9 章）                                │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │ Server-Sent Events
┌──────────────────────────────▼────────────────────────────────────────────┐
│  接入层  main.py                                                           │
│  ├─ POST /api/chat    按 agent_enabled 二选一：固定管道 / Agent 模式        │
│  ├─ POST /api/agent   非流式跑一次，返回完整事件轨迹（评测专用）             │
│  ├─ GET  /api/health  报告 agent 是否启用、依赖是否齐备                     │
│  └─ build_agent_runner()  每请求装配一次（配置快照，用完即弃）              │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────────────┐
│  Agent 层  agent/                                                          │
│                                                                            │
│  __init__.py   门面：is_available() / build_runner()                       │
│                惰性导入，未装 langchain 时给一条可照做的提示                 │
│       │                                                                    │
│       ▼                                                                    │
│  runner.py     ├─ build_model()      端点解析 → ChatOpenAI（带缓存）        │
│                ├─ build_agent()      create_agent + 中间件 + checkpointer   │
│                ├─ AgentRunner        驱动图，翻译成事件流                    │
│                └─ trace_handler()    全链路 trace 接入 logging              │
│       │                                                                    │
│       ▼                                                                    │
│  tools.py      ├─ build_tools()      工厂闭包，把 engine 绑进工具            │
│                ├─ 三个工具实现        永不抛异常，失败也返回可读文本          │
│                └─ estimate_tokens()  工具返回值的 token 预算与截断           │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │ 只读调用，不修改既有逻辑
┌──────────────────────────────▼────────────────────────────────────────────┐
│  既有 RAG 底座（本次改造一行未动其原有逻辑，只新增 4 个只读方法）             │
│                                                                            │
│  rag_engine.py   retrieve_with_diagnostics()  ← search_docs 用（原有）      │
│                  fetch_section()      新增 ← expand_section 用              │
│                  find_literal()       新增 ← find_literal 用                │
│                  document_exists()    新增 ← 错误诊断用                     │
│                  chunk_overlap        新增 ← 拼接整节时去重叠用              │
│                                                                            │
│  config.py       resolve_llm()  端点解析（云端/自建二选一，只此一处）        │
│  retrieval.py    BM25 + RRF + 重排                                          │
│  embeddings.py   本地/远程嵌入                                              │
│  vector_store.py Qdrant 向量库，按文档/章节的过滤下推到这里                  │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `agent/__init__.py` | 58 | 门面 + 惰性导入保护 |
| `agent/tools.py` | 393 | 三个工具 + token 预算 |
| `agent/runner.py` | 459 | 模型装配 + 中间件 + 事件流 |
| `tests/test_agent.py` | ~430 | 23 个用例，全离线 |
| `scripts/agent_smoke.py` | ~400 | 对真实模型测行为与代价 |

依赖只有两个包（`requirements-agent.txt`）：

```
langchain>=1.0,<2.0          # create_agent + 中间件
langchain-openai>=1.0,<2.0   # OpenAI 兼容端点适配
```

`langgraph` 是 `langchain` 的传递依赖，提供 `InMemorySaver` 和 `recursion_limit`，
但我们**不直接使用它的图 API**——这是这次改造的核心变化，见第 13 章。

---

## 3. 一次提问的完整生命周期

以「REQUEST_TIMEOUT 默认多少秒？」为例，下面是真实运行的完整时序
（时间取自 `logs/app.log`，模型是内网 Ollama 上的 qwen3.8:27b）：

```
时刻      参与者              动作
─────────────────────────────────────────────────────────────────────────────
 0.00s   浏览器              POST /api/chat {"question": "REQUEST_TIMEOUT 默认多少秒？"}
 0.00s   main.chat()         读 settings，agent_enabled=true
 0.01s   build_agent_runner  装配 AgentRunner（编译图，毫秒级）
 0.01s   AgentRunner.astream 构造输入 {"messages": [HumanMessage("REQUEST_TIMEOUT...")]}
         │
         ├─ 进入 LangGraph ────────────────────────────────────────────────
         │
 0.02s   ModelCallLimit      before_model：本次 run 已调 0 次 < 6，放行
         .before_model
 0.02s   Summarization       before_model：消息数 2 < 20，不触发压缩，返回 None
         .before_model
 0.03s   model 节点          调 ChatOpenAI，带上 system_prompt + 3 个工具的 schema
20.94s   ← 模型返回          content="" ，tool_calls=[search_docs(query="REQUEST_TIMEOUT 默认值 秒")]
         │
20.94s   ModelCallLimit      after_model：run_model_call_count=1
         .after_model        有 tool_calls → 路由到 tools 节点
         │
20.94s   AgentRunner         检测到 tool_calls，发出 {"type":"reset"} + {"type":"plan",...}
         │
20.95s   tools 节点          执行 search_docs（首次调用要加载嵌入模型+建 BM25+加载重排）
27.55s   ← 工具返回          "[search_docs] 检索「...」命中 5 个片段\n文档: ... ｜ 章节: ..."
         │
27.55s   AgentRunner         发出 {"type":"tool","tool":"search_docs","ok":true,...}
         │
27.56s   ModelCallLimit      before_model：1 < 6，放行
27.56s   Summarization       消息数 4 < 20，不触发
27.56s   model 节点          第二次调模型，这次消息里多了 AIMessage(tool_calls) + ToolMessage
47.58s   ← 模型返回          content="根据知识库，**REQUEST_TIMEOUT 默认为 60 秒**。\n\n参考来源：..."
                             tool_calls=[] → 循环结束
         │
47.58s   ModelCallLimit      after_model：无 tool_calls → 路由到 __end__
         .after_model
         │
         └─ 离开 LangGraph ────────────────────────────────────────────────
         │
47.60s   AgentRunner         发出 {"type":"done","stages":{...}}
47.60s   浏览器              收到 [DONE]，渲染最终答案
```

对应的 `stages`（这是 `/api/agent` 真实返回的内容）：

```json
{
  "agent_tool_calls": 1,
  "agent_model_calls": 2,
  "agent_tools_used": ["search_docs"],
  "agent_retrieval_skipped": false
}
```

**从这条时间线能读出的三件事**：

1. **瓶颈 100% 在 LLM**。两次模型调用共 41 秒，工具本身在预热后只要 3～5 毫秒。
   优化延迟的方向是换模型或降调用次数，不是优化检索。
2. **首次工具调用有 6.6 秒的冷启动**，来自懒加载的嵌入模型、BM25 索引构建、
   重排模型加载。第二次起降到毫秒级。
3. **`reset` 事件发生在工具调用之前**。因为模型这一轮虽然 `content=""`，
   但换个模型可能会先说一句「我查一下」——那段文本已经流给用户了，必须撤销。

---

## 4. create_agent 到底生成了什么

这是最容易产生误解的地方。`create_agent` 不是一个「包装器」或「执行器」，
它**编译出一张真正的 LangGraph 状态图**，中间件会变成图上的真实节点。

### 4.1 我们的调用

> 来源：`agent/runner.py` 第 163–188 行

```python
def build_agent(engine: RAGEngine, settings: dict, model=None, checkpointer=None):
    middleware = [
        ModelCallLimitMiddleware(
            run_limit=max(2, int(settings.get("agent_max_model_calls", 6))),
            exit_behavior="end",
        ),
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
```

### 4.2 编译出的真实图

下面的拓扑不是猜的，是用 `agent.get_graph()` 打印出来的：

```
                            ┌─────────────┐
                            │  __start__  │
                            └──────┬──────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │   ModelCallLimitMiddleware.before_model          │◀────────┐
        │   ─────────────────────────────────────          │         │
        │   已经调了几次模型？超过 run_limit 就不再调       │         │
        └───────────┬──────────────────────┬───────────────┘         │
                    │ 未超限                │ 超限（exit_behavior=end）│
                    ▼                      ▼                         │
        ┌──────────────────────────┐   ┌──────────┐                  │
        │ SummarizationMiddleware  │   │ __end__  │                  │
        │      .before_model       │   └──────────┘                  │
        │ ───────────────────────  │                                 │
        │ 消息数 ≥ trigger ？       │                                 │
        │  是 → 折叠早期消息为摘要  │                                 │
        │  否 → 原样放行            │                                 │
        └───────────┬──────────────┘                                 │
                    │                                                │
                    ▼                                                │
        ┌──────────────────────────┐                                 │
        │        model 节点         │  ← 真正调 LLM 的地方             │
        │  带 system_prompt          │    （流式 token 从这里产生）     │
        │  带 3 个工具的 JSON Schema │                                 │
        └───────────┬──────────────┘                                 │
                    │                                                │
                    ▼                                                │
        ┌──────────────────────────────────────────────────┐         │
        │   ModelCallLimitMiddleware.after_model           │         │
        │   ─────────────────────────────────────          │         │
        │   计数 +1，然后按模型输出路由                     │         │
        └──┬──────────────────┬───────────────────┬────────┘         │
           │ 有 tool_calls     │ 无 tool_calls     │ 已达上限          │
           ▼                  ▼                   ▼                  │
    ┌─────────────┐      ┌──────────┐       ┌──────────┐             │
    │ tools 节点   │      │ __end__  │       │ __end__  │             │
    │ 并行执行     │      └──────────┘       └──────────┘             │
    │ 所有工具调用 │                                                  │
    └──────┬──────┘                                                  │
           │                                                         │
           └─────────────────────────────────────────────────────────┘
                              带着 ToolMessage 回到开头
```

**三个必须记住的点**：

1. **中间件是节点，不是装饰器。** `ModelCallLimitMiddleware.before_model` 和
   `SummarizationMiddleware.before_model` 在图上是独立节点，有自己的执行记录、
   自己的耗时、自己的 trace 条目。这解释了为什么 `logs/app.log` 里会看到
   `chain:LangGraph > chain:SummarizationMiddleware.before_model [1ms]` 这样的行。

2. **`before_model` 每轮都跑。** 不是只在开始跑一次。上图里 tools → before_model
   的回边意味着：每次取材完回到模型之前，压缩判定和次数检查都会再做一遍。
   这正是压缩能在长对话中持续生效的原因。

3. **中间件顺序有意义。** 数组里靠前的包在外层。我们把 `ModelCallLimitMiddleware`
   放在前面，所以次数检查先于压缩——已经决定不再调模型了，就没必要再花一次
   LLM 调用去生成摘要。反过来放就会白白多一次调用。

### 4.3 图状态长什么样

`create_agent` 的状态核心只有一个字段：

```python
{"messages": [SystemMessage, HumanMessage, AIMessage, ToolMessage, AIMessage, ...]}
```

`messages` 用 `add_messages` reducer，语义是**按 id 追加或替换**。
这一点在第 6.5 节讲压缩时至关重要——摘要就是靠这个 reducer 的特殊语义实现的。

对比一下改造前手写 langgraph 时的状态（已删除的 `agent/state.py`）：

```python
# 旧实现：11 个字段，每个都要单独想清楚 reducer 语义
{
  "turns": Annotated[list, append],      "summary": str,
  "session_id": str | None,              "question": str,
  "history": list,                       "messages": Annotated[list, append],
  "traces": Annotated[list, append],     "hits": Annotated[list, merge_hits],
  "stages": Annotated[dict, merge],      "tool_rounds": int,
  "pending_calls": list, "grade_feedback": str, "stop_reason": str | None,
}
```

还需要自己造 `Reset` / `ResetDict` 两个哨兵类，来区分「累加」和「整体替换」。
这 170 行状态定义现在全部消失了——**这是改用 `create_agent` 最大的单项收益**。

---

## 5. 模块逐个拆解

### 5.1 `agent/__init__.py` — 门面与惰性导入

只有 58 行，做三件事。

**一、把「没装依赖」变成一条可照做的提示。**

```python
MISSING_DEPS = (
    "Agent 模式需要 langchain，请先安装：pip install -r requirements-agent.txt"
)

def is_available() -> bool:
    import importlib.util
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("langchain", "langchain_openai", "langgraph")
    )
```

`is_available()` 用 `find_spec` 而不是 `try: import`，因为它会被健康检查频繁调用，
`find_spec` 不执行模块代码、开销小得多。

**二、惰性导入。** `build_runner()` 内部才 `from .runner import AgentRunner`：

```python
def build_runner(engine, settings=None, model=None, with_session=False):
    try:
        from .runner import AgentRunner, default_checkpointer
    except ImportError as exc:
        raise RuntimeError(MISSING_DEPS) from exc
    return AgentRunner(
        engine, settings=settings, model=model,
        checkpointer=default_checkpointer() if with_session else None,
    )
```

这样做的收益：`agent/tools.py` 只依赖 `langchain_core`，
所以**没装完整 langchain 时，工具层的单元测试仍然能跑**。

**三、把 checkpointer 的单例逻辑挡在外面。** 调用方只需要传一个布尔
`with_session`，不需要知道 checkpointer 是什么、为什么必须是单例。

**为什么 `main.py` 里是 `import agent` 而不是顶层 import？**
因为 `main.py` 在 agent 关闭时也要能正常启动。顶层 import 会让一个
可选功能的缺失变成整个服务起不来。

---

### 5.2 `agent/tools.py` — 三个工具

#### 5.2.1 工具怎么绑定到引擎：工厂闭包

LangChain 的工具签名只能是「模型能填的参数」，但工具实现需要 `engine` 和
`settings`——这两个不能暴露给模型。解法是工厂闭包：

> 来源：`agent/tools.py` 第 331–360 行

```python
def build_tools(engine: RAGEngine, settings: dict) -> list[BaseTool]:
    def _guard(name: str, fn):
        async def run(**kwargs):
            try:
                return await fn(engine, settings, **kwargs)
            except Exception as exc:
                logger.exception("工具 %s 执行异常", name)
                return _fail(name, f"工具执行异常：{exc}", "可以换一个工具或调整参数重试")
        return run

    return [
        StructuredTool.from_function(
            coroutine=_guard("search_docs", search_docs),
            name="search_docs",
            args_schema=SearchDocsArgs,
            description="按语义检索知识库，返回最相关的若干文本片段。...",
        ),
        # ... 另外两个
    ]
```

三个设计点：

**为什么用工厂而不是模块级 `@tool` 装饰器？**
`engine` 和 `settings` 是**每次请求的快照**（配置支持热重载）。
做成模块级全局就会出现「配置改了、工具还绑着旧引擎」的撕裂状态。
闭包让工具的绑定关系和请求生命周期严格一致。

**为什么用 `coroutine=` 而不是 `func=`？**
底层检索是异步的（`await engine.retrieve_with_diagnostics(...)`）。
传给 `coroutine=` 参数后，LangChain 在 `ainvoke` 路径上会直接 await 它，
不会退化成线程池里跑同步函数。

**`_guard` 为什么必须存在？**
见下一节的失败契约。

#### 5.2.2 失败契约：永远不抛异常

这是整个工具层最重要的一条规则：

> **工具永远返回字符串，即使失败。失败时必须同时给出「怎么改」。**

理由：向上抛异常只能让整轮对话失败；返回一句可读的失败说明，
模型还能换个工具或改参数再试一次。

> 来源：`agent/tools.py` 第 136–142 行

```python
def _fail(tool: str, error: str, hint: str = "") -> str:
    """渲染一次失败。hint 必须跟着 error 一起给，只说「失败了」不说怎么改等于没说。"""
    text = f"[{tool} 失败] {error}"
    if hint:
        text += f"\n提示：{hint}"
    return text
```

实际效果对比：

```
❌ 只说失败（模型只能重试同样的参数，撞满次数上限）
   [expand_section 失败] 在 '运维手册.md' 里找不到章节 '不存在的章节'

✅ 带上可用选项（模型下一步就能选对）
   [expand_section 失败] 在 '运维手册.md' 里找不到章节 '不存在的章节'
   提示：该文档的章节有：超时设置、故障处理流程、配置核对
```

**还有一条同样重要的规则：「执行失败」与「执行成功但无结果」严格分开。**

```
执行失败      →  "[find_literal 失败] 没有名为 '不存在.md' 的文档\n提示：可用文档：..."
                 语义：参数写错了，该改参数重试

成功但无结果  →  "[find_literal] 全库没有正文包含 'ZZZ'。注意大小写敏感；..."
                 语义：库里确实没有，这是**拒答的依据**，不要重试
```

混成一个错误码的后果：模型在空结果上反复重检索，撞满次数上限之后
给出一个比直接拒答更差的答案。

#### 5.2.3 参数 schema 怎么变成模型看得懂的东西

用 Pydantic 模型声明，LangChain 自动转成 OpenAI function calling 的 JSON Schema：

> 来源：`agent/tools.py` 第 117–133 行

```python
class SearchDocsArgs(BaseModel):
    query: str = Field(description="检索用的完整问句，不要带指代词")
    top_k: Optional[int] = Field(default=None, description="返回条数，默认使用系统配置")


class ExpandSectionArgs(BaseModel):
    document: str = Field(description="文档名，抄自检索结果里的「文档」")
    section: str = Field(description="章节名，抄自检索结果里的「章节」")


class FindLiteralArgs(BaseModel):
    text: str = Field(description="要查找的字面串，如 REQUEST_TIMEOUT")
    document: Optional[str] = Field(default=None, description="限定某个文档名；留空则查全部")
```

发到模型的实际形态（从 trace 日志里能看到完整版本）：

```json
{
  "type": "function",
  "function": {
    "name": "expand_section",
    "description": "取出某个章节的完整正文。**只用于扩展 search_docs 已经命中的片段**，...",
    "parameters": {
      "type": "object",
      "properties": {
        "document": {"type": "string", "description": "文档名，抄自检索结果里的「文档」"},
        "section":  {"type": "string", "description": "章节名，抄自检索结果里的「章节」"}
      },
      "required": ["document", "section"]
    }
  }
}
```

**`description` 是 prompt 工程的一部分，不是注释。**
它是模型选择工具的唯一依据。注意 `expand_section` 的描述里写死了
「只用于扩展 search_docs 已经命中的片段」——这句话是在告诉模型
**工具之间的依赖顺序**，因为 `expand_section` 需要精确的文档名和章节名，
凭空猜是猜不中的。

同理，`find_literal` 的描述里明确写了「大小写敏感，不支持通配符或正则」——
这三条性质来自底层实现（字面子串匹配），不写出来模型就会
拿它去做模糊查找然后得到零命中。

#### 5.2.4 工具返回值的 token 预算

**为什么工具自己要管 token？** 因为 `search_docs` 老实返回 20 个候选的全文
会直接撑爆上下文窗口，而调用方（模型循环）没有能力在事后补救。

> ⚠️ **这和第 6 章的上下文压缩是两件事，不能互相替代。**
> `SummarizationMiddleware` 折叠的是**历史消息**（多轮累积的问答），
> 它管不到**单条工具返回的大小**。一次返回 5 万字，压缩中间件是来不及救的——
> 那条消息已经进了这一轮的 prompt。

**token 估算**：

> 来源：`agent/tools.py` 第 50–62 行

```python
def estimate_tokens(text: str) -> int:
    """
    估算 token 数：CJK 字符按 1 个、其余按 4 字符 1 个。

    不用 tiktoken：它的分词器对应 OpenAI 系模型，而本项目的对话端点常常指向
    Ollama 上的 qwen，词表完全不同，算出来并不更准却要多背一个依赖。
    这里需要的只是一个**单调且可复现**的估计量，宁可略微高估。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(text) - cjk + 3) // 4
```

**打包策略**——这是个有讲究的小算法：

> 来源：`agent/tools.py` 第 87–112 行

```python
def _pack_blocks(blocks, budget_tokens, sep="\n\n---\n\n") -> tuple[str, bool]:
    """
    在 token 预算内尽量多装几块，装不下就整块丢弃而不是半截截断。

    只有最后一块允许被截断——半截的文本块仍然有参考价值，
    而中间丢一块会让后面的块错位。
    """
    kept, used, truncated = [], 0, False
    for block in blocks:
        cost = estimate_tokens(block)
        if used + cost <= budget_tokens:
            kept.append(block); used += cost; continue
        remaining = budget_tokens - used
        if not kept and remaining > 0:
            kept.append(truncate_to_tokens(block, remaining))
        truncated = True
        break
    return sep.join(kept), truncated
```

图示（预算 1200 token，五个检索片段）：

```
输入：[块1:400] [块2:500] [块3:600] [块4:300] [块5:200]

累加：400 ✓ → 900 ✓ → 1500 ✗ 超了

结果：保留 块1 + 块2 （共 900 token），块3/4/5 整块丢弃
      truncated=True → 返回文本末尾追加「…（结果已按长度上限截断）」

为什么不把块3截成 300 token 塞进去？
  因为检索结果已经按相关度排过序，块3 本来就不如块1/2 重要，
  塞半个进去只会稀释注意力。只有「一块都装不下」时才截断那一块。
```

**截断必须被声明出来**，这是全项目的一条主线：把静默失败转成显式失败。
上下文被悄悄截掉一半，表现出来只是「答案质量莫名下降」，没有异常、
没有日志，是最难归因的一类问题。

两档预算：

| 配置 | 默认值 | 适用 |
|------|--------|------|
| `agent_tool_payload_tokens` | 1200 | `search_docs` / `find_literal` |
| `agent_expand_payload_tokens` | 2400 | `expand_section` |

`expand_section` 单独给更宽的预算，因为完整章节本来就比检索片段长得多。
用同一个上限会把「完整步骤」截成半截——那等于**没有完成这个工具存在的目的**。

#### 5.2.5 `search_docs`：最简单的那个

> 来源：`agent/tools.py` 第 204–226 行

```python
async def search_docs(engine, settings, query: str, top_k=None) -> str:
    query = (query or "").strip()
    if not query:
        return _fail("search_docs", "query 不能为空", "请给出完整的检索问句")

    # 不传 history：agent 模式下检索问句由模型自己写成不含指代的完整问句，
    # 再套一层基于历史的改写只会把它改回去。
    result = await engine.retrieve_with_diagnostics(query)
    hits = result["hits"]
    if top_k is not None and top_k > 0:
        hits = hits[:top_k]

    if not hits:
        return f"[search_docs] 检索「{query}」未命中任何内容，知识库里没有相关资料。"

    blocks = [f"{_label(h.source, h.heading)}\n{h.text}" for h in hits]
    payload, truncated = _pack_blocks(blocks, _payload_budget(settings))
    head = f"[search_docs] 检索「{query}」命中 {len(hits)} 个片段"
    tail = "\n…（结果已按长度上限截断）" if truncated else ""
    return f"{head}\n{payload}{tail}"
```

它**完整复用现有的三段式管道**（混合检索 + RRF + 重排），一行检索逻辑都没有重写。

注意 `_label()` 生成的标注格式：

> 来源：`agent/tools.py` 第 153–155 行

```python
def _label(source: str, heading: str) -> str:
    """片段的定位标注。写成带字段名的形式，让模型能原样抄进工具参数。"""
    return f"文档: {source} ｜ 章节: {heading or '（无标题正文）'}"
```

写成 `文档: xxx ｜ 章节: yyy` 而不是 `xxx › yyy`，是为了让模型能**原样抄**进
`expand_section` 的参数。字段名和参数名对齐，这是一条隐形的工具链接线。

不传 `history` 的决定值得展开：固定管道里有一个 query rewrite 步骤，
把「那它呢？」改写成「REQUEST_TIMEOUT 是多少？」。但在 agent 模式下，
模型看得到完整对话历史，它写出来的 `query` 本来就是完整问句。
再套一层改写是多余的一次 LLM 调用，还可能把好问句改坏。

#### 5.2.6 `expand_section`：两级匹配 + 去重叠拼接

这是三个工具里最复杂的一个。

**第一步：两级章节匹配**

```
        模型给的 (document, section)
                 │
                 ▼
    ┌────────────────────────────┐
    │ engine.fetch_section(      │   Qdrant payload 精确匹配
    │   document, section)       │   where={"source":doc, "heading":sec}
    └────────────┬───────────────┘
                 │ 空？
        ┌────────┴────────┐
        │ 命中            │ 未命中
        ▼                 ▼
    直接拼接      ┌────────────────────────────┐
                  │ engine.fetch_section(      │  取整篇文档
                  │   document)   ← 不传章节    │
                  └────────────┬───────────────┘
                               │ 空？
                      ┌────────┴────────┐
                      │ 非空            │ 空
                      ▼                 ▼
          在这篇文档内做              文档根本不存在
          章节名「包含」匹配          → 失败 + 列出全部可用文档
                      │
             ┌────────┴────────┐
             │ 命中            │ 未命中
             ▼                 ▼
          拼接              → 失败 + 列出这篇文档的所有章节
```

为什么要第二级的包含匹配？因为**模型抄标题时丢掉层级前缀是常见情况**：
真实标题是「配置说明 › 超时设置」，模型只抄了「超时设置」。
兜底的扫描范围严格限定在单篇文档内，量级可控。

还有一个性能细节：

> 来源：`agent/tools.py` 第 242–245 行

```python
    # 不预先校验文档名。校验要扫全部块的 metadata 才能算出文档清单，
    # 而正常路径（名字是从上一步检索结果里抄来的）根本用不上它。
    # 先查，查不到再解释——只有真出错时才付那份代价。
```

**第二步：去掉 chunk_overlap 造成的重复**

切分文档时相邻块刻意保留了 `chunk_overlap`（默认 50）个字符的重叠，
避免跨块语义被切断——这是**为检索服务**的。但一次性读整节时，
这段重叠就是纯粹的重复内容，既占 token 又让模型看到两遍同样的话。

```
原始切分（overlap=50）：
  块1: "第一步是确认告警的真实性，核对同一时间窗口内是否有其它关联告警同时触发。第二步是采集"
  块2: "。第二步是采集现场，包括进程状态、最近一小时的日志与当前连接数。第三步是判断"
        └──────── 重叠部分 ────────┘

_stitch 之后：
  "第一步是确认告警的真实性，核对同一时间窗口内是否有其它关联告警同时触发。第二步是采集
   现场，包括进程状态、最近一小时的日志与当前连接数。第三步是判断..."
```

实现是求「前一块的后缀」与「当前块的前缀」的最长公共长度：

> 来源：`agent/tools.py` 第 158–184 行

```python
def _overlap_len(prev: str, current: str, max_overlap: int) -> int:
    """求 prev 的后缀与 current 的前缀的最长公共长度，上限 max_overlap。"""
    limit = min(max_overlap, len(prev), len(current))
    for size in range(limit, 0, -1):
        if prev[-size:] == current[:size]:
            return size
    return 0


def _stitch(chunks: list[dict], max_overlap: int) -> str:
    parts, previous = [], ""
    for chunk in chunks:
        text = chunk["text"]
        if previous and max_overlap > 0:
            cut = _overlap_len(previous, text, max_overlap)
            if cut:
                text = text[cut:]
        parts.append(text)
        previous = chunk["text"]   # 注意：存原文，不是裁剪后的
    return "\n".join(p for p in parts if p.strip())
```

⚠️ 最后一行 `previous = chunk["text"]` 存的是**裁剪前的原文**。
如果存裁剪后的版本，下一轮比较的基准就错了，会导致连续裁剪时逐渐偏移。

`max_overlap` 从 `engine.chunk_overlap` 读，而不是让调用方自己翻 settings——
这样配置热重载后，拼接用的重叠值和实际切分时用的始终是同一个。

#### 5.2.7 `find_literal`：为什么「命中 137 处」里的 137 是真的

这个工具存在的全部意义是**不悄悄截断**，所以「总数」必须是真实的全库命中数。

实现用了**两段式查询**：

> 来源：`rag_engine.py` 第 646–680 行

```python
        where = {"source": source} if source else None
        matched = await asyncio.to_thread(
            self._state.collection.get,
            where=where,
            where_document={"$contains": text},
            include=[],          # ← 第一段：只要 id，不要正文
        )
        ids = list(matched.get("ids") or [])
        if not ids:
            return {"total": 0, "chunks": []}

        raw = await asyncio.to_thread(
            self._state.collection.get,
            ids=ids[: max(1, limit)],   # ← 第二段：只取前 limit 条的正文
            include=["documents", "metadatas"],
        )
        return {"total": len(ids), "chunks": self._as_chunks(raw)}
```

```
第一段查询  include=[]           → 拿到 137 个 id，不传输任何正文
                                   代价极小，但 total 是真实的

第二段查询  ids=ids[:40]         → 只取前 40 条的正文
                                   传输量可控

返回        "'NSRReadTimeOut' 在正文中命中 137 处（展示前 40 处）"
                                   ↑ 真实数字        ↑ 诚实声明
```

如果只做一次带 `limit` 的查询，拿到的「总数」就是被 limit 截断后的 40，
那个数字**没人说得清是什么意思**——既不是全部，也不知道差多少。

**零命中的两种成因必须区分**：

> 来源：`agent/tools.py` 第 283–320 行

```python
    if not total:
        # 零命中有两种成因，对下一步完全不同：
        # 文档名写错了（该改参数重试）vs 确实没有这个串（该据实拒答）。
        # 文档存在性用 limit=1 的查询判断，不去算全量文档清单。
        if document and not await engine.document_exists(document):
            available = [d["filename"] for d in engine.list_documents()]
            return _fail("find_literal", f"没有名为 {document!r} 的文档",
                         f"可用文档：{_format_items(available)}")
        return (f"[find_literal] 全库没有正文包含 {text!r}。"
                "注意大小写敏感；若要按语义或章节名查找请改用 search_docs。")
```

注意 `document_exists()` 为什么要单独存在：

> 来源：`rag_engine.py` 第 630–645 行

```python
    async def document_exists(self, source: str) -> bool:
        """
        单独给一个方法而不是让调用方去 list_documents()：后者要扫全部块的
        metadata 才能算出文档清单，而这里只需要知道「有没有」，
        `limit=1` 一条就够。几万个块时这是「一次全表扫描」和「一次索引命中」的差别。
        """
```

**两条必须写进工具描述的性质**：

| 性质 | 成因 | 后果 |
|------|------|------|
| 大小写敏感 | 最初来自 Chroma `$contains`；迁到 Qdrant 后改为进程内 `text in chunk`，刻意保持同一语义（Qdrant 的 `MatchText` 是分词 + 转小写，中文整段算一个词，不能用） | 对 `REQUEST_TIMEOUT`、`nokia-imshss` 这类标识符**正好是想要的** |
| 只匹配正文，不匹配标题 | 库里存的正文不含标题（标题只进 metadata） | 不是缺陷：标题本来就参与向量化，按章节名找是 `search_docs` 的职责 |

---

### 5.3 `agent/runner.py` — 装配与驱动

#### 5.3.1 `build_model()`：接现有的端点解析

项目的对话端点有云端/自建两套配置，「哪套在生效」这个判断**只应该存在一处**
（`config.resolve_llm`）。agent 如果直接读 `settings["llm_model"]`，
切到自建服务后就会把云端的模型名发到内网去。

> 来源：`agent/runner.py` 第 84–129 行

```python
def build_model(settings: dict, *, temperature: Optional[float] = None) -> ChatOpenAI:
    endpoint = config.resolve_llm(settings)          # ← 唯一的端点解析入口
    timeout = float(settings.get("request_timeout", 60))
    if temperature is None:
        temperature = float(settings.get("temperature", 0.2))
    bypass = bool(settings.get("bypass_proxy_for_internal")) and config.is_internal_host(
        endpoint.base_url
    )

    key = (endpoint.base_url, endpoint.model, endpoint.api_key,
           temperature, timeout, bypass)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    http_async_client = None
    if bypass:
        import httpx
        logger.info("端点 %s 属内网地址，已绕过 HTTP 代理", endpoint.base_url)
        http_async_client = httpx.AsyncClient(trust_env=False, timeout=timeout)

    model = ChatOpenAI(
        model=endpoint.model, base_url=endpoint.base_url, api_key=endpoint.api_key,
        temperature=temperature, timeout=timeout, max_retries=2,
        http_async_client=http_async_client,
    )
    _MODEL_CACHE[key] = model
    return model
```

**内网代理绕过**（`trust_env=False`）：企业环境普遍设置了 `HTTP_PROXY`，
而 httpx 默认读取该环境变量，于是访问内网推理服务的请求被发去代理，
返回一段 HTML 错误页。报错信息里只有 `InternalServerError` 和一段 HTML，
**极难联想到是代理问题**。

**为什么要缓存？** 接入层每次请求都重新装配 agent（配置可热重载），
而每个 `ChatOpenAI` 在内网场景下会带一个自己的 `httpx.AsyncClient`。
不缓存的话，每次请求都新建一个从不关闭的连接池——这类泄漏不会立刻报错，
只会在服务跑上几天后表现成「连接数莫名涨满」。

缓存 key 包含了全部会影响行为的参数，所以**配置一变 key 就变，
自然拿到新实例，缓存不会掩盖热重载**。

#### 5.3.2 System Prompt

> 来源：`agent/runner.py` 第 54–68 行

```python
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
```

三段的作用各不相同：

| 段落 | 作用 | 如果删掉 |
|------|------|---------|
| 取材策略 | 补充工具 description 里放不下的**跨工具决策规则** | 模型会对所有问题都用 `search_docs` |
| 「闲聊不需要取材」 | 省掉一整轮检索，对应 `agent_retrieval_skipped` 指标 | 「你好」也会触发一次检索 |
| 引用格式 | 改造前引用是代码拼接的，现在由模型生成 | 答案没有出处，RAG 的可溯源性丢失 |

⚠️ **引用来源现在依赖模型遵守指令**，不再有代码兜底。
这是从「代码拼接引用」改成「模型自己标注」的直接代价，见第 15 章。

#### 5.3.3 `AgentRunner`：把图的输出翻译成事件流

> 来源：`agent/runner.py` 第 320–367 行

```python
    async def astream(self, question, history=None, session_id=None):
        stages = {"agent_tool_calls": 0, "agent_model_calls": 0}
        tools_used: list[str] = []

        try:
            async for mode, chunk in self._agent.astream(
                self._inputs(question, history, session_id),
                config=self._config(session_id),
                stream_mode=["messages", "updates"],     # ← 关键
            ):
                if mode == "messages":
                    message, meta = chunk
                    # 只推主模型节点的文本。摘要中间件也会调模型，
                    # 那段输出是内部产物，推给用户就是一段莫名其妙的英文摘要。
                    if meta.get("langgraph_node") != "model":
                        continue
                    text = _text_of(message)
                    if text:
                        yield {"type": "token", "content": text}
                elif mode == "updates":
                    for node, update in (chunk or {}).items():
                        for event in _update_events(node, update, stages, tools_used):
                            yield event
        except GraphRecursionError as exc:
            ...
```

**`stream_mode=["messages", "updates"]` 是两条并行的信息流：**

```
 messages 流                             updates 流
 ───────────                             ──────────
 粒度：token 级                          粒度：节点级
 内容：(AIMessageChunk, metadata)        内容：{节点名: {"messages": [...]}}
 用途：流式吐字给用户                     用途：知道工具调了没、调的哪个

 ┌─ 模型开始输出 ─────────────────────────────────────────────┐
 │  messages: ("根",  {node:"model"})                          │
 │  messages: ("据",  {node:"model"})   ← 逐字推给前端          │
 │  messages: ("知识库", {node:"model"})                       │
 │  ...                                                        │
 │  updates:  {"model": {"messages":[AIMessage(完整, tool_calls)]}}
 │                                       ↑ 节点结束时才到      │
 └─────────────────────────────────────────────────────────────┘
```

**`meta["langgraph_node"] != "model"` 这个过滤为什么必须有？**
因为 `SummarizationMiddleware` 也会调模型生成摘要，那次调用的 token
同样会出现在 messages 流里，但它的 `langgraph_node` 是
`SummarizationMiddleware.before_model`。不过滤的话，用户会在答案中间
突然看到一段英文的对话摘要。

#### 5.3.4 事件翻译器 `_update_events`

> 来源：`agent/runner.py` 第 425–458 行

```python
def _update_events(node: str, update, stages: dict, tools_used: list[str]):
    """把一次节点更新翻成对外事件，并顺带累计 stages。"""
    messages = (update or {}).get("messages") or [] if isinstance(update, dict) else []

    if node == "model":
        stages["agent_model_calls"] = int(stages.get("agent_model_calls", 0)) + 1
        calls = []
        for message in messages:
            calls.extend(getattr(message, "tool_calls", None) or [])
        if calls:
            # 这一轮要去取材，说明刚才流出去的文本是开场白而不是答案
            yield {"type": "reset"}
            yield {"type": "plan", "tools": [c.get("name", "") for c in calls]}
            stages["agent_tool_calls"] += len(calls)

    elif node == "tools":
        for message in messages:
            name = getattr(message, "name", "") or "(未知工具)"
            tools_used.append(name)
            summary = _text_of(message).splitlines()
            ok = not summary or not summary[0].startswith(f"[{name} 失败]")
            if not ok:
                stages["agent_tool_failures"] = int(stages.get("agent_tool_failures", 0)) + 1
            yield {
                "type": "tool", "tool": name, "ok": ok,
                # 工具返回的第一行就是设计好的一行摘要，正文不推给前端
                "summary": (summary[0] if summary else "")[:300],
            }
```

**注意 `ok` 是怎么判断的**：靠工具返回文本的第一行是不是以 `[工具名 失败]` 开头。
这是 5.2.2 那个失败契约的直接回报——**统一的失败格式让编排层不需要
异常机制就能识别失败**。

**为什么只推第一行给前端？** 因为工具返回的第一行按设计就是一行摘要
（`[search_docs] 检索「...」命中 5 个片段`），正文是给模型看的，
推给前端只会把进度区变成一屏文档原文。

---

### 5.4 `rag_engine.py` 新增的四个只读方法

本次改造对底座**只做加法**，既有逻辑一行未改。

| 新增 | 位置 | 服务于 | 关键设计 |
|------|------|--------|---------|
| `chunk_overlap` | `rag_engine.py:317` | `expand_section` 去重叠 | 读引擎快照而非 settings，与热重载保持一致 |
| `_as_chunks()` | `rag_engine.py` | 上面三个共用 | 把向量库记录按 `(source, chunk_index)` 排序 |
| `fetch_section()` | `rag_engine.py` | `expand_section` | 过滤下推给向量库，不把语料拉进内存 |
| `document_exists()` | `rag_engine.py` | 错误诊断 | `limit=1`，避免全表扫描 |
| `find_literal()` | `rag_engine.py` | `find_literal` | 遍历全部候选计数，保证 total 真实 |

> 行号已随 2026-10-09 的向量库迁移（ChromaDB → Qdrant，见 DESIGN.md 11.1）变动，这里不再标注。

**贯穿这几个方法的一条原则：按文档/章节的过滤一律下推给向量库。**

```
❌ 拉进内存筛                        ✅ 下推给向量库
   chunks = collection.get()            collection.get(
   [c for c in chunks                     where={"source": doc})
    if c.metadata["source"] == doc]       # → Qdrant FieldCondition，
                                          #   服务端有 keyword 索引
   几万个块 = 每次调用几十 MB          代价与语料规模解耦
```

唯一的例外是 `find_literal` 的子串判断：Qdrant 的 `MatchText` 是分词匹配，
不是子串语义，所以文档过滤下推、子串判断在进程内做（见 5.2 节 `find_literal` 的性质表）。

`fetch_section` 的 where 条件构造：

```python
        where: dict = {"source": source, "heading": heading} if heading else {"source": source}
        raw = await asyncio.to_thread(self._state.collection.get, where=where)
        return self._as_chunks(raw)
```

`asyncio.to_thread` 是因为 qdrant-client 是同步客户端。
不包一层的话，一次检索会阻塞整个事件循环，多用户并发时互相拖死。

---

### 5.5 `main.py` — 接入层

#### 5.5.1 一个端点，两种模式

> 来源：`main.py` 第 316–358 行

```python
@api.post("/chat")
async def chat(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    settings = config.load_settings()
    agent_on = bool(settings.get("agent_enabled"))
    session_id = req.session_id if (agent_on and _sessions_enabled(settings)) else None

    async def generate():
        try:
            if agent_on:
                runner = build_agent_runner(
                    engine, settings, with_session=session_id is not None
                )
                async for event in runner.astream(req.question, req.history, session_id):
                    yield _sse(event)
            else:
                async for chunk in engine.query(req.question, req.history):
                    yield _sse({"type": "token", "content": chunk})
        except Exception as exc:
            logger.exception("问答失败")
            yield _sse({"type": "error", "error": str(exc)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**向后兼容的设计**：两种模式的正文都放在 `content` 字段里，
进度事件不带 `content`。所以只认 `content` / `error` 的旧客户端
在两种模式下都能正常工作。

`X-Accel-Buffering: no` 是给 Nginx 看的，不加的话反向代理会缓冲 SSE，
流式效果全部消失。

#### 5.5.2 为什么每请求重新装配

> 来源：`main.py` 第 299–314 行

```python
def build_agent_runner(engine: RAGEngine, settings: dict, with_session: bool = False):
    """
    每次请求重新装配而不是缓存：编译一张两节点的图是毫秒级开销，
    而缓存会引入「配置已热重载、agent 还绑着旧引擎」这种撕裂状态。
    取一次配置快照、用完即弃最简单。

    会话存储则相反，必须是进程级单例（见 agent.runner.default_checkpointer）：
    agent 可以重装，checkpointer 不能，否则每个请求都是一张空白的会话表。
    """
    import agent
    return agent.build_runner(engine, settings, with_session=with_session)
```

这是一对方向相反的决定，值得记住：

```
 agent 图        每请求重建   ← 要跟上热重载
 checkpointer    进程级单例   ← 重建就等于清空所有会话
 ChatOpenAI      按参数缓存   ← 重建会泄漏连接池
```

#### 5.5.3 `/api/agent`：为评测而存在

> 来源：`main.py` 第 360–385 行

```python
@api.post("/agent")
async def agent_run(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    """
    非流式跑一次 agent，返回答案、完整事件轨迹与 stages。

    这个接口不受 agent_enabled 约束——评测要能在开关关闭时照样跑对照组。
    """
    settings = config.load_settings()
    try:
        runner = build_agent_runner(engine, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    result = await runner.run(req.question, req.history)
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return {
        "question": req.question,
        "answer": result["answer"],
        "stages": result["stages"],
        "events": [e for e in result["events"] if e["type"] != "token"],
    }
```

**把「可评估性」当架构需求。** 工具选择正确率、平均调用次数、不检索比例
这几个指标只能从完整事件序列里统计，从最终答案反推是推不出来的。

真实返回示例：

```json
{
  "question": "REQUEST_TIMEOUT 默认多少秒？",
  "answer": "根据知识库，**REQUEST_TIMEOUT 默认为 60 秒**。\n\n参考来源：docker-verify.md › Docker 验证文档 › 超时设置",
  "stages": {
    "agent_tool_calls": 1,
    "agent_model_calls": 2,
    "agent_tools_used": ["search_docs"],
    "agent_retrieval_skipped": false
  },
  "events": [
    {"type": "reset"},
    {"type": "plan", "tools": ["search_docs"]},
    {"type": "tool", "tool": "search_docs", "ok": true,
     "summary": "[search_docs] 检索「REQUEST_TIMEOUT 默认值 秒」命中 5 个片段"},
    {"type": "done", "session_id": null, "stages": {...}}
  ]
}
```

---

### 5.6 前端 `static/index.html`

只有三处改动。

**一、渲染进度轨迹**（agent 取材时用户要盯着空白等十几秒）：

```javascript
function describeAgentEvent(obj) {
  switch (obj.type) {
    case 'plan': return `· 选用工具：${(obj.tools || []).join('、')}`;
    case 'tool': return `· ${obj.tool}：${obj.summary}`;
    default:     return '';
  }
}
```

**二、处理 `reset`**：

```javascript
const obj = JSON.parse(payload);
if (obj.error) { full += `\n\n**错误:** ${obj.error}`; break; }
/* reset：刚才那段文本是模型在取材前说的开场白，不是答案 */
if (obj.type === 'reset') { full = ''; continue; }
if (obj.content) full += obj.content;
else {
  const line = describeAgentEvent(obj);
  if (line) trace.push(line);
}
```

**三、会话 id**（仅在服务端报告 `session_enabled` 时启用）：

```javascript
async function initSession() {
  const health = await (await fetch('/api/health')).json();
  if (!health.agent || !health.agent.session_enabled) return;
  sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
}
```

---

## 6. 上下文压缩：完整机制

> 这一章是文档里信息密度最高的部分。上下文压缩用的是
> `langchain.agents.middleware.SummarizationMiddleware`，
> 下面把它的**完整算法**拆开讲——包括源码层面的实现细节，
> 因为「摘要之后消息列表变成什么样」直接决定了会不会出现
> 「模型突然失忆」「工具调用对不上」这类难查的问题。

### 6.1 为什么需要压缩

Agent 的消息列表增长得比普通对话快得多：

```
普通对话一轮 = 2 条消息
  [Human: 问题] [AI: 答案]

Agent 一轮 = 4 ～ 6 条消息
  [Human: 问题]
  [AI: tool_calls=[search_docs]]         ← 空 content，但占位
  [Tool: 1200 token 的检索结果]           ← 大头
  [AI: tool_calls=[expand_section]]
  [Tool: 2400 token 的整节正文]           ← 更大的大头
  [AI: 最终答案]
```

**十轮对话就是 40～60 条消息、几万 token。** 超出窗口后的表现不是报错，
而是被服务端静默截断——最前面的 system prompt 可能就没了，
模型突然不知道自己有工具可用。

### 6.2 我们的配置

> 来源：`agent/runner.py` 第 174–178 行

```python
        SummarizationMiddleware(
            model=model or build_model(settings, temperature=0.0),
            trigger=("messages", max(4, int(settings.get("agent_summary_trigger", 20)))),
            keep=("messages", max(2, int(settings.get("agent_summary_keep", 8)))),
        ),
```

| 参数 | 我们的值 | 含义 |
|------|---------|------|
| `model` | 同一端点，**temperature=0** | 摘要要的是稳定复现，不是多样性 |
| `trigger` | `("messages", 20)` | 消息数 ≥ 20 就触发 |
| `keep` | `("messages", 8)` | 保留最近 8 条原文 |
| `token_counter` | 默认 `count_tokens_approximately` | 我们没改 |
| `trim_tokens_to_summarize` | 默认 4000 | 送去生成摘要的内容自身的上限 |

`trigger` 和 `keep` 都支持三种单位，我们选了 `messages`：

```python
("messages", 20)    # 按消息条数    ← 我们用这个，最直观可预测
("tokens", 8000)    # 按 token 总数
("fraction", 0.8)   # 按模型窗口的百分比（需要模型 profile 数据）
```

### 6.3 触发判定：`_should_summarize`

`abefore_model` 在**每次进入 model 节点前**都会被调用。第一步是判断要不要压缩：

```python
# langchain/agents/middleware/summarization.py（简化）
def _should_summarize(self, messages, total_tokens) -> bool:
    for clause in self._trigger_clauses:
        clause_met = True
        for kind, value in clause.items():
            if kind == "messages" and len(messages) < value:
                clause_met = False
                break
            if kind == "tokens":
                # 两条路：本地估算超阈值，或模型自报的 usage_metadata 超阈值
                if total_tokens < value and not self._should_summarize_based_on_reported_tokens(...):
                    clause_met = False
                    break
            ...
        if clause_met:
            return True
    return False
```

我们用 `("messages", 20)`，所以判定简化成一行：**`len(messages) >= 20`**。

> 💡 `tokens` 模式有一个额外的优点：它会优先采信**模型自己返回的
> `usage_metadata.total_tokens`**，比本地估算准。但它要求 provider
> 正确上报，而 Ollama 的上报并不总是可靠，所以我们没用。

### 6.4 切点选择：`_find_safe_cutoff`

判定要压缩之后，第二步是决定**从哪里切**。这里有一个必须处理的陷阱。

```python
def _find_safe_cutoff(self, messages, messages_to_keep: int) -> int:
    if len(messages) <= messages_to_keep:
        return 0                                    # 不够长，不压缩
    target_cutoff = len(messages) - messages_to_keep
    return self._find_safe_cutoff_point(messages, target_cutoff)
```

朴素想法是 `cutoff = len(messages) - 8`，但这会**切断 AI/Tool 配对**：

```
❌ 朴素切法（keep=8）

  索引:   0      1      2      3      4      5      6      7      8      9
       [Human][AI(tc)][Tool][AI(tc)][Tool][ AI ][Human][AI(tc)][Tool][ AI ]
                                              ↑
                                         cutoff = 10 - 8 = 2
                                         正好切在 Tool 前面！

  结果：保留的部分以 [Tool] 开头，但它对应的 [AI(tool_calls)] 被摘要掉了
        → OpenAI 兼容端点直接报错："tool message must follow an assistant
          message with tool_calls"
        → 表现为「对话到第 N 轮就 400 错误」，而且只在特定轮数触发
```

`_find_safe_cutoff_point` 的处理策略是**向前推进，宁可多摘要**：

```python
@staticmethod
def _find_safe_cutoff_point(messages, cutoff_index: int) -> int:
    """
    If the message at `cutoff_index` is a ToolMessage, search backward for the
    AIMessage containing the corresponding tool_calls and adjust the cutoff to
    include it.

    This is aggressive with summarization - if the target cutoff lands in the
    middle of tool messages, we advance past all of them (summarizing more).
    """
```

```
✅ 安全切法

  索引:   0      1      2      3      4      5      6      7      8      9
       [Human][AI(tc)][Tool][AI(tc)][Tool][ AI ][Human][AI(tc)][Tool][ AI ]
                        ↑
              目标 cutoff=2 落在 Tool 上
              → 回溯找到它的 AI(tool_calls)（索引 1）
              → 把这一对整体划进「要摘要」的部分

  最终 cutoff 落在完整的消息边界上，保留段永远以合法的消息类型开头
```

**代价**：实际保留的消息可能少于 `keep` 指定的条数。
所以 `keep` 是一个「至多保留」的软约束，不是精确值。

### 6.5 替换消息列表：`RemoveMessage` 的妙用

这是整个机制里最巧妙、也最容易看不懂的一步。

问题：`messages` 字段的 reducer 是 `add_messages`，语义是**追加**。
中间件返回什么才能让消息列表变成「摘要 + 保留段」，而不是在后面追加？

答案是一个特殊哨兵：

```python
return {
    "messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),   # ← 哨兵：先清空整个列表
        *new_messages,                           # ← 摘要（1 条 HumanMessage）
        *preserved_messages,                     # ← 保留的最近若干条原文
    ]
}
```

`add_messages` reducer 看到 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 时，
会先把现有列表整个清空，再依次追加后面的内容。

摘要本身被包装成一条 `HumanMessage`：

```python
@staticmethod
def _build_new_messages(summary: str) -> list[HumanMessage]:
    return [
        HumanMessage(
            content=f"Here is a summary of the conversation to date:\n\n{summary}",
            additional_kwargs={"lc_source": "summarization"},
        )
    ]
```

> 💡 **为什么摘要用 `HumanMessage` 而不是 `SystemMessage`？**
> 因为大多数 OpenAI 兼容端点只接受消息列表开头有一条 system 消息。
> 在中间插一条 system 会被部分实现拒绝。用 human 角色则在哪都合法。
> `additional_kwargs["lc_source"]` 是给程序识别用的标记，不影响模型。

压缩前后的完整对比：

```
━━━ 压缩前（22 条，触发阈值 20）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [0]  System   你是一个基于知识库的文档助手...
  [1]  Human    REQUEST_TIMEOUT 默认多少秒？
  [2]  AI       tool_calls=[search_docs(...)]
  [3]  Tool     [search_docs] 命中 5 个片段 ... (1200 token)
  [4]  AI       默认 60 秒。参考来源：...
  [5]  Human    那 NSR_REFRESH_TIMER 呢？
  [6]  AI       tool_calls=[search_docs(...)]
  [7]  Tool     [search_docs] 命中 3 个片段 ... (1100 token)
  ...
  [14] Human    把故障处理流程的完整步骤给我
  [15] AI       tool_calls=[search_docs(...)]
  [16] Tool     [search_docs] 命中 5 个片段 ...
  [17] AI       tool_calls=[expand_section(...)]
  [18] Tool     [expand_section] 章节「...」共 7 个片段 ... (2400 token)
  [19] AI       完整步骤如下：第一步...
  [20] Human    第三步里说的「可用区」是什么意思？
  [21] AI       tool_calls=[search_docs(...)]

                          总计 ≈ 14000 token

━━━ SummarizationMiddleware.before_model 执行 ━━━━━━━━━━━━━━━━━━━

  1. _ensure_message_ids()        给没有 id 的消息补 UUID
  2. token_counter(messages)      ≈ 14000
  3. _should_summarize()          len=22 >= 20  → True
  4. _determine_cutoff_index()    target = 22-8 = 14
                                  messages[14] 是 Human → 已是安全边界
                                  cutoff = 14
  5. _partition_messages()        summarize = [0..13]，preserve = [14..21]
  6. _acreate_summary([0..13])    ↓ 见 6.6
  7. _build_new_messages(summary)
  8. 返回 RemoveMessage + 摘要 + 保留段

━━━ 压缩后（9 条）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [0]  Human    Here is a summary of the conversation to date:
                ## SESSION INTENT
                用户在查询运维手册中的超时与故障处理配置...
                ## SUMMARY
                - REQUEST_TIMEOUT 默认 60 秒（docker-verify.md › 超时设置）
                - NSR_REFRESH_TIMER 默认 2100 秒...
                ...
  [1]  Human    把故障处理流程的完整步骤给我
  [2]  AI       tool_calls=[search_docs(...)]
  [3]  Tool     [search_docs] 命中 5 个片段 ...
  [4]  AI       tool_calls=[expand_section(...)]
  [5]  Tool     [expand_section] 章节「...」共 7 个片段 ...
  [6]  AI       完整步骤如下：第一步...
  [7]  Human    第三步里说的「可用区」是什么意思？
  [8]  AI       tool_calls=[search_docs(...)]

                          总计 ≈ 5000 token
```

⚠️ **注意 System 消息也被摘要掉了**。这不是 bug：`create_agent` 的
`system_prompt` 是在每次调模型时**重新注入**的，不依赖消息列表里那一条。

### 6.6 摘要是怎么生成的：`_create_summary`

```python
def _create_summary(self, messages_to_summarize) -> str:
    if not messages_to_summarize:
        return "No previous conversation history."

    trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
    if not trimmed_messages:
        return "Previous conversation was too long to summarize."

    # Serialize as XML so URL-based multimodal blocks remain visible in the summary
    # prompt while excluding raw message metadata from the token budget.
    formatted_messages = get_buffer_string(trimmed_messages, format="xml")

    response = self._summary_model.invoke(
        self.summary_prompt.format(messages=formatted_messages).rstrip(), ...
    )
    return response.text
```

四个步骤：

**① 先裁剪要摘要的内容本身**（`trim_tokens_to_summarize=4000`）：

```python
trim_messages(
    messages,
    max_tokens=4000,
    token_counter=self.token_counter,
    start_on="human",        # 保证裁剪后第一条是 Human，消息序列合法
    strategy="last",         # 从尾部保留（近的更重要）
    allow_partial=True,
    include_system=True,
)
# 异常时退回：messages[-15:]
```

**为什么摘要的输入也要裁剪？** 因为要摘要的部分本身可能有几万 token，
直接发过去会**在压缩这一步就撑爆窗口**——压缩机制自己成了 OOM 的来源。

**② XML 序列化**：`get_buffer_string(trimmed, format="xml")` 把消息列表
转成带标签的文本，这样多模态 URL 块仍然可见，同时把原始 metadata
排除在 token 预算之外。

**③ 调模型**。`self._summary_model` 是包过 `.with_retry` 的（最多 3 次尝试），
应对限流和超时这类瞬时错误。**重试用尽后错误会直接向上抛，而不是编一个假摘要**——
这是个好设计：假摘要会让对话静默失真，比失败更糟。

**④ 默认 prompt**。LangChain 的默认 `summary_prompt` 是一段很长的英文
「Context Extraction Assistant」指令，要求模型按四个固定小节输出：

```
## SESSION INTENT     用户的总体目标是什么
## SUMMARY            重要决定、结论、策略，以及被否决的方案和原因
## ARTIFACTS          创建/修改/访问过的文件与资源
## NEXT STEPS         还剩什么没做
```

⚠️ **这个默认 prompt 是为编码 agent 设计的，对我们这个中文文档问答场景并不贴切**
（`ARTIFACTS` 和 `NEXT STEPS` 两节基本永远是 "None"，而且输出是英文）。
这是一个明确的改进点，见第 15 章。

### 6.7 两层压缩，各管各的

这是最容易混淆的一点，用一张图说清楚：

```
                         ┌─────────────────────────────┐
                         │      模型的上下文窗口        │
                         └─────────────────────────────┘
                                      ▲
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
   ┌────────────────────┐                          ┌────────────────────────┐
   │  第一层：工具层截断  │                          │ 第二层：历史消息压缩    │
   │  agent/tools.py    │                          │ SummarizationMiddleware│
   ├────────────────────┤                          ├────────────────────────┤
   │ 管什么：            │                          │ 管什么：                │
   │   单条工具返回的大小 │                          │   多轮累积的消息总量    │
   │                    │                          │                        │
   │ 什么时候生效：       │                          │ 什么时候生效：           │
   │   工具返回的那一刻   │                          │   每次进 model 节点前   │
   │                    │                          │                        │
   │ 手段：              │                          │ 手段：                  │
   │   整块丢弃 + 末块截断│                          │   LLM 生成摘要替换       │
   │   （不调 LLM）      │                          │   （要多花一次 LLM 调用）│
   │                    │                          │                        │
   │ 配置：              │                          │ 配置：                  │
   │   TOOL_PAYLOAD_TOKENS│                         │   SUMMARY_TRIGGER      │
   │   EXPAND_PAYLOAD_TOKENS│                       │   SUMMARY_KEEP         │
   └────────────────────┘                          └────────────────────────┘

   缺了它会怎样：                                    缺了它会怎样：
   一次 search_docs 返回 20 个候选的全文             第 10 轮对话时消息列表
   → 单条消息 5 万 token                            → 累积 4 万 token
   → 这一轮直接爆窗口，压缩中间件来不及救            → 逐渐挤掉早期上下文
```

**一句话：工具层管「一次别给太多」，中间件管「累积别太多」。**

---

## 7. 会话持久化

### 7.1 机制

```
┌──────────┐   session_id    ┌─────────────┐  thread_id  ┌──────────────┐
│ 浏览器   │ ──────────────▶ │  /api/chat  │ ──────────▶ │ InMemorySaver│
│          │  crypto         │             │  config     │              │
│ localSt- │  .randomUUID()  │             │  .configur- │ {thread-1: [消息列表快照],
│ orage    │                 │             │   able      │  thread-2: [...]}
└──────────┘                 └─────────────┘             └──────────────┘
```

每次图运行结束，LangGraph 会把完整状态（就是 `messages` 列表）
以 `thread_id` 为键存进 checkpointer。下一次带同样的 `thread_id` 进来时，
状态被恢复，新的 `HumanMessage` 追加在后面。

> 来源：`agent/runner.py` 第 299–318 行

```python
    @staticmethod
    def _inputs(question: str, history, session_id) -> dict:
        """
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
```

**角色白名单是一道防注入措施**：只放行 `user` / `assistant`。
如果原样接受前端传来的 role，攻击者可以塞一条 `system` 消息覆盖指令。

### 7.2 为什么 checkpointer 必须是进程级单例

> 来源：`agent/runner.py` 第 137–151 行

```python
#: 进程级单例。接入层每次请求都会重新装配 agent（配置可热重载），
#: checkpointer 若跟着重建，会话就只能活一个请求。
_CHECKPOINTER: Optional[InMemorySaver] = None


def default_checkpointer() -> InMemorySaver:
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


def reset_default_checkpointer():
    """丢弃单例。供测试隔离用例之间的会话状态。"""
    global _CHECKPOINTER
    _CHECKPOINTER = None
```

`reset_default_checkpointer()` 存在的唯一理由是测试隔离——
单例在测试之间会渗透，导致用例顺序依赖这种最难查的问题。
测试里用 autouse fixture 每个用例前后各清一次。

### 7.3 存内存的代价与收益

**重启即清空，不落盘。** 这既是代价也是特性：
「文档默认不出本机」是本项目的第一条设计约束，而对话内容比文档更敏感。
存内存意味着没有额外的磁盘留痕，也就不需要回答
「这些对话存多久、谁能读、怎么删」这三个合规问题。

要换成持久化，装 `langgraph-checkpoint-sqlite` 并替换 checkpointer 即可，
**其余代码一行不用改**——这正是用标准 checkpointer 接口而不是自己造轮子的收益。

### 7.4 ⚠️ 安全边界

> **`session_id` 是 bearer 凭据，不是身份认证。**
> 当前实现里 id 由**前端生成**，服务端不做归属校验。
> 拿到 id 的人就是这段对话的主人。公网暴露时**必须**同时配置 `APP_API_TOKEN`。

这是一个明确的简化取舍。更严格的做法是服务端用
`secrets.token_urlsafe(32)`（256 位熵）生成 id 并维护一张登记簿，
客户端提交的未知 id 一律不采纳（否则等于允许客户端自选 `thread_id`，
猜中别人的就能读走整段对话）。当前版本为了少 200 行代码放弃了这一层。

另外 `InMemorySaver` **只增不减**，没有 TTL 也没有数量上限——
长期运行的多用户服务上这是一条内存泄漏。见第 15 章。

---

## 8. 终止条件：三道闸

模型循环本身没有上限。三道闸管的是**三种不同的失效**，
不能用其中一个推算出另一个。

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 闸 1  ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")         │
│ ──────────────────────────────────────────────────────────────────────── │
│ 管什么：模型反复调工具停不下来（prompt 写歪、工具描述有歧义）              │
│ 怎么生效：after_model 计数 +1；before_model 发现超限就直接路由到 __end__  │
│ 达到后：带着现有材料**直接作答**，不报错                                  │
│ 缺了会怎样：一个写歪的 prompt 就能让它调到请求超时                        │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ 闸 2  recursion_limit = 25（在 run config 里）                            │
│ ──────────────────────────────────────────────────────────────────────── │
│ 管什么：**图本身**的执行步数，包括中间件节点                              │
│ 怎么生效：LangGraph 内部计数，超限抛 GraphRecursionError                  │
│ 达到后：AgentRunner 捕获，转成一条可读的 error 事件                       │
│ 缺了会怎样：图路由写错成环时，一个笔误就能挂死服务                        │
│ 为什么不能用闸 1 替代：中间件节点也算步数，6 次模型调用对应的图步数        │
│                       远不止 6（每轮至少 4 个节点）                       │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ 闸 3  工具层 token 预算（见 5.2.4）                                       │
│ ──────────────────────────────────────────────────────────────────────── │
│ 管什么：单次工具返回过大                                                  │
│ 怎么生效：_pack_blocks 在返回前就截断                                     │
│ 达到后：截断并在文本里声明「结果已按长度上限截断」                        │
│ 缺了会怎样：撑爆窗口，报错或被服务端静默截断                              │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 闸 1 的单位换算：调用次数 ≠ 取材轮数

这里容易算错，而算错会让人以为闸卡得比实际紧：

> **模型调用次数 = 取材轮数 + 1**

「读上一轮的工具结果」和「决定这一轮调什么」**是同一次模型调用**——模型看到
`ToolMessage` 之后，在同一次响应里就决定了继续调工具还是作答。多出来的那 1 次
是最后生成答案的调用。`logs/requests.jsonl` 里的实际分布正是这个形状：

| `agent_model_calls` | 工具调用数 | 含义 |
|---|---|---|
| 1 | 0 | 直接作答，不取材 |
| 2 | 1 | 1 轮取材 |
| 3 | 2 | 2 轮取材 |

所以 `AGENT_MAX_MODEL_CALLS=6` 允许的是**最多 5 轮取材**，不是两轮。

**与改造前的对照**：手写图版本的 `AGENT_MAX_TOOL_ROUNDS=3` 直接以轮为单位，
所以迁移后这道闸实际放宽了约 67%。要复刻原来的语义就设 `4`（3 轮 + 1 次作答）。

**两个版本共有的盲区**：轮数/调用次数都限制不了**单轮内的工具调用个数**。
手写图的 `call_tools` 会把 `pending_calls` 全跑一遍，`create_agent` 的 tools 节点
同样并行执行一条 `AIMessage` 上的全部 `tool_calls`。真要约束工具调用总量，
得另加 `ToolCallLimitMiddleware`（它还提供 `thread_limit`，补会话级累计的空白——
闸 1 的 `run_limit` 每问重新计数）。

闸 1 的 `exit_behavior` 有两个选项，我们选了 `"end"`：

| 值 | 行为 | 我们为什么不选 |
|----|------|--------------|
| `"end"` | 停止循环，带现有材料作答 | ✅ 选了这个。不阻断问答比硬失败更稳 |
| `"error"` | 抛异常 | 用户看到的是一次彻底失败的对话，而模型本来能给出一个「基于已有资料」的答案 |

闸 2 的捕获处理：

> 来源：`agent/runner.py` 第 351–359 行

```python
        except GraphRecursionError as exc:
            logger.error("Agent 触发递归上限: %s", exc)
            yield {
                "type": "error",
                "error": (
                    f"Agent 达到递归上限 {self.recursion_limit} 仍未结束，已中止。"
                ),
            }
            return
```

单独捕获 `GraphRecursionError` 而不是让它落进通用 `except`，
是为了给出一条**说明原因**的错误信息。通用异常只能吐一个
`Recursion limit of 25 reached...` 的英文栈信息。

---

## 9. 流式事件协议

### 9.1 事件类型

| 类型 | 字段 | 语义 | 前端怎么处理 |
|------|------|------|------------|
| `token` | `content` | 一段生成的文本 | 追加到答案缓冲 |
| `reset` | — | 之前的文本是取材前的开场白 | **清空答案缓冲** |
| `plan` | `tools[]` | 这一轮决定调哪些工具 | 渲染成进度行 |
| `tool` | `tool`, `ok`, `summary` | 某个工具返回了 | 渲染成进度行 |
| `done` | `session_id`, `stages` | 结束 | 收尾 |
| `error` | `error` | 运行失败 | 显示错误 |

### 9.2 `reset` 为什么必须存在

这是整个协议里唯一一个不直观的设计。

```
问题场景：模型在决定调工具那一轮，也可能先说一句话

  时刻 t1   模型开始输出："好的，我先查一下文档。"
            → 这些 token 通过 messages 流**已经推给用户了**
  时刻 t2   同一条消息里带出 tool_calls=[search_docs]
            → updates 流才告诉我们「这一轮是去取材的」
  时刻 t3   工具返回
  时刻 t4   模型第二次输出："根据知识库，默认 60 秒。"

  不发 reset 的话，用户看到的最终答案是：
     "好的，我先查一下文档。根据知识库，默认 60 秒。"
                ↑ 这半句不该在答案里
```

```
  时间轴与事件流
  ─────────────────────────────────────────────────────────────────
  t1  ──▶ token("好的，")  token("我先")  token("查一下文档。")
                                              前端缓冲 full = "好的，我先查一下文档。"
  t2  ──▶ reset                              前端 full = ""          ← 撤销
      ──▶ plan(tools=["search_docs"])        前端 trace += "· 选用工具：search_docs"
  t3  ──▶ tool(search_docs, ok=true, ...)    前端 trace += "· search_docs：命中 5 个片段"
  t4  ──▶ token("根据")  token("知识库，")   前端 full = "根据知识库，..."
      ──▶ done(stages={...})
  ─────────────────────────────────────────────────────────────────
                                最终答案 = "根据知识库，默认 60 秒。" ✓
```

**为什么不能在服务端缓冲后再决定推不推？**
因为那就不是流式了——用户要等模型这一轮完全输出完才能看到第一个字。
`reset` 是「先推、必要时撤销」，保住了流式体验。

评测路径 `run()` 用同样的规则还原答案：

> 来源：`agent/runner.py` 第 384–398 行

```python
        async for event in self.astream(question, history, session_id):
            events.append(event)
            kind = event["type"]
            if kind == "token":
                answer.append(event["content"])
            elif kind == "reset":
                answer.clear()          # ← 同一套语义
            elif kind == "done":
                stages = event["stages"]
            elif kind == "error":
                error = event["error"]
```

### 9.3 SSE 线格式

```
data: {"type":"token","content":"根据"}

data: {"type":"reset"}

data: {"type":"plan","tools":["search_docs"]}

data: {"type":"tool","tool":"search_docs","ok":true,"summary":"[search_docs] 命中 5 个片段"}

data: {"type":"done","session_id":null,"stages":{"agent_model_calls":2,...}}

data: [DONE]
```

---

## 10. 配置项全表

### 10.1 Agent 行为

| 环境变量 | 默认 | 范围 | 作用 |
|---------|------|------|------|
| `AGENT_ENABLED` | `false` | bool | 总开关。关闭时 `/api/chat` 走固定管道 |
| `AGENT_MAX_MODEL_CALLS` | `6` | 2–20 | 闸 1。调用次数 = 取材轮数 + 1，所以 6 ≈ 最多 5 轮取材 |
| `AGENT_RECURSION_LIMIT` | `25` | 4–100 | 闸 2。图执行步数上限 |
| `AGENT_TOOL_PAYLOAD_TOKENS` | `1200` | 100–50000 | `search_docs` / `find_literal` 单次返回上限 |
| `AGENT_EXPAND_PAYLOAD_TOKENS` | `2400` | 100–50000 | `expand_section` 的独立预算 |
| `AGENT_SUMMARY_TRIGGER` | `20` | 4–200 | 消息数达到此值触发压缩 |
| `AGENT_SUMMARY_KEEP` | `8` | 2–100 | 压缩时保留最近几条原文 |
| `AGENT_SESSION_ENABLED` | `false` | bool | 服务端会话持久化 |

### 10.2 日志与排障

| 环境变量 | 默认 | 作用 |
|---------|------|------|
| `LOG_LEVEL` | `INFO` | `DEBUG` 会把 httpx、qdrant-client 的内部动作也打出来 |
| `LOG_FILE` | 空 | 留空只输出控制台。设了就滚动落盘 |
| `LOG_MAX_MB` | `20` | 单文件上限 |
| `LOG_BACKUPS` | `3` | 保留份数 |
| `AGENT_TRACE` | `false` | 全链路 trace。⚠️ 会把提问与文档原文写进日志 |

这五项**刻意不进 `DEFAULTS`**（也就是不能从界面热改）：
日志要在第一行代码跑起来之前就定下来，改它必须重启才有意义。

### 10.3 复用的既有配置

| 配置 | Agent 层怎么用 |
|------|--------------|
| `LLM_PROVIDER` / `LLM_MODEL` / `LOCAL_LLM_*` | 经 `config.resolve_llm()` 解析成端点 |
| `TEMPERATURE` | 主模型温度（摘要模型固定 0） |
| `REQUEST_TIMEOUT` | httpx 超时 |
| `BYPASS_PROXY_FOR_INTERNAL` | 内网端点绕过 HTTP_PROXY |
| `CHUNK_OVERLAP` | `expand_section` 拼接时去重叠 |
| `TOP_K` / 混合检索 / 重排等 | `search_docs` 内部完整复用 |

---

## 11. 可观测性与排障

### 11.1 三个层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 层次 1  stages（每次问答的结构化指标）                                   │
│ 获取：POST /api/agent 的响应，或 SSE 的 done 事件                        │
│ 内容：agent_model_calls / agent_tool_calls / agent_tools_used /          │
│       agent_tool_failures / agent_retrieval_skipped                     │
│ 适合：快速判断「它选对工具了没有」                                       │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 层次 2  应用日志（logs/app.log，LOG_LEVEL=INFO）                         │
│ 内容：端点解析、httpx 请求、BM25 索引构建、模型加载、异常栈               │
│ 适合：看系统层面发生了什么                                               │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 层次 3  全链路 trace（AGENT_TRACE=true，同一个日志文件）                 │
│ 内容：每次模型调用的**完整 prompt 与响应**、每个工具的入参与返回          │
│ 适合：「为什么它选了这个工具」「模型到底看到了什么」                     │
│ ⚠️ 会把提问和文档原文完整写进磁盘，排查完建议关掉                        │
└─────────────────────────────────────────────────────────────────────────┘
```

上面三层回答的都是**单次**问答的问题（「那一问怎么了」）。下面两层回答的是
**趋势**问题（「最近怎么样」），两者不能互相替代：翻一万条 trace 也算不出
P95，而看着一条变高的曲线也定位不到是哪一问出的错。

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 层次 4  结构化请求日志（REQUEST_LOG_FILE，一行一个 JSON）                │
│ 内容：trace_id / route / 端到端与首token延迟 / 分阶段耗时 /              │
│       prompt+completion token / tokens_source / 成本 / 全部 stages      │
│ 适合：事后用 jq / pandas 做任意切分（「昨天走 agent 的题平均花了多少」）  │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 层次 5  Prometheus 指标（GET /metrics）                                  │
│ 内容：请求数、延迟直方图（端到端 / 首token / 分阶段）、token、费用、      │
│       模型调用次数、工具成功失败数、不检索次数、反馈计数                 │
│ 适合：报警与看板；「该不该开 agent」要比的正是两条 route 的分位数         │
└─────────────────────────────────────────────────────────────────────────┘
```

**trace_id 是把这五层缝起来的那根线**，也是把它们和用户反馈缝起来的那根线。
每个请求分配一个，从响应头 `X-Trace-Id` 回传（SSE 流末尾另发一个 `meta`
事件做备份，因为反向代理和跨域漏配 `expose_headers` 都会让响应头静默丢失）。
它同时出现在人读日志的每一行前缀、结构化日志的字段里、和 `feedback.db`
的主键上——所以「用户说昨天下午那一问答错了」这件事第一次变得可查。

客户端传入的 `X-Trace-Id` 会被采纳（便于串联上游调用），但必须过白名单：
它会被写进日志文件，不校验就是一条日志注入，一个换行符足以伪造出一整条
不存在的记录。

### 11.1.1 token 用量：精确值与估算值必须分开

`tokens_source` 这个字段比 token 数本身更重要。

流式响应默认**不返回** usage，必须显式索要（`stream_options.include_usage`）。
拿到了就是 `"usage"`，拿不到就退回估算并标成 `"estimated"`。两者绝不混用，
而且精确值优先是硬规则——估算覆盖精确值之后 `tokens_source` 仍然显示
`"usage"`，账单口径就无声地退化了，而且没有任何迹象表明它错了。

估算用的是 `observability.estimate_tokens`，也就是工具层做 token 预算的
**同一个函数**。这一点是有意的：两边各算一套的结果是「预算说 1200、
账单说 1800」这种对不上账、又无从排查的情况。

两条路径拿用量的方式不同，因为能插手的位置不同：

| 路径 | 精确值来源 | 端点不支持时 |
|------|-----------|-------------|
| 固定管道 | 流的最后一个分片的 `chunk.usage` | `_open_stream` 自己探测 400、记住结论、去掉参数重试一次 |
| Agent | 每条 `AIMessage.usage_metadata` 逐次累加 | 请求由 langchain 构造，插不进重试，只能报错并提示关 `LLM_STREAM_USAGE` |

⚠️ 固定管道那个分片的 `choices` 是**空的**。判断条件必须是 `if chunk.choices`，
不能先取 `choices[0]` 再判断内容——后者会在最后一个分片上抛 `IndexError`，
表现成一次「答案已经流完了但请求最后失败」的诡异故障。

agent 侧还多一个 `agent_tokens_exact`：只有每一次模型调用都如实报了用量，
总数才能用来对账。有一次没报，整个总数就只是个下界，必须让读数的人知道。

### 11.1.2 分节点耗时：瓶颈到底在哪

3 节里那条「瓶颈 100% 在 LLM，工具预热后只要 3～5 毫秒」的结论原本是手工读
日志得出的。现在它是 `stages` 里的两个字段（`agent_model_seconds` /
`agent_tools_seconds`），因此是一个**能被回归发现**的事实而不是一次观察。

实现是个近似值：LangGraph 的 `updates` 流在节点**完成时**推送，所以
「上一次更新到这一次更新」的间隔就约等于刚结束那个节点的耗时。不含框架自身的
调度开销，但足以回答唯一重要的那个问题。

第一条真实数据（qwen3.8:27b，冷启动，两轮取材）就推翻了一半的原结论：

```
agent_model_seconds   11.57      ← 三次模型调用
agent_tools_seconds   13.28      ← 两次工具调用
first_token_ms        29599      ← 几乎等于端到端耗时
```

工具耗时**超过**了模型耗时——因为首次调用要加载嵌入模型、建 BM25 索引、
加载重排模型（3 节里记的那 6.6 秒冷启动）。原结论只在预热之后成立，
而用户遇到的第一个请求恰恰不是预热之后的。

`first_token_ms ≈ latency_ms` 这一条同样值得单独看：agent 要等两轮取材结束
才开始吐字，所以「首 token 延迟」这个通常用来代表体感的指标，在 agent 模式下
退化成了端到端耗时。这就是 9.1 节那条进度轨迹必须存在的量化依据。

### 11.2 trace 是怎么接的

> 来源：`agent/runner.py` 第 206–244 行

```python
def _build_trace_handler():
    from langchain_core.tracers.stdout import FunctionCallbackHandler

    class _TraceHandler(FunctionCallbackHandler):
        """
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
```

三个实现决定：

**① 为什么不用 `langchain.globals.set_debug(True)`？**
它是直接 `print` 到 stdout 的，没有时间戳、进不了日志文件。
而 agent 要排查的恰恰是「昨天那一问为什么走错了工具」这类事后问题。
接到标准 logging 上之后，trace 和应用日志同在一个文件、同一套时间戳，
可以直接按时间对齐。

**② 为什么要剥 ANSI 颜色码？**
`FunctionCallbackHandler` 的输出里带 `\x1b[32;1m` 这类转义序列。
终端上好看，写进文件就是一堆乱码，而且会让 grep 的模式匹配不上。

**③ 挂载方式**是按请求挂进 run config，而不是全局安装：

> 来源：`agent/runner.py` 第 290–297 行

```python
    def _config(self, session_id: Optional[str]) -> dict:
        cfg: dict = {"recursion_limit": self.recursion_limit}
        if session_id:
            cfg["configurable"] = {"thread_id": session_id}
        handler = trace_handler()
        if handler is not None:
            cfg["callbacks"] = [handler]
        return cfg
```

### 11.3 实战：一次排障的完整过程

**现象**：客户端等了 400 秒超时。

**第一步**，看每次调用的耗时：

```powershell
Select-String -Path logs/app.log -Pattern "llm:ChatOpenAI\] \[.+s\]" -Encoding UTF8
```

```
LLM  124.26s
LLM  162.29s
LLM   57.60s
LLM   73.65s
LLM   89.05s
```

**第二步**，看工具耗时：

```powershell
Select-String -Path logs/app.log -Pattern "tool:\w+\] \[\d+ms\]" -Encoding UTF8
```

```
TOOL find_literal     4ms
TOOL find_literal     4ms
TOOL expand_section   5ms
TOOL find_literal     3ms
```

**结论**：瓶颈 100% 在 LLM（每次 57–162 秒），工具本身 3–5 毫秒。
按 `AGENT_MAX_MODEL_CALLS=6` 算，最坏情况接近十分钟。
优化方向是换小模型或把上限调到 4，**不是**优化检索。

### 11.4 常用命令

```powershell
# 实时跟
Get-Content logs/app.log -Wait -Tail 50 -Encoding UTF8

# 这一问用了哪些工具
Select-String -Path logs/app.log -Pattern "tool/start|tool/end" -Encoding UTF8

# 模型每次看到的完整 prompt
Select-String -Path logs/app.log -Pattern "llm/start" -Context 0,40 -Encoding UTF8

# 不想翻日志：直接要结构化结果
curl -X POST http://127.0.0.1:8001/api/agent -H "Content-Type: application/json" `
     -d '{"question":"..."}'

# 按 trace_id 把一次问答的所有痕迹捞出来（人读日志 + 结构化日志）
Select-String -Path logs/app.log,logs/requests.jsonl -Pattern "38cf5f7f" -Encoding UTF8

# 结构化日志的常用切分（需要 jq）
#   走 agent 且慢于 30 秒的
Get-Content logs/requests.jsonl | jq 'select(.route=="agent" and .latency_ms>30000)'
#   估算而非精确的 token 占比 —— 高说明端点没在报 usage，成本数字不可用来对账
Get-Content logs/requests.jsonl | jq -s 'group_by(.tokens_source)|map({(.[0].tokens_source):length})'
#   按路径看平均耗时与 token
Get-Content logs/requests.jsonl | jq -s 'group_by(.route)|map({route:.[0].route,n:length,
  avg_ms:(map(.latency_ms)|add/length),avg_tok:(map(.total_tokens)|add/length)})'

# 指标（设了 APP_API_TOKEN 就要带同一个头）
curl http://127.0.0.1:8001/metrics | Select-String "^rag_"

# 满意度与分路径代价
curl http://127.0.0.1:8001/api/feedback/stats
```

### 11.5 对真实模型测代价

```bash
# 内置夹具（自带一篇能把三个工具都逼出来的文档，跑完自动删除）
python scripts/agent_smoke.py --base-url http://127.0.0.1:8001

# 调参前后对照
python scripts/agent_smoke.py --json eval/agent-calls6.json
# 改完 .env 重启
python scripts/agent_smoke.py --json eval/agent-calls4.json --baseline eval/agent-calls6.json
```

报告的指标：工具选择正确率、平均模型调用次数、平均工具调用次数、
工具失败次数、不检索比例、首 token 中位数/最大值、总耗时中位数/最大值，
并按题型分组（均值会把类别之间的差距藏起来）。

> 判定口径是「该工具**出现在本次的工具序列里**」，而不是「它是第一个被调用的」。
> 因为 `expand_section` 按设计不能当首工具——它必须扩展 `search_docs` 已经命中的
> 片段，按首工具判这个指标对它结构性地永远是 0。

---

## 12. 评测：怎么知道它准不准

> 第 11 章的 trace 回答「它做了什么」，这一章回答「它做得对不对」。
> 两者是不同的问题：trace 能告诉你模型调了 `search_docs`，
> 但不能告诉你这道题**本该**调 `find_literal`。

### 12.1 先说结论：原有的两套评测还能跑，但它们看不见 agent

这是一个必须先讲清楚的陷阱。

```
┌──────────────────────────────────────────────────────────────────────────┐
│  retrieval_sweep.py                                                      │
│      engine = RAGEngine(settings=settings, reranker=injected)            │
│      result = await engine.retrieve_with_diagnostics(question)           │
│                                                                          │
│  ragas_eval.py                                                           │
│      engine = RAGEngine(settings=settings, reranker=injected)            │
│      result = await engine.answer(question)     ← 固定管道               │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        两个脚本都**直接构造引擎**，不走 HTTP，不 import agent。
        所以即使 AGENT_ENABLED=true，它们评的仍然是固定管道。
        没有报错、没有警告，数字看起来完全正常。
```

| 脚本 | 还能跑吗 | 实际测的是 | 还要不要留 |
|------|---------|-----------|-----------|
| `retrieval_sweep.py` | ✅ 无需改动 | `search_docs` 内部那条三段式管道的召回质量 | **要**。它是 `search_docs` 能力的上界，底层召回不准的话 agent 选对工具也救不回来 |
| `ragas_eval.py` | ✅ 无需改动 | **固定管道**的检索与生成质量 | **要**。它是 agent 的对照组 |
| `agent_smoke.py` | ✅ | agent 的**代价**（耗时、首 token、调用次数） | 要 |
| `agent_eval.py` | 新增 | agent 的**准确性**（路由 + 导出 RAGAS 样本） | — |

### 12.2 agent 引入的新失败维度：路由

固定管道只有一条路，不存在「走错路」。agent 多了一个决策点，也就多了四类失败：

```
                  ┌─ 选错工具 ──────► 用 search_docs 回答「列出所有 X」
                  │                   答案不全，但看起来很完整
                  │                   ⚠️ RAGAS 的 faithfulness 会给它**高分**
                  │                      ——每句话确实都有出处，它只是漏了一半
                  │
  模型的决策 ─────┼─ 该取材却没取 ──► 直接用自身知识编造
                  │
                  ├─ 不该取材却取了 ► 「你好」也检索一轮，白等 20 秒
                  │
                  └─ 过早停手 ──────► search_docs 命中了章节名，
                                      却没跟进 expand_section 取全
```

第一条是最危险的，因为**基于 LLM 评委的指标对它是盲的**。
所以路由必须用确定性判定单独测，不能指望 RAGAS。

### 12.3 三层评测框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 第 1 层  路由：取材方式选对了吗                                          │
│ ─────────────────────────────────────────────────────────────────────── │
│ 指标   工具选择正确率 / 不检索比例 / 平均模型调用次数 / 工具失败次数      │
│ 手段   确定性判定，**不调 LLM**，可复现、零成本、秒出                    │
│ 数据   问题集里的 expected_tool                                          │
│ 工具   scripts/agent_eval.py                                            │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 第 2 层  取材：资料够不够、准不准                                        │
│ ─────────────────────────────────────────────────────────────────────── │
│ 快指标 来源命中率（expected_sources ∩ contexts 的 source）——确定性      │
│ 慢指标 context_precision / context_recall ——RAGAS，要 LLM 评委          │
│ 数据   agent 实际看到的 contexts                                         │
│ 工具   agent_eval.py 出快指标；--dump 后交给 ragas_eval.py 出慢指标      │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 第 3 层  生成：答案忠实吗、答在点上吗                                    │
│ ─────────────────────────────────────────────────────────────────────── │
│ 快指标 关键词覆盖率 / 空手作答率 ——确定性粗筛                            │
│ 慢指标 faithfulness / answer_relevancy ——RAGAS                          │
│ 工具   同上                                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么要分快慢两档？** LLM 评委一条样本要几秒到几十秒，22 道题四个指标
就是 88 次评判。调参迭代时你需要一个几秒就能出结果的信号，
确定性指标就是干这个的；RAGAS 留给「要下结论」的那一次。

### 12.4 让 RAGAS 能评 agent：contexts 是怎么导出来的

RAGAS 四个指标的输入需求：

| 指标 | 需要 | agent 能给吗 |
|------|------|-------------|
| `context_precision` | 问题 + 参考答案 + **contexts** | 需要导出 |
| `context_recall` | 问题 + 参考答案 + **contexts** | 需要导出 |
| `faithfulness` | 问题 + 答案 + **contexts** | 需要导出 |
| `answer_relevancy` | 问题 + 答案 | ✅ 现成 |

卡点在 `contexts`：agent 的「检索结果」是三个工具返回内容的并集，
而这份数据原先只存在于给模型的那段文本里。

**解法：`response_format="content_and_artifact"`。**
这是 LangChain 工具的一个返回模式——工具返回 `(文本, 结构化数据)` 二元组，
文本进 prompt，结构化数据挂在 `ToolMessage.artifact` 上**不进 prompt**，
但留在图状态里供编排层读取。

```
   工具执行
      │
      ├──► 文本  "[search_docs] 命中 5 个片段\n文档: ... ｜ 章节: ...\n正文..."
      │           └─► 进 prompt，给模型看
      │
      └──► artifact {"tool":"search_docs", "ok":true, "empty":false,
                     "truncated":false,
                     "contexts":[{"text":..., "source":..., "heading":...}, ...],
                     "meta":{"hits":5}}
                    └─► 挂在 ToolMessage.artifact 上，不进 prompt
                        AgentRunner 收集它 → /api/agent 的 contexts 字段
```

三个实现要点：

**① contexts 必须是截断之后的那一份。** 这是正确性问题，不是细节：

> 来源：`agent/tools.py`（`_pack_items` 的文档字符串）

```text
在**结构化片段**上做打包而不是在拼好的字符串上做，是为了让
「模型实际看到了哪些内容」能被如实导出。评测 faithfulness 时，
contexts 必须是模型真正看过的那一份：拿截断前的原文去评分，
等于在评一个不存在的输入。
```

分数会虚高，而且**没有任何迹象表明它错了**。

**② 不从文本反向切分。** 按 `\n\n---\n\n` 去 split 工具返回也能拿到块，
但那样一来渲染格式和评测就隐式耦合了——改一次标注格式，评测静默失真。

**③ artifact 不推给前端。** 它含文档原文，推到 SSE 里只会让浏览器的
进度区变成一屏正文。所以 `astream()` 用一个显式的出参来传：

> 来源：`agent/runner.py`（`astream` 的 `collect` 参数）

```python
async def astream(self, question, history=None, session_id=None,
                  collect: Optional[list[dict]] = None):
    """
    collect 非空时，每次工具返回的 artifact 会被追加进去。
    做成显式的出参而不是塞进事件里，是因为 artifact 含文档原文：
    推给浏览器只会让进度区变成一屏正文，而评测又确实需要它。
    谁要谁自己递一个篮子来。
    """
```

`run()` 递篮子，SSE 路径不递。

**④ 跨轮去重。** 多轮取材几乎必然重复命中同一块——第二次的检索问句是
基于第一次的材料改的，语义上更接近。不去重会让 `context_precision`
被同一段内容反复拉高或拉低。`run()` 按 `(source, heading, text)` 去重。

### 12.5 和现有 RAGAS 的对接：一行都不用改

`ragas_eval.py` 有一个 `--samples` 模式，直接对一个 JSONL 打分，**完全不碰引擎**：

```python
if args.samples:
    all_samples = load_samples(Path(args.samples))
```

它期望的就是 `ragas.SingleTurnSample` 的字段名。所以 `agent_eval.py --dump`
只要产出这个格式，现有的整套评测设施（四个指标、评委模型切换、
`ragas_compare.py` 对照、Cursor agent 当第二评委）全部可以复用：

```
┌───────────────────┐   POST /api/agent    ┌─────────────────┐
│ questions.jsonl   │ ───────────────────► │  运行中的服务    │
│ + expected_tool   │                      │  （真实模型）    │
└───────────────────┘                      └────────┬────────┘
                                                    │ answer + contexts
                                                    │ + stages + events
                                           ┌────────▼────────┐
                                           │ agent_eval.py   │
                                           └───┬─────────┬───┘
                        确定性判定（秒出）      │         │  --dump
                    ┌───────────────────────────┘         └──────────────┐
                    ▼                                                    ▼
        ┌───────────────────────┐                        ┌──────────────────────────┐
        │ 路由 / 来源 / 关键词   │                        │ agent_samples.jsonl      │
        │ 直接打印成报告         │                        │ user_input / response /  │
        └───────────────────────┘                        │ retrieved_contexts /     │
                                                         │ reference                │
                                                         └────────────┬─────────────┘
                                                                      │
                                                      ragas_eval.py --samples
                                                                      │
                                                                      ▼
                                                    ┌──────────────────────────────┐
                                                    │ faithfulness                 │
                                                    │ answer_relevancy             │
                                                    │ context_precision            │
                                                    │ context_recall               │
                                                    └──────────────────────────────┘
```

**`retrieved_contexts` 只取正文，不带「文档: X ｜ 章节: Y」标注**——
这样和 `ragas_eval.py` 里固定管道那条路径（`[h.text for h in hits]`）格式一致，
「agent vs 固定管道」才能用同一套指标横向比较。

### 12.6 确定性指标的定义

| 指标 | 怎么算 | 为什么这么算 |
|------|--------|------------|
| 工具选择正确率 | `expected_tool in tools_used` | 看**有没有用到**，不看是不是第一个用的。`expand_section` 按设计必须跟在 `search_docs` 之后，按首工具判它结构性地永远是 0 |
| 不检索比例 | `agent_retrieval_skipped` 的占比 | 闲聊类问题省掉一整轮检索是 agent 的主要收益之一 |
| 来源命中率 | `expected_sources ∩ contexts 的 source` 非空 | 不需要 LLM，比 RAGAS 快且确定 |
| 关键词覆盖率 | `expected_keywords` 在答案里的比例 | 粗筛。细评交给 RAGAS |
| 空手作答率 | 取材为空、答案却不含拒答措辞 | **幻觉的高危信号** |
| 工具失败次数 | `agent_tool_failures` | 高说明工具描述有歧义或参数 schema 不清楚 |

「空手作答」这一项值得解释：没取到资料时，正确行为是明说查不到。
如果模型照样给出了肯定回答，那几乎一定是在用自身知识编造。
判定靠一张拒答措辞表（「没有相关」「未找到」「知识库中没有」等）。
注意它不能直接判错——闲聊类问题本来就该空手回答，所以只在
「调了工具但没取到资料」时才统计。

### 12.7 问题集格式

`eval/questions.agent.example.jsonl`，在原有字段上加了 `expected_tool`：

```json
{"question": "把「检索结果为空」这一节的完整排查步骤原原本本给我，一步都不要漏",
 "expected_tool": "expand_section",
 "type": "full-section",
 "ground_truth": "依次确认：知识库中已有文档；嵌入模型未被中途更换……",
 "expected_keywords": ["MIN_SIMILARITY"],
 "expected_sources": ["运维手册.md"]}
```

| 字段 | 必填 | 用途 |
|------|------|------|
| `question` | ✅ | — |
| `expected_tool` | — | 路由判定。写 `null` 表示**不该取材**；整个字段省略 = 不参与统计 |
| `type` | — | 分组统计。均值会把类别之间的差距藏起来 |
| `ground_truth` | — | RAGAS 的两个 reference 指标 |
| `expected_keywords` | — | 确定性粗筛 |
| `expected_sources` | — | 来源命中率 |

> ⚠️ **「没标注」与「明确标成 null」必须分开。** 都当成 `None` 的话，
> 所有没标注的题都会被算成「不该取材」，正确率瞬间虚高。
> 脚本里用一个单独的 `has_expected_tool` 标志来区分。

示例题集覆盖五类题型，比例是有意的：

| 题型 | 条数 | 期望工具 | 考什么 |
|------|------|---------|--------|
| `single-hop` / `paraphrase` | 8 | `search_docs` | 基本路由，别过度使用花哨工具 |
| `full-section` | 4 | `expand_section` | 认不认得「要整节」 |
| `enumerate` | 4 | `find_literal` | 认不认得「要列全」 |
| `chitchat` | 3 | `null` | 会不会为闲聊白跑一轮检索 |
| `unanswerable` | 3 | `search_docs` | 查不到时会不会老实拒答 |

### 12.8 ⚠️ 语料规模决定了题集的有效性

示例语料只有一篇 3.5KB 的文档，切出 11 个块。这对
`expand_section` 和 `find_literal` 是**不够**的：

```
  默认 top_k = 5，而全库只有 11 块
        │
        ▼
  一次 search_docs 就召回了接近一半的语料
        │
        ▼
  「要整节」的问题用 search_docs 也能答对
        │
        ▼
  expand_section 结构性地很难被逼出来
        │
        ▼
  量到的低分是**语料的缺陷**，不是 agent 的行为
```

这和 `agent_smoke.py` 里记录过的那个坑是同一个：内置夹具第一版只有 4 个块，
导致「工具选择正确率」这个指标量的是夹具自己的缺陷。

**所以示例题集只用来演示格式。要下结论，必须指向你自己语料上的题集**
（几百篇文档、几万个块的量级），那时 `top_k` 覆盖不了全库，三个工具的
差异才真正显现出来。

### 12.9 完整工作流

```bash
# 0) 前提：服务已启动，知识库里有对应语料
#    /api/agent 不受 AGENT_ENABLED 约束，关着也能跑对照组

# 1) 快循环：只看路由，几分钟，不花评委成本
python scripts/agent_eval.py --questions eval/questions.agent.example.jsonl

# 2) 要下结论：导出样本，交给现有 RAGAS
python scripts/agent_eval.py --questions eval/questions.hss.jsonl \
    --dump eval/agent_samples.jsonl
python scripts/ragas_eval.py --samples eval/agent_samples.jsonl

# 3) 对照组：同一批问题在固定管道上跑一次
python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
    --dump-only --output eval/baseline_samples.jsonl
python scripts/ragas_eval.py --samples eval/baseline_samples.jsonl

# 4) 调参前后对照
python scripts/agent_eval.py --json eval/agent_before.json
#    改 .env，重启
python scripts/agent_eval.py --json eval/agent_after.json \
    --baseline eval/agent_before.json

# 5) 代价（准确性之外的另一半，收益和代价必须一起报）
python scripts/agent_smoke.py --base-url http://127.0.0.1:8001
```

**第 3 步的对照组是整套评测里最重要的一步。** agent 每问多 1～3 次 LLM 调用、
延迟高一个数量级，如果 RAGAS 分数相对固定管道没有明显提升，
那结论就是这个场景不该开 agent——这正是 `AGENT_ENABLED` 默认为 `false` 的原因。

### 12.10 报告长什么样

```
========================================================================
样本 22 条
========================================================================

  【路由】agent 特有的失败模式，RAGAS 测不到
    工具选择正确率       18/22（82%）
    不检索比例           0.14
    平均模型调用次数     2.36
    工具失败次数         1

  【取材】资料对不对，不需要 LLM 评委
    来源命中率           19/19（100%）
    平均 contexts 条数   4.2
    零资料轮次           3

  【答案】粗筛，细评交给 RAGAS
    关键词覆盖率         86%
    空手作答（幻觉高危） 0/3

  按题型（均值会把类别之间的差距藏起来）
    类型                 条数  工具选择正确率
    chitchat                3            100%
    enumerate               4             50%
    full-section            4             75%
    single-hop              6            100%
    unanswerable            3            100%
```

按题型分组是必须的：上面这个例子里总正确率 82% 看着还行，
但 `enumerate` 只有 50%——一半的「列出所有 X」被 `search_docs` 接走了，
而那正是会产生「看起来完整、实际不全」答案的那类失败。
混在一起报均值会把它完全藏住。

---

## 13. 测试策略

23 个用例，**全程离线**：嵌入用 hashing，模型用按脚本回话的替身。

### 13.1 假模型

```python
class ScriptedChatModel(BaseChatModel):
    """
    按脚本依次返回预设消息的假模型。

    脚本用完之后重复最后一条，这样「模型调用次数超出预期」不会变成
    IndexError——那种报错只会让人去查测试自己的越界，而不是去看
    真正的问题：循环没有按预期停下来。
    """
    script: list = Field(default_factory=list)
    log: dict = Field(default_factory=lambda: {"calls": 0, "bound_tools": []})

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.log["bound_tools"] = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log["calls"] += 1
        self.log.setdefault("seen", []).append(messages)
        index = min(self.log["calls"] - 1, len(self.script) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])
```

`bind_tools` 必须实现——`create_agent` 会调它把工具 schema 绑上去。
顺便记下绑了哪些工具，这样「少挂一个工具」能被测出来
（这种错不会报错，只会让对应那类问题悄悄退化回语义检索）。

### 13.2 两条测试线

**工具层**（不需要模型）——重点测**失败路径**，因为工具的契约是
「永远返回可读文本，失败时带上怎么改」，这条契约破了不会抛异常，
只会让模型在错误参数上反复重试：

```
test_search_docs_命中时标注文档与章节       标注是 expand_section 的输入来源
test_search_docs_空结果不是失败             空结果是拒答依据，不是错误
test_search_docs_超预算时截断并声明         截断必须可见
test_expand_section_拼回完整章节            三步必须都在
test_expand_section_章节名只给一半也能命中   两级匹配的回退路径
test_expand_section_章节找不到时列出可用章节 失败提示必须可操作
test_expand_section_文档不存在时列出可用文档
test_find_literal_如实报告命中总数           两段式查询的意义
test_find_literal_区分文档名写错与确实没有
test_工具内部异常转成可读失败而不上抛        _guard 的契约
test_三个工具都被挂上去
```

**编排层**（用假模型）——`create_agent` 的循环是 langchain 的代码，
不需要我们测；要测的是我们接在它两端的东西：

```
test_取材一轮后作答的完整事件序列
test_取材前的开场白不会粘进答案              reset 机制
test_闲聊不取材                             agent_retrieval_skipped
test_工具失败被记进stages并照样收尾
test_同一thread_id跨轮记住上一问             checkpointer 真的生效
test_不同thread_id互不串话
test_无会话时用前端传来的history
test_模型调用次数上限能刹住循环              闸 1
test_三个工具被绑给模型
```

**接口层**：

```
test_health报告agent状态
test_agent端点返回事件轨迹
test_chat端点在agent关闭时走固定管道         默认路径不能被带偏
```

### 13.3 两个测试基建上的坑

**① 单例必须隔离**：

```python
@pytest.fixture(autouse=True)
def _isolate_checkpointer():
    """进程级 checkpointer 是单例，用例之间必须清掉，否则会话会互相渗透。"""
    reset_default_checkpointer()
    yield
    reset_default_checkpointer()
```

**② 开发者的 `.env` 会渗进测试**。`conftest.py` 里必须固定：

```python
# 同理固定 agent 开关：开着的话 /api/chat 会走 agent 路径去调真实端点，
# 表现成一个莫名其妙的 401。需要 agent 的用例自己注入假模型。
os.environ["AGENT_ENABLED"] = "false"
os.environ["AGENT_SESSION_ENABLED"] = "false"
```

这条是实际踩出来的：开发者 `.env` 里 `AGENT_ENABLED=true`，
结果 `test_chat端点在agent关闭时走固定管道` 去打了真实的云端端点，
报了一个和测试内容毫无关系的 401。

### 13.4 运行

```powershell
# 必须用项目的虚拟环境，系统 Python 缺依赖
.\.venv\Scripts\python.exe -m pytest tests/test_agent.py -q     # 23 passed
.\.venv\Scripts\python.exe -m pytest -q                         # 224 passed, 4 skipped
```

---

## 14. 设计取舍备忘

### 14.1 为什么放弃手写 langgraph 状态图

改造前的实现是一张手写的五节点状态图：

```
  plan ──▶ call_tools ──▶ grade ──▶ generate ──▶ finalize
   ▲                        │
   └────────────────────────┘  资料不够且未达上限，带着「还缺什么」回灌
```

改造后：

```
  model ⇄ tools   （create_agent 编译，中间件插在两端）
```

| 维度 | 手写 langgraph | create_agent |
|------|---------------|--------------|
| `agent/` 文件数 | 6 | 3 |
| `agent/` 代码行数 | ~2100 | ~900 |
| 测试文件大小 | 56 KB | 14 KB |
| 状态字段 | 11 个 + 2 个哨兵类 | 1 个（`messages`） |
| 每问 LLM 调用 | 2～4（含 grade） | 2～6 |

**被删掉的东西与理由**：

| 删掉的 | 原来做什么 | 为什么可以删 |
|--------|-----------|------------|
| `graph.py` 的 plan / grade 节点 | 规划取材、判定资料够不够 | 「资料够不够」本来就是模型每一轮都在做的判断。额外花一次 LLM 调用去问它，是把同一件事做了两遍 |
| `state.py`（174 行） | 11 个字段的 reducer + Reset 哨兵 | `create_agent` 只有 `messages` 一个字段，reducer 是标准的 `add_messages` |
| `context.py`（243 行） | token 预算分配、工具结果衰减 | 换成 `SummarizationMiddleware`。工具层的 token 截断保留了（两者管的不是一回事） |
| `session.py`（235 行） | SessionStore + TTL/LRU + 三个端点 | 简化成裸 `InMemorySaver`。代价见 7.4 |
| `prompts.py`（113 行） | 版本化 prompt 资源 | 只剩一条 system prompt，直接写在 `runner.py` 里 |
| `<think>` 剥离 + JSON 抠取 | grade 节点要从模型输出里解析 JSON 判定 | 没有 grade 节点了，也就不需要解析结构化输出 |

**保留下来的**：三个工具的取材逻辑与失败提示原样保留，
只是返回类型从 `ToolResult` dataclass 改成了 LangChain 工具要的字符串。

### 14.2 create_agent 仍然跑在 langgraph 上

这一点容易误解：`create_agent` **不是** langgraph 的替代品，它是建立在
langgraph 之上的一层。`InMemorySaver`、`recursion_limit`、
`GraphRecursionError` 都来自 langgraph，我们照常使用。

区别在于**我们不再直接写图**：不定义节点、不定义边、不定义 reducer。
这是「用框架的高层 API」和「用框架的底层 API」的区别，不是换框架。

### 14.3 其他关键取舍一览

| 取舍 | 选择 | 理由 |
|------|------|------|
| 工具划分依据 | 按失败模式 | 每个工具的存在都对应一类必然失败的问题，「要不要加工具」有了判断标准 |
| 工具失败处理 | 返回文本，不抛异常 | 抛异常整轮对话失败；返回文本模型还能换个工具 |
| 工具绑定 engine | 工厂闭包 | 模块级全局会和热重载撕裂 |
| LLM 端点 | 走 `config.resolve_llm` | 「哪套配置生效」只解析一处 |
| ChatOpenAI | 按参数缓存 | 每请求新建会泄漏 httpx 连接池 |
| agent 图 | 每请求重建 | 毫秒级开销，换来和热重载一致 |
| checkpointer | 进程级单例 | 重建等于清空所有会话 |
| 上下文压缩 | 用 langchain 自带 | 自己写一遍只会得到更差的版本 |
| 工具返回截断 | 自己做 | 中间件管不到单条消息的大小 |
| 引用来源 | 写进 system prompt | 让模型自己标注（代价见 15.1） |
| 流式开场白 | `reset` 事件 | 保住流式体验的同时不让开场白粘进答案 |
| 会话存储 | 内存，不落盘 | 对话比文档更敏感，没有磁盘留痕就不需要回答合规问题 |
| trace 输出 | 标准 logging | `set_debug` 直接 print，进不了文件、没有时间戳 |

### 14.4 框架选型的常见追问

14.1 讲的是「选了什么」，这一节是「被追着问的时候怎么答」。

#### 先把问题重新定义一次

「对比了 LangChain 和 LangGraph，最后选了 LangChain」这个说法本身是错的，
照它讲一定会被一句「你知道 `create_agent` 跑在 LangGraph 上吗」问住（见 14.2）。

正确的表述是：**两版都跑在 LangGraph 上，选择发生在 API 层级上**——
是自己定义节点、边与 reducer，还是用 `create_agent` 封装好的 model ⇄ tools 循环。

决策依据一句话：**图里没有任何一个节点，做的判断是模型做不到的。**
所以那张五节点图的复杂度没有换来能力，只是把工具调用循环手写了一遍。
可量化的结果是 `agent/` 从 7 个文件 1948 行降到 3 个文件 950 行（14.1 的表）。

#### 追问一：什么情况下你会选回手写图

答：当图里出现**模型不该决定、也决定不了的硬编排**时。例如

- 某个工具调用前必须人工审批
- 必须先查权限才能决定检索范围
- 要并行跑多路取材再投票
- 要支持从中间某一步恢复重跑

这类约束写进 system prompt 靠模型遵守是不可接受的，必须由图结构保证。
判别标准还是上面那句：**有没有一个节点做的判断是模型做不到或做不准的。**
本项目的 `grade` 节点不属于这一类——充分性判定模型每轮都在做，这正是它被删掉的理由。

#### 追问二：`create_agent` 不就是以前的 `AgentExecutor` 吗

不是，区别是**能不能往循环中间插东西**。

`AgentExecutor` 是一个 while 循环，插不进节点；`create_agent` 编译出真正的状态图，
中间件是图上的真实节点，有自己的执行记录与耗时（4.2 的拓扑是 `get_graph()` 打出来的）。

举一个能证明「顺序有语义」的例子：`ModelCallLimitMiddleware` 放在
`SummarizationMiddleware` 前面，因为数组靠前的包在外层——已经决定不再调模型了，
就不该再花一次 LLM 调用去生成摘要。反过来放就白多一次调用（见 4.2 第 3 点）。

#### 追问三：交给模型自己循环，怎么防止它跑飞

三道闸，管的是三类不同的失效（完整版见第 8 章）：

| 闸 | 管什么 | 失效方式 |
|----|--------|---------|
| `ModelCallLimitMiddleware` | 模型调用次数 | 功能降级：`exit_behavior="end"` 带现有材料作答 |
| `recursion_limit` | 图层面兜底 | 硬失败：`GraphRecursionError` |
| 工具内 token 截断 | 单条消息过大 | 中间件管不到这一层，只能自己做 |

前两者不能互相推算。手写图版本里专门有一条用例证明过：轮数上限设 50、
递归上限设 5，必须是中止报错而不是一直跑下去。

#### 追问四：迁移后 LLM 调用次数降了吗

**没有，上界反而从 2～4 涨到 2～6**（14.1 的表里就是这么写的）。
砍掉 `grade` 省了一次，但轮数控制权交给模型后上界更松。

收益不在调用次数，在两处：一是代码量与可维护性，
二是去掉了那次非流式判定的阻塞延迟——27B 实测单次约 38 秒，而每轮至少判定一次。

#### 必须主动讲的代价

只讲收益的迁移故事像营销，主动报代价才像工程。两条都在第 15 章：

- **引用溯源从确定性变成概率性**（15.1）。原来由代码从结构化命中拼出「参考来源」，
  现在写进 system prompt 靠模型自觉。对文档问答而言溯源错比答错更伤信任——
  用户会以为有出处就可信。
- **去重失去了着力点**。原来状态里带结构化命中，按 `doc_id` 去重、按 `chunk_overlap`
  消重叠；现在只有 `ToolMessage` 文本，而 `expand_section` 拼回整节时必然包含上一轮
  `search_docs` 已召回的片段，这段内容会重复进 prompt。

两条都**不需要回退到手写图**：引用校验可以做成 `after_model` 钩子
（拿工具返回过的「文档 › 章节」集合校验模型标注，不在集合里即幻觉）；
去重要落在 `awrap_tool_call` 上——本版本没有 `after_tools` 钩子，
工具侧的扩展点是环绕式的那一个（钩子清单见 14.5）。
保住 950 行的简洁，同时拿回确定性。

#### 三句不要说的话

| 别说 | 为什么 | 改成 |
|------|--------|------|
| 「LangGraph 太复杂了」 | 像嫌麻烦，不像做取舍 | 「复杂度没有换来能力」 |
| 「LangChain 比 LangGraph 好」 | 层级搞错，暴露理解不到位 | 「同一套栈的两层 API」 |
| 「选了业界主流」 | 对方要判断依据，不是跟风能力 | 给出上面那条判别标准 |

---

## 15. 已知限制与演进方向

### 15.1 引用来源依赖模型遵守指令

改造前引用是代码拼接的：取材完由 `RAGEngine` 组装 prompt，
末尾用代码追加「参考来源：」。现在改成写进 system prompt 让模型自己标注。

**风险**：模型可能漏标、标错、或者编一个不存在的章节名。

**可能的改进**：在 `AgentRunner` 里记录所有工具返回中出现过的
`文档 › 章节`，收尾时校验模型标注的来源是否在这个集合里，
不在就是幻觉，可以剔除或告警。

### 15.2 摘要用的是英文默认 prompt

`SummarizationMiddleware` 的默认 `summary_prompt` 是为编码 agent 设计的
「Context Extraction Assistant」，要求按 `SESSION INTENT` / `SUMMARY` /
`ARTIFACTS` / `NEXT STEPS` 四节输出。对我们这个中文文档问答场景：

- `ARTIFACTS` 和 `NEXT STEPS` 基本永远是 "None"，纯浪费 token
- 输出是英文，混进中文对话里

**改进**：传一个自定义的 `summary_prompt`，只保留「用户问过什么、
已经确认的事实与出处」两节，用中文。这是个小改动，收益明确。

### 15.3 `InMemorySaver` 没有上限

只增不减，没有 TTL 也没有数量上限。长期运行的多用户服务上这是一条内存泄漏。

**改进**：包一层带 TTL + LRU 淘汰的登记簿（改造前的 `session.py` 就是这么做的），
或者换成 `langgraph-checkpoint-sqlite` 并配合定期清理。

### 15.4 会话 id 由客户端生成

见 7.4。服务端不校验归属，拿到 id 就能读走整段对话。

**改进**：服务端用 `secrets.token_urlsafe(32)` 生成，维护一张登记簿，
客户端提交的未知 id 一律发新的而不是采纳。

### 15.5 token 消耗报不出真实值 ✅ 已解决

> 原文：对话端点返回了 `usage`，但服务端没有把它累计进 `stages`，
> 所以只能报调用次数，报不了真实 token 消耗。

现已实现，见 [11.1.1](#1111-token-用量精确值与估算值必须分开)。
`_update_events` 从 `AIMessage.usage_metadata` 逐次累加，固定管道从流的最后
一个分片取 `chunk.usage`，两条路径都带 `tokens_source` 声明数字是精确还是估算。
成本按模型名查 `LLM_PRICING` 换算，未配置单价时报 0 而不是瞎猜。

实现过程中多出来两个当初没想到的要点：

1. **流式响应默认不返回 usage**，必须显式索要 `stream_options.include_usage`，
   而个别 OpenAI 兼容实现会对这个参数返回 400。固定管道自己探测并回退；
   agent 路径的请求由 langchain 构造，插不进重试，只能把报错翻成一条
   「请设置 `LLM_STREAM_USAGE=false`」的提示。
2. **精确值和估算值必须分开标注，且精确值优先是硬规则**。估算覆盖精确值
   不会报错，只会让 `tokens_source` 照旧显示 `"usage"`——等于骗自己。

### 15.6 没有在线反馈闭环 ✅ 已解决

> 这一条是补充上去的，原文没有列。它是整套评测设施里唯一缺的一环，
> 而缺得最不显眼：RAGAS 四指标、确定性路由指标、消融开关、对照基线全都有了，
> 但**输入这些评测的题目全部由人手写**。于是「线上答错了」和「回归集里多一条
> 用例」之间隔着一个人的记性。

```
用户点 👎  →  feedback.db  →  badcase_export.py  →  questions.jsonl  →  既有评测链路
                                                     （格式见 12.7，评测侧一行未改）
```

实现要点：

| 决定 | 选择 | 理由 |
|------|------|------|
| 评价粒度 | 只有 👍 / 👎 | 星级在小样本上几乎不可用（所有人都点 4 星），而 badcase 库只需要「这条要不要进回归集」这一个比特 |
| 差评原因 | 预置选项，非自由文本 | 「38 条差评」不指向任何动作，「其中 24 条是答案不全」直接指向 `expand_section` 的路由准确率 |
| 存储 | SQLite，两张表 | 主要用法是查询（「最近所有 👎 的、enumerate 题型的、走了 agent 的」）而不是追加；两张表是因为反馈**后来**才到、而且多数请求根本没有 |
| 落库内容 | 含答案与 contexts | 缺了「当时取回了什么资料」就无法区分这是检索问题还是生成问题，而两者的修法完全不同 |
| 未知 trace_id | 照样入库，不报 404 | 留痕在流结束后才写，用户可能在那之前就点了评价。丢掉它等于惩罚手快的用户 |
| 留痕失败 | 吞掉异常只记日志 | 磁盘满了的时候用户应该照样能得到答案。观测手段自己成为故障源是这类改动最典型的翻车方式 |
| 导出的 `ground_truth` | 一律留空 | 只能由人判断——能自动生成正确答案的话就不需要这个系统了。`validate_questions.py` 会把每条都报成「缺 ground_truth」，那个列表就是待标注清单 |
| 记录上限 | 20000 条，淘汰没被评价过的最老记录 | 「只增不减」是一条磁盘泄漏（和日志要滚动、会话要 TTL 同理）；而被评价过的记录正是这张表存在的理由，不参与淘汰 |

⚠️ 它把提问、答案与文档原文写进磁盘，和 `AGENT_TRACE` 是同一类暴露面。
默认开启是因为没有反馈就没有闭环，但这仍然是一个需要显式知道的取舍。

**还没做的**：反馈本身没有身份（谁点的不知道），也没有把「同一个问题被反复
点差评」聚合成一个优先级。这两件事都要等真身份接进来之后才有意义。

### 15.7 没有做工具调用的并发

`create_agent` 的 `tools` 节点本身支持并行执行同一轮的多个工具调用，
但我们的三个工具在典型流程里是**串行依赖**的
（`search_docs` → 抄出文档章节 → `expand_section`），
所以实际很少触发并发。这不是缺陷，是工具设计决定的。

---

## 附录 A：快速索引

| 想知道什么 | 看哪一节 |
|-----------|---------|
| 整体长什么样 | [2.1 分层架构](#21-分层架构) |
| 一次提问发生了什么 | [3. 完整生命周期](#3-一次提问的完整生命周期) |
| `create_agent` 生成的图 | [4.2 编译出的真实图](#42-编译出的真实图) |
| 三个工具为什么是这三个 | [1.1](#11-固定管道的结构性缺陷) + [5.2](#52-agenttoolspy--三个工具) |
| 工具返回怎么截断 | [5.2.4](#524-工具返回值的-token-预算) |
| **上下文怎么压缩的** | [6. 完整机制](#6-上下文压缩完整机制)（尤其 6.4 / 6.5） |
| 会话怎么存的 | [7. 会话持久化](#7-会话持久化) |
| 怎么防止死循环 | [8. 三道闸](#8-终止条件三道闸) |
| `reset` 事件是干嘛的 | [9.2](#92-reset-为什么必须存在) |
| 怎么打开 trace | [11.2](#112-trace-是怎么接的) |
| **一次问答多少钱、多少秒** | [11.1.1](#1111-token-用量精确值与估算值必须分开) + [11.1.2](#1112-分节点耗时瓶颈到底在哪) |
| 线上差评怎么变成回归用例 | [15.6](#156-没有在线反馈闭环--已解决) |
| 为什么不用 langgraph 手写图 | [14.1](#141-为什么放弃手写-langgraph-状态图) |
| 这个选型被追问时怎么答 | [14.4 框架选型的常见追问](#144-框架选型的常见追问) |

## 附录 B：关键文件与行号

| 内容 | 位置 |
|------|------|
| System prompt | `agent/runner.py:54` |
| 模型构造与缓存 | `agent/runner.py:84` |
| checkpointer 单例 | `agent/runner.py:141` |
| **中间件装配** | `agent/runner.py:163` |
| trace handler | `agent/runner.py:206` |
| 事件流主循环 | `agent/runner.py:320` |
| 事件翻译 | `agent/runner.py:425` |
| token 估算 | `agent/tools.py:50` |
| 打包与截断 | `agent/tools.py:87` |
| 参数 schema | `agent/tools.py:117` |
| 去重叠拼接 | `agent/tools.py:167` |
| 三个工具实现 | `agent/tools.py:204` / `228` / `283` |
| 工具工厂 | `agent/tools.py:331` |
| `chunk_overlap` | `rag_engine.py:317` |
| `fetch_section` | `rag_engine.py:602` |
| `document_exists` | `rag_engine.py:630` |
| `find_literal` | `rag_engine.py:646` |
| trace_id 与请求记录 | `observability.py` |
| 反馈库与 badcase 导出 | `feedback.py` / `scripts/badcase_export.py` |
| token 用量累计 | `agent/runner.py` 的 `_accumulate_usage` |
| usage 的端点能力探测 | `rag_engine.py` 的 `_open_stream` |
| `/api/health` | `main.py:262` |
| `build_agent_runner` | `main.py:299` |
| `/api/chat` | `main.py:316` |
| `/api/agent` | `main.py:360` |
