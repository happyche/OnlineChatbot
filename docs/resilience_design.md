# 熔断与降级设计

## 1. 目标与原则

1. **下游挂了，用户仍然拿到东西**。能用 agent 就用 agent，不行退固定管道，
   LLM 也不行就直接给检索片段，检索也不行才报错——每一级都比下一级有用。
2. **熔断按「下游依赖」划分，不按工具划分**。三个工具共用 Qdrant，
   Qdrant 一挂三个一起不可用；按工具各自计数要各失败 3 轮才熔断，白白多打 9 次。
3. **只有「下游坏了」才计入熔断**。400 参数错、401 key 错、模型名写错、代码 bug
   重试多少次都不会好，计进去只会让一个配置错误把服务熔断掉。
4. **重试只在一处发生**。SDK 重试、工具重试、上层重试叠加会把一次失败放大成 9 次调用。
5. **熔断判断在基础设施层，不交给 LLM**。依赖不可用时把工具从工具列表里拿掉，
   而不是让模型调一次、读一句「失败了」、再换一个试。

## 2. 依赖与熔断器

| 熔断器 key | 保护对象 | 使用方 | 启用条件 |
|---|---|---|---|
| `qdrant` | 远程 Qdrant | 三个工具、固定管道检索 | 配了 `QDRANT_URL`（本地模式无网络故障，不设） |
| `embedding` | 远程 Embedding API | `search_docs`、固定管道检索 | `embedding_provider=openai` |
| `llm:<base_url>\|<model>` | 对话 / 改写模型端点 | agent 每次模型调用、固定管道生成、query rewrite | 总是 |
| `route:agent` | agent 整条链路 | `/api/chat` 路由选择 | `agent_enabled=true` |

- `rewrite_model` 与对话模型不同时，自然得到两个不同的 `llm:` key，改写挂了不连累生成。
- `route:agent` 对应原设计里「LLM 连续失败 5 次就退回原始检索」：agent 一次要调多次 LLM、
  prompt 更长、更容易超时或撞递归上限，**单次 LLM 调用能成功不代表 agent 能跑完**，
  所以单独一个熔断器，打开后走固定管道。
- Reranker、BM25 是本进程内的计算，不设熔断器，出错直接跳过该阶段（见 §6）。

## 3. 熔断器状态机

```
           失败达到阈值                    冷却到期
 CLOSED ───────────────▶ OPEN ───────────────▶ HALF_OPEN（只放 1 个探测请求）
   ▲                      ▲                        │
   │                      └── 探测失败，冷却×2 ─────┤
   └────────────── 探测成功，冷却重置 ──────────────┘
```

- **熔断条件**（二选一）：连续失败 N 次；或滑动窗口 60s 内调用 ≥ 10 次且失败率 ≥ 50%。
  低流量时前者生效，并发高时后者更稳（夹一次成功就清零的问题由它兜住）。
- **冷却**：60s → 120s → 240s … 上限 600s，加 0～10% 随机抖动。
- **Half-Open 只放一个请求**，其余请求继续按 OPEN 处理，避免恢复瞬间被打垮。
- OPEN 时调用方拿到 `CircuitOpenError`（立即抛出，不发网络请求）。
- 状态为进程内存（当前部署单 uvicorn 进程）；多进程/多副本时可换 Redis，接口不变。
- 读写加 `threading.Lock`：Qdrant 调用跑在 `asyncio.to_thread` 的线程里。

## 4. 错误分类

`resilience.classify_error(exc) -> ErrorKind`：

| ErrorKind | 来源 | 重试 | 计入熔断 |
|---|---|---|---|
| `RETRYABLE` | 5xx、408、超时、连接错误、`ResponseHandlingException`、200 但返回内容不对（如嵌入条数不一致） | 是 | 是 |
| `RATE_LIMIT` | 429 限流（RPM / TPM / 并发数超限），且 SDK 自带重试后仍失败 | 退避（遵守 Retry-After） | 是（但只按失败率，不按连续次数） |
| `CLIENT` | 400 / 422（参数错、上下文超长、filter 写错） | 否 | 否 |
| `CONFIG` | 401 / 403 / 404、**429 且错误码为额度耗尽（`insufficient_quota` 等）**、本地模型加载失败、Qdrant 本地目录被占 | 否 | 否，打 ERROR 日志告警 |
| `CANCELLED` | 客户端断开（`asyncio.CancelledError`） | 否 | 否 |
| `BUG` | `KeyError` / `TypeError` / `ValueError` 等其余异常 | 否 | 否 |

识别依据：`openai.APIStatusError.status_code`、`qdrant_client.http.exceptions.UnexpectedResponse.status_code`、
异常类型本身。分类结果写进日志、工具 artifact 的 `meta.error_kind` 和 stages。

**429 要按响应体再分一次**：同样是 429，「请求太快」等一会儿就好，「额度/余额用完」等多久都不会好。
后者按 `exc.code` / `exc.body["error"]["code"]`（如 OpenAI 的 `insufficient_quota`）识别，归入 `CONFIG`。
另外 SDK 会自动重试 429，所以业务代码只在 SDK 重试用完后才会看到这个异常，日志里出现它说明限流已持续一段时间。

## 5. 重试策略（每类调用只在一处重试）

| 调用 | 在哪重试 | 次数 | 说明 |
|---|---|---|---|
| OpenAI 兼容接口（对话、改写、嵌入、agent 模型调用） | **SDK 自带** `max_retries` | 默认 1（可配 `llm_max_retries`） | SDK 只在建立请求阶段重试，流式输出开始后不会重放，不会产生重复 token；自带指数退避和 Retry-After |
| Qdrant | `VectorStore` 内的 `with_retry` | 2 次，0.3s / 0.6s + 抖动 | qdrant_client 自身不重试；只读操作才重试，`add`/`delete` 不重试 |
| 工具 `_guard` | **不重试** | — | 下游已各自重试过，再套一层就是 3×3 |
| query rewrite | 不重试 | — | 本来就有回退（用原问题），失败直接回退即可 |

熔断器统计的是**重试之后的最终结果**：一次逻辑调用 = 熔断器上的一次成功/失败。

## 6. 降级等级

| 等级 | 名称 | 行为 | 调 LLM 次数 |
|---|---|---|---|
| L0 | Agent | agent 多轮取材 + 作答（`agent_enabled=true` 时的正常路径） | 2～6 |
| L1 | 固定管道 | 一次检索 + 一次生成（即现有 `engine.query`）；改写端点熔断时跳过改写 | 1（+ 可选改写 1） |
| L2 | 仅检索 | 不调 LLM，把召回片段按「文档 › 章节」排版返回，顶部提示「AI 服务暂时不可用，以下为相关文档片段」 | 0 |
| L3 | 不可用 | 返回友好提示「知识库服务暂时不可用，请稍后再试」 | 0 |

### 6.1 请求开始时：按熔断器状态选起始等级（不浪费一次尝试）

```
qdrant OPEN                         → L3
llm 端点 OPEN                        → L2
agent_enabled 且 route:agent 未 OPEN → L0
否则                                 → L1
```

`embedding` OPEN 时：L0 中 `search_docs` 被摘掉（`expand_section` / `find_literal` 仍可用）；
L1/L2 的检索退回 **BM25-only**（BM25 索引不依赖嵌入）——这是可选增强，见 §9 第二期。

### 6.2 请求进行中：失败就逐级下降

```
L0 失败 ──▶ 发 reset 事件 ──▶ L1
            （GraphRecursionError、超出 agent 总时限、LLM 失败、任何异常都算 agent 失败，
             计入 route:agent；若错误是 CONFIG 类或 llm 端点已 OPEN，直接跳 L2）
L1 生成失败 ──▶ 已有 token 则发 reset ──▶ L2（复用 L1 已检索到的 hits，不重新检索）
L1/L2 检索失败 ──▶ L3
```

- 复用现有 SSE 协议里的 `reset` 事件清掉已经流出的半截答案，旧前端也能正确处理。
- 新增一个 `{"type": "degraded", "level": "L1", "reason": "..."}` 事件；旧前端忽略未知事件，不受影响。
- `done` 事件的 stages、`RequestRecord`、反馈库里记录 `degrade_level` 与 `degrade_reason`，
  badcase 复查时能区分「答得差」和「降级了」。

### 6.3 进程内阶段级降级（不涉及熔断器）

- **Reranker 出错**：跳过重排，用 RRF/余弦顺序，stages 记 `rerank_skipped=<原因>`。
- **BM25 出错**：退回纯向量检索，stages 记 `bm25_skipped`。
- **query rewrite 失败或端点 OPEN**：用原问题（现有行为，只需加一个 OPEN 判断提前跳过）。

## 7. Agent 内部的接入方式

新增 `ResilienceMiddleware`（与现有 `FinalAnswerOnLimitMiddleware` 同一套 `awrap_model_call` 写法）：

1. **调用前**：按工具→依赖映射过滤 `request.tools`，依赖处于 OPEN 的工具不给模型看。
   全部工具都被摘掉时由系统提示告知「检索暂不可用」。
2. **调用 LLM**：检查 `llm:` 熔断器，OPEN 直接抛 `CircuitOpenError`（agent 失败 → 降级）。
3. **调用后**：成功 `on_success()`；异常按分类决定是否 `on_failure()`，然后原样抛出。

工具 → 依赖映射：

```python
TOOL_DEPENDENCIES = {
    "search_docs":    ("qdrant", "embedding"),
    "expand_section": ("qdrant",),
    "find_literal":   ("qdrant",),
}
```

`_guard` 的职责收窄为：依赖 OPEN 时返回「工具暂时不可用」；异常时分类并把
`error_kind` / `status_code` 写进 artifact，返回可读失败文本给模型。**计数不在这里做**，
在下游边界（VectorStore / Embedder）做，固定管道与 agent 共享同一份熔断状态。

## 8. 配置项（`config.DEFAULTS`，支持热重载）

| 键 | 默认 | 说明 |
|---|---|---|
| `resilience_enabled` | `true` | 总开关，关掉后行为与现在一致 |
| `llm_max_retries` | `1` | OpenAI SDK 重试次数（原硬编码 2） |
| `qdrant_retries` | `2` | Qdrant 只读调用重试次数 |
| `breaker_fail_threshold` | `3` | qdrant / embedding 连续失败阈值 |
| `llm_breaker_fail_threshold` | `5` | LLM 端点连续失败阈值 |
| `agent_breaker_fail_threshold` | `5` | agent 链路连续失败阈值 |
| `breaker_window_seconds` / `breaker_min_calls` / `breaker_failure_rate` | `60` / `10` / `0.5` | 失败率熔断 |
| `breaker_base_cooldown` / `breaker_max_cooldown` | `60` / `600` | 冷却时间（秒） |
| `agent_deadline_seconds` | `120` | agent 单次请求总时限，超时按 agent 失败降级到 L1 |

## 9. 改动清单

### 第一期（核心，建议先做）

| 文件 | 改动 | 规模 |
|---|---|---|
| `resilience.py`（新增） | `ErrorKind` + `classify_error`、`CircuitBreaker`（状态机 + 失败率窗口 + 线程锁）、`BreakerRegistry` 进程级单例、`CircuitOpenError`、`with_retry` | 新增约 220 行 |
| `config.py` | §8 的配置项 | +20 行 |
| `vector_store.py` | 远程模式下只读方法（`search`/`get`/`count`/`find_contains`）包一层「检查熔断 → 重试 → 记结果」 | +40 行 |
| `embeddings.py` | `OpenAIEmbedder.embed_documents` 接 `embedding` 熔断器；条数不一致的 `RuntimeError` 改为可识别的下游错误 | +20 行 |
| `rag_engine.py` | `_make_client` 的 `max_retries` 读配置；`_stream_chat` 接 `llm:` 熔断器；`_apply_rerank` 出错跳过；`query()` 生成失败时抛出带 hits 的异常（或经 `collect` 带出），供 L2 复用 | +60 行 / 改 20 行 |
| `query_rewrite.py` | 调用前检查改写端点熔断器，OPEN 直接回退；结果计入熔断；加 `timeout` | +15 行 |
| `agent/tools.py` | `_guard` 改造（§7）；`list_documents` 两处调用包 `asyncio.to_thread`（顺带修事件循环阻塞） | +30 行 / 改 10 行 |
| `agent/runner.py` | `ResilienceMiddleware`；`ChatOpenAI` 的 `max_retries` 读配置；error 事件带 `error_kind`；agent 总时限 | +80 行 |
| `main.py` | `/api/chat` 的 `generate()` 改成 L0→L1→L2→L3 降级编排；`degraded` 事件；`route:agent` 熔断计数；`/api/admin/health/detail` 输出各熔断器状态；`/api/health` 在有熔断器 OPEN 时报 `degraded` | +120 行 / 改 40 行 |
| `tests/test_resilience.py`（新增） | 状态机、冷却翻倍与上限、Half-Open 只放一个、错误分类表 | 新增约 200 行 |
| `tests/test_api_e2e.py` / `test_agent.py` | 注入失败的模型/工具，验证 L0→L1、L1→L2、qdrant OPEN→L3、工具被摘除 | +150 行 |

### 第二期（可选增强）

- `embedding` OPEN 时检索退回 BM25-only（`retrieve_with_diagnostics` 加一个分支）。
- `observability.py`：熔断器状态 gauge、状态切换计数、各降级等级请求数，接到 `/metrics`。
- `static/index.html`：收到 `degraded` 事件时在气泡上方显示一条提示横幅。
- 多副本部署时熔断状态迁到 Redis。

## 10. 工作量评估

- **改动规模：中等**。新增 1 个模块（约 220 行）+ 约 9 个文件的局部修改（约 400 行增改），
  测试约 350 行。
- **核心结构不动**：检索管道、agent 图、工具实现、SSE 协议都保持原样，
  只是在下游调用边界加熔断、在 `/api/chat` 加降级编排。
- **风险最大的一处是 `main.py` 的降级编排**：要处理「已经流出部分 token 后失败」，
  依赖 `reset` 事件，需要端到端测试覆盖。
- **预计工期**：第一期 2～3 人天（含测试），第二期 1～1.5 人天。
- **可灰度**：`resilience_enabled=false` 时所有新逻辑旁路，行为与现在完全一致。
