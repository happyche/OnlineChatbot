# -*- coding: utf-8 -*-
"""
[RAGAS] 把本项目的配置与模型接到 RAGAS 上的适配层。

整个文件都是为 RAGAS 评测新增的，不参与服务运行。

为什么需要单独一层：
  RAGAS 的评委（judge）需要两样东西——一个能产出结构化输出的 LLM，
  以及一个把文本变成向量的 embedding。这两样本项目都已经有了，
  但接口形状和 RAGAS 期望的不一样。与其在评测脚本里重复拼装客户端、
  重复读一遍 .env，不如在这里统一从 config 派生，
  这样「评测用的模型」和「线上跑的模型」永远来自同一份配置，
  不会出现改了 .env 却只有一半生效的情况。

所有 ragas 的 import 都是惰性的（写在函数体内），
目的是让没装 ragas 的环境仍能导入本模块——
评测脚本的 --dump-only 与离线单测都依赖这一点。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from embeddings import Embedder  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402

#: 未安装 ragas 时给出的提示。直接抛 ModuleNotFoundError 的话，
#: 报错栈会指向 langchain 之类的间接依赖，看不出该装什么。
INSTALL_HINT = (
    "未安装 RAGAS 评测依赖。请先运行：\n"
    "    pip install -r requirements-eval.txt\n"
    "（该依赖较重，会引入 langchain / datasets 等包，故未并入 requirements-dev.txt）"
)


def ensure_ragas() -> None:
    """检查 ragas 是否可用，不可用时抛出可操作的提示。"""
    try:
        import ragas  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(INSTALL_HINT) from exc


def resolve_judge_endpoint(
    settings: dict, provider: str = "auto", model: Optional[str] = None
) -> "config.Endpoint":
    """
    [RAGAS] 解析评委该用哪个端点。

    默认沿用被测系统的 llm_provider，但允许单独指定，因为评委和被测模型
    未必该跑在同一个地方：被测的自建服务可能显存吃紧跑不动大模型，
    而评委恰恰应该用更强的模型才判得准。
    """
    if provider not in ("auto", "openai", "local"):
        raise ValueError(f"未知的 judge provider: {provider!r}（可选 auto/openai/local）")

    effective = dict(settings)
    if provider != "auto":
        effective["llm_provider"] = provider

    endpoint = config.resolve_llm(effective)
    if model:
        endpoint = config.Endpoint(
            base_url=endpoint.base_url,
            model=model,
            api_key=endpoint.api_key,
            provider=endpoint.provider,
        )
    return endpoint


#: 评委的输出长度上限。
#:
#: ragas 自己的默认值是 1024，对 GPT-4o 那类直接吐 JSON 的模型够用，
#: 但会思考的模型（qwen3、deepseek-r1 等）在给出 JSON 前先输出一大段推理，
#: 1024 经常不够，结果是 instructor 抛 IncompleteOutputException，
#: 表现为「某些问题上指标随机缺失」——很容易被误当成模型能力问题。
#: ragas 文档对 GPT-5 / o 系列也建议调到 4096 以上。
#: max_tokens 是上限不是目标值，调高不会让正常请求变慢或变贵。
DEFAULT_JUDGE_MAX_TOKENS = 4096


class _RunFailed(RuntimeError):
    """[RAGAS] agent run 跑起来了但没正常结束。与「输出解析不了」区分开。"""


def build_judge_llm(
    settings: dict,
    model: Optional[str] = None,
    json_mode: bool = False,
    provider: str = "auto",
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
):
    """
    [RAGAS] 构造评委 LLM。

    默认复用 config.resolve_llm 解析出的端点，因此 llm_provider=local 时
    评委也会走自建服务，不会偷偷把内容发去云端。

    参数:
        model      : 覆盖评委模型名。默认与被测模型相同——
                     同模型自评存在偏袒风险（模型倾向于认为自己的输出是对的），
                     真要下结论时建议换一个更强的模型做评委。
        json_mode  : 走 instructor 的 Markdown-JSON 模式而不是原生
                     response_format。DashScope 等 OpenAI 兼容端点对
                     json_schema 的支持参差不齐，原生模式报 400 时用这个降级。
        provider   : auto=沿用 llm_provider｜openai=强制云端｜local=强制自建。
        max_tokens : 见 DEFAULT_JUDGE_MAX_TOKENS。

    返回: InstructorBaseRagasLLM
    """
    ensure_ragas()
    from ragas.llms import llm_factory

    endpoint = resolve_judge_endpoint(settings, provider=provider, model=model)
    timeout = settings.get("request_timeout", 60)
    # 复用引擎的客户端构造逻辑，连内网绕代理这类细节也一并继承
    client = RAGEngine._make_client(endpoint, settings, timeout)

    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if json_mode:
        import instructor

        kwargs["mode"] = instructor.Mode.MD_JSON

    return llm_factory(endpoint.model, client=client, **kwargs)


def build_judge_embeddings(embedder: Embedder):
    """
    [RAGAS] 把本项目的 Embedder 包装成 RAGAS 的 BaseRagasEmbedding。

    只有 AnswerRelevancy 用得到：它把「答案反推出的问题」与原问题做余弦比较。

    参数 embedder 应当直接传 RAGEngine.embedder，而不是另外构造一个。
    另建一个即便配置相同也是白费——远程嵌入会多开一条连接，
    本地嵌入会把 90MB 的模型再加载一遍。

    为什么不用 ragas 自带的 embedding_factory：
      那个工厂只认 openai / litellm 这类远程 provider，而本项目默认用的是
      本地 fastembed 的 bge-small-zh-v1.5。走自带工厂等于评测时换了一套
      向量空间，而且把语料发去云端，与本项目「可离线运行」的前提冲突。
      包一层反而更简单——BaseRagasEmbedding 只要求实现单条文本的同步/异步编码。

    类定义写在函数体内，是因为它必须继承 ragas 的基类，
    而模块级 import ragas 会让没装 ragas 的环境连本文件都导入不了。
    """
    ensure_ragas()
    from ragas.embeddings.base import BaseRagasEmbedding

    class _EmbedderAdapter(BaseRagasEmbedding):
        """把 Embedder 的 embed_query 暴露成 RAGAS 期望的形状。"""

        def __init__(self, wrapped: Embedder):
            super().__init__()
            self._wrapped = wrapped
            self.signature = wrapped.signature

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await self._wrapped.embed_query(text)

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            # 基类要求提供同步版本。RAGAS 算指标时走的是异步路径，
            # 这里基本不会被调用，但不能不实现。
            # 已在事件循环中时 asyncio.run 会直接抛 RuntimeError，
            # 故换一个线程跑独立循环。
            coro = self._wrapped.embed_query(text)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            # 基类默认逐条串行，而本项目的 embedder 本来就支持批量，
            # 覆盖掉能少很多次远程调用
            return await self._wrapped.embed_documents(list(texts))

    return _EmbedderAdapter(embedder)


# ===================================================================
# [RAGAS] Cursor Agent 作为评委
# ===================================================================
#
# RAGAS 对评委的要求其实很窄，只有两个方法：
#     generate(prompt: str, response_model: type[BaseModel]) -> BaseModel
#     agenerate(...)  同上，异步
# 也就是「给一段文本、返回一个结构化对象」。任何能产出文本的后端都能接，
# 包括 Cursor 的 agent —— 只要自己把 JSON Schema 塞进 prompt、
# 再从回复里把 JSON 抠出来校验，就补上了 instructor 平时干的活。
#
# 但要清楚代价，这条路和 OpenAI 兼容端点不是等价替换：
#   1. 没有 temperature / seed，分数不可复现。评测最主要的用途是
#      「改了配置分数变了没有」，评委自身有噪声就分不清是改动还是抖动。
#   2. agent 自带工具。一个能上网、能读文件的评委在判 faithfulness 时
#      可能引入 contexts 之外的知识，那测的就不是忠实度而是事实性了。
#      下面用 tools=[] 关掉工具，并把 cwd 指向空临时目录做第二道防线。
#   3. 一次 agent run 的开销远大于一次 chat completion，不适合上千次的评测。
#
# 所以它的定位是「第二个独立评委，用来和主评委交叉验证」，
# 不是回归评测的基准线。

#: 附加在 RAGAS 原始 prompt 后面的输出约束。
#: 平时这活由 instructor 通过 response_format 完成，这里只能靠提示词。
_JSON_INSTRUCTION = """
---
Respond with a single JSON object that validates against the JSON Schema below.

Rules:
- Output ONLY the JSON object. No prose before or after, no code fences.
- Do not use any tools. Answer directly from the text above.
- Do not consult outside knowledge. Judge only from what is given.

JSON Schema:
{schema}
"""


def extract_json_object(text: str) -> str:
    """
    [RAGAS] 从自由文本里抠出最外层的 JSON 对象。

    agent 不保证只吐 JSON —— 可能裹上 ```json 围栏，也可能前后带一句说明。
    不能简单地取第一个 { 到最后一个 }，因为字符串字面量里也可能有花括号
    （评判理由里引用了含 { 的原文就会踩到），所以扫描时要跳过字符串与转义。

    找不到合法的 JSON 对象时抛 ValueError，由调用方决定重试还是记失败。
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    raise ValueError(f"回复里找不到完整的 JSON 对象：{text[:200]!r}")


def build_cursor_judge(
    model: str = "gpt-5.6-sol",
    api_key: Optional[str] = None,
    attempts: int = 3,
):
    """
    [RAGAS] 用 Cursor 的本地 agent 充当评委。

    参数:
        model    : Cursor 模型 id。评委要尽量强 —— LLM 裁判的可靠性
                   直接取决于裁判本身，在评委上省钱省出来的是不可信的分数。
                   但**能列出不等于能用**：模型目录反映的是账号能看到什么，
                   未开通的模型会返回 status=error（实测该账号下 Claude
                   系列全部不可用）。故默认取一个实测可用的强模型，
                   并在开跑前做一次预检。
                   务必钉死具体型号，不要用 auto / default ——
                   它们会在请求之间换模型，两次评测就没有可比性了。
        api_key  : 留空则读环境变量 CURSOR_API_KEY。
        attempts : JSON 解析失败时的重试次数。失败原因会回灌给模型。

    类定义写在函数体内的原因同 build_judge_embeddings：要继承 ragas 的基类。
    """
    ensure_ragas()
    from ragas.llms.base import InstructorBaseRagasLLM

    try:
        from cursor_sdk import (
            AgentOptions,
            AsyncAgent,
            AsyncClient,
            CursorAgentError,
            LocalAgentOptions,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未安装 Cursor SDK。请运行：\n"
            "    pip install cursor-sdk\n"
            "（仅 --judge-provider cursor 需要，其余评委不依赖它）"
        ) from exc

    key = api_key or os.getenv("CURSOR_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "使用 Cursor 评委需要 API Key。请在 Cursor Dashboard → API Keys 生成后，"
            "写入环境变量或本项目 .env 中的 CURSOR_API_KEY。"
        )

    class _CursorAgentJudge(InstructorBaseRagasLLM):
        """把 Cursor 的 agent run 包装成 RAGAS 期望的结构化生成器。"""

        def __init__(self):
            self.model = model
            self._attempts = max(1, attempts)
            # agent 在一个空目录里跑：即便某个工具没被 tools=[] 拦住，
            # 它也读不到本项目的代码和文档，不会把语料之外的信息带进评判
            self._sandbox = tempfile.mkdtemp(prefix="ragas-cursor-judge-")
            self._client = None
            #: 累计的 agent run 次数与耗时，供跑完后核对成本
            self.runs = 0
            self.total_ms = 0

        async def _ensure_client(self):
            """
            惰性启动 SDK 的本地 bridge。

            必须走 AsyncClient 而不是同步的 Agent.prompt：
            cursor-sdk 1.0.30 的同步 bridge 在 Windows 上起不来 ——
            它把子进程的 stderr 管道注册进 selectors.DefaultSelector()，
            而 Windows 的 select() 只接受 socket，于是抛
            OSError: [WinError 10038]。异步那条路用的是
            asyncio.create_subprocess_exec，没有这个限制。
            """
            if self._client is None:
                self._client = await AsyncClient.launch_bridge(workspace=self._sandbox)
            return self._client

        async def _ask(self, text: str) -> str:
            client = await self._ensure_client()
            options = AgentOptions(
                model=self.model,
                api_key=key,
                local=LocalAgentOptions(cwd=self._sandbox),
                # 评委不该有工具：能上网就不再是「只依据 contexts 判断」了
                tools=[],
            )
            result = await AsyncAgent.prompt(text, options, client=client)
            self.runs += 1
            self.total_ms += int(getattr(result, "duration_ms", 0) or 0)

            if result.status != "finished":
                raise _RunFailed(
                    f"Cursor agent run 未正常结束：status={result.status} "
                    f"id={result.id} result={result.result!r}"
                )
            return result.result or ""

        async def preflight(self) -> None:
            """
            开跑前用一个最小请求验证模型确实可用。

            必要性来自一次实测教训：Cursor 的模型目录（models.list）返回的是
            账号能「看到」的模型，不等于能「用」—— 未开通的模型不会报鉴权错，
            而是每次都返回 status=error 且 result 为空。没有预检的话，
            这种失败会被当成单格失败重试三次，20 格就白烧 60 次 agent run
            才让人看出不对。预检把代价压到 1 次。
            """
            try:
                await self._ask('Reply with exactly this JSON and nothing else: {"ok": true}')
            except _RunFailed as exc:
                raise RuntimeError(
                    f"Cursor 模型 {self.model!r} 不可用：{exc}\n"
                    "模型目录里能列出不代表账号有权使用（实测该账号下 Claude 系列"
                    "全部返回 status=error）。请换一个模型，或在 Cursor 后台确认权限。"
                ) from exc

        def generate(self, prompt: str, response_model):
            # 基类要求提供同步版本，但 AsyncClient 与创建它的事件循环绑定，
            # 每次 asyncio.run 都会另起一个循环、也就要重启一次 bridge，
            # 代价高到没有实用价值。RAGAS 算指标走的是 ascore → agenerate，
            # 这条路正常情况下不会被调用。
            raise NotImplementedError(
                "Cursor 评委只支持异步调用。请用 metric.ascore(...) 而不是 metric.score(...)。"
            )

        async def agenerate(self, prompt: str, response_model):
            schema = json.dumps(
                response_model.model_json_schema(), ensure_ascii=False, indent=2
            )
            message = prompt + _JSON_INSTRUCTION.format(schema=schema)
            last_error: Optional[Exception] = None

            for attempt in range(self._attempts):
                try:
                    reply = await self._ask(message)
                    return response_model.model_validate_json(extract_json_object(reply))
                except CursorAgentError:
                    # 压根没跑起来（鉴权、网络、配置），重试无意义，直接上抛
                    raise
                except _RunFailed as exc:
                    # 跑起来但中途失败，可能是限流之类的瞬时问题，值得重试。
                    # 但不能回灌「你上次的回复解析不了」—— 根本没有回复，
                    # 那句话只会把模型带偏。
                    last_error = exc
                except Exception as exc:
                    # 有回复但不合规。把错误回灌给模型往往一次就能纠正，
                    # 比直接判这一格失败要划算
                    last_error = exc
                    if attempt + 1 < self._attempts:
                        message = (
                            prompt
                            + _JSON_INSTRUCTION.format(schema=schema)
                            + f"\n\nYour previous reply could not be parsed: {exc}\n"
                            "Return only the JSON object this time."
                        )

            raise RuntimeError(
                f"Cursor 评委连续 {self._attempts} 次未产出合法 JSON：{last_error}"
            )

        async def aclose(self) -> None:
            """关闭 bridge 子进程并清理沙箱目录。不关会漏子进程。"""
            import shutil

            if self._client is not None:
                await self._client.aclose()
                self._client = None
            shutil.rmtree(self._sandbox, ignore_errors=True)

    return _CursorAgentJudge()
