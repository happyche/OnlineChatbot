# -*- coding: utf-8 -*-
"""
Agent 层测试
============
两条线，分别对应两类会真正出错的地方：

**工具层**（不需要模型）。工具的契约是「永远返回可读文本，失败时带上怎么改」，
这条契约一旦破了，表现出来不是异常而是模型在错误参数上反复重试直到撞上限——
最难归因的那种失败。所以重点测失败路径与边界，而不是顺利路径。

**编排层**（用脚本化的假模型）。create_agent 的循环本身是 langchain 的代码，
不需要我们测；要测的是我们接在它两端的东西：事件序列对不对、
取材前的开场白有没有被 reset 掉、checkpointer 是不是真的跨轮记住了对话。

全程离线：嵌入用 hashing，模型用按脚本回话的替身，不访问网络。
"""
from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent import tools as agent_tools
from agent.runner import AgentRunner, reset_default_checkpointer

DOC = """# 运维手册

## 超时设置

REQUEST_TIMEOUT 默认为 60 秒，控制单次上游调用的最长等待时间。
NSR_REFRESH_TIMER 默认为 2100 秒，决定缓存的刷新周期。

## 故障处理流程

第一步是确认告警的真实性，核对同一时间窗口内是否有其它关联告警同时触发。
第二步是采集现场，包括进程状态、最近一小时的日志与当前连接数。
第三步是判断影响范围，对照另外两个可用区的同名服务指标。

## 配置核对

发布前需要逐项核对 REQUEST_TIMEOUT 与 NSR_REFRESH_TIMER 两项配置。
"""


@pytest.fixture
async def engine(make_engine):
    """装好一篇文档的引擎。三个工具都对着它取材。"""
    eng = make_engine(chunk_size=200, chunk_overlap=20)
    await eng.add_document(DOC, "运维手册.md")
    return eng


@pytest.fixture
def settings():
    return {"agent_tool_payload_tokens": 1200, "agent_expand_payload_tokens": 2400}


# ======================================================================
# 假模型
# ======================================================================

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
        names = [getattr(t, "name", str(t)) for t in tools]
        self.log["bound_tools"] = names
        # 逐次记录，供「最后一轮有没有被摘掉工具」这类断言查看
        self.log.setdefault("bind_history", []).append(names)
        # 返回 bind 出来的 Runnable 而不是 self：真实模型把工具清单透传进这一次
        # 调用，_generate 才看得见「这一次有没有工具」。返回 self 会让
        # 「上一轮绑过工具」一直生效——而 langchain 在 tools 为空时根本不调
        # bind_tools，于是「最后一轮摘掉工具」这件事就永远测不出来。
        return self.bind(tools=names)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log["calls"] += 1
        self.log.setdefault("seen", []).append(messages)
        # 这一次没有工具就不能返回 tool_calls。真实模型做不到的事替身也不该做，
        # 否则「摘掉工具逼它作答」这条路径测不出来。
        if not kwargs.get("tools"):
            msg = AIMessage(content="（无工具可用）基于现有资料的收尾回答")
        else:
            index = min(self.log["calls"] - 1, len(self.script) - 1)
            msg = self.script[index]
        return ChatResult(generations=[ChatGeneration(message=msg)])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def make_runner(engine, script, settings=None, checkpointer=None) -> AgentRunner:
    model = ScriptedChatModel(script=script)
    runner = AgentRunner(
        engine,
        settings={
            "agent_max_model_calls": 6,
            "agent_recursion_limit": 25,
            "agent_summary_trigger": 20,
            "agent_summary_keep": 8,
            **(settings or {}),
        },
        model=model,
        checkpointer=checkpointer,
    )
    runner.model = model  # 便于用例检查模型看到了什么
    return runner


# ======================================================================
# 工具层
# ======================================================================

async def test_search_docs_命中时标注文档与章节(engine, settings):
    """标注是 expand_section 的输入来源，格式错了整条工具链就断了。"""
    out, _ = await agent_tools.search_docs(
        engine, settings, query="REQUEST_TIMEOUT 是多少"
    )

    assert "[search_docs]" in out
    assert "文档: 运维手册.md" in out
    assert "章节:" in out


async def test_search_docs_空结果不是失败(make_engine, settings):
    """
    「库里确实没有」是有效信息，是拒答的依据。
    把它写成失败会让模型以为该重试，在空结果上白跑几轮。
    """
    out, artifact = await agent_tools.search_docs(
        make_engine(), settings, query="Kubernetes 集群怎么部署"
    )

    assert "失败" not in out
    assert "未命中" in out
    # ok=True 而 empty=True：执行成功，只是确实没有内容
    assert artifact["ok"] is True and artifact["empty"] is True
    assert artifact["contexts"] == []


async def test_search_docs_空结果不替模型断言知识库里没有(make_engine, settings):
    """
    语义检索没命中也可能只是问句用词和文档对不上。把「知识库里没有相关资料」
    写死在工具返回里，等于用一次向量检索的结果替模型下了终局结论——
    而这正是 README 里点名要避免的那类静默失败。
    """
    out, _ = await agent_tools.search_docs(
        make_engine(), settings, query="Kubernetes 集群怎么部署"
    )

    assert "不一定代表" in out
    assert "find_literal" in out, "没命中时要给出下一步，而不是只说没命中"


# ---- 命中质量信号 ----
#
# 向量检索的语义是「返回最相似的 K 条」而不是「返回相关的条目」，所以
# 知识库里没有的问题照样拿到 top_k 个片段。没有质量信号时模型分辨不出来，
# 会挨个换工具重试直到撞上调用次数上限。

def _hit(rerank_score, score_type="rerank"):
    """构造一条带指定重排分的命中。"""
    from rag_engine import Retrieved

    return Retrieved(
        text="正文",
        source="运维手册.md",
        heading="超时设置",
        similarity=0.6,
        bm25_score=None,
        rrf_score=None,
        rerank_score=rerank_score,
        score=rerank_score if rerank_score is not None else 0.0,
        score_type=score_type,
        doc_id="d1",
    )


def test_命中质量高时不附提示(settings):
    note = agent_tools._confidence_note(
        [_hit(5.7)], {**settings, "agent_low_confidence_score": 1.0}
    )

    assert note == ""


def test_命中质量存疑时给出一条明确的下一步(settings):
    """
    提示要给**一条**具体的下一步。泛泛说「再找找」只会让模型把剩下的工具
    都试一遍——那正是这个提示要消除的行为。
    """
    note = agent_tools._confidence_note(
        [_hit(-3.6)], {**settings, "agent_low_confidence_score": 1.0}
    )

    assert "find_literal" in note
    assert "硬凑" in note, "低分片段最危险的用法是被当成依据编出有出处的答案"


def test_重排关闭时不提示(settings):
    """
    余弦受语料与模型分布影响（实测无关问题的余弦能比库内问题还高），
    而 RRF 只用名次、完全丢弃幅度——任何查询的 top1 拿到的分都一样。
    两者都不能当阈值，所以宁可不提示，也不换个分数凑合。
    """
    low = {**settings, "agent_low_confidence_score": 1.0}

    assert agent_tools._confidence_note([_hit(None, "cosine")], low) == ""
    assert agent_tools._confidence_note([_hit(None, "rrf")], low) == ""


async def test_命中质量存疑时结果照常返回(engine, settings, monkeypatch):
    """
    只提示、不过滤——这是校准数据定的。硬过滤能挡住全部「同领域但库里没有」
    的问题，但会误杀库内两类文档：标题带格式噪声的、以及只有两三块的小文档。
    漏答比慢几分钟严重得多。
    """
    from dataclasses import replace

    real = engine.retrieve_with_diagnostics

    async def low_scored(query, *args, **kwargs):
        # 测试用 hashing 嵌入、不加载重排模型，所以命中的分数要自己伪造
        result = await real(query, *args, **kwargs)
        result["hits"] = [
            replace(hit, rerank_score=-2.0, score_type="rerank")
            for hit in result["hits"]
        ]
        return result

    monkeypatch.setattr(engine, "retrieve_with_diagnostics", low_scored)

    out, artifact = await agent_tools.search_docs(
        engine, {**settings, "agent_low_confidence_score": 1.0},
        query="REQUEST_TIMEOUT 是多少",
    )

    assert "相关性偏低" in out
    assert artifact["contexts"], "提示不等于丢结果，片段必须还在"
    assert artifact["meta"]["low_confidence"] is True
    assert artifact["meta"]["top_rerank_score"] == -2.0


async def test_search_docs_超预算时截断并声明(engine, settings):
    """
    静默截断是最难归因的失败：答案质量下降，但没有异常也没有日志。
    所以截断必须写在返回文本里让模型看得见。
    """
    out, artifact = await agent_tools.search_docs(
        engine, {"agent_tool_payload_tokens": 100}, query="故障处理流程"
    )

    assert "已按长度上限截断" in out
    assert agent_tools.estimate_tokens(out) < 400
    assert artifact["truncated"] is True


async def test_expand_section_拼回完整章节(engine, settings):
    """
    这个工具存在的全部理由：检索只给最相似的一两个片段，而步骤是跨片段的。
    三步必须都在，少一步就等于没完成它的职责。
    """
    out, _ = await agent_tools.expand_section(
        engine, settings, document="运维手册.md", section="故障处理流程"
    )

    assert "第一步" in out and "第二步" in out and "第三步" in out


async def test_expand_section_章节名只给一半也能命中(engine, settings):
    """模型抄标题时丢掉层级前缀是常见情况，精确匹配失败要退回包含匹配。"""
    out, _ = await agent_tools.expand_section(
        engine, settings, document="运维手册.md", section="故障处理"
    )

    assert "失败" not in out
    assert "第一步" in out


async def test_expand_section_章节找不到时列出可用章节(engine, settings):
    """只说「找不到」而不说有哪些，等于告诉模型失败了却不说怎么改。"""
    out, artifact = await agent_tools.expand_section(
        engine, settings, document="运维手册.md", section="不存在的章节"
    )

    assert "[expand_section 失败]" in out
    assert "超时设置" in out and "故障处理流程" in out
    assert artifact["ok"] is False


async def test_expand_section_文档不存在时列出可用文档(engine, settings):
    out, _ = await agent_tools.expand_section(
        engine, settings, document="不存在.md", section="随便"
    )

    assert "[expand_section 失败]" in out
    assert "运维手册.md" in out


async def test_find_literal_如实报告命中总数(engine, settings):
    """
    这个工具的意义就是「不悄悄截断」。总数必须是真实的全库命中数，
    而不是被 limit 截断后没人说得清的那个数。
    """
    out, artifact = await agent_tools.find_literal(
        engine, settings, text="NSR_REFRESH_TIMER"
    )

    assert "命中 2 处" in out
    assert artifact["meta"]["total"] == 2


async def test_find_literal_区分文档名写错与确实没有(engine, settings):
    """两种零命中对下一步完全不同：一个该改参数重试，一个该据实拒答。"""
    wrong_doc, wrong_art = await agent_tools.find_literal(
        engine, settings, text="REQUEST_TIMEOUT", document="不存在.md"
    )
    truly_absent, absent_art = await agent_tools.find_literal(
        engine, settings, text="ZZZ_NOT_IN_ANY_DOC"
    )

    assert "[find_literal 失败]" in wrong_doc and "可用文档" in wrong_doc
    assert "失败" not in truly_absent and "大小写敏感" in truly_absent
    # 这组断言才是这条规则的要害：两者的 ok 必须不同
    assert wrong_art["ok"] is False
    assert absent_art["ok"] is True and absent_art["empty"] is True


# ----------------------------------------------------------------------
# artifact：评测取 contexts 的唯一来源
# ----------------------------------------------------------------------

async def test_artifact_里的contexts是截断之后的那一份(engine):
    """
    RAGAS 的 faithfulness 要拿「模型实际看到的资料」去评分。
    如果 artifact 报的是截断前的原文，评的就是一个根本不存在的输入——
    分数会虚高，而且没有任何迹象表明它错了。
    """
    out, artifact = await agent_tools.search_docs(
        engine, {"agent_tool_payload_tokens": 120}, query="故障处理流程"
    )

    assert artifact["truncated"] is True
    for ctx in artifact["contexts"]:
        # 每一段 context 都必须能在给模型的文本里原样找到
        assert ctx["text"] in out


async def test_artifact_带上文档与章节(engine, settings):
    """source / heading 要能对上 expected_sources，这是不需要 LLM 评委的那类指标。"""
    _, artifact = await agent_tools.search_docs(
        engine, settings, query="REQUEST_TIMEOUT 是多少"
    )

    assert artifact["contexts"]
    assert all(c["source"] == "运维手册.md" for c in artifact["contexts"])
    assert all("text" in c and "heading" in c for c in artifact["contexts"])


async def test_工具对象会把artifact挂到ToolMessage上(engine, settings):
    """
    contexts 能不能被编排层拿到，取决于 response_format 有没有声明对。
    漏声明不会报错，只会让 artifact 静默变成 None、评测拿不到任何资料。
    """
    tool = next(
        t for t in agent_tools.build_tools(engine, settings) if t.name == "search_docs"
    )
    message = await tool.ainvoke(
        {
            "name": "search_docs",
            "args": {"query": "REQUEST_TIMEOUT 是多少"},
            "id": "call_1",
            "type": "tool_call",
        }
    )

    assert message.artifact is not None
    assert message.artifact["tool"] == "search_docs"
    assert message.artifact["contexts"]


async def test_工具内部异常转成可读失败而不上抛(engine):
    """
    向上抛异常只能让整轮对话失败；转成结构化失败，模型还能换个工具再试。
    这条契约由 build_tools 的 _guard 保证，所以要从工具对象这一层测。
    """
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("向量库炸了")

    tool = next(t for t in agent_tools.build_tools(Boom(), {}) if t.name == "search_docs")
    out = await tool.ainvoke({"query": "任何问题"})

    assert "[search_docs 失败]" in out
    assert "向量库炸了" in out


async def test_三个工具都被挂上去(engine, settings):
    """少挂一个不会报错，只会让对应那类问题悄悄退化回语义检索。"""
    names = {t.name for t in agent_tools.build_tools(engine, settings)}

    assert names == {"search_docs", "expand_section", "find_literal"}


# ======================================================================
# 编排层
# ======================================================================

async def test_取材一轮后作答的完整事件序列(engine):
    runner = make_runner(
        engine,
        [
            tool_call("search_docs", {"query": "REQUEST_TIMEOUT 默认值"}),
            AIMessage(content="默认 60 秒。\n参考来源：运维手册.md › 超时设置"),
        ],
    )

    result = await runner.run("REQUEST_TIMEOUT 默认多少秒？")

    assert result["error"] is None
    kinds = [e["type"] for e in result["events"]]
    assert "plan" in kinds and "tool" in kinds and kinds[-1] == "done"
    assert result["stages"]["agent_tools_used"] == ["search_docs"]
    assert result["stages"]["agent_retrieval_skipped"] is False
    assert "60 秒" in result["answer"]


async def test_模型如实报了用量时累计进stages(engine):
    """
    design_LangChainAgent.md 15.5 记的正是这条缺口：端点返回了 usage，
    但服务端没把它累计起来，于是「一次问答多少钱」答不出来。
    """
    runner = make_runner(
        engine,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_docs", "args": {"query": "超时"}, "id": "c1"}],
                usage_metadata={"input_tokens": 400, "output_tokens": 30, "total_tokens": 430},
            ),
            AIMessage(
                content="默认 60 秒。",
                usage_metadata={"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280},
            ),
        ],
    )

    stages = (await runner.run("REQUEST_TIMEOUT 默认多少秒？"))["stages"]

    # 两次模型调用的用量要相加，不是只记最后一次
    assert stages["agent_prompt_tokens"] == 1600
    assert stages["agent_completion_tokens"] == 110
    assert stages["agent_tokens_exact"] is True


async def test_有一次没报用量就不声称精确(engine):
    """
    继续累加一个残缺的总数、同时声称它精确，比干脆没有这个数字更糟：
    它会让「换了模型之后成本降三成」这种结论建立在一个下界上。
    """
    runner = make_runner(
        engine,
        [
            tool_call("search_docs", {"query": "超时"}),  # 没有 usage_metadata
            AIMessage(
                content="默认 60 秒。",
                usage_metadata={"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280},
            ),
        ],
    )

    stages = (await runner.run("REQUEST_TIMEOUT 默认多少秒？"))["stages"]

    assert stages["agent_tokens_exact"] is False
    assert stages["agent_prompt_tokens"] == 1200  # 拿到的那次照样记，只是不算精确


async def test_分节点耗时被记录(engine):
    """
    「这几十秒花在模型上还是检索上」是优化方向的唯一依据。
    3 节里那条「瓶颈 100% 在 LLM」的结论原本是手工读日志得出的，
    记进 stages 之后它才是一个能被回归发现的事实。
    """
    runner = make_runner(
        engine,
        [
            tool_call("search_docs", {"query": "超时"}),
            AIMessage(content="默认 60 秒。"),
        ],
    )

    stages = (await runner.run("REQUEST_TIMEOUT 默认多少秒？"))["stages"]

    assert stages["agent_model_seconds"] >= 0
    assert stages["agent_tools_seconds"] >= 0


async def test_端点不认stream_options时给出可照做的提示(engine, monkeypatch):
    """
    我们为了拿 token 用量默认带上了 stream_options，而个别 OpenAI 兼容实现
    会直接返回 400——那个报错里只字不提「用量」，用户完全无从联想到该关哪个开关。
    """
    runner = make_runner(engine, [AIMessage(content="不会走到这里")])

    async def boom(*args, **kwargs):
        raise RuntimeError("400 unrecognized request argument: stream_options")
        yield  # pragma: no cover - 让它成为异步生成器

    monkeypatch.setattr(runner.agent, "astream", boom)
    result = await runner.run("随便问一句")

    assert "LLM_STREAM_USAGE=false" in result["error"]


async def test_run返回可直接喂给RAGAS的contexts(engine):
    """
    RAGAS 的 context_precision / context_recall / faithfulness 都要这一份。
    它必须是模型实际看到的资料，而不是从答案里反推的。
    """
    runner = make_runner(
        engine,
        [
            tool_call("search_docs", {"query": "REQUEST_TIMEOUT 默认值"}),
            AIMessage(content="默认 60 秒。"),
        ],
    )

    result = await runner.run("REQUEST_TIMEOUT 默认多少秒？")

    assert result["contexts"], "取材了却没有导出 contexts，RAGAS 无法评分"
    assert all({"text", "source", "heading"} <= set(c) for c in result["contexts"])
    assert any("REQUEST_TIMEOUT" in c["text"] for c in result["contexts"])


async def test_contexts跨多轮取材去重(engine):
    """
    多轮取材几乎必然重复命中同一块：第二次的检索问句是基于第一次的材料改的，
    语义上更接近。不去重会让 context_precision 被同一段内容反复拉高或拉低。
    """
    runner = make_runner(
        engine,
        [
            tool_call("search_docs", {"query": "REQUEST_TIMEOUT"}, "c1"),
            tool_call("search_docs", {"query": "REQUEST_TIMEOUT 超时"}, "c2"),
            AIMessage(content="默认 60 秒。"),
        ],
    )

    result = await runner.run("REQUEST_TIMEOUT 默认多少秒？")

    keys = [(c["source"], c["heading"], c["text"]) for c in result["contexts"]]
    assert len(keys) == len(set(keys))


async def test_不取材时contexts为空(engine):
    """闲聊没有资料可依据。空 contexts 是正确答案，不是缺陷。"""
    runner = make_runner(engine, [AIMessage(content="我可以回答知识库里的问题。")])

    result = await runner.run("你好，你能做什么？")

    assert result["contexts"] == []


async def test_取材前的开场白不会粘进答案(engine):
    """
    模型常常在决定调工具那一轮先说一句「我查一下」，这段话会先于工具调用
    被流式推出去。不发 reset 的话它就粘在最终答案前面。
    """
    runner = make_runner(
        engine,
        [
            AIMessage(
                content="好的，我先查一下文档。",
                tool_calls=[
                    {"name": "search_docs", "args": {"query": "超时"}, "id": "c1"}
                ],
            ),
            AIMessage(content="默认 60 秒。"),
        ],
    )

    result = await runner.run("REQUEST_TIMEOUT 默认多少秒？")

    assert "我先查一下" not in result["answer"]
    assert result["answer"].strip() == "默认 60 秒。"
    assert any(e["type"] == "reset" for e in result["events"])


async def test_思考内容不进答案而是单独成事件(engine):
    """
    qwen3 这类模型在 OpenAI 兼容端点上把思考直接写在正文里用 <think> 包着。
    不切开的话，答案前面挂着一大段自言自语，落进反馈库和 chatHistory 的
    也是这段脏文本——而它恰恰是排查「模型为什么想歪了」最有用的东西，
    所以要留下来，只是不能留在答案里。
    """
    runner = make_runner(
        engine,
        [AIMessage(content="<think>用户问的是超时配置，先查文档。</think>默认 60 秒。")],
    )

    result = await runner.run("REQUEST_TIMEOUT 默认多少秒？")

    assert result["answer"].strip() == "默认 60 秒。"
    thinking = "".join(
        e["content"] for e in result["events"] if e["type"] == "reasoning"
    )
    assert "先查文档" in thinking


async def test_思考走独立字段时也能取到(engine):
    """DeepSeek 系叫 reasoning_content，Ollama 叫 reasoning，两种都得认。"""
    runner = make_runner(
        engine,
        [
            AIMessage(
                content="默认 60 秒。",
                additional_kwargs={"reasoning_content": "先确认是哪个超时。"},
            )
        ],
    )

    result = await runner.run("REQUEST_TIMEOUT 默认多少秒？")

    assert result["answer"].strip() == "默认 60 秒。"
    assert any(
        e["type"] == "reasoning" and "先确认" in e["content"] for e in result["events"]
    )


def test_标签被切片切断时不漏字也不串型():
    """
    流式下 "<think>" 会被切成 "<thi" + "nk>"。切分器若按片判断，
    半个标签要么被当正文吐出去、要么整段思考被误判成答案——
    两种都是用户直接看得见的错。
    """
    from agent.runner import _ThinkSplitter

    splitter = _ThinkSplitter()
    out = []
    for piece in ["<thi", "nk>想一", "想</thi", "nk>答案", "在此"]:
        out.extend(splitter.feed(piece))
    out.extend(splitter.flush())

    joined = {}
    for kind, text in out:
        joined[kind] = joined.get(kind, "") + text
    assert joined == {"reasoning": "想一想", "token": "答案在此"}


def test_标签没闭合时剩下的文字照样吐出去():
    """流断在标签中间是常态。宁可多一段思考，也不能把字吞掉。"""
    from agent.runner import _ThinkSplitter

    splitter = _ThinkSplitter()
    out = splitter.feed("<think>想到一半就断了")
    out.extend(splitter.flush())

    assert "".join(text for _, text in out) == "想到一半就断了"


async def test_闲聊不取材(engine):
    """
    省掉一整轮检索的这条路径是 agent 相对固定管道的主要收益之一，
    「不检索比例」也是靠这个 stage 统计出来的。
    """
    runner = make_runner(engine, [AIMessage(content="我可以回答知识库里的问题。")])

    result = await runner.run("你好，你能做什么？")

    assert result["stages"]["agent_retrieval_skipped"] is True
    assert result["stages"]["agent_tool_calls"] == 0
    assert not any(e["type"] == "tool" for e in result["events"])


async def test_工具失败被记进stages并照样收尾(engine):
    """工具失败不该让整轮对话挂掉——模型拿到失败文本后还能换个工具。"""
    runner = make_runner(
        engine,
        [
            tool_call("expand_section", {"document": "不存在.md", "section": "x"}),
            AIMessage(content="知识库里没有这篇文档。"),
        ],
    )

    result = await runner.run("把不存在.md 的内容给我")

    assert result["error"] is None
    assert result["stages"]["agent_tool_failures"] == 1
    tool_event = next(e for e in result["events"] if e["type"] == "tool")
    assert tool_event["ok"] is False


async def test_同一thread_id跨轮记住上一问(engine):
    """
    这正是接 checkpointer 的目的。第二轮只传新问题，
    上一轮的问答必须由 checkpointer 恢复进消息列表。

    这里用 InMemorySaver 而不是落盘实现：跨轮恢复是 BaseCheckpointSaver
    的接口语义，两种存储都得满足，用内存版跑得更快。落盘特有的行为
    （重启后还在）由 test_会话库重开后仍能读到上一轮 覆盖。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    script = [AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答")]

    first = make_runner(engine, script, checkpointer=saver)
    await first.run("REQUEST_TIMEOUT 是多少？", session_id="thread-1")

    second = make_runner(engine, script, checkpointer=saver)
    await second.run("那 NSR_REFRESH_TIMER 呢？", session_id="thread-1")

    seen = second.model.log["seen"][-1]
    texts = [str(getattr(m, "content", "")) for m in seen]
    assert any("REQUEST_TIMEOUT 是多少？" in t for t in texts)
    assert any("第一轮回答" in t for t in texts)


async def test_不同thread_id互不串话(engine):
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    script = [AIMessage(content="回答")]

    a = make_runner(engine, script, checkpointer=saver)
    await a.run("甲的问题", session_id="thread-a")

    b = make_runner(engine, script, checkpointer=saver)
    await b.run("乙的问题", session_id="thread-b")

    texts = [str(getattr(m, "content", "")) for m in b.model.log["seen"][-1]]
    assert not any("甲的问题" in t for t in texts)


async def test_会话库重开后仍能读到上一轮(engine, tmp_path):
    """
    换掉内存存储换来的唯一东西：重启进程不丢会话。

    两个先后打开、指向同一个文件的 saver 就是「重启」的最小模型——
    第二个实例是全新的连接和全新的进程内状态，能读到上一轮只可能来自磁盘。
    """
    from agent.runner import open_checkpointer

    db = tmp_path / "sessions.db"
    script = [AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答")]

    async with open_checkpointer(db) as saver:
        first = make_runner(engine, script, checkpointer=saver)
        await first.run("REQUEST_TIMEOUT 是多少？", session_id="thread-1")

    assert db.exists(), "会话库文件没有被创建"

    async with open_checkpointer(db) as saver:
        second = make_runner(engine, script, checkpointer=saver)
        await second.run("那 NSR_REFRESH_TIMER 呢？", session_id="thread-1")

    texts = [str(getattr(m, "content", "")) for m in second.model.log["seen"][-1]]
    assert any("REQUEST_TIMEOUT 是多少？" in t for t in texts)
    assert any("第一轮回答" in t for t in texts)


async def test_会话库未打开时退回内存存储(tmp_path):
    """
    没有 ASGI 生命周期的入口（评测脚本、单元测试）也要能跑多轮会话。
    退回内存是有意的降级，不是抛错——但块内必须拿到落盘那一个，
    否则 lifespan 注入的实例就白开了。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from agent.runner import default_checkpointer, open_checkpointer

    assert isinstance(default_checkpointer(), InMemorySaver)

    reset_default_checkpointer()
    async with open_checkpointer(tmp_path / "sessions.db") as saver:
        assert default_checkpointer() is saver

    # 出了块连接已经关掉，绝不能继续把它当单例发出去
    assert isinstance(default_checkpointer(), InMemorySaver)


def test_会话库打不开时服务仍然起得来(monkeypatch, tmp_path):
    """
    部署准则和引擎一致：配置问题不能让进程起不来，否则用户没有任何途径
    通过 UI 补救。失败要变成 /api/health/detail 里的一条原因，而不是崩溃。
    """
    from fastapi.testclient import TestClient

    import agent
    import config
    import main

    # 走 settings.json 而不是改 DEFAULTS：lifespan 读的是 load_settings()，
    # 而 isolated_settings_file 已经把它指向了本用例独占的临时文件。
    settings = config.load_settings()
    settings.update(agent_enabled=True, agent_session_enabled=True)
    config.save_settings(settings)
    monkeypatch.setattr(config, "AGENT_SESSION_DB", tmp_path / "sessions.db")

    def boom(_path):
        raise RuntimeError("磁盘只读")

    monkeypatch.setattr(agent, "open_session_store", boom)

    with TestClient(main.app) as client:
        body = client.get("/api/health/detail").json()

    assert "磁盘只读" in body["agent"]["session_error"]


async def test_无会话时用前端传来的history(engine):
    """agent_session_enabled 关闭时沿用原有行为：历史由前端维护、服务端不存。"""
    runner = make_runner(engine, [AIMessage(content="回答")])

    await runner.run(
        "那它呢？",
        history=[
            {"role": "user", "content": "REQUEST_TIMEOUT 是多少？"},
            {"role": "assistant", "content": "60 秒。"},
        ],
    )

    texts = [str(getattr(m, "content", "")) for m in runner.model.log["seen"][-1]]
    assert any("REQUEST_TIMEOUT 是多少？" in t for t in texts)


async def test_模型调用次数上限能刹住循环(engine):
    """
    没有这道闸，一个写歪的 prompt 就能让模型反复调工具直到请求超时。
    刹住之后是带着现有材料收尾，而不是报错。
    """
    runner = make_runner(
        engine,
        [tool_call("search_docs", {"query": "超时"})],  # 脚本重复最后一条 = 永远要工具
        settings={"agent_max_model_calls": 3},
    )

    result = await runner.run("死循环测试")

    assert result["error"] is None
    assert runner.model.log["calls"] <= 3


#: 每轮换一个 tool_call id，模拟「模型每轮都真的要工具」这种最坏情况。
#: 复用同一个 id 会让 tools 节点只跑一次，把图走短了，测不出真实步数。
_GREEDY_SCRIPT = [
    tool_call("search_docs", {"query": f"超时{i}"}, call_id=f"call_{i}")
    for i in range(120)
]


@pytest.mark.parametrize("calls", [2, 6, 20])
async def test_递归上限自动宽到让模型调用上限先触发(engine, calls):
    """
    两道终止条件的**顺序**：调用上限带着材料作答，递归上限直接报错中止。
    兜底先触发的话，那条优雅收尾的路径就是死代码——用户看到的是
    「Agent 达到递归上限 25 仍未结束」，而不是答案。

    这是真实踩过的坑：默认 max_model_calls=6 需要 32 步，而默认兜底是 25，
    于是默认配置下每一次「模型停不下来」都报错收场。

    这里故意把兜底配成一个明显不够的值，验证运行期会把它抬到下限之上。
    参数覆盖最小值、生产默认值和 /api/settings 允许的最大值——最后这个
    算出来是 102，已经超过接口对该字段的 le=100，所以只能在运行期兜住。
    """
    runner = make_runner(
        engine,
        _GREEDY_SCRIPT,
        settings={"agent_max_model_calls": calls, "agent_recursion_limit": 4},
    )

    assert runner.recursion_limit >= 5 * calls + 2

    result = await runner.run("死循环测试")

    assert result["error"] is None, "兜底先触发了，优雅收尾没走到"
    # 用 >= 而不是 ==：消息数超过 summary_trigger 后 SummarizationMiddleware
    # 也会调模型，而替身是同一个实例，那几次摘要调用一并记在这个计数里。
    # 它们不占图的 step，所以不影响上面的下限。
    assert runner.model.log["calls"] >= calls, "没跑满就停了，上限不是被它刹住的"


async def test_到达调用上限时给出答案而不是空回答(engine):
    """
    刹住循环之后必须有答案。

    ModelCallLimitMiddleware 的 exit_behavior="end" 只注入一条固定提示就跳到
    end，而那条消息不经过 model 节点、拿不到 token，用户看到的是一个空气泡——
    这是真实踩过的坑，比报错更难理解。所以最后一轮要摘掉工具逼模型作答。
    """
    runner = make_runner(
        engine,
        _GREEDY_SCRIPT,
        settings={"agent_max_model_calls": 4},
    )

    result = await runner.run("死循环测试")

    assert result["error"] is None
    assert result["answer"].strip(), "刹住了但没有答案，用户看到的是空气泡"
    assert runner.model.log["calls"] == 4
    # 前三轮照常带着三个工具；最后一轮的 tools 是空的，而 langchain 在工具为空
    # 时干脆不调 bind_tools，所以这里应该只有 3 条记录——少的那条就是被摘掉的。
    history = runner.model.log["bind_history"]
    assert len(history) == 3, f"摘工具的轮次不对: {history}"
    assert all(len(h) == 3 for h in history), f"前几轮不该被动过: {history}"


async def test_递归上限够大时不被改动(engine):
    """抬高只在配置不够时发生。配得比下限大就该原样生效，它仍是独立开关。"""
    runner = make_runner(
        engine,
        _GREEDY_SCRIPT,
        settings={"agent_max_model_calls": 6, "agent_recursion_limit": 80},
    )

    assert runner.recursion_limit == 80


async def test_三个工具被绑给模型(engine):
    runner = make_runner(engine, [AIMessage(content="ok")])
    await runner.run("随便问问")

    assert set(runner.model.log["bound_tools"]) == {
        "search_docs",
        "expand_section",
        "find_literal",
    }


# ======================================================================
# 接口层
# ======================================================================

def test_health报告agent状态(client):
    body = client.get("/api/health").json()

    assert "agent" in body
    assert body["agent"]["available"] is True
    assert body["agent"]["enabled"] is False  # 默认关闭


def test_agent端点返回事件轨迹(client, monkeypatch):
    """
    /api/agent 不受 agent_enabled 约束：评测要能在开关关闭时跑对照组。
    它返回的 events 是工具选择正确率那几个指标的唯一来源。
    """
    import main

    client.post("/api/upload", files=[("files", ("运维手册.md", DOC.encode(), "text/markdown"))])

    def fake_runner(engine, settings, with_session=False):
        return make_runner(
            engine,
            [
                tool_call("search_docs", {"query": "REQUEST_TIMEOUT"}),
                AIMessage(content="默认 60 秒。"),
            ],
        )

    monkeypatch.setattr(main, "build_agent_runner", fake_runner)
    body = client.post("/api/agent", json={"question": "超时是多少？"}).json()

    assert body["answer"] == "默认 60 秒。"
    assert body["stages"]["agent_tools_used"] == ["search_docs"]
    # token 事件被剔除，留下的才是可统计的轨迹
    assert all(e["type"] != "token" for e in body["events"])
    # contexts 是 RAGAS 三个指标的唯一来源，格式要和 /api/retrieve 对齐
    assert body["contexts"]
    assert all({"text", "source", "heading"} <= set(c) for c in body["contexts"])


def test_chat端点在agent关闭时走固定管道(client):
    """两种模式共用一个端点，默认那条路径不能被 agent 改动带偏。"""
    client.post("/api/upload", files=[("files", ("运维手册.md", DOC.encode(), "text/markdown"))])

    response = client.post("/api/chat", json={"question": "超时是多少？"})

    assert response.status_code == 200
    assert "模拟回答" in response.text


@pytest.fixture(autouse=True)
def _isolate_checkpointer():
    """进程级 checkpointer 是单例，用例之间必须清掉，否则会话会互相渗透。"""
    reset_default_checkpointer()
    yield
    reset_default_checkpointer()
