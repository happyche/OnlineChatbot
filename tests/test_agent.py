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
        self.log["bound_tools"] = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log["calls"] += 1
        self.log.setdefault("seen", []).append(messages)
        index = min(self.log["calls"] - 1, len(self.script) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])


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
    这正是接 InMemorySaver 的目的。第二轮只传新问题，
    上一轮的问答必须由 checkpointer 恢复进消息列表。
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
