# -*- coding: utf-8 -*-
"""
可观测性与在线反馈的测试。

重点不在「数字算得对不对」——那部分是几行算术。重点在三件容易静默失效的事：

  1. **精确用量不会被估算值覆盖。** 覆盖了也不会报错，只会让 tokens_source
     照旧显示 "usage"，等于骗自己。
  2. **trace_id 一路贯通**（响应头 + 流末尾的 meta 事件 + 落库 + 反馈关联）。
     断在任何一环，反馈按钮都只是个装饰品。
  3. **留痕失败不影响问答。** 观测手段变成故障来源是这类改动最典型的翻车方式。
"""
from __future__ import annotations

import json

import pytest

import config
import feedback as feedback_mod
import observability


# ====================================================================
# token 与成本
# ====================================================================

class Test用量与成本:
    def test_精确用量不会被估算值覆盖(self):
        """
        这是正确性问题，不是细节：估算覆盖精确值之后 tokens_source 仍然显示
        "usage"，账单口径就无声地退化了，而且没有任何迹象表明它错了。
        """
        record = observability.RequestRecord(route="pipeline", question="问题")
        record.set_usage(1000, 200)
        record.estimate_usage(["一大段资料" * 100], "一段答案")

        assert record.usage.prompt_tokens == 1000
        assert record.usage.completion_tokens == 200
        assert record.usage.source == "usage"

    def test_没有精确值时才估算(self):
        record = observability.RequestRecord(route="pipeline", question="问题")
        record.estimate_usage(["知识库资料"], "答案")

        assert record.usage.source == "estimated"
        assert record.usage.prompt_tokens > 0
        assert record.usage.completion_tokens > 0

    def test_估算口径与工具层的预算一致(self):
        """
        工具层的 token 预算和请求级的计费必须用同一个函数，否则会出现
        「预算说 1200、账单说 1800」这种对不上账、又无从排查的情况。
        """
        from agent.tools import estimate_tokens as tools_estimate

        assert tools_estimate is observability.estimate_tokens

    def test_未配置单价时成本为零而不是瞎猜(self, monkeypatch):
        """自建服务确实不按 token 计费，此时报 0 比报一个编出来的数字诚实。"""
        monkeypatch.setattr(config, "LLM_PRICING", {})
        assert observability.cost_of("qwen3.8:27b", 10000, 5000) == 0.0

    def test_按模型名取单价(self, monkeypatch):
        """llm_model 可以热改，而云端与自建模型的成本差着几个数量级。"""
        monkeypatch.setattr(
            config,
            "LLM_PRICING",
            {
                "default": {"prompt": 0.001, "completion": 0.002},
                "qwen-plus": {"prompt": 0.0008, "completion": 0.002},
            },
        )
        assert observability.cost_of("qwen-plus", 1000, 1000) == pytest.approx(0.0028)
        # 表里没有的模型回落到 default，而不是静默算成 0
        assert observability.cost_of("某个没列的模型", 1000, 1000) == pytest.approx(0.003)

    def test_单价配置格式错误不挡启动(self, monkeypatch):
        """配置写错该告警，但不能让服务起不来——那时连改配置的界面都没有。"""
        monkeypatch.setenv("LLM_PRICING", "{这不是 json")
        with pytest.warns(UserWarning):
            assert config._load_pricing() == {}


class Test请求记录:
    def test_首token延迟只认第一次(self):
        record = observability.RequestRecord(route="pipeline", question="问题")
        record.mark_first_token()
        first = record.first_token_seconds
        record.mark_first_token()
        assert record.first_token_seconds == first

    def test_没有首token时不上报该指标(self):
        """出错的请求一个字都没吐出来，首 token 延迟不存在——不能记成 0，
        否则分位数会被一堆 0 拉下来，看起来像是变快了。"""
        record = observability.RequestRecord(route="agent", question="问题")
        data = record.finish(status="error", error="炸了")
        assert data["first_token_ms"] is None
        assert data["status"] == "error"

    def test_结构化日志是一行合法json(self, caplog):
        record = observability.RequestRecord(route="pipeline", question="问题")
        record.set_usage(100, 20)
        record.add_stage_seconds("retrieve", 0.5)
        with caplog.at_level("INFO", logger="request"):
            record.finish()

        lines = [r.getMessage() for r in caplog.records if r.name == "request"]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["route"] == "pipeline"
        assert payload["total_tokens"] == 120
        assert payload["tokens_source"] == "usage"
        assert payload["retrieve_ms"] == 500.0

    def test_stages里的诊断字段原样进日志(self):
        """「走错了工具」和「延迟高」要能在同一行日志里对上。"""
        record = observability.RequestRecord(route="agent", question="问题")
        record.stages = {"agent_tools_used": ["search_docs"], "agent_model_calls": 2}
        data = record.finish()
        assert data["agent_tools_used"] == ["search_docs"]
        assert data["agent_model_calls"] == 2


class Test指标导出:
    def test_指标接口返回文本格式(self, client):
        res = client.get("/metrics")
        if not observability.METRICS_AVAILABLE:  # pragma: no cover
            assert res.status_code == 501
            return
        assert res.status_code == 200
        assert "rag_requests_total" in res.text

    def test_关掉开关后返回404(self, client, monkeypatch):
        monkeypatch.setattr(config, "METRICS_ENABLED", False)
        assert client.get("/metrics").status_code == 404

    def test_问答之后请求计数增加(self, client, manual_text):
        if not observability.METRICS_AVAILABLE:  # pragma: no cover
            pytest.skip("未安装 prometheus_client")
        client.post("/api/upload", files={"files": ("手册.md", manual_text, "text/markdown")})
        before = client.get("/metrics").text
        client.post("/api/chat", json={"question": "Python 要什么版本？"})
        after = client.get("/metrics").text
        assert before != after
        assert 'rag_requests_total{route="pipeline",status="ok"}' in after


# ====================================================================
# trace_id
# ====================================================================

class Test链路追踪:
    def test_响应头带回trace_id(self, client):
        res = client.get("/api/health")
        assert res.headers.get("X-Trace-Id")

    def test_采纳客户端传来的trace_id(self, client):
        res = client.get("/api/health", headers={"X-Trace-Id": "upstream-abc123"})
        assert res.headers["X-Trace-Id"] == "upstream-abc123"

    def test_拒绝带换行的trace_id(self, client):
        """
        trace_id 会被写进日志文件。不校验就是一条日志注入——
        一个换行符足以在 app.log 里伪造出一整条不存在的记录。
        """
        res = client.get("/api/health", headers={"X-Trace-Id": "bad\nINFO fake line"})
        returned = res.headers["X-Trace-Id"]
        assert "\n" not in returned
        assert returned != "bad\nINFO fake line"

    def test_chat的trace_id在响应头和流末尾都能拿到(self, client, manual_text):
        """
        两个来源都要有：响应头是主渠道，但反向代理和跨域漏配 expose_headers
        都会让它静默变成 null，而丢了就等于这一问无法被反馈。
        """
        client.post("/api/upload", files={"files": ("手册.md", manual_text, "text/markdown")})
        res = client.post("/api/chat", json={"question": "Python 要什么版本？"})

        header_id = res.headers["X-Trace-Id"]
        metas = [
            json.loads(line[6:])
            for line in res.text.splitlines()
            if line.startswith("data: ") and '"meta"' in line
        ]
        assert len(metas) == 1
        assert metas[0]["trace_id"] == header_id


# ====================================================================
# 在线反馈
# ====================================================================

@pytest.fixture
def chatted(client, manual_text):
    """跑一次问答，返回 (client, trace_id)。反馈必须挂在真实的一次交互上。"""
    client.post("/api/upload", files={"files": ("手册.md", manual_text, "text/markdown")})
    res = client.post("/api/chat", json={"question": "Python 要什么版本？"})
    return client, res.headers["X-Trace-Id"]


class Test反馈接口:
    def test_点赞被记录(self, chatted):
        client, trace_id = chatted
        res = client.post("/api/feedback", json={"trace_id": trace_id, "verdict": "up"})
        assert res.status_code == 200
        assert client.get("/api/feedback/stats").json()["up"] == 1

    def test_差评带原因并进入统计分布(self, chatted):
        """总数没有行动含义，分布才有：「24 条是答案不全」直接指向路由准确率。"""
        client, trace_id = chatted
        client.post(
            "/api/feedback",
            json={"trace_id": trace_id, "verdict": "down", "reason": "incomplete"},
        )
        stats = client.get("/api/feedback/stats").json()
        assert stats["down"] == 1
        assert stats["down_reasons"]["incomplete"] == 1

    def test_重复提交按最后一次算(self, chatted):
        """用户点错了要能改，而不是攒出两条互相矛盾的记录。"""
        client, trace_id = chatted
        client.post("/api/feedback", json={"trace_id": trace_id, "verdict": "up"})
        client.post("/api/feedback", json={"trace_id": trace_id, "verdict": "down"})
        stats = client.get("/api/feedback/stats").json()
        assert (stats["up"], stats["down"]) == (0, 1)

    def test_未知trace_id照样入库(self, client):
        """
        留痕在流结束后才写，而用户可能在那之前就点了评价。
        报 404 等于惩罚手快的用户。
        """
        res = client.post(
            "/api/feedback", json={"trace_id": "还没落库的id", "verdict": "down"}
        )
        assert res.status_code == 200

    def test_非法verdict被拒(self, client):
        res = client.post("/api/feedback", json={"trace_id": "x", "verdict": "maybe"})
        assert res.status_code == 422

    def test_未知reason被拒(self, chatted):
        """原因是预置选项，放任自由值进来这张表就聚合不出任何结论。"""
        client, trace_id = chatted
        res = client.post(
            "/api/feedback",
            json={"trace_id": trace_id, "verdict": "down", "reason": "我自己编的"},
        )
        assert res.status_code == 400

    def test_原因选项由服务端给出(self, client):
        """前后端各维护一份迟早对不上，而对不上的表现是聚合结果静默失真。"""
        reasons = client.get("/api/feedback/reasons").json()["reasons"]
        assert reasons == feedback_mod.REASONS

    def test_没有反馈时满意度是未知而不是零(self, client):
        assert client.get("/api/feedback/stats").json()["satisfaction"] is None

    def test_健康检查报告反馈是否可用(self, client):
        """配置错误不预检，就只会在用户点下按钮那一刻表现成一次莫名的失败。"""
        obs = client.get("/api/health").json()["observability"]
        assert obs["feedback_enabled"] is True
        # metrics 的依赖状态只有管理员需要知道，搬去了 detail
        assert "metrics_available" in (
            client.get("/api/health/detail").json()["observability"]
        )

    def test_关闭后反馈接口返回503(self, client, monkeypatch):
        monkeypatch.setattr(config, "FEEDBACK_ENABLED", False)
        feedback_mod.reset_store()
        res = client.post("/api/feedback", json={"trace_id": "x", "verdict": "up"})
        assert res.status_code == 503


class Test交互留痕:
    def test_问答现场被落库并可导出(self, chatted):
        client, trace_id = chatted
        client.post(
            "/api/feedback",
            json={"trace_id": trace_id, "verdict": "down", "reason": "wrong"},
        )
        badcases = client.get("/api/feedback/badcases").json()["badcases"]
        assert len(badcases) == 1

        case = badcases[0]
        assert case["trace_id"] == trace_id
        assert case["question"] == "Python 要什么版本？"
        # 答案必须落库：没有答案的 badcase 是无法复查的
        assert case["answer"].startswith("这是模拟回答。")
        # contexts 同样必须在：缺了它就无法区分这是检索问题还是生成问题
        assert case["contexts"], "badcase 必须带上当时取回的资料"
        assert case["route"] == "pipeline"
        assert case["latency_ms"] > 0

    def test_落库的答案与用户看到的完全一致(self, chatted):
        """
        包括 query() 末尾追加的「参考来源」脚注。

        这里和 answer()（给 RAGAS 用的那条路径）的取舍正好相反：RAGAS 要剥掉
        脚注，因为那是 UI 元素，计入忠实度会被当作无出处的凭空断言。而 badcase
        要的是**现场**——用户抱怨「引用来源不对」时，被抱怨的就是这段脚注，
        剥掉它等于把证据删了。
        """
        client, trace_id = chatted
        client.post("/api/feedback", json={"trace_id": trace_id, "verdict": "up"})
        case = client.get("/api/feedback/badcases?verdict=up").json()["badcases"][0]
        assert case["answer"].startswith("这是模拟回答。")
        assert "参考来源" in case["answer"]

    def test_落库答案等于流式内容(self, client, manual_text):
        """服务端自己攒的那份答案必须和推给浏览器的逐字一致。"""
        client.post("/api/upload", files={"files": ("手册.md", manual_text, "text/markdown")})
        res = client.post("/api/chat", json={"question": "Python 要什么版本？"})
        trace_id = res.headers["X-Trace-Id"]

        streamed = "".join(
            json.loads(line[6:]).get("content") or ""
            for line in res.text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        )
        client.post("/api/feedback", json={"trace_id": trace_id, "verdict": "up"})
        case = client.get("/api/feedback/badcases?verdict=up").json()["badcases"][0]
        assert case["answer"] == streamed

    def test_固定管道的token有估算值兜底(self, chatted):
        """
        假模型不返回 usage，此时必须退回估算而不是报 0——
        报 0 会让「一次问答多少钱」这个问题得到一个错误的答案。
        """
        client, trace_id = chatted
        client.post("/api/feedback", json={"trace_id": trace_id, "verdict": "up"})
        case = client.get("/api/feedback/badcases?verdict=up").json()["badcases"][0]
        assert case["tokens_source"] == "estimated"
        # prompt 估算必须包含检索到的资料，不能只算问题正文
        assert case["prompt_tokens"] > 50

    def test_留痕失败不影响问答(self, client, manual_text, monkeypatch):
        """观测手段自己成了故障源，是这类改动最典型的翻车方式。"""
        store = feedback_mod.get_store()

        def boom(*args, **kwargs):
            raise RuntimeError("磁盘满了")

        monkeypatch.setattr(store, "_conn", None)
        monkeypatch.setattr(type(store), "_prune", boom, raising=False)

        client.post("/api/upload", files={"files": ("手册.md", manual_text, "text/markdown")})
        res = client.post("/api/chat", json={"question": "Python 要什么版本？"})
        assert res.status_code == 200
        assert "模拟回答" in res.text


class Test淘汰策略:
    def test_超出上限时淘汰没人评价过的最老记录(self, tmp_path, monkeypatch):
        """
        「只增不减」是一条磁盘泄漏（和日志要滚动、会话要 TTL 同理），
        但被点过评价的记录正是这张表存在的理由，不能参与淘汰。
        """
        monkeypatch.setattr(config, "FEEDBACK_MAX_ROWS", 3)
        store = feedback_mod.FeedbackStore(tmp_path / "fb.db")
        try:
            for i in range(6):
                store.record_interaction(
                    {"trace_id": f"t{i}", "route": "pipeline", "status": "ok",
                     "question": f"问题{i}"},
                    answer=f"答案{i}",
                )
                if i == 0:
                    # 最老的那条被评价过，必须留下来
                    store.record_feedback("t0", "down", "wrong")

            kept = {
                row["trace_id"]
                for row in store._conn.execute(
                    "SELECT trace_id FROM interactions"
                ).fetchall()
            }
            assert "t0" in kept, "被评价过的记录不该被淘汰"
            assert "t5" in kept, "最新的记录不该被淘汰"
            assert len(kept) <= 4  # 3 条上限 + 受保护的 t0
        finally:
            store.close()
