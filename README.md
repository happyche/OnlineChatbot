# RAG 智能文档助手

基于 Qwen 大模型的企业文档问答系统。上传 Markdown 格式的内部说明书，系统自动构建向量索引，支持实时检索增强生成（RAG）。

> 设计决策、取舍依据、踩过的坑与验证结果见 [DESIGN.md](DESIGN.md)。

## 功能

- **智能问答** — 检索内部文档并结合大模型生成回答，回答附带精确到章节的出处
- **文档管理** — 拖拽上传 Markdown，按标题层级切分并建立向量索引
- **流式输出** — 回答实时流式展示，支持 Markdown 渲染与代码高亮
- **模型配置** — 通过 UI 实时切换 LLM、Embedding 模型与检索参数，无需重启
- **索引重建** — 更换嵌入模型或调整切分参数后一键重新索引

## 快速开始

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1
                                  # Git Bash:   source .venv/Scripts/activate
                                  # macOS/Linux: source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp env.example .env
# 编辑 .env：云端模型填 LLM_API_KEY；自建服务填 LOCAL_LLM_BASE_URL

# 4. 启动服务
python main.py
```

浏览器访问 `http://localhost:8000`。

> **`ModuleNotFoundError: No module named 'chromadb'`**
> 说明当前用的不是虚拟环境里的解释器。Windows 上 `python` 常指向应用商店版
> （`WindowsApps\python.exe`），它没有本项目的依赖。
> 要么先激活虚拟环境，要么直接指定解释器：
>
> ```bash
> .venv\Scripts\python.exe main.py
> ```
>
> 用 `python -c "import sys; print(sys.prefix)"` 可以确认当前解释器 ——
> 输出应指向项目下的 `.venv`。

## 配置说明

### 环境变量 (.env)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | `openai`（云端 API）/ `local`（自建服务） | `openai` |
| `LLM_API_KEY` | 云端 API Key（`openai` 时生效） | - |
| `LLM_BASE_URL` | 云端接口地址（`openai` 时生效） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | 云端对话模型名 | `qwen-plus` |
| `LOCAL_LLM_BASE_URL` | 自建服务地址（`local` 时生效） | `http://localhost:11434/v1` |
| `LOCAL_LLM_MODEL` | 自建服务的模型名 | `qwen2.5:7b` |
| `LOCAL_LLM_API_KEY` | 自建服务令牌，多数情况留空 | 空 |
| `EMBEDDING_PROVIDER` | `local`（本地 ONNX）/ `openai`（远程） | `local` |
| `LOCAL_EMBEDDING_MODEL` | 本地嵌入模型（`local` 时生效） | `BAAI/bge-small-zh-v1.5` |
| `EMBEDDING_CACHE_DIR` | 本地模型权重缓存目录 | `~/.cache/fastembed` |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOC_PREFIX` | 查询/文档指令前缀，bge-v1.5 无需 | 空 |
| `EMBEDDING_MODEL` | 远程嵌入模型（`openai` 时生效） | `text-embedding-v3` |
| `EMBEDDING_BASE_URL` | 嵌入接口地址，留空则复用对话端点 | 空 |
| `EMBEDDING_API_KEY` | 嵌入接口 Key，留空则复用对话端点的 Key | 空 |
| `EMBEDDING_BATCH_SIZE` | 单次嵌入请求的文本条数 | `10` |
| `TEMPERATURE` | 生成温度 | `0.7` |
| `CHUNK_SIZE` | 分片大小（字符） | `500` |
| `CHUNK_OVERLAP` | 相邻分片重叠字符数 | `50` |
| `TOP_K` | 检索返回片段数 | `5` |
| `MIN_SIMILARITY` | 向量路的相似度下限，低于此值的片段被丢弃 | `0.2` |
| `HYBRID_SEARCH_ENABLED` | 开启混合检索（BM25 + 向量，RRF 融合） | `false` |
| `BM25_TOKENIZER` | `jieba`（词粒度）/ `bigram`（字符二元组） | `jieba` |
| `RRF_K` | RRF 平滑常数 | `60` |
| `RERANK_ENABLED` | 开启交叉编码器重排 | `false` |
| `RERANK_PROVIDER` | `local`（交叉编码器）/ `lexical`（字面重叠，冒烟用） | `local` |
| `RERANK_MODEL` | 重排模型 | `BAAI/bge-reranker-base` |
| `MIN_RERANK_SCORE` | 重排分数下限，留空不过滤 | 空 |
| `CANDIDATE_POOL_SIZE` | 融合/重排前每路的候选数 | `20` |
| `HISTORY_LIMIT` | 送入模型的历史消息条数（2 条为一轮） | `10` |
| `REQUEST_TIMEOUT` | 上游请求超时（秒） | `60` |
| `BYPASS_PROXY_FOR_INTERNAL` | 访问内网/本机地址时绕过 `HTTP_PROXY` | `true` |
| `CORS_ORIGINS` | 允许跨域的来源，逗号分隔 | 本机地址 |
| `APP_API_TOKEN` | 设置后 `/api/*` 需带 `X-API-Token` | 空（不鉴权） |

### 对话模型：云端 API 或自建服务

两条路都支持，在页面的「模型配置」里切换，或用 `LLM_PROVIDER` 配置。

**自建服务**（`LLM_PROVIDER=local`）指任何暴露了 OpenAI 兼容接口的推理服务，
在本机或内网服务器上均可。只需填服务地址与模型名：

```bash
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://192.168.1.10:11434/v1   # 填服务器实际地址
LOCAL_LLM_MODEL=qwen2.5:7b
```

地址必须是 OpenAI 兼容路径，一般以 `/v1` 结尾。常见默认端口：

| 推理服务 | 地址示例 |
|---------|---------|
| Ollama | `http://<host>:11434/v1` |
| vLLM | `http://<host>:8000/v1` |
| LM Studio | `http://<host>:1234/v1` |
| llama.cpp server | `http://<host>:8080/v1` |

多数自建服务不校验 API Key（Ollama 直接忽略）；vLLM 若启用了 `--api-key`，
在 `LOCAL_LLM_API_KEY` 中填写。本地模型首字延迟通常更长，必要时调大
`REQUEST_TIMEOUT`。

#### 企业代理会拦截内网请求

企业环境普遍设置了 `HTTP_PROXY`，而 HTTP 客户端默认读取该环境变量，
于是访问内网自建服务的请求会被发到代理上，返回一段 HTML 错误页 ——
报错形如 `InternalServerError: <!-- IE friendly error message...`，
完全指向错误的方向（看起来像服务端 500，其实是本地代理配置）。

本项目会按私有地址段（`10.x`、`172.16-31.x`、`192.168.x`、`127.x`、
`localhost`）**自动绕过代理**，无需手工配置 `NO_PROXY`。
少数确实需要经代理访问内网的场景可以关掉：

```bash
BYPASS_PROXY_FOR_INTERNAL=false
# 此时改用标准做法，把内网地址加进 NO_PROXY
NO_PROXY=localhost,127.0.0.1,10.67.34.44
```

注意主机名（而非 IP）无法判断是否内网，这种情况仍需显式配置 `NO_PROXY`。

#### 从别的机器连 Ollama 连不上？

Ollama 默认只监听 `127.0.0.1`，本机能访问但局域网不行。需要在**服务器上**
让它监听所有网卡后重启服务：

```bash
# Linux (systemd)：systemctl edit ollama.service 添加
Environment="OLLAMA_HOST=0.0.0.0:11434"

# 或直接以环境变量启动
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

再确认防火墙放行了 11434 端口。模型名必须带 tag（如 `qwen2.5:7b` 而不是
`qwen2.5`），用 `ollama list` 或本项目的 `scripts/check_llm.py` 可以看到准确名称。

#### 嵌入模型也可以放在服务器上

Ollama 的 `/v1` 同样提供 `/embeddings`，因此嵌入也能走服务器，省掉本地下载权重：

```bash
# 服务器上： ollama pull bge-m3
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=http://192.168.1.10:11434/v1
EMBEDDING_MODEL=bge-m3
```

`bge-m3` 是 1024 维的多语言检索模型，中文质量优于本地默认的
`bge-small-zh-v1.5`，代价是每次检索多一次网络往返。
注意切换嵌入模型后需要重新索引。

接入前建议先单独验证端点，能把「地址错」「模型名错」「需要令牌」区分开：

```bash
python scripts/check_llm.py --base-url http://192.168.1.10:11434/v1 --model qwen2.5:7b
# 或直接检查当前配置生效的端点
python scripts/check_llm.py
```

页面「模型配置」里也有「测试连接并列出模型」按钮，作用相同。
`GET /api/settings` 与 `GET /api/health` 的 `active` 字段会回报当前**实际生效**
的端点与模型——两套配置并存时，这是判断哪套在生效的依据。

**对话与嵌入相互独立**：可以对话走内网自建服务、嵌入走云端，反之亦可。
此时注意用 `EMBEDDING_BASE_URL` 与 `EMBEDDING_API_KEY` 单独配置嵌入端点，
否则会拿自建服务的占位令牌去请求云端。

### 嵌入：默认走本地，文本不出本机

默认使用本地 `BAAI/bge-small-zh-v1.5`（约 90MB、512 维）做 ONNX 推理：
离线可用、无调用费用、文档内容不会离开本机 —— 内部文档场景通常需要这一点。
模型在首次索引文档时惰性加载，因此服务启动很快。

底层用 `fastembed` 而非 `sentence-transformers`：后者依赖 PyTorch，
Windows 上要额外装数百 MB；前者只需 `onnxruntime`，而 ChromaDB 已带该依赖。

建议部署前先预下载权重，避免首次使用时被 Hugging Face 限流拖慢：

```bash
python scripts/download_model.py
# 缓存损坏时： python scripts/download_model.py --clean
```

若网络受限，可设置 `HF_TOKEN` 提高速率上限，或用镜像
`HF_ENDPOINT=https://hf-mirror.com`。

**嵌入模型必须与语料语言匹配。** 中文知识库不要使用纯英文模型
（如 `all-MiniLM-L6-v2`），否则检索结果接近随机。

### 改用远程嵌入

设置 `EMBEDDING_PROVIDER=openai` 即切换为远程接口（如 `text-embedding-v3`）。
适合本地资源紧张或希望免维护模型文件的场景，代价是需要联网、按量计费，
且文档内容会发送到第三方。

### 对话与嵌入可能不在同一端点

DashScope 的 coding 专用端点（`coding.dashscope.aliyuncs.com/v1`）只提供
chat，请求 `/embeddings` 会返回 404。若 `LLM_BASE_URL` 指向该端点，
必须把 `EMBEDDING_BASE_URL` 设为标准端点
`https://dashscope.aliyuncs.com/compatible-mode/v1`。

### 更换嵌入模型后必须重新索引

向量库会记录产出这批向量的嵌入模型指纹。更换模型后旧向量与新查询向量
既不在同一语义空间、维度也不同，检索结果没有意义，因此服务启动时会检测到
不兼容并**清空重建** collection。此时需要重新索引：

```bash
curl -X POST http://localhost:8000/api/reindex
```

或在 UI 保存配置后按提示确认。索引会从 `uploads/` 目录下的文档重建。

## 检索增强与消融评估

检索管道分三段，后两段各由一个开关控制，可任意组合：

```
向量检索 →（可选）BM25 + RRF 融合 →（可选）交叉编码器重排 → 截到 Top K
```

**混合检索**（`HYBRID_SEARCH_ENABLED`）在向量检索之外并行跑一路 BM25。
向量检索靠语义相似度，对精确串很弱 —— 技术文档里的 `kubectl exec`、
`REQUEST_TIMEOUT`、错误码这类词，字面命中往往比语义相近更可靠。
两路结果用 RRF（Reciprocal Rank Fusion）融合：只依赖排名而非原始分数，
因此不需要为余弦相似度和 BM25 分数这两种量纲做归一化调参。

中文 BM25 必须先分词，否则整句会变成一个 token。默认用 jieba，
也可切到字符二元组（`BM25_TOKENIZER=bigram`）—— 后者不受词典覆盖率影响，
产品术语和新词多的语料可以对比后再定。

**重排**（`RERANK_ENABLED`）用交叉编码器对候选精排。向量检索是双塔结构，
问题与文档分别编码，无法建模两者的细粒度交互；交叉编码器把
(问题, 文档) 拼在一起打分，精度更高但成本也高，因此只对候选池里的
少量文档执行。首次启用需下载约 1GB 权重，建议先预下载：

```bash
python scripts/download_model.py --reranker
```

`MIN_SIMILARITY` 只作用于向量这一路：BM25 命中的片段即使余弦分低也会保留，
这正是混合检索的意义 —— 捕捉语义相近度不高但字面精确匹配的内容。

### 量化对比各环节的贡献

是否值得开启取决于具体语料与问题分布，只能靠数据决定。`POST /api/retrieve`
只做检索不生成回答，返回命中片段与各阶段得分明细（余弦分、BM25 分、
RRF 分、重排分），既省掉生成的时间与费用，也避免生成环节的波动干扰归因。

配套的扫描脚本会把同一批问题跑遍各开关组合，导出可直接喂给 RAGAS 的 JSONL：

```bash
python scripts/retrieval_sweep.py \
    --questions eval/questions.hss.jsonl \
    --corpus eval/corpus               # 现建临时索引，不动现有向量库
```

输出形如（49 条问题、196 个文本块的真实语料、`top_k=5`）：

```
组合                 问题数      平均条数    Hit@5     MRR     P@5      首位命中
baseline            49      5.00    75.0%   0.588   0.195     50.0%
hybrid              49      5.00    86.4%   0.749   0.250     68.2%
rerank              49      5.00    81.8%   0.726   0.232     65.9%
hybrid+rerank       49      5.00    86.4%   0.790   0.291     72.7%

相对 baseline 的 MRR 变化：
  hybrid          +0.161 (+27.5%)
  rerank          +0.139 (+23.6%)
  hybrid+rerank   +0.202 (+34.4%)
```

**判断重排是否有效要看 MRR**，不能只看 Hit@K：重排改变的是名次而非召回
集合，只有 MRR 和 Precision@K 才能反映出「正确片段被提到多前面」。
上面 `hybrid` 与 `hybrid+rerank` 的 Hit@5 完全相同（都是 86.4%），
但 MRR 差了 0.041，正是这个现象的实例。
问题集里给出 `expected_keywords` 后脚本会自动算这几个指标。

延迟方面重排在 CPU 上大约每候选 70–90 ms，默认池大小（20）
会给每次查询增加约 1.6 秒；BM25 则是毫秒级。

自带的问题集针对英文技术手册语料，问题用中文提出但保留英文技术术语，
这既是中文工程师查英文文档的真实提问方式，也让 BM25 有标识符可命中。
换成自己的语料时，用 `scripts/validate_questions.py` 先校验一遍：
`expected_keywords` 若在语料里根本不存在，那道题的 Hit@K 会恒为 0
而且不报任何错，只是让指标悄悄偏低。

### 用 RAGAS 做 LLM 裁判评测

关键词匹配零成本但粗糙（片段里出现某个词不代表它真的回答了问题）。
要判断「召回的片段到底相不相关」「答案有没有编造」，得让 LLM 来评。

评测依赖较重（会引入 langchain、datasets 等包），因此单独成文件：

```bash
pip install -r requirements-eval.txt

# 完整评测：检索质量 + 生成质量
python scripts/ragas_eval.py \
    --questions eval/questions.hss.jsonl \
    --corpus eval/corpus

# 只评检索，跳过答案生成，调用量减半
python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
    --metrics context_precision,context_recall

# 评委独立于被测模型（推荐：同模型自评会偏袒自己）
python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
    --llm-model qwen2.5:latest --judge-model qwen3:8b

# 先只导出样本、不算指标，用来确认数据管道通了（无需安装 ragas）
python scripts/ragas_eval.py --questions eval/questions.hss.jsonl --dump-only
```

问题集格式与消融扫描相同，其中 `ground_truth` 会作为 RAGAS 的 `reference`；
缺 `ground_truth` 的问题会自动跳过需要参考答案的两个指标。

四个指标：`context_precision`（相关片段是否排在前面）、
`context_recall`（该召回的是否都召回了）、
`faithfulness`（答案的每个断言是否有出处）、
`answer_relevancy`（答案是否答在点上）。前两个只看检索，不需要生成答案。

输出 `eval/ragas_results.jsonl`（逐条明细）和 `eval/ragas_report.md`（汇总表）。
实测（196 块语料、49 条问题、评委 `gpt-oss:20b`、只跑检索类指标）：

```
| 组合          | 问题数 | context_precision | context_recall |
| baseline      |  49   |   0.599 (n=44)    |  0.706 (n=44)  |
| hybrid        |  49   |   0.721 (n=44)    |  0.871 (n=44)  |
| rerank        |  49   |   0.755 (n=44)    |  0.804 (n=43)  |
| hybrid+rerank |  49   |   0.782 (n=44)    |  0.869 (n=44)  |
```

结论方向与关键词指标一致（baseline 最差、hybrid+rerank 最好），
两种独立的相关性判定方法互相印证。

`n=44` 是因为拒答类问题（`type` 为 `unanswerable`）故意不带 `ground_truth`，
两个需要 reference 的指标会自动跳过它们；它们只在跑 `faithfulness`
时才有意义 —— 看模型会不会在知识库没有答案时硬编。
392 次评判耗时约 64 分钟，这个量级说明为什么日常迭代该用消融扫描、
只在要下结论时才跑 RAGAS。

两个使用上的注意点：

- **`answer_relevancy` 的绝对值不可跨配置比较**。它是把答案反推出的问题
  与原问题比向量余弦，数值高度依赖所用的嵌入模型
  （`bge-small-zh` 对中文近义句本来就集中在 0.5–0.7）。
- **评委输出被截断会导致指标随机缺失**。会思考的模型（qwen3、deepseek-r1）
  容易触发，脚本跑完会归类提示，按提示调大 `--judge-max-tokens` 即可。
  汇总表里的 `(n=7)` 表示该指标只有 7 个样本算成功，别当成 10 个的均值读。

### 用第二个评委交叉验证

LLM 裁判的可靠性取决于评委本身，而评委水平没法直接测量。间接办法是
让两个独立评委评同一批样本：一致不能证明都对，但不一致足以说明至少有一个不可信。

两个评委必须评**同一批样本**——重跑 RAG 会生成不同的答案，
那时比出来的是答案差异而不是评委差异。所以先 dump 再分别打分：

```bash
python scripts/ragas_eval.py --questions eval/questions.hss.jsonl \
    --corpus eval/corpus --dump-only --output eval/samples.jsonl

python scripts/ragas_eval.py --samples eval/samples.jsonl \
    --judge-model qwen3:8b    --output eval/a.jsonl
python scripts/ragas_eval.py --samples eval/samples.jsonl \
    --judge-model gpt-oss:20b --output eval/b.jsonl

python scripts/ragas_compare.py eval/a.jsonl eval/b.jsonl
```

对比报告给三层信息：均值差（两个评委的宽严差异）、**Spearman 秩相关**
（对样本好坏的排序是否一致）、分歧最大的几条。看秩相关而不是绝对分，
是因为评测真正关心的是「A 配置是否优于 B」这个次序问题——
只要排序一致，评委偏松偏严都不影响横向比较的结论。

想要一个与本地 Ollama 完全独立的第三方意见，可以用 Cursor 的 agent 当评委：

```bash
pip install cursor-sdk
# 在 Cursor Dashboard → API Keys 生成后写入 .env 的 CURSOR_API_KEY
python scripts/ragas_eval.py --samples eval/samples.jsonl \
    --judge-provider cursor --cursor-model gpt-5.6-sol --output eval/c.jsonl
```

它会关掉 agent 的所有工具、并让 agent 在一个空临时目录里运行——一个能上网
的评委在判 faithfulness 时会引入 contexts 之外的知识，测出来的就不是忠实度了。

两个注意点：

- **模型能列出不等于能用。** `models.list()` 给的是账号看得到的目录，
  未开通的模型不报鉴权错，只是每次 run 静默返回 `status=error`。
  脚本会在开跑前预检一次，不可用会立刻报错而不是每格重试三次白烧几十次调用。
- **agent run 没有 temperature/seed，分数不可复现**，开销也大得多
  （实测 10 题 × 2 指标约 60 次 agent run、5 分钟，因为 `context_precision`
  逐个 context 各判一次，调用量随 `top_k` 线性增长）。
  所以只适合交叉验证，不能当回归基准线；模型 id 也别用 `auto`。

**实测四个评委在同一批 10 条样本上评 `context_precision`：**

| 评委 | 均值 | | 评委对 | 秩相关 |
|---|---|---|---|---|
| qwen3:8b | 0.890 | | gpt-oss:20b ↔ gpt-5.6-sol | **+0.995** |
| gpt-oss:20b | 0.873 | | composer-2.5 ↔ gpt-5.6-sol | +0.818 |
| composer-2.5 | 0.903 | | gpt-oss:20b ↔ composer-2.5 | +0.766 |
| gpt-5.6-sol | 0.887 | | qwen3:8b ↔ 其余三个 | +0.088 ~ +0.361 |

四个均值的极差只有 0.030，看上去高度吻合；但秩相关分成两层：后三个评委
构成紧密的共识簇，`qwen3:8b` 跟三者全对不上——**三对一，它是离群评委**，
而且证据来自两个不同供应方，不是同源模型的共同偏差。
只做一组两两对比得不出这个结论。

更实用的一条：本地免费的 `gpt-oss:20b` 与前沿的 `gpt-5.6-sol` 秩相关高达
**+0.995**。所以**日常回归评测用本地模型就够了**，只在阶段性下结论时用强模型
抽查锚定一次。本地评委还有个 agent 给不了的性质：两个本地评委各自重跑一遍，
逐条分数完全一致——回归评测要的正是这种确定性，分数变了就一定是被测系统变了。

结论：换评委前先跑一遍交叉验证，确认秩相关够高再引用数字。

**指标定义、相关性判定方法、与业界标准（nDCG / BEIR / MTEB）的对照，
以及这套方法的局限**，见 [DESIGN.md 第 8 节](DESIGN.md)：
8.3 是评估方法与指标定义，8.4 是关键词指标的实测解读，
8.5 是 RAGAS 接入的设计取舍与实测结果。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查，含引擎状态与生效端点 |
| GET | `/api/models` | 列出对话端点可用模型（兼作连通性检查） |
| POST | `/api/chat` | 流式问答（SSE） |
| POST | `/api/retrieve` | 只检索不生成，返回片段与各阶段得分明细 |
| POST | `/api/upload` | 上传 Markdown 并索引 |
| POST | `/api/reindex` | 用当前配置重建全部索引 |
| GET | `/api/documents` | 列出已入库文档 |
| DELETE | `/api/documents/{name}` | 删除文档 |
| GET | `/api/settings` | 获取配置（不含 Key 明文） |
| POST | `/api/settings` | 更新配置并热重载 |

## 安全须知

- `.env` 与 `settings.json` 均含 API Key 明文，已在 `.gitignore` 中排除，**切勿提交**
- 若 Key 曾经进入过版本库或被分享，请立即在控制台吊销并重建
- 服务默认不鉴权，仅适合本机使用；对外暴露前请设置 `APP_API_TOKEN` 并收紧 `CORS_ORIGINS`

## 测试

```bash
pip install -r requirements-dev.txt

# 单元测试与接口测试（离线，使用确定性嵌入，无需 API Key 与模型权重）
# 含 RAGAS 评测链路的装配测试，不需要装 requirements-eval.txt
pytest

# 额外跑真实加载 bge 中文模型的检索质量用例（需已下载权重）
RUN_LOCAL_EMBED=1 pytest tests/test_local_embedder.py

# 针对运行中的服务做冒烟测试（真实 HTTP + 真实嵌入模型）
python main.py                     # 另开一个终端
python scripts/smoke_test.py
```

没有可用的推理服务时，可以用内置的 OpenAI 兼容测试替身跑通完整链路：

```bash
python scripts/mock_llm_server.py --port 9911     # 终端 1
LLM_PROVIDER=local LOCAL_LLM_BASE_URL=http://127.0.0.1:9911/v1 \
  LOCAL_LLM_MODEL=mock-model python main.py       # 终端 2
python scripts/smoke_test.py                      # 终端 3
```

该替身不做推理，只回显收到的参考资料条数与出处，因此可以直接验证
「检索到的内容有没有真的进入 prompt」。

## 技术栈

- **后端**: FastAPI + uvicorn，全异步（`AsyncOpenAI`），流式响应不阻塞事件循环
- **向量数据库**: ChromaDB（本地持久化，余弦相似度）
- **Embedding**: 默认本地 `bge-small-zh-v1.5`（fastembed / ONNX），可切换为远程接口
- **LLM**: 通过 OpenAI 兼容接口调用 Qwen（DashScope），也支持任何兼容端点
- **前端**: 原生 HTML/CSS/JS 单页应用，`marked` 渲染 + `DOMPurify` 净化
