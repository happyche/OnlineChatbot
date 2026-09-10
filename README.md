# RAG 智能文档助手

面向企业内部文档的在线问答系统：上传资料、自动建库，结合检索增强生成（RAG）流式回答，并附章节级出处。

## 功能

- **智能问答** — 混合检索 + 大模型生成，回答带参考来源
- **文档管理** — 拖拽上传、自动切分索引，支持一键重建
- **多轮对话** — 支持上下文追问；检索侧可改写指代问题
- **模型配置** — 页面内切换 LLM / 嵌入 / 检索开关，热重载无需重启
- **Docker 部署** — `docker compose up` 一键启动（模型构建时预下载）

## 界面

<img width="1278" alt="问答界面" src="https://github.com/user-attachments/assets/39e853e8-874c-4df5-afa4-2ade0e1166e6" />

<img width="888" alt="嵌入与检索参数" src="docs/screenshots/settings-embedding.png" />

<img width="888" alt="检索增强开关" src="docs/screenshots/settings-retrieval.png" />

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp env.example .env             # 填入 API Key 或自建 LLM 地址
python main.py
```

浏览器打开 `http://localhost:8000`。

**Docker：**

```bash
cp env.docker.example .env
docker compose up --build
```

常用环境变量见 `env.example`（对话走云端或 Ollama/vLLM、嵌入默认本地 BGE、混合检索与重排开关等）。

### 依赖文件

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 跑服务（Docker 也只装这份） |
| `requirements-dev.txt` | 开发 + 单元测试 |
| `requirements-eval.txt` | 可选：RAGAS 评测与 PDF 转语料（体积大，按需安装） |

## 测试

```bash
pip install -r requirements-dev.txt
pytest                          # 离线测试，无需 API Key / 模型权重
```

针对**已启动的服务**做 HTTP 冒烟（真实嵌入 + 真实接口）：

```bash
python main.py                  # 另开终端
python scripts/smoke_test.py    # 默认 http://127.0.0.1:8000
```

接入 LLM 前可先检查端点是否通：

```bash
python scripts/check_llm.py
python scripts/check_llm.py --base-url http://192.168.1.10:11434/v1 --model qwen2.5:7b
```

首次使用本地嵌入/重排前，可预下载模型权重（避免首次索引时卡住）：

```bash
python scripts/download_model.py
python scripts/download_model.py --reranker   # 重排模型约 1GB
```

## 评测

公开示例数据在 `eval/corpus.example/` 与 `eval/questions.example.jsonl`，可直接跑通流程。

**1. 召回消融（Hit@K / MRR，不调 LLM 生成）**

```bash
pip install -r requirements-dev.txt   # 已含主线依赖

python scripts/validate_questions.py eval/questions.example.jsonl --corpus eval/corpus.example
python scripts/retrieval_sweep.py --questions eval/questions.example.jsonl --corpus eval/corpus.example
```

**2. RAGAS（LLM 裁判，评检索与生成质量）**

```bash
pip install -r requirements-eval.txt

# 只评检索类指标（省调用）
python scripts/ragas_eval.py --questions eval/questions.example.jsonl \
    --corpus eval/corpus.example --metrics context_precision,context_recall

# 完整评测（需配置 .env 中的 LLM）
python scripts/ragas_eval.py --questions eval/questions.example.jsonl --corpus eval/corpus.example
```

仅导出样本、不装 ragas 也可：

```bash
python scripts/ragas_eval.py --questions eval/questions.example.jsonl \
    --corpus eval/corpus.example --dump-only
```

**3. 其他**

| 命令 | 说明 |
|------|------|
| `python scripts/pdf_to_markdown.py 手册.pdf` | PDF 转可入库文本（评测语料准备） |
| `python scripts/docker_verify.py` | Docker 部署后健康检查 |
| `POST /api/retrieve` | 只检索不生成，便于调试召回 |

本地若有完整语料与问题集（未上传 Git），把路径换成 `eval/corpus`、`eval/questions.hss.jsonl` 即可。详见 `eval/README.md`。

## 技术栈

| 层次 | 选型 |
|------|------|
| 后端 | FastAPI · uvicorn · 全异步 |
| 向量库 | ChromaDB（本地持久化） |
| 嵌入 | fastembed / BGE-small-zh（默认本地 ONNX） |
| 检索 | 稠密向量 · BM25 · RRF 融合 · Cross-Encoder 重排 |
| 大模型 | OpenAI 兼容接口（DashScope / Ollama 等） |
| 前端 | 原生 HTML/CSS/JS · SSE 流式 · DOMPurify |

## 评测结果

在真实企业语料（196 块、49 条中文问题）上，对四种检索组合做消融（`top_k=5`，嵌入 `bge-small-zh-v1.5`）：

**关键词指标（Hit@K / MRR）**

| 组合 | Hit@5 | MRR | 相对 baseline |
|------|-------|-----|---------------|
| baseline（仅向量） | 75.0% | 0.588 | — |
| + hybrid（BM25+RRF） | 86.4% | 0.749 | MRR +27.5% |
| + rerank | 81.8% | 0.726 | MRR +23.6% |
| hybrid + rerank | **86.4%** | **0.790** | **MRR +34.4%** |

**RAGAS 检索质量（LLM 裁判，49 题）**

| 组合 | context_precision | context_recall |
|------|-------------------|----------------|
| baseline | 0.599 | 0.706 |
| hybrid | 0.721 | 0.871 |
| rerank | 0.755 | 0.804 |
| hybrid + rerank | **0.782** | **0.869** |

复现命令见上文「评测」一节；完整 HSS 语料因敏感信息未上传仓库。
