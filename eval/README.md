# 评测数据说明

## 会提交到 Git 的内容

| 路径 | 说明 |
|------|------|
| `questions.example.jsonl` | 公开示例问题集（8 条），对应下方示例语料 |
| `questions.multiturn.example.jsonl` | 多轮追问示例（4 条，含 `history` 字段） |
| `corpus.example/` | 公开示例语料（合成运维手册，无版权限制） |

快速验证检索扫描：

```bash
python scripts/validate_questions.py eval/questions.example.jsonl --corpus eval/corpus.example
python scripts/retrieval_sweep.py --questions eval/questions.example.jsonl --corpus eval/corpus.example
```

## 不会提交的内容（见根目录 `.gitignore`）

以下文件**保留在本地**供后续评测使用，仅不进入 Git：

| 路径 | 原因 |
|------|------|
| `corpus/` | 可能含受限/内部手册原文 |
| `questions.hss.jsonl` | 与内部 HSS 语料绑定 |
| `sweep_*.jsonl`、`ragas_*.jsonl`、`ragas_*report.md`、`samples*.jsonl` 等 | 评测运行产物，可重复生成 |

本地若已有上述文件，可照常跑完整 HSS 评测；重新跑脚本会覆盖同名输出文件。
