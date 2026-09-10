# RAG Chatbot — 生产镜像（嵌入 + 重排模型在构建期预下载，启动即可用）
#
# 构建:
#   docker build -t onlinechatbot .
# 国内网络建议在构建时传入镜像站:
#   docker build --build-arg HF_ENDPOINT=https://hf-mirror.com -t onlinechatbot .

FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 模型缓存固定在镜像内只读路径；运行时勿把空 volume 挂到这里。
ENV EMBEDDING_CACHE_DIR=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG HF_ENDPOINT=
ARG HF_TOKEN=
ENV HF_ENDPOINT=${HF_ENDPOINT} \
    HF_TOKEN=${HF_TOKEN}

RUN mkdir -p /app/models \
    && python scripts/download_model.py --cache-dir /app/models --retries 5 \
    && python scripts/download_model.py --cache-dir /app/models --reranker --retries 5

ENV UPLOADS_DIR=/data/uploads \
    CHROMA_DIR=/data/chroma_db \
    SETTINGS_FILE=/data/settings.json \
    HOST=0.0.0.0 \
    PORT=8000 \
    HYBRID_SEARCH_ENABLED=true \
    RERANK_ENABLED=true \
    EMBEDDING_PROVIDER=local \
    RERANK_PROVIDER=local

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
