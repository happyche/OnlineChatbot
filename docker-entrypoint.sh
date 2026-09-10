#!/bin/sh
set -e

# 持久化目录（由 compose volume 挂载）；模型权重在 /app/models，不挂 volume。
mkdir -p /data/uploads /data/chroma_db

exec "$@"
