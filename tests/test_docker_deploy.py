# -*- coding: utf-8 -*-
"""Docker 部署产物存在性检查（不依赖 Docker daemon）。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-entrypoint.sh",
    ".dockerignore",
    "env.docker.example",
    "DOCKER_DEPLOY_CHANGELOG.md",
    "scripts/docker_verify.py",
]


def test_docker_deploy_files_exist():
    for name in REQUIRED:
        assert (ROOT / name).is_file(), f"missing {name}"


def test_dockerfile_bakes_models():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "download_model.py" in text
    assert "--reranker" in text
    assert "/app/models" in text


def test_compose_does_not_mount_model_cache():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    volumes_block = text.split("volumes:", 1)[-1]
    assert "/app/models" not in volumes_block
