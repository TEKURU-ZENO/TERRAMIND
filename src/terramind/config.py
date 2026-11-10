"""Paths and config loading."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def repo_root() -> Path:
    env = os.environ.get("TERRAMIND_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def data_dir() -> Path:
    return Path(os.environ.get("TERRAMIND_DATA", repo_root() / "data"))


def load_yaml(name: str) -> dict:
    with open(repo_root() / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)
