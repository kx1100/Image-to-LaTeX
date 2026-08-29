"""Configuration loading.

Configuration lives in YAML (configs/data.yaml) and is parsed into frozen dataclasses
so that a typo in a key fails at load time rather than deep inside a pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations for each stage of the data pipeline."""

    raw: Path
    interim: Path
    processed: Path

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path) -> PathsConfig:
        return cls(
            raw=_resolve(data["raw"], root),
            interim=_resolve(data["interim"], root),
            processed=_resolve(data["processed"], root),
        )


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    paths: PathsConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path) -> Config:
        return cls(paths=PathsConfig.from_dict(data["paths"], root))


def _resolve(value: str, root: Path) -> Path:
    """Resolve a configured path against the repository root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load configuration from a YAML file.

    Relative paths inside the file are resolved against the file's parent directory's
    parent (i.e. the repository root), so runs are independent of the caller's cwd.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {config_path} must contain a YAML mapping")

    return Config.from_dict(data, root=config_path.parent.parent)
