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
class PreprocessConfig:
    """The preprocessing stage sequence and each stage's parameters (R-8).

    Order is configuration rather than code so that it can be changed and measured
    without touching the pipeline — see D-008 for why the current order is what it is.
    """

    stages: tuple[str, ...]
    params: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessConfig:
        stages = tuple(data["stages"])
        if not stages:
            raise ValueError("preprocessing.stages must list at least one stage")

        # Imported here rather than at module scope: pipeline imports this module, and
        # naming the stage registry at import time would close the cycle.
        from im2latex.preprocessing.pipeline import STAGES

        unknown = [stage for stage in stages if stage not in STAGES]
        if unknown:
            known = ", ".join(sorted(STAGES))
            raise ValueError(f"Unknown preprocessing stage(s): {unknown}. Known stages: {known}")

        duplicates = {stage for stage in stages if stages.count(stage) > 1}
        if duplicates:
            raise ValueError(f"Preprocessing stage(s) listed more than once: {sorted(duplicates)}")

        params = dict(data.get("params") or {})
        orphaned = [stage for stage in params if stage not in stages]
        if orphaned:
            raise ValueError(
                f"preprocessing.params has entries for stages that are not run: {orphaned}"
            )

        return cls(stages=stages, params={key: dict(value) for key, value in params.items()})


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    paths: PathsConfig
    preprocessing: PreprocessConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path) -> Config:
        return cls(
            paths=PathsConfig.from_dict(data["paths"], root),
            preprocessing=PreprocessConfig.from_dict(data["preprocessing"]),
        )


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
