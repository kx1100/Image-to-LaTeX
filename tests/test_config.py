"""Tests for configuration loading and the CLI entry point."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from im2latex import __version__
from im2latex.cli import app
from im2latex.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

runner = CliRunner()


def test_version_command_reports_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_repository_config_loads():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    assert config.paths.raw == REPO_ROOT / "data" / "raw"
    assert config.paths.interim == REPO_ROOT / "data" / "interim"
    assert config.paths.processed == REPO_ROOT / "data" / "processed"


def test_relative_paths_resolve_against_repository_root_not_cwd(tmp_path, monkeypatch):
    """Paths must not depend on where the command was invoked from."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "data.yaml").write_text(
        "paths:\n  raw: data/raw\n  interim: data/interim\n  processed: data/processed\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path / "configs")
    config = load_config(config_dir / "data.yaml")
    assert config.paths.raw == tmp_path / "data" / "raw"


def test_absolute_paths_are_left_alone(tmp_path):
    absolute = (tmp_path / "elsewhere" / "raw").as_posix()
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(
        f"paths:\n  raw: {absolute}\n  interim: b\n  processed: c\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.paths.raw == Path(absolute)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_non_mapping_config_raises(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_file)


def test_missing_required_key_fails_at_load_time(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("paths:\n  raw: data/raw\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(config_file)
