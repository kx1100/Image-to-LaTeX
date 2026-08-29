"""Tests for configuration loading and the CLI entry point."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from im2latex import __version__
from im2latex.cli import app
from im2latex.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

MINIMAL_PREPROCESSING = """
preprocessing:
  stages:
    - grayscale
    - binarize
  params:
    binarize:
      method: otsu
"""

runner = CliRunner()


def write_config(directory: Path, paths_block: str, preprocessing: str = MINIMAL_PREPROCESSING):
    """Write a config file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    config_file = directory / "data.yaml"
    config_file.write_text(paths_block + preprocessing, encoding="utf-8")
    return config_file


DEFAULT_PATHS = "paths:\n  raw: data/raw\n  interim: data/interim\n  processed: data/processed\n"


# ---------------------------------------------------------------------------------- cli


def test_version_command_reports_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


# -------------------------------------------------------------------------------- paths


def test_repository_config_loads():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    assert config.paths.raw == REPO_ROOT / "data" / "raw"
    assert config.paths.interim == REPO_ROOT / "data" / "interim"
    assert config.paths.processed == REPO_ROOT / "data" / "processed"


def test_relative_paths_resolve_against_repository_root_not_cwd(tmp_path, monkeypatch):
    """Paths must not depend on where the command was invoked from."""
    config_file = write_config(tmp_path / "configs", DEFAULT_PATHS)
    monkeypatch.chdir(tmp_path / "configs")

    config = load_config(config_file)
    assert config.paths.raw == tmp_path / "data" / "raw"


def test_absolute_paths_are_left_alone(tmp_path):
    absolute = (tmp_path / "elsewhere" / "raw").as_posix()
    config_file = write_config(
        tmp_path / "configs",
        f"paths:\n  raw: {absolute}\n  interim: b\n  processed: c\n",
    )
    assert load_config(config_file).paths.raw == Path(absolute)


# ------------------------------------------------------------------------------ failure


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_non_mapping_config_raises(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_file)


def test_missing_required_key_fails_at_load_time(tmp_path):
    config_file = write_config(tmp_path / "configs", "paths:\n  raw: data/raw\n")
    with pytest.raises(KeyError):
        load_config(config_file)


# ------------------------------------------------------------------------- preprocessing


def test_preprocessing_stages_and_params_load():
    preprocessing = load_config(REPO_ROOT / "configs" / "data.yaml").preprocessing
    assert preprocessing.stages[0] == "grayscale"
    assert preprocessing.params["binarize"]["method"] == "adaptive"


def test_unknown_stage_is_rejected_with_the_known_names(tmp_path):
    config_file = write_config(
        tmp_path / "configs",
        DEFAULT_PATHS,
        "preprocessing:\n  stages:\n    - grayscale\n    - sharpen\n",
    )
    with pytest.raises(ValueError, match="Unknown preprocessing stage"):
        load_config(config_file)


def test_duplicate_stage_is_rejected(tmp_path):
    config_file = write_config(
        tmp_path / "configs",
        DEFAULT_PATHS,
        "preprocessing:\n  stages:\n    - grayscale\n    - grayscale\n",
    )
    with pytest.raises(ValueError, match="more than once"):
        load_config(config_file)


def test_empty_stage_list_is_rejected(tmp_path):
    config_file = write_config(
        tmp_path / "configs", DEFAULT_PATHS, "preprocessing:\n  stages: []\n"
    )
    with pytest.raises(ValueError, match="at least one stage"):
        load_config(config_file)


def test_params_for_a_stage_that_is_not_run_are_rejected(tmp_path):
    """Catches the silent failure of tuning a stage you removed from the sequence."""
    config_file = write_config(
        tmp_path / "configs",
        DEFAULT_PATHS,
        "preprocessing:\n  stages:\n    - grayscale\n"
        "  params:\n    deskew:\n      max_angle_deg: 5\n",
    )
    with pytest.raises(ValueError, match="not run"):
        load_config(config_file)
