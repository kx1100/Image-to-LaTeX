"""Preparation of im2latex-100k: download, extract, pair, split, record.

One command takes the project from nothing to prepared, split data with a manifest
describing it (the M1 gate). Every stage is idempotent, so re-running costs a checksum
pass rather than a re-download, and an interrupted run resumes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from im2latex import __version__
from im2latex.config import Config
from im2latex.data import sources
from im2latex.data.manifest import Manifest, SplitRecord
from im2latex.data.splits import (
    SPLIT_NAMES,
    Sample,
    parse_split_file,
    read_formulas,
    write_jsonl,
)

DATASET_NAME = "im2latex-100k"

#: Directory the image archive unpacks into, per the upstream tarball layout.
IMAGE_DIRECTORY = "formula_images"

Reporter = Callable[[str], None]


@dataclass
class PreparedDataset:
    """Where a prepared dataset landed, and what it contains."""

    image_directory: Path
    split_files: dict[str, Path]
    manifest_path: Path
    counts: dict[str, int]


def _noop(_: str) -> None:
    """Default reporter: say nothing."""


def prepare(config: Config, report: Reporter = _noop) -> PreparedDataset:
    """Download, verify, extract and split im2latex-100k.

    Returns the locations of everything produced. Raises
    :class:`im2latex.data.sources.ChecksumMismatch` if no source can supply a file that
    matches its published checksum — a dataset that cannot be verified is not prepared.
    """
    raw_directory = config.paths.raw / DATASET_NAME
    interim_directory = config.paths.interim / DATASET_NAME
    processed_directory = config.paths.processed / DATASET_NAME

    report(f"Fetching {DATASET_NAME} into {raw_directory}")
    for remote in sources.IM2LATEX_100K:
        sources.fetch(remote, raw_directory, on_event=report)

    report("Extracting images")
    sources.extract_archive(
        raw_directory / "formula_images.tar.gz",
        interim_directory,
        sentinel=IMAGE_DIRECTORY,
    )
    image_directory = interim_directory / IMAGE_DIRECTORY

    # One directory listing beats ~103k individual stat calls, which is the difference
    # between a second and a minute on Windows.
    available_images = {path.name for path in image_directory.iterdir()}
    report(f"Found {len(available_images):,} images")

    formulas, encoding = read_formulas(raw_directory / "im2latex_formulas.lst")
    report(f"Read {len(formulas):,} formulas ({encoding})")

    manifest = Manifest.create(
        dataset=DATASET_NAME,
        doi=sources.IM2LATEX_100K_DOI,
        license_=sources.IM2LATEX_100K_LICENSE,
        tool_version=__version__,
    )
    manifest.formula_encoding = encoding
    manifest.formula_count = len(formulas)
    manifest.source_urls = {remote.name: remote.urls[0] for remote in sources.IM2LATEX_100K}
    manifest.checksums = {remote.name: remote.md5 for remote in sources.IM2LATEX_100K}

    split_files: dict[str, Path] = {}
    counts: dict[str, int] = {}

    for split in SPLIT_NAMES:
        parsed = parse_split_file(raw_directory / f"im2latex_{split}.lst", formulas)

        kept: list[Sample] = []
        missing = 0
        for sample in parsed.samples:
            if sample.image in available_images:
                kept.append(sample)
            else:
                missing += 1

        destination = processed_directory / f"{split}.jsonl"
        written = write_jsonl(kept, destination)

        split_files[split] = destination
        counts[split] = written
        manifest.splits[split] = SplitRecord(
            samples=written,
            unique_formulas=len({sample.formula for sample in kept}),
            missing_images=missing,
            malformed_lines=parsed.malformed_lines,
            out_of_range_indices=parsed.out_of_range_indices,
            path=config.relative(destination),
        )

        note = f" ({missing:,} images missing)" if missing else ""
        report(f"  {split:<9} {written:>7,} samples{note}")

    manifest_path = manifest.write(config.paths.manifest)
    report(f"Wrote manifest to {manifest_path}")

    return PreparedDataset(
        image_directory=image_directory,
        split_files=split_files,
        manifest_path=manifest_path,
        counts=counts,
    )
