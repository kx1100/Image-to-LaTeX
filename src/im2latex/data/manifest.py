"""The dataset manifest: what was prepared, from where, and whether it verified.

N-3 requires every training run to record its dataset version, and N-5 requires
evaluation to be reproducible from a checkpoint and a split. Both need the dataset
itself to be identifiable, which a directory of images is not. The manifest is the
identifier: source URLs, published checksums, counts, and the encoding actually used.

It is the one artifact under ``data/`` that is committed, so a prepared dataset can be
verified against the repository rather than trusted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


@dataclass
class SplitRecord:
    """What one prepared split contains."""

    samples: int
    unique_formulas: int
    missing_images: int
    malformed_lines: int
    out_of_range_indices: int
    path: str


@dataclass
class Manifest:
    """A prepared dataset, described well enough to be reproduced or verified."""

    dataset: str
    doi: str
    license: str
    prepared_at: str
    tool_version: str
    manifest_version: int = MANIFEST_VERSION
    formula_encoding: str = ""
    formula_count: int = 0
    source_urls: dict[str, str] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    splits: dict[str, SplitRecord] = field(default_factory=dict)

    @classmethod
    def create(cls, dataset: str, doi: str, license_: str, tool_version: str) -> Manifest:
        return cls(
            dataset=dataset,
            doi=doi,
            license=license_,
            prepared_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tool_version=tool_version,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["splits"] = {name: asdict(record) for name, record in self.splits.items()}
        return data

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest back as plain data."""
    return json.loads(path.read_text(encoding="utf-8"))
