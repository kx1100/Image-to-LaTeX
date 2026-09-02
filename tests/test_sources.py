"""Tests for verified fetching, extraction, and the manifest.

No test here touches the network: ``_download_to`` is substituted so that the retry,
verification and fallback logic is exercised deterministically and offline.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.error
from pathlib import Path

import pytest

from im2latex.data import sources
from im2latex.data.manifest import Manifest, SplitRecord, read_manifest
from im2latex.data.sources import ChecksumMismatch, RemoteFile

PAYLOAD = b"formula list contents"
PAYLOAD_MD5 = hashlib.md5(PAYLOAD).hexdigest()  # noqa: S324 - matching production usage


@pytest.fixture
def remote() -> RemoteFile:
    return RemoteFile(name="thing.lst", md5=PAYLOAD_MD5, description="test file")


def fake_downloader(content_by_url: dict[str, bytes | Exception]):
    """Build a stand-in for ``_download_to`` driven by a per-URL script."""

    def download(url: str, destination: Path, on_progress=None) -> None:
        outcome = content_by_url.get(url)
        if outcome is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if isinstance(outcome, Exception):
            raise outcome
        destination.write_bytes(outcome)

    return download


# ------------------------------------------------------------------------------ md5


def test_md5_of_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    assert sources.md5_of(path) == PAYLOAD_MD5


def test_md5_of_reads_in_chunks_without_changing_the_result(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD * 1000)
    assert sources.md5_of(path, chunk_bytes=7) == sources.md5_of(path, chunk_bytes=1 << 20)


def test_is_present_and_valid_is_false_for_a_missing_file(tmp_path):
    assert not sources.is_present_and_valid(tmp_path / "nope", PAYLOAD_MD5)


def test_is_present_and_valid_is_false_for_wrong_contents(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"different")
    assert not sources.is_present_and_valid(path, PAYLOAD_MD5)


# ---------------------------------------------------------------------------- fetch


def test_fetch_downloads_and_verifies(tmp_path, monkeypatch, remote):
    monkeypatch.setattr(
        sources, "_download_to", fake_downloader({remote.urls[0]: PAYLOAD})
    )
    path = sources.fetch(remote, tmp_path)
    assert path.read_bytes() == PAYLOAD


def test_fetch_skips_a_file_already_present_and_valid(tmp_path, monkeypatch, remote):
    (tmp_path / remote.name).write_bytes(PAYLOAD)

    def explode(*args, **kwargs):
        raise AssertionError("should not download an already-valid file")

    monkeypatch.setattr(sources, "_download_to", explode)
    assert sources.fetch(remote, tmp_path).read_bytes() == PAYLOAD


def test_fetch_redownloads_a_corrupted_local_file(tmp_path, monkeypatch, remote):
    (tmp_path / remote.name).write_bytes(b"corrupted")
    monkeypatch.setattr(
        sources, "_download_to", fake_downloader({remote.urls[0]: PAYLOAD})
    )
    assert sources.fetch(remote, tmp_path).read_bytes() == PAYLOAD


def test_fetch_falls_back_to_the_mirror_when_the_primary_fails(tmp_path, monkeypatch, remote):
    monkeypatch.setattr(
        sources,
        "_download_to",
        fake_downloader({
            remote.urls[0]: urllib.error.URLError("zenodo down"),
            remote.urls[1]: PAYLOAD,
        }),
    )
    assert sources.fetch(remote, tmp_path).read_bytes() == PAYLOAD


def test_fetch_treats_a_bad_checksum_as_a_failed_source(tmp_path, monkeypatch, remote):
    """A mirror serving different bytes must fall through, not poison the dataset."""
    monkeypatch.setattr(
        sources,
        "_download_to",
        fake_downloader({remote.urls[0]: b"wrong bytes", remote.urls[1]: PAYLOAD}),
    )
    assert sources.fetch(remote, tmp_path).read_bytes() == PAYLOAD


def test_fetch_raises_when_no_source_verifies(tmp_path, monkeypatch, remote):
    monkeypatch.setattr(
        sources,
        "_download_to",
        fake_downloader({remote.urls[0]: b"bad", remote.urls[1]: b"also bad"}),
    )
    with pytest.raises(ChecksumMismatch) as error:
        sources.fetch(remote, tmp_path)

    message = str(error.value)
    assert remote.urls[0] in message and remote.urls[1] in message


def test_fetch_leaves_no_file_behind_when_every_source_fails(tmp_path, monkeypatch, remote):
    """A failed prepare must not leave a corrupt file that later looks cached."""
    monkeypatch.setattr(
        sources, "_download_to", fake_downloader({remote.urls[0]: b"bad", remote.urls[1]: b"bad"})
    )
    with pytest.raises(ChecksumMismatch):
        sources.fetch(remote, tmp_path)
    assert not (tmp_path / remote.name).exists()


def test_remote_urls_put_zenodo_first(remote):
    assert "zenodo.org" in remote.urls[0]
    assert len(remote.urls) > 1


def test_the_pinned_dataset_lists_every_file_the_pipeline_needs():
    names = {r.name for r in sources.IM2LATEX_100K}
    assert names >= {
        "formula_images.tar.gz",
        "im2latex_formulas.lst",
        "im2latex_train.lst",
        "im2latex_validate.lst",
        "im2latex_test.lst",
    }
    assert all(len(r.md5) == 32 for r in sources.IM2LATEX_100K)


# -------------------------------------------------------------------------- extraction


def build_tar(path: Path, entries: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


def test_extract_archive_unpacks(tmp_path):
    archive = build_tar(tmp_path / "a.tar.gz", {"images/one.png": b"x"})
    sources.extract_archive(archive, tmp_path / "out", sentinel="images")
    assert (tmp_path / "out" / "images" / "one.png").read_bytes() == b"x"


def test_extract_archive_skips_when_the_sentinel_exists(tmp_path):
    archive = build_tar(tmp_path / "a.tar.gz", {"images/one.png": b"x"})
    destination = tmp_path / "out"
    (destination / "images").mkdir(parents=True)

    sources.extract_archive(archive, destination, sentinel="images")
    assert not (destination / "images" / "one.png").exists()


def test_extract_archive_refuses_a_path_escaping_the_destination(tmp_path):
    archive = build_tar(tmp_path / "evil.tar.gz", {"../escaped.txt": b"x"})
    with pytest.raises(ValueError, match="escaping path"):
        sources.extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_archive_refuses_a_symlink_member(tmp_path):
    archive_path = tmp_path / "link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError, match="non-regular member"):
        sources.extract_archive(archive_path, tmp_path / "out")


# ---------------------------------------------------------------------------- manifest


def test_manifest_round_trips(tmp_path):
    manifest = Manifest.create("im2latex-100k", "10.5281/zenodo.56198", "CC0-1.0", "0.1.0")
    manifest.formula_encoding = "latin-1"
    manifest.formula_count = 103_559
    manifest.checksums = {"a.lst": "0" * 32}
    manifest.splits["train"] = SplitRecord(
        samples=83_884,
        unique_formulas=83_872,
        missing_images=0,
        malformed_lines=0,
        out_of_range_indices=0,
        path="data/processed/train.jsonl",
    )

    path = manifest.write(tmp_path / "manifest.json")
    loaded = read_manifest(path)

    assert loaded["dataset"] == "im2latex-100k"
    assert loaded["formula_encoding"] == "latin-1"
    assert loaded["splits"]["train"]["samples"] == 83_884
    assert loaded["manifest_version"] == 1


def test_manifest_is_written_deterministically(tmp_path):
    """Sorted keys keep the committed manifest's diff meaningful across re-runs."""
    manifest = Manifest.create("d", "doi", "CC0-1.0", "0.1.0")
    text = manifest.write(tmp_path / "m.json").read_text(encoding="utf-8")
    assert list(json.loads(text)) == sorted(json.loads(text))
