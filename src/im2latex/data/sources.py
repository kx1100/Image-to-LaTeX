"""Dataset sources, and verified fetching of them.

im2latex-100k is pinned to the Zenodo deposit rather than the author's web mirror: it
has a DOI, immutable versioned records, published MD5s for every file, and a CC0
licence.

The mirror is kept as a fallback for Zenodo outages, with one measured caveat: its
copies of the three ``.lst`` split files are one byte longer than Zenodo's (a trailing
newline), so they cannot satisfy the published MD5 and the fallback will be rejected for
those files. That is the safe failure — verification is never relaxed to accommodate a
source — and it is recorded here so the behaviour is not mistaken for a bug. The two
large files, ``formula_images.tar.gz`` and ``im2latex_formulas.lst``, match Zenodo
exactly and do fall back cleanly.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

CHUNK_BYTES = 1 << 20  # 1 MiB

ZENODO_RECORD = "56198"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
MIRROR_BASE = "https://im2markup.yuntiandeng.com/data"

#: DOI of the pinned deposit, recorded in the manifest so a prepared dataset is citable.
IM2LATEX_100K_DOI = "10.5281/zenodo.56198"
IM2LATEX_100K_LICENSE = "CC0-1.0"


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable file, pinned by checksum."""

    name: str
    md5: str
    description: str

    @property
    def urls(self) -> tuple[str, ...]:
        """Primary URL first, then fallbacks. All are expected to be byte-identical."""
        return (
            f"{ZENODO_BASE}/{self.name}?download=1",
            f"{MIRROR_BASE}/{self.name}",
        )


#: The canonical im2latex-100k distribution (Deng et al., 2017).
#:
#: The split files are the published ones, so that M2's numbers are comparable with the
#: literature rather than to a re-split only this project uses (N-4).
IM2LATEX_100K: tuple[RemoteFile, ...] = (
    RemoteFile(
        name="formula_images.tar.gz",
        md5="cf25f2408f1ea09bbd096890a6361533",
        description="Rendered formula images (~292 MB)",
    ),
    RemoteFile(
        name="im2latex_formulas.lst",
        md5="974c0a14f0daa6d91ecd0e625f1ddf52",
        description="Untokenized formulas, one per line, indexed by the split files",
    ),
    RemoteFile(
        name="im2latex_train.lst",
        md5="d5607c37aa00576098a9e4bad84a7040",
        description="Canonical training split",
    ),
    RemoteFile(
        name="im2latex_validate.lst",
        md5="cf6eeee02bc443b1b9557685fbfe7ea5",
        description="Canonical validation split",
    ),
    RemoteFile(
        name="im2latex_test.lst",
        md5="1bc17b865796dca5df15250b4da7804f",
        description="Canonical test split",
    ),
    RemoteFile(
        name="readme.txt",
        md5="3d4cb64d8c403148ff06370d71072cdc",
        description="Upstream README, kept so the raw directory is self-describing",
    ),
)


class ChecksumMismatch(RuntimeError):
    """A downloaded file did not match its published checksum."""


def md5_of(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Compute a file's MD5 without reading it entirely into memory."""
    digest = hashlib.md5()  # noqa: S324 - integrity check against a published digest,
    with path.open("rb") as handle:  # not a security primitive
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def is_present_and_valid(path: Path, expected_md5: str) -> bool:
    """True when the file already exists and matches its checksum."""
    return path.is_file() and md5_of(path) == expected_md5


def _download_to(url: str, destination: Path, on_progress=None) -> None:
    """Stream one URL to a temporary file, then move it into place.

    Downloading via a ``.part`` file means an interrupted run cannot leave a truncated
    file that looks complete on the next invocation.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "im2latex/0.1"})

    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned https URLs
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        with partial.open("wb") as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total)

    partial.replace(destination)


def fetch(remote: RemoteFile, directory: Path, on_progress=None, on_event=None) -> Path:
    """Download one file into ``directory`` and verify it, returning its path.

    A file already present with the right checksum is left alone, which is what makes
    ``data prepare`` cheap to re-run. Each URL is tried in turn; a checksum failure is
    treated like a download failure so a corrupted mirror falls through to the next
    source rather than poisoning the dataset.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / remote.name

    if is_present_and_valid(destination, remote.md5):
        if on_event:
            on_event(f"{remote.name}: already present and verified")
        return destination

    failures: list[str] = []
    for url in remote.urls:
        try:
            if on_event:
                on_event(f"{remote.name}: fetching from {url}")
            _download_to(url, destination, on_progress=on_progress)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            failures.append(f"{url}: {error}")
            continue

        actual = md5_of(destination)
        if actual != remote.md5:
            destination.unlink(missing_ok=True)
            failures.append(f"{url}: checksum {actual}, expected {remote.md5}")
            continue

        if on_event:
            on_event(f"{remote.name}: verified {remote.md5}")
        return destination

    raise ChecksumMismatch(
        f"Could not obtain a verified copy of {remote.name}. Tried:\n  "
        + "\n  ".join(failures)
    )


def fetch_all(
    remotes: Iterable[RemoteFile], directory: Path, on_progress=None, on_event=None
) -> Iterator[Path]:
    """Fetch several files, yielding each verified path as it completes."""
    for remote in remotes:
        yield fetch(remote, directory, on_progress=on_progress, on_event=on_event)


def _is_within(directory: Path, target: Path) -> bool:
    """True when ``target`` resolves to somewhere inside ``directory``."""
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return True


def extract_archive(archive: Path, destination: Path, sentinel: str | None = None) -> Path:
    """Extract a tar archive, skipping the work when it is already extracted.

    ``sentinel`` names a path inside ``destination`` whose presence means the archive
    has already been unpacked.

    Members are screened rather than trusted. ``tarfile``'s ``filter="data"`` would do
    this, but it only reached ``shutil.unpack_archive`` in Python 3.12 and this project
    supports 3.10, so the check is explicit: nothing may escape the destination, and
    links and device nodes — which a formula-image archive has no reason to contain —
    are refused outright.
    """
    if sentinel is not None and (destination / sentinel).exists():
        return destination / sentinel

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"{archive.name} contains a non-regular member: {member.name}")
            if not _is_within(root, (destination / member.name).resolve()):
                raise ValueError(f"{archive.name} contains an escaping path: {member.name}")
            members.append(member)
        tar.extractall(destination, members=members)  # noqa: S202 - members screened above

    return destination / sentinel if sentinel else destination
