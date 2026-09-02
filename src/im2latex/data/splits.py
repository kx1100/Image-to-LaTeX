"""Split parsing and the leakage audit that evidences N-4.

The splits themselves are the canonical published ones — they are read, not invented.
What this module adds is *verification*: N-4 requires splits that are fixed, documented,
and free of distribution leakage, and an audit that is never run is an assumption rather
than a guarantee.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Canonical split names, in the order they are reported.
SPLIT_NAMES: tuple[str, ...] = ("train", "validate", "test")

#: Approximate LaTeX token pattern, for auditing only.
#:
#: A control sequence (``\frac``, ``\alpha``), an escaped literal (``\{``), or a single
#: non-space character. This deliberately is NOT the model's tokenizer, which is built
#: from the train split in the next slice per D-007. Auditing with the tokenizer would
#: mean using the train split to judge whether the train split is contaminated.
TOKEN_PATTERN = re.compile(r"\\[a-zA-Z]+|\\.|\S")


@dataclass(frozen=True)
class Sample:
    """One image/formula pair."""

    image: str
    formula_index: int
    formula: str
    render_type: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def normalize_whitespace(formula: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim.

    This is the whitespace normalization the exact-match metric assumes (PLAN.md §2.4).
    It also removes the stray carriage returns and tabs the upstream corpus carries
    inside formula text. It is not tokenization — that is D-007's job, in the next
    slice — and it changes no LaTeX semantics, since TeX treats any whitespace run as a
    single separator.
    """
    return " ".join(formula.split())


def tokenize_for_audit(formula: str) -> list[str]:
    """Split a formula into approximate LaTeX tokens for distribution reporting."""
    return TOKEN_PATTERN.findall(formula)


def read_formulas(path: Path) -> tuple[list[str], str]:
    """Read the formula list, returning the formulas and the encoding that worked.

    Two upstream quirks are handled here, and both are silent-corruption traps.

    **Encoding.** The file is Latin-1, not UTF-8; decoding it as UTF-8 raises. UTF-8 is
    still attempted first so a future re-encoded release reads correctly, and the
    encoding actually used is returned so the manifest records it rather than leaving it
    as folklore.

    **Line splitting.** The split files address formulas *by line number*, so the line
    count has to be exact. ``str.splitlines()`` must not be used: it breaks on bare
    carriage returns, and this corpus contains 1,005 of them inside formula text. That
    read yields 104,564 lines instead of 103,559, shifting every formula after the first
    stray byte and silently pairing images with the wrong LaTeX. Splitting on ``\\n``
    alone gives 103,559, which is exactly train + validate + test.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        lines = text.split("\n")
        if lines and lines[-1] == "":  # trailing newline, not a formula
            lines.pop()
        return lines, encoding
    raise ValueError(f"Could not decode {path} as UTF-8 or Latin-1")


@dataclass
class ParseResult:
    """Samples parsed from a split file, and what was dropped getting there."""

    samples: list[Sample] = field(default_factory=list)
    malformed_lines: int = 0
    out_of_range_indices: int = 0


def parse_split_file(path: Path, formulas: Sequence[str]) -> ParseResult:
    """Parse one ``.lst`` split file.

    Each line is ``<formula_index> <image_name> <render_type>``, per the upstream
    readme, where ``image_name`` has no extension. Unusable lines are counted rather
    than raised on, so that a partially malformed release is reported as a number in the
    manifest instead of aborting the run or vanishing unnoticed.
    """
    result = ParseResult()
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2 or not fields[0].lstrip("-").isdigit():
                if line.strip():
                    result.malformed_lines += 1
                continue

            index = int(fields[0])
            if not 0 <= index < len(formulas):
                result.out_of_range_indices += 1
                continue

            name = fields[1]
            result.samples.append(
                Sample(
                    image=name if name.endswith(".png") else f"{name}.png",
                    formula_index=index,
                    formula=normalize_whitespace(formulas[index]),
                    render_type=fields[2] if len(fields) > 2 else "",
                )
            )
    return result


def write_jsonl(samples: Iterable[Sample], path: Path) -> int:
    """Write samples as JSON Lines, returning the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[Sample]:
    """Read samples back from a prepared split file."""
    with path.open("r", encoding="utf-8") as handle:
        return [Sample(**json.loads(line)) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[Sample]:
    """Stream samples from a prepared split file, for when the whole list is not needed."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Sample(**json.loads(line))


# ------------------------------------------------------------------------------- audit


@dataclass(frozen=True)
class SplitStatistics:
    """Distribution summary for one split."""

    name: str
    samples: int
    unique_formulas: int
    distinct_tokens: int
    total_tokens: int
    mean_token_length: float
    median_token_length: float
    max_token_length: int
    token_frequencies: Counter


def summarize(name: str, samples: Sequence[Sample]) -> SplitStatistics:
    """Compute the distribution summary for one split."""
    lengths: list[int] = []
    frequencies: Counter = Counter()
    for sample in samples:
        tokens = tokenize_for_audit(sample.formula)
        lengths.append(len(tokens))
        frequencies.update(tokens)

    return SplitStatistics(
        name=name,
        samples=len(samples),
        unique_formulas=len({sample.formula for sample in samples}),
        distinct_tokens=len(frequencies),
        total_tokens=sum(lengths),
        mean_token_length=statistics.fmean(lengths) if lengths else 0.0,
        median_token_length=float(statistics.median(lengths)) if lengths else 0.0,
        max_token_length=max(lengths, default=0),
        token_frequencies=frequencies,
    )


@dataclass(frozen=True)
class LeakageReport:
    """What the splits share with each other (N-4).

    ``shared_formula_indices`` must be empty. An index in two splits means the same
    formula is both trained and tested on, which would make the test numbers meaningless.

    ``shared_formula_text`` is a softer signal. The upstream corpus contains formulas
    duplicated under different indices — arXiv preprints repeat common expressions — so
    a non-zero count here is a property of the published splits rather than a defect
    introduced by this pipeline. It is reported because it caps how much an exact-match
    score can be trusted, not because it is actionable on its own.

    ``shared_images`` is the one that actually bites, and it is not implied by the other
    two. The published splits are disjoint by formula index, but the same rendered PNG
    is reachable from two different indices whose LaTeX differs only trivially (a
    leading ``%``, say). Measured on the canonical release: 9 images sit in both train
    and an evaluation split. That is 0.009% of the corpus and will not move a metric,
    but it is literal train-on-test data and is reported rather than left implicit.

    ``tokens_unseen_in_train`` is what D-007's vocabulary will not cover: symbols that
    appear only in validation or test are guaranteed out-of-vocabulary, since the
    vocabulary is derived from train alone.
    """

    shared_formula_indices: dict[str, int]
    shared_formula_text: dict[str, int]
    shared_images: dict[str, int]
    tokens_unseen_in_train: dict[str, list[str]]

    @property
    def has_index_leakage(self) -> bool:
        return any(count > 0 for count in self.shared_formula_indices.values())

    @property
    def has_image_leakage(self) -> bool:
        return any(count > 0 for count in self.shared_images.values())


def audit(splits: dict[str, list[Sample]]) -> tuple[dict[str, SplitStatistics], LeakageReport]:
    """Produce per-split statistics and the cross-split leakage report (N-4)."""
    stats = {name: summarize(name, samples) for name, samples in splits.items()}

    indices = {name: {s.formula_index for s in samples} for name, samples in splits.items()}
    texts = {name: {s.formula for s in samples} for name, samples in splits.items()}
    images = {name: {s.image for s in samples} for name, samples in splits.items()}

    shared_indices: dict[str, int] = {}
    shared_text: dict[str, int] = {}
    shared_images: dict[str, int] = {}
    names = list(splits)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            pair = f"{left}|{right}"
            shared_indices[pair] = len(indices[left] & indices[right])
            shared_text[pair] = len(texts[left] & texts[right])
            shared_images[pair] = len(images[left] & images[right])

    train_tokens = set(stats["train"].token_frequencies) if "train" in stats else set()
    unseen = {
        name: sorted(set(split_stats.token_frequencies) - train_tokens)
        for name, split_stats in stats.items()
        if name != "train"
    }

    return stats, LeakageReport(
        shared_formula_indices=shared_indices,
        shared_formula_text=shared_text,
        shared_images=shared_images,
        tokens_unseen_in_train=unseen,
    )
