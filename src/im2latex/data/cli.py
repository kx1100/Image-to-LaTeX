"""``im2latex data`` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from im2latex.config import DEFAULT_CONFIG_PATH, load_config
from im2latex.data.im2latex import prepare
from im2latex.data.splits import SPLIT_NAMES, audit, read_jsonl

app = typer.Typer(help="Dataset download, preparation and split management.", no_args_is_help=True)

ConfigOption = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to the config file.")


@app.command("prepare")
def prepare_command(config_path: Path = ConfigOption) -> None:
    """Download, verify, extract and split im2latex-100k.

    Idempotent: files already present and matching their published checksum are not
    re-downloaded, and an already-extracted archive is not unpacked again.
    """
    config = load_config(config_path)
    result = prepare(config, report=typer.echo)

    typer.echo("")
    typer.echo(f"Prepared {sum(result.counts.values()):,} samples:")
    for split, count in result.counts.items():
        typer.echo(f"  {split:<9} {count:>7,}  {result.split_files[split]}")


@app.command("audit")
def audit_command(
    config_path: Path = ConfigOption,
    top: int = typer.Option(15, help="How many of the most frequent tokens to show."),
    unseen: int = typer.Option(20, help="How many train-unseen tokens to list per split."),
) -> None:
    """Report split distributions and cross-split leakage (N-4).

    Exits non-zero if any formula index appears in more than one split, so the check can
    gate a pipeline rather than merely inform a human.
    """
    config = load_config(config_path)
    processed = config.paths.processed / "im2latex-100k"

    splits = {}
    for name in SPLIT_NAMES:
        path = processed / f"{name}.jsonl"
        if not path.is_file():
            raise typer.BadParameter(f"{path} not found. Run `im2latex data prepare` first.")
        splits[name] = read_jsonl(path)

    statistics, leakage = audit(splits)

    typer.echo(f"{'split':<10}{'samples':>10}{'unique':>10}{'tokens':>9}{'mean':>8}"
               f"{'median':>8}{'max':>7}")
    for name in SPLIT_NAMES:
        stats = statistics[name]
        typer.echo(
            f"{name:<10}{stats.samples:>10,}{stats.unique_formulas:>10,}"
            f"{stats.distinct_tokens:>9,}{stats.mean_token_length:>8.1f}"
            f"{stats.median_token_length:>8.0f}{stats.max_token_length:>7,}"
        )

    typer.echo("\nMost frequent tokens in train:")
    for token, count in statistics["train"].token_frequencies.most_common(top):
        typer.echo(f"  {token:<12} {count:>9,}")

    typer.echo("\nShared formula indices between splits (must all be 0):")
    for pair, count in leakage.shared_formula_indices.items():
        marker = "  LEAK" if count else ""
        typer.echo(f"  {pair:<20} {count:>7,}{marker}")

    typer.echo("\nShared formula text between splits (upstream duplicates, informational):")
    for pair, count in leakage.shared_formula_text.items():
        typer.echo(f"  {pair:<20} {count:>7,}")

    typer.echo("\nShared IMAGES between splits (literal train-on-test, upstream):")
    for pair, count in leakage.shared_images.items():
        marker = "  <-- contaminated" if count else ""
        typer.echo(f"  {pair:<20} {count:>7,}{marker}")

    typer.echo("\nTokens absent from train (guaranteed out-of-vocabulary under D-007):")
    for name, tokens in leakage.tokens_unseen_in_train.items():
        shown = ", ".join(tokens[:unseen]) if tokens else "none"
        more = f" (+{len(tokens) - unseen} more)" if len(tokens) > unseen else ""
        typer.echo(f"  {name:<10} {len(tokens):>4}  {shown}{more}")

    if leakage.has_index_leakage:
        typer.echo("\nFAIL: a formula index appears in more than one split (N-4).")
        raise typer.Exit(code=1)

    typer.echo("\nOK: no formula index is shared between splits (N-4).")
    if leakage.has_image_leakage:
        # Not a failure: this is a property of the canonical published splits, and
        # failing here would mean the check can never pass on the dataset we chose.
        # It is surfaced so the number is known when M2's results are read.
        total = sum(leakage.shared_images.values())
        typer.echo(
            f"NOTE: {total} image(s) appear in more than one split. This is upstream, "
            "not introduced here; see DECISIONS.md D-009."
        )
