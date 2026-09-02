"""Command-line entry point.

Subcommand groups are registered by the modules that implement them, so that each
milestone slice adds its commands without this file becoming a hub of stubs.
"""

from __future__ import annotations

import typer

from im2latex import __version__
from im2latex.data.cli import app as data_app

app = typer.Typer(
    help="Image-to-LaTeX data pipeline and recognition tooling.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Root callback.

    Present so that Typer keeps `im2latex <command>` dispatch even while only one
    command is registered; without it a single-command app collapses into a bare
    command and `im2latex version` becomes an unexpected argument.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
