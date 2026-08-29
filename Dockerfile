# Reproducible environment for the data pipeline and (later) training. Satisfies N-2.
# The image installs from uv.lock only, so it resolves to the same versions as the
# Windows development environment.
FROM python:3.10-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so a source-only change does not re-resolve the environment.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --no-dev

ENTRYPOINT ["im2latex"]
CMD ["--help"]
