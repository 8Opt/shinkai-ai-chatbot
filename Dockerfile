# Base stage with common dependencies
FROM python:3.10-slim-bullseye AS base

# No system dependencies needed anymore
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build stage
FROM base AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

# Configure uv
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy only necessary files
COPY app/ /app/app/
COPY entrypoint.sh /app/entrypoint.sh
COPY main.py /app/main.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Runtime stage
FROM base AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Add Python optimizations
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2

# Copy only the necessary files from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/entrypoint.sh /app/entrypoint.sh
COPY --from=builder /app/main.py /app/main.py

# Ensure the entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Define entrypoint and default command
ENTRYPOINT ["bash"]
CMD ["entrypoint.sh"]