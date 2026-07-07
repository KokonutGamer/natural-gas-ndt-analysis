# Build the project on debian trixie
FROM python:3.13.14-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    clang \
    cmake \
    ninja-build \
    git \
    libopencv-dev \
    lldb \
    gdbserver \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Cache dependencies; freeze the lock file so dependencies stay absolutely the same
# ENV instruction ensures path variable sets up venv
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT="/opt/.venv"
RUN uv sync --frozen --no-install-project
ENV PATH="/opt/.venv/bin:$PATH"

WORKDIR /app
RUN mkdir -p processed
