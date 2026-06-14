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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p processed

# Cache dependencies; keep freeze the lock file so dependencies stay absolutely the same
# ENV instruction ensures path variable sets up venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Configure and build (both Debug and Release)
COPY . .
RUN uv sync --frozen
RUN cmake --preset linux-x64-debug
RUN cmake --build --preset linux-x64-debug
RUN cmake --preset linux-x64-release
RUN cmake --build --preset linux-x64-release

WORKDIR /app/bin

# Invoke preferred executable using ./${config}/#{executable}
# 
# ./Debug/imgproc --input 
# ./Debug/unit_tests