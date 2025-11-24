FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# Copy uv and uvx (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /uvx /bin/

# Install system dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libclang-dev \
        git \
        curl \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
# The official install script usually detects the architecture (ARM64 vs AMD64) automatically.
RUN curl -fsSL https://ollama.com/install.sh | sh

# The vscode user already exists in the base image with UID 1000.
USER vscode

WORKDIR /workspace
