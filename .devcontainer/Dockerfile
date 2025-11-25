ARG PYTHON_VERSION=3.11
ARG INSTALL_OLLAMA=false

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}-bookworm

# Copy uv and uvx (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /uvx /bin/

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Enable Docker BuildKit for cache mounts
ENV DOCKER_BUILDKIT=1

# Install system dependencies with cache mount for faster rebuilds (pinned versions)
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt/lists \
  apt-get update \
  && apt-get install -y --no-install-recommends \
  cmake=3.25.1-1 \
  ninja-build=1.11.1-2~deb12u1 \
  libclang-dev=1:14.0-55.7~deb12u1 \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
# The official install script usually detects the architecture (ARM64 vs AMD64) automatically.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ "$INSTALL_OLLAMA" = "true" ]; then \
  curl -fsSL https://ollama.com/install.sh | sh; \
  else \
  echo "Skipping Ollama installation."; \
  fi

# The vscode user already exists in the base image with UID 1000.
USER vscode

WORKDIR /workspace
