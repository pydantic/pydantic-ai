ARG PYTHON_VERSION=3.12
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}-trixie

# Copy uv and uvx (latest version)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Enable Docker BuildKit for cache mounts
ENV DOCKER_BUILDKIT=1

# Install system dependencies with cache mount for faster rebuilds (latest versions)
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt/lists \
  apt-get update \
  && apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  libclang-dev \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# The vscode user already exists in the base image with UID 1000.
USER vscode

WORKDIR /workspaces/pydantic-ai
