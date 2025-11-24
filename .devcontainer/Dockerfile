FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

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

# The vscode user already exists in the base image with UID 1000. No need to create a new user.
USER vscode

WORKDIR /workspace
