FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get -y install --no-install-recommends \
  cmake \
  libclang-dev \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

