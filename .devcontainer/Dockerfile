# Pydantic AI Development Container
# Optimized for AI coding agents (Claude Code, Cursor)

FROM mcr.microsoft.com/devcontainers/base:debian-12

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    make \
    build-essential \
    ca-certificates \
    wget \
    unzip \
    # For Python builds
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    tk-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch to vscode user for non-root operations
USER vscode

# Install uv and deno
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    curl -fsSL https://deno.land/install.sh | sh

# Set all PATH and environment variables together
ENV PATH="/home/vscode/.local/bin:/home/vscode/.deno/bin:${PATH}"
ENV DENO_INSTALL="/home/vscode/.deno"
ENV UV_PYTHON=3.12

# Install Python tools (now that PATH is set)
RUN uv tool install pre-commit && \
    uv python install 3.12

# Pre-create cache directories with correct permissions
RUN mkdir -p /home/vscode/.cache/uv /home/vscode/.cache/pre-commit && \
    chown -R vscode:vscode /home/vscode/.cache

# Set working directory
WORKDIR /workspace

# Pre-create .venv directory with correct permissions
RUN mkdir -p /workspace/.venv && chown vscode:vscode /workspace/.venv

# Set environment variables for optimal development
ENV COLUMNS=150
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv

# Switch back to noninteractive
ENV DEBIAN_FRONTEND=dialog

# Default command
CMD ["/bin/bash"]
