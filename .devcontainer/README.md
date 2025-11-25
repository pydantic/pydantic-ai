## Requirements

Please ensure both [Docker](https://docs.docker.com/desktop/) and [Ollama](https://ollama.com/download) are installed and running on the host machine. Please configure your IDE by mounting your own `.vscode/` or `.cursor/` folders. Simply uncomment the `mounts` section in `devcontainer.json`.

## Overview

In the construction of the dev container, we take a hybrid approach: `Dockerfile` for low-level customization and `devcontainer.json` for fine-tuning and IDE integration. A minor change in `devcontainer.json` does not require a full rebuild of the entire image, thus speeding up the development workflow.

1. `Dockerfile` is based on a Debian-style [Bookworm image for Python 3.11](https://hub.docker.com/r/microsoft/devcontainers-python) by Microsoft. It ships with a non-root user `vscode`. The Dockerfile installs system dependencies, `uv` and the Ollama client. Note that the Ollama instance runs on the host machine for performance reasons.

2. `devcontainer.json` is based on a [dev container template](https://github.com/devcontainers/templates/tree/main/src/python) for Python 3 by Microsoft. It install further dev tools via `features`, sets important environment variables and runs `uv sync`. The container is rather unopinionated about IDE configurations. The user is encouraged to mount their own `.vscode/` or `.cursor/` folders externally.

## Building and testing the container locally

You can build and test the container locally using the devcontainer CLI tool. This process works independently of any specific IDE, such as VS Code or Cursor.

```bash
# brew install devcontainer
# npm install -g @devcontainers/cli

devcontainer read-configuration --workspace-folder . # Validates devcontainer.json configuration
devcontainer build --workspace-folder . # Builds the dev container

devcontainer up --workspace-folder . # Starts the dev container including postCreateCommand. Complete startup test.
```

## Building and testing the container in the CI pipeline

The container build and startup process is tested in the CI pipeline defined in `.github/workflows/ci.yml`. The availability of the major dev tools and the successful execution of `make lint`, `make typecheck` and `make test` are verified.
