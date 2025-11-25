## Requirements

Please ensure that both [Docker](https://docs.docker.com/desktop/) and [Ollama](https://ollama.com/download) are installed and running on the host machine. To configure your IDE, mount your own `.vscode/` or `.cursor/` folders by uncommenting the `mounts` section in `devcontainer.json`.

## Overview

The dev container is built using a hybrid approach: the `Dockerfile` provides low-level customization, while `devcontainer.json` is used for fine-tuning and IDE integration. Minor changes to `devcontainer.json` do not require a full rebuild of the entire image, which speeds up the development workflow.

1. The `Dockerfile` is based on Microsoft’s Debian-style [Bookworm image for Python 3.11](https://hub.docker.com/r/microsoft/devcontainers-python). It uses a non-root user named `vscode` and installs system dependencies, `uv`, and the Ollama client. Note that the Ollama instance runs on the host machine for performance reasons.

2. The `devcontainer.json` is based on Microsoft’s [dev container template](https://github.com/devcontainers/templates/tree/main/src/python) for Python 3. It installs additional development tools via `features`, sets important environment variables, and runs `uv sync`. The container does not enforce any specific IDE configuration; you are encouraged to mount your own `.vscode/` or `.cursor/` folders externally.

## Building and testing the container locally

You can build and test the container locally using the `devcontainer` CLI tool. This process works independently of any specific IDE, such as VS Code or Cursor.

```bash
# brew install devcontainer
# npm install -g @devcontainers/cli

devcontainer read-configuration --workspace-folder .  # Validates devcontainer.json configuration
devcontainer build --workspace-folder .  # Builds the dev container
devcontainer up --workspace-folder .  # Starts the dev container and runs postCreateCommand. Complete startup test.
```

## Building and testing the container in the CI pipeline

The container build and startup process are tested in the CI pipeline defined in `.github/workflows/ci.yml`. The availability of major development tools and the successful execution of `make lint`, `make typecheck`, and `make test` are verified.

## Known Issue in Cursor IDE

Occasionally, the dev container may fail to start properly in the Cursor IDE. A [suggested workaround](https://forum.cursor.com/t/dev-containers-support/1510/13) is:

1. Start the container using VS Code.
2. In Cursor, attach to the already running container.
3. Inside the container, navigate to `/workspaces/pydantic-ai`.
