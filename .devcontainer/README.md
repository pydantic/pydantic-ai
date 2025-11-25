The dev container is using Debian Bookworm as the base image. It comes with a non-root user `vscode`. The customizations is using a hybrid approach: `Dockerfile` for low-level customization and `devcontainer.json` for fine-tuning and IDE integration. Regarding IDE configurations, the container is rather unopinionated. `.vscode/` or `.cursor/` folders need to be mounted externally. The Ollama CLI is installed in the container, but the Ollama instance on the host machine is used. Building and startup of the container are tested in the CI pipeline. The `make lint`, `make typecheck` and `make test` targets run successfully in the container.

## Requirements

Please ensure both [Docker](https://docs.docker.com/desktop/) and [Ollama](https://ollama.com/download) are installed and running on the host machine.

## Building and testing the Dev Container locally

Build and test the dev container locally using the `devcontainer` CLI tool. No need for any IDE such as VS Code or Cursor.
```bash
# brew install devcontainer
# npm install -g @devcontainers/cli

devcontainer read-configuration --workspace-folder . # Validates devcontainer.json configuration
devcontainer build --workspace-folder . # Builds the dev container

devcontainer up --workspace-folder . # Starts the dev container including postCreateCommand. Complete startup test.
```

## IDE Configuration

The dev container is unopinionated about IDE configs. Please uncomment the `mounts` section in `devcontainer.json` and provide your own `.vscode/` or `.cursor/` configurations.
