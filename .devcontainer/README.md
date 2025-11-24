## Building and testing the Dev Container locally

Build and test the dev container locally using the `devcontainer` CLI tool. No need for any IDE such as VS Code or Cursor.
```bash
# brew install devcontainer
# npm install -g @devcontainers/cli

devcontainer read-configuration --workspace-folder . # Validates devcontainer.json configuration
devcontainer build --workspace-folder . # Builds the dev container

devcontainer up --workspace-folder . # Starts the dev container including postCreateCommand. Complete startup test.
```
