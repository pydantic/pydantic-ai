# Pydantic AI DevContainer

This directory contains a complete development container configuration optimized for AI coding agents like **Claude Code** and **Cursor**.

## Why Use a DevContainer?

Running AI coding agents inside containers is [Anthropic's official best practice](https://docs.anthropic.com/en/docs/claude-code) because it provides:

- **Security isolation**: Protects your host system from accidental changes
- **Environment fidelity**: Claude experiences the exact same Python interpreter, dependencies, and tools as production
- **Reproducibility**: Every developer and AI agent works in an identical environment
- **Clean separation**: Keeps project dependencies isolated from your system

## What's Included

This devcontainer setup includes:

- **Python 3.12** (the project's default version)
- **uv** - Fast Python package manager
- **pre-commit** - Git hooks for code quality
- **deno** - JavaScript runtime for documentation tools
- **All development dependencies** pre-installed via `make install`
- **All model provider SDKs** (OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Bedrock)
- **VS Code extensions** for Python, Ruff, MyPy, Pyright, GitLens, and more
- **MCP proxy support** (optional) for Model Context Protocol integration

## Platform Compatibility

### ARM64 / Apple Silicon

The devcontainer is configured to use the **`linux/amd64` (x86_64) platform** by default for maximum compatibility with Full mode (which includes ML frameworks like `mlx`, PyTorch, transformers).

**On Apple Silicon Macs**: Docker automatically uses Rosetta/QEMU emulation to run x86_64 containers. This works seamlessly but may be slightly slower than native ARM64.

**For better performance on Apple Silicon (Standard mode users):**
1. Edit `docker-compose.yml`
2. Change `platform: linux/amd64` to `platform: linux/arm64`
3. Use Standard mode installation (default)
4. ⚠️ **Note**: If you later need Full mode, you must switch back to `linux/amd64` due to ML framework dependencies

**Full mode requires amd64** because packages like `mlx` (Apple's ML framework) don't have Linux ARM64 wheels. Standard mode has no such restrictions.

### Why x86_64 by Default?

Using x86_64 ensures compatibility with Full mode (ML frameworks) out of the box. On Apple Silicon, this uses emulation which is slightly slower but works for both Standard and Full modes.

## Installation Modes

The devcontainer supports two installation modes to balance speed and functionality:

### Standard Mode (Default, Recommended)
- **Includes**: Cloud API providers + all development tools
- **Excludes**: Heavy ML frameworks (PyTorch, transformers, vLLM, outlines extras)
- **Use case**: PR testing, bug fixes, feature development (95% of users)
- **What's installed**:
  - All cloud model providers (OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Bedrock, HuggingFace)
  - Development tools (CLI, MCP, FastMCP)
  - Optional integrations (Logfire, retries, Temporal, UI, AG-UI, evals)
  - Testing tools, linters, type checkers

### Full Mode (ML Development)
- **Includes**: Everything including heavy ML frameworks for local model inference
- **Use case**: Working on outlines integration, local model features, ML framework development
- **What's installed**: Everything in Standard + PyTorch, transformers, vLLM, SGLang, MLX, LlamaCPP, and all workspace packages

### How Installation Works

**Interactive Mode** (VS Code, local development):
- When you first open the devcontainer, you'll be prompted to choose Standard or Full
- Your choice applies to the initial setup only
- You can always install ML frameworks later by running `make install`

**Non-Interactive Mode** (AI agents, CI, remote servers):
- Automatically uses **Standard mode** (excludes heavy ML frameworks)
- Override by setting environment variable: `INSTALL_MODE=full`

**Switching Modes Later**:
```bash
# Install everything including ML frameworks
make install

# Or manually install all extras for pydantic-ai-slim only
uv sync --frozen --package pydantic-ai-slim --all-extras --group lint --group docs

# Or install everything across all packages
uv sync --frozen --all-extras --all-packages --group lint --group docs
```

## Getting Started

### Using VS Code

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this repository in VS Code
4. Click the prompt to "Reopen in Container" (or press `F1` → "Dev Containers: Reopen in Container")
5. Wait for the container to build
6. **Choose installation mode** when prompted (Standard recommended for most users)
7. Once built, the environment is ready!

### Using Claude Code

Claude Code automatically detects and uses devcontainers when configured. Simply:

1. Ensure Docker is running
2. Open the project - Claude Code will prompt to use the devcontainer
3. Accept the prompt, and Claude will operate inside the container
4. **Installation mode**: Automatically uses **Standard mode** (excludes ML frameworks, agents are non-interactive)
   - To use Full mode, set `INSTALL_MODE=full` in the container's environment variables

This ensures Claude has access to the same tools and environment as your tests.

### Using Cursor

Similar to Claude Code:

1. Ensure Docker is running
2. Open the project in Cursor
3. When prompted, select "Reopen in Container"
4. **Installation mode**: Automatically uses **Standard mode** (excludes ML frameworks, agents are non-interactive)
   - To use Full mode, set `INSTALL_MODE=full` in the container's environment variables
5. Cursor's AI will now operate within the containerized environment

## Setting Up API Keys

Most model providers require API keys. The devcontainer includes a comprehensive `.env.example` file documenting all supported providers.

### Quick Setup

1. **Copy the example file**:
   ```bash
   cp .devcontainer/.env.example .devcontainer/.env
   ```

2. **Add your API keys** to `.devcontainer/.env`:
   ```bash
   # Required for testing with specific models
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=AI...
   # Add others as needed
   ```

3. **Update docker-compose** to load the `.env` file (add to the `dev` service):
   ```yaml
   dev:
     env_file:
       - .env
   ```

4. **Rebuild the container**: `Dev Containers: Rebuild Container`

### Supported Providers

See `.devcontainer/.env.example` for the complete list of API keys for:
- **Major providers**: OpenAI, Anthropic (Claude), Google (Gemini), Groq, Mistral, Cohere
- **AWS Bedrock**: Configure via AWS credentials
- **OpenAI-compatible**: DeepSeek, Grok, OpenRouter, Fireworks, Together, and more
- **Search tools**: Brave Search, Tavily (optional)
- **Observability**: Logfire (optional)

### Testing Without API Keys

You can develop and test without paid API keys using:
- **Ollama** (see "Using Ollama Locally" below) - Free local models
- **Test models** - The test suite includes mock models
- **VCR cassettes** - Pre-recorded API interactions in `tests/cassettes/`

## Available Commands

Once inside the container, all standard project commands work:

```bash
# Run all checks (format, lint, typecheck, tests)
make

# Run tests only
make test

# Run tests with coverage report
make testcov

# Format code
make format

# Lint code
make lint

# Type check with Pyright
make typecheck

# Build and serve documentation locally
make docs-serve  # Available at http://localhost:8000
```

## Container Architecture

### File Structure

```
.devcontainer/
├── Dockerfile              # Container image definition
├── devcontainer.json       # VS Code configuration
├── docker-compose.yml      # Service orchestration
├── mcp-proxy-config.json   # MCP server proxy config (optional)
└── README.md               # This file
```

### Volumes

The setup uses several volumes for optimal performance:

- **Workspace**: Your code is mounted at `/workspace`
- **Virtual environment**: `.venv/` persists across container restarts
- **Cache directories**: `uv` and `pre-commit` caches are preserved for faster operations

### Networking

The container uses `host` networking mode for simplicity. This means:

- The docs server at `localhost:8000` is directly accessible
- No port mapping configuration needed
- Direct access to any services you run

## MCP Integration (Advanced)

The devcontainer includes optional MCP (Model Context Protocol) proxy support. This allows Claude Code to communicate with stdio-based MCP servers from inside the container.

### Enabling MCP Proxy

1. Edit `docker-compose.yml` and uncomment the `mcp-proxy` service
2. Edit `mcp-proxy-config.json` to configure your MCP servers:
   - Set `disabled: false` for servers you want to enable
   - Add environment variables for API keys
3. Rebuild the container: `Dev Containers: Rebuild Container`
4. The proxy will be available at `http://localhost:3000`

### Configuring MCP Servers

Example configuration in `mcp-proxy-config.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      },
      "disabled": false
    }
  }
}
```

**Security Note**: Never commit API keys. Use environment variables or Docker secrets instead.

## Using Ollama Locally

Ollama allows you to run open-source LLMs locally without API keys or usage costs. The devcontainer includes an optional Ollama service.

### Enabling Ollama

1. **Edit `docker-compose.yml`** and uncomment the `ollama` service section
2. **Uncomment the volume** in the `volumes` section: `ollama-models:`
3. **Rebuild the container**: `Dev Containers: Rebuild Container`

### Pulling Models

Once Ollama is running, pull models:

```bash
# Pull a small, fast model (used in CI tests)
docker exec pydantic-ai-ollama ollama pull qwen2:0.5b

# Pull other popular models
docker exec pydantic-ai-ollama ollama pull llama3.2:3b
docker exec pydantic-ai-ollama ollama pull phi3:mini
```

### Using Ollama in Code

Ollama uses the OpenAI-compatible API:

```python
from pydantic_ai import Agent

# Using the docker-compose ollama service
agent = Agent(
    'openai:qwen2:0.5b',
    openai_base_url='http://localhost:11434/v1/'
)
```

Or set the environment variable:
```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1/
```

### Checking Available Models

```bash
docker exec pydantic-ai-ollama ollama list
```

## Running Examples with Databases

Some examples in `examples/` require PostgreSQL. The devcontainer includes optional database services.

### SQL Generation Example (`examples/sql_gen.py`)

1. **Edit `docker-compose.yml`** and uncomment the `postgres` service section
2. **Uncomment the volume**: `postgres-data:`
3. **Rebuild the container**: `Dev Containers: Rebuild Container`
4. **Run the example**:
   ```bash
   # The example expects PostgreSQL at localhost:54320
   cd examples
   uv run pydantic_ai_examples/sql_gen.py
   ```

### RAG Example (`examples/rag.py`)

This example requires PostgreSQL with the pgvector extension:

1. **Edit `docker-compose.yml`** and uncomment the `pgvector` service section
2. **Uncomment the volume**: `pgvector-data:`
3. **Rebuild the container**: `Dev Containers: Rebuild Container`
4. **Run the example**:
   ```bash
   # The example expects pgvector at localhost:54321
   cd examples
   uv run pydantic_ai_examples/rag.py
   ```

### Connection Strings

The default credentials for both services are:
```
User: postgres
Password: postgres
Database: postgres
```

- **Standard PostgreSQL**: `postgresql://postgres:postgres@localhost:54320/postgres`
- **PostgreSQL + pgvector**: `postgresql://postgres:postgres@localhost:54321/postgres`

## Optional Services Summary

The devcontainer supports these optional services (all commented out by default):

| Service | Port | Use Case | Enable By |
|---------|------|----------|-----------|
| **Ollama** | 11434 | Local LLM testing | Uncomment in `docker-compose.yml` |
| **PostgreSQL** | 54320 | SQL generation example | Uncomment in `docker-compose.yml` |
| **pgvector** | 54321 | RAG example | Uncomment in `docker-compose.yml` |
| **MCP Proxy** | 3000 | MCP server integration | Uncomment in `docker-compose.yml` |

**Why commented out?** To keep the default setup minimal and fast. Enable only what you need.

## Git Configuration

### How Git Works in the DevContainer

This devcontainer follows **2025 best practices** for git configuration with AI coding agents:

**Automatic Credential Forwarding** (No manual setup needed)
- VS Code automatically forwards your SSH agent and git credentials to the container
- No need to mount `.gitconfig` or `.ssh` directories
- Works seamlessly with SSH keys, Personal Access Tokens, and credential helpers
- **Source**: [VS Code: Sharing Git credentials](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials)

**Git Identity via Environment Variables**
- The container uses environment variables for git commits made by AI agents
- Default identity: `AI Agent <ai@devcontainer.local>`
- Customize in `.devcontainer/devcontainer.json` under `remoteEnv`:
  ```json
  "GIT_AUTHOR_NAME": "Your Preferred Name",
  "GIT_AUTHOR_EMAIL": "your@email.com",
  "GIT_COMMITTER_NAME": "Your Preferred Name",
  "GIT_COMMITTER_EMAIL": "your@email.com"
  ```
- **Source**: [Git Environment Variables](https://git-scm.com/book/en/v2/Git-Internals-Environment-Variables)

**Safe Directory Configuration**
- The `postStartCommand` automatically trusts the workspace directory
- This resolves git's "dubious ownership" security check (CVE-2022-24765)
- **Source**: [Avoiding Dubious Ownership in Dev Containers](https://www.kenmuse.com/blog/avoiding-dubious-ownership-in-dev-containers/)

**Why This Approach?**
- **More secure**: Host's gitconfig remains untouched by the container
- **Cleaner**: No file mounts needed - VS Code handles everything
- **Flexible**: Easy to customize AI commit identity via environment variables
- **Modern**: Follows 2025 devcontainer best practices

## Customization

### Adding Python Packages

Install packages using `uv`:

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Sync with lockfile
uv sync
```

### Adding VS Code Extensions

Edit `devcontainer.json` and add extension IDs to the `extensions` array:

```json
"extensions": [
  "ms-python.python",
  "your-extension-id"
]
```

### Changing Python Version

To use a different Python version:

1. Edit `Dockerfile`: Change `ENV UV_PYTHON=3.12` to your desired version
2. Rebuild: `Dev Containers: Rebuild Container`

## Troubleshooting

### ARM64 Platform Compatibility Error

**Issue**: Full mode installation fails with error about `mlx` or other ML packages not having wheels for Linux ARM64

**Solution**: This only affects Full mode (ML frameworks). If you see this error:
1. Verify `docker-compose.yml` has `platform: linux/amd64` under the `dev` service's `build` section
2. Rebuild: `Dev Containers: Rebuild Container`
3. On Apple Silicon, Docker will use Rosetta emulation automatically
4. Alternatively, use Standard mode which has no ARM64 restrictions

### Container Build Fails

**Issue**: Docker build fails with network errors

**Solution**: Check your internet connection and Docker proxy settings. Try:
```bash
docker system prune -a  # Clean Docker cache
```

### `make install` Fails

**Issue**: Post-create command fails during container creation

**Solution**:
1. Open a terminal in the container
2. Run `make install` manually to see detailed error messages
3. Check that `uv`, `pre-commit`, and `deno` are installed: `which uv pre-commit deno`

### Tests Fail with Import Errors

**Issue**: Tests can't find installed packages

**Solution**: Ensure the virtual environment is activated:
```bash
source .venv/bin/activate
python -c "import sys; print(sys.prefix)"  # Should show /workspace/.venv
```

### Git Operations Fail

**Issue**: Git commands show "permission denied" or "unsafe repository"

**Solution**: The devcontainer automatically runs `git config --global --add safe.directory /workspace` on start. If issues persist:
```bash
git config --global --add safe.directory /workspace
```

### Port 8000 Already in Use

**Issue**: Can't access docs at `localhost:8000`

**Solution**: Check if another service is using port 8000:
```bash
# On host machine
lsof -i :8000
# Kill the process or use a different port in mkdocs serve
```

### Slow Performance

**Issue**: Container operations are slow

**Solution**:
- Ensure Docker has adequate resources (4+ GB RAM recommended)
- Use Docker Desktop's built-in resource settings
- Consider using WSL2 backend on Windows for better performance

## Performance Tips

1. **Persistent volumes**: The `.venv` volume persists across rebuilds, making subsequent starts much faster
2. **Cached mounts**: Workspace is mounted with `cached` consistency for better I/O performance
3. **Prune regularly**: Run `docker system prune` periodically to free disk space

## Security Considerations

- The container runs as non-root user `vscode` (UID 1000)
- Git credentials are mounted read-only
- SSH keys are mounted read-only
- MCP proxy should only expose necessary servers
- Never commit secrets or API keys to configuration files

## Additional Resources

- [Pydantic AI Contributing Guide](../docs/contributing.md)
- [Dev Containers Documentation](https://containers.dev/)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [uv Documentation](https://docs.astral.sh/uv/)

## Support

If you encounter issues with the devcontainer setup:

1. Check this README's troubleshooting section
2. Review container logs: `docker logs <container-id>`
3. Rebuild the container: `Dev Containers: Rebuild Container`
4. Open an issue in the [Pydantic AI repository](https://github.com/pydantic/pydantic-ai/issues)
