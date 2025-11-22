# DevContainer Maintenance Guide

## About This Codebase

- **Pydantic AI**: Agent framework for building LLM-powered applications with Pydantic
- **Workspace structure**: uv monorepo with multiple packages
  - `pydantic-ai-slim`: Core framework (minimal dependencies)
  - `pydantic-evals`: Evaluation framework
  - `pydantic-graph`: Graph execution engine
  - `examples/`: Example applications
  - `clai/`: CLI tool
- **Primary users**: Contributors, AI coding agents (Claude Code, Cursor), PR reviewers

## DevContainer Purpose

- Provides isolated, reproducible development environment
- Matches exact dependencies and tools across all developers and AI agents
- Prevents "works on my machine" issues
- Ensures AI agents have proper access to testing/building tools
- Security isolation for AI agents

## Platform Configuration

- **Default platform**: `linux/amd64` (x86_64)
- **Why not ARM64**: Some Python packages (e.g., mlx) lack Linux ARM64 wheels
- **Apple Silicon**: Uses Rosetta/QEMU emulation automatically (slightly slower but compatible)
- **Change if needed**: Edit `docker-compose.yml` platform setting

## Installation Modes

### Standard Mode (Default)
- Installs: Cloud API providers + dev tools
- Excludes: PyTorch, transformers, vLLM, outlines ML extras
- Use case: 95% of development (PR testing, features, bug fixes)
- Why: Saves significant install time and disk space
- Command: Uses explicit `--extra` flags in `install.sh`

### Full Mode
- Installs: Everything including ML frameworks
- Use case: Working on outlines integration, local model features
- Command: `--all-extras --all-packages`

### Mode Selection
- Interactive (VSCode): User prompted to choose
- Non-interactive (agents/CI): Defaults to Standard
- Override: Set `INSTALL_MODE=standard|full` environment variable

## Key Files

### `.devcontainer/devcontainer.json`
- VSCode configuration for the devcontainer
- Editor settings, extensions, port forwarding
- Lifecycle commands (`postCreateCommand`, `postStartCommand`)
- Environment variables (`UV_LINK_MODE`, `UV_PROJECT_ENVIRONMENT`, etc.)
- Git identity for AI commits

### `.devcontainer/Dockerfile`
- Base image: `mcr.microsoft.com/devcontainers/base:debian-12`
- System dependencies for Python builds
- Installs: uv, deno, pre-commit, Python 3.12
- Runs as non-root user `vscode`

### `.devcontainer/docker-compose.yml`
- Service orchestration
- Platform specification (`linux/amd64`)
- Optional services (commented out): Ollama, PostgreSQL, pgvector, MCP proxy
- Volume management for persistence

### `.devcontainer/install.sh`
- Interactive installation script
- Detects interactive vs non-interactive mode
- Implements Standard vs Full installation logic
- Installs pre-commit hooks
- Called by `postCreateCommand` in devcontainer.json

## Environment Variables

### Critical Variables (devcontainer.json)
- `UV_PROJECT_ENVIRONMENT=/workspace/.venv`: Virtual environment location
- `UV_LINK_MODE=copy`: Suppress hardlink warnings in Docker volumes
- `PYTHONUNBUFFERED=1`: Ensure Python output appears immediately
- `COLUMNS=150`: Terminal width for better output formatting
- `GIT_AUTHOR_*`, `GIT_COMMITTER_*`: Git identity for AI commits

### Optional Variables
- `INSTALL_MODE=standard|full`: Override installation mode
- API keys: Should be set in `.devcontainer/.env` (not committed)

## Dependencies and Extras

### Always Installed (Both Modes)
- Core: pydantic, httpx, opentelemetry-api
- Cloud APIs: openai, anthropic, google, groq, mistral, cohere, bedrock, huggingface
- Dev tools: cli, mcp, fastmcp, logfire, retries, temporal, ui, ag-ui, evals
- Build tools: lint group, docs group (ruff, mypy, pyright, mkdocs)

### Only in Full Mode
- `outlines-transformers`: PyTorch + Transformers library
- `outlines-vllm-offline`: vLLM inference engine
- `outlines-sglang`: SGLang framework
- `outlines-mlxlm`: Apple MLX framework
- `outlines-llamacpp`: LlamaCPP bindings

## Common Maintenance Tasks

### Adding New System Dependencies
- Edit `Dockerfile`: Add to `apt-get install` command
- Rebuild container required

### Adding Python Packages
- Use `uv add package-name` (not manual pyproject.toml edits)
- For new extras: Add to `pydantic_ai_slim/pyproject.toml` optional-dependencies
- Update `install.sh` if extra should be in Standard mode

### Adding VSCode Extensions
- Edit `devcontainer.json`: Add to `customizations.vscode.extensions` array
- Rebuild container required

### Updating Base Image/Tools
- `Dockerfile`: Change base image tag
- Update uv/deno install commands if needed
- Test with both Standard and Full modes

### Adding Optional Services
- Uncomment service in `docker-compose.yml`
- Uncomment corresponding volume if needed
- Document in README.md optional services section

## Troubleshooting

### Container Build Fails
- Check Docker daemon is running
- Check internet connectivity
- Try: `docker system prune -a` to clean cache
- Check Dockerfile for syntax errors: `docker build -f .devcontainer/Dockerfile .`

### Installation Script Fails
- Check `install.sh` syntax: `bash -n .devcontainer/install.sh`
- Run manually in container to see detailed errors
- Check uv lockfile is up to date: `uv lock`

### Performance Issues
- Verify Docker resources (4+ GB RAM recommended)
- Check platform setting (amd64 vs arm64)
- Volume cache consistency setting in docker-compose.yml

### UV Warnings
- Hardlink warning: Ensure `UV_LINK_MODE=copy` is set
- Lockfile conflicts: Run `uv lock` to regenerate

## Best Practices

### When Changing install.sh
- Test both Standard and Full modes
- Test interactive and non-interactive flows
- Verify syntax with `bash -n install.sh`
- Update README.md to match

### When Changing Dependencies
- Keep Standard mode lean (exclude heavy ML frameworks)
- Update install.sh if adding new extras
- Document in README.md what's included/excluded
- Test install time impact

### When Updating Documentation
- Keep README.md user-facing and comprehensive
- Keep CLAUDE.md maintainer-focused and concise
- No time estimates (machine-dependent)
- Link to official docs where applicable

## Git Configuration

- Credentials forwarded automatically by VSCode (no manual setup needed)
- Git identity set via environment variables (not .gitconfig file)
- Safe directory configured in `postStartCommand`
- AI commits use identity from `GIT_AUTHOR_*` variables

## Testing the Setup

### Manual Test
1. Make changes to devcontainer files
2. Rebuild container: "Dev Containers: Rebuild Container"
3. Test Standard mode installation
4. Test Full mode: `INSTALL_MODE=full` or run `make install`
5. Verify: `uv run pytest tests/test_agent.py::test_simple_sync`

### CI Considerations
- Container should work in non-interactive mode
- Default Standard mode should cover 95% of test suite
- Full mode needed only for outlines/ML framework tests
