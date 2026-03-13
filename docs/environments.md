# Execution Environments & Sandboxes

Pydantic AI provides [`ExecutionEnvironment`][pydantic_ai.environments.ExecutionEnvironment] — an abstraction for environments where agents can execute commands, read/write files, and search the filesystem — along with [`ExecutionEnvironmentToolset`][pydantic_ai.environments.ExecutionEnvironmentToolset], a ready-made [toolset](toolsets.md) that exposes these capabilities as tools.

This is the foundation for building coding agents, data analysis bots, and other agents that need to interact with a shell and filesystem.

## Quick Start

```python {title="environments_quickstart.py"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionEnvironmentToolset
from pydantic_ai.environments.local import LocalEnvironment

env = LocalEnvironment(root_dir='/tmp/workspace')
toolset = ExecutionEnvironmentToolset(env)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    async with env:
        result = await agent.run('Create a Python script that prints the first 10 Fibonacci numbers, then run it.')
        print(result.output)
        #> Done! The first 10 Fibonacci numbers are: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

## Environments

An [`ExecutionEnvironment`][pydantic_ai.environments.ExecutionEnvironment] defines where and how commands run. Three implementations are included:

| Environment | Isolation | Use case |
|---|---|---|
| [`LocalEnvironment`][pydantic_ai.environments.local.LocalEnvironment] | None — runs on host | Development, testing, trusted agents |
| [`DockerEnvironment`][pydantic_ai.environments.docker.DockerEnvironment] | Container-level | Production, untrusted code |
| [`MemoryEnvironment`][pydantic_ai.environments.memory.MemoryEnvironment] | In-memory (no filesystem) | Unit testing |

All environments are async context managers. Enter the environment before running the agent, and exit it to clean up:

```python {title="environments_lifecycle.py"}
from pydantic_ai.environments.docker import DockerEnvironment

env = DockerEnvironment(image='python:3.12-slim')

async def main():
    async with env:
        result = await env.shell('python -c "print(42)"')
        print(result.output)
        """
        42
        """
        #>
```

### LocalEnvironment

[`LocalEnvironment`][pydantic_ai.environments.local.LocalEnvironment] runs commands as local subprocesses within a specified root directory. It provides no isolation — use it for development, testing, and trusted agents.

```python {title="environments_local.py"}
from pydantic_ai.environments.local import LocalEnvironment

env = LocalEnvironment(
    root_dir='/tmp/workspace',
    env_vars={'PYTHONPATH': '/tmp/workspace/lib'},
    inherit_env=True,  # inherit host environment variables (default)
)
```

File operations (read, write, edit, ls, glob, grep) are confined to the root directory — path traversal attempts raise `PermissionError`.

!!! info "Environment variable inheritance"
    By default, `LocalEnvironment` inherits the host's environment variables. Set `inherit_env=False` for a clean environment where only explicitly provided `env_vars` (and per-call `env` overrides) are available. This is useful for reproducibility and testing.

### DockerEnvironment

[`DockerEnvironment`][pydantic_ai.environments.docker.DockerEnvironment] runs commands inside a Docker container with configurable resource limits, security options, and network access.

Requires the `docker` package: `pip install pydantic-ai-slim[docker-environment]`

```python {title="environments_docker.py"}
from pydantic_ai.environments.docker import DockerEnvironment

env = DockerEnvironment(
    image='my-sandbox:latest',
    env_vars={'MPLBACKEND': 'Agg'},
    memory_limit='512m',
    cpu_limit=1.0,
    network_disabled=True,
)
```

#### Building a custom Docker image

`DockerEnvironment` runs whatever image you give it — it doesn't install packages at startup. Pre-build a custom image with any libraries your agent needs, so containers start fast and reproducibly.

**Example Dockerfile** — a Python data-science sandbox:

```dockerfile {title="Dockerfile" test="skip" lint="skip"}
FROM python:3.12-slim

# Install OS-level tools the agent might use (optional)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir numpy pandas matplotlib requests

WORKDIR /workspace
```

Build and tag the image:

```bash
docker build -t my-sandbox:latest .
```

Then pass the tag to `DockerEnvironment`:

```python {title="environments_docker_custom.py"}
from pydantic_ai.environments.docker import DockerEnvironment

env = DockerEnvironment(image='my-sandbox:latest')
```

!!! tip "Tips for custom images"

    - **Start from a slim base** (`python:3.12-slim`, `node:22-slim`, etc.) to keep image size and attack surface small.
    - **Pin package versions** (e.g. `numpy==2.2.3`) for reproducible builds.
    - **Use `--no-cache-dir`** with pip to avoid bloating the image with cached wheels.
    - **Build once, run many times.** The image is pulled from the local Docker cache on each `DockerEnvironment` startup — no rebuild needed.
    - **Use a registry** for team or CI workflows: push your image to Docker Hub, GitHub Container Registry, or a private registry, then reference it by its full name (e.g. `ghcr.io/myorg/my-sandbox:latest`).
    - **For Node.js** or other runtimes, adjust the base image and install command accordingly:

        ```dockerfile {test="skip" lint="skip"}
        FROM node:22-slim
        RUN npm install -g typescript ts-node express
        WORKDIR /workspace
        ```

For running untrusted code, you can harden the container with Linux security options:

```python {title="environments_docker_hardened.py"}
from pydantic_ai.environments.docker import DockerEnvironment

env = DockerEnvironment.hardened(image='python:3.12-slim')
```

This uses the [`hardened()`][pydantic_ai.environments.docker.DockerEnvironment.hardened] convenience constructor, which sets sensible security defaults: network disabled, read-only root filesystem, all capabilities dropped, no privilege escalation, runs as `nobody`, uses an init process, and limits PIDs, memory, and CPU. You can customize the resource limits:

```python {title="environments_docker_hardened_custom.py"}
from pydantic_ai.environments.docker import DockerEnvironment

env = DockerEnvironment.hardened(
    image='my-sandbox:latest',
    memory_limit='1g',
    cpu_limit=2.0,
    pids_limit=512,
)
```

## ExecutionEnvironmentToolset

[`ExecutionEnvironmentToolset`][pydantic_ai.environments.ExecutionEnvironmentToolset] wraps an environment and exposes coding-agent-style tools that models are well-trained on (matching tools that popular coding agents expose):

| Tool | Description |
|---|---|
| `ls` | List directory contents |
| `shell` | Execute shell commands |
| `read_file` | Read files with line numbers (renders images for multimodal models) |
| `write_file` | Create or overwrite files |
| `edit_file` | Edit files by exact string replacement |
| `glob` | Find files by pattern |
| `grep` | Search file contents with regex |

Tools are dynamically registered based on the environment's capabilities. You can selectively include or exclude capabilities:

```python {title="environments_selective_tools.py"}
from pydantic_ai.environments import ExecutionEnvironmentToolset
from pydantic_ai.environments.memory import MemoryEnvironment

# Only file tools — no shell or search
toolset = ExecutionEnvironmentToolset(
    MemoryEnvironment(),
    include=['read_file', 'write_file', 'edit_file'],
)
```

### Using with an Agent

The toolset manages the environment lifecycle when used as a context manager:

```python {title="environments_agent.py"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionEnvironmentToolset
from pydantic_ai.environments.docker import DockerEnvironment

env = DockerEnvironment(image='python:3.12-slim')
toolset = ExecutionEnvironmentToolset(env)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    async with toolset:  # starts the Docker container
        result = await agent.run('Fetch https://httpbin.org/get and print the response')
        print(result.output)
        """
        Successfully fetched the URL. The response contains request metadata including headers and origin IP.
        """
    # container cleaned up automatically
```

!!! tip "Pre-starting the environment"
    Using `async with toolset:` starts the environment once and keeps it alive across all agent runs. Without it, the environment is started and stopped on each `agent.run()` call — for Docker, that means creating and destroying a container every time. Pre-start the toolset for better performance when running the agent multiple times.

!!! note "Shared environment"
    When you pass an environment directly, all concurrent `agent.run()` calls share the same environment instance (same container, filesystem, and processes). For isolated concurrent runs, use `environment_factory` — see [Concurrent Runs](#concurrent-runs) below.

### Environment Overrides

You can swap the backing environment at runtime using [`use_environment()`][pydantic_ai.environments.ExecutionEnvironmentToolset.use_environment]:

```python {title="environments_override.py"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionEnvironmentToolset
from pydantic_ai.environments.docker import DockerEnvironment
from pydantic_ai.environments.local import LocalEnvironment

toolset = ExecutionEnvironmentToolset(LocalEnvironment('/tmp/dev'))

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    # Default: local environment
    async with LocalEnvironment('/tmp/dev') as local_env:
        with toolset.use_environment(local_env):
            await agent.run('echo "running locally"')

    # Override: Docker environment for untrusted input
    async with DockerEnvironment() as docker_env:
        with toolset.use_environment(docker_env):
            await agent.run('echo "running in Docker"')
```

### Concurrent Runs

When multiple `agent.run()` calls execute concurrently (e.g. via `asyncio.gather`), a shared environment means they all operate on the same filesystem and processes, which can cause interference. Use `environment_factory` to create a fresh, isolated environment for each run:

```python {title="environments_concurrent.py"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionEnvironmentToolset
from pydantic_ai.environments.docker import DockerEnvironment

# Each concurrent run gets its own container
toolset = ExecutionEnvironmentToolset(
    environment_factory=lambda: DockerEnvironment(image='python:3.12-slim')
)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    # Each agent.run() enters its own `async with toolset:`, creating a separate container
    await asyncio.gather(
        agent.run('task A'),
        agent.run('task B'),
    )
```

The factory is called once per `async with toolset:` entry, and the created environment is automatically cleaned up on exit.

## Per-Call Environment Variables

All environments support per-call environment variables via the `env` parameter on [`shell()`][pydantic_ai.environments.ExecutionEnvironment.shell] and [`create_process()`][pydantic_ai.environments.ExecutionEnvironment.create_process]. These are merged on top of any baseline `env_vars`:

```python {title="environments_env_vars.py" test="skip"}
from pydantic_ai.environments.local import LocalEnvironment

environment = LocalEnvironment(env_vars={'BASE_URL': 'https://api.example.com'})

async def main():
    async with environment:
        # Uses BASE_URL from baseline + API_KEY from per-call
        result = await environment.shell(
            'curl -H "Authorization: Bearer $API_KEY" $BASE_URL/data',
            env={'API_KEY': 'sk-test-123'},
        )
        print(result.output)
```

## Interactive Processes

For long-running or interactive workloads, use [`create_process()`][pydantic_ai.environments.ExecutionEnvironment.create_process] to get an [`ExecutionProcess`][pydantic_ai.environments.ExecutionProcess] with bidirectional streaming I/O:

```python {title="environments_process.py" test="skip"}
from pydantic_ai.environments.local import LocalEnvironment

env = LocalEnvironment()

async def main():
    async with env:
        async with await env.create_process('python3 -u worker.py') as proc:
            await proc.send(b'{"task": "analyze"}\n')
            response = await proc.recv(timeout=10.0)
            print(response.decode())
```

## Execution Model

Each call to `shell()` or `create_process()` starts a fresh process. Shell state (like `cd`, shell variables) does not persist between calls. This is the same model used by other coding agents like Claude Code and Codex.

To run commands in a specific directory, chain them:

```python {title="environments_chaining.py" test="skip" lint="skip"}
result = await env.shell('cd /some/path && python script.py')
```

Filesystem changes (created files, installed packages) persist for the lifetime of the environment.

## Building a Custom Environment

You can implement [`ExecutionEnvironment`][pydantic_ai.environments.ExecutionEnvironment] to integrate with any execution backend. The only abstract member is `capabilities`; override the methods that match your declared capabilities. Override [`create_process()`][pydantic_ai.environments.ExecutionEnvironment.create_process] if you need interactive process support.

```python {title="environments_custom.py" test="skip" lint="skip"}
from typing import Literal

from pydantic_ai.environments import EnvToolName, ExecutionEnvironment, ExecutionProcess, ExecutionResult, FileInfo

class MyCloudEnvironment(ExecutionEnvironment):
    @property
    def capabilities(self) -> frozenset[EnvToolName]:
        return frozenset({'shell', 'read_file', 'write_file', 'edit_file', 'ls', 'glob', 'grep'})

    async def shell(
        self, command: str, *, timeout: float | None = 120, env: dict[str, str] | None = None
    ) -> ExecutionResult:
        # Run a command in your cloud environment
        ...

    async def read_file(
        self, path: str, *, offset: int = 0, limit: int = 2000
    ) -> str | bytes:
        ...

    async def write_file(self, path: str, content: str | bytes) -> None:
        ...

    async def replace_str(
        self, path: str, old: str, new: str, *, replace_all: bool = False
    ) -> int:
        ...

    async def ls(self, path: str = '.') -> list[FileInfo]:
        ...

    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        ...

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
    ) -> str:
        ...
```
