# Execution Environments & Sandboxes

Pydantic AI provides [`ExecutionEnvironment`][pydantic_ai.environments.ExecutionEnvironment] — an abstraction for environments where agents can execute commands, read/write files, and search the filesystem — along with [`ExecutionToolset`][pydantic_ai.environments.ExecutionToolset], a ready-made [toolset](toolsets.md) that exposes these capabilities as tools.

This is the foundation for building coding agents, data analysis bots, and other agents that need to interact with a shell and filesystem.

## Quick Start

```python {title="environments_quickstart.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionToolset
from pydantic_ai.environments.local import LocalEnvironment

env = LocalEnvironment(root_dir='/tmp/workspace')
toolset = ExecutionToolset(env)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    async with env:
        result = await agent.run('Create a Python script that prints the first 10 Fibonacci numbers, then run it.')
        print(result.output)
```

## Environments

An [`ExecutionEnvironment`][pydantic_ai.environments.ExecutionEnvironment] defines where and how commands run. Four implementations are included:

| Environment | Isolation | Use case |
|---|---|---|
| [`LocalEnvironment`][pydantic_ai.environments.local.LocalEnvironment] | None — runs on host | Development, testing, trusted agents |
| [`DockerSandbox`][pydantic_ai.environments.docker.DockerSandbox] | Container-level | Production, untrusted code |
| [`E2BSandbox`][pydantic_ai.environments.e2b.E2BSandbox] | Cloud VM | Production, zero local setup |
| [`MemoryEnvironment`][pydantic_ai.environments.memory.MemoryEnvironment] | In-memory (no filesystem) | Unit testing |

All environments are async context managers. Enter the environment before running the agent, and exit it to clean up:

```python {title="environments_lifecycle.py" test="skip"}
from pydantic_ai.environments.docker import DockerSandbox

sandbox = DockerSandbox(image='python:3.12-slim', packages=['numpy'])

async def main():
    async with sandbox:
        result = await sandbox.execute('python -c "import numpy; print(numpy.__version__)"')
        print(result.output)
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

### DockerSandbox

[`DockerSandbox`][pydantic_ai.environments.docker.DockerSandbox] runs commands inside a Docker container with configurable images, packages, resource limits, and network access.

Requires the `docker` package: `pip install pydantic-ai-slim[docker-sandbox]`

```python {title="environments_docker.py" test="skip"}
from pydantic_ai.environments.docker import DockerSandbox

sandbox = DockerSandbox(
    image='python:3.12-slim',
    packages=['pandas', 'matplotlib'],
    setup_commands=['apt-get update && apt-get install -y git'],
    env_vars={'MPLBACKEND': 'Agg'},
    memory_limit='512m',
    cpu_limit=1.0,
    network_disabled=True,
)
```

Built images are cached by default (keyed on the image, packages, and setup commands) so subsequent starts are fast.

### E2BSandbox

[`E2BSandbox`][pydantic_ai.environments.e2b.E2BSandbox] runs commands in a cloud-hosted micro-VM via [E2B](https://e2b.dev). It provides full Linux isolation with no local Docker required.

Requires the `e2b-code-interpreter` package: `pip install pydantic-ai-slim[e2b-sandbox]`

```python {title="environments_e2b.py" test="skip"}
from pydantic_ai.environments.e2b import E2BSandbox

sandbox = E2BSandbox(
    template='base',
    # api_key defaults to E2B_API_KEY env var
    timeout=300,
    env_vars={'MY_VAR': 'value'},
)
```

## ExecutionToolset

[`ExecutionToolset`][pydantic_ai.environments.ExecutionToolset] wraps an environment and exposes coding-agent-style tools that models are well-trained on (matching tools that popular coding agents expose):

| Tool | Description |
|---|---|
| `bash` | Execute shell commands |
| `read_file` | Read files with line numbers (renders images for multimodal models) |
| `write_file` | Create or overwrite files |
| `edit_file` | Edit files by exact string replacement |
| `glob` | Find files by pattern |
| `grep` | Search file contents with regex |

Tools can be selectively included:

```python {title="environments_selective_tools.py"}
from pydantic_ai.environments import ExecutionToolset
from pydantic_ai.environments.memory import MemoryEnvironment

# Only bash — no file or search tools
toolset = ExecutionToolset(
    MemoryEnvironment(),
    include_bash=True,
    include_file_tools=False,
    include_search_tools=False,
)
```

### System Prompt

The toolset provides a [`system_prompt`][pydantic_ai.environments.ExecutionToolset.system_prompt] property with a suggested prompt describing the available tools and best practices. You can use it as part of your agent's system prompt:

```python {title="environments_system_prompt.py"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionToolset
from pydantic_ai.environments.memory import MemoryEnvironment

toolset = ExecutionToolset(MemoryEnvironment())

agent = Agent(
    'test',
    system_prompt=toolset.system_prompt,
    toolsets=[toolset],
)
```

### Using with an Agent

The toolset manages the environment lifecycle when used as a context manager:

```python {title="environments_agent.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionToolset
from pydantic_ai.environments.docker import DockerSandbox

sandbox = DockerSandbox(image='python:3.12-slim', packages=['requests'])
toolset = ExecutionToolset(sandbox)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    async with toolset:  # starts the Docker container
        result = await agent.run('Fetch https://httpbin.org/get and print the response')
        print(result.output)
    # container cleaned up automatically
```

### Environment Overrides

You can swap the backing environment at runtime using [`use_environment()`][pydantic_ai.environments.ExecutionToolset.use_environment]:

```python {title="environments_override.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.environments import ExecutionToolset
from pydantic_ai.environments.docker import DockerSandbox
from pydantic_ai.environments.local import LocalEnvironment

toolset = ExecutionToolset(LocalEnvironment('/tmp/dev'))

agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    # Default: local environment
    async with LocalEnvironment('/tmp/dev') as local_env:
        with toolset.use_environment(local_env):
            await agent.run('echo "running locally"')

    # Override: Docker sandbox for untrusted input
    async with DockerSandbox() as docker_env:
        with toolset.use_environment(docker_env):
            await agent.run('echo "running in Docker"')
```

## Per-Call Environment Variables

All environments support per-call environment variables via the `env` parameter on [`execute()`][pydantic_ai.environments.ExecutionEnvironment.execute] and [`create_process()`][pydantic_ai.environments.ExecutionEnvironment.create_process]. These are merged on top of any baseline `env_vars`:

```python {title="environments_env_vars.py" test="skip"}
from pydantic_ai.environments.local import LocalEnvironment

env = LocalEnvironment(env_vars={'BASE_URL': 'https://api.example.com'})

async def main():
    async with env:
        # Uses BASE_URL from baseline + API_KEY from per-call
        result = await env.execute(
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

Each call to `execute()` or `create_process()` starts a fresh process. Shell state (like `cd`, shell variables) does not persist between calls. This is the same model used by other coding agents like Claude Code and Codex.

To run commands in a specific directory, chain them:

```python {title="environments_chaining.py" test="skip" lint="skip"}
result = await env.execute('cd /some/path && python script.py')
```

Filesystem changes (created files, installed packages) persist for the lifetime of the environment.

## Building a Custom Environment

You can implement [`ExecutionEnvironment`][pydantic_ai.environments.ExecutionEnvironment] to integrate with any execution backend. The required methods are [`execute()`][pydantic_ai.environments.ExecutionEnvironment.execute] and the file/search operations (`read_file`, `read_file_bytes`, `write_file`, `edit_file`, `ls`, `glob`, `grep`). Override [`create_process()`][pydantic_ai.environments.ExecutionEnvironment.create_process] if you need interactive process support.

```python {title="environments_custom.py" test="skip" lint="skip"}
from typing import Literal

from pydantic_ai.environments import ExecutionEnvironment, ExecutionProcess, ExecuteResult, FileInfo

class MyCloudEnvironment(ExecutionEnvironment):
    async def execute(
        self, command: str, *, timeout: float | None = 120, env: dict[str, str] | None = None
    ) -> ExecuteResult:
        # Run a command in your cloud environment
        ...

    async def read_file(
        self, path: str, *, offset: int = 0, limit: int = 2000
    ) -> str:
        ...

    async def read_file_bytes(self, path: str) -> bytes:
        ...

    async def write_file(self, path: str, content: str | bytes) -> None:
        ...

    async def edit_file(
        self, path: str, old_string: str, new_string: str, *, replace_all: bool = False
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
