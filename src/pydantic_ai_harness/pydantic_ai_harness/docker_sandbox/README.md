# Docker Sandbox

Run your agent's commands and file edits in a local [Docker](https://www.docker.com/) or Podman container, created on first use, away from your machine's files and processes.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/docker_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[anthropic]"
```

No extra is needed for Docker itself: the capability runs the `docker` CLI already on your machine, so there is no SDK to install. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra, and set `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
```

`Coder`'s shell and file tools now run in a container, not on your machine. The container is created on the run's first use of the workspace. With `Coder`, `RepoContext` uses it when the run starts, so use `Coder(repo_context=False)` to create it only when a tool needs it. The container keeps running after the run ends; see [Clean up](#clean-up).

The image needs a POSIX `sh` and the usual file utilities, and Docker pulls it the first time. Commands run as the image's user, usually `root`. They start in `/workspace`, which is created if it's missing.

A new container is on Docker's default bridge network, which reaches the internet, but also your local network and services on this machine. Pass `allow_internet_access=False` to run it with `--network none`.

## Continue in the same container

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox(image='python:3.13-slim'), Coder()])
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')

followup = agent.run_sync('Now make it print the first 100.', message_history=result.all_messages())
```

The follow-up run finds the container in the message history and works in it, so the script is still there. Without the history, a run creates a new container. If the container has been removed, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](https://pydantic.dev/docs/ai/harness/shell/) and [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) instead of `Coder`, or write your own tool:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.docker_sandbox import DockerSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox(image='python:3.13-slim'), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the container."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Work on a project directory

A new container starts empty. To let the agent work on a directory on your machine, mount it with `docker_args`, extra `docker run` arguments:

```python {names="defined"}
from pathlib import Path

from pydantic_ai_harness.docker_sandbox import DockerSandbox

project = Path('.').resolve()
sandbox = DockerSandbox(
    image='python:3.13-slim',
    docker_args=['--volume', f'{project}:/workspace', '--memory', '2g', '--cpus', '2'],
    allow_internet_access=False,
)
```

The mount is the only part of your machine's files the commands can change; the network is a separate setting. For a network of your own, leave `allow_internet_access` on and pass `--network` in `docker_args`, since Docker refuses a second `--network` next to `--network none`. `docker_args` can't change the container's name, its `ai.pydantic.workspace=true` label or its entrypoint. Set the directory commands start in with `working_dir`.

A new container gets `--pids-limit 2048`, so a runaway fork loop can't use up the process table of the machine (or VM) running Docker. A command that hits it fails with `Cannot fork` or `Resource temporarily unavailable`; pass your own `--pids-limit` in `docker_args` to change it, or `-1` to remove it.

## Use a remote Docker host

The capability runs the `docker` CLI on your machine, so it uses whichever daemon that CLI points at. To run containers on another host, point the CLI there before the agent starts; your code doesn't change:

```bash
export DOCKER_HOST=ssh://dev@build-box  # or: docker context use build-box
```

The host needs Docker, and `ssh` must log in without a prompt, through a key or your SSH agent. `DOCKER_HOST` overrides `DOCKER_CONTEXT`, which overrides `docker context use`, so unset the ones you don't use. Paths in `docker_args`, such as a `--volume` source, are paths on that host, and `allow_internet_access` reaches that host's network. Each run reads these settings when it starts, and `destroy` when it's called, so keep them the same while a container is in use.

## Reattach later

To come back to the container without the message history, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox(image='python:3.13-slim'), Coder()])

result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Now make it print the first 100.', workspace=ref)
```

The ref is `WorkspaceRef(provider='docker', id='pydantic-ai-<hex>')`, the container's name. Attaching starts a stopped container on first use, so it comes back with its files. Pass `workspace='new'` to create a fresh container even when the message history names one. `image`, `allow_internet_access` and `docker_args` only shape a new container. `working_dir` and `env` apply to every command.

A ref is trusted input: whoever passes one runs commands in that container. `DockerSandbox` refuses a container without its `ai.pydantic.workspace=true` label, so a ref can't reach the other containers the Docker daemon serves. The label isn't proof of origin, though: any `DockerSandbox` on this machine, or anyone who can run `docker`, can create a container with it, so only pass refs your application stored itself.

## Prepare a container before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox(image='python:3.13-slim'), Coder()])


async def main() -> None:
    backend = DockerSandboxBackend(image='python:3.13-slim')
    try:
        workspace = Workspace(backend)
        await workspace.write_text('calc.py', 'def add(a, b):\n    return a - b\n')
        install = await workspace.run(['pip', 'install', 'pytest'], timeout=300)
        if install.exit_code != 0:
            raise RuntimeError(f'pip install failed: {install.stderr}')
        result = await agent.run('Fix the bug in calc.py.', workspace=backend)
        print(result.output)
    finally:
        ref = backend.ref  # set once the container is being created
        if ref is not None:
            await DockerSandbox().destroy(ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` removes the container even when setup or the run fails. To keep it instead, store `backend.ref` and pass it as `workspace=` to reattach later. `DockerSandbox`'s settings don't apply to a backend you pass as `workspace=`; give the backend its own.

## Clean up

A run never removes its container. Remove it, with its anonymous volumes, with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.docker_sandbox import DockerSandbox


async def remove_container(ref: WorkspaceRef) -> None:
    await DockerSandbox().destroy(ref)
```

Removing a container that is already gone returns quietly. `DockerSandbox()` without an `image` attaches to and removes containers by ref, and leaves a run without a ref to the agent's other workspace capabilities.

Nothing removes a container for you, and there is no idle or lifetime limit. `docker ps --all --filter label=ai.pydantic.workspace=true` finds the containers `DockerSandbox` created, including any left behind without a stored ref.

`backend.ref` is set when the first operation starts creating the container, and goes back to `None` if creating it fails before a container exists. If the create is cancelled, Docker may still finish creating the container after `destroy` has found nothing to remove; the label filter above finds it. A failed run returns no result, so there is no ref to store. Remove the container in an `on_run_error` hook, as shown for [E2B](https://pydantic.dev/docs/ai/harness/e2b-sandbox/#clean-up), or create the backend yourself as in [Prepare a container before the run](#prepare-a-container-before-the-run).

## How commands run

The container doesn't run the image's own entrypoint or command, so a service the image would start on boot isn't running; start it with a command. Creating or starting the container, including pulling the image, is not bounded by a command's `timeout`. File operations run as commands in the container, so large files are slow to read and write: mount a directory with `docker_args` for large data.

A container that can't be created or started, or that stops mid-command, raises `WorkspaceUnavailableError`. The exit code is the command's own, even when it is one `docker` also uses for its own errors.

On a timeout or cancellation, the command and the processes it started are stopped in the container; a command that detached into a session of its own, like a [Shell](https://pydantic.dev/docs/ai/harness/shell/) background job, keeps running.

With Docker, a command that leaves a background process holding its output open, such as `server &` without redirecting the server's output, doesn't return until that process exits: redirect it, as in `server > server.log 2>&1 &`. With Podman, it returns when the command itself exits.

Podman works too: pass `executable='podman'`. The CLI is looked up on `PATH` once, when the capability or backend is created. The `docker` client gets `DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_CONFIG`, `SSH_AUTH_SOCK` and the other variables that select and reach a daemon from your environment, besides the `PATH`, `HOME` and locale every local command gets, and nothing else from it.

## Security model

A container shares your machine's kernel, so it is a weaker boundary than a VM such as [Smol Sandbox](https://pydantic.dev/docs/ai/harness/smol-sandbox/), [E2B](https://pydantic.dev/docs/ai/harness/e2b-sandbox/) or [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/). Inside, commands run as the image's user, usually `root`, so anything the agent runs can change anything in the container.

- Don't mount the Docker socket or directories holding secrets. A writable mount also lets commands change files your machine runs later, such as a virtualenv, the `docker` CLI inside one, or git hooks. Add hardening such as `--cap-drop ALL`, `--user` or `--read-only` through `docker_args` when the image allows it.
- A missing `working_dir` is created as the container's user, so with `--user`, point `working_dir` at a directory that user can write, such as a mount, or pass `working_dir=None` to use the image's own. With `--read-only`, mount a writable working directory.
- On Linux, files the commands write to a mounted directory belong to the container's user on your machine too, which is `root` unless you pass `--user` with your own IDs, such as `--user 1000:1000` from `id -u` and `id -g`.
- While a command runs, its `env` values are visible in process listings (`ps`) on your machine.
- Docker adds the proxy settings in your `~/.docker/config.json` to every new container as environment variables.
- `image` is pulled by tag; pin a digest, such as `python:3.13-slim@sha256:...`, to get the same image every time.

## Configuration

| Option | What it does |
| --- | --- |
| `image` | Image for a new container, such as `'python:3.13-slim'`. Default: `None`, which only attaches to and removes containers by ref. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against, created on first use if missing. Default: `/workspace`; `None` uses the image's own (`WORKDIR`, or `/`). |
| `env` | Environment variables every command gets, on top of the image's. Default: `None`. Nothing from your machine's environment reaches the container except Docker's proxy settings, and `env` is kept out of the capability's `repr`, but the agent's commands can read it. |
| `allow_internet_access` | Whether a new container gets outbound network access, through Docker's default bridge. Default: `True`; `False` runs it with `--network none`. |
| `docker_args` | Extra `docker run` arguments for a new container, such as mounts, `--memory` or `--user`; a `--pids-limit` here replaces the default of 2048. Default: `()`. |
| `executable` | The container CLI to run, such as `'podman'`, looked up on `PATH` when the capability is created. Default: `'docker'`. |

## Platforms

The machine running the agent must be POSIX (Linux or macOS), like [`LocalWorkspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#platforms). On Windows, `DockerSandboxBackend` raises `NotImplementedError` when it is constructed, so a run with `DockerSandbox` does too.

## Telemetry

`DockerSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the container on the agent run span as `pydantic_ai.workspace.provider` (`docker`) and `pydantic_ai.workspace.id` (its name). Each command or file operation a tool makes runs inside that tool call's span.

## API reference

::: pydantic_ai_harness.docker_sandbox.DockerSandbox

::: pydantic_ai_harness.docker_sandbox.DockerSandboxBackend
