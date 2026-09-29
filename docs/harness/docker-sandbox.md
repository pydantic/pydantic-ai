---
title: Docker Sandbox
description: "Run a Pydantic AI agent's commands and file edits in a local Docker or Podman container, created on first use."
---

# Docker Sandbox

Run your agent's commands and file edits in a local [Docker](https://www.docker.com/) or Podman container, created on first use, so they can't touch the rest of your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/docker_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

No extra is needed for Docker itself: the capability runs the `docker` CLI already on your machine, so there is no SDK to install. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra.

## Quick start

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import Coder, DockerSandbox

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[DockerSandbox('python:3.13-slim', env={'PIP_NO_CACHE_DIR': '1'}), Coder()],
)
result = agent.run_sync('Write a script that prints the first 10 primes, and run it.')
```

`Coder`'s shell and file tools now run in a fresh container from `python:3.13-slim`, in `/workspace`. For a single run, pass `workspace=DockerSandboxBackend('python:3.13-slim')` (also imported from `pydantic_ai_harness`) to `agent.run` instead.

## Working on a project

The container starts empty. To let the agent work on a directory on your machine, mount it with `docker_args`, which go after the capability's own `docker run` arguments:

```python {test="skip"}
from pathlib import Path

from pydantic_ai_harness import DockerSandbox

project = Path('.').resolve()
sandbox = DockerSandbox(
    'python:3.13-slim',
    docker_args=['--volume', f'{project}:/workspace', '--memory', '2g', '--cpus', '2'],
    network=False,
)
```

The mount is the only part of your machine the commands can change. `network=False` runs the container with `--network none`.

## How commands run

The first operation runs `docker run --detach --init --entrypoint sh <image>` with a keep-alive loop, so the image needs a POSIX `sh` and the usual file utilities. Every command is a `docker exec` into that container, and file operations run as shell commands there. Pulling the image is not bounded by a command's `timeout`. Reading or writing a file takes one `docker exec` per 64 KiB, so mount a directory with `docker_args` for large data.

A container that can't be created or started, or that stops mid-command, raises `WorkspaceUnavailableError`. The exit code is the command's own, even when it is one `docker` also uses for its own errors.

On a timeout or cancellation, a second `docker exec` stops the command's process group in the container; a command that detached into a session of its own, like a [Shell](shell.md) background job, keeps running.

A command that leaves a background process holding its output open, such as `server &` without redirecting the server's output, doesn't return until that process exits, because `docker exec` waits for the output to close: redirect it, as in `server > server.log 2>&1 &`.

Podman works too: pass `executable='podman'`. The `docker` client gets `DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_CONFIG` and the other variables that select a daemon from your environment, besides the `PATH`, `HOME` and locale every local command gets, and nothing else from it.

## Reattach and clean up

The ref is `WorkspaceRef(provider='docker', id='pydantic-ai-<hex>')`, the container's name, set as soon as the container is created. `DockerSandbox(...).backend(ref)` attaches to it without I/O; the first operation runs `docker start`, so a stopped container comes back with its files, and a removed one raises `WorkspaceUnavailableError`.

The container outlives the run, like every workspace. Remove it, with its anonymous volumes, when you're done:

```python {test="skip"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness import DockerSandbox


async def remove_sandbox(ref: WorkspaceRef) -> None:  # for example `result.workspace.ref`
    await DockerSandbox('python:3.13-slim').destroy(ref)
```

Containers the capability created carry the label `ai.pydantic.workspace=true`, so `docker ps --all --filter label=ai.pydantic.workspace=true` finds any you've lost track of. Attaching to a ref and `destroy` both check that label first and refuse any other container, because a ref can come from stored message history and the Docker daemon serves every container on your machine.

## Security

A container shares your machine's kernel, so it is a weaker boundary than a VM such as [E2B](e2b-sandbox.md) or [Modal](modal-sandbox.md). Commands run as the image's user, usually `root` inside the container. Don't mount the Docker socket or directories holding secrets, and add hardening such as `--cap-drop ALL`, `--read-only` or `--user` through `docker_args` when the image allows it.

`env=` values are passed to `docker exec` as arguments, so while a command runs they are visible in process listings (`ps`) on your machine. `env=` is kept out of the capability's `repr`.

## Platforms

The machine running the agent must be POSIX (Linux or macOS), like [`LocalWorkspace`](../workspace.md#platforms): constructing `DockerSandbox` on Windows raises `NotImplementedError`.

## Telemetry

`DockerSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the workspace on the agent run span as `pydantic_ai.workspace.provider` (`docker`) and `pydantic_ai.workspace.id` (the container name, recorded even without `include_content`, because it identifies the environment rather than content), and each command or file operation a tool makes runs inside that tool call's span.

## API reference

::: pydantic_ai_harness.docker_sandbox.DockerSandbox

::: pydantic_ai_harness.docker_sandbox.DockerSandboxBackend
