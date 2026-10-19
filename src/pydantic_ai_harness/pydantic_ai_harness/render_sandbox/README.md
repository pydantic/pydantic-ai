# Render Sandbox

Run your agent's commands and file edits in an isolated [Render sandbox](https://render.com/docs/sandboxes) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[render,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[render,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `RENDER_API_KEY` to a [Render API key](https://render.com/docs/api#1-create-an-api-key), `RENDER_WORKSPACE_ID` to the `tea-...` ID of a [workspace](https://render.com/docs/workspaces) with Sandboxes access, and `ANTHROPIC_API_KEY` for the examples.

Render Sandboxes are in early access, and their SDK API sits under `render.experimental`, so it may change between SDK releases; this integration is tested against Render SDK 1.2.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(allow_internet_access=True), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. The sandbox keeps running, and billing, after the run ends; see [Clean up](#clean-up).

A new sandbox has no internet access unless you pass `allow_internet_access=True`, so `git clone` and `pip install` fail without it. It lives for 1 hour by default, then times out. Pass up to `sandbox_timeout=86400` for longer work.

Commands start in `/workspace`, which is created on a new sandbox. Pass `working_dir=` for another absolute directory.

## Choose the tools

`RenderSandbox` supplies the workspace only. Add [`Coder`](../coder/) for a coding agent, or the narrower [`Shell`](../shell/) and [`FileSystem`](../filesystem/).

A string command runs under `/bin/sh -c`; an argument list runs as given. Standard input is at EOF, and a non-zero exit is a normal result. A command whose combined output passes 1 MiB is stopped and raises `WorkspaceOutputLimitError` with the captured beginning in `stdout` and `stderr`; redirect large output to a file instead. A command and its environment must fit in about 110 KB, the limit on one exec argument, where each non-ASCII character takes 6 bytes (12 for emoji); write larger content with the file tools.

Render's exec API cannot stop a command, so the backend runs each command under a small Python supervisor sent with the exec request. On a timeout or cancellation, it kills the command's process group and leaves the sandbox usable. A background child that detaches into its own session keeps running. The supervisor and the file helpers need `python3` in the sandbox image, which Render's default image provides.

File contents travel through Render's `copy_to` and `copy_from`. A write is staged and then moved over the destination, so an interrupted upload leaves the old file in place. Metadata operations, such as listing a directory, run as shell commands.

Render allows 100 API requests a minute. A command is one request, a file read three, and a file write four. A rate-limited request raises the SDK's `RateLimitError`, which is not retried for you; durable engines retry it as a transient error.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(), Coder()])

result = agent.run_sync('Write a fizzbuzz.py and run it.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Add tests for fizzbuzz.py.', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `RENDER_API_KEY` and `RENDER_WORKSPACE_ID` too. A sandbox that has been terminated raises `WorkspaceUnavailableError`; the backend never creates a replacement. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`allow_internet_access` and `sandbox_timeout` only shape a new sandbox. `working_dir` and `env` apply to every command, and an attached sandbox must already have `working_dir`.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(), Coder()])


async def main() -> None:
    backend = RenderSandboxBackend(allow_internet_access=True)
    try:
        workspace = Workspace(backend)
        await workspace.write_text('calc.py', 'def add(a, b):\n    return a - b\n')
        install = await workspace.run(['pip', 'install', 'pytest'], timeout=300)
        if install.exit_code != 0:
            raise RuntimeError(f'pip install failed: {install.stderr}')
        result = await agent.run('Fix the bug in calc.py.', workspace=backend)
        print(result.output)
    finally:
        ref = backend.ref  # set once the sandbox exists
        if ref is not None:
            await RenderSandbox().destroy(ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` terminates the sandbox even when setup or the run fails. Read any files you want to keep through `workspace` before it.

## Clean up

The sandbox keeps running, and billing, after the run ends, until `sandbox_timeout` runs out. Pydantic AI never terminates it. Terminate it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.render_sandbox import RenderSandbox


async def terminate_sandbox(ref: WorkspaceRef) -> None:
    await RenderSandbox().destroy(ref)
```

`RenderSandbox.backend(ref)` constructs an attached backend without I/O; `destroy(ref)` terminates by ID without attaching, and returns quietly for a sandbox that is already terminated or unknown. Rejected credentials raise `WorkspaceUnavailableError`. If a create is cancelled while in flight, the backend still records the ref when Render responds; a lost response may leave a sandbox without a ref, which `sandbox_timeout` then bounds. Creating a sandbox logs `Created Render sandbox <id>` at INFO on the `pydantic_ai_harness.render_sandbox._backend` logger, so the ID is on record even then.

To terminate every run's sandbox, including failed runs, which return no result to read the ref from, use `after_run` and `on_run_error` hooks:

```python {names="defined"}
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

sandbox = RenderSandbox()
hooks = Hooks()


@hooks.on.after_run
async def terminate_sandbox(ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await sandbox.destroy(ctx.workspace.ref)
    return result


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await sandbox.destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[sandbox, Coder(), hooks])
```

To check or download results before the sandbox goes, read them through `ctx.workspace` in `after_run` before calling `destroy`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A caller-owned `render.RenderAsync`, never closed for you. Default: `None`, a client per workspace operation from `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `False`. Does not affect the host's model requests. |
| `sandbox_timeout` | Seconds a new sandbox lives before Render terminates it, at most `86_400`. Default: `3_600`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `/workspace`. Created on a new sandbox; an attached sandbox must already have it. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. |

Every operation, including `destroy`, needs the asyncio event loop; on Trio it raises `UserError`.

## Render Workflows

`RenderSandbox` composes with [`RenderWorkflows`](../render/): the workspace ref crosses task boundaries, so model and tool tasks share one sandbox. The same `render` extra covers both; set the Render variables on every worker. A root-task retry starts the agent again and can repeat work, so terminate the sandbox from the task that runs the agent, as in [Clean up](#clean-up).

## Telemetry

`RenderSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span.

## API reference

::: pydantic_ai_harness.render_sandbox.RenderSandbox

::: pydantic_ai_harness.render_sandbox.RenderSandboxBackend
