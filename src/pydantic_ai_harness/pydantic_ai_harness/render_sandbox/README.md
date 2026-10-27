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

Render Sandboxes are in early access. This integration is tested with Render SDK 1.2.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox. With `Coder`, the sandbox is created when the run starts; pass `Coder(repo_context=False)` to create it on the first tool call instead. The sandbox keeps running, and billing, after the run ends; see [Clean up](#clean-up).

A new sandbox can reach the internet; pass `allow_internet_access=False` to block it. It lives for 1 hour; pass up to `sandbox_timeout=86400` for longer.

Commands start in `/workspace`, which is created on a new sandbox. Pass `working_dir=` for another absolute directory.

## Choose the tools

`RenderSandbox` supplies the workspace only. Add [`Coder`](../coder/) for a coding agent, or the narrower [`Shell`](../shell/) and [`FileSystem`](../filesystem/).

A string command runs under `/bin/sh -c`; a list runs as given. Standard input is empty, and a non-zero exit is a normal result. Output over 10 MiB stops the command and raises `WorkspaceOutputLimitError` with the first 64 KiB; redirect large output to a file. A command and its environment must fit in about 110 KB; write larger content with the file tools.

The sandbox image needs `python3`. Render's default image has it.

Render allows 100 API requests a minute. A rate-limited request raises the SDK's `RateLimitError`, and is not retried.

### What a timeout stops

On a timeout or cancellation, the command and its child processes are killed, and the sandbox stays usable. A background process that starts its own session keeps running.

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

The process that reattaches needs `RENDER_API_KEY` and `RENDER_WORKSPACE_ID` too. A terminated sandbox raises `WorkspaceUnavailableError`, and no replacement is created. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`allow_internet_access` and `sandbox_timeout` apply only to a new sandbox. `working_dir` and `env` apply to every command; an attached sandbox must already have `working_dir`.

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
    backend = RenderSandboxBackend()
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

The `finally` terminates the sandbox even if setup or the run fails. Read any files you want to keep before it.

## Clean up

The sandbox keeps running, and billing, until `sandbox_timeout` runs out. Pydantic AI never terminates it. Terminate it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.render_sandbox import RenderSandbox


async def terminate_sandbox(ref: WorkspaceRef) -> None:
    await RenderSandbox().destroy(ref)
```

`destroy(ref)` terminates the sandbox by ID. It returns quietly if the sandbox is already gone, and raises `WorkspaceUnavailableError` if Render rejects the credentials. Each new sandbox's ID is also logged at INFO as `Created Render sandbox <id>`.

To terminate the sandbox after every run, including failed ones, use `after_run` and `run_error` hooks:

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

To keep results, read them through `ctx.workspace` in `after_run` before calling `destroy`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `render.RenderAsync` you own; it is never closed for you. Default: `None`, which reads `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `sandbox_timeout` | Seconds a new sandbox lives before Render terminates it, at most `86_400`. Default: `3_600`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve from. Default: `/workspace`, created on a new sandbox. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment is passed. |

`RenderSandbox` needs asyncio; on Trio it raises `UserError`.

## Render Workflows

`RenderSandbox` composes with [`RenderWorkflows`](../render/): model and tool tasks share one sandbox. The `render` extra covers both; set the Render variables on every worker. A retried root task runs the agent again, so terminate the sandbox from the task that runs the agent, as in [Clean up](#clean-up).

## Telemetry

`RenderSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span.

## API reference

::: pydantic_ai_harness.render_sandbox.RenderSandbox

::: pydantic_ai_harness.render_sandbox.RenderSandboxBackend
