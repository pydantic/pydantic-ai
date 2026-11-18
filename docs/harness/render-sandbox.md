---
title: Render Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated Render sandbox instead of on your machine."
---

# Render Sandbox

Run your agent's commands and file edits in an isolated [Render sandbox](https://render.com/docs/sandboxes) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,render,anthropic]"
```

Set `RENDER_API_KEY` to a [Render API key](https://render.com/docs/api#1-create-an-api-key), and `RENDER_WORKSPACE_ID` to the `tea-...` ID of a [workspace](https://render.com/docs/team-members) with Sandboxes access and billing configured. The examples also use `Coder` and an Anthropic model, hence the `coder` and `anthropic` extras and `ANTHROPIC_API_KEY`; swap them for your own tools and model provider.

## Quick start

<span id="choose-the-tools"></span>

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[RenderSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox. For a narrower agent, use [`Shell`](shell.md), [`FileSystem`](filesystem.md) or your own tools on [`ctx.workspace`](../workspace.md#using-the-workspace-in-tools).

- **`Coder` creates it when the run starts**, to load the repository's instructions, even if no tool is called. Pass `Coder(repo_context=False)` to wait for the first tool call.
- **It keeps running, and billing, after the run**, until `sandbox_timeout` runs out: `RenderSandbox` turns off Render's idle suspension. See [Clean up](#clean-up).
- **It lives for 1 hour.** Pass up to `sandbox_timeout=86400` for longer.
- **Commands start in the sandbox's home directory.** Pass `working_dir=` for another absolute directory, created on a new sandbox.
- **It can reach the internet.** Pass `allow_internet_access=False` to block it.
- **The image needs `python3`.** Render's default image has it.

## Continue in the same sandbox

<span id="reattach-later"></span>

Pass the message history, and the next run works in the same sandbox:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[RenderSandbox(), Coder()])
result = agent.run_sync('Write a fizzbuzz.py and run it.')

followup = agent.run_sync('Add tests for fizzbuzz.py.', message_history=result.all_messages())
ref = result.workspace.ref  # or store this, and pass `workspace=ref` to a later run
```

If the sandbox has been terminated, the run raises `WorkspaceUnavailableError`; pass `workspace='new'` to start over. [Workspaces](../workspace.md#continuing-in-the-same-workspace) covers how runs find their workspace.

The ref holds no credentials: the process that reattaches needs `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`, or a `client=`, too.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

agent = Agent('{{ anthropic:large }}', capabilities=[RenderSandbox(), Coder()])


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

The `finally` terminates the sandbox even when setup or the run fails, so read any files you want to keep before it. To keep the sandbox, store `backend.ref` instead.

## Clean up

Terminate a sandbox with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.render_sandbox import RenderSandbox


async def terminate_sandbox(ref: WorkspaceRef) -> None:
    await RenderSandbox().destroy(ref)
```

`destroy(ref)` raises `WorkspaceUnavailableError` if Render rejects the credentials. To terminate the sandbox after every run, including failed ones, use `after_run` and `run_error` hooks:

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


agent = Agent('{{ anthropic:large }}', capabilities=[sandbox, Coder(), hooks])
```

To keep results, read them through `ctx.workspace` in `after_run` before calling `destroy`.

New sandbox ids are logged at INFO. A connection failure during creation can leave a sandbox running without a ref, so check your Render workspace for unused sandboxes.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `render.RenderAsync` you own; it is never closed for you. Default: `None`, which reads `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `sandbox_timeout` | Seconds a new sandbox lives before Render terminates it, at most `86_400`. Default: `3_600`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve from. Default: `None`, the sandbox's home directory. Created on a new sandbox; on an attached one it must exist. |
| `env` | Environment variables every command gets. Default: `None`; nothing from your machine reaches the sandbox. The agent's commands can read these, so leave out secrets it shouldn't see. |
| `authority` | Who wrote the sandbox's files, for capabilities that check [authority](../workspace-custom.md#workspace-authority): `'none'` for files nobody on your side wrote, such as a cloned third-party repository, `'user'` for the user's own. Default: `'operator'`. |
| `path_authority` | Levels for some directories and files, overriding `authority` under them; the longest matching path wins. Paths are sandbox paths, relative to the working directory or absolute; `~` isn't expanded. Default: `None`. |

`client`, `working_dir`, `env`, `authority` and `path_authority` also apply when you reattach, and `destroy` uses `client`; the other options only shape a new sandbox.

## Limits

<span id="what-a-timeout-stops"></span>

- Sandbox setup timeouts raise `TimeoutError`; missing sandboxes and rejected access raise `WorkspaceUnavailableError`.
- Render allows 100 API requests a minute. A rate-limited request raises the SDK's `RateLimitError` and is not retried.
- A command and its environment must fit in about 110 KB. Write larger content with the file tools.
- A command whose output passes 10 MiB raises `WorkspaceOutputLimitError`. Redirect large output to a file.
- Stopping a command on a timeout or cancellation is best effort, and a warning names the sandbox if it can't be confirmed. Background processes that detach can keep running.
- `RenderSandbox` needs asyncio; on Trio it raises `UserError`.

## Durable execution

<span id="render-workflows"></span>`RenderSandbox` composes with [`RenderWorkflows`](render-workflows.md): model and tool tasks share one sandbox. The `render` extra covers both; set the Render variables on every worker. A retried root task runs the agent again, so terminate the sandbox from the task that runs the agent, as in [Clean up](#clean-up).

## Telemetry

`RenderSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id` on the agent run span, and each command or file operation runs inside its tool call's span.

## API reference

::: pydantic_ai_harness.render_sandbox.RenderSandbox

::: pydantic_ai_harness.render_sandbox.RenderSandboxBackend
