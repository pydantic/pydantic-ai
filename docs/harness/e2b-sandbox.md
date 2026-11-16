---
title: E2B Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated E2B cloud sandbox instead of on your machine."
---

# E2B Sandbox

Run your agent's commands and file edits in an isolated [E2B](https://e2b.dev) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/e2b_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

```mermaid
flowchart TB
    accTitle: E2B Sandbox: your agent's commands run in an E2B cloud sandbox
    accDescr: On your machine, the model calls Coder, Shell or FileSystem tools such as run_command. Via ctx.workspace, E2BSandbox's get_workspace gives the run an E2B cloud sandbox over the E2B API with E2B_API_KEY, paused at its timeout and killed by destroy(ref). A later run reattaches with the WorkspaceRef in the message history or result.workspace.ref and resumes the sandbox if paused.
    subgraph machine["Your machine"]
        model(["model"])
        tools["Coder, Shell or<br/>FileSystem tools,<br/>such as run_command"]
        ref["WorkspaceRef<br/>in the message history,<br/>result.workspace.ref"]
        e2b["E2BSandbox()<br/>get_workspace gives<br/>the run an E2B sandbox"]:::capability
    end
    sandbox["E2B cloud sandbox<br/>paused at its timeout,<br/>killed by destroy(ref)"]:::external
    model <-->|"tool call, result"| tools
    tools -->|"ctx.workspace"| e2b
    ref -.->|"a later run reattaches<br/>and resumes it if paused"| e2b
    e2b <-->|"E2B API, E2B_API_KEY"| sandbox
    classDef capability stroke:#E520E9,stroke-width:2px
    classDef external stroke:#0E9C86,stroke-width:2px
```

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,e2b,anthropic]"
```

Set `E2B_API_KEY` to your E2B API key. The examples also use `Coder` and an Anthropic model, hence the `coder` and `anthropic` extras and `ANTHROPIC_API_KEY`; swap them for your own tools and model provider.

## Quick start

<span id="choose-the-tools"></span>

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox. For a narrower agent, use [`Shell`](shell.md), [`FileSystem`](filesystem.md) or your own tools on [`ctx.workspace`](../workspace.md#using-the-workspace-in-tools).

- **It keeps running, and billing, after the run.** See [Clean up](#clean-up).
- **It pauses after 1 hour** (`sandbox_timeout=3600`, the most E2B's Hobby plan allows), and the next run resumes it. Pro plans allow up to `sandbox_timeout=86400`.
- **Commands start in `/home/user`.** Pass `working_dir='/home/user/project'` to keep the project out of the home directory; it is created on a new sandbox.
- **The default template has Python and git**, but not your project's dependencies or ripgrep. Bake them into an E2B [template](https://e2b.dev/docs/template/quickstart) and pass `template='my-template'`, or install them [before the run](#prepare-a-sandbox-before-the-run).

## Continue in the same sandbox

<span id="reattach-later"></span>

Pass the message history, and the next run works in the same sandbox, resuming it if paused:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
ref = result.workspace.ref  # or store this, and pass `workspace=ref` to a later run
```

If the sandbox has been killed, the run raises `WorkspaceUnavailableError`; pass `workspace='new'` to start over. [Workspaces](../workspace.md#continuing-in-the-same-workspace) covers how runs find their workspace.

The ref holds no credentials: the process that reattaches needs `E2B_API_KEY` or a `client=` too.

Already have an `e2b.AsyncSandbox`? Pass `workspace=E2BSandboxBackend(sandbox=sandbox, working_dir=..., env=...)` to a run; `E2BSandbox`'s settings don't apply to it.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

agent = Agent('{{ anthropic:large }}', capabilities=[E2BSandbox(), Coder()])


async def main() -> None:
    backend = E2BSandboxBackend(working_dir='/home/user/project')
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
            await E2BSandbox().destroy(ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` kills the sandbox even when setup or the run fails. To keep it, store `backend.ref` instead.

## Clean up

Kill a sandbox with the ref you stored. This works on a paused sandbox too, without resuming it:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.e2b_sandbox import E2BSandbox


async def kill_sandbox(ref: WorkspaceRef) -> None:
    await E2BSandbox().destroy(ref)
```

A failed run returns no result, so kill its sandbox in an `on_run_error` hook:

```python
from typing import Any

import anyio

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        # Shielded so a deadline around the run can't cut the kill short, bounded so a hung call can't hang it.
        with anyio.move_on_after(30, shield=True):
            await E2BSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[E2BSandbox(), Coder(), hooks])
```

A sandbox you don't kill pauses when its `sandbox_timeout` runs out; see [E2B's sandbox lifecycle](https://docs.e2b.dev/sandbox). Each new sandbox's id is logged as `Created E2B sandbox <id>` at INFO on the `pydantic_ai_harness.e2b_sandbox._backend` logger, so one created by a run that was cancelled before storing it can still be found; if E2B's answer never arrived, find it in the E2B dashboard.

## Use different credentials for each user

`E2BSandbox(client=E2B(api_key=...))` sets one client for the capability. When each user has their own E2B account, supply the workspace from a capability of your own that builds the client from the run's `deps`:

```python {names="defined"}
from dataclasses import dataclass

from e2b import E2B
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend


@dataclass
class User:
    e2b_api_key: str


@dataclass
class UsersE2B(AbstractCapability[User]):
    def get_workspace(self, ctx: RunContext[User], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is not None and ref.provider != 'e2b':
            return None
        if not ctx.deps.e2b_api_key:  # E2B would fall back to `E2B_API_KEY`
            raise ValueError('This user has no E2B API key.')
        return E2BSandboxBackend(client=E2B(api_key=ctx.deps.e2b_api_key), ref=ref)


agent = Agent('{{ anthropic:large }}', deps_type=User, capabilities=[UsersE2B(), Coder()])
```

- E2B's client falls back to `E2B_API_KEY` when its key is empty, so check for a key first, as above, or a user without one runs on that account.
- Destroy a user's sandbox with that user's key: `await E2BSandbox(client=E2B(api_key=user.e2b_api_key)).destroy(ref)`. With another account's client the sandbox isn't found, `destroy` returns without an error, and the sandbox keeps billing.
- Under [Temporal](../durable_execution/temporal.md), `deps` are stored in the workflow's history. Put a user ID there, and look the key up in `get_workspace` on the worker.

## Preview a dev server

With `Shell`, have the agent start the server with `start_command`, such as `npm run dev -- --host 0.0.0.0 --port 3000`. Then get its public hostname from the sandbox, with `capability` being the agent's `E2BSandbox`:

```text
sandbox = await capability.backend(ref).get_sandbox()
url = f'https://{sandbox.get_host(3000)}'
```

The preview is public. Stop it with `stop_command` while the workspace is attached, and kill the sandbox when you're done.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | An `e2b.E2B` for the API key and connection settings. Default: `None`, which reads `E2B_API_KEY` and the rest of the environment. |
| `template` | E2B template name or ID for a new sandbox. Default: `None`, E2B's `base`. An unknown template fails on first use. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `sandbox_timeout` | Seconds until E2B pauses the sandbox, set on create and on reattach. Default: `3_600`. Over your plan's limit raises `WorkspaceUnavailableError`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, `/home/user` on the default template. Created on a new sandbox; on an attached one it must exist. |
| `env` | Environment variables every command gets. Default: `None`; nothing from your machine reaches the sandbox. The agent's commands can read these, so leave out secrets it shouldn't see. The default locale is POSIX; pass `{'LC_ALL': 'C.UTF-8'}` for tools that should count characters rather than bytes. |

`template` and `allow_internet_access` only shape a new sandbox; the other options also apply when you reattach, and `destroy` uses `client`.

## Limits

<span id="what-a-timeout-stops"></span>

- E2B's file service runs as root, so Unix permissions don't restrict file tools. Commands run as `user`.
- Concurrent writes to one path can interleave, and a reader can see a partial file. Write to separate paths.
- A background process that keeps the command's stdout or stderr open keeps `run()` waiting. Redirect its output to a file; `Shell.start_command` does this for you.
- A command whose output passes 10 MiB is stopped with `WorkspaceOutputLimitError`. Redirect large output to a file.
- On a timeout, the command's process group is stopped. Custom templates need `/bin/bash`, `/bin/sh` and `setsid` for that; without `setsid`, only the command itself is stopped.
- Stopping is best effort: if E2B can't be reached, the command may keep running, so keep the ref to inspect or kill the sandbox.
- The E2B SDK needs asyncio; under Trio, every operation raises `UserError`.

## Durable execution

Add `TemporalDurability()` next to `E2BSandbox()` and install the `temporal` extra; [Workspaces](../workspace.md#durable-execution) covers what changes. A durable run creates its sandbox when it starts.

## Telemetry

`E2BSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id` on the agent run span, and each command or file operation runs inside its tool call's span.

## Further reading

- [Run Pydantic AI coding agents in E2B sandboxes](https://pydantic.dev/articles/pydantic-ai-e2b-sandboxes) (Pydantic): a coding agent in an E2B sandbox

## API reference

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend
