# boxd Sandbox

Run your agent's commands and file edits in an isolated [boxd](https://boxd.sh) cloud machine instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/boxd_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

```mermaid
flowchart TB
    accTitle: boxd Sandbox: your agent's commands run in a boxd cloud machine
    accDescr: On your machine, the model calls Coder, Shell or FileSystem tools like run_command. Via ctx.workspace, BoxdSandbox's get_workspace hands the run a boxd cloud machine over the boxd API with BOXD_API_KEY. Later runs reattach with the WorkspaceRef in the message history or result.workspace.ref, destroy(ref) deletes the machine and fork(ref) copies it with its files and processes.
    subgraph machine["Your machine"]
        model(["model"])
        tools["Coder, Shell or<br/>FileSystem tools,<br/>such as run_command"]
        ref["WorkspaceRef<br/>in the message history,<br/>result.workspace.ref"]
        boxd["BoxdSandbox()<br/>get_workspace hands<br/>the run a boxd machine"]:::capability
    end
    sandbox["boxd cloud machine<br/>created on first use,<br/>deleted by destroy(ref)"]:::external
    copy["copy of the machine<br/>starts with the same<br/>files and processes"]:::external
    model <-->|"tool call, result"| tools
    tools -->|"ctx.workspace"| boxd
    ref -.->|"a later run reattaches"| boxd
    boxd <-->|"boxd API,<br/>BOXD_API_KEY"| sandbox
    sandbox -->|"fork(ref)"| copy
    classDef capability stroke:#E520E9,stroke-width:2px
    classDef external stroke:#0E9C86,stroke-width:2px
```

## Install

uv:

```bash
uv add "pydantic-ai-harness[coder,boxd,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder,boxd,anthropic]"
```

Set `BOXD_API_KEY` to an API key from `boxd auth keys create`; each key belongs to one boxd organization. The examples also use `Coder` and an Anthropic model, hence the `coder` and `anthropic` extras and `ANTHROPIC_API_KEY`; swap them for your own tools and model provider.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxdSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in a boxd machine. For a narrower agent, use [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) or your own tools on [`ctx.workspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#using-the-workspace-in-tools).

- **It keeps its disk until you delete it.** boxd suspends it when idle, but never deletes it unless your organization's defaults say so. See [Clean up](#clean-up).
- **Commands run as the non-root `boxd` user in `/home/boxd`**, with passwordless `sudo`. Pass `working_dir='/home/boxd/project'` to give the agent a project directory; it is created on a new machine.
- **The default image is Ubuntu 24.04 with git, Python 3, uv, Node.js and `rg`**, but not your project's dependencies. Install them in a machine, save it with `boxd snapshots save`, and pass `snapshot='my-snapshot'`. An unknown snapshot, or one still saving, fails on first use with `WorkspaceUnavailableError`.
- **It has a public HTTPS address** that serves the port its code listens on, and can reach the internet. See [Security model](#security-model).

## Continue in the same machine

Pass the message history, and the next run works in the same machine, waking or starting it if needed:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxdSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
ref = result.workspace.ref  # or store this, and pass `workspace=ref` to a later run
```

If the machine has been deleted, the run raises `WorkspaceUnavailableError`; pass `workspace='new'` to start over. [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/#continuing-in-the-same-workspace) covers how runs find their workspace.

The ref holds no credentials: the process that reattaches needs `BOXD_API_KEY` or a `client=` too.

## Prepare a machine before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxdSandbox(), Coder()])


async def run_in_prepared_machine() -> str:
    backend = BoxdSandboxBackend(working_dir='/home/boxd/project')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the machine exists
        if ref is not None:
            await BoxdSandbox().destroy(ref)
```

The run leaves a backend you built open, so close it with `aclose()`; that closes the client it opened from the environment, not the machine. A `client=` you passed is never closed. To keep the machine, store `backend.ref` instead of destroying it.

For boxd features the workspace API doesn't cover, such as snapshots and public URLs, `await backend.get_sandbox()` returns the machine's `boxd.Machine` record; use its `id` with your own `boxd.AsyncBoxd`.

## Clean up

Delete a machine with the ref you stored, whatever state it is in:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox


async def delete_machine(ref: WorkspaceRef) -> None:
    await BoxdSandbox().destroy(ref)
```

A failed run returns no result, so delete its machine in an `on_run_error` hook:

```python
from typing import Any

import anyio

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_machine(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        # Shielded so a deadline around the run can't cut the delete short, bounded so a hung call can't hang it.
        with anyio.move_on_after(30, shield=True):
            await BoxdSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoxdSandbox(), Coder(), hooks])
```

A machine you don't delete is suspended, and later hibernated, after your organization's idle time; set `auto_suspend_timeout=` in seconds for new machines, or `0` to turn it off. `BoxdSandbox` sets no delete timer, so boxd deletes the machine only if your organization's defaults give new machines one; `boxd machine get` shows a machine's timers.

A run that ends before its ref is stored, such as a crash, Ctrl-C or a killed worker, leaves the machine behind. Machines the backend creates are named `pydantic-ai-<hex>` and logged as `Created boxd machine <id> (<name>)`, or `Forked boxd machine ...` for a copy, at INFO on the `pydantic_ai_harness.boxd_sandbox._backend` logger. Find leftovers with `boxd list` and delete one with `await BoxdSandbox().destroy(WorkspaceRef(provider='boxd', id=...))`.

## Fork a machine

`fork(ref)` copies a machine into a new isolated one, with its files and running processes as they are at that moment, and returns the copy's ref. The two then change independently, so agents can try different approaches from the same starting point at once:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

sandbox = BoxdSandbox()
agent = Agent('anthropic:claude-opus-5-5', capabilities=[sandbox, Coder()])


async def main() -> None:
    setup = await agent.run('Clone https://github.com/pydantic/pydantic-ai and install it with `uv sync`.')
    ref = setup.workspace.ref
    assert ref is not None  # set once the run has used the machine
    copy = await sandbox.fork(ref)

    await asyncio.gather(
        agent.run('Speed up the slowest test by caching what it builds.', workspace=ref),
        agent.run('Speed up the slowest test by mocking its network calls.', workspace=copy),
    )
    await asyncio.gather(sandbox.destroy(ref), sandbox.destroy(copy))


asyncio.run(main())
```

Destroy each machine when you are done; the copy isn't tied to the original.

## Security model

- Machines `BoxdSandbox` creates, and copies `fork` makes, are isolated: code the agent runs holds no boxd credentials and can't reach your other boxd machines.
- They can reach the internet, and like every boxd machine have a public HTTPS address that serves the port their code listens on.
- `BoxdSandbox` refuses to attach to or fork a machine that isn't isolated or is on a named boxd network, and `destroy` refuses one that isn't isolated.
- A machine from a snapshot keeps the networks of the machine the snapshot was saved from, so save snapshots from a machine on no network.
- A ref can name any isolated machine your credentials can see, so keep refs server-side.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `boxd.AsyncBoxd` to share across runs on one event loop, or to point at another boxd cluster. Default: `None`, a client each run opens and closes when it ends, from `BOXD_TOKEN`, else `BOXD_API_KEY`, else the boxd machine it runs in. You close a client you pass. |
| `org` | boxd organization a new machine belongs to. Default: `None`, the API key's organization, or your default one for other credentials. |
| `snapshot` | boxd snapshot a new machine starts from, by name or id. Default: `None`, boxd's default image. An unknown snapshot fails on first use. |
| `vcpu` | Size of a new machine: `1`, `2`, `4` or `8` vCPUs, with 4 GiB of memory each. Default: `None`, the organization's default size. Can't be combined with `snapshot`: a machine from a snapshot keeps the size it was saved at. |
| `auto_suspend_timeout` | Seconds without network activity before boxd suspends a new machine; `0` disables it. Default: `None`, the organization's default. A command wakes a suspended machine. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, `/home/boxd`. Created on a new machine; on an attached one it must exist. |
| `env` | Environment variables every command gets. Default: `None`; nothing from your machine reaches the boxd machine. The agent's commands can read these, so leave out secrets it shouldn't see. |

`client`, `working_dir` and `env` also apply when you reattach, and `destroy` uses `client`; the other options only shape a new machine.

## Limits

- boxd ends a command that prints nothing for 30 minutes, even with `timeout=None`, raising `WorkspaceTimeoutError`. Have long builds print progress.
- A timeout or cancellation stops the command's process group, not the machine.
- `run()` returns when the command exits, so output a background child writes afterwards is lost. Redirect it to a file; `Shell.start_command` does this for you.
- A command whose output passes 10 MiB is stopped with `WorkspaceOutputLimitError`. Redirect large output to a file.
- The boxd SDK needs asyncio; under Trio, every operation raises `UserError`.

## Durable execution

Add a durability capability, such as `TemporalDurability()` from the `temporal` extra, next to `BoxdSandbox()`; [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) covers what changes. Pass one client, as in `BoxdSandbox(client=boxd.AsyncBoxd())`: without it, every workspace operation logs in to boxd again, and boxd allows 30 logins a minute from one IP address, so a busy worker fails with `RateLimitError`.

## Telemetry

`BoxdSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id` on the agent run span, and each command or file operation runs inside its tool call's span.

## API reference

::: pydantic_ai_harness.boxd_sandbox.BoxdSandbox

::: pydantic_ai_harness.boxd_sandbox.BoxdSandboxBackend
