---
title: boxd Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated boxd cloud machine instead of on your machine."
---

# boxd Sandbox

Run your agent's commands and file edits in an isolated [boxd](https://boxd.sh) cloud machine instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/boxd_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![The model calls Coder, Shell or FileSystem tools such as run_command and edit_file. BoxdSandbox's get_workspace hook gives the run a boxd cloud machine as ctx.workspace, so each command and file operation goes over the boxd API, authenticated with BOXD_API_KEY, and runs in the machine as the boxd user; output comes back to the tool. The first use creates the machine, its WorkspaceRef is saved in the message history and on the result, a later run reattaches with it, and destroy(ref) deletes it. fork(ref) copies a machine, with its files and running processes, into a new isolated one.](../img/harness/boxd-sandbox.svg)

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,boxd,anthropic]"
```

The `coder` extra is there because the examples use `Coder`; the sandbox itself doesn't need it. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `BOXD_API_KEY` to an API key from `boxd auth keys create` (each key belongs to one boxd organization), and `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoxdSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the machine, not on your machine. With `Coder`, `RepoContext` creates the machine when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It keeps running after the run ends until boxd suspends or hibernates it for being idle, and keeps its disk until you delete it; see [Clean up](#clean-up).

Give the agent a project directory with `BoxdSandbox(working_dir='/home/boxd/project')`: a new machine gets it created for you, and commands and relative paths start there.

boxd's default image is Ubuntu 24.04 with git, Python 3, uv, Node.js, and ripgrep (`rg`), which `Coder` uses for searches. For your project's dependencies, install them in a machine, save it as a snapshot with `boxd snapshots save`, and pass `BoxdSandbox(snapshot='my-snapshot')`. A snapshot boxd doesn't know, or one it is still saving, fails on first use with `WorkspaceUnavailableError`. `BoxdSandbox(vcpu=4)` picks a larger machine: 1, 2, 4, or 8 vCPUs, with 4 GiB of memory each. It can't be combined with `snapshot`, since a machine from a snapshot keeps the size it was saved at.

## Continue in the same machine

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoxdSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the machine in the message history and works in it, so the clone is still there. Without the history, a run starts a new machine. If the machine was suspended or stopped, the follow-up run wakes or starts it again. If the machine has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool that runs in the machine:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('{{ anthropic:large }}', capabilities=[BoxdSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the machine."""
    result = await ctx.workspace.run(['python3', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Commands run under `sh -c` as the non-root `boxd` user, which has passwordless `sudo`. File writes create missing parent directories, write through a symlink to its target, and keep an existing file's mode, and file reads refuse FIFOs rather than waiting for a writer. File operations use boxd's file API as the same user, so Unix permissions apply to them as they do to commands. On a machine whose boxd agent predates that API, such as one created from an older snapshot, they run as shell commands until the machine reboots.

### What a timeout stops

A command timeout starts after the machine is ready. A timeout or a cancelled run stops the command and the processes in its process group, and the machine keeps running for other commands. `WorkspaceTimeoutError` carries the output the command printed so far. `timeout=None` doesn't lift boxd's own limit: it ends a command that prints nothing for 30 minutes, which also raises `WorkspaceTimeoutError`, so have long builds print progress.

`run()` returns when the command exits, even if a background child it started is still running; output that child writes afterwards is not returned. A command whose combined output passes 10 MiB is stopped and raises `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large or background output to a file instead; `Shell.start_command` manages its own output log.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the machine without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoxdSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `BOXD_API_KEY` too. Pass `workspace='new'` to start a fresh machine even when the message history names one.

`org`, `snapshot`, `vcpu`, and `auto_suspend_timeout` only shape a new machine; `working_dir` and `env` apply to every command, including after you reattach. A new machine gets `working_dir` created for it; on an attached machine it must already exist, or commands fail with `WorkspaceError`.

A run leaves a backend you built open. Call `await backend.aclose()` when you are done with it: that closes the client it opened from the environment, not the machine, and a later operation opens a new one. A `client=` you passed is never closed.

## Fork a machine

`fork(ref)` copies a machine into a new isolated one, with its files and running processes as they are at that moment, and returns the copy's ref. After that the two change independently, so agents can try different approaches from the same starting point at once:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

sandbox = BoxdSandbox()
agent = Agent('{{ anthropic:large }}', capabilities=[sandbox, Coder()])


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

The copy starts with everything the machine had, including what the agent wrote to disk, and isn't tied to it afterwards: destroy each when you are done.

## Prepare a machine before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoxdSandbox(), Coder()])


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

The run uses the backend as it is, so `BoxdSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the machine instead, store `backend.ref` and pass it as `workspace=` to reattach later. For boxd features the workspace API doesn't cover, such as snapshots and public URLs, `await backend.get_sandbox()` returns the machine's `boxd.Machine` record; use its `id` with your own `boxd.AsyncBoxd`.

## Clean up

Pydantic AI never stops or deletes the machine. boxd suspends, and later hibernates, a machine that has had no network activity for as long as your organization's defaults say (set `auto_suspend_timeout=` in seconds to change suspending for new machines, or `0` to turn it off). `BoxdSandbox` sets no delete timer, so boxd deletes the machine only if your organization's defaults give new machines an idle-destroy or delete-after timer; `boxd machine get` shows a machine's timers. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox


async def delete_machine(ref: WorkspaceRef) -> None:
    await BoxdSandbox().destroy(ref)
```

`destroy(ref)` deletes the machine whatever state it is in; a machine that no longer exists returns quietly. Use `backend(ref)` to construct a lazy backend for an existing machine.

A failed run returns no result, so there is no ref to store. To delete its machine, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_machine(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await BoxdSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[BoxdSandbox(), Coder(), hooks])
```

A run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the machine behind. Machines the backend creates are named `pydantic-ai-<hex>` and logged at `INFO` (`Created boxd machine <id> (<name>)`, or `Forked boxd machine ...` for a copy); find leftovers with `boxd list` and delete one with `await BoxdSandbox().destroy(WorkspaceRef(provider='boxd', id=...))`.

## Security model

Machines `BoxdSandbox` creates, and copies `fork` makes, are isolated: code the agent runs holds no boxd credentials and can't reach your other boxd machines. It can reach the internet, and like every boxd machine it has a public HTTPS address that serves the port its code listens on.

`BoxdSandbox` refuses to attach to or fork a machine that isn't isolated or is on a named boxd network, and `destroy` refuses one that isn't isolated. A machine from a snapshot keeps the networks of the machine the snapshot was saved from, so save snapshots from a machine on no network. A ref can still name any isolated machine your credentials can see, so keep refs server-side.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `boxd.AsyncBoxd` to share across runs on one event loop, or to point at another boxd cluster. Default: `None`, a client each run opens and closes when it ends, from `BOXD_TOKEN`, else `BOXD_API_KEY`, else the boxd machine it runs in. You close a client you pass; `BoxdSandbox` never does. |
| `org` | boxd organization a new machine belongs to. Default: `None`, the API key's organization, or your default one for other credentials. |
| `snapshot` | boxd snapshot a new machine starts from, by name or id. Default: `None`, boxd's default image. An unknown snapshot fails on first use. |
| `vcpu` | Size of a new machine: `1`, `2`, `4`, or `8` vCPUs, with 4 GiB of memory each. Default: `None`, the organization's default size. It can't be combined with `snapshot`: a machine from a snapshot keeps the size it was saved at. |
| `auto_suspend_timeout` | Seconds without network activity before boxd suspends a new machine; `0` disables it. Default: `None`, the organization's default. A command wakes a suspended machine. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the machine's own (`/home/boxd`, where commands run as non-root `boxd`); set a project directory, or use relative paths for portable code. Created on a new machine; on an attached machine it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the machine. |

boxd runs on the asyncio event loop only; under Trio the backend raises `UserError`.

## Durable execution

Pass one `client=`, as below. Without it, every workspace operation opens a new client that logs in to boxd again, and boxd allows 30 logins a minute from one IP address, so a busy worker fails with `RateLimitError`. With one client, every activity on the worker's event loop shares its session token. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import uuid

import boxd
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.coder import Coder
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    '{{ anthropic:large }}',
    name='boxd_coder',
    capabilities=[BoxdSandbox(client=boxd.AsyncBoxd()), Coder(), TemporalDurability()],
)


@workflow.defn
class SandboxWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='sandbox', workflows=[SandboxWorkflow]):
        print(
            await client.execute_workflow(
                SandboxWorkflow.run,
                'Use the shell tool to run pwd.',
                id=f'sandbox-{uuid.uuid4()}', task_queue='sandbox',
            )
        )


if __name__ == '__main__':
    asyncio.run(main())
```

`Coder`, `Shell`, and `FileSystem` work under DBOS, Temporal and Prefect. See the [Coder](coder.md#durable-execution), [Shell](shell.md#durable-execution), and [FileSystem](filesystem.md#durable-execution) guides for engine-specific limits.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`BoxdSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the machine on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a machine logs its id and name at `INFO` (`Created boxd machine <id> (<name>)`) on the `pydantic_ai_harness.boxd_sandbox._backend` logger.

## API reference

::: pydantic_ai_harness.boxd_sandbox.BoxdSandbox

::: pydantic_ai_harness.boxd_sandbox.BoxdSandboxBackend
