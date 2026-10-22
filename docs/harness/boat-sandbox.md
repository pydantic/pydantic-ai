---
title: Boat Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated Boat cloud sandbox instead of on your machine."
---

# Boat Sandbox

Run your agent's commands and file edits in an isolated [Boat](https://boat.dev) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/boat_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

No extra is needed for Boat itself: the capability talks to Boat's REST API with `httpx`, which the harness already depends on. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `BOAT_API_KEY` to a [Boat API key](https://docs.boat.dev/api-keys), and `ANTHROPIC_API_KEY` for the examples. A key created with Boat's `ci` preset is enough for the sandboxes it creates: it can create, resume, use, and delete them. Boat added resume and delete to `ci` later, so an older `ci` key cannot resume or delete; create a new one ([Boat's scopes](https://docs.boat.dev/api-keys/scopes#presets)).

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boat_sandbox import BoatSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoatSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It keeps running, and costs money, after the run ends until Boat stops it, one hour after it was created by default; see [Clean up](#clean-up).

Give the agent a project directory with `BoatSandbox(working_dir='/home/user/project')`: a new sandbox gets it created for you, and commands and relative paths start there.

A new sandbox runs Ubuntu 24.04 with git, Python, Node.js, Docker, and ripgrep (`rg`), which `Coder` uses for searches; see [Boat's pre-installed tools](https://docs.boat.dev/machines/tools). Commands run as `user`, who has passwordless `sudo`, so the agent can install what else it needs. For your project's dependencies, start from a named snapshot; see [Prepare a sandbox before the run](#prepare-a-sandbox-before-the-run).

A new sandbox is created with Boat's `noEnv` setting. Without it, Boat gives every command your GitHub token and Boat tokens, which code the agent runs could read and use. With it, the sandbox holds none of your account's secrets and cannot act on your account or your other sandboxes; the one Boat token it keeps, `ASCII_TOKEN`, can only expose ports of that sandbox ([Boat's scopes](https://docs.boat.dev/api-keys/scopes#in-sandbox-api-key)). Pass the secrets a task needs with `env=`; its values travel in each command's command line, so the sandbox's process list and Boat's command history show them (see [Configuration](#configuration)).

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boat_sandbox import BoatSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoatSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If Boat stopped the sandbox, the follow-up run resumes it: files on its disk are kept, but running programs, `/tmp`, and `~/.cache` may not be. If the sandbox has been deleted, or Boat deleted its disk 14 days after it stopped, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool that runs in the sandbox:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.boat_sandbox import BoatSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('{{ anthropic:large }}', capabilities=[BoatSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python3', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Commands run under `sh -c` as `user`. File reads and writes under `/home/user` and `/tmp` go through Boat's file API; other paths go through shell commands run as `user`. Boat's file API reads as root, so a read can return a file that `user`'s commands cannot open; writes run as `user`. File reads and writes refuse FIFOs with `OSError`: a read at once, and a write once its request times out after 30 seconds, since Boat's file API waits for a reader before it answers a write. File writes create missing parent directories, write through a symlink to its target, and keep an existing file's mode.

### What a timeout stops

A command timeout starts after the sandbox is ready. Each command runs as a detached Boat command, so Boat's 600-second limit on commands that wait for their result does not apply. A timeout or a cancelled run stops the command through Boat's process API, which ends its whole process tree, background children included, and the sandbox keeps running for other commands. `WorkspaceTimeoutError` carries the last 4 KiB of each stream's output so far. If Boat does not confirm the stop within 15 seconds, the command may still be running: inspect the sandbox or delete it yourself. `timeout=None` removes the command deadline, not Boat's auto-stop; a sandbox that Boat stops during a command makes `run()` raise `WorkspaceUnavailableError`. Boat limits API requests to 1,000 per minute from one IP address, and the checks on a running command count toward that limit: they start 0.2 seconds apart and back off to one every 2 seconds.

`run()` returns when the command exits, even if a background child it started is still running; output that child writes afterwards is not returned. A command whose stdout or stderr passes 8 MiB, Boat's limit per stream, raises `WorkspaceOutputLimitError` once it exits, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large or background output to a file instead; `Shell.start_command` manages its own output log.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.boat_sandbox import BoatSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoatSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `BOAT_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`machine_type`, `snapshot`, and `auto_stop` only shape a new sandbox; a resumed sandbox keeps the auto-stop it was created with. `working_dir` and `env` apply to every command, including after you reattach. A new sandbox gets `working_dir` created for it; on an attached sandbox it must already exist, or commands fail with `WorkspaceError`.

Already have a Boat sandbox, such as one from `boat new`? Pass `workspace=WorkspaceRef(provider='boat', id='bx_...')` to a run, with `WorkspaceRef` from `pydantic_ai.workspaces`. The sandbox keeps the settings it was created with: unless it was created with `noEnv`, its commands can read your account's credentials. A `ci` key cannot resume or delete a sandbox it did not create.

A run leaves a backend you built open. Call `await backend.aclose()` when you are done with it: that closes the client it opened from `BOAT_API_KEY`, not the sandbox, and a later operation opens a new one. A `client=` you passed is never closed.

## Prepare a sandbox before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.boat_sandbox import BoatSandbox, BoatSandboxBackend
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[BoatSandbox(), Coder()])


async def run_in_prepared_sandbox() -> str:
    backend = BoatSandboxBackend(working_dir='/home/user/project')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the sandbox exists
        if ref is not None:
            await BoatSandbox().destroy(ref)
```

The run uses the backend as it is, so `BoatSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later.

To start every sandbox with your project's dependencies installed, set them up once in a sandbox, save it as a [named snapshot](https://docs.boat.dev/snapshots/templates) with `boat snapshot <id> <name>`, and pass `BoatSandbox(snapshot='<name>')`: each new sandbox starts from that disk. A named snapshot does not expire, and it keeps every file on its disk, so save it from a sandbox that holds nothing you would not give the agent. A snapshot Boat doesn't know fails on first use with `WorkspaceUnavailableError`.

## Clean up

Pydantic AI never stops or deletes the sandbox. Boat stops it `auto_stop` seconds after it was created or last resumed: one hour by default, or never with `auto_stop=0`. The deadline does not move while commands run, so a run still working at that point fails with `WorkspaceUnavailableError`, and a later run that attaches resumes the sandbox. A stopped sandbox costs nothing and keeps its disk for 14 days; after that Boat deletes the disk and the sandbox can no longer be resumed. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.boat_sandbox import BoatSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await BoatSandbox().destroy(ref)
```

`destroy(ref)` deletes by sandbox ID without resuming a stopped sandbox, and a sandbox that no longer exists returns quietly. Only destroy sandboxes you own. Use `backend(ref)` to construct a lazy backend for an existing sandbox.

A failed run returns no result, so there is no ref to store. To delete its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.boat_sandbox import BoatSandbox
from pydantic_ai_harness.coder import Coder

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await BoatSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[BoatSandbox(), Coder(), hooks])
```

A run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the sandbox behind. Its ID is logged at `INFO` as `Created Boat sandbox <id>`, and `boat list --all` shows your sandboxes, stopped ones included; delete it with `await BoatSandbox().destroy(WorkspaceRef(provider='boat', id=...))`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | An `httpx.AsyncClient` to share across runs on one event loop, or to route through a proxy. Give it `base_url='https://boat.dev/api/v1'` and an `Authorization: Bearer <key>` header. Default: `None`, a client each run opens from `BOAT_API_KEY` and closes when it ends. You close a client you pass; `BoatSandbox` never does. |
| `machine_type` | Size of a new sandbox: `'small'`, `'default'`, `'large'`, or `'xlarge'`. See [Boat's sizes](https://docs.boat.dev/machines). Default: `None`, Boat's `default` size, or the snapshot's size when `snapshot` is set. A size your plan doesn't allow fails on first use. |
| `snapshot` | Named snapshot a new sandbox starts from. Default: `None`, Boat's base image. An unknown snapshot fails on first use. |
| `auto_stop` | Seconds after a new sandbox is created, and after each resume, before Boat stops it, keeping its disk. Command activity does not move the deadline. `0` disables it; on Boat's free trial, `0` or a value over 7200 fails on first use. Default: `None`, Boat's default of 1 hour. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the sandbox's own (`/home/user`, where commands run as `user`); set a project directory, or use relative paths for portable code. Created on a new sandbox; on an attached sandbox it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. Values travel in each command's command line, so the sandbox's process list and Boat's command history show them. |

Boat runs on the asyncio event loop only: the backend finishes sandbox creation and command cleanup on asyncio tasks of its own, so under Trio it raises `UserError`.

## Durable execution

Each activity opens its own Boat client and closes it once its operation ends, so the worker has nothing to close. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.boat_sandbox import BoatSandbox
from pydantic_ai_harness.coder import Coder
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    '{{ anthropic:large }}',
    name='boat_coder',
    capabilities=[BoatSandbox(), Coder(), TemporalDurability()],
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

Creating a sandbox sends Boat an `Idempotency-Key`, so when a reply is lost the backend asks again without creating a second sandbox. The key lasts as long as the backend, and Temporal builds a new backend for each activity, so a retried activity that creates the sandbox can leave the first one behind; the `Created Boat sandbox <id>` log line finds it. A command is sent again only when Boat reports that it did not start, or when the request could not reach Boat. When a reply about a command is lost, `run()` raises `WorkspaceError`, which durable engines record rather than retry, since the command may have run.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`BoatSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox logs its ID at `INFO` (`Created Boat sandbox <id>`) on the `pydantic_ai_harness.boat_sandbox._backend` logger, so a sandbox that outlives its run can still be found.

## API reference

::: pydantic_ai_harness.boat_sandbox.BoatSandbox

::: pydantic_ai_harness.boat_sandbox.BoatSandboxBackend
