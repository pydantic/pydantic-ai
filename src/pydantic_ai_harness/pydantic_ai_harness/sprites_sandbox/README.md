# Sprites Sandbox

Run your agent's commands and file edits in a persistent [Fly.io Sprite](https://sprites.dev) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/sprites_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[sprites,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[sprites,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `SPRITE_TOKEN` to your Sprites API token, and `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in a Sprite, not on your machine. With `Coder`, `RepoContext` creates the Sprite when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It has no lifetime limit: it sleeps when idle and keeps its files, and costs money, until you delete it; see [Clean up](#clean-up).

Give the agent a project directory with `SpritesSandbox(working_dir='/home/sprite/project')`: a new Sprite gets it created for you, and commands and relative paths start there.

A new Sprite comes with git, Python, and Node.js ([preinstalled tools](https://docs.fly.io/sprites/working-with-sprites/)), but not pytest or ripgrep (`rg`). Install what your project needs, such as pytest; Sprites retain installed packages. For faster `Coder` searches, run `sudo apt-get update && sudo apt-get install ripgrep` once in the Sprite and store its ref to reuse it on later runs.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the Sprite in the message history and works in it, so the clone is still there. Without the history, a run starts a new Sprite. If the Sprite has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one. A command exiting 137 after confirmed Sprite deletion also raises this error; a SIGKILLed command in a live Sprite returns exit 137.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool that runs in the Sprite:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the Sprite."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

A Sprite pauses processes between commands unless you run them as a [Sprites service](https://docs.sprites.dev/working-with-sprites/services/).

Commands run as the non-root `sprite` user, but Unix permissions do not restrict it: a file with mode `000` is still readable. Do not rely on them to keep tools out of a path.

File reads refuse FIFOs rather than waiting for a writer. File writes go through the Sprites filesystem API, which writes through a symlink to its target and creates missing parent directories.

### What a timeout stops

A command timeout starts after the Sprite is ready. The backend closes that command's exec connection and asks Sprites to stop it after one second; it does not delete the Sprite. Stopping is best effort: the command's process group, plain `&` children of a shell included, is not guaranteed to have stopped. If stopping is uncertain, inspect the Sprite or delete it explicitly. `timeout=None` removes the command deadline, not the Sprite's idle pause or transport limits.

A background child that inherits stdout or stderr keeps `run()` waiting until that child exits, because Sprites reports the exit status only after the output stream closes. Redirect background output to a file when starting a long-running job; `Shell.start_command` manages its own output log.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the Sprite without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `SPRITE_TOKEN` too. Pass `workspace='new'` to start a fresh Sprite even when the message history names one.

`runtime` only shapes a new Sprite, and an unknown one raises a clear error on first use; `working_dir` and `env` apply to every command, including after you reattach. A new Sprite gets `working_dir` created for it; on an attached or caller-supplied Sprite it must already exist, or commands fail with `WorkspaceError`.

Already have a `sprites.AsyncSprite`? Pass `workspace=SpritesSandboxBackend(sandbox=sprite)` to a run, with `SpritesSandboxBackend` from `pydantic_ai_harness.sprites_sandbox`. `SpritesSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

A run leaves a backend you built open. Call `await backend.aclose()` when you are done with it: that closes the client it opened from `SPRITE_TOKEN`, not the Sprite, and a later operation opens a new one. A `client=` or `sandbox=` you passed is never closed.

## Prepare a Sprite before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox, SpritesSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])


async def run_in_prepared_sprite() -> str:
    backend = SpritesSandboxBackend(working_dir='/home/sprite/project')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the Sprite exists
        if ref is not None:
            await SpritesSandbox().destroy(ref)
```

The run uses the backend as it is, so `SpritesSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the Sprite instead, store `backend.ref` and pass it as `workspace=` to reattach later.

## Clean up

The Sprite keeps its files and installed packages after the run ends. Pydantic AI never deletes it. It sleeps when idle and wakes on the next command: you pay for compute while it is active and for storage until you delete it. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox


async def delete_sprite(ref: WorkspaceRef) -> None:
    await SpritesSandbox().destroy(ref)
```

`destroy(ref)` deletes by Sprite id without attaching or waking it, and a Sprite that no longer exists returns quietly. Only destroy Sprites you own. Use `backend(ref)` to construct a lazy backend for an existing Sprite; `get_sandbox()` attaches to it and returns its native `sprites.AsyncSprite` (on a backend with no Sprite yet, it creates one). A backend you build yourself outside a run opens its own client, which you close with `aclose()`; after that, call `get_sandbox()` again for a working handle. See [Sprite lifecycle](https://docs.sprites.dev/concepts/lifecycle/).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails. If creation's reply was lost, the ref may identify a Sprite that is not yet visible to the API; a lookup failure does not prove it was never created:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await SpritesSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder(), hooks])
```

Unlike Modal and E2B sandboxes, a Sprite has no lifetime timeout: it persists until deleted. So a run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the Sprite behind. Its id is logged at INFO as `Created Sprite <id>` when it is created; delete it with `await SpritesSandbox().destroy(WorkspaceRef(provider='sprites', id=...))`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `sprites.AsyncSpritesClient` to share across runs on one event loop, or to set its base URL or timeout. Default: `None`, a client each run opens from `SPRITE_TOKEN` and closes when it ends. You close a client you pass; `SpritesSandbox` never does. |
| `runtime` | Runtime for a new Sprite. Default: `None`, Sprites' default. An unknown runtime fails on first use. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the Sprite's own (`/home/sprite`, where commands run as non-root `sprite`); set a project directory, or use relative paths for portable code. Created on a new Sprite; on an attached or caller-supplied Sprite it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the Sprite. |

Sprites runs on the asyncio event loop only: its SDK uses asyncio tasks, so under Trio the backend raises `UserError`.

## Durable execution

If the exec socket drops after connection but before an exit status arrives, the command may have run. This raises a non-retryable workspace error rather than replaying a potentially non-idempotent command. A handshake that fails before the socket opens cannot have started the command: the backend retries it three times over about five seconds, then lets it propagate as retryable.

A shared client is created at module import so activities on this worker reuse it. Close it when the worker stops. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import os
import uuid


from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

# The provider SDK must not be re-imported inside Temporal's restricted workflow sandbox.
with workflow.unsafe.imports_passed_through():
    from sprites import AsyncSpritesClient

CLIENT = AsyncSpritesClient(token=os.environ['SPRITE_TOKEN'])
agent = Agent(
    'anthropic:claude-opus-5-5',
    name='sprites_coder',
    capabilities=[SpritesSandbox(client=CLIENT), Coder(), TemporalDurability()],
)


@workflow.defn
class SandboxWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with CLIENT:
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

`Coder`, `Shell`, and `FileSystem` work under DBOS, Temporal and Prefect. See the [Coder](https://pydantic.dev/docs/ai/harness/coder/#durable-execution), [Shell](https://pydantic.dev/docs/ai/harness/shell/#durable-execution), and [FileSystem](https://pydantic.dev/docs/ai/harness/filesystem/#durable-execution) guides for engine-specific limits.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`SpritesSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the Sprite on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a Sprite logs its name at `INFO` (`Created Sprite <name>`) on the `pydantic_ai_harness.sprites_sandbox._backend` logger, so a Sprite that outlives its run can still be found.

## API reference

::: pydantic_ai_harness.sprites_sandbox.SpritesSandbox

::: pydantic_ai_harness.sprites_sandbox.SpritesSandboxBackend
