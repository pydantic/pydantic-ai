# Daytona Sandbox

Run your agent's commands and file edits in an isolated [Daytona](https://www.daytona.io) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/daytona_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[daytona,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[daytona,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `DAYTONA_API_KEY` to an API key from your Daytona dashboard, and `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It keeps running after the run ends until Daytona stops it for being idle, and keeps its disk, and costs money, until you delete it; see [Clean up](#clean-up).

Give the agent a project directory with `DaytonaSandbox(working_dir='/home/daytona/project')`: a new sandbox gets it created for you, and commands and relative paths start there.

Daytona's default snapshot comes with git, Python, and ripgrep (`rg`), which `Coder` uses for searches. For your project's dependencies, build a Daytona snapshot with them installed and pass `DaytonaSandbox(snapshot='my-snapshot')`. A snapshot Daytona doesn't know fails on first use with `WorkspaceUnavailableError`.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If Daytona stopped the sandbox while it was idle, the follow-up run starts it again. If the sandbox has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool that runs in the sandbox:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Commands run under `sh -c` as the non-root `daytona` user. File reads refuse FIFOs rather than waiting for a writer. File writes create missing parent directories, write through a symlink to its target, and keep an existing file's mode.

### What a timeout stops

A command timeout starts after the sandbox is ready. A timeout or a cancelled run stops the command, and the sandbox keeps running for other commands. `WorkspaceTimeoutError` carries the last 4 KiB of each stream's output so far. If Daytona does not confirm the stop, the command may still be running: inspect the sandbox or delete it yourself. `timeout=None` removes the command deadline, not Daytona's idle stop.

A command's output is collected once it exits, so `run()` returns then even if a background child it started is still running; output that child writes afterwards is not returned. A command whose combined output passes 10 MiB raises `WorkspaceOutputLimitError` once it exits, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large or background output to a file instead; `Shell.start_command` manages its own output log.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `DAYTONA_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`snapshot`, `auto_stop_interval`, and `allow_internet_access` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach. A new sandbox gets `working_dir` created for it; on an attached or caller-supplied sandbox it must already exist, or commands fail with `WorkspaceError`.

Already have a `daytona.AsyncSandbox`? Pass `workspace=DaytonaSandboxBackend(sandbox=sandbox)` to a run, with `DaytonaSandboxBackend` from `pydantic_ai_harness.daytona_sandbox`. `DaytonaSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

A run leaves a backend you built open. Call `await backend.aclose()` when you are done with it: that closes the client it opened from `DAYTONA_API_KEY`, not the sandbox, and a later operation opens a new one. A `client=` or `sandbox=` you passed is never closed.

## Prepare a sandbox before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])


async def run_in_prepared_sandbox() -> str:
    backend = DaytonaSandboxBackend(working_dir='/home/daytona/project')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the sandbox exists
        if ref is not None:
            await DaytonaSandbox().destroy(ref)
```

The run uses the backend as it is, so `DaytonaSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later.

## Clean up

Pydantic AI never stops or deletes the sandbox. Daytona stops a container sandbox, such as one from the default snapshot, after 15 idle minutes by default (set `auto_stop_interval=` to change that, or `0` to keep it running). A stopped sandbox keeps its disk and is archived after 7 days. A sandbox from a VM snapshot is paused after 60 idle minutes instead, and is never archived. Either way, a later run starts it again, and Daytona never deletes it. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await DaytonaSandbox().destroy(ref)
```

`destroy(ref)` deletes by sandbox ID without starting a stopped sandbox, and a sandbox that no longer exists returns quietly. Only destroy sandboxes you own. Use `backend(ref)` to construct a lazy backend for an existing sandbox; `get_sandbox()` attaches to it and returns its native `daytona.AsyncSandbox` (on a backend with no sandbox yet, it creates one).

A failed run returns no result, so there is no ref to store. To delete its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await DaytonaSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder(), hooks])
```

A run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the sandbox behind. Every sandbox the backend creates is labelled `created-by=pydantic-ai`, and its ID is logged at `INFO` as `Created Daytona sandbox <id>`; delete it with `await DaytonaSandbox().destroy(WorkspaceRef(provider='daytona', id=...))`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `daytona.AsyncDaytona` to share across runs on one event loop, or to configure its API URL or target. Default: `None`, a client each run opens from `DAYTONA_API_KEY` and closes when it ends. You close a client you pass; `DaytonaSandbox` never does. |
| `snapshot` | Daytona snapshot a new sandbox starts from. Default: `None`, Daytona's default. An unknown snapshot fails on first use. |
| `auto_stop_interval` | Idle minutes before Daytona stops a new sandbox; `0` disables it. Default: `None`, Daytona's default: 15 minutes for a container, while a VM pauses after 60 minutes instead. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. When `False`, what is blocked depends on your Daytona organization's tier: on Tier 3 and 4 everything, while lower tiers keep package registries, GitHub, and AI APIs reachable, so there it is not an exfiltration boundary. See [Daytona's network limits](https://www.daytona.io/docs/en/network-limits/). Default: `True`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the sandbox's own (`/home/daytona`, where commands run as non-root `daytona`); set a project directory, or use relative paths for portable code. Created on a new sandbox; on an attached or caller-supplied sandbox it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. |

Daytona runs on the asyncio event loop only: its SDK uses asyncio tasks, so under Trio the backend raises `UserError`.

## Durable execution

Each activity opens its own Daytona client and closes it once its operation ends, so the worker has nothing to close. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='daytona_coder',
    capabilities=[DaytonaSandbox(), Coder(), TemporalDurability()],
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

`Coder`, `Shell`, and `FileSystem` work under DBOS, Temporal and Prefect. See the [Coder](https://pydantic.dev/docs/ai/harness/coder/#durable-execution), [Shell](https://pydantic.dev/docs/ai/harness/shell/#durable-execution), and [FileSystem](https://pydantic.dev/docs/ai/harness/filesystem/#durable-execution) guides for engine-specific limits.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`DaytonaSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox logs its ID at `INFO` (`Created Daytona sandbox <id>`) on the `pydantic_ai_harness.daytona_sandbox._backend` logger, so a sandbox that outlives its run can still be found.

## API reference

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandbox

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend
