---
title: Vercel Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated Vercel Sandbox instead of on your machine."
---

# Vercel Sandbox

Run your agent's commands and file edits in an isolated [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/vercel_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![The model calls Coder, Shell or FileSystem tools such as run_command and edit_file. VercelSandbox's get_workspace hook gives the run a Vercel sandbox as ctx.workspace, so each command and file operation goes over the Vercel API, authenticated with VERCEL_OIDC_TOKEN or VERCEL_TOKEN, and runs in the sandbox's Firecracker microVM; output comes back to the tool. The first use creates the sandbox, its WorkspaceRef is saved in the message history and on the result, a later run reattaches with it and resumes a stopped persistent sandbox, and destroy(ref) destroys it.](../img/harness/vercel-sandbox.svg)

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,vercel,anthropic]"
```

The `coder` extra is there because the examples use `Coder`; the sandbox itself doesn't need it. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then give the SDK Vercel credentials: inside a Vercel project, `vercel env pull` writes a `VERCEL_OIDC_TOKEN`; elsewhere, set `VERCEL_TOKEN` to an access token along with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`. Set `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. The sandbox's session keeps running after the run ends, until `execution_time_limit` (45 minutes by default); see [Clean up](#clean-up).

Give the agent a project directory with `VercelSandbox(working_dir='/vercel/sandbox/project')`: a new sandbox gets it created for you, and commands and relative paths start there.

Vercel's default image, `vercel/sandbox/universal`, has Node.js, Python 3, and common command-line tools. Pass `VercelSandbox(image=...)` to boot another one; an image Vercel doesn't know fails on first use with `WorkspaceUnavailableError`. `VercelSandbox(vcpus=4)` picks a larger sandbox, with 2 GB of memory per vCPU; Vercel accepts 1 or an even number up to your plan's limit.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If the sandbox's session has ended, the follow-up run resumes it from the snapshot Vercel saved, with its files but not its running processes. If the sandbox has been destroyed, or was created with `persistent=False` and its session has ended, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool that runs in the sandbox:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python3', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Commands run under `sh -c` in the sandbox. File operations run as shell commands there too, through [`Workspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/), so they see the same files and permissions as commands, and a large file moves in pieces of at most 64 KiB.

### What a timeout stops

A command timeout starts after the sandbox is ready. A timeout or a cancelled run kills the command, and the sandbox keeps running for other commands. `WorkspaceTimeoutError` carries the output the command printed so far. `timeout=None` doesn't lift Vercel's own limit: when the session reaches `execution_time_limit`, Vercel stops it and any command still running ends with `WorkspaceUnavailableError`. The next operation resumes a persistent sandbox.

A command whose combined output passes 10 MiB is killed and raises `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large or background output to a file instead; `Shell.start_command` manages its own output log.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds the sandbox's name and no credentials, so the process that reattaches needs credentials for the same Vercel project. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`image`, `vcpus`, `persistent`, and `execution_time_limit` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach. A new sandbox gets `working_dir` created for it; on an attached sandbox it must already exist, or commands fail with `WorkspaceError`.

A run leaves a backend you built open. Call `await backend.aclose()` when you are done with it: that closes the client it opened from the environment, not the sandbox, and a later operation opens a new one. A `client=` you passed is never closed.

## Prepare a sandbox before the run

To seed files before the agent starts, build the backend yourself, write through a `Workspace`, and pass the backend to the run:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder()])


async def run_in_prepared_sandbox() -> str:
    backend = VercelSandboxBackend(working_dir='/vercel/sandbox/project')
    try:
        await Workspace(backend).write_text('TASK.md', 'Add a `--version` flag to cli.py.\n')
        result = await agent.run('Do the task in TASK.md.', workspace=backend)
        return result.output
    finally:
        await backend.aclose()
        ref = backend.ref  # set once the sandbox exists
        if ref is not None:
            await VercelSandbox().destroy(ref)
```

The run uses the backend as it is, so `VercelSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later. For Vercel features the workspace API doesn't cover, such as ports, snapshots, and network policies, `await backend.get_sandbox()` returns the sandbox's `vercel.sandbox.Sandbox`. The handle works through the backend's client, so use it before `aclose()`. For a sandbox the run created, such as `result.workspace`, open your own `SandboxClient` and call `get_sandbox(name=ref.id)` with its ref.

## Clean up

Pydantic AI never stops or destroys the sandbox. Vercel stops its session at `execution_time_limit`; a persistent sandbox is snapshotted then and kept until you destroy it, or until it has been inactive for as long as Vercel keeps idle sandboxes. Destroy it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.vercel_sandbox import VercelSandbox


async def destroy_sandbox(ref: WorkspaceRef) -> None:
    await VercelSandbox().destroy(ref)
```

`destroy(ref)` destroys the sandbox whatever state it is in, keeping its snapshots until they expire; a sandbox that no longer exists returns quietly. Use `backend(ref)` to construct a lazy backend for an existing sandbox.

A failed run returns no result, so there is no ref to store. To destroy its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

hooks = Hooks()


@hooks.on.run_error
async def destroy_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await VercelSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder(), hooks])
```

A run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the sandbox behind. Sandboxes the backend creates are named `pydantic-ai-<hex>` and logged at `INFO` (`Creating Vercel sandbox <name>`); find leftovers in your project's Sandboxes dashboard and destroy one with `await VercelSandbox().destroy(WorkspaceRef(provider='vercel', id=...))`.

## Security model

Each sandbox is its own Firecracker microVM. Code the agent runs gets no Vercel credentials: nothing from your machine's environment reaches the sandbox except the `env` you pass. It can reach the internet.

A ref names a sandbox by its name in your Vercel project, and attaching or `destroy` acts on any sandbox those credentials can see in that project, so keep refs server-side.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `vercel.sandbox.SandboxClient` to share across runs on one event loop. Default: `None`, a client each run opens and closes when it ends, from `VERCEL_OIDC_TOKEN`, else `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`. You close a client you pass; `VercelSandbox` never does. |
| `image` | Image a new sandbox boots from. Default: `None`, Vercel's `vercel/sandbox/universal`. An unknown image fails on first use. |
| `vcpus` | vCPUs of a new sandbox, with 2 GB of memory each: 1 or an even number up to your plan's limit. Default: `None`, Vercel's default (2). |
| `persistent` | Whether a new sandbox is snapshotted when its session ends, so a later run resumes it with its files. Default: `True`. With `False`, the sandbox is gone once its session ends. |
| `execution_time_limit` | Seconds a new sandbox's session runs before Vercel stops it, ending any command still running. Default: `2700` (45 minutes), the most Vercel's Hobby plan allows; Pro and Enterprise plans allow up to `86400`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the sandbox's own (`/vercel/sandbox`); set a project directory, or use relative paths for portable code. Created on a new sandbox; on an attached sandbox it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. |

The backend runs on the asyncio event loop only; under Trio it raises `UserError`.

## Durable execution

Pass one `client=`, as below. Without it, every workspace operation opens a new client and a new connection to Vercel; with one client, every activity on the worker's event loop shares its connections. Run a Temporal dev server on `localhost:7233` first.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from vercel.sandbox import SandboxClient

agent = Agent(
    '{{ anthropic:large }}',
    name='vercel_coder',
    capabilities=[VercelSandbox(client=SandboxClient.create()), Coder(), TemporalDurability()],
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

`VercelSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox logs its name at `INFO` (`Creating Vercel sandbox <name>`) on the `pydantic_ai_harness.vercel_sandbox._backend` logger.

## API reference

::: pydantic_ai_harness.vercel_sandbox.VercelSandbox

::: pydantic_ai_harness.vercel_sandbox.VercelSandboxBackend
