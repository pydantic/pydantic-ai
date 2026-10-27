---
title: Vercel Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated Vercel Sandbox microVM instead of on your machine."
---

# Vercel Sandbox

Run your agent's commands and file edits in an isolated [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) microVM instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/vercel_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[vercel,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then give the Vercel SDK credentials for a Vercel project: run `vercel link` and `vercel env pull` to get a `VERCEL_OIDC_TOKEN`, or set an access token as `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`. See [Vercel's authentication guide](https://vercel.com/docs/sandbox/concepts/authentication). Set `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. The sandbox keeps running after the run ends until its session runs out, then Vercel saves its files and keeps it, and its snapshot storage costs money, until you delete it; see [Clean up](#clean-up).

Give the agent a project directory with `VercelSandbox(working_dir='/vercel/project')`: a new sandbox gets it created for you, and commands and relative paths start there.

The default image, `vercel/sandbox/universal`, comes with git, Python, Node.js, and ripgrep (`rg`), which `Coder` uses for searches. For your project's dependencies, build an image with them installed, push it to [Vercel Container Registry](https://vercel.com/docs/sandbox/concepts/images#custom-images), and pass `VercelSandbox(image='my-repository:latest')`. An image Vercel doesn't know fails on first use with `WorkspaceUnavailableError`.

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

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If the sandbox's session ended in between, its first command resumes it from the saved files in a new session; processes that were running, such as a dev server, are not restarted. If the sandbox has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

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

Commands run under `sh -c` as the image's default user: on the default image that is `ubuntu`, which can use `sudo`. Vercel gives commands every Linux capability, so file permissions do not restrict them, even as `ubuntu`. File operations run as commands in the sandbox too, so they see the same files and permissions; a file moves in 64 KiB pieces, one command each, so reading or writing many megabytes through the workspace is slow. For bulk transfers, use the SDK's filesystem API on the sandbox handle from `get_sandbox()`.

### What a timeout stops

A command timeout starts after the sandbox is ready. A timeout or a cancelled run kills the command's process, and the sandbox keeps running for other commands; Vercel also kills it 10 seconds past the timeout, in case the client's kill does not arrive. `WorkspaceTimeoutError` carries the output the command printed. A background job it started can keep running. `timeout=None` removes the command deadline, not the session's `sandbox_timeout`.

Vercel reports a command's exit only once every process holding its output has closed it, so a background job that inherits stdout or stderr keeps `run()` waiting until that job ends. Redirect a background job's output, as in `npm run dev > dev.log 2>&1 &`; `Shell.start_command` manages its own output log. A command whose combined output passes 10 MiB is killed and raises `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`; redirect large output to a file and read part of it instead.

A command still running when its session ends, because `sandbox_timeout` ran out or someone stopped the sandbox, fails with `WorkspaceError`, and the next command resumes the sandbox. One running when the sandbox is deleted raises `WorkspaceUnavailableError`.

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

The ref's `id` is the sandbox's name. It holds no credentials, so the process that reattaches needs credentials for the same Vercel project too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`image`, `sandbox_timeout`, and `allow_internet_access` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach. A new sandbox gets `working_dir` created for it; on an attached sandbox it must already exist, or commands fail with `WorkspaceError`.

Already have a sandbox, such as one you created from a Git repository or a snapshot with the Vercel SDK? Pass its name: `workspace=WorkspaceRef(provider='vercel', id=sandbox.name)`.

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
    backend = VercelSandboxBackend(working_dir='/vercel/project')
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

The run uses the backend as it is, so `VercelSandbox`'s settings don't apply to it, and leaves it open: close it with `aclose()`. To keep the sandbox instead, store `backend.ref` and pass it as `workspace=` to reattach later.

## Clean up

Pydantic AI never stops or deletes the sandbox. Vercel stops a session when its `sandbox_timeout` runs out and saves the sandbox's files as a snapshot, which a later run resumes from. Snapshots are billed as storage and, by default, expire 30 days after their last use; Vercel removes a sandbox that can no longer resume after 14 days without activity. See [Vercel's persistence guide](https://vercel.com/docs/sandbox/concepts/persistent-sandboxes). Delete the sandbox with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.vercel_sandbox import VercelSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await VercelSandbox().destroy(ref)
```

`destroy(ref)` deletes the sandbox by name without resuming it, together with its snapshots that no other sandbox uses, and returns quietly for a sandbox that no longer exists. Only destroy sandboxes you own. To end a session early and keep the sandbox, stop it through the SDK: `await (await backend.get_sandbox()).stop()`. Use `VercelSandbox().backend(ref)` to construct a lazy backend for an existing sandbox; `get_sandbox()` attaches to it and returns its native `vercel.sandbox.Sandbox` (on a backend with no sandbox yet, it creates one).

A failed run returns no result, so there is no ref to store. To delete its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await VercelSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[VercelSandbox(), Coder(), hooks])
```

A run that ends before its ref is stored, such as a crash, Ctrl-C, or a killed worker right after creation, leaves the sandbox behind. Every sandbox the backend creates is named `pydantic-ai-<hex>`, and its name is logged at `INFO` as `Created Vercel sandbox <name>`; delete it with `await VercelSandbox().destroy(WorkspaceRef(provider='vercel', id=...))`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `vercel.sandbox.SandboxClient` to share across runs on one event loop, or to set its region or credentials. Default: `None`, a client each run opens, reading its credentials from the environment, and closes when it ends. You close a client you pass; `VercelSandbox` never does. |
| `image` | Image a new sandbox runs. Default: `None`, Vercel's default, `vercel/sandbox/universal`. An unknown image fails on first use. |
| `sandbox_timeout` | Seconds each session of a new sandbox runs before Vercel stops it and saves its files; the next command resumes it. Default: `2700` (45 minutes), the most Vercel's Hobby plan allows; Pro and Enterprise allow up to `86400`. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. When `False`, it gets Vercel's `deny-all` network policy. For an allow-list of domains, create the sandbox with the SDK's `NetworkPolicy.custom(...)` and attach to it by name. Default: `True`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the image's own; set a project directory, or use relative paths for portable code. Created on a new sandbox; on an attached sandbox it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the sandbox. |

Vercel Sandbox runs on the asyncio event loop only: the backend creates a sandbox in an asyncio task so a cancelled run still records it, so under Trio it raises `UserError`.

## Durable execution

Each activity opens its own Vercel client and closes it once its operation ends, so the worker has nothing to close. Run a Temporal dev server on `localhost:7233` first.

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

agent = Agent(
    '{{ anthropic:large }}',
    name='vercel_coder',
    capabilities=[VercelSandbox(), Coder(), TemporalDurability()],
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

`VercelSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox logs its name at `INFO` (`Created Vercel sandbox <name>`) on the `pydantic_ai_harness.vercel_sandbox._backend` logger, so a sandbox that outlives its run can still be found.

## API reference

::: pydantic_ai_harness.vercel_sandbox.VercelSandbox

::: pydantic_ai_harness.vercel_sandbox.VercelSandboxBackend
