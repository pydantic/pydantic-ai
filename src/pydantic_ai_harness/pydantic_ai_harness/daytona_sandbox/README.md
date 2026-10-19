# Daytona Sandbox

Run your agent's commands and file edits in an isolated [Daytona](https://www.daytona.io) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/daytona_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[daytona]"
```

pip:

```bash
pip install "pydantic-ai-harness[daytona]"
```

Then set `DAYTONA_API_KEY` to an API key from your Daytona dashboard.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It keeps running, and billing, after the run ends; see [Clean up](#clean-up).

Daytona's default snapshot includes Python and `git`. `Coder` searches with ripgrep (`rg`) when the sandbox has it and with its built-in search otherwise; installing it in a snapshot of your own (`snapshot=`) makes searches faster. A snapshot Daytona doesn't know fails on first use with a clear error.

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

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If Daytona stopped the sandbox while it was idle, attaching in a new run starts it again. A held backend also retries command setup once after an idle stop. If the sandbox has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool:

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

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, keep the run's workspace ref (in your database, say) and pass it back as `workspace=`:

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

`snapshot`, `auto_stop_interval`, and `network_block_all` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach.

Already have a `daytona.AsyncSandbox`? Pass `workspace=DaytonaSandboxBackend(workspace=sandbox)` to a run, with `DaytonaSandboxBackend` from `pydantic_ai_harness.daytona_sandbox`. `DaytonaSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never stops or deletes it. Sandboxes created by this backend carry the `created-by=pydantic-ai` label; filter by that label when auditing your Daytona account. Delete one with the ref you kept:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await DaytonaSandbox().destroy(ref)
```

`destroy(ref)` deletes by ref without starting a stopped sandbox. If you pass `client=`, you own and must close that client, including after a cancelled run; otherwise the capability closes the client it opened when the run ends.

Daytona SDK 0.198.0 can leave one HTTP connection open after its event socket reconnects, which Python reports as an `Unclosed connector` `ResourceWarning`. The harness cannot close it; it is a Daytona SDK issue.

By default, Daytona stops a sandbox after 15 idle minutes (a later run starts it again); set `auto_stop_interval=` to a smaller number of minutes to stop it sooner. A stopped sandbox keeps its disk, is archived after 7 days, and is never deleted. See [Daytona's SDK docs](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await delete_sandbox(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder(), hooks])
```

## What a timeout stops

A `run(timeout=...)` deadline starts after sandbox acquisition. A FIFO read fails rather than waiting for a writer. Daytona deletes the command session to stop its foreground work, leaving the sandbox available for other commands; a failed deletion is retried for up to two seconds during timeout or cancellation cleanup. `WorkspaceTimeoutError` carries partial stdout and stderr. Cancellation also attempts to stop the session. If Daytona cannot confirm deletion, the command may still be running: inspect or delete the sandbox yourself. `timeout=None` removes the command deadline, not Daytona's idle auto-stop or transport limits.

A background child that inherits stdout or stderr keeps `run()` waiting until that child exits, because Daytona keeps the session command open while its output is held. Redirect background output to a file when starting a long-running job; `Shell.start_command` manages its own output log.

Daytona stages `write_bytes` uploads beside the resolved target before replacing it, preserving an existing file's mode and symlinks; interrupted uploads can leave a temporary sibling after a host kill.

## Configuration

| Option | What it does |
| --- | --- |
| `snapshot` | Daytona snapshot a new sandbox starts from. Default: Daytona's. |
| `auto_stop_interval` | Idle minutes before Daytona stops a new sandbox; `0` disables it. Default: Daytona's, 15 minutes. |
| `network_block_all` | Block DNS and non-allowlisted outbound traffic from a new sandbox. Daytona still allows package registries, GitHub, and AI APIs, so this is not an exfiltration boundary. See [Daytona's network limits](https://www.daytona.io/docs/en/network-limits/). Default: `False`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `/home/daytona` for the non-root `daytona` user in Daytona's default image. Prefer relative paths or set `working_dir=` for portable code. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the sandbox. |
| `client` | An `AsyncDaytona` client to share across runs. You close it; `DaytonaSandbox` never does. |

## Durable execution

A shared client is created at module import so activities on this worker reuse it. Close it when the worker stops. Run a Temporal dev server on `localhost:7233` first.

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

# The provider SDK must not be re-imported inside Temporal's restricted workflow sandbox.
with workflow.unsafe.imports_passed_through():
    from daytona import AsyncDaytona

CLIENT = AsyncDaytona()
agent = Agent(
    'anthropic:claude-opus-5-5',
    name='daytona_coder',
    capabilities=[DaytonaSandbox(client=CLIENT), Coder(), TemporalDurability()],
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


## API reference

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandbox

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend
