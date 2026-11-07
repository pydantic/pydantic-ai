---
title: AgentCore Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an Amazon Bedrock AgentCore Code Interpreter session instead of on your machine."
---

# AgentCore Sandbox

Run your agent's commands and file edits in an isolated [Amazon Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html) session in your AWS account instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/agentcore_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![The model calls Coder, Shell or FileSystem tools such as run_command and edit_file. AgentCoreSandbox's get_workspace hook gives the run an AgentCore Code Interpreter session as ctx.workspace, so each command and file operation goes over the AgentCore API, authenticated with your AWS credentials, and runs in the session's microVM; output comes back to the tool. The first use starts the session, its WorkspaceRef is saved in the message history and on the result, a later run reattaches with it, and destroy(ref) stops it. When session_timeout runs out, 15 minutes by default, the session ends with its files, and a run that names it raises WorkspaceUnavailableError.](../img/harness/agentcore-sandbox.svg)

## Install

```bash
pip/uv-add "pydantic-ai-harness[coder,agentcore,anthropic]"
```

The `agentcore` extra installs `boto3`. The `coder` extra is there because the examples use `Coder`; the sandbox itself doesn't need it. The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra, and set `ANTHROPIC_API_KEY` for the examples.

AWS credentials and the region come from boto3's usual configuration: environment variables, `~/.aws`, or an instance role. They need the `bedrock-agentcore:StartCodeInterpreterSession`, `bedrock-agentcore:InvokeCodeInterpreter` and `bedrock-agentcore:StopCodeInterpreterSession` permissions.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[AgentCoreSandbox(region_name='us-east-1'), Coder()])
result = agent.run_sync('Write a script that prints the first 20 prime numbers, run it, and show me the output.')
```

`Coder`'s shell and file tools now run in the session, not on your machine. With `Coder`, `RepoContext` starts the session when the run starts, even without a tool call; use `Coder(repo_context=False)` to start it on the first tool call instead. It keeps running, and billing, after the run ends; see [Clean up](#clean-up).

A session lives for 15 minutes (`session_timeout=900`), counted from its start, not from its last use. AgentCore allows up to `AgentCoreSandbox(session_timeout=28800)`, 8 hours. When it runs out, the session ends and its files go with it: it cannot be resumed.

Commands run as `sh -c` and start in the session's home directory, `/opt/amazon/genesis1p-tools/var`. Pass a project directory, such as `AgentCoreSandbox(working_dir='/tmp/project')`, to keep the project apart from it; it is created for you in a new session.

The AWS-managed code interpreter, `aws.codeinterpreter.v1`, has Python 3.12 with pip, Node.js and ripgrep, but not git. It runs in sandbox network mode, which reaches Amazon S3 but not the internet, so `pip install` from PyPI fails there. For internet access or your VPC, [create a code interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-resource-management.html) in public or VPC network mode and pass its id as `code_interpreter=`.

## Continue in the same session

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[AgentCoreSandbox(region_name='us-east-1'), Coder()])
result = agent.run_sync('Write a script that prints the first 20 prime numbers, run it, and show me the output.')

followup = agent.run_sync('Make the script take the count as an argument.', message_history=result.all_messages())
```

The follow-up run finds the session in the message history and works in it, so the script is still there. Without the history, a run starts a new session. If the session has ended, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('{{ anthropic:large }}', capabilities=[AgentCoreSandbox(region_name='us-east-1'), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the session."""
    result = await ctx.workspace.run(['python3', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

### How commands and files run

AgentCore runs a command on a pseudo-terminal, which merges stdout and stderr. So each command runs as a background task with stdin at EOF and its output redirected to files, which are read when it exits. On a timeout or cancellation, the backend stops the task, which kills its process, and the timeout error carries up to the last 4 KiB of each stream. A command whose combined output passes 4 MiB raises `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`; redirect large output to a file instead. A command that replaces one of its output files with something other than a regular file, such as a FIFO, fails with `WorkspaceError` and leaves the session usable.

File operations run as short Python programs in the session's code kernel, which runs one at a time. They follow symlinks, and reading a FIFO or another file that is not a regular file raises `OSError` instead of waiting. AgentCore returns large output slowly, so a file is read 1 MiB per request: writing an 8 MiB file and reading it back takes about a minute.

See [Workspaces](../workspace.md) for more.

## Reattach later

To come back to the session without the message history, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[AgentCoreSandbox(region_name='us-east-1'), Coder()])

result = agent.run_sync('Write a script that prints the first 20 prime numbers, run it, and show me the output.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Make the script take the count as an argument.', workspace=ref)
```

The ref names the code interpreter and the session, but not the region or any credentials, so the process that reattaches needs the same region and credentials that can reach the session. Pass `workspace='new'` to start a fresh session even when the message history names one.

`code_interpreter` and `session_timeout` only shape a new session; `working_dir` and `env` apply to every command.

## Prepare a session before the run

To seed files before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox, AgentCoreSandboxBackend
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', capabilities=[AgentCoreSandbox(region_name='us-east-1'), Coder()])


async def main() -> None:
    backend = AgentCoreSandboxBackend(region_name='us-east-1', working_dir='/tmp/project')
    try:
        workspace = Workspace(backend)
        await workspace.write_text('calc.py', 'def add(a, b):\n    return a - b\n')
        result = await agent.run('Fix the bug in calc.py and show that it works.', workspace=backend)
        print(result.output)
    finally:
        ref = backend.ref  # set once the session exists
        if ref is not None:
            await AgentCoreSandbox(region_name='us-east-1').destroy(ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` stops the session even when setup or the run fails. To keep it instead, store `backend.ref` and pass it as `workspace=` to reattach later, before `session_timeout` runs out.

## Clean up

The session keeps running, and billing, after the run ends, until its `session_timeout` runs out. Pydantic AI does not stop it. If a run is cancelled while AgentCore starts the session, the ref is still recorded once AgentCore answers; the id is also logged (see [Telemetry](#telemetry)). Stop a session early with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox


async def stop_session(ref: WorkspaceRef) -> None:
    await AgentCoreSandbox(region_name='us-east-1').destroy(ref)
```

`AgentCoreSandbox.backend(ref)` constructs an attached backend without I/O; `destroy(ref)` stops the session by its id without attaching. Stopping a session that has already ended returns quietly.

A failed run returns no result, so there is no ref to store. To stop its session, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

import anyio

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox
from pydantic_ai_harness.coder import Coder

sandbox = AgentCoreSandbox(region_name='us-east-1')
hooks = Hooks()


@hooks.on.run_error
async def stop_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        # Shielded, so an `anyio` deadline around the run doesn't cut the stop short,
        # and bounded, so a hung API call can't hang the run indefinitely.
        with anyio.move_on_after(30, shield=True):
            await sandbox.destroy(ctx.workspace.ref)
    raise error


agent = Agent('{{ anthropic:large }}', capabilities=[sandbox, Coder(), hooks])
```

## Configuration

| Option | What it does |
| --- | --- |
| `region_name` | AWS region of the code interpreter. Default: `None`, boto3's configured region. A ref does not record its region, so reattach with the region the session was started in. |
| `code_interpreter` | The code interpreter a new session runs in: `aws.codeinterpreter.v1`, or the id of one you created, for example with public or VPC network mode. Default: `'aws.codeinterpreter.v1'`. An unknown one fails on first use. |
| `session_timeout` | Seconds a new session lives, counted from its start. Default: `900` (15 minutes); at most `28800` (8 hours). |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the session's own (`/opt/amazon/genesis1p-tools/var` on `aws.codeinterpreter.v1`). Created in a new session; in an attached session it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine's environment reaches the session. The agent's commands can read these values, so leave out secrets the agent should not see. |

`AgentCoreSandbox` works under asyncio and Trio: boto3 calls run in worker threads.

## Durable execution

Install the `temporal` extra too:

```bash
pip/uv-add "pydantic-ai-harness[coder,agentcore,anthropic,temporal]"
```

Run a Temporal dev server on `localhost:7233` first. The agent and workflow must be defined at module level for activity registration.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox
from pydantic_ai_harness.coder import Coder
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    '{{ anthropic:large }}',
    name='agentcore_coder',
    capabilities=[AgentCoreSandbox(region_name='us-east-1'), Coder(), TemporalDurability()],
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
                SandboxWorkflow.run, 'Use the shell tool to run pwd.',
                id=f'sandbox-{uuid.uuid4()}', task_queue='sandbox',
            )
        )


if __name__ == '__main__':
    asyncio.run(main())
```

A workflow that runs longer than `session_timeout` loses its session: keep the timeout above the longest run you expect.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`AgentCoreSandbox` emits no spans of its own. Core's [instrumentation](../capabilities/instrumentation.md) records the session on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Starting a session also logs `Started AgentCore Code Interpreter session <id>` at INFO on the `pydantic_ai_harness.agentcore_sandbox._backend` logger, so the id is on record even if the run ends before it is stored.

## API reference

::: pydantic_ai_harness.agentcore_sandbox.AgentCoreSandbox

::: pydantic_ai_harness.agentcore_sandbox.AgentCoreSandboxBackend
