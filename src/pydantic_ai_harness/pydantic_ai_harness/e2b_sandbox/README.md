# E2B Sandbox

Run your agent's commands and file edits in an isolated [E2B](https://e2b.dev) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/e2b_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[e2b,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[e2b,anthropic]"
```

The `anthropic` extra is there because the examples use an Anthropic model; swap it for your model provider's extra. Then set `E2B_API_KEY` to your E2B API key, and `ANTHROPIC_API_KEY` for the examples.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call; use `Coder(repo_context=False)` for lazy creation. It keeps running, and billing, after the run ends; see [Clean up](#clean-up).

A sandbox lives for 1 hour by default. When that runs out it pauses, and the next run resumes it. On E2B's Pro plan you can pass up to `E2BSandbox(sandbox_timeout=86400)`. A plan-limit hint is added only when E2B rejects the requested lifetime; other create errors, including network timeouts, retain the upstream error.

Commands start in the home directory, `/home/user`, which holds dotfiles and the pip cache. Pass a project directory, such as `E2BSandbox(working_dir='/home/user/project')`, so the project is not the home directory; it is created for you on a new sandbox.

E2B's default template does not include ripgrep (`rg`). `Coder` can search without it, but installing `rg` makes searches faster. Build a reusable [template](https://e2b.dev/docs/template/quickstart) once, outside an agent run (building can take a minute):

```text
from e2b import AsyncTemplate, Template
await AsyncTemplate.build(Template().from_base_image().apt_install(['ripgrep']), 'my-rg-template')
agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(template='my-rg-template'), Coder()])
```

The build snippet is illustrative and not part of the runnable agent examples below.

The default template has Python and git but not your project's dependencies, not even pytest. Install them in a custom template, or with `await workspace.run(['pip', 'install', ...])` before the run, as in [Prepare a sandbox before the run](#prepare-a-sandbox-before-the-run).

| Default template | Value |
| --- | --- |
| User | `user` |
| `HOME` and default working directory | `/home/user` |
| `rg` | Not installed |
| `pytest` | Not installed |

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. A paused sandbox resumes. If the sandbox has been killed, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Shell commands run as the `user` account, and files the backend creates are owned by it. E2B's file service still reads and writes with its own root privileges, so Unix permissions do not restrict file operations; do not rely on them to keep file tools out of a path.

E2B file reads reject FIFOs rather than waiting for a writer. An ordinary read performs a shell FIFO check before downloading the file. E2B's file API omits looping symlinks from directory listings.

A background child that inherits stdout or stderr can keep `run()` waiting for the SDK stream to close after the main command exits. Redirect background output to a file when starting a long-running job; `Shell.start_command` manages its own output log.

### What a timeout stops

On a command deadline or cancellation, the backend sends TERM then KILL to the command's foreground process group, including children that have not detached into a separate session. Custom templates and attached sandboxes are probed once for `setsid` (util-linux); without it, only the command leader is stopped, so child processes may continue. Custom templates also need `/bin/bash` and `/bin/sh`. Intentionally detached background jobs are not stopped. It does not destroy the shared sandbox. Stop is best effort: if E2B cannot be reached, or a start won registration but does not respond to the stop request, work may still be running. A cancelled start that arrives after stop is fenced before user code runs. Retain the sandbox ref to inspect or kill it explicitly.

A command whose combined output passes 10 MiB is stopped the same way and raises `WorkspaceOutputLimitError`, with the first 65,536 characters of each stream in `stdout` and `stderr`. Redirect large output to a file instead.

A command killed by a signal may return `exit_code=-1` in E2B; the SDK does not identify the signal, so this is not converted to `128+signal`.

Concurrent writes to the same path are not atomic on E2B: uploads may interleave, and a reader can see a partial file. Coordinate writers or write to separate paths when using `FileSystem` or `Coder`.

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `E2B_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`template` and `allow_internet_access` only shape a new sandbox. `sandbox_timeout` also applies when you reattach, and `working_dir` and `env` apply to every command.

Already have an `e2b.AsyncSandbox`? Pass `workspace=E2BSandboxBackend(sandbox=sandbox)` to a run, with `E2BSandboxBackend` from `pydantic_ai_harness.e2b_sandbox`. `E2BSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])


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

The `finally` kills the sandbox even when setup or the run fails. To keep it instead, store `backend.ref` and pass it as `workspace=` to reattach later.

## Preview a dev server

With `Shell`, ask the agent to use `start_command` for `npm run dev -- --host 0.0.0.0 --port 3000`, then poll `check_command` and `curl http://localhost:3000/health` until ready. Save the returned command ID. Given the workspace ref, connect with `sandbox = await e2b.AsyncSandbox.connect(ref.id)` and use `sandbox.get_host(3000)` for the public hostname (prefix with `https://` for the preview URL). When done, call `stop_command` with the ID while the workspace is attached, then delete the sandbox with `await E2BSandbox().destroy(ref)`. Do not leave a public preview running longer than necessary.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never kills it. If acquisition is cancelled while creation is in flight, the backend finishes recording the ref when E2B responds; a lost response may still leave a sandbox without a ref. Kill it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.e2b_sandbox import E2BSandbox


async def kill_sandbox(ref: WorkspaceRef) -> None:
    await E2BSandbox().destroy(ref)
```

`E2BSandbox.backend(ref)` constructs an attached backend without I/O; `destroy(ref)` kills by ID without attaching. This kills a paused sandbox too, without resuming it. A sandbox you don't kill is paused when its `sandbox_timeout` runs out. See [E2B's sandbox lifecycle](https://docs.e2b.dev/sandbox).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await E2BSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder(), hooks])
```

## Configuration

| Option | What it does |
| --- | --- |
| `template` | E2B template name or ID for a new sandbox. Default: `None`, E2B's `base`. An unknown template fails on first use. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `sandbox_timeout` | Seconds the sandbox lives before E2B pauses it, set on create and on reattach. Default: `3_600` (1 hour, the most E2B's Hobby plan allows). |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, the sandbox's own (`/home/user` on the default template, where commands run as `user`); prefer relative paths or set `working_dir` for portable code. Created on a new sandbox; on an attached or caller-supplied sandbox it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Output is decoded as UTF-8 either way; the default template's locale is POSIX, so pass `env={'LC_ALL': 'C.UTF-8'}` if a tool such as `wc -m` should count characters rather than bytes. Nothing from your machine's environment reaches the sandbox. |

Commands run on the asyncio event loop only: the E2B SDK reads their output with asyncio tasks, so under Trio `run` raises `UserError`.

## Durable execution

Install the `temporal` extra too:

uv:

```bash
uv add "pydantic-ai-harness[e2b,anthropic,temporal]"
```

pip:

```bash
pip install "pydantic-ai-harness[e2b,anthropic,temporal]"
```

Run a Temporal dev server on `localhost:7233` first. The agent and workflow must be defined at module level for activity registration.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='e2b_coder',
    capabilities=[E2BSandbox(), Coder(), TemporalDurability()],
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

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## Telemetry

`E2BSandbox` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox also logs `Created E2B sandbox <id>` at INFO on the `pydantic_ai_harness.e2b_sandbox._backend` logger, so the id is on record even if the run ends before it is stored.

## API reference

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend
