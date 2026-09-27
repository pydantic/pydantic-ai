# Modal Sandbox

Run your agent's commands and file edits in an isolated [Modal](https://modal.com) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/modal_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[modal]"
uv run modal token new
```

pip:

```bash
pip install "pydantic-ai-harness[modal]"
modal token new
```

This saves your Modal credentials; in CI, set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` instead.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. With `Coder`, `RepoContext` creates the sandbox when the run starts, even without a tool call. Use `Coder(repo_context=False)` for lazy creation. It keeps running, and billing, after the run ends; see [Clean up](#clean-up).

A new sandbox lives for up to 24 hours, Modal's maximum; pass `ModalSandbox(sandbox_timeout=3600)` to end it sooner. Commands start in the image user's home directory (`/root` on the default image). A first use may take several minutes while Modal builds or pulls an image. If Modal cannot start the sandbox, for example because the image does not exist, the first tool call raises an error that says why.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If the sandbox has been terminated or has expired, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool that runs in the sandbox:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.modal_sandbox import ModalSandbox
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

Cancellation and command deadlines attempt to stop the foreground command without ending the sandbox; see [What a timeout stops](#what-a-timeout-stops) for the best-effort caveat. `Shell` sets a 30-second timeout; in your own tools, pass a `timeout`, as above.

If only your own tools use the sandbox, pass `ModalSandbox(warn_if_no_tools=False)` to silence the missing-tools warning.

## Reattach later

To come back to the sandbox without the message history, store its ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs your Modal credentials too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`image`, `app_name`, `create_app_if_missing`, `sandbox_timeout`, and `idle_timeout` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach.

Already have a `modal.Sandbox`? Pass `workspace=ModalSandboxBackend(sandbox=sandbox)` to a run, with `ModalSandboxBackend` from `pydantic_ai_harness.modal_sandbox`. `ModalSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never terminates it. Terminate it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.modal_sandbox import ModalSandbox


async def terminate_sandbox(ref: WorkspaceRef) -> None:
    await ModalSandbox().destroy(ref)
```

`ModalSandbox().backend(ref)` constructs a backend for an existing ref without I/O. `destroy(ref)` uses the sandbox ID directly; it does not resume an expired sandbox or run its tools. Only destroy sandboxes you own.

### What a timeout stops

`run(timeout=...)` starts its clock after sandbox acquisition and includes command start, execution, and output collection. On its deadline or cancellation, the backend attempts to stop the command's foreground process group, not the shared sandbox or detached background jobs. Partial stdout and stderr are available on `WorkspaceTimeoutError`; if the stop RPC fails, the command may still run, so retain the sandbox ref for explicit cleanup. `timeout=None` has no command deadline; the sandbox's own lifetime and idle settings still apply. Custom or attached images without `setsid -w` still run commands, but cancellation can signal only the wrapper process, not its descendants. Install util-linux (`setsid`) in the image for process-group stopping.

File reads refuse FIFOs rather than waiting indefinitely for a writer. Writes through symlinks update the target, including when the target was created by a shell command.

`get_sandbox()` returns the `modal.Sandbox`. A sandbox you don't terminate ends when its `sandbox_timeout` runs out, or after `idle_timeout` seconds without activity if you set one. See [Modal's timeouts](https://modal.com/docs/guide/sandbox#timeouts).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await terminate_sandbox(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Coder(), hooks])
```

## Configuration

| Option | What it does |
| --- | --- |
| `image` | Image for a new sandbox: a registry tag or a `modal.Image`. Default: Debian slim with Python 3.12, `git`, and `ripgrep`; the first sandbox in a Modal workspace takes a few extra seconds while it builds. |
| `app_name` | Modal app a new sandbox belongs to. Default: `'pydantic-ai-harness'`. |
| `create_app_if_missing` | Create that app if it doesn't exist. Default: `True`. |
| `sandbox_timeout` | Seconds a new sandbox lives before Modal stops it (10-86,400). Default: `86_400` (24 hours, Modal's maximum). |
| `idle_timeout` | Seconds without activity before Modal stops a new sandbox. Default: `None`, no idle limit. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: the image user's home directory (`/root` on the default Debian slim image), or the image's working directory if it has none. |
| `defer_loading` | `defer_loading=True` is unsupported: workspace selection happens at run setup. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the sandbox. |
| `warn_if_no_tools` | Warn when the agent has no `Shell` or `FileSystem` tool. Default: `True`. |

## Upgrading from the previous `ModalSandbox`

The previous `ModalSandbox` registered its own `run_command`, `read_file`, `write_file`, and `list_directory` tools and terminated its sandbox when the run ended. Now it only supplies the sandbox, so add `Coder()`, or `Shell()` and `FileSystem()`, as shown above. Old arguments and imports fail with an error that names the replacement.

### What changed in the lifecycle

- A run no longer terminates the sandbox. It runs until you terminate it or its `sandbox_timeout` ends it (see [Clean up](#clean-up)).
- `sandbox_timeout` defaults to 24 hours instead of 5 minutes, so a later run can continue in the same sandbox.
- A run that continues a `message_history` reattaches to the previous run's sandbox. Pass `workspace='new'` for a fresh one.
- Reattaching to an expired or terminated sandbox raises `WorkspaceUnavailableError`. No empty replacement is created. If a command exits 137 because the sandbox was terminated mid-command, it raises the same error; a SIGKILLed command in a running sandbox returns exit 137.

### Migration table

| Previous API | Now |
| --- | --- |
| `image`, `app_name`, `create_app_if_missing`, `env` | Unchanged. `image` also takes a `modal.Image`, and its default now has `git` and `ripgrep`. |
| `sandbox_timeout` | Unchanged name. The default is now `86_400` (24 hours) instead of `300`. |
| `workdir` | Renamed `working_dir`. `workdir=` still works, with a deprecation warning. |
| `sandbox_id` | Removed. Use `agent.run(..., workspace=WorkspaceRef(provider='modal', id=sandbox_id))`. |
| `session`, `ModalSandboxSession` | Removed. Use `agent.run(..., workspace=ModalSandboxBackend(sandbox=<modal.Sandbox>))`. |
| `default_command_timeout` | Removed. Use `Shell(default_timeout=...)`. |
| `max_command_timeout` | Removed, with no direct equivalent: nothing caps a timeout the model asks for. `Shell(default_timeout=...)` sets the timeout of commands that don't give one; `sandbox_timeout` limits the lifetime of a new sandbox and does not apply to attached sandboxes. |
| `max_output_bytes`, `max_output_lines` | Removed. Use `Shell(max_output_chars=...)` or `ToolOutputLimits`. |
| `max_read_bytes` | Removed. Use `FileSystem(max_read_lines=..., max_read_chars=...)`. |
| `instructions` | Removed. Use the agent's `instructions`. |
| `run_command` tool | Removed. Use `Shell()`. |
| `read_file`, `write_file`, `list_directory` tools | Removed. Use `FileSystem()`. |
| `ModalSandboxExecResult` | Removed. Use `pydantic_ai.workspaces.CommandResult`. |
| `ModalSandboxError` | Removed. Catch `pydantic_ai.workspaces.WorkspaceError`. |
| `ModalSandboxTerminalError`, `ModalSandboxUnavailableError`, `ModalSandboxAuthError` | Removed. Catch `pydantic_ai.workspaces.WorkspaceUnavailableError`. |

## Durable execution

Run a Temporal dev server on `localhost:7233` first. The agent and workflow must be defined at module level for activity registration.

```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='modal_coder',
    capabilities=[ModalSandbox(), Coder(), TemporalDurability()],
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

For a lazily created sandbox outside this example, use `Coder(repo_context=False)`; by default Coder reads repository instructions when the run starts.

Removing a capability while workflows using it are still running changes their replay history. Drain those workflows or use [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) before deploying the change.

## API reference

::: pydantic_ai_harness.modal_sandbox.ModalSandbox

::: pydantic_ai_harness.modal_sandbox.ModalSandboxBackend
