# Render Sandbox

Run your agent's commands and file edits in an isolated [Render sandbox](https://render.com/docs/sandboxes) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[coder,render,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder,render,anthropic]"
```

Set `RENDER_API_KEY` to a [Render API key](https://render.com/docs/api#1-create-an-api-key), and `RENDER_WORKSPACE_ID` to the `tea-...` ID of your [Render workspace](https://render.com/docs/team-members), Render's name for a team. The workspace needs Sandboxes access and billing set up. The examples also use `Coder` and an Anthropic model, so they need the `coder` and `anthropic` extras and `ANTHROPIC_API_KEY`. Swap them for your own tools and model provider.

## Quick start

<span id="choose-the-tools"></span>

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox. For a narrower agent, use [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) or your own tools on [`ctx.workspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#using-the-workspace-in-tools).

- `Coder` creates it when the run starts, to load the repository's instructions, even if no tool is called. Pass `Coder(repo_context=False)` to wait for the first tool call.
- It keeps running, and billing, after the run, even when idle, until `sandbox_timeout` runs out. `RenderSandbox` turns off Render's idle suspension. See [Clean up](#clean-up).
- It lives for 1 hour. Pass up to `sandbox_timeout=86400` for longer.
- Commands start in the sandbox's home directory. Pass `working_dir=` for another absolute directory. It is created on a new sandbox.
- It can reach the internet. Pass `allow_internet_access=False` to block it.
- The image needs `python3`. Render's default image has it.

## Continue in the same sandbox

<span id="reattach-later"></span>

Pass the message history, and the next run works in the same sandbox:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(), Coder()])
result = agent.run_sync('Write a fizzbuzz.py and run it.')

followup = agent.run_sync('Add tests for fizzbuzz.py.', message_history=result.all_messages())
ref = result.workspace.ref  # or store this, and pass `workspace=ref` to a later run
```

The history records the sandbox as a `WorkspaceRef`, or "ref": the provider's name and the sandbox's ID. `result.workspace.ref` is the same ref, for continuing without the history.

If the sandbox has been terminated, the run raises `WorkspaceUnavailableError`. Pass `workspace='new'` to start over. [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/#continuing-in-the-same-workspace) covers how runs find their workspace.

The ref holds no credentials, so the process that reattaches needs `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`, or a `client=`, too.

## Prepare a sandbox before the run

To seed files or install packages before the agent starts, create the backend yourself, work in it through `Workspace`, and pass it to the run:

```python {names="defined"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RenderSandbox(), Coder()])


async def main() -> None:
    backend = RenderSandboxBackend()
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
            await RenderSandbox().destroy(ref)


if __name__ == '__main__':
    asyncio.run(main())
```

The `finally` terminates the sandbox even when setup or the run fails, so read any files you want to keep before it. To keep the sandbox, store `backend.ref` instead.

## Give the agent a desktop

Pass `desktop=True` and add [`ComputerUse`](../computer_use/):
`capabilities=[RenderSandbox(desktop=True), ComputerUse()]`. The screen is 1280x800; pass a size such as
`desktop=(1440, 900)` for another.

Render's default image has none of [the display's programs](../computer_use/#what-the-display-needs), so the first use of the screen installs them with `apt-get`, with no web browser. That needs `allow_internet_access`: creating a new sandbox without it raises `UserError`. An attached sandbox without internet access must already have the programs, or the first use of the screen raises `WorkspaceUnavailableError` naming what's missing.

To give the agent a browser or other programs, install them before the run, as in [Prepare a sandbox before the run](#prepare-a-sandbox-before-the-run) with `RenderSandboxBackend(desktop=True)`, or let the agent install them with `Shell`.

The agent can type only ASCII: an action that types or presses a character such as `é` fails, and the model is told why.

## Clean up

Terminate a sandbox with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.render_sandbox import RenderSandbox


async def terminate_sandbox(ref: WorkspaceRef) -> None:
    await RenderSandbox().destroy(ref)
```

`destroy(ref)` raises `WorkspaceUnavailableError` if Render rejects the credentials.

To terminate the sandbox after every run, including failed ones, use `after_run` and `run_error` hooks:

```python {names="defined"}
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.render_sandbox import RenderSandbox

sandbox = RenderSandbox()
hooks = Hooks()


@hooks.on.after_run
async def terminate_sandbox(ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await sandbox.destroy(ctx.workspace.ref)
    return result


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await sandbox.destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[sandbox, Coder(), hooks])
```

To keep results, read them through `ctx.workspace` in `after_run` before calling `destroy`.

### Find a lost sandbox

If your program stops before it stores the ref, for example after a crash or Ctrl-C, the sandbox runs until its `sandbox_timeout`. To find it:

- Search your logs for `Created Render sandbox <id>`, logged at INFO by the `pydantic_ai_harness` logger.
- A connection failure while Render creates the sandbox can leave it running with no ref and no log line. Check your Render workspace for unused sandboxes.

Then terminate it: `await RenderSandbox().destroy(WorkspaceRef(provider='render', id='<id>'))`.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `render.RenderAsync` you own. It is never closed for you. Default: `None`, which reads `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `sandbox_timeout` | Seconds a new sandbox lives before Render terminates it, at most `86_400`. Default: `3_600`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve from. Default: `None`, the sandbox's home directory. Created on a new sandbox. On an attached one, it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine reaches the sandbox. The agent's commands can read these, so leave out secrets it shouldn't see. |
| `desktop` | Give the sandbox a graphical display for [`ComputerUse`](../computer_use/): `True` for 1280x800, or its `(width, height)`. The first use installs the desktop's programs with `apt-get`, so a new sandbox needs `allow_internet_access`. The size applies when the display first starts. Default: `False`. |
| `authority` | Who wrote the sandbox's files. A capability that puts a file's text in the prompt gives it no more [authority](https://pydantic.dev/docs/ai/core-concepts/workspace-custom/#workspace-authority) than this. Default: `'operator'`, your own files. Use `'user'` for the user's files, and `'none'` for files nobody on your side wrote, such as a cloned third-party repository. |
| `path_authority` | Levels for some directories and files, overriding `authority` under them. The longest matching path wins. Paths are in the sandbox, relative to the working directory or absolute, and `~` isn't expanded. Default: `None`. |

`client`, `working_dir`, `env`, `desktop`, `authority` and `path_authority` also apply when you reattach, and `destroy` uses `client`. The other options only shape a new sandbox.

## Limits

<span id="what-a-timeout-stops"></span>

- A sandbox that doesn't finish setting up in time raises `TimeoutError`.
- A missing sandbox or rejected access raises `WorkspaceUnavailableError`.
- Render allows 100 API requests a minute. A rate-limited request raises the SDK's `RateLimitError` and is not retried.
- A command and its environment must fit in about 110 KB. Write larger content with the file tools.
- A command whose output passes 10 MiB raises `WorkspaceOutputLimitError`. Redirect large output to a file.
- Stopping a command on a timeout or cancellation is best effort. If the stop can't be confirmed, a warning names the sandbox.
- Background processes that detach can keep running.
- `RenderSandbox` needs asyncio. Under Trio, it raises `UserError`.

## Durable execution

<span id="render-workflows"></span>`RenderSandbox` works with [`RenderWorkflows`](https://pydantic.dev/docs/ai/harness/render-workflows/), and its model and tool tasks share one sandbox.

- The `render` extra covers both. Set the Render variables on every worker.
- A retried root task runs the agent again, so terminate the sandbox from the task that runs the agent, as in [Clean up](#clean-up).

## Telemetry

`RenderSandbox` traces creating or reconnecting to the sandbox, until it is running, as `workspace create` or `workspace attach`, under the call that needed it, and `destroy(ref)` called inside a run as `workspace destroy`; see [workspace spans](https://pydantic.dev/docs/ai/integrations/logfire/#workspace-spans). With [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/), the agent run span identifies the sandbox with `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span.

## API reference

::: pydantic_ai_harness.render_sandbox.RenderSandbox

::: pydantic_ai_harness.render_sandbox.RenderSandboxBackend
