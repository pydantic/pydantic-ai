# Daytona Sandbox

Run your agent's commands and file edits in an isolated [Daytona](https://www.daytona.io) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/daytona_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

```mermaid
flowchart TB
    accTitle: Daytona Sandbox: your agent's commands run in a Daytona cloud sandbox
    accDescr: On your machine, the model calls Coder, Shell or FileSystem tools such as run_command. DaytonaSandbox's get_workspace hands the run a Daytona cloud sandbox as ctx.workspace over the Daytona API with DAYTONA_API_KEY. It stops when idle, destroy(ref) deletes it, and later runs reattach with the WorkspaceRef in the message history or result.workspace.ref, starting it if stopped.
    subgraph machine["Your machine"]
        model(["model"])
        tools["Coder, Shell or<br/>FileSystem tools,<br/>such as run_command"]
        ref["WorkspaceRef<br/>in the message history,<br/>result.workspace.ref"]
        daytona["DaytonaSandbox()<br/>get_workspace hands<br/>the run the sandbox"]:::capability
    end
    sandbox["Daytona cloud sandbox<br/>stops when idle,<br/>deleted by destroy(ref)"]:::external
    model <-->|"tool call, result"| tools
    tools -->|"ctx.workspace"| daytona
    ref -.->|"a later run reattaches,<br/>starting it if stopped"| daytona
    daytona <-->|"Daytona API,<br/>DAYTONA_API_KEY"| sandbox
    classDef capability stroke:#E520E9,stroke-width:2px
    classDef external stroke:#0E9C86,stroke-width:2px
```

## Install

uv:

```bash
uv add "pydantic-ai-harness[coder,daytona,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder,daytona,anthropic]"
```

Set `DAYTONA_API_KEY` to an API key from your Daytona dashboard. The examples also use `Coder` and an Anthropic model, so they need the `coder` and `anthropic` extras and `ANTHROPIC_API_KEY`. Swap them for your own tools and model provider.

## Quick start

<span id="choose-the-tools"></span>

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox. For a narrower agent, use [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) or your own tools on [`ctx.workspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#using-the-workspace-in-tools).

- `Coder` creates it when the run starts, to load the repository's instructions, even if no tool is called. Pass `Coder(repo_context=False)` to wait for the first tool call.
- It keeps its disk, and costs money, until you delete it. Daytona stops it when idle, but doesn't delete it. See [Clean up](#clean-up).
- Commands run as the non-root `daytona` user in `/home/daytona`. Pass `working_dir='/home/daytona/project'` to give the agent a project directory. It is created on a new sandbox.
- The default snapshot has git, Python and `rg`, but not your project's dependencies. A snapshot is the image a new sandbox starts from. Build a Daytona snapshot with them and pass `snapshot='my-snapshot'`, or install them [before the run](#prepare-a-sandbox-before-the-run). An unknown snapshot fails on first use with `WorkspaceUnavailableError`.

## Continue in the same sandbox

<span id="reattach-later"></span>

Pass the message history, and the next run works in the same sandbox, starting it again if Daytona stopped it:

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
ref = result.workspace.ref  # or store this, and pass `workspace=ref` to a later run
```

The history records the sandbox as a `WorkspaceRef`, or "ref": the provider's name and the sandbox's ID. `result.workspace.ref` is the same ref, for continuing without the history.

If the sandbox has been deleted, the run raises `WorkspaceUnavailableError`. Pass `workspace='new'` to start over. [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/#continuing-in-the-same-workspace) covers how runs find their workspace.

The ref holds no credentials, so the process that reattaches needs `DAYTONA_API_KEY` or a `client=` too.

Already have a `daytona.AsyncSandbox`? Pass `workspace=DaytonaSandboxBackend(sandbox=sandbox, working_dir=..., env=...)` to a run. `DaytonaSandbox`'s settings don't apply to it. A backend's `get_sandbox()` returns its `daytona.AsyncSandbox` for Daytona features the workspace API doesn't cover. On a backend with no sandbox yet, it creates one, which Daytona bills.

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

The run leaves a backend you built open, so close it with `aclose()`. That closes the client it opened from `DAYTONA_API_KEY`, not the sandbox. A `client=` or `sandbox=` you passed is never closed. To keep the sandbox, store `backend.ref` instead of destroying it.

## Give the agent a desktop

Pass `desktop=True` and add [`ComputerUse`](../computer_use/):
`capabilities=[DaytonaSandbox(desktop=True), ComputerUse()]`. The screen is 1280x800; pass a size such as
`desktop=(1440, 900)` for another. The size is set when the sandbox is created.

Daytona's default snapshot has a desktop with a terminal and Chromium. A custom snapshot needs [Daytona's desktop packages](https://www.daytona.io/docs/en/computer-use/).

- The desktop runs a VNC server with no password on one of the sandbox's preview ports.
- On a private sandbox, anyone who holds a preview token for that port can see the screen and control the mouse and keyboard.
- On a public sandbox, Daytona's preview links need no token, so anyone with the preview URL can see and control the desktop.
- Where the snapshot's Chromium can't run its own sandbox, as in Daytona's default snapshot, the desktop start adds `--no-sandbox` to Chromium's flags through a file in `/etc/chromium.d/`. It writes the file as root or with passwordless `sudo`, on sandboxes it creates and on ones you attach with `ref=` or `sandbox=`. The file stays after the run, so a web page that breaks out of Chromium's renderer can reach the rest of the sandbox. A custom snapshot whose Chromium sandbox works is left alone.
- Keep the sandbox private if others shouldn't see or control the desktop.
- On Daytona's Tier 1 and Tier 2, a sandbox can reach only package registries, GitHub and AI APIs, even with `allow_internet_access=True`, so its browser can't open other sites. See [Daytona's network limits](https://www.daytona.io/docs/en/network-limits/).
- The desktop can't press print screen, pause, scroll lock or the menu key, or type characters beyond U+FFFF, such as most emoji. The agent is told when it asks for one.

## Clean up

Delete a sandbox with the ref you stored. This doesn't start a stopped sandbox:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox


async def delete_sandbox(ref: WorkspaceRef) -> None:
    await DaytonaSandbox().destroy(ref)
```

A failed run returns no result, so delete its sandbox in an `on_run_error` hook:

```python
from typing import Any

import anyio

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

hooks = Hooks()


@hooks.on.run_error
async def delete_failed_run_sandbox(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        # Shielded so a deadline around the run can't cut the delete short, bounded so a hung call can't hang it.
        with anyio.move_on_after(30, shield=True):
            await DaytonaSandbox().destroy(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder(), hooks])
```

A sandbox you don't delete keeps its disk, and a later run starts it again:

- A container sandbox is stopped after 15 idle minutes and archived after 7 days. Set `auto_stop_interval=` in minutes to change the idle time, or `0` to keep it running.
- A sandbox from a VM snapshot is paused after 60 idle minutes instead, and is never archived.

### Find a lost sandbox

If your program stops before it stores the ref, for example after a crash or Ctrl-C, the sandbox is left behind. To find it:

- List your sandboxes in Daytona. Sandboxes Pydantic AI creates have the label `created-by=pydantic-ai`.
- Or search your logs for `Created Daytona sandbox <id>`, logged at INFO by the `pydantic_ai_harness` logger.

Then delete it: `await DaytonaSandbox().destroy(WorkspaceRef(provider='daytona', id='<id>'))`.

## Keep a token out of the sandbox

Pass `secrets` to give a new sandbox Daytona [Secrets](https://www.daytona.io/docs/en/secrets), as a map from environment variable name to Secret name. The variable holds a placeholder, and Daytona swaps in the real value as requests leave for the Secret's `hosts`. Create the Secret once, then use it:

```python {test="skip"}
from daytona import AsyncDaytona, CreateSecretParams

from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox


async def create_github_secret(token: str) -> None:  # a GitHub token, such as the output of `gh auth token`
    async with AsyncDaytona() as daytona:
        await daytona.secret.create(
            CreateSecretParams(
                name='github-token',
                value=token,
                hosts=['github.com', 'api.github.com', 'uploads.github.com'],
            )
        )


sandbox = DaytonaSandbox(secrets={'GH_TOKEN': 'github-token'})
```

`gh` then works as the token's owner, while `echo $GH_TOKEN` in the sandbox prints the placeholder. For `git push`, have git send the placeholder as its password:

```bash
git config --global credential.https://github.com.helper '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
```

- Always set `hosts` on the Secret. Without them, Daytona sends the real value to any host.
- Daytona swaps the value only in HTTPS request headers, including git's password. Request bodies, query strings and plain HTTP keep the placeholder. Use `https://` remotes, not `git@github.com:owner/repo.git`.
- The agent can't see the token, but it can use it for anything the token allows.
- An attached sandbox keeps the Secrets it was created with. To swap the token, call `daytona.secret.update` with the Secret's ID. Running sandboxes pick it up within seconds.
- To change which Secrets a running sandbox has, call `update_secrets` on `await backend.get_sandbox()`. It replaces the whole set. New variables reach only commands started after it. A sandbox created without Secrets must restart first.

## Configuration

| Option | What it does |
| --- | --- |
| `client` | A `daytona.AsyncDaytona` to share across runs on one event loop, or to configure its API URL or target. Default: `None`, a client each run opens from `DAYTONA_API_KEY` and closes when it ends. You close a client you pass. |
| `snapshot` | Daytona snapshot a new sandbox starts from. Default: `None`, Daytona's default. An unknown snapshot fails on first use. |
| `auto_stop_interval` | Idle minutes before Daytona stops a new sandbox. `0` disables it. Default: `None`, Daytona's default of 15 minutes for a container. A VM pauses after 60 minutes instead. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. `False` blocks everything only for Tier 3 and 4 organizations. Lower tiers can still reach package registries, GitHub and AI APIs, so `False` doesn't stop data leaving the sandbox there. See [Daytona's network limits](https://www.daytona.io/docs/en/network-limits/). |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `None`, `/home/daytona`. Created on a new sandbox. On an attached one, it must already exist. |
| `env` | Environment variables every command gets. Default: `None`. Nothing from your machine reaches the sandbox. The agent's commands can read these, so leave out secrets it shouldn't see. |
| `secrets` | Daytona Secrets for a new sandbox, as a map from environment variable name to the name of an existing organization Secret. Passed to `AsyncDaytona.create` unchanged. Default: `None`. An attached sandbox keeps the ones it was created with. See [Keep a token out of the sandbox](#keep-a-token-out-of-the-sandbox). |
| `authority` | Who wrote the sandbox's files. A capability that puts a file's text in the prompt gives it no more [authority](https://pydantic.dev/docs/ai/core-concepts/workspace-custom/#workspace-authority) than this. Default: `'operator'`, your own files. Use `'user'` for the user's files, and `'none'` for files nobody on your side wrote, such as a cloned third-party repository. |
| `path_authority` | Levels for some directories and files, overriding `authority` under them. The longest matching path wins. Paths are in the sandbox, relative to the working directory or absolute, and `~` isn't expanded. Default: `None`. |
| `desktop` | Give the sandbox a graphical display for [`ComputerUse`](../computer_use/): `True` for 1280x800, or its `(width, height)`. The size applies to a new sandbox. Default: `False`. |

`client`, `working_dir`, `env`, `desktop` (but not its size), `authority` and `path_authority` also apply when you reattach, and `destroy` uses `client`. The other options only shape a new sandbox.

## Limits

<span id="what-a-timeout-stops"></span>

- A timeout or cancellation stops the command, not the sandbox. If Daytona doesn't confirm the stop, the command may still be running, so inspect or delete the sandbox.
- `WorkspaceTimeoutError` carries the last 4 KiB of each stream.
- Output is collected when the command exits, so output a background child writes afterwards is lost. Redirect it to a file. `Shell.start_command` does this for you.
- A command whose output passes 10 MiB raises `WorkspaceOutputLimitError` once it exits. Redirect large output to a file.
- `timeout=None` removes the command deadline, not Daytona's idle stop.
- The Daytona SDK needs asyncio. Under Trio, every operation raises `UserError`.

## Durable execution

Add `TemporalDurability()` next to `DaytonaSandbox()` and install the `temporal` extra. [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) covers what changes.

## Telemetry

`DaytonaSandbox` traces starting or reconnecting to the sandbox as `workspace create` or `workspace attach`, under the call that needed it, and `destroy(ref)` called inside a run as `workspace destroy`; see [workspace spans](https://pydantic.dev/docs/ai/integrations/logfire/#workspace-spans). Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) also records the sandbox on the agent run span as `pydantic_ai.workspace.provider` and `pydantic_ai.workspace.id`, and each command or file operation a tool makes runs inside that tool call's span; operations at run start, such as `RepoContext` loading repo instructions, run in the agent run span. Creating a sandbox logs its ID at `INFO` (`Created Daytona sandbox <id>`) on the `pydantic_ai_harness.daytona_sandbox._backend` logger, so a sandbox that outlives its run can still be found.

## API reference

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandbox

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend
