# Workspaces

A workspace is where your agent runs commands and reads and writes files: a directory on your machine,
a container, or a cloud sandbox. Tools reach it through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace],
so the same tools work in any of them, and you switch by changing one capability.

## Give an agent a workspace

<span id="give-an-agent-an-environment"></span>

```python {title="workspace_agent.py" dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.workspaces import WorkspaceError

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[LocalWorkspace('.')],
)


@agent.tool
async def execute(ctx: RunContext, command: list[str]) -> str:
    """Run a command in the project directory."""
    try:
        result = await ctx.workspace.run(command, timeout=60)
    except WorkspaceError as error:  # e.g. a timeout, or a read-only workspace
        raise ModelRetry(str(error))
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    result = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    print(result.output)
    #> fizzbuzz.py is written and runs clean.


if __name__ == '__main__':
    asyncio.run(main())
```

- `LocalWorkspace('.')` gives every run the current directory. Commands start there and relative
  paths resolve against it. The directory must already exist.
- `ctx.workspace.run(...)` returns the command's `exit_code`, `stdout` and `stderr`. A failing command
  is a normal result, so the model sees what went wrong.
- `timeout=60` stops a stuck command with a
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError], which carries the output so
  far. The clock starts once the sandbox is ready, so creating it doesn't count.
- Turning any [`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] into `ModelRetry` lets the model
  try something else. Without it, the error ends the run.

You rarely write these tools yourself: [`Coder`](harness/coder.md), [`Shell`](harness/shell.md) and
[`FileSystem`](harness/filesystem.md) from the Pydantic AI Harness give the model file, search and shell
tools that work in whatever workspace the run has.

## Run it in a sandbox

<span id="move-it-into-a-sandbox"></span>

`LocalWorkspace` runs the model's commands on your machine, as you. To run them somewhere isolated, swap
it for a sandbox. Your tools, such as the `execute` tool above or `Coder` here, now run there:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and run its tests.')
```

The sandbox is created the first time the run uses it:

- With `Coder`, that's when the run starts, because `Coder` loads the repository's instructions. Pass
  `Coder(repo_context=False)` to wait for the first tool call instead.
- A [durable run](#durable-execution) also creates it when the run starts.

The sandbox keeps running after the run until you delete it or its provider stops it. [Cleaning
up](#cleaning-up) shows how to handle that.

## Supported workspaces {#supported-workspaces}

`LocalWorkspace` and `WindowsWorkspace` come with `pydantic-ai`. The rest come with
[Pydantic AI Harness](harness/index.md), each behind its own install extra.

| Workspace | Where commands run | Isolated from your machine | You need |
|---|---|---|---|
| [`LocalWorkspace`](#your-machine-or-a-sandbox) | A directory on this machine (macOS, Linux) | No | Nothing |
| [`WindowsWorkspace`](#platforms) | A directory on this machine, in PowerShell | No | Windows |
| [Docker Container](harness/docker-container.md) | A local Docker or Podman container | Yes, sharing your machine's kernel | Docker or Podman |
| [Smol Sandbox](harness/smol-sandbox.md) | A smolvm microVM on this machine | Yes | Apple silicon Mac, or Linux with KVM |
| [BoxLite Sandbox](harness/boxlite-sandbox.md) | A BoxLite micro-VM on this machine, with an optional egress allow-list | Yes | Apple silicon Mac, or Linux with KVM |
| [OpenSandbox](harness/opensandbox-sandbox.md) | A sandbox on your own OpenSandbox server, on Docker or Kubernetes | Yes | An OpenSandbox server |
| [Kubernetes Sandbox](harness/kubernetes-sandbox.md) | A pod from an agent-sandbox warm pool | Yes | A cluster with agent-sandbox |
| [Modal Sandbox](harness/modal-sandbox.md) | A Modal cloud sandbox | Yes | A Modal account |
| [E2B Sandbox](harness/e2b-sandbox.md) | An E2B cloud sandbox | Yes | An E2B account |
| [Daytona Sandbox](harness/daytona-sandbox.md) | A Daytona cloud sandbox | Yes | A Daytona account |
| [Render Sandbox](harness/render-sandbox.md) | A Render sandbox | Yes | A Render workspace with Sandboxes access |
| [Sprites Sandbox](harness/sprites-sandbox.md) | A persistent Fly.io Sprite | Yes | A Sprites API token |
| [boxd Sandbox](harness/boxd-sandbox.md) | A boxd cloud machine | Yes | A boxd account |
| [SSH Workspace](harness/ssh-workspace.md) | Another machine, over your `ssh` client | Yes, on another host, but not sandboxed there: commands run as your SSH user | SSH access |
| [Bubblewrapped Workspace](harness/bubblewrap-sandbox.md) | Wraps another workspace and runs its commands in Linux `bwrap` | Best effort, not a security boundary | Linux with `bwrap` on that host |

<span id="third-party-workspaces"></span>Other packages implement the same interface:
[`pydantic-ai-backend`](https://github.com/vstorm-co/pydantic-ai-backend), a community package by
[Vstorm](https://vstorm.co/), provides workspaces backed by Kubernetes pods, Docker, Daytona, a `sandboxd` service that holds the
Docker socket for you, and in-memory documents you keep in your own database. To list a package here,
open a pull request.

<span id="writing-a-backend"></span><span id="resolving-symlinks"></span><span id="checking-a-backend"></span>For anything else, [write a backend](workspace-custom.md#writing-a-backend): one class that runs a
command, plus a test suite that checks it behaves like the others.

<span id="supplying-a-workspace-from-a-capability"></span><span id="workspace-authority"></span><span id="files-you-didnt-write"></span><span id="authority-by-path"></span><span id="path-authority"></span>[Custom
Workspaces](workspace-custom.md) also shows how to:

- [Supply a workspace from a capability](workspace-custom.md#supplying-a-workspace-from-a-capability) to
  give each user their own.
- [Mark who wrote its files](workspace-custom.md#workspace-authority), for the whole workspace or
  [by path](workspace-custom.md#path-authority), so the model doesn't take a third-party file's text as
  your instructions.
- [Wrap a workspace](workspace-custom.md#wrapping-a-workspace) to add a policy of your own.

## Your machine or a sandbox

`LocalWorkspace` isolates nothing: a command can read and change anything you can, and the directory only
sets where commands start. Use it for your own, trusted work, and a sandbox for code you don't trust.

A sandbox is a container or a VM. A container shares your machine's kernel, so a kernel flaw can let a
command escape it. A VM runs its own kernel, which makes it the stronger boundary for code you don't trust.
Docker's [Containers are not VMs](https://www.docker.com/blog/containers-are-not-vms/) explains the difference.

Commands in a `LocalWorkspace` get your `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`, and nothing else
from the agent's environment, so your API keys don't reach the commands the model runs. Pass what they
need with `env=`, never the whole `os.environ`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[LocalWorkspace('.', env={'UV_OFFLINE': '1'})],
)
```

## Using the workspace in tools

`ctx.workspace` has the same methods in every workspace:

- [`run`][pydantic_ai.workspaces.Workspace.run] runs a command.
- [`read_text`][pydantic_ai.workspaces.Workspace.read_text] and
  [`write_text`][pydantic_ai.workspaces.Workspace.write_text] read and write text.
- [`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] and
  [`write_bytes`][pydantic_ai.workspaces.Workspace.write_bytes] read and write exact bytes.
- [`read_file`][pydantic_ai.workspaces.Workspace.read_file] reads a file as
  [`BinaryContent`][pydantic_ai.messages.BinaryContent], with its media type, to hand on to a user or a model.
  See [Get files out of the workspace](#get-files-out).
- Writing creates missing parent directories.
- [`list_dir`][pydantic_ai.workspaces.Workspace.list_dir], [`stat`][pydantic_ai.workspaces.Workspace.stat],
  [`exists`][pydantic_ai.workspaces.Workspace.exists], [`make_dir`][pydantic_ai.workspaces.Workspace.make_dir]
  and [`remove`][pydantic_ai.workspaces.Workspace.remove] work with directories and entries.

```python {title="file_tools.py" requires="workspace_agent.py"}
from pydantic_ai import ModelRetry, RunContext

from workspace_agent import agent


@agent.tool
async def read_source(ctx: RunContext, path: str) -> str:
    """Read a text file from the project."""
    try:
        return await ctx.workspace.read_text(path)
    except FileNotFoundError:
        raise ModelRetry(f'{path} does not exist.')


@agent.tool
async def save_notes(ctx: RunContext, notes: str) -> str:
    """Save notes for later steps."""
    await ctx.workspace.write_text('NOTES.md', notes)
    return 'Saved to NOTES.md.'
```

- Relative paths resolve against the working directory. It is a starting point, not a boundary: `..` and
  absolute paths reach the rest of the environment.
- `run(..., cwd='subdir')` starts one command in another directory, relative to the working directory. The
  directory must already exist.
- Sandbox providers use different default users and directories, so prefer relative paths, or set
  `working_dir=` on the sandbox.
- A bad path raises the usual error, such as `FileNotFoundError` or `IsADirectoryError`. Raise `ModelRetry`
  from it so the model can try again. Uncaught, it ends the run. On Temporal, it follows the
  [retry policy](#durable-execution) instead.
- An environment that is gone, such as a sandbox deleted during a command, raises
  `WorkspaceUnavailableError` and ends the run. A command killed by a signal while its environment stays
  live returns its exit code instead.

## Pick up where you left off {#continuing-in-the-same-workspace}

Pass the message history to the next run, and it continues in the same workspace. With a sandbox, the
files and installed packages from the first run are still there:

```python {requires="workspace_agent.py"}
from workspace_agent import agent


async def main() -> None:
    first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    await agent.run('Now add a test for it.', message_history=first.all_messages())
```

This works because each model response records which workspace it ran in. The record is a
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef], a reference (or "ref") holding the provider's name and
the environment's ID, stored on [`workspace_ref`][pydantic_ai.messages.ModelResponse.workspace_ref]. It is
saved with the messages, so this works however you store them:

```python {requires="workspace_agent.py"}
from pydantic_ai import ModelMessagesTypeAdapter

from workspace_agent import agent


async def main() -> None:
    first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    print(first.response.workspace_ref == first.workspace.ref)
    #> True
    saved = first.all_messages_json()  # store it in your database

    history = ModelMessagesTypeAdapter.validate_json(saved)
    await agent.run('Now add a test for it.', message_history=history)
```

To continue without the history, for example in a new conversation about the same project, save the
reference and pass it as `workspace=`:

```python {requires="workspace_agent.py"}
from workspace_agent import agent


async def main() -> None:
    first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    ref = first.workspace.ref  # a small dataclass: `provider` and `id`

    await agent.run('Explain what fizzbuzz.py does.', workspace=ref)
```

- A sandbox's `ref` is `None` until the run first uses it. A run that never touches the workspace
  records no reference, and a later turn that doesn't use it keeps the earlier one.
- If the sandbox a reference names is gone, the run raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] instead of starting
  over in an empty one. Pass `workspace='new'` to start over on purpose.
- [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] and the [UI adapters](ui/overview.md)
  strip references from history a client sends you, so a client can't choose where your agent works.
  Save the reference on your server and pass it as `workspace=`.

## Get files out of the workspace {#get-files-out}

A file the agent made, such as a chart or a spreadsheet, stays in the workspace until you read it.
[`read_file`][pydantic_ai.workspaces.Workspace.read_file] returns it as
[`BinaryContent`][pydantic_ai.messages.BinaryContent], ready to save, send to a user, or pass to a model.
Right after the run, read it from the result:

```python {title="get_chart.py" requires="workspace_agent.py" names="defined"}
from pydantic_ai import BinaryContent

from workspace_agent import agent


async def plot_sales() -> BinaryContent:
    result = await agent.run('Plot the monthly sales in sales.csv to chart.png.')
    return await result.workspace.read_file('chart.png')
```

Later, for example when a user clicks a download link, you may have only the reference you stored.
`backend(ref)` on the capability that created the sandbox, such as `DockerContainer` or `E2BSandbox`,
attaches to it without starting a run:

```python {title="download.py" names="defined"}
from pydantic_ai_harness.docker_container import DockerContainer

from pydantic_ai import BinaryContent
from pydantic_ai.workspaces import Workspace, WorkspaceRef


async def download(ref: WorkspaceRef, path: str) -> BinaryContent:
    workspace = Workspace(DockerContainer().backend(ref))
    return await workspace.read_file(path)
```

- The media type comes from the file name's extension, as in
  [`BinaryContent.from_path`][pydantic_ai.messages.BinaryContent.from_path]. An unknown extension
  gives `application/octet-stream`, and an image comes back as a
  [`BinaryImage`][pydantic_ai.messages.BinaryImage].
- The whole file is read into memory, whatever its size. To refuse large files, check
  [`stat`][pydantic_ai.workspaces.Workspace.stat]`(path).size` first.
- Use the capability that created the sandbox. If the sandbox is gone, reading raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- Serve only paths your app chose, such as the file name you asked the agent to write. `read_file` doesn't
  keep a path inside the working directory: an absolute path or `..` can name any file the environment can
  read, and with [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] that is the host. A path from the
  client or from a model response (which the prompt or fetched content can steer) needs a check that
  `await workspace.realpath(path)` is under `await workspace.working_dir()` before you read it.

## Cleaning up

Pydantic AI never deletes an environment. A sandbox keeps running, and billing, after the run until you
delete it or its provider's lifetime settings stop it. Each sandbox page shows how to delete one.

- After a successful run that used the sandbox, `result.workspace.ref` names it, so you can continue in it or
  delete it. If the ref is `None`, the run never created a sandbox, and calling `result.workspace` methods
  creates a new one that no message records.
- A failed run returns no result. Delete its sandbox in an `on_run_error` [hook](hooks.md), where
  `ctx.workspace.ref` names it. `after_run` doesn't run when a run fails.
- If you delete the sandbox after each run, pass `workspace='new'` when you continue its history, or the
  next run raises `WorkspaceUnavailableError`.

## Hand the workspace to another agent

`workspace=` also takes a live workspace. Here the coding agent asks a reviewer to check its work. The
reviewer reads the same files, and [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] stops
it from changing them:

```python {title="reviewer.py" requires="workspace_agent.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import ReadOnlyWorkspace

from workspace_agent import agent

reviewer = Agent('{{ anthropic:large }}', instructions='Review code. Be brief.')


@reviewer.tool
async def read_file(ctx: RunContext, path: str) -> str:
    return await ctx.workspace.read_text(path)


@agent.tool
async def ask_reviewer(ctx: RunContext, request: str) -> str:
    """Ask a reviewer to check files in this workspace."""
    review = await reviewer.run(request, workspace=ReadOnlyWorkspace(ctx.workspace), usage=ctx.usage)
    return review.output


async def main() -> None:
    result = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    result = await agent.run('Ask the reviewer to check fizzbuzz.py.', message_history=result.all_messages())
    print(result.output)
    #> The reviewer says fizzbuzz.py is correct.
```

The reviewer has no workspace capability of its own, so it works wherever it is told to: a local checkout
in one run, a sandbox in the next.

## Read-only access

Pass `read_only=True` to let tools read and list files while refusing commands and changes:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('{{ anthropic:medium }}', capabilities=[LocalWorkspace('.', read_only=True)])
```

A refused change raises [`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError].
Commands are refused too, because a command can change files. A capability can check
[`ctx.workspace.read_only`][pydantic_ai.workspaces.Workspace.read_only] to leave its write tools out.

For a single run, pass `workspace=ReadOnlyWorkspace(...)`, as the
[reviewer above](#hand-the-workspace-to-another-agent) does. For a policy of your own, see
[Wrapping a workspace](workspace-custom.md#wrapping-a-workspace).

## Choosing a run's workspace

A run uses the first of these that applies:

1. The `workspace=` argument.
2. The reference on the latest response in `message_history`.
3. The agent's workspace capability: its directory for `LocalWorkspace`, a new sandbox for a sandbox.

`workspace=` takes one of:

- `'new'`, to start fresh and ignore the history.
- A [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef], for the agent's capability to attach to.
- A live workspace, such as `result.workspace` or `ctx.workspace`, or a
  [backend](workspace-custom.md#writing-a-backend), used as is.

To move to a new directory or provider without breaking old conversations, list both capabilities. They are
asked in order, and only the one that recognizes a reference attaches to it. New conversations use the
first. Conversations that started in the second continue there.

```python {title="two_workspaces.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[
        LocalWorkspace('~/projects/current', id='current'),  # new conversations
        LocalWorkspace('~/projects/archive', id='archive'),  # conversations that started here
    ],
)
```

`LocalWorkspace` only attaches to a reference to its own directory, so message history can't point the
agent at another directory on your machine. The directory is compared as written, without resolving
symlinks, so if you change `~/proj` to the path it links to, old conversations no longer match.

When no capability supplies the workspace:

- A reference in `message_history` that no capability recognizes raises `UserError`, for example after
  you switch providers. Pass `workspace='new'`, or keep the old capability. An agent with no workspace
  capability, such as one that summarizes the conversation, ignores the reference.
- `workspace=` with a `WorkspaceRef` or `'new'` raises `UserError`.
- Without a reference, the run has no workspace, and tools that use it raise `WorkspaceUnavailableError`.

In tests, `with agent.override(workspace=LocalWorkspaceBackend(tmp_path)):` puts every run in that directory,
whatever the agent's capability, the history or `workspace=` say. `LocalWorkspaceBackend` is the class that
does `LocalWorkspace`'s work, without the capability around it.

## Durable execution

<span id="harness-workspace-capabilities"></span>

Under [Temporal](durable_execution/temporal.md), [DBOS](durable_execution/dbos.md) or
[Prefect](durable_execution/prefect.md), attach the workspace capability next to the durability capability:

```python {title="durable_workspace.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability

agent = Agent(
    '{{ anthropic:medium }}',
    name='fizzbuzz',  # names the agent's Temporal activities
    capabilities=[LocalWorkspace('.'), TemporalDurability()],
)
```

Swap `TemporalDurability` for `DBOSDurability` or `PrefectDurability` and nothing else in the agent changes.
The [engine guides](durable_execution/overview.md) show where each run starts.

Tools use `ctx.workspace` as in a plain run:

- Harness tools such as `Coder`, `Shell` and `FileSystem` work on all three engines, including background
  jobs, sticky `cd` and [`FileChangeRequestedEvent`](harness/filesystem.md#events) approval.
- On Temporal, live events and delegation have [limits](durable_execution/temporal.md#harness-workspace-capabilities).
- `FileSystem` and the `shell` tool also work under [Azure Durable Task](harness/azure-durable-task.md) and
  [Render Workflows](harness/render-workflows.md).

An engine normally runs each tool call as one recorded unit: a Temporal activity, a DBOS step or a Prefect
task. A tool that must run in the workflow code itself, for example to approve a file change before writing
it, can set [`durable=False`](durable_execution/overview.md#running-a-tool-outside-the-durable-unit). The
workspace calls it makes are still recorded as activities, steps or tasks.

### Creating and reattaching

- A durable run creates or attaches its environment when it starts, even if no tool uses it. Retries,
  replays and recovery reattach to it.
- If a worker dies after creating a sandbox but before that is recorded, the retry can create a second one.
  Delete the first with the provider's tools.
- Without an explicit `run_id`, a durable run takes its ID from the Temporal run ID, DBOS workflow ID or
  Prefect flow run ID, so workspace state such as a sticky `cd` or a background job survives a worker
  restart.
- On Temporal, `LocalWorkspace` calls run on whichever worker picks them up, so every worker needs the
  directory at the same absolute path, on shared storage.

### Retries and timeouts

- A workspace call can run again if a worker dies during it.
- Timeouts, output-limit failures, read-only refusals and a lost environment are not retried.
- Other errors, such as `FileNotFoundError`, follow the engine's retry policy. On Temporal that retries
  without limit by default. Raise `ModelRetry` from them, or set `maximum_attempts` on the
  [retry policy](durable_execution/temporal.md#activity-configuration).
- On Temporal, a command must finish within its activity's `start_to_close_timeout`, 60 seconds by
  default, whatever its own `timeout`. Raise it for a tool with
  `metadata={'temporal': ActivityConfig(start_to_close_timeout=...)}`.
- Workspace calls from workflow code use `activity_config`. If you set it, keep a `start_to_close_timeout`
  in it.
- On DBOS, tool calls in a run with a workspace run one at a time, so their steps replay in order.

### Harness shell state

- A sticky `cd` is kept in the workspace under `.pydantic-ai-harness/shell/run-state/`, keyed by the run,
  so a new worker continues in the same directory. It is removed when the run ends.
- Commands in one run that change directory at the same time can overwrite each other's state.
- A retried `start_command` reattaches to its job instead of starting a second process.
- If a launch failed partway, the retry reports a pending launch. Once the process has stopped, remove its
  files under `.pydantic-ai-harness/shell/`.

### Settings belong on the capability

Each step rebuilds the workspace from the agent's capability, and `workspace=` passes only the reference.
Anything you wrap around a workspace for one run would be dropped, so:

- Put policy on the capability, such as `LocalWorkspace(..., read_only=True)` or
  `LocalWorkspace(..., authority='none')`.
- A `ReadOnlyWorkspace(...)` or `AuthorityWorkspace(...)` passed to the run raises `UserError`, unless the
  capability already applies the same policy.
- A live workspace passed to the run that has no ref yet raises `UserError`. A previous `result.workspace`
  without a ref is the exception: it starts a fresh workspace.
- A workspace capability passed to the run raises `UserError` if it selects a different kind of workspace
  or policy. Its other settings, such as a sandbox image, are ignored.
- Workspace calls made from workflow code store their arguments, `env=` and file contents included, in
  durable history. Keep secrets in the capability's `env=` or use them inside a tool.
- `workspace.backend` is not available in workflow code, which includes DBOS function tools and tools with
  `durable=False`. Reach the provider's own API from a tool that runs as an activity or step.
- The deprecated `TemporalAgent`, `DBOSAgent` and `PrefectAgent` refuse a workspace. Use the durability
  capability instead.

!!! warning "Adding a workspace with runs in flight"
    Adding or removing a workspace capability changes what runs record, so runs started before the change
    no longer match their history: Temporal replay fails with a nondeterminism error, and DBOS can't recover
    the workflow. Let in-flight runs finish first, or deploy the change with Temporal worker versioning, a
    new task queue, or a new DBOS application version. On Prefect, let running flows finish before you
    deploy.

## Platforms

`LocalWorkspace` needs macOS or Linux. On Windows, use
[`WindowsWorkspace`][pydantic_ai.capabilities.WindowsWorkspace]. It takes the same options and isolates
nothing either.

- Commands run in `pwsh` when it is on `PATH`, and in Windows PowerShell otherwise.
- Commands get only the variables Windows tools need to start, plus `env`.
- A timeout or cancellation ends the processes a command started.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import WindowsWorkspace

agent = Agent('{{ anthropic:medium }}', capabilities=[WindowsWorkspace('.')])
```

A tool that builds paths or shell commands itself can read
[`ctx.workspace.platform`][pydantic_ai.workspaces.Workspace.platform] and use
[`ctx.workspace.paths`][pydantic_ai.workspaces.Workspace.paths] (`posixpath` or `ntpath`).

## Limits

<span id="security-choices"></span><span id="timeouts-and-clocks"></span>

- A run has one workspace.
- A local command whose combined output passes 10 MiB raises `WorkspaceOutputLimitError`, with the first
  64 KiB of each stream. Redirect large output to a file.
- Commands get stdin at EOF, so use non-interactive flags such as `-y`.
- `LocalWorkspace` background processes outlive the run. Redirect their output to a file so the command
  doesn't wait for them, and stop them before your program exits.
- Run spans record the workspace id, a `LocalWorkspace`'s directory path included, even with
  `include_content=False`.
- Realtime sessions don't select a workspace yet.
