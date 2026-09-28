# Workspaces

A workspace is an environment your agent can use: it runs commands and reads and writes files there.
It can be a directory on your machine or a sandbox in the cloud, and your tools don't need to know
which, because they all use it through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace].

## Give an agent an environment

```python {title="workspace_agent.py" dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.workspaces import WorkspaceError

agent = Agent(
    'anthropic:claude-sonnet-5',
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

- `LocalWorkspace('.')` gives every run the current directory. Commands start there, and relative
  paths resolve against it.
- `ctx.workspace.run(...)` returns the command's `exit_code`, `stdout` and `stderr`. A failing command
  is a normal result, so the model sees what went wrong and can fix it.
- `timeout=60` stops a stuck command with a
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError]. Turning any
  [`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] into `ModelRetry` lets the model try
  something else instead of ending the run.

You rarely need to write these tools yourself. The [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/)
capabilities `Coder`, `FileSystem` and `Shell` give the model file, search and shell tools that work in
whatever workspace the run has.

## Move it into a sandbox

`LocalWorkspace` runs the model's commands on your machine, as you. To run them somewhere isolated,
swap it for a sandbox from the harness. Nothing else changes: the same tools now run in the sandbox.

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and run its tests.')
```

The [harness](https://pydantic.dev/docs/ai/harness/) has sandboxes for Modal, E2B and Sprites. A
sandbox is created the first time the run uses it (a tool call, or `Coder` loading repo
instructions at run start; durable runs create it when they start), and keeps running after the run until you or its
provider stop it; see [Cleaning up](#cleaning-up). If the run never used it, calling a method on `result.workspace` after a run with `ref=None` may create a new sandbox that no message records. Check the ref before using the workspace after a run, and arrange to clean up any new sandbox you create.

## Pick up where you left off {#continuing-in-the-same-workspace}

Pass the message history to the next run, and it continues in the same workspace. With a sandbox,
the files and installed packages from the first run are still there:

```python {requires="workspace_agent.py"}
from workspace_agent import agent


async def main() -> None:
    first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    await agent.run('Now add a test for it.', message_history=first.all_messages())
```

This works because each model response records the workspace it ran in, as a
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] on
[`workspace_ref`][pydantic_ai.messages.ModelResponse.workspace_ref]. The next run finds it on the
latest response and attaches to that workspace. The reference is part of the messages, so it survives
however you store them:

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
reference the run used and pass it as `workspace=`:

```python {requires="workspace_agent.py"}
from workspace_agent import agent


async def main() -> None:
    first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    ref = first.workspace.ref  # a small dataclass: `provider` and `id`

    await agent.run('Explain what fizzbuzz.py does.', workspace=ref)
```

- A sandbox's `ref` is `None` until the run first uses it, so a run that never touched the
  workspace records no reference. A later turn that doesn't use it keeps the earlier reference.
- If the sandbox a reference names no longer exists, the run's first use of it (at run start for
  durable runs) raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] instead of quietly
  starting over in an empty one. Pass `workspace='new'` to start over on purpose.
- [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] and the [UI adapters](ui/overview.md)
  strip references from history a client sends you, so a client can't choose the environment your
  agent works in. Save the reference on your server and pass it as `workspace=` instead.

## Hand the workspace to another agent

`workspace=` also takes a live workspace. Here the coding agent asks a second agent to review its
work. The reviewer reads the same files, and
[`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] stops it from changing them:

```python {title="reviewer.py" requires="workspace_agent.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import ReadOnlyWorkspace

from workspace_agent import agent

reviewer = Agent('anthropic:claude-opus-5-5', instructions='Review code. Be brief.')


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

The reviewer has no workspace capability of its own: it works wherever it is told to. The same
agent can review a local checkout in one run and a sandbox in the next.

## Your machine or a sandbox

`LocalWorkspace` is not a sandbox: a command can read and change anything you can, and the directory
you pass only sets where commands start. The directory must already exist. Use it for your own,
trusted work, and a sandbox for code you don't trust.

Commands in a `LocalWorkspace` get your `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`, so they find
host tools and configuration and use your locale. They don't inherit the rest of the agent process's
environment: pass what they need with `env=`, never the whole `os.environ`, which would hand your API
keys to the commands the model runs:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspace('.', env={'UV_OFFLINE': '1'})],
)
```

## Using the workspace in tools

`ctx.workspace` has the same methods whatever the environment:

- [`run`][pydantic_ai.workspaces.Workspace.run] runs a command.
- [`read_text`][pydantic_ai.workspaces.Workspace.read_text] and
  [`write_text`][pydantic_ai.workspaces.Workspace.write_text] read and write text files;
  [`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] and
  [`write_bytes`][pydantic_ai.workspaces.Workspace.write_bytes] do the same with exact bytes. Writing creates
  any missing parent directories.
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

Relative paths resolve against the workspace's working directory. That is a starting point, not a
boundary: `..` and absolute paths reach the rest of the environment. Prefer relative paths or
set `working_dir=` for portable commands: sandbox providers use different default users and
working directories.

A bad path raises the usual error, such as `FileNotFoundError` or `IsADirectoryError`. Catch it and
raise `ModelRetry` so the model can try again; uncaught, it ends the run, except on Temporal, where it
follows the activity retry policy (see [Durable execution](#durable-execution)). An environment that is gone,
such as a sandbox deleted during a command, raises `WorkspaceUnavailableError` and ends the run.
A command killed by a signal while its environment stays live returns its exit code instead.

## Read-only access

Pass `read_only=True` to let tools read and list files while refusing commands and changes:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace('.', read_only=True)])
```

A refused change raises [`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError].
Commands are refused too, because a command could change files. A capability can check
[`ctx.workspace.read_only`][pydantic_ai.workspaces.Workspace.read_only] to leave its write tools out.

For a single run, wrap its workspace in `ReadOnlyWorkspace` and pass it as `workspace=`, as the
[reviewer above](#hand-the-workspace-to-another-agent) does. To write your own policy, subclass
[`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace], override the operations you want to
change, and call `self.wrapped` for the rest. `run()` bypasses file-method policies (including
shell/grep tools built on it) unless the wrapper also overrides or refuses commands. Symlinks can
also lead outside a file root; check `realpath` before a write, for example:

```python
import posixpath

from pydantic_ai.workspaces import WorkspaceReadOnlyError, WrapperWorkspace


class RootedWrites(WrapperWorkspace):
    async def write_bytes(self, path: str, data: bytes) -> None:
        root = await self.wrapped.realpath(await self.working_dir())
        target = await self.wrapped.realpath(await self.resolve(path))
        if posixpath.commonpath((root, target)) != root:
            raise WorkspaceReadOnlyError('outside the allowed root')
        await self.wrapped.write_bytes(path, data)

    async def run(self, command, **kwargs):
        raise WorkspaceReadOnlyError('commands bypass file policy')
```

This is a preflight check, not a security boundary against concurrent symlink changes; use an
isolated backend for untrusted commands. On a backend without commands or `SupportsRealpath`,
`realpath` only normalizes text; see [Resolving symlinks](#resolving-symlinks).

## Choosing a run's workspace

A run picks its workspace from the first of these that applies:

1. The `workspace=` argument.
2. The reference on the latest response in `message_history`.
3. The agent's workspace capabilities: its directory for `LocalWorkspace`, a new sandbox for a sandbox
   provider.

`workspace=` takes:

- `'new'`, to start a fresh environment and ignore the one in the history.
- A [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef], such as a saved `result.workspace.ref`, for
  the agent's capabilities to attach to.
- A workspace or backend, used as is: `result.workspace`, `ctx.workspace`, or a backend such as
  `LocalWorkspaceBackend('/tmp/scratch')`.

An agent can have several workspace capabilities. Those passed to the run are asked before the
agent's own, each in order. With a reference, the first that recognizes it supplies the workspace;
without one, the first that returns a workspace does. This is how you move to a new provider without
breaking old conversations: list the new capability first, so new conversations use it, and keep the
old one, so conversations that started there continue in it.

```python {title="two_workspaces.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('~/projects/current', id='current'),  # new conversations
        LocalWorkspace('~/projects/archive', id='archive'),  # conversations that started here
    ],
)
```

A capability only attaches to references it recognizes. `LocalWorkspace` accepts a reference to its
own directory and no other, so message history can't point the agent at another directory on your
machine.

When no capability supplies a workspace:

- A reference from `message_history` raises `UserError` if the agent has workspace capabilities and
  none recognizes it, for example after you switch providers. Pass `workspace='new'`, or keep the old
  capability. An agent with no workspace capability, such as one that summarizes the conversation,
  ignores the reference.
- A `WorkspaceRef` passed as `workspace=` raises `UserError`, and so does `workspace='new'`.
- Without a reference, the run has no workspace, and tools that use it raise `WorkspaceUnavailableError`.

For file inspection without changes, use a read-only workspace and read tools. To answer
without file access, use a no-file-tools agent instead: workspace-backed tools need an attached
workspace, and `UnavailableWorkspace` deliberately refuses all operations.

## Cleaning up

Pydantic AI never deletes an environment: a sandbox keeps running after the run until you delete it
or its provider's lifetime settings stop it. Each sandbox page shows how to delete one.

A successful run returns `result.workspace.ref`, so you can continue in the sandbox or delete it
later. A failed run returns no result, so delete its sandbox in an `on_run_error` hook, where
`ctx.workspace.ref` names it. `after_run` doesn't run when a run fails. If you delete the sandbox after each run, pass `workspace='new'` when continuing its history; otherwise the next run raises `WorkspaceUnavailableError` because the recorded environment is gone.

## Durable execution

Under [Temporal](durable_execution/temporal.md), [DBOS](durable_execution/dbos.md) or
[Prefect](durable_execution/prefect.md), attach the workspace capability when you construct the
agent, next to the durability capability:

```python {title="durable_workspace.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability

agent = Agent(
    'anthropic:claude-sonnet-5',
    name='fizzbuzz',  # names the agent's Temporal activities
    capabilities=[LocalWorkspace('.'), TemporalDurability()],
)
```

Tools use `ctx.workspace` as they would in a plain run. On Temporal, a tool that must run
its orchestration in the workflow (for example, to approve a file change before writing) can
set `metadata={'temporal': False}` on its tool definition; workspace calls it makes still run as
activities.

- Inside a workflow or flow, a run creates or attaches its environment when it starts, even if no
  tool uses it; retries, replays and recovery reattach to that same environment. Plain runs stay
  lazy. If a worker dies after creating the environment but before that is recorded, the retry can
  create a second one.
- Without an explicit `run_id`, a run of an agent with a workspace capability inside a workflow or flow gets an ID derived from the Temporal
  execution run ID, the DBOS workflow ID or the Prefect flow run ID, so its workspace state stays
  addressable after a worker restart or flow retry.
- On DBOS, a run with a workspace runs its tool calls one at a time: DBOS numbers steps as they
  start, so parallel tools making several workspace calls each could not be replayed reliably.
- A workspace call may run again if a worker dies mid-call: inside a tool it retries with the tool,
  and from workflow code it follows the engine's activity or step settings. On Temporal and Prefect,
  workspace timeouts, output-limit failures, read-only refusals and a lost environment are never
  retried. Other exceptions, such as `FileNotFoundError`, follow the retry policy, which on Temporal
  retries without limit by default: catch them and raise `ModelRetry`, or set `maximum_attempts`.
- On Temporal, a command runs within an activity's `start_to_close_timeout`, 60 seconds by default,
  whatever its own `timeout`. Inside a tool that is the tool's activity; raise it with
  `metadata={'temporal': ActivityConfig(start_to_close_timeout=...)}`. From workflow code it is
  `activity_config`; if you set that, keep a `start_to_close_timeout` in it, for example
  `activity_config={'start_to_close_timeout': timedelta(seconds=60), 'retry_policy': RetryPolicy(maximum_attempts=3)}`.
- `workspace=` passes on only a reference, and the run rebuilds the workspace from the agent's own
  capabilities. A `ReadOnlyWorkspace(...)` argument raises `UserError` if rebuilding would drop its
  read-only policy, and so does a live backend that has no ref yet or a capability passed to the run
  that changes the workspace's type or policy. Other settings of a run-level workspace capability,
  such as a sandbox image, do not apply: every unit rebuilds the workspace from the agent's. A previous `result.workspace` without a ref starts a fresh workspace. Put policy on the agent's capability instead, such as
  `LocalWorkspace(..., read_only=True)`.
- `workspace.backend` is not available in workflow code, which includes DBOS function tools and
  Temporal tools with `metadata={'temporal': False}`. Reach the provider's own API from a tool that
  runs as an activity or task.
- Inside a workflow or flow, the deprecated `TemporalAgent`, `DBOSAgent` and `PrefectAgent` wrappers
  refuse a workspace; use the durability capability instead.
- On Temporal, `LocalWorkspace` calls run in activities on whichever worker picks them up, so every
  worker needs the directory at the same absolute path, on storage they share.

!!! warning "Adding a workspace with runs in flight"
    Adding a workspace to an agent changes what its runs record, so runs started before the change no
    longer match their history: Temporal replay fails with a nondeterminism error, and DBOS can't
    recover the workflow. Let in-flight runs finish first, or deploy the change separately: with
    Temporal worker versioning or a new task queue, or a new DBOS application version. On Prefect,
    let running flows finish before you deploy.

The engine guides cover the rest, such as matching capabilities across Temporal workers.

## Supplying a workspace from a capability

To supply a workspace from your own capability, implement
[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace]. This one points each
user at their own directory, named by the user ID the run gets as `deps`:

```python {title="per_user_workspace.py"}
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceBackend, WorkspaceRef


@dataclass
class UserDirectory(AbstractCapability[str]):
    base_dir: Path

    def get_workspace(self, ctx: RunContext[str], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if not ctx.deps.isalnum():  # keeps an ID like '../other-user' from leaving base_dir
            raise ValueError(f'Invalid user ID: {ctx.deps!r}')
        backend = LocalWorkspaceBackend(self.base_dir / ctx.deps)
        if ref is not None and ref != backend.ref:
            return None  # not this user's directory
        return backend
```

`LocalWorkspaceBackend` never creates its directory, and every operation, absolute paths included,
raises `WorkspaceUnavailableError` until it exists, so create each user's directory when you create the user. This separates where users start; like `LocalWorkspace`, it isolates nothing.

- With `ref=None`, return the backend for a new or default environment.
- With a `ref` you recognize, return a backend that attaches to it. Return `None` for any other.
  Ref ids can come from stored or untrusted history: validate them before using them as paths or identifiers.
- To apply a policy, return a `Workspace` around the backend, such as
  `ReadOnlyWorkspace(Workspace(backend))`.
- Don't do I/O or keep state in `get_workspace`: it can be called more than once per run. Connect on
  the backend's first operation.
- A capability that supplies a workspace can't use `defer_loading=True`: the workspace is chosen when
  the run starts.

A capability that needs a workspace can check
[`ctx.workspace.attached`][pydantic_ai.workspaces.Workspace.attached] in `before_run`. It is `False`
when nothing supplied a workspace, so the run fails at its start, naming what to attach, instead of on
the first tool call:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError


class ProjectNotes(AbstractCapability):
    async def before_run(self, ctx: RunContext) -> None:
        if not ctx.workspace.attached:
            raise UserError("`ProjectNotes` needs a workspace. Attach one, such as `LocalWorkspace('.')`.")
```

Realtime sessions do not currently select workspaces, even when the agent has a workspace capability.
Workspace tools in a realtime session report this limit instead of suggesting a second capability.

For tests, `with agent.override(workspace=LocalWorkspaceBackend(path)):` temporarily uses a local
workspace instead of the agent's capability, the ref in history, or a per-run `workspace=`, as
`override(model=)` does for the model.

## Writing a backend

A backend is the object that talks to one environment; `Workspace` wraps it to give tools the
`ctx.workspace` methods. A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements
`ref` and `working_dir`, then adds [`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. With commands only,
`ctx.workspace` derives the file operations through the shell. With a filesystem only, file tools work
and `ctx.workspace.run` raises `UserError`. Shell-derived reads require regular files (not FIFOs or devices). They transfer large files and directory listings in bounded chunks, but shell-derived file operations are slower than native provider file APIs. Local and shell-derived file operations refuse to remove the workspace root or an ancestor, and local writes do not recreate a removed workspace directory. `LocalWorkspace` continues only in a ref naming its configured directory exactly as written, so a ref spelled through a symlink is declined.
With both, they must reach the same environment. A filesystem-only backend whose storage can hold
symlinks should also implement `SupportsRealpath`; see [Resolving symlinks](#resolving-symlinks).

### Resolving symlinks

`Workspace.realpath` answers "which file does this path really lead to?". Code that keeps an agent
inside a directory needs that answer, because a symlink can make a path that looks inside lead
outside. File methods open [`resolve(path)`][pydantic_ai.workspaces.Workspace.resolve], which collapses
`..` as text, so to check the file a file method opens, use `realpath(await ws.resolve(path))`, as the
`RootedWrites` example above does. `realpath(path)` on the raw path follows the kernel, where `..`
climbs from a symlink's target, so it answers for commands. For example, with a check that only allows paths under `/app` and a link
`/app/data -> /secrets`, `/app/data/key` looks inside, but opens `/secrets/key`.

Backends answer it in one of three ways:

- `LocalWorkspaceBackend` uses `os.path.realpath`.
- A backend with commands but without [`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath]
  gets a shell fallback that follows links with `readlink`, in one command.
- A backend with neither only gets the path normalized as text. That is exact for storage that
  cannot hold symlinks, such as an object store. Where symlinks can exist, a path check such as the
  harness `FileSystem`'s `root_dir` then sees only the text, and the example above leads outside.

The check and the later file operation are separate calls, so a link swapped in between them is not
caught; for untrusted code, rely on the sandbox, not on path checks.

A backend for a cloud sandbox has this shape (`sandbox_sdk` stands in for the provider's SDK):

```python {title="my_sandbox.py" test="skip" lint="skip"}
from collections.abc import Mapping

import anyio
import sandbox_sdk

from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceUnavailableError,
)


class MySandbox(WorkspaceBackend, SupportsCommands):
    def __init__(self, ref: WorkspaceRef | None = None):
        self._ref = ref
        self._sandbox: sandbox_sdk.Sandbox | None = None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def _connect(self) -> sandbox_sdk.Sandbox:
        async with self._lock:  # concurrent first operations share one sandbox
            if self._sandbox is None:
                if self._ref is None:
                    self._sandbox = await sandbox_sdk.Sandbox.create()
                    self._ref = WorkspaceRef(provider='my-sandbox', id=self._sandbox.id)
                else:
                    try:
                        self._sandbox = await sandbox_sdk.Sandbox.connect(self._ref.id)
                    except sandbox_sdk.NotFound as error:
                        raise WorkspaceUnavailableError(f'sandbox {self._ref.id!r} no longer exists') from error
            return self._sandbox

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        sandbox = await self._connect()
        result = await sandbox.exec(command, shell=shell, env=env, timeout=timeout)
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    async def working_dir(self) -> str:
        return (await self._connect()).working_dir
```

- The constructor does no I/O. The first operation creates the environment, or attaches to the one
  `ref` names.
- `ref` is `None` until the environment exists, then names it and never changes. Set it as soon as
  the create call returns, before any setup that can fail, so a failed setup still leaves a ref to
  clean up. Durable execution needs it set once any operation has completed.
- Concurrent first operations must create only one environment and share its ref. `Workspace` holds
  no lock, so the backend owns its get-or-create lock.
- A caller cancelled while the environment is being created must not lose its ref: finish recording
  it even though the caller has gone. The example above skips this; the harness sandboxes show one way.
- A `ref` whose environment is gone raises `WorkspaceUnavailableError`. The backend never creates a
  replacement for it.
- The backend never destroys the environment when a run ends. Whoever holds the ref decides when to
  delete it; see [Cleaning up](#cleaning-up).
- A non-zero exit is a result, not an exception. A program that can't start is exit 127 (not found)
  or 126 (not executable), as in `sh`. A path-level failure raises the builtin file error, and a
  timeout raises `WorkspaceTimeoutError`. Let a provider SDK's own transient errors propagate, so a
  durable engine can retry them.

A capability's `get_workspace` returns this backend, and users reach its provider-specific methods
through `ctx.workspace.backend`. This is a provider API escape hatch and bypasses wrapper policies such as
`read_only`; use `ctx.workspace` file/command methods when policies must apply.

### Checking a backend

Subclass [`WorkspaceBackendSuite`][pydantic_ai.workspaces.conformance.WorkspaceBackendSuite] in your
pytest suite and provide its `backend` fixture. Each test checks one rule of the backend contract:

```python {test="skip" lint="skip"}
import pytest
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite


class TestMyBackend(WorkspaceBackendSuite):
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls, tmp_path_factory: pytest.TempPathFactory) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(working_dir=tmp_path_factory.mktemp('ws'))
```

The suite needs the anyio pytest plugin, which runs each rule on every installed async backend (asyncio,
and Trio when installed). For an asyncio-only SDK, override the `anyio_backend` fixture to return
`'asyncio'`, class-scoped when your `backend` fixture is. A class-scoped fixture starts one environment for the whole
suite instead of one per rule. Under durable execution every workspace call rebuilds your backend from
its ref, so provide `attach_backend`, a factory that builds a backend for a ref the way your capability's
`get_workspace` does: it enables the rule that a backend attached by ref reaches the same files and
commands and keeps that ref. To check destruction,
also provide `destroy_environment` and `destructive_backend`, a factory for an independent environment.
The latter defaults to `fresh_backend` when supplied. The destructive rule never uses the shared `backend` fixture.

Before shipping a backend, use the suite to check:

- Concurrent first use creates one environment (supply `fresh_backend` for this rule), with a stable ref.
- Command errors use the right types: invalid arguments, missing paths and unavailable environments;
  timeouts raise `WorkspaceTimeoutError`; stdin at EOF and non-zero exits remain normal results.
- Cancellation stops the foreground command, and background children holding output pipes do not
  indefinitely delay a finished command.
- File operations preserve bytes, follow symlinks where appropriate, and report path errors;
  understand containment limits: a working directory is not a jail, and symlinks may cross roots.
- Supply `attach_backend`, `destroy_environment`, and an independent `destructive_backend` to check
  reattachment, a destroyed reference, and destruction during a command. These optional rules
  skip when their fixtures are absent.

A skipped rule is not a passed rule: run pytest with `-rs` to see what went unchecked. Four boolean
fixtures, all `True` by default, declare a known limit; return `False` only for a limit you also
document for users:

- `can_detect_exit_with_inherited_output_pipes`: the provider reports a command's exit only after
  every process holding its output has exited.
- `filesystem_honors_shell_permissions`: the provider's file API bypasses the command user's permissions.
- `has_real_posix_shell` and `enforces_parent_file_errors`: for in-memory test doubles only.

The suite's class, fixtures, switches and rule names are stable in 2.x, so renaming one is a breaking
change. New or stricter rules may be added in minor releases; pin `pydantic-ai` if a new rule failing
your CI is a problem.

## Timeouts and clocks

`run(timeout=...)` starts its clock after the sandbox is ready (and, locally, after resolving the
working directory). Local process startup counts as command time. The deadline covers the command only;
`timeout=None` has no command deadline. A provider's sandbox lifetime and idle limits are
separate. Commands receive stdin at EOF, so use non-interactive flags (such as `-y`). On its
own deadline the foreground command is stopped and `WorkspaceTimeoutError` carries any partial
`stdout` and `stderr` collected so far. For the local backend, the command deadline also
bounds subprocess startup; even with `timeout=None`, startup and process reaping have finite
safety bounds.

## Security choices

`LocalWorkspace` has the full authority of the host user. Neither `working_dir` nor a harness
`FileSystem` root jails shell commands. Do not pass `os.environ` to untrusted workspaces:
explicitly choose the variables the command needs. Arguments to workflow-side workspace calls,
including `env=` and file contents, are stored in durable history; do not pass secrets there
without a suitable payload codec.

Agent run spans record `pydantic_ai.workspace.id` (the sandbox id, or the directory path for
`LocalWorkspace`) even with `include_content=False`, because it identifies the environment, not content.

## Platforms

Local background jobs outlive `run()`; redirect their output to a file to avoid the two-second drain grace when they inherit stdout or stderr. The caller must clean up jobs when the host exits.

Local commands that exceed the 10 MiB combined output limit raise `WorkspaceOutputLimitError`, with the first 64 KiB of each stream in `stdout` and `stderr`. Redirect large output to a file instead.

The local backend and command-backed shell fallback require POSIX. A filesystem-only backend
works without shell support, but cannot run commands.

## Limits

- `LocalWorkspace` runs only on POSIX systems (macOS and Linux) and isolates nothing. Create its root
  first: every operation needs it. While it exists, absolute paths outside it are allowed.
  `defer_loading=True` is rejected because the workspace must be selected before deferred
  capabilities load. Its ref normalizes `.` and `..` without resolving symlinks, so differently
  spelled symlink roots have distinct refs even if they point to the same directory.
- A `LocalWorkspace` backend serializes its own reads and writes of one file; other runs, processes and commands can still interleave with them.
- A run has one workspace.
- Non-durable runs do not create or delete sandboxes solely at run boundaries; durable runs eagerly create or attach an environment at their start, even without tool use. In either case, deletion remains the caller's job.
- How a timed-out command is stopped depends on the provider.
