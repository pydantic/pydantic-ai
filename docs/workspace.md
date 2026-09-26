# Workspaces

A workspace is a computer your agent can use: it runs commands and reads and writes files there.
It can be a directory on your machine or a sandbox in the cloud, and your tools don't need to know
which, because they all use it through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace].

## Give an agent a computer

```python {title="workspace_agent.py"}
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
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
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
sandbox is created the first time a tool uses it, and keeps running after the run until you or its
provider stop it; see [Cleaning up](#cleaning-up).

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

- A sandbox's `ref` is `None` until a tool first uses it, so a run whose tools never touched the
  workspace records no reference. A later turn that doesn't use it keeps the earlier reference.
- If the sandbox a reference names has been deleted or has expired, the run raises
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

Commands in a `LocalWorkspace` get your `PATH` and `HOME`, so they find your tools and your git and
package-manager configuration. Nothing else from your environment reaches them. Pass other variables
with `env=`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspace('.', env={'UV_OFFLINE': '1'})],
)
```

!!! warning
    Don't pass `os.environ` itself: that hands the model's commands every secret in the process,
    LLM API keys included.

## Using the workspace in tools

`ctx.workspace` has the same methods whatever the environment:

- [`run`][pydantic_ai.workspaces.Workspace.run] runs a command.
- [`read_text`][pydantic_ai.workspaces.Workspace.read_text] and
  [`write_text`][pydantic_ai.workspaces.Workspace.write_text] read and write text files;
  [`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] and
  [`write_bytes`][pydantic_ai.workspaces.Workspace.write_bytes] do the same with exact bytes.
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
boundary: `..` and absolute paths reach the rest of the environment.

A bad path raises the usual error, such as `FileNotFoundError` or `IsADirectoryError`. Catch it and
raise `ModelRetry` so the model can try again; uncaught, it ends the run. An environment that is gone,
such as a deleted sandbox, raises `WorkspaceUnavailableError` and ends the run.

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
change, and call `self.wrapped` for the rest.

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
- Without a reference, the run has no workspace, and tools that use it raise `UserError`.

To answer a question without letting tools touch files, pass an
[`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace]. Every workspace operation then
raises `WorkspaceUnavailableError` with the reason you give:

```python {requires="workspace_agent.py"}
from pydantic_ai.workspaces import UnavailableWorkspace

from workspace_agent import agent


async def main() -> None:
    no_files = UnavailableWorkspace(reason='This run has no file access.')
    await agent.run('Explain what fizzbuzz.py does.', workspace=no_files)
```

## Cleaning up

Pydantic AI never deletes an environment: a sandbox keeps running after the run until you delete it
or its provider's lifetime settings stop it. Each sandbox page shows how to delete one.

A successful run returns `result.workspace.ref`, so you can continue in the sandbox or delete it
later. A failed run returns no result, so delete its sandbox in an `on_run_error` hook, where
`ctx.workspace.ref` names it. `after_run` doesn't run when a run fails.

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

Tools use `ctx.workspace` as they would in a plain run.

- Every run creates or attaches its environment at its start, even if no tool uses it. Retries,
  replays and recovery then reattach to that same environment.
- Workspace calls retry like tools do, so a command or write may run again if a worker dies mid-call,
  and a `run(timeout=...)` must fit within the call's own timeout (Temporal's `start_to_close_timeout`).
- `workspace=` passes on only a reference, and the run rebuilds the workspace from the agent's own
  capabilities. A wrapper such as `ReadOnlyWorkspace(...)` passed to the run is dropped, and a
  capability passed to the run that changes the workspace raises `UserError`. Put policy on the
  agent's capability instead, such as `LocalWorkspace(..., read_only=True)`.
- `workspace.backend` is not available in workflow code. Reach the provider's own API from a tool.
- The deprecated `TemporalAgent`, `DBOSAgent` and `PrefectAgent` wrappers refuse a workspace.

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
        backend = LocalWorkspaceBackend(self.base_dir / ctx.deps)
        if ref is not None and ref != backend.ref:
            return None  # not this user's directory
        return backend
```

`LocalWorkspaceBackend` never creates its directory, so create each user's directory when you create
the user. This separates where users start; like `LocalWorkspace`, it isolates nothing.

- With `ref=None`, return the backend for a new or default environment.
- With a `ref` you recognize, return a backend that attaches to it. Return `None` for any other.
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

## Writing a backend

A backend is the object that talks to one environment; `Workspace` wraps it to give tools the
`ctx.workspace` methods. A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements
`ref` and `working_dir`, then adds [`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. With commands only,
`ctx.workspace` derives the file operations through the shell. With a filesystem only, file tools work
and `ctx.workspace.run` raises `UserError`. With both, they must reach the same environment.
Implement [`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath] if your platform can resolve
symlinks natively; with commands only, `realpath` uses the shell, and with neither, symlinks aren't
resolved, so a root-directory check such as the harness `FileSystem`'s is textual only.

This backend gives each environment its own directory under `base_dir`:

```python {title="host_workspace.py"}
import uuid
from collections.abc import Mapping
from pathlib import Path

import anyio

from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceUnavailableError,
)


class HostWorkspaceBackend(WorkspaceBackend):
    def __init__(self, base_dir: Path, ref: WorkspaceRef | None = None):
        self._base_dir = base_dir
        self._ref = ref
        self._local: LocalWorkspaceBackend | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def _directory(self) -> LocalWorkspaceBackend:
        if self._local is None:
            if self._ref is None:
                # No reference: create the environment, then report its identity.
                directory = anyio.Path(self._base_dir / uuid.uuid4().hex)
                await directory.mkdir()
                self._ref = WorkspaceRef(provider='host', id=directory.name)
            else:
                # A reference: attach to the environment it names, or fail. Never create a replacement.
                directory = anyio.Path(self._base_dir / self._ref.id)
                if not await directory.is_dir():
                    raise WorkspaceUnavailableError(f'workspace {self._ref.id!r} no longer exists')
            self._local = LocalWorkspaceBackend(Path(directory))
        return self._local

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        local = await self._directory()
        return await local.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        local = await self._directory()
        return await local.working_dir()
```

- The constructor does no I/O. The first operation creates the environment, or attaches to the one
  `ref` names.
- `ref` is `None` until the environment exists, then names it. Durable execution needs it set once
  any operation has completed.
- A `ref` whose environment is gone raises `WorkspaceUnavailableError`. The backend never creates a
  replacement for it.
- A non-zero exit is a result, not an exception. A program that can't start is exit 127 (not found)
  or 126 (not executable), as in `sh`. A path-level failure raises the builtin file error, and a
  timeout raises `WorkspaceTimeoutError`. Let a provider SDK's own transient errors propagate, so a
  durable engine can retry them.

A capability's `get_workspace` returns this backend, and users reach its provider-specific methods
through `ctx.workspace.backend`.

### Checking a backend

Subclass [`WorkspaceBackendSuite`][pydantic_ai.workspaces.testing.WorkspaceBackendSuite] in your
pytest suite and provide its `backend` fixture. Each test checks one rule of the backend contract:

```python {test="skip" lint="skip"}
import pytest
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai.workspaces.testing import WorkspaceBackendSuite


class TestMyBackend(WorkspaceBackendSuite):
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls, tmp_path_factory: pytest.TempPathFactory) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(working_dir=tmp_path_factory.mktemp('ws'))
```

The suite needs the anyio pytest plugin. A class-scoped fixture starts one environment for the whole
suite instead of one per rule. Provide the optional `attach_backend` and `destroy_environment`
fixtures to enable the reattachment rules.

## Limits

- `LocalWorkspace` isolates nothing, never creates its directory, and runs only on POSIX systems
  (macOS and Linux).
- A run has one workspace.
- Pydantic AI never creates or deletes a sandbox at run boundaries: cleanup is yours.
- How a timed-out command is stopped depends on the provider.
- Under durable execution, `workspace=` passes on only a reference: per-run wrappers such as
  `ReadOnlyWorkspace` don't apply.
