# Workspaces

A workspace is where an agent does its work: an environment in which it can run commands and read
and write files. Tools, and hooks that receive a [`RunContext`][pydantic_ai.tools.RunContext], use it
through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace].

Here is an agent that can run commands in the current directory:

```python {title="workspace_agent.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspace('.')],
)


@agent.tool
async def execute(ctx: RunContext, command: list[str]) -> str:
    """Run a command in the project directory."""
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
```

- `LocalWorkspace('.')` gives every run of the agent the current directory as its workspace.
  Commands start there, and relative file paths resolve against it.
- `ctx.workspace.run(...)` runs a command and returns its `exit_code`, `stdout` and `stderr`. A command
  that fails is a normal result, so the tool can show the model what went wrong.
- `timeout=60` stops a command that runs too long and raises
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError].

You rarely need to write these tools yourself: the [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/)
capabilities, such as `Coder`, `FileSystem` and `Shell`, give the model file, search and shell tools
that all work in the run's workspace.

## Your machine or a sandbox

`LocalWorkspace` runs commands on your own machine, as your user. It is not a sandbox: a command can
read and change anything you can, and `working_dir` only sets where commands start. Use it for your
own, trusted work.

To run code the model writes in isolation, attach a sandbox capability from the harness instead, such
as Modal, E2B, Daytona or Sprites. Nothing else about the agent changes: tools keep using
`ctx.workspace`. An agent has one workspace capability; attaching a second raises `UserError`.

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

```python {title="file_tools.py"}
from pydantic_ai import RunContext


async def read_source(ctx: RunContext, path: str) -> str:
    """Read a text file from the project."""
    return await ctx.workspace.read_text(path)


async def save_notes(ctx: RunContext, notes: str) -> str:
    """Save notes for later steps."""
    await ctx.workspace.write_text('NOTES.md', notes)
    return 'Saved to NOTES.md.'
```

A missing file raises `FileNotFoundError`, and a directory where a file was expected raises
`IsADirectoryError`, as they would locally. Relative paths resolve against the workspace's working
directory. That is only a starting point, not a boundary: `..` and absolute paths reach the rest of
the environment.

## Read-only access

Pass `read_only=True` to let tools read and list files while refusing commands and changes:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace('/srv/data', read_only=True)])
```

A refused change raises [`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError],
and [`ctx.workspace.read_only`][pydantic_ai.workspaces.Workspace.read_only] lets a capability leave its
write tools out. Commands are refused too, because a command could change files.

To make a single run read-only, wrap its workspace in
[`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] and pass it to the run:

```python {requires="workspace_agent.py"}
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace

from workspace_agent import agent


async def main() -> None:
    workspace = ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend('/srv/project')))
    await agent.run('Explain what fizzbuzz.py does.', workspace=workspace)
```

[`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace] is the base class for writing your own
policy wrapper: override the operations you want to change and call `self.wrapped` for the rest.

## Continuing in the same workspace

Each response records where the run worked, as `workspace_ref`. A later run given the same
`message_history` continues in that workspace, so files the agent made are still there:

```python {requires="workspace_agent.py"}
from workspace_agent import agent


async def main() -> None:
    first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
    await agent.run('Now add a test for it.', message_history=first.all_messages())
```

- Pass `workspace='new'` to start in a fresh environment instead, ignoring the one in the history.
- Pass `result.workspace` as `workspace=` to work in the same environment without the history, or
  pass `ctx.workspace` to a subagent's run so it works where its parent does.
- Pass a backend, such as `LocalWorkspaceBackend('/tmp/scratch')`, as `workspace=` to choose the
  workspace for one run.

Pydantic AI never creates or deletes a sandbox at run boundaries. A sandbox capability creates one
on the first operation that needs it, and its provider's own lifetime settings decide when it stops.

## Durable execution

Under [Temporal](durable_execution/temporal.md), [DBOS](durable_execution/dbos.md) or
[Prefect](durable_execution/prefect.md), a workspace supplied by a capability works everywhere a
plain run's does, and workspace I/O never runs in workflow code. Attach the capability when the agent
is constructed, so the durability capability can register one durable unit per `Workspace` method:

```python {title="durable_workspace.py" test="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability


class ShareTree(AbstractCapability):
    async def before_run(self, ctx: RunContext) -> None:
        # Runs in workflow code: this write is one durable unit.
        await ctx.workspace.write_text('TASK.md', 'Summarize the repository.')


agent = Agent(
    'anthropic:claude-sonnet-5',
    name='summarizer',
    capabilities=[ShareTree(), LocalWorkspace('~/project'), TemporalDurability()],
)


@agent.tool
async def read_task(ctx: RunContext) -> str:
    # Runs inside a durable unit: the workspace is used directly.
    return await ctx.workspace.read_text('TASK.md')
```

Inside a durable container, [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace] and
[`result.workspace`][pydantic_ai.agent.AgentRunResult.workspace] are a wrapper around the selected
workspace whose operations each run as their own durable unit (a Temporal activity, a DBOS step, a
Prefect task). Errors the workspace raises, such as a missing file, a read-only refusal or a timeout, come
back as the same exception types. Inside a durable unit, such as a tool, `ctx.workspace` is the
workspace itself: on Temporal it is rebuilt inside the activity from the run's serialized
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] through the same capabilities, wrappers such
as `LocalWorkspace(..., read_only=True)` included; on DBOS and Prefect it is the run's live workspace.

One `ensure` unit runs at the start of every run in a container. It forces the environment to exist
(so a run provisions one even if no tool ends up using it) and records its `WorkspaceRef` and
canonical working directory. From then on every unit of the run carries the same ref, so parallel
tool calls, retries, replay and recovery all reattach to one environment, and `working_dir()` and
`resolve()` answer from the recorded value without a unit. A backend used under durable execution
must therefore report a `ref` once any operation has completed. `resolve()` with an absolute path
stays local and does not consult an inner wrapper's override; the next operation resolves the path
inside the unit either way.

A durable unit can run more than once if the process fails between the side effect and its
checkpoint. Reads keep the engine's retry policy; `run`, `write_bytes`, `write_text`, `make_dir`
and `remove` are attempted once by default, so a command or write is never repeated by a retry.
Each engine's `workspace_*_config` knob changes that; see the engine guides.

Inside a container, `workspace=` accepts `None`, `'new'`, a `WorkspaceRef`, a previous result's
workspace, or a live instance whose ref an attached capability recognizes; the run then uses the
capability-built workspace for that environment. Any other live backend or wrapper raises
`UserError`: it cannot cross the durable boundary, and a wrapper applied around the argument, such
as `ReadOnlyWorkspace(...)`, would not be reapplied on the other side. Policy belongs on the
capability. To share an environment between durable agents, give them the same workspace capability
and pass `result.workspace` (or its ref) along. `workspace.backend` is not available in workflow
code, as calling the provider directly would bypass durability; reach it from a tool.

The deprecated `TemporalAgent`, `DBOSAgent` and `PrefectAgent` wrappers have no durability
capability and refuse a workspace inside their container.

## Supplying a workspace from a capability

To supply a workspace from your own capability, implement
[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace]. Return a backend
configured from the capability's settings: with `ref=None` it creates a fresh environment on its first
operation, and with a `ref` it attaches to that environment. Return `None` for a ref from another
provider.

`get_workspace` must not do I/O or keep state. Under a durable engine it is called in workflow code,
where I/O is not allowed, and again inside each activity to rebuild the workspace from its ref. The
backend connects on its first operation instead.

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

### Workspace references {#workspace-references}

A [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] names an environment that exists, and it
exists only once the environment does:

- A backend built without a reference reports `ref` as `None`. Its first operation creates the
  environment, and the backend sets `ref` as soon as the creation call returns. A run whose tools
  never touch a fresh workspace therefore ends without a reference, because nothing was created.
- A backend built with a reference is bound to that environment. Its first operation attaches, and
  if the environment is gone (deleted, expired, or unknown to the provider) that operation raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError]. It never
  creates a replacement, so a stale reference in message history fails loudly instead of quietly
  starting over; pass `workspace='new'` to start over on purpose.
- A reference is never a label. A backend does not derive one from a name, a conversation id or a
  random id ahead of creating the environment, and reports one it was not given only after the
  environment exists.

When a run ends, the workspace's `ref` at that point is recorded as `workspace_ref` on the run's
last `ModelResponse`, which is what lets the next run with the same `message_history` continue in
the same environment, and what `result.workspace.ref` hands to another process.
[`sanitize_messages`][pydantic_ai.messages.sanitize_messages] strips it from client-supplied
history by default.

A run continues in the environment named by the latest response's `workspace_ref`. A run on an agent
without that provider's capability records no reference, and that latest `None` hides the older one
from later runs. To continue after such a run, pass the earlier run's `result.workspace` (or its ref)
as `workspace=`.

A local workspace is the one case where the reference precedes any operation: its directory is the
environment, so `WorkspaceRef(provider='local', id=...)`, where `id` is the absolute `working_dir`
with `~` expanded, is known from construction, and responses from a local run record which
directory they worked in. The first operation still checks the environment: `working_dir()` raises
`WorkspaceUnavailableError` when the directory does not exist, and nothing creates it.
[`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] claims that reference only when it names
its own `working_dir`, which is how a run continued from message history lands in the same
directory. It declines a local reference for any other directory, so message history can never point
the agent at an arbitrary directory on the host; that run gets no workspace unless you pass
`workspace='new'`.

To disable workspace access explicitly, pass an
[`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] as `workspace=`:

```python
from pydantic_ai.workspaces import UnavailableWorkspace

disabled = UnavailableWorkspace(reason='Workspace access is disabled for this run.')
```


### Writing a backend

A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements `ref` and `working_dir`,
then adds [`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. `Workspace` adds path
resolution and text helpers. For a command-only backend it derives file operations
through shell commands. A filesystem-only backend works without a shell; calling `ctx.workspace.run` on it
raises `UserError`. When both capabilities are present, commands and file operations must
use the same environment.

The constructor takes the backend's configuration plus an optional reference and does no I/O; the
first operation creates or attaches, following the [reference rules](#workspace-references). This
backend gives each environment its own directory on the host under `base_dir`:

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

Reading `ref` does not run a command. Keep the concrete backend when you need provider-specific
methods: `Workspace.backend` reaches the wrapped backend.

Pydantic AI does not create or destroy environments at run boundaries. Provision and clean them up
in `before_run`, `after_run`, or `wrap_run` hooks, or with the provider's SDK. Pass
`result.workspace` to a later run, or `ctx.workspace` to a subagent, to keep using the same
environment.

#### Checking a backend

Subclass [`WorkspaceBackendSuite`][pydantic_ai.workspaces.testing.WorkspaceBackendSuite] in your
pytest suite and provide its `backend` fixture:

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

The fixture can be synchronous or asynchronous, and function- or class-scoped. A class-scoped
fixture means a remote backend does not start a sandbox per rule. Write it as a `classmethod`, as
above: pytest requires that for class-scoped fixtures on a class. The ref rule accepts a ref that
already exists, and the destroy rule runs last.

The suite checks these rules. Command rules skip when the backend does not implement
`SupportsCommands`. Filesystem rules run against the file operations `Workspace` derives through
the shell when the backend only implements `SupportsCommands`, and skip only when it implements
neither protocol. Reattachment rules skip until their fixtures are provided.

- `test_required_members`: The backend provides the structural `WorkspaceBackend` members.
- `test_string_command_requires_shell`: A string command without `shell=True` raises `TypeError`.
- `test_argv_command_rejects_shell`: An argv command with `shell=True` raises `TypeError`.
- `test_relative_cwd_is_rejected`: A relative `cwd` raises `ValueError`.
- `test_shell_result_is_honest`: A shell result carries the real `exit_code`, `stdout`, and `stderr`.
- `test_argv_arguments_are_literal`: Argv items are passed literally, without shell interpretation.
- `test_default_working_dir_is_canonical`: `working_dir` is an absolute, canonical POSIX path.
- `test_timeout_raises_workspace_timeout_error`: A timeout raises `WorkspaceTimeoutError` carrying the enforced deadline, which is at least the requested one.
- `test_env_is_added`: Extra environment variables reach the command.
- `test_absolute_cwd_is_used`: An absolute `cwd` is honored.
- `test_ref_is_stable_across_operations`: Once assigned, `ref` remains stable across operations.
- `test_filesystem_bytes_round_trip`: Byte writes, replacements, and reads round trip exactly.
- `test_filesystem_exists_is_truthful`: `exists` distinguishes present and absent paths.
- `test_filesystem_entries_are_truthful`: `stat` and `list_dir` return truthful, non-recursive entries.
- `test_filesystem_make_dir_has_mkdir_p_semantics`: `make_dir` creates parents and is idempotent.
- `test_run_and_filesystem_share_one_environment`: Commands and filesystem methods see the same files.
- `test_filesystem_missing_paths_raise_file_not_found`: Missing paths raise `FileNotFoundError` from `read_bytes`, `stat`, `list_dir`, and `remove`.
- `test_filesystem_reading_directory_raises_is_a_directory`: Reading a directory raises `IsADirectoryError`.
- `test_filesystem_remove_file_and_tree`: `remove` deletes a file or a directory tree.
- `test_realpath_resolves_symlinks`: `Workspace.realpath` follows a symlink the command created, and keeps a missing tail under a symlinked directory as written.
- `test_ref_is_none_until_the_environment_exists_then_stable`: A lazily created environment gets a stable, non-`None` ref on first use.
- `test_reattach_by_ref_sees_the_same_files`: A second backend attached by ref sees the same files.
- `test_reattach_after_destroy_raises_unavailable`: A ref to a destroyed environment raises `WorkspaceUnavailableError` on use.

### Errors

The exception a workspace raises says whether the environment is still usable, so a backend must
raise the right one in each case (this is the contract capabilities and the durable engines rely
on; the full list is on the [`pydantic_ai.workspaces`][pydantic_ai.workspaces] module):

- A non-zero command exit is a normal result reported on `exit_code`, not an exception.
- An environment that is gone or unreachable raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError], including on the
  first operation of a backend whose reference no longer names a live environment. Retrying cannot
  help, and a run cannot continue in that environment, so this error ends the agent run rather than
  reaching the model.
- A command that exceeds its deadline raises
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] with the output received so
  far. How the command is terminated depends on the provider.
- A path-level failure raises the builtin file error: a missing file `FileNotFoundError`, a
  directory where a file was expected `IsADirectoryError`, and so on. The environment itself is
  fine, so a tool can report the failure to the model and carry on.
- A mutation refused by a read-only workspace raises
  [`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError], a `PermissionError`; the
  environment is fine and the tool reports the refusal to the model.
- Any other failure the workspace layer refuses deliberately, such as command output over a
  backend's limit, raises [`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError]; invalid
  arguments raise `TypeError` or `ValueError`.
- A provider SDK's own transient errors propagate unchanged and are treated as infrastructure
  failures: under durable execution the unit is retried. A backend whose platform reports a dead
  environment and a failed operation with the same exception should probe, for example with
  `working_dir()`, and raise `WorkspaceUnavailableError` when the environment is gone.
- [`UserError`][pydantic_ai.exceptions.UserError] comes from `Workspace` and policy wrappers, such
  as an unattached workspace, not from a backend operation.

