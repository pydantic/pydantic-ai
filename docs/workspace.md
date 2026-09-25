# Workspaces

A workspace gives an agent an environment where it can run commands, read and write files, or both.
Tools use it through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace]. A backend implements
[`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. When command execution
is available but native filesystem access is not, [`Workspace`][pydantic_ai.workspaces.Workspace]
performs file operations through the shell.

```python {title="workspace_agent.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspace('.')],
)


@agent.tool
async def execute(ctx: RunContext, command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
```

## Choosing a workspace

The [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] capability gives every run of an
agent a workspace on this machine: commands are host subprocesses and files are the host's files.
It isolates nothing and is not a jail. Its `working_dir` is only where commands start and what
relative paths resolve against, not a security boundary: absolute paths and commands reach anywhere
on the host that this process can. Use it for trusted, local work; run untrusted code in a container
or VM through a provider workspace. `working_dir` is required. A relative path such as `'.'`
resolves against the current directory when the workspace is constructed, so a later change of
directory doesn't move it, and a leading `~` is expanded. The caller owns that directory's creation
and cleanup.

Pass `read_only=True` to let tools read and list files while refusing commands and file changes:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace('/srv/data', read_only=True)])
```

An agent has one `LocalWorkspace`: like other capabilities with a default `id`, a second one
replaces the first unless you give it its own `id`.

The capability supplies a [`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend].
To choose the workspace for a single run instead, pass a backend through `workspace=`, which takes
precedence over the agent's capabilities:

```python {requires="workspace_agent.py"}
from pydantic_ai.workspaces import LocalWorkspaceBackend

from workspace_agent import agent


async def main() -> None:
    workspace = LocalWorkspaceBackend('.')
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=workspace)
```

Commands inherit only `PATH` and `HOME` from the agent process's environment, so they find the
host's tools and the user's configuration. The workspace's `env` is layered on top, then any `env`
passed to `run`; pass other variables a command needs the same way.

!!! warning
    Don't pass `os.environ` itself: that hands the model's commands every secret in the process,
    LLM API keys included.

`LocalWorkspaceBackend` caps captured command output at 10 MiB and raises
[`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] above that limit.

You can write a workspace for another environment by implementing a small backend. See
[Writing a backend](#writing-a-backend). Provider integrations such as Modal and E2B are available
as separate packages.

## Reading and writing files

Relative paths resolve against the workspace's working directory. Path resolution normalizes
spelling, including `..`, but does not confine access; isolation comes from the workspace itself.
[`resolve`][pydantic_ai.workspaces.Workspace.resolve] is textual and never looks at the filesystem.
To learn where a path actually leads, use [`realpath`][pydantic_ai.workspaces.Workspace.realpath]:
it asks the environment to resolve symlinks in the components that exist, and keeps the ones that
don't as written.
[`read_text`][pydantic_ai.workspaces.Workspace.read_text] and
[`write_text`][pydantic_ai.workspaces.Workspace.write_text] read and write whole text files.
[`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] returns exact bytes.

For large command output, redirect the output to a file and read a window with
[`read_file`][pydantic_ai.workspaces.Workspace.read_file]. This keeps the result passed to the model
small for every workspace.

`read_file` returns a line window and decodes text leniently.

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext, path: str, offset: int = 1) -> str:
    window = await ctx.workspace.read_file(path, offset=offset, limit=200)
    return window.text
```

By default, `read_file` returns at most 2000 lines or 50 KiB, whichever comes first. Pass
`limit=None` and `max_bytes=None` together to read through the end of the file.

The result is a [`FileWindow`][pydantic_ai.workspaces.FileWindow]. Its `text` field contains the
selected text and a continuation notice when needed. `truncated` says whether the window is
incomplete, and `truncated_by` is `'lines'`, `'bytes'`, or `None`. `remaining_lines` reports how many
lines remain when the total is known. `first_line_exceeds_limit` is true when the first requested
line alone exceeds `max_bytes`. `binary` is true when the file is binary; in that case `text`
reports its size instead of decoding it.

Remote workspaces slice the window inside their environment. Use `read_bytes` or `read_text` when
you need the exact, uncapped contents.

## Wrapping a workspace

[`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] allows reads and directory listings.
It refuses commands and file changes with
[`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError].
[`Workspace.read_only`][pydantic_ai.workspaces.Workspace.read_only] reports this policy, including
through outer wrappers, so tool providers can omit mutation tools.
[`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace] is a base class for adding behavior
around workspace operations. Helpers such as `read_text` and `read_file` use an overridden
`read_bytes` method.

```python
import logging
from pathlib import Path

from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WrapperWorkspace,
)

logger = logging.getLogger(__name__)


class LoggingWorkspace(WrapperWorkspace):
    async def read_bytes(self, path: str) -> bytes:
        logger.debug('Reading %s', path)
        return await self.wrapped.read_bytes(path)


async def main() -> None:
    source = Workspace(LocalWorkspaceBackend(Path.cwd()))
    await source.write_text('message.txt', 'hello')
    workspace = ReadOnlyWorkspace(LoggingWorkspace(source))
    assert await workspace.read_text('message.txt') == 'hello'
```

`ReadOnlyWorkspace` refuses commands because a command could change files. It is not isolation.

## Selecting a workspace for a run

Pass a workspace with the `workspace=` argument. An explicit backend is used directly, and so is a
[`Workspace`][pydantic_ai.workspaces.Workspace] facade or any
[`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace] around one, such as
`ReadOnlyWorkspace(...)`, a previous run's `result.workspace`, or a parent run's `ctx.workspace`:
it reaches tools as the same object, wrappers included. An explicit
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] is offered to the
[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace] hook of each
non-deferred capability, in capability order, until one returns a workspace. If no capability
recognizes an explicit reference, the run raises `UserError`.

The first capability that returns a workspace wins, and the ones after it are not asked. Attach
several workspace capabilities, such as one per provider, to let one agent continue in an
environment from any of them: each returns `None` for a reference it does not own, so the capability
that recognizes the reference supplies the workspace. Without a reference, the first workspace
capability in the list creates the fresh environment.

This ordering supports provider migration. Listing the new provider's capability before the old
one's sends new runs to the new provider, while a history carrying an old ref is still claimed by the
old provider's capability. A capability only answers for refs whose `provider` it owns. This mirrors
[`get_model()`][pydantic_ai.capabilities.AbstractCapability.get_model].

Pass `workspace='new'` to start in a fresh environment: like `conversation_id='new'`, it ignores any
`workspace_ref` in `message_history` and calls the hook without a reference, so the first workspace
capability creates one. Because the caller asked for a workspace, the run raises `UserError` if no
capability returns one.

With `workspace=None`, the hook receives the `workspace_ref` on the most recent `ModelResponse` in
`message_history`, or `None` if there is no such reference. If no capability returns a workspace,
the run continues with a placeholder whose operations explain how to attach one. An unrecognized
historical reference does not raise an error.

Selection happens after each capability's `for_run` hook. `for_run` sees the placeholder unless the
caller passed an explicit backend or `Workspace`. `before_run`, `wrap_run`, and tools see the
selected workspace.

`get_workspace` is synchronous and must have no side effects.

To supply workspaces from your own capability, implement `get_workspace`: return a backend
configured from the capability's own settings, carrying `ref` when one was passed in, and `None`
for a ref you do not recognize. `get_workspace` is the only place a reference is turned back into a
workspace, so a capability that creates environments must also recognize the references they get.

A capability that needs a workspace can check
[`Workspace.attached`][pydantic_ai.workspaces.Workspace.attached] in `before_run`, so a run without
one fails at the start instead of on the first tool call:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError


class ProjectNotes(AbstractCapability):
    async def before_run(self, ctx: RunContext) -> None:
        if not ctx.workspace.attached:
            raise UserError("`ProjectNotes` needs a workspace. Attach one, such as `LocalWorkspace('.')`.")
```

## Workspace references {#workspace-references}

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

The same `workspace=` argument is available on the streaming, CLI, and web interfaces. The CLI and web
interfaces apply it to every run of a session, so they do not accept `'new'`.

## Writing a backend

A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements `ref` and `working_dir`,
then adds [`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. `Workspace` adds path
resolution, text helpers, and windowed reads. For a command-only backend it derives file operations
through shell commands. A filesystem-only backend works without a shell; calling `run` on its
facade raises `UserError`. When both capabilities are present, commands and file operations must
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

### Checking a backend

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

## Errors

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
- [`UserError`][pydantic_ai.exceptions.UserError] comes from the facade and policy wrappers, such
  as an unattached workspace, not from a backend operation.

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
Prefect task). Errors the workspace raises — a missing file, a read-only refusal, a timeout — come
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
