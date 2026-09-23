# Workspaces

A workspace gives an agent an environment where it can run commands, read and write files, or both.
Tools use it through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace]. A backend implements
[`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. When command execution
is available but native filesystem access is not, [`Workspace`][pydantic_ai.workspaces.Workspace]
performs file operations through the shell.

```python {title="workspace_agent.py"}
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace(Path.cwd())])


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
```

## Choosing a workspace

The [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] capability gives every run of an
agent a workspace on this machine: commands are host subprocesses and files are the host's files.
It isolates nothing and is not a jail. Its `working_dir` is only where commands start and what
relative paths resolve against, so absolute paths and commands reach anywhere on the host that this
process can. Use it for trusted, local work; run untrusted code in a container or VM through a
provider workspace. `working_dir` is required and must be absolute, such as `Path.cwd()` or
`'~/project'` (a leading `~` is expanded), so a run never lands in the host process's working
directory implicitly. The caller owns that directory's creation and cleanup.

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
from pathlib import Path

from pydantic_ai.workspaces import LocalWorkspaceBackend

from workspace_agent import agent


async def main() -> None:
    workspace = LocalWorkspaceBackend(Path.cwd())
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=workspace)
```

`LocalWorkspaceBackend` passes only `PATH`, `HOME`, `LANG`, and `TMPDIR` through to commands, plus any `env`
you supply. It caps captured command output at 10 MiB and raises
[`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] above that limit.

You can write a workspace for another environment by implementing a small backend. See
[Writing a backend](#writing-a-backend). Provider integrations such as Modal and E2B are available
as separate packages.

## Reading and writing files

Relative paths resolve against the workspace's working directory. Path resolution normalizes
spelling, including `..`, but does not confine access; isolation comes from the workspace itself.
[`read_text`][pydantic_ai.workspaces.Workspace.read_text] and
[`write_text`][pydantic_ai.workspaces.Workspace.write_text] read and write whole text files.
[`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] returns exact bytes.

For large command output, redirect the output to a file and read a window with
[`read_file`][pydantic_ai.workspaces.Workspace.read_file]. This keeps the result passed to the model
small for every workspace.

`read_file` returns a line window and decodes text leniently.

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext[None], path: str, offset: int = 1) -> str:
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

A local workspace's reference is `WorkspaceRef(provider='local', id=...)`, where `id` is its
`working_dir` with `~` expanded, so responses from a local run record which directory they worked
in. [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] claims that reference only when it
names its own `working_dir`, which is how a run continued from message history lands in the same
directory. It declines a local reference for any other directory, so message history can never point
the agent at an arbitrary directory on the host; that run gets no workspace unless you pass
`workspace='new'`. To supply workspaces from your own capability, implement `get_workspace`: return a
backend configured from the capability's own settings, carrying `ref` when one was passed in, and
`None` for a ref you do not recognize.

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

This backend runs commands on the host under a directory selected from its reference:

```python {title="host_workspace.py"}
from collections.abc import Awaitable, Mapping
from pathlib import Path

from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
)


class HostWorkspaceBackend(WorkspaceBackend):
    def __init__(self, base_dir: Path, ref: WorkspaceRef):
        self._ref = ref
        self._local = LocalWorkspaceBackend(base_dir / ref.id)

    @property
    def ref(self) -> WorkspaceRef:
        return self._ref

    @property
    def workspace(self) -> Awaitable[str]:
        return self._local.working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        return await self._local.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        return await self.workspace
```

Reading `ref` does not run a command. Keep the concrete backend when you need provider-specific
methods: `await backend.workspace` returns its native handle, and `Workspace.backend` reaches the
wrapped backend.

Pydantic AI does not create or destroy environments at run boundaries. Provision and clean them up
in `before_run`, `after_run`, or `wrap_run` hooks, or with the provider's SDK. Pass
`result.workspace` to a later run, or `ctx.workspace` to a subagent, to keep using the same
environment.

### Checking a backend

Subclass [`WorkspaceBackendSuite`][pydantic_ai.workspaces.testing.WorkspaceBackendSuite] in your
pytest suite and provide its `backend` fixture:

```python {test="skip" lint="skip"}
import pytest
from pydantic_ai.workspaces.testing import WorkspaceBackendSuite


class TestMyBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self) -> MyBackend:
        return MyBackend()
```

The fixture can be synchronous or asynchronous. It can also be class-scoped, so a remote backend
does not start a sandbox per rule: the ref rule accepts a ref that already exists, and the destroy
rule runs last. The suite checks these rules. Command and
filesystem rules skip when the backend does not implement
the corresponding optional protocol; reattachment rules skip until their fixtures are provided.

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
- `test_ref_is_none_until_the_environment_exists_then_stable`: A lazily created environment gets a stable, non-`None` ref on first use.
- `test_reattach_by_ref_sees_the_same_files`: A second backend attached by ref sees the same files.
- `test_reattach_after_destroy_raises_unavailable`: A ref to a destroyed environment raises `WorkspaceUnavailableError` on use.

## Errors

- A non-zero command exit is a normal result reported on `exit_code`.
- A missing file raises `FileNotFoundError`.
- A mutation refused by a read-only workspace raises
  [`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError].
- An unavailable environment raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- A command that exceeds its deadline raises
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] with the output received so
  far. How the command is terminated depends on the provider.

## Durable execution

Under [Temporal](durable_execution/temporal.md), [DBOS](durable_execution/dbos.md) or
[Prefect](durable_execution/prefect.md), a workspace supplied by a capability works everywhere a
plain run's does, and workspace I/O never runs in workflow code. Attach the capability when the agent
is constructed, so the durability capability can register one durable unit per `Workspace` method:

```python {title="durable_workspace.py" test="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability


class ShareTree(AbstractCapability[None]):
    async def before_run(self, ctx: RunContext[None]) -> None:
        # Runs in workflow code: this write is one durable unit.
        await ctx.workspace.write_text('TASK.md', 'Summarize the repository.')


agent = Agent(
    'anthropic:claude-sonnet-5',
    name='summarizer',
    capabilities=[ShareTree(), LocalWorkspace('~/project'), TemporalDurability()],
)


@agent.tool
async def read_task(ctx: RunContext[None]) -> str:
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
