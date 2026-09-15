# Workspaces

A workspace gives an agent an environment where it can run commands and read and write files.
Tools use it through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace]. Every workspace
supports all of these operations. A backend only has to run commands; when it has no filesystem
implementation of its own, [`Workspace`][pydantic_ai.workspaces.Workspace] performs file operations
through the shell.

```python
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    workspace = LocalWorkspace(Path.cwd())
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=workspace)
```

## Choosing a workspace

[`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace] runs commands on this machine and uses its
filesystem. It isolates nothing. Use it for trusted, local work; run untrusted code in a container
or VM through a provider workspace. The caller must give it an absolute root directory and owns
that directory's creation and cleanup.

`LocalWorkspace` passes only `PATH`, `HOME`, `LANG`, and `TMPDIR` through to commands, plus any `env`
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
It refuses commands and file changes. [`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace]
is a base class for adding behavior around workspace operations. Helpers such as `read_text` and
`read_file` use an overridden `read_bytes` method.

```python
import logging
from pathlib import Path

from pydantic_ai.workspaces import (
    LocalWorkspace,
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
    source = Workspace(LocalWorkspace(Path.cwd()))
    await source.write_text('message.txt', 'hello')
    workspace = ReadOnlyWorkspace(LoggingWorkspace(source))
    assert await workspace.read_text('message.txt') == 'hello'
```

`ReadOnlyWorkspace` refuses commands because a command could change files. It is not isolation.

## Selecting a workspace for a run

Pass a workspace with the `workspace=` argument. An explicit backend or
[`Workspace`][pydantic_ai.workspaces.Workspace] is used directly. An explicit
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] is offered to each non-deferred capability's
[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace] hook. If no capability
recognizes an explicit reference, the run raises `UserError`. If two capabilities return a
workspace, the run also raises `UserError`.

With `workspace=None`, the hook receives the `workspace_ref` on the most recent `ModelResponse` in
`message_history`, or `None` if there is no such reference. If no capability returns a workspace,
the run continues with a placeholder whose operations explain how to attach one. An unrecognized
historical reference does not raise an error.

Selection happens after each capability's `for_run` hook. `for_run` sees the placeholder unless the
caller passed an explicit backend or `Workspace`. `before_run`, `wrap_run`, and tools see the
selected workspace.

`get_workspace` is synchronous and must have no side effects. It is called on every non-deferred
capability, even after another capability has returned a workspace.

```python
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import LocalWorkspace, WorkspaceBackend, WorkspaceRef


@dataclass
class LocalWorkspaceCapability(AbstractCapability[None]):
    root: Path

    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is not None:
            return None
        return LocalWorkspace(self.root)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspaceCapability(root=Path.cwd())],
)
```

To disable workspace access explicitly, pass an
[`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] as `workspace=`:

```python
from pydantic_ai.workspaces import UnavailableWorkspace

disabled = UnavailableWorkspace(reason='Workspace access is disabled for this run.')
```

The same `workspace=` argument is available on the streaming, CLI, and web interfaces.

## Writing a backend

A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements `ref`, `run`, and
`working_dir`. [`Workspace`][pydantic_ai.workspaces.Workspace] adds path resolution, text helpers,
and windowed reads. A backend that implements
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem] supplies file operations directly;
otherwise `Workspace` uses shell commands. Commands and file operations must use the same
environment.

This backend runs commands on the host under a directory selected from its reference:

```python {title="host_workspace.py"}
from collections.abc import Awaitable, Mapping
from pathlib import Path

from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
)


class HostWorkspaceBackend(WorkspaceBackend):
    def __init__(self, base_dir: Path, ref: WorkspaceRef):
        self._ref = ref
        self._local = LocalWorkspace(base_dir / ref.id)

    @property
    def ref(self) -> WorkspaceRef:
        return self._ref

    @property
    def workspace(self) -> Awaitable[Path]:
        return self._local.root

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
        return str(await self.workspace)
```

Reading `ref` does not run a command. Keep the concrete backend when you need provider-specific
methods: `await backend.workspace` returns its native handle, and `Workspace.backend` reaches the
wrapped backend.

Pydantic AI does not create or destroy environments at run boundaries. Provision and clean them up
in `before_run`, `after_run`, or `wrap_run` hooks, or with the provider's SDK. Pass
`result.workspace` to a later run, or `ctx.workspace` to a subagent, to keep using the same
environment.

## Errors

- A non-zero command exit is a normal result reported on `exit_code`.
- A missing file raises `FileNotFoundError`.
- An unavailable environment raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- A command that exceeds its deadline raises
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] with the output received so
  far. How the command is terminated depends on the provider.

## Durable execution

Under a durable executor, the application owns provisioning, retries, and cleanup. Persist a stable
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] before a workflow needs to reconnect. A live
handle cannot cross a serialized boundary.

In a Temporal workflow, tool calls already run as activities. The default context serializes the
workspace reference. Subclass
[`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] to rebuild the backend
from worker configuration, and pass that context type to
[`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]:

```python {requires="host_workspace.py"}
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec.temporal import TemporalDurability, TemporalRunContext
from pydantic_ai.workspaces import (
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
)

from host_workspace import HostWorkspaceBackend

workspace_root = Path.cwd() / 'workspaces'


@dataclass
class HostWorkspaces(AbstractCapability[None]):
    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is None or ref.provider != 'host':
            return None
        return HostWorkspaceBackend(workspace_root, ref)


class AppTemporalContext(TemporalRunContext[None]):
    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: None) -> 'AppTemporalContext':
        data = dict(ctx)
        ref = TypeAdapter(WorkspaceRef).validate_python(data.pop('workspace_ref'))
        backend = HostWorkspaceBackend(workspace_root, ref)
        workspace = ReadOnlyWorkspace(Workspace(backend))
        return cls(**{**data, 'workspace': workspace}, deps=deps)


agent = Agent(
    'anthropic:claude-sonnet-5',
    name='report_reader',
    capabilities=[HostWorkspaces(), TemporalDurability(run_context_type=AppTemporalContext)],
)


@agent.tool
async def read_report(ctx: RunContext[None]) -> str:
    return await ctx.workspace.read_text('report.txt')
```

In the workflow, pass the saved reference through `agent.run(workspace=ref, ...)`. Provider
credentials come from worker configuration. Rebuild any wrappers when deserializing the context.
See the [Temporal guide](durable_execution/temporal.md), [DBOS guide](durable_execution/dbos.md), and
[Prefect guide](durable_execution/prefect.md) for setup.
