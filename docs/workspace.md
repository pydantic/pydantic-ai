# Workspaces

A workspace gives an agent an environment to work in: somewhere to run commands and, usually, a
filesystem to read and write. Tools reach it through
[`ctx.workspace`][pydantic_ai.tools.RunContext.workspace]. Running commands is the one thing every
workspace can do; a filesystem is optional. Your application chooses the environment and writes the
tools that use it.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalWorkspace() as workspace:
        await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=workspace)
```

## Choosing a workspace

[`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace] is the built-in one, used above. It runs
host subprocesses and reads and writes the host filesystem, so it provides no isolation: use it for
trusted local development and tests, and an isolated environment (a container or VM) for untrusted
code. You can write your own workspace for any environment by implementing a small backend (see
[Writing a backend](#writing-a-backend)), and provider integrations such as Modal and E2B ship as
separate packages.

`LocalWorkspace` passes only `PATH`, `HOME`, `LANG`, and `TMPDIR` through to commands, plus any `env`
you supply, so the framework's own credentials are not inherited. A command's captured output is
limited to 10 MiB; redirect larger output to a file and read a window of it.

## Reading and writing files

Relative paths resolve against the workspace's working directory.
[`read_text`][pydantic_ai.workspaces.Workspace.read_text] and
[`write_text`][pydantic_ai.workspaces.Workspace.write_text] read and write whole text files, and
[`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] returns exact bytes.

[`read_file`][pydantic_ai.workspaces.Workspace.read_file] is the read to hand a model: it returns a
line window, decodes leniently, and is bounded so a mistaken read cannot flood the model or drag a
large file across the network.

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext[None], path: str, offset: int = 1) -> str:
    window = await ctx.workspace.read_file(path, offset=offset, limit=200)
    return window.text
```

By default `read_file` returns at most 2000 lines or 50 KiB, whichever comes first (the same
defaults as Pi and OpenCode; Claude Code and Gemini CLI also default to 2000 lines). Pass
`limit=None` and `max_bytes=None` together to read through the end of the file.

The result is a [`FileWindow`][pydantic_ai.workspaces.FileWindow], not a bare string, so a cap
cannot be missed: `truncated` is true when the window is incomplete, `truncated_by` names the cap
that fired (`'lines'` or `'bytes'`), `remaining_lines` is set when the total is known, and
`text` includes a continuation notice. A single line longer than `max_bytes` yields an empty
window with `first_line_exceeds_limit=True` rather than a partial line presented as complete.
A file whose head contains a NUL byte is reported as binary (`window.binary` is `True`, and
`window.text` names its size instead of decoding the bytes). For remote workspaces the window is
sliced inside the environment, so only the window crosses the wire, never the whole file. When
you do want the exact, uncapped contents, use `read_bytes` or `read_text`.

## Policy wrappers

[`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] allows reads and directory listings
and refuses commands and file changes. [`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace]
composes an existing workspace so you can add behavior around an operation; helpers such as `read_text`
and `read_file` go through the overridden `read_bytes`.

```python
import logging

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
    async with LocalWorkspace() as backend:
        source = Workspace(backend)
        await source.write_text('message.txt', 'hello')
        workspace = ReadOnlyWorkspace(LoggingWorkspace(source))
        assert await workspace.read_text('message.txt') == 'hello'
```

`ReadOnlyWorkspace` refuses commands because a command could change the same filesystem. This is a
policy applied to calls made through the workspace, not operating-system isolation.

## Selecting a workspace for a run

Pass a workspace to a run with the `workspace=` argument. It accepts a backend or a `Workspace` you
already have (from `result.workspace` or a subagent's `ctx.workspace`), a
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] to reconnect to an environment through a
configured provider, or `None` (the default) to use normal selection.

With `None` and no explicit workspace, a configured capability chooses one. Exactly one capability may
supply a workspace; more than one raises `UserError`, and a `WorkspaceRef` that no capability
recognizes also raises. With no capability and no reference, operations run against a placeholder that
explains how to attach a workspace rather than falling back to the host.

A capability supplies a workspace from its
[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace] hook. The hook is
synchronous and does no I/O: it returns a configured backend, never a live environment, and the
backend creates or attaches on its first operation.

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
        # `LocalWorkspace` runs on the host, so it has no reference to reconnect to: decline a ref
        # and let another capability handle it. A provider backend that can reconnect would instead
        # construct itself from the ref, e.g. `ModalWorkspaceBackend(ref=ref)`.
        if ref is not None:
            return None
        return LocalWorkspace(self.root)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspaceCapability(root=Path.cwd())],
)
```

The reference passed to `get_workspace` is an explicit `WorkspaceRef`, or the most recent one recorded
in message history, or `None`. Returning `None` declines a reference (for example one belonging to
another provider). Selection happens after each capability's `for_run` has run, so a `for_run` hook
that reads `ctx.workspace` sees the placeholder; `before_run`, `wrap_run`, and tools see the selected
workspace.

To disable workspace access explicitly, pass an
[`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] as `workspace=`; its operations
raise the reason your tools surface:

```python
from pydantic_ai.workspaces import UnavailableWorkspace

disabled = UnavailableWorkspace(reason='Workspace access is disabled by policy.')
```

The same `workspace=` argument is available on the streaming, CLI, and web interfaces.

## Writing a backend

A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements three members: `ref`,
`run`, and `working_dir`. `Workspace` wraps a backend and adds path resolution and the text and
windowed-read helpers; a backend that also implements
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem] gets native file operations, and
otherwise file operations are derived from `run` using standard shell utilities. Commands and file
operations must address the same environment.

A backend that reconnects to a remote environment keeps its live handle behind an awaitable property
and stores only configuration until first use:

```python {test="skip" lint="skip"}
class ExampleBackend(WorkspaceBackend):
    def __init__(self, *, ref: WorkspaceRef | None = None):
        self._ref = ref  # configuration only; nothing is created yet

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref  # reading it does not create anything

    @property
    def workspace(self):
        return self._get_workspace()  # awaited; creates or attaches on first use
```

On first use it attaches when given a reference (and fails if that environment is gone, rather than
creating an empty replacement) or creates a new one and publishes its reference. Reading `ref` never
triggers acquisition. Retain the concrete backend when you need provider-specific methods: `await
backend.workspace` returns its native handle, and `Workspace.backend` reaches the wrapped layer.

Pydantic AI does not create or destroy environments at the start or end of a run. Provision and clean
up in ordinary `before_run`, `after_run`, or `wrap_run` hooks, or with the provider's own SDK. Pass
`result.workspace` to a later run, or `ctx.workspace` to a subagent, to keep working in the same
environment with the same policies.

## Errors

- A non-zero command exit is a normal result, reported on `exit_code`, not an exception.
- A missing file raises `FileNotFoundError`.
- An unavailable environment raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- A command that exceeds its deadline raises
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError], which carries the output
  received so far. How a command is terminated depends on the provider.

`resolve()` normalizes path spelling, including `..`, but does not enforce confinement: isolation is
the workspace's responsibility, not the path helper's.

## Durable execution

Under a durable executor the application owns provisioning, retries, and cleanup. Persist a stable
`WorkspaceRef` before a workflow needs to reconnect, because a live handle cannot cross a serialized
boundary.

A tool's workspace I/O needs no special handling to be durable: in a Temporal workflow every tool call
already runs as an activity, so the `ctx.workspace` operations inside it are part of that durable unit
and replay skips their side effects. What the application must do is reconstruct the backend on the
worker. Temporal's default context serializes the reference; customize
`TemporalRunContext.deserialize_run_context` to rebuild the backend from worker configuration and
restore any policy wrappers, and select it with `TemporalDurability(run_context_type=...)`:

```python {test="skip"}
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai_harness.modal_workspace import ModalWorkspace, ModalWorkspaceBackend

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import TemporalDurability, TemporalRunContext
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceRef


class AppTemporalContext(TemporalRunContext[None]):
    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: None) -> 'AppTemporalContext':
        data = dict(ctx)
        ref = TypeAdapter(WorkspaceRef).validate_python(data.pop('workspace_ref'))
        # Rebuild the lazy backend from the reference and restore the run's policy wrappers: the
        # reference identifies the environment, not its access policy or provider credentials.
        workspace = ReadOnlyWorkspace(Workspace(ModalWorkspaceBackend(ref=ref)))
        return cls(**{**data, 'workspace': workspace}, deps=deps)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[ModalWorkspace(), TemporalDurability(run_context_type=AppTemporalContext)],
)


@agent.tool
async def read_report(ctx: RunContext[None]) -> str:
    # Runs inside the tool's activity, so this workspace read is already durable.
    return await ctx.workspace.read_text('report.txt')
```

In the workflow, pass the saved reference through `agent.run(workspace=ref, ...)`. Provider
credentials come from worker configuration, not the reference, and policy wrappers are restored on
each worker. See the [Temporal guide](durable_execution/temporal.md) for workflow and worker setup,
and the [DBOS](durable_execution/dbos.md) and [Prefect](durable_execution/prefect.md) guides for their
boundaries.
