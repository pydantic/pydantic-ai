# Workspaces

A workspace gives tools access to an execution environment through
[`ctx.workspace`][pydantic_ai.tools.RunContext.workspace]. Your application chooses the environment
and supplies the tools. For trusted local development:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalWorkspace() as backend:
        await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=backend)
```

`LocalWorkspace` runs host subprocesses and accesses the host filesystem. It provides no isolation.
Use an isolated environment for untrusted code. Local commands inherit `PATH`, `HOME`, `LANG`, and
`TMPDIR` when present, plus the explicit `env` overlay. Their combined captured output is limited to
10 MiB. Redirect larger output to a file and read a window of it.

## Files and policy wrappers

Relative paths resolve against the workspace's working directory. Use `read_text` and `write_text`
for complete text files, or `read_file` for a line window:

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext[None], path: str, offset: int = 1) -> str:
    window = await ctx.workspace.read_file(path, offset=offset, limit=200)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix
```

[`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace] composes an existing `Workspace`.
Override an operation to add behavior before or after it. Helpers such as `read_text` and `read_file`
use the overridden `read_bytes` operation:

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

`ReadOnlyWorkspace` allows reads and directory listings, and refuses commands and file changes.
Commands are blocked because they could change the same filesystem. This is a policy for calls
through the workspace interface; it does not provide operating-system isolation.

## Selecting a workspace

An explicit backend or facade passed through `workspace=` is used directly. Otherwise, configured
capabilities receive an explicit `WorkspaceRef`, the latest `ModelResponse.workspace_ref` from
message history, or `None` when there is no reference. A latest value of `None` suppresses older
references. History supplies identity, not provider configuration.

Exactly one capability may supply a workspace. Multiple suppliers raise `UserError`. An unrecognized
explicit `WorkspaceRef` also raises. With no reference and no supplier, operations on the unavailable
default explain how to attach a workspace; they do not fall back to the host.

[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace] is synchronous and does
no I/O. It runs after `for_run` has resolved the per-run capability instances, and is skipped for an
explicit backend or facade. Return `None` to decline a reference belonging to another provider:

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

Pass `result.workspace` to another run, or `ctx.workspace` to a subagent, to preserve the same
facade and its policies. A reconnectable backend exposes a `WorkspaceRef(provider=..., id=...)`;
pass that reference to reconnect through a configured provider. Local workspaces have no such
reference. Passing `workspace=None` uses normal selection, including message history.

To disable workspace access explicitly, pass
`UnavailableWorkspace(reason='Workspace access is disabled by application policy.')`.
The same `workspace=` argument is available on the streaming, CLI, and web interfaces.

## Backend and lifecycle responsibilities

A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] implements `ref`, `run`, and
`working_dir`. `Workspace` supplies path, text, and windowed-read helpers. It prefers native
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem] methods and otherwise derives file
operations from command execution using standard shell utilities. Commands and file operations
must address the same environment.

Provider constructors only store configuration. The provider keeps its typed native SDK handle
behind an awaitable `workspace` property. Its private `_get_workspace()` locks first acquisition
and caches the handle; `_create_or_attach(ref)` performs the SDK calls. Operations run outside that
lock. A supplied reference attaches to that environment and fails if it is gone; it must not
silently create an empty replacement. Without a reference, first use can create an environment
and publish its reference. Reading `ref` does not trigger acquisition.

Core does not automatically provision or tear down environments at run boundaries. Use ordinary
`before_run`, `after_run`, or `wrap_run` hooks for application lifecycle work. Retain the concrete
provider backend when you need SDK-specific methods: `await backend.workspace` returns its native
handle. `Workspace.backend` exposes the immediate wrapped layer, which may itself be a workspace.

A non-zero command exit is a normal result. Missing files raise `FileNotFoundError`; unavailable
environments raise `WorkspaceUnavailableError`. `WorkspaceTimeoutError` carries available output
when a command deadline is exceeded. Command termination behavior depends on the provider.
`resolve()` normalizes path spelling, including `..`; it does not enforce confinement.

## Durable execution

The application owns durable provisioning, retries, and cleanup. Persist a stable reference before
a workflow needs to reconnect to the environment. A live SDK handle does not cross a serialized
boundary. Core does not automatically route workspace operations into activities or steps.

Temporal's default context serializes the known reference. An application customizes
`TemporalRunContext.deserialize_run_context` to validate that JSON value, construct its lazy backend
from worker configuration, and restore any wrappers. Configure the context through
`TemporalDurability(run_context_type=...)`. Tools then use `ctx.workspace` inside their activity.

For example, with the optional Modal workspace integration installed:

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
        backend = ModalWorkspaceBackend(ref=ref)
        workspace = ReadOnlyWorkspace(Workspace(backend))
        return cls(**{**data, 'workspace': workspace}, deps=deps)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[ModalWorkspace(), TemporalDurability(run_context_type=AppTemporalContext)],
)


@agent.tool
async def read_report(ctx: RunContext[None]) -> str:
    return await ctx.workspace.read_text('report.txt')
```

In the workflow, pass the saved `WorkspaceRef` through `agent.run(workspace=ref, ...)`.
Provider credentials come from worker configuration, not the reference. Restore policy wrappers
on each worker: a reference identifies the environment, not its access policy. A tool performing
commands uses `ctx.workspace.run()` in the same way, with a policy that permits commands.

See the [Temporal guide](durable_execution/temporal.md) for workflow and worker setup, and the
[DBOS](durable_execution/dbos.md) and [Prefect](durable_execution/prefect.md) guides for their durable
execution boundaries. Provisioning a new environment and persisting its reference must follow the
application's retry and recovery rules.
