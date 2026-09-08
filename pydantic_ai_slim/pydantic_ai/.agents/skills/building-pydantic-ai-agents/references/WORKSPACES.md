# Workspaces

Use a workspace when an agent needs a workspace for commands and files. Attach an environment to
the run, then use `ctx.workspace` inside tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('openai:gpt-5.2')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=30)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


@agent.tool
async def read_file(ctx: RunContext[None], path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a bounded line window from a workspace file."""
    window = await ctx.workspace.read_file(path, offset=offset, limit=limit)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix


async def main() -> None:
    async with LocalWorkspace() as workspace:
        await agent.run('Inspect the project and fix the failing test.', workspace=workspace)
```

The same tools work with a container, VM, or remote workspace. Only the attached environment
changes.

`LocalWorkspace` runs host subprocesses and isolates nothing. Use it only for trusted development
and tests. Attach an isolated backend before running untrusted code.

Workspace access is opt-in. Without an attached environment, operations raise `UserError` with
instructions for attaching one; Pydantic AI never silently uses the host. Keep approval, command
restrictions, output limits, and path rules in the tool layer.

## Choose the environment

- Pass a live backend or `WorkspaceRef` through `workspace=` for one run. The caller owns its
  lifecycle, and this explicit value wins over capability-provided workspaces.
- Add one workspace capability to provision or connect environments automatically.
- Pass `UnavailableWorkspace(reason=...)` to disable execution with an application-specific
  explanation.

Tools always call the same flat `ctx.workspace` methods. Native `SupportsFilesystem` methods are
preferred when the backend provides them; otherwise `Workspace` derives complete filesystem access
from `run()` with standard shell utilities.

Backends raise `WorkspaceError` for deliberate recoverable operation failures,
`WorkspaceTimeoutError` when a command exceeds its deadline, and
`WorkspaceUnavailableError` when retrying against the same environment cannot succeed.

## Supply a workspace from a capability

A capability supplies the run's workspace through one hook, `get_workspace`. It is synchronous and
must do no I/O: it returns a backend built from the capability's own settings, and that backend
creates or attaches the first time an operation runs.

`ref` is the identity of an environment the run should continue in when the caller passes a
`WorkspaceRef` through `workspace=`. `None` means make a fresh one. Pydantic AI does not infer workspace
identity from message history; pass `result.workspace` or its `ref` explicitly to continue.

```python
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import anyio

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
)


class MyBackend:
    """Configuration plus an optional identity. Nothing here touches the network."""

    def __init__(self, *, client: Any, ref: WorkspaceRef | None, name: str | None):
        self.client, self.ref, self.name = client, ref, name
        self._workspace: Any | None = None
        self._lock = anyio.Lock()

    @property
    def workspace(self) -> Awaitable[Any]:
        """Returns something you can only await, so no method can skip connecting."""
        return self._resolve()

    async def _resolve(self) -> Any:
        async with self._lock:
            if self._workspace is None:
                if self.ref is not None:
                    self._workspace = await self.client.connect(self.ref.workspace_id)
                else:
                    self._workspace = await self.client.create(name=self.name)
                    self.ref = WorkspaceRef(workspace_id=self._workspace.id)
        return self._workspace

    async def run(self, command: WorkspaceCommand, **kwargs: Any) -> CommandResult:
        workspace = await self.workspace
        return await workspace.exec(command, **kwargs)

    async def working_dir(self) -> str:
        workspace = await self.workspace
        return workspace.workdir


@dataclass
class MyWorkspaceCapability(AbstractCapability[Any]):
    client: Any  # credentials stay here, never in the ref

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
        return MyBackend(client=self.client, ref=ref, name=ctx.conversation_id)
```

Exactly one attached capability may return a backend; two raise `UserError`. Deferred capabilities
take no part.

Pydantic AI never creates, closes, destroys or pauses an environment. A conversation can span many
runs, so the end of a run does not mean the workspace is finished with. Use `before_run` to warm up
or copy files in, `after_run` to copy results out or pause, and `wrap_run` with `try`/`finally` when
cleanup must also happen after a failure or a cancellation.

Pass `UnavailableWorkspace(reason='Local execution is disabled by application policy.')` to
disable workspace access with a useful error.

Wrap a caller-managed backend in `ReadOnlyWorkspace` to allow file reads and listings while
blocking commands and file changes. When a capability manages the backend, apply the wrapper in
`get_workspace` so every reconnection remains read-only.

## Durable execution

The live backend never crosses a durable boundary; its `WorkspaceRef`, method arguments, and the
serializable run context do. Tools and capability hooks still call `ctx.workspace` normally; configure
workspace use inside the durable tool or capability activity that owns the provider boundary. Reconnection
goes through the exact supplying capability. Give the workspace a stable reference, and do not access
`workspace.backend` from workflow code.

Pass `WorkspaceRef(workspace_id=...)` through `workspace=` when the environment is provisioned elsewhere
and outlives the run. The agent must have a capability whose `get_workspace` connects it. Do not pass
a live backend or `LocalWorkspace` into a durable run.

Capability author rules:

- `get_workspace` does no I/O. Everything that talks to a provider belongs in the backend, behind
  the awaitable property, so it happens inside a durable unit rather than in workflow code.
- Make create-or-attach safe to run twice: durable operations may retry.
- When a ref was given and its environment is gone, raise. Do not quietly make an empty one in its
  place, because the message history says files are there that no longer are.
- Keep credentials on the capability, not in `WorkspaceRef` or workflow history.
- Always configure a server-side TTL or reaper: nothing in Pydantic AI destroys an environment, and
  a cancelled workflow will not do it either.

See the full [workspace guide](https://ai.pydantic.dev/workspace/) for protocol contracts,
lifecycle rules, and implementation guidance.
