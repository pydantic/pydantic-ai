# Sandboxes

Use a sandbox when an agent needs a workspace for commands and files. Attach an environment to
the run, then use `ctx.sandbox` inside tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.sandboxes import LocalSandbox

agent = Agent('openai:gpt-5.2')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.sandbox.run(command, timeout=30)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


@agent.tool
async def read_file(ctx: RunContext[None], path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a bounded line window from a workspace file."""
    window = await ctx.sandbox.read_file(path, offset=offset, limit=limit)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix


async def main() -> None:
    async with LocalSandbox() as sandbox:
        await agent.run('Inspect the project and fix the failing test.', sandbox=sandbox)
```

The same tools work with a container, VM, or remote sandbox. Only the attached environment
changes.

`LocalSandbox` runs host subprocesses and isolates nothing. Use it only for trusted development
and tests. Attach an isolated backend before running untrusted code.

Sandbox access is opt-in. Without an attached environment, operations raise `UserError` with
instructions for attaching one; Pydantic AI never silently uses the host. Keep approval, command
restrictions, output limits, and path rules in the tool layer.

## Choose the environment

- Pass a live backend or `SandboxRef` through `sandbox=` for one run. The caller owns its
  lifecycle, and this explicit value wins over capability-provided sandboxes.
- Add one sandbox capability to provision or connect environments automatically.
- Pass `UnavailableSandbox(reason=...)` to disable execution with an application-specific
  explanation.

Tools always call the same `ctx.sandbox` methods. `fs` needs a backend with
`SupportsFilesystem`; unsupported filesystem operations raise `NotImplementedError`.

Backends raise `SandboxError` for deliberate recoverable operation failures,
`SandboxTimeoutError` when a command exceeds its deadline, and
`SandboxUnavailableError` when retrying against the same environment cannot succeed.

## Supply a sandbox from a capability

A capability supplies the run's sandbox through one hook, `get_sandbox`. It is synchronous and
must do no I/O: it returns a backend built from the capability's own settings, and that backend
creates or attaches the first time an operation runs.

`ref` is the identity of an environment the run should continue in, from an explicit `sandbox=`
argument or an earlier run in the conversation. `None` means make a fresh one.

```python
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import anyio

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import (
    CommandResult,
    SandboxBackend,
    SandboxCommand,
    SandboxRef,
)


class MyBackend:
    """Configuration plus an optional identity. Nothing here touches the network."""

    def __init__(self, *, client: Any, ref: SandboxRef | None, name: str | None):
        self.client, self.ref, self.name = client, ref, name
        self._sandbox: Any | None = None
        self._lock = anyio.Lock()

    @property
    def sandbox(self) -> Awaitable[Any]:
        """Returns something you can only await, so no method can skip connecting."""
        return self._resolve()

    async def _resolve(self) -> Any:
        async with self._lock:
            if self._sandbox is None:
                if self.ref is not None:
                    self._sandbox = await self.client.connect(self.ref.sandbox_id)
                else:
                    self._sandbox = await self.client.create(name=self.name)
                    self.ref = SandboxRef(sandbox_id=self._sandbox.id)
        return self._sandbox

    async def run(self, command: SandboxCommand, **kwargs: Any) -> CommandResult:
        sandbox = await self.sandbox
        return await sandbox.exec(command, **kwargs)

    async def working_dir(self) -> str:
        sandbox = await self.sandbox
        return sandbox.workdir


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    client: Any  # credentials stay here, never in the ref

    def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
        return MyBackend(client=self.client, ref=ref, name=ctx.conversation_id)
```

Exactly one attached capability may return a backend; two raise `UserError`. Deferred capabilities
take no part.

Pydantic AI never creates, closes, destroys or pauses an environment. A conversation can span many
runs, so the end of a run does not mean the workspace is finished with. Use `before_run` to warm up
or copy files in, `after_run` to copy results out or pause, and `wrap_run` with `try`/`finally` when
cleanup must also happen after a failure or a cancellation.

Pass `UnavailableSandbox(reason='Local execution is disabled by application policy.')` to
disable sandbox access with a useful error.

Wrap a caller-managed backend in `ReadOnlySandbox` to allow file reads and listings while
blocking commands and file changes. When a capability manages the backend, apply the wrapper in
`get_sandbox` so every reconnection remains read-only.

## Durable execution

Only the `SandboxRef` crosses a durable boundary. Tools still call `ctx.sandbox` normally, and the
worker reconnects on first use.

Pass `SandboxRef(sandbox_id=...)` through `sandbox=` when the environment is provisioned elsewhere
and outlives the run. The agent must have a capability whose `get_sandbox` connects it. Do not pass
a live backend or `LocalSandbox` into a durable run.

Capability author rules:

- `get_sandbox` does no I/O. Everything that talks to a provider belongs in the backend, behind
  the awaitable property, so it happens inside a durable unit rather than in workflow code.
- Make create-or-attach safe to run twice: durable operations may retry.
- When a ref was given and its environment is gone, raise. Do not quietly make an empty one in its
  place, because the message history says files are there that no longer are.
- Keep credentials on the capability, not in `SandboxRef` or workflow history.
- Always configure a server-side TTL or reaper: nothing in Pydantic AI destroys an environment, and
  a cancelled workflow will not do it either.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for protocol contracts,
lifecycle rules, and implementation guidance.
