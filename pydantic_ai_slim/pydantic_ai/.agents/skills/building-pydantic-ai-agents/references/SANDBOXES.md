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
`SupportsFilesystem`, while `start()` needs `SupportsStart`; unsupported operations raise
`NotImplementedError`. `ctx.sandbox.attached` is `False` only when the run has no sandbox at all,
for tools that fall back to another path.

Backends raise `SandboxTimeoutError` when a command exceeds its deadline and
`SandboxUnavailableError` when retrying against the same environment cannot succeed.

## Manage sandbox lifecycle with a capability

A capability can create an environment for each run, reconnect an existing one, share a warm
environment, or manage a pool. It does this through three hooks:

- `acquire_sandbox`: provision, select, or check out an environment once per run, or return `None`.
- `get_sandbox`: open a connection when an operation first needs it; reconnect, never create.
- `release_sandbox`: destroy, return, or detach the environment after a run that acquired a ref.

When acquisition returns a `SandboxRef`, the run stores only that serializable identity. The
live connection opens on first use:

```python
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    client: Any  # your provider's SDK client; credentials stay here, never in the ref

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        """Acquire for this run by provisioning, checking out, or selecting."""
        sandbox = await self.client.create(idempotency_key=ctx.run_id)
        return SandboxRef(sandbox_id=sandbox.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        """Connect, never create."""
        return await self.client.connect(ref.sandbox_id)

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        """Release after the run, including failure. This may destroy, check in, or do nothing."""
        await self.client.destroy(ref.sandbox_id)
```

Choose the lifecycle that matches the application:

- For a fresh sandbox per run, create it in `acquire_sandbox` and destroy it in `release_sandbox`.
- For a warm sandbox shared across runs, return its ref from `acquire_sandbox` and leave
  `release_sandbox` unchanged.
- For a pool, check out in `acquire_sandbox` and return it in `release_sandbox`.
- For a sandbox provisioned elsewhere, implement `get_sandbox` to connect its `SandboxRef`; the
  caller owns its lifecycle.

Exactly one capability may return a ref from `acquire_sandbox`, and exactly one may connect a given
ref; more raises `UserError`. Deferred capabilities take no part until loaded.

Pydantic AI closes the connection `get_sandbox` returned when the run ends; a live backend passed
through `sandbox=` stays open and caller-owned.

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

- `get_sandbox` re-opens only; raise when the environment expired. Never silently
  open-or-create: a replacement environment would contradict the model's message history.
- `release_sandbox` must be idempotent. It may destroy the sandbox, return it to a pool,
  decrement a reference count, or do nothing for a warm environment.
- Keep credentials on the capability, not in `SandboxRef` or workflow history.
- Always configure a server-side TTL or reaper: a terminated or cancelled workflow may not run cleanup.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for protocol contracts,
lifecycle rules, and implementation guidance.
