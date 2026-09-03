# Sandboxes

Use a sandbox when an agent needs a workspace for commands and files. Attach an environment to
the run, then use `ctx.sandbox` inside tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalSandbox

agent = Agent('openai:gpt-5.2', capabilities=[LocalSandbox()])


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
    await agent.run('Inspect the project and fix the failing test.')
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

## Manage sandbox lifecycle with a capability

A capability can create an environment for each run, reconnect an existing one, share a warm
environment, or manage a pool. It does this through three hooks:

- `acquire_sandbox`: return a ref or `None`; decorate it with `@durable_operation` when provisioning or checkout performs I/O.
- `resolve_sandbox`: synchronously construct a backend whose client connects lazily on its first operation.
- `release_sandbox`: destroy, return, or do nothing; decorate it with `@durable_operation` when it performs I/O.

When acquisition returns a `SandboxRef`, the run stores that serializable identity and constructs
the backend before run hooks:

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

    def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        """Construct a handle without I/O; the handle connects lazily on first use."""
        return self.client.sandbox(ref.sandbox_id)

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        """Release after the run, including failure. This may destroy, check in, or do nothing."""
        await self.client.destroy(ref.sandbox_id)
```

Choose the lifecycle that matches the application:

- For a fresh sandbox per run, create it in `acquire_sandbox` and destroy it in `release_sandbox`.
- For a warm sandbox shared across runs, return its ref from `acquire_sandbox` and leave
  `release_sandbox` unchanged.
- For a pool, check out in `acquire_sandbox` and return it in `release_sandbox`.
- For a sandbox provisioned elsewhere, implement `resolve_sandbox` to connect its `SandboxRef`; the
  caller owns its lifecycle.

Capabilities are asked in list order and the first ref returned from `acquire_sandbox` wins. The
framework stamps its capability ID on the ref and routes connection and release back to it. For a
caller-built unstamped ref, exactly one capability may connect. Deferred capabilities take no part.

Pydantic AI closes the backend `resolve_sandbox` returned when the run ends; a live backend passed
through `sandbox=` stays open and caller-owned.

Pass `UnavailableSandbox(reason='Local execution is disabled by application policy.')` to
disable sandbox access with a useful error.

Wrap a caller-managed backend in `ReadOnlySandbox` to allow file reads and listings while
blocking commands and file changes. When a capability manages the backend, apply the wrapper in
`get_wrapper_sandbox` so every reconstructed backend remains read-only.

## Durable execution

Only the `SandboxRef` crosses a durable boundary. Tools still call `ctx.sandbox` normally, and the
worker constructs a fresh backend inside every durable unit.

Pass `SandboxRef(sandbox_id=...)` through `sandbox=` when the environment is provisioned elsewhere
and outlives the run. The agent must have a capability whose `resolve_sandbox` connects it. Do not pass
a live backend or `LocalSandboxBackend` into a durable run.

Capability author rules:

- Decorate `acquire_sandbox` and `release_sandbox` with `@durable_operation` to make them durable
  units. Undecorated overrides run inline and must be deterministic and free of I/O.
- `resolve_sandbox` must be construct-only. Its backend should raise `SandboxUnavailableError` from
  the first operation when the environment has expired, never silently create a replacement.
- Async `release_sandbox` must be idempotent. It may destroy the sandbox, return it to a pool,
  decrement a reference count, or do nothing for a warm environment.
- Keep credentials on the capability, not in `SandboxRef` or workflow history.
- Always configure a server-side TTL or reaper: a terminated or cancelled workflow may not run cleanup.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for protocol contracts,
lifecycle rules, and implementation guidance.
