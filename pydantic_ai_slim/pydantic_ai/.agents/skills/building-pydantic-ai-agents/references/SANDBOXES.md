# Sandboxes

Every run exposes a concrete `Sandbox` facade at `ctx.sandbox`. By default it is an
`UnavailableSandbox`: every operation raises `UserError` with instructions for attaching a real
environment. Execution is always an explicit opt-in; no run silently touches the host.

`LocalSandbox` executes host subprocesses and isolates nothing. Pass it explicitly
(`agent.run(..., sandbox=LocalSandbox())`) only for trusted development and tests. Attach a
container, VM, or remote backend for untrusted execution.

## Use the attached facade

Tools use `ctx.sandbox` directly:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5.2')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    result = await ctx.sandbox.run(command, shell=True, timeout=30)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


@agent.tool
async def read_file(ctx: RunContext[None], path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a bounded line window from a workspace file."""
    window = await ctx.sandbox.read_file(path, offset=offset, limit=limit)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix
```

`Sandbox.fs` requires the backend to implement `SupportsFilesystem` and `Sandbox.start`
requires `SupportsStart`; both raise `NotImplementedError` when the backend omits them. Keep
approval, command restrictions, output limits, and path policy in the tool layer.

## Resolution and lifecycle

Sandbox resolution happens before capability and toolset `for_run`:

1. The `sandbox=` run argument: a caller-owned live backend or a serializable `SandboxRef`.
2. A capability's `acquire_sandbox` contribution. The latest supplier in the resolved chain
   wins; deferred capabilities are not consulted; returning `None` falls through.
3. The framework default: `UnavailableSandbox` with attachment instructions.

A capability supplies a sandbox through three lifecycle hooks; only the serializable
`SandboxRef` is held by the run, and the live connection is (re)established on first use:

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
        """Acquire for this run by provisioning, checking out, or selecting; None declines."""
        sandbox = await self.client.create()
        return SandboxRef(provider='docker', sandbox_id=sandbox.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        """Connect, never create. None = not my ref; the capability chain continues."""
        if ref.provider != 'docker':
            return None
        return await self.client.connect(ref.sandbox_id)

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        """Release after the run, including failure. This may destroy, check in, or do nothing."""
        await self.client.destroy(ref.sandbox_id)
```

The same hooks cover every lifecycle: per-run (as above), warm (acquisition returns the held
backend's ref, no release override), pooled per conversation (acquisition checks out by
`ctx.conversation_id` and release checks it back in), and connect-only (only `get_sandbox` overridden, which serves
`SandboxRef` run arguments). With a ref run argument, `acquire_sandbox` is skipped (the caller
owns the lifecycle) but `get_sandbox` still connects.

The handle is present on every `RunContext`, including capability and toolset `for_run` hooks
and initial metadata factories.

To replace the default reason with an application policy, explicitly pass:

```python
from pydantic_ai import UnavailableSandbox

sandbox = UnavailableSandbox(reason='Local execution is disabled by application policy.')
```

To give a run read access to an environment without letting it change anything, wrap the
backend in `ReadOnlySandbox`: file reads, listings, and `working_dir` pass through, while
command execution and file mutation raise `UserError` with the reason. Commands are blocked
along with writes because they run against the same filesystem. The wrapper keeps the wrapped
backend's identity, so a capability applies it in `get_sandbox` on every (re)connection.

```python
from pydantic_ai import LocalSandbox, ReadOnlySandbox

backend = LocalSandbox()
read_only = ReadOnlySandbox(backend)  # backend stays read-write for the application
```

## Durable execution

The lifecycle hooks already fit durable execution: only the ref crosses serialization
boundaries, and the worker's capability tree already knows how to connect. The same capability
instance exists on the agent the worker constructed, with its credentials, so nothing needs a
separate registration.

- Under durable execution, `acquire_sandbox` and `release_sandbox` run as Temporal activities,
  DBOS steps, or Prefect tasks. Only the ref returns to workflow code, so replay reuses the
  recorded acquisition. A physical attempt may still repeat after a worker failure, so both
  hooks must be idempotent. Inside each durable operation, `ctx.sandbox` is rebuilt from the ref
  and reconnects through the owning capability's `get_sandbox` on first use.
- `SandboxRef(provider=..., sandbox_id=...)` passed as `sandbox=` works on every engine for a
  sandbox that outlives the run, as long as a capability whose `get_sandbox` recognizes it is
  attached to the agent. Framework-acquired refs also carry the owning capability's stable ID
  for exact reconnection and release routing.

Either way tools keep calling `await ctx.sandbox.run(...)`; the deferred facade connects once
on its first operation inside the engine's I/O boundary. Do not call sandbox operations in
Temporal workflow code (connecting is I/O), and wrap effectful DBOS sandbox tools as steps.
Live backends are rejected inside durable containers, and `LocalSandbox` is intentionally not
reconnectable because its temporary directory is worker-local.

Capability author rules:

- `get_sandbox` re-opens only; raise when the environment expired. Never silently
  open-or-create: a replacement environment would contradict the model's message history.
- `release_sandbox` must be idempotent. It may destroy the sandbox, return it to a pool,
  decrement a reference count, or do nothing for a warm environment.
- Keep credentials on the capability, not in `SandboxRef` or workflow history.
- Always configure a server-side TTL or reaper: a terminated or cancelled workflow runs no cleanup.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for protocol contracts,
lifecycle rules, and implementation guidance.
