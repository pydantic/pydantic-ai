# Sandboxes

Use a sandbox when an agent needs a workspace for commands and files. Attach an environment to
the run, then use `ctx.sandbox` inside tools:

```python
from pydantic_ai import Agent, LocalSandbox, RunContext

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
`NotImplementedError`.

## Manage sandbox lifecycle with a capability

A capability can create an environment for each run, reconnect an existing one, share a warm
environment, or manage a pool. It does this through three hooks:

- `acquire_sandbox`: provision, select, or check out an environment once per run.
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
        sandbox = await self.client.create()
        return SandboxRef(provider='docker', sandbox_id=sandbox.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
        """Connect, never create. With no ref, an implementation may return an already-live backend."""
        if ref is None or ref.provider != 'docker':
            return None
        return await self.client.connect(ref.sandbox_id)

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        """Release after the run, including failure. This may destroy, check in, or do nothing."""
        await self.client.destroy(ref.sandbox_id)
```

For durable execution, construct the capability with a stable ID, for example
`MySandboxCapability(client, id='my-sandbox')`, so workers can route the ref back to it.

The same hooks cover every lifecycle: per-run (as above), warm (acquisition returns the held
backend's ref, no release override), pooled per conversation (acquisition checks out by
`ctx.conversation_id` and release checks it back in), already-live (only `get_sandbox` is
overridden and returns a backend for `None`), and connect-only (only `get_sandbox` is overridden
and serves `SandboxRef` run arguments). With a ref run argument, `acquire_sandbox` is skipped (the
caller owns the lifecycle) but `get_sandbox` still connects. Exactly one active capability may
define sandbox hooks; ambiguity raises before any acquisition side effect.

`ctx.sandbox` is present on every `RunContext`, including capability and toolset `for_run` hooks
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

Either way tools keep calling `await ctx.sandbox.run(...)`; the `Sandbox` object connects once
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
