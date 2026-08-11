# Sandboxes

Every run exposes a concrete `Sandbox` facade at
`ctx.sandbox`. On POSIX, the default is a fresh per-run
`LocalSandbox` whose lazy temporary directory is removed
when the run ends. Elsewhere, the default is
`UnavailableSandbox`, whose operations raise
`UserError` with platform guidance.

`LocalSandbox` executes host subprocesses and isolates nothing. Use it only for trusted
development and tests. Attach a container, VM, or remote backend for untrusted execution.

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
requires `SupportsStart` — both raise `NotImplementedError` when the backend omits them. Keep
approval, command restrictions, output limits, and path policy in the tool layer.

## Resolution and lifecycle

Sandbox resolution happens before capability and toolset `for_run`:

1. The `sandbox=` run argument: a caller-owned live backend or a serializable `SandboxRef`.
2. A capability's `get_sandbox(ctx)` contribution. The latest supplier in the resolved chain
   wins; deferred capabilities are not consulted.
3. The framework default.

A capability may return an async context manager yielding a fresh backend; the run enters it at
run start and exits it at run end. A bare backend is treated as warm and shared: the capability
retains lifecycle ownership. Returning an existing `Sandbox` passes the facade through unchanged.
Provisioning and teardown happen inside the agent-run span through `wrap_entire_run`, so startup
failures and slow setup are visible in traces.

The handle is present on every `RunContext`, including capability and toolset `for_run` hooks and
initial metadata factories. `wrap_entire_run` and `get_sandbox` receive the earlier
`RunPreparationContext`, whose `sandbox` contains only an explicitly supplied run argument.

To prohibit local execution, explicitly pass:

```python
from pydantic_ai import UnavailableSandbox

sandbox = UnavailableSandbox(reason='Local execution is disabled by application policy.')
```

## Durable execution

A `SandboxProvider` subclass is the glue for one provider, holding worker-side credentials:
`create()` provisions, `connect()` re-opens an existing environment, `teardown()` destroys.
Only `connect()` is required; `create()` raises `NotImplementedError` by default and
`teardown()` is a no-op.

Two ways to use it:

- `ManagedSandbox(provider)` as a capability: the run creates a sandbox at start and destroys
  it at end, including on failure. No reference is ever handled by hand. Supported under
  `TemporalDurability` (both halves run as activities); DBOS and Prefect reject it.
- `SandboxRef(provider=..., sandbox_id=...)` passed as `sandbox=` for a sandbox that outlives the run,
  with the provider registered via `TemporalDurability(sandbox_providers=[...])`,
  `DBOSDurability(sandbox_providers=[...])`, or `PrefectDurability(sandbox_providers=[...])`.

Either way tools keep calling `await ctx.sandbox.run(...)`; the deferred facade connects once on
its first operation inside the engine's I/O boundary.

Temporal serializes the identity into activities. DBOS pickles it as a workflow input and
reconnects to the same environment when recovery re-executes the workflow. Sandbox I/O must still
run in the engine's I/O boundary: do not connect in Temporal workflow code, and wrap effectful
DBOS sandbox tools as steps. Prefect includes deferred `(provider, sandbox_id)` identity in cache
keys without connecting.

Without either, Temporal and DBOS retain `UnavailableSandbox`. They reject live backends.
`LocalSandbox` is intentionally not reconnectable because its temporary directory is worker-local.

Provider rules:

- `connect()` re-opens only; raise when the environment expired. Never silently open-or-create.
- `teardown()` must tolerate an already-gone sandbox.
- Keep credentials on the provider, not in `SandboxRef` or workflow history.
- Always configure a server-side TTL or reaper: a terminated or cancelled workflow runs no cleanup.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for protocol contracts,
lifecycle rules, and implementation guidance.
