# Sandboxes

Use a sandbox when tools need an execution environment for commands or files. Providers implement the small structural [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] command-execution floor. Pydantic AI wraps the backend once and exposes the rich [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade at `ctx.sandbox`, reconstructing filesystem operations over POSIX shell commands unless the backend implements [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem]. It does not decide which command or file tools the model may call.

## Attach and expose

The caller owns a sandbox passed directly to a run:

```python
from pydantic_ai import Agent, LocalSandbox, RunContext, UserError

agent = Agent('openai:gpt-5.2')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    result = await sandbox.run(command, shell=True, timeout=30)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


@agent.tool
async def read_file(ctx: RunContext[None], path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a bounded line window from a workspace file."""
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    window = await sandbox.read_file(path, offset=offset, limit=limit)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix


async def main() -> None:
    async with LocalSandbox() as sandbox:
        await agent.run('Create and run hello.py.', sandbox=sandbox)
```

`LocalSandbox` runs on the host and isolates nothing. Use it only for trusted development and tests; use a container, VM, or remote implementation for untrusted code. [`Sandbox.read_file`][pydantic_ai.sandboxes.Sandbox.read_file] uses [`SupportsReadBytesRange`][pydantic_ai.sandboxes.SupportsReadBytesRange] for bounded transfer; the shell filesystem provides that range operation with `tail`, `head`, and `base64`. [`Sandbox.start`][pydantic_ai.sandboxes.Sandbox.start] requires the backend to implement [`SupportsStart`][pydantic_ai.sandboxes.SupportsStart]. Keep approval, command restrictions, output limits, and path policy in the tool layer.

## Ownership and precedence

- The caller of `run(sandbox=...)` owns the backend: create it before the run, tear it down after (typically an `async with` around the run). It wins over any capability contribution. Inside tools, `ctx.sandbox.backend` is that backend.
- A sandbox-supplying capability overrides `get_sandbox(ctx)` and returns a [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] or async context manager yielding one: the run enters it at run start and exits it at run end, like a capability toolset, then wraps the yielded backend once. Capabilities that only use the sandbox do not override this method. Prefer a fresh context manager for a per-run backend, or return the backend itself for a warm sandbox shared across runs. Resolution happens before capability and toolset `for_run`, so every `RunContext` sees the final sandbox. Serving an existing [`Sandbox`][pydantic_ai.sandboxes.Sandbox] passes it through unchanged.
- Among sandbox suppliers, the latest in the resolved chain wins, and deferred capabilities are never consulted.
- The handle is available on every `RunContext`, including capability and toolset `for_run` hooks and initial metadata factories. Initial metadata factories run after sandbox resolution and entry but before per-run capability resolution. `wrap_entire_run` and `get_sandbox` instead receive the earlier `RunPreparationContext`.

## Durable execution

Live sandbox handles do not cross durable boundaries:

- Temporal workflows reject `sandbox=` and sandbox-contributing capabilities. Carry a serializable `{provider, sandbox_id}` reference and re-open it inside an activity.
- DBOS durable `run` and `run_sync` reject both routes. Re-open by reference inside a tool decorated with `@DBOS.step()`.
- Prefect includes provider-qualified sandbox identity in tool-task cache keys, but the caller or capability still owns lifecycle.

Keep credentials worker-side, make create/open operations idempotent, and use a server-side TTL because terminated workflows do not run cleanup.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for the protocol, lifecycle rules, and implementation contracts.
