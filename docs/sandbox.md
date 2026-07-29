# Sandboxes

Sandboxes give agents a workspace for running commands and working with files. Attach one to an agent run, then access it from tools and capabilities through the read-only [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] field. The environment behind that interface might be a subprocess jail, a container, a microVM, or a remote worker.

Pydantic AI splits this surface into two layers. Sandbox implementers conform structurally to the small, frozen [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol — no base class, registration, or Pydantic AI dependency is required. Runs wrap that backend once and expose the rich concrete [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade at `ctx.sandbox`. The facade adds model-friendly windowed reads and text helpers with a full-read fallback, accelerated by [`SupportsReadBytesRange`][pydantic_ai.sandboxes.SupportsReadBytesRange] when the backend provides it.

Wire up a Docker container or cloud sandbox SDK, or start with the shipped, isolation-free [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] ([below](#a-minimal-local-implementation)).

## Backend protocol and facade

A sandbox backend exposes two things: **command execution** and **bytes-only file access**. The facade delegates that complete surface and adds uniform decoding and file-window semantics.

```python
from pydantic_ai.sandboxes import Sandbox


async def analyze(sandbox: Sandbox) -> str:
    await sandbox.write_text('count.py', 'print(sum(range(10)))')
    path = await sandbox.resolve('count.py')  # relative to the sandbox's working directory
    result = await sandbox.run(['python', path], timeout=30)
    if result.exit_code != 0:
        return f'failed: {result.stderr}'
    return result.stdout
```

| Member | What it does |
|---|---|
| [`run(command, ...)`][pydantic_ai.sandboxes.SandboxBackend.run] | Execute an argv sequence (or, with `shell=True`, a shell string — a mismatch raises `TypeError`) and wait for the [result][pydantic_ai.sandboxes.SandboxResult]. |
| [`start(command, ...)`][pydantic_ai.sandboxes.SandboxBackend.start] | Start a command and return a [`SandboxProcess`][pydantic_ai.sandboxes.SandboxProcess] with `wait()`, `stream()`, and `kill()`. |
| [`fs`][pydantic_ai.sandboxes.SandboxBackend.fs] | A bytes-only [`SandboxFilesystem`][pydantic_ai.sandboxes.SandboxFilesystem]: `read_bytes`, `write_bytes`, `stat`, `list_dir`, `make_dir`, `remove`, `exists`. |
| [`working_dir()`][pydantic_ai.sandboxes.SandboxBackend.working_dir] / [`resolve(path)`][pydantic_ai.sandboxes.SandboxBackend.resolve] | The default working directory, and a helper to make model-supplied relative paths absolute. |
| [`provider`][pydantic_ai.sandboxes.SandboxBackend.provider] / [`sandbox_id`][pydantic_ai.sandboxes.SandboxBackend.sandbox_id] | Identity for logs and serialized references. |
| [`SupportsReadBytesRange`][pydantic_ai.sandboxes.SupportsReadBytesRange] (optional) | A separate extension protocol that lets the facade fetch bounded byte ranges for windowed reads. |

Three contracts to know when writing code against the protocol (implementers: see the [API reference][pydantic_ai.sandboxes] for the full set):

- **Optional operations raise `NotImplementedError`.** Not every backend can stream output, kill a process, or bound retained output (`output_limit=`). Treat `NotImplementedError` as "use the fallback": `wait()` instead of `stream()`, `timeout=` instead of `kill()`.
- **`timeout=` is a kill guarantee** — the command is terminated and an exception deriving from `TimeoutError` is raised. Merely cancelling the awaiting task is *not* guaranteed to stop the remote command.
- **Results are honest.** A non-zero `exit_code` is a normal result, not an exception; check it.

!!! warning "A sandbox protocol is not a security boundary"
    Isolation comes from the environment the implementation provides (the container, the VM, the jail) — not from this interface. In particular [`resolve()`][pydantic_ai.sandboxes.SandboxBackend.resolve] is a textual path convenience: `..` can escape the base directory and symlinks are not inspected. If you need path confinement, enforce it in the sandbox itself.

## Reading files

Use [`Sandbox.read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] to give a model a bounded line window:

```python
from pydantic_ai import RunContext, UserError


async def read_source(ctx: RunContext[None]) -> str:
    if ctx.sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    window = await ctx.sandbox.read_file('src/app.py', offset=120, limit=40)

    print(window.lines)  # requested lines without trailing newlines
    print(window.start_line)  # 120, even if the window is empty
    print(window.has_more)  # whether content follows this window
    print(window.total_lines)  # known at EOF, otherwise None
    return window.text  # the same lines joined with "\n"
```

Relative paths resolve against the backend's working directory. Backends implementing [`SupportsReadBytesRange`][pydantic_ai.sandboxes.SupportsReadBytesRange] transfer only the chunks needed for a bounded window; other backends transparently fall back to [`read_bytes()`][pydantic_ai.sandboxes.SandboxFilesystem.read_bytes]. For strict text decoding and writing, use [`Sandbox.read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] and [`Sandbox.write_text()`][pydantic_ai.sandboxes.Sandbox.write_text].

## Attaching a sandbox to a run

There are two routes.

**1. Directly, per run** — you create it, you own it. Pass a sandbox to any run method via `sandbox=`; it is then available on `ctx.sandbox` for the whole run, from the earliest hooks through `after_run`. Pydantic AI never touches its lifecycle — create the sandbox before the run and tear it down after, typically with an `async with` around the run:

```python
from my_sandboxes import make_docker_sandbox  # your sandbox library

from pydantic_ai import Agent, RunContext, UserError

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    result = await sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with make_docker_sandbox() as sandbox:  # any SandboxBackend-conforming object
        result = await agent.run('Profile the script and fix the hot spot.', sandbox=sandbox)
        print(result.output)
        #> Optimized the hot loop; the profile is clean now.
```

Because the caller owns the sandbox, sharing one across several runs (state persists between conversations) is just passing the same handle to each run.
The run wraps the backend once; inside tools, `ctx.sandbox.backend is sandbox`.

**2. From a capability** — a sandbox-supplying capability overrides [`serve_sandbox`][pydantic_ai.capabilities.AbstractCapability.serve_sandbox]. Capabilities that only use the sandbox do not override this method; they read the live sandbox from `ctx.sandbox`. Serve a **per-run sandbox** *as an async context manager*: the run enters it when it starts and exits it when it ends — exactly like a capability [toolset][pydantic_ai.capabilities.AbstractCapability.get_toolset], whose enter/exit the run also owns. Teardown is guaranteed by the run (even when the run fails to start), and `ctx.sandbox` is live everywhere except run assembly — `for_run` on capabilities and toolsets, and initial metadata factories, which all resolve before the sandbox is entered:

```python {title="sandbox_capability.py"}
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from my_sandboxes import make_docker_sandbox  # your sandbox library

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    def serve_sandbox(self) -> AbstractAsyncContextManager[SandboxBackend]:
        # A fresh sandbox per run: entered when the run starts, exited when it ends.
        return make_docker_sandbox()
```

```python {requires="sandbox_capability.py"}
from pydantic_ai import Agent

from sandbox_capability import MySandboxCapability

agent = Agent('anthropic:claude-sonnet-5', capabilities=[MySandboxCapability()])
```

The run wraps the value yielded by the context manager once; inside tools,
`ctx.sandbox.backend is my_backend`.

For a **warm sandbox shared across runs**, serve the backend itself — a bare [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] is wrapped once and never entered or exited, so it keeps running between conversations; `ctx.sandbox.backend is my_backend`. A backend that is itself an async context manager is entered, so serve `contextlib.nullcontext(backend)` to keep that one warm. Serving an existing [`Sandbox`][pydantic_ai.sandboxes.Sandbox] passes it through unchanged.

A sandbox passed to the run method **wins** over anything a capability would serve — the hook is then never called, exactly like run-level `model_settings` beating capability settings. Among sandbox suppliers, the one **latest in the resolved chain** wins (with no [ordering constraints][pydantic_ai.capabilities.CapabilityOrdering] in play, that's the last one registered), and losing suppliers never build a sandbox. Deferred capabilities are never consulted: the run's sandbox is resolved once, before the first model request.

Provisioning and teardown happen inside the agent-run span through [`wrap_iter`][pydantic_ai.capabilities.AbstractCapability.wrap_iter], so startup failures and slow setup are visible in traces.

## Durable execution

A live sandbox handle **does not survive a durable-execution boundary** — this is inherent, not an implementation gap:

- **[Temporal](durable_execution/temporal.md)**: tool bodies run in activities, where `RunContext` is rebuilt from a serialized allowlist; a live handle can't cross, and a capability-contributed sandbox would be entered as workflow code, where I/O is forbidden. Temporal agents therefore **reject** both `run(sandbox=...)` and sandbox-contributing capabilities inside a workflow with a clear error (the capability check is static, so a contributor produced at run time by a dynamic capability function fails inside the workflow sandbox instead), and `ctx.sandbox` inside an activity raises [`UserError`][pydantic_ai.exceptions.UserError] instead of silently returning `None`.
- **[DBOS](durable_execution/dbos.md)**: run arguments are pickled as workflow inputs and workflow code is replayed during recovery, so DBOS durable `run`/`run_sync` **reject** both `sandbox=` and sandbox-contributing capabilities. Re-open a sandbox by serializable reference inside a tool decorated as a DBOS step.
- **[Prefect](durable_execution/prefect.md)**: tool calls are tasks with input-hash caching; the sandbox's provider-qualified identity (`provider` + `sandbox_id` — ids are only unique within a provider) participates in the cache key so a flow-run retry with a fresh sandbox can't silently replay results recorded against a dead one.

The portable pattern is to carry a **serializable reference** and re-open the sandbox in the durable engine's I/O boundary: a Temporal activity, a DBOS step, or a Prefect task.

```python {title="durable_sandbox_pattern.py"}
from dataclasses import dataclass

from my_sandboxes import open_sandbox  # worker-side factory holding the credentials

from pydantic_ai import Agent, RunContext


@dataclass
class SandboxRef:
    provider: str
    sandbox_id: str  # ids only — keep credentials worker-side, out of workflow history


agent = Agent('anthropic:claude-sonnet-5', deps_type=SandboxRef)


@agent.tool
async def sh(ctx: RunContext[SandboxRef], command: str) -> str:
    # Re-open by id using your implementation's worker-side reconnection API.
    # With DBOS, decorate this I/O-performing tool with `@DBOS.step()` as well.
    sandbox = await open_sandbox(ctx.deps.provider, ctx.deps.sandbox_id)
    result = await sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'
```

Rules of thumb for the reference pattern:

- **Create the sandbox in an activity** (or before the workflow starts), keyed idempotently (e.g. on the workflow id) so an activity retry can't create duplicates.
- **Destroy in a workflow `finally` — and still set a server-side TTL.** A terminated workflow runs no cleanup; without a TTL/reaper, that's a guaranteed leak.
- **Ids only in `deps`/`metadata`.** Both are serialized into every activity payload and recorded in workflow history; credentials belong in worker-side configuration, mirroring the Temporal `provider_factory` pattern.
- **Fail loudly on expiry.** If the sandbox was reaped while the workflow slept, an open-*or-create* fallback silently swaps in an empty environment that the model's message history contradicts. Recreate only as an explicit, logged decision.

Re-opening by `sandbox_id` inside each tool call is exactly why the protocol requires `provider` and `sandbox_id` — they are the durable half of an otherwise live-only object. First-class rehydration inside the durable integrations (a worker-side sandbox factory, sandbox creation as a managed activity) is planned as a follow-up.

## A minimal local implementation

Pydantic AI ships one batteries-included implementation, [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox]: host subprocesses and the host filesystem behind the protocol surface. It **isolates nothing** — use it for trusted workloads, tests, and development, and swap in a real sandbox for anything else. POSIX only; construction raises `NotImplementedError` elsewhere:

```python
from pydantic_ai import Agent, LocalSandbox, RunContext, UserError

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    result = await sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalSandbox() as sandbox:  # a temp directory, removed on exit
        await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', sandbox=sandbox)
```

It is also the reference for implementing the protocol yourself: the floor is deliberately small — [its source][pydantic_ai.sandboxes.LocalSandbox] is one page over `asyncio.subprocess` and `pathlib`, and most of that page is spent honoring the contracts rather than filling in the surface: the process-group kill behind the `timeout=` guarantee, `env=` overlaying the host environment instead of replacing it, `TypeError` on command/shell mismatches, and honest `NotImplementedError`s for what it can't do. Implement the same surface over whatever backend you have, and let the type checker verify conformance — a single assignment is the whole "registration" story:

```python
from my_sandboxes import DockerSandbox  # any object with the right surface

from pydantic_ai.sandboxes import SandboxBackend

sandbox: SandboxBackend = DockerSandbox(image='python:3.13')  # type-checked structurally
```

## Building tools on the sandbox

The framework attaches the sandbox but ships no sandbox tools: what to expose (`execute`? `read_file`? approval gates? output truncation policy?) is an application decision. A tool reads `ctx.sandbox` and goes:

```python
from pydantic_ai import Agent, RunContext, UserError

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str, timeout: float = 30.0) -> str:
    """Run a shell command in the workspace."""
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError(
            'No sandbox on this run: pass `sandbox=` to the run method '
            'or register a sandbox-contributing capability.'
        )
    result = await sandbox.run(command, shell=True, timeout=min(timeout, 120.0))
    output = result.stdout + (f'\n[stderr]\n{result.stderr}' if result.stderr else '')
    return output if result.exit_code == 0 else f'[exit code: {result.exit_code}]\n{output}'
```

Keep policy (allow/deny lists, path rules, output budgets) in the tool layer; keep isolation in the sandbox. Denylists over free-form shell strings are security theater — if commands must be constrained, use argv form and validate arguments.
