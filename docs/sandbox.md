# Sandboxes

Sandboxes give agents a workspace for running commands and working with files. Attach one to an agent run, then access it from tools and capabilities through the read-only [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] field. The environment behind that interface might be a subprocess jail, a container, a microVM, or a remote worker.

Pydantic AI defines the **concept, not an implementation**: [`Sandbox`][pydantic_ai.sandbox.Sandbox] is a [structural protocol](https://typing.python.org/en/latest/spec/protocol.html), so any object with the right surface conforms — no base class, no registration, no dependency on any particular sandboxing library. Wire up a Docker container or a cloud sandbox SDK — or start with the shipped, isolation-free [`LocalSandbox`][pydantic_ai.local_sandbox.LocalSandbox] ([below](#a-minimal-local-implementation)).

## The protocol

A sandbox exposes two things: **command execution** and **file access**.

```python
from pydantic_ai.sandbox import Sandbox


async def analyze(sandbox: Sandbox) -> str:
    await sandbox.fs.write_text('/workspace/count.py', 'print(sum(range(10)))')
    result = await sandbox.run(['python', '/workspace/count.py'], timeout=30)
    if result.exit_code != 0:
        return f'failed: {result.stderr}'
    return result.stdout
```

| Member | What it does |
|---|---|
| [`run(command, ...)`][pydantic_ai.sandbox.Sandbox.run] | Execute an argv sequence (or, with `shell=True`, a shell string — a mismatch raises `TypeError`) and wait for the [result][pydantic_ai.sandbox.SandboxResult]. |
| [`start(command, ...)`][pydantic_ai.sandbox.Sandbox.start] | Start a command and return a [`SandboxProcess`][pydantic_ai.sandbox.SandboxProcess] with `wait()`, `stream()`, and `kill()`. |
| [`fs`][pydantic_ai.sandbox.Sandbox.fs] | A [`SandboxFilesystem`][pydantic_ai.sandbox.SandboxFilesystem]: `read_bytes`/`write_bytes`, `read_text`/`write_text`, `stat`, `list_dir`, `make_dir`, `remove`, `exists`. |
| [`working_dir()`][pydantic_ai.sandbox.Sandbox.working_dir] / [`resolve(path)`][pydantic_ai.sandbox.Sandbox.resolve] | The default working directory, and a helper to make model-supplied relative paths absolute. |
| [`provider`][pydantic_ai.sandbox.Sandbox.provider] / [`sandbox_id`][pydantic_ai.sandbox.Sandbox.sandbox_id] | Identity for logs and serialized references. |

Three contracts to know when writing code against the protocol (implementers: see the [API reference][pydantic_ai.sandbox] for the full set):

- **Optional operations raise `NotImplementedError`.** Not every backend can stream output, kill a process, or bound retained output (`output_limit=`). Treat `NotImplementedError` as "use the fallback": `wait()` instead of `stream()`, `timeout=` instead of `kill()`.
- **`timeout=` is a kill guarantee** — the command is terminated and an exception deriving from `TimeoutError` is raised. Merely cancelling the awaiting task is *not* guaranteed to stop the remote command.
- **Results are honest.** A non-zero `exit_code` is a normal result, not an exception; check it.

!!! warning "A sandbox protocol is not a security boundary"
    Isolation comes from the environment the implementation provides (the container, the VM, the jail) — not from this interface. In particular [`resolve()`][pydantic_ai.sandbox.Sandbox.resolve] is a textual path convenience: `..` can escape the base directory and symlinks are not inspected. If you need path confinement, enforce it in the sandbox itself.

## Attaching a sandbox to a run

Pass a sandbox to any run method via `sandbox=`; it is then available on `ctx.sandbox` for the whole run, from the earliest hooks through `after_run`. **You create it, you own it**: Pydantic AI never creates, enters, closes, or destroys a sandbox — it only carries the reference for the duration of the run. Create the sandbox before the run and tear it down after, typically with an `async with` around the run:

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
    async with make_docker_sandbox() as sandbox:  # any Sandbox-conforming object
        result = await agent.run('Profile the script and fix the hot spot.', sandbox=sandbox)
        print(result.output)
        #> Optimized the hot loop; the profile is clean now.
```

Because the caller owns the sandbox, sharing one across several runs (state persists between conversations) is just passing the same handle to each run.

## Durable execution

A live sandbox handle **does not survive a durable-execution boundary** — this is inherent, not an implementation gap:

- **[Temporal](durable_execution/temporal.md)**: tool bodies run in activities, where `RunContext` is rebuilt from a serialized allowlist; a live handle can't cross. Temporal agents therefore **reject** `run(sandbox=...)` inside a workflow with a clear error, and `ctx.sandbox` inside an activity raises [`UserError`][pydantic_ai.exceptions.UserError] instead of silently returning `None`.
- **[DBOS](durable_execution/dbos.md)**: run arguments are pickled as workflow inputs and workflow code is replayed during recovery, so DBOS durable `run`/`run_sync` also **reject** `sandbox=`. Re-open a sandbox by serializable reference inside a tool decorated as a DBOS step.
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

Pydantic AI ships one batteries-included implementation, [`LocalSandbox`][pydantic_ai.local_sandbox.LocalSandbox]: host subprocesses and the host filesystem behind the protocol surface. It **isolates nothing** — use it for trusted workloads, tests, and development, and swap in a real sandbox for anything else. POSIX only; construction raises `NotImplementedError` elsewhere:

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

It is also the reference for implementing the protocol yourself: the floor is deliberately small — [its source][pydantic_ai.local_sandbox.LocalSandbox] is one page over `asyncio.subprocess` and `pathlib`, and most of that page is spent honoring the contracts rather than filling in the surface: the process-group kill behind the `timeout=` guarantee, `env=` overlaying the host environment instead of replacing it, `TypeError` on command/shell mismatches, and honest `NotImplementedError`s for what it can't do. Implement the same surface over whatever backend you have, and let the type checker verify conformance — a single assignment is the whole "registration" story:

```python
from my_sandboxes import DockerSandbox  # any object with the right surface

from pydantic_ai.sandbox import Sandbox

sandbox: Sandbox = DockerSandbox(image='python:3.13')  # type-checked structurally
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
        raise UserError('No sandbox on this run: pass `sandbox=` to the run method.')
    result = await sandbox.run(command, shell=True, timeout=min(timeout, 120.0))
    output = result.stdout + (f'\n[stderr]\n{result.stderr}' if result.stderr else '')
    return output if result.exit_code == 0 else f'[exit code: {result.exit_code}]\n{output}'
```

Keep policy (allow/deny lists, path rules, output budgets) in the tool layer; keep isolation in the sandbox. Denylists over free-form shell strings are security theater — if commands must be constrained, use argv form and validate arguments.
