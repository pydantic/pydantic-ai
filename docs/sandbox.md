# Sandboxes

Every agent run has a sandbox, exposed through the read-only
[`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] field. By default it is an
[`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox]: every operation raises
[`UserError`][pydantic_ai.exceptions.UserError] with instructions for attaching a real
environment. No run ever gets implicit access to the host machine: in a multi-tenant
deployment, reading or writing the server's filesystem is never safe to imply, so execution is
always an explicit opt-in.

Pydantic AI resolves the sandbox once, before capability and toolset `for_run` hooks:

1. The `sandbox=` run argument, when provided as a live
   [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] or serializable
   [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef].
2. A capability's
   [`acquire_sandbox`][pydantic_ai.capabilities.AbstractCapability.acquire_sandbox] contribution.
   The latest supplier in the capability chain wins. Its effective capability ID is recorded on
   the returned reference so reconnection and release route back to the same supplier. Give the
   supplier an explicit stable `id` when the ref will cross runs or processes.
3. The framework default: an `UnavailableSandbox` explaining how to attach one.

The selected supplier remains the lifecycle owner for the whole run. A capability returned by
that supplier's later `for_run()` may contribute other run-specific behavior, but it does not
replace the owner used for `acquire_sandbox`, `get_sandbox`, or `release_sandbox`. Durable workers
recover the owner by the stable capability ID recorded on the ref.

Tools and capabilities can therefore use `ctx.sandbox` directly:

```python
from pydantic_ai import Agent, LocalSandbox, RunContext

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    result = await ctx.sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', sandbox=LocalSandbox())
```

!!! warning "`LocalSandbox` does not isolate code"
    [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] runs commands as host subprocesses and
    reads and writes the host filesystem. It is suitable only for trusted development and
    tests, which is why it is opt-in. Attach a container-, VM-, or remote-backed sandbox before
    exposing command execution or file access to untrusted input.

The framework attaches the environment; it deliberately ships no model-facing command or file
tools. Applications decide which operations to expose and where to enforce approval, command,
path, timeout, and output policies. Keep policy (allow/deny lists, path rules, output budgets)
in the tool layer and isolation in the sandbox. Denylists over free-form shell strings are
security theater: if commands must be constrained, use argv form and validate arguments.

## Reading files

Use [`Sandbox.read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] to give a model a bounded
line window:

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext[None]) -> str:
    window = await ctx.sandbox.read_file('src/app.py', offset=120, limit=40)

    print(window.lines)  # requested lines without trailing newlines
    print(window.start_line)  # 120, even if the window is empty
    print(window.has_more)  # whether content follows this window
    print(window.total_lines)  # known at EOF, otherwise None
    return window.text  # the same lines joined with "\n"
```

Relative paths resolve against the backend's working directory. `total_lines` may be `None`
when the read was served without reaching EOF. For strict text decoding and writing, use
[`Sandbox.read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] and
[`Sandbox.write_text()`][pydantic_ai.sandboxes.Sandbox.write_text].

## Attaching a sandbox

### Directly, per run

Pass any structurally conforming
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] through `sandbox=`. The caller owns its
lifecycle; create it before the run and tear it down after. An explicit sandbox wins over every
capability contribution.

```python
from my_sandboxes import make_docker_sandbox

from pydantic_ai import Agent

agent = Agent('anthropic:claude-sonnet-5')


async def main() -> None:
    async with make_docker_sandbox() as sandbox:
        result = await agent.run('Profile the script and fix the hot spot.', sandbox=sandbox)
        print(result.output)
        #> Optimized the hot loop; the profile is clean now.
```

The run wraps the backend once in the rich [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade.
Inside tools, `ctx.sandbox.backend is sandbox`. Passing the same backend to several runs shares
its state between those runs.

### From a capability

A capability supplies a sandbox through up to three lifecycle hooks. The run holds only the
serializable [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef]; the live connection is
(re)established wherever it is needed, which is what makes the same capability work unchanged
under [durable execution](#durable-execution).

- [`acquire_sandbox`][pydantic_ai.capabilities.AbstractCapability.acquire_sandbox], once per run:
  provision, check out, or select an environment and return its identity, or `None` to not contribute.
- [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox]: connect, never
  create. Called lazily on the first sandbox operation, and again in each durable unit that
  touches the sandbox.
- [`release_sandbox`][pydantic_ai.capabilities.AbstractCapability.release_sandbox], once
  after the run ends, including on failure. It may destroy the environment, return it to a pool,
  decrement a reference count, or do nothing. The inherited no-op suits warm sandboxes and
  platforms that clean up on their own.

```python {title="sandbox_capability.py"}
from dataclasses import dataclass
from typing import Any

from my_sandboxes import SandboxClient

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    client: SandboxClient  # credentials stay here, never in the ref

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        sandbox = await self.client.create()
        return SandboxRef(provider='docker', sandbox_id=sandbox.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        if ref.provider != 'docker':
            return None  # not ours; resolution continues along the capability chain
        # Re-open only: raise if the environment expired. Never create a replacement here.
        return await self.client.connect(ref.sandbox_id)

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        await self.client.destroy(ref.sandbox_id)
```

```python {requires="sandbox_capability.py"}
from my_sandboxes import SandboxClient

from pydantic_ai import Agent

from sandbox_capability import MySandboxCapability

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[MySandboxCapability(SandboxClient.from_environment())],
)
```

The same three hooks cover every lifecycle without further concepts:

| Lifecycle | `acquire_sandbox` | `get_sandbox` | `release_sandbox` |
|---|---|---|---|
| Fresh sandbox per run | provision, return its ref | connect by id | destroy |
| Warm, shared across runs | return the held backend's ref | return the held backend | inherited no-op |
| Pooled per conversation | check out or create by `ctx.conversation_id` | connect | return to pool or decrement a reference count |
| Connect-only (provisioned elsewhere) | don't override | connect by id | don't override |

Among capability suppliers, the latest in the chain wins; a supplier that returns
`None` falls through to the next. Deferred capabilities cannot contribute a sandbox because
sandbox resolution happens before deferred capabilities can load. Acquisition and release happen
inside the agent-run span, so startup failures and slow provisioning are visible in traces.

"After the run ends" includes a run that ends early with
[`DeferredToolRequests`](deferred-tools.md): the approval round-trip spans two runs, so a
fresh-per-run supplier destroys the environment before the approved call executes, and the
resumed run provisions a new, empty one. When state must survive an approval (or any other
multi-run conversation), pick a lifecycle that spans the runs — the pooled-per-conversation row
above, or create the sandbox outside the runs entirely and pass the same `sandbox=` to each.

### Disabling execution with a policy reason

Pass [`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox] explicitly to replace the
default attachment instructions with your own policy reason. Run-argument precedence keeps
capability suppliers from attaching anything, and every attempted operation surfaces the
configured reason:

```python
from pydantic_ai import Agent, UnavailableSandbox

agent = Agent('anthropic:claude-sonnet-5')


async def main() -> None:
    await agent.run(
        'What is the capital of France?',
        sandbox=UnavailableSandbox(reason='Local execution is disabled by application policy.'),
    )
```

### Making a sandbox read-only

Wrap any backend in [`ReadOnlySandbox`][pydantic_ai.sandboxes.ReadOnlySandbox] to let a run
read files without being able to change anything. File reads, directory listings, and
`working_dir` pass through; command execution and file mutation raise
[`UserError`][pydantic_ai.exceptions.UserError] explaining the restriction. The same backend
stays fully usable outside the wrapper, so one environment can be read-write for your
application and read-only for the agent:

```python
from pydantic_ai import Agent, LocalSandbox, ReadOnlySandbox

agent = Agent('anthropic:claude-sonnet-5')


async def main() -> None:
    async with LocalSandbox() as sandbox:
        root = await sandbox.working_dir()
        await sandbox.fs.write_bytes(f'{root}/data.csv', b'a,b\n1,2\n')
        await agent.run(
            'Summarize data.csv in the working directory.',
            sandbox=ReadOnlySandbox(sandbox),
        )
```

Commands are blocked along with writes because they execute in the same environment as the
filesystem: a sandbox that refused writes but ran `rm` would not be read-only. If the model
needs to execute commands against protected data, enforce read-only in the environment itself
instead (e.g. a read-only mount).

The wrapper keeps the wrapped backend's `provider` and `sandbox_id`: a
[`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] names the environment, never the policy. A
capability that supplies read-only access applies the wrapper in
[`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox], so the restriction is
re-applied on every (re)connection, including under [durable execution](#durable-execution).

## Backend protocol and facade

A backend is required to implement only four members: `provider`, `sandbox_id`, command
execution, and its working directory. The concrete [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade adds
decoding, path-resolution, and file-window behavior on top, and delegates filesystem work to
the backend's own [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] implementation.

| Need | [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade | Native opt-in |
|---|---|---|
| Execute a command | [`run()`][pydantic_ai.sandboxes.Sandbox.run] | — |
| Background process | [`start()`][pydantic_ai.sandboxes.Sandbox.start] | [`SupportsStart`][pydantic_ai.sandboxes.SupportsStart] (else `NotImplementedError`) |
| Read/write files | `fs` / [`read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] / [`write_text()`][pydantic_ai.sandboxes.Sandbox.write_text] | [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] (else `NotImplementedError`) |
| Windowed read | [`read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] | `sed` over `run()` (else read-all + slice) |
| Working directory | [`working_dir()`][pydantic_ai.sandboxes.Sandbox.working_dir] | — |
| Path resolution | [`resolve()`][pydantic_ai.sandboxes.Sandbox.resolve] | Facade-owned normalization |

Implement `SupportsStart` when the backend can return a real process handle.

Three protocol contracts matter to callers:

- Optional operations raise `NotImplementedError`; use the documented fallback.
- `timeout=` guarantees the command is terminated before an exception deriving from
  `TimeoutError` is raised.
- A non-zero `exit_code` is a normal result, not an exception.

!!! warning "The sandbox protocol is not a security boundary"
    Isolation comes from the backend environment. In particular,
    [`resolve()`][pydantic_ai.sandboxes.Sandbox.resolve] only normalizes text: `..` can escape the
    base directory and symlinks are not inspected. Enforce confinement in the sandbox itself.

[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] is also the reference implementation: one
page over `asyncio.subprocess` and `pathlib`. `LocalSandbox()` construction raises
`NotImplementedError` on non-POSIX platforms, because its timeout contract must kill the whole
process group. Structural conformance needs no registration:

```python
from my_sandboxes import DockerSandbox

from pydantic_ai.sandboxes import SandboxBackend

sandbox: SandboxBackend = DockerSandbox(image='python:3.13')
```

## Durable execution

Durable engines can't hold a live sandbox handle: workflow code is replayed, and the handle
can't cross into the activities, steps, or tasks that do the I/O. The lifecycle hooks are
already the right shape for this: only the [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef]
crosses, and `get_sandbox` re-opens it wherever the work actually runs. The worker's capability
tree already knows how to connect, so nothing needs a second registration.

### A sandbox per run

Attach a lifecycle-owning capability like `MySandboxCapability`
[above](#from-a-capability) and the run owns the whole lifecycle. Under
[Temporal](durable_execution/temporal.md), `acquire_sandbox` and `release_sandbox` each run as
their own activity and only the ref returns to workflow code, so a replay reuses the recorded
ref instead of provisioning again. The activity itself is at-least-once — Temporal retries it
if the worker crashes after provisioning but before the ref reaches history — so `acquire_sandbox`
should be idempotent: create-or-reuse keyed by
[`ctx.run_id`][pydantic_ai.tools.RunContext.run_id] (most platforms accept a caller-chosen name
or tag), with a server-side TTL as the backstop for the copy that lost the race.
Tool code calls `await ctx.sandbox.run(...)` exactly as it does
outside a workflow: the ref rides along in the serialized run context, and the first operation
inside each activity reconnects through the capability chain's `get_sandbox`.

```python {title="durable_sandbox_pattern.py" test="skip" lint="skip"}
from my_sandboxes import SandboxClient
from temporalio import workflow

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import TemporalDurability

from sandbox_capability import MySandboxCapability

agent = Agent(
    'anthropic:claude-sonnet-5',
    name='workspace_agent',
    capabilities=[
        TemporalDurability(),
        MySandboxCapability(SandboxClient.from_environment()),
    ],
)


@agent.tool
async def sh(ctx: RunContext[None], command: str) -> str:
    result = await ctx.sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


@workflow.defn
class WorkspaceWorkflow:
    @workflow.run
    async def run(self, task: str) -> str:
        result = await agent.run(task)
        return result.output
```

Release runs at the end of the run, including a failed one, but a cancelled workflow may skip
it, so **always configure a server-side idle timeout or reaper as the backstop.**
Temporal runs sandbox lifecycle operations in activities, DBOS runs them in steps, and Prefect
runs them in tasks. Each engine records one acquire and one release operation in the logical run
history, but failed physical attempts may be retried. Both hooks must therefore be idempotent.
Temporal and Prefect use bounded three-attempt defaults for capability operations; customize them
with `capability_activity_config` or `capability_task_config`. After durable release retries are
exhausted, cleanup is logged and the provider-side timeout or reaper remains the final backstop.

### A sandbox that outlives the run

When the environment is provisioned elsewhere (by an operator, another service, or an earlier
workflow), pass its [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] through `sandbox=`. The
same capability that can supply sandboxes also connects references: with a ref run argument its
`acquire_sandbox` is skipped (the caller owns the lifecycle), but its `get_sandbox` still does
the connecting. A capability that only ever connects simply doesn't override `acquire_sandbox`.

```python {title="durable_sandbox_ref_pattern.py" test="skip" lint="skip"}
from my_sandboxes import SandboxClient
from temporalio import workflow

from pydantic_ai import Agent, SandboxRef
from pydantic_ai.durable_exec.temporal import TemporalDurability

from sandbox_capability import MySandboxCapability

agent = Agent(
    'anthropic:claude-sonnet-5',
    name='workspace_agent',
    capabilities=[
        TemporalDurability(),
        MySandboxCapability(SandboxClient.from_environment()),
    ],
)


@workflow.defn
class ExistingWorkspaceWorkflow:
    @workflow.run
    async def run(self, sandbox_id: str) -> str:
        result = await agent.run(
            'Inspect the workspace and fix the failing tests.',
            sandbox=SandboxRef(provider='docker', sandbox_id=sandbox_id),
        )
        return result.output
```

A run argument wins over every capability contribution, so the run uses the referenced sandbox
and never destroys it. Pydantic AI reconstructs a deferred
[`Sandbox`][pydantic_ai.sandboxes.Sandbox] inside the durable I/O boundary; the first operation
connects once and caches the live backend for that activity, step, or task. Framework-created
refs also carry `capability_id`, which routes reconnection directly to the supplier. Caller-created
refs may omit it and use normal chain precedence for backward compatibility. `provider` and
`sandbox_id` are readable before connection, but the synchronous `sandbox.backend` property raises
until an async operation has connected the facade.

- **[Temporal](durable_execution/temporal.md)** serializes the reference into
  `TemporalRunContext` and rebuilds the deferred facade in activities. Workflow code may inspect
  its identity, but must not call sandbox operations because connecting performs I/O.
- **[DBOS](durable_execution/dbos.md)** pickles the reference as a workflow input. Recovery
  rebuilds a fresh facade that reconnects to the same `sandbox_id` through the capability
  chain. Effectful sandbox tools must still run as DBOS steps, like any other tool I/O.
- **[Prefect](durable_execution/prefect.md)** runs tools in-process. A deferred facade
  contributes `(provider, sandbox_id, capability_id)` to task cache keys without connecting;
  `UnavailableSandbox` (including the framework default) adds no sandbox component.

A live backend is still rejected inside durable containers because it cannot cross their
boundaries, and `LocalSandbox` has no meaningful ref: its worker-local temporary directory
cannot survive worker replacement.

Rules of thumb for capability authors:

- **`get_sandbox` re-opens, never creates.** If the platform deleted the sandbox while the
  workflow slept, an open-or-create fallback silently swaps in an empty environment that the
  model's message history contradicts. Recreate only as an explicit, logged decision.
- **`release_sandbox` is idempotent.** It also runs after a failure that may have destroyed the
  environment already, and a retry may repeat it. For pooled sandboxes, repeated release must not
  return or decrement the same lease twice.
- **Still set a server-side TTL.** A terminated workflow runs no cleanup; without a TTL or
  reaper, the sandbox leaks.
- **Ids only in `SandboxRef`.** The reference is recorded in workflow history; credentials
  belong on the capability.
