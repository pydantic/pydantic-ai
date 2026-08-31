# Sandboxes

Sandboxes give an agent a workspace where its tools can run commands and work with files. Attach
an environment to a run, then use [`ctx.sandbox`][pydantic_ai.tools.RunContext.sandbox] inside
your tools. For trusted local development, the smallest complete example uses
[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox]:

```python
from pydantic_ai import Agent, LocalSandbox, RunContext

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.sandbox.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalSandbox() as sandbox:
        await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', sandbox=sandbox)
```

The tool does not need to know whether the attached environment is a local process, container,
VM, or remote service. It uses the same [`Sandbox`][pydantic_ai.sandboxes.Sandbox] methods in
each case:

- [`run()`][pydantic_ai.sandboxes.Sandbox.run] runs a command and returns its output and exit code.
- [`read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] returns a line window suitable for
  model context.
- [`read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] and
  [`write_text()`][pydantic_ai.sandboxes.Sandbox.write_text] work with complete text files.
- [`fs`][pydantic_ai.sandboxes.Sandbox.fs] provides byte-level reads, writes, listings, and
  metadata when the environment supports filesystem access.
- [`start()`][pydantic_ai.sandboxes.Sandbox.start] starts a background process when the
  environment supports it.

!!! warning "`LocalSandbox` does not isolate code"
    [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] runs commands as host subprocesses and
    reads and writes the host filesystem. It is suitable only for trusted development and
    tests, which is why it is opt-in. Attach a container-, VM-, or remote-backed sandbox before
    exposing command execution or file access to untrusted input.

## Read files without flooding model context

Use [`Sandbox.read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] to give a model a bounded
line-count window:

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

## Safety and policy

Sandbox access is opt-in. If you do not attach one, sandbox operations raise
[`UserError`][pydantic_ai.exceptions.UserError] with instructions for attaching an environment;
Pydantic AI never silently runs commands or reads files on the host.

Pydantic AI provides the connection, not model-facing command or file tools. Your application
chooses which operations to expose and enforces approval, command, path, timeout, and output
rules in those tools. If commands must be constrained, use argv form and validate each argument;
do not rely on a denylist over free-form shell strings.

## Choose where the code runs

Pydantic AI chooses one sandbox for the run, in this order:

1. The environment passed through `sandbox=`.
2. The environment supplied by one active capability.
3. The unavailable default, which explains how to attach one when a tool tries to use it.

If more than one active capability defines sandbox hooks, the run raises before any hook
executes. Deferred capabilities cannot supply a sandbox because the environment is chosen before
they load.

### Directly, per run

Pass any [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] through `sandbox=`. Create it
before the run and tear it down afterward.

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

Inside tools, `ctx.sandbox` is the run's [`Sandbox`][pydantic_ai.sandboxes.Sandbox] object and
`ctx.sandbox.backend is sandbox`. Passing the same backend to several runs shares its state
between those runs.

### From a capability

A capability supplies a sandbox through up to three lifecycle hooks. A managed environment uses
a serializable [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef]; a provider configured with an
existing environment may skip the ref and open a connection directly from `get_sandbox`. In both
cases the run-scoped connection is established lazily wherever it is needed, which is what makes
the same capability work unchanged under [durable execution](#durable-execution).

- [`acquire_sandbox`][pydantic_ai.capabilities.AbstractCapability.acquire_sandbox], once per run:
  provision, check out, or select an environment and return its identity, or `None` to not contribute.
- [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox]: connect, never
  create. With a ref, reconnect that environment; with `None`, open a backend connection using
  capability configuration or `ctx.deps`. Called lazily on the first sandbox operation, and again
  in each durable unit that touches the sandbox. Each call returns a detachable connection handle.
- [`release_sandbox`][pydantic_ai.capabilities.AbstractCapability.release_sandbox], once
  after a run that acquired a ref ends, including on failure. It may destroy the environment,
  return it to a pool, decrement a reference count, or do nothing. The inherited no-op suits
  warm sandboxes and platforms that clean up on their own.

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

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
        if ref is None or ref.provider != 'docker':
            return None  # this implementation requires an acquired or caller-provided ref
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
    capabilities=[
        MySandboxCapability(SandboxClient.from_environment(), id='my-sandbox'),
    ],
)
```

The explicit `id` lets durable workers route a sandbox reference back to the same capability.

The same three hooks cover every lifecycle without further concepts:

| Lifecycle | `acquire_sandbox` | `get_sandbox` | `release_sandbox` |
|---|---|---|---|
| Fresh sandbox per run | provision, return its ref | connect by id | destroy |
| Warm environment shared across runs | return its ref | open a run-scoped connection | inherited no-op |
| Pooled per conversation | check out or create by `ctx.conversation_id` | connect | return to pool or decrement a reference count |
| Provider-configured environment | don't override | open a run-scoped connection for `None` | don't override |
| Connect-only (provisioned elsewhere) | don't override | connect by id | don't override |

Without a ref there is no provider or sandbox identity to expose before connection. The first
async sandbox operation calls `get_sandbox(ctx, None)` and caches its backend; after that,
`sandbox.provider`, `sandbox.sandbox_id`, and `sandbox.backend` reflect the returned backend.
When the run ends, Pydantic AI calls `close(terminate=False)` if that backend supports it. This
detaches the run's connection without terminating the environment. Because each run caches its
own connection, `get_sandbox` should return a fresh detachable handle rather than one backend
object shared by multiple runs. For a caller-owned live backend object that should remain open
and be reused directly, pass it through `sandbox=`; run teardown never closes that object.

Returning `None` from `acquire_sandbox` skips acquisition. If the capability implements
`get_sandbox`, the run asks it for an open connection with `ref=None`; otherwise the run uses the
unavailable default. Acquisition and release happen inside the agent-run span, so startup
failures and slow provisioning are visible in traces.

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

## Build a sandbox backend

A backend is required to implement only four members: `provider`, `sandbox_id`, command
execution, and its working directory. Pydantic AI exposes it to tools through a
[`Sandbox`][pydantic_ai.sandboxes.Sandbox] object, which adds text decoding, path resolution,
and line-window reads. Filesystem operations use the backend's
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] implementation.

| Need | [`Sandbox`][pydantic_ai.sandboxes.Sandbox] API | Backend support |
|---|---|---|
| Execute a command | [`run()`][pydantic_ai.sandboxes.Sandbox.run] | — |
| Background process | [`start()`][pydantic_ai.sandboxes.Sandbox.start] | [`SupportsStart`][pydantic_ai.sandboxes.SupportsStart] (else `NotImplementedError`) |
| Read/write files | [`fs`][pydantic_ai.sandboxes.Sandbox.fs] / [`read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] / [`write_text()`][pydantic_ai.sandboxes.Sandbox.write_text] | [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] (else `NotImplementedError`) |
| Windowed read | [`read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] | `sed` over `run()`, with [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] fallback (else `NotImplementedError`) |
| Working directory | [`working_dir()`][pydantic_ai.sandboxes.Sandbox.working_dir] | — |
| Path resolution | [`resolve()`][pydantic_ai.sandboxes.Sandbox.resolve] | Handled by `Sandbox` |

Implement `SupportsStart` when the backend can return a real process handle.

Three protocol contracts matter to callers:

- Optional operations raise `NotImplementedError`; use the documented fallback.
- `timeout=` guarantees the command is terminated before an exception deriving from
  `TimeoutError` is raised.
- A non-zero `exit_code` is a normal result, not an exception.

[`SandboxBackend.run()`][pydantic_ai.sandboxes.SandboxBackend.run] returns complete captured
output. Truncating its result in a tool bounds model context, but does not bound the backend's
transfer or memory use. For commands whose output volume is not trusted, use a backend-native
streaming or bounded-execution facility rather than assuming a tool-side character limit is a
process resource limit.

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
        MySandboxCapability(SandboxClient.from_environment(), id='my-sandbox'),
    ],
)


@agent.tool
async def sh(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.sandbox.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


@workflow.defn
class WorkspaceWorkflow:
    @workflow.run
    async def run(self, task: str) -> str:
        result = await agent.run(task)
        return result.output
```

After a successful acquisition, release runs at the end of the run, including a failed one, but
a cancelled workflow may skip it, so **always configure a server-side idle timeout or reaper as
the backstop.**
Temporal runs sandbox lifecycle operations in activities, DBOS runs them in steps, and Prefect
runs them in tasks. Each engine records one acquire and one release operation in the logical run
history, but failed physical attempts may be retried. Both hooks must therefore be idempotent.
Temporal's default activity retry policy is unbounded; set an explicit retry policy through
`TemporalDurability(activity_config=...)` when needed. Prefect capability tasks currently use
Pydantic AI's default Prefect task configuration, which sets zero retries. The provider-side
timeout or reaper remains the final cleanup backstop in every engine.

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
        MySandboxCapability(SandboxClient.from_environment(), id='my-sandbox'),
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
refs also carry `capability_id`, which routes reconnection directly to the provider. Caller-created
refs may omit it; the run's sole sandbox provider is then asked to connect. `provider` and
`sandbox_id` are readable before connection, but the synchronous `sandbox.backend` property raises
until an async operation has opened the connection.

- **[Temporal](durable_execution/temporal.md)** serializes the reference into
  `TemporalRunContext` and rebuilds the run's `Sandbox` object in activities. Workflow code may
  inspect its identity, but must not call sandbox operations because connecting performs I/O.
- **[DBOS](durable_execution/dbos.md)** pickles the reference as a workflow input. Recovery
  rebuilds a fresh `Sandbox` object that reconnects to the same `sandbox_id` through the capability
  chain. Effectful sandbox tools must still run as DBOS steps, like any other tool I/O.
- **[Prefect](durable_execution/prefect.md)** runs tools in-process. A not-yet-connected sandbox
  contributes `(provider, sandbox_id, capability_id)` to task cache keys without connecting;
  a provider-only sandbox contributes its capability ID. `UnavailableSandbox` (including the
  framework default) adds no sandbox component.

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
