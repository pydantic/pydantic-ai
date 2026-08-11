# Sandboxes

Every agent run has a sandbox, exposed through the read-only
[`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] field. On POSIX platforms, the
default is a fresh [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] for each run. Its
temporary working directory is created lazily on first use and removed when the run ends. On
other platforms, the default is an
[`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox], whose operations raise
[`UserError`][pydantic_ai.exceptions.UserError] with guidance to attach another backend.

!!! warning "The default sandbox does not isolate code"
    `LocalSandbox` runs commands as host subprocesses and reads and writes the host filesystem.
    It is suitable only for trusted development and tests. Attach a container-, VM-, or
    remote-backed sandbox before exposing command execution or file access to untrusted input.

Pydantic AI resolves the sandbox once, before capability and toolset `for_run` hooks:

1. The `sandbox=` run argument, when provided as a live
   [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] or serializable
   [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef].
2. A capability's
   [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox] contribution. The
   latest supplier in the resolved capability chain wins.
3. The framework default: a per-run `LocalSandbox` on POSIX, or `UnavailableSandbox` elsewhere.

Tools and capabilities can therefore use `ctx.sandbox` directly:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    result = await ctx.sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    await agent.run('Write fizzbuzz to fizzbuzz.py and run it.')
```

The framework attaches the environment; it deliberately ships no model-facing command or file
tools. Applications decide which operations to expose and where to enforce approval, command,
path, timeout, and output policies.

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

Relative paths resolve against the backend's working directory. For a bounded window the facade
slices with `sed` inside the sandbox so only the requested lines cross the wire — backends
guarantee `run` and `fs` see the same files (see the
[protocol contracts][pydantic_ai.sandboxes]). If the environment cannot slice (no usable `sed`,
a non-POSIX shell), the facade reads the full bytes and slices itself. `total_lines` may be
`None` on the sliced path when the window provably didn't reach EOF. For strict
text decoding and writing, use
[`Sandbox.read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] and
[`Sandbox.write_text()`][pydantic_ai.sandboxes.Sandbox.write_text].

## Attaching a sandbox

The default local backend is convenient for trusted work. Attach an isolated backend for
untrusted execution or when a run needs a persistent or remote environment.

### Directly, per run

Pass any structurally conforming
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] through `sandbox=`. The caller owns its
lifecycle; create it before the run and tear it down after. An explicit backend wins over every
capability contribution.

```python
from my_sandboxes import make_docker_sandbox

from pydantic_ai import Agent, RunContext

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    result = await ctx.sandbox.run(command, shell=True, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


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

For a fresh sandbox per run from a provider you can create and destroy, reach for the ready-made
[`ManagedSandbox`][pydantic_ai.sandboxes.ManagedSandbox] capability, which is also the one path
that works inside a Temporal workflow — see [durable execution](#durable-execution).

To supply a sandbox some other way, write a capability that overrides
[`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox]. Capabilities that only
consume the sandbox read `ctx.sandbox` and do not override this method.

Prefer returning a fresh async context manager for a per-run backend. The run enters it before
`for_run`, exits it after all run hooks and toolsets, and guarantees teardown when setup or
execution fails:

```python {title="sandbox_capability.py"}
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from my_sandboxes import make_docker_sandbox

from pydantic_ai import RunPreparationContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    def get_sandbox(
        self, ctx: RunPreparationContext[Any]
    ) -> AbstractAsyncContextManager[SandboxBackend]:
        return make_docker_sandbox()
```

```python {requires="sandbox_capability.py"}
from pydantic_ai import Agent

from sandbox_capability import MySandboxCapability

agent = Agent('anthropic:claude-sonnet-5', capabilities=[MySandboxCapability()])
```

For a warm sandbox shared across runs, return the backend itself. A bare backend is wrapped but
never entered or exited — even when it also implements the async context manager protocol — so
the capability retains lifecycle ownership. Run-managed lifecycle is requested only by returning
an actual context manager. Returning an existing `Sandbox` facade preserves that facade unchanged.

Among capability suppliers, the latest in the resolved chain wins and losing suppliers are not
consulted. Deferred capabilities cannot contribute a sandbox because sandbox resolution happens
before deferred capabilities can load.

Provisioning and teardown happen inside the agent-run span through
[`wrap_entire_run`][pydantic_ai.capabilities.AbstractCapability.wrap_entire_run], so startup
failures and slow setup are visible in traces.

### Disabling the default

Pass [`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox] explicitly when local
execution is prohibited by application policy. Run-argument precedence prevents the framework
from creating its default local backend, and every attempted operation surfaces the configured
reason:

```python
from pydantic_ai import Agent, UnavailableSandbox

agent = Agent('anthropic:claude-sonnet-5')


async def main() -> None:
    await agent.run(
        'What is the capital of France?',
        sandbox=UnavailableSandbox(reason='Local execution is disabled by application policy.'),
    )
```

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

[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] is also the reference implementation. It is
one page over `asyncio.subprocess` and `pathlib`, including process-group termination for the
`timeout=` guarantee, environment overlays, command/shell validation, and clear
`NotImplementedError`s for unsupported operations. Direct `LocalSandbox()` construction raises `NotImplementedError` on non-POSIX
platforms. This is distinct from the run default, which attaches `UnavailableSandbox` there.
Structural conformance needs no registration:

```python
from my_sandboxes import DockerSandbox

from pydantic_ai.sandboxes import SandboxBackend

sandbox: SandboxBackend = DockerSandbox(image='python:3.13')
```

## Building tools on the sandbox

A tool reads `ctx.sandbox` directly:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: str, timeout: float = 30.0) -> str:
    """Run a shell command in the workspace."""
    result = await ctx.sandbox.run(command, shell=True, timeout=min(timeout, 120.0))
    output = result.stdout + (f'\n[stderr]\n{result.stderr}' if result.stderr else '')
    return output if result.exit_code == 0 else f'[exit code: {result.exit_code}]\n{output}'
```

Keep policy (allow/deny lists, path rules, output budgets) in the tool layer; keep isolation in
the sandbox. Denylists over free-form shell strings are security theater — if commands must be
constrained, use argv form and validate arguments.

## Durable execution

Durable engines can't hold a live sandbox handle: workflow code is replayed, and the handle
can't cross into the activities, steps, or tasks that do the I/O. Two pieces solve that:

- [`SandboxProvider`][pydantic_ai.sandboxes.SandboxProvider] is the glue for one provider,
  holding worker-side credentials and configuration. `create()` provisions an environment,
  `connect()` re-opens an existing one, and `teardown()` destroys it.
- [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] is pure serializable identity — `provider`
  plus `sandbox_id` — with no credentials or live client.

### A sandbox per run

Attach [`ManagedSandbox`][pydantic_ai.sandboxes.ManagedSandbox] and the run owns the whole
lifecycle: it creates a sandbox before any hook sees `ctx.sandbox` and destroys it after the last
one, including when the run fails. Tool code calls `await ctx.sandbox.run(...)` exactly as it does
outside a workflow, and no reference is ever handled by hand.

```python {title="managed_sandbox_pattern.py" test="skip" lint="skip"}
from my_sandboxes import SandboxClient
from temporalio import workflow

from pydantic_ai import Agent, RunContext, SandboxBackend
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.sandboxes import ManagedSandbox, SandboxProvider


class MySandboxProvider(SandboxProvider):
    def __init__(self, client: SandboxClient):
        self.client = client  # credentials stay on the worker

    @property
    def provider(self) -> str:
        return 'my-sandbox'

    async def create(self) -> SandboxBackend:
        return await self.client.create()

    async def connect(self, sandbox_id: str) -> SandboxBackend:
        # Re-open only: raise if the environment expired. Never create a replacement here.
        return await self.client.connect(sandbox_id)

    async def teardown(self, sandbox_id: str) -> None:
        await self.client.destroy(sandbox_id)


agent = Agent(
    'anthropic:claude-sonnet-5',
    name='workspace_agent',
    capabilities=[
        TemporalDurability(),
        ManagedSandbox(MySandboxProvider(SandboxClient.from_environment())),
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

Only `connect()` is required. `create()` is for providers that can provision, and `teardown()`
defaults to a no-op for platforms that reap idle sandboxes themselves.

Under [Temporal](durable_execution/temporal.md), creation and destruction each run as their own
activity and only the resulting identity returns to workflow code, so creation happens exactly
once per run even across replays. Teardown runs at the end of the run, including a failed one, but
a cancelled workflow may skip it — **always configure a server-side idle timeout or reaper as the
backstop.** [DBOS](durable_execution/dbos.md) and [Prefect](durable_execution/prefect.md) have no
equivalent boundary and reject `ManagedSandbox`; use a reference there.

### A sandbox that outlives the run

When the environment is provisioned elsewhere — by an operator, another service, or an earlier
workflow — pass its [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] through `sandbox=` and
register its provider on the durability capability:

```python {title="durable_sandbox_ref_pattern.py" test="skip" lint="skip"}
from temporalio import workflow

from pydantic_ai import Agent, SandboxRef
from pydantic_ai.durable_exec.temporal import TemporalDurability

from managed_sandbox_pattern import MySandboxProvider, SandboxClient

durability = TemporalDurability(
    sandbox_providers=[MySandboxProvider(SandboxClient.from_environment())]
)
agent = Agent('anthropic:claude-sonnet-5', name='workspace_agent', capabilities=[durability])


@workflow.defn
class ExistingWorkspaceWorkflow:
    @workflow.run
    async def run(self, sandbox_id: str) -> str:
        result = await agent.run(
            'Inspect the workspace and fix the failing tests.',
            sandbox=SandboxRef(provider='my-sandbox', sandbox_id=sandbox_id),
        )
        return result.output
```

A run argument wins over every capability contribution, so the run uses the referenced sandbox and
never destroys it. Pydantic AI reconstructs a deferred [`Sandbox`][pydantic_ai.sandboxes.Sandbox]
inside the durable I/O boundary; the first operation connects once and caches the live backend for
that activity, step, or task. `provider` and `sandbox_id` are readable before connection, but the
synchronous `sandbox.backend` property raises until an async operation has connected the facade.

- **[Temporal](durable_execution/temporal.md)** serializes the reference into
  `TemporalRunContext` and rebuilds the deferred facade in activities. Workflow code may inspect
  its identity, but must not call sandbox operations because connecting performs I/O.
- **[DBOS](durable_execution/dbos.md)** pickles the reference as a workflow input. Recovery
  rebuilds a fresh facade whose provider reaches the same `sandbox_id`. Effectful sandbox tools
  must still run as DBOS steps, like any other tool I/O.
- **[Prefect](durable_execution/prefect.md)** runs tools in-process, so the default local sandbox
  remains available. A deferred facade contributes `(provider, sandbox_id)` to task cache keys
  without connecting. The fresh framework default and `UnavailableSandbox` add no sandbox
  component.

With neither a `ManagedSandbox` nor a `SandboxRef`, Temporal and DBOS keep their
`UnavailableSandbox` default. A live backend is still rejected because it cannot cross their
durable boundaries. `LocalSandbox` has no provider: its worker-local temporary directory cannot
survive worker replacement.

Rules of thumb for provider authors:

- **`connect()` re-opens, never creates.** If the sandbox was reaped while the workflow slept, an
  open-or-create fallback silently swaps in an empty environment that the model's message history
  contradicts. Recreate only as an explicit, logged decision.
- **`teardown()` tolerates an already-gone sandbox.** It also runs after a failure that may have
  destroyed the environment already.
- **Still set a server-side TTL.** A terminated workflow runs no cleanup; without a TTL or reaper,
  the sandbox leaks.
- **Ids only in `SandboxRef`.** The reference is recorded in workflow history; credentials belong
  on the provider.
