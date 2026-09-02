# Sandboxes

Sandboxes give an agent a workspace where its tools can run commands and work with files. Attach
an environment to a run, then use [`ctx.sandbox`][pydantic_ai.tools.RunContext.sandbox] inside
your tools. For trusted local development, the smallest complete example uses
[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox]:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.sandboxes import LocalSandbox

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

!!! warning "`LocalSandbox` does not isolate code"
    [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] runs commands as host subprocesses and
    reads and writes the host filesystem. It is suitable only for trusted development and
    tests, which is why it is opt-in. Attach a container-, VM-, or remote-backed sandbox before
    exposing command execution or file access to untrusted input.

Commands run by `LocalSandbox` inherit only `PATH`, `HOME`, `LANG` and `TMPDIR`, plus `env`; other
parent variables, including provider API keys, are not inherited by default, but `HOME` and the
host filesystem remain available. Output is capped at 10 MiB: redirect noisy output to a file and
read a window instead. A background process can delay return by up to the two-second drain grace.

Every interface that owns a run takes the same `sandbox=` argument, and none of them attaches a
sandbox for you:

```python {title="sandbox_to_cli.py" test="skip"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.sandboxes import LocalSandbox

agent = Agent('anthropic:claude-sonnet-5')
agent.to_cli_sync(sandbox=LocalSandbox(root=Path.cwd()))
```

## Read files without flooding model context

Use [`Sandbox.read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] instead of loading a whole
file into the model's context:

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext[None], path: str, offset: int = 1) -> str:
    window = await ctx.sandbox.read_file(path, offset=offset, limit=200)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix
```

Relative paths resolve against the sandbox's working directory. For complete files, use
[`read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] and
[`write_text()`][pydantic_ai.sandboxes.Sandbox.write_text].

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

If more than one capability returns a reference, the run releases them all and raises. Deferred
capabilities are not asked, because they load after the sandbox is chosen.

### Directly, per run

Pass any [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] through `sandbox=`. Create it
before the run and tear it down afterward, as shown in the [first example](#sandboxes). Pass the
same backend to several runs when they should share one workspace.

### Manage sandboxes automatically

A capability can provision a sandbox automatically for each run. This is useful for applications
that create containers or remote environments on demand:

- [`acquire_sandbox`][pydantic_ai.capabilities.AbstractCapability.acquire_sandbox] creates or
  selects an environment and returns its serializable identity, or `None` to decline.
- [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox] connects to it when a
  sandbox operation first runs.
- [`release_sandbox`][pydantic_ai.capabilities.AbstractCapability.release_sandbox] cleans up after
  a run whose acquisition returned a reference.

```python {title="sandbox_capability.py"}
from dataclasses import dataclass
from typing import Any

from my_sandboxes import SandboxClient

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    client: SandboxClient  # credentials stay here, never in the ref

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        sandbox = await self.client.create(idempotency_key=ctx.run_id)
        return SandboxRef(sandbox_id=sandbox.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        # Re-open only: raise if the environment expired. Never create a replacement here.
        return await self.client.connect(ref.sandbox_id)

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        await self.client.destroy(ref.sandbox_id)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[MySandboxCapability(SandboxClient.from_environment())],
)
```

Pydantic AI closes the connection `get_sandbox` returned when the run ends; a live backend passed
through `sandbox=` stays open and caller-owned.

Choose the lifecycle that matches your application:

| Lifecycle | `acquire_sandbox` | `get_sandbox` | `release_sandbox` |
|---|---|---|---|
| Fresh sandbox per run | provision, return its ref | connect by id | destroy |
| Warm environment shared across runs | return its ref | open a run-scoped connection | inherited no-op |
| Pooled per conversation | check out or create by `ctx.conversation_id` | connect | return to pool or decrement a reference count |
| Environment fixed by configuration | return its ref | connect by id | don't override |
| Connect-only (the caller passes a `SandboxRef`) | don't override | connect by id | don't override |

If a workspace must survive several runs, use a warm or pooled lifecycle rather than creating a
fresh sandbox per run. This includes [tool approval](deferred-tools.md), where pausing and resuming
spans two runs.

### Disabling execution with a policy reason

Pass [`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox] as
`sandbox=UnavailableSandbox(reason='Local execution is disabled by application policy.')` to
prevent capabilities from attaching a sandbox and give attempted operations a useful error.

### Making a sandbox read-only

Wrap a backend in [`ReadOnlySandbox`][pydantic_ai.sandboxes.ReadOnlySandbox] when an agent should
inspect a workspace without changing it:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.sandboxes import LocalSandbox, ReadOnlySandbox

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def read_workspace_file(ctx: RunContext[None], path: str) -> str:
    return await ctx.sandbox.read_text(path)


async def main() -> None:
    async with LocalSandbox() as sandbox:
        root = await sandbox.working_dir()
        await sandbox.fs.write_bytes(f'{root}/data.csv', b'a,b\n1,2\n')
        await agent.run(
            'Summarize data.csv in the working directory.',
            sandbox=ReadOnlySandbox(sandbox),
        )
```

File reads and directory listings work; commands and file changes raise
[`UserError`][pydantic_ai.exceptions.UserError]. If the agent must run commands against protected
data, enforce read-only access in the environment itself, for example with a read-only mount.

## Build a sandbox integration

A backend is required to implement only three members: `sandbox_id`, command
execution, and its working directory. Pydantic AI exposes it to tools through a
[`Sandbox`][pydantic_ai.sandboxes.Sandbox] object, which adds text decoding, path resolution,
and line-window reads. Filesystem operations use the backend's
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] implementation.

| Need | [`Sandbox`][pydantic_ai.sandboxes.Sandbox] API | Backend support |
|---|---|---|
| Execute a command | [`run()`][pydantic_ai.sandboxes.Sandbox.run] | — |
| Read/write files | [`fs`][pydantic_ai.sandboxes.Sandbox.fs] / [`read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] / [`write_text()`][pydantic_ai.sandboxes.Sandbox.write_text] | [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] (else `NotImplementedError`) |
| Windowed read | [`read_file()`][pydantic_ai.sandboxes.Sandbox.read_file] | `sed` over `run()`, with [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] fallback (else `NotImplementedError`) |
| Working directory | [`working_dir()`][pydantic_ai.sandboxes.Sandbox.working_dir] | — |
| Path resolution | [`resolve()`][pydantic_ai.sandboxes.Sandbox.resolve] | Handled by `Sandbox` |

The protocol contracts that matter to callers:

- Optional operations raise `NotImplementedError`; use the documented fallback.
- `timeout=` guarantees the command is terminated before
  [`SandboxTimeoutError`][pydantic_ai.sandboxes.SandboxTimeoutError] is raised; its `stdout` and
  `stderr` attributes contain output produced before termination when the backend can recover it.
- Backends raise [`SandboxUnavailableError`][pydantic_ai.sandboxes.SandboxUnavailableError] when
  the environment is permanently unusable and consumers should stop retrying it.
- Backends raise [`SandboxError`][pydantic_ai.sandboxes.SandboxError] for deliberate recoverable
  operation failures; catch specific subclasses before the base class.
- A filesystem reports a missing path with the builtin `FileNotFoundError`, and its `stat()` and
  `list_dir()` entries can reuse the concrete [`FileEntry`][pydantic_ai.sandboxes.FileEntry]
  carrier instead of declaring their own.
- A non-zero `exit_code` is a normal result, not an exception. `run()` results can reuse the
  concrete [`CommandResult`][pydantic_ai.sandboxes.CommandResult] carrier instead of declaring
  their own.

These translations are a backend's whole error-handling duty; wrapping other SDK failures is
optional.

[`SandboxBackend.run()`][pydantic_ai.sandboxes.SandboxBackend.run] returns complete captured
output. Truncating it in a tool bounds model context, not the backend's memory; bound untrusted
output in the command itself (for example with `tail`).

!!! warning "The sandbox protocol is not a security boundary"
    Isolation comes from the backend environment. In particular,
    [`resolve()`][pydantic_ai.sandboxes.Sandbox.resolve] only normalizes text: `..` can escape the
    base directory and symlinks are not inspected. Enforce confinement in the sandbox itself.

[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] is the reference implementation. A custom
backend works without registration when it implements the relevant protocols.

## Durable execution

Tools still use `ctx.sandbox` unchanged under [Temporal](durable_execution/temporal.md),
[DBOS](durable_execution/dbos.md), and [Prefect](durable_execution/prefect.md). Only the
[`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] crosses the durable boundary; the worker reconnects
through `get_sandbox` when a durable activity, step, or task uses the sandbox.

Use the lifecycle capability [shown above](#manage-sandboxes-automatically) for a sandbox owned by
one run. If the sandbox is provisioned elsewhere and should outlive the run, pass its reference
instead:

```python
from pydantic_ai import SandboxRef

sandbox = SandboxRef(sandbox_id='sandbox-123')
```

Pass that value through `sandbox=`. The agent must also have a capability whose `get_sandbox` can
connect the reference. Because the caller owns this sandbox, the run does not acquire or release
it. Do not pass a live backend or `LocalSandbox` into a durable run; they cannot cross the durable
boundary.

For reliable durable lifecycles:

- make `acquire_sandbox` and `release_sandbox` idempotent because durable operations may retry;
- make `get_sandbox` reconnect an existing environment rather than silently create an empty one;
- keep credentials on the capability, never in `SandboxRef` or workflow history;
- configure a provider-side TTL or reaper because a cancelled workflow may not run cleanup.

See the relevant durable-execution guide for engine-specific retry and task configuration.
