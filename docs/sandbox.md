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

### Supply a sandbox from a capability

A capability can supply the run's sandbox, which is useful for applications that create containers
or remote environments on demand. There is one hook:

[`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox] runs once per run, before
any other hook, and returns a backend or `None` to decline. It is synchronous and must not touch
the network: it hands back a backend built from your own settings, and that backend creates or
attaches the first time somebody runs a command.

`ref` is the identity of an environment the run should continue in, either from an explicit
`sandbox=` argument or from a previous run in the same conversation. `None` means make a fresh one.

The backend holds your settings and, if the run is continuing an environment, its identity. Keep
the environment behind a property so no method can use it without connecting first:

```python {title="my_backend.py"}
from collections.abc import Awaitable
from typing import Any

import anyio

from pydantic_ai.sandboxes import CommandResult, SandboxCommand, SandboxRef


class MyBackend:
    def __init__(self, *, client: Any, ref: SandboxRef | None, name: str | None):
        self.client, self.ref, self.name = client, ref, name
        self._sandbox: Any | None = None
        self._lock = anyio.Lock()

    @property
    def sandbox(self) -> Awaitable[Any]:
        return self._resolve()

    async def _resolve(self) -> Any:
        async with self._lock:
            if self._sandbox is None:
                if self.ref is not None:
                    self._sandbox = await self.client.connect(self.ref.sandbox_id)
                else:
                    self._sandbox = await self.client.create(name=self.name)
                    self.ref = SandboxRef(sandbox_id=self._sandbox.id)
        return self._sandbox

    async def run(self, command: SandboxCommand, **kwargs: Any) -> CommandResult:
        sandbox = await self.sandbox
        return await sandbox.exec(command, **kwargs)

    async def working_dir(self) -> str:
        sandbox = await self.sandbox
        return sandbox.working_dir
```

The capability then just builds one:

```python {title="sandbox_capability.py" requires="my_backend.py"}
from dataclasses import dataclass
from typing import Any

from my_backend import MyBackend
from my_sandboxes import SandboxClient

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef


@dataclass
class MySandboxCapability(AbstractCapability[Any]):
    client: SandboxClient  # credentials stay here, never in the ref

    def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
        return MyBackend(client=self.client, ref=ref, name=ctx.conversation_id)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[MySandboxCapability(SandboxClient.from_environment())],
)
```

`sandbox` returns something you can only `await`, never call a method on directly, so the connect
step cannot be skipped by accident. The lock means two tools running at once still produce one
environment.

Exactly one attached capability may return a backend. Two raise
[`UserError`][pydantic_ai.exceptions.UserError] naming both.

#### Starting and stopping is yours

Pydantic AI never creates, closes, destroys or pauses an environment. A conversation can span many
runs, so ending a run does not mean the workspace is finished with.

If you want something to happen around a run, use the ordinary hooks:

| You want | Where it goes |
|---|---|
| Warm the sandbox up before the model runs | `before_run`, calling something harmless like `await ctx.sandbox.working_dir()` |
| Copy files in, or mount storage | `before_run` |
| Copy results out, or pause the environment | `after_run` |
| Clean up even when the run fails or is cancelled | `wrap_run`, with `try`/`finally` |
| Destroy it for good | your own code, after the run, through `result.sandbox` |

Most providers stop charging for an idle environment on their own, so doing nothing is usually the
right answer.

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

A backend is required to implement only three members: its `ref`, command execution, and its
working directory. Pydantic AI exposes it to tools through a
[`Sandbox`][pydantic_ai.sandboxes.Sandbox] object, which adds text decoding, path resolution,
and line-window reads. Filesystem operations use the backend's
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] implementation.

| Need | [`Sandbox`][pydantic_ai.sandboxes.Sandbox] API | Backend support |
|---|---|---|
| Execute a command | [`run()`][pydantic_ai.sandboxes.Sandbox.run] | — |
| Read/write files | [`read_bytes()`][pydantic_ai.sandboxes.Sandbox.read_bytes] / [`read_text()`][pydantic_ai.sandboxes.Sandbox.read_text] / [`write_text()`][pydantic_ai.sandboxes.Sandbox.write_text] | [`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem] (else `NotImplementedError`) |
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

Use the capability [shown above](#supply-a-sandbox-from-a-capability) to have the agent pick the
sandbox. If the environment is made elsewhere, pass its reference instead:

```python
from pydantic_ai import SandboxRef

sandbox = SandboxRef(sandbox_id='sandbox-123')
```

Pass that value through `sandbox=`. The agent must also have a capability whose `get_sandbox`
recognizes the reference. Do not pass a live backend or `LocalSandbox` into a durable run; neither
can cross the durable boundary.

For reliable durable lifecycles:

- make the backend's create-or-attach step safe to run twice, because durable operations may retry;
- reconnect an existing environment rather than quietly making an empty one in its place;
- keep credentials on the capability, never in `SandboxRef` or workflow history;
- configure a provider-side TTL or reaper because a cancelled workflow may not run cleanup.

See the relevant durable-execution guide for engine-specific retry and task configuration.
