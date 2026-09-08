# Workspaces

Workspaces give an agent a workspace where its tools can run commands and work with files. Attach
an environment to a run, then use [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace] inside
your tools. For trusted local development, the smallest complete example uses
[`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace]:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalWorkspace() as workspace:
        await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=workspace)
```

The tool does not need to know whether the attached environment is a local process, container,
VM, or remote service. It uses the same [`Workspace`][pydantic_ai.workspaces.Workspace] methods in
each case:

- [`run()`][pydantic_ai.workspaces.Workspace.run] runs a command and returns its output and exit code.
- [`read_file()`][pydantic_ai.workspaces.Workspace.read_file] returns a line window suitable for
  model context.
- [`read_text()`][pydantic_ai.workspaces.Workspace.read_text] and
  [`write_text()`][pydantic_ai.workspaces.Workspace.write_text] work with complete text files.

!!! warning "`LocalWorkspace` does not isolate code"
    [`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace] runs commands as host subprocesses and
    reads and writes the host filesystem. It is suitable only for trusted development and
    tests, which is why it is opt-in. Attach a container-, VM-, or remote-backed workspace before
    exposing command execution or file access to untrusted input.

Commands run by `LocalWorkspace` inherit only `PATH`, `HOME`, `LANG` and `TMPDIR`, plus `env`; other
parent variables, including provider API keys, are not inherited by default, but `HOME` and the
host filesystem remain available. Output is capped at 10 MiB: redirect noisy output to a file and
read a window instead. A background process can delay return by up to the two-second drain grace.

Every interface that owns a run takes the same `workspace=` argument, and none of them attaches a
workspace for you:

```python {title="workspace_to_cli.py" test="skip"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5')
agent.to_cli_sync(workspace=LocalWorkspace(root=Path.cwd()))
```

## Read files without flooding model context

Use [`Workspace.read_file()`][pydantic_ai.workspaces.Workspace.read_file] instead of loading a whole
file into the model's context:

```python
from pydantic_ai import RunContext


async def read_source(ctx: RunContext[None], path: str, offset: int = 1) -> str:
    window = await ctx.workspace.read_file(path, offset=offset, limit=200)
    suffix = '\n[more lines available]' if window.has_more else ''
    return window.text + suffix
```

Relative paths resolve against the workspace's working directory. For complete files, use
[`read_text()`][pydantic_ai.workspaces.Workspace.read_text] and
[`write_text()`][pydantic_ai.workspaces.Workspace.write_text].

## Safety and policy

Workspace access is opt-in. If you do not attach one, workspace operations raise
[`UserError`][pydantic_ai.exceptions.UserError] with instructions for attaching an environment;
Pydantic AI never silently runs commands or reads files on the host.

Pydantic AI provides the connection, not model-facing command or file tools. Your application
chooses which operations to expose and enforces approval, command, path, timeout, and output
rules in those tools. If commands must be constrained, use argv form and validate each argument;
do not rely on a denylist over free-form shell strings.

## Choose where the code runs

Pydantic AI chooses one workspace for the run, in this order:

1. The environment passed through `workspace=`.
2. The environment supplied by one active capability.
3. The unavailable default, which explains how to attach one when a tool tries to use it.

If more than one capability returns a backend, the run raises when the second answer is found.
Backends are lazy, so this selection does not create an environment that needs cleanup. Deferred
capabilities are not asked, because they load after the workspace is chosen.

### Directly, per run

Pass any [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] through `workspace=`. Create it
before the run and tear it down afterward, as shown in the [first example](#workspaces). Pass the
same backend to several runs when they should share one workspace.

### Supply a workspace from a capability

A capability can supply the run's workspace, which is useful for applications that create containers
or remote environments on demand. There is one hook:

[`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace] runs once per run after
[`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] has selected the per-run
capability instances, but before the run lifecycle hooks. It returns a backend or `None` to
decline. It is synchronous and must not touch the network: it hands back a backend built from your
own settings, and that backend creates or attaches the first time somebody runs a command.

`ref` is the identity of an environment the run should continue in when the caller passed a
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] through `workspace=`. `None` means make a fresh one.
Pydantic AI does not infer workspace identity from message history; pass `result.workspace` to another
run to reuse its live backend, or pass its `ref` when the next run should reconnect through a
capability.

The backend holds your settings and, if the run is continuing an environment, its identity. Remote
providers should keep their typed native workspace behind a property, acquire it lazily in a private
`_get_workspace()` method, and implement `_create_or_attach(ref)` with provider-specific locking,
caching, cleanup, and reconnect behavior. [`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace]
shows the lazy acquisition pattern for a host directory; provider integrations show the corresponding
typed native SDK implementation.

The capability then just builds one:

```python {title="workspace_capability.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces import LocalWorkspace


@dataclass
class MyWorkspaceCapability(AbstractCapability[Any]):
    root: str

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
        return LocalWorkspace(self.root)


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[MyWorkspaceCapability(root='/tmp/my-workspace', id='my_workspace')],
)
```

`workspace` returns something you can only `await`, never call a method on directly, so the connect
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
| Warm the workspace up before the model runs | `before_run`, calling something harmless like `await ctx.workspace.working_dir()` |
| Copy files in, or mount storage | `before_run` |
| Copy results out, or pause the environment | `after_run` |
| Clean up even when the run fails or is cancelled | `wrap_run`, with `try`/`finally` |
| Destroy it for good | your own code, after the run, through `result.workspace` |

Environment lifetime and idle cleanup are provider and application configuration.

#### Carrying on where a run left off

A finished run hands back the workspace it used, so the environment and its files are still there
afterwards. Read an artifact out of it, or pass it to the next run and keep working in the same
workspace:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=60)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalWorkspace() as workspace:
        first = await agent.run('Write fizzbuzz to fizzbuzz.py and run it.', workspace=workspace)
        await agent.run('Now add a test for it.', workspace=first.workspace)
```

The same value works for a subagent: pass `ctx.workspace` from a tool and the subagent shares the
workspace instead of getting one of its own.

### Disabling execution with a policy reason

Pass [`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] as
`workspace=UnavailableWorkspace(reason='Local execution is disabled by application policy.')` to
prevent capabilities from attaching a workspace and give attempted operations a useful error.

### Making a workspace read-only

Wrap a workspace in [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] when an agent should
inspect a workspace without changing it:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace, ReadOnlyWorkspace, Workspace

agent = Agent('anthropic:claude-sonnet-5')


@agent.tool
async def read_workspace_file(ctx: RunContext[None], path: str) -> str:
    return await ctx.workspace.read_text(path)


async def main() -> None:
    async with LocalWorkspace() as workspace:
        root = await workspace.working_dir()
        await workspace.write_bytes(f'{root}/data.csv', b'a,b\n1,2\n')
        await agent.run(
            'Summarize data.csv in the working directory.',
            workspace=ReadOnlyWorkspace(Workspace(workspace)),
        )
```

File reads and directory listings work; commands and file changes raise
[`UserError`][pydantic_ai.exceptions.UserError]. If the agent must run commands against protected
data, enforce read-only access in the environment itself, for example with a read-only mount.

Policies compose by overriding primitive operations on [`WrapperWorkspace`][pydantic_ai.workspaces.WrapperWorkspace]:

```python
from pydantic_ai.workspaces import WrapperWorkspace, Workspace


class LoggingWorkspace(WrapperWorkspace):
    async def read_bytes(self, path: str) -> bytes:
        print(f'reading {path}')
        return await self.wrapped.read_bytes(path)
```

Higher-level helpers such as `read_text()` and `read_file()` use the overridden primitive. A
wrapper can therefore add behavior around reads or writes without forwarding every helper method.

## Build a workspace integration

A backend is required to implement only three members: its `ref`, command execution, and its
working directory. Pydantic AI exposes it to tools through a
[`Workspace`][pydantic_ai.workspaces.Workspace] object, which adds text decoding, path resolution,
and line-window reads. Filesystem methods are always available: `Workspace` prefers a backend's
flat, native [`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem] implementation and
otherwise derives the same operations from `run()` using standard shell utilities.

| Need | [`Workspace`][pydantic_ai.workspaces.Workspace] API | Backend support |
|---|---|---|
| Execute a command | [`run()`][pydantic_ai.workspaces.Workspace.run] | — |
| Read/write files | [`read_bytes()`][pydantic_ai.workspaces.Workspace.read_bytes] / [`read_text()`][pydantic_ai.workspaces.Workspace.read_text] / [`write_text()`][pydantic_ai.workspaces.Workspace.write_text] | Native [`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or shell fallback over `run()` |
| Windowed read | [`read_file()`][pydantic_ai.workspaces.Workspace.read_file] | Bounded `sed` over `run()`, then the ordinary native-or-shell filesystem path |
| Working directory | [`working_dir()`][pydantic_ai.workspaces.Workspace.working_dir] | — |
| Path resolution | [`resolve()`][pydantic_ai.workspaces.Workspace.resolve] | Handled by `Workspace` |

The protocol contracts that matter to callers:

- `WorkspaceBackend` stays small; implement flat `SupportsFilesystem` methods when the provider has
  a better native file API. `Workspace` supplies a complete shell fallback when it does not.
- `timeout=` guarantees the command is terminated before
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] is raised; its `stdout` and
  `stderr` attributes contain output produced before termination when the backend can recover it.
- Backends raise [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] when
  the environment is permanently unusable and consumers should stop retrying it.
- Backends raise [`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] for deliberate recoverable
  operation failures; catch specific subclasses before the base class.
- A filesystem reports a missing path with the builtin `FileNotFoundError`, and its `stat()` and
  `list_dir()` entries can reuse the concrete [`FileEntry`][pydantic_ai.workspaces.FileEntry]
  carrier instead of declaring their own.
- A non-zero `exit_code` is a normal result, not an exception. `run()` results can reuse the
  concrete [`CommandResult`][pydantic_ai.workspaces.CommandResult] carrier instead of declaring
  their own.

These translations are a backend's whole error-handling duty; wrapping other SDK failures is
optional.

[`WorkspaceBackend.run()`][pydantic_ai.workspaces.WorkspaceBackend.run] returns complete captured
output. Truncating it in a tool bounds model context, not the backend's memory; bound untrusted
output in the command itself (for example with `tail`).

!!! warning "The workspace protocol is not a security boundary"
    Isolation comes from the backend environment. In particular,
    [`resolve()`][pydantic_ai.workspaces.Workspace.resolve] only normalizes text: `..` can escape the
    base directory and symlinks are not inspected. Enforce confinement in the workspace itself.

[`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace] is the reference implementation. A custom
backend works without registration when it implements the relevant protocols.

## Durable execution

Tools and capability hooks still use `ctx.workspace` unchanged under
[Temporal](durable_execution/temporal.md), [DBOS](durable_execution/dbos.md), and
[Prefect](durable_execution/prefect.md). A `Workspace` method called directly from replayed workflow
code executes inside the durable tool or capability activity you configure. The live backend never
crosses the boundary; its
[`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef], the method arguments, and the serializable run
context do. The worker reconnects through the exact capability that supplied the workspace.
Accessing `workspace.backend` in workflow code is rejected because calling a provider-specific
method directly would bypass durable execution.

The supplying capability needs an explicit stable reference. Use the capability [shown above](#supply-a-workspace-from-a-capability) to have the
agent pick the workspace. If the environment is made elsewhere, pass its reference instead:

```python
from pydantic_ai import WorkspaceRef

workspace = WorkspaceRef(provider='my-provider', id='workspace-123')
```

Pass that value through `workspace=`. The agent must also have a capability whose `get_workspace`
recognizes the reference. Do not pass a live backend or `LocalWorkspace` into a durable run; neither
can cross the durable boundary.

For reliable durable lifecycles:

- make the backend's create-or-attach step safe to run twice, because durable operations may retry;
- reconnect an existing environment rather than quietly making an empty one in its place;
- keep credentials on the capability, never in `WorkspaceRef` or workflow history;
- configure a provider-side TTL or reaper because a cancelled workflow may not run cleanup.

See the relevant durable-execution guide for engine-specific retry and task configuration.
