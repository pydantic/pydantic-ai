# Workspaces

Attach a workspace capability to the agent and use `ctx.workspace` in tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace

agent = Agent('openai:gpt-5.2', capabilities=[LocalWorkspace('~/project')])


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=30)
    return result.stdout
```

`LocalWorkspace(working_dir, *, read_only=False, env=None)` runs host subprocesses and provides no
isolation: `working_dir` is only the default directory and the base for relative paths, not a jail
or a security boundary. Use it only for trusted work. `working_dir` is required; a relative path
such as `'.'` resolves against the current directory at construction, and a leading `~` is
expanded; the caller owns that directory. Commands inherit only `PATH` and
`HOME` from the agent process, with `env` and then the per-call `env` layered on top. Never pass `os.environ` wholesale: it hands the
model's commands every secret in the process, LLM API keys included. `read_only=True` wraps it in `ReadOnlyWorkspace`. Two
`LocalWorkspace`s share the default id `local_workspace` and combine into the last one (none of
the earlier one's settings carry over); give one a distinct `id` to keep both.
Its ref is `WorkspaceRef(provider='local', id=<absolute working_dir>)` from construction (the
directory is not checked then; the first operation raises `WorkspaceUnavailableError` if it is
missing), and the capability
claims only that exact ref: a foreign ref, or a local ref for another directory, gets `None`, so message history
cannot redirect the agent to another host directory (pass `workspace='new'` to start over in the
configured one). For a single run, pass the backend instead:
`agent.run(..., workspace=LocalWorkspaceBackend('.'))`, or
`workspace=ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend('.')))` for a read-only run (outside
durable execution; see below).
Without an attached workspace, operations raise `UserError`; a capability that needs one checks
`ctx.workspace.attached` in `before_run` and raises a `UserError` naming what to attach. `Workspace` offers the same run
and file methods for every backend; `WrapperWorkspace` is the base for policy wrappers (override
operations, delegate the rest to `self.wrapped`), `ReadOnlyWorkspace` blocks commands and changes,
and `workspace.read_only` lets a tool provider leave write tools out. To disable workspace access
for a run on purpose, pass `workspace=UnavailableWorkspace(reason=...)`.

`resolve()` is textual; `realpath()` asks the environment to resolve symlinks in the existing
components (native through `SupportsRealpath`, `readlink` in the shell otherwise).

Outside a durable container, an explicit backend passed through `workspace=` is used directly, and
a `Workspace` facade or wrapper (`ReadOnlyWorkspace(...)`, `result.workspace`, `ctx.workspace`) is
kept as-is (inside one, see below). An explicit
`WorkspaceRef` is offered to configured capabilities, and raises if none recognizes it.
`workspace='new'` ignores any ref in message history and asks capabilities with `ref=None` for a
fresh workspace, raising if none supplies one. With `workspace=None`, capabilities receive the
latest `ModelResponse.workspace_ref` from message history; a latest `None` suppresses older ones. History
supplies identity, not provider configuration. Precedence is: explicit `workspace=`, then the
history ref, then a fresh workspace from the capability. An agent may have several workspace
capabilities, like `resolve_model_id`: they are asked in order and the first that returns a
workspace wins, so listing a new provider's capability before the old one moves new conversations
while old ones continue where they started. A history ref that no capability recognizes raises
`UserError` (pass `workspace='new'` to start fresh), unless the agent has no workspace capability at
all (a summarizer given the history), which ignores it. With no ref and no supplier the run gets an
unattached placeholder whose operations raise `UserError` explaining how to attach one.
`get_workspace` runs before `for_run` (a capability that only a `for_run` contributes is asked
afterwards, and `for_run` may not change a selection made before it), is synchronous, and must have
no side effects or I/O. A capability must return `None` for references it does not own.

A `WorkspaceRef` names an environment that exists, and exists only once it does. A backend built
without a ref reports `ref is None`, creates the environment on its first operation, and sets `ref`
as soon as the create call returns; a backend built with a ref attaches on its first operation and
raises `WorkspaceUnavailableError` if the environment is gone, never creating a replacement. A ref
is never a label: no name-, conversation- or uuid-derived ref before the environment exists.
`get_workspace` is the only path from a ref back to a workspace. The local backend is the one
exception where the ref precedes any operation (the directory is the environment); `working_dir()`
raises `WorkspaceUnavailableError` if the directory is missing. When a run ends, `workspace.ref` is
recorded as `workspace_ref` on its last `ModelResponse` (`None` if no environment was created). A
run on an agent without that provider's capability records `None`, which hides the older ref, so pass
`workspace=result.workspace` (or its ref) to continue after it.

The core does not create or destroy environments at run boundaries; the application owns
SDK retries (outside durable execution), cleanup, TTL and pause/stop through the provider's SDK or run hooks. `Workspace.backend` reaches
the concrete backend for provider-specific methods (not from workflow code under durable
execution). Sandbox providers (Modal, E2B, Daytona, Sprites) ship as capabilities in the
[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/). Check a custom backend by
subclassing `pydantic_ai.workspaces.testing.WorkspaceBackendSuite` and providing its `backend`
fixture.

Exception contract a backend must follow: `WorkspaceUnavailableError` when the environment is gone
or unreachable (ends the run; never reaches the model); `WorkspaceTimeoutError` for a command over
its `timeout=`; builtin file errors (`FileNotFoundError`, `IsADirectoryError`, ...) for path-level
failures the model can act on; `WorkspaceReadOnlyError` for a mutation on a read-only workspace;
`WorkspaceError` for other deliberate refusals; `TypeError`/`ValueError`
for bad arguments. SDK transient errors propagate unchanged (durable engines retry them); if the SDK
cannot tell a dead environment from a failed operation, probe with `working_dir()` and raise
`WorkspaceUnavailableError`. `UserError` belongs to the facade and policy wrappers, not backends.

`result.workspace` continues with a live backend; `result.workspace.ref` lets another worker attach.

Under `TemporalDurability`, `DBOSDurability` or `PrefectDurability`, attach the workspace capability
at agent construction and nothing else is needed: in workflow code (hooks, output functions,
`result.workspace`) every `Workspace` method runs as its own durable unit, and inside a unit (a
tool) `ctx.workspace` is the plain workspace, rebuilt on Temporal from the serialized ref through the
same capabilities (policy wrappers included). One `ensure` unit at run start creates or attaches the
environment and records its ref and working directory, so all units share one environment and
`working_dir()`/`resolve()` need no unit. Inside a container `workspace=` takes `None`, `'new'`, a
`WorkspaceRef`, a previous `result.workspace`, or a live instance whose ref a capability recognizes,
which is rebuilt through that capability. Any wrapper around it, such as `ReadOnlyWorkspace`, is
silently dropped, so put policy on the capability (e.g. `LocalWorkspace(..., read_only=True)`). A
live instance without a recognized ref raises `UserError`. `run`/writes/`make_dir`/`remove` are attempted once by
default; configure with `workspace_activity_config`, `workspace_step_config` or
`workspace_task_config`. The deprecated `TemporalAgent`/`DBOSAgent`/`PrefectAgent` wrappers refuse
workspaces in their container.

See the [workspace guide](https://pydantic.dev/docs/ai/core-concepts/workspace/) for protocol details and lifecycle
examples.
