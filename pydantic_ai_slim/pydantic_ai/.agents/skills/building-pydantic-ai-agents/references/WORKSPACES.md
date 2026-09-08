# Workspaces

Attach a workspace to a run and use `ctx.workspace` in tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import LocalWorkspace

agent = Agent('openai:gpt-5.2')


@agent.tool
async def execute(ctx: RunContext[None], command: list[str]) -> str:
    result = await ctx.workspace.run(command, timeout=30)
    return result.stdout


async def main() -> None:
    async with LocalWorkspace() as workspace:
        await agent.run('Inspect the project.', workspace=workspace)
```

`LocalWorkspace` runs host subprocesses and provides no isolation. Use it only for trusted work.
Without an attached workspace, operations raise `UserError`. `Workspace` offers the same run,
file, and bounded-read methods for every backend; wrappers can override primitives and
`ReadOnlyWorkspace` blocks commands and changes.

An explicit backend or facade passed through `workspace=` is used directly. Otherwise configured
capabilities receive an explicit `WorkspaceRef`, the latest `ModelResponse.workspace_ref` from
message history, or `None` when there is no reference. A latest `None` suppresses older references.
History supplies identity, not provider configuration. Exactly one capability may supply a workspace;
multiple suppliers and unrecognized explicit `WorkspaceRef` inputs raise. With no supplier, the unavailable
default explains how to attach a workspace. `get_workspace` runs after `for_run`, is synchronous, and
must perform no I/O. A capability should return `None` for references it does not own.

A provider backend keeps credentials and its SDK client, exposes a typed awaitable native handle as
`workspace`, and owns a lock/cache plus private `_create_or_attach(ref)`. No ref creates and publishes
`WorkspaceRef(provider=..., id=...)`; a ref attaches or raises if the environment is gone. The core
does not manage provider lifecycle at run boundaries. The application owns SDK retries, cleanup, TTL,
and pause/stop operations.

`result.workspace` continues with a live backend; `result.workspace.ref` lets another worker attach.
Persist the ref in application state when a durable workflow must resume. Temporal's default context
serializes `workspace_ref` as JSON. Restore a live backend and any wrappers in an application custom
`TemporalRunContext.deserialize_run_context` configured through
`TemporalDurability(run_context_type=...)`; tools still call `ctx.workspace.run(...)` inside the
durable boundary.

See the [workspace guide](https://ai.pydantic.dev/workspace/) for protocol details and lifecycle
examples.
