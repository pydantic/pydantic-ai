---
title: Capability Storage
description: "Keep the files harness capabilities store for themselves, such as spilled tool output, in a workspace of their own instead of the run's workspace."
---

# Capability Storage

Keep the files harness capabilities store for themselves in a location you choose, instead of the
run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/). Use it when the model's
file and shell tools should not reach those files, when the run's workspace is read-only, when a
run has no workspace at all, or when each tenant of a service needs its own location.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import CapabilityStorage, ToolOutputLimits

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        CapabilityStorage(LocalWorkspaceBackend('/var/lib/myapp/agent-files')),
        ToolOutputLimits(),
    ],
)
```

`CapabilityStorage` takes a workspace backend, not a workspace capability such as `LocalWorkspace`.
Any backend that supports file operations works, including a sandbox provider's backend. It
contributes no tools, instructions, or hooks.

## What moves

Without `CapabilityStorage`, harness capabilities keep their own files in the run's workspace,
under `.pydantic-ai-harness/` in its working directory. With it, that directory is created in the
storage workspace's working directory instead, with the same layout and a `.gitignore` holding `*`.

| Capability | Files | Where they go with `CapabilityStorage` |
|---|---|---|
| [Tool Output Limits](tool-output-limits.md) | Spilled tool output | `.pydantic-ai-harness/tool-output/` in the storage workspace, unless its `store=` names a location of its own |
| [Shell](shell.md) | Background job logs | Unchanged: the run's workspace, because the commands that write them run there |

Spills in the storage workspace are outside the model's file tools, so the model reads them with
`read_tool_result`. A run with `CapabilityStorage` can spill even when it has no workspace of its own
or its workspace is read-only.

A store passed to a capability explicitly takes precedence: `ToolOutputLimits(store=LocalFileStore())`
or `ToolOutputLimits(store=WorkspaceStore(workspace=...))` keeps writing where that store points.

## One location per tenant

Pass `CapabilityStorage` per run to give each tenant, user, or conversation its own location:

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import CapabilityStorage, ToolOutputLimits

agent = Agent('anthropic:claude-fable-5', capabilities=[ToolOutputLimits()])


async def answer(tenant_id: str, prompt: str) -> str:
    storage = CapabilityStorage(LocalWorkspaceBackend(f'/var/lib/myapp/tenants/{tenant_id}'))
    result = await agent.run(prompt, capabilities=[storage])
    return result.output
```

Two `CapabilityStorage` capabilities on one run are one configuration stated twice: the later one
wins, so a run keeps its capabilities' files in one place.

## Durable execution

The storage workspace's I/O runs in-process. Under a durable engine such as Temporal it is not
routed through the workflow, unlike the run's own workspace. A durable agent also refuses
`capabilities=` on a run, so set `CapabilityStorage` on the agent there. See
[Durable execution](durable-execution.md).

## Telemetry

`CapabilityStorage` emits no spans. It makes no decisions of its own; the capabilities that read and
write through it report their own work.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/capability_storage/).

## API reference

::: pydantic_ai_harness.capability_storage.CapabilityStorage
