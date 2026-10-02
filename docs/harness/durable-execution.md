---
title: Durable Execution
description: "Run Pydantic AI Harness workspace capabilities such as Coder, Shell, and FileSystem under Temporal, DBOS, or Prefect, and see exactly which features work on each engine."
---

# Durable Execution

Harness capabilities that act in a [workspace](../workspace.md), such as [`Coder`](coder.md),
[`Shell`](shell.md), [`FileSystem`](filesystem.md), and [`RepoContext`](repo-context.md), work under
[Temporal](../durable_execution/temporal.md), [DBOS](../durable_execution/dbos.md), and
[Prefect](../durable_execution/prefect.md). The run records a reference to its workspace, and every
activity or step reattaches to the environment it names, whichever worker runs it. A later run that
continues the conversation from `message_history`, in a new workflow or flow, works in the
[same environment](../workspace.md#continuing-in-the-same-workspace).

Attach the workspace, the harness capabilities, and the durability capability when you construct the
agent. The examples below run the same `Coder` agent on each engine. Swap `LocalWorkspace` for a
sandbox capability such as [`ModalSandbox`](modal-sandbox.md) to run it in an isolated machine;
nothing else changes.

## Temporal

```python {title="coder_temporal.py"}
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='coder',  # names the agent's Temporal activities
    capabilities=[LocalWorkspace('.'), Coder(), TemporalDurability()],
)


@workflow.defn
class CoderWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='coder', workflows=[CoderWorkflow]):
        output = await client.execute_workflow(
            CoderWorkflow.run, 'Add a test for fizzbuzz.py and run it.', id=f'coder-{uuid.uuid4()}', task_queue='coder'
        )
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
```

## DBOS

```python {title="coder_dbos.py"}
import asyncio

from dbos import DBOS, DBOSConfig
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai_harness.coder import Coder

dbos_config: DBOSConfig = {'name': 'coder', 'system_database_url': 'sqlite:///coder.sqlite'}
DBOS(config=dbos_config)

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='coder',
    capabilities=[LocalWorkspace('.'), Coder(), DBOSDurability()],
)


@DBOS.workflow()
async def run(prompt: str) -> str:
    return (await agent.run(prompt)).output


async def main() -> None:
    DBOS.launch()
    print(await run('Add a test for fizzbuzz.py and run it.'))


if __name__ == '__main__':
    asyncio.run(main())
```

## Prefect

```python {title="coder_prefect.py"}
import asyncio

from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.prefect import PrefectDurability
from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='coder',
    capabilities=[LocalWorkspace('.'), Coder(), PrefectDurability()],
)


@flow
async def run(prompt: str) -> str:
    return (await agent.run(prompt)).output


if __name__ == '__main__':
    print(asyncio.run(run('Add a test for fizzbuzz.py and run it.')))
```

## What works where

| Feature | Temporal | DBOS | Prefect |
|---|---|---|---|
| File tools: read, write, edit, search | Yes | Yes | Yes |
| Commands: `run_command`, `shell` | Yes, within the [activity timeout](#temporal-timeouts) | Yes | Yes |
| Background jobs: `start_command`, `check_command`, `stop_command` | Yes | Yes | Yes |
| Sticky `cd`: `Shell(persist_cwd=True)` | Yes | Yes | Yes |
| File-change approval: [`FileChangeRequestEvent`](filesystem.md#events) | Yes | Yes | Yes |
| Live tool events | [File-change requests and write events only](#live-events-on-temporal) | Yes | Yes |
| Delegation: `delegate_task` | [To `self` only](#delegation-on-temporal) | Yes | Yes |

### Live events on Temporal

Tools that write files ask for approval and report the write from the workflow, so
`FileChangeRequestEvent`, `FileWrittenEvent`, `FileEditedEvent`, and `DirectoryCreatedEvent` reach your
listeners as they happen. Other tools run in activities, and their events are not delivered:
`FileReadEvent`, `DirectoryListedEvent`, and `FilesSearchedEvent` from the file tools, and
`CommandStartedEvent`, `CommandOutputEvent`, and `CommandFinishedEvent` from `shell`
([pydantic-ai#7971](https://github.com/pydantic/pydantic-ai/issues/7971)). The tools themselves work
and return their results as usual.

When Temporal replays a workflow, approval listeners run again, so make their external effects
idempotent.

### Delegation on Temporal

`Coder`'s delegate, `self`, runs in the workflow, so the child's model requests and tools become
activities of the same agent. Sub-agents read from `agent_folders` are not supported: reading the
folders at run start adds a toolset at run time, which Temporal refuses, so pass
`SubAgents(include_self=True, agent_folders=None)` when you compose `SubAgents` yourself.

### Temporal timeouts

An activity has 60 seconds by default. `run_command` and `shell` get 300 seconds, which covers the
`shell` tool's 270-second foreground wait. A `run_command` given a longer timeout, through
`Shell(default_timeout=...)` or the model's `timeout_seconds`, needs a longer activity:

```python
from datetime import timedelta

from pydantic_ai.capabilities import SetToolMetadata
from temporalio.workflow import ActivityConfig

SetToolMetadata(tools=['run_command'], temporal=ActivityConfig(start_to_close_timeout=timedelta(minutes=30)))
```

Add it to the agent's capabilities next to `TemporalDurability()`. For work longer than that, use
`start_command` or `shell(mode='background')`, which return while the command keeps running.

### Engine notes

- **DBOS** runs a workspace run's tool calls one at a time, so recovery replays each call's recorded result.
- **Capabilities that make requests from their tools**, [`ExaSearch` and `ExaAgent`](exa-search.md),
  [`YouSearch` and `YouResearch`](youdotcom.md), [`LocalStack`](localstack.md) against an external instance, and
  [`CapabilityCreation`](capability-creation.md), record each request on every engine, including DBOS,
  which otherwise runs function tools in workflow code. Recovery reuses the recorded result. The records
  are named after each capability's default `id`, so keep it stable once deployed.
- **Sticky `cd`** is kept in the workspace under `.pydantic-ai-harness/shell/run-state/`, keyed by the
  run ID, so a new worker continues in the same directory after a restart. Without an explicit
  `run_id`, a durable run's ID comes from its workflow or flow run and survives worker recovery. The
  file is removed when the run completes. Commands in one run should execute in order; simultaneous
  commands that change directory can overwrite each other's state.
- **Background jobs** are keyed by the run and tool-call IDs, so a retried `start_command` reattaches to
  the job instead of starting a second process. If a launch fails after claiming its job directory, the
  retry reports a pending launch; once the process has stopped, remove the stale job files under
  `.pydantic-ai-harness/shell/`.
- **Sandboxes**: if a worker dies after creating a sandbox but before the run records its reference,
  the retry creates a second sandbox and the first is left running. Clean it up with the provider's
  tools. See [Durable execution](../workspace.md#durable-execution) in the workspace guide for how
  workspace calls retry and time out on each engine.
- **Deploying changes**: removing a capability while workflows that use it are still running changes
  their replay history. Drain those workflows, or on Temporal use
  [worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning),
  before you deploy the change.
