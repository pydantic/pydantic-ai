# Durable Execution with Prefect

[Prefect](https://www.prefect.io/) is a workflow orchestration framework for building resilient data pipelines in Python, natively integrated with Pydantic AI.

## Durable Execution

Prefect 3.0 brings [transactional semantics](https://www.prefect.io/blog/transactional-ml-pipelines-with-prefect-3-0) to your Python workflows, allowing you to group tasks into atomic units and define failure modes. If any part of a transaction fails, the entire transaction can be rolled back to a clean state.

* **Flows** are the top-level entry points for your workflow. They can contain tasks and other flows.
* **Tasks** are individual units of work that can be retried, cached, and monitored independently.

Prefect 3.0's approach to transactional orchestration makes your workflows automatically **idempotent**: rerunnable without duplication or inconsistency across any environment. Every task is executed within a transaction that governs when and where the task's result record is persisted. If the task runs again under an identical context, it will not re-execute but instead load its previous result.

The diagram below shows the overall architecture of an agentic application with Prefect.
Prefect uses client-side task orchestration by default, with optional server connectivity for advanced features like scheduling and monitoring.

```text
            +---------------------+
            |   Prefect Server    |      (Monitoring,
            |      or Cloud       |       scheduling, UI,
            +---------------------+       orchestration)
                     ^
                     |
        Flow state,  |   Schedule flows,
        metadata,    |   track execution
        logs         |
                     |
+------------------------------------------------------+
|               Application Process                    |
|   +----------------------------------------------+   |
|   |              Flow (Agent.run)                |   |
|   +----------------------------------------------+   |
|          |          |                |               |
|          v          v                v               |
|   +-----------+ +------------+ +-------------+       |
|   |   Task    | |    Task    | |    Task     |       |
|   |  (Tool)   | | (MCP Tool) | | (Model API) |       |
|   +-----------+ +------------+ +-------------+       |
|         |           |                |               |
|       Cache &     Cache &          Cache &           |
|       persist     persist          persist           |
|         to           to               to             |
|         v            v                v              |
|   +----------------------------------------------+   |
|   |     Result Storage (Local FS, S3, etc.)     |    |
|   +----------------------------------------------+   |
+------------------------------------------------------+
          |           |                |
          v           v                v
      [External APIs, services, databases, etc.]
```

See the [Prefect documentation](https://docs.prefect.io/) for more information.

## Durable Agent

Add durable execution to any [`Agent`][pydantic_ai.agent.Agent] by attaching the [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] [capability](../capabilities.md). When the agent runs inside a Prefect flow, the capability routes [model requests](../models/overview.md), [tool calls](../tools.md), and [MCP communication](../mcp/client.md) through Prefect tasks. To make a run durable, call `agent.run()` inside a `@flow`.

The agent stays a normal `Agent` everywhere — outside a Prefect flow the capability is transparent, and the original agent, model, and MCP server can still be used as normal.

See [Streaming](#streaming) for event handling inside tasks and flow code.

Here is a simple but complete example of attaching durable execution to an agent. All it requires is to install Pydantic AI with Prefect:

```bash
pip/uv-add pydantic-ai[prefect]
```

Or if you're using the slim package, you can install it with the `prefect` optional group:

```bash
pip/uv-add pydantic-ai-slim[prefect]
```

```python {title="prefect_durability.py" test="skip"}
from prefect import flow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectDurability

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions="You're an expert in geography.",
    name='geography',  # (1)!
    capabilities=[PrefectDurability()],  # (2)!
)


@flow  # (3)!
async def answer(question: str) -> str:
    result = await agent.run(question)
    return result.output


async def main():
    answer_text = await answer('What is the capital of Mexico?')
    print(answer_text)
    #> Mexico City (Ciudad de México, CDMX)
```

1. The agent's `name` is used to uniquely identify its flows and tasks.
2. Attach durability via `capabilities=[...]`. The capability routes model requests, tool calls, and MCP communication through Prefect tasks when the agent runs inside a flow.
3. Wrap `agent.run()` in your own `@flow` to make the run durable.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Because the same agent works inside and outside a Prefect flow, [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] composes with all other [capabilities](../capabilities.md) without each needing a Prefect-specific wrapper variant.

For more information on how to use Prefect in Python applications, see their [Python documentation](https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run).

### Wrapper-agent path (deprecated)

!!! warning "Deprecated"
    [`PrefectAgent`][pydantic_ai.durable_exec.prefect.PrefectAgent] is the original wrapper-agent path for Prefect integration and will be removed in v3. New code should use the [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] capability shown above.

    **When migrating, you must wrap the run in a flow yourself.** `PrefectAgent` wrapped `run` / `run_sync` as a Prefect flow automatically; `PrefectDurability` deliberately does not — a run is only durable when `agent.run()` is called inside your own `@flow`. Porting the constructor arguments but calling `agent.run()` directly produces a run that works but is **not durable**.

Any agent can be wrapped in a [`PrefectAgent`][pydantic_ai.durable_exec.prefect.PrefectAgent] to get a durable agent variant that routes model requests, tool calls, and MCP communication through Prefect tasks:

```python {title="prefect_agent.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent

agent = Agent('openai:gpt-5.6-sol', name='geography')
prefect_agent = PrefectAgent(agent)  # Use `prefect_agent` in place of `agent`.
```

Migrating to the capability means attaching `PrefectDurability` and adding the flow decorator that `PrefectAgent` used to apply for you:

```diff
-prefect_agent = PrefectAgent(agent)
-result = await prefect_agent.run(prompt)
+agent = Agent(..., capabilities=[PrefectDurability()])
+
+@flow
+async def answer(prompt: str) -> str:
+    result = await agent.run(prompt)
+    return result.output
```

## Prefect Integration Considerations

When using Prefect with Pydantic AI agents, there are a few important considerations to ensure workflows behave correctly.

### Agent Requirements

Each agent instance must have a unique `name` so Prefect can correctly identify and track its flows and tasks.

Toolsets that implement their own tool listing and calling (i.e. [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPToolset`][pydantic_ai.mcp.MCPToolset]) must have a unique [`id`][pydantic_ai.toolsets.AbstractToolset.id] set, which is used to identify their tasks within the flow.

### Model Selection at Runtime

[`Agent.run(model=...)`][pydantic_ai.agent.Agent.run] supports both model strings (like `'openai:gpt-5.6-sol'`) and model instances. A model instance can't be serialized across the task boundary, so it's sent as its `model_id` string and rebuilt inside the task. That faithfully reproduces model-name strings and models with standard providers, but not an instance whose exact behavior depends on a custom provider, client, or settings — pre-register those by passing a `models` dict to [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] and reference them by key (or pass the registered instance). The agent's own model, set at construction, is always available as the default.

To customize how a model string is built — a custom provider, or per-user credentials carried on the run's `deps` — add a [`ResolveModelId`](../capabilities.md#resolvemodelid) capability before `PrefectDurability`: it gets first crack at every string, and the resolver runs again inside the task with the run's actual `deps`, so it must be deterministic for a given `(model_id, deps)` and must not perform external I/O.

### Tool Wrapping

Agent tools are automatically wrapped as Prefect tasks, which means they benefit from:

* **Retry logic**: Failed tool calls can be retried automatically
* **Caching**: Tool results are cached based on their inputs
* **Observability**: Tool execution is tracked in the Prefect UI

A default [`TaskConfig`][pydantic_ai.durable_exec.prefect.TaskConfig] for all tools can be passed as `tool_task_config` to the [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] constructor. Per-tool config lives on the tool's [`metadata`][pydantic_ai.toolsets.FunctionToolset.tool] field — `PrefectDurability` looks for a `'prefect'` key. You can set the metadata directly on the tool definition, or apply it across a selection of tools via the [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] capability. See the [capabilities documentation][pydantic_ai.capabilities.SetToolMetadata] for the full selector vocabulary.

```python {title="prefect_per_tool_config.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.durable_exec.prefect import PrefectDurability, TaskConfig
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset(id='research')


@toolset.tool(metadata={'prefect': TaskConfig(timeout_seconds=10.0)})  # (1)!
def fetch_data(url: str) -> str: ...


@toolset.tool(metadata={'prefect': False})  # (2)!
def simple_tool() -> str: ...


agent = Agent(
    'openai:gpt-5.6-sol',
    name='research',
    toolsets=[toolset],
    capabilities=[
        SetToolMetadata(  # (3)!
            tools=['fetch_data', 'fetch_dataset'],
            prefect=TaskConfig(timeout_seconds=10.0),
        ),
        PrefectDurability(tool_task_config=TaskConfig(retries=3)),  # (4)!
    ],
)
```

1. Inline: declare the task config alongside the tool definition. Per-tool config merges on top of the base `tool_task_config`.
2. Set `'prefect': False` to skip task wrapping entirely for that tool.
3. Selector-based: [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] applies the same metadata across a selection of tools (`'all'`, a name list, a dict, or a callable).
4. `tool_task_config` sets the default config for every tool.

### Streaming

[`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream], [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events], and [`Agent.iter()`][pydantic_ai.agent.Agent.iter] work inside a Prefect flow, but their events are buffered rather than delivered in real time. The model stream runs inside the durable task, and its events are replayed to the flow after the task completes.

For handlers with I/O side effects, pass `event_stream_handler=` to [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability]. Model events are delivered live inside each model-request task, while each tool event is delivered in its own event-handler task. Configure those per-event tasks with `event_stream_handler_task_config=`. As with any Prefect task, a handler may run more than once if a task retries, so keep its side effects idempotent.

Alternatively, register [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream]. Its handler runs in flow code and must be deterministic because it re-runs on flow replay. Tool and final-output events arrive live, while the real captured model events are replayed after each model request completes. For examples, see the [streaming docs](../agent.md#streaming-all-events).

A durability `event_stream_handler=` and a separately registered `ProcessEventStream` are two distinct handlers, and each fires once. The durability handler receives live events inside the durable task, while `ProcessEventStream` sees the buffered replay in flow code.

A per-run handler passed to `Agent.run(event_stream_handler=...)` also runs flow-side against replayed model events.

Because the model stream is consumed inside the task, cancelling it from the flow side (e.g. with [`AgentStream.cancel()`][pydantic_ai.result.AgentStream.cancel]) is not available across the durable boundary.

### Suspended Turns and Background Mode

When a provider pauses a model turn mid-flight (Anthropic `pause_turn`) or runs it as a server-side job that's polled until it's ready ([OpenAI background mode](../models/openai.md#background-mode)), each segment runs in a separate model request task. The suspended [`ModelResponse`][pydantic_ai.messages.ModelResponse] and background job ID are checkpointed between segments, while the final response is merged and usage is recorded once. A [`message_history`](../message-history.md) ending in a suspended response is passed to the first task. Size `timeout_seconds` in [Task Configuration](#task-configuration) for one provider round trip. If an error abandons a suspended job, its provider teardown runs in a dedicated cancellation task.

### Toolsets at Runtime

Additional toolsets can be passed per run via `agent.run(toolsets=...)`, but only toolsets that don't need durable wrapping are supported: non-executing toolsets like [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset], whose tools are executed outside the agent run, and [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset]s whose tools all opt out of task wrapping with `metadata={'prefect': False}`. Other executing toolsets ([`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPToolset`][pydantic_ai.mcp.MCPToolset]) and dynamic toolsets must be set when constructing the agent so their tasks are registered before the flow runs; passing them at runtime raises a `UserError`.

## Task Configuration

You can customize Prefect task behavior, such as retries and timeouts, by passing [`TaskConfig`][pydantic_ai.durable_exec.prefect.TaskConfig] objects to the [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] constructor:

- `mcp_task_config`: Configuration for MCP server communication tasks
- `model_task_config`: Configuration for model request tasks
- `event_stream_handler_task_config`: Configuration for event stream handler tasks
- `tool_task_config`: Default configuration for all tool calls (per-tool overrides go on the tool's `'prefect'` metadata — see [Tool Wrapping](#tool-wrapping) above)

Available `TaskConfig` options:

- `retries`: Maximum number of retries for the task (default: `0`)
- `retry_delay_seconds`: Delay between retries in seconds (can be a single value or list for exponential backoff, default: `1.0`)
- `timeout_seconds`: Maximum time in seconds for the task to complete
- `cache_policy`: Custom Prefect cache policy for the task
- `persist_result`: Whether to persist the task result
- `result_storage`: Prefect result storage for the task (e.g., `'s3-bucket/my-storage'` or a `WritableFileSystem` block)
- `log_prints`: Whether to log print statements from the task (default: `False`)

Example:

```python {title="prefect_durability_task_config.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectDurability, TaskConfig

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions="You're an expert in geography.",
    name='geography',
    capabilities=[
        PrefectDurability(
            model_task_config=TaskConfig(
                retries=3,
                retry_delay_seconds=[1.0, 2.0, 4.0],  # Exponential backoff
                timeout_seconds=30.0,
            ),
        ),
    ],
)


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> Paris
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

### Retry Considerations

Pydantic AI and provider API clients have their own retry logic. When using Prefect, you may want to:

* Disable [HTTP Request Retries](../retries.md) in Pydantic AI
* Turn off your provider API client's retry logic (e.g., `max_retries=0` on a [custom OpenAI client](../models/openai.md#custom-openai-client))
* Rely on Prefect's task-level retry configuration for consistency

This prevents requests from being retried multiple times at different layers.

## Caching and Idempotency

Prefect 3.0 provides built-in caching and transactional semantics. Tasks with identical inputs will not re-execute if their results are already cached, making workflows naturally idempotent and resilient to failures.

* **Task inputs**: Messages, settings, parameters, tool arguments, and serializable dependencies

**Note**: For user dependencies to be included in cache keys, they must be serializable (e.g., Pydantic models or basic Python types). Non-serializable dependencies are automatically excluded from cache computation.

## Observability with Prefect and Logfire

Prefect provides a built-in UI for monitoring flow runs, task executions, and failures. You can:

* View real-time flow run status
* Debug failures with full stack traces
* Set up alerts and notifications

To access the Prefect UI, you can either:

1. Use [Prefect Cloud](https://www.prefect.io/cloud) (managed service)
2. Run a local [Prefect server](https://docs.prefect.io/v3/how-to-guides/self-hosted/server-cli) with `prefect server start`

You can also use [Pydantic Logfire](../logfire.md) for detailed observability. When using both Prefect and Logfire, you'll get complementary views:

* **Prefect**: Workflow-level orchestration, task status, and retry history
* **Logfire**: Fine-grained tracing of agent runs, model requests, and tool invocations

When using Logfire with Prefect, you can enable distributed tracing to see spans for your Prefect runs included with your agent runs, model requests, and tool invocations.

For more information about Prefect monitoring, see the [Prefect documentation](https://docs.prefect.io/).

## Deployments and Scheduling

To deploy and schedule a Prefect-durable agent, wrap it in a Prefect flow and use the flow's [`serve()`](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-serve) or [`deploy()`](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python) methods:

```python {title="serve_agent.py" test="skip"}
from prefect import flow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectDurability


@flow
async def daily_report_flow(user_prompt: str):
    """Generate a daily report using the agent."""
    agent = Agent(  # (1)!
        'openai:gpt-5.6-sol',
        name='daily_report_agent',
        instructions='Generate a daily summary report.',
        capabilities=[PrefectDurability()],
    )

    result = await agent.run(user_prompt)
    return result.output


# Serve the flow with a daily schedule
if __name__ == '__main__':
    daily_report_flow.serve(
        name='daily-report-deployment',
        cron='0 9 * * *',  # Run daily at 9am
        parameters={'user_prompt': "Generate today's report"},
        tags=['production', 'reports'],
    )
```

1. Each flow run executes in an isolated process, and all inputs and dependencies must be serializable. Because Agent instances cannot be serialized, instantiate the agent inside the flow rather than at the module level.

The `serve()` method accepts scheduling options:

- **`cron`**: Cron schedule string (e.g., `'0 9 * * *'` for daily at 9am)
- **`interval`**: Schedule interval in seconds or as a timedelta
- **`rrule`**: iCalendar RRule schedule string

For production deployments with Docker, Kubernetes, or other infrastructure, use the flow's [`deploy()`](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python) method. See the [Prefect deployment documentation](https://docs.prefect.io/v3/how-to-guides/deployments/create-deploymentsy) for more information.
