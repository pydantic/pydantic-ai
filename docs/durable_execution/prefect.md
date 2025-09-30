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
                    Clients
            (HTTP, CLI, API, etc.)
                        |
                        v
+------------------------------------------------------+
|               Application Process                    |
|                                                      |
|   +----------------------------------------------+   |
|   |        Pydantic AI + Prefect Libraries       |   |
|   |                                              |   |
|   |  [ Flows (Agent Run Loop) ]                  |   |
|   |  [ Tasks (Tool, MCP, Model) ]                |   |
|   |  [ Result Persistence ] [ Caching ]          |   |
|   +----------------------------------------------+   |
|                                                      |
+------------------------------------------------------+
                        |
                        v (optional)
+------------------------------------------------------+
|                  Prefect Server/Cloud                |
|      (Monitoring, Scheduling, Orchestration)         |
+------------------------------------------------------+
```

See the [Prefect documentation](https://docs.prefect.io/) for more information.

## Durable Agent

Any agent can be wrapped in a [`PrefectAgent`][pydantic_ai.durable_exec.prefect.PrefectAgent] to get durable execution. `PrefectAgent` automatically:

* Wraps [`Agent.run`][pydantic_ai.Agent.run] and [`Agent.run_sync`][pydantic_ai.Agent.run_sync] as Prefect flows.
* Wraps [model requests](../models/overview.md) and [MCP communication](../mcp/client.md) as Prefect tasks.

Custom tool functions and event stream handlers are **not automatically wrapped** by Prefect.
If they involve I/O or non-deterministic behavior, you can explicitly decorate them with `@task` from Prefect.

The original agent, model, and MCP server can still be used as normal outside the Prefect flow.

Here is a simple but complete example of wrapping an agent for durable execution. All it requires is to install Pydantic AI with Prefect:

```bash
pip/uv-add pydantic-ai[prefect]
```

Or if you're using the slim package, you can install it with the `prefect` optional group:

```bash
pip/uv-add pydantic-ai-slim[prefect]
```

```python {title="prefect_agent.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent

agent = Agent(
    'gpt-4o',
    instructions="You're an expert in geography.",
    name='geography',  # (1)!
)

prefect_agent = PrefectAgent(agent)  # (2)!

async def main():
    result = await prefect_agent.run('What is the capital of Mexico?')  # (3)!
    print(result.output)
    #> Mexico City (Ciudad de México, CDMX)
```

1. The agent's `name` is used to uniquely identify its flows and tasks.
2. Wrapping the agent with `PrefectAgent` enables durable execution for all agent runs.
3. [`PrefectAgent.run()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run] works like [`Agent.run()`][pydantic_ai.Agent.run], but runs as a Prefect flow and executes model requests, decorated tool calls, and MCP communication as Prefect tasks.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

For more information on how to use Prefect in Python applications, see their [Python documentation](https://docs.prefect.io/v3/develop/write-flows).

## Prefect Integration Considerations

When using Prefect with Pydantic AI agents, there are a few important considerations to ensure workflows behave correctly.

### Agent and Toolset Requirements

Each agent instance must have a unique `name` so Prefect can correctly identify and track its flows and tasks.

Tools and event stream handlers are not automatically wrapped by Prefect. You can decide how to integrate them:

* Decorate with `@task` from Prefect if the function involves I/O or needs retry/caching behavior.
* Skip the decorator if the function is simple and doesn't need task-level durability.
* Prefect tasks can be nested, so you can structure your tooling as needed.

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

Prefect persists task results using [Pydantic's serialization](https://docs.pydantic.dev/latest/concepts/serialization/). This means the [dependencies](../dependencies.md) object provided to [`PrefectAgent.run()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run] or [`PrefectAgent.run_sync()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run_sync], and tool outputs should be serializable using Pydantic's `TypeAdapter`. You may also want to keep the inputs and outputs reasonably sized for optimal performance.

### Streaming

Because Prefect tasks consume their entire execution before returning results, [`Agent.run_stream()`][pydantic_ai.Agent.run_stream] is not supported when running inside of a Prefect flow.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `PrefectAgent` instance and using [`PrefectAgent.run()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run].
The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../agents.md#streaming-all-events).

## Task Configuration

You can customize Prefect task behavior, such as retries and timeouts, by passing [`TaskConfig`][pydantic_ai.durable_exec.prefect.TaskConfig] objects to the `PrefectAgent` constructor:

- `mcp_task_config`: The Prefect task config to use for MCP server communication tasks.
- `model_task_config`: The Prefect task config to use for model request tasks.

For custom tools, you can annotate them directly with [`@task`](https://docs.prefect.io/3.0/develop/write-tasks) from Prefect as needed. These decorators have no effect outside Prefect flows, so tools remain usable in non-Prefect agents.

Example with task configuration:

```python {title="prefect_agent_config.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

agent = Agent(
    'gpt-4o',
    instructions="You're an expert in geography.",
    name='geography',
)

prefect_agent = PrefectAgent(
    agent,
    model_task_config=TaskConfig(
        retries=3,  # Retry up to 3 times
        retry_delay_seconds=1.0,  # Wait 1 second between retries
        timeout_seconds=30.0,  # Timeout after 30 seconds
        persist_result=True,  # Persist task results
    ),
)

async def main():
    result = await prefect_agent.run('What is the capital of France?')
    print(result.output)
    #> Paris
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Task Retries

Prefect provides automatic retry capabilities for tasks that fail. By default, tasks are configured with Prefect's standard retry behavior. You can customize retry policies using [task configuration](#task-configuration).

On top of Prefect's retries, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause requests to be retried more often than expected.

When using Prefect, consider:

* Disabling [HTTP Request Retries](../retries.md) in Pydantic AI
* Turning off your provider API client's retry logic (e.g., setting `max_retries=0` on a [custom `OpenAIProvider` API client](../models/openai.md#custom-openai-client))
* Relying on Prefect's task-level retry configuration for consistency

## Caching and Idempotency

Prefect 3.0 provides built-in caching and transactional semantics. Tasks with identical inputs will not re-execute if their results are already cached. This makes workflows naturally idempotent and resilient to failures.

### Serializable RunContext

The Prefect integration uses a `SerializableRunContext` wrapper to handle cache key serialization. This wrapper:

* **Includes cacheable fields** in the cache key:
  - `deps`: User-provided dependencies (if serializable)
  - `prompt`: User prompt string
  - `tool_call_id`: Tool call identifier
  - `tool_name`: Tool name
  - `retry`: Current retry count
  - `max_retries`: Maximum retry count
  - `run_step`: Current step in the run
  - `tool_call_approved`: Approval status

* **Excludes non-serializable fields** from the cache key:
  - `model`: Contains HTTP clients and other non-serializable objects
  - `usage`: Contains internal state that shouldn't affect caching
  - `messages`: May contain binary data or non-serializable content
  - `tracer`: OpenTelemetry tracer object
  - `retries`: Internal state dict

The wrapper is transparent - all attribute access is delegated to the wrapped `RunContext`, so it can be used as a drop-in replacement wherever `RunContext` is expected.

### Cache Behavior

The Prefect integration uses cache policies that match Prefect's `DEFAULT` behavior (`TASK_SOURCE + RUN_ID + Inputs`):

* **Model requests**: Cached based on task source, run ID, messages, model settings, and request parameters
* **Streaming model requests**: Same as above, plus the serializable fields from `RunContext`
* **MCP tool listing**: Cached based on task source, run ID, and serializable fields from `RunContext`
* **MCP tool calls**: Cached based on task source, run ID, tool name, tool arguments, and serializable fields from `RunContext`

This ensures that:
- Tasks don't return stale cached results inappropriately (due to including task source and run ID)
- Non-serializable objects like HTTP clients don't cause serialization errors
- All relevant inputs are considered for cache invalidation
- User dependencies are included in cache keys when they're serializable

You can override caching behavior using the `cache_policy` parameter in [`TaskConfig`][pydantic_ai.durable_exec.prefect.TaskConfig]. See [Prefect's caching documentation](https://docs.prefect.io/3.0/develop/task-caching) for details.

**Note**: If your `deps` type contains non-serializable objects, those will be automatically excluded from the cache key. To ensure your dependencies are included in cache keys, make sure they're serializable using Pydantic's serialization (e.g., use Pydantic models or basic Python types).

## Observability with Prefect UI

Prefect provides a built-in UI for monitoring flow runs, task executions, and failures. You can:

* View real-time flow run status
* Inspect task execution history and outputs
* Debug failures with full stack traces
* Set up alerts and notifications

To access the Prefect UI, you can either:

1. Use [Prefect Cloud](https://www.prefect.io/cloud) (managed service)
2. Run a local [Prefect server](https://docs.prefect.io/v3/manage/self-host) with `prefect server start`

Pydantic AI also integrates with [Pydantic Logfire](../logfire.md) for detailed observability. When using both Prefect and Logfire, you'll get complementary views:

* **Prefect**: Workflow-level orchestration, task status, and retry history
* **Logfire**: Fine-grained tracing of agent runs, model requests, and tool invocations

For more information about Prefect monitoring, see the [Prefect documentation](https://docs.prefect.io/).

## Advanced: Deployments and Scheduling

Prefect supports deploying flows for scheduled or event-driven execution. While this is beyond the scope of basic Pydantic AI integration, you can:

* Create [Prefect deployments](https://docs.prefect.io/v3/deploy/infrastructure-examples/docker) to run agents on a schedule
* Use [Prefect work pools](https://docs.prefect.io/v3/deploy/dynamic-infra/push-runs-serverless) for distributed execution
* Trigger agent runs via [Prefect automations](https://docs.prefect.io/v3/automate/events/automations-triggers)

For more information, see the [Prefect deployment guides](https://docs.prefect.io/v3/deploy/infrastructure-examples/docker).
