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
* Wraps [model requests](../models/overview.md) as Prefect tasks.
* Wraps [tool calls](../tools.md) as Prefect tasks (configurable per-tool).
* Wraps [MCP communication](../mcp/client.md) as Prefect tasks.

Event stream handlers are **not automatically wrapped** by Prefect. If they involve I/O or non-deterministic behavior, you can explicitly decorate them with `@task` from Prefect.

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

### Agent Requirements

Each agent instance must have a unique `name` so Prefect can correctly identify and track its flows and tasks.

### Tool Wrapping

Agent tools are automatically wrapped as Prefect tasks, which means they benefit from:

* **Retry logic**: Failed tool calls can be retried automatically
* **Caching**: Tool results are cached based on their inputs
* **Observability**: Tool execution is tracked in the Prefect UI

You can customize tool task behavior using `tool_task_config` (applies to all tools) or `tool_task_config_by_name` (per-tool configuration):

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

agent = Agent('gpt-4o', name='my_agent')

@agent.tool_plain
def fetch_data(url: str) -> str:
    # This tool will be wrapped as a Prefect task
    ...

prefect_agent = PrefectAgent(
    agent,
    tool_task_config=TaskConfig(retries=3),  # Default for all tools
    tool_task_config_by_name={
        'fetch_data': TaskConfig(timeout_seconds=10.0),  # Specific to fetch_data
        'simple_tool': None,  # Disable task wrapping for simple_tool
    },
)
```

Set a tool's config to `None` in `tool_task_config_by_name` to disable task wrapping for that specific tool.

### Agent Run Context and Dependencies

Prefect persists task results using [Pydantic's serialization](https://docs.pydantic.dev/latest/concepts/serialization/). This means the [dependencies](../dependencies.md) object provided to [`PrefectAgent.run()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run] or [`PrefectAgent.run_sync()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run_sync], and tool outputs should be serializable using Pydantic's `TypeAdapter`. You may also want to keep the inputs and outputs reasonably sized for optimal performance.

### Streaming

Because Prefect tasks consume their entire execution before returning results, [`Agent.run_stream()`][pydantic_ai.Agent.run_stream] is not supported when running inside of a Prefect flow.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `PrefectAgent` instance and using [`PrefectAgent.run()`][pydantic_ai.durable_exec.prefect.PrefectAgent.run].
The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../agents.md#streaming-all-events).

## Task Configuration

You can customize Prefect task behavior, such as retries and timeouts, by passing [`TaskConfig`][pydantic_ai.durable_exec.prefect.TaskConfig] objects to the `PrefectAgent` constructor:

- `mcp_task_config`: Configuration for MCP server communication tasks
- `model_task_config`: Configuration for model request tasks
- `tool_task_config`: Default configuration for all tool calls
- `tool_task_config_by_name`: Per-tool task configuration (overrides `tool_task_config`)

Available `TaskConfig` options:

- `retries`: Maximum number of retries for the task
- `retry_delay_seconds`: Delay between retries in seconds (can be a single value or list for exponential backoff)
- `timeout_seconds`: Maximum time in seconds for the task to complete
- `cache_policy`: Custom Prefect cache policy for the task
- `persist_result`: Whether to persist the task result
- `log_prints`: Whether to log print statements from the task

Example:

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
        retries=3,
        retry_delay_seconds=[1.0, 2.0, 4.0],  # Exponential backoff
        timeout_seconds=30.0,
    ),
)

async def main():
    result = await prefect_agent.run('What is the capital of France?')
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

### How Caching Works

The Prefect integration caches tasks based on:

* **Task source code**: Different versions of the same task won't share cached results
* **Run ID**: Tasks are scoped to their flow run
* **Task inputs**: Messages, settings, parameters, tool arguments, and serializable dependencies

The integration automatically handles serialization of complex objects like `RunContext` by extracting only the cacheable fields (prompt, tool information, retry counts, and user dependencies if serializable) while excluding non-serializable objects like HTTP clients and tracers.

### Customizing Cache Behavior

You can override caching using the `cache_policy` parameter in [`TaskConfig`][pydantic_ai.durable_exec.prefect.TaskConfig]:

```python
from pydantic_ai.durable_exec.prefect import TaskConfig
from prefect.cache_policies import NONE

prefect_agent = PrefectAgent(
    agent,
    model_task_config=TaskConfig(cache_policy=NONE),  # Disable caching
)
```

See [Prefect's caching documentation](https://docs.prefect.io/3.0/develop/task-caching) for more cache policy options.

**Note**: For user dependencies to be included in cache keys, they must be serializable (e.g., Pydantic models or basic Python types). Non-serializable dependencies are automatically excluded from cache computation.

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
