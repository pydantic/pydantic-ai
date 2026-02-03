# Durable Execution Reference

Source: `docs/durable_execution/overview.md`, `docs/durable_execution/temporal.md`, `docs/durable_execution/dbos.md`, `docs/durable_execution/prefect.md`

## Overview

Durable execution frameworks ensure agent runs survive failures, restarts, and deployments.
PydanticAI integrates with three durable execution platforms, each using only public interfaces.

Key benefits:
- **Fault tolerance**: Automatic recovery from transient API failures
- **State persistence**: Preserve progress across application restarts
- **Long-running workflows**: Support for human-in-the-loop and async workflows
- **Full streaming support**: Works with MCP servers and event stream handlers

## Temporal

Install: `pydantic-ai-slim[temporal]`

Temporal provides workflow orchestration with automatic retries and state persistence via a replay mechanism.

**Key concepts:**
- **Workflows**: Deterministic code that orchestrates activities (the agent run loop)
- **Activities**: Non-deterministic operations with I/O (model requests, tool calls, MCP)

### Basic Setup

```python
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    PydanticAIPlugin,
    PydanticAIWorkflow,
    TemporalAgent,
)

agent = Agent(
    'openai:gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # Required: uniquely identifies activities
)

temporal_agent = TemporalAgent(agent)  # Wrap for durable execution


@workflow.defn
class GeographyWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [temporal_agent]  # Register agents

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent.run(prompt)
        return result.output


async def main():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin()],  # Pydantic serialization + activity registration
    )

    async with Worker(
        client,
        task_queue='geography',
        workflows=[GeographyWorkflow],
    ):
        output = await client.execute_workflow(
            GeographyWorkflow.run,
            args=['What is the capital of Mexico?'],
            id='geography-unique-id',
            task_queue='geography',
        )
        print(output)
```

### Agent and Toolset Requirements

Each agent must have a unique `name` for stable activity identification:

```python
agent = Agent('openai:gpt-5', name='my_unique_agent')

@agent.toolset(id='my_tools')  # Required ID for @agent.toolset
async def my_dynamic_toolset(ctx: RunContext[Deps]) -> Toolset:
    return FunctionToolset([my_tool])
```

### Activity Configuration

Customize timeouts and retries per activity type:

```python
from temporalio.workflow import ActivityConfig

temporal_agent = TemporalAgent(
    agent,
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)),
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    toolset_activity_config={'my_tools': ActivityConfig(retry_policy=RetryPolicy(maximum_attempts=5))},
    tool_activity_config={('my_tools', 'slow_tool'): ActivityConfig(start_to_close_timeout=timedelta(minutes=5))},
)
```

### Model Selection at Runtime

Pre-register models for Temporal's replay mechanism:

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel

temporal_agent = TemporalAgent(
    agent,
    models={
        'fast': AnthropicModel('claude-sonnet-4-5'),
        'reasoning': OpenAIResponsesModel('gpt-5.2'),
    },
)

# In workflow:
result = await temporal_agent.run(prompt, model='fast')  # Use registered model
result = await temporal_agent.run(prompt, model='openai:gpt-4.1-mini')  # Model strings work too
```

### Streaming with Event Stream Handler

Since `run_stream()` is not supported, use event stream handlers:

```python
async def my_stream_handler(ctx: RunContext[Deps], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in events:
        # Write to external system (message queue, websocket, etc.)
        await my_queue.send(event)

agent = Agent('openai:gpt-5', name='streaming_agent', event_stream_handler=my_stream_handler)
temporal_agent = TemporalAgent(agent)
```

### Temporal Characteristics

- Supports workflows spanning minutes to hours/days
- Replay-based recovery from failures
- Built-in compensation logic for rollbacks
- Requires external Temporal Server infrastructure
- Strong consistency guarantees via event sourcing

## DBOS

Install: `pydantic-ai-slim[dbos]`

DBOS provides durable execution backed by PostgreSQL or SQLite with in-process checkpointing.

**Key concepts:**
- **Workflows**: Deterministic code checkpointed in database
- **Steps**: I/O operations with automatic checkpoint and recovery

### Basic Setup

```python
from dbos import DBOS, DBOSConfig

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent

dbos_config: DBOSConfig = {
    'name': 'pydantic_dbos_agent',
    'system_database_url': 'sqlite:///dbostest.sqlite',  # SQLite for dev
    # 'system_database_url': 'postgresql://...',  # Postgres for production
}
DBOS(config=dbos_config)

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # Required: uniquely identifies workflows
)

dbos_agent = DBOSAgent(agent)  # Must be defined before DBOS.launch()

async def main():
    DBOS.launch()
    result = await dbos_agent.run('What is the capital of Mexico?')
    print(result.output)
```

### Decorating Tools with DBOS Steps

Tools with I/O should be decorated as DBOS steps:

```python
from dbos import DBOS

@agent.tool_plain
@DBOS.step  # Checkpoint this tool's execution
def fetch_weather(city: str) -> str:
    """Fetch weather data from external API."""
    response = requests.get(f'https://api.weather.com/{city}')
    return response.json()

@agent.tool_plain  # No decorator if no I/O or durability not needed
def calculate_score(data: dict) -> int:
    """Pure computation, no checkpoint needed."""
    return sum(data.values())
```

### Step Configuration

Configure retries for model and MCP steps:

```python
from pydantic_ai.durable_exec.dbos import DBOSAgent, StepConfig

dbos_agent = DBOSAgent(
    agent,
    model_step_config=StepConfig(retries=3),
    mcp_step_config=StepConfig(retries=2),
)
```

### Parallel Tool Execution

By default, tools run in parallel with ordered events:

```python
dbos_agent = DBOSAgent(
    agent,
    parallel_execution_mode='sequential',  # Or use 'parallel_ordered_events' (default)
)
```

### Streaming with Event Stream Handler

Use event stream handlers since `run_stream()` is not supported:

```python
async def my_handler(ctx: RunContext[Deps], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in events:
        print(event)

agent = Agent('openai:gpt-5', name='agent', event_stream_handler=my_handler)
dbos_agent = DBOSAgent(agent)
```

### DBOS Characteristics

- In-process library (no external orchestrator)
- Uses PostgreSQL or SQLite for state persistence
- Checkpoint-based recovery
- Lower infrastructure complexity
- Tools require manual `@DBOS.step` decoration for durability

## Prefect

Install: `pydantic-ai-slim[prefect]`

Prefect provides workflow orchestration with transactional semantics and built-in caching.

**Key concepts:**
- **Flows**: Top-level entry points for workflows (agent runs)
- **Tasks**: Individual units of work with retry, caching, observability (tools, model requests)

### Basic Setup

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # Required: identifies flows and tasks
)

prefect_agent = PrefectAgent(agent)

async def main():
    result = await prefect_agent.run('What is the capital of Mexico?')
    print(result.output)
```

### Task Configuration

Configure retries, timeouts, and caching:

```python
from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

prefect_agent = PrefectAgent(
    agent,
    model_task_config=TaskConfig(
        retries=3,
        retry_delay_seconds=[1.0, 2.0, 4.0],  # Exponential backoff
        timeout_seconds=30.0,
    ),
    tool_task_config=TaskConfig(retries=2),  # Default for all tools
    tool_task_config_by_name={
        'slow_tool': TaskConfig(timeout_seconds=60.0),
        'fast_tool': None,  # Disable task wrapping for this tool
    },
    mcp_task_config=TaskConfig(retries=2),
    event_stream_handler_task_config=TaskConfig(log_prints=True),
)
```

### TaskConfig Options

| Option | Type | Description |
|--------|------|-------------|
| `retries` | `int` | Maximum retry attempts (default: 0) |
| `retry_delay_seconds` | `float \| list[float]` | Delay between retries |
| `timeout_seconds` | `float` | Maximum execution time |
| `cache_policy` | `CachePolicy` | Custom Prefect cache policy |
| `persist_result` | `bool` | Persist task result |
| `result_storage` | `str \| WritableFileSystem` | Where to store results |
| `log_prints` | `bool` | Log print statements |

### Streaming Behavior

`run_stream()` works but doesn't provide real-time streaming inside Prefect flows. Use event stream handlers for real-time streaming:

```python
async def my_handler(ctx: RunContext[Deps], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in events:
        # Each event is wrapped as a Prefect task for durability
        print(event)

agent = Agent('openai:gpt-5', name='agent', event_stream_handler=my_handler)
prefect_agent = PrefectAgent(agent)
```

### Deployments and Scheduling

Deploy agents with Prefect's scheduling:

```python
from prefect import flow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent


@flow
async def daily_report_flow(user_prompt: str):
    agent = Agent('openai:gpt-5', name='daily_report')  # Create inside flow
    prefect_agent = PrefectAgent(agent)
    result = await prefect_agent.run(user_prompt)
    return result.output


if __name__ == '__main__':
    daily_report_flow.serve(
        name='daily-report-deployment',
        cron='0 9 * * *',  # Daily at 9am
        parameters={'user_prompt': "Generate today's report"},
    )
```

### Prefect Characteristics

- Transactional semantics with automatic idempotency
- Built-in task caching based on inputs
- Optional server connectivity (works standalone)
- Built-in scheduling (cron, interval, rrule)
- Automatic tool wrapping as Prefect tasks
- Native UI for monitoring flow runs

## Choosing a Platform

| Feature | Temporal | DBOS | Prefect |
|---------|----------|------|---------|
| **State backend** | Temporal Server | PostgreSQL/SQLite | Local/Cloud storage |
| **Deployment** | Requires server | In-process library | Optional server |
| **Complexity** | High | Low-Medium | Medium |
| **Best for** | Complex workflows | DB-centric apps | Data pipelines |
| **Streaming** | Event handler only | Event handler only | Event handler (wrapped) |
| **Tool wrapping** | Automatic | Manual `@DBOS.step` | Automatic |
| **Caching** | Via activities | Via steps | Built-in |
| **Scheduling** | Via server | Via queues | Built-in |

## Retry Considerations

All platforms have retry capabilities that may conflict with Pydantic AI and provider SDK retries. Recommendation:

```python
# Disable Pydantic AI HTTP retries when using durable execution
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(api_key='...', http_client=httpx.AsyncClient())
# Don't use AsyncTenacityTransport

# Disable provider SDK retries
from openai import AsyncOpenAI
client = AsyncOpenAI(max_retries=0)

# Rely on durable execution platform retries instead
```

## Observability

All platforms integrate with Logfire for comprehensive tracing:

**Temporal:**
```python
from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin

client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin(), LogfirePlugin()])
```

**DBOS:** Configure OpenTelemetry export in DBOS settings (see DBOS docs).

**Prefect:** Use Logfire alongside Prefect UI for workflow-level and trace-level observability.

## When NOT to Use Durable Execution

- Simple request-response agents (just use `agent.run()`)
- Short-lived operations that can be retried at the application level
- When the overhead of a durable execution framework is not justified
- Latency-sensitive applications where checkpoint overhead is unacceptable

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `TemporalAgent` | `pydantic_ai.durable_exec.temporal` | Temporal wrapper |
| `PydanticAIWorkflow` | `pydantic_ai.durable_exec.temporal` | Base workflow class |
| `PydanticAIPlugin` | `pydantic_ai.durable_exec.temporal` | Temporal client plugin |
| `LogfirePlugin` | `pydantic_ai.durable_exec.temporal` | Logfire integration |
| `DBOSAgent` | `pydantic_ai.durable_exec.dbos` | DBOS wrapper |
| `StepConfig` | `pydantic_ai.durable_exec.dbos` | DBOS step configuration |
| `PrefectAgent` | `pydantic_ai.durable_exec.prefect` | Prefect wrapper |
| `TaskConfig` | `pydantic_ai.durable_exec.prefect` | Prefect task configuration |
