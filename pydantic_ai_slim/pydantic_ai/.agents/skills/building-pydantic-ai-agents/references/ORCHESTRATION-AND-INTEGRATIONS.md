# Orchestration and Integrations

Read this file when the user wants multi-agent coordination, graphs, direct model calls, A2A, durable execution, embeddings, evals, or third-party integrations.

## Coordinate Multiple Agents

Use agent delegation when one agent should call another and return the result.

```python
from pydantic_ai import Agent, RunContext

parent = Agent('openai:gpt-5.2', name='parent_agent')
researcher = Agent('openai:gpt-5.2', name='researcher_agent', output_type=str)


@parent.tool
async def research(ctx: RunContext, topic: str) -> str:
    result = await researcher.run(f'Research: {topic}', usage=ctx.usage)
    return result.output
```

Delegating tools and output functions must be `async def` and use `await delegate.run(...)`; never call
`run_sync()` or `run_stream_sync()` inside them. The parent may still use `run_sync()` at the application boundary.

Good split:

- delegation via tools when the parent keeps control
- output functions or programmatic hand-off when control should move elsewhere

## Build Multi-Step Workflows with Graphs

Use `pydantic_graph` when the workflow is a state machine rather than a single agent loop. Compose graphs with `GraphBuilder` and typed step functions:

```python
from pydantic_graph import GraphBuilder, StepContext

g = GraphBuilder(input_type=int, output_type=int)


@g.step
async def increment(ctx: StepContext[None, None, int]) -> int:
    return ctx.inputs + 1


@g.step
async def double(ctx: StepContext[None, None, int]) -> int:
    return ctx.inputs * 2


g.add(
    g.edge_from(g.start_node).to(increment),
    g.edge_from(increment).to(double),
    g.edge_from(double).to(g.end_node),
)

graph = g.build()
result = graph.run_sync(inputs=3)
```

Use `await graph.run(inputs=...)` from async code.

## Call the Model Without Using an Agent

Use the direct API when the user wants a single model request without agent orchestration.

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

response = model_request_sync(
    'openai:gpt-5.2',
    [ModelRequest.user_text_prompt('Summarize this in one sentence.')],
)
```

Reach for this when there is no need for tools, retries, or agent loop state.

## Use Durable Execution

Use the durable execution integrations when the run must survive crashes, retries, or long-lived workflows.

Temporal entry points:

- `Agent(..., capabilities=[TemporalDurability(...)])`
- `PydanticAIWorkflow`
- `PydanticAIPlugin`
- `AgentPlugin`

`TemporalAgent`, `DBOSAgent`, and `PrefectAgent` are deprecated wrapper agents.

Pass every executing toolset that needs durable wrapping to the agent constructor. In particular, construct a `DynamicToolset` with an explicit `id` and pass it to `Agent(toolsets=[...])`; the `@agent.toolset` decorator registers after the engine's durable units were created. Toolsets that arrive later — via the decorator, `run(toolsets=...)`, `override(toolsets=...)`, or a per-run capability — are never wrapped for durable execution: under the `*Durability` capability, using one inside a workflow or flow raises a `UserError` (DBOS still accepts a `FunctionToolset`, whose tools it runs inline either way). The deprecated wrapper agents don't run this check — inside a workflow or flow they run the toolset list frozen at wrap time, so a toolset registered that late is silently left out. A toolset added at run time also cannot reuse a construction-time toolset's `id`.

Temporal and DBOS register durable units before their workers start, so attach capabilities at agent construction time. Passing `run(capabilities=[...])` inside one of their workflows raises a `UserError` unless the capability is the observer-only `Instrumentation`; broader support for observer-only capabilities is tracked in [#5477](https://github.com/pydantic/pydantic-ai/issues/5477), where users can share their use cases. Prefect creates tasks per call, so it has no such registration boundary and accepts a per-run capability that contributes no executing toolset; one that does is still rejected by the runtime-toolset guard.

A run-time `model=` inside a workflow must be a model-name string or an instance registered in the durability capability's `models=`. An unregistered `Model` instance raises a `UserError`: it can't be serialized into the activity/step/task, and rebuilding it from its `model_id` would build a different model. To build a specific instance inside the durable unit (e.g. per-user credentials from `deps`), pass a string and use a `ResolveModelId` capability.

## Handle MCP Tool Errors

Set `MCPToolset(tool_error_behavior=...)` according to the server error semantics:

- `'retry'` asks the model to correct and retry the call. This is the default.
- `'failed'` reports a completed failed result via `ToolFailed` without consuming retry budget.
- `'error'` propagates the underlying exception to application code.

Structured server error content is serialized as JSON for both `'retry'` and `'failed'`. Protocol and transport errors (as opposed to completed tool errors) stay retryable even under `'failed'`.

## Use Embeddings for RAG

Use `Embedder(...)` for query/document embeddings when the user is building retrieval or semantic search.

```python
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')
```

## Use LangChain Tools

Third-party integrations to reach for:

- `tool_from_langchain`
- `LangChainToolset`

Use these when the user explicitly wants the LangChain ecosystem instead of native Pydantic AI tools.

## Systematically Verify Agent Behavior with Evals

Use `pydantic_evals` when the user wants repeatable evaluation datasets and evaluators rather than ad hoc tests.

Common entry points:

- `Case`
- `Dataset`
- evaluator classes from `pydantic_evals.evaluators`

## Build Custom Toolsets, Models, or Agents

Extensibility entry points:

- `AbstractToolset` / `WrapperToolset`
- `Model` / `WrapperModel`
- `AbstractAgent` / `WrapperAgent`
- `AbstractCapability`

Reach for these only when the built-in primitives are genuinely insufficient.
