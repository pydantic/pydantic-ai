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

- `TemporalAgent`
- `PydanticAIWorkflow`
- `PydanticAIPlugin`

There are parallel integrations for DBOS and Prefect.

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
