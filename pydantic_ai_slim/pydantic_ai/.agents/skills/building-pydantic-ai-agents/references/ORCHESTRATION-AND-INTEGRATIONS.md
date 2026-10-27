# Orchestration and Integrations

Read this file when the user wants multi-agent coordination, graphs, direct model calls, A2A, durable execution, embeddings, image generation, evals, or third-party integrations.

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

For runs long enough to approach Temporal's history limit, pass `TemporalDurability(continue_as_new=...)`. Before each model request where Temporal suggests continuing as new, it calls your function with the `RunContext`. The function calls `workflow.continue_as_new()` with everything the workflow's run method needs to resume, typically `None` as the prompt plus `ctx.messages`, `ctx.usage`, and `ctx.usage_limits`, or returns to defer.

To delegate to a nested durable agent under Temporal, run the delegating async function tool as a child workflow with `metadata={'temporal': {'child_workflow': ChildWorkflowConfig(...)}}` (or the same metadata through `SetToolMetadata`). The nested agent's model requests and tool calls then become activities in the child's own history instead of the parent's. List the nested agent in `__pydantic_ai_agents__` too, and register the worker through `PydanticAIPlugin` or `AgentPlugin`: `TemporalDurability.temporal_activities` raises when a tool uses a child workflow, because it would leave the child workflow definition unregistered. The metadata must be in place when the worker is configured; tools added with `@agent.tool` count. An exception in the tool body fails the child workflow, like a failed activity. Child workflows aren't supported for sync, dynamic, or MCP tools.

`TemporalAgent`, `DBOSAgent`, and `PrefectAgent` are deprecated wrapper agents.

For latency-tolerant work at batch prices, add the `BatchModelRequests` capability from `pydantic_ai.capabilities`
(Anthropic, OpenAI Responses and Gemini API models have batch APIs; not through Pydantic AI Gateway) and run the
agent as usual; runs started together in one process share batches, and so do agents and runs given the same
instance. Runs awaited one after another each wait `max_wait` (default 30 seconds) alone and get a batch each, so
start them together, for example with `asyncio.gather`. Don't write a submission or polling loop: a waiting request
suspends the turn through the agent graph's continuation loop, so it works in a plain `agent.run(...)`, and a run
resumed from its message history picks up the same batch. Streamed runs and event stream handlers work, getting each
response's events all at once when its batch result arrives, and each tool turn or output retry is another batch
round trip. For a job that submits and exits, add `PendingModelResponse` to `output_type`: the run ends with it once
the request is submitted, and running the agent again with `result.all_messages()` checks the batch once and
continues or ends pending again; the collecting agent needs `BatchModelRequests` too, or resuming raises
`UserError`, and a model chosen dynamically (e.g. by `SelectModel`) must be passed as `run(model=...)`. A
submit-only script can lower `max_wait` so it doesn't wait for more requests to join the batch. The same works for
OpenAI background mode.

Under durable execution, put the capability on the agent. With Temporal, DBOS or Prefect, submitting and each check
are durable steps and a recovered run continues with the same batch; the waits between checks are durable on
Temporal and DBOS, and plain sleeps on Prefect. The Harness engines (AWS Lambda, Absurd, pgtask, Render Workflows)
sleep in-process between checks, so the process must stay up for the whole batch. Restate and Airflow don't record
batched requests, so recovery submits them again, in a new batch that's billed too. Where an engine can't wait out a long
batch, use `PendingModelResponse` and schedule the next run from the stored history yourself. See
https://pydantic.dev/docs/ai/core-concepts/batch/#durable-execution for each engine's details.

Don't write resubmission logic either: requests a batch didn't run (expired or missing, not cancelled) or that failed
with a rate limit or server error are submitted again automatically, up to `max_resubmissions` (default 2), a check
that fails transiently is retried at the next poll, and a history whose batch the provider no longer has raises
`SuspendedResponseExpired`.

Capability hooks that must behave differently in deterministic workflow code check `ctx.in_durable_context`: `True` inside the durable workflow or flow, `False` in Temporal activities and DBOS steps (where tools run) and outside durable execution. Prefect tasks inherit their flow's context, so it is `True` there. Don't detect the engine by module or class name.

Pass every executing toolset that needs durable wrapping to the agent constructor. In particular, construct a `DynamicToolset` with an explicit `id` and pass it to `Agent(toolsets=[...])`; the `@agent.toolset` decorator registers after the engine's durable units were created. Toolsets that arrive later — via the decorator, `run(toolsets=...)`, `override(toolsets=...)`, or a per-run capability — are never wrapped for durable execution. Inside a workflow or flow, Temporal and Prefect reject runtime `MCPToolset` and `DynamicToolset` leaves, plus `FunctionToolset` leaves unless every async tool opts out of durable wrapping with `metadata={'temporal': False}` or `metadata={'prefect': False}` respectively; DBOS accepts a `FunctionToolset`, whose tools it runs inline either way, but rejects `MCPToolset` and `DynamicToolset`. A custom executing `AbstractToolset` leaf is not recognized by this guard and runs unwrapped, so do not add one at run time. The deprecated wrapper agents don't run this check — inside a workflow or flow they run the toolset list frozen at wrap time, so a toolset registered that late is silently left out. A toolset added at run time also cannot reuse a construction-time toolset's `id`.

Temporal and DBOS register durable units before their workers start, so attach capabilities at agent construction time. Passing `run(capabilities=[...])` inside one of their workflows raises a `UserError` unless the capability is the observer-only `Instrumentation`; broader support for observer-only capabilities is tracked in [#5477](https://github.com/pydantic/pydantic-ai/issues/5477), where users can share their use cases. Prefect creates tasks per call, so it has no such registration boundary and accepts a per-run capability that contributes no executing toolset; one that does is still rejected by the runtime-toolset guard.

A run-time `model=` inside a workflow must be a model-name string or an instance registered in the durability capability's `models=`. An unregistered `Model` instance raises a `UserError`: it can't be serialized into the activity/step/task, and rebuilding it from its `model_id` would build a different model. To build a specific instance inside the durable unit (e.g. per-user credentials from `deps`), pass a string and use a `ResolveModelId` capability.

## Handle MCP Tool Errors

Set `MCPToolset(tool_error_behavior=...)` according to the server error semantics:

- `'retry'` asks the model to correct and retry the call. This is the default.
- `'failed'` reports a completed failed result via `ToolFailed` without consuming retry budget.
- `'error'` propagates the underlying exception to application code.

Structured server error content is serialized as JSON for both `'retry'` and `'failed'`. Protocol and transport errors (as opposed to completed tool errors) stay retryable even under `'failed'`.

## Render MCP Apps in a Frontend

[MCP Apps](https://modelcontextprotocol.io/extensions/apps/overview) are HTML pages a server attaches to tools. `MCPToolset` needs no setup: a call to a tool with `_meta.ui.resourceUri` keeps an `MCPAppToolResult` in `ToolReturnPart.metadata['mcp_app']` (resource URI, tool input, full result, toolset `id`), which the model never sees. The AG-UI adapter sends it as an `mcp-apps` activity, the Vercel AI adapter as a `data-mcp-apps` chunk. Read the page with `read_resource_for_app(uri)`, which returns the MCP result as is, keeping the MIME type and `_meta.ui` (CSP, permissions).

Relay the page's requests with `call_tool_for_app(name, arguments)` (app-visible tools only, `isError` results returned, no `process_tool_call`) and `read_resource_for_app(uri)` (any resource); both return the MCP wire result to send back as is. Use one `MCPToolset` per server.

## Use Embeddings for RAG

Use `Embedder(...)` for query/document embeddings when the user is building retrieval or semantic search.

```python
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')
```

## Generate Images

Use `ImageGenerator(...)` when the application, rather than an agent, decides that an image should be created or edited.

```python
from pydantic_ai import ImageGenerator

generator = ImageGenerator('openai:gpt-image-2')
result = generator.generate_sync('A watercolor map of a floating city.')
image_bytes = result.image.data
```

Use `await generator.generate(...)` from async code. `result.image` is the first generated image as a `BinaryImage`; use `result.images` when you asked for several or need per-image metadata such as `revised_prompt`.

`ImageGenerationSettings` carries only the portable settings — `dimensions`, `aspect_ratio`, `extra_headers`, `extra_body`. Everything else, including image count, quality, output format, and background, is provider-prefixed on `OpenAIImageGenerationSettings`, `GoogleImageGenerationSettings`, or `XaiImageGenerationSettings`.

Pass reference images through `images=[...]` to edit or transform them. A provider content block raises `ContentFilterError` instead of returning an empty result, so a rejected prompt can be retried explicitly.

When the agent rather than the application should decide, use the `ImageGeneration` capability with `fallback_image_model='openai:gpt-image-2'` (or an `ImageGenerationModel`; an `ImageGenerator` goes on `local`), which calls the direct image model as a tool when the conversational model has no native image generation.

The image such a run generates is in its message history, not in `result.output`: a local fallback returns it from the `generate_image` tool call (`ToolReturnPart.files`), and the native tool puts it in that response's `ModelResponse.images`. `output_type=BinaryImage` requires the agent's model itself to support image output, and the output is then that model's own image, even beside `native=False`, not the fallback's.

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

For bulk model requests without an agent, use `pydantic_ai.direct.model_request_batch()`. To submit in one process and collect in another, use `model.batch_encode()`, `batch_create()`, `batch_status()`, `batch_results()`, `batch_cancel()` and `batch_decode()`, and store the returned `Batch` handle with the requests' settings and parameters.
