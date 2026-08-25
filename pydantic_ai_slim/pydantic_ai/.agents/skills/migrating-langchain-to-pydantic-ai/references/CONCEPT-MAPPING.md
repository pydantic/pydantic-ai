# LangChain and Pydantic AI Concept Mapping

Use this reference to select Pydantic AI primitives. Preserve behavior rather than matching class names.

## Contents

- [Core agent loop](#core-agent-loop)
- [Typed invariants at unstable seams](#typed-invariants-at-unstable-seams)
- [Tools and runtime context](#tools-and-runtime-context)
- [Structured output](#structured-output)
- [Middleware and lifecycle](#middleware-and-lifecycle)
- [State, memory, and persistence](#state-memory-and-persistence)
- [Graphs and multi-agent systems](#graphs-and-multi-agent-systems)
- [Streaming, testing, and observability](#streaming-testing-and-observability)
- [Transitional bridges](#transitional-bridges)
- [Migration traps](#migration-traps)

## Core agent loop

| LangChain / LangGraph | Pydantic AI default | Migration note |
|---|---|---|
| LCEL prompt/retriever/parser pipeline | plain Python around a model call or `Agent` | Keep deterministic retrieval, formatting, and routing explicit. Do not introduce an agent loop when the source performs one model call. |
| `create_agent(model, tools, system_prompt=...)` | `Agent(model, tools=..., instructions=...)` | Keep one reusable agent unless construction genuinely varies per request. |
| `agent.invoke({"messages": ...})` | `await agent.run(prompt, deps=..., message_history=...)` | Consume `result.output`; do not expose Pydantic AI message objects at the public boundary. |
| `ainvoke` | `await agent.run(...)` | Pydantic AI is async-first; use `run_sync` only at synchronous edges. |
| `RunnableConfig.configurable` | typed dependency object and explicit run arguments | Split application configuration from model-visible content. |
| `context_schema` / `ToolRuntime.context` | `deps_type` / `RunContext.deps` | Put DB clients, authenticated identity, config, and service gateways in dependencies. |
| prompt templates | `instructions` or `system_prompt` | Use `instructions` for current-agent policy and `system_prompt` only when prior prompts must survive history; multiple LangChain dynamic prompts may replace rather than compose. |
| `init_chat_model` | provider-prefixed model string or model instance | Preserve provider settings explicitly; verify the installed provider API. |

Preserve model transport as well as the model name. LangChain's `ChatOpenAI`, an OpenAI Responses model, an Azure deployment, and an OpenAI-compatible local endpoint can share a model label while using different request protocols and settings. Inspect the installed Pydantic AI provider constructors and test every configured branch without making a live request; leave unknown custom endpoints as an explicit integration gap.

Minimal translation:

```python {lint="skip"}
# ruff: noqa: F704, F821, Q000
# LangChain
from langchain.agents import create_agent

agent = create_agent(
    model="provider:model",
    tools=[lookup_order],
    system_prompt="Help authenticated customers with orders.",
)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Where is order 123?"}]},
    context=runtime_context,
)
```

```python {noqa="F704,F821"}
# ruff: noqa: F704, F821
# Pydantic AI
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    customer_id: str
    orders: 'OrderService'


agent = Agent(
    'provider:model',
    deps_type=Deps,
    instructions='Help authenticated customers with orders.',
)


@agent.tool
async def lookup_order(ctx: RunContext[Deps], order_id: str) -> str:
    return await ctx.deps.orders.lookup_for_customer(ctx.deps.customer_id, order_id)


result = await agent.run('Where is order 123?', deps=deps)
print(result.output)
```

The dependency boundary is a security boundary: the model chooses `order_id`, but cannot choose `customer_id` or the service client.

## Typed invariants at unstable seams

Use types to make an observed migration decision executable, not to remodel stable application code. The following seams caused real ambiguity in migrated applications:

| Observed source ambiguity | Target invariant | Evidence to require |
|---|---|---|
| free-form config or runtime context mixes trusted identity, services, and model inputs | a dependency dataclass plus `Agent[DepsT, OutputT]`; only model-chosen values appear in tool parameters | static checking plus a captured tool schema and model request |
| a string, dictionary, or graph control value can mean several terminal states, such as answer versus request-more-input | a Pydantic model or explicit output union for the states the source actually reaches | every variant validates and every application branch is exercised; remember that including `str` permits plain text to terminate the run |
| graph state mixes public messages, model protocol history, pending resume state, owner, and persistence version | distinct public DTOs, `list[ModelMessage]`, and a typed workflow record; serialize model history with `ModelMessagesTypeAdapter` when the installed version provides it | storage round-trip, continuation, ownership, and resume tests against the real backend |
| provider names and settings are passed through loose strings and dictionaries | retain the source enum or typed configuration and map it to a concrete Pydantic AI `Model` | construct every configured branch without a live request and probe provider-specific settings |

Do not claim semantic parity from static typing alone. Types can expose missing cases and validate boundaries; they cannot prove timing, retries, persistence, side effects, or framework lifecycle behavior.

## Tools and runtime context

| Source pattern | Target pattern |
|---|---|
| `@tool` plain function | `@agent.tool_plain` or `Tool(fn)` |
| tool with `ToolRuntime` | `@agent.tool` with first parameter `RunContext[Deps]` |
| `BaseTool` subclass | Prefer a normal typed function; use `Tool` or `Tool.from_schema` only when dynamic schema is necessary. |
| toolkit | `FunctionToolset`, another `AbstractToolset`, or a focused capability |
| MCP adapter | Pydantic AI MCP toolset or `MCP` capability |
| tools filtered by user/state | tool `prepare=...`, `PrepareTools`, a wrapper toolset, or capability loading |
| runtime-discovered large catalog | deferred tools/tool search; avoid rebuilding the schema every turn |
| tool retry middleware | `Hooks.on.tool_execute` or service-client retry for same-handler local retry; `ModelRetry`, tool `retries=...`, validators, and transport retry remain separate domains |
| approval middleware | `DeferredToolRequests` and `DeferredToolResults` |
| tool artifacts | `ToolReturn` with distinct model content, return value, and metadata |

Preserve tool name, description, JSON schema, concurrency, idempotency, timeout, retry, approval, auth, and error-to-model semantics. A successful wrapper import does not prove tool parity.

## Structured output

| LangChain | Pydantic AI |
|---|---|
| `response_format=Schema` | `output_type=Schema`, but select the output transport explicitly when parity matters |
| provider strategy | `NativeOutput(Schema)` when native enforcement is required |
| tool strategy | `ToolOutput(Schema)` when tool transport is required |
| manual parser / non-JSON text | `TextOutput(parser)` |
| dynamic JSON schema | `StructuredDict(schema, name=...)` |
| retry after invalid response | output validation and `ModelRetry` / configured retries |

Do not include `str` in a union when the run must end with structured output; plain text would be a valid terminal result.

LangChain may auto-select provider-native structured output for a bare schema. A bare Pydantic AI structured `output_type` follows the selected model profile's default, which may be tool, native, or prompted output. Use `NativeOutput`, `ToolOutput`, `PromptedOutput`, or `TextOutput` explicitly when wire behavior matters, and characterize the chosen provider/model.

## Middleware and lifecycle

Map each middleware by the behavior it owns:

| LangChain middleware behavior | Pydantic AI target |
|---|---|
| dynamic system prompt | dynamic `@agent.instructions` |
| before/after model or tool | `Hooks` lifecycle hook; wrapper decorators include `on.model_request` and `on.tool_execute` |
| reusable prompt + tools + hooks + settings | custom `AbstractCapability` |
| trim/summarize messages | `ProcessHistory` with an explicitly tested summary and pairing policy |
| filter/rename tool definitions | tool `prepare`, `PrepareTools`, or wrapper toolset |
| validate tool arguments | `args_validator` or tool-validation hook |
| dynamic model selection/fallback | model instance/wrapper such as `FallbackModel`, or select the model at the app boundary |
| model call limit | `UsageLimits` and explicit application limits |
| tool error conversion | catch the expected exception in the tool and raise `ModelRetry` only for model-correctable failures |
| logging/tracing | Logfire instrumentation or hooks for application metrics |
| guardrail | input/output validation hook; keep authorization inside tools/services |
| inject a mid-run user message | `RunContext.enqueue` or `AgentRun.enqueue` |

Reproduce hook ordering explicitly. Combining several source middleware objects into one opaque hook makes parity harder to inspect and test.

## State, memory, and persistence

Separate four concepts that LangGraph often stores together:

1. **Run dependencies:** immutable or service-like values used during one run -> `deps_type` and `RunContext.deps`.
2. **Conversation messages:** model request/response history -> persist serialized Pydantic AI messages and pass `message_history`.
3. **Workflow state:** plan, counters, fan-out results, approvals, domain progress -> typed application state or `pydantic_graph` state.
4. **Long-term memory:** cross-thread facts -> an explicit repository/service in dependencies.

| LangGraph feature | Migration choice |
|---|---|
| `MessagesState` | Pydantic AI message history plus separately typed workflow state |
| custom reducers | plain update functions or `pydantic_graph` joins/reducers |
| checkpointer/thread | application persistence or a durable execution integration |
| store | explicit storage service in typed dependencies |
| time travel/fork | durable workflow-specific implementation; do not infer this from message history |
| `interrupt()` for missing user input | application-owned pending conversational state and resume |
| `interrupt()` or HITL around a protected tool | deferred tool request plus authenticated, durable application correlation and resume |
| replay after failure | Temporal, DBOS, Prefect, Restate, or another explicit durable boundary |

Never label a migration complete because chat messages survive if the old system also promised checkpoint replay, pending writes, thread forks, or exactly-once side-effect protection.

Pydantic AI may repair dangling tool calls and orphaned tool results before the model request, while LangGraph `add_messages` merges by message ID. Compare the actual model-visible history when converting stored threads.

## Graphs and multi-agent systems

Choose by topology:

- Use one Pydantic AI `Agent` for a normal model/tool loop.
- Use plain async Python for a fixed sequence, bounded loop, or `asyncio.gather` fan-out.
- Use delegation via tools when a parent agent remains in control and consumes child output.
- Use programmatic hand-off when application code chooses the next specialist.
- Use `pydantic_graph` when explicit typed nodes, branches, joins, or inspectable workflow state add value.
- Use parent tools that call typed child agents for model-selected delegation; define child history, dependencies, usage, result, and failure propagation explicitly.
- Add a durable execution integration when the workflow must survive process failure; a graph alone is not durability.

Translate `Command(goto=..., update=...)` into an explicit next-node value plus a typed state update. Preserve fan-out limits, cancellation, exception aggregation, and ordering when translating LangGraph `Send` or parallel branches. For Pydantic Graph fan-out, return branch-local results, carry a source index through the join when order matters, and use `ReducerContext.cancel_sibling_tasks()` only when early completion is the intended reducer contract.

## Streaming, testing, and observability

| LangChain ecosystem | Pydantic ecosystem |
|---|---|
| `stream` / `astream` values, updates, messages | `run_stream`, `run_stream_events`, `event_stream_handler`, or `iter` |
| fake chat models | `TestModel` or `FunctionModel` under `agent.override(...)` |
| trajectory/eval datasets | `pydantic_evals` cases, datasets, and evaluators |
| LangSmith traces | Pydantic AI Logfire instrumentation and application-owned OpenTelemetry spans; LangSmith can export source traces to the same backend for comparison |
| graph state inspection | typed state plus application persistence/graph inspection |

Define an application-owned event schema at the UI/API boundary. Adapt both implementations to it during migration; do not make clients depend directly on either framework's event classes.

Use Logfire to inspect model calls, tools, retries, errors, usage, and timing, then prove public contracts with executable tests. In particular, model time to first chunk is not client time to first event. See [Logfire-Assisted Migration Verification](LOGFIRE-VERIFICATION.md).

Do not substitute `run_stream()` for `run()` without a separate trajectory test. `run_stream()` commits the first matching output as it streams; co-emitted tools and retries can therefore produce a different terminal result from a complete `run()`.

## Transitional bridges

Pydantic AI can wrap LangChain tools:

```python {noqa="F821"}
# ruff: noqa: F821
from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset, tool_from_langchain

single_tool = tool_from_langchain(existing_langchain_tool)
toolset = LangChainToolset(existing_toolkit.get_tools())

agent = Agent('provider:model', tools=[single_tool], toolsets=[toolset])
```

Use this only to create a vertical slice while tool internals are ported. The wrapper delegates argument validation to the LangChain tool, retains LangChain dependencies, and can conceal framework-specific callbacks or runtime assumptions. Add a removal issue and a parity test for every bridge.

Audit LangChain-only flags such as `return_direct`: the wrapper invokes the tool but does not automatically preserve the source agent's stop-after-tool routing. When the call is a model-selected terminal action, represent it as a named output function with `ToolOutput` and prove one execution plus the source model-call count.

Keep an existing retriever or LCEL pipeline behind a narrow tool/service interface if rewriting it would block agent migration. Port it later when the agent boundary is stable.

## Migration traps

- Translating `state_schema` to `deps_type` and then mutating dependencies as workflow state.
- Passing authenticated identity, tenant, or credentials as model-chosen tool parameters.
- Treating all middleware as hooks even when behavior belongs in a tool or model wrapper.
- Reusing LangChain message objects in Pydantic AI history.
- Moving deterministic branches into prompts to avoid learning `pydantic_graph`.
- Replacing checkpointers with an in-memory message list.
- Keeping both observability SDKs without defining trace ownership and correlation.
- Testing only final text while tool trajectory, approval, and side effects changed.

Primary references: [Pydantic AI agents](https://pydantic.dev/docs/ai/core-concepts/agent/), [tools](https://pydantic.dev/docs/ai/tools-toolsets/tools/), [hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/), [third-party tools](https://pydantic.dev/docs/ai/tools-toolsets/third-party-tools/), [multi-agent patterns](https://pydantic.dev/docs/ai/guides/multi-agent-applications/), and [durable execution](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/).
