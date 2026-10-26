# Agents Core

Read this file when the user needs the core `Agent` workflow: creating agents, choosing output types, using dependencies, defining specs, selecting models, or choosing how to run/stream an agent.

## Create a Basic Agent

Include Logfire instrumentation in new applications from the start, so every run is visible; see [Debug and Validate Agent Behavior](./TESTING-AND-DEBUGGING.md#debug-and-validate-agent-behavior) for credentials and other OpenTelemetry backends.

```python
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'anthropic:claude-fable-5-1',
    name='hello_world_agent',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

Pass an explicit `name=` to each agent: it labels the agent's run span in Logfire. When omitted, the name is inferred from the variable the agent is assigned to and falls back to `'agent'` when it can't be (e.g. agents kept in a list or dict), which matters once more than one agent runs in the same app.

## Structured Output with Pydantic Models

Use `output_type=MyModel` when the model should return validated structured data.

```python
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google:gemini-3-flash-preview', name='city_location_agent', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
```

If the user is choosing between output modes:

- `output_type=str` for plain text
- `output_type=MyModel` for structured output
- `TextOutput` for custom text parsing
- `NativeOutput` or `ToolOutput` when they need explicit output-mode control

## Picking One of a Run-Time Set

Use `Choices({key: description})` when the model has to pick one of a set that only exists once the run is under
way — the records a search returned, the actions available on a screen. Each option carries its meaning into the
schema, the output is validated against the keys, and it is a type, so the same value also works as a model field
or a tool parameter — there, write `Annotated[str, Intent]` so type checkers see a `str`. For a set you know when you write the code, use a `Literal` or an `Enum` (with
`UseEnumMemberDocstrings` for per-member descriptions), which give exhaustiveness checking.

```python
from pydantic_ai import Agent, Choices

agent = Agent('openai:gpt-5.2', name='triage_agent')

result = agent.run_sync(
    'The blender arrived smashed. Just send me another one.',
    output_type=Choices(
        {'refund': 'They want their money back.', 'replace': 'They want a working unit.'},
        description='What the customer is asking for.',
    ),
)
print(result.output)
```

`Choice(description, value=...)` makes an option stand for something other than its key. When that value is
callable, picking it *calls* it — sync or async, with no arguments — so the run's output is what the action
returned, the way an output function's is, and `ModelRetry` from it sends the model back for another pick. A set
with a callable value is an `output_type` only; as a field or a parameter it raises `UserError`.

## Validate Output Against Your Domain

The output type's Pydantic validation checks shape. Checks that need the run's deps or I/O — an id the model returned
must be one the app gave it, a referenced record must exist — go in an `@agent.output_validator`. Raising `ModelRetry`
there sends the message back to the model, which corrects its output within the same run. **Domain checks belong in
the validator, not after `run()`**: code that rejects `result.output` once the run has returned has lost the retry and
can only start over.

```python
from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass
class TriageDeps:
    open_ticket_ids: set[str]


class Triage(BaseModel):
    ticket_id: str
    summary: str


agent = Agent('openai:gpt-5.2', name='triage_agent', deps_type=TriageDeps, output_type=Triage)


@agent.output_validator
def check_ticket_id(ctx: RunContext[TriageDeps], output: Triage) -> Triage:
    if output.ticket_id not in ctx.deps.open_ticket_ids:
        raise ModelRetry(f'{output.ticket_id!r} is not an open ticket. Use one of {sorted(ctx.deps.open_ticket_ids)}.')
    return output
```

Validators can be async, so a lookup can use a database client from deps. Look up all the ids in one query, so the
number of queries doesn't grow with the size of the model's output:

```python
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext


class OrderStore(Protocol):
    async def existing_order_ids(self, order_ids: list[str]) -> set[str]: ...


@dataclass
class RefundDeps:
    orders: OrderStore


class Refund(BaseModel):
    order_id: str
    reason: str


agent = Agent('openai:gpt-5.2', name='refund_agent', deps_type=RefundDeps, output_type=list[Refund])


@agent.output_validator
async def check_orders_exist(ctx: RunContext[RefundDeps], output: list[Refund]) -> list[Refund]:
    if ctx.partial_output:  # streamed runs also validate partial output; check only the final one
        return output
    order_ids = [refund.order_id for refund in output]
    missing = set(order_ids) - await ctx.deps.orders.existing_order_ids(order_ids)
    if missing:
        raise ModelRetry(f'No orders with ids {sorted(missing)}. Refund only orders that exist.')
    return output
```

- Each `ModelRetry` uses one output retry; the default budget is 1, raised with `retries={'output': N}` on the agent
  or the run. When it runs out, the run raises `UnexpectedModelBehavior`.
- The validator may return a modified output; what it returns is `result.output`.
- When the valid ids are known as the run starts and the model must pick one, `Choices` (above) puts them in the
  schema itself; a validator also covers ids nested in larger outputs and checks that need I/O.

## Dependency Injection

Use `deps_type=...` plus `RunContext[...]` when tools or instructions need app state.

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5.2', name='greeting_agent', deps_type=str)


@agent.instructions
def add_user_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."
```

Use `@agent.tool` when the tool needs `RunContext`. Use `@agent.tool_plain` when it does not.

For instructions built from operator-controlled state that changes mid-conversation (todo lists, memory, per-run context), use `on_change='append'` to preserve the instruction prefix and the cache behind it; the cost is the block's complete new text added to history per change.
Use `@agent.instructions(name='state', on_change='append')` on the agent.
The callback still runs before each request; changed values become full replacements in history, sent as a diff when that's shorter.
Return `None` or an empty string to withdraw the block.
Use the same decorator on an always-on capability with an explicit `id`.
Toolsets can return `InstructionPart(content=..., name='state', on_change='append')` from `get_instructions()` under their own `id`, or use `@toolset.instructions(name='state', on_change='append')` on a `FunctionToolset` with an `id`.
Any instruction function (passed to `Agent(instructions=...)`, returned from a capability's `get_instructions()`, ...) can return `InstructionPart(..., name='state', on_change='append')` to opt in from inside the function.
Per-run context passed to `agent.run(..., instructions=...)` appends when it's a named part: `instructions=InstructionPart(..., name='page', on_change='append')` is keyed `run:page`; unnamed runtime instructions always rewrite.
A custom capability's `get_instructions()` can also return such an `InstructionPart` directly, for text that changes between runs when `for_run` returns a replacement carrying the new text (otherwise the construction-time result is reused).
Literal text in `Agent(instructions=...)`, `Capability(instructions=...)` and `FunctionToolset(instructions=...)` opts in with `InstructionPart('...', name='config', on_change='append')`.
Template strings can't opt in; use an instruction function that returns an `InstructionPart` instead.
Build instruction text from operator-controlled state, never from tool arguments, tool output, user input or retrieved content, even in append mode; send untrusted values in user messages or tool results.
Keep serialized message history in trusted server storage, including its `InstructionBaselinePart`s and `InstructionDeltaPart`s.
Compaction or loss of the baseline starts a new prefix from current instructions.
Use a durable operation for instruction callbacks that read external state during durable execution.
Deferred capabilities and realtime sessions do not support this update policy.
Leave the default `on_change='rewrite'` when prefix preservation is unnecessary.
The deferred capability catalog and Harness `SubAgents` and `RepoContext` instructions already append their changes.
So do MCP server instructions from `MCPToolset(..., include_instructions=True)`, keyed `toolset:<id>` (a URL toolset's `id` defaults to its hostname; durable execution needs an explicit one); stdio and in-process servers need an `id=` to append. Only enable `include_instructions` for servers you trust.

## Define Agents Declaratively with Specs

Use YAML or JSON specs when configuration should live outside Python code.

```yaml
model: anthropic:claude-opus-4-6
instructions: "You are helping {{user_name}} with research."
capabilities:
  - WebSearch
  - Thinking:
      effort: high
```

```python
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class UserContext:
    user_name: str


agent = Agent.from_file('agent.yaml', deps_type=UserContext)
result = agent.run_sync('Find recent papers on AI safety', deps=UserContext(user_name='Alice'))
```

Template strings are part of the spec flow, so route template-string questions here too.

## Choose or Configure Models

Model strings use the `"provider:model-name"` format.

Examples:

- `openai:gpt-5.2`
- `anthropic:claude-sonnet-4-6`
- `google:gemini-3-pro-preview`

Through the [Pydantic AI Gateway](https://pydantic.dev/docs/ai/overview/gateway/), one `PYDANTIC_AI_GATEWAY_API_KEY` reaches models from several providers, with spending limits and cost monitoring in Logfire. Prefix the model string with `gateway/`, using the provider's API format:

- `gateway/anthropic:claude-fable-5-1`
- `gateway/openai:gpt-6-sol`

Use a model instance instead of a string when the user needs provider-specific constructor arguments.

## Run Methods and Streaming

Pick a run method based on the interaction pattern:

- `run()` for async runs that complete normally
- `run_sync()` for synchronous scripts and notebooks
- `run_stream()` for streaming final output
- `run_stream_sync()` for sync streaming
- `run_stream_events()` when the caller needs the typed event stream directly
- `iter()` when the caller needs step-by-step control over the agent loop

Use `event_stream_handler=` with `run()` or `run_stream()` when the user wants progress updates without manually consuming the event stream. The stream includes model deltas, tool call/result events, and framework events such as `EnqueuedMessagesEvent` when queued messages enter run history.

Realtime sessions do not use `event_stream_handler`; iterate the session to consume realtime-only `RealtimeEvent` members.

```python
from collections.abc import AsyncIterable

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, RunContext

agent = Agent('openai:gpt-5.2', name='streaming_agent')


async def stream_handler(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]):
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            print(f'Calling {event.part.tool_name}...')


async def main():
    await agent.run('Do the task', event_stream_handler=stream_handler)
```

Deferred tool calls also surface as batch-level events: `DeferredToolRequestsEvent` (once per batch of deferred calls, before any `HandleDeferredToolCalls` handler runs) and `DeferredToolResultsEvent` (when a handler resolves requests inline). Use these to tell a frontend the run is paused waiting for approvals or external calls.

To surface progress or intermediate results from an async tool into the same event stream without polluting the model's context, define a dataclass subclass of `CustomEvent` (its fields are the payload; the event name derives from the class name) and await `ctx.emit(event)`. Sync tools cannot emit events. It reaches the `event_stream_handler`, `run_stream_events()`, `iter()` streaming, the AG-UI/Vercel AI adapters, and a realtime session's event stream (where events emitted from `before_run` lead the stream, and those emitted from `after_run`, after the stream has ended, aren't delivered, as in `iter()`); when emitted from a tool, its `tool_call_id` and `tool_name` are auto-stamped, and consumers use `isinstance()` against the class. Code driving `agent.iter()` can inject events by awaiting `AgentRun.emit()`. The payload can't reuse the envelope's own field names: `data`, `tool_call_id`, `tool_name`, and `event_kind` are rejected at class definition.

`CustomEvent` is for application-owned code only. Code that lives inside a capability must define namespaced `CapabilityEvent` subclasses instead; emitting either family from the other's side raises `UserError`. See CAPABILITIES-AND-HOOKS.md.

Custom events reach the AG-UI and Vercel AI frontends by default. For an event that should stay server-side (metrics, audit logs), declare the class `ui=False` — `class IndexProgressEvent(CustomEvent, ui=False)` — and every UI adapter skips it while in-process consumers still receive it. Declaring a `ui` field or `ClassVar` on an event class is rejected, since it would shadow that flag. The flag is class-level, not on the wire, so adapters also skip an `UnknownCustomEvent` (a class this process never imported): when events reach the frontend from another process, import their defining modules there or none of them are forwarded.

```python
from dataclasses import dataclass

from pydantic_ai import Agent, CustomEvent, RunContext

agent = Agent('openai:gpt-5.2', name='progress_agent')


@dataclass(kw_only=True)
class ProgressEvent(CustomEvent):
    done: int
    total: int


@agent.tool
async def process(ctx: RunContext, count: int) -> str:
    for i in range(count):
        await ctx.emit(ProgressEvent(done=i + 1, total=count))
    return 'done'
```

## Handle Provider Failures

Use `FallbackModel` when the user wants automatic provider or model failover.

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

fallback = FallbackModel(
    OpenAIChatModel('gpt-5.2'),
    AnthropicModel('claude-sonnet-4-6'),
)

agent = Agent(fallback, name='fallback_agent')
```

Good defaults:

- primary expensive/strong model, cheaper fallback for resilience
- same prompt/output contract across both models
- per-model settings only when the user actually needs them
- `model_settings={'max_retries': 0}` on the agent, or on every candidate but the last, so the provider SDK's own
  retries (2 by default on OpenAI, Anthropic, Groq and Cohere) don't hold up the fallback; Bedrock and xAI ignore it
  with an `UnsupportedSettingWarning`, so with one of those in the chain set it on the other candidates' `settings`

The answering response's `failed_attempts` lists the attempts the fallback moved on from (model, provider,
`'error'` or `'rejected'` outcome, error, start time, duration, usage); when every model fails they're on
`FallbackExceptionGroup.attempts`. A rejected response's tokens and cost count in `RunUsage` and towards token and
cost limits, but not in the answering response's `usage`. Neither they nor failed attempts count towards
`UsageLimits.request_limit`, which bounds the model responses the agent acts on, not requests sent to the provider.

To react to a specific kind of failure regardless of provider, catch an error category rather than inspecting status codes or provider exceptions: `ModelRateLimitError` (with `retry_after`), `ModelQuotaExceededError` (exhausted quota or billing, only when the provider says so; waiting won't help), `ModelUnavailableError` and its `ModelOverloadedError` (only when the provider says overloaded, e.g. HTTP 529), `ModelServerError` (other 5xx), `ModelConnectionError` (and its `ModelTimeoutError`), or `ModelRequestRejectedError` (a 400/422 or other rejection that resending unchanged won't fix) and its `ModelAuthenticationError` (401/403, bad or unauthorized credentials), `ModelNotFoundError` (404, unknown model or resource) and `ModelContextWindowExceededError`. A specific category wins over the generic rejection: a 400 context overflow is a `ModelContextWindowExceededError`, and Anthropic's out-of-credits 400 is a `ModelQuotaExceededError`, not a rejection. A retry predicate can skip permanent failures with `isinstance(e, ModelRequestRejectedError | ModelQuotaExceededError)`. All are `ModelAPIError`s, so `FallbackModel` falls back on them by default; one that came with an HTTP status is also a `ModelHTTPError`, and a custom model can raise such a combined error with `ModelHTTPError.for_category(category, status_code=..., model_name=...)`. An error inside an already open stream gets the status the same error has before the stream opens, with `in_stream=True`, so errors don't depend on whether the request was streamed. Provider details are on `provider_error_code`, `provider_error_type`, and `body`; `retry_after` is on every `ModelAPIError`, `ModelHTTPError.provider_retry_hint` reads `x-should-retry`, and `ModelConnectionError.phase` (`'pool'`, `'connect'`, `'write'`, `'read'`) says whether the request may have reached the provider. Embedding models (`Embedder`) raise the same errors.

When a completed model response omits usage, configured token or cost limits emit `UsageLimitUnavailableWarning` and return the result with known totals as lower bounds. Promote the warning to an error when the application must abort if a limit cannot be fully checked:

```python
import warnings

from pydantic_ai import UsageLimitUnavailableWarning

warnings.filterwarnings('error', category=UsageLimitUnavailableWarning)
```


To bound how long one model can hold up the fallback, set `model_settings={'request_timeout': seconds}`. Unlike
`timeout`, which the provider SDK applies to each attempt and re-arms on every retry, it is one deadline enforced by
Pydantic AI over all attempts at a request to one model, and over reading a streamed response through its last chunk
(consumer time included). Expiry raises `ModelRequestTimeout` (a `ModelAPIError`), which `FallbackModel` falls back on
by default; each model it tries gets a fresh deadline. It applies to any model, with or without a fallback. Under
Temporal, a plain model's deadline runs as a workflow timer that cancels the model activity; a `FallbackModel`'s
per-model deadlines run inside the activity instead.
