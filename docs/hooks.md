
# Hooks

Hooks let you intercept and modify agent behavior at every stage of a run — model requests, tool calls, streaming events — using simple decorators or constructor arguments. No subclassing needed.

The [`Hooks`][pydantic_ai.capabilities.Hooks] capability is the recommended way to add [lifecycle hooks](capabilities.md#hooking-into-the-lifecycle) for application-level concerns like logging, metrics, and lightweight validation. For reusable capabilities that combine hooks with tools, instructions, or model settings, subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] instead — see [Building custom capabilities](capabilities.md#building-custom-capabilities).

## Quick start

Create a [`Hooks`][pydantic_ai.capabilities.Hooks] instance, register hooks via `@hooks.on.*` decorators, and pass it to your agent:

```python {title="hooks_decorator.py"}
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages to the model')
    #> Sending 1 messages to the model
    return request_context


agent = Agent('test', capabilities=[hooks])
result = agent.run_sync('Hello!')
print(result.output)
#> success (no tool calls)
```

## Registering hooks

### Decorator registration

The `hooks.on` namespace provides decorator methods for every lifecycle hook. Use them as bare decorators or with parameters:

```python {test="skip" lint="skip"}
# Bare decorator
@hooks.on.before_model_request
async def my_hook(ctx, request_context):
    return request_context

# With parameters (timeout, tool filter)
@hooks.on.before_model_request(timeout=5.0)
async def my_timed_hook(ctx, request_context):
    return request_context
```

Multiple hooks can be registered for the same event — they fire in registration order.

### Constructor kwargs

You can also pass hook functions directly to the [`Hooks`][pydantic_ai.capabilities.Hooks] constructor:

```python {title="hooks_constructor.py"}
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks


async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages to the model')
    #> Sending 1 messages to the model
    return request_context


agent = Agent('test', capabilities=[Hooks(before_model_request=log_request)])
result = agent.run_sync('Hello!')
print(result.output)
#> success (no tool calls)
```

Both sync and async hook functions are accepted. Sync functions are automatically wrapped for async execution.

## Lifecycle overview

The following sequence diagram shows all hooks firing during a complete run with one tool call (happy path). Error hooks (`on_*_error`) are mutually exclusive with `after_*` — see [error hooks](capabilities.md#error-hooks) for that flow.

??? note "Expand sequence diagram"

    ```mermaid
    sequenceDiagram
        participant App as Application
        participant R as Run Hooks
        participant N as Node Hooks
        participant M as Model Hooks
        participant P as prepare_tools
        participant TV as Tool Validate Hooks
        participant TEx as Tool Execute Hooks
        participant LLM as LLM Provider
        participant Fn as Tool Function

        App->>R: before_run(ctx)
        activate R
        Note right of R: wrap_run enters

        Note over R,Fn: ── UserPromptNode ──
        R->>N: before_node_run(ctx, node)
        activate N
        Note right of N: wrap_node_run enters
        Note over N: Build user prompt message
        Note right of N: wrap_node_run exits
        N-->>R: after_node_run → next: ModelRequestNode
        deactivate N

        Note over R,Fn: ── ModelRequestNode ──
        R->>N: before_node_run(ctx, node)
        activate N
        Note right of N: wrap_node_run enters
        N->>P: prepare_tools(ctx, tool_defs)
        P-->>N: filtered tool_defs
        N->>M: before_model_request(ctx, request_context)
        activate M
        Note right of M: wrap_model_request enters
        M->>LLM: HTTP request
        LLM-->>M: Response (with tool_calls)
        Note right of M: wrap_model_request exits
        M-->>N: after_model_request(ctx, request_context, response)
        deactivate M
        Note right of N: wrap_node_run exits
        N-->>R: after_node_run → next: CallToolsNode
        deactivate N

        Note over R,Fn: ── CallToolsNode ──
        R->>N: before_node_run(ctx, node)
        activate N
        Note right of N: wrap_node_run enters

        Note over TV,TEx: For each tool call
        N->>TV: before_tool_validate(ctx, call, tool_def, raw_args)
        activate TV
        Note right of TV: wrap_tool_validate enters
        Note over TV: Parse & validate args against schema
        Note right of TV: wrap_tool_validate exits
        TV-->>N: after_tool_validate(ctx, call, tool_def, validated_args)
        deactivate TV

        N->>TEx: before_tool_execute(ctx, call, tool_def, args)
        activate TEx
        Note right of TEx: wrap_tool_execute enters
        TEx->>Fn: call tool function
        Fn-->>TEx: result
        Note right of TEx: wrap_tool_execute exits
        TEx-->>N: after_tool_execute(ctx, call, tool_def, args, result)
        deactivate TEx

        Note right of N: wrap_node_run exits
        N-->>R: after_node_run → next: ModelRequestNode
        deactivate N

        Note over R,Fn: ── ModelRequestNode (with tool results) ──
        R->>N: before_node_run(ctx, node)
        activate N
        Note right of N: wrap_node_run enters
        N->>P: prepare_tools(ctx, tool_defs)
        P-->>N: filtered tool_defs
        N->>M: before_model_request(ctx, request_context)
        activate M
        Note right of M: wrap_model_request enters
        M->>LLM: HTTP request
        LLM-->>M: Response (text only, no tool calls)
        Note right of M: wrap_model_request exits
        M-->>N: after_model_request(ctx, request_context, response)
        deactivate M
        Note right of N: wrap_node_run exits
        N-->>R: after_node_run → next: CallToolsNode
        deactivate N

        Note over R,Fn: ── CallToolsNode (no tool calls) ──
        R->>N: before_node_run(ctx, node)
        activate N
        Note right of N: wrap_node_run enters
        Note over N: No tool calls → End(FinalResult)
        Note right of N: wrap_node_run exits
        N-->>R: after_node_run → End
        deactivate N

        Note right of R: wrap_run exits
        R-->>App: after_run(ctx, result)
        deactivate R
        Note over App: AgentRunResult
    ```

## Hook types

### Run hooks

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `before_run` | `before_run=` | `before_run` |
| `after_run` | `after_run=` | `after_run` |
| `run` | `run=` | `wrap_run` |
| `run_error` | `run_error=` | `on_run_error` |

Run hooks fire once per agent run. `wrap_run` (registered via `hooks.on.run`) wraps the entire run and supports error recovery.

### Node hooks

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `before_node_run` | `before_node_run=` | `before_node_run` |
| `after_node_run` | `after_node_run=` | `after_node_run` |
| `node_run` | `node_run=` | `wrap_node_run` |
| `node_run_error` | `node_run_error=` | `on_node_run_error` |

Node hooks fire for each graph step ([`UserPromptNode`][pydantic_ai.agent.UserPromptNode], [`ModelRequestNode`][pydantic_ai.agent.ModelRequestNode], [`CallToolsNode`][pydantic_ai.agent.CallToolsNode]).

!!! note
    `wrap_node_run` hooks are called automatically by [`agent.run()`][pydantic_ai.agent.AbstractAgent.run], [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], and [`agent_run.next()`][pydantic_ai.run.AgentRun.next], but **not** when iterating with bare `async for node in agent_run:`.

### Model request hooks

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `before_model_request` | `before_model_request=` | `before_model_request` |
| `after_model_request` | `after_model_request=` | `after_model_request` |
| `model_request` | `model_request=` | `wrap_model_request` |
| `model_request_error` | `model_request_error=` | `on_model_request_error` |

Model request hooks fire around each LLM call. [`ModelRequestContext`][pydantic_ai.models.ModelRequestContext] bundles `model`, `messages`, `model_settings`, and `model_request_parameters`. To swap the model for a given request, set `request_context.model` to a different [`Model`][pydantic_ai.models.Model] instance.

To skip the model call entirely, raise [`SkipModelRequest(response)`][pydantic_ai.exceptions.SkipModelRequest] from `before_model_request` or `model_request` (wrap).

### Tool validation hooks

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `before_tool_validate` | `before_tool_validate=` | `before_tool_validate` |
| `after_tool_validate` | `after_tool_validate=` | `after_tool_validate` |
| `tool_validate` | `tool_validate=` | `wrap_tool_validate` |
| `tool_validate_error` | `tool_validate_error=` | `on_tool_validate_error` |

Validation hooks fire when the model's JSON arguments are parsed and validated. All tool hooks receive `call` ([`ToolCallPart`][pydantic_ai.messages.ToolCallPart]) and `tool_def` ([`ToolDefinition`][pydantic_ai.tools.ToolDefinition]) parameters.

!!! note
    Tool validation and execution hooks only fire for function tools. Internal output tools (used to deliver structured output) are not user-facing and are skipped.

To skip validation, raise [`SkipToolValidation(args)`][pydantic_ai.exceptions.SkipToolValidation] from `before_tool_validate` or `tool_validate` (wrap).

### Tool execution hooks

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `before_tool_execute` | `before_tool_execute=` | `before_tool_execute` |
| `after_tool_execute` | `after_tool_execute=` | `after_tool_execute` |
| `tool_execute` | `tool_execute=` | `wrap_tool_execute` |
| `tool_execute_error` | `tool_execute_error=` | `on_tool_execute_error` |

Execution hooks fire when the tool function runs. `args` is always the validated `dict[str, Any]`.

To skip execution, raise [`SkipToolExecution(result)`][pydantic_ai.exceptions.SkipToolExecution] from `before_tool_execute` or `tool_execute` (wrap).

### Tool preparation

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `prepare_tools` | `prepare_tools=` | `prepare_tools` |

Filters or modifies tool definitions the model sees on each step. Controls visibility, not execution.

### Event stream hooks

| `hooks.on.` | Constructor kwarg | `AbstractCapability` method |
|---|---|---|
| `run_event_stream` | `run_event_stream=` | `wrap_run_event_stream` |
| `event` | `event=` | _(per-event convenience)_ |

`run_event_stream` wraps the full event stream as an async generator. `event` is a convenience — it fires for each individual event during a streamed run:

```python {title="hooks_event.py"}
from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()
event_count = 0


@hooks.on.event
async def count_events(ctx: RunContext[None], event: AgentStreamEvent) -> AgentStreamEvent:
    global event_count
    event_count += 1
    return event


agent = Agent('test', capabilities=[hooks])
```

## Tool hook filtering

Tool hooks (validation and execution) support a `tools` parameter to target specific tools by name:

```python {title="hooks_tool_filter.py"}
from typing import Any

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart

hooks = Hooks()
call_log: list[str] = []


@hooks.on.before_tool_execute(tools=['send_email'])
async def audit_dangerous_tools(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    call_log.append(f'audit: {call.tool_name}')
    return args


agent = Agent('test', capabilities=[hooks])


@agent.tool_plain
def send_email(to: str) -> str:
    return f'sent to {to}'


result = agent.run_sync('Send an email to test@example.com')
print(call_log)
#> ['audit: send_email']
```

The `tools` parameter accepts a sequence of tool names. The hook only fires for matching tools — other tool calls pass through unaffected.

## Timeouts

Each hook supports an optional `timeout` in seconds. If the hook exceeds the timeout, a [`HookTimeoutError`][pydantic_ai.capabilities.HookTimeoutError] is raised:

```python {title="hooks_timeout.py"}
import asyncio

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks, HookTimeoutError

hooks = Hooks()


@hooks.on.before_model_request(timeout=0.01)
async def slow_hook(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    await asyncio.sleep(10)  # Will be interrupted by timeout
    return request_context  # pragma: no cover


agent = Agent('test', capabilities=[hooks])
try:
    agent.run_sync('Hello')
except HookTimeoutError as e:
    print(f'Hook timed out: {e.hook_name} after {e.timeout}s')
    #> Hook timed out: before_model_request after 0.01s
```

Timeouts are set via the decorator parameter (`@hooks.on.before_model_request(timeout=5.0)`) or via the constructor when using kwargs.

## Wrap hooks

Wrap hooks let you surround an operation with setup/teardown logic. In the `hooks.on` namespace, wrap hooks drop the `wrap_` prefix — `hooks.on.model_request` corresponds to `wrap_model_request`:

```python {title="hooks_wrap.py"}
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse

hooks = Hooks()
wrap_log: list[str] = []


@hooks.on.model_request
async def log_request(
    ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
) -> ModelResponse:
    wrap_log.append('before')
    response = await handler(request_context)
    wrap_log.append('after')
    return response


agent = Agent('test', capabilities=[hooks])
result = agent.run_sync('Hello!')
print(wrap_log)
#> ['before', 'after']
```

## Hook ordering

Within a single [`Hooks`][pydantic_ai.capabilities.Hooks] instance, `before_*`, `after_*`, and `on_*_error` fire in **registration order** (the order they were defined or passed to the constructor). `wrap_*` nests as middleware, with the first-registered wrapper as the outermost layer.

Across multiple capabilities, the [composition rules](capabilities.md#composition) apply: `before_*` fires in capability order, `after_*` fires in reverse capability order, and `wrap_*` nests as middleware.

## Error hooks

Error hooks (`*_error` in the `hooks.on` namespace, `on_*_error` on `AbstractCapability`) use **raise-to-propagate, return-to-recover** semantics:

- **Raise the original error** — propagates unchanged *(default)*
- **Raise a different exception** — transforms the error
- **Return a result** — suppresses the error

See [Error hooks](capabilities.md#error-hooks) for the full pattern and recovery types.

## Triggering retries with `ModelRetry`

Hooks can raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to try again with a custom message — the same exception used in [tool functions](tools.md#model-retry) and output validators.

**Model request hooks** (`after_model_request`, `wrap_model_request`, `on_model_request_error`):

- The retry message is sent back to the model as a [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart]
- `after_model_request`: the original response is preserved in message history so the model can see what it said
- `wrap_model_request`: the response is preserved only if the handler was called
- Retries count against the agent's `output_retries` limit

**Tool hooks** (`before/after_tool_validate`, `before/after_tool_execute`, `wrap_tool_execute`, `on_tool_execute_error`):

- Converted to tool retry prompts, same as when a tool function raises `ModelRetry`
- Retries count against the tool's `max_retries` limit

`ModelRetry` from `wrap_model_request` and `wrap_tool_execute` is treated as control flow — it bypasses `on_model_request_error` and `on_tool_execute_error` respectively.

```python {title="hooks_model_retry.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()


@hooks.on.after_model_request
async def check_response(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    if 'PLACEHOLDER' in str(response.parts):
        raise ModelRetry('Response contains placeholder text. Please provide real data.')
    return response


agent = Agent('test', capabilities=[hooks])
result = agent.run_sync('Hello')
print(result.output)
#> success (no tool calls)
```

## When to use `Hooks` vs `AbstractCapability`

| Use [`Hooks`][pydantic_ai.capabilities.Hooks] | Use [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] |
|---|---|
| Application-level hooks (logging, metrics) | Reusable, packaged capabilities |
| Quick one-off interceptors | Combined tools + hooks + instructions + settings |
| No configuration state needed | Complex per-run state management |
| Single-file scripts | Multi-agent shared behavior |
