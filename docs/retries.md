# Retries

"Retry" means five different things in an agent run, at five different layers, and they don't share budgets. Mixing them up is the usual cause of a run that retries far more (or far less) than expected. This page is the map; each layer links to the page that configures it in detail.

## The layers

| Layer | What it re-attempts | Configured with | What it adds to message history |
|---|---|---|---|
| [Transport](#transport-retries) | The same HTTP request to the provider | [`AsyncHTTPX2TenacityTransport`][pydantic_ai.retries.AsyncHTTPX2TenacityTransport] on your HTTP client | Nothing — the agent never sees the attempts |
| [Model fallback](#model-fallback-is-not-a-retry) | The same request against a *different* model | [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] | Only the winning response |
| [Tool](#tool-retries) | One tool call, by asking the model to correct it | `retries={'tools': N}` and per-tool limits | A [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] in place of the tool's result |
| [Output](#output-retries) | The model's final answer, by asking it to correct it | `retries={'output': N}` and [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries] | A `RetryPromptPart` — see [below](#output-retries) for where it lands |
| [Model-request hooks](hooks.md) | The model request, from `after_model_request`, `wrap_model_request`, or `on_model_request_error` raising `ModelRetry` | The hook itself; it draws on the **output** budget | A new request carrying a `RetryPromptPart` |

Only the last three are "agent retries" — they cost a model round trip each, because a retry *is* another request. The first two are invisible to the model.

## Transport retries

Transport retries live below the model client: a failed HTTP request is re-sent without the agent ever knowing. Nothing retries at this layer unless you install a retrying transport on the HTTP client you pass to the provider, and you decide which errors qualify.

This is the right layer for rate limits, connection resets, and 5xx responses. See [HTTP Request Retries](models/http-request-retries.md) for the transports, the `Retry-After`-aware wait strategy, and per-provider notes — including AWS Bedrock, which retries through boto3 rather than `httpx2`.

When you build your own backoff outside a transport, [`ModelHTTPError.retry_after`][pydantic_ai.exceptions.ModelHTTPError.retry_after] gives you the provider's `Retry-After` header already parsed into seconds.

## Model fallback is not a retry

[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] moves to the *next* model when the current one fails; it never re-attempts the same one. Pair it with transport retries rather than treating it as a substitute: retry the same provider for transient failures, fall back to a different provider when it's genuinely down. See [Fallback Model](models/overview.md#fallback-model).

## Tool retries

A tool retry is a message to the model: the call didn't work, here is why, try again. It is triggered by a Pydantic `ValidationError` on the tool's arguments, by the tool (or its `args_validator`, or a tool hook) raising [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], by a [tool timeout](timeouts.md#bounding-how-long-a-step-takes), and by the model calling a tool that doesn't exist.

[Tool Execution, Retries, and Failures](tools-advanced.md#tool-retries) documents the configuration: the default budget of `1`, the per-tool / per-toolset / per-run / agent-wide precedence ladder, and the choice between `ModelRetry` and [`ToolFailed`][pydantic_ai.exceptions.ToolFailed]. Three properties of the *counter* matter when you're reasoning about a run:

- **The counter is keyed by tool name, and it resets on success.** Each tool has its own count; there is no run-wide tool-retry budget. When a tool succeeds, its count is cleared — so a tool that alternates failure and success can fail many times in one run without ever exhausting a budget of `1`.
- **`max_retries=N` allows N retries, so N+1 attempts.** `max_retries=0` raises on the first failure without ever sending a retry prompt.
- **A tool name the model invented gets its own budget.** An unknown tool name produces a retry prompt listing the available tools, and consumes a budget keyed under the invented name, bounded by the agent-wide `tools` budget. So a model that hallucinates a *different* name each time keeps getting a fresh budget.

Exhausting a tool's budget raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].

### What a retry looks like in message history

A retried tool call has no [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] — the [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] takes its place, carrying the same `tool_call_id`. There is never both:

```python {title="retry_prompt_history.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    ModelRetry,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def lookup_then_answer(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('lookup_user', {'name': 'John'})])
    elif len(messages) == 3:
        return ModelResponse(
            parts=[ToolCallPart('lookup_user', {'name': 'John Doe'})]
        )
    return ModelResponse(parts=[TextPart('John Doe is user 123.')])


agent = Agent(FunctionModel(lookup_then_answer))


@agent.tool_plain
def lookup_user(name: str) -> int:
    if ' ' not in name:
        raise ModelRetry('Provide the full name.')
    return 123


result = agent.run_sync('Who is John?')
print([type(p).__name__ for m in result.all_messages() for p in m.parts])
"""
[
    'UserPromptPart',
    'ToolCallPart',
    'RetryPromptPart',
    'ToolCallPart',
    'ToolReturnPart',
    'TextPart',
]
"""
```

_(This example is complete, it can be run "as is")_

A [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] carries the failure as either a string (from `ModelRetry`) or a list of Pydantic error details (from a `ValidationError`), and renders for the model with `'Fix the errors and try again.'` appended. Its `tool_name` is set when the retry belongs to a specific tool call, and `None` when it belongs to the run's output.

Because the retry prompts stay in the history, [reusing that history](message-history.md) in a later run replays the failures to the model. If you don't want the model to see its earlier mistakes, filter them out with a [`ProcessHistory`](capabilities/process-history.md) capability.

[`ToolFailed`][pydantic_ai.exceptions.ToolFailed] is the deliberate opposite: it records a `ToolReturnPart` with `outcome='failed'` and does **not** consume the retry budget, so repeated failures are bounded by [`UsageLimits`][pydantic_ai.usage.UsageLimits] rather than by a retry count. See [Reporting a Failed Tool Result](tools-advanced.md#tool-failed).

## Output retries

The output budget is separate from the tool budget, and how it's enforced depends on how the model returns its final answer. [How output retries are enforced](agent.md#how-output-retries-are-enforced) covers both paths; the difference that matters for message history is:

- **Text path** (`output_type=str`, [`TextOutput`](output.md#text-output), [`NativeOutput`](output.md#native-output), [`PromptedOutput`](output.md#prompted-output), and responses with no usable output): one budget shared across the whole run. The retry becomes a new [`ModelRequest`][pydantic_ai.messages.ModelRequest] whose only part is a `RetryPromptPart` with `tool_name=None`.
- **Tool path** ([`ToolOutput`](output.md#tool-output)): the output budget acts as the default limit *per output tool*, overridable with [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries]. The retry prompt is bound to the output tool's `tool_call_id`, exactly like a function tool's.

Both are triggered by validation failures, by an [output function](output.md#output-functions) or [output validator](output.md#output-validator-functions) raising `ModelRetry`, and by a model response with nothing actionable in it. Both raise [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] when the budget runs out.

The last of those triggers has an exception: if the output type allows `None` — `output_type=str | None`, for instance — an empty or thinking-only response is a valid final result of `None` rather than a retry. Models that finish their work in a tool call and then emit only thinking would otherwise be pushed into producing filler text. Output validators still run on that `None`, so they can force a retry themselves by raising `ModelRetry`.

Both budgets are configured through one argument:

```python {title="retry_budgets.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', retries=3)  # (1)!

strict_output = Agent('openai:gpt-5.2', retries={'tools': 5, 'output': 1})  # (2)!
```

1. A bare `int` sets both the tool and output budgets.
2. An [`AgentRetries`][pydantic_ai.agent.AgentRetries] dict sets only the keys it names; unnamed keys keep the default of `1`.

The same argument is accepted per run — `agent.run(..., retries=...)` and friends — and for a block of runs via [`agent.override()`][pydantic_ai.agent.Agent.override]. [Which retry limit wins](tools-advanced.md#which-retry-limit-wins) has the full precedence table.

## What is never retried

- **`prepare` callbacks.** An exception raised by a per-tool `prepare=`, by [`PrepareTools`](capabilities/prepare-tools.md), or by a [dynamic toolset](toolsets.md) propagates out of the run unchanged — including `ModelRetry`, which is *not* turned into a retry prompt there. To hide a tool for a turn, return `None` from the callback rather than raising.
- **The `before_model_request` hook.** It runs while the request is still being assembled, before the model is called, so a `ModelRetry` raised there propagates out of the run instead of becoming a retry prompt — there is no response to retry yet. Raise it from one of the [other model-request hooks](hooks.md#model-request-hooks) instead: `hooks.on.after_model_request` to reject a response the model *did* produce (the rejected response stays in the message history, so the model can see what it said), `hooks.on.model_request` (`wrap_model_request`), or `hooks.on.model_request_error` (`on_model_request_error`).
- **Exceptions other than `ModelRetry` and `ToolFailed`.** Anything else a tool raises propagates out of the run rather than becoming a retry — *unless* a [capability](capabilities/overview.md) implements `on_tool_execute_error`, which sees the exception first and can return a replacement tool result or raise `ModelRetry` to keep the run going. [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] are the exceptions that are neither: they're control flow, not errors, and end the run with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output instead of propagating — except in a [realtime session](realtime/overview.md), which can't pause and instead answers the model with an explanation that the tool can't complete during the session. [Ending a run from inside a tool](timeouts.md#ending-a-run-from-inside-a-tool) has the full table.
- **Whole agent runs.** Nothing re-runs an agent for you. [Pydantic Evals](evals.md) has its own `retry_task` and `retry_evaluators` options for retrying a whole task or evaluator during an evaluation — see [Retry Strategies](evals/how-to/retry-strategies.md). Those sit outside the agent, so a retried task starts with fresh tool and output budgets.
