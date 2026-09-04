# Retries

"Retry" means seven different things in an agent run, at seven different layers, and they don't share budgets. Mixing them up is the usual cause of a run that retries far more (or far less) than expected. This page is the map; each layer links to the page that configures it in detail.

## The layers

| Layer | What it re-attempts | Configured with | What it adds to message history |
|---|---|---|---|
| [Transport](#transport-retries) | The same HTTP request to the provider | [`AsyncHTTPX2TenacityTransport`][pydantic_ai.retries.AsyncHTTPX2TenacityTransport] on your HTTP client | Nothing — the agent never sees the attempts |
| [Provider SDK](#provider-sdk-retries) | The same HTTP request, re-issued by the provider SDK's own client | The SDK client itself; defaults and configuration are provider-specific | Nothing — the agent never sees the attempts |
| [Durable execution](durable_execution/overview.md) | The whole model request, re-executed by the workflow engine — re-entering every layer nearer the wire; unbounded by default on Temporal (`maximum_attempts=0`) | `retry_policy` in Temporal's `ActivityConfig`, `max_attempts` in DBOS's `StepConfig`, `retries` in Prefect's `TaskConfig` | Nothing — the engine replays the step |
| [Model fallback](#model-fallback-is-not-a-retry) | The same request against a *different* model | [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] | Only the winning response |
| [Tool](#tool-retries) | One tool call, by asking the model to correct it | `retries={'tools': N}` and per-tool limits | A [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] with `outcome='retried'` as that call's result |
| [Output](#output-retries) | The model's final answer, by asking it to correct it | `retries={'output': N}` and [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries] | A [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart], or a retried `ToolReturnPart` — see [below](#output-retries) |
| [Model-request hooks](hooks.md) | The model request, from `after_model_request`, `wrap_model_request`, or `on_model_request_error` raising `ModelRetry` | The hook itself; it draws on the **output** budget | A new request carrying a `RetryFeedbackPart` |

Only the last three are "agent retries" — they cost a model round trip each, because a retry *is* another request. The other four are invisible to the model: it never sees an attempt fail.

## Retry multiplication

The layers don't share budgets, but they stack: a retry at one layer wraps the attempts of every layer nearer the wire. If a logical call can issue up to `N` model requests — the initial attempt plus one follow-up per tool call and whatever the [tool](#tool-retries) and [output](#output-retries) retry budgets add — each model request is sent by a provider SDK client allowed up to `M` attempts, and each attempt travels on a transport allowed up to `K` attempts, so one logical call can put up to `N`×`M`×`K` wire requests on the network. Under [durable execution](durable_execution/overview.md) the step that runs the model request retries too, re-entering `M` and `K` each time — and on Temporal that retry count is unbounded unless you set `maximum_attempts` yourself.

- `N` — model requests per logical call: the initial attempt, one follow-up per tool call (even a successful tool call queues another request), and any retry prompts the [tool](#tool-retries) and [output](#output-retries) budgets add
- `M` — attempts per model request inside the provider SDK client. The SDK determines this budget; for example, an OpenAI client configured with `max_retries=N` allows `1 + N` attempts. See [provider SDK retries](#provider-sdk-retries) for the provider-specific settings.
- `K` — attempts per request on the wire: the transport's stop strategy, so `stop_after_attempt(N)` allows `N` total attempts (`K = N`), not one plus retries — see [transport retries](#transport-retries)

Every wire request pays its own latency — and bills tokens once the request reaches the model — so the worst case, not the happy path, is what your budgets must absorb. [`UsageLimits`][pydantic_ai.usage.UsageLimits] bounds only `N`: its `request_limit` (default `50`) counts model requests per run and never sees the wire requests the SDK client and transport add beneath them. [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout] applies per attempt — a retrying SDK client re-arms it for every retry — and only on the [model classes that forward it](timeouts.md#bounding-how-long-a-step-takes). See [Timeouts](timeouts.md#bounding-how-long-a-step-takes) for the time side.

A run with `retries={'output': 2}` (up to 3 model requests for the final answer alone), the OpenAI SDK's default `max_retries=2` (3 attempts per request), and a transport stopped by `stop_after_attempt(2)` (2 attempts per wire request) can put `3` × `3` × `2 = 18` requests on the network. Each of those carries its own `ModelSettings(timeout=10)` deadline — 180 seconds of request time in the worst case, before any backoff wait between retries.

## Transport retries

Transport retries live below the model client: a failed HTTP request is re-sent without the agent ever knowing. Nothing retries at this layer unless you install a retrying transport on the HTTP client you pass to the provider, and you decide which errors qualify.

This is the right layer for rate limits, connection resets, and 5xx responses. See [HTTP Request Retries](models/http-request-retries.md) for the transports, the `Retry-After`-aware wait strategy, and per-provider notes — including AWS Bedrock, which retries through boto3 rather than `httpx2`.

When you build your own backoff outside a transport, [`ModelHTTPError.retry_after`][pydantic_ai.exceptions.ModelHTTPError.retry_after] gives you the provider's `Retry-After` header already parsed into seconds.

## Provider SDK retries {#provider-sdk-retries}

Between the transport and the model sits one more layer the agent never sees: the provider SDK's own client, which re-issues failed requests before your code hears about them. Its defaults, retryable errors, and configuration differ by provider, so size `M` from the client you use.

See the provider-specific settings for [OpenAI](models/openai.md#custom-openai-client), [Anthropic](models/anthropic.md#custom-http-client), [Google](models/google.md#http-retries), [Groq](models/groq.md#sdk-retries), [Cohere](models/cohere.md#sdk-retries), and [AWS Bedrock](models/bedrock.md#configuring-retries).

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

A retry that belongs to a tool call *is* that call's result: a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] carrying the call's `tool_call_id`, marked `outcome='retried'`. So a retried call and a successful one look the same in history apart from the outcome, and there is never a second part standing in for the result:

```python {title="retry_history.py"}
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
    'ToolReturnPart',
    'ToolCallPart',
    'ToolReturnPart',
    'TextPart',
]
"""
```

_(This example is complete, it can be run "as is")_

The part's `content` is the feedback itself — a string (from `ModelRetry`) or a list of Pydantic error details (from a `ValidationError`) — and it reaches the model on the provider's native error channel, the same one a failed result uses: Anthropic's `is_error`, Bedrock's `status='error'`, Gemini's `error` key, or an `{"error": ...}` wrapper where the provider has none. A retry that answers no call is a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart] instead — see [Feedback that belongs to no tool call](#feedback-that-belongs-to-no-tool-call). A stored history, or your own code, can still carry a [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart]; one naming a tool reaches the model on that same channel.

Because retries stay in the history, [reusing that history](message-history.md) in a later run replays the failures to the model. If you don't want the model to see its earlier mistakes, filter them out with a [`ProcessHistory`](capabilities/process-history.md) capability.

[`ToolFailed`][pydantic_ai.exceptions.ToolFailed] records the same kind of part with `outcome='failed'` instead, and does **not** consume the retry budget, so repeated failures are bounded by [`UsageLimits`][pydantic_ai.usage.UsageLimits] rather than by a retry count. The outcome is the whole difference: `'retried'` asks the model to correct the call, `'failed'` tells it to adapt. See [Reporting a Failed Tool Result](tools-advanced.md#tool-failed).

## Output retries

The output budget is separate from the tool budget, and how it's enforced depends on how the model returns its final answer. [How output retries are enforced](agent.md#how-output-retries-are-enforced) covers both paths; the difference that matters for message history is:

- **Text path** (`output_type=str`, [`TextOutput`](output.md#text-output), [`NativeOutput`](output.md#native-output), [`PromptedOutput`](output.md#prompted-output), and responses with no usable output): one budget shared across the whole run. There is no tool call to answer, so the retry becomes a new [`ModelRequest`][pydantic_ai.messages.ModelRequest] whose only part is a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart] — see [Feedback that belongs to no tool call](#feedback-that-belongs-to-no-tool-call).
- **Tool path** ([`ToolOutput`](output.md#tool-output)): the output budget acts as the default limit *per output tool*, overridable with [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries]. The retry is bound to the output tool's `tool_call_id` and looks exactly like a function tool's.

Both are triggered by validation failures, by an [output function](output.md#output-functions) or [output validator](output.md#output-validator-functions) raising `ModelRetry`, and by a model response with nothing actionable in it. Both raise [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] when the budget runs out.

The last of those triggers has an exception: if the output type allows `None` — `output_type=str | None`, for instance — an empty or thinking-only response is a valid final result of `None` rather than a retry. Models that finish their work in a tool call and then emit only thinking would otherwise be pushed into producing filler text. Output validators still run on that `None`, so they can force a retry themselves by raising `ModelRetry`.

### Feedback that belongs to no tool call

A text-path retry, and a `ModelRetry` from a [model-request hook](hooks.md#triggering-retries-with-modelretry), have no tool call to answer. They become a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart], which records *why* the response couldn't be used and nothing about how to say it:

```python {title="retry_feedback_history.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    ModelRetry,
    RetryFeedbackPart,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('7' if len(messages) == 1 else 'seven')])


agent = Agent(FunctionModel(answer))


@agent.output_validator
def spell_it_out(output: str) -> str:
    if output.isdigit():
        raise ModelRetry('answer with the number spelled out as a word')
    return output


result = agent.run_sync('How many continents are there?')
print([p for m in result.all_messages() for p in m.parts if isinstance(p, RetryFeedbackPart)])
"""
[
    RetryFeedbackPart(
        content='answer with the number spelled out as a word',
        cause='model_retry',
        timestamp=datetime.datetime(...),
    )
]
"""
```

_(This example is complete, it can be run "as is")_

The stored part carries no wording at all; each model turns it into a sendable part when the request is built, and its [`cause`][pydantic_ai.messages.RetryFeedbackPart.cause] decides which one. That is why the same history can be replayed against any model.

| `cause` | What the model is sent | Why |
|---|---|---|
| `'validation_error'` | a user turn whose text is wrapped in `<validation_errors>…</validation_errors>` | The feedback quotes back the output the model itself wrote, which may in turn quote untrusted user input. The tag is what marks those words off from a person's turn. |
| `'no_output'`, `'model_retry'` | a mid-conversation system prompt, degraded to `<system>`-tagged user text where the provider takes no system message mid-conversation | The text is a message your own code wrote knowing it would be shown — the same trust you already give [`instructions`][pydantic_ai.Agent]. |

A closing tag inside either wrapper is escaped, so text the model had a hand in cannot end the statement that carries it.

!!! warning "What goes in the system voice"
    The system channel is the highest-privilege text a model reads, and a [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] message goes there verbatim. Interpolating model output into that message hands the model's words the system voice; pass a fixed message and let the validation errors carry the specifics — those go out in the user voice, inside the fence, precisely because they quote the model.

    A `RetryFeedbackPart` that opens a history — through a hand-built `message_history`, an adapter load, [compaction](capabilities/compaction.md), or [`ProcessHistory`](capabilities/process-history.md) filtering — is treated exactly like a [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] written in that position, so a `'model_retry'` one there joins the run's standing prompt and reaches the provider's own system field. Put your own system prompt first if that matters.

!!! note "`RetryPromptPart` is no longer emitted"
    Retries used to be a single [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] for both cases, which meant a tool-less retry arrived at the model as ordinary user text it couldn't tell from something a person wrote. A retry that answers a tool call is now that call's [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] with `outcome='retried'` carrying the feedback unwrapped; one that answers no call is a `RetryFeedbackPart`. The class is still accepted and still deserializes from stored histories, so existing histories and code that hands one back through [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] keep working — a stored one is translated into whichever of the two parts above it always meant before the request goes out.

## What is never retried

- **`prepare` callbacks.** An exception raised by a per-tool `prepare=`, by [`PrepareTools`](capabilities/prepare-tools.md), or by a [dynamic toolset](toolsets.md) propagates out of the run unchanged — including `ModelRetry`, which is *not* turned into a retry prompt there. To hide a tool for a turn, return `None` from the callback rather than raising.
- **The `before_model_request` hook.** It runs while the request is still being assembled, before the model is called, so a `ModelRetry` raised there propagates out of the run instead of becoming a retry prompt — there is no response to retry yet. Raise it from one of the [other model-request hooks](hooks.md#model-request-hooks) instead: `hooks.on.after_model_request` to reject a response the model *did* produce (the rejected response stays in the message history, so the model can see what it said), `hooks.on.model_request` (`wrap_model_request`), or `hooks.on.model_request_error` (`on_model_request_error`).
- **Exceptions other than `ModelRetry` and `ToolFailed`.** Anything else a tool raises propagates out of the run rather than becoming a retry — *unless* a [capability](capabilities/overview.md) implements `on_tool_execute_error`, which sees the exception first and can return a replacement tool result or raise `ModelRetry` to keep the run going. [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] are the exceptions that are neither: they're control flow, not errors, and end the run with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output instead of propagating — except in a [realtime session](realtime/overview.md), which can't pause and instead answers the model with an explanation that the tool can't complete during the session. [Ending a run from inside a tool](timeouts.md#ending-a-run-from-inside-a-tool) has the full table.
- **Whole agent runs.** Nothing re-runs an agent for you. [Pydantic Evals](evals.md) has its own `retry_task` and `retry_evaluators` options for retrying a whole task or evaluator during an evaluation — see [Retry Strategies](evals/how-to/retry-strategies.md). Those sit outside the agent, so a retried task starts with fresh tool and output budgets.
