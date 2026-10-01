---
description: "Fall back to other models when a Pydantic AI model request fails or its response is rejected, with the Fallback capability."
---

# Fallback

[`Fallback`][pydantic_ai.capabilities.Fallback] is a [capability](overview.md) that attempts other models when the one serving a request fails, or returns a response you reject. The chain is the model selected for the step, followed by the capability's models:

```python {title="fallback_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Fallback

# Try GPT-5.6 Sol first, then Claude Opus 5.5.
agent = Agent('openai:gpt-5.6-sol', capabilities=[Fallback('anthropic:claude-opus-5-5')])
```

Because the chain starts from the model selected for the step, `Fallback` composes with the agent's model, a model passed to `agent.run(model=...)`, and [`SelectModel`](select-model.md): whichever model they choose is attempted first. When the agent has no model and nothing else selects one, the first of `Fallback`'s models is used for the first attempt:

```python {title="fallback_without_agent_model.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Fallback

agent = Agent(capabilities=[Fallback('openai:gpt-5.6-sol', 'anthropic:claude-opus-5-5')])
```

A model that was already attempted in the step is skipped, whichever capability attempted it, so listing the agent's own model among the candidates doesn't cause a duplicate request.

## When to fall back

By default, `Fallback` moves on when a request raises [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], the error a model raises when the provider's API fails, such as a 4xx or 5xx response. The `fallback_on` parameter takes the same forms as [`FallbackModel`'s](../models/overview.md#fallback-model):

- a tuple of exception types
- an exception handler, `(Exception) -> bool`
- a response handler, `(ModelResponse) -> bool`, which rejects a response that was returned successfully
- a sequence mixing any of the above

Handlers may be sync or async. A handler whose first parameter is annotated as [`ModelResponse`][pydantic_ai.messages.ModelResponse] is a response handler; any other handler is an exception handler.

When every model has failed or been rejected, `Fallback` raises [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] with every exception, plus a [`ResponseRejected`][pydantic_ai.models.fallback.ResponseRejected] counting any rejected responses.

!!! warning "Temporal"
    Under [Temporal](../durable_execution/temporal.md), a failed model request currently reaches the workflow as Temporal's `ActivityError` rather than the [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] the default `fallback_on` matches. Until model errors are rebuilt on the workflow side, widen `fallback_on` to match the errors you want to fall back on, or keep using [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], which falls back inside a single activity.

!!! note
    Provider SDKs often retry failed requests themselves before raising, which delays falling back. See [The layers](../retries.md#the-layers) in the retries guide for how to turn those retries off.

## How attempts work

Each model attempted is an *attempt* within the same request step, not a new step:

- `before_model_request` and `wrap_model_request` run once per step, and the model never sees a retry prompt.
- [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] runs again for every attempt, with `request_context.model` set to the model about to serve it, so model-specific preparation (message translation for the provider, compaction, context-window fitting) is redone for each candidate rather than inherited from the previous one. See [Retrying and falling back](custom.md#model-request-attempts).
- `after_model_request` hooks of capabilities outside `Fallback` only ever see the response that was accepted. `Fallback` is positioned innermost, so it judges a response before them.
- `usage.requests` counts the step once, so [`UsageLimits.request_limit`][pydantic_ai.usage.UsageLimits.request_limit] bounds how many steps the agent takes rather than how many models it tried. The tokens and cost of every attempt that produced a response, including rejected ones, are counted in the run's usage and checked against token and cost limits before the next attempt.

## Streaming

A streamed request falls back if the stream fails to open: `Fallback` makes the first request to the provider before any output reaches you, so a connection error or an error status can still be handled. Once output has been streamed, a failure is raised to you rather than retried.

A response handler can't reject a streamed response, because it has already been streamed by the time it could be judged. Raising [`RetryModelRequest`][pydantic_ai.exceptions.RetryModelRequest] from `after_model_request` on a streamed request raises [`UserError`][pydantic_ai.exceptions.UserError].

## Continuing suspended responses

Some providers suspend a response and continue it on a later request, like Anthropic's `pause_turn` or OpenAI's background mode. A suspended response `Fallback` accepts is pinned to the model that started it, so when a later run resumes it, the continuation goes to that model rather than to the first model in the chain. If that continuation fails, the suspended response is dropped and the turn is generated again from the start of the chain.

## `Fallback` or `FallbackModel`

Prefer the `Fallback` capability inside an agent. [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] wraps its models into a single model, so hooks that read the model, like [`model.profile`][pydantic_ai.models.Model.profile] or token counting, can't tell which model will serve the request, and preparation can't be redone per candidate by capabilities. `FallbackModel` remains the way to fall back outside an agent, with [direct model requests](../direct.md) or a standalone [`Model`][pydantic_ai.models.Model].
