---
description: "Fall back to other models when a Pydantic AI model request fails or its response is rejected, with the ModelFallback capability."
---

# Model Fallback

[`ModelFallback`][pydantic_ai.capabilities.ModelFallback] is a [capability](overview.md) that attempts other models when the one serving a request fails, or returns a response you reject. The chain is the model selected for the step, followed by the capability's models:

```python {title="model_fallback_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback

# Try the OpenAI model first, then the Anthropic one.
agent = Agent('{{ openai:medium }}', capabilities=[ModelFallback('{{ anthropic:medium }}')])
```

Because the chain starts from the model selected for the step, `ModelFallback` composes with the agent's model, a model passed to `agent.run(model=...)`, and [`SelectModel`](select-model.md): whichever model they choose is attempted first. When the agent has no model and nothing else selects one, the first of `ModelFallback`'s models is used for the first attempt:

```python {title="model_fallback_without_agent_model.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback

agent = Agent(capabilities=[ModelFallback('{{ openai:medium }}', '{{ anthropic:medium }}')])
```

A model that was already attempted in the step is skipped, whichever capability attempted it, so listing the agent's own model among the candidates doesn't cause a duplicate request.

## When to fall back

By default, `ModelFallback` moves on when a request raises [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], the error a model raises when the provider's API fails, such as a 4xx or 5xx response. The `fallback_on` parameter takes the same forms as [`FallbackModel`'s](../models/overview.md#fallback-model):

- a tuple of exception types
- an exception handler, `(Exception) -> bool`
- a response handler, `(ModelResponse) -> bool`, which rejects a response that was returned successfully
- a sequence mixing any of the above

Handlers may be sync or async. A handler whose first parameter is annotated as [`ModelResponse`][pydantic_ai.messages.ModelResponse] is a response handler; any other handler is an exception handler.

When every model has failed or been rejected, `ModelFallback` raises [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] with every exception, plus a [`ResponseRejected`][pydantic_ai.models.fallback.ResponseRejected] counting any rejected responses.

!!! note
    Provider SDKs often retry failed requests themselves before raising, which delays falling back. Set [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) to `0` on the agent, or on each candidate model but the last. As with `FallbackModel`, a candidate's own `settings` apply only to that candidate, and settings on the agent or run apply to every candidate. See [The layers](../retries.md#the-layers) in the retries guide for the full retry map.

A run that reaches its `timeout=` stops with [`RunTimedOut`][pydantic_ai.exceptions.RunTimedOut] wherever it is, including in the middle of a candidate's request: the remaining candidates aren't attempted. Likewise, a request that exceeds the run's [usage limits](../agent.md#usage-limits) raises [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] whatever `fallback_on` says, as another model wouldn't change that.

## In agent specs {#agent-specs}

`ModelFallback` can be declared in an [agent spec](../agent-spec.md), with its models as a list of model names:

```yaml {title="fallback_agent.yaml" test="skip"}
model: {{ openai:medium }}
capabilities:
  - ModelFallback: [{{ anthropic:medium }}]
```

As in Python, the spec's `model:` is attempted first, and a spec without one uses the first of `ModelFallback`'s models for the first attempt.

To choose when to fall back, give the models and `fallback_on` as keyword arguments. In a spec, `fallback_on` takes the names of [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] and its subclasses in [`pydantic_ai.exceptions`][pydantic_ai.exceptions], such as [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError] and [`ModelUnavailableError`][pydantic_ai.exceptions.ModelUnavailableError]. Exception and response handlers can only be passed in Python:

```yaml {title="fallback_on_agent.yaml" test="skip"}
model: {{ openai:medium }}
capabilities:
  - ModelFallback:
      models: [{{ anthropic:medium }}, {{ google:medium }}]
      fallback_on: [ModelRateLimitError, ModelUnavailableError]
```

Model names are resolved when they're attempted, like those passed to `ModelFallback(...)`, so a spec can also list model IDs of your own that a [`ResolveModelId`](resolve-model-id.md) capability resolves, passed to [`Agent.from_spec()`][pydantic_ai.agent.Agent.from_spec] or [`Agent.from_file()`][pydantic_ai.agent.Agent.from_file] with `capabilities=`.

## How attempts work

Each model attempted is an *attempt* within the same request step, not a new step:

- `before_model_request` and `wrap_model_request` run once per step, and the model never sees a retry prompt.
- [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] runs again for every attempt, with `request_context.model` set to the model about to serve it. The model's own message preparation and token counting are redone for each candidate, and so is any preparation a capability does in `prepare_model_request`, rather than inherited from the previous one. The built-in compaction capabilities, including the Harness ones, still run in `before_model_request` for now, so they decide once per step for the step's model, and every candidate inherits that decision. See [Retrying and falling back](custom.md#model-request-attempts).
- `after_model_request` hooks of capabilities outside `ModelFallback` only ever see the response that was accepted. `ModelFallback` is positioned innermost, so it judges a response before them.
- `usage.requests` counts the step once, so [`UsageLimits.request_limit`][pydantic_ai.usage.UsageLimits.request_limit] bounds how many steps the agent takes rather than how many models it tried. The tokens and cost of every attempt that produced a response, including rejected ones, are counted in the run's usage and checked against token and cost limits before the next attempt.

## Failed attempts and telemetry

Failed attempts are recorded the same way [`FallbackModel` records them](../models/overview.md#failed-attempts-and-usage). The response that answers lists every attempt `ModelFallback` moved on from in [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], and when every model fails, the [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] lists them in `attempts`. A rejected response's tokens and cost are counted once, in the run's [`RunUsage`][pydantic_ai.usage.RunUsage], whichever of the two you use.

When the agent is [instrumented](../logfire.md), the model request (`chat`) span is named after the model that answered and describes it. Each failed attempt gets its own `model request attempt <model name>` child span with an ERROR status. When every model fails, the `chat` span is named after the last model attempted.

## Streaming

A streamed request falls back if it fails before any output reaches you: `ModelFallback` waits for each candidate's stream to produce its first event before handing it over, so a connection error, an error status, or an error the provider reports right after accepting the request (like an overload) can still be handled. The first event is then replayed, so you see the same stream. Once output has been streamed, a failure is raised to you rather than retried. `FallbackModel` streams the same way.

Response handlers aren't consulted for a streamed response, which is accepted as is with a `UserWarning`: it has already been streamed by the time it could be judged. A capability of your own that raises [`RetryModelRequest`][pydantic_ai.exceptions.RetryModelRequest] from `after_model_request` on a streamed request gets a [`UserError`][pydantic_ai.exceptions.UserError].

## Continuing suspended responses

Some providers suspend a response and continue it on a later request, like Anthropic's `pause_turn` or OpenAI's background mode. A suspended response `ModelFallback` accepts is pinned to the model that started it, so when a later run resumes it, the continuation goes to that model rather than to the first model in the chain. Response handlers judge the continued response like any other. If that continuation fails or is rejected, the suspended response is dropped and the turn is generated again by the next model in the chain after the one it was pinned to, as it would have been had the response failed in the run that started it. The chain doesn't start over: with [batching](../batch.md), the first model would only accept a new batch, and the request would never reach the models after it. A rejected response the provider is still working on, like a batch request that hasn't finished, has its job cancelled.

## `ModelFallback` or `FallbackModel`

Prefer the `ModelFallback` capability inside an agent. [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] wraps its models into a single model, so hooks that read the model, like [`model.profile`][pydantic_ai.models.Model.profile] or token counting, can't tell which model will serve the request, and preparation can't be redone per candidate by capabilities. `FallbackModel` remains the way to fall back outside an agent, with [direct model requests](../direct.md) or a standalone [`Model`][pydantic_ai.models.Model].
