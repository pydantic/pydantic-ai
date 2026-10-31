---
description: "Retry the same model when a Pydantic AI model request fails with a transient error, with the RetryOnModelErrors capability."
---

# Retry On Model Errors

[`RetryOnModelErrors`][pydantic_ai.capabilities.RetryOnModelErrors] is a [capability](overview.md) that attempts the same model again when a request to it fails with an error that waiting may fix, like a rate limit, an overloaded model or a dropped connection:

```python {title="retry_on_model_errors.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import RetryOnModelErrors

agent = Agent('{{ openai:medium }}', capabilities=[RetryOnModelErrors()])
```

It's the agent's own retry layer: it sees the [normalized model errors](../models/overview.md#handling-model-api-errors) every model raises rather than status codes, retries a stream that fails before any output reaches you, and records every attempt on the response that answers. See [Retries](../retries.md#the-layers) for how it relates to the other layers that retry.

## Retry or fall back

`RetryOnModelErrors` attempts the *same* model again, for a failure that is likely to be gone a moment later. [`ModelFallback`](model-fallback.md) moves on to a *different* model, for a failure that won't be, like a model that's down, an exhausted quota or a request the model rejects. They're separate capabilities that compose: with both, each model is retried before the request falls back to the next one, and the next model gets a fresh budget of attempts of its own.

```python {title="retry_then_fall_back.py"}
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.capabilities import ModelFallback, RetryOnModelErrors
from pydantic_ai.exceptions import ModelOverloadedError
from pydantic_ai.models.function import AgentInfo, FunctionModel


def overloaded(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelOverloadedError('primary', 'Overloaded', retry_after=0.1)


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('The capital of France is Paris.')])


agent = Agent(
    # These deterministic models stand in for two providers.
    FunctionModel(overloaded, model_name='primary'),
    capabilities=[
        ModelFallback(FunctionModel(answer, model_name='backup')),
        RetryOnModelErrors(max_attempts=3),
    ],
)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
for attempt in result.response.failed_attempts or []:
    print(attempt.model_name, attempt.kind, attempt.retry_after)
    #> primary retry 0.1
    #> primary retry 0.1
    #> primary fallback 0.1
```

The order you list them in doesn't matter: `RetryOnModelErrors` always judges an attempt's failure before `ModelFallback` does. An error `RetryOnModelErrors` doesn't retry, or stops retrying, goes on to `ModelFallback`'s `fallback_on` as it would without it.

## When to retry

By default, a request is retried when it fails with one of these [error categories](../models/overview.md#handling-model-api-errors):

- [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError]
- [`ModelUnavailableError`][pydantic_ai.exceptions.ModelUnavailableError], including [`ModelOverloadedError`][pydantic_ai.exceptions.ModelOverloadedError]
- [`ModelServerError`][pydantic_ai.exceptions.ModelServerError]
- [`ModelConnectionError`][pydantic_ai.exceptions.ModelConnectionError], including the transport's [`ModelTimeoutError`][pydantic_ai.exceptions.ModelTimeoutError]

Errors that fail the same way when the request is sent again aren't retried by default: a [`ModelRequestRejectedError`][pydantic_ai.exceptions.ModelRequestRejectedError] (a bad request, bad credentials, an unknown model or an overflowing context window), a [`ModelQuotaExceededError`][pydantic_ai.exceptions.ModelQuotaExceededError], a [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout] for a request that ran out its whole [`request_timeout`](../timeouts.md#model-request-deadlines), or a plain `ModelAPIError` that falls in no category.

The `retry_on` parameter chooses differently. It takes the same forms as `ModelFallback`'s [`fallback_on`](model-fallback.md#when-to-fall-back):

- a tuple of exception types
- an exception handler, `(Exception) -> bool`
- a response handler, `(ModelResponse) -> bool`, which rejects a response that was returned successfully, so the same model is asked again
- a sequence mixing any of the above

```python {title="retry_on_rate_limits.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import RetryOnModelErrors
from pydantic_ai.exceptions import ModelOverloadedError, ModelRateLimitError

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[RetryOnModelErrors(retry_on=(ModelRateLimitError, ModelOverloadedError), max_attempts=5)],
)
```

Whatever `retry_on` says, a request is not retried when:

- the provider sent an `x-should-retry: false` header, as OpenAI and Anthropic do for errors that retrying won't fix (see [`provider_retry_hint`][pydantic_ai.exceptions.ModelHTTPError.provider_retry_hint]). Their SDKs heed the header too, so turning the SDK's retries off doesn't lose it. A `ModelFallback` can still move on to another model.
- the run's [usage limits](../agent.md#usage-limits) stopped it, as another attempt wouldn't change that.

Once a model has been attempted `max_attempts` times (`3` by default, the first attempt included), the last error is raised as is, or the last rejected response accepted, so that a `ModelFallback` can take over. Two `RetryOnModelErrors` capabilities, say one with its own `retry_on` for rate limits, count the same attempts, each against its own `max_attempts`.

## Waiting between attempts

Before each retry, the agent waits as long as the error's [`retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] says the provider asked for. Without one, it backs off exponentially: `initial_delay` seconds (`1` by default) before the first retry, doubling for each retry after it, up to `max_delay` seconds (`30` by default). A random part of up to half of each wait is taken off, so that many runs failing at the same moment don't all retry at the same moment too.

A request is not retried, and the error is raised at once, when the wait would be longer than `max_delay`, as a provider asking for minutes rather than seconds usually means a quota resets later, or when the wait would end after the request's or the run's [deadline](../timeouts.md#bounding-a-whole-run). Either way, a `ModelFallback` can then move on to another model while the run still has time.

A [`request_timeout`](../timeouts.md#model-request-deadlines) covers a request to a model with all its retries and the waits between them: each retry gets what's left of the deadline the first attempt started with, and a wait that would end after it isn't made. A model `ModelFallback` moves on to gets a fresh deadline, as without `RetryOnModelErrors`.

## How attempts work

Each retry is an *attempt* within the same request step, as each model `ModelFallback` tries is. The model never sees a retry prompt, `before_model_request` and `wrap_model_request` run once per step, and [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] runs again for each attempt. See [How attempts work](model-fallback.md#how-attempts-work) for the details, which are the same.

`usage.requests` counts the step once, so [`UsageLimits.request_limit`][pydantic_ai.usage.UsageLimits.request_limit] isn't spent on retries. The tokens and cost of a response a response handler rejected are counted in the run's usage and checked against its token and cost limits before the next attempt.

The response that answers lists every attempt before it in [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], retries and fallbacks in one list, in the order they were made. Each one's [`kind`][pydantic_ai.messages.ModelRequestAttempt.kind] says whether the same model was attempted after it (`'retry'`) or the request moved on to another model (`'fallback'`), and its [`retry_after`][pydantic_ai.messages.ModelRequestAttempt.retry_after] records the wait the provider asked for. When the agent is [instrumented](../logfire.md), each failed attempt also gets its own `model request attempt` span, as described under [Failed attempts and telemetry](model-fallback.md#failed-attempts-and-telemetry).

## Provider SDK retries

Most provider SDKs retry failed requests themselves before raising, so each attempt could otherwise be several requests on the wire, with waits of their own. `RetryOnModelErrors` turns those retries off for the requests it makes, by sending them with [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) set to `0`, so it's the one layer that retries and `max_attempts` is the number of requests sent. A `max_retries` you set yourself, on the agent, the run or a model, is kept, and a model that doesn't support the setting isn't sent it.

## Streaming

A streamed request is retried when it fails before any of its output reaches you: the agent waits for the stream's first event before handing the stream over, so a connection error, an error status, or an error the provider reports right after accepting the request can still be retried. Once output has been streamed, a failure is raised to you rather than retried. Response handlers aren't consulted for a streamed response, which is accepted as is with a `UserWarning`. See [Streaming](model-fallback.md#streaming) for when a request is streamed.

## In agent specs {#agent-specs}

`RetryOnModelErrors` can be declared in an [agent spec](../agent-spec.md). Its `retry_on` takes the names of [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] and its subclasses in [`pydantic_ai.exceptions`][pydantic_ai.exceptions]; exception and response handlers can only be passed in Python:

```yaml {title="retry_agent.yaml" test="skip"}
model: {{ openai:medium }}
capabilities:
  - RetryOnModelErrors:
      retry_on: [ModelRateLimitError, ModelOverloadedError]
      max_attempts: 5
  - ModelFallback: [{{ anthropic:medium }}]
```

## Durable execution

Under a [durability capability](../durable_execution/overview.md), each attempt is a durable model operation of its own, as under `ModelFallback`: on [Temporal](../durable_execution/temporal.md), an activity per attempt. The wait before a retry is a durable timer on Temporal and a durable sleep on [DBOS](../durable_execution/dbos.md), so a workflow that is recovered while it waits doesn't wait again from the start. On [Prefect](../durable_execution/prefect.md), the wait is an ordinary sleep in the flow.

On Temporal, an agent with `RetryOnModelErrors` makes every model error [non-retryable](../durable_execution/temporal.md#activity-retries) in the model request activity, as if it were listed in `TemporalDurability`'s `non_retryable_model_errors`: the error fails the activity on its first attempt and reaches `RetryOnModelErrors` as its own class, so the capability alone retries it. The activity's retry policy still covers what never reaches a hook, like a worker crash or an activity timeout, so keep it, with a bounded `maximum_attempts` in [`model_activity_config`](../durable_execution/temporal.md#activity-configuration), as Temporal's default retries without limit. Don't set `maximum_attempts=1`: a timed-out activity would then fail the workflow.

On DBOS and Prefect, the step's own retries, off by default, retry a failed request before the error reaches `RetryOnModelErrors`, and the two multiply, so leave them off.
