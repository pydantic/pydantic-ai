---
description: "Fall back to other models when a Pydantic AI model request fails or its response is rejected, with the ModelFallback capability."
---

# Model Fallback

[`ModelFallback`][pydantic_ai.capabilities.ModelFallback] is a [capability](overview.md) that attempts other models when the one serving a request fails, or returns a response you reject. It's the way to fall back to other models in an agent; [`FallbackModel`](../models/overview.md#fallback-model) remains for [direct model requests](../direct.md) and standalone models (see [Migrating from `FallbackModel`](#migrating-from-fallbackmodel)). The chain is the model selected for the step, followed by the capability's models:

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

Handlers may be sync or async. A handler whose first parameter is annotated as [`ModelResponse`][pydantic_ai.messages.ModelResponse] is a response handler; any other handler is an exception handler. See [Response-Based Fallback](../models/overview.md#response-based-fallback) for examples of handlers, which work the same for both.

!!! warning "A response handler on its own replaces the default"
    Passing only a response handler as `fallback_on` replaces the default `(ModelAPIError,)`, so API errors are raised rather than fallen back from. To keep falling back on them, pass both in a list, as below.

Here the first model's response is cut off and rejected, the second model fails, and the third answers. The answering response lists both attempts it moved on from, and the rejected response's tokens count in the run's usage:

```python {title="model_fallback_mixed_policy.py"}
from pydantic_ai import Agent, ModelAPIError, ModelMessage, ModelResponse, TextPart
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage


def truncated(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('The capital of France is')],
        usage=RequestUsage(input_tokens=60, output_tokens=100),
        finish_reason='length',
    )


def unavailable(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError('second', 'Service unavailable')


def complete(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('The capital of France is Paris.')],
        usage=RequestUsage(input_tokens=60, output_tokens=8),
    )


def was_truncated(response: ModelResponse) -> bool:
    return response.finish_reason == 'length'


agent = Agent(
    # These deterministic models stand in for three providers.
    FunctionModel(truncated, model_name='primary'),
    capabilities=[
        ModelFallback(
            FunctionModel(unavailable, model_name='second'),
            FunctionModel(complete, model_name='backup'),
            fallback_on=[ModelAPIError, was_truncated],
        )
    ],
)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
print(result.usage)
#> RunUsage(input_tokens=120, output_tokens=108, requests=1)
for attempt in result.response.failed_attempts or []:
    print(attempt.model_name, attempt.outcome, attempt.usage)
    #> primary rejected RequestUsage(input_tokens=60, output_tokens=100)
    #> second error None
```

When every model has failed or been rejected, `ModelFallback` raises [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] with every exception, plus a [`ResponseRejected`][pydantic_ai.models.fallback.ResponseRejected] counting any rejected responses.

!!! note
    Provider SDKs often retry failed requests themselves before raising, which delays falling back. Set [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) to `0` on the agent, or on each candidate model but the last. As with `FallbackModel`, a candidate's own `settings` apply only to that candidate, and settings on the agent or run apply to every candidate. See [The layers](../retries.md#the-layers) in the retries guide for the full retry map.

A run that reaches its `timeout=` stops with [`RunTimedOut`][pydantic_ai.exceptions.RunTimedOut] wherever it is, including in the middle of a candidate's request: the remaining candidates aren't attempted. Likewise, a request that exceeds the run's [usage limits](../agent.md#usage-limits) raises [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] whatever `fallback_on` says, as another model wouldn't change that.

## Output validation retries

Set `output_retries` to a positive integer to move to the next candidate when the serving model repeatedly fails output validation. `output_retries=1` moves on after the first accepted output retry; the default `None` keeps request-only fallback.

```python {title="model_fallback_output_retries.py"}
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_output_text='draft'),
    capabilities=[ModelFallback(TestModel(custom_output_text='ready'), output_retries=1)],
    retries={'output': 2},
)


@agent.output_validator
def validate(output: str) -> str:
    if output == 'draft':
        raise ModelRetry('Finish the draft.')
    return output


print(agent.run_sync('Answer.').output)
#> ready
```

Unlike `fallback_on`, this happens across ordinary request steps: the next model sees the rejected response and validation feedback. Successful tools are retained, usage keeps accumulating, and the logical run and trace continue. The next model is the next candidate in the chain; it need not be larger or more capable.

An integer applies the same threshold to every model. For independent thresholds, pass a mapping from model IDs to positive integers. Keys match the string used to select the model, including a `ResolveModelId` alias, or [`Model.model_id`][pydantic_ai.models.Model.model_id] for a model instance. Unlisted models keep request-only fallback; an empty mapping disables output-driven switching. The capability copies the mapping when constructed.

- Schema failures, output-validator or output-function `ModelRetry`, invalid output-tool arguments, and unusable responses count when core accepts their retry. Function-tool retries, request-hook `ModelRetry`, request errors, and cancellation do not count towards this threshold.
- Each serving model has its own count. Several failed output calls in one response count against their producer and trigger at most one move; they cannot skip a candidate that has not run.
- After switching, later steps start at the chosen candidate. Request failures can still use `fallback_on` to try later candidates, but the chain does not return to candidates retired by output failures. At the end of the chain, output retries stay on the last serving model until core's limits end the run.
- The initial model still comes from the agent, `run(model=...)`, `override(model=...)`, or `SelectModel`. After an output-driven switch, the persistent candidate takes precedence for later steps in that run. A suspended response continues on the model that started it before output-driven routing is considered.

Configure `retries={'output': ...}` explicitly to allow the desired switches and any retries on the final model. `output_retries` does not replenish that allowance, change per-output-tool overrides, or bypass usage limits. Request-hook retries consume the same core budget without advancing this policy, so no threshold guarantees that a later candidate is reached.

The policy state is isolated per run. Passing saved message history into a new run starts the policy afresh. Under durable execution, replay reconstructs the state by replaying accepted-output notifications outside journaled model operations.

Graph-level streaming through `run(event_stream_handler=...)`, `run_stream_events()`, and `iter()` can continue to the next candidate after validation rejects a completed response. `run_stream()` does not replace final output already delivered to its consumer.

For models with a configured threshold, [`OutputRetryEvent`][pydantic_ai.capabilities.OutputRetryEvent] reports each accepted retry through the capability-event stream. It includes `source`, the rejected `response`, core's `retry` metadata, and output-tool identity when applicable. A zero-duration `model_fallback.output_fallback` span records a switch when its candidate is prepared, with `model_fallback.model`, `model_fallback.source`, and the optional `model_fallback.producer`. These contain no response content and add no usage charges. Request-attempt failures retain their existing telemetry.

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

The mapping form also accepts `output_retries`:

```yaml
retries: {output: 3}
capabilities:
  - ModelFallback:
      models: [{{ openai:medium }}, {{ anthropic:medium }}]
      output_retries: 1
```

For different thresholds, use the selected model IDs as keys:

```yaml
model: {{ openai:medium }}
retries: {output: 4}
capabilities:
  - ModelFallback:
      models: [{{ anthropic:medium }}]
      output_retries:
        {{ openai:medium }}: 2
        {{ anthropic:medium }}: 1
```

## How attempts work

Each model attempted is an *attempt* within the same request step, not a new step:

- `before_model_request` and `wrap_model_request` run once per step, and the model never sees a retry prompt.
- [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] runs again for every attempt, with `request_context.model` set to the model about to serve it. [`RunContext.model`][pydantic_ai.tools.RunContext.model] stays the model selected for the step, so a hook that depends on the serving model, its [profile](../models/overview.md#inspecting-a-models-profile) or its context window reads `request_context.model`. The model's own message preparation and token counting are redone for each candidate, and so is any preparation a capability does in `prepare_model_request`, rather than inherited from the previous one. The built-in compaction capabilities, including the Harness ones, still run in `before_model_request` for now, so they decide once per step for the step's model, and every candidate inherits that decision. See [Retrying and falling back](custom.md#model-request-attempts).
- A [native tool factory](../native-tools.md#dynamic-configuration) is called again for each model after the first, with that model as `ctx.model`, so each model is sent the native tools it would be sent on its own, and a [provider-adaptive tool](overview.md#provider-adaptive-tools)'s local fallback is offered to a model its native tool doesn't apply to. A native tool a `before_model_request` hook removed stays removed.
- `after_model_request` hooks of capabilities outside `ModelFallback` only ever see the response that was accepted. `ModelFallback` is positioned innermost, so it judges a response before them.
- `usage.requests` counts the step once, so [`UsageLimits.request_limit`][pydantic_ai.usage.UsageLimits.request_limit] bounds how many steps the agent takes rather than how many models it tried. The tokens and cost of every attempt that produced a response, including rejected ones, are counted in the run's usage and checked against token and cost limits before the next attempt.

## Failed attempts and telemetry

Failed attempts are recorded the same way [`FallbackModel` records them](../models/overview.md#failed-attempts-and-usage). The response that answers lists every attempt `ModelFallback` moved on from in [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], and when every model fails, the [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] lists them in `attempts`. A rejected response's tokens and cost are counted once, in the run's [`RunUsage`][pydantic_ai.usage.RunUsage], whichever of the two you use.

When the agent is [instrumented](../logfire.md), the model request (`chat`) span is named after the model that answered and describes it, with its usage. Each failed attempt gets its own `model request attempt <model name>` child span with an ERROR status: an error is recorded on it as an exception, and a rejected response carries its usage, cost and finish reason. The span names the model it tried in its `gen_ai.request.model` and `gen_ai.provider.name` attributes, and its zero-based position among the request's attempts in `pydantic_ai.model_request.attempt`. When every model fails, the `chat` span is named after the last model attempted. See [Failed attempts and usage](../models/overview.md#failed-attempts-and-usage) for how these spans relate to the ones the tried model opens itself.

## Streaming

A streamed request falls back if it fails before any output reaches you: `ModelFallback` waits for each candidate's stream to produce its first event before handing it over, so a connection error, an error status, or an error the provider reports right after accepting the request (like an overload) can still be handled. The first event is then replayed, so you see the same stream. Once output has been streamed, a failure is raised to you rather than retried. `FallbackModel` streams the same way.

Response handlers aren't consulted for a streamed response, which is accepted as is with a `UserWarning`: it has already been streamed by the time it could be judged. A request is streamed whenever something consumes the run's events: [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] (which the [UI adapters](../ui/overview.md) use), [`agent.iter()`][pydantic_ai.agent.Agent.iter] with `node.stream()`, and [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] or [`agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] with an `event_stream_handler`, an [`agent.on_event`][pydantic_ai.agent.Agent.on_event] listener, or a capability that reacts to events. A capability of your own that raises [`RetryModelRequest`][pydantic_ai.exceptions.RetryModelRequest] from `after_model_request` on a streamed request gets a [`UserError`][pydantic_ai.exceptions.UserError].

## Continuing suspended responses

Some providers suspend a response and continue it on a later request, like Anthropic's `pause_turn` or OpenAI's background mode. Every response records the model that served it as [`model_id`][pydantic_ai.messages.ModelResponse.model_id], so when a later run resumes a suspended response one of `ModelFallback`'s models started, the continuation goes to that model rather than to the first model in the chain. A suspended response started by a model outside the chain is continued by the step's model, as without `ModelFallback`. Response handlers judge the continued response like any other. If that continuation fails or is rejected, the suspended response is dropped and the turn is generated again by the next model in the chain after the one it was pinned to, as it would have been had the response failed in the run that started it. The chain doesn't start over: with [batching](../batch.md), the first model would only accept a new batch, and the request would never reach the models after it. A rejected response the provider is still working on, like a batch request that hasn't finished, has its job cancelled.

## Request deadlines

A [`request_timeout`](../timeouts.md#model-request-deadlines) in the model settings bounds each model's attempt as a whole, across the provider SDK's own retries: each model `ModelFallback` tries gets a fresh deadline, and the [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout] it raises when the deadline passes moves on to the next model by default. A model's own `settings` can set a different `request_timeout` for it.

## Batching

With [`BatchModelRequests`](../batch.md), every attempt is batched with the model that serves it, so when the first model's batch request fails, the request goes into a batch on the next model. Batching submits a request again itself after a transient failure, so `ModelFallback` only sees a batch failure once resubmitting is used up, or one resubmitting wouldn't fix.

## Durable execution

Under a [durability capability](../durable_execution/overview.md), each attempt is a durable model operation of its own: on [Temporal](../durable_execution/temporal.md), an activity per model tried. The activity's retry policy retries a failed request before the error reaches `ModelFallback`'s `fallback_on`, and Temporal's default retry policy retries without limit: set `maximum_attempts` in the retry policy of `TemporalDurability`'s [`model_activity_config`](../durable_execution/temporal.md#activity-configuration) so that a failing model doesn't hold up the fallback. Each model's `request_timeout` deadline runs as a durable timer around its own activity.

A streamed request is consumed inside its durable unit, where its events reach a durability capability's `event_stream_handler`, so a stream that fails after producing output there is raised rather than fallen back from, as outside durable execution, and what it produced counts towards the run's usage.

A model `ModelFallback` moves to is rebuilt for its activity like any model a run uses: a model name string is built on the worker, and a model instance must be registered with the durability capability's `models`, then given to `ModelFallback` by that name:

```python {title="model_fallback_temporal.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

backup = AnthropicModel('{{ anthropic:medium.name }}', provider=AnthropicProvider(api_key='...'))

agent = Agent(
    '{{ openai:medium }}',
    name='resilient_agent',
    capabilities=[
        ModelFallback('backup'),
        TemporalDurability(models={'backup': backup}),
    ],
)
```

A `FallbackModel` instead runs its whole chain inside one activity, along with the deadlines it gives each model, and when every model fails, its [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] reaches the workflow without the attempts it lists, so the usage of the responses it rejected isn't recorded.

## Migrating from `FallbackModel`

`ModelFallback` is the recommended way to fall back to other models inside an agent. [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] isn't deprecated and doesn't warn: like [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel] next to the [`Instrumentation`](instrumentation.md) capability, it remains the way to fall back with [direct model requests](../direct.md) or a standalone [`Model`][pydantic_ai.models.Model], outside an agent.

To move an agent to the capability, keep the first model as the agent's model and pass the rest, with the same `fallback_on`, to `ModelFallback`:

```python {title="model_fallback_migration.py" test="skip" lint="skip"}
# Before
agent = Agent(FallbackModel(primary, backup, fallback_on=[ModelAPIError, was_truncated]))

# After
agent = Agent(primary, capabilities=[ModelFallback(backup, fallback_on=[ModelAPIError, was_truncated])])
```

Everything a `FallbackModel` does carries over:

- `fallback_on` takes the same exception types, exception handlers and response handlers, mixed in a list, sync or async.
- Each model keeps its own `settings`, and settings on the agent or run apply to every model.
- The answering response lists the attempts moved on from in [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], and when every model fails, [`FallbackExceptionGroup.attempts`][pydantic_ai.exceptions.FallbackExceptionGroup] lists them.
- A rejected response's tokens and cost count in the run's usage and towards its limits, also when the run is cancelled, times out or ends on an error `fallback_on` doesn't cover.
- Each model gets a fresh `request_timeout` deadline.
- A stream that fails before any output falls back, and response handlers don't apply to a streamed response.
- A suspended response is continued on the model that started it.
- The same spans: a `chat` span for the model that answered, and a `model request attempt` span for each attempt moved on from.

And a few things only the capability can do:

- [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] runs for each model, so hooks see the model that will serve the request, its profile and context window, and prepare the request for it. The model's own message preparation and token counting are redone for each model too, so [`count_tokens_before_request`][pydantic_ai.usage.UsageLimits.count_tokens_before_request] works.
- A [native tool factory](../native-tools.md#dynamic-configuration) sees the model each attempt is at as `ctx.model`, where under a `FallbackModel` it sees the `FallbackModel` itself.
- The chain starts from the model selected for the step, so it composes with `agent.run(model=...)` and [`SelectModel`](select-model.md).
- Under [durable execution](#durable-execution), each attempt is a durable model operation of its own.
- With [batching](#batching), each attempt goes into a batch on its own model.
- A failed or rejected continuation of a resumed suspended response moves on to the next model rather than starting the chain over.
