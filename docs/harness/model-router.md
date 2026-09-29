---
title: Model Router
description: Choose an agent's model from a named menu using another Pydantic AI model.
---

# Model Router

Choose the model for an agent run from a named menu, using another Pydantic AI model to make the choice.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/model_router/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

Declare each model with a stable key and a description of when it should be used. `default` is the choice used if routing fails:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.model_router import ModelChoice, ModelRouter

router = ModelRouter(
    choices={
        'fast': ModelChoice(
            'openai:gpt-5.6-luna',
            'Lookups, extraction, or a change confined to one place.',
        ),
        'capable': ModelChoice(
            'openai:gpt-5.6-sol',
            'Architecture, security, or a decision that is expensive to get wrong.',
        ),
    },
    router_model='openai:gpt-5.6-luna',
    default='capable',
)

agent = Agent(
    instructions='You are a helpful engineering assistant.',
    capabilities=[router],
)

result = agent.run_sync('Explain why this authentication design is safe.')
print(result.output)
```

This example needs the OpenAI provider group:

```bash
pip/uv-add pydantic-ai-harness 'pydantic-ai-slim[openai]'
```

The router creates an internal agent named `model_router`. Its output has one `choice` field, whose options are the choice keys, and each option carries its description in the output schema. Language models answer through structured output, while typed models such as TypeSafe's decision models pick one option with the description of each in hand, without generating text.

## Routing input

The router agent is given the run's message history as its own `message_history`, ending with the request being routed: the user's prompt on the first step, and tool results or a retry on later ones. A router on another provider therefore receives the conversation content used to make the decision, including any files in it. A router model that cannot take a part of that history raises `UserError`, as a decision model does for files, and the run fails rather than routing to `default`.

That input is not bounded. In `'per_step'` mode each routing request grows with the conversation, so a long run eventually spends more on routing than the choice is worth, and a history that outgrows the router's context window fails the request and falls back to `default`. Pair `'per_step'` routing with [compaction](compaction.md) to bound it: the compacted history is what the router reads, which is usually what you wanted it to read anyway.

Routing reads that history one step behind compaction. Pydantic AI selects the step's model before `before_model_request` runs, and `before_model_request` is where compaction rewrites the history, so on the step that compacts, the router is handed the history compaction is about to replace -- the largest the history ever gets. Size the router's context window against the point compaction triggers, not against the compacted result.

## When routing runs

`mode` controls the decision cadence:

| Mode | Router cost | Behavior |
|---|---|---|
| `'once'` | One router model request per run. | Keeps the first choice for every model request in the run. It cannot react to later message history. |
| `'per_step'` | One router model request per logical model request step. | Reconsiders the choice as Pydantic AI exposes more message history, so later steps can use a different model. |

`'once'` is the default. It keeps routing cost and latency fixed. Use `'per_step'` when a run can begin with routine work and later reach a step that needs a different model.

Pydantic AI calls model selectors once per logical step. Provider polling or continuation inside one step stays on the selected model.

## Probability and failure

Set `probability_threshold` when constructing `ModelRouter` to reject a pick the router was unsure of. For example, with `probability_threshold=0.8`, a pick given a probability below `0.8` falls back to `default`.

The probability is read from `provider_details['probabilities']['choice'][<pick>]`, which is what a decision model such as TypeSafe reports for a pick-one field. It is not the field's `provider_details['confidence']`, which measures something else. A router model that reports no probability for its pick keeps its pick, so ordinary language models route without provider-specific metadata, and the threshold has no effect on them.

If the router model raises at request time, returns invalid model behavior after its normal output retries, or returns an unknown key, `ModelRouter` selects `default` and the main run continues. A `UserError` is the exception: it means the request can never succeed as configured, so it propagates. Cancellation is not converted into a fallback either.

Configuration mistakes do not fall back. An empty menu, an unknown `default`, a `probability_threshold` outside `0` to `1`, and a `router_model` Pydantic AI cannot resolve all raise `UserError` from the `ModelRouter` constructor, so a typo surfaces immediately instead of routing every request to `default` for the life of the agent.

## Options

| Option | Default | Behavior |
|---|---:|---|
| `choices` | required | Mapping of keys to `ModelChoice(model, description)`. Keys and descriptions must be non-empty. |
| `router_model` | required | Any model name or `Model` instance accepted by Pydantic AI. |
| `default` | required | Configured choice used for router errors and low-probability picks. |
| `mode` | `'once'` | Route `'once'` per run or `'per_step'`. |
| `probability_threshold` | `None` | Minimum probability, from `0` to `1`, the router must give its pick. A missing probability accepts the pick. |

Both the router and choice models may be model names or configured `Model` instances. Model instances preserve their clients, credentials, base URLs, and instrumentation.

## Observability

Each routing decision emits a `model_router.select` span on the parent run's tracer. It has these attributes:

| Attribute | Meaning |
|---|---|
| `model_router.choice` | Choice actually used after fallback. |
| `model_router.probability` | The probability the router gave its pick, when reported. |
| `model_router.fallback_reason` | `none`, `low_probability`, or `error`. |
| `model_router.mode` | `once` or `per_step`. |
| `model_router.run_step` | Logical request step being routed. |
| `model_router.error.type` | Exception class when the router request failed, whether the default handled it or, for a `UserError`, it propagated. |

The router run is instrumented with the parent run's instrumentation settings, from `Agent.instrument_all()`, `agent.instrument`, or an `Instrumentation` capability, so its `invoke_agent` and model request spans nest under `model_router.select` when the parent run is traced and are not emitted when it is not. The internal agent name `model_router` also lets Logfire group its requests, token use, cost, and latency separately from the main agent.

## Composition and execution constraints

`ModelRouter` implements Pydantic AI's existing [`get_model()`](../capabilities/custom.md#selecting-the-model) hook. A model passed directly to `run(model=...)`, through a run spec, or through `agent.override(model=...)` takes precedence and skips capability routing. `ModelRouter` takes precedence over the model passed to the `Agent` constructor. When several capabilities contribute a model, Pydantic AI uses the last contribution. Two `ModelRouter`s on one agent share the default `id` `'model_router'`, so they are merged into one: their menus are combined and the later one's other settings win. Give each its own `id` to keep them apart, in which case the later one routes.

Selection also runs before `before_model_request` and `wrap_model_request`, so a capability that gates or rewrites the prompt at either point does not cover the router request. An [`InputGuardrail`](guardrails.md) acts when the parent model request is made, by which time the router has already sent the original prompt to `router_model` and been billed for it, and it is still sent when the guard then blocks the parent call. Keep `router_model` inside the same trust boundary as the models in `choices`, and enforce any policy about content that must never leave that boundary before the run rather than inside it.

The router request shares the parent run's `RunUsage` and usage limits, so its tokens, cost, and requests count toward the run limits and totals. It reserves one request for the pending parent model call.

Under a durable execution capability such as `TemporalDurability` or `DBOSDurability`, the router request is a durable operation: it runs in the engine's activity or step, and replay restores the recorded choice instead of asking the router again and possibly choosing another model. A failed router request is recorded as the `default` fallback rather than retried under the engine's retry policy. Register the model of every `ModelChoice` in the durability capability's `models=`, like any model a durable run selects. The durability capability also needs the agent to have a `model` of its own. `ModelRouter` carries the stable default `id='model_router'`, so durable recovery works without configuration.

`ModelRouter` is configured in Python and does not publish an AgentSpec entry. Its choice and router fields can contain live `Model` instances, and the output type depends on the runtime choice keys.

## Why this is in Harness

Pydantic AI core owns the selection mechanism: `ModelSelectionContext`, `ModelSelector`, the `get_model()` hook, and `SelectModel`. `ModelRouter` composes those primitives with an internal typed agent, a named menu, decision cadence, probability fallback, error fallback, and telemetry. Those are routing policy and reusable agent behavior, so the composition belongs in Harness. Moving it into core would duplicate `SelectModel` with one opinionated way to implement its callable.

## Pydantic AI references

- [Selecting the model](../capabilities/custom.md#selecting-the-model)
- [Select Model](../capabilities/select-model.md)
- [Capabilities](../capabilities/overview.md)
- [Instrumentation](../capabilities/instrumentation.md)

## API reference

::: pydantic_ai_harness.model_router.ModelRouter

::: pydantic_ai_harness.model_router.ModelChoice
