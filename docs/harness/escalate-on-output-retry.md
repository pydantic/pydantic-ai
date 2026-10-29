---
title: Escalate On Output Retry
description: "Try successive models after output validation failures while retaining the agent run and its retry limits."
---

# Escalate On Output Retry

Try a stronger model when responses keep failing output validation, while retaining the same conversation, successful tool results, accumulated usage, and retry limits. Use `EscalateOnOutputRetry` for typed-output agents that can benefit from another model after receiving validation feedback.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/escalate_on_output_retry/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness pydantic-ai-slim[openai]
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import EscalateOnOutputRetry, EscalationLevel

escalation = EscalateOnOutputRetry(
    levels=[
        EscalationLevel('{{ openai:small }}', after_failures=2),
        EscalationLevel('{{ openai:large }}'),
    ],
)
agent = Agent(
    output_type=int,
    capabilities=[escalation],
    retries={'output': escalation.required_output_retries},
)
```

The first model gets two output failures before the next request goes to the second model. That request sees the usual validation feedback. The second model remains selected through later function-tool steps and continues until the run succeeds or core stops it.

Each non-final `EscalationLevel` requires a positive `after_failures`; the final level omits it. A single-level ladder is valid. The capability copies the supplied level sequence so later list edits do not change its configuration.

## What advances the ladder

One accepted output retry counts as one failure:

- Text, Native Output, and Prompted Output schema validation failures.
- Output validators and output functions raising `ModelRetry`.
- Malformed output-tool arguments and rejected output-tool results.
- Empty responses and other responses with no usable output.

Function-tool retries, request-hook `ModelRetry`, request or transport errors, unrelated exceptions, and cancellation do not advance it. A recovered validation error that core accepts as output does not count. Neither does an empty response accepted as `None`.

Several output calls can fail in one response. Each accepted retry counts, so a batch can cross several thresholds and skip an intermediate level. A failed output call can also accompany a successful sibling: its retry is reported, but a successful run makes no further request.

## Budgets

Core owns retry enforcement. Set `retries={'output': ...}` on the agent or run explicitly. `required_output_retries` sums the non-final thresholds: the minimum allowance to reach the final model on a serial output path. Choose a larger allowance for retries on the final model.

The minimum is not a reachability guarantee: request-hook retries draw on the same core budget without advancing the ladder, and usage limits or cancellation can stop the run earlier. Output tools keep their own limits, including `ToolOutput(max_retries=...)` overrides. Changing models or output tools does not reset their counters. The capability neither increases these limits nor starts another run.

## Composition

- An explicit run `model=` or `Agent.override(model=...)` takes precedence over capability model selection. A constructor `Agent(model=...)` is a lower-precedence default.
- With several model selectors, the last non-`None` contribution wins. `SelectModel` and escalation do not automatically chain; choose their order deliberately.
- `ModelFallback` handles request failures within each step, starting from the escalation level's model. A successful fallback alone does not advance escalation. If that response then fails output validation, it can advance the ladder.
- Models are resolved and prepared through the existing agent machinery. Each selected model gets its own profile and settings. IDs can use a `ResolveModelId` capability; ordinary runs can also use model instances.
- The agent's explicit output configuration stays fixed. Automatic output-mode preparation still follows the selected model's profile.
- Separate instances have separate state. A new run, including one supplied with earlier `message_history`, starts on the first level. Saved message history alone does not restore escalation state.

## Durable execution

Orchestration replay rebuilds the escalation count from settled response processing before choosing the next model. Use model IDs or models registered with the durability capability, following that engine's model-resolution requirements.

The capability follows each engine's checkpoint contract. DBOS does not automatically journal ordinary function tools; decorate side-effecting tools with `@DBOS.step`. Render Workflows dispatches model and tool operations as tasks, but retrying the entry task starts the agent again. Restoring message history into a new run also starts escalation at the first level.

Replay can repeat notifications and decision spans according to the engine's event-delivery behavior. Use the engine's delivery contract when writing an external event sink.

## Specs

Register `EscalateOnOutputRetry` in `custom_capability_types` when loading an agent spec. Levels in specs use model IDs:

```yaml
retries:
  output: 2
capabilities:
  - EscalateOnOutputRetry:
      levels:
        - model: {{ openai:small }}
          after_failures: 2
        - model: {{ openai:large }}
```

## Streaming

`run`, `run_sync`, streamed graph iteration, and event streaming can advance through output candidates where core already retries them. Event consumers can see failed candidates before the run succeeds.

`run_stream` retains core's delivery boundary: once a final output stream reaches the caller, a later validation failure remains an error. The capability does not retract that output or supply a replacement stream. Partial validation observations do not advance escalation.

## Events and telemetry

`OutputRetryEvent` reports an accepted output retry through the agent's existing capability-event machinery. It carries `source` (`'response_output'` or `'output_tool'`), the rejected `response`, and core's `retry` metadata (`cause`, `attempt`, `max_retries`). Output-tool events also carry `tool_name` and `tool_call_id`. The response identifies the actual producer when request fallback served the step. Terminal budget exhaustion does not report an accepted retry.

A zero-duration `escalate_on_output_retry` span records a transition that contributes to a new request. Attributes under `escalate_on_output_retry.` are `from_level`, `to_level` (zero-based), `failures`, `model`, `producer`, and `source`. It includes no response content. Core's run and request spans retain usage accounting; the capability adds no model-request span and charges no usage itself.

## API reference

::: pydantic_ai_harness.EscalateOnOutputRetry

::: pydantic_ai_harness.EscalationLevel

::: pydantic_ai_harness.OutputRetryEvent
