---
title: Refine Output
description: "Have a Pydantic AI agent review and revise its answer before the run ends, for a set number of rounds."
---

# Refine Output

Have an agent review and revise its answer before the run ends. Use it where a second pass catches mistakes that a first draft misses and the extra model request is worth it.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import RefineOutput

agent = Agent('{{ anthropic:large }}', capabilities=[RefineOutput()])
```

When the run would end with an output, `RefineOutput` adds a system prompt to the conversation that tells the model to review its last answer and give the complete answer again, revised. The model answers once more, with its tools available, and that answer is the run's output. The earlier answer stays in the [message history](../message-history.md).

| Field | Default | Meaning |
|---|---|---|
| `rounds` | `1` | How many times per run the agent reviews its output |
| `instructions` | `DEFAULT_REFINE_OUTPUT_INSTRUCTIONS` | What the agent is told at the start of each review round |

```python
from pydantic_ai_harness import RefineOutput

RefineOutput(rounds=2, instructions='Check every figure against the source documents, then answer again.')
```

## Review rounds

- A round costs at least one more model request, more when the model calls tools during it, and counts toward [`UsageLimits`](../agent.md#usage-limits) like any other request.
- With structured output the model produces the output again. [Output validators and output functions](../output.md) run on every answer, not only the last, so keep them free of side effects.
- The instruction is a `SystemPromptPart` whose `metadata` holds `'origin': 'pydantic-ai-harness.refine_output.v1'` and the `round` number. It stays in the message history. Rounds are counted from the current run's messages, so a run that continues an earlier conversation starts with none used.

Two kinds of run end without a review:

- A run that pauses for [deferred tools](../deferred-tools.md) and returns `DeferredToolRequests`.
- `Agent.run_stream()`, which ends at the first output it streams. To stream a run that is reviewed, use `Agent.run()` with an `event_stream_handler`, or `Agent.iter()`.

## Durable execution

`RefineOutput` keeps no state of its own. It counts rounds from the run's messages, so there is nothing to set up for a durable run.

## Telemetry

`RefineOutput` emits no spans of its own. On an [instrumented](../logfire.md) agent, each round shows as an ordinary model request in the run's trace.

## Composition

- Several `RefineOutput` on one agent share the round count. Each adds its instructions to every round up to its own `rounds`, so two with different instructions have the model review for both in the same round.
- [Trajectory Judge](trajectory-judge.md) has a second model review the run while it is in progress. `RefineOutput` needs no second model and acts only on the output.
- An [output guardrail](guardrails.md) rejects an output that breaks a rule. `RefineOutput` asks for a second look at an output that was accepted.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/refine_output/).

## API reference

::: pydantic_ai_harness.RefineOutput

::: pydantic_ai_harness.refine_output.DEFAULT_REFINE_OUTPUT_INSTRUCTIONS
