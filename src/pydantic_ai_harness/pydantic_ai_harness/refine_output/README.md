# Refine Output

Have an agent review and revise its answer before the run ends. Use it where a second pass catches mistakes that a first draft misses and the extra model request is worth it.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/refine_output/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import RefineOutput

agent = Agent('anthropic:claude-opus-5-5', capabilities=[RefineOutput()])
```

When the run would end with an output, `RefineOutput` adds an instruction in the application's voice to the conversation that tells the model to review its last answer and give the complete answer again, revised. The model answers once more, with its tools available, and that answer is the run's output. The earlier answer stays in the [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/).

| Field | Default | Meaning |
|---|---|---|
| `rounds` | `1` | How many times per run the agent reviews its output |
| `instructions` | `DEFAULT_REFINE_OUTPUT_INSTRUCTIONS` | What the agent is told at the start of each review round |

```python
from pydantic_ai_harness import RefineOutput

RefineOutput(rounds=2, instructions='Check every figure against the source documents, then answer again.')
```

## Review rounds

- A round costs at least one more model request, more when the model calls tools during it, and counts toward [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) like any other request.
- With structured output the model produces the output again. [Output validators and output functions](https://pydantic.dev/docs/ai/core-concepts/output/) run on every answer, not only the last, so keep them free of side effects.
- The instruction is a `ReviewRoundPart`, an [`InstructionCapabilityPart`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.InstructionCapabilityPart) kind (a [capability part](https://pydantic.dev/docs/ai/capabilities/custom/#capability-parts) in the application's voice) whose `round` field holds the round number and never reaches the model. It keeps the default conversation scope, so it stays in the message history. Find review rounds by `isinstance(part, ReviewRoundPart)` (from `pydantic_ai_harness.refine_output`), not by their text. Each run starts with no rounds used.
- Code that streams the run sees every answer as the model writes it: the first one, then each revised one. Only the last is the run's output.

Four kinds of run end without a review:

- A run that pauses for [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) and returns `DeferredToolRequests`.
- A run that ends while the provider is still processing a model request, such as a batch or background response, and returns [`PendingModelResponse`](https://pydantic.dev/docs/ai/core-concepts/batch/#ending-a-run-while-it-waits).
- A run whose output did not come from a model request, such as the refusal of an [input guardrail](../guardrails/) that blocked the prompt. A review would send the model the prompt the guardrail kept from it. `RefineOutput` recognizes such an output by a response with no `model_name`, so a custom `Model` should set `model_name` on the responses it returns, or its runs are not reviewed.
- `Agent.run_stream()`, which ends at the first output it streams. To stream a run that is reviewed, use `Agent.run()` with an `event_stream_handler`, or `Agent.iter()`.

## Durable execution

The rounds a run has used live in memory, in the instance `for_run` creates for that run. Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) the hook that counts them (`after_node_run`) runs in workflow code, so when Temporal, DBOS, or Prefect replays a run, it runs again in the same order over the journaled responses and counts the same rounds.

## Telemetry

`RefineOutput` emits no spans of its own. On an [instrumented](https://pydantic.dev/docs/ai/integrations/logfire/) agent, each round shows as an ordinary model request in the run's trace.

## Composition

- Several `RefineOutput` on one agent each count their own rounds, and each adds its instructions to every round up to its own `rounds`. Two with different instructions have the model review for both in the same round.
- The count does not depend on the message history, so [compaction](../compaction/) that summarizes a review instruction away does not start the rounds again.
- [Trajectory Judge](../trajectory_judge/) has a second model review the run while it is in progress. `RefineOutput` needs no second model and acts only on the output.
- An [output guardrail](../guardrails/) rejects an output that breaks a rule. `RefineOutput` asks for a second look at an output that was accepted.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/refine_output/).
