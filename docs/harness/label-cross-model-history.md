---
title: Label Cross Model History
description: "Tell a Pydantic AI agent's model when earlier assistant turns in its history came from another model, with an ephemeral instruction that leaves the history unchanged."
---

# Label Cross Model History

Tell the model when earlier assistant turns in its conversation came from another model, so after a model upgrade, an A/B handoff, or a fallback, it treats them as inherited context instead of statements it made itself.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import LabelCrossModelHistory

agent = Agent('{{ anthropic:large }}', capabilities=[LabelCrossModelHistory()])


async def main():
    first = await agent.run('Draft a migration plan.')
    # The second model is told the earlier answer came from another model.
    await agent.run('Review the plan.', message_history=first.all_messages(), model='{{ openai:large }}')
```

## How it decides

Before each model request, `LabelCrossModelHistory` compares the `model_id` of every `ModelResponse` in the request's history with the model the request is about to go to. While any of them came from another model, it adds a notice to that request's instructions:

> Some earlier assistant turns in this conversation were written by a different model, not by you. Treat them as inherited context: check their claims and commitments instead of assuming you made them.

`ModelResponse.model_id` is the identifier the agent sent the request to, as you configured it, not the dated version a provider reports as `model_name`, so a provider snapshot name doesn't count as another model.

- **Fresh or single-model history**: no notice.
- **Responses without a `model_id`**, such as history saved before the field existed or a response a capability returned without a model request: not counted, since they don't say which model wrote them.
- **[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]**: responses from any of its candidates count as the current model's own. Which candidate will serve a request isn't known before it is sent, so a switch between candidates isn't labeled.
- **[`ModelFallback`][pydantic_ai.capabilities.ModelFallback]**: the notice is decided for the model the request goes to first. If the request then falls back to another candidate, that candidate gets the instructions decided for the first.

The notice stays on every request while a turn from another model is in the history, and goes away once compaction or a sliding window drops the last one. It never names the other model: model identifiers from replayed or client-supplied history are not put into the prompt.

## Options

- `notice` (default `DEFAULT_CROSS_MODEL_NOTICE`): the instruction added while the history has a turn from another model.

## Composition

- The notice is a dynamic instruction, not a message part. Each request computes its own, so the message history you pass in is not changed and a later model only sees the notice its own request carries. A request records the instructions it was sent with, as with any instructions.
- Adding the notice changes the instructions, which moves the prompt-cache prefix. Switching models already starts a new cache, but returning to an earlier model adds a notice its cached requests didn't have, so that one request reads less from the cache.
- It adds no tools, model settings, or spans. Core's model request span records the instructions each request was sent with, notice included, when content is included in traces.
- It has no default `id`: two instances each add their own notice.

## API reference

- [`pydantic_ai_harness.label_cross_model_history` source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/label_cross_model_history/)
- [Pydantic AI capabilities](../capabilities/overview.md)
- [Message history](../message-history.md)

::: pydantic_ai_harness.label_cross_model_history.LabelCrossModelHistory
