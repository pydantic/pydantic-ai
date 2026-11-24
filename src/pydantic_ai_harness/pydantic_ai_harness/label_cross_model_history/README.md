# Label Cross Model History

Tell the model when earlier assistant turns in its conversation came from another model, so after a model upgrade, an A/B handoff, or a fallback, it treats them as inherited context instead of statements it made itself.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/label_cross_model_history/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import LabelCrossModelHistory

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LabelCrossModelHistory()])


async def main():
    first = await agent.run('Draft a migration plan.')
    # The second model is told the earlier answer came from another model.
    await agent.run('Review the plan.', message_history=first.all_messages(), model='openai:gpt-6-astra')
```

## How it decides

Before each model request, `LabelCrossModelHistory` compares the `model_id` of every `ModelResponse` in the request's history with the model the request is about to go to. While any of them came from another model, it states this notice:

> Some earlier assistant turns in this conversation were written by a different model, not by you. Treat them as inherited context: check their claims and commitments instead of assuming you made them.

`ModelResponse.model_id` is the identifier the agent sent the request to, as you configured it, not the dated version a provider reports as `model_name`, so a provider snapshot name doesn't count as another model. The match is exact, so the same model selected by a different identifier, such as another API prefix or through the gateway, counts as another model.

- **Fresh or single-model history**: no notice.
- **Responses without a `model_id`**, such as history saved before the field existed or a response a capability returned without a model request: not counted, since they don't say which model wrote them.
- **`FallbackModel`**: responses from any of its candidates count as the current model's own. Which candidate will serve a request isn't known before it is sent, so a switch between candidates isn't labeled. The responses a `FallbackModel` writes carry its own composite `model_id`, not the candidate that served them, so continuing that history on one of its candidates, or on a `FallbackModel` with other candidates, states the notice.
- **`ModelFallback`**: the notice is decided again for each candidate a request falls back to, so a candidate that wrote the earlier turns itself gets none.

The notice is an instruction block named `notice` with `on_change='append'`, returned empty from the first request on. When it starts to apply, it is appended to the history at that point as an `InstructionDeltaPart`, and it stays in effect while a turn from another model is in the history. When compaction or a sliding window drops the last one, an appended change withdraws it. It never names the other model: model identifiers from replayed or client-supplied history are not put into the prompt.

## Options

- `notice` (default `DEFAULT_CROSS_MODEL_NOTICE`): the text stated while the history has a turn from another model.
- `id` (default `'label_cross_model_history'`): the capability's id, which makes the `notice` block addressable as `capability:label_cross_model_history:notice`. Without an id, core can't append changes to the block and rewrites the instructions instead.

## Composition

- The notice and its withdrawal are appended where they apply instead of rewriting the instructions, so the cached prompt prefix doesn't move, including when a conversation returns to an earlier model. Models also follow an appended change more reliably than one rewritten into a prompt they continue their own history under. The message history you pass in is not changed; the run's new messages carry the appended change.
- It adds no tools, model settings, or spans. The appended change is part of the recorded message history, which core's spans include when content is included in traces.
- Its default `id` is `'label_cross_model_history'`, which qualifies the block's id as `capability:label_cross_model_history:notice`. Two instances merge into one, so the notice is stated once.
