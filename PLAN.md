# Propose structured input for decision models

Proposal for [#9640](https://github.com/pydantic/pydantic-ai/issues/9640). This PR contains the API plan; runtime implementation follows agreement on the input and history shapes.

## Problem

Jev can judge JSON objects natively, but an agent prompt containing serialized JSON reaches `DecisionRequest.state` as a string. The reporter measured representative AUC of 0.910 with an object and 0.883 with the same object serialized as text. These are the reporter's measurements, not an independently reproduced benchmark.

The backend boundary already supports `JsonValue`. The missing capability is an explicit structured input that survives message storage, retries, and model fallback. Existing string prompts must keep their current meaning.

## Proposed API

Add `JsonContent(value: JsonValue)` in `pydantic_ai.messages`, with the discriminator `kind='json-content'`, as an explicit `UserContent` item:

```python
from pydantic import JsonValue

from pydantic_ai import Agent
from pydantic_ai.messages import JsonContent
from pydantic_ai.models.typesafe import TypeSafeModel

agent = Agent(TypeSafeModel('jev-1.13.0'), output_type=bool, instructions='Are the answers equivalent?')
state: JsonValue = {'question': 'What is 2 + 2?', 'expected_answer': '4', 'generated_answer': 'Four'}
result = await agent.run([JsonContent(state)])
```

`JsonContent` accepts JSON values, including objects, arrays, scalars, and null. Arbitrary Python objects still require caller serialization. Plain strings, including JSON-looking strings, remain text. Bare dictionaries do not become valid `Agent.run()` prompts.

The value is stored in the normalized user message, not in a model setting or application metadata. This keeps replay and fallback tied to the input the caller supplied.

## Model behavior

Add `ModelProfile.supports_json_input`, defaulting to `False`, and enable native input in the decision-model profile. At the existing `Model.prepare_messages()` boundary, project `JsonContent` to compact JSON text for models without native JSON input. Project an outgoing copy; retain `JsonContent` in recorded history. This reuses the existing cross-model message preparation instead of adding JSON handling independently to every text adapter.

For decision models, allow one `JsonContent` as the prompt's sole data item; ignore `CachePoint` markers as today. Reject mixed text/JSON input and multiple JSON items with an actionable `UserError`. Text models can serialize JSON items alongside other content in the original order.

Use the same mapped state for route selection and field filling. Forward native JSON through both `TypeSafeModel` and `SystemOneModel`.

## Decision history and retries

Keep every existing text-state shape unchanged. For structured input, use these shapes:

| Context | `DecisionRequest.state` |
| --- | --- |
| No history or subsequent steps | The JSON value itself |
| Earlier conversation | `{'history': history, 'state': value}` |
| Tool activity or retry after the prompt | `{'history': history, 'state': value, 'done': done}` |

Omit `history` when empty in the retry shape, as the current mapper does. Record a previous structured user prompt as `{'user': value}` within `history`. Identify a supplied JSON value by its presence, so null and empty values remain valid input.

Use an envelope when context exists. Do not merge `history` or `done` into the caller's object: those keys might belong to the data being judged. Preserve the structured value through resumed turns and retries, including requests containing multiple `UserPromptPart` instances; reject an ambiguous active prompt instead of concatenating JSON with text.

## Implementation and validation

- Add the content type, discriminator validation, exports, and message JSON round-trip coverage in `messages.py` and `tests/test_messages.py`.
- Add the profile fact and non-mutating text projection in the existing model preparation path. Verify request, streaming, token-counting, and direct model API paths use a consistent projection; include fallback from a decision model to a text model.
- Update decision message mapping in `models/decision.py`. Cover objects, arrays, scalars, null, empty values, mixed-input rejection, previous conversation, retries, tool turns, resumed turns, and route/fill requests in `tests/models/test_decision.py`.
- Record TypeSafe and System One wire tests with their own test functions. Assert native object state and unchanged legacy text payloads.
- Verify instrumentation describes the effective structured input, and message persistence retains the JSON value. Audit UI, realtime, and durable-execution consumers of `UserContent`; document or implement explicit handling where required, rather than silently dropping the new item.
- Update decision-model and input/history documentation and the building-Pydantic-AI-agents skill when implementing.

Adding a public `UserContent` arm is compatibility-relevant for exhaustive consumers. Run the API compatibility gate and apply the version policy's warning, migration, release-note, and waiver requirements where needed before shipping. This proposal changes no runtime API or behavior.

## Scope

This proposal covers structured input and its preservation. Changes to `LLMJudge` probability handling or rubric placement in #9641 are separate. Automatic parsing of JSON-looking text and model-setting overrides of the input are outside this proposal.

## Decision requested

Agree on `JsonContent` as the explicit input, text projection for models without native JSON input, and the contextual `state` envelope before implementation.

## References

- [Issue and benchmark](https://github.com/pydantic/pydantic-ai/issues/9640)
- [TypeSafe state documentation](https://docs.typesafe.ai/concepts/state)
- [Retry-state fix #8967](https://github.com/pydantic/pydantic-ai/pull/8967)
