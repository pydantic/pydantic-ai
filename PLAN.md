# Plan: Decide how structured Decisions instructions should be rendered

## Question for maintainers

Should `OpenAIDecisionsModel` keep sending structured instruction values as JSON-formatted text, or render those values as prose before sending them to the OpenAI Decisions API?

## Evidence and uncertainty

- The [Decisions guide](https://developers.openai.com/api/docs/guides/decisions) uses prose for ordinary text-only instructions. It also documents an iterable of text and image parts for question-specific image instructions.
- The current [OpenAI Python SDK request type](https://github.com/openai/openai-python/blob/main/src/openai/types/decision_create_params.py) accepts `str | Iterable[DecisionQuestionInputPart]`. A JSON-formatted string is therefore valid for the declared wire type; neither the type nor the examples establish that JSON text is rejected.
- The issue author reports 93.2% for OpenAI Decisions versus 96.2% for OpenAI and TypeSafe on an approximately 400-case labeled set. That result has not been independently reproduced, and no controlled A/B comparison of the two instruction serializations has been run.
- The issue contains similar prose-rendering proposals from other contributors, including a report of a local patch. No public PR was linked when checked, and the issue remains unassigned.

## Options

1. **Keep JSON-formatted text.** This preserves the current prompt and its structured labels, but may expose internal keys such as `field` and differs from the prose style in the guide.
2. **Render stable-order prose.** This follows the guide's text-only examples and can omit internal labels, but changes the prompt seen by the model and may change routing outcomes. Measure before adopting this behavior.

## Proposed sequence if maintainers prefer prose

1. Confirm field order and handling of unknown instruction keys. A proposal already described by yftx293 uses `background`, `goal`, `context`, `premise`, `question`, `option`, `yes`, then `no`, and omits `field` because the question name already carries it; maintainers should confirm or revise this order.
2. Confirm whether a controlled A/B on the issue's labeled set is required before implementation or before merge. Compare the current serialization and proposed prose using the same model/version and dataset, and report the result without claiming improvement in advance.
3. Assign the issue to the contributor if the direction is approved.
4. Submit a minimal OpenAI Decisions-only serializer change with request-level regressions for structured dictionaries, lists, plain strings, and absent instructions. Leave other providers unchanged.
5. Run the focused model tests and repository checks remotely; include the A/B result or clearly state if it remains unavailable.

## Requested maintainer decision

Please choose the intended wire representation, confirm the evaluation bar, and assign an implementer if the prose approach is accepted. No runtime code is included in this plan PR.
