# Report probability and gate uncertain Boolean judgments

Plan for [#9641](https://github.com/pydantic/pydantic-ai/issues/9641). This PR proposes an API; implementation follows maintainer agreement.

## Problem and scope

`LLMJudge` turns a textless model's verdict into a binary score and omits confidence information from the evaluator result. The reporter's gated judge improved agreement with labelled answers from 308/390 to 356/390 while Jev decided 60% of cases. Those measurements have not been independently reproduced.

Preserve existing score, assertion, and `rubric: str` behavior. Add explicit probability reporting and an opt-in Boolean confidence gate first. Add `LLMJudge.criteria: BoolCriteria | None = None` after the structured input design in [#9640](https://github.com/pydantic/pydantic-ai/issues/9640) is agreed.

## Recommended API

| Surface | Proposed contract |
| --- | --- |
| `LLMJudge.probability` | `bool \| OutputConfig`, default `False`; emit a separate native probability metric. |
| `DecisionModelSettings.decision_boolean_confidence_threshold` | `float \| None`, default `None`; accept values in `[0, 1]` and hand off native Boolean answers below the threshold. |
| `UnsureBoolean` in `pydantic_ai.models.decision` | A new `ModelAPIError` carrying the field name, verdict, raw probability, confidence, and configured threshold. |
| `LLMJudge.criteria` | Add `BoolCriteria \| None`, default `None`; keep the existing `rubric: str` parameter and string-rubric behavior unchanged. |

The first three surfaces can ship before structured input. `LLMJudge.criteria` follows #9640.

```python
from pydantic_ai.models.decision import DecisionHandOff, DecisionModelSettings, UnsureBoolean
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_evals.evaluators import LLMJudge

# Proposed API; llm is the configured fallback model.
judge = LLMJudge(
    rubric='The output is equivalent to the expected output for the input.',
    include_input=True,
    include_expected_output=True,
    model=FallbackModel(
        TypeSafeModel('jev-1.13.0'), llm,
        fallback_on=(DecisionHandOff, UnsureBoolean),
    ),
    model_settings=DecisionModelSettings(decision_boolean_confidence_threshold=0.6),
    probability=True,
)
```

### Probability reporting

Add previously absent scalar Boolean distributions to `DecisionModel`'s existing `provider_details['probabilities'][field]` mapping as `{'true': p, 'false': 1 - p}`. This adds Boolean field entries to response metadata even when the evaluator's probability metric is disabled; no existing metadata entry is removed or renamed. Retain existing confidence, choice distributions, and rubric scores. Do not reconstruct raw probability from the rounded confidence value.

When requested, `LLMJudge` emits the final selected response's `pass` probability under a separate default name ending in `_probability`. `OutputConfig` can override that name. Preserve the existing score/assertion naming rules and values. A generative fallback has no native probability, so omit this metric for that row; do not substitute its generated score or the rejected primary response's probability. Document that probability aggregates cover only rows with this metric.

### Boolean confidence gate

Use the existing Boolean confidence definition relative to `decision_boolean_threshold`, including its six-decimal reporting precision. For multiple native Boolean fields, hand off the whole response if any field is below the configured confidence threshold, before output acceptance or tool execution. Leave list/mapping membership decisions, rubric scores, and route thresholds unchanged. No configured threshold means no new handoff.

At the default verdict cutoff of `0.5`, confidence `0.6` accepts approximately `p <= 0.2` or `p >= 0.8`; threshold equality is accepted. With a different verdict cutoff, use its existing scaled distance rather than hard-coded probability bands. Validate thresholds before calling the backend.

`UnsureBoolean` extends `ModelAPIError` directly. Existing `DecisionHandOff` requires route/probability attributes that describe route selection; Boolean uncertainty must not populate those attributes with unrelated values. Raise `UnsureBoolean` only after answer decoding succeeds and outside `_fill`'s backend-error wrapper, so `FallbackModel` can receive the uncertainty. Existing backend failures remain terminal. For this opt-in threshold, uncertainty in a selected route's filled output causes whole-step fallback after route filling and before tool execution. `FallbackModel` already handles response predicates and exception tuples, so no change to its dispatch is needed.

### Explicit criteria and material

Keep `rubric: str` unchanged and add `LLMJudge.criteria: BoolCriteria | None = None`. When `criteria` is provided, the caller's rubric becomes the question description and `BoolCriteria` supplies the true/false criteria. This opt-in path follows #9640; with `criteria=None`, the existing string-rubric path and prompts stay unchanged.

Reuse the existing criteria type, exported from pydantic_ai:

```python
from pydantic_ai import BoolCriteria

criteria = BoolCriteria(
    true='The output conveys the facts requested by the input and expected_output.',
    false='The output contradicts or omits a requested fact.',
)
```

When `criteria` is provided, use named material fields `output`, `input`, and `expected_output`, honoring the existing inclusion flags. Serialize material to JSON with Pydantic serialization and raise a clear error for unsupported values. Send that material through #9640's explicit structured content API. Put the caller's rubric in the Boolean field description and the true/false criteria in `BoolCriteria`; exclude the criteria from state.

A text-capable judge renders the same explicit criteria together with the rubric in its grading instructions. The mixed fallback path depends on #9640 defining JSON-to-text projection without mutating stored history. Keep both models judging the same material and criteria. Do not parse free-form rubric strings into criteria or change their prompt encoding.

## Implementation and verification

| Criterion | Verification to add |
| --- | --- |
| Existing users retain binary scores, assertions, names, prompts, and the `rubric: str` type | Existing judge snapshots, a default-settings regression, and a rubric type assertion in `tests/evals/test_llm_as_a_judge.py`. |
| Native probability remains exact and separate from confidence | Boolean response metadata cases in `tests/models/test_decision.py`; preserve other distribution shapes. |
| The confidence gate respects custom cutoffs, endpoints, equality, and multiple fields | Targeted decision-model cases for forced routes, two-request route fills, streaming, validation before backend calls, unchanged non-Boolean paths, and unchanged backend failures. |
| Mixed fallback accepts uncertain Boolean outputs without inventing probability | End-to-end judge cases in `tests/evals/test_typesafe_judges.py`; test predicate and exception-tuple fallback, metric omission, and backend-error propagation. |
| Structured material and explicit criteria reach both models correctly when `criteria` is provided | After #9640, record each backend test with its own test function; assert object state, question criteria, and text fallback projection. |

Before accepting default thresholds for an application, rerun the reporter's benchmark with its rubric and labelled sample. Confidence is a model signal, not a guarantee of calibration. The unlabelled benchmark's similar pass rates do not prove accuracy.

Update the judge and decision-model documentation and relevant agent skills with the implementation. This plan-only PR changes no runtime code, tests, or published API.

## References

- [#8480](https://github.com/pydantic/pydantic-ai/pull/8480) intentionally introduced binary textless grading.
- [Existing fallback response handlers](https://github.com/pydantic/pydantic-ai/pull/8450#issuecomment-5723244252) already support confidence-based fallback.
- [Decision settings, handoff contracts, and answer metadata](pydantic_ai_slim/pydantic_ai/models/decision.py) define the existing primitives.
- [`BoolCriteria`](pydantic_ai_slim/pydantic_ai/output.py) already supplies Boolean question criteria.
- [TypeSafe primitives](https://docs.typesafe.ai/primitives) separate material, instructions, and criteria.
- [The reporter's implemented judge](https://github.com/ggozad/haiku.rag/blob/main/evaluations/evaluations/evaluators/system_one.py) exposes probability and gates fallback.
