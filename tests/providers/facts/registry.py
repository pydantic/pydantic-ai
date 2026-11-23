"""Hand-written provider facts, each with the recorded test that proves it (proposal R32).

A provider fact is something the library hard-codes about a provider's API: a model it excludes from a feature, a
model that can't turn reasoning off, the error shape it maps to a category. Each `ProviderFact` names where the fact
is written and the recorded probe that shows the provider behaving that way, so a stale fact fails a re-recording
instead of a user's request. A member of a set that `scripts/check_provider_facts.py` registers carries
`# fact: <id>` to point at its row.

`tests/providers/facts/test_registry.py` checks that every row's fact and probe exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FactKind = Literal['rejects', 'takes_effect', 'error_shape', 'mandatory_reasoning']
"""What the probe records.

- `'rejects'`: whether the provider rejects a request the fact names, or accepts it once an exclusion is lifted.
- `'takes_effect'`: a setting changes what the provider does, like `max_tokens` ending a response with
  `finish_reason='length'`.
- `'error_shape'`: the error body the provider sends for a failure, and the error category it maps to.
- `'mandatory_reasoning'`: whether a model can turn reasoning off, shown by the provider accepting what
  `thinking=False` sends.
"""


@dataclass(frozen=True)
class ProviderFact:
    id: str
    """The name a `# fact: <id>` comment uses."""
    claim: str
    """What the provider does, as the probe records it."""
    where: str
    """Where the fact is written, as `path::symbol`, the symbol dotted for a method."""
    probe: str
    """The pytest node id of the recorded test that proves the claim; without a parameter id, every case of it."""
    expect: FactKind


FACTS: tuple[ProviderFact, ...] = (
    ProviderFact(
        id='azure-realtime-hang-up',
        claim=(
            'Azure OpenAI ends a WebRTC call from the server: the `/openai/v1` hangup endpoint answers 200, and a '
            'second hangup answers 404 `call_id_not_found`. The exclusion that made `hang_up()` raise was removed in '
            'https://github.com/pydantic/pydantic-ai/pull/11075.'
        ),
        where='pydantic_ai_slim/pydantic_ai/realtime/azure.py::AzureRealtimeModel.hang_up',
        probe='tests/realtime/test_azure_ws.py::test_webrtc_hang_up_ends_the_call',
        expect='rejects',
    ),
    ProviderFact(
        id='mistral-prompt-too-long',
        claim=(
            'Mistral answers a prompt over the context window with a 400 of type `invalid_request_prompt_too_long` '
            'and code `3059`, a `ModelContextWindowExceededError` (https://github.com/pydantic/pydantic-ai/pull/12882).'
        ),
        where='pydantic_ai_slim/pydantic_ai/models/mistral.py::_status_error',
        probe='tests/providers/facts/test_error_shapes.py::test_recorded_error_shape[mistral-prompt-too-long]',
        expect='error_shape',
    ),
    ProviderFact(
        id='mistral-invalid-args',
        claim=(
            "`3051` is Mistral's generic invalid arguments code, so a `3051` whose message is not about the context "
            'length is a `ModelRequestRejectedError` (https://github.com/pydantic/pydantic-ai/pull/12882).'
        ),
        where='pydantic_ai_slim/pydantic_ai/models/mistral.py::_status_error',
        probe='tests/providers/facts/test_error_shapes.py::test_recorded_error_shape[mistral-invalid-args]',
        expect='error_shape',
    ),
    ProviderFact(
        id='deepseek-max-tokens',
        claim=(
            'DeepSeek limits output with `max_tokens` and ignores `max_completion_tokens`, so the `max_tokens` setting '
            "goes out as `max_tokens` and the response stops at it with `finish_reason='length'` "
            '(https://github.com/pydantic/pydantic-ai/pull/12385).'
        ),
        where='pydantic_ai_slim/pydantic_ai/providers/deepseek.py::DeepSeekProvider.model_profile',
        probe='tests/models/test_deepseek.py::test_deepseek_max_tokens_limits_output',
        expect='takes_effect',
    ),
    ProviderFact(
        id='mistral-large-4-reasoning-floor',
        claim="Mistral Large 4 can't turn reasoning off: `thinking=False` sends its lowest effort, `'low'`, which it accepts.",
        where='pydantic_ai_slim/pydantic_ai/providers/mistral.py::_LARGE_4_MODELS',
        probe='tests/test_thinking_wire_contract.py::test_reasoning_wire_contract[mistral-large-4-disable]',
        expect='mandatory_reasoning',
    ),
    ProviderFact(
        id='mistral-small-reasoning-off',
        claim="`mistral-small-latest` turns reasoning off: it accepts the `reasoning_effort='none'` `thinking=False` sends.",
        where='pydantic_ai_slim/pydantic_ai/providers/mistral.py::_ADJUSTABLE_REASONING_MODELS',
        probe='tests/test_thinking_wire_contract.py::test_reasoning_wire_contract[mistral-small-disable]',
        expect='mandatory_reasoning',
    ),
    ProviderFact(
        id='magistral-medium-reasoning-off',
        claim=(
            "`magistral-medium-latest` turns reasoning off: it accepts the `reasoning_effort='none'` `thinking=False` "
            'sends.'
        ),
        where='pydantic_ai_slim/pydantic_ai/providers/mistral.py::_ADJUSTABLE_REASONING_MODELS',
        probe='tests/test_thinking_wire_contract.py::test_reasoning_wire_contract[mistral-magistral-latest-disable]',
        expect='mandatory_reasoning',
    ),
    ProviderFact(
        id='gemini-2-5-pro-min-thinking-budget',
        claim=(
            "Gemini 2.5 Pro can't turn thinking off, so `thinking=False` sends its minimum `thinking_budget`, 128, "
            'which it accepts.'
        ),
        where='pydantic_ai_slim/pydantic_ai/models/google.py::_GEMINI_2_5_PRO_MIN_THINKING_BUDGET',
        probe='tests/test_thinking_wire_contract.py::test_reasoning_wire_contract[google-gemini-25-pro-false-min-budget]',
        expect='mandatory_reasoning',
    ),
    ProviderFact(
        id='openrouter-mandatory-reasoning-floor',
        claim=(
            "OpenRouter makes reasoning mandatory on these routes, so `thinking=False` sends `effort='minimal'`, "
            'which they accept.'
        ),
        where='pydantic_ai_slim/pydantic_ai/providers/openrouter.py::_MANDATORY_REASONING_MODEL_PREFIXES',
        probe='tests/models/test_openrouter.py::test_openrouter_thinking_false_on_mandatory_reasoning_route',
        expect='mandatory_reasoning',
    ),
    ProviderFact(
        id='openrouter-forced-tool-choice-while-reasoning',
        claim="Qwen3.8 Flash on OpenRouter rejects `tool_choice='required'` while it reasons, with a 400.",
        where='pydantic_ai_slim/pydantic_ai/providers/openrouter.py::_FORCED_TOOL_CHOICE_REJECTED_WHILE_REASONING_MODEL_PREFIXES',
        probe='tests/models/test_openrouter.py::test_openrouter_forced_tool_choice_while_reasoning_provider_400_without_flag',
        expect='rejects',
    ),
    ProviderFact(
        id='gemini-api-key-invalid',
        claim=(
            'The Gemini API answers an invalid key with a 400 `INVALID_ARGUMENT` whose reason is `API_KEY_INVALID`, a '
            '`ModelAuthenticationError`.'
        ),
        where='pydantic_ai_slim/pydantic_ai/_model_errors.py::google_status_category',
        probe=(
            'tests/test_retrying_transport_error_categories.py::'
            'test_retrying_transport_gave_up_recorded_error_category[google-api-key-invalid]'
        ),
        expect='error_shape',
    ),
)
