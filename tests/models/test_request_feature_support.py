"""Check what each model class does with each request feature against what its profile says.

`models/AGENTS.md` asks a model class to raise on a request feature it can't form, never to drop it silently, and the
`ModelProfile` flags are where a class says what it can form. This module checks both against the wire, with the probes
and cases of `test_model_settings_support.py`: for each feature it sends a minimal request with the feature and the same
request without it, and compares the two payloads.

- **sent:** the payload moved. A fallback that renders the feature as text, such as instructions moved into a user
  message, also moves the payload and counts as sent.
- **raises:** the request raised before reaching the wire. Any exception counts, except a warning, which counts as
  dropped: which class an adapter should raise is https://github.com/pydantic/pydantic-ai/issues/14027 (F1).
- **dropped:** the payload didn't move and nothing was raised: the feature was silently ignored.

A feature with a profile flag is expected to be sent when the flag is on and to raise when it's off, so a class states
what it supports by setting the flag, the same way for every class. `SENT_DESPITE_FLAG` lists the exceptions to that
with their reasons. A feature without a flag is expected to be sent, and `RAISES_WITHOUT_FLAG` lists the ones that raise
with their reasons. Every `dropped` outcome needs its reason in `DROPPED_BECAUSE`: a documented filter, a deliberate
skip, or a tracked gap.

`ToolDefinition.strict` is left out: a provider that drops it is handled by retrying on the invalid arguments, so it
stays best-effort.

This is not a VCR test, and can't be one: a cassette is a frozen recording, so it keeps matching after the code stops
sending a feature, which is the silent drop this file exists to catch. Nothing is recorded either, since every probe
aborts before a response.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.tools import ToolDefinition

from .test_model_settings_support import CASES, PROBE_TOOL, Case, RequestShape

Outcome = Literal['sent', 'dropped', 'raises']

ProfileFlag = Literal[
    'supports_tools',
    'supports_json_schema_output',
    'supports_image_output',
    'supports_image_input',
    'supports_document_input',
    'supports_audio_file_input',
]

OUTPUT_SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']}

OUTPUT_TOOL = ToolDefinition(name='final_result', description='The final answer.', parameters_json_schema=OUTPUT_SCHEMA)

OUTPUT_OBJECT = OutputObjectDefinition(json_schema=OUTPUT_SCHEMA, name='probe_output', description='The answer.')

BARE = ModelRequestParameters()
"""No tools and text output: the parameters every feature is compared against, unless it changes them itself."""

PDF = BinaryContent(b'%PDF-1.4\n%probe\n', media_type='application/pdf', identifier='probe_pdf')
WAV = BinaryContent(b'RIFF\x24\x00\x00\x00WAVEfmt ', media_type='audio/wav', identifier='probe_wav')
PNG = BinaryContent(b'\x89PNG\r\n\x1a\nprobe', media_type='image/png', identifier='probe_png')


def _replayed_answer(*, with_thinking: bool) -> Callable[[Model], list[ModelMessage]]:
    """An earlier turn whose answer is replayed, optionally with a signed thinking part from the probed model itself."""

    def history(model: Model) -> list[ModelMessage]:
        parts: list[TextPart | ThinkingPart] = [TextPart('Probe answer.')]
        if with_thinking:
            # OpenRouter replays only a part carrying the reasoning detail type its own responses record.
            details = (
                {'type': 'reasoning.text', 'format': 'unknown', 'index': 0} if model.system == 'openrouter' else None
            )
            parts.insert(
                0,
                ThinkingPart(
                    'Probe thought.',
                    signature='cHJvYmUtc2lnbmF0dXJl',
                    provider_name=model.system,
                    provider_details=details,
                ),
            )
        return [
            ModelRequest.user_text_prompt('Earlier probe.'),
            ModelResponse(parts=parts, model_name=model.model_name, provider_name=model.system),
        ]

    return history


@dataclass(frozen=True)
class Feature:
    """One request feature, as the request that carries it and the same request without it."""

    id: str
    with_feature: RequestShape
    without: RequestShape = RequestShape(parameters=BARE)
    flag: ProfileFlag | None = None
    """The profile flag that says whether the model supports the feature, if there is one."""


FEATURES = [
    Feature(
        'function_tools',
        RequestShape(parameters=ModelRequestParameters(function_tools=[PROBE_TOOL])),
        flag='supports_tools',
    ),
    Feature(
        'output_tools',
        RequestShape(
            parameters=ModelRequestParameters(output_mode='tool', output_tools=[OUTPUT_TOOL], allow_text_output=False)
        ),
        flag='supports_tools',
    ),
    Feature(
        'native_output',
        RequestShape(
            parameters=ModelRequestParameters(
                output_mode='native', output_object=OUTPUT_OBJECT, allow_text_output=False
            )
        ),
        flag='supports_json_schema_output',
    ),
    Feature(
        'prompted_output',
        RequestShape(
            parameters=ModelRequestParameters(
                output_mode='prompted', output_object=OUTPUT_OBJECT, allow_text_output=False
            )
        ),
    ),
    Feature(
        'instruction_parts',
        RequestShape(parameters=ModelRequestParameters(instruction_parts=[InstructionPart('Probe instructions.')])),
    ),
    Feature(
        'allow_image_output',
        RequestShape(parameters=ModelRequestParameters(allow_image_output=True)),
        flag='supports_image_output',
    ),
    Feature('cache_point', RequestShape(('probe', CachePoint()), parameters=BARE)),
    Feature('image_input', RequestShape(('probe', PNG), parameters=BARE), flag='supports_image_input'),
    Feature('document_input', RequestShape(('probe', PDF), parameters=BARE), flag='supports_document_input'),
    Feature('audio_input', RequestShape(('probe', WAV), parameters=BARE), flag='supports_audio_file_input'),
    Feature(
        'replayed_signed_thinking',
        RequestShape(history=_replayed_answer(with_thinking=True), parameters=BARE),
        RequestShape(history=_replayed_answer(with_thinking=False), parameters=BARE),
    ),
]

FEATURE_IDS = {feature.id for feature in FEATURES}

DECISION_MODELS = frozenset({'TypeSafeModel', 'SystemOneModel', 'OpenAIDecisionsModel', 'GoogleDecisionsModel'})
"""Decision models judge a conversation through one output tool rather than answer it, so these features don't apply."""

FEATURE_CASES = [case for case in CASES if case.id not in DECISION_MODELS]

SENT_DESPITE_FLAG: dict[tuple[str, str], str] = {
    ('OpenAIChatModel', 'audio_input'): (
        "documented: a file in a user prompt is sent as it is, whatever the model's flag says (`supports_audio_file_input`)"
    ),
    ('SnowflakeModel', 'function_tools'): (
        'tracked gap: only `GoogleModel` checks `supports_tools` for function tools, '
        'https://github.com/pydantic/pydantic-ai/issues/14027 (F5)'
    ),
}
"""Why each feature that's sent although its profile flag is off is allowed to be sent."""

RAISES_WITHOUT_FLAG: dict[tuple[str, str], str] = {
    ('MCPSamplingModel', 'cache_point'): 'tracked gap: https://github.com/pydantic/pydantic-ai/issues/14857',
}
"""Why each feature without a profile flag that raises is allowed to raise."""

CHAT_IMAGE_OUTPUT_GAP = 'tracked gap: https://github.com/pydantic/pydantic-ai/issues/14790'
CACHE_POINT_HINT = 'documented: "Models that don\'t support caching will filter these out" (`CachePoint`)'
MCP_GAP = 'tracked gap: https://github.com/pydantic/pydantic-ai/issues/14774'

DROPPED_BECAUSE: dict[tuple[str, str], str] = {
    ('OpenAIChatModel', 'allow_image_output'): CHAT_IMAGE_OUTPUT_GAP,
    ('GitHubCopilotModel', 'allow_image_output'): CHAT_IMAGE_OUTPUT_GAP,
    ('OpenRouterModel', 'allow_image_output'): CHAT_IMAGE_OUTPUT_GAP,
    ('GoogleInteractionsModel', 'cache_point'): CACHE_POINT_HINT,
    ('GoogleModel', 'cache_point'): CACHE_POINT_HINT,
    ('XaiModel', 'cache_point'): CACHE_POINT_HINT,
    ('HuggingFaceModel', 'native_output'): 'tracked gap: https://github.com/pydantic/pydantic-ai/issues/14789',
    ('MCPSamplingModel', 'function_tools'): MCP_GAP,
    ('MCPSamplingModel', 'output_tools'): MCP_GAP,
    ('MCPSamplingModel', 'prompted_output'): f'{MCP_GAP} (rides on `instruction_parts`)',
    ('MCPSamplingModel', 'instruction_parts'): MCP_GAP,
    (
        'MCPSamplingModel',
        'replayed_signed_thinking',
    ): 'deliberate: MCP sampling messages have no thinking content (`_mcp.py`)',
}
"""Why each feature that's silently dropped is allowed to be: a documented filter, a deliberate skip, or a tracked gap."""


async def _outcome(case: Case, feature: Feature) -> tuple[Outcome, bool | None, str]:
    """The feature's outcome, the value of its profile flag if it has one, and the error message if it raised."""
    without = await case.probe({}, feature.without)
    assert without.payload is not None, f'{case.id} sent nothing for the request without {feature.id}'
    profile = without.observed.profile
    flag = None if feature.flag is None else bool(profile.get(feature.flag, DEFAULT_PROFILE.get(feature.flag)))
    with_feature = await case.probe({}, feature.with_feature)
    if with_feature.payload is None:
        error = with_feature.observed.error
        assert error is not None, f'{case.id} sent nothing for {feature.id} and raised nothing'
        # The suite turns warnings into errors, but outside it a warning lets the request go on without the feature.
        return ('dropped' if isinstance(error, Warning) else 'raises'), flag, f'{type(error).__name__}: {error}'
    return ('dropped' if with_feature.payload == without.payload else 'sent'), flag, ''


def _expected(case: Case, feature: Feature, flag: bool | None) -> str:
    """The outcome the profile and the exception tables call for, or why an exception entry is stale."""
    key = (case.id, feature.id)
    if key in DROPPED_BECAUSE:
        return 'dropped'
    if key in SENT_DESPITE_FLAG:
        return 'sent' if not flag else 'no `SENT_DESPITE_FLAG` entry, as the flag is on'
    if flag is None:
        return 'raises' if key in RAISES_WITHOUT_FLAG else 'sent'
    return 'sent' if flag else 'raises'


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in FEATURE_CASES])
async def test_request_features_match_the_profile(case: Case, allow_model_requests: None):
    """A flagged feature is sent when its flag is on and raises when it's off; an unflagged one is sent.

    Each exception table entry overrides that for its pair, and fails once it's no longer needed.
    """
    observed = {feature.id: (feature, *await _outcome(case, feature)) for feature in FEATURES}
    outcomes = {feature_id: outcome for feature_id, (_, outcome, _, _) in observed.items()}
    expected = {feature_id: _expected(case, feature, flag) for feature_id, (feature, _, flag, _) in observed.items()}
    assert outcomes == expected, f'{case.id}:\n' + '\n'.join(
        f'{feature_id} (`{feature.flag}={flag}`): expected {expected[feature_id]}, got {outcome} {message}'
        for feature_id, (feature, outcome, flag, message) in observed.items()
        if outcome != expected[feature_id]
    )


def test_exceptions_name_known_cases_and_features():
    case_ids = {case.id for case in FEATURE_CASES}
    flagged = {feature.id for feature in FEATURES if feature.flag is not None}
    for table in (DROPPED_BECAUSE, SENT_DESPITE_FLAG, RAISES_WITHOUT_FLAG):
        assert {case for case, _ in table} <= case_ids
        assert {feature for _, feature in table} <= FEATURE_IDS
    assert {feature for _, feature in SENT_DESPITE_FLAG} <= flagged
    assert not {feature for _, feature in RAISES_WITHOUT_FLAG} & flagged
