"""Pin what each model class does with each request feature: send it, or raise.

`models/AGENTS.md` asks a model class to raise on a request feature it can't form, never to drop it silently. This
module checks that against the wire, with the probes and cases of `test_model_settings_support.py`: for each feature it
sends a minimal request with the feature and the same request without it, and compares the two payloads.

- **sent:** the payload moved. A fallback that renders the feature as text, such as instructions moved into a user
  message, also moves the payload and counts as sent.
- **raises:** the request raised a `UserError` before reaching the wire.
- **dropped:** the payload didn't move and nothing was raised: the feature was silently ignored.

`EXPECTED` lists every pair whose outcome isn't `sent`, so a class that starts dropping a feature, or stops, fails the
test until the table is updated. Every `dropped` entry there has its reason in `DROPPED_BECAUSE`: a documented filter, a deliberate skip, or a
known gap.

`ToolDefinition.strict` is left out: a provider that drops it is handled by retrying on the invalid arguments, so it
stays best-effort.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
from pydantic_ai.tools import ToolDefinition

from .test_model_settings_support import CASES, PROBE_TOOL, Case, RequestShape

Outcome = str
"""`'sent'`, `'dropped'`, or `'raises <exception class>'`."""

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


FEATURES = [
    Feature('function_tools', RequestShape(parameters=ModelRequestParameters(function_tools=[PROBE_TOOL]))),
    Feature(
        'output_tools',
        RequestShape(
            parameters=ModelRequestParameters(output_mode='tool', output_tools=[OUTPUT_TOOL], allow_text_output=False)
        ),
    ),
    Feature(
        'native_output',
        RequestShape(
            parameters=ModelRequestParameters(
                output_mode='native', output_object=OUTPUT_OBJECT, allow_text_output=False
            )
        ),
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
    Feature('allow_image_output', RequestShape(parameters=ModelRequestParameters(allow_image_output=True))),
    Feature('cache_point', RequestShape(('probe', CachePoint()), parameters=BARE)),
    Feature('image_input', RequestShape(('probe', PNG), parameters=BARE)),
    Feature('document_input', RequestShape(('probe', PDF), parameters=BARE)),
    Feature('audio_input', RequestShape(('probe', WAV), parameters=BARE)),
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

EXPECTED: dict[str, dict[str, Outcome]] = {
    'OpenAIChatModel': {
        'allow_image_output': 'dropped',
    },
    'OpenAIResponsesModel': {
        'audio_input': 'raises NotImplementedError',
    },
    'OpenAICodexModel': {
        'audio_input': 'raises NotImplementedError',
    },
    'OpenAIResponsesModel (ChatGPT plan usage)': {
        'audio_input': 'raises NotImplementedError',
    },
    'CerebrasModel': {
        'native_output': 'raises UserError',
        'allow_image_output': 'raises UserError',
    },
    'CrusoeModel': {
        'allow_image_output': 'raises UserError',
    },
    'GitHubCopilotModel': {
        'allow_image_output': 'dropped',
    },
    'OllamaModel': {
        'allow_image_output': 'raises UserError',
    },
    'OpenRouterModel': {
        'allow_image_output': 'dropped',
    },
    'SnowflakeModel': {
        'output_tools': 'raises UserError',
        'native_output': 'raises UserError',
        'allow_image_output': 'raises UserError',
    },
    'ZaiModel': {
        'native_output': 'raises UserError',
        'allow_image_output': 'raises UserError',
    },
    'BedrockMantleChatModel': {
        'allow_image_output': 'raises UserError',
    },
    'BedrockMantleResponsesModel': {
        'allow_image_output': 'raises UserError',
        'audio_input': 'raises NotImplementedError',
    },
    'AnthropicModel': {
        'allow_image_output': 'raises UserError',
        'audio_input': 'raises NotImplementedError',
    },
    'GroqModel': {
        'native_output': 'raises UserError',
        'allow_image_output': 'raises UserError',
        'document_input': 'raises NotImplementedError',
        'audio_input': 'raises NotImplementedError',
    },
    'MistralModel': {
        'native_output': 'raises UserError',
        'allow_image_output': 'raises UserError',
        'audio_input': 'raises NotImplementedError',
    },
    'CohereModel': {
        'native_output': 'raises UserError',
        'allow_image_output': 'raises UserError',
        'image_input': 'raises RuntimeError',
        'document_input': 'raises RuntimeError',
        'audio_input': 'raises RuntimeError',
    },
    'GoogleInteractionsModel': {
        'allow_image_output': 'raises UserError',
        'cache_point': 'dropped',
    },
    'GoogleModel': {
        'allow_image_output': 'raises UserError',
        'cache_point': 'dropped',
    },
    'BedrockConverseModel': {
        'allow_image_output': 'raises UserError',
        'audio_input': 'raises NotImplementedError',
    },
    'HuggingFaceModel': {
        'native_output': 'dropped',
        'allow_image_output': 'raises UserError',
        'document_input': 'raises RuntimeError',
        'audio_input': 'raises RuntimeError',
    },
    'XaiModel': {
        'allow_image_output': 'raises UserError',
        'cache_point': 'dropped',
        'audio_input': 'raises NotImplementedError',
    },
    'MCPSamplingModel': {
        'function_tools': 'dropped',
        'output_tools': 'dropped',
        'native_output': 'raises UserError',
        'prompted_output': 'dropped',
        'instruction_parts': 'dropped',
        'allow_image_output': 'raises UserError',
        'cache_point': 'raises NotImplementedError',
        'document_input': 'raises NotImplementedError',
        'audio_input': 'raises NotImplementedError',
        'replayed_signed_thinking': 'dropped',
    },
}
"""Per case, every feature whose outcome isn't `sent`. Every `dropped` entry has its reason in `DROPPED_BECAUSE`."""

CHAT_IMAGE_OUTPUT_GAP = 'known gap: the profile allows image output for the Responses image tool, and Chat neither asks for nor parses images'
CACHE_POINT_HINT = 'documented: "Models that don\'t support caching will filter these out" (`CachePoint`)'

DROPPED_BECAUSE: dict[tuple[str, str], str] = {
    ('OpenAIChatModel', 'allow_image_output'): CHAT_IMAGE_OUTPUT_GAP,
    ('GitHubCopilotModel', 'allow_image_output'): CHAT_IMAGE_OUTPUT_GAP,
    ('OpenRouterModel', 'allow_image_output'): CHAT_IMAGE_OUTPUT_GAP,
    ('GoogleInteractionsModel', 'cache_point'): CACHE_POINT_HINT,
    ('GoogleModel', 'cache_point'): CACHE_POINT_HINT,
    ('XaiModel', 'cache_point'): CACHE_POINT_HINT,
    (
        'HuggingFaceModel',
        'native_output',
    ): 'known gap: the Qwen 3.5 profile allows native output, which the adapter never sends',
    ('MCPSamplingModel', 'function_tools'): 'known gap: https://github.com/pydantic/pydantic-ai/issues/14774',
    ('MCPSamplingModel', 'output_tools'): 'known gap: https://github.com/pydantic/pydantic-ai/issues/14774',
    (
        'MCPSamplingModel',
        'prompted_output',
    ): 'known gap: rides on `instruction_parts`, https://github.com/pydantic/pydantic-ai/issues/14774',
    ('MCPSamplingModel', 'instruction_parts'): 'known gap: https://github.com/pydantic/pydantic-ai/issues/14774',
    (
        'MCPSamplingModel',
        'replayed_signed_thinking',
    ): 'deliberate: MCP sampling messages have no thinking content (`_mcp.py`)',
}
"""Why each `dropped` entry in `EXPECTED` is allowed: a documented filter, a deliberate skip, or a known gap."""


async def _outcome(case: Case, feature: Feature) -> tuple[Outcome, str]:
    without = await case.probe({}, feature.without)
    assert without.payload is not None, f'{case.id} sent nothing for the request without {feature.id}'
    with_feature = await case.probe({}, feature.with_feature)
    if with_feature.payload is None:
        error = with_feature.observed.error
        assert error is not None, f'{case.id} sent nothing for {feature.id} and raised nothing'
        return f'raises {type(error).__name__}', str(error)
    return ('dropped' if with_feature.payload == without.payload else 'sent'), ''


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in FEATURE_CASES])
async def test_request_features_are_sent_or_raise(case: Case, allow_model_requests: None):
    """Each request feature moves the payload or raises a `UserError`; any other outcome is listed in `EXPECTED`."""
    observed = {feature.id: await _outcome(case, feature) for feature in FEATURES}
    not_sent = {feature: outcome for feature, (outcome, _) in observed.items() if outcome != 'sent'}
    assert not_sent == EXPECTED.get(case.id, {}), '\n'.join(
        f'{case.id} {feature}: {outcome} {message}' for feature, (outcome, message) in observed.items()
    )


def test_expected_names_known_cases_and_features():
    assert set(EXPECTED) <= {case.id for case in FEATURE_CASES}
    assert {feature for outcomes in EXPECTED.values() for feature in outcomes} <= FEATURE_IDS


def test_every_drop_has_a_reason():
    """A silent drop is only allowed with a reason: a documented filter, a deliberate skip, or a tracked gap."""
    drops = {
        (case, feature) for case, outcomes in EXPECTED.items() for feature, o in outcomes.items() if o == 'dropped'
    }
    assert drops == set(DROPPED_BECAUSE)
