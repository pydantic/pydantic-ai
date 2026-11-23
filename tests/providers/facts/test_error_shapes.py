"""A provider's recorded error body raises the error category its hand-written mapping names (proposal R32).

Each case records a request the provider rejects before running a model, so a re-recording shows whether the provider
still sends that shape. `tests/models/test_error_categories.py` checks the same mappings against fixed wire shapes.
"""

from __future__ import annotations as _annotations

import json
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.settings import ModelSettings

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

pytestmark = [
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='mistral not installed'),
]


@dataclass(frozen=True)
class Case:
    id: str
    model_name: str
    prompt: str
    settings: ModelSettings
    expected: dict[str, object]
    """The error's category, the provider error fields it carries, and the error object the provider sent."""


CASES = [
    Case(
        # `voxtral-small-latest` has the smallest context window of Mistral's chat models, 32768 tokens, and each
        # digit is one token, so the prompt is just over it.
        id='mistral-prompt-too-long',
        model_name='voxtral-small-latest',
        prompt='0' * 33_000,
        settings=ModelSettings(),
        expected=snapshot(
            {
                'category': 'ModelContextWindowExceededError',
                'status_code': 400,
                'provider_error_code': '3059',
                'provider_error_type': 'invalid_request_prompt_too_long',
                'body': {
                    'object': 'error',
                    'message': 'Prompt 33003 > 32768 maximum context length',
                    'type': 'invalid_request_prompt_too_long',
                    'param': None,
                    'code': '3059',
                    'raw_status_code': 400,
                },
            }
        ),
    ),
    Case(
        # `max_tokens=0` asks for a prefill-only request, which Mistral rejects as an invalid argument, with the
        # generic code `3051`.
        id='mistral-invalid-args',
        model_name='mistral-small-latest',
        prompt='hello',
        settings=ModelSettings(max_tokens=0),
        expected=snapshot(
            {
                'category': 'ModelRequestRejectedError',
                'status_code': 400,
                'provider_error_code': '3051',
                'provider_error_type': 'invalid_request_invalid_args',
                'body': {
                    'object': 'error',
                    'message': 'prompt_logprobs must be True when max_tokens=0 (prefill-only request), got '
                    'request.prompt_logprobs=None',
                    'type': 'invalid_request_invalid_args',
                    'param': None,
                    'code': '3051',
                    'raw_status_code': 400,
                },
            }
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_recorded_error_shape(allow_model_requests: None, mistral_api_key: str, case: Case):
    model = MistralModel(case.model_name, provider=MistralProvider(api_key=mistral_api_key))

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run(case.prompt, model_settings=case.settings)

    exc = exc_info.value
    assert isinstance(exc.body, str)
    assert {
        'category': type(exc).__name__,
        'status_code': exc.status_code,
        'provider_error_code': exc.provider_error_code,
        'provider_error_type': exc.provider_error_type,
        'body': json.loads(exc.body),
    } == case.expected
