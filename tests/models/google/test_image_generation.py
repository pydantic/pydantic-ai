"""Tests for `ImageGenerationTool` support on `GoogleModel`.

Only Gemini image models generate images natively; on a text model an `ImageGeneration` capability runs
its local fallback instead.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.profiles import ModelProfile

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.vcr,
]


async def test_google_image_generation_text_model_runs_local_fallback(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
):
    """A Gemini text model has no native image generation, so `ImageGeneration` sends its local tool instead."""
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=30))
    prompts: list[str] = []

    def generate_image(prompt: str) -> str:
        """Generate an image from a text prompt."""
        prompts.append(prompt)
        return 'The image was generated and shown to the user.'

    agent = Agent(
        GoogleModel('gemini-2.5-flash', provider=provider), capabilities=[ImageGeneration(local=generate_image)]
    )
    await agent.run('Generate an image of an axolotl.')

    assert prompts == snapshot(['axolotl'])
    body = request_capture.body(':generateContent')
    assert body['tools'] == snapshot(
        [
            {
                'functionDeclarations': [
                    {
                        'description': 'Generate an image from a text prompt.',
                        'name': 'generate_image',
                        'parameters_json_schema': {
                            'additionalProperties': False,
                            'properties': {'prompt': {'type': 'string'}},
                            'required': ['prompt'],
                            'type': 'object',
                        },
                    }
                ]
            }
        ]
    )
    assert body['generationConfig'] == snapshot({'responseModalities': ['TEXT']})


@pytest.mark.parametrize(
    ('model_name', 'profile', 'supports_native'),
    [
        ('gemini-2.5-flash', None, False),
        ('gemini-3.1-flash-image', None, True),
        ('gemini-2.5-flash', ModelProfile(supports_image_output=True), True),
    ],
)
def test_google_image_generation_tool_follows_supports_image_output(
    gemini_api_key: str, model_name: str, profile: ModelProfile | None, supports_native: bool
):
    """`ImageGenerationTool` is supported exactly when the resolved `supports_image_output` is true.

    Pinned on the profile because the flag is resolved there, including a user `profile=` override. The
    request paths are recorded in `test_google_image_generation_text_model_runs_local_fallback` (flag off)
    and `tests/models/test_google.py::test_google_image_or_text_output` (flag on).
    """
    model = GoogleModel(model_name, provider=GoogleProvider(api_key=gemini_api_key), profile=profile)
    assert (ImageGenerationTool in model.profile.get('supported_native_tools', frozenset())) is supports_native


def test_google_optional_image_generation_tool_dropped_on_text_model(gemini_api_key: str):
    """An optional `ImageGenerationTool` on a text model is dropped instead of raising.

    Not a VCR test: the tool is resolved in `prepare_request`, before any request is built.
    """
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
    _, params = model.prepare_request(None, ModelRequestParameters(native_tools=[ImageGenerationTool(optional=True)]))
    assert params.native_tools == []
