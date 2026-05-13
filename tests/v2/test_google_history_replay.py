"""Cross-history-replay coverage for the Google provider rename.

`GoogleProvider.name` returns `'google'` / `'google-cloud'` post-rename, but historical
`ModelMessage` records may have `provider_name='google-gla'` / `'google-vertex'`. The Google
model class accepts both via the `GEMINI_API_SYSTEMS` / `GOOGLE_CLOUD_SYSTEMS` sets, so
replay still routes thinking signatures and built-in tool parts correctly. These tests pin
that contract.

TODO: generalize this into a cross-module history-replay test suite (V2-RULES rule 21).
"""

from __future__ import annotations as _annotations

import base64

import pytest

from pydantic_ai import ModelResponse, NativeToolCallPart, NativeToolReturnPart, TextPart, ThinkingPart
from pydantic_ai.native_tools import CodeExecutionTool

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel, _content_model_response  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


@pytest.fixture(autouse=True)
def _set_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')


def _gemini_api_model() -> GoogleModel:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider())


def _google_cloud_model() -> GoogleModel:
    return GoogleModel('gemini-2.5-flash', provider=GoogleCloudProvider(project='p', location='us-central1'))


@pytest.mark.parametrize(
    ('model_factory', 'historical_provider_name', 'current_provider_name'),
    [
        (_gemini_api_model, 'google-gla', 'google'),
        (_google_cloud_model, 'google-vertex', 'google-cloud'),
    ],
)
def test_history_with_legacy_provider_name_still_routes_thinking_signature(
    model_factory: type, historical_provider_name: str, current_provider_name: str
) -> None:
    """A `ThinkingPart` captured against the old provider name still has its signature replayed."""
    model = model_factory()
    assert model.system == current_provider_name

    signature = base64.b64encode(b'sig').decode('ascii')
    response = ModelResponse(
        parts=[
            ThinkingPart(content='reasoning', provider_name=historical_provider_name, signature=signature),
            TextPart(content='final'),
        ],
        provider_name=historical_provider_name,
    )

    accepted = model._matching_provider_names
    assert historical_provider_name in accepted
    assert current_provider_name in accepted

    content = _content_model_response(response, accepted)
    assert content is not None
    parts = content.get('parts') or []
    text_part = next(p for p in parts if p.get('text') == 'final')
    assert 'thought_signature' in text_part


@pytest.mark.parametrize(
    ('model_factory', 'historical_provider_name', 'current_provider_name'),
    [
        (_gemini_api_model, 'google-gla', 'google'),
        (_google_cloud_model, 'google-vertex', 'google-cloud'),
    ],
)
def test_history_with_legacy_provider_name_still_replays_builtin_tool_parts(
    model_factory: type, historical_provider_name: str, current_provider_name: str
) -> None:
    """A `NativeToolCallPart` / `NativeToolReturnPart` carrying the old name still round-trips."""
    model = model_factory()
    assert model.system == current_provider_name

    response = ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name=CodeExecutionTool.kind,
                args={'code': "print('hi')"},
                provider_name=historical_provider_name,
            ),
            NativeToolReturnPart(
                tool_name=CodeExecutionTool.kind,
                content={'output': 'hi\n', 'outcome': 'OUTCOME_OK'},
                provider_name=historical_provider_name,
            ),
        ],
        provider_name=historical_provider_name,
    )

    accepted = model._matching_provider_names
    content = _content_model_response(response, accepted)
    assert content is not None
    parts = content.get('parts') or []
    assert any('executable_code' in p for p in parts)
    assert any('code_execution_result' in p for p in parts)
