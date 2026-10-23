"""The unified `thinking_summary` setting against recorded provider responses.

What comes back is the claim — readable thinking, empty signed thinking, or none at all — so these record real
requests. The outbound field is read off a request hook on the provider's HTTP client: the shared `httpx2` one, or for
Groq, whose provider only takes a legacy `httpx` client, that client.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.settings import ModelSettings

from ._inline_snapshot import snapshot
from .conftest import RequestCapture, try_import

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as groq_imports:
    import httpx

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.vcr

PROMPT = (
    'How many integers from 1 to 1000 inclusive are divisible by 3 or 5 but not by 7? '
    'Work it out carefully, then answer with just the number.'
)


def _thinking_parts(messages: list[ModelMessage]) -> list[ThinkingPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ThinkingPart)
    ]


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
async def test_anthropic_thinking_returns_summaries_by_default(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """Claude Opus 4.7 defaults to `display: 'omitted'`, so unified `thinking` alone asks for summaries.

    Without them the response holds signed thinking blocks with no text, which is the issue this setting fixes.
    """
    model = AnthropicModel(
        'claude-opus-4-7', provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client)
    )
    result = await Agent(model, model_settings=ModelSettings(thinking='high')).run(PROMPT)

    assert request_capture.body('/v1/messages')['thinking'] == snapshot({'type': 'adaptive', 'display': 'summarized'})
    thinking = _thinking_parts(result.all_messages())
    assert thinking and all(part.content and part.signature for part in thinking)


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
@pytest.mark.parametrize('thinking_summary', [True, False])
async def test_anthropic_thinking_summary_on_a_model_that_thinks_by_default(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture, thinking_summary: bool
):
    """Claude Opus 5.5 thinks without a `thinking` parameter, so `thinking_summary` sends the adaptive block it uses.

    `True` returns readable summaries; `False` returns the same signed blocks with no text, since the model still
    thinks.
    """
    model = AnthropicModel(
        'claude-opus-5-5', provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client)
    )
    result = await Agent(model, model_settings=ModelSettings(thinking_summary=thinking_summary)).run(PROMPT)

    assert request_capture.body('/v1/messages')['thinking'] == {
        'type': 'adaptive',
        'display': 'summarized' if thinking_summary else 'omitted',
    }
    thinking = _thinking_parts(result.all_messages())
    assert thinking and all(part.signature for part in thinking)
    assert all(bool(part.content) is thinking_summary for part in thinking)


@pytest.mark.skipif(not google_imports(), reason='google-genai not installed')
async def test_google_thinking_summary_on_a_model_that_thinks_by_default(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
):
    """Gemini thinks without a thinking config but leaves the thoughts out, so `True` alone asks for them."""
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=60))
    model = GoogleModel('gemini-3.8-flash', provider=provider)
    result = await Agent(model, model_settings=ModelSettings(thinking_summary=True)).run(PROMPT)

    generation_config = request_capture.body(':generateContent')['generationConfig']
    assert isinstance(generation_config, dict)
    assert generation_config['thinkingConfig'] == snapshot({'include_thoughts': True})
    thinking = _thinking_parts(result.all_messages())
    assert thinking and all(part.content for part in thinking)


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_responses_thinking_summary(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    """OpenAI's Responses API only returns a reasoning summary when asked, and `True` asks with `auto`."""
    model = OpenAIResponsesModel(
        'gpt-6.1-sol', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    result = await Agent(model, model_settings=ModelSettings(thinking='high', thinking_summary=True)).run(PROMPT)

    assert request_capture.body('/responses')['reasoning'] == snapshot(
        {'effort': 'high', 'context': 'all_turns', 'summary': 'auto'}
    )
    thinking = _thinking_parts(result.all_messages())
    assert thinking and any(part.content for part in thinking)


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
async def test_groq_thinking_summary_false_hides_reasoning(allow_model_requests: None, groq_api_key: str):
    """gpt-oss on Groq returns its reasoning by default; `False` asks for `reasoning_format='hidden'`, so none comes back.

    `GroqProvider` takes a legacy `httpx` client rather than the shared `httpx2` one, so the outbound body is read
    off that client's request hook, which also fires on replay.
    """
    sent: list[dict[str, Any]] = []

    async def record(request: httpx.Request) -> None:
        sent.append(json.loads(request.content))

    async with httpx.AsyncClient(event_hooks={'request': [record]}) as http_client:
        model = GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key=groq_api_key, http_client=http_client))
        result = await Agent(model, model_settings=ModelSettings(thinking_summary=False)).run(PROMPT)

    assert [body.get('reasoning_format') for body in sent] == snapshot(['hidden'])
    assert _thinking_parts(result.all_messages()) == []
