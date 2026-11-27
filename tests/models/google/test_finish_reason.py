"""Tests for how `GoogleModel` maps the finish reason Gemini ends a response with onto `ModelResponse.finish_reason`."""

from __future__ import annotations as _annotations

import json

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import FinishReason, ModelResponse

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


def _model(finish_reason: str, *parts: dict[str, str]) -> GoogleModel:
    """A model whose requests are answered with one response that has `parts` and ends with `finish_reason`."""
    body = {
        'candidates': [{'content': {'role': 'model', 'parts': list(parts)}, 'finishReason': finish_reason, 'index': 0}],
        'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 1},
        'modelVersion': 'gemini-2.5-flash',
        'responseId': 'response-1',
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if 'streamGenerateContent' in request.url.path:
            content = f'data: {json.dumps(body)}\r\n\r\n'.encode()
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=content)
        return httpx2.Response(200, json=body)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test', http_client=client))


async def _run(agent: Agent[None, str], stream: bool) -> tuple[str, ModelResponse]:
    """Run `agent`, streamed or not, returning its output and its last response."""
    if stream:
        async with agent.run_stream('Hello') as result:
            output = await result.get_output()
        return output, result.response
    run_result = await agent.run('Hello')
    return run_result.output, run_result.response


# google-genai lists these in `FinishReason` from 3.0 or 3.3; an earlier release warns as it parses one.
_NOT_IN_OLDER_SDK = pytest.mark.filterwarnings('ignore:.* is not a valid FinishReason')


@pytest.mark.parametrize(
    ('raw_finish_reason', 'finish_reason'),
    [
        ('TOO_MANY_TOOL_CALLS', 'error'),
        ('IMAGE_RECITATION', 'content_filter'),
        ('IMAGE_OTHER', None),
        pytest.param('MALFORMED_RESPONSE', 'error', marks=_NOT_IN_OLDER_SDK),
        pytest.param('MISSING_THOUGHT_SIGNATURE', 'error', marks=_NOT_IN_OLDER_SDK),
        pytest.param('ESCALATION', 'content_filter', marks=_NOT_IN_OLDER_SDK),
        pytest.param('PUP_LIMITED_DISABLED', 'content_filter', marks=_NOT_IN_OLDER_SDK),
    ],
)
@pytest.mark.parametrize('stream', [False, True])
async def test_finish_reason(
    raw_finish_reason: str, finish_reason: FinishReason | None, stream: bool, allow_model_requests: None
):
    """A response that ends with `raw_finish_reason` reports `finish_reason`, and keeps the raw value.

    `IMAGE_OTHER` is a catch-all with no counterpart, so it maps to `None`, as `OTHER` does.

    Not a VCR test: a live request can't be made to end with a given finish reason.
    """
    output, response = await _run(Agent(_model(raw_finish_reason, {'text': 'Hi'})), stream)

    assert output == 'Hi'
    assert response.finish_reason == finish_reason
    assert response.provider_details == {'finish_reason': raw_finish_reason}


@pytest.mark.parametrize('stream', [False, True])
async def test_empty_image_recitation_raises_content_filter_error(stream: bool, allow_model_requests: None):
    """An empty response that ends with `IMAGE_RECITATION` is refused content, not an empty response to retry.

    Not a VCR test: a live request can't be made to end with a given finish reason.
    """
    with pytest.raises(ContentFilterError, match="Finish reason: 'IMAGE_RECITATION'"):
        await _run(Agent(_model('IMAGE_RECITATION')), stream)
