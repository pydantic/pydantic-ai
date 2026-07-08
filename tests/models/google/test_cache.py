"""Tests for `google_cached_content` context caching."""

from __future__ import annotations as _annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import (
        Candidate,
        Content,
        CreateCachedContentConfig,
        FinishReason as GoogleFinishReason,
        GenerateContentResponse,
        Part,
    )

    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

if TYPE_CHECKING:
    GoogleModelFactory = Callable[..., GoogleModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_google_model_cached_content(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
):
    """End-to-end contract for `google_cached_content`: the cache resource
    owns `system_instruction`, `tools`, and `tool_config`, and both Gemini
    and Vertex return `400 INVALID_ARGUMENT` if those fields are sent
    alongside `cached_content`. Pydantic AI therefore strips them from the
    outgoing request, emitting a `UserWarning` whenever stripping actually
    drops a populated field.

    One cassette covers both branches: first run carries instructions + a
    registered tool (warning fires, request still succeeds, response shows a
    cache hit); second run is minimal (nothing to strip, no warning — the
    suite's `filterwarnings = ['error']` setting turns any stray warning
    into a failure). The shared `_build_content_and_config` helper means a
    non-streaming test covers the streaming path too.

    See issue #5671.
    """
    model_name = 'gemini-2.5-flash'
    long_text = 'Paris is the capital of France. The Eiffel Tower is in Paris. ' * 250

    model = google_model(model_name)
    cache = await model.client.aio.caches.create(
        model=model_name,
        config=CreateCachedContentConfig(
            system_instruction='You are a geography expert. Be concise.',
            contents=[Content(role='user', parts=[Part(text=long_text)])],
            ttl='120s',
        ),
    )
    cache_name = cache.name
    assert cache_name is not None
    try:
        settings = GoogleModelSettings(google_cached_content=cache_name)

        agent_with_extras = Agent(
            model=model,
            instructions='These instructions get stripped — the cache owns the system_instruction.',
            model_settings=settings,
        )

        @agent_with_extras.tool_plain
        def unused_tool(x: str) -> str:
            return x  # pragma: no cover

        with pytest.warns(UserWarning, match='`google_cached_content` is set'):
            result = await agent_with_extras.run('What is the capital of France?')

        assert 'Paris' in result.output
        assert (result.usage.details or {}).get('cached_content_tokens', 0) > 0

        agent_minimal = Agent(model=model, model_settings=settings)
        result_minimal = await agent_minimal.run('Say the capital one more time.')
        assert 'Paris' in result_minimal.output
    finally:
        await model.client.aio.caches.delete(name=cache_name)


async def test_google_model_cached_content_prompted_output_enables_json_mode(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
    mocker: MockerFixture,
):
    """`prompted` output mode normally only switches the request to JSON mode when no
    tools are registered (the model has to dedicate its output to JSON instead of
    tool calls). When `google_cached_content` strips the tools, the post-strip request
    *is* tool-less, so JSON mode should kick in — otherwise the agent gets free-form
    text back and the prompted-JSON parser fails downstream with no clear link to the
    cache. Regression test for the interaction flagged on #5681.
    """
    cache_name = 'projects/p/locations/global/cachedContents/test-cache'
    model = google_model('gemini-2.5-pro')

    class CityLocation(BaseModel):
        city: str
        country: str

    chunk = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[Part(text='{"city": "Paris", "country": "France"}')], role='model'),
                finish_reason=GoogleFinishReason.STOP,
            )
        ],
        response_id='cached',
        model_version='gemini-2.5-pro',
    )
    mock = mocker.patch.object(model.client.aio.models, 'generate_content', return_value=chunk)

    agent = Agent(
        model=model,
        output_type=PromptedOutput(CityLocation),
        model_settings=GoogleModelSettings(google_cached_content=cache_name),
    )

    @agent.tool_plain
    def unused_tool(x: str) -> str:
        return x  # pragma: no cover

    with pytest.warns(UserWarning, match='`google_cached_content` is set'):
        await agent.run('Where is the Eiffel Tower?')

    assert mock.call_count == 1
    _, kwargs = mock.call_args
    config = kwargs['config']
    assert not config.get('tools')
    assert config['response_mime_type'] == 'application/json'
