"""Tests for the `openai_safety_identifier` setting on the Chat Completions and Responses APIs."""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent
from pydantic_ai.models import Model

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[OpenAIProvider], Model]
    path: str
    settings: OpenAIChatModelSettings
    expected_output: str
    expected_bodies: list[dict[str, str]]


CASES = [
    Case(
        id='chat',
        model=lambda provider: OpenAIChatModel('gpt-4o', provider=provider),
        path='/chat/completions',
        settings={'openai_user': 'user_id', 'openai_safety_identifier': 'safety_id'},
        expected_output=snapshot('2 + 2 equals 4.'),
        expected_bodies=snapshot([{'safety_identifier': 'safety_id', 'user': 'user_id'}, {}]),
    ),
    Case(
        id='responses',
        model=lambda provider: OpenAIResponsesModel('gpt-4o', provider=provider),
        path='/responses',
        settings={'openai_safety_identifier': 'safety_id'},
        expected_output=snapshot('2 + 2 equals 4.'),
        expected_bodies=snapshot([{'safety_identifier': 'safety_id'}, {}]),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_openai_safety_identifier(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture, case: Case
):
    """`openai_safety_identifier` goes out as `safety_identifier`, and only when set.

    The Chat case sends `user` and `safety_identifier` together, and OpenAI accepts them in one request.
    """
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    agent = Agent(case.model(provider))

    result = await agent.run('What is 2+2?', model_settings=case.settings)
    assert result.output == case.expected_output
    await agent.run('What is 2+2?')

    assert [
        {key: value for key, value in body.items() if key in ('user', 'safety_identifier')}
        for body in request_capture.bodies(case.path)
    ] == case.expected_bodies
