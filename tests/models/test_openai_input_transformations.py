"""Reasoning items the OpenAI Responses API withheld from the input, from `Response.input_transformations`.

On newer models, `gpt-6.2-sol` among them, the API drops a replayed reasoning item whose earlier
conversation changed, and lists it in `input_transformations`. Changing the output format between
runs is such a change: https://developers.openai.com/api/docs/guides/keep-prefix-unchanged lists
`text.format`. The recordings use `openai_store=False`.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from opentelemetry import trace
from pydantic import BaseModel

from pydantic_ai import Agent, ModelResponse, NativeOutput, ThinkingPart
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

from ..conftest import message, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed'),
    pytest.mark.vcr,
    pytest.mark.moves_cache_prefix(reason='the second run changes the output format, which is the edit under test'),
]


class Answer(BaseModel):
    total: int


_QUESTION = 'What is the smallest prime number strictly greater than 7741? Reason it through, then reply with it.'
_FOLLOW_UP = 'Add 1000 to that prime.'


def _agent(openai_api_key: str, *, instrument: bool) -> Agent[None, str]:
    return Agent(
        OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(api_key=openai_api_key)),
        model_settings=OpenAIResponsesModelSettings(thinking='medium', openai_store=False),
        capabilities=[Instrumentation(settings=InstrumentationSettings())] if instrument else [],
    )


def _span_events(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    return [
        event
        for span in capfire.exporter.exported_spans_as_dict()
        for event in span.get('events', [])
        if event['name'] == 'openai.input_transformations'
    ]


@pytest.mark.parametrize('stream', [False, True])
async def test_openai_responses_reports_withheld_reasoning(
    allow_model_requests: None, openai_api_key: str, capfire: CaptureLogfire, stream: bool
):
    """The request after the format change reports the first run's reasoning item as removed.

    The report goes in the response's `provider_details` and on its model request span. A response
    whose `input_transformations` is empty has no `input_transformations` key.
    """
    agent = _agent(openai_api_key, instrument=True)
    first = await agent.run(_QUESTION)
    if stream:
        async with agent.run_stream(
            _FOLLOW_UP, message_history=first.all_messages(), output_type=NativeOutput(Answer)
        ) as streamed:
            await streamed.get_output()
        messages = streamed.all_messages()
    else:
        messages = (
            await agent.run(_FOLLOW_UP, message_history=first.all_messages(), output_type=NativeOutput(Answer))
        ).all_messages()

    first_response = message(first.all_messages(), ModelResponse, index=-1)
    reasoning_id = next(part.id for part in first_response.parts if isinstance(part, ThinkingPart))
    assert 'input_transformations' not in (first_response.provider_details or {})
    second_response = message(messages, ModelResponse, index=-1)
    assert second_response.provider_details is not None
    withheld = [{'id': reasoning_id, 'action': 'removed', 'reason': 'prefix_mismatch', 'type': 'reasoning'}]
    assert second_response.provider_details['input_transformations'] == withheld
    assert [json.loads(event['attributes']['openai.input_transformations']) for event in _span_events(capfire)] == [
        withheld
    ]


@pytest.mark.vcr('test_openai_responses_reports_withheld_reasoning[False].yaml')
async def test_openai_responses_withheld_reasoning_skips_an_ambient_span(
    allow_model_requests: None, openai_api_key: str, capfire: CaptureLogfire
):
    """Without instrumentation, the report stays in `provider_details` and never reaches a caller's span.

    Replays the non-streamed recording above: the requests are the same, only the instrumentation differs.
    """
    agent = _agent(openai_api_key, instrument=False)
    with trace.get_tracer(__name__).start_as_current_span('ambient'):
        first = await agent.run(_QUESTION)
        second = await agent.run(_FOLLOW_UP, message_history=first.all_messages(), output_type=NativeOutput(Answer))

    second_response = message(second.all_messages(), ModelResponse, index=-1)
    assert second_response.provider_details is not None
    assert [entry['reason'] for entry in second_response.provider_details['input_transformations']] == [
        'prefix_mismatch'
    ]
    assert _span_events(capfire) == []
