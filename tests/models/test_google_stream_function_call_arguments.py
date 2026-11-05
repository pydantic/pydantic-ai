"""Tests for Vertex streamed function-call arguments (`google_stream_function_call_arguments`)."""

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai.messages import (
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from google import genai
    from google.genai.types import GenerateContentResponse

    from pydantic_ai.models.google import GoogleModel, GoogleModelProfile, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.vcr,
]


class ModelRequestParametersStub:
    """Minimal parameters describing a text-and-tool-call request."""

    @staticmethod
    def parameters() -> ModelRequestParameters:
        return ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name='save_doc',
                    description='Save a document.',
                    parameters_json_schema={
                        'type': 'object',
                        'properties': {
                            'body': {'type': 'string'},
                            'meta': {'type': 'object'},
                            'priority': {'type': 'integer'},
                            'tags': {'type': 'array', 'items': {'type': 'string'}},
                        },
                    },
                )
            ],
            allow_text_output=True,
            output_tools=[],
            output_mode='tool',
        )


async def _stream_chunks(
    model: GoogleModel, chunks: list[dict[str, Any]], capture: list[dict[str, Any]]
) -> AsyncIterator[ModelResponseStreamEvent]:
    """Yield the model stream's events, stubbing the transport boundary and capturing request configs."""

    async def generate_content_stream(*args: Any, **kwargs: Any) -> AsyncIterator[GenerateContentResponse]:
        capture.append(cast(dict[str, Any], kwargs.get('config', args[-1] if args else None)))

        async def _chunks() -> AsyncIterator[GenerateContentResponse]:
            for chunk in chunks:
                yield GenerateContentResponse.model_validate(chunk)

        return _chunks()

    model.client.aio.models.generate_content_stream = generate_content_stream

    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='Hello')])],
        model_settings=ModelSettings(),
        model_request_parameters=ModelRequestParametersStub.parameters(),
    ) as stream:
        async for event in stream:
            yield event


@pytest.fixture
def vertex_model() -> GoogleModel:
    client = genai.Client(vertexai=True, project='test-project', location='us-central1')
    provider = GoogleProvider(client=client)
    return GoogleModel(
        'gemini-2.5-flash',
        provider=provider,
        profile=GoogleModelProfile(google_stream_function_call_arguments=True),
    )


_OPENING_CHUNK: dict[str, Any] = {
    'candidates': [
        {
            'content': {
                'parts': [{'functionCall': {'id': 'call-1', 'name': 'save_doc', 'willContinue': True}}],
                'role': 'model',
            }
        }
    ]
}


def _fragment_chunk(partial_args: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'candidates': [
            {
                'content': {
                    'parts': [{'functionCall': {'partialArgs': partial_args, 'willContinue': True}}],
                    'role': 'model',
                },
            }
        ]
    }


_CLOSING_CHUNK: dict[str, Any] = {
    'candidates': [{'content': {'parts': [{'functionCall': {}}], 'role': 'model'}, 'finishReason': 'STOP'}]
}


async def test_google_model_iter_stream_function_call_arguments_partial_args_deltas(
    allow_model_requests: None, vertex_model: GoogleModel
):
    """Fragments decode to JSON-text deltas under one vendor part id, reconciled into final arguments."""
    chunks = [
        _OPENING_CHUNK,
        _fragment_chunk([{'json_path': '$.body', 'string_value': 'Body start. ', 'will_continue': True}]),
        _fragment_chunk([{'json_path': '$.body', 'string_value': 'The rest.', 'will_continue': True}]),
        _fragment_chunk([{'json_path': '$.body', 'string_value': '', 'will_continue': False}]),
        _fragment_chunk([{'json_path': '$.meta.author', 'string_value': 'Pydantic', 'will_continue': False}]),
        _fragment_chunk([{'json_path': '$.meta.pages', 'number_value': 3.0, 'will_continue': False}]),
        _fragment_chunk([{'json_path': '$.priority', 'number_value': 2.0, 'will_continue': False}]),
        _fragment_chunk([{'json_path': '$.tags[0]', 'string_value': 'a', 'will_continue': False}]),
        _fragment_chunk([{'json_path': '$.tags[1]', 'string_value': 'b', 'will_continue': False}]),
        _fragment_chunk([{'json_path': '$.tags[]', 'will_continue': False}]),
        _CLOSING_CHUNK,
    ]
    capture: list[dict[str, Any]] = []
    events = [event async for event in _stream_chunks(vertex_model, chunks, capture)]

    # Gating: Vertex streaming with the flag on sends the option inside the function-calling config.
    tool_config = capture[0].get('tool_config')
    assert tool_config is not None
    assert tool_config.get('function_calling_config', {}).get('stream_function_call_arguments') is True

    deltas = [
        event.delta
        for event in events
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta)
    ]
    args_text = ''.join(d.args_delta for d in deltas if isinstance(d.args_delta, str))
    assert 'Body start. The rest.' in args_text

    final_parts = [
        event.part
        for event in events
        if isinstance(event, PartStartEvent)
        and isinstance(event.part, ToolCallPart)
        and event.part.tool_name == 'save_doc'
    ]
    assert len(final_parts) >= 1
    final_part = final_parts[-1]
    assert final_part.args == {
        'body': 'Body start. The rest.',
        'meta': {'author': 'Pydantic', 'pages': 3.0},
        'priority': 2.0,
        'tags': ['a', 'b'],
    }


async def test_google_model_iter_stream_function_call_arguments_gating(
    allow_model_requests: None, vertex_model: GoogleModel
):
    """The flag only reaches Vertex streaming requests: never Gemini API, never unary requests."""
    # Gemini API client: the profile narrows the flag away even when set.
    gemini_model = GoogleModel(
        'gemini-2.5-flash',
        provider=GoogleProvider(api_key='mock-api-key'),
        profile=GoogleModelProfile(google_stream_function_call_arguments=True),
    )
    assert gemini_model.profile.get('google_stream_function_call_arguments') is False

    # Unary Vertex request: no flag even when the profile enables it.
    tool_config = vertex_model._get_tool_config(  # pyright: ignore[reportPrivateUsage]
        ModelRequestParametersStub.parameters(), GoogleModelSettings(), stream=False
    )
    assert tool_config[1] is None or (
        (tool_config[1].get('function_calling_config') or {}).get('stream_function_call_arguments') is None
    )

    # Vertex streaming without the flag: option absent.
    client = genai.Client(vertexai=True, project='p', location='l')
    vertex_model_no_flag = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(client=client))
    tool_config = vertex_model_no_flag._get_tool_config(  # pyright: ignore[reportPrivateUsage]
        ModelRequestParametersStub.parameters(), GoogleModelSettings(), stream=True
    )
    assert tool_config[1] is None or (
        (tool_config[1].get('function_calling_config') or {}).get('stream_function_call_arguments') is None
    )


async def test_google_model_iter_stream(allow_model_requests: None, vertex_model: GoogleModel):
    """Sequential streamed calls never share a vendor part identity; the default path is unchanged."""
    chunks = [
        _OPENING_CHUNK,
        _fragment_chunk([{'json_path': '$.body', 'string_value': 'One', 'will_continue': False}]),
        _CLOSING_CHUNK,
        {
            'candidates': [
                {
                    'content': {
                        'parts': [{'functionCall': {'id': 'call-2', 'name': 'save_doc', 'willContinue': True}}],
                        'role': 'model',
                    }
                }
            ]
        },
        _fragment_chunk([{'json_path': '$.body', 'string_value': 'Two', 'will_continue': False}]),
        _CLOSING_CHUNK,
    ]
    capture: list[dict[str, Any]] = []
    events = [event async for event in _stream_chunks(vertex_model, chunks, capture)]
    finals = [
        event.part
        for event in events
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart) and event.part.args
    ]
    assert [part.args for part in finals] == [{'body': 'One'}, {'body': 'Two'}]
