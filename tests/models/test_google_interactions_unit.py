from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    AgentInstructionSource,
    BinaryContent,
    CachePoint,
    CompactionPart,
    FilePart,
    ImageUrl,
    InstructionDeltaPart,
    InstructionId,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    RetryFeedbackPart,
    RetryInfo,
    SpeechPart,
    StructuredContent,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.native_tools import FileSearchTool, ImageGenerationTool, MCPServerTool, MemoryTool
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai import interactions as gi
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError, APITimeoutError
    from google.genai._gaos.utils.serializers import ALLOW_UNKNOWN_UNION_VARIANTS

    from pydantic_ai.models.google_interactions import (
        GoogleInteractionsModel,
        GoogleInteractionsModelSettings,
        GoogleInteractionsStreamedResponse,
        _replay,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.profiles.google import GoogleModelProfile
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

_JSON = TypeAdapter(dict[str, JsonValue])


@dataclass
class MockGoogle:
    body: dict[str, JsonValue] = field(
        default_factory=lambda: {
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'hello'}]}],
        }
    )
    requests: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    status: int = 200

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(_JSON.validate_json(request.read()))
        return httpx2.Response(self.status, json=self.body)


@pytest.fixture
def mock_google() -> MockGoogle:
    return MockGoogle()


@pytest.fixture
def mock_model(mock_google: MockGoogle) -> GoogleInteractionsModel:
    return GoogleInteractionsModel(
        'gemini-3.8-flash',
        provider=GoogleProvider(
            api_key='test-key', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(mock_google.handle))
        ),
    )


def test_inference_preserves_instance(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    model = infer_model('google-interactions:gemini-3.8-flash')
    assert isinstance(model, GoogleInteractionsModel)
    assert infer_model(model) is model
    assert model.system == 'google'
    assert model.canonical_system == 'google'
    assert model.model_name == 'gemini-3.8-flash'


async def test_settings_and_usage(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    """Mock the wire for exact parameter and usage semantics that cassette matching does not check."""
    mock_google.body['usage'] = {
        'total_input_tokens': 20,
        'total_output_tokens': 5,
        'total_thought_tokens': 3,
        'total_cached_tokens': 7,
        'input_tokens_by_modality': [{'modality': 'audio', 'tokens': 4}],
        'cached_tokens_by_modality': [{'modality': 'audio', 'tokens': 2}],
        'output_tokens_by_modality': [{'modality': 'audio', 'tokens': 1}],
        'grounding_tool_count': [{'type': 'google_search', 'count': 2}],
    }
    settings: GoogleInteractionsModelSettings = {
        'temperature': 0.1,
        'top_p': 0.8,
        'seed': 7,
        'max_tokens': 32,
        'stop_sequences': ['DONE'],
        'thinking': 'high',
        'service_tier': 'default',
        'google_safety_settings': [{'type': 'harassment', 'threshold': 'block_only_high'}],
        'google_labels': {'purpose': 'test'},
    }
    response = await model_request(mock_model, [ModelRequest(parts=[UserPromptPart('hello')])], model_settings=settings)
    body = mock_google.requests[-1]
    assert body['generation_config'] == {
        'temperature': 0.1,
        'top_p': 0.8,
        'seed': 7,
        'max_output_tokens': 32,
        'stop_sequences': ['DONE'],
        'thinking_level': 'high',
        'thinking_summaries': 'auto',
    }
    assert body['store'] is False
    assert body['service_tier'] == 'standard'
    assert body['safety_settings'] == settings['google_safety_settings']
    assert body['labels'] == {'purpose': 'test'}
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 8
    assert response.usage.output_reasoning_tokens == 3
    assert response.usage.cache_read_tokens == 7
    assert response.usage.input_audio_tokens == 4
    assert response.usage.output_audio_tokens == 1
    assert response.usage.cache_audio_read_tokens == 2


async def test_history_edits_are_sent(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    mock_google.body['steps'] = [
        {
            'type': 'model_output',
            'content': [
                {'type': 'text', 'text': 'first'},
                {'type': 'image', 'mime_type': 'image/png', 'data': base64.b64encode(b'png').decode()},
                {'type': 'text', 'text': 'last'},
            ],
        }
    ]
    first = await model_request(mock_model, [ModelRequest(parts=[UserPromptPart('hello')])])
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([first]))
    response = history[0]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], TextPart)
    response.parts[0].content = 'edited'
    response.parts = response.parts[:-1]
    await model_request(mock_model, [*history, ModelRequest(parts=[UserPromptPart('continue')])])
    assert mock_google.requests[-1]['input'] == [
        {
            'type': 'model_output',
            'content': [
                {'type': 'text', 'text': 'edited'},
                {'type': 'image', 'mime_type': 'image/png', 'data': base64.b64encode(b'png').decode()},
            ],
        },
        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'continue'}]},
    ]


async def test_foreign_history_does_not_replay_signatures(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hello')]),
        ModelResponse(
            parts=[ThinkingPart('reasoning', signature='foreign', provider_name='other'), TextPart('answer')],
            provider_name='other',
        ),
        ModelRequest(parts=[UserPromptPart('continue')]),
    ]
    await model_request(mock_model, messages)
    assert 'foreign' not in str(mock_google.requests[-1]['input'])
    assert 'reasoning' in str(mock_google.requests[-1]['input'])


@pytest.mark.parametrize(
    'status,reason',
    [
        ('completed', 'stop'),
        ('requires_action', 'tool_call'),
        ('incomplete', 'length'),
        ('budget_exceeded', 'length'),
        ('failed', 'error'),
        ('cancelled', 'error'),
    ],
)
async def test_finish_status(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, status: str, reason: str, allow_model_requests: None
):
    mock_google.body['status'] = status
    response = await model_request(mock_model, [ModelRequest(parts=[UserPromptPart('hello')])])
    assert response.finish_reason == reason
    assert response.usage.unmeasured_requests == 1


async def iter_events(values: list[dict[str, JsonValue]]) -> AsyncIterator[gi.InteractionSSEEvent]:
    adapter = TypeAdapter(gi.InteractionSSEEvent)
    for value in values:
        yield adapter.validate_python(value, context={ALLOW_UNKNOWN_UNION_VARIANTS: True})


async def test_stream_mixed_content_and_signature_fragments():
    """Synthetic fragments exercise ordering, signatures, annotations and binary chunk boundaries."""
    values: list[dict[str, JsonValue]] = [
        {'event_type': 'step.start', 'index': 0, 'step': {'type': 'thought'}},
        {
            'event_type': 'step.delta',
            'index': 0,
            'delta': {'type': 'thought_summary', 'content': {'type': 'text', 'text': 'thinking'}},
        },
        {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'thought_signature', 'signature': 'abc'}},
        {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'thought_signature', 'signature': 'def'}},
        {'event_type': 'step.stop', 'index': 0},
        {'event_type': 'step.start', 'index': 1, 'step': {'type': 'model_output'}},
        {'event_type': 'step.delta', 'index': 1, 'delta': {'type': 'text', 'text': 'first'}},
        {
            'event_type': 'step.delta',
            'index': 1,
            'delta': {
                'type': 'text_annotation_delta',
                'annotations': [
                    {
                        'type': 'url_citation',
                        'url': 'https://example.com',
                        'title': 'Example',
                        'start_index': 0,
                        'end_index': 5,
                    }
                ],
            },
        },
        {'event_type': 'step.delta', 'index': 1, 'delta': {'type': 'image', 'mime_type': 'image/png', 'data': 'YWJj'}},
        {'event_type': 'step.delta', 'index': 1, 'delta': {'type': 'image', 'mime_type': 'image/png', 'data': 'ZGVm'}},
        {'event_type': 'step.delta', 'index': 1, 'delta': {'type': 'text', 'text': 'last'}},
        {'event_type': 'step.stop', 'index': 1},
        {
            'event_type': 'interaction.completed',
            'interaction': {
                'id': 'unit-interaction',
                'status': 'completed',
                'usage': {'total_input_tokens': 1, 'total_output_tokens': 2, 'total_thought_tokens': 3},
            },
        },
    ]
    response = GoogleInteractionsStreamedResponse(
        ModelRequestParameters(), iter_events(values), 'gemini-test', 'google', 'https://example.com'
    )
    events = [event async for event in response]
    starts = [(event.index, event.part.part_kind) for event in events if isinstance(event, PartStartEvent)]
    ends = [(event.index, event.next_part_kind) for event in events if isinstance(event, PartEndEvent)]
    assert starts == [(0, 'thinking'), (1, 'text'), (2, 'file'), (3, 'text')]
    assert ends == [(0, 'text'), (1, 'file'), (3, None)]
    parts = response.get().parts
    assert len(parts) == 4
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].signature == 'abcdef'
    assert isinstance(parts[1], TextPart) and parts[1].content == 'first' and parts[1].citations
    assert isinstance(parts[2], FilePart) and parts[2].content.data == b'abcdef'
    assert isinstance(parts[3], TextPart) and parts[3].content == 'last'
    assert response.usage.output_tokens == 5


@pytest.mark.parametrize(
    'values,match',
    [
        ([], 'ended before'),
        ([{'event_type': 'error', 'error': {'message': 'broken', 'code': 'internal'}}], 'stream failed'),
        (
            [{'event_type': 'future_event'}],
            'Unexpected Google Interactions stream event',
        ),
        (
            [
                {'event_type': 'step.start', 'index': 0, 'step': {'type': 'thought'}},
                {
                    'event_type': 'step.delta',
                    'index': 0,
                    'delta': {'type': 'thought_summary', 'content': {'type': 'image'}},
                },
            ],
            'Expected text',
        ),
        (
            [
                {'event_type': 'step.start', 'index': 0, 'step': {'type': 'model_output'}},
                {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'text_annotation_delta'}},
            ],
            'without a text content block',
        ),
    ],
)
async def test_stream_errors(values: list[dict[str, JsonValue]], match: str):
    response = GoogleInteractionsStreamedResponse(
        ModelRequestParameters(), iter_events(values), 'gemini-test', 'google', 'https://example.com'
    )
    with pytest.raises(UnexpectedModelBehavior, match=match):
        async for _ in response:
            pass


async def test_owned_http_client_refresh_updates_interactions_resource():
    """The lazy SDK resource captures the original HTTP client, which a later run can replace."""
    provider = GoogleProvider(api_key='test-key')
    model = GoogleInteractionsModel('gemini-3.8-flash', provider=provider)
    resource = provider.client.aio.interactions
    original = resource.sdk_configuration.async_client
    assert original is not None
    await original.aclose()
    model._refresh_http_client()  # pyright: ignore[reportPrivateUsage]
    replacement = provider.client.aio.interactions.sdk_configuration.async_client
    assert replacement is not original
    assert replacement is not None
    await replacement.aclose()


@pytest.mark.parametrize('thinking', [True, False])
async def test_boolean_thinking_and_native_overrides(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, thinking: bool, allow_model_requests: None
):
    """Pin precedence on the wire; the cassette matcher does not compare these settings."""
    await model_request(
        mock_model,
        [ModelRequest.user_text_prompt('hello')],
        model_settings={'thinking': thinking},
    )
    assert mock_google.requests[-1]['generation_config'] == (
        {'thinking_summaries': 'auto'} if thinking else {'thinking_level': 'minimal', 'thinking_summaries': 'none'}
    )
    settings: GoogleInteractionsModelSettings = {
        'thinking': 'low',
        'google_thinking_level': 'high',
        'google_generation_config': {'thinking_level': 'medium', 'thinking_summaries': 'auto', 'temperature': 0.3},
    }
    await model_request(mock_model, [ModelRequest.user_text_prompt('hello')], model_settings=settings)
    assert mock_google.requests[-1]['generation_config'] == {
        'thinking_level': 'medium',
        'thinking_summaries': 'auto',
        'temperature': 0.3,
    }
    settings['google_generation_config'] = {'temperature': 0.4}
    await model_request(mock_model, [ModelRequest.user_text_prompt('hello')], model_settings=settings)
    assert mock_google.requests[-1]['generation_config'] == {
        'thinking_level': 'high',
        'thinking_summaries': 'auto',
        'temperature': 0.4,
    }


async def test_restricted_tools_and_native_parameters(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    """Assert exact restriction, authentication and provider-specific settings forwarding."""
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='first', parameters_json_schema={'type': 'object'}),
            ToolDefinition(name='second', parameters_json_schema={'type': 'object'}),
        ],
        native_tools=[
            FileSearchTool(file_store_ids=['stores/test'], max_num_results=4),
            MCPServerTool(
                id='docs',
                url='https://example.com/mcp',
                headers={'X-Test': 'yes'},
                authorization_token='Bearer test',
                allowed_tools=['lookup'],
            ),
        ],
    )
    settings: GoogleInteractionsModelSettings = {
        'tool_choice': ['first'],
        'extra_body': {'custom': True, 'store': True},
        'timeout': httpx.Timeout(10),
    }
    await model_request(
        mock_model, [ModelRequest.user_text_prompt('hello')], model_settings=settings, model_request_parameters=params
    )
    body = mock_google.requests[-1]
    assert body['generation_config'] == {'tool_choice': {'allowed_tools': {'mode': 'any', 'tools': ['first']}}}
    assert body['tools'] == [
        {'type': 'function', 'name': 'first', 'description': '', 'parameters': {'type': 'object'}},
        {'type': 'function', 'name': 'second', 'description': '', 'parameters': {'type': 'object'}},
        {'type': 'file_search', 'file_search_store_names': ['stores/test'], 'top_k': 4},
        {
            'type': 'mcp_server',
            'url': 'https://example.com/mcp',
            'name': 'docs',
            'headers': {'X-Test': 'yes', 'Authorization': 'Bearer test'},
            'allowed_tools': [{'tools': ['lookup']}],
        },
    ]
    assert body['custom'] is True and body['store'] is False
    with pytest.raises(UserError, match='MemoryTool'):
        mock_model._map_tools(ModelRequestParameters(native_tools=[MemoryTool()]))  # pyright: ignore[reportPrivateUsage]


async def test_image_generation_settings(mock_google: MockGoogle, allow_model_requests: None):
    """Image settings select a response format rather than declaring a function tool."""
    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(
            api_key='test-key', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(mock_google.handle))
        ),
    )
    params = ModelRequestParameters(
        allow_text_output=False,
        allow_image_output=True,
        native_tools=[ImageGenerationTool(aspect_ratio='16:9', size='2K', output_format='jpeg')],
    )
    await model_request(model, [ModelRequest.user_text_prompt('a landscape')], model_request_parameters=params)
    assert mock_google.requests[-1]['response_format'] == {
        'type': 'image',
        'aspect_ratio': '16:9',
        'image_size': '2K',
        'mime_type': 'image/jpeg',
    }
    params.native_tools = []
    await model_request(model, [ModelRequest.user_text_prompt('a landscape')], model_request_parameters=params)
    assert mock_google.requests[-1]['response_format'] == {'type': 'image'}
    params.native_tools = [ImageGenerationTool(size='auto', output_format='png')]
    params.allow_text_output = True
    await model_request(model, [ModelRequest.user_text_prompt('a landscape')], model_request_parameters=params)
    assert mock_google.requests[-1]['response_format'] == [{'type': 'image'}, {'type': 'text'}]


async def test_prompt_content_and_file_references(
    mock_model: GoogleInteractionsModel,
    mock_google: MockGoogle,
    monkeypatch: pytest.MonkeyPatch,
    allow_model_requests: None,
):
    """Exercise reference ownership and forced downloads without an external file service."""
    downloader = AsyncMock(return_value={'data': b'png', 'data_type': 'image/png'})
    monkeypatch.setattr('pydantic_ai.models.google_interactions.download_item', downloader)
    parts: list[ModelRequestPart] = [
        SystemPromptPart('system'),
        UserPromptPart(''),
        UserPromptPart(
            [
                '',
                'hello',
                TextContent('tagged'),
                StructuredContent({'x': 1}),
                CachePoint(),
                UploadedFile(
                    'https://generativelanguage.googleapis.com/v1beta/files/test',
                    provider_name='google',
                    media_type='image/png',
                ),
                ImageUrl('https://example.com/image.png'),
                ImageUrl('https://example.com/download.png', force_download=True),
            ]
        ),
    ]
    await model_request(mock_model, [ModelRequest(parts=parts)])
    body = mock_google.requests[-1]
    assert body['system_instruction'] == 'system'
    content = TypeAdapter(list[dict[str, JsonValue]]).validate_python(body['input'])[0]['content']
    assert content == [
        {'type': 'text', 'text': 'hello'},
        {'type': 'text', 'text': 'tagged'},
        {'type': 'text', 'text': '{"x":1}'},
        {
            'type': 'image',
            'uri': 'https://generativelanguage.googleapis.com/v1beta/files/test',
            'mime_type': 'image/png',
        },
        {'type': 'image', 'uri': 'https://example.com/image.png', 'mime_type': 'image/png'},
        {'type': 'image', 'data': 'cG5n', 'mime_type': 'image/png'},
    ]
    downloader.assert_awaited_once()


@pytest.mark.parametrize('user_turn', [False, True])
async def test_unstored_history_and_cross_api_signatures(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, user_turn: bool, allow_model_requests: None
):
    """History without complete Interactions metadata still carries compatible Google signatures."""
    mock_model = GoogleInteractionsModel(
        mock_model.model_name,
        provider=GoogleProvider(client=mock_model.client),
        profile={'mimics_assistant_message_formatting': user_turn},
    )
    history = ModelResponse(
        parts=[
            ThinkingPart('signed summary', signature='signed', provider_name='google'),
            ThinkingPart('unsigned summary'),
            ThinkingPart(''),
            ThinkingPart('', signature='empty-summary', provider_name='google'),
            TextPart(''),
            ToolCallPart(
                'lookup', {}, 'call', provider_name='google', provider_details={'thought_signature': 'inline'}
            ),
            NativeToolCallPart('foreign', {}, 'native', provider_name='other'),
            NativeToolReturnPart('foreign', 'done', 'native', provider_name='other'),
        ],
        provider_name='google',
    )
    await model_request(mock_model, [history, ModelRequest.user_text_prompt('continue')])
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(mock_google.requests[-1]['input'])
    assert steps[0] == {
        'type': 'thought',
        'signature': 'signed',
        'summary': [{'type': 'text', 'text': 'signed summary'}],
    }
    assert steps[1]['type'] == ('user_input' if user_turn else 'model_output')
    assert steps[2] == {'type': 'thought', 'signature': 'empty-summary'}
    assert steps[3] == {'type': 'thought', 'signature': 'inline'}
    assert steps[4] == {'type': 'function_call', 'id': 'call', 'name': 'lookup', 'arguments': {}}
    assert len(steps) == 6


async def test_native_history_edits_preserve_opaque_fields(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    """Persisted metadata is replayed with user edits, including invisible signed steps."""
    mock_google.body['steps'] = [
        {'type': 'thought', 'signature': 'signature', 'summary': [{'type': 'text', 'text': 'original'}]},
        {
            'type': 'code_execution_call',
            'id': 'call',
            'arguments': {'code': '1+1', 'language': 'python'},
            'signature': 'opaque',
        },
        {'type': 'code_execution_result', 'call_id': 'call', 'result': '2'},
    ]
    response = await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])
    thought, call, result = response.parts
    assert isinstance(thought, ThinkingPart)
    thought.signature = None
    thought.content = ''
    assert isinstance(call, NativeToolCallPart)
    edited_call = NativeToolCallPart(
        call.tool_name,
        {'code': '2+2', 'language': 'python'},
        call.tool_call_id,
        provider_name=call.provider_name,
        provider_details=call.provider_details,
    )
    assert isinstance(result, NativeToolReturnPart)
    edited_result = NativeToolReturnPart(
        result.tool_name,
        '4',
        result.tool_call_id,
        provider_name=result.provider_name,
        provider_details=result.provider_details,
    )
    response.parts = [thought, edited_call, edited_result]
    await model_request(mock_model, [response, ModelRequest.user_text_prompt('continue')])
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(mock_google.requests[-1]['input'])
    assert steps[:3] == [
        {'type': 'thought', 'summary': []},
        {
            'type': 'code_execution_call',
            'id': 'call',
            'arguments': {'code': '2+2', 'language': 'python'},
            'signature': 'opaque',
        },
        {'type': 'code_execution_result', 'call_id': 'call', 'result': '4'},
    ]


@pytest.mark.parametrize('index', [-1, 3, 'bad'])
def test_invalid_persisted_content_index(index: int | str):
    """Reject corrupt persisted references instead of selecting unrelated content."""
    part = TextPart('edited', provider_details={'google_interactions_content_index': index})
    with pytest.raises(UnexpectedModelBehavior, match='Invalid content index'):
        _replay.replay_content({'content': [{'type': 'text', 'text': 'original'}]}, [part])


def test_invalid_persisted_part_type():
    with pytest.raises(UnexpectedModelBehavior, match='does not match'):
        _replay.replay_step({'type': 'thought'}, [CompactionPart()])
    part = ThinkingPart('invalid', provider_details={'google_interactions_content_index': 0})
    with pytest.raises(UnexpectedModelBehavior, match='Invalid content part'):
        _replay.replay_content({'content': [{'type': 'text', 'text': 'original'}]}, [part])


@pytest.mark.parametrize('media', ['image', 'audio', 'video', 'document'])
async def test_uri_media_response_and_replay(
    mock_model: GoogleInteractionsModel,
    mock_google: MockGoogle,
    monkeypatch: pytest.MonkeyPatch,
    media: str,
    allow_model_requests: None,
):
    """Download URI output once, retain its URI on replay, and send edited bytes inline."""
    downloader = AsyncMock(return_value={'data': b'original', 'data_type': 'application/octet-stream'})
    monkeypatch.setattr('pydantic_ai.models.google_interactions._response.download_item', downloader)
    raw: dict[str, JsonValue] = {'type': media, 'uri': 'https://example.com/output'}
    mock_google.body['steps'] = [{'type': 'model_output', 'content': [raw]}]
    response = await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])
    part = response.parts[0]
    assert isinstance(part, FilePart) and part.content.data == b'original'
    mock_google.body['steps'] = []
    await model_request(mock_model, [response, ModelRequest.user_text_prompt('continue')])
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(mock_google.requests[-1]['input'])
    assert steps[0] == {'type': 'model_output', 'content': [raw]}
    part.content = BinaryContent(b'edited', media_type='image/png')
    await model_request(mock_model, [response, ModelRequest.user_text_prompt('continue')])
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(mock_google.requests[-1]['input'])
    assert steps[0] == {
        'type': 'model_output',
        'content': [{'type': 'image', 'data': 'ZWRpdGVk', 'mime_type': 'image/png'}],
    }
    downloader.assert_awaited_once()


async def test_citations_and_error_metadata(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    """Pin UTF-8 citation offsets and retain diagnostic fields without relying on live grounding."""
    mock_google.body.update(
        errors=[{'code': 'internal', 'message': 'provider diagnostic'}],
        steps=[
            {
                'type': 'model_output',
                'content': [
                    {
                        'type': 'text',
                        'text': 'éword',
                        'annotations': [
                            {'type': 'url_citation'},
                            {
                                'type': 'place_citation',
                                'url': 'https://example.com/place',
                                'name': 'Place',
                                'start_index': 2,
                                'end_index': 6,
                            },
                            {'type': 'file_citation', 'document_uri': 'files/source', 'file_name': 'Source'},
                            {'type': 'url_citation', 'url': 'https://example.com', 'start_index': 99, 'end_index': 100},
                            {'type': 'future_annotation'},
                        ],
                    }
                ],
            }
        ],
    )
    response = await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])
    part = response.parts[0]
    assert isinstance(part, TextPart) and part.citations is not None
    assert len(part.citations) == 3
    assert part.citations[0].anchor is not None
    assert part.citations[0].anchor.start == 1 and part.citations[0].anchor.end == 5
    assert part.citations[1].anchor is None and part.citations[2].anchor is None
    assert response.provider_details is not None and response.provider_details['errors'] == [
        {'code': 'internal', 'message': 'provider diagnostic'}
    ]


@pytest.mark.parametrize(
    'step,match',
    [
        ({'type': 'future_step'}, 'response step'),
        ({'type': 'model_output', 'content': [{'type': 'future_content'}]}, 'response content'),
        ({'type': 'model_output', 'content': [{'type': 'image'}]}, 'without data or a URI'),
    ],
)
async def test_invalid_response_data(
    mock_model: GoogleInteractionsModel,
    mock_google: MockGoogle,
    step: dict[str, JsonValue],
    match: str,
    allow_model_requests: None,
):
    mock_google.body['steps'] = [step]
    with pytest.raises(UnexpectedModelBehavior, match=match):
        await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])


@pytest.mark.parametrize('error', ['timeout', 'connection', 'api'])
async def test_sdk_error_mapping(
    mock_model: GoogleInteractionsModel, monkeypatch: pytest.MonkeyPatch, error: str, allow_model_requests: None
):
    """SDK exceptions are synthetic; a live API cannot reliably reproduce a transport outage."""
    request = httpx.Request('POST', 'https://example.com/interactions')
    failures = {
        'timeout': APITimeoutError(request),
        'connection': APIConnectionError(request=request),
        'api': APIError('broken response', request=request, body={'message': 'broken'}),
    }
    monkeypatch.setattr(mock_model.client.aio.interactions, 'create', AsyncMock(side_effect=failures[error]))
    with pytest.raises(ModelAPIError):
        await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])


async def test_unprepared_history_guards(mock_model: GoogleInteractionsModel, allow_model_requests: None):
    """Direct Model.request callers must run prepare_messages before provider serialization."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[InstructionDeltaPart(id=InstructionId(AgentInstructionSource()), content='instruction')]),
        ModelRequest(
            parts=[RetryFeedbackPart('retry', retry=RetryInfo(cause='model_retry', attempt=1, max_retries=1))]
        ),
        ModelResponse(parts=[CompactionPart('summary')]),
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='hello')]),
    ]
    for message in messages:
        with pytest.raises(UserError, match='prepare_messages'):
            await mock_model.request([message], None, ModelRequestParameters())


@pytest.mark.parametrize('stream', [False, True])
async def test_sdk_response_mode_guard(
    mock_model: GoogleInteractionsModel, monkeypatch: pytest.MonkeyPatch, stream: bool, allow_model_requests: None
):
    """A mismatched SDK result must fail before exposing a misleading model response."""
    result = gi.Interaction(status='completed') if stream else iter_events([])
    monkeypatch.setattr(mock_model.client.aio.interactions, 'create', AsyncMock(return_value=result))
    with pytest.raises(UnexpectedModelBehavior, match=r'Expected a .*streaming Google interaction'):
        if stream:
            async with model_request_stream(mock_model, [ModelRequest.user_text_prompt('hello')]):
                pass
        else:
            await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])


async def test_streamed_namespaced_mcp_calls():
    """Provider-run functions can share the function-call wire shape with client tools."""
    values: list[dict[str, JsonValue]] = [
        {
            'event_type': 'step.start',
            'index': 0,
            'step': {'type': 'function_call', 'id': 'mcp-call', 'name': 'docs:lookup', 'arguments': {}},
        },
        {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'arguments_delta', 'arguments': '{"query":"test"}'}},
        {'event_type': 'step.stop', 'index': 0},
        {
            'event_type': 'step.start',
            'index': 1,
            'step': {
                'type': 'function_result',
                'call_id': 'mcp-call',
                'name': 'docs:lookup',
                'result': [{'type': 'text', 'text': 'found'}],
            },
        },
        {'event_type': 'step.stop', 'index': 1},
        {'event_type': 'interaction.completed', 'interaction': {'status': 'completed'}},
    ]
    response = GoogleInteractionsStreamedResponse(
        ModelRequestParameters(native_tools=[MCPServerTool(id='docs', url='https://example.com/mcp')]),
        iter_events(values),
        'gemini-test',
        'google',
        'https://example.com',
    )
    async for _ in response:
        pass
    call, result = response.get().parts
    assert isinstance(call, NativeToolCallPart) and call.args_as_dict() == {'query': 'test'}
    assert isinstance(result, NativeToolReturnPart) and result.content == [{'type': 'text', 'text': 'found'}]


@pytest.mark.parametrize('strict_profile,strict_tool', [(False, None), (True, False)])
async def test_strict_tool_opt_out(
    mock_model: GoogleInteractionsModel,
    mock_google: MockGoogle,
    strict_profile: bool,
    strict_tool: bool | None,
    allow_model_requests: None,
):
    """Either provider or per-tool opt-out keeps Google's automatic mode unvalidated."""
    model = GoogleInteractionsModel(
        mock_model.model_name,
        provider=GoogleProvider(client=mock_model.client),
        profile=GoogleModelProfile(google_supports_strict_tool_definition=strict_profile),
    )
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='lookup', strict=strict_tool, parameters_json_schema={'type': 'object'})]
    )
    settings: GoogleInteractionsModelSettings = {
        'google_thinking_summaries': 'none',
        'google_service_tier': 'priority',
        'service_tier': 'flex',
    }
    await model_request(
        model, [ModelRequest.user_text_prompt('hello')], model_settings=settings, model_request_parameters=params
    )
    assert mock_google.requests[-1]['generation_config'] == {'thinking_summaries': 'none', 'tool_choice': 'auto'}
    assert mock_google.requests[-1]['service_tier'] == 'priority'


async def test_appended_history_part_preserves_existing_steps(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    """A history processor can add an ordinary part beside persisted provider steps."""
    response = await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])
    response.parts = [
        *response.parts,
        TextPart('added', provider_name='google', provider_details={'application': 'label'}),
    ]
    await model_request(mock_model, [response, ModelRequest.user_text_prompt('continue')])
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(mock_google.requests[-1]['input'])
    assert steps[:2] == [
        {'type': 'model_output', 'content': [{'type': 'text', 'text': 'hello'}]},
        {'type': 'model_output', 'content': [{'type': 'text', 'text': 'added'}]},
    ]


async def test_partial_usage_and_metadata_only_media_delta():
    """Missing thought/grounding counts and a metadata-only media chunk are valid partial data."""
    values: list[dict[str, JsonValue]] = [
        {'event_type': 'step.start', 'index': 0, 'step': {'type': 'model_output'}},
        {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'image', 'mime_type': 'image/png', 'data': 'cG5n'}},
        {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'image', 'mime_type': 'image/png'}},
        {'event_type': 'step.stop', 'index': 0},
        {
            'event_type': 'interaction.completed',
            'interaction': {'status': 'completed', 'usage': {'total_input_tokens': 1, 'grounding_tool_count': [{}]}},
        },
    ]
    response = GoogleInteractionsStreamedResponse(
        ModelRequestParameters(), iter_events(values), 'gemini-test', 'google', 'https://example.com'
    )
    async for _ in response:
        pass
    part = response.get().parts[0]
    assert isinstance(part, FilePart) and part.content.data == b'png' and part.content.media_type == 'image/png'
    assert response.usage.input_tokens == 1 and response.usage.output_tokens == 0 and response.usage.details == {}


@pytest.mark.parametrize('mcp', [False, True])
async def test_native_replay_without_response_metadata(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, mcp: bool, allow_model_requests: None
):
    """Per-part replay data survives a history processor dropping response-level metadata."""
    steps: list[dict[str, JsonValue]] = (
        [
            {'type': 'function_call', 'name': 'docs:lookup', 'id': 'call', 'arguments': {}, 'signature': 'opaque'},
            {
                'type': 'function_result',
                'name': 'docs:lookup',
                'call_id': 'call',
                'result': [{'type': 'text', 'text': 'found'}],
            },
        ]
        if mcp
        else [
            {
                'type': 'code_execution_call',
                'id': 'call',
                'arguments': {'code': '1+1', 'language': 'python'},
                'signature': 'opaque',
            },
            {'type': 'code_execution_result', 'call_id': 'call', 'result': '2'},
        ]
    )
    mock_google.body['steps'] = TypeAdapter(JsonValue).validate_python(steps)
    response = await model_request(mock_model, [ModelRequest.user_text_prompt('hello')])
    response.provider_details = None
    await model_request(mock_model, [response, ModelRequest.user_text_prompt('continue')])
    sent = TypeAdapter(list[dict[str, JsonValue]]).validate_python(mock_google.requests[-1]['input'])
    assert sent[:2] == steps


async def test_unknown_own_native_tool_is_not_silently_dropped(
    mock_model: GoogleInteractionsModel, allow_model_requests: None
):
    response = ModelResponse(
        parts=[NativeToolCallPart('unknown', {}, 'call', provider_name='google')], provider_name='google'
    )
    with pytest.raises(UnexpectedModelBehavior, match='unsupported Google Interactions replay step'):
        await model_request(mock_model, [response, ModelRequest.user_text_prompt('continue')])


async def test_provider_environment_and_foreign_file_history(
    mock_model: GoogleInteractionsModel, mock_google: MockGoogle, allow_model_requests: None
):
    """Wire-only options and normalized foreign media must not depend on replay metadata."""
    history: list[ModelMessage] = [
        ModelResponse(parts=[FilePart(BinaryContent(b'png', media_type='image/png'))], provider_name='other'),
        ModelRequest.user_text_prompt('describe this'),
    ]
    params = ModelRequestParameters(
        native_tools=[MCPServerTool(id='docs', url='https://example.com/mcp', authorization_token='Bearer test')]
    )
    settings: GoogleInteractionsModelSettings = {'google_environment': 'environments/test'}
    await model_request(mock_model, history, model_settings=settings, model_request_parameters=params)
    body = mock_google.requests[-1]
    assert body['environment'] == 'environments/test'
    assert body['tools'] == [
        {
            'type': 'mcp_server',
            'name': 'docs',
            'url': 'https://example.com/mcp',
            'headers': {'Authorization': 'Bearer test'},
        }
    ]
    assert TypeAdapter(list[dict[str, JsonValue]]).validate_python(body['input'])[0] == {
        'type': 'model_output',
        'content': [{'type': 'image', 'data': 'cG5n', 'mime_type': 'image/png'}],
    }
