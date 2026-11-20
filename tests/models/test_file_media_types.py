"""How adapters send files by media type: the formats an API takes, and aliases sent under their registered spelling.

The request tests are recorded, and assert the request the SDK sent through `request_capture`. The refusal tests
use a mocked client: they assert a `UserError` raised before any request, so a recording would add nothing. The
tool-return note test does too, as it needs the model to call a tool on cue, and so does the `+yaml` document test,
which patches the mime table so the type is a document on every platform.
"""

from __future__ import annotations as _annotations

import copy
import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired
from unittest.mock import patch

import httpx2
import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai import Agent, AudioUrl, BinaryContent, messages as _messages
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import
from .conftest import AnthropicModelFactory, content_blocks

with try_import() as openai_available:
    from openai import AsyncOpenAI
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
class TestOpenAIChatAudio:
    """Chat Completions' `input_audio` takes WAV and MP3 only."""

    def _model(self, mock_client: AsyncOpenAI) -> OpenAIChatModel:
        return OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client))

    def test_other_format_in_a_user_prompt_is_refused(self, allow_model_requests: None):
        """The adapter refuses before any request is built."""
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))

        with pytest.raises(UserError, match="takes WAV and MP3 audio only, not 'audio/flac'"):
            Agent(self._model(mock_client)).run_sync(['Listen.', BinaryContent(b'fLaC', media_type='audio/flac')])

    @pytest.mark.parametrize(
        ('url', 'media_type'),
        [('https://example.com/audio.flac', 'audio/flac'), ('https://example.com/audio.m4a', 'audio/mp4')],
    )
    async def test_other_format_url_in_a_user_prompt_is_refused_without_downloading(
        self, allow_model_requests: None, url: str, media_type: str
    ):
        """The adapter refuses by the URL's media type, before downloading it or building a request."""
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))

        with patch('pydantic_ai.models.openai._chat_model.download_item') as mock_download:
            with pytest.raises(UserError, match=f"takes WAV and MP3 audio only, not '{media_type}'"):
                await Agent(self._model(mock_client)).run(['Listen.', AudioUrl(url)])
        mock_download.assert_not_called()

    def test_other_format_from_a_tool_is_noted(self, allow_model_requests: None):
        """The note replaces the file in the request body, which is what this asserts."""
        tool_call = ChatCompletionMessageFunctionToolCall(
            id='call_1', function=Function(name='get_audio', arguments='{}'), type='function'
        )
        mock_client = MockOpenAI.create_mock(
            [
                completion_message(ChatCompletionMessage(content=None, role='assistant', tool_calls=[tool_call])),
                completion_message(ChatCompletionMessage(content='done', role='assistant')),
            ]
        )
        agent = Agent(self._model(mock_client))

        @agent.tool_plain
        def get_audio() -> BinaryContent:
            return BinaryContent(b'fLaC', media_type='audio/flac', identifier='song')

        agent.run_sync('Play the song.')

        assert get_mock_chat_completion_kwargs(mock_client)[1]['messages'][-1] == snapshot(
            {
                'role': 'tool',
                'tool_call_id': 'call_1',
                'content': 'File song (audio/flac, 4 bytes) was not sent: this model cannot read audio/flac files.',
            }
        )


class _InputAudio(TypedDict):
    format: str


class _ChatPart(TypedDict):
    type: str
    input_audio: NotRequired[_InputAudio]
    text: NotRequired[str]


class _ChatMessage(TypedDict):
    content: list[_ChatPart]


class _ChatBodyDict(TypedDict):
    messages: list[_ChatMessage]


_ChatBody = TypeAdapter(_ChatBodyDict)
"""The parts of a Chat Completions request body these tests read."""


class _ImageSource(TypedDict):
    media_type: str


class _ImageBlockDict(TypedDict):
    source: _ImageSource


_ImageBlock = TypeAdapter(_ImageBlockDict)
"""The part of an Anthropic image block these tests read."""


def _wav(pcm: bytes) -> bytes:
    """16 kHz mono 16-bit PCM wrapped as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm)
    return buffer.getvalue()


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.vcr()
@pytest.mark.parametrize(('media_type', 'asset'), [('audio/mp3', 'marcelo.mp3'), ('audio/x-wav', 'marcelo_16khz.pcm')])
async def test_openai_chat_sends_an_audio_alias_in_its_format(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    assets_path: Path,
    media_type: str,
    asset: str,
):
    data = assets_path.joinpath(asset).read_bytes()
    if asset.endswith('.pcm'):
        data = _wav(data)
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    agent = Agent(OpenAIChatModel('gpt-audio-mini', provider=provider))

    await agent.run(['What is said in this clip? One sentence.', BinaryContent(data, media_type=media_type)])

    [message] = _ChatBody.validate_python(request_capture.body('/chat/completions'))['messages']
    formats = [part['input_audio']['format'] for part in message['content'] if 'input_audio' in part]
    assert formats == [{'audio/mp3': 'mp3', 'audio/x-wav': 'wav'}[media_type]]


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.vcr()
async def test_anthropic_sends_image_jpg_as_jpeg(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    assets_path: Path,
):
    """The Anthropic API rejects `image/jpg` with a 400, so the request carries `image/jpeg`."""
    model: AnthropicModel = anthropic_model('claude-haiku-5-5', capture=True)
    agent = Agent(model)
    image = BinaryContent(assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpg')

    await agent.run(['What fruit is this? One word.', image])

    [block] = content_blocks(request_capture.body('/v1/messages'), 'image')
    assert _ImageBlock.validate_python(block)['source']['media_type'] == 'image/jpeg'


@dataclass(frozen=True)
class _YamlAdapterCase:
    id: str
    marks: tuple[pytest.MarkDecorator, ...]


_YAML_ADAPTER_CASES = [
    _YamlAdapterCase(id='openai', marks=(pytest.mark.skipif(not openai_available(), reason='openai not installed'),)),
    _YamlAdapterCase(
        id='mistral', marks=(pytest.mark.skipif(not mistral_available(), reason='mistral not installed'),)
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in _YAML_ADAPTER_CASES])
async def test_yaml_structured_suffix_is_inlined_as_text(
    allow_model_requests: None,
    openai_api_key: str,
    mistral_api_key: str,
    request_capture: RequestCapture,
    case: _YamlAdapterCase,
):
    """Chat Completions and Mistral send a `+yaml` type as a text file block, as they send `application/yaml`.

    Both APIs accept the block, and the model answers from the file.
    """
    if case.id == 'openai':
        provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
        model: Model = OpenAIChatModel('gpt-4o-mini', provider=provider)
    else:
        model = MistralModel(
            'mistral-large-latest',
            provider=MistralProvider(api_key=mistral_api_key, http_client=request_capture.client),
        )
    spec = BinaryContent(
        data=b'openapi: 3.1.0\ninfo:\n  title: Demo API\n  version: 1.0.0\npaths: {}\n',
        media_type='application/openapi+yaml',
    )

    result = await Agent(model).run(['What is the title of this API? Answer with the title only.', spec])

    assert result.output == snapshot('Demo API')
    [message] = _ChatBody.validate_python(request_capture.body('/chat/completions'))['messages']
    assert [part['text'] for part in message['content'] if 'text' in part] == snapshot(
        [
            'What is the title of this API? Answer with the title only.',
            """\
-----BEGIN FILE id="f23449" type="application/openapi+yaml"-----
openapi: 3.1.0
info:
  title: Demo API
  version: 1.0.0
paths: {}

-----END FILE id="f23449"-----\
""",
        ]
    )


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in _YAML_ADAPTER_CASES])
async def test_yaml_structured_suffix_image_is_sent_as_an_image(allow_model_requests: None, case: _YamlAdapterCase):
    """An `image/*+yaml` type is an image, so Chat Completions and Mistral send it as an image part, not as text.

    Unit test, not VCR: no registered `image/*+yaml` type exists for an API to read, so the request goes to a
    mocked transport.
    """
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        message = {'role': 'assistant', 'content': 'ok'}
        choice = {'index': 0, 'finish_reason': 'stop', 'message': message}
        usage = {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}
        body = {'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': 'm', 'choices': [choice], 'usage': usage}
        return httpx2.Response(200, json=body)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        if case.id == 'openai':
            model: Model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(api_key='test', http_client=client))
        else:
            model = MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test', http_client=client))
        image = BinaryContent(data=b'x: 1\n', media_type='image/x+yaml')

        await Agent(model).run(['Describe this image.', image])

    [message] = _ChatBody.validate_python(json.loads(requests[0].content))['messages']
    assert [part['type'] for part in message['content']] == ['text', 'image_url']


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_openai_chat_sends_a_yaml_structured_suffix_document_as_a_file(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
):
    """A `+yaml` type the mime table knows an extension for is a document, and is sent as a file part."""
    mime_types = copy.deepcopy(_messages._mime_types)  # pyright: ignore[reportPrivateUsage]
    mime_types.add_type('application/cwl+yaml', '.cwl.yaml')
    monkeypatch.setattr(_messages, '_mime_types', mime_types)
    cwl_content = BinaryContent(data=b'cwlVersion: v1.2\n', media_type='application/cwl+yaml')
    assert cwl_content.is_document
    mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))

    await Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))).run(
        ['What is this file?', cwl_content]
    )

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'What is this file?', 'type': 'text'},
                    {
                        'file': {
                            'file_data': 'data:application/cwl+yaml;base64,Y3dsVmVyc2lvbjogdjEuMgo=',
                            'filename': 'filename.cwl.yaml',
                        },
                        'type': 'file',
                    },
                ],
            }
        ]
    )
