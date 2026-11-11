"""How adapters send files by media type: the formats an API takes, and aliases sent under their registered spelling.

The request tests are recorded, and assert the request the SDK sent through `request_capture`. The refusal tests
use a mocked client: they assert a `UserError` raised before any request, so a recording would add nothing. The
tool-return note test does too, as it needs the model to call a tool on cue.
"""

from __future__ import annotations as _annotations

import io
import wave
from pathlib import Path
from typing import NotRequired
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai import Agent, AudioUrl, BinaryContent
from pydantic_ai.exceptions import UserError

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as openai_available:
    from openai import AsyncOpenAI
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as anthropic_available:
    from .conftest import AnthropicModelFactory, content_blocks


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
    input_audio: NotRequired[_InputAudio]


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
    agent = Agent(anthropic_model('claude-haiku-5-5', capture=True))
    image = BinaryContent(assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpg')

    await agent.run(['What fruit is this? One word.', image])

    [block] = content_blocks(request_capture.body('/v1/messages'), 'image')
    assert _ImageBlock.validate_python(block)['source']['media_type'] == 'image/jpeg'
