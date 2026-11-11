"""How adapters send files by media type: the formats an API takes, and aliases sent under their registered spelling.

These are unit tests against mocked clients. Each asserts the request body an adapter builds, or a refusal raised
before any request, so a recording would add a live call without testing more.
"""

from __future__ import annotations as _annotations

from unittest.mock import patch

import pytest

from pydantic_ai import Agent, AudioUrl, BinaryContent
from pydantic_ai.exceptions import UserError

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as openai_available:
    from openai import AsyncOpenAI
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as anthropic_available:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import (
        MockAnthropic,
        completion_message as anthropic_completion_message,
        get_mock_chat_completion_kwargs as anthropic_request_kwargs,
    )


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

    async def test_other_format_url_in_a_user_prompt_is_refused(self, allow_model_requests: None):
        """The adapter refuses after downloading the URL, before any request is built."""
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))

        with patch('pydantic_ai.models.openai._chat_model.download_item') as mock_download:
            mock_download.return_value = {'data': 'ZkxhQw==', 'data_type': 'flac'}
            with pytest.raises(UserError, match="takes WAV and MP3 audio only, not 'audio/flac'"):
                await Agent(self._model(mock_client)).run(['Listen.', AudioUrl('https://example.com/audio.flac')])

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

    @pytest.mark.parametrize(('media_type', 'audio_format'), [('audio/mp3', 'mp3'), ('audio/x-wav', 'wav')])
    def test_alias_is_sent_in_its_format(self, allow_model_requests: None, media_type: str, audio_format: str):
        """The request body's `format` is the behavior under test."""
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))

        Agent(self._model(mock_client)).run_sync(['Listen.', BinaryContent(b'audio', media_type=media_type)])

        user_message = get_mock_chat_completion_kwargs(mock_client)[0]['messages'][0]
        audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')
        assert audio_part['input_audio']['format'] == audio_format


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
def test_anthropic_sends_image_jpg_as_jpeg(allow_model_requests: None):
    """The Anthropic API rejects `image/jpg` with a 400, so the request body carries `image/jpeg`."""
    mock_client = MockAnthropic.create_mock(
        anthropic_completion_message(
            [BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1)
        )
    )
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=mock_client))

    Agent(model).run_sync(['Describe.', BinaryContent(b'jpg', media_type='image/jpg')])

    [user_message] = anthropic_request_kwargs(mock_client)[0]['messages']
    image = next(block for block in user_message['content'] if block['type'] == 'image')
    assert image['source']['media_type'] == 'image/jpeg'
