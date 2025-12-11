from __future__ import annotations as _annotations

import base64
from unittest.mock import patch

import pytest

from pydantic_ai import Agent, AudioUrl, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from ..conftest import try_import
from .mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_openai_chat_audio_default_base64(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    # BinaryContent
    audio_data = b'fake_audio_data'
    binary_audio = BinaryContent(audio_data, media_type='audio/wav')

    agent.run_sync(['Process this audio', binary_audio])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect raw base64
    expected_data = base64.b64encode(audio_data).decode('utf-8')
    assert audio_part['input_audio']['data'] == expected_data
    assert audio_part['input_audio']['format'] == 'wav'


def test_openai_chat_audio_uri_encoding(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)

    # Set profile to use URI encoding
    profile = OpenAIModelProfile(openai_chat_audio_input_encoding='uri')
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client), profile=profile)
    agent = Agent(model)

    # BinaryContent
    audio_data = b'fake_audio_data'
    binary_audio = BinaryContent(audio_data, media_type='audio/wav')

    agent.run_sync(['Process this audio', binary_audio])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect Data URI
    expected_data = f'data:audio/wav;base64,{base64.b64encode(audio_data).decode("utf-8")}'
    assert audio_part['input_audio']['data'] == expected_data
    assert audio_part['input_audio']['format'] == 'wav'


async def test_openai_chat_audio_url_default_base64(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    audio_url = AudioUrl('https://example.com/audio.mp3')

    # Mock download_item to return base64 data
    fake_base64_data = base64.b64encode(b'fake_downloaded_audio').decode('utf-8')

    with patch('pydantic_ai.models.openai.download_item') as mock_download:
        mock_download.return_value = {'data': fake_base64_data, 'data_type': 'mp3'}

        await agent.run(['Process this audio url', audio_url])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect raw base64 (which is what download_item returns in this mock)
    assert audio_part['input_audio']['data'] == fake_base64_data
    assert audio_part['input_audio']['format'] == 'mp3'


async def test_openai_chat_audio_url_uri_encoding(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)

    # Set profile to use URI encoding
    profile = OpenAIModelProfile(openai_chat_audio_input_encoding='uri')
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client), profile=profile)
    agent = Agent(model)

    audio_url = AudioUrl('https://example.com/audio.mp3')

    # Mock download_item to return Data URI (since we're calling with data_format='base64_uri')
    fake_base64_data = base64.b64encode(b'fake_downloaded_audio').decode('utf-8')
    data_uri = f'data:audio/mpeg;base64,{fake_base64_data}'

    with patch('pydantic_ai.models.openai.download_item') as mock_download:
        mock_download.return_value = {'data': data_uri, 'data_type': 'mp3'}

        await agent.run(['Process this audio url', audio_url])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect Data URI with correct MIME type for mp3
    assert audio_part['input_audio']['data'] == data_uri
    assert audio_part['input_audio']['format'] == 'mp3'


async def test_openai_chat_audio_url_custom_media_type(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)

    # Set profile to use URI encoding
    profile = OpenAIModelProfile(openai_chat_audio_input_encoding='uri')
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client), profile=profile)
    agent = Agent(model)

    # AudioUrl with explicit media_type that differs from default extension mapping
    # e.g., .mp3 extension but we want to force a specific weird mime type
    audio_url = AudioUrl('https://example.com/audio.mp3', media_type='audio/custom-weird-format')

    fake_base64_data = base64.b64encode(b'fake_downloaded_audio').decode('utf-8')
    # download_item with data_format='base64_uri' should use the custom media_type from AudioUrl
    data_uri = f'data:audio/custom-weird-format;base64,{fake_base64_data}'

    with patch('pydantic_ai.models.openai.download_item') as mock_download:
        mock_download.return_value = {'data': data_uri, 'data_type': 'mp3'}

        await agent.run(['Process this audio url', audio_url])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect Data URI with the CUSTOM MIME type
    assert audio_part['input_audio']['data'] == data_uri
    assert audio_part['input_audio']['format'] == 'mp3'
