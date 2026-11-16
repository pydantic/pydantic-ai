"""Hugging Face logprobs: the settings sent to the SDK and the `provider_details['logprobs']` they return.

The tests mock the SDK client rather than replay a cassette: no Hugging Face recording carries logprobs yet.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from huggingface_hub import (
        ChatCompletionOutputLogprob,
        ChatCompletionOutputLogprobs,
        ChatCompletionOutputMessage,
        ChatCompletionOutputTopLogprob,
        ChatCompletionStreamOutput,
        ChatCompletionStreamOutputDelta,
        ChatCompletionStreamOutputUsage,
    )

    from pydantic_ai.models.huggingface import HuggingFaceModel, HuggingFaceModelSettings
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .test_huggingface import FinishReason, MockHuggingFace, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed'),
    pytest.mark.filterwarnings('ignore::ResourceWarning'),
]


async def test_huggingface_logprobs(allow_model_requests: None):
    c = completion_message(
        ChatCompletionOutputMessage(content='hello', role='assistant'),
        logprobs=ChatCompletionOutputLogprobs(
            content=[
                ChatCompletionOutputLogprob(
                    token='hello',
                    logprob=-0.25,
                    top_logprobs=[
                        ChatCompletionOutputTopLogprob(token='hello', logprob=-0.25),
                        ChatCompletionOutputTopLogprob(token='hi', logprob=-2.5),
                    ],
                )
            ]
        ),
    )
    mock_client = MockHuggingFace.create_mock(c)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m, model_settings=HuggingFaceModelSettings(huggingface_logprobs=True, huggingface_top_logprobs=2))

    result = await agent.run('hello')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert (kwargs['logprobs'], kwargs['top_logprobs']) == (True, 2)
    provider_details = result.response.provider_details
    assert provider_details is not None
    assert provider_details['logprobs'] == snapshot(
        [
            {
                'token': 'hello',
                'logprob': -0.25,
                'top_logprobs': [{'token': 'hello', 'logprob': -0.25}, {'token': 'hi', 'logprob': -2.5}],
            }
        ]
    )


async def test_huggingface_logprobs_not_requested(allow_model_requests: None):
    c = completion_message(ChatCompletionOutputMessage(content='hello', role='assistant'))
    mock_client = MockHuggingFace.create_mock(c)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)

    result = await agent.run('hello')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert (kwargs['logprobs'], kwargs['top_logprobs']) == (None, None)
    provider_details = result.response.provider_details
    assert provider_details is not None
    assert 'logprobs' not in provider_details


async def test_huggingface_logprobs_null_top_logprobs(allow_model_requests: None):
    # The SDK types `top_logprobs` as required, so only an unvalidated parse produces the `null` some providers send.
    logprobs = ChatCompletionOutputLogprobs.parse_obj_as_instance(  # pyright: ignore[reportUnknownMemberType]
        {'content': [{'token': 'hello', 'logprob': -0.25, 'top_logprobs': None}]}
    )
    c = completion_message(ChatCompletionOutputMessage(content='hello', role='assistant'), logprobs=logprobs)
    mock_client = MockHuggingFace.create_mock(c)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m, model_settings=HuggingFaceModelSettings(huggingface_logprobs=True))

    result = await agent.run('hello')

    provider_details = result.response.provider_details
    assert provider_details is not None
    assert provider_details['logprobs'] == snapshot([{'token': 'hello', 'logprob': -0.25, 'top_logprobs': []}])


def logprobs_chunk(text: str, logprob: float, finish_reason: FinishReason | None = None) -> ChatCompletionStreamOutput:
    """A streamed chunk carrying the logprobs for its own token, as the API returns them."""
    return ChatCompletionStreamOutput.parse_obj_as_instance(  # pyright: ignore[reportUnknownMemberType]
        {
            'id': 'x',
            'choices': [
                {
                    'index': 0,
                    'delta': ChatCompletionStreamOutputDelta(content=text, role='assistant'),
                    'finish_reason': finish_reason,
                    'logprobs': {
                        'content': [
                            {'token': text, 'logprob': logprob, 'top_logprobs': [{'token': text, 'logprob': logprob}]}
                        ]
                    },
                }
            ],
            'created': 1704067200,  # 2024-01-01
            'model': 'hf-model',
            'object': 'chat.completion.chunk',
            'usage': ChatCompletionStreamOutputUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
        }
    )


async def test_huggingface_logprobs_streaming(allow_model_requests: None):
    stream = [logprobs_chunk('hello ', -0.25), logprobs_chunk('world', -1.5, finish_reason='stop')]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m, model_settings=HuggingFaceModelSettings(huggingface_logprobs=True, huggingface_top_logprobs=1))

    async with agent.run_stream('') as result:
        await result.get_output()

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert (kwargs['logprobs'], kwargs['top_logprobs']) == (True, 1)
    provider_details = result.response.provider_details
    assert provider_details is not None
    assert provider_details['logprobs'] == snapshot(
        [
            {'token': 'hello ', 'logprob': -0.25, 'top_logprobs': [{'token': 'hello ', 'logprob': -0.25}]},
            {'token': 'world', 'logprob': -1.5, 'top_logprobs': [{'token': 'world', 'logprob': -1.5}]},
        ]
    )
