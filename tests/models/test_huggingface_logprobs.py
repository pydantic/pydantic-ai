"""Hugging Face logprobs: the settings sent to the SDK and the `provider_details['logprobs']` they return.

The recorded tests go through the Hugging Face router to Novita. The mocked tests pin what no recording shows:
unset settings reach the SDK as `None`, a `null` `top_logprobs` maps to an empty list, and streamed logprobs
collect across chunks.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import try_import

with try_import() as imports_successful:
    from cassetter import Cassette
    from huggingface_hub import (
        ChatCompletionOutputLogprobs,
        ChatCompletionOutputMessage,
        ChatCompletionStreamOutput,
        ChatCompletionStreamOutputDelta,
        ChatCompletionStreamOutputUsage,
    )
    from huggingface_hub.hf_api import InferenceProviderMapping
    from huggingface_hub.inference._providers._common import HARDCODED_MODEL_INFERENCE_MAPPING

    from pydantic_ai.models.huggingface import HuggingFaceModel, HuggingFaceModelSettings
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .test_huggingface import FinishReason, MockHuggingFace, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed'),
    pytest.mark.filterwarnings('ignore::ResourceWarning'),
]


@pytest.fixture
def novita_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map the model to Novita without the Hub lookup, which these recordings don't carry."""
    mapping = InferenceProviderMapping(
        provider='novita',
        hf_model_id='meta-llama/Llama-3.1-8B-Instruct',
        providerId='meta-llama/llama-3.1-8b-instruct',
        status='live',
        task='conversational',
    )
    monkeypatch.setitem(HARDCODED_MODEL_INFERENCE_MAPPING, 'novita', {mapping.hf_model_id: mapping})


@pytest.mark.vcr(additional_matchers=['body'])
async def test_huggingface_logprobs(
    allow_model_requests: None, huggingface_api_key: str, vcr: Cassette, novita_mapping: None
):
    model = HuggingFaceModel(
        'meta-llama/Llama-3.1-8B-Instruct',
        provider=HuggingFaceProvider(provider_name='novita', api_key=huggingface_api_key),
    )
    agent = Agent(model, model_settings=HuggingFaceModelSettings(huggingface_logprobs=True, huggingface_top_logprobs=2))

    result = await agent.run('What is the capital of France? Answer in one word.')

    body = single_request_body(vcr)
    assert (body['logprobs'], body['top_logprobs']) == (True, 2)
    provider_details = result.response.provider_details
    assert provider_details is not None
    assert provider_details['logprobs'] == snapshot(
        [
            {
                'token': 'Paris',
                'logprob': 0,
                'top_logprobs': [{'token': 'Paris', 'logprob': 0}, {'token': 'Par', 'logprob': -17.142858505249023}],
            },
            {
                'token': '.',
                'logprob': -0.008023055270314217,
                'top_logprobs': [
                    {'token': '.', 'logprob': -0.008023055270314217},
                    {'token': '<|eot_id|>', 'logprob': -4.82944917678833},
                ],
            },
            {
                'token': '<|eot_id|>',
                'logprob': 0,
                'top_logprobs': [{'token': '<|eot_id|>', 'logprob': 0}, {'token': ' ', 'logprob': -17.14285659790039}],
            },
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


@pytest.mark.vcr(additional_matchers=['body'])
async def test_huggingface_logprobs_streaming(
    allow_model_requests: None, huggingface_api_key: str, vcr: Cassette, novita_mapping: None
):
    model = HuggingFaceModel(
        'meta-llama/Llama-3.1-8B-Instruct',
        provider=HuggingFaceProvider(provider_name='novita', api_key=huggingface_api_key),
    )
    agent = Agent(model, model_settings=HuggingFaceModelSettings(huggingface_logprobs=True, huggingface_top_logprobs=2))

    async with agent.run_stream('What is the capital of France? Answer in one word.') as result:
        await result.get_output()

    body = single_request_body(vcr)
    assert (body['stream'], body['logprobs'], body['top_logprobs']) == (True, True, 2)
    provider_details = result.response.provider_details
    assert provider_details is not None
    # Novita leaves the token's `logprob` out of the streamed chunk, though the non-streamed response gives 0.
    assert provider_details['logprobs'] == snapshot(
        [
            {
                'token': 'Paris',
                'logprob': None,
                'top_logprobs': [{'token': 'Paris', 'logprob': 0}, {'token': 'Par', 'logprob': -17.142858505249023}],
            }
        ]
    )


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


async def test_huggingface_logprobs_streaming_accumulates(allow_model_requests: None):
    """The streamed recording carries logprobs in one chunk only, so this pins collecting them across chunks."""
    stream = [logprobs_chunk('hello ', -0.25), logprobs_chunk('world', -1.5, finish_reason='stop')]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m, model_settings=HuggingFaceModelSettings(huggingface_logprobs=True, huggingface_top_logprobs=1))

    async with agent.run_stream('') as result:
        await result.get_output()

    provider_details = result.response.provider_details
    assert provider_details is not None
    assert provider_details['logprobs'] == snapshot(
        [
            {'token': 'hello ', 'logprob': -0.25, 'top_logprobs': [{'token': 'hello ', 'logprob': -0.25}]},
            {'token': 'world', 'logprob': -1.5, 'top_logprobs': [{'token': 'world', 'logprob': -1.5}]},
        ]
    )
