"""Hugging Face logprobs: the settings sent to the SDK and the `provider_details['logprobs']` they return.

The recorded tests go through the Hugging Face router to Novita. The mocked tests pin what no recording shows:
unset settings reach the SDK as `None`, and a `null` `top_logprobs` maps to an empty list.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import IsList, try_import

with try_import() as imports_successful:
    from cassetter import Cassette
    from huggingface_hub import ChatCompletionOutputLogprobs, ChatCompletionOutputMessage

    from pydantic_ai.models.huggingface import HuggingFaceModel, HuggingFaceModelSettings
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .test_huggingface import MockHuggingFace, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed'),
    pytest.mark.filterwarnings('ignore::ResourceWarning'),
]


@pytest.mark.vcr()
async def test_huggingface_logprobs(allow_model_requests: None, huggingface_api_key: str, vcr: Cassette):
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
    assert provider_details['logprobs'] == IsList(length=(1, ...))


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


@pytest.mark.vcr()
async def test_huggingface_logprobs_streaming(allow_model_requests: None, huggingface_api_key: str, vcr: Cassette):
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
    assert provider_details['logprobs'] == IsList(length=(1, ...))
