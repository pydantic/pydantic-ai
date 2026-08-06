import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse
from pydantic_ai.direct import model_request_stream

from .._inline_snapshot import snapshot
from ..conftest import message, try_import
from .mock_openai import MockOpenAI

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.vercel import VercelModel
    from pydantic_ai.providers.vercel import VercelProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_vercel_model_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.models import infer_model

    monkeypatch.setenv('VERCEL_AI_GATEWAY_API_KEY', 'mock-api-key')
    model = infer_model('vercel:deepseek/deepseek-v4-flash')
    assert isinstance(model, VercelModel)
    assert model.model_name == 'deepseek/deepseek-v4-flash'


async def test_vercel_reports_cost(allow_model_requests: None, vercel_api_key: str) -> None:
    model = VercelModel('deepseek/deepseek-v4-flash', provider=VercelProvider(api_key=vercel_api_key))
    agent = Agent(model)

    result = await agent.run('What is 2+2? Reply with just the number.')

    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details is not None
    assert response.provider_details['cost'] == snapshot(1.652e-05)


async def test_vercel_stream_reports_cost(allow_model_requests: None, vercel_api_key: str) -> None:
    model = VercelModel('deepseek/deepseek-v4-flash', provider=VercelProvider(api_key=vercel_api_key))

    async with model_request_stream(model, [ModelRequest.user_text_prompt('Who are you?')]) as stream:
        _ = [chunk async for chunk in stream]

        assert stream.provider_details is not None
        assert stream.provider_details['cost'] == snapshot(0.0001029)


async def test_vercel_stream_cost_on_usage_only_chunk(allow_model_requests: None) -> None:
    """The cost must survive arriving on a spec-shaped final chunk with empty `choices`.

    The base `OpenAIStreamedResponse` event loop skips such chunks before its
    `_map_provider_details` hook, so `VercelStreamedResponse` lifts the cost in `_map_usage`.
    """
    stream = [
        ChatCompletionChunk(
            id='chunk-1',
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content='4', role='assistant'))],
            created=1704067200,
            model='deepseek/deepseek-v4-flash',
            object='chat.completion.chunk',
        ),
        ChatCompletionChunk(
            id='chunk-2',
            choices=[],
            created=1704067200,
            model='deepseek/deepseek-v4-flash',
            object='chat.completion.chunk',
            usage=CompletionUsage.model_validate(
                {'prompt_tokens': 14, 'completion_tokens': 1, 'total_tokens': 15, 'cost': 1.316e-05}
            ),
        ),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    model = VercelModel('deepseek/deepseek-v4-flash', provider=VercelProvider(openai_client=mock_client))

    async with model_request_stream(model, [ModelRequest.user_text_prompt('What is 2+2?')]) as streamed:
        _ = [chunk async for chunk in streamed]

        assert streamed.provider_details is not None
        assert streamed.provider_details['cost'] == snapshot(1.316e-05)


async def test_vercel_no_cost_when_gateway_omits_it(allow_model_requests: None) -> None:
    stream = [
        ChatCompletionChunk(
            id='chunk-1',
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content='4', role='assistant'))],
            created=1704067200,
            model='deepseek/deepseek-v4-flash',
            object='chat.completion.chunk',
            usage=CompletionUsage(prompt_tokens=14, completion_tokens=1, total_tokens=15),
        ),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    model = VercelModel('deepseek/deepseek-v4-flash', provider=VercelProvider(openai_client=mock_client))

    async with model_request_stream(model, [ModelRequest.user_text_prompt('What is 2+2?')]) as streamed:
        _ = [chunk async for chunk in streamed]

        assert (streamed.provider_details or {}).get('cost') is None
