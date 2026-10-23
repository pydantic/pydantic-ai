"""How `OpenAIChatModel` and the model classes that validate with its `_ChatCompletion` read a Chat Completions `finish_reason`."""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, ModelResponse, TextPart, UnexpectedModelBehavior
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.bedrock_mantle import BedrockMantleChatModel
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.models.crusoe import CrusoeModel
    from pydantic_ai.models.github_copilot import GitHubCopilotModel
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.crusoe import CrusoeProvider
    from pydantic_ai.providers.github_copilot import GitHubCopilotProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


MODELS: list[tuple[str, Callable[[AsyncOpenAI], OpenAIChatModel]]] = (
    [
        ('openai', lambda client: OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))),
        ('ollama', lambda client: OllamaModel('qwen3:0.6b', provider=OllamaProvider(openai_client=client))),
        (
            'bedrock-mantle',
            lambda client: BedrockMantleChatModel(
                'openai.gpt-oss-safeguard-120b', provider=BedrockMantleProvider(openai_client=client)
            ),
        ),
        (
            'github-copilot',
            lambda client: GitHubCopilotModel('gpt-5.2', provider=GitHubCopilotProvider(openai_client=client)),
        ),
        ('cerebras', lambda client: CerebrasModel('gpt-oss-120b', provider=CerebrasProvider(openai_client=client))),
        (
            'crusoe',
            lambda client: CrusoeModel('openai/gpt-oss-120b', provider=CrusoeProvider(openai_client=client)),
        ),
    ]
    if imports_successful()
    else []
)


def _completion_handler(finish_reason: object) -> Callable[[httpx2.Request], httpx2.Response]:
    """Answers `Hello!` with `finish_reason`, as a complete response or as a stream, whichever the request asks for."""
    envelope: dict[str, Any] = {'id': 'chatcmpl-1', 'created': 0, 'model': 'test-model'}
    usage = {'prompt_tokens': 9, 'completion_tokens': 2, 'total_tokens': 11}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if not json.loads(request.content).get('stream'):
            choice = {'index': 0, 'message': {'role': 'assistant', 'content': 'Hello!'}, 'finish_reason': finish_reason}
            return httpx2.Response(
                200, json={**envelope, 'object': 'chat.completion', 'choices': [choice], 'usage': usage}
            )
        chunks: list[dict[str, Any]] = [
            {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'Hello!'}, 'finish_reason': None}]},
            {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]},
            {'choices': [], 'usage': usage},
        ]
        events = [f'data: {json.dumps({**envelope, "object": "chat.completion.chunk", **c})}\n\n' for c in chunks]
        return httpx2.Response(
            200, content=''.join([*events, 'data: [DONE]\n\n']), headers={'content-type': 'text/event-stream'}
        )

    return handler


async def _run(model: OpenAIChatModel, stream: bool) -> ModelResponse:
    agent = Agent(model)
    if stream:
        async with agent.run_stream('Hello') as result:
            await result.get_output()
        return result.response
    return (await agent.run('Hello')).response


@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize('finish_reason', ['guardrail_intervened', 'MALFORMED_FUNCTION_CALL'])
@pytest.mark.parametrize('model_factory', [pytest.param(factory, id=name) for name, factory in MODELS])
async def test_unknown_finish_reason(
    allow_model_requests: None,
    model_factory: Callable[[AsyncOpenAI], OpenAIChatModel],
    finish_reason: str,
    stream: bool,
) -> None:
    """A `finish_reason` outside OpenAI's five keeps the response, leaving `finish_reason` unset and the raw value in `provider_details`.

    Complete and streamed responses build the same result. `guardrail_intervened` is a Bedrock value and
    `MALFORMED_FUNCTION_CALL` a Gemini one, from https://github.com/pydantic/pydantic-ai/issues/7816.
    A mock transport stands in for a cassette because no provider returns a non-standard value on demand.
    """
    async with AsyncOpenAI(
        api_key='test',
        base_url='https://example.com/v1',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_completion_handler(finish_reason))),
    ) as client:
        response = await _run(model_factory(client), stream)

    assert response.parts == [TextPart(content='Hello!')]
    assert response.usage == RequestUsage(input_tokens=9, output_tokens=2)
    assert response.finish_reason is None
    assert response.provider_details == {'finish_reason': finish_reason}


@pytest.mark.parametrize('finish_reason', ['', 123, ['stop']])
async def test_invalid_finish_reason_raises(allow_model_requests: None, finish_reason: object) -> None:
    """A `finish_reason` that isn't a non-empty string still fails validation."""
    async with AsyncOpenAI(
        api_key='test',
        base_url='https://example.com/v1',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_completion_handler(finish_reason))),
    ) as client:
        model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))
        with pytest.raises(UnexpectedModelBehavior) as exc_info:
            await Agent(model).run('Hello')

    assert 'choices.0.finish_reason' in exc_info.value.message
