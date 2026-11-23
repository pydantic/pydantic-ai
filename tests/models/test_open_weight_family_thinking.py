"""Requests for the open-weight thinking families on the OpenAI-compatible gateways that resolve their profiles.

`test_open_weight_family_thinking_on_openrouter` is recorded against OpenRouter's OpenAI-compatible API. The other
tests are not VCR tests: there are no recordings for the other gateways, and their claims are about the request the
code builds, which cassette matching doesn't check. A `MockTransport` serves synthetic responses, and
`request_capture` records the requests. See https://github.com/pydantic/pydantic-ai/issues/12096.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
import pytest
from pydantic import BaseModel, JsonValue

from pydantic_ai import Agent, ModelResponse, NativeOutput, ThinkingPart
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.litellm import LiteLLMProvider
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
    from pydantic_ai.providers.together import TogetherProvider
    from pydantic_ai.providers.vercel import VercelProvider

    from .mock_openai import completion_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

BASE_URL = 'https://gateway.example/v1'


def gateway_client(request_capture: RequestCapture) -> httpx2.AsyncClient:
    """A client that answers each request with a text response, and records it in `request_capture`."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        message = ChatCompletionMessage(content='{"text": "hello"}', role='assistant')
        return httpx2.Response(200, json=completion_message(message).model_dump(mode='json'))

    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    )


@dataclass(frozen=True)
class ThinkingCase:
    id: str
    model_name: str
    thinking: bool | None
    expected: dict[str, JsonValue]


THINKING_CASES = [
    ThinkingCase(
        'nemotron-3-on', 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8', True, snapshot({'reasoning_effort': 'medium'})
    ),
    # On by default: leaving `thinking` unset sends no effort, so the model keeps its own default.
    ThinkingCase('nemotron-3-default', 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8', None, snapshot({})),
    # Hybrid: thinking can be turned off.
    ThinkingCase('hy3-off', 'tencent/Hy3', False, snapshot({'reasoning_effort': 'none'})),
    ThinkingCase('step-5-default', 'stepfun/step-5-preview', None, snapshot({})),
    ThinkingCase('step-5-on', 'stepfun/step-5-preview', True, snapshot({'reasoning_effort': 'medium'})),
    # Step 5 always reasons, and the family profile names no lowest effort, so turning thinking off sends nothing.
    ThinkingCase('step-5-off', 'stepfun/step-5-preview', False, snapshot({})),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in THINKING_CASES])
async def test_open_weight_family_thinking_wire_shape(
    allow_model_requests: None, request_capture: RequestCapture, case: ThinkingCase
):
    async with gateway_client(request_capture) as client:
        model = OpenAIChatModel(case.model_name, provider=TogetherProvider(api_key='test', http_client=client))
        settings: ModelSettings = {} if case.thinking is None else {'thinking': case.thinking}
        await Agent(model).run('hello', model_settings=settings)

    body = request_capture.body('/chat/completions')
    assert {key: value for key, value in body.items() if key == 'reasoning_effort'} == case.expected


@dataclass(frozen=True)
class VercelEffortCase:
    id: str
    model_name: str
    thinking: ThinkingLevel
    expected: dict[str, JsonValue]


VERCEL_EFFORT_CASES = [
    # Vercel lists Hy3's efforts as `none`, `low` and `high`, so a unified level goes to the lowest of those at or above it.
    VercelEffortCase('hy3-on', 'tencent/hy3', True, snapshot({'reasoning_effort': 'high'})),
    VercelEffortCase('hy3-medium', 'tencent/hy3', 'medium', snapshot({'reasoning_effort': 'high'})),
    VercelEffortCase('hy3-low', 'tencent/hy3', 'low', snapshot({'reasoning_effort': 'low'})),
    VercelEffortCase('hy3-off', 'tencent/hy3', False, snapshot({'reasoning_effort': 'none'})),
    # Step 5 always reasons and Vercel lists `low` as its lowest effort.
    VercelEffortCase('step-5-off', 'stepfun/step-5-preview', False, snapshot({'reasoning_effort': 'low'})),
    # Vercel's catalog lists no reasoning controls for Nemotron 3 Super, so no effort goes out.
    VercelEffortCase('nemotron-3-super-on', 'nvidia/nemotron-3-super-120b-a12b', True, snapshot({})),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in VERCEL_EFFORT_CASES])
async def test_vercel_open_weight_family_sends_listed_efforts(
    allow_model_requests: None, request_capture: RequestCapture, case: VercelEffortCase
):
    """Vercel's docs don't say what it does with an effort the model's catalog entry doesn't list, so only listed
    efforts go out: https://vercel.com/docs/ai-gateway/models-and-providers/reasoning."""
    async with gateway_client(request_capture) as client:
        model = OpenAIChatModel(case.model_name, provider=VercelProvider(api_key='test', http_client=client))
        await Agent(model).run('hello', model_settings={'thinking': case.thinking})

    body = request_capture.body('/chat/completions')
    assert {key: value for key, value in body.items() if key == 'reasoning_effort'} == case.expected


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize('provider_name', ['openai-compatible', 'litellm'])
async def test_open_weight_family_native_output(
    allow_model_requests: None, request_capture: RequestCapture, provider_name: str
):
    """The family profile keeps the OpenAI-compatible default under it, so `NativeOutput` still sends a JSON schema."""
    async with gateway_client(request_capture) as client:
        provider = (
            OpenAICompatibleProvider('gateway', base_url=BASE_URL, api_key='test', http_client=client)
            if provider_name == 'openai-compatible'
            else LiteLLMProvider(api_base=BASE_URL, api_key='test', http_client=client)
        )
        model = OpenAIChatModel('nvidia/nemotron-3-super-120b-a12b', provider=provider)
        result = await Agent(model, output_type=NativeOutput(Answer)).run('hello', model_settings={'thinking': True})

    assert result.output == Answer(text='hello')
    body = request_capture.body('/chat/completions')
    assert {key: body[key] for key in ('reasoning_effort', 'response_format')} == snapshot(
        {
            'reasoning_effort': 'medium',
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'Answer',
                    'schema': {
                        'properties': {'text': {'type': 'string'}},
                        'required': ['text'],
                        'type': 'object',
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
        }
    )


@dataclass(frozen=True)
class OpenRouterCase:
    id: str
    model_name: str
    thinking: bool
    expected: dict[str, JsonValue]


OPENROUTER_CASES = [
    OpenRouterCase(
        'nemotron-3',
        'nvidia/nemotron-3-super-120b-a12b',
        True,
        snapshot({'reasoning_effort': 'medium', 'reasoned': True}),
    ),
    OpenRouterCase(
        'nemotron-3.5',
        'nvidia/nemotron-3.5-lightning',
        True,
        snapshot({'reasoning_effort': 'medium', 'reasoned': True}),
    ),
    OpenRouterCase(
        'inkling', 'thinkingmachines/inkling', True, snapshot({'reasoning_effort': 'medium', 'reasoned': True})
    ),
    OpenRouterCase('mimo', 'xiaomi/mimo-v2.6-flash', True, snapshot({'reasoning_effort': 'medium', 'reasoned': True})),
    OpenRouterCase(
        'step-5', 'stepfun/step-5-preview', True, snapshot({'reasoning_effort': 'medium', 'reasoned': True})
    ),
    OpenRouterCase(
        'step-5-off', 'stepfun/step-5-preview', False, snapshot({'reasoning_effort': None, 'reasoned': True})
    ),
    OpenRouterCase('hy3', 'tencent/hy3', True, snapshot({'reasoning_effort': 'medium', 'reasoned': True})),
    OpenRouterCase(
        'ling-3', 'inclusionai/ling-3.1-flash', True, snapshot({'reasoning_effort': 'medium', 'reasoned': True})
    ),
]


@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in OPENROUTER_CASES])
async def test_open_weight_family_thinking_on_openrouter(
    allow_model_requests: None, openrouter_api_key: str, request_capture: RequestCapture, case: OpenRouterCase
):
    """`OpenAICompatibleProvider` sends the family profile's `reasoning_effort` as a top-level field, which OpenRouter's
    Chat Completions API takes as "Shorthand for setting reasoning effort. Equivalent to setting reasoning.effort":
    https://openrouter.ai/docs/api-reference/chat-completion.

    Each case pins the effort sent and whether the response reasoned (reasoning tokens or a `ThinkingPart`).
    """
    provider = OpenAICompatibleProvider(
        'gateway',
        base_url='https://openrouter.ai/api/v1',
        api_key=openrouter_api_key,
        http_client=request_capture.http_client(timeout=120),
    )
    agent = Agent(OpenAIChatModel(case.model_name, provider=provider))
    result = await agent.run('What is 17 * 23? Reply with the number only.', model_settings={'thinking': case.thinking})

    body = request_capture.body('/chat/completions')
    thinking_parts = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ThinkingPart)
    ]
    reasoned = result.usage.details.get('reasoning_tokens', 0) > 0 or bool(thinking_parts)
    assert {'reasoning_effort': body.get('reasoning_effort'), 'reasoned': reasoned} == case.expected
