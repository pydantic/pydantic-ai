"""Thinking requests for the open-weight families the family map resolves on OpenAI-compatible gateways.

Not VCR tests: there are no recordings for these gateways, and the claims are about the request the code builds,
which cassette matching doesn't check. A `MockTransport` serves synthetic responses, and `request_capture` records
the requests. See https://github.com/pydantic/pydantic-ai/issues/12096.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
import pytest
from pydantic import JsonValue

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.together import TogetherProvider

    from .mock_openai import completion_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def gateway_client(request_capture: RequestCapture) -> httpx2.AsyncClient:
    """A client that answers each request with a text response, and records it in `request_capture`."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        message = ChatCompletionMessage(content='hello', role='assistant')
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
