"""Shared helper for the gateway Kimi regression tests: capture the request body of one agent run."""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx

from pydantic_ai import Agent
from pydantic_ai.models import allow_model_requests
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

_COMPLETION = {
    'id': 'chatcmpl-kimi-test',
    'object': 'chat.completion',
    'created': 1700000000,
    'model': 'kimi',
    'choices': [
        {
            'index': 0,
            'message': {'role': 'assistant', 'content': 'world'},
            'logprobs': None,
            'finish_reason': 'stop',
        }
    ],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}


def capture_request_body(provider_factory: Callable[..., Any], model_name: str) -> dict[str, Any]:
    """Run one agent request against `model_name` through a mocked transport and return the request body."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_COMPLETION)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = provider_factory(http_client=http_client)
    model = OpenAIChatModel(model_name, provider=provider)
    agent = Agent(model)
    with allow_model_requests():
        agent.run_sync('Hello', model_settings=ModelSettings(thinking=True))
    return bodies[0]
