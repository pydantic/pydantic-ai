"""Check the Python host against the runs of the real `pydantic-ai`.

Each scenario in `fixtures/runs.json` is replayed through the host. The transport checks that the host sends
the request `pydantic-ai` sent, then answers with the response `pydantic-ai` got.

Run it from the repository root: `uv run python experiments/pydantic-ai-core/python/check.py`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
from pydantic import JsonValue
from pydantic_ai_core import Agent, CoreError

from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse

RUNS = Path(__file__).parent.parent / 'fixtures/runs.json'


async def check_scenario(scenario: dict[str, Any]) -> int:
    """Replay one scenario through the host and return the number of requests it made."""
    steps = iter(scenario['steps'])
    tool_results = [result for step in scenario['steps'] for result in step.get('tool_results', [])]
    sent = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal sent
        sent += 1
        step = next(steps)
        assert json.loads(request.content) == step['request'], scenario['name']
        assert request.url == 'https://api.openai.com/v1/chat/completions'
        return httpx2.Response(200, json=step['response'])

    start = scenario['start']
    agent = Agent(
        start['model'],
        instructions=start['instructions'],
        api_key='-',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )

    async def get_temperature(city: str) -> dict[str, Any]:
        result = tool_results.pop(0)
        if 'retry' in result:
            raise ModelRetry(result['retry'])
        return result['content']

    if start['tools']:
        agent.tool_plain(get_temperature)

    history: list[ModelMessage] | None = None
    if 'message_history' in start:
        history = ModelMessagesTypeAdapter.validate_python(start['message_history'])
    result = await agent.run(start['user_prompt'], message_history=history)

    assert result.output == scenario['output'], scenario['name']
    assert result.usage == scenario['usage'], scenario['name']
    # The host returns real `pydantic-ai` messages, so they are compared after a round trip through its types.
    new_messages = result.messages[len(history or []) :]
    assert all(isinstance(message, ModelRequest | ModelResponse) for message in new_messages)
    dumped = ModelMessagesTypeAdapter.dump_python(new_messages, mode='json')
    assert contains(dumped, scenario['new_messages']), scenario['name']
    return sent


def contains(actual: JsonValue, expected: JsonValue) -> bool:
    """Whether `actual` has everything in `expected`: every key of a dict, and every item of a list in order."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and contains(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(map(contains, actual, expected))
    return actual == expected


async def check_error() -> None:
    """A provider error ends the run with the core's error."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, text='slow down')

    agent = Agent('gpt-4.1-mini', api_key='-', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    try:
        await agent.run('hello')
    except CoreError as error:
        assert str(error).startswith('status_code: 429'), error
    else:
        raise AssertionError('the run should have failed')


async def main() -> None:
    """Replay every scenario."""
    scenarios = json.loads(RUNS.read_text())['scenarios']
    requests = sum([await check_scenario(scenario) for scenario in scenarios])
    await check_error()
    print(f'{len(scenarios)} scenarios, {requests} requests, 0 failures')


if __name__ == '__main__':
    asyncio.run(main())
