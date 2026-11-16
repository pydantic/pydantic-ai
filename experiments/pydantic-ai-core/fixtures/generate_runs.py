"""Dump agent-run fixtures from the real `pydantic-ai`, for the Rust core and its Python and TypeScript hosts.

Each scenario runs a real `Agent` on `OpenAIChatModel`. The HTTP transport replays responses recorded from
the OpenAI API in an existing test cassette, and the requests `pydantic-ai` sends are captured.

Run it from the repository root: `uv run python experiments/pydantic-ai-core/fixtures/generate_runs.py`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import yaml
from openai import AsyncOpenAI

from pydantic_ai import Agent, ModelRetry, Tool
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

REPOSITORY = Path(__file__).parents[3]
CASSETTE = (
    REPOSITORY / 'tests/models/cassettes/test_openai/test_openai_instructions_with_tool_calls_keep_instructions.yaml'
)
MODEL = 'gpt-4.1-mini'
INSTRUCTIONS = 'You are a helpful assistant.'
RETRY_MESSAGE = 'The weather service is down.'


async def run_scenario(
    name: str,
    responses: list[dict[str, Any]],
    user_prompt: str,
    *,
    with_tool: bool = True,
    retry_first_call: bool = False,
    message_history: list[ModelMessage] | None = None,
) -> tuple[dict[str, Any], list[ModelMessage]]:
    """Run one scenario on the real `Agent`, and return its fixture and the messages of the run."""
    requests: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    scripted = iter(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=next(scripted))

    client = AsyncOpenAI(api_key='-', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    agent = Agent(OpenAIChatModel(MODEL, provider=OpenAIProvider(openai_client=client)), instructions=INSTRUCTIONS)

    async def get_temperature(city: str) -> dict[str, Any]:
        if retry_first_call and not tool_results:
            tool_results.append({'retry': RETRY_MESSAGE})
            raise ModelRetry(RETRY_MESSAGE)
        # An integer, not `20.0`: JavaScript has one number type, so the TypeScript host would send back `20`.
        temperature = {'city': city, 'celsius': 20}
        tool_results.append({'content': temperature})
        return temperature

    tool_def = Tool(get_temperature).tool_def
    if with_tool:
        agent.tool_plain(get_temperature)

    result = await agent.run(user_prompt, message_history=message_history)

    results = iter(tool_results)
    steps: list[dict[str, Any]] = []
    for request, response in zip(requests, responses, strict=True):
        step: dict[str, Any] = {'request': request, 'response': response}
        if tool_calls := response['choices'][0]['message'].get('tool_calls'):
            step['tool_calls'] = [
                {
                    'tool_call_id': call['id'],
                    'tool_name': call['function']['name'],
                    'args': call['function']['arguments'],
                }
                for call in tool_calls
            ]
            step['tool_results'] = [{'tool_call_id': call['id'], **next(results)} for call in tool_calls]
        steps.append(step)

    start: dict[str, Any] = {
        'type': 'start',
        'model': MODEL,
        'instructions': INSTRUCTIONS,
        'user_prompt': user_prompt,
        'tools': [
            {
                'name': tool_def.name,
                'description': tool_def.description,
                'parameters_json_schema': tool_def.parameters_json_schema,
                'strict': tool_def.strict,
            }
        ]
        if with_tool
        else [],
    }
    if message_history is not None:
        # The history goes in exactly as `pydantic-ai` serializes it.
        start['message_history'] = ModelMessagesTypeAdapter.dump_python(message_history, mode='json')

    usage = result.usage
    fixture = {
        'name': name,
        'start': start,
        'steps': steps,
        'output': result.output,
        'usage': {'requests': usage.requests, 'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens},
        'new_messages': [
            project_message(message)
            for message in ModelMessagesTypeAdapter.dump_python(result.new_messages(), mode='json')
        ],
    }
    return fixture, result.all_messages()


def project_message(message: dict[str, Any]) -> dict[str, Any]:
    """The fields of a serialized message that the core models."""
    parts = [project_part(part) for part in message['parts']]
    if message['kind'] == 'request':
        return {'kind': 'request', 'parts': parts, 'instructions': message['instructions']}
    return {
        'kind': 'response',
        'parts': parts,
        'usage': {key: message['usage'][key] for key in ('input_tokens', 'output_tokens')},
        'model_name': message['model_name'],
        'provider_name': message['provider_name'],
        'provider_response_id': message['provider_response_id'],
        'finish_reason': message['finish_reason'],
    }


def project_part(part: dict[str, Any]) -> dict[str, Any]:
    """The fields of a serialized part that the core models."""
    keys = {
        'user-prompt': ('part_kind', 'content'),
        'text': ('part_kind', 'content'),
        'tool-call': ('part_kind', 'tool_name', 'args', 'tool_call_id'),
        'tool-return': ('part_kind', 'tool_name', 'content', 'tool_call_id'),
    }[part['part_kind']]
    projected = {key: part[key] for key in keys}
    if part.get('outcome', 'success') != 'success':
        projected['outcome'] = part['outcome']
        projected['retry'] = part['retry']
    return projected


async def main() -> None:
    """Write `runs.json` next to this file."""
    interactions = yaml.safe_load(CASSETTE.read_text())['interactions']
    tool_call, final_text = (interaction['response']['parsed_body'] for interaction in interactions)
    question = 'What is the temperature in Tokyo?'

    first, messages = await run_scenario('tool_call', [tool_call, final_text], question)
    scenarios = [
        first,
        (await run_scenario('tool_retry', [tool_call, tool_call, final_text], question, retry_first_call=True))[0],
        (await run_scenario('text_only', [final_text], question, with_tool=False))[0],
        (await run_scenario('message_history', [final_text], 'And in Paris?', message_history=messages))[0],
    ]

    runs = Path(__file__).parent / 'runs.json'
    runs.write_text(json.dumps({'scenarios': scenarios}, indent=1) + '\n')
    print(f'{len(scenarios)} scenarios, {sum(len(s["steps"]) for s in scenarios)} requests -> {runs}')


if __name__ == '__main__':
    asyncio.run(main())
