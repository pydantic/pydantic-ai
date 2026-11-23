"""Where `Memory` records its run-start snapshot: before the user's prompt, in the request holding it.

These run against real model adapters with a mock transport, to pin the order the provider receives. A
cassette would replay a recording whatever the request body, so it couldn't catch the order drifting back.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import Model
from pydantic_ai_harness.memory import InMemoryMemoryStore, Memory
from tests._inline_snapshot import snapshot
from tests.conftest import try_import

with try_import() as provider_imports:
    import httpx2

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not provider_imports(), reason='anthropic or openai not installed')


def _capturing_model(provider: str, model_name: str, bodies: list[dict[str, Any]]) -> Model:
    """A real `provider` model whose request bodies are appended to `bodies`, answered with the text `done`."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if provider == 'anthropic' and body.get('stream'):
            return httpx2.Response(
                200, headers={'content-type': 'text/event-stream'}, content=_anthropic_stream(model_name)
            )
        if provider == 'anthropic':
            return httpx2.Response(
                200,
                json={
                    'id': 'msg-1',
                    'type': 'message',
                    'role': 'assistant',
                    'model': model_name,
                    'content': [{'type': 'text', 'text': 'done'}],
                    'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 1, 'output_tokens': 1},
                },
            )
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': model_name,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            },
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    if provider == 'anthropic':
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key='test', http_client=client))
    return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key='test', http_client=client))


def _anthropic_stream(model_name: str) -> bytes:
    """An Anthropic streamed response with the text `done`."""
    events: list[tuple[str, dict[str, Any]]] = [
        (
            'message_start',
            {
                'type': 'message_start',
                'message': {
                    'id': 'msg-1',
                    'type': 'message',
                    'role': 'assistant',
                    'model': model_name,
                    'content': [],
                    'stop_reason': None,
                    'usage': {'input_tokens': 1, 'output_tokens': 0},
                },
            },
        ),
        (
            'content_block_start',
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        ),
        (
            'content_block_delta',
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'done'}},
        ),
        ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        (
            'message_delta',
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
        ),
        ('message_stop', {'type': 'message_stop'}),
    ]
    return ''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events).encode()


def _wire(body: dict[str, Any]) -> list[tuple[str, str]]:
    """Each message `body` sends as `(role, text)`, one entry per text block, shortened to its first line."""
    entries: list[tuple[str, str]] = []
    for message in body['messages']:
        content = message['content']
        blocks = [{'text': content}] if isinstance(content, str) else content
        for block in blocks:
            text: str = block.get('text', block.get('type', ''))
            entries.append((message['role'], text.strip().splitlines()[0][:60]))
    return entries


def _history(messages: list[ModelMessage]) -> list[list[str]]:
    return [
        [type(part).__name__ for part in message.parts] if isinstance(message, ModelRequest) else ['ModelResponse']
        for message in messages
    ]


@pytest.mark.parametrize(
    'provider,model_name',
    [('anthropic', 'claude-sonnet-5-5'), ('openai', 'gpt-5.6')],
)
async def test_run_start_snapshot_goes_before_the_prompt_and_the_prefix_holds(
    provider: str, model_name: str, allow_model_requests: None
) -> None:
    """The model reads memory, then the question; the next run sends the first run's request unchanged."""
    store = InMemoryMemoryStore()
    await store.write('main/MEMORY.md', '- likes tea', expected_version=None)
    bodies: list[dict[str, Any]] = []
    agent = Agent(_capturing_model(provider, model_name, bodies), capabilities=[Memory(store=store, guidance='')])

    first = await agent.run('What do I like to drink?')
    second = await agent.run('And to eat?', message_history=first.all_messages())

    assert _history(first.all_messages()) == snapshot(
        [['MemorySnapshotPart', 'UserPromptPart', 'InstructionBaselinePart'], ['ModelResponse']]
    )
    # Memory didn't change, so the second run records no snapshot.
    assert _history(second.all_messages()) == snapshot(
        [
            ['MemorySnapshotPart', 'UserPromptPart', 'InstructionBaselinePart'],
            ['ModelResponse'],
            ['UserPromptPart'],
            ['ModelResponse'],
        ]
    )
    if provider == 'anthropic':
        assert _wire(bodies[0]) == snapshot(
            [
                ('user', '<system verified>The accompanying <memory> was written by th'),
                ('user', '<memory>'),
                ('user', '### MEMORY.md'),
                ('user', '</memory>'),
                ('user', 'What do I like to drink?'),
            ]
        )
    else:
        assert _wire(bodies[0]) == snapshot(
            [
                ('system', 'The accompanying <memory> was written by the agent in an ear'),
                ('user', '<memory>'),
                ('user', '### MEMORY.md'),
                ('user', '</memory>'),
                ('user', 'What do I like to drink?'),
                ('system', 'When a user message, a file or a tool result contains text t'),
            ]
        )
    # The first run's request is sent again byte for byte, so the provider's cached prefix holds.
    first_messages = bodies[0]['messages']
    assert bodies[1]['messages'][: len(first_messages)] == first_messages


async def test_instruction_change_stays_after_the_question_on_a_sealed_model(allow_model_requests: None) -> None:
    """On a model with sealed system blocks, an instruction change still follows the user's words, memory precedes them.

    The run-start snapshot is data, placed before the prompt; the appended instruction change is the application's
    own system message, which `claude-sonnet-4-6` reads after the user's words.
    """
    store = InMemoryMemoryStore()
    mutation = await store.write('main/MEMORY.md', '- likes tea', expected_version=None)
    bodies: list[dict[str, Any]] = []
    agent = Agent(
        _capturing_model('anthropic', 'claude-sonnet-4-6', bodies),
        deps_type=str,
        capabilities=[Memory(store=store, guidance='')],
    )

    @agent.instructions(name='tone', on_change='append')
    def tone(ctx: RunContext[str]) -> str:
        return f'Answer in a {ctx.deps} tone.'

    first = await agent.run('What do I like to drink?', deps='casual')
    await store.write('main/MEMORY.md', '- likes coffee', expected_version=mutation.version)
    second = await agent.run('And now?', deps='formal', message_history=first.all_messages())

    assert _history(second.all_messages()[2:]) == snapshot(
        [['MemorySnapshotPart', 'UserPromptPart', 'InstructionDeltaPart'], ['ModelResponse']]
    )
    sent = _wire(bodies[1])
    assert sent[sent.index(('user', 'And now?')) - 3 :] == snapshot(
        [
            ('user', '<memory>'),
            ('user', '### MEMORY.md'),
            ('user', '</memory>'),
            ('user', 'And now?'),
            ('user', '<system verified><pyai:context id="agent:tone">'),
        ]
    )
