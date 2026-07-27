"""Mid-conversation system prompts on Anthropic.

The Anthropic Messages API accepts a `{'role': 'system'}` entry inside `messages`, so an application
can add an operator instruction partway through a conversation without rewriting the top-level
`system` parameter (and with it the whole cached prefix). Pydantic AI renders a `SystemPromptPart`
outside the first `ModelRequest` that way when the model and client support it, and keeps the
`<system>`-tagged user rendering everywhere else.

These tests pin both sides of the `supports_inline_system_prompts` profile flag, the placement
fallback for the positions the API rejects, the client-transport gate, and the cache breakpoint that
now lands on the new message.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicVertex
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

INSTRUCTION = 'From now on, every suggestion must include explicit type annotations.'


def message_history() -> list[ModelMessage]:
    """A reviewed-once conversation that a new operator instruction is appended to.

    The trailing request carries only the `SystemPromptPart`; the agent's own user prompt merges
    into it, which is the shape `ctx.enqueue(SystemPromptPart(...))` produces as well.
    """
    return [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a code reviewer.'),
                UserPromptPart(content='Review `def add(a, b): return a + b`.'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='Looks fine.')]),
        ModelRequest(parts=[SystemPromptPart(content=INSTRUCTION)]),
    ]


@dataclass(frozen=True)
class Case:
    """A model on one side of the `supports_inline_system_prompts` flag, and the wire it produces."""

    id: str
    model: str
    expected_messages: list[dict[str, Any]]


CASES = [
    Case(
        id='inline-system-supported',
        model='claude-sonnet-5',
        expected_messages=snapshot(
            [
                {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
                {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
                {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
                {
                    'role': 'system',
                    'content': [
                        {
                            'text': 'From now on, every suggestion must include explicit type annotations.',
                            'type': 'text',
                        }
                    ],
                },
            ]
        ),
    ),
    Case(
        id='inline-system-unsupported',
        model='claude-sonnet-4-6',
        expected_messages=snapshot(
            [
                {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
                {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': '<system>From now on, every suggestion must include explicit type annotations.</system>',
                            'type': 'text',
                        },
                        {'text': 'Review it again.', 'type': 'text'},
                    ],
                },
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_mid_conversation_system_prompt(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, case: Case
):
    """A non-leading `SystemPromptPart` gets its own `system` entry only on models that accept the role.

    Both cases send the same history, so the snapshots show the only difference: Sonnet 5 gets a
    fourth `{'role': 'system'}` message, Sonnet 4.6 gets the instruction folded into the user turn as
    `<system>`-tagged text. The leading system prompt hoists to the top-level `system` parameter
    either way — the point of the feature is that adding the instruction leaves it untouched.

    Both recordings show the model acting on the instruction, so the annotated signature is asserted
    on the output: the two renderings are equivalent in effect, they differ in what they cost.
    """
    agent = Agent(AnthropicModel(case.model, provider=AnthropicProvider(api_key=anthropic_api_key)))

    result = await agent.run('Review it again.', message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == case.expected_messages


async def test_mid_conversation_system_prompt_takes_cache_breakpoint(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
):
    """`anthropic_cache_messages` puts its breakpoint on the trailing `system` entry, and the API takes it.

    The setting caches the last content block of the last message, which the new entry now is, so this
    pins that a `cache_control` on a `system`-role block isn't rejected — the shape is only reachable
    since mid-conversation system prompts started getting their own message.
    """
    agent = Agent(
        AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key)),
        model_settings=AnthropicModelSettings(anthropic_cache_messages=True),
    )

    result = await agent.run('Review it again.', message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert body['messages'][-1] == snapshot(
        {
            'role': 'system',
            'content': [
                {
                    'text': 'From now on, every suggestion must include explicit type annotations.',
                    'type': 'text',
                    'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                }
            ],
        }
    )


async def test_mid_conversation_system_prompt_without_user_turn(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
):
    """Without a user turn to follow, the instruction degrades to the `<system>`-tagged rendering.

    Anthropic rejects a `system` entry that directly follows an assistant turn, so a system prompt
    that lands at the end of the history on its own — here, a run with no new user prompt — must not
    be sent as one. The request has to keep succeeding, which is what the recording proves.

    The recorded reply spells out what the degradation costs: the model reads the tagged text as "a
    stated preference from you rather than a higher-privilege instruction", and follows it anyway.
    """
    agent = Agent(AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key)))

    result = await agent.run(message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {
                        'text': '<system>From now on, every suggestion must include explicit type annotations.</system>',
                        'type': 'text',
                    }
                ],
            },
        ]
    )


async def test_mid_conversation_system_prompt_unsupported_client(allow_model_requests: None):
    """Vertex AI doesn't serve the `system` role, so a supported model still gets the `<system>` wrap.

    The transport isn't visible to `AnthropicProvider.model_profile()`, which is a `staticmethod`, so
    the gate lives in the model adapter. Mocked rather than recorded: we have no Vertex credentials,
    and the assertion is about the request we build, not the response.
    """
    response = BetaMessage(
        id='msg_1',
        content=[BetaTextBlock(text='Understood.', type='text')],
        model='claude-sonnet-5',
        role='assistant',
        type='message',
        usage=BetaUsage(input_tokens=5, output_tokens=3),
    )
    client = MagicMock()
    client.__class__ = AsyncAnthropicVertex  # pyright: ignore[reportAttributeAccessIssue]
    client.base_url = 'https://us-central1-aiplatform.googleapis.com'
    client.beta.messages.create = AsyncMock(return_value=response)

    agent = Agent(AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(anthropic_client=client)))
    await agent.run('Review it again.', message_history=message_history())

    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs['system'] == 'You are a code reviewer.'
    assert [message['role'] for message in kwargs['messages']] == ['user', 'assistant', 'user']
    assert kwargs['messages'][-1]['content'][0]['text'] == f'<system>{INSTRUCTION}</system>'
