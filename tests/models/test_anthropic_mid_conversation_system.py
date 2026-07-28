"""Mid-conversation system prompts on Anthropic.

The Anthropic Messages API accepts a `{'role': 'system'}` entry inside `messages`, so an application
can add an operator instruction partway through a conversation without rewriting the top-level
`system` parameter (and with it the whole cached prefix). Pydantic AI renders a `SystemPromptPart`
outside the first `ModelRequest` that way when the model and client support it, and keeps the
`<system>`-tagged user rendering everywhere else.

The API accepts the entry only directly behind a user turn and directly ahead of an assistant turn
(or the end of the array), so two transforms keep it legal without giving up its authority: a
minimal `.` user turn when nothing precedes it, and sliding it past user turns that ended up behind
it. These tests pin both, both sides of the `supports_inline_system_prompts` profile flag, the
client-transport gate, and the cache breakpoint that now lands on the new message.
"""

from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    UserPromptPart,
)

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicBedrock

    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.native_tools._tool_search import ToolSearchTool
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.tools import ToolDefinition

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

INSTRUCTION = 'From now on, every suggestion must include explicit type annotations.'


@pytest.fixture
def rendered_messages(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, Any]]]:
    """The `messages` array the adapter renders, per request, as it goes out.

    `single_request_body` reads the request stored in the *cassette*, and VCR is configured to match
    on method, path and host only, so a rendering change still replays its recording and keeps
    asserting the recorded body — the trap `tests/AGENTS.md` calls out. Every test below therefore
    also asserts that what the adapter produces today equals what the recording captured, which is
    what makes the snapshots evidence about live behavior rather than about the file.
    """
    rendered: list[list[dict[str, Any]]] = []
    map_message = AnthropicModel._map_message  # pyright: ignore[reportPrivateUsage]

    async def capture(self: AnthropicModel, *args: Any, **kwargs: Any) -> Any:
        system_prompt, anthropic_messages = await map_message(self, *args, **kwargs)
        rendered.append(cast('list[dict[str, Any]]', anthropic_messages))
        return system_prompt, anthropic_messages

    monkeypatch.setattr(AnthropicModel, '_map_message', capture)
    return rendered


@pytest.fixture
def anthropic_bedrock_client() -> AsyncAnthropicBedrock:
    """An `AsyncAnthropicBedrock` client authenticated by bearer token, defaulted for replay.

    The SDK reads `AWS_BEARER_TOKEN_BEDROCK` into `api_key`, and refuses it alongside SigV4
    credentials, so the token is the only auth this passes. `botocore` only ships under the
    `bedrock` extra.
    """
    pytest.importorskip('botocore')

    return AsyncAnthropicBedrock(
        api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'test-bedrock-token'),
        aws_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
    )


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


def message_history_with_paired_instruction() -> list[ModelMessage]:
    """The same conversation, with the instruction arriving alongside a user prompt.

    The pairing is what puts a native `system` entry directly behind the request's own user turn.
    Without one, the entry gets the `.` anchor instead (see `..._without_user_turn`), so these are
    the histories where `_place_system_messages_before_generation` decides where it ends up.
    """
    history = message_history()
    history[-1] = ModelRequest(
        parts=[SystemPromptPart(content=INSTRUCTION), UserPromptPart(content='Review it again.')]
    )
    return history


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
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
    rendered_messages: list[list[dict[str, Any]]],
    case: Case,
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
    assert rendered_messages == [body['messages']]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == case.expected_messages


async def test_mid_conversation_system_prompt_takes_cache_breakpoint(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_messages: list[list[dict[str, Any]]]
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
    assert rendered_messages == [body['messages']]
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
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_messages: list[list[dict[str, Any]]]
):
    """Without a user turn to follow, the instruction gets a minimal one rather than degrading.

    Anthropic rejects a `system` entry that directly follows an assistant turn, so a system prompt
    that lands at the end of the history on its own — here, a run with no new user prompt, the shape
    `ctx.enqueue(SystemPromptPart(...))` produces — has nothing legal to sit behind. A `.` user turn
    is the cheapest thing that satisfies the rule, and it asserts nothing on the user's behalf.

    The alternative was the `<system>`-tagged rendering, and the recording it replaced is why we
    don't: the model read the tagged text as "a stated preference from you rather than a
    higher-privilege instruction". Here it just complies.
    """
    agent = Agent(AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key)))

    result = await agent.run(message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert rendered_messages == [body['messages']]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': '.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
        ]
    )


async def test_mid_conversation_system_prompt_before_another_request(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_messages: list[list[dict[str, Any]]]
):
    """A system entry that a *user* turn would follow slides past it instead of degrading.

    `[user, system, user]` is rejected outright, and a second `ModelRequest` directly after the one
    carrying the instruction produces exactly that. `_merge_consecutive_messages` folds consecutive
    requests into one for agent runs, but leaves them unmerged when their instructions differ, and
    `Model.request` is callable directly — the adapter can't assume every request is followed by a
    response.

    Moving the entry past the user turn costs nothing: it governs the same generation either way, and
    it stays an operator instruction rather than becoming text the model can overrule.

    Driven through `Model.request` rather than `Agent.run` precisely because the agent's history
    cleaning would merge the two requests.
    """
    model = AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelRequest(parts=[UserPromptPart(content='Review it once more.')]),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def add(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_messages == [body['messages']]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it once more.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
        ]
    )


async def test_mid_conversation_system_prompt_kept_mid_history(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_messages: list[list[dict[str, Any]]]
):
    """An instruction with a real response after it keeps the native entry, mid-history.

    This is the shape a stored conversation replays on every later turn — the entry sits between the
    user turn it arrived with and the assistant turn it governed, and stays there as the history
    grows past it. The cheap alternative (only allow the entry in trailing position) would
    re-render the whole conversation's instructions on every request, which is the cache churn the
    feature exists to avoid.
    """
    model = AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelResponse(parts=[TextPart(content='def add(a: int, b: int) -> int: return a + b')]),
            ModelRequest(parts=[UserPromptPart(content='Now do `def mul(a, b): return a * b`.')]),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def mul(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_messages == [body['messages']]
    assert [message['role'] for message in body['messages']] == snapshot(
        ['user', 'assistant', 'user', 'system', 'assistant', 'user']
    )
    assert body['messages'][3] == snapshot(
        {
            'content': [
                {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
            ],
            'role': 'system',
        }
    )


async def test_mid_conversation_system_prompt_before_empty_response(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_messages: list[list[dict[str, Any]]]
):
    """A `ModelResponse` that renders to nothing doesn't count as the assistant turn to follow.

    A response whose parts all drop out — here an empty `TextPart` — appends no assistant message,
    so a request after it lands directly behind the system entry and the API rejects the whole
    thing. Reading ahead in the message list would call this placement legal; only checking what was
    actually rendered gets it right, which is why the decision is a pass over the wire messages.

    This is the case that makes the pass necessary rather than merely convenient: the history here
    is byte-identical to `..._kept_mid_history` apart from the response being empty, and that one
    leaves its entry where it is.
    """
    model = AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelResponse(parts=[TextPart(content='')]),
            ModelRequest(parts=[UserPromptPart(content='Review it once more.')]),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def add(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_messages == [body['messages']]
    assert [message['role'] for message in body['messages']] == snapshot(
        ['user', 'assistant', 'user', 'user', 'system']
    )
    assert body['messages'][-1] == snapshot(
        {
            'role': 'system',
            'content': [
                {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
            ],
        }
    )


async def test_two_mid_conversation_system_prompts_keep_their_order(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_messages: list[list[dict[str, Any]]]
):
    """Two instructions that both have to move end up adjacent, in the order they were given.

    Each slides only past *user* turns, so the earlier one stops behind the later one rather than
    overtaking it — sliding past everything would invert them, and "ignore the previous instruction"
    cases would then resolve backwards. Consecutive `system` entries are a placement the API takes:
    the group as a whole still precedes the generation, and the recording shows both obeyed.
    """
    model = AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Also always state the time complexity.'),
                    UserPromptPart(content='Review it once more.'),
                ]
            ),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def add(a: int, b: int) -> int:' in reply.content
    assert 'O(1)' in reply.content

    body = single_request_body(vcr)
    assert rendered_messages == [body['messages']]
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it once more.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
            {'role': 'system', 'content': [{'text': 'Also always state the time complexity.', 'type': 'text'}]},
        ]
    )


async def test_mid_conversation_system_prompt_unsupported_client(
    allow_model_requests: None,
    anthropic_bedrock_client: AsyncAnthropicBedrock,
    vcr: Cassette,
    rendered_messages: list[list[dict[str, Any]]],
):
    """Bedrock doesn't serve the `system` role, so a supported model still gets the `<system>` wrap.

    `us.anthropic.claude-sonnet-5` normalizes to a model name the profile flag covers, so this
    isolates the transport gate from the model gate: same model, same history as the supported case
    above, and the instruction still ends up `<system>`-tagged because of the client it's sent
    through. Vertex AI and Microsoft Foundry share the branch.
    """
    model = AnthropicModel(
        'us.anthropic.claude-sonnet-5', provider=AnthropicProvider(anthropic_client=anthropic_bedrock_client)
    )
    # The Bedrock provider segment is stripped before the prefix check, so the model half of the
    # gate is open here and the transport half is the only thing left to close it.
    assert model.profile.get('supports_inline_system_prompts') is True

    result = await Agent(model).run('Review it again.', message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert rendered_messages == [body['messages']]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}], 'role': 'user'},
            {'content': [{'text': 'Looks fine.', 'type': 'text'}], 'role': 'assistant'},
            {
                'content': [
                    {
                        'text': '<system>From now on, every suggestion must include explicit type annotations.</system>',
                        'type': 'text',
                    },
                    {'text': 'Review it again.', 'type': 'text'},
                ],
                'role': 'user',
            },
        ]
    )


async def test_native_tool_availability_delta(allow_model_requests: None, anthropic_api_key: str, vcr: Cassette):
    """Supported models accept a framework tool reveal and can call the newly available tool.

    A delta arriving on its own has the same problem a lone system prompt does — nothing legal to
    sit behind — and takes the same `.` anchor, rather than the bespoke `<tool-availability-change>`
    user message it used to get. The `tool_addition` block already says what changed; a second,
    vaguer statement of it in the user's voice added nothing.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))
    tool = ToolDefinition(
        name='lookup_refund_policy',
        description='Look up the refund policy for an order.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'order_id': {'type': 'string'}},
            'required': ['order_id'],
        },
        defer_loading=True,
        with_native='tool_search',
    )
    old_tool = ToolDefinition(
        name='old_refund_tool',
        description='Old refund lookup.',
        parameters_json_schema={'type': 'object', 'properties': {}},
    )

    await model.request(
        [
            ModelRequest(parts=[UserPromptPart(content='I need help with a refund.')]),
            ModelResponse(parts=[TextPart(content='I can check that.')]),
            ModelRequest(
                parts=[
                    ToolAvailabilityDeltaPart(
                        added=['lookup_refund_policy'],
                        removed=['old_refund_tool'],
                    )
                ]
            ),
        ],
        None,
        ModelRequestParameters(function_tools=[tool, old_tool], native_tools=[ToolSearchTool()]),
    )

    body = single_request_body(vcr)
    assert body['messages'][-2] == snapshot({'role': 'user', 'content': [{'text': '.', 'type': 'text'}]})
    assert body['messages'][-1] == snapshot(
        {
            'role': 'system',
            'content': [
                {
                    'type': 'tool_addition',
                    'tool': {
                        'type': 'tool_reference',
                        'name': 'lookup_refund_policy',
                    },
                },
                {
                    'type': 'tool_removal',
                    'tool': {'type': 'tool_reference', 'name': 'old_refund_tool'},
                },
            ],
        }
    )
