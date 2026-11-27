"""`NativeCompaction` against the providers' own compaction APIs.

Each test has a two-turn conversation compacted, from a capability or with `compact=` on the run, then asks a
question only the compacted history can answer. The `FunctionModel` tests in
`tests/test_capability_native_compaction.py` cover the mapping in detail.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from cassetter import Cassette

from pydantic_ai import Agent, CompactionPart, ModelMessage, ModelRequest, ModelResponse, RunContext, ToolReturnPart
from pydantic_ai.capabilities import AbstractCapability, NativeCompaction
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset

from .._inline_snapshot import snapshot
from ..cassette_utils import request_json
from ..conftest import RequestCapture, try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from .conftest import AnthropicModelFactory

pytestmark = [
    pytest.mark.vcr,
    pytest.mark.moves_cache_prefix(reason='compaction replaces the history the next request is built on'),
]


@dataclass
class RequestOnce(AbstractCapability[Any]):
    """Requests compaction for the run's first request."""

    async def before_run(self, ctx: RunContext[Any]) -> None:
        ctx.request_compaction(reason='threshold')


@pytest.fixture(
    params=[
        pytest.param('openai', marks=pytest.mark.skipif(not openai_available(), reason='openai not installed')),
        pytest.param(
            'anthropic', marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
        ),
    ]
)
def compacting_model(request: pytest.FixtureRequest, openai_api_key: str, anthropic_api_key: str) -> Model:
    if request.param == 'openai':
        return OpenAIResponsesModel('gpt-5.4-nano', provider=OpenAIProvider(api_key=openai_api_key))
    return AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))


def compaction_part(messages: list[ModelMessage]) -> CompactionPart:
    """The provider's compaction part the compacted history starts with."""
    first = messages[0]
    assert isinstance(first, ModelResponse)
    [part] = first.parts
    assert isinstance(part, CompactionPart)
    return part


def compacted_on_the_wire(vcr: Cassette, model: Model) -> dict[str, Any]:
    """Assert the recorded compaction request went out as the provider's compaction API, and return the follow-up's body.

    The follow-up is the request sent after the compaction, which carries the provider's compaction item first.
    """
    bodies = [(request.uri, request_json(request)) for request in vcr.requests]
    if model.system == 'openai':
        [compact] = [body for uri, body in bodies if uri.endswith('/responses/compact')]
        # `/responses/compact` answers with one `CompactedResponse`: the SDK's `responses.compact()` takes no `stream`.
        assert 'stream' not in compact
        follow_ups = [
            body
            for uri, body in bodies
            if uri.endswith('/responses') and body['input'] and body['input'][0].get('type') == 'compaction'
        ]
    else:
        [compact] = [body for _, body in bodies if 'compaction' in body]
        config = compact['compaction']
        assert config['type'] == 'summarize'
        # No focus, so Anthropic's own summarization prompt; a focus is asserted in
        # `test_run_compact_sends_the_conversations_own_request`.
        assert 'instructions' not in config
        follow_ups = [
            body
            for _, body in bodies
            if 'compaction' not in body
            and body['messages'][0]['role'] == 'assistant'
            and body['messages'][0]['content'][0]['type'] == 'compaction'
        ]
    [follow_up] = follow_ups
    return follow_up


async def test_native_compaction_in_a_run(allow_model_requests: None, compacting_model: Model, vcr: Cassette):
    agent = Agent(compacting_model, instructions='Answer in one short sentence.', capabilities=[NativeCompaction()])
    first = await agent.run('My name is Bob and my favorite color is teal.')

    second = await agent.run(
        'What is my favorite color?', message_history=first.all_messages(), capabilities=[RequestOnce()]
    )

    assert 'teal' in second.output.lower()
    messages = second.all_messages()
    assert compaction_part(messages).provider_name == compacting_model.system
    [record] = second.compactions
    assert (record.handler, record.reason, record.messages_before, record.messages_after, record.kept_from) == snapshot(
        ('native', 'threshold', 3, 2, 1)
    )
    assert record.summarized == first.all_messages()
    assert len(messages) == 3
    compacted_on_the_wire(vcr, compacting_model)


async def test_run_compact(allow_model_requests: None, compacting_model: Model, vcr: Cassette):
    """`compact=True` compacts the history before the run's first request, which is answered from it."""
    agent = Agent(compacting_model, instructions='Answer in one short sentence.', capabilities=[NativeCompaction()])
    first = await agent.run('My name is Bob and my favorite color is teal.')

    second = await agent.run('What is my favorite color?', message_history=first.all_messages(), compact=True)

    assert 'teal' in second.output.lower()
    assert compaction_part(second.all_messages()).provider_name == compacting_model.system
    [record] = second.compactions
    assert (record.handler, record.reason, record.model_name, record.messages_before, record.kept_from) == (
        'native',
        'manual',
        compacting_model.model_name,
        3,
        1,
    )
    assert record.summarized == first.all_messages()
    assert record.part == compaction_part(second.all_messages())
    compacted_on_the_wire(vcr, compacting_model)


async def test_native_compaction_with_a_tool_requesting_it(
    allow_model_requests: None, compacting_model: Model, vcr: Cassette
):
    """A tool requests compaction mid-run: the provider accepts its compaction item followed by the kept tool turn.

    The compacted history is the compaction response, then the response with the tool call and the request with its
    result, which the follow-up request is answered from.
    """
    toolset = FunctionToolset[Any]()

    @toolset.tool
    def compact_conversation(ctx: RunContext[Any]) -> str:
        """Compact the conversation, then report the user's favorite color."""
        ctx.request_compaction()
        return 'Compacted. Reminder: the favorite color is teal.'

    agent = Agent(
        compacting_model,
        instructions='Answer in one short sentence.',
        toolsets=[toolset],
        capabilities=[NativeCompaction()],
    )
    first = await agent.run('My name is Bob and my favorite color is teal.')

    second = await agent.run(
        'Call the compact_conversation tool, then tell me my favorite color.', message_history=first.all_messages()
    )

    assert 'teal' in second.output.lower()
    [record] = second.compactions
    assert (record.handler, record.reason, record.kept_from) == ('native', 'manual', 1)
    messages = second.all_messages()
    assert compaction_part(messages).provider_name == compacting_model.system
    _, call_response, result_request, final_response = messages
    assert isinstance(call_response, ModelResponse)
    assert [call.tool_name for call in call_response.tool_calls] == ['compact_conversation']
    assert isinstance(result_request, ModelRequest)
    [tool_return] = [part for part in result_request.parts if isinstance(part, ToolReturnPart)]
    assert tool_return.tool_call_id == call_response.tool_calls[0].tool_call_id
    assert isinstance(final_response, ModelResponse)
    # The provider accepted its compaction followed by the kept tool turn.
    follow_up = compacted_on_the_wire(vcr, compacting_model)
    if compacting_model.system == 'openai':
        assert [item.get('type') for item in follow_up['input']] == [
            'compaction',
            'function_call',
            'function_call_output',
        ]
    else:
        assert [
            (message['role'], [block['type'] for block in message['content']]) for message in follow_up['messages']
        ] == [
            ('assistant', ['compaction']),
            ('assistant', ['tool_use']),
            ('user', ['tool_result']),
        ]


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_run_compact_sends_the_conversations_own_request(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """The compaction request has the conversation's system prompt and tools, so it can read its cache.

    A focus reaches Anthropic as the compaction's `instructions`, inside a full summarization prompt.
    """
    agent = Agent(
        anthropic_model('claude-haiku-5-5', capture=True),
        instructions='Answer in one short sentence.',
        capabilities=[NativeCompaction()],
    )

    @agent.tool_plain
    def favorite_color() -> str:
        """Look up the user's favorite color."""
        return 'teal'  # pragma: no cover

    first = await agent.run('My name is Bob and my favorite color is teal.')
    second = await agent.run(
        'What is my favorite color?',
        message_history=first.all_messages(),
        compact={'focus': 'the favorite color'},
    )

    assert 'teal' in second.output.lower()
    conversation_body, compaction_body, _ = request_capture.bodies('/v1/messages')
    assert 'compaction' not in conversation_body
    assert compaction_body['system'] == conversation_body['system']
    assert compaction_body['tools'] == conversation_body['tools']
    compaction_config = compaction_body['compaction']
    assert isinstance(compaction_config, dict)
    assert compaction_config['type'] == 'summarize'
    assert 'the favorite color' in str(compaction_config['instructions'])
