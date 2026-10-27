"""Tests for `CompactionRequestedEvent` and `request_compaction`.

A capability asks for compaction with `request_compaction`; the compaction capabilities that listen
replace the request's history, and the first one to do so handles the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CompactionRequestedEvent, on_event, request_compaction
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext

from .conftest import RequestCapture, try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


@dataclass
class Requester(AbstractCapability[None]):
    """Requests compaction before every request that follows a model response."""

    events: list[CompactionRequestedEvent] = field(default_factory=list[CompactionRequestedEvent])

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if any(isinstance(message, ModelResponse) for message in request_context.messages):
            self.events.append(await request_compaction(ctx, request_context, reason='test'))
        return request_context


@dataclass
class KeepLastRequest(AbstractCapability[None]):
    """A compaction that keeps only the request being sent."""

    name: str = 'KeepLastRequest'

    @on_event(CompactionRequestedEvent)
    async def compact(self, ctx: RunContext[None], event: CompactionRequestedEvent) -> None:
        if event.compacted_by is None:
            event.request_context.messages = event.request_context.messages[-1:]
            event.compacted_by = self.name


def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(f'saw {len(messages)} messages')])


def _agent(model: Model, *capabilities: AbstractCapability[None], instructions: str | None = None) -> Agent[None, str]:
    return Agent(model, deps_type=type(None), instructions=instructions, capabilities=list(capabilities))


async def test_a_listener_compacts_and_the_run_keeps_the_history() -> None:
    """The compacted history goes out with the request and replaces the run's history."""
    requester = Requester()
    agent = _agent(FunctionModel(_reply), requester, KeepLastRequest())
    first = await agent.run('one')
    second = await agent.run('two', message_history=first.all_messages())

    assert second.output == 'saw 1 messages'
    assert [(event.reason, event.compacted_by) for event in requester.events] == [('test', 'KeepLastRequest')]
    assert len(second.all_messages()) == 2


async def test_without_a_listener_nothing_changes() -> None:
    requester = Requester()
    agent = _agent(FunctionModel(_reply), requester)
    first = await agent.run('one')
    second = await agent.run('two', message_history=first.all_messages())

    assert second.output == 'saw 3 messages'
    assert [event.compacted_by for event in requester.events] == [None]


def test_request_context_is_only_available_to_listeners() -> None:
    with pytest.raises(UserError, match='only available to its listeners'):
        CompactionRequestedEvent(reason='test').request_context


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_compaction_declines_another_providers_model() -> None:
    """`OpenAICompaction` compacts through OpenAI's endpoint, so it leaves another provider's request alone."""
    requester = Requester()
    agent = _agent(FunctionModel(_reply), requester, OpenAICompaction())
    first = await agent.run('one')
    second = await agent.run('two', message_history=first.all_messages())

    assert second.output == 'saw 3 messages'
    assert [event.compacted_by for event in requester.events] == [None]


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='The requested compaction replaces the history with a compaction item.')
@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_compaction_handles_a_request(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    """`OpenAICompaction` honours a request through OpenAI's `/responses/compact`, once per request."""
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = OpenAIResponsesModel('gpt-5.6', provider=provider)
    requester = Requester()
    agent = _agent(
        model,
        requester,
        # Its own trigger fires for every request too: having compacted on the request, it doesn't compact again.
        OpenAICompaction(trigger=lambda messages: len(messages) > 1),
        instructions='You are a helpful math assistant. Give short answers.',
    )
    first = await agent.run('What is 2+2?')
    second = await agent.run('And 3+3?', message_history=first.all_messages())

    assert '6' in second.output
    assert [event.compacted_by for event in requester.events] == ['OpenAICompaction']
    assert len(request_capture.bodies('/responses/compact')) == 1
    messages = second.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    first_message = messages[0]
    assert isinstance(first_message, ModelResponse)
    assert isinstance(first_message.parts[0], CompactionPart)
