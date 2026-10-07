"""Native steering through the real SDK and a deterministic local WebSocket peer."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

import anyio
import pytest

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelRetry,
    RunContext,
    SessionStateTypeAdapter,
    ToolOutput,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    BinaryImage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from ..conftest import try_import
from .test_openai_responses_ws import Peer, imports_successful, text_frames

with try_import():
    from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from .test_openai_responses_ws import peer as peer

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI or websockets not installed')


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('steered', [False, True])
async def test_native_successor_is_a_separate_response(peer: Peer, stream: bool, steered: bool):
    initial = text_frames('resp_a', 'Original')
    if steered:
        initial[-1]['type'] = 'response.incomplete'
        initial[-1]['response']['status'] = 'incomplete'
        initial[-1]['response']['incomplete_details'] = {'reason': 'steered'}
    peer.scripts = [
        initial[:2],
        [
            {
                'type': 'response.steer.accepted',
                'sequence_number': 2,
                'steer': {'id': 'steer_1', 'previous_response_id': 'resp_a'},
            },
            initial[-1],
            *text_frames('resp_b', 'Updated'),
        ],
    ]
    delivered: list[str] = []
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and not delivered:
                    delivered.append(await ctx.steer('New requirement'))
                    checkpoint = session.state
                    assert checkpoint.steering[0].status == 'sent'
                    # The active response is checkpointed, but native input is not history until committed.
                    assert len(checkpoint.conversation.messages) == 2
                    response = checkpoint.conversation.messages[-1]
                    assert isinstance(response, ModelResponse)
                    assert response.provider_response_id == 'resp_a'
                    assert response.state == 'incomplete'

        with anyio.fail_after(10):
            if stream:
                async with session.run_stream('Start', event_stream_handler=handle) as result:
                    assert await result.get_output() == 'Updated'
            else:
                assert (await session.run('Start', event_stream_handler=handle)).output == 'Updated'
        state = session.state
        assert len(state.conversation.messages) == 4
        assert [type(message) for message in state.conversation.messages] == [
            ModelRequest,
            ModelResponse,
            ModelRequest,
            ModelResponse,
        ]
        part = state.conversation.messages[2].parts[0]
        assert isinstance(part, UserPromptPart) and part.content == ['New requirement']
        assert state.conversation.usage.requests == 2
        assert state.conversation.usage.input_tokens == 6
        assert state.conversation.usage.output_tokens == 4
        delivery = state.steering[0]
        assert delivery.delivery_id == delivered[0]
        assert delivery.status == 'committed'
        assert delivery.successor_response_id == 'resp_b'
        assert [body['type'] for _, body in peer.requests] == ['response.create', 'response.steer']
        assert len(peer.connections) == 1


def tool_frames(response_id: str, name: str, arguments: str = '{}') -> list[dict[str, Any]]:
    frames = text_frames(response_id)
    call = {
        'type': 'function_call',
        'name': name,
        'arguments': arguments,
        'call_id': f'call_{response_id}',
        'id': f'fc_{response_id}',
    }
    frames[-1]['response']['output'] = [call]
    return [
        frames[0],
        {'type': 'response.output_item.added', 'sequence_number': 1, 'output_index': 0, 'item': call},
        {'type': 'response.output_item.done', 'sequence_number': 2, 'output_index': 0, 'item': call},
        frames[-1],
    ]


def acceptance() -> dict[str, Any]:
    return {
        'type': 'response.steer.accepted',
        'sequence_number': 2,
        'steer': {'id': 'steer_1', 'previous_response_id': 'resp_a'},
    }


@pytest.mark.parametrize('output_tool', [False, True])
@pytest.mark.parametrize('stream', [False, True])
async def test_native_tool_continuation_does_not_resend_input(peer: Peer, output_tool: bool, stream: bool):
    calls: list[int] = []

    def finish(value: int) -> int:
        calls.append(value)
        return value

    agent = Agent(
        peer.model(),
        deps_type=type(None),
        output_type=ToolOutput(finish, name='finish') if output_tool else str,
        model_settings=OpenAIResponsesModelSettings(openai_steering=True),
    )
    if not output_tool:
        agent.tool_plain(finish)
    initial = tool_frames('resp_a', 'finish', '{"value":1}')
    peer.scripts = [
        initial[:2],
        [
            acceptance(),
            *initial[2:],
            {
                'type': 'response.steer.pending',
                'sequence_number': 4,
                'steer': {'id': 'steer_1', 'previous_response_id': 'resp_a'},
                'reason': 'waiting_for_required_input',
                'required_input': [{'type': 'function_call_output', 'call_id': 'call_resp_a'}],
            },
        ],
        tool_frames('resp_b', 'finish', '{"value":2}') if output_tool else text_frames('resp_b', 'Updated'),
    ]
    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await session.steer('New requirement')

        with anyio.fail_after(10):
            if stream:
                async with session.run_stream('Start', event_stream_handler=handle) as result:
                    assert await result.get_output() == (2 if output_tool else 'Updated')
            else:
                assert (await session.run('Start', event_stream_handler=handle)).output == (
                    2 if output_tool else 'Updated'
                )
        assert calls == ([1, 2] if output_tool else [1])
        assert [body['type'] for _, body in peer.requests] == ['response.create', 'response.steer', 'response.create']
        body = peer.requests[-1][1]
        assert body['previous_response_id'] == 'resp_a'
        assert len(body['input']) == 1 and body['input'][0]['type'] == 'function_call_output'
        state = session.state
        assert state.steering[0].status == 'committed'
        part = state.conversation.messages[2].parts[0]
        assert isinstance(part, UserPromptPart) and part.content == ['New requirement']
        assert len(peer.connections) == 1


async def test_native_successor_keeps_advertised_tools(peer: Peer):
    preparations: list[int] = []
    executions: list[int] = []
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))

    async def prepare(ctx: RunContext[None], definition: ToolDefinition) -> ToolDefinition | None:
        preparations.append(ctx.run_step)
        return definition if ctx.run_step == 1 else None

    @agent.tool(prepare=prepare)
    async def lookup(ctx: RunContext[None]) -> str:
        executions.append(ctx.run_step)
        return 'found'

    initial = text_frames('resp_a', 'Original')
    peer.scripts = [initial[:2], [acceptance(), initial[-1], *tool_frames('resp_b', 'lookup')], text_frames('resp_c')]
    submitted = False

    async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        nonlocal submitted
        async for event in events:
            if isinstance(event, PartStartEvent) and not submitted:
                submitted = True
                await ctx.steer('Look up the answer')

    with anyio.fail_after(10):
        result = await agent.run('Start', event_stream_handler=handle)
    assert result.output == 'Hello'
    assert executions == [2]
    assert preparations == [1, 3]
    assert result.usage.requests == 3


@pytest.mark.parametrize('accepted', [False, True])
async def test_native_disconnect_preserves_uncommitted_input(peer: Peer, accepted: bool):
    initial = text_frames('resp_a', 'Original')
    peer.scripts = [initial[:2], [*([acceptance()] if accepted else []), initial[-1], None]]
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await ctx.steer('Keep this input')

        with anyio.fail_after(10), pytest.raises(ModelAPIError):
            await session.run('Start', event_stream_handler=handle)
        checkpoint = session.state
        delivery = checkpoint.steering[0]
        assert delivery.status == 'uncertain'
        part = delivery.messages[0].parts[0]
        assert isinstance(part, UserPromptPart) and part.content == ['Keep this input']
        assert len(checkpoint.conversation.messages) == 2
        with pytest.raises(UserError, match='unresolved'):
            await session.run('Do not silently lose the input')
        recovered = checkpoint.recover(steering={delivery.delivery_id: 'replay'})
        assert recovered.pending[0].enqueue_id == delivery.delivery_id
        assert len(peer.requests) == 2


@pytest.mark.parametrize('retry_response', ['resp_a', 'resp_b', None])
async def test_native_successor_cannot_be_replaced_by_request_middleware(peer: Peer, retry_response: str | None):
    wrapped: list[int] = []
    observed: list[str | None] = []

    class Middleware(AbstractCapability[None]):
        async def wrap_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            wrapped.append(ctx.run_step)
            assert len(wrapped) == 1, 'A successor must not be short-circuited by a cache'
            return await handler(request_context)

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            observed.append(response.provider_response_id)
            if response.provider_response_id == retry_response:
                raise ModelRetry('Try again')
            return response

    initial = text_frames('resp_a', 'Original')
    peer.scripts = [initial[:2], [acceptance(), initial[-1], *text_frames('resp_b', 'Updated')]]
    agent = Agent(
        peer.model(),
        deps_type=type(None),
        capabilities=[Middleware()],
        model_settings=OpenAIResponsesModelSettings(openai_steering=True),
    )
    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await ctx.steer('Change direction')

        with anyio.fail_after(10):
            if retry_response is None:
                assert (await session.run('Start', event_stream_handler=handle)).output == 'Updated'
            else:
                with pytest.raises(UserError, match=r'native|steering'):
                    await session.run('Start', event_stream_handler=handle)
        assert wrapped == [1]
        assert observed == (['resp_a'] if retry_response == 'resp_a' else ['resp_a', 'resp_b'])
        assert len(peer.requests) == 2
        assert session.state.steering[0].status == ('uncertain' if retry_response == 'resp_a' else 'committed')


@pytest.mark.parametrize('preflight', [False, True])
async def test_native_admission_precedes_wire_send(peer: Peer, preflight: bool):
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    attempted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal attempted
            async for event in events:
                if isinstance(event, PartStartEvent) and not attempted:
                    attempted = True
                    assert ctx.usage_limits is not None
                    if preflight:
                        # Initial request has already passed preflight; test steering's own guard.
                        ctx.usage_limits.count_tokens_before_request = True
                    with pytest.raises(UserError if preflight else UsageLimitExceeded):
                        await ctx.steer('Must not be sent')

        result = await session.run('Start', usage_limits=UsageLimits(request_limit=1), event_stream_handler=handle)
        assert result.output == 'Hello'
        assert attempted
        assert session.state.steering == []
        assert [body['type'] for _, body in peer.requests] == ['response.create']


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('mutation', ['replace', 'in_place', 'wrap', 'error'])
async def test_native_parent_hooks_preserve_recoverable_history(peer: Peer, stream: bool, mutation: str):
    class Rewrite(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if mutation == 'error':
                raise ValueError('hook failed')
            if mutation == 'in_place':
                response.parts = [TextPart('redacted')]
                return response
            return response if mutation == 'wrap' else ModelResponse(parts=[TextPart('redacted')])

        async def wrap_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            response = await handler(request_context)
            return ModelResponse(parts=[TextPart('redacted')]) if mutation == 'wrap' else response

    initial = text_frames('resp_a', 'Original')
    peer.scripts = [initial[:2], [acceptance(), initial[-1], *text_frames('resp_b', 'Updated')]]
    agent = Agent(
        peer.model(),
        deps_type=type(None),
        capabilities=[Rewrite()],
        model_settings=OpenAIResponsesModelSettings(openai_steering=True),
    )
    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await ctx.steer('Keep this input')

        with anyio.fail_after(10), pytest.raises(ValueError if mutation == 'error' else UserError):
            if stream:
                async with session.run_stream('Start', event_stream_handler=handle):
                    assert False, 'invalid native continuation must not yield a result'
            else:
                await session.run('Start', event_stream_handler=handle)
        state = session.state
        assert state.steering[0].status == 'uncertain'
        assert len(state.conversation.messages) == 2
        response = state.conversation.messages[-1]
        assert isinstance(response, ModelResponse)
        assert response.provider_response_id == 'resp_a'
        assert response.parts == [TextPart('Original', id='msg_resp_a', provider_name='openai')]
        assert state.conversation.usage.requests == 1
        assert state.recover(steering={state.steering[0].delivery_id: 'replay'}).pending


@pytest.mark.parametrize('stream', [False, True])
async def test_native_tool_continuation_rejects_model_switch_before_request(peer: Peer, stream: bool):
    replacement = TestModel(custom_output_text='wrong model')

    class Switch(AbstractCapability[None]):
        async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext):
            if ctx.run_step == 2:
                request_context.model = replacement
            return request_context

    initial = tool_frames('resp_a', 'lookup')
    peer.scripts = [initial[:2], [acceptance(), *initial[2:]]]
    agent = Agent(
        peer.model(),
        deps_type=type(None),
        capabilities=[Switch()],
        model_settings=OpenAIResponsesModelSettings(openai_steering=True, timeout=0.2),
    )

    @agent.tool_plain
    def lookup() -> str:
        return 'found'

    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await ctx.steer('Keep this input')

        with anyio.fail_after(10), pytest.raises(UserError, match=r'native|steering'):
            if stream:
                async with session.run_stream('Start', event_stream_handler=handle):
                    assert False, 'invalid native continuation must not yield a result'
            else:
                await session.run('Start', event_stream_handler=handle)
        assert replacement.last_model_request_parameters is None
        assert len(peer.requests) == 2
        assert session.state.steering[0].status == 'uncertain'


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('policy', ['cache', 'skip', 'wrapper', 'history'])
async def test_native_tool_continuation_middleware(peer: Peer, stream: bool, policy: str):
    class Middleware(AbstractCapability[None]):
        async def wrap_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            if ctx.run_step == 2 and policy == 'cache':
                return ModelResponse(parts=[TextPart('cached')])
            return await handler(request_context)

        async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext):
            if ctx.run_step == 2:
                if policy == 'skip':
                    raise SkipModelRequest(ModelResponse(parts=[TextPart('cached')]))
                if policy == 'history':
                    request_context.messages = [request_context.messages[-1]]
                else:
                    assert policy == 'wrapper'
                    request_context.model = WrapperModel(request_context.model)
            return request_context

    initial = tool_frames('resp_a', 'lookup')
    peer.scripts = [initial[:2], [acceptance(), *initial[2:]], text_frames('resp_b', 'Updated')]
    agent = Agent(
        peer.model(),
        deps_type=type(None),
        capabilities=[Middleware()],
        model_settings=OpenAIResponsesModelSettings(openai_steering=True),
    )

    @agent.tool_plain
    def lookup() -> str:
        return 'found'

    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await ctx.steer('New requirement')

        async def run() -> str:
            if stream:
                async with session.run_stream('Start', event_stream_handler=handle) as result:
                    return await result.get_output()
            return (await session.run('Start', event_stream_handler=handle)).output

        with anyio.fail_after(10):
            if policy == 'wrapper':
                assert await run() == 'Updated'
            else:
                with pytest.raises(UserError, match='native'):
                    await run()
        assert len(peer.requests) == (3 if policy == 'wrapper' else 2)
        assert session.state.steering[0].status == ('committed' if policy == 'wrapper' else 'uncertain')


async def test_native_successor_in_place_hook_cannot_change_history(peer: Peer):
    class Rewrite(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if response.provider_response_id == 'resp_b':
                response.parts = [TextPart('fabricated')]
            return response

    initial = text_frames('resp_a', 'Original')
    peer.scripts = [initial[:2], [acceptance(), initial[-1], *text_frames('resp_b', 'Updated')]]
    agent = Agent(
        peer.model(),
        deps_type=type(None),
        capabilities=[Rewrite()],
        model_settings=OpenAIResponsesModelSettings(openai_steering=True),
    )
    submitted = False
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal submitted
            async for event in events:
                if isinstance(event, PartStartEvent) and not submitted:
                    submitted = True
                    await ctx.steer('New requirement')

        with pytest.raises(UserError, match='native steering'):
            await session.run('Start', event_stream_handler=handle)
        state = session.state
        assert state.steering[0].status == 'committed'
        assert state.conversation.messages[-1].parts == [TextPart('Updated', id='msg_resp_b', provider_name='openai')]
        assert state.conversation.usage.requests == 2


@pytest.mark.parametrize('failure', ['rejected', 'cancelled'])
async def test_native_failed_exchange_retains_input(peer: Peer, failure: str):
    initial = text_frames('resp_a', 'Original')
    peer.scripts = [
        initial[:2],
        [
            {
                'type': 'response.steer.failed',
                'sequence_number': 2,
                'steer': {'previous_response_id': 'resp_a', 'input': 'Keep this'},
                'error': {
                    'code': 'steering_not_supported',
                    'message': 'Not supported',
                    'type': 'invalid_request_error',
                },
            },
        ],
    ]
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            assert isinstance(await anext(aiter(events)), PartStartEvent)
            await ctx.steer('Keep this')
            if failure == 'cancelled':
                raise ValueError('Consumer stopped')

        with pytest.raises(ValueError if failure == 'cancelled' else ModelAPIError):
            await session.run('Start', event_stream_handler=handle)
        state = session.state
        assert state.steering[0].status == ('uncertain' if failure == 'cancelled' else 'failed')
        assert len(state.conversation.messages) == 2
        assert state.recover(steering={state.steering[0].delivery_id: 'replay'}).pending
        assert len(peer.requests) == 2


async def test_native_multiple_successors_and_binary_input(peer: Peer):
    first = text_frames('resp_a', 'Original')
    second = text_frames('resp_b', 'Updated')
    peer.scripts = [
        first[:2],
        [acceptance(), first[-1], *second[:2]],
        [
            {
                'type': 'response.steer.accepted',
                'sequence_number': 2,
                'steer': {'id': 'steer_2', 'previous_response_id': 'resp_b'},
            },
            second[-1],
            *text_frames('resp_c', 'Final'),
        ],
    ]
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    submitted: list[str] = []
    image = BinaryImage(b'\xff\x00', media_type='image/png')
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                if isinstance(event, PartStartEvent) and len(submitted) < 2:
                    submitted.append(await ctx.steer('Use this image', image))
                    with pytest.raises(UserError, match='current steering submission'):
                        await ctx.steer('Not yet')

        result = await session.run('Start', event_stream_handler=handle)
        assert result.output == 'Final'
        assert result.usage.requests == 3
        state = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(session.state))
        assert [delivery.successor_response_id for delivery in state.steering] == ['resp_b', 'resp_c']
        assert len(state.conversation.messages) == 6
        for index in (2, 4):
            part = state.conversation.messages[index].parts[0]
            assert isinstance(part, UserPromptPart) and part.content == ['Use this image', image]
        assert [request['type'] for _, request in peer.requests] == [
            'response.create',
            'response.steer',
            'response.steer',
        ]
        assert peer.requests[1][1]['input'][0]['content'][1]['image_url'] == 'data:image/png;base64,/wA='


async def test_native_invalid_content_does_not_create_delivery(peer: Peer):
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    async with agent.session() as session:

        async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                if isinstance(event, PartStartEvent):
                    with pytest.raises(NotImplementedError, match='audio is not supported'):
                        await ctx.steer(BinaryContent(b'audio', media_type='audio/wav'))

        assert (await session.run('Start', event_stream_handler=handle)).output == 'Hello'
        assert session.state.steering == []
        assert len(peer.requests) == 1


@pytest.mark.parametrize('failure', [None, 'consumer', 'disconnect'])
async def test_native_successor_can_be_driven_by_graph_iteration(peer: Peer, failure: str | None):
    """After streaming the parent, callers can drive the successor with ordinary graph steps."""
    initial = text_frames('resp_a', 'Original')
    successor = text_frames('resp_b', 'Updated')
    peer.scripts = [
        initial[:2],
        [acceptance(), initial[-1], *successor[:2], None]
        if failure == 'disconnect'
        else [acceptance(), initial[-1], *successor],
    ]
    agent = Agent(peer.model(), model_settings=OpenAIResponsesModelSettings(openai_steering=True))
    async with agent.session() as session:
        async with session.iter('Start') as run:
            first_node = run.next_node
            assert Agent.is_user_prompt_node(first_node)
            node = await run.next(first_node)
            assert Agent.is_model_request_node(node)
            async with node.stream(run.ctx) as events:
                async for event in events:
                    if isinstance(event, PartStartEvent):
                        await session.steer('New requirement')
            node = await run.next(node)
            assert Agent.is_call_tools_node(node)
            node = await run.next(node)
            assert Agent.is_model_request_node(node)
            if failure == 'consumer':
                with pytest.raises(ValueError, match='consumer stopped'):
                    async with node.stream(run.ctx) as events:
                        assert isinstance(await anext(aiter(events)), PartStartEvent)
                        raise ValueError('consumer stopped')
            elif failure == 'disconnect':
                with pytest.raises(ModelAPIError):
                    await run.next(node)
            else:
                node = await run.next(node)
                assert Agent.is_call_tools_node(node)
                await run.next(node)
                assert run.result is not None
                assert run.result.output == 'Updated'
        state = session.state
        assert state.steering[0].status == 'committed'
        assert state.conversation.usage.requests == 2
        assert state.conversation.messages[-1].parts == [TextPart('Updated', id='msg_resp_b', provider_name='openai')]
        assert [body['type'] for _, body in peer.requests] == ['response.create', 'response.steer']


@pytest.mark.parametrize('failure', ['unsolicited', 'wrong_parent', 'rejected_with_id'])
async def test_native_protocol_errors_do_not_lose_delivery_state(peer: Peer, failure: str):
    event = acceptance()
    if failure == 'wrong_parent':
        event['steer']['previous_response_id'] = 'unrelated'
    elif failure == 'rejected_with_id':
        event['type'] = 'response.steer.failed'
        event['error'] = {'code': 'invalid_input', 'message': 'Rejected input', 'type': 'invalid_request_error'}
    initial = text_frames('resp_a', 'Original')
    peer.scripts = [[event]] if failure == 'unsolicited' else [initial[:2], [event]]
    agent = Agent(peer.model(), deps_type=type(None), model_settings=OpenAIResponsesModelSettings(openai_steering=True))

    async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            assert isinstance(event, PartStartEvent)
            await ctx.steer('Keep this input')

    async with agent.session() as session:
        with pytest.raises(ModelAPIError if failure == 'rejected_with_id' else UnexpectedModelBehavior):
            await session.run('Start', event_stream_handler=handle)
        if failure == 'unsolicited':
            assert session.state.steering == []
        else:
            (delivery,) = session.state.steering
            assert delivery.status == ('failed' if failure == 'rejected_with_id' else 'uncertain')
            assert delivery.provider_id == ('steer_1' if failure == 'rejected_with_id' else None)
            assert session.state.recover(steering={delivery.delivery_id: 'replay'}).pending
