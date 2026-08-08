"""Tests for deferred-capability loading, `load_capability`, and tool reveal / availability deltas.

Split out of `test_capabilities.py`, which had grown past the repository's file-size limit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    Capability,
    ProcessHistory,
    ToolSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import (
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolAvailabilityDeltaEvent,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    ModelRequestContext,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    AbstractNativeTool,
)
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._deferred_capability_loader import (
    LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE,
    LOAD_CAPABILITY_TOOL_NAME,
)
from pydantic_ai.usage import RequestUsage, RunUsage

from ._inline_snapshot import snapshot
from .capability_models import (
    make_text_response,
)
from .conftest import IsDatetime, IsStr, iter_message_parts

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


def _build_run_context(deps: Any = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), run_step=0)

async def test_deferred_capability_loads_instructions_and_tools_e2e() -> None:
    """A deferred capability starts as a catalog entry and becomes usable after `load_capability`."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    def add_account_context(ctx: RunContext) -> str:
        return f'Load-time account context for run step {ctx.run_step}.'

    def empty_instruction(ctx: RunContext) -> None:
        return None

    always_on = Capability[object](
        id='always-on',
        description='Visible billing guidance.',
        instructions='Visible billing instructions.',
    )
    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        instructions=[
            'Use the refund policy before answering refund questions.',
            add_account_context,
            empty_instruction,
        ],
        toolsets=[toolset],
        defer_loading=True,
    )

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ]
            )

        if not any(part.tool_name == 'lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )

        refund_result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund_policy')
        return make_text_response(f'final: {refund_result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[always_on, refunds])

    result = await agent.run('Can I get a refund?')

    assert result.output == snapshot('final: order-123: refund allowed for 30 days')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Can I get a refund?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    LoadCapabilityCallPart(
                        tool_name='load_capability',
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ],
                usage=RequestUsage(input_tokens=55, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    LoadCapabilityReturnPart(
                        content={
                            'instructions': """\
Use the refund policy before answering refund questions.

Load-time account context for run step 1.\
"""
                        },
                        tool_call_id='load-refunds',
                        timestamp=IsDatetime(),
                    ),
                    ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='load-refunds'),
                ],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy', args={'order_id': 'order-123'}, tool_call_id='lookup-refund'
                    )
                ],
                usage=RequestUsage(input_tokens=80, output_tokens=10),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='lookup_refund_policy',
                        content='order-123: refund allowed for 30 days',
                        tool_call_id='lookup-refund',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final: order-123: refund allowed for 30 days')],
                usage=RequestUsage(input_tokens=86, output_tokens=17),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_tool_return_reveals_deferred_tool_without_capability() -> None:
    """A user tool can reveal a deferred tool and records the delta beside its return."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for part in iter_message_parts(messages, ModelRequest, ToolReturnPart)
            if part.tool_name in {'reveal_weather', 'get_weather'}
        ]
        if not returns:
            assert info.model_request_parameters.revealed_tool_names == set()
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_weather', args={}, tool_call_id='reveal')])
        if len(returns) == 1:
            assert info.model_request_parameters.revealed_tool_names == {'get_weather'}
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='weather')]
            )
        return make_text_response(str(returns[-1].content))

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def reveal_weather() -> ToolReturn[str]:
        return ToolReturn(return_value='Weather tools are ready.', tools=['get_weather'])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    result = await agent.run('What is the weather?')

    assert result.output == 'Sunny in Paris'
    reveal_request = next(
        message
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_call_id == 'reveal' for part in message.parts)
    )
    assert reveal_request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='reveal_weather',
                content='Weather tools are ready.',
                tool_call_id='reveal',
                timestamp=IsDatetime(),
            ),
            ToolAvailabilityDeltaPart(tools_added=['get_weather'], tool_call_id='reveal'),
        ]
    )


async def test_processed_history_determines_request_reveal_state() -> None:
    """Removing a reveal from outgoing history also removes it from request parameters."""
    seen: list[set[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.revealed_tool_names)
        assert 'hidden_tool' not in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[TextPart(content='done')])

    def strip_deltas(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(message, parts=[part for part in message.parts if not isinstance(part, ToolAvailabilityDeltaPart)])
            if isinstance(message, ModelRequest)
            else message
            for message in messages
        ]

    agent = Agent(FunctionModel(model_fn), capabilities=[ProcessHistory(strip_deltas)])

    @agent.tool_plain(defer_loading=True)
    def hidden_tool() -> str:  # pragma: no cover
        return 'hidden'

    await agent.run(
        'continue',
        message_history=[ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden_tool'])])],
    )

    assert seen == [set()]


async def test_orphaned_reveal_evidence_stripped_by_cleanup_does_not_count_as_revealed() -> None:
    """Evidence orphaned by a history processor is stripped before reveal derivation.

    A processor that drops the response carrying a `ToolSearchCallPart` leaves an orphaned
    `ToolSearchReturnPart`; history cleanup removes the orphan before the request ships, so the
    derived reveal state must not count it — otherwise the request would declare a revealed tool
    with zero reveal evidence on the outgoing wire.
    """
    seen: list[set[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.revealed_tool_names)
        assert 'hidden_tool' not in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[TextPart(content='done')])

    def drop_search_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            message
            for message in messages
            if not (
                isinstance(message, ModelResponse)
                and any(isinstance(part, ToolSearchCallPart) for part in message.parts)
            )
        ]

    agent = Agent(FunctionModel(model_fn), capabilities=[ProcessHistory(drop_search_calls)])

    @agent.tool_plain(defer_loading=True)
    def hidden_tool() -> str:  # pragma: no cover
        return 'hidden'

    await agent.run(
        'continue',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='find tools')]),
            ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['hidden']}, tool_call_id='search-1')]),
            ModelRequest(
                parts=[
                    ToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'hidden_tool'}]},
                        tool_call_id='search-1',
                    )
                ]
            ),
        ],
    )

    assert seen == [set()]


async def test_model_calling_a_withheld_tool_executes_without_revealing_it() -> None:
    """Calling a hidden tool by (guessed) name executes it and authors no reveal.

    Pins the documented no-trust-boundary stance: hiding is prompt engineering, not access
    control, so execution is accepted — but execution is not discovery, and the tool stays off
    the wire afterwards.
    """
    wire_tools: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        wire_tools.append(sorted(tool.name for tool in info.function_tools))
        if not list(iter_message_parts(messages, ModelRequest, ToolReturnPart)):
            return ModelResponse(parts=[ToolCallPart(tool_name='hidden_tool', args={}, tool_call_id='guess')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain(defer_loading=True)
    def hidden_tool() -> str:
        return 'secret'

    result = await agent.run('guess the hidden tool')

    assert result.output == 'done'
    returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert [(part.tool_name, part.content) for part in returns] == [('hidden_tool', 'secret')]
    deltas = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert deltas == []
    assert all('hidden_tool' not in tools for tools in wire_tools)


async def test_tool_return_deduplicates_new_reveals() -> None:
    """Duplicate names and repeated reveals author one ordered availability delta.

    A fully repeated reveal drops out entirely; a partial overlap keeps only the genuinely new
    names, in order.
    """

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(parts=[ToolCallPart('revealer', {}, tool_call_id='first')])
        if len(returns) == 1:
            return ModelResponse(parts=[ToolCallPart('revealer', {}, tool_call_id='second')])
        if len(returns) == 2:
            return ModelResponse(parts=[ToolCallPart('partial_revealer', {}, tool_call_id='third')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def revealer() -> ToolReturn[str]:
        return ToolReturn(return_value='ready', tools=['tool_b', 'tool_a', 'tool_b'])

    @agent.tool_plain
    def partial_revealer() -> ToolReturn[str]:
        return ToolReturn(return_value='partially new', tools=['tool_a', 'tool_c'])

    result = await agent.run('reveal')
    deltas = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert deltas == [
        ToolAvailabilityDeltaPart(tools_added=['tool_b', 'tool_a'], tool_call_id='first'),
        ToolAvailabilityDeltaPart(tools_added=['tool_c'], tool_call_id='third'),
    ]


@pytest.mark.parametrize(
    'tools',
    ['get_weather', 1, [1], [[]]],
    ids=['bare-string', 'non-sequence', 'non-string-element', 'unhashable-element'],
)
async def test_tool_return_rejects_invalid_tools(tools: object) -> None:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if list(iter_message_parts(messages, ModelRequest, ToolReturnPart)):  # pragma: no cover
            return make_text_response('done')
        return ModelResponse(parts=[ToolCallPart(tool_name='reveal_weather', args={}, tool_call_id='reveal')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def reveal_weather() -> ToolReturn[str]:
        return ToolReturn(return_value='Weather tools are ready.', tools=cast(Any, tools))

    with pytest.raises(UserError, match=r'`ToolReturn\.tools` must be a list of tool names'):
        await agent.run('Reveal the weather tool.')


async def test_parallel_tool_returns_keep_each_availability_delta_adjacent() -> None:
    """Parallel execution reorders each return together with its own sibling delta."""

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='reveal_b', args={}, tool_call_id='b'),
                    ToolCallPart(tool_name='reveal_a', args={}, tool_call_id='a'),
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    async def reveal_a() -> ToolReturn[str]:
        await asyncio.sleep(0)
        return ToolReturn(return_value='a', tools=['tool_a'])

    @agent.tool_plain
    async def reveal_b() -> ToolReturn[str]:
        await asyncio.sleep(0.01)
        return ToolReturn(return_value='b', tools=['tool_b'])

    result = await agent.run('reveal both')
    request = next(
        message
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_call_id == 'b' for part in message.parts)
    )
    assert [(type(part).__name__, getattr(part, 'tool_call_id', None)) for part in request.parts] == snapshot(
        [
            ('ToolReturnPart', 'b'),
            ('ToolAvailabilityDeltaPart', 'b'),
            ('ToolReturnPart', 'a'),
            ('ToolAvailabilityDeltaPart', 'a'),
        ]
    )


async def test_parallel_tool_returns_dedupe_same_reveal_in_history_order() -> None:
    """When parallel calls reveal the same tool, the first call in emitted history owns the delta.

    Deduplication must not depend on task completion order: the first-emitted call finishes
    LAST here, and must still be the one that carries the availability delta — otherwise the
    durable history (and the reveal's wire anchor) would vary with scheduling.
    """

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='slow_first', args={}, tool_call_id='first'),
                    ToolCallPart(tool_name='fast_second', args={}, tool_call_id='second'),
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    async def slow_first() -> ToolReturn[str]:
        await asyncio.sleep(0.01)
        return ToolReturn(return_value='slow', tools=['revealed'])

    @agent.tool_plain
    async def fast_second() -> ToolReturn[str]:
        return ToolReturn(return_value='fast', tools=['revealed'])

    events: list[AgentStreamEvent] = []
    async with agent.iter('reveal in parallel') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    events.extend([event async for event in stream])

    assert agent_run.result is not None
    result = agent_run.result
    deltas = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert deltas == [ToolAvailabilityDeltaPart(tools_added=['revealed'], tool_call_id='first')]
    assert [event for event in events if isinstance(event, ToolAvailabilityDeltaEvent)] == [
        ToolAvailabilityDeltaEvent(part=deltas[0])
    ]


async def test_deferred_capability_tool_registered_after_construction_defers_until_load() -> None:
    """A tool registered via `@cap.tool` *after* construction defers like a constructor tool: hidden until load.

    Deferred tools stay tagged `defer_loading=True`; current visibility is tracked separately.
    """
    refunds = Capability[object](id='refunds', description='Refund policy tools.', defer_loading=True)

    # Register on the deferred capability *after* construction (decorator path, not the `tools=` arg).
    @refunds.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    defer_flag_by_phase: dict[str, bool | None] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        loaded = any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns)
        refund_def = next((tool for tool in info.function_tools if tool.name == 'lookup_refund_policy'), None)
        defer_flag_by_phase['after_load' if loaded else 'before_load'] = (
            refund_def.defer_loading if refund_def else None
        )

        if not loaded:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        if not any(part.tool_name == 'lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='lookup_refund_policy', args={'order_id': 'order-1'}, tool_call_id='look')
                ]
            )
        result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund_policy')
        return make_text_response(f'final: {result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('Can I get a refund?')

    assert result.output == snapshot('final: order-1: refund allowed for 30 days')
    assert defer_flag_by_phase == snapshot({'before_load': None, 'after_load': True})


async def test_deferred_capability_tool_stays_available_across_turns() -> None:
    """A capability-owned tool stays callable across every turn after `load_capability`.

    Regression guard: the `available_tool_names`/`discovered_tool_names` split must keep a
    loaded deferred tool non-deferred on the second (and later) post-load model request,
    not just on the turn immediately following the load.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        toolsets=[toolset],
        defer_loading=True,
    )
    hooks = Hooks()
    available_per_turn: list[set[str]] = []

    @hooks.on.before_model_request
    async def record_available_tools(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext:
        available_per_turn.append(ctx.available_tool_names)
        return request_context

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        # Turn 1: load the capability.
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )

        lookup_calls = [part for part in tool_returns if part.tool_name == 'lookup_refund_policy']

        # Turns 2 and 3: call the loaded tool twice, so we exercise two post-load turns.
        if len(lookup_calls) < 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': f'order-{len(lookup_calls)}'},
                        tool_call_id=f'lookup-{len(lookup_calls)}',
                    )
                ]
            )

        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds, hooks])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done'
    assert 'lookup_refund_policy' not in available_per_turn[0]
    assert len(available_per_turn[1:]) >= 2
    assert all('lookup_refund_policy' in names for names in available_per_turn[1:])


async def test_run_context_tools_exposes_deferred_definitions_as_name_keyed_dict() -> None:
    """`ctx.tools` is the full name-keyed dict of `ToolDefinition`s, including entries
    that are still deferred (and therefore absent from `ctx.available_tool_names`)."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        return f'{order_id}: refund allowed'

    refunds = Capability[object](id='refunds', toolsets=[toolset], defer_loading=True)

    seen_tools: list[dict[str, ToolDefinition]] = []

    @dataclass
    class CaptureCtxToolsCap(AbstractCapability):
        async def before_model_request(
            self, ctx: RunContext, request_context: ModelRequestContext
        ) -> ModelRequestContext:
            seen_tools.append(ctx.tools)
            return request_context

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds, CaptureCtxToolsCap()])
    await agent.run('hi')

    [tools] = seen_tools
    # The deferred tool is keyed by its own name and carries `defer_loading=True`,
    # even though it's absent from `available_tool_names` until the capability loads.
    assert tools['lookup_refund_policy'].name == 'lookup_refund_policy'
    assert tools['lookup_refund_policy'].defer_loading is True


async def test_deferred_capability_tool_delta_persists_in_history() -> None:
    """The tool delta after a capability load persists, without duplication on resume."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        toolsets=[toolset],
        defer_loading=True,
    )

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    events: list[AgentStreamEvent] = []
    async with agent.iter('Can I get a refund?') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    events.extend([event async for event in stream])

    assert agent_run.result is not None
    result = agent_run.result

    def availability_deltas(messages: list[ModelMessage]) -> list[ToolAvailabilityDeltaPart]:
        return [part for message in messages for part in message.parts if isinstance(part, ToolAvailabilityDeltaPart)]

    messages = result.all_messages()
    assert availability_deltas(messages) == [
        ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='load')
    ]
    assert [event for event in events if isinstance(event, ToolAvailabilityDeltaEvent)] == [
        ToolAvailabilityDeltaEvent(part=availability_deltas(messages)[0])
    ]

    # Idempotence: feeding the resulting history back in does not inject a duplicate pair
    # (the deterministic call_id means it's recognized as already discovered).
    result2 = await agent.run('And another refund?', message_history=messages)
    new_messages = result2.all_messages()[len(messages) :]
    assert availability_deltas(new_messages) == []


async def test_capability_load_history_without_delta_is_backfilled() -> None:
    """An ID-only capability load history gains one delta before the resumed model request."""
    refunds = Capability[object](id='refunds', defer_loading=True)
    visibility: list[tuple[bool, set[str]]] = []

    @refunds.tool_plain
    def lookup_refund_policy() -> str:  # pragma: no cover
        return 'refund allowed'

    @dataclass
    class CaptureVisibility(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            visibility.append((ctx.is_tool_available('lookup_refund_policy'), ctx.available_tool_names))
            return request_context

    history: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='old-load')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='old-load')]),
    ]

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.revealed_tool_names == {'lookup_refund_policy'}
        return make_text_response('done')

    result = await Agent(FunctionModel(model_fn), capabilities=[refunds, CaptureVisibility()]).run(
        'Continue.', message_history=history
    )

    assert visibility == [(True, {'load_capability', 'lookup_refund_policy'})]
    new_deltas = [
        part
        for message in result.new_messages()
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert new_deltas == [ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'])]


class _NoNativeToolSearchModel(FunctionModel):
    """`FunctionModel` that forces the local `search_tools` function path.

    `FunctionModel` reports support for every native tool (including native tool search),
    which would route deferred standalone tools through the provider rather than the
    synthetic `search_tools` function. Dropping `ToolSearchTool` mirrors a model without
    native tool-search support, exercising the function-tool discovery path.
    """

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset(super().supported_native_tools()) - {ToolSearchTool}


async def test_two_deferred_capabilities_loaded_sequentially_both_stay_available() -> None:
    """Loading a second deferred capability does not drop the first one's tool.

    Trajectory: load A and call A's tool, then on a later turn load B and call B's tool,
    then one more turn. Both capabilities' tools must be non-deferred on every turn after
    their respective loads, proving loads are additive and sticky.
    """
    toolset_a = FunctionToolset()

    @toolset_a.tool_plain
    def alpha_tool() -> str:
        """Capability A's tool."""
        return 'alpha-result'

    toolset_b = FunctionToolset()

    @toolset_b.tool_plain
    def beta_tool() -> str:
        """Capability B's tool."""
        return 'beta-result'

    cap_a = Capability[object](id='alpha', description='Alpha tools.', toolsets=[toolset_a], defer_loading=True)
    cap_b = Capability[object](id='beta', description='Beta tools.', toolsets=[toolset_b], defer_loading=True)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        names = {part.tool_name for part in tool_returns}

        # Turn 1: load A.
        if 'alpha' not in {part.capability_id for part in _load_calls(messages)}:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'alpha'}, tool_call_id='load-a')]
            )
        # Turn 2: use A's tool.
        if 'alpha_tool' not in names:
            return ModelResponse(parts=[ToolCallPart(tool_name='alpha_tool', args={}, tool_call_id='call-a')])
        # Turn 3: load B.
        if 'beta' not in {part.capability_id for part in _load_calls(messages)}:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'beta'}, tool_call_id='load-b')]
            )
        # Turn 4: use B's tool.
        if 'beta_tool' not in names:
            return ModelResponse(parts=[ToolCallPart(tool_name='beta_tool', args={}, tool_call_id='call-b')])
        # Turn 5+: just respond.
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap_a, cap_b])
    result = await agent.run('Use both capabilities.')

    assert result.output == 'done'


async def test_tool_search_discovery_and_capability_load_coexist() -> None:
    """A tool-search-discovered standalone tool and a load_capability tool coexist and persist.

    Trajectory: discover a standalone deferred tool via `search_tools`, load a deferred
    capability via `load_capability`, then continue for extra turns. Both the searched tool
    and the capability's tool must be available together and stay available afterwards.
    """
    standalone = FunctionToolset()

    @standalone.tool_plain(defer_loading=True)
    def searchable_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'{city}: sunny'

    cap_toolset = FunctionToolset()

    @cap_toolset.tool_plain
    def lookup_refund(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refundable'

    refunds = Capability[object](id='refunds', description='Refund tools.', toolsets=[cap_toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        names = {part.tool_name for part in tool_returns}

        # Turn 1: search for the standalone deferred tool.
        if not any(part.tool_name == _SEARCH_TOOLS_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=_SEARCH_TOOLS_NAME, args={'queries': ['weather']}, tool_call_id='search')]
            )
        # Turn 2: load the deferred capability.
        if not _load_calls(messages):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        # Turn 3: use the discovered standalone tool.
        if 'searchable_weather' not in names:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='searchable_weather', args={'city': 'Paris'}, tool_call_id='call-w')]
            )
        # Turn 4: use the capability's tool.
        if 'lookup_refund' not in names:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='lookup_refund', args={'order_id': 'o1'}, tool_call_id='call-r')]
            )
        # Turn 5+: respond.
        return make_text_response('done')

    agent = Agent(_NoNativeToolSearchModel(model_fn), capabilities=[standalone_capability(standalone), refunds])
    result = await agent.run('Find weather and refund tools.')

    assert result.output == 'done'


async def test_deferred_capability_tool_delta_not_duplicated_over_long_trajectory() -> None:
    """The tool availability delta for a loaded capability appears exactly once.

    Extends the persistence test to >= 3 model-request turns after the load: the delta must
    remain singular across the whole trajectory, and the capability's tool stays available
    on every post-load turn.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds', description='Refund policy tools.', toolsets=[toolset], defer_loading=True
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )

        # Three post-load turns that each call the loaded tool, then respond.
        lookup_calls = [part for part in tool_returns if part.tool_name == 'lookup_refund_policy']
        if len(lookup_calls) < 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': f'order-{len(lookup_calls)}'},
                        tool_call_id=f'lookup-{len(lookup_calls)}',
                    )
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('Refund please.')

    assert result.output == 'done'

    messages = result.all_messages()
    tool_deltas = [
        part for message in messages for part in message.parts if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert tool_deltas == [ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='load')]


async def test_deferred_capability_tool_available_on_turn_that_does_not_call_it() -> None:
    """A loaded capability's tool stays available on a turn that does not call it.

    After loading, the model calls an unrelated visible tool (not the capability's tool) and
    then responds. The capability's tool must remain non-deferred on those turns — loading is
    sticky, not gated on per-turn usage.
    """
    visible_toolset = FunctionToolset()

    @visible_toolset.tool_plain
    def ping() -> str:
        """An always-visible tool unrelated to the capability."""
        return 'pong'

    cap_toolset = FunctionToolset()

    @cap_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds', description='Refund policy tools.', toolsets=[cap_toolset], defer_loading=True
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        names = {part.tool_name for part in tool_returns}

        # Turn 1: load the capability.
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        # Turn 2: call an UNRELATED tool, never the capability's tool.
        if 'ping' not in names:
            return ModelResponse(parts=[ToolCallPart(tool_name='ping', args={}, tool_call_id='call-ping')])
        # Turn 3: respond without ever calling the capability's tool.
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), tools=[ping], capabilities=[refunds])
    # `ping` is registered via a function tool on the agent; ensure both paths see it.
    result = await agent.run('Load refunds but use ping.')

    assert result.output == 'done'


def _load_calls(messages: list[ModelMessage]) -> list[LoadCapabilityCallPart]:
    """All `load_capability` call parts in the message history."""
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, LoadCapabilityCallPart)
    ]


def standalone_capability(toolset: FunctionToolset) -> Capability:
    """Wrap a toolset of standalone deferred tools in an eager capability (tools keep their own defer flag)."""
    return Capability[object](id='standalone', description='Standalone searchable tools.', toolsets=[toolset])


async def test_deferred_capability_load_includes_toolset_instructions() -> None:
    """Instructions declared on a deferred capability's toolset surface via the `load_capability` return.

    The wrapping `CapabilityOwnedToolset` silences `get_instructions` for deferred-loading
    capabilities (so toolset hints don't leak into the prompt), then re-emits them on load
    alongside the capability's own instructions.
    """
    toolset = FunctionToolset(instructions='Use the refund tool with the order id, not the customer id.')

    @toolset.tool_plain
    def lookup_refund(order_id: str) -> str:
        return f'{order_id}: ok'

    refunds = Capability[object](
        id='refunds',
        description='Refund tools.',
        instructions='Quote the refund policy verbatim.',
        toolsets=[toolset],
        defer_loading=True,
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        already_loaded = any(
            isinstance(part, LoadCapabilityReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if not already_loaded:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ]
            )
        if not any(part.tool_name == 'lookup_refund' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )
        refund_result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund')
        return make_text_response(str(refund_result))

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('hi')

    assert result.output == 'order-123: ok'
    [load_return] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert load_return.instructions == snapshot("""\
Quote the refund policy verbatim.

Use the refund tool with the order id, not the customer id.\
""")
    first_request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))
    assert first_request.instructions == snapshot(
        """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
"""
    )
    assert first_request.instructions is not None
    assert 'Use the refund tool' not in first_request.instructions


async def test_deferred_capability_load_drops_empty_toolset_instructions() -> None:
    """Empty toolset instructions are filtered from load returns."""
    from dataclasses import dataclass

    from pydantic_ai.messages import InstructionPart
    from pydantic_ai.toolsets.wrapper import WrapperToolset

    @dataclass
    class _LiteralInstructionsToolset(WrapperToolset):
        raw: tuple[str | InstructionPart, ...] = ()

        async def get_instructions(self, ctx: RunContext) -> list[str | InstructionPart]:
            return list(self.raw)

    toolset = _LiteralInstructionsToolset(
        wrapped=FunctionToolset(),
        raw=(
            InstructionPart(content='   ', dynamic=False),
            InstructionPart(content='Real hint from toolset.', dynamic=False),
            '',
        ),
    )
    cap = Capability[object](id='cap', description='Custom-toolset cap.', toolsets=[toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        already_loaded = any(
            isinstance(part, LoadCapabilityReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if not already_loaded:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'cap'},
                        tool_call_id='load',
                    )
                ]
            )
        return make_text_response('ok')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])
    result = await agent.run('hi')

    [load_return] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert load_return.instructions == 'Real hint from toolset.'


async def test_unknown_deferred_capability_id_does_not_reveal_hidden_tools() -> None:
    toolset = FunctionToolset()

    @toolset.tool_plain
    def hidden_tool() -> str:
        return 'hidden'  # pragma: no cover

    hidden = Capability[object](
        id='hidden',
        description='Hidden tool access.',
        toolsets=[toolset],
        defer_loading=True,
    )
    seen_tool_state: list[list[tuple[str, bool]]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_state.append([(t.name, bool(t.defer_loading)) for t in info.function_tools])
        # Give up on the first signal of tool feedback — either a `ToolReturnPart`
        # (success, which can't happen here) or a `RetryPromptPart` (the framework
        # signaling the bad cap id). Without the retry branch, we'd loop past
        # `max_retries` and raise `UnexpectedModelBehavior` instead of giving up.
        if not any(
            isinstance(part, (ToolReturnPart, RetryPromptPart))
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'missing'},
                        tool_call_id='load-missing',
                    )
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[hidden])
    result = await agent.run('load missing')

    assert result.output == snapshot('done')
    assert seen_tool_state == snapshot(
        [
            [('load_capability', False)],
            [('load_capability', False)],
        ]
    )
    history_parts = [part for message in result.all_messages() for part in message.parts]
    assert not any(isinstance(part, LoadCapabilityReturnPart) for part in history_parts)
    [retry] = [part for part in history_parts if isinstance(part, RetryPromptPart)]
    assert retry.content == snapshot("No capability found with id 'missing'.")


async def test_load_capability_retries_for_already_available_capability() -> None:
    always_on = Capability[object](
        id='always-on',
        description='Already visible.',
        instructions='Already visible instructions.',
    )
    deferred = Capability[object](
        id='deferred',
        description='Deferred.',
        instructions='Deferred instructions.',
        defer_loading=True,
    )
    expected_retry = LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE.format(capability_id='always-on')
    retry_messages: list[str] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        retries = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart) and isinstance(part.content, str)
        ]
        if retries:
            retry_messages.extend(retries)
            return make_text_response('done')

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=LOAD_CAPABILITY_TOOL_NAME,
                    args={'id': 'always-on'},
                    tool_call_id='load-always-on',
                )
            ]
        )

    agent = Agent(FunctionModel(model_fn), capabilities=[always_on, deferred])
    result = await agent.run('load always-on')

    assert result.output == 'done'
    assert retry_messages == [expected_retry]
    assert not any(
        isinstance(part, LoadCapabilityReturnPart) for message in result.all_messages() for part in message.parts
    )


async def test_load_capability_retries_when_capability_is_already_loaded() -> None:
    deferred = Capability[object](
        id='deferred',
        description='Deferred.',
        instructions='Deferred instructions.',
        defer_loading=True,
    )
    expected_retry = LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE.format(capability_id='deferred')
    retry_messages: list[str] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        retries = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart) and isinstance(part.content, str)
        ]
        if retries:
            retry_messages.extend(retries)
            return make_text_response('done')

        load_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, LoadCapabilityReturnPart)
        ]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=LOAD_CAPABILITY_TOOL_NAME,
                    args={'id': 'deferred'},
                    tool_call_id=f'load-deferred-{len(load_returns)}',
                )
            ]
        )

    agent = Agent(FunctionModel(model_fn), capabilities=[deferred])
    result = await agent.run('load twice')

    assert result.output == 'done'
    assert retry_messages == [expected_retry]
    load_returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert len(load_returns) == 1
    assert load_returns[0].instructions == 'Deferred instructions.'


