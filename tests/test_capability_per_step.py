"""Tests for on-demand capabilities that a `DynamicCapability(per_run_step=True)` returns per run step."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from pydantic_ai import (
    Agent,
    AgentInstructionSource,
    CapabilityInstructionSource,
    InstructionBaselinePart,
    InstructionDigest,
    InstructionId,
    InstructionPart,
    RunContext,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CombinedCapability,
    DynamicCapability,
    NativeTool,
    durable_operation,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityLoadDeltaPart,
    FunctionToolResultEvent,
    InstructionDeltaPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryInfo,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, ApprovalRequiredToolset
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .capability_models import make_text_response
from .conftest import IsDatetime, IsStr


def refund_status(order_id: str) -> str:
    """Look up the refund status for an order."""
    return f'Order {order_id}: refund issued.'


def make_refunds() -> Capability[object]:
    return Capability[object](
        id='refunds',
        description='Refund tools.',
        instructions='Confirm the order ID first.',
        tools=[refund_status],
        defer_loading=True,
    )


def responses(messages: list[ModelMessage]) -> int:
    return sum(isinstance(message, ModelResponse) for message in messages)


@dataclass
class Seen:
    """What the model was offered on each request."""

    tools: list[list[str]] = field(default_factory=list[list[str]])
    instructions: list[str | None] = field(default_factory=list[str | None])

    def record(self, messages: list[ModelMessage], info: AgentInfo) -> None:
        self.tools.append(sorted(tool_def.name for tool_def in info.function_tools))
        self.instructions.append(info.instructions)


async def test_capability_appearing_mid_run_is_loaded_by_the_model() -> None:
    """A capability the factory first returns at a later step joins the catalog as an appended change."""
    seen = Seen()

    def capability_for_step(ctx: RunContext[object]) -> Capability[object] | None:
        # A fresh instance on every step, as a factory reading a database would build.
        return make_refunds() if ctx.run_step >= 2 else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.record(messages, info)
        step = responses(messages)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='ping')])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'}, tool_call_id='load')])
        if step == 2:
            return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'}, tool_call_id='status')])
        return make_text_response('done')

    agent = Agent(
        FunctionModel(model_fn),
        instructions='Help with orders.',
        capabilities=[DynamicCapability(capability_for_step, per_run_step=True)],
    )

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('Refund order A1')

    # `load_capability` is offered from the first request, before any on-demand capability exists, and the
    # instructions in the prefix never change: the catalog arrives as an appended change instead.
    assert seen.tools == snapshot(
        [
            ['load_capability', 'ping'],
            ['load_capability', 'ping'],
            ['load_capability', 'ping', 'refund_status'],
            ['load_capability', 'ping', 'refund_status'],
        ]
    )
    assert seen.instructions == snapshot(
        ['Help with orders.', 'Help with orders.', 'Help with orders.', 'Help with orders.']
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Refund order A1', timestamp=IsDatetime()),
                    InstructionBaselinePart(
                        instructions=[
                            InstructionDigest(
                                id=InstructionId(source=AgentInstructionSource()),
                                content_hash='aa34b7d719714d806f37d27cbd9a37cc65e8cc6d6f647acf899ec04b4b540d44',
                            ),
                            InstructionPart(
                                content='',
                                dynamic=True,
                                on_change='append',
                                id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                            ),
                        ]
                    ),
                ],
                timestamp=IsDatetime(),
                instructions='Help with orders.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ping', args={}, tool_call_id='ping')],
                usage=RequestUsage(input_tokens=53, output_tokens=2),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='ping', content='pong', tool_call_id='ping', timestamp=IsDatetime()),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                        content="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                    ),
                ],
                timestamp=IsDatetime(),
                instructions='Help with orders.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='load')],
                usage=RequestUsage(input_tokens=88, output_tokens=7),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    LoadCapabilityReturnPart(
                        content={'message': "Capability 'refunds' loaded. Its instructions, if any, follow."},
                        tool_call_id='load',
                        timestamp=IsDatetime(),
                    ),
                    ToolAvailabilityDeltaPart(tools_added=['refund_status'], tool_call_id='load'),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='refunds')),
                        content='Confirm the order ID first.',
                    ),
                ],
                timestamp=IsDatetime(),
                instructions='Help with orders.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='refund_status', args={'order_id': 'A1'}, tool_call_id='status')],
                usage=RequestUsage(input_tokens=119, output_tokens=12),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='refund_status',
                        content='Order A1: refund issued.',
                        tool_call_id='status',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Help with orders.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=124, output_tokens=13),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_capability_appearing_mid_run_is_loaded_from_code() -> None:
    """`ctx.load_capability()` loads a capability that appeared mid-run, leaving the records a load from code leaves."""
    refunds = make_refunds()

    def capability_for_step(ctx: RunContext[object]) -> Capability[object] | None:
        return refunds if ctx.run_step >= 2 else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = responses(messages)
        if step < 2:
            return ModelResponse(parts=[ToolCallPart('escalate', {}, tool_call_id=f'escalate-{step}')])
        if step == 2:
            return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'}, tool_call_id='status')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[DynamicCapability(capability_for_step, per_run_step=True)])

    @agent.tool
    def escalate(ctx: RunContext[object]) -> str:
        if 'refunds' not in ctx.capabilities:
            # Not registered on the first step, so there's nothing to load yet.
            with pytest.raises(UserError, match="No capability with id 'refunds' is registered in this run"):
                ctx.load_capability('refunds')
            return 'not yet'
        assert ctx.load_capability('refunds')
        return 'escalated'

    result = await agent.run('Refund order A1')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Refund order A1', timestamp=IsDatetime()),
                    InstructionBaselinePart(
                        instructions=[
                            InstructionPart(
                                content='',
                                dynamic=True,
                                on_change='append',
                                id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                            )
                        ]
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='escalate', args={}, tool_call_id='escalate-0')],
                usage=RequestUsage(input_tokens=53, output_tokens=2),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='escalate', content='not yet', tool_call_id='escalate-0', timestamp=IsDatetime()
                    ),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                        content="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='escalate', args={}, tool_call_id='escalate-1')],
                usage=RequestUsage(input_tokens=89, output_tokens=4),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='escalate', content='escalated', tool_call_id='escalate-1', timestamp=IsDatetime()
                    ),
                    CapabilityLoadDeltaPart(loaded=['refunds'], tool_call_id='escalate-1'),
                    ToolAvailabilityDeltaPart(tools_added=['refund_status'], tool_call_id='escalate-1'),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='refunds')),
                        content='Confirm the order ID first.',
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='refund_status', args={'order_id': 'A1'}, tool_call_id='status')],
                usage=RequestUsage(input_tokens=110, output_tokens=9),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='refund_status',
                        content='Order A1: refund issued.',
                        tool_call_id='status',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=115, output_tokens=10),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.moves_cache_prefix(reason='a loaded capability that leaves mid-run takes its tools out of the request')
async def test_capability_leaving_mid_run_is_withdrawn() -> None:
    """A loaded capability the factory stops returning leaves the catalog, its instructions, and its tools."""
    seen = Seen()
    refunds = make_refunds()
    loaded: list[set[str]] = []

    def capability_for_step(ctx: RunContext[object]) -> Capability[object] | None:
        return refunds if ctx.run_step <= 2 else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.record(messages, info)
        step = responses(messages)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'}, tool_call_id='load')])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'}, tool_call_id='status')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[DynamicCapability(capability_for_step, per_run_step=True)])

    @agent.instructions
    def record_loaded(ctx: RunContext[object]) -> None:
        loaded.append(set(ctx.loaded_capability_ids))

    result = await agent.run('Refund order A1')

    assert loaded == snapshot([set(), {'refunds'}, set()])
    assert seen.tools == snapshot([['load_capability'], ['load_capability', 'refund_status'], ['load_capability']])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Refund order A1', timestamp=IsDatetime()),
                    InstructionBaselinePart(
                        instructions=[
                            InstructionPart(
                                content="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                                dynamic=True,
                                on_change='append',
                                id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                            )
                        ]
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='load')],
                usage=RequestUsage(input_tokens=53, output_tokens=5),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    LoadCapabilityReturnPart(
                        content={'message': "Capability 'refunds' loaded. Its instructions, if any, follow."},
                        tool_call_id='load',
                        timestamp=IsDatetime(),
                    ),
                    ToolAvailabilityDeltaPart(tools_added=['refund_status'], tool_call_id='load'),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='refunds')),
                        content='Confirm the order ID first.',
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='refund_status', args={'order_id': 'A1'}, tool_call_id='status')],
                usage=RequestUsage(input_tokens=84, output_tokens=10),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='refund_status',
                        content='Order A1: refund issued.',
                        tool_call_id='status',
                        timestamp=IsDatetime(),
                    ),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')), content=None
                    ),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='refunds')), content=None
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=126, output_tokens=11),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_loaded_capability_returning_after_leaving_is_loaded_again() -> None:
    """A capability's earlier load counts again when the factory returns it again, and its instructions are restated."""
    refunds = Capability[object](
        id='refunds', description='Refund policy.', instructions='Refunds take 5 days.', defer_loading=True
    )
    loaded: list[set[str]] = []

    def capability_for_step(ctx: RunContext[object]) -> Capability[object] | None:
        return None if ctx.run_step == 3 else refunds

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = responses(messages)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'}, tool_call_id='load')])
        if step < 3:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id=f'ping-{step}')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[DynamicCapability(capability_for_step, per_run_step=True)])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    @agent.instructions
    def record_loaded(ctx: RunContext[object]) -> None:
        loaded.append(set(ctx.loaded_capability_ids))

    result = await agent.run('Refund order A1')

    assert loaded == snapshot([set(), {'refunds'}, set(), {'refunds'}])
    assert [
        (part.id, part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, InstructionDeltaPart)
    ] == snapshot(
        [
            (InstructionId(source=CapabilityInstructionSource(id='refunds')), 'Refunds take 5 days.'),
            (InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')), None),
            (InstructionId(source=CapabilityInstructionSource(id='refunds')), None),
            (
                InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy.\
""",
            ),
            (InstructionId(source=CapabilityInstructionSource(id='refunds')), 'Refunds take 5 days.'),
        ]
    )


async def test_unloaded_capability_leaving_mid_run_leaves_the_catalog() -> None:
    """An on-demand capability that was never loaded leaves the catalog through an appended change.

    It has no tools: a capability's hidden tools are declared to providers that defer them natively, so they'd
    leave the request with it.
    """
    refunds = Capability[object](
        id='refunds', description='Refund policy.', instructions='Refunds take 5 days.', defer_loading=True
    )

    def capability_for_step(ctx: RunContext[object]) -> Capability[object] | None:
        return refunds if ctx.run_step <= 1 else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if responses(messages) == 0:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='ping')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[DynamicCapability(capability_for_step, per_run_step=True)])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('Refund order A1')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Refund order A1', timestamp=IsDatetime()),
                    InstructionBaselinePart(
                        instructions=[
                            InstructionPart(
                                content="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy.\
""",
                                dynamic=True,
                                on_change='append',
                                id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                            )
                        ]
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ping', args={}, tool_call_id='ping')],
                usage=RequestUsage(input_tokens=53, output_tokens=2),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='ping', content='pong', tool_call_id='ping', timestamp=IsDatetime()),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')), content=None
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=76, output_tokens=3),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@dataclass
class Deps:
    refunds_enabled: bool


@pytest.mark.moves_cache_prefix(
    reason='the last run no longer has the loaded capability, so its tools leave the request'
)
async def test_loaded_capability_is_resolved_again_when_the_conversation_resumes() -> None:
    """The factory derives the same capability from the same run context, so a later run finds it loaded."""

    def capability_for_step(ctx: RunContext[Deps]) -> Capability[object] | None:
        return make_refunds() if ctx.deps.refunds_enabled else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if responses(messages) == 0:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'}, tool_call_id='load')])
        return make_text_response(f'done after {responses(messages)} responses')

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=Deps,
        capabilities=[DynamicCapability(capability_for_step, per_run_step=True)],
    )
    first = await agent.run('Refund order A1', deps=Deps(refunds_enabled=True))

    # Resumed without a prompt, on a response calling the capability's tool: the tool runs, as the capability is
    # registered and loaded again.
    second = await agent.run(
        message_history=[
            *first.all_messages(),
            ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'}, tool_call_id='status')]),
        ],
        deps=Deps(refunds_enabled=True),
    )
    assert second.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='refund_status',
                        content='Order A1: refund issued.',
                        tool_call_id='status',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done after 3 responses')],
                usage=RequestUsage(input_tokens=89, output_tokens=18),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # A run whose context no longer produces it doesn't have it, even with its load in history.
    third = await agent.run(
        message_history=[
            *first.all_messages(),
            ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'}, tool_call_id='status')]),
        ],
        deps=Deps(refunds_enabled=False),
    )
    assert third.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='refund_status',
                        content="Unknown tool name: 'refund_status'. Available tools: 'load_capability'",
                        tool_call_id='status',
                        timestamp=IsDatetime(),
                        outcome='retried',
                        retry=RetryInfo(cause='unknown_tool', attempt=1, max_retries=1),
                    ),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')), content=None
                    ),
                    InstructionDeltaPart(
                        id=InstructionId(source=CapabilityInstructionSource(id='refunds')), content=None
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done after 3 responses')],
                usage=RequestUsage(input_tokens=131, output_tokens=18),
                model_name='function:model_fn:',
                model_id='function:function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_factory_is_called_once_per_step_including_a_resumed_suspended_response() -> None:
    steps: list[int] = []

    def capability_for_step(ctx: RunContext[object]) -> Capability[object] | None:
        steps.append(ctx.run_step)
        return make_refunds()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return make_text_response('resumed')

    agent = Agent(FunctionModel(model_fn), capabilities=[DynamicCapability(capability_for_step, per_run_step=True)])
    result = await agent.run(
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='Refund order A1')]),
            ModelResponse(parts=[TextPart('paused')], state='suspended'),
        ]
    )
    assert result.output == 'pausedresumed'
    assert steps == snapshot([0, 1])


@dataclass
class Precise(AbstractCapability[object]):
    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(temperature=0.0)


async def test_model_settings_of_a_capability_appearing_mid_run_apply_once_loaded() -> None:
    temperatures: list[float | None] = []

    def capability_for_step(ctx: RunContext[object]) -> AbstractCapability[object] | None:
        if ctx.run_step < 2:
            return None
        return Precise(id='precise', defer_loading=True)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        temperatures.append((info.model_settings or {}).get('temperature'))
        step = responses(messages)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='ping')])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'precise'}, tool_call_id='load')])
        return make_text_response('done')

    agent = Agent(
        FunctionModel(model_fn),
        model_settings={'temperature': 1.0},
        capabilities=[DynamicCapability(capability_for_step, per_run_step=True)],
    )

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    await agent.run('Be precise')
    assert temperatures == snapshot([1.0, 1.0, 0.0])


@dataclass
class NeedsApproval(AbstractCapability[object]):
    def get_wrapper_toolset(self, toolset: AbstractToolset[object]) -> AbstractToolset[object]:
        return ApprovalRequiredToolset(toolset)


@dataclass
class Audited(AbstractCapability[object]):
    @durable_operation('audit')
    async def audit(self) -> None:  # pragma: no cover
        pass


@pytest.mark.parametrize(
    'returned, error',
    [
        (
            Capability[object](id='always-on', instructions='Always.'),
            "A `DynamicCapability` with `per_run_step=True` can only return on-demand capabilities, but it returned Capability(id='always-on') without `defer_loading=True`.",
        ),
        (
            NativeTool[object](WebSearchTool(), id='search', defer_loading=True),
            "A `DynamicCapability` with `per_run_step=True` does not support native tools yet, but it returned NativeTool(id='search') with them.",
        ),
        (
            CombinedCapability[object]([make_refunds(), Capability[object](id='always-on', instructions='Always.')]),
            "A `DynamicCapability` with `per_run_step=True` can only return on-demand capabilities, but it returned Capability(id='always-on') without `defer_loading=True`.",
        ),
        (
            CombinedCapability[object](
                [make_refunds(), NativeTool[object](WebSearchTool(), id='search', defer_loading=True)]
            ),
            "A `DynamicCapability` with `per_run_step=True` does not support native tools yet, but it returned NativeTool(id='search') with them.",
        ),
        (
            NeedsApproval(id='approvals', defer_loading=True),
            "A `DynamicCapability` with `per_run_step=True` does not support a toolset wrapper (`get_wrapper_toolset`) yet, but it returned NeedsApproval(id='approvals') with them.",
        ),
        (
            Audited(id='audited', defer_loading=True),
            "A `DynamicCapability` with `per_run_step=True` does not support durable operations yet, but it returned Audited(id='audited') with them.",
        ),
    ],
)
async def test_per_step_factory_rejects_what_it_cannot_return(returned: AbstractCapability[object], error: str) -> None:
    agent = Agent(
        FunctionModel(lambda messages, info: make_text_response('done')),
        capabilities=[DynamicCapability(lambda ctx: returned, per_run_step=True)],
    )
    with pytest.raises(UserError) as exc_info:
        await agent.run('Hello')
    assert str(exc_info.value) == error


@pytest.mark.moves_cache_prefix(reason='capabilities that leave mid-run take their hidden tools out of the request')
async def test_combined_capabilities_join_and_leave_the_catalog_together() -> None:
    """A `CombinedCapability` of on-demand capabilities registers each of them, and they leave together."""
    seen = Seen()
    policy = Capability[object](
        id='policy', description='Refund policy.', instructions='Refunds take 5 days.', defer_loading=True
    )
    both = CombinedCapability[object]([make_refunds(), policy])

    def capability_for_step(ctx: RunContext[object]) -> CombinedCapability[object] | None:
        return both if 2 <= ctx.run_step <= 3 else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.record(messages, info)
        step = responses(messages)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='ping-0')])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'policy'}, tool_call_id='load')])
        if step == 2:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='ping-2')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[DynamicCapability(capability_for_step, per_run_step=True)])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('Refund order A1')

    assert seen.tools == snapshot(
        [
            ['load_capability', 'ping'],
            ['load_capability', 'ping'],
            ['load_capability', 'ping'],
            ['load_capability', 'ping'],
        ]
    )
    assert [
        (part.id, part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, InstructionDeltaPart)
    ] == snapshot(
        [
            (
                InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')),
                """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.
- policy: Refund policy.\
""",
            ),
            (InstructionId(source=CapabilityInstructionSource(id='policy')), 'Refunds take 5 days.'),
            (InstructionId(source=CapabilityInstructionSource(id='deferred_capabilities')), None),
            (InstructionId(source=CapabilityInstructionSource(id='policy')), None),
        ]
    )


@dataclass
class Audit(AbstractCapability[object]):
    """Records the steps its hooks run on and the tool results it hears about."""

    requests: list[int] = field(default_factory=list[int])
    events: list[str] = field(default_factory=list[str])

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.requests.append(ctx.run_step)
        return request_context

    async def on_event(self, ctx: RunContext[object], *, event: AgentStreamEvent) -> None:
        if isinstance(event, FunctionToolResultEvent):
            self.events.append(f'{ctx.run_step}: {event.part.tool_name}')


async def test_hooks_and_event_listeners_apply_while_loaded_and_present() -> None:
    """A per-step capability's hooks and event listeners run only once it's loaded, and stop once it's gone."""
    audit = Audit(id='audit', defer_loading=True)

    def capability_for_step(ctx: RunContext[object]) -> Audit | None:
        return audit if 2 <= ctx.run_step <= 4 else None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = responses(messages)
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'audit'}, tool_call_id='load')])
        if step < 5:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id=f'ping-{step}')])
        return make_text_response('done')

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        # A run with an event listener streams its steps, from the step the listener appears on.
        for part in model_fn(messages, info).parts:
            if isinstance(part, ToolCallPart):
                yield {
                    0: DeltaToolCall(
                        name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                    )
                }
            else:
                assert isinstance(part, TextPart)
                yield part.content

    agent = Agent(
        FunctionModel(model_fn, stream_function=stream_fn),
        capabilities=[DynamicCapability(capability_for_step, per_run_step=True)],
    )

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    await agent.run('Hello')

    assert audit.requests == snapshot([3, 4])
    assert audit.events == snapshot(['3: ping', '4: ping'])
