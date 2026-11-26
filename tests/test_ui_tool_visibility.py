"""Per-tool UI visibility: what the UI adapters send a client for each tool, live and from `dump_messages`.

These run the real adapters on a `FunctionModel`: what is under test is the adapters' projection of tool calls and
returns, which no provider response changes, so there is nothing for a cassette to record.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import anyio
import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import Agent, AgentRunResult, CancellationToken, ToolKind, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, Capability, PrepareTools, WrapModelRequestHandler
from pydantic_ai.conversation import ConversationTypeAdapter
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, SkipModelRequest, ToolFailed
from pydantic_ai.messages import (
    AgentStreamEvent,
    CustomEvent as PydanticAICustomEvent,
    DeferredToolRequestsEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    sanitize_messages,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, BuiltinToolCallsReturns, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolDefinition,
    ToolDenied,
    ToolUIVisibility,
)
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset, FunctionToolset

from ._inline_snapshot import snapshot
from .conftest import legacy_retry_prompt_part, sse_data, tool_part_stamps, try_import

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPAppToolResult

with try_import() as imports_successful:
    from ag_ui.core import CustomEvent, Message, RunAgentInput, UserMessage
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream
    from pydantic_ai.ui.ag_ui._utils import (
        ACTIVITY_EVENTS_VERSION,
        DEFAULT_AG_UI_VERSION,
        INTERRUPT_ID_PREFIX,
        parse_ag_ui_version,
    )
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage
    from pydantic_ai.ui.vercel_ai.response_types import DataChunk

with try_import() as interrupts_imports_successful:
    # AG-UI interrupts, which approve a call from the client, came with ag-ui-protocol 0.1.19.
    from ag_ui.core import ResumeEntry

with try_import() as mcp_imports_successful:
    from fastmcp import FastMCP

    from pydantic_ai.mcp import MCPToolset

pytestmark = pytest.mark.skipif(not imports_successful(), reason='ag-ui-protocol or starlette not installed')

ProtocolName = Literal['ag_ui', 'vercel_ai']
PROTOCOLS: list[ProtocolName] = ['ag_ui', 'vercel_ai']

SECRET = 'internal-margin-42'
"""What the server-side tools return: it must never reach the client unless the tool is public."""


def calling(*calls: tuple[str, str]) -> FunctionModel:
    """A model whose first response makes `calls`, as `(tool name, JSON args)`, and whose next one answers `'Done.'`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and not any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield {
                index: DeltaToolCall(name=name, json_args=args, tool_call_id=f'call_{index}')
                for index, (name, args) in enumerate(calls)
            }
        else:
            yield 'Done.'

    return FunctionModel(stream_function=stream)


@dataclass
class Run:
    """What a client got from one adapter run, and the history the run produced."""

    protocol: ProtocolName
    events: list[dict[str, Any]]
    messages: list[ModelMessage]

    @property
    def wire(self) -> str:
        return json.dumps(self.events)

    def tool_events(self) -> list[dict[str, Any]]:
        """The run's tool events, with the fields this file asserts on."""
        if self.protocol == 'ag_ui':
            keep = ('type', 'toolCallId', 'toolCallName', 'delta', 'content', 'activityType', 'name', 'value')
            return [
                {key: event[key] for key in keep if key in event}
                for event in self.events
                if event['type'].startswith('TOOL_CALL') or event['type'] in ('ACTIVITY_SNAPSHOT', 'CUSTOM')
            ]
        keep = ('type', 'toolCallId', 'toolName', 'inputTextDelta', 'input', 'output', 'errorText', 'data')
        return [
            {key: event[key] for key in keep if key in event}
            for event in self.events
            if event['type'].startswith('tool-') or event['type'].startswith('data-')
        ]


async def run_adapter(
    protocol: ProtocolName,
    agent: Agent[None, Any],
    *,
    manage_message_history: Literal['client', 'server'] = 'server',
    ag_ui_version: str | None = None,
    approving: Sequence[ModelMessage] | None = None,
    **adapter_kwargs: Any,
) -> Run:
    """Run `agent` through the protocol's adapter for one user message, decoding the SSE wire it writes.

    The history is held on the server, as non-public tools need it to be. With `approving`, the request instead
    carries that paused history the way a client holding it sends it back, approving `call_0`. `ag_ui_version`
    defaults to the adapter's own.
    """
    adapter_kwargs['manage_message_history'] = manage_message_history
    messages: list[ModelMessage] = []

    async def keep_history(result: AgentRunResult[Any]) -> None:
        messages.extend(result.all_messages())

    if protocol == 'ag_ui':
        ag_ui_version = ag_ui_version or DEFAULT_AG_UI_VERSION
        client_messages: list[Message] = [UserMessage(id='msg_1', content='Where is my order?')]
        resume: dict[str, Any] = {}
        if approving is not None:
            client_messages = AGUIAdapter.dump_messages(approving, ag_ui_version=ag_ui_version)
            resume['resume'] = [
                ResumeEntry(interrupt_id=f'{INTERRUPT_ID_PREFIX}call_0', status='resolved', payload={'approved': True})
            ]
        run_input = RunAgentInput(
            thread_id='thread_1',
            run_id='run_1',
            messages=client_messages,
            state={},
            tools=[],
            context=[],
            forwarded_props={},
            **resume,
        )
        ag_ui_adapter = AGUIAdapter(agent, run_input=run_input, ag_ui_version=ag_ui_version, **adapter_kwargs)
        wire = [
            chunk async for chunk in ag_ui_adapter.encode_stream(ag_ui_adapter.run_stream(on_complete=keep_history))
        ]
    else:
        request = SubmitMessage(
            trigger='submit-message',
            id='chat_1',
            messages=[UIMessage(id='msg_1', role='user', parts=[TextUIPart(text='Where is my order?')])],
        )
        if approving is not None:
            client_history = [
                message.model_dump(mode='json', by_alias=True, exclude_none=True)
                for message in VercelAIAdapter.dump_messages(approving, sdk_version=6)
            ]
            for part in (part for message in client_history for part in message['parts']):
                if part.get('toolCallId') == 'call_0':
                    approval_id = part.get('approval', {}).get('id', 'call_0')
                    part.update(state='approval-responded', approval={'id': approval_id, 'approved': True})
            request = SubmitMessage.model_validate(
                {'trigger': 'submit-message', 'id': 'chat_1', 'messages': client_history}
            )
        vercel_adapter = VercelAIAdapter(agent, run_input=request, sdk_version=6, **adapter_kwargs)
        wire = [
            chunk async for chunk in vercel_adapter.encode_stream(vercel_adapter.run_stream(on_complete=keep_history))
        ]

    return Run(protocol, sse_data(wire), messages)


async def resume(
    protocol: ProtocolName, agent: Agent[None, Any], history: list[ModelMessage], results: DeferredToolResults
) -> Run:
    """Resume a paused run from its stored history through the protocol's event stream, as an endpoint holding it does.

    The history goes through `ModelMessagesTypeAdapter` first, as a conversation store keeps it.
    """
    stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    messages: list[ModelMessage] = []

    async def keep_history(result: AgentRunResult[Any]) -> None:
        messages.extend(result.all_messages())

    async with agent.run_stream_events(message_history=stored, deferred_tool_results=results) as events:
        wire = await encode_standalone(protocol, events, on_complete=keep_history)
    return Run(protocol, sse_data(wire), messages)


async def encode_standalone(
    protocol: ProtocolName,
    events: AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]],
    *,
    on_complete: Callable[[AgentRunResult[Any]], Awaitable[None]] | None = None,
) -> list[str]:
    """`events` as the protocol's event stream encodes them, with no adapter around it."""
    if protocol == 'ag_ui':
        ag_ui_stream = AGUIEventStream[None, Any]()
        return [ag_ui_stream.encode_event(e) async for e in ag_ui_stream.transform_stream(events, on_complete)]
    vercel_stream = VercelAIEventStream[None, Any](sdk_version=6)
    return [vercel_stream.encode_event(e) async for e in vercel_stream.transform_stream(events, on_complete)]


def dump(
    protocol: ProtocolName, messages: Sequence[ModelMessage], default_ui_visibility: ToolUIVisibility = 'public'
) -> str:
    """What a "load chat" endpoint would send for `messages`, as JSON."""
    if protocol == 'ag_ui':
        dumped = AGUIAdapter.dump_messages(messages, default_ui_visibility=default_ui_visibility)
    else:
        dumped = VercelAIAdapter.dump_messages(messages, default_ui_visibility=default_ui_visibility)
    return json.dumps([message.model_dump(mode='json', by_alias=True, exclude_none=True) for message in dumped])


def secret_agent(ui_visibility: ToolUIVisibility | None, **agent_kwargs: Any) -> Agent[None, Any]:
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')), **agent_kwargs)

    @agent.tool_plain(ui_visibility=ui_visibility)
    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    return agent


EXPECTED_REDACTED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '[REDACTED]'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_redacted_tool_keeps_the_call_and_replaces_the_result(protocol: ProtocolName):
    run = await run_adapter(protocol, secret_agent('redacted'))

    assert SECRET not in run.wire
    assert run.tool_events() == EXPECTED_REDACTED_EVENTS[protocol]
    assert SECRET not in dump(protocol, run.messages)
    assert '[REDACTED]' in dump(protocol, run.messages)

    # The stored history keeps the real return; only what goes to the client is projected.
    tool_return = next(part for part in run.messages[2].parts if isinstance(part, ToolReturnPart))
    assert tool_return.content == {'order_id': 'o_1', 'margin': SECRET}
    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-return', 'lookup_order', 'redacted'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_private_tool_sends_neither_call_nor_result(protocol: ProtocolName):
    run = await run_adapter(protocol, secret_agent('private'))

    assert 'lookup_order' not in run.wire
    assert 'call_0' not in run.wire
    assert run.tool_events() == []
    dumped = dump(protocol, run.messages)
    assert 'lookup_order' not in dumped
    assert 'Done.' in dumped
    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'private'),
        ('tool-return', 'lookup_order', 'private'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_public_tool_is_unchanged(protocol: ProtocolName):
    """A tool with no setting under the default adapter is sent in full, and its parts stay unstamped."""
    run = await run_adapter(protocol, secret_agent(None))

    assert SECRET in run.wire
    assert SECRET in dump(protocol, run.messages)
    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', None),
        ('tool-return', 'lookup_order', None),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_adapter_default_applies_to_tools_without_a_setting(protocol: ProtocolName):
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}'), ('order_status', '{"order_id": "o_1"}')))

    @agent.tool_plain
    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    @agent.tool_plain(ui_visibility='public')
    def order_status(order_id: str) -> str:
        return 'shipped'

    run = await run_adapter(protocol, agent, default_ui_visibility='redacted')

    assert SECRET not in run.wire
    assert 'shipped' in run.wire
    # The default is written into the stamp, so a reload without it projects the same way.
    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-call', 'order_status', 'public'),
        ('tool-return', 'lookup_order', 'redacted'),
        ('tool-return', 'order_status', 'public'),
    ]
    dumped = dump(protocol, run.messages)
    assert SECRET not in dumped
    assert 'shipped' in dumped


CLIENT_ROUND_TRIPS: dict[ProtocolName, Callable[[Sequence[ModelMessage]], list[ModelMessage]]] = {
    'ag_ui': lambda messages: AGUIAdapter.load_messages(AGUIAdapter.dump_messages(messages)),
    'vercel_ai': lambda messages: VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(messages)),
}
"""The history a client holding it sends back, after it was sent `messages`."""


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_dump_reads_the_stamp_not_the_tool_definitions(protocol: ProtocolName):
    """A dump projects from the stamps alone, never the tool definitions, and its default doesn't override a stamp."""
    run = await run_adapter(protocol, secret_agent('redacted'))

    assert SECRET not in dump(protocol, run.messages)
    assert 'lookup_order' in dump(protocol, run.messages, default_ui_visibility='private')

    # The stamps survive the stored history's own round trip.
    stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(run.messages))
    assert tool_part_stamps(stored) == tool_part_stamps(run.messages)
    assert SECRET not in dump(protocol, stored)

    # The protocols don't carry them, so a history the client sends back has none, and only the placeholder.
    client_copy = CLIENT_ROUND_TRIPS[protocol](run.messages)
    assert [stamp for *_, stamp in tool_part_stamps(client_copy)] == [None, None]
    assert SECRET not in dump(protocol, client_copy)

    # History with no stamps, such as one from an earlier version, takes the dump's default.
    unstamped = (await run_adapter(protocol, secret_agent(None))).messages
    assert SECRET in dump(protocol, unstamped)
    assert SECRET not in dump(protocol, unstamped, default_ui_visibility='redacted')
    assert 'lookup_order' not in dump(protocol, unstamped, default_ui_visibility='private')


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('change', ['renamed', 'removed'])
async def test_history_is_sent_as_streamed_after_its_tool_changes(
    protocol: ProtocolName, change: Literal['renamed', 'removed']
):
    """A history continued after its tool was renamed or removed is still sent by its stamps, as it was streamed."""
    run = await run_adapter(protocol, secret_agent('private'))
    later = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Anything else?')])))
    if change == 'renamed':

        @later.tool_plain(ui_visibility='public')
        def lookup_order_v2(order_id: str) -> str:
            return SECRET  # pragma: no cover

    result = await later.run('Thanks.', message_history=run.messages)

    dumped = dump(protocol, result.all_messages())
    assert SECRET not in dumped
    assert 'lookup_order' not in dumped
    assert 'Anything else?' in dumped


class OrderArgs(TypedDict):
    order_id: str


class OrderResult(TypedDict):
    margin: str


order_kind = ToolKind('ui_visibility_test', 'order', args=OrderArgs, content=OrderResult)


class OrderCallPart(ToolCallPart[OrderArgs], kind=order_kind):
    pass


class OrderReturnPart(ToolReturnPart[OrderResult], kind=order_kind):
    pass


async def test_stamps_survive_the_conversation_round_trip_on_typed_parts():
    """A stored `Conversation` keeps every stamp: on typed parts, and on the calls a paused run waits on.

    A unit test of serialization: the stamps never reach a provider, so no recording can show them.
    """
    agent = Agent(
        calling(('lookup_order', '{"order_id": "o_1"}'), ('escalate', '{"order_id": "o_1"}')),
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(tool_kind=order_kind, ui_visibility='redacted')
    def lookup_order(args: OrderArgs) -> OrderResult:
        return {'margin': SECRET}

    @agent.tool_plain(ui_visibility='private')
    def escalate(order_id: str) -> str:
        raise CallDeferred()

    async with agent.run_stream_events('Where is my order?') as events:
        result = [event async for event in events if isinstance(event, AgentRunResultEvent)][-1].result
    stored = ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(result.conversation))

    assert [type(part) for message in stored.messages for part in message.parts if part.part_kind != 'user-prompt'] == [
        OrderCallPart,
        ToolCallPart,
        OrderReturnPart,
    ]
    assert tool_part_stamps(stored.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-call', 'escalate', 'redacted'),
        ('tool-return', 'lookup_order', 'redacted'),
    ]
    assert stored.deferred_tool_requests is not None
    assert [(call.tool_name, call.ui_visibility) for call in stored.deferred_tool_requests.calls] == [
        ('escalate', 'redacted')
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_external_and_output_tools_stay_public(protocol: ProtocolName):
    """Results of external (frontend) tools come from the client, and output tool args are the run's answer."""
    frontend = ExternalToolset(
        [ToolDefinition(name='show_map', parameters_json_schema={'type': 'object', 'properties': {}})]
    )
    agent = Agent(
        calling(('show_map', '{}')),
        toolsets=[frontend],
        output_type=[ToolOutput(str, name='final_result'), DeferredToolRequests],
    )

    run = await run_adapter(protocol, agent, default_ui_visibility='private')

    assert 'show_map' in run.wire
    assert tool_part_stamps(run.messages) == [('tool-call', 'show_map', 'public')]

    answer = Agent(calling(('final_result', '{"response": "It shipped."}')), output_type=ToolOutput(str))
    answered = await run_adapter(protocol, answer, default_ui_visibility='private')

    assert 'It shipped.' in answered.wire
    assert tool_part_stamps(answered.messages) == [
        ('tool-call', 'final_result', 'public'),
        ('tool-return', 'final_result', 'public'),
    ]


EXPECTED_APPROVAL_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'refund'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'refund'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'refund',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-approval-request', 'toolCallId': 'call_0'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_approval_required_private_tool_is_shown_redacted(protocol: ProtocolName):
    """The user can't approve a call they can't see, so a `'private'` tool that needs approval is `'redacted'`."""
    agent = Agent(calling(('refund', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True, ui_visibility='private')
    def refund(order_id: str) -> str:
        return SECRET  # pragma: no cover

    run = await run_adapter(protocol, agent)

    assert run.tool_events() == EXPECTED_APPROVAL_EVENTS[protocol]
    assert tool_part_stamps(run.messages) == [('tool-call', 'refund', 'redacted')]
    assert 'refund' in dump(protocol, run.messages)


EXPECTED_DENIED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot([{'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'}]),
    'vercel_ai': snapshot([{'type': 'tool-output-denied', 'toolCallId': 'call_0'}]),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_denied_redacted_tool_sends_the_placeholder_for_the_denial(protocol: ProtocolName):
    """A denial message is the call's result, so a `'redacted'` tool's denial goes out as the placeholder.

    Vercel AI's denied chunk carries no text, so only AG-UI sends the placeholder.
    """
    denial = f'Refunds above {SECRET} need a manager.'
    agent = Agent(calling(('refund', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True, ui_visibility='redacted')
    def refund(order_id: str) -> str:
        return 'Refunded.'  # pragma: no cover

    paused = await run_adapter(protocol, agent)
    run = await resume(protocol, agent, paused.messages, DeferredToolResults(approvals={'call_0': ToolDenied(denial)}))

    assert SECRET not in run.wire
    assert run.tool_events() == EXPECTED_DENIED_EVENTS[protocol]
    assert SECRET not in dump(protocol, run.messages)


ApprovalSetup = Literal[
    'requires_approval_redacted',
    'requires_approval_private',
    'approval_required_at_runtime_private',
    'requires_approval_unset',
]
CLIENT_APPROVALS: dict[ApprovalSetup, tuple[ToolUIVisibility | None, bool]] = {
    'requires_approval_redacted': ('redacted', False),
    'requires_approval_private': ('private', False),
    'approval_required_at_runtime_private': ('private', True),
    'requires_approval_unset': (None, False),
}
"""Each way a tool asks for approval: its `ui_visibility`, and whether it raises `ApprovalRequired` itself."""


def approving_agent(setup: ApprovalSetup) -> Agent[None, Any]:
    """An agent whose `refund` tool asks for approval as `setup` says, and returns the secret once approved."""
    ui_visibility, at_runtime = CLIENT_APPROVALS[setup]
    agent = Agent(
        calling(('refund', '{"order_id": "o_1"}')), deps_type=type(None), output_type=[str, DeferredToolRequests]
    )

    @agent.tool(requires_approval=not at_runtime, ui_visibility=ui_visibility)
    def refund(ctx: RunContext[None], order_id: str) -> str:
        if at_runtime and not ctx.tool_call_approved:
            raise ApprovalRequired()
        return SECRET

    return agent


EXPECTED_CLIENT_APPROVED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot([{'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'}]),
    'vercel_ai': snapshot(
        [
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'refund',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '[REDACTED]'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('setup', [setup for setup, (ui_visibility, _) in CLIENT_APPROVALS.items() if ui_visibility])
async def test_call_approved_from_a_client_held_history_is_stamped_before_it_runs(
    protocol: ProtocolName, setup: ApprovalSetup
):
    """A history the client sends back has no stamps, so the approved call is stamped from its tool before it runs.

    An approved `'private'` call is `'redacted'`, as it was shown to be approved, so its result is the placeholder.
    """
    if protocol == 'ag_ui' and not interrupts_imports_successful():
        pytest.skip('this ag-ui-protocol has no interrupts')  # pragma: lax no cover
    agent = approving_agent(setup)
    paused = await run_adapter(protocol, agent, ag_ui_version='0.1.19')
    with pytest.warns(UserWarning, match="`manage_message_history='server'`"):
        run = await run_adapter(
            protocol, agent, manage_message_history='client', ag_ui_version='0.1.19', approving=paused.messages
        )

    assert SECRET not in run.wire
    assert run.tool_events() == EXPECTED_CLIENT_APPROVED_EVENTS[protocol]
    assert tool_part_stamps(run.messages)[-1] == ('tool-return', 'refund', 'redacted')
    assert SECRET not in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_call_approved_from_a_client_held_history_without_a_setting_is_unchanged(protocol: ProtocolName):
    """A tool with no `ui_visibility` stays unstamped when its approved call runs, and its result is sent."""
    if protocol == 'ag_ui' and not interrupts_imports_successful():
        pytest.skip('this ag-ui-protocol has no interrupts')  # pragma: lax no cover
    agent = approving_agent('requires_approval_unset')
    paused = await run_adapter(protocol, agent, ag_ui_version='0.1.19')
    run = await run_adapter(
        protocol, agent, manage_message_history='client', ag_ui_version='0.1.19', approving=paused.messages
    )

    assert SECRET in run.wire
    assert {stamp for *_, stamp in tool_part_stamps(run.messages)} == {None}


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('client_stamp', [None, 'public'])
async def test_sanitizing_resets_the_stamp_a_client_sends_on_its_history(
    protocol: ProtocolName, client_stamp: ToolUIVisibility | None
):
    """A client can't make an approved call's result public: `sanitize_messages` resets the stamp it sends.

    The protocols don't carry the stamp, so the client here sends Pydantic AI's own message format, as an endpoint
    taking `message_history` from a browser would accept it. The approved call is then stamped from its tool's
    definition, so the `'redacted'` tool's result is the placeholder whatever the client claimed.
    """
    agent = approving_agent('requires_approval_redacted')
    sent: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Where is my order?')]),
        ModelResponse(
            parts=[ToolCallPart('refund', {'order_id': 'o_1'}, tool_call_id='call_0', ui_visibility=client_stamp)]
        ),
    ]

    run = await resume(
        protocol,
        agent,
        sanitize_messages(sent, resolved_tool_call_ids={'call_0'}),
        DeferredToolResults(approvals={'call_0': True}),
    )

    assert SECRET not in run.wire
    assert run.tool_events() == EXPECTED_CLIENT_APPROVED_EVENTS[protocol]
    assert tool_part_stamps(run.messages)[-1] == ('tool-return', 'refund', 'redacted')
    assert SECRET not in dump(protocol, run.messages)


DeferralName = Literal['CallDeferred', 'ApprovalRequired', 'question']
DEFERRALS: dict[DeferralName, Callable[[], CallDeferred | ApprovalRequired]] = {
    'CallDeferred': CallDeferred,
    'ApprovalRequired': ApprovalRequired,
    'question': lambda: CallDeferred(kind='question'),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('deferral', list(DEFERRALS))
async def test_private_tool_that_defers_at_runtime_is_shown_redacted(protocol: ProtocolName, deferral: DeferralName):
    """The call was hidden when it streamed; once it defers, the client has to see what it is answering."""
    agent = Agent(calling(('escalate', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(ui_visibility='private')
    def escalate(order_id: str) -> str:
        raise DEFERRALS[deferral]()

    run = await run_adapter(protocol, agent)

    assert run.tool_events() == EXPECTED_DEFERRED_EVENTS[(protocol, deferral)]
    # The response in history holds a `'redacted'` copy of the call; the part that was streamed is left as it was.
    assert tool_part_stamps(run.messages) == [('tool-call', 'escalate', 'redacted')]
    assert 'escalate' in dump(protocol, run.messages)


EXPECTED_RESUMED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot([{'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'}]),
    'vercel_ai': snapshot([{'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '[REDACTED]'}]),
}


class Answer(BaseModel):
    text: str


async def test_redacted_copy_of_a_deferred_call_leaves_a_call_sharing_its_id_alone():
    """Only the `'private'` call that deferred is shown `'redacted'`, not another call the response gave the same id.

    A response can repeat a `tool_call_id` across the batches `end_strategy='graceful'` runs, here split by an
    output call that fails validation.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart('final_result', {'text': 'Done.'}, tool_call_id='out_1')])
        return ModelResponse(
            parts=[
                ToolCallPart('escalate', {'order_id': 'o_1'}, tool_call_id='x'),
                ToolCallPart('final_result', {'text': 7, 'extra': []}, tool_call_id='out_1'),
                ToolCallPart('lookup_order', {'order_id': 'o_2'}, tool_call_id='x'),
            ]
        )

    agent = Agent(FunctionModel(respond), output_type=[Answer, DeferredToolRequests], end_strategy='graceful')

    @agent.tool_plain(ui_visibility='private')
    def escalate(order_id: str) -> str:
        raise CallDeferred()

    @agent.tool_plain
    def lookup_order(order_id: str) -> str:
        return f'Found {order_id}.'

    result = await agent.run('Where is my order?')

    calls = [
        (part.tool_name, part.args, part.ui_visibility)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_call_id == 'x'
    ]
    assert calls == [('escalate', {'order_id': 'o_1'}, 'redacted'), ('lookup_order', {'order_id': 'o_2'}, None)]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_private_tool_deferred_at_runtime_resumes_redacted(protocol: ProtocolName):
    """The result of a `'private'` call that deferred follows the call's `'redacted'` stamp when the run resumes."""
    agent = Agent(calling(('escalate', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(ui_visibility='private')
    def escalate(order_id: str) -> str:
        raise CallDeferred()

    paused = await run_adapter(protocol, agent)
    run = await resume(protocol, agent, paused.messages, DeferredToolResults(calls={'call_0': SECRET}))

    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'escalate', 'redacted'),
        ('tool-return', 'escalate', 'redacted'),
    ]
    assert SECRET not in run.wire
    assert run.tool_events() == EXPECTED_RESUMED_EVENTS[protocol]
    assert SECRET not in dump(protocol, run.messages)


EXPECTED_DEFERRED_EVENTS: dict[tuple[ProtocolName, DeferralName], Any] = {
    ('ag_ui', 'question'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    ('vercel_ai', 'question'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
        ]
    ),
    ('ag_ui', 'CallDeferred'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    ('ag_ui', 'ApprovalRequired'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    ('vercel_ai', 'CallDeferred'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
        ]
    ),
    ('vercel_ai', 'ApprovalRequired'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-approval-request', 'toolCallId': 'call_0'},
        ]
    ),
}


EXPECTED_REDACTED_METADATA_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'forecast'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"city": "Paris"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_1', 'toolCallName': 'notify'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_1', 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_1'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_1', 'content': '[REDACTED]'},
            {'type': 'CUSTOM', 'name': 'progress', 'value': 'notified'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'forecast'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"city": "Paris"}'},
            {'type': 'tool-input-start', 'toolCallId': 'call_1', 'toolName': 'notify'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_1', 'inputTextDelta': '{}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'forecast',
                'input': {'city': 'Paris'},
            },
            {'type': 'tool-input-available', 'toolCallId': 'call_1', 'toolName': 'notify', 'input': {}},
            {'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '[REDACTED]'},
            {'type': 'tool-output-available', 'toolCallId': 'call_1', 'output': '[REDACTED]'},
            {'type': 'data-progress', 'data': 'notified'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_redacted_tool_drops_the_mcp_app_and_keeps_metadata_events(protocol: ProtocolName):
    """The MCP Apps record holds the full result, so it goes; events the tool put in its metadata are its own channel."""
    agent = Agent(calling(('forecast', '{"city": "Paris"}'), ('notify', '{}')))
    app: MCPAppToolResult = {
        'resource_uri': 'ui://weather/forecast',
        'tool_input': {'city': 'Paris'},
        'result': {'content': [{'type': 'text', 'text': SECRET}]},
        'toolset_id': 'weather',
    }

    @agent.tool_plain(ui_visibility='redacted', sequential=True)
    def forecast(city: str) -> ToolReturn:
        return ToolReturn(return_value=SECRET, metadata={'mcp_app': app})

    event: CustomEvent | DataChunk = (
        CustomEvent(name='progress', value='notified')
        if protocol == 'ag_ui'
        else DataChunk(type='data-progress', data='notified')
    )

    @agent.tool_plain(ui_visibility='redacted', sequential=True)
    def notify() -> ToolReturn:
        return ToolReturn(return_value=SECRET, metadata=event)

    run = await run_adapter(protocol, agent)

    assert SECRET not in run.wire
    assert 'ui://weather/forecast' not in run.wire
    assert run.tool_events() == EXPECTED_REDACTED_METADATA_EVENTS[protocol]
    assert SECRET not in dump(protocol, run.messages)
    assert 'ui://weather/forecast' not in dump(protocol, run.messages)


@dataclass(kw_only=True)
class StockChecked(PydanticAICustomEvent, name='stock_checked'):
    """What a `'private'` tool tells the client while it runs."""

    in_stock: int


EXPECTED_EMITTED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot([{'type': 'CUSTOM', 'name': 'stock_checked', 'value': {'in_stock': 7}}]),
    'vercel_ai': snapshot([{'type': 'data-stock_checked', 'data': {'in_stock': 7}}]),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_private_tool_sends_no_metadata_events_and_can_emit_instead(protocol: ProtocolName):
    """A `'private'` result isn't sent, so neither are the events in its `metadata`; `ctx.emit` still reaches the client."""
    agent = Agent(calling(('check_stock', '{}')))
    event: CustomEvent | DataChunk = (
        CustomEvent(name='progress', value='from-metadata')
        if protocol == 'ag_ui'
        else DataChunk(type='data-progress', data='from-metadata')
    )

    @agent.tool(ui_visibility='private')
    async def check_stock(ctx: RunContext[object]) -> ToolReturn:
        await ctx.emit(StockChecked(in_stock=7))
        return ToolReturn(return_value=SECRET, metadata=event)

    run = await run_adapter(protocol, agent)

    assert 'from-metadata' not in run.wire
    assert 'from-metadata' not in dump(protocol, run.messages)
    assert run.tool_events() == EXPECTED_EMITTED_EVENTS[protocol]
    assert 'call_0' not in run.wire


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('ui_visibility', ['redacted', 'private'])
async def test_tool_return_content_is_a_user_message_visibility_does_not_cover(
    protocol: ProtocolName, ui_visibility: ToolUIVisibility
):
    """`ui_visibility` covers the call and its return value; `ToolReturn.content` is a separate user message.

    The model gets it as one, and `dump_messages` sends it as one, so data the client must not see belongs in the
    return value. See https://github.com/pydantic/pydantic-ai/issues/6662#issuecomment-5443082639.
    """
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility=ui_visibility)
    def lookup_order(order_id: str) -> ToolReturn:
        return ToolReturn(return_value=SECRET, content='Order o_1 is on its way.')

    run = await run_adapter(protocol, agent)

    assert SECRET not in run.wire
    # The adapters' streams never send a result event's `content`, for any tool.
    assert 'Order o_1 is on its way.' not in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert 'Order o_1 is on its way.' in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_tool_parts_without_a_definition_follow_the_adapter_default(protocol: ProtocolName):
    """A call to an unknown tool, and a native tool's parts, have no definition to take a setting from."""

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[BuiltinToolCallsReturns | DeltaToolCalls | str]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and not any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield {
                0: NativeToolCallPart(
                    tool_name='web_search', args={'query': 'o_1'}, tool_call_id='native_1', provider_name='function'
                )
            }
            yield {
                1: NativeToolReturnPart(
                    tool_name='web_search', content=SECRET, tool_call_id='native_1', provider_name='function'
                )
            }
            yield {2: DeltaToolCall(name='made_up', json_args='{}', tool_call_id='call_0')}
        else:
            yield 'Done.'

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain(ui_visibility='private')
    def internal_lookup() -> str:
        return SECRET  # pragma: no cover

    run = await run_adapter(protocol, agent, default_ui_visibility='redacted')

    assert SECRET not in run.wire
    # The retry for the unknown tool lists the available tools, a private one among them.
    assert 'internal_lookup' not in run.wire
    assert tool_part_stamps(run.messages) == [
        ('builtin-tool-call', 'web_search', 'redacted'),
        ('builtin-tool-return', 'web_search', 'redacted'),
        ('tool-call', 'made_up', 'redacted'),
        ('tool-return', 'made_up', 'redacted'),
    ]
    assert 'internal_lookup' not in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize(
    ('ui_visibility', 'manage_message_history', 'default_ui_visibility', 'expected_warning'),
    [
        pytest.param(
            None, 'client', 'redacted', "`manage_message_history='server'`", id='client-held-redacted-default'
        ),
        pytest.param('private', 'client', 'public', "`manage_message_history='server'`", id='client-held-private-tool'),
        pytest.param('private', 'server', 'public', None, id='server-held-private-tool'),
        pytest.param(None, 'client', 'public', None, id='client-held-public-tools'),
    ],
)
async def test_client_held_history_warns_about_non_public_tools(
    protocol: ProtocolName,
    ui_visibility: ToolUIVisibility | None,
    manage_message_history: Literal['client', 'server'],
    default_ui_visibility: ToolUIVisibility,
    expected_warning: str | None,
):
    """A client that only saw the placeholder would send it back as the result, so a run warns once; others don't."""
    run = run_adapter(
        protocol,
        secret_agent(ui_visibility),
        manage_message_history=manage_message_history,
        default_ui_visibility=default_ui_visibility,
    )
    if expected_warning is None:
        await run  # Any warning fails the test (`filterwarnings = ['error']`).
    else:
        with pytest.warns(UserWarning, match=expected_warning) as record:
            await run
        assert len(record) == 1


EXPECTED_INTERRUPTED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-error', 'toolCallId': 'call_0', 'errorText': '[REDACTED]'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_client_held_history_warns_when_an_interrupted_redacted_call_is_closed(protocol: ProtocolName):
    """A redacted call is sent in full, so the first thing hidden can be the placeholder that closes it on an error."""
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility='redacted')
    def lookup_order(order_id: str) -> str:
        raise RuntimeError('The order service is down.')

    with pytest.warns(UserWarning, match="`manage_message_history='server'`") as record:
        run = await run_adapter(protocol, agent, manage_message_history='client')
    assert len(record) == 1
    assert run.tool_events() == EXPECTED_INTERRUPTED_EVENTS[protocol]


async def breaking_off(
    agent: Agent[None, Any], after: type[AgentStreamEvent], error: Exception
) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]:
    """The agent run's events up to its first `after` event, then `error`, as a stream that breaks off sends them."""
    async with agent.run_stream_events('Where is my order?') as events:
        async for event in events:
            yield event
            if isinstance(event, after):
                raise error


EXPECTED_DEFERRED_CLOSEOUT_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-error', 'toolCallId': 'call_0', 'errorText': '[REDACTED]'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_private_call_shown_when_it_defers_is_closed_when_the_stream_breaks(protocol: ProtocolName):
    """A `'private'` call the client is shown once it defers gets the closeout any shown call gets on an error."""
    agent = Agent(calling(('escalate', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(ui_visibility='private')
    def escalate(order_id: str) -> str:
        raise ApprovalRequired()

    wire = await encode_standalone(
        protocol, breaking_off(agent, DeferredToolRequestsEvent, RuntimeError('The client went away.'))
    )

    assert Run(protocol, sse_data(wire), []).tool_events() == EXPECTED_DEFERRED_CLOSEOUT_EVENTS[protocol]


RUN_ERROR_EVENTS: dict[ProtocolName, str] = {'ag_ui': 'RUN_ERROR', 'vercel_ai': 'error'}
"""The event each protocol ends a failed run with."""


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_client_held_history_warning_comes_after_the_run_error(protocol: ProtocolName):
    """Raised as an error, the warning about a closeout it hid can't take the place of the run's own error event."""
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility='redacted')
    def lookup_order(order_id: str) -> str:
        raise RuntimeError('The order service is down.')

    wire: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning, match="`manage_message_history='server'`"):
            async with agent.run_stream_events('Where is my order?') as events:
                # Each stream is set as an adapter run with client-held history sets it.
                if protocol == 'ag_ui':
                    ag_ui_stream = AGUIEventStream[None, Any]()
                    ag_ui_stream._warns_on_hidden_tools = True  # pyright: ignore[reportPrivateUsage]
                    chunks = (ag_ui_stream.encode_event(e) async for e in ag_ui_stream.transform_stream(events))
                else:
                    vercel_stream = VercelAIEventStream[None, Any](sdk_version=6)
                    vercel_stream._warns_on_hidden_tools = True  # pyright: ignore[reportPrivateUsage]
                    chunks = (vercel_stream.encode_event(e) async for e in vercel_stream.transform_stream(events))
                async for chunk in chunks:
                    wire.append(chunk)

    assert sse_data(wire)[-1]['type'] == RUN_ERROR_EVENTS[protocol]
    assert Run(protocol, sse_data(wire), []).tool_events() == EXPECTED_INTERRUPTED_EVENTS[protocol]


EXPECTED_CANCELLED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '[REDACTED]'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_cancelled_run_closes_a_redacted_call_and_sends_nothing_for_a_private_one(protocol: ProtocolName):
    """A run cancelled with both calls running closes the `'redacted'` one with the placeholder; the client knows nothing of the other."""
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}'), ('escalate', '{"order_id": "o_1"}')))
    token = CancellationToken()

    @agent.tool_plain(ui_visibility='redacted')
    async def lookup_order(order_id: str) -> str:
        token.cancel()
        await anyio.sleep_forever()
        return SECRET  # pragma: no cover

    @agent.tool_plain(ui_visibility='private')
    async def escalate(order_id: str) -> str:
        await anyio.sleep_forever()
        return SECRET  # pragma: no cover

    async with agent.run_stream_events('Where is my order?', cancellation_token=token) as events:
        wire = await encode_standalone(protocol, events)

    assert SECRET not in ''.join(wire)
    assert 'escalate' not in ''.join(wire)
    assert Run(protocol, sse_data(wire), []).tool_events() == EXPECTED_CANCELLED_EVENTS[protocol]


EXPECTED_ENQUEUED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [{'type': 'ACTIVITY_SNAPSHOT', 'content': {'messages': []}, 'activityType': 'pydantic_ai_enqueued_messages'}]
    ),
    'vercel_ai': snapshot([{'type': 'data-enqueued-messages', 'data': {'messages': []}}]),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_enqueued_tool_exchange_follows_the_stream_default(protocol: ProtocolName):
    """A tool exchange a tool enqueues has no definition to be stamped from, so the stream's default decides it."""
    if protocol == 'ag_ui' and parse_ag_ui_version(DEFAULT_AG_UI_VERSION) < ACTIVITY_EVENTS_VERSION:
        pytest.skip('this ag-ui-protocol has no activity events')  # pragma: lax no cover

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, ToolReturnPart) and part.tool_name == 'escalate' for m in messages for part in m.parts):
            yield 'Done.'
        else:
            yield {0: DeltaToolCall(name='escalate', json_args='{}', tool_call_id='call_0')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool
    def escalate(ctx: RunContext[object]) -> str:
        ctx.enqueue(
            ModelResponse(parts=[ToolCallPart('lookup_notes', {}, tool_call_id='notes_0')]),
            ModelRequest(parts=[ToolReturnPart('lookup_notes', SECRET, tool_call_id='notes_0')]),
        )
        return 'Escalated.'

    run = await run_adapter(protocol, agent, default_ui_visibility='private')

    assert run.tool_events() == EXPECTED_ENQUEUED_EVENTS[protocol]
    assert 'Escalated.' not in run.wire
    assert 'lookup_notes' not in run.wire
    assert SECRET not in run.wire


async def test_standalone_stream_closes_no_output_call_it_hid():
    """An output call a standalone stream's non-public default hid has nothing to close when the run fails."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name='final_result', json_args='{"answer": ', tool_call_id='call_0')}
        raise RuntimeError('The model went away.')

    agent = Agent(FunctionModel(stream_function=stream), output_type=dict[str, str])
    event_stream = VercelAIEventStream[None, Any](default_ui_visibility='private')
    async with agent.run_stream_events('Where is my order?') as events:
        chunks = [chunk async for chunk in event_stream.encode_stream(event_stream.transform_stream(events))]

    wire = ''.join(chunks)
    assert 'The model went away.' in wire
    assert 'final_result' not in wire


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('prefix_tool', ['public', 'default_redacted'])
async def test_tool_name_streamed_in_pieces_is_stamped_from_the_complete_name(
    protocol: ProtocolName, prefix_tool: Literal['public', 'default_redacted']
):
    """A name that first matches another tool, `lookup` for `lookup_secret`, takes the complete name's stamp.

    The call has already started under the shorter name by then, which the docs call out; its result never goes out.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, ToolReturnPart) for m in messages for part in m.parts):
            yield 'Done.'
        else:
            yield {0: DeltaToolCall(name='lookup', tool_call_id='call_0')}
            yield {0: DeltaToolCall(name='_secret')}
            yield {0: DeltaToolCall(json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain(ui_visibility='public' if prefix_tool == 'public' else None)
    def lookup() -> str:
        return 'Nothing to see.'  # pragma: no cover

    @agent.tool_plain(ui_visibility='private')
    def lookup_secret() -> str:
        return SECRET

    default: ToolUIVisibility = 'redacted' if prefix_tool == 'default_redacted' else 'public'
    run = await run_adapter(protocol, agent, default_ui_visibility=default)

    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_secret', 'private'),
        ('tool-return', 'lookup_secret', 'private'),
    ]
    assert SECRET not in run.wire
    assert 'lookup_secret' not in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert 'lookup_secret' not in dump(protocol, run.messages)


EXPECTED_FAILURE_EVENTS: dict[tuple[ProtocolName, ToolUIVisibility, str], Any] = {
    ('ag_ui', 'redacted', 'ToolFailed'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    ('ag_ui', 'redacted', 'ModelRetry'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    ('ag_ui', 'private', 'ToolFailed'): snapshot([]),
    ('ag_ui', 'private', 'ModelRetry'): snapshot([]),
    ('vercel_ai', 'redacted', 'ToolFailed'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-error', 'toolCallId': 'call_0', 'errorText': '[REDACTED]'},
        ]
    ),
    ('vercel_ai', 'redacted', 'ModelRetry'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-error', 'toolCallId': 'call_0', 'errorText': '[REDACTED]'},
        ]
    ),
    ('vercel_ai', 'private', 'ToolFailed'): snapshot([]),
    ('vercel_ai', 'private', 'ModelRetry'): snapshot([]),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('ui_visibility', ['redacted', 'private'])
@pytest.mark.parametrize('failure', [ToolFailed, ModelRetry])
async def test_failure_a_tool_reports_follows_its_visibility(
    protocol: ProtocolName, ui_visibility: ToolUIVisibility, failure: type[ToolFailed | ModelRetry]
):
    """A `ToolFailed` or `ModelRetry` message is the tool's result, so it stays off the stream and out of the dump.

    A `'redacted'` tool's failure still goes out, with the placeholder as its error text.
    """
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility=ui_visibility)
    def lookup_order(order_id: str) -> str:
        raise failure(SECRET)

    run = await run_adapter(protocol, agent)

    assert 'Done.' in run.wire
    assert SECRET not in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert run.tool_events() == EXPECTED_FAILURE_EVENTS[(protocol, ui_visibility, failure.__name__)]


EXPECTED_INVALID_ARGS_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{}'},
            {
                'type': 'tool-input-error',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {},
                'errorText': '[REDACTED]',
            },
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_redacted_call_with_invalid_args_gets_the_placeholder_as_its_error(protocol: ProtocolName):
    """The validation error a call gets is its result, so a `'redacted'` tool's goes out as the placeholder.

    The arguments are still sent, as they are for every `'redacted'` call.
    """
    agent = Agent(calling(('lookup_order', '{}')))

    @agent.tool_plain(ui_visibility='redacted')
    def lookup_order(order_id: str) -> str:
        return SECRET  # pragma: no cover

    run = await run_adapter(protocol, agent)

    assert run.tool_events() == EXPECTED_INVALID_ARGS_EVENTS[protocol]


async def test_standalone_event_stream_default_applies_only_to_unstamped_parts():
    """Without an adapter, nothing writes the default into the stamps, so only unstamped parts take it.

    That includes calls to external and output tools, which is why the adapter, not the stream, applies the default
    to tool definitions: under `'private'`, an output tool's call isn't sent, and a frontend tool's call is sent only
    once the run defers it, since the client has to run it.
    """
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}'), ('order_status', '{"order_id": "o_1"}')))

    @agent.tool_plain
    def lookup_order(order_id: str) -> str:
        return SECRET

    @agent.tool_plain(ui_visibility='public')
    def order_status(order_id: str) -> str:
        return 'shipped'

    stream = AGUIEventStream[None, Any](default_ui_visibility='private')
    async with agent.run_stream_events('Where is my order?') as events:
        wire = [stream.encode_event(event) async for event in stream.transform_stream(events)]

    assert 'lookup_order' not in ''.join(wire)
    assert SECRET not in ''.join(wire)
    assert 'shipped' in ''.join(wire)

    frontend = ExternalToolset(
        [ToolDefinition(name='show_map', parameters_json_schema={'type': 'object', 'properties': {}})]
    )
    external = Agent(calling(('show_map', '{}')), toolsets=[frontend], output_type=[str, DeferredToolRequests])
    output = Agent(calling(('final_result', '{"response": "It shipped."}')), output_type=ToolOutput(str))
    edge_wires: dict[str, str] = {}
    for name, edge_agent in (('external', external), ('output', output)):
        edge_stream = AGUIEventStream[None, Any](default_ui_visibility='private')
        async with edge_agent.run_stream_events('Where is my order?') as events:
            edge_wires[name] = ''.join(
                [edge_stream.encode_event(event) async for event in edge_stream.transform_stream(events)]
            )

    assert 'show_map' in edge_wires['external']
    assert 'final_result' not in edge_wires['output']
    assert 'It shipped.' not in edge_wires['output']


def test_ui_visibility_flows_from_each_tool_setter_into_the_tool_definition() -> None:
    """A unit test: `ToolDefinition.ui_visibility` never reaches the provider, so a recorded request can't observe it."""

    def plain() -> str:
        return 'never called'  # pragma: no cover

    def with_ctx(ctx: RunContext[None]) -> str:
        return 'never called'  # pragma: no cover

    toolset = FunctionToolset[None](ui_visibility='private')
    toolset.tool_plain(plain)
    toolset.tool(ui_visibility='public')(with_ctx)
    toolset.add_tool(Tool(plain, name='as_tool'))
    toolset.add_tool(Tool(plain, name='kept', ui_visibility='redacted'))
    toolset.add_function(plain, name='as_function', ui_visibility='redacted')
    toolset.add_function(plain, name='function_default')
    assert {name: tool.tool_def.ui_visibility for name, tool in toolset.tools.items()} == {
        'plain': 'private',
        'with_ctx': 'public',
        'as_tool': 'private',
        'kept': 'redacted',
        'as_function': 'redacted',
        'function_default': 'private',
    }
    assert FunctionToolset[None]([plain]).tools['plain'].tool_def.ui_visibility is None

    seen: dict[str, ToolUIVisibility | None] = {}

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.update({tool_def.name: tool_def.ui_visibility for tool_def in info.function_tools})
        return ModelResponse(parts=[TextPart('Done.')])

    agent = Agent(FunctionModel(answer), deps_type=type(None))
    agent.tool_plain(ui_visibility='private')(plain)
    agent.tool(ui_visibility='redacted')(with_ctx)
    capability = Capability[None]()
    capability.tool_plain(ui_visibility='private')(plain)
    capability.tool(ui_visibility='redacted')(with_ctx)
    for tooled in (agent, Agent(FunctionModel(answer), deps_type=type(None), capabilities=[capability])):
        seen.clear()
        tooled.run_sync('Hi')
        assert seen == {'plain': 'private', 'with_ctx': 'redacted'}


@pytest.fixture(
    params=[
        'decorator',
        'toolset',
        pytest.param('mcp', marks=pytest.mark.skipif(not mcp_imports_successful(), reason='fastmcp not installed')),
    ]
)
def bulk_redacted_agent(request: pytest.FixtureRequest) -> Agent[None, Any]:
    """An agent whose `lookup_order` comes from a decorator, a `FunctionToolset` or an MCP server, redacted in bulk.

    A sync fixture, so FastMCP's task-extension import runs outside the event loop.
    """

    def redact_lookups(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [replace(td, ui_visibility='redacted') if td.name.startswith('lookup_') else td for td in tool_defs]

    capabilities = [PrepareTools(redact_lookups)]
    if request.param == 'decorator':
        return secret_agent(None, capabilities=capabilities)

    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    toolset: AbstractToolset[Any]
    if request.param == 'toolset':
        toolset = FunctionToolset([lookup_order])
    else:
        server: FastMCP[None] = FastMCP('orders')
        server.tool(lookup_order)
        toolset = MCPToolset(server)
    return Agent(
        calling(('lookup_order', '{"order_id": "o_1"}')),
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=capabilities,
    )


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_prepare_tools_sets_visibility_in_bulk(protocol: ProtocolName, bulk_redacted_agent: Agent[None, Any]):
    """`PrepareTools` reaches every tool: a decorator's, a toolset's and an MCP server's."""
    run = await run_adapter(protocol, bulk_redacted_agent)

    assert SECRET not in run.wire
    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-return', 'lookup_order', 'redacted'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_dispatch_request_passes_the_default_to_the_adapter(protocol: ProtocolName):
    """`dispatch_request` hands `default_ui_visibility` to `from_request`, which sets it on the adapter it builds."""
    if protocol == 'ag_ui':
        adapter_type: type[AGUIAdapter[None, Any] | VercelAIAdapter[None, Any]] = AGUIAdapter
        body = RunAgentInput(
            thread_id='thread_1',
            run_id='run_1',
            messages=[UserMessage(id='msg_1', content='Where is my order?')],
            state={},
            tools=[],
            context=[],
            forwarded_props={},
        ).model_dump_json(by_alias=True)
    else:
        adapter_type = VercelAIAdapter
        body = SubmitMessage(
            trigger='submit-message',
            id='chat_1',
            messages=[UIMessage(id='msg_1', role='user', parts=[TextUIPart(text='Where is my order?')])],
        ).model_dump_json(by_alias=True)

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': body.encode()}

    request = Request(
        scope={'type': 'http', 'method': 'POST', 'headers': [(b'content-type', b'application/json')]},
        receive=receive,
    )
    response = await adapter_type.dispatch_request(
        request, agent=secret_agent(None), manage_message_history='server', default_ui_visibility='redacted'
    )

    assert isinstance(response, StreamingResponse)
    wire = ''.join(
        [chunk if isinstance(chunk, str) else bytes(chunk).decode() async for chunk in response.body_iterator]
    )
    assert 'lookup_order' in wire
    assert SECRET not in wire


LEGACY_RETRY_SENT: dict[ToolUIVisibility, list[str]] = {'redacted': ['lookup_order', '[REDACTED]'], 'private': []}
"""What a dump sends of a call and its legacy retry, by the call's stamp: the call and the placeholder, or nothing."""


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('ui_visibility', ['redacted', 'private'])
def test_dump_projects_a_legacy_retry_as_its_calls_result(protocol: ProtocolName, ui_visibility: ToolUIVisibility):
    """A deprecated `RetryPromptPart` bound to a call is that call's result, so it takes the call's stamp.

    A unit test: no run makes one any more, and a stored one is translated when it loads.
    """
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Where is my order?'),
        ModelResponse(parts=[ToolCallPart('lookup_order', {}, tool_call_id='call_0', ui_visibility=ui_visibility)]),
        ModelRequest(parts=[legacy_retry_prompt_part(SECRET, tool_name='lookup_order', tool_call_id='call_0')]),
    ]

    dumped = dump(protocol, messages)
    assert SECRET not in dumped
    assert [text for text in ('lookup_order', '[REDACTED]') if text in dumped] == LEGACY_RETRY_SENT[ui_visibility]


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_redacted_native_return_drops_the_provider_details(protocol: ProtocolName):
    """The provider's details on a native tool's return can hold its result, so a `'redacted'` return drops them.

    A unit test: no shipped provider puts result data there, so no recording can show it.
    """
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Search the docs.'),
        ModelResponse(
            parts=[
                NativeToolCallPart(
                    'web_search', {'query': 'margins'}, tool_call_id='native_0', provider_name='function'
                ),
                NativeToolReturnPart(
                    'web_search',
                    'Found the margins page.',
                    tool_call_id='native_0',
                    provider_name='function',
                    provider_details={'raw_result': SECRET},
                    ui_visibility='redacted',
                ),
                TextPart('Found it.'),
            ]
        ),
    ]

    assert SECRET not in dump(protocol, messages)
    assert '[REDACTED]' in dump(protocol, messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_error_closeout_hides_a_call_whose_id_a_private_call_reused(protocol: ProtocolName):
    """A `'private'` call that reuses a pending call's `tool_call_id` takes over the id, so the closeout is hidden too.

    A unit test of the stream: no model in the test suite repeats a `tool_call_id`, so the events are built by hand.
    """
    public = ToolCallPart('lookup_order', {}, tool_call_id='dup')
    private = ToolCallPart('lookup_margin', {}, tool_call_id='dup', ui_visibility='private')

    async def events() -> AsyncIterator[AgentStreamEvent]:
        yield PartStartEvent(index=0, part=public)
        yield PartEndEvent(index=0, part=public)
        yield FunctionToolCallEvent(public)
        yield PartStartEvent(index=1, part=private)
        raise RuntimeError('The model provider is down.')

    wire = ''.join(await encode_standalone(protocol, events()))

    assert 'The model provider is down.' in wire
    assert 'lookup_margin' not in wire
    assert 'interrupted' not in wire


def test_unknown_visibility_raises_rather_than_sends_the_result():
    """A value outside `ToolUIVisibility`, which only code that ignores its types can set, never reads as `'public'`."""
    call = ToolCallPart('lookup_order', {}, tool_call_id='call_0', ui_visibility='Private')  # pyright: ignore[reportArgumentType]

    with pytest.raises(AssertionError):
        dump('vercel_ai', [ModelResponse(parts=[call])])


AnswerPath = Literal['wrap', 'skip', 'recover']
"""How a capability answers a model request: by wrapping it, by skipping it, or by recovering from its error."""


def _lookup_call() -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('lookup_order', {'order_id': 'o_1'}, tool_call_id='call_0')])


@dataclass
class AnswersForTheModel(AbstractCapability[Any]):
    """Answers the first request with a call to `lookup_order` instead of the model, the way `path` says."""

    path: AnswerPath

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        if self.path == 'recover' or any(
            isinstance(part, ToolReturnPart) for message in request_context.messages for part in message.parts
        ):
            return await handler(request_context)
        if self.path == 'skip':
            raise SkipModelRequest(_lookup_call())
        return _lookup_call()

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        return _lookup_call()


def answered_agent(path: AnswerPath) -> Agent[None, Any]:
    """A `'private'` `lookup_order` whose call comes from `AnswersForTheModel`; the model only fails, then answers."""

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            raise RuntimeError('The model is down.')
        return ModelResponse(parts=[TextPart('Done.')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        answer(messages, info)
        yield 'Done.'

    agent = Agent(FunctionModel(answer, stream_function=stream), capabilities=[AnswersForTheModel(path)])

    @agent.tool_plain(ui_visibility='private')
    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    return agent


ANSWER_PATHS: list[AnswerPath] = ['wrap', 'skip', 'recover']


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('path', ANSWER_PATHS)
async def test_response_a_capability_supplies_is_stamped(protocol: ProtocolName, path: AnswerPath):
    """A call in a response a capability gives instead of the model's is stamped from its tool like the model's own."""
    run = await run_adapter(protocol, answered_agent(path))

    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'private'),
        ('tool-return', 'lookup_order', 'private'),
    ]
    assert 'o_1' not in run.wire
    assert SECRET not in run.wire
    assert SECRET not in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_recovered_response_keeps_its_tools_setting_under_a_non_public_default(protocol: ProtocolName):
    """A recovered response is stamped before `after_model_request`, so the adapter's default can't write over it."""
    run = await run_adapter(protocol, answered_agent('recover'), default_ui_visibility='redacted')

    assert tool_part_stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'private'),
        ('tool-return', 'lookup_order', 'private'),
    ]
    assert 'o_1' not in run.wire


@pytest.mark.parametrize('path', ANSWER_PATHS)
async def test_response_a_capability_supplies_is_stamped_without_streaming(path: AnswerPath):
    """A run that doesn't stream stamps the supplied response too, so a later dump with no default leaves it out."""
    result = await answered_agent(path).run('Where is my order?')

    assert SECRET not in dump('vercel_ai', result.all_messages())


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_dump_drops_a_response_left_empty_by_private_parts(protocol: ProtocolName):
    """A response that only held a private call leaves no assistant message behind."""
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Where is my order?'),
        ModelResponse(parts=[ToolCallPart('lookup_order', {'order_id': 'o_1'}, 'call_0', ui_visibility='private')]),
        ModelRequest(parts=[ToolReturnPart('lookup_order', SECRET, 'call_0', ui_visibility='private')]),
        ModelResponse(parts=[TextPart('Done.')]),
    ]
    dumped = json.loads(dump(protocol, messages))
    assert [message['role'] for message in dumped] == ['user', 'assistant']
    assert 'Done.' in json.dumps(dumped[-1])
    assert 'lookup_order' not in json.dumps(dumped)
